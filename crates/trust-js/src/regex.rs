//! A from-scratch regular-expression engine (no dependencies).
//!
//! Pipeline: [`parse`] turns a pattern string into a [`Node`] AST, `compile` lowers it to a flat
//! [`Inst`] program, and [`Regex::exec_at`] runs a backtracking matcher over it, keeping its
//! choice points on a heap stack rather than the native one. Supports the commonly-used syntax:
//! literals, `.`, character classes (`[...]`, `\d\w\s` and negations), anchors (`^ $ \b \B`),
//! quantifiers (`* + ? {n} {n,} {n,m}`, greedy + lazy), groups (capturing, `(?:)`), alternation,
//! backreferences, and lookaround, with the `d g i m s u v y` flags. As ECMA-262 requires,
//! backtracking is bounded neither in time nor in depth: a pathological pattern runs until it
//! finishes or the host interrupts the script, which the matcher polls for. Only absurd growth of
//! the backtracking state is cut off, as an explicit resource error rather than an ordinary
//! no-match.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::OnceLock;

const MAX_REPEAT: usize = 1000;
const INLINE_CAPTURES: usize = 4;
const INTERRUPT_POLL_MASK: usize = 0x3fff;
/// Keep cold patterns on the compact matcher program. A pattern is promoted only after repeated
/// execution and only when an ASCII or one-byte subject has actually been observed.
const REGEXP_TIER_UP_THRESHOLD: u32 = 64;
/// Override the feedback threshold for local A/B runs. Zero disables the experimental tier.
fn regexp_tier_up_threshold() -> u32 {
    std::env::var("TRUST_JS_REGEXP_TIER_UP_AT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(REGEXP_TIER_UP_THRESHOLD)
}

const SUBJECT_ASCII: u8 = 1 << 0;
const SUBJECT_ONE_BYTE: u8 = 1 << 1;
const SUBJECT_TWO_BYTE: u8 = 1 << 2;
const SUBJECT_ASTRAL: u8 = 1 << 3;

/// Number of [`Inst`] variants, one counter per kind for the opt-in matcher profiling report.
const REGEXP_PROF_INST_KINDS: usize = 26;
const REGEXP_PROF_INST_NAMES: [&str; REGEXP_PROF_INST_KINDS] = [
    "char",
    "any",
    "class",
    "string_set",
    "string_set_repeat",
    "save",
    "split",
    "jmp",
    "match",
    "assert_start",
    "assert_end",
    "word_boundary",
    "backref",
    "backref_alt",
    "clear_caps",
    "look",
    "look_behind",
    "many",
    "push_flags",
    "pop_flags",
    "set_mark",
    "check_progress",
    "counter_reset",
    "counter_incr",
    "counted_branch",
    "check_counted_progress",
];

/// Opt-in matcher profiling (`TRUST_JS_REGEXP_PROF=1`): per-instruction dispatch counts, scan
/// positions, candidate attempts, and backtracking entries, reported once at process exit.
/// Disabled matching costs one relaxed load per exec plus a single predicted branch per
/// instruction; the counters live in a thread-local so the engine's single-threaded Agent
/// boundary keeps them consistent.
pub(crate) struct RegexpProf {
    pub(crate) inst: [u64; REGEXP_PROF_INST_KINDS],
    pub(crate) scan_positions: u64,
    pub(crate) attempts: u64,
    pub(crate) backtrack_entries: u64,
    pub(crate) tier_up_attempts: u64,
    pub(crate) tier_up_successes: u64,
    pub(crate) native_execs: u64,
    pub(crate) native_code_bytes: u64,
}

impl RegexpProf {
    const fn zero() -> Self {
        Self {
            inst: [0; REGEXP_PROF_INST_KINDS],
            scan_positions: 0,
            attempts: 0,
            backtrack_entries: 0,
            tier_up_attempts: 0,
            tier_up_successes: 0,
            native_execs: 0,
            native_code_bytes: 0,
        }
    }
}

impl Default for RegexpProf {
    fn default() -> Self {
        Self::zero()
    }
}

static REGEXP_PROF_GATE: OnceLock<bool> = OnceLock::new();

fn regexp_prof_enabled() -> bool {
    *REGEXP_PROF_GATE.get_or_init(|| std::env::var_os("TRUST_JS_REGEXP_PROF").is_some())
}

fn regexp_prof_tier_up(success: bool) {
    if regexp_prof_enabled() {
        REGEXP_PROF.with(|cell| {
            let mut prof = cell.borrow_mut();
            prof.tier_up_attempts += 1;
            if success {
                prof.tier_up_successes += 1;
            }
        });
    }
}

fn regexp_prof_native_exec() {
    if regexp_prof_enabled() {
        REGEXP_PROF.with(|cell| cell.borrow_mut().native_execs += 1);
    }
}

fn regexp_prof_native_code(bytes: usize) {
    if regexp_prof_enabled() {
        REGEXP_PROF.with(|cell| {
            let mut prof = cell.borrow_mut();
            prof.native_code_bytes = prof.native_code_bytes.saturating_add(bytes as u64);
        });
    }
}

thread_local! {
    static REGEXP_PROF: std::cell::RefCell<RegexpProf> =
        const { std::cell::RefCell::new(RegexpProf::zero()) };
}

/// Index the one instruction kind that dispatched (see [`REGEXP_PROF_INST_NAMES`]).
fn regexp_prof_kind(inst: &Inst) -> usize {
    match inst {
        Inst::Char(_) => 0,
        Inst::Any => 1,
        Inst::Class(_) => 2,
        Inst::StringSet(_) => 3,
        Inst::StringSetRepeat { .. } => 4,
        Inst::Save(_) => 5,
        Inst::Split(..) => 6,
        Inst::Jmp(_) => 7,
        Inst::Match => 8,
        Inst::AssertStart => 9,
        Inst::AssertEnd => 10,
        Inst::WordBoundary(_) => 11,
        Inst::Backref(_) => 12,
        Inst::BackrefAlt(_) => 13,
        Inst::ClearCaps(..) => 14,
        Inst::Look { .. } => 15,
        Inst::LookBehind { .. } => 16,
        Inst::Many { .. } => 17,
        Inst::PushFlags(..) => 18,
        Inst::PopFlags => 19,
        Inst::SetMark(_) => 20,
        Inst::CheckProgress(_) => 21,
        Inst::CounterReset(_) => 22,
        Inst::CounterIncr(_) => 23,
        Inst::CountedBranch { .. } => 24,
        Inst::CheckCountedProgress { .. } => 25,
    }
}

/// A matcher implementation limit is not an ECMAScript match failure. RegExpBuiltinExec may
/// return `null` only after its matcher returns failure (ECMA-262 §22.2.7.2); collapsing exhausted
/// backtracking into failure can silently choose the wrong program branch. Host interruption is
/// likewise control flow, not a JavaScript no-match result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatchError {
    ResourceExhausted,
    Interrupted(crate::InterruptReason),
}

pub(crate) type MatchResult<T> = Result<Option<T>, MatchError>;

#[inline]
fn poll_interrupt(control: &crate::RuntimeInterrupt) -> Result<(), MatchError> {
    match control.current_reason() {
        Some(reason) => Err(MatchError::Interrupted(reason)),
        None => Ok(()),
    }
}

pub(crate) enum Captures {
    Inline {
        len: u8,
        spans: [Option<(usize, usize)>; INLINE_CAPTURES],
    },
    Heap(Box<[Option<(usize, usize)>]>),
}

impl Captures {
    fn from_slots(slots: &[Option<usize>], groups: usize) -> Self {
        let len = groups + 1;
        if len <= INLINE_CAPTURES {
            let mut inline = [None; INLINE_CAPTURES];
            for (group, span) in inline[..len].iter_mut().enumerate() {
                *span = match (slots[2 * group], slots[2 * group + 1]) {
                    (Some(a), Some(b)) => Some((a.min(b), a.max(b))),
                    _ => None,
                };
            }
            Captures::Inline {
                len: len as u8,
                spans: inline,
            }
        } else {
            let mut spans = Vec::with_capacity(len);
            for group in 0..len {
                spans.push(match (slots[2 * group], slots[2 * group + 1]) {
                    (Some(a), Some(b)) => Some((a.min(b), a.max(b))),
                    _ => None,
                });
            }
            Captures::Heap(spans.into_boxed_slice())
        }
    }

    fn one(span: (usize, usize)) -> Self {
        let mut spans = [None; INLINE_CAPTURES];
        spans[0] = Some(span);
        Captures::Inline { len: 1, spans }
    }
}

impl std::ops::Deref for Captures {
    type Target = [Option<(usize, usize)>];
    fn deref(&self) -> &Self::Target {
        match self {
            Captures::Inline { len, spans } => &spans[..*len as usize],
            Captures::Heap(spans) => spans,
        }
    }
}

impl AsRef<[Option<(usize, usize)>]> for Captures {
    fn as_ref(&self) -> &[Option<(usize, usize)>] {
        self
    }
}

/// A compiled regular expression.
pub struct Regex {
    prog: Vec<Inst>,
    nmarks: usize,
    /// Start-position prescan derived from the program (see [`first_filter`]): lets the scan
    /// skip positions that cannot begin a match instead of running the backtracker at each.
    first: FirstFilter,
    /// Exact leading ASCII byte, when the first-set proof has only one case-sensitive literal.
    /// The byte input scans this eight bytes at a time before entering the backtracker.
    first_byte: Option<u8>,
    /// Capture-free, case-sensitive ASCII literal program. Searching it directly is equivalent
    /// to executing `Save(0), Char*, Save(1), Match`, without paying the backtracking VM dispatch.
    literal_ascii: Option<Box<[u8]>>,
    /// Capture-free case-insensitive (non-Unicode-mode) literal program, UTF-8 encoded. ECMA-262
    /// canonicalization for a non-`u` `i` pattern folds only ASCII letters, so a byte-level
    /// fold-aware search over an ASCII subject is exact — including non-ASCII pattern characters
    /// (e.g. `ß`), which match by byte equality because canonicalization never folds them.
    literal_fold: Option<Box<[u8]>>,
    /// Capture-free programs containing a string-valued UnicodeSets class can memoize failed
    /// continuation states without changing observable captures.
    memo_string_failures: bool,
    /// [`FirstFilter::Atoms`] baked into a byte-indexed table (elements < 256): the scan loop
    /// becomes one load per position.
    first_lut: Option<Box<[bool; 256]>>,
    /// Besides the [`FirstFilter::Atoms`], a match may begin wherever `^` holds.
    first_line_start: bool,
    /// Per-pattern execution feedback. These cells add no allocation to cold patterns; the native
    /// straight-line program is compiled lazily after the tier-up threshold.
    tier_ticks: Cell<u32>,
    subject_shapes: Cell<u8>,
    tier_up_attempted: Cell<bool>,
    tier_up_at: u32,
    one_byte_native: RefCell<Option<OneByteNativeProgram>>,
    pub unicode: bool,
    pub ngroups: usize,
    pub source: String,
    pub flags: String,
    pub global: bool,
    pub ignore_case: bool,
    pub multiline: bool,
    pub dotall: bool,
    pub sticky: bool,
    /// `(?<name>…)` group names paired with their capture index.
    pub names: Vec<(String, usize)>,
}

/// The first native RegExp subset: a capture-free, straight-line sequence over one-byte
/// characters, dot, and precomputed ASCII/Latin-1 classes. It has no control-flow or capture
/// state, so its candidate loop is semantically equivalent to the corresponding matcher bytecode
/// and can return the group-0 span directly. Unsupported instructions stay on the general matcher.
#[derive(Clone)]
enum OneByteNativeOp {
    Char(u32),
    Any,
    Class(Rc<CharClass>),
    AssertStart,
    AssertEnd,
    /// A terminal repeated single-element matcher. This keeps the native boundary free of
    /// continuation/backtracking state: the whole match is the run selected by `greedy`.
    Many {
        rep: Rep,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
}

impl OneByteNativeOp {
    fn matches(&self, element: u32) -> bool {
        match self {
            OneByteNativeOp::Char(expected) => element == *expected,
            OneByteNativeOp::Any => !is_line_terminator_u32(element),
            OneByteNativeOp::Class(class) => class.matches(element, false, false),
            OneByteNativeOp::AssertStart | OneByteNativeOp::AssertEnd => true,
            OneByteNativeOp::Many { rep, .. } => one_byte_rep_matches(rep, element),
        }
    }

    fn consumes_element(&self) -> bool {
        !matches!(
            self,
            OneByteNativeOp::AssertStart | OneByteNativeOp::AssertEnd
        )
    }
}

fn one_byte_rep_matches(rep: &Rep, element: u32) -> bool {
    match rep {
        Rep::Char(expected) => element == *expected,
        Rep::Any => !is_line_terminator_u32(element),
        Rep::Class(class) => class.matches(element, false, false),
    }
}

trait OneByteSubject {
    fn len(&self) -> usize;
    fn element_at(&self, index: usize) -> u32;
}

impl OneByteSubject for [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    fn element_at(&self, index: usize) -> u32 {
        self[index] as u32
    }
}

impl OneByteSubject for [u32] {
    fn len(&self) -> usize {
        <[u32]>::len(self)
    }

    fn element_at(&self, index: usize) -> u32 {
        self[index]
    }
}

#[repr(C)]
struct OneByteNativeMatch {
    from: usize,
    to: usize,
}

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
type OneByteNativeEntry = unsafe extern "C" fn(
    subject: *const u8,
    len: usize,
    start: usize,
    sticky: u8,
    control: *const crate::RuntimeInterrupt,
    out: *mut OneByteNativeMatch,
) -> u8;

#[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
extern "C" fn regexp_native_poll(control: *const crate::RuntimeInterrupt) -> u32 {
    // The generated code returns 0 for no interruption and 1..=3 in the same priority order as
    // RuntimeInterrupt. The helper keeps deadline/mutex semantics out of the generated loop.
    // Return a full word so all of w0 is defined for the AArch64 caller's comparison.
    match unsafe { control.as_ref() }.and_then(crate::RuntimeInterrupt::current_reason) {
        None => 0,
        Some(crate::InterruptReason::Cancelled) => 1,
        Some(crate::InterruptReason::UserNavigation) => 2,
        Some(crate::InterruptReason::DeadlineExceeded) => 3,
    }
}

struct OneByteNativeCode {
    #[allow(dead_code)] // Ownership keeps both W^X mappings alive until the pattern is evicted.
    byte_executable: crate::jit::ExecutableBuffer,
    #[allow(dead_code)]
    word_executable: crate::jit::ExecutableBuffer,
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
    byte_entry: OneByteNativeEntry,
    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
    word_entry: OneByteNativeEntry,
}

impl OneByteNativeCode {
    fn compile(ops: &[OneByteNativeOp]) -> Option<Self> {
        // Terminal Many has a specialized Rust loop for now. Keep it out of the fixed-width
        // machine emitter until the emitter grows a bounded repeat counter and interruption
        // poll path; the representation remains shared with the generated differential gate.
        if ops
            .iter()
            .any(|op| matches!(op, OneByteNativeOp::Many { .. }))
        {
            return None;
        }
        #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
        {
            // Keep the straight-line emitter's relocation and branch ranges bounded. Live bytes
            // are charged by the shared W^X allocation owner below, alongside ordinary JIT code.
            if ops.len() > 256 {
                return None;
            }
            let byte_code = {
                #[cfg(target_arch = "aarch64")]
                {
                    compile_one_byte_native_aarch64(ops, false)?
                }
                #[cfg(target_arch = "x86_64")]
                {
                    compile_one_byte_native_x64(ops, false)?
                }
            };
            let word_code = {
                #[cfg(target_arch = "aarch64")]
                {
                    compile_one_byte_native_aarch64(ops, true)?
                }
                #[cfg(target_arch = "x86_64")]
                {
                    compile_one_byte_native_x64(ops, true)?
                }
            };
            let byte_executable = crate::jit::ExecutableBuffer::from_bytes(&byte_code)?;
            let word_executable = crate::jit::ExecutableBuffer::from_bytes(&word_code)?;
            // The emitter produces one entry at byte zero and the allocation is immutable after
            // W^X publication. Its only embedded pointers refer to immutable class LUTs retained
            // by `ops` in the owning OneByteNativeProgram.
            let byte_entry = unsafe {
                std::mem::transmute::<*const u8, OneByteNativeEntry>(byte_executable.as_ptr())
            };
            let word_entry = unsafe {
                std::mem::transmute::<*const u8, OneByteNativeEntry>(word_executable.as_ptr())
            };
            regexp_prof_native_code(byte_executable.len());
            regexp_prof_native_code(word_executable.len());
            Some(Self {
                byte_executable,
                word_executable,
                byte_entry,
                word_entry,
            })
        }
        #[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix)))]
        {
            let _ = ops;
            None
        }
    }

    fn find_ascii(
        &self,
        subject: &[u8],
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
        {
            self.run_entry(
                self.byte_entry,
                subject.as_ptr(),
                subject.len(),
                start,
                sticky,
                control,
            )
        }
        #[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix)))]
        {
            let _ = (subject, start, sticky, control);
            unreachable!("machine RegExp code is unavailable on this target")
        }
    }

    fn find_one_byte(
        &self,
        subject: &[u32],
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
        {
            self.run_entry(
                self.word_entry,
                subject.as_ptr() as *const u8,
                subject.len(),
                start,
                sticky,
                control,
            )
        }
        #[cfg(not(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix)))]
        {
            let _ = (subject, start, sticky, control);
            unreachable!("machine RegExp code is unavailable on this target")
        }
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
    fn run_entry(
        &self,
        entry: OneByteNativeEntry,
        subject: *const u8,
        len: usize,
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        let mut out = OneByteNativeMatch { from: 0, to: 0 };
        let status = unsafe { (entry)(subject, len, start, sticky as u8, control, &mut out) };
        match status {
            0 => Ok(None),
            1 => Ok(Some((out.from, out.to))),
            2 => Err(MatchError::Interrupted(crate::InterruptReason::Cancelled)),
            3 => Err(MatchError::Interrupted(
                crate::InterruptReason::UserNavigation,
            )),
            4 => Err(MatchError::Interrupted(
                crate::InterruptReason::DeadlineExceeded,
            )),
            _ => Err(MatchError::ResourceExhausted),
        }
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
struct OneByteNativeAssembler {
    code: Vec<u8>,
    labels: Vec<Option<usize>>,
    branches: Vec<(usize, usize)>,
}

#[cfg(all(target_arch = "x86_64", unix))]
impl OneByteNativeAssembler {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            labels: Vec::new(),
            branches: Vec::new(),
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

    fn imm32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn imm64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn rel32(&mut self, label: usize) {
        self.branches.push((self.code.len(), label));
        self.imm32(0);
    }

    fn jmp(&mut self, label: usize) {
        self.bytes(&[0xe9]);
        self.rel32(label);
    }

    fn jcc(&mut self, condition: u8, label: usize) {
        self.bytes(&[0x0f, condition]);
        self.rel32(label);
    }

    fn finish(mut self) -> Option<Vec<u8>> {
        for (at, label) in self.branches {
            let target = self.labels.get(label).and_then(|offset| *offset)?;
            let next = at.checked_add(4)?;
            let displacement = (target as isize).checked_sub(next as isize)?;
            let displacement = i32::try_from(displacement).ok()?;
            self.code[at..at + 4].copy_from_slice(&displacement.to_le_bytes());
        }
        Some(self.code)
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
fn compile_one_byte_native_x64(ops: &[OneByteNativeOp], word_elements: bool) -> Option<Vec<u8>> {
    // SysV ABI arguments enter in rdi/rsi/rdx/rcx/r8/r9. Move them into callee-saved registers so
    // the poll helper can use the ordinary argument registers without losing the scan state.
    let mut a = OneByteNativeAssembler::new();
    let no_match = a.label();
    let interrupted = a.label();
    let ret = a.label();
    let fail = a.label();
    let loop_start = a.label();

    a.bytes(&[
        0x53, // push rbx
        0x55, // push rbp
        0x41, 0x54, // push r12
        0x41, 0x55, // push r13
        0x41, 0x56, // push r14
        0x41, 0x57, // push r15
        0x48, 0x83, 0xec, 0x08, // sub rsp, 8 (align calls)
        0x49, 0x89, 0xfc, // mov r12, rdi (subject)
        0x49, 0x89, 0xf5, // mov r13, rsi (length)
        0x49, 0x89, 0xd6, // mov r14, rdx (current start)
        0x49, 0x89, 0xcf, // mov r15, rcx (sticky)
        0x4c, 0x89, 0xc3, // mov rbx, r8 (interrupt)
        0x4c, 0x89, 0xcd, // mov rbp, r9 (result)
        0x4d, 0x39, 0xee, // cmp r14, r13
    ]);
    a.jcc(0x87, no_match); // ja
    let width = ops.iter().filter(|op| op.consumes_element()).count();
    a.bytes(&[0x4d, 0x89, 0xeb]); // mov r11, r13 (last = length - width)
    a.bytes(&[0x49, 0x81, 0xeb]);
    a.imm32(width as u32);
    a.bytes(&[0x4c, 0x89, 0xe8]); // mov rax, r13
    a.bytes(&[0x4c, 0x29, 0xf0]); // sub rax, r14
    a.bytes(&[0x48, 0x81, 0xf8]); // cmp rax, width
    a.imm32(width as u32);
    a.jcc(0x82, no_match); // jb

    a.bind(loop_start);
    a.bytes(&[0x48, 0x89, 0xdf]); // mov rdi, rbx
    a.bytes(&[0x48, 0xb8]); // movabs rax, regexp_native_poll
    a.imm64(regexp_native_poll as *const () as usize as u64);
    a.bytes(&[0xff, 0xd0]); // call rax
    a.bytes(&[0x84, 0xc0]); // test al, al
    a.jcc(0x85, interrupted); // jne
    if word_elements {
        a.bytes(&[0x4f, 0x8d, 0x14, 0xb4]); // lea r10, [r12 + r14 * 4]
    } else {
        a.bytes(&[0x4f, 0x8d, 0x14, 0x34]); // lea r10, [r12 + r14]
    }

    let mut offset = 0i32;
    for op in ops {
        match op {
            OneByteNativeOp::Char(expected) => {
                debug_assert!(*expected < 0x100);
                emit_x64_cmp_mem_imm(&mut a, offset, *expected as u8, word_elements);
                a.jcc(0x85, fail); // jne
                offset += 1;
            }
            OneByteNativeOp::Any => {
                emit_x64_cmp_mem_imm(&mut a, offset, b'\n', word_elements);
                a.jcc(0x84, fail); // je
                emit_x64_cmp_mem_imm(&mut a, offset, b'\r', word_elements);
                a.jcc(0x84, fail); // je
                offset += 1;
            }
            OneByteNativeOp::Class(class) => {
                let lut = class.ascii_lut.as_ref()?.as_ptr() as usize as u64;
                emit_x64_load_mem(&mut a, offset, word_elements); // ecx = subject element
                a.bytes(&[0x48, 0xb8]); // movabs rax, LUT
                a.imm64(lut);
                a.bytes(&[0x0f, 0xb6, 0x04, 0x08]); // movzx eax, byte [rax + rcx]
                a.bytes(&[0x84, 0xc0]); // test al, al
                a.jcc(0x84, fail); // je
                offset += 1;
            }
            OneByteNativeOp::AssertStart => {
                a.bytes(&[0x4d, 0x85, 0xf6]); // test r14, r14
                a.jcc(0x85, fail); // jne
            }
            OneByteNativeOp::AssertEnd => {
                a.bytes(&[0x4d, 0x39, 0xde]); // cmp r14, r11
                a.jcc(0x85, fail); // jne
            }
            OneByteNativeOp::Many { .. } => return None,
        }
    }

    a.bytes(&[0x4c, 0x89, 0x75, 0x00]); // mov [rbp], r14
    a.bytes(&[0x4c, 0x89, 0xf0]); // mov rax, r14
    a.bytes(&[0x48, 0x81, 0xc0]);
    a.imm32(width as u32);
    a.bytes(&[0x48, 0x89, 0x45, 0x08]); // mov [rbp + 8], rax
    a.bytes(&[0xb8, 1, 0, 0, 0]); // match
    a.jmp(ret);

    a.bind(fail);
    a.bytes(&[0x45, 0x84, 0xff]); // test r15b, r15b
    a.jcc(0x85, no_match); // jne
                           // r11 is caller-saved and the polling helper may clobber it. Recompute the final candidate
                           // after every poll instead of trusting the value initialized before the loop.
    a.bytes(&[0x4d, 0x89, 0xeb]); // mov r11, r13 (last = length - width)
    a.bytes(&[0x49, 0x81, 0xeb]);
    a.imm32(width as u32);
    a.bytes(&[0x4d, 0x39, 0xde]); // cmp r14, r11
    a.jcc(0x83, no_match); // jae
    a.bytes(&[0x49, 0xff, 0xc6]); // inc r14
    a.jmp(loop_start);

    a.bind(interrupted);
    a.bytes(&[0x83, 0xc0, 0x01]); // status = interruption reason + match/no-match offset
    a.jmp(ret);

    a.bind(no_match);
    a.bytes(&[0x31, 0xc0]); // no match

    a.bind(ret);
    a.bytes(&[
        0x48, 0x83, 0xc4, 0x08, // add rsp, 8
        0x41, 0x5f, // pop r15
        0x41, 0x5e, // pop r14
        0x41, 0x5d, // pop r13
        0x41, 0x5c, // pop r12
        0x5d, // pop rbp
        0x5b, // pop rbx
        0xc3, // ret
    ]);
    a.finish()
}

#[cfg(all(target_arch = "x86_64", unix))]
fn emit_x64_cmp_mem_imm(
    a: &mut OneByteNativeAssembler,
    offset: i32,
    value: u8,
    word_elements: bool,
) {
    if word_elements {
        if (0..=127).contains(&(offset * 4)) {
            a.bytes(&[0x41, 0x81, 0x7a, (offset * 4) as u8]); // cmp dword [r10 + disp8], imm32
        } else {
            a.bytes(&[0x41, 0x81, 0xba]); // cmp dword [r10 + disp32], imm32
            a.imm32((offset * 4) as u32);
        }
        a.imm32(u32::from(value));
        return;
    }
    if (0..=127).contains(&offset) {
        a.bytes(&[0x41, 0x80, 0x7a, offset as u8, value]); // cmp byte [r10 + disp8], imm8
    } else {
        a.bytes(&[0x41, 0x80, 0xba]); // cmp byte [r10 + disp32], imm8
        a.imm32(offset as u32);
        a.bytes(&[value]);
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
fn emit_x64_load_mem(a: &mut OneByteNativeAssembler, offset: i32, word_elements: bool) {
    let (opcode, scale) = if word_elements { (0x8b, 4) } else { (0xb6, 1) };
    let byte_offset = offset * scale;
    if (0..=127).contains(&byte_offset) {
        if word_elements {
            a.bytes(&[0x41, 0x8b, 0x4a, byte_offset as u8]); // mov ecx, dword [r10 + disp8]
        } else {
            a.bytes(&[0x41, 0x0f, opcode, 0x4a, byte_offset as u8]);
            // movzx ecx, byte [r10 + disp8]
        }
    } else if word_elements {
        a.bytes(&[0x41, 0x8b, 0x8a]); // mov ecx, dword [r10 + disp32]
        a.imm32(byte_offset as u32);
    } else {
        a.bytes(&[0x41, 0x0f, opcode, 0x8a]); // movzx ecx, byte [r10 + disp32]
        a.imm32(byte_offset as u32);
    }
}

#[cfg(all(target_arch = "aarch64", unix))]
struct OneByteNativeArm64Assembler {
    code: Vec<u32>,
    labels: Vec<Option<usize>>,
    branches: Vec<(usize, usize, bool, u8)>,
}

#[cfg(all(target_arch = "aarch64", unix))]
impl OneByteNativeArm64Assembler {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            labels: Vec::new(),
            branches: Vec::new(),
        }
    }

    fn label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn bind(&mut self, label: usize) {
        self.labels[label] = Some(self.code.len());
    }

    fn insn(&mut self, instruction: u32) {
        self.code.push(instruction);
    }

    fn branch(&mut self, label: usize) {
        self.branches.push((self.code.len(), label, false, 0));
        self.insn(0);
    }

    fn branch_cond(&mut self, condition: u8, label: usize) {
        self.branches
            .push((self.code.len(), label, true, condition));
        self.insn(0);
    }

    fn finish(mut self) -> Option<Vec<u8>> {
        for (at, label, conditional, condition) in self.branches {
            let target = self.labels.get(label).and_then(|offset| *offset)?;
            let delta = isize::try_from(target)
                .ok()?
                .checked_sub(isize::try_from(at).ok()?)?;
            let instruction = if conditional {
                let delta = i32::try_from(delta).ok()?;
                if !(-(1 << 18)..(1 << 18)).contains(&delta) {
                    return None;
                }
                0x5400_0000 | (((delta as u32) & 0x7ffff) << 5) | u32::from(condition)
            } else {
                let delta = i32::try_from(delta).ok()?;
                if !(-(1 << 25)..(1 << 25)).contains(&delta) {
                    return None;
                }
                0x1400_0000 | ((delta as u32) & 0x03ff_ffff)
            };
            self.code[at] = instruction;
        }
        Some(self.code.into_iter().flat_map(u32::to_le_bytes).collect())
    }
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_mov_reg(a: &mut OneByteNativeArm64Assembler, dst: u8, src: u8) {
    // ORR Xd, XZR, Xm (the architectural MOV register alias).
    a.insn(0xaa00_03e0 | (u32::from(src) << 16) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_mov_imm64(a: &mut OneByteNativeArm64Assembler, reg: u8, value: u64) {
    for halfword in 0..4 {
        let immediate = ((value >> (halfword * 16)) & 0xffff) as u32;
        let opcode = if halfword == 0 {
            0xd280_0000
        } else {
            0xf280_0000
        };
        a.insn(opcode | (halfword << 21) | (immediate << 5) | u32::from(reg));
    }
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_add_imm(a: &mut OneByteNativeArm64Assembler, dst: u8, src: u8, immediate: u32) {
    debug_assert!(immediate < 1 << 12);
    a.insn(0x9100_0000 | (immediate << 10) | (u32::from(src) << 5) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_add_imm_w(a: &mut OneByteNativeArm64Assembler, dst: u8, src: u8, immediate: u32) {
    debug_assert!(immediate < 1 << 12);
    a.insn(0x1100_0000 | (immediate << 10) | (u32::from(src) << 5) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_add_shifted_reg(
    a: &mut OneByteNativeArm64Assembler,
    dst: u8,
    left: u8,
    right: u8,
    shift: u8,
) {
    debug_assert!(shift < 64);
    a.insn(
        0x8b00_0000
            | (u32::from(right) << 16)
            | (u32::from(shift) << 10)
            | (u32::from(left) << 5)
            | u32::from(dst),
    );
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_sub_imm(a: &mut OneByteNativeArm64Assembler, dst: u8, src: u8, immediate: u32) {
    debug_assert!(immediate < 1 << 12);
    a.insn(0xd100_0000 | (immediate << 10) | (u32::from(src) << 5) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_sub_reg(a: &mut OneByteNativeArm64Assembler, dst: u8, left: u8, right: u8) {
    a.insn(0xcb00_0000 | (u32::from(right) << 16) | (u32::from(left) << 5) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_cmp_reg(a: &mut OneByteNativeArm64Assembler, left: u8, right: u8) {
    a.insn(0xeb00_001f | (u32::from(right) << 16) | (u32::from(left) << 5));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_cmp_imm_w(a: &mut OneByteNativeArm64Assembler, reg: u8, immediate: u32) {
    debug_assert!(immediate < 1 << 12);
    a.insn(0x7100_001f | (immediate << 10) | (u32::from(reg) << 5));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_cmp_imm(a: &mut OneByteNativeArm64Assembler, reg: u8, immediate: u32) {
    debug_assert!(immediate < 1 << 12);
    a.insn(0xf100_001f | (immediate << 10) | (u32::from(reg) << 5));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_ldrb(a: &mut OneByteNativeArm64Assembler, dst: u8, base: u8, offset: u32) {
    debug_assert!(offset < 1 << 12);
    a.insn(0x3940_0000 | (offset << 10) | (u32::from(base) << 5) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_ldr_w(a: &mut OneByteNativeArm64Assembler, dst: u8, base: u8, offset: u32) {
    debug_assert!(offset < 1 << 12);
    a.insn(0xb940_0000 | (offset << 10) | (u32::from(base) << 5) | u32::from(dst));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn arm64_str64(a: &mut OneByteNativeArm64Assembler, src: u8, base: u8, offset: u32) {
    debug_assert_eq!(offset % 8, 0);
    debug_assert!(offset / 8 < 1 << 12);
    a.insn(0xf900_0000 | ((offset / 8) << 10) | (u32::from(base) << 5) | u32::from(src));
}

#[cfg(all(target_arch = "aarch64", unix))]
fn compile_one_byte_native_aarch64(
    ops: &[OneByteNativeOp],
    word_elements: bool,
) -> Option<Vec<u8>> {
    // AArch64 Unix ABI arguments enter in x0..x5. x19..x24 retain them, x25 retains the last
    // candidate, and x16 is the indirect-call scratch register.
    let mut a = OneByteNativeArm64Assembler::new();
    let no_match = a.label();
    let interrupted = a.label();
    let ret = a.label();
    let fail = a.label();
    let loop_start = a.label();

    for (left, right) in [(19u8, 20u8), (21, 22), (23, 24), (25, 30)] {
        a.insn(0xa9bf_0000 | (u32::from(right) << 10) | (31u32 << 5) | u32::from(left));
    }
    for (dst, src) in [(19, 0), (20, 1), (21, 2), (22, 3), (23, 4), (24, 5)] {
        arm64_mov_reg(&mut a, dst, src);
    }
    arm64_cmp_reg(&mut a, 21, 20);
    a.branch_cond(8, no_match); // HI: start > length
    let width = ops.iter().filter(|op| op.consumes_element()).count();
    arm64_sub_imm(&mut a, 25, 20, width as u32);
    arm64_sub_reg(&mut a, 9, 20, 21);
    arm64_cmp_imm(&mut a, 9, width as u32);
    a.branch_cond(3, no_match); // LO: remaining length < width

    a.bind(loop_start);
    arm64_mov_reg(&mut a, 0, 23);
    arm64_mov_imm64(&mut a, 16, regexp_native_poll as *const () as usize as u64);
    // AAPCS64 6.1.1: NZCV is undefined across a public call. Test the returned
    // status, not flags left by Rust; a false interrupt with w0 == 0 would be
    // encoded below as a successful empty match and could stall global replace.
    a.insn(0xd63f_0200); // blr x16
    arm64_cmp_imm_w(&mut a, 0, 0);
    a.branch_cond(1, interrupted); // NE: poll returned a reason
    if word_elements {
        arm64_add_shifted_reg(&mut a, 10, 19, 21, 2); // add x10, x19, x21, lsl #2
    } else {
        arm64_add_shifted_reg(&mut a, 10, 19, 21, 0); // add x10, x19, x21
    }

    let mut offset = 0u32;
    for op in ops {
        match op {
            OneByteNativeOp::Char(expected) => {
                debug_assert!(*expected < 0x100);
                if word_elements {
                    arm64_ldr_w(&mut a, 9, 10, offset * 4);
                } else {
                    arm64_ldrb(&mut a, 9, 10, offset);
                }
                arm64_cmp_imm_w(&mut a, 9, *expected);
                a.branch_cond(1, fail); // NE
                offset += 1;
            }
            OneByteNativeOp::Any => {
                if word_elements {
                    arm64_ldr_w(&mut a, 9, 10, offset * 4);
                } else {
                    arm64_ldrb(&mut a, 9, 10, offset);
                }
                arm64_cmp_imm_w(&mut a, 9, b'\n' as u32);
                a.branch_cond(0, fail); // EQ
                arm64_cmp_imm_w(&mut a, 9, b'\r' as u32);
                a.branch_cond(0, fail); // EQ
                offset += 1;
            }
            OneByteNativeOp::Class(class) => {
                let lut = class.ascii_lut.as_ref()?.as_ptr() as usize as u64;
                if word_elements {
                    arm64_ldr_w(&mut a, 9, 10, offset * 4);
                } else {
                    arm64_ldrb(&mut a, 9, 10, offset);
                }
                arm64_mov_imm64(&mut a, 16, lut);
                a.insn(0x8b09_0210); // add x16, x16, x9 (the byte is zero-extended)
                arm64_ldrb(&mut a, 9, 16, 0);
                arm64_cmp_imm_w(&mut a, 9, 0);
                a.branch_cond(0, fail); // EQ
                offset += 1;
            }
            OneByteNativeOp::AssertStart => {
                arm64_cmp_imm(&mut a, 21, 0);
                a.branch_cond(1, fail); // NE
            }
            OneByteNativeOp::AssertEnd => {
                arm64_cmp_reg(&mut a, 21, 25);
                a.branch_cond(1, fail); // NE
            }
            OneByteNativeOp::Many { .. } => return None,
        }
    }

    arm64_str64(&mut a, 21, 24, 0);
    arm64_add_imm(&mut a, 0, 21, width as u32);
    arm64_str64(&mut a, 0, 24, 8);
    a.insn(0x5280_0020); // mov w0, #1
    a.branch(ret);

    a.bind(fail);
    arm64_cmp_imm_w(&mut a, 22, 0);
    a.branch_cond(1, no_match); // NE: sticky
    arm64_cmp_reg(&mut a, 21, 25);
    a.branch_cond(2, no_match); // HS: current >= last
    arm64_add_imm(&mut a, 21, 21, 1);
    a.branch(loop_start);

    a.bind(interrupted);
    arm64_add_imm_w(&mut a, 0, 0, 1);
    a.branch(ret);

    a.bind(no_match);
    a.insn(0x5280_0000); // mov w0, #0

    a.bind(ret);
    for (left, right) in [(25u8, 30u8), (23, 24), (21, 22), (19, 20)] {
        a.insn(0xa8c1_0000 | (u32::from(right) << 10) | (31u32 << 5) | u32::from(left));
    }
    a.insn(0xd65f_03c0); // ret
    a.finish()
}

struct OneByteNativeProgram {
    ops: Box<[OneByteNativeOp]>,
    machine_code: Option<OneByteNativeCode>,
}

impl OneByteNativeProgram {
    fn compile(prog: &[Inst], multiline: bool) -> Option<Self> {
        let body = prog.get(1..prog.len().checked_sub(2)?)?;
        if body.is_empty() {
            return None;
        }
        if body.len() == 1 {
            if let Inst::Many {
                rep,
                min,
                max,
                greedy,
            } = &body[0]
            {
                if one_byte_native_rep(rep) {
                    let ops = vec![OneByteNativeOp::Many {
                        rep: rep.clone(),
                        min: *min,
                        max: *max,
                        greedy: *greedy,
                    }]
                    .into_boxed_slice();
                    return Some(Self {
                        machine_code: None,
                        ops,
                    });
                }
            }
        }
        let mut has_non_literal = false;
        let mut ops = Vec::with_capacity(body.len());
        for instruction in body {
            match instruction {
                Inst::Char(c) if *c < 0x100 => ops.push(OneByteNativeOp::Char(*c)),
                Inst::Any => {
                    has_non_literal = true;
                    ops.push(OneByteNativeOp::Any);
                }
                Inst::Class(class) if class.ascii_lut.is_some() => {
                    has_non_literal = true;
                    ops.push(OneByteNativeOp::Class(class.clone()));
                }
                Inst::AssertStart | Inst::AssertEnd if !multiline => {
                    has_non_literal = true;
                    ops.push(if matches!(instruction, Inst::AssertStart) {
                        OneByteNativeOp::AssertStart
                    } else {
                        OneByteNativeOp::AssertEnd
                    });
                }
                _ => return None,
            }
        }
        // Pure literals already use the eager SIMD-friendly literal path. The tiered path is for
        // the next useful subset, where the old matcher paid one instruction dispatch per atom.
        if !has_non_literal {
            return None;
        }
        let ops = ops.into_boxed_slice();
        // The AArch64 class emitter still needs an optimized-release audit: on this host it can
        // lose termination on a no-match tail even though the same program is sound in debug.
        // Keep the checked native Rust matcher (and terminal Many specialization) available while
        // declining only the unsafe LUT machine mapping. The byte/word fixed-width path remains
        // enabled for the class-free subset.
        // Assertions need to participate in the candidate scan, including the failed-candidate
        // path. Keep those programs on the checked native Rust matcher until the architecture
        // emitters have a dedicated assertion differential gate; this preserves the tier-up win
        // for the atom matcher without allowing an optimized executable mapping to turn an
        // anchored no-match into a false empty match (which affects jQuery's vendor-property
        // detection).
        let machine_code = if ops.iter().any(|op| {
            matches!(
                op,
                OneByteNativeOp::AssertStart | OneByteNativeOp::AssertEnd
            )
        }) || (cfg!(target_arch = "aarch64")
            && ops.iter().any(|op| matches!(op, OneByteNativeOp::Class(_))))
        {
            None
        } else {
            OneByteNativeCode::compile(&ops)
        };
        Some(Self { machine_code, ops })
    }

    fn retained_bytes(&self) -> usize {
        self.ops
            .len()
            .saturating_mul(std::mem::size_of::<OneByteNativeOp>())
    }

    #[cfg(test)]
    fn find<S: OneByteSubject + ?Sized>(
        &self,
        subject: &S,
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        self.find_interpreted(subject, start, sticky, control)
    }

    fn find_ascii(
        &self,
        subject: &[u8],
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        if let Some(code) = &self.machine_code {
            return code.find_ascii(subject, start, sticky, control);
        }
        self.find_interpreted(subject, start, sticky, control)
    }

    fn find_one_byte(
        &self,
        subject: &[u32],
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        if let Some(code) = &self.machine_code {
            return code.find_one_byte(subject, start, sticky, control);
        }
        self.find_interpreted(subject, start, sticky, control)
    }

    fn find_interpreted<S: OneByteSubject + ?Sized>(
        &self,
        subject: &S,
        start: usize,
        sticky: bool,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        if let [OneByteNativeOp::Many {
            rep,
            min,
            max,
            greedy,
        }] = self.ops.as_ref()
        {
            return find_terminal_one_byte_many(
                subject, start, rep, *min, *max, *greedy, sticky, control,
            );
        }
        let width = self.ops.iter().filter(|op| op.consumes_element()).count();
        if start > subject.len() || width > subject.len().saturating_sub(start) {
            return Ok(None);
        }
        let last = subject.len() - width;
        let mut from = start;
        while from <= last {
            if from & INTERRUPT_POLL_MASK == 0 {
                poll_interrupt(control)?;
            }
            let mut matched = true;
            let mut offset = 0;
            for (index, op) in self.ops.iter().enumerate() {
                if index & INTERRUPT_POLL_MASK == 0 {
                    poll_interrupt(control)?;
                }
                let hit = match op {
                    OneByteNativeOp::AssertStart => from == 0,
                    OneByteNativeOp::AssertEnd => from == last,
                    _ => {
                        let element = subject.element_at(from + offset);
                        offset += 1;
                        op.matches(element)
                    }
                };
                if !hit {
                    matched = false;
                    break;
                }
            }
            if matched {
                return Ok(Some((from, from + width)));
            }
            if sticky {
                return Ok(None);
            }
            from += 1;
        }
        Ok(None)
    }
}

fn one_byte_native_rep(rep: &Rep) -> bool {
    match rep {
        Rep::Char(c) => *c < 0x100,
        Rep::Any => true,
        Rep::Class(class) => class.ascii_lut.is_some(),
    }
}

fn find_terminal_one_byte_many<S: OneByteSubject + ?Sized>(
    subject: &S,
    start: usize,
    rep: &Rep,
    min: usize,
    max: Option<usize>,
    greedy: bool,
    sticky: bool,
    control: &crate::RuntimeInterrupt,
) -> MatchResult<(usize, usize)> {
    if start > subject.len() {
        return Ok(None);
    }
    let cap = max.unwrap_or(usize::MAX);
    let mut from = start;
    while from <= subject.len() {
        if from & INTERRUPT_POLL_MASK == 0 {
            poll_interrupt(control)?;
        }
        let room = subject.len() - from;
        let mut avail = 0usize;
        while avail < cap
            && avail < room
            && one_byte_rep_matches(rep, subject.element_at(from + avail))
        {
            avail += 1;
            if avail & INTERRUPT_POLL_MASK == 0 {
                poll_interrupt(control)?;
            }
        }
        if avail >= min {
            let count = if greedy { avail } else { min };
            return Ok(Some((from, from + count)));
        }
        if sticky {
            return Ok(None);
        }
        from += 1;
    }
    Ok(None)
}

#[derive(Clone)]
enum Inst {
    Char(u32),
    Any,
    Class(Rc<CharClass>),
    StringSet(Rc<StringSet>),
    StringSetRepeat {
        set: Rc<StringSet>,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
    Save(usize),
    Split(usize, usize),
    Jmp(usize),
    Match,
    AssertStart,
    AssertEnd,
    WordBoundary(bool),
    Backref(usize),
    /// `\k<name>` where the name is shared by several groups: matches via whichever captured.
    BackrefAlt(Rc<Vec<usize>>),
    /// Reset capture slots for groups `lo..=hi` at the start of a quantifier iteration.
    ClearCaps(usize, usize),
    Look {
        negate: bool,
        prog: Rc<Vec<Inst>>,
    },
    /// `(?<=…)` / `(?<!…)`: the body must match text ending at the current position.
    LookBehind {
        negate: bool,
        prog: Rc<Vec<Inst>>,
    },
    /// A repeated single-character matcher (`a*`, `\w+`, `.{2,5}`, `\p{L}+`). Consumed iteratively,
    /// with one backtrack record for the whole run rather than one per character.
    Many {
        rep: Rep,
        min: usize,
        max: Option<usize>,
        greedy: bool,
    },
    /// `(?ims-ims:…)` inline modifiers: push a new `(icase, multiline, dotall)` flag set for the
    /// group body (`Some` = add/remove, `None` = inherit), then `PopFlags` restores it.
    PushFlags(Option<bool>, Option<bool>, Option<bool>),
    PopFlags,
    /// RepeatMatcher's empty-iteration rule: `SetMark` records the position entering an optional
    /// quantifier iteration; `CheckProgress` FAILS (forcing backtracking into the body or out of
    /// the loop) when the iteration consumed nothing.
    SetMark(usize),
    CheckProgress(usize),
    /// ECMA-262 RepeatMatcher for a counted quantifier too large to unroll (`(?:ab){0,5000}`):
    /// `CounterReset` starts the loop's iteration count, kept in a `marks` slot; `CounterIncr`
    /// ends an iteration. Both roll back on backtracking. `CountedBranch` is the loop head: it
    /// takes a mandatory iteration while the count is below `min`, leaves the loop once the count
    /// reaches `max`, and otherwise branches between an optional iteration and `exit`, greedy
    /// first or lazy first. Every iteration runs the same `body`.
    CounterReset(usize),
    CounterIncr(usize),
    CountedBranch {
        id: usize,
        min: usize,
        max: Option<usize>,
        greedy: bool,
        body: usize,
        exit: usize,
    },
    /// `CheckProgress` at the end of a counted loop's body: only an optional iteration, one that
    /// began with the count in mark `counter` at or past `min`, must have consumed input.
    CheckCountedProgress {
        mark: usize,
        counter: usize,
        min: usize,
    },
}

/// A single-codepoint matcher, for the `Inst::Many` fast path.
#[derive(Clone)]
enum Rep {
    Char(u32),
    Any,
    Class(Rc<CharClass>),
}

/// What the compiled program says about how a match can begin.
enum FirstFilter {
    /// No usable information — the scan tries every position.
    None,
    /// Every path first asserts `^` in non-multiline mode: a match can only begin at position 0,
    /// so one attempt decides the whole scan.
    Anchored,
    /// Every path begins by consuming one element matching one of these atoms; positions whose
    /// element matches none can be skipped without entering the backtracker. The predicate is a
    /// superset of what the matcher accepts, so a pass is never wrong — only a reject is binding.
    Atoms(Vec<Rep>),
}

/// Compute the [`FirstFilter`] by ε-walking the program from its entry: through saves, jumps,
/// splits, capture clears and marks, collecting the first thing each path does. A path that
/// begins with `^` can only begin where `^` holds; the returned flag says such paths exist, so a
/// match may also begin at the start of the input (or of a line, in multiline mode), as in
/// `(^|\s)name`. Anything else not modelled (other assertions, backrefs, lookarounds, inline flags,
/// or an ε-reachable `Match` — an empty-matchable pattern) disables the filter.
fn first_filter(prog: &[Inst], multiline: bool) -> (FirstFilter, bool) {
    let mut atoms: Vec<Rep> = Vec::new();
    let mut asserts = 0usize;
    let mut stack = vec![0usize];
    let mut seen = vec![false; prog.len()];
    while let Some(pc) = stack.pop() {
        if seen[pc] {
            continue;
        }
        seen[pc] = true;
        match &prog[pc] {
            Inst::Save(_)
            | Inst::ClearCaps(..)
            | Inst::SetMark(_)
            | Inst::CounterReset(_)
            | Inst::CounterIncr(_) => stack.push(pc + 1),
            // Every way out of a counted loop's head may begin a path.
            Inst::CountedBranch { body, exit, .. } => {
                stack.push(*body);
                stack.push(*exit);
            }
            Inst::Jmp(t) => stack.push(*t),
            Inst::Split(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Inst::Char(c) => atoms.push(Rep::Char(*c)),
            Inst::Any => atoms.push(Rep::Any),
            Inst::Class(cc) => atoms.push(Rep::Class(cc.clone())),
            Inst::StringSet(set) => {
                atoms.push(Rep::Class(Rc::new(clone_class(&set.first))));
                if set.empty {
                    stack.push(pc + 1);
                }
            }
            Inst::StringSetRepeat { set, min, .. } => {
                atoms.push(Rep::Class(Rc::new(clone_class(&set.first))));
                if *min == 0 || set.empty {
                    stack.push(pc + 1);
                }
            }
            Inst::Many { rep, min, .. } => {
                atoms.push(rep.clone());
                if *min == 0 {
                    stack.push(pc + 1); // may consume nothing — the next inst also "begins" a path
                }
            }
            Inst::AssertStart => asserts += 1,
            _ => return (FirstFilter::None, false),
        }
    }
    if asserts > 0 {
        if atoms.is_empty() && !multiline {
            (FirstFilter::Anchored, false)
        } else {
            (FirstFilter::Atoms(atoms), true)
        }
    } else if !atoms.is_empty() {
        (FirstFilter::Atoms(atoms), false)
    } else {
        (FirstFilter::None, false)
    }
}

#[derive(Default, Clone)]
struct CharClass {
    negate: bool,
    ranges: Vec<(u32, u32)>,
    /// Builtin sub-classes by letter: 'd','w','s' (and uppercase negated forms expanded inline).
    builtins: Vec<char>,
    /// Unicode property escapes `\p{…}` / `\P{…}`: `(negated, sorted codepoint ranges)`.
    props: Vec<(bool, &'static [(u32, u32)])>,
    /// Exact membership for non-case-insensitive inputs in the byte range. Compiled classes
    /// exercise this table for one-byte subjects; all other inputs retain the range/property path.
    ascii_lut: Option<Box<[bool; 256]>>,
}

impl CharClass {
    fn matches(&self, u: u32, icase: bool, unicode: bool) -> bool {
        if !icase {
            if let Some(lut) = &self.ascii_lut {
                if let Some(&hit) = lut.get(u as usize) {
                    return hit;
                }
            }
        }
        let mut hit = self.matches_raw2(u, icase, unicode);
        let c = char::from_u32(u);
        if !hit && icase {
            if let Some(c) = c {
                if unicode {
                    // Try every member of the character's case-fold orbit.
                    for alt in fold_orbit(u) {
                        if alt != u && self.matches_raw2(alt, icase, unicode) {
                            hit = true;
                            break;
                        }
                    }
                } else {
                    // Legacy Canonicalize: compare via simple uppercase, never folding a
                    // non-ASCII character onto an ASCII one.
                    let cu = canonicalize_legacy(c);
                    if cu != c && self.matches_raw2(cu as u32, icase, unicode) {
                        hit = true;
                    }
                    // A member whose canonical form equals cu also matches (/[k]/i vs 'K').
                    if !hit {
                        for alt in c.to_lowercase().chain(c.to_uppercase()) {
                            if alt != c
                                && canonicalize_legacy(alt) == cu
                                && self.matches_raw2(alt as u32, icase, unicode)
                            {
                                hit = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        hit ^ self.negate
    }
    fn matches_raw2(&self, u: u32, icase: bool, unicode: bool) -> bool {
        // Class membership is decided in true code-point space: smuggled surrogate atoms in the
        // class's own ranges decode to their surrogate values.
        for &(lo, hi) in &self.ranges {
            if u >= lo && u <= hi {
                return true;
            }
        }
        for &b in &self.builtins {
            if builtin_matches_ic(b, u, icase, unicode) {
                return true;
            }
        }
        for &(neg, ranges) in &self.props {
            // Ranges are sorted and disjoint: binary-search for the one that could contain `u`.
            let in_range = ranges
                .binary_search_by(|&(lo, hi)| {
                    if u < lo {
                        std::cmp::Ordering::Greater
                    } else if u > hi {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .is_ok();
            if in_range ^ neg {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Default)]
struct StringTrieNode {
    edges: Vec<(u32, usize)>,
    terminal: bool,
}

#[derive(Clone)]
struct StringTrie {
    nodes: Vec<StringTrieNode>,
    max_depth: usize,
}

impl StringTrie {
    fn build(strings: &[Vec<char>], backwards: bool) -> Self {
        let mut trie = Self {
            nodes: vec![StringTrieNode::default()],
            max_depth: 0,
        };
        for string in strings {
            debug_assert!(string.len() > 1);
            trie.max_depth = trie.max_depth.max(string.len());
            let mut node = 0usize;
            if backwards {
                for character in string.iter().rev() {
                    node = trie.insert_edge(node, *character as u32);
                }
            } else {
                for character in string {
                    node = trie.insert_edge(node, *character as u32);
                }
            }
            trie.nodes[node].terminal = true;
        }
        trie
    }

    fn insert_edge(&mut self, node: usize, code_point: u32) -> usize {
        match self.nodes[node]
            .edges
            .binary_search_by_key(&code_point, |edge| edge.0)
        {
            Ok(index) => self.nodes[node].edges[index].1,
            Err(index) => {
                let child = self.nodes.len();
                self.nodes.push(StringTrieNode::default());
                self.nodes[node].edges.insert(index, (code_point, child));
                child
            }
        }
    }
}

/// ECMA-262 §22.2.2.7 compiles a UnicodeSets class's string elements as alternatives sorted by
/// descending length, followed by its singleton class and then its empty element. Two tries share
/// common prefixes for forward and backwards matching while [`Matcher::run`] still tries
/// the continuation at every matching length in that normative order.
#[derive(Clone)]
struct StringSet {
    forward: StringTrie,
    backward: StringTrie,
    singles: CharClass,
    first: CharClass,
    empty: bool,
}

impl StringSet {
    fn new(strings: Vec<Vec<char>>, singles: CharClass, empty: bool) -> Self {
        let forward = StringTrie::build(&strings, false);
        let backward = StringTrie::build(&strings, true);
        let mut first = clone_class(&singles);
        for &(code_point, _) in &forward.nodes[0].edges {
            first.ranges.push((code_point, code_point));
        }
        Self {
            forward,
            backward,
            singles,
            first,
            empty,
        }
    }
}

fn builtin_matches_ic(b: char, u: u32, icase: bool, unicode: bool) -> bool {
    let c = char::from_u32(u);
    match b {
        'd' => c.map(|c| c.is_ascii_digit()).unwrap_or(false),
        'D' => !c.map(|c| c.is_ascii_digit()).unwrap_or(false),
        'w' => is_word_ic(u, icase, unicode),
        'W' => !is_word_ic(u, icase, unicode),
        's' => c.map(js_whitespace).unwrap_or(false),
        'S' => !c.map(js_whitespace).unwrap_or(false),
        _ => false,
    }
}

/// A JS LineTerminator code point.
fn is_line_terminator_u32(c: u32) -> bool {
    matches!(c, 0x0A | 0x0D | 0x2028 | 0x2029)
}

fn is_word(c: u32) -> bool {
    char::from_u32(c)
        .map(|c| c.is_ascii_alphanumeric() || c == '_')
        .unwrap_or(false)
}

/// GetWordCharacters: under unicode case-insensitive matching, characters whose case fold lands
/// in [A-Za-z0-9_] (ſ, K) are word characters too.
fn is_word_ic(c: u32, icase: bool, unicode: bool) -> bool {
    if is_word(c) {
        return true;
    }
    if !(icase && unicode) {
        return false;
    }
    fold_orbit(c).any(is_word)
}

/// The canonical full case-folding representative of a code point (identity outside any orbit).
fn fold_canon(u: u32) -> u32 {
    match crate::regex_fold::FOLD_CANON.binary_search_by_key(&u, |&(m, _)| m) {
        Ok(k) => crate::regex_fold::FOLD_CANON[k].1,
        Err(_) => u,
    }
}

/// Every member of `u`'s case-fold orbit (just `u` when it has none).
fn fold_orbit(u: u32) -> impl Iterator<Item = u32> {
    let canon = fold_canon(u);
    let t = crate::regex_fold::FOLD_ORBITS;
    let lo = t.partition_point(|&(c, _)| c < canon);
    let hi = t.partition_point(|&(c, _)| c <= canon);
    let mut own = if lo == hi { Some(u) } else { None };
    t[lo..hi]
        .iter()
        .map(|&(_, m)| m)
        .chain(std::iter::from_fn(move || own.take()))
}

/// The JS WhiteSpace + LineTerminator set: includes U+FEFF and NBSP, but NOT U+0085 (NEL) or
/// other control characters Rust's `is_whitespace` accepts.
pub(crate) fn js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{0B}' | '\u{0C}' | '\r' | ' ' | '\u{A0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

fn uprop_has(name: &str, c: char) -> bool {
    let u = c as u32;
    crate::unicode_props::lookup(name, None).is_some_and(|r| {
        r.binary_search_by(|&(lo, hi)| {
            if u < lo {
                std::cmp::Ordering::Greater
            } else if u > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
    })
}
/// IdentifierStart for a RegExp capture-group name (ID_Start ∪ {$, _}).
/// The legacy (non-Unicode) Canonicalize: the simple uppercase mapping, except that a non-ASCII
/// character never canonicalizes onto an ASCII one (so /\u212a/i does not match 'K' without /u).
fn canonicalize_legacy(c: char) -> char {
    let mut up = c.to_uppercase();
    let (first, rest) = (up.next(), up.next());
    match (first, rest) {
        (Some(u), None) => {
            if (c as u32) >= 128 && (u as u32) < 128 {
                c
            } else {
                u
            }
        }
        _ => c,
    }
}

/// A regular-expression SyntaxCharacter (the only chars an identity escape may name in /u mode).
fn is_regex_syntax_char(c: char) -> bool {
    matches!(
        c,
        '^' | '$' | '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|'
    )
}

fn regex_ident_start(c: char) -> bool {
    if c.is_ascii() {
        return c == '$' || c == '_' || c.is_ascii_alphabetic();
    }
    uprop_has("ID_Start", c)
}
/// IdentifierPart for a capture-group name (ID_Continue ∪ {$, _, ZWNJ, ZWJ}).
fn regex_ident_part(c: char) -> bool {
    if c.is_ascii() {
        return c == '$' || c == '_' || c.is_ascii_alphanumeric();
    }
    c == '\u{200C}' || c == '\u{200D}' || uprop_has("ID_Continue", c)
}

// ---------------------------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------------------------

enum Node {
    Empty,
    Char(u32),
    Any,
    Class(CharClass),
    /// A UnicodeSets character class containing multi-code-point strings. The compact trie keeps
    /// the specification's descending-length alternatives without a flat O(strings) Split chain.
    StringSet(Rc<StringSet>),
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    Group(Option<usize>, Box<Node>),
    Repeat(Box<Node>, usize, Option<usize>, bool),
    Start,
    End,
    WordB(bool),
    Backref(usize),
    /// `\k<name>` — resolved to a group index after the whole pattern is parsed.
    NamedBackref(String),
    /// `\k<name>` naming several duplicate groups — matches via whichever of them captured.
    BackrefAlt(Vec<usize>),
    Look(bool, Box<Node>),
    /// `(?<=…)` / `(?<!…)` lookbehind: assert the body matches text *ending* at the current position.
    LookBehind(bool, Box<Node>),
    /// `(?ims-ims:…)` inline-modifier group: `(add, remove)` flag deltas over `(i, m, s)`.
    Modifier {
        add: (bool, bool, bool),
        remove: (bool, bool, bool),
        inner: Box<Node>,
    },
}

/// Dropping a tree through the derived glue recurses once per level. Detach nested children into a
/// worklist instead, so dropping a deeply nested pattern uses constant native stack.
impl Drop for Node {
    fn drop(&mut self) {
        let mut pending = Vec::new();
        self.detach_children(&mut pending);
        while let Some(mut node) = pending.pop() {
            node.detach_children(&mut pending);
        }
    }
}

impl Node {
    fn detach_children(&mut self, out: &mut Vec<Node>) {
        match self {
            Node::Concat(nodes) | Node::Alt(nodes) => out.append(nodes),
            Node::Group(_, inner)
            | Node::Repeat(inner, ..)
            | Node::Look(_, inner)
            | Node::LookBehind(_, inner)
            | Node::Modifier { inner, .. } => {
                out.push(std::mem::replace(&mut **inner, Node::Empty))
            }
            _ => {}
        }
    }
}

struct Parser {
    chars: Vec<char>,
    /// Total capturing groups in the whole pattern (prescanned): Annex B decides decimal escapes
    /// (backreference vs legacy octal) against this count.
    total_groups: usize,
    pos: usize,
    ngroups: usize,
    names: Vec<(String, usize)>,
    /// `u` or `v` flag: enables Unicode mode (notably `\p{…}` property escapes).
    unicode: bool,
    /// Whether `\k` is a named back-reference here: true in Unicode mode, or when the pattern
    /// contains a named group (`(?<name>…)`). Otherwise `\k` is the literal character `k` (Annex B).
    /// The `v` flag: classes are ClassSetExpressions (nested classes, `&&`, `--`, `\q{}`).
    unicode_sets: bool,
    named_mode: bool,
    /// `\k<name>` references collected during parsing, validated against `names` afterwards.
    name_refs: Vec<String>,
    /// Groups and nested classes open at the current position (see [`MAX_PATTERN_NESTING`]).
    depth: usize,
    /// Lookarounds open at the current position (see [`MAX_LOOKAROUND_NESTING`]).
    look_depth: usize,
}

/// How deeply groups and UnicodeSets classes may nest; a deeper pattern is a SyntaxError, as V8
/// reports an oversized pattern. The parser, the compiler and the other passes over the pattern
/// tree recurse once per level, growing the native stack on demand (see [`with_pattern_stack`]),
/// so this bounds the memory they can claim. Real patterns, including generated ones such as word
/// tries, nest a few dozen levels.
#[cfg(not(target_arch = "wasm32"))]
const MAX_PATTERN_NESTING: usize = 1000;
/// wasm32 cannot switch to a fresh stack segment, so there the bound alone must keep the
/// recursion within the default 1 MiB stack.
#[cfg(target_arch = "wasm32")]
const MAX_PATTERN_NESTING: usize = 250;

/// How deeply lookarounds may nest. Each compiles to a sub-program that dropping, memory
/// accounting and analysis of the compiled matcher walk recursively, so they are bounded more
/// tightly than groups; real patterns nest lookarounds two or three levels.
const MAX_LOOKAROUND_NESTING: usize = 128;

/// Headroom one recursion level over a pattern may use before checking again (a debug build's
/// parser takes about 10 KiB per nesting level, an optimized one about 1.5 KiB), and the size of
/// each stack segment added when it runs out.
#[cfg(not(target_arch = "wasm32"))]
const PATTERN_STACK_RED_ZONE: usize = 128 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const PATTERN_STACK_SEGMENT: usize = 1024 * 1024;

/// Run one level of a recursive pass over a pattern, on a fresh stack segment when the current
/// one is nearly exhausted: a pattern compiled deep in a script's recursion, or nested
/// [`MAX_PATTERN_NESTING`] levels deep, must not overflow the native stack.
#[inline]
fn with_pattern_stack<R>(pass: impl FnOnce() -> R) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    {
        stacker::maybe_grow(PATTERN_STACK_RED_ZONE, PATTERN_STACK_SEGMENT, pass)
    }
    #[cfg(target_arch = "wasm32")]
    {
        pass()
    }
}

/// The element sequence regular expressions operate over. In unicode (`u`/`v`) mode an element
/// is a code point; otherwise it is a UTF-16 code unit. Surrogate units/code points are carried
/// as their jstr-smuggled plane-16 scalars so every element is a valid `char` — an astral
/// character in a non-unicode pattern or subject is therefore TWO elements (its two halves).
pub fn pattern_elements(unicode: bool, s: &str) -> Vec<char> {
    if unicode {
        crate::jstr::code_points(s)
            .into_iter()
            .map(elem_of_cp)
            .collect()
    } else {
        crate::jstr::units(s)
            .into_iter()
            .map(|u| {
                if (0xD800..0xE000).contains(&(u as u32)) {
                    crate::jstr::smuggle(u)
                } else {
                    char::from_u32(u as u32).unwrap()
                }
            })
            .collect()
    }
}

/// The true code-point value of a pattern/subject element (smuggled surrogates decode).
fn cp_of_elem(c: char) -> u32 {
    match crate::jstr::smuggled(c) {
        Some(u) => u as u32,
        None => c as u32,
    }
}

fn elem_of_cp(cp: u32) -> char {
    if (0xD800..0xE000).contains(&cp) {
        crate::jstr::smuggle(cp as u16)
    } else {
        char::from_u32(cp).unwrap()
    }
}

/// The source string a prepared subject keeps (and its window's parent's), the GC edges of a
/// cached subject (gc-heap).
#[cfg(feature = "gc-heap")]
impl crate::gc_trace::GcTrace for ReText {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        crate::gc_trace::GcTrace::trace_gc(&self.ascii_src, sink);
        if let Some(parent) = &self.parent {
            crate::gc_trace::GcTrace::trace_gc(&**parent, sink);
        }
    }
}

/// A subject string prepared for matching: its elements plus each element's unit offset.
/// A flat subject's `unit_of` is `None` when element index == unit offset; otherwise it records
/// each element's offset and the final unit length. A window resolves both elements and offsets
/// through one flat parent. JS-visible indices (lastIndex, match.index) remain relative units.
pub struct ReText {
    /// Flat wide elements — empty for ASCII subjects and windows. Matchers use `elements()`;
    /// ASCII subjects match over `ascii_src`'s bytes with no element materialization.
    pub elems: Vec<u32>,
    pub unit_of: Option<Vec<usize>>,
    /// Visible element count, including windows with no privately owned elements.
    n_elems: usize,
    subject_shape: u8,
    unicode: bool,
    /// The source string when it is pure ASCII (element index == byte index): matching runs
    /// over its bytes and `slice` copies straight out of it.
    ascii_src: Option<crate::lstr::LStr>,
    /// A window owns one flat prepared subject, never another window. Matcher input is sliced
    /// before entry, so anchors, lookbehind and bounds cannot inspect the excluded prefix/tail.
    parent: Option<std::rc::Rc<ReText>>,
    elem_base: usize,
    unit_base: usize,
}

impl ReText {
    /// Retained allocation size excluding the source string itself (the identity cache owns and
    /// accounts that allocation once alongside this prepared view).
    pub(crate) fn heap_bytes(&self) -> usize {
        self.own_heap_bytes().saturating_add(
            self.parent
                .as_ref()
                .map_or(0, |parent| parent.own_heap_bytes()),
        )
    }

    fn own_heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(
                self.elems
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.unit_of.as_ref().map_or(0, |offsets| {
                offsets
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>())
            }))
    }

    /// Attribute the prepared view and its pinned source through the managed-memory visitor.
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        if let Some(source) = &self.ascii_src {
            visitor.lstr(source);
        }
        if let Some(parent) = &self.parent {
            visitor.re_text(parent);
        }
        self.own_heap_bytes()
    }

    /// The caller proves `source` is precisely the parent's unit range. Exact element
    /// boundaries are checked here; roots and their tables remain immutable and pinned.
    pub(crate) fn window(
        parent: std::rc::Rc<ReText>,
        source: &crate::lstr::LStr,
        start: usize,
        end: usize,
    ) -> Option<Self> {
        if parent.parent.is_some() || end > parent.unit_index(parent.len()) || start > end {
            return None;
        }
        debug_assert_eq!(crate::jstr::unit_len(source), end - start);
        let first = parent.elem_at_unit(start);
        let last = parent.elem_at_unit(end);
        if parent.unit_index(first) != start || parent.unit_index(last) != end {
            return None;
        }
        // Every non-ASCII UTF-8 scalar needs more bytes than UTF-16 units, including
        // astral characters and smuggled surrogates. Equality for this proven unit
        // range therefore proves the visible source is ASCII without scanning it.
        // A wide prefix/tail must not prevent the slice from using the byte matcher.
        let ascii = source.len() == end - start;
        Some(Self {
            elems: Vec::new(),
            unit_of: None,
            n_elems: last - first,
            subject_shape: if ascii {
                SUBJECT_ASCII
            } else {
                parent.subject_shape
            },
            unicode: parent.unicode,
            ascii_src: ascii.then(|| source.clone()),
            parent: (!ascii).then_some(parent),
            elem_base: first,
            unit_base: start,
        })
    }

    #[inline]
    fn elements(&self) -> &[u32] {
        match &self.parent {
            Some(parent) => &parent.elems[self.elem_base..self.elem_base + self.n_elems],
            None => &self.elems,
        }
    }

    /// Prepare `s` for matching, keeping the caller's `Rc` for zero-copy ASCII slicing.
    pub fn new_rc(unicode: bool, s: &crate::lstr::LStr) -> ReText {
        // Engine strings maintain an exact one-way ASCII hint in their allocation header.
        // RegExp workloads commonly stream many distinct ASCII subjects through the tiny
        // identity cache; consulting the hint avoids rescanning every subject just to select
        // the byte matcher.
        if s.ascii_hint() {
            return ReText {
                elems: Vec::new(),
                unit_of: None,
                n_elems: s.len(),
                subject_shape: SUBJECT_ASCII,
                unicode,
                ascii_src: Some(s.clone()),
                parent: None,
                elem_base: 0,
                unit_base: 0,
            };
        }
        // Keep the engine string itself: `LStr::clone` copies a handle, and its immutable
        // bytes can be matched and sliced directly.
        Self::build(unicode, s, Some(s.clone()))
    }

    fn build(unicode: bool, s: &str, src: Option<crate::lstr::LStr>) -> ReText {
        // ASCII: elements are the bytes, and element index == unit offset in both modes.
        if s.is_ascii() {
            return ReText {
                elems: Vec::new(),
                unit_of: None,
                n_elems: s.len(),
                subject_shape: SUBJECT_ASCII,
                unicode,
                ascii_src: Some(src.unwrap_or_else(|| crate::lstr::LStr::from(s))),
                parent: None,
                elem_base: 0,
                unit_base: 0,
            };
        }
        if unicode {
            let cps = crate::jstr::code_points(s);
            if cps.iter().all(|&cp| cp < 0x10000) {
                // BMP-only: one unit per element.
                let subject_shape = if cps.iter().all(|&cp| cp <= 0xff) {
                    SUBJECT_ONE_BYTE
                } else {
                    SUBJECT_TWO_BYTE
                };
                return ReText {
                    n_elems: cps.len(),
                    elems: cps,
                    unit_of: None,
                    subject_shape,
                    unicode,
                    ascii_src: None,
                    parent: None,
                    elem_base: 0,
                    unit_base: 0,
                };
            }
            let mut unit_of = Vec::with_capacity(cps.len() + 1);
            let mut u = 0usize;
            for &cp in &cps {
                unit_of.push(u);
                u += if cp >= 0x10000 { 2 } else { 1 };
            }
            unit_of.push(u);
            ReText {
                n_elems: cps.len(),
                elems: cps,
                unit_of: Some(unit_of),
                subject_shape: SUBJECT_ASTRAL,
                unicode,
                ascii_src: None,
                parent: None,
                elem_base: 0,
                unit_base: 0,
            }
        } else {
            let units = crate::jstr::units(s);
            ReText {
                n_elems: units.len(),
                elems: units.iter().map(|&u| u as u32).collect(),
                unit_of: None,
                subject_shape: if units.iter().all(|&unit| unit <= 0xff) {
                    SUBJECT_ONE_BYTE
                } else {
                    SUBJECT_TWO_BYTE
                },
                unicode,
                ascii_src: None,
                parent: None,
                elem_base: 0,
                unit_base: 0,
            }
        }
    }

    fn subject_shape(&self) -> u8 {
        self.subject_shape
    }

    /// The element index containing unit offset `u` (== len when `u` is at/past the end).
    pub fn elem_at_unit(&self, u: usize) -> usize {
        if let Some(parent) = &self.parent {
            let u = u.min(self.unit_index(self.n_elems));
            return parent.elem_at_unit(self.unit_base + u) - self.elem_base;
        }
        match &self.unit_of {
            None => u.min(self.n_elems),
            Some(unit_of) => match unit_of.binary_search(&u) {
                Ok(k) => k.min(self.n_elems),
                Err(k) => k - 1,
            },
        }
    }

    /// The unit offset of element `e`.
    pub fn unit_index(&self, e: usize) -> usize {
        if let Some(parent) = &self.parent {
            return parent.unit_index(self.elem_base + e.min(self.n_elems)) - self.unit_base;
        }
        match &self.unit_of {
            None => e.min(self.n_elems),
            Some(unit_of) => unit_of[e.min(self.n_elems)],
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.n_elems
    }

    /// [`Self::slice`] as an engine string, sharing an ASCII source's storage where `LStr` can.
    pub(crate) fn slice_lstr(&self, a: usize, b: usize) -> crate::lstr::LStr {
        match &self.ascii_src {
            Some(src) => src.slice_bytes(a, b),
            None => self.slice(a, b).into(),
        }
    }

    /// Append [`Self::slice`] to `out`, copying straight out of an ASCII source.
    pub(crate) fn push_slice(&self, out: &mut String, a: usize, b: usize) {
        match &self.ascii_src {
            Some(src) => out.push_str(&src.as_str()[a..b]),
            None => out.push_str(&self.slice(a, b)),
        }
    }

    /// The canonical string for elements `a..b` (surrogate halves recombine).
    pub fn slice(&self, a: usize, b: usize) -> String {
        // ASCII subject: element index == byte index — copy straight from the source.
        if let Some(src) = &self.ascii_src {
            return src[a..b].to_string();
        }
        let elems = &self.elements()[a..b];
        // ASCII fast path: elements are the bytes.
        if elems.iter().all(|&e| e < 0x80) {
            let bytes: Vec<u8> = elems.iter().map(|&e| e as u8).collect();
            return String::from_utf8(bytes).unwrap();
        }
        if self.unicode {
            crate::jstr::from_code_points(elems)
        } else {
            let units: Vec<u16> = elems.iter().map(|&e| e as u16).collect();
            crate::jstr::from_units(&units)
        }
    }
}

/// Count the capturing groups in a pattern (escapes and classes skipped): `(` not followed by
/// `?`, plus named groups `(?<name>`.
fn count_capture_groups(chars: &[char]) -> usize {
    let mut n = 0;
    let mut i = 0;
    let mut in_class = false;
    while i < chars.len() {
        match chars[i] {
            '\\' => i += 1,
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => {
                let plain = chars.get(i + 1) != Some(&'?');
                let named = chars.get(i + 2) == Some(&'<')
                    && !matches!(chars.get(i + 3), Some('=') | Some('!'));
                if plain || named {
                    n += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    n
}

/// Whether `pattern` contains a named capture group `(?<name>…)` (not a lookbehind `(?<=`/`(?<!`).
fn has_named_group(pattern: &str) -> bool {
    let b: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i + 2 < b.len() {
        if b[i] == '(' && b[i + 1] == '?' && b[i + 2] == '<' {
            let after = b.get(i + 3).copied();
            if after != Some('=') && after != Some('!') {
                return true;
            }
        }
        i += 1;
    }
    false
}

impl Regex {
    /// Requested payload/capacity retained by this compiled matcher. The visitor owns one
    /// identity registry per shared allocation family, so first-set aliases and nested program
    /// references are credited exactly once across every cache and Realm in the Agent.
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        let first = match &self.first {
            FirstFilter::Atoms(atoms) => atoms
                .capacity()
                .saturating_mul(std::mem::size_of::<Rep>())
                .saturating_add(
                    atoms
                        .iter()
                        .map(|rep| scan_rep_retained_memory(rep, visitor))
                        .fold(0usize, usize::saturating_add),
                ),
            _ => 0,
        };
        std::mem::size_of::<Self>()
            .saturating_add(
                self.prog
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Inst>()),
            )
            .saturating_add(scan_program_references(&self.prog, visitor))
            .saturating_add(first)
            .saturating_add(self.literal_ascii.as_ref().map_or(0, |bytes| bytes.len()))
            .saturating_add(self.literal_fold.as_ref().map_or(0, |bytes| bytes.len()))
            .saturating_add(
                self.one_byte_native
                    .borrow()
                    .as_ref()
                    .map_or(0, OneByteNativeProgram::retained_bytes),
            )
            .saturating_add(self.first_lut.as_ref().map_or(0, |_| 256))
            .saturating_add(self.source.capacity())
            .saturating_add(self.flags.capacity())
            .saturating_add(
                self.names
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(String, usize)>()),
            )
            .saturating_add(
                self.names
                    .iter()
                    .map(|(name, _)| name.capacity())
                    .sum::<usize>(),
            )
    }

    /// Conservative retained heap size of this immutable compiled matcher. Shared character
    /// classes/lookaround programs may be counted more than once; over-accounting only evicts a
    /// reconstructible cache entry sooner and avoids an expensive graph-dedup pass on insertion.
    pub(crate) fn heap_bytes(&self) -> usize {
        let first = match &self.first {
            FirstFilter::Atoms(atoms) => atoms
                .capacity()
                .saturating_mul(std::mem::size_of::<Rep>())
                .saturating_add(atoms.iter().map(rep_heap_bytes).sum::<usize>()),
            _ => 0,
        };
        std::mem::size_of::<Self>()
            .saturating_add(program_heap_bytes(&self.prog, self.prog.capacity()))
            .saturating_add(first)
            .saturating_add(self.literal_ascii.as_ref().map_or(0, |bytes| bytes.len()))
            .saturating_add(self.literal_fold.as_ref().map_or(0, |bytes| bytes.len()))
            .saturating_add(
                self.one_byte_native
                    .borrow()
                    .as_ref()
                    .map_or(0, OneByteNativeProgram::retained_bytes),
            )
            .saturating_add(self.first_lut.as_ref().map_or(0, |_| 256))
            .saturating_add(self.source.capacity())
            .saturating_add(self.flags.capacity())
            .saturating_add(
                self.names
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(String, usize)>()),
            )
            .saturating_add(
                self.names
                    .iter()
                    .map(|(name, _)| name.capacity())
                    .sum::<usize>(),
            )
    }

    pub fn new(pattern: &str, flags: &str) -> Result<Regex, String> {
        let mut seen = String::new();
        for f in flags.chars() {
            if !"dgimsuvy".contains(f) {
                return Err(format!("invalid regular expression flag {f}"));
            }
            if seen.contains(f) {
                return Err(format!("duplicate regular expression flag {f}"));
            }
            seen.push(f);
        }
        if flags.contains('u') && flags.contains('v') {
            return Err("the u and v regular expression flags are mutually exclusive".into());
        }
        let unicode = flags.contains('u') || flags.contains('v');
        let unicode_sets = flags.contains('v');
        let named_mode = unicode || has_named_group(pattern);
        let elems = pattern_elements(unicode, pattern);
        let total_groups = count_capture_groups(&elems);
        let mut p = Parser {
            chars: elems,
            pos: 0,
            total_groups,
            ngroups: 0,
            names: Vec::new(),
            unicode,
            unicode_sets,
            named_mode,
            name_refs: Vec::new(),
            depth: 0,
            look_depth: 0,
        };
        let mut ast = p.parse_alt()?;
        if p.pos != p.chars.len() {
            return Err("unexpected character in pattern".into());
        }
        // Resolve `\k<name>` references now that every group name is known.
        for name in &p.name_refs {
            if !p.names.iter().any(|(n, _)| n == name) {
                return Err(format!("invalid named back reference <{name}>"));
            }
        }
        // Duplicate group names are allowed only across distinct alternation branches.
        validate_group_names(&ast, &p.names)?;
        // In Unicode mode a decimal escape must name an existing capture group.
        if unicode {
            let mut max_ref = 0usize;
            max_backref(&ast, &mut max_ref);
            if max_ref > p.ngroups {
                return Err(format!(
                    "back reference \\{max_ref} exceeds the number of capture groups"
                ));
            }
        }
        resolve_named_backrefs(&mut ast, &p.names);
        // Wrap the whole match in group-0 saves.
        let mut prog = vec![Inst::Save(0)];
        let mut lowering = Lowering {
            marks: 0,
            unroll_budget: MAX_UNROLLED_LEN,
        };
        compile(&ast, &mut prog, &mut lowering, false)?;
        prog.push(Inst::Save(1));
        prog.push(Inst::Match);
        // The `flags` accessor returns flags in canonical order.
        let canonical: String = "dgimsuvy".chars().filter(|c| flags.contains(*c)).collect();
        let (first, first_line_start) = first_filter(&prog, flags.contains('m'));
        let first_byte = if !flags.contains('i') && !first_line_start {
            match &first {
                FirstFilter::Atoms(atoms) if atoms.len() == 1 => match &atoms[0] {
                    Rep::Char(c) if *c < 0x80 => Some(*c as u8),
                    _ => None,
                },
                _ => None,
            }
        } else {
            None
        };
        let literal_ascii = if !flags.contains('i') && p.ngroups == 0 {
            let chars = if prog.len() >= 4
                && matches!(prog.first(), Some(Inst::Save(0)))
                && matches!(prog.get(prog.len() - 2), Some(Inst::Save(1)))
                && matches!(prog.last(), Some(Inst::Match))
            {
                prog[1..prog.len() - 2]
                    .iter()
                    .map(|inst| match inst {
                        Inst::Char(c) if *c < 0x80 => Some(*c as u8),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
            } else {
                None
            }
            .filter(|literal| !literal.is_empty());
            chars.map(Vec::into_boxed_slice)
        } else {
            None
        };
        // Non-`u` case-insensitive literal: the same `Save(0), Char*, Save(1), Match` shape
        // becomes a UTF-8 fold-aware byte search (canonicalization only folds ASCII letters). A
        // Char that is not a scalar value (a lone surrogate in the source) declines the fast
        // path rather than guessing an encoding.
        let literal_fold = if flags.contains('i') && !flags.contains('u') && p.ngroups == 0 {
            let bytes = if prog.len() >= 4
                && matches!(prog.first(), Some(Inst::Save(0)))
                && matches!(prog.get(prog.len() - 2), Some(Inst::Save(1)))
                && matches!(prog.last(), Some(Inst::Match))
            {
                let mut out = Vec::new();
                let mut ok = true;
                for inst in &prog[1..prog.len() - 2] {
                    match inst {
                        Inst::Char(c) => match char::from_u32(*c) {
                            Some(ch) => {
                                out.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes())
                            }
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                ok.then_some(out)
            } else {
                None
            }
            .filter(|literal| !literal.is_empty());
            bytes.map(Vec::into_boxed_slice)
        } else {
            None
        };
        let memo_string_failures = p.ngroups == 0 && program_contains_string_set(&prog);
        let mut re = Regex {
            unicode,
            nmarks: lowering.marks,
            first,
            first_byte,
            literal_ascii,
            literal_fold,
            memo_string_failures,
            first_lut: None,
            first_line_start,
            tier_ticks: Cell::new(0),
            subject_shapes: Cell::new(0),
            tier_up_attempted: Cell::new(false),
            tier_up_at: regexp_tier_up_threshold(),
            one_byte_native: RefCell::new(None),
            prog,
            ngroups: p.ngroups,
            source: if pattern.is_empty() {
                "(?:)".into()
            } else {
                pattern.to_string()
            },
            flags: canonical,
            global: flags.contains('g'),
            ignore_case: flags.contains('i'),
            multiline: flags.contains('m'),
            dotall: flags.contains('s'),
            sticky: flags.contains('y'),
            names: p.names,
        };
        let lut = if let FirstFilter::Atoms(atoms) = &re.first {
            let mut lut = Box::new([false; 256]);
            for (c, slot) in lut.iter_mut().enumerate() {
                *slot = re.first_matches(atoms, c as u32);
            }
            Some(lut)
        } else {
            None
        };
        re.first_lut = lut;
        Ok(re)
    }

    fn note_execution(&self, text: &ReText) {
        self.subject_shapes
            .set(self.subject_shapes.get() | text.subject_shape());
        self.tier_ticks.set(self.tier_ticks.get().saturating_add(1));
        if self.tier_up_attempted.get()
            || self.tier_up_at == 0
            || self.tier_ticks.get() < self.tier_up_at
            || self.subject_shapes.get() & (SUBJECT_ASCII | SUBJECT_ONE_BYTE) == 0
        {
            return;
        }
        self.tier_up_attempted.set(true);
        // The native subset currently uses legacy exact byte membership. Case folding, dotAll,
        // and Unicode/UnicodeSets mode change those predicates, so those patterns remain on the
        // general matcher until their semantics have dedicated native operations.
        if self.ignore_case || self.dotall || self.unicode {
            regexp_prof_tier_up(false);
            return;
        }
        if let Some(program) = OneByteNativeProgram::compile(&self.prog, self.multiline) {
            *self.one_byte_native.borrow_mut() = Some(program);
            regexp_prof_tier_up(true);
        } else {
            regexp_prof_tier_up(false);
        }
    }

    #[cfg(test)]
    fn tier_feedback(&self) -> (u32, u8, bool, bool) {
        (
            self.tier_ticks.get(),
            self.subject_shapes.get(),
            self.tier_up_attempted.get(),
            self.one_byte_native.borrow().is_some(),
        )
    }

    /// Whether a match could begin with element `c` — the [`FirstFilter::Atoms`] predicate.
    /// Deliberately a superset of what the matcher accepts (e.g. `Any` only excludes `\n`), so a
    /// pass costs a wasted attempt at worst; only a reject skips work.
    fn first_matches(&self, atoms: &[Rep], c: u32) -> bool {
        let unicode = self.unicode || self.flags.contains('v');
        atoms.iter().any(|rep| match rep {
            Rep::Char(ch) => {
                *ch == c
                    || (self.ignore_case && {
                        match (char::from_u32(c), char::from_u32(*ch)) {
                            (Some(x), Some(y)) => {
                                if unicode {
                                    fold_canon(x as u32) == fold_canon(y as u32)
                                } else {
                                    canonicalize_legacy(x) == canonicalize_legacy(y)
                                }
                            }
                            _ => false,
                        }
                    })
            }
            Rep::Any => self.dotall || c != '\n' as u32,
            Rep::Class(cc) => cc.matches(c, self.ignore_case, unicode),
        })
    }

    /// Match a prepared subject and return shared capture spans.
    #[cfg(test)]
    pub fn exec_text_shared(
        &self,
        text: &ReText,
        start: usize,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<Captures> {
        poll_interrupt(control)?;
        self.exec_text_shared_entry_polled(text, start, control)
    }

    /// Match after the surrounding execution entry has already force-polled host interruption.
    /// Long scans and backtracking still poll internally; browser/native call sites use this to
    /// avoid a duplicate atomic read for every tiny match.
    pub(crate) fn exec_text_shared_entry_polled(
        &self,
        text: &ReText,
        start: usize,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<Captures> {
        self.note_execution(text);
        match &text.ascii_src {
            Some(s) => {
                if let Some(literal) = &self.literal_ascii {
                    Ok(
                        find_ascii_literal(s.as_bytes(), start, literal, self.sticky, control)?
                            .map(Captures::one),
                    )
                } else if let Some(literal) = &self.literal_fold {
                    Ok(
                        find_ascii_fold_literal(
                            s.as_bytes(),
                            start,
                            literal,
                            self.sticky,
                            control,
                        )?
                        .map(Captures::one),
                    )
                } else if let Some(native) = self.one_byte_native.borrow().as_ref() {
                    regexp_prof_native_exec();
                    Ok(native
                        .find_ascii(s.as_bytes(), start, self.sticky, control)?
                        .map(Captures::one))
                } else {
                    self.exec_impl(s.as_bytes(), start, control)
                }
            }
            None if text.subject_shape() == SUBJECT_ONE_BYTE => {
                if let Some(native) = self.one_byte_native.borrow().as_ref() {
                    regexp_prof_native_exec();
                    Ok(native
                        .find_one_byte(text.elements(), start, self.sticky, control)?
                        .map(Captures::one))
                } else {
                    self.exec_impl(text.elements(), start, control)
                }
            }
            None => self.exec_impl(text.elements(), start, control),
        }
    }

    pub(crate) fn exec_text_discard_shared_entry_polled(
        &self,
        text: &ReText,
        start: usize,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<Captures> {
        self.exec_text_shared_entry_polled(text, start, control)
    }

    /// Whole-match-only search for operations whose JavaScript result is dead. Capture groups
    /// can be recovered lazily if a legacy RegExp static is subsequently observed.
    pub(crate) fn find_text_shared_entry_polled(
        &self,
        text: &ReText,
        start: usize,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        self.note_execution(text);
        match &text.ascii_src {
            Some(s) => {
                if let Some(literal) = &self.literal_ascii {
                    find_ascii_literal(s.as_bytes(), start, literal, self.sticky, control)
                } else if let Some(literal) = &self.literal_fold {
                    find_ascii_fold_literal(s.as_bytes(), start, literal, self.sticky, control)
                } else if let Some(native) = self.one_byte_native.borrow().as_ref() {
                    regexp_prof_native_exec();
                    native.find_ascii(s.as_bytes(), start, self.sticky, control)
                } else {
                    self.find_impl(s.as_bytes(), start, control)
                }
            }
            None if text.subject_shape() == SUBJECT_ONE_BYTE => {
                if let Some(native) = self.one_byte_native.borrow().as_ref() {
                    regexp_prof_native_exec();
                    native.find_one_byte(text.elements(), start, self.sticky, control)
                } else {
                    self.find_impl(text.elements(), start, control)
                }
            }
            None => self.find_impl(text.elements(), start, control),
        }
    }

    fn exec_impl<I: ReInput>(
        &self,
        input: I,
        start: usize,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<Captures> {
        let groups = self.ngroups;
        self.exec_impl_with(input, start, control, move |slots| {
            Captures::from_slots(slots, groups)
        })
    }

    /// Execute the matcher while retaining capture slots for backreferences and assertions, but
    /// project only the group-0 span. RegExpBuiltinExec still creates the public match array when
    /// required; discarded native/JIT calls use this projection so they do not allocate a second
    /// capture-result container (ECMA-262 §22.2.7.1–2).
    fn find_impl<I: ReInput>(
        &self,
        input: I,
        start: usize,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<(usize, usize)> {
        self.exec_impl_with(input, start, control, |slots| {
            let start = slots[0].expect("successful regexp match has a start");
            let end = slots[1].expect("successful regexp match has an end");
            (start.min(end), start.max(end))
        })
    }

    fn exec_impl_with<I: ReInput, T, F: Fn(&[Option<usize>]) -> T>(
        &self,
        input: I,
        start: usize,
        control: &crate::RuntimeInterrupt,
        project: F,
    ) -> MatchResult<T> {
        if start > input.len() {
            return Ok(None);
        }
        // One matcher for the whole scan, its working buffers recycled across `exec` calls via a
        // thread-local (the engine is single-threaded per Interp).
        let mut scratch = MATCH_SCRATCH
            .with(|s| s.borrow_mut().take())
            .unwrap_or_default();
        scratch.caps.clear();
        scratch.caps.resize(2 * (self.ngroups + 1), None);
        scratch.marks.clear();
        scratch.marks.resize(self.nmarks, None);
        scratch.flags.clear();
        scratch
            .flags
            .push((self.ignore_case, self.multiline, self.dotall));
        let mut m = Matcher {
            input,
            caps: scratch.caps,
            marks: scratch.marks,
            steps: 0,
            bt: scratch.bt,
            saved: scratch.saved,
            choices: 0,
            floor: 0,
            looks: Vec::new(),
            state_limit: backtrack_state_limit(),
            back: false,
            flags: scratch.flags,
            unicode: self.flags.contains('u') || self.flags.contains('v'),
            control,
            abort: None,
            // With no observable capture groups, failure at a StringSet instruction depends only
            // on program position, subject position, direction, and active flags. Memoizing those
            // failures turns ambiguous emoji-sequence repetition into dynamic programming while
            // preserving the specification's alternative order.
            string_failures: self.memo_string_failures.then(Default::default),
            prof: regexp_prof_enabled(),
        };
        let mut from = start;
        let prof = m.prof;
        let result = 'scan: loop {
            if from > input.len() {
                break 'scan Ok(None);
            }
            if prof {
                REGEXP_PROF.with(|p| p.borrow_mut().attempts += 1);
            }
            // Prescan: skip positions that cannot begin a match. Sticky regexes get exactly one
            // attempt at `start`, so the filter only ever saves that single attempt for them.
            if !self.sticky {
                if let Some(byte) = self.first_byte {
                    let Some(found) = input.find_byte(from, byte, control)? else {
                        break 'scan Ok(None);
                    };
                    from = found;
                } else {
                    match &self.first {
                        FirstFilter::Anchored => {
                            // `^` (non-multiline) can only match at position 0: one attempt at
                            // `from` decides the scan (any later position fails the assert too).
                            if from > 0 {
                                break 'scan Ok(None);
                            }
                        }
                        FirstFilter::Atoms(atoms) => {
                            // Every path consumes an element first, or begins where `^`
                            // holds: find the next viable position. Small elements go
                            // through the precomputed table (one load per position).
                            let len = input.len();
                            let line_start = |from: usize| {
                                self.first_line_start
                                    && (from == 0
                                        || (self.multiline
                                            && is_line_terminator_u32(input.at(from - 1))))
                            };
                            loop {
                                if from >= len {
                                    // A path through `^` may match the empty end of the input.
                                    if from == len && line_start(from) {
                                        break;
                                    }
                                    break 'scan Ok(None);
                                }
                                let c = input.at(from);
                                let viable = match &self.first_lut {
                                    Some(lut) if (c as usize) < 256 => lut[c as usize],
                                    _ => self.first_matches(atoms, c),
                                };
                                if viable || line_start(from) {
                                    break;
                                }
                                from += 1;
                                if prof {
                                    REGEXP_PROF.with(|p| p.borrow_mut().scan_positions += 1);
                                }
                                if from & INTERRUPT_POLL_MASK == 0 {
                                    poll_interrupt(control)?;
                                }
                            }
                        }
                        FirstFilter::None => {}
                    }
                }
            }
            // A failed attempt unwinds its backtrack stack, lookarounds and direction. Captures,
            // marks and flags changed without undo records (see `Matcher::floor`) are reset here.
            debug_assert!(m.bt.is_empty() && m.saved.is_empty() && m.looks.is_empty());
            debug_assert!(m.choices == 0 && m.floor == 0 && !m.back);
            m.caps.fill(None);
            m.marks.fill(None);
            m.flags.truncate(1);
            if m.run(&self.prog, from) {
                break 'scan Ok(Some(project(&m.caps)));
            }
            if let Some(error) = m.abort {
                break 'scan Err(error);
            }
            if self.sticky {
                break 'scan Ok(None);
            }
            from += 1;
        };
        // A match leaves its backtrack state behind; a huge one's storage is released.
        m.bt.clear();
        m.bt.shrink_to(RETAINED_BACKTRACK_STATE);
        m.saved.clear();
        m.saved.shrink_to(RETAINED_BACKTRACK_STATE);
        MATCH_SCRATCH.with(|s| {
            *s.borrow_mut() = Some(MatchScratch {
                caps: m.caps,
                marks: m.marks,
                flags: m.flags,
                bt: m.bt,
                saved: m.saved,
            });
        });
        result
    }
}

/// One-line matcher profiling report (`TRUST_JS_REGEXP_PROF=1`), mirroring the JIT opstat style:
/// per-instruction dispatch counts, scan positions, candidate attempts, and backtracking entries.
/// `None` (and zero work) when the gate is off or nothing matched.
pub(crate) fn regexp_prof_report() -> Option<String> {
    if !regexp_prof_enabled() {
        return None;
    }
    REGEXP_PROF.with(|cell| {
        let prof = cell.borrow();
        if prof.inst.iter().all(|c| *c == 0)
            && prof.scan_positions == 0
            && prof.attempts == 0
            && prof.backtrack_entries == 0
            && prof.tier_up_attempts == 0
            && prof.tier_up_successes == 0
            && prof.native_execs == 0
            && prof.native_code_bytes == 0
        {
            return None;
        }
        let mut body = String::new();
        for (name, count) in REGEXP_PROF_INST_NAMES.iter().zip(prof.inst.iter()) {
            if *count > 0 {
                body.push_str(&format!(" {name}:{count}"));
            }
        }
        Some(format!(
            "matcher inst[{}] scan:{} attempts:{} backtrack:{} tier_attempts:{} tier_successes:{} native_execs:{} native_code_bytes:{}",
            body.trim(),
            prof.scan_positions,
            prof.attempts,
            prof.backtrack_entries,
            prof.tier_up_attempts,
            prof.tier_up_successes,
            prof.native_execs,
            prof.native_code_bytes
        ))
    })
}

fn scan_program_references(program: &[Inst], visitor: &mut crate::memory::Visitor) -> usize {
    program
        .iter()
        .map(|instruction| scan_inst_retained_memory(instruction, visitor))
        .fold(0usize, usize::saturating_add)
}

fn scan_inst_retained_memory(instruction: &Inst, visitor: &mut crate::memory::Visitor) -> usize {
    match instruction {
        Inst::Class(class) => scan_char_class_retained_memory(class, visitor),
        Inst::StringSet(set) | Inst::StringSetRepeat { set, .. } => {
            scan_string_set_retained_memory(set, visitor)
        }
        Inst::BackrefAlt(groups) => {
            let identity = Rc::as_ptr(groups) as usize;
            if visitor.regexp_backref_groups_allocation(identity) {
                std::mem::size_of::<Vec<usize>>().saturating_add(
                    groups
                        .capacity()
                        .saturating_mul(std::mem::size_of::<usize>()),
                )
            } else {
                0
            }
        }
        Inst::Look { prog, .. } | Inst::LookBehind { prog, .. } => {
            let identity = Rc::as_ptr(prog) as usize;
            if visitor.regexp_program_allocation(identity) {
                std::mem::size_of::<Vec<Inst>>()
                    .saturating_add(prog.capacity().saturating_mul(std::mem::size_of::<Inst>()))
                    .saturating_add(scan_program_references(prog, visitor))
            } else {
                0
            }
        }
        Inst::Many { rep, .. } => scan_rep_retained_memory(rep, visitor),
        _ => 0,
    }
}

fn scan_rep_retained_memory(rep: &Rep, visitor: &mut crate::memory::Visitor) -> usize {
    match rep {
        Rep::Class(class) => scan_char_class_retained_memory(class, visitor),
        Rep::Char(_) | Rep::Any => 0,
    }
}

fn scan_char_class_retained_memory(
    class: &Rc<CharClass>,
    visitor: &mut crate::memory::Visitor,
) -> usize {
    let identity = Rc::as_ptr(class) as usize;
    if visitor.regexp_char_class_allocation(identity) {
        char_class_requested_bytes(class)
    } else {
        0
    }
}

fn scan_string_set_retained_memory(
    set: &Rc<StringSet>,
    visitor: &mut crate::memory::Visitor,
) -> usize {
    let identity = Rc::as_ptr(set) as usize;
    if !visitor.regexp_string_set_allocation(identity) {
        return 0;
    }
    let trie_requested_bytes = |trie: &StringTrie| {
        trie.nodes
            .capacity()
            .saturating_mul(std::mem::size_of::<StringTrieNode>())
            .saturating_add(trie.nodes.iter().fold(0usize, |bytes, node| {
                bytes.saturating_add(
                    node.edges
                        .capacity()
                        .saturating_mul(std::mem::size_of::<(u32, usize)>()),
                )
            }))
    };
    std::mem::size_of::<StringSet>()
        .saturating_add(trie_requested_bytes(&set.forward))
        .saturating_add(trie_requested_bytes(&set.backward))
        .saturating_add(char_class_requested_capacity(&set.singles))
        .saturating_add(char_class_requested_capacity(&set.first))
}

fn char_class_requested_bytes(class: &CharClass) -> usize {
    std::mem::size_of::<CharClass>().saturating_add(char_class_requested_capacity(class))
}

fn char_class_requested_capacity(class: &CharClass) -> usize {
    class
        .ranges
        .capacity()
        .saturating_mul(std::mem::size_of::<(u32, u32)>())
        .saturating_add(
            class
                .builtins
                .capacity()
                .saturating_mul(std::mem::size_of::<char>()),
        )
        .saturating_add(
            class
                .props
                .capacity()
                .saturating_mul(std::mem::size_of::<(bool, &'static [(u32, u32)])>()),
        )
        .saturating_add(class.ascii_lut.as_ref().map_or(0, |_| 256))
}

fn program_heap_bytes(program: &[Inst], capacity: usize) -> usize {
    capacity
        .saturating_mul(std::mem::size_of::<Inst>())
        .saturating_add(program.iter().map(inst_heap_bytes).sum::<usize>())
}

fn program_contains_string_set(program: &[Inst]) -> bool {
    program.iter().any(|instruction| match instruction {
        Inst::StringSet(_) | Inst::StringSetRepeat { .. } => true,
        Inst::Look { prog, .. } | Inst::LookBehind { prog, .. } => {
            program_contains_string_set(prog)
        }
        _ => false,
    })
}

fn inst_heap_bytes(inst: &Inst) -> usize {
    match inst {
        Inst::Class(class) => char_class_heap_bytes(class),
        Inst::StringSet(set) => string_set_heap_bytes(set),
        Inst::StringSetRepeat { set, .. } => string_set_heap_bytes(set),
        Inst::BackrefAlt(groups) => std::mem::size_of::<Vec<usize>>().saturating_add(
            groups
                .capacity()
                .saturating_mul(std::mem::size_of::<usize>()),
        ),
        Inst::Look { prog, .. } | Inst::LookBehind { prog, .. } => std::mem::size_of::<Vec<Inst>>()
            .saturating_add(program_heap_bytes(prog, prog.capacity())),
        Inst::Many { rep, .. } => rep_heap_bytes(rep),
        _ => 0,
    }
}

fn string_set_heap_bytes(set: &StringSet) -> usize {
    let trie_bytes = |trie: &StringTrie| {
        trie.nodes
            .capacity()
            .saturating_mul(std::mem::size_of::<StringTrieNode>())
            .saturating_add(trie.nodes.iter().fold(0usize, |bytes, node| {
                bytes.saturating_add(
                    node.edges
                        .capacity()
                        .saturating_mul(std::mem::size_of::<(u32, usize)>()),
                )
            }))
    };
    std::mem::size_of::<StringSet>()
        .saturating_add(trie_bytes(&set.forward))
        .saturating_add(trie_bytes(&set.backward))
        .saturating_add(char_class_heap_bytes(&set.singles))
        .saturating_add(char_class_heap_bytes(&set.first))
}

fn rep_heap_bytes(rep: &Rep) -> usize {
    match rep {
        Rep::Class(class) => char_class_heap_bytes(class),
        _ => 0,
    }
}

fn char_class_heap_bytes(class: &CharClass) -> usize {
    std::mem::size_of::<CharClass>()
        .saturating_add(
            class
                .ranges
                .capacity()
                .saturating_mul(std::mem::size_of::<(u32, u32)>()),
        )
        .saturating_add(
            class
                .builtins
                .capacity()
                .saturating_mul(std::mem::size_of::<char>()),
        )
        .saturating_add(
            class
                .props
                .capacity()
                .saturating_mul(std::mem::size_of::<(bool, &'static [(u32, u32)])>()),
        )
        .saturating_add(class.ascii_lut.as_ref().map_or(0, |_| 256))
}

/// Recycled matcher working buffers (see `Regex::exec_at`).
#[derive(Default)]
struct MatchScratch {
    caps: Vec<Option<usize>>,
    marks: Vec<Option<usize>>,
    flags: Vec<(bool, bool, bool)>,
    bt: Vec<Backtrack>,
    saved: Vec<Option<usize>>,
}

thread_local! {
    static MATCH_SCRATCH: std::cell::RefCell<Option<MatchScratch>> =
        const { std::cell::RefCell::new(None) };
}

// ---------------------------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------------------------

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// Parse a group or nested class body one nesting level deeper.
    fn nested<T>(
        &mut self,
        parse: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.depth == MAX_PATTERN_NESTING {
            return Err("regular expression is nested too deeply".into());
        }
        self.depth += 1;
        let result = with_pattern_stack(|| parse(self));
        self.depth -= 1;
        result
    }

    /// Parse a lookaround body (see [`MAX_LOOKAROUND_NESTING`]).
    fn parse_lookaround_body(&mut self) -> Result<Node, String> {
        if self.look_depth == MAX_LOOKAROUND_NESTING {
            return Err("regular expression is nested too deeply".into());
        }
        self.look_depth += 1;
        let body = self.parse_alt();
        self.look_depth -= 1;
        body
    }

    fn parse_alt(&mut self) -> Result<Node, String> {
        let mut branches = vec![self.parse_concat()?];
        while self.peek() == Some('|') {
            self.bump();
            branches.push(self.parse_concat()?);
        }
        if branches.len() == 1 {
            Ok(branches.pop().unwrap())
        } else {
            Ok(Node::Alt(branches))
        }
    }

    fn parse_concat(&mut self) -> Result<Node, String> {
        let mut seq = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            seq.push(self.parse_quantified()?);
        }
        match seq.len() {
            0 => Ok(Node::Empty),
            1 => Ok(seq.pop().unwrap()),
            _ => Ok(Node::Concat(seq)),
        }
    }

    fn parse_quantified(&mut self) -> Result<Node, String> {
        // A quantifier at the start of a term (after `(`, `|`, or `^`) has nothing to repeat.
        if matches!(self.peek(), Some('*' | '+' | '?')) {
            return Err("nothing to repeat".into());
        }
        // A *braced* quantifier at term start too (`/{2}/`); a non-quantifier `{` stays a
        // literal (Annex B) and is handled by parse_atom.
        if self.peek() == Some('{') && self.try_parse_brace()?.is_some() {
            return Err("nothing to repeat".into());
        }
        let atom = self.parse_atom()?;
        let (min, max) = match self.peek() {
            Some('*') => {
                self.bump();
                (0, None)
            }
            Some('+') => {
                self.bump();
                (1, None)
            }
            Some('?') => {
                self.bump();
                (0, Some(1))
            }
            Some('{') => match self.try_parse_brace()? {
                Some(mm) => mm,
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        // A lookbehind can never be quantified; a lookahead only outside Unicode mode
        // (the Annex B QuantifiableAssertion carve-out).
        if matches!(atom, Node::LookBehind(..)) || (self.unicode && matches!(atom, Node::Look(..)))
        {
            return Err("quantifier on an assertion".into());
        }
        let greedy = if self.peek() == Some('?') {
            self.bump();
            false
        } else {
            true
        };
        // A quantifier cannot itself be quantified (`a**`, `a+?` is lazy and already consumed).
        if matches!(self.peek(), Some('*' | '+' | '?')) {
            return Err("nothing to repeat".into());
        }
        Ok(Node::Repeat(Box::new(atom), min, max, greedy))
    }

    /// `{n}` / `{n,}` / `{n,m}`. Returns `None` (and leaves position) if it is not a valid quantifier
    /// (a literal `{`).
    fn try_parse_brace(&mut self) -> Result<Option<(usize, Option<usize>)>, String> {
        let save = self.pos;
        self.bump(); // {
        let mut digits = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                digits.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if digits.is_empty() {
            self.pos = save;
            return Ok(None);
        }
        let min: usize = digits.parse().unwrap_or(0);
        let max = if self.peek() == Some(',') {
            self.bump();
            let mut d2 = String::new();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    d2.push(c);
                    self.bump();
                } else {
                    break;
                }
            }
            if d2.is_empty() {
                None
            } else {
                Some(d2.parse().unwrap_or(min))
            }
        } else {
            Some(min)
        };
        if self.peek() != Some('}') {
            self.pos = save;
            return Ok(None);
        }
        self.bump(); // }
        if let Some(mx) = max {
            if min > mx {
                return Err("numbers out of order in {} quantifier".into());
            }
        }
        Ok(Some((min, max)))
    }

    fn parse_atom(&mut self) -> Result<Node, String> {
        match self.bump() {
            None => Ok(Node::Empty),
            Some('.') => Ok(Node::Any),
            Some('^') => Ok(Node::Start),
            Some('$') => Ok(Node::End),
            Some('(') => self.nested(Self::parse_group),
            Some('[') => self.parse_class(),
            Some('\\') => self.parse_escape(),
            // In Unicode mode a PatternCharacter excludes the remaining SyntaxCharacters.
            Some(c @ ('{' | '}' | ']')) if self.unicode => {
                Err(format!("lone '{c}' is not valid in a unicode pattern"))
            }
            Some(c) => Ok(Node::Char(cp_of_elem(c))),
        }
    }

    fn parse_group(&mut self) -> Result<Node, String> {
        // Detect (?:...), (?=...), (?!...), (?<name>...), and lookbehind (?<= / (?<! .
        if self.peek() == Some('?') {
            self.bump();
            match self.peek() {
                Some(':') => {
                    self.bump();
                    let inner = self.parse_alt()?;
                    self.expect(')')?;
                    Ok(Node::Group(None, Box::new(inner)))
                }
                Some('=') => {
                    self.bump();
                    let inner = self.parse_lookaround_body()?;
                    self.expect(')')?;
                    Ok(Node::Look(false, Box::new(inner)))
                }
                Some('!') => {
                    self.bump();
                    let inner = self.parse_lookaround_body()?;
                    self.expect(')')?;
                    Ok(Node::Look(true, Box::new(inner)))
                }
                Some('<') => {
                    self.bump();
                    // Lookbehind (?<= / (?<! (a backward-matching assertion), else a named
                    // capturing group (?<name>...).
                    match self.peek() {
                        Some(c @ ('=' | '!')) => {
                            self.bump();
                            let inner = self.parse_lookaround_body()?;
                            self.expect(')')?;
                            Ok(Node::LookBehind(c == '!', Box::new(inner)))
                        }
                        _ => {
                            let name = self.parse_group_name()?;
                            self.ngroups += 1;
                            let idx = self.ngroups;
                            // Duplicate names are allowed (ES2025) — they're distinct capture groups
                            // in different alternatives; the `groups` object reports whichever matched.
                            self.names.push((name, idx));
                            let inner = self.parse_alt()?;
                            self.expect(')')?;
                            Ok(Node::Group(Some(idx), Box::new(inner)))
                        }
                    }
                }
                Some('i' | 'm' | 's' | '-') => self.parse_modifier_group(),
                _ => Err("unsupported group".into()),
            }
        } else {
            self.ngroups += 1;
            let idx = self.ngroups;
            let inner = self.parse_alt()?;
            self.expect(')')?;
            Ok(Node::Group(Some(idx), Box::new(inner)))
        }
    }

    /// Parse `(?ims-ims:body)` after the `(?`. Flags before `-` are added, after `-` removed.
    fn parse_modifier_group(&mut self) -> Result<Node, String> {
        let mut add = (false, false, false);
        let mut remove = (false, false, false);
        let mut neg = false;
        let mut seen_any = false;
        loop {
            match self.peek() {
                Some('-') if !neg => {
                    self.bump();
                    neg = true;
                }
                Some(c @ ('i' | 'm' | 's')) => {
                    self.bump();
                    seen_any = true;
                    let slot = if neg { &mut remove } else { &mut add };
                    let f = match c {
                        'i' => &mut slot.0,
                        'm' => &mut slot.1,
                        _ => &mut slot.2,
                    };
                    if *f {
                        return Err("duplicate inline modifier flag".into());
                    }
                    *f = true;
                }
                Some(':') => break,
                _ => return Err("invalid inline modifier".into()),
            }
        }
        self.bump(); // ':'
        let _ = seen_any;
        // Only a wholly-empty modifier list (`(?:` is handled elsewhere; `(?-:` reaches here) is
        // invalid — `(?s-:…)` (add some, remove none) is fine.
        if add == (false, false, false) && remove == (false, false, false) {
            return Err("empty inline modifier".into());
        }
        // A flag may not be both added and removed.
        if (add.0 && remove.0) || (add.1 && remove.1) || (add.2 && remove.2) {
            return Err("inline modifier flag added and removed".into());
        }
        let inner = self.parse_alt()?;
        self.expect(')')?;
        Ok(Node::Modifier {
            add,
            remove,
            inner: Box::new(inner),
        })
    }

    /// `v`-mode `[...]`: parse a ClassSetExpression, computing the concrete set, and compile it
    /// to a match node (an alternation of its strings — longest first — plus a range class).
    fn parse_class_set(&mut self) -> Result<Node, String> {
        let negate = if self.peek() == Some('^') {
            self.bump();
            true
        } else {
            false
        };
        let mut set = self.parse_class_set_expression()?;
        self.expect(']')?;
        if negate {
            set = set.complement()?;
        }
        Ok(class_set_to_node(set))
    }

    fn parse_class_set_expression(&mut self) -> Result<ClassSet, String> {
        // Empty class.
        if self.peek() == Some(']') {
            return Ok(ClassSet::default());
        }
        let first = self.parse_class_set_operand()?;
        // Decide the expression kind from the following operator.
        if self.peek() == Some('&') && self.chars.get(self.pos + 1) == Some(&'&') {
            let mut acc = first;
            while self.peek() == Some('&') && self.chars.get(self.pos + 1) == Some(&'&') {
                self.bump();
                self.bump();
                if self.peek() == Some('&') {
                    return Err("unexpected '&&&' in class set".into());
                }
                let rhs = self.parse_class_set_operand()?;
                acc = acc.intersect(rhs);
            }
            return Ok(acc);
        }
        if self.peek() == Some('-') && self.chars.get(self.pos + 1) == Some(&'-') {
            let mut acc = first;
            while self.peek() == Some('-') && self.chars.get(self.pos + 1) == Some(&'-') {
                self.bump();
                self.bump();
                let rhs = self.parse_class_set_operand()?;
                acc = acc.subtract(rhs);
            }
            return Ok(acc);
        }
        // Union (with a-z ranges).
        let mut acc = self.maybe_class_set_range(first)?;
        while self.peek() != Some(']') && self.peek().is_some() {
            if self.peek() == Some('&') && self.chars.get(self.pos + 1) == Some(&'&') {
                return Err("cannot mix '&&' with a union in a class set".into());
            }
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) == Some(&'-') {
                return Err("cannot mix '--' with a union in a class set".into());
            }
            let next = self.parse_class_set_operand()?;
            let next = self.maybe_class_set_range(next)?;
            acc = acc.union(next);
        }
        Ok(acc)
    }

    /// After a single-character operand, `-x` extends it to a range.
    fn maybe_class_set_range(&mut self, operand: ClassSet) -> Result<ClassSet, String> {
        let single = operand.strings.is_empty()
            && operand.ranges.len() == 1
            && operand.ranges[0].0 == operand.ranges[0].1;
        if single
            && self.peek() == Some('-')
            && self.chars.get(self.pos + 1) != Some(&'-')
            && self.chars.get(self.pos + 1) != Some(&']')
        {
            self.bump(); // '-'
            let hi = self.parse_class_set_operand()?;
            let hi_single =
                hi.strings.is_empty() && hi.ranges.len() == 1 && hi.ranges[0].0 == hi.ranges[0].1;
            if !hi_single {
                return Err("invalid character class range".into());
            }
            let (a, b) = (operand.ranges[0].0, hi.ranges[0].0);
            if a > b {
                return Err("range out of order in character class".into());
            }
            return Ok(ClassSet {
                ranges: vec![(a, b)],
                strings: Vec::new(),
            });
        }
        Ok(operand)
    }

    fn parse_class_set_operand(&mut self) -> Result<ClassSet, String> {
        match self.peek() {
            None => Err("unterminated character class".into()),
            Some('[') => {
                self.bump();
                let negate = if self.peek() == Some('^') {
                    self.bump();
                    true
                } else {
                    false
                };
                let mut set = self.nested(Self::parse_class_set_expression)?;
                self.expect(']')?;
                if negate {
                    set = set.complement()?;
                }
                Ok(set)
            }
            Some('\\') => {
                self.bump();
                match self.peek() {
                    Some('q') => {
                        self.bump();
                        if self.bump() != Some('{') {
                            return Err("expected '{' after \\q".into());
                        }
                        let mut set = ClassSet::default();
                        let mut cur: Vec<char> = Vec::new();
                        loop {
                            match self.peek() {
                                None => return Err("unterminated \\q{...}".into()),
                                Some('}') => {
                                    self.bump();
                                    push_q_alternative(&mut set, std::mem::take(&mut cur));
                                    break;
                                }
                                Some('|') => {
                                    self.bump();
                                    push_q_alternative(&mut set, std::mem::take(&mut cur));
                                }
                                Some('\\') => {
                                    self.bump();
                                    let v = self.class_set_escape_char()?;
                                    cur.push(char::from_u32(v).unwrap_or('\u{FFFD}'));
                                }
                                Some(c) => {
                                    self.bump();
                                    cur.push(c);
                                }
                            }
                        }
                        set.normalize();
                        Ok(set)
                    }
                    Some(b @ ('d' | 'D' | 'w' | 'W' | 's' | 'S')) => {
                        self.bump();
                        Ok(builtin_class_set(b))
                    }
                    Some(pc @ ('p' | 'P')) => {
                        self.bump();
                        self.parse_class_set_property(pc == 'P')
                    }
                    _ => Ok(ClassSet::from_cp(self.class_set_escape_char()?)),
                }
            }
            // ClassSetSyntaxCharacters may not appear literally.
            Some(c @ ('(' | ')' | '{' | '}' | '/' | '|' | '-')) => {
                Err(format!("'{c}' must be escaped in a v-mode class"))
            }
            Some(c) => {
                // Doubled punctuators are reserved.
                if "&!#$%*+,.:;<=>?@^`~\"'".contains(c) && self.chars.get(self.pos + 1) == Some(&c)
                {
                    return Err(format!("reserved doubled punctuator '{c}{c}' in class set"));
                }
                self.bump();
                Ok(ClassSet::from_cp(cp_of_elem(c)))
            }
        }
    }

    /// A single-character escape inside a v-mode class (`\n`, `\u{...}`, `\-`, identity escapes).
    fn class_set_escape_char(&mut self) -> Result<u32, String> {
        match self.bump() {
            None => Err("trailing backslash in class".into()),
            Some('n') => Ok('\n' as u32),
            Some('t') => Ok('\t' as u32),
            Some('r') => Ok('\r' as u32),
            Some('f') => Ok(0x0C),
            Some('v') => Ok(0x0B),
            Some('b') => Ok(0x08),
            Some('0') => Ok(0),
            Some('x') => self.hex_strict(2),
            Some('u') => self.unicode_escape_strict(),
            Some('c') => match self.peek() {
                Some(l) if l.is_ascii_alphabetic() => {
                    self.bump();
                    Ok((l as u8 % 32) as u32)
                }
                _ => Err("invalid \\c escape in class set".into()),
            },
            Some(c) if is_regex_syntax_char(c) || "/-&!#%,:;<=>@`~\"'".contains(c) => Ok(c as u32),
            Some(c) => Err(format!("invalid identity escape \\{c} in v-mode class")),
        }
    }

    fn parse_class_set_property(&mut self, negate: bool) -> Result<ClassSet, String> {
        if self.bump() != Some('{') {
            return Err("invalid property escape: expected '{'".into());
        }
        let mut body = String::new();
        loop {
            match self.bump() {
                Some('}') => break,
                Some(c) if c.is_ascii_alphanumeric() || c == '_' || c == '=' => body.push(c),
                Some(_) => return Err("invalid character in property escape".into()),
                None => return Err("unterminated property escape".into()),
            }
        }
        let (name, value) = match body.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (body.as_str(), None),
        };
        if value.is_none() {
            if let Some(set) = property_of_strings(name) {
                if negate {
                    return Err("\\P of a property of strings is invalid".into());
                }
                return Ok(set);
            }
        }
        match crate::unicode_props::lookup_strict(name, value) {
            Some((complement, ranges)) => {
                let set = ClassSet {
                    ranges: ranges.to_vec(),
                    strings: Vec::new(),
                };
                if negate != complement {
                    set.complement()
                } else {
                    Ok(set)
                }
            }
            None => Err(format!("invalid unicode property {body}")),
        }
    }

    fn parse_class(&mut self) -> Result<Node, String> {
        if self.unicode_sets {
            return self.nested(Self::parse_class_set);
        }
        let mut cc = CharClass::default();
        if self.peek() == Some('^') {
            self.bump();
            cc.negate = true;
        }
        // `]` always closes — `[]` is the empty class (matches nothing), `[^]` matches anything.
        loop {
            match self.peek() {
                None => return Err("unterminated character class".into()),
                Some(']') => {
                    self.bump();
                    break;
                }
                _ => {}
            }
            let lo = self.class_atom()?;
            // Range a-z (but `-` at end or before `]` is literal).
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.bump();
                let hi = self.class_atom()?;
                match (lo, hi) {
                    (ClassAtom::Char(a), ClassAtom::Char(b)) => {
                        if a > b {
                            return Err("range out of order in character class".into());
                        }
                        cc.ranges.push((a, b));
                    }
                    (a, b) => {
                        // In Unicode mode a class escape (`\d`, `\p{…}`) can't be a range bound.
                        if self.unicode {
                            return Err("invalid character class range".into());
                        }
                        push_class_atom(&mut cc, a);
                        cc.ranges.push(('-' as u32, '-' as u32));
                        push_class_atom(&mut cc, b);
                    }
                }
            } else {
                push_class_atom(&mut cc, lo);
            }
        }
        Ok(Node::Class(cc))
    }

    fn class_atom(&mut self) -> Result<ClassAtom, String> {
        match self.bump() {
            None => Err("unterminated character class".into()),
            Some('\\') => match self.bump() {
                None => Err("bad escape in class".into()),
                Some(c @ ('d' | 'D' | 'w' | 'W' | 's' | 'S')) => Ok(ClassAtom::Builtin(c)),
                Some(c @ ('p' | 'P')) if self.unicode => {
                    let prop = self.parse_prop_escape(c == 'P')?;
                    Ok(ClassAtom::Prop(prop))
                }
                Some('n') => Ok(ClassAtom::Char('\n' as u32)),
                Some('t') => Ok(ClassAtom::Char('\t' as u32)),
                Some('r') => Ok(ClassAtom::Char('\r' as u32)),
                Some('f') => Ok(ClassAtom::Char(0x0C)),
                Some('v') => Ok(ClassAtom::Char(0x0B)),
                Some('0') => {
                    if self.unicode && self.peek().is_some_and(|d| d.is_ascii_digit()) {
                        return Err("legacy octal escape in unicode pattern".into());
                    }
                    // Annex B: `\0` continues as a LegacyOctalEscapeSequence in a class.
                    let mut v = 0u32;
                    if !self.unicode {
                        for _ in 0..2 {
                            match self.peek() {
                                Some(d @ '0'..='7') => {
                                    v = v * 8 + d.to_digit(8).unwrap();
                                    self.bump();
                                }
                                _ => break,
                            }
                        }
                    }
                    Ok(ClassAtom::Char(v))
                }
                Some(c) if !self.unicode && c.is_ascii_digit() => {
                    // Annex B class octal escape; \8 and \9 are identity digits.
                    if c >= '8' {
                        return Ok(ClassAtom::Char(c as u32));
                    }
                    let mut v = c.to_digit(8).unwrap();
                    let max_more = if c <= '3' { 2 } else { 1 };
                    for _ in 0..max_more {
                        match self.peek() {
                            Some(d @ '0'..='7') => {
                                v = v * 8 + d.to_digit(8).unwrap();
                                self.bump();
                            }
                            _ => break,
                        }
                    }
                    Ok(ClassAtom::Char(v))
                }
                Some('b') => Ok(ClassAtom::Char(0x08)),
                Some('c') => match self.peek() {
                    Some(l) if l.is_ascii_alphabetic() => {
                        self.bump();
                        Ok(ClassAtom::Char((l as u8 % 32) as u32))
                    }
                    // Annex B ClassControlLetter also admits digits and '_'.
                    Some(l) if !self.unicode && (l.is_ascii_digit() || l == '_') => {
                        self.bump();
                        Ok(ClassAtom::Char((l as u8 % 32) as u32))
                    }
                    _ if self.unicode => Err("invalid \\c escape in unicode pattern".into()),
                    _ => {
                        self.pos -= 1; // un-consume the 'c': `\` is a literal backslash member
                        Ok(ClassAtom::Char('\\' as u32))
                    }
                },
                Some('x') => {
                    if self.unicode {
                        Ok(ClassAtom::Char(self.hex_strict(2)?))
                    } else {
                        Ok(ClassAtom::Char(self.hex(2, 'x')))
                    }
                }
                Some('u') => {
                    if self.unicode {
                        Ok(ClassAtom::Char(self.unicode_escape_strict()?))
                    } else {
                        Ok(ClassAtom::Char(self.unicode_escape()))
                    }
                }
                Some(c) if self.unicode && !is_regex_syntax_char(c) && c != '/' && c != '-' => {
                    Err(format!("invalid identity escape \\{c} in unicode class"))
                }
                Some(c) => Ok(ClassAtom::Char(cp_of_elem(c))),
            },
            Some(c) => Ok(ClassAtom::Char(cp_of_elem(c))),
        }
    }

    fn parse_escape(&mut self) -> Result<Node, String> {
        match self.bump() {
            None => Err("trailing backslash".into()),
            Some(c @ ('d' | 'D' | 'w' | 'W' | 's' | 'S')) => Ok(Node::Class(CharClass {
                builtins: vec![c],
                ..Default::default()
            })),
            Some(c @ ('p' | 'P')) if self.unicode => {
                // In v-mode a property escape may be a property of *strings* (a computed set).
                if self.unicode_sets {
                    let set = self.parse_class_set_property(c == 'P')?;
                    return Ok(class_set_to_node(set));
                }
                let prop = self.parse_prop_escape(c == 'P')?;
                Ok(Node::Class(CharClass {
                    props: vec![prop],
                    ..Default::default()
                }))
            }
            Some('b') => Ok(Node::WordB(true)),
            Some('B') => Ok(Node::WordB(false)),
            Some('k') if self.named_mode => {
                // `\k<name>` — a named back-reference (resolved after the full parse).
                if self.peek() != Some('<') {
                    return Err("expected '<' in named back reference".into());
                }
                self.bump();
                let name = self.parse_group_name()?;
                self.name_refs.push(name.clone());
                Ok(Node::NamedBackref(name))
            }
            Some('n') => Ok(Node::Char('\n' as u32)),
            Some('t') => Ok(Node::Char('\t' as u32)),
            Some('r') => Ok(Node::Char('\r' as u32)),
            Some('f') => Ok(Node::Char(0x0C)),
            Some('v') => Ok(Node::Char(0x0B)),
            Some('0') => {
                // `\0` may not be followed by a digit in Unicode mode (a legacy octal escape).
                if self.unicode && self.peek().is_some_and(|d| d.is_ascii_digit()) {
                    return Err("legacy octal escape in unicode pattern".into());
                }
                // Annex B: `\0` continues as a LegacyOctalEscapeSequence (up to 2 more digits).
                let mut v = 0u32;
                if !self.unicode {
                    for _ in 0..2 {
                        match self.peek() {
                            Some(d @ '0'..='7') => {
                                v = v * 8 + d.to_digit(8).unwrap();
                                self.bump();
                            }
                            _ => break,
                        }
                    }
                }
                Ok(Node::Char(v))
            }
            Some('c') => {
                // `\cX` (a letter) is a control escape; otherwise Annex B treats the `\` as a
                // literal backslash and reparses the `c` as a plain character.
                match self.peek() {
                    Some(l) if l.is_ascii_alphabetic() => {
                        self.bump();
                        Ok(Node::Char((l as u8 % 32) as u32))
                    }
                    _ if self.unicode => Err("invalid \\c escape in unicode pattern".into()),
                    _ => {
                        self.pos -= 1; // un-consume the 'c'
                        Ok(Node::Char('\\' as u32))
                    }
                }
            }
            Some('x') => {
                if self.unicode {
                    Ok(Node::Char(self.hex_strict(2)?))
                } else {
                    Ok(Node::Char(self.hex(2, 'x')))
                }
            }
            Some('u') => {
                if self.unicode {
                    Ok(Node::Char(self.unicode_escape_strict()?))
                } else {
                    Ok(Node::Char(self.unicode_escape()))
                }
            }
            Some(c) if c.is_ascii_digit() => {
                let start = self.pos;
                let mut num = c.to_digit(10).unwrap() as usize;
                while let Some(d) = self.peek() {
                    if d.is_ascii_digit() {
                        num = num.saturating_mul(10) + d.to_digit(10).unwrap() as usize;
                        self.bump();
                    } else {
                        break;
                    }
                }
                if self.unicode || (num >= 1 && num <= self.total_groups) {
                    return Ok(Node::Backref(num));
                }
                // Annex B: a decimal escape naming no capture group is a LegacyOctalEscapeSequence
                // (\8 and \9 are identity escapes); trailing digits reparse as literal atoms.
                self.pos = start;
                if c >= '8' {
                    return Ok(Node::Char(c as u32));
                }
                let mut v = c.to_digit(8).unwrap();
                let max_more = if c <= '3' { 2 } else { 1 };
                for _ in 0..max_more {
                    match self.peek() {
                        Some(d @ '0'..='7') => {
                            v = v * 8 + d.to_digit(8).unwrap();
                            self.bump();
                        }
                        _ => break,
                    }
                }
                Ok(Node::Char(v))
            }
            // IdentityEscape in Unicode mode is a SyntaxCharacter or '/' only.
            Some(c) if self.unicode && !is_regex_syntax_char(c) && c != '/' => {
                Err(format!("invalid identity escape \\{c} in unicode pattern"))
            }
            Some(c) => Ok(Node::Char(cp_of_elem(c))),
        }
    }

    /// Parse a `\p{Name}` / `\p{Name=Value}` body (the `\p`/`\P` already consumed). `negate` is true
    /// for `\P`. Returns `(negated, ranges)`. Only valid in Unicode mode; an unknown property errors.
    fn parse_prop_escape(&mut self, negate: bool) -> Result<(bool, &'static [(u32, u32)]), String> {
        if self.bump() != Some('{') {
            return Err("invalid property escape: expected '{'".into());
        }
        let mut body = String::new();
        loop {
            match self.bump() {
                Some('}') => break,
                // The grammar is `[A-Za-z0-9_]` names, optionally `name=value` — no spaces or other
                // characters (so `\p{ Gc=L }` with spaces is a SyntaxError, not loose-matched).
                Some(c) if c.is_ascii_alphanumeric() || c == '_' || c == '=' => body.push(c),
                Some(_) => return Err("invalid character in property escape".into()),
                None => return Err("unterminated property escape".into()),
            }
        }
        let (name, value) = match body.split_once('=') {
            Some((n, v)) => (n, Some(v)),
            None => (body.as_str(), None),
        };
        // Exact spellings only — `\p{…}` does not do UAX44 loose matching.
        match crate::unicode_props::lookup_strict(name, value) {
            Some((complement, ranges)) => Ok((negate != complement, ranges)),
            None => Err(format!("invalid unicode property {body}")),
        }
    }

    /// Read a `(?<name>` capture-group name (the `>` is consumed). A name is a `RegExpIdentifierName`:
    /// an IdentifierName, optionally using `\u` escapes, validated against ID_Start / ID_Continue.
    fn parse_group_name(&mut self) -> Result<String, String> {
        let mut name = String::new();
        loop {
            match self.peek() {
                Some('>') => {
                    self.bump();
                    break;
                }
                Some('\\') => {
                    self.bump();
                    if self.peek() == Some('u') {
                        self.bump();
                        let mut cp = self.unicode_escape_u32();
                        // A `\uD8xx\uDCxx` lead/trail escape pair combines into one code point.
                        if (0xD800..=0xDBFF).contains(&cp)
                            && self.peek() == Some('\\')
                            && self.chars.get(self.pos + 1) == Some(&'u')
                        {
                            let save = self.pos;
                            self.bump();
                            self.bump();
                            let trail = self.unicode_escape_u32();
                            if (0xDC00..=0xDFFF).contains(&trail) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (trail - 0xDC00);
                            } else {
                                self.pos = save;
                            }
                        }
                        name.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    } else {
                        return Err("invalid escape in capture group name".into());
                    }
                }
                Some(c) => {
                    self.bump();
                    // In non-unicode mode the elements are code units: recombine a smuggled
                    // surrogate pair into the character it encodes.
                    if let Some(&next) = self.chars.get(self.pos) {
                        if let Some(real) = crate::jstr::paired_char(c, next) {
                            self.bump();
                            name.push(real);
                            continue;
                        }
                    }
                    match crate::jstr::smuggled(c) {
                        // A truly lone surrogate can never be part of an identifier.
                        Some(_) => return Err("invalid capture group name".into()),
                        None => name.push(c),
                    }
                }
                None => return Err("unterminated capture group name".into()),
            }
        }
        let mut chars = name.chars();
        let valid =
            matches!(chars.next(), Some(c) if regex_ident_start(c)) && chars.all(regex_ident_part);
        if !valid {
            return Err(format!("invalid capture group name <{name}>"));
        }
        Ok(name)
    }

    /// Annex B ExtendedHexEscapeSequence: `\x` needs exactly `n` hex digits, otherwise the whole
    /// escape is an IdentityEscape for `esc` (consuming nothing past it).
    fn hex(&mut self, n: usize, esc: char) -> u32 {
        let save = self.pos;
        let mut s = String::new();
        for _ in 0..n {
            match self.peek() {
                Some(c) if c.is_ascii_hexdigit() => {
                    s.push(c);
                    self.bump();
                }
                _ => {
                    self.pos = save;
                    return esc as u32;
                }
            }
        }
        u32::from_str_radix(&s, 16).unwrap_or(0xFFFD)
    }

    /// Four hex digits as a raw value (surrogate halves pass through).
    fn hex4_u32(&mut self) -> u32 {
        let mut s = String::new();
        for _ in 0..4 {
            if let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() {
                    s.push(c);
                    self.bump();
                }
            }
        }
        u32::from_str_radix(&s, 16).unwrap_or(0xFFFD)
    }

    /// A non-strict (Annex B) `\u` escape: exactly four hex digits or `{…}`, otherwise the
    /// whole escape is an IdentityEscape for `u` (consuming nothing).
    fn unicode_escape(&mut self) -> u32 {
        // Annex B (no `u` flag): `\u{` is an identity escape for `u` followed by a quantifier —
        // braced code-point escapes exist only in Unicode mode.
        let save = self.pos;
        let mut v: u32 = 0;
        for _ in 0..4 {
            match self.peek().and_then(|c| c.to_digit(16)) {
                Some(d) => {
                    v = v * 16 + d;
                    self.bump();
                }
                None => {
                    self.pos = save;
                    return 'u' as u32;
                }
            }
        }
        v
    }

    /// Exactly `n` hex digits, or a SyntaxError (Unicode mode).
    fn hex_strict(&mut self, n: usize) -> Result<u32, String> {
        let mut v: u32 = 0;
        for _ in 0..n {
            match self.peek().and_then(|c| c.to_digit(16)) {
                Some(d) => {
                    v = v * 16 + d;
                    self.bump();
                }
                None => return Err("invalid hexadecimal escape".into()),
            }
        }
        Ok(v)
    }

    /// A Unicode-mode `\u` escape: `{…}` bodies are strictly hex and capped at U+10FFFF, plain
    /// escapes are exactly four hex digits, and a lead/trail surrogate escape pair combines into
    /// one code point.
    fn unicode_escape_strict(&mut self) -> Result<u32, String> {
        if self.peek() == Some('{') {
            self.bump();
            let mut v: u32 = 0;
            let mut any = false;
            loop {
                match self.peek() {
                    Some('}') => {
                        self.bump();
                        break;
                    }
                    Some(c) if c.is_ascii_hexdigit() => {
                        any = true;
                        v = v.saturating_mul(16).saturating_add(c.to_digit(16).unwrap());
                        self.bump();
                    }
                    _ => return Err("invalid code point escape".into()),
                }
            }
            if !any || v > 0x10FFFF {
                return Err("invalid code point escape".into());
            }
            return Ok(v);
        }
        let mut lead: u32 = 0;
        for _ in 0..4 {
            match self.peek().and_then(|c| c.to_digit(16)) {
                Some(d) => {
                    lead = lead * 16 + d;
                    self.bump();
                }
                None => return Err("invalid unicode escape".into()),
            }
        }
        // Combine a surrogate escape pair into a single code point.
        if (0xD800..=0xDBFF).contains(&lead)
            && self.peek() == Some('\\')
            && self.chars.get(self.pos + 1) == Some(&'u')
        {
            let save = self.pos;
            self.bump();
            self.bump();
            let mut trail: u32 = 0;
            let mut ok = true;
            for _ in 0..4 {
                match self.peek().and_then(|c| c.to_digit(16)) {
                    Some(d) => {
                        trail = trail * 16 + d;
                        self.bump();
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && (0xDC00..=0xDFFF).contains(&trail) {
                let cp = 0x10000 + ((lead - 0xD800) << 10) + (trail - 0xDC00);
                return Ok(cp);
            }
            self.pos = save;
        }
        Ok(lead)
    }

    /// The raw code-point value of a `\u` escape body (surrogate values pass through).
    fn unicode_escape_u32(&mut self) -> u32 {
        if self.peek() == Some('{') {
            self.bump();
            let mut s = String::new();
            while let Some(c) = self.peek() {
                if c == '}' {
                    self.bump();
                    break;
                }
                s.push(c);
                self.bump();
            }
            u32::from_str_radix(&s, 16).unwrap_or(0xFFFD)
        } else {
            self.hex4_u32()
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.bump() == Some(c) {
            Ok(())
        } else {
            Err(format!("expected '{c}' in pattern"))
        }
    }
}

enum ClassAtom {
    Char(u32),
    Builtin(char),
    Prop((bool, &'static [(u32, u32)])),
}

fn push_class_atom(cc: &mut CharClass, a: ClassAtom) {
    match a {
        ClassAtom::Char(c) => cc.ranges.push((c, c)),
        ClassAtom::Builtin(b) => cc.builtins.push(b),
        ClassAtom::Prop(p) => cc.props.push(p),
    }
}

// ---------------------------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------------------------

/// Lower `node` onto `prog`. A `backward` node belongs to a lookbehind body, which runs
/// right-to-left: its concatenations are emitted in reverse, while alternative order, greed and
/// nested lookarounds (which set their own direction) are unchanged.
fn compile(
    node: &Node,
    prog: &mut Vec<Inst>,
    cx: &mut Lowering,
    backward: bool,
) -> Result<(), String> {
    with_pattern_stack(|| compile_node(node, prog, cx, backward))
}

fn compile_node(
    node: &Node,
    prog: &mut Vec<Inst>,
    cx: &mut Lowering,
    backward: bool,
) -> Result<(), String> {
    match node {
        Node::Empty => {}
        Node::Char(c) => prog.push(Inst::Char(*c)),
        Node::Any => prog.push(Inst::Any),
        Node::Class(cc) => prog.push(Inst::Class(Rc::new(clone_class(cc)))),
        Node::StringSet(set) => prog.push(Inst::StringSet(set.clone())),
        Node::Start => prog.push(Inst::AssertStart),
        Node::End => prog.push(Inst::AssertEnd),
        Node::WordB(b) => prog.push(Inst::WordBoundary(*b)),
        Node::Backref(n) => prog.push(Inst::Backref(*n)),
        Node::BackrefAlt(v) => prog.push(Inst::BackrefAlt(Rc::new(v.clone()))),
        // Resolved to `Backref` before compile; treat any stray one as group 0 (never matches).
        Node::NamedBackref(_) => prog.push(Inst::Backref(0)),
        Node::Modifier { add, remove, inner } => {
            let opt = |a: bool, r: bool| {
                if a {
                    Some(true)
                } else if r {
                    Some(false)
                } else {
                    None
                }
            };
            prog.push(Inst::PushFlags(
                opt(add.0, remove.0),
                opt(add.1, remove.1),
                opt(add.2, remove.2),
            ));
            compile(inner, prog, cx, backward)?;
            prog.push(Inst::PopFlags);
        }
        Node::Concat(v) => {
            if backward {
                for n in v.iter().rev() {
                    compile(n, prog, cx, backward)?;
                }
            } else {
                for n in v {
                    compile(n, prog, cx, backward)?;
                }
            }
        }
        Node::Alt(v) => {
            let mut jmp_ends = Vec::new();
            for (i, alt) in v.iter().enumerate() {
                if i < v.len() - 1 {
                    let sp = prog.len();
                    prog.push(Inst::Split(0, 0));
                    let a_start = prog.len();
                    compile(alt, prog, cx, backward)?;
                    jmp_ends.push(prog.len());
                    prog.push(Inst::Jmp(0));
                    let next = prog.len();
                    prog[sp] = Inst::Split(a_start, next);
                } else {
                    compile(alt, prog, cx, backward)?;
                }
            }
            let end = prog.len();
            for j in jmp_ends {
                prog[j] = Inst::Jmp(end);
            }
        }
        Node::Group(idx, inner) => {
            if let Some(i) = idx {
                prog.push(Inst::Save(2 * i));
            }
            compile(inner, prog, cx, backward)?;
            if let Some(i) = idx {
                prog.push(Inst::Save(2 * i + 1));
            }
        }
        Node::Look(negate, inner) => {
            // A lookahead body always matches forward, even inside a lookbehind.
            let mut sub = Vec::new();
            compile(inner, &mut sub, cx, false)?;
            sub.push(Inst::Match);
            prog.push(Inst::Look {
                negate: *negate,
                prog: Rc::new(sub),
            });
        }
        Node::LookBehind(negate, inner) => {
            // The body is compiled backward and executed right-to-left.
            let mut sub = Vec::new();
            compile(inner, &mut sub, cx, true)?;
            sub.push(Inst::Match);
            prog.push(Inst::LookBehind {
                negate: *negate,
                prog: Rc::new(sub),
            });
        }
        Node::Repeat(inner, min, max, greedy) => {
            compile_repeat(inner, *min, *max, *greedy, prog, cx, backward)?
        }
    }
    Ok(())
}

/// Lowering state threaded through [`compile`].
struct Lowering {
    /// Mark and counter slots allocated so far.
    marks: usize,
    /// Instructions that unrolled repetitions may still add (see [`MAX_UNROLLED_LEN`]).
    unroll_budget: usize,
}

impl Lowering {
    fn mark(&mut self) -> usize {
        self.marks += 1;
        self.marks - 1
    }
}

/// How many instructions unrolling may add to a whole program. Unrolled copies are straight-line
/// code, but they multiply: each nesting level of counted repetition would multiply the program
/// again. Repetitions beyond the budget run as counted loops over a single copy of their body,
/// keeping the program linear in the pattern size.
const MAX_UNROLLED_LEN: usize = 1 << 16;

/// Append `body`, compiled as a program of its own, at the end of `prog`, rebasing its jumps.
fn append_relocated(prog: &mut Vec<Inst>, body: &[Inst]) {
    let base = prog.len();
    prog.extend(body.iter().map(|instruction| match instruction {
        Inst::Split(a, b) => Inst::Split(a + base, b + base),
        Inst::Jmp(target) => Inst::Jmp(target + base),
        Inst::CountedBranch {
            id,
            min,
            max,
            greedy,
            body,
            exit,
        } => Inst::CountedBranch {
            id: *id,
            min: *min,
            max: *max,
            greedy: *greedy,
            body: body + base,
            exit: exit + base,
        },
        other => other.clone(),
    }));
}

/// ECMA-262 RepeatMatcher (§22.2.2.3.1) as a loop whose iteration count lives in a `marks` slot
/// (see `Inst::CountedBranch`), for a repetition too large to unroll:
///
/// ```text
///       CounterReset(c)
/// head: CountedBranch { c, min, max, greedy, body, exit }
/// body: SetMark(m); ClearCaps; atom; CheckCountedProgress(m, c, min); CounterIncr(c); Jmp(head)
/// exit:
/// ```
///
/// Each iteration clears the captures inside the atom first (RepeatMatcher step 4), and only an
/// optional iteration, one past `min`, fails when it consumes nothing (step 2.b).
fn compile_counted_repeat(
    body: &[Inst],
    min: usize,
    max: Option<usize>,
    greedy: bool,
    prog: &mut Vec<Inst>,
    cx: &mut Lowering,
) {
    let counter = cx.mark();
    let progress = cx.mark();
    prog.push(Inst::CounterReset(counter));
    let head = prog.len();
    prog.push(Inst::Match); // Patched to the loop head below.
    let start = prog.len();
    prog.push(Inst::SetMark(progress));
    append_relocated(prog, body);
    prog.push(Inst::CheckCountedProgress {
        mark: progress,
        counter,
        min,
    });
    prog.push(Inst::CounterIncr(counter));
    prog.push(Inst::Jmp(head));
    prog[head] = Inst::CountedBranch {
        id: counter,
        min,
        max,
        greedy,
        body: start,
        exit: prog.len(),
    };
}

fn compile_repeat(
    inner: &Node,
    min: usize,
    max: Option<usize>,
    greedy: bool,
    prog: &mut Vec<Inst>,
    cx: &mut Lowering,
    backward: bool,
) -> Result<(), String> {
    // A variable-length UnicodeSets atom needs explicit heap state: compiling it through the
    // general Split/SetMark loop recurses once per matched string and exhausts the native matcher
    // stack on otherwise linear, standards-generated emoji corpora.
    if let Node::StringSet(set) = inner {
        prog.push(Inst::StringSetRepeat {
            set: set.clone(),
            min,
            max,
            greedy,
        });
        return Ok(());
    }
    // Fast path: a repeated single-character atom consumes iteratively (no per-character
    // recursion), so arbitrarily large counts (up to 2^53-1) cost nothing to compile.
    if let Some(rep) = single_char_rep(inner) {
        prog.push(Inst::Many {
            rep,
            min,
            max,
            greedy,
        });
        return Ok(());
    }
    // The atom is compiled once, as a program of its own, and copied into place. RepeatMatcher
    // clears the captures inside it at the start of every iteration.
    let mut body = Vec::new();
    if let Some((lo, hi)) = cap_span(inner) {
        body.push(Inst::ClearCaps(lo, hi));
    }
    compile(inner, &mut body, cx, backward)?;
    // Short repetitions unroll into `min` copies and one optional copy per further iteration,
    // which keeps them straight-line. ECMA-262 places no bound on the counts, so larger ones run
    // as a counted loop whose size doesn't depend on them.
    let optional_len = match max {
        None => body.len() + 4,
        Some(max) => (max - min).saturating_mul(body.len() + 3),
    };
    let unrolled_len = min.saturating_mul(body.len()).saturating_add(optional_len);
    let added = unrolled_len.saturating_sub(body.len());
    if min > MAX_REPEAT || max.is_some_and(|max| max > MAX_REPEAT) || added > cx.unroll_budget {
        compile_counted_repeat(&body, min, max, greedy, prog, cx);
        return Ok(());
    }
    cx.unroll_budget -= added;
    for _ in 0..min {
        append_relocated(prog, &body);
    }
    // Optional iterations enforce the RepeatMatcher empty-iteration rule: an iteration that
    // consumes nothing fails, backtracking into the body's other alternatives or out of the loop.
    // The iterations run one after another, so they share one progress mark.
    match max {
        None => {
            // Greedy: L1: Split(body, end); body; Jmp(L1); end.
            let id = cx.mark();
            let l1 = prog.len();
            prog.push(Inst::Split(0, 0));
            let start = prog.len();
            prog.push(Inst::SetMark(id));
            append_relocated(prog, &body);
            prog.push(Inst::CheckProgress(id));
            prog.push(Inst::Jmp(l1));
            let end = prog.len();
            prog[l1] = if greedy {
                Inst::Split(start, end)
            } else {
                Inst::Split(end, start)
            };
        }
        Some(max) if max > min => {
            let id = cx.mark();
            let mut splits = Vec::new();
            for _ in min..max {
                let split = prog.len();
                prog.push(Inst::Split(0, 0));
                splits.push((split, prog.len()));
                prog.push(Inst::SetMark(id));
                append_relocated(prog, &body);
                prog.push(Inst::CheckProgress(id));
            }
            let end = prog.len();
            for (split, start) in splits {
                prog[split] = if greedy {
                    Inst::Split(start, end)
                } else {
                    Inst::Split(end, start)
                };
            }
        }
        Some(_) => {}
    }
    Ok(())
}

/// The min/max capture-group indices inside `node`, if any (for per-iteration capture resets).
fn cap_span(node: &Node) -> Option<(usize, usize)> {
    let merge = |a: Option<(usize, usize)>, b: Option<(usize, usize)>| match (a, b) {
        (Some((l1, h1)), Some((l2, h2))) => Some((l1.min(l2), h1.max(h2))),
        (x, None) | (None, x) => x,
    };
    with_pattern_stack(|| match node {
        Node::Group(idx, inner) => merge(idx.map(|i| (i, i)), cap_span(inner)),
        Node::Concat(v) | Node::Alt(v) => v.iter().fold(None, |acc, n| merge(acc, cap_span(n))),
        Node::Repeat(inner, ..)
        | Node::Look(_, inner)
        | Node::LookBehind(_, inner)
        | Node::Modifier { inner, .. } => cap_span(inner),
        _ => None,
    })
}

/// The largest numeric back reference in the pattern (0 when there are none).
fn max_backref(node: &Node, out: &mut usize) {
    with_pattern_stack(|| match node {
        Node::Backref(n) => *out = (*out).max(*n),
        Node::Concat(items) | Node::Alt(items) => {
            for n in items {
                max_backref(n, out);
            }
        }
        Node::Group(_, inner)
        | Node::Repeat(inner, _, _, _)
        | Node::Look(_, inner)
        | Node::LookBehind(_, inner)
        | Node::Modifier { inner, .. } => max_backref(inner, out),
        _ => {}
    })
}

/// Replace each `\k<name>` (`Node::NamedBackref`) with the numeric `Backref` of its group. Names are
/// validated before this runs, so an unknown name resolves to group 0 (never matches), harmlessly.
/// Reject same-name capture groups that could both match (i.e. live in the same concatenation);
/// duplicates spread across different alternation branches are allowed (ES2025).
fn validate_group_names(node: &Node, names: &[(String, usize)]) -> Result<(), String> {
    collect_group_names(node, names)?;
    Ok(())
}

fn collect_group_names(
    node: &Node,
    names: &[(String, usize)],
) -> Result<std::collections::HashSet<String>, String> {
    use std::collections::HashSet;
    with_pattern_stack(|| match node {
        Node::Group(idx, inner) => {
            let mut s = collect_group_names(inner, names)?;
            if let Some(idx) = idx {
                if let Some((name, _)) = names.iter().find(|(_, i)| i == idx) {
                    if !s.insert(name.clone()) {
                        return Err(format!("duplicate group name {name}"));
                    }
                }
            }
            Ok(s)
        }
        Node::Look(_, inner) | Node::LookBehind(_, inner) | Node::Repeat(inner, _, _, _) => {
            collect_group_names(inner, names)
        }
        Node::Modifier { inner, .. } => collect_group_names(inner, names),
        Node::Concat(children) => {
            let mut all = HashSet::new();
            for c in children {
                for n in collect_group_names(c, names)? {
                    if !all.insert(n.clone()) {
                        return Err(format!("duplicate group name {n}"));
                    }
                }
            }
            Ok(all)
        }
        Node::Alt(branches) => {
            let mut union = HashSet::new();
            for b in branches {
                union.extend(collect_group_names(b, names)?);
            }
            Ok(union)
        }
        _ => Ok(std::collections::HashSet::new()),
    })
}

fn resolve_named_backrefs(node: &mut Node, names: &[(String, usize)]) {
    with_pattern_stack(|| match node {
        Node::NamedBackref(name) => {
            let idxs: Vec<usize> = names
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, i)| *i)
                .collect();
            *node = match idxs.len() {
                0 => Node::Backref(0),
                1 => Node::Backref(idxs[0]),
                _ => Node::BackrefAlt(idxs),
            };
        }
        Node::Concat(v) | Node::Alt(v) => {
            v.iter_mut().for_each(|n| resolve_named_backrefs(n, names))
        }
        Node::Group(_, inner)
        | Node::Repeat(inner, ..)
        | Node::Look(_, inner)
        | Node::LookBehind(_, inner)
        | Node::Modifier { inner, .. } => resolve_named_backrefs(inner, names),
        _ => {}
    })
}

/// If `node` matches exactly one code point, return it as a `Rep` (for the `Inst::Many` fast path).
fn single_char_rep(node: &Node) -> Option<Rep> {
    match node {
        Node::Char(c) => Some(Rep::Char(*c)),
        Node::Any => Some(Rep::Any),
        Node::Class(cc) => Some(Rep::Class(Rc::new(clone_class(cc)))),
        _ => None,
    }
}

fn clone_class(cc: &CharClass) -> CharClass {
    let mut cloned = CharClass {
        negate: cc.negate,
        ranges: cc.ranges.clone(),
        builtins: cc.builtins.clone(),
        props: cc.props.clone(),
        ascii_lut: None,
    };
    let mut lut = Box::new([false; 256]);
    for (code, slot) in lut.iter_mut().enumerate() {
        *slot = cloned.matches_raw2(code as u32, false, false) ^ cloned.negate;
    }
    cloned.ascii_lut = Some(lut);
    cloned
}

// ---------------------------------------------------------------------------------------------
// Backtracking matcher
// ---------------------------------------------------------------------------------------------

fn find_ascii_literal(
    subject: &[u8],
    start: usize,
    literal: &[u8],
    sticky: bool,
    control: &crate::RuntimeInterrupt,
) -> MatchResult<(usize, usize)> {
    if start > subject.len() || literal.len() > subject.len().saturating_sub(start) {
        return Ok(None);
    }
    if sticky {
        return Ok(subject[start..]
            .starts_with(literal)
            .then_some((start, start + literal.len())));
    }
    let mut from = start;
    while from + literal.len() <= subject.len() {
        let Some(found) = subject.find_byte(from, literal[0], control)? else {
            return Ok(None);
        };
        if found + literal.len() > subject.len() {
            return Ok(None);
        }
        if subject[found..].starts_with(literal) {
            return Ok(Some((found, found + literal.len())));
        }
        from = found + 1;
    }
    Ok(None)
}

/// Canonicalize equivalence for a non-`u` case-insensitive pattern (ECMA-262 §22.2.8.3
/// Canonicalize with `unicode` false): only ASCII letters fold, `A`↔`a` … `Z`↔`z`.
#[inline(always)]
fn icase_bytes_eq(a: u8, b: u8) -> bool {
    a == b || (a.is_ascii_alphabetic() && b.is_ascii_alphabetic() && a | 0x20 == b | 0x20)
}

/// Case-insensitive literal search over an ASCII subject: the needle is the UTF-8 encoding of
/// the pattern's literal characters, compared byte-wise under canonicalize equivalence. Every
/// needle starts at a valid scalar value, so in TRust JS's surrogate-smuggled, always-valid UTF-8
/// subject a byte match can only align at a scalar boundary; non-ASCII pattern bytes (e.g. `ß`)
/// match by byte equality because canonicalization never folds them. Semantically identical to
/// `find_ascii_literal` plus the fold at every byte.
fn find_ascii_fold_literal(
    subject: &[u8],
    start: usize,
    literal: &[u8],
    sticky: bool,
    control: &crate::RuntimeInterrupt,
) -> MatchResult<(usize, usize)> {
    if start > subject.len() || literal.len() > subject.len().saturating_sub(start) {
        return Ok(None);
    }
    if sticky {
        return Ok((subject[start..].len() >= literal.len()
            && match_literal_fold(&subject[start..start + literal.len()], literal))
        .then_some((start, start + literal.len())));
    }
    let mut from = start;
    while from + literal.len() <= subject.len() {
        let Some(found) = find_fold_byte(subject, from, literal[0], control)? else {
            return Ok(None);
        };
        if found + literal.len() > subject.len() {
            return Ok(None);
        }
        if match_literal_fold(&subject[found..found + literal.len()], literal) {
            return Ok(Some((found, found + literal.len())));
        }
        from = found + 1;
    }
    Ok(None)
}

/// Byte-equality fast path (memcmp) first, then the per-byte fold scan — most candidate hits
/// agree exactly, so the common case avoids the per-byte branches.
fn match_literal_fold(hay: &[u8], literal: &[u8]) -> bool {
    hay == literal || hay.iter().zip(literal).all(|(a, b)| icase_bytes_eq(*a, *b))
}

/// First fold-byte scan, mirroring `ReInput::find_byte`'s interrupt-poll cadence.
fn find_fold_byte(
    subject: &[u8],
    mut from: usize,
    byte: u8,
    control: &crate::RuntimeInterrupt,
) -> MatchResult<usize> {
    while from < subject.len() {
        if icase_bytes_eq(subject[from], byte) {
            return Ok(Some(from));
        }
        from += 1;
        if from & INTERRUPT_POLL_MASK == 0 {
            poll_interrupt(control)?;
        }
    }
    Ok(None)
}

/// The matcher's view of a subject: element `i` as a code point / code unit. Monomorphized for
/// bytes (an ASCII subject — the common case, matched with no `Vec<u32>` materialization at all)
/// and for wide elements (anything non-ASCII).
pub trait ReInput: Copy {
    fn len(&self) -> usize;
    fn at(&self, i: usize) -> u32;

    #[inline]
    fn find_byte(
        &self,
        mut from: usize,
        byte: u8,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<usize> {
        while from < self.len() {
            if self.at(from) == byte as u32 {
                return Ok(Some(from));
            }
            from += 1;
            if from & INTERRUPT_POLL_MASK == 0 {
                poll_interrupt(control)?;
            }
        }
        Ok(None)
    }
}

impl ReInput for &[u8] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }
    #[inline(always)]
    fn at(&self, i: usize) -> u32 {
        self[i] as u32
    }

    #[inline]
    fn find_byte(
        &self,
        from: usize,
        byte: u8,
        control: &crate::RuntimeInterrupt,
    ) -> MatchResult<usize> {
        let Some(bytes) = self.get(from..) else {
            return Ok(None);
        };
        let repeated = u64::from_ne_bytes([byte; 8]);
        let low_bits = 0x0101_0101_0101_0101u64;
        let high_bits = 0x8080_8080_8080_8080u64;
        let bulk_len = bytes.len() / 8 * 8;
        for (chunk_index, chunk) in bytes[..bulk_len].chunks(8).enumerate() {
            if chunk_index & (INTERRUPT_POLL_MASK / 8) == 0 {
                poll_interrupt(control)?;
            }
            let word = u64::from_ne_bytes(chunk.try_into().unwrap());
            let different = word ^ repeated;
            if different.wrapping_sub(low_bits) & !different & high_bits != 0 {
                if let Some(offset) = chunk.iter().position(|candidate| *candidate == byte) {
                    return Ok(Some(from + chunk_index * 8 + offset));
                }
            }
        }
        let tail_start = from + bulk_len;
        Ok(bytes[bulk_len..]
            .iter()
            .position(|candidate| *candidate == byte)
            .map(|offset| tail_start + offset))
    }
}

impl ReInput for &[u32] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[u32]>::len(self)
    }
    #[inline(always)]
    fn at(&self, i: usize) -> u32 {
        self[i]
    }
}

/// Sentinel for an unset capture or mark slot inside a [`Backtrack`] record. Subject positions
/// and iteration counts never reach it.
const UNSET: usize = usize::MAX;

#[inline(always)]
fn pack_slot(slot: Option<usize>) -> usize {
    slot.unwrap_or(UNSET)
}

#[inline(always)]
fn unpack_slot(slot: usize) -> Option<usize> {
    (slot != UNSET).then_some(slot)
}

/// One entry of the matcher's backtrack stack: either a choice point (an untried alternative to
/// resume) or an undo record for state changed since the choice point below it. Every entry is
/// 16 bytes; a choice needing a third word keeps it in the [`Backtrack::Extra`] directly below.
#[derive(Clone, Copy)]
enum Backtrack {
    /// Choice: resume the current program at `pc` and `pos`.
    Alt {
        pc: u32,
        pos: usize,
    },
    /// Undo a capture write (`old` is [`UNSET`] for an unset slot).
    Capture {
        slot: u32,
        old: usize,
    },
    /// Undo a mark or counter write.
    Mark {
        id: u32,
        old: usize,
    },
    /// Undo `PushFlags` / `PopFlags`.
    PopFlags,
    PushFlags(bool, bool, bool),
    /// Restore the capture slots from `2 * lo` on from the newest snapshot, `saved[at..]`.
    Captures {
        lo: u32,
        at: usize,
    },
    /// The operand of the choice directly above: a `Many` start position, or a `StringSetRepeat`
    /// iteration count.
    Extra(usize),
    /// Choice: a greedy `Many` at `pc` that now ends at `end`; retry it with fewer elements.
    Greedy {
        pc: u32,
        end: usize,
    },
    /// Choice: a lazy `Many` at `pc` that now ends at `end`; retry it with more elements.
    Lazy {
        pc: u32,
        end: usize,
    },
    /// Choice: enter the `StringSetRepeat` at `pc` at `pos` after `Extra` iterations.
    SetIteration {
        pc: u32,
        pos: usize,
    },
    /// Every alternative of the `StringSet`/`StringSetRepeat` state at `pc`, `pos` (with the
    /// memoized iteration count in `Extra`) failed: record it in the failure memo.
    SetFailure {
        pc: u32,
        pos: usize,
    },
    /// The innermost lookaround's body failed (see [`LookFrame`]).
    LookFail,
}

/// Ceiling on the matcher's backtracking state: backtrack records plus saved capture slots,
/// 16 bytes each, so 512 MiB. ECMA-262 bounds neither the time nor the space a match may take;
/// this only stops absurd growth (a quantifier repeated millions of times with an untried
/// alternative left at every iteration), which becomes a RangeError instead of exhausting memory.
pub(crate) const MAX_BACKTRACK_STATE: usize = 32 << 20;

/// Backtracking state capacity a thread keeps between `exec` calls; a huge match's is released.
const RETAINED_BACKTRACK_STATE: usize = 1 << 12;

#[cfg(test)]
thread_local! {
    static BACKTRACK_STATE_LIMIT: Cell<usize> = const { Cell::new(MAX_BACKTRACK_STATE) };
}

/// Lower this thread's backtracking state ceiling so tests can reach it cheaply.
#[cfg(test)]
pub(crate) fn set_backtrack_state_limit(limit: usize) {
    BACKTRACK_STATE_LIMIT.with(|cell| cell.set(limit));
}

#[inline]
fn backtrack_state_limit() -> usize {
    #[cfg(test)]
    {
        BACKTRACK_STATE_LIMIT.with(Cell::get)
    }
    #[cfg(not(test))]
    {
        MAX_BACKTRACK_STATE
    }
}

/// An active lookaround. Its body runs on the same backtrack stack above a `LookFail` record:
/// reaching the body's `Match` completes it with success (the body's choice points are
/// discarded, lookarounds being atomic), and backtracking down to `LookFail` completes it with
/// failure.
struct LookFrame<'a> {
    /// The program containing the lookaround instruction, and its pc.
    prog: &'a [Inst],
    pc: usize,
    /// Subject position the body started from.
    pos: usize,
    negate: bool,
    /// Matching direction of the enclosing program.
    back: bool,
    /// Index of this frame's `LookFail` record.
    base: usize,
    /// Choice records below the body, and the enclosing program's undo floor.
    choices: u32,
    floor: u32,
    /// Start of the capture snapshot taken on entry, in `saved`.
    saved: usize,
    /// Depth of the flag stack on entry.
    flags: usize,
}

struct Matcher<'a, I: ReInput> {
    input: I,
    caps: Vec<Option<usize>>,
    marks: Vec<Option<usize>>,
    /// Executed instructions, counted only to poll for host interruption: ECMA-262 bounds no
    /// matcher's running time, so a long match runs until it ends or the host stops the script.
    steps: u64,
    /// The backtrack stack: no native recursion per choice point, so neither a long subject nor
    /// a deeply nested pattern consumes native stack.
    bt: Vec<Backtrack>,
    /// Capture snapshots referenced by `Backtrack::Captures` records and lookaround frames.
    saved: Vec<Option<usize>>,
    /// Choice records currently on `bt`.
    choices: u32,
    /// Choice count at entry to the running (sub)program. While `choices` equals it, a failure
    /// unwinds the whole program, whose caller resets the state itself, so state changes need no
    /// undo records.
    floor: u32,
    /// Active lookarounds, innermost last.
    looks: Vec<LookFrame<'a>>,
    /// Ceiling on `bt.len() + saved.len()` (see [`MAX_BACKTRACK_STATE`]).
    state_limit: usize,
    /// Matching direction: a lookbehind body (compiled from the reversed AST) consumes leftward.
    back: bool,
    /// `(icase, multiline, dotall)` stack — the base flags, plus an entry per active `(?ims-ims:…)`
    /// inline-modifier group. Reads use the top; the group's Push/Pop instructions undo on backtrack.
    flags: Vec<(bool, bool, bool)>,
    /// Unicode mode (`u`/`v`): case-insensitive matching uses full case folding instead of the
    /// legacy Canonicalize (simple uppercase, never folding non-ASCII to ASCII).
    unicode: bool,
    control: &'a crate::RuntimeInterrupt,
    abort: Option<MatchError>,
    string_failures: Option<std::collections::HashSet<(usize, usize, usize, usize, bool, u8)>>,
    /// Opt-in instruction/backtrack profiling gate (see [`REGEXP_PROF`]).
    prof: bool,
}

/// Failed string-set states one match records in its failure memo. The memo only saves repeated
/// work, so further failures past this many go unrecorded rather than growing it without bound.
const MAX_STRING_FAILURES: usize = 1 << 20;

/// How far ahead [`Matcher::continuation_viable`] may look before giving up on a proof.
const VIABILITY_BUDGET: usize = 16;

impl<'a, I: ReInput> Matcher<'a, I> {
    #[inline(always)]
    fn tick(&mut self) -> bool {
        if self.abort.is_some() {
            return false;
        }
        self.steps += 1;
        if self.steps as usize & INTERRUPT_POLL_MASK == 0 {
            return self.checkpoint();
        }
        true
    }

    /// The periodic slow path of [`Self::tick`]: host interruption and the state ceiling.
    #[cold]
    fn checkpoint(&mut self) -> bool {
        if let Err(error) = poll_interrupt(self.control) {
            self.abort = Some(error);
            return false;
        }
        if self.bt.len() + self.saved.len() > self.state_limit {
            self.abort = Some(MatchError::ResourceExhausted);
            return false;
        }
        true
    }

    #[inline]
    fn poll_linear(&mut self, index: usize) -> bool {
        if index == 0 || index & INTERRUPT_POLL_MASK != 0 {
            return self.abort.is_none();
        }
        if let Err(error) = poll_interrupt(self.control) {
            self.abort = Some(error);
            return false;
        }
        true
    }

    fn input_ranges_eq(&mut self, left: usize, right: usize, len: usize) -> bool {
        for index in 0..len {
            if !self.poll_linear(index)
                || !self.eqc_uu(self.input.at(left + index), self.input.at(right + index))
            {
                return false;
            }
        }
        true
    }
    #[inline(always)]
    fn icase(&self) -> bool {
        self.flags.last().unwrap().0
    }
    #[inline(always)]
    fn multiline(&self) -> bool {
        self.flags.last().unwrap().1
    }
    #[inline(always)]
    fn dotall(&self) -> bool {
        self.flags.last().unwrap().2
    }
    /// Compare two subject/pattern code points under the active case rules.
    #[inline(always)]
    fn eqc_uu(&self, a: u32, b: u32) -> bool {
        if a == b {
            return true;
        }
        if self.icase() {
            let (ca, cb) = match (char::from_u32(a), char::from_u32(b)) {
                (Some(x), Some(y)) => (x, y),
                _ => return false, // lone surrogates have no case
            };
            if self.unicode {
                // Full case folding via the generated orbit table (ſ≡s, ΐ≡ΐ, K≡k, ...).
                return fold_canon(ca as u32) == fold_canon(cb as u32);
            }
            return canonicalize_legacy(ca) == canonicalize_legacy(cb);
        }
        false
    }

    /// The next element to consume and the position after it, honouring the match direction.
    #[inline(always)]
    fn step(&self, pos: usize) -> Option<(u32, usize)> {
        if self.back {
            if pos > 0 {
                Some((self.input.at(pos - 1), pos - 1))
            } else {
                None
            }
        } else if pos < self.input.len() {
            Some((self.input.at(pos), pos + 1))
        } else {
            None
        }
    }

    /// The position `n` elements from `pos` in the match direction.
    #[inline(always)]
    fn advance(&self, pos: usize, n: usize) -> usize {
        if self.back {
            pos - n
        } else {
            pos + n
        }
    }

    /// Elements left to consume from `pos` in the match direction.
    #[inline(always)]
    fn room(&self, pos: usize) -> usize {
        if self.back {
            pos
        } else {
            self.input.len() - pos
        }
    }

    /// The `k`-th element from `pos` in the match direction.
    #[inline(always)]
    fn element(&self, pos: usize, k: usize) -> u32 {
        if self.back {
            self.input.at(pos - 1 - k)
        } else {
            self.input.at(pos + k)
        }
    }

    /// How many consecutive elements from `pos` in the match direction, at most `limit`, `rep`
    /// matches; `None` if the host interrupted the scan.
    #[inline]
    fn run_length(&mut self, rep: &Rep, pos: usize, limit: usize) -> Option<usize> {
        let back = self.back;
        let mut n = 0;
        while n < limit {
            let element = if back {
                self.input.at(pos - 1 - n)
            } else {
                self.input.at(pos + n)
            };
            if !self.rep_matches(rep, element) {
                break;
            }
            n += 1;
            if !self.poll_linear(n) {
                return None;
            }
        }
        Some(n)
    }

    #[inline(always)]
    fn rep_matches(&self, rep: &Rep, c: u32) -> bool {
        match rep {
            Rep::Char(ch) => self.eqc_uu(c, *ch),
            Rep::Any => self.dotall() || !is_line_terminator_u32(c),
            Rep::Class(cc) => cc.matches(c, self.icase(), self.unicode),
        }
    }

    #[inline(always)]
    fn push_choice(&mut self, record: Backtrack) {
        self.choices += 1;
        self.bt.push(record);
    }

    /// Record how to undo a state change, unless no choice point above the floor could resume
    /// with the old state (see [`Matcher::floor`]).
    #[inline(always)]
    fn push_undo(&mut self, record: Backtrack) {
        if self.choices != self.floor {
            self.bt.push(record);
        }
    }

    /// Whether the branch at `pc` may match at `pos`, judged by its first instruction past
    /// capture and mark bookkeeping; `false` proves it fails before consuming anything. Unlike
    /// [`Self::continuation_viable`] this never looks through another branch point, so it is cheap
    /// enough for every alternation the matcher meets.
    #[inline]
    fn branch_may_match(&self, prog: &[Inst], mut pc: usize, pos: usize) -> bool {
        for _ in 0..4 {
            match &prog[pc] {
                Inst::Char(expected) => {
                    return self
                        .step(pos)
                        .is_some_and(|(found, _)| self.eqc_uu(found, *expected));
                }
                Inst::Class(class) => {
                    return self.step(pos).is_some_and(|(found, _)| {
                        class.matches(found, self.icase(), self.unicode)
                    });
                }
                Inst::Any => {
                    return self
                        .step(pos)
                        .is_some_and(|(found, _)| self.dotall() || !is_line_terminator_u32(found));
                }
                Inst::Many { rep, min, .. } if *min > 0 => {
                    return self
                        .step(pos)
                        .is_some_and(|(found, _)| self.rep_matches(rep, found));
                }
                Inst::Save(_)
                | Inst::ClearCaps(..)
                | Inst::SetMark(_)
                | Inst::CounterReset(_)
                | Inst::CounterIncr(_) => pc += 1,
                Inst::Jmp(target) => pc = *target,
                _ => return true,
            }
        }
        true
    }

    /// Take the first of two branches at `pos`, leaving the second to backtrack into. A branch
    /// that provably fails at once is skipped: it could only fail again, and skipping it keeps
    /// matches that never backtrack free of backtrack records. Returns the pc to continue at.
    #[inline]
    fn branch(&mut self, prog: &[Inst], first: usize, second: usize, pos: usize) -> usize {
        if !self.branch_may_match(prog, first, pos) {
            return second;
        }
        if self.branch_may_match(prog, second, pos) {
            self.push_choice(Backtrack::Alt {
                pc: second as u32,
                pos,
            });
        }
        first
    }

    /// Restore capture slots from `2 * lo` on from the snapshot at `saved[at..]`, dropping it.
    #[inline]
    fn restore_captures(&mut self, lo: usize, at: usize) {
        let len = self.saved.len() - at;
        self.caps[2 * lo..2 * lo + len].copy_from_slice(&self.saved[at..]);
        self.saved.truncate(at);
    }

    /// Key of the failure memo for a string-set state of the current program and flags.
    fn memo_key(
        &self,
        prog: &[Inst],
        pc: usize,
        pos: usize,
        count: usize,
    ) -> (usize, usize, usize, usize, bool, u8) {
        let flags =
            (self.icase() as u8) | ((self.multiline() as u8) << 1) | ((self.dotall() as u8) << 2);
        (prog.as_ptr() as usize, pc, pos, count, self.back, flags)
    }

    fn memo_failed(&self, key: &(usize, usize, usize, usize, bool, u8)) -> bool {
        self.string_failures
            .as_ref()
            .is_some_and(|failures| failures.contains(key))
    }

    /// Return every matching UnicodeSets string length in the normative order: multi-code-point
    /// elements by descending length, then a singleton, then the empty element. The exact-mode
    /// path follows one trie edge per subject code point; ignore-case mode retains the small set
    /// of canonically equivalent branches.
    fn string_set_lengths(&self, set: &StringSet, pos: usize) -> Vec<usize> {
        let trie = if self.back {
            &set.backward
        } else {
            &set.forward
        };
        let mut lengths = Vec::new();
        if self.icase() {
            let mut nodes = vec![0usize];
            let mut cursor = pos;
            for depth in 1..=trie.max_depth {
                let Some((found, next)) = self.step(cursor) else {
                    break;
                };
                let mut next_nodes = Vec::new();
                for node in nodes {
                    for &(expected, child) in &trie.nodes[node].edges {
                        if self.eqc_uu(found, expected) && !next_nodes.contains(&child) {
                            next_nodes.push(child);
                        }
                    }
                }
                if next_nodes.is_empty() {
                    break;
                }
                if next_nodes.iter().any(|node| trie.nodes[*node].terminal) {
                    lengths.push(depth);
                }
                nodes = next_nodes;
                cursor = next;
            }
        } else {
            let mut node = 0usize;
            let mut cursor = pos;
            for depth in 1..=trie.max_depth {
                let Some((found, next)) = self.step(cursor) else {
                    break;
                };
                let Ok(edge) = trie.nodes[node]
                    .edges
                    .binary_search_by_key(&found, |edge| edge.0)
                else {
                    break;
                };
                node = trie.nodes[node].edges[edge].1;
                if trie.nodes[node].terminal {
                    lengths.push(depth);
                }
                cursor = next;
            }
        }
        lengths.reverse();
        if self
            .step(pos)
            .is_some_and(|(found, _)| set.singles.matches(found, self.icase(), self.unicode))
        {
            lengths.push(1);
        }
        if set.empty {
            lengths.push(0);
        }
        lengths
    }

    /// RepeatMatcher for a variable-length UnicodeSets atom (the `StringSetRepeat` at `pc`),
    /// entered at `pos` after `count` iterations. Each state's alternatives are tried in the
    /// order of ECMA-262's greedy or lazy RepeatMatcher: the first is taken here, the rest become
    /// backtrack records. Capture-free failures share the failure memo with a standalone
    /// `StringSet`. Returns the position the continuation starts from, or `None` on failure.
    fn string_set_repeat(
        &mut self,
        prog: &[Inst],
        pc: usize,
        mut pos: usize,
        mut count: usize,
    ) -> Option<usize> {
        let Inst::StringSetRepeat {
            set,
            min,
            max,
            greedy,
        } = &prog[pc]
        else {
            unreachable!("string-set repetition state outside its instruction");
        };
        let (min, max, greedy) = (*min, *max, *greedy);
        loop {
            if !self.tick() {
                return None;
            }
            // Beyond `min` every further iteration must consume an element, so once `max` is
            // out of reach of the remaining input the state no longer depends on the count.
            let memo_count =
                if count >= min && max.is_none_or(|limit| limit - count >= self.room(pos)) {
                    min
                } else {
                    count
                };
            let key = self.memo_key(prog, pc, pos, memo_count);
            if self.memo_failed(&key) {
                return None;
            }
            let continuation = count >= min;
            let mut lengths = if max.is_none_or(|limit| count < limit) {
                self.string_set_lengths(set, pos)
            } else {
                Vec::new()
            };
            // RepeatMatcher rejects a further empty iteration once min is satisfied, allowing the
            // atom to try any later candidate before the repeat exits.
            if continuation {
                lengths.retain(|&length| length != 0);
            }
            let next_count = if lengths.is_empty() {
                count
            } else {
                let Some(next) = count.checked_add(1) else {
                    self.abort = Some(MatchError::ResourceExhausted);
                    return None;
                };
                next
            };
            if self.string_failures.is_some() {
                self.bt.push(Backtrack::Extra(memo_count));
                self.push_choice(Backtrack::SetFailure { pc: pc as u32, pos });
            }
            // Alternative `i` in trial order: `Some(length)` iterates, `None` continues.
            let total = lengths.len() + continuation as usize;
            let alternative = |i: usize| -> Option<usize> {
                if greedy {
                    lengths.get(i).copied()
                } else if continuation {
                    i.checked_sub(1).map(|i| lengths[i])
                } else {
                    Some(lengths[i])
                }
            };
            for i in (1..total).rev() {
                match alternative(i) {
                    Some(length) => {
                        let next = self.advance(pos, length);
                        self.bt.push(Backtrack::Extra(next_count));
                        self.push_choice(Backtrack::SetIteration {
                            pc: pc as u32,
                            pos: next,
                        });
                    }
                    None => self.push_choice(Backtrack::Alt {
                        pc: pc as u32 + 1,
                        pos,
                    }),
                }
            }
            if total == 0 {
                return None;
            }
            match alternative(0) {
                Some(length) => {
                    pos = self.advance(pos, length);
                    count = next_count;
                }
                None => return Some(pos),
            }
        }
    }

    /// Conservative viability test for the continuation at `pc` and `pos`.
    ///
    /// `Some(false)` proves that its first consuming instruction cannot match here; `Some(true)`
    /// is only a possible match, and `None` means stateful bytecode prevented a proof. The walk
    /// shares one `budget` of instructions across both arms of every split it meets. Quantifiers
    /// and alternatives use this to skip impossible retry positions without changing match order.
    fn continuation_viable(
        &self,
        prog: &[Inst],
        mut pc: usize,
        pos: usize,
        budget: &mut usize,
        mut tainted_groups: u128,
    ) -> Option<bool> {
        while *budget > 0 {
            *budget -= 1;
            match &prog[pc] {
                Inst::Char(expected) => {
                    return Some(
                        self.step(pos)
                            .is_some_and(|(found, _)| self.eqc_uu(found, *expected)),
                    );
                }
                Inst::Any => {
                    return Some(self.step(pos).is_some_and(|(found, _)| {
                        self.dotall() || !is_line_terminator_u32(found)
                    }));
                }
                Inst::Class(class) => {
                    return Some(self.step(pos).is_some_and(|(found, _)| {
                        class.matches(found, self.icase(), self.unicode)
                    }));
                }
                // `first` holds the elements a string can begin with, which a backward match
                // consumes last.
                Inst::StringSet(set) => {
                    if set.empty || self.back {
                        return None;
                    }
                    return Some(self.step(pos).is_some_and(|(found, _)| {
                        set.first.matches(found, self.icase(), self.unicode)
                    }));
                }
                Inst::StringSetRepeat { set, min, .. } => {
                    if *min == 0 || set.empty || self.back {
                        return None;
                    }
                    return Some(self.step(pos).is_some_and(|(found, _)| {
                        set.first.matches(found, self.icase(), self.unicode)
                    }));
                }
                Inst::Many { rep, min, .. } => {
                    let matches_here = self
                        .step(pos)
                        .is_some_and(|(found, _)| self.rep_matches(rep, found));
                    if *min > 0 || matches_here {
                        return Some(matches_here);
                    }
                    pc += 1;
                }
                Inst::Backref(group) => {
                    let group = *group;
                    if group >= 128 || tainted_groups & (1u128 << group) != 0 {
                        return None;
                    }
                    if group == 0 || 2 * group + 1 >= self.caps.len() {
                        pc += 1;
                        continue;
                    }
                    match (self.caps[2 * group], self.caps[2 * group + 1]) {
                        (Some(a), Some(b)) if a != b => {
                            // A backward backreference consumes the capture from its end.
                            let first = if self.back {
                                self.input.at(a.max(b) - 1)
                            } else {
                                self.input.at(a.min(b))
                            };
                            return Some(
                                self.step(pos)
                                    .is_some_and(|(found, _)| self.eqc_uu(found, first)),
                            );
                        }
                        _ => {
                            pc += 1; // an empty or unset backreference consumes nothing
                        }
                    }
                }
                Inst::BackrefAlt(groups) => {
                    if groups
                        .iter()
                        .any(|&group| group >= 128 || tainted_groups & (1u128 << group) != 0)
                    {
                        return None;
                    }
                    let captured = groups.iter().copied().find_map(|group| {
                        match (self.caps[2 * group], self.caps[2 * group + 1]) {
                            (Some(a), Some(b)) => Some((a.min(b), a.max(b))),
                            _ => None,
                        }
                    });
                    match captured {
                        Some((a, b)) if a != b => {
                            let first = if self.back {
                                self.input.at(b - 1)
                            } else {
                                self.input.at(a)
                            };
                            return Some(
                                self.step(pos)
                                    .is_some_and(|(found, _)| self.eqc_uu(found, first)),
                            );
                        }
                        _ => pc += 1,
                    }
                }
                Inst::AssertStart => {
                    if pos != 0
                        && !(self.multiline() && is_line_terminator_u32(self.input.at(pos - 1)))
                    {
                        return Some(false);
                    }
                    pc += 1;
                }
                Inst::AssertEnd => {
                    if pos != self.input.len()
                        && !(self.multiline() && is_line_terminator_u32(self.input.at(pos)))
                    {
                        return Some(false);
                    }
                    pc += 1;
                }
                Inst::WordBoundary(want) => {
                    let before =
                        pos > 0 && is_word_ic(self.input.at(pos - 1), self.icase(), self.unicode);
                    let after = pos < self.input.len()
                        && is_word_ic(self.input.at(pos), self.icase(), self.unicode);
                    if (before != after) != *want {
                        return Some(false);
                    }
                    pc += 1;
                }
                Inst::Jmp(target) => pc = *target,
                Inst::Split(a, b) => {
                    let left = self.continuation_viable(prog, *a, pos, budget, tainted_groups);
                    if left == Some(true) {
                        return left;
                    }
                    let right = self.continuation_viable(prog, *b, pos, budget, tainted_groups);
                    return match (left, right) {
                        (Some(false), Some(false)) => Some(false),
                        (_, Some(true)) => Some(true),
                        _ => None,
                    };
                }
                Inst::Match => return Some(true),
                // Capture writes do not themselves consume input. Keep walking, but remember
                // which groups have changed so a following backreference never consults stale
                // state. This is especially valuable for `.*?\k` continuations: the compiler's
                // group-end Save no longer hides the backreference's first-character filter.
                Inst::Save(slot) => {
                    let group = *slot / 2;
                    if group >= 128 {
                        return None;
                    }
                    tainted_groups |= 1u128 << group;
                    pc += 1;
                }
                Inst::ClearCaps(lo, hi) => {
                    if *hi >= 128 {
                        return None;
                    }
                    for group in *lo..=*hi {
                        tainted_groups |= 1u128 << group;
                    }
                    pc += 1;
                }
                Inst::SetMark(_) | Inst::CounterReset(_) | Inst::CounterIncr(_) => pc += 1,
                // These affect the next predicate or can branch on mutable state.
                Inst::Look { .. }
                | Inst::LookBehind { .. }
                | Inst::PushFlags(..)
                | Inst::PopFlags
                | Inst::CheckProgress(_)
                | Inst::CheckCountedProgress { .. }
                | Inst::CountedBranch { .. } => return None,
            }
        }
        None
    }

    /// The largest count in `min..n` a greedy `Many` at `pc` (entered at `start`) can retry
    /// with: the next one whose continuation's first instruction may match.
    fn greedy_retry(
        &mut self,
        prog: &[Inst],
        pc: usize,
        start: usize,
        mut n: usize,
        min: usize,
    ) -> Option<usize> {
        let mut tried = 0usize;
        while n > min {
            n -= 1;
            tried += 1;
            if !self.poll_linear(tried) {
                return None;
            }
            if self.branch_may_match(prog, pc + 1, self.advance(start, n)) {
                return Some(n);
            }
        }
        None
    }

    /// The count a lazy `Many` at `pc` (entered at `start`) tries next: `n` itself when
    /// `include`, otherwise the smallest larger count, skipping counts whose continuation cannot
    /// match. The run is extended one element at a time, so a lazy quantifier never scans
    /// further ahead than the match needs.
    fn lazy_next(
        &mut self,
        prog: &[Inst],
        pc: usize,
        start: usize,
        mut n: usize,
        include: bool,
    ) -> Option<usize> {
        let Inst::Many { rep, max, .. } = &prog[pc] else {
            unreachable!("lazy repetition state outside its instruction");
        };
        let limit = max.unwrap_or(usize::MAX).min(self.room(start));
        let mut extend = !include;
        loop {
            if extend {
                if n >= limit || !self.rep_matches(rep, self.element(start, n)) {
                    return None;
                }
                n += 1;
                if !self.poll_linear(n) {
                    return None;
                }
            }
            extend = true;
            let mut budget = VIABILITY_BUDGET;
            let end = self.advance(start, n);
            if self.continuation_viable(prog, pc + 1, end, &mut budget, 0) != Some(false) {
                return Some(n);
            }
        }
    }

    /// Match `root` from `pos`. On success the capture slots hold the match. Backtracking state
    /// lives in `bt` (see [`Backtrack`]) and lookarounds in `looks`, so the native stack depth
    /// is constant whatever the subject or pattern.
    fn run(&mut self, root: &'a [Inst], pos: usize) -> bool {
        let mut prog = root;
        let mut pc = 0usize;
        let mut pos = pos;
        'exec: loop {
            // Run forward until the program matches or this path fails.
            loop {
                if !self.tick() {
                    return false;
                }
                if self.prof {
                    REGEXP_PROF.with(|cell| {
                        let mut prof = cell.borrow_mut();
                        prof.inst[regexp_prof_kind(&prog[pc])] += 1;
                    });
                }
                match &prog[pc] {
                    Inst::Match => {
                        let Some(frame) = self.looks.pop() else {
                            return true;
                        };
                        // The lookaround body matched. Lookarounds are atomic: drop the body's
                        // choice points and undo records along with the frame's `LookFail`.
                        self.bt.truncate(frame.base);
                        self.choices = frame.choices;
                        self.floor = frame.floor;
                        self.back = frame.back;
                        self.flags.truncate(frame.flags);
                        self.saved.truncate(frame.saved + self.caps.len());
                        prog = frame.prog;
                        if frame.negate {
                            self.restore_captures(0, frame.saved);
                            break;
                        }
                        // Captures the body set persist, undone only if the enclosing program
                        // backtracks past the lookaround.
                        if self.choices != self.floor {
                            self.bt.push(Backtrack::Captures {
                                lo: 0,
                                at: frame.saved,
                            });
                        } else {
                            self.saved.truncate(frame.saved);
                        }
                        pc = frame.pc + 1;
                        pos = frame.pos;
                    }
                    Inst::Char(c) => match self.step(pos) {
                        Some((e, next)) if self.eqc_uu(e, *c) => {
                            pc += 1;
                            pos = next;
                        }
                        _ => break,
                    },
                    Inst::Any => match self.step(pos) {
                        Some((e, next)) if self.dotall() || !is_line_terminator_u32(e) => {
                            pc += 1;
                            pos = next;
                        }
                        _ => break,
                    },
                    Inst::Class(cc) => match self.step(pos) {
                        Some((e, next)) if cc.matches(e, self.icase(), self.unicode) => {
                            pc += 1;
                            pos = next;
                        }
                        _ => break,
                    },
                    Inst::StringSet(set) => {
                        let key = self.memo_key(prog, pc, pos, usize::MAX);
                        if self.memo_failed(&key) {
                            break;
                        }
                        let lengths = self.string_set_lengths(set, pos);
                        if self.string_failures.is_some() {
                            self.bt.push(Backtrack::Extra(usize::MAX));
                            self.push_choice(Backtrack::SetFailure { pc: pc as u32, pos });
                        }
                        for &length in lengths.iter().skip(1).rev() {
                            let next = self.advance(pos, length);
                            self.push_choice(Backtrack::Alt {
                                pc: pc as u32 + 1,
                                pos: next,
                            });
                        }
                        let Some(&length) = lengths.first() else {
                            break;
                        };
                        pos = self.advance(pos, length);
                        pc += 1;
                    }
                    Inst::StringSetRepeat { .. } => {
                        match self.string_set_repeat(prog, pc, pos, 0) {
                            Some(next) => {
                                pc += 1;
                                pos = next;
                            }
                            None => break,
                        }
                    }
                    Inst::Save(slot) => {
                        let slot = *slot;
                        self.push_undo(Backtrack::Capture {
                            slot: slot as u32,
                            old: pack_slot(self.caps[slot]),
                        });
                        self.caps[slot] = Some(pos);
                        pc += 1;
                    }
                    Inst::Split(a, b) => pc = self.branch(prog, *a, *b, pos),
                    Inst::SetMark(id) => {
                        let id = *id;
                        self.push_undo(Backtrack::Mark {
                            id: id as u32,
                            old: pack_slot(self.marks[id]),
                        });
                        self.marks[id] = Some(pos);
                        pc += 1;
                    }
                    Inst::CheckProgress(id) => {
                        if self.marks[*id] == Some(pos) {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::CheckCountedProgress { mark, counter, min } => {
                        if self.marks[*counter].unwrap_or(0) >= *min
                            && self.marks[*mark] == Some(pos)
                        {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::CounterReset(id) | Inst::CounterIncr(id) => {
                        let id = *id;
                        let old = self.marks[id];
                        self.push_undo(Backtrack::Mark {
                            id: id as u32,
                            old: pack_slot(old),
                        });
                        self.marks[id] = Some(match &prog[pc] {
                            Inst::CounterReset(_) => 0,
                            _ => old.map_or(1, |count| count + 1),
                        });
                        pc += 1;
                    }
                    Inst::CountedBranch {
                        id,
                        min,
                        max,
                        greedy,
                        body,
                        exit,
                    } => {
                        let count = self.marks[*id].unwrap_or(0);
                        if count < *min {
                            pc = *body;
                        } else if max.is_some_and(|max| count >= max) {
                            pc = *exit;
                        } else {
                            let (first, second) = if *greedy {
                                (*body, *exit)
                            } else {
                                (*exit, *body)
                            };
                            pc = self.branch(prog, first, second, pos);
                        }
                    }
                    Inst::Many {
                        rep,
                        min,
                        max,
                        greedy,
                    } => {
                        let min = *min;
                        let limit = max.unwrap_or(usize::MAX).min(self.room(pos));
                        if *greedy {
                            // Consume as many as the input allows (up to `max`), iteratively;
                            // backtracking retries smaller counts (see `Backtrack::Greedy`).
                            let Some(avail) = self.run_length(rep, pos, limit) else {
                                return false;
                            };
                            if avail < min {
                                break;
                            }
                            let end = self.advance(pos, avail);
                            if avail > min {
                                self.bt.push(Backtrack::Extra(pos));
                                self.push_choice(Backtrack::Greedy { pc: pc as u32, end });
                            }
                            pc += 1;
                            pos = end;
                        } else {
                            if min > limit
                                || !(0..min).all(|k| self.rep_matches(rep, self.element(pos, k)))
                            {
                                break;
                            }
                            let Some(n) = self.lazy_next(prog, pc, pos, min, true) else {
                                break;
                            };
                            let end = self.advance(pos, n);
                            self.bt.push(Backtrack::Extra(pos));
                            self.push_choice(Backtrack::Lazy { pc: pc as u32, end });
                            pc += 1;
                            pos = end;
                        }
                    }
                    Inst::PushFlags(i, m, s) => {
                        let cur = *self.flags.last().unwrap();
                        self.flags.push((
                            i.unwrap_or(cur.0),
                            m.unwrap_or(cur.1),
                            s.unwrap_or(cur.2),
                        ));
                        self.push_undo(Backtrack::PopFlags);
                        pc += 1;
                    }
                    Inst::PopFlags => {
                        let (i, m, s) = self.flags.pop().unwrap();
                        self.push_undo(Backtrack::PushFlags(i, m, s));
                        pc += 1;
                    }
                    Inst::Jmp(t) => pc = *t,
                    Inst::AssertStart => {
                        let ok = pos == 0
                            || (self.multiline() && is_line_terminator_u32(self.input.at(pos - 1)));
                        if !ok {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::AssertEnd => {
                        let ok = pos == self.input.len()
                            || (self.multiline() && is_line_terminator_u32(self.input.at(pos)));
                        if !ok {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::WordBoundary(want) => {
                        let (icase, unicode) = (self.icase(), self.unicode);
                        let before = pos > 0 && is_word_ic(self.input.at(pos - 1), icase, unicode);
                        let after = pos < self.input.len()
                            && is_word_ic(self.input.at(pos), icase, unicode);
                        if (before != after) != *want {
                            break;
                        }
                        pc += 1;
                    }
                    Inst::Backref(g) => {
                        let g = *g;
                        if g == 0 || 2 * g + 1 >= self.caps.len() {
                            pc += 1; // invalid group: matches empty
                            continue;
                        }
                        match (self.caps[2 * g], self.caps[2 * g + 1]) {
                            (Some(a), Some(b)) => match self.backref_end(pos, a.min(b), a.max(b)) {
                                Some(next) => {
                                    pc += 1;
                                    pos = next;
                                }
                                None => break,
                            },
                            _ => pc += 1, // unset group matches empty
                        }
                    }
                    Inst::BackrefAlt(idxs) => {
                        // At most one same-named group can have captured; match through that one.
                        let g = idxs.iter().copied().find(|&g| {
                            2 * g + 1 < self.caps.len()
                                && self.caps[2 * g].is_some()
                                && self.caps[2 * g + 1].is_some()
                        });
                        match g {
                            None => pc += 1, // no group captured: matches empty
                            Some(g) => {
                                let (a, b) =
                                    (self.caps[2 * g].unwrap(), self.caps[2 * g + 1].unwrap());
                                match self.backref_end(pos, a.min(b), a.max(b)) {
                                    Some(next) => {
                                        pc += 1;
                                        pos = next;
                                    }
                                    None => break,
                                }
                            }
                        }
                    }
                    Inst::ClearCaps(lo, hi) => {
                        let slots = 2 * lo..2 * hi + 2;
                        if self.caps[slots.clone()].iter().any(Option::is_some) {
                            if self.choices != self.floor {
                                let at = self.saved.len();
                                self.saved.extend_from_slice(&self.caps[slots.clone()]);
                                self.bt.push(Backtrack::Captures { lo: *lo as u32, at });
                            }
                            self.caps[slots].fill(None);
                        }
                        pc += 1;
                    }
                    Inst::Look { negate, prog: sub } | Inst::LookBehind { negate, prog: sub } => {
                        // The body runs above a `LookFail` record with its own undo floor. A
                        // lookbehind body (compiled from the reversed AST) matches right-to-left
                        // from `pos`; a nested lookahead always matches forward.
                        let saved = self.saved.len();
                        self.saved.extend_from_slice(&self.caps);
                        self.looks.push(LookFrame {
                            prog,
                            pc,
                            pos,
                            negate: *negate,
                            back: self.back,
                            base: self.bt.len(),
                            choices: self.choices,
                            floor: self.floor,
                            saved,
                            flags: self.flags.len(),
                        });
                        self.bt.push(Backtrack::LookFail);
                        self.floor = self.choices;
                        self.back = matches!(prog[pc], Inst::LookBehind { .. });
                        prog = sub;
                        pc = 0;
                    }
                }
            }
            // Backtrack: undo state changes down to the most recent choice point and resume it.
            loop {
                let Some(record) = self.bt.pop() else {
                    return false;
                };
                match record {
                    Backtrack::Alt { pc: p, pos: q } => {
                        self.choices -= 1;
                        pc = p as usize;
                        pos = q;
                    }
                    Backtrack::Capture { slot, old } => {
                        self.caps[slot as usize] = unpack_slot(old);
                        continue;
                    }
                    Backtrack::Mark { id, old } => {
                        self.marks[id as usize] = unpack_slot(old);
                        continue;
                    }
                    Backtrack::PopFlags => {
                        self.flags.pop();
                        continue;
                    }
                    Backtrack::PushFlags(i, m, s) => {
                        self.flags.push((i, m, s));
                        continue;
                    }
                    Backtrack::Captures { lo, at } => {
                        self.restore_captures(lo as usize, at);
                        continue;
                    }
                    Backtrack::Extra(_) => {
                        unreachable!("backtrack operand without its choice record")
                    }
                    Backtrack::Greedy { pc: p, end } => {
                        self.choices -= 1;
                        let p = p as usize;
                        let Some(&Backtrack::Extra(start)) = self.bt.last() else {
                            unreachable!("greedy choice without its start position");
                        };
                        let Inst::Many { min, .. } = &prog[p] else {
                            unreachable!("greedy choice outside its instruction");
                        };
                        let n = if self.back { start - end } else { end - start };
                        match self.greedy_retry(prog, p, start, n, *min) {
                            Some(n) => {
                                let end = self.advance(start, n);
                                if n > *min {
                                    self.push_choice(Backtrack::Greedy { pc: p as u32, end });
                                } else {
                                    self.bt.pop();
                                }
                                pc = p + 1;
                                pos = end;
                            }
                            None => {
                                if self.abort.is_some() {
                                    return false;
                                }
                                self.bt.pop();
                                continue;
                            }
                        }
                    }
                    Backtrack::Lazy { pc: p, end } => {
                        self.choices -= 1;
                        let p = p as usize;
                        let Some(&Backtrack::Extra(start)) = self.bt.last() else {
                            unreachable!("lazy choice without its start position");
                        };
                        let n = if self.back { start - end } else { end - start };
                        match self.lazy_next(prog, p, start, n, false) {
                            Some(n) => {
                                let end = self.advance(start, n);
                                self.push_choice(Backtrack::Lazy { pc: p as u32, end });
                                pc = p + 1;
                                pos = end;
                            }
                            None => {
                                if self.abort.is_some() {
                                    return false;
                                }
                                self.bt.pop();
                                continue;
                            }
                        }
                    }
                    Backtrack::SetIteration { pc: p, pos: q } => {
                        self.choices -= 1;
                        let Some(Backtrack::Extra(count)) = self.bt.pop() else {
                            unreachable!("string-set iteration without its count");
                        };
                        let p = p as usize;
                        match self.string_set_repeat(prog, p, q, count) {
                            Some(next) => {
                                pc = p + 1;
                                pos = next;
                            }
                            None if self.abort.is_some() => return false,
                            None => continue,
                        }
                    }
                    Backtrack::SetFailure { pc: p, pos: q } => {
                        self.choices -= 1;
                        let Some(Backtrack::Extra(count)) = self.bt.pop() else {
                            unreachable!("string-set failure without its count");
                        };
                        let key = self.memo_key(prog, p as usize, q, count);
                        if let Some(failures) = self
                            .string_failures
                            .as_mut()
                            .filter(|failures| failures.len() < MAX_STRING_FAILURES)
                        {
                            failures.insert(key);
                        }
                        continue;
                    }
                    Backtrack::LookFail => {
                        // The lookaround body failed: restore the state it was entered with.
                        let frame = self
                            .looks
                            .pop()
                            .expect("lookaround failure without a frame");
                        self.floor = frame.floor;
                        self.back = frame.back;
                        self.flags.truncate(frame.flags);
                        self.restore_captures(0, frame.saved);
                        prog = frame.prog;
                        if !frame.negate {
                            continue;
                        }
                        pc = frame.pc + 1;
                        pos = frame.pos;
                    }
                }
                if self.prof {
                    REGEXP_PROF.with(|p| p.borrow_mut().backtrack_entries += 1);
                }
                continue 'exec;
            }
        }
    }

    /// Match a backreference to the captured `input[a..b]` at `pos`, returning the position after
    /// it (a backward backreference ends at `pos`).
    #[inline]
    fn backref_end(&mut self, pos: usize, a: usize, b: usize) -> Option<usize> {
        let n = b - a;
        let start = if self.back {
            pos.checked_sub(n)?
        } else {
            if pos + n > self.input.len() {
                return None;
            }
            pos
        };
        self.input_ranges_eq(start, a, n)
            .then(|| self.advance(pos, n))
    }
}

// ---------------------------------------------------------------------------------------------
// `v`-flag (unicodeSets) character classes: ClassSetExpressions are evaluated at parse time into
// a concrete set of code-point ranges plus a set of multi-code-point strings.
// ---------------------------------------------------------------------------------------------

/// A `v`-mode class set: sorted, disjoint code-point ranges plus multi-code-point strings.
#[derive(Default, Clone)]
struct ClassSet {
    ranges: Vec<(u32, u32)>,
    strings: Vec<Vec<char>>,
}

impl ClassSet {
    fn normalize(&mut self) {
        self.ranges.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(self.ranges.len());
        for &(lo, hi) in &self.ranges {
            if let Some(last) = out.last_mut() {
                if lo <= last.1.saturating_add(1) {
                    last.1 = last.1.max(hi);
                    continue;
                }
            }
            out.push((lo, hi));
        }
        self.ranges = out;
        self.strings.sort();
        self.strings.dedup();
    }

    fn union(mut self, other: ClassSet) -> ClassSet {
        self.ranges.extend(other.ranges);
        self.strings.extend(other.strings);
        self.normalize();
        self
    }

    fn intersect(mut self, other: ClassSet) -> ClassSet {
        let mut ranges = Vec::new();
        for &(a, b) in &self.ranges {
            for &(c, d) in &other.ranges {
                let lo = a.max(c);
                let hi = b.min(d);
                if lo <= hi {
                    ranges.push((lo, hi));
                }
            }
        }
        self.strings.retain(|s| other.strings.contains(s));
        self.ranges = ranges;
        self.normalize();
        self
    }

    fn subtract(mut self, other: ClassSet) -> ClassSet {
        let mut ranges = self.ranges.clone();
        for &(c, d) in &other.ranges {
            let mut next = Vec::with_capacity(ranges.len() + 1);
            for &(a, b) in &ranges {
                if d < a || c > b {
                    next.push((a, b));
                    continue;
                }
                if a < c {
                    next.push((a, c - 1));
                }
                if b > d {
                    next.push((d + 1, b));
                }
            }
            ranges = next;
        }
        self.strings.retain(|s| !other.strings.contains(s));
        self.ranges = ranges;
        self.normalize();
        self
    }

    /// Complement over the full code-point space. A set containing strings may not be negated.
    fn complement(mut self) -> Result<ClassSet, String> {
        if !self.strings.is_empty() {
            return Err("cannot negate a class set containing strings".into());
        }
        self.normalize();
        let mut out = Vec::new();
        let mut next = 0u32;
        for &(lo, hi) in &self.ranges {
            if lo > next {
                out.push((next, lo - 1));
            }
            next = hi.saturating_add(1);
        }
        if next <= 0x10FFFF {
            out.push((next, 0x10FFFF));
        }
        self.ranges = out;
        Ok(self)
    }

    fn from_cp(c: u32) -> ClassSet {
        ClassSet {
            ranges: vec![(c, c)],
            strings: Vec::new(),
        }
    }
}

/// The concrete ranges of a `\d`/`\w`/`\s` class escape (for `v`-mode set arithmetic).
fn builtin_class_set(b: char) -> ClassSet {
    let base = match b.to_ascii_lowercase() {
        'd' => vec![(0x30, 0x39)],
        'w' => vec![(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)],
        's' => {
            let mut r = vec![
                (0x09, 0x0D),
                (0x20, 0x20),
                (0x85, 0x85),
                (0xA0, 0xA0),
                (0x1680, 0x1680),
                (0x2000, 0x200A),
                (0x2028, 0x2029),
                (0x202F, 0x202F),
                (0x205F, 0x205F),
                (0x3000, 0x3000),
                (0xFEFF, 0xFEFF),
            ];
            r.sort_unstable();
            r
        }
        _ => Vec::new(),
    };
    let mut set = ClassSet {
        ranges: base,
        strings: Vec::new(),
    };
    if b.is_ascii_uppercase() {
        set = set.complement().unwrap();
    }
    set
}

/// The Unicode "properties of strings" (ECMA-262 table-binary-unicode-properties-of-strings): the
/// UTS #51 definitions derivable from the bundled emoji binary-property tables, and the curated
/// RGI_* sequence lists from `crate::regex_emoji`.
fn property_of_strings(name: &str) -> Option<ClassSet> {
    let ranges_of = |prop: &str| -> Vec<(u32, u32)> {
        crate::unicode_props::lookup(prop, None)
            .map(|r| r.to_vec())
            .unwrap_or_default()
    };
    match name {
        "Basic_Emoji" => {
            // Emoji_Presentation singletons, plus (Emoji minus Emoji_Presentation) + FE0F.
            let ep = ClassSet {
                ranges: ranges_of("Emoji_Presentation"),
                strings: Vec::new(),
            };
            let emoji = ClassSet {
                ranges: ranges_of("Emoji"),
                strings: Vec::new(),
            };
            let text_only = emoji.subtract(ep.clone());
            let mut strings = Vec::new();
            for &(lo, hi) in &text_only.ranges {
                for u in lo..=hi {
                    if let Some(c) = char::from_u32(u) {
                        strings.push(vec![c, '\u{FE0F}']);
                    }
                }
            }
            let mut set = ep;
            set.strings = strings;
            set.normalize();
            Some(set)
        }
        "Emoji_Keycap_Sequence" => {
            let mut strings = Vec::new();
            for c in "#*0123456789".chars() {
                strings.push(vec![c, '\u{FE0F}', '\u{20E3}']);
            }
            Some(ClassSet {
                ranges: Vec::new(),
                strings,
            })
        }
        "RGI_Emoji_Modifier_Sequence" => {
            let bases = ranges_of("Emoji_Modifier_Base");
            let mut strings = Vec::new();
            for &(lo, hi) in &bases {
                for u in lo..=hi {
                    if let Some(c) = char::from_u32(u) {
                        for m in 0x1F3FB..=0x1F3FF {
                            strings.push(vec![c, char::from_u32(m).unwrap()]);
                        }
                    }
                }
            }
            Some(ClassSet {
                ranges: Vec::new(),
                strings,
            })
        }
        "RGI_Emoji_Flag_Sequence" => Some(ClassSet {
            ranges: Vec::new(),
            strings: crate::regex_emoji::RGI_FLAG_SEQUENCES
                .iter()
                .map(|s| s.chars().collect())
                .collect(),
        }),
        "RGI_Emoji_ZWJ_Sequence" => Some(ClassSet {
            ranges: Vec::new(),
            strings: crate::regex_emoji::RGI_ZWJ_SEQUENCES
                .iter()
                .map(|s| s.chars().collect())
                .collect(),
        }),
        "RGI_Emoji" => {
            // The union table: single code points join the ranges, sequences the strings.
            let mut set = ClassSet {
                ranges: Vec::new(),
                strings: Vec::new(),
            };
            for s in crate::regex_emoji::RGI_EMOJI_ALL {
                let cs: Vec<char> = s.chars().collect();
                if cs.len() == 1 {
                    set.ranges.push((cs[0] as u32, cs[0] as u32));
                } else {
                    set.strings.push(cs);
                }
            }
            set.normalize();
            Some(set)
        }
        "RGI_Emoji_Tag_Sequence" => {
            // The three RGI tag sequences: england, scotland, wales.
            let mk = |tags: &str| {
                let mut v = vec!['\u{1F3F4}'];
                for c in tags.chars() {
                    v.push(char::from_u32(0xE0000 + c as u32).unwrap());
                }
                v.push('\u{E007F}');
                v
            };
            Some(ClassSet {
                ranges: Vec::new(),
                strings: vec![mk("gbeng"), mk("gbsct"), mk("gbwls")],
            })
        }
        _ => None,
    }
}

/// A `\q{...}` alternative: a single char joins the ranges; longer sequences join the strings.
fn push_q_alternative(set: &mut ClassSet, alt: Vec<char>) {
    match alt.len() {
        0 => set.strings.push(Vec::new()),
        1 => set.ranges.push((alt[0] as u32, alt[0] as u32)),
        _ => set.strings.push(alt),
    }
}

/// Compile a computed class set: an alternation of its strings (longest first, so the greedy
/// match prefers the longest sequence) plus a plain range class. Lone-surrogate ranges are
/// dropped (input is scalar values).
fn class_set_to_node(mut set: ClassSet) -> Node {
    set.normalize();
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for &(lo, hi) in &set.ranges {
        let mut push = |a: u32, b: u32| {
            if a <= b {
                ranges.push((a, b));
            }
        };
        if lo <= 0xD7FF && hi >= 0xE000 {
            push(lo, 0xD7FF);
            push(0xE000, hi);
        } else if !(0xD800..=0xDFFF).contains(&lo) || !(0xD800..=0xDFFF).contains(&hi) {
            push(lo.clamp(0, 0x10FFFF), hi.min(0x10FFFF));
        }
    }
    let class = CharClass {
        negate: false,
        ranges,
        builtins: Vec::new(),
        props: Vec::new(),
        ascii_lut: None,
    };
    if set.strings.is_empty() {
        return Node::Class(class);
    }
    let empty = set.strings.iter().any(Vec::is_empty);
    let strings: Vec<Vec<char>> = set
        .strings
        .into_iter()
        .filter(|string| string.len() > 1)
        .collect();
    if strings.is_empty() && !empty {
        return Node::Class(class);
    }
    Node::StringSet(Rc::new(StringSet::new(strings, class, empty)))
}

#[cfg(test)]
mod internal_engine_diagnostics {
    include!("regex_native_generated.rs");

    #[test]
    fn ascii_windows_of_wide_roots_match_the_reference_with_relative_boundaries() {
        use super::*;
        let body = format!("x\nA1.z\n{}z", "a".repeat(1024));
        let prefix = "😀é";
        let root: crate::lstr::LStr = format!("{prefix}{body}é").into();
        let source = root.slice_bytes(prefix.len(), prefix.len() + body.len());
        assert!(source.view_range().is_some());
        assert!(
            !source.ascii_hint(),
            "native string byte readers still decline views"
        );
        let control = crate::RuntimeInterrupt::default();
        let wide: Vec<u32> = body.bytes().map(u32::from).collect();
        for flags in ["", "g", "y", "i", "m", "u", "v", "iu", "duy"] {
            let unicode = flags.contains('u') || flags.contains('v');
            let parent = Rc::new(ReText::new_rc(unicode, &root));
            let start = crate::jstr::unit_len(prefix);
            let window =
                ReText::window(parent.clone(), &source, start, start + body.len()).unwrap();
            assert!(window.ascii_src.is_some());
            assert_eq!(window.subject_shape(), SUBJECT_ASCII);
            assert_eq!(
                Rc::strong_count(&parent),
                1,
                "ASCII needs no parent table owner"
            );
            assert_eq!(window.slice(0, window.len()), body);
            for pattern in [
                "^x",
                "^A\\d",
                "\\bA\\d\\b",
                "(?<=\\n)A\\d",
                "(?<=é)x",
                "(?<!é)x",
                "(?<n>A\\d)",
                "z$",
                "z(?=é)",
                "[a-z]+",
                "[^a]+",
            ] {
                let mut re = Regex::new(pattern, flags).unwrap();
                re.tier_up_at = 1;
                for at in [0, 1, 2, 6, body.len() - 1, body.len(), body.len() + 1] {
                    let actual = re
                        .exec_text_shared(&window, at, &control)
                        .unwrap()
                        .map(|captures| captures.to_vec());
                    let expected = re
                        .exec_impl(&wide[..], at, &control)
                        .unwrap()
                        .map(|captures| captures.to_vec());
                    assert_eq!(actual, expected, "/{pattern}/{flags}, start={at}");
                }
            }
        }
    }

    #[test]
    fn wide_windows_keep_unit_mapping_and_surrogate_boundary_guards() {
        use super::*;
        let root: crate::lstr::LStr = format!("😀{}éz", "a".repeat(1024)).into();
        let source = root.slice_bytes(4, root.len());
        for unicode in [false, true] {
            let parent = Rc::new(ReText::new_rc(unicode, &root));
            let end = crate::jstr::unit_len(&root);
            let window = ReText::window(parent.clone(), &source, 2, end).unwrap();
            assert!(window.ascii_src.is_none());
            assert!(window.parent.is_some());
            assert_eq!(window.unit_index(window.len()), end - 2);
            assert_eq!(window.slice(0, window.len()), source.as_str());
            if unicode {
                let empty = crate::lstr::LStr::from("");
                assert!(ReText::window(parent, &empty, 1, 1).is_none());
            }
        }
    }

    #[test]
    fn backtracking_exhaustion_is_distinct_from_no_match() {
        // Only the size of the backtracking state is bounded. Every iteration here leaves an
        // untried exit (the lookahead defeats the viability proof), so a lowered ceiling is
        // reached; that is an implementation limit, not a match failure.
        let re = super::Regex::new("(?:a|b)*(?=c)", "y").unwrap();
        let input = crate::lstr::LStr::from("ab".repeat(20_000));
        let text = super::ReText::new_rc(false, &input);
        let control = crate::RuntimeInterrupt::default();
        super::set_backtrack_state_limit(1 << 12);
        let limited = re.exec_text_shared(&text, 0, &control);
        super::set_backtrack_state_limit(super::MAX_BACKTRACK_STATE);
        assert!(matches!(limited, Err(super::MatchError::ResourceExhausted)));
        assert!(matches!(re.exec_text_shared(&text, 0, &control), Ok(None)));
    }

    #[test]
    fn deep_backtracking_runs_on_the_heap_stack() {
        // ECMA-262 bounds no matcher's depth. Choice points and lookarounds live on the heap,
        // so subjects far longer than any native recursion bound match exactly, including the
        // last iteration's captures, in either direction.
        let control = crate::RuntimeInterrupt::default();
        let quoted = format!("\"{}\"", "ab\\\"c".repeat(20_000));
        let cases: [(&str, &str, String, Vec<Option<(usize, usize)>>); 6] = [
            (
                "(a|b)*",
                "y",
                "ab".repeat(100_000),
                vec![Some((0, 200_000)), Some((199_999, 200_000))],
            ),
            (
                "\"(?:[^\"\\\\]|\\\\.)*\"",
                "",
                quoted.clone(),
                vec![Some((0, quoted.len()))],
            ),
            (
                "(?:(?=a)a|b)*c",
                "",
                "ab".repeat(50_000) + "c",
                vec![Some((0, 100_001))],
            ),
            (
                "z(?<=^(?:x|y)*z)",
                "",
                "xy".repeat(50_000) + "z",
                vec![Some((100_000, 100_001))],
            ),
            (
                "(?:ab){0,100000}?c",
                "",
                "ab".repeat(60_000) + "c",
                vec![Some((0, 120_001))],
            ),
            (
                "(a|ab)*?c",
                "",
                "ab".repeat(30_000) + "c",
                vec![Some((0, 60_001)), Some((59_998, 60_000))],
            ),
        ];
        for (pattern, flags, subject, expected) in cases {
            let re = super::Regex::new(pattern, flags).unwrap();
            let input = crate::lstr::LStr::from(subject.as_str());
            let text = super::ReText::new_rc(false, &input);
            let found = re
                .exec_text_shared(&text, 0, &control)
                .unwrap_or_else(|error| panic!("/{pattern}/{flags}: {error:?}"))
                .map(|captures| captures.to_vec());
            assert_eq!(found, Some(expected), "/{pattern}/{flags}");
        }
    }

    #[test]
    fn nested_counted_repetitions_compile_to_bounded_programs() {
        // Unrolling multiplies the program at every level of counted repetition. Past the unroll
        // budget the repetitions become counted loops over one copy of their body, so these stay
        // small instead of needing billions of instructions, and still match exactly.
        let control = crate::RuntimeInterrupt::default();
        let cases = [
            (
                "(?:(?:(?:ab){1000}){1000}){1000}".to_string(),
                "ab".repeat(1000),
                None,
            ),
            (
                "(?:".repeat(30) + "a|b" + &"){2}".repeat(30),
                "ab".repeat(100),
                None,
            ),
            (
                "(?:".repeat(10) + "a|b" + &"){2}".repeat(10),
                "ab".repeat(600),
                Some((0, 1024)),
            ),
            (
                "(?:x(?:ab){1001}y){3}".to_string(),
                format!("x{}y", "ab".repeat(1001)).repeat(3),
                Some((0, 6012)),
            ),
        ];
        for (pattern, subject, expected) in cases {
            let re = super::Regex::new(&pattern, "").unwrap();
            assert!(
                re.prog.len() < 2 * super::MAX_UNROLLED_LEN,
                "{} instructions",
                re.prog.len()
            );
            let input = crate::lstr::LStr::from(subject.as_str());
            let text = super::ReText::new_rc(false, &input);
            let found = re.exec_text_shared(&text, 0, &control).unwrap();
            assert_eq!(found.map(|captures| captures[0].unwrap()), expected);
        }
    }

    #[test]
    fn deeply_nested_patterns_compile_on_a_small_stack_or_fail_cleanly() {
        // The parser, the compiler and the other passes over the pattern tree recurse once per
        // nesting level. At the nesting limits every form compiles and matches on a small native
        // stack, which they grow on demand; one level deeper is a SyntaxError, never a stack
        // overflow.
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let control = crate::RuntimeInterrupt::default();
                let depth = super::MAX_PATTERN_NESTING;
                let looks = super::MAX_LOOKAROUND_NESTING;
                let named: String = (0..depth).map(|i| format!("(?<g{i}>")).collect();
                let cases = [
                    ("(".repeat(depth) + "a" + &")".repeat(depth), "", "(", ")"),
                    (
                        "(?:".repeat(depth) + "a" + &")*".repeat(depth),
                        "",
                        "(?:",
                        ")",
                    ),
                    (
                        "(?i:".repeat(depth) + "a" + &")".repeat(depth),
                        "",
                        "(?:",
                        ")",
                    ),
                    (named + "a" + &")".repeat(depth), "", "(?:", ")"),
                    ("[".repeat(depth) + "a" + &"]".repeat(depth), "v", "[", "]"),
                    (
                        "(?=".repeat(looks) + "a" + &")".repeat(looks),
                        "",
                        "(?!",
                        ")",
                    ),
                    (
                        "(?<=".repeat(looks) + "a" + &")".repeat(looks) + "b",
                        "",
                        "(?<!",
                        ")",
                    ),
                ];
                for (pattern, flags, open, close) in cases {
                    let re = super::Regex::new(&pattern, flags)
                        .unwrap_or_else(|error| panic!("{flags}: {error}"));
                    let input = crate::lstr::LStr::from("ab");
                    let text = super::ReText::new_rc(flags.contains('v'), &input);
                    assert!(
                        matches!(re.exec_text_shared(&text, 0, &control), Ok(Some(_))),
                        "{flags} {open}"
                    );
                    let deeper = format!("{open}{pattern}{close}");
                    assert_eq!(
                        super::Regex::new(&deeper, flags).err().as_deref(),
                        Some("regular expression is nested too deeply"),
                        "{flags} {open}"
                    );
                }
            })
            .expect("spawn small-stack thread")
            .join()
            .expect("nested patterns on a small stack");
    }

    #[test]
    fn long_backtracking_runs_to_its_ecmascript_result() {
        // ECMA-262 bounds no matcher's running time: exponential backtracking that used to stop
        // at a 2,000,000-step budget (a RangeError) now finds its result like other engines.
        let re = super::Regex::new("(a|aa)*b", "y").unwrap();
        let input = crate::lstr::LStr::from("a".repeat(28));
        let text = super::ReText::new_rc(false, &input);
        assert!(matches!(
            re.exec_text_shared(&text, 0, &crate::RuntimeInterrupt::default()),
            Ok(None)
        ));
    }

    #[test]
    fn case_insensitive_ascii_literal_uses_fold_search() {
        let re = super::Regex::new("abc", "i").unwrap();
        assert!(
            re.literal_fold.is_some(),
            "an ASCII icase literal gets the fold needle"
        );
        assert!(re.literal_ascii.is_none());
        let text = |s: &str| super::ReText::new_rc(false, &crate::lstr::LStr::from(s));
        let control = crate::RuntimeInterrupt::default();
        // ASCII subjects run the fold-aware byte search: any case mix, correct spans.
        assert_eq!(
            re.exec_text_shared(&text("xxABCyy"), 0, &control)
                .unwrap()
                .unwrap()[0],
            Some((2, 5))
        );
        assert_eq!(
            re.exec_text_shared(&text("aBc"), 0, &control)
                .unwrap()
                .unwrap()[0],
            Some((0, 3))
        );
        assert!(re
            .exec_text_shared(&text("Abx"), 0, &control)
            .unwrap()
            .is_none());
        // A non-ASCII subject keeps the matcher path but stays equivalent.
        assert_eq!(
            re.exec_text_shared(&text("éABC"), 0, &control)
                .unwrap()
                .unwrap()[0],
            Some((1, 4))
        );
    }

    #[test]
    fn case_insensitive_literal_with_non_ascii_character() {
        let re = super::Regex::new("straße", "i").unwrap();
        assert!(
            re.literal_fold.is_some(),
            "UTF-8 needle with a non-ASCII fold-free char"
        );
        let text = |s: &str| super::ReText::new_rc(false, &crate::lstr::LStr::from(s));
        let control = crate::RuntimeInterrupt::default();
        // 'ß' does not fold to 'ss': simple folding only (verified against V8/test262).
        assert!(re
            .exec_text_shared(&text("STRASSE"), 0, &control)
            .unwrap()
            .is_none());
        // Capital eszett (U+1E9E) is not folded by non-u canonicalization either.
        assert!(re
            .exec_text_shared(&text("STRAẞE"), 0, &control)
            .unwrap()
            .is_none());
        // An exact fold hit with the raw char matches on the wide subject.
        assert_eq!(
            re.exec_text_shared(&text("XSTRAßE"), 0, &control)
                .unwrap()
                .unwrap()[0],
            Some((1, 7))
        );
    }

    #[test]
    fn bounded_string_set_repetition_memoizes_by_position() {
        // Each `a` can end a one- or two-element string, so the failing repetition can reach
        // every position with many different counts. Once `max` is out of reach of the rest of
        // the subject the count is irrelevant, and the failure memo stays linear in the subject
        // instead of growing with position × count.
        let re = super::Regex::new(r"^[\q{aa|a}]{0,100000}$", "v").unwrap();
        let input = crate::lstr::LStr::from("a".repeat(20_000) + "b");
        let text = super::ReText::new_rc(true, &input);
        let start = std::time::Instant::now();
        let found = re.exec_text_shared(&text, 0, &crate::RuntimeInterrupt::default());
        assert!(matches!(found, Ok(None)));
        assert!(start.elapsed() < std::time::Duration::from_secs(20));
    }

    #[test]
    fn repeated_string_properties_use_heap_backtracking_state() {
        for property in ["Basic_Emoji", "RGI_Emoji"] {
            let set = super::property_of_strings(property).expect("known string property");
            let mut subject = String::new();
            for (start, end) in set.ranges {
                for code_point in start..=end {
                    if let Some(character) = char::from_u32(code_point) {
                        subject.push(character);
                    }
                }
            }
            for string in set.strings {
                subject.extend(string);
            }
            let re = super::Regex::new(&format!(r"^\p{{{property}}}+$"), "v").unwrap();
            let input = crate::lstr::LStr::from(subject.as_str());
            let text = super::ReText::new_rc(true, &input);
            assert!(matches!(
                re.exec_text_shared(&text, 0, &crate::RuntimeInterrupt::default()),
                Ok(Some(_))
            ));
        }
    }

    #[test]
    fn matcher_honors_deadlines_and_cross_thread_cancellation() {
        let re = super::Regex::new("needle", "").unwrap();

        let deadline = crate::RuntimeInterrupt::default();
        deadline.set_deadline(Some(std::time::Instant::now()));
        let short_input = crate::lstr::LStr::from("haystack");
        let short_text = super::ReText::new_rc(false, &short_input);
        assert!(matches!(
            re.exec_text_shared(&short_text, 0, &deadline),
            Err(super::MatchError::Interrupted(
                crate::InterruptReason::DeadlineExceeded
            ))
        ));

        // The first poll occurs before scanning. Cancelling after the call has started therefore
        // proves the literal prescan continues to poll while processing a long subject.
        let control = std::sync::Arc::new(crate::RuntimeInterrupt::default());
        let canceller = std::thread::spawn({
            let control = control.clone();
            move || {
                std::thread::sleep(std::time::Duration::from_millis(1));
                control.cancel();
            }
        });
        let long_input = crate::lstr::LStr::from("a".repeat(32 * 1024 * 1024));
        let long_text = super::ReText::new_rc(false, &long_input);
        let outcome = re.exec_text_shared(&long_text, 0, &control);
        canceller.join().expect("canceller thread");
        assert!(matches!(
            outcome,
            Err(super::MatchError::Interrupted(
                crate::InterruptReason::Cancelled
            ))
        ));
    }

    #[test]
    fn escaped_open_bracket_patterns_compile() {
        for pattern in [r"\s*([+>~\s])\s*([a-zA-Z#.*:\[])", r"^[\s[]?shapgvba"] {
            super::Regex::new(pattern, "g")
                .unwrap_or_else(|error| panic!("internal matcher rejected {pattern:?}: {error}"));
        }
    }

    #[test]
    fn legacy_identity_escaped_punctuation_compiles() {
        super::Regex::new(r#"(^|[^\\])\"\\\/Qngr\((-?[0-9]+)\)\\\/\""#, "g")
            .expect("internal matcher should accept legacy identity escapes");
    }

    #[test]
    fn guaranteed_ascii_backreference_matches() {
        let re =
            super::Regex::new(r#"^(\[) *@?([\w-]+) *([!*$^~=]*) *('?"?)(.*?)\4 *\]"#, "").unwrap();
        let input = crate::lstr::LStr::from("[glcr=fhozvg]");
        let text = super::ReText::new_rc(false, &input);
        let caps = re
            .exec_text_shared(&text, 0, &crate::RuntimeInterrupt::default())
            .unwrap()
            .unwrap();
        assert_eq!(caps[0], Some((0, 13)));
    }

    #[test]
    fn whole_match_projection_preserves_public_match_span() {
        for (pattern, flags, input, start, expected) in [
            ("(a)(b)(c)(d)(e)", "", "xxabcde", 0, Some((2, 7))),
            ("needle", "", "xxneedle", 1, Some((2, 8))),
            ("(😀)", "", "x😀y", 0, Some((1, 3))),
            ("(😀)", "u", "x😀y", 0, Some((1, 2))),
            ("needle", "", "haystack", 0, None),
        ] {
            let re = super::Regex::new(pattern, flags).unwrap();
            let input = crate::lstr::LStr::from(input);
            let text = super::ReText::new_rc(re.unicode, &input);
            let captures = re
                .exec_text_shared(&text, start, &crate::RuntimeInterrupt::default())
                .unwrap()
                .and_then(|caps| caps[0]);
            let whole = re
                .find_text_shared_entry_polled(&text, start, &crate::RuntimeInterrupt::default())
                .unwrap();
            assert_eq!(captures, expected);
            assert_eq!(whole, expected);
        }
    }

    #[test]
    fn capture_free_ascii_lookahead_matches() {
        let re = super::Regex::new("HF(?=;)", "i").unwrap();
        let input = crate::lstr::LStr::from("xhf;y");
        let text = super::ReText::new_rc(false, &input);
        assert_eq!(
            re.exec_text_shared(&text, 0, &crate::RuntimeInterrupt::default())
                .unwrap()
                .unwrap()[0],
            Some((1, 3))
        );
    }

    #[test]
    fn legacy_pattern_uses_utf16_element_offsets() {
        let re = super::Regex::new(r"Qngr\((-?[0-9]+)\)", "").unwrap();
        let input = crate::lstr::LStr::from("‰Qngr(-12)");
        let text = super::ReText::new_rc(false, &input);
        let caps = re
            .exec_text_shared(&text, 0, &crate::RuntimeInterrupt::default())
            .unwrap()
            .unwrap();
        assert_eq!(caps[0], Some((1, 10)));
        assert_eq!(caps[1], Some((6, 9)));
    }

    #[test]
    fn internal_literal_plan_honors_start_and_sticky() {
        let input = crate::lstr::LStr::from("xxneedle--needle");
        let text = super::ReText::new_rc(false, &input);
        let search = super::Regex::new("needle", "").unwrap();
        assert_eq!(
            search
                .exec_text_shared(&text, 3, &crate::RuntimeInterrupt::default())
                .unwrap()
                .unwrap()[0],
            Some((10, 16))
        );

        let sticky = super::Regex::new("needle", "y").unwrap();
        assert!(sticky
            .exec_text_shared(&text, 3, &crate::RuntimeInterrupt::default())
            .unwrap()
            .is_none());
        assert_eq!(
            sticky
                .exec_text_shared(&text, 10, &crate::RuntimeInterrupt::default())
                .unwrap()
                .unwrap()[0],
            Some((10, 16))
        );
    }

    #[test]
    fn straight_line_ascii_tier_is_lazy_and_semantically_exact() {
        let re = super::Regex::new("a.c", "").unwrap();
        let control = crate::RuntimeInterrupt::default();
        let input = crate::lstr::LStr::from("xxaXcyy");
        let text = super::ReText::new_rc(false, &input);

        assert_eq!(re.tier_feedback(), (0, 0, false, false));
        for _ in 0..(super::REGEXP_TIER_UP_THRESHOLD - 1) {
            assert_eq!(
                re.exec_text_shared(&text, 0, &control).unwrap().unwrap()[0],
                Some((2, 5))
            );
        }
        let (ticks, shapes, attempted, native) = re.tier_feedback();
        assert_eq!(ticks, super::REGEXP_TIER_UP_THRESHOLD - 1);
        assert_ne!(shapes & super::SUBJECT_ASCII, 0);
        assert!(!attempted);
        assert!(!native);

        assert_eq!(
            re.exec_text_shared(&text, 0, &control).unwrap().unwrap()[0],
            Some((2, 5))
        );
        let (ticks, _, attempted, native) = re.tier_feedback();
        assert_eq!(ticks, super::REGEXP_TIER_UP_THRESHOLD);
        assert!(attempted);
        assert!(native);

        // Dot still rejects both ASCII line terminators, and the sticky path only tries its exact
        // requested position after tier-up.
        let newline = super::ReText::new_rc(false, &crate::lstr::LStr::from("a\nc"));
        assert!(re
            .exec_text_shared(&newline, 0, &control)
            .unwrap()
            .is_none());
        let sticky = super::Regex::new("a.c", "y").unwrap();
        let sticky_text = super::ReText::new_rc(false, &crate::lstr::LStr::from("xaXc"));
        for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
            sticky.exec_text_shared(&sticky_text, 1, &control).unwrap();
        }
        assert_eq!(
            sticky
                .find_text_shared_entry_polled(&sticky_text, 1, &control)
                .unwrap(),
            Some((1, 4))
        );
    }

    #[test]
    fn ascii_tier_records_subject_shapes_and_rejects_unsafe_flags() {
        let re = super::Regex::new("a.c", "").unwrap();
        let control = crate::RuntimeInterrupt::default();
        let ascii = super::ReText::new_rc(false, &crate::lstr::LStr::from("aXc"));
        let one_byte = super::ReText::new_rc(false, &crate::lstr::LStr::from("aéc"));
        assert!(re.exec_text_shared(&ascii, 0, &control).unwrap().is_some());
        assert!(re
            .exec_text_shared(&one_byte, 0, &control)
            .unwrap()
            .is_some());
        let (_, shapes, _, _) = re.tier_feedback();
        assert_ne!(shapes & super::SUBJECT_ASCII, 0);
        assert_ne!(shapes & super::SUBJECT_ONE_BYTE, 0);

        let latin = super::Regex::new("é.", "").unwrap();
        let latin_text = super::ReText::new_rc(false, &crate::lstr::LStr::from("éX"));
        for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
            assert!(latin
                .exec_text_shared(&latin_text, 0, &control)
                .unwrap()
                .is_some());
        }
        assert!(latin.tier_feedback().3);

        for flags in ["i", "s", "u", "v"] {
            let guarded = super::Regex::new("a.c", flags).unwrap();
            let input = super::ReText::new_rc(guarded.unicode, &crate::lstr::LStr::from("aXc"));
            for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
                guarded.exec_text_shared(&input, 0, &control).unwrap();
            }
            let (_, _, attempted, native) = guarded.tier_feedback();
            assert!(attempted, "unsafe flag should close the tier-up attempt");
            assert!(!native, "unsafe flag must remain on the exact matcher");
        }

        let anchored = super::Regex::new("^a.c$", "").unwrap();
        for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
            anchored.exec_text_shared(&ascii, 0, &control).unwrap();
        }
        assert!(anchored.tier_feedback().3);

        let multiline = super::Regex::new("^a.c$", "m").unwrap();
        for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
            multiline.exec_text_shared(&ascii, 0, &control).unwrap();
        }
        assert!(!multiline.tier_feedback().3);
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
    #[test]
    fn native_regexp_machine_poll_with_armed_deadline_preserves_match_offsets() {
        let control = crate::RuntimeInterrupt::default();
        control.set_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_secs(60),
        ));
        // Exercise the executable mappings directly: the checked native Rust matcher
        // cannot reveal an ABI error at the generated code's host-interrupt call.
        let code = super::OneByteNativeCode::compile(&[super::OneByteNativeOp::Any])
            .expect("the fixed-width dot pattern must have executable code");
        for sticky in [false, true] {
            for start in 0..=4 {
                let expected = (start < 3).then_some((start, start + 1));
                assert_eq!(
                    code.find_ascii(b"abc", start, sticky, &control).unwrap(),
                    expected,
                    "byte mapping at {start}, sticky={sticky}"
                );
                assert_eq!(
                    code.find_one_byte(&[0xe9, 0x78, 0xff], start, sticky, &control)
                        .unwrap(),
                    expected,
                    "word mapping at {start}, sticky={sticky}"
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", unix))]
    #[test]
    fn native_regexp_aarch64_poll_explicitly_tests_return_value() {
        // Debug Rust can coincidentally leave useful flags; keep an emitter-level
        // gate as well as the release executable-mapping and JS regressions.
        for word_elements in [false, true] {
            let bytes = super::compile_one_byte_native_aarch64(
                &[super::OneByteNativeOp::Any],
                word_elements,
            )
            .unwrap();
            let instructions: Vec<u32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect();
            let poll = instructions
                .iter()
                .position(|&word| word == 0xd63f_0200)
                .expect("poll call must be emitted");
            assert_eq!(instructions[poll + 1], 0x7100_001f, "cmp w0, #0");
        }
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
    #[test]
    fn native_regexp_machine_poll_preserves_host_interrupt_reasons() {
        let code = super::OneByteNativeCode::compile(&[super::OneByteNativeOp::Any])
            .expect("the fixed-width dot pattern must have executable code");
        for expected in [
            crate::InterruptReason::Cancelled,
            crate::InterruptReason::UserNavigation,
            crate::InterruptReason::DeadlineExceeded,
        ] {
            let control = crate::RuntimeInterrupt::default();
            match expected {
                crate::InterruptReason::Cancelled => control.cancel(),
                crate::InterruptReason::UserNavigation => control.request_user_navigation(),
                crate::InterruptReason::DeadlineExceeded => control.set_deadline(Some(
                    std::time::Instant::now() - std::time::Duration::from_secs(1),
                )),
            }
            for result in [
                code.find_ascii(b"abc", 1, false, &control),
                code.find_one_byte(&[0xe9, 0x78, 0xff], 1, false, &control),
            ] {
                assert!(
                    matches!(result, Err(super::MatchError::Interrupted(reason)) if reason == expected),
                    "incorrect poll completion: {result:?}, expected {expected:?}"
                );
            }
        }
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix))]
    #[test]
    fn machine_code_subset_matches_reference_and_preserves_interrupts() {
        let control = crate::RuntimeInterrupt::default();
        for (pattern, subject, expected) in [
            ("a.c", "xxaXcyy", Some((2, 5))),
            ("^a.c$", "aXc", Some((0, 3))),
            ("a.c$", "xxaXc", Some((2, 5))),
            ("^a.c", "xxaXcyy", None),
            ("^-ms-", "background-image", None),
            ("[a-z].", "123aYzz", Some((3, 5))),
            (r"\d[a-z]", "xx7ayzz", Some((2, 4))),
            ("[^a].", "aabYzz", Some((2, 4))),
        ] {
            let re = super::Regex::new(pattern, "").unwrap();
            let input = crate::lstr::LStr::from(subject);
            let text = super::ReText::new_rc(false, &input);
            for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
                re.exec_text_shared(&text, 0, &control).unwrap();
            }
            assert!(
                re.tier_feedback().3,
                "machine subset was not emitted for /{pattern}/"
            );
            let machine = re
                .find_text_shared_entry_polled(&text, 0, &control)
                .unwrap();
            let reference = re.find_impl(subject.as_bytes(), 0, &control).unwrap();
            assert_eq!(machine, expected);
            assert_eq!(
                machine, reference,
                "machine/reference mismatch for /{pattern}/"
            );
        }

        for (pattern, subject, expected) in [
            ("é.", "xxéXyy", Some((2, 4))),
            ("[é].", "xxéYzz", Some((2, 4))),
        ] {
            let re = super::Regex::new(pattern, "").unwrap();
            let input = crate::lstr::LStr::from(subject);
            let text = super::ReText::new_rc(false, &input);
            assert_eq!(text.subject_shape(), super::SUBJECT_ONE_BYTE);
            for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
                re.exec_text_shared(&text, 0, &control).unwrap();
            }
            assert!(re.tier_feedback().3);
            let machine = re
                .find_text_shared_entry_polled(&text, 0, &control)
                .unwrap();
            let reference = re.find_impl(text.elements(), 0, &control).unwrap();
            assert_eq!(machine, expected);
            assert_eq!(machine, reference, "word-load machine/reference mismatch");
        }

        let re = super::Regex::new("a.c", "").unwrap();
        let input = crate::lstr::LStr::from("aXc");
        let text = super::ReText::new_rc(false, &input);
        for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
            re.exec_text_shared(&text, 0, &control).unwrap();
        }
        let cancelled = crate::RuntimeInterrupt::default();
        cancelled.cancel();
        assert!(matches!(
            re.find_text_shared_entry_polled(&text, 0, &cancelled),
            Err(super::MatchError::Interrupted(
                crate::InterruptReason::Cancelled
            ))
        ));
    }

    #[test]
    fn one_byte_native_subset_differentially_matches_bytecode() {
        let subjects: &[&[u8]] = &[
            b"", b"a", b"aXc", b"xxaXcyy", b"a\nc", b"a\rc", b"0a", b"za", b"_9", b"!a",
        ];
        let control = crate::RuntimeInterrupt::default();
        for pattern in ["a.c", "[a-z].", r"\d[a-z]", r"[^a]."] {
            for &subject in subjects {
                for sticky in [false, true] {
                    let re = super::Regex::new(pattern, if sticky { "y" } else { "" }).unwrap();
                    let native = super::OneByteNativeProgram::compile(&re.prog, re.multiline)
                        .expect("subset pattern must compile to the native representation");
                    for (instruction, op) in
                        re.prog[1..re.prog.len() - 2].iter().zip(native.ops.iter())
                    {
                        for byte in 0..=u8::MAX {
                            let reference = match instruction {
                                super::Inst::Char(c) => *c == byte as u32,
                                super::Inst::Any => !super::is_line_terminator_u32(byte as u32),
                                super::Inst::Class(class) => {
                                    class.matches(byte as u32, false, false)
                                }
                                _ => panic!("unsupported instruction in subset test"),
                            };
                            assert_eq!(
                                op.matches(byte as u32),
                                reference,
                                "operation predicate mismatch for /{pattern}/ and byte {byte}"
                            );
                        }
                    }
                    for start in 0..=subject.len() {
                        let fast = native.find(subject, start, sticky, &control);
                        let reference = re.find_impl(subject, start, &control);
                        assert_eq!(
                            fast, reference,
                            "native/reference mismatch for /{pattern}/ at {start}, sticky={sticky}, subject={subject:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn terminal_many_native_subset_differentially_matches_bytecode() {
        let subjects: &[&[u8]] = &[
            b"", b"a", b"aa", b"xxaaay", b"bbb", b"12abc", b"a\na", b"a\ra",
        ];
        let control = crate::RuntimeInterrupt::default();
        for pattern in [
            "a+", "a+?", "a{2,4}", "a{2,4}?", "[a-z]+", r"\d{2,4}", ".*", ".*?",
        ] {
            for &subject in subjects {
                for sticky in [false, true] {
                    let re = super::Regex::new(pattern, if sticky { "y" } else { "" }).unwrap();
                    let native = super::OneByteNativeProgram::compile(&re.prog, re.multiline)
                        .expect("terminal Many pattern must compile to the native representation");
                    assert!(matches!(
                        native.ops.as_ref(),
                        [super::OneByteNativeOp::Many { .. }]
                    ));
                    for start in 0..=subject.len() {
                        let fast = native.find(subject, start, sticky, &control);
                        let reference = re.find_impl(subject, start, &control);
                        assert_eq!(
                            fast, reference,
                            "terminal Many mismatch for /{pattern}/ at {start}, sticky={sticky}, subject={subject:?}"
                        );
                    }
                }
            }
        }

        let re = super::Regex::new("é+", "").unwrap();
        let input = crate::lstr::LStr::from("xxééZ");
        let text = super::ReText::new_rc(false, &input);
        assert_eq!(text.subject_shape(), super::SUBJECT_ONE_BYTE);
        let native = super::OneByteNativeProgram::compile(&re.prog, re.multiline)
            .expect("one-byte non-ASCII terminal Many must compile");
        for start in 0..=text.elements().len() {
            let fast = native.find(text.elements(), start, false, &control);
            let reference = re.find_impl(text.elements(), start, &control);
            assert_eq!(
                fast, reference,
                "non-ASCII terminal Many mismatch at {start}"
            );
        }
    }

    #[test]
    fn terminal_many_native_path_promotes_and_polls() {
        let control = crate::RuntimeInterrupt::default();
        let re = super::Regex::new("[a-z]+", "").unwrap();
        let input = crate::lstr::LStr::from("abcxyz");
        let text = super::ReText::new_rc(false, &input);
        for _ in 0..super::REGEXP_TIER_UP_THRESHOLD {
            re.exec_text_shared(&text, 0, &control).unwrap();
        }
        assert!(re.tier_feedback().3);
        assert_eq!(
            re.find_text_shared_entry_polled(&text, 0, &control)
                .unwrap(),
            Some((0, 6))
        );

        let interrupted = crate::RuntimeInterrupt::default();
        interrupted.cancel();
        let long = vec![b'a'; super::INTERRUPT_POLL_MASK + 1];
        let native = super::OneByteNativeProgram::compile(&re.prog, re.multiline).unwrap();
        assert!(matches!(
            native.find(&long[..], 0, false, &interrupted),
            Err(super::MatchError::Interrupted(
                crate::InterruptReason::Cancelled
            ))
        ));
    }
}
