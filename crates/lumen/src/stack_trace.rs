//! Source positions in `Error.prototype.stack`, formatted like V8's frames:
//! `    at name (url:line:column)`, `    at url:line:column` for anonymous functions and top-level
//! code, `    at eval (<anonymous>:line:column)` for eval code and `    at name (<anonymous>)` for
//! built-ins.
//!
//! Positions are byte offsets into the text of a [`ScriptSource`]. The parser records them on
//! call, `new`, property-access, tagged-template and assignment nodes, and each compiled chunk
//! keeps an operation-to-position table (see `bytecode::SourcePositions`). Nothing is tracked
//! while code runs normally; only these events record a position:
//!
//! - An explicit call (`f()`, `new C()`, `super()`, a tagged template) stores its position in
//!   [`Interp::call_site`] for the duration of the call. Pushing a function frame moves it into
//!   [`crate::interpreter::FnFrame::call_site`] (the native template JIT's direct calls write the
//!   compile-time position there instead), and popping restores it, so every frame records where
//!   its caller was. Outside an explicit call the field holds [`NO_POSITION`].
//! - An error the engine creates while an operation runs (a TypeError for `null.f`, a
//!   ReferenceError, …) is captured with an unknown top position. The innermost operation with
//!   a recorded position that the error propagates through places it ([`Interp::place_thrown`]):
//!   the tree-walker at the AST node, the bytecode VM and the JIT helpers at the operation that
//!   failed. Every tier attributes an operation to the same node, so they agree.
//!
//! Line and column are computed only when a stack is formatted: columns count UTF-16 code units
//! like V8, from a line table built on first use. Frames that were entered through an implicit
//! call (a getter, `valueOf`, an iterator step, a Proxy trap) record no call position for their
//! caller; that caller reports its function's start instead.
//!
//! Captured frames hold names, positions and shared [`ScriptSource`]s, never JavaScript objects,
//! and nothing here is Realm-specific: compiled code and its position tables serve every Realm.

use crate::interpreter::{Abrupt, Interp};
use crate::lstr::LStr;
use crate::value::{Callable, Exotic, Value};
use std::rc::Rc;

/// The position of an operation that has none (see the module documentation).
pub(crate) const NO_POSITION: u32 = u32::MAX;

/// The location V8 prints for code without a URL and for built-in frames.
const ANONYMOUS: &str = "<anonymous>";

/// Where an embedder's script text came from, for the locations its stack frames report.
///
/// `line` and `column` are the zero-based position at which the text starts within the resource
/// named by `url` (V8's `ScriptOrigin` offsets): an inline `<script>` element whose text starts
/// on line 12 of its document passes `line: 11`. `column` applies to the first line only.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceOrigin<'a> {
    pub url: &'a str,
    pub line: u32,
    pub column: u32,
}

impl<'a> SourceOrigin<'a> {
    /// Text that starts at the beginning of the resource named by `url`.
    pub fn new(url: &'a str) -> Self {
        SourceOrigin {
            url,
            line: 0,
            column: 0,
        }
    }
}

/// What a [`ScriptSource`] holds, which decides how its top-level frame is named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceKind {
    Script,
    Module,
    /// Direct or indirect eval code (and ShadowRealm evaluation).
    Eval,
    /// The source text CreateDynamicFunction assembles for `new Function(…)`.
    Function,
}

/// One parsed source text: the URL its frames report and the text their positions index. Every
/// function parsed from it shares it. Holds no JavaScript values.
pub(crate) struct ScriptSource {
    /// Engine text (see `crate::jstr`), like the frame names it is printed beside.
    url: Box<str>,
    text: Rc<str>,
    line_offset: u32,
    column_offset: u32,
    kind: SourceKind,
    /// Byte offsets of the line starts, built when a position is first formatted.
    line_starts: std::cell::OnceCell<Box<[u32]>>,
    /// UTF-16 unit counts at sampled byte offsets, built when a column far from its line start
    /// is first formatted (minified bundles are a few very long lines).
    unit_samples: std::cell::OnceCell<Box<[(u32, u32)]>>,
    /// The ScriptOrModule of code from this source, as the host-defined base its `import()`
    /// calls resolve against (see [`Interp::active_script_or_module`]): set when a Script or
    /// Module starts to evaluate, and for eval code and dynamic functions to the active
    /// ScriptOrModule where they are created (ECMA-262 PerformEval, OrdinaryFunctionCreate).
    /// Unset is null, as for host snapshot code.
    referrer: std::cell::OnceCell<Rc<str>>,
}

impl std::fmt::Debug for ScriptSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptSource")
            .field("url", &self.url)
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

impl ScriptSource {
    pub(crate) fn new(
        text: Rc<str>,
        kind: SourceKind,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Rc<ScriptSource> {
        let origin = origin.copied().unwrap_or(SourceOrigin::new(ANONYMOUS));
        let url = if origin.url.is_empty() {
            ANONYMOUS
        } else {
            origin.url
        };
        Rc::new(ScriptSource {
            url: crate::jstr::from_text(url).into(),
            text,
            line_offset: origin.line,
            column_offset: origin.column,
            kind,
            line_starts: std::cell::OnceCell::new(),
            unit_samples: std::cell::OnceCell::new(),
            referrer: std::cell::OnceCell::new(),
        })
    }

    pub(crate) fn kind(&self) -> SourceKind {
        self.kind
    }

    /// The referrer of this source's ScriptOrModule, None when that is null.
    pub(crate) fn referrer(&self) -> Option<&Rc<str>> {
        self.referrer.get()
    }

    /// Record the referrer of this source's ScriptOrModule. A source evaluates as one
    /// ScriptOrModule, so only the first record counts.
    pub(crate) fn set_referrer(&self, referrer: Rc<str>) {
        let _ = self.referrer.set(referrer);
    }

    pub(crate) fn text(&self) -> &Rc<str> {
        &self.text
    }

    /// The one-based line and UTF-16 column of byte `offset`, counting ECMAScript line
    /// terminators (LF, CR, CRLF, U+2028, U+2029) and the origin's offsets.
    pub(crate) fn line_column(&self, offset: u32) -> (u32, u32) {
        let starts = self.line_starts.get_or_init(|| line_starts(&self.text));
        let offset = (offset as usize).min(self.text.len());
        let line = starts.partition_point(|&start| start as usize <= offset) - 1;
        let start = starts[line] as usize;
        // Positions are token starts, so always character boundaries; stay total regardless.
        let units = self
            .text
            .get(start..offset)
            .map_or(offset - start, |slice| self.units_in(start, offset, slice))
            as u32;
        let column = if line == 0 {
            units.saturating_add(self.column_offset)
        } else {
            units
        };
        (
            (line as u32)
                .saturating_add(self.line_offset)
                .saturating_add(1),
            column.saturating_add(1),
        )
    }

    /// The UTF-16 length of `text[start..offset]` (`slice`), both ends character boundaries.
    /// A short span is counted directly; a long one, such as a column deep into a minified
    /// bundle's single line, is the difference of two prefix counts that each scan at most
    /// one sample block, so formatting a stack frame no longer reads the whole line.
    fn units_in(&self, start: usize, offset: usize, slice: &str) -> usize {
        if slice.len() <= 2 * UNIT_BLOCK {
            return crate::jstr::unit_len(slice);
        }
        let samples = self.unit_samples.get_or_init(|| unit_samples(&self.text));
        let units_before = |at: usize| {
            let k = samples.partition_point(|&(byte, _)| byte as usize <= at) - 1;
            let (byte, units) = samples[k];
            units as usize + crate::jstr::unit_len(&self.text[byte as usize..at])
        };
        units_before(offset) - units_before(start)
    }

    fn write_location(&self, out: &mut String, offset: u32) {
        use std::fmt::Write as _;
        let (line, column) = self.line_column(offset);
        let _ = write!(out, "{}:{line}:{column}", self.url);
    }

    /// Heap bytes this record owns beside the shared text.
    pub(crate) fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.url.len()
            + self
                .line_starts
                .get()
                .map_or(0, |starts| starts.len() * std::mem::size_of::<u32>())
            + self.unit_samples.get().map_or(0, |samples| {
                samples.len() * std::mem::size_of::<(u32, u32)>()
            })
    }
}

/// Bytes between the sampled UTF-16 prefix counts of [`ScriptSource::units_in`].
const UNIT_BLOCK: usize = 4096;

/// (byte offset, UTF-16 units before it) at every `UNIT_BLOCK` bytes, rounded up to a character
/// boundary; the first entry is (0, 0).
fn unit_samples(text: &str) -> Box<[(u32, u32)]> {
    let mut samples = vec![(0u32, 0u32)];
    let mut at = 0usize;
    let mut units = 0usize;
    while at < text.len() {
        let mut next = (at + UNIT_BLOCK).min(text.len());
        while !text.is_char_boundary(next) {
            next += 1;
        }
        units += crate::jstr::unit_len(&text[at..next]);
        at = next;
        if at < text.len() {
            samples.push((at as u32, units as u32));
        }
    }
    samples.into_boxed_slice()
}

fn line_starts(text: &str) -> Box<[u32]> {
    let mut starts = vec![0u32];
    let mut chars = text.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        let next = match c {
            '\r' if matches!(chars.peek(), Some((_, '\n'))) => continue,
            '\n' | '\r' | '\u{2028}' | '\u{2029}' => at + c.len_utf8(),
            _ => continue,
        };
        starts.push(next as u32);
    }
    starts.into_boxed_slice()
}

/// How a captured frame prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameKind {
    /// An ordinary function call: `at name (location)`, or `at location` when anonymous.
    Function,
    /// A `[[Construct]]` call: `at new name (location)`.
    Constructor,
    /// Script or module code: `at location`.
    TopLevel,
    /// Eval code: `at eval (location)`.
    Eval,
    /// A built-in implemented in JavaScript: `at name (<anonymous>)`.
    Native,
}

/// One captured stack frame.
#[derive(Clone)]
pub(crate) struct Frame {
    kind: FrameKind,
    name: Option<LStr>,
    source: Option<Rc<ScriptSource>>,
    /// The byte offset of the operation the frame was running, or [`NO_POSITION`] until
    /// [`Interp::place_thrown`] supplies it.
    position: u32,
    /// What an unknown position reports: the start of the function (or of the source).
    start: u32,
    /// The frame's execution-context depth ([`Interp::context_depth`] while it runs): only an
    /// operation of that context can place its position.
    depth: u32,
}

impl Frame {
    fn write(&self, out: &mut String) {
        out.push_str("\n    at ");
        let name = self.name.as_deref().filter(|name| !name.is_empty());
        let Some(source) = self.source.as_deref() else {
            out.push_str(name.unwrap_or(ANONYMOUS));
            if name.is_some() {
                out.push_str(" (");
                out.push_str(ANONYMOUS);
                out.push(')');
            }
            return;
        };
        let offset = if self.position == NO_POSITION {
            self.start
        } else {
            self.position
        };
        let label = match (self.kind, name) {
            (FrameKind::Eval, _) => Some("eval"),
            (FrameKind::Constructor, name) => {
                out.push_str("new ");
                Some(name.unwrap_or(ANONYMOUS))
            }
            (FrameKind::Function | FrameKind::Native, name) => name,
            (FrameKind::TopLevel, _) => None,
        };
        match label {
            Some(label) => {
                out.push_str(label);
                out.push_str(" (");
                source.write_location(out, offset);
                out.push(')');
            }
            None => source.write_location(out, offset),
        }
    }
}

/// The frames captured for an Error object (or by `Error.captureStackTrace`).
#[derive(Clone)]
pub(crate) struct StackTrace {
    frames: Vec<Frame>,
    /// The formatted frame lines, built on first use.
    lines: Option<Rc<str>>,
}

impl StackTrace {
    /// The `\n    at …` lines.
    pub(crate) fn lines(&mut self) -> Rc<str> {
        if let Some(lines) = &self.lines {
            return lines.clone();
        }
        let mut out = String::new();
        for frame in &self.frames {
            frame.write(&mut out);
        }
        let lines: Rc<str> = out.into();
        self.lines = Some(lines.clone());
        lines
    }

    /// The formatted lines when they are already known, for side-effect-free diagnostics.
    pub(crate) fn formatted(&self) -> String {
        match &self.lines {
            Some(lines) => lines.to_string(),
            None => {
                let mut out = String::new();
                for frame in &self.frames {
                    frame.write(&mut out);
                }
                out
            }
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.frames.capacity() * std::mem::size_of::<Frame>()
            + self.lines.as_ref().map_or(0, |lines| lines.len())
    }
}

/// A script, module or eval context, or a resumed generator or async function body: an
/// execution context without a function frame. Function frames live on `Interp::fn_frames`;
/// these interleave with them by `depth`.
pub(crate) struct CodeFrame {
    kind: FrameKind,
    name: Option<LStr>,
    source: Option<Rc<ScriptSource>>,
    /// See [`Frame::start`].
    start: u32,
    /// `fn_frames.len()` on entry: the context runs above those function frames.
    depth: u32,
    /// The caller's [`Interp::call_site`] on entry, restored on exit.
    call_site: u32,
}

/// The frame a generator, async function or module body reports while it runs after resuming:
/// by then the call that created it has returned, so no function frame stands for it.
#[derive(Clone)]
pub(crate) struct ResumedFrame {
    kind: FrameKind,
    name: Option<LStr>,
    source: Rc<ScriptSource>,
    start: u32,
    /// `Rc::as_ptr` identity of the function object whose call created the body, if any: its
    /// first step runs inside that call's frame (see [`Interp::in_resumed`]).
    function: usize,
}

/// The `Error.stackTraceLimit` that applies when none is installed (V8's default).
pub(crate) const DEFAULT_STACK_TRACE_LIMIT: f64 = 10.0;

/// Which frames a capture leaves out.
#[derive(Clone, Copy)]
pub(crate) enum Skip {
    None,
    /// Every frame up to and including the innermost call of the function with this
    /// `Rc::as_ptr` identity (V8's SKIP_UNTIL_SEEN); everything when it is not on the stack.
    UntilSeen(usize),
}

impl Interp {
    /// Pop the innermost function frame, restoring the caller's explicit-call position.
    #[inline]
    pub(crate) fn pop_fn_frame(&mut self) {
        if let Some(frame) = self.fn_frames.pop() {
            self.call_site = frame.call_site;
        }
    }

    /// The number of execution contexts above the host: function frames plus code frames.
    pub(crate) fn context_depth(&self) -> u32 {
        (self.fn_frames.len() + self.code_frames.len()) as u32
    }

    /// Enter script, module or eval code from `source` (None for code without positions, such
    /// as a decoded snapshot). Pair with [`Interp::leave_code`].
    pub(crate) fn enter_code(&mut self, source: Option<Rc<ScriptSource>>) {
        let kind = match source.as_deref().map(ScriptSource::kind) {
            Some(SourceKind::Eval) => FrameKind::Eval,
            _ => FrameKind::TopLevel,
        };
        self.code_frames.push(CodeFrame {
            kind,
            name: None,
            source,
            start: 0,
            depth: self.fn_frames.len() as u32,
            call_site: std::mem::replace(&mut self.call_site, NO_POSITION),
        });
    }

    /// The frame of a generator or async function body that the innermost call (of `func`)
    /// creates now, for when it resumes later. None for functions without a source.
    pub(crate) fn resumed_frame(&self, func: &crate::ast::Function) -> Option<Box<ResumedFrame>> {
        let source = func.script.clone().filter(|_| !func.self_hosted)?;
        let callee = self
            .fn_frames
            .last()
            .map(|frame| (frame.fn_ptr, frame.callee()));
        let name = callee.as_ref().and_then(|(_, callee)| {
            match callee
                .borrow()
                .props
                .get("name")
                .map(|property| property.value())
            {
                Some(Value::Str(name)) if !name.is_empty() => Some(name),
                _ => None,
            }
        });
        Some(Box::new(ResumedFrame {
            kind: FrameKind::Function,
            name,
            source,
            start: func.start,
            function: callee.map_or(0, |(identity, _)| identity),
        }))
    }

    /// The frame of a module body that runs as a coroutine (top-level await).
    pub(crate) fn resumed_module_frame(source: Rc<ScriptSource>) -> Box<ResumedFrame> {
        Box::new(ResumedFrame {
            kind: FrameKind::TopLevel,
            name: None,
            source,
            start: 0,
            function: 0,
        })
    }

    /// Run a step of a coroutine body in its own code frame, unless the step runs inside the
    /// function frame of the call that created the body (an async function's first step).
    pub(crate) fn in_resumed<T>(
        &mut self,
        frame: Option<&ResumedFrame>,
        run: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let framed = frame.is_some_and(|frame| {
            frame.function != 0
                && self.fn_frames.last().map(|top| top.fn_ptr) == Some(frame.function)
                && self
                    .code_frames
                    .last()
                    .is_none_or(|code| (code.depth as usize) < self.fn_frames.len())
        });
        let Some(frame) = frame.filter(|_| !framed) else {
            return run(self);
        };
        self.code_frames.push(CodeFrame {
            kind: frame.kind,
            name: frame.name.clone(),
            source: Some(frame.source.clone()),
            start: frame.start,
            depth: self.fn_frames.len() as u32,
            call_site: std::mem::replace(&mut self.call_site, NO_POSITION),
        });
        let result = run(self);
        self.leave_code();
        result
    }

    pub(crate) fn leave_code(&mut self) {
        if let Some(frame) = self.code_frames.pop() {
            self.call_site = frame.call_site;
        }
    }

    /// ECMA-262 GetActiveScriptOrModule: the source of the innermost execution context whose
    /// ScriptOrModule is not null. Function contexts take their function's (built-in, self-
    /// hosted and host snapshot functions have none); below them lie the script execution
    /// contexts a host pushed for job callbacks (HTML HostCallJobCallback).
    pub(crate) fn active_script_or_module(&self) -> Option<Rc<ScriptSource>> {
        let mut function = self.fn_frames.len();
        let mut code = self.code_frames.len();
        let mut pushed = self.job_script_contexts.len();
        loop {
            let depth = function + code;
            // A pushed context entered at depth d runs above the d contexts below it.
            if pushed > 0 && self.job_script_contexts[pushed - 1].0 as usize >= depth {
                pushed -= 1;
                return Some(self.job_script_contexts[pushed].1.clone());
            }
            if depth == 0 {
                return None;
            }
            let source = if code > 0 && self.code_frames[code - 1].depth as usize >= function {
                code -= 1;
                self.code_frames[code].source.clone()
            } else {
                function -= 1;
                match &self.fn_frames[function].callee().borrow().call {
                    Callable::User(user) if !user.func.self_hosted => user.func.script.clone(),
                    _ => None,
                }
            };
            if let Some(source) = source.filter(|source| source.referrer().is_some()) {
                return Some(source);
            }
        }
    }

    /// The referrer of the active ScriptOrModule (see [`Self::active_script_or_module`]).
    pub(crate) fn active_script_referrer(&self) -> Option<Rc<str>> {
        self.active_script_or_module()
            .and_then(|source| source.referrer().cloned())
    }

    /// HTML HostCallJobCallback steps 4-6: run `operation` with the script execution context
    /// HostMakeJobCallback saved (`script`) pushed below it, when there is one.
    pub(crate) fn with_job_script_context<R>(
        &mut self,
        script: Option<&Rc<ScriptSource>>,
        operation: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let Some(script) = script else {
            return operation(self);
        };
        self.job_script_contexts
            .push((self.context_depth(), script.clone()));
        let result = operation(self);
        self.job_script_contexts.pop();
        result
    }

    /// Run script, module or eval code from `source` in its own code frame.
    pub(crate) fn in_code<T>(
        &mut self,
        source: Option<Rc<ScriptSource>>,
        run: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.enter_code(source);
        let result = run(self);
        self.leave_code();
        result
    }

    /// `Error.stackTraceLimit` of the current Realm's %Error%, read without side effects like
    /// V8 (an own data property; an accessor counts as absent). None when it is not a Number:
    /// no stack is captured then.
    fn stack_trace_limit(&self) -> Option<usize> {
        let ctor = self.extra_protos.get("%ErrorCtor%")?;
        let ctor = ctor.borrow();
        let property = ctor.props.get("stackTraceLimit")?;
        if property.accessor() {
            return None;
        }
        match property.value() {
            Value::Num(limit) if limit.is_nan() || limit <= 0.0 => Some(0),
            // Infinity (and anything that large) keeps every frame.
            Value::Num(limit) => Some(if limit >= usize::MAX as f64 {
                usize::MAX
            } else {
                limit as usize
            }),
            _ => None,
        }
    }

    /// Capture the current execution contexts, innermost first, for a new error. None when
    /// `Error.stackTraceLimit` is not a Number.
    pub(crate) fn capture_stack_trace(&self, skip: Skip) -> Option<Box<StackTrace>> {
        let limit = self.stack_trace_limit()?;
        let mut frames = Vec::new();
        let mut skipping = matches!(skip, Skip::UntilSeen(_));
        // The position of the context being visited: the innermost runs at `call_site`, each
        // outer one where it called the context above it.
        let mut position = self.call_site;
        let mut function = self.fn_frames.len();
        let mut code = self.code_frames.len();
        while frames.len() < limit && (function > 0 || code > 0) {
            let depth = (function + code) as u32;
            // A code frame entered at depth d runs above function frames [0, d).
            let frame = if code > 0 && self.code_frames[code - 1].depth as usize >= function {
                code -= 1;
                let entry = &self.code_frames[code];
                let frame = entry.source.as_ref().map(|source| Frame {
                    kind: entry.kind,
                    name: entry.name.clone(),
                    source: Some(source.clone()),
                    position,
                    start: entry.start,
                    depth,
                });
                position = entry.call_site;
                frame
            } else {
                function -= 1;
                let entry = &self.fn_frames[function];
                if let Skip::UntilSeen(target) = skip {
                    if skipping {
                        skipping = entry.fn_ptr != target;
                        position = entry.call_site;
                        continue;
                    }
                }
                let frame = function_frame(&entry.callee(), entry.construct, position, depth);
                position = entry.call_site;
                frame
            };
            // Code frames above a skipped function are left out with it.
            if let Some(frame) = frame.filter(|_| !skipping) {
                frames.push(frame);
            }
        }
        if skipping {
            frames.clear();
        }
        Some(Box::new(StackTrace {
            frames,
            lines: None,
        }))
    }

    /// The operation at byte `position` of the current context failed with `thrown`: give the
    /// error's frame for this context that position if it has none yet. That frame is the top
    /// one for an error the engine created while the operation ran, or the frame that called
    /// the function that threw through an implicit call (a getter, `valueOf`, …) the operation
    /// made. Errors without such a frame, or not Errors at all, are left alone.
    #[cold]
    #[inline(never)]
    pub(crate) fn place_thrown(&self, thrown: &Abrupt, position: u32) {
        if position == NO_POSITION {
            return;
        }
        let Abrupt::Throw(thrown) = thrown else {
            return;
        };
        if let Some((mut trace, index)) = self.pending_frame(thrown) {
            trace.frames[index].position = position;
            trace.lines = None;
        }
    }

    /// A `catch` (or `finally`) of the current context took `caught`: whatever position the
    /// error's frame for this context still lacks stays unknown, and no explicit call is in
    /// progress any more.
    #[cold]
    #[inline(never)]
    pub(crate) fn settle_caught(&mut self, caught: &Value) {
        self.call_site = NO_POSITION;
        if let Some((mut trace, index)) = self.pending_frame(caught) {
            trace.frames[index].depth = u32::MAX;
        }
    }

    /// The thrown Error's unplaced frame for the current context, as (trace, frame index).
    fn pending_frame<'a>(
        &self,
        thrown: &'a Value,
    ) -> Option<(std::cell::RefMut<'a, StackTrace>, usize)> {
        let Value::Obj(object) = thrown else {
            return None;
        };
        let object = object.try_borrow_mut().ok()?;
        let depth = self.context_depth();
        let trace = std::cell::RefMut::filter_map(object, |object| match &mut object.exotic {
            Exotic::Error(Some(trace)) => Some(&mut **trace),
            _ => None,
        })
        .ok()?;
        let index = trace
            .frames
            .iter()
            .position(|frame| frame.depth == depth && frame.position == NO_POSITION)?;
        Some((trace, index))
    }
}

/// The frame of a call to `callee`, or None for a function that stays out of stack traces:
/// platform code from a host snapshot, which has neither source text nor a position.
fn function_frame(
    callee: &crate::value::Gc,
    construct: bool,
    position: u32,
    depth: u32,
) -> Option<Frame> {
    let object = callee.borrow();
    let name = match object.props.get("name").map(|property| property.value()) {
        Some(Value::Str(name)) if !name.is_empty() => Some(name),
        _ => None,
    };
    let kind = if construct {
        FrameKind::Constructor
    } else {
        FrameKind::Function
    };
    let Callable::User(user) = &object.call else {
        return Some(Frame {
            kind: FrameKind::Native,
            name,
            source: None,
            position,
            start: 0,
            depth,
        });
    };
    let func = &user.func;
    match &func.script {
        Some(source) if !func.self_hosted => Some(Frame {
            kind,
            name,
            source: Some(source.clone()),
            position,
            start: func.start,
            depth,
        }),
        // Self-hosted built-ins, and snapshot code that kept its source text, print as built-ins.
        _ if func.self_hosted || func.source.is_some() => Some(Frame {
            kind: FrameKind::Native,
            name,
            source: None,
            position,
            start: 0,
            depth,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_and_utf16_column() {
        let text: Rc<str> = "ab\ncαβ😀x\r\ny\u{2028}z".into();
        let source = ScriptSource::new(text.clone(), SourceKind::Script, None);
        let at = |needle: &str| text.find(needle).unwrap() as u32;
        assert_eq!(source.line_column(0), (1, 1));
        assert_eq!(source.line_column(at("b")), (1, 2));
        assert_eq!(source.line_column(at("c")), (2, 1));
        // Greek letters are one unit each, the emoji two.
        assert_eq!(source.line_column(at("x")), (2, 6));
        assert_eq!(source.line_column(at("y")), (3, 1));
        assert_eq!(source.line_column(at("z")), (4, 1));
        let origin = SourceOrigin {
            url: "https://example.com/",
            line: 9,
            column: 4,
        };
        let shifted = ScriptSource::new(text.clone(), SourceKind::Script, Some(&origin));
        assert_eq!(shifted.line_column(at("b")), (10, 6));
        assert_eq!(shifted.line_column(at("y")), (12, 1));
    }

    /// A column far from its line start is the difference of sampled prefix counts: it must
    /// equal the direct UTF-16 count at every character boundary, across sample blocks, for
    /// text mixing ASCII, two- and three-byte characters, astral characters and the scalars
    /// that stand for lone surrogates (ECMA-262 §6.1.4 string code units).
    #[test]
    fn long_line_columns_match_a_direct_utf16_count() {
        let pieces = ["a", "bc", "é", "€", "😀", "\u{10F800}", "\u{10FFFF}", "xyz"];
        let mut text = String::new();
        let mut seed = 0x2545f491u32;
        while text.len() < 5 * UNIT_BLOCK + 300 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            text.push_str(pieces[(seed >> 24) as usize % pieces.len()]);
            if seed.is_multiple_of(2503) {
                text.push('\n');
            }
        }
        let text: Rc<str> = text.into();
        let source = ScriptSource::new(text.clone(), SourceKind::Script, None);
        let starts = line_starts(&text);
        for (offset, _) in text.char_indices().step_by(7) {
            let line = starts.partition_point(|&start| start as usize <= offset) - 1;
            let start = starts[line] as usize;
            let expected = crate::jstr::unit_len(&text[start..offset]) as u32 + 1;
            assert_eq!(
                source.line_column(offset as u32).1,
                expected,
                "offset {offset}"
            );
        }
    }
}
