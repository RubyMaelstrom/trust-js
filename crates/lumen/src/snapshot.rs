//! A compact binary (de)serializer for the parse AST, so the runtime's static JS glue can be
//! parsed once at build time and *decoded* (much cheaper than lex+parse) on every boot.
//!
//! Design points:
//! - **Optimization, not source of truth.** [`decode`] can fail (version skew, truncation); the
//!   caller falls back to re-parsing the original source, so a codec bug can never miscompile —
//!   at worst it costs a parse. A `MAGIC`+`VERSION` header makes skew a clean decode error.
//! - **Only parser output is encoded.** `Function`'s `scan`/`hoist`/`calls`/`code` are lazy
//!   runtime caches (`Cell`/`OnceCell`); decode initializes them empty, exactly as the parser
//!   leaves them. Ordinary snapshots preserve reflection source; the explicit host-bootstrap
//!   variant omits only function/class source text, leaving executable AST nodes intact.
//! - **Function bodies are length-prefixed** (a little-endian `u32` byte count before the
//!   statements), so [`decode_static`] can leave each body of a `&'static` snapshot encoded and
//!   [`decode_function_body`] can decode it when the function first needs it (see
//!   [`FunctionBody`]). The first static decode of a snapshot in the process still decodes, and
//!   so validates, every body under the same budgets as [`decode`]; it drops each body after
//!   validating it. Later decodes of those immutable bytes skip the bodies by their lengths.
//! - **Run-once bootstrap functions are marked** in a second flags byte: the encoder knows which
//!   function expressions run once per evaluation of the Script (`Function::run_once`). Only a
//!   static decode keeps the mark, which changes how the function is compiled, never what it
//!   does.
//! - Interned `&'static str` operators are re-interned from [`KEYWORDS`]/[`PUNCTUATORS`] on
//!   decode (every op the parser emits comes from those tables).
//!
//! The format is deliberately dumb — a preorder walk with a `u8` tag per enum, LEB128 lengths,
//! little-endian `f64`. It is an internal build/runtime contract, never persisted across builds.

use std::cell::{Cell, OnceCell};
use std::rc::Rc;

use crate::ast::*;
use crate::bigint::JsBigInt;
use crate::stack_trace::NO_POSITION;
use crate::token::{KEYWORDS, PUNCTUATORS};

const MAGIC: u32 = 0x4c_53_4e_31; // "LSN1"
/// Bump on any AST or format change. A mismatch makes `decode` fail → caller re-parses.
const VERSION: u32 = 5;
/// Snapshots are an optimization for trusted build-time glue, but the public embedding API can be
/// handed arbitrary bytes. Keep corrupt data from turning its length fields into unbounded work.
const MAX_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;
const MAX_SNAPSHOT_ALLOC_BYTES: usize = 256 * 1024 * 1024;
// Keep malformed recursion bounded independently of the caller's native stack headroom.
const MAX_SNAPSHOT_DEPTH: usize = 128;
const ACCOUNTED_NODE_BYTES: usize = 64;

// Debug codec frames contain large AST enum temporaries: dec_expr_inner measured 11,936 bytes
// on ARM64. A host can enter the codec with suspended JS/VM frames already occupying most of
// its stack. Check every recursive AST gateway *before* entering its large inner frame; the
// intervening collection/property helpers do not recurse without crossing another gateway.
// 128 KiB leaves substantial room for one such edge, allocation failure, and AST cleanup. New
// segments are 1 MiB and live only for that continuation. Decode's existing depth/byte/allocation
// budgets remain authoritative, rather than silently reducing valid input depth on re-entry.
// This is implementation storage only: ECMA-262 ParseScript/ScriptEvaluation and HTML's
// run-a-classic-script ordering are unchanged (local ECMA-262 snapshot e28783d5fc9d).
#[cfg(not(target_arch = "wasm32"))]
const CODEC_STACK_RED_ZONE: usize = 128 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const CODEC_STACK_SEGMENT: usize = 1024 * 1024;

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static CODEC_STACK_SEGMENTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

#[inline]
fn with_codec_stack<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(not(target_arch = "wasm32"))]
    if stacker::remaining_stack().is_none_or(|remaining| remaining < CODEC_STACK_RED_ZONE) {
        #[cfg(test)]
        struct SegmentGuard;
        #[cfg(test)]
        impl Drop for SegmentGuard {
            fn drop(&mut self) {
                CODEC_STACK_SEGMENTS.with(|count| {
                    let (live, total) = count.get();
                    count.set((live - 1, total));
                });
            }
        }
        #[cfg(test)]
        let _guard = {
            CODEC_STACK_SEGMENTS.with(|count| {
                let (live, total) = count.get();
                count.set((live + 1, total + 1));
            });
            SegmentGuard
        };
        return stacker::grow(CODEC_STACK_SEGMENT, f);
    }
    f()
}

// ---- writer / reader --------------------------------------------------------------------------

struct Writer {
    buf: Vec<u8>,
    include_source: bool,
    /// The code being encoded runs at most once per evaluation of the Script: its top-level
    /// statements and the bodies of the run-once functions they call immediately, outside
    /// loops, classes and other functions (see `Function::run_once`).
    run_once: bool,
    /// The function expression an immediate call in run-once code is about to encode.
    run_once_callee: *const Function,
}

impl Writer {
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    /// LEB128 unsigned varint (small counts/tags stay one byte).
    fn uv(&mut self, mut v: u64) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.buf.push(byte);
                break;
            }
            self.buf.push(byte | 0x80);
        }
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.buf.push(v as u8);
    }
    fn str(&mut self, s: &str) {
        self.uv(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    allocated: usize,
    depth: usize,
    /// Set when `buf` is static and function bodies stay encoded (see [`decode_static`]).
    lazy: Option<LazyBodies>,
}

#[derive(Clone, Copy)]
struct LazyBodies {
    /// The bytes `Reader::buf` reads, as the `'static` slice that encoded bodies keep.
    buf: &'static [u8],
    /// Decode each body (and drop it) before leaving it encoded, exactly as [`decode`] would
    /// have decoded it: the process's first decode of these bytes.
    validate: bool,
}

type R<T> = Result<T, String>;

impl Reader<'_> {
    fn u8(&mut self) -> R<u8> {
        let b = *self.buf.get(self.pos).ok_or("snapshot: truncated")?;
        self.pos += 1;
        Ok(b)
    }
    fn uv(&mut self) -> R<u64> {
        let mut result = 0u64;
        let mut shift = 0;
        loop {
            let byte = self.u8()?;
            if shift == 63 && byte & 0x7e != 0 {
                return Err("snapshot: varint overflow".into());
            }
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err("snapshot: varint overflow".into());
            }
        }
    }
    fn u32(&mut self) -> R<u32> {
        let end = self
            .pos
            .checked_add(4)
            .ok_or("snapshot: position overflow")?;
        let bytes = self
            .buf
            .get(self.pos..end)
            .ok_or("snapshot: truncated u32")?;
        self.pos = end;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }
    fn f64(&mut self) -> R<f64> {
        let end = self
            .pos
            .checked_add(8)
            .ok_or("snapshot: position overflow")?;
        let bytes = self
            .buf
            .get(self.pos..end)
            .ok_or("snapshot: truncated f64")?;
        self.pos = end;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }
    fn bool(&mut self) -> R<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("snapshot: invalid boolean".into()),
        }
    }
    fn option(&mut self) -> R<bool> {
        self.bool()
    }
    fn usize(&mut self) -> R<usize> {
        usize::try_from(self.uv()?).map_err(|_| "snapshot: integer too large".into())
    }
    fn charge(&mut self, bytes: usize) -> R<()> {
        self.allocated = self
            .allocated
            .checked_add(bytes)
            .filter(|&total| total <= MAX_SNAPSHOT_ALLOC_BYTES)
            .ok_or("snapshot: allocation budget exceeded")?;
        Ok(())
    }
    fn collection<T>(&mut self) -> R<(usize, Vec<T>)> {
        let len = self.usize()?;
        // Every encoded collection element consumes at least one tag/field byte. This cheap check
        // rejects hostile capacities before even consulting the wider allocation budget.
        if len > self.buf.len().saturating_sub(self.pos) {
            return Err("snapshot: collection length exceeds input".into());
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("snapshot: allocation size overflow")?;
        self.charge(bytes)?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| "snapshot: allocation failed".to_string())?;
        Ok((len, out))
    }
    fn enter_node(&mut self) -> R<()> {
        let depth = self
            .depth
            .checked_add(1)
            .filter(|&depth| depth <= MAX_SNAPSHOT_DEPTH)
            .ok_or("snapshot: nesting limit exceeded")?;
        self.charge(ACCOUNTED_NODE_BYTES)?;
        self.depth = depth;
        Ok(())
    }
    fn leave_node(&mut self) {
        self.depth -= 1;
    }
    fn str(&mut self) -> R<String> {
        let len = self.usize()?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or("snapshot: position overflow")?;
        let bytes = self
            .buf
            .get(self.pos..end)
            .ok_or("snapshot: truncated str")?;
        self.pos = end;
        self.charge(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| "snapshot: bad utf8".into())
    }
    fn rcstr(&mut self) -> R<Rc<str>> {
        let len = self.usize()?;
        let end = self
            .pos
            .checked_add(len)
            .ok_or("snapshot: position overflow")?;
        let bytes = self
            .buf
            .get(self.pos..end)
            .ok_or("snapshot: truncated str")?;
        self.pos = end;
        self.charge(len)?;
        Ok(Rc::from(
            std::str::from_utf8(bytes).map_err(|_| "snapshot: bad utf8")?,
        ))
    }
}

/// Re-intern an operator string to the `&'static str` the parser would have used.
fn intern_op(s: &str) -> R<&'static str> {
    KEYWORDS
        .iter()
        .chain(PUNCTUATORS.iter())
        .find(|k| **k == s)
        .copied()
        .ok_or_else(|| format!("snapshot: unknown operator {s:?}"))
}

// ---- public API -------------------------------------------------------------------------------

/// Encode a parsed script body to a snapshot blob.
pub fn encode(body: &[Stmt]) -> Vec<u8> {
    encode_with_source(body, true)
}

/// Host implementation code has no author-visible source. Keep the executable
/// AST intact, including nested functions/classes created after bootstrap.
pub(crate) fn encode_host(body: &[Stmt]) -> Vec<u8> {
    encode_with_source(body, false)
}

fn encode_with_source(body: &[Stmt], include_source: bool) -> Vec<u8> {
    let perf_started = crate::jit::perf_stage_start();
    let mut w = Writer {
        buf: Vec::with_capacity(body.len() * 32),
        include_source,
        run_once: true,
        run_once_callee: std::ptr::null(),
    };
    w.uv(MAGIC as u64);
    w.uv(VERSION as u64);
    enc_stmts(&mut w, body);
    crate::jit::perf_snapshot_encode_end(perf_started);
    w.buf
}

/// Decode a snapshot blob back into a script body. `Err` (skew/truncation/corruption) tells the
/// caller to fall back to parsing the original source.
pub fn decode(bytes: &[u8]) -> R<Vec<Stmt>> {
    decode_with(bytes, None)
}

/// [`decode`] for a static snapshot, leaving every function body encoded in `bytes` until the
/// function first needs it (see [`FunctionBody`]). The process's first static decode of these
/// bytes also decodes every body, under the same depth, allocation and structure checks as
/// [`decode`], so it accepts and rejects exactly the snapshots [`decode`] does; it drops each body
/// once it has validated it. Later decodes of the same bytes skip the bodies by their lengths.
#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
pub(crate) fn decode_static(bytes: &'static [u8]) -> R<Vec<Stmt>> {
    let validate = !validated_static_snapshot(bytes);
    let body = decode_with(
        bytes,
        Some(LazyBodies {
            buf: bytes,
            validate,
        }),
    )?;
    if validate {
        remember_validated_static_snapshot(bytes);
    }
    Ok(body)
}

fn decode_with(bytes: &[u8], lazy: Option<LazyBodies>) -> R<Vec<Stmt>> {
    let perf_started = crate::jit::perf_stage_start();
    let result = (|| {
        if bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err("snapshot: input exceeds byte limit".into());
        }
        let mut r = Reader {
            buf: bytes,
            pos: 0,
            allocated: 0,
            depth: 0,
            lazy,
        };
        if r.uv()? != MAGIC as u64 {
            return Err("snapshot: bad magic".into());
        }
        if r.uv()? != VERSION as u64 {
            return Err("snapshot: version mismatch".into());
        }
        let body = dec_stmts(&mut r)?;
        if r.pos != bytes.len() {
            return Err("snapshot: trailing bytes".into());
        }
        Ok(body)
    })();
    crate::jit::perf_snapshot_decode_end(perf_started, result.is_ok());
    result
}

/// Decode the statements of a function body that [`decode_static`] left encoded. Its own nested
/// function bodies stay encoded in turn.
pub(crate) fn decode_function_body(encoded: &'static [u8]) -> R<Vec<Stmt>> {
    let mut r = Reader {
        buf: encoded,
        pos: 0,
        allocated: 0,
        depth: 0,
        lazy: Some(LazyBodies {
            buf: encoded,
            validate: false,
        }),
    };
    let body = dec_stmts(&mut r)?;
    if r.pos != encoded.len() {
        return Err("snapshot: function body length mismatch".into());
    }
    Ok(body)
}

#[cfg(test)]
thread_local! {
    /// Function bodies decoded for validation by [`decode_static`] on this thread.
    static VALIDATED_BODIES: Cell<usize> = const { Cell::new(0) };
}

/// Static snapshots this process has decoded with full validation, by address and length.
/// `&'static` bytes are immutable and never freed, so the key identifies their content for the
/// rest of the process. Embedders install a handful of static bootstraps; past the bound, the
/// oldest entry is forgotten and that snapshot is simply validated again on its next decode.
#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
static VALIDATED_STATIC_SNAPSHOTS: std::sync::Mutex<Vec<(usize, usize)>> =
    std::sync::Mutex::new(Vec::new());
#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
const MAX_VALIDATED_STATIC_SNAPSHOTS: usize = 16;

#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
fn static_snapshot_key(bytes: &'static [u8]) -> (usize, usize) {
    (bytes.as_ptr() as usize, bytes.len())
}

#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
fn validated_static_snapshot(bytes: &'static [u8]) -> bool {
    let key = static_snapshot_key(bytes);
    VALIDATED_STATIC_SNAPSHOTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&key)
}

#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
fn remember_validated_static_snapshot(bytes: &'static [u8]) {
    let key = static_snapshot_key(bytes);
    let mut validated = VALIDATED_STATIC_SNAPSHOTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if validated.contains(&key) {
        return;
    }
    if validated.len() == MAX_VALIDATED_STATIC_SNAPSHOTS {
        validated.remove(0);
    }
    validated.push(key);
}

// ---- Vec / Option helpers (monomorphized by hand to keep the reader borrow simple) ------------

fn enc_stmts(w: &mut Writer, v: &[Stmt]) {
    w.uv(v.len() as u64);
    for s in v {
        enc_stmt(w, s);
    }
}
fn dec_stmts(r: &mut Reader) -> R<Vec<Stmt>> {
    let (n, mut out) = r.collection()?;
    for _ in 0..n {
        out.push(dec_stmt(r)?);
    }
    Ok(out)
}

fn enc_exprs(w: &mut Writer, v: &[Expr]) {
    w.uv(v.len() as u64);
    for e in v {
        enc_expr(w, e);
    }
}
fn dec_exprs(r: &mut Reader) -> R<Vec<Expr>> {
    let (n, mut out) = r.collection()?;
    for _ in 0..n {
        out.push(dec_expr(r)?);
    }
    Ok(out)
}

fn enc_opt_expr(w: &mut Writer, o: &Option<Expr>) {
    match o {
        Some(e) => {
            w.u8(1);
            enc_expr(w, e);
        }
        None => w.u8(0),
    }
}
fn dec_opt_expr(r: &mut Reader) -> R<Option<Expr>> {
    Ok(if r.option()? {
        Some(dec_expr(r)?)
    } else {
        None
    })
}

fn enc_opt_str(w: &mut Writer, o: &Option<String>) {
    match o {
        Some(s) => {
            w.u8(1);
            w.str(s);
        }
        None => w.u8(0),
    }
}
fn dec_opt_str(r: &mut Reader) -> R<Option<String>> {
    Ok(if r.option()? { Some(r.str()?) } else { None })
}

fn enc_opt_rcstr(w: &mut Writer, o: &Option<Rc<str>>) {
    match o {
        Some(s) => {
            w.u8(1);
            w.str(s);
        }
        None => w.u8(0),
    }
}
fn dec_opt_rcstr(r: &mut Reader) -> R<Option<Rc<str>>> {
    Ok(if r.option()? { Some(r.rcstr()?) } else { None })
}

// ---- Stmt -------------------------------------------------------------------------------------

fn enc_stmt(w: &mut Writer, s: &Stmt) {
    // A loop's parts can run any number of times per evaluation of the code around it.
    let run_once = w.run_once;
    w.run_once = run_once && s.loop_body().is_none();
    with_codec_stack(|| enc_stmt_inner(w, s));
    w.run_once = run_once;
}

#[inline(never)]
fn enc_stmt_inner(w: &mut Writer, s: &Stmt) {
    match s {
        Stmt::Expr(e) => {
            w.u8(0);
            enc_expr(w, e);
        }
        Stmt::VarDecl { kind, decls } => {
            w.u8(1);
            enc_declkind(w, *kind);
            w.uv(decls.len() as u64);
            for (pat, init) in decls {
                enc_pattern(w, pat);
                enc_opt_expr(w, init);
            }
        }
        Stmt::FuncDecl(f) => {
            w.u8(2);
            enc_function(w, f);
        }
        Stmt::Return(e) => {
            w.u8(3);
            enc_opt_expr(w, e);
        }
        Stmt::If { test, cons, alt } => {
            w.u8(4);
            enc_expr(w, test);
            enc_stmt(w, cons);
            match alt {
                Some(a) => {
                    w.u8(1);
                    enc_stmt(w, a);
                }
                None => w.u8(0),
            }
        }
        Stmt::Block(b) => {
            w.u8(5);
            enc_stmts(w, b);
        }
        Stmt::While { test, body, .. } => {
            w.u8(6);
            enc_expr(w, test);
            enc_stmt(w, body);
        }
        Stmt::DoWhile { body, test, .. } => {
            w.u8(7);
            enc_stmt(w, body);
            enc_expr(w, test);
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            w.u8(8);
            match init {
                Some(fi) => {
                    w.u8(1);
                    enc_forinit(w, fi);
                }
                None => w.u8(0),
            }
            enc_opt_expr(w, test);
            enc_opt_expr(w, update);
            enc_stmt(w, body);
        }
        Stmt::ForInOf {
            decl,
            left,
            right,
            of,
            is_await,
            body,
            ..
        } => {
            w.u8(9);
            match decl {
                Some(k) => {
                    w.u8(1);
                    enc_declkind(w, *k);
                }
                None => w.u8(0),
            }
            enc_pattern(w, left);
            enc_expr(w, right);
            w.bool(*of);
            w.bool(*is_await);
            enc_stmt(w, body);
        }
        Stmt::Break(l) => {
            w.u8(10);
            enc_opt_str(w, l);
        }
        Stmt::Continue(l) => {
            w.u8(11);
            enc_opt_str(w, l);
        }
        Stmt::Throw(e) => {
            w.u8(12);
            enc_expr(w, e);
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            w.u8(13);
            enc_stmts(w, block);
            match handler {
                Some((param, hbody)) => {
                    w.u8(1);
                    match param {
                        Some(p) => {
                            w.u8(1);
                            enc_pattern(w, p);
                        }
                        None => w.u8(0),
                    }
                    enc_stmts(w, hbody);
                }
                None => w.u8(0),
            }
            match finalizer {
                Some(f) => {
                    w.u8(1);
                    enc_stmts(w, f);
                }
                None => w.u8(0),
            }
        }
        Stmt::Switch { disc, cases } => {
            w.u8(14);
            enc_expr(w, disc);
            w.uv(cases.len() as u64);
            for c in cases {
                enc_opt_expr(w, &c.test);
                enc_stmts(w, &c.body);
            }
        }
        Stmt::Labeled { label, body } => {
            w.u8(15);
            w.str(label);
            enc_stmt(w, body);
        }
        Stmt::With { obj, body } => {
            w.u8(16);
            enc_expr(w, obj);
            enc_stmt(w, body);
        }
        Stmt::ClassDecl(c) => {
            w.u8(17);
            enc_class(w, c);
        }
        Stmt::Empty => w.u8(18),
        Stmt::Debugger => w.u8(19),
        Stmt::Import(d) => {
            w.u8(20);
            w.str(&d.source);
            w.uv(d.specs.len() as u64);
            for s in &d.specs {
                enc_importspec(w, s);
            }
            enc_opt_str(w, &d.attr_type);
        }
        Stmt::ExportNamed { specs, source } => {
            w.u8(21);
            w.uv(specs.len() as u64);
            for s in specs {
                w.str(&s.local);
                w.str(&s.exported);
            }
            enc_opt_rcstr(w, source);
        }
        Stmt::ExportDecl(s) => {
            w.u8(22);
            enc_stmt(w, s);
        }
        Stmt::ExportDefault(s) => {
            w.u8(23);
            enc_stmt(w, s);
        }
        Stmt::ExportAll { source, exported } => {
            w.u8(24);
            w.str(source);
            enc_opt_str(w, exported);
        }
    }
}

fn dec_stmt(r: &mut Reader) -> R<Stmt> {
    Ok(*dec_boxed_stmt(r)?)
}

fn dec_boxed_stmt(r: &mut Reader) -> R<Box<Stmt>> {
    r.enter_node()?;
    let result = with_codec_stack(|| dec_stmt_inner(r).map(Box::new));
    r.leave_node();
    result
}

fn dec_boxed_loop_body(r: &mut Reader) -> R<Box<LoopBody>> {
    r.enter_node()?;
    let result = with_codec_stack(|| dec_stmt_inner(r).map(|stmt| Box::new(LoopBody::new(stmt))));
    r.leave_node();
    result
}

#[inline(never)]
fn dec_stmt_inner(r: &mut Reader) -> R<Stmt> {
    Ok(match r.u8()? {
        0 => Stmt::Expr(dec_expr(r)?),
        1 => {
            let kind = dec_declkind(r)?;
            let (n, mut decls) = r.collection()?;
            for _ in 0..n {
                decls.push((dec_pattern(r)?, dec_opt_expr(r)?));
            }
            Stmt::VarDecl { kind, decls }
        }
        2 => Stmt::FuncDecl(Rc::new(dec_function(r)?)),
        3 => Stmt::Return(dec_opt_expr(r)?),
        4 => {
            let test = dec_expr(r)?;
            let cons = dec_boxed_stmt(r)?;
            let alt = if r.option()? {
                Some(dec_boxed_stmt(r)?)
            } else {
                None
            };
            Stmt::If { test, cons, alt }
        }
        5 => Stmt::Block(dec_stmts(r)?),
        6 => Stmt::While {
            test: dec_expr(r)?,
            body: dec_boxed_loop_body(r)?,
        },
        7 => Stmt::DoWhile {
            body: dec_boxed_loop_body(r)?,
            test: dec_expr(r)?,
        },
        8 => {
            let init = if r.option()? {
                Some(Box::new(dec_forinit(r)?))
            } else {
                None
            };
            Stmt::For {
                init,
                test: dec_opt_expr(r)?,
                update: dec_opt_expr(r)?,
                body: dec_boxed_loop_body(r)?,
            }
        }
        9 => {
            let decl = if r.option()? {
                Some(dec_declkind(r)?)
            } else {
                None
            };
            Stmt::ForInOf {
                decl,
                left: dec_pattern(r)?,
                right: dec_expr(r)?,
                of: r.bool()?,
                is_await: r.bool()?,
                body: dec_boxed_loop_body(r)?,
            }
        }
        10 => Stmt::Break(dec_opt_str(r)?),
        11 => Stmt::Continue(dec_opt_str(r)?),
        12 => Stmt::Throw(dec_expr(r)?),
        13 => {
            let block = dec_stmts(r)?;
            let handler = if r.option()? {
                let param = if r.option()? {
                    Some(dec_pattern(r)?)
                } else {
                    None
                };
                Some((param, dec_stmts(r)?))
            } else {
                None
            };
            let finalizer = if r.option()? {
                Some(dec_stmts(r)?)
            } else {
                None
            };
            Stmt::Try {
                block,
                handler,
                finalizer,
            }
        }
        14 => {
            let disc = dec_expr(r)?;
            let (n, mut cases) = r.collection()?;
            for _ in 0..n {
                cases.push(SwitchCase {
                    test: dec_opt_expr(r)?,
                    body: dec_stmts(r)?,
                });
            }
            Stmt::Switch { disc, cases }
        }
        15 => Stmt::Labeled {
            label: r.str()?,
            body: dec_boxed_stmt(r)?,
        },
        16 => Stmt::With {
            obj: dec_expr(r)?,
            body: dec_boxed_stmt(r)?,
        },
        17 => Stmt::ClassDecl(Rc::new(dec_class(r)?)),
        18 => Stmt::Empty,
        19 => Stmt::Debugger,
        20 => {
            let source = r.rcstr()?;
            let (n, mut specs) = r.collection()?;
            for _ in 0..n {
                specs.push(dec_importspec(r)?);
            }
            Stmt::Import(ImportDecl {
                source,
                specs,
                attr_type: dec_opt_str(r)?,
            })
        }
        21 => {
            let (n, mut specs) = r.collection()?;
            for _ in 0..n {
                specs.push(ExportSpec {
                    local: r.str()?,
                    exported: r.str()?,
                });
            }
            Stmt::ExportNamed {
                specs,
                source: dec_opt_rcstr(r)?,
            }
        }
        22 => Stmt::ExportDecl(dec_boxed_stmt(r)?),
        23 => Stmt::ExportDefault(dec_boxed_stmt(r)?),
        24 => Stmt::ExportAll {
            source: r.rcstr()?,
            exported: dec_opt_str(r)?,
        },
        t => return Err(format!("snapshot: bad Stmt tag {t}")),
    })
}

// ---- Expr -------------------------------------------------------------------------------------

fn enc_expr(w: &mut Writer, e: &Expr) {
    with_codec_stack(|| enc_expr_inner(w, e));
}

#[inline(never)]
fn enc_expr_inner(w: &mut Writer, e: &Expr) {
    match e {
        Expr::Paren(x) => {
            w.u8(0);
            enc_expr(w, x);
        }
        Expr::Num(n) => {
            w.u8(1);
            w.f64(*n);
        }
        Expr::BigInt(b) => {
            w.u8(2);
            w.str(&b.to_string_radix(16));
        }
        Expr::Str(s) => {
            w.u8(3);
            w.str(s);
        }
        Expr::ToStr(x) => {
            w.u8(4);
            enc_expr(w, x);
        }
        Expr::Bool(b) => {
            w.u8(5);
            w.bool(*b);
        }
        Expr::Null => w.u8(6),
        Expr::Undefined => w.u8(7),
        Expr::Ident(n, _) => {
            w.u8(8);
            w.str(n);
        }
        Expr::This => w.u8(9),
        Expr::Regex { body, flags } => {
            w.u8(10);
            w.str(body);
            w.str(flags);
        }
        Expr::Array(elems) => {
            w.u8(11);
            enc_array_elems(w, elems);
        }
        Expr::Object(props) => {
            w.u8(12);
            w.uv(props.len() as u64);
            for p in props {
                enc_propdef(w, p);
            }
        }
        Expr::Func(f) => {
            w.u8(13);
            enc_function(w, f);
        }
        Expr::Class(c) => {
            w.u8(14);
            enc_class(w, c);
        }
        Expr::Yield { delegate, arg } => {
            w.u8(15);
            w.bool(*delegate);
            match arg {
                Some(a) => {
                    w.u8(1);
                    enc_expr(w, a);
                }
                None => w.u8(0),
            }
        }
        Expr::Await(x) => {
            w.u8(16);
            enc_expr(w, x);
        }
        Expr::Super => w.u8(17),
        Expr::Unary { op, arg } => {
            w.u8(18);
            w.str(op);
            enc_expr(w, arg);
        }
        Expr::Update { op, prefix, arg } => {
            w.u8(19);
            w.str(op);
            w.bool(*prefix);
            enc_expr(w, arg);
        }
        Expr::Binary { op, left, right } => {
            w.u8(20);
            w.str(op);
            enc_expr(w, left);
            enc_expr(w, right);
        }
        Expr::Logical { op, left, right } => {
            w.u8(21);
            w.str(op);
            enc_expr(w, left);
            enc_expr(w, right);
        }
        Expr::Assign {
            op, target, value, ..
        } => {
            w.u8(22);
            w.str(op);
            enc_expr(w, target);
            enc_expr(w, value);
        }
        Expr::Cond { test, cons, alt } => {
            w.u8(23);
            enc_expr(w, test);
            enc_expr(w, cons);
            enc_expr(w, alt);
        }
        Expr::Call {
            callee,
            args,
            optional,
            ..
        } => {
            w.u8(24);
            // `(function () { ... })()` in run-once code: the function object the callee
            // creates is called here, once, and nowhere else.
            let mut immediate = &**callee;
            while let Expr::Paren(inner) = immediate {
                immediate = inner;
            }
            if let (true, Expr::Func(function)) = (w.run_once, immediate) {
                w.run_once_callee = Rc::as_ptr(function);
            }
            enc_expr(w, callee);
            enc_array_elems(w, args);
            w.bool(*optional);
        }
        Expr::New { callee, args, .. } => {
            w.u8(25);
            enc_expr(w, callee);
            enc_array_elems(w, args);
        }
        Expr::Member {
            obj,
            prop,
            optional,
            ..
        } => {
            w.u8(26);
            enc_expr(w, obj);
            w.str(prop);
            w.bool(*optional);
        }
        Expr::Index {
            obj,
            index,
            optional,
            ..
        } => {
            w.u8(27);
            enc_expr(w, obj);
            enc_expr(w, index);
            w.bool(*optional);
        }
        Expr::Seq(exprs) => {
            w.u8(28);
            enc_exprs(w, exprs);
        }
        Expr::TaggedTemplate {
            tag,
            site: _,
            quasis,
            subs,
            ..
        } => {
            w.u8(29);
            enc_expr(w, tag);
            w.uv(quasis.len() as u64);
            for (cooked, raw) in quasis {
                enc_opt_str(w, cooked);
                w.str(raw);
            }
            enc_exprs(w, subs);
        }
        Expr::OptionalChain(x) => {
            w.u8(30);
            enc_expr(w, x);
        }
        Expr::PrivateIn { name, obj } => {
            w.u8(31);
            w.str(name);
            enc_expr(w, obj);
        }
        Expr::ImportCall {
            spec,
            phase,
            options,
        } => {
            w.u8(32);
            enc_expr(w, spec);
            w.u8(match phase {
                ImportPhase::Evaluation => 0,
                ImportPhase::Source => 1,
                ImportPhase::Defer => 2,
            });
            match options {
                Some(o) => {
                    w.u8(1);
                    enc_expr(w, o);
                }
                None => w.u8(0),
            }
        }
        Expr::ImportMeta => w.u8(33),
        Expr::NewTarget => w.u8(34),
    }
}

fn dec_expr(r: &mut Reader) -> R<Expr> {
    Ok(*dec_boxed_expr(r)?)
}

fn dec_boxed_expr(r: &mut Reader) -> R<Box<Expr>> {
    r.enter_node()?;
    let result = with_codec_stack(|| dec_expr_inner(r).map(Box::new));
    r.leave_node();
    result
}

#[inline(never)]
fn dec_expr_inner(r: &mut Reader) -> R<Expr> {
    Ok(match r.u8()? {
        0 => Expr::Paren(dec_boxed_expr(r)?),
        1 => Expr::Num(r.f64()?),
        2 => Expr::BigInt(JsBigInt::parse_radix(&r.str()?, 16).ok_or("snapshot: bad bigint")?),
        3 => Expr::Str(r.rcstr()?),
        4 => Expr::ToStr(dec_boxed_expr(r)?),
        5 => Expr::Bool(r.bool()?),
        6 => Expr::Null,
        7 => Expr::Undefined,
        8 => Expr::Ident(r.str()?, NO_POSITION),
        9 => Expr::This,
        10 => Expr::Regex {
            body: r.rcstr()?,
            flags: r.rcstr()?,
        },
        11 => Expr::Array(dec_array_elems(r)?),
        12 => {
            let (n, mut props) = r.collection()?;
            for _ in 0..n {
                props.push(dec_propdef(r)?);
            }
            Expr::Object(props)
        }
        13 => Expr::Func(Rc::new(dec_function(r)?)),
        14 => Expr::Class(Rc::new(dec_class(r)?)),
        15 => {
            let delegate = r.bool()?;
            let arg = if r.option()? {
                Some(dec_boxed_expr(r)?)
            } else {
                None
            };
            Expr::Yield { delegate, arg }
        }
        16 => Expr::Await(dec_boxed_expr(r)?),
        17 => Expr::Super,
        18 => Expr::Unary {
            op: intern_op(&r.str()?)?,
            arg: dec_boxed_expr(r)?,
        },
        19 => Expr::Update {
            op: intern_op(&r.str()?)?,
            prefix: r.bool()?,
            arg: dec_boxed_expr(r)?,
        },
        20 => Expr::Binary {
            op: intern_op(&r.str()?)?,
            left: dec_boxed_expr(r)?,
            right: dec_boxed_expr(r)?,
        },
        21 => Expr::Logical {
            op: intern_op(&r.str()?)?,
            left: dec_boxed_expr(r)?,
            right: dec_boxed_expr(r)?,
        },
        22 => Expr::Assign {
            op: intern_op(&r.str()?)?,
            target: dec_boxed_expr(r)?,
            value: dec_boxed_expr(r)?,
            pos: NO_POSITION,
        },
        23 => Expr::Cond {
            test: dec_boxed_expr(r)?,
            cons: dec_boxed_expr(r)?,
            alt: dec_boxed_expr(r)?,
        },
        24 => Expr::Call {
            callee: dec_boxed_expr(r)?,
            args: dec_array_elems(r)?,
            optional: r.bool()?,
            pos: NO_POSITION,
        },
        25 => Expr::New {
            callee: dec_boxed_expr(r)?,
            args: dec_array_elems(r)?,
            pos: NO_POSITION,
        },
        26 => Expr::Member {
            obj: dec_boxed_expr(r)?,
            prop: r.str()?,
            optional: r.bool()?,
            pos: NO_POSITION,
        },
        27 => Expr::Index {
            obj: dec_boxed_expr(r)?,
            index: dec_boxed_expr(r)?,
            optional: r.bool()?,
            pos: NO_POSITION,
        },
        28 => Expr::Seq(dec_exprs(r)?),
        29 => {
            let tag = dec_boxed_expr(r)?;
            // Decoding constructs a fresh Parse Node, so it receives a fresh identity just as a
            // normal parse would. The site identity is intentionally not serialized.
            let site = crate::parser::next_template_site_id();
            let (n, mut quasis) = r.collection()?;
            for _ in 0..n {
                quasis.push((dec_opt_str(r)?, r.str()?));
            }
            Expr::TaggedTemplate {
                tag,
                site,
                quasis,
                subs: dec_exprs(r)?.into_boxed_slice(),
                pos: NO_POSITION,
            }
        }
        30 => Expr::OptionalChain(dec_boxed_expr(r)?),
        31 => Expr::PrivateIn {
            name: r.str()?,
            obj: dec_boxed_expr(r)?,
        },
        32 => Expr::ImportCall {
            spec: dec_boxed_expr(r)?,
            phase: match r.u8()? {
                0 => ImportPhase::Evaluation,
                1 => ImportPhase::Source,
                2 => ImportPhase::Defer,
                t => return Err(format!("snapshot: bad ImportPhase {t}")),
            },
            options: if r.option()? {
                Some(dec_boxed_expr(r)?)
            } else {
                None
            },
        },
        33 => Expr::ImportMeta,
        34 => Expr::NewTarget,
        t => return Err(format!("snapshot: bad Expr tag {t}")),
    })
}

// ---- shared sub-structures --------------------------------------------------------------------

fn enc_array_elems(w: &mut Writer, elems: &[ArrayElem]) {
    w.uv(elems.len() as u64);
    for el in elems {
        match el {
            ArrayElem::Item(e) => {
                w.u8(0);
                enc_expr(w, e);
            }
            ArrayElem::Spread(e) => {
                w.u8(1);
                enc_expr(w, e);
            }
            ArrayElem::Hole => w.u8(2),
        }
    }
}
fn dec_array_elems(r: &mut Reader) -> R<Vec<ArrayElem>> {
    let (n, mut out) = r.collection()?;
    for _ in 0..n {
        out.push(match r.u8()? {
            0 => ArrayElem::Item(dec_expr(r)?),
            1 => ArrayElem::Spread(dec_expr(r)?),
            2 => ArrayElem::Hole,
            t => return Err(format!("snapshot: bad ArrayElem {t}")),
        });
    }
    Ok(out)
}

fn enc_propdef(w: &mut Writer, p: &PropDef) {
    match p {
        PropDef::KeyValue { key, value } => {
            w.u8(0);
            enc_propkey(w, key);
            enc_expr(w, value);
        }
        PropDef::Cover { key, value } => {
            w.u8(1);
            enc_propkey(w, key);
            enc_expr(w, value);
        }
        PropDef::Method { key, func } => {
            w.u8(2);
            enc_propkey(w, key);
            enc_function(w, func);
        }
        PropDef::Getter { key, func } => {
            w.u8(3);
            enc_propkey(w, key);
            enc_function(w, func);
        }
        PropDef::Setter { key, func } => {
            w.u8(4);
            enc_propkey(w, key);
            enc_function(w, func);
        }
        PropDef::Spread(e) => {
            w.u8(5);
            enc_expr(w, e);
        }
        PropDef::Proto(e) => {
            w.u8(6);
            enc_expr(w, e);
        }
    }
}
fn dec_propdef(r: &mut Reader) -> R<PropDef> {
    Ok(match r.u8()? {
        0 => PropDef::KeyValue {
            key: dec_propkey(r)?,
            value: dec_expr(r)?,
        },
        1 => PropDef::Cover {
            key: dec_propkey(r)?,
            value: dec_expr(r)?,
        },
        2 => PropDef::Method {
            key: dec_propkey(r)?,
            func: Rc::new(dec_function(r)?),
        },
        3 => PropDef::Getter {
            key: dec_propkey(r)?,
            func: Rc::new(dec_function(r)?),
        },
        4 => PropDef::Setter {
            key: dec_propkey(r)?,
            func: Rc::new(dec_function(r)?),
        },
        5 => PropDef::Spread(dec_expr(r)?),
        6 => PropDef::Proto(dec_expr(r)?),
        t => return Err(format!("snapshot: bad PropDef {t}")),
    })
}

fn enc_propkey(w: &mut Writer, k: &PropKey) {
    match k {
        PropKey::Ident(n) => {
            w.u8(0);
            w.str(n);
        }
        PropKey::Str(s) => {
            w.u8(1);
            w.str(s);
        }
        PropKey::Num(n) => {
            w.u8(2);
            w.f64(*n);
        }
        PropKey::Computed(e) => {
            w.u8(3);
            enc_expr(w, e);
        }
    }
}
fn dec_propkey(r: &mut Reader) -> R<PropKey> {
    Ok(match r.u8()? {
        0 => PropKey::Ident(r.str()?),
        1 => PropKey::Str(r.rcstr()?),
        2 => PropKey::Num(r.f64()?),
        3 => PropKey::Computed(dec_expr(r)?),
        t => return Err(format!("snapshot: bad PropKey {t}")),
    })
}

fn enc_pattern(w: &mut Writer, p: &Pattern) {
    with_codec_stack(|| enc_pattern_inner(w, p));
}

#[inline(never)]
fn enc_pattern_inner(w: &mut Writer, p: &Pattern) {
    match p {
        Pattern::Ident(n) => {
            w.u8(0);
            w.str(n);
        }
        Pattern::Array(elems) => {
            w.u8(1);
            w.uv(elems.len() as u64);
            for el in elems {
                match el {
                    ArrayPatElem::Hole => w.u8(0),
                    ArrayPatElem::Elem { pattern, default } => {
                        w.u8(1);
                        enc_pattern(w, pattern);
                        enc_opt_expr(w, default);
                    }
                    ArrayPatElem::Rest(p) => {
                        w.u8(2);
                        enc_pattern(w, p);
                    }
                }
            }
        }
        Pattern::Object(op) => {
            w.u8(2);
            w.uv(op.props.len() as u64);
            for prop in &op.props {
                enc_propkey(w, &prop.key);
                enc_pattern(w, &prop.value);
                enc_opt_expr(w, &prop.default);
            }
            enc_opt_str(w, &op.rest);
        }
        Pattern::Member(e) => {
            w.u8(3);
            enc_expr(w, e);
        }
    }
}
fn dec_pattern(r: &mut Reader) -> R<Pattern> {
    r.enter_node()?;
    let result = with_codec_stack(|| dec_pattern_inner(r));
    r.leave_node();
    result
}

#[inline(never)]
fn dec_pattern_inner(r: &mut Reader) -> R<Pattern> {
    Ok(match r.u8()? {
        0 => Pattern::Ident(r.str()?),
        1 => {
            let (n, mut elems) = r.collection()?;
            for _ in 0..n {
                elems.push(match r.u8()? {
                    0 => ArrayPatElem::Hole,
                    1 => ArrayPatElem::Elem {
                        pattern: dec_pattern(r)?,
                        default: dec_opt_expr(r)?,
                    },
                    2 => ArrayPatElem::Rest(dec_pattern(r)?),
                    t => return Err(format!("snapshot: bad ArrayPatElem {t}")),
                });
            }
            Pattern::Array(elems)
        }
        2 => {
            let (n, mut props) = r.collection()?;
            for _ in 0..n {
                props.push(ObjPatProp {
                    key: dec_propkey(r)?,
                    value: dec_pattern(r)?,
                    default: dec_opt_expr(r)?,
                });
            }
            Pattern::Object(ObjectPat {
                props,
                rest: dec_opt_str(r)?,
            })
        }
        3 => Pattern::Member(dec_boxed_expr(r)?),
        t => return Err(format!("snapshot: bad Pattern {t}")),
    })
}

fn enc_function(w: &mut Writer, f: &Function) {
    let run_once = std::mem::replace(&mut w.run_once_callee, std::ptr::null()) == f;
    let outer = std::mem::replace(&mut w.run_once, run_once);
    with_codec_stack(|| enc_function_inner(w, f, run_once));
    w.run_once = outer;
}

#[inline(never)]
fn enc_function_inner(w: &mut Writer, f: &Function, run_once: bool) {
    enc_opt_str(w, &f.name);
    w.uv(f.params.len() as u64);
    for p in &f.params {
        enc_pattern(w, &p.pattern);
        enc_opt_expr(w, &p.default);
        w.bool(p.rest);
    }
    enc_function_body(w, &f.body);
    // Flags packed into one byte.
    let flags = (f.is_arrow as u8)
        | (f.is_strict as u8) << 1
        | (f.expr_body as u8) << 2
        | (f.is_generator as u8) << 3
        | (f.is_async as u8) << 4
        | (f.is_method as u8) << 5
        | (f.is_fn_expr as u8) << 6
        | (f.default_ctor as u8) << 7;
    w.u8(flags);
    w.u8(f.self_hosted as u8 | (run_once as u8) << 1);
    if w.include_source {
        enc_opt_rcstr(w, &f.source);
    } else {
        w.u8(0);
    }
}
fn dec_function(r: &mut Reader) -> R<Function> {
    r.enter_node()?;
    let result = with_codec_stack(|| dec_function_inner(r));
    r.leave_node();
    result
}

#[inline(never)]
fn dec_function_inner(r: &mut Reader) -> R<Function> {
    let name = dec_opt_str(r)?;
    let (n, mut params) = r.collection()?;
    for _ in 0..n {
        params.push(Param {
            pattern: dec_pattern(r)?,
            default: dec_opt_expr(r)?,
            rest: r.bool()?,
        });
    }
    let body = dec_function_body(r)?;
    let flags = r.u8()?;
    let more_flags = r.u8()?;
    if more_flags > 3 {
        return Err("snapshot: invalid function flags".into());
    }
    let self_hosted = more_flags & 1 != 0;
    // Only a static snapshot's bootstrap runs in each Realm that evaluates it.
    let run_once = more_flags & 2 != 0 && r.lazy.is_some();
    let source = dec_opt_rcstr(r)?;
    Ok(Function {
        name,
        params,
        body,
        is_arrow: flags & 1 != 0,
        is_strict: flags & 2 != 0,
        expr_body: flags & 4 != 0,
        is_generator: flags & 8 != 0,
        is_async: flags & 16 != 0,
        is_method: flags & 32 != 0,
        is_fn_expr: flags & 64 != 0,
        default_ctor: flags & 128 != 0,
        self_hosted,
        run_once,
        source,
        // A snapshot keeps no source positions: its frames print like built-ins (or not at all).
        script: None,
        start: 0,
        // Lazy runtime caches — start empty, exactly as the parser leaves them.
        scan: Cell::new(0),
        hoist: OnceCell::new(),
        calls: Cell::new(0),
        code: OnceCell::new(),
        code2: OnceCell::new(),
        fn_maps: OnceCell::new(),
        env_layouts: OnceCell::new(),
    })
}

/// A function body: its statements' byte length as a little-endian `u32`, then the statements.
fn enc_function_body(w: &mut Writer, body: &FunctionBody) {
    let at = w.buf.len();
    w.buf.extend_from_slice(&[0; 4]);
    // A lazily decoded body is re-encoded from a temporary decode, as `include_source` may
    // differ from the encoding it was read from.
    body.with(|stmts| enc_stmts(w, stmts));
    // No representable snapshot holds a 4 GiB body; an impossible length fails decoding.
    let len = u32::try_from(w.buf.len() - at - 4).unwrap_or(u32::MAX);
    w.buf[at..at + 4].copy_from_slice(&len.to_le_bytes());
}

fn dec_function_body(r: &mut Reader) -> R<FunctionBody> {
    let len = usize::try_from(r.u32()?).map_err(|_| "snapshot: integer too large")?;
    let start = r.pos;
    let end = start
        .checked_add(len)
        .filter(|&end| end <= r.buf.len())
        .ok_or("snapshot: truncated function body")?;
    let Some(lazy) = r.lazy else {
        let body = dec_stmts(r)?;
        if r.pos != end {
            return Err("snapshot: function body length mismatch".into());
        }
        return Ok(FunctionBody::new(body));
    };
    if lazy.validate {
        // Exactly the decode `decode` performs, charged to the same budgets.
        let body = dec_stmts(r)?;
        if r.pos != end {
            return Err("snapshot: function body length mismatch".into());
        }
        drop(body);
        #[cfg(test)]
        VALIDATED_BODIES.with(|count| count.set(count.get() + 1));
    } else {
        r.pos = end;
    }
    Ok(FunctionBody::encoded(&lazy.buf[start..end]))
}

fn enc_class(w: &mut Writer, c: &Class) {
    // Class elements run per instance or per class evaluation.
    let run_once = std::mem::replace(&mut w.run_once, false);
    with_codec_stack(|| enc_class_inner(w, c));
    w.run_once = run_once;
}

#[inline(never)]
fn enc_class_inner(w: &mut Writer, c: &Class) {
    enc_opt_str(w, &c.name);
    match &c.superclass {
        Some(sc) => {
            w.u8(1);
            enc_expr(w, sc);
        }
        None => w.u8(0),
    }
    w.uv(c.members.len() as u64);
    for m in &c.members {
        enc_propkey(w, &m.key);
        w.u8(match m.kind {
            MemberKind::Constructor => 0,
            MemberKind::Method => 1,
            MemberKind::Get => 2,
            MemberKind::Set => 3,
            MemberKind::Field => 4,
            MemberKind::Accessor => 5,
            MemberKind::StaticBlock => 6,
        });
        w.bool(m.is_static);
        match &m.func {
            Some(f) => {
                w.u8(1);
                enc_function(w, f);
            }
            None => w.u8(0),
        }
        enc_opt_expr(w, &m.value);
        enc_exprs(w, &m.decorators);
    }
    enc_exprs(w, &c.decorators);
    if w.include_source {
        enc_opt_rcstr(w, &c.source);
    } else {
        w.u8(0);
    }
}
fn dec_class(r: &mut Reader) -> R<Class> {
    r.enter_node()?;
    let result = with_codec_stack(|| dec_class_inner(r));
    r.leave_node();
    result
}

#[inline(never)]
fn dec_class_inner(r: &mut Reader) -> R<Class> {
    let name = dec_opt_str(r)?;
    let superclass = if r.option()? {
        Some(dec_boxed_expr(r)?)
    } else {
        None
    };
    let (n, mut members) = r.collection()?;
    for _ in 0..n {
        let key = dec_propkey(r)?;
        let kind = match r.u8()? {
            0 => MemberKind::Constructor,
            1 => MemberKind::Method,
            2 => MemberKind::Get,
            3 => MemberKind::Set,
            4 => MemberKind::Field,
            5 => MemberKind::Accessor,
            6 => MemberKind::StaticBlock,
            t => return Err(format!("snapshot: bad MemberKind {t}")),
        };
        let is_static = r.bool()?;
        let func = if r.option()? {
            Some(Rc::new(dec_function(r)?))
        } else {
            None
        };
        members.push(ClassMember {
            key,
            kind,
            is_static,
            func,
            value: dec_opt_expr(r)?,
            decorators: dec_exprs(r)?,
        });
    }
    Ok(Class {
        name,
        superclass,
        members,
        decorators: dec_exprs(r)?,
        source: dec_opt_rcstr(r)?,
        script: None,
        start: 0,
    })
}

fn enc_forinit(w: &mut Writer, fi: &ForInit) {
    match fi {
        ForInit::VarDecl { kind, decls } => {
            w.u8(0);
            enc_declkind(w, *kind);
            w.uv(decls.len() as u64);
            for (pat, init) in decls {
                enc_pattern(w, pat);
                enc_opt_expr(w, init);
            }
        }
        ForInit::Expr(e) => {
            w.u8(1);
            enc_expr(w, e);
        }
    }
}
fn dec_forinit(r: &mut Reader) -> R<ForInit> {
    Ok(match r.u8()? {
        0 => {
            let kind = dec_declkind(r)?;
            let (n, mut decls) = r.collection()?;
            for _ in 0..n {
                decls.push((dec_pattern(r)?, dec_opt_expr(r)?));
            }
            ForInit::VarDecl { kind, decls }
        }
        1 => ForInit::Expr(dec_expr(r)?),
        t => return Err(format!("snapshot: bad ForInit {t}")),
    })
}

fn enc_importspec(w: &mut Writer, s: &ImportSpec) {
    match s {
        ImportSpec::Default(n) => {
            w.u8(0);
            w.str(n);
        }
        ImportSpec::Namespace(n) => {
            w.u8(1);
            w.str(n);
        }
        ImportSpec::DeferNamespace(n) => {
            w.u8(2);
            w.str(n);
        }
        ImportSpec::Source(n) => {
            w.u8(3);
            w.str(n);
        }
        ImportSpec::Named { imported, local } => {
            w.u8(4);
            w.str(imported);
            w.str(local);
        }
    }
}
fn dec_importspec(r: &mut Reader) -> R<ImportSpec> {
    Ok(match r.u8()? {
        0 => ImportSpec::Default(r.str()?),
        1 => ImportSpec::Namespace(r.str()?),
        2 => ImportSpec::DeferNamespace(r.str()?),
        3 => ImportSpec::Source(r.str()?),
        4 => ImportSpec::Named {
            imported: r.str()?,
            local: r.str()?,
        },
        t => return Err(format!("snapshot: bad ImportSpec {t}")),
    })
}

fn enc_declkind(w: &mut Writer, k: DeclKind) {
    w.u8(match k {
        DeclKind::Var => 0,
        DeclKind::Let => 1,
        DeclKind::Const => 2,
        DeclKind::Using => 3,
        DeclKind::AwaitUsing => 4,
    });
}
fn dec_declkind(r: &mut Reader) -> R<DeclKind> {
    Ok(match r.u8()? {
        0 => DeclKind::Var,
        1 => DeclKind::Let,
        2 => DeclKind::Const,
        3 => DeclKind::Using,
        4 => DeclKind::AwaitUsing,
        t => return Err(format!("snapshot: bad DeclKind {t}")),
    })
}

#[cfg(test)]
mod tests {
    use super::{decode, decode_static, encode, Writer, MAGIC, VERSION};
    use crate::ast::{Expr, Function, Stmt};
    use std::rc::Rc;

    fn snapshot_writer() -> Writer {
        let mut writer = Writer {
            buf: Vec::new(),
            include_source: true,
            run_once: true,
            run_once_callee: std::ptr::null(),
        };
        writer.uv(MAGIC as u64);
        writer.uv(VERSION as u64);
        writer
    }

    /// `encode` and `decode` must be exact inverses: re-encoding a decoded tree reproduces the
    /// bytes. This catches every tag/field-order asymmetry (a *dropped* field is caught instead
    /// by the behavioral suites that run the decoded glue).
    fn assert_roundtrips(src: &str) {
        let blob = crate::compile_snapshot(src).unwrap_or_else(|e| panic!("compile {src:?}: {e}"));
        let ast = decode(&blob).unwrap_or_else(|e| panic!("decode {src:?}: {e}"));
        assert_eq!(blob, encode(&ast), "re-encode differs for: {src}");
    }

    #[test]
    fn roundtrip_diverse_constructs() {
        for src in [
            "1; 'two'; true; null; undefined; 0xffn; 1.5e10;",
            "let { a, b: [c, ...d] = [], ...rest } = obj; const [x = 1, , z] = arr;",
            "class C extends B { #f = 1; static s = 2; get g(){return this.#f} set g(v){} accessor a; static { init(); } has(o){ return #f in o } ['x'+y]() {} }",
            "async function* gen(a, b = 1, ...c) { yield* a; await b; for await (const x of c) {} }",
            "const f = (x) => x ? a?.b.c?.() : `t${x}${'raw'}`; tag`a${1}b`;",
            "try { throw new E(); } catch { } finally { } with (o) { o.p = 1; } label: for (k in o) break label;",
            "a ??= b; c ||= d; e &&= f; g **= h; -x; !y; typeof z; void 0; delete o.p; a in b; a instanceof B;",
            "switch (x) { case 1: break; default: } do { i++ } while (i < 10); function ctor(){ return new.target; }",
            "const p = import('m'); const q = import('m', { with: { type: 'json' } });",
            "obj = { __proto__: p, shorthand, key: v, [comp]: w, m() {}, get g() {}, *gen() {}, async am() {}, ...spread };",
        ] {
            assert_roundtrips(src);
        }
    }

    #[test]
    fn roundtrip_real_web_glue() {
        // The actual runtime glue — the thing the build-time snapshot will encode.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../lumen-web/src/js/");
        for file in [
            "events.js",
            "encoding.js",
            "url.js",
            "streams.js",
            "fetch.js",
            "server.js",
            "crypto.js",
        ] {
            let src = std::fs::read_to_string(format!("{dir}{file}")).unwrap();
            assert_roundtrips(&src);
        }
    }

    #[test]
    fn corrupt_lengths_and_varints_fail_before_allocation() {
        let mut huge_collection = snapshot_writer();
        huge_collection.uv(u64::MAX);
        assert!(decode(&huge_collection.buf)
            .unwrap_err()
            .contains("collection length"));

        let mut overflowing_varint = snapshot_writer();
        overflowing_varint.buf.extend_from_slice(&[0xff; 9]);
        overflowing_varint.buf.push(2);
        assert!(decode(&overflowing_varint.buf)
            .unwrap_err()
            .contains("varint overflow"));
    }

    #[test]
    fn corrupt_structure_is_strictly_bounded() {
        let mut trailing = encode(&[]);
        trailing.push(0);
        assert!(decode(&trailing).unwrap_err().contains("trailing bytes"));

        let mut invalid_bool = snapshot_writer();
        invalid_bool.uv(1); // one statement
        invalid_bool.u8(0); // expression statement
        invalid_bool.u8(5); // boolean expression
        invalid_bool.u8(2); // invalid Boolean encoding
        assert!(decode(&invalid_bool.buf)
            .unwrap_err()
            .contains("invalid boolean"));

        let mut nested = snapshot_writer();
        nested.uv(1); // one statement
        nested.u8(0); // expression statement
        for _ in 0..600 {
            nested.u8(0); // parenthesized expression
        }
        nested.u8(6); // null
        assert!(decode(&nested.buf).unwrap_err().contains("nesting limit"));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn codec_reentered_on_a_small_native_stack_roundtrips() {
        // A host may decode a child Realm's bootstrap while its caller's JS contexts are
        // suspended. Model that occupied stack without requiring a browser or generated code.
        // The snapshot is valid and remains below the existing codec depth budget.
        let mut writer = snapshot_writer();
        writer.uv(1);
        writer.u8(0); // expression statement
        for _ in 0..96 {
            writer.u8(20); // binary expression
            writer.str("+");
        }
        writer.u8(1);
        writer.f64(1.0);
        for _ in 0..96 {
            writer.u8(1);
            writer.f64(2.0);
        }
        #[inline(never)]
        fn reentered(bytes: Vec<u8>) {
            let occupied = [0x7bu8; 64 * 1024];
            std::hint::black_box(&occupied);
            let ast = decode(&bytes).expect("valid nested snapshot");
            assert_eq!(encode(&ast), bytes);
            drop(ast);
            assert_eq!(std::hint::black_box(&occupied)[0], 0x7b);
            super::CODEC_STACK_SEGMENTS.with(|count| {
                let (live, total) = count.get();
                assert_eq!(live, 0, "all codec stack continuations returned");
                if cfg!(debug_assertions) {
                    assert!(total > 0, "the debug regression must exercise stack growth");
                }
            });
        }
        std::thread::Builder::new()
            .name("snapshot-reentered".into())
            .stack_size(512 * 1024)
            // A pthread implementation may reuse a larger cached stack for a smaller
            // request. An explicit segment makes the occupied-headroom regression
            // deterministic even after other tests have created large-stack threads.
            .spawn(move || stacker::grow(512 * 1024, || reentered(writer.buf)))
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn codec_recursive_gateways_and_corrupt_inputs_release_small_stacks() {
        // Exercise every mutually recursive gateway, not just the Binary edge in the browser
        // reproducer. Build deep fixtures iteratively so parser recursion is not under test.
        use crate::ast::{ArrayPatElem, Class, DeclKind, Expr, Pattern, Stmt};
        use std::rc::Rc;
        let mut expr = Expr::Num(1.0);
        let mut stmt = Stmt::Empty;
        let mut pattern = Pattern::Ident("x".into());
        for _ in 0..96 {
            expr = Expr::Paren(Box::new(expr));
            stmt = Stmt::Block(vec![stmt]);
            pattern = Pattern::Array(vec![ArrayPatElem::Rest(pattern)]);
        }
        let function_template = crate::parser::parse_script("function f() {}", false)
            .ok()
            .expect("function template parses");
        let Stmt::FuncDecl(template) = &function_template[0] else {
            panic!("function fixture");
        };
        let mut function_stmt = Stmt::Return(Some(Expr::Num(1.0)));
        for _ in 0..40 {
            let mut function = (**template).clone();
            function.body = vec![function_stmt].into();
            function_stmt = Stmt::FuncDecl(Rc::new(function));
        }
        let mut class_expr = Expr::Ident("Object".into(), crate::stack_trace::NO_POSITION);
        for _ in 0..40 {
            class_expr = Expr::Class(Rc::new(Class {
                name: None,
                superclass: Some(Box::new(class_expr)),
                members: Vec::new(),
                decorators: Vec::new(),
                source: None,
                script: None,
                start: 0,
            }));
        }
        let programs = [
            vec![Stmt::Expr(expr)],
            vec![stmt],
            vec![Stmt::VarDecl {
                kind: DeclKind::Let,
                decls: vec![(pattern, None)],
            }],
            vec![function_stmt],
            vec![Stmt::Expr(class_expr)],
        ];
        let blobs: Vec<_> = programs.iter().map(|body| encode(body)).collect();
        std::thread::Builder::new()
            .name("snapshot-all-gateways".into())
            .stack_size(128 * 1024)
            .spawn(move || {
                stacker::grow(128 * 1024, || {
                    for bytes in blobs {
                        for _ in 0..3 {
                            let ast = decode(&bytes).expect("valid recursive snapshot");
                            assert_eq!(encode(&ast), bytes);
                            drop(ast);
                        }
                        // A failed reader must release completed siblings and partial trees, too.
                        for end in [bytes.len() / 2, bytes.len() - 1] {
                            assert!(decode(&bytes[..end]).is_err());
                        }
                    }
                    corrupt_lengths_and_varints_fail_before_allocation();
                    corrupt_structure_is_strictly_bounded();
                    super::CODEC_STACK_SEGMENTS.with(|count| {
                        let (live, total) = count.get();
                        assert_eq!(live, 0);
                        assert!(total > 0);
                    });
                })
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn codec_stack_switches_preserve_ownership_and_unwind_cleanup() {
        std::thread::Builder::new()
            .name("snapshot-ownership".into())
            .stack_size(128 * 1024)
            .spawn(|| {
                stacker::grow(128 * 1024, || {
                    use crate::ast::{Expr, Stmt};
                    use std::rc::Rc;
                    let text: Rc<str> = Rc::from("retained only by its original owner");
                    let original = vec![Stmt::Expr(Expr::Str(text.clone()))];
                    let bytes = encode(&original);
                    assert_eq!(Rc::strong_count(&text), 2);
                    let decoded = decode(&bytes).unwrap();
                    let Stmt::Expr(Expr::Str(decoded_text)) = &decoded[0] else {
                        panic!("snapshot AST changed");
                    };
                    let weak = Rc::downgrade(decoded_text);
                    assert_eq!(&**decoded_text, &*text);
                    drop(decoded);
                    assert!(weak.upgrade().is_none());
                    drop(original);
                    assert_eq!(Rc::strong_count(&text), 1);
                    let caught = std::panic::catch_unwind(|| {
                        super::with_codec_stack(|| panic!("codec stack unwind test"));
                    });
                    assert!(caught.is_err());
                    super::CODEC_STACK_SEGMENTS.with(|count| {
                        let (live, total) = count.get();
                        assert_eq!(live, 0);
                        assert!(
                            total > 0,
                            "ownership and unwind checks must grow the codec stack"
                        );
                    });
                    assert_eq!(encode(&decode(&bytes).unwrap()), bytes);
                })
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// Leak a copy: static decoding keeps `&'static` slices of its input.
    fn leak(bytes: &[u8]) -> &'static [u8] {
        Box::leak(bytes.to_vec().into_boxed_slice())
    }

    fn validated_bodies() -> usize {
        super::VALIDATED_BODIES.with(std::cell::Cell::get)
    }

    /// The function declared by the first statement of `body`.
    fn declared(body: &[Stmt]) -> Rc<Function> {
        match &body[0] {
            Stmt::FuncDecl(function) => function.clone(),
            other => panic!("expected a function declaration, found {other:?}"),
        }
    }

    const NESTED: &str = r#"function outer(a, b = () => a) {
        let unused;
        const inner = function named(x) { return x + a; };
        class C extends Object { m() { return super.toString; } get g() { return 1; } static { this.s = 2; } }
        function* gen() { yield* [1, 2]; }
        return [inner, C, gen, { method() { return this; }, arrow: (y) => y * 2 }];
    }
    var later = async (z) => { await z; return z; };"#;

    #[test]
    fn static_decode_leaves_function_bodies_encoded_until_first_use() {
        let bytes = crate::compile_snapshot(NESTED).unwrap();
        let eager = decode(&bytes).unwrap();
        let lazy = decode_static(leak(&bytes)).unwrap();
        let outer = declared(&lazy);
        assert!(
            outer.body.materialized().is_none(),
            "the body stays encoded"
        );
        assert_eq!(
            outer.params.len(),
            2,
            "parameters are decoded with the function"
        );
        // An analysis decodes the statements without keeping them.
        assert_eq!(outer.body.with(|stmts| stmts.len()), 5);
        assert!(outer.body.materialized().is_none());
        // Debug output decodes transiently and is the eager tree's.
        assert_eq!(format!("{lazy:?}"), format!("{eager:?}"));
        assert!(outer.body.materialized().is_none());
        // Reading the statements materializes them once; nested bodies stay encoded.
        assert_eq!(outer.body.len(), 5);
        assert!(outer.body.materialized().is_some());
        let first = outer.body.as_ptr();
        assert_eq!(
            outer.body.as_ptr(),
            first,
            "the statements keep their identity"
        );
        let Stmt::VarDecl { decls, .. } = &outer.body[1] else {
            panic!("const inner")
        };
        let Some(Expr::Func(inner)) = &decls[0].1 else {
            panic!("function expression")
        };
        assert!(
            inner.body.materialized().is_none(),
            "nested bodies stay encoded"
        );
        assert_eq!(format!("{lazy:?}"), format!("{eager:?}"));
        // The parameter default arrow was decoded with the parameters; its body stays encoded.
        let Some(Expr::Func(default)) = &outer.params[1].default else {
            panic!("arrow default")
        };
        assert!(default.body.materialized().is_none());
    }

    #[test]
    fn static_decode_reencodes_to_the_same_bytes() {
        for source in [
            NESTED,
            "function f() { return function g() { return () => this; }; }",
        ] {
            let bytes = crate::compile_snapshot(source).unwrap();
            let lazy = decode_static(leak(&bytes)).unwrap();
            assert_eq!(encode(&lazy), bytes, "unmaterialized: {source}");
            let outer = declared(&lazy);
            assert!(!outer.body.is_empty());
            assert_eq!(encode(&lazy), bytes, "materialized: {source}");
            let host = crate::compile_host_snapshot(source).unwrap();
            assert_eq!(
                super::encode_host(&decode_static(leak(&host)).unwrap()),
                host
            );
        }
    }

    /// Static decoding validates exactly as eager decoding: every single-byte corruption of a
    /// snapshot with nested bodies is rejected by both or accepted by both, with the same error
    /// and, when accepted, the same tree.
    #[test]
    fn static_decode_accepts_and_rejects_what_decode_does() {
        let bytes = crate::compile_snapshot(
            "function f(a) { function g() { return [a, 1.5, 'x']; } return g; } var v = f;",
        )
        .unwrap();
        for index in 0..bytes.len() {
            for value in [0x00, 0x01, 0x7f, 0x80, 0xff, bytes[index].wrapping_add(1)] {
                let mut mutated = bytes.clone();
                mutated[index] = value;
                let eager = decode(&mutated);
                let lazy = decode_static(leak(&mutated));
                match (&eager, &lazy) {
                    (Ok(eager), Ok(lazy)) => {
                        assert_eq!(format!("{lazy:?}"), format!("{eager:?}"), "byte {index}");
                    }
                    (Err(eager), Err(lazy)) => assert_eq!(eager, lazy, "byte {index}"),
                    _ => panic!("byte {index} = {value:#x}: {eager:?} / {lazy:?}"),
                }
            }
        }
        // Nesting and length checks inside a body that stays encoded. Build the deep fixture
        // directly, so parser recursion is not under test.
        let template = crate::parser::parse_script("function f() {}", false)
            .ok()
            .expect("function template parses");
        let mut function = (*declared(&template)).clone();
        let mut expr = Expr::Num(1.0);
        for _ in 0..200 {
            expr = Expr::Paren(Box::new(expr));
        }
        function.body = vec![Stmt::Return(Some(expr))].into();
        let deep = encode(&[Stmt::FuncDecl(Rc::new(function))]);
        let eager = decode(&deep).unwrap_err();
        assert!(eager.contains("nesting limit"), "{eager}");
        assert_eq!(decode_static(leak(&deep)).unwrap_err(), eager);
        let mut short = crate::compile_host_snapshot("function f() { return 1; }").unwrap();
        // The body length follows the MAGIC/VERSION header, the statement count and tag, and
        // the function's name and empty parameter list.
        let header = encode(&[]).len() - 1;
        let at = header + 1 + 1 + 3 + 1;
        let len = u32::from_le_bytes(short[at..at + 4].try_into().unwrap());
        assert_eq!(
            len as usize,
            short.len() - at - 4 - 3,
            "the body precedes 3 flag bytes"
        );
        short[at..at + 4].copy_from_slice(&(len - 1).to_le_bytes());
        let eager = decode(&short).unwrap_err();
        assert!(eager.contains("length mismatch"), "{eager}");
        assert_eq!(decode_static(leak(&short)).unwrap_err(), eager);
    }

    #[test]
    fn static_decode_validates_bodies_once_per_process() {
        let bytes = leak(&crate::compile_snapshot(NESTED).unwrap());
        let before = validated_bodies();
        let first = decode_static(bytes).unwrap();
        let validated = validated_bodies() - before;
        // Every function of the snapshot, nested ones included, exactly once.
        let functions = format!("{:?}", decode(bytes).unwrap())
            .matches("Function {")
            .count();
        assert_eq!(validated, functions);
        assert!(functions >= 10, "{functions}");
        let second = decode_static(bytes).unwrap();
        assert_eq!(validated_bodies() - before, validated, "already validated");
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        // The same content at another address is validated again.
        let copy = leak(bytes);
        decode_static(copy).unwrap();
        assert_eq!(validated_bodies() - before, 2 * validated);
    }

    #[test]
    fn edited_and_cloned_bodies_keep_their_statements() {
        let bytes = crate::compile_snapshot(NESTED).unwrap();
        let lazy = decode_static(leak(&bytes)).unwrap();
        let outer = declared(&lazy);
        let clone = (*outer).clone();
        assert!(
            clone.body.materialized().is_none(),
            "a clone of an encoded body stays encoded"
        );
        let mut edited = (*outer).clone();
        edited.body.truncate(1);
        assert!(edited.body.materialized().is_some());
        assert_eq!(edited.body.len(), 1);
        assert_eq!(clone.body.len(), 5);
        assert!(
            outer.body.materialized().is_none(),
            "the original is untouched"
        );
    }

    /// Collect `(name, run_once)` for every named function the fixtures below contain.
    fn function_marks(body: &[Stmt], out: &mut Vec<(String, bool)>) {
        use crate::ast::{ArrayElem, ClassMember, ForInit, MemberKind, PropDef, PropKey};
        fn function(f: &Function, out: &mut Vec<(String, bool)>) {
            named(f, f.name.clone().unwrap_or_default(), out);
        }
        fn member(member: &ClassMember, out: &mut Vec<(String, bool)>) {
            if let Some(func) = &member.func {
                match (&member.kind, &member.key) {
                    (MemberKind::StaticBlock, _) => named(func, "<static block>".into(), out),
                    (_, PropKey::Ident(key)) => named(func, key.clone(), out),
                    _ => function(func, out),
                }
            }
        }
        fn named(f: &Function, name: String, out: &mut Vec<(String, bool)>) {
            out.push((name, f.run_once));
            for param in &f.params {
                if let Some(default) = &param.default {
                    expr(default, out);
                }
            }
            f.body.with(|body| function_marks(body, out));
        }
        fn expr(e: &Expr, out: &mut Vec<(String, bool)>) {
            match e {
                Expr::Func(f) => function(f, out),
                Expr::Paren(inner) | Expr::Unary { arg: inner, .. } => expr(inner, out),
                Expr::Call { callee, args, .. } => {
                    expr(callee, out);
                    for arg in args {
                        if let ArrayElem::Item(arg) | ArrayElem::Spread(arg) = arg {
                            expr(arg, out);
                        }
                    }
                }
                Expr::Member { obj, .. } => expr(obj, out),
                Expr::Array(items) => {
                    for item in items {
                        if let ArrayElem::Item(item) | ArrayElem::Spread(item) = item {
                            expr(item, out);
                        }
                    }
                }
                Expr::Object(props) => {
                    for prop in props {
                        match prop {
                            PropDef::Method {
                                key: PropKey::Ident(key),
                                func,
                            } => named(func, key.clone(), out),
                            PropDef::KeyValue { value, .. } => expr(value, out),
                            _ => {}
                        }
                    }
                }
                Expr::Class(class) => {
                    for each in &class.members {
                        member(each, out);
                    }
                }
                _ => {}
            }
        }
        for statement in body {
            match statement {
                Stmt::Expr(e) => expr(e, out),
                Stmt::FuncDecl(f) => function(f, out),
                Stmt::VarDecl { decls, .. } => {
                    for (_, init) in decls {
                        if let Some(init) = init {
                            expr(init, out);
                        }
                    }
                }
                Stmt::If { test, cons, alt } => {
                    expr(test, out);
                    function_marks(std::slice::from_ref(&**cons), out);
                    if let Some(alt) = alt {
                        function_marks(std::slice::from_ref(&**alt), out);
                    }
                }
                Stmt::Block(body) => function_marks(body, out),
                Stmt::Try {
                    block, finalizer, ..
                } => {
                    function_marks(block, out);
                    if let Some(finalizer) = finalizer {
                        function_marks(finalizer, out);
                    }
                }
                Stmt::For { init, body, .. } => {
                    if let Some(ForInit::Expr(e)) = init.as_deref() {
                        expr(e, out);
                    }
                    function_marks(std::slice::from_ref(&body.statement), out);
                }
                Stmt::While { body, .. } => {
                    function_marks(std::slice::from_ref(&body.statement), out)
                }
                Stmt::ClassDecl(class) => {
                    for each in &class.members {
                        member(each, out);
                    }
                }
                _ => {}
            }
        }
    }

    /// A static snapshot marks exactly the function expressions that its top-level code, or the
    /// body of another marked function, calls immediately outside loops and classes: each runs
    /// once per evaluation of the Script.
    #[test]
    fn run_once_marks_only_immediately_invoked_bootstrap_functions() {
        const SOURCE: &str = r#"(function wrapper() {
            (function block() {})();
            (function crockford() {}());
            !function bang() {}();
            var assigned = (function value() { return 1; })();
            if (assigned) { (function inIf() {})(); }
            try { (function inTry() {})(); } finally { (function inFinally() {})(); }
            for (let i = 0; i < 2; i++) (function inFor() {})();
            while (false) (function inWhile() {})();
            function declared() { (function inDeclared() {})(); }
            var expression = function notCalled() { (function inNotCalled() {})(); };
            class C {
                static { (function inStaticBlock() {})(); }
                m() { (function inMethod() {})(); }
            }
            [1].forEach(function callback() { (function inCallback() {})(); });
            ({ method() { (function inLiteralMethod() {})(); } }).method();
            (function outer(f = function inDefault() {}) { (function inner() {})(); })();
            (() => { (function inArrow() {})(); })();
        })();
        function topDeclared() { (function inTopDeclared() {})(); }
        (function second() {})();"#;
        let expected = [
            ("wrapper", true),
            ("block", true),
            ("crockford", true),
            ("bang", true),
            ("value", true),
            ("inIf", true),
            ("inTry", true),
            ("inFinally", true),
            ("inFor", false),
            ("inWhile", false),
            ("declared", false),
            ("inDeclared", false),
            ("notCalled", false),
            ("inNotCalled", false),
            ("<static block>", false),
            ("inStaticBlock", false),
            ("m", false),
            ("inMethod", false),
            ("callback", false),
            ("inCallback", false),
            ("method", false),
            ("inLiteralMethod", false),
            ("outer", true),
            ("inDefault", false),
            ("inner", true),
            ("", true), // the arrow
            ("inArrow", true),
            ("topDeclared", false),
            ("inTopDeclared", false),
            ("second", true),
        ];
        let bytes = crate::compile_host_snapshot(SOURCE).unwrap();
        let mut marks = Vec::new();
        function_marks(&decode_static(leak(&bytes)).unwrap(), &mut marks);
        let mut sorted_marks = marks.clone();
        sorted_marks.sort();
        let mut sorted_expected: Vec<(String, bool)> = expected
            .iter()
            .map(|(name, mark)| (name.to_string(), *mark))
            .collect();
        sorted_expected.sort();
        assert_eq!(sorted_marks, sorted_expected);
        // Only a static decode keeps the marks; the encoding is the same either way.
        let eager = decode(&bytes).unwrap();
        let mut eager_marks = Vec::new();
        function_marks(&eager, &mut eager_marks);
        assert!(eager_marks.iter().all(|(_, mark)| !mark), "{eager_marks:?}");
        assert_eq!(super::encode_host(&eager), bytes);
    }
}
