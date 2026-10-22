//! TRust JS — a JavaScript engine written in Rust.
//!
//! The tree-walking interpreter is the semantic reference for bytecode and native JIT tiers. The
//! tc39/test262 conformance suite drives language work (see `crates/test262-runner`).
//!
//! ## Shape
//! - [`lexer`] tokenizes, [`parser`] builds the [`ast`], [`interpreter`] + `eval` walk it.
//! - [`value`] is the prototype-based object model: `RefCell<Object>` cells on a mark-sweep heap
//!   (conservative only for the native stack), with bounded live-object pressure.
//! - [`builtins`] installs the realm (`globalThis`, `Object`/`Array`/`Function`/`Math`, the error
//!   constructors, global functions).
//!
//! ## Public API
//! [`Engine::new`] builds a fresh realm; [`Engine::eval`] runs a script and reports a [`Completion`]
//! (a value, or a thrown error with its constructor name + message) or a parse-phase [`ParseError`].
//! The error name + phase distinction is exactly what a test262 negative-test matcher needs.

// The ECMAScript abstract operations (`to_number`/`to_string`/`to_primitive`/…) take `&mut self`
// on purpose: converting an object can run user `valueOf`/`toString`/getters, which mutate the
// realm. That trips clippy's `wrong_self_convention`, which assumes `to_*` is a cheap borrow.
#![allow(clippy::wrong_self_convention)]
// Without `embed`, the engine's embedder entry points (host state, the internals behind the scoped
// API) have no caller: the `embed` build is the one whose dead code the lint checks (S6 step 11b).
#![cfg_attr(not(feature = "embed"), allow(dead_code))]
// Handles are `Copy` (P1 S2) and packed words have no drop glue (S5): code written for
// reference-counted handles still clones, drops or forgets them where such a handle was
// retained or released.
#![allow(
    dropping_copy_types,
    forgetting_copy_types,
    clippy::clone_on_copy,
    clippy::drop_non_drop
)]

/// A process-lifetime diagnostic switch: whether environment variable `$name` is set. The first
/// read is cached, because a libc environment lookup scans every variable under a lock and the
/// compilers consult these switches per chunk and per candidate region.
macro_rules! env_flag {
    ($name:literal) => {{
        static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *FLAG.get_or_init(|| std::env::var_os($name).is_some())
    }};
}

/// The cached value of a process-lifetime diagnostic environment variable (see [`env_flag`]).
macro_rules! env_value {
    ($name:literal) => {{
        static VALUE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        VALUE.get_or_init(|| std::env::var($name).ok()).as_deref()
    }};
}

#[cfg(test)]
mod agent_activation_tests;
mod array_binding;
#[cfg(test)]
mod array_binding_tests;
mod ast;
mod atomic_compat;
mod bigint;
#[cfg(test)]
mod binding_reference_tests;
#[cfg(test)]
mod bootstrap_operation_tests;
mod builtins;
pub use interpreter::SharedMem;
mod bytecode;
/// The execution tiers ([`Engine::set_tier`]).
pub use bytecode::Tier;
mod cache;
mod callback;
mod callee_name;
#[cfg(test)]
mod chain_bail_tests;
#[cfg(test)]
mod code_point_intrinsic_tests;
#[cfg(test)]
mod computed_reference_tests;
mod coroutine;
#[cfg(test)]
mod data_copy_tests;
#[cfg(test)]
mod dense_slot_tests;
mod env_handle;
mod eval;
mod execution_storage;
/// The engine's size-class caching allocator, for its many small Rust-heap blocks (spilled
/// property and binding storage, keys, argument lists, boxed internal state). NOT registered
/// here: a library must not preempt an embedder's `#[global_allocator]` (the test262 runner caps
/// worker allocations with its own). Binaries opt in:
/// `#[global_allocator] static A: trust_js::fastalloc::ClassAlloc = trust_js::fastalloc::ClassAlloc;`
#[cfg(not(target_arch = "wasm32"))]
pub mod fastalloc;
mod fasthash;
mod feedback;
mod finalization_cells;
#[cfg(test)]
mod for_in_cache_tests;
#[cfg(test)]
mod function_realm_identity_tests;
#[cfg(test)]
mod function_realm_lifetime_tests;
mod gc_box;
mod gc_core;
#[cfg(test)]
mod gc_test;
#[cfg(test)]
mod payload_cell_tests;
mod rooting;

/// Run an embedder entry (a script, a microtask checkpoint) between two stack scrubs: its
/// frames start without stale cell addresses below the entry, and leave none behind for
/// whatever the embedder calls next, such as an explicit collection.
#[inline(never)]
fn scrubbed<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    gc_core::scrub_stack(gc_core::ENTRY_SCRUB_BYTES);
    let result = f();
    #[cfg(not(target_arch = "wasm32"))]
    gc_core::scrub_stack(gc_core::ENTRY_SCRUB_BYTES);
    result
}
mod gc_handle;
mod gc_native;
#[cfg(test)]
mod gc_scan_spike;
mod gc_sweep;
mod gc_trace;
#[cfg(test)]
mod group_by_tests;
#[cfg(all(test, feature = "embed"))]
mod handle_tests;
mod handles;
mod host;
mod host_collections;
mod host_memory;
#[cfg(test)]
mod host_string_tests;
mod inline_vec;
#[cfg(test)]
mod internal_slot_tests;
mod interpreter;
mod interrupt;
#[cfg(feature = "intl")]
mod intl;
mod jit;
mod jit_ir;
#[cfg(test)]
mod json_stringify_space_tests;
mod jstr;
#[cfg(test)]
mod lazy_snapshot_body_tests;
mod lexer;
#[cfg(test)]
mod loop_completion_tests;
#[cfg(test)]
mod loop_fragment_tests;
mod lstr;
mod memory;
#[cfg(test)]
mod minor_collection_tests;
mod modules;
#[cfg(test)]
mod native_captures;
#[cfg(all(test, feature = "embed"))]
mod native_constructor_tests;
mod native_typed_array;
#[cfg(feature = "intl")]
mod numbering;
#[cfg(test)]
mod object_alloc_bench;
#[cfg(test)]
mod object_literal_jit_tests;
mod ordered_collection;
#[cfg(test)]
mod ordered_collection_tests;
mod parser;
#[cfg(test)]
mod private_name_cache_tests;
#[cfg(test)]
mod realm_inline_guard_tests;
#[cfg(test)]
mod record_enumeration_tests;
mod regex;
mod regex_emoji;
mod regex_fold;
#[cfg(test)]
mod regexp_tests;
#[cfg(feature = "embed")]
mod scope;
#[cfg(all(test, feature = "embed"))]
mod scope_tests;
#[cfg(all(test, feature = "embed"))]
mod script_caller_tests;
#[cfg(test)]
mod script_entry_tests;
#[cfg(test)]
mod script_scope_tests;
#[cfg(test)]
mod script_tiering_tests;
mod self_hosted;
#[cfg(test)]
mod shared_buffer_lifetime_tests;
#[cfg(test)]
mod shared_layout_tests;
#[cfg(test)]
mod shared_snapshot_realm_tests;
mod snapshot;
mod spread;
mod stack_trace;
#[cfg(test)]
mod string_concat_tests;
#[cfg(test)]
mod string_replace_split_tests;
#[cfg(test)]
mod string_slice_tests;
#[cfg(test)]
mod table_record_tests;
mod tagged;
mod temporal;
mod tiering;
mod token;
#[cfg(test)]
mod typedarray_allocation_tests;
#[cfg(test)]
mod typedarray_conversion_tests;
#[cfg(test)]
mod typedarray_search_tests;
mod tz;
mod weak_metadata;
#[rustfmt::skip]
// The daylight-time flags serve Intl's time zone names.
#[cfg_attr(not(feature = "intl"), allow(dead_code))]
mod tzdata;
#[rustfmt::skip]
mod tzrules;
#[rustfmt::skip]
#[cfg(any(windows, test))]
mod cldr_windows_zones;
#[rustfmt::skip]
mod umalqura;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod cldr_likely;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod cldr_locales;
#[cfg(feature = "intl")]
mod cldr_locale_info;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod cldr_dates;
#[cfg(feature = "intl")]
mod cldr_collation;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod cldr_datetime_patterns;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod cldr_tz_names;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod cldr_units;
#[rustfmt::skip]
#[cfg(feature = "intl")]
mod units;
#[cfg(feature = "intl")]
mod unicode_collation;
mod unicode_norm;
mod unicode_norm_impl;
mod unicode_props;
#[cfg(feature = "intl")]
#[rustfmt::skip]
mod unicode_segment;
#[cfg(feature = "intl")]
#[rustfmt::skip]
mod cldr_plurals;
#[cfg(feature = "intl")]
#[rustfmt::skip]
mod cldr_lists;
#[cfg(feature = "intl")]
#[rustfmt::skip]
mod cldr_relative_time;
#[cfg(feature = "intl")]
#[rustfmt::skip]
mod cldr_display_names;
#[cfg(feature = "intl")]
#[rustfmt::skip]
mod cldr_numbers;
mod value;
mod workload_metrics;
#[cfg(test)]
mod write_barrier_tests;

use interpreter::Interp;
use std::time::{Duration, Instant};
use value::Value;

pub use interrupt::{InterruptReason, RuntimeInterrupt};
pub use stack_trace::SourceOrigin;

/// Unstable, opt-in process JIT/collector and per-Agent managed-memory diagnostics for TRust JS's
/// own benchmark tooling.
///
/// Returns `None` unless `TRUST_JS_PERF_METRICS` was present when the process first checked its
/// diagnostic state. `engine` selects the Agent-owned memory snapshot; process JIT/GC counters
/// remain aggregate diagnostics. This is intentionally not a stable embedder API.
#[doc(hidden)]
/// Opt-in matcher profiling report (`TRUST_JS_REGEXP_PROF=1`); `None` when disabled or idle.
pub fn unstable_regexp_prof_report() -> Option<String> {
    crate::regex::regexp_prof_report()
}

pub fn unstable_performance_metrics_json(engine: &Engine) -> Option<String> {
    let managed_memory = memory::json(&engine.interp);
    jit::performance_metrics_json(&managed_memory)
}

/// An opt-in, bounded sampler for diagnostics from a long-lived Agent.
///
/// Performance counters are intentionally cumulative, while the managed-memory record is a
/// post-collection snapshot. `maybe_dump` advances the existing diagnostic collection boundary
/// only when the interval expires and returns one bounded JSON envelope. Hosts can call it from
/// their normal task/idle loop; normal execution never creates a sampler or pays for its clock.
/// The JIT/GC counters inside the envelope remain process aggregates; the managed-memory record
/// carries the sampled Agent/heap identity. The envelope is deliberately unstable tooling output,
/// not an ECMAScript-visible operation.
#[doc(hidden)]
pub struct PerformanceMetricsSampler {
    interval: Duration,
    started: Instant,
    next_dump: Instant,
    sequence: u64,
    enabled: bool,
}

impl PerformanceMetricsSampler {
    const DEFAULT_INTERVAL: Duration = Duration::from_secs(10);

    /// Construct a sampler when the host has already opted into `TRUST_JS_PERF_METRICS`.
    ///
    /// A zero interval is normalized to one second so a configuration mistake cannot turn a
    /// long-lived browser loop into an unbounded stream of collections and JSON allocations.
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        let interval = if interval.is_zero() {
            Duration::from_secs(1)
        } else {
            interval
        };
        let now = Instant::now();
        let enabled = std::env::var_os("TRUST_JS_PERF_METRICS").is_some();
        Self {
            interval,
            started: now,
            next_dump: now,
            sequence: 0,
            enabled,
        }
    }

    /// Construct a sampler from the diagnostic environment, or return `None` when diagnostics are
    /// disabled. `TRUST_JS_PERF_METRICS_INTERVAL_SECONDS` accepts a finite positive floating-point
    /// duration and defaults to ten seconds. Invalid values are ignored rather than changing host
    /// behavior or disabling the normal metrics switch.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        std::env::var_os("TRUST_JS_PERF_METRICS")?;
        let interval = std::env::var("TRUST_JS_PERF_METRICS_INTERVAL_SECONDS")
            .ok()
            .and_then(|raw| raw.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .unwrap_or(Self::DEFAULT_INTERVAL);
        Some(Self::new(interval))
    }

    /// Return a machine-readable diagnostic envelope once the configured interval has elapsed.
    /// The first call emits an immediate snapshot. The caller owns the output destination; no
    /// process-global stream or background thread is created.
    pub fn maybe_dump(&mut self, engine: &mut Engine) -> Option<String> {
        let now = Instant::now();
        let (sequence, elapsed_seconds) = self.claim(now)?;
        engine.unstable_collect_for_performance_metrics();
        let metrics = unstable_performance_metrics_json(engine)?;
        Some(format!(
            "{{\"schema_version\":1,\"dump_sequence\":{sequence},\"elapsed_seconds\":{elapsed_seconds:.6},\"interval_seconds\":{:.6},\"metrics\":{metrics}}}",
            self.interval.as_secs_f64()
        ))
    }

    fn claim(&mut self, now: Instant) -> Option<(u64, f64)> {
        if !self.enabled || now < self.next_dump {
            return None;
        }
        self.sequence = self.sequence.saturating_add(1);
        // Schedule from the completed sample, not from the old deadline. If a host is paused for
        // minutes, the next poll emits one record instead of replaying every missed interval.
        self.next_dump = now.checked_add(self.interval).unwrap_or(now);
        Some((
            self.sequence,
            now.duration_since(self.started).as_secs_f64(),
        ))
    }

    #[cfg(test)]
    fn new_for_test(interval: Duration, enabled: bool, now: Instant) -> Self {
        let mut sampler = Self::new(interval);
        sampler.started = now;
        sampler.next_dump = now;
        sampler.enabled = enabled;
        sampler
    }
}

#[cfg(test)]
mod performance_sampler_tests {
    use super::PerformanceMetricsSampler;
    use std::time::{Duration, Instant};

    #[test]
    fn sampler_emits_immediately_then_respects_interval() {
        let start = Instant::now();
        let mut sampler =
            PerformanceMetricsSampler::new_for_test(Duration::from_secs(5), true, start);
        assert_eq!(sampler.claim(start), Some((1, 0.0)));
        assert_eq!(sampler.claim(start + Duration::from_secs(4)), None);
        let (sequence, elapsed) = sampler
            .claim(start + Duration::from_secs(5))
            .expect("second interval");
        assert_eq!(sequence, 2);
        assert_eq!(elapsed, 5.0);
        assert_eq!(sampler.claim(start + Duration::from_secs(6)), None);
    }

    #[test]
    fn disabled_sampler_never_claims_or_allocates_a_sample() {
        let start = Instant::now();
        let mut sampler =
            PerformanceMetricsSampler::new_for_test(Duration::from_secs(1), false, start);
        assert_eq!(sampler.claim(start), None);
        assert_eq!(sampler.claim(start + Duration::from_secs(60)), None);
    }

    #[test]
    fn zero_interval_is_bounded() {
        let sampler = PerformanceMetricsSampler::new(Duration::ZERO);
        assert_eq!(sampler.interval, Duration::from_secs(1));
    }
}

/// Internal-stage entry points, exposed only for benchmarking (`bench` feature). These reach past
/// the stable public API to time individual compilation stages (lex → parse → snapshot encode →
/// decode) — the breakdown behind cold-boot cost. Not a stability commitment; do not depend on it.
#[cfg(feature = "bench")]
pub mod bench_api {
    pub use crate::ast::Stmt;
    pub use crate::lexer::tokenize;
    pub use crate::parser::{parse_module, parse_script};
    pub use crate::snapshot::{decode, encode};
}

/// Legacy process-wide wall-clock fallback, retained for embedders which cannot use the `embed`
/// feature. Browser/runtime embedders should prefer [`Engine::set_wall_clock`]: HTML's
/// `HostSystemUTCEpochNanoseconds(global)` hook is global-sensitive, so unrelated realms must not
/// compete for a first-call-wins process singleton.
static HOST_CLOCK: std::sync::OnceLock<fn() -> f64> = std::sync::OnceLock::new();

/// Install a process-wide wall-clock source (first call wins). The embedder's `f` returns
/// milliseconds since the Unix epoch.
pub fn set_host_clock(f: fn() -> f64) {
    let _ = HOST_CLOCK.set(f);
}

/// The installed host clock's current time, if one was set.
pub(crate) fn host_now_ms() -> Option<f64> {
    HOST_CLOCK.get().map(|f| f())
}

/// Parse Script source text supplied by the host. The text is Rust text, so a character in the
/// range the engine reserves for lone surrogates is a real character (see `Value::str`); eval
/// and the Function constructor parse engine strings directly.
fn parse_host_script(src: &str, strict: bool) -> Result<Vec<ast::Stmt>, parser::ParseError> {
    parser::parse_script(&jstr::from_text(src), strict)
}

/// Parse an embedder's script whose stack frames report `origin` (`<anonymous>` when None).
fn parse_host_script_from(
    src: &str,
    strict: bool,
    origin: Option<&SourceOrigin<'_>>,
) -> Result<(Vec<ast::Stmt>, std::rc::Rc<stack_trace::ScriptSource>), ParseError> {
    parser::parse_script_from(
        &jstr::from_text(src),
        strict,
        stack_trace::SourceKind::Script,
        origin,
    )
    .map_err(|error| ParseError {
        message: error.message,
        line: error.line,
        at_eof: error.at_eof,
    })
}

/// Parse `src` as a script and encode its AST to a snapshot blob — a build-time helper (used
/// from op crates' `build.rs`) so static JS glue is parsed once at build and decoded, not
/// re-parsed, on every boot. Decode it at runtime with [`Engine::eval_snapshot`]. `Err` is a
/// parse-error message.
pub fn compile_snapshot(src: &str) -> Result<Vec<u8>, String> {
    let body =
        parse_host_script(src, false).map_err(|e| format!("{} (line {})", e.message, e.line))?;
    Ok(snapshot::encode(&body))
}

/// Compile implementation-owned bootstrap code without author-visible function
/// source, including nested functions and class constructors. This supplies the
/// source-unavailable behavior permitted by ECMA-262 HostHasSourceTextAvailable:
/// Function.prototype.toString uses NativeFunction syntax for these callables.
/// Ordinary snapshots and subsequently parsed author/eval/Function code retain
/// their source. This does not alter execution, constructibility, or descriptors,
/// and does not by itself implement the rest of Web IDL's function-object rules.
pub fn compile_host_snapshot(src: &str) -> Result<Vec<u8>, String> {
    let body =
        parse_host_script(src, false).map_err(|e| format!("{} (line {})", e.message, e.line))?;
    Ok(snapshot::encode_host(&body))
}

/// A parse-phase failure. test262 reports these as a `SyntaxError` thrown during parsing.
#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub line: u32,
    /// The parse failed only because the input ended too soon (e.g. an unclosed block or
    /// template). A REPL treats this as "keep reading lines", not a SyntaxError.
    pub at_eof: bool,
}

/// The outcome of evaluating a script.
pub enum Completion {
    /// Ran to completion; the last statement value rendered to a string (best-effort).
    Value(String),
    /// A value was thrown. `name` is the error's constructor name (`"TypeError"`, …) when the
    /// thrown value is an Error object, else `""`.
    Throw { name: String, message: String },
}

/// The outcome returned by interruption-aware evaluation entry points.
///
/// This is separate from [`Completion`] so adding host control flow does not break embedders
/// which exhaustively match the original public enum. An interruption is not a JavaScript throw:
/// author `catch` and `finally` blocks do not observe it (HTML §8.1.4.5 "Killing scripts").
pub enum ExecutionOutcome {
    Value(String),
    Throw { name: String, message: String },
    Interrupted { reason: InterruptReason },
}

/// A JavaScript engine instance: one realm (global object + intrinsics) that persists across
/// [`eval`](Engine::eval) calls.
/// Destroying the engine dismantles its Agent's object and environment graph, including all
/// child realms. Opaque object handles belong to this engine; retaining one does not keep the
/// realm executable after the engine has been destroyed.
pub struct Engine {
    // Thread-backed generator bodies hold a pointer to the interpreter while suspended. Boxing
    // makes that address stable even when ordinary Rust code moves the public `Engine` value.
    interp: Box<Interp>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Engine {
        Self::new_with_interrupt(Default::default())
    }

    /// Build a realm around a host-created control handle.
    ///
    /// This closes the startup race for dedicated workers: their owner can cancel the handle
    /// before the worker thread has finished constructing its realm or begun evaluating its
    /// entry script.
    pub fn new_with_interrupt(interrupt: std::sync::Arc<RuntimeInterrupt>) -> Engine {
        let mut interp = Interp::new();
        interp.runtime_interrupt = interrupt;
        Engine {
            interp: Box::new(interp),
        }
    }

    /// A thread-safe handle for cancelling this realm, yielding to user navigation, or setting a
    /// host execution deadline. The handle remains valid while JavaScript is running on another
    /// thread (notably a dedicated worker).
    pub fn interrupt_handle(&self) -> std::sync::Arc<RuntimeInterrupt> {
        self.interp.runtime_interrupt.clone()
    }

    /// Force the post-collection safepoint required by the unstable performance record.
    ///
    /// The CLI calls this only when `TRUST_JS_PERF_METRICS` is enabled. It is hidden rather than part
    /// of the supported embedder API; browser hosts use their normal idle collection entry point.
    #[doc(hidden)]
    pub fn unstable_collect_for_performance_metrics(&mut self) {
        self.interp.collect_garbage_for_host();
    }

    /// Replace the realm's control handle before evaluating author code.
    ///
    /// Runtime assemblers use this after their finite, trusted bootstrap has installed built-ins:
    /// an owner may already have cancelled the supplied handle, in which case the first author
    /// script entry observes that cancellation immediately.
    pub fn set_interrupt_handle(&mut self, interrupt: std::sync::Arc<RuntimeInterrupt>) {
        self.interp.runtime_interrupt = interrupt;
    }

    /// Run `src` as a spawned `$262.agent`: the agent may block in `Atomics.wait`, receives
    /// SharedArrayBuffer broadcasts on `broadcast_rx`, and reports back via `report_tx`.
    /// Whether this agent may block in `Atomics.wait` (test262's CanBlockIsTrue flag).
    pub fn set_can_block(&mut self, b: bool) {
        self.interp.can_block = b;
    }

    pub fn run_as_agent(
        &mut self,
        src: &str,
        broadcast_rx: std::sync::mpsc::Receiver<(SharedMem, usize)>,
        report_tx: std::sync::mpsc::Sender<String>,
    ) {
        self.interp.can_block = true;
        self.interp.agent = Some(Box::new(interpreter::AgentChannels {
            agent_broadcast_txs: Vec::new(),
            report_rx: None,
            report_tx,
            broadcast_rx: Some(broadcast_rx),
        }));
        if let Ok(Completion::Value(_)) = self.eval(src, false) {
            let _ = self.run_agent_event_loop();
        }
    }

    /// The event loop of a host whose only work is this agent's, such as a test262 runner: run
    /// jobs until no `Atomics.waitAsync` promise or `$262.agent.setTimeout` timer is pending,
    /// blocking between them for the next timeout or a notification from another agent. It gives
    /// up after four seconds in which nothing settles. Evaluation entry points never block on
    /// this work; an embedder with its own event loop uses its checkpoints instead.
    pub fn run_agent_event_loop(&mut self) -> Result<(), crate::InterruptReason> {
        self.interp.activate_gc_heap();
        let result = self.interp.run_agent_event_loop();
        self.interp.gc_task_boundary();
        result
    }

    /// Parse and run `src`. `strict` forces strict mode (used for the test262 strict variant); a
    /// `"use strict"` directive in the source also enables it.
    pub fn eval(&mut self, src: &str, strict: bool) -> Result<Completion, ParseError> {
        self.eval_interruptible(src, strict)
            .map(Self::legacy_completion)
    }

    /// [`Engine::eval`] with host interruption kept distinct from a JavaScript throw.
    pub fn eval_interruptible(
        &mut self,
        src: &str,
        strict: bool,
    ) -> Result<ExecutionOutcome, ParseError> {
        self.eval_interruptible_with_origin(src, strict, None)
    }

    /// [`Engine::eval_interruptible`] for a script whose `Error.prototype.stack` frames report
    /// `origin` (a file path or URL, with the text's starting line and column), or
    /// `<anonymous>` when None.
    pub fn eval_interruptible_with_origin(
        &mut self,
        src: &str,
        strict: bool,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Result<ExecutionOutcome, ParseError> {
        scrubbed(|| self.eval_interruptible_with_origin_inner(src, strict, origin))
    }

    fn eval_interruptible_with_origin_inner(
        &mut self,
        src: &str,
        strict: bool,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Result<ExecutionOutcome, ParseError> {
        let (body, source) = parse_host_script_from(src, strict, origin)?;
        // A top-level `"use strict"` directive prologue turns on strict mode for the whole script.
        let directive_strict = matches!(
            body.first(),
            Some(ast::Stmt::Expr(ast::Expr::Str(s))) if &**s == "use strict"
        );
        self.interp.strict = strict || directive_strict;
        let result = self.interp.run_program_from(&body, Some(source));
        match result {
            Ok(v) => {
                // Run queued promise reactions (the microtask checkpoint after the script).
                if let Err(reason) = self.interp.drain_microtasks_interruptible() {
                    self.interp.gc_task_boundary();
                    return Ok(ExecutionOutcome::Interrupted { reason });
                }
                self.interp.gc_task_boundary();
                Ok(ExecutionOutcome::Value(self.render(&v)))
            }
            Err(interpreter::Abrupt::Throw(thrown)) => {
                crate::jit::perf_error_escaped();
                if let Err(reason) = self.interp.drain_microtasks_interruptible() {
                    self.interp.gc_task_boundary();
                    return Ok(ExecutionOutcome::Interrupted { reason });
                }
                self.interp.gc_task_boundary();
                let Completion::Throw { name, message } = self.describe_throw(thrown) else {
                    unreachable!()
                };
                Ok(ExecutionOutcome::Throw { name, message })
            }
            Err(interpreter::Abrupt::Interrupt(reason)) => {
                self.interp.gc_task_boundary();
                Ok(ExecutionOutcome::Interrupted { reason })
            }
            Err(_) => Ok(ExecutionOutcome::Value(String::new())),
        }
    }

    /// Like [`eval`](Engine::eval), but the script body comes from a precompiled snapshot blob
    /// (see [`compile_snapshot`]) instead of parsing source — the runtime uses this to skip
    /// re-lexing/parsing its static JS glue on every boot. A decode failure (version skew,
    /// corruption) surfaces as `Err(ParseError)` so the caller can fall back to `eval` on the
    /// original source; the resulting AST is otherwise identical to a parsed one, so execution
    /// is byte-for-byte the same.
    pub fn eval_snapshot(&mut self, bytes: &[u8], strict: bool) -> Result<Completion, ParseError> {
        self.eval_snapshot_interruptible(bytes, strict)
            .map(Self::legacy_completion)
    }

    /// [`Engine::eval_snapshot`] with host interruption kept distinct from a JavaScript throw.
    pub fn eval_snapshot_interruptible(
        &mut self,
        bytes: &[u8],
        strict: bool,
    ) -> Result<ExecutionOutcome, ParseError> {
        let body = snapshot::decode(bytes).map_err(|message| ParseError {
            message,
            line: 0,
            at_eof: false,
        })?;
        let directive_strict = matches!(
            body.first(),
            Some(ast::Stmt::Expr(ast::Expr::Str(s))) if &**s == "use strict"
        );
        self.interp.strict = strict || directive_strict;
        let result = self.interp.run_program(&body);
        match result {
            Ok(v) => {
                if let Err(reason) = self.interp.drain_microtasks_interruptible() {
                    self.interp.gc_task_boundary();
                    return Ok(ExecutionOutcome::Interrupted { reason });
                }
                self.interp.gc_task_boundary();
                Ok(ExecutionOutcome::Value(self.render(&v)))
            }
            Err(interpreter::Abrupt::Throw(thrown)) => {
                crate::jit::perf_error_escaped();
                if let Err(reason) = self.interp.drain_microtasks_interruptible() {
                    self.interp.gc_task_boundary();
                    return Ok(ExecutionOutcome::Interrupted { reason });
                }
                self.interp.gc_task_boundary();
                let Completion::Throw { name, message } = self.describe_throw(thrown) else {
                    unreachable!()
                };
                Ok(ExecutionOutcome::Throw { name, message })
            }
            Err(interpreter::Abrupt::Interrupt(reason)) => {
                self.interp.gc_task_boundary();
                Ok(ExecutionOutcome::Interrupted { reason })
            }
            Err(_) => Ok(ExecutionOutcome::Value(String::new())),
        }
    }

    /// Install a host module loader used by dynamic `import()` (and `eval_module`). `loader(specifier,
    /// referrer)` returns the imported module's `(canonical_key, source)`.
    pub fn set_module_loader(
        &mut self,
        loader: impl Fn(&str, &str) -> Option<(String, String)> + 'static,
    ) {
        self.interp.module_loader = Some(std::rc::Rc::new(
            move |s: &str, r: &str, _a: Option<&str>| loader(s, r),
        ));
    }

    /// [`Engine::set_module_loader`], with import attributes: the loader also receives the
    /// import's `with { type: ... }` attribute (`Some("json" | "text" | "bytes")` for the types
    /// the engine synthesizes). An attribute-aware host returns the RAW file contents for those —
    /// text/JSON per its own decoding policy, binary latin-1-decoded (one char per byte) — and
    /// the engine builds the synthetic module (default-exporting the parsed JSON, the string, or
    /// an immutable-backed `Uint8Array`) keyed separately from any ordinary module of the file.
    pub fn set_module_loader_attrs(
        &mut self,
        loader: impl Fn(&str, &str, Option<&str>) -> Option<(String, String)> + 'static,
    ) {
        self.interp.module_loader = Some(std::rc::Rc::new(loader));
    }

    /// Install the asynchronous HostLoadImportedModule hook used by browser embedders. Returning
    /// `true` accepts the request; the embedder must later call [`Engine::finish_dynamic_module_load`]
    /// with the same id. Returning `false` lets the engine fall back to its synchronous loader.
    pub fn set_async_dynamic_module_loader(
        &mut self,
        loader: impl Fn(u64, &str, &str, Option<&str>) -> bool + 'static,
    ) {
        self.set_async_dynamic_module_loader_with_context(
            move |id, _, specifier, referrer, attributes| {
                loader(id, specifier, referrer, attributes)
            },
        );
    }

    /// HostLoadImportedModule retains the referrer's host-defined environment
    /// (ECMA-262 #sec-HostLoadImportedModule). Browser embedders use the second
    /// argument, the initiating host job context, to authorize resource access.
    /// A URL alone cannot distinguish the same module in different Window realms.
    pub fn set_async_dynamic_module_loader_with_context(
        &mut self,
        loader: impl Fn(u64, u64, &str, &str, Option<&str>) -> bool + 'static,
    ) {
        self.interp.dynamic_module_loader = Some(std::rc::Rc::new(loader));
    }

    /// Complete a dynamic-import host request with `(canonical key, source)` or a loading failure.
    /// The corresponding JavaScript promise settles through the normal module link/evaluate path.
    pub fn finish_dynamic_module_load(
        &mut self,
        request_id: u64,
        result: Option<(String, String)>,
    ) -> bool {
        self.interp.activate_gc_heap();
        self.interp.finish_dynamic_module_load(request_id, result)
    }

    /// The base URL of the classic scripts evaluated from now on (HTML: a classic script's base
    /// URL): the referrer of `import()` in their code, including functions they create, which
    /// keep it after the base changes (ECMA-262 GetActiveScriptOrModule). It is also the referrer
    /// when no Script or Module is active (the current Realm's), so a host restores the settings
    /// object's API base URL after running a script.
    pub fn set_import_base(&mut self, base: &str) {
        self.interp.import_base = base.to_string();
    }

    /// Evaluate `src` as an ES module identified by `key`. `loader(specifier, referrer)` resolves an
    /// imported specifier to its `(canonical_key, source)`; it is consulted for every dependency.
    pub fn eval_module(
        &mut self,
        src: &str,
        key: &str,
        loader: impl Fn(&str, &str) -> Option<(String, String)> + 'static,
    ) -> Result<Completion, ParseError> {
        self.eval_module_attrs(src, key, move |s, r, _a| loader(s, r))
    }

    /// [`Engine::eval_module`] with an attribute-aware loader (see
    /// [`Engine::set_module_loader_attrs`] for the contract).
    pub fn eval_module_attrs(
        &mut self,
        src: &str,
        key: &str,
        loader: impl Fn(&str, &str, Option<&str>) -> Option<(String, String)> + 'static,
    ) -> Result<Completion, ParseError> {
        self.eval_module_attrs_interruptible(src, key, loader)
            .map(Self::legacy_completion)
    }

    /// [`Engine::eval_module_attrs`] with host interruption kept distinct from a JavaScript
    /// throw.
    pub fn eval_module_attrs_interruptible(
        &mut self,
        src: &str,
        key: &str,
        loader: impl Fn(&str, &str, Option<&str>) -> Option<(String, String)> + 'static,
    ) -> Result<ExecutionOutcome, ParseError> {
        self.interp.activate_gc_heap();
        self.interp.module_loader = Some(std::rc::Rc::new(loader));
        let result = self.interp.load_module(key, src);
        Ok(match result {
            Ok(_) => match self.interp.drain_microtasks_interruptible() {
                Ok(()) => {
                    self.interp.gc_task_boundary();
                    ExecutionOutcome::Value(String::new())
                }
                Err(reason) => {
                    self.interp.gc_task_boundary();
                    ExecutionOutcome::Interrupted { reason }
                }
            },
            Err(interpreter::Abrupt::Throw(value)) => {
                crate::jit::perf_error_escaped();
                if let Err(reason) = self.interp.drain_microtasks_interruptible() {
                    self.interp.gc_task_boundary();
                    return Ok(ExecutionOutcome::Interrupted { reason });
                }
                self.interp.gc_task_boundary();
                let Completion::Throw { name, message } = self.describe_throw(value) else {
                    unreachable!()
                };
                ExecutionOutcome::Throw { name, message }
            }
            Err(interpreter::Abrupt::Interrupt(reason)) => {
                self.interp.gc_task_boundary();
                ExecutionOutcome::Interrupted { reason }
            }
            Err(_) => ExecutionOutcome::Value(String::new()),
        })
    }

    /// Select the execution tier (see [`Tier`]). `Interp` never touches any codegen
    /// path; `Bytecode` compiles eligible functions after
    /// [`set_tier_threshold`](Engine::set_tier_threshold) calls; `Jit` is the default and lowers
    /// eligible bytecode to native code where the backend supports the host architecture.
    pub fn set_tier(&mut self, tier: Tier) {
        self.interp.tier = tier;
    }

    /// Calls before a function is considered for bytecode compilation (0 = immediately).
    pub fn set_tier_threshold(&mut self, threshold: u32) {
        self.interp.tier_threshold = threshold;
    }

    /// Set the ceiling on simultaneously live objects; values below 100,000 count as 100,000.
    /// An allocation safe point that finds more live objects collects, and if more than the
    /// ceiling survive, throws a `RangeError` instead of exhausting memory. The default scales
    /// with the memory available to the process (at least three million).
    pub fn set_live_object_limit(&mut self, limit: u64) {
        let floor = interpreter::MIN_CONFIGURED_LIVE_OBJECT_LIMIT as u64;
        let limit = limit.clamp(floor, i64::MAX as u64) as i64;
        self.interp.live_object_limit = limit;
        self.interp.gc_next = limit;
    }

    /// The engine's current live-object ceiling (see [`Engine::set_live_object_limit`]).
    pub fn live_object_limit(&self) -> u64 {
        self.interp.live_object_limit as u64
    }

    /// Drain anything written to `console.*` since the last call, as Rust text (lone
    /// surrogates become U+FFFD).
    pub fn take_console(&mut self) -> Vec<String> {
        std::mem::take(&mut self.interp.console)
            .into_iter()
            .map(|line| match jstr::to_text(&line) {
                std::borrow::Cow::Borrowed(_) => line,
                std::borrow::Cow::Owned(text) => text,
            })
            .collect()
    }

    fn render(&mut self, v: &Value) -> String {
        self.interp
            .to_string(v)
            .map(|s| jstr::to_text(&s).into_owned())
            .unwrap_or_default()
    }

    fn describe_throw(&mut self, thrown: Value) -> Completion {
        // Pull the constructor name + message off an Error object; fall back to the rendered value.
        let name = match self.interp.get_member_key(&thrown, "name") {
            Ok(Value::Undefined) | Err(_) => {
                // No own/inherited `name` (e.g. Test262Error): use the constructor's name.
                match self.interp.get_member_key(&thrown, "constructor") {
                    Ok(ctor @ Value::Obj(_)) => match self.interp.get_member_key(&ctor, "name") {
                        Ok(Value::Undefined) | Err(_) => String::new(),
                        Ok(v) => self.render(&v),
                    },
                    _ => String::new(),
                }
            }
            Ok(v) => self.render(&v),
        };
        let message = match &thrown {
            Value::Obj(_) => match self.interp.get_member_key(&thrown, "message") {
                Ok(Value::Undefined) | Err(_) => String::new(),
                Ok(v) => self.render(&v),
            },
            other => self.render(other),
        };
        Completion::Throw { name, message }
    }

    fn legacy_completion(outcome: ExecutionOutcome) -> Completion {
        match outcome {
            ExecutionOutcome::Value(value) => Completion::Value(value),
            ExecutionOutcome::Throw { name, message } => Completion::Throw { name, message },
            ExecutionOutcome::Interrupted { reason } => Completion::Throw {
                name: Self::legacy_interrupt_name(reason).to_string(),
                message: reason.message().to_string(),
            },
        }
    }

    fn legacy_interrupt_name(reason: InterruptReason) -> &'static str {
        match reason {
            InterruptReason::DeadlineExceeded => "QuotaExceededError",
            InterruptReason::Cancelled | InterruptReason::UserNavigation => "AbortError",
        }
    }
}

#[cfg(feature = "embed")]
pub mod embed {
    //! The curated embedder surface (`feature = "embed"`), for runtime layers (event loop, host
    //! APIs) built on top of the engine. Gated because everything here is a semver commitment on a
    //! published crate; it stabilizes together with the `trust-js-host`/`trust-js-runtime` crates.
    //!
    //! # Scopes and values
    //!
    //! A JavaScript value in Rust is a [`Value<'s>`](Value): a primitive, or an opaque handle to a
    //! string, Symbol, BigInt or object cell ([`JsString`], [`Symbol`], [`BigInt`], [`Object`]).
    //! The lifetime `'s` names the *scope* that produced the value, and every value lives exactly
    //! as long as its scope: it cannot be stored in a `'static` structure (host state, a
    //! thread-local, a closure kept by the engine), returned from its scope, mixed with another
    //! scope's values (the brand is invariant, so two scopes' values never unify), or sent to
    //! another thread. A scope is
    //!
    //! - one native function call: a [`NativeFn`] is
    //!   `for<'s> fn(&mut Ctx<'s>, Value<'s>, &[Value<'s>]) -> Result<Value<'s>, Value<'s>>`, so
    //!   its receiver, arguments and results belong to the call; or
    //! - one [`Engine::scope`](crate::Engine::scope) closure, where engine-level host code (an
    //!   event loop firing a callback, a test reading a global) runs:
    //!
    //! ```
    //! let mut engine = trust_js::Engine::new();
    //! engine.eval("globalThis.double = (x) => x * 2", false).unwrap();
    //! let n = engine.scope(|cx| {
    //!     let global = cx.global_this();
    //!     let double = cx.member_get(&global, "double").unwrap();
    //!     let result = cx.call_function(&double, trust_js::embed::Value::Undefined,
    //!                                   &[trust_js::embed::Value::Num(21.0)]);
    //!     result.ok().and_then(|value| value.as_num_opt())
    //! });
    //! assert_eq!(n, Some(42.0));
    //! ```
    //!
    //! The scope's [`Ctx<'s>`](Ctx) carries every method that takes or returns a value; strings
    //! and BigInts are made through it ([`Ctx::str`], [`Ctx::bigint_from_i64`]), so they belong to
    //! its heap. While a scope runs, the values in its Rust stack frames are roots of every
    //! collection. Values in Rust heap memory are not: a list of values held across a call that can
    //! run JavaScript (and so collect) is a [`ValueList`], whose values stay rooted.
    //!
    //! # Handles
    //!
    //! A value kept past its scope is kept in a rooted handle: a [`Handle`] keeps its value alive
    //! wherever it is stored, a [`WeakHandle`] names one without keeping it (and is cleared, before
    //! the cell is freed, once a collection finds it dead), and a [`CountedHandle`] is strong while
    //! its count is above 0 (Node-API's references). Each is made from a value through a scope's
    //! context and read back into a scope ([`Handle::new`], [`Handle::get`]), is `!Send`, and is
    //! refused ([`HandleError`]) with another Engine's context or once its Engine has dropped,
    //! when it is inert.
    //!
    //! ```
    //! use trust_js::embed::{Handle, Value, ValueList};
    //! let mut engine = trust_js::Engine::new();
    //! // A list of values built while JavaScript runs (and may collect) between the pushes.
    //! let kept = engine.scope(|cx| {
    //!     let mut list = ValueList::new();
    //!     for n in 0..3 {
    //!         list.push(cx.eval_value(&format!("({{n: {n}}})")).unwrap().unwrap());
    //!         cx.collect_garbage_for_host();
    //!     }
    //!     let array = list.into_array(cx);
    //!     Handle::new(cx, &array).unwrap()
    //! });
    //! engine.collect_garbage_at_idle();
    //! let length = engine.scope(|cx| {
    //!     let array = kept.get(cx).unwrap();
    //!     cx.member_get(&array, "length").unwrap().as_num_opt()
    //! });
    //! assert_eq!(length, Some(3.0));
    //! ```
    //!
    //! What a scoped value cannot do is checked at compile time. A value cannot be stored in a
    //! `'static` structure such as host state, so it cannot outlive its call:
    //!
    //! ```compile_fail,E0521
    //! use trust_js::embed::{Ctx, Value};
    //! struct Kept<'s>(Value<'s>);
    //! fn keep<'s>(
    //!     cx: &mut Ctx<'s>,
    //!     _: Value<'s>,
    //!     args: &[Value<'s>],
    //! ) -> Result<Value<'s>, Value<'s>> {
    //!     // `OpState` holds `'static` types only.
    //!     cx.op_state().put(Kept(args[0]));
    //!     Ok(Value::Undefined)
    //! }
    //! ```
    //!
    //! It cannot be returned out of its scope, so a value kept past its Engine does not compile
    //! either (a [`Handle`] kept past its Engine is inert):
    //!
    //! ```compile_fail
    //! let kept = {
    //!     let mut engine = trust_js::Engine::new();
    //!     engine.scope(|cx| cx.global_this())
    //! };
    //! ```
    //!
    //! nor be kept in a collection outside its scope, across the collections between scopes (a
    //! [`ValueList`] keeps values inside one):
    //!
    //! ```compile_fail,E0521
    //! let mut engine = trust_js::Engine::new();
    //! let mut kept = Vec::new();
    //! engine.scope(|cx| kept.push(cx.new_object()));
    //! engine.collect_garbage_at_idle();
    //! ```
    //!
    //! Two scopes' values never meet, even of the same thread's Engines:
    //!
    //! ```compile_fail,E0521
    //! let (mut first, mut second) = (trust_js::Engine::new(), trust_js::Engine::new());
    //! first.scope(|a| {
    //!     second.scope(|b| {
    //!         let object = trust_js::embed::Value::Obj(a.new_object());
    //!         let global = b.global_this();
    //!         let _ = b.member_set(&global, "foreign", object);
    //!     })
    //! });
    //! ```
    //!
    //! and neither a value nor a handle crosses to another thread:
    //!
    //! ```compile_fail,E0277
    //! let mut engine = trust_js::Engine::new();
    //! let handle = engine.scope(|cx| {
    //!     let global = cx.global_this();
    //!     trust_js::embed::Handle::new(cx, &global).unwrap()
    //! });
    //! std::thread::spawn(move || drop(handle));
    //! ```
    //!
    //! ```compile_fail,E0277
    //! let mut engine = trust_js::Engine::new();
    //! engine.scope(|cx| {
    //!     let global = cx.global_this();
    //!     std::thread::scope(|threads| {
    //!         threads.spawn(move || drop(global));
    //!     });
    //! });
    //! ```
    //!
    //! # Host state and the host traits
    //!
    //! Rust state lives in [`OpState`] (typed slots and a [`ResourceTable`]), reached through
    //! [`Ctx::op_state`] or [`Engine::op_state`](crate::Engine::op_state). It holds JavaScript
    //! values in handles and needs no tracing. A host whose values must live exactly as long as
    //! their JavaScript owners do (a DOM) reports a graph instead, by registering a [`HostTrace`]
    //! type ([`OpState::register_trace`]): at every collection its visitor ([`HostGcVisitor`])
    //! receives weak handles as roots, as logical edges from an owner, and as the JavaScript ends
    //! of a native graph ([`NativeGcId`]). A [`HostGc`] type also hears about each job's end and
    //! sweeps its caches between a collection's phases, after the weak handles whose values died
    //! have been cleared. These callbacks run no JavaScript and allocate nothing in the engine.
    //! The retained-memory reporters ([`RetainedBytes`], [`RetainedMemory`],
    //! [`RetainedExternalMemory`], [`NativeCallableRetained`]) describe Rust allocations for the
    //! managed-memory diagnostics; the JavaScript values a host keeps are its handles.

    pub use crate::handles::{CountedHandle, Handle, HandleError, WeakHandle};
    pub use crate::host::{
        HostGc, HostGcVisitor, HostRetainedMemoryVisitor, HostTrace, NativeGcId, OpState,
        ResourceId, ResourceTable, RetainedBytes, RetainedExternalAllocation,
        RetainedExternalMemory, RetainedMemory,
    };
    pub use crate::interpreter::ArrayBufferBytes;
    pub use crate::scope::{
        BigInt, Ctx, EvalError, HostJobContextEnter, HostJobContextLeave, IndexedLength,
        IndexedNames, JsString, NativeCaptureFn, NativeFn, Object, Symbol, Value, ValueList,
    };
    pub use crate::value::{
        NativeCallableRetained, NativeCaptures, NativeRetainedMemoryVisitor,
        RetainedManagedAllocation,
    };
    pub use crate::{RealmEntry, RealmError};

    /// Rust text in the engine representation used by [`Ctx::string_from_engine_text`]: every
    /// character in U+10F800..=U+10FFFF becomes its pair of private-use scalars. Use it where
    /// a host stores decoded text (a parsed document, a network message) beside engine text.
    pub fn text_to_engine(text: &str) -> std::borrow::Cow<'_, str> {
        crate::jstr::from_text(text)
    }

    /// Engine text (from [`JsString::as_engine_text`] or [`Ctx::coerce_engine_text`]) as Rust
    /// text: the Web IDL `USVString` conversion, in which every lone surrogate becomes U+FFFD.
    pub fn engine_to_text(engine_text: &str) -> std::borrow::Cow<'_, str> {
        crate::jstr::to_text(engine_text)
    }
}

/// The engine form of an interrupt-aware entry point's failure (`embed::EvalError`).
#[cfg(feature = "embed")]
pub(crate) enum EvalFailure {
    Throw(Value),
    Interrupted(InterruptReason),
}

/// The engine side of the embedder's evaluation and callback entry points (`embed::Ctx`).
#[cfg(feature = "embed")]
impl interpreter::Interp {
    /// Set the opaque host settings token captured by subsequently-created promise reactions.
    ///
    /// ECMA-262 associates a PromiseReactionJob with its handler Realm, while HTML uses that
    /// Realm to restore the corresponding environment settings object. Embedders that represent
    /// more than one host settings object inside a single engine Realm can use this token to
    /// preserve that additional distinction without changing ECMAScript-visible values.
    pub(crate) fn set_host_job_context(&mut self, context: u64) {
        self.switch_host_job_context(context);
    }

    /// Parse and evaluate one ECMAScript Script Record in this realm while a host operation is
    /// already running, its `Error.prototype.stack` frames reporting `origin`.
    ///
    /// HTML's "run a classic script" algorithm can synchronously enter a new ScriptEvaluation
    /// from parser or DOM insertion work. This is deliberately not `eval`: ScriptEvaluation uses
    /// the Realm's persistent `[[GlobalEnv]]` as both its lexical and variable environment, while
    /// indirect eval gives lexical declarations a fresh eval-only environment. As with
    /// `Ctx::eval_value_interruptible`, the embedder owns the following microtask checkpoint.
    pub(crate) fn eval_classic_script(
        &mut self,
        src: &str,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Result<Result<Value, EvalFailure>, ParseError> {
        let (body, source) = parse_host_script_from(src, false, origin)?;
        Ok(self.eval_classic_body(&body, Some(source)))
    }

    /// Decode and evaluate a precompiled ECMAScript Script Record in the active Realm: the
    /// snapshot counterpart of [`Self::eval_classic_script`], with which browser embedders
    /// install the same static platform bootstrap in many Window Realms without reparsing it.
    pub(crate) fn eval_classic_snapshot(
        &mut self,
        bytes: &[u8],
    ) -> Result<Result<Value, EvalFailure>, ParseError> {
        let body = snapshot::decode(bytes).map_err(|message| ParseError {
            message,
            line: 0,
            at_eof: false,
        })?;
        Ok(self.eval_classic_body(&body, None))
    }

    /// [`Self::eval_classic_snapshot`] for static host bootstrap code that many Realms of this
    /// Agent install, such as a browser's Window platform prelude.
    ///
    /// The first call decodes the snapshot; the Agent retains the program and every later call
    /// with the same `&'static` bytes evaluates that same Script body in the active Realm.
    /// Each evaluation is a separate ScriptEvaluation (ECMA-262 ScriptEvaluation): Global-
    /// DeclarationInstantiation, function objects, environments and template objects belong to
    /// the Realm that runs it, as when the source is parsed for that Realm. What is shared is
    /// implementation work cached on the immutable AST: hoisting plans, bytecode and native
    /// code. Compiled code owns no JavaScript objects, and each of its caches is validated
    /// against the live receiver, environment or Realm before it is used.
    ///
    /// Because the Parse Nodes are shared, evaluating the same snapshot twice in one Realm
    /// behaves like evaluating one Script twice rather than parsing its source again: tagged
    /// templates reuse that Realm's template objects (GetTemplateObject keys the Realm's
    /// [[TemplateMap]] by Parse Node). Different Realms always get their own.
    pub(crate) fn eval_shared_classic_snapshot(
        &mut self,
        bytes: &'static [u8],
    ) -> Result<Result<Value, EvalFailure>, ParseError> {
        let body = self
            .shared_snapshot_program(bytes)
            .map_err(|message| ParseError {
                message,
                line: 0,
                at_eof: false,
            })?;
        // The bootstrap compiles its outermost functions one inside the other; each compilation
        // scans the functions nested in it. Summarize each still-encoded one once.
        let _summaries = crate::bytecode::capture_summaries();
        Ok(self.eval_classic_body(&body, None))
    }

    /// Evaluate a parsed or decoded Script body as one ScriptEvaluation in the active Realm.
    fn eval_classic_body(
        &mut self,
        body: &[ast::Stmt],
        source: Option<std::rc::Rc<stack_trace::ScriptSource>>,
    ) -> Result<Value, EvalFailure> {
        match self.run_classic_program(body, source) {
            Ok(Value::Empty) => Ok(Value::Undefined),
            Ok(value) => Ok(value),
            Err(interpreter::Abrupt::Throw(value)) => {
                crate::jit::perf_error_escaped();
                Err(EvalFailure::Throw(value))
            }
            Err(interpreter::Abrupt::Interrupt(reason)) => Err(EvalFailure::Interrupted(reason)),
            Err(_) => Ok(Value::Undefined),
        }
    }

    /// `Ctx::eval_value_interruptible_with_origin`: one synchronous script job whose completion
    /// comes back as a value, with no microtask checkpoint (the caller owns the event loop). The
    /// spec's EMPTY completion (a value-less final statement) is lowered to `undefined`.
    pub(crate) fn eval_value_entry(
        &mut self,
        src: &str,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Result<Result<Value, EvalFailure>, ParseError> {
        let (body, source) = parse_host_script_from(src, false, origin)?;
        let directive_strict = matches!(
            body.first(),
            Some(ast::Stmt::Expr(ast::Expr::Str(s))) if &**s == "use strict"
        );
        self.strict = directive_strict;
        let result = match self.run_program_from(&body, Some(source)) {
            Ok(Value::Empty) => Ok(Value::Undefined),
            Ok(value) => Ok(value),
            Err(interpreter::Abrupt::Throw(value)) => {
                crate::jit::perf_error_escaped();
                Err(EvalFailure::Throw(value))
            }
            Err(interpreter::Abrupt::Interrupt(reason)) => {
                self.gc_task_boundary();
                Err(EvalFailure::Interrupted(reason))
            }
            Err(_) => Ok(Value::Undefined),
        };
        // This embedding entry runs one synchronous ECMAScript job. Promise jobs, if any, are
        // deliberately owned by the caller and begin with a fresh [[KeptAlive]] list.
        self.clear_kept_objects();
        Ok(result)
    }

    /// `Ctx::eval_snapshot_value_interruptible`: [`Self::eval_value_entry`] for a snapshot.
    pub(crate) fn eval_snapshot_value_entry(
        &mut self,
        bytes: &[u8],
    ) -> Result<Result<Value, EvalFailure>, ParseError> {
        let result = self.eval_classic_snapshot(bytes)?;
        if matches!(&result, Err(EvalFailure::Interrupted(_))) {
            self.gc_task_boundary();
        }
        self.clear_kept_objects();
        Ok(result)
    }

    /// `Ctx::eval_shared_snapshot_value_interruptible`: [`Self::eval_snapshot_value_entry`]
    /// through the Agent's shared program table.
    pub(crate) fn eval_shared_snapshot_value_entry(
        &mut self,
        bytes: &'static [u8],
    ) -> Result<Result<Value, EvalFailure>, ParseError> {
        let result = self.eval_shared_classic_snapshot(bytes)?;
        if matches!(&result, Err(EvalFailure::Interrupted(_))) {
            self.gc_task_boundary();
        }
        self.clear_kept_objects();
        Ok(result)
    }

    /// `Ctx::call_callback_interruptible`: call `func` as the event loop fires a callback, with
    /// a captured host callback source context; host interruption stays control flow.
    pub(crate) fn call_callback_entry(
        &mut self,
        func: &Value,
        this: Value,
        args: &[Value],
        script_caller: &Value,
    ) -> Result<Value, EvalFailure> {
        if let Err(abrupt) = self.interrupt_poll_force() {
            let error = match abrupt {
                interpreter::Abrupt::Interrupt(reason) => EvalFailure::Interrupted(reason),
                interpreter::Abrupt::Throw(value) => EvalFailure::Throw(value),
                _ => EvalFailure::Throw(Value::Undefined),
            };
            self.gc_task_boundary();
            return Err(error);
        }
        let result = self
            .with_callback_script_caller(script_caller, |interp| {
                interp.call_callback(func.clone(), this, args)
            })
            .map_err(|abrupt| match abrupt {
                interpreter::Abrupt::Throw(value) => EvalFailure::Throw(value),
                interpreter::Abrupt::Interrupt(reason) => EvalFailure::Interrupted(reason),
                _ => EvalFailure::Throw(Value::Undefined),
            });
        self.clear_kept_objects();
        if matches!(result, Err(EvalFailure::Interrupted(_))) {
            self.gc_task_boundary();
        }
        result
    }

    /// Drain the promises rejected without a handler, as `(context, promise, reason)` in
    /// rejection order (HTML's about-to-be-notified rejected promises list).
    pub(crate) fn take_unhandled_rejection_values(&mut self) -> Vec<(u64, Value, Value)> {
        if self.unhandled_rejections.is_empty() {
            return Vec::new();
        }
        let mut rejections: Vec<_> = std::mem::take(&mut self.unhandled_rejections)
            .into_values()
            .collect();
        rejections.sort_unstable_by_key(|(order, ..)| *order);
        rejections
            .into_iter()
            .map(|(_, promise, reason, context)| (context, promise, reason))
            .collect()
    }

    /// `Engine::run_microtasks`: a microtask checkpoint, then the task boundary.
    pub(crate) fn run_microtasks_entry(&mut self) {
        self.activate_gc_heap();
        self.drain_microtasks();
        self.gc_task_boundary();
    }

    /// `Engine::run_microtasks_interruptible`: a checkpoint keeping a host interruption as
    /// control flow (pending jobs are discarded when a running job is killed).
    pub(crate) fn run_microtasks_interruptible_entry(
        &mut self,
    ) -> Result<(), crate::InterruptReason> {
        self.activate_gc_heap();
        if let Err(reason) = self.drain_microtasks_interruptible() {
            self.gc_task_boundary();
            return Err(reason);
        }
        self.gc_task_boundary();
        Ok(())
    }

    /// Whether promise-reaction jobs are queued.
    pub(crate) fn has_pending_jobs(&self) -> bool {
        !self.microtasks.is_empty()
            || !self.pending_finalization_cleanup.is_empty()
            || self.async_waits_ready()
    }

    /// `Ctx::enter_realm`: make `realm_global`'s Realm the active one, keeping the caller's on
    /// the heap's stack of entered Realms (`GcState::entered_realms`), whose states are roots.
    pub(crate) fn enter_embed_realm_entry(
        &mut self,
        realm_global: &Value,
    ) -> Result<RealmEntry, Value> {
        let (key, saved) = self.enter_embed_realm(realm_global)?;
        let mut entered = self.gc_heap.entered_realms.borrow_mut();
        entered.push(crate::value::EnteredRealm {
            interp: std::ptr::from_ref(self) as usize,
            key,
            saved,
        });
        Ok(RealmEntry {
            heap: crate::value::heap_id(&self.gc_heap),
            depth: entered.len(),
        })
    }

    /// `Ctx::leave_realm`: restore the Realm that was active when `entry` was entered. Panics
    /// unless `entry` is this interpreter's innermost.
    pub(crate) fn leave_embed_realm_entry(&mut self, entry: RealmEntry) {
        let innermost = {
            let mut entered = self.gc_heap.entered_realms.borrow_mut();
            assert!(
                entry.heap == crate::value::heap_id(&self.gc_heap)
                    && entry.depth == entered.len()
                    && entered
                        .last()
                        .is_some_and(|last| last.interp == std::ptr::from_ref(self) as usize),
                "a RealmEntry is left by its own Engine, innermost first"
            );
            entered.pop().expect("the innermost entry")
        };
        self.leave_embed_realm(innermost.key, innermost.saved);
    }
}

/// Why a Realm could not be entered from the Engine level ([`Engine::enter_realm`],
/// [`Engine::with_embed_realm`]).
#[cfg(feature = "embed")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealmError {
    /// The handle naming the Realm's global could not be read: another Engine's, or its Engine
    /// has dropped.
    Handle(embed::HandleError),
    /// The value is not the global object of one of this Engine's Realms.
    NotARealm,
}

#[cfg(feature = "embed")]
impl std::fmt::Display for RealmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealmError::Handle(error) => write!(f, "{error}"),
            RealmError::NotARealm => {
                f.write_str("not the global object of one of this engine's realms")
            }
        }
    }
}

#[cfg(feature = "embed")]
impl std::error::Error for RealmError {}

/// Embedder methods (`feature = "embed"`). Engine-level code that takes or returns JavaScript
/// values runs in a [`scope`](Engine::scope); what needs no value (configuration, checkpoints,
/// host state) is here too.
#[cfg(feature = "embed")]
impl Engine {
    /// Run `f` in a new scope of this Engine: its [`embed::Ctx`] makes and uses values, which
    /// live as long as the scope (`'s`), so none of them can be stored, returned from `f` or
    /// meet another scope's. Values the embedder keeps past the scope are kept in
    /// [`embed::Handle`]s.
    pub fn scope<R>(&mut self, f: impl for<'s> FnOnce(&mut embed::Ctx<'s>) -> R) -> R {
        self.interp.activate_gc_heap();
        // SAFETY: `f` is generic over the scope's lifetime, so no value of the scope leaves it
        // or meets another scope's, and the context borrows the Engine while the scope runs.
        f(unsafe { embed::Ctx::new(&mut self.interp) })
    }

    /// Host state ([`embed::OpState`]): typed per-subsystem slots and the resource table.
    pub fn op_state(&mut self) -> &mut embed::OpState {
        self.interp.op_state()
    }

    /// Shorthand for `op_state().get_mut::<T>()`.
    pub fn host_mut<T: std::any::Any>(&mut self) -> Option<&mut T> {
        self.interp.host_mut()
    }

    /// Shorthand for `op_state().get::<T>()`.
    pub fn host<T: std::any::Any>(&self) -> Option<&T> {
        self.interp.host()
    }

    /// Set the opaque host settings token captured by subsequently-created promise reactions
    /// (see [`embed::Ctx::set_host_job_context`]).
    pub fn set_host_job_context(&mut self, context: u64) {
        self.interp.activate_gc_heap();
        self.interp.set_host_job_context(context);
    }

    /// The opaque host settings token of the active host operation.
    pub fn host_job_context(&self) -> u64 {
        self.interp.host_job_context()
    }

    /// Retire one embedder settings object after its owning Document is destroyed (see
    /// [`embed::Ctx::release_host_job_context`]).
    pub fn release_host_job_context(&mut self, context: u64) -> bool {
        self.interp.activate_gc_heap();
        self.interp.release_host_job_context(context)
    }

    /// Run an engine-level host operation in the Realm whose global `realm_global` holds, then
    /// restore the caller's Realm. This is the task counterpart of
    /// [`embed::Ctx::with_embed_realm`]: embedders use it when an asynchronous completion needs
    /// entry points that live on [`Engine`] itself, such as module evaluation.
    pub fn with_embed_realm<R>(
        &mut self,
        realm_global: &embed::Handle,
        operation: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, RealmError> {
        let entry = self.enter_realm(realm_global)?;
        let result = operation(self);
        self.leave_realm(entry);
        Ok(result)
    }

    /// [`Engine::with_embed_realm`] for an operation that spans its embedder's own state as
    /// well as the engine: the Realm whose global `realm_global` holds is the active one until
    /// the returned entry is passed to [`Engine::leave_realm`]. Entries nest like
    /// `with_embed_realm` calls: leave the innermost first.
    pub fn enter_realm(&mut self, realm_global: &embed::Handle) -> Result<RealmEntry, RealmError> {
        self.scope(|cx| {
            let global = realm_global.get(cx).map_err(RealmError::Handle)?;
            cx.enter_realm(&global).map_err(|_| RealmError::NotARealm)
        })
    }

    /// Restore the Realm that was active when `entry` was entered, keeping what the entered
    /// Realm's operations changed. Panics unless `entry` is this Engine's innermost.
    pub fn leave_realm(&mut self, entry: RealmEntry) {
        self.interp.activate_gc_heap();
        self.interp.leave_embed_realm_entry(entry);
    }

    /// Install this realm's wall-clock source, in milliseconds since the Unix epoch.
    ///
    /// The callback is realm-local and may capture mutable embedder state. `Date`, `Date.now`,
    /// and `Temporal.Now` all read this same source. This is the engine side of HTML
    /// `HostSystemUTCEpochNanoseconds(global)`; it deliberately takes precedence over the legacy
    /// process-wide [`set_host_clock`] fallback.
    pub fn set_wall_clock(&mut self, clock: impl Fn() -> f64 + 'static) {
        self.interp.wall_clock = Some(std::rc::Rc::new(clock));
    }

    /// Install callbacks that enter and leave an embedder's opaque settings context around a
    /// promise job. The callbacks are omitted for ordinary single-settings embedders and add no
    /// work to their job path.
    pub fn set_host_job_context_hooks(
        &mut self,
        enter: embed::HostJobContextEnter,
        leave: embed::HostJobContextLeave,
    ) {
        let (enter, leave) = crate::scope::host_job_context_hooks(enter, leave);
        self.interp.host_job_context_enter = Some(enter);
        self.interp.host_job_context_leave = Some(leave);
    }

    /// Opt in to HTML HostMakeJobCallback/HostCallJobCallback: JobCallback registration saves
    /// the incumbent script-caller global and the active script (as a script execution context
    /// whose ScriptOrModule is the active Script or Module), and invoking the callback restores
    /// both. The script caller preserves source identity independently of the callback's
    /// execution Realm; the script execution context lets `import()` reached from a built-in
    /// callback (`promise.then(eval)`) resolve against the registering script's base.
    /// Non-browser hosts must leave this disabled, as required by ECMA-262 §9.5.2–9.5.3.
    pub fn set_job_callback_script_caller_capture(&mut self, enabled: bool) {
        self.interp.capture_job_script_caller = enabled;
    }

    /// Define `globalThis.<name>` as a native function (non-enumerable, like built-ins).
    pub fn define_global(&mut self, name: &str, len: usize, f: embed::NativeFn) {
        self.interp.activate_gc_heap();
        let global = self.interp.global.clone();
        self.interp
            .def_method(&global, name, len, crate::scope::native_fn(f));
    }

    /// Define `globalThis.<name>` as a namespace object (like `Math`) with the given
    /// `(name, arity, fn)` native methods.
    pub fn define_namespace(&mut self, name: &str, ops: &[(&str, usize, embed::NativeFn)]) {
        self.interp.activate_gc_heap();
        let ns = self.interp.new_object();
        for (op, len, f) in ops {
            let f = crate::scope::native_fn(*f);
            self.interp.def_method(&ns, op, *len, f);
            // Qualify the diagnostic identity (`ns.op`); the bare `op` label recorded by
            // `def_method` would otherwise collide across namespaces sharing method names.
            crate::jit::perf_native_relabel(f as usize, format!("{name}.{op}"));
        }
        self.interp.global.borrow_mut().props.insert(
            crate::value::encode_string_key(name).as_ref(),
            crate::value::Property::builtin(Value::Obj(ns)),
        );
    }

    /// Drain the microtask (promise-reaction) queue to quiescence.
    pub fn run_microtasks(&mut self) {
        self.interp.run_microtasks_entry();
    }

    /// Run a microtask checkpoint while preserving host interruption as control flow. Pending jobs
    /// are discarded when a running job is killed; they must not be resumed as a later task.
    pub fn run_microtasks_interruptible(&mut self) -> Result<(), crate::InterruptReason> {
        scrubbed(|| self.interp.run_microtasks_interruptible_entry())
    }

    /// Collect garbage when the host event loop is about to block for external input.
    ///
    /// This complements allocation-triggered task checks: garbage spread across many
    /// individually low-churn timer tasks is reclaimed once, at the natural idle boundary.
    #[inline(always)]
    pub fn collect_garbage_at_idle(&mut self) -> i64 {
        self.interp.collect_garbage_for_host()
    }

    /// Defer optional task-boundary collections during a short latency-sensitive
    /// host interval. This does not suppress allocation-triggered collection,
    /// memory limits, microtasks, or ClearKeptObjects. The host must end the
    /// interval and service pending collection when its input queue quiets down.
    pub fn defer_task_garbage_collection(&mut self, deferred: bool) {
        self.interp.gc_task_deferred = deferred;
    }

    /// Whether a checkpoint requested an optional collection that is pending.
    pub fn has_pending_task_garbage_collection(&self) -> bool {
        self.interp.gc_task_pending
    }

    /// Service a deferred request between host tasks, after their microtask
    /// checkpoints. Returns zero while deferral remains enabled or no request
    /// is pending. Unlike an explicit idle collection, this never forces a collection
    /// of an otherwise quiet heap. Inlined: the stack below the host's frame is zeroed before
    /// the boundary's frames are built (see `Interp::gc_collect`).
    #[inline(always)]
    pub fn collect_pending_task_garbage(&mut self) -> i64 {
        if !self.interp.gc_task_pending || self.interp.gc_task_deferred {
            return 0;
        }
        #[cfg(not(target_arch = "wasm32"))]
        gc_core::scrub_stack(gc_core::ENTRY_SCRUB_BYTES);
        self.interp.activate_gc_heap();
        self.interp.gc_task_boundary()
    }

    /// Drain and return the reasons of promises rejected without a handler (after a microtask
    /// checkpoint, these are genuine unhandled rejections), each held by a handle. The runtime
    /// reports them; the bare engine ignores them, so test262 semantics are unaffected.
    pub fn take_unhandled_rejections(&mut self) -> Vec<embed::Handle> {
        self.take_unhandled_rejections_full()
            .into_iter()
            .map(|(_promise, reason)| reason)
            .collect()
    }

    /// [`Engine::take_unhandled_rejections`], keeping the promise alongside each reason (what a
    /// global `unhandledrejection` handler receives as `event.promise` / `event.reason`).
    pub fn take_unhandled_rejections_full(&mut self) -> Vec<(embed::Handle, embed::Handle)> {
        self.take_unhandled_rejections_in_contexts()
            .into_iter()
            .map(|(_, promise, reason)| (promise, reason))
            .collect()
    }

    /// [`Engine::take_unhandled_rejections_full`] as `(context, promise, reason)`, with the host
    /// settings context active when each promise was rejected: HTML's HostPromiseRejectionTracker
    /// puts the promise on that settings object's global's about-to-be-notified rejected promises
    /// list, so an embedder with several Window Realms notifies each global of its own.
    pub fn take_unhandled_rejections_in_contexts(
        &mut self,
    ) -> Vec<(u64, embed::Handle, embed::Handle)> {
        self.scope(|cx| cx.take_unhandled_rejections_in_contexts())
    }

    /// The HostPromiseRejectionTracker "handle" operations on promises the host added to its
    /// outstanding rejected promises weak set ([`embed::Ctx::add_outstanding_rejection`]), as
    /// `(promise, reason)`: HTML queues a `rejectionhandled` event for each.
    pub fn take_handled_rejections(&mut self) -> Vec<(embed::Handle, embed::Handle)> {
        self.scope(|cx| cx.take_handled_rejections())
    }

    /// [`Engine::take_handled_rejections`] as `(context, promise, reason)`, with the host
    /// settings context whose global's outstanding rejected promises weak set held each promise:
    /// its `rejectionhandled` event fires at that global.
    pub fn take_handled_rejections_in_contexts(
        &mut self,
    ) -> Vec<(u64, embed::Handle, embed::Handle)> {
        self.scope(|cx| cx.take_handled_rejections_in_contexts())
    }

    /// Whether promise-reaction jobs are queued (the loop uses this to decide when a turn is
    /// really over).
    pub fn has_pending_jobs(&self) -> bool {
        self.interp.has_pending_jobs()
    }

    /// When the earliest pending `Atomics.waitAsync` times out. Its promise resolves at the first
    /// job checkpoint from then on, so a host event loop should run one at this time
    /// (HostEnqueueTimeoutJob). A pending wait does not otherwise keep the loop alive.
    pub fn next_atomics_wait_deadline(&self) -> Option<std::time::Instant> {
        self.interp.next_async_wait_deadline()
    }

    /// Run a single queued job; `false` when the queue was empty or a host interruption stopped
    /// the job (use [`Self::run_one_job_interruptible`] to tell the two apart).
    pub fn run_one_job(&mut self) -> bool {
        self.run_one_job_interruptible().unwrap_or(false)
    }

    /// Run one queued job, preserving a host interruption from the job's JavaScript callback.
    pub fn run_one_job_interruptible(&mut self) -> Result<bool, crate::InterruptReason> {
        self.interp.activate_gc_heap();
        // Resolving the Atomics.waitAsync promises that can settle is a job, which may queue
        // reaction jobs.
        let settled = self.interp.settle_async_waits();
        match self.interp.microtasks.pop_front() {
            Some(job) => {
                self.interp.run_job_interruptible(job)?;
                self.interp.clear_kept_objects();
                Ok(true)
            }
            None => match self.interp.pending_finalization_cleanup.pop_front() {
                Some(registry) => {
                    self.interp.run_finalization_cleanup_job(registry)?;
                    Ok(true)
                }
                None => Ok(settled),
            },
        }
    }
}

#[cfg(test)]
mod call_error_name_tests;
#[cfg(test)]
mod call_guard_tests;
#[cfg(test)]
mod entry_tests;
#[cfg(test)]
mod tail_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod iterator_result_tests;

#[cfg(all(test, feature = "embed"))]
mod module_host_context_tests {
    use super::*;
    use crate::bytecode::Tier;
    #[test]
    fn asynchronous_module_hook_receives_initiating_host_context() {
        use std::{cell::RefCell, rc::Rc};
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            let seen = Rc::new(RefCell::new(Vec::new()));
            let captured = seen.clone();
            engine.set_async_dynamic_module_loader_with_context(
                move |id, context, specifier, _, _| {
                    captured
                        .borrow_mut()
                        .push((id, context, specifier.to_string()));
                    true
                },
            );
            engine.set_host_job_context(17);
            engine
                .eval("import('first').catch(() => {});", false)
                .unwrap();
            engine.set_host_job_context(23);
            engine
                .eval("import('second').catch(() => {});", false)
                .unwrap();
            let seen = seen.borrow();
            assert_eq!(seen.len(), 2);
            assert_eq!((seen[0].1, seen[0].2.as_str()), (17, "first"));
            assert_eq!((seen[1].1, seen[1].2.as_str()), (23, "second"));
            assert_ne!(seen[0].0, seen[1].0);
            assert!(engine.finish_dynamic_module_load(seen[0].0, None));
            assert!(engine.finish_dynamic_module_load(seen[1].0, None));
        }
    }
}

/// A Realm [`embed::Ctx::enter_realm`] or [`Engine::enter_realm`] made active, which
/// [`Engine::leave_realm`] (or [`embed::Ctx::leave_realm`]) leaves again: a token of its Engine's
/// stack of entered Realms. The Engine keeps the state of the Realm it replaced, as a root of
/// its collections, until the entry is left.
#[cfg(feature = "embed")]
#[must_use = "a Realm entered must be left"]
#[derive(Debug)]
pub struct RealmEntry {
    /// The heap whose stack holds the entry (`value::heap_id`).
    heap: u64,
    /// The stack's height with the entry on top.
    depth: usize,
}
