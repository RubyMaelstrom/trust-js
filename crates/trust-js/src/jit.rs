//! Native template JIT: the third execution tier.
//!
//! A compiled [`crate::bytecode::Chunk`] lowers to machine code one bytecode op at a time. Most
//! ops become a call into [`crate::bytecode::jit_exec`] — the single slow-path helper that runs
//! exactly one op against a raw operand-stack pointer — with the op index baked in as an
//! immediate. Control flow (jumps, conditional branches, returns, try/catch) is real machine
//! branches between per-op labels, so the interpreter's fetch/dispatch loop disappears entirely.
//! Hot ops gain inline fast paths over the templates in later passes.
//!
//! The operand stack is a pre-sized flat buffer (its maximum depth is computed statically from
//! the op stream), held in a callee-saved register; helpers return the new stack top, or null to
//! signal a throw, which routes through a shared unwind block that consults the try-handler
//! stack recorded by `PushHandler` templates.
//!
//! The mature backend emits ARM64 on desktop operating systems. A correctness-first x86-64
//! backend emits native control flow on Intel macOS, Linux, and Windows while its hot inline
//! templates are filled in. Other targets retain the bytecode VM.

#![cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use crate::bytecode::Chunk;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
use crate::bytecode::UpdKind;
use crate::execution_storage::SlotAccess;
use crate::interpreter::{Abrupt, Env, Interp};
use crate::value::{PackedValue, Value};

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "jit_names.rs"]
mod names;

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "jit_typed_array.rs"]
mod typed_array;

#[path = "jit_profiler.rs"]
mod profiler;

#[path = "jit_cache.rs"]
pub(crate) mod cache;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "jit_cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
#[path = "jit_coverage_tests.rs"]
mod coverage_tests;

#[cfg(test)]
#[path = "jit_assignment_tests.rs"]
mod assignment_tests;

#[cfg(test)]
#[path = "jit_update_elem_tests.rs"]
mod update_elem_tests;

#[cfg(test)]
#[path = "jit_call_reference_tests.rs"]
mod call_reference_tests;

#[cfg(test)]
#[path = "jit_call_closure_tests.rs"]
mod call_closure_tests;

#[cfg(test)]
#[path = "jit_osr_entry_tests.rs"]
mod osr_entry_tests;

#[cfg(test)]
#[path = "jit_continuation_tests.rs"]
mod continuation_tests;

#[cfg(test)]
#[path = "jit_inline_context_tests.rs"]
mod inline_context_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "bytecode_inline_prepared_tests.rs"]
mod inline_prepared_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "jit_inline_realm_selection_tests.rs"]
mod inline_realm_selection_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "call_global_env_pin_tests.rs"]
mod call_global_env_pin_tests;

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_math_intrinsic_tests.rs"]
mod math_intrinsic_tests;

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_int_typed_tests.rs"]
mod int_typed_tests;

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_constant_tests.rs"]
mod constant_tests;

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_name_dispatch_tests.rs"]
mod name_dispatch_tests;

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_loop_latch_tests.rs"]
mod loop_latch_tests;

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_word_routing_tests.rs"]
mod word_routing_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_this_tests.rs"]
mod this_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_string_constant_tests.rs"]
mod string_constant_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_traced_tests.rs"]
mod traced_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_rooting_tests.rs"]
pub(crate) mod rooting_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_poll_tests.rs"]
mod poll_tests;

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_regions.rs"]
mod regions;

#[cfg(test)]
pub(crate) fn test_local_array_fusion_hits() -> [u64; 5] {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        regions::local_array_hits()
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )))]
    {
        [0, 0, 0, 0, 0]
    }
}

#[cfg(test)]
pub(crate) fn test_tdz_array_pair_hits() -> [u64; 2] {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        regions::tdz_array_pair_hits()
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )))]
    {
        [0, 0]
    }
}

#[cfg(test)]
pub(crate) fn test_tdz_effect_paths() -> [u64; 5] {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        regions::tdz_effect_paths()
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )))]
    {
        [0, 0, 0, 0, 0]
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_operations.rs"]
mod operations;

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_exec_values.rs"]
mod exec_values;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
use exec_values::*;

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_storage.rs"]
mod storage;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
use storage::*;

/// Opt-in process counters used by the reproducible benchmark runner. Compilation,
/// native operations, conversions, and selected allocation/lookup helpers check this
/// flag; disabled paths do no counter updates, timing, or diagnostic traversal.
/// Relaxed atomics are sufficient: these are aggregate diagnostics, not engine state.
// 0 = not initialized, 1 = disabled, 2 = enabled. The diagnostics switch is process-scoped and
// sampled once; a relaxed byte load keeps disabled instrumentation at a predictable branch cost
// without taking the `OnceLock` fast path on every iterator/conversion helper.
static PERF_METRICS_ENABLED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static PERF_COMPILE_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_COMPILE_SUCCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_COMPILE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_GENERATED_CODE_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_LARGEST_CODE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_INLINE_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_INLINE_EMPTY_PLANS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_INLINE_PLAN_SITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_INLINE_SUCCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_INLINE_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_INLINE_SUPPRESSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_LEX_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_LEX_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_LEX_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_PARSE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_PARSE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_PARSE_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_BYTECODE_COMPILE_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_BYTECODE_COMPILE_SUCCESSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_BYTECODE_COMPILE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_SNAPSHOT_ENCODE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_SNAPSHOT_ENCODE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_SNAPSHOT_DECODE_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_SNAPSHOT_DECODE_SUCCESSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_SNAPSHOT_DECODE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_NATIVE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_NATIVE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_NATIVE_FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Stable per-operation native-call inventory (Phase 1 host/native observability).
///
/// Raw function-pointer addresses vary with ASLR and code layout, so they are never a reported
/// identity and never leave the process: they serve only as in-process lookup keys from a
/// registration site (which knows the static operation name) to the diagnostic label. The
/// reported identity is always the label string. Bare `NativeFn` builtins that share one
/// implementation address through identical-code folding keep the first-registered label, and
/// overloaded names such as `toString` aggregate across prototypes by design; both limits are
/// covered by tests below. Data-carrying callables and embedder namespaces carry their own
/// immutable labels (resolved inside `perf_native_end` via `NativeLabelSrc`). Phase 8's generated host ABI owns true
/// per-operation identities; this table is the migration inventory, not its replacement.
#[derive(Default)]
struct NativeOpStats {
    calls: u64,
    failures: u64,
    nanos: u64,
}

#[derive(Default)]
struct NativeOpTables {
    names: HashMap<usize, String>,
    /// Label-aggregated counters in deterministic byte order for stable JSON emission.
    stats: BTreeMap<String, NativeOpStats>,
}

/// Process-aggregate tables behind one mutex. Initialized on first registration; never touched
/// on the disabled per-call path (callers return before any lock, timestamp, or allocation).
/// Registration itself runs only on cold paths (realm/extension setup) with a bounded number of
/// distinct entries, so disabled processes pay no per-event cost.
static NATIVE_OP_TABLES: std::sync::OnceLock<std::sync::Mutex<NativeOpTables>> =
    std::sync::OnceLock::new();

/// Bound on distinct operation labels. Embedders may register arbitrary names, so diagnostics
/// must not become an unbounded-growth vector: past the cap, further labels aggregate into a
/// single `<overflow>` row while call/failure/time totals stay exact.
const NATIVE_OP_LABEL_CAP: usize = 1024;
/// Fallback label for fn addresses with no registration (never an address rendering).
const NATIVE_OP_UNKNOWN: &str = "<native>";
/// Aggregation row for labels past the distinct-label cap.
const NATIVE_OP_OVERFLOW: &str = "<overflow>";
static PERF_ERROR_CONSTRUCTIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_CAUGHT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_ESCAPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_OBJECT_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_MESSAGE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_STACK_CAPTURE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_STACK_CAPTURE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_STACK_FORMAT_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ERROR_STACK_FORMAT_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_GET_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_GET_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_GET_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_STEP_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_STEP_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_STEP_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_CLOSE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATOR_CLOSE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_TO_PRIMITIVE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_TO_PRIMITIVE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_TO_PRIMITIVE_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_TO_STRING_OBJECT_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_TO_STRING_OBJECT_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_TO_STRING_OBJECT_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATE_FAST_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATE_FAST_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATE_PROTOCOL_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATE_PROTOCOL_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static PERF_ITERATE_PROTOCOL_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[inline]
pub(crate) fn perf_metrics_enabled() -> bool {
    use std::sync::atomic::Ordering::{Relaxed, Release};
    match PERF_METRICS_ENABLED.load(Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let enabled = std::env::var_os("TRUST_JS_PERF_METRICS").is_some();
            let state = if enabled { 2 } else { 1 };
            let _ = PERF_METRICS_ENABLED.compare_exchange(0, state, Release, Relaxed);
            enabled
        }
    }
}

#[inline]
pub(crate) fn perf_stage_start() -> Option<std::time::Instant> {
    perf_metrics_enabled().then(std::time::Instant::now)
}

#[inline]
pub(crate) fn perf_lex_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_LEX_CALLS.fetch_add(1, Relaxed);
    PERF_LEX_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_LEX_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_parse_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_PARSE_CALLS.fetch_add(1, Relaxed);
    PERF_PARSE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_PARSE_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_bytecode_compile_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_BYTECODE_COMPILE_ATTEMPTS.fetch_add(1, Relaxed);
    PERF_BYTECODE_COMPILE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if success {
        PERF_BYTECODE_COMPILE_SUCCESSES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_snapshot_encode_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_SNAPSHOT_ENCODE_CALLS.fetch_add(1, Relaxed);
    PERF_SNAPSHOT_ENCODE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_snapshot_decode_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_SNAPSHOT_DECODE_ATTEMPTS.fetch_add(1, Relaxed);
    PERF_SNAPSHOT_DECODE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if success {
        PERF_SNAPSHOT_DECODE_SUCCESSES.fetch_add(1, Relaxed);
    }
}

/// Native-call funnel exit. Stays tiny so it inlines to a single enabled-gate branch at both
/// dispatch sites; all counting, timing, and per-operation work lives in
/// `perf_native_end_enabled`, which runs only when diagnostics are on. Disabled dispatch therefore
/// keeps the exact shape (and cost) of the previous aggregate-only funnel.
#[inline]
pub(crate) fn perf_native_end(
    started: Option<std::time::Instant>,
    success: bool,
    label_src: NativeLabelSrc<'_>,
) {
    let Some(started) = started else { return };
    perf_native_end_enabled(started, success, label_src);
}

fn perf_native_end_enabled(
    started: std::time::Instant,
    success: bool,
    label_src: NativeLabelSrc<'_>,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let nanos = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    PERF_NATIVE_CALLS.fetch_add(1, Relaxed);
    PERF_NATIVE_NANOS.fetch_add(nanos, Relaxed);
    if !success {
        PERF_NATIVE_FAILURES.fetch_add(1, Relaxed);
    }
    let mut tables = native_op_tables();
    record_native_op_src(&mut tables, label_src, nanos, success);
}

/// Attribute one call to its stable label. Takes `&mut NativeOpTables` directly (not the
/// mutex guard) so the names lookup and the stats update borrow disjoint struct fields.
fn record_native_op_src(
    tables: &mut NativeOpTables,
    src: NativeLabelSrc<'_>,
    nanos: u64,
    success: bool,
) {
    let label: &str = match src {
        NativeLabelSrc::Call(call) => match call {
            // Immutable registration identity, independent of the mutable `name` property.
            crate::value::Callable::NativeData(data) => &data.identity,
            crate::value::Callable::Native(f) => tables
                .names
                .get(&(*f as usize))
                .map(String::as_str)
                .unwrap_or(NATIVE_OP_UNKNOWN),
            _ => return,
        },
        NativeLabelSrc::Addr(addr) => tables
            .names
            .get(&addr)
            .map(String::as_str)
            .unwrap_or(NATIVE_OP_UNKNOWN),
    };
    record_native_op(&mut tables.stats, label, nanos, success);
}

/// Lock the process tables, recovering from a poisoned mutex rather than panicking diagnostics.
fn native_op_tables() -> std::sync::MutexGuard<'static, NativeOpTables> {
    NATIVE_OP_TABLES
        .get_or_init(|| std::sync::Mutex::new(NativeOpTables::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Record a registration label for a bare native fn address. First registration wins, which is
/// deterministic per binary because builtin installation order is fixed.
///
/// Runtime operations also create native functions (Promise resolving functions, iterator and
/// generator helpers), so this is not only a setup path. Each thread remembers the addresses it
/// has already registered and takes the process-wide lock only for a new one; later
/// registrations of the same address could never change its first-wins label.
pub(crate) fn perf_native_register(addr: usize, label: &str) {
    thread_local! {
        static REGISTERED: std::cell::RefCell<crate::fasthash::FastSet<usize>> =
            std::cell::RefCell::new(Default::default());
    }
    let first_on_thread = REGISTERED
        .try_with(|registered| registered.borrow_mut().insert(addr))
        .unwrap_or(true);
    if first_on_thread {
        register_native_name(&mut native_op_tables(), addr, label);
    }
}

fn register_native_name(tables: &mut NativeOpTables, addr: usize, label: &str) {
    tables
        .names
        .entry(addr)
        .or_insert_with(|| label.to_string());
}

/// Overwrite the label for a fn address (embedder namespaces qualify `op` as `ns.op`).
/// Only called from `feature = "embed"` registration paths (plus tests).
#[cfg(any(feature = "embed", test))]
// The sole caller lives in the optional `embed` module, so a default-feature test build sees the
// `test` cfg without that caller; the embed build and CI keep it genuinely live.
#[cfg_attr(
    not(feature = "embed"),
    expect(dead_code, reason = "caller is behind feature `embed`")
)]
pub(crate) fn perf_native_relabel(addr: usize, label: String) {
    relabel_native_name(&mut native_op_tables(), addr, label);
}

#[cfg(any(feature = "embed", test))]
fn relabel_native_name(tables: &mut NativeOpTables, addr: usize, label: String) {
    tables.names.insert(addr, label);
}

/// What to attribute one native call to. Passed by value (`Copy`, pointer-sized, no drop glue)
/// so disabled dispatch pays only register traffic: callers construct it unconditionally and
/// `perf_native_end` resolves it only after the enabled gate.
#[derive(Clone, Copy)]
pub(crate) enum NativeLabelSrc<'a> {
    /// Ordinary dispatch: the dispatched callable is still borrowed by the caller.
    Call(&'a crate::value::Callable),
    /// Native-entry IC funnel: only the bare fn address is available in scope.
    Addr(usize),
}

fn record_native_op(
    stats: &mut BTreeMap<String, NativeOpStats>,
    label: &str,
    nanos: u64,
    success: bool,
) {
    // New labels allocate exactly once (first sighting); steady-state hits borrow and update in
    // place, so enabled profiling adds no per-call allocation on hot operations.
    if stats.len() >= NATIVE_OP_LABEL_CAP && !stats.contains_key(label) {
        let entry = stats.entry(NATIVE_OP_OVERFLOW.to_string()).or_default();
        entry.calls += 1;
        entry.nanos += nanos;
        if !success {
            entry.failures += 1;
        }
        return;
    }
    if let Some(entry) = stats.get_mut(label) {
        entry.calls += 1;
        entry.nanos += nanos;
        if !success {
            entry.failures += 1;
        }
        return;
    }
    let entry = NativeOpStats {
        calls: 1,
        nanos,
        failures: u64::from(!success),
    };
    stats.insert(label.to_string(), entry);
}

fn json_escape_into(out: &mut String, text: &str) {
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let v = c as u32;
                out.push_str("\\u00");
                out.push(HEX[(v >> 4) as usize] as char);
                out.push(HEX[(v & 0xF) as usize] as char);
            }
            c => out.push(c),
        }
    }
}

fn render_native_ops(tables: &NativeOpTables) -> String {
    let mut out = String::from("[");
    for (index, (label, stats)) in tables.stats.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        let mut escaped = String::with_capacity(label.len());
        json_escape_into(&mut escaped, label);
        out.push_str(&format!(
            "{{\"operation\":\"{escaped}\",\"calls\":{},\"failures\":{},\"seconds\":{:.9}}}",
            stats.calls,
            stats.failures,
            stats.nanos as f64 / 1_000_000_000.0
        ));
    }
    out.push(']');
    out
}

fn native_op_json() -> String {
    render_native_ops(&native_op_tables())
}

#[inline]
pub(crate) fn perf_error_construction() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_ERROR_CONSTRUCTIONS.fetch_add(1, Relaxed);
    }
}

/// Count a thrown completion consumed by an actual `catch` clause. Finalizers and iterator
/// cleanup handlers deliberately do not call this: they preserve/transform completions but do not
/// consume a throw under ECMA-262's TryStatement algorithm.
#[inline]
pub(crate) fn perf_error_caught() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_ERROR_CAUGHT.fetch_add(1, Relaxed);
    }
}

/// Count a throw that reaches a public synchronous script boundary without a matching catch.
/// This is a completion-lifecycle metric, separate from error-object construction.
#[inline]
pub(crate) fn perf_error_escaped() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_ERROR_ESCAPED.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_error_message_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ERROR_MESSAGE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_error_object_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ERROR_OBJECT_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_error_stack_capture_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ERROR_STACK_CAPTURE_CALLS.fetch_add(1, Relaxed);
    PERF_ERROR_STACK_CAPTURE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_error_stack_format_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ERROR_STACK_FORMAT_CALLS.fetch_add(1, Relaxed);
    PERF_ERROR_STACK_FORMAT_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_iterator_get_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ITERATOR_GET_CALLS.fetch_add(1, Relaxed);
    PERF_ITERATOR_GET_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_ITERATOR_GET_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_iterator_step_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ITERATOR_STEP_CALLS.fetch_add(1, Relaxed);
    PERF_ITERATOR_STEP_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_ITERATOR_STEP_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_iterator_close_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ITERATOR_CLOSE_CALLS.fetch_add(1, Relaxed);
    PERF_ITERATOR_CLOSE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_to_primitive_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_TO_PRIMITIVE_CALLS.fetch_add(1, Relaxed);
    PERF_TO_PRIMITIVE_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_TO_PRIMITIVE_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_to_string_object_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_TO_STRING_OBJECT_CALLS.fetch_add(1, Relaxed);
    PERF_TO_STRING_OBJECT_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_TO_STRING_OBJECT_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_iterate_fast_end(started: Option<std::time::Instant>) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ITERATE_FAST_CALLS.fetch_add(1, Relaxed);
    PERF_ITERATE_FAST_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
}

#[inline]
pub(crate) fn perf_iterate_protocol_end(started: Option<std::time::Instant>, success: bool) {
    let Some(started) = started else { return };
    use std::sync::atomic::Ordering::Relaxed;
    PERF_ITERATE_PROTOCOL_CALLS.fetch_add(1, Relaxed);
    PERF_ITERATE_PROTOCOL_NANOS.fetch_add(
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Relaxed,
    );
    if !success {
        PERF_ITERATE_PROTOCOL_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_inline_attempt() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_INLINE_ATTEMPTS.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_inline_empty_plan() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_INLINE_EMPTY_PLANS.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_inline_plan_sites(sites: usize) {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_INLINE_PLAN_SITES.fetch_add(sites as u64, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_inline_success() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_INLINE_SUCCESSES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_inline_failure() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_INLINE_FAILURES.fetch_add(1, Relaxed);
    }
}

#[inline]
pub(crate) fn perf_inline_suppressed() {
    if perf_metrics_enabled() {
        use std::sync::atomic::Ordering::Relaxed;
        PERF_INLINE_SUPPRESSED.fetch_add(1, Relaxed);
    }
}

/// Machine-readable process summary printed by the CLI at normal exit. Kept as a single JSON line
/// so an external runner can separate it from other diagnostics without a serializer dependency.
pub(crate) fn performance_metrics_json(managed_memory: &str) -> Option<String> {
    if !perf_metrics_enabled() {
        return None;
    }
    use std::sync::atomic::Ordering::Relaxed;
    let attempts = PERF_COMPILE_ATTEMPTS.load(Relaxed);
    let successes = PERF_COMPILE_SUCCESSES.load(Relaxed);
    let nanos = PERF_COMPILE_NANOS.load(Relaxed);
    let generated = PERF_GENERATED_CODE_BYTES.load(Relaxed);
    let largest = PERF_LARGEST_CODE_BYTES.load(Relaxed);
    let inline_attempts = PERF_INLINE_ATTEMPTS.load(Relaxed);
    let inline_empty = PERF_INLINE_EMPTY_PLANS.load(Relaxed);
    let inline_sites = PERF_INLINE_PLAN_SITES.load(Relaxed);
    let inline_successes = PERF_INLINE_SUCCESSES.load(Relaxed);
    let inline_failures = PERF_INLINE_FAILURES.load(Relaxed);
    let inline_suppressed = PERF_INLINE_SUPPRESSED.load(Relaxed);
    let lex_calls = PERF_LEX_CALLS.load(Relaxed);
    let lex_nanos = PERF_LEX_NANOS.load(Relaxed);
    let lex_failures = PERF_LEX_FAILURES.load(Relaxed);
    let parse_calls = PERF_PARSE_CALLS.load(Relaxed);
    let parse_nanos = PERF_PARSE_NANOS.load(Relaxed);
    let parse_failures = PERF_PARSE_FAILURES.load(Relaxed);
    let bytecode_attempts = PERF_BYTECODE_COMPILE_ATTEMPTS.load(Relaxed);
    let bytecode_successes = PERF_BYTECODE_COMPILE_SUCCESSES.load(Relaxed);
    let bytecode_nanos = PERF_BYTECODE_COMPILE_NANOS.load(Relaxed);
    let snapshot_encode_calls = PERF_SNAPSHOT_ENCODE_CALLS.load(Relaxed);
    let snapshot_encode_nanos = PERF_SNAPSHOT_ENCODE_NANOS.load(Relaxed);
    let snapshot_decode_attempts = PERF_SNAPSHOT_DECODE_ATTEMPTS.load(Relaxed);
    let snapshot_decode_successes = PERF_SNAPSHOT_DECODE_SUCCESSES.load(Relaxed);
    let snapshot_decode_nanos = PERF_SNAPSHOT_DECODE_NANOS.load(Relaxed);
    let native_calls = PERF_NATIVE_CALLS.load(Relaxed);
    let native_nanos = PERF_NATIVE_NANOS.load(Relaxed);
    let native_failures = PERF_NATIVE_FAILURES.load(Relaxed);
    let native_by_op = native_op_json();
    let error_constructions = PERF_ERROR_CONSTRUCTIONS.load(Relaxed);
    let error_caught = PERF_ERROR_CAUGHT.load(Relaxed);
    let error_escaped = PERF_ERROR_ESCAPED.load(Relaxed);
    let error_object_nanos = PERF_ERROR_OBJECT_NANOS.load(Relaxed);
    let error_message_nanos = PERF_ERROR_MESSAGE_NANOS.load(Relaxed);
    let error_stack_capture_calls = PERF_ERROR_STACK_CAPTURE_CALLS.load(Relaxed);
    let error_stack_capture_nanos = PERF_ERROR_STACK_CAPTURE_NANOS.load(Relaxed);
    let error_stack_format_calls = PERF_ERROR_STACK_FORMAT_CALLS.load(Relaxed);
    let error_stack_format_nanos = PERF_ERROR_STACK_FORMAT_NANOS.load(Relaxed);
    let iterator_get_calls = PERF_ITERATOR_GET_CALLS.load(Relaxed);
    let iterator_get_nanos = PERF_ITERATOR_GET_NANOS.load(Relaxed);
    let iterator_get_failures = PERF_ITERATOR_GET_FAILURES.load(Relaxed);
    let iterator_step_calls = PERF_ITERATOR_STEP_CALLS.load(Relaxed);
    let iterator_step_nanos = PERF_ITERATOR_STEP_NANOS.load(Relaxed);
    let iterator_step_failures = PERF_ITERATOR_STEP_FAILURES.load(Relaxed);
    let iterator_close_calls = PERF_ITERATOR_CLOSE_CALLS.load(Relaxed);
    let iterator_close_nanos = PERF_ITERATOR_CLOSE_NANOS.load(Relaxed);
    let to_primitive_calls = PERF_TO_PRIMITIVE_CALLS.load(Relaxed);
    let to_primitive_nanos = PERF_TO_PRIMITIVE_NANOS.load(Relaxed);
    let to_primitive_failures = PERF_TO_PRIMITIVE_FAILURES.load(Relaxed);
    let to_string_object_calls = PERF_TO_STRING_OBJECT_CALLS.load(Relaxed);
    let to_string_object_nanos = PERF_TO_STRING_OBJECT_NANOS.load(Relaxed);
    let to_string_object_failures = PERF_TO_STRING_OBJECT_FAILURES.load(Relaxed);
    let iterate_fast_calls = PERF_ITERATE_FAST_CALLS.load(Relaxed);
    let iterate_fast_nanos = PERF_ITERATE_FAST_NANOS.load(Relaxed);
    let iterate_protocol_calls = PERF_ITERATE_PROTOCOL_CALLS.load(Relaxed);
    let iterate_protocol_nanos = PERF_ITERATE_PROTOCOL_NANOS.load(Relaxed);
    let iterate_protocol_failures = PERF_ITERATE_PROTOCOL_FAILURES.load(Relaxed);
    let gc = crate::value::gc_performance_metrics_json_fields();
    let (code_used, code_limit, code_denials) = executable_code_stats();
    let workload_metrics = crate::workload_metrics::json_fields();
    let cache_metadata = cache::shared_metadata_json();
    // Read an existing counter only when diagnostics are requested; no extra per-call work.
    // This epoch is process-wide, unlike the thread-owned native residency registry.
    let call_ic_epoch = crate::bytecode::CALL_IC_EPOCH.load(Relaxed);
    let shape_transitions = crate::value::active_shape_transition_count();
    let code_metrics = format!(
        "\"executable_code_live_bytes\":{code_used},\"executable_code_limit_bytes\":{code_limit},\"executable_code_budget_denials\":{code_denials},\"call_ic_process_epoch\":{call_ic_epoch},\"shape_transitions\":{shape_transitions},\"native_cache_thread_metadata\":{cache_metadata},{workload_metrics}"
    );
    Some(format!(
        "{{\"schema_version\":1,\"jit_compile_attempts\":{attempts},\"jit_compile_successes\":{successes},\"jit_compile_failures\":{},\"jit_compile_seconds\":{:.9},\"jit_generated_code_bytes\":{generated},\"jit_largest_code_bytes\":{largest},\"jit_inline_attempts\":{inline_attempts},\"jit_inline_empty_plans\":{inline_empty},\"jit_inline_plan_sites\":{inline_sites},\"jit_inline_successes\":{inline_successes},\"jit_inline_failures\":{inline_failures},\"jit_inline_suppressed\":{inline_suppressed},\"lex_calls\":{lex_calls},\"lex_seconds\":{:.9},\"lex_failures\":{lex_failures},\"parse_calls\":{parse_calls},\"parse_seconds\":{:.9},\"parse_failures\":{parse_failures},\"bytecode_compile_attempts\":{bytecode_attempts},\"bytecode_compile_successes\":{bytecode_successes},\"bytecode_compile_failures\":{},\"bytecode_compile_seconds\":{:.9},\"snapshot_encode_calls\":{snapshot_encode_calls},\"snapshot_encode_seconds\":{:.9},\"snapshot_decode_attempts\":{snapshot_decode_attempts},\"snapshot_decode_successes\":{snapshot_decode_successes},\"snapshot_decode_failures\":{},\"snapshot_decode_seconds\":{:.9},\"native_calls\":{native_calls},\"native_failures\":{native_failures},\"native_seconds\":{:.9},\"error_constructions\":{error_constructions},\"error_caught\":{error_caught},\"error_escaped\":{error_escaped},\"error_object_seconds\":{:.9},\"error_message_seconds\":{:.9},\"error_stack_capture_calls\":{error_stack_capture_calls},\"error_stack_capture_seconds\":{:.9},\"error_stack_format_calls\":{error_stack_format_calls},\"error_stack_format_seconds\":{:.9},\"iterator_get_calls\":{iterator_get_calls},\"iterator_get_failures\":{iterator_get_failures},\"iterator_get_seconds\":{:.9},\"iterator_step_calls\":{iterator_step_calls},\"iterator_step_failures\":{iterator_step_failures},\"iterator_step_seconds\":{:.9},\"iterator_close_calls\":{iterator_close_calls},\"iterator_close_seconds\":{:.9},\"to_primitive_object_calls\":{to_primitive_calls},\"to_primitive_object_failures\":{to_primitive_failures},\"to_primitive_object_seconds\":{:.9},\"to_string_object_calls\":{to_string_object_calls},\"to_string_object_failures\":{to_string_object_failures},\"to_string_object_seconds\":{:.9},\"iterate_fast_calls\":{iterate_fast_calls},\"iterate_fast_seconds\":{:.9},\"iterate_protocol_calls\":{iterate_protocol_calls},\"iterate_protocol_failures\":{iterate_protocol_failures},\"iterate_protocol_seconds\":{:.9},{gc},{code_metrics},\"managed_memory\":{managed_memory},\"native_by_operation\":{native_by_op}}}",
        attempts.saturating_sub(successes),
        nanos as f64 / 1_000_000_000.0,
        lex_nanos as f64 / 1_000_000_000.0,
        parse_nanos as f64 / 1_000_000_000.0,
        bytecode_attempts.saturating_sub(bytecode_successes),
        bytecode_nanos as f64 / 1_000_000_000.0,
        snapshot_encode_nanos as f64 / 1_000_000_000.0,
        snapshot_decode_attempts.saturating_sub(snapshot_decode_successes),
        snapshot_decode_nanos as f64 / 1_000_000_000.0,
        native_nanos as f64 / 1_000_000_000.0,
        error_object_nanos as f64 / 1_000_000_000.0,
        error_message_nanos as f64 / 1_000_000_000.0,
        error_stack_capture_nanos as f64 / 1_000_000_000.0,
        error_stack_format_nanos as f64 / 1_000_000_000.0,
        iterator_get_nanos as f64 / 1_000_000_000.0,
        iterator_step_nanos as f64 / 1_000_000_000.0,
        iterator_close_nanos as f64 / 1_000_000_000.0,
        to_primitive_nanos as f64 / 1_000_000_000.0,
        to_string_object_nanos as f64 / 1_000_000_000.0,
        iterate_fast_nanos as f64 / 1_000_000_000.0,
        iterate_protocol_nanos as f64 / 1_000_000_000.0,
    ))
}

// ---------------------------------------------------------------------------------------------
// Executable memory (platform W^X policy)
// ---------------------------------------------------------------------------------------------

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    target_os = "macos"
))]
mod sys {
    extern "C" {
        pub fn mmap(
            addr: *mut u8,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut u8;
        fn munmap(addr: *mut u8, len: usize) -> i32;
        fn pthread_jit_write_protect_np(enabled: i32);
        fn sys_icache_invalidate(start: *mut u8, len: usize);
    }
    const PROT_RWX: i32 = 0x1 | 0x2 | 0x4;
    const MAP_PRIVATE_ANON_JIT: i32 = 0x0002 | 0x1000 | 0x0800;

    pub unsafe fn alloc_exec(src: *const u8, len: usize) -> *mut u8 {
        let mem = mmap(
            std::ptr::null_mut(),
            len,
            PROT_RWX,
            MAP_PRIVATE_ANON_JIT,
            -1,
            0,
        );
        if mem as isize == -1 {
            return std::ptr::null_mut();
        }
        pthread_jit_write_protect_np(0);
        std::ptr::copy_nonoverlapping(src, mem, len);
        pthread_jit_write_protect_np(1);
        sys_icache_invalidate(mem, len);
        mem
    }

    pub unsafe fn free_exec(mem: *mut u8, len: usize) {
        munmap(mem, len);
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod sys {
    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;

    pub unsafe fn alloc_exec(src: *const u8, len: usize) -> *mut u8 {
        let mem = mmap(
            std::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE_ANON,
            -1,
            0,
        );
        if mem as isize == -1 {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(src, mem, len);
        if mprotect(mem, len, PROT_READ | PROT_EXEC) != 0 {
            munmap(mem, len);
            return std::ptr::null_mut();
        }
        mem
    }

    pub unsafe fn free_exec(mem: *mut u8, len: usize) {
        munmap(mem, len);
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod sys {
    use core::arch::asm;

    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;

    unsafe fn flush_icache(start: *mut u8, len: usize) {
        let ctr: usize;
        asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nostack, preserves_flags));
        let dline = 4usize << ((ctr >> 16) & 0xf);
        let iline = 4usize << (ctr & 0xf);
        let end = start as usize + len;
        let mut p = (start as usize) & !(dline - 1);
        while p < end {
            asm!("dc cvau, {p}", p = in(reg) p, options(nostack, preserves_flags));
            p += dline;
        }
        asm!("dsb ish", options(nostack, preserves_flags));
        p = (start as usize) & !(iline - 1);
        while p < end {
            asm!("ic ivau, {p}", p = in(reg) p, options(nostack, preserves_flags));
            p += iline;
        }
        asm!("dsb ish", "isb", options(nostack, preserves_flags));
    }

    pub unsafe fn alloc_exec(src: *const u8, len: usize) -> *mut u8 {
        let mem = mmap(
            std::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE_ANON,
            -1,
            0,
        );
        if mem as isize == -1 {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(src, mem, len);
        flush_icache(mem, len);
        if mprotect(mem, len, PROT_READ | PROT_EXEC) != 0 {
            munmap(mem, len);
            return std::ptr::null_mut();
        }
        mem
    }

    pub unsafe fn free_exec(mem: *mut u8, len: usize) {
        munmap(mem, len);
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    target_os = "windows"
))]
mod sys {
    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, len: usize, kind: u32, protect: u32) -> *mut u8;
        fn VirtualProtect(addr: *mut u8, len: usize, protect: u32, old: *mut u32) -> i32;
        fn VirtualFree(addr: *mut u8, len: usize, kind: u32) -> i32;
        fn FlushInstructionCache(process: *mut u8, addr: *const u8, len: usize) -> i32;
        fn GetCurrentProcess() -> *mut u8;
    }
    const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 0x04;
    const PAGE_EXECUTE_READ: u32 = 0x20;

    pub unsafe fn alloc_exec(src: *const u8, len: usize) -> *mut u8 {
        let mem = VirtualAlloc(
            std::ptr::null_mut(),
            len,
            MEM_COMMIT_RESERVE,
            PAGE_READWRITE,
        );
        if mem.is_null() {
            return mem;
        }
        std::ptr::copy_nonoverlapping(src, mem, len);
        let mut old = 0;
        if VirtualProtect(mem, len, PAGE_EXECUTE_READ, &mut old) == 0
            || FlushInstructionCache(GetCurrentProcess(), mem, len) == 0
        {
            VirtualFree(mem, 0, MEM_RELEASE);
            return std::ptr::null_mut();
        }
        mem
    }

    pub unsafe fn free_exec(mem: *mut u8, _len: usize) {
        VirtualFree(mem, 0, MEM_RELEASE);
    }
}

/// The process-wide executable-code budget shared by the bytecode JIT and native RegExp tiers.
///
/// This caps requested bytes in live executable mappings, not a cumulative total
/// of code ever emitted (OS page rounding is separate). Browser bootstrap can
/// exceed 16 MiB before application hot functions even compile. Keep bounded
/// headroom; pressure first reclaims inactive, unleased native code. A failed
/// reservation uses the checked fallback, and dropping each owning mapping
/// returns capacity to later hot code.
pub(crate) const EXECUTABLE_CODE_BUDGET: usize = 128 << 20;

fn configured_executable_code_budget(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|mib| (1..=1024).contains(mib))
        .and_then(|mib| mib.checked_mul(1 << 20))
        .unwrap_or(EXECUTABLE_CODE_BUDGET)
}

struct ExecutableCodeBudget {
    limit: usize,
    remaining: std::sync::atomic::AtomicUsize,
    denials: std::sync::atomic::AtomicU64,
}

impl ExecutableCodeBudget {
    const fn new(limit: usize) -> Self {
        Self {
            limit,
            remaining: std::sync::atomic::AtomicUsize::new(limit),
            denials: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

fn executable_code_budget() -> &'static ExecutableCodeBudget {
    static BUDGET: std::sync::OnceLock<ExecutableCodeBudget> = std::sync::OnceLock::new();
    BUDGET.get_or_init(|| {
        ExecutableCodeBudget::new(configured_executable_code_budget(
            std::env::var("TRUST_JS_JIT_CODE_BUDGET_MB").ok().as_deref(),
        ))
    })
}

pub(crate) fn executable_code_can_fit(bytes: usize) -> bool {
    executable_code_budget()
        .remaining
        .load(std::sync::atomic::Ordering::Relaxed)
        >= bytes
}

/// Reclaim only inactive, unleased code owned by this thread. Another thread's
/// live mappings or a fully active working set may still require VM fallback.
pub(crate) fn ensure_executable_capacity(bytes: usize) -> bool {
    let budget = executable_code_budget();
    if bytes > budget.limit {
        return false;
    }
    let remaining = budget.remaining.load(std::sync::atomic::Ordering::Relaxed);
    if remaining < bytes {
        cache::reclaim_for_capacity(bytes, remaining);
    }
    executable_code_can_fit(bytes)
}

thread_local! {
    // Scoped to compilation, never JS execution. A nested compiler invocation
    // restores its caller's observation, and other threads have separate slots.
    static COMPILATION_DENIED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

struct CompilationPressureScope(usize);

impl CompilationPressureScope {
    fn enter() -> Self {
        Self(COMPILATION_DENIED_BYTES.with(|slot| slot.replace(0)))
    }

    fn denied_bytes(&self) -> usize {
        COMPILATION_DENIED_BYTES.with(std::cell::Cell::get)
    }
}

impl Drop for CompilationPressureScope {
    fn drop(&mut self) {
        COMPILATION_DENIED_BYTES.with(|slot| slot.set(self.0));
    }
}

/// Live requested executable bytes, hard limit, and rejected reservations. This
/// intentionally excludes page rounding and ordinary compilation metadata.
pub(crate) fn executable_code_stats() -> (usize, usize, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let budget = executable_code_budget();
    (
        budget.limit - budget.remaining.load(Relaxed),
        budget.limit,
        budget.denials.load(Relaxed),
    )
}

pub(crate) struct ExecutableCodeReservation<'a> {
    bytes: usize,
    budget: &'a ExecutableCodeBudget,
}

impl<'a> ExecutableCodeReservation<'a> {
    fn try_new(budget: &'a ExecutableCodeBudget, bytes: usize) -> Option<Self> {
        if bytes == 0 {
            return None;
        }
        let mut remaining = budget.remaining.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            if remaining < bytes {
                COMPILATION_DENIED_BYTES.with(|slot| slot.set(bytes));
                budget
                    .denials
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return None;
            }
            match budget.remaining.compare_exchange_weak(
                remaining,
                remaining - bytes,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => return Some(Self { bytes, budget }),
                Err(next) => remaining = next,
            }
        }
    }
}

impl Drop for ExecutableCodeReservation<'_> {
    fn drop(&mut self) {
        self.budget
            .remaining
            .fetch_add(self.bytes, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(test)]
mod executable_code_budget_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn executable_code_budget_is_bounded_and_validated() {
        for invalid in [
            None,
            Some("0"),
            Some("-1"),
            Some("1025"),
            Some("1.5"),
            Some("bad"),
        ] {
            assert_eq!(
                configured_executable_code_budget(invalid),
                EXECUTABLE_CODE_BUDGET
            );
        }
        assert_eq!(configured_executable_code_budget(Some("1")), 1 << 20);
        assert_eq!(configured_executable_code_budget(Some("128")), 128 << 20);
        assert_eq!(configured_executable_code_budget(Some("1024")), 1024 << 20);
    }

    #[test]
    fn executable_code_reservations_release_capacity_and_count_pressure() {
        let budget = ExecutableCodeBudget::new(100);
        assert!(ExecutableCodeReservation::try_new(&budget, 0).is_none());
        let first = ExecutableCodeReservation::try_new(&budget, 60).unwrap();
        assert!(ExecutableCodeReservation::try_new(&budget, 41).is_none());
        let second = ExecutableCodeReservation::try_new(&budget, 40).unwrap();
        assert_eq!(budget.remaining.load(Relaxed), 0);
        assert_eq!(budget.denials.load(Relaxed), 1);
        drop(first);
        assert_eq!(budget.remaining.load(Relaxed), 60);
        let replacement = ExecutableCodeReservation::try_new(&budget, 60).unwrap();
        drop((second, replacement));
        assert_eq!(budget.remaining.load(Relaxed), 100);
    }

    #[test]
    fn executable_code_budget_is_shared_safely_between_threads() {
        let budget = ExecutableCodeBudget::new(128);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        let reservation = ExecutableCodeReservation::try_new(&budget, 64);
                        assert!(budget.remaining.load(Relaxed) <= 128);
                        std::thread::yield_now();
                        drop(reservation);
                    }
                });
            }
        });
        assert_eq!(budget.remaining.load(Relaxed), 128);
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    #[test]
    fn jit_compilation_recovers_after_executable_budget_pressure() {
        // The budget is process-wide. Isolate this forced-pressure test rather
        // than stealing executable capacity from concurrently running tests.
        const CHILD: &str = "TRUST_JS_TEST_CODE_PRESSURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "jit::executable_code_budget_tests::jit_compilation_recovers_after_executable_budget_pressure",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env("TRUST_JS_JIT_CODE_BUDGET_MB", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        use crate::value::{Callable, Value};
        let mut engine = crate::Engine::new();
        engine.set_tier(crate::bytecode::Tier::Jit);
        engine.set_tier_threshold(0);
        engine
            .eval("function hot(value) { return value + 1; }", false)
            .unwrap();
        let global = Value::Obj(engine.interp.global.clone());
        let function = engine
            .interp
            .get_member_key(&global, "hot")
            .unwrap_or_else(|_| panic!("hot function binding exists"));
        let budget = executable_code_budget();
        let held =
            ExecutableCodeReservation::try_new(budget, budget.remaining.load(Relaxed)).unwrap();
        assert!(
            matches!(engine.interp.call(function.clone(), Value::Undefined, &[Value::Num(41.0)]),
            Ok(Value::Num(value)) if value == 42.0)
        );
        let Value::Obj(object) = &function else {
            panic!("function object")
        };
        let chunk = match &object.borrow().call {
            Callable::User(user) => user
                .func
                .runtime()
                .code
                .get()
                .and_then(Option::as_ref)
                .unwrap()
                .clone(),
            _ => panic!("ordinary user function"),
        };
        assert!(
            chunk.jit.get().is_none(),
            "pressure must not cache a permanent JIT failure"
        );
        assert!(chunk.jit_budget_wait_bytes.get() > 0);
        let denials = budget.denials.load(Relaxed);
        for _ in 0..32 {
            assert!(
                matches!(engine.interp.call(function.clone(), Value::Undefined, &[Value::Num(41.0)]),
                Ok(Value::Num(value)) if value == 42.0)
            );
        }
        assert_eq!(
            budget.denials.load(Relaxed),
            denials,
            "no repeated emission while capacity is unavailable"
        );
        // OSR is equally recoverable: a denied compilation must preserve the settled
        // Script frame and finish in bytecode, without replaying its side effects.
        engine.set_tier_threshold(32);
        const SCRIPT: &str = "var pressureCount=0;for(var pressureIndex=0;pressureIndex<1000;pressureIndex++){pressureCount++;}pressureCount;";
        TEST_OSR_ENTRIES.with(|count| count.set(0));
        crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.set(0));
        assert!(matches!(engine.eval(SCRIPT,false).unwrap(),
            crate::Completion::Value(value) if value=="1000"));
        assert_eq!(
            TEST_OSR_ENTRIES.with(|count| count.get()),
            0,
            "a full executable budget must leave the exact Script in the VM"
        );
        assert!(
            crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get()) > 0,
            "cold Script must reach the VM fragment despite native-code pressure"
        );
        drop(held);
        assert!(
            matches!(engine.interp.call(function, Value::Undefined, &[Value::Num(41.0)]),
            Ok(Value::Num(value)) if value == 42.0)
        );
        assert!(
            chunk.jit.get().is_some_and(|code| code.is_some()),
            "released capacity permits a real JIT entry"
        );
        assert_eq!(chunk.jit_budget_wait_bytes.get(), 0);
        TEST_OSR_ENTRIES.with(|count| count.set(0));
        crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.set(0));
        assert!(matches!(engine.eval(SCRIPT,false).unwrap(),
            crate::Completion::Value(value) if value=="1000"));
        assert!(
            TEST_OSR_ENTRIES.with(|count| count.get()) > 0,
            "released capacity permits real same-activation native execution"
        );
        assert!(
            crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get()) > 0,
            "recovered Script must retain its real fragment handoff"
        );
    }
}

/// A W^X-protected executable allocation shared by the regular JIT and native RegExp tiers. The
/// allocation owns its live-code reservation, so every mapping has one accounting and reclamation
/// path regardless of which execution engine requested it.
pub(crate) struct ExecutableBuffer {
    mem: *mut u8,
    len: usize,
    _reservation: ExecutableCodeReservation<'static>,
}

impl ExecutableBuffer {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }
        ensure_executable_capacity(bytes.len());
        let reservation =
            ExecutableCodeReservation::try_new(executable_code_budget(), bytes.len())?;
        #[cfg(any(
            all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ),
            all(
                target_arch = "x86_64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            )
        ))]
        {
            let mem = unsafe { sys::alloc_exec(bytes.as_ptr(), bytes.len()) };
            if mem.is_null() {
                return None;
            }
            Some(Self {
                mem,
                len: bytes.len(),
                _reservation: reservation,
            })
        }
        #[cfg(not(any(
            all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ),
            all(
                target_arch = "x86_64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            )
        )))]
        {
            let _ = (bytes, reservation);
            None
        }
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.mem
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        #[cfg(any(
            all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ),
            all(
                target_arch = "x86_64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            )
        ))]
        unsafe {
            sys::free_exec(self.mem, self.len);
        }
    }
}

/// The native frame contract, independent of the body's language suspension eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeEntryKind {
    FreshFrame,
    BorrowedFrame,
}

/// A finished JIT compilation and validated bytecode-to-machine landing maps.
#[repr(C)]
pub struct JitCode {
    mem: *mut u8,
    len: usize,
    /// Code byte offset of each bytecode pc (catch targets and branch targets).
    pc_offsets: Vec<u32>,
    /// Statically computed maximum operand-stack depth.
    pub max_stack: usize,
    /// Whether any template reads `JitCtx::global_body` (free-name caches): frame setup skips
    /// the realm-global borrow otherwise.
    pub needs_global: bool,
    /// Checked bytecode resumption depths; empty for ordinary-call code. Suspended state
    /// retains PCs, never executable addresses that could outlive a recompilation.
    resume_depths: Vec<Option<usize>>,
    /// Calling convention is independent of language Await/Yield eligibility. OSR code and
    /// coroutine slices borrow canonical VM storage; ordinary CallIc entries create a frame.
    entry_kind: NativeEntryKind,
    /// Only taken backward-edge headers admit a first VM-to-native transfer, at exactly this
    /// depth. Later completion resumptions use resume_depths' separate upper-bound contract.
    osr_entry_depths: Vec<Option<usize>>,
    /// Owns the W^X mapping and its shared live executable-code reservation. The duplicate `mem`
    /// and `len` fields above are retained because generated direct-call templates read the
    /// stable prefix of this struct by offset.
    #[allow(dead_code)]
    executable: ExecutableBuffer,
    /// Stable native-entry counter and CLOCK reference bit. The code embeds its
    /// address; keep the allocation alive until after the mapping is unmapped.
    residency: Box<cache::CodeResidency>,
}

impl Drop for JitCode {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.residency.active.get(),
            0,
            "active native mapping dropped"
        );
        // Unload profiler symbols before the executable mapping is released.
        profiler::unregister(self.mem);
    }
}

impl JitCode {
    #[cfg(test)]
    pub(crate) fn test_residency_marks(&self, reset_reference: bool) -> (u64, u8) {
        if reset_reference {
            self.residency.referenced.set(0);
        }
        (self.residency.active.get(), self.residency.referenced.get())
    }

    /// Heap-requested metadata only. The executable mapping is intentionally excluded and remains
    /// visible through the independent generated-code byte metric.
    pub(crate) fn retained_heap_metadata_bytes(&self) -> usize {
        std::mem::size_of::<JitCode>()
            .saturating_add(std::mem::size_of::<cache::CodeResidency>())
            .saturating_add(
                self.pc_offsets
                    .capacity()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(self.resume_depths.capacity() * std::mem::size_of::<Option<usize>>())
            .saturating_add(self.osr_entry_depths.capacity() * std::mem::size_of::<Option<usize>>())
    }

    /// The machine-code entry address (for CallIc fills — the direct-call sequence branches
    /// to it through the swapped ctx).
    /// Size of the generated machine code in bytes.
    pub(crate) fn code_bytes(&self) -> usize {
        self.len
    }

    pub(crate) fn mem_ptr(&self) -> *const u8 {
        assert_eq!(
            self.entry_kind,
            NativeEntryKind::FreshFrame,
            "borrowed-frame code cannot enter an ordinary CallIc"
        );
        self.mem
    }
    /// The pc→code-offset table's data pointer (same purpose).
    pub(crate) fn pc_offsets_ptr(&self) -> *const u32 {
        self.pc_offsets.as_ptr()
    }

    pub(crate) fn osr_entry_depth(&self, pc: usize) -> Option<usize> {
        self.osr_entry_depths.get(pc).copied().flatten()
    }
}

// ---------------------------------------------------------------------------------------------
// The runtime context shared between JIT code and its Rust helpers
// ---------------------------------------------------------------------------------------------

/// Passed to the JIT entry in x0. The leading fields are read from assembly by fixed offset —
/// keep their order in sync with the prologue/epilogue emitters below. Emitters access fields
/// after that prefix through `offset_of!`, never assumed Rust enum or struct sizes.
#[repr(C)]
pub struct JitCtx {
    /// [0] Helper function table (see `HELPER_*` indices).
    pub helpers: *const usize,
    /// [8] Operand-stack base; the JIT keeps the live top in a register and stores it back here
    /// on every exit path.
    pub stack_base: *mut PackedValue,
    /// [16] Final stack top, written by the epilogues (for leftover-value cleanup on throw).
    pub final_sp: *mut PackedValue,
    /// [24] Local slots base (the inline LoadLocal/StoreLocal templates index off this).
    pub slots: *mut PackedValue,
    /// [32] Points at `Interp::inline_ic_safe` (a `Cell<bool>` byte): raw prototype-walk
    /// templates read it live and fall to the helper after a Proxy is created.
    pub inline_ic_safe: *const u8,
    /// [40] `Env::as_ptr` of the activation env — what the inline LoadName template compares
    /// against the per-site name cache (see `bytecode::NameIc`).
    pub env_raw: *const u8,
    /// [48] The activation's `this` itself, as a packed word that owns nothing. Generated code
    /// chooses its LoadThis, receiver, direct-call and finish-stub forms for it; Rust reads and
    /// writes it through the `this` accessors below.
    pub(crate) this_word: PackedValue,
    /// [56] The current realm's global `Object` (inside its cell's RefCell): the LoadName
    /// templates' global-mode path validates the cached shape/slot against it.
    pub global_body: *const u8,
    /// [64] `Env::as_ptr` of the active realm's global scope: the call template's inline probe
    /// compares it against the CallIc's fill-time `global_env` (the same-realm proof).
    pub genv: usize,
    // ---- Rust-only fields ----
    pub interp: *mut Interp,
    pub chunk: *const Chunk,
    pub n_slots: usize,
    /// Active catch/finally/iterator regions, shared with bytecode completion routing.
    pub(crate) handlers: Vec<crate::bytecode::Handler>,
    /// The handler-stack watermark of THIS activation: `jit_unwind` propagates out (instead of
    /// popping) once `handlers.len()` reaches it. Always 0 for a `run`/`run_moved` activation
    /// (each owns a fresh Vec); the direct-call sequence shares the caller's ctx — and its
    /// handlers Vec — so it swaps this to the live length for the callee's duration.
    pub handler_floor: usize,
    pub code_base: *const u8,
    pub pc_offsets: *const u32,
    pub error: Option<Abrupt>,
    /// Owned native return slot. Vacant is the complete PACK_UNDEFINED word, never a
    /// zero tag byte; direct calls move it back to the packed operand stack unchanged.
    pub ret: PackedValue,
    /// Parent of `env_raw` as an `Env::as_ptr`, or null. Depth-1 free-name caches use this to
    /// validate a fresh activation without baking that per-call allocation's identity.
    pub env_parent_raw: *const u8,
    /// Hoisted process-wide debug switch. Keeping it in the activation avoids a `OnceLock`
    /// lookup in every JIT helper when operation statistics are disabled.
    pub opstat_enabled: bool,
    pub callstat_enabled: bool,
    pub propstat_enabled: bool,
    pub inline_recompile_at: u32,
    /// Address of this Agent heap's live-object counter. Compiled chunks may cross thread
    /// boundaries with generator/async interpreter handoff, so this heap address must be captured
    /// when an activation starts rather than embedded while machine code is compiled. The
    /// heap's due word follows the count (`value::LiveCount`): polls load both.
    pub live_objects: *const i64,
    /// The active heap's fixed shape-directory array. Creation ICs index this directly instead
    /// of following an owning heap pointer in each receiver. Layouts remain pinned by the heap.
    pub shape_layouts: *const u8,
    pub(crate) activation: Option<Box<crate::bytecode::NativeActivation>>,
    /// Native slices borrow VmCoro's authoritative heap state only while running. Ordinary
    /// entries leave this null; chunk identity prevents a nested direct call from aliasing it.
    pub(crate) resume_activation: *mut crate::bytecode::NativeContinuation,
    pub(crate) resume_pc: usize,
    pub(crate) resume_step: Option<crate::bytecode::VmStep>,
    /// Canonical Reference owners for this frame; null until a fresh activation needs them.
    pub(crate) references_raw: *mut crate::eval::PreparedReferenceSlot,
    /// Validated accessor holder (`Gc::as_ptr` as usize) and slot, set by the ARM64
    /// property template's accessor landing immediately before it calls `jit_get_prop`.
    /// The helper consumes (clears) both before any JavaScript can run, so no validated
    /// state can outlive the synchronous helper call that observed it. 0 = none.
    pub accessor_holder: usize,
    pub accessor_slot: u32,
}

/// `this` as the moved JIT entries (`run_moved` and its forms) take it and a [`JitCtx`] holds
/// it: the packed word.
pub(crate) type JitThis = PackedValue;

/// An activation's `this` lent to Rust code that takes a `&Value` (see [`JitCtx::this_view`]):
/// the view holds the widened word.
pub(crate) struct JitThisView {
    this: Value,
}

impl std::ops::Deref for JitThisView {
    type Target = Value;

    #[inline]
    fn deref(&self) -> &Value {
        &self.this
    }
}

// The activation's `this`. Rust reads and writes it only here.
impl JitCtx {
    /// The activation's `this` as a wide Value: the word widened.
    #[inline]
    pub(crate) fn this_value(&self) -> Value {
        self.this_word.unpack()
    }

    /// Lend the activation's `this` as a wide Value without borrowing the context, so the caller
    /// can lend the context's other fields beside it: the view holds the widened word.
    #[inline]
    pub(crate) fn this_view(&self) -> JitThisView {
        JitThisView {
            this: self.this_word.unpack(),
        }
    }

    /// Install the activation's `this`; the previous word owns nothing, so nothing is released.
    #[inline]
    pub(crate) fn set_this(&mut self, this: JitThis) {
        self.this_word = this;
    }

    /// Reset the activation's `this` to Undefined, a pooled record's invariant: one store of the
    /// complete Undefined word.
    #[inline]
    pub(crate) fn clear_this(&mut self) {
        self.this_word = PackedValue::pack(Value::Undefined);
    }
}

impl JitCtx {
    #[inline]
    pub(crate) fn take_ret(&mut self) -> PackedValue {
        std::mem::replace(&mut self.ret, PackedValue::pack(Value::Undefined))
    }

    /// Clone one local without changing the representation of the rest of the frame.
    /// A runtime miss for one operation must not copy every local in a large function.
    pub(crate) unsafe fn clone_slot(&self, slot: usize) -> Value {
        debug_assert!(slot < self.n_slots);
        unsafe { (*self.slots.add(slot)).unpack() }
    }
}

/// The values a running context holds in its own fields, traced precisely. Whoever
/// makes the context a root scans its slot and operand words: an active [`JitFrame`] record its
/// whole word buffer, a Rust entry the ranges it registers (`gc_trace::JitFrameRoots`). The raw
/// identities it caches are kept alive by their owners, as classified below.
impl crate::gc_trace::GcTrace for JitCtx {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let JitCtx {
            helpers: _,
            stack_base: _, // the operand words: scanned by the registration
            final_sp: _,
            slots: _, // the slot words: scanned by the registration
            inline_ic_safe: _,
            // The running environment, owned by the entry's frame (`run`'s environment, a
            // moved entry's activation or callee closure), by the callee closure a direct call
            // keeps on the caller's operand stack and in its FnFrame, by `activation`, or by
            // the VM frame a borrowed entry runs (registered by `drive_vm`).
            env_raw: _,
            this_word,
            global_body: _, // the realm's global object and scope
            genv: _,
            interp: _,
            chunk: _, // code, not cells
            n_slots: _,
            handlers: _, // pcs and depths
            handler_floor: _,
            code_base: _,
            pc_offsets: _,
            error,
            ret,
            env_parent_raw: _, // reached through `env_raw`
            opstat_enabled: _,
            callstat_enabled: _,
            propstat_enabled: _,
            inline_recompile_at: _,
            live_objects: _,
            shape_layouts: _, // Rust-owned layouts, not managed cell edges
            activation,
            // `drive_vm`'s borrow of its own frame state, which it registers (VM_FRAMES) for
            // its whole call: the references, environments and stack this points at.
            resume_activation: _,
            resume_pc: _,
            resume_step,
            references_raw: _,  // into `activation` or the borrowed continuation
            accessor_holder: _, // consumed before any JavaScript runs
            accessor_slot: _,
        } = self;
        this_word.trace_gc(sink);
        ret.trace_gc(sink);
        error.trace_gc(sink);
        activation.trace_gc(sink);
        resume_step.trace_gc(sink);
    }
}

thread_local! {
    /// Attribution markers of the chunk most recently finished by [`asm::Asm::finish_with_offsets`].
    static LAST_MARKS: std::cell::RefCell<Vec<(u32, &'static str)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether `TRUST_JS_JIT_MAP` profiler output is requested (attribution markers are emitted).
pub(crate) fn jit_map_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TRUST_JS_JIT_MAP").is_some())
}

#[inline]
/// Whether generated code may use FJCVTZS (ARMv8.3 FEAT_JSCVT) for ECMAScript ToInt32
/// (ECMA-262 §7.1.6). `TRUST_JS_JIT_NO_JSCVT` forces the portable guarded sequence.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) fn jscvt_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_aarch64_feature_detected!("jsconv")
            && std::env::var_os("TRUST_JS_JIT_NO_JSCVT").is_none()
    })
}

fn jit_env_parent_raw(env: &Env) -> *const u8 {
    env.borrow()
        .parent
        .as_ref()
        .map_or(std::ptr::null(), |parent| Env::as_ptr(parent) as *const u8)
}

/// Byte offset of the live-length word in `JitCtx::handlers`.
///
/// `Vec` does not expose a stable field layout, so the ARM64 direct-call emitter probes the
/// monomorphized representation instead of assuming one. The three distinct values make the
/// length word unambiguous for every supported Rust layout.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn jit_handlers_len_offset() -> Option<usize> {
    use std::mem::offset_of;

    let mut handlers: Vec<crate::bytecode::Handler> = Vec::with_capacity(5);
    handlers.extend((1..=3).map(|pc| crate::bytecode::Handler {
        target: crate::bytecode::HandlerTarget::Catch { throw_pc: pc },
        stack_depth: pc,
    }));
    let words: [usize; 3] = unsafe { std::mem::transmute_copy(&handlers) };
    words
        .iter()
        .position(|word| *word == handlers.len())
        .map(|word| offset_of!(JitCtx, handlers) + word * size_of::<usize>())
}

/// The helper function table the emitted code indexes (see `JitCtx::helpers`); built once per
/// `Interp` (`Interp::jit_helpers`) so calls don't re-materialize it.
pub(crate) fn helper_table() -> [usize; N_HELPERS] {
    [
        crate::bytecode::jit_exec as *const () as usize,
        crate::bytecode::jit_cond as *const () as usize,
        crate::bytecode::jit_return as *const () as usize,
        crate::bytecode::jit_push_handler as *const () as usize,
        crate::bytecode::jit_pop_handler as *const () as usize,
        crate::bytecode::jit_unwind as *const () as usize,
        crate::bytecode::jit_call as *const () as usize,
        crate::bytecode::jit_call_hit as *const () as usize,
        crate::bytecode::jit_direct_finish as *const () as usize,
        crate::bytecode::jit_iteration as *const () as usize,
        crate::bytecode::jit_make_object as *const () as usize,
        crate::bytecode::jit_set_prop as *const () as usize,
        crate::bytecode::jit_get_prop as *const () as usize,
        crate::bytecode::jit_intrinsic as *const () as usize,
        crate::bytecode::jit_new as *const () as usize,
        crate::bytecode::jit_add_strings as *const () as usize,
        crate::bytecode::jit_instanceof as *const () as usize,
        crate::bytecode::jit_make_array as *const () as usize,
        crate::bytecode::jit_set_elem as *const () as usize,
        crate::bytecode::string_accumulation::jit_try as *const () as usize,
        crate::bytecode::jit_strict_eq as *const () as usize,
        crate::bytecode::jit_make_regexp as *const () as usize,
        crate::bytecode::jit_interrupt as *const () as usize,
        crate::bytecode::jit_loop_backedge as *const () as usize,
        crate::bytecode::jit_get_element as *const () as usize,
        crate::bytecode::jit_complete as *const () as usize,
        crate::bytecode::jit_slice_op as *const () as usize,
        crate::bytecode::jit_reference_op as *const () as usize,
        crate::bytecode::jit_make_closure as *const () as usize,
        crate::bytecode::jit_object_literal as *const () as usize,
        crate::bytecode::jit_abstract_operation as *const () as usize,
    ]
}

/// Helper table indices (multiplied by 8 in the emitted `ldr`).
pub const H_EXEC: usize = 0;
pub const H_COND: usize = 1;
// Index2 retains the synthetic return helper; native returns publish ctx.ret directly.
pub const H_PUSH_HANDLER: usize = 3;
pub const H_POP_HANDLER: usize = 4;
pub const H_UNWIND: usize = 5;
pub const H_CALL: usize = 6;
/// The call template's inline way-1 probe hit: skips the helper-side decode and probe loop.
pub const H_CALL_HIT: usize = 7;
/// Teardown for a direct (shared-ctx) call: drops, frame-pool return, FnFrame pop, tail drain.
pub const H_DIRECT_FINISH: usize = 8;
/// Iterator-protocol operations (see `bytecode::jit_iteration`).
pub const H_ITERATION: usize = 9;
/// Dedicated `Op::MakeObject` entry: template clone + stack-direct value writes, no op decode.
pub const H_MAKE_OBJECT: usize = 10;
/// Dedicated property-store entry (`SetProp`/`SetPropDrop`/`SetPropThisDrop`/`SetPropLocalDrop`
/// misses): straight into `set_prop_ic`, no generic op decode.
pub const H_SET_PROP: usize = 11;
/// Dedicated property-read entry (`GetProp`/`GetPropThis`/`GetPropLocal`/`GetMethod` misses):
/// straight into `get_prop_ic`.
pub const H_GET_PROP: usize = 12;
/// Specialized native calls (`String#slice`, `Object.hasOwn`) after the call IC identity probe.
pub const H_INTRINSIC: usize = 13;
/// Dedicated `Op::New` entry: constructor-cache probe and dispatch without generic op decode.
pub const H_NEW: usize = 14;
pub const H_ADD_STRINGS: usize = 15;
pub const H_INSTANCEOF: usize = 16;
/// Dedicated `Op::MakeArray` entry: moves stack values directly into dense storage.
pub const H_MAKE_ARRAY: usize = 17;
/// Dedicated element-store entry for dense misses and fresh indexed writes.
pub const H_SET_ELEM: usize = 18;
/// Complete primitive-string declarative accumulation. flag1 is an unchanged miss,
/// not an exception; only its guarded region emitters may call this helper.
pub const H_STRING_ACCUMULATION: usize = 19;
/// Strict equality misses after the generated fast path: handles content comparison and BigInts
/// without entering the generic bytecode decoder.
pub const H_STRICT_EQ: usize = 20;
/// Fresh RegExp-literal allocation using the chunk's immutable compiled-program cache.
pub const H_MAKE_REGEXP: usize = 21;
/// Amortized host cancellation/deadline poll at generated loop backedges.
pub const H_INTERRUPT: usize = 22;
/// Exact-PC loop back-edge feedback in diagnostic mode.
pub const H_LOOP_BACKEDGE: usize = 23;

/// Pure representation preparation: no author code or engine allocation/safepoint. The
/// activation owns `raw`; the optional f64 mirror leaves canonical Property addresses intact.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
unsafe extern "C" fn jit_prepare_packed_numeric_mirror(
    raw: *const std::cell::RefCell<crate::value::Object>,
) {
    // SAFETY: the activation owns `raw` (see above).
    let object = unsafe { &*raw };
    let Ok(mut body) = object.try_borrow_mut() else {
        return;
    };
    if matches!(body.exotic, crate::value::Exotic::Array) && body.ic_plain.get() {
        body.props.prepare_packed_numeric_mirror();
    }
}

pub const H_GET_METHOD_ELEM: usize = 24;
/// The historical method-read slot now serves every compact computed-read shape.
/// Keep its diagnostic identity and ABI index stable.
pub const H_GET_ELEM: usize = H_GET_METHOD_ELEM;
pub const H_COMPLETE: usize = 25;
pub const H_SLICE_OP: usize = 26;
pub const H_REFERENCE_OP: usize = 27;
/// `Op::MakeClosure` (see `bytecode::jit_make_closure`).
pub const H_MAKE_CLOSURE: usize = 28;
/// Incremental object-literal builder ops (see `bytecode::jit_object_literal`).
pub const H_OBJECT_LITERAL: usize = 29;
/// Self-hosted built-ins' abstract operations (see `bytecode::jit_abstract_operation`).
pub const H_ABSTRACT: usize = 30;
pub const N_HELPERS: usize = 31;

/// Stable diagnostic identities for the generated helper ABI. These labels are part of the
/// profile vocabulary; they intentionally do not expose helper addresses or Rust symbol names.
#[allow(dead_code)]
pub(crate) const HELPER_NAMES: [&str; N_HELPERS] = [
    "exec",
    "cond",
    "return",
    "push_handler",
    "pop_handler",
    "unwind",
    "call",
    "call_hit",
    "direct_finish",
    "iteration",
    "make_object",
    "set_prop",
    "get_prop",
    "intrinsic",
    "new",
    "add_strings",
    "instanceof",
    "make_array",
    "set_elem",
    "string_accumulation",
    "strict_eq",
    "make_regexp",
    "interrupt",
    "loop_backedge",
    "get_method_elem",
    "complete",
    "slice_op",
    "reference_op",
    "make_closure",
    "object_literal",
    "abstract_operation",
];

#[inline]
#[allow(dead_code)]
pub(crate) fn helper_name(index: usize) -> &'static str {
    HELPER_NAMES.get(index).copied().unwrap_or("invalid")
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod shared_stub_tests {
    use super::*;

    /// Loop bodies run from a backward branch's target through the branch; sequential and
    /// nested loops are both covered, forward branches and code between loops are not.
    #[test]
    fn loop_body_mask_spans_each_backward_branch() {
        use crate::bytecode::Op;
        let ops = [
            Op::Undef,              // 0
            Op::JumpIfFalse(8),     // 1  outer header (forward exit)
            Op::JumpIfFalsePeek(5), // 2  inner header
            Op::Pop,                // 3
            Op::Jump(2),            // 4  inner backedge
            Op::Pop,                // 5
            Op::Jump(1),            // 6  outer backedge
            Op::Undef,              // 7
            Op::Pop,                // 8
            Op::JumpIfTruePeek(8),  // 9  self-contained do-while
            Op::ReturnUndef,        // 10
        ];
        let expected = [
            false, true, true, true, true, true, true, false, true, true, false,
        ];
        assert_eq!(loop_body_mask(&ops), expected);
    }

    /// A loop body longer than the hot-loop bound (a flattened dispatch loop) keeps its
    /// sites on the shared stubs; a small loop nested inside it is still hot.
    #[test]
    fn loop_body_mask_leaves_oversized_loop_bodies_cold() {
        use crate::bytecode::Op;
        let span = hot_loop_ops() + 8;
        let mut ops = vec![Op::Undef; span + 2];
        ops[3] = Op::JumpIfFalsePeek(6); // inner header
        ops[5] = Op::Jump(3); // inner backedge
        ops[span] = Op::Jump(1); // outer backedge spanning more than the bound
        ops[span + 1] = Op::ReturnUndef;
        let mask = loop_body_mask(&ops);
        assert!(!mask[1] && !mask[2] && !mask[6] && !mask[span]);
        assert!(mask[3] && mask[4] && mask[5]);
    }

    /// Chunks compiled separately branch to one process-wide copy of each shared stub, and code
    /// running through those copies (free names, property ways, direct calls, intrinsics)
    /// computes what the tree-walker does.
    #[test]
    fn shared_stub_bodies_are_assembled_once_per_process() {
        let source = r#"
            var base = 40, text = 'abc';
            function name() { return base + 1; }
            function prop(o) { return o.x + o.y; }
            function leaf(v) { return v * 2; }
            function direct(v) { var r = leaf(v); return r; }
            function intr(s, n) { return s.charCodeAt(n); }
            var t = 0;
            for (var k = 0; k < 50; k++) {
                t += name() + prop({ x: k, y: 1 }) + direct(k) + intr(text, k % 3);
            }
            t
        "#;
        let mut results = Vec::new();
        for tier in [crate::bytecode::Tier::Interp, crate::bytecode::Tier::Jit] {
            let mut engine = crate::Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            match engine.eval(source, false).expect("stub fixture parses") {
                crate::Completion::Value(value) => results.push(value),
                crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
            }
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], "10674");
        if env_flag!("TRUST_JS_JIT_CHUNK_STUBS") || !shared_stubs_enabled() {
            return;
        }
        let keys: Vec<u32> = with_global_stubs(|table| {
            table
                .as_ref()
                .expect("compiled chunks requested shared stubs")
                .addresses
                .keys()
                .copied()
                .collect()
        })
        .unwrap();
        let name_stub = SharedStub::NameValuePtr { packed_ok: true }.key();
        assert!(keys.contains(&name_stub));
        assert!(keys
            .iter()
            .any(|&key| matches!(SharedStub::from_key(key), SharedStub::DirectCall { .. })));
    }

    #[test]
    fn shared_stubs_are_requested_once_and_forgotten_with_a_rewound_emission() {
        let mut a = asm::Asm::new();
        let name = SharedStub::NameValuePtr { packed_ok: true }.key();
        let first = a.shared_stub(name);
        assert_eq!(a.shared_stub(name), first, "one entry label per stub");
        let checkpoint = a.checkpoint();
        let transient = a.shared_stub(SharedStub::NameValuePtr { packed_ok: false }.key());
        assert_ne!(transient, first);
        a.rewind(checkpoint);
        let requested = a.take_shared_stubs();
        assert_eq!(
            requested,
            vec![(name, first)],
            "a rewound request is dropped"
        );
        assert!(a.take_shared_stubs().is_empty());
    }

    #[test]
    fn shared_stub_keys_round_trip() {
        let mut keys = vec![
            SharedStub::NameValuePtr { packed_ok: false }.key(),
            SharedStub::NameValuePtr { packed_ok: true }.key(),
            SharedStub::NameValuePtrIn.key(),
        ];
        #[cfg(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        ))]
        {
            for bits in 0..16u32 {
                keys.push(
                    SharedStub::PropWays(PropProbeFlags {
                        arr_ok: bits & 1 != 0,
                        wrap_ok: bits & 2 != 0,
                        method: bits & 4 != 0,
                        kc: bits & 8 != 0,
                    })
                    .key(),
                );
            }
            keys.push(SharedStub::CallSecondary.key());
            for argc in [0u8, 1, 8, 64] {
                for with_this in [false, true] {
                    keys.push(SharedStub::DirectCall { argc, with_this }.key());
                }
            }
        }
        for &key in &keys {
            assert_eq!(SharedStub::from_key(key).key(), key);
        }
        let mut unique = keys.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            unique.len(),
            keys.len(),
            "distinct stubs have distinct keys"
        );
    }
}

#[cfg(test)]
mod helper_identity_tests {
    use super::*;

    #[test]
    fn helper_identity_table_matches_the_machine_abi() {
        assert_eq!(HELPER_NAMES.len(), N_HELPERS);
        assert_eq!(helper_name(H_EXEC), "exec");
        assert_eq!(helper_name(H_CALL_HIT), "call_hit");
        assert_eq!(helper_name(H_LOOP_BACKEDGE), "loop_backedge");
        assert_eq!(helper_name(H_ITERATION), "iteration");
        assert_eq!(helper_name(H_STRING_ACCUMULATION), "string_accumulation");
        assert_eq!(helper_name(N_HELPERS), "invalid");
    }
}

#[cfg(test)]
mod native_op_tests {
    use super::*;

    fn tables() -> NativeOpTables {
        NativeOpTables::default()
    }

    #[test]
    fn first_registration_wins_for_folded_addresses() {
        fn probe_op(_: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
            Ok(Value::Undefined)
        }
        let mut t = tables();
        register_native_name(&mut t, probe_op as *const () as usize, "first");
        register_native_name(&mut t, probe_op as *const () as usize, "second");
        let call = crate::value::Callable::Native(probe_op);
        record_native_op_src(&mut t, NativeLabelSrc::Call(&call), 5, true);
        let rendered = render_native_ops(&t);
        assert!(
            rendered.contains("\"operation\":\"first\",\"calls\":1"),
            "{rendered}"
        );
        assert!(!rendered.contains("\"operation\":\"second\""), "{rendered}");
    }

    #[test]
    fn relabel_overwrites_for_namespaces() {
        fn probe_ns_op(_: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
            Ok(Value::Undefined)
        }
        let mut t = tables();
        register_native_name(&mut t, probe_ns_op as *const () as usize, "read");
        relabel_native_name(
            &mut t,
            probe_ns_op as *const () as usize,
            "fs.read".to_string(),
        );
        record_native_op_src(
            &mut t,
            NativeLabelSrc::Addr(probe_ns_op as *const () as usize),
            5,
            true,
        );
        let rendered = render_native_ops(&t);
        assert!(rendered.contains("\"operation\":\"fs.read\""), "{rendered}");
    }

    #[test]
    fn data_callables_report_their_immutable_identity() {
        let func: Rc<crate::value::NativeClosure> =
            Rc::new(|_, _, _| Err(crate::value::Value::Undefined));
        let data = Rc::new(crate::value::NativeCallable {
            body: crate::value::NativeCallableBody::Opaque(func),
            retained: None,
            identity: Rc::from("test.extension.op"),
        });
        let call = crate::value::Callable::NativeData(data);
        let mut t = tables();
        record_native_op_src(&mut t, NativeLabelSrc::Call(&call), 5, true);
        let rendered = render_native_ops(&t);
        assert!(
            rendered.contains("\"operation\":\"test.extension.op\""),
            "{rendered}"
        );
    }

    #[test]
    fn unknown_addresses_never_render_as_numbers() {
        let mut t = tables();
        record_native_op_src(&mut t, NativeLabelSrc::Addr(0xDEAD), 5, true);
        let rendered = render_native_ops(&t);
        assert!(
            rendered.contains("\"operation\":\"<native>\""),
            "{rendered}"
        );
        assert!(!rendered.contains("0x"), "{rendered}");
        assert!(!rendered.contains("57005"), "{rendered}");
    }

    #[test]
    fn aggregation_counts_failures_and_time() {
        let mut t = tables();
        record_native_op(&mut t.stats, "Math.abs", 10, true);
        record_native_op(&mut t.stats, "Math.abs", 20, false);
        let rendered = render_native_ops(&t);
        assert!(
            rendered.contains("\"operation\":\"Math.abs\",\"calls\":2,\"failures\":1"),
            "{rendered}"
        );
    }

    #[test]
    fn rendering_is_deterministic_and_escapes_labels() {
        let mut t = tables();
        for label in ["b", "a\"q", "c\\d", "e\nf", "g\u{1}h"] {
            record_native_op(&mut t.stats, label, 0, true);
        }
        let rendered = render_native_ops(&t);
        let a = rendered.find("\"operation\":\"a").expect("a row");
        let b = rendered.find("\"operation\":\"b\"").expect("b row");
        let c = rendered.find("\"operation\":\"c").expect("c row");
        assert!(a < b && b < c, "{rendered}");
        assert!(rendered.contains("a\\\"q"), "{rendered}");
        assert!(rendered.contains("c\\\\d"), "{rendered}");
        assert!(rendered.contains("e\\nf"), "{rendered}");
        assert!(rendered.contains("g\\u0001h"), "{rendered}");
        assert!(!rendered.contains("0x"), "{rendered}");
    }

    #[test]
    fn overflow_bucket_bounds_distinct_labels() {
        let mut t = tables();
        for i in 0..(NATIVE_OP_LABEL_CAP + 3) {
            let label = format!("op{i}");
            record_native_op(&mut t.stats, &label, 1, true);
        }
        assert_eq!(t.stats.len(), NATIVE_OP_LABEL_CAP + 1);
        let rendered = render_native_ops(&t);
        assert!(
            rendered.contains("\"operation\":\"<overflow>\",\"calls\":3"),
            "{rendered}"
        );
    }
}

/// ARM64 condition codes used by the inline templates.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_EQ: u32 = 0;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_NE: u32 = 1;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_HS: u32 = 2;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_LO: u32 = 3;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_MI: u32 = 4;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_HI: u32 = 8;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_LS: u32 = 9;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_GE: u32 = 10;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_GT: u32 = 12;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const C_VS: u32 = 6;

/// Condition-helper modes (the `w1` immediate for `H_COND`).
pub const COND_POP_TRUTHY: u32 = 0;
pub const COND_PEEK_TRUTHY: u32 = 1;
#[cfg_attr(all(test, not(target_arch = "x86_64")), allow(dead_code))]
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub const COND_PEEK_NOT_NULLISH: u32 = 2;

// The inline fast paths read Value directly: repr(u8) tag byte at offset 0, payload at
// offset 8, 16 bytes total on 64-bit. Tags 0..=4 (Undefined/Empty/Null/Bool/Num) are trivially
// copyable. Only 64-bit desktop JIT targets depend on this; on wasm32 `Value` is smaller.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const _: () = assert!(std::mem::size_of::<Value>() == 16);
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const _: () = assert!(std::mem::align_of::<Value>() == 8);
// The offsets below bake 8-byte pointers into the emitted templates: JIT-platform only (on
// wasm32 pointers are 4 bytes and none of this code exists).
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod layout_asserts {
    use super::JitCtx;
    // Both backends read the context's leading words by fixed offset (x64: the helper table at
    // [r12], the operand base or final top into r13 and back, the slots into r15, the
    // environment, `this`, the global scope and the interpreter).
    const _: () = assert!(std::mem::offset_of!(JitCtx, helpers) == 0);
    const _: () = assert!(std::mem::offset_of!(JitCtx, stack_base) == 8);
    const _: () = assert!(std::mem::offset_of!(JitCtx, final_sp) == 16);
    const _: () = assert!(std::mem::offset_of!(JitCtx, slots) == 24);
    const _: () = assert!(std::mem::offset_of!(JitCtx, inline_ic_safe) == 32);
    const _: () = assert!(std::mem::offset_of!(JitCtx, env_raw) == 40);
    // [48]: the 8-byte packed `this` word.
    const _: () = assert!(
        std::mem::offset_of!(JitCtx, this_word) == 48
            && std::mem::size_of::<crate::value::PackedValue>() == 8
    );
    const _: () = assert!(std::mem::offset_of!(JitCtx, global_body) == 56);
    const _: () = assert!(std::mem::offset_of!(JitCtx, genv) == 64);
    // 3b reads Interp state from machine code through ctx.interp.
    const _: () = assert!(std::mem::offset_of!(JitCtx, interp) == 72);
    // ARM64's call, frame-push and direct-call sequences.
    #[cfg(target_arch = "aarch64")]
    mod arm64 {
        // The call template's inline way-1 probe reads these CallIc fields by fixed offset.
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, callee) == 0);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, global_env) == 32);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, epoch) == 56);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, n_params) == 42);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, n_slots) == 44);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, direct) == 46);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, chunk_raw) == 64);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, code_mem) == 72);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, pc_offs_ptr) == 80);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, native) == 88);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, intrinsic) == 96);
        const _: () = assert!(std::mem::offset_of!(crate::bytecode::CallIc, realm) == 104);
        const _: () =
            assert!(std::mem::size_of::<std::cell::Cell<crate::bytecode::CallIc>>() == 112);
        // The asm frame push writes FnFrame fields by fixed offset.
        const _: () = assert!(std::mem::offset_of!(crate::interpreter::FnFrame, fn_ptr) == 0);
        const _: () = assert!(std::mem::offset_of!(crate::interpreter::FnFrame, call_site) == 8);
        const _: () = assert!(std::mem::offset_of!(crate::interpreter::FnFrame, strict) == 12);
        const _: () = assert!(std::mem::offset_of!(crate::interpreter::FnFrame, construct) == 13);
        const _: () = assert!(std::mem::offset_of!(crate::interpreter::FnFrame, extra) == 16);
        const _: () = assert!(std::mem::size_of::<crate::interpreter::FnFrame>() == 24);
        // The direct-call sequence reads the callee's code/pc_offsets straight from its JitCode.
        const _: () = assert!(std::mem::offset_of!(crate::jit::JitCode, mem) == 0);
        const _: () = assert!(std::mem::offset_of!(crate::jit::JitCode, pc_offsets) == 16);
    }
}

/// Two-register return for helpers that produce (new sp, flag) — x0/x1 under the C ABI.
#[repr(C)]
pub struct SpFlag {
    pub sp: *mut PackedValue,
    pub flag: u64,
}

// ---------------------------------------------------------------------------------------------
// ARM64 assembler (the ~20 encodings the templates need)
// ---------------------------------------------------------------------------------------------

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod asm {
    /// Instruction buffer with label/patch support. Registers are plain u32 numbers (x0..x30,
    /// sp=31 where encodable); labels are indices into `patches`.
    #[cfg_attr(test, derive(Clone))]
    pub struct Asm {
        pub buf: Vec<u32>,
        /// (instruction index, label id, kind) — resolved in `finish`.
        patches: Vec<(usize, usize, PatchKind)>,
        labels: Vec<Option<usize>>, // label id → instruction index
        /// Out-of-line routines shared by every site of this compilation, as (stub key, entry
        /// label). The compiler emits each once after the body (see `super::SharedStub`).
        stubs: Vec<(u32, usize)>,
        /// The code being emitted runs once per loop iteration: sites keep their fast paths in
        /// line instead of calling a shared stub (see `super::use_shared_stub`).
        hot: bool,
        /// The chunk compiler emits the shared stubs sites request (see `super::SharedStub`);
        /// template tests that assemble loose code leave this clear and get inline sequences.
        stubs_ok: bool,
        /// Profiler attribution markers (`TRUST_JS_JIT_MAP` only): a bound label and the template
        /// family whose code starts there. Nested families restore their parent when they end.
        marks: Vec<(usize, &'static str)>,
        mark_stack: Vec<&'static str>,
        /// Immutable scalar original-PC/slot/exit-label plans; no JavaScript owners.
        /// Checked numeric replay uses the same before-read transaction as ordinary entry.
        string_accumulations: Vec<(u32, u16, usize)>,
        /// Whether the code being emitted is a chunk body's baseline code, where a call to an
        /// allocating helper ends in the explicit poll (see `super::emit_helper_end_poll`).
        /// Clear in region code, shared stubs and loose test assemblies.
        end_polls: bool,
    }

    #[derive(Clone, Copy)]
    enum PatchKind {
        /// Unconditional B: imm26.
        B,
        /// B.cond/CBZ/CBNZ: imm19.
        Cb,
    }

    impl Asm {
        pub fn new() -> Asm {
            Asm {
                buf: Vec::new(),
                patches: Vec::new(),
                labels: Vec::new(),
                stubs: Vec::new(),
                hot: false,
                stubs_ok: false,
                marks: Vec::new(),
                mark_stack: Vec::new(),
                string_accumulations: Vec::new(),
                end_polls: false,
            }
        }
        /// Set whether helper calls emitted from here on end in the explicit poll (see
        /// the `end_polls` field); returns the previous setting.
        pub fn set_end_polls(&mut self, on: bool) -> bool {
            std::mem::replace(&mut self.end_polls, on)
        }
        pub fn end_polls(&self) -> bool {
            self.end_polls
        }
        /// Start a profiler attribution region (a no-op unless `TRUST_JS_JIT_MAP` is set).
        pub fn mark_push(&mut self, name: &'static str) {
            if !super::jit_map_enabled() {
                return;
            }
            self.mark_stack.push(name);
            let label = self.new_label();
            self.bind(label);
            self.marks.push((label, name));
        }
        /// Make `name` the attribution of everything emitted from here on, outside any template
        /// family that is open (`TRUST_JS_JIT_MAP` only).
        pub fn mark_op(&mut self, name: &'static str) {
            if !super::jit_map_enabled() {
                return;
            }
            self.mark_stack.clear();
            self.mark_stack.push(name);
            let label = self.new_label();
            self.bind(label);
            self.marks.push((label, name));
        }
        /// End the innermost attribution region, resuming its parent's.
        pub fn mark_pop(&mut self) {
            if !super::jit_map_enabled() {
                return;
            }
            self.mark_stack.pop();
            let parent = self.mark_stack.last().copied().unwrap_or("-");
            let label = self.new_label();
            self.bind(label);
            self.marks.push((label, parent));
        }
        /// Whether any emitted branch targets `label`.
        pub fn label_referenced(&self, label: usize) -> bool {
            self.patches.iter().any(|&(_, target, _)| target == label)
        }
        /// Mark the following code as (not) running once per loop iteration.
        pub fn set_hot(&mut self, hot: bool) {
            self.hot = hot;
        }
        pub fn hot(&self) -> bool {
            self.hot
        }
        /// Declare that the compiler will emit every shared stub this code requests.
        pub fn set_stubs_ok(&mut self, ok: bool) {
            self.stubs_ok = ok;
        }
        pub fn stubs_ok(&self) -> bool {
            self.stubs_ok
        }
        pub fn add_string_accumulation(&mut self, pc: u32, slot: u16, exit: usize) {
            self.string_accumulations.push((pc, slot, exit));
        }
        pub fn string_accumulation(&self, pc: u32) -> Option<(u16, usize)> {
            self.string_accumulations
                .binary_search_by_key(&pc, |&(start, _, _)| start)
                .ok()
                .map(|index| {
                    let (_, slot, exit) = self.string_accumulations[index];
                    (slot, exit)
                })
        }
        /// The entry label of the shared stub `key`, requesting its emission on first use.
        pub fn shared_stub(&mut self, key: u32) -> usize {
            if let Some(&(_, label)) = self.stubs.iter().find(|&&(k, _)| k == key) {
                return label;
            }
            let label = self.new_label();
            self.stubs.push((key, label));
            label
        }
        /// Take the stubs requested so far (the compiler binds and emits each one).
        pub fn take_shared_stubs(&mut self) -> Vec<(u32, usize)> {
            std::mem::take(&mut self.stubs)
        }
        /// Transaction for an optional region. The region may allocate/bind only NEW labels;
        /// branches to existing baseline labels are fine. This caps added native bytes before
        /// executable allocation without cloning the function's growing instruction buffer.
        pub fn checkpoint(&self) -> (usize, usize, usize) {
            (self.buf.len(), self.patches.len(), self.labels.len())
        }
        pub fn rewind(&mut self, checkpoint: (usize, usize, usize)) {
            self.buf.truncate(checkpoint.0);
            self.patches.truncate(checkpoint.1);
            self.labels.truncate(checkpoint.2);
            // A stub first requested by the discarded emission loses its label with it.
            self.stubs.retain(|&(_, label)| label < checkpoint.2);
            self.marks.retain(|&(label, _)| label < checkpoint.2);
        }
        /// Labels actually referenced by an optional emission, before relaxation. A region
        /// uses this to omit unreachable deoptimization stubs and unnecessary baseline entries.
        pub fn referenced_since(
            &self,
            checkpoint: (usize, usize, usize),
        ) -> impl Iterator<Item = usize> + '_ {
            self.patches[checkpoint.1..]
                .iter()
                .map(|&(_, label, _)| label)
        }
        pub fn new_label(&mut self) -> usize {
            self.labels.push(None);
            self.labels.len() - 1
        }
        pub fn bind(&mut self, label: usize) {
            self.labels[label] = Some(self.buf.len());
        }
        fn emit(&mut self, i: u32) {
            self.buf.push(i);
        }
        /// A data word in the instruction stream, placed where execution never reaches it (after
        /// an unconditional branch). Branch relaxation inserts only after conditional branches
        /// it widens, so the word keeps its offset from the instructions before it.
        pub fn word(&mut self, value: u32) {
            self.emit(value);
        }

        /// Best-effort hot-code alignment before branch relaxation. Padding executes only on
        /// fallthrough; bind the loop's back-edge label after it. NOP preserves registers/NZCV.
        pub fn align_hot(&mut self, bytes: usize) {
            debug_assert!(bytes >= 4 && bytes.is_power_of_two());
            while !(self.buf.len() * 4).is_multiple_of(bytes) {
                self.emit(0xD503_201F);
            }
        }

        /// movz xd, #imm16, lsl #(shift*16)
        /// str wt, [xn, #imm] (scaled, imm/4)
        pub fn str_w_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(4) && imm_bytes / 4 < 4096);
            self.emit(0xB900_0000 | ((imm_bytes / 4) << 10) | (rn << 5) | rt);
        }
        /// ldr xt, [xn, xm, lsl #3] (register-offset, scaled)
        pub fn ldr_x_lsl3(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xF860_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldrb wt, [xn, xm] (register-offset, unscaled)
        pub fn ldrb_reg(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0x3860_6800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldrh wt, [xn, #imm] (scaled, imm/2)
        pub fn ldrh_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(2) && imm_bytes / 2 < 4096);
            self.emit(0x7940_0000 | ((imm_bytes / 2) << 10) | (rn << 5) | rt);
        }
        pub fn movz(&mut self, rd: u32, imm16: u32, shift: u32) {
            self.emit(0xD280_0000 | (shift << 21) | (imm16 << 5) | rd);
        }
        /// movk xd, #imm16, lsl #(shift*16)
        #[allow(dead_code)] // the inline fast-path pass uses these
        pub fn movk(&mut self, rd: u32, imm16: u32, shift: u32) {
            self.emit(0xF280_0000 | (shift << 21) | (imm16 << 5) | rd);
        }
        /// mov xd, xn (ORR xd, xzr, xn)
        pub fn mov(&mut self, rd: u32, rn: u32) {
            self.emit(0xAA00_03E0 | (rn << 16) | rd);
        }
        /// Exact 64-bit materialization using the shortest MOVZ/MOVN + MOVK chain.
        /// MOVZ zero-fills other lanes; MOVN one-fills them (Arm 100076, D2.106–108).
        /// Never reinterpret the bits: callers also pass pointers, packed tags and binary64
        /// signed zero/NaN encodings (ECMA-262 e28783d5, sec-ecmascript-language-types-number-type).
        #[allow(dead_code)]
        pub fn mov_imm64(&mut self, rd: u32, v: u64) {
            debug_assert!(rd < 32);
            let lanes = std::array::from_fn::<_, 4, _>(|lane| ((v >> (lane * 16)) & 0xffff) as u32);
            let zero_cost = lanes.iter().filter(|&&lane| lane != 0).count();
            let ones_cost = lanes.iter().filter(|&&lane| lane != 0xffff).count();
            let inverted = ones_cost < zero_cost;
            let fill = if inverted { 0xffff } else { 0 };
            let first = lanes.iter().position(|&lane| lane != fill).unwrap_or(0);
            if inverted {
                // MOVN Xd, #imm16, LSL #(first*16); unlike MOVK it initializes every bit.
                self.emit(
                    0x9280_0000 | ((first as u32) << 21) | ((!lanes[first] & 0xffff) << 5) | rd,
                );
            } else {
                self.movz(rd, lanes[first], first as u32);
            }
            for (lane, &bits) in lanes.iter().enumerate() {
                if lane != first && bits != fill {
                    self.movk(rd, bits, lane as u32);
                }
            }
        }
        /// ldr xd, [xn, #imm] (imm = byte offset, multiple of 8, unsigned)
        pub fn ldr_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(8) && imm_bytes / 8 < 4096);
            self.emit(0xF940_0000 | ((imm_bytes / 8) << 10) | (rn << 5) | rt);
        }
        /// str xt, [xn, #imm]
        pub fn str_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(8) && imm_bytes / 8 < 4096);
            self.emit(0xF900_0000 | ((imm_bytes / 8) << 10) | (rn << 5) | rt);
        }
        /// stp xt1, xt2, [sp, #-imm]! (pre-index, imm = positive byte count, multiple of 8)
        pub fn stp_pre(&mut self, rt1: u32, rt2: u32, imm_bytes: i32) {
            debug_assert!(
                imm_bytes % 8 == 0 && (-512..=504).contains(&imm_bytes),
                "AArch64 STP pre-index offset out of signed imm7 range: {imm_bytes}"
            );
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0xA980_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt1);
        }
        /// ldp xt1, xt2, [sp], #imm (post-index)
        pub fn ldp_post(&mut self, rt1: u32, rt2: u32, imm_bytes: i32) {
            debug_assert!(
                imm_bytes % 8 == 0 && (-512..=504).contains(&imm_bytes),
                "AArch64 LDP post-index offset out of signed imm7 range: {imm_bytes}"
            );
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0xA8C0_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt1);
        }
        /// stp xt1, xt2, [sp, #imm] (signed offset form)
        pub fn stp_off(&mut self, rt1: u32, rt2: u32, imm_bytes: i32) {
            debug_assert!(
                imm_bytes % 8 == 0 && (-512..=504).contains(&imm_bytes),
                "AArch64 STP signed offset out of imm7 range: {imm_bytes}"
            );
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0xA900_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt1);
        }
        /// ldp xt1, xt2, [sp, #imm]
        pub fn ldp_off(&mut self, rt1: u32, rt2: u32, imm_bytes: i32) {
            debug_assert!(
                imm_bytes % 8 == 0 && (-512..=504).contains(&imm_bytes),
                "AArch64 LDP signed offset out of imm7 range: {imm_bytes}"
            );
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0xA940_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt1);
        }
        /// stp xt1, xt2, [xn, #imm] (signed offset, any base register)
        pub fn stp_base(&mut self, rt1: u32, rt2: u32, rn: u32, imm_bytes: i32) {
            debug_assert!(
                imm_bytes % 8 == 0 && (-512..=504).contains(&imm_bytes),
                "AArch64 STP signed offset out of imm7 range: {imm_bytes}"
            );
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0xA900_0000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1);
        }
        /// ldp xt1, xt2, [xn, #imm] (signed offset, any base register; xn may be xt1 or xt2)
        pub fn ldp_base(&mut self, rt1: u32, rt2: u32, rn: u32, imm_bytes: i32) {
            debug_assert!(
                imm_bytes % 8 == 0 && (-512..=504).contains(&imm_bytes),
                "AArch64 LDP signed offset out of imm7 range: {imm_bytes}"
            );
            debug_assert_ne!(
                rt1, rt2,
                "an LDP loading one register twice is unpredictable"
            );
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0xA940_0000 | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1);
        }
        pub fn blr(&mut self, rn: u32) {
            self.emit(0xD63F_0000 | (rn << 5));
        }
        pub fn br(&mut self, rn: u32) {
            self.emit(0xD61F_0000 | (rn << 5));
        }
        pub fn ret(&mut self) {
            self.emit(0xD65F_03C0);
        }
        /// b label (patched)
        pub fn b(&mut self, label: usize) {
            self.patches.push((self.buf.len(), label, PatchKind::B));
            self.emit(0x1400_0000);
        }
        /// bl label (patched; same imm26 shape as B). The callee stub must preserve x19..x22
        /// and, if it calls out itself, spill/reload x30.
        pub fn bl_label(&mut self, label: usize) {
            self.patches.push((self.buf.len(), label, PatchKind::B));
            self.emit(0x9400_0000);
        }
        /// cbz x/w reg, label (patched); `is64` selects X vs W.
        pub fn cbz(&mut self, rt: u32, is64: bool, label: usize) {
            self.patches.push((self.buf.len(), label, PatchKind::Cb));
            self.emit(if is64 { 0xB400_0000 } else { 0x3400_0000 } | rt);
        }
        /// cbnz x/w reg, label (patched)
        pub fn cbnz(&mut self, rt: u32, is64: bool, label: usize) {
            self.patches.push((self.buf.len(), label, PatchKind::Cb));
            self.emit(if is64 { 0xB500_0000 } else { 0x3500_0000 } | rt);
        }

        /// ldrb wt, [xn, #imm] (unsigned byte offset)
        pub fn ldrb_imm(&mut self, rt: u32, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x3940_0000 | (imm << 10) | (rn << 5) | rt);
        }
        /// strb wt, [xn, #imm]
        #[allow(dead_code)]
        pub fn strb_imm(&mut self, rt: u32, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x3900_0000 | (imm << 10) | (rn << 5) | rt);
        }
        /// sturh wt, [xn, #simm9] (unscaled)
        pub fn sturh(&mut self, rt: u32, rn: u32, simm9: i32) {
            debug_assert!((-256..256).contains(&simm9));
            self.emit(0x7800_0000 | (((simm9 as u32) & 0x1FF) << 12) | (rn << 5) | rt);
        }
        /// ldurb wt, [xn, #simm9]
        pub fn ldurb(&mut self, rt: u32, rn: u32, simm9: i32) {
            self.emit(0x3840_0000 | (((simm9 as u32) & 0x1FF) << 12) | (rn << 5) | rt);
        }
        /// ldur xt, [xn, #simm9]
        pub fn ldur(&mut self, rt: u32, rn: u32, simm9: i32) {
            self.emit(0xF840_0000 | (((simm9 as u32) & 0x1FF) << 12) | (rn << 5) | rt);
        }
        /// stur xt, [xn, #simm9]
        pub fn stur(&mut self, rt: u32, rn: u32, simm9: i32) {
            self.emit(0xF800_0000 | (((simm9 as u32) & 0x1FF) << 12) | (rn << 5) | rt);
        }
        /// ldr wt, [xn, #imm] (32-bit, unsigned scaled by 4)
        pub fn ldr_w_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(4) && imm_bytes / 4 < 4096);
            self.emit(0xB940_0000 | ((imm_bytes / 4) << 10) | (rn << 5) | rt);
        }
        /// FCVT Sd, Dn / FCVT Dd, Sn. Narrowing uses FPCR's roundTiesToEven default.
        pub fn fcvt_float_width(&mut self, rd: u32, rn: u32, narrow: bool) {
            self.emit((if narrow { 0x1E62_4000 } else { 0x1E22_C000 }) | (rn << 5) | rd);
        }
        /// sdiv wd, wn, wm (truncating; wn/0 = 0 and i32::MIN/-1 = i32::MIN, no trap)
        pub fn sdiv_w(&mut self, rd: u32, rn: u32, rm: u32) {
            self.emit(0x1AC0_0C00 | (rm << 16) | (rn << 5) | rd);
        }
        /// msub wd, wn, wm, wa  (wd = wa - wn*wm, wrapping)
        pub fn msub_w(&mut self, rd: u32, rn: u32, rm: u32, ra: u32) {
            self.emit(0x1B00_8000 | (rm << 16) | (ra << 10) | (rn << 5) | rd);
        }
        /// madd xd, xn, xm, xa  (xd = xn*xm + xa)
        pub fn madd(&mut self, rd: u32, rn: u32, rm: u32, ra: u32) {
            self.emit(0x9B00_0000 | (rm << 16) | (ra << 10) | (rn << 5) | rd);
        }
        /// cmp wn, wm  (SUBS wzr, wn, wm)
        pub fn cmp_reg_w(&mut self, rn: u32, rm: u32) {
            self.emit(0x6B00_001F | (rm << 16) | (rn << 5));
        }
        /// cmp xn, #imm12
        pub fn cmp_imm_x(&mut self, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0xF100_001F | (imm << 10) | (rn << 5));
        }
        /// ldur dt, [xn, #simm9]
        pub fn ldur_d(&mut self, rt: u32, rn: u32, simm9: i32) {
            self.emit(0xFC40_0000 | (((simm9 as u32) & 0x1FF) << 12) | (rn << 5) | rt);
        }
        /// stur dt, [xn, #simm9]
        pub fn stur_d(&mut self, rt: u32, rn: u32, simm9: i32) {
            self.emit(0xFC00_0000 | (((simm9 as u32) & 0x1FF) << 12) | (rn << 5) | rt);
        }
        /// ldr dt, [xn, #imm] (scaled)
        pub fn ldr_d_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(8) && imm_bytes / 8 < 4096);
            self.emit(0xFD40_0000 | ((imm_bytes / 8) << 10) | (rn << 5) | rt);
        }
        /// str dt, [xn, #imm] (scaled)
        pub fn str_d_imm(&mut self, rt: u32, rn: u32, imm_bytes: u32) {
            debug_assert!(imm_bytes.is_multiple_of(8) && imm_bytes / 8 < 4096);
            self.emit(0xFD00_0000 | ((imm_bytes / 8) << 10) | (rn << 5) | rt);
        }
        /// add xd, xn, #imm12
        pub fn add_imm(&mut self, rd: u32, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x9100_0000 | (imm << 10) | (rn << 5) | rd);
        }
        /// sub xd, xn, #imm12
        pub fn sub_imm(&mut self, rd: u32, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0xD100_0000 | (imm << 10) | (rn << 5) | rd);
        }
        /// cmp wn, #imm12
        pub fn cmp_imm_w(&mut self, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x7100_001F | (imm << 10) | (rn << 5));
        }
        /// b.cond label (patched; imm19 shares the CBZ patch shape)
        pub fn b_cond(&mut self, cond: u32, label: usize) {
            self.patches.push((self.buf.len(), label, PatchKind::Cb));
            self.emit(0x5400_0000 | cond);
        }
        /// fadd/fsub/fmul/fdiv dd, dn, dm — op: 0=add,1=sub,2=mul,3=div
        pub fn f_arith(&mut self, op: u32, rd: u32, rn: u32, rm: u32) {
            let bits = match op {
                0 => 0x1E60_2800u32,
                1 => 0x1E60_3800,
                2 => 0x1E60_0800,
                _ => 0x1E60_1800,
            };
            self.emit(bits | (rm << 16) | (rn << 5) | rd);
        }
        /// fcmp dn, dm
        pub fn fcmp(&mut self, rn: u32, rm: u32) {
            self.emit(0x1E60_2000 | (rm << 16) | (rn << 5));
        }
        /// cset wd, cond (CSINC wd, wzr, wzr, !cond)
        pub fn cset_w(&mut self, rd: u32, cond: u32) {
            self.emit(0x1A9F_07E0 | ((cond ^ 1) << 12) | rd);
        }
        /// fmov dd, #1.0
        pub fn fmov_one(&mut self, rd: u32) {
            self.emit(0x1E6E_1000 | rd);
        }
        /// fcvtzu wd, dn (float → unsigned 32-bit, round toward zero, saturating)
        pub fn fcvtzu_w_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E79_0000 | (rn << 5) | rd);
        }
        /// ucvtf dd, wn (unsigned 32-bit → double, exact)
        pub fn ucvtf_d_w(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E63_0000 | (rn << 5) | rd);
        }
        /// fcvtns wd, dn (float → signed 32-bit, round to nearest with ties to even, saturating;
        /// NaN gives 0): the rounding of ECMA-262 ToUint8Clamp before its [0, 255] clamp.
        pub fn fcvtns_w_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E60_0000 | (rn << 5) | rd);
        }
        /// fcvtzs xd, dn (float → signed 64-bit, round toward zero, saturating)
        pub fn fcvtzs_x_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x9E78_0000 | (rn << 5) | rd);
        }
        /// fcvtzs wd, dn (float → signed 32-bit, round toward zero, saturating)
        pub fn fcvtzs_w_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E78_0000 | (rn << 5) | rd);
        }
        /// fjcvtzs wd, dn (FEAT_JSCVT): ECMAScript ToInt32 of any double in one instruction —
        /// truncate toward zero, keep the low 32 bits (mod 2^32), NaN and ±Infinity give 0.
        /// Emit only when [`super::jscvt_available`].
        pub fn fjcvtzs_w_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E7E_0000 | (rn << 5) | rd);
        }
        /// scvtf dd, xn (signed 64-bit → double, round to nearest)
        pub fn scvtf_d_x(&mut self, rd: u32, rn: u32) {
            self.emit(0x9E62_0000 | (rn << 5) | rd);
        }
        /// scvtf dd, wn (signed 32-bit → double, exact)
        pub fn scvtf_d_w(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E62_0000 | (rn << 5) | rd);
        }
        /// frintz dd, dn (round toward zero to integral)
        pub fn frintz(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E65_C000 | (rn << 5) | rd);
        }
        /// frintm dd, dn (round toward minus infinity to integral)
        pub fn frintm(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E65_4000 | (rn << 5) | rd);
        }
        /// frintp dd, dn (round toward plus infinity to integral)
        pub fn frintp(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E64_C000 | (rn << 5) | rd);
        }
        /// fabs dd, dn
        pub fn fabs(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E60_C000 | (rn << 5) | rd);
        }
        /// fmin/fmax dd, dn, dm — NaN in, NaN out; -0 orders below +0 (ECMA-262 §21.3.2.25-26).
        pub fn fminmax(&mut self, max: bool, rd: u32, rn: u32, rm: u32) {
            let bits = if max { 0x1E60_4800u32 } else { 0x1E60_5800 };
            self.emit(bits | (rm << 16) | (rn << 5) | rd);
        }
        /// clz wd, wn
        pub fn clz_w(&mut self, rd: u32, rn: u32) {
            self.emit(0x5AC0_1000 | (rn << 5) | rd);
        }
        /// mul wd, wn, wm (MADD wd, wn, wm, wzr; the low 32 bits of the product)
        pub fn mul_w(&mut self, rd: u32, rn: u32, rm: u32) {
            self.emit(0x1B00_7C00 | (rm << 16) | (rn << 5) | rd);
        }
        /// fmov dd, xn (bit move)
        pub fn fmov_d_x(&mut self, rd: u32, rn: u32) {
            self.emit(0x9E67_0000 | (rn << 5) | rd);
        }
        /// fneg dd, dn
        pub fn fneg(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E61_4000 | (rn << 5) | rd);
        }
        /// fsqrt dd, dn
        pub fn fsqrt(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E61_C000 | (rn << 5) | rd);
        }
        /// fmov dd, dn
        pub fn fmov_d_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x1E60_4000 | (rn << 5) | rd);
        }
        /// stp dt1, dt2, [sp, #imm] (SIMD&FP 64-bit, signed offset)
        pub fn stp_d_off(&mut self, rt1: u32, rt2: u32, imm_bytes: i32) {
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0x6D00_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt1);
        }
        /// ldp dt1, dt2, [sp, #imm]
        pub fn ldp_d_off(&mut self, rt1: u32, rt2: u32, imm_bytes: i32) {
            let imm7 = ((imm_bytes / 8) & 0x7f) as u32;
            self.emit(0x6D40_0000 | (imm7 << 15) | (rt2 << 10) | (31 << 5) | rt1);
        }
        /// and/orr/eor wd, wn, wm — op: 0=and, 1=orr, 2=eor
        pub fn logic_w(&mut self, op: u32, rd: u32, rn: u32, rm: u32) {
            let bits = match op {
                0 => 0x0A00_0000u32,
                1 => 0x2A00_0000,
                _ => 0x4A00_0000,
            };
            self.emit(bits | (rm << 16) | (rn << 5) | rd);
        }
        /// mvn wd, wm (ORN wd, wzr, wm): bitwise complement of the low 32 bits.
        pub fn mvn_w(&mut self, rd: u32, rm: u32) {
            self.emit(0x2A20_03E0 | (rm << 16) | rd);
        }
        /// orr xd, xn, #(2^ones - 1): sets the low `ones` bits (ORR (immediate), sf=N=1, immr=0,
        /// imms=ones-1; Arm 100076, D2.110).
        pub fn orr_low_ones_x(&mut self, rd: u32, rn: u32, ones: u32) {
            assert!(rd < 32 && rn < 32 && (1..64).contains(&ones));
            self.emit(0xB240_0000 | ((ones - 1) << 10) | (rn << 5) | rd);
        }
        /// and/orr/eor xd, xn, xm — op: 0=and, 1=orr, 2=eor
        pub fn logic_x(&mut self, op: u32, rd: u32, rn: u32, rm: u32) {
            let bits = match op {
                0 => 0x8A00_0000u32,
                1 => 0xAA00_0000,
                _ => 0xCA00_0000,
            };
            self.emit(bits | (rm << 16) | (rn << 5) | rd);
        }
        /// lslv/lsrv/asrv wd, wn, wm (shift amount = wm mod 32, matching JS) — op: 0=lsl, 1=lsr, 2=asr
        pub fn shift_w(&mut self, op: u32, rd: u32, rn: u32, rm: u32) {
            let bits = match op {
                0 => 0x1AC0_2000u32,
                1 => 0x1AC0_2400,
                _ => 0x1AC0_2800,
            };
            self.emit(bits | (rm << 16) | (rn << 5) | rd);
        }
        /// add xd, xn, xm, lsl #shift
        pub fn add_shifted(&mut self, rd: u32, rn: u32, rm: u32, shift: u32) {
            debug_assert!(shift < 64);
            self.emit(0x8B00_0000 | (rm << 16) | (shift << 10) | (rn << 5) | rd);
        }
        /// cmp xn, xm (SUBS xzr, xn, xm)
        pub fn cmp_reg_x(&mut self, rn: u32, rm: u32) {
            self.emit(0xEB00_001F | (rm << 16) | (rn << 5));
        }
        /// lsr xd, xn, #shift (UBFM xd, xn, #shift, #63)
        pub fn lsr_imm(&mut self, rd: u32, rn: u32, shift: u32) {
            debug_assert!(shift < 64);
            self.emit(0xD340_FC00 | (shift << 16) | (rn << 5) | rd);
        }
        /// UBFX Xd, Xn, #lsb, #width: zero-extend the selected field, without changing NZCV.
        /// Arm Instruction Set Reference Guide 100076_0100, D2.180–181: UBFM with
        /// sf=N=1, immr=lsb, imms=lsb+width-1. Register 31 denotes XZR, not SP.
        pub fn ubfx(&mut self, rd: u32, rn: u32, lsb: u32, width: u32) {
            assert!(rd < 32 && rn < 32);
            assert!(lsb < 64 && width > 0 && width <= 64 - lsb);
            self.emit(0xD340_0000 | (lsb << 16) | ((lsb + width - 1) << 10) | (rn << 5) | rd);
        }
        /// lsrv xd, xn, xm (64-bit logical variable shift).
        pub fn lsr_reg(&mut self, rd: u32, rn: u32, rm: u32) {
            self.emit(0x9AC0_2400 | (rm << 16) | (rn << 5) | rd);
        }
        /// lsl xd, xn, #shift (UBFM alias)
        pub fn lsl_imm(&mut self, rd: u32, rn: u32, shift: u32) {
            debug_assert!(shift < 64);
            let immr = (64 - shift) & 63;
            let imms = 63 - shift;
            self.emit(0xD340_0000 | (immr << 16) | (imms << 10) | (rn << 5) | rd);
        }
        /// mov wd, wm (ORR wd, wzr, wm — zero-extends into the x register)
        pub fn mov_w(&mut self, rd: u32, rm: u32) {
            self.emit(0x2A00_03E0 | (rm << 16) | rd);
        }
        /// cmn wn, #imm12 (ADDS wzr, wn, #imm — `cmn wn, #1` tests for 0xFFFF_FFFF)
        pub fn cmn_imm_w(&mut self, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x3100_001F | (imm << 10) | (rn << 5));
        }
        /// cmn xn, #imm12 (ADDS xzr, xn, #imm — `cmn xn, #1` sets V exactly for xn == i64::MAX)
        pub fn cmn_imm_x(&mut self, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0xB100_001F | (imm << 10) | (rn << 5));
        }
        /// fcmp dn, #0.0
        pub fn fcmp_zero(&mut self, rn: u32) {
            self.emit(0x1E60_2008 | (rn << 5));
        }
        /// sub xd, xn, xm
        pub fn sub_reg(&mut self, rd: u32, rn: u32, rm: u32) {
            self.emit(0xCB00_0000 | (rm << 16) | (rn << 5) | rd);
        }
        /// sxtw xd, wn (SBFM xd, xn, #0, #31)
        pub fn sxtw(&mut self, rd: u32, rn: u32) {
            self.emit(0x9340_7C00 | (rn << 5) | rd);
        }
        /// asr wd, wn, #shift (SBFM wd, wn, #shift, #31)
        pub fn asr_imm_w(&mut self, rd: u32, rn: u32, shift: u32) {
            debug_assert!(shift < 32);
            self.emit(0x1300_7C00 | (shift << 16) | (rn << 5) | rd);
        }
        /// lsr wd, wn, #shift (UBFM wd, wn, #shift, #31)
        pub fn lsr_imm_w(&mut self, rd: u32, rn: u32, shift: u32) {
            debug_assert!(shift < 32);
            self.emit(0x5300_7C00 | (shift << 16) | (rn << 5) | rd);
        }
        /// lsl wd, wn, #shift (UBFM wd, wn, #(32-shift)%32, #(31-shift))
        pub fn lsl_imm_w(&mut self, rd: u32, rn: u32, shift: u32) {
            debug_assert!(shift < 32);
            let immr = (32 - shift) & 31;
            let imms = 31 - shift;
            self.emit(0x5300_0000 | (immr << 16) | (imms << 10) | (rn << 5) | rd);
        }
        /// and/orr/eor wd, wn, #imm (logical immediate; `field` from [`logical_imm_w`]) —
        /// op: 0=and, 1=orr, 2=eor
        pub fn logic_imm_w(&mut self, op: u32, rd: u32, rn: u32, field: u32) {
            let bits = match op {
                0 => 0x1200_0000u32,
                1 => 0x3200_0000,
                _ => 0x5200_0000,
            };
            self.emit(bits | (field << 10) | (rn << 5) | rd);
        }
        /// fmov xd, dn (bit move)
        pub fn fmov_x_d(&mut self, rd: u32, rn: u32) {
            self.emit(0x9E66_0000 | (rn << 5) | rd);
        }
        /// Register-offset element accesses of the typed-array templates: the index register is
        /// scaled by the element size (`lsl #log2(size)`; bytes are unscaled). Checked against
        /// the Arm assembler's output by `typed_element_encodings_match_the_assembler`.
        /// ldr wt, [xn, xm, lsl #2]
        pub fn ldr_w_lsl2(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xB860_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldrsb wt, [xn, xm]
        pub fn ldrsb_reg(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0x38E0_6800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldrh wt, [xn, xm, lsl #1]
        pub fn ldrh_lsl1(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0x7860_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldrsh wt, [xn, xm, lsl #1]
        pub fn ldrsh_lsl1(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0x78E0_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldr st, [xn, xm, lsl #2]
        pub fn ldr_s_lsl2(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xBC60_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// strb wt, [xn, xm]
        pub fn strb_reg(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0x3820_6800 | (rm << 16) | (rn << 5) | rt);
        }
        /// strh wt, [xn, xm, lsl #1]
        pub fn strh_lsl1(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0x7820_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// str wt, [xn, xm, lsl #2]
        pub fn str_w_lsl2(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xB820_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// str st, [xn, xm, lsl #2]
        pub fn str_s_lsl2(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xBC20_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// ldr dt, [xn, xm, lsl #3]
        pub fn ldr_d_lsl3(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xFC60_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// str dt, [xn, xm, lsl #3]
        pub fn str_d_lsl3(&mut self, rt: u32, rn: u32, rm: u32) {
            self.emit(0xFC20_7800 | (rm << 16) | (rn << 5) | rt);
        }
        /// adds wd, wn, #imm12 (sets flags; V on i32 overflow)
        pub fn adds_imm_w(&mut self, rd: u32, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x3100_0000 | (imm << 10) | (rn << 5) | rd);
        }
        /// subs wd, wn, wm (sets flags)
        pub fn subs_reg_w(&mut self, rd: u32, rn: u32, rm: u32) {
            self.emit(0x6B00_0000 | (rm << 16) | (rn << 5) | rd);
        }
        /// subs wd, wn, #imm12 (sets flags; V on i32 overflow)
        pub fn subs_imm_w(&mut self, rd: u32, rn: u32, imm: u32) {
            debug_assert!(imm < 4096);
            self.emit(0x7100_0000 | (imm << 10) | (rn << 5) | rd);
        }

        #[cfg(test)]
        pub fn finish(self) -> Vec<u32> {
            self.finish_with_offsets(&[]).0
        }

        /// Resolve patches and export byte offsets from the final, relaxed label layout.
        /// Recording `here()` before relaxation is unsafe for catch destinations: inserted
        /// veneers move those destinations just as they move ordinary branch labels.
        /// Panics on an unbound label (a compiler bug).
        pub fn finish_with_offsets(mut self, exported: &[usize]) -> (Vec<u32>, Vec<u32>) {
            self.relax_branches(8);
            for (at, label, kind) in std::mem::take(&mut self.patches) {
                let target = self.labels[label].expect("unbound jit label");
                let delta = target as i64 - at as i64; // in instructions
                match kind {
                    PatchKind::B => {
                        assert!(
                            (-(1 << 25)..(1 << 25)).contains(&delta),
                            "JIT imm26 branch out of range: {delta} instructions"
                        );
                        let imm26 = (delta as u32) & 0x03FF_FFFF;
                        self.buf[at] |= imm26;
                    }
                    PatchKind::Cb => {
                        debug_assert!((-(1 << 18)..(1 << 18)).contains(&delta));
                        let imm19 = ((delta as u32) & 0x7FFFF) << 5;
                        self.buf[at] |= imm19;
                    }
                }
            }
            let offsets = exported
                .iter()
                .map(|&label| {
                    let insn = self.labels[label].expect("unbound exported jit label");
                    u32::try_from(insn * 4).expect("JIT label offset exceeds u32")
                })
                .collect();
            if !self.marks.is_empty() {
                let resolved = self
                    .marks
                    .iter()
                    .filter_map(|&(label, name)| Some((self.labels[label]? as u32 * 4, name)))
                    .collect();
                super::LAST_MARKS.with(|cell| *cell.borrow_mut() = resolved);
            }
            (self.buf, offsets)
        }

        /// Resolve widening in immutable instruction coordinates, then copy code once. The old
        /// per-branch Vec::insert plus complete label/patch rescans was quadratic on large
        /// functions. Prefix counts relocate branches and catch labels consistently (ECMA-262
        /// TryStatement Evaluation); the emitted imm19/imm26 encodings are unchanged.
        fn relax_branches(&mut self, exact_batches: usize) {
            let fits = |delta: i64| (-(1 << 18)..(1 << 18)).contains(&delta);
            if !self.patches.iter().any(|&(at, label, kind)| {
                matches!(kind, PatchKind::Cb)
                    && !fits(self.labels[label].expect("unbound jit label") as i64 - at as i64)
            }) {
                return; // Ordinary functions need no extra allocation or code-buffer copy.
            }
            debug_assert!(self.patches.windows(2).all(|p| p[0].0 < p[1].0));
            let count = self.patches.len();
            // Inserting after instruction P moves labels strictly after P, not labels at P.
            // Unbound, unused labels are legal; only referenced/exported labels must be bound.
            // One sweep over instruction positions counts the patches before each position;
            // a binary search per label is several times slower on multi-megabyte bodies,
            // which carry hundreds of thousands of labels and patches.
            let mut patches_before = Vec::with_capacity(self.buf.len() + 1);
            let mut next = 0usize;
            for position in 0..=self.buf.len() {
                while next < count && self.patches[next].0 < position {
                    next += 1;
                }
                patches_before.push(next as u32);
            }
            let cuts: Vec<_> = self
                .labels
                .iter()
                .map(|label| label.map(|at| patches_before[at] as usize))
                .collect();
            debug_assert!(self.labels.iter().zip(&cuts).all(|(label, cut)| {
                *cut == label.map(|at| self.patches.partition_point(|p| p.0 < at))
            }));
            drop(patches_before);
            let mut wide = vec![false; count];
            let mut prefix = vec![0usize; count + 1];
            let rebuild = |prefix: &mut [usize], wide: &[bool]| {
                for (k, &widen) in wide.iter().enumerate() {
                    prefix[k + 1] = prefix[k] + usize::from(widen);
                }
            };
            let mut settled = false;
            for _ in 0..exact_batches {
                let mut changed = false;
                for (k, &(at, label, kind)) in self.patches.iter().enumerate() {
                    if wide[k] || !matches!(kind, PatchKind::Cb) {
                        continue;
                    }
                    let target = self.labels[label].expect("unbound jit label")
                        + prefix[cuts[label].expect("unbound jit label")];
                    if !fits(target as i64 - (at + prefix[k]) as i64) {
                        wide[k] = true;
                        changed = true;
                    }
                }
                rebuild(&mut prefix, &wide);
                if !changed {
                    settled = true;
                    break;
                }
            }
            if !settled {
                // Bound even adversarial one-branch-per-round cascades. Hypothetically widening
                // EVERY conditional gives an upper bound on each branch's absolute distance.
                // Widen any remaining branch whose bound does not fit. All retained short
                // branches then provably fit for the actual subset, without more iterations.
                // Only this rare fallback may choose a larger layout than the least fixed point.
                for (k, &(_, _, kind)) in self.patches.iter().enumerate() {
                    prefix[k + 1] = prefix[k] + usize::from(matches!(kind, PatchKind::Cb));
                }
                for (k, &(at, label, kind)) in self.patches.iter().enumerate() {
                    if !wide[k] && matches!(kind, PatchKind::Cb) {
                        let target = self.labels[label].expect("unbound jit label")
                            + prefix[cuts[label].expect("unbound jit label")];
                        wide[k] = !fits(target as i64 - (at + prefix[k]) as i64);
                    }
                }
                rebuild(&mut prefix, &wide);
            }
            for (label, cut) in self.labels.iter_mut().zip(cuts) {
                if let (Some(at), Some(cut)) = (label, cut) {
                    *at += prefix[cut];
                }
            }
            let mut code = Vec::with_capacity(self.buf.len() + prefix[count]);
            let mut from = 0;
            for (k, (at, _, kind)) in self.patches.iter_mut().enumerate() {
                let old_at = *at;
                code.extend_from_slice(&self.buf[from..=old_at]);
                *at = code.len() - 1;
                if wide[k] {
                    let insn = code[*at];
                    code[*at] = if insn & 0xff00_0000 == 0x5400_0000 {
                        insn ^ 1 // B.cond: invert the low condition bit.
                    } else {
                        insn ^ 0x0100_0000 // CBZ <-> CBNZ, preserving register and width.
                    } | (2 << 5); // The inverted branch skips the following B.
                    code.push(0x1400_0000);
                    *at += 1;
                    *kind = PatchKind::B;
                }
                from = old_at + 1;
            }
            code.extend_from_slice(&self.buf[from..]);
            self.buf = code;
        }
    }

    /// Encode a 32-bit logical immediate for AND/ORR/EOR (immediate form): the 12-bit
    /// `immr:imms` field to OR into the instruction at bit 10 (N is always 0 for the 32-bit
    /// variant). `None` when `v` is not a repeating rotated ones-run (0 and !0 included).
    pub fn logical_imm_w(v: u32) -> Option<u32> {
        if v == 0 || v == u32::MAX {
            return None;
        }
        // Smallest power-of-two period.
        let mut p = 32u32;
        while p > 2 {
            let h = p / 2;
            let mask = (1u64 << h) - 1;
            let mut periodic = true;
            let mut i = h;
            while i < 32 {
                if (v as u64 >> i) & mask != v as u64 & mask {
                    periodic = false;
                    break;
                }
                i += h;
            }
            if !periodic {
                break;
            }
            p = h;
        }
        let emask = if p == 32 { u32::MAX } else { (1u32 << p) - 1 };
        let elem = v & emask;
        let len = elem.count_ones();
        if len == 0 || len == p {
            return None;
        }
        let ones = ((1u64 << len) - 1) as u32;
        // The element must be ones(len) rotated right by immr (within p bits).
        for r in 0..p {
            let ror = if r == 0 {
                ones
            } else {
                ((ones >> r) | (ones << (p - r))) & emask
            };
            if ror == elem {
                let imms = match p {
                    32 => 0x00,
                    16 => 0x20,
                    8 => 0x30,
                    4 => 0x38,
                    _ => 0x3C,
                } | (len - 1);
                return Some((r << 6) | imms);
            }
        }
        None
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn optional_region_reference_inventory_ignores_unused_bailouts() {
            let mut a = super::Asm::new();
            let entry = a.new_label();
            a.bind(entry);
            a.b(entry);
            let checkpoint = a.checkpoint();
            let unused = a.new_label();
            let needed = a.new_label();
            a.b_cond(super::super::C_NE, needed);
            a.b(entry);
            assert_eq!(
                a.referenced_since(checkpoint).collect::<Vec<_>>(),
                [needed, entry]
            );
            assert!(!a.referenced_since(checkpoint).any(|label| label == unused));
            a.bind(needed);
            a.ret();
            // An unreferenced, unbound label must not require a fake stub or exported PC.
            assert_eq!(a.finish().len(), 4);
        }

        #[test]
        fn optional_region_rollback_preserves_baseline_labels_patches_and_bytes() {
            fn baseline(with_discarded_region: bool) -> (Vec<u32>, Vec<u32>) {
                let mut a = super::Asm::new();
                let start = a.new_label();
                let finish = a.new_label();
                a.bind(start);
                a.b(finish); // a preexisting fixup to a still-unbound baseline label
                if with_discarded_region {
                    let before = a.checkpoint();
                    let internal = a.new_label();
                    let unused = a.new_label();
                    a.b_cond(super::super::C_EQ, internal);
                    a.b(start);
                    a.bind(internal);
                    a.b(finish);
                    a.bind(unused);
                    a.mov(7, 8);
                    a.rewind(before);
                    assert_eq!(a.checkpoint(), before);
                }
                a.mov(0, 1);
                a.bind(finish);
                a.b(start);
                a.finish_with_offsets(&[start, finish])
            }
            let expected = baseline(false);
            assert_eq!(
                expected,
                (vec![0x1400_0002, 0xaa01_03e0, 0x17ff_fffe], vec![0, 8])
            );
            assert_eq!(baseline(true), expected);
        }

        /// Brute-force decoder for the 32-bit logical-immediate field (N=0).
        fn decode(field: u32) -> Option<u32> {
            let immr = (field >> 6) & 0x3F;
            let imms = field & 0x3F;
            // Element size from the leading-ones pattern of imms.
            let (p, len) = match imms {
                s if s & 0x20 == 0 => (32u32, (s & 0x1F) + 1),
                s if s & 0x30 == 0x20 => (16, (s & 0x0F) + 1),
                s if s & 0x38 == 0x30 => (8, (s & 0x07) + 1),
                s if s & 0x3C == 0x38 => (4, (s & 0x03) + 1),
                s if s & 0x3E == 0x3C => (2, (s & 0x01) + 1),
                _ => return None,
            };
            if len >= p || immr >= p {
                return None;
            }
            let ones = ((1u64 << len) - 1) as u32;
            let emask = if p == 32 { u32::MAX } else { (1u32 << p) - 1 };
            let elem = if immr == 0 {
                ones
            } else {
                ((ones >> immr) | (ones << (p - immr))) & emask
            };
            let mut v = 0u32;
            let mut i = 0;
            while i < 32 {
                v |= elem << i;
                i += p;
            }
            Some(v)
        }

        #[test]
        fn logical_imm_w_roundtrip() {
            // Every encodable field decodes back to a value that re-encodes to itself.
            let mut seen = std::collections::HashMap::new();
            for field in 0u32..(1 << 12) {
                if let Some(v) = decode(field) {
                    seen.entry(v).or_insert(field);
                }
            }
            for &v in seen.keys() {
                let enc = super::logical_imm_w(v).unwrap_or_else(|| {
                    panic!("0x{v:08x} should be encodable");
                });
                assert_eq!(decode(enc), Some(v), "0x{v:08x} enc {enc:03x}");
            }
            // Common masks used by the emitter.
            for m in [0x3fffu32, 0xfffffff, 0x7fff, 0xff, 1, 0x3fffffff] {
                assert!(super::logical_imm_w(m).is_some(), "0x{m:x}");
            }
            // Non-encodable values.
            for m in [0u32, u32::MAX, 0x12345678, 5] {
                if let Some(enc) = super::logical_imm_w(m) {
                    assert_eq!(decode(enc), Some(m));
                }
            }
            assert!(super::logical_imm_w(0).is_none());
            assert!(super::logical_imm_w(u32::MAX).is_none());
            assert!(super::logical_imm_w(0x12345678).is_none());
        }

        #[test]
        fn far_condition_uses_an_unconditional_veneer() {
            let mut a = super::Asm::new();
            let target = a.new_label();
            a.b_cond(super::super::C_EQ, target);
            // Exceed imm19's positive limit (262,143 instructions). A raw conditional branch
            // would wrap into unrelated generated code; the veneer keeps its conditional local.
            for _ in 0..270_000 {
                a.mov(0, 0);
            }
            a.bind(target);
            let code = a.finish();
            assert_eq!((code[0] >> 5) & 0x7ffff, 2); // inverted condition skips the B
            assert_eq!(code[0] & 0xf, super::super::C_NE);
            assert_eq!(code[1] >> 26, 0b000101); // unconditional B (imm26)
        }

        #[test]
        fn relaxed_branches_relocate_exported_catch_offsets() {
            let mut a = super::Asm::new();
            let start = a.new_label();
            let middle = a.new_label();
            let end = a.new_label();
            a.bind(start);
            a.b_cond(super::super::C_EQ, end);
            a.bind(middle);
            for _ in 0..270_000 {
                a.mov(0, 0);
            }
            a.cbnz(0, true, middle);
            a.bind(end);
            a.ret();

            let (code, offsets) = a.finish_with_offsets(&[start, middle, end, middle]);
            // Both the forward B.cond and backward CBNZ require an inserted instruction.
            // Exporting a label twice must preserve aliases (fused bytecode PCs use these).
            assert_eq!(offsets, [0, 8, 270_004 * 4, 8]);
            assert_eq!(code[offsets[2] as usize / 4], 0xd65f03c0); // RET at catch entry
            assert_eq!(
                1 + (code[1] & 0x03ff_ffff) as usize,
                offsets[2] as usize / 4
            );
            let back = code.len() - 2;
            let delta = ((code[back] << 6) as i32) >> 6;
            assert_eq!(back as i64 + delta as i64, offsets[1] as i64 / 4);
        }

        /// The previous insertion algorithm is retained only as a differential oracle.
        fn reference_finish(mut a: super::Asm, exported: &[usize]) -> (Vec<u32>, Vec<u32>) {
            while let Some(k) = a.patches.iter().position(|&(at, label, kind)| {
                matches!(kind, super::PatchKind::Cb)
                    && !(-(1 << 18)..(1 << 18))
                        .contains(&(a.labels[label].unwrap() as i64 - at as i64))
            }) {
                let (at, label, _) = a.patches[k];
                let insn = a.buf[at];
                a.buf[at] = if insn & 0xff00_0000 == 0x5400_0000 {
                    insn ^ 1
                } else {
                    insn ^ 0x0100_0000
                } | (2 << 5);
                a.buf.insert(at + 1, 0x1400_0000);
                for bound in a.labels.iter_mut().flatten() {
                    if *bound > at {
                        *bound += 1;
                    }
                }
                for (patch_at, _, _) in &mut a.patches {
                    if *patch_at > at {
                        *patch_at += 1;
                    }
                }
                a.patches[k] = (at + 1, label, super::PatchKind::B);
            }
            a.finish_with_offsets(exported)
        }

        fn checked_relaxation(mut a: super::Asm, batches: usize) -> (Vec<u32>, Vec<u32>) {
            let original: Vec<_> = a.patches.iter().map(|p| (a.buf[p.0], p.2)).collect();
            a.relax_branches(batches);
            let patches = a.patches.clone();
            let labels = a.labels.clone();
            let exported: Vec<_> = labels
                .iter()
                .enumerate()
                .filter_map(|(k, p)| p.map(|_| k))
                .collect();
            let result = a.finish_with_offsets(&exported);
            for ((insn, old_kind), &(at, label, kind)) in original.into_iter().zip(&patches) {
                let word = result.0[at];
                let delta = match kind {
                    super::PatchKind::B => ((word << 6) as i32 >> 6) as i64,
                    super::PatchKind::Cb => ((word << 8) as i32 >> 13) as i64,
                };
                assert_eq!(at as i64 + delta, labels[label].unwrap() as i64);
                match (old_kind, kind) {
                    (super::PatchKind::Cb, super::PatchKind::B) => {
                        let inverse = if insn & 0xff00_0000 == 0x5400_0000 {
                            insn ^ 1
                        } else {
                            insn ^ 0x0100_0000
                        };
                        assert_eq!(result.0[at - 1], inverse | (2 << 5));
                        assert_eq!(word & 0xfc00_0000, 0x1400_0000);
                    }
                    (super::PatchKind::B, super::PatchKind::B) => {
                        assert_eq!(word & 0xfc00_0000, insn); // Preserve B versus BL.
                    }
                    (super::PatchKind::Cb, super::PatchKind::Cb) => {
                        assert_eq!(word & !(0x7ffff << 5), insn);
                    }
                    _ => panic!("unconditional branch was narrowed"),
                }
            }
            for (&label, &offset) in exported.iter().zip(&result.1) {
                assert_eq!(offset as usize, labels[label].unwrap() * 4);
            }
            result
        }

        #[test]
        fn batched_relaxation_cascades_at_both_signed_boundaries() {
            for backward in [false, true] {
                let mut a = super::Asm::new();
                let start = a.new_label();
                let middle = a.new_label();
                let end = a.new_label();
                a.bind(start);
                if !backward {
                    a.b_cond(super::super::C_EQ, middle);
                }
                a.cbz(7, true, end); // Widening this pushes the initially fitting branch out.
                a.buf
                    .resize(if backward { 1 << 18 } else { (1 << 18) - 1 }, 0xaa00_03e0);
                a.bind(middle);
                if backward {
                    a.cbnz(9, false, start);
                }
                a.buf.resize((1 << 18) + 100, 0xaa00_03e0);
                a.bind(end);
                a.ret();
                let expected = reference_finish(a.clone(), &[start, middle, end]);
                assert_eq!(checked_relaxation(a.clone(), 8), expected);
                // Force the bounded conservative fallback as well as the ordinary fixed point.
                checked_relaxation(a, 0);
            }
        }

        #[test]
        fn batched_relaxation_matches_reference_and_preserves_every_patch_family() {
            for seed in 0u64..12 {
                let mut a = super::Asm::new();
                let mut random = seed + 1;
                for k in 0..96 {
                    random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                    a.buf.resize(k * 3500, 0xaa00_03e0);
                    let label = a.new_label();
                    a.labels[label] = Some((random as usize % 101) * 3500);
                    match k % 6 {
                        0 => a.b_cond((k / 6 % 14) as u32, label),
                        1 => a.cbz(k as u32 % 32, k % 4 == 1, label),
                        2 => a.cbnz(k as u32 % 32, k % 4 == 2, label),
                        3 => a.b(label),
                        4 => a.bl_label(label),
                        _ => a.b_cond(super::super::C_NE, label),
                    }
                }
                a.buf.resize(360_000, 0xaa00_03e0);
                let labels: Vec<_> = (0..a.labels.len()).collect();
                let expected = reference_finish(a.clone(), &labels);
                assert_eq!(checked_relaxation(a.clone(), 8), expected, "seed {seed}");
                checked_relaxation(a, 0);
            }
        }

        #[test]
        fn batched_relaxation_handles_many_far_branches_and_unused_labels() {
            let mut a = super::Asm::new();
            let end = a.new_label();
            a.new_label(); // Unbound but unused: do not demand an exported destination for it.
            for k in 0..16_000 {
                a.cbz(k % 32, k % 2 == 0, end);
            }
            a.buf.resize(280_000, 0xaa00_03e0);
            a.bind(end);
            a.ret();
            let (words, offsets) = checked_relaxation(a, 8);
            assert_eq!(words.len(), 296_001);
            assert_eq!(offsets, [296_000 * 4]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------------------------

/// Compile `chunk` to machine code, or `None` when unsupported (non-macOS/ARM64, suspending bodies,
/// or an op stream whose stack depths don't line up — a compiler bug caught defensively).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub fn compile(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
) -> Option<JitCode> {
    compile_entry(
        chunk,
        layout,
        ilayout,
        if chunk.jit_is_resumable() {
            NativeEntryKind::BorrowedFrame
        } else {
            NativeEntryKind::FreshFrame
        },
    )
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn compile_entry(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    entry_kind: NativeEntryKind,
) -> Option<JitCode> {
    use crate::bytecode::{Op, UpdKind};

    if !ilayout.valid {
        return None; // Cannot establish the body's lexical mode without a validated layout.
    }
    let ops = chunk.jit_ops();
    if ops.len() > 0xFFFF {
        return None; // op index must fit one movz
    }
    let borrowed_entry = entry_kind == NativeEntryKind::BorrowedFrame;
    if !borrowed_entry && ops.iter().any(|op| matches!(op, Op::FragmentExit(_))) {
        return None; // exact outer completion belongs to the borrowed AST/VM driver
    }
    let return_needs_unwind = ops
        .iter()
        .any(|op| matches!(op, Op::PushFinally(..) | Op::PushIterator(..)));
    let cfg = if borrowed_entry && !chunk.jit_is_resumable() {
        crate::jit_ir::Cfg::build_osr(chunk)
    } else {
        crate::jit_ir::Cfg::build(chunk)
    }
    .ok()?;
    let max_stack = cfg.jit_stack_capacity();
    // Debug: `TRUST_JS_JIT_DUMP=<substr>` prints the op stream of chunks whose leading slot names
    // contain the substring (empty value = all chunks) as they compile.
    if let Some(pat) = env_value!("TRUST_JS_JIT_DUMP") {
        let head: Vec<&str> = chunk
            .jit_slot_names()
            .iter()
            .take(4)
            .map(|s| &**s)
            .collect();
        let name = head.join(",");
        if pat.is_empty() || name.contains(pat) {
            eprintln!("[jit-dump] fn({name}) {} ops", ops.len());
            for (pc, op) in ops.iter().enumerate() {
                eprintln!("[jit-dump]   {pc:>4}  {op:?}");
            }
        }
    }
    let mut fast: u32 = env_value!("TRUST_JS_JIT_FAST")
        .and_then(|v| v.parse().ok())
        .unwrap_or(u32::MAX);
    // A slice borrows a continuation-owned activation. Calls remain native through their
    // ordinary entry, but cannot swap this context out from under the borrowed state view.
    if borrowed_entry {
        fast &= !(1 << 20);
    }
    if chunk.jit_detailed_feedback_enabled() {
        // Detailed arithmetic observation runs in the exact-PC helper. Keep ordinary builds on
        // their inline and register-region paths; profile-enabled chunks route numeric/update
        // templates, fusions, and chains through the helper so successful JIT operations cannot
        // disappear. The broader name/property bits also cover non-update operations, but this
        // diagnostic-only mode favors complete attribution over profiling throughput.
        // Route all element templates through the exact-PC helper as well: GetElem (1024),
        // SetElemDrop (2048), and SetElem (4096) otherwise perform the operation without a
        // semantic trace. Register/region/property fusions are disabled for the same reason.
        // Branch templates and compare/loop fusions are routed through exact-PC observation
        // helpers below so optimized execution cannot disappear from branch/back-edge counts.
        fast &= !(1
            | 2
            | 4
            | 32
            | 1024
            | 2048
            | 4096
            | 8192
            | 16384
            | 32768
            | 65536
            | 524288
            | (1 << 20)
            | (1 << 21));
    }
    let array_intrinsics_on = !env_flag!("TRUST_JS_JIT_NO_ARRAY_INTRINSICS");
    let function_call_intrinsic_on = !env_flag!("TRUST_JS_JIT_NO_FUNCTION_CALL_INTRINSIC");
    // Direct shared-ctx calls, on by default like every other emitter feature (mask bit 20 off
    // for debugging). Requires the inline call probe (bit 524288) to emit at all. The readable
    // kill switch preserves an emergency layered-call path independently of the numeric mask.
    let direct_on = fast & (1 << 20) != 0 && crate::bytecode::direct_shared_context_enabled();
    let mut a = asm::Asm::new();
    a.set_stubs_ok(true);
    let residency = Box::<cache::CodeResidency>::default();
    let residency_ptr = &*residency as *const cache::CodeResidency as u64;
    // One label per bytecode pc (branch/catch targets bind as we emit).
    let pc_labels: Vec<usize> = (0..ops.len()).map(|_| a.new_label()).collect();
    for pc in 0..ops.len() {
        if let Some(plan) = crate::bytecode::string_accumulation::Plan::at(chunk, pc)
            .filter(|plan| plan.end < ops.len())
        {
            a.add_string_accumulation(pc as u32, plan.slot, pc_labels[plan.end]);
        }
    }
    let l_unwind = a.new_label();
    let l_ret_ok = a.new_label();
    let l_ret_throw = a.new_label();
    // The direct-call teardown stub (one per chunk, `bl`-reached; emitted after the epilogues).
    let l_direct_finish = a.new_label();

    // ---- prologue ----
    // Frame: fp/lr + x19..x22 + d8..d15 + saved caller strictness (16-byte alignment).
    const FRAME_SIZE: i32 = 128;
    const DSAVE: i32 = 48;
    const STRICT_SAVE: u32 = 112;
    a.stp_pre(29, 30, -FRAME_SIZE);
    a.stp_off(19, 20, 16);
    a.stp_off(21, 22, 32);
    a.stp_d_off(8, 9, DSAVE);
    a.stp_d_off(10, 11, DSAVE + 16);
    a.stp_d_off(12, 13, DSAVE + 32);
    a.stp_d_off(14, 15, DSAVE + 48);
    a.mov(19, 0); // ctx
    a.ldr_imm(21, 19, 0); // helpers table
    a.ldr_imm(20, 19, if borrowed_entry { 16 } else { 8 }); // live top or empty stack
    a.ldr_imm(22, 19, 24); // local slots base

    // Protect this mapping even when a direct callee re-enters compilation.
    // No helper can run before this mark or after its matching epilogue release.
    emit_residency_mark(&mut a, residency_ptr, true);

    // ECMA-262 Strict Mode Code / PutValue: semantic helpers must see THIS body's lexical
    // mode, including cached calls and direct shared-context calls. Save on the native frame,
    // not JitCtx (which direct callees share), and restore on both normal and throwing exits.
    a.ldr_imm(13, 19, std::mem::offset_of!(JitCtx, interp) as u32);
    let (strict_base, strict_off) = byte_field_address(&mut a, 13, ilayout.strict, 16);
    a.ldrb_imm(9, strict_base, strict_off);
    a.strb_imm(9, 31, STRICT_SAVE);
    if !borrowed_entry {
        a.movz(9, u32::from(chunk.jit_is_strict()), 0);
        a.strb_imm(9, strict_base, strict_off);
    }
    // Function entry is a safepoint, like a loop back-edge: allocation pressure and host
    // interruption are observed here on the shared amortized cadence. Code that never takes a
    // back-edge (loop-free recursion makes billions of calls) must still be interruptible, since
    // a user agent may abort a running script at any time (HTML §8.1.4.5). The frame is fully
    // established and the operand stack empty, so the helper sees every owned root. A direct
    // call `blr`s to the code start: this is its safepoint too.
    if !borrowed_entry {
        emit_interrupt_poll(&mut a, ilayout, POLL_ENTRY, l_unwind);
    }
    if borrowed_entry {
        // The Rust entry validates the bytecode PC and its settled stack before dispatch.
        a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, resume_pc) as u32);
        a.ldr_imm(10, 19, std::mem::offset_of!(JitCtx, pc_offsets) as u32);
        a.lsl_imm(9, 9, 2);
        a.add_shifted(9, 10, 9, 0);
        a.ldr_w_imm(9, 9, 0);
        a.ldr_imm(10, 19, std::mem::offset_of!(JitCtx, code_base) as u32);
        a.add_shifted(9, 10, 9, 0);
        a.br(9);
    }

    // Branch/catch targets: a fused compare+branch may only swallow a following JumpIfFalse if
    // nothing can land on the branch op itself.
    let mut targeted = vec![false; ops.len() + 1];
    let mut interrupt_targets = vec![false; ops.len() + 1];
    for (pc, op) in ops.iter().enumerate() {
        if borrowed_entry && crate::bytecode::jit_slice_exit_op(op) && pc + 1 < ops.len() {
            targeted[pc + 1] = true;
        }
        match op {
            Op::Jump(t)
            | Op::AbruptJump(t, _)
            | Op::JumpIfFalse(t)
            | Op::JumpIfFalsePeek(t)
            | Op::JumpIfTruePeek(t)
            | Op::JumpIfNotNullishPeek(t)
            | Op::InlineGuard(_, t)
            | Op::PushHandler(t) => targeted[*t as usize] = true,
            Op::PushFinally(
                throw_target,
                return_target,
                bare_return_target,
                resume_return_target,
                jump_target,
            ) => {
                targeted[*throw_target as usize] = true;
                targeted[*return_target as usize] = true;
                targeted[*bare_return_target as usize] = true;
                targeted[*resume_return_target as usize] = true;
                targeted[*jump_target as usize] = true;
            }
            Op::PushIterator(
                throw_target,
                return_target,
                bare_return_target,
                resume_return_target,
            ) => {
                targeted[*throw_target as usize] = true;
                targeted[*return_target as usize] = true;
                targeted[*bare_return_target as usize] = true;
                targeted[*resume_return_target as usize] = true;
            }
            _ => {}
        }
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
    // Emit primitive bitwise exits after the body, preserving numeric loop code layout.
    let mut primitive_bit_paths = Vec::new();
    // Whole-loop plans start only at their guarded header. Extra exact bailout labels added
    // by an enclosing region must inhibit baseline fusions, not destroy a stronger nested
    // loop's planning vocabulary (for example a checked, discarded local read).
    let source_targets = targeted.clone();
    // The transaction can enter only at its original read. Keep every ordinary interior
    // template independently resumable, including borrowed frames and exact-PC recovery.
    // Established Number chains get a separate forward copy rather than losing their lowering.
    for pc in 0..ops.len() {
        if let Some(plan) = crate::bytecode::string_accumulation::Plan::at(chunk, pc)
            .filter(|plan| plan.end < ops.len())
        {
            targeted[pc..plan.end].fill(true);
        }
    }
    // ---- op templates ----
    let mut skip = 0usize;
    // Baseline labels remain independently enterable; don't emit another optimized
    // forward copy at each interior block of an already selected region.
    let mut region_covered = vec![false; ops.len()];
    let mut acyclic_budget = crate::jit_ir::AcyclicBudget::new(&cfg, ops.len());
    // All optional general-region copies share one budget, including loop regions.
    // The preserved baseline and established whole-loop lowerings remain independent.
    let mut region_copy_bytes = 0usize;
    let mut stronger_header_cache = vec![None; ops.len()];
    let in_loop = loop_body_mask(ops);
    // The body's baseline templates end each call to an allocating helper in the
    // explicit poll; region emission clears this for its own code.
    a.set_end_polls(true);
    for (pc, op) in ops.iter().enumerate() {
        a.bind(pc_labels[pc]);
        a.set_hot(in_loop[pc]);
        if interrupt_targets[pc] {
            emit_interrupt_poll(&mut a, ilayout, POLL_LOOP, l_unwind);
        }
        if skip > 0 {
            // Consumed by a fusion (chain / compare+branch / key-producer pair). The label and
            // pc-offset still bind here (harmless: nothing jumps into a fused region — checked).
            skip -= 1;
            continue;
        }
        if borrowed_entry && crate::bytecode::jit_slice_exit_op(op) {
            emit_op_helper(&mut a, H_SLICE_OP, pc as u32, l_ret_throw);
            a.b(l_ret_ok);
            continue;
        }
        emit_string_accumulation(&mut a, pc as u32, false, l_unwind);
        // Loop-spanning chain: a fully-chainable, branch-free loop headed here runs with its
        // locals register-resident across the back edge. The plain templates for the region are
        // still emitted below (starting at `plain_h`) as the bail target; the head's canonical
        // label points at the chain entry, so plain back-edge jumps re-enter the chain.
        // Policy: multi-block native loop lowerings must be justified by production traces and
        // accepted by the general CFG/SSA region builder. Do not key a lowering to benchmark
        // function names, class layouts, or fixture-specific object graphs.
        if fast & 32768 != 0 && layout.valid && targeted[pc] {
            if env_flag!("TRUST_JS_JIT_REGIONLOG") {
                match crate::jit_ir::RegionIr::build_loop(chunk, &cfg, pc) {
                    Ok(region) => {
                        let op_count: usize = region.blocks.iter().map(|b| b.insts.len()).sum();
                        let phi_count = region
                            .values
                            .iter()
                            .filter(|v| matches!(v.def, crate::jit_ir::ValueDef::BlockParam { .. }))
                            .count();
                        eprintln!(
                            "[jit-region] head {pc}: {} blocks, {op_count} ops, {} values, {phi_count} params, {} exits",
                            region.blocks.len(),
                            region.values.len(),
                            region.exits.len()
                        );
                    }
                    // Most targeted bytecodes are ordinary jump destinations rather than loop
                    // headers.  `NoLoop` is therefore expected and would drown out actionable
                    // region diagnostics on large functions.
                    Err(crate::jit_ir::IrError::NoLoop) => {}
                    Err(err) => eprintln!("[jit-region] head {pc}: reject {err:?}"),
                }
            }
            let mut emitted_region = false;
            if let Some(plan) = plan_linked_scan(chunk, ops, pc, &cfg, layout, fast) {
                let plain_h = emit_linked_scan_region(&mut a, layout, ilayout, &plan, &pc_labels);
                a.bind(plain_h);
                for p in pc + 1..pc + 9 {
                    targeted[p] = true;
                }
                emitted_region = true;
                if env_flag!("TRUST_JS_JIT_REGIONLOG") {
                    eprintln!("[jit-region] head {pc}: EMITTED linked scan");
                }
            }
            if !emitted_region {
                // Prefer the fully helper-free lowering when it covers the entire loop.
                // General CFG lowering adds effect/call coverage, not a reason to spill
                // already-proven numeric and element loops at every operation.
                if let Some(plan) = plan_loop(chunk, ops, pc, &source_targets, layout, fast, &cfg) {
                    let plain_h =
                        emit_loop_chain(&mut a, layout, ilayout, &plan, &pc_labels, &mut targeted);
                    a.bind(plain_h);
                } else if let Some(plan) = regions::Plan::build(chunk, &cfg, pc) {
                    for &(start, end) in plan.ranges() {
                        region_covered[start..end].fill(true);
                    }
                    let checkpoint = a.checkpoint();
                    let emission = regions::emit(
                        &mut a,
                        chunk,
                        &cfg,
                        layout,
                        ilayout,
                        &plan,
                        &pc_labels,
                        &source_targets,
                        &mut stronger_header_cache,
                        fast,
                        array_intrinsics_on,
                        function_call_intrinsic_on,
                        l_unwind,
                        l_direct_finish,
                        entry_kind,
                    );
                    if emission.profitable() && emission.bytes <= 128 * 1024 - region_copy_bytes {
                        region_copy_bytes += emission.bytes;
                        emission.publish(&mut targeted);
                        a.bind(emission.plain);
                        if env_flag!("TRUST_JS_JIT_REGIONLOG") {
                            eprintln!(
                                "[jit-region] head {pc}: EMITTED general CFG ({} bytes, {} saved-work units)",
                                emission.bytes, emission.saved_work
                            );
                        }
                    } else {
                        a.rewind(checkpoint);
                        if env_flag!("TRUST_JS_JIT_REGIONLOG") {
                            eprintln!(
                                "[jit-region] head {pc}: retained baseline ({} optional bytes, {} saved-work units)",
                                emission.bytes, emission.saved_work
                            );
                        }
                    }
                }
            }
        }
        if fast & 32768 != 0
            && layout.valid
            && !region_covered[pc]
            && region_copy_bytes < 128 * 1024
        {
            if let Some(plan) = regions::Plan::build_acyclic(chunk, &cfg, pc, &mut acyclic_budget) {
                for &(start, end) in plan.ranges() {
                    region_covered[start..end].fill(true);
                }
                let checkpoint = a.checkpoint();
                let emission = regions::emit(
                    &mut a,
                    chunk,
                    &cfg,
                    layout,
                    ilayout,
                    &plan,
                    &pc_labels,
                    &source_targets,
                    &mut stronger_header_cache,
                    fast,
                    array_intrinsics_on,
                    function_call_intrinsic_on,
                    l_unwind,
                    l_direct_finish,
                    entry_kind,
                );
                let added = emission.bytes;
                if !emission.profitable() || added > 128 * 1024 - region_copy_bytes {
                    // Code, fixups and private labels roll back together. Crucially, no
                    // speculative bailout destinations have been published: the mature
                    // baseline retains every independently legal span fusion.
                    a.rewind(checkpoint);
                    if env_flag!("TRUST_JS_JIT_REGIONLOG") {
                        eprintln!(
                            "[jit-region] head {pc}: retained baseline ({added} optional bytes, {} saved-work units)",
                            emission.saved_work
                        );
                    }
                } else {
                    region_copy_bytes += added;
                    emission.publish(&mut targeted);
                    a.bind(emission.plain);
                    if env_flag!("TRUST_JS_JIT_REGIONLOG") {
                        eprintln!(
                            "[jit-region] head {pc}: EMITTED acyclic CFG ({added} bytes, total {region_copy_bytes})"
                        );
                    }
                }
            }
        }
        // Local identity/nullish comparison feeding a branch. Reading the two frame slots in
        // place avoids two Value copies, their stack traffic and the equality helper for hot
        // `if (object != excluded)` loops. Unsupported/coercing pairs replay all three value ops
        // plus the condition from untouched state.
        if fast & 2 != 0 && eq_inlinable(layout) && pc + 3 < ops.len() {
            if let (
                Op::LoadLocal(lhs),
                Some(Op::LoadLocal(rhs)),
                Some(cmp @ (Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq)),
                Some(Op::JumpIfFalse(target)),
            ) = (op, ops.get(pc + 1), ops.get(pc + 2), ops.get(pc + 3))
            {
                let lhs_off = *lhs as u32 * 8;
                let rhs_off = *rhs as u32 * 8;
                if !targeted[pc + 1]
                    && !targeted[pc + 2]
                    && !targeted[pc + 3]
                    && lhs_off + 8 < 4096
                    && rhs_off + 8 < 4096
                {
                    emit_local_eq_branch(
                        &mut a,
                        layout,
                        lhs_off,
                        rhs_off,
                        pc as u32,
                        l_unwind,
                        matches!(cmp, Op::StrictEq | Op::StrictNotEq),
                        matches!(cmp, Op::NotEq | Op::StrictNotEq),
                        pc_labels[*target as usize],
                    );
                    skip = 3;
                    continue;
                }
            }
        }
        // Numeric register chain: a run of ops whose values stay in FP registers end to end.
        if fast & 16384 != 0 && layout.valid {
            let accumulation = a.string_accumulation(pc as u32).is_some();
            let chain_targets = if accumulation {
                &source_targets
            } else {
                &targeted
            };
            if let Some((chain, consumed)) =
                build_chain(chunk, ops, pc, chain_targets, layout, fast)
            {
                emit_chain(&mut a, layout, &chain, &pc_labels, l_unwind);
                if accumulation && pc + consumed < ops.len() {
                    a.b(pc_labels[pc + consumed]);
                } else {
                    skip = consumed - 1;
                }
                continue;
            }
        }
        // Jump-threaded equality condition: short-circuit lowering can produce
        // `Eq; Jump(shared-cond)` where the destination is a JumpIfFalse. Drive both outcomes
        // directly from the equality template, bypassing the temporary Bool, the forwarding
        // jump, and its immediate pop. Other predecessors still enter the shared condition's
        // ordinary template. The skipped forwarding Jump must have no incoming edge of its own.
        if fast & 2 != 0 && eq_inlinable(layout) && !targeted[pc + 1] {
            if let (
                Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq,
                Some(Op::Jump(cond_pc)),
            ) = (op, ops.get(pc + 1))
            {
                let cond_pc = *cond_pc as usize;
                if let Some(Op::JumpIfFalse(false_pc)) = ops.get(cond_pc) {
                    emit_eq_inline(
                        &mut a,
                        layout,
                        pc as u32,
                        l_unwind,
                        matches!(op, Op::StrictEq | Op::StrictNotEq),
                        matches!(op, Op::NotEq | Op::StrictNotEq),
                        Some(pc_labels[*false_pc as usize]),
                    );
                    a.b(pc_labels[cond_pc + 1]);
                    skip = 1;
                    continue;
                }
            }
        }
        // Fused equality + JumpIfFalse: the full inline equality template drives the branch
        // directly — numbers, nullish, identity, Bool payloads, string length — no intermediate
        // bool. (The ordered relations below keep their number-only fusion: any other operand
        // type coerces, which is the helper's job.)
        if fast & 2 != 0 && eq_inlinable(layout) {
            if let (
                Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq,
                Some(Op::JumpIfFalse(t)),
            ) = (op, ops.get(pc + 1))
            {
                if !targeted[pc + 1] {
                    emit_eq_inline(
                        &mut a,
                        layout,
                        pc as u32,
                        l_unwind,
                        matches!(op, Op::StrictEq | Op::StrictNotEq),
                        matches!(op, Op::NotEq | Op::StrictNotEq),
                        Some(pc_labels[*t as usize]),
                    );
                    skip = 1;
                    continue;
                }
            }
        }
        // Fused number-compare + JumpIfFalse: fcmp and branch directly on the negated condition
        // (IEEE unordered must jump for the ordered relations and for ==; must fall through for
        // !=) — the intermediate bool never materializes. Types other than two numbers take the
        // unfused pair via the helpers.
        if fast & 2 != 0 {
            if let (
                Op::Lt
                | Op::Gt
                | Op::Le
                | Op::Ge
                | Op::StrictEq
                | Op::StrictNotEq
                | Op::EqEq
                | Op::NotEq,
                Some(Op::JumpIfFalse(t)),
            ) = (op, ops.get(pc + 1))
            {
                if !targeted[pc + 1] {
                    let neg = match op {
                        Op::Lt => 5,                  // PL: !(a<b), true for unordered (NaN must jump)
                        Op::Gt => 13,                 // LE: !(a>b), true for unordered
                        Op::Le => 8,                  // HI: !(a<=b), true for unordered
                        Op::Ge => 11,                 // LT: !(a>=b), true for unordered
                        Op::StrictEq | Op::EqEq => 1, // NE: !(a==b), true for unordered
                        _ => 0, // EQ: !(a!=b); unordered IS "!=" → correctly no jump
                    };
                    let slow = a.new_label();
                    let done = a.new_label();
                    emit_exec_word_load(&mut a, 9, 20, -16);
                    emit_exec_word_load(&mut a, 10, 20, -8);
                    emit_exec_number_guard(&mut a, 9, 0, 11, slow);
                    emit_exec_number_guard(&mut a, 10, 1, 11, slow);
                    a.sub_imm(20, 20, 16); // pop both compact operands
                    a.fcmp(0, 1);
                    a.b_cond(neg, pc_labels[*t as usize]);
                    a.b(done);
                    a.bind(slow);
                    // Unfused fallback: generic compare (pushes a bool), then pop-and-branch.
                    emit_exec(&mut a, pc as u32, l_unwind);
                    emit_cond(&mut a, COND_POP_TRUTHY, l_unwind);
                    a.cbz(1, false, pc_labels[*t as usize]);
                    a.bind(done);
                    skip = 1;
                    continue;
                }
            }
        }
        // Fused key-producer + element read: `x0[cur]` (LoadLocal;GetElemLocal) and `x[++cur]`
        // (UpdateLocal-pre;GetElemLocal) skip the key's stack round-trip entirely. All guards run
        // before any state is written (the pre-increment commits with the element copy), so the
        // slow path can re-run both ops through the helper cleanly.
        if fast & 1024 != 0 && get_elem_inlinable(layout) && !targeted[pc + 1] {
            let in_range = |s: u16| (s as u32) * 8 + 8 < 4096;
            let pair = match (op, ops.get(pc + 1)) {
                (Op::LoadLocal(k), Some(Op::GetElemLocal(x))) if in_range(*k) && in_range(*x) => {
                    Some((*x as u32 * 8, KeySrc::Slot(*k as u32 * 8)))
                }
                (
                    Op::UpdateLocal(k, kind @ (UpdKind::PreInc | UpdKind::PreDec)),
                    Some(Op::GetElemLocal(x)),
                ) if in_range(*k) && in_range(*x) => Some((
                    *x as u32 * 8,
                    KeySrc::SlotPre(*k as u32 * 8, matches!(kind, UpdKind::PreDec)),
                )),
                _ => None,
            };
            if let Some((x_off, key)) = pair {
                emit_elem_local_keyed(
                    &mut a,
                    layout,
                    x_off,
                    &[pc as u32, pc as u32 + 1],
                    l_unwind,
                    ElemLocalKind::Get,
                    key,
                    None,
                    0,
                    None,
                );
                skip = 1;
                continue;
            }
        }
        // Store-and-reload and assignment-result pairs keep one owned copy on the
        // operand stack. Fuse them without assuming a numeric value or reloading the
        // slot, while retaining the ordinary fallback for BigInt and unusual layouts.
        if fast & (8 | 16 | 64) == (8 | 16 | 64) {
            if let Some(slot) = local_store_pair(ops, pc, &targeted) {
                emit_store_local(
                    &mut a,
                    slot as u32 * 8,
                    &[pc as u32, pc as u32 + 1],
                    l_unwind,
                    true,
                    layout.valid,
                );
                skip = 1;
                continue;
            }
        }
        // An unused local read still checks the TDZ, but copies nothing: no word owns
        // anything, and the slot keeps its word where the collector traces it. In particular
        // this is the guard before a plain lexical assignment. Keep independent Pop entries
        // intact.
        if fast & (8 | 64) == (8 | 64) {
            if let Some(slot) = local_read_discard_pair(ops, pc, &targeted) {
                emit_discard_local_read(&mut a, slot as u32 * 8, pc as u32, l_unwind);
                skip = 1;
                continue;
            }
        }
        if operations::emit(
            &mut a,
            chunk,
            layout,
            ilayout,
            pc,
            fast,
            array_intrinsics_on,
            function_call_intrinsic_on,
            l_unwind,
            l_direct_finish,
        ) {
            continue;
        }
        match op {
            Op::Jump(t) => {
                if chunk.jit_detailed_feedback_enabled() && (*t as usize) <= pc {
                    emit_loop_backedge(&mut a, pc as u32);
                }
                a.b(pc_labels[*t as usize]);
            }
            Op::AbruptJump(..) | Op::ResumeReturn | Op::ResumeJump => {
                emit_completion(&mut a, pc as u32, l_ret_ok);
            }
            Op::JumpIfFalse(t) if chunk.jit_detailed_feedback_enabled() => {
                emit_cond_profiled(&mut a, COND_POP_TRUTHY, pc as u32, false, l_unwind);
                a.cbz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfFalse(t) if fast & 4 != 0 => {
                // Packed Bool on top (the compare fast paths produce one).
                let slow = a.new_label();
                let done = a.new_label();
                emit_exec_word_load(&mut a, 9, 20, -8);
                emit_exec_tag_guard(&mut a, 9, crate::value::PACK_BOOL, 10, slow);
                a.movz(10, 1, 0);
                a.logic_x(0, 9, 9, 10);
                a.sub_imm(20, 20, 8);
                a.cbz(9, false, pc_labels[*t as usize]);
                a.b(done);
                a.bind(slow);
                emit_cond(&mut a, COND_POP_TRUTHY, l_unwind);
                a.cbz(1, false, pc_labels[*t as usize]);
                a.bind(done);
            }
            Op::JumpIfFalse(t) => {
                emit_cond(&mut a, COND_POP_TRUTHY, l_unwind);
                a.cbz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfFalsePeek(t) if chunk.jit_detailed_feedback_enabled() => {
                emit_cond_profiled(&mut a, COND_PEEK_TRUTHY, pc as u32, false, l_unwind);
                a.cbz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfFalsePeek(t) => {
                emit_peek_cond_inline(&mut a, layout, false, l_unwind);
                a.cbz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfTruePeek(t) if chunk.jit_detailed_feedback_enabled() => {
                emit_cond_profiled(&mut a, COND_PEEK_TRUTHY, pc as u32, true, l_unwind);
                a.cbnz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfTruePeek(t) => {
                emit_peek_cond_inline(&mut a, layout, false, l_unwind);
                a.cbnz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfNotNullishPeek(t) if chunk.jit_detailed_feedback_enabled() => {
                emit_cond_profiled(&mut a, 2, pc as u32, true, l_unwind);
                a.cbnz(1, false, pc_labels[*t as usize]);
            }
            Op::JumpIfNotNullishPeek(t) => {
                emit_peek_cond_inline(&mut a, layout, true, l_unwind);
                a.cbnz(1, false, pc_labels[*t as usize]);
            }
            Op::Return => {
                if return_needs_unwind {
                    emit_completion(&mut a, pc as u32, l_ret_ok);
                } else {
                    emit_return(&mut a, 1, l_ret_ok);
                }
            }
            Op::ReturnBare => {
                if return_needs_unwind {
                    emit_completion(&mut a, pc as u32, l_ret_ok);
                } else {
                    emit_return(&mut a, 0, l_ret_ok);
                }
            }
            Op::ReturnUndef => {
                emit_return(&mut a, 0, l_ret_ok);
            }
            Op::PushHandler(..) | Op::PushFinally(..) | Op::PushIterator(..) => {
                emit_helper(&mut a, H_PUSH_HANDLER, pc as u32);
            }
            Op::PushDisposeFrame | Op::AddDisposable(_) => {
                emit_exec(&mut a, pc as u32, l_unwind);
            }
            Op::DisposeNormal
            | Op::DisposeThrow
            | Op::DisposeReturn
            | Op::DisposeBareReturn
            | Op::DisposeResumeReturn
            | Op::DisposeJump => {
                unreachable!("resource-disposal ops are rejected before JIT emission")
            }
            Op::PushWith | Op::PopEnv => {
                emit_exec(&mut a, pc as u32, l_unwind);
            }
            Op::PopHandler => {
                emit_helper(&mut a, H_POP_HANDLER, 0);
            }
            Op::Throw => {
                // The generic executor sets ctx.error and returns null.
                emit_exec(&mut a, pc as u32, l_unwind);
            }
            Op::Await | Op::Yield | Op::YieldStar => {
                unreachable!("suspending chunks are rejected above")
            }
            // Int32 ops on primitive numbers/booleans: ToInt32 = truncate + wrap to 32 bits.
            // ECMA-262 ToNumeric/ToNumber maps primitive false/true to 0/1 without user code.
            // Objects, strings, Symbols and BigInts still take the ordered coercion helper.
            // fcvtzs to x
            // truncates; taking the low 32 bits is the mod-2^32 wrap. The scvtf/frintz
            // round-trip proves no i64 saturation happened (NaN/±Inf/|x|≥2^63 all fail it and
            // take the helper, which applies the spec's zero/wrap semantics).
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr if fast & 1 != 0 => {
                let slow = a.new_label();
                let done = a.new_label();
                let primitives = a.new_label();
                let calculate = a.new_label();
                emit_exec_word_load(&mut a, 9, 20, -16);
                emit_exec_word_load(&mut a, 10, 20, -8);
                emit_exec_number_guard(&mut a, 9, 0, 11, primitives);
                emit_exec_number_guard(&mut a, 10, 1, 11, primitives);
                if jscvt_available() {
                    // Exact ToInt32 of every Number, so no operand needs the helper.
                    a.fjcvtzs_w_d(9, 0);
                    a.fjcvtzs_w_d(10, 1);
                } else {
                    a.fcvtzs_x_d(9, 0);
                    a.scvtf_d_x(2, 9);
                    a.frintz(3, 0);
                    a.fcmp(2, 3);
                    a.b_cond(C_NE, slow);
                    // x == +2^63 exactly saturates yet passes the round-trip (2^63-1 re-rounds
                    // to 2^63): cmn #1 sets V only for i64::MAX — send it to the helper.
                    a.cmn_imm_x(9, 1);
                    a.b_cond(6, slow); // VS
                    a.fcvtzs_x_d(10, 1);
                    a.scvtf_d_x(2, 10);
                    a.frintz(3, 1);
                    a.fcmp(2, 3);
                    a.b_cond(C_NE, slow);
                    a.cmn_imm_x(10, 1);
                    a.b_cond(6, slow); // VS
                }
                a.bind(calculate);
                match op {
                    Op::BitAnd => a.logic_w(0, 11, 9, 10),
                    Op::BitOr => a.logic_w(1, 11, 9, 10),
                    Op::BitXor => a.logic_w(2, 11, 9, 10),
                    Op::Shl => a.shift_w(0, 11, 9, 10),
                    Op::UShr => a.shift_w(1, 11, 9, 10),
                    _ => a.shift_w(2, 11, 9, 10), // Shr
                }
                if matches!(op, Op::UShr) {
                    a.ucvtf_d_w(0, 11); // >>> yields an unsigned 32-bit result
                } else {
                    a.scvtf_d_w(0, 11);
                }
                a.stur_d(0, 20, -16); // finite Int32/Uint32: already canonical
                a.sub_imm(20, 20, 8);
                a.b(done);
                primitive_bit_paths.push((primitives, calculate, slow));
                a.bind(slow);
                emit_exec(&mut a, pc as u32, l_unwind);
                a.bind(done);
            }
            // Speculative-inline guard: the callee (argc+1 deep) must be the pinned function —
            // a tag compare and a pointer compare; mismatch branches to the generic call.
            // Shallow tail positions run their ordinary call (see `Op::TailDeep`): compare the
            // live execution-context depth and, when the following JumpIfFalse cannot be entered
            // on its own, branch straight to the ordinary form.
            Op::TailDeep
                if ilayout.depth.is_multiple_of(4)
                    && ilayout.depth / 4 < 4096
                    && !chunk.jit_detailed_feedback_enabled() =>
            {
                let limit = crate::bytecode::ordinary_tail_call_depth();
                a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, interp) as u32);
                a.ldr_w_imm(9, 9, ilayout.depth as u32);
                a.mov_imm64(10, u64::from(limit));
                a.cmp_reg_w(9, 10);
                match ops.get(pc + 1) {
                    Some(Op::JumpIfFalse(t)) if !targeted[pc + 1] => {
                        a.b_cond(C_LO, pc_labels[*t as usize]);
                        skip = 1;
                    }
                    _ => {
                        a.cset_w(9, C_HS);
                        a.mov_imm64(10, crate::value::PACK_BOOL);
                        a.logic_x(1, 9, 9, 10); // orr: tag the 0/1 payload
                        emit_exec_word_store(&mut a, 9, 20, 0);
                        a.add_imm(20, 20, 8);
                    }
                }
            }
            Op::InlineGuard(t, target) => {
                let it = chunk.jit_inline_target(*t);
                // A Value::Obj payload holds the stored handle word, the cell's address: read
                // the expected stored word out of an Option<Gc> exactly like `value::jit_layout`
                // probes it. A dead callee (or an unprobed layout) degrades to the generic call
                // unconditionally.
                let stored = it.pin.upgrade().filter(|_| layout.valid).map(|o| {
                    let some: Option<crate::value::Gc> = Some(o);
                    unsafe { *(&some as *const Option<crate::value::Gc> as *const usize) }
                });
                match stored {
                    None => a.b(pc_labels[*target as usize]),
                    Some(s) => {
                        if it.expected_env != 0 {
                            a.ldr_imm(11, 19, 40); // ctx.env_raw
                            a.mov_imm64(12, it.expected_env as u64);
                            a.cmp_reg_x(11, 12);
                            a.b_cond(C_NE, pc_labels[*target as usize]);
                        }
                        // Same-Realm proof: the splice runs in the planning Realm only.
                        a.ldr_imm(11, 19, std::mem::offset_of!(JitCtx, genv) as u32);
                        a.mov_imm64(12, it.expected_genv as u64);
                        a.cmp_reg_x(11, 12);
                        a.b_cond(C_NE, pc_labels[*target as usize]);
                        let dm = (it.argc as i32 + 1) * 8;
                        // Inline plans bound arity; a far-negative address is materialized
                        // once rather than truncating an unscaled load displacement.
                        a.mov_imm64(11, dm as u64);
                        a.sub_reg(11, 20, 11);
                        emit_exec_word_load(&mut a, 9, 11, 0);
                        a.mov_imm64(10, crate::value::PACK_OBJ | s as u64);
                        a.cmp_reg_x(9, 10);
                        a.b_cond(C_NE, pc_labels[*target as usize]);
                        if it.check_this {
                            emit_exec_word_load(&mut a, 9, 11, -8);
                            emit_exec_tag_guard(
                                &mut a,
                                9,
                                crate::value::PACK_OBJ,
                                10,
                                pc_labels[*target as usize],
                            );
                        }
                    }
                }
            }
            _ => {
                emit_exec(&mut a, pc as u32, l_unwind);
            }
        }
    }
    a.set_hot(false);
    a.set_end_polls(false);
    // Fall off the end: return undefined (compile() always terminates with ReturnUndef, but be
    // safe about it).
    if borrowed_entry {
        emit_helper(&mut a, H_SLICE_OP, ops.len() as u32);
        a.b(l_ret_ok);
    } else {
        emit_return(&mut a, 0, l_ret_ok);
    }

    for (primitives, calculate, slow) in primitive_bit_paths {
        a.bind(primitives);
        for (offset, reg) in [(-16, 9), (-8, 10)] {
            let boolean = a.new_label();
            let converted = a.new_label();
            emit_exec_word_load(&mut a, 11, 20, offset);
            emit_exec_number_guard(&mut a, 11, 0, 12, boolean);
            a.fcvtzs_x_d(reg, 0);
            a.scvtf_d_x(2, reg);
            a.frintz(3, 0);
            a.fcmp(2, 3);
            a.b_cond(C_NE, slow);
            // +2^63 saturates yet passes the round-trip (MAX re-rounds to +2^63).
            a.cmn_imm_x(reg, 1);
            a.b_cond(C_VS, slow);
            a.b(converted);
            a.bind(boolean);
            emit_exec_tag_guard(&mut a, 11, crate::value::PACK_BOOL, 12, slow);
            a.movz(12, 1, 0);
            a.logic_x(0, reg, 11, 12);
            a.bind(converted);
        }
        a.b(calculate);
    }

    // ---- unwind: route a throw to the innermost try handler, or out ----
    a.bind(l_unwind);
    if borrowed_entry {
        a.b(l_ret_throw);
    }
    a.mov(0, 19);
    a.movz(1, 0, 0);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_UNWIND * 8) as u32);
    a.blr(16);
    a.cbz(0, true, l_ret_throw);
    a.mov(20, 1);
    a.br(0);

    // ---- epilogues ----
    a.bind(l_ret_ok);
    a.str_imm(20, 19, 16); // ctx.final_sp = sp
    a.ldr_imm(13, 19, std::mem::offset_of!(JitCtx, interp) as u32);
    let (strict_base, strict_off) = byte_field_address(&mut a, 13, ilayout.strict, 16);
    a.ldrb_imm(9, 31, STRICT_SAVE);
    a.strb_imm(9, strict_base, strict_off);
    emit_residency_mark(&mut a, residency_ptr, false);
    a.movz(0, 1, 0);
    a.ldp_d_off(8, 9, DSAVE);
    a.ldp_d_off(10, 11, DSAVE + 16);
    a.ldp_d_off(12, 13, DSAVE + 32);
    a.ldp_d_off(14, 15, DSAVE + 48);
    a.ldp_off(21, 22, 32);
    a.ldp_off(19, 20, 16);
    a.ldp_post(29, 30, FRAME_SIZE);
    a.ret();
    a.bind(l_ret_throw);
    a.str_imm(20, 19, 16);
    a.ldr_imm(13, 19, std::mem::offset_of!(JitCtx, interp) as u32);
    let (strict_base, strict_off) = byte_field_address(&mut a, 13, ilayout.strict, 16);
    a.ldrb_imm(9, 31, STRICT_SAVE);
    a.strb_imm(9, strict_base, strict_off);
    emit_residency_mark(&mut a, residency_ptr, false);
    a.movz(0, 0, 0);
    a.ldp_d_off(8, 9, DSAVE);
    a.ldp_d_off(10, 11, DSAVE + 16);
    a.ldp_d_off(12, 13, DSAVE + 32);
    a.ldp_d_off(14, 15, DSAVE + 48);
    a.ldp_off(21, 22, 32);
    a.ldp_off(19, 20, 16);
    a.ldp_post(29, 30, FRAME_SIZE);
    a.ret();

    // ---- shared out-of-line stubs requested by this chunk's sites ----
    a.set_hot(false);
    emit_shared_stubs(
        &mut a,
        &StubContext {
            layout,
            ilayout: Some(ilayout),
            direct: direct_on.then(|| DirectCallContext {
                attempted_off: chunk.jit_inline_attempted_off(),
                runs_off: chunk.jit_runs_off(),
                retry_off: chunk.jit_inline_retry_at_off(),
                finish_stub: l_direct_finish,
            }),
        },
    );
    // ---- direct-call teardown stub (only reachable from emitted direct sequences) ----
    // In-line direct sequences and per-chunk direct-call stubs reach it; process-wide stubs
    // carry their own, so a chunk with neither omits it.
    a.bind(l_direct_finish);
    if direct_on && a.label_referenced(l_direct_finish) {
        emit_direct_finish_stub(&mut a, ilayout, layout.valid);
    }

    // The unwinder (ECMA-262 14.15.3) needs the same final catch addresses as patched
    // branches, including every word inserted while relaxing long conditional branches.
    let (words, pc_offsets) = a.finish_with_offsets(&pc_labels[..ops.len()]);
    // Debug: `TRUST_JS_JIT_CODEDUMP=<substr>` prints the finished code words (hex, one per line)
    // of chunks whose leading slot names contain the substring (`*`: every chunk) — round-trip
    // them through `clang -c` + `objdump -d` for a disassembly of exactly what runs. `names@N`
    // selects the function that starts at source offset N. Any value also prints
    // a `[jit-map]` line per compiled chunk (runtime base + length), which joins a `sample`
    // profile's raw addresses to chunk-relative offsets.
    let codedump_pat = env_value!("TRUST_JS_JIT_CODEDUMP");
    if let Some(pat) = codedump_pat {
        let head: Vec<&str> = chunk
            .jit_slot_names()
            .iter()
            .take(4)
            .map(|s| &**s)
            .collect();
        let name = head.join(",");
        // `names@origin` also requires the function's source start offset (the `@N` that
        // `TRUST_JS_JIT_MAP` prints), which singles out one function among many sharing names.
        let (names_pat, origin) = match pat.split_once('@') {
            Some((names, origin)) => (names, origin.parse::<u32>().ok()),
            None => (pat, None),
        };
        if (names_pat == "*" || !names_pat.is_empty() && name.contains(names_pat))
            && origin.is_none_or(|origin| chunk.jit_debug_origin().0 == origin)
        {
            eprintln!("[jit-codedump] fn({name}) {} words", words.len());
            for w in &words {
                eprintln!("[jit-codedump] {w:08x}");
            }
        }
    }
    let len = words.len() * 4;
    let executable = ExecutableBuffer::from_bytes(unsafe {
        std::slice::from_raw_parts(words.as_ptr() as *const u8, len)
    })?;
    let mem = executable.as_ptr() as *mut u8;
    let len = executable.len();
    {
        if codedump_pat.is_some() {
            let head: Vec<&str> = chunk
                .jit_slot_names()
                .iter()
                .take(4)
                .map(|s| &**s)
                .collect();
            eprintln!(
                "[jit-map] fn({}) base={:#x} len={len}",
                head.join(","),
                mem as usize
            );
        }
        if env_flag!("TRUST_JS_JIT_MAP") {
            let head: Vec<&str> = chunk
                .jit_slot_names()
                .iter()
                .take(4)
                .map(|s| &**s)
                .collect();
            let name = head.join("|");
            let (origin, source_len) = chunk.jit_debug_origin();
            eprintln!(
                "[jit-map-range] {:x} {:x} {name}@{origin}+{source_len}/{}ops/{}slots",
                mem as usize,
                len,
                ops.len(),
                chunk.jit_slot_names().len()
            );
            for (pc, (&offset, op)) in pc_offsets.iter().zip(ops).enumerate() {
                let property = match *op {
                    Op::GetProp(name, _)
                    | Op::GetMethod(name, _)
                    | Op::GetPropThis(name, _)
                    | Op::GetPropLocal(_, name, _)
                    | Op::SetProp(name, _)
                    | Op::SetPropDrop(name, _)
                    | Op::SetPropThisDrop(name, _)
                    | Op::SetPropLocalDrop(_, name, _) => {
                        format!(" key={:?}", chunk.jit_name(name))
                    }
                    _ => String::new(),
                };
                eprintln!(
                    "[jit-map-pc] {:x} {:x} {pc} {op:?} {name}{property}",
                    mem as usize, offset
                );
            }
            LAST_MARKS.with(|cell| {
                for &(offset, family) in cell.borrow().iter() {
                    eprintln!("[jit-map-mark] {:x} {offset:x} {family}", mem as usize);
                }
                cell.borrow_mut().clear();
            });
        }
        Some(JitCode {
            entry_kind,
            osr_entry_depths: if borrowed_entry && !chunk.jit_is_resumable() {
                (0..ops.len()).map(|pc| cfg.osr_entry_depth(pc)).collect()
            } else {
                Vec::new()
            },
            resume_depths: if borrowed_entry {
                (0..ops.len()).map(|pc| cfg.stack_depth_at(pc)).collect()
            } else {
                Vec::new()
            },
            needs_global: ops.iter().any(|o| {
                matches!(
                    o,
                    Op::LoadName(..)
                        | Op::LoadNameForCall(..)
                        | Op::LoadNameIn(..)
                        | Op::LoadNameForCallIn(..)
                )
            }),
            mem,
            len,
            pc_offsets,
            max_stack,
            executable,
            residency,
        })
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_residency_mark(a: &mut asm::Asm, residency: u64, enter: bool) {
    const _: () = assert!(std::mem::offset_of!(cache::CodeResidency, active) == 0);
    a.mov_imm64(9, residency);
    a.ldr_imm(10, 9, 0);
    if enter {
        a.add_imm(10, 10, 1);
    } else {
        a.sub_imm(10, 10, 1);
    }
    a.str_imm(10, 9, 0);
    if enter {
        a.movz(10, 1, 0);
        a.strb_imm(
            10,
            9,
            std::mem::offset_of!(cache::CodeResidency, referenced) as u32,
        );
    }
}

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
#[path = "jit_x64.rs"]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
mod x64;

#[cfg(test)]
#[path = "jit_exec_memory_tests.rs"]
mod exec_memory_tests;

#[cfg(all(test, target_arch = "aarch64"))]
#[path = "jit_call_epoch_tests.rs"]
mod call_epoch_tests;
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "native_array_local_tests.rs"]
mod native_array_local_tests;
#[cfg(test)]
#[path = "jit_numeric_array_tests.rs"]
mod numeric_array_tests;
#[cfg(test)]
#[path = "jit_property_store_tests.rs"]
mod property_store_tests;
#[cfg(test)]
#[path = "jit_return_tests.rs"]
mod return_tests;

#[cfg(all(
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub fn compile(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
) -> Option<JitCode> {
    x64::compile(chunk, layout, ilayout)
}

#[cfg(all(
    target_arch = "x86_64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn compile_entry(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    entry_kind: NativeEntryKind,
) -> Option<JitCode> {
    x64::compile_entry(chunk, layout, ilayout, entry_kind)
}

#[cfg(not(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
)))]
pub fn compile(
    _chunk: &Chunk,
    _layout: &crate::value::JitLayout,
    _ilayout: &crate::interpreter::InterpLayout,
) -> Option<JitCode> {
    None
}

#[cfg(not(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
)))]
fn compile_entry(
    _chunk: &Chunk,
    _layout: &crate::value::JitLayout,
    _ilayout: &crate::interpreter::InterpLayout,
    _entry_kind: NativeEntryKind,
) -> Option<JitCode> {
    None
}

/// Distinguish a temporary capacity shortage from a permanent native-tier bail.
pub(crate) enum JitCompileOutcome {
    Compiled(JitCode),
    /// The body is supported, but its executable reservation did not fit.
    Deferred {
        required_bytes: usize,
    },
    Unavailable,
}

/// Compile one chunk and, when explicitly requested, account for compilation latency and emitted
/// instruction bytes. Keeping this wrapper outside the target-specific emitters ensures AArch64,
/// x86-64, successful compilations, and defensive fallbacks use one measurement definition.
pub(crate) fn compile_profiled(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
) -> JitCompileOutcome {
    compile_profiled_with(chunk, || compile(chunk, layout, ilayout))
}

pub(crate) fn compile_borrowed_profiled(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
) -> JitCompileOutcome {
    if chunk.jit_is_resumable() {
        return JitCompileOutcome::Unavailable;
    }
    compile_profiled_with(chunk, || {
        compile_entry(chunk, layout, ilayout, NativeEntryKind::BorrowedFrame)
    })
}

fn compile_profiled_with(
    chunk: &Chunk,
    compile: impl FnOnce() -> Option<JitCode>,
) -> JitCompileOutcome {
    let pressure = CompilationPressureScope::enter();
    let started = perf_metrics_enabled().then(std::time::Instant::now);
    let result = compile();
    if let Some(code) = &result {
        profiler::register(chunk, code.mem, code.len);
    }
    if let Some(started) = started {
        let elapsed = started.elapsed();
        use std::sync::atomic::Ordering::Relaxed;
        PERF_COMPILE_ATTEMPTS.fetch_add(1, Relaxed);
        PERF_COMPILE_NANOS.fetch_add(elapsed.as_nanos().min(u64::MAX as u128) as u64, Relaxed);
        if let Some(code) = &result {
            let bytes = code.len as u64;
            PERF_COMPILE_SUCCESSES.fetch_add(1, Relaxed);
            PERF_GENERATED_CODE_BYTES.fetch_add(bytes, Relaxed);
            PERF_LARGEST_CODE_BYTES.fetch_max(bytes, Relaxed);
        }
    }
    match result {
        Some(code) => JitCompileOutcome::Compiled(code),
        None => {
            let required_bytes = pressure.denied_bytes();
            if required_bytes > 0 && required_bytes <= executable_code_budget().limit {
                JitCompileOutcome::Deferred { required_bytes }
            } else {
                JitCompileOutcome::Unavailable
            }
        }
    }
}

/// Whether `layout` is usable for the inline GetProp template: valid (probed std layouts hold)
/// and every offset it bakes fits its instruction's immediate range.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn get_prop_inlinable(layout: &crate::value::JitLayout) -> bool {
    let sh = layout.obj_props + layout.props_shape;
    layout.valid
        // Packed property values are eight bytes. Until these templates decode them, keep only
        // property/name/element operations on their checked paths; unrelated JIT templates stay
        // enabled. In the old wide layout `meta` followed the full 16-byte Value.
        && layout.entry_accessor >= layout.entry_value + 8
        && layout.obj_from_cell < 4096
        && layout.obj_exotic < 4096
        && layout.obj_ic_plain < 4096
        && sh.is_multiple_of(4)
        && sh / 4 < 4096
        && fields_data_fits(layout)
        && fields_len_fits(layout)
        && layout.entry_accessor < 4096
        && layout.entry_value + 16 < 256
        && layout.entry_size < 0x1_0000
        && (layout.obj_props + layout.props_layout).is_multiple_of(8)
        && (layout.obj_props + layout.props_layout) / 8 < 4096
        && (layout.layout_data_off + layout.vec_ptr_off).is_multiple_of(8)
        && (layout.layout_data_off + layout.vec_len_off).is_multiple_of(8)
        && layout.layout_data_off + layout.vec_ptr_off < 4096
        && layout.layout_data_off + layout.vec_len_off < 4096
        && std::mem::offset_of!(JitCtx, shape_layouts).is_multiple_of(8)
        && std::mem::offset_of!(JitCtx, shape_layouts) / 8 < 4096
        && layout.shape_layout_entry_size == 16
        && layout.shape_layout_entry_id == 0
        && layout.shape_layout_entry_keys == 8
        && crate::value::SHAPE_LAYOUT_PAGE_BITS == 6
        && crate::value::SHAPE_LAYOUT_PAGE_SIZE == 64
        && crate::value::SHAPE_LAYOUT_PAGE_COUNT == 64
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn guard_prop_data(a: &mut asm::Asm, reg: u32, base: u32, flags: u32, slow: usize) {
    a.ldrb_imm(reg, base, flags);
    let bit = asm::logical_imm_w(crate::value::PROP_ACCESSOR as u32).unwrap();
    a.logic_imm_w(0, reg, reg, bit);
    a.cbnz(reg, false, slow);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn guard_prop_writable(a: &mut asm::Asm, reg: u32, base: u32, flags: u32, slow: usize) {
    a.ldrb_imm(reg, base, flags);
    let bit = asm::logical_imm_w(crate::value::PROP_WRITABLE as u32).unwrap();
    a.logic_imm_w(0, reg, reg, bit);
    a.cbz(reg, false, slow);
}

/// Whether generated code admits a Map's or Set's `Exotic::Collection` wherever it
/// admits an ordinary object: [`crate::value::Exotic::is_ordinary`] accepts it, and the
/// interpreter's caches treat its named fields and elements as ordinary. Its tag must directly
/// follow None's (`JitRepr::exotic_ordinary_contiguous`) for one range test to admit both;
/// otherwise a Collection takes the checked path. Both backends' ordinary-object guards ask it.
#[cfg(any(
    test,
    all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
fn exotic_collection_admitted(layout: &crate::value::JitLayout) -> bool {
    layout.repr.exotic_collection_tag.is_some() && layout.repr.exotic_ordinary_contiguous
}

/// Branch to `ok` when the exotic discriminant byte loaded in `reg` is one
/// [`crate::value::Exotic::is_ordinary`] accepts; otherwise fall through. Without Collection
/// admitted: `cmp reg, #none; b.eq ok`. With it ([`exotic_collection_admitted`]), one
/// unsigned range test from None's tag: `sub scratch, reg, #none; cmp scratch, #1; b.ls ok`,
/// which preserves `reg` unless `scratch` is `reg`. Clobbers NZCV and `scratch`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn branch_exotic_ordinary(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    reg: u32,
    scratch: u32,
    ok: usize,
) {
    let none_tag = layout.exotic_none_tag as u32;
    if exotic_collection_admitted(layout) {
        a.sub_imm(scratch, reg, none_tag);
        a.cmp_imm_w(scratch, 1);
        a.b_cond(C_LS, ok);
    } else {
        a.cmp_imm_w(reg, none_tag);
        a.b_cond(C_EQ, ok);
    }
}

/// Branch to `miss` unless the exotic discriminant byte loaded in `reg` is one
/// [`crate::value::Exotic::is_ordinary`] accepts (see [`branch_exotic_ordinary`]):
/// `cmp reg, #none; b.ne miss`, or with Collection admitted
/// `sub scratch, reg, #none; cmp scratch, #1; b.hi miss`. `scratch` may be `reg` when the tag
/// is dead afterwards. Clobbers NZCV and `scratch`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn guard_exotic_ordinary(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    reg: u32,
    scratch: u32,
    miss: usize,
) {
    let none_tag = layout.exotic_none_tag as u32;
    if exotic_collection_admitted(layout) {
        a.sub_imm(scratch, reg, none_tag);
        a.cmp_imm_w(scratch, 1);
        a.b_cond(C_HI, miss);
    } else {
        a.cmp_imm_w(reg, none_tag);
        a.b_cond(C_NE, miss);
    }
}

/// [`guard_exotic_ordinary`] that also admits `Exotic::Array` (the element templates and the
/// key-checked Array holder): the ordinary test, then `cmp reg, #array; b.ne miss`, so
/// `scratch` must differ from `reg`. Clobbers NZCV and `scratch`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn guard_exotic_ordinary_or_array(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    reg: u32,
    scratch: u32,
    miss: usize,
) {
    debug_assert_ne!(reg, scratch, "the Array compare reads the tag");
    let ok = a.new_label();
    branch_exotic_ordinary(a, layout, reg, scratch, ok);
    a.cmp_imm_w(reg, layout.exotic_array_tag as u32);
    a.b_cond(C_NE, miss);
    a.bind(ok);
}

/// Branch to `miss` unless the exotic discriminant byte already loaded in `reg` is ordinary
/// (`Exotic::None`, or a Collection: see [`branch_exotic_ordinary`]) or an
/// admitted primitive-wrapper tag. Wrapper tags are admitted only at sites whose compile-time
/// name passed the wrapper name gates (not `length`/`description`, not digit-leading): StrWrap
/// answers those names outside its entries, Num/Bool wrappers answer none. Shape, `ic_plain`,
/// slot bounds and the live accessor bit are still checked by the caller around this guard, so
/// a mutation after the fill re-routes to the helper. Clobbers NZCV and `scratch`, which must
/// differ from `reg` when a wrapper is admitted.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn guard_exotic_none_or_wrap(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    reg: u32,
    scratch: u32,
    str_wrap: bool,
    num_bool_wrap: bool,
    miss: usize,
) {
    if !str_wrap && !num_bool_wrap {
        guard_exotic_ordinary(a, layout, reg, scratch, miss);
        return;
    }
    debug_assert_ne!(reg, scratch, "the wrapper compares read the tag");
    let ok = a.new_label();
    branch_exotic_ordinary(a, layout, reg, scratch, ok);
    if str_wrap {
        a.cmp_imm_w(reg, layout.exotic_strwrap_tag as u32);
        a.b_cond(C_EQ, ok);
    }
    if num_bool_wrap {
        a.cmp_imm_w(reg, layout.exotic_numwrap_tag as u32);
        a.b_cond(C_EQ, ok);
        a.cmp_imm_w(reg, layout.exotic_boolwrap_tag as u32);
        a.b_cond(C_EQ, ok);
    }
    a.b(miss);
    a.bind(ok);
}

/// x10 = the ACTIVE realm's wrapper prototype at `proto_off` within the `Interp` reached
/// through ctx.interp — never an embedded prototype identity, so a realm swap between the
/// fill and the hit is observed exactly as the Rust helper observes it.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_load_active_proto(a: &mut asm::Asm, proto_off: usize) {
    a.ldr_imm(10, 19, 72); // ctx.interp
    a.ldr_imm(10, 10, proto_off as u32);
}

/// Inline shape-validated property load, unified over `GetProp` (`method == false`: pop the
/// receiver, push the value in its slot) and `GetMethod` (`method == true`: the receiver stays —
/// it is re-used as `this` — and the method pushes above it), and over IC depths 0..=2:
/// the value may live on the receiver itself, its prototype, or two hops up (a subclass
/// hierarchy). Every hop re-follows the live proto pointer and re-validates exotic-None +
/// `ic_plain` + shape — a shape match on a non-holder hop proves it still lacks the name (see
/// [`crate::bytecode::IcState`]); depth 2 additionally requires the recorded `mid_shape`
/// (`mid_ok`). Every guard branches to `slow` before any state is written, so the fallback
/// re-runs the op cleanly. An accessor, a function's deferred `prototype` or any guard miss
/// falls to the checked helper.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// Where a property read's receiver comes from.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy, PartialEq)]
enum PropRecv {
    /// Operand stack top (classic GetProp/GetMethod): consumed.
    Stack,
    /// The frame's `this` binding: the packed word at [48] of the context, kept by the frame.
    This,
    /// A local slot (alive for the whole frame).
    Slot(u32),
}

/// Receiver/holder variants of the generic property-cache way probe (see
/// [`emit_prop_way_probe`]); each distinct combination is one shared stub per chunk.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy)]
struct PropProbeFlags {
    /// An `Exotic::Array` receiver may shape-validate (the site's name is not an element key).
    arr_ok: bool,
    /// A primitive read in a non-consuming form probes the active realm's wrapper
    /// prototype (and Str/Num/Bool wrapper receivers, mids and holders pass).
    wrap_ok: bool,
    /// A method load: cached absence is not served (the helper throws for a missing method).
    method: bool,
    /// Key-checked Array holder states route to the site's key compare.
    kc: bool,
}

/// One generic property-cache way probe: x12 = the way's `IcState` cell, x10 = the receiver's
/// cell address (preserved). A validated data hit branches to `load` (x11 = holder object
/// base, x13 = slot), a key-checked holder to `load_kc` (same registers; the site verifies the
/// entry key), a validated cached absence to `absent_hit`; anything else to `miss`. Clobbers
/// x9, x11..x17 only.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
fn emit_prop_way_probe(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    flags: PropProbeFlags,
    miss: usize,
    load: usize,
    load_kc: usize,
    absent_hit: usize,
) {
    use crate::bytecode::{
        IC_OFF_DEPTH, IC_OFF_HOLDER_SHAPE, IC_OFF_MID2_SHAPE, IC_OFF_MID3_SHAPE, IC_OFF_MID4_SHAPE,
        IC_OFF_MID_OK, IC_OFF_MID_SHAPE, IC_OFF_RECV_SHAPE, IC_OFF_SLOT,
    };
    let PropProbeFlags {
        arr_ok,
        wrap_ok,
        method,
        kc,
    } = flags;
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let pr = layout.obj_proto as u32;
    let sh = (layout.obj_props + layout.props_shape) as u32;
    let plain = layout.obj_ic_plain as u32;
    let d1 = a.new_label();
    a.ldrb_imm(9, 12, IC_OFF_DEPTH);
    a.ldr_w_imm(13, 12, IC_OFF_SLOT);
    // receiver hop: exotic None (or Array when `arr_ok` — but only as a NON-holder, so an
    // Array receiver additionally requires depth ≥ 1: its shape proves named-key ABSENCE,
    // not slot positions, because element entries occupy slots without transitioning the
    // shape; or a primitive wrapper when `wrap_ok` — StrWrap intercepts only index and
    // `length` reads and Num/Bool wrappers intercept none, all excluded by the wrap_ok
    // name gates), plain, shape == recv_shape; x11 = receiver object base
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(14, 11, ex);
    if arr_ok || wrap_ok {
        let ex_ok = a.new_label();
        branch_exotic_ordinary(a, layout, 14, 15, ex_ok);
        if arr_ok {
            let not_arr = a.new_label();
            a.cmp_imm_w(14, layout.exotic_array_tag as u32);
            a.b_cond(C_NE, not_arr);
            a.cbz(9, false, miss); // Array receiver must not be the holder (w9 = depth)
            a.b(ex_ok);
            a.bind(not_arr);
        }
        if wrap_ok {
            a.cmp_imm_w(14, layout.exotic_strwrap_tag as u32);
            a.b_cond(C_EQ, ex_ok);
            a.cmp_imm_w(14, layout.exotic_numwrap_tag as u32);
            a.b_cond(C_EQ, ex_ok);
            a.cmp_imm_w(14, layout.exotic_boolwrap_tag as u32);
            a.b_cond(C_EQ, ex_ok);
        }
        a.b(miss);
        a.bind(ex_ok);
    } else {
        guard_exotic_ordinary(a, layout, 14, 14, miss);
    }
    a.ldrb_imm(14, 11, plain);
    a.cbz(14, false, miss);
    a.ldr_w_imm(14, 11, sh);
    a.ldr_w_imm(16, 12, IC_OFF_RECV_SHAPE);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, miss);
    // depth routing: 0 → holder is the receiver; 1 → one hop; 2/3 validate their recorded
    // intermediate shapes then fall to d1 for the holder. Non-plain depths divert to the
    // key-checked decoder (`kc_route`) so the common depths pay nothing for its existence.
    // Accessor states (IC_ACCESSOR | depth, never key-checked here) route identically:
    // the chain validation is descriptor-kind agnostic, and the load landing re-checks
    // the live descriptor kind and diverts accessors to the validated accessor landing.
    let kc_route = if kc { a.new_label() } else { miss };
    let depth2 = a.new_label();
    let depth3 = a.new_label();
    let other = a.new_label();
    a.cbz(9, false, load);
    a.cmp_imm_w(9, 1);
    a.b_cond(C_EQ, d1);
    a.cmp_imm_w(9, crate::bytecode::IC_ACCESSOR as u32);
    a.b_cond(C_EQ, load);
    a.cmp_imm_w(9, (crate::bytecode::IC_ACCESSOR | 1) as u32);
    a.b_cond(C_EQ, d1);
    a.cmp_imm_w(9, (crate::bytecode::IC_ACCESSOR | 2) as u32);
    a.b_cond(C_EQ, depth2);
    a.cmp_imm_w(9, (crate::bytecode::IC_ACCESSOR | 3) as u32);
    a.b_cond(C_EQ, depth3);
    a.cmp_imm_w(9, 2);
    a.b_cond(C_EQ, depth2);
    a.cmp_imm_w(9, 3);
    a.b_cond(C_NE, other);
    a.bind(depth3);
    a.ldrb_imm(14, 12, IC_OFF_MID_OK);
    a.cmp_imm_w(14, 3);
    a.b_cond(C_NE, miss);
    // depth-3 first intermediate hop.
    a.ldr_imm(17, 11, pr);
    a.cbz(17, true, miss);
    a.add_imm(11, 17, obj_off);
    a.ldrb_imm(14, 11, ex);
    guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
    a.ldrb_imm(14, 11, plain);
    a.cbz(14, false, miss);
    a.ldr_w_imm(14, 11, sh);
    a.ldr_w_imm(16, 12, IC_OFF_MID_SHAPE);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, miss);
    // depth-3 second intermediate hop; the common holder validation follows at d1.
    a.ldr_imm(17, 11, pr);
    a.cbz(17, true, miss);
    a.add_imm(11, 17, obj_off);
    a.ldrb_imm(14, 11, ex);
    guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
    a.ldrb_imm(14, 11, plain);
    a.cbz(14, false, miss);
    a.ldr_w_imm(14, 11, sh);
    a.ldr_w_imm(16, 12, IC_OFF_MID2_SHAPE);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, miss);
    a.b(d1);
    a.bind(depth2);
    a.ldrb_imm(14, 12, IC_OFF_MID_OK);
    a.cbz(14, false, miss);
    // depth-2 mid hop: follow the live proto, validate against mid_shape
    a.ldr_imm(17, 11, pr); // Option<Gc> niche: pointer or 0
    a.cbz(17, true, miss);
    a.add_imm(11, 17, obj_off);
    a.ldrb_imm(14, 11, ex);
    guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
    a.ldrb_imm(14, 11, plain);
    a.cbz(14, false, miss);
    a.ldr_w_imm(14, 11, sh);
    a.ldr_w_imm(16, 12, IC_OFF_MID_SHAPE);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, miss);
    // holder hop (depth 1 entry point; depth 2 falls through): validate holder_shape
    a.bind(d1);
    a.ldr_imm(17, 11, pr);
    a.cbz(17, true, miss);
    a.add_imm(11, 17, obj_off);
    a.ldrb_imm(14, 11, ex);
    guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
    a.ldrb_imm(14, 11, plain);
    a.cbz(14, false, miss);
    a.ldr_w_imm(14, 11, sh);
    a.ldr_w_imm(16, 12, IC_OFF_HOLDER_SHAPE);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, miss);
    a.b(load);
    // Cached ABSENCE (`IC_ABSENT`, the AST-shaped read `node.optionalField`): re-walk the
    // live chain — every level ic-plain with a shape matching the recorded walk and an
    // exotic that answers no name outside its entries (None or Num/Bool wrapper; ABSENT
    // states fill only from such chains, so re-require that on receivers the
    // `arr_ok`/`wrap_ok` gates let through — an Array or StrWrap level keeps the helper,
    // exactly as in `ic_shape_probe`) — and the chain must END where the fill saw it end.
    // Then the read is `undefined` with no entry scan at all. Method loads keep the
    // helper (an absent method throws there anyway).
    a.bind(other);
    if !method {
        a.cmp_imm_w(9, crate::bytecode::IC_ABSENT as u32);
        a.b_cond(C_NE, kc_route);
        if arr_ok || wrap_ok {
            a.ldrb_imm(14, 11, ex);
            guard_exotic_none_or_wrap(a, layout, 14, 15, false, true, miss);
        }
        let chain_end = a.new_label();
        for (lvl, shape_off) in [
            (2u32, IC_OFF_MID_SHAPE),
            (3u32, IC_OFF_MID2_SHAPE),
            (4u32, IC_OFF_MID3_SHAPE),
            (5u32, IC_OFF_MID4_SHAPE),
            (6u32, IC_OFF_HOLDER_SHAPE),
        ] {
            a.cmp_imm_w(13, lvl);
            a.b_cond(C_LO, chain_end);
            a.ldr_imm(17, 11, pr);
            a.cbz(17, true, miss); // chain ended before the recorded level count
            a.add_imm(11, 17, obj_off);
            a.ldrb_imm(14, 11, ex);
            guard_exotic_none_or_wrap(a, layout, 14, 15, false, true, miss);
            a.ldrb_imm(14, 11, plain);
            a.cbz(14, false, miss);
            a.ldr_w_imm(14, 11, sh);
            a.ldr_w_imm(16, 12, shape_off);
            a.cmp_reg_w(14, 16);
            a.b_cond(C_NE, miss);
        }
        a.bind(chain_end);
        a.ldr_imm(17, 11, pr);
        a.cbz(17, true, absent_hit);
        a.b(miss); // a proto was attached where the fill saw the end
    } else if !kc {
        a.b(miss);
    }
    // key-checked states (`IC_ARR_KEYCHK`): 0x40 = the array receiver IS the holder
    // (`arr.length`); 0x41 = one hop to an array holder (Array.prototype methods — itself
    // an Array exotic). The receiver-side Array gate above already passed 0x40 (nonzero
    // depth). Deeper key-checked states → helper.
    if kc {
        a.bind(kc_route);
        a.cmp_imm_w(9, 0x40);
        a.b_cond(C_EQ, load_kc);
        a.cmp_imm_w(9, 0x41);
        a.b_cond(C_NE, miss);
        // one proto hop; the holder may be ordinary or an Array (its entry key gets
        // re-checked, which is what makes an array holder's slot trustworthy at all)
        a.ldr_imm(17, 11, pr);
        a.cbz(17, true, miss);
        a.add_imm(11, 17, obj_off);
        a.ldrb_imm(14, 11, ex);
        guard_exotic_ordinary_or_array(a, layout, 14, 15, miss);
        a.ldrb_imm(14, 11, plain);
        a.cbz(14, false, miss);
        a.ldr_w_imm(14, 11, sh);
        a.ldr_w_imm(16, 12, IC_OFF_HOLDER_SHAPE);
        a.cmp_reg_w(14, 16);
        a.b_cond(C_NE, miss);
        a.b(load_kc);
    }
}

/// Status codes the shared property way loop returns in w9 (0 = no way validated).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const PROP_PROBE_LOAD: u32 = 1;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const PROP_PROBE_LOAD_KC: u32 = 2;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const PROP_PROBE_ABSENT: u32 = 3;

/// Probe every cache way of a site (x8 = its first `IcState` cell, x10 = receiver): x8 is the
/// way cursor and w7 the ways left, both untouched by the probe body. Clobbers x7..x9, x11..x17.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_prop_way_loop(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    flags: PropProbeFlags,
    miss: usize,
    load: usize,
    load_kc: usize,
    absent_hit: usize,
) {
    let l_way = a.new_label();
    let l_way_next = a.new_label();
    a.movz(7, crate::bytecode::PROP_IC_WAYS as u32, 0);
    a.bind(l_way);
    a.mov(12, 8);
    emit_prop_way_probe(a, layout, flags, l_way_next, load, load_kc, absent_hit);
    a.bind(l_way_next);
    let ic_stride = std::mem::size_of::<std::cell::Cell<crate::bytecode::IcState>>();
    a.add_imm(8, 8, ic_stride as u32);
    a.sub_imm(7, 7, 1);
    a.cbnz(7, false, l_way);
    a.b(miss);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// `str.length` for a String primitive in execution word `raw` whose flat buffer is known
/// ASCII: one UTF-16 code unit per byte, so the byte length is the answer. Any other String
/// (a view, or text that may hold non-ASCII code points) goes to `slow`, which counts code
/// units; a non-String falls through with `raw` preserved. `consume` is the stack form: the
/// receiver word at [x20-8] is replaced by the length. Otherwise the receiver is a borrowed
/// binding and the length is pushed. Clobbers x11-x14, d0 and NZCV.
fn emit_ascii_string_length(a: &mut asm::Asm, raw: u32, consume: bool, done: usize, slow: usize) {
    let not_string = a.new_label();
    a.lsr_imm(11, raw, 48);
    a.movz(12, (crate::value::PACK_STR >> 48) as u32, 0);
    a.cmp_reg_w(11, 12);
    a.b_cond(C_NE, not_string);
    emit_exec_payload(a, raw, 11);
    a.ldr_w_imm(14, 11, crate::lstr::CAP_OFF as u32);
    a.lsr_imm(14, 14, 31);
    a.cbz(14, false, slow);
    a.ldr_w_imm(14, 11, crate::lstr::LEN_OFF as u32);
    a.ucvtf_d_w(0, 14);
    if consume {
        a.stur_d(0, 20, -8);
    } else {
        a.str_d_imm(0, 20, 0);
        a.add_imm(20, 20, 8);
    }
    a.b(done);
    a.bind(not_string);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_prop_load_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    // Interp field offsets: string-primitive method receivers resolve against the ACTIVE
    // realm's String.prototype through ctx.interp.
    il: &crate::interpreter::InterpLayout,
    cache_ptr: usize,
    preferred: Option<crate::bytecode::IcState>,
    // The site's interned name (`chunk.jit_name(n)`): its data pointer keys the stub-cache
    // arm, and its bytes are the key-checked arm's compare immediates. Pinned by the chunk
    // the emitted code belongs to.
    name: &str,
    pc: u32,
    l_unwind: usize,
    method: bool,
    // Whether an `Exotic::Array` receiver may shape-validate: true when the site's (compile-time)
    // name cannot be an element key — element inserts don't transition an array's shape, but
    // element keys are all canonical indices, so a name that doesn't start with a digit cannot
    // collide with one. Prototype hops stay `Exotic::None`-only.
    arr_ok: bool,
    recv: PropRecv,
) {
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let pr = layout.obj_proto as u32;
    let sh = (layout.obj_props + layout.props_shape) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let es = layout.entry_size as u64;

    let plain = layout.obj_ic_plain as u32;
    // Key-checked entries (`IC_ARR_KEYCHK`: the holder is an Array — `arr.length`, or a method
    // on the Array-exotic `Array.prototype`): validated by an inline byte-compare of the entry's
    // key against the site's compile-time name. Only for short names with a good fat-pointer
    // probe; otherwise those states keep falling to the helper.
    let kc = layout.key_probe_ok && !name.is_empty() && name.len() <= 8;
    let slow = a.new_label();
    let done = a.new_label();
    let load = a.new_label();
    let load_data = a.new_label();
    let load_kc = a.new_label();
    let absent_hit = a.new_label();
    // 1. receiver must be an Obj (tag 8); x10 = its stored pointer — hop walking uses x17. The
    //    this/slot forms read a binding the frame keeps.
    // Primitive receivers in the NON-CONSUMING forms only (a method load keeps its stack
    // receiver as `this`): resolve against the active realm's wrapper prototype, whose stored
    // pointer plays the receiver role for the probe. The Rust helper fills the SAME site cache
    // with proto-based states (see get_prop_ic's primitive arm), so shapes align. The consuming
    // stack GetProp keeps the helper for primitives. Name gates mirror the helper's primitive
    // arm (`length`/`description`/digit-leading stay checked).
    let fits_proto = |off: usize| off & 7 == 0 && off / 8 < 4096;
    let wrap_ok = (method || !matches!(recv, PropRecv::Stack))
        && il.valid
        && fits_proto(il.string_proto)
        && fits_proto(il.number_proto)
        && fits_proto(il.boolean_proto)
        && name != "length"
        && name != "description"
        && !name.as_bytes().first().is_some_and(|b| b.is_ascii_digit());
    // Packed receiver word in x9 → x10 = the object the probe walks: the Obj payload itself,
    // or the active realm's wrapper prototype for a Str/Num/Bool primitive (kinds are the
    // wide-Value discriminants `emit_exec_kind` decodes: Bool 3, Num 4, Str 6, Obj 8).
    // Every other kind (nullish, Symbol, BigInt, Empty) branches to `slow` before anything
    // is written, where the helper reproduces OrdinaryGet — including its TypeError — with
    // the original primitive receiver for any accessor it runs.
    let emit_packed_probe_receiver = |a: &mut asm::Asm| {
        let obj_recv = a.new_label();
        let str_recv = a.new_label();
        let bool_recv = a.new_label();
        let probe_go = a.new_label();
        // Obj is the overwhelmingly common receiver: single out with the cheap top-16
        // compare first (as the string-method path always did) so ordinary object reads
        // pay nothing for the primitive cases. Kind discrimination below runs only for
        // non-object receivers.
        a.lsr_imm(11, 9, 48);
        a.movz(12, (crate::value::PACK_OBJ >> 48) as u32, 0);
        a.cmp_reg_w(11, 12);
        a.b_cond(C_EQ, obj_recv);
        emit_exec_kind(a, 9, 11, 12, slow);
        a.cmp_imm_w(11, 6);
        a.b_cond(C_EQ, str_recv);
        a.cmp_imm_w(11, 3);
        a.b_cond(C_EQ, bool_recv);
        a.cmp_imm_w(11, 4); // Num is the only remaining admissible kind
        a.b_cond(C_NE, slow);
        emit_load_active_proto(a, il.number_proto);
        a.b(probe_go);
        a.bind(str_recv);
        emit_load_active_proto(a, il.string_proto);
        a.b(probe_go);
        a.bind(bool_recv);
        emit_load_active_proto(a, il.boolean_proto);
        a.b(probe_go);
        a.bind(obj_recv);
        emit_exec_payload(a, 9, 10);
        a.bind(probe_go);
    };
    // `length` of a String primitive is its own non-writable, non-configurable data property
    // (ECMA-262 §10.4.3.5 StringGetOwnProperty via ToObject), so no user code can intervene.
    let string_length = !method && name == "length";
    match recv {
        PropRecv::Stack => {
            emit_exec_word_load(a, 9, 20, -8);
            if string_length {
                emit_ascii_string_length(a, 9, true, done, slow);
            }
            if wrap_ok {
                // Stack + wrap_ok implies a method load: the receiver word stays below the
                // method as its `this`.
                debug_assert!(method);
                emit_packed_probe_receiver(a);
            } else {
                emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
                emit_exec_payload(a, 9, 10);
            }
        }
        PropRecv::This => {
            // The packed word at [48]: the slot form's guard and payload, and its packed probe
            // for primitive receivers.
            emit_exec_word_load(a, 9, 19, 48); // ctx.this_word
            if wrap_ok {
                emit_packed_probe_receiver(a);
            } else {
                emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
                emit_exec_payload(a, 9, 10);
            }
        }
        PropRecv::Slot(off) => {
            emit_exec_word_load(a, 9, 22, off as i32);
            if string_length {
                emit_ascii_string_length(a, 9, false, done, slow);
            }
            if wrap_ok {
                emit_packed_probe_receiver(a);
            } else {
                emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
                emit_exec_payload(a, 9, 10);
            }
        }
    }
    // After bytecode warmup, ordinary OO sites overwhelmingly have one stable depth/shape/slot.
    // Bake that state into a compact probe instead of embedding the full four-way/deep/exotic
    // state machine at every site. Every fact that made the cached slot authoritative is guarded
    // live; a miss goes to the checked helper, which observes mutations and arbitrary alternate
    // shapes exactly like the generic JIT miss path.
    let bakeable = |st: &crate::bytecode::IcState| {
        st.depth <= 3
            && (st.depth < 2
                || (st.depth == 2 && st.mid_ok & 1 != 0)
                || (st.depth == 3 && st.mid_ok & 3 == 3))
    };
    let compact = preferred.filter(bakeable);
    // A polymorphic site (virtual methods over a few receiver classes) bakes each warm way into
    // its own compact probe, tried in cache order; a miss on all of them falls to the generic
    // way loop, which still sees ways filled after compilation, and then to the helper.
    let baked: Vec<crate::bytecode::IcState> = match compact {
        Some(st) => vec![st],
        None if cache_ptr != 0 => (0..crate::bytecode::PROP_IC_WAYS)
            .map(|way| unsafe {
                (*(cache_ptr as *const std::cell::Cell<crate::bytecode::IcState>).add(way)).get()
            })
            .filter(|st| st.has_cacheable_shapes() && bakeable(st))
            .collect(),
        None => Vec::new(),
    };
    let emit_baked = |a: &mut asm::Asm, st: crate::bytecode::IcState, miss: usize| {
        a.add_imm(11, 10, obj_off);
        a.ldrb_imm(14, 11, ex);
        if arr_ok || wrap_ok {
            let recv_exotic_ok = a.new_label();
            branch_exotic_ordinary(a, layout, 14, 15, recv_exotic_ok);
            if arr_ok && st.depth >= 1 {
                a.cmp_imm_w(14, layout.exotic_array_tag as u32);
                a.b_cond(C_EQ, recv_exotic_ok);
            }
            if wrap_ok {
                a.cmp_imm_w(14, layout.exotic_strwrap_tag as u32);
                a.b_cond(C_EQ, recv_exotic_ok);
                a.cmp_imm_w(14, layout.exotic_numwrap_tag as u32);
                a.b_cond(C_EQ, recv_exotic_ok);
                a.cmp_imm_w(14, layout.exotic_boolwrap_tag as u32);
                a.b_cond(C_EQ, recv_exotic_ok);
            }
            a.b(miss);
            a.bind(recv_exotic_ok);
        } else {
            guard_exotic_ordinary(a, layout, 14, 14, miss);
        }
        a.ldrb_imm(14, 11, plain);
        a.cbz(14, false, miss);
        a.ldr_w_imm(14, 11, sh);
        a.mov_imm64(16, st.recv_shape as u64);
        a.cmp_reg_w(14, 16);
        a.b_cond(C_NE, miss);
        if st.depth >= 1 {
            a.ldr_imm(17, 11, pr);
            a.cbz(17, true, miss);
            a.add_imm(11, 17, obj_off);
            a.ldrb_imm(14, 11, ex);
            guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
            a.ldrb_imm(14, 11, plain);
            a.cbz(14, false, miss);
            a.ldr_w_imm(14, 11, sh);
            let expected = if st.depth == 1 {
                st.holder_shape
            } else {
                st.mid_shape
            };
            a.mov_imm64(16, expected as u64);
            a.cmp_reg_w(14, 16);
            a.b_cond(C_NE, miss);
        }
        if st.depth >= 2 {
            a.ldr_imm(17, 11, pr);
            a.cbz(17, true, miss);
            a.add_imm(11, 17, obj_off);
            a.ldrb_imm(14, 11, ex);
            guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
            a.ldrb_imm(14, 11, plain);
            a.cbz(14, false, miss);
            a.ldr_w_imm(14, 11, sh);
            let expected = if st.depth == 2 {
                st.holder_shape
            } else {
                st.mid2_shape
            };
            a.mov_imm64(16, expected as u64);
            a.cmp_reg_w(14, 16);
            a.b_cond(C_NE, miss);
        }
        if st.depth == 3 {
            a.ldr_imm(17, 11, pr);
            a.cbz(17, true, miss);
            a.add_imm(11, 17, obj_off);
            a.ldrb_imm(14, 11, ex);
            guard_exotic_none_or_wrap(a, layout, 14, 15, wrap_ok, wrap_ok, miss);
            a.ldrb_imm(14, 11, plain);
            a.cbz(14, false, miss);
            a.ldr_w_imm(14, 11, sh);
            a.mov_imm64(16, st.holder_shape as u64);
            a.cmp_reg_w(14, 16);
            a.b_cond(C_NE, miss);
        }
        a.mov_imm64(13, st.slot as u64);
        // Baked states are data-only by construction and their shapes now encode the
        // descriptor class, so the validated slot is provably a data property: take
        // the guard-free landing.
        a.b(load_data);
    };
    let generic_entry = a.new_label();
    for (index, st) in baked.iter().enumerate() {
        let miss = if index + 1 < baked.len() {
            a.new_label()
        } else if compact.is_some() {
            slow
        } else {
            generic_entry
        };
        emit_baked(a, *st, miss);
        if miss != slow && miss != generic_entry {
            a.bind(miss);
        }
    }
    a.bind(generic_entry);
    // 2-5. probe every cache way (sites allocate PROP_IC_WAYS consecutive cells; the fill path
    // demotes ways one step, so a site rotating through up to that many shapes stabilizes with
    // one shape per way). Each probe is self-contained: it recomputes the receiver base from
    // x10 and jumps to `load` with x11 = holder base, x13 = slot.
    // `cache_ptr` = the IcState cell address, or 0 for "x12 already holds it" (the stub-cache
    // arm computes the entry address at run time).
    // Probe every way through the chunk's shared out-of-line way loop (x8 = the site's first
    // cell, x10 = receiver); it reports which landing validated in w9. With stubs disabled, the
    // same loop is emitted at the site.
    if compact.is_none() {
        let flags = PropProbeFlags {
            arr_ok,
            wrap_ok,
            method,
            kc,
        };
        a.mov_imm64(8, cache_ptr as u64);
        if shared_stubs_enabled() {
            let stub = a.shared_stub(SharedStub::PropWays(flags).key());
            a.bl_label(stub);
            a.cbz(9, false, slow);
            a.cmp_imm_w(9, PROP_PROBE_LOAD);
            a.b_cond(C_EQ, load);
            if kc {
                a.cmp_imm_w(9, PROP_PROBE_LOAD_KC);
                a.b_cond(C_EQ, load_kc);
            }
            if !method {
                a.cmp_imm_w(9, PROP_PROBE_ABSENT);
                a.b_cond(C_EQ, absent_hit);
            }
            a.b(slow);
        } else {
            emit_prop_way_loop(a, layout, flags, slow, load, load_kc, absent_hit);
        }
    }
    // 6. x11 = holder base: bounds-check the cached slot against the live entries length
    //    (defense in depth — fills only record exact-slot holders, but an OOB read through a
    //    stale cache would be memory-unsafe, so verify), then entry = entries + slot*size;
    //    data property; never a deferred prototype
    let val = a.new_label();
    let val_body = a.new_label();
    let accessor_hit = a.new_label();
    let emit_entry = |a: &mut asm::Asm| {
        fields_len(a, layout, 16, 11);
        a.cmp_reg_x(13, 16);
        a.b_cond(C_HS, slow);
        fields_data(a, layout, 15, 11);
        a.mov_imm64(16, es);
        a.madd(15, 13, 16, 15);
    };
    // Baked (data-only, shape-proven-class) hits skip the descriptor-kind guard.
    a.bind(load_data);
    emit_entry(a);
    a.b(val_body);
    a.bind(load);
    emit_entry(a);
    a.bind(val);
    // An accessor at the validated slot is not a miss: the chain shapes, holder and slot
    // are already proven, so record them for the helper's validated accessor path instead
    // of discarding the validation and re-deriving the resolution in Rust.
    guard_prop_data(a, 9, 15, ea, accessor_hit);
    a.bind(val_body);
    if layout.entry_accessor == layout.entry_value + 8 {
        // Heap and execution words share the encoding.
        a.ldur(12, 15, ev);
    } else {
        a.ldur(9, 15, ev);
        a.ldur(13, 15, ev + 8);
        emit_exec_encode_wide(a, 9, 13, 12, 16, 17, 0, slow);
    }
    if name == "prototype" {
        // Only this key can hold a function's deferred prototype.
        property_word_guard(a, 12, 13, slow);
    }
    // --- commit: everything validated; from here only writes ---
    if !matches!(recv, PropRecv::Stack) {
        // this/slot receivers were never on the stack: just push the value.
        emit_exec_word_store(a, 12, 20, 0);
        a.add_imm(20, 20, 8);
    } else if method {
        // receiver stays at [-16]; push the method above it
        emit_exec_word_store(a, 12, 20, 0);
        a.add_imm(20, 20, 8);
    } else {
        // overwrite the receiver slot with the value (pop obj + push value = same depth)
        emit_exec_word_store(a, 12, 20, -8);
    }
    #[cfg(test)]
    regions::emit_property_event(a, 0);
    a.b(done);
    // 6kc. Key-checked landing (out of the hit path's fall-through line): same bounds + entry
    // compute as `load`, then verify the entry's key IS the site's name (length, then content
    // against immediates) — an array's slots aren't pinned by its shape, so the key is the
    // authority. Mismatch (slot shifted since fill) → helper re-derives. Ends by jumping back
    // into the shared value path.
    a.bind(load_kc);
    if kc {
        fields_len(a, layout, 16, 11);
        a.cmp_reg_x(13, 16);
        a.b_cond(C_HS, slow);
        fields_data(a, layout, 15, 11);
        a.mov_imm64(16, es);
        a.madd(15, 13, 16, 15);
        // Named fields have no inline key: follow the holder's shared layout at the same slot.
        a.ldr_imm(17, 11, (layout.obj_props + layout.props_layout) as u32);
        a.cbz(17, true, slow);
        a.ldr_imm(17, 17, (layout.layout_data_off + layout.vec_ptr_off) as u32);
        a.add_shifted(17, 17, 13, 4); // sizeof(Rc<str>) == 16
        let klen = layout.str_len_word as i32;
        let kptr = layout.str_ptr_word as i32;
        a.ldur(16, 17, klen);
        a.cmp_imm_x(16, name.len() as u32);
        a.b_cond(C_NE, slow);
        a.ldur(16, 17, kptr); // stored Rc<str> word (RcBox base)
        let d = layout.str_data_off as u32;
        let bytes = name.as_bytes();
        let mut off = 0usize;
        while bytes.len() - off >= 4 {
            let imm =
                u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            a.ldr_w_imm(17, 16, d + off as u32);
            a.movz(9, imm & 0xFFFF, 0);
            a.movk(9, imm >> 16, 1);
            a.cmp_reg_w(17, 9);
            a.b_cond(C_NE, slow);
            off += 4;
        }
        if bytes.len() - off >= 2 {
            let imm = u16::from_le_bytes([bytes[off], bytes[off + 1]]) as u32;
            a.ldrh_imm(17, 16, d + off as u32);
            a.movz(9, imm, 0);
            a.cmp_reg_w(17, 9);
            a.b_cond(C_NE, slow);
            off += 2;
        }
        if bytes.len() - off == 1 {
            a.ldrb_imm(17, 16, d + off as u32);
            a.cmp_imm_w(17, bytes[off] as u32);
            a.b_cond(C_NE, slow);
        }
        a.b(val);
    }
    // 6a. absent landing: the read is `undefined`. Tag-only write (stale payload is fine —
    // Undefined drops touch nothing; the Tdz template sets the same precedent).
    a.bind(absent_hit);
    if !method {
        a.mov_imm64(9, crate::value::PACK_UNDEFINED);
        match recv {
            PropRecv::Stack => {
                // overwrite the receiver in place
                emit_exec_word_store(a, 9, 20, -8);
            }
            PropRecv::This | PropRecv::Slot(_) => {
                emit_exec_word_store(a, 9, 20, 0);
                a.add_imm(20, 20, 8);
            }
        }
        #[cfg(test)]
        regions::emit_property_event(a, 1);
        a.b(done);
    }
    // Validated accessor landing: x11 = holder object base, x13 = slot. Publish both for
    // `jit_get_prop`'s validated accessor path, then take the ordinary helper call. No
    // JavaScript can run between the validation above and the helper consuming these.
    a.bind(accessor_hit);
    // x11 is the Object base; the helper wants the `Gc::as_ptr` form (the RefCell, at the
    // cell address), which sits `obj_from_cell` (the RefCell value offset) before it.
    a.sub_imm(9, 11, obj_off);
    a.str_imm(9, 19, std::mem::offset_of!(JitCtx, accessor_holder) as u32);
    a.str_w_imm(13, 19, std::mem::offset_of!(JitCtx, accessor_slot) as u32);
    a.bind(slow);
    #[cfg(test)]
    regions::emit_property_event(a, 2);
    emit_op_helper(a, H_GET_PROP, pc, l_unwind);
    a.bind(done);
}

/// Method read `receiver[key]` for a Number key (ECMA-262 EvaluatePropertyAccessWithExpressionKey,
/// GetValue): when the key is exactly an array index (u32) of a plain Object or Array receiver
/// with a dense own data element, replace the key with that element's word and keep the
/// receiver as the call's `this`. Stack `[receiver, key]` becomes `[receiver, element]`.
/// Everything else (holes and absent elements, which consult the prototype chain; accessors;
/// proxies, typed arrays and other side-table receivers; non-Object receivers; fractional,
/// negative or huge keys) branches to `slow` before any state changes. Enters with d0 = the
/// validated Number key; falls to `done` after the commit.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_method_elem_dense(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    done: usize,
) {
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let es = layout.entry_size as u64;
    let plain = layout.obj_ic_plain as u32;
    let commit = a.new_label();
    emit_exec_word_load(a, 10, 20, -16);
    emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 11, slow);
    emit_exec_payload(a, 10, 10);
    // The key must be exactly a u32 (round-trip compare; NaN/negative/fractional/huge all miss).
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    // Object base; exotic must be ordinary (None, or a Map/Set) or Array, and plain
    // (no side-table behavior).
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, slow);
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);
    if packed_elem_inlinable(layout) {
        let classic_dense = a.new_label();
        emit_packed_elements_base(a, layout, classic_dense);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.add_shifted(15, 15, 9, 4); // property_size == 16 (gate above)
        guard_prop_data(a, 14, 15, layout.property_meta as u32, slow);
        a.ldur(13, 15, layout.property_value as i32);
        a.mov_imm64(14, crate::value::PACK_EMPTY);
        a.cmp_reg_x(13, 14);
        a.b_cond(C_EQ, slow); // an absent own element must still consult the prototype chain
        a.mov(12, 13);
        a.b(commit);
        a.bind(classic_dense);
    }
    // Dense bounds, then slot = elems[n]; NO_SLOT (0xFFFF_FFFF) is a hole.
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 9, 15, ea, slow);
    if layout.entry_accessor == layout.entry_value + 8 {
        a.ldur(12, 15, ev);
    } else {
        a.ldur(9, 15, ev);
        a.ldur(13, 15, ev + 8);
        emit_exec_encode_wide(a, 9, 13, 12, 14, 16, 0, slow);
    }
    // Commit: the element simply replaces the Number key.
    a.bind(commit);
    emit_exec_word_store(a, 12, 20, -8);
    a.b(done);
}

/// Computed method read: a pinned string identity selects a bounded resolution, then every
/// live shape/descriptor is revalidated. The receiver stays owned on the operand stack; only
/// the key is replaced with the method. No state changes before all guards have succeeded.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_computed_method_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    il: &crate::interpreter::InterpLayout,
    pc: u32,
    l_unwind: usize,
) {
    use crate::bytecode::{
        ComputedReadCache, IC_OFF_DEPTH, IC_OFF_HOLDER_SHAPE, IC_OFF_MID2_SHAPE, IC_OFF_MID_OK,
        IC_OFF_MID_SHAPE, IC_OFF_RECV_SHAPE, IC_OFF_SLOT,
    };
    let fits = |offset: usize| offset.is_multiple_of(8) && offset / 8 < 4096;
    if !fits(il.computed_read_sets)
        || !il.computed_read_mask.is_multiple_of(4)
        || il.computed_read_mask / 4 >= 4096
        || !fits(il.string_proto)
        || !fits(layout.obj_proto)
        || layout.entry_accessor != layout.entry_value + 8
    {
        emit_op_helper(a, H_GET_METHOD_ELEM, pc, l_unwind);
        return;
    }
    let slow = a.new_label();
    let done = a.new_label();
    let object = a.new_label();
    let receiver = a.new_label();
    let string_key = a.new_label();
    let shape = (layout.obj_props + layout.props_shape) as u32;
    let exotic = layout.obj_exotic as u32;
    let plain = layout.obj_ic_plain as u32;
    let object_data = layout.obj_from_cell as u32;
    if get_elem_inlinable(layout) {
        // A Number key is an element read of a dense array: `table[i & mask](args)`, the
        // shape of every function-table and virtual call of Emscripten output.
        emit_exec_word_load(a, 9, 20, -8);
        emit_exec_number_guard(a, 9, 0, 11, string_key);
        emit_method_elem_dense(a, layout, slow, done);
    }
    a.bind(string_key);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 11, slow);
    emit_exec_payload(a, 9, 8); // LStr identity, pinned by a matching cache way
    emit_exec_word_load(a, 9, 20, -16);
    a.lsr_imm(11, 9, 48);
    a.movz(12, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_w(11, 12);
    a.b_cond(C_EQ, object);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 11, slow);
    a.ldr_imm(10, 19, 72);
    a.ldr_imm(10, 10, il.string_proto as u32); // active realm, never embedded prototype identity
    a.b(receiver);
    a.bind(object);
    emit_exec_payload(a, 9, 10);
    a.bind(receiver);
    a.add_imm(11, 10, object_data);
    a.ldr_w_imm(13, 11, shape);
    a.ldr_imm(12, 19, 72);
    a.ldr_w_imm(6, 12, il.computed_read_mask as u32);
    a.ldr_imm(12, 12, il.computed_read_sets as u32);
    a.cbz(12, true, slow);
    a.lsr_imm(7, 8, 3);
    a.lsr_imm(14, 7, 7);
    a.logic_x(2, 7, 7, 14);
    a.logic_x(2, 7, 7, 13);
    a.logic_w(0, 7, 7, 6);
    a.mov_imm64(14, ComputedReadCache::SET_STRIDE as u64);
    a.madd(12, 7, 14, 12);
    let probe = a.new_label();
    let next = a.new_label();
    let hit = a.new_label();
    a.movz(15, ComputedReadCache::WAYS as u32, 0);
    a.bind(probe);
    a.ldr_imm(14, 12, ComputedReadCache::KEY_OFF as u32);
    a.cmp_reg_x(14, 8);
    a.b_cond(C_NE, next);
    a.ldr_w_imm(
        14,
        12,
        ComputedReadCache::STATE_OFF as u32 + IC_OFF_RECV_SHAPE,
    );
    a.cmp_reg_w(14, 13);
    a.b_cond(C_EQ, hit);
    a.bind(next);
    a.add_imm(12, 12, ComputedReadCache::WAY_STRIDE as u32);
    a.sub_imm(15, 15, 1);
    a.cbnz(15, false, probe);
    a.b(slow);
    a.bind(hit);
    a.add_imm(12, 12, ComputedReadCache::STATE_OFF as u32);
    a.ldrb_imm(7, 12, IC_OFF_DEPTH);
    a.cmp_imm_w(7, 3);
    a.b_cond(C_HI, slow); // array key-checked holders, absence and deep/exotic paths remain checked
    let ordinary = a.new_label();
    a.ldrb_imm(14, 11, exotic);
    branch_exotic_ordinary(a, layout, 14, 16, ordinary);
    a.cmp_imm_w(14, layout.exotic_strwrap_tag as u32);
    a.b_cond(C_EQ, ordinary);
    a.cmp_imm_w(14, layout.exotic_array_tag as u32);
    a.b_cond(C_NE, slow);
    a.cbz(7, false, slow); // an array is allowed only below a named-property holder
    a.bind(ordinary);
    a.ldrb_imm(14, 11, plain);
    a.cbz(14, false, slow);
    let load = a.new_label();
    let holder = a.new_label();
    a.cbz(7, false, load);
    for hop in 1..=3 {
        a.ldr_imm(17, 11, layout.obj_proto as u32);
        a.cbz(17, true, slow);
        a.add_imm(11, 17, object_data);
        let hop_ordinary = a.new_label();
        a.ldrb_imm(14, 11, exotic);
        branch_exotic_ordinary(a, layout, 14, 16, hop_ordinary);
        a.cmp_imm_w(14, layout.exotic_strwrap_tag as u32);
        a.b_cond(C_NE, slow);
        a.bind(hop_ordinary);
        a.ldrb_imm(14, 11, plain);
        a.cbz(14, false, slow);
        a.ldr_w_imm(14, 11, shape);
        a.cmp_imm_w(7, hop);
        a.b_cond(C_EQ, holder);
        if hop < 3 {
            a.ldrb_imm(16, 12, IC_OFF_MID_OK);
            a.logic_imm_w(0, 16, 16, asm::logical_imm_w(1 << (hop - 1)).unwrap());
            a.cbz(16, false, slow);
            a.ldr_w_imm(
                16,
                12,
                if hop == 1 {
                    IC_OFF_MID_SHAPE
                } else {
                    IC_OFF_MID2_SHAPE
                },
            );
            a.cmp_reg_w(14, 16);
            a.b_cond(C_NE, slow);
        }
    }
    a.bind(holder);
    a.ldr_w_imm(16, 12, IC_OFF_HOLDER_SHAPE);
    a.cmp_reg_w(14, 16);
    a.b_cond(C_NE, slow);
    a.bind(load);
    a.ldr_w_imm(13, 12, IC_OFF_SLOT);
    fields_len(a, layout, 16, 11);
    a.cmp_reg_x(13, 16);
    a.b_cond(C_HS, slow);
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, layout.entry_size as u64);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 9, 15, layout.entry_accessor as u32, slow);
    a.ldur(12, 15, layout.entry_value as i32);
    // The key is known only at run time: it may be `prototype` on a function.
    property_word_guard(a, 12, 13, slow);
    // The method overwrites the key; the original receiver stays for Call.
    emit_exec_word_store(a, 12, 20, -8);
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, H_GET_METHOD_ELEM, pc, l_unwind);
    a.bind(done);
}

/// Shared call-site lowering for canonical code and optimized CFG regions. A region
/// must not replace a warmed direct call/inline intrinsic with a generic Rust dispatch.
/// Both entries own canonical operands and use the same guarded IC and unwind contract.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
fn emit_call_inline(
    a: &mut asm::Asm,
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    pc: usize,
    fast: u32,
    array_intrinsics_on: bool,
    function_call_intrinsic_on: bool,
    l_unwind: usize,
    l_direct_finish: usize,
) {
    a.mark_push("call");
    emit_call_inline_marked(
        a,
        chunk,
        layout,
        ilayout,
        pc,
        fast,
        array_intrinsics_on,
        function_call_intrinsic_on,
        l_unwind,
        l_direct_finish,
    );
    a.mark_pop();
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
fn emit_call_inline_marked(
    a: &mut asm::Asm,
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    pc: usize,
    fast: u32,
    array_intrinsics_on: bool,
    function_call_intrinsic_on: bool,
    l_unwind: usize,
    l_direct_finish: usize,
) {
    use crate::bytecode::Op;
    let ops = chunk.jit_ops();
    let op = &ops[pc];
    let (Op::Call(argc, c) | Op::CallWithThis(argc, c)) = op else {
        unreachable!("call emitter requires a call opcode");
    };
    let direct_on = fast & (1 << 20) != 0 && crate::bytecode::direct_shared_context_enabled();
    let inline_probe = fast & 524288 != 0;
    let slow = a.new_label();
    let done = a.new_label();
    if inline_probe {
        let depth = *argc as u32 + 1; // callee sits under the args
        let off = depth as i32 * -8;
        if depth <= 65 {
            let ic0 = chunk.jit_call_cache_ptr(*c);
            if off >= -256 {
                emit_exec_word_load(a, 9, 20, off);
            } else {
                a.sub_imm(9, 20, depth * 8);
                a.ldr_imm(9, 9, 0);
            }
            emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, slow);
            // x10 = the callee's payload, its cell address; x13 = the same word, its
            // `Gc::as_ptr` identity, which the ways record.
            emit_exec_payload(a, 9, 10);
            a.mov(13, 10);
            // Probe ALL 4 ways (a stable polymorphic site — e.g. one dispatch
            // loop over a handful of receiver classes — otherwise pays the full
            // helper on every call that isn't way 1): x12 = entry cursor,
            // w14 = ways left, w15 = live epoch, x17 = ctx.genv.
            a.mov_imm64(12, ic0 as u64);
            a.mov_imm64(11, &crate::bytecode::CALL_IC_EPOCH as *const _ as u64);
            a.ldr_w_imm(15, 11, 0);
            // LDR W15 zero-extends the u32 epoch: use W arithmetic, preserve X15.
            a.cmn_imm_w(15, 1);
            a.b_cond(C_EQ, slow); // exhausted generations are never raw-cache proofs
            a.ldr_imm(17, 19, 64); // ctx.genv
            a.movz(14, crate::bytecode::CALL_IC_WAYS as u32, 0);
            let l_probe = a.new_label();
            let l_next = a.new_label();
            let l_hit = a.new_label();
            let l_primary_hit = a.new_label();
            a.bind(l_probe);
            a.ldur(11, 12, 0); // ic.callee (a `Gc::as_ptr` identity)
            a.cmp_reg_x(13, 11);
            a.b_cond(C_NE, l_next);
            a.ldr_w_imm(11, 12, 56); // ic.epoch
            a.cmp_reg_w(11, 15);
            a.b_cond(C_NE, l_next);
            a.ldr_imm(11, 12, 32); // ic.global_env
            a.cmp_reg_x(11, 17);
            a.b_cond(C_EQ, l_primary_hit);
            a.bind(l_next);
            // entry stride (size compile-asserted below the JitCtx asserts)
            let stride = std::mem::size_of::<std::cell::Cell<crate::bytecode::CallIc>>();
            a.add_imm(12, 12, stride as u32);
            a.sub_imm(14, 14, 1);
            a.cbnz(14, false, l_probe);
            // Secondary probes (the engine-wide identity overflow cache, then code-keyed ways):
            // out of line in the chunk's shared stub. The primary loop left x12 exactly
            // CALL_IC_WAYS strides past the site's first way.
            if shared_stubs_enabled() {
                a.sub_imm(12, 12, (crate::bytecode::CALL_IC_WAYS * stride) as u32);
                let stub = a.shared_stub(SharedStub::CallSecondary.key());
                a.bl_label(stub);
                a.cbz(9, false, slow);
                a.b(l_hit);
            } else {
                a.mov_imm64(6, ic0 as u64);
                emit_call_secondary_probes(a, layout, ilayout, l_hit, slow);
            }
            // x15 = the hit way (ways-left counter → index), kept live through
            // the direct sequence's NO-MUTATION gate checks (they never touch
            // x15; every route into hit_slow happens before any blr).
            a.bind(l_primary_hit);
            a.movz(15, crate::bytecode::CALL_IC_WAYS as u32, 0);
            a.sub_reg(15, 15, 14);
            a.bind(l_hit);
            let with_this = matches!(op, Op::CallWithThis(..));
            let hit_slow = a.new_label();
            let ordinary_hit = a.new_label();
            // A user function (and an unrecognized native) has intrinsic id zero.
            // After the live IC proof, bypass every native-id comparison at once.
            // Keep the hit entry and way registers for the ordinary call below.
            if with_this
                && ((*argc == 1 && layout.valid)
                    || (*argc == 0 && array_intrinsics_on)
                    || ((1..=8).contains(argc) && function_call_intrinsic_on)
                    || *argc == 2)
            {
                a.ldrb_imm(9, 12, 96);
                a.cbz(9, false, ordinary_hit);
            }
            // Inline intrinsics: a native entry the template can finish without
            // leaving machine code. charCodeAt on a known-ASCII receiver with an
            // exact in-bounds u32 index is a byte load — meriyah-style scanners
            // make millions of these per parse. Any miss (intrinsic id, receiver
            // tag/hint, index shape, bounds) takes the H_CALL_HIT form, whose Rust
            // side handles native entries generally.
            if with_this && *argc == 1 && layout.valid {
                let discard = matches!(ops.get(pc + 1), Some(Op::Pop));
                let no_intr = a.new_label();
                if use_shared_stub(a) {
                    let stub = a.shared_stub(
                        SharedStub::Intrinsics1 {
                            array_push: array_intrinsics_on,
                            discard,
                        }
                        .key(),
                    );
                    emit_intrinsic_stub_call(a, pc as u32, stub, no_intr, hit_slow, done, l_unwind);
                } else {
                    emit_call_intrinsics1(
                        a,
                        layout,
                        IntrinsicPc::Imm(pc as u32),
                        array_intrinsics_on,
                        discard,
                        no_intr,
                        hit_slow,
                        done,
                        l_unwind,
                    );
                }
                a.bind(no_intr);
            }
            if with_this && *argc == 0 && array_intrinsics_on {
                let no_intr = a.new_label();
                a.ldrb_imm(9, 12, 96);
                a.cmp_imm_w(9, crate::bytecode::INTRINSIC_ARRAY_POP as u32);
                a.b_cond(C_NE, no_intr);
                emit_exec_word_load(a, 9, 20, -16);
                emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
                a.mov(0, 19);
                a.movz(1, pc as u32, 0);
                a.movk(1, crate::bytecode::INTRINSIC_ARRAY_POP as u32, 1);
                a.mov(2, 20);
                a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
                a.blr(16);
                a.mov(20, 0);
                a.cbnz(1, false, l_unwind);
                a.b(done);
                a.bind(no_intr);
            }
            if with_this && (1..=8).contains(argc) && function_call_intrinsic_on {
                let no_intr = a.new_label();
                a.ldrb_imm(9, 12, 96);
                a.cmp_imm_w(9, crate::bytecode::INTRINSIC_FUNCTION_CALL as u32);
                a.b_cond(C_NE, no_intr);
                let receiver_off = -((*argc as i32 + 2) * 8);
                emit_exec_word_load(a, 9, 20, receiver_off);
                emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
                a.mov(0, 19);
                a.movz(1, pc as u32, 0);
                a.movk(
                    1,
                    crate::bytecode::INTRINSIC_FUNCTION_CALL as u32 | ((*argc as u32) << 8),
                    1,
                );
                a.mov(2, 20);
                a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
                a.blr(16);
                a.mov(20, 0);
                a.cbnz(1, false, l_unwind);
                a.b(done);
                a.bind(no_intr);
            }
            if with_this && *argc == 2 {
                let discard = matches!(ops.get(pc + 1), Some(Op::Pop));
                let no_intr = a.new_label();
                if use_shared_stub(a) {
                    let stub = a.shared_stub(SharedStub::Intrinsics2 { discard }.key());
                    emit_intrinsic_stub_call(a, pc as u32, stub, no_intr, hit_slow, done, l_unwind);
                } else {
                    emit_call_intrinsics2(
                        a,
                        layout,
                        IntrinsicPc::Imm(pc as u32),
                        discard,
                        no_intr,
                        hit_slow,
                        done,
                        l_unwind,
                    );
                }
                a.bind(no_intr);
            }
            a.bind(ordinary_hit);
            // Direct shared-ctx call: its own gate misses land on `hit_slow` =
            // the H_CALL_HIT form below.
            if direct_on {
                let attempted_off = chunk.jit_inline_attempted_off();
                let runs_off = chunk.jit_runs_off();
                let retry_off = chunk.jit_inline_retry_at_off();
                let argc = *argc as usize;
                if use_shared_stub(a)
                    && direct_call_supported(
                        ilayout,
                        layout,
                        attempted_off,
                        runs_off,
                        retry_off,
                        argc,
                    )
                {
                    // The chunk's shared sequence for this arity and receiver form.
                    let stub = a.shared_stub(
                        SharedStub::DirectCall {
                            argc: argc as u8,
                            with_this,
                        }
                        .key(),
                    );
                    a.bl_label(stub);
                    let not_returned = a.new_label();
                    const _: () = assert!(DIRECT_CALL_RETURNED == 0);
                    a.cbnz(9, false, not_returned);
                    a.b(done);
                    // Never executed: the stub reads the call's position two instructions past
                    // its return address (see `DirectCallSite::AfterReturn`). Neither branch
                    // above can widen: both targets are in range.
                    a.word(chunk.call_position(*c));
                    a.bind(not_returned);
                    a.cmp_imm_w(9, DIRECT_CALL_THREW);
                    a.b_cond(C_EQ, l_unwind);
                    // A declined gate falls through to the H_CALL_HIT form below.
                } else {
                    emit_direct_call(
                        a,
                        ilayout,
                        layout,
                        attempted_off,
                        runs_off,
                        retry_off,
                        argc,
                        with_this,
                        hit_slow,
                        l_unwind,
                        done,
                        l_direct_finish,
                        DirectCallSite::Position(chunk.call_position(*c)),
                    );
                }
            }
            a.bind(hit_slow);
            // Secondary ways are not indices into the caller's four primary
            // entries. A non-direct/non-intrinsic secondary hit re-probes through
            // the checked helper; it must never alias primary way zero. A code-keyed
            // hit's tagged index still names its primary way in the low bits.
            a.cmp_imm_w(15, crate::bytecode::CALL_IC_WAYS as u32);
            a.b_cond(C_EQ, slow);
            a.mov(0, 19);
            // x1 = pc | way << 16 (pcs are < 65536: every helper call encodes
            // the pc as one movz)
            a.movz(1, pc as u32, 0);
            a.add_shifted(1, 1, 15, 16);
            a.mov(2, 20);
            a.ldr_imm(16, 21, (H_CALL_HIT * 8) as u32);
            a.blr(16);
            a.mov(20, 0);
            a.cbnz(1, false, l_unwind);
            a.b(done);
        }
    }
    a.bind(slow);
    a.mov(0, 19);
    a.movz(1, pc as u32, 0);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_CALL * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.bind(done);
    // Every completed form of the call (H_CALL, H_CALL_HIT, an intrinsic, a direct
    // call) lands here with its result pushed. A native or intrinsic can allocate after the
    // call's last safepoint.
    emit_helper_end_poll(a, l_unwind);
}
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
thread_local! { pub(super) static TEST_FAST_MATH_INTRINSICS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
extern "C" fn record_fast_math_intrinsic() {
    TEST_FAST_MATH_INTRINSICS.with(|count| count.set(count.get() + 1));
}

#[cfg(all(test, target_arch = "aarch64"))]
thread_local! { pub(super) static TEST_CHAIN_PROPERTY_STORES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(all(test, target_arch = "aarch64"))]
extern "C" fn record_chain_property_store() {
    TEST_CHAIN_PROPERTY_STORES.with(|count| count.set(count.get() + 1));
}

/// Inline Math numeric intrinsics (ECMA-262 §21.3.2: floor, ceil, trunc, abs, fround, clz32,
/// imul, max, min) at a `CallWithThis(argc)` hit whose call IC proved the exact builtin
/// (x9 = its `INTRINSIC_MATH_*` id, x10 = the callee's cell address). Stack, top last:
/// `[this, callee, args...]`. Every guard precedes the single commit: all arguments must be
/// Numbers (so ToNumber/ToInt32 run no author code). The exact builtin identity permits any
/// receiver, which these Math operations ignore. The result Number replaces `this`.
/// Clobbers x9, x11, x13, x14, x16, x17, d0, d1; preserves x10, x12 and x15 for the ordinary
/// call on a decline.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_math_intrinsic(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    argc: u32,
    no_intr: usize,
    hit_slow: usize,
    done: usize,
) {
    use crate::bytecode::{
        INTRINSIC_MATH_ABS, INTRINSIC_MATH_CEIL, INTRINSIC_MATH_CLZ32, INTRINSIC_MATH_FLOOR,
        INTRINSIC_MATH_FROUND, INTRINSIC_MATH_IMUL, INTRINSIC_MATH_MAX, INTRINSIC_MATH_MIN,
        INTRINSIC_MATH_TRUNC,
    };
    if !layout.valid {
        a.b(no_intr);
        return;
    }
    // Operand slots: argc arguments, the callee below them, `this` below that.
    let arg0 = -(argc as i32) * 8;
    let this_off = arg0 - 16;
    // Arguments: both Numbers, in d0/d1.
    emit_exec_word_load(a, 13, 20, arg0);
    emit_exec_number_guard(a, 13, 0, 16, hit_slow);
    if argc == 2 {
        emit_exec_word_load(a, 13, 20, arg0 + 8);
        emit_exec_number_guard(a, 13, 1, 16, hit_slow);
    }
    // Compute (d0 <- result). Declines here still precede every state change.
    let commit = a.new_label();
    let ops: &[u8] = if argc == 1 {
        &[
            INTRINSIC_MATH_FLOOR,
            INTRINSIC_MATH_CEIL,
            INTRINSIC_MATH_TRUNC,
            INTRINSIC_MATH_ABS,
            INTRINSIC_MATH_FROUND,
            INTRINSIC_MATH_CLZ32,
        ]
    } else {
        &[INTRINSIC_MATH_IMUL, INTRINSIC_MATH_MAX, INTRINSIC_MATH_MIN]
    };
    let labels: Vec<usize> = ops.iter().map(|_| a.new_label()).collect();
    for (&id, &label) in ops.iter().zip(&labels) {
        a.cmp_imm_w(9, id as u32);
        a.b_cond(C_EQ, label);
    }
    a.b(hit_slow);
    for (&id, &label) in ops.iter().zip(&labels) {
        a.bind(label);
        match id {
            INTRINSIC_MATH_FLOOR => a.frintm(0, 0),
            INTRINSIC_MATH_CEIL => a.frintp(0, 0),
            INTRINSIC_MATH_TRUNC => a.frintz(0, 0),
            INTRINSIC_MATH_ABS => a.fabs(0, 0),
            INTRINSIC_MATH_FROUND => {
                a.fcvt_float_width(0, 0, true); // FCVT S0, D0 (round to nearest, ties to even)
                a.fcvt_float_width(0, 0, false); // FCVT D0, S0
            }
            INTRINSIC_MATH_CLZ32 => {
                if !jscvt_available() {
                    a.b(hit_slow);
                    continue;
                }
                a.fjcvtzs_w_d(16, 0); // ToUint32 has the same bits as ToInt32
                a.clz_w(16, 16);
                a.ucvtf_d_w(0, 16);
            }
            INTRINSIC_MATH_IMUL => {
                if !jscvt_available() {
                    a.b(hit_slow);
                    continue;
                }
                a.fjcvtzs_w_d(16, 0);
                a.fjcvtzs_w_d(17, 1);
                a.mul_w(16, 16, 17);
                a.scvtf_d_w(0, 16);
            }
            INTRINSIC_MATH_MAX => a.fminmax(true, 0, 0, 1),
            _ => a.fminmax(false, 0, 0, 1),
        }
        a.b(commit);
    }
    // Commit: replace `this` with the result.
    a.bind(commit);
    emit_exec_number_store(a, 0, 20, this_off, 9);
    a.sub_imm(20, 20, (argc + 1) * 8);
    #[cfg(test)]
    regions::emit_lowering_event(a, record_fast_math_intrinsic as *const () as u64);
    a.b(done);
}

/// Where an intrinsic block finds its call site's bytecode pc for the helper's operand word:
/// an immediate in the site's own code, or w8 in the chunk's shared intrinsic stub.
#[derive(Clone, Copy)]
enum IntrinsicPc {
    Imm(u32),
    InW8,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_intrinsic_pc(a: &mut asm::Asm, pc: IntrinsicPc) {
    match pc {
        IntrinsicPc::Imm(pc) => a.movz(1, pc, 0),
        IntrinsicPc::InW8 => a.mov(1, 8),
    }
}

/// The one-argument method intrinsics of a `CallWithThis(1)` site after its live call-IC hit
/// (x10 = callee payload, x12 = the hit way, x15 = its index; preserved on every exit except a
/// completed or throwing intrinsic). `no_intr` continues with the ordinary call, `hit_slow` with
/// the H_CALL_HIT form; `done`/`l_unwind` follow a completed/throwing intrinsic. `discard`: the
/// result is popped (RegExp#exec and String#split elide it). Emitted at the site, or once per
/// chunk in [`SharedStub::Intrinsics1`].
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
fn emit_call_intrinsics1(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: IntrinsicPc,
    array_push: bool,
    discard: bool,
    no_intr: usize,
    hit_slow: usize,
    done: usize,
    l_unwind: usize,
) {
    let char_at = a.new_label();
    let char_code = a.new_label();
    let char_code_helper = a.new_label();
    let code_point = a.new_label();
    let sqrt = a.new_label();
    let is_nan = a.new_label();
    let from_char_code = a.new_label();
    let regexp_exec = discard.then(|| a.new_label());
    let string_split = discard.then(|| a.new_label());
    let array_push = array_push.then(|| a.new_label());
    a.ldrb_imm(9, 12, 96); // ic.intrinsic (offset compile-asserted)
    let math1 = a.new_label();
    a.sub_imm(16, 9, crate::bytecode::INTRINSIC_MATH_FLOOR as u32);
    a.cmp_imm_x(
        16,
        (crate::bytecode::INTRINSIC_MATH_CLZ32 - crate::bytecode::INTRINSIC_MATH_FLOOR) as u32 + 1,
    );
    a.b_cond(C_LO, math1);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_CHAR_AT as u32);
    a.b_cond(C_EQ, char_at);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_CHAR_CODE_AT as u32);
    a.b_cond(C_EQ, char_code);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_CODE_POINT_AT as u32);
    a.b_cond(C_EQ, code_point);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_MATH_SQRT as u32);
    a.b_cond(C_EQ, sqrt);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_IS_NAN as u32);
    a.b_cond(C_EQ, is_nan);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_FROM_CHAR_CODE as u32);
    a.b_cond(C_EQ, from_char_code);
    if let Some(array_push) = array_push {
        a.cmp_imm_w(9, crate::bytecode::INTRINSIC_ARRAY_PUSH as u32);
        a.b_cond(C_EQ, array_push);
    }
    if let Some(regexp_exec) = regexp_exec {
        a.cmp_imm_w(9, crate::bytecode::INTRINSIC_REGEXP_EXEC_DISCARD as u32);
        a.b_cond(C_EQ, regexp_exec);
    }
    if let Some(string_split) = string_split {
        a.cmp_imm_w(9, crate::bytecode::INTRINSIC_STRING_SPLIT_DISCARD as u32);
        a.b_cond(C_EQ, string_split);
    }
    a.b(no_intr);

    // String#charAt(number): exact builtin identity is already proven.
    // The dedicated helper handles truncation, UTF-16 units, and the
    // interned ASCII result while consuming the three operands directly.
    a.bind(char_at);
    emit_exec_word_load(a, 9, 20, -24);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_number_guard(a, 9, 0, 16, hit_slow);
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_CHAR_AT as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    // Exact builtin identity is proven by the live call IC. Numeric
    // indices need no author conversion; all UTF-16 cases use the
    // compact helper with the established native activation boundaries.
    a.bind(code_point);
    emit_exec_word_load(a, 9, 20, -24);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_number_guard(a, 9, 0, 16, hit_slow);
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_CODE_POINT_AT as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    a.bind(char_code);
    // receiver: Str; index: Num. Every other miss below (no ASCII hint, a
    // fractional or out-of-range index) is still a String receiver with a
    // Number index, which the operand-only helper finishes.
    emit_exec_word_load(a, 9, 20, -24);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_number_guard(a, 9, 0, 16, hit_slow);
    emit_exec_word_load(a, 11, 20, -24);
    emit_exec_payload(a, 11, 11);
    a.ldr_w_imm(14, 11, crate::lstr::CAP_OFF as u32);
    a.lsr_imm(14, 14, 31);
    a.cbz(14, false, char_code_helper);
    // index: exact u32
    a.ldur_d(0, 20, -8);
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, char_code_helper);
    // bounds (ASCII: byte index == unit index); OOB answers NaN in the
    // helper
    a.ldr_w_imm(14, 11, crate::lstr::LEN_OFF as u32);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, char_code_helper);
    // ---- commit: byte load, Num over the receiver slot ----
    a.add_imm(16, 11, crate::lstr::DATA_OFF as u32);
    a.ldrb_reg(16, 16, 9);
    a.ucvtf_d_w(0, 16);
    emit_exec_number_store(a, 0, 20, -24, 9);
    a.sub_imm(20, 20, 16);
    a.b(done);

    // String receiver, Number index: UTF-16 units of a non-ASCII receiver,
    // truncation and out-of-range indices in the operand-only helper.
    a.bind(char_code_helper);
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_CHAR_CODE_AT as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    // isNaN(number) / Number.isNaN(number): both answer SameValue(n, NaN) for a
    // Number argument (ToNumber is the identity). The exact builtin ignores its receiver.
    a.bind(is_nan);
    {
        emit_exec_word_load(a, 9, 20, -8);
        emit_exec_number_guard(a, 9, 0, 16, hit_slow);
        // ---- commit: Boolean over the receiver slot ----
        a.fcmp(0, 0);
        a.cset_w(9, C_VS);
        a.mov_imm64(16, crate::value::PACK_BOOL);
        a.logic_x(1, 9, 9, 16);
        emit_exec_word_store(a, 9, 20, -24);
        a.sub_imm(20, 20, 16);
        #[cfg(test)]
        regions::emit_lowering_event(a, record_fast_math_intrinsic as *const () as u64);
        a.b(done);
    }

    // String.fromCharCode(number): ToUint16 and the interned one-unit strings in
    // the operand-only helper.
    a.bind(from_char_code);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_number_guard(a, 9, 0, 16, hit_slow);
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_FROM_CHAR_CODE as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    // Math.sqrt(number): the call IC already proved builtin identity.
    // All receiver kinds are accepted by the exact builtin.
    a.bind(math1);
    emit_math_intrinsic(a, layout, 1, no_intr, hit_slow, done);

    a.bind(sqrt);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_number_guard(a, 9, 0, 16, hit_slow);
    a.ldur_d(0, 20, -8);
    a.fsqrt(0, 0);
    emit_exec_number_store(a, 0, 20, -24, 9);
    a.sub_imm(20, 20, 16);
    #[cfg(test)]
    regions::emit_lowering_event(a, record_fast_math_intrinsic as *const () as u64);
    a.b(done);

    // Array#push(value): builtin identity is proven by the call IC. The
    // helper moves `value` into dense storage after live array/prototype/
    // length guards, and restores the operand before the exact builtin on
    // any miss.
    if let Some(array_push) = array_push {
        a.bind(array_push);
        emit_exec_word_load(a, 9, 20, -24);
        emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
        a.mov(0, 19);
        emit_intrinsic_pc(a, pc);
        a.movk(1, crate::bytecode::INTRINSIC_ARRAY_PUSH as u32, 1);
        a.mov(2, 20);
        a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
        a.blr(16);
        a.mov(20, 0);
        a.cbnz(1, false, l_unwind);
        a.b(done);
    }

    if let Some(regexp_exec) = regexp_exec {
        a.bind(regexp_exec);
        // Exact built-in identity is already proven by the call IC.
        // The helper additionally validates the ordinary RegExp object
        // and lastIndex shape before taking its allocation-free path.
        emit_exec_word_load(a, 9, 20, -24);
        emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
        emit_exec_word_load(a, 9, 20, -8);
        emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
        a.mov(0, 19);
        emit_intrinsic_pc(a, pc);
        a.movk(1, crate::bytecode::INTRINSIC_REGEXP_EXEC_DISCARD as u32, 1);
        a.mov(2, 20);
        a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
        a.blr(16);
        a.mov(20, 0);
        a.cbnz(1, false, l_unwind);
        a.b(done);
    }

    if let Some(string_split) = string_split {
        a.bind(string_split);
        // The call IC proves String#split identity. The helper validates
        // the separator's complete RegExp protocol/species dependency
        // chain before eliding only the dead result allocations.
        emit_exec_word_load(a, 9, 20, -24);
        emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
        emit_exec_word_load(a, 9, 20, -8);
        emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
        a.mov(0, 19);
        emit_intrinsic_pc(a, pc);
        a.movk(1, crate::bytecode::INTRINSIC_STRING_SPLIT_DISCARD as u32, 1);
        a.mov(2, 20);
        a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
        a.blr(16);
        a.mov(20, 0);
        a.cbnz(1, false, l_unwind);
        a.b(done);
    }

    a.b(no_intr);
}

/// The two-argument method intrinsics of a `CallWithThis(2)` site; the register contract and
/// exits are those of [`emit_call_intrinsics1`]. Emitted at the site, or once per chunk in
/// [`SharedStub::Intrinsics2`].
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
fn emit_call_intrinsics2(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: IntrinsicPc,
    discard: bool,
    no_intr: usize,
    hit_slow: usize,
    done: usize,
    l_unwind: usize,
) {
    let slice = a.new_label();
    let has_own = a.new_label();
    let apply = a.new_label();
    let replace = discard.then(|| a.new_label());
    a.ldrb_imm(9, 12, 96);
    let math2 = a.new_label();
    a.sub_imm(16, 9, crate::bytecode::INTRINSIC_MATH_IMUL as u32);
    a.cmp_imm_x(
        16,
        (crate::bytecode::INTRINSIC_MATH_MIN - crate::bytecode::INTRINSIC_MATH_IMUL) as u32 + 1,
    );
    a.b_cond(C_LO, math2);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_STRING_SLICE as u32);
    a.b_cond(C_EQ, slice);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_OBJECT_HAS_OWN as u32);
    a.b_cond(C_EQ, has_own);
    a.cmp_imm_w(9, crate::bytecode::INTRINSIC_FUNCTION_APPLY as u32);
    a.b_cond(C_EQ, apply);
    if let Some(replace) = replace {
        a.cmp_imm_w(9, crate::bytecode::INTRINSIC_STRING_REPLACE_DISCARD as u32);
        a.b_cond(C_EQ, replace);
    }
    a.b(no_intr);

    a.bind(math2);
    emit_math_intrinsic(a, layout, 2, no_intr, hit_slow, done);

    // ASCII String#slice(start, end), both bounds already Numbers: no
    // user code or exotic conversion can run in the dedicated helper.
    a.bind(slice);
    emit_exec_word_load(a, 9, 20, -32);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
    emit_exec_word_load(a, 11, 20, -32);
    emit_exec_payload(a, 11, 11);
    a.ldr_w_imm(9, 11, crate::lstr::CAP_OFF as u32);
    a.lsr_imm(9, 9, 31);
    a.cbz(9, false, hit_slow);
    for off in [-16i32, -8] {
        emit_exec_word_load(a, 9, 20, off);
        emit_exec_number_guard(a, 9, 0, 16, hit_slow);
    }
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_STRING_SLICE as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    // Object.hasOwn(obj, string): the named intrinsic's implementation
    // is exactly an own-map lookup for this non-coercing argument shape.
    a.bind(has_own);
    emit_exec_word_load(a, 9, 20, -16);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_tag_guard(a, 9, crate::value::PACK_STR, 16, hit_slow);
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_OBJECT_HAS_OWN as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    // Function#apply(targetThis, arguments): builtin identity is already
    // proven. Restrict the intrinsic helper to an object target and
    // object list; it performs the ordinary/unmapped/dense guards before
    // moving entries directly into a compiled target frame.
    a.bind(apply);
    emit_exec_word_load(a, 9, 20, -32);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 16, hit_slow);
    a.mov(0, 19);
    emit_intrinsic_pc(a, pc);
    a.movk(1, crate::bytecode::INTRINSIC_FUNCTION_APPLY as u32, 1);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    a.b(done);

    if let Some(replace) = replace {
        a.bind(replace);
        for (off, tag) in [
            (-32i32, crate::value::PACK_STR),
            (-16, crate::value::PACK_OBJ),
            (-8, crate::value::PACK_STR),
        ] {
            emit_exec_word_load(a, 9, 20, off);
            emit_exec_tag_guard(a, 9, tag, 16, hit_slow);
        }
        a.mov(0, 19);
        emit_intrinsic_pc(a, pc);
        a.movk(
            1,
            crate::bytecode::INTRINSIC_STRING_REPLACE_DISCARD as u32,
            1,
        );
        a.mov(2, 20);
        a.ldr_imm(16, 21, (H_INTRINSIC * 8) as u32);
        a.blr(16);
        a.mov(20, 0);
        a.cbnz(1, false, l_unwind);
        a.b(done);
    }
    a.b(no_intr);
}

/// Enter the chunk's shared intrinsic stub from a call site (w8 = the site's pc) and route its
/// status: completed, no intrinsic (the ordinary call), declined (H_CALL_HIT) or thrown.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_intrinsic_stub_call(
    a: &mut asm::Asm,
    pc: u32,
    stub: usize,
    no_intr: usize,
    hit_slow: usize,
    done: usize,
    l_unwind: usize,
) {
    a.movz(8, pc, 0);
    a.bl_label(stub);
    const _: () = assert!(INTRINSIC_DONE == 0);
    a.cbz(9, false, done);
    a.cmp_imm_w(9, INTRINSIC_NONE);
    a.b_cond(C_EQ, no_intr);
    a.cmp_imm_w(9, INTRINSIC_DECLINED);
    a.b_cond(C_EQ, hit_slow);
    a.b(l_unwind);
}

/// The call site's secondary probes after its primary identity ways miss: the engine-wide
/// identity overflow cache, then the code-keyed ways (x6 = the site's first way). Inputs and hit
/// outputs are those of the primary probe (see `emit_call_overflow_probe`,
/// `emit_call_code_probe`). Clobbers x7, x9, x11, x12, x14 and x16 only.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_call_secondary_probes(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    hit: usize,
    miss: usize,
) {
    let code_keyed = a.new_label();
    emit_call_overflow_probe(a, ilayout, hit, code_keyed);
    a.bind(code_keyed);
    emit_call_code_probe(a, layout, hit, miss);
}

/// Code-keyed probe, after the identity probes miss: accept another closure of a primary way's
/// function. ECMA-262 OrdinaryFunctionCreate gives every evaluation of a function expression a
/// new object with its own [[Environment]] and [[Realm]], but all of them run the same code, so
/// a way filled for one closure serves the others once the live callee is proven to be an
/// ordinary user closure (`Callable::User`, `ic_plain`) of the way's function (`CallIc::func`,
/// which the site pins against address reuse — see `CallSite`), in the way's Realm and the
/// fill-time epoch/active Realm, and not under a `with` (see `Interp::call_jit_fast`). Class
/// constructors never match: every closure of a class constructor's function is a class
/// constructor, and such closures are never filled.
///
/// Inputs match the primary probe (x10 = stored callee pointer, w15 = live epoch, x17 = the
/// active Realm's global scope), plus x6 = the site's first way. On a hit, x12 points at the way
/// and x15 is its index tagged
/// with [`CALL_CODE_HIT`]; the call then runs the LIVE closure (the direct sequence and
/// `jit_call_hit` take its environment and identity from the callee, never from the way). No
/// calls, allocation or state changes occur.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_call_code_probe(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    hit: usize,
    miss: usize,
) {
    use std::mem::offset_of;
    let call_tag = layout.obj_from_cell + layout.obj_call;
    let call_payload = call_tag + 8;
    let ic_plain = layout.obj_from_cell + layout.obj_ic_plain;
    let fits8 = |o: usize| o.is_multiple_of(8) && o / 8 < 4096;
    let ic_func = offset_of!(crate::bytecode::CallIc, func);
    let ic_realm = offset_of!(crate::bytecode::CallIc, realm);
    if !(layout.valid
        && layout.call_probe_valid
        && layout.scope_parent_valid
        && call_tag < 4096
        && ic_plain < 4096
        && fits8(call_payload)
        && fits8(layout.user_func)
        && fits8(layout.user_env)
        && fits8(layout.user_realm)
        && layout.func_data_off < 4096
        && layout.scope_under_with < 4096)
    {
        a.b(miss);
        return;
    }
    a.ldrb_imm(9, 10, call_tag as u32);
    a.cmp_imm_w(9, crate::value::CALLABLE_USER_TAG as u32);
    a.b_cond(C_NE, miss);
    a.ldrb_imm(9, 10, ic_plain as u32);
    a.cbz(9, false, miss);
    a.ldr_imm(9, 10, call_payload as u32); // the closure cell (a `UserFn`)
    a.ldr_imm(11, 9, layout.user_func as u32);
    a.add_imm(11, 11, layout.func_data_off as u32); // the closure's function identity
    a.ldr_imm(16, 9, layout.user_realm as u32);
    a.mov(12, 6);
    a.movz(14, crate::bytecode::CALL_IC_WAYS as u32, 0);
    let probe = a.new_label();
    let next = a.new_label();
    let found = a.new_label();
    a.bind(probe);
    a.ldr_imm(7, 12, ic_func as u32);
    a.cmp_reg_x(7, 11);
    a.b_cond(C_NE, next);
    a.ldr_w_imm(7, 12, 56); // epoch
    a.cmp_reg_w(7, 15);
    a.b_cond(C_NE, next);
    a.ldr_imm(7, 12, 32); // global_env
    a.cmp_reg_x(7, 17);
    a.b_cond(C_NE, next);
    a.ldr_imm(7, 12, ic_realm as u32);
    a.cmp_reg_x(7, 16);
    a.b_cond(C_EQ, found);
    a.bind(next);
    let stride = std::mem::size_of::<std::cell::Cell<crate::bytecode::CallIc>>();
    a.add_imm(12, 12, stride as u32);
    a.sub_imm(14, 14, 1);
    a.cbnz(14, false, probe);
    a.b(miss);
    a.bind(found);
    a.ldr_imm(7, 9, layout.user_env as u32); // the closure's environment, its `Env::as_ptr`
    a.ldrb_imm(7, 7, layout.scope_under_with as u32);
    a.cbnz(7, false, miss);
    a.movz(15, crate::bytecode::CALL_IC_WAYS as u32 + CALL_CODE_HIT, 0);
    a.sub_reg(15, 15, 14);
    a.b(hit);
}

/// Way-index tag (x15 at a call template's hit label) marking a code-keyed hit (see
/// [`emit_call_code_probe`]). The direct sequence then installs the live closure's environment,
/// and the H_CALL_HIT form still reads the way from the index's low bits (`jit_call_hit`, which
/// also runs the live closure). The overflow probe's sentinel is exactly `CALL_IC_WAYS`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const CALL_CODE_HIT: u32 = 8;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const _: () = assert!(crate::bytecode::CALL_IC_WAYS.is_power_of_two());
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const _: () = assert!((crate::bytecode::CALL_IC_WAYS as u32) < CALL_CODE_HIT);

/// Secondary identity probe. Inputs match the primary probe: x13 = live callee identity,
/// w15 = epoch, x17 = realm, x10 = stored callee pointer. On hit x12 points at a CallIc and
/// w15 is the non-primary sentinel. No calls, allocation, GC or observable state changes occur.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_call_overflow_probe(
    a: &mut asm::Asm,
    layout: &crate::interpreter::InterpLayout,
    hit: usize,
    miss: usize,
) {
    if !layout.valid
        || !layout.call_overflow_sets.is_multiple_of(8)
        || layout.call_overflow_sets / 8 >= 4096
        || !layout.call_overflow_shift.is_multiple_of(4)
        || layout.call_overflow_shift / 4 >= 4096
    {
        a.b(miss);
        return;
    }
    a.ldr_imm(9, 19, 72); // live ctx.interp
    a.ldr_imm(12, 9, layout.call_overflow_sets as u32);
    a.cbz(12, true, miss);
    a.ldr_w_imm(9, 9, layout.call_overflow_shift as u32);
    a.mov_imm64(11, crate::bytecode::CALL_OVERFLOW_HASH);
    a.madd(11, 13, 11, 31);
    a.lsr_reg(11, 11, 9);
    a.mov_imm64(9, crate::bytecode::CallOverflow::SET_STRIDE as u64);
    a.madd(12, 11, 9, 12);
    a.movz(14, crate::bytecode::CALL_OVERFLOW_WAYS as u32, 0);
    let probe = a.new_label();
    let next = a.new_label();
    let found = a.new_label();
    a.bind(probe);
    a.ldr_imm(11, 12, 0);
    a.cmp_reg_x(11, 13);
    a.b_cond(C_NE, next);
    a.ldr_w_imm(11, 12, 56);
    a.cmp_reg_w(11, 15);
    a.b_cond(C_NE, next);
    a.ldr_imm(11, 12, 32);
    a.cmp_reg_x(11, 17);
    a.b_cond(C_EQ, found);
    a.bind(next);
    a.add_imm(12, 12, crate::bytecode::CallOverflow::ENTRY_STRIDE as u32);
    a.sub_imm(14, 14, 1);
    a.cbnz(14, false, probe);
    a.b(miss);
    a.bind(found);
    a.movz(15, crate::bytecode::CALL_IC_WAYS as u32, 0);
    a.b(hit);
}

/// Emission-time preconditions of [`emit_direct_call`]: every probed Interp/JitCtx offset the
/// sequence addresses exists and fits its addressing mode, and the arity is within the unrolled
/// argument-move ceiling (real-world call sites are overwhelmingly below 64 arguments).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn direct_call_supported(
    ilayout: &crate::interpreter::InterpLayout,
    layout: &crate::value::JitLayout,
    attempted_off: usize,
    runs_off: usize,
    retry_off: usize,
    argc: usize,
) -> bool {
    use std::mem::offset_of;
    let fits8 = |o: usize| o & 7 == 0 && o / 8 < 4096;
    let fits4 = |o: usize| o & 3 == 0 && o / 4 < 4096;
    let il = ilayout;
    il.valid
        && FRAME_WORDS_OFF < 4096
        && argc <= 64
        && layout.scope_parent_valid
        && layout.scope_parent.is_multiple_of(8)
        && layout.scope_parent / 8 < 4096
        && fits4(il.depth)
        && fits4(il.direct_call_depth)
        && fits4(il.gc_tick)
        && fits8(il.gc_next)
        && fits8(il.new_target)
        && fits8(il.fn_frames + il.fnf_ptr_word)
        && fits8(il.fn_frames + il.fnf_len_word)
        && fits8(il.fn_frames + il.fnf_cap_word)
        && fits8(il.frame_pool + il.fp_ptr_word)
        && fits8(il.frame_pool + il.fp_len_word)
        && fits8(il.new_target + 8)
        // The handlers Vec's length-word offset within JitCtx (per-instantiation, probed).
        && jit_handlers_len_offset().is_some_and(fits8)
        && fits8(offset_of!(JitCtx, ret))
        && fits8(offset_of!(JitCtx, live_objects))
        && attempted_off < 4096
        && fits4(runs_off)
        && fits4(retry_off)
}

/// The source of a direct call's position (see `crate::stack_trace`), which the sequence stores
/// in the callee's `FnFrame::call_site`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy)]
pub(crate) enum DirectCallSite {
    /// The sequence is emitted for one call site: its position is a constant.
    Position(u32),
    /// The sequence is a shared stub: each site places its position word two instructions past
    /// the stub's return address, after the unconditional branch that leaves the site (see
    /// `Asm::word`). Returning to the unmodified address keeps return prediction intact.
    AfterReturn,
}

/// The direct JIT→JIT call sequence, emitted after a guarded cache hit when the fill-time gates
/// allow it (see [`crate::bytecode::CallIc::direct`]). Everything the layered path does
/// survives — recursion depth, the `FnFrame`, constructing/new.target clearing, the callee's
/// own handler watermark — and its safepoint is the callee's entry poll (see
/// [`emit_interrupt_poll`]), which runs once the call has committed: the sequence checks no
/// allocation pressure or gc tick of its own. The callee runs on an activation record popped
/// from the frame pool, entered by a bare `blr`: no helper dispatch, no probe re-read, no
/// `run_moved`. Teardown (pool return, frame pop, tail drain) is the chunk's finish stub; the
/// sequence then restores the interpreter's construct state and either pushes the return value
/// or routes to the caller's unwind. Falls back to `hit_slow` (the H_CALL_HIT path) on any gate
/// failure, with NO state mutated.
///
/// Returns false (nothing emitted) when an emission-time precondition fails — the caller then
/// emits only the probe + H_CALL_HIT form.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_direct_call(
    a: &mut asm::Asm,
    ilayout: &crate::interpreter::InterpLayout,
    // A stored handle word is the cell's `Gc::as_ptr` identity: FnFrame.fn_ptr records the
    // callee's as it is (FnFrame::callee reconstructs a `Gc` from it).
    layout: &crate::value::JitLayout,
    // Byte offset of `Chunk::inline_attempted` (computed by the caller from its own chunk —
    // same monomorphized layout as the callee's): the sequence requires the callee's one-shot
    // recompile to have happened (or been attempted), else it keeps taking H_CALL_HIT, which
    // is what bumps jit_runs toward the trigger. A landed code2 bumps the epoch and refills
    // the site with the new chunk anyway.
    attempted_off: usize,
    runs_off: usize,
    retry_off: usize,
    argc: usize,
    with_this: bool,
    hit_slow: usize,
    l_unwind: usize,
    done: usize,
    // Label of the chunk's shared direct-finish stub (`emit_direct_finish_stub`).
    finish_stub: usize,
    // Where the call's source position comes from (the callee's `FnFrame::call_site`).
    call_site: DirectCallSite,
) -> bool {
    use std::mem::offset_of;
    if !direct_call_supported(ilayout, layout, attempted_off, runs_off, retry_off, argc) {
        return false;
    }
    let fits8 = |o: usize| o & 7 == 0 && o / 8 < 4096;
    let il = ilayout;
    const IC_ENV: i32 = 8;
    const IC_STRICT: u32 = 40;
    const IC_USES_THIS: u32 = 41;
    const IC_NPARAMS: u32 = 42;
    const IC_NSLOTS: u32 = 44;
    const IC_DIRECT: i32 = 46;
    const IC_CHUNK_RAW: i32 = 64;
    const IC_CODE_MEM: i32 = 72;
    const IC_PC_OFFS: i32 = 80;
    let cx_stack_base = offset_of!(JitCtx, stack_base) as u32;
    let cx_env_raw = offset_of!(JitCtx, env_raw) as u32;
    let cx_env_parent_raw = offset_of!(JitCtx, env_parent_raw) as u32;
    let cx_chunk = offset_of!(JitCtx, chunk) as u32;
    let cx_n_slots = offset_of!(JitCtx, n_slots) as u32;
    let cx_code_base = offset_of!(JitCtx, code_base) as u32;
    let cx_pc_offsets = offset_of!(JitCtx, pc_offsets) as u32;
    let cx_ret = offset_of!(JitCtx, ret) as u32;
    // The code-keyed probe is emitted only under these conditions (see `emit_call_code_probe`);
    // without it every hit is an identity hit and the way's recorded environment is the callee's.
    let call_env_from_callee = layout.call_probe_valid
        && fits8(layout.obj_from_cell + layout.obj_call + 8)
        && fits8(layout.user_env);

    // ---- checks (entry state: x12 = ic0 ptr, x10 = callee cell address; NO mutations) ----
    // direct bits 0 (no force resets) and 2 (recompile settled)
    a.ldurb(9, 12, IC_DIRECT);
    let field1b = asm::logical_imm_w(1).unwrap();
    a.logic_imm_w(0, 11, 9, field1b); // bit 0: no force resets
    a.cbz(11, false, hit_slow);
    let field8 = asm::logical_imm_w(8).unwrap();
    a.logic_imm_w(0, 11, 9, field8); // bit 3: frame fits FRAME_BUF
    a.cbz(11, false, hit_slow);
    // recompile settled (live chunk byte — see attempted_off)
    a.ldur(11, 12, IC_CHUNK_RAW);
    a.ldrb_imm(11, 11, attempted_off as u32);
    a.cbz(11, false, hit_slow);
    // needs_global (bit 1) requires a live ctx.global_body
    let no_glob = a.new_label();
    let field2 = asm::logical_imm_w(2).unwrap();
    a.logic_imm_w(0, 11, 9, field2);
    a.cbz(11, false, no_glob);
    a.ldr_imm(11, 19, 56); // ctx.global_body
    a.cbz(11, true, hit_slow);
    a.bind(no_glob);
    // Arity needs no gate: the sequence moves min(argc, n_params) arguments into their slots,
    // tags every remaining slot Undefined (the missing-argument binding), and leaves any
    // over-applied surplus on the caller's operand stack, which pops it after the call.
    // this binding: a this-using SLOPPY callee needs boxing/global fallback unless the
    // incoming receiver is already an object.
    a.ldrb_imm(9, 12, IC_USES_THIS);
    let this_ok = a.new_label();
    a.cbz(9, false, this_ok);
    a.ldrb_imm(9, 12, IC_STRICT);
    a.cbnz(9, false, this_ok);
    if with_this {
        a.sub_imm(9, 20, ((argc + 2) * 8) as u32);
        a.ldr_imm(9, 9, 0);
        emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, hit_slow);
    } else {
        a.b(hit_slow); // no receiver + sloppy this-user: global boxing → generic
    }
    a.bind(this_ok);
    // interp-side room: depth, gc tick, fn_frames capacity, frame pool
    a.ldr_imm(14, 19, 72); // ctx.interp
    a.ldr_w_imm(11, 14, il.depth as u32);
    a.ldr_w_imm(13, 14, il.direct_call_depth as u32);
    a.cmp_reg_x(11, 13); // w-load zero-extends; the compare stays 64-bit
    a.b_cond(C_HS, hit_slow);
    // Allocation pressure and host interruption are observed by the callee's entry safepoint
    // (see the prologue), after this call has committed.
    a.ldr_imm(16, 14, (il.fn_frames + il.fnf_len_word) as u32);
    a.ldr_imm(17, 14, (il.fn_frames + il.fnf_cap_word) as u32);
    a.cmp_reg_x(16, 17);
    a.b_cond(C_HS, hit_slow);
    a.ldr_imm(7, 14, (il.frame_pool + il.fp_len_word) as u32);
    a.cbz(7, true, hit_slow);

    // Late feedback can make a previously empty inline plan useful. Count only direct
    // entries with a pending retry; on the threshold call the helper owns the increment.
    // This gate runs after every other fallback check, before committing any call state.
    let retry_done = a.new_label();
    a.ldur(4, 12, IC_CHUNK_RAW);
    a.ldr_w_imm(5, 4, retry_off as u32);
    a.cbz(5, false, retry_done);
    a.ldr_w_imm(6, 4, runs_off as u32);
    a.add_imm(6, 6, 1);
    a.cmp_reg_w(6, 5);
    a.b_cond(C_HS, hit_slow);
    a.str_w_imm(6, 4, runs_off as u32);
    a.bind(retry_done);

    // x13 = the callee's [[Environment]] (`Env::as_ptr`), kept until the install below. An
    // identity hit's way recorded exactly this closure's environment; a code-keyed hit
    // (CALL_CODE_HIT in x15) matched another closure of the same function, so read the LIVE
    // closure's (x10 is the stored callee pointer).
    if call_env_from_callee {
        let code_env = a.new_label();
        let env_done = a.new_label();
        let code_bit = asm::logical_imm_w(CALL_CODE_HIT).unwrap();
        a.logic_imm_w(0, 9, 15, code_bit);
        a.cbnz(9, false, code_env);
        a.ldur(13, 12, IC_ENV);
        a.b(env_done);
        a.bind(code_env);
        a.ldr_imm(13, 10, (layout.obj_from_cell + layout.obj_call + 8) as u32);
        a.ldr_imm(13, 13, layout.user_env as u32);
        a.bind(env_done);
    } else {
        a.ldur(13, 12, IC_ENV);
    }

    // ---- mutations ----
    a.add_imm(11, 11, 1);
    a.str_w_imm(11, 14, il.depth as u32); // depth++ (u32 field)
                                          // FnFrame push: entry = ptr + len*24
    a.ldr_imm(6, 14, (il.fn_frames + il.fnf_ptr_word) as u32);
    a.movz(5, 24, 0);
    a.madd(6, 16, 5, 6);
    a.mov(4, 10);
    a.stur(4, 6, 0); // fn_ptr = the callee's as_ptr identity
    match call_site {
        DirectCallSite::Position(position) => {
            a.movz(5, position & 0xffff, 0);
            if position >> 16 != 0 {
                a.movk(5, position >> 16, 1);
            }
        }
        // Nothing has called out of this sequence yet: x30 is still the site's return address.
        DirectCallSite::AfterReturn => a.ldr_w_imm(5, 30, 8),
    }
    a.str_w_imm(5, 6, 8); // call_site
    a.ldrb_imm(5, 12, IC_STRICT);
    a.sturh(5, 6, 12); // strict, and construct = false (w5 is 0 or 1)
    a.stur(31, 6, 16); // extra = None (xzr)
    a.add_imm(16, 16, 1);
    a.str_imm(16, 14, (il.fn_frames + il.fnf_len_word) as u32);
    // Pop the callee's activation record: x9 = the record (and its JitCtx), x15 = its slot
    // storage. The callee runs on this record; the caller's context is never modified.
    a.sub_imm(7, 7, 1);
    a.str_imm(7, 14, (il.frame_pool + il.fp_len_word) as u32);
    a.ldr_imm(5, 14, (il.frame_pool + il.fp_ptr_word) as u32);
    a.ldr_x_lsl3(9, 5, 7);
    a.add_imm(15, 9, FRAME_WORDS_OFF as u32);
    // Move packed argument words byte-for-byte; there is no conversion or possible miss.
    // ECMA-262 OrdinaryCallBindThis/FunctionDeclarationInstantiation bind the first n_params
    // values; an over-applied call's surplus words stay on the caller's operand stack, which
    // pops them after the call.
    a.sub_imm(8, 20, (argc * 8) as u32);
    let surplus = a.new_label();
    let moved = a.new_label();
    a.ldrh_imm(6, 12, IC_NPARAMS);
    a.cmp_imm_w(6, argc as u32);
    a.b_cond(C_LO, surplus);
    for k in 0..argc {
        a.ldr_imm(4, 8, (k * 8) as u32);
        a.str_imm(4, 15, (k * 8) as u32);
    }
    a.movz(6, argc as u32, 0); // first slot left to initialize
    a.b(moved);
    a.bind(surplus);
    // w6 = n_params < argc: move exactly those arguments.
    {
        let move_loop = a.new_label();
        let move_done = a.new_label();
        a.movz(3, 0, 0);
        a.bind(move_loop);
        a.cmp_reg_w(3, 6);
        a.b_cond(C_HS, move_done);
        a.ldr_x_lsl3(4, 8, 3);
        a.add_shifted(16, 15, 3, 3);
        a.str_imm(4, 16, 0);
        a.add_imm(3, 3, 1);
        a.b(move_loop);
        a.bind(move_done);
    }
    a.bind(moved);
    // Initialize every remaining local (from slot w6) to a complete packed Undefined word.
    // The count w4 = n_slots - w6 is stored four words per iteration (two STPs), then a pair
    // and a single word for the remainder; x5 = n_slots survives for the installation below.
    a.ldrh_imm(5, 12, IC_NSLOTS);
    a.mov_imm64(17, crate::value::PACK_UNDEFINED);
    let init_done = a.new_label();
    a.subs_reg_w(4, 5, 6);
    a.b_cond(C_LS, init_done); // n_slots <= first uninitialized slot: nothing left
    a.add_shifted(3, 15, 6, 3);
    let init_pair = a.new_label();
    let init_single = a.new_label();
    let init_quad = a.new_label();
    a.lsr_imm_w(16, 4, 2);
    a.cbz(16, false, init_pair);
    a.bind(init_quad);
    a.stp_base(17, 17, 3, 0);
    a.stp_base(17, 17, 3, 16);
    a.add_imm(3, 3, 32);
    a.subs_imm_w(16, 16, 1);
    a.b_cond(C_NE, init_quad);
    a.bind(init_pair);
    a.logic_imm_w(0, 16, 4, asm::logical_imm_w(2).unwrap());
    a.cbz(16, false, init_single);
    a.stp_base(17, 17, 3, 0);
    a.add_imm(3, 3, 16);
    a.bind(init_single);
    a.logic_imm_w(0, 16, 4, asm::logical_imm_w(1).unwrap());
    a.cbz(16, false, init_done);
    a.str_imm(17, 3, 0);
    a.bind(init_done);

    // ---- install the callee's per-frame fields in its own record ----
    // Interpreter-constant fields (helpers, interp, counters, slots) were written when the
    // record was created; this/ret/error/handlers are vacant while it is pooled.
    // stack_base = slots + n_slots*8
    a.movz(4, 8, 0);
    a.madd(3, 5, 4, 15);
    a.str_imm(3, 9, cx_stack_base);
    // env_raw (selected into x13 above), and its parent cache from that environment: the
    // parent's stored word, its `Env::as_ptr`, or null.
    a.str_imm(13, 9, cx_env_raw);
    a.ldr_imm(6, 13, layout.scope_parent as u32);
    a.str_imm(6, 9, cx_env_parent_raw);
    a.ldur(4, 12, IC_CHUNK_RAW);
    a.str_imm(4, 9, cx_chunk);
    a.str_imm(5, 9, cx_n_slots);
    a.ldur(4, 12, IC_CODE_MEM);
    a.str_imm(4, 9, cx_code_base);
    a.ldur(4, 12, IC_PC_OFFS);
    a.str_imm(4, 9, cx_pc_offsets);
    // Same Realm as the caller (the probe compared genv): its global body and scope.
    a.ldr_imm(4, 19, 56);
    a.str_imm(4, 9, 56);
    a.ldr_imm(4, 19, 64);
    a.str_imm(4, 9, 64);
    if with_this {
        // Move the receiver's word as it is into the callee's [48] (its `this_word`), even when
        // the callee never reads `this`: it owns nothing, and the finish stub resets it.
        let receiver = ((argc + 2) * 8) as u32;
        if receiver <= 256 {
            a.ldur(4, 20, -(receiver as i32));
        } else {
            a.sub_imm(4, 20, receiver);
            a.ldr_imm(4, 4, 0);
        }
        a.str_imm(4, 9, 48);
    }
    // Save area: [record, constructing (byte), new_target (16B)]. constructing and new_target
    // live on the interpreter and are cleared for the callee.
    a.sub_imm(31, 31, 32);
    a.stur(9, 31, 0);
    let (construct_base, construct_offset) = byte_field_address(a, 14, il.constructing, 16);
    a.ldrb_imm(4, construct_base, construct_offset);
    a.stur(4, 31, 8);
    a.strb_imm(31, construct_base, construct_offset);
    a.ldr_imm(4, 14, il.new_target as u32);
    a.ldr_imm(5, 14, (il.new_target + 8) as u32);
    a.stur(4, 31, 16);
    a.stur(5, 31, 24);
    let (target_base, target_offset) = byte_field_address(a, 14, il.new_target, 16);
    let lexical_target = a.new_label();
    a.ldurb(4, 12, IC_DIRECT);
    let lexical_bit = asm::logical_imm_w(crate::bytecode::CALL_IC_LEXICAL_THIS as u32).unwrap();
    a.logic_imm_w(0, 4, 4, lexical_bit);
    a.cbnz(4, false, lexical_target);
    a.strb_imm(31, target_base, target_offset); // ordinary calls clear new.target
    a.bind(lexical_target); // arrows retain their lexical function environment's new.target

    // ---- run the callee on its own record ----
    a.mov(0, 9);
    a.ldur(16, 12, IC_CODE_MEM);
    a.blr(16);
    // w0 = 1 ok / 0 threw → w1 = threw for the finish stub; x10 = the callee record.
    let field1 = asm::logical_imm_w(1).unwrap();
    a.logic_imm_w(2, 1, 0, field1); // eor w1, w0, #1
    a.ldur(10, 31, 0);
    // Teardown (completion transfer, record return, frame pop, tail drain, depth--): one
    // shared per-chunk stub (see `emit_direct_finish_stub`) whose fast path never leaves
    // machine code. w8 = threw.
    a.bl_label(finish_stub);

    // ---- restore the interpreter's construct state ----
    a.ldr_imm(14, 19, 72); // ctx.interp (x14 was clobbered by the callee/helpers)
    a.ldur(4, 31, 8);
    let (construct_base, construct_offset) = byte_field_address(a, 14, il.constructing, 16);
    a.strb_imm(4, construct_base, construct_offset);
    a.ldur(4, 31, 16);
    a.ldur(5, 31, 24);
    a.str_imm(4, 14, il.new_target as u32);
    a.str_imm(5, 14, (il.new_target + 8) as u32);
    a.add_imm(31, 31, 32);

    // ---- pop the arguments and the callee (and skip the consumed this slot); dispatch on
    // threw ----
    let popped = ((argc + 1 + with_this as usize) * 8) as u32;
    a.sub_imm(20, 20, popped);
    a.cbnz(8, true, l_unwind); // threw → caller unwind (fields restored)
                               // Move the complete word; no decode/encode.
    a.ldr_imm(4, 19, cx_ret);
    a.stur(4, 20, 0);
    a.mov_imm64(5, crate::value::PACK_UNDEFINED);
    a.str_imm(5, 19, cx_ret);
    a.add_imm(20, 20, 8);
    #[cfg(test)]
    {
        a.mov_imm64(16, record_direct_packed_return as *const () as u64);
        a.blr(16);
    }
    a.b(done);
    true
}

#[cfg(all(test, target_arch = "aarch64"))]
thread_local! {
    static TEST_DIRECT_PACKED_RETURNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Numeric-chain property operations compiled with `this` as their receiver.
    static TEST_THIS_CHAIN_OPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, target_arch = "aarch64"))]
extern "C" fn record_direct_packed_return() {
    TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(count.get() + 1));
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod direct_call_tests {
    #[test]
    fn computed_method_native_probe_reads_live_values_and_preserves_owners() {
        use crate::value::{Object, PackedValue, Property, Value};
        unsafe extern "C" fn miss(
            ctx: *mut super::JitCtx,
            _pc: u32,
            sp: *mut PackedValue,
        ) -> super::SpFlag {
            unsafe {
                *ctx.cast::<usize>() += 1;
            }
            super::SpFlag { sp, flag: 0 }
        }
        let mut engine = crate::Engine::new();
        let il = crate::interpreter::interp_layout(&mut engine.interp);
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        assert!(super::get_method_inlinable(&layout));
        let mut asm = super::asm::Asm::new();
        asm.stp_pre(29, 30, -16);
        asm.stp_pre(19, 20, -16);
        asm.stp_pre(21, 22, -16);
        asm.mov(19, 0);
        asm.mov(20, 1);
        asm.mov(21, 2);
        let unwind = asm.new_label();
        super::emit_computed_method_inline(&mut asm, &layout, &il, 0, unwind);
        asm.bind(unwind);
        asm.mov(0, 20);
        asm.ldp_post(21, 22, 16);
        asm.ldp_post(19, 20, 16);
        asm.ldp_post(29, 30, 16);
        asm.ret();
        let bytes: Vec<u8> = asm
            .finish()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let code = super::ExecutableBuffer::from_bytes(&bytes).unwrap();
        let entry: unsafe extern "C" fn(
            *mut usize,
            *mut PackedValue,
            *const usize,
        ) -> *mut PackedValue = unsafe { std::mem::transmute(code.as_ptr()) };
        let mut helpers = [0usize; super::N_HELPERS];
        helpers[super::H_GET_METHOD_ELEM] = miss as *const () as usize;
        // Only the activation-relative Interp pointer is read on this no-call hit path.
        let mut ctx = [0usize; 16];
        ctx[std::mem::offset_of!(super::JitCtx, interp) / 8] =
            &mut *engine.interp as *mut _ as usize;
        for depth in 0..=3 {
            let key = crate::lstr::LStr::from(format!("method{depth}"));
            let holder = Object::new(None);
            holder
                .borrow_mut()
                .props
                .insert(key.as_str(), Property::plain(Value::Num(17.0)));
            let mut object = holder.clone();
            for _ in 0..depth {
                object = Object::new(Some(object));
            }
            let receiver = Value::Obj(object.clone());
            assert!(matches!(
                engine.interp.get_computed_property(&receiver, &key),
                Ok(Value::Num(17.0))
            ));
            for value in [Value::Num(29.0), Value::Str(key.clone()), receiver.clone()] {
                holder.borrow_mut().props.set_value(&key, value.clone());
                let mut operands = [
                    PackedValue::pack(receiver.clone()),
                    PackedValue::pack(Value::Str(key.clone())),
                ];
                let end = unsafe { operands.as_mut_ptr().add(2) };
                assert_eq!(
                    unsafe { entry(ctx.as_mut_ptr(), end, helpers.as_ptr()) },
                    end
                );
                assert_eq!(ctx[0], 0, "expected a native cache hit, depth {depth}");
                match (&operands[1].unpack(), &value) {
                    (Value::Num(a), Value::Num(b)) => assert_eq!(a, b),
                    (Value::Str(a), Value::Str(b)) => assert!(crate::lstr::LStr::ptr_eq(a, b)),
                    (Value::Obj(a), Value::Obj(b)) => assert!(crate::value::Gc::ptr_eq(a, b)),
                    _ => panic!("wrong computed value"),
                }
                drop(operands);
                // Owners carry no count: the probe leaves the key's, the holder's and
                // the receiver's cells intact, which a collection (and, in debug builds, its
                // heap verifier) confirms.
                engine.interp.gc_collect();
                assert_eq!(key.as_str(), format!("method{depth}"));
                assert!(holder.try_borrow_mut().is_ok() && object.try_borrow_mut().is_ok());
            }
            holder
                .borrow_mut()
                .props
                .get_mut(&key)
                .unwrap()
                .set_value(Value::Num(31.0));
            // Same string contents at a different allocation must miss the identity probe.
            let fresh = crate::lstr::LStr::from(key.as_str());
            let mut operands = [
                PackedValue::pack(receiver),
                PackedValue::pack(Value::Str(fresh)),
            ];
            unsafe {
                entry(
                    ctx.as_mut_ptr(),
                    operands.as_mut_ptr().add(2),
                    helpers.as_ptr(),
                );
            }
            assert_eq!(ctx[0], 1);
            ctx[0] = 0;
        }

        // The already-compiled probe must read the current table and mask after growth.
        // Populate real, stable property resolutions, including enough names to evict ways.
        let holder = Object::new(None);
        let keys: Vec<_> = (0..4096)
            .map(|n| crate::lstr::LStr::from(format!("growth{n}")))
            .collect();
        for (n, key) in keys.iter().enumerate() {
            holder
                .borrow_mut()
                .props
                .insert(key.as_str(), Property::plain(Value::Num(n as f64)));
        }
        let receiver = Value::Obj(holder.clone());
        let initial_mask = engine.interp.computed_reads.mask;
        for key in &keys {
            assert!(engine.interp.get_computed_property(&receiver, key).is_ok());
        }
        assert!(engine.interp.computed_reads.mask > initial_mask);
        let shape = holder.borrow().props.shape();
        for (n, key) in keys.iter().enumerate() {
            let cached = engine
                .interp
                .computed_reads
                .lookup(key.as_ptr() as usize, shape)
                .is_some();
            let mut operands = [
                PackedValue::pack(receiver.clone()),
                PackedValue::pack(Value::Str(key.clone())),
            ];
            let end = unsafe { operands.as_mut_ptr().add(2) };
            assert_eq!(
                unsafe { entry(ctx.as_mut_ptr(), end, helpers.as_ptr()) },
                end
            );
            assert_eq!(
                ctx[0],
                usize::from(!cached),
                "native/Rust probe disagreement after growth"
            );
            if cached {
                assert!(matches!(operands[1].unpack(), Value::Num(value) if value == n as f64));
            }
            ctx[0] = 0;
        }
        for (n, key) in keys.iter().enumerate().take(256) {
            assert!(engine.interp.get_computed_property(&receiver, key).is_ok());
            let mut operands = [
                PackedValue::pack(receiver.clone()),
                PackedValue::pack(Value::Str(key.clone())),
            ];
            unsafe {
                entry(
                    ctx.as_mut_ptr(),
                    operands.as_mut_ptr().add(2),
                    helpers.as_ptr(),
                );
            }
            assert_eq!(ctx[0], 0, "a refilled key must hit the grown table");
            assert!(matches!(operands[1].unpack(), Value::Num(value) if value == n as f64));
        }
    }

    #[test]
    fn secondary_probe_matches_rust_across_growth_epoch_and_realm_misses() {
        use crate::bytecode::{CallIc, CallOverflow};
        let mut engine = crate::Engine::new();
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let mut asm = super::asm::Asm::new();
        asm.stp_pre(19, 30, -16);
        asm.mov(19, 0); // synthetic JitCtx header
        asm.mov(13, 1); // key
        asm.mov_w(15, 2); // epoch
        asm.mov(17, 3); // realm
        let hit = asm.new_label();
        let miss = asm.new_label();
        let done = asm.new_label();
        super::emit_call_overflow_probe(&mut asm, &layout, hit, miss);
        asm.bind(hit);
        asm.mov(0, 12);
        asm.b(done);
        asm.bind(miss);
        asm.movz(0, 0, 0);
        asm.bind(done);
        asm.ldp_post(19, 30, 16);
        asm.ret();
        let bytes: Vec<u8> = asm
            .finish()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let executable = super::ExecutableBuffer::from_bytes(&bytes).unwrap();
        let probe: unsafe extern "C" fn(*const usize, usize, u32, usize) -> *const CallIc =
            unsafe { std::mem::transmute(executable.as_ptr()) };
        let mut context = [0usize; 10];
        // Engine owns a Box<Interp>; JitCtx contains the pointee, not the Box handle's address.
        context[72 / 8] = &mut *engine.interp as *mut _ as usize;
        assert!(unsafe { probe(context.as_ptr(), 100, 7, 123) }.is_null());
        let mut objects = Vec::new();
        for batch in 0..4 {
            for _ in 0..512 {
                let object = crate::value::Object::new(None);
                engine.interp.call_overflow.insert(
                    CallIc {
                        callee: crate::value::Gc::as_ptr(&object) as usize,
                        epoch: 7,
                        global_env: 123,
                        ..CallIc::EMPTY
                    },
                    crate::value::Gc::downgrade(&object),
                );
                objects.push(object);
            }
            for object in &objects {
                let key = crate::value::Gc::as_ptr(object) as usize;
                let expected = engine.interp.call_overflow.lookup(key, 123, 7);
                let found = unsafe { probe(context.as_ptr(), key, 7, 123) };
                assert_eq!(!found.is_null(), expected.is_some(), "batch {batch}");
                if !found.is_null() {
                    assert_eq!(unsafe { (*found).callee }, key);
                }
                assert!(unsafe { probe(context.as_ptr(), key, 8, 123) }.is_null());
                assert!(unsafe { probe(context.as_ptr(), key, 7, 124) }.is_null());
            }
        }
        assert!(
            engine.interp.call_overflow.hash_shift < 57,
            "exercise growth with already-compiled probes"
        );
        assert!(engine.interp.call_overflow.retained_bytes() <= 512 * CallOverflow::SET_STRIDE);
    }

    #[test]
    fn direct_call_sequence_is_emitted_for_the_live_interpreter_layout() {
        let mut engine = crate::Engine::new();
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let value_layout = crate::value::jit_layout(&engine.interp.object_proto);
        let mut asm = super::asm::Asm::new();
        let hit_slow = asm.new_label();
        let unwind = asm.new_label();
        let done = asm.new_label();
        let finish = asm.new_label();
        let attempted = std::mem::offset_of!(crate::bytecode::Chunk, inline_attempted);
        assert!(
            super::emit_direct_call(
                &mut asm, &layout, &value_layout, attempted,
                std::mem::offset_of!(crate::bytecode::Chunk, jit_runs),
                std::mem::offset_of!(crate::bytecode::Chunk, inline_retry_at), 1, false,
                hit_slow, unwind, done, finish,
                super::DirectCallSite::Position(crate::stack_trace::NO_POSITION),
            ),
            "direct calls silently disabled: valid={}, depth={}, direct_depth={}, gc_tick={}, gc_next={}, call_site={}, constructing={}, new_target={}, frames={}, pool={}, attempted={}",
            layout.valid, layout.depth, layout.direct_call_depth, layout.gc_tick,
            layout.gc_next, layout.call_site, layout.constructing, layout.new_target,
            layout.fn_frames, layout.frame_pool, attempted,
        );
    }

    #[test]
    fn byte_field_address_preserves_large_offset_loads_and_stores() {
        for offset in [0usize, 4095, 4096, 4400, 32768] {
            let mut asm = super::asm::Asm::new();
            // extern C fn(base: *mut u8, value: u8) -> u8, x0/w1 -> w0.
            let (base, imm) = super::byte_field_address(&mut asm, 0, offset, 16);
            asm.strb_imm(1, base, imm);
            asm.ldrb_imm(0, base, imm);
            asm.ret();
            let code: Vec<u8> = asm
                .finish()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
            let executable = super::ExecutableBuffer::from_bytes(&code).unwrap();
            let entry: unsafe extern "C" fn(*mut u8, u8) -> u8 =
                unsafe { std::mem::transmute(executable.as_ptr()) };
            let mut bytes = vec![0u8; offset + 2];
            assert_eq!(unsafe { entry(bytes.as_mut_ptr(), 173) }, 173);
            assert_eq!(bytes[offset], 173);
            assert_eq!(bytes.iter().filter(|&&byte| byte != 0).count(), 1);
        }
    }

    #[test]
    fn direct_finish_clears_the_callee_records_handlers() {
        let handlers_len = super::jit_handlers_len_offset().expect("Vec length word") as u32;
        let mut asm = super::asm::Asm::new();
        super::emit_handler_clear(&mut asm, 10, handlers_len);

        let words = asm.finish();
        // str xzr, [x10, #handlers_len]
        let store_len = 0xF900_0000 | ((handlers_len / 8) << 10) | (10 << 5) | 31;
        assert_eq!(words, [store_len]);
    }
}

/// Recognize value-preserving local stores without crossing an independently entered PC.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn local_store_pair(ops: &[crate::bytecode::Op], pc: usize, targeted: &[bool]) -> Option<u16> {
    use crate::bytecode::Op;
    if targeted.get(pc + 1).copied().unwrap_or(true) {
        return None;
    }
    let slot = match (ops.get(pc)?, ops.get(pc + 1)?) {
        (Op::Dup, Op::StoreLocal(slot)) => *slot,
        (Op::StoreLocal(stored), Op::LoadLocal(loaded)) if stored == loaded => *stored,
        _ => return None,
    };
    ((slot as u32) * 8 + 8 < 4096).then_some(slot)
}

/// Canonical local update shared by baseline and general-region checked effects. The Number
/// path has no ownership changes; every coercing kind keeps the authoritative opcode helper.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_update_local(
    a: &mut asm::Asm,
    slot: u16,
    kind: crate::bytecode::UpdKind,
    pc: u32,
    unwind: usize,
) {
    use crate::bytecode::UpdKind;
    let off = slot as i32 * 8;
    let slow = a.new_label();
    let done = a.new_label();
    emit_exec_word_load(a, 9, 22, off);
    emit_exec_number_guard(a, 9, 0, 10, slow);
    a.fmov_one(1);
    let dec = matches!(
        kind,
        UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
    );
    a.f_arith(if dec { 1 } else { 0 }, 2, 0, 1);
    emit_exec_number_store(a, 2, 22, off, 10);
    match kind {
        UpdKind::PreInc | UpdKind::PreDec => {
            emit_exec_number_store(a, 2, 20, 0, 10);
            a.add_imm(20, 20, 8);
        }
        UpdKind::PostInc | UpdKind::PostDec => {
            emit_exec_number_store(a, 0, 20, 0, 10);
            a.add_imm(20, 20, 8);
        }
        UpdKind::IncDiscard | UpdKind::DecDiscard => {}
    }
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, unwind);
    a.bind(done);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn local_read_discard_pair(
    ops: &[crate::bytecode::Op],
    pc: usize,
    targeted: &[bool],
) -> Option<u16> {
    use crate::bytecode::Op;
    if targeted.get(pc + 1).copied().unwrap_or(true) {
        return None;
    }
    match (ops.get(pc)?, ops.get(pc + 1)?) {
        (Op::LoadLocal(slot), Op::Pop) if (*slot as u32) * 8 + 8 < 4096 => Some(*slot),
        _ => None,
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_discard_local_read(a: &mut asm::Asm, off: u32, pc: u32, l_unwind: usize) {
    let done = a.new_label();
    a.ldr_imm(9, 22, off);
    a.mov_imm64(10, crate::value::PACK_EMPTY);
    a.cmp_reg_x(9, 10);
    a.b_cond(C_NE, done);
    emit_exec(a, pc, l_unwind);
    emit_exec(a, pc + 1, l_unwind);
    a.bind(done);
}

/// Move the stack's top value into a local, optionally retaining the expression result.
/// `pcs` is the exact unfused sequence for a guard miss. The store is a move and the retained
/// copy a bit copy, so with a valid layout (`valid`) only an Empty (TDZ) result replays the
/// original sequence. Without one, a word above a Number's kind (the slot's old word, or a
/// retained result) replays it too.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_store_local(
    a: &mut asm::Asm,
    off: u32,
    pcs: &[u32],
    l_unwind: usize,
    keep: bool,
    valid: bool,
) {
    let slow = a.new_label();
    let done = a.new_label();
    if keep {
        // Empty must replay the original sequence so StoreLocal/LoadLocal retains its TDZ
        // error.
        a.ldur(9, 20, -8);
        emit_exec_kind(a, 9, 10, 11, slow);
        a.cmp_imm_w(10, 1);
        a.b_cond(C_EQ, slow);
        if !valid {
            a.cmp_imm_w(10, 4);
            a.b_cond(C_HI, slow);
        }
    }
    if !valid {
        a.ldr_imm(9, 22, off);
        emit_exec_kind(a, 9, 10, 11, slow);
        a.cmp_imm_w(10, 4);
        a.b_cond(C_HI, slow);
    }
    a.ldur(9, 20, -8);
    a.str_imm(9, 22, off);
    if !keep {
        a.sub_imm(20, 20, 8);
    }
    // A valid layout's store with nothing kept has no guard, hence no checked path.
    if keep || !valid {
        a.b(done);
        a.bind(slow);
        for &pc in pcs {
            emit_exec(a, pc, l_unwind);
        }
    }
    a.bind(done);
}

/// Byte loads/stores have an unscaled 12-bit offset, unlike the wider range of
/// word/pointer accesses. Interp's Rust layout can put a byte field beyond 4 KiB;
/// form its address in an explicitly reserved scratch register instead of
/// silently disabling every direct call. ADD does not alter condition flags.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn byte_field_address(a: &mut asm::Asm, base: u32, offset: usize, scratch: u32) -> (u32, u32) {
    if offset < 4096 {
        (base, offset as u32)
    } else {
        debug_assert_ne!(base, scratch);
        a.mov_imm64(scratch, offset as u64);
        a.add_shifted(scratch, base, scratch, 0);
        (scratch, 0)
    }
}

/// Discard handler records left in a finished direct callee's own record (register `record`).
/// A `return` is an abrupt completion that leaves its surrounding `try` (ECMA-262 14.10.1 and
/// 14.15.3), so a direct callee may legitimately bypass its lexical `PopHandler` operation.
/// `Handler` has no destructor, so clearing the length releases nothing else.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_handler_clear(a: &mut asm::Asm, record: u32, handlers_len_off: u32) {
    a.str_imm(31, record, handlers_len_off);
}

/// The direct-call teardown stub, emitted ONCE per chunk (sites reach it by `bl`; per-site
/// inlining would grow every call site by ~90 instructions). Entry: w1 = threw, x10 = the
/// callee's finished activation record (its `JitCtx`), x19 = the caller's context, x21 =
/// helpers. Exit: w8 = final threw, with the completion moved into the caller's context
/// (`ret` or `error`) and the record back in the pool; everything else caller-saved is
/// clobbered. The fast path replicates `jit_direct_finish` for a normal return without a
/// pending tail call or materialized `extra`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_direct_finish_stub(
    a: &mut asm::Asm,
    il: &crate::interpreter::InterpLayout,
    // Whether the templates' layout is valid (`JitLayout::valid`): without it every teardown
    // takes the helper.
    layout_valid: bool,
) {
    use std::mem::offset_of;
    let cx_ret = offset_of!(JitCtx, ret) as u32;
    let cx_activation = offset_of!(JitCtx, activation) as u32;
    let slow = a.new_label();
    let fits8 = |o: usize| o & 7 == 0 && o / 8 < 4096;
    let handlers_len_off = jit_handlers_len_offset().filter(|&off| fits8(off));
    let fast_ok = layout_valid
        && fits8(cx_activation as usize)
        && il.valid
        && handlers_len_off.is_some()
        && fits8(cx_ret as usize)
        && fits8(il.pending_tail)
        && fits8(il.fn_frames + il.fnf_ptr_word)
        && fits8(il.fn_frames + il.fnf_len_word)
        && fits8(il.frame_pool + il.fp_ptr_word)
        && fits8(il.frame_pool + il.fp_len_word)
        && fits8(il.frame_pool + il.fp_cap_word)
        && il.depth & 3 == 0
        && il.depth / 4 < 4096;
    // The stub calls out (the full helper), so lr is spilled for the whole body, with the record
    // at [sp, 16]; both exits share the epilogue.
    let done = a.new_label();
    a.stp_pre(29, 30, -32);
    a.stur(10, 31, 16);
    if fast_ok {
        let handlers_len_off = handlers_len_off.expect("checked above") as u32;
        a.cbnz(1, false, slow); // threw → helper
        a.ldr_imm(14, 19, 72); // ctx.interp
                               // operand stack clean (a clean return always leaves final_sp == stack_base)
        a.ldr_imm(9, 10, 16); // record.final_sp
        a.ldr_imm(11, 10, 8); // record.stack_base
        a.cmp_reg_x(9, 11);
        a.b_cond(C_NE, slow);
        // no pending proper-tail-call (Option<Box> niche: None = 0)
        a.ldr_imm(9, 14, il.pending_tail as u32);
        a.cbnz(9, true, slow);
        // No materialized native activation (Option<Box> niche: None = 0): bridge operations
        // create it on demand, and only `JitFrame::release` may drop it and restore the pooled
        // invariant.
        a.ldr_imm(9, 10, cx_activation);
        a.cbnz(9, true, slow);
        // FnFrame top: no materialized `extra` (the asm push wrote None; only the callee's own
        // arguments-object materialization could have filled it)
        a.ldr_imm(16, 14, (il.fn_frames + il.fnf_len_word) as u32);
        a.ldr_imm(6, 14, (il.fn_frames + il.fnf_ptr_word) as u32);
        a.sub_imm(16, 16, 1);
        a.movz(5, 24, 0);
        a.madd(6, 16, 5, 6);
        a.ldur(9, 6, 16); // FnFrame.extra
        a.cbnz(9, true, slow);
        // pool room: len < FRAME_POOL_LIMIT and len < capacity (a push must not reallocate the
        // Vec from machine code). Nothing below touches the pool, the frame stack, or the depth
        // before the commit, so validating here stays sound.
        a.ldr_imm(7, 14, (il.frame_pool + il.fp_len_word) as u32);
        a.cmp_imm_x(7, FRAME_POOL_LIMIT as u32);
        a.b_cond(C_HS, slow);
        a.ldr_imm(4, 14, (il.frame_pool + il.fp_cap_word) as u32);
        a.cmp_reg_x(7, 4);
        a.b_cond(C_HS, slow);
        // ---- commit ----
        emit_handler_clear(a, 10, handlers_len_off);
        // Move the completion into the caller: its `ret` is vacant while it runs, so the
        // record's word moves as it is (it owns nothing).
        a.ldr_imm(4, 10, cx_ret);
        a.str_imm(4, 19, cx_ret);
        a.mov_imm64(5, crate::value::PACK_UNDEFINED);
        a.str_imm(5, 10, cx_ret);
        // Reset the record's `this` (its packed word at [48]) to Undefined, the pooled invariant
        // a receiver-less direct call relies on, since it does not write the callee's `this`: x5
        // still holds the complete Undefined word (a zero word would be +0.0). Nothing is
        // released, and the record's slots keep their words, which own nothing once it is back
        // in the pool.
        a.str_imm(5, 10, 48);
        // Bookkeeping (re-read everything).
        a.ldr_imm(14, 19, 72); // ctx.interp
                               // FnFrame pop
        a.ldr_imm(16, 14, (il.fn_frames + il.fnf_len_word) as u32);
        a.sub_imm(16, 16, 1);
        a.str_imm(16, 14, (il.fn_frames + il.fnf_len_word) as u32);
        // record push: ptr[len] = record; len++ (room validated above)
        a.ldr_imm(7, 14, (il.frame_pool + il.fp_len_word) as u32);
        a.ldr_imm(4, 14, (il.frame_pool + il.fp_ptr_word) as u32);
        a.add_shifted(4, 4, 7, 3);
        a.stur(10, 4, 0);
        a.add_imm(7, 7, 1);
        a.str_imm(7, 14, (il.frame_pool + il.fp_len_word) as u32);
        // depth--
        a.ldr_w_imm(4, 14, il.depth as u32);
        a.sub_imm(4, 4, 1);
        a.str_w_imm(4, 14, il.depth as u32);
        a.movz(8, 0, 0); // not threw
        a.b(done);
    }
    // ---- helper fallback (nothing mutated above: `slow` is only reachable pre-commit) ----
    a.bind(slow);
    a.mov(2, 1); // threw
    a.mov(1, 10); // callee record
    a.mov(0, 19); // caller context
    a.ldr_imm(16, 21, (H_DIRECT_FINISH * 8) as u32);
    a.blr(16);
    a.mov(8, 0);
    a.bind(done);
    a.ldp_post(29, 30, 32);
    a.ret();
}

/// Same immediate-range gate as [`get_prop_inlinable`] plus the `proto` offset (GetMethod walks
/// one prototype hop).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn get_method_inlinable(layout: &crate::value::JitLayout) -> bool {
    get_prop_inlinable(layout) && layout.obj_proto < 4096
}

/// Same gate as [`get_prop_inlinable`] plus the `writable` byte (the store re-checks it — an
/// in-place defineProperty can flip attributes without changing the shape) and the creation
/// commit's layout-cell stores.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn set_prop_inlinable(layout: &crate::value::JitLayout) -> bool {
    get_prop_inlinable(layout)
        && layout.entry_writable < 256
        && layout.entry_accessor < 256
        && layout.obj_extensible < 4096
        && layout.obj_proto.is_multiple_of(8)
        && layout.obj_proto / 8 < 4096
        && layout.obj_props + layout.props_proto_flag < 4096
        && layout.props_elems.is_multiple_of(8)
        && layout.props_elems / 8 < 4096
        && fields_capacity_fits(layout)
        && layout.str_ptr_word < 256
        && layout.str_len_word < 256
}

/// Inline `this.x++` / `--` (`UpdateProp`): the read and the write both target the cached own
/// data slot — exactly what a depth-0 IC hit on the VM path does (`get_prop_ic` then
/// `set_prop_ic`) — so a shape-validated receiver whose slot holds a Num updates in place with
/// one FP add. Anything else (accessor, non-writable, non-Num old value, shape/depth miss,
/// exotic receiver) falls to the checked helper before any state is written.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_update_prop_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    kind: UpdKind,
    pc: u32,
    l_unwind: usize,
) {
    use crate::bytecode::{IC_OFF_DEPTH, IC_OFF_RECV_SHAPE, IC_OFF_SLOT};
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let sh = (layout.obj_props + layout.props_shape) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;

    let plain = layout.obj_ic_plain as u32;
    let slow = a.new_label();
    let done = a.new_label();
    // 1. stack: [obj @ -16] — receiver must be an Obj
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    emit_exec_payload(a, 9, 10);
    // 2. cache: depth 0, slot + shape
    a.mov_imm64(12, cache_ptr as u64);
    a.ldrb_imm(9, 12, IC_OFF_DEPTH);
    a.cbnz(9, false, slow);
    a.ldr_w_imm(13, 12, IC_OFF_SLOT);
    a.ldr_w_imm(14, 12, IC_OFF_RECV_SHAPE);
    // 3. ordinary receiver, shape match
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(9, 11, ex);
    guard_exotic_ordinary(a, layout, 9, 9, slow);
    a.ldrb_imm(9, 11, plain);
    a.cbz(9, false, slow);
    a.ldr_w_imm(9, 11, sh);
    a.cmp_reg_w(9, 14);
    a.b_cond(C_NE, slow);
    // 4. bounds-check the cached slot, then entry: data property, writable, holding a Num
    fields_len(a, layout, 16, 11);
    a.cmp_reg_x(13, 16);
    a.b_cond(C_HS, slow);
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 9, 15, ea, slow);
    guard_prop_writable(a, 9, 15, ew, slow);
    if layout.entry_accessor == layout.entry_value + 8 {
        a.ldur(16, 15, ev);
        a.lsr_imm(9, 16, 48);
        let number = a.new_label();
        a.movz(14, (crate::value::PACK_OBJ >> 48) as u32, 0);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.movz(14, (crate::value::PACK_UNDEFINED >> 48) as u32, 0);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_LO, number);
        a.movz(14, (crate::value::PACK_SYM >> 48) as u32, 0);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_LS, slow);
        a.bind(number);
    } else {
        a.ldurb(9, 15, ev);
        a.cmp_imm_w(9, 4);
        a.b_cond(C_NE, slow);
    }
    // --- commit: d0 = old, d2 = old ± 1, written in place ---
    a.ldur_d(
        0,
        15,
        if layout.entry_accessor == layout.entry_value + 8 {
            ev
        } else {
            ev + 8
        },
    );
    a.fmov_one(1);
    let dec = matches!(
        kind,
        UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
    );
    a.f_arith(if dec { 1 } else { 0 }, 2, 0, 1);
    if layout.entry_accessor == layout.entry_value + 8 {
        emit_exec_number_store(a, 2, 15, ev, 16);
    } else {
        a.stur_d(2, 15, ev + 8);
    }
    // result per kind: Pre* push the new value, Post* the old, *Discard nothing.
    match kind {
        UpdKind::PreInc | UpdKind::PreDec => {
            emit_exec_number_store(a, 2, 20, -8, 9);
        }
        UpdKind::PostInc | UpdKind::PostDec => {
            emit_exec_number_store(a, 0, 20, -8, 9);
        }
        UpdKind::IncDiscard | UpdKind::DecDiscard => {
            a.sub_imm(20, 20, 8);
        }
    }
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
}

/// Inline `obj[key]++` / `obj[key]--` (`Op::UpdateElem`, all six kinds) on an existing own
/// writable data element holding a Number, of a plain object or Array.
///
/// ECMA-262 e28783d5 §13.4.2.1-§13.4.5.1 (Postfix/Prefix Increment/Decrement): GetValue(lhs),
/// ToNumeric, Number::add/subtract(oldValue, 1), PutValue(lhs, newValue). For such an element
/// the [[Get]] reads [[Value]] without author code, ToNumeric of a Number is the identity, and
/// PutValue's OrdinarySetWithOwnDescriptor ends in ValidateAndApplyPropertyDescriptor on an
/// existing writable data property (for an Array, an index below `length`), which replaces
/// only [[Value]]: exactly `Props::set_index_value`, including its element-mirror upkeep.
/// Stack `[obj @ -16, key @ -8]`: the key must be a Number that is exactly a u32, so ToPropertyKey
/// is its canonical index text and runs no author code. Holes, accessors, non-writable or
/// non-Number elements, typed arrays, proxies and other side-table objects, other exotics and
/// every other key take the exact-PC helper; every guard precedes the first write, so that
/// re-run is clean.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_update_elem_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    kind: UpdKind,
    pc: u32,
    l_unwind: usize,
) {
    debug_assert_eq!(layout.entry_accessor, layout.entry_value + 8);
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;
    let plain = layout.obj_ic_plain as u32;
    let slow = a.new_label();
    let done = a.new_label();
    let element = a.new_label();
    a.mark_push("update-elem");
    // 1. an Obj receiver
    emit_exec_word_load(a, 9, 20, -16);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    // 2. the key is exactly a u32 (every non-Number word is a NaN pattern, and the unordered
    // compare is "not equal"). d4 keeps it for the mirror store; x9 holds the index.
    a.ldur_d(4, 20, -8);
    a.fcvtzu_w_d(9, 4);
    a.ucvtf_d_w(1, 9);
    a.fcmp(4, 1);
    a.b_cond(C_NE, slow);
    // 3. x10 = the receiver's payload
    emit_exec_word_load(a, 10, 20, -16);
    emit_exec_payload(a, 10, 10);
    // 4. ordinary or Array exotic, not in any side table (proxy, typed array, host object)
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, slow);
    // 5. dense element storage; x15 → the element's Property
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);
    if packed_elem_inlinable(layout)
        && layout.entry_value == layout.property_value
        && layout.entry_accessor == layout.property_meta
    {
        let classic = a.new_label();
        emit_packed_elements_base(a, layout, classic);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.add_shifted(15, 15, 9, 4); // property_size == 16 (packed_elem_inlinable)
        guard_prop_data(a, 14, 15, layout.property_meta as u32, slow);
        guard_prop_writable(a, 14, 15, layout.property_meta as u32, slow);
        a.b(element);
        a.bind(classic);
    }
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1); // NO_SLOT: a hole consults the prototype chain
    a.b_cond(C_EQ, slow);
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 14, 15, ea, slow);
    guard_prop_writable(a, 14, 15, ew, slow);
    a.bind(element);
    // 6. the old value is a Number (a packed hole and every other kind are tagged NaNs) → d5
    a.ldur(16, 15, ev);
    emit_exec_number_guard(a, 16, 5, 13, slow);
    // --- commit: d6 = old ± 1 replaces [[Value]]; the old Number needs no drop ---
    a.fmov_one(7);
    let dec = matches!(
        kind,
        UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
    );
    a.f_arith(if dec { 1 } else { 0 }, 6, 5, 7);
    emit_exec_number_store(a, 6, 15, ev, 13);
    // Keep the element mirror coherent (clobbers x9/x12/x13/d1/d2 only).
    emit_mirror_store(
        a,
        layout,
        11,
        MirrorKey::F64InDreg(4),
        MirrorVal::Num(6, false),
    );
    match kind {
        UpdKind::PreInc | UpdKind::PreDec => {
            emit_exec_number_store(a, 6, 20, -16, 9);
            a.sub_imm(20, 20, 8);
        }
        UpdKind::PostInc | UpdKind::PostDec => {
            emit_exec_number_store(a, 5, 20, -16, 9);
            a.sub_imm(20, 20, 8);
        }
        UpdKind::IncDiscard | UpdKind::DecDiscard => {
            a.sub_imm(20, 20, 16);
        }
    }
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
    a.mark_pop();
}

/// Gate for the inline equality / Not templates: the Obj arms read the receiver's `ic_plain`
/// byte, so those offsets must fit their instructions' immediate ranges.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn eq_inlinable(layout: &crate::value::JitLayout) -> bool {
    layout.valid
        && layout.obj_from_cell < 4096
        && layout.obj_ic_plain < 4096
        && crate::lstr::LEN_OFF.is_multiple_of(4)
        && crate::lstr::LEN_OFF / 4 < 4096
}

/// Fused `LoadLocal(a); LoadLocal(b); equality; JumpIfFalse`: compare borrowed frame values
/// directly for same-type primitives, object identity and nullish cases. These need no coercion or
/// ownership changes. Any TDZ value, coercing mixed pair, or HTMLDDA/nullish pair replays the
/// original operations through their checked helpers before the frame is touched.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_local_eq_branch(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    lhs_off: u32,
    rhs_off: u32,
    first_pc: u32,
    l_unwind: usize,
    strict: bool,
    negate: bool,
    target: usize,
) {
    let slow = a.new_label();
    let done = a.new_label();
    let lhs_obj = a.new_label();
    let rhs_obj = a.new_label();
    let both_obj = a.new_label();
    let lhs_nullish = a.new_label();
    let rhs_nullish = a.new_label();
    let equal = a.new_label();
    let unequal = a.new_label();
    let same_primitive = a.new_label();

    // w9/w10 are the borrowed Value tags. Empty is a TDZ sentinel, so it must retain the
    // checked LoadLocal path and its precise ReferenceError.
    emit_exec_word_load(a, 12, 22, lhs_off as i32);
    emit_exec_word_load(a, 13, 22, rhs_off as i32);
    emit_exec_kind(a, 12, 9, 14, slow);
    emit_exec_kind(a, 13, 10, 14, slow);
    a.cmp_imm_w(9, 1);
    a.b_cond(C_EQ, slow);
    a.cmp_imm_w(10, 1);
    a.b_cond(C_EQ, slow);
    a.cmp_imm_w(9, 8);
    a.b_cond(C_EQ, lhs_obj);
    a.cmp_imm_w(10, 8);
    a.b_cond(C_EQ, rhs_obj);
    a.cmp_reg_w(9, 10);
    a.b_cond(C_EQ, same_primitive);

    // Neither side is an object. Null/undefined compare loosely equal only to each other;
    // strictly they must have the same tag. Other strict different-tag pairs are definitively
    // unequal, while same-tag and loose primitive pairs retain their value/coercion helpers.
    a.cmp_imm_w(9, 0);
    a.b_cond(C_EQ, lhs_nullish);
    a.cmp_imm_w(9, 2);
    a.b_cond(C_EQ, lhs_nullish);
    a.cmp_imm_w(10, 0);
    a.b_cond(C_EQ, rhs_nullish);
    a.cmp_imm_w(10, 2);
    a.b_cond(C_EQ, rhs_nullish);
    if strict {
        a.cmp_reg_w(9, 10);
        a.b_cond(C_NE, unequal);
    }
    a.b(slow);

    a.bind(same_primitive);
    // IsStrictlyEqual / IsLooselyEqual delegate same-type Numbers to Number::equal.
    // FP equality handles both signed zeroes and makes every NaN compare unequal.
    let not_number = a.new_label();
    a.cmp_imm_w(9, 4);
    a.b_cond(C_NE, not_number);
    a.ldr_d_imm(0, 22, lhs_off);
    a.ldr_d_imm(1, 22, rhs_off);
    a.fcmp(0, 1);
    a.b_cond(C_EQ, equal);
    a.b(unequal);
    a.bind(not_number);
    a.cmp_imm_w(9, 2);
    a.b_cond(C_LS, equal); // both Undefined or both Null (Empty was rejected)
    let not_bool = a.new_label();
    a.cmp_imm_w(9, 3);
    a.b_cond(C_NE, not_bool);
    a.ldrb_imm(12, 22, lhs_off);
    a.ldrb_imm(13, 22, rhs_off);
    a.cmp_reg_w(12, 13);
    a.b_cond(C_EQ, equal);
    a.b(unequal);
    a.bind(not_bool);
    a.cmp_imm_w(9, 7);
    a.b_cond(C_EQ, both_obj); // Symbols also compare by identity
    a.b(slow); // String content and BigInt use their established helpers

    a.bind(lhs_nullish);
    if strict {
        a.cmp_reg_w(9, 10);
        a.b_cond(C_EQ, equal);
        a.b(unequal);
    } else {
        a.cmp_imm_w(10, 0);
        a.b_cond(C_EQ, equal);
        a.cmp_imm_w(10, 2);
        a.b_cond(C_EQ, equal);
        a.b(unequal);
    }

    a.bind(rhs_nullish);
    a.b(unequal); // lhs was already proven non-nullish

    // Object/object equality is identity for both strict and loose operators.
    a.bind(lhs_obj);
    a.cmp_imm_w(10, 8);
    a.b_cond(C_EQ, both_obj);
    if strict {
        a.b(unequal);
    } else {
        a.cmp_imm_w(10, 0);
        let lhs_null_cmp = a.new_label();
        a.b_cond(C_EQ, lhs_null_cmp);
        a.cmp_imm_w(10, 2);
        a.b_cond(C_NE, slow);
        a.bind(lhs_null_cmp);
        // The sole object/nullish exception is [[IsHTMLDDA]]. `ic_plain` false sends it back to
        // the helper; an ordinary object is definitively unequal to nullish.
        emit_exec_word_load(a, 12, 22, lhs_off as i32);
        emit_exec_payload(a, 12, 12);
        a.add_imm(12, 12, layout.obj_from_cell as u32);
        a.ldrb_imm(12, 12, layout.obj_ic_plain as u32);
        a.cbz(12, false, slow);
        a.b(unequal);
    }

    a.bind(rhs_obj);
    if strict {
        a.b(unequal);
    } else {
        a.cmp_imm_w(9, 0);
        let rhs_null_cmp = a.new_label();
        a.b_cond(C_EQ, rhs_null_cmp);
        a.cmp_imm_w(9, 2);
        a.b_cond(C_NE, slow);
        a.bind(rhs_null_cmp);
        emit_exec_word_load(a, 12, 22, rhs_off as i32);
        emit_exec_payload(a, 12, 12);
        a.add_imm(12, 12, layout.obj_from_cell as u32);
        a.ldrb_imm(12, 12, layout.obj_ic_plain as u32);
        a.cbz(12, false, slow);
        a.b(unequal);
    }

    a.bind(both_obj);
    emit_exec_word_load(a, 12, 22, lhs_off as i32);
    emit_exec_payload(a, 12, 12);
    emit_exec_word_load(a, 13, 22, rhs_off as i32);
    emit_exec_payload(a, 13, 13);
    a.cmp_reg_x(12, 13);
    a.b_cond(C_EQ, equal);
    a.b(unequal);

    // JumpIfFalse branches when `(equal XOR negate)` is false.
    a.bind(equal);
    if negate {
        a.b(target);
    } else {
        a.b(done);
    }
    a.bind(unequal);
    if negate {
        a.b(done);
    } else {
        a.b(target);
    }

    a.bind(slow);
    emit_exec(a, first_pc, l_unwind);
    emit_exec(a, first_pc + 1, l_unwind);
    emit_exec(a, first_pc + 2, l_unwind);
    emit_cond(a, COND_POP_TRUTHY, l_unwind);
    a.cbz(1, false, target);
    a.bind(done);
}

/// Gate for the ordinary-constructor `instanceof` template. The current heap property layout is
/// NaN-boxed; require that exact form so decoding `.prototype` remains fail-closed if storage is
/// changed again.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn instanceof_inlinable(
    layout: &crate::value::JitLayout,
    il: &crate::interpreter::InterpLayout,
) -> bool {
    let sh = layout.obj_props + layout.props_shape;
    layout.valid
        && il.valid
        && layout.entry_accessor == layout.entry_value + 8
        && layout.obj_from_cell < 4096
        && layout.obj_proto.is_multiple_of(8)
        && layout.obj_proto / 8 < 4096
        && layout.obj_exotic < 4096
        && layout.obj_ic_plain < 4096
        && layout.obj_is_constructor < 4096
        && sh.is_multiple_of(4)
        && sh / 4 < 4096
        && fields_data_fits(layout)
        && fields_len_fits(layout)
        && layout.entry_accessor < 4096
        && layout.entry_value < 256
        && layout.entry_size < 0x1_0000
        && il.function_proto.is_multiple_of(8)
        && il.function_proto / 8 < 4096
}

/// Inline the default OrdinaryHasInstance case. A single cache cell proves that the RHS still
/// has the key set observed by the checked path (notably, no own `@@hasInstance`); live guards
/// additionally validate constructor identity facts and decode its current `.prototype` value.
/// The LHS prototype walk is raw only while the realm-wide proxy latch remains clear. Every miss
/// occurs before the stack is written and therefore cleanly replays through `jit_exec`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_instanceof_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    il: &crate::interpreter::InterpLayout,
    cache_ptr: usize,
    pc: u32,
    l_unwind: usize,
) {
    use crate::bytecode::{IC_OFF_DEPTH, IC_OFF_RECV_SHAPE, IC_OFF_SLOT};
    let slow = a.new_label();
    let done = a.new_label();
    let refs_ok = a.new_label();
    let walk = a.new_label();
    let yes = a.new_label();
    let no = a.new_label();
    let have = a.new_label();
    let sh = (layout.obj_props + layout.props_shape) as u32;

    // The RHS must be an object. A primitive LHS is immediately false once the
    // ordinary-constructor guards below pass; accepting it here avoids a helper call for the
    // ubiquitous linked-list tail check `atom instanceof Pair`. Every LHS kind (BigInt
    // included) is admitted; unknown tags retain the checked path. w7 keeps the LHS kind.
    emit_exec_word_load(a, 8, 20, -16);
    emit_exec_kind(a, 8, 7, 9, slow);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    a.ldr_imm(9, 19, 32); // ctx.inline_ic_safe
    a.ldrb_imm(9, 9, 0);
    a.cbz(9, false, slow);
    emit_exec_word_load(a, 11, 20, -8);
    emit_exec_payload(a, 11, 11);

    // x10 = the payload of a String, Symbol or Object LHS; the walk below reads an object's.
    a.cmp_imm_w(7, 6);
    a.b_cond(C_LO, refs_ok);
    emit_exec_word_load(a, 10, 20, -16);
    emit_exec_payload(a, 10, 10);
    a.bind(refs_ok);

    // RHS: ordinary/plain constructor, canonical Function.prototype, and cached shape.
    a.add_imm(12, 11, layout.obj_from_cell as u32);
    a.ldrb_imm(9, 12, layout.obj_exotic as u32);
    a.cmp_imm_w(9, layout.exotic_none_tag as u32);
    a.b_cond(C_NE, slow);
    a.ldrb_imm(9, 12, layout.obj_ic_plain as u32);
    a.cbz(9, false, slow);
    a.ldrb_imm(9, 12, layout.obj_is_constructor as u32);
    a.cbz(9, false, slow);
    a.ldr_imm(14, 12, layout.obj_proto as u32);
    a.ldr_imm(15, 19, 72); // ctx.interp
    a.ldr_imm(15, 15, il.function_proto as u32);
    a.cmp_reg_x(14, 15);
    a.b_cond(C_NE, slow);
    a.mov_imm64(13, cache_ptr as u64);
    a.ldrb_imm(9, 13, IC_OFF_DEPTH);
    a.cmp_imm_w(9, 0);
    a.b_cond(C_NE, slow);
    a.ldr_w_imm(14, 12, sh);
    a.ldr_w_imm(15, 13, IC_OFF_RECV_SHAPE);
    a.cmp_reg_w(14, 15);
    a.b_cond(C_NE, slow);

    // Resolve the cached own `.prototype` slot defensively, require a data property holding an
    // object, and untag its cell address from the packed heap value.
    a.ldr_w_imm(13, 13, IC_OFF_SLOT);
    fields_len(a, layout, 14, 12);
    a.cmp_reg_x(13, 14);
    a.b_cond(C_HS, slow);
    fields_data(a, layout, 15, 12);
    a.mov_imm64(16, layout.entry_size as u64);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 9, 15, layout.entry_accessor as u32, slow);
    a.ldur(15, 15, layout.entry_value as i32);
    a.lsr_imm(16, 15, 48);
    a.movz(17, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(16, 17);
    a.b_cond(C_NE, slow);
    a.lsl_imm(15, 15, 16);
    a.lsr_imm(15, 15, 16); // x15 = the target prototype's cell address

    // A scalar LHS is false. Otherwise walk lhs.[[Prototype]] until target or null. Cycles are
    // rejected by SetPrototypeOf, and the proxy latch above makes every hop a direct field read.
    a.cmp_imm_w(7, 8);
    a.b_cond(C_NE, no);
    a.mov(12, 10);
    a.bind(walk);
    a.add_imm(12, 12, layout.obj_from_cell as u32);
    a.ldr_imm(12, 12, layout.obj_proto as u32);
    a.cbz(12, true, no);
    a.cmp_reg_x(12, 15);
    a.b_cond(C_EQ, yes);
    a.b(walk);
    a.bind(yes);
    a.movz(9, 1, 0);
    a.b(have);
    a.bind(no);
    a.movz(9, 0, 0);
    a.bind(have);

    // Commit: all guards have passed. Replace the two inputs with one Bool, and leave the stack
    // in the normal binary-op shape.
    a.mov_imm64(10, crate::value::PACK_BOOL);
    a.logic_x(1, 9, 9, 10);
    emit_exec_word_store(a, 9, 20, -16);
    a.sub_imm(20, 20, 8);
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, H_INSTANCEOF, pc, l_unwind);
    a.bind(done);
}

/// Probe one polymorphic property-creation way: x12 = the way's `IcState` cell. Entry has x11
/// at the receiver Object and the incoming value at sp-16. A hit leaves x12 at its IcState and
/// x13 holding the current entries length and x7 at a replacement layout cell (zero keeps the
/// current prediction), then branches to `commit`; a miss has no side effects. Clobbers x7, x9
/// and x12..x17 only.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_prop_create_probe(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    name: &str,
    miss: usize,
    commit: usize,
) {
    use crate::bytecode::{
        IC_CREATE, IC_OFF_DEPTH, IC_OFF_HOLDER_SHAPE, IC_OFF_MID2_SHAPE, IC_OFF_MID_SHAPE,
        IC_OFF_RECV_SHAPE, IC_OFF_SLOT,
    };
    let sh = (layout.obj_props + layout.props_shape) as u32;
    a.ldrb_imm(9, 12, IC_OFF_DEPTH);
    a.cmp_imm_w(9, IC_CREATE as u32);
    a.b_cond(C_NE, miss);
    a.ldr_w_imm(13, 11, sh);
    a.ldr_w_imm(14, 12, IC_OFF_RECV_SHAPE);
    a.cmp_reg_w(13, 14);
    a.b_cond(C_NE, miss);
    a.ldrb_imm(9, 11, layout.obj_extensible as u32);
    a.cbz(9, false, miss);
    a.ldrb_imm(9, 11, (layout.obj_props + layout.props_proto_flag) as u32);
    a.cbnz(9, false, miss);
    // Named-only small map: no DenseStorage sidecar/index, <=8 entries, and spare capacity.
    a.ldr_imm(9, 11, (layout.obj_props + layout.props_elems) as u32);
    a.cbnz(9, true, miss);
    fields_len(a, layout, 13, 11);
    a.cmp_imm_x(13, 8);
    a.b_cond(C_HS, miss);
    fields_capacity(a, layout, 14, 11);
    a.cmp_reg_x(13, 14);
    a.b_cond(C_HS, miss);
    let transition_layout = a.new_label();
    let key_ready = a.new_label();
    a.movz(7, 0, 0);
    // A predicted layout already owns the next key. Otherwise the creation IC's destination
    // shape can supply a cached layout, with no allocation or string-keyed lookup.
    a.ldr_imm(16, 11, (layout.obj_props + layout.props_layout) as u32);
    a.cbz(16, true, transition_layout);
    a.ldr_imm(17, 16, (layout.layout_data_off + layout.vec_len_off) as u32);
    a.cmp_reg_x(13, 17);
    a.b_cond(C_HS, transition_layout);
    a.ldr_imm(16, 16, (layout.layout_data_off + layout.vec_ptr_off) as u32);
    a.add_shifted(16, 16, 13, 4);
    a.ldur(17, 16, layout.str_len_word as i32);
    a.mov_imm64(14, name.len() as u64);
    a.cmp_reg_x(17, 14);
    a.b_cond(C_NE, transition_layout);
    a.ldur(17, 16, layout.str_ptr_word as i32);
    a.mov_imm64(14, (name.as_ptr() as usize - layout.str_data_off) as u64);
    a.cmp_reg_x(17, 14);
    // Layouts can be learned before compilation or shared with another creation site. Equal
    // property names need not own the same Rc allocation. Keep identity as the cheap case,
    // then compare short names by content without calling out or changing any ownership.
    if name.len() <= 32 {
        let key_matches = a.new_label();
        a.b_cond(C_EQ, key_matches);
        let d = layout.str_data_off as u32;
        let bytes = name.as_bytes();
        let mut off = 0usize;
        while bytes.len() - off >= 4 {
            let imm = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            a.ldr_w_imm(16, 17, d + off as u32);
            a.movz(14, imm & 0xFFFF, 0);
            a.movk(14, imm >> 16, 1);
            a.cmp_reg_w(16, 14);
            a.b_cond(C_NE, transition_layout);
            off += 4;
        }
        if bytes.len() - off >= 2 {
            let imm = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) as u32;
            a.ldrh_imm(16, 17, d + off as u32);
            a.movz(14, imm, 0);
            a.cmp_reg_w(16, 14);
            a.b_cond(C_NE, transition_layout);
            off += 2;
        }
        if bytes.len() - off == 1 {
            a.ldrb_imm(16, 17, d + off as u32);
            a.cmp_imm_w(16, bytes[off] as u32);
            a.b_cond(C_NE, transition_layout);
        }
        a.bind(key_matches);
    } else {
        a.b_cond(C_NE, transition_layout);
    }
    a.b(key_ready);
    a.bind(transition_layout);
    // This activation's heap pins cached layouts. The shape table cannot change between
    // this probe and commit: no helper, allocation, GC or user code runs in that interval.
    a.ldr_imm(16, 19, std::mem::offset_of!(JitCtx, shape_layouts) as u32);
    a.ldr_w_imm(14, 12, IC_OFF_HOLDER_SHAPE);
    // A bounded paged directory caches globally unique shape identities. Indexing chooses
    // only a candidate: the complete stored ID must match before borrowing its key layout.
    a.ubfx(17, 14, 6, 6);
    a.add_shifted(16, 16, 17, 3);
    a.ldr_imm(16, 16, 0); // Option<Box<LayoutPage>> is a nullable allocation pointer.
    a.cbz(16, true, miss);
    a.ubfx(17, 14, 0, 6);
    a.add_shifted(16, 16, 17, 4);
    a.ldr_w_imm(17, 16, layout.shape_layout_entry_id as u32);
    a.cmp_reg_w(17, 14);
    a.b_cond(C_NE, miss);
    a.ldr_imm(7, 16, layout.shape_layout_entry_keys as u32);
    a.cbz(7, true, miss);
    a.ldr_imm(17, 7, (layout.layout_data_off + layout.vec_len_off) as u32);
    a.cmp_reg_x(13, 17);
    a.b_cond(C_HS, miss);
    a.bind(key_ready);
    // Same live global epoch recorded by the fill; saturation is never cacheable.
    a.mov_imm64(16, crate::value::proto_epoch_ptr() as usize as u64);
    a.ldr_w_imm(16, 16, 0);
    a.ldr_w_imm(17, 12, IC_OFF_MID_SHAPE);
    a.cmp_reg_w(16, 17);
    a.b_cond(C_NE, miss);
    a.mov_imm64(9, u32::MAX as u64);
    a.cmp_reg_w(16, 9);
    a.b_cond(C_EQ, miss);
    // The cache stores `Gc::as_ptr(proto)`; Object::proto stores the cell address, the same
    // word (or null).
    a.ldr_w_imm(14, 12, IC_OFF_SLOT);
    a.ldr_w_imm(15, 12, IC_OFF_MID2_SHAPE);
    a.lsl_imm(15, 15, 32);
    a.logic_x(1, 14, 14, 15);
    a.ldr_imm(16, 11, layout.obj_proto as u32);
    a.cmp_reg_x(14, 16);
    a.b_cond(C_NE, miss);
    // Every execution word transfers unchanged.
    a.b(commit);
}

/// The write barrier's card store (S6 step 10b; `gc_core::card_mark`), emitted after a store of
/// a word that can be a reference into an object, environment or their storage, before any
/// safepoint: the card of the cell whose start (an object's or environment's) or Object address
/// (its cell's start + `obj_from_cell`, in the same 128-byte card) is in `owner` is set, from the
/// address alone. With base = owner & !0xffff and card = (owner >> 7) & 511, the byte at base +
/// 65,024 + card becomes 0xff:
///
/// ```text
/// ubfx s2, owner, #7, #9      ; the card
/// orr  s1, owner, #0xffff     ; base + 65,535, whose low byte (0xff) is the byte stored
/// sub  s2, s2, #511           ; card - 511: base + 65,535 + (card - 511) = base + 65,024 + card
/// strb w(s1), [s1, s2]        ; strb's immediate cannot reach 65,024: a register offset
/// ```
///
/// `s1` may be `owner`; `s2` differs from both. One emitter for every site.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_card_mark(a: &mut asm::Asm, owner: u32, s1: u32, s2: u32) {
    use crate::gc_core::{CARD_SHIFT, CARD_TABLE_BYTES, PAGE_SHIFT, PAGE_SIZE};
    debug_assert!(s2 != owner && s2 != s1);
    // The arithmetic below: base + (PAGE_SIZE - 1) + card - (CARD_TABLE_BYTES - 1) is the card's
    // byte in the table that ends the page, and the low byte of base + (PAGE_SIZE - 1) is set.
    const _: () = assert!(
        PAGE_SIZE - CARD_TABLE_BYTES == crate::gc_core::CARD_TABLE_OFFSET
            && PAGE_SIZE.is_power_of_two()
            && CARD_TABLE_BYTES.is_power_of_two()
            && PAGE_SIZE >= 256
    );
    a.ubfx(s2, owner, CARD_SHIFT, CARD_TABLE_BYTES.trailing_zeros());
    a.orr_low_ones_x(s1, owner, PAGE_SHIFT);
    a.sub_imm(s2, s2, (CARD_TABLE_BYTES - 1) as u32);
    a.strb_reg(s1, s1, s2);
}

/// Debug check for [`emit_card_mark`]'s use of an Object address: it lies in its cell's card.
/// Object cells (192 bytes) start 0 or 64 bytes into a 128-byte card.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn debug_assert_object_in_cell_card(layout: &crate::value::JitLayout) {
    debug_assert!(
        layout.obj_from_cell < (1 << crate::gc_core::CARD_SHIFT) / 2,
        "an Object's address must share its cell's card"
    );
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// Native own-data store, including the assignment expression's RHS result.
/// Every declining guard precedes mutation. `keep` also moves the RHS word over
/// the receiver as the expression's result. Descriptor/shape/exotic misses
/// retain the exact checked op.
/// ECMA-262 e28783d5: OrdinarySetWithOwnDescriptor and assignment Evaluation.
fn emit_set_prop_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    name: &str,
    pc: u32,
    l_unwind: usize,
    recv: PropRecv,
    keep: bool,
) {
    debug_assert!(!keep || matches!(recv, PropRecv::Stack));
    use crate::bytecode::{IC_OFF_DEPTH, IC_OFF_RECV_SHAPE, IC_OFF_SLOT};
    if layout.entry_accessor != layout.entry_value + 8 {
        emit_op_helper(a, H_SET_PROP, pc, l_unwind);
        return;
    }
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let sh = (layout.obj_props + layout.props_shape) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;

    let plain = layout.obj_ic_plain as u32;
    let slow = a.new_label();
    let done = a.new_label();
    if keep {
        // Never admit TDZ/property-only words as a JavaScript result.
        emit_exec_word_load(a, 16, 20, -8);
        emit_exec_kind(a, 16, 9, 14, slow);
        a.cmp_imm_w(9, 1);
        a.b_cond(C_EQ, slow);
    }
    // Stack form: [obj @ -16, v @ -8]; this/slot forms have [v @ -8] only and the frame keeps
    // the receiver.
    match recv {
        PropRecv::Stack => {
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
            emit_exec_payload(a, 9, 10); // receiver pointer
        }
        PropRecv::This => {
            emit_exec_word_load(a, 9, 19, 48); // ctx.this_word
            emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
            emit_exec_payload(a, 9, 10);
        }
        PropRecv::Slot(off) => {
            emit_exec_word_load(a, 9, 22, off as i32);
            emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
            emit_exec_payload(a, 9, 10);
        }
    }
    // 2. object base; exotic ordinary (None, or a Map/Set), and not a side-table exotic
    //    (proxy/typed-array/namespace). Receiver-wide facts — validated once, shared by both
    //    ways.
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(9, 11, ex);
    guard_exotic_ordinary(a, layout, 9, 9, slow);
    a.ldrb_imm(9, 11, plain);
    a.cbz(9, false, slow);
    // Creation IC: constructors repeatedly assign a named field to a fresh receiver whose map
    // lacks it (see `emit_prop_create_probe`). Its ways share the site's cells with ordinary
    // update ways; the single way loop below dispatches on each way's kind.
    let create_ok = layout.key_probe_ok
        && !name.is_empty()
        && !name.as_bytes()[0].is_ascii_digit()
        && name != "length"
        && name != "prototype";
    let create_commit = a.new_label();
    // 3-8 per way (sites allocate PROP_IC_WAYS consecutive cells; the Rust fast path probes
    // all of them, so the template must too or a rotating store site helper-calls forever):
    // depth 0, shape match, slot bounds, data+writable. Guard misses jump to the next way, the
    // last way's to the helper. Register results consumed by the commit below: x13 slot, x15
    // entry. `cache_ptr` = the IcState cell address, or 0 for "x12 already holds it" (the way
    // loop keeps its cursor in x8).
    let commit = a.new_label();
    let way = |a: &mut asm::Asm, cache_ptr: usize, miss: usize| {
        if cache_ptr != 0 {
            a.mov_imm64(12, cache_ptr as u64);
        }
        a.ldrb_imm(9, 12, IC_OFF_DEPTH);
        a.cbnz(9, false, miss);
        a.ldr_w_imm(13, 12, IC_OFF_SLOT);
        a.ldr_w_imm(14, 12, IC_OFF_RECV_SHAPE);
        a.ldr_w_imm(9, 11, sh);
        a.cmp_reg_w(9, 14);
        a.b_cond(C_NE, miss);
        fields_len(a, layout, 16, 11);
        a.cmp_reg_x(13, 16);
        a.b_cond(C_HS, miss);
        fields_data(a, layout, 15, 11);
        a.mov_imm64(16, es);
        a.madd(15, 13, 16, 15);
        guard_prop_data(a, 9, 15, ea, miss);
        guard_prop_writable(a, 9, 15, ew, miss);
        // The incoming word replaces the old value without observing it, including a
        // deferred prototype. Shape/data/writability and the write barrier remain in force.
    };
    // One pass over the ways (x8 = way cursor, w6 = ways left; neither probe clobbers them,
    // and the creation probe returns its layout in x7): each way is examined once, by kind — an
    // update way (depth 0) or, for a creation-eligible name, a creation way. The two kinds prove
    // different receiver shapes (without and with the property), so at most one validates.
    {
        use crate::bytecode::IC_CREATE;
        let l_way = a.new_label();
        let l_way_next = a.new_label();
        a.mov_imm64(8, cache_ptr as u64);
        a.movz(6, crate::bytecode::PROP_IC_WAYS as u32, 0);
        a.bind(l_way);
        a.mov(12, 8);
        if create_ok {
            let update_way = a.new_label();
            a.ldrb_imm(9, 12, IC_OFF_DEPTH);
            a.cmp_imm_w(9, IC_CREATE as u32);
            a.b_cond(C_NE, update_way);
            emit_prop_create_probe(a, layout, name, l_way_next, create_commit);
            a.bind(update_way);
        }
        way(a, 0, l_way_next);
        a.b(commit);
        a.bind(l_way_next);
        let ic_stride = std::mem::size_of::<std::cell::Cell<crate::bytecode::IcState>>();
        a.add_imm(8, 8, ic_stride as u32);
        a.sub_imm(6, 6, 1);
        a.cbnz(6, false, l_way);
        a.b(slow);
    }
    a.bind(commit);
    // --- commit: everything validated; from here only writes ---
    // Move v (the stack word, in x16) into the entry.
    emit_exec_word_load(a, 16, 20, -8);
    a.stur(16, 15, ev);
    // The write barrier (site A): the receiver's cell starts at x10.
    emit_card_mark(a, 10, 9, 17);
    if matches!(recv, PropRecv::Stack) {
        if keep {
            // The RHS word is also the expression's result, over the receiver.
            emit_exec_word_store(a, 16, 20, -16);
            a.sub_imm(20, 20, 8);
        } else {
            a.sub_imm(20, 20, 16);
        }
    } else {
        // pop just the value
        a.sub_imm(20, 20, 8);
    }
    a.b(done);
    // Creation commit (out of the update hit's fall-through line).
    if create_ok {
        use crate::bytecode::IC_OFF_HOLDER_SHAPE;
        a.bind(create_commit);
        // All creation proofs are complete. Preserve x7 (layout), x12 (IC), x13 (length) and
        // x10/x11 (receiver).
        // The heap roots this immutable shared layout. Replacing a private old layout
        // leaves reclamation to the collector; no helper or ownership traffic is required.
        let layout_installed = a.new_label();
        a.cbz(7, true, layout_installed);
        a.str_imm(7, 11, (layout.obj_props + layout.props_layout) as u32);
        a.bind(layout_installed);
        // From here no branch can fail: compute the vacant entry and pack the incoming value.
        fields_data(a, layout, 15, 11);
        a.mov_imm64(16, es);
        a.madd(15, 13, 16, 15);
        emit_exec_word_load(a, 16, 20, -8);
        // The predicted key is already owned by the shared layout (validated pre-commit).
        // Only install Property::plain; no per-instance key ownership operation is needed.
        a.stur(16, 15, ev);
        // The write barrier (site B): the receiver, which can be old (a constructor's `this`
        // made before the callee's entry poll), starts at x10.
        emit_card_mark(a, 10, 9, 17);
        a.movz(
            14,
            (crate::value::PROP_WRITABLE
                | crate::value::PROP_ENUMERABLE
                | crate::value::PROP_CONFIGURABLE) as u32,
            0,
        );
        a.stur(14, 15, ea as i32);
        // Publish the entry by updating shape then length. No allocation or side structure is
        // touched.
        a.ldr_w_imm(14, 12, IC_OFF_HOLDER_SHAPE);
        a.str_w_imm(14, 11, sh);
        a.add_imm(13, 13, 1);
        fields_set_len(a, layout, 13, 11);
        if matches!(recv, PropRecv::Stack) {
            if keep {
                emit_exec_word_store(a, 16, 20, -16);
                a.sub_imm(20, 20, 8);
            } else {
                a.sub_imm(20, 20, 16);
            }
        } else {
            a.sub_imm(20, 20, 8);
        }
        a.b(done);
    }
    a.bind(slow);
    emit_op_helper(a, H_SET_PROP, pc, l_unwind);
    a.bind(done);
}

/// Inline equality (`==` / `!=` / `===` / `!==`): every case the helper would resolve *without
/// coercion or content compares*, in machine code. Both-number pairs FCMP (IEEE: unordered is
/// unequal); loose nullish operands resolve by the other side's tag; same-tag Bools compare
/// payloads; same-tag Sym/Obj compare identity; same-tag Strs compare identity, then length (a
/// length mismatch is a definitive "not equal"; equal lengths fall to the helper's content
/// compare); strict different-tag pairs are unequal outright. Everything else — BigInt pairs,
/// coercing mixed-type pairs, a loose nullish-vs-object compare on a non-ordinary object
/// (`ic_plain` off — which includes the `[[IsHTMLDDA]]` object) — takes the helper. Every guard
/// branches to `slow` before any state is written. With `branch`, the result drives a fused
/// `JumpIfFalse` directly (no Bool materializes); otherwise the Bool pushes in place of the
/// operands.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_eq_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: u32,
    l_unwind: usize,
    strict: bool,
    negate: bool,
    branch: Option<usize>,
) {
    let len_off = crate::lstr::LEN_OFF as u32;
    let slow = a.new_label();
    let done = a.new_label();
    let l_num = a.new_label();
    let l_sametag = a.new_label();
    let l_bool = a.new_label();
    let l_str = a.new_label();
    let l_ptr = a.new_label();
    let l_ptr_same = a.new_label();
    let l_true = a.new_label();
    let l_false = a.new_label();
    let l_have = a.new_label();
    // Packed numeric operands need no semantic-kind decoding. Keep the guard's d0/d1 live
    // through the fused comparison; canonical NaN remains a Number but compares unequal.
    emit_exec_word_load(a, 12, 20, -16);
    emit_exec_word_load(a, 13, 20, -8);
    let tagged = a.new_label();
    let full_kind = a.new_label();
    let scalar_bool = a.new_label();
    emit_exec_number_guard(a, 12, 0, 14, tagged);
    emit_exec_number_guard(a, 13, 1, 14, tagged);
    a.b(l_num);
    a.bind(tagged);
    // Undefined/Null/Bool pairs are complete scalar comparisons, with no owners to release.
    // Do not generalize raw-bit equality to Numbers (NaN/±0) or reference-valued operands.
    a.lsr_imm(9, 12, 48);
    a.lsr_imm(10, 13, 48);
    let scalar_pairs = a.new_label();
    a.cmp_reg_w(9, 10);
    a.b_cond(C_NE, scalar_pairs);
    for (tag, target) in [
        (crate::value::PACK_OBJ, l_ptr),
        (crate::value::PACK_SYM, l_ptr),
        (crate::value::PACK_STR, l_str),
    ] {
        a.movz(14, (tag >> 48) as u32, 0);
        a.cmp_reg_w(9, 14);
        a.b_cond(C_EQ, target);
    }
    a.bind(scalar_pairs);
    a.movz(14, (crate::value::PACK_BOOL >> 48) as u32, 0);
    a.cmp_reg_w(9, 14);
    a.b_cond(C_EQ, scalar_bool);
    a.logic_imm_w(1, 9, 9, asm::logical_imm_w(2).unwrap());
    a.movz(14, (crate::value::PACK_NULL >> 48) as u32, 0);
    a.cmp_reg_w(9, 14);
    a.b_cond(C_NE, full_kind);
    a.logic_imm_w(1, 10, 10, asm::logical_imm_w(2).unwrap());
    a.cmp_reg_w(10, 14);
    a.b_cond(C_NE, full_kind);
    if strict {
        a.cmp_reg_x(12, 13);
        a.cset_w(11, C_EQ);
    } else {
        a.movz(11, 1, 0);
    }
    a.b(l_have);
    a.bind(scalar_bool);
    a.cmp_reg_w(10, 14);
    a.b_cond(C_NE, full_kind);
    a.cmp_reg_x(12, 13);
    a.cset_w(11, C_EQ);
    a.b(l_have);
    a.bind(full_kind);
    emit_exec_kind(a, 12, 9, 14, slow);
    emit_exec_kind(a, 13, 10, 14, slow);
    let l_notnum = a.new_label();
    a.cmp_imm_w(9, 4);
    a.b_cond(C_NE, l_notnum);
    a.cmp_imm_w(10, 4);
    a.b_cond(C_EQ, l_num);
    a.bind(l_notnum);
    if !strict {
        // Loose nullish: undefined/null equal each other and nothing else (helper handles the
        // IsHTMLDDA exception via the inline_ic_safe gate below).
        let la_null = a.new_label();
        let lb_null = a.new_label();
        a.cmp_imm_w(9, 0);
        a.b_cond(C_EQ, la_null);
        a.cmp_imm_w(9, 2);
        a.b_cond(C_EQ, la_null);
        a.cmp_imm_w(10, 0);
        a.b_cond(C_EQ, lb_null);
        a.cmp_imm_w(10, 2);
        a.b_cond(C_EQ, lb_null);
        a.b(l_sametag);
        // a is nullish: equal iff b is nullish; otherwise false.
        a.bind(la_null);
        a.cmp_imm_w(10, 0);
        a.b_cond(C_EQ, l_true);
        a.cmp_imm_w(10, 2);
        a.b_cond(C_EQ, l_true);
        // A BigInt is unequal to nullish like Bool/Num, so it falls into the scalar arm.
        a.cmp_imm_w(10, 6);
        a.b_cond(C_LO, l_false); // Bool/Num/BigInt
        emit_exec_word_load(a, 13, 20, -8);
        emit_exec_payload(a, 13, 13);
        let la_drop = a.new_label();
        a.cmp_imm_w(10, 8);
        a.b_cond(C_NE, la_drop);
        // nullish == Obj is only false for an ordinary object (ic_plain rules out IsHTMLDDA)
        a.add_imm(11, 13, layout.obj_from_cell as u32);
        a.ldrb_imm(11, 11, layout.obj_ic_plain as u32);
        a.cbz(11, false, slow);
        a.bind(la_drop);
        a.b(l_false);
        // b is nullish (a is not): false (a BigInt a is unequal like Bool/Num).
        a.bind(lb_null);
        a.cmp_imm_w(9, 6);
        a.b_cond(C_LO, l_false);
        emit_exec_word_load(a, 12, 20, -16);
        emit_exec_payload(a, 12, 12);
        let lb_drop = a.new_label();
        a.cmp_imm_w(9, 8);
        a.b_cond(C_NE, lb_drop);
        a.add_imm(11, 12, layout.obj_from_cell as u32);
        a.ldrb_imm(11, 11, layout.obj_ic_plain as u32);
        a.cbz(11, false, slow);
        a.bind(lb_drop);
        a.b(l_false);
    }
    a.bind(l_sametag);
    let l_diff = a.new_label();
    a.cmp_reg_w(9, 10);
    a.b_cond(C_NE, if strict { l_diff } else { slow });
    if strict {
        // Same-tag undefined/null are equal (loose routed them above).
        a.cmp_imm_w(9, 2);
        a.b_cond(C_LS, l_true);
    }
    a.cmp_imm_w(9, 3);
    a.b_cond(C_EQ, l_bool);
    a.cmp_imm_w(9, 6);
    a.b_cond(C_EQ, l_str);
    a.cmp_imm_w(9, 7);
    a.b_cond(C_HS, l_ptr); // Sym/Obj: identity
    a.b(slow); // BigInt
    a.bind(l_bool);
    a.ldurb(12, 20, -16);
    a.ldurb(13, 20, -8);
    a.cmp_reg_w(12, 13);
    a.cset_w(11, C_EQ);
    a.b(l_have);
    // Sym/Obj identity: same pointer → equal, different → unequal.
    a.bind(l_ptr);
    emit_exec_word_load(a, 12, 20, -16);
    emit_exec_payload(a, 12, 12);
    emit_exec_word_load(a, 13, 20, -8);
    emit_exec_payload(a, 13, 13);
    a.cmp_reg_x(12, 13);
    a.b_cond(C_EQ, l_ptr_same);
    a.b(l_false);
    a.bind(l_ptr_same);
    a.b(l_true);
    // Str: identity → equal; different lengths → unequal; same length → helper (content).
    a.bind(l_str);
    emit_exec_word_load(a, 12, 20, -16);
    emit_exec_payload(a, 12, 12);
    emit_exec_word_load(a, 13, 20, -8);
    emit_exec_payload(a, 13, 13);
    a.cmp_reg_x(12, 13);
    a.b_cond(C_EQ, l_ptr_same);
    a.ldr_w_imm(14, 12, len_off);
    a.ldr_w_imm(15, 13, len_off);
    a.cmp_reg_w(14, 15);
    a.b_cond(C_EQ, slow);
    a.b(l_false);
    if strict {
        // Different tags (both-number already peeled off): strictly unequal, whatever the
        // operands are.
        a.bind(l_diff);
        a.b(l_false);
    }
    a.bind(l_num);
    if let Some(target) = branch {
        // Straight-line fused numeric compare — branch on the negated condition, matching the
        // ordered-relation fusion (IEEE unordered must jump for == and fall through for !=).
        a.sub_imm(20, 20, 16);
        a.fcmp(0, 1);
        a.b_cond(if negate { C_EQ } else { C_NE }, target);
        a.b(done);
    } else {
        a.fcmp(0, 1);
        a.cset_w(11, C_EQ); // unordered (NaN) → 0: correctly unequal
        a.b(l_have);
    }
    a.bind(l_true);
    a.movz(11, 1, 0);
    a.b(l_have);
    a.bind(l_false);
    a.movz(11, 0, 0);
    a.bind(l_have);
    a.sub_imm(20, 20, 16);
    match branch {
        Some(target) => {
            // JumpIfFalse jumps when `eq ^ negate` is 0 — fold the negate into branch polarity.
            if negate {
                a.cbnz(11, false, target);
            } else {
                a.cbz(11, false, target);
            }
            a.b(done);
        }
        None => {
            if negate {
                a.movz(12, 1, 0);
                a.logic_w(2, 11, 11, 12); // eor: flip the pushed bool
            }
            a.mov_imm64(10, crate::value::PACK_BOOL);
            a.logic_x(1, 11, 11, 10);
            emit_exec_word_store(a, 11, 20, 0);
            a.add_imm(20, 20, 8);
            a.b(done);
        }
    }
    a.bind(slow);
    if strict {
        emit_op_helper(a, H_STRICT_EQ, pc, l_unwind);
    } else {
        emit_exec(a, pc, l_unwind);
    }
    if let Some(target) = branch {
        // Both equality helpers always push a Bool. Consume it directly instead of calling
        // ToBoolean through a second helper.
        a.ldurb(1, 20, -8);
        a.sub_imm(20, 20, 8);
        a.cbz(1, false, target);
    }
    a.bind(done);
}

/// Inline `!x` (ToBoolean + negate): Bool flips its payload; a Number is falsy iff ±0 or NaN;
/// undefined/null are falsy; a Str is falsy iff empty (length read through the header); Sym/Obj
/// are truthy — except a possible `[[IsHTMLDDA]]` object, so the Obj arm requires the
/// receiver's `ic_plain` byte. BigInt takes the helper. Guards all branch to `slow` before any
/// state is written.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_not_inline(a: &mut asm::Asm, layout: &crate::value::JitLayout, pc: u32, l_unwind: usize) {
    let len_off = crate::lstr::LEN_OFF as u32;
    let slow = a.new_label();
    let done = a.new_label();
    let l_bool = a.new_label();
    let l_num = a.new_label();
    let l_str = a.new_label();
    let l_objsym = a.new_label();
    let l_obj = a.new_label();
    let l_sym = a.new_label();
    let l_true = a.new_label();
    let l_have = a.new_label();
    emit_exec_word_load(a, 12, 20, -8);
    let tagged = a.new_label();
    emit_exec_number_guard(a, 12, 0, 10, tagged);
    a.b(l_num);
    a.bind(tagged);
    a.lsr_imm(9, 12, 48);
    for (tag, target) in [
        (crate::value::PACK_OBJ, l_obj),
        (crate::value::PACK_BOOL, l_bool),
        (crate::value::PACK_UNDEFINED, l_true),
        (crate::value::PACK_NULL, l_true),
        (crate::value::PACK_STR, l_str),
        (crate::value::PACK_SYM, l_sym),
        (crate::value::PACK_EMPTY, l_true),
    ] {
        a.movz(10, (tag >> 48) as u32, 0);
        a.cmp_reg_w(9, 10);
        a.b_cond(C_EQ, target);
    }
    a.b(slow); // BigInt and property-only/reserved tags retain the checked path.
    a.bind(l_obj);
    a.movz(9, 8, 0);
    a.b(l_objsym);
    a.bind(l_sym);
    a.movz(9, 7, 0);
    a.b(l_objsym);
    a.bind(l_bool);
    a.ldurb(11, 20, -8);
    a.movz(12, 1, 0);
    a.logic_w(2, 11, 11, 12); // eor: flip
    a.b(l_have);
    a.bind(l_num);
    a.movz(12, 0, 0);
    a.fmov_d_x(1, 12); // d1 = +0.0
    a.fcmp(0, 1);
    a.cset_w(11, C_EQ); // ±0 → falsy
    a.cset_w(12, C_VS); // NaN (unordered) → falsy
    a.logic_w(1, 11, 11, 12); // orr
    a.b(l_have);
    a.bind(l_str);
    emit_exec_word_load(a, 12, 20, -8);
    emit_exec_payload(a, 12, 12);
    a.ldr_w_imm(11, 12, len_off);
    a.cmp_imm_w(11, 0);
    a.cset_w(11, C_EQ); // empty → falsy
    a.b(l_have);
    a.bind(l_objsym);
    emit_exec_word_load(a, 12, 20, -8);
    emit_exec_payload(a, 12, 12);
    let os_drop = a.new_label();
    a.cmp_imm_w(9, 8);
    a.b_cond(C_NE, os_drop);
    // an Obj is only reliably truthy when it is ordinary (ic_plain rules out IsHTMLDDA)
    a.add_imm(11, 12, layout.obj_from_cell as u32);
    a.ldrb_imm(11, 11, layout.obj_ic_plain as u32);
    a.cbz(11, false, slow);
    a.bind(os_drop);
    a.movz(11, 0, 0);
    a.b(l_have);
    a.bind(l_true);
    a.movz(11, 1, 0);
    a.bind(l_have);
    a.mov_imm64(10, crate::value::PACK_BOOL);
    a.logic_x(1, 11, 11, 10);
    emit_exec_word_store(a, 11, 20, -8);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
}

/// Gate for the inline LoadName template: probed layouts hold and every baked offset fits its
/// instruction's immediate range.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn load_name_inlinable(layout: &crate::value::JitLayout) -> bool {
    // The global-mode path additionally bakes the property-IC offsets (shape/entries/accessor),
    // so it shares that gate.
    get_prop_inlinable(layout)
        && layout.scope_gen.is_multiple_of(4)
        && layout.scope_gen / 4 < 4096
        && layout.scope_layout.is_multiple_of(4)
        && layout.scope_layout / 4 < 4096
        && layout.binding_value + 16 < 256
        && layout.binding_value < 4096
        && layout.binding_init < 4096
        && layout.binding_import < 4096
        && layout.scope_with_valid
        && layout.scope_with < 4096
}

/// The cached numeric name update additionally writes through the resolved binding/property.
/// Require the descriptor and binding-mutability bytes to be directly addressable, and the
/// packed global-property representation understood by the emitted Number guard.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn update_name_inlinable(layout: &crate::value::JitLayout) -> bool {
    load_name_inlinable(layout)
        && layout.entry_accessor == layout.entry_value + 8
        && layout.entry_writable >= layout.entry_value
        && layout.entry_writable - layout.entry_value < 4096
        && layout.binding_mutable >= layout.binding_value
        && layout.binding_mutable - layout.binding_value < 4096
}

/// Inline `++`/`--` on a cached free name holding a Number. Cache validation proves the live
/// resolution and leaves x14 at the binding/property value. Mutable/writable and Number guards
/// all run before the FP update is committed; any mismatch replays the original op in Rust.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_update_name_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    kind: UpdKind,
    pc: u32,
    l_unwind: usize,
) {
    let slow = a.new_label();
    let done = a.new_label();
    let bits = a.new_label();
    let scope = a.new_label();

    // x14 -> wide Binding.value (x7=0) or packed global Property value (x7=1).
    emit_name_ic_value_ptr(a, layout, cache_ptr, slow, true);
    a.cbz(7, false, scope);

    // Global-object mode: the live data property must remain writable and contain a Number.
    guard_prop_writable(
        a,
        9,
        14,
        (layout.entry_writable - layout.entry_value) as u32,
        slow,
    );
    a.ldur(16, 14, 0);
    a.lsr_imm(9, 16, 48);
    let packed_number = a.new_label();
    a.movz(13, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_w(9, 13);
    a.b_cond(C_HS, slow);
    a.movz(13, (crate::value::PACK_UNDEFINED >> 48) as u32, 0);
    a.cmp_reg_w(9, 13);
    a.b_cond(C_LO, packed_number);
    a.movz(13, (crate::value::PACK_SYM >> 48) as u32, 0);
    a.cmp_reg_w(9, 13);
    a.b_cond(C_HI, packed_number);
    a.b(slow);
    a.bind(packed_number);
    a.b(bits);

    // Scope mode: mutability is live (const/named-expression bindings must take the slow path);
    // wide execution Values carry their Number tag beside the payload.
    a.bind(scope);
    a.ldrb_imm(
        9,
        14,
        (layout.binding_mutable - layout.binding_value) as u32,
    );
    a.cbz(9, false, slow);
    a.ldurb(9, 14, 0);
    a.cmp_imm_w(9, 4);
    a.b_cond(C_NE, slow);
    a.add_imm(14, 14, 8);

    // x14 now points at the f64 bits in either storage mode.
    a.bind(bits);
    a.ldur_d(0, 14, 0);
    a.fcmp(0, 0);
    a.b_cond(C_VS, slow); // keep NaN boxing/canonicalization on the checked path
    a.fmov_one(1);
    let dec = matches!(
        kind,
        UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
    );
    a.f_arith(if dec { 1 } else { 0 }, 2, 0, 1);
    a.stur_d(2, 14, 0);
    match kind {
        UpdKind::PreInc | UpdKind::PreDec => {
            emit_exec_number_store(a, 2, 20, 0, 9);
            a.add_imm(20, 20, 8);
        }
        UpdKind::PostInc | UpdKind::PostDec => {
            emit_exec_number_store(a, 0, 20, 0, 9);
            a.add_imm(20, 20, 8);
        }
        UpdKind::IncDiscard | UpdKind::DecDiscard => {}
    }
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
}

/// Inline a cached free-name store. Immutable bindings, non-writable/accessor globals, NaN
/// packing, and every cache miss replay through the checked executor before any stack or target
/// mutation.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_store_name_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    pc: u32,
    l_unwind: usize,
) {
    let slow = a.new_label();
    let done = a.new_label();

    emit_name_ic_value_ptr(a, layout, cache_ptr, slow, true);
    emit_store_name_value(a, layout, NameStoreOwner::Cache(cache_ptr), slow, done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
}

/// Where [`emit_store_name_value`]'s write barrier finds the store's owner, which no register
/// holds at the store (S6 step 10b): it reloads it.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy)]
enum NameStoreOwner {
    /// A free-name or captured-binding site, whose `NameIc` cell is at this address: a global
    /// property's owner is the realm's global object, a binding's the environment the cache's
    /// mode names ([`emit_name_owner_env`]).
    Cache(usize),
    /// A Reference op on this slot: the Reference's object, or its scope.
    Reference(u16),
}

/// The environment owning the binding a free-name or captured-binding cache's proof resolved
/// (`emit_name_ic_value_ptr_in` from the running environment), into x17: by the cache's mode,
/// the running environment (direct and fixed-layout modes), its parent (depth one: `ic.env` =
/// parent | 2, or | 6), or the deep descriptor's holder, the running environment's `len - 1`th
/// parent. The cache is unchanged since the proof. Clobbers x13, x15 and x16.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_name_owner_env(a: &mut asm::Asm, layout: &crate::value::JitLayout, cache_ptr: usize) {
    use crate::bytecode::lexical_cache::{NativeNameDescriptor, DEEP_NAME_IC};
    use crate::bytecode::{NAME_IC_OFF_BINDING, NAME_IC_OFF_ENV};
    use std::mem::offset_of;
    let ready = a.new_label();
    let not_depth_one = a.new_label();
    a.mov_imm64(15, cache_ptr as u64);
    a.ldr_imm(16, 15, NAME_IC_OFF_ENV);
    a.ldr_imm(17, 19, offset_of!(JitCtx, env_raw) as u32);
    a.ubfx(13, 16, 1, 1);
    a.cbz(13, true, not_depth_one);
    a.ldr_imm(17, 19, offset_of!(JitCtx, env_parent_raw) as u32);
    a.bind(not_depth_one);
    if names::deep_layout_supported(layout) {
        // Otherwise the deep proof always declines.
        let walk = a.new_label();
        a.cmp_imm_x(16, DEEP_NAME_IC as u32);
        a.b_cond(C_NE, ready);
        a.ldr_imm(15, 15, NAME_IC_OFF_BINDING);
        a.ldr_imm(15, 15, offset_of!(NativeNameDescriptor, len) as u32);
        a.bind(walk);
        a.sub_imm(15, 15, 1);
        a.cbz(15, true, ready);
        a.ldr_imm(17, 17, layout.scope_parent as u32);
        a.b(walk);
    }
    a.bind(ready);
}

/// Store to an already resolved, checked x14 value address; x7 distinguishes property/binding.
/// No subsequent operation may resolve the identifier again. Shared by name and Reference ICs;
/// `owner` says where the write barrier reloads the store's owner.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_store_name_value(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    owner: NameStoreOwner,
    slow: usize,
    done: usize,
) {
    let scope = a.new_label();
    a.cbz(7, false, scope);

    // Packed global property: the moved stack word overwrites the old one.
    guard_prop_writable(
        a,
        9,
        14,
        (layout.entry_writable - layout.entry_value) as u32,
        slow,
    );
    // Both stores hold the same compact word; no repacking or allocation is necessary.
    emit_exec_word_load(a, 10, 20, -8);
    // The old word is released by overwriting it, a function's deferred prototype included (a
    // cell, which dies once no word names it): no load, test or helper call.
    a.stur(10, 14, 0);
    // The write barrier (site C): the property's owner, reloaded.
    match owner {
        NameStoreOwner::Cache(_) => {
            // The realm's global Object (its cell's start + obj_from_cell, in the cell's card).
            debug_assert_object_in_cell_card(layout);
            a.ldr_imm(15, 19, std::mem::offset_of!(JitCtx, global_body) as u32);
        }
        NameStoreOwner::Reference(_) => {
            // x8 = the PreparedReference; its object is a cell start.
            a.ldr_imm(
                15,
                8,
                std::mem::offset_of!(crate::eval::PreparedReference, object) as u32,
            );
        }
    }
    emit_card_mark(a, 15, 15, 16);
    a.sub_imm(20, 20, 8);
    a.b(done);

    // Wide scope binding: validate mutability. The incoming Value moves from the operand stack
    // into the binding, and the old value is released by overwriting it.
    a.bind(scope);
    a.ldrb_imm(
        9,
        14,
        (layout.binding_mutable - layout.binding_value) as u32,
    );
    a.cbz(9, false, slow);
    emit_exec_word_load(a, 8, 20, -8);
    // A Number stored over a Number (the common update of a closure variable such as a stack
    // pointer) overwrites the payload: the discriminant stays and neither word owns anything.
    {
        let generic = a.new_label();
        let number = a.new_label();
        a.fmov_d_x(0, 8);
        a.fcmp(0, 0);
        a.b_cond(7, number);
        a.mov_imm64(9, crate::value::PACK_CANON_NAN);
        a.cmp_reg_x(8, 9);
        a.b_cond(C_NE, generic);
        a.bind(number);
        a.ldurb(9, 14, 0);
        a.cmp_imm_w(9, 4);
        a.b_cond(C_NE, generic);
        a.stur(8, 14, 8);
        a.sub_imm(20, 20, 8);
        a.b(done);
        a.bind(generic);
    }
    // The decode below precedes both stores.
    emit_exec_word_load(a, 8, 20, -8);
    emit_exec_decode_wide(a, 8, 9, 10, 11, 13, slow);
    a.stur(9, 14, 0);
    a.stur(10, 14, 8);
    // The write barrier (site D): the binding's environment, reloaded.
    match owner {
        NameStoreOwner::Cache(cache_ptr) => emit_name_owner_env(a, layout, cache_ptr),
        NameStoreOwner::Reference(reference) => {
            use crate::eval::{PreparedReference as R, PreparedReferenceSlot as S};
            use std::mem::{offset_of, size_of};
            // x8 held the PreparedReference until the operand load above.
            a.ldr_imm(17, 19, offset_of!(JitCtx, references_raw) as u32);
            a.mov_imm64(
                16,
                (reference as usize * size_of::<S>()
                    + offset_of!(S, reference)
                    + offset_of!(R, scope)) as u64,
            );
            a.add_shifted(17, 17, 16, 0);
            a.ldr_imm(17, 17, 0);
        }
    }
    emit_card_mark(a, 17, 17, 16);
    a.sub_imm(20, 20, 8);
    a.b(done);
}

/// Inline free-name read (`LoadName`) against the per-site [`crate::bytecode::NameIc`]. Ordinary
/// scope/global/depth-one modes and bounded deep-chain proofs validate the current resolution
/// before cloning its live value, without hashing or a helper on a supported hit. Cold/stale
/// proofs, live with/import/TDZ state and unsupported value encodings take the checked helper.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_load_name_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    pc: u32,
    l_unwind: usize,
    for_call: bool,
) {
    a.mark_push("load-name");
    emit_load_name_inline_marked(a, layout, cache_ptr, pc, l_unwind, for_call);
    a.mark_pop();
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_load_name_inline_marked(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    pc: u32,
    l_unwind: usize,
    // `LoadNameForCall`: the fast path pushes the `this` slot (Undefined — a depth-0 hit can't
    // come through a `with` object) below the value; the slow path runs the full op.
    for_call: bool,
) {
    let slow = a.new_label();
    let done = a.new_label();
    if use_shared_stub(a) {
        // Straight-line code: the whole read is one call of the chunk's shared stub, which
        // validates the cache and pushes the clone; a miss falls through to the checked helper.
        a.mov_imm64(12, cache_ptr as u64);
        let stub = a.shared_stub(SharedStub::NameLoad { for_call }.key());
        a.bl_label(stub);
        a.cbnz(9, false, done);
        emit_exec(a, pc, l_unwind);
        a.bind(done);
        return;
    }
    // Validate the cache and leave a pointer to the resolved Value in x14 (either mode).
    emit_name_ic_value_ptr(a, layout, cache_ptr, slow, true);
    emit_load_name_value(a, layout, slow, for_call, false);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
}

/// `Op::LoadNameIn` / `Op::LoadNameForCallIn`: [`emit_load_name_inline`] resolving from a fixed
/// inline target's [[Environment]] (`env`, an `Env::as_ptr` scope address). Emitted at each site
/// (in the chunk's shared fixed-environment name stub unless stubs are inlined).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[allow(clippy::too_many_arguments)]
fn emit_load_name_in_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    env: u64,
    pc: u32,
    l_unwind: usize,
    for_call: bool,
) {
    let slow = a.new_label();
    let done = a.new_label();
    a.mov_imm64(12, cache_ptr as u64);
    a.mov_imm64(9, env);
    if use_shared_stub(a) {
        let stub = a.shared_stub(SharedStub::NameValuePtrIn.key());
        a.bl_label(stub);
        a.cbz(9, false, slow);
    } else {
        emit_name_ic_value_ptr_in(a, layout, slow, true, NameEnv::InX9);
    }
    emit_load_name_value(a, layout, slow, for_call, false);
    a.b(done);
    a.bind(slow);
    emit_exec(a, pc, l_unwind);
    a.bind(done);
}

/// Clone from a checked x14 value address into the canonical operand stack (x7=storage kind).
/// `prototype_key`: a `LoadRef` whose Reference names `prototype`. Its packed holder can be a
/// `with` object that is a function whose prototype is still deferred; the packed holder of a
/// free-name read is the global object, never a function.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_load_name_value(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    for_call: bool,
    prototype_key: bool,
) {
    // Lexical Binding.value stays wide; object properties already share execution encoding.
    let loaded = a.new_label();
    if layout.entry_accessor == layout.entry_value + 8 {
        let wide = a.new_label();
        a.cbz(7, false, wide);
        a.ldur(12, 14, 0);
        if prototype_key {
            property_word_guard(a, 12, 13, slow);
        }
        a.b(loaded);
        a.bind(wide);
        // The wide lexical binding of a closure variable usually holds a Number (a counter, a
        // stack pointer) or an Object (an array view): decode those two without the general
        // encoder. A Number word is its payload (a NaN is the canonical one); an Object
        // word is the Object tag over the stored pointer.
        let generic = a.new_label();
        let wide_number = a.new_label();
        a.ldurb(10, 14, 0);
        a.cmp_imm_w(10, 4);
        a.b_cond(C_EQ, wide_number);
        a.cmp_imm_w(10, 8);
        a.b_cond(C_NE, generic);
        a.ldur(11, 14, 8);
        a.mov_imm64(12, crate::value::PACK_OBJ);
        a.logic_x(1, 12, 12, 11);
        a.b(loaded);
        a.bind(wide_number);
        a.ldur(12, 14, 8);
        a.fmov_d_x(0, 12);
        a.fcmp(0, 0);
        a.b_cond(7, loaded);
        a.mov_imm64(12, crate::value::PACK_CANON_NAN);
        a.b(loaded);
        a.bind(generic);
    }
    a.ldur(10, 14, 0);
    a.ldur(11, 14, 8);
    emit_exec_encode_wide(a, 10, 11, 12, 13, 16, 0, slow);
    a.bind(loaded);
    if for_call {
        a.mov_imm64(9, crate::value::PACK_UNDEFINED);
        emit_exec_word_store(a, 9, 20, 0);
        a.add_imm(20, 20, 8);
    }
    emit_exec_word_store(a, 12, 20, 0);
    a.add_imm(20, 20, 8);
}

/// Shared LoadName cache validation: on success x14 points at the resolved `Value` (the binding's
/// value in scope mode, the global entry's value in global mode) and execution falls through; any
/// mismatch branches to `slow`. Clobbers x7 and x9-x17.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// Out-of-line routines emitted once per compiled chunk and reached from each site with `bl`.
/// The site passes arguments in fixed registers; a stub is a leaf (it calls nothing and keeps the
/// machine stack untouched), clobbers exactly the registers its inline form would, and reports
/// hit/miss in w9. Sharing them keeps per-site code small: before, every free-name read carried
/// the complete multi-mode cache proof, and compiled applications reached megabytes of
/// generated code (instruction-fetch stalls). `TRUST_JS_JIT_INLINE_STUBS=1` restores inline
/// emission for comparison.
#[derive(Clone, Copy)]
enum SharedStub {
    /// [`emit_name_ic_value_ptr_body`]: x12 = `NameIc` cell → w9 = hit, x14 = value, x7 = kind.
    NameValuePtr { packed_ok: bool },
    /// [`emit_name_ic_value_ptr_in`] from the fixed scope in x9 (`Op::LoadNameIn`), with the
    /// outputs of [`SharedStub::NameValuePtr`].
    NameValuePtrIn,
    /// A whole free-name read (`Op::LoadName`, or `Op::LoadNameForCall` with its `this` slot):
    /// the cache proof of [`SharedStub::NameValuePtr`], then the value's clone onto the
    /// operand stack at x20 (advanced). x12 = the site's `NameIc` cell → w9 = 1 when the value
    /// is pushed; w9 = 0 leaves every operand untouched for the site's checked helper.
    NameLoad { for_call: bool },
    /// The checked helper call of an operation's slow path (`emit_op_helper`): x1 = the
    /// site's pc → x20 = the helper's new operand top, x1 = its flag (nonzero: threw).
    Helper { idx: u8 },
    /// [`emit_prop_way_loop`]: x8 = a site's first `IcState` cell, x10 = receiver → w9 = 0 or a
    /// `PROP_PROBE_*` landing, with x11 = holder base and x13 = slot for the data landings.
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    PropWays(PropProbeFlags),
    /// [`emit_call_secondary_probes`]: the call-site probe inputs plus x12 = the site's first
    /// way → w9 = hit, with x12/x15 set as at the primary hit label.
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    CallSecondary,
    /// [`emit_direct_call`] for one arity and receiver form, entered at a call site's primary
    /// hit with its registers → w9 = `DIRECT_CALL_*` (a declined gate keeps x10/x12/x15 for
    /// the site's H_CALL_HIT form).
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    DirectCall { argc: u8, with_this: bool },
    /// [`emit_call_intrinsics1`] for a `CallWithThis(1)` site's live call-IC hit, with w8 = the
    /// site's pc → w9 = `INTRINSIC_*` (no-intrinsic and declined keep x10/x12/x15).
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    Intrinsics1 { array_push: bool, discard: bool },
    /// [`emit_call_intrinsics2`] for a `CallWithThis(2)` site, with the same contract.
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    Intrinsics2 { discard: bool },
    /// [`typed_array::emit_access`]: x11 = a typed array Object, x9 = its exact unsigned index
    /// (and d1 the Number to store) → w13 = 1 once the element is read (x12 = packed Number,
    /// d1 = value) or written, 0 when the access is declined before anything is written.
    TypedElement { write: bool },
    /// [`regions::emit_name_element_body`]: a fused `LoadName K; key; GetElem/SetElemDrop` access
    /// with x12 = the name's `NameIc` cell, d3 = the key and (for a write) d1 = the Number to
    /// store → w13 = 1 after the access (a read leaves x12 = packed Number, d1 = value), 2 when
    /// the binding validated but is not for the native path (x14 = its value, x7 = its storage
    /// kind), 0 otherwise.
    NameElement { write: bool },
}

/// Status codes of the shared direct-call stub. The common completion is zero, so a site pays
/// one untaken branch for it.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const DIRECT_CALL_RETURNED: u32 = 0;
/// A gate declined before anything was mutated (the site continues with H_CALL_HIT).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const DIRECT_CALL_DECLINED: u32 = 1;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const DIRECT_CALL_THREW: u32 = 2;

/// Status codes of the shared intrinsic stubs (see [`emit_intrinsic_stub_call`]).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const INTRINSIC_DONE: u32 = 0;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const INTRINSIC_NONE: u32 = 1;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const INTRINSIC_DECLINED: u32 = 2;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const INTRINSIC_THREW: u32 = 3;

/// What shared stubs may need beyond the object layout (see [`SharedStub::emit`]).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
struct StubContext<'a> {
    layout: &'a crate::value::JitLayout,
    /// The interpreter layout, for call stubs (absent in layout-only test harnesses).
    ilayout: Option<&'a crate::interpreter::InterpLayout>,
    /// Chunk-derived direct-call inputs (see [`DirectCallContext`]).
    direct: Option<DirectCallContext>,
}

/// The direct-call sequence's chunk-level inputs: the `Chunk` field offsets it reads from the
/// callee (every chunk shares the monomorphized layout) and the chunk's finish-stub label.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy)]
struct DirectCallContext {
    attempted_off: usize,
    runs_off: usize,
    retry_off: usize,
    finish_stub: usize,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
impl SharedStub {
    const NAME_VALUE_PTR: u32 = 0x000;
    const PROP_WAYS: u32 = 0x100;
    const CALL_SECONDARY: u32 = 0x200;
    const DIRECT_CALL: u32 = 0x300;
    const NAME_VALUE_PTR_IN: u32 = 0x400;
    const INTRINSICS1: u32 = 0x500;
    const INTRINSICS2: u32 = 0x600;
    const NAME_LOAD: u32 = 0x700;
    const HELPER: u32 = 0x800;
    const TYPED_ELEMENT: u32 = 0x900;
    const NAME_ELEMENT: u32 = 0xa00;

    fn key(self) -> u32 {
        match self {
            SharedStub::NameValuePtr { packed_ok } => Self::NAME_VALUE_PTR | u32::from(packed_ok),
            SharedStub::NameValuePtrIn => Self::NAME_VALUE_PTR_IN,
            SharedStub::NameLoad { for_call } => Self::NAME_LOAD | u32::from(for_call),
            SharedStub::Helper { idx } => Self::HELPER | u32::from(idx),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::PropWays(flags) => {
                Self::PROP_WAYS
                    | u32::from(flags.arr_ok)
                    | u32::from(flags.wrap_ok) << 1
                    | u32::from(flags.method) << 2
                    | u32::from(flags.kc) << 3
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::CallSecondary => Self::CALL_SECONDARY,
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::DirectCall { argc, with_this } => {
                debug_assert!(argc < 128);
                Self::DIRECT_CALL | u32::from(argc) | u32::from(with_this) << 7
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::Intrinsics1 {
                array_push,
                discard,
            } => Self::INTRINSICS1 | u32::from(array_push) | u32::from(discard) << 1,
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::Intrinsics2 { discard } => Self::INTRINSICS2 | u32::from(discard),
            SharedStub::TypedElement { write } => Self::TYPED_ELEMENT | u32::from(write),
            SharedStub::NameElement { write } => Self::NAME_ELEMENT | u32::from(write),
        }
    }

    fn from_key(key: u32) -> SharedStub {
        match key & !0xff {
            Self::NAME_VALUE_PTR => SharedStub::NameValuePtr {
                packed_ok: key & 1 != 0,
            },
            Self::NAME_VALUE_PTR_IN => SharedStub::NameValuePtrIn,
            Self::NAME_LOAD => SharedStub::NameLoad {
                for_call: key & 1 != 0,
            },
            Self::HELPER => SharedStub::Helper {
                idx: (key & 0xff) as u8,
            },
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            Self::PROP_WAYS => SharedStub::PropWays(PropProbeFlags {
                arr_ok: key & 1 != 0,
                wrap_ok: key & 2 != 0,
                method: key & 4 != 0,
                kc: key & 8 != 0,
            }),
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            Self::CALL_SECONDARY => SharedStub::CallSecondary,
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            Self::DIRECT_CALL => SharedStub::DirectCall {
                argc: (key & 0x7f) as u8,
                with_this: key & 0x80 != 0,
            },
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            Self::INTRINSICS1 => SharedStub::Intrinsics1 {
                array_push: key & 1 != 0,
                discard: key & 2 != 0,
            },
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            Self::INTRINSICS2 => SharedStub::Intrinsics2 {
                discard: key & 1 != 0,
            },
            Self::TYPED_ELEMENT => SharedStub::TypedElement {
                write: key & 1 != 0,
            },
            Self::NAME_ELEMENT => SharedStub::NameElement {
                write: key & 1 != 0,
            },
            _ => unreachable!("shared stub keys are produced by SharedStub::key"),
        }
    }

    /// The stub body at the current position (its entry label is already bound).
    fn emit(self, a: &mut asm::Asm, context: &StubContext<'_>) {
        let layout = context.layout;
        match self {
            SharedStub::NameValuePtr { packed_ok } => {
                let miss = a.new_label();
                emit_name_ic_value_ptr_body(a, layout, miss, packed_ok);
                a.movz(9, 1, 0);
                a.ret();
                a.bind(miss);
                a.movz(9, 0, 0);
                a.ret();
            }
            SharedStub::NameValuePtrIn => {
                let miss = a.new_label();
                emit_name_ic_value_ptr_in(a, layout, miss, true, NameEnv::InX9);
                a.movz(9, 1, 0);
                a.ret();
                a.bind(miss);
                a.movz(9, 0, 0);
                a.ret();
            }
            SharedStub::Helper { idx } => {
                // The site has set x1 = pc; the call preserves nothing the body keeps in
                // caller-saved registers, exactly like the inline sequence it replaces.
                a.mov(0, 19);
                a.mov(2, 20);
                a.ldr_imm(16, 21, (usize::from(idx) * 8) as u32);
                a.stp_pre(29, 30, -16);
                a.blr(16);
                a.ldp_post(29, 30, 16);
                a.mov(20, 0);
                a.ret();
            }
            SharedStub::NameLoad { for_call } => {
                let miss = a.new_label();
                emit_name_ic_value_ptr_body(a, layout, miss, true);
                emit_load_name_value(a, layout, miss, for_call, false);
                a.movz(9, 1, 0);
                a.ret();
                a.bind(miss);
                a.movz(9, 0, 0);
                a.ret();
            }
            SharedStub::NameElement { write } => {
                let hit = a.new_label();
                let value_miss = a.new_label();
                let name_miss = a.new_label();
                regions::emit_name_element_body(a, layout, write, true, hit, name_miss, value_miss);
                a.bind(hit);
                a.movz(13, 1, 0);
                a.ret();
                a.bind(value_miss);
                a.movz(13, 2, 0);
                a.ret();
                a.bind(name_miss);
                a.movz(13, 0, 0);
                a.ret();
            }
            SharedStub::TypedElement { write } => {
                let hit = a.new_label();
                let miss = a.new_label();
                match layout.native_typed_array {
                    Some(native) => typed_array::emit_access(a, &native, write, hit, miss),
                    None => a.b(miss),
                }
                a.bind(hit);
                a.movz(13, 1, 0);
                a.ret();
                a.bind(miss);
                a.movz(13, 0, 0);
                a.ret();
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::PropWays(flags) => {
                let miss = a.new_label();
                let load = a.new_label();
                let load_kc = a.new_label();
                let absent = a.new_label();
                emit_prop_way_loop(a, layout, flags, miss, load, load_kc, absent);
                for (landing, status) in [
                    (miss, 0),
                    (load, PROP_PROBE_LOAD),
                    (load_kc, PROP_PROBE_LOAD_KC),
                    (absent, PROP_PROBE_ABSENT),
                ] {
                    a.bind(landing);
                    a.movz(9, status, 0);
                    a.ret();
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::CallSecondary => {
                let ilayout = context
                    .ilayout
                    .expect("call stubs are requested only by chunk compilation");
                let hit = a.new_label();
                let miss = a.new_label();
                a.mov(6, 12);
                emit_call_secondary_probes(a, layout, ilayout, hit, miss);
                a.bind(hit);
                a.movz(9, 1, 0);
                a.ret();
                a.bind(miss);
                a.movz(9, 0, 0);
                a.ret();
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::Intrinsics1 { .. } | SharedStub::Intrinsics2 { .. } => {
                // The intrinsic helpers are calls: keep the site's return address in a frame of
                // the stub's own, released before every status return.
                a.stp_pre(29, 30, -16);
                let none = a.new_label();
                let declined = a.new_label();
                let done = a.new_label();
                let threw = a.new_label();
                match self {
                    SharedStub::Intrinsics1 {
                        array_push,
                        discard,
                    } => emit_call_intrinsics1(
                        a,
                        layout,
                        IntrinsicPc::InW8,
                        array_push,
                        discard,
                        none,
                        declined,
                        done,
                        threw,
                    ),
                    SharedStub::Intrinsics2 { discard } => emit_call_intrinsics2(
                        a,
                        layout,
                        IntrinsicPc::InW8,
                        discard,
                        none,
                        declined,
                        done,
                        threw,
                    ),
                    _ => unreachable!(),
                }
                for (landing, status) in [
                    (done, INTRINSIC_DONE),
                    (none, INTRINSIC_NONE),
                    (declined, INTRINSIC_DECLINED),
                    (threw, INTRINSIC_THREW),
                ] {
                    a.bind(landing);
                    a.ldp_post(29, 30, 16);
                    a.movz(9, status, 0);
                    a.ret();
                }
            }
            #[cfg(all(
                target_arch = "aarch64",
                any(target_os = "macos", target_os = "linux", target_os = "windows")
            ))]
            SharedStub::DirectCall { argc, with_this } => {
                let ilayout = context
                    .ilayout
                    .expect("call stubs are requested only by chunk compilation");
                let direct = context
                    .direct
                    .expect("direct-call stubs are requested only by chunk compilation");
                // The sequence calls the callee and the finish stub: keep the site's return
                // address in a 16-byte frame of its own. Every landing below is reached with the
                // sequence's own save area already released.
                a.stp_pre(29, 30, -16);
                let declined = a.new_label();
                let returned = a.new_label();
                let threw = a.new_label();
                let emitted = emit_direct_call(
                    a,
                    ilayout,
                    layout,
                    direct.attempted_off,
                    direct.runs_off,
                    direct.retry_off,
                    usize::from(argc),
                    with_this,
                    declined,
                    threw,
                    returned,
                    direct.finish_stub,
                    DirectCallSite::AfterReturn,
                );
                assert!(emitted, "sites request the stub only when it is supported");
                for (landing, status) in [
                    (returned, DIRECT_CALL_RETURNED),
                    (declined, DIRECT_CALL_DECLINED),
                    (threw, DIRECT_CALL_THREW),
                ] {
                    a.bind(landing);
                    a.ldp_post(29, 30, 16);
                    a.movz(9, status, 0);
                    a.ret();
                }
            }
        }
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn shared_stubs_enabled() -> bool {
    static INLINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*INLINE.get_or_init(|| std::env::var_os("TRUST_JS_JIT_INLINE_STUBS").is_some())
}

/// Whether the site being emitted calls the chunk's shared stub for a sequence it runs on every
/// execution (name-cache validation, the direct-call sequence) rather than carrying it in line.
/// Shared stubs keep straight-line code — most of an application's code, run once per call —
/// compact; in a loop body the call and return cost more than the code size saves. Stubs a
/// site reaches only after its primary cache way misses (the property way loop, the secondary
/// call probes) stay shared everywhere: inlining them in loops measured +6-8% code for under 2%
/// on polymorphic loops.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn use_shared_stub(a: &asm::Asm) -> bool {
    shared_stubs_enabled() && !a.hot()
}

/// Loops whose bodies span at most this many operations keep their sites' shared-stub
/// sequences in line (see [`loop_body_mask`]). A larger body, such as the dispatch loop of
/// control-flow-flattened code, spends far more per iteration than a stub call and return, and
/// inlining every site in it multiplies the emitted code. `TRUST_JS_JIT_HOT_LOOP_OPS` overrides it.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn hot_loop_ops() -> usize {
    static OPS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *OPS.get_or_init(|| {
        std::env::var("TRUST_JS_JIT_HOT_LOOP_OPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(HOT_LOOP_OPS)
    })
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const HOT_LOOP_OPS: usize = 512;

/// The operations of `ops` inside a loop body of at most [`hot_loop_ops`] operations: from a
/// backward branch's target through the branch itself (nested and overlapping loops alike).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn loop_body_mask(ops: &[crate::bytecode::Op]) -> Vec<bool> {
    use crate::bytecode::Op;
    let limit = hot_loop_ops();
    let mut depth = vec![0i32; ops.len() + 1];
    for (pc, op) in ops.iter().enumerate() {
        if let Op::Jump(target)
        | Op::AbruptJump(target, _)
        | Op::JumpIfFalse(target)
        | Op::JumpIfFalsePeek(target)
        | Op::JumpIfTruePeek(target)
        | Op::JumpIfNotNullishPeek(target) = *op
        {
            if target as usize <= pc && pc - target as usize <= limit {
                depth[target as usize] += 1;
                depth[pc + 1] -= 1;
            }
        }
    }
    let mut open = 0;
    ops.iter()
        .enumerate()
        .map(|(pc, _)| {
            open += depth[pc];
            open > 0
        })
        .collect()
}

/// Emit every shared stub the chunk's sites requested (a stub may itself request others). A
/// stub with a process-wide copy (see [`global_stub_address`]) gets a two-word-target veneer
/// instead of its body.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_shared_stubs(a: &mut asm::Asm, context: &StubContext<'_>) {
    loop {
        let requested = a.take_shared_stubs();
        if requested.is_empty() {
            return;
        }
        for (key, label) in requested {
            a.bind(label);
            match global_stub_address(key, context) {
                // Every stub's contract lets it clobber x16, and `br` keeps the site's return
                // address in x30 for the stub's own `ret`.
                Some(address) => {
                    a.mov_imm64(16, address);
                    a.br(16);
                }
                None => SharedStub::from_key(key).emit(a, context),
            }
        }
    }
}

/// The inputs a shared stub's body is assembled from. They are process constants (probed type
/// layouts and `Chunk` field offsets); a context that differs from the first one keeps its
/// stubs per chunk.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
struct GlobalStubTable {
    layout: crate::value::JitLayout,
    ilayout: crate::interpreter::InterpLayout,
    direct: Option<(usize, usize, usize)>,
    addresses: crate::fasthash::FastMap<u32, u64>,
}

#[cfg(all(
    not(test),
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
static GLOBAL_STUBS: std::sync::Mutex<Option<GlobalStubTable>> = std::sync::Mutex::new(None);

// Test builds instrument some stub bodies with the addresses of thread-local counters, so a
// body is only valid on the thread that assembled it: keep one table per thread there.
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
thread_local! {
    static GLOBAL_STUBS: std::cell::RefCell<Option<GlobalStubTable>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn with_global_stubs<R>(f: impl FnOnce(&mut Option<GlobalStubTable>) -> R) -> Option<R> {
    #[cfg(not(test))]
    {
        let mut table = GLOBAL_STUBS.lock().ok()?;
        Some(f(&mut table))
    }
    #[cfg(test)]
    {
        GLOBAL_STUBS.with(|table| Some(f(&mut table.borrow_mut())))
    }
}

/// Shared stubs read no chunk state: their inputs arrive in registers and their outputs are
/// status codes, and the direct-call stub's finish stub is equally chunk-independent. So each
/// stub body is assembled once per process into executable memory that is never reclaimed, and
/// chunks branch to it through a veneer instead of carrying a copy each (the Cloudflare
/// challenge emitted about 8 MB of such copies per load). `TRUST_JS_JIT_CHUNK_STUBS=1` keeps
/// per-chunk copies for diagnosis.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn global_stub_address(key: u32, context: &StubContext<'_>) -> Option<u64> {
    if env_flag!("TRUST_JS_JIT_CHUNK_STUBS") {
        return None;
    }
    let ilayout = *context.ilayout?;
    let direct = context
        .direct
        .map(|direct| (direct.attempted_off, direct.runs_off, direct.retry_off));
    let is_direct = matches!(SharedStub::from_key(key), SharedStub::DirectCall { .. });
    if is_direct && direct.is_none() {
        return None;
    }
    with_global_stubs(|table| {
        let table = table.get_or_insert_with(|| GlobalStubTable {
            layout: *context.layout,
            ilayout,
            direct,
            addresses: Default::default(),
        });
        if table.layout != *context.layout || table.ilayout != ilayout {
            return None;
        }
        if is_direct {
            match table.direct {
                Some(known) if Some(known) != direct => return None,
                Some(_) => {}
                None => table.direct = direct,
            }
        }
        if let Some(&address) = table.addresses.get(&key) {
            return Some(address);
        }
        let address = assemble_global_stub(key, context)?;
        table.addresses.insert(key, address);
        Some(address)
    })?
}

/// Assemble one shared stub (with any stubs it requests, and the direct-call finish stub) into
/// its own permanent executable buffer; the entry address, or `None` when memory is refused.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn assemble_global_stub(key: u32, context: &StubContext<'_>) -> Option<u64> {
    let mut a = asm::Asm::new();
    let entry = a.new_label();
    let finish = a.new_label();
    let local = StubContext {
        layout: context.layout,
        ilayout: context.ilayout,
        direct: context.direct.map(|direct| DirectCallContext {
            finish_stub: finish,
            ..direct
        }),
    };
    a.bind(entry);
    SharedStub::from_key(key).emit(&mut a, &local);
    loop {
        let requested = a.take_shared_stubs();
        if requested.is_empty() {
            break;
        }
        for (nested, label) in requested {
            a.bind(label);
            SharedStub::from_key(nested).emit(&mut a, &local);
        }
    }
    a.bind(finish);
    if matches!(SharedStub::from_key(key), SharedStub::DirectCall { .. }) {
        emit_direct_finish_stub(&mut a, context.ilayout?, context.layout.valid);
    }
    let (words, offsets) = a.finish_with_offsets(&[entry]);
    let executable = ExecutableBuffer::from_bytes(unsafe {
        std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4)
    })?;
    let address = executable.as_ptr() as u64 + u64::from(offsets[0]);
    // Permanent: any compiled chunk may branch here for the rest of the process.
    std::mem::forget(executable);
    Some(address)
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_name_ic_value_ptr(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    cache_ptr: usize,
    slow: usize,
    packed_ok: bool,
) {
    a.mov_imm64(12, cache_ptr as u64);
    if use_shared_stub(a) {
        let stub = a.shared_stub(SharedStub::NameValuePtr { packed_ok }.key());
        a.bl_label(stub);
        a.cbz(9, false, slow);
    } else {
        emit_name_ic_value_ptr_body(a, layout, slow, packed_ok);
    }
}

/// The name-cache validation shared by every free-name site (see
/// [`emit_name_ic_value_ptr`]): x12 = the site's `NameIc` cell. On success x14 points at the
/// resolved Value (x7 = 0 for a wide scope binding, 1 for a packed global property); a failed
/// proof branches to `slow`. Clobbers x7 and x9..x17 only — callers keep chain values in x2..x8
/// and FP registers — and calls nothing.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_name_ic_value_ptr_body(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    packed_ok: bool,
) {
    emit_name_ic_value_ptr_in(a, layout, slow, packed_ok, NameEnv::Running);
}

/// Where a free-name cache probe starts its Environment Record walk.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum NameEnv {
    /// The running environment (`ctx.env_raw`).
    Running,
    /// A fixed scope (`Env::as_ptr`) the site already placed in x9 (`Op::LoadNameIn`).
    InX9,
}

/// [`emit_name_ic_value_ptr_body`] with the starting Environment Record either the running one
/// or a fixed scope in x9 (`Op::LoadNameIn`: an inline target's [[Environment]], alive while
/// its identity guard holds). A fixed scope derives the depth-1 parent from its live `parent`
/// field rather than the frame's cached one.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_name_ic_value_ptr_in(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    packed_ok: bool,
    env: NameEnv,
) {
    use crate::bytecode::{
        NAME_IC_OFF_ACT_GEN, NAME_IC_OFF_BINDING, NAME_IC_OFF_ENV, NAME_IC_OFF_GEN,
    };
    let sg = layout.scope_gen as u32;
    let bv = layout.binding_value as u32;
    let bi = layout.binding_init as u32;
    let g_ex = layout.obj_exotic as u32;
    let g_sh = (layout.obj_props + layout.props_shape) as u32;
    let g_ea = layout.entry_accessor as u32;
    let g_ev = layout.entry_value as u32;
    let g_es = layout.entry_size as u64;
    let none_tag = layout.exotic_none_tag as u32;

    if env == NameEnv::Running {
        a.ldr_imm(9, 19, 40); // ctx.env_raw
    }
    a.ldrb_imm(17, 9, layout.scope_with as u32);
    a.cmp_imm_w(17, layout.scope_with_none as u32);
    a.b_cond(C_NE, slow);
    a.ldr_imm(10, 12, NAME_IC_OFF_ENV);
    let deep = a.new_label();
    let have = a.new_label();
    let fixed_scope = a.new_label();
    let binding_ready = a.new_label();
    // Classify the cache mode before choosing which scope's generation `gen` describes.
    a.ldr_w_imm(11, 9, sg);
    a.ldr_w_imm(13, 12, NAME_IC_OFF_GEN);
    // Saturated generations are never pointer proofs, even if a malformed/stale IC matches.
    // ADDS WZR, Wgen, #1 is zero exactly at u32::MAX; no temporary constant is needed.
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    let scope = a.new_label();
    let direct_scope = a.new_label();
    let depth_one = a.new_label();
    a.cmp_reg_x(9, 10);
    a.b_cond(C_EQ, direct_scope);
    // Tagged depth-1 mode (`ic.env = parent|2`) validates the fresh activation by its
    // chunk-stable generation, then the live parent identity and generation.
    a.movz(15, 2, 0);
    a.logic_x(0, 16, 10, 15);
    a.cbnz(16, false, depth_one);
    // --- global mode: ic.env == env|1 (env is ≥8-aligned, so +1 sets the tag bit) ---
    // Test identity before generation: DEEP_NAME_IC publishes gen=0, which need not match
    // this scope. Keep w11 intact for the ordinary global generation proof below.
    a.add_imm(17, 9, 1);
    a.cmp_reg_x(17, 10);
    a.b_cond(C_NE, deep);
    a.cmp_reg_w(11, 13);
    a.b_cond(C_NE, slow);
    if layout.entry_accessor == layout.entry_value + 8 && !packed_ok {
        // Global bindings live in packed properties; scope bindings below remain wide. Keep the
        // global arm checked until it shares the packed decoder with GetProp.
        a.b(slow);
    }
    a.ldr_imm(14, 19, 56); // the realm's global Object
    a.ldrb_imm(15, 14, g_ex);
    a.cmp_imm_w(15, none_tag);
    a.b_cond(C_NE, slow);
    a.ldrb_imm(15, 14, layout.obj_ic_plain as u32); // not side-table masked
    a.cbz(15, false, slow);
    a.ldr_w_imm(15, 14, g_sh); // live shape vs cached (packed high half)
    a.ldr_imm(16, 12, NAME_IC_OFF_BINDING);
    a.lsr_imm(17, 16, 32);
    a.cmp_reg_w(15, 17);
    a.b_cond(C_NE, slow);
    a.mov_w(16, 16); // zero-extend the slot half
    fields_data(a, layout, 15, 14);
    a.mov_imm64(17, g_es);
    a.madd(15, 16, 17, 15);
    guard_prop_data(a, 14, 15, g_ea, slow);
    a.add_imm(14, 15, g_ev); // x14 → the entry's Value
    a.movz(7, 1, 0); // packed global property
    a.b(have);
    // The uncommon chain proof lives off the successful ordinary paths. GetValue still
    // checks live with/import/TDZ state; only mode dispatch has moved (ECMA-262 sec-getvalue,
    // sec-getidentifierreference). Empty or stale unrelated identities cannot reach it.
    a.bind(deep);
    a.cmp_imm_x(10, crate::bytecode::FIXED_NAME_IC as u32);
    a.b_cond(C_EQ, fixed_scope);
    a.cmp_imm_x(10, crate::bytecode::lexical_cache::DEEP_NAME_IC as u32);
    a.b_cond(C_NE, slow);
    a.ldr_imm(12, 12, NAME_IC_OFF_BINDING);
    names::emit_deep_name_value_ptr(a, layout, slow, packed_ok);
    a.b(have);
    // Same published ordered layout, different activation allocation. Derive
    // this invocation's Binding address; no previous invocation is retained.
    a.bind(fixed_scope);
    if scope_entries_fit(layout) {
        a.cbz(13, false, slow); // layout ID zero never proves a slot
        a.ldr_w_imm(11, 9, layout.scope_layout as u32);
        a.cmp_reg_w(11, 13);
        a.b_cond(C_NE, slow);
        a.ldrb_imm(11, 9, layout.scope_small_tag as u32);
        a.cmp_imm_w(11, 1); // reject Large; Small(0) and Indexed(2) share the probed entries
        a.b_cond(C_EQ, slow);
        a.cmp_imm_w(11, 2);
        a.b_cond(C_HI, slow);
        a.ldr_imm(14, 12, NAME_IC_OFF_BINDING);
        scope_entries_len(a, layout, 11, 9);
        a.cmp_reg_x(14, 11);
        a.b_cond(C_HS, slow);
        scope_entries_data(a, layout, 11, 9);
        a.mov_imm64(15, layout.scope_binding_stride as u64);
        a.madd(14, 14, 15, 11);
        a.add_imm(14, 14, layout.scope_binding_offset as u32);
        a.b(binding_ready);
    } else {
        a.b(slow);
    }
    // --- depth-1 mode: `act_gen` belongs to current env; `gen` belongs to parent ---
    a.bind(depth_one);
    a.ldr_w_imm(15, 12, NAME_IC_OFF_ACT_GEN);
    let activation_guard = a.new_label();
    let parent_tagged = a.new_label();
    a.movz(16, 4, 0);
    a.logic_x(0, 16, 10, 16);
    a.cbz(16, false, activation_guard);
    // Layout mode (parent|6): invalidation is represented by layout ID zero.
    // Read the creator's published key-set identity, not a coincidental count
    // of mutations in an unrelated activation.
    a.cbz(15, false, slow);
    a.ldr_w_imm(11, 9, layout.scope_layout as u32);
    a.bind(activation_guard);
    // A layout token is independently non-recycled; only the generation-mode activation
    // needs the exhaustion guard.
    let activation_not_saturated = a.new_label();
    a.cbnz(16, false, activation_not_saturated);
    a.cmn_imm_w(15, 1);
    a.b_cond(C_EQ, slow);
    a.bind(activation_not_saturated);
    a.cmp_reg_w(11, 15);
    a.b_cond(C_NE, slow);
    if env == NameEnv::InX9 {
        // The fixed scope's live parent (a stored handle word: its `Env::as_ptr`).
        if layout.scope_parent_valid
            && layout.scope_parent.is_multiple_of(8)
            && layout.scope_parent / 8 < 4096
        {
            a.ldr_imm(14, 9, layout.scope_parent as u32);
            a.cbz(14, true, slow);
        } else {
            a.b(slow);
        }
    } else {
        a.ldr_imm(14, 19, std::mem::offset_of!(JitCtx, env_parent_raw) as u32);
    }
    a.cbz(14, true, slow);
    a.cbz(16, false, parent_tagged);
    a.sub_imm(10, 10, 4); // strip the optional layout tag
    a.bind(parent_tagged);
    a.sub_imm(16, 10, 2); // strip the depth-1 tag
    a.cmp_reg_x(14, 16);
    a.b_cond(C_NE, slow);
    a.ldr_w_imm(15, 14, sg);
    a.ldrb_imm(17, 14, layout.scope_with as u32);
    a.cmp_imm_w(17, layout.scope_with_none as u32);
    a.b_cond(C_NE, slow);
    a.cmp_reg_w(15, 13);
    a.b_cond(C_NE, slow);
    a.b(scope);
    // --- direct scope mode: `gen` belongs to the current env; falls into the scope tail ---
    a.bind(direct_scope);
    a.cmp_reg_w(11, 13);
    a.b_cond(C_NE, slow);
    // --- scope mode: binding initialized (TDZ) ---
    a.bind(scope);
    a.ldr_imm(14, 12, NAME_IC_OFF_BINDING);
    a.bind(binding_ready);
    a.ldrb_imm(9, 14, bi);
    a.cbz(9, false, slow);
    a.ldrb_imm(9, 14, layout.binding_import as u32);
    a.cbnz(9, false, slow);
    a.add_imm(14, 14, bv); // x14 → the binding's Value
    a.movz(7, 0, 0); // wide scope binding
    a.bind(have);
}

/// Same gate as [`get_prop_inlinable`] plus the dense-element (`Props::elems`) and
/// writable-flag offsets the element templates bake in.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn elem_inlinable(layout: &crate::value::JitLayout) -> bool {
    let elems = layout.obj_props + layout.props_elems;
    get_prop_inlinable(layout)
        && (layout.entry_accessor == layout.entry_value + 8
            || layout.entry_accessor >= layout.entry_value + 16)
        && [
            elems,
            layout.dense_elems + layout.vec_ptr_off,
            layout.dense_elems + layout.vec_len_off,
            layout.dense_mirror + layout.vec_ptr_off,
            layout.dense_mirror + layout.vec_len_off,
        ]
        .into_iter()
        .all(|off| off.is_multiple_of(8) && off / 8 < 4096)
        && layout.entry_writable < 4096
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn packed_elem_inlinable(layout: &crate::value::JitLayout) -> bool {
    layout.packed_elems_valid
        && layout.property_size == 16
        && layout.property_value < 256
        && layout.property_meta < 4096
        && layout.dense_packed.is_multiple_of(8)
        && layout.dense_packed / 8 < 4096
        && layout.dense_inline_len < 4096
        && layout.dense_inline_data < 4096
        && layout.dense_inline_data.is_multiple_of(8)
}

/// Resolve either current packed-element representation. x12 owns the live DenseBuffers;
/// output x15 = Property pointer, x14 = length. A zero inline length means no packed storage.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_packed_elements_base(a: &mut asm::Asm, layout: &crate::value::JitLayout, legacy: usize) {
    let heap = a.new_label();
    let ready = a.new_label();
    a.ldr_imm(15, 12, layout.dense_packed as u32);
    a.cbnz(15, true, heap);
    a.ldrb_imm(14, 12, layout.dense_inline_len as u32);
    a.cbz(14, false, legacy);
    a.add_imm(15, 12, layout.dense_inline_data as u32);
    a.b(ready);
    a.bind(heap);
    a.ldr_imm(14, 15, layout.vec_len_off as u32);
    a.ldr_imm(15, 15, layout.vec_ptr_off as u32);
    a.bind(ready);
}

/// Packed entries can still use the numeric mirror read; the classic entry chase falls back.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn get_elem_inlinable(layout: &crate::value::JitLayout) -> bool {
    let elems = layout.obj_props + layout.props_elems;
    get_prop_inlinable(layout)
        && [
            elems,
            layout.dense_elems + layout.vec_ptr_off,
            layout.dense_elems + layout.vec_len_off,
            layout.dense_mirror + layout.vec_ptr_off,
            layout.dense_mirror + layout.vec_len_off,
        ]
        .into_iter()
        .all(|off| off.is_multiple_of(8) && off / 8 < 4096)
}

/// Inline dense-element read (`a[i]`): an own data element of a plain object/array, indexed
/// through `Props::elems` without hashing or stringifying the key — the machine-code mirror of
/// `Interp::fast_get_elem`. Every guard branches to `slow` before any state is written. Handles a
/// Num key that is exactly a u32 in dense bounds and a non-accessor slot; the live
/// `inline_ic_safe` flag rules out proxies / typed arrays / module namespaces existing at all.
/// Everything else falls to the checked helper.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_get_elem_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: u32,
    l_unwind: usize,
) {
    a.mark_push("get-elem");
    emit_get_elem_inline_marked(a, layout, pc, l_unwind);
    a.mark_pop();
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_get_elem_inline_marked(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: u32,
    l_unwind: usize,
) {
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let mvp = (layout.dense_mirror + layout.vec_ptr_off) as u32;
    let mvl = (layout.dense_mirror + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let es = layout.entry_size as u64;

    let plain = layout.obj_ic_plain as u32;
    let slow = a.new_label();
    let done = a.new_label();
    let typed = a.new_label();
    let mirror_hit = a.new_label();
    // 1. stack: [obj @ -32, key @ -16] — receiver must be Obj
    emit_exec_word_load(a, 9, 20, -16);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    // 2. key must be exactly a u32 (round-trip compare; NaN/negative/fractional/huge all miss).
    // Every non-Number word is a NaN pattern (ECMA-262 PackedValue tags), and an unordered
    // FCMP is "not equal", so this test alone also proves the key is a Number.
    a.ldur_d(0, 20, -8);
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    // 3. x10 = the receiver's payload
    emit_exec_word_load(a, 10, 20, -16);
    emit_exec_payload(a, 10, 10);
    // 4. object base; exotic must be ordinary (None, or a Map/Set) or Array, and plain
    //    (no side-table behavior)
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, typed);
    // 5. mirror read: coherent + hole-free ⇒ bounds + one indexed load of a known Num — no
    // entry chase, no tag check. A miss answers classically below.
    let classic = a.new_label();

    let mf = (layout.obj_props + layout.props_mirror_flags) as u32;
    let mirror = el;
    a.ldrb_imm(12, 11, mf);
    let mask = asm::logical_imm_w((crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32)
        .unwrap();
    a.logic_imm_w(0, 12, 12, mask);
    a.cmp_imm_w(
        12,
        (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32,
    );
    a.b_cond(C_NE, classic);
    a.ldr_imm(12, 11, mirror);
    a.cbz(12, true, classic);
    a.ldr_imm(14, 12, mvl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, classic);
    a.ldr_imm(12, 12, mvp);
    a.ldr_d_lsl3(0, 12, 9);
    a.fmov_x_d(12, 0);
    a.fcmp(0, 0);
    let mirror_number = a.new_label();
    a.b_cond(7, mirror_number);
    a.mov_imm64(12, crate::value::PACK_CANON_NAN);
    a.bind(mirror_number);
    a.b(mirror_hit); // a Num
    a.bind(classic);
    // 5b. dense bounds: n < elems.len (x9's upper bits are zero from the w-form fcvtzu)
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);
    if packed_elem_inlinable(layout) {
        let classic_dense = a.new_label();
        emit_packed_elements_base(a, layout, classic_dense);
        // Packed elements are a keyless Vec<Property>: Empty remains a semantic hole, while a
        // live data slot can be decoded directly without an index string or entry-table chase.
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.add_shifted(15, 15, 9, 4); // property_size == 16 (gate above)
        guard_prop_data(a, 14, 15, layout.property_meta as u32, slow);
        a.ldur(13, 15, layout.property_value as i32);
        a.mov_imm64(14, crate::value::PACK_EMPTY);
        a.cmp_reg_x(13, 14);
        a.b_cond(C_EQ, slow); // an absent own element must still consult the prototype chain
        a.mov(12, 13);
        a.b(mirror_hit);
        a.bind(classic_dense);
    }
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    // 6. slot = elems[n]; NO_SLOT (0xFFFF_FFFF) = hole → slow
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    // 7. entry base = entries data ptr + slot*entry_size
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    // 8. not an accessor
    guard_prop_data(a, 9, 15, ea, slow);
    // Copy the same compact word the heap holds; wide legacy layouts encode once.
    if layout.entry_accessor == layout.entry_value + 8 {
        a.ldur(12, 15, ev);
    } else {
        a.ldur(9, 15, ev);
        a.ldur(13, 15, ev + 8);
        emit_exec_encode_wide(a, 9, 13, 12, 14, 16, 0, slow);
    }
    // --- commit: everything validated; from here only writes ---
    a.bind(mirror_hit);
    // pop obj+key, push value → value lands at the obj slot, sp drops one
    emit_exec_word_store(a, 12, 20, -16);
    a.sub_imm(20, 20, 8);
    a.b(done);
    a.bind(typed);
    typed_array::emit(a, layout, None, mirror_hit, slow);
    a.bind(slow);
    emit_op_helper(a, H_GET_ELEM, pc, l_unwind);
    a.bind(done);
}

/// Inline dense-element write (`a[i] = v`, and the value-keeping `SetElem` when `keep`): the
/// machine-code mirror of `Interp::fast_set_elem` — overwrite an existing own writable data
/// element. `v` *moves* into the slot (under `keep` it also stays on the stack as the expression
/// result). An incoming non-Number (see `emit_packed_stack_encode`), an accessor or
/// non-writable slot, or any dense miss falls to the
/// checked helper.
/// Where a mirror store's key index comes from.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
enum MirrorKey {
    /// Exact u32 already in an x register.
    U32InReg(u32),
    /// The key Value's f64 payload at `[x20 + off]` (already validated as an exact u32).
    StackF64(i32),
    /// A validated u32 key in a d register.
    F64InDreg(u32),
    /// Compile-time constant index.
    Const(u32),
}

/// What a mirror store writes.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
enum MirrorVal {
    /// A Value at `[x20 + off]` (tag at `off`, payload at `off+8`); tag unknown — a non-Num
    /// invalidates the mirror.
    Stack(i32),
    /// A proven-Num f64 in a d register; `bool` = proven exact-i32 (keeps `MIRROR_ALL_I32`).
    Num(u32, bool),
}

/// The element-mirror side of a dense element store the caller has already committed to the
/// entry (see `value::Props::mirror`): keep `mirror[n]` coherent, drop `MIRROR_ALL_I32` for
/// unproven values, and invalidate outright on a non-Num or the hole sentinel. Bounds are
/// re-checked against the mirror's own length as corruption insurance (the lockstep invariant
/// should make it redundant). Clobbers x9, x12, x13 and d1/d2 only.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_mirror_store(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    base: u32,
    key: MirrorKey,
    val: MirrorVal,
) {
    let mf = (layout.obj_props + layout.props_mirror_flags) as u32;
    let mirror = (layout.obj_props + layout.props_elems) as u32;
    let mvp = (layout.dense_mirror + layout.vec_ptr_off) as u32;
    let mvl = (layout.dense_mirror + layout.vec_len_off) as u32;
    let done = a.new_label();
    let inval = a.new_label();
    let inactive = a.new_label();
    a.ldrb_imm(13, base, mf);
    let ok_bit = asm::logical_imm_w(crate::value::MIRROR_OK as u32).unwrap();
    a.logic_imm_w(0, 12, 13, ok_bit);
    a.cbz(12, false, inactive);
    // Value → d1 (or reuse the proven register).
    let (dv, proven_num, proven_i32) = match val {
        MirrorVal::Stack(off) => {
            emit_exec_word_load(a, 9, 20, off);
            emit_exec_number_guard(a, 9, 1, 12, inval);
            (1u32, false, false)
        }
        MirrorVal::Num(d, i32_proven) => (d, true, i32_proven),
    };
    let _ = proven_num;
    if !proven_i32 {
        // MIRROR_ALL_I32 upkeep, flag-first: float-heavy code (flag long cleared) pays two
        // instructions. No hole-sentinel screen: hole accounting is structural (see
        // `Props::mirror_sync`), a data value equal to the sentinel bits is just a NaN to JIT
        // readers, and Rust readers fall back to the authoritative entry.
        let i32_done = a.new_label();
        let i32_bit = asm::logical_imm_w(crate::value::MIRROR_ALL_I32 as u32).unwrap();
        a.logic_imm_w(0, 9, 13, i32_bit);
        a.cbz(9, false, i32_done);
        a.fcvtzs_w_d(9, dv);
        // MirrorVal::Stack owns d1 and a local-element key may still own d0. Test the
        // integer round-trip in d2: clobbering d1 used to truncate the stored mirror value
        // (and lose -0) even though the authoritative property retained the exact Number.
        a.scvtf_d_w(2, 9);
        a.fmov_x_d(9, 2);
        a.fmov_x_d(12, dv);
        a.cmp_reg_x(9, 12);
        a.b_cond(C_EQ, i32_done);
        let clear = asm::logical_imm_w(!(crate::value::MIRROR_ALL_I32 as u32)).unwrap();
        a.logic_imm_w(0, 13, 13, clear);
        a.strb_imm(13, base, mf);
        a.bind(i32_done);
    }
    // Key index → x9.
    match key {
        MirrorKey::U32InReg(r) => {
            if r != 9 {
                a.mov(9, r);
            }
        }
        MirrorKey::StackF64(off) => {
            a.ldur_d(0, 20, off);
            a.fcvtzu_w_d(9, 0);
        }
        MirrorKey::F64InDreg(d) => a.fcvtzu_w_d(9, d),
        MirrorKey::Const(n) => a.mov_imm64(9, n as u64),
    }
    // Insurance bounds check, then the store.
    a.ldr_imm(12, base, mirror);
    a.cbz(12, true, inval);
    a.ldr_imm(13, 12, mvl);
    a.cmp_reg_x(9, 13);
    a.b_cond(C_HS, inval);
    a.ldr_imm(12, 12, mvp);
    a.add_shifted(12, 12, 9, 3);
    a.str_d_imm(dv, 12, 0);
    a.b(done);
    a.bind(inactive);
    // Even a Number-to-Number indexed mutation ends a failed preparation's retry
    // suppression, just as Props::set_index_value does. In particular an earlier
    // fallible allocation may now succeed. Zero flags need no write; an inactive
    // view supplies no live proof bits that must survive this invalidation.
    a.cbz(13, false, done);
    a.bind(inval);
    a.strb_imm(31, base, mf);
    a.bind(done);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_set_elem_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: u32,
    l_unwind: usize,
    keep: bool,
) {
    a.mark_push("set-elem");
    emit_set_elem_inline_marked(a, layout, pc, l_unwind, keep);
    a.mark_pop();
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_set_elem_inline_marked(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: u32,
    l_unwind: usize,
    keep: bool,
) {
    if layout.entry_accessor != layout.entry_value + 8 {
        emit_op_helper(a, H_SET_ELEM, pc, l_unwind);
        return;
    }
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;

    let plain = layout.obj_ic_plain as u32;
    let slow = a.new_label();
    let done = a.new_label();
    let typed = a.new_label();
    // 1. stack: [obj @ -48, key @ -32, v @ -16]
    emit_exec_word_load(a, 9, 20, -24);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    // 2. key must be exactly a u32; the unordered compare of a non-Number (NaN pattern) misses
    a.ldur_d(0, 20, -16);
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    // 3. x10 = the receiver's payload
    emit_exec_word_load(a, 10, 20, -24);
    emit_exec_payload(a, 10, 10);
    // 4. object base; exotic ordinary (None, or a Map/Set) or Array, and plain
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, typed);
    // 5. dense bounds
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);
    emit_packed_element_store(a, layout, slow, done, true, keep);
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    // 6. slot = elems[n]; hole → slow
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    // 7. entry base
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    // 8. data property, writable
    guard_prop_data(a, 9, 15, ea, slow);
    guard_prop_writable(a, 9, 15, ew, slow);
    // --- commit ---
    // Move v into the entry.
    emit_packed_stack_encode(a, -8, slow);
    a.stur(16, 15, ev);
    // Keep the element mirror coherent (x9/x12/x13/d0/d1 are dead here; v words in 14/17,
    // bases in 10/11/15 stay live).
    emit_mirror_store(
        a,
        layout,
        11,
        MirrorKey::StackF64(-16),
        MirrorVal::Stack(-8),
    );
    if keep {
        // [obj, key, v] → [v]: the result lands at the obj slot
        emit_exec_word_load(a, 14, 20, -8);
        emit_exec_word_store(a, 14, 20, -24);
        a.sub_imm(20, 20, 16);
    } else {
        a.sub_imm(20, 20, 24);
    }
    a.b(done);
    a.bind(typed);
    let typed_hit = a.new_label();
    typed_array::emit(
        a,
        layout,
        Some(typed_array::WriteValue::Stack),
        typed_hit,
        slow,
    );
    a.bind(typed_hit);
    if keep {
        emit_exec_word_load(a, 14, 20, -8);
        emit_exec_word_store(a, 14, 20, -24);
        a.sub_imm(20, 20, 16);
    } else {
        a.sub_imm(20, 20, 24);
    }
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, H_SET_ELEM, pc, l_unwind);
    a.bind(done);
}

/// Numeric overwrites of the current keyless dense storage. ECMA-262 OrdinarySetWithOwnDescriptor
/// and Array.[[DefineOwnProperty]] permit an existing writable data element to be overwritten
/// without consulting prototypes or changing length. Holes, accessors, non-writable properties
/// retain the checked path. No guard follows the first write.
/// Entry: x12 = DenseElems, x9 = index, x10 = the receiver's cell address, x11 = Object.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_packed_element_store(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    done: usize,
    stack_receiver: bool,
    keep: bool,
) {
    if !packed_elem_inlinable(layout)
        || layout.entry_value != layout.property_value
        || layout.entry_accessor != layout.property_meta
    {
        return;
    }
    let classic = a.new_label();
    emit_packed_elements_base(a, layout, classic);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    a.add_shifted(15, 15, 9, 4);
    guard_prop_data(a, 14, 15, layout.property_meta as u32, slow);
    guard_prop_writable(a, 14, 15, layout.property_meta as u32, slow);
    emit_packed_existing_guard(a, layout, 15, slow);
    emit_packed_stack_encode(a, -8, slow);
    a.stur(16, 15, layout.property_value as i32);
    // The canonical packed property stays authoritative. Hot numeric arrays can also own an
    // f64 mirror; this numeric write preserves it exactly, including -0/NaN.
    emit_mirror_store(
        a,
        layout,
        11,
        MirrorKey::StackF64(-16),
        MirrorVal::Stack(-8),
    );
    let consumed = if stack_receiver { 24 } else { 16 };
    if keep {
        // The expression result retains the original Number, including signed zero and NaN.
        emit_exec_word_load(a, 14, 20, -8);
        emit_exec_word_store(a, 14, 20, -consumed);
        a.sub_imm(20, 20, (consumed - 8) as u32);
    } else {
        a.sub_imm(20, 20, consumed as u32);
    }
    a.b(done);
    a.bind(classic);
}

/// Inline `lval in rval` (ECMA-262 §13.10.1) when lval is a Number that is exactly an array
/// index and rval a plain object or Array whose own element storage holds that element — the
/// machine-code form of `Interp::plain_own_element_present`. HasProperty then answers *true*
/// with no key string, no prototype walk and nothing observable. Holes (which consult the
/// prototype chain), out-of-bounds indices, and other keys and receivers all take the checked
/// operation, which also produces every *false* answer.
/// Every guard branches to `slow` before any state is written.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_in_inline(a: &mut asm::Asm, layout: &crate::value::JitLayout, pc: u32, l_unwind: usize) {
    emit_in_keyed(a, layout, None, KeySrc::Stack, &[pc], l_unwind, None, None);
}

/// Local-source form used by loop regions for `LoadLocal(key); LoadLocal(receiver); In`.
/// Both local reads remain non-owning on a hit; the checked miss publishes all private
/// numeric homes and replays the three original operations in source order.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_in_local_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    receiver_off: u32,
    key: KeySrc,
    pcs: &[u32],
    l_unwind: usize,
    slow_plan: &regions::Plan,
    slow_reload_fail: usize,
    fast_event: Option<u64>,
) {
    emit_in_keyed(
        a,
        layout,
        Some(receiver_off),
        key,
        pcs,
        l_unwind,
        Some((slow_plan, slow_reload_fail)),
        fast_event,
    );
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_in_keyed(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    receiver_slot: Option<u32>,
    key: KeySrc,
    pcs: &[u32],
    l_unwind: usize,
    slow_plan: Option<(&regions::Plan, usize)>,
    fast_event: Option<u64>,
) {
    #[cfg(not(test))]
    let _ = fast_event;
    let stack_receiver = receiver_slot.is_none();
    debug_assert!(stack_receiver == matches!(key, KeySrc::Stack));
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let plain = layout.obj_ic_plain as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let slow = a.new_label();
    let present = a.new_label();
    let done = a.new_label();
    // 1. The receiver must be an Object; the key must be a Number that is exactly a u32.
    //    NaN, negative, fractional and huge values fail the round trip; 2^32 − 1, which is not
    //    an array index, fails every bounds check below.
    if let Some(slot) = receiver_slot {
        emit_exec_word_load(a, 9, 22, slot as i32);
    } else {
        emit_exec_word_load(a, 9, 20, -8);
    }
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    match key {
        KeySrc::Stack => {
            emit_exec_word_load(a, 9, 20, -16);
            emit_exec_number_guard(a, 9, 0, 11, slow);
            a.ldur_d(0, 20, -16);
        }
        KeySrc::Slot(slot) => {
            emit_exec_word_load(a, 9, 22, slot as i32);
            emit_exec_number_guard(a, 9, 0, 11, slow);
        }
        KeySrc::DReg(reg) => a.fmov_d_d(0, reg),
        KeySrc::SlotPre(..) => unreachable!("In does not use a deferred local update"),
    }
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    // x10 = the receiver's payload. A stack receiver is popped on the commit path; local-source
    // regions do not pop the receiver at all.
    if stack_receiver {
        emit_exec_word_load(a, 10, 20, -8);
        emit_exec_payload(a, 10, 10);
    } else {
        emit_exec_word_load(a, 10, 22, receiver_slot.unwrap() as i32);
        emit_exec_payload(a, 10, 10);
    }
    // 3. object base; exotic ordinary (None, or a Map/Set) or Array, and plain (no
    //    side-table internal methods).
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, slow);
    // 4. element storage sidecar (none: no elements to find here).
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);
    if packed_elem_inlinable(layout) {
        // Packed: a non-Empty Property at n (data or accessor) is a present own element.
        let classic = a.new_label();
        emit_packed_elements_base(a, layout, classic);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.add_shifted(15, 15, 9, 4); // property_size == 16 (gate above)
        a.ldur(13, 15, layout.property_value as i32);
        a.mov_imm64(14, crate::value::PACK_EMPTY);
        a.cmp_reg_x(13, 14);
        a.b_cond(C_EQ, slow);
        a.b(present);
        a.bind(classic);
    }
    // Classic: elems[n] names the element's entry slot, NO_SLOT (0xFFFF_FFFF) a hole.
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    // --- commit: present own element means [[HasProperty]] is true, with no observable
    // prototype walk, key coercion or getter. Local-source form pushes the result without
    // popping an operand. ---
    a.bind(present);
    a.mov_imm64(9, crate::value::PACK_BOOL | 1);
    if stack_receiver {
        emit_exec_word_store(a, 9, 20, -16);
        a.sub_imm(20, 20, 8);
    } else {
        emit_exec_word_store(a, 9, 20, 0);
        a.add_imm(20, 20, 8);
    }
    #[cfg(test)]
    if let Some(function) = fast_event {
        regions::emit_lowering_event(a, function);
    }
    a.b(done);
    a.bind(slow);
    if let Some((plan, _)) = slow_plan {
        plan.flush_locals(a);
    }
    for (index, &pc) in pcs.iter().enumerate() {
        if index + 1 == pcs.len() {
            emit_op_helper(a, H_EXEC, pc, l_unwind);
        } else {
            emit_exec(a, pc, l_unwind);
        }
    }
    if let Some((plan, fail)) = slow_plan {
        plan.reload(a, fail);
    }
    a.bind(done);
}

/// Whether [`emit_define_elem_inline`]'s offsets fit its addressing forms and its packed-element
/// and length-entry assumptions match the probed layout.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn define_elem_inlinable(layout: &crate::value::JitLayout) -> bool {
    let props = layout.obj_props;
    let mirror = layout.dense_mirror;
    crate::value::dense_elements_enabled()
        && elem_inlinable(layout)
        && packed_elem_inlinable(layout)
        // The length entry and the packed elements are both `Property` values.
        && layout.entry_size == layout.property_size
        && layout.entry_value == layout.property_value
        && layout.entry_accessor == layout.property_meta
        && layout.dense_inline_capacity < 256
        && [
            layout.obj_extensible,
            props + layout.props_proto_flag,
            props + layout.props_elem_mode,
            props + layout.props_has_far,
            props + layout.props_mirror_flags,
        ]
        .into_iter()
        .all(|off| off < 4096)
        && slot_memo_fits(layout, props + layout.props_len_slot)
        && fields_data_fits(layout)
        && [
            mirror + layout.vec_ptr_off,
            mirror + layout.vec_len_off,
            mirror + layout.vec_cap_off,
            layout.vec_cap_off,
        ]
        .into_iter()
        .all(|off| off.is_multiple_of(8) && off / 8 < 4096)
}

/// Inline CreateDataPropertyOrThrow(A, k, V) ([`crate::bytecode::AbstractOp`]; ECMA-262 §7.3.7)
/// for the case every element of a self-hosted `map`/`filter` result takes: A is a plain,
/// extensible Array — no prototype, so no proof depends on its shape — whose packed elements
/// end exactly at k with spare capacity and whose `length` is a writable data property.
/// Array.[[DefineOwnProperty]] (§10.4.2.1) then appends { [[Value]]: V, [[Writable]],
/// [[Enumerable]], [[Configurable]]: true } and, at or past `length`, sets it to k + 1: the
/// machine-code form of `Props::try_define_dense_element`'s append, including its numeric
/// mirror. V's word moves into the element. Every guard branches to the checked operation
/// before any write; it also starts, grows and converts element storage.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_define_elem_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    pc: u32,
    l_unwind: usize,
) {
    use crate::value::{MIRROR_ALL_I32, MIRROR_HOLE, MIRROR_OK, MIRROR_PACKED};
    debug_assert!(define_elem_inlinable(layout));
    let props = layout.obj_props as u32;
    let mirror = layout.dense_mirror as u32;
    let (vec_ptr, vec_len, vec_cap) = (
        layout.vec_ptr_off as u32,
        layout.vec_len_off as u32,
        layout.vec_cap_off as u32,
    );
    let slow = a.new_label();
    let done = a.new_label();
    let inline = a.new_label();
    let slot_ready = a.new_label();
    let no_mirror = a.new_label();
    let mirror_ready = a.new_label();
    // 1. stack: [A @ -24, k @ -16, V @ -8]. A an Object; k a Number that is exactly a u32 (x9,
    //    d0); V an ordinary value word (x13) that can move into a property.
    emit_exec_word_load(a, 9, 20, -24);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    emit_exec_word_load(a, 9, 20, -16);
    emit_exec_number_guard(a, 9, 0, 11, slow);
    a.ldur_d(0, 20, -16);
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    emit_exec_word_load(a, 13, 20, -8);
    a.lsr_imm(14, 13, 48);
    // V is an operand word, which is never property-only: only the TDZ marker is rejected.
    a.mov_imm64(15, crate::value::PACK_EMPTY >> 48);
    a.cmp_reg_w(14, 15);
    a.b_cond(C_EQ, slow);
    // 2. x10 = A's payload.
    emit_exec_word_load(a, 10, 20, -24);
    emit_exec_payload(a, 10, 10);
    // 3. An extensible, plain Array (x11 = Object) ...
    a.add_imm(11, 10, layout.obj_from_cell as u32);
    a.ldrb_imm(12, 11, layout.obj_exotic as u32);
    a.cmp_imm_w(12, layout.exotic_array_tag as u32);
    a.b_cond(C_NE, slow);
    a.ldrb_imm(12, 11, layout.obj_ic_plain as u32);
    a.cbz(12, false, slow);
    a.ldrb_imm(12, 11, layout.obj_extensible as u32);
    a.cbz(12, false, slow);
    // ... that is no marked prototype, keeps every index in element storage ...
    a.ldrb_imm(12, 11, props + layout.props_proto_flag as u32);
    a.cbnz(12, false, slow);
    a.ldrb_imm(12, 11, props + layout.props_elem_mode as u32);
    a.cbz(12, false, slow);
    a.ldrb_imm(12, 11, props + layout.props_has_far as u32);
    a.cbnz(12, false, slow);
    // ... and has a memoized, writable data `length` holding a Number (x14 = entry, d1 = L).
    slot_memo_load(a, layout, 12, 11, layout.obj_props + layout.props_len_slot);
    slot_memo_branch_none(a, layout, 12, 15, slow);
    fields_data(a, layout, 14, 11);
    a.mov_imm64(15, layout.entry_size as u64);
    a.madd(14, 12, 15, 14);
    guard_prop_data(a, 15, 14, layout.entry_accessor as u32, slow);
    guard_prop_writable(a, 15, 14, layout.entry_writable as u32, slow);
    a.ldur(15, 14, layout.entry_value as i32);
    emit_exec_number_guard(a, 15, 1, 16, slow);
    // 4. Packed storage ending at k with room for one more (x16 = length, x17 = the new slot;
    //    x15 = the heap Vec header, or null for inline slots).
    a.ldr_imm(12, 11, props + layout.props_elems as u32);
    a.cbz(12, true, slow);
    a.ldr_imm(15, 12, layout.dense_packed as u32);
    a.cbz(15, true, inline);
    a.ldr_imm(16, 15, vec_len);
    a.cmp_reg_x(9, 16);
    a.b_cond(C_NE, slow);
    a.ldr_imm(17, 15, vec_cap);
    a.cmp_reg_x(16, 17);
    a.b_cond(C_HS, slow);
    a.ldr_imm(17, 15, vec_ptr);
    a.add_shifted(17, 17, 16, 4); // property_size == 16 (packed_elem_inlinable)
    a.b(slot_ready);
    a.bind(inline);
    a.ldrb_imm(16, 12, layout.dense_inline_len as u32);
    a.cbz(16, false, slow); // no packed storage yet: the checked operation starts it
    a.cmp_reg_x(9, 16);
    a.b_cond(C_NE, slow);
    a.cmp_imm_w(16, layout.dense_inline_capacity as u32);
    a.b_cond(C_HS, slow);
    a.add_imm(17, 12, layout.dense_inline_data as u32);
    a.add_shifted(17, 17, 16, 4);
    a.bind(slot_ready);
    // 5. The numeric mirror (w0 = flags): inactive, or coherent with the packed elements with
    //    room for V, which must be a Number other than the hole pattern (d2). Anything else is
    //    the checked operation's to invalidate.
    a.ldrb_imm(0, 11, props + layout.props_mirror_flags as u32);
    a.cbz(0, false, no_mirror);
    for bit in [MIRROR_OK, MIRROR_PACKED] {
        a.logic_imm_w(0, 1, 0, asm::logical_imm_w(bit as u32).unwrap());
        a.cbz(1, false, slow);
    }
    emit_exec_number_guard(a, 13, 2, 1, slow);
    a.mov_imm64(1, MIRROR_HOLE);
    a.cmp_reg_x(13, 1);
    a.b_cond(C_EQ, slow);
    a.ldr_imm(2, 12, mirror + vec_len);
    a.cmp_reg_x(2, 16);
    a.b_cond(C_NE, slow);
    a.ldr_imm(3, 12, mirror + vec_cap);
    a.cmp_reg_x(2, 3);
    a.b_cond(C_HS, slow);
    // --- commit: element, mirror, length; nothing below can fail ---
    a.ldr_imm(3, 12, mirror + vec_ptr);
    a.str_d_lsl3(2, 3, 2);
    a.add_imm(2, 2, 1);
    a.str_imm(2, 12, mirror + vec_len);
    // MIRROR_ALL_I32 survives only an exact i32 (bit-identical round trip: not -0, NaN or a
    // fraction).
    a.fcvtzs_w_d(1, 2);
    a.scvtf_d_w(3, 1);
    a.fmov_x_d(1, 3);
    a.cmp_reg_x(1, 13);
    a.b_cond(C_EQ, mirror_ready);
    let keep = asm::logical_imm_w(!(MIRROR_ALL_I32 as u32)).unwrap();
    a.logic_imm_w(0, 0, 0, keep);
    a.strb_imm(0, 11, props + layout.props_mirror_flags as u32);
    a.b(mirror_ready);
    a.bind(no_mirror);
    // --- commit (no mirror) ---
    a.bind(mirror_ready);
    a.stur(13, 17, layout.property_value as i32);
    a.mov_imm64(
        1,
        (crate::value::PROP_WRITABLE
            | crate::value::PROP_ENUMERABLE
            | crate::value::PROP_CONFIGURABLE) as u64,
    );
    a.stur(1, 17, layout.property_meta as i32);
    // The write barrier (site E): the array's cell starts at x10.
    emit_card_mark(a, 10, 9, 17);
    a.add_imm(16, 16, 1);
    let inline_appended = a.new_label();
    let appended = a.new_label();
    a.cbz(15, true, inline_appended);
    a.str_imm(16, 15, vec_len);
    a.b(appended);
    a.bind(inline_appended);
    a.strb_imm(16, 12, layout.dense_inline_len as u32);
    a.bind(appended);
    // k ≥ length: length = k + 1 (exact: k < 2^32 − 1 because it equals a storage length).
    let length_kept = a.new_label();
    a.fcmp(0, 1);
    a.b_cond(C_LO, length_kept);
    a.ucvtf_d_w(1, 16);
    a.stur_d(1, 14, layout.entry_value as i32);
    a.bind(length_kept);
    // [A, k, V] → [true].
    a.mov_imm64(1, crate::value::PACK_BOOL | 1);
    emit_exec_word_store(a, 1, 20, -24);
    a.sub_imm(20, 20, 16);
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, H_ABSTRACT, pc, l_unwind);
    a.bind(done);
}

/// Which fused parameter-slot element op to emit.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy, PartialEq)]
enum ElemLocalKind {
    /// `x[k]` → pops the key, pushes the element (net stack unchanged).
    Get,
    /// `x[k] = v` statement → pops key and value.
    SetDrop,
    /// `x[k] = v` expression → pops key and value, pushes `v` back.
    SetKeep,
}

/// Where a fused element read's key comes from.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy, PartialEq)]
enum KeySrc {
    /// On the operand stack (the plain op forms).
    Stack,
    /// Read straight from a local slot (peephole-fused `LoadLocal k; GetElemLocal x`).
    Slot(u32),
    /// A numeric local already held in one of the region's callee-saved FP homes.
    /// The slow path publishes all homes before re-running the original bytecodes.
    DReg(u32),
    /// Pre-increment/-decrement a numeric local slot in place and use the new value
    /// (peephole-fused `UpdateLocal(k, Pre*); GetElemLocal x`). The slot store is deferred to
    /// the commit point so a slow-path re-run never sees a half-applied update.
    SlotPre(u32, bool),
}

/// Load and guard a Number for a numeric READ. On success x12 holds its bits, which the
/// read/write region transfers to its FP home. Stores need only the separate hole guard.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_packed_number_read_guard(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    entry: u32,
    slow: usize,
) {
    a.ldur(12, entry, layout.entry_value as i32);
    a.lsr_imm(9, 12, 48);
    // PACK_OBJ sorts above the tagged scalar range, so reject it before the two numeric ranges.
    a.movz(14, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    let number = a.new_label();
    a.movz(14, (crate::value::PACK_UNDEFINED >> 48) as u32, 0);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_LO, number);
    a.movz(14, (crate::value::PACK_SYM >> 48) as u32, 0);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_LS, slow);
    a.bind(number);
}

/// A packed dense hole is not an existing own element. Retain its prototype-sensitive slow
/// path when overwriting any other value kind without consulting ownership or payload data.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_packed_existing_guard(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    entry: u32,
    slow: usize,
) {
    a.ldur(12, entry, layout.entry_value as i32);
    a.mov_imm64(14, crate::value::PACK_EMPTY);
    a.cmp_reg_x(12, 14);
    a.b_cond(C_EQ, slow);
}

/// Encode a wide Number at `off` into x16. Other kinds stay on the checked path, keeping the
/// per-site packed-write template compact; Number payload bits are already NaN-box compatible.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_packed_stack_encode(a: &mut asm::Asm, off: i32, slow: usize) {
    emit_exec_word_load(a, 16, 20, off);
    emit_exec_number_guard(a, 16, 1, 13, slow);
}

/// Inline fused element access where the receiver lives in a *parameter* slot
/// ([`crate::bytecode::Op::GetElemLocal`] and friends): like [`emit_get_elem_inline`] /
/// [`emit_set_elem_inline`] but the receiver is read straight out of the slot — it never crosses
/// the operand stack, so no receiver word is copied or dropped at all (the slot holds it, and
/// the collector traces the slot; no user code runs inside the fast path). A non-Obj slot
/// (including a defensive TDZ Empty) falls to the checked helper, which re-runs the op
/// generically.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_elem_local_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slot_off: u32,
    pc: u32,
    l_unwind: usize,
    kind: ElemLocalKind,
) {
    emit_elem_local_keyed(
        a,
        layout,
        slot_off,
        &[pc],
        l_unwind,
        kind,
        KeySrc::Stack,
        None,
        0,
        None,
    );
}

/// [`emit_elem_local_inline`] parameterized on the key source (see [`KeySrc`]) — the peephole
/// pairs fuse the key-producing op into the element read, so their slow path re-runs *both*
/// original ops via the helper (`pcs` lists them in order; every guard runs before any state
/// is written, so the re-run is always clean).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_elem_local_keyed(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slot_off: u32,
    pcs: &[u32],
    l_unwind: usize,
    kind: ElemLocalKind,
    key: KeySrc,
    slow_plan: Option<&regions::Plan>,
    slow_reload_fail: usize,
    fast_event: Option<u64>,
) {
    #[cfg(not(test))]
    let _ = fast_event;
    if kind != ElemLocalKind::Get && layout.entry_accessor != layout.entry_value + 8 {
        for &pc in pcs {
            emit_op_helper(a, H_SET_ELEM, pc, l_unwind);
        }
        return;
    }
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let mvp = (layout.dense_mirror + layout.vec_ptr_off) as u32;
    let mvl = (layout.dense_mirror + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;
    let get = kind == ElemLocalKind::Get;
    debug_assert!(get || key == KeySrc::Stack);
    // Stack-keyed layout: Get → [key @ -16]; Set* → [key @ -32, v @ -16].
    let key_off = if get { -8 } else { -16 };

    let plain = layout.obj_ic_plain as u32;
    let slow = a.new_label();
    let done = a.new_label();
    let typed = a.new_label();
    let mirror_hit = a.new_label();
    // 1. slot holds an Obj; key (from its source) is a Num, loaded into d0
    emit_exec_word_load(a, 9, 22, slot_off as i32);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    match key {
        KeySrc::Stack => {
            emit_exec_word_load(a, 9, 20, key_off);
            emit_exec_number_guard(a, 9, 0, 11, slow);
        }
        KeySrc::Slot(k_off) => {
            emit_exec_word_load(a, 9, 22, k_off as i32);
            emit_exec_number_guard(a, 9, 0, 11, slow);
        }
        KeySrc::DReg(reg) => a.fmov_d_d(0, reg),
        KeySrc::SlotPre(k_off, dec) => {
            emit_exec_word_load(a, 9, 22, k_off as i32);
            emit_exec_number_guard(a, 9, 0, 11, slow);
            a.fmov_one(1);
            a.f_arith(if dec { 1 } else { 0 }, 0, 0, 1); // d0 = slot ± 1 (store deferred)
        }
    }
    // 2. key must be exactly a u32
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);
    // 3. the receiver's cell address straight from the slot (nothing is counted or dropped)
    emit_exec_word_load(a, 10, 22, slot_off as i32);
    emit_exec_payload(a, 10, 10);
    // 4. object base; exotic ordinary (None, or a Map/Set) or Array, and plain
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, typed);

    let classic = a.new_label();
    if get {
        // 5. mirror read: bounds + one indexed load of a known Num (see emit_get_elem_inline).
        let mf = (layout.obj_props + layout.props_mirror_flags) as u32;
        let mirror = el;
        a.ldrb_imm(12, 11, mf);
        let mask =
            asm::logical_imm_w((crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32)
                .unwrap();
        a.logic_imm_w(0, 12, 12, mask);
        a.cmp_imm_w(
            12,
            (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32,
        );
        a.b_cond(C_NE, classic);
        a.ldr_imm(12, 11, mirror);
        a.cbz(12, true, classic);
        a.ldr_imm(14, 12, mvl);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, classic);
        a.ldr_imm(12, 12, mvp);
        a.ldr_d_lsl3(1, 12, 9);
        a.fmov_x_d(12, 1);
        a.fcmp(1, 1);
        let mirror_number = a.new_label();
        a.b_cond(7, mirror_number);
        a.mov_imm64(12, crate::value::PACK_CANON_NAN);
        a.bind(mirror_number);
        a.b(mirror_hit);
    }
    a.bind(classic);
    // 5b. dense bounds
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);
    if !get {
        emit_packed_element_store(a, layout, slow, done, false, kind == ElemLocalKind::SetKeep);
    }
    if get && packed_elem_inlinable(layout) {
        let classic_dense = a.new_label();
        emit_packed_elements_base(a, layout, classic_dense);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.add_shifted(15, 15, 9, 4);
        guard_prop_data(a, 14, 15, layout.property_meta as u32, slow);
        a.ldur(13, 15, layout.property_value as i32);
        a.mov_imm64(14, crate::value::PACK_EMPTY);
        a.cmp_reg_x(13, 14);
        a.b_cond(C_EQ, slow);
        a.mov(12, 13);
        a.b(mirror_hit);
        a.bind(classic_dense);
    }
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    // 6. slot = elems[n]; hole → slow
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    // 7. entry base
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    // 8. data property (+ writable for the set forms)
    guard_prop_data(a, 9, 15, ea, slow);
    if get {
        // Copy directly into a compact word; preserve d0 (the deferred numeric key).
        if layout.entry_accessor == layout.entry_value + 8 {
            a.ldur(12, 15, ev);
        } else {
            a.ldur(9, 15, ev);
            a.ldur(13, 15, ev + 8);
            emit_exec_encode_wide(a, 9, 13, 12, 14, 16, 1, slow);
        }
        a.bind(mirror_hit);
        match key {
            KeySrc::Stack => {
                // pop key, push value → result replaces the key slot
                emit_exec_word_store(a, 12, 20, -8);
            }
            KeySrc::Slot(_) | KeySrc::DReg(_) | KeySrc::SlotPre(..) => {
                if let KeySrc::SlotPre(k_off, _) = key {
                    a.str_d_imm(0, 22, k_off); // commit the deferred ±1 to the slot
                }
                // nothing was on the stack: push the value
                emit_exec_word_store(a, 12, 20, 0);
                a.add_imm(20, 20, 8);
            }
        }
    } else {
        guard_prop_writable(a, 9, 15, ew, slow);
        // --- commit: move v into the entry ---
        emit_packed_stack_encode(a, -8, slow);
        a.stur(16, 15, ev);
        // Element mirror (x9/x12/x13/d1 dead here; the key f64 survives in d0).
        emit_mirror_store(a, layout, 11, MirrorKey::F64InDreg(0), MirrorVal::Stack(-8));
        if kind == ElemLocalKind::SetKeep {
            // The packed store guard proves Number: the result is the stored word.
            emit_exec_word_load(a, 14, 20, -8);
            emit_exec_word_store(a, 14, 20, -16);
            a.sub_imm(20, 20, 8);
        } else {
            a.sub_imm(20, 20, 16);
        }
    }
    #[cfg(test)]
    if let Some(function) = fast_event {
        regions::emit_lowering_event(a, function);
    }
    a.b(done);
    a.bind(typed);
    if get {
        typed_array::emit(a, layout, None, mirror_hit, slow);
    } else {
        let typed_hit = a.new_label();
        typed_array::emit(
            a,
            layout,
            Some(typed_array::WriteValue::Stack),
            typed_hit,
            slow,
        );
        a.bind(typed_hit);
        if kind == ElemLocalKind::SetKeep {
            emit_exec_word_load(a, 14, 20, -8);
            emit_exec_word_store(a, 14, 20, -16);
            a.sub_imm(20, 20, 8);
        } else {
            a.sub_imm(20, 20, 16);
        }
        a.b(done);
    }
    a.bind(slow);
    if let Some(plan) = slow_plan {
        // The checked operation may call user code. Publish every numeric home before
        // replaying the original source operations, and route throws through the ordinary
        // unwinder instead of a private-home flush (the callback may have changed a home).
        plan.flush_locals(a);
    }
    for (index, &p) in pcs.iter().enumerate() {
        if index + 1 == pcs.len() {
            emit_op_helper(a, if get { H_GET_ELEM } else { H_SET_ELEM }, p, l_unwind);
        } else {
            emit_exec(a, p, l_unwind);
        }
    }
    if let Some(plan) = slow_plan {
        // A successful checked operation can re-enter and mutate bindings. Re-guard homes
        // from their canonical post-call slots before execution continues in the region.
        plan.reload(a, slow_reload_fail);
    }
    a.bind(done);
}

/// Guard the one-iteration `HasProperty`/`Get` pair used by self-hosted array methods.
/// On success, x12 contains a cloneable own data value (borrowed from the rooted array);
/// no owner is acquired and no frame state is changed. Every failure happens before the
/// caller runs either TDZ clear, so it can replay both local loads and the original `In`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_local_own_dense_data_guard(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    receiver_off: u32,
    key: KeySrc,
    slow: usize,
) {
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let plain = layout.obj_ic_plain as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let ea = layout.entry_accessor as u32;
    let es = layout.entry_size as u64;

    // In source order the key local is loaded before the receiver local. Merely probing
    // either canonical slot is non-observable; any TDZ value misses to the original PC,
    // which performs the reads and throws in the specified order.
    emit_exec_word_load(a, 9, 22, receiver_off as i32);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, slow);
    match key {
        KeySrc::Slot(offset) => {
            emit_exec_word_load(a, 9, 22, offset as i32);
            emit_exec_number_guard(a, 9, 0, 11, slow);
        }
        KeySrc::DReg(reg) => a.fmov_d_d(0, reg),
        KeySrc::Stack | KeySrc::SlotPre(..) => {
            unreachable!("the TDZ array pair has a local, non-updating key")
        }
    }
    // Only a Number which is exactly a u32 can name the guarded dense slot. This also
    // rejects fractional, negative, NaN, infinite, and out-of-range keys without invoking
    // ToPropertyKey; all of those replay the original operation before either TDZ clear.
    a.fcvtzu_w_d(9, 0);
    a.ucvtf_d_w(1, 9);
    a.fcmp(0, 1);
    a.b_cond(C_NE, slow);

    emit_exec_word_load(a, 10, 22, receiver_off as i32);
    emit_exec_payload(a, 10, 10);
    a.add_imm(11, 10, obj_off);
    a.ldrb_imm(12, 11, ex);
    guard_exotic_ordinary_or_array(a, layout, 12, 13, slow);
    // `ic_plain` excludes proxies and all objects whose property lookup can invoke an
    // exotic internal method. Array's indexed exotic behavior is already represented by
    // its live dense sidecar, while this flag rules out any other side-table behavior.
    a.ldrb_imm(12, 11, plain);
    a.cbz(12, false, slow);
    a.ldr_imm(12, 11, el);
    a.cbz(12, true, slow);

    let classic_dense = a.new_label();
    let entry_ready = a.new_label();
    if packed_elem_inlinable(layout) {
        // Current packed element words place the descriptor flags beside its value.
        // The legacy branch below covers dense buffers without packed element storage.
        emit_packed_elements_base(a, layout, classic_dense);
        a.cmp_reg_x(9, 14);
        a.b_cond(C_HS, slow);
        a.add_shifted(15, 15, 9, 4);
        guard_prop_data(a, 14, 15, layout.property_meta as u32, slow);
        a.ldur(12, 15, layout.property_value as i32);
        a.b(entry_ready);
    }
    a.bind(classic_dense);
    // A classic dense index names an entry slot; NO_SLOT is a hole. Inherited values are
    // deliberately not accepted here: the checked HasProperty path handles them.
    a.ldr_imm(14, 12, evl);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_HS, slow);
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, slow);
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 14, 15, ea, slow);
    if layout.entry_accessor == layout.entry_value + 8 {
        a.ldur(12, 15, ev);
    } else {
        a.ldur(9, 15, ev);
        a.ldur(13, 15, ev + 8);
        emit_exec_encode_wide(a, 9, 13, 12, 14, 16, 1, slow);
    }
    a.bind(entry_ready);
    // PackedValue's Empty word is the deleted-slot marker, not a language value.
    a.mov_imm64(14, crate::value::PACK_EMPTY);
    a.cmp_reg_x(12, 14);
    a.b_cond(C_EQ, slow);
}

/// After the two TDZ stores, reconstruct the already-guarded descriptor address from the
/// still-rooted receiver and unchanged numeric key: the stores run no author code and change
/// no descriptor, so only address derivation repeats. x12 receives the borrowed word.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_local_assumed_dense_data_value(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    receiver_off: u32,
    key: KeySrc,
    slow: usize,
) {
    let obj_off = layout.obj_from_cell as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let ev = layout.entry_value as i32;
    let es = layout.entry_size as u64;

    match key {
        KeySrc::Slot(offset) => {
            emit_exec_word_load(a, 9, 22, offset as i32);
            a.fmov_d_x(0, 9);
        }
        KeySrc::DReg(reg) => a.fmov_d_d(0, reg),
        KeySrc::Stack | KeySrc::SlotPre(..) => {
            unreachable!("the TDZ array pair has a local, non-updating key")
        }
    }
    // These conversions are guard-free because the pre-effect arm proved an exact u32 and
    // no author code/re-entrant operation can modify an unexposed numeric home or slot.
    a.fcvtzu_w_d(9, 0);
    emit_exec_word_load(a, 10, 22, receiver_off as i32);
    emit_exec_payload(a, 10, 10);
    a.add_imm(11, 10, obj_off);
    a.ldr_imm(12, 11, el);

    let classic_dense = a.new_label();
    let entry_ready = a.new_label();
    if packed_elem_inlinable(layout) {
        emit_packed_elements_base(a, layout, classic_dense);
        a.add_shifted(15, 15, 9, 4);
        a.ldur(12, 15, layout.property_value as i32);
        a.b(entry_ready);
    }
    a.bind(classic_dense);
    a.ldr_imm(12, 12, evp);
    a.add_shifted(12, 12, 9, 2);
    a.ldr_w_imm(13, 12, 0);
    fields_data(a, layout, 15, 11);
    a.mov_imm64(16, es);
    a.madd(15, 13, 16, 15);
    if layout.entry_accessor == layout.entry_value + 8 {
        a.ldur(12, 15, ev);
    } else {
        a.ldur(9, 15, ev);
        a.ldur(13, 15, ev + 8);
        emit_exec_encode_wide(a, 9, 13, 12, 14, 16, 1, slow);
    }
    a.bind(entry_ready);
}

/// One op of a numeric register chain (see [`build_chain`]). Every value the chain produces is a
/// proven Num held in a callee-saved FP register (d8..d15) instead of the operand stack.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy)]
enum ChainOp {
    /// Push a Num constant (f64 bits).
    ConstNum(u64),
    /// Push a numeric local (slot byte offset).
    Load(u32),
    /// `++`/`--` a numeric local in place (slot byte offset); pushes per the kind.
    Update(u32, UpdKind),
    /// Dense element read: virtual key → virtual Num element (receiver slot byte offset).
    GetElem(u32),
    /// Dense element write from virtual `[key, v]` (receiver slot byte offset); `true` = keep
    /// `v` as the virtual result (`SetElemLocal` vs `SetElemLocalDrop`).
    SetElem(u32, bool),
    /// fadd/fsub/fmul/fdiv on the two virtual tops (same encoding as [`asm::Asm::f_arith`]).
    Arith(u32),
    /// Int32 op on the two virtual tops: 0=and 1=or 2=xor 3=shl 4=ushr 5=shr. Operands convert
    /// via guarded ToInt32 (guard-free when the virtual is known int-valued); the result is a
    /// known int-valued Num.
    Bit(u32),
    Neg,
    /// Store the virtual top into a local slot (byte offset).
    Store(u32),
    Pop,
    /// Duplicate the virtual top (compound element assignment's key copy).
    Dup,
    /// A proven no-op: numeric ToPropKeyLocal, or an unused read guarded by the loop preamble.
    KeyNop,
    /// Cached free-name read that must currently hold a Num (the `NameIc` cell address).
    LoadName(usize),
    /// Monomorphic shape-validated property read whose live value must be a Num. `u32::MAX`
    /// denotes the frame's `this` binding; every other value is a local-slot byte offset.
    /// The IC state is baked only as a guarded lookup recipe: live shapes, bounds, attributes,
    /// and the value tag are still checked on every execution.
    LoadProp(u32, crate::bytecode::IcState),
    /// Own writable numeric property store. Receiver encoding matches `LoadProp`; the virtual
    /// top is consumed and written without materializing a wide stack Value.
    StoreProp(u32, crate::bytecode::IcState),
    /// Terminal fused compare+branch: negated ARM condition + target pc.
    CmpBranch(u32, usize),
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
struct LinkedScanPlan {
    exit_pc: usize,
    next_off: u32,
    peek_off: u32,
    link: crate::bytecode::IcState,
    loose_null_compare: bool,
}

/// Plan `while ((peek = next.link) != null) next = peek`.  The optimized region borrows linked
/// objects through the still-rooted initial list, does no per-node work on handles, then
/// materializes the two locals exactly once at the exit or a guarded side exit.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn plan_linked_scan(
    chunk: &Chunk,
    ops: &[crate::bytecode::Op],
    head: usize,
    cfg: &crate::jit_ir::Cfg,
    layout: &crate::value::JitLayout,
    fast: u32,
) -> Option<LinkedScanPlan> {
    use crate::bytecode::Op;
    if fast & (1 << 21) == 0
        || env_flag!("TRUST_JS_JIT_NO_CFG_REGION")
        || !get_prop_inlinable(layout)
        || layout.entry_accessor != layout.entry_value + 8
    {
        return None;
    }
    let end = head.checked_add(9)?;
    let [Op::GetPropLocal(next, _, cache), Op::Dup, Op::StoreLocal(peek), Op::Const(null), cmp @ (Op::NotEq | Op::StrictNotEq), Op::JumpIfFalse(exit), Op::LoadLocal(peek_read), Op::StoreLocal(next_store), Op::Jump(back)] =
        ops.get(head..end)?
    else {
        return None;
    };
    let loose_null_compare = matches!(cmp, Op::NotEq);
    if next != next_store
        || peek != peek_read
        || next == peek
        || *back as usize != head
        || (*exit as usize) < end
        || !chunk.jit_const_copyable(*null)
        || chunk.jit_const_bits(*null) != (2, 0)
    {
        return None;
    }
    let lp = cfg.loop_at_header(head)?;
    if lp.latches.len() != 1
        || cfg.blocks()[lp.latches[0].0 as usize].end != end
        || lp.blocks.iter().any(|id| {
            let block = &cfg.blocks()[id.0 as usize];
            block.start < head
                || block.end > end
                || block.stack_in != Some(0)
                || block.stack_out != Some(0)
        })
        || crate::jit_ir::RegionIr::build_loop(chunk, cfg, head).is_err()
    {
        return None;
    }
    let link = chunk.jit_cache_preferred(*cache)?;
    if link.depth != 0 {
        return None;
    }
    let next_off = *next as u32 * 8;
    let peek_off = *peek as u32 * 8;
    if next_off + 8 >= 4096 || peek_off + 8 >= 4096 {
        return None;
    }
    Some(LinkedScanPlan {
        exit_pc: *exit as usize,
        next_off,
        peek_off,
        link,
        loose_null_compare,
    })
}

/// Try to recognize a *numeric register chain* starting at `start`: a maximal run of ops whose
/// intermediate values can live entirely in FP registers — locals, dense elements, float
/// arithmetic, cached names — ending either naturally or in a fused compare+branch. Every op
/// consumes only values produced *within* the chain (tracked by `vdepth`), so each value is a
/// proven Num in a register: arithmetic needs no tag checks at all and the compare+branch needs
/// no guards whatsoever. Returns the chain and how many bytecode ops it covers (`None` if
/// shorter than 3 ops — plain templates are fine for those).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn build_chain(
    chunk: &Chunk,
    ops: &[crate::bytecode::Op],
    start: usize,
    targeted: &[bool],
    layout: &crate::value::JitLayout,
    fast: u32,
) -> Option<(Vec<(ChainOp, usize)>, usize)> {
    use crate::bytecode::Op;
    let in_range = |s: u16| (s as u32) * 8 + 8 < 4096;
    let elem_ok = fast & 1024 != 0 && get_elem_inlinable(layout);
    let name_ok = fast & 8192 != 0 && load_name_inlinable(layout);
    let prop_ok =
        fast & 256 != 0 && get_prop_inlinable(layout) && !env_flag!("TRUST_JS_JIT_NO_PROP_CHAIN");
    let prop_store_ok = prop_ok
        && fast & 65536 != 0
        && set_prop_inlinable(layout)
        && !env_flag!("TRUST_JS_JIT_NO_PROP_STORE_CHAIN");
    let mut chain: Vec<(ChainOp, usize)> = Vec::new();
    let mut vdepth = 0usize;
    let mut pc = start;
    while pc < ops.len() {
        if pc > start && targeted[pc] {
            break; // a jump lands here: the canonical (memory) stack state must hold
        }
        if local_read_discard_pair(ops, pc, targeted).is_some() {
            // The observable effect of this pair is only its TDZ check. Its value
            // is not a numeric input: leave it to the tag-only ordinary template,
            // and keep any following stores out of a speculative helper suffix.
            break;
        }
        let (op, push, pop): (ChainOp, usize, usize) = match &ops[pc] {
            Op::Const(k) => match chunk.jit_const_num(*k) {
                Some(bits) => (ChainOp::ConstNum(bits), 1, 0),
                None => break,
            },
            Op::LoadLocal(s) if in_range(*s) => (ChainOp::Load(*s as u32 * 8), 1, 0),
            Op::UpdateLocal(s, kind) if in_range(*s) => {
                let pushes = !matches!(kind, UpdKind::IncDiscard | UpdKind::DecDiscard);
                (ChainOp::Update(*s as u32 * 8, *kind), pushes as usize, 0)
            }
            Op::GetElemLocal(x) if elem_ok && in_range(*x) && vdepth >= 1 => {
                (ChainOp::GetElem(*x as u32 * 8), 1, 1)
            }
            Op::SetElemLocal(x) if elem_ok && in_range(*x) && vdepth >= 2 => {
                (ChainOp::SetElem(*x as u32 * 8, true), 1, 2)
            }
            Op::SetElemLocalDrop(x) if elem_ok && in_range(*x) && vdepth >= 2 => {
                (ChainOp::SetElem(*x as u32 * 8, false), 0, 2)
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div if vdepth >= 2 => {
                let f = match ops[pc] {
                    Op::Add => 0,
                    Op::Sub => 1,
                    Op::Mul => 2,
                    _ => 3,
                };
                (ChainOp::Arith(f), 1, 2)
            }
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr if vdepth >= 2 => {
                let code = match ops[pc] {
                    Op::BitAnd => 0,
                    Op::BitOr => 1,
                    Op::BitXor => 2,
                    Op::Shl => 3,
                    Op::UShr => 4,
                    _ => 5, // Shr
                };
                (ChainOp::Bit(code), 1, 2)
            }
            Op::Neg if vdepth >= 1 => (ChainOp::Neg, 1, 1),
            Op::StoreLocal(s) if in_range(*s) => {
                if vdepth >= 1 {
                    (ChainOp::Store(*s as u32 * 8), 0, 1)
                } else {
                    break;
                }
            }
            Op::Pop if vdepth >= 1 => (ChainOp::Pop, 0, 1),
            Op::Dup if vdepth >= 1 => (ChainOp::Dup, 1, 0),
            Op::ToPropKeyLocal(_) if vdepth >= 1 => (ChainOp::KeyNop, 0, 0),
            Op::LoadName(_, c) if name_ok => {
                (ChainOp::LoadName(chunk.jit_name_cache_ptr(*c)), 1, 0)
            }
            Op::GetPropThis(_, c) if prop_ok => {
                let Some(st) = chunk.jit_cache_preferred(*c) else {
                    break;
                };
                if st.depth > 2 || (st.depth == 2 && st.mid_ok & 1 == 0) {
                    break;
                }
                (ChainOp::LoadProp(u32::MAX, st), 1, 0)
            }
            Op::GetPropLocal(s, _, c) if prop_ok && in_range(*s) => {
                let Some(st) = chunk.jit_cache_preferred(*c) else {
                    break;
                };
                if st.depth > 2 || (st.depth == 2 && st.mid_ok & 1 == 0) {
                    break;
                }
                (ChainOp::LoadProp(*s as u32 * 8, st), 1, 0)
            }
            Op::SetPropThisDrop(n, c)
                if prop_store_ok
                    && vdepth >= 1
                    && !chunk
                        .jit_name(*n)
                        .as_bytes()
                        .first()
                        .is_some_and(|b| b.is_ascii_digit()) =>
            {
                let Some(st) = chunk.jit_cache_preferred(*c) else {
                    break;
                };
                if st.depth != 0 {
                    break;
                }
                (ChainOp::StoreProp(u32::MAX, st), 0, 1)
            }
            Op::SetPropLocalDrop(s, n, c)
                if prop_store_ok
                    && vdepth >= 1
                    && in_range(*s)
                    && !chunk
                        .jit_name(*n)
                        .as_bytes()
                        .first()
                        .is_some_and(|b| b.is_ascii_digit()) =>
            {
                let Some(st) = chunk.jit_cache_preferred(*c) else {
                    break;
                };
                if st.depth != 0 {
                    break;
                }
                (ChainOp::StoreProp(*s as u32 * 8, st), 0, 1)
            }
            Op::Lt
            | Op::Gt
            | Op::Le
            | Op::Ge
            | Op::StrictEq
            | Op::StrictNotEq
            | Op::EqEq
            | Op::NotEq
                if vdepth == 2 =>
            {
                match ops.get(pc + 1) {
                    Some(Op::JumpIfFalse(t)) if !targeted[pc + 1] => {
                        let neg = match ops[pc] {
                            Op::Lt => 5,                  // PL (unordered jumps)
                            Op::Gt => 13,                 // LE
                            Op::Le => 8,                  // HI
                            Op::Ge => 11,                 // LT
                            Op::StrictEq | Op::EqEq => 1, // NE
                            _ => 0,                       // EQ
                        };
                        chain.push((ChainOp::CmpBranch(neg, *t as usize), pc));
                    }
                    _ => {}
                }
                break;
            }
            _ => break,
        };
        if vdepth - pop + push > 8 {
            break; // out of d-registers
        }
        vdepth = vdepth - pop + push;
        chain.push((op, pc));
        pc += 1;
    }
    // Trim trailing pure producers: a Load/Const/LoadName whose value nothing in the chain
    // consumes would only be spilled back to the stack — zero benefit, and for an *object*
    // local (an array receiver feeding a non-chain GetElem/SetElem) the Num guard would fail
    // every execution, sending the whole bail tail through the generic helper. Emitting them
    // as plain templates instead is both faster and type-agnostic.
    while matches!(
        chain.last(),
        Some((
            ChainOp::ConstNum(_) | ChainOp::Load(_) | ChainOp::LoadName(_) | ChainOp::LoadProp(..),
            _
        ))
    ) {
        chain.pop();
    }
    // Same idea anywhere in the chain: a pure producer whose value nothing in the chain consumes
    // (a call argument, an array receiver below the real work — `x.am(i, a[i], r, 2*i, 0, 1)`)
    // would only be spilled — and when the value is an object, its Num guard fails every single
    // execution, condemning the whole tail to the generic helper. Cut the chain just before the
    // earliest such producer; the main loop emits it as a plain template and re-attempts a chain
    // right after it. Iterate: each cut can orphan earlier consumers.
    loop {
        let mut sim: Vec<usize> = Vec::new();
        for (idx, &(op, _)) in chain.iter().enumerate() {
            let (pops, pushes): (usize, usize) = match op {
                ChainOp::ConstNum(_)
                | ChainOp::Load(_)
                | ChainOp::LoadName(_)
                | ChainOp::LoadProp(..) => (0, 1),
                ChainOp::StoreProp(..) => (1, 0),
                ChainOp::Update(_, k) => (
                    0,
                    !matches!(k, UpdKind::IncDiscard | UpdKind::DecDiscard) as usize,
                ),
                ChainOp::GetElem(_) => (1, 1),
                ChainOp::SetElem(_, keep) => (2, keep as usize),
                ChainOp::Arith(_) | ChainOp::Bit(_) => (2, 1),
                ChainOp::Neg => (1, 1),
                ChainOp::Store(_) | ChainOp::Pop => (1, 0),
                ChainOp::Dup => (0, 1),
                ChainOp::KeyNop => (0, 0),
                ChainOp::CmpBranch(..) => (2, 0),
            };
            for _ in 0..pops {
                sim.pop();
            }
            for _ in 0..pushes {
                sim.push(idx);
            }
        }
        let cut = sim
            .iter()
            .copied()
            .filter(|&idx| {
                matches!(
                    chain[idx].0,
                    ChainOp::ConstNum(_)
                        | ChainOp::Load(_)
                        | ChainOp::LoadName(_)
                        | ChainOp::LoadProp(..)
                )
            })
            .min();
        match cut {
            Some(idx) => chain.truncate(idx),
            None => break,
        }
        if chain.is_empty() {
            return None;
        }
    }
    if chain.len() < 3 {
        return None;
    }
    // Element transfers do not benefit from assuming the value is numeric. A common
    // table decoder does `result = values[index]`: GetElem/Dup/Store/Pop then fail
    // on every string result and replay through generic helpers, despite the ordinary
    // element/local templates already handling every value type in machine code.
    // Keep numerical speculation for computations, not just moving array values.
    let has_elem = chain
        .iter()
        .any(|(op, _)| matches!(op, ChainOp::GetElem(_) | ChainOp::SetElem(..)));
    let has_computation = chain.iter().any(|(op, _)| {
        matches!(
            op,
            ChainOp::Arith(_)
                | ChainOp::Bit(_)
                | ChainOp::Neg
                | ChainOp::Update(..)
                | ChainOp::CmpBranch(..)
        )
    });
    if has_elem && !has_computation {
        return None;
    }
    // A speculative property producer is worthwhile when the chain actually computes with it.
    // Property-to-local transfer runs are common in generated code (EarleyBoyer in particular),
    // but they merely replace one compact property template with a larger guarded chain and can
    // repeatedly bail when the field is non-numeric. Leave those to the ordinary templates.
    let has_prop = chain
        .iter()
        .any(|(op, _)| matches!(op, ChainOp::LoadProp(..)));
    if has_prop {
        let useful = match chain.last() {
            Some((ChainOp::Bit(_), _)) => true,
            Some((ChainOp::StoreProp(..), _)) => {
                chain.iter().any(|(op, _)| matches!(op, ChainOp::Bit(_)))
            }
            Some((ChainOp::CmpBranch(..), cmp_pc)) => match ops[*cmp_pc] {
                // Ordered comparisons necessarily request numeric coercion on the ordinary
                // path, so a numeric property guard is a productive speculation.
                crate::bytecode::Op::Lt
                | crate::bytecode::Op::Gt
                | crate::bytecode::Op::Le
                | crate::bytecode::Op::Ge => true,
                // Equality is frequently object/string identity in generated programs. Admit
                // only the global-constant form seen in numeric dispatch kernels; local/local
                // and property/property equality stay on the type-generic template.
                crate::bytecode::Op::StrictEq
                | crate::bytecode::Op::StrictNotEq
                | crate::bytecode::Op::EqEq
                | crate::bytecode::Op::NotEq => chain
                    .iter()
                    .any(|(op, _)| matches!(op, ChainOp::LoadName(_))),
                _ => false,
            },
            _ => false,
        };
        if !useful {
            return None;
        }
    }
    if env_flag!("TRUST_JS_JIT_PROP_CHAIN_LOG") && has_prop {
        let desc: Vec<String> = chain
            .iter()
            .map(|(_op, pc)| format!("{pc}:{:?}", ops[*pc]))
            .collect();
        eprintln!("[jit-prop-chain] {}", desc.join(" | "));
    }
    let consumed = chain.last().map_or(0, |&(op, p)| {
        p - start
            + if matches!(op, ChainOp::CmpBranch(..)) {
                2
            } else {
                1
            }
    });
    Some((chain, consumed))
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod numeric_chain_tests {
    use super::{build_chain, ChainOp, Chunk};

    fn compile_function(source: &str) -> std::rc::Rc<Chunk> {
        let statements = crate::parser::parse_script(source, false)
            .ok()
            .expect("parse");
        let function = statements
            .iter()
            .find_map(|statement| match statement {
                crate::ast::Stmt::FuncDecl(function) => Some(function.clone()),
                _ => None,
            })
            .expect("function declaration");
        crate::bytecode::compile(&function).expect("compile")
    }

    #[test]
    fn local_store_pairs_keep_jump_entries_and_distinct_slots() {
        use crate::bytecode::Op;
        for pair in [
            [Op::Dup, Op::StoreLocal(3)],
            [Op::StoreLocal(3), Op::LoadLocal(3)],
        ] {
            assert_eq!(super::local_store_pair(&pair, 0, &[false; 3]), Some(3));
            assert_eq!(
                super::local_store_pair(&pair, 0, &[false, true, false]),
                None
            );
            assert_eq!(super::local_store_pair(&pair, 1, &[false; 3]), None);
        }
        assert_eq!(
            super::local_store_pair(&[Op::StoreLocal(3), Op::LoadLocal(4)], 0, &[false; 3]),
            None
        );
        assert_eq!(
            super::local_store_pair(&[Op::Dup, Op::StoreLocal(511)], 0, &[false; 3]),
            None
        );
    }

    #[test]
    fn unused_local_reads_keep_tdz_checks_and_independent_pop_entries() {
        use crate::bytecode::Op;
        let pair = [Op::LoadLocal(3), Op::Pop];
        assert_eq!(
            super::local_read_discard_pair(&pair, 0, &[false; 3]),
            Some(3)
        );
        assert_eq!(
            super::local_read_discard_pair(&pair, 0, &[false, true, false]),
            None
        );
        assert_eq!(super::local_read_discard_pair(&pair, 1, &[false; 3]), None);
        assert_eq!(
            super::local_read_discard_pair(&[Op::LoadLocal(511), Op::Pop], 0, &[false; 3]),
            None
        );
        assert_eq!(
            super::local_read_discard_pair(&[Op::LoadLocal(3), Op::Dup], 0, &[false; 3]),
            None
        );
    }

    #[test]
    fn hot_loop_padding_preserves_instruction_words_and_branch_labels() {
        for alignment in [16, 32, 64, 128] {
            for prefix in 0..32 {
                let mut assembler = super::asm::Asm::new();
                for _ in 0..prefix {
                    assembler.movz(0, 7, 0);
                }
                let body = assembler.new_label();
                assembler.b(body);
                let before_padding = assembler.buf.len();
                assembler.align_hot(alignment);
                assembler.bind(body);
                assembler.ret();
                let (words, offsets) = assembler.finish_with_offsets(&[body]);
                let body_word = offsets[0] as usize / 4;
                assert!((offsets[0] as usize).is_multiple_of(alignment));
                assert!(words[before_padding..body_word]
                    .iter()
                    .all(|word| *word == 0xD503_201F));
                assert_eq!(words[prefix] & 0x03ff_ffff, (body_word - prefix) as u32);
                assert_eq!(words[body_word], 0xD65F_03C0);
            }
        }
    }

    #[test]
    fn loop_fallback_entries_leave_non_entry_discard_pairs_fusible() {
        use crate::bytecode::Op;
        let engine = crate::Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let chunk = compile_function(
            "function read(a, count) { let result = ''; for (let k = 0; k < count; k++) result = a[k & 127]; return result; }",
        );
        let ops = chunk.jit_ops();
        let mut targeted = vec![false; ops.len() + 1];
        let mut head = None;
        for (pc, op) in ops.iter().enumerate() {
            match op {
                Op::Jump(to) | Op::JumpIfFalse(to) => {
                    targeted[*to as usize] = true;
                    if (*to as usize) < pc {
                        head = Some(*to as usize);
                    }
                }
                _ => {}
            }
        }
        let cfg = crate::jit_ir::Cfg::build(&chunk).expect("CFG");
        let plan = super::plan_loop(
            &chunk,
            ops,
            head.expect("loop head"),
            &targeted,
            &layout,
            u32::MAX,
            &cfg,
        )
        .expect("numeric loop plan");
        let mut assembler = super::asm::Asm::new();
        let labels: Vec<_> = (0..=ops.len()).map(|_| assembler.new_label()).collect();
        let mut poll_engine = crate::Engine::new();
        let ilayout = crate::interpreter::interp_layout(&mut poll_engine.interp);
        super::emit_loop_chain(
            &mut assembler,
            &layout,
            &ilayout,
            &plan,
            &labels,
            &mut targeted,
        );
        let guard_pc = ops
            .windows(2)
            .position(|pair| matches!(pair, [Op::LoadLocal(_), Op::Pop]))
            .expect("lexical assignment guard");
        let Op::LoadLocal(guard_slot) = ops[guard_pc] else {
            unreachable!()
        };
        assert!(plan
            .initialization_guards
            .contains(&(guard_slot as u32 * 8)));
        assert!(plan
            .chain
            .iter()
            .any(|(op, pc)| *pc == guard_pc && matches!(op, ChainOp::KeyNop)));
        assert!(plan
            .slots
            .iter()
            .any(|slot| slot.off == guard_slot as u32 * 8 && slot.virgin && !slot.preload));
        assert!(!targeted[guard_pc + 1], "Pop has no bailout edge");
        assert!(super::local_read_discard_pair(ops, guard_pc, &targeted).is_some());
        let element_pc = ops
            .iter()
            .enumerate()
            .find_map(|(pc, op)| {
                (pc > plan.head && matches!(op, Op::GetElemLocal(_))).then_some(pc)
            })
            .expect("element access");
        assert!(
            targeted[element_pc],
            "element guard must retain its exact entry"
        );
    }

    #[test]
    fn discarded_local_reads_do_not_introduce_numeric_speculation() {
        let engine = crate::Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let chunk = compile_function(
            "function read(a, k) { let result = ''; result = a[k & 127]; return result; }",
        );
        let ops = chunk.jit_ops();
        let targeted = vec![false; ops.len() + 1];
        let guard_pc = ops
            .windows(2)
            .position(|pair| {
                matches!(
                    pair,
                    [crate::bytecode::Op::LoadLocal(_), crate::bytecode::Op::Pop]
                )
            })
            .expect("lexical assignment guard");
        for start in 0..=guard_pc {
            if let Some((chain, _)) = build_chain(&chunk, ops, start, &targeted, &layout, u32::MAX)
            {
                assert!(
                    chain.iter().all(|(_, pc)| *pc != guard_pc),
                    "unused read at {guard_pc} became a numeric guard from {start}: {ops:?}"
                );
            }
        }
    }

    #[test]
    fn element_value_transfers_do_not_speculate_on_the_value_type() {
        let engine = crate::Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let chunk =
            compile_function("function read(a, k, result) { return result = a[k], result; }");
        let ops = chunk.jit_ops();
        assert!(ops
            .iter()
            .any(|op| matches!(op, crate::bytecode::Op::GetElemLocal(_))));
        let targeted = vec![false; ops.len() + 1];
        for start in 0..ops.len() {
            if let Some((chain, _)) = build_chain(&chunk, ops, start, &targeted, &layout, u32::MAX)
            {
                assert!(
                    !chain
                        .iter()
                        .any(|(op, _)| matches!(op, ChainOp::GetElem(_))),
                    "pure element transfer became a numeric chain at {start}: {ops:?}"
                );
            }
        }
    }

    #[test]
    fn element_arithmetic_still_uses_numeric_chains() {
        let engine = crate::Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        for source in [
            "function sum(a, k) { return a[k] + 7; }",
            "function bits(a, k) { return a[k] ^ 255; }",
            "function negative(a, k) { return -a[k]; }",
            "function less(a, k) { if (a[k] < 7) return 1; return 0; }",
        ] {
            let chunk = compile_function(source);
            let ops = chunk.jit_ops();
            let targeted = vec![false; ops.len() + 1];
            assert!(
                (0..ops.len()).any(|start| {
                    build_chain(&chunk, ops, start, &targeted, &layout, u32::MAX).is_some_and(
                        |(chain, _)| {
                            chain
                                .iter()
                                .any(|(op, _)| matches!(op, ChainOp::GetElem(_)))
                        },
                    )
                }),
                "element computation lost its numeric chain: {source}: {ops:?}"
            );
        }
    }
}

/// Emit a numeric register chain (see [`build_chain`]): the virtual operand stack lives in
/// d8..d15 (callee-saved — the prologue preserves them), scratch math uses d0..d3. Any guard
/// failure spills the virtual values to the real operand stack — in stack order, exactly the
/// state the ops would have produced — and re-runs the failing op and everything after it
/// through the generic helper, so semantics are identical on every path. Side-effecting ops
/// (slot stores, element writes) commit only after all their guards pass, which is what makes
/// the spill-and-rerun always clean.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_chain(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    chain: &[(ChainOp, usize)],
    pc_labels: &[usize],
    l_unwind: usize,
) {
    a.mark_push("chain");
    emit_chain_marked(a, layout, chain, pc_labels, l_unwind);
    a.mark_pop();
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_chain_marked(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    chain: &[(ChainOp, usize)],
    pc_labels: &[usize],
    l_unwind: usize,
) {
    let mf = (layout.obj_props + layout.props_mirror_flags) as u32;
    let mirror = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let mvp = (layout.dense_mirror + layout.vec_ptr_off) as u32;
    let mvl = (layout.dense_mirror + layout.vec_len_off) as u32;
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let pr = layout.obj_proto as u32;
    let sh = (layout.obj_props + layout.props_shape) as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let ev = layout.entry_value as i32;
    let num_ev = if layout.entry_accessor == layout.entry_value + 8 {
        ev
    } else {
        ev + 8
    };
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;
    let plain = layout.obj_ic_plain as u32;

    let done = a.new_label();
    // Virtual stack: (d-register, known-int-valued). Int-valued means the f64 is integral and in
    // i64 range, so a ToInt32 conversion is a bare fcvtzs with no round-trip guard.
    let mut vregs: Vec<(u32, bool)> = Vec::new();
    let mut free: Vec<u32> = vec![15, 14, 13, 12, 11, 10, 9, 8];
    // Receiver cache: slot byte offset → registers holding validated receiver state. The chain
    // fast path calls no helpers, so between element ops nothing can change the slot's tag, the
    // object's exotic status, the ic-safe flag — or, in Mirror mode, the mirror's coherence,
    // length, or data pointer (the in-chain slim store only overwrites payloads; growth and
    // hole-creation bail). Mirror mode pins the whole element fast path in registers: `base` +
    // the mirror data pointer and length, validated MIRROR_OK|NO_HOLES once — so later reads
    // are a bounds check + one indexed load, the shape one dispatch loop hits 5-6 times per
    // iteration on the same one or two arrays (NavierStokes' lin_solve). Classic mode caches
    // only the base under register pressure. A null cached mirror pointer selects the native
    // packed/classic path: an unavailable mirror must not replay a whole chain in helpers.
    // Cache registers live in x2-x8 except x7: `emit_name_ic_value_ptr` (in-chain LoadName)
    // clobbers x7 (the packed-value flag) and x9-x17. An in-chain Store/Update to the receiver
    // slot drops its entry.
    enum RcMode {
        Classic,
        Mirror { mpreg: u32, mlreg: u32 },
    }
    struct RcEnt {
        off: u32,
        base: u32,
        mode: RcMode,
    }
    let mut rcache: Vec<RcEnt> = Vec::new();
    let mut rfree: Vec<u32> = vec![8, 6, 5, 4, 3, 2];
    // (chain index, bail label, virtual stack *before* the op) — slow paths follow the fast body.
    let mut bails: Vec<(usize, usize, Vec<(u32, bool)>)> = Vec::new();

    for (idx, (cop, _pc)) in chain.iter().enumerate() {
        // One bail label per chain op. The snapshot is the virtual stack before the op runs: the
        // emitter pops from `vregs` up front, but every guard fires before the op writes any
        // register or memory, so the snapshot registers still hold the pre-op values at any bail.
        let bail = a.new_label();
        let pre_op: Vec<(u32, bool)> = vregs.clone();
        let mut used = 0u32;
        macro_rules! guard {
            () => {{
                used += 1;
                bail
            }};
        }
        match *cop {
            ChainOp::ConstNum(bits) => {
                let rd = free.pop().expect("chain reg underflow");
                a.mov_imm64(9, bits);
                a.fmov_d_x(rd, 9);
                let f = f64::from_bits(bits);
                let iv =
                    f.fract() == 0.0 && (-9.223372036854776e18..9.223372036854776e18).contains(&f);
                vregs.push((rd, iv));
            }
            ChainOp::Load(off) => {
                a.ldr_imm(9, 22, off);
                emit_exec_number_guard(a, 9, 0, 10, guard!());
                let rd = free.pop().expect("chain reg underflow");
                a.ldr_d_imm(rd, 22, off);
                vregs.push((rd, false));
            }
            ChainOp::LoadProp(off, st) => {
                // Receiver-direct compact property probe. Unlike the ordinary property
                // template, a successful read never materializes a wide Value or pushes the
                // property's word: the numeric payload enters the chain's FP register stack
                // directly.
                #[cfg(all(test, target_arch = "aarch64"))]
                if off == u32::MAX {
                    TEST_THIS_CHAIN_OPS.with(|ops| ops.set(ops.get() + 1));
                }
                if off == u32::MAX {
                    a.ldr_imm(10, 19, 48); // ctx.this_word
                    emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 12, guard!());
                    emit_exec_payload(a, 10, 10);
                } else {
                    a.ldr_imm(10, 22, off);
                    emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 12, guard!());
                    emit_exec_payload(a, 10, 10);
                }
                a.add_imm(11, 10, obj_off);
                a.ldrb_imm(14, 11, ex);
                guard_exotic_ordinary(a, layout, 14, 14, guard!());
                a.ldrb_imm(14, 11, plain);
                a.cbz(14, false, guard!());
                a.ldr_w_imm(14, 11, sh);
                a.mov_imm64(16, st.recv_shape as u64);
                a.cmp_reg_w(14, 16);
                a.b_cond(C_NE, guard!());
                if st.depth >= 1 {
                    a.ldr_imm(17, 11, pr);
                    a.cbz(17, true, guard!());
                    a.add_imm(11, 17, obj_off);
                    a.ldrb_imm(14, 11, ex);
                    guard_exotic_ordinary(a, layout, 14, 14, guard!());
                    a.ldrb_imm(14, 11, plain);
                    a.cbz(14, false, guard!());
                    a.ldr_w_imm(14, 11, sh);
                    let expected = if st.depth == 1 {
                        st.holder_shape
                    } else {
                        st.mid_shape
                    };
                    a.mov_imm64(16, expected as u64);
                    a.cmp_reg_w(14, 16);
                    a.b_cond(C_NE, guard!());
                }
                if st.depth == 2 {
                    a.ldr_imm(17, 11, pr);
                    a.cbz(17, true, guard!());
                    a.add_imm(11, 17, obj_off);
                    a.ldrb_imm(14, 11, ex);
                    guard_exotic_ordinary(a, layout, 14, 14, guard!());
                    a.ldrb_imm(14, 11, plain);
                    a.cbz(14, false, guard!());
                    a.ldr_w_imm(14, 11, sh);
                    a.mov_imm64(16, st.holder_shape as u64);
                    a.cmp_reg_w(14, 16);
                    a.b_cond(C_NE, guard!());
                }
                fields_len(a, layout, 16, 11);
                a.mov_imm64(13, st.slot as u64);
                a.cmp_reg_x(13, 16);
                a.b_cond(C_HS, guard!());
                fields_data(a, layout, 15, 11);
                a.mov_imm64(16, es);
                a.madd(15, 13, 16, 15);
                guard_prop_data(a, 9, 15, ea, guard!());
                let rd = free.pop().expect("chain reg underflow");
                if layout.entry_accessor == layout.entry_value + 8 {
                    // Packed Property value: all bit patterns outside the reserved tagged
                    // prefixes are Numbers (including the canonical NaN prefix). Object has a
                    // negative prefix and therefore needs its own rejection before the range.
                    a.ldur(13, 15, ev);
                    a.lsr_imm(9, 13, 48);
                    a.movz(16, (crate::value::PACK_OBJ >> 48) as u32, 0);
                    a.cmp_reg_x(9, 16);
                    a.b_cond(C_HS, guard!());
                    let is_num = a.new_label();
                    a.movz(16, (crate::value::PACK_UNDEFINED >> 48) as u32, 0);
                    a.cmp_reg_x(9, 16);
                    a.b_cond(C_LO, is_num);
                    a.movz(16, (crate::value::PACK_SYM >> 48) as u32, 0);
                    a.cmp_reg_x(9, 16);
                    a.b_cond(C_LS, guard!());
                    a.bind(is_num);
                    a.fmov_d_x(rd, 13);
                } else {
                    a.ldrb_imm(9, 15, ev as u32);
                    a.cmp_imm_w(9, 4);
                    a.b_cond(C_NE, guard!());
                    a.ldur_d(rd, 15, ev + 8);
                }
                vregs.push((rd, false));
            }
            ChainOp::StoreProp(off, st) => {
                let (dv, _) = vregs.pop().expect("chain vstack");
                #[cfg(all(test, target_arch = "aarch64"))]
                if off == u32::MAX {
                    TEST_THIS_CHAIN_OPS.with(|ops| ops.set(ops.get() + 1));
                }
                if off == u32::MAX {
                    a.ldr_imm(10, 19, 48); // ctx.this_word
                    emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 12, guard!());
                    emit_exec_payload(a, 10, 10);
                } else {
                    a.ldr_imm(10, 22, off);
                    emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 12, guard!());
                    emit_exec_payload(a, 10, 10);
                }
                a.add_imm(11, 10, obj_off);
                a.ldrb_imm(14, 11, ex);
                guard_exotic_ordinary(a, layout, 14, 14, guard!());
                a.ldrb_imm(14, 11, plain);
                a.cbz(14, false, guard!());
                a.ldr_w_imm(14, 11, sh);
                a.mov_imm64(16, st.recv_shape as u64);
                a.cmp_reg_w(14, 16);
                a.b_cond(C_NE, guard!());
                fields_len(a, layout, 16, 11);
                a.mov_imm64(13, st.slot as u64);
                a.cmp_reg_x(13, 16);
                a.b_cond(C_HS, guard!());
                fields_data(a, layout, 15, 11);
                a.mov_imm64(16, es);
                a.madd(15, 13, 16, 15);
                guard_prop_data(a, 9, 15, ea, guard!());
                guard_prop_writable(a, 9, 15, ew, guard!());
                if layout.entry_accessor == layout.entry_value + 8 {
                    emit_exec_number_store(a, dv, 15, ev, 13);
                } else {
                    a.movz(9, 4, 0);
                    a.stur(9, 15, ev);
                    a.stur_d(dv, 15, ev + 8);
                }
                #[cfg(test)]
                {
                    // The chain's receiver cache can keep x8 live across this test-only
                    // callback; the common event helper saves x2..x7 and d16..d31.
                    a.sub_imm(31, 31, 16);
                    a.stp_off(8, 9, 0);
                    regions::emit_lowering_event(
                        a,
                        record_chain_property_store as *const () as u64,
                    );
                    a.ldp_off(8, 9, 0);
                    a.add_imm(31, 31, 16);
                }
                free.push(dv);
            }
            ChainOp::Update(off, kind) => {
                if let Some(k) = rcache.iter().position(|c| c.off == off) {
                    let ent = rcache.remove(k);
                    rfree.push(ent.base);
                    if let RcMode::Mirror { mpreg, mlreg } = ent.mode {
                        rfree.push(mpreg);
                        rfree.push(mlreg);
                    }
                }
                a.ldr_imm(9, 22, off);
                emit_exec_number_guard(a, 9, 0, 10, guard!());
                let dec = matches!(
                    kind,
                    UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
                );
                let f = if dec { 1 } else { 0 };
                match kind {
                    UpdKind::PreInc | UpdKind::PreDec => {
                        let rd = free.pop().expect("chain reg underflow");
                        a.ldr_d_imm(rd, 22, off);
                        a.fmov_one(0);
                        a.f_arith(f, rd, rd, 0);
                        emit_exec_number_store(a, rd, 22, off as i32, 9);
                        vregs.push((rd, false));
                    }
                    UpdKind::PostInc | UpdKind::PostDec => {
                        let rd = free.pop().expect("chain reg underflow");
                        a.ldr_d_imm(rd, 22, off);
                        a.fmov_one(0);
                        a.f_arith(f, 1, rd, 0);
                        emit_exec_number_store(a, 1, 22, off as i32, 9);
                        vregs.push((rd, false)); // the old value is the result
                    }
                    UpdKind::IncDiscard | UpdKind::DecDiscard => {
                        a.ldr_d_imm(0, 22, off);
                        a.fmov_one(1);
                        a.f_arith(f, 0, 0, 1);
                        emit_exec_number_store(a, 0, 22, off as i32, 9);
                    }
                }
            }
            ChainOp::GetElem(xoff) | ChainOp::SetElem(xoff, _) => {
                let is_set = matches!(*cop, ChainOp::SetElem(..));
                let keep = matches!(*cop, ChainOp::SetElem(_, true));
                let (dv, viv) = if is_set {
                    vregs.pop().expect("chain vstack")
                } else {
                    (0, false)
                };
                let (dk, _) = vregs.pop().expect("chain vstack");
                // key is exactly a u32
                a.fcvtzu_w_d(9, dk);
                a.ucvtf_d_w(0, 9);
                a.fcmp(dk, 0);
                a.b_cond(C_NE, guard!());
                let packed_done = a.new_label();
                let typed = a.new_label();
                let typed_hit = a.new_label();
                let cached = rcache.iter().position(|c| c.off == xoff);
                let mode: Option<(u32, u32)> = match cached {
                    Some(k) => {
                        let ent = &rcache[k];
                        a.mov(11, ent.base);
                        match ent.mode {
                            RcMode::Mirror { mpreg, mlreg } => {
                                // A negative length is the typed-view marker, never an array length.
                                a.cmp_imm_x(mlreg, 0);
                                a.b_cond(C_MI, typed);
                                Some((mpreg, mlreg))
                            }
                            RcMode::Classic => {
                                a.ldrb_imm(12, 11, plain);
                                a.cbz(12, false, typed);
                                None
                            }
                        }
                    }
                    None => {
                        // First access to this receiver in the chain: validate once.
                        a.ldr_imm(10, 22, xoff);
                        emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 12, guard!());
                        emit_exec_payload(a, 10, 10);
                        a.add_imm(11, 10, obj_off);
                        a.ldrb_imm(12, 11, ex);
                        guard_exotic_ordinary_or_array(a, layout, 12, 13, guard!());
                        a.ldrb_imm(12, 11, plain); // no side-table behavior
                        a.cbz(12, false, typed);
                        if rfree.len() >= 3 {
                            // Retain the existing numeric-kernel optimization when a mirror
                            // is coherent. Its absence selects native element access below,
                            // rather than forcing either representation through slow helpers.
                            let base = rfree.pop().unwrap();
                            let mpreg = rfree.pop().unwrap();
                            let mlreg = rfree.pop().unwrap();
                            let no_mirror = a.new_label();
                            a.mov(base, 11);
                            a.movz(mpreg, 0, 0);
                            a.movz(mlreg, 0, 0);
                            a.ldrb_imm(12, 11, mf);
                            let mask = asm::logical_imm_w(
                                (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32,
                            )
                            .unwrap();
                            a.logic_imm_w(0, 12, 12, mask);
                            a.cmp_imm_w(
                                12,
                                (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32,
                            );
                            a.b_cond(C_NE, no_mirror);
                            a.ldr_imm(12, 11, mirror);
                            a.cbz(12, true, no_mirror);
                            if packed_elem_inlinable(layout) {
                                // Only indexed entry storage is updated alongside this pinned
                                // mirror. A keyless packed store invalidates its optional mirror;
                                // never retain such a pointer across a store through an alias.
                                a.ldr_imm(13, 12, layout.dense_packed as u32);
                                a.cbnz(13, true, no_mirror);
                                a.ldrb_imm(13, 12, layout.dense_inline_len as u32);
                                a.cbnz(13, false, no_mirror);
                            }
                            a.ldr_imm(mpreg, 12, mvp);
                            a.ldr_imm(mlreg, 12, mvl);
                            a.bind(no_mirror);
                            rcache.push(RcEnt {
                                off: xoff,
                                base,
                                mode: RcMode::Mirror { mpreg, mlreg },
                            });
                            Some((mpreg, mlreg))
                        } else {
                            if let Some(base) = rfree.pop() {
                                a.mov(base, 11);
                                rcache.push(RcEnt {
                                    off: xoff,
                                    base,
                                    mode: RcMode::Classic,
                                });
                            }
                            None
                        }
                    }
                };
                if let Some((mpreg, mlreg)) = mode {
                    // Mirror-pinned receiver: bounds against the register copy, then one
                    // indexed access. Stores also sync the canonical entry payload (readers
                    // outside the chain trust entries) and keep the ALL_I32 flag honest.
                    let native_elements = a.new_label();
                    a.cbz(mpreg, true, native_elements);
                    a.cmp_reg_x(9, mlreg);
                    a.b_cond(C_HS, guard!());
                    if !is_set {
                        a.ldr_d_lsl3(dk, mpreg, 9);
                    } else {
                        a.ldr_imm(12, 11, el);
                        a.cbz(12, true, guard!());
                        a.ldr_imm(12, 12, evp);
                        a.add_shifted(12, 12, 9, 2);
                        a.ldr_w_imm(13, 12, 0);
                        a.cmn_imm_w(13, 1);
                        a.b_cond(C_EQ, guard!()); // hole: property creation → plain path
                        fields_data(a, layout, 15, 11);
                        a.movz(14, es as u32, 0);
                        a.madd(15, 13, 14, 15);
                        if layout.entry_accessor == layout.entry_value + 8 {
                            a.fmov_x_d(14, dv);
                            a.fcmp(dv, dv);
                            let encoded = a.new_label();
                            a.b_cond(C_VS ^ 1, encoded);
                            a.mov_imm64(14, f64::NAN.to_bits());
                            a.bind(encoded);
                            a.stur(14, 15, num_ev);
                        } else {
                            a.stur_d(dv, 15, num_ev);
                        }
                        a.str_d_lsl3(dv, mpreg, 9);
                        // Flag-first ALL_I32 upkeep (dv int-ness is unknown in this tier).
                        let i32_done = a.new_label();
                        a.ldrb_imm(13, 11, mf);
                        let i32_bit =
                            asm::logical_imm_w(crate::value::MIRROR_ALL_I32 as u32).unwrap();
                        a.logic_imm_w(0, 12, 13, i32_bit);
                        a.cbz(12, false, i32_done);
                        a.fcvtzs_w_d(12, dv);
                        a.scvtf_d_w(1, 12);
                        a.fmov_x_d(12, 1);
                        a.fmov_x_d(14, dv);
                        a.cmp_reg_x(12, 14);
                        a.b_cond(C_EQ, i32_done);
                        let clear =
                            asm::logical_imm_w(!(crate::value::MIRROR_ALL_I32 as u32)).unwrap();
                        a.logic_imm_w(0, 13, 13, clear);
                        a.strb_imm(13, 11, mf);
                        a.bind(i32_done);
                    }
                    a.b(packed_done);
                    a.bind(native_elements);
                }
                if packed_elem_inlinable(layout)
                    && layout.entry_value == layout.property_value
                    && layout.entry_accessor == layout.property_meta
                {
                    // The receiver is already validated (and cached across this helper-free
                    // chain). Packed array literals have no legacy numeric mirror.
                    let indexed = a.new_label();
                    a.ldr_imm(12, 11, el);
                    a.cbz(12, true, guard!());
                    emit_packed_elements_base(a, layout, indexed);
                    a.cmp_reg_x(9, 14);
                    a.b_cond(C_HS, guard!());
                    a.add_shifted(15, 15, 9, 4);
                    guard_prop_data(a, 14, 15, layout.property_meta as u32, guard!());
                    if is_set {
                        guard_prop_writable(a, 14, 15, layout.property_meta as u32, guard!());
                    }
                    if is_set {
                        emit_packed_existing_guard(a, layout, 15, guard!());
                    } else {
                        emit_packed_number_read_guard(a, layout, 15, guard!());
                    }
                    if is_set {
                        a.fmov_x_d(16, dv);
                        a.fcmp(dv, dv);
                        let encoded = a.new_label();
                        a.b_cond(C_VS ^ 1, encoded);
                        a.mov_imm64(16, f64::NAN.to_bits());
                        a.bind(encoded);
                        a.stur(16, 15, layout.property_value as i32);
                        emit_mirror_store(
                            a,
                            layout,
                            11,
                            MirrorKey::F64InDreg(dk),
                            MirrorVal::Num(dv, false),
                        );
                    } else {
                        a.fmov_d_x(dk, 12);
                    }
                    a.b(packed_done);
                    a.bind(indexed);
                }
                let mirror_done = a.new_label();
                let classic = a.new_label();
                if !is_set {
                    // Mirror read: coherent + hole-free ⇒ bounds + one indexed load, value
                    // known Num. Any miss (flags, range) answers classically below.
                    a.ldrb_imm(12, 11, mf);
                    let mask = asm::logical_imm_w(
                        (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32,
                    )
                    .unwrap();
                    a.logic_imm_w(0, 12, 12, mask);
                    a.cmp_imm_w(
                        12,
                        (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32,
                    );
                    a.b_cond(C_NE, classic);
                    a.ldr_imm(12, 11, mirror);
                    a.cbz(12, true, classic);
                    a.ldr_imm(14, 12, mvl);
                    a.cmp_reg_x(9, 14);
                    a.b_cond(C_HS, classic);
                    a.ldr_imm(12, 12, mvp);
                    a.ldr_d_lsl3(dk, 12, 9);
                    a.b(mirror_done);
                } else {
                    // Mirror-slim store: MIRROR_OK proves every (non-hole) element is a plain
                    // writable data Num, so the accessor/writable/old-value dance collapses to
                    // a payload overwrite in the entry plus the mirror word. A hole (elems
                    // NO_SLOT) would CREATE a property — classic handles it.
                    a.ldrb_imm(12, 11, mf);
                    let ok_bit = asm::logical_imm_w(crate::value::MIRROR_OK as u32).unwrap();
                    a.logic_imm_w(0, 12, 12, ok_bit);
                    a.cbz(12, false, classic);
                    a.ldr_imm(12, 11, mirror);
                    a.cbz(12, true, classic);
                    a.ldr_imm(14, 12, mvl);
                    a.cmp_reg_x(9, 14);
                    a.b_cond(C_HS, classic);
                    a.ldr_imm(12, 11, el);
                    a.cbz(12, true, classic);
                    a.ldr_imm(12, 12, evp);
                    a.add_shifted(12, 12, 9, 2);
                    a.ldr_w_imm(13, 12, 0);
                    a.cmn_imm_w(13, 1);
                    a.b_cond(C_EQ, classic);
                    fields_data(a, layout, 15, 11);
                    a.movz(14, es as u32, 0);
                    a.madd(15, 13, 14, 15);
                    if layout.entry_accessor == layout.entry_value + 8 {
                        a.fmov_x_d(14, dv);
                        a.fcmp(dv, dv);
                        let encoded = a.new_label();
                        a.b_cond(C_VS ^ 1, encoded);
                        a.mov_imm64(14, f64::NAN.to_bits());
                        a.bind(encoded);
                        a.stur(14, 15, num_ev);
                    } else {
                        a.stur_d(dv, 15, num_ev);
                    }
                    a.ldr_imm(12, 11, mirror);
                    a.ldr_imm(12, 12, mvp);
                    a.str_d_lsl3(dv, 12, 9);
                    // Flag-first ALL_I32 upkeep (dv int-ness is unknown in this tier).
                    let i32_done = a.new_label();
                    a.ldrb_imm(13, 11, mf);
                    let i32_bit = asm::logical_imm_w(crate::value::MIRROR_ALL_I32 as u32).unwrap();
                    a.logic_imm_w(0, 12, 13, i32_bit);
                    a.cbz(12, false, i32_done);
                    a.fcvtzs_w_d(12, dv);
                    a.scvtf_d_w(1, 12);
                    a.fmov_x_d(12, 1);
                    a.fmov_x_d(14, dv);
                    a.cmp_reg_x(12, 14);
                    a.b_cond(C_EQ, i32_done);
                    let clear = asm::logical_imm_w(!(crate::value::MIRROR_ALL_I32 as u32)).unwrap();
                    a.logic_imm_w(0, 13, 13, clear);
                    a.strb_imm(13, 11, mf);
                    a.bind(i32_done);
                    a.b(mirror_done);
                }
                a.bind(classic);
                a.ldr_imm(12, 11, el);
                a.cbz(12, true, guard!());
                a.ldr_imm(14, 12, evl);
                a.cmp_reg_x(9, 14);
                a.b_cond(C_HS, guard!());
                a.ldr_imm(12, 12, evp);
                a.add_shifted(12, 12, 9, 2);
                a.ldr_w_imm(13, 12, 0);
                a.cmn_imm_w(13, 1);
                a.b_cond(C_EQ, guard!());
                fields_data(a, layout, 15, 11);
                a.movz(9, es as u32, 0); // entry stride (< 65536; the key index in x9 is dead)
                a.madd(15, 13, 9, 15);
                guard_prop_data(a, 9, 15, ea, guard!());
                if layout.entry_accessor == layout.entry_value + 8 {
                    // A length update or descriptor operation can invalidate the optional
                    // mirror without changing ordinary numeric elements. Read authoritative
                    // packed properties directly instead of replaying the entire chain.
                    if is_set {
                        guard_prop_writable(a, 14, 15, ew, guard!());
                    }
                    if is_set {
                        emit_packed_existing_guard(a, layout, 15, guard!());
                    } else {
                        emit_packed_number_read_guard(a, layout, 15, guard!());
                    }
                    if is_set {
                        a.fmov_x_d(16, dv);
                        a.fcmp(dv, dv);
                        let encoded = a.new_label();
                        a.b_cond(C_VS ^ 1, encoded);
                        a.mov_imm64(16, f64::NAN.to_bits());
                        a.bind(encoded);
                        a.stur(16, 15, ev);
                        a.strb_imm(31, 11, mf);
                    } else {
                        a.fmov_d_x(dk, 12);
                    }
                    a.b(mirror_done);
                }
                if is_set {
                    guard_prop_writable(a, 9, 15, ew, guard!());
                    // commit: entry = Num(dv)
                    a.movz(9, 4, 0);
                    a.stur(9, 15, ev);
                    a.stur_d(dv, 15, ev + 8);
                    // Element mirror: dv is a proven Num; int-ness is unknown in this tier.
                    emit_mirror_store(
                        a,
                        layout,
                        11,
                        MirrorKey::F64InDreg(dk),
                        MirrorVal::Num(dv, false),
                    );
                    a.bind(mirror_done);
                    free.push(dk);
                    if keep {
                        vregs.push((dv, viv)); // v stays the virtual result (a Num)
                    } else {
                        free.push(dv);
                    }
                } else {
                    // element must be a Num to stay in a register
                    a.ldrb_imm(9, 15, ev as u32);
                    a.cmp_imm_w(9, 4);
                    a.b_cond(C_NE, guard!());
                    a.ldur_d(dk, 15, ev + 8); // reuse the key's register for the element
                    a.bind(mirror_done);
                    vregs.push((dk, false));
                }
                a.b(packed_done);
                a.bind(typed);
                if cached.is_none() {
                    if let Some(entry) = rcache.iter().find(|entry| entry.off == xoff) {
                        a.mov(entry.base, 11);
                        if let RcMode::Mirror { mpreg, mlreg } = entry.mode {
                            a.movz(mpreg, 0, 0);
                            a.mov_imm64(mlreg, usize::MAX as u64);
                        }
                    }
                }
                typed_array::emit(
                    a,
                    layout,
                    is_set.then_some(typed_array::WriteValue::Register(dv)),
                    typed_hit,
                    guard!(),
                );
                a.bind(typed_hit);
                if !is_set {
                    a.fmov_d_d(dk, 1);
                }
                a.bind(packed_done);
            }
            ChainOp::Arith(f) => {
                let (rm, _) = vregs.pop().expect("chain vstack");
                let (rn, _) = vregs.pop().expect("chain vstack");
                a.f_arith(f, rn, rn, rm);
                vregs.push((rn, false));
                free.push(rm);
            }
            ChainOp::Bit(code) => {
                let (rm, mi) = vregs.pop().expect("chain vstack");
                let (rn, ni) = vregs.pop().expect("chain vstack");
                // ToInt32 each operand: fcvtzs truncates; the low 32 bits are the mod-2^32 wrap.
                // Known int-valued skips the round-trip guard (the conversion is exact by
                // construction); otherwise guard like the standalone template.
                for (src, iv, out) in [(rn, ni, 9u32), (rm, mi, 10u32)] {
                    if !iv && jscvt_available() {
                        a.fjcvtzs_w_d(out, src); // exact ToInt32: no guard needed
                        continue;
                    }
                    a.fcvtzs_x_d(out, src);
                    if !iv {
                        a.scvtf_d_x(0, out);
                        a.frintz(1, src);
                        a.fcmp(0, 1);
                        a.b_cond(C_NE, guard!());
                        a.cmn_imm_x(out, 1);
                        a.b_cond(6, guard!()); // VS: the +2^63 saturation edge
                    }
                }
                match code {
                    0 => a.logic_w(0, 11, 9, 10),
                    1 => a.logic_w(1, 11, 9, 10),
                    2 => a.logic_w(2, 11, 9, 10),
                    3 => a.shift_w(0, 11, 9, 10),
                    4 => a.shift_w(1, 11, 9, 10),
                    _ => a.shift_w(2, 11, 9, 10),
                }
                if code == 4 {
                    a.ucvtf_d_w(rn, 11); // >>> yields an unsigned 32-bit result
                } else {
                    a.scvtf_d_w(rn, 11);
                }
                vregs.push((rn, true));
                free.push(rm);
            }
            ChainOp::Neg => {
                let (rt, _) = *vregs.last().expect("chain vstack");
                a.fneg(rt, rt);
                // Clear the int-valued flag: -(-2^63) = +2^63 escapes the guard-free i64 range.
                let top = vregs.len() - 1;
                vregs[top].1 = false;
            }
            ChainOp::Store(off) => {
                if let Some(k) = rcache.iter().position(|c| c.off == off) {
                    let ent = rcache.remove(k);
                    rfree.push(ent.base);
                    if let RcMode::Mirror { mpreg, mlreg } = ent.mode {
                        rfree.push(mpreg);
                        rfree.push(mlreg);
                    }
                }
                let (dv, _) = vregs.pop().expect("chain vstack");
                // The overwrite is the old word's release; no bail is needed.
                emit_exec_number_store(a, dv, 22, off as i32, 9);
                free.push(dv);
            }
            ChainOp::Pop => {
                let (r, _) = vregs.pop().expect("chain vstack");
                free.push(r);
            }
            ChainOp::Dup => {
                let &(src, iv) = vregs.last().expect("chain vstack");
                let rd = free.pop().expect("chain reg underflow");
                a.fmov_d_d(rd, src);
                vregs.push((rd, iv));
            }
            ChainOp::KeyNop => {}
            ChainOp::LoadName(cache_ptr) => {
                // The validator clobbers x9-x17 only — receiver caches (x2-x8) survive it.
                // Shared cache validation (scope or global mode) leaves x14 → the Value.
                emit_name_ic_value_ptr(a, layout, cache_ptr, guard!(), true);
                let rd = free.pop().expect("chain reg underflow");
                let loaded = a.new_label();
                if layout.entry_accessor == layout.entry_value + 8 {
                    let wide = a.new_label();
                    a.cbz(7, false, wide);
                    a.ldur(9, 14, 0);
                    a.lsr_imm(10, 9, 48);
                    let number = a.new_label();
                    a.movz(11, (crate::value::PACK_OBJ >> 48) as u32, 0);
                    a.cmp_reg_x(10, 11);
                    a.b_cond(C_HS, guard!());
                    a.movz(11, (crate::value::PACK_UNDEFINED >> 48) as u32, 0);
                    a.cmp_reg_x(10, 11);
                    a.b_cond(C_LO, number);
                    a.movz(11, (crate::value::PACK_SYM >> 48) as u32, 0);
                    a.cmp_reg_x(10, 11);
                    a.b_cond(C_LS, guard!());
                    a.bind(number);
                    a.fmov_d_x(rd, 9);
                    a.b(loaded);
                    a.bind(wide);
                }
                a.ldurb(9, 14, 0);
                a.cmp_imm_w(9, 4);
                a.b_cond(C_NE, guard!()); // only a Num can live in a register
                a.ldur_d(rd, 14, 8);
                a.bind(loaded);
                vregs.push((rd, false));
            }
            ChainOp::CmpBranch(neg, target) => {
                let (rm, _) = vregs.pop().expect("chain vstack");
                let (rn, _) = vregs.pop().expect("chain vstack");
                a.fcmp(rn, rm);
                a.b_cond(neg, pc_labels[target]);
                free.push(rm);
                free.push(rn);
            }
        }
        if used > 0 {
            bails.push((idx, bail, pre_op));
        }
    }
    // Chain finished: spill any remaining virtual values to the real stack, in stack order.
    for &(r, _) in &vregs {
        emit_exec_number_store(a, r, 20, 0, 9);
        a.add_imm(20, 20, 8);
    }
    a.b(done);
    // ---- bail paths: spill the pre-op virtual stack, then re-run the rest via the helper ----
    // Every bail replays the same suffix of the chain from its op onward, and each replay step
    // depends only on its pc and the live operand stack. One shared ladder of replay steps
    // serves all bails (each enters at its own op), keeping fallback code linear in the chain
    // length rather than quadratic.
    let Some(first) = bails.iter().map(|&(idx, _, _)| idx).min() else {
        a.bind(done);
        return;
    };
    let rungs: Vec<usize> = (first..chain.len()).map(|_| a.new_label()).collect();
    for (idx, label, snap) in bails {
        a.bind(label);
        for &(r, _) in &snap {
            emit_exec_number_store(a, r, 20, 0, 9);
            a.add_imm(20, 20, 8);
        }
        a.b(rungs[idx - first]);
    }
    for (k, (cop2, pc2)) in chain.iter().enumerate().skip(first) {
        a.bind(rungs[k - first]);
        match cop2 {
            ChainOp::GetElem(_) => emit_op_helper(a, H_GET_ELEM, *pc2 as u32, l_unwind),
            ChainOp::CmpBranch(_, target) => {
                // generic compare (pushes a bool) + pop-and-branch, like the unfused pair
                emit_exec(a, *pc2 as u32, l_unwind);
                emit_cond(a, COND_POP_TRUTHY, l_unwind);
                a.cbz(1, false, pc_labels[*target]);
            }
            _ => emit_exec(a, *pc2 as u32, l_unwind),
        }
    }
    a.b(done);
    a.bind(done);
}

// ---------------------------------------------------------------------------------------------
// Loop-spanning chains: a fully-chainable, branch-free loop keeps its locals in registers
// across the back edge. Slot loads and type guards hoist into a one-time preamble; memory is
// written only on loop exit or on a bail, which flushes and jumps into the plain templates of
// the same region (still emitted as usual — the loop head's canonical label points at the chain
// entry, so both the fallthrough entry and plain back-edge jumps re-enter the chain). The loop
// is rotated: the condition runs once at entry (copy A, exits with nothing dirty) and again at
// the bottom of the body (copy B, exits through a flush), so the back edge is a single branch.
//
// Value kinds (decided by the planner, followed verbatim by the emitter):
//   K — compile-time f64 constant, materialized lazily (bit-op immediates are free)
//   I — exact integer in an x-register (x2..x8): keys and bit ops are single instructions;
//       float uses convert with one scvtf. `neg` = may be negative (sign-correctness matters).
//   D — f64 in a d-register (transients d16..; residents d8..d15); `iv` = proven integral with
//       |v| < 2^62, so ToInt32 is a bare fcvtzs with no round-trip guard.
//
// Residency: slots read before written preload behind a tag guard (a failed guard runs the
// whole loop through the plain templates); ±1-update targets whose stores stay integer live as
// I with a per-update magnitude guard that keeps them exact (JS numbers stop moving under ±1 at
// 2^53, so exceeding it must bail rather than diverge); everything else numeric lives as F.
// Slots written before read ("virgins") get no preamble load: every flush is a plain overwrite.
// ---------------------------------------------------------------------------------------------

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// What a chain op pushes, precomputed by the planner (see the module comment above).
#[derive(Clone, Copy, PartialEq, Debug)]
enum PushKind {
    None,
    K(u64),
    I { neg: bool },
    D { iv: bool },
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// Where a loop-touched numeric slot lives during the run.
#[derive(Clone, Copy, PartialEq, Debug)]
enum SlotRes {
    /// f64 home in a d-register (d8..d15).
    F(u32),
    /// Exact-integer home in an x-register (x2..x8).
    I(u32),
    /// Not register-resident: per-access guarded memory ops, like a plain chain.
    None,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Debug)]
struct SlotPlan {
    off: u32,
    res: SlotRes,
    /// Read (or ±1-updated) before any region store: preamble tag-guard + load.
    preload: bool,
    /// Some Store/Update writes it in the region (it must flush on exits and bails).
    stored: bool,
    /// Stored before ever read: the preamble does not load it, and flushes plain-overwrite it.
    virgin: bool,
    /// F resident with a one-time exact-int entry check: loads carry `integral, |v| ≤ 2^31`,
    /// so integer arithmetic takes them with a bare fcvtzs.
    int_checked: bool,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
struct LoopPlan {
    head: usize,
    exit_pc: usize,
    /// Discarded local reads checked once at entry. This helper-free region has no Tdz op;
    /// failure resumes the original loop so zero-trip and RHS exception order stay unchanged.
    initialization_guards: Vec<u32>,
    /// Translated ops from the head to the latch; CmpBranch ends the condition prefix.
    chain: Vec<(ChainOp, usize)>,
    /// Chain entries `[0, cond_len)` are the condition (emitted twice: entry + bottom).
    cond_len: usize,
    /// Per chain index: what the op pushes (kind agreement between planner and emitter).
    kinds: Vec<PushKind>,
    slots: Vec<SlotPlan>,
    /// Receiver slots validated once. In mirror mode the cached register holds the raw-f64
    /// element buffer's data pointer (`Props::mirror`) with the length in `len_reg`, and
    /// element reads are one indexed load; classic mode caches the object base and walks
    /// entries per access.
    receivers: Vec<ReceiverPlan>,
    /// GetElem chain idx → pin register holding its (guarded) result for later reuse.
    elem_retain: Vec<(usize, u32)>,
    /// GetElem chain idx → the retaining chain idx whose pin it copies from.
    elem_reuse: Vec<(usize, usize)>,
    /// Bit (chain idx, operand side) → pin register: retain the guarded ToInt32 result / reuse.
    conv_retain: Vec<((usize, u8), u32)>,
    conv_reuse: Vec<((usize, u8), u32)>,
    /// Per SetElem chain idx: the stored value is a proven exact-i32 (mirror flag upkeep).
    setelem_i32: crate::fasthash::FastMap<usize, bool>,
    /// Cached free names read in the region, pinned once in the preamble. The region is
    /// helper-free and its op vocabulary writes only locals and elements, so a name binding
    /// cannot change while the loop spins: one validation covers every iteration (loop bounds
    /// are typically closure vars — `i < width` — and paid a full name-IC probe per iteration
    /// as plain chains).
    names: Vec<NamePlan>,
    /// Some pin drew from x23-x28 (callee-saved): the loop brackets itself with save/restore
    /// pairs — a preamble spill, and a reload on every exit and bail path.
    uses_ext: bool,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
struct NamePlan {
    /// The `NameIc` cell address (`Chunk::jit_name_cache_ptr`).
    ptr: usize,
    /// f64 home in a d-register (allocated from the same d8..d15 bank as `SlotRes::F`).
    dreg: u32,
    /// Preamble adds the one-time exact-int proof (the name feeds a key or bit op).
    int_checked: bool,
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// How a loop-chain receiver is cached (see `LoopPlan::receivers`).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
struct ReceiverPlan {
    off: u32,
    /// x16/x17 (receivers 3-4 draw from the pin pool): the validated object base.
    reg: u32,
    /// Element accesses go through the raw-f64 mirror (preamble-proven coherent and hole-free;
    /// the buffer pointer/length load per access — same cache line, far cheaper than the
    /// entry chase they replace).
    mirror: bool,
    /// Any int-typed element read flows from this receiver (preamble then requires
    /// `MIRROR_ALL_I32`, letting those reads use a bare fcvtzs).
    int_reads: bool,
    /// A numeric store in the region can clear ALL_I32 on this receiver through an alias.
    /// Such reads must revalidate the fact at the read, not truncate under the entry proof.
    recheck_i32: bool,
    /// Leftover pin registers holding the mirror length / mirror data / elems data / entries
    /// data pointers (all stable in-region: the vocabulary is helper-free and slim stores
    /// never grow or reallocate). Each pin shaves a dependent load off every element access —
    /// the hot lin_solve-shape loop hits one receiver 5-6 times per iteration.
    mlreg: Option<u32>,
    mpreg: Option<u32>,
    elpreg: Option<u32>,
    enreg: Option<u32>,
}

/// Integer-range bookkeeping for iv decisions: |v| ≤ 2^exp and integral. 255 = unknown/not
/// integral. Kept crude on purpose — it only has to prove products/sums of masked values stay
/// under 2^62.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[derive(Clone, Copy)]
struct NumInfo {
    integral: bool,
    exp: u32,
    neg: bool,
}
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
impl NumInfo {
    fn unknown() -> NumInfo {
        NumInfo {
            integral: false,
            exp: 255,
            neg: true,
        }
    }
    fn iv(&self) -> bool {
        self.integral && self.exp <= 62
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn plan_loop(
    chunk: &Chunk,
    ops: &[crate::bytecode::Op],
    head: usize,
    targeted: &[bool],
    layout: &crate::value::JitLayout,
    fast: u32,
    cfg: &crate::jit_ir::Cfg,
) -> Option<LoopPlan> {
    use crate::bytecode::Op;
    if fast & 32768 == 0 {
        return None;
    }
    macro_rules! reject {
        ($why:expr) => {{
            if env_flag!("TRUST_JS_JIT_LOOPLOG") {
                eprintln!("[jit-loop] head {head}: reject: {}", $why);
            }
            return None;
        }};
    }
    let in_range = |s: u16| (s as u32) * 8 + 8 < 4096;
    let name_ok = fast & 8192 != 0 && load_name_inlinable(layout);

    // ---- region discovery: the shared CFG admits only the old emitter's linear-loop shape.
    // Forward diamonds are intentionally left for the SSA/fixed-home region lowering.
    let jump_pc = cfg.linear_loop_latch(ops, head)?;
    if jump_pc == head + 1 {
        reject!("empty region");
    }
    debug_assert!(targeted[head]);

    // ---- translate the region; require full coverage and exactly one fused exit branch
    let mut chain: Vec<(ChainOp, usize)> = Vec::new();
    let mut vdepth = 0usize;
    let mut exit_pc = None;
    let mut cond_len = None;
    let mut initialization_guards = Vec::new();
    let mut pc = head;
    while pc < jump_pc {
        if let Some(slot) = local_read_discard_pair(ops, pc, targeted) {
            let off = slot as u32 * 8;
            if !initialization_guards.contains(&off) {
                initialization_guards.push(off);
            }
            // This read's value is discarded, not used numerically. The preamble only
            // checks initialization; a miss replays the ordinary instructions. Keep both
            // PCs in the plan for exact original-bytecode mapping without a numeric home.
            chain.push((ChainOp::KeyNop, pc));
            chain.push((ChainOp::KeyNop, pc + 1));
            pc += 2;
            continue;
        }
        let (cop, push, pop): (ChainOp, usize, usize) = match &ops[pc] {
            Op::Const(k) => (ChainOp::ConstNum(chunk.jit_const_num(*k)?), 1, 0),
            Op::LoadLocal(s) if in_range(*s) => (ChainOp::Load(*s as u32 * 8), 1, 0),
            Op::UpdateLocal(s, kind) if in_range(*s) => {
                let pushes = !matches!(kind, UpdKind::IncDiscard | UpdKind::DecDiscard);
                (ChainOp::Update(*s as u32 * 8, *kind), pushes as usize, 0)
            }
            Op::GetElemLocal(x) if in_range(*x) && vdepth >= 1 => {
                (ChainOp::GetElem(*x as u32 * 8), 1, 1)
            }
            Op::SetElemLocal(x) if in_range(*x) && vdepth >= 2 => {
                (ChainOp::SetElem(*x as u32 * 8, true), 1, 2)
            }
            Op::SetElemLocalDrop(x) if in_range(*x) && vdepth >= 2 => {
                (ChainOp::SetElem(*x as u32 * 8, false), 0, 2)
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div if vdepth >= 2 => {
                let f = match ops[pc] {
                    Op::Add => 0,
                    Op::Sub => 1,
                    Op::Mul => 2,
                    _ => 3,
                };
                (ChainOp::Arith(f), 1, 2)
            }
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr if vdepth >= 2 => {
                let code = match ops[pc] {
                    Op::BitAnd => 0,
                    Op::BitOr => 1,
                    Op::BitXor => 2,
                    Op::Shl => 3,
                    Op::UShr => 4,
                    _ => 5,
                };
                (ChainOp::Bit(code), 1, 2)
            }
            Op::Neg if vdepth >= 1 => (ChainOp::Neg, 1, 1),
            Op::StoreLocal(s) if in_range(*s) && vdepth >= 1 => {
                (ChainOp::Store(*s as u32 * 8), 0, 1)
            }
            Op::Pop if vdepth >= 1 => (ChainOp::Pop, 0, 1),
            Op::Dup if vdepth >= 1 => (ChainOp::Dup, 1, 0),
            Op::ToPropKeyLocal(_) if vdepth >= 1 => (ChainOp::KeyNop, 0, 0),
            Op::LoadName(_, c) if name_ok => {
                (ChainOp::LoadName(chunk.jit_name_cache_ptr(*c)), 1, 0)
            }
            Op::Lt
            | Op::Gt
            | Op::Le
            | Op::Ge
            | Op::StrictEq
            | Op::StrictNotEq
            | Op::EqEq
            | Op::NotEq
                if vdepth == 2 =>
            {
                match ops.get(pc + 1) {
                    Some(Op::JumpIfFalse(t)) if (*t as usize) > jump_pc => {
                        if exit_pc.is_some() {
                            return None; // one exit only
                        }
                        let neg = match ops[pc] {
                            Op::Lt => 5,                  // PL (unordered jumps)
                            Op::Gt => 13,                 // LE
                            Op::Le => 8,                  // HI
                            Op::Ge => 11,                 // LT
                            Op::StrictEq | Op::EqEq => 1, // NE
                            _ => 0,                       // EQ
                        };
                        exit_pc = Some(*t as usize);
                        chain.push((ChainOp::CmpBranch(neg, *t as usize), pc));
                        cond_len = Some(chain.len());
                        vdepth = 0;
                        pc += 2;
                        continue;
                    }
                    _ => return None,
                }
            }
            _ => reject!(format!("unchainable op at pc {pc}: {:?}", ops[pc])),
        };
        if vdepth - pop + push > 8 {
            reject!("vdepth > 8");
        }
        vdepth = vdepth - pop + push;
        chain.push((cop, pc));
        pc += 1;
    }
    let exit_pc = exit_pc?;
    let cond_len = cond_len?;
    if vdepth != 0 || cond_len == chain.len() {
        reject!("unbalanced or empty body");
    }

    // ---- value graph: per produced value, its consumers (for elem-int and residency choices)
    #[derive(Clone, Copy, PartialEq)]
    enum Use {
        Bit,
        Key,
        Cmp,
        Arith,
        Other,
    }
    let n = chain.len();
    // Node ids: one per chain index that pushes (Dup aliases its source).
    let mut consumers: Vec<Vec<Use>> = vec![Vec::new(); n];
    let mut slot_src: crate::fasthash::FastMap<u32, usize> = Default::default(); // off → node
    let mut slot_bind: crate::fasthash::FastMap<u32, usize> = Default::default();
    // Free names are loop-invariant (nothing in the vocabulary writes a binding): every read
    // of one cache ptr is the same node.
    let mut name_src: crate::fasthash::FastMap<usize, usize> = Default::default();
    let mut names_order: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut elem_nodes: Vec<usize> = Vec::new(); // GetElem chain indices
    let mut receivers: Vec<u32> = Vec::new();
    let mut stored: Vec<u32> = Vec::new();
    let mut updated: Vec<u32> = Vec::new();
    // Raw memo inputs: element reads as (chain idx, receiver, key node), element writes as
    // (chain idx, receiver), bit ops as (chain idx, lhs node, rhs node).
    let mut elem_reads: Vec<(usize, u32, usize)> = Vec::new();
    let mut elem_writes: Vec<(usize, u32)> = Vec::new();
    let mut bit_uses: Vec<(usize, usize, usize)> = Vec::new();
    // Result → operand edges for the needs-int propagation below.
    let mut flow_edges: Vec<(usize, usize)> = Vec::new();
    for (idx, (cop, _)) in chain.iter().enumerate() {
        match *cop {
            ChainOp::ConstNum(_) => stack.push(idx),
            ChainOp::Load(off) => {
                let node = match slot_bind.get(&off) {
                    Some(&b) => b,
                    None => *slot_src.entry(off).or_insert(idx),
                };
                stack.push(node);
            }
            ChainOp::Update(off, kind) => {
                slot_src.entry(off).or_insert(idx);
                if !updated.contains(&off) {
                    updated.push(off);
                }
                if !stored.contains(&off) {
                    stored.push(off);
                }
                // The update's own read counts as an int-friendly use.
                let cur = slot_bind.get(&off).copied().or(slot_src.get(&off).copied());
                if let Some(c) = cur {
                    consumers[c].push(Use::Arith);
                }
                slot_bind.insert(off, idx);
                let pushes = !matches!(kind, UpdKind::IncDiscard | UpdKind::DecDiscard);
                if pushes {
                    // Post forms push the OLD value — the same node as the pre-update binding,
                    // so a later identical use (an element key, typically) can be deduplicated.
                    match kind {
                        UpdKind::PostInc | UpdKind::PostDec => stack.push(cur.unwrap_or(idx)),
                        _ => stack.push(idx),
                    }
                }
            }
            ChainOp::GetElem(xoff) => {
                let k = stack.pop().expect("loop plan stack");
                consumers[k].push(Use::Key);
                if !receivers.contains(&xoff) {
                    receivers.push(xoff);
                }
                elem_reads.push((idx, xoff, k));
                elem_nodes.push(idx);
                stack.push(idx);
            }
            ChainOp::SetElem(xoff, keep) => {
                let v = stack.pop().expect("loop plan stack");
                let k = stack.pop().expect("loop plan stack");
                consumers[v].push(Use::Other);
                consumers[k].push(Use::Key);
                if !receivers.contains(&xoff) {
                    receivers.push(xoff);
                }
                elem_writes.push((idx, xoff));
                if keep {
                    stack.push(v);
                }
            }
            ChainOp::Arith(_) => {
                let b = stack.pop().expect("loop plan stack");
                let a_ = stack.pop().expect("loop plan stack");
                consumers[a_].push(Use::Arith);
                consumers[b].push(Use::Arith);
                flow_edges.push((idx, a_));
                flow_edges.push((idx, b));
                stack.push(idx);
            }
            ChainOp::Bit(_) => {
                let b = stack.pop().expect("loop plan stack");
                let a_ = stack.pop().expect("loop plan stack");
                consumers[a_].push(Use::Bit);
                consumers[b].push(Use::Bit);
                bit_uses.push((idx, a_, b));
                stack.push(idx);
            }
            ChainOp::Neg => {
                let v = stack.pop().expect("loop plan stack");
                consumers[v].push(Use::Arith);
                flow_edges.push((idx, v));
                stack.push(idx);
            }
            ChainOp::Store(off) => {
                let v = stack.pop().expect("loop plan stack");
                consumers[v].push(Use::Other);
                slot_bind.insert(off, v);
                if !stored.contains(&off) {
                    stored.push(off);
                }
            }
            ChainOp::Pop => {
                let v = stack.pop().expect("loop plan stack");
                consumers[v].push(Use::Other);
            }
            ChainOp::Dup => {
                let v = *stack.last().expect("loop plan stack");
                stack.push(v);
            }
            ChainOp::KeyNop => {}
            ChainOp::CmpBranch(..) => {
                let b = stack.pop().expect("loop plan stack");
                let a_ = stack.pop().expect("loop plan stack");
                consumers[a_].push(Use::Cmp);
                consumers[b].push(Use::Cmp);
            }
            ChainOp::LoadName(ptr) => {
                let node = *name_src.entry(ptr).or_insert(idx);
                if !names_order.contains(&ptr) {
                    names_order.push(ptr);
                }
                stack.push(node);
            }
            ChainOp::LoadProp(..) | ChainOp::StoreProp(..) => {
                unreachable!("loop discovery never admits property operations")
            }
        }
    }

    // Elem ops present require the inline layout; receivers must never be written in-region.
    if (!elem_nodes.is_empty() || !receivers.is_empty())
        && (fast & 1024 == 0 || !get_elem_inlinable(layout))
    {
        reject!("elem layout");
    }
    if receivers.len() > 4 {
        reject!("too many receivers");
    }
    for r in &receivers {
        if stored.contains(r) {
            reject!("stored receiver");
        }
    }
    // ---- slot classification
    let mut slot_offs: Vec<u32> = Vec::new();
    for (cop, _) in &chain {
        match *cop {
            ChainOp::Load(off) | ChainOp::Update(off, _) | ChainOp::Store(off)
                if !slot_offs.contains(&off) && !receivers.contains(&off) =>
            {
                slot_offs.push(off);
            }
            _ => {}
        }
    }
    // Read-before-store per slot: first access wins.
    let mut first_access: crate::fasthash::FastMap<u32, bool> = Default::default(); // true=read
    for (cop, _) in &chain {
        match *cop {
            ChainOp::Load(off) | ChainOp::Update(off, _) => {
                first_access.entry(off).or_insert(true);
            }
            ChainOp::Store(off) => {
                first_access.entry(off).or_insert(false);
            }
            _ => {}
        }
    }

    // needs-int: a value feeds a bit op or key, directly or through arithmetic whose result
    // does. This is what justifies speculative exact-int guards: a float here would have been
    // truncated (or bailed) downstream anyway, so proving int early only moves the check.
    let mut needs_int = vec![false; n];
    for (idx, uses) in consumers.iter().enumerate() {
        if uses.iter().any(|u| matches!(u, Use::Bit | Use::Key)) {
            needs_int[idx] = true;
        }
    }
    loop {
        let mut changed = false;
        for &(r, op) in &flow_edges {
            if needs_int[r] && !needs_int[op] {
                needs_int[op] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Elem-int decision: the value (transitively) feeds an int context.
    let elem_int: Vec<bool> = elem_nodes.iter().map(|&idx| needs_int[idx]).collect();

    // Names feeding int contexts get the one-time exact-int preamble proof (like int_checked
    // slots), so integer consumers take them with a bare fcvtzs.
    let int_checked_names: Vec<usize> = names_order
        .iter()
        .copied()
        .filter(|p| name_src.get(p).is_some_and(|&nd| needs_int[nd]))
        .collect();

    // Residency policy. x-registers are scarce (9 shared with transients), so they go where
    // integer latency matters: counters (±1 updates), loop-carried accumulators (read before
    // stored — the cross-iteration critical path), and stored slots whose values feed bit ops
    // or keys directly. Read-only preloads that feed int contexts stay in d-registers behind a
    // one-time exact-int entry check (`int_checked`): integer arithmetic takes them with a bare
    // fcvtzs. The sim rounds below demote any I candidate whose stores turn out non-integer.
    let mut i_slots: Vec<u32> = updated.clone();
    let mut int_checked: Vec<u32> = Vec::new();
    // Store-value nodes per slot, for the direct-consumer test.
    let mut store_nodes: crate::fasthash::FastMap<u32, Vec<usize>> = Default::default();
    {
        let mut stack2: Vec<usize> = Vec::new();
        let mut bind2: crate::fasthash::FastMap<u32, usize> = Default::default();
        let mut src2: crate::fasthash::FastMap<u32, usize> = Default::default();
        for (idx, (cop, _)) in chain.iter().enumerate() {
            let (pops, pushes): (usize, usize) = match *cop {
                ChainOp::ConstNum(_) | ChainOp::LoadName(_) => (0, 1),
                ChainOp::Load(_) => (0, 1),
                ChainOp::Update(_, k) => (
                    0,
                    !matches!(k, UpdKind::IncDiscard | UpdKind::DecDiscard) as usize,
                ),
                ChainOp::GetElem(_) => (1, 1),
                ChainOp::SetElem(_, keep) => (2, keep as usize),
                ChainOp::Arith(_) | ChainOp::Bit(_) => (2, 1),
                ChainOp::Neg => (1, 1),
                ChainOp::Store(_) | ChainOp::Pop => (1, 0),
                ChainOp::Dup => (0, 1),
                ChainOp::KeyNop => (0, 0),
                ChainOp::CmpBranch(..) => (2, 0),
                ChainOp::LoadProp(..) | ChainOp::StoreProp(..) => {
                    unreachable!("loop discovery never admits property operations")
                }
            };
            let mut popped: Vec<usize> = Vec::new();
            for _ in 0..pops {
                popped.push(stack2.pop().expect("residency stack"));
            }
            match *cop {
                ChainOp::Load(off) => {
                    let nd = bind2
                        .get(&off)
                        .copied()
                        .unwrap_or_else(|| *src2.entry(off).or_insert(idx));
                    stack2.push(nd);
                }
                ChainOp::Update(off, kind) => {
                    let cur = bind2.get(&off).copied().or(src2.get(&off).copied());
                    src2.entry(off).or_insert(idx);
                    bind2.insert(off, idx);
                    if pushes == 1 {
                        match kind {
                            UpdKind::PostInc | UpdKind::PostDec => stack2.push(cur.unwrap_or(idx)),
                            _ => stack2.push(idx),
                        }
                    }
                }
                ChainOp::Store(off) => {
                    store_nodes.entry(off).or_default().push(popped[0]);
                    bind2.insert(off, popped[0]);
                }
                ChainOp::Dup => {
                    let v = *stack2.last().expect("residency stack");
                    stack2.push(v);
                }
                ChainOp::SetElem(_, true) => stack2.push(popped[0]),
                _ => {
                    for _ in 0..pushes {
                        stack2.push(idx);
                    }
                }
            }
        }
    }
    for &off in &slot_offs {
        if i_slots.contains(&off) {
            continue;
        }
        let preloaded = first_access.get(&off).copied().unwrap_or(false);
        let is_stored = stored.contains(&off);
        if preloaded && !is_stored {
            if slot_src.get(&off).is_some_and(|&nd| needs_int[nd]) {
                int_checked.push(off);
            }
            continue;
        }
        if !is_stored {
            continue;
        }
        let carried = preloaded; // read before stored: loop-carried accumulator
        let bit_fed = store_nodes.get(&off).is_some_and(|nodes| {
            nodes.iter().any(|&nd| {
                consumers[nd]
                    .iter()
                    .any(|u| matches!(u, Use::Bit | Use::Key))
            })
        });
        if carried || bit_fed {
            i_slots.push(off);
        }
    }

    // ---- kind simulation (multiple rounds: residency demotions can change kinds, and the
    // loop-carried exponent bounds of int-resident slots need a cross-iteration fixed point)
    let mut plan_kinds: Vec<PushKind> = Vec::new();
    let mut i_peak = 0usize;
    let mut d_peak = 0usize;
    // Bit-operand kinds per (chain idx, side), from the final round (conversion memos below).
    let mut bit_kinds: crate::fasthash::FastMap<(usize, u8), PushKind> = Default::default();
    // Per SetElem chain idx: stored value proven exact-i32 (final round).
    let mut setelem_i32: crate::fasthash::FastMap<usize, bool> = Default::default();
    // Loop-head |value| ≤ 2^exp bound per int-resident slot: entry guards prove 31; stores
    // widen it; iterate until stable (or the slot demotes to float residency).
    let mut slot_exp_head: crate::fasthash::FastMap<u32, u32> = Default::default();
    for &off in &i_slots {
        slot_exp_head.insert(off, 31);
    }
    // One precise widening per slot; a second jumps past the int cap so the slot demotes and
    // the rounds terminate (a slot can otherwise creep +1 per round forever).
    let mut widened: Vec<u32> = Vec::new();
    #[allow(unused_assignments)]
    let mut stable = false;
    // Integer registers available to chains: x2..x8 plus x0/x1 — nothing in a chain fast path
    // calls out or scratches them (helpers only run on bail/exit stubs, after the flush).
    const I_UNIVERSE: [u32; 9] = [2, 3, 4, 5, 6, 7, 8, 0, 1];
    let use_count = |off: u32, chain: &[(ChainOp, usize)]| {
        chain
            .iter()
            .filter(|(c, _)| {
                matches!(*c, ChainOp::Load(o) | ChainOp::Update(o, _) | ChainOp::Store(o) if o == off)
            })
            .count()
    };
    // Whether an int-kind duplicate element read exists (it would want an x pin — worth
    // demoting one resident for, at ~20 instructions per iteration saved).
    let want_pin: usize = {
        let mut last: Vec<(u32, usize, bool)> = Vec::new();
        let mut dups = 0usize;
        let mut w = 0usize;
        for (k, &(idx, rcv, key)) in elem_reads.iter().enumerate() {
            while w < elem_writes.len() && elem_writes[w].0 < idx {
                last.clear();
                w += 1;
            }
            if last
                .iter()
                .any(|&(r, kn, wi)| r == rcv && kn == key && wi == elem_int[k])
            {
                if elem_int[k] {
                    dups += 1;
                }
            } else {
                last.push((rcv, key, elem_int[k]));
            }
        }
        dups.min(1)
    };
    let mut pins_demoted = 0usize;
    let pins_wanted = want_pin;
    'budget: loop {
        widened.clear();
        stable = false;
        for _round in 0..64 {
            plan_kinds = vec![PushKind::None; n];
            bit_kinds.clear();
            setelem_i32.clear();
            // (kind, info) per virtual value; slot state per off.
            let mut vstack: Vec<(PushKind, NumInfo)> = Vec::new();
            let mut slot_iv: crate::fasthash::FastMap<u32, NumInfo> = Default::default();
            for &off in &int_checked {
                slot_iv.insert(
                    off,
                    NumInfo {
                        integral: true,
                        exp: 31,
                        neg: true,
                    },
                );
            }
            let mut slot_exp: crate::fasthash::FastMap<u32, u32> = slot_exp_head.clone();
            let mut stored_exp: crate::fasthash::FastMap<u32, u32> = Default::default();
            let mut demote: Option<u32> = None;
            let mut i_live = 0usize;
            let mut d_live = 0usize;
            i_peak = 0;
            d_peak = 0;
            let mut elem_seen = 0usize;
            macro_rules! track {
            ($k:expr, $dir:tt) => {
                match $k {
                    PushKind::I { .. } => i_live = (i_live as isize $dir 1) as usize,
                    PushKind::D { .. } => d_live = (d_live as isize $dir 1) as usize,
                    _ => {}
                }
            };
        }
            for (idx, (cop, _)) in chain.iter().enumerate() {
                let (i_start, d_start) = (i_live, d_live);
                let mut i_pushed = 0usize;
                let mut d_pushed = 0usize;
                macro_rules! push {
                ($k:expr, $inf:expr) => {{
                    let (k, inf) = ($k, $inf);
                    track!(k, +);
                    match k {
                        PushKind::I { .. } => i_pushed += 1,
                        PushKind::D { .. } => d_pushed += 1,
                        _ => {}
                    }
                    plan_kinds[idx] = k;
                    vstack.push((k, inf));
                }};
            }
                macro_rules! pop {
                () => {{
                    let (k, inf) = vstack.pop().expect("loop kind stack");
                    track!(k, -);
                    (k, inf)
                }};
            }
                match *cop {
                    ChainOp::ConstNum(bits) => {
                        let f = f64::from_bits(bits);
                        let integral = f.fract() == 0.0 && f.abs() < 9.0e18;
                        let exp = if integral {
                            (f.abs().max(1.0)).log2().ceil() as u32
                        } else {
                            255
                        };
                        push!(
                            PushKind::K(bits),
                            NumInfo {
                                integral,
                                exp,
                                neg: f < 0.0
                            }
                        );
                    }
                    ChainOp::Load(off) => {
                        if i_slots.contains(&off) {
                            let exp = slot_exp.get(&off).copied().unwrap_or(31);
                            push!(
                                PushKind::I { neg: true },
                                NumInfo {
                                    integral: true,
                                    exp,
                                    neg: true
                                }
                            );
                        } else {
                            let inf = slot_iv.get(&off).copied().unwrap_or(NumInfo::unknown());
                            push!(PushKind::D { iv: inf.iv() }, inf);
                        }
                    }
                    ChainOp::Update(off, kind) => {
                        if !i_slots.contains(&off) {
                            slot_iv.insert(off, NumInfo::unknown());
                        }
                        if !matches!(kind, UpdKind::IncDiscard | UpdKind::DecDiscard) {
                            if i_slots.contains(&off) {
                                push!(
                                    PushKind::I { neg: true },
                                    NumInfo {
                                        integral: true,
                                        exp: 31,
                                        neg: true
                                    }
                                );
                            } else {
                                push!(PushKind::D { iv: false }, NumInfo::unknown());
                            }
                        }
                    }
                    ChainOp::GetElem(_) => {
                        pop!();
                        let want_int = elem_int[elem_seen];
                        elem_seen += 1;
                        if want_int {
                            // The w-form conversion guard proves exact i32.
                            push!(
                                PushKind::I { neg: true },
                                NumInfo {
                                    integral: true,
                                    exp: 31,
                                    neg: true
                                }
                            );
                        } else {
                            push!(PushKind::D { iv: false }, NumInfo::unknown());
                        }
                    }
                    ChainOp::SetElem(_, keep) => {
                        let (vk, vinf) = pop!();
                        pop!();
                        // The abstract bound is |v| <= 2^exp, inclusive. exp==31 therefore
                        // admits positive 2^31, which is not i32: the runtime round-trip must
                        // maintain ALL_I32 in that case. Int kinds cannot carry -0.0.
                        setelem_i32.insert(idx, matches!(vk, PushKind::I { .. }) && vinf.exp < 31);
                        if keep {
                            push!(vk, vinf);
                        }
                    }
                    ChainOp::Arith(f) => {
                        let (bk, binf) = pop!();
                        let (ak, ainf) = pop!();
                        let integral = ainf.integral && binf.integral && f != 3;
                        let exp = match f {
                            0 | 1 => ainf.exp.max(binf.exp).saturating_add(1),
                            2 => ainf.exp.saturating_add(binf.exp),
                            _ => 255,
                        };
                        // Integer lowering: both operands are exact ints in registers (or int
                        // constants) and the result provably fits 2^52, so 64-bit integer add/sub/
                        // mul is exact and equals the f64 result — no guards, 1-cycle latency.
                        let int_side = |k: PushKind, inf: NumInfo| match k {
                            PushKind::I { .. } => true,
                            // -0.0 is "integral" but has no integer representation: its sign would
                            // erase through int arithmetic.
                            PushKind::K(b) => {
                                inf.integral && inf.exp <= 52 && b != (-0.0f64).to_bits()
                            }
                            // Proven-integral f64 (entry-checked preload or tracked store): a bare
                            // fcvtzs is exact (the entry guards reject -0.0).
                            PushKind::D { .. } => inf.integral && inf.exp <= 52,
                            _ => false,
                        };
                        if f != 3 && exp <= 52 && int_side(ak, ainf) && int_side(bk, binf) {
                            let neg = ainf.neg || binf.neg || f == 1;
                            push!(
                                PushKind::I { neg },
                                NumInfo {
                                    integral: true,
                                    exp,
                                    neg
                                }
                            );
                        } else {
                            let inf = NumInfo {
                                integral: integral && exp <= 62,
                                exp,
                                neg: true,
                            };
                            push!(PushKind::D { iv: inf.iv() }, inf);
                        }
                    }
                    ChainOp::Bit(code) => {
                        let (bk, binf) = pop!();
                        let (ak, ainf) = pop!();
                        let _ = binf;
                        bit_kinds.insert((idx, 0), ak);
                        bit_kinds.insert((idx, 1), bk);
                        let kbits = |k: PushKind| match k {
                            PushKind::K(b) => {
                                let f = f64::from_bits(b);
                                if f.fract() == 0.0 && (0.0..2147483648.0).contains(&f) {
                                    Some(f as u32)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        let inf = match code {
                            0 => {
                                // and: a nonneg constant mask bounds the result
                                match kbits(ak).into_iter().chain(kbits(bk)).min() {
                                    Some(m) => NumInfo {
                                        integral: true,
                                        exp: 32 - m.leading_zeros(),
                                        neg: false,
                                    },
                                    None => NumInfo {
                                        integral: true,
                                        exp: 32,
                                        neg: true,
                                    },
                                }
                            }
                            5 => {
                                // shr by a constant: |x >> k| ≤ max(|x| / 2^k, 1) with sign
                                // preserved (after the i32 wrap, so the input bound caps at 31).
                                match kbits(bk) {
                                    Some(k) => {
                                        let e0 = ainf.exp.min(31);
                                        NumInfo {
                                            integral: true,
                                            exp: e0.saturating_sub(k.min(31)).max(1),
                                            neg: if ainf.exp <= 31 { ainf.neg } else { true },
                                        }
                                    }
                                    None => NumInfo {
                                        integral: true,
                                        exp: 32,
                                        neg: true,
                                    },
                                }
                            }
                            3 => {
                                // shl by a constant of a small nonneg value can't wrap
                                match (kbits(bk), ainf.neg) {
                                    (Some(k), false) if ainf.exp + k.min(31) <= 31 => NumInfo {
                                        integral: true,
                                        exp: ainf.exp + k.min(31),
                                        neg: false,
                                    },
                                    _ => NumInfo {
                                        integral: true,
                                        exp: 32,
                                        neg: true,
                                    },
                                }
                            }
                            4 => NumInfo {
                                integral: true,
                                exp: 32,
                                neg: false,
                            },
                            _ => NumInfo {
                                integral: true,
                                exp: 32,
                                neg: true,
                            },
                        };
                        push!(PushKind::I { neg: inf.neg }, inf);
                    }
                    ChainOp::Neg => {
                        let (_, vinf) = pop!();
                        let inf = NumInfo {
                            integral: vinf.integral,
                            exp: vinf.exp,
                            neg: true,
                        };
                        push!(PushKind::D { iv: inf.iv() }, inf);
                    }
                    ChainOp::Store(off) => {
                        let (vk, vinf) = pop!();
                        if i_slots.contains(&off) {
                            // A non-integer store demotes the slot: kinds must be re-simulated.
                            // Counter slots (±1 updates) additionally require i32 stores — the
                            // update sequence relies on the w-form overflow check.
                            let int_ok = match vk {
                                PushKind::I { .. } => true,
                                PushKind::K(b) => {
                                    let f = f64::from_bits(b);
                                    f.fract() == 0.0 && f.abs() < 9.0e15
                                }
                                _ => false,
                            };
                            let exp_cap = if updated.contains(&off) { 31 } else { 52 };
                            if (!int_ok || vinf.exp > exp_cap) && demote.is_none() {
                                demote = Some(off);
                            }
                            slot_exp.insert(off, vinf.exp);
                            let e = stored_exp.entry(off).or_insert(0);
                            *e = (*e).max(vinf.exp);
                        }
                        slot_iv.insert(off, vinf);
                    }
                    ChainOp::Pop => {
                        pop!();
                    }
                    ChainOp::Dup => {
                        let &(vk, vinf) = vstack.last().expect("loop kind stack");
                        push!(vk, vinf);
                    }
                    ChainOp::KeyNop => {}
                    ChainOp::CmpBranch(..) => {
                        pop!();
                        pop!();
                    }
                    ChainOp::LoadName(ptr) => {
                        let inf = if int_checked_names.contains(&ptr) {
                            NumInfo {
                                integral: true,
                                exp: 31,
                                neg: true,
                            }
                        } else {
                            NumInfo::unknown()
                        };
                        push!(PushKind::D { iv: inf.iv() }, inf);
                    }
                    ChainOp::LoadProp(..) | ChainOp::StoreProp(..) => {
                        unreachable!("loop discovery never admits property operations")
                    }
                }
                // Operand registers are freed only at op end, so an op needs its start-of-op
                // live set plus everything it pushes, simultaneously.
                i_peak = i_peak.max(i_live).max(i_start + i_pushed);
                d_peak = d_peak.max(d_live).max(d_start + d_pushed);
            }
            match demote {
                Some(off) => {
                    if env_flag!("TRUST_JS_JIT_LOOPLOG") {
                        eprintln!("[jit-loop] head {head}: demote I slot {}", off / 8);
                    }
                    i_slots.retain(|&o| o != off);
                    slot_exp_head.remove(&off);
                }
                None => {
                    // Widen loop-head exponent bounds with what this round stored; a stable set of
                    // bounds means the kinds are final.
                    let mut changed = false;
                    for (&off, &e) in &stored_exp {
                        if !i_slots.contains(&off) {
                            continue;
                        }
                        let entry = slot_exp_head.entry(off).or_insert(31);
                        let mut new = (*entry).max(e);
                        if new != *entry && widened.contains(&off) {
                            new = 53; // second widening: force the demotion path
                        }
                        if new != *entry {
                            *entry = new;
                            widened.push(off);
                            changed = true;
                        }
                    }
                    if !changed {
                        stable = true;
                        break;
                    }
                }
            }
        }
        if !stable {
            reject!("kind rounds did not converge");
        }
        // Register budget: demote the least-used I resident and re-simulate when over; also
        // give up (a bounded number of) residents so the receiver/memo pins fit — a length
        // pin turns every element access into one load, worth far more than a counter's home.
        let over = i_peak + i_slots.len() > I_UNIVERSE.len();
        let pin_squeeze = pins_demoted < pins_wanted
            && i_peak + i_slots.len() + (pins_wanted - pins_demoted) > I_UNIVERSE.len();
        if over || pin_squeeze {
            let victim = i_slots
                .iter()
                .copied()
                .min_by_key(|&off| use_count(off, &chain));
            match victim {
                Some(v) => {
                    if env_flag!("TRUST_JS_JIT_LOOPLOG") {
                        eprintln!(
                            "[jit-loop] head {head}: demote I slot {} ({})",
                            v / 8,
                            if over { "pressure" } else { "pin" }
                        );
                    }
                    if !over {
                        pins_demoted += 1;
                    }
                    i_slots.retain(|&o| o != v);
                    slot_exp_head.remove(&v);
                    continue 'budget;
                }
                None if over => reject!(format!("i pressure: peak {i_peak}")),
                None => break,
            }
        }
        break;
    }
    if d_peak + 1 > 8 {
        reject!(format!("d pressure: peak {d_peak}"));
    }
    let f_slots: Vec<u32> = slot_offs
        .iter()
        .copied()
        .filter(|o| !i_slots.contains(o))
        .collect();
    if f_slots.len() + names_order.len() > 8 {
        reject!(format!(
            "f pressure: {} slots + {} names",
            f_slots.len(),
            names_order.len()
        ));
    }

    let mut slots: Vec<SlotPlan> = Vec::new();
    let mut next_d = 8u32;
    let mut next_x = 0usize; // index into I_UNIVERSE
    for &off in &slot_offs {
        let res = if i_slots.contains(&off) {
            let r = SlotRes::I(I_UNIVERSE[next_x]);
            next_x += 1;
            r
        } else if f_slots.contains(&off) {
            let r = SlotRes::F(next_d);
            next_d += 1;
            r
        } else {
            SlotRes::None
        };
        let preload = first_access.get(&off).copied().unwrap_or(false);
        let is_stored = stored.contains(&off);
        slots.push(SlotPlan {
            off,
            res,
            preload,
            stored: is_stored,
            virgin: is_stored && !preload,
            int_checked: int_checked.contains(&off),
        });
    }
    // Name homes come from the same d8..d15 bank, after the slot homes.
    let names: Vec<NamePlan> = names_order
        .iter()
        .map(|&ptr| {
            let dreg = next_d;
            next_d += 1;
            NamePlan {
                ptr,
                dreg,
                int_checked: int_checked_names.contains(&ptr),
            }
        })
        .collect();
    // Sanity: kinds recorded for the final residency sets. The last sim round used exactly
    // `i_slots`/all-resident F, matching the assignment above.

    // Which receivers feed int-typed element reads (their mirror mode also needs ALL_I32).
    let mut rcv_int: Vec<u32> = Vec::new();
    {
        let mut seen = 0usize;
        for (cop, _) in &chain {
            if let ChainOp::GetElem(off) = *cop {
                if elem_int[seen] && !rcv_int.contains(&off) {
                    rcv_int.push(off);
                }
                seen += 1;
            }
        }
    }

    // ---- memoization: duplicate element reads and repeated guarded ToInt32 conversions.
    // Node ids are SSA-like (an id never changes value), so a second element read with the same
    // (receiver, key id) — with no intervening element write — and a second Bit-op use of the
    // same unproven-f64 id can reuse the first result from a pinned register. Pins live in the
    // leftover resident registers; memos are dropped when none are free.
    // x pins: whatever the universe leaves after I residents and the transient reserve; d pins
    // from the resident bank's leftovers (d transients live in d16.. and never collide).
    // Pin pool: the caller-saved leftovers, then x23-x27 (x28 is the poll countdown).
    // These are callee-saved — using any obliges
    // the loop to bracket itself with save/restore pairs, a fixed ~6-instruction cost per
    // loop ENTRY against one shaved load per element access per iteration). Pops take the
    // caller-saved ones first.
    let mut free_pin_x: Vec<u32> = [27u32, 26, 25, 24, 23]
        .into_iter()
        .chain(
            I_UNIVERSE
                .iter()
                .copied()
                .filter(|x| !slots.iter().any(|s| s.res == SlotRes::I(*x)))
                .skip(i_peak),
        )
        .collect();
    let mut free_pin_d: Vec<u32> = (next_d..16).collect();
    // Receivers 1-2 take x16/x17; 3-4 draw from the pin pool BEFORE the memo pins (a receiver
    // base is worth more than a memo — it carries every element access on that array). What
    // the pool still has after that pins the per-receiver vector fields, most-used first:
    // mirror length and data (every access), then elems/entries data (stores only).
    let mut rplans: Vec<ReceiverPlan> = Vec::new();
    let written: Vec<u32> = elem_writes.iter().map(|&(_, off)| off).collect::<Vec<_>>();
    for (k, &off) in receivers.iter().enumerate() {
        let reg = if k < 2 {
            16 + k as u32
        } else {
            match free_pin_x.pop() {
                Some(r) => r,
                None => reject!("too many receivers for the pin pool"),
            }
        };
        rplans.push(ReceiverPlan {
            off,
            reg,
            mirror: fast & 262144 != 0,
            int_reads: rcv_int.contains(&off),
            recheck_i32: rcv_int.contains(&off) && setelem_i32.values().any(|&exact| !exact),
            mlreg: None,
            mpreg: None,
            elpreg: None,
            enreg: None,
        });
    }
    if fast & 262144 != 0 {
        // Heaviest-accessed receiver first, and its FULL pin set before the next receiver
        // gets any: in the lin_solve shape one array carries 5 of 6 accesses per iteration —
        // splitting pins evenly left its mirror data pointer reloading every access.
        let weight = |off: u32| {
            elem_reads.iter().filter(|&&(_, r, _)| r == off).count()
                + elem_writes.iter().filter(|&&(_, r)| r == off).count()
        };
        let mut order: Vec<usize> = (0..rplans.len()).collect();
        order.sort_by_key(|&k| std::cmp::Reverse(weight(rplans[k].off)));
        for k in order {
            let rp = &mut rplans[k];
            rp.mlreg = free_pin_x.pop();
            rp.mpreg = free_pin_x.pop();
            if written.contains(&rp.off) {
                rp.elpreg = free_pin_x.pop();
                rp.enreg = free_pin_x.pop();
            }
        }
    }
    let receivers = rplans;
    let mut elem_retain: Vec<(usize, u32)> = Vec::new();
    let mut elem_reuse: Vec<(usize, usize)> = Vec::new(); // (dup idx, retain idx)
    {
        // (rcv, key node, want-int) → retain chain idx
        let mut last: Vec<((u32, usize, bool), usize)> = Vec::new();
        let mut w = 0usize;
        for (k, &(idx, rcv, key)) in elem_reads.iter().enumerate() {
            // Any element write invalidates every pending read: two receiver slots can hold the
            // same array at runtime, so same-receiver screening would be unsound.
            while w < elem_writes.len() && elem_writes[w].0 < idx {
                last.clear();
                w += 1;
            }
            let want = elem_int[k];
            match last
                .iter()
                .find(|((r, kn, wi), _)| *r == rcv && *kn == key && *wi == want)
            {
                Some(&(_, ridx)) => elem_reuse.push((idx, ridx)),
                None => last.push(((rcv, key, want), idx)),
            }
        }
        // Only reads that are actually reused get pins.
        for &(_, ridx) in &elem_reuse {
            if !elem_retain.iter().any(|(i, _)| *i == ridx) {
                let k = elem_reads.iter().position(|&(i, _, _)| i == ridx).unwrap();
                let pin = if elem_int[k] {
                    free_pin_x.pop()
                } else {
                    free_pin_d.pop()
                };
                if let Some(r) = pin {
                    elem_retain.push((ridx, r));
                }
            }
        }
        // Drop reuses whose retain got no pin.
        elem_reuse.retain(|&(_, ridx)| elem_retain.iter().any(|(i, _)| *i == ridx));
    }
    let mut conv_retain: Vec<((usize, u8), u32)> = Vec::new();
    let mut conv_reuse: Vec<((usize, u8), u32)> = Vec::new();
    {
        // Guarded conversions only (D with iv=false): the 7-instruction guard is worth a pin.
        let mut by_id: crate::fasthash::FastMap<usize, Vec<(usize, u8)>> = Default::default();
        for &(idx, aid, bid) in &bit_uses {
            for (side, id) in [(0u8, aid), (1u8, bid)] {
                if matches!(bit_kinds.get(&(idx, side)), Some(PushKind::D { iv: false })) {
                    by_id.entry(id).or_default().push((idx, side));
                }
            }
        }
        let mut ids: Vec<(usize, Vec<(usize, u8)>)> =
            by_id.into_iter().filter(|(_, v)| v.len() >= 2).collect();
        ids.sort_by_key(|(id, _)| *id);
        for (_, mut uses) in ids {
            let Some(pin) = free_pin_x.pop() else { break };
            uses.sort();
            conv_retain.push((uses[0], pin));
            for &u in &uses[1..] {
                conv_reuse.push((u, pin));
            }
        }
    }

    if env_flag!("TRUST_JS_JIT_LOOPLOG") {
        let vec_pins: usize = receivers
            .iter()
            .map(|r| {
                [r.mlreg, r.mpreg, r.elpreg, r.enreg]
                    .iter()
                    .filter(|p| p.is_some())
                    .count()
            })
            .sum();
        eprintln!(
            "[jit-loop] head {head}: CHAINED {} ops, {} slots ({} I), {} receivers ({} vec pins), {} names, memo elem {}r/{}u conv {}r/{}u",
            chain.len(),
            slots.len(),
            slots
                .iter()
                .filter(|s| matches!(s.res, SlotRes::I(_)))
                .count(),
            receivers.len(),
            vec_pins,
            names.len(),
            elem_retain.len(),
            elem_reuse.len(),
            conv_retain.len(),
            conv_reuse.len()
        );
    }
    let uses_ext = receivers.iter().any(|r| {
        r.reg >= 23
            || [r.mlreg, r.mpreg, r.elpreg, r.enreg]
                .iter()
                .flatten()
                .any(|&x| x >= 23)
    }) || elem_retain.iter().any(|&(_, p)| p >= 23)
        || conv_retain.iter().any(|&(_, p)| p >= 23);
    Some(LoopPlan {
        head,
        exit_pc,
        initialization_guards,
        chain,
        cond_len,
        kinds: plan_kinds,
        slots,
        receivers,
        elem_retain,
        elem_reuse,
        conv_retain,
        conv_reuse,
        setelem_i32,
        names,
        uses_ext,
    })
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
/// A virtual value during loop-chain emission.
#[derive(Clone, Copy)]
enum LV {
    K(u64),
    I(u32, bool), // x-register, may-be-negative
    D(u32, bool), // d-register, integral-valued
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_region_own_entry(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    rc_reg: u32,
    body_reg: u32,
    entry_reg: u32,
    state: crate::bytecode::IcState,
    writable: bool,
    fail: usize,
) {
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let plain = layout.obj_ic_plain as u32;
    let shape = (layout.obj_props + layout.props_shape) as u32;
    a.add_imm(body_reg, rc_reg, obj_off);
    a.ldrb_imm(9, body_reg, ex);
    guard_exotic_ordinary(a, layout, 9, 9, fail);
    a.ldrb_imm(9, body_reg, plain);
    a.cbz(9, false, fail);
    a.ldr_w_imm(9, body_reg, shape);
    a.mov_imm64(16, state.recv_shape as u64);
    a.cmp_reg_w(9, 16);
    a.b_cond(C_NE, fail);
    fields_len(a, layout, 16, body_reg);
    a.mov_imm64(13, state.slot as u64);
    a.cmp_reg_x(13, 16);
    a.b_cond(C_HS, fail);
    fields_data(a, layout, entry_reg, body_reg);
    a.mov_imm64(16, layout.entry_size as u64);
    a.madd(entry_reg, 13, 16, entry_reg);
    guard_prop_data(a, 9, entry_reg, layout.entry_accessor as u32, fail);
    if writable {
        guard_prop_writable(a, 9, entry_reg, layout.entry_writable as u32, fail);
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_linked_scan_materialize(a: &mut asm::Asm, plan: &LinkedScanPlan, peek_object: bool) {
    // x0 is a borrowed current-node pointer: it becomes the one or two frame words naming the
    // node, which overwrite the old `next` and `peek` words.
    a.mov_imm64(9, crate::value::PACK_OBJ);
    a.logic_x(1, 9, 9, 0);
    a.str_imm(9, 22, plan.next_off);
    if peek_object {
        a.str_imm(9, 22, plan.peek_off);
    } else {
        a.mov_imm64(9, crate::value::PACK_NULL);
        a.str_imm(9, 22, plan.peek_off);
    }
}

/// Linked-list SSA region.  The initial `next` frame owner roots the complete property chain, so
/// x0 can walk borrowed packed object pointers without a clone/drop pair at every node.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_linked_scan_region(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    plan: &LinkedScanPlan,
    pc_labels: &[usize],
) -> usize {
    let plain_h = a.new_label();
    let body = a.new_label();
    let fail = a.new_label();
    let found_null = a.new_label();

    // `next`, the walk's first node, must be an Object; the baseline path remains untouched on
    // rejection. The old `peek` word is simply overwritten.
    a.ldr_imm(9, 22, plan.next_off);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 10, plain_h);
    emit_exec_payload(a, 9, 3);

    a.mov(0, 3); // borrowed current node
    a.movz(8, 0, 0); // no successful assignment yet
    a.bind(body);
    emit_region_own_entry(a, layout, 0, 1, 2, plan.link, false, fail);
    a.ldur(13, 2, layout.entry_value as i32);
    a.mov_imm64(14, crate::value::PACK_NULL);
    a.cmp_reg_x(13, 14);
    a.b_cond(C_EQ, found_null);
    a.lsr_imm(9, 13, 48);
    a.movz(14, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(9, 14);
    a.b_cond(C_NE, fail);
    a.lsl_imm(15, 13, 16);
    a.lsr_imm(15, 15, 16);
    if plan.loose_null_compare {
        // Loose equality has the one object/null exception: an HTMLDDA object compares equal to
        // null.  `ic_plain` is false for that object (and for other exotic receivers), so replay
        // the comparison in the baseline before committing `next = peek`.  Keep x0 unchanged
        // until after this guard so the materialized loop-head state remains exact.
        a.add_imm(14, 15, layout.obj_from_cell as u32);
        a.ldrb_imm(9, 14, layout.obj_ic_plain as u32);
        a.cbz(9, false, fail);
    }
    a.mov(0, 15);
    a.movz(8, 1, 0);
    emit_region_poll_guard(a, ilayout, fail, false);
    a.b(body);

    a.bind(found_null);
    emit_linked_scan_materialize(a, plan, false);
    a.b(pc_labels[plan.exit_pc]);

    a.bind(fail);
    a.cbz(8, false, plain_h); // first iteration: the original frame is still canonical
    emit_linked_scan_materialize(a, plan, true);
    a.b(plain_h); // later failure: resume baseline at the loop head with materialized state
    plain_h
}

#[cfg(all(test, target_arch = "aarch64"))]
thread_local! {
    static TEST_NUMERIC_REGION_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, target_arch = "aarch64"))]
extern "C" fn record_numeric_region_entry() {
    TEST_NUMERIC_REGION_ENTRIES.with(|count| count.set(count.get() + 1));
}

/// Emit the loop chain for `plan`. Returns the label for the plain fallback of the head op —
/// the caller binds it immediately after and continues emitting the plain region. Every emitted
/// interior fallback destination is marked in `targeted` before plain-template fusion begins.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_loop_chain(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    plan: &LoopPlan,
    pc_labels: &[usize],
    targeted: &mut [bool],
) -> usize {
    // A helper-free region can keep its poll divider in a register, writing the
    // shared tick at every exit. Unsupported layouts retain canonical polling.
    if !ilayout.valid
        || !ilayout.interrupt_poll_tick.is_multiple_of(4)
        || ilayout.interrupt_poll_tick / 4 >= 4096
    {
        return a.new_label();
    }
    let obj_off = layout.obj_from_cell as u32;
    let ex = layout.obj_exotic as u32;
    let el = (layout.obj_props + layout.props_elems) as u32;
    let evp = (layout.dense_elems + layout.vec_ptr_off) as u32;
    let evl = (layout.dense_elems + layout.vec_len_off) as u32;
    let mvp = (layout.dense_mirror + layout.vec_ptr_off) as u32;
    let mvl = (layout.dense_mirror + layout.vec_len_off) as u32;
    let ev = layout.entry_value as i32;
    let num_ev = if layout.entry_accessor == layout.entry_value + 8 {
        ev
    } else {
        ev + 8
    };
    let ea = layout.entry_accessor as u32;
    let ew = layout.entry_writable as u32;
    let es = layout.entry_size as u64;
    let plain = layout.obj_ic_plain as u32;
    let mf = (layout.obj_props + layout.props_mirror_flags) as u32;
    let mirror = el;

    let plain_h = a.new_label();
    let body_l = a.new_label();
    let exit_a = a.new_label();
    let exit_b = a.new_label();
    let poll_bail = a.new_label();
    // x23-x28 bracket (see LoopPlan::uses_ext): saved before ANY preamble step (receiver bases
    // may live in ext registers), reloaded on every path out — preamble failures route through
    // `pre_fail`, exits and bails emit the reload inline.
    let pre_fail = a.new_label();
    if plan.uses_ext {
        a.stp_pre(23, 24, -48);
        a.stp_off(25, 26, 16);
        a.stp_off(27, 28, 32);
    } else {
        a.stp_pre(28, 31, -16);
    }
    let restore_ext = |a: &mut asm::Asm| {
        if plan.uses_ext {
            a.ldp_off(25, 26, 16);
            a.ldp_off(27, 28, 32);
            a.ldp_post(23, 24, 48);
        } else {
            a.ldp_post(28, 31, 16);
        }
    };

    let slot = |off: u32| plan.slots.iter().find(|s| s.off == off);
    let rcv_plan = |off: u32| plan.receivers.iter().find(|r| r.off == off);
    // Virgins stored within the condition prefix (they flush even on the entry exit).
    let cond_virgins: Vec<u32> = plan.chain[..plan.cond_len]
        .iter()
        .filter_map(|(c, _)| match *c {
            ChainOp::Store(off) if slot(off).is_some_and(|s| s.virgin) => Some(off),
            _ => None,
        })
        .collect();
    let all_virgins: Vec<u32> = plan
        .slots
        .iter()
        .filter(|s| s.virgin)
        .map(|s| s.off)
        .collect();

    // ---- preamble --------------------------------------------------------------------------
    // Hoist only the non-throwing guard, never the observable TDZ error. On Empty, return to
    // the original loop before its condition/RHS: it may be zero-trip or throw from the RHS
    // first. Once admitted, no operation in the helper-free loop can uninitialize a local.
    for off in &plan.initialization_guards {
        a.ldr_imm(9, 22, *off);
        a.mov_imm64(10, crate::value::PACK_EMPTY);
        a.cmp_reg_x(9, 10);
        a.b_cond(C_EQ, pre_fail);
    }
    // Validate names before populating integer local homes: the shared IC probe uses x7 as its
    // packed/wide result marker in addition to x9-x17. Large regions can legitimately assign
    // a resident local to x7, so reversing this order would silently overwrite that local.
    for np in &plan.names {
        emit_name_ic_value_ptr(a, layout, np.ptr, pre_fail, true);
        let loaded = a.new_label();
        if layout.entry_accessor == layout.entry_value + 8 {
            let wide = a.new_label();
            a.cbz(7, false, wide);
            a.ldur(9, 14, 0);
            a.lsr_imm(10, 9, 48);
            let number = a.new_label();
            a.movz(11, (crate::value::PACK_OBJ >> 48) as u32, 0);
            a.cmp_reg_x(10, 11);
            a.b_cond(C_HS, pre_fail);
            a.movz(11, (crate::value::PACK_UNDEFINED >> 48) as u32, 0);
            a.cmp_reg_x(10, 11);
            a.b_cond(C_LO, number);
            a.movz(11, (crate::value::PACK_SYM >> 48) as u32, 0);
            a.cmp_reg_x(10, 11);
            a.b_cond(C_LS, pre_fail);
            a.bind(number);
            a.fmov_d_x(np.dreg, 9);
            a.b(loaded);
            a.bind(wide);
        }
        a.ldurb(9, 14, 0);
        a.cmp_imm_w(9, 4);
        a.b_cond(C_NE, pre_fail); // only a Num can live in a register
        a.ldur_d(np.dreg, 14, 8);
        a.bind(loaded);
        if np.int_checked {
            a.fcvtzs_w_d(9, np.dreg);
            a.scvtf_d_w(1, 9);
            a.fmov_x_d(10, 1);
            a.fmov_x_d(11, np.dreg);
            a.cmp_reg_x(10, 11);
            a.b_cond(C_NE, pre_fail);
        }
    }
    for s in &plan.slots {
        if !s.preload {
            continue;
        }
        a.ldr_imm(9, 22, s.off);
        emit_exec_number_guard(a, 9, 0, 10, pre_fail);
        match s.res {
            SlotRes::F(d) => {
                a.ldr_d_imm(d, 22, s.off);
                if s.int_checked {
                    // One-time exact-i32 proof (bit-compare: -0.0 must not pass); the value
                    // stays in its d home and integer consumers convert with a bare fcvtzs.
                    a.fcvtzs_w_d(9, d);
                    a.scvtf_d_w(1, 9);
                    a.fmov_x_d(10, 1);
                    a.fmov_x_d(11, d);
                    a.cmp_reg_x(10, 11);
                    a.b_cond(C_NE, pre_fail);
                }
            }
            SlotRes::I(x) => {
                // Exact i32 (w-form conversion + compare-back): counters keep the invariant
                // with a flag-setting ±1, and the planner's range analysis starts from 2^31.
                a.ldr_d_imm(0, 22, s.off);
                a.fcvtzs_w_d(x, 0);
                a.scvtf_d_w(1, x);
                a.fmov_x_d(9, 1);
                a.fmov_x_d(10, 0);
                a.cmp_reg_x(9, 10);
                a.b_cond(C_NE, pre_fail);
                a.sxtw(x, x);
            }
            SlotRes::None => {}
        }
    }
    // Receiver bases come last because the name probe also clobbers x16/x17.
    for rp in &plan.receivers {
        let (off, r) = (rp.off, rp.reg);
        a.ldr_imm(10, 22, off);
        emit_exec_tag_guard(a, 10, crate::value::PACK_OBJ, 12, pre_fail);
        emit_exec_payload(a, 10, 10);
        a.add_imm(r, 10, obj_off);
        a.ldrb_imm(12, r, ex);
        guard_exotic_ordinary_or_array(a, layout, 12, 13, pre_fail);
        a.ldrb_imm(12, r, plain);
        a.cbz(12, false, pre_fail);
        if rp.mirror {
            // The element buffer must be coherent, hole-free, and (for int-read receivers)
            // all-i32: element reads become bounds + one indexed load with no tag check.
            let mut need = (crate::value::MIRROR_OK | crate::value::MIRROR_NO_HOLES) as u32;
            if rp.int_reads {
                need |= crate::value::MIRROR_ALL_I32 as u32;
            }
            a.ldrb_imm(12, r, mf);
            a.mov(13, 12);
            let field = asm::logical_imm_w(need).expect("mirror mask encodable");
            a.logic_imm_w(0, 12, 12, field);
            a.cmp_imm_w(12, need);
            if packed_elem_inlinable(layout) {
                let ready = a.new_label();
                a.b_cond(C_EQ, ready);
                // Existing coherent non-i32 mirrors cannot satisfy an int-read proof. An
                // earlier failed packed preparation must not rescan on every slow iteration.
                for bit in [crate::value::MIRROR_OK, crate::value::MIRROR_PACKED_FAILED] {
                    a.logic_imm_w(0, 12, 13, asm::logical_imm_w(bit as u32).unwrap());
                    a.cbnz(12, false, pre_fail);
                }
                a.ldr_imm(12, r, el);
                a.cbz(12, true, pre_fail);
                emit_packed_elements_base(a, layout, pre_fail);
                // Cold representation preparation preserves all already established integer
                // homes/receiver pins. d8..d15 are preserved by the C ABI; no transient FP
                // operand exists in this preamble. This helper cannot execute JS or GC.
                a.stp_pre(0, 1, -144);
                for register in (2..18).step_by(2) {
                    a.stp_off(register, register + 1, register as i32 * 8);
                }
                // The helper takes `Gc::as_ptr`: the cell address, `obj_off` bytes before the
                // Object.
                a.sub_imm(0, r, obj_off);
                a.mov_imm64(16, jit_prepare_packed_numeric_mirror as *const () as u64);
                a.blr(16);
                for register in (2..18).step_by(2) {
                    a.ldp_off(register, register + 1, register as i32 * 8);
                }
                a.ldp_post(0, 1, 144);
                a.ldrb_imm(12, r, mf);
                a.logic_imm_w(0, 12, 12, field);
                a.cmp_imm_w(12, need);
                a.b_cond(C_NE, pre_fail);
                a.bind(ready);
            } else {
                a.b_cond(C_NE, pre_fail);
            }
        }
        // Pinned vector fields (stable for the whole region — helper-free vocabulary, and slim
        // stores never grow or reallocate).
        if rp.mlreg.is_some() || rp.mpreg.is_some() {
            a.ldr_imm(12, r, mirror);
            a.cbz(12, true, pre_fail);
            if let Some(x) = rp.mlreg {
                a.ldr_imm(x, 12, mvl);
            }
            if let Some(x) = rp.mpreg {
                a.ldr_imm(x, 12, mvp);
            }
        }
        if let Some(x) = rp.elpreg {
            a.ldr_imm(12, r, el);
            a.cbz(12, true, pre_fail);
            a.ldr_imm(x, 12, evp);
        }
        if let Some(x) = rp.enreg {
            fields_data(a, layout, x, r);
        }
    }

    #[cfg(test)]
    {
        // Runtime evidence after every numeric/receiver guard, not a compilation counter.
        // This test-only observer does not run JS or GC. Preserve all volatile homes;
        // generated release code has no observer, storage, or additional instructions.
        a.stp_pre(0, 1, -400);
        for register in (2..18).step_by(2) {
            a.stp_off(register, register + 1, register as i32 * 8);
        }
        for register in (0..32).step_by(2) {
            a.stp_d_off(register, register + 1, 144 + register as i32 * 8);
        }
        a.mov_imm64(16, record_numeric_region_entry as *const () as u64);
        a.blr(16);
        for register in (0..32).step_by(2) {
            a.ldp_d_off(register, register + 1, 144 + register as i32 * 8);
        }
        for register in (2..18).step_by(2) {
            a.ldp_off(register, register + 1, register as i32 * 8);
        }
        a.ldp_post(0, 1, 400);
    }

    // ---- emission state --------------------------------------------------------------------
    // (chain idx, bail label, vstack snapshot, virgins stored at that point)
    let mut bails: Vec<(usize, usize, Vec<LV>, Vec<u32>)> = Vec::new();
    let mut vstack: Vec<LV> = Vec::new();
    let pinned = |x: u32| {
        plan.elem_retain.iter().any(|&(_, p)| p == x)
            || plan.conv_retain.iter().any(|&(_, p)| p == x)
    };
    let mut free_i: Vec<u32> = [1u32, 0, 8, 7, 6, 5, 4, 3, 2]
        .into_iter()
        .filter(|x| {
            !plan.slots.iter().any(|s| s.res == SlotRes::I(*x))
                && !pinned(*x)
                && !plan.receivers.iter().any(|r| {
                    r.reg == *x || [r.mlreg, r.mpreg, r.elpreg, r.enreg].contains(&Some(*x))
                })
        })
        .collect();
    let mut free_d: Vec<u32> = (16..24).rev().collect();
    // Loads push ALIASES of resident registers (zero-copy: consumers never clobber their
    // operands — Arith/Neg/Bit write fresh destinations). The pools only take back their own:
    // a freed alias of an I/F home or a name home silently stays out.
    let pool_i: Vec<u32> = free_i.clone();
    let is_pool_i = |x: u32| pool_i.contains(&x);
    let is_pool_d = |d: u32| (16..24).contains(&d);

    macro_rules! emit_pass {
        ($range:expr, $exit:expr, $base_virgins:expr) => {{
            let mut stores_seen: Vec<u32> = $base_virgins;
            for idx in $range {
                let (ref cop, _) = plan.chain[idx];
                let bail = a.new_label();
                #[allow(unused_assignments)]
                let mut used = false;
                let snap = vstack.clone();
                let seen_snap = stores_seen.clone();
                // Operand registers freed by this op return to the pools only once the op has
                // emitted its last guard — a bail spills the pre-op snapshot, so no operand
                // register may be reused (and clobbered) while a guard can still fire.
                let mut dead: Vec<LV> = Vec::new();
                macro_rules! guard {
                    () => {{
                        #[allow(unused_assignments)]
                        {
                            used = true;
                        }
                        bail
                    }};
                }
                // Convert helpers ------------------------------------------------------------
                macro_rules! to_w {
                    // Value into a w-usable scratch gpr; returns the register number.
                    ($v:expr, $scr:expr) => {{
                        match $v {
                            LV::I(x, _) => x,
                            LV::K(bits) => {
                                let iv = f64::from_bits(bits) as i64;
                                a.mov_imm64($scr, iv as u64);
                                $scr
                            }
                            LV::D(d, iv) => {
                                a.fcvtzs_x_d($scr, d);
                                if !iv {
                                    a.scvtf_d_x(0, $scr);
                                    a.frintz(1, d);
                                    a.fcmp(0, 1);
                                    a.b_cond(C_NE, guard!());
                                    a.cmn_imm_x($scr, 1);
                                    a.b_cond(C_VS, guard!());
                                }
                                $scr
                            }
                        }
                    }};
                }
                macro_rules! free_v {
                    ($v:expr) => {
                        dead.push($v)
                    };
                }
                // Materialize any live vstack alias of a resident register about to be
                // overwritten (an Update/Store to its slot): the pushed value must keep the
                // OLD contents. Runs after the pre-op snapshot (bails read the still-unmutated
                // resident) and before the mutation.
                macro_rules! flush_aliases {
                    ($home:expr, $is_f:expr) => {{
                        for k in 0..vstack.len() {
                            match vstack[k] {
                                LV::D(d, iv) if $is_f && d == $home => {
                                    let dt = free_d.pop().expect("loop d pool");
                                    a.fmov_d_d(dt, d);
                                    vstack[k] = LV::D(dt, iv);
                                }
                                LV::I(x, ng) if !$is_f && x == $home => {
                                    let xt = free_i.pop().expect("loop i pool");
                                    a.mov(xt, x);
                                    vstack[k] = LV::I(xt, ng);
                                }
                                _ => {}
                            }
                        }
                    }};
                }
                macro_rules! to_d {
                    // Value into a d-register; the original register is deferred-freed, so the
                    // caller owns the result only if the source was already D.
                    ($v:expr) => {{
                        match $v {
                            LV::D(d, _) => d,
                            LV::I(x, _) => {
                                let d = free_d.pop().expect("loop d pool");
                                a.scvtf_d_x(d, x);
                                dead.push(LV::I(x, false));
                                d
                            }
                            LV::K(bits) => {
                                let d = free_d.pop().expect("loop d pool");
                                a.mov_imm64(9, bits);
                                a.fmov_d_x(d, 9);
                                d
                            }
                        }
                    }};
                }
                macro_rules! key_to_x9 {
                    ($v:expr) => {
                        match $v {
                            LV::I(x, _neg) => {
                                // No explicit negative check: LV::I is a sign-extended exact
                                // i32, and every consumer's FIRST use of x9 is an unsigned
                                // bounds compare against a vector length — a negative reads
                                // as ≥ 2^63 and takes the same bail the explicit check did.
                                a.mov(9, x);
                                dead.push(LV::I(x, false));
                            }
                            LV::K(bits) => {
                                let f = f64::from_bits(bits);
                                if f.fract() == 0.0 && (0.0..2147483648.0).contains(&f) {
                                    a.mov_imm64(9, f as u64);
                                } else {
                                    a.mov_imm64(9, bits);
                                    a.fmov_d_x(0, 9);
                                    a.fcvtzu_w_d(9, 0);
                                    a.ucvtf_d_w(1, 9);
                                    a.fcmp(0, 1);
                                    a.b_cond(C_NE, guard!());
                                }
                            }
                            LV::D(d, _) => {
                                a.fcvtzu_w_d(9, d);
                                a.ucvtf_d_w(0, 9);
                                a.fcmp(d, 0);
                                a.b_cond(C_NE, guard!());
                                dead.push(LV::D(d, false));
                            }
                        }
                    };
                }
                // Element lookup: key index in x9, receiver base in `r` → entry pointer in x15.
                macro_rules! elem_entry {
                    ($r:expr) => {{
                        a.ldr_imm(12, $r, el);
                        a.cbz(12, true, guard!());
                        a.ldr_imm(14, 12, evl);
                        a.cmp_reg_x(9, 14);
                        a.b_cond(C_HS, guard!());
                        a.ldr_imm(12, 12, evp);
                        a.add_shifted(12, 12, 9, 2);
                        a.ldr_w_imm(13, 12, 0);
                        a.cmn_imm_w(13, 1);
                        a.b_cond(C_EQ, guard!());
                        fields_data(a, layout, 15, $r);
                        a.movz(9, es as u32, 0);
                        a.madd(15, 13, 9, 15);
                        guard_prop_data(a, 9, 15, ea, guard!());
                    }};
                }

                match *cop {
                    ChainOp::ConstNum(bits) => vstack.push(LV::K(bits)),
                    ChainOp::Load(off) => {
                        let s = slot(off).expect("planned slot");
                        match s.res {
                            SlotRes::F(dres) => {
                                // Zero-copy alias of the home (see the pool filter).
                                let iv = matches!(plan.kinds[idx], PushKind::D { iv: true });
                                vstack.push(LV::D(dres, iv));
                            }
                            SlotRes::I(xres) => {
                                vstack.push(LV::I(xres, true));
                            }
                            SlotRes::None => {
                                a.ldr_imm(9, 22, off);
                                emit_exec_number_guard(a, 9, 0, 10, guard!());
                                let dt = free_d.pop().expect("loop d pool");
                                a.ldr_d_imm(dt, 22, off);
                                let iv = matches!(plan.kinds[idx], PushKind::D { iv: true });
                                vstack.push(LV::D(dt, iv));
                            }
                        }
                    }
                    ChainOp::Update(off, kind) => {
                        let s = slot(off).expect("planned slot");
                        match s.res {
                            SlotRes::F(d) => flush_aliases!(d, true),
                            SlotRes::I(x) => flush_aliases!(x, false),
                            SlotRes::None => {}
                        }
                        let dec = matches!(
                            kind,
                            UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
                        );
                        match s.res {
                            SlotRes::I(xres) => {
                                // The entry guard proved exact i32; a flag-setting w-form ±1
                                // keeps it (V = left i32 = bail), far from f64's 2^53 edge.
                                // The guard fires before any mutation, so the sign-extend can
                                // land straight in the resident.
                                if dec {
                                    a.subs_imm_w(9, xres, 1);
                                } else {
                                    a.adds_imm_w(9, xres, 1);
                                }
                                a.b_cond(C_VS, guard!());
                                match kind {
                                    UpdKind::PostInc | UpdKind::PostDec => {
                                        let xt = free_i.pop().expect("loop i pool");
                                        a.mov(xt, xres);
                                        a.sxtw(xres, 9);
                                        vstack.push(LV::I(xt, true));
                                    }
                                    UpdKind::PreInc | UpdKind::PreDec => {
                                        a.sxtw(xres, 9);
                                        vstack.push(LV::I(xres, true));
                                    }
                                    _ => a.sxtw(xres, 9),
                                }
                            }
                            SlotRes::F(dres) => {
                                let f = if dec { 1 } else { 0 };
                                a.fmov_one(0);
                                match kind {
                                    UpdKind::PostInc | UpdKind::PostDec => {
                                        let dt = free_d.pop().expect("loop d pool");
                                        a.fmov_d_d(dt, dres);
                                        a.f_arith(f, dres, dres, 0);
                                        vstack.push(LV::D(dt, false));
                                    }
                                    UpdKind::PreInc | UpdKind::PreDec => {
                                        a.f_arith(f, dres, dres, 0);
                                        let dt = free_d.pop().expect("loop d pool");
                                        a.fmov_d_d(dt, dres);
                                        vstack.push(LV::D(dt, false));
                                    }
                                    _ => a.f_arith(f, dres, dres, 0),
                                }
                            }
                            SlotRes::None => {
                                a.ldr_imm(9, 22, off);
                                emit_exec_number_guard(a, 9, 0, 10, guard!());
                                let f = if dec { 1 } else { 0 };
                                a.ldr_d_imm(0, 22, off);
                                a.fmov_one(1);
                                a.f_arith(f, 1, 0, 1);
                                emit_exec_number_store(a, 1, 22, off as i32, 9);
                                match kind {
                                    UpdKind::PostInc | UpdKind::PostDec => {
                                        let dt = free_d.pop().expect("loop d pool");
                                        a.fmov_d_d(dt, 0);
                                        vstack.push(LV::D(dt, false));
                                    }
                                    UpdKind::PreInc | UpdKind::PreDec => {
                                        let dt = free_d.pop().expect("loop d pool");
                                        a.fmov_d_d(dt, 1);
                                        vstack.push(LV::D(dt, false));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    ChainOp::GetElem(xoff) => {
                        let key = vstack.pop().expect("loop vstack");
                        // A read the planner proved identical to an earlier one (same receiver,
                        // same key value, no element write between) copies the pinned result —
                        // its guards already passed this iteration.
                        if let Some(&(_, ridx)) = plan.elem_reuse.iter().find(|&&(d, _)| d == idx) {
                            let pin = plan
                                .elem_retain
                                .iter()
                                .find(|&&(i, _)| i == ridx)
                                .expect("planned retain")
                                .1;
                            free_v!(key);
                            if matches!(plan.kinds[idx], PushKind::I { .. }) {
                                let xt = free_i.pop().expect("loop i pool");
                                a.mov(xt, pin);
                                vstack.push(LV::I(xt, true));
                            } else {
                                let dt = free_d.pop().expect("loop d pool");
                                a.fmov_d_d(dt, pin);
                                vstack.push(LV::D(dt, false));
                            }
                        } else {
                            key_to_x9!(key);
                            let rp = rcv_plan(xoff).expect("planned receiver");
                            let pin = plan
                                .elem_retain
                                .iter()
                                .find(|&&(i, _)| i == idx)
                                .map(|p| p.1);
                            if rp.mirror {
                                // Mirror: bounds + one indexed load. Preamble proved coherent
                                // + hole-free (+ all-i32 for int reads): no tag check, and int
                                // reads need no exactness guard. Pinned length/data registers
                                // shave the two dependent loads when the planner had room.
                                match rp.mlreg {
                                    Some(x) => a.cmp_reg_x(9, x),
                                    None => {
                                        a.ldr_imm(12, rp.reg, mirror);
                                        a.cbz(12, true, guard!());
                                        a.ldr_imm(14, 12, mvl);
                                        a.cmp_reg_x(9, 14);
                                    }
                                }
                                a.b_cond(C_HS, guard!());
                                let mpr = match rp.mpreg {
                                    Some(x) => x,
                                    None => {
                                        a.ldr_imm(12, rp.reg, mirror);
                                        a.cbz(12, true, guard!());
                                        a.ldr_imm(14, 12, mvp);
                                        14
                                    }
                                };
                                if matches!(plan.kinds[idx], PushKind::I { .. }) {
                                    if rp.recheck_i32 {
                                        // A previous numeric write may target this same object
                                        // through any receiver local. Guard before consuming
                                        // this read; earlier stores remain committed on bailout.
                                        a.ldrb_imm(12, rp.reg, mf);
                                        a.logic_imm_w(
                                            0,
                                            12,
                                            12,
                                            asm::logical_imm_w(crate::value::MIRROR_ALL_I32 as u32)
                                                .unwrap(),
                                        );
                                        a.cbz(12, false, guard!());
                                    }
                                    a.ldr_d_lsl3(0, mpr, 9);
                                    let xt = free_i.pop().expect("loop i pool");
                                    a.fcvtzs_w_d(xt, 0);
                                    a.sxtw(xt, xt);
                                    if let Some(p) = pin {
                                        a.mov(p, xt);
                                    }
                                    vstack.push(LV::I(xt, true));
                                } else {
                                    let dt = free_d.pop().expect("loop d pool");
                                    a.ldr_d_lsl3(dt, mpr, 9);
                                    if let Some(p) = pin {
                                        a.fmov_d_d(p, dt);
                                    }
                                    vstack.push(LV::D(dt, false));
                                }
                            } else {
                                let r = rp.reg;
                                if layout.entry_accessor == layout.entry_value + 8 {
                                    a.b(guard!());
                                }
                                elem_entry!(r);
                                a.ldrb_imm(9, 15, ev as u32);
                                a.cmp_imm_w(9, 4);
                                a.b_cond(C_NE, guard!());
                                if matches!(plan.kinds[idx], PushKind::I { .. }) {
                                    // w-form: the exactness compare-back also proves i32 (the
                                    // planner's range analysis relies on that bound).
                                    a.ldur_d(0, 15, ev + 8);
                                    let xt = free_i.pop().expect("loop i pool");
                                    a.fcvtzs_w_d(xt, 0);
                                    a.scvtf_d_w(1, xt);
                                    // Bit-compare, not fcmp: IEEE equality would accept -0.0
                                    // and erase its sign through the int-typed value.
                                    a.fmov_x_d(9, 1);
                                    a.fmov_x_d(10, 0);
                                    a.cmp_reg_x(9, 10);
                                    a.b_cond(C_NE, guard!());
                                    a.sxtw(xt, xt);
                                    if let Some(p) = pin {
                                        a.mov(p, xt);
                                    }
                                    vstack.push(LV::I(xt, true));
                                } else {
                                    let dt = free_d.pop().expect("loop d pool");
                                    a.ldur_d(dt, 15, ev + 8);
                                    if let Some(p) = pin {
                                        a.fmov_d_d(p, dt);
                                    }
                                    vstack.push(LV::D(dt, false));
                                }
                            }
                        }
                    }
                    ChainOp::SetElem(xoff, keep) => {
                        let val = vstack.pop().expect("loop vstack");
                        let key = vstack.pop().expect("loop vstack");
                        // Stage the value into d2 before the key conversion (d0/d1 scratch).
                        match val {
                            LV::D(d, _) => a.fmov_d_d(2, d),
                            LV::I(x, _) => a.scvtf_d_x(2, x),
                            LV::K(bits) => {
                                a.mov_imm64(9, bits);
                                a.fmov_d_x(2, 9);
                            }
                        }
                        key_to_x9!(key);
                        let rp = rcv_plan(xoff).expect("planned receiver");
                        let r = rp.reg;
                        let i32_proven = plan.setelem_i32.get(&idx).copied().unwrap_or(false);
                        if rp.mirror {
                            // Mirror invariant (preamble-proven): every mirrored element is a
                            // plain writable data Num — the accessor/writable/old-value checks
                            // and the tag write all collapse. A hole (elems NO_SLOT) bails:
                            // that store would CREATE a property. Pinned vector registers
                            // shave the four dependent loads when the planner had room.
                            match rp.mlreg {
                                Some(x) => a.cmp_reg_x(9, x),
                                None => {
                                    a.ldr_imm(12, r, mirror);
                                    a.cbz(12, true, guard!());
                                    a.ldr_imm(14, 12, mvl);
                                    a.cmp_reg_x(9, 14);
                                }
                            }
                            a.b_cond(C_HS, guard!());
                            let canonical_stored = a.new_label();
                            if packed_elem_inlinable(layout) {
                                let classic_store = a.new_label();
                                a.ldrb_imm(12, r, mf);
                                a.logic_imm_w(
                                    0,
                                    12,
                                    12,
                                    asm::logical_imm_w(crate::value::MIRROR_PACKED as u32).unwrap(),
                                );
                                a.cbz(12, false, classic_store);
                                a.ldr_imm(12, r, el);
                                emit_packed_elements_base(a, layout, guard!());
                                a.add_shifted(15, 15, 9, 4);
                                emit_exec_number_store(a, 2, 15, layout.property_value as i32, 14);
                                a.b(canonical_stored);
                                a.bind(classic_store);
                            }
                            let elpr = match rp.elpreg {
                                Some(x) => x,
                                None => {
                                    a.ldr_imm(12, r, el);
                                    a.cbz(12, true, guard!());
                                    a.ldr_imm(14, 12, evp);
                                    14
                                }
                            };
                            a.add_shifted(12, elpr, 9, 2);
                            a.ldr_w_imm(13, 12, 0);
                            a.cmn_imm_w(13, 1);
                            a.b_cond(C_EQ, guard!());
                            let enr = match rp.enreg {
                                Some(x) => x,
                                None => {
                                    fields_data(a, layout, 15, r);
                                    15
                                }
                            };
                            a.movz(12, es as u32, 0);
                            a.madd(15, 13, 12, enr);
                            if layout.entry_accessor == layout.entry_value + 8 {
                                emit_exec_number_store(a, 2, 15, num_ev, 14);
                            } else {
                                a.stur_d(2, 15, num_ev);
                            }
                            a.bind(canonical_stored);
                            match rp.mpreg {
                                Some(x) => a.str_d_lsl3(2, x, 9),
                                None => {
                                    a.ldr_imm(12, r, mirror);
                                    a.ldr_imm(12, 12, mvp);
                                    a.str_d_lsl3(2, 12, 9);
                                }
                            }
                            if !i32_proven {
                                // MIRROR_ALL_I32 upkeep, flag-first (no sentinel screen —
                                // hole accounting is structural, see Props::mirror_sync).
                                let i32_done = a.new_label();
                                a.ldrb_imm(13, r, mf);
                                let i32_bit =
                                    asm::logical_imm_w(crate::value::MIRROR_ALL_I32 as u32)
                                        .unwrap();
                                a.logic_imm_w(0, 12, 13, i32_bit);
                                a.cbz(12, false, i32_done);
                                a.fcvtzs_w_d(12, 2);
                                a.scvtf_d_w(1, 12);
                                a.fmov_x_d(12, 1);
                                a.fmov_x_d(14, 2);
                                a.cmp_reg_x(12, 14);
                                a.b_cond(C_EQ, i32_done);
                                let clear =
                                    asm::logical_imm_w(!(crate::value::MIRROR_ALL_I32 as u32))
                                        .unwrap();
                                a.logic_imm_w(0, 13, 13, clear);
                                a.strb_imm(13, r, mf);
                                a.bind(i32_done);
                            }
                        } else {
                            if layout.entry_accessor == layout.entry_value + 8 {
                                a.b(guard!());
                            }
                            elem_entry!(r);
                            guard_prop_writable(a, 9, 15, ew, guard!());
                            a.movz(9, 4, 0);
                            a.stur(9, 15, ev);
                            a.stur_d(2, 15, ev + 8);
                            // Element mirror: the value was staged in d2; key registers are
                            // still intact (operand frees are deferred to op end).
                            let mkey = match key {
                                LV::I(x, _) => MirrorKey::U32InReg(x),
                                LV::D(d, _) => MirrorKey::F64InDreg(d),
                                // A K key reaching the commit passed the exact-u32 runtime
                                // check, so the compile-time conversion is exact.
                                LV::K(bits) => MirrorKey::Const(f64::from_bits(bits) as u32),
                            };
                            emit_mirror_store(a, layout, r, mkey, MirrorVal::Num(2, i32_proven));
                        }
                        if keep {
                            vstack.push(val);
                        } else {
                            free_v!(val);
                        }
                    }
                    ChainOp::Arith(f) => {
                        let b = vstack.pop().expect("loop vstack");
                        let a_ = vstack.pop().expect("loop vstack");
                        if let PushKind::I { neg } = plan.kinds[idx] {
                            // Range-proven exact integer arithmetic: no guards needed.
                            let to_x = |a: &mut asm::Asm, v: LV, scr: u32| match v {
                                LV::I(x, _) => x,
                                LV::K(bits) => {
                                    a.mov_imm64(scr, f64::from_bits(bits) as i64 as u64);
                                    scr
                                }
                                // Planner-proven integral: exact without a guard.
                                LV::D(d, _) => {
                                    a.fcvtzs_x_d(scr, d);
                                    scr
                                }
                            };
                            let xb = to_x(a, b, 10);
                            let xa = to_x(a, a_, 9);
                            let xt = free_i.pop().expect("loop i pool");
                            match f {
                                0 => a.add_shifted(xt, xa, xb, 0),
                                1 => a.sub_reg(xt, xa, xb),
                                _ => a.madd(xt, xa, xb, 31),
                            }
                            free_v!(a_);
                            free_v!(b);
                            vstack.push(LV::I(xt, neg));
                        } else {
                            // Fresh destination: operands may be zero-copy aliases of resident
                            // registers (f_arith is 3-operand, so this costs nothing).
                            let db = to_d!(b);
                            let da = to_d!(a_);
                            let dt = free_d.pop().expect("loop d pool");
                            a.f_arith(f, dt, da, db);
                            dead.push(LV::D(da, false));
                            dead.push(LV::D(db, false));
                            let iv = matches!(plan.kinds[idx], PushKind::D { iv: true });
                            vstack.push(LV::D(dt, iv));
                        }
                    }
                    ChainOp::Bit(code) => {
                        let b = vstack.pop().expect("loop vstack");
                        let a_ = vstack.pop().expect("loop vstack");
                        let neg = matches!(plan.kinds[idx], PushKind::I { neg: true });
                        // A guarded ToInt32 the planner proved repeats an earlier one reuses the
                        // pinned result; the first instance converts into its pin.
                        macro_rules! conv {
                            ($v:expr, $side:expr, $scr:expr) => {{
                                let reuse = plan
                                    .conv_reuse
                                    .iter()
                                    .find(|&&((i, s), _)| i == idx && s == $side)
                                    .map(|p| p.1);
                                match (reuse, $v) {
                                    // The operand register is untouched; the arm's free_v!
                                    // releases it at op end like any other operand.
                                    (Some(pin), LV::D(..)) => pin,
                                    _ => {
                                        let scr = plan
                                            .conv_retain
                                            .iter()
                                            .find(|&&((i, s), _)| i == idx && s == $side)
                                            .map(|p| p.1)
                                            .unwrap_or($scr);
                                        to_w!($v, scr)
                                    }
                                }
                            }};
                        }
                        // Immediate forms when the rhs is a suitable constant.
                        let imm = match b {
                            LV::K(bits) => {
                                let f = f64::from_bits(bits);
                                if f.fract() == 0.0 && (0.0..4294967296.0).contains(&f) {
                                    Some(f as u32)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        let enc = imm.and_then(|m| match code {
                            0..=2 => asm::logical_imm_w(m),
                            _ => Some(m & 31),
                        });
                        let xt;
                        if let Some(field) = enc {
                            let wa = conv!(a_, 0, 9);
                            xt = free_i.pop().expect("loop i pool");
                            match code {
                                0..=2 => a.logic_imm_w(code, xt, wa, field),
                                3 => a.lsl_imm_w(xt, wa, field),
                                4 => a.lsr_imm_w(xt, wa, field),
                                _ => a.asr_imm_w(xt, wa, field),
                            }
                            free_v!(a_);
                        } else {
                            let wb = conv!(b, 1, 10);
                            let wa = conv!(a_, 0, 9);
                            xt = free_i.pop().expect("loop i pool");
                            match code {
                                0 => a.logic_w(0, xt, wa, wb),
                                1 => a.logic_w(1, xt, wa, wb),
                                2 => a.logic_w(2, xt, wa, wb),
                                3 => a.shift_w(0, xt, wa, wb),
                                4 => a.shift_w(1, xt, wa, wb),
                                _ => a.shift_w(2, xt, wa, wb),
                            }
                            free_v!(a_);
                            free_v!(b);
                        }
                        if neg {
                            a.sxtw(xt, xt);
                        }
                        vstack.push(LV::I(xt, neg));
                    }
                    ChainOp::Neg => {
                        let v = vstack.pop().expect("loop vstack");
                        let d = to_d!(v);
                        let dt = free_d.pop().expect("loop d pool");
                        a.fneg(dt, d);
                        dead.push(LV::D(d, false));
                        let iv = matches!(plan.kinds[idx], PushKind::D { iv: true });
                        vstack.push(LV::D(dt, iv));
                    }
                    ChainOp::Store(off) => {
                        let v = vstack.pop().expect("loop vstack");
                        let s = slot(off).expect("planned slot");
                        match s.res {
                            SlotRes::F(d) => flush_aliases!(d, true),
                            SlotRes::I(x) => flush_aliases!(x, false),
                            SlotRes::None => {}
                        }
                        match s.res {
                            SlotRes::F(dres) => match v {
                                LV::D(d, _) => {
                                    a.fmov_d_d(dres, d);
                                    dead.push(LV::D(d, false));
                                }
                                LV::I(x, _) => {
                                    a.scvtf_d_x(dres, x);
                                    dead.push(LV::I(x, false));
                                }
                                LV::K(bits) => {
                                    a.mov_imm64(9, bits);
                                    a.fmov_d_x(dres, 9);
                                }
                            },
                            SlotRes::I(xres) => match v {
                                LV::I(x, _) => {
                                    a.mov(xres, x);
                                    dead.push(LV::I(x, false));
                                }
                                LV::K(bits) => {
                                    let f = f64::from_bits(bits);
                                    a.mov_imm64(xres, f as i64 as u64);
                                }
                                LV::D(..) => unreachable!("planner demotes float-stored I slots"),
                            },
                            SlotRes::None => {
                                let dv = to_d!(v);
                                // The overwrite is the old word's release.
                                emit_exec_number_store(a, dv, 22, off as i32, 9);
                                dead.push(LV::D(dv, false));
                            }
                        }
                        if s.virgin && !stores_seen.contains(&off) {
                            stores_seen.push(off);
                        }
                    }
                    ChainOp::Pop => {
                        let v = vstack.pop().expect("loop vstack");
                        free_v!(v);
                    }
                    ChainOp::Dup => {
                        let v = *vstack.last().expect("loop vstack");
                        match v {
                            LV::K(bits) => vstack.push(LV::K(bits)),
                            // Aliases duplicate for free (nothing clobbers them; the pool
                            // filter blocks their double-free). Owned temps still copy — the
                            // two entries free independently.
                            LV::I(x, neg) => {
                                if is_pool_i(x) {
                                    let xt = free_i.pop().expect("loop i pool");
                                    a.mov(xt, x);
                                    vstack.push(LV::I(xt, neg));
                                } else {
                                    vstack.push(LV::I(x, neg));
                                }
                            }
                            LV::D(d, iv) => {
                                if is_pool_d(d) {
                                    let dt = free_d.pop().expect("loop d pool");
                                    a.fmov_d_d(dt, d);
                                    vstack.push(LV::D(dt, iv));
                                } else {
                                    vstack.push(LV::D(d, iv));
                                }
                            }
                        }
                    }
                    ChainOp::KeyNop => {}
                    ChainOp::CmpBranch(neg, _) => {
                        let b = vstack.pop().expect("loop vstack");
                        let a_ = vstack.pop().expect("loop vstack");
                        let k_imm12 = |v: LV| match v {
                            LV::K(bits) => {
                                let f = f64::from_bits(bits);
                                if f.fract() == 0.0 && (0.0..4096.0).contains(&f) {
                                    Some(f as u32)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        let int_neg = match neg {
                            5 => 10, // !(a<b) → GE
                            8 => 12, // !(a<=b) → GT
                            n => n,  // LE/LT/NE/EQ hold for signed ints
                        };
                        match (a_, b) {
                            (LV::I(xa, _), LV::I(xb, _)) => {
                                a.cmp_reg_x(xa, xb);
                                a.b_cond(int_neg, $exit);
                                dead.push(LV::I(xa, false));
                                dead.push(LV::I(xb, false));
                            }
                            (LV::I(xa, _), kb) if k_imm12(kb).is_some() => {
                                a.cmp_imm_x(xa, k_imm12(kb).unwrap());
                                a.b_cond(int_neg, $exit);
                                dead.push(LV::I(xa, false));
                            }
                            // One side exact-int in a register, the other a PROVEN-integral
                            // f64 (an int-checked name/preload): a bare x-form fcvtzs is exact,
                            // so the compare stays integer — the loop-head `i < width` pattern,
                            // otherwise a per-iteration scvtf + fcmp on the branch path.
                            (LV::I(xa, _), LV::D(db, true)) => {
                                a.fcvtzs_x_d(9, db);
                                a.cmp_reg_x(xa, 9);
                                a.b_cond(int_neg, $exit);
                                dead.push(LV::I(xa, false));
                                dead.push(LV::D(db, false));
                            }
                            (LV::D(da, true), LV::I(xb, _)) => {
                                a.fcvtzs_x_d(9, da);
                                a.cmp_reg_x(9, xb);
                                a.b_cond(int_neg, $exit);
                                dead.push(LV::D(da, false));
                                dead.push(LV::I(xb, false));
                            }
                            (a2, LV::K(bits)) if f64::from_bits(bits) == 0.0 => {
                                let da = to_d!(a2);
                                a.fcmp_zero(da);
                                a.b_cond(neg, $exit);
                                dead.push(LV::D(da, false));
                            }
                            (a2, b2) => {
                                let db = to_d!(b2);
                                let da = to_d!(a2);
                                a.fcmp(da, db);
                                a.b_cond(neg, $exit);
                                dead.push(LV::D(da, false));
                                dead.push(LV::D(db, false));
                            }
                        }
                    }
                    ChainOp::LoadName(ptr) => {
                        // Preamble-pinned, never written in-region: a zero-copy alias.
                        let np = plan
                            .names
                            .iter()
                            .find(|n| n.ptr == ptr)
                            .expect("planned name");
                        let iv = matches!(plan.kinds[idx], PushKind::D { iv: true });
                        vstack.push(LV::D(np.dreg, iv));
                    }
                    ChainOp::LoadProp(..) | ChainOp::StoreProp(..) => {
                        unreachable!("loop discovery never admits property operations")
                    }
                }
                if used {
                    bails.push((idx, bail, snap, seen_snap));
                }
                for v in dead {
                    match v {
                        LV::I(x, _) if is_pool_i(x) => free_i.push(x),
                        LV::D(d, _) if is_pool_d(d) => free_d.push(d),
                        _ => {}
                    }
                }
            }
        }};
    }

    // ---- rotated loop ----------------------------------------------------------------------
    emit_region_poll_start(a, ilayout, 28);
    emit_pass!(0..plan.cond_len, exit_a, Vec::new());
    // Keep hot-loop placement stable when cold preamble checks grow. The override is for
    // release A/B diagnostics, not a semantic switch; 0 disables padding. Large-branch
    // relaxation can still move this boundary, so correctness never depends on alignment.
    let alignment = env_value!("TRUST_JS_JIT_LOOP_ALIGNMENT")
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value == 0 || (16..=128).contains(value) && value.is_power_of_two())
        .unwrap_or(64);
    if alignment != 0 {
        a.align_hot(alignment);
    }
    a.bind(body_l);
    emit_pass!(
        plan.cond_len..plan.chain.len(),
        exit_b,
        cond_virgins.clone()
    );
    a.subs_imm_w(28, 28, 1);
    a.b_cond(C_EQ, poll_bail);
    emit_pass!(0..plan.cond_len, exit_b, all_virgins.clone());
    a.b(body_l);

    // ---- exits and bails -------------------------------------------------------------------
    let emit_flush = |a: &mut asm::Asm, virgins: &[u32]| {
        for s in &plan.slots {
            if !s.stored {
                continue;
            }
            if s.virgin && !virgins.contains(&s.off) {
                continue;
            }
            let d = match s.res {
                SlotRes::F(d) => d,
                SlotRes::I(x) => {
                    a.scvtf_d_x(0, x);
                    0
                }
                SlotRes::None => continue, // stores wrote through
            };
            emit_exec_number_store(a, d, 22, s.off as i32, 9);
        }
    };
    a.bind(exit_a);
    emit_flush(a, &cond_virgins);
    emit_region_poll_finish(a, ilayout, 28);
    restore_ext(a);
    a.b(pc_labels[plan.exit_pc]);
    a.bind(exit_b);
    emit_flush(a, &all_virgins);
    emit_region_poll_finish(a, ilayout, 28);
    restore_ext(a);
    a.b(pc_labels[plan.exit_pc]);
    a.bind(poll_bail);
    emit_flush(a, &all_virgins);
    // Leave the due increment to the canonical header, with every root materialized.
    a.movz(28, 1, 0);
    emit_region_poll_finish(a, ilayout, 28);
    restore_ext(a);
    a.b(pc_labels[plan.head]);
    a.bind(pre_fail);
    restore_ext(a);
    a.b(plain_h);

    for (idx, label, snap, seen) in bails {
        a.bind(label);
        for v in &snap {
            match *v {
                LV::K(bits) => {
                    a.mov_imm64(
                        9,
                        if f64::from_bits(bits).is_nan() {
                            crate::value::PACK_CANON_NAN
                        } else {
                            bits
                        },
                    );
                    a.stur(9, 20, 0);
                }
                LV::I(x, _) => {
                    a.scvtf_d_x(0, x);
                    a.stur_d(0, 20, 0);
                }
                LV::D(d, _) => {
                    emit_exec_number_store(a, d, 20, 0, 9);
                }
            }
            a.add_imm(20, 20, 8);
        }
        emit_flush(a, &seen);
        emit_region_poll_finish(a, ilayout, 28);
        restore_ext(a);
        let pc = plan.chain[idx].1;
        if pc == plan.head {
            a.b(plain_h);
        } else {
            targeted[pc] = true;
            a.b(pc_labels[pc]);
        }
    }
    plain_h
}

/// The generic per-op helper call: `jit_exec(ctx, pc, sp)` → (new sp, threw?). The sp is taken
/// unconditionally — it reflects consumed operands even when the op threw, which is what keeps
/// the unwinder's cleanup from re-dropping moved-out slots.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_exec(a: &mut asm::Asm, pc: u32, l_unwind: usize) {
    emit_op_helper(a, H_EXEC, pc, l_unwind);
}

/// [`emit_exec`] through a DEDICATED helper slot (same `(ctx, pc, sp) → SpFlag` contract):
/// hot op families skip the generic decode.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_op_helper(a: &mut asm::Asm, idx: usize, pc: u32, l_unwind: usize) {
    a.mark_push("helper-call");
    emit_op_helper_marked(a, idx, pc, l_unwind);
    a.mark_pop();
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_op_helper_marked(a: &mut asm::Asm, idx: usize, pc: u32, l_unwind: usize) {
    if idx == H_EXEC {
        // A chain's replay ladder enters with canonical owners before this original read.
        // General CFG emitters also flush all homes at candidate reads before calling here.
        emit_string_accumulation(a, pc, true, l_unwind);
    }
    if a.stubs_ok() && shared_stubs_enabled() && idx < N_HELPERS {
        // A slow path runs rarely and costs far more than a call and return: the sequence
        // lives once in a shared stub, leaving three instructions at every site.
        a.movz(1, pc, 0);
        let stub = a.shared_stub(SharedStub::Helper { idx: idx as u8 }.key());
        a.bl_label(stub);
        a.cbnz(1, false, l_unwind);
        if helper_ends_in_poll(idx) {
            emit_helper_end_poll(a, l_unwind);
        }
        return;
    }
    a.mov(0, 19);
    a.movz(1, pc, 0);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (idx * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
    a.cbnz(1, false, l_unwind);
    if helper_ends_in_poll(idx) {
        emit_helper_end_poll(a, l_unwind);
    }
}

/// Whether a call to helper `idx` ends in the explicit poll, in either backend: ARM64's
/// `emit_op_helper` (see `emit_helper_end_poll`) and x64's `Asm::helper_spflag` (see
/// `x64::Asm::helper_end_poll`). The census of helper families:
/// - Polled after their call: the ones that can allocate cells without reaching a safepoint
///   (`gc_check`, `gc_check_amortized` or a call into JavaScript) before they return, or that
///   reach one only before allocating: `exec` (generic operations), `make_object`,
///   `make_array`, `make_regexp`, `make_closure`, `object_literal`, `abstract_operation`,
///   `iteration`, `add_strings`, `get_prop`, `set_prop`, `get_method_elem`/`get_elem`,
///   `set_elem`, `instanceof` and `reference_op` (natives, host accessors, lazy prototypes,
///   key and string conversions). Calls (`call`, `call_hit`, `intrinsic`, direct calls) poll
///   once at their site's end, `new` after its own sequence and `string_accumulation` on its
///   committed exit, outside this table.
/// - Not polled: `cond`, `return`, `push_handler`, `pop_handler`, `strict_eq` and
///   `loop_backedge` allocate no cell; `unwind` and `complete` route a completion already made
///   (their ABIs return a code address); `direct_finish` runs inside a direct call, whose site
///   polls at its end (a tail call goes through `gc_check_amortized` first); `slice_op` ends the
///   native slice, and the code that resumes or receives it polls; `interrupt` is the poll.
fn helper_ends_in_poll(idx: usize) -> bool {
    matches!(
        idx,
        H_EXEC
            | H_MAKE_OBJECT
            | H_MAKE_ARRAY
            | H_MAKE_REGEXP
            | H_MAKE_CLOSURE
            | H_OBJECT_LITERAL
            | H_ABSTRACT
            | H_ITERATION
            | H_ADD_STRINGS
            | H_GET_PROP
            | H_SET_PROP
            | H_GET_ELEM
            | H_SET_ELEM
            | H_INSTANCEOF
            | H_REFERENCE_OP
    )
}

/// The explicit poll that ends a call to an allocating helper in a chunk body's
/// baseline code (the recorded S5 deviation: allocation never collects, I9, and these polls
/// take the place of "the allocation slow path may collect"). It runs once the helper has
/// returned: the helper's results then sit in canonical places (operand words below the
/// returned sp, slots, context fields) and nothing it produced lives elsewhere, since baseline
/// code keeps no value in a register across a helper call. Region code, whose checked effects
/// run with virtual operands and private homes, and shared stubs, which region sites share,
/// emit none (see `Asm::set_end_polls`): a region collects at its canonical header.
///
/// Three instructions while the heap's due word is clear (the context's count pointer, the
/// word, a CBZ). A set word calls `H_INTERRUPT` (`POLL_HELPER_END`), whose `gc_check` collects
/// and so clears it; a throw from there (the live-object limit's RangeError, a host interrupt)
/// unwinds from here. The object count is left to the entry and loop polls.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_helper_end_poll(a: &mut asm::Asm, l_unwind: usize) {
    #[cfg(test)]
    if TEST_NO_END_POLLS.with(std::cell::Cell::get) {
        return;
    }
    if !a.end_polls() {
        return;
    }
    let done = a.new_label();
    a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, live_objects) as u32);
    a.ldr_imm(9, 9, crate::value::LIVE_DUE_OFF);
    a.cbz(9, true, done);
    emit_op_helper(a, H_INTERRUPT, POLL_HELPER_END, l_unwind);
    a.bind(done);
}

/// Try only primitive RHS strings. Miss preserves the canonical state and continues the
/// original operation; success commits the entire region then takes its original continuation
/// (in baseline code, after the helper-end poll; a throw from that poll goes to `l_unwind`).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_string_accumulation(a: &mut asm::Asm, pc: u32, replay: bool, l_unwind: usize) {
    #[cfg(test)]
    if !replay && crate::bytecode::string_accumulation::TEST_REPLAY_ONLY.with(std::cell::Cell::get)
    {
        return;
    }
    #[cfg(not(test))]
    let _ = replay;
    let Some((slot, exit)) = a.string_accumulation(pc) else {
        return;
    };
    let ordinary = a.new_label();
    emit_exec_word_load(a, 9, 22, slot as i32 * 8);
    a.lsr_imm(10, 9, 48);
    a.movz(11, (crate::value::PACK_STR >> 48) as u32, 0);
    a.cmp_reg_x(10, 11);
    a.b_cond(C_NE, ordinary);
    emit_helper(a, H_STRING_ACCUMULATION, pc);
    if a.end_polls() {
        a.cbnz(1, false, ordinary); // an unchanged miss
        emit_helper_end_poll(a, l_unwind);
        a.b(exit);
        a.bind(ordinary);
        return;
    }
    a.cbz(1, false, exit);
    a.bind(ordinary);
}

/// `H_INTERRUPT`'s immediate: which poll calls it. A loop header's poll (where a region's poll
/// bail lands too) passes `POLL_LOOP`; a function entry's passes `POLL_ENTRY` and
/// the end of an allocating helper's call `POLL_HELPER_END`, so tests can count the collections
/// each kind runs.
pub(crate) const POLL_LOOP: u32 = 0;
pub(crate) const POLL_ENTRY: u32 = 1;
pub(crate) const POLL_HELPER_END: u32 = 2;

/// Test-only: what generated code's `H_INTERRUPT` calls on this thread did, indexed by
/// the caller's `POLL_*` kind: the calls, the collections they ran, and the most bytes the heap
/// had allocated past its trigger when a call collected.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PollCensus {
    pub(crate) calls: [u64; 3],
    pub(crate) collections: [u64; 3],
    pub(crate) max_overshoot: usize,
}

#[cfg(test)]
thread_local! {
    static POLL_CENSUS: std::cell::Cell<PollCensus> = const {
        std::cell::Cell::new(PollCensus {
            calls: [0; 3],
            collections: [0; 3],
            max_overshoot: 0,
        })
    };
    /// Test-only switch: chunks compiled while it is set end no helper call in a poll (see
    /// `emit_helper_end_poll`), so a test can watch the entry and loop polls alone.
    pub(crate) static TEST_NO_END_POLLS: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

/// The poll census so far on this thread, which then starts over (tests).
#[cfg(test)]
pub(crate) fn take_poll_census() -> PollCensus {
    POLL_CENSUS.with(|census| census.replace(PollCensus::default()))
}

/// `jit_interrupt`'s record of one call (tests): its caller, whether it collected, and the
/// heap's allocated bytes past the trigger before it ran.
#[cfg(test)]
pub(crate) fn record_poll(kind: u32, collected: bool, overshoot: usize) {
    POLL_CENSUS.with(|census| {
        let mut current = census.get();
        let kind = kind as usize;
        current.calls[kind] += 1;
        if collected {
            current.collections[kind] += 1;
            current.max_overshoot = current.max_overshoot.max(overshoot);
        }
        census.set(current);
    });
}

/// Load the live-object count into x9 from its address in x9 (`JitCtx::live_objects`). The
/// cell heap's due word follows the count (`value::LiveCount`): one LDP loads both (the word
/// into x10) and a set word takes `slow`, so the poll sees the heap's byte budget, stress
/// interval and deferred requests, none of which moves the count. x10 is clobbered, which both
/// polls reload with `gc_next` next.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_live_count(a: &mut asm::Asm, slow: usize) {
    const _: () = assert!(crate::value::LIVE_DUE_OFF == 8);
    a.ldp_base(9, 10, 9, 0);
    a.cbnz(10, true, slow);
}

/// Generated-loop safepoint. Allocation pressure must not wait for the interrupt divider:
/// call-free loops can allocate cycles, and deferring optional task collection does not
/// defer the live-object ceiling. The cold helper sees canonical owned frame roots.
///
/// The fast path loads `ctx.interp`, the count and the heap's due word ([`emit_live_count`]),
/// compares the count with `gc_next`, then bumps the shared interrupt divider (12
/// instructions); the slow path is `H_INTERRUPT` (`jit_interrupt`: the host interrupt, then
/// `Interp::gc_check`), taken when the count passes `gc_next`, every 16384 ticks and while the
/// heap's due word is set. There `gc_check` collects, which clears the word,
/// unless collections are suppressed or deferred (a no-GC region, an outstanding borrow, an
/// interpreter that does not own its heap): then the word stays set and every poll calls the
/// helper until one can collect. `kind` is `POLL_LOOP` or `POLL_ENTRY`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_interrupt_poll(
    a: &mut asm::Asm,
    ilayout: &crate::interpreter::InterpLayout,
    kind: u32,
    l_unwind: usize,
) {
    let offset = ilayout.interrupt_poll_tick;
    if !ilayout.valid
        || !offset.is_multiple_of(4)
        || offset / 4 >= 4096
        || !ilayout.gc_next.is_multiple_of(8)
        || ilayout.gc_next / 8 >= 4096
    {
        emit_op_helper(a, H_INTERRUPT, kind, l_unwind);
        return;
    }
    let done = a.new_label();
    let slow = a.new_label();
    a.ldr_imm(14, 19, 72); // ctx.interp
    a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, live_objects) as u32);
    emit_live_count(a, slow);
    a.ldr_imm(10, 14, ilayout.gc_next as u32);
    a.cmp_reg_x(9, 10);
    a.b_cond(C_GT, slow);
    a.ldr_w_imm(9, 14, offset as u32);
    a.add_imm(9, 9, 1);
    a.str_w_imm(9, 14, offset as u32);
    let mask = asm::logical_imm_w(0x3fff).expect("interrupt divider mask is encodable");
    a.logic_imm_w(0, 10, 9, mask);
    a.cbnz(10, false, done);
    a.bind(slow);
    emit_op_helper(a, H_INTERRUPT, kind, l_unwind);
    a.bind(done);
}

/// Start a helper-free region's register countdown. Since its body cannot enter author
/// code, helpers or GC, the shared tick cannot change before its matching finish.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_region_poll_start(
    a: &mut asm::Asm,
    ilayout: &crate::interpreter::InterpLayout,
    counter: u32,
) {
    a.ldr_imm(14, 19, 72);
    a.ldr_w_imm(9, 14, ilayout.interrupt_poll_tick as u32);
    a.logic_imm_w(0, 9, 9, asm::logical_imm_w(0x3fff).unwrap());
    a.movz(counter, 16384, 0);
    a.sub_reg(counter, counter, 9);
}

/// Commit all private backedges, including wraparound, to the shared divider. Cold exits
/// amortize this memory traffic; the hot loop needs only SUBS + conditional branch.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_region_poll_finish(
    a: &mut asm::Asm,
    ilayout: &crate::interpreter::InterpLayout,
    counter: u32,
) {
    a.ldr_imm(14, 19, 72);
    a.ldr_w_imm(9, 14, ilayout.interrupt_poll_tick as u32);
    a.logic_imm_w(1, 9, 9, asm::logical_imm_w(0x3fff).unwrap());
    a.add_imm(9, 9, 1);
    a.sub_reg(9, 9, counter);
    a.str_w_imm(9, 14, ilayout.interrupt_poll_tick as u32);
}

/// Private optimized backedges must participate in the same cancellation cadence as
/// canonical headers. On a due tick materialize and revisit that header without storing
/// the tick: it performs the poll exactly once with all owned roots visible to helpers.
///
/// A general region (`check_allocation`) also bails while the live-object count passes
/// `gc_next` and while the heap's due word is set (+1 instruction, see
/// [`emit_live_count`]); its checked effects never poll themselves. The bail lands on the
/// canonical header, whose poll collects. While collections are suppressed or deferred the
/// word stays set and every backedge bails, but each bail follows a whole iteration run in the
/// region, and the header re-enters the region after its poll: the loop makes progress at
/// the cost of a materialization, a helper call and a reload per iteration.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_region_poll_guard(
    a: &mut asm::Asm,
    ilayout: &crate::interpreter::InterpLayout,
    bail: usize,
    check_allocation: bool,
) {
    let offset = ilayout.interrupt_poll_tick;
    if !ilayout.valid || !offset.is_multiple_of(4) || offset / 4 >= 4096 {
        a.b(bail);
        return;
    }
    a.ldr_imm(14, 19, 72);
    // A general region may allocate through checked effects. Bail to its canonical
    // header before collection, so virtual operands/private homes are materialized.
    // Helper-free numeric regions use their separate register-only countdown.
    if check_allocation {
        if !ilayout.gc_next.is_multiple_of(8) || ilayout.gc_next / 8 >= 4096 {
            a.b(bail);
            return;
        }
        a.ldr_imm(9, 19, std::mem::offset_of!(JitCtx, live_objects) as u32);
        emit_live_count(a, bail);
        a.ldr_imm(10, 14, ilayout.gc_next as u32);
        a.cmp_reg_x(9, 10);
        a.b_cond(C_GT, bail);
    }
    a.ldr_w_imm(9, 14, offset as u32);
    a.add_imm(9, 9, 1);
    let mask = asm::logical_imm_w(0x3fff).unwrap();
    a.logic_imm_w(0, 10, 9, mask);
    a.cbz(10, false, bail);
    a.str_w_imm(9, 14, offset as u32);
}

/// An infallible helper (returns the new sp): return/handler bookkeeping.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_helper(a: &mut asm::Asm, idx: usize, imm: u32) {
    a.mov(0, 19);
    a.movz(1, imm, 0);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (idx * 8) as u32);
    a.blr(16);
    a.mov(20, 0);
}

/// Publish a return only after the completion router has established that no finalizer or
/// iterator pad must execute (ECMA-262 ReturnStatement and TryStatement Evaluation). The
/// result word may already be occupied after a failed tail call; GC owns its displaced cell.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_return(a: &mut asm::Asm, mode: u32, ret_ok: usize) {
    let ret = std::mem::offset_of!(JitCtx, ret) as u32;
    if mode == 1 {
        a.ldur(9, 20, -8);
        a.sub_imm(20, 20, 8);
    } else {
        a.mov_imm64(9, crate::value::PACK_UNDEFINED);
    }
    a.str_imm(9, 19, ret);
    a.b(ret_ok);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_completion(a: &mut asm::Asm, pc: u32, ret_ok: usize) {
    emit_helper(a, H_COMPLETE, pc);
    a.cbz(1, true, ret_ok);
    a.br(1);
}

/// Record one unconditional loop back-edge without changing the operand-stack pointer. This is
/// emitted only for detailed chunks; normal JIT loops retain their direct branch instruction.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_loop_backedge(a: &mut asm::Asm, pc: u32) {
    emit_helper(a, H_LOOP_BACKEDGE, pc);
}

/// Condition helper: leaves the flag in w1, new sp in x0 (null = threw during ToBoolean).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_cond(a: &mut asm::Asm, mode: u32, l_unwind: usize) {
    emit_cond_imm(a, mode, l_unwind);
}

/// Condition helper variant carrying an exact baseline PC and branch polarity for detailed
/// branch feedback. The low bits retain the ordinary condition mode; higher bits are diagnostic
/// metadata decoded by `jit_cond` and do not affect the predicate or stack behavior.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_cond_profiled(a: &mut asm::Asm, mode: u32, pc: u32, take_when_true: bool, l_unwind: usize) {
    let packed = mode | 1 << 2 | u32::from(take_when_true) << 3 | (pc << 4);
    emit_cond_imm(a, packed, l_unwind);
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_cond_imm(a: &mut asm::Asm, mode: u32, l_unwind: usize) {
    a.mov(0, 19);
    a.mov_imm64(1, mode as u64);
    a.mov(2, 20);
    a.ldr_imm(16, 21, (H_COND * 8) as u32);
    a.blr(16);
    a.cbz(0, true, l_unwind);
    a.mov(20, 0);
}

/// Non-owning conditional check for the short-circuit peek ops. ToBoolean cannot throw and the
/// value stays on the operand stack, where the collector traces it (the word owns nothing), so
/// common tags need no helper transition. BigInt and a possible HTMLDDA object retain the
/// canonical helper path.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn emit_peek_cond_inline(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    not_nullish: bool,
    l_unwind: usize,
) {
    let done = a.new_label();
    let l_false = a.new_label();
    emit_exec_word_load(a, 12, 20, -8);
    if not_nullish {
        // Empty is an internal completion marker, not nullish; preserve the helper's exact
        // `Undefined | Null` predicate even though Empty should not escape onto this stack.
        a.mov_imm64(9, crate::value::PACK_UNDEFINED);
        a.cmp_reg_x(12, 9);
        a.b_cond(C_EQ, l_false);
        a.mov_imm64(9, crate::value::PACK_NULL);
        a.cmp_reg_x(12, 9);
        a.cset_w(1, C_NE);
        a.b(done);
        a.bind(l_false);
        a.movz(1, 0, 0);
        a.bind(done);
        return;
    }

    let slow = a.new_label();
    let l_bool = a.new_label();
    let l_num = a.new_label();
    let l_str = a.new_label();
    let l_obj = a.new_label();
    let l_true = a.new_label();
    let tagged = a.new_label();
    emit_exec_number_guard(a, 12, 0, 10, tagged);
    a.b(l_num);
    a.bind(tagged);
    a.lsr_imm(9, 12, 48);
    for (tag, target) in [
        (crate::value::PACK_OBJ, l_obj),
        (crate::value::PACK_BOOL, l_bool),
        (crate::value::PACK_UNDEFINED, l_false),
        (crate::value::PACK_NULL, l_false),
        (crate::value::PACK_STR, l_str),
        (crate::value::PACK_SYM, l_true),
        (crate::value::PACK_EMPTY, l_false),
    ] {
        a.movz(10, (tag >> 48) as u32, 0);
        a.cmp_reg_w(9, 10);
        a.b_cond(C_EQ, target);
    }
    a.b(slow); // BigInt and property-only/reserved tags retain the checked path.

    a.bind(l_bool);
    a.ldurb(1, 20, -8);
    a.b(done);
    a.bind(l_num);
    a.movz(12, 0, 0);
    a.fmov_d_x(1, 12);
    a.fcmp(0, 1);
    a.cset_w(11, C_EQ);
    a.cset_w(12, C_VS);
    a.logic_w(1, 11, 11, 12); // zero or NaN = falsy
    a.movz(12, 1, 0);
    a.logic_w(2, 1, 11, 12); // invert to truthy
    a.b(done);
    a.bind(l_str);
    emit_exec_word_load(a, 12, 20, -8);
    emit_exec_payload(a, 12, 12);
    a.ldr_w_imm(11, 12, crate::lstr::LEN_OFF as u32);
    a.cmp_imm_w(11, 0);
    a.cset_w(1, C_NE);
    a.b(done);
    a.bind(l_obj);
    emit_exec_word_load(a, 12, 20, -8);
    emit_exec_payload(a, 12, 12);
    a.add_imm(11, 12, layout.obj_from_cell as u32);
    a.ldrb_imm(11, 11, layout.obj_ic_plain as u32);
    a.cbz(11, false, slow); // includes the engine's possible HTMLDDA object
    a.bind(l_true);
    a.movz(1, 1, 0);
    a.b(done);
    a.bind(l_false);
    a.movz(1, 0, 0);
    a.b(done);
    a.bind(slow);
    emit_cond(a, COND_PEEK_TRUTHY, l_unwind);
    a.bind(done);
}

// ---------------------------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------------------------

/// `ctx.global_body` for a fresh frame: the live global object's body pointer. Populated even
/// when this chunk never reads it, because a direct (shared-ctx) JIT→JIT call can only enter a
/// `needs_global` CALLEE if the caller's ctx already carries the pointer — a null here forces
/// every such call through the layered path. Falls back to null (never needed) or the original
/// panicking borrow (needed, but the global is mutably borrowed — same failure as before).
fn jit_global_body(i: &Interp, code: &JitCode) -> *const u8 {
    if let Ok(b) = i.global.try_borrow() {
        &*b as *const crate::value::Object as *const u8
    } else if code.needs_global {
        let b = i.global.borrow();
        &*b as *const crate::value::Object as *const u8
    } else {
        std::ptr::null()
    }
}

/// Compile a resumable body lazily with the same executable-budget retry policy as ordinary
/// functions. Unsupported architectures keep the authoritative bytecode continuation.
pub(crate) fn continuation_code(i: &mut Interp, chunk: &Chunk) -> Option<Rc<JitCode>> {
    debug_assert!(chunk.jit_is_resumable());
    let permit = if chunk.jit.ready_to_compile()
        && ensure_executable_capacity(chunk.jit_budget_wait_bytes.get())
    {
        chunk.jit.begin_compile()
    } else {
        None
    };
    if let Some(permit) = permit {
        let layout = *i
            .jit_layout
            .get_or_init(|| crate::value::jit_layout(&i.object_proto));
        if !i.interp_layout.get().valid {
            let layout = crate::interpreter::interp_layout(i);
            i.interp_layout.set(layout);
        }
        match compile_profiled(chunk, &layout, &i.interp_layout.get()) {
            JitCompileOutcome::Compiled(code) => {
                chunk.jit_budget_wait_bytes.set(0);
                permit.commit(Some(Rc::new(code)));
            }
            JitCompileOutcome::Deferred { required_bytes } => {
                chunk.jit_budget_wait_bytes.set(required_bytes);
            }
            JitCompileOutcome::Unavailable => {
                permit.commit(None);
            }
        }
    }
    chunk.jit.get().flatten()
}

/// Run native instructions until the next exact VmStep, borrowing the heap continuation's
/// owned-word buffers. Await/GeneratorYield and abrupt-completion routing remain in VmCoro;
/// there is no suspended native stack or duplicated scope/reference/disposal state.
///
/// # Safety
/// `state` points at distinct live fields of the same VmCoro for this complete call. No one
/// else accesses its stack while the generated code owns the initialized prefix.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_continuation(
    i: &mut Interp,
    chunk: &Chunk,
    code: &JitCode,
    state: &mut crate::bytecode::NativeContinuation,
    slots: &[PackedValue],
    pc: &mut usize,
    this_val: &Value,
    handlers: &mut Vec<crate::bytecode::Handler>,
) -> Result<crate::bytecode::VmStep, Abrupt> {
    assert!(
        chunk.jit_is_resumable(),
        "ordinary code cannot enter a continuation"
    );
    run_borrowed_frame(i, chunk, code, state, slots, pc, this_val, handlers, false)
}

/// Borrow the existing execution context without fresh-function setup or replaying any op.
/// ScriptEvaluation's Realm/environments/completion state remain owned by the VM driver;
/// this is a tier transition, not a new ECMAScript call or suspension (§§9.4, 16.1.6).
///
/// # Safety
/// All state fields must belong to this exact live Chunk/frame and remain pinned throughout
/// native execution. The initial OSR caller must prove osr_entry_depth(pc) == stack.len();
/// later calls resume only canonical completion state returned by this code/its VM driver.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_borrowed_frame(
    i: &mut Interp,
    chunk: &Chunk,
    code: &JitCode,
    state: &mut crate::bytecode::NativeContinuation,
    slots: &[PackedValue],
    pc: &mut usize,
    this_val: &Value,
    handlers: &mut Vec<crate::bytecode::Handler>,
    first_transfer: bool,
) -> Result<crate::bytecode::VmStep, Abrupt> {
    assert_eq!(
        code.entry_kind,
        NativeEntryKind::BorrowedFrame,
        "fresh-frame code cannot borrow live execution storage"
    );
    assert_eq!(
        state.chunk, chunk as *const Chunk,
        "borrowed state must belong to the compiled chunk"
    );
    let stack = &mut *state.stack;
    if first_transfer && code.osr_entry_depth(*pc) != Some(stack.len()) {
        // Defensive validation happens before lending owners or changing the frame. Normal
        // unsupported/ineligible headers are filtered by the driver and continue in the VM.
        return Err(i.throw("InternalError", "invalid initial native OSR state"));
    }
    stack.reserve(code.max_stack.saturating_sub(stack.len()));
    let stack_base = stack.as_mut_ptr();
    let initial_depth = stack.len();
    let env = &*state.env;
    let mut ctx = JitCtx {
        helpers: i.jit_helpers.as_ptr(),
        stack_base,
        final_sp: stack_base.add(initial_depth),
        slots: slots.as_ptr().cast_mut(),
        inline_ic_safe: &i.inline_ic_safe as *const _ as *const u8,
        env_raw: Env::as_ptr(env) as *const u8,
        // The VM frame's `this` stays wide; the native run reads it packed.
        this_word: PackedValue::pack(this_val.clone()),
        global_body: jit_global_body(i, code),
        genv: Env::as_ptr(&i.global_env) as usize,
        interp: i,
        chunk,
        n_slots: slots.len(),
        handlers: std::mem::take(handlers),
        handler_floor: 0,
        code_base: code.mem,
        pc_offsets: code.pc_offsets.as_ptr(),
        error: None,
        ret: PackedValue::pack(Value::Undefined),
        env_parent_raw: jit_env_parent_raw(env),
        opstat_enabled: crate::bytecode::jit_opstat_enabled(),
        callstat_enabled: crate::bytecode::jit_callstat_enabled(),
        propstat_enabled: crate::bytecode::jit_propstat_enabled(),
        inline_recompile_at: crate::bytecode::inline_recompile_at(),
        live_objects: crate::value::live_objects_ptr(&i.gc_heap),
        shape_layouts: crate::value::shape_layouts_ptr(&i.gc_heap),
        activation: None,
        resume_activation: state,
        references_raw: (&mut *state.references).as_mut_ptr(),
        accessor_holder: 0,
        accessor_slot: 0,
        resume_pc: *pc,
        resume_step: None,
    };
    stack.set_len(0);
    // The lent operand words leave the VM frame's traced length for the native run,
    // so they are this context's roots until it ends: the buffer's whole capacity, which covers
    // the entry depth and `max_stack` and is initialized (see `ValueStack`). The VM frame's
    // slots, environments and references stay registered by `drive_vm`.
    let roots = crate::gc_trace::enter_jit_frame(crate::gc_trace::JitFrameRoots {
        ctx: std::ptr::addr_of!(ctx),
        words: [
            (std::ptr::null(), 0),
            (stack_base.cast_const().cast(), stack.capacity()),
        ],
    });
    let entry: extern "C" fn(*mut JitCtx) -> u64 = std::mem::transmute(code.mem);
    let saved_strict = i.strict;
    let outcome = loop {
        if ctx.resume_pc == chunk.jit_ops().len() {
            break Ok(crate::bytecode::VmStep::Done(Value::Undefined));
        }
        let depth = ctx.final_sp.offset_from(stack_base) as usize;
        // Fail closed before indirect control transfer if a future compiler/driver violates
        // the settled-stack contract. No unvalidated native code address is ever entered.
        // Handler unwinding truncates to min(saved_depth, current_depth): an operation can
        // consume operands before throwing. Its catch (and a later yield in that catch) can
        // therefore legitimately have fewer live values than the static handler-root bound.
        if code
            .resume_depths
            .get(ctx.resume_pc)
            .copied()
            .flatten()
            .is_none_or(|bound| depth > bound)
        {
            break Err(i.throw("InternalError", "invalid native continuation state"));
        }
        if let Err(error) = i.interrupt_poll_force() {
            break Err(error);
        }
        i.strict = if (&*state.class_states).iter().any(Option::is_some) {
            true
        } else {
            chunk.jit_is_strict()
        };
        chunk.jit_runs.set(chunk.jit_runs.get().saturating_add(1));
        #[cfg(test)]
        if chunk.jit_is_resumable() {
            TEST_NATIVE_SLICES.with(|count| count.set(count.get() + 1));
        } else {
            TEST_OSR_ENTRIES.with(|count| count.set(count.get() + 1));
        }
        let ok = entry(&mut ctx);
        if let Some(error) = ctx.error.take() {
            break Err(error);
        }
        assert_eq!(ok, 1, "native continuation exit must retain its error");
        if let Some(step) = ctx.resume_step.take() {
            break Ok(step);
        }
        // A synchronous DisposeNormal can finish without suspension. Continue from its
        // settled bytecode successor using the same borrowed activation and no allocation.
    };
    drop(roots);
    i.strict = saved_strict;
    (*state.stack).set_len(ctx.final_sp.offset_from(stack_base) as usize);
    *handlers = std::mem::take(&mut ctx.handlers);
    *pc = ctx.resume_pc;
    outcome
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_NATIVE_SLICES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static TEST_OSR_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Execute a JIT-compiled chunk: mirrors `bytecode::run` (activation env, pooled slot buffer),
/// with the operand stack in a pooled flat buffer sized by the static analysis.
#[cfg(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub fn run(
    i: &mut Interp,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    env: &Env,
    this_val: Value,
    args: &[Value],
) -> Result<Value, Abrupt> {
    assert_eq!(
        code.entry_kind,
        NativeEntryKind::FreshFrame,
        "ordinary entry cannot borrow a live frame"
    );
    let env = chunk.jit_make_run_env(i, env, &this_val, args);
    let (mut slots, mut stack) = i.vm_pool.pop().unwrap_or_default();
    let (n_params, n_slots) = chunk.jit_frame();
    let seed = n_params.min(args.len());
    slots.extend(args[..seed].iter().cloned().map(PackedValue::pack));
    slots.resize_with(n_slots, || PackedValue::pack(Value::Undefined));
    if let Some(rest) = chunk.rest_slot {
        let rest_values = args.get(n_params..).unwrap_or(&[]).to_vec();
        slots.write_value(rest as usize, i.make_array(rest_values));
    }
    if let Some(s) = chunk.jit_arguments_slot() {
        slots.write_value(s as usize, chunk.arguments_slot_value(i, args, &env));
    }
    stack.clear();
    stack.reserve(code.max_stack);

    let stack_base = stack.as_mut_ptr();
    let env_raw = Env::as_ptr(&env) as *const u8;
    let env_parent_raw = jit_env_parent_raw(&env);
    let mut ctx = JitCtx {
        helpers: i.jit_helpers.as_ptr(),
        stack_base,
        final_sp: stack_base,
        env_raw,
        this_word: PackedValue::pack(this_val),
        global_body: jit_global_body(i, code),
        genv: Env::as_ptr(&i.global_env) as usize,
        env_parent_raw,
        opstat_enabled: crate::bytecode::jit_opstat_enabled(),
        callstat_enabled: crate::bytecode::jit_callstat_enabled(),
        propstat_enabled: crate::bytecode::jit_propstat_enabled(),
        inline_recompile_at: crate::bytecode::inline_recompile_at(),
        live_objects: crate::value::live_objects_ptr(&i.gc_heap),
        shape_layouts: crate::value::shape_layouts_ptr(&i.gc_heap),
        interp: i as *mut Interp,
        chunk: Rc::as_ptr(chunk),
        slots: slots.as_mut_ptr(),
        inline_ic_safe: &i.inline_ic_safe as *const std::cell::Cell<bool> as *const u8,
        n_slots,
        handlers: Vec::new(),
        activation: None,
        resume_activation: std::ptr::null_mut(),
        references_raw: std::ptr::null_mut(),
        accessor_holder: 0,
        accessor_slot: 0,
        resume_pc: 0,
        resume_step: None,
        handler_floor: 0,
        code_base: code.mem,
        pc_offsets: code.pc_offsets.as_ptr(),
        error: None,
        ret: PackedValue::pack(Value::Undefined),
    };
    let entry: extern "C" fn(*mut JitCtx) -> u64 = unsafe { std::mem::transmute(code.mem) };
    // The pooled buffers are this context's roots for the native run (the stack scan
    // sees the context itself): every slot, initialized above, and the operand words up to
    // `max_stack`, initialized because a ValueStack's whole capacity is (see `ValueStack`).
    let roots = crate::gc_trace::enter_jit_frame(crate::gc_trace::JitFrameRoots {
        ctx: std::ptr::addr_of!(ctx),
        words: [
            (slots.as_ptr().cast(), slots.len()),
            (stack_base.cast_const().cast(), code.max_stack),
        ],
    });
    let ok = entry(&mut ctx);
    drop(roots);
    drop(env); // the env handle must outlive the run (ctx.env_ref aliases it)
               // Drop any operands left on the raw stack (a throw can leave temporaries).
    unsafe {
        let mut p = ctx.stack_base;
        while p < ctx.final_sp {
            std::ptr::drop_in_place(p);
            p = p.add(1);
        }
    }
    slots.clear();
    stack.clear();
    if i.vm_pool.len() < 64 {
        i.vm_pool.push((slots, stack));
    }
    if ok == 1 {
        Ok(ctx.take_ret().into_value())
    } else {
        Err(ctx
            .error
            .take()
            .unwrap_or_else(|| Abrupt::Throw(Value::Undefined)))
    }
}

/// The per-frame storage size (in packed words) of a [`JitFrame`]: slots + operand stack of a
/// JIT frame carve one fixed raw buffer, so frame setup is a freelist pop + pointer math instead
/// of `Vec` bookkeeping. Frames that need more fall back to the pooled-`Vec` path.
pub(crate) const FRAME_BUF: usize = 256;

/// Pooled activation records kept per interpreter. Records beyond this bound are freed on
/// release; deeper recursion allocates (and later frees) the excess.
pub(crate) const FRAME_POOL_LIMIT: usize = 256;

/// A pooled JIT activation record: the frame's own [`JitCtx`] followed by its fixed
/// [`FRAME_BUF`]-word slot and operand-stack storage.
///
/// Every activation that fits runs on a record of its own — Rust entries and direct JIT→JIT
/// calls alike — so entering a frame writes only the callee's per-frame fields. Nothing is
/// saved, swapped into a shared context, or restored afterwards. Interpreter-constant fields
/// (helper table, interpreter, live-object counter, IC-safety byte, diagnostic switches and the
/// `slots` self-pointer) are written once, when the record is created; a record never leaves its
/// interpreter's pool.
///
/// Pooled records hold: `this` (the complete Undefined word at [48]) and `ret` Undefined;
/// `error`, `activation` and `resume_step` None; an empty `handlers` Vec (capacity retained);
/// null resume/reference pointers; and a zero handler floor.
/// Generated code relies on these to skip the corresponding stores on entry: a receiver-less
/// direct call leaves the callee's `this` as the pool left it.
///
/// Rooting: the interpreter's `FramePool` registers every record, and the collector
/// traces each one that is not pooled — from the pop that starts an activation (by a Rust entry
/// or a direct call's machine code) to the push or free that ends it. It traces the context's
/// values precisely (`GcTrace for JitCtx`) and all [`FRAME_BUF`] words conservatively, as packed
/// words: the slots and the operand stack, including the caller's operands under a direct call
/// (the callee closure among them) and the stale words a finished callee or a popped operand
/// leaves, which can only retain. No stack pointer is needed. `alloc` zero-fills the words, so
/// every scanned word has been written.
#[repr(C)]
pub(crate) struct JitFrame {
    pub(crate) ctx: JitCtx,
    words: [std::mem::MaybeUninit<PackedValue>; FRAME_BUF],
}

/// Byte offset of a record's slot storage from the record (and its context) base.
pub(crate) const FRAME_WORDS_OFF: usize = std::mem::offset_of!(JitFrame, words);

/// An active record (see [`JitFrame`] and `GcTrace for FramePool`).
impl crate::gc_trace::GcTrace for JitFrame {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let JitFrame { ctx, words } = self;
        ctx.trace_gc(sink);
        // SAFETY: `alloc` zero-filled every word; later writes store complete packed words.
        unsafe { crate::gc_trace::scan_words(words.as_ptr().cast::<u64>(), FRAME_BUF, sink) };
    }
}

impl JitFrame {
    /// A new record owned by `i`'s pool, with its interpreter-constant fields initialized
    /// (registered with the pool, its words zero-filled).
    pub(crate) fn alloc(i: &mut Interp) -> std::ptr::NonNull<JitFrame> {
        let mut record = Box::<JitFrame>::new_uninit();
        let base = record.as_mut_ptr();
        let record = unsafe {
            let words = std::ptr::addr_of_mut!((*base).words).cast::<PackedValue>();
            // The collector scans an active record's words whole (see `JitFrame`).
            words.write_bytes(0, FRAME_BUF);
            std::ptr::addr_of_mut!((*base).ctx).write(JitCtx {
                helpers: i.jit_helpers.as_ptr(),
                stack_base: words,
                final_sp: words,
                slots: words,
                inline_ic_safe: &i.inline_ic_safe as *const std::cell::Cell<bool> as *const u8,
                env_raw: std::ptr::null(),
                this_word: PackedValue::pack(Value::Undefined),
                global_body: std::ptr::null(),
                genv: 0,
                interp: i as *mut Interp,
                chunk: std::ptr::null(),
                n_slots: 0,
                handlers: Vec::new(),
                handler_floor: 0,
                code_base: std::ptr::null(),
                pc_offsets: std::ptr::null(),
                error: None,
                ret: PackedValue::pack(Value::Undefined),
                env_parent_raw: std::ptr::null(),
                opstat_enabled: crate::bytecode::jit_opstat_enabled(),
                callstat_enabled: crate::bytecode::jit_callstat_enabled(),
                propstat_enabled: crate::bytecode::jit_propstat_enabled(),
                inline_recompile_at: crate::bytecode::inline_recompile_at(),
                live_objects: crate::value::live_objects_ptr(&i.gc_heap),
                shape_layouts: crate::value::shape_layouts_ptr(&i.gc_heap),
                activation: None,
                resume_activation: std::ptr::null_mut(),
                resume_pc: 0,
                resume_step: None,
                references_raw: std::ptr::null_mut(),
                accessor_holder: 0,
                accessor_slot: 0,
            });
            std::ptr::NonNull::new_unchecked(Box::into_raw(record.assume_init()))
        };
        i.frame_pool.register(record);
        record
    }

    /// Restore the pooled invariants of a record whose activation has finished (its slots and
    /// operand stack already released) and return it to `i`'s pool.
    ///
    /// # Safety
    /// `frame` must come from [`JitFrame::alloc`] for `i` and no longer be executing.
    pub(crate) unsafe fn release(i: &mut Interp, frame: std::ptr::NonNull<JitFrame>) {
        let ctx = unsafe { &mut (*frame.as_ptr()).ctx };
        // A throw can escape through open try regions; a pooled record owns no handlers.
        ctx.handlers.clear();
        ctx.handler_floor = 0;
        // Owned values are released here, not by the next user of the record.
        ctx.clear_this();
        drop(ctx.take_ret());
        ctx.error = None;
        ctx.activation = None;
        ctx.resume_step = None;
        ctx.resume_activation = std::ptr::null_mut();
        ctx.resume_pc = 0;
        ctx.references_raw = std::ptr::null_mut();
        if i.frame_pool.len() < FRAME_POOL_LIMIT {
            i.frame_pool.push(frame);
        } else {
            i.frame_pool.unregister(frame);
            drop(unsafe { Box::from_raw(frame.as_ptr()) });
        }
    }
}

/// [`run`] for the JIT→JIT fast call: takes ownership of `argc` argument packed words at `args`
/// (moved off the caller's operand stack — the caller must NOT drop them), seeding parameter
/// slots by move instead of clone and dropping any surplus. Only for chunks with no activation
/// environment (`Chunk::jit_no_activation`), so the arguments have exactly one consumer.
/// `env` is borrowed raw: the caller keeps the aliased handle alive across the run.
///
/// # Safety
/// `args..args+argc` must be initialized packed owners the caller relinquishes entirely; `*env` must
/// outlive the run.
#[cfg(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) unsafe fn run_moved(
    i: &mut Interp,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    env: *const Env,
    this_val: JitThis,
    args: *mut PackedValue,
    argc: usize,
    // `chunk.jit_frame()`, precomputed by the caller (the cached call reads it from its IC).
    frame: (usize, usize),
) -> Result<PackedValue, Abrupt> {
    unsafe { run_moved_inner(i, chunk, code, env, this_val, args, argc, frame, None) }
}

/// Constructor entry from the `Op::New` helper, after allocation and `prototype` validation
/// have committed. Runs the no-activation constructor on its own pooled record exactly like
/// [`run_moved`]; the caller's context is left untouched.
///
/// # Safety
/// Same moved-argument and environment lifetime contract as [`run_moved`].
#[cfg(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_moved_shared(
    i: &mut Interp,
    _caller: &mut JitCtx,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    env: *const Env,
    this_val: JitThis,
    args: *mut PackedValue,
    argc: usize,
    frame: (usize, usize),
) -> Result<Value, Abrupt> {
    unsafe { run_moved_inner(i, chunk, code, env, this_val, args, argc, frame, None) }
        .map(PackedValue::into_value)
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_PACKED_ENV_ENTRIES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

/// Moved-frame entry for a callee that needs a real activation environment. Decode/clone only
/// captured parameters into that environment, then move the call's owned arguments into the
/// fixed frame buffer. A complete Value list is needed only to materialize an `arguments` exotic
/// before the move; ordinary captured calls require no temporary argument owners or Vec.
#[cfg(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) unsafe fn run_moved_env(
    i: &mut Interp,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    definition_env: *const Env,
    this_val: JitThis,
    args: *mut PackedValue,
    argc: usize,
    frame: (usize, usize),
) -> Result<PackedValue, Abrupt> {
    #[cfg(test)]
    TEST_PACKED_ENV_ENTRIES.with(|entries| entries.set(entries.get() + 1));
    let args_ref = unsafe { std::slice::from_raw_parts(args, argc) };
    let activation =
        chunk.jit_make_run_env_packed(i, unsafe { &*definition_env }, &this_val, args_ref);
    let arguments = chunk.jit_arguments_slot().map(|slot| {
        (
            slot as usize,
            crate::execution_storage::CallArgs::Packed(args_ref)
                .with_values(|values| chunk.arguments_slot_value(i, values, &activation)),
        )
    });
    unsafe {
        run_moved_inner(
            i,
            chunk,
            code,
            &activation as *const Env,
            this_val,
            args,
            argc,
            frame,
            arguments,
        )
    }
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
unsafe fn run_moved_inner(
    i: &mut Interp,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    env: *const Env,
    this_val: JitThis,
    args: *mut PackedValue,
    argc: usize,
    (n_params, n_slots): (usize, usize),
    arguments: Option<(usize, Value)>,
) -> Result<PackedValue, Abrupt> {
    assert_eq!(
        code.entry_kind,
        NativeEntryKind::FreshFrame,
        "moved ordinary entry cannot borrow a live frame"
    );
    if n_slots + code.max_stack > FRAME_BUF {
        return unsafe {
            run_moved_oversized(
                i,
                chunk,
                code,
                env,
                this_val,
                args,
                argc,
                (n_params, n_slots),
                arguments,
            )
        };
    }
    let seed = n_params.min(argc);
    // The activation runs on a pooled record of its own (see `JitFrame`): only per-frame fields
    // are written here, and the interpreter-constant ones were set when the record was created.
    let frame = i.frame_pool.pop().unwrap_or_else(|| JitFrame::alloc(i));
    let ctx = unsafe { &mut (*frame.as_ptr()).ctx };
    let slots_ptr = ctx.slots;
    let stack_base = unsafe { slots_ptr.add(n_slots) };
    unsafe {
        std::ptr::copy_nonoverlapping(args, slots_ptr, seed);
        // A packed Undefined is a complete tagged word, never a zeroed wide discriminant.
        for k in seed..n_slots {
            slots_ptr.add(k).write(PackedValue::pack(Value::Undefined));
        }
        // Surplus arguments were still moved to us: they become the rest parameter's Array
        // (CreateArrayFromList), or are dropped.
        if let Some(rest) = chunk.rest_slot {
            let array = i.make_array_from_raw(args.add(seed), argc - seed);
            std::ptr::drop_in_place(slots_ptr.add(rest as usize));
            slots_ptr.add(rest as usize).write(PackedValue::pack(array));
        } else {
            for k in seed..argc {
                std::ptr::drop_in_place(args.add(k));
            }
        }
        if let Some((slot, value)) = arguments {
            if slot < seed {
                std::ptr::drop_in_place(slots_ptr.add(slot));
            }
            slots_ptr.add(slot).write(PackedValue::pack(value));
        }
    }
    let env = unsafe { &*env };
    ctx.stack_base = stack_base;
    ctx.final_sp = stack_base;
    ctx.env_raw = Env::as_ptr(env) as *const u8;
    ctx.env_parent_raw = jit_env_parent_raw(env);
    ctx.global_body = jit_global_body(i, code);
    ctx.genv = Env::as_ptr(&i.global_env) as usize;
    ctx.chunk = Rc::as_ptr(chunk);
    ctx.set_this(this_val);
    ctx.n_slots = n_slots;
    ctx.code_base = code.mem;
    ctx.pc_offsets = code.pc_offsets.as_ptr();
    let entry: extern "C" fn(*mut JitCtx) -> u64 = unsafe { std::mem::transmute(code.mem) };
    let ok = entry(ctx);
    unsafe {
        let mut p = ctx.stack_base;
        while p < ctx.final_sp {
            std::ptr::drop_in_place(p);
            p = p.add(1);
        }
        // Every local is an initialized packed word, which owns nothing: dropping it releases
        // nothing, without widening the frame.
        for k in 0..n_slots {
            std::ptr::drop_in_place(slots_ptr.add(k));
        }
    }
    let result = if ok == 1 {
        Ok(ctx.take_ret())
    } else {
        Err(ctx
            .error
            .take()
            .unwrap_or_else(|| Abrupt::Throw(Value::Undefined)))
    };
    unsafe { JitFrame::release(i, frame) };
    result
}

/// [`run_moved_inner`] for a frame larger than a pooled record: pooled `Vec` storage and a
/// context of its own on the native stack.
#[cfg(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
#[allow(clippy::too_many_arguments)]
#[cold]
unsafe fn run_moved_oversized(
    i: &mut Interp,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    env: *const Env,
    this_val: JitThis,
    args: *mut PackedValue,
    argc: usize,
    (n_params, n_slots): (usize, usize),
    arguments: Option<(usize, Value)>,
) -> Result<PackedValue, Abrupt> {
    let seed = n_params.min(argc);
    let (mut slots, mut stack) = i.vm_pool.pop().unwrap_or_default();
    slots.reserve(n_slots);
    stack.clear();
    stack.reserve(code.max_stack);
    let (slots_ptr, stack_base) = (slots.as_mut_ptr(), stack.as_mut_ptr());
    unsafe {
        std::ptr::copy_nonoverlapping(args, slots_ptr, seed);
        for k in seed..n_slots {
            slots_ptr.add(k).write(PackedValue::pack(Value::Undefined));
        }
        if let Some(rest) = chunk.rest_slot {
            let array = i.make_array_from_raw(args.add(seed), argc - seed);
            std::ptr::drop_in_place(slots_ptr.add(rest as usize));
            slots_ptr.add(rest as usize).write(PackedValue::pack(array));
        } else {
            for k in seed..argc {
                std::ptr::drop_in_place(args.add(k));
            }
        }
        if let Some((slot, value)) = arguments {
            if slot < seed {
                std::ptr::drop_in_place(slots_ptr.add(slot));
            }
            slots_ptr.add(slot).write(PackedValue::pack(value));
        }
    }
    let env = unsafe { &*env };
    let mut ctx = JitCtx {
        helpers: i.jit_helpers.as_ptr(),
        stack_base,
        final_sp: stack_base,
        env_raw: Env::as_ptr(env) as *const u8,
        this_word: this_val,
        global_body: jit_global_body(i, code),
        genv: Env::as_ptr(&i.global_env) as usize,
        env_parent_raw: jit_env_parent_raw(env),
        opstat_enabled: crate::bytecode::jit_opstat_enabled(),
        callstat_enabled: crate::bytecode::jit_callstat_enabled(),
        propstat_enabled: crate::bytecode::jit_propstat_enabled(),
        inline_recompile_at: crate::bytecode::inline_recompile_at(),
        live_objects: crate::value::live_objects_ptr(&i.gc_heap),
        shape_layouts: crate::value::shape_layouts_ptr(&i.gc_heap),
        interp: i as *mut Interp,
        chunk: Rc::as_ptr(chunk),
        slots: slots_ptr,
        inline_ic_safe: &i.inline_ic_safe as *const std::cell::Cell<bool> as *const u8,
        n_slots,
        handlers: Vec::new(),
        activation: None,
        resume_activation: std::ptr::null_mut(),
        references_raw: std::ptr::null_mut(),
        accessor_holder: 0,
        accessor_slot: 0,
        resume_pc: 0,
        resume_step: None,
        handler_floor: 0,
        code_base: code.mem,
        pc_offsets: code.pc_offsets.as_ptr(),
        error: None,
        ret: PackedValue::pack(Value::Undefined),
    };
    let entry: extern "C" fn(*mut JitCtx) -> u64 = unsafe { std::mem::transmute(code.mem) };
    // As in `run`. The slot Vec's length stays 0, so its `n_slots` words, all written
    // above, are named here; the operand words are initialized up to `max_stack`.
    let roots = crate::gc_trace::enter_jit_frame(crate::gc_trace::JitFrameRoots {
        ctx: std::ptr::addr_of!(ctx),
        words: [
            (slots_ptr.cast_const().cast(), n_slots),
            (stack_base.cast_const().cast(), code.max_stack),
        ],
    });
    let ok = entry(&mut ctx);
    drop(roots);
    unsafe {
        let mut p = ctx.stack_base;
        while p < ctx.final_sp {
            std::ptr::drop_in_place(p);
            p = p.add(1);
        }
        for k in 0..n_slots {
            std::ptr::drop_in_place(slots_ptr.add(k));
        }
        // The values were dropped above; the Vec must not double-drop them.
        slots.set_len(0);
    }
    if i.vm_pool.len() < 64 {
        i.vm_pool.push((slots, stack));
    }
    if ok == 1 {
        Ok(ctx.take_ret())
    } else {
        Err(ctx
            .error
            .take()
            .unwrap_or_else(|| Abrupt::Throw(Value::Undefined)))
    }
}

#[cfg(not(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
)))]
pub fn run(
    _i: &mut Interp,
    _chunk: &Rc<Chunk>,
    _code: &JitCode,
    _env: &Env,
    _this_val: Value,
    _args: &[Value],
) -> Result<Value, Abrupt> {
    unreachable!("jit code cannot exist on this platform")
}

/// See the aarch64-macos definition; without machine code the fast call never commits.
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
)))]
pub(crate) unsafe fn run_moved(
    _i: &mut Interp,
    _chunk: &Rc<Chunk>,
    _code: &JitCode,
    _env: *const Env,
    _this_val: JitThis,
    _args: *mut PackedValue,
    _argc: usize,
    _frame: (usize, usize),
) -> Result<PackedValue, Abrupt> {
    unreachable!("jit code cannot exist on this platform")
}

/// See the native-code definition; without machine code the fast constructor never commits.
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
)))]
pub(crate) unsafe fn run_moved_shared(
    _i: &mut Interp,
    _caller: &mut JitCtx,
    _chunk: &Rc<Chunk>,
    _code: &JitCode,
    _env: *const Env,
    _this_val: JitThis,
    _args: *mut PackedValue,
    _argc: usize,
    _frame: (usize, usize),
) -> Result<Value, Abrupt> {
    unreachable!("jit code cannot exist on this platform")
}

/// See the aarch64-macos definition; without machine code the fast call never commits.
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ),
    all(
        target_arch = "x86_64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
)))]
pub(crate) unsafe fn run_moved_env(
    _i: &mut Interp,
    _chunk: &Rc<Chunk>,
    _code: &JitCode,
    _env: *const Env,
    _this_val: JitThis,
    _args: *mut PackedValue,
    _argc: usize,
    _frame: (usize, usize),
) -> Result<PackedValue, Abrupt> {
    unreachable!("jit code cannot exist on this platform")
}

#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[path = "jit_count_guard_tests.rs"]
mod count_guard_tests;
