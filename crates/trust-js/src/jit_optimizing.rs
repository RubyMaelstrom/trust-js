//! Whole-function native lowering, sharing TRust JS's canonical activation and code owner.
//!
//! This is an opt-in development backend, NOT yet a replacement for the production template
//! tier. Cranelift sees a function's entire control flow, calls and memory effects. JavaScript
//! specialization, effect-aware frame SSA, speculative inlining and deoptimization belong above
//! this layer; adding a code generator alone does not implement those optimizations.
//!
//! ECMA-262 snapshot e28783d5: Execution Contexts, GetValue/PutValue, EvaluateCall, TryStatement,
//! Number operations and WeakRef liveness govern the boundary. Every checked call sees canonical
//! owning locals/operands, and publishes the exact remaining operand prefix even on a throw.
//! Failing a numeric guard invokes that operation once, before effects, with its original inputs.
//! Exceptions/finalizers dispatch by bytecode PC, never by an assumed machine block address.
//! The production Rc graph remains authoritative: owning local words may live in SSA only
//! between publication barriers. Their exact owners are restored before any checked call.

#[path = "jit_optimizing_bitwise.rs"]
mod bitwise;
#[cfg(test)]
#[path = "jit_optimizing_call_tests.rs"]
mod call_tests;
#[path = "jit_optimizing_creation.rs"]
mod creation;
#[cfg(test)]
#[path = "jit_optimizing_creation_tests.rs"]
mod creation_tests;
#[cfg(test)]
#[path = "jit_optimizing_deopt_tests.rs"]
mod deopt_tests;
#[cfg(test)]
#[path = "jit_optimizing_diagnostic_tests.rs"]
mod diagnostic_tests;
#[path = "jit_optimizing_effects.rs"]
mod effects;
#[path = "jit_optimizing_elements.rs"]
mod elements;
#[path = "jit_optimizing_frame.rs"]
mod frame;
#[cfg(test)]
#[path = "jit_optimizing_heap_tests.rs"]
mod heap_tests;
#[path = "jit_optimizing_names.rs"]
mod names;
#[cfg(test)]
#[path = "jit_optimizing_names_tests.rs"]
mod names_tests;
#[path = "jit_optimizing_ownership.rs"]
mod ownership;
#[cfg(test)]
#[path = "jit_optimizing_ownership_tests.rs"]
mod ownership_tests;
#[path = "jit_optimizing_property.rs"]
mod property;
#[cfg(test)]
#[path = "jit_optimizing_property_tests.rs"]
mod property_tests;
#[path = "jit_optimizing_specialization.rs"]
mod specialization;
#[cfg(test)]
#[path = "jit_optimizing_specialization_tests.rs"]
mod specialization_tests;
#[path = "jit_optimizing_stack.rs"]
mod stack;
#[path = "jit_optimizing_stack_lowering.rs"]
mod stack_lowering;
#[cfg(test)]
#[path = "jit_optimizing_stack_tests.rs"]
mod stack_tests;
#[path = "jit_optimizing_stores.rs"]
mod stores;
#[path = "jit_optimizing_values.rs"]
mod value_facts;

#[path = "jit_optimizing_loop.rs"]
pub(super) mod loop_entry;
#[path = "jit_optimizing_loop_facts.rs"]
mod loop_facts;
#[cfg(test)]
#[path = "jit_optimizing_loop_facts_tests.rs"]
mod loop_facts_tests;

use super::{
    cache::CodeResidency, optimizing_diagnostics as diagnostics, ExecutableBuffer, JitCode, JitCtx,
    NativeEntryKind,
};
use crate::bytecode::{self, Chunk, Op, UpdKind};
use crate::interpreter::InterpLayout;
use crate::jit_ir::Cfg;
use crate::value::{PackedValue, PACK_BOOL, PACK_CANON_NAN, PACK_EMPTY, PACK_NULL, PACK_UNDEFINED};
use cranelift_codegen::ir::{
    condcodes::{FloatCC, IntCC},
    types, AbiParam, Block, Function, InstBuilder, MemFlagsData as MemFlags, SigRef, Signature,
    UserFuncName, Value,
};
use cranelift_codegen::{
    isa::TargetIsa,
    settings::{self, Configurable},
    Context,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use std::mem::offset_of;
use std::sync::{Arc, OnceLock};

use effects::SlotWrites;

static COMPILE_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMPILE_SUCCESSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMPILE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GENERATED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HOT_REQUESTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HOT_PUBLISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A subset of the enclosing JIT compilation counters, not additional exclusive time. The
/// disabled path never reads a clock or updates an atomic. Counts describe compilations, not
/// execution coverage: unsupported chunks may still execute through the established tier.
pub(super) fn metrics_json() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let attempts = COMPILE_ATTEMPTS.load(Relaxed);
    let successes = COMPILE_SUCCESSES.load(Relaxed);
    let nanos = COMPILE_NANOS.load(Relaxed);
    let bytes = GENERATED_BYTES.load(Relaxed);
    let hot_requests = HOT_REQUESTS.load(Relaxed);
    let hot_published = HOT_PUBLISHED.load(Relaxed);
    let call_stubs = super::call_stub::metrics_json();
    let diagnostics = diagnostics::snapshot_json();
    format!(
        "{{\"compile_attempts\":{attempts},\"compile_successes\":{successes},\"compile_declines\":{},\"compile_seconds\":{:.9},\"generated_code_bytes\":{bytes},\"hot_requests\":{hot_requests},\"hot_published\":{hot_published},\"hot_declines\":{},\"shared_call_stubs\":{call_stubs},\"diagnostics\":{diagnostics}}}",
        attempts.saturating_sub(successes),
        nanos as f64 / 1_000_000_000.0,
        hot_requests.saturating_sub(hot_published)
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    Forced,
    Hot,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("TRUST_JS_OPT_JIT").as_deref() {
        Ok("1") => Mode::Forced,
        Ok("hot") => Mode::Hot,
        _ => Mode::Off,
    })
}

pub(super) fn selected(chunk: &Chunk) -> bool {
    chunk.optimizing_misses.get() < SPECIALIZATION_MISS_LIMIT
        && (mode() == Mode::Forced || (mode() == Mode::Hot && chunk.optimizing_candidate))
}

use crate::bytecode::SPECIALIZATION_MISS_LIMIT;

/// Called only after a failed specialization guard has published the canonical frame.
/// Retire this version after bounded misses instead of sending every future invocation
/// through a terminal VM continuation. Old frames/leases remain protected by the cache.
unsafe extern "C" fn specialization_miss(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
    let chunk = &*(*ctx).chunk;
    let misses = chunk.optimizing_misses.get().saturating_add(1);
    chunk.optimizing_misses.set(misses);
    if misses == SPECIALIZATION_MISS_LIMIT {
        chunk.jit.request_retirement();
    }
    bytecode::native_deopt::resume_before(ctx, pc, sp)
}

fn admission_override() -> Option<u32> {
    static AT: OnceLock<Option<u32>> = OnceLock::new();
    *AT.get_or_init(|| {
        std::env::var("TRUST_JS_OPT_JIT_HOT_AT")
            .ok()?
            .parse::<u32>()
            .ok()
            .map(|n| n.max(1))
    })
}

pub(super) fn hot_threshold(op_count: usize) -> u32 {
    if mode() != Mode::Hot || op_count > 256 {
        return 0;
    }
    // Four separated observations, without instrumenting any bytecode fast path. Only
    // these countdown expirations call Rust; later checkpoints are benefit-dependent.
    admission_override().unwrap_or(256).div_ceil(4).clamp(1, 64)
}

pub(super) fn hot_counter(chunk: &Chunk) -> Option<*const std::cell::Cell<u32>> {
    (mode() == Mode::Hot
        && !chunk.optimizing_candidate
        && chunk.jit_ops().len() <= 256
        && !chunk.jit_detailed_feedback_enabled()
        && bytecode::native_deopt::supported(chunk)
        && has_sampling_source(chunk)
        && chunk.optimizing_remaining.get() != 0)
        .then_some(&chunk.optimizing_remaining as *const _)
}

fn has_sampling_source(chunk: &Chunk) -> bool {
    // A necessary condition for specialization::Plan::build: facts can enter only
    // through used sampled formals or the supported property-read sites. Captured
    // names and unused formals cannot provide those facts. Do not put a permanent
    // entry branch on such bodies just to discover that at a later checkpoint.
    // This changes tier bookkeeping only; ECMA-262 GetValue/OrdinaryGet and Call
    // still execute through the complete established tier, including getters.
    let formals = chunk.jit_frame().0.min(16);
    chunk.jit_ops().iter().any(|op| match op {
        Op::LoadLocal(slot) | Op::UpdateLocal(slot, _) | Op::GetPropLocal(slot, ..) => {
            usize::from(*slot) < formals
        }
        Op::GetPropThis(..) => true,
        _ => false,
    })
}

/// Called at bounded baseline entry checkpoints, before executing any bytecode.
/// The current mapping remains resident/active and is NEVER overwritten. Function::code2
/// publishes a separate canonical chunk; epoch invalidation makes FUTURE callers re-resolve.
/// ECMA-262 Execution Contexts / [[Call]]: identity comes from the live invocation, not a
/// function-name lookup or a closure instance remembered by a compilation cache.
pub(super) unsafe extern "C" fn request_hot_upgrade(ctx: *mut JitCtx) {
    let ctx = &mut *ctx;
    let chunk = &*ctx.chunk;
    if chunk.jit_ops().len() > 256
        || chunk.jit_detailed_feedback_enabled()
        || !bytecode::native_deopt::supported(chunk)
        || !has_sampling_source(chunk)
    {
        return;
    }
    let samples = chunk.optimizing_samples.get_or_init(|| {
        let mut samples = crate::feedback::EntrySamples::new(hot_threshold(chunk.jit_ops().len()));
        samples.properties = chunk
            .jit_ops()
            .iter()
            .enumerate()
            .filter_map(|(pc, op)| match op {
                Op::GetPropLocal(slot, ..) if usize::from(*slot) < chunk.jit_frame().0.min(16) => {
                    Some(pc)
                }
                Op::GetPropThis(..) => Some(pc),
                _ => None,
            })
            .take(16)
            .map(|pc| (pc as u32, std::cell::Cell::new(0)))
            .collect();
        Box::new(samples)
    });
    samples.entries.set(
        samples
            .entries
            .get()
            .saturating_add(u64::from(samples.interval.get())),
    );
    samples.samples.set(samples.samples.get().saturating_add(1));
    let count = chunk
        .jit_frame()
        .0
        .min(samples.classes.len())
        .min(ctx.n_slots);
    for (slot, observed) in samples.classes[..count].iter().enumerate() {
        observed.set(observed.get() | sampled_class(&*ctx.slots.add(slot)));
    }
    for (pc, observed) in &samples.properties {
        observed.set(
            observed.get() | sampled_property(ctx, *pc as usize).map_or(u32::MAX, sampled_number),
        );
    }
    if count == 0 && samples.properties.is_empty() {
        return;
    }
    let rearm = |distance: u64| {
        let interval = distance.clamp(1, u64::from(u32::MAX)) as u32;
        samples.interval.set(interval);
        chunk.optimizing_remaining.set(interval);
    };
    if samples.samples.get() < 4 {
        rearm(u64::from(hot_threshold(chunk.jit_ops().len())));
        return;
    }
    let inputs: Vec<_> = samples.classes[..count]
        .iter()
        .enumerate()
        .map(|(slot, bits)| (slot as u16, bits.get()))
        .collect();
    let properties: Vec<_> = samples
        .properties
        .iter()
        .map(|(pc, bits)| (*pc, bits.get()))
        .collect();
    let Ok(cfg) = Cfg::build(chunk) else {
        return;
    };
    let Some(stack) = stack::StackPlan::build(chunk, &cfg) else {
        return;
    };
    let Some(plan) = specialization::Plan::build(chunk, &cfg, &stack, &inputs, &properties) else {
        return;
    };
    let target = admission_override()
        .map(u64::from)
        .unwrap_or_else(|| plan.admission_entries(chunk).max(4096));
    if samples.entries.get() < target {
        rearm(target - samples.entries.get());
        return;
    }
    let i = &mut *ctx.interp;
    let Some(frame) = i.fn_frames.last() else {
        return;
    };
    if frame.fn_ptr == 0 {
        return;
    }
    let callee = frame.callee();
    let function = {
        let Ok(object) = callee.try_borrow() else {
            return;
        };
        let crate::value::Callable::User(user) = &object.call else {
            return;
        };
        user.func.clone()
    };
    // Script/continuation entries do not manufacture a Function execution context. A foreign
    // or already upgraded frame cannot justify publishing this chunk as that function's tier.
    if function.code2.get().is_some()
        || function
            .code
            .get()
            .and_then(Option::as_ref)
            .is_none_or(|primary| !std::ptr::eq(&**primary, chunk))
    {
        return;
    }
    let counted = super::perf_metrics_enabled();
    if counted {
        HOT_REQUESTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let Some(next) = bytecode::compile_for_optimizer(&function, chunk, &inputs, &properties) else {
        return;
    };
    let Some(permit) = next.jit.begin_compile() else {
        return;
    };
    let values = *i
        .jit_layout
        .get_or_init(|| crate::value::jit_layout(&i.object_proto));
    if !i.interp_layout.get().valid {
        let layout = crate::interpreter::interp_layout(i);
        i.interp_layout.set(layout);
    }
    // Do not invalidate every caller just to publish an uncompiled or declined version.
    // The old mapping is active/pinned; the new slot is reserved before any code emission.
    let super::JitCompileOutcome::Compiled(code) =
        super::compile_profiled_with(&next, || compile(&next, &values, &i.interp_layout.get()))
    else {
        return;
    };
    permit.commit(Some(std::rc::Rc::new(code)));
    if function.code2.set(Some(next)).is_ok() {
        bytecode::invalidate_call_caches();
        if counted {
            HOT_PUBLISHED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn sampled_class(value: &PackedValue) -> u32 {
    use crate::feedback::ValueClass;
    if let Some(number) = value.number() {
        sampled_number(number)
    } else if value.is_boolean() {
        ValueClass::Boolean.bit()
    } else {
        // Do not manufacture a Value or inspect a JS object to classify a rejected input.
        // A generic mask is permanently ineligible for this version's scalar guards.
        u32::MAX
    }
}

fn sampled_number(number: f64) -> u32 {
    use crate::feedback::ValueClass;
    if number >= i32::MIN as f64
        && number <= i32::MAX as f64
        && number.fract() == 0.0
        && !(number == 0.0 && number.is_sign_negative())
    {
        ValueClass::NumberInt32.bit()
    } else {
        ValueClass::NumberDouble.bit()
    }
}

/// Inspect only an already cached own data descriptor. No [[Get]], getter, proxy,
/// conversion or deferred-value materialization occurs at a sampling checkpoint.
unsafe fn sampled_property(ctx: &JitCtx, pc: usize) -> Option<f64> {
    let chunk = &*ctx.chunk;
    let (receiver, name, cache) = match *chunk.jit_ops().get(pc)? {
        Op::GetPropLocal(slot, name, cache) if usize::from(slot) < ctx.n_slots => (
            (&*ctx.slots.add(usize::from(slot))).sampled_object()?,
            name,
            cache,
        ),
        Op::GetPropThis(name, cache) => (ctx.this_val.as_obj()?.clone(), name, cache),
        _ => return None,
    };
    let state = chunk
        .jit_cache_preferred(cache)
        .filter(|state| state.depth == 0)?;
    let object = receiver.try_borrow().ok()?;
    if !object.ic_plain.get()
        || !matches!(object.exotic, crate::value::Exotic::None)
        || object.props.shape() != state.recv_shape
    {
        return None;
    }
    let (key, property) = object.props.entry_at(state.slot as usize)?;
    if key.as_ref() != chunk.jit_name(name) {
        return None;
    }
    property.number_value()
}

fn isa() -> Option<&'static Arc<dyn TargetIsa>> {
    static ISA: OnceLock<Option<Arc<dyn TargetIsa>>> = OnceLock::new();
    ISA.get_or_init(|| {
        let mut flags = settings::builder();
        flags.set("opt_level", "speed").ok()?;
        // ECMA-262 e28783d5 #sec-ecmascript-language-types-number-type:
        // preserve IEEE operations and both zero signs. packed_number canonicalizes
        // every result before it enters the owning-word ABI. Cranelift's separate
        // deterministic-Wasm pass would canonicalize each operation a second time;
        // disabling that redundant pass does not enable fast-math reassociation.
        flags.set("enable_nan_canonicalization", "false").ok()?;
        flags.set("enable_verifier", "true").ok()?;
        flags.set("preserve_frame_pointers", "true").ok()?;
        flags.set("is_pic", "false").ok()?;
        flags.set("use_colocated_libcalls", "false").ok()?;
        flags.set("enable_probestack", "true").ok()?;
        flags.set("probestack_strategy", "inline").ok()?;
        cranelift_native::builder()
            .ok()?
            .finish(settings::Flags::new(flags))
            .ok()
    })
    .as_ref()
}

#[derive(Debug)]
enum CompileError {
    Unsupported,
    ControlFlow(crate::jit_ir::BuildError),
    Codegen(String),
    Relocation,
    Capacity,
    LoweringBudget,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("unsupported entry or target"),
            Self::ControlFlow(error) => write!(f, "invalid CFG: {error}"),
            Self::Codegen(error) => write!(f, "code generation: {error}"),
            Self::Relocation => f.write_str("unexpected external relocation"),
            Self::Capacity => f.write_str("executable allocation unavailable"),
            Self::LoweringBudget => f.write_str("whole-function IR work budget exceeded"),
        }
    }
}

pub(super) fn compile(
    chunk: &Chunk,
    values: &crate::value::JitLayout,
    layout: &InterpLayout,
) -> Option<JitCode> {
    let started = super::perf_metrics_enabled().then(std::time::Instant::now);
    let result = compile_checked(chunk, values, layout);
    if let Some(started) = started {
        use std::sync::atomic::Ordering::Relaxed;
        COMPILE_ATTEMPTS.fetch_add(1, Relaxed);
        COMPILE_NANOS.fetch_add(
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Relaxed,
        );
        if let Ok(code) = &result {
            COMPILE_SUCCESSES.fetch_add(1, Relaxed);
            GENERATED_BYTES.fetch_add(code.len as u64, Relaxed);
        }
    }
    match result {
        Ok(code) => Some(code),
        Err(error) => {
            if std::env::var_os("TRUST_JS_OPT_JIT_LOG").is_some() {
                eprintln!("[optimizing-jit] {} ops: {error}", chunk.jit_ops().len());
            }
            None
        }
    }
}

/// Scalar helper ABI works on both SysV/AAPCS64 and Windows x64. Do not assume Rust's
/// two-word aggregate return uses the same registers on all supported targets.
type Helper = unsafe extern "C" fn(*mut JitCtx, u32, *mut PackedValue) -> u64;

macro_rules! checked_helper {
    ($name:ident, $implementation:path, $heap_kind:expr) => {
        unsafe extern "C" fn $name(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
            #[cfg(test)]
            elements::TEST_HEAP_HELPERS.with(|count| {
                let mut counts = count.get();
                counts[$heap_kind] += 1;
                count.set(counts);
            });
            let result = $implementation(ctx, pc, sp);
            (*ctx).final_sp = result.sp;
            result.flag
        }
    };
    ($name:ident, $implementation:path) => {
        unsafe extern "C" fn $name(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
            let result = $implementation(ctx, pc, sp);
            (*ctx).final_sp = result.sp;
            result.flag
        }
    };
}
unsafe extern "C" fn exec(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
    #[cfg(test)]
    {
        let op = &(&*(*ctx).chunk).jit_ops()[pc as usize];
        if matches!(
            op,
            Op::LoadName(..) | Op::LoadNameForCall(..) | Op::LoadCap(_)
        ) {
            names::TEST_NAME_HELPERS.with(|count| count.set(count.get() + 1));
        }
        let kind = match op {
            Op::LoadLocal(_) => Some(0),
            Op::Dup => Some(1),
            Op::Pop => Some(2),
            Op::StoreLocal(_) => Some(3),
            _ => None,
        };
        if let Some(kind) = kind {
            ownership::TEST_OWNERSHIP_HELPERS.with(|count| {
                let mut counts = count.get();
                counts[kind] += 1;
                count.set(counts);
            });
        }
        // An opaque operand prefix deliberately retains generic execution. Count that
        // path too: a zero dedicated-helper count alone must not masquerade as native coverage.
        let heap_kind = match op {
            Op::SetProp(..)
            | Op::SetPropDrop(..)
            | Op::SetPropThisDrop(..)
            | Op::SetPropLocalDrop(..) => Some(0),
            Op::GetElem | Op::GetElemLocal(_) | Op::GetMethodElem => Some(1),
            Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(_) | Op::SetElemLocalDrop(_) => {
                Some(2)
            }
            _ => None,
        };
        if let Some(kind) = heap_kind {
            elements::TEST_HEAP_HELPERS.with(|count| {
                let mut counts = count.get();
                counts[kind] += 1;
                count.set(counts);
            });
        }
    }
    let result = bytecode::jit_exec(ctx, pc, sp);
    (*ctx).final_sp = result.sp;
    result.flag
}
checked_helper!(call, bytecode::jit_call);
checked_helper!(condition, bytecode::jit_cond);
checked_helper!(interrupt, bytecode::jit_interrupt);
unsafe extern "C" fn get_property(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
    #[cfg(test)]
    property::TEST_READ_HELPERS.with(|count| count.set(count.get() + 1));
    let result = bytecode::jit_get_prop(ctx, pc, sp);
    (*ctx).final_sp = result.sp;
    result.flag
}
checked_helper!(set_property, bytecode::jit_set_prop, 0);
checked_helper!(get_element, bytecode::jit_get_element, 1);
checked_helper!(set_element, bytecode::jit_set_elem, 2);
checked_helper!(make_array, bytecode::jit_make_array);
checked_helper!(make_object, bytecode::jit_make_object);
checked_helper!(construct, bytecode::jit_new);

unsafe extern "C" fn push_handler(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
    (*ctx).final_sp = bytecode::jit_push_handler(ctx, pc, sp);
    0
}

unsafe extern "C" fn pop_handler(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) -> u64 {
    (*ctx).final_sp = bytecode::jit_pop_handler(ctx, pc, sp);
    0
}

unsafe extern "C" fn return_value(ctx: *mut JitCtx, mode: u32, sp: *mut PackedValue) -> u64 {
    (*ctx).final_sp = bytecode::jit_return(ctx, mode, sp);
    0
}

#[cfg(test)]
thread_local! {
    /// Emitted local reloads / all-tracked alternative, not runtime timings.
    static TEST_LOCAL_RELOADS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static TEST_VALUE_PROOFS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

struct Lowering<'a, 'b> {
    b: FunctionBuilder<'a>,
    chunk: &'b Chunk,
    values: &'b crate::value::JitLayout,
    owners_valid: bool,
    ctx: Value,
    interp: Value,
    slots: Value,
    locals: Vec<Option<Variable>>,
    frame_plan: &'b frame::FramePlan,
    value_plan: Option<&'b value_facts::ValuePlan>,
    property_results: &'b [(u32, value_facts::Types)],
    loop_facts: Option<&'b loop_facts::Plan>,
    loop_field_addresses: Vec<Variable>,
    loop_field_values: Vec<Variable>,
    loop_validation: Option<Block>,
    loop_validation_returns: Vec<Block>,
    loop_scalars: bool,
    sparse_reloads: bool,
    local_effects: bool,
    native_heap: bool,
    native_creation: bool,
    direct_returns: bool,
    dirty: &'b [u16],
    stack_plan: &'b stack::StackPlan,
    stack_vars: Vec<Variable>,
    /// Non-owning Number representation paired with each canonical operand word.
    /// It is meaningful only after that word is proven/guarded to be a Number.
    stack_numbers: Vec<Variable>,
    stack_state: stack::StackState,
    pc: usize,
    sp: Variable,
    helper_sig: SigRef,
    call_sig: SigRef,
    call_stubs: &'b [std::rc::Rc<super::call_stub::SharedCallStub>],
    diagnostics: Option<&'b diagnostics::Code>,
    unwind: Block,
    ok: Block,
    error: Block,
    dispatch: Block,
    pcs: Vec<Block>,
}

impl Lowering<'_, '_> {
    fn count_runtime(&mut self, counter: usize, amount: usize) {
        if let Some(record) = self.diagnostics {
            emit_diagnostic_count(&mut self.b, record, counter, amount);
        }
    }

    fn count_publication(&mut self, words: usize) {
        if let Some(record) = self.diagnostics {
            record
                .emitted_publications
                .set(record.emitted_publications.get() + words as u64);
        }
        self.count_runtime(diagnostics::PUBLICATION, words);
    }

    fn count_heap_guard(&mut self) {
        if let Some(record) = self.diagnostics {
            record.emitted_guards.set(record.emitted_guards.get() + 1);
        }
        self.count_runtime(diagnostics::HEAP_GUARD, 1);
    }

    fn input_types(&self) -> value_facts::Inputs {
        self.value_plan
            .and_then(|plan| plan.at.get(self.pc))
            .copied()
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn count_value_proof(kind: usize, count: usize) {
        TEST_VALUE_PROOFS.with(|counts| {
            let mut values = counts.get();
            values[kind] += count;
            counts.set(values);
        });
    }

    fn local(&mut self, slot: u16) -> Value {
        self.b
            .use_var(self.locals[slot as usize].expect("tracked local"))
    }

    fn set_local(&mut self, slot: u16, value: Value) {
        self.b
            .def_var(self.locals[slot as usize].expect("tracked local"), value);
    }

    fn publish_locals(&mut self) {
        self.count_publication(self.dirty.len());
        for &slot in self.dirty {
            let value = self
                .b
                .use_var(self.locals[slot as usize].expect("dirty tracked local"));
            self.b
                .ins()
                .store(MemFlags::trusted(), value, self.slots, i32::from(slot) * 8);
        }
    }

    fn reload_locals(&mut self, live: &[u16], writes: SlotWrites) {
        let live = if self.sparse_reloads {
            live
        } else {
            &self.frame_plan.tracked
        };
        let writes = if self.local_effects && self.sparse_reloads {
            writes
        } else {
            SlotWrites::All
        };
        #[cfg(test)]
        TEST_LOCAL_RELOADS.with(|count| {
            let (actual, full) = count.get();
            count.set((
                actual + live.iter().filter(|&&slot| writes.contains(slot)).count(),
                full + self.frame_plan.tracked.len(),
            ));
        });
        for &slot in live {
            if !writes.contains(slot) {
                continue;
            }
            let value = self.b.ins().load(
                types::I64,
                MemFlags::trusted(),
                self.slots,
                i32::from(slot) * 8,
            );
            self.b.def_var(
                self.locals[usize::from(slot)].expect("live tracked local"),
                value,
            );
        }
    }

    fn load(&mut self, base: Value, offset: i32) -> Value {
        self.b
            .ins()
            .load(types::I64, MemFlags::trusted(), base, offset)
    }

    fn store(&mut self, value: Value, base: Value, offset: i32) {
        self.b.ins().store(MemFlags::trusted(), value, base, offset);
    }

    fn top(&mut self) -> Value {
        self.b.use_var(self.sp)
    }

    fn change_top(&mut self, bytes: i64) {
        assert_eq!(bytes % 8, 0);
        self.stack_state.depth = self
            .stack_state
            .depth
            .checked_add_signed((bytes / 8) as isize)
            .expect("verified stack effect");
        self.stack_state.floor = self.stack_state.floor.min(self.stack_state.depth);
        let old = self.top();
        let new = self.b.ins().iadd_imm_s(old, bytes);
        self.b.def_var(self.sp, new);
    }

    fn push(&mut self, value: Value) {
        self.b
            .def_var(self.stack_vars[self.stack_state.depth], value);
        let number = self.b.ins().bitcast(types::F64, MemFlags::new(), value);
        self.b
            .def_var(self.stack_numbers[self.stack_state.depth], number);
        self.change_top(8);
    }

    /// All observers see the canonical owners before entry. The helper may execute user code,
    /// update eval/argument-visible bindings, collect, consume operands, or throw. Only
    /// proven private frame words can retain register copies across that boundary; no
    /// heap-property/environment/type assumption follows from preserving their identity.
    fn helper(&mut self, helper: Helper, immediate: u32) -> Value {
        self.publish_loop_fields();
        let writes = self
            .chunk
            .jit_ops()
            .get(self.pc)
            .map_or(SlotWrites::All, effects::checked_writes);
        self.observing_helper(
            helper,
            immediate,
            &self.frame_plan.live_after[self.pc],
            writes,
        )
    }

    fn observing_helper(
        &mut self,
        helper: Helper,
        immediate: u32,
        live: &[u16],
        writes: SlotWrites,
    ) -> Value {
        self.publish_locals();
        self.publish_stack(self.stack_state.depth);
        let flag = self.canonical_helper(helper, immediate);
        self.reload_locals(live, writes);
        flag
    }

    /// Only the shared unwind dispatcher may enter here without publishing: every predecessor
    /// has already published before its throwing helper, whose effects must not be overwritten.
    fn canonical_helper(&mut self, helper: Helper, immediate: u32) -> Value {
        let immediate = self.b.ins().iconst(types::I32, immediate as i64);
        self.canonical_helper_value(helper, immediate)
    }

    fn canonical_helper_value(&mut self, helper: Helper, immediate: Value) -> Value {
        if let Some(record) = self.diagnostics {
            record.emitted_helpers.set(record.emitted_helpers.get() + 1);
            let category = if std::ptr::fn_addr_eq(helper, bytecode::jit_unwind_pc as Helper) {
                9
            } else {
                self.chunk
                    .jit_ops()
                    .get(self.pc)
                    .map_or(9, diagnostics::category)
            };
            self.count_runtime(category, 1);
        }
        let callee = self.b.ins().iconst(types::I64, helper as usize as i64);
        let sp = self.top();
        let call = self
            .b
            .ins()
            .call_indirect(self.helper_sig, callee, &[self.ctx, immediate, sp]);
        let flag = self.b.inst_results(call)[0];
        let sp = self.load(self.ctx, offset_of!(JitCtx, final_sp) as i32);
        self.b.def_var(self.sp, sp);
        flag
    }

    fn checked(&mut self, helper: Helper, immediate: u32, next: Block) {
        let flag = self.helper(helper, immediate);
        let normal = self.b.create_block();
        self.b.ins().brif(flag, self.unwind, &[], normal, &[]);
        self.b.switch_to_block(normal);
        // Never load normal-result owners before testing the exceptional completion. A
        // throwing helper can have consumed more words than its installed handler saved.
        self.reload_stack(self.stack_plan.after[self.pc]);
        if self.loop_facts.is_some_and(|plan| {
            plan.observes && plan.contains(self.pc) && !plan.native_write(self.pc)
        }) && effects::checked_heap_write(&self.chunk.jit_ops()[self.pc])
        {
            self.validate_loop_fields(self.pc + 1);
        }
        self.jump_normal(next);
    }

    fn call(&mut self, pc: usize, argc: u16, cache: u32, with_this: bool, next: Block) {
        let entry = self
            .call_stubs
            .iter()
            .find(|stub| stub.matches(argc, with_this))
            .map(|stub| stub.entry());
        let Some(entry) = entry else {
            self.checked(call, pc as u32, next);
            return;
        };
        self.publish_loop_fields();
        self.publish_locals();
        self.publish_stack(self.stack_state.depth);
        let entry = self.b.ins().iconst(types::I64, entry as i64);
        self.count_runtime(diagnostics::SHARED_CALL, 1);
        let pc = self.b.ins().iconst(types::I32, pc as i64);
        let cache = self
            .b
            .ins()
            .iconst(types::I64, self.chunk.jit_call_cache_ptr(cache) as i64);
        let sp = self.top();
        let call = self
            .b
            .ins()
            .call_indirect(self.call_sig, entry, &[self.ctx, pc, sp, cache]);
        let flag = self.b.inst_results(call)[0];
        let sp = self.load(self.ctx, offset_of!(JitCtx, final_sp) as i32);
        self.b.def_var(self.sp, sp);
        self.reload_locals(&self.frame_plan.live_after[self.pc], SlotWrites::None);
        let normal = self.b.create_block();
        self.b.ins().brif(flag, self.unwind, &[], normal, &[]);
        self.b.switch_to_block(normal);
        self.reload_stack(self.stack_plan.after[self.pc]);
        if self
            .loop_facts
            .is_some_and(|plan| plan.observes && plan.contains(self.pc))
        {
            self.validate_loop_fields(self.pc + 1);
        }
        self.jump_normal(next);
    }

    fn slow(&mut self, pc: usize, block: Block, next: Block) {
        self.b.switch_to_block(block);
        self.b.set_cold_block(block);
        self.stack_state = self.stack_plan.at[pc];
        self.checked(exec, pc as u32, next);
    }

    /// Terminal, before-effect bailout. The VM finishes this SAME invocation, so neither
    /// the normal successor nor the native completion router may run afterwards.
    fn deopt_before(&mut self, pc: usize, error: Block) {
        self.publish_locals();
        self.publish_stack(self.stack_state.depth);
        let success = self.canonical_helper(bytecode::native_deopt::resume_before, pc as u32);
        self.b.ins().brif(success, self.ok, &[], error, &[]);
    }

    fn specialization_bailout(&mut self, pc: usize) {
        self.count_runtime(diagnostics::SPECIALIZATION_BAILOUT, 1);
        self.publish_locals();
        self.publish_stack(self.stack_state.depth);
        let success = self.canonical_helper(specialization_miss, pc as u32);
        self.b.ins().brif(success, self.ok, &[], self.error, &[]);
    }

    /// Numbers occupy the IEEE range through the canonical NaN, with either sign. All
    /// boxed tags are above it. No heap owner can enter an unboxed floating-point operation.
    fn number(&mut self, value: Value) -> Value {
        let high = self.b.ins().band_imm_s(value, 0x7fff_0000_0000_0000);
        self.b
            .ins()
            .icmp_imm_s(IntCC::UnsignedLessThanOrEqual, high, PACK_CANON_NAN as i64)
    }

    /// Immediate words need no retain/drop. Empty is copyable for destruction but is NOT
    /// a successful GetBindingValue; LoadLocal checks the TDZ separately.
    fn immediate(&mut self, value: Value) -> Value {
        let high = self.b.ins().band_imm_s(value, 0x7fff_0000_0000_0000);
        let positive = self.b.ins().icmp_imm_s(
            IntCC::UnsignedLessThanOrEqual,
            value,
            (PACK_BOOL | 1) as i64,
        );
        let number =
            self.b
                .ins()
                .icmp_imm_s(IntCC::UnsignedLessThanOrEqual, high, PACK_CANON_NAN as i64);
        self.b.ins().bor(positive, number)
    }

    fn packed_number(&mut self, number: Value) -> Value {
        let bits = self.b.ins().bitcast(types::I64, MemFlags::new(), number);
        let nan = self.b.ins().fcmp(FloatCC::Unordered, number, number);
        let canonical = self.b.ins().iconst(types::I64, PACK_CANON_NAN as i64);
        self.b.ins().select(nan, canonical, bits)
    }

    /// ECMA-262 e28783d5, ReturnStatement / TryStatement / IteratorClose: a return
    /// can transfer its value directly only when this body installs no handler.
    /// Caller handlers stay below the activation's floor. The empty-destination
    /// guard preserves displaced-owner destruction after failed tail calls.
    /// There is no observer between publishing remaining owners and the transfer;
    /// the shared exit restores strictness, residency and the exact operand top.
    fn direct_return(&mut self, value: bool) {
        let state = self.stack_state;
        if self.direct_returns && (!value || state.known() != 0) {
            let ret = self.load(self.ctx, offset_of!(JitCtx, ret) as i32);
            let empty = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, ret, PACK_UNDEFINED as i64);
            let fast = self.b.create_block();
            let slow = self.b.create_block();
            self.b.ins().brif(empty, fast, &[], slow, &[]);
            self.b.switch_to_block(fast);
            self.publish_locals();
            self.publish_stack(state.depth - usize::from(value));
            if value {
                let result = self.stack_read(1);
                self.store(result, self.ctx, offset_of!(JitCtx, ret) as i32);
                self.change_top(-8);
            }
            self.b.ins().jump(self.ok, &[]);
            self.b.switch_to_block(slow);
            self.b.set_cold_block(slow);
            self.stack_state = state;
        }
        if matches!(
            self.chunk.jit_ops().get(self.pc),
            Some(Op::Return | Op::ReturnBare)
        ) {
            let target = self.helper(bytecode::jit_complete_pc, self.pc as u32);
            self.b
                .ins()
                .brif(target, self.dispatch, &[target.into()], self.ok, &[]);
        } else {
            debug_assert!(!value);
            self.helper(return_value, 0);
            self.b.ins().jump(self.ok, &[]);
        }
    }

    fn numeric(&mut self, pc: usize, op: &Op, next: Block) {
        let right = self.stack_read(1);
        let left = self.stack_read(2);
        let inputs = self.input_types();
        let left_known = inputs.second.is_number();
        let right_known = inputs.top.is_number();
        #[cfg(test)]
        Self::count_value_proof(0, usize::from(left_known) + usize::from(right_known));
        let slow = if left_known && right_known {
            None
        } else {
            let both = match (left_known, right_known) {
                (true, false) => self.number(right),
                (false, true) => self.number(left),
                _ => {
                    let lnum = self.number(left);
                    let rnum = self.number(right);
                    self.b.ins().band(lnum, rnum)
                }
            };
            let fast = self.b.create_block();
            let slow = self.b.create_block();
            self.b.ins().brif(both, fast, &[], slow, &[]);
            self.b.switch_to_block(fast);
            Some(slow)
        };
        let left_word = left;
        let right_word = right;
        let left = self.stack_number(2);
        let right = self.stack_number(1);
        let mut numeric_result = None;
        let result = match op {
            Op::Add | Op::Sub | Op::Mul | Op::Div => {
                let number = match op {
                    Op::Add => self.b.ins().fadd(left, right),
                    Op::Sub => self.b.ins().fsub(left, right),
                    Op::Mul => self.b.ins().fmul(left, right),
                    _ => self.b.ins().fdiv(left, right),
                };
                numeric_result = Some(number);
                self.packed_number(number)
            }
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr => {
                let number = self.bitwise_number(op, left_word, right_word, left, right);
                numeric_result = Some(number);
                self.b.ins().bitcast(types::I64, MemFlags::new(), number)
            }
            _ => {
                let cc = match op {
                    Op::Lt => FloatCC::LessThan,
                    Op::Le => FloatCC::LessThanOrEqual,
                    Op::Gt => FloatCC::GreaterThan,
                    Op::Ge => FloatCC::GreaterThanOrEqual,
                    Op::StrictEq => FloatCC::Equal,
                    Op::StrictNotEq => FloatCC::NotEqual,
                    _ => unreachable!("numeric lowering selected by opcode"),
                };
                let bit = self.b.ins().fcmp(cc, left, right);
                let bit = self.b.ins().uextend(types::I64, bit);
                self.b.ins().bor_imm_s(bit, PACK_BOOL as i64)
            }
        };
        self.stack_write(2, result);
        if let Some(number) = numeric_result {
            self.b
                .def_var(self.stack_numbers[self.stack_state.depth - 2], number);
        }
        self.change_top(-8);
        self.jump_normal(next);
        if let Some(slow) = slow {
            self.slow(pc, slow, next);
        }
    }

    fn poll(&mut self, layout: &InterpLayout) {
        let live = self.load(self.ctx, offset_of!(JitCtx, live_objects) as i32);
        let live = self.load(live, 0);
        let limit = self.load(self.interp, layout.gc_next as i32);
        let pressure = self.b.ins().icmp(IntCC::SignedGreaterThan, live, limit);
        let tick = self.b.ins().load(
            types::I32,
            MemFlags::trusted(),
            self.interp,
            layout.interrupt_poll_tick as i32,
        );
        let tick = self.b.ins().iadd_imm_s(tick, 1);
        self.store(tick, self.interp, layout.interrupt_poll_tick as i32);
        let divided = self.b.ins().band_imm_s(tick, 0x3fff);
        let due = self.b.ins().icmp_imm_s(IntCC::Equal, divided, 0);
        let due = self.b.ins().bor(pressure, due);
        let slow = self.b.create_block();
        let next = self.b.create_block();
        self.b.ins().brif(due, slow, &[], next, &[]);
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        // This runs BEFORE the opcode. The non-moving collector/interrupt check
        // observes canonical roots without replacing private frame words. The
        // all-clobber ablation must reload this opcode's inputs, not successors.
        self.publish_loop_fields();
        let flag = self.observing_helper(
            interrupt,
            0,
            &self.frame_plan.live_at[self.pc],
            SlotWrites::None,
        );
        let normal = self.b.create_block();
        self.b.ins().brif(flag, self.unwind, &[], normal, &[]);
        self.b.switch_to_block(normal);
        self.reload_stack(self.stack_plan.at[self.pc]);
        if self.loop_facts.is_some_and(|plan| plan.contains(self.pc)) {
            self.validate_loop_fields(self.pc);
        }
        self.b.ins().jump(next, &[]);
        self.b.switch_to_block(next);
    }

    fn conditional_jump(&mut self, op: &Op, target: Block, next: Block) {
        let take_on_false = matches!(op, Op::JumpIfFalse(_) | Op::JumpIfFalsePeek(_));
        let (yes, no) = if take_on_false {
            (next, target)
        } else {
            (target, next)
        };
        let yes = self.loop_exit(yes);
        let no = self.loop_exit(no);
        let slow;
        if self.stack_state.known() > 0 {
            let value = self.stack_read(1);
            if matches!(op, Op::JumpIfNotNullishPeek(_)) {
                // CoalesceExpression tests Undefined/Null identity, not ToBoolean/IsHTMLDDA.
                let defined =
                    self.b
                        .ins()
                        .icmp_imm_s(IntCC::NotEqual, value, PACK_UNDEFINED as i64);
                let nonnull = self
                    .b
                    .ins()
                    .icmp_imm_s(IntCC::NotEqual, value, PACK_NULL as i64);
                let keep = self.b.ins().band(defined, nonnull);
                self.finish_normal();
                self.b.ins().brif(keep, yes, &[], no, &[]);
                return;
            } else {
                let known = self.input_types().top;
                let scalar = known.is_copyable() && known.initialized();
                slow = if scalar {
                    #[cfg(test)]
                    Self::count_value_proof(3, 1);
                    None
                } else {
                    let slow = self.b.create_block();
                    let immediate = self.immediate(value);
                    let fast = self.b.create_block();
                    self.b.ins().brif(immediate, fast, &[], slow, &[]);
                    self.b.switch_to_block(fast);
                    Some(slow)
                };
                let truthy = if known.is_boolean() {
                    self.b
                        .ins()
                        .icmp_imm_s(IntCC::Equal, value, (PACK_BOOL | 1) as i64)
                } else if known.is_nullish() {
                    self.b.ins().iconst(types::I8, 0)
                } else {
                    // OrderedNotEqual intentionally rejects NaN and both zero signs.
                    let number = self.b.ins().bitcast(types::F64, MemFlags::new(), value);
                    let zero = self.b.ins().f64const(0.0);
                    let nonzero = self.b.ins().fcmp(FloatCC::OrderedNotEqual, number, zero);
                    if known.is_number() {
                        nonzero
                    } else {
                        let boolean =
                            self.b
                                .ins()
                                .icmp_imm_s(IntCC::Equal, value, (PACK_BOOL | 1) as i64);
                        self.b.ins().bor(nonzero, boolean)
                    }
                };
                if matches!(op, Op::JumpIfFalse(_)) {
                    self.change_top(-8);
                }
                self.finish_normal();
                self.b.ins().brif(truthy, yes, &[], no, &[]);
            }
        } else {
            // The opaque prefix may be shorter than its logical CFG labels. Only the
            // checked operation may consume/read it and establish the normal result suffix.
            let checked = self.b.create_block();
            self.b.ins().jump(checked, &[]);
            slow = Some(checked);
        }
        let Some(slow) = slow else { return };
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        self.stack_state = self.stack_plan.at[self.pc];
        let mode = match op {
            Op::JumpIfFalse(_) => super::COND_POP_TRUTHY,
            Op::JumpIfNotNullishPeek(_) => super::COND_PEEK_NOT_NULLISH,
            _ => super::COND_PEEK_TRUTHY,
        };
        let flag = self.helper(condition, mode);
        self.reload_stack(self.stack_plan.after[self.pc]);
        self.finish_normal();
        self.b.ins().brif(flag, yes, &[], no, &[]);
    }

    fn lower(&mut self, pc: usize, op: &Op) {
        let next = self.pcs[pc + 1];
        if self.loop_field_operation(pc, op, next) {
            return;
        }
        if self.loop_facts.is_some_and(|plan| plan.native_write(pc)) {
            self.publish_loop_fields();
        }
        if !self.native_heap {
            let checked = match op {
                Op::SetProp(..)
                | Op::SetPropDrop(..)
                | Op::SetPropThisDrop(..)
                | Op::SetPropLocalDrop(..) => Some(set_property as Helper),
                Op::GetElem | Op::GetElemLocal(_) | Op::GetMethodElem => {
                    Some(get_element as Helper)
                }
                Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(_) | Op::SetElemLocalDrop(_) => {
                    Some(set_element as Helper)
                }
                _ => None,
            };
            if let Some(helper) = checked {
                self.checked(helper, pc as u32, next);
                return;
            }
        }
        let inspected = match op {
            Op::Dup
            | Op::Pop
            | Op::StoreLocal(_)
            | Op::GetProp(..)
            | Op::GetMethod(..)
            | Op::SetPropThisDrop(..)
            | Op::SetPropLocalDrop(..)
            | Op::GetElemLocal(_) => 1,
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::Shl
            | Op::Shr
            | Op::UShr
            | Op::Lt
            | Op::Le
            | Op::Gt
            | Op::Ge
            | Op::StrictEq
            | Op::StrictNotEq
            | Op::SetProp(..)
            | Op::SetPropDrop(..)
            | Op::GetElem
            | Op::GetMethodElem
            | Op::SetElemLocal(_)
            | Op::SetElemLocalDrop(_) => 2,
            Op::SetElem | Op::SetElemDrop => 3,
            _ => 0,
        };
        if inspected > self.stack_state.known() {
            self.checked(exec, pc as u32, next);
            return;
        }
        match op {
            Op::Const(k) if self.chunk.jit_const_copyable(*k) => {
                let bits = self
                    .chunk
                    .jit_const_packed_bits(*k)
                    .expect("copyable constant");
                let value = self.b.ins().iconst(types::I64, bits as i64);
                self.push(value);
                self.jump_normal(next);
            }
            Op::Undef => {
                let value = self.b.ins().iconst(types::I64, PACK_UNDEFINED as i64);
                self.push(value);
                self.jump_normal(next);
            }
            Op::LoadLocal(_) | Op::Dup | Op::Pop | Op::StoreLocal(_) => {
                self.ownership(pc, op, next)
            }
            Op::LoadName(..) | Op::LoadNameForCall(..) | Op::LoadCap(_) => {
                self.name_read(pc, op, next)
            }
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::Shl
            | Op::Shr
            | Op::UShr
            | Op::Lt
            | Op::Le
            | Op::Gt
            | Op::Ge
            | Op::StrictEq
            | Op::StrictNotEq => self.numeric(pc, op, next),
            Op::UpdateLocal(slot, kind) => {
                let old = self.local(*slot);
                let slow = if self.input_types().local.is_number() {
                    #[cfg(test)]
                    Self::count_value_proof(1, 1);
                    None
                } else {
                    let number = self.number(old);
                    let fast = self.b.create_block();
                    let slow = self.b.create_block();
                    self.b.ins().brif(number, fast, &[], slow, &[]);
                    self.b.switch_to_block(fast);
                    Some(slow)
                };
                let old_number = self.b.ins().bitcast(types::F64, MemFlags::new(), old);
                let one = self.b.ins().f64const(1.0);
                let updated = if matches!(
                    kind,
                    UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
                ) {
                    self.b.ins().fsub(old_number, one)
                } else {
                    self.b.ins().fadd(old_number, one)
                };
                let updated = self.packed_number(updated);
                self.set_local(*slot, updated);
                match kind {
                    UpdKind::PreInc | UpdKind::PreDec => self.push(updated),
                    UpdKind::PostInc | UpdKind::PostDec => self.push(old),
                    UpdKind::IncDiscard | UpdKind::DecDiscard => {}
                }
                self.jump_normal(next);
                if let Some(slow) = slow {
                    self.slow(pc, slow, next);
                }
            }
            Op::Jump(target) | Op::InlineGuard(_, target) => {
                // Spliced baseline inlining has a complete original-call fallback. This
                // backend takes that path until its own effect/frame-aware inliner lands.
                self.jump_normal(self.pcs[*target as usize]);
            }
            Op::JumpIfFalse(target)
            | Op::JumpIfFalsePeek(target)
            | Op::JumpIfTruePeek(target)
            | Op::JumpIfNotNullishPeek(target) => {
                let target = self.pcs[*target as usize];
                self.conditional_jump(op, target, next);
            }
            Op::Return | Op::ReturnBare => self.direct_return(matches!(op, Op::Return)),
            Op::ResumeReturn | Op::AbruptJump(..) | Op::ResumeJump => {
                let target = self.helper(bytecode::jit_complete_pc, pc as u32);
                self.b
                    .ins()
                    .brif(target, self.dispatch, &[target.into()], self.ok, &[]);
            }
            Op::ReturnUndef => self.direct_return(false),
            Op::PushHandler(_) | Op::PushFinally(..) | Op::PushIterator(..) => {
                self.helper(push_handler, pc as u32);
                self.reload_stack(self.stack_plan.after[pc]);
                self.jump_normal(next);
            }
            Op::PopHandler => {
                self.helper(pop_handler, 0);
                self.reload_stack(self.stack_plan.after[pc]);
                self.jump_normal(next);
            }
            Op::Throw => {
                self.helper(exec, pc as u32);
                self.b.ins().jump(self.unwind, &[]);
            }
            Op::Call(argc, cache) | Op::CallWithThis(argc, cache) => {
                self.call(pc, *argc, *cache, matches!(op, Op::CallWithThis(..)), next);
            }
            Op::GetProp(..) | Op::GetPropThis(..) | Op::GetPropLocal(..) | Op::GetMethod(..) => {
                self.property_read(pc, op, next)
            }
            Op::SetProp(..)
            | Op::SetPropDrop(..)
            | Op::SetPropThisDrop(..)
            | Op::SetPropLocalDrop(..) => self.property_write(pc, op, next),
            Op::GetElem | Op::GetElemLocal(_) | Op::GetMethodElem => {
                self.element_read(pc, op, next)
            }
            Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(_) | Op::SetElemLocalDrop(_) => {
                self.element_write(pc, op, next)
            }
            Op::MakeArray(_) => self.checked(make_array, pc as u32, next),
            Op::MakeObject(..) => self.checked(make_object, pc as u32, next),
            Op::New(argc, _) => self.checked(construct, pc as u32 | (u32::from(*argc) << 16), next),
            _ => self.checked(exec, pc as u32, next),
        }
    }
}

fn compile_checked(
    chunk: &Chunk,
    values: &crate::value::JitLayout,
    layout: &InterpLayout,
) -> Result<JitCode, CompileError> {
    // Explicit diagnostic stress point, not a speculation/profitability policy. There is no
    // extra generated branch or call when unset. Tests select PCs directly, without env races.
    let deopt_at = std::env::var("TRUST_JS_OPT_JIT_DEOPT_AT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&pc| pc <= chunk.jit_ops().len());
    compile_with_deopt(chunk, values, layout, deopt_at)
}

fn compile_with_deopt(
    chunk: &Chunk,
    values: &crate::value::JitLayout,
    layout: &InterpLayout,
    deopt_at: Option<usize>,
) -> Result<JitCode, CompileError> {
    let instructions = if chunk.optimizing_candidate {
        chunk
            .jit_ops()
            .len()
            .saturating_mul(40)
            .saturating_add(names::instruction_allowance(chunk))
            .min(16_384)
    } else {
        262_144
    };
    compile_with_limits(chunk, values, layout, deopt_at, instructions, 65_536)
}

// Each native body pins these Cells through its JitCode. They are engine-thread local,
// and neither this write nor the optional directory can keep a JavaScript object alive.
fn emit_diagnostic_count(
    b: &mut FunctionBuilder<'_>,
    record: &diagnostics::Code,
    counter: usize,
    amount: usize,
) {
    if amount == 0 || !record.runtime_counters {
        return;
    }
    let address = b.ins().iconst(types::I64, record.counter(counter) as i64);
    let previous = b.ins().load(types::I64, MemFlags::trusted(), address, 0);
    let next = b.ins().iadd_imm_s(previous, amount as i64);
    b.ins().store(MemFlags::trusted(), next, address, 0);
}

/// Bound expanded IR as well as source bytecode. A single guarded heap opcode can generate
/// many blocks; the bytecode limit alone cannot bound Cranelift's memory/work. Exhaustion
/// declines only this compiler attempt, before the function's executable allocation/publication, and
/// the established JIT/VM retains the entire function's semantics. These are NOT heap or
/// JavaScript program-size limits. Explicit arguments let tests exercise refusal cheaply.
fn compile_with_limits(
    chunk: &Chunk,
    values: &crate::value::JitLayout,
    layout: &InterpLayout,
    deopt_at: Option<usize>,
    instruction_limit: usize,
    block_limit: usize,
) -> Result<JitCode, CompileError> {
    compile_entry_with_limits(
        chunk,
        values,
        layout,
        deopt_at,
        instruction_limit,
        block_limit,
        None,
    )
}

fn compile_entry_with_limits(
    chunk: &Chunk,
    values: &crate::value::JitLayout,
    layout: &InterpLayout,
    deopt_at: Option<usize>,
    instruction_limit: usize,
    block_limit: usize,
    continuation: Option<(usize, &[(u16, value_facts::Types)], &[u32])>,
) -> Result<JitCode, CompileError> {
    let ops = chunk.jit_ops();
    if !layout.valid
        || [layout.strict, layout.gc_next, layout.interrupt_poll_tick]
            .into_iter()
            .any(|o| o > i32::MAX as usize)
        || ops.is_empty()
        || ops.len() > 65_535
        || chunk.jit_is_resumable()
        || chunk.jit_detailed_feedback_enabled()
        || ((!chunk.optimizing_inputs.is_empty() || !chunk.optimizing_properties.is_empty())
            && !bytecode::native_deopt::supported(chunk))
        || deopt_at.is_some_and(|pc| pc > ops.len() || !bytecode::native_deopt::supported(chunk))
        || ops.iter().any(|op| {
            matches!(
                op,
                Op::FragmentExit(_)
                    | Op::Await
                    | Op::Yield
                    | Op::YieldStar
                    | Op::AsyncIterStepL(..)
                    | Op::AsyncIterResumeL(..)
                    | Op::AsyncIterCloseL(..)
                    | Op::DisposeNormal
                    | Op::DisposeThrow
                    | Op::DisposeReturn
                    | Op::DisposeBareReturn
                    | Op::DisposeResumeReturn
                    | Op::DisposeJump
            )
        })
    {
        return Err(CompileError::Unsupported);
    }
    // The entry, every bytecode boundary, unwind/dispatch and two exits already need
    // these blocks, before expanding even one opcode. Refuse impossible budgets before
    // analyses or shared call-stub preparation; an ordinary body always needs instructions.
    if instruction_limit == 0 || ops.len().saturating_add(6) > block_limit {
        return Err(CompileError::LoweringBudget);
    }
    let diagnostic = diagnostics::Code::new(chunk);
    let analysis_started = diagnostic.as_ref().map(|_| std::time::Instant::now());
    let cfg = Cfg::build(chunk).map_err(CompileError::ControlFlow)?;
    let entry_pc = continuation.map_or(0, |(pc, _, _)| pc);
    if continuation.is_some()
        && (!bytecode::native_deopt::supported(chunk)
            || cfg
                .block_at(entry_pc)
                .is_none_or(|block| cfg.blocks()[block.0 as usize].start != entry_pc)
            || cfg.stack_depth_at(entry_pc).is_none())
    {
        return Err(CompileError::Unsupported);
    }
    let local_effects = std::env::var("TRUST_JS_OPT_JIT_LOCAL_EFFECTS").as_deref() != Ok("0");
    let frame =
        frame::FramePlan::build(chunk, &cfg, local_effects).ok_or(CompileError::Unsupported)?;
    let stack_plan = stack::StackPlan::build(chunk, &cfg).ok_or(CompileError::Unsupported)?;
    // A failed/budget-exhausted value proof retains all dynamic guards, not a less
    // capable function or a different execution tier. The switch also skips analysis cost.
    let use_facts = std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() != Ok("0");
    let specialization = (use_facts && continuation.is_none())
        .then(|| {
            specialization::Plan::build(
                chunk,
                &cfg,
                &stack_plan,
                &chunk.optimizing_inputs,
                &chunk.optimizing_properties,
            )
        })
        .flatten();
    let generic_values = (use_facts && specialization.is_none() && continuation.is_none())
        .then(|| value_facts::ValuePlan::build(chunk, &cfg, &stack_plan))
        .flatten();
    let continuation_values = match continuation {
        Some((pc, inputs, _)) => Some(
            value_facts::ValuePlan::at_entry(chunk, &cfg, &stack_plan, pc, inputs)
                .ok_or(CompileError::Unsupported)?,
        ),
        None => None,
    };
    let loop_facts = continuation.and_then(|(pc, inputs, properties)| {
        (property::supported(values) && deopt_at.is_none())
            .then(|| loop_facts::Plan::build(chunk, &cfg, &stack_plan, pc, inputs, properties))
            .flatten()
    });
    if continuation.is_some()
        && std::env::var("TRUST_JS_OPT_JIT_LOOP_FACTS").as_deref() == Ok("only")
        && loop_facts.is_none()
    {
        return Err(CompileError::Unsupported);
    }
    let value_plan = loop_facts
        .as_ref()
        .map(|plan| &plan.values)
        .or_else(|| specialization.as_ref().map(|plan| &plan.values))
        .or(generic_values.as_ref())
        .or(continuation_values.as_ref());
    if chunk.optimizing_candidate && specialization.is_none() {
        return Err(CompileError::Unsupported);
    }
    if std::env::var_os("TRUST_JS_OPT_JIT_LOG").is_some() {
        if let Some(plan) = &loop_facts {
            eprintln!(
                "[optimizing-loop-facts] entry={entry_pc} fields={}",
                plan.field_count()
            );
        }
        if let Some(plan) = &specialization {
            eprintln!("[optimizing-specialization] ops={} entry_guards={} saved_tests={} opportunities={:?}",
                ops.len(), plan.inputs.len(), plan.saved_tests,
                plan.opportunities.iter().map(|item| (item.pc, item.site.map(|id| id.index()), item.removed_tests)).collect::<Vec<_>>());
        }
    }
    if std::env::var_os("TRUST_JS_OPT_JIT_DUMP").is_some() {
        for (pc, op) in ops.iter().enumerate() {
            eprintln!("[optimizing-jit] {pc}: {op:?}");
        }
    }
    let isa = isa().ok_or(CompileError::Unsupported)?;
    let call_stubs = super::call_stub::prepare(chunk, values, layout);
    if let (Some(record), Some(started)) = (&diagnostic, analysis_started) {
        record.analysis_ns.set(diagnostics::nanos(started));
    }
    let lowering_started = diagnostic.as_ref().map(|_| std::time::Instant::now());
    let convention = isa.default_call_conv();
    let mut signature = Signature::new(convention);
    signature.params.push(AbiParam::new(types::I64));
    signature.returns.push(AbiParam::new(types::I64));
    let mut context = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        signature,
    ));
    let mut helper_sig = Signature::new(convention);
    helper_sig.params.extend([
        AbiParam::new(types::I64),
        AbiParam::new(types::I32),
        AbiParam::new(types::I64),
    ]);
    helper_sig.returns.push(AbiParam::new(types::I64));
    let mut call_sig = helper_sig.clone();
    call_sig.params.push(AbiParam::new(types::I64));
    let call_sig = context.func.import_signature(call_sig);
    let helper_sig = context.func.import_signature(helper_sig);
    let residency = Box::<CodeResidency>::default();
    let mut builder_context = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut context.func, &mut builder_context);
        let entry = b.create_block();
        let pcs: Vec<_> = (0..=ops.len()).map(|_| b.create_block()).collect();
        let unwind = b.create_block();
        let dispatch = b.create_block();
        b.append_block_param(dispatch, types::I64);
        let ok = b.create_block();
        let error = b.create_block();
        b.switch_to_block(entry);
        b.append_block_params_for_function_params(entry);
        if let Some(record) = &diagnostic {
            emit_diagnostic_count(&mut b, record, 0, 1);
        }
        let ctx = b.block_params(entry)[0];
        let interp = b.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx,
            offset_of!(JitCtx, interp) as i32,
        );
        let slots = b.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx,
            offset_of!(JitCtx, slots) as i32,
        );
        let stack = b.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx,
            offset_of!(JitCtx, stack_base) as i32,
        );
        let sp = b.declare_var(types::I64);
        let initial_sp = if continuation.is_some() {
            b.ins()
                .iadd_imm_s(stack, (stack_plan.at[entry_pc].depth * 8) as i64)
        } else {
            stack
        };
        b.def_var(sp, initial_sp);
        // Protect direct-call entries against nested compilation/reclamation before any helper.
        let owner = b
            .ins()
            .iconst(types::I64, (&*residency as *const CodeResidency) as i64);
        let active = b.ins().load(
            types::I64,
            MemFlags::trusted(),
            owner,
            offset_of!(CodeResidency, active) as i32,
        );
        let active = b.ins().iadd_imm_s(active, 1);
        b.ins().store(
            MemFlags::trusted(),
            active,
            owner,
            offset_of!(CodeResidency, active) as i32,
        );
        let one = b.ins().iconst(types::I8, 1);
        b.ins().store(
            MemFlags::trusted(),
            one,
            owner,
            offset_of!(CodeResidency, referenced) as i32,
        );
        let saved_strict =
            b.ins()
                .load(types::I8, MemFlags::trusted(), interp, layout.strict as i32);
        let strict = b
            .ins()
            .iconst(types::I8, u8::from(chunk.jit_is_strict()) as i64);
        b.ins()
            .store(MemFlags::trusted(), strict, interp, layout.strict as i32);
        let mut locals = vec![None; chunk.jit_frame().1];
        for &slot in &frame.tracked {
            let var = b.declare_var(types::I64);
            let value = b
                .ins()
                .load(types::I64, MemFlags::trusted(), slots, i32::from(slot) * 8);
            b.def_var(var, value);
            locals[slot as usize] = Some(var);
        }
        let stack_vars = (0..stack_plan.capacity)
            .map(|_| b.declare_var(types::I64))
            .collect();
        let stack_numbers = (0..stack_plan.capacity)
            .map(|_| b.declare_var(types::F64))
            .collect();
        let mut lower = Lowering {
            b,
            chunk,
            values,
            owners_valid: crate::value::jit_packed_owners_supported(values),
            ctx,
            interp,
            slots,
            locals,
            frame_plan: &frame,
            value_plan,
            property_results: specialization
                .as_ref()
                .map_or(&[], |plan| plan.properties.as_slice()),
            loop_facts: loop_facts.as_ref(),
            loop_field_addresses: Vec::new(),
            loop_field_values: Vec::new(),
            loop_validation: None,
            loop_validation_returns: Vec::new(),
            loop_scalars: std::env::var("TRUST_JS_OPT_JIT_LOOP_SCALARS").as_deref() != Ok("0"),
            sparse_reloads: std::env::var("TRUST_JS_OPT_JIT_SPARSE_RELOADS").as_deref() != Ok("0"),
            local_effects,
            native_heap: std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0"),
            native_creation: std::env::var("TRUST_JS_OPT_JIT_CREATION").as_deref() != Ok("0"),
            direct_returns: !ops.iter().any(|op| {
                matches!(
                    op,
                    Op::PushHandler(_) | Op::PushFinally(..) | Op::PushIterator(..)
                )
            }),
            dirty: &[],
            stack_plan: &stack_plan,
            stack_vars,
            stack_numbers,
            stack_state: stack_plan.at[entry_pc],
            pc: entry_pc,
            sp,
            helper_sig,
            call_sig,
            call_stubs: &call_stubs,
            diagnostics: diagnostic.as_deref(),
            unwind,
            ok,
            error,
            dispatch,
            pcs,
        };
        if continuation.is_some() {
            lower.reload_stack(stack_plan.at[entry_pc]);
        }
        if let Some(plan) = &loop_facts {
            lower.loop_field_addresses = (0..plan.field_count())
                .map(|_| lower.b.declare_var(types::I64))
                .collect();
            lower.loop_field_values = (0..plan.field_count())
                .map(|_| lower.b.declare_var(types::I64))
                .collect();
        }
        let entry_inputs = continuation
            .map(|(_, inputs, _)| inputs)
            .or_else(|| specialization.as_ref().map(|plan| plan.inputs.as_slice()));
        if let Some(inputs) = entry_inputs.filter(|inputs| !inputs.is_empty()) {
            let miss = lower.b.create_block();
            for &(slot, class) in inputs {
                let value = lower.local(slot);
                let matches = if class.is_number() {
                    lower.number(value)
                } else {
                    debug_assert!(class.is_boolean());
                    let tag = lower.b.ins().band_imm_s(value, -2);
                    lower
                        .b
                        .ins()
                        .icmp_imm_s(IntCC::Equal, tag, PACK_BOOL as i64)
                };
                let next = lower.b.create_block();
                lower.b.ins().brif(matches, next, &[], miss, &[]);
                lower.b.switch_to_block(next);
            }
            lower.validate_loop_fields(entry_pc);
            lower.b.ins().jump(lower.pcs[entry_pc], &[]);
            lower.b.switch_to_block(miss);
            lower.b.set_cold_block(miss);
            // No bytecode, getter, coercion or owner transfer has happened. Preserve the
            // actual inputs and finish this invocation once through its ordinary VM.
            if continuation.is_some() {
                lower.publish_locals();
                lower.publish_stack(lower.stack_state.depth);
                let success = lower.canonical_helper(loop_entry::miss, entry_pc as u32);
                lower.b.ins().brif(success, lower.ok, &[], error, &[]);
            } else {
                lower.specialization_bailout(0);
            }
        } else {
            lower.validate_loop_fields(entry_pc);
            lower.b.ins().jump(lower.pcs[entry_pc], &[]);
        }
        let mut poll = vec![false; ops.len()];
        for (pc, op) in ops.iter().enumerate() {
            if let Some(target) = crate::jit_ir::jump_target(op) {
                if target <= pc {
                    poll[target] = true;
                }
            }
        }
        for (pc, op) in ops.iter().enumerate() {
            if lower.b.func.dfg.num_insts() > instruction_limit
                || lower.b.func.dfg.num_blocks() > block_limit
            {
                return Err(CompileError::LoweringBudget);
            }
            lower.b.switch_to_block(lower.pcs[pc]);
            if !stack_plan.reachable[pc] {
                // No invented operands in dead bytecode. In particular, no unchecked reads
                // from zero-depth synthetic stack states to satisfy the SSA builder.
                lower.b.ins().jump(error, &[]);
                continue;
            }
            lower.pc = pc;
            lower.stack_state = stack_plan.at[pc];
            lower.dirty = &frame.dirty_at[pc];
            if poll[pc] {
                lower.poll(layout);
            }
            if deopt_at == Some(pc) {
                lower.deopt_before(pc, error);
            } else {
                lower.lower(pc, op);
            }
        }
        lower.b.switch_to_block(lower.pcs[ops.len()]);
        if stack_plan.reachable[ops.len()] {
            lower.pc = ops.len();
            lower.stack_state = stack_plan.at[ops.len()];
            lower.dirty = &frame.written;
            if deopt_at == Some(ops.len()) {
                lower.deopt_before(ops.len(), error);
            } else {
                lower.direct_return(false);
            }
        } else {
            lower.b.ins().jump(error, &[]);
        }
        lower.emit_loop_validation();
        lower.b.switch_to_block(unwind);
        lower.b.set_cold_block(unwind);
        let target = lower.canonical_helper(bytecode::jit_unwind_pc, 0);
        lower
            .b
            .ins()
            .brif(target, dispatch, &[target.into()], error, &[]);
        lower.b.switch_to_block(dispatch);
        let target = lower.b.block_params(dispatch)[0];
        let mut switch = Switch::new();
        let mut destinations = std::collections::BTreeSet::new();
        for root in cfg.handler_roots() {
            destinations.insert(cfg.blocks()[root.target.0 as usize].start);
        }
        for op in ops {
            if let Op::AbruptJump(target, _) = op {
                destinations.insert(*target as usize);
            }
        }
        let destinations: Vec<_> = destinations
            .into_iter()
            .map(|pc| {
                let landing = lower.b.create_block();
                switch.set_entry(pc as u128 + 1, landing);
                (pc, landing)
            })
            .collect();
        switch.emit(&mut lower.b, target, error);
        for (pc, landing) in destinations {
            lower.b.switch_to_block(landing);
            // Both throw routing and normal return/break/continue finalizers
            // can arrive here. Their canonical frame, not stale predecessor
            // SSA variables, supplies precisely the destination's register uses.
            lower.reload_locals(&frame.live_at[pc], SlotWrites::All);
            // Completion helpers returned the exact sp. Load only the proven suffix,
            // independent of whether the restored prefix is shorter than its CFG depth.
            lower.reload_stack(stack_plan.at[pc]);
            lower.b.ins().jump(lower.pcs[pc], &[]);
        }
        for (block, success) in [(ok, 1), (error, 0)] {
            lower.b.switch_to_block(block);
            let sp = lower.top();
            lower.store(sp, ctx, offset_of!(JitCtx, final_sp) as i32);
            lower.store(saved_strict, interp, layout.strict as i32);
            let active = lower.load(owner, offset_of!(CodeResidency, active) as i32);
            let active = lower.b.ins().iadd_imm_s(active, -1);
            lower.store(active, owner, offset_of!(CodeResidency, active) as i32);
            let success = lower.b.ins().iconst(types::I64, success);
            lower.b.ins().return_(&[success]);
        }
        if lower.b.func.dfg.num_insts() > instruction_limit
            || lower.b.func.dfg.num_blocks() > block_limit
        {
            return Err(CompileError::LoweringBudget);
        }
        lower.b.seal_all_blocks();
        lower.b.finalize(isa.frontend_config());
    }
    // SSA sealing/finalization can add instructions of its own. The backend must never
    // receive IR above the stated bound just because that work followed the final opcode.
    if context.func.dfg.num_insts() > instruction_limit
        || context.func.dfg.num_blocks() > block_limit
    {
        return Err(CompileError::LoweringBudget);
    }
    if let (Some(record), Some(started)) = (&diagnostic, lowering_started) {
        record.lowering_ns.set(diagnostics::nanos(started));
        record.ir_instructions.set(context.func.dfg.num_insts());
        record.ir_blocks.set(context.func.dfg.num_blocks());
    }
    if std::env::var_os("TRUST_JS_OPT_JIT_DUMP").is_some() {
        eprintln!("{}", context.func.display());
    }
    let passes = diagnostic.as_deref().map(diagnostics::passes::Scope::start);
    let codegen_started = diagnostic.as_ref().map(|_| std::time::Instant::now());
    let result = context.compile(&**isa, &mut Default::default());
    if let (Some(record), Some(started)) = (&diagnostic, codegen_started) {
        record.codegen_ns.set(diagnostics::nanos(started));
    }
    drop(passes);
    let code = result.map_err(|e| CompileError::Codegen(format!("{e:?}")))?;
    let finish_started = diagnostic.as_ref().map(|_| std::time::Instant::now());
    // All runtime calls are explicit indirect calls. Refuse any unexpected libcall/relocation
    // before allocating executable memory; never publish an unlinked buffer.
    if !code.buffer.relocs().is_empty() {
        return Err(CompileError::Relocation);
    }
    let executable =
        ExecutableBuffer::from_bytes(code.code_buffer()).ok_or(CompileError::Capacity)?;
    if let (Some(record), Some(started)) = (&diagnostic, finish_started) {
        record.native_bytes.set(executable.len());
        record.finish_ns.set(diagnostics::nanos(started));
        record.compiled();
    }
    Ok(JitCode {
        mem: executable.as_ptr() as *mut u8,
        len: executable.len(),
        pc_offsets: Vec::new(),
        max_stack: cfg.jit_stack_capacity(),
        needs_global: ops.iter().any(|op| {
            matches!(
                op,
                Op::LoadName(..)
                    | Op::LoadNameForCall(..)
                    | Op::LoadNameIn(..)
                    | Op::LoadNameForCallIn(..)
            )
        }),
        resume_depths: Vec::new(),
        entry_kind: if continuation.is_some() {
            NativeEntryKind::OptimizingContinuation
        } else {
            NativeEntryKind::FreshFrame
        },
        osr_entry_depths: Vec::new(),
        executable,
        residency,
        call_stubs,
        optimizing_diagnostics: diagnostic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value as JsValue;
    use crate::{bytecode::Tier, Completion, Engine};

    #[test]
    fn hot_sampling_skips_bodies_without_a_semantic_feedback_source() {
        const CHILD: &str = "TRUST_JS_TEST_OPT_STATIC_ADMISSION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "jit::optimizing::tests::hot_sampling_skips_bodies_without_a_semantic_feedback_source",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("TRUST_JS_OPT_JIT", "hot")
                .env("TRUST_JS_OPT_JIT_HOT_AT", "8")
                .env("TRUST_JS_INLINE_AT", "0")
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
        assert_eq!(mode(), Mode::Hot);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let result = engine
            .eval(
                r#"
            var calls = 0;
            function factory(x) {
                return {get value() { calls++; return x; }, set value(v) { x = v; }};
            }
            var object = factory(3);
            var captured = Object.getOwnPropertyDescriptor(object, 'value').get;
            function unused(x) { return calls; }
            function scalar(x) { return x*x+x*x; }
            function field(o) { return o.x*o.x+o.x*o.x; }
            function receiver() { return this.x*this.x+this.x*this.x; }
            var total = 0;
            for (var n = 0; n < 128; n++) {
                total += object.value;
                unused(n); scalar(n); field({x:n}); receiver.call({x:n});
            }
            object.value = 7;
            [total, calls, object.value].join('|');
        "#,
                false,
            )
            .unwrap();
        assert!(matches!(result, Completion::Value(ref value) if value == "384|128|7"));
        for (name, expected) in [
            ("captured", false),
            ("unused", false),
            ("scalar", true),
            ("field", true),
            ("receiver", true),
        ] {
            let env = engine.interp.global_env.clone();
            let value = engine.interp.get_var(name, &env).ok().unwrap();
            let function = {
                let object = value.as_obj().unwrap().borrow();
                let crate::value::Callable::User(user) = &object.call else {
                    panic!("user function");
                };
                user.func.clone()
            };
            let primary = function.code.get().and_then(Option::as_ref).unwrap();
            assert_eq!(has_sampling_source(primary), expected, "{name}");
            assert_eq!(
                primary.optimizing_samples.get().is_some(),
                expected,
                "{name}"
            );
            if !expected {
                assert!(hot_counter(primary).is_none(), "{name}");
                assert!(function.code2.get().is_none(), "{name}");
                assert!(!primary.jit.get().flatten().unwrap().pc_offsets.is_empty());
            } else {
                assert!(
                    function
                        .code2
                        .get()
                        .and_then(Option::as_ref)
                        .is_some_and(|chunk| chunk.optimizing_candidate),
                    "{name}"
                );
            }
        }
        engine.interp.gc_collect();
        let result = engine
            .eval("object.value = 11; [object.value, calls].join('|')", false)
            .unwrap();
        assert!(matches!(result, Completion::Value(ref value) if value == "11|130"));
    }

    #[test]
    fn hot_admission_preserves_profitable_baseline_inlining() {
        const CHILD: &str = "TRUST_JS_TEST_OPT_INLINE_COEXIST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "jit::optimizing::tests::hot_admission_preserves_profitable_baseline_inlining",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("TRUST_JS_OPT_JIT", "hot")
                .env("TRUST_JS_OPT_JIT_HOT_AT", "4096")
                .env("TRUST_JS_INLINE_AT", "16")
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
        assert_eq!(mode(), Mode::Hot);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let result = engine
            .eval(
                r#"
            function leaf(x){return x+3;}
            function subject(x){return leaf(x)*2;}
            var total=0;
            for(var k=0;k<200;k++)total+=subject(k);
            total;
        "#,
                false,
            )
            .unwrap();
        assert!(matches!(result, Completion::Value(ref value) if value == "41000"));
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let function = {
            let object = value.as_obj().unwrap().borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("user function")
            };
            user.func.clone()
        };
        let primary = function.code.get().and_then(Option::as_ref).unwrap();
        let inlined = function
            .code2
            .get()
            .and_then(Option::as_ref)
            .expect("an unadmitted optimizer must leave baseline inlining available");
        assert!(!inlined.optimizing_candidate);
        assert!(!std::rc::Rc::ptr_eq(primary, inlined));
        assert!(inlined
            .jit_ops()
            .iter()
            .any(|op| matches!(op, Op::InlineGuard(..))));
        let code = inlined.jit.get().flatten().expect("inlined body executed");
        assert!(
            !code.pc_offsets.is_empty(),
            "baseline native body remains selected"
        );
        assert_eq!(code.residency.active.get(), 0);
        engine.interp.gc_collect();
        let result = engine
            .eval("leaf=function(x){return x+7;};subject(5)", false)
            .unwrap();
        assert!(
            matches!(result, Completion::Value(ref value) if value == "24"),
            "the inliner's original-call fallback preserves live callee replacement"
        );
    }

    #[test]
    fn hot_admission_covers_direct_calls_callbacks_recursion_and_live_frames() {
        const CHILD: &str = "TRUST_JS_TEST_OPT_HOT_ADMISSION_CHILD";
        if std::env::var_os(CHILD).is_none() {
            // Mode/threshold are process-scoped. Do not race other tests' engines by changing
            // their environment or once-initialized tier controls in this test process.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "jit::optimizing::tests::hot_admission_covers_direct_calls_callbacks_recursion_and_live_frames", "--nocapture"])
                .env(CHILD, "1")
                .env("TRUST_JS_OPT_JIT", "hot")
                .env("TRUST_JS_OPT_JIT_HOT_AT", "8")
                .env("TRUST_JS_JIT_CODE_BUDGET_MB", "1")
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        assert_eq!(mode(), Mode::Hot);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let result = engine
            .eval(
                r#"
            function cold(x){return x+1;}
            function direct(x){return x+2;}
            function callback(x){return x*2;}
            function fieldRead(o){return o.x*2+o.y*3;}
            function fieldThis(){return this.x*2+this.y*3;}
            function getterRead(o){return o.value*2;}
            function recursive(n){if(n<2)return n;return recursive(n-1)+recursive(n-2);}
            var trace=[];
            function protectedCall(n){
                try { if(n===0)throw 'bottom';return protectedCall(n-1); }
                finally { trace.push(n); }
            }
            function makeReader(x){return function reader(y){return x+y;};}
            var first=makeReader(10),second=makeReader(20);
            class Parent {read(){return this.n+4;}}
            class Child extends Parent {read(x){return super.read()+x;}}
            var instance=new Child();instance.n=7;
            var method=Child.prototype.read,methods=0;
            var total=0;
            for(var k=0;k<80;k++){total+=direct(k);methods+=method.call(instance,k);}
            var values=[];for(var k=0;k<80;k++)values.push(k);
            var mapped=values.map(callback);
            var fields={x:2,y:3},fieldTotal=0,thisTotal=0,getterCount=0;
            var accessor={get value(){getterCount++;return 7;}};
            for(var k=0;k<80;k++){
                fieldTotal+=fieldRead(fields);thisTotal+=fieldThis.call(fields);getterRead(accessor);
            }
            var fib=recursive(9),caught='';
            try{protectedCall(12);}catch(e){caught=e;}
            var closures=0;for(var k=0;k<40;k++)closures+=first(k)+second(k);
            [cold(2),total,mapped[79],fib,caught,trace.join(','),closures,methods,fieldTotal,thisTotal,getterCount].join('|');
        "#,
                false,
            )
            .unwrap();
        match result {
            Completion::Value(value) => assert_eq!(
                value,
                "3|3320|158|34|bottom|0,1,2,3,4,5,6,7,8,9,10,11,12|2760|4040|1040|1040|80"
            ),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        let function = |engine: &mut Engine, name: &str| {
            let env = engine.interp.global_env.clone();
            let value = engine.interp.get_var(name, &env).ok().unwrap();
            let object = value.as_obj().unwrap().borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("user function")
            };
            user.func.clone()
        };
        let cold = function(&mut engine, "cold");
        assert!(
            cold.code2.get().is_none(),
            "one call must retain baseline code"
        );
        for name in [
            "direct",
            "callback",
            "fieldRead",
            "fieldThis",
            "recursive",
            "protectedCall",
            "first",
            "second",
        ] {
            let function = function(&mut engine, name);
            let primary = function
                .code
                .get()
                .and_then(Option::as_ref)
                .expect("baseline chunk");
            assert_eq!(
                primary.optimizing_remaining.get(),
                0,
                "{name} reached hot admission"
            );
            let upgraded = function
                .code2
                .get()
                .and_then(Option::as_ref)
                .expect("upgraded chunk");
            assert!(
                upgraded.optimizing_candidate,
                "{name} must use the whole-function tier"
            );
            assert!(
                !std::rc::Rc::ptr_eq(primary, upgraded),
                "active chunk is immutable"
            );
            let native = upgraded
                .jit
                .get()
                .flatten()
                .expect("hot fixture must execute native code");
            assert!(
                native.pc_offsets.is_empty(),
                "no silent template fallback for {name}"
            );
            assert_eq!(native.residency.active.get(), 0);
            if let Some(native) = primary.jit.get().flatten() {
                assert_eq!(native.residency.active.get(), 0);
            }
        }
        let method = function(&mut engine, "method");
        assert!(
            method.code2.get().is_none(),
            "prepared super-method parameters are environment-backed, not guarded private inputs"
        );
        let getter = function(&mut engine, "getterRead");
        assert!(
            getter.code2.get().is_none(),
            "sampling never executes or treats a getter as own data"
        );
        engine.interp.gc_collect();
        assert_eq!(
            engine.interp.depth, 0,
            "promotion/throws must unwind all call depths"
        );
        assert!(
            engine.interp.fn_frames.is_empty(),
            "promotion must not leak invocation frames"
        );
    }

    /// Compile the first declaration explicitly: a silent template/VM fallback cannot pass
    /// these tests. Nested functions use the selected existing tier, testing both directions
    /// of the call boundary without a process-wide environment-variable race.
    pub(super) fn check(source: &str, expected: &str) {
        check_with_gc_pressure(source, expected, false);
    }

    pub(super) fn check_with_gc_pressure(source: &str, expected: &str, pressure: bool) {
        check_with_options(source, expected, pressure, None, None);
    }

    pub(super) fn check_warmed(source: &str, expected: &str) {
        check_with_options(source, expected, false, Some((expected, "")), None);
    }

    /// Seed the real operation caches using an ordinary invocation, then change realm state
    /// before the explicitly compiled invocation. No cache state is fabricated by the test.
    pub(super) fn check_warmed_then(
        source: &str,
        warm_expected: &str,
        before_run: &str,
        expected: &str,
    ) {
        check_with_options(
            source,
            expected,
            false,
            Some((warm_expected, before_run)),
            None,
        );
    }

    pub(super) fn check_with_installer(source: &str, expected: &str, install: fn(&mut Engine)) {
        check_with_options(source, expected, false, None, Some(install));
    }

    /// General-entry bodies require FunctionDeclarationInstantiation before native entry.
    /// Install the exact compiled body on its real function and require an execution mark;
    /// do not bypass the call contract by invoking jit::run with a global environment.
    pub(super) fn check_real_call(source: &str, expected: &str) {
        let mut oracle = Engine::new();
        oracle.set_tier(Tier::Interp);
        assert!(matches!(
            oracle.eval(&format!("{source}\nsubject();"), false).unwrap(),
            Completion::Value(actual) if actual == expected
        ));
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        assert!(matches!(
            engine.eval(source, false).unwrap(),
            Completion::Value(_)
        ));
        let code = super::call_tests::optimized(&mut engine, "subject", None);
        code.residency.referenced.set(0);
        match engine.eval("subject();", false).unwrap() {
            Completion::Value(actual) => assert_eq!(actual, expected),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        assert_eq!(code.residency.referenced.get(), 1, "actual native entry");
        assert_eq!(code.residency.active.get(), 0);
        assert_eq!(engine.interp.depth, 0);
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
    }

    fn check_with_options(
        source: &str,
        expected: &str,
        pressure: bool,
        warm: Option<(&str, &str)>,
        install: Option<fn(&mut Engine)>,
    ) {
        let statements = crate::parser::parse_script(source, false).ok().unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("first statement must declare subject")
        };
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            if let Some(install) = install {
                install(&mut engine);
            }
            engine.eval(source, false).unwrap();
            let chunk = bytecode::compile(function).expect("fixture bytecode");
            assert!(
                !chunk.prepared_entry,
                "prepared-entry fixtures must use check_real_call for proper instantiation"
            );
            let env = engine.interp.global_env.clone();
            if let Some((warm_expected, before_run)) = warm {
                let value =
                    bytecode::run(&mut engine.interp, &chunk, &env, JsValue::Undefined, &[])
                        .unwrap_or_else(|_| panic!("warmup fixture threw"));
                assert_eq!(
                    engine.interp.to_string(&value).ok().unwrap().as_ref(),
                    warm_expected
                );
                assert!(matches!(
                    engine.eval(before_run, false).unwrap(),
                    Completion::Value(_)
                ));
            }
            let layout = crate::interpreter::interp_layout(&mut engine.interp);
            let values = crate::value::jit_layout(&engine.interp.object_proto);
            let code = compile_checked(&chunk, &values, &layout).unwrap_or_else(|e| panic!("{e}"));
            assert!(code.len > 0);
            if pressure {
                engine.interp.gc_next = -1;
                engine.interp.interrupt_poll_tick = 0x3fff;
            }
            let mut result = crate::jit::run(
                &mut engine.interp,
                &chunk,
                &code,
                &env,
                JsValue::Undefined,
                &[],
            )
            .unwrap_or_else(|_| panic!("compiled fixture threw, nested tier {tier:?}"));
            // run() is the native BODY entry, below Interp's ordinary Call trampoline.
            // Strict-mode tail positions intentionally leave that trampoline a pending call.
            while let Some(pending) = engine.interp.pending_tail.take() {
                let (callee, this, arguments) = *pending;
                result = engine
                    .interp
                    .call_tail(callee, this, &arguments)
                    .unwrap_or_else(|_| panic!("fixture tail call threw"));
            }
            engine.interp.gc_collect();
            let actual = engine.interp.to_string(&result).ok().unwrap();
            assert_eq!(actual.as_ref(), expected, "nested tier {tier:?}");
            assert_eq!(code.residency.active.get(), 0);
            assert_eq!(code.residency.referenced.get(), 1);
            // Independent fresh realm: effects in the compiled invocation cannot influence
            // the oracle. Agreement is necessary but does not replace the expected result.
            let mut oracle = Engine::new();
            oracle.set_tier(Tier::Interp);
            if let Some(install) = install {
                install(&mut oracle);
            }
            let before_run = warm.map_or("", |(_, before_run)| before_run);
            let result = oracle
                .eval(&format!("{source}\n{before_run}\nsubject();"), false)
                .unwrap();
            match result {
                Completion::Value(value) => assert_eq!(value, expected, "interpreter oracle"),
                Completion::Throw { name, message } => panic!("oracle threw {name}: {message}"),
            }
        }
    }

    #[test]
    fn whole_function_numeric_control_flow_and_updates() {
        check(
            r#"
            function subject() {
                var total=0;
                for (var i=0;i<80;i++) {
                    if (i<20) total=total+i*2;
                    else total=total-i/2;
                }
                var x=2, a=x++, b=++x, c=x--, d=--x;
                return [total,a,b,c,d,x].join('|');
            }
        "#,
            "-1105|2|4|4|2|2",
        );
    }

    #[test]
    fn whole_function_scalar_branches_keep_local_ssa_without_condition_helpers() {
        bytecode::TEST_JIT_COND_HELPERS.with(|count| count.set(0));
        check(
            r#"
            function subject() {
                var sum=0;
                for(var k=0;k<23;k++) {
                    if(k<4) sum=sum+k; else sum=sum-k;
                }
                return sum;
            }
            "#,
            "-241",
        );
        assert_eq!(bytecode::TEST_JIT_COND_HELPERS.with(|count| count.get()), 0);
    }

    #[test]
    fn whole_function_short_circuit_values_and_effect_counts() {
        check(
            r#"
            function subject() {
                var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,'',0n,3n,Symbol('s'),{}];
                var and=0,or=0,coalesce=0,identities=0;
                for(var k=0;k<values.length;k++) {
                    var value=values[k];
                    value && and++;
                    value || or++;
                    value ?? coalesce++;
                    var token={};
                    var chosen=value ?? token;
                    if(Object.is(chosen,value) || ((value===null || value===undefined) && chosen===token))
                        identities++;
                }
                return [and,or,coalesce,identities].join('|');
            }
            "#,
            "6|8|2|14",
        );
    }

    #[test]
    fn whole_function_ssa_locals_merge_fast_slow_and_exceptional_edges() {
        check(
            r#"
            function subject() {
                var a=0,b=1;
                for(var i=0;i<6;i++) {
                    if(i===2) a='x'; else a=a+1;
                    b=b*2;
                    try { if(i===3) throw b; }
                    catch(e) { a=a+':'+e; }
                    finally { b=b+1; }
                }
                return [a,b,i].join('|');
            }
            "#,
            "x1:3011|127|6",
        );
    }

    #[test]
    fn whole_function_sparse_reload_emission_omits_unused_register_copies() {
        TEST_LOCAL_RELOADS.with(|count| count.set((0, 0)));
        check(
            r#"
            function subject() {
                var a={n:1}, b={n:2}, c={n:3}, d={n:4};
                var sum=a.n+b.n+c.n+d.n;
                tick();tick();tick();
                for(var k=0;k<4;k++) {sum+=k;tick();}
                return sum;
            }
            function tick() { return 1; }
            "#,
            "16",
        );
        let (actual, full) = TEST_LOCAL_RELOADS.with(|count| count.get());
        eprintln!("[local-reloads] {actual} emitted / {full} all-tracked alternative");
        assert!(full > 0, "fixture must actually emit reload sites");
        if std::env::var("TRUST_JS_OPT_JIT_SPARSE_RELOADS").as_deref() == Ok("0") {
            assert_eq!(actual, full, "diagnostic ablation must retain full reloads");
        } else {
            assert!(
                actual < full,
                "must omit emitted loads, not just compute smaller sets"
            );
        }
    }

    #[test]
    fn whole_function_ssa_locals_publish_at_loop_collection_and_interrupt_polls() {
        check_with_gc_pressure(
            r#"
            function subject() {
                var held={value:41}, total=17;
                for(var i=0;i<17000;i++) total=total+1;
                return [total,i,held.value].join('|');
            }
            "#,
            "17017|17000|41",
            true,
        );
    }

    #[test]
    fn whole_function_ssa_locals_preserve_completed_effects_when_a_getter_throws() {
        check(
            r#"
            function subject() {
                var captured=1,local=2,result=[];
                var source={get value(){captured=7;throw 'caught';}};
                try { captured=3;local=11;var ignored=source.value; }
                catch(e) { result.push(e,captured,local);local=local+2; }
                finally { captured=captured+2;local=local*3; }
                return result.join('|')+'|'+captured+'|'+local;
            }
            "#,
            "caught|7|11|9|39",
        );
    }

    #[test]
    fn whole_function_framework_style_reducer_callbacks_and_identity() {
        check(
            r#"
            function subject() {
                var state=[{id:0,done:false},{id:1,done:false}];
                for(var round=0;round<20;round++) {
                    state=state.map(function(item){return {id:item.id,done:!item.done};});
                    state.push({id:round+2,done:round%2===0});
                }
                var chosen=state.filter(function(item){return item.done;});
                var index=new Map();state.forEach(function(item){index.set(item.id,item);});
                return [state.length,chosen.length,index.get(4)===state[4],state[0].done].join('|');
            }
        "#,
            "22|0|true|false",
        );
    }

    #[test]
    fn whole_function_getters_proxies_and_coercion_are_not_replayed() {
        check(
            r#"
            function subject() {
                var trace=[];
                var a={valueOf:function(){trace.push('left');return 4;}};
                var b={valueOf:function(){trace.push('right');return 5;}};
                var target={get x(){trace.push('get');return a;},set x(v){trace.push('set:'+v);}};
                var p=new Proxy(target,{get:function(t,k,r){trace.push('proxy');return Reflect.get(t,k,r);}});
                target.x=p.x+b;
                return trace.join('|');
            }
        "#,
            "proxy|get|left|right|set:9",
        );
    }

    #[test]
    fn whole_function_nested_finalizers_preserve_completion_kinds() {
        check(
            r#"
            function subject() {
                var trace=[];
                outer:for(var i=0;i<4;i++) {
                    try {
                        try { if(i===1)continue outer;if(i===3)break outer;throw i; }
                        catch(e) {trace.push('caught:'+e);}
                        finally {trace.push('inner:'+i);}
                    } finally {trace.push('outer:'+i);}
                }
                try {return trace.join('|');} finally { trace.push('late'); }
            }
        "#,
            "caught:0|inner:0|outer:0|inner:1|outer:1|caught:2|inner:2|outer:2|inner:3|outer:3",
        );
    }

    #[test]
    fn whole_function_iterator_close_and_return_override() {
        check(
            r#"
            function subject() {
                var trace=[];
                var iterable={[Symbol.iterator]:function(){return {
                    next:function(){return {value:7,done:false};},
                    return:function(){trace.push('closed');return {};}
                };}};
                try { for(var value of iterable) { return value; } }
                finally { return trace.join('|'); }
            }
        "#,
            "closed",
        );
    }

    #[test]
    fn whole_function_heap_owners_survive_mutation_and_collection() {
        check(
            r#"
            function subject() {
                var marker={}, held=marker, other=marker, symbol=Symbol('s'),big=12345678901234567890n;
                for(var i=0;i<40;i++){other={parent:held};held=other;}
                other=0;
                for(var i=0;i<40;i++)held=held.parent;
                var weak=new WeakRef(marker);marker=null;
                return [weak.deref()===held,typeof symbol,String(big),other].join('|');
            }
        "#,
            "true|symbol|12345678901234567890|0",
        );
    }

    #[test]
    fn whole_function_primitive_tags_nan_signed_zero_and_bigint_fallback() {
        check(
            r#"
            function subject() {
                var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,'text',0n,3n,Symbol('s'),{}];
                var truth=[];for(var i=0;i<values.length;i++){var copy=values[i];truth.push(copy?1:0);}
                var zero=-0,inf=Infinity,big=3n;big++;
                var caught='no';try{var invalid=big+1;}catch(e){caught=e.name;}
                var nan=inf-inf;
                return [truth.join(''),1/(zero*2),nan!==nan,nan===nan,big+2n,caught].join('|');
            }
        "#,
            "00010001110111|-Infinity|true|false|6|TypeError",
        );
    }

    #[test]
    fn whole_function_binding_tdz_and_captured_environment_barriers() {
        check(
            r#"
            function subject() {
                var result=[];
                try { result.push(local); let local=4; } catch(e){result.push(e.name);}
                var x=1;var change=function(){x=7;};
                result.push(x);change();result.push(x);
                var callbacks=[];for(let k=0;k<3;k++)callbacks.push(function(){return k;});
                result.push(callbacks.map(function(f){return f();}).join(','));
                return result.join('|');
            }
        "#,
            "ReferenceError|1|7|0,1,2",
        );
    }

    #[test]
    fn whole_function_strict_mode_is_saved_across_nested_calls() {
        check(
            r#"
            function subject() {
                'use strict';
                var frozen=Object.freeze({x:1}),result=[];
                function sloppy(o){o.x=2;return o.x;}
                var external=Function('o','o.x=2;return o.x');
                result.push(external(frozen));
                try {frozen.x=3;}catch(e){result.push(e.name);}
                try {sloppy(frozen);}catch(e){result.push(e.name);}
                return result.join('|');
            }
        "#,
            "1|TypeError|TypeError",
        );
    }
}
