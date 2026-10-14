//! Shared non-observing cache probes. Results are non-owning packed words.
//!
//! ECMA-262 e28783d5 GetBindingValue, OrdinaryGet and WeakRef liveness: a probe
//! neither invokes the language operation nor owns a value. Live canonical or
//! register owners keep every input alive. A success precedes the caller's
//! existing retain/consume transaction; a miss changes no state or result word.
//! No allocation, callback, collection, owner clone/drop or unwind occurs here.

use super::*;
use crate::bytecode::{IcState, NameIc, IC_MAX_DEPTH, PROP_IC_WAYS};
use crate::value::{JitLayout, PACK_BIGINT, PACK_OBJ, PACK_STR, PACK_SYM, PROP_ACCESSOR};
use std::cell::Cell;

const SUCCESS: u64 = 1;
const GUARD_SHIFT: u32 = 8;
const PAYLOAD: u64 = 0x0000_ffff_ffff_ffff;

#[cfg(test)]
thread_local! {
    // name calls/property calls/name misses/property misses; actual leaf entries.
    static TEST_CALLS: Cell<[usize; 4]> = const { Cell::new([0; 4]) };
    static TEST_LAST_OWNER_PROBES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
fn record(property: bool, status: u64) {
    TEST_CALLS.with(|calls| {
        let mut counts = calls.get();
        let kind = usize::from(property);
        counts[kind] += 1;
        if status & SUCCESS == 0 {
            counts[kind + 2] += 1;
        }
        calls.set(counts);
    });
}

#[cfg(test)]
unsafe fn record_last_owner_probe(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    receiver: u64,
    status: u64,
) {
    // Only a successful probe has proved the live Object/non-null receiver.
    // This records the caller's consumed-stack case before its count rejection.
    if status & SUCCESS != 0
        && matches!((&*(*ctx).chunk).jit_ops()[pc as usize], Op::GetProp(..))
        && load::<usize>((receiver & PAYLOAD) as usize, (*l).rc_strong_off) == 1
    {
        TEST_LAST_OWNER_PROBES.with(|count| count.set(count.get() + 1));
    }
}

/// Only offsets/representation flags; no object, environment, cache or Realm identity.
static LAYOUT: OnceLock<JitLayout> = OnceLock::new();

pub(super) fn requested() -> bool {
    std::env::var("TRUST_JS_OPT_JIT_SHARED_READ_GUARDS").as_deref() == Ok("1")
}

pub(super) fn layout(values: &JitLayout) -> Option<&'static JitLayout> {
    if !requested() || !property::supported(values) || !names::supported(values) {
        return None;
    }
    let stored = LAYOUT.get_or_init(|| *values);
    (stored == values).then_some(stored)
}

struct Probe<const COUNT: bool> {
    guards: u64,
}

impl<const COUNT: bool> Probe<COUNT> {
    #[inline(always)]
    fn guard(&mut self, pass: bool) -> bool {
        if COUNT {
            self.guards += 1;
        }
        pass
    }

    #[inline(always)]
    fn status(&self, success: bool) -> u64 {
        u64::from(success) | (self.guards << GUARD_SHIFT)
    }
}

/// Loads only initialized fields. In particular do not read repr(Value) padding
/// into a Rust integer: Bool has a one-byte payload, and nullish values have none.
#[inline(always)]
unsafe fn load<T: Copy>(base: usize, offset: usize) -> T {
    ((base + offset) as *const T).read()
}

#[inline(always)]
unsafe fn plain<const COUNT: bool>(p: &mut Probe<COUNT>, l: &JitLayout, object: usize) -> bool {
    p.guard(load::<u8>(object, l.obj_ic_plain) != 0)
}

#[inline(always)]
unsafe fn ordinary<const COUNT: bool>(p: &mut Probe<COUNT>, l: &JitLayout, object: usize) -> bool {
    p.guard(load::<u8>(object, l.obj_exotic) == l.exotic_none_tag) && plain(p, l, object)
}

#[inline(always)]
unsafe fn shape<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    object: usize,
    expected: u32,
) -> bool {
    p.guard(load::<u32>(object, l.obj_props + l.props_shape) == expected)
}

#[inline(always)]
unsafe fn property_word<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    object: usize,
    slot: usize,
) -> Option<u64> {
    let len = load::<usize>(object, l.obj_props + l.props_entries + l.vec_len_off);
    if !p.guard(slot < len) {
        return None;
    }
    let entries = load::<usize>(object, l.obj_props + l.props_entries + l.vec_ptr_off);
    let entry = entries + slot * l.entry_size;
    if !p.guard(load::<u8>(entry, l.entry_accessor) & PROP_ACCESSOR as u8 == 0) {
        return None;
    }
    Some(load::<u64>(entry, l.entry_value))
}

/// Safety: layout was validated against live Rust types; ctx names the immutable
/// current Chunk; receiver is a canonical owning word whose source remains live;
/// out is an aligned dedicated native stack slot, disjoint from engine storage.
#[inline(always)]
unsafe fn property_probe<const COUNT: bool>(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    receiver: u64,
    out: *mut u64,
) -> u64 {
    let l = &*l;
    let mut p = Probe::<COUNT> { guards: 0 };
    if !p.guard(receiver >> 48 == PACK_OBJ >> 48) {
        return p.status(false);
    }
    let owner = (receiver & PAYLOAD) as usize;
    if !p.guard(owner != 0) {
        return p.status(false);
    }
    let chunk = &*(*ctx).chunk;
    let (name, cache) = match chunk.jit_ops()[pc as usize] {
        Op::GetProp(name, cache)
        | Op::GetMethod(name, cache)
        | Op::GetPropThis(name, cache)
        | Op::GetPropLocal(_, name, cache) => (name, cache),
        _ => return p.status(false),
    };
    let object = owner + l.obj_from_rc;
    if !plain(&mut p, l, object) {
        return p.status(false);
    }
    let exotic = load::<u8>(object, l.obj_exotic);
    let ordinary_receiver = exotic == l.exotic_none_tag;
    let non_index = !chunk
        .jit_name(name)
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_digit);
    let array = exotic == l.exotic_array_tag && non_index;
    if !p.guard(ordinary_receiver || array) {
        return p.status(false);
    }
    let cache = chunk.jit_cache_ptr(cache) as *const Cell<IcState>;
    for way in 0..PROP_IC_WAYS {
        let state = (*cache.add(way)).get();
        if !p.guard(state.depth <= IC_MAX_DEPTH)
            || (non_index && !p.guard(ordinary_receiver || (array && state.depth != 0)))
            || !shape(&mut p, l, object, state.recv_shape)
        {
            continue;
        }
        let mut holder = object;
        let mut valid = true;
        for hop in 0..usize::from(state.depth) {
            let proto = load::<usize>(holder, l.obj_proto);
            if !p.guard(proto != 0) {
                valid = false;
                break;
            }
            holder = proto + l.obj_from_rc;
            if !ordinary(&mut p, l, holder) {
                valid = false;
                break;
            }
            let last = hop + 1 == usize::from(state.depth);
            if hop < 4 && !p.guard(last || state.mid_ok & (1 << hop) != 0) {
                valid = false;
                break;
            }
            let expected = if last {
                state.holder_shape
            } else {
                [
                    state.mid_shape,
                    state.mid2_shape,
                    state.mid3_shape,
                    state.mid4_shape,
                ][hop]
            };
            if !shape(&mut p, l, holder, expected) {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }
        // Existing generic emission checks bounds/accessor after selecting a way;
        // a changed descriptor uses the checked operation, not another stale way.
        let Some(word) = property_word(&mut p, l, holder, state.slot as usize) else {
            return p.status(false);
        };
        out.write(word);
        return p.status(true);
    }
    p.status(false)
}

#[inline(never)]
pub(super) unsafe extern "C" fn property_uncounted(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    receiver: u64,
    out: *mut u64,
) -> u64 {
    let status = property_probe::<false>(l, ctx, pc, receiver, out);
    #[cfg(test)]
    record(true, status);
    #[cfg(test)]
    record_last_owner_probe(l, ctx, pc, receiver, status);
    status
}

#[inline(never)]
pub(super) unsafe extern "C" fn property_counted(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    receiver: u64,
    out: *mut u64,
) -> u64 {
    let status = property_probe::<true>(l, ctx, pc, receiver, out);
    #[cfg(test)]
    record(true, status);
    #[cfg(test)]
    record_last_owner_probe(l, ctx, pc, receiver, status);
    status
}

#[inline(always)]
unsafe fn scope_plain<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    scope: usize,
) -> bool {
    p.guard(load::<u8>(scope, l.scope_with) == l.scope_with_none)
}

#[inline(always)]
unsafe fn generation<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    scope: usize,
    expected: u32,
) -> bool {
    p.guard(expected != u32::MAX) && p.guard(load::<u32>(scope, l.scope_gen) == expected)
}

#[inline(always)]
unsafe fn scope_layout<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    scope: usize,
    expected: u32,
) -> bool {
    p.guard(expected != 0) && p.guard(load::<u32>(scope, l.scope_layout) == expected)
}

#[inline(always)]
unsafe fn fixed_binding<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    scope: usize,
    slot: usize,
) -> Option<usize> {
    if !l.scope_small_valid {
        return None;
    }
    let tag = load::<u8>(scope, l.scope_small_tag);
    if !p.guard(tag == 0 || tag == 2) {
        return None;
    }
    let len = load::<usize>(scope, l.scope_small_vec + l.vec_len_off);
    if !p.guard(slot < len) {
        return None;
    }
    let entries = load::<usize>(scope, l.scope_small_vec + l.vec_ptr_off);
    Some(entries + slot * l.scope_binding_stride + l.scope_binding_offset)
}

#[inline(always)]
unsafe fn binding_word<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    binding: usize,
) -> Option<u64> {
    if !p.guard(binding != 0)
        || !p.guard(load::<u8>(binding, l.binding_init) != 0)
        || !p.guard(load::<u8>(binding, l.binding_import) == 0)
    {
        return None;
    }
    let value = binding + l.binding_value;
    Some(match load::<u8>(value, 0) {
        0 => PACK_UNDEFINED,
        1 => PACK_EMPTY,
        2 => PACK_NULL,
        3 => PACK_BOOL | u64::from(load::<u8>(value, 8) & 1),
        4 => {
            let bits = load::<u64>(value, 8);
            if f64::from_bits(bits).is_nan() {
                PACK_CANON_NAN
            } else {
                bits
            }
        }
        5 => PACK_BIGINT | load::<usize>(value, 8) as u64,
        6 => PACK_STR | load::<usize>(value, 8) as u64,
        7 => PACK_SYM | load::<usize>(value, 8) as u64,
        8 => PACK_OBJ | load::<usize>(value, 8) as u64,
        _ => return None,
    })
}

#[inline(always)]
unsafe fn global_word<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    ctx: &JitCtx,
    slot: usize,
    expected: u32,
) -> Option<u64> {
    let object = ctx.global_body as usize;
    if !p.guard(object != 0) || !ordinary(p, l, object) || !shape(p, l, object, expected) {
        return None;
    }
    property_word(p, l, object, slot)
}

#[inline(always)]
unsafe fn name_word<const COUNT: bool>(
    p: &mut Probe<COUNT>,
    l: &JitLayout,
    ctx: &JitCtx,
    state: NameIc,
) -> Option<u64> {
    use crate::bytecode::lexical_cache::{
        NativeNameDescriptor, DEEP_NAME_IC, NAME_GUARD_GENERATION, NAME_GUARD_IDENTITY,
        NAME_GUARD_LAYOUT, NAME_GUARD_SIZE,
    };
    let scope = ctx.env_raw as usize;
    if !scope_plain(p, l, scope) {
        return None;
    }
    let pointer = if state.env == scope {
        if !generation(p, l, scope, state.gen) {
            return None;
        }
        state.binding as usize
    } else if state.env == scope | 1 {
        if !generation(p, l, scope, state.gen) {
            return None;
        }
        return global_word(
            p,
            l,
            ctx,
            (state.binding & u64::from(u32::MAX)) as usize,
            (state.binding >> 32) as u32,
        );
    } else if state.env == bytecode::FIXED_NAME_IC {
        if !scope_layout(p, l, scope, state.gen) {
            return None;
        }
        fixed_binding(p, l, scope, state.binding as usize)?
    } else if state.env == DEEP_NAME_IC {
        if !l.scope_parent_valid || !p.guard(state.binding != 0) {
            return None;
        }
        let descriptor = &*(state.binding as *const NativeNameDescriptor);
        if !p.guard(descriptor.len != 0)
            || !p.guard(descriptor.len <= 16)
            || !p.guard(!descriptor.guards.is_null())
        {
            return None;
        }
        let mut current = scope;
        let mut exact = false;
        for hop in 0..descriptor.len {
            if !scope_plain(p, l, current) {
                return None;
            }
            let guard = descriptor.guards as usize + hop * NAME_GUARD_SIZE;
            let identity = load::<usize>(guard, NAME_GUARD_IDENTITY);
            let expected_generation = load::<u32>(guard, NAME_GUARD_GENERATION);
            exact = identity == current
                && expected_generation != u32::MAX
                && load::<u32>(current, l.scope_gen) == expected_generation;
            if !exact && !scope_layout(p, l, current, load::<u32>(guard, NAME_GUARD_LAYOUT)) {
                return None;
            }
            if hop + 1 < descriptor.len {
                let parent = load::<usize>(current, l.scope_parent);
                if !p.guard(parent != 0) {
                    return None;
                }
                current = parent + l.scope_data_off;
            }
        }
        if descriptor.kind == 0 {
            if exact {
                descriptor.binding as usize
            } else {
                fixed_binding(p, l, current, descriptor.slot)?
            }
        } else {
            if !p.guard(descriptor.kind == 1) || !p.guard(ctx.genv == current) {
                return None;
            }
            return global_word(p, l, ctx, descriptor.slot, descriptor.shape);
        }
    } else if state.env & 2 != 0 {
        let activation_valid = if state.env & 4 != 0 {
            scope_layout(p, l, scope, state.act_gen)
        } else {
            generation(p, l, scope, state.act_gen)
        };
        if !activation_valid {
            return None;
        }
        let parent = ctx.env_parent_raw as usize;
        if !p.guard(parent != 0)
            || !p.guard(parent == state.env & !7)
            || !scope_plain(p, l, parent)
            || !generation(p, l, parent, state.gen)
        {
            return None;
        }
        state.binding as usize
    } else {
        return None;
    };
    binding_word(p, l, pointer)
}

#[inline(always)]
unsafe fn name_probe<const COUNT: bool>(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    _receiver: u64,
    out: *mut u64,
) -> u64 {
    let l = &*l;
    let ctx = &*ctx;
    let chunk = &*ctx.chunk;
    let cache = match chunk.jit_ops()[pc as usize] {
        Op::LoadName(_, cache) | Op::LoadNameForCall(_, cache) => chunk.jit_name_cache_ptr(cache),
        Op::LoadCap(name) if !chunk.jit_needs_activation_state() => chunk.jit_cap_cache_ptr(name),
        _ => return 0,
    };
    let mut p = Probe::<COUNT> { guards: 0 };
    let state = (*(cache as *const Cell<NameIc>)).get();
    let Some(word) = name_word(&mut p, l, ctx, state) else {
        return p.status(false);
    };
    out.write(word);
    p.status(true)
}

#[inline(never)]
pub(super) unsafe extern "C" fn name_uncounted(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    receiver: u64,
    out: *mut u64,
) -> u64 {
    let status = name_probe::<false>(l, ctx, pc, receiver, out);
    #[cfg(test)]
    record(false, status);
    status
}

#[inline(never)]
pub(super) unsafe extern "C" fn name_counted(
    l: *const JitLayout,
    ctx: *const JitCtx,
    pc: u32,
    receiver: u64,
    out: *mut u64,
) -> u64 {
    let status = name_probe::<true>(l, ctx, pc, receiver, out);
    #[cfg(test)]
    record(false, status);
    status
}

type ReadGuard = unsafe extern "C" fn(*const JitLayout, *const JitCtx, u32, u64, *mut u64) -> u64;

impl Lowering<'_, '_> {
    /// These calls touch no canonical owner or language effect. ABI register
    /// preservation is Cranelift's responsibility; this is not a helper observer.
    fn shared_read_guard(&mut self, receiver: Value, property: bool, miss: Block) -> Value {
        // Scalar/proof-only bodies gain neither a signature nor a native stack
        // reservation. Create these only at the first actual shared probe.
        let signature = *self.read_guard_sig.get_or_insert_with(|| {
            let mut signature = Signature::new(self.b.func.signature.call_conv);
            signature.params.extend([
                AbiParam::new(types::I64),
                AbiParam::new(types::I64),
                AbiParam::new(types::I32),
                AbiParam::new(types::I64),
                AbiParam::new(types::I64),
            ]);
            signature.returns.push(AbiParam::new(types::I64));
            self.b.func.import_signature(signature)
        });
        let slot = *self.read_guard_result.get_or_insert_with(|| {
            self.b
                .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    8,
                    3,
                ))
        });
        let layout = self.read_guard_layout.expect("validated scalar layout");
        let counted = self
            .diagnostics
            .is_some_and(|record| record.runtime_counters);
        let helper: ReadGuard = match (property, counted) {
            (true, false) => property_uncounted,
            (true, true) => property_counted,
            (false, false) => name_uncounted,
            (false, true) => name_counted,
        };
        if let Some(record) = self.diagnostics {
            record.emitted_helpers.set(record.emitted_helpers.get() + 1);
            record
                .emitted_read_guard_calls
                .set(record.emitted_read_guard_calls.get() + 1);
        }
        let callee = self.b.ins().iconst(types::I64, helper as usize as i64);
        let layout = self
            .b
            .ins()
            .iconst(types::I64, (layout as *const JitLayout) as i64);
        let pc = self.b.ins().iconst(types::I32, self.pc as i64);
        let output = self.b.ins().stack_addr(types::I64, slot, 0);
        let call = self.b.ins().call_indirect(
            signature,
            callee,
            &[layout, self.ctx, pc, receiver, output],
        );
        let status = self.b.inst_results(call)[0];
        if counted {
            let guards = self.b.ins().ushr_imm_u(status, i64::from(GUARD_SHIFT));
            self.owner_leaf_count(diagnostics::HEAP_GUARD, guards);
        }
        let success = self.b.ins().band_imm_s(status, SUCCESS as i64);
        let hit = self.b.create_block();
        self.b.ins().brif(success, hit, &[], miss, &[]);
        self.b.switch_to_block(hit);
        // The leaf initializes this word only on success. No stale/miss output read.
        self.b.ins().stack_load(types::I64, types::I64, slot, 0)
    }

    pub(super) fn shared_name_read(&mut self, pc: usize, op: &Op, next: Block) -> bool {
        if self.read_guard_layout.is_none() {
            return false;
        }
        let slow = self.b.create_block();
        let unused = self.b.ins().iconst(types::I64, 0);
        let word = self.shared_read_guard(unused, false, slow);
        self.retain_word(word, slow);
        if matches!(op, Op::LoadNameForCall(..)) {
            let receiver = self.b.ins().iconst(types::I64, PACK_UNDEFINED as i64);
            self.push(receiver);
        }
        self.push(word);
        self.jump_normal(next);
        self.slow(pc, slow, next);
        true
    }

    pub(super) fn shared_property_read(&mut self, pc: usize, op: &Op, next: Block) -> bool {
        if self.read_guard_layout.is_none() {
            return false;
        }
        let slow = self.b.create_block();
        let receiver = if matches!(op, Op::GetPropThis(..)) {
            let this = self.load(self.ctx, offset_of!(JitCtx, this_raw) as i32);
            let tag = self.property_load(types::I8, this, 0);
            let object = self.b.ins().icmp_imm_s(IntCC::Equal, tag, 8);
            self.property_guard(object, slow);
            let owner = self.load(this, 8);
            self.b.ins().bor_imm_s(owner, PACK_OBJ as i64)
        } else if let Op::GetPropLocal(slot, ..) = op {
            self.local(*slot)
        } else {
            self.stack_read(1)
        };
        let word = self.shared_read_guard(receiver, true, slow);
        let owner = if matches!(op, Op::GetProp(..)) {
            // A successful probe proved Object/non-null before this dereference.
            let owner = self.b.ins().band_imm_s(receiver, PAYLOAD as i64);
            let count = self.load(owner, self.values.rc_strong_off as i32);
            let shared = self
                .b
                .ins()
                .icmp_imm_s(IntCC::UnsignedGreaterThan, count, 1);
            self.property_guard(shared, slow);
            Some(owner)
        } else {
            None
        };
        self.retain_word(word, slow);
        if let Some(owner) = owner {
            // Re-read after retain because the property may name the same owner.
            let count = self.load(owner, self.values.rc_strong_off as i32);
            let count = self.b.ins().iadd_imm_s(count, -1);
            self.store(count, owner, self.values.rc_strong_off as i32);
            self.count_runtime(diagnostics::RELEASE, 1);
            self.stack_write(1, word);
        } else {
            self.push(word);
        }
        self.count_native_site_success(pc, "data_property_read");
        self.jump_normal(next);
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        self.stack_state = self.stack_plan.at[pc];
        self.checked(get_property, pc as u32, next);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controlled_children(name: &str) -> bool {
        controlled_children_with_env(name, &[])
    }

    fn controlled_children_with_env(name: &str, envs: &[(&str, &str)]) -> bool {
        const CHILD: &str = "TRUST_JS_TEST_SHARED_READ_GUARD_CHILD";
        if let Some(flag) = std::env::var_os(CHILD) {
            assert_eq!(requested(), flag == "1");
            return false;
        }
        for flag in ["0", "1"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture", "--test-threads=1"])
                .env(CHILD, flag)
                .env("TRUST_JS_OPT_JIT_SHARED_READ_GUARDS", flag)
                .envs(envs.iter().copied())
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "flag{flag}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        }
        true
    }

    #[test]
    fn wide_binding_reads_initialized_variant_bytes_without_owner_changes() {
        let engine = crate::Engine::new();
        let l = crate::value::jit_layout(&engine.interp.object_proto);
        assert!(property::supported(&l) && names::supported(&l));
        for (value, expected) in [
            (crate::value::Value::Undefined, PACK_UNDEFINED),
            (crate::value::Value::Null, PACK_NULL),
            (crate::value::Value::Bool(false), PACK_BOOL),
            (crate::value::Value::Bool(true), PACK_BOOL | 1),
            (crate::value::Value::Num(0.0), 0),
            (crate::value::Value::Num(-0.0), (-0.0f64).to_bits()),
            (crate::value::Value::Num(f64::NAN), PACK_CANON_NAN),
        ] {
            let binding = crate::interpreter::Binding {
                value,
                mutable: true,
                initialized: true,
                import_ref: None,
                imported: false,
                deletable: false,
                strict_immutable: false,
            };
            let mut counted = Probe::<true> { guards: 0 };
            let word = unsafe { binding_word(&mut counted, &l, &binding as *const _ as usize) };
            assert_eq!(word, Some(expected));
            assert_eq!(counted.guards, 3);
            let mut uncounted = Probe::<false> { guards: 0 };
            assert_eq!(
                unsafe { binding_word(&mut uncounted, &l, &binding as *const _ as usize) },
                word
            );
            assert_eq!(uncounted.guards, 0);
        }
    }

    #[test]
    fn unused_shared_probes_reserve_no_stack_result_or_call_signature() {
        if controlled_children("jit::optimizing::read_guards::tests::unused_shared_probes_reserve_no_stack_result_or_call_signature") {
            return;
        }
        for (source, has_name) in [
            ("function subject(){return 1+2}", false),
            (
                "function subject(){return outside}globalThis.outside=3;",
                true,
            ),
        ] {
            let statements = crate::parser::parse_script(source, false).ok().unwrap();
            let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
                panic!("subject declaration")
            };
            let chunk = bytecode::compile(function).unwrap();
            let mut engine = crate::Engine::new();
            engine.eval(source, false).unwrap();
            let values = crate::value::jit_layout(&engine.interp.object_proto);
            let layout = crate::interpreter::interp_layout(&mut engine.interp);
            let prepared =
                prepare_entry_with_limits(&chunk, &values, &layout, None, 16_384, 65_536, None)
                    .unwrap();
            let func = &prepared.context.func;
            let expected = usize::from(requested() && has_name);
            assert_eq!(
                func.sized_stack_slots.len(),
                expected,
                "dedicated borrowed-result slot"
            );
            let signatures = func
                .dfg
                .signatures
                .values()
                .filter(|sig| {
                    sig.params
                        .iter()
                        .map(|param| param.value_type)
                        .collect::<Vec<_>>()
                        == [types::I64, types::I64, types::I32, types::I64, types::I64]
                        && sig
                            .returns
                            .iter()
                            .map(|param| param.value_type)
                            .collect::<Vec<_>>()
                            == [types::I64]
                })
                .count();
            assert_eq!(
                signatures, expected,
                "shared probe ABI imported only on actual emission"
            );
            eprintln!("shared probe requested={} actual={has_name} stack_slots={} read_signatures={signatures}", requested(), func.sized_stack_slots.len());
        }
    }

    #[test]
    fn successful_last_owned_receiver_probes_use_checked_transfer_and_keep_result_alive() {
        if controlled_children("jit::optimizing::read_guards::tests::successful_last_owned_receiver_probes_use_checked_transfer_and_keep_result_alive") {
            return;
        }
        for mode in ["0", "1"] {
            diagnostics::with_test_mode(mode, || {
                TEST_CALLS.with(|calls| calls.set([0; 4]));
                TEST_LAST_OWNER_PROBES.with(|count| count.set(0));
                property::TEST_READ_HELPERS.with(|count| count.set(0));
                super::super::tests::check_with_gc_pressure(
                    "function subject(){var pin={x:null},kept;for(var k=0;k<3;k++){kept=({x:{token:'live'}}).x;$262.gc()}return typeof pin+':'+kept.token}",
                    "object:live", true,
                );
                let last = TEST_LAST_OWNER_PROBES.with(Cell::get);
                assert_eq!(
                    last,
                    if requested() { 6 } else { 0 },
                    "two successful sole-owner probes in each of three actual native bodies"
                );
                assert_eq!(property::TEST_READ_HELPERS.with(Cell::get), 15, "three consumed receiver transfers, cold gc method, cold token read per native body");
                eprintln!("sole-owner mode={mode} requested={} successful_probes={last} calls={:?} checked_reads={}", requested(), TEST_CALLS.with(Cell::get), property::TEST_READ_HELPERS.with(Cell::get));
            });
        }
    }

    #[test]
    fn native_shared_nan_values_keep_counted_and_uncounted_fallbacks_exact() {
        if controlled_children("jit::optimizing::read_guards::tests::native_shared_nan_values_keep_counted_and_uncounted_fallbacks_exact") {
            return;
        }
        for mode in ["0", "1"] {
            diagnostics::with_test_mode(mode, || {
                TEST_CALLS.with(|calls| calls.set([0; 4]));
                property::TEST_READ_HELPERS.with(|count| count.set(0));
                super::super::tests::check(
                    "function subject(){var o={x:0/0},s='';for(var k=0;k<3;k++)s+=o.x!==o.x;return s}",
                    "truetruetrue",
                );
                assert_eq!(
                    property::TEST_READ_HELPERS.with(Cell::get),
                    6,
                    "one cold miss at each of two read PCs per native body"
                );
                let calls = TEST_CALLS.with(Cell::get);
                assert_eq!(calls[1], if requested() { 18 } else { 0 });
                assert_eq!(calls[3], if requested() { 6 } else { 0 });
                eprintln!(
                    "NaN mode={mode} shared={} calls={calls:?} checked_reads=6",
                    requested()
                );
            });
        }
    }

    #[test]
    fn shared_continuation_reads_publish_buffered_aliases_before_leaf_probes() {
        use crate::value::Callable;
        if controlled_children_with_env(
            "jit::optimizing::read_guards::tests::shared_continuation_reads_publish_buffered_aliases_before_leaf_probes",
            &[("TRUST_JS_OPT_JIT", "0"), ("TRUST_JS_OPT_JIT_OSR", "1"),
              ("TRUST_JS_OPT_JIT_OSR_AT", "16"), ("TRUST_JS_OPT_JIT_LOOP_FACTS", "1"),
              ("TRUST_JS_OPT_JIT_LOOP_SCALARS", "1"), ("TRUST_JS_INLINE_AT", "0"),
              ("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution")],
        ) { return; }
        for (source, property) in [
            ("globalThis.x=0;function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;seen=x;}return seen}subject(globalThis)+':'+globalThis.x", false),
            ("function subject(o){var seen=0;for(var k=0;k<100;k++){o.x=o.x+1;if(k>=80)seen=this.x;}return seen}var o={x:0};subject.call(o,o)+':'+o.x", true),
        ] {
            for tier in [bytecode::Tier::Interp, bytecode::Tier::Bytecode, bytecode::Tier::Jit] {
                TEST_CALLS.with(|calls| calls.set([0; 4]));
                let mut engine = crate::Engine::new();
                engine.set_tier(tier);
                engine.set_tier_threshold(0);
                assert!(matches!(engine.eval(source,false).unwrap(),crate::Completion::Value(value) if value=="100:100"));
                if !matches!(tier, bytecode::Tier::Jit) { continue; }
                let env = engine.interp.global_env.clone();
                let value = engine.interp.get_var("subject", &env).ok().unwrap();
                let object = value.as_obj().unwrap().borrow();
                let Callable::User(user) = &object.call else { panic!("author subject") };
                let chunk = user.func.execution_code().and_then(Option::as_ref).unwrap();
                let state = chunk.optimizing_loop.get().expect("actual continuation");
                assert!(state.entries.get() > 0);
                assert_eq!(state.misses.get(), 0);
                let code = state.code.get().flatten().unwrap();
                let record = code.optimizing_diagnostics.as_ref().unwrap();
                assert!(record.counters[0].get() > 0);
                assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 0);
                let calls = TEST_CALLS.with(Cell::get);
                if requested() {
                    assert!(calls[usize::from(property)] > 0, "must execute the relevant shared native leaf: {calls:?}");
                    if property {
                        assert!(record.site_stats().iter().any(|(_, kind, count)| *kind == "data_property_read" && *count > 0));
                    }
                } else { assert_eq!(calls, [0; 4]); }
                eprintln!("OSR shared={} property={property} entries={} calls={calls:?} sites={:?}", requested(), state.entries.get(), record.site_stats());
            }
        }
    }

    #[test]
    fn native_shared_reads_distinguish_zero_results_and_counted_guard_misses() {
        if controlled_children("jit::optimizing::read_guards::tests::native_shared_reads_distinguish_zero_results_and_counted_guard_misses") {
            return;
        }
        for mode in ["0", "compile", "1", "execution"] {
            diagnostics::with_test_mode(mode, || {
                TEST_CALLS.with(|calls| calls.set([0; 4]));
                property::TEST_READ_HELPERS.with(|count| count.set(0));
                super::super::tests::check(
                    "function subject(){var o={x:0},s='',gets=0;for(var k=0;k<3;k++){s+=1/o.x+',';}Object.defineProperty(o,'x',{get(){gets++;return -0}});s+=1/o.x;return s+'|'+gets}",
                    "Infinity,Infinity,Infinity,-Infinity|1",
                );
                let counts = TEST_CALLS.with(Cell::get);
                if !requested() {
                    assert_eq!(
                        counts, [0; 4],
                        "off control retains original native lowering"
                    );
                    return;
                }
                assert!(
                    counts[1] >= 4,
                    "actual native shared property probes: {counts:?}"
                );
                assert!(
                    counts[3] >= 2,
                    "cold IC and accessor both use exact fallback: {counts:?}"
                );
                assert!(property::TEST_READ_HELPERS.with(Cell::get) >= 2);
            });
        }
    }

    #[test]
    fn native_shared_names_preserve_f64_owners_tdz_and_live_getters() {
        if controlled_children("jit::optimizing::read_guards::tests::native_shared_names_preserve_f64_owners_tdz_and_live_getters") {
            return;
        }
        for mode in ["0", "1"] {
            diagnostics::with_test_mode(mode, || {
                TEST_CALLS.with(|calls| calls.set([0; 4]));
                super::super::tests::check_warmed(
                    "function subject(){return (external*1.25+other*2.5)+marker}let external=4,other=2,marker=3;",
                    "13",
                );
                super::super::tests::check_warmed(
                    "function subject(){var x=1.25*4+2.5*2;return [x+outside,1/negative,yes,text,object===object].join(':')}let outside=3,negative=-0,yes=true,text='live',object={}",
                    "13:-Infinity:true:live:true",
                );
                let calls = TEST_CALLS.with(Cell::get);
                if requested() {
                    assert!(calls[0] >= 5, "actual native name leaves: {calls:?}");
                } else {
                    assert_eq!(calls, [0; 4]);
                }
                TEST_CALLS.with(|calls| calls.set([0; 4]));
                super::super::tests::check_real_call(
                    "function subject(){function keeper(){return later}try{return later}catch(e){return e.name}let later=3}",
                    "ReferenceError",
                );
                let calls = TEST_CALLS.with(Cell::get);
                if requested() {
                    assert!(
                        calls[0] >= 1 && calls[2] >= 1,
                        "environment TDZ reaches a failed shared name leaf: {calls:?}"
                    );
                } else {
                    assert_eq!(calls, [0; 4]);
                }
                super::super::tests::check_warmed_then(
                    "function subject(){return outside+':'+reads}globalThis.outside=7;var reads=0;",
                    "7:0",
                    "Object.defineProperty(globalThis,'outside',{get(){reads++;return 9},configurable:true});",
                    "9:1",
                );
            });
        }
    }

    #[test]
    fn native_shared_read_guards_preserve_polymorphism_aliases_and_gc() {
        if controlled_children("jit::optimizing::read_guards::tests::native_shared_read_guards_preserve_polymorphism_aliases_and_gc") { return; }
        TEST_CALLS.with(|calls| calls.set([0; 4]));
        super::super::tests::check_with_gc_pressure(
            "function subject(){var held={},rows=[{x:held},{tag:1,x:held},{before:0,tag:1,x:held},{z:0,before:0,tag:1,x:held}],n=0;for(var k=0;k<24;k++){var row=rows[k%4];if(row.x===held)n++;$262.gc()}var cycle={};cycle.self=cycle;var ok=cycle.self===cycle;delete cycle.self;var last=({x:held}).x;return n+':'+ok+':'+(last===held)}",
            "24:true:true", true,
        );
        let calls = TEST_CALLS.with(Cell::get);
        if requested() {
            assert!(
                calls[1] >= 24,
                "actual polymorphic/alias native leaves: {calls:?}"
            );
        } else {
            assert_eq!(calls, [0; 4]);
        }
        super::super::tests::check(
            "function subject(){var trace='',proto={x:3},o=Object.create(proto),sum=0;for(var k=0;k<8;k++){if(k===2)Object.defineProperty(proto,'x',{get(){trace+='g';return 5},configurable:true});if(k===4)o=new Proxy(o,{get(t,key,r){trace+='p';return Reflect.get(t,key,r)}});if(k===6)Object.setPrototypeOf(proto,{x:7});sum+=o.x}return sum+':'+trace}",
            "36:ggpgpgpgpg",
        );
    }

    #[test]
    fn native_shared_name_imports_read_live_export_bindings() {
        if controlled_children("jit::optimizing::read_guards::tests::native_shared_name_imports_read_live_export_bindings") { return; }
        TEST_CALLS.with(|calls| calls.set([0; 4]));
        let mut engine = crate::Engine::new();
        engine.set_tier(bytecode::Tier::Jit);
        engine.set_tier_threshold(0);
        let completion = engine.eval_module(
            "import {current,bump} from 'dep';globalThis.subject=function subject(){return current};globalThis.bump=bump;",
            "root.mjs",
            |name,_| (name=="dep").then(|| ("dep.mjs".into(), "export let current=3;export function bump(){current=9}".into())),
        ).unwrap_or_else(|_| panic!("module parses"));
        assert!(!matches!(completion, crate::Completion::Throw { .. }));
        let code = super::super::call_tests::optimized(&mut engine, "subject", None);
        code.residency.referenced.set(0);
        for (source, expected) in [
            ("subject()", "3"),
            ("bump();subject()", "9"),
            ("subject()", "9"),
        ] {
            assert!(
                matches!(engine.eval(source,false).unwrap(),crate::Completion::Value(value) if value==expected)
            );
            engine.interp.gc_collect();
        }
        assert_eq!(
            code.residency.referenced.get(),
            1,
            "actual native import reader"
        );
        assert_eq!(code.residency.active.get(), 0);
        let calls = TEST_CALLS.with(Cell::get);
        if requested() {
            assert!(
                calls[0] >= 3 && calls[2] >= 3,
                "live imports stay checked after failed shared probes: {calls:?}"
            );
        } else {
            assert_eq!(calls, [0; 4]);
        }
    }

    #[test]
    fn shared_guard_layout_never_captures_a_realm_or_global_value() {
        use crate::realm_inline_guard_tests::{eval_string, in_realm};
        if controlled_children("jit::optimizing::read_guards::tests::shared_guard_layout_never_captures_a_realm_or_global_value") { return; }
        const SOURCE:&str="(function(){'use strict';globalThis.subject=function subject(){return outside+':'+record.x};globalThis.outside=3;globalThis.record={x:5};})();";
        static SNAPSHOT: OnceLock<Vec<u8>> = OnceLock::new();
        let snapshot: &'static [u8] =
            SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).expect("shared snapshot"));
        let run = |interp: &mut crate::interpreter::Interp| {
            let body = interp.shared_snapshot_program(snapshot).unwrap();
            assert!(interp.run_classic_program(&body, None).is_ok());
        };
        let mut engine = crate::Engine::new();
        engine.set_tier(bytecode::Tier::Jit);
        engine.set_tier_threshold(0);
        run(&mut engine.interp);
        let code = super::super::call_tests::optimized(&mut engine, "subject", None);
        assert_eq!(eval_string(&mut engine.interp, "subject()"), "3:5");
        let child = engine.interp.create_realm();
        code.residency.referenced.set(0);
        in_realm(&mut engine.interp, &child, |interp| {
            run(interp);
            assert_eq!(
                eval_string(interp, "outside=13;record.x=17;subject()"),
                "13:17"
            );
            interp.gc_collect();
            assert_eq!(eval_string(interp, "subject()"), "13:17");
        });
        assert_eq!(
            code.residency.referenced.get(),
            1,
            "same native shared AST body in child Realm"
        );
        assert_eq!(code.residency.active.get(), 0);
        assert_eq!(eval_string(&mut engine.interp, "subject()"), "3:5");
    }
}
