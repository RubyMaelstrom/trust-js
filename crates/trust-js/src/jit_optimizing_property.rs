//! Native property reads for the whole-function tier.
//!
//! ECMA-262 OrdinaryGet / GetValue / EvaluateCall (official local snapshot e28783d5).
//! A shape identifies keys, not descriptor kind or value. Rewalk live prototype links,
//! recheck descriptor metadata, and read the current value. No cached getter result or
//! prototype identity is substituted. All guards precede ownership changes; misses call
//! the checked operation exactly once with its original receiver and canonical frame.

use super::*;
use crate::bytecode::{
    IcState, IC_MAX_DEPTH, IC_OFF_DEPTH, IC_OFF_HOLDER_SHAPE, IC_OFF_MID2_SHAPE, IC_OFF_MID3_SHAPE,
    IC_OFF_MID4_SHAPE, IC_OFF_MID_OK, IC_OFF_MID_SHAPE, IC_OFF_RECV_SHAPE, IC_OFF_SLOT,
    PROP_IC_WAYS,
};
use crate::value::{JitLayout, PACK_OBJ, PROP_ACCESSOR};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_READ_HELPERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_COMPACT_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// No Rust layout is guessed from a target name. The parent obtains these offsets from the
/// same live-type probe used by the template tier; unsupported layouts retain checked reads.
pub(super) fn supported(layout: &JitLayout) -> bool {
    layout.valid
        && layout.entry_accessor == layout.entry_value + 8
        && [
            layout.obj_from_rc,
            layout.rc_strong_off,
            layout.obj_proto,
            layout.obj_exotic,
            layout.obj_ic_plain,
            layout.obj_props + layout.props_shape,
            layout.obj_props + layout.props_len_slot,
            layout.obj_props + layout.props_entries + layout.vec_ptr_off,
            layout.obj_props + layout.props_entries + layout.vec_len_off,
            layout.entry_size,
            layout.entry_accessor,
            layout.entry_value,
        ]
        .into_iter()
        .all(|offset| offset <= i32::MAX as usize)
}

impl Lowering<'_, '_> {
    pub(super) fn property_guard(&mut self, pass: Value, miss: Block) {
        self.count_heap_guard();
        let next = self.b.create_block();
        self.b.ins().brif(pass, next, &[], miss, &[]);
        self.b.switch_to_block(next);
    }

    pub(super) fn property_load(
        &mut self,
        ty: cranelift_codegen::ir::Type,
        base: Value,
        offset: usize,
    ) -> Value {
        self.b
            .ins()
            .load(ty, MemFlags::trusted(), base, offset as i32)
    }

    pub(super) fn property_plain(&mut self, object: Value, miss: Block) {
        let plain = self.property_load(types::I8, object, self.values.obj_ic_plain);
        self.property_guard(plain, miss);
    }

    pub(super) fn property_ordinary(&mut self, object: Value, miss: Block) {
        let tag = self.property_load(types::I8, object, self.values.obj_exotic);
        let ordinary =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, i64::from(self.values.exotic_none_tag));
        self.property_guard(ordinary, miss);
        self.property_plain(object, miss);
    }

    pub(super) fn property_shape(&mut self, object: Value, expected: Value, miss: Block) {
        let actual = self.property_load(
            types::I32,
            object,
            self.values.obj_props + self.values.props_shape,
        );
        let matches = self.b.ins().icmp(IntCC::Equal, actual, expected);
        self.property_guard(matches, miss);
    }

    pub(super) fn property_read(&mut self, pc: usize, op: &Op, next: Block) {
        let numeric_result = self
            .property_results
            .iter()
            .any(|&(site, ty)| site as usize == pc && ty.is_number());
        if !supported(self.values) {
            if numeric_result {
                self.specialization_bailout(pc);
            } else {
                self.checked(get_property, pc as u32, next);
            }
            return;
        }
        let (name, cache) = match *op {
            Op::GetProp(name, cache)
            | Op::GetPropThis(name, cache)
            | Op::GetMethod(name, cache)
            | Op::GetPropLocal(_, name, cache) => (name, cache),
            _ => unreachable!("named property read"),
        };
        let slow = self.b.create_block();
        let receiver = if matches!(op, Op::GetPropThis(..)) {
            // Value is repr(u8) with fixed discriminants, separate from packed local storage.
            let this = self.load(self.ctx, offset_of!(JitCtx, this_raw) as i32);
            let tag = self.property_load(types::I8, this, 0);
            let object = self.b.ins().icmp_imm_s(IntCC::Equal, tag, 8);
            self.property_guard(object, slow);
            self.load(this, 8)
        } else {
            let word = if let Op::GetPropLocal(slot, ..) = op {
                self.local(*slot)
            } else {
                self.stack_read(1)
            };
            let tag = self.b.ins().ushr_imm_u(word, 48);
            let object = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, (PACK_OBJ >> 48) as i64);
            self.property_guard(object, slow);
            self.b.ins().band_imm_s(word, 0x0000_ffff_ffff_ffff)
        };
        if matches!(op, Op::GetProp(..)) {
            // A consumed last receiver can run real destructors; leave it to the helper.
            let count = self.load(receiver, self.values.rc_strong_off as i32);
            let shared = self
                .b
                .ins()
                .icmp_imm_s(IntCC::UnsignedGreaterThan, count, 1);
            self.property_guard(shared, slow);
        }
        let receiver_object = self
            .b
            .ins()
            .iadd_imm_s(receiver, self.values.obj_from_rc as i64);
        self.property_plain(receiver_object, slow);
        let exotic = self.property_load(types::I8, receiver_object, self.values.obj_exotic);
        let ordinary =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, exotic, self.values.exotic_none_tag as i64);
        // Arrays may prove ABSENCE of a non-index named property below an ordinary holder,
        // never a holder slot: indexed elements do not participate in their named shape.
        let array = if !self
            .chunk
            .jit_name(name)
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_digit)
        {
            Some(
                self.b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, exotic, self.values.exotic_array_tag as i64),
            )
        } else {
            None
        };
        let receiver_ok = array.map_or(ordinary, |array| self.b.ins().bor(ordinary, array));
        self.property_guard(receiver_ok, slow);

        let hit = self.b.create_block();
        self.b.append_block_param(hit, types::I64);
        self.b.append_block_param(hit, types::I64);
        let compact = self.chunk.jit_cache_preferred(cache).filter(|state| {
            state.depth <= IC_MAX_DEPTH
                && (state.depth < 2 || {
                    let needed = (1u8 << (state.depth - 1)) - 1;
                    state.mid_ok & needed == needed
                })
        });
        if let Some(state) = compact {
            // Compile observed monomorphic shape facts, not an entire empty-cache resolver.
            // Identity is NOT embedded: every hit rewalks live links and reads live flags/value.
            // Changed shapes take the exact operation's fallback before any ownership effect.
            #[cfg(test)]
            TEST_COMPACT_READS.with(|count| count.set(count.get() + 1));
            if state.depth == 0 {
                self.property_guard(ordinary, slow);
            }
            let shape = self.b.ins().iconst(types::I32, state.recv_shape as i64);
            self.property_shape(receiver_object, shape, slow);
            let mut holder = receiver_object;
            for hop in 1..=state.depth {
                let proto = self.load(holder, self.values.obj_proto as i32);
                self.property_guard(proto, slow);
                holder = self
                    .b
                    .ins()
                    .iadd_imm_s(proto, self.values.obj_from_rc as i64);
                self.property_ordinary(holder, slow);
                let shape = if hop == state.depth {
                    state.holder_shape
                } else {
                    [
                        state.mid_shape,
                        state.mid2_shape,
                        state.mid3_shape,
                        state.mid4_shape,
                    ][hop as usize - 1]
                };
                let shape = self.b.ins().iconst(types::I32, shape as i64);
                self.property_shape(holder, shape, slow);
            }
            let slot = self.b.ins().iconst(types::I64, state.slot as i64);
            self.b.ins().jump(hit, &[holder.into(), slot.into()]);
        } else {
            // One bounded native probe body for the entire polymorphic site. No helper and no
            // author code can mutate these cells/prototypes while the probe is in progress.
            let probe = self.b.create_block();
            self.b.append_block_param(probe, types::I64);
            self.b.append_block_param(probe, types::I32);
            let miss = self.b.create_block();
            let cache = self
                .b
                .ins()
                .iconst(types::I64, self.chunk.jit_cache_ptr(cache) as i64);
            let ways = self.b.ins().iconst(types::I32, PROP_IC_WAYS as i64);
            self.b.ins().jump(probe, &[cache.into(), ways.into()]);
            self.b.switch_to_block(probe);
            let cache = self.b.block_params(probe)[0];
            let remaining = self.b.block_params(probe)[1];
            let slot = self.property_load(types::I32, cache, IC_OFF_SLOT as usize);
            let slot = self.b.ins().uextend(types::I64, slot);
            let depth = self.property_load(types::I8, cache, IC_OFF_DEPTH as usize);
            let data =
                self.b
                    .ins()
                    .icmp_imm_s(IntCC::UnsignedLessThanOrEqual, depth, IC_MAX_DEPTH as i64);
            self.property_guard(data, miss); // empty/absent/accessor/array-slot states stay checked
            if let Some(array) = array {
                let inherited = self.b.ins().icmp_imm_s(IntCC::NotEqual, depth, 0);
                let inherited_array = self.b.ins().band(array, inherited);
                let receiver_ok = self.b.ins().bor(ordinary, inherited_array);
                self.property_guard(receiver_ok, miss);
            }
            let shape = self.property_load(types::I32, cache, IC_OFF_RECV_SHAPE as usize);
            self.property_shape(receiver_object, shape, miss);
            let mut holder = receiver_object;
            for hop in 0..=usize::from(IC_MAX_DEPTH) {
                let reached = self.b.ins().icmp_imm_s(IntCC::Equal, depth, hop as i64);
                let deeper = self.b.create_block();
                self.b
                    .ins()
                    .brif(reached, hit, &[holder.into(), slot.into()], deeper, &[]);
                self.b.switch_to_block(deeper);
                if hop == usize::from(IC_MAX_DEPTH) {
                    self.b.ins().jump(miss, &[]);
                    break;
                }
                let proto = self.load(holder, self.values.obj_proto as i32);
                self.property_guard(proto, miss);
                holder = self
                    .b
                    .ins()
                    .iadd_imm_s(proto, self.values.obj_from_rc as i64);
                self.property_ordinary(holder, miss);
                let last = self
                    .b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, depth, (hop + 1) as i64);
                let holder_shape =
                    self.property_load(types::I32, cache, IC_OFF_HOLDER_SHAPE as usize);
                let expected = if hop < 4 {
                    let valid = self.property_load(types::I8, cache, IC_OFF_MID_OK as usize);
                    let valid = self.b.ins().band_imm_s(valid, 1 << hop);
                    let valid = self.b.ins().bor(valid, last);
                    self.property_guard(valid, miss);
                    let offset = [
                        IC_OFF_MID_SHAPE,
                        IC_OFF_MID2_SHAPE,
                        IC_OFF_MID3_SHAPE,
                        IC_OFF_MID4_SHAPE,
                    ][hop];
                    let mid = self.property_load(types::I32, cache, offset as usize);
                    self.b.ins().select(last, holder_shape, mid)
                } else {
                    holder_shape
                };
                self.property_shape(holder, expected, miss);
            }
            self.b.switch_to_block(miss);
            let remaining = self.b.ins().iadd_imm_s(remaining, -1);
            let next_cache = self
                .b
                .ins()
                .iadd_imm_s(cache, std::mem::size_of::<IcState>() as i64);
            self.b.ins().brif(
                remaining,
                probe,
                &[next_cache.into(), remaining.into()],
                slow,
                &[],
            );
        }
        self.b.switch_to_block(hit);
        let holder = self.b.block_params(hit)[0];
        let slot = self.b.block_params(hit)[1];
        let length = self.load(
            holder,
            (self.values.obj_props + self.values.props_entries + self.values.vec_len_off) as i32,
        );
        let valid = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, length);
        self.property_guard(valid, slow);
        let entries = self.load(
            holder,
            (self.values.obj_props + self.values.props_entries + self.values.vec_ptr_off) as i32,
        );
        let offset = self.b.ins().imul_imm_s(slot, self.values.entry_size as i64);
        let entry = self.b.ins().iadd(entries, offset);
        let flags = self.property_load(types::I8, entry, self.values.entry_accessor);
        let accessor = self.b.ins().band_imm_s(flags, PROP_ACCESSOR as i64);
        let data = self.b.ins().icmp_imm_s(IntCC::Equal, accessor, 0);
        self.property_guard(data, slow);
        let value = self.load(entry, self.values.entry_value as i32);
        if numeric_result {
            let number = self.number(value);
            self.property_guard(number, slow);
        } else {
            self.retain_word(value, slow);
        }
        // Commit: from here no guard, helper, safepoint or other observer may intervene.
        if matches!(op, Op::GetProp(..)) {
            // Re-read after retaining: a self-valued property can name this same owner.
            let count = self.load(receiver, self.values.rc_strong_off as i32);
            let count = self.b.ins().iadd_imm_s(count, -1);
            self.store(count, receiver, self.values.rc_strong_off as i32);
            self.count_runtime(diagnostics::RELEASE, 1);
            self.stack_write(1, value);
        } else {
            self.push(value);
        }
        self.jump_normal(next);
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        self.stack_state = self.stack_plan.at[pc];
        if numeric_result {
            // The complete frame is still before this access. In particular an accessor
            // miss must execute its getter once in the VM, never replay an observed call.
            self.specialization_bailout(pc);
        } else {
            self.checked(get_property, pc as u32, next);
        }
    }
}
