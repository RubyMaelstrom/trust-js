//! Native GetBindingValue for live declarative/global name-cache proofs.
//!
//! ECMA-262 e28783d5 #sec-getidentifierreference,
//! #sec-declarative-environment-records-getbindingvalue-n-s, #sec-getvalue,
//! #sec-object-environment-records-hasbinding-n and #sec-evaluatecall.
//! Every load checks the current environment, resolution proof and value. With
//! environments, imports, TDZ, accessors and unsupported cache modes use the
//! original checked operation, once. No cached value or binding pointer is
//! embedded in generated code. Retains follow all resolution/representation guards.

use super::*;
use crate::bytecode::lexical_cache::{
    NativeNameDescriptor, DEEP_NAME_IC, NAME_GUARD_GENERATION, NAME_GUARD_IDENTITY,
    NAME_GUARD_LAYOUT, NAME_GUARD_SIZE,
};
use crate::bytecode::{NAME_IC_OFF_ACT_GEN, NAME_IC_OFF_BINDING, NAME_IC_OFF_ENV, NAME_IC_OFF_GEN};
use crate::value::{JitLayout, PACK_BIGINT, PACK_OBJ, PACK_STR, PACK_SYM, PROP_ACCESSOR};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_NAME_HELPERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TRUST_JS_OPT_JIT_NAMES").as_deref() != Ok("0"))
}

/// Native resolution expands one bytecode into live environment, descriptor and
/// representation guards. Charge this work to admission as well as compilation;
/// callers retain their absolute IR caps.
pub(super) fn instruction_allowance(chunk: &Chunk) -> usize {
    if !enabled() {
        return 0;
    }
    chunk
        .jit_ops()
        .iter()
        .filter(|op| match op {
            Op::LoadName(..) | Op::LoadNameForCall(..) => true,
            Op::LoadCap(_) => !chunk.jit_needs_activation_state(),
            _ => false,
        })
        .count()
        .saturating_mul(384)
}

fn supported(layout: &JitLayout) -> bool {
    property::supported(layout)
        && layout.scope_with_valid
        && [
            layout.scope_gen,
            layout.scope_layout,
            layout.scope_with,
            layout.scope_small_tag,
            layout.scope_small_vec + layout.vec_ptr_off,
            layout.scope_small_vec + layout.vec_len_off,
            layout.scope_binding_stride,
            layout.scope_binding_offset,
            layout.binding_value + 8,
            layout.binding_init,
            layout.binding_import,
        ]
        .into_iter()
        .all(|offset| offset <= i32::MAX as usize)
}

impl Lowering<'_, '_> {
    fn name_scope_plain(&mut self, scope: Value, slow: Block) {
        let with = self.property_load(types::I8, scope, self.values.scope_with);
        let absent =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, with, self.values.scope_with_none as i64);
        self.property_guard(absent, slow);
    }

    fn name_generation(&mut self, scope: Value, expected: Value, slow: Block) {
        let valid = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, expected, u32::MAX as i64);
        self.property_guard(valid, slow);
        let actual = self.property_load(types::I32, scope, self.values.scope_gen);
        let matches = self.b.ins().icmp(IntCC::Equal, actual, expected);
        self.property_guard(matches, slow);
    }

    fn name_layout(&mut self, scope: Value, expected: Value, slow: Block) {
        self.property_guard(expected, slow);
        let actual = self.property_load(types::I32, scope, self.values.scope_layout);
        let matches = self.b.ins().icmp(IntCC::Equal, actual, expected);
        self.property_guard(matches, slow);
    }

    fn name_fixed_binding(&mut self, scope: Value, slot: Value, slow: Block) -> Value {
        if !self.values.scope_small_valid {
            self.b.ins().jump(slow, &[]);
            let dead = self.b.create_block();
            self.b.switch_to_block(dead);
            return self.b.ins().iconst(types::I64, 0);
        }
        let tag = self.property_load(types::I8, scope, self.values.scope_small_tag);
        let small = self.b.ins().icmp_imm_s(IntCC::Equal, tag, 0);
        let indexed = self.b.ins().icmp_imm_s(IntCC::Equal, tag, 2);
        let ordered = self.b.ins().bor(small, indexed);
        self.property_guard(ordered, slow);
        let length = self.property_load(
            types::I64,
            scope,
            self.values.scope_small_vec + self.values.vec_len_off,
        );
        let inside = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, length);
        self.property_guard(inside, slow);
        let entries = self.property_load(
            types::I64,
            scope,
            self.values.scope_small_vec + self.values.vec_ptr_off,
        );
        let offset = self
            .b
            .ins()
            .imul_imm_s(slot, self.values.scope_binding_stride as i64);
        let binding = self.b.ins().iadd(entries, offset);
        self.b
            .ins()
            .iadd_imm_s(binding, self.values.scope_binding_offset as i64)
    }

    fn name_wide_word(&mut self, binding: Value, slow: Block) -> Value {
        self.property_guard(binding, slow);
        let initialized = self.property_load(types::I8, binding, self.values.binding_init);
        self.property_guard(initialized, slow);
        let imported = self.property_load(types::I8, binding, self.values.binding_import);
        let ordinary = self.b.ins().icmp_imm_s(IntCC::Equal, imported, 0);
        self.property_guard(ordinary, slow);
        let tag_word = self.property_load(types::I64, binding, self.values.binding_value);
        let payload = self.property_load(types::I64, binding, self.values.binding_value + 8);
        let tag = self.b.ins().band_imm_s(tag_word, 255);
        let number = self.b.create_block();
        let boolean = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        let mut switch = Switch::new();
        switch.set_entry(4, number);
        switch.set_entry(3, boolean);
        let mut cases = Vec::new();
        for (tag, bits, owner) in [
            (0, PACK_UNDEFINED, false),
            (1, PACK_EMPTY, false),
            (2, PACK_NULL, false),
            (5, PACK_BIGINT, true),
            (6, PACK_STR, true),
            (7, PACK_SYM, true),
            (8, PACK_OBJ, true),
        ] {
            let block = self.b.create_block();
            switch.set_entry(tag, block);
            cases.push((block, bits, owner));
        }
        switch.emit(&mut self.b, tag, slow);
        for (block, bits, owner) in cases {
            self.b.switch_to_block(block);
            let bits = self.b.ins().iconst(types::I64, bits as i64);
            let value = if owner {
                self.b.ins().bor(payload, bits)
            } else {
                bits
            };
            self.b.ins().jump(done, &[value.into()]);
        }
        self.b.switch_to_block(number);
        let number = self.b.ins().bitcast(types::F64, MemFlags::new(), payload);
        let value = self.packed_number(number);
        self.b.ins().jump(done, &[value.into()]);
        self.b.switch_to_block(boolean);
        // Value::Bool keeps its byte at the payload offset; the rest of that word is padding.
        let bit = self.b.ins().band_imm_s(payload, 1);
        let value = self.b.ins().bor_imm_s(bit, PACK_BOOL as i64);
        self.b.ins().jump(done, &[value.into()]);
        self.b.switch_to_block(done);
        self.b.block_params(done)[0]
    }

    fn name_deep_read(
        &mut self,
        descriptor: Value,
        first_scope: Value,
        slow: Block,
        binding: Block,
        global: Block,
    ) {
        if !self.values.scope_parent_valid {
            self.b.ins().jump(slow, &[]);
            return;
        }
        self.property_guard(descriptor, slow);
        let guards = self.property_load(
            types::I64,
            descriptor,
            offset_of!(NativeNameDescriptor, guards),
        );
        let length = self.property_load(
            types::I64,
            descriptor,
            offset_of!(NativeNameDescriptor, len),
        );
        self.property_guard(length, slow);
        let bounded = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedLessThanOrEqual, length, 16);
        self.property_guard(bounded, slow);
        let head = self.b.create_block();
        for _ in 0..3 {
            self.b.append_block_param(head, types::I64);
        }
        self.b
            .ins()
            .jump(head, &[first_scope.into(), guards.into(), length.into()]);
        self.b.switch_to_block(head);
        let scope = self.b.block_params(head)[0];
        let guard = self.b.block_params(head)[1];
        let left = self.b.block_params(head)[2];
        self.name_scope_plain(scope, slow);
        let identity = self.property_load(types::I64, guard, NAME_GUARD_IDENTITY);
        let generation = self.property_load(types::I32, guard, NAME_GUARD_GENERATION);
        let actual = self.property_load(types::I32, scope, self.values.scope_gen);
        let same = self.b.ins().icmp(IntCC::Equal, identity, scope);
        let fresh = self.b.ins().icmp(IntCC::Equal, actual, generation);
        let valid = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, generation, u32::MAX as i64);
        let exact = self.b.ins().band(same, fresh);
        let exact = self.b.ins().band(exact, valid);
        let layout = self.b.create_block();
        let checked = self.b.create_block();
        self.b.ins().brif(exact, checked, &[], layout, &[]);
        self.b.switch_to_block(layout);
        let layout = self.property_load(types::I32, guard, NAME_GUARD_LAYOUT);
        self.name_layout(scope, layout, slow);
        self.b.ins().jump(checked, &[]);
        self.b.switch_to_block(checked);
        let remaining = self.b.ins().iadd_imm_s(left, -1);
        let advance = self.b.create_block();
        let holder = self.b.create_block();
        self.b.ins().brif(remaining, advance, &[], holder, &[]);
        self.b.switch_to_block(advance);
        let parent = self.property_load(types::I64, scope, self.values.scope_parent);
        self.property_guard(parent, slow);
        let parent = self
            .b
            .ins()
            .iadd_imm_s(parent, self.values.scope_data_off as i64);
        let next_guard = self.b.ins().iadd_imm_s(guard, NAME_GUARD_SIZE as i64);
        self.b
            .ins()
            .jump(head, &[parent.into(), next_guard.into(), remaining.into()]);
        self.b.switch_to_block(holder);
        let kind = self.property_load(
            types::I32,
            descriptor,
            offset_of!(NativeNameDescriptor, kind),
        );
        let lexical = self.b.create_block();
        let global_holder = self.b.create_block();
        self.b.ins().brif(kind, global_holder, &[], lexical, &[]);
        self.b.switch_to_block(lexical);
        let direct = self.b.create_block();
        let indexed = self.b.create_block();
        self.b.ins().brif(exact, direct, &[], indexed, &[]);
        self.b.switch_to_block(direct);
        let pointer = self.property_load(
            types::I64,
            descriptor,
            offset_of!(NativeNameDescriptor, binding),
        );
        self.b.ins().jump(binding, &[pointer.into()]);
        self.b.switch_to_block(indexed);
        let slot = self.property_load(
            types::I64,
            descriptor,
            offset_of!(NativeNameDescriptor, slot),
        );
        let pointer = self.name_fixed_binding(scope, slot, slow);
        self.b.ins().jump(binding, &[pointer.into()]);
        self.b.switch_to_block(global_holder);
        let ordinary_global = self.b.ins().icmp_imm_s(IntCC::Equal, kind, 1);
        self.property_guard(ordinary_global, slow);
        let global_scope = self.load(self.ctx, offset_of!(JitCtx, genv) as i32);
        let same = self.b.ins().icmp(IntCC::Equal, global_scope, scope);
        self.property_guard(same, slow);
        let shape = self.property_load(
            types::I32,
            descriptor,
            offset_of!(NativeNameDescriptor, shape),
        );
        let slot = self.property_load(
            types::I64,
            descriptor,
            offset_of!(NativeNameDescriptor, slot),
        );
        self.b.ins().jump(global, &[slot.into(), shape.into()]);
    }

    pub(super) fn name_read(&mut self, pc: usize, op: &Op, next: Block) {
        if !supported(self.values) || !enabled() {
            self.checked(exec, pc as u32, next);
            return;
        }
        let cache_address = match *op {
            Op::LoadName(_, cache) | Op::LoadNameForCall(_, cache) => {
                self.chunk.jit_name_cache_ptr(cache)
            }
            Op::LoadCap(name) if !self.chunk.jit_needs_activation_state() => {
                self.chunk.jit_cap_cache_ptr(name)
            }
            _ => {
                self.checked(exec, pc as u32, next);
                return;
            }
        };
        let slow = self.b.create_block();
        let direct = self.b.create_block();
        let global = self.b.create_block();
        let global_value = self.b.create_block();
        self.b.append_block_param(global_value, types::I64);
        self.b.append_block_param(global_value, types::I32);
        let fixed = self.b.create_block();
        let parent = self.b.create_block();
        let deep = self.b.create_block();
        let binding = self.b.create_block();
        self.b.append_block_param(binding, types::I64);
        let loaded = self.b.create_block();
        self.b.append_block_param(loaded, types::I64);
        let cache = self.b.ins().iconst(types::I64, cache_address as i64);
        let scope = self.load(self.ctx, offset_of!(JitCtx, env_raw) as i32);
        self.name_scope_plain(scope, slow);
        let cached_scope = self.property_load(types::I64, cache, NAME_IC_OFF_ENV as usize);
        let generation = self.property_load(types::I32, cache, NAME_IC_OFF_GEN as usize);
        let location = self.property_load(types::I64, cache, NAME_IC_OFF_BINDING as usize);
        let is_direct = self.b.ins().icmp(IntCC::Equal, cached_scope, scope);
        let dispatch = self.b.create_block();
        self.b.ins().brif(is_direct, direct, &[], dispatch, &[]);
        self.b.switch_to_block(dispatch);
        let global_tag = self.b.ins().bor_imm_s(scope, 1);
        let is_global = self.b.ins().icmp(IntCC::Equal, cached_scope, global_tag);
        let dispatch = self.b.create_block();
        self.b.ins().brif(is_global, global, &[], dispatch, &[]);
        self.b.switch_to_block(dispatch);
        let is_fixed =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, cached_scope, bytecode::FIXED_NAME_IC as i64);
        let dispatch = self.b.create_block();
        self.b.ins().brif(is_fixed, fixed, &[], dispatch, &[]);
        self.b.switch_to_block(dispatch);
        let is_deep = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, cached_scope, DEEP_NAME_IC as i64);
        let dispatch = self.b.create_block();
        self.b.ins().brif(is_deep, deep, &[], dispatch, &[]);
        self.b.switch_to_block(dispatch);
        let is_parent = self.b.ins().band_imm_s(cached_scope, 2);
        self.b.ins().brif(is_parent, parent, &[], slow, &[]);

        self.b.switch_to_block(direct);
        self.name_generation(scope, generation, slow);
        self.b.ins().jump(binding, &[location.into()]);

        self.b.switch_to_block(fixed);
        self.name_layout(scope, generation, slow);
        let target = self.name_fixed_binding(scope, location, slow);
        self.b.ins().jump(binding, &[target.into()]);

        self.b.switch_to_block(parent);
        let activation = self.property_load(types::I32, cache, NAME_IC_OFF_ACT_GEN as usize);
        let layout_mode = self.b.ins().band_imm_s(cached_scope, 4);
        let layout_guard = self.b.create_block();
        let generation_guard = self.b.create_block();
        let parent_ready = self.b.create_block();
        self.b
            .ins()
            .brif(layout_mode, layout_guard, &[], generation_guard, &[]);
        self.b.switch_to_block(layout_guard);
        self.name_layout(scope, activation, slow);
        self.b.ins().jump(parent_ready, &[]);
        self.b.switch_to_block(generation_guard);
        self.name_generation(scope, activation, slow);
        self.b.ins().jump(parent_ready, &[]);
        self.b.switch_to_block(parent_ready);
        let parent = self.load(self.ctx, offset_of!(JitCtx, env_parent_raw) as i32);
        self.property_guard(parent, slow);
        let expected = self.b.ins().band_imm_s(cached_scope, -8);
        let matches = self.b.ins().icmp(IntCC::Equal, parent, expected);
        self.property_guard(matches, slow);
        self.name_scope_plain(parent, slow);
        self.name_generation(parent, generation, slow);
        self.b.ins().jump(binding, &[location.into()]);

        self.b.switch_to_block(deep);
        self.name_deep_read(location, scope, slow, binding, global_value);

        self.b.switch_to_block(global);
        self.name_generation(scope, generation, slow);
        let shape = self.b.ins().ushr_imm_u(location, 32);
        let shape = self.b.ins().ireduce(types::I32, shape);
        let slot = self.b.ins().band_imm_s(location, u32::MAX as i64);
        self.b
            .ins()
            .jump(global_value, &[slot.into(), shape.into()]);
        self.b.switch_to_block(global_value);
        let slot = self.b.block_params(global_value)[0];
        let shape = self.b.block_params(global_value)[1];
        let object = self.load(self.ctx, offset_of!(JitCtx, global_body) as i32);
        self.property_guard(object, slow);
        self.property_ordinary(object, slow);
        self.property_shape(object, shape, slow);
        let length = self.property_load(
            types::I64,
            object,
            self.values.obj_props + self.values.props_entries + self.values.vec_len_off,
        );
        let inside = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, length);
        self.property_guard(inside, slow);
        let entries = self.property_load(
            types::I64,
            object,
            self.values.obj_props + self.values.props_entries + self.values.vec_ptr_off,
        );
        let offset = self.b.ins().imul_imm_s(slot, self.values.entry_size as i64);
        let entry = self.b.ins().iadd(entries, offset);
        let flags = self.property_load(types::I8, entry, self.values.entry_accessor);
        let accessor = self.b.ins().band_imm_s(flags, PROP_ACCESSOR as i64);
        let data = self.b.ins().icmp_imm_s(IntCC::Equal, accessor, 0);
        self.property_guard(data, slow);
        let value = self.property_load(types::I64, entry, self.values.entry_value);
        self.b.ins().jump(loaded, &[value.into()]);

        self.b.switch_to_block(binding);
        let target = self.b.block_params(binding)[0];
        let value = self.name_wide_word(target, slow);
        self.b.ins().jump(loaded, &[value.into()]);
        self.b.switch_to_block(loaded);
        let value = self.b.block_params(loaded)[0];
        self.retain_word(value, slow);
        if matches!(op, Op::LoadNameForCall(..)) {
            let receiver = self.b.ins().iconst(types::I64, PACK_UNDEFINED as i64);
            self.push(receiver);
        }
        self.push(value);
        self.jump_normal(next);
        self.slow(pc, slow, next);
    }
}
