//! Own writable-data property stores with guarded, alias-aware packed owner transfers.
//! ECMA-262 e28783d5 OrdinarySetWithOwnDescriptor: a cached key/slot is not an attribute
//! proof. Live descriptor flags always win, and every miss retains full [[Set]] semantics.

use super::*;
use crate::bytecode::{
    IcState, IC_CREATE, IC_OFF_DEPTH, IC_OFF_RECV_SHAPE, IC_OFF_SLOT, PROP_IC_WAYS,
};
use crate::value::PACK_OBJ;

impl Lowering<'_, '_> {
    fn store_receiver(&mut self, op: &Op, slow: Block) -> Value {
        if matches!(op, Op::SetPropThisDrop(..)) {
            let this = self.load(self.ctx, offset_of!(JitCtx, this_raw) as i32);
            let tag = self.property_load(types::I8, this, 0);
            let object = self.b.ins().icmp_imm_s(IntCC::Equal, tag, 8);
            self.property_guard(object, slow);
            let owner = self.load(this, 8);
            self.b.ins().bor_imm_s(owner, PACK_OBJ as i64)
        } else if let Op::SetPropLocalDrop(slot, ..) = op {
            self.local(*slot)
        } else {
            self.stack_read(2)
        }
    }

    pub(super) fn property_write(&mut self, pc: usize, op: &Op, next: Block) {
        let (name, cache) = match *op {
            Op::SetProp(name, cache)
            | Op::SetPropDrop(name, cache)
            | Op::SetPropThisDrop(name, cache)
            | Op::SetPropLocalDrop(_, name, cache) => (name, cache),
            _ => unreachable!("named property write"),
        };
        let index = crate::value::canonical_index(self.chunk.jit_name(name));
        if !elements::supported(self.values) {
            self.checked(set_property, pc as u32, next);
            return;
        }
        let slow = self.b.create_block();
        let receiver = self.store_receiver(op, slow);
        let incoming = self.stack_read(1);
        let object = self.element_receiver(receiver, slow);
        let (property, old, index) = if let Some(index) = index {
            // A constant canonical array index has the same dense-storage and mirror
            // obligations as a computed numeric key, even when the receiver is ordinary.
            let index = self.b.ins().iconst(types::I64, i64::from(index));
            let (property, old) = self.element_property(object, index, slow);
            (property, old, Some(index))
        } else {
            // Array named shapes do not identify arbitrary entry slots. Until that path
            // has its own key proof, only ordinary receivers use the named-slot cache.
            self.property_ordinary(object, slow);
            let creation =
                self.native_creation && creation::supported(self.values, self.chunk.jit_name(name));
            let probe = self.b.create_block();
            self.b.append_block_param(probe, types::I64);
            self.b.append_block_param(probe, types::I32);
            let miss = self.b.create_block();
            let hit = self.b.create_block();
            self.b.append_block_param(hit, types::I64);
            let cache = self
                .b
                .ins()
                .iconst(types::I64, self.chunk.jit_cache_ptr(cache) as i64);
            let ways = self.b.ins().iconst(types::I32, PROP_IC_WAYS as i64);
            self.b.ins().jump(probe, &[cache.into(), ways.into()]);
            self.b.switch_to_block(probe);
            let cache = self.b.block_params(probe)[0];
            let ways = self.b.block_params(probe)[1];
            let depth = self.property_load(types::I8, cache, IC_OFF_DEPTH as usize);
            let shape = self.property_load(types::I32, cache, IC_OFF_RECV_SHAPE as usize);
            self.property_shape(object, shape, miss);
            if creation {
                let create = self.b.create_block();
                let overwrite = self.b.create_block();
                let creates = self
                    .b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, depth, IC_CREATE as i64);
                self.b.ins().brif(creates, create, &[], overwrite, &[]);
                self.b.switch_to_block(create);
                self.property_create(
                    op,
                    object,
                    receiver,
                    incoming,
                    cache,
                    self.chunk.jit_name(name),
                    miss,
                    next,
                );
                // Both native arms start with the original SSA operands. A successful
                // creation already terminates at `next`; misses have made no effects.
                self.stack_state = self.stack_plan.at[pc];
                self.b.switch_to_block(overwrite);
            }
            let own = self.b.ins().icmp_imm_s(IntCC::Equal, depth, 0);
            self.property_guard(own, miss);
            let slot = self.property_load(types::I32, cache, IC_OFF_SLOT as usize);
            let slot = self.b.ins().uextend(types::I64, slot);
            self.b.ins().jump(hit, &[slot.into()]);
            self.b.switch_to_block(miss);
            let remaining = self.b.ins().iadd_imm_s(ways, -1);
            let cache = self
                .b
                .ins()
                .iadd_imm_s(cache, std::mem::size_of::<IcState>() as i64);
            self.b.ins().brif(
                remaining,
                probe,
                &[cache.into(), remaining.into()],
                slow,
                &[],
            );
            self.b.switch_to_block(hit);
            let slot = self.b.block_params(hit)[0];
            let length = self.load(
                object,
                (self.values.obj_props + self.values.props_entries + self.values.vec_len_off)
                    as i32,
            );
            let valid = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, length);
            self.property_guard(valid, slow);
            let data = self.load(
                object,
                (self.values.obj_props + self.values.props_entries + self.values.vec_ptr_off)
                    as i32,
            );
            let offset = self.b.ins().imul_imm_s(slot, self.values.entry_size as i64);
            let property = self.b.ins().iadd(data, offset);
            let old = self.load(property, self.values.entry_value as i32);
            (property, old, None)
        };
        self.property_data(property, true, slow);
        let consumed_receiver = matches!(op, Op::SetProp(..) | Op::SetPropDrop(..));
        let keep = matches!(op, Op::SetProp(..));
        let mut owners = vec![(old, false)];
        if consumed_receiver {
            owners.push((receiver, false));
        }
        if keep && !self.input_types().top.is_copyable() {
            owners.push((incoming, true));
        }
        self.transfer_owners(&owners, slow);
        self.store(incoming, property, self.values.entry_value as i32);
        if let Some(index) = index {
            self.element_mirror_store(object, index, incoming);
        }
        self.finish_property_store(op, incoming, next);
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        self.stack_state = self.stack_plan.at[pc];
        self.checked(set_property, pc as u32, next);
    }

    /// Complete either a native overwrite or creation, after all owner transfers.
    pub(super) fn finish_property_store(&mut self, op: &Op, incoming: Value, next: Block) {
        let consumed_receiver = matches!(op, Op::SetProp(..) | Op::SetPropDrop(..));
        let keep = matches!(op, Op::SetProp(..));
        if keep {
            self.stack_write(2, incoming);
        }
        let consumed = 1 + usize::from(consumed_receiver) - usize::from(keep);
        self.change_top(-8 * consumed as i64);
        self.jump_normal(next);
    }
}
