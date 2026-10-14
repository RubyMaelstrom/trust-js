//! Loop-scoped heap facts above Cranelift, not key-shape guesses.
//!
//! ECMA-262 e28783d5 #sec-ordinaryget and #sec-ordinarysetwithowndescriptor:
//! and #sec-array-exotic-objects: sampled own numeric data properties become
//! guarded scalar values between effects. Private receiver identities remain
//! fixed. Calls and checked operations publish fields before observing them;
//! normal returns reload the live descriptors, and a miss resumes AFTER that
//! operation. General native writes also invalidate the proof on their fast
//! paths. Polls publish roots and revalidate. No effect or getter is replayed.

use super::*;
use crate::value::{PACK_OBJ, PROP_ACCESSOR, PROP_WRITABLE};

/// Inspect a live own descriptor without invoking [[Get]], coercion or user code.
/// Array key layouts do not identify a named slot: the live key is checked here,
/// and generated code separately checks the maintained length-slot memo.
pub(super) unsafe fn sample(ctx: &JitCtx, pc: usize) -> bool {
    let chunk = &*ctx.chunk;
    let Some(&Op::GetPropLocal(slot, name, cache)) = chunk.jit_ops().get(pc) else {
        return false;
    };
    if usize::from(slot) >= ctx.n_slots {
        return false;
    }
    let Some(receiver) = (&*ctx.slots.add(usize::from(slot))).sampled_object() else {
        return false;
    };
    let Some(state) = chunk.jit_cache_preferred(cache) else {
        return false;
    };
    let Ok(object) = receiver.try_borrow() else {
        return false;
    };
    if !object.ic_plain.get() || object.props.shape() != state.recv_shape {
        return false;
    }
    let kind = match object.exotic {
        crate::value::Exotic::None => state.depth == 0,
        crate::value::Exotic::Array => {
            chunk.jit_name(name) == "length" && state.depth == bytecode::IC_ARR_KEYCHK
        }
        _ => false,
    };
    if !kind {
        return false;
    }
    let Some((key, property)) = object.props.entry_at(state.slot as usize) else {
        return false;
    };
    key.as_ref() == chunk.jit_name(name) && property.number_value().is_some()
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_PLANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy)]
pub(super) struct Field {
    receiver: u16,
    shape: u32,
    slot: u32,
    name: u32,
    writable: bool,
    array_length: bool,
}

pub(super) struct Plan {
    pub(super) values: value_facts::ValuePlan,
    fields: Vec<Field>,
    sites: Vec<Option<usize>>,
    region: Vec<bool>,
    borrowed_loads: Vec<bool>,
    owned_stores: Vec<bool>,
    pub(super) observes: bool,
    native_writes: Vec<bool>,
}

impl Plan {
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        entry: usize,
        inputs: &[(u16, value_facts::Types)],
        sampled: &[u32],
    ) -> Option<Self> {
        if std::env::var("TRUST_JS_OPT_JIT_LOOP_FACTS").as_deref() == Ok("0")
            || std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0")
            || !cfg.handler_roots().is_empty()
        {
            return None;
        }
        let natural = cfg
            .loops()
            .iter()
            .find(|natural| cfg.blocks()[natural.header.0 as usize].start == entry)?;
        let ops = chunk.jit_ops();
        let mut region = vec![false; ops.len()];
        for block in &natural.blocks {
            let block = &cfg.blocks()[block.0 as usize];
            region[block.start..block.end].fill(true);
        }
        // Reject irreducible side entries, and any exit that could later reenter
        // after running code outside this effect proof.
        for id in &natural.blocks {
            if *id != natural.header
                && cfg.blocks()[id.0 as usize]
                    .predecessors
                    .iter()
                    .any(|predecessor| !natural.blocks.contains(predecessor))
            {
                return None;
            }
        }
        let mut pending: Vec<_> = natural.exits.iter().map(|&(_, to)| to).collect();
        let mut seen = vec![false; cfg.blocks().len()];
        while let Some(id) = pending.pop() {
            if natural.blocks.contains(&id) {
                return None;
            }
            if !std::mem::replace(&mut seen[id.0 as usize], true) {
                pending.extend_from_slice(&cfg.blocks()[id.0 as usize].successors);
            }
        }
        let mut fields: Vec<Field> = Vec::new();
        let mut sites = vec![None; ops.len()];
        let mut borrowed_loads = vec![false; ops.len()];
        let mut owned_stores = vec![false; ops.len()];
        let mut native_writes = vec![false; ops.len()];
        let mut store_receivers = vec![None; ops.len()];
        for id in &natural.blocks {
            let block = &cfg.blocks()[id.0 as usize];
            // Origins never cross a control-flow edge or a duplicate. A
            // receiver used by one later store can remain borrowed through
            // intervening scalar calculations in this same effect-closed block.
            let mut origins = vec![None; stack.at[block.start].depth];
            for pc in block.start..block.end {
                if matches!(ops[pc], Op::SetProp(..) | Op::SetPropDrop(..)) {
                    store_receivers[pc] = origins
                        .get(origins.len().checked_sub(2)?)
                        .copied()
                        .flatten();
                }
                let (pops, pushes) = chunk.jit_stack_effect(pc)?;
                origins.truncate(origins.len().checked_sub(pops)?);
                let origin = match ops[pc] {
                    Op::LoadLocal(slot) => Some((slot, pc)),
                    _ => None,
                };
                origins.extend(std::iter::repeat_n(origin, pushes));
            }
        }
        let mut properties = Vec::new();
        for &pc in sampled {
            let pc = pc as usize;
            if !region.get(pc).copied().unwrap_or(false) {
                continue;
            }
            let Op::GetPropLocal(receiver, name, cache) = ops[pc] else {
                continue;
            };
            if crate::value::canonical_index(chunk.jit_name(name)).is_some()
                || ops
                    .iter()
                    .enumerate()
                    .any(|(at, op)| region[at] && effects::checked_writes(op).contains(receiver))
            {
                continue;
            }
            let state = chunk.jit_cache_preferred(cache)?;
            let array_length =
                state.depth == bytecode::IC_ARR_KEYCHK && chunk.jit_name(name) == "length";
            if state.depth != 0 && !array_length {
                continue;
            }
            let field = if let Some(index) = fields.iter().position(|field| {
                field.receiver == receiver
                    && field.shape == state.recv_shape
                    && field.slot == state.slot
            }) {
                index
            } else {
                if fields.len() == 16 {
                    return None;
                }
                fields.push(Field {
                    receiver,
                    shape: state.recv_shape,
                    slot: state.slot,
                    name,
                    writable: false,
                    array_length,
                });
                fields.len() - 1
            };
            sites[pc] = Some(field);
            properties.push((pc as u32, value_facts::Types::NUMBER));
        }
        if fields.is_empty() || properties.is_empty() {
            return None;
        }
        let values = value_facts::ValuePlan::at_entry_with_properties(
            chunk,
            cfg,
            stack,
            entry,
            inputs,
            &properties,
        )?;
        // Only proved Number stores can stay in scalar variables. Other stores
        // execute their complete ordinary lowering and invalidate live heap facts.
        for (pc, op) in ops.iter().enumerate().filter(|(pc, _)| region[*pc]) {
            let receiver = match *op {
                Op::SetPropLocalDrop(receiver, name, cache) => Some((receiver, name, cache)),
                Op::SetProp(name, cache) | Op::SetPropDrop(name, cache) => {
                    store_receivers[pc].map(|(receiver, _)| (receiver, name, cache))
                }
                _ => None,
            };
            if let Some((receiver, _, cache)) = receiver {
                if let Some(state) = chunk.jit_cache_preferred(cache) {
                    if values.at[pc].top.is_number() && state.depth == 0 {
                        if let Some(index) = fields.iter().position(|field| {
                            field.receiver == receiver
                                && field.shape == state.recv_shape
                                && field.slot == state.slot
                                && !field.array_length
                        }) {
                            fields[index].writable = true;
                            sites[pc] = Some(index);
                        }
                    }
                }
            }
            // Existing own element overwrites cannot change named numeric fields
            // or Array length. Guard misses still take the full checked barrier.
            // Native named creation uses reserved storage without moving fields;
            // only the same property key can alias a scalar value on a fast hit.
            let written_name = match op {
                Op::SetProp(name, _)
                | Op::SetPropDrop(name, _)
                | Op::SetPropThisDrop(name, _)
                | Op::SetPropLocalDrop(_, name, _) => Some(*name),
                _ => None,
            };
            native_writes[pc] = sites[pc].is_none()
                && written_name.is_some_and(|name| {
                    fields
                        .iter()
                        .any(|field| chunk.jit_name(field.name) == chunk.jit_name(name))
                });
        }
        // Borrow receiver operands only when the WHOLE region remains closed.
        // Otherwise normal owners protect operands across every checked boundary.
        let closed = |pc: usize, op: &Op| {
            let ty = values.at[pc];
            let copy = |ty: value_facts::Types| ty.is_copyable() && ty.initialized();
            match op {
                Op::Const(index) => chunk.jit_const_copyable(*index),
                Op::Undef | Op::Jump(_) => true,
                Op::LoadLocal(_)
                    if store_receivers.iter().enumerate().any(|(at, origin)| {
                        sites[at].is_some() && origin.is_some_and(|(_, producer)| producer == pc)
                    }) =>
                {
                    true
                }
                Op::LoadLocal(_) => copy(ty.local),
                Op::StoreLocal(slot) => {
                    copy(ty.local)
                        && copy(ty.top)
                        && fields.iter().all(|field| field.receiver != *slot)
                }
                Op::UpdateLocal(slot, _) => {
                    ty.local.is_number() && fields.iter().all(|field| field.receiver != *slot)
                }
                Op::Dup
                | Op::Pop
                | Op::JumpIfFalse(_)
                | Op::JumpIfFalsePeek(_)
                | Op::JumpIfTruePeek(_)
                | Op::JumpIfNotNullishPeek(_) => copy(ty.top),
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
                | Op::StrictNotEq => ty.top.is_number() && ty.second.is_number(),
                Op::GetPropLocal(..) => sites[pc].is_some(),
                Op::SetPropLocalDrop(..) | Op::SetProp(..) | Op::SetPropDrop(..) => {
                    sites[pc].is_some() && ty.top.is_number()
                }
                _ => false,
            }
        };
        let observes = ops
            .iter()
            .enumerate()
            .any(|(pc, op)| region[pc] && !closed(pc, op));
        for (pc, op) in ops.iter().enumerate().filter(|(pc, _)| region[*pc]) {
            // Calls/ordinary checked operations preserve private receiver words.
            // Handlers, suspension and unreviewed slot bridges remain excluded.
            if matches!(effects::checked_writes(op), SlotWrites::All)
                || stack.at[pc].floor != 0
                || stack.forget_to[pc] != 0
            {
                return None;
            }
            if sites[pc].is_some() && matches!(op, Op::SetProp(..) | Op::SetPropDrop(..)) {
                let (_, producer) = store_receivers[pc]?;
                if observes {
                    owned_stores[pc] = true;
                } else {
                    borrowed_loads[producer] = true;
                }
            }
        }
        #[cfg(test)]
        TEST_PLANS.with(|count| count.set(count.get() + 1));
        Some(Self {
            values,
            fields,
            sites,
            region,
            borrowed_loads,
            owned_stores,
            observes,
            native_writes,
        })
    }

    pub(super) fn field_count(&self) -> usize {
        self.fields.len()
    }

    pub(super) fn contains(&self, pc: usize) -> bool {
        self.region.get(pc).copied().unwrap_or(false)
    }

    pub(super) fn native_write(&self, pc: usize) -> bool {
        self.native_writes.get(pc).copied().unwrap_or(false)
    }
}

impl Lowering<'_, '_> {
    /// Called only at entry and after an effect boundary, with no borrowed
    /// operands. The owning private locals protect every derived address.
    pub(super) fn validate_loop_fields(&mut self, resume_pc: usize) {
        if self.loop_facts.is_none() {
            return;
        }
        let validation = if let Some(block) = self.loop_validation {
            block
        } else {
            let block = self.b.create_block();
            self.b.append_block_param(block, types::I32);
            self.b.append_block_param(block, types::I32);
            self.loop_validation = Some(block);
            block
        };
        let next = self.b.create_block();
        let index = self.loop_validation_returns.len();
        self.loop_validation_returns.push(next);
        // A shared failure block receives heterogeneous stack depths. Publish
        // each exact state at its own boundary, before joining the validator.
        self.publish_locals();
        self.publish_stack(self.stack_state.depth);
        let index = self.b.ins().iconst(types::I32, index as i64);
        let resume = self.b.ins().iconst(types::I32, resume_pc as i64);
        self.b
            .ins()
            .jump(validation, &[index.into(), resume.into()]);
        self.b.switch_to_block(next);
    }

    /// One native validator per region, shared by all effect/poll boundaries.
    /// The continuation token routes normal completion back to its exact site;
    /// the independent bytecode PC supplies the canonical VM recovery point.
    pub(super) fn emit_loop_validation(&mut self) {
        let Some(validation) = self.loop_validation else {
            return;
        };
        let plan = self.loop_facts.expect("validation requires field proof");
        self.b.switch_to_block(validation);
        let token = self.b.block_params(validation)[0];
        let resume_pc = self.b.block_params(validation)[1];
        let miss = self.b.create_block();
        for (index, field) in plan.fields.iter().copied().enumerate() {
            let word = self.local(field.receiver);
            let tag = self.b.ins().ushr_imm_u(word, 48);
            let object = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, (PACK_OBJ >> 48) as i64);
            self.property_guard(object, miss);
            let raw = self.b.ins().band_imm_s(word, 0x0000_ffff_ffff_ffff);
            let object = self.b.ins().iadd_imm_s(raw, self.values.obj_from_rc as i64);
            if field.array_length {
                self.property_plain(object, miss);
                let exotic = self.property_load(types::I8, object, self.values.obj_exotic);
                let array = self.b.ins().icmp_imm_s(
                    IntCC::Equal,
                    exotic,
                    self.values.exotic_array_tag as i64,
                );
                self.property_guard(array, miss);
                let slot = self.property_load(
                    types::I32,
                    object,
                    self.values.obj_props + self.values.props_len_slot,
                );
                let key = self
                    .b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, slot, field.slot as i64);
                self.property_guard(key, miss);
            } else {
                self.property_ordinary(object, miss);
                let shape = self.b.ins().iconst(types::I32, field.shape as i64);
                self.property_shape(object, shape, miss);
            }
            let len = self.property_load(
                types::I64,
                object,
                self.values.obj_props + self.values.props_entries + self.values.vec_len_off,
            );
            let inside =
                self.b
                    .ins()
                    .icmp_imm_s(IntCC::UnsignedGreaterThan, len, field.slot as i64);
            self.property_guard(inside, miss);
            let data = self.property_load(
                types::I64,
                object,
                self.values.obj_props + self.values.props_entries + self.values.vec_ptr_off,
            );
            let address = self
                .b
                .ins()
                .iadd_imm_s(data, field.slot as i64 * self.values.entry_size as i64);
            let flags = self.property_load(types::I8, address, self.values.entry_accessor);
            let mask = PROP_ACCESSOR | if field.writable { PROP_WRITABLE } else { 0 };
            let flags = self.b.ins().band_imm_s(flags, mask as i64);
            let data = self.b.ins().icmp_imm_s(
                IntCC::Equal,
                flags,
                if field.writable {
                    PROP_WRITABLE as i64
                } else {
                    0
                },
            );
            self.property_guard(data, miss);
            let value = self.property_load(types::I64, address, self.values.entry_value);
            let number = self.number(value);
            self.property_guard(number, miss);
            self.b.def_var(self.loop_field_addresses[index], address);
            if self.loop_scalars {
                // Different private receiver slots may alias. Distinct cached
                // fields may share an address only when neither is written.
                for previous in 0..index {
                    if field.writable || plan.fields[previous].writable {
                        let other = self.b.use_var(self.loop_field_addresses[previous]);
                        let separate = self.b.ins().icmp(IntCC::NotEqual, address, other);
                        self.property_guard(separate, miss);
                    }
                }
                self.b.def_var(self.loop_field_values[index], value);
            }
        }
        let mut dispatch = Switch::new();
        for (index, &target) in self.loop_validation_returns.iter().enumerate() {
            dispatch.set_entry(index as u128, target);
        }
        dispatch.emit(&mut self.b, token, self.error);
        self.b.switch_to_block(miss);
        self.b.set_cold_block(miss);
        self.count_runtime(diagnostics::SPECIALIZATION_BAILOUT, 1);
        self.count_loop_retirement_request();
        self.count_terminal_recovery_start();
        let success = self.canonical_helper_value(loop_entry::miss, resume_pc);
        self.count_terminal_recovery_result(success);
        self.b.ins().brif(success, self.ok, &[], self.error, &[]);
    }

    /// Publish Number fields before effects, safepoints and region exits. Roots
    /// remain unchanged; refresh before using these addresses after an effect.
    pub(super) fn publish_loop_fields(&mut self) {
        if !self.loop_scalars {
            return;
        }
        let Some(plan) = self.loop_facts.filter(|plan| plan.contains(self.pc)) else {
            return;
        };
        for (index, field) in plan.fields.iter().enumerate() {
            if field.writable {
                let address = self.b.use_var(self.loop_field_addresses[index]);
                let value = self.b.use_var(self.loop_field_values[index]);
                self.store(value, address, self.values.entry_value as i32);
            }
        }
    }

    pub(super) fn loop_exit(&mut self, target: Block) -> Block {
        if !self.loop_scalars {
            return target;
        }
        let Some(plan) = self.loop_facts.filter(|plan| plan.contains(self.pc)) else {
            return target;
        };
        let Some(pc) = self.pcs.iter().position(|&block| block == Some(target)) else {
            return target;
        };
        if plan.contains(pc) {
            return target;
        }
        let from = self.b.current_block().expect("live native block");
        let exit = self.b.create_block();
        self.b.switch_to_block(exit);
        self.publish_loop_fields();
        self.b.ins().jump(target, &[]);
        self.b.switch_to_block(from);
        exit
    }

    pub(super) fn loop_field_operation(&mut self, pc: usize, op: &Op, next: Block) -> bool {
        let Some(plan) = self.loop_facts else {
            return false;
        };
        if plan.borrowed_loads.get(pc).copied().unwrap_or(false) {
            let Op::LoadLocal(slot) = *op else {
                unreachable!()
            };
            let receiver = self.local(slot);
            self.push(receiver);
            self.jump_normal(next);
            return true;
        }
        let Some(field) = plan.sites.get(pc).copied().flatten() else {
            return false;
        };
        let address = self.b.use_var(self.loop_field_addresses[field]);
        if matches!(op, Op::GetPropLocal(..)) {
            let number = if self.loop_scalars {
                self.b.use_var(self.loop_field_values[field])
            } else {
                self.property_load(types::I64, address, self.values.entry_value)
            };
            self.push(number);
        } else {
            let number = self.stack_read(1);
            if plan.owned_stores[pc] {
                // The original private receiver is never assigned in this
                // region, so its canonical owner survives this operand release.
                let receiver = self.stack_read(2);
                let receiver = self.b.ins().band_imm_s(receiver, 0x0000_ffff_ffff_ffff);
                let count = self.load(receiver, self.values.rc_strong_off as i32);
                let count = self.b.ins().iadd_imm_s(count, -1);
                self.store(count, receiver, self.values.rc_strong_off as i32);
            }
            if self.loop_scalars {
                self.b.def_var(self.loop_field_values[field], number);
            } else {
                self.store(number, address, self.values.entry_value as i32);
            }
            match op {
                Op::SetProp(..) => {
                    // The lower word is the non-owning receiver from the proven
                    // LoadLocal pair; the surviving result is an immediate Number.
                    self.stack_write(2, number);
                    self.change_top(-8);
                }
                Op::SetPropDrop(..) => self.change_top(-16),
                Op::SetPropLocalDrop(..) => self.change_top(-8),
                _ => unreachable!(),
            }
        }
        self.jump_normal(next);
        true
    }
}
