//! Native own-element reads and overwrites, for every packed JavaScript value category.
//!
//! ECMA-262 e28783d5: OrdinaryGet, OrdinarySetWithOwnDescriptor and Array exotic
//! [[DefineOwnProperty]] (#sec-ordinaryget, #sec-ordinarysetwithowndescriptor,
//! #sec-array-exotic-objects-defineownproperty-p-desc). An existing own data element wins
//! before prototype lookup; a hole DOES NOT. Writes require its live writable attribute.
//! Number and canonical array-index String guards perform no coercion. Every miss,
//! including creation, accessors, proxies, typed arrays and other keys, executes the
//! original checked operation. "-0" and noncanonical numeric strings are not array indices.
//!
//! The canonical Property remains authoritative. Numeric read mirrors are maintained on
//! numeric writes, not discarded indiscriminately. No observer separates owner transfers,
//! the property/mirror write and the corresponding operand-stack move.

use super::*;
use crate::value::{
    JitLayout, MIRROR_ALL_I32, MIRROR_HOLE, MIRROR_OK, PACK_OBJ, PACK_STR, PROP_ACCESSOR,
    PROP_WRITABLE,
};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_HEAP_HELPERS: std::cell::Cell<[usize; 3]> =
        const { std::cell::Cell::new([0; 3]) };
}

pub(super) fn supported(layout: &JitLayout) -> bool {
    property::supported(layout)
        && layout.packed_elems_valid
        && layout.property_size == layout.entry_size
        && layout.property_value == layout.entry_value
        && layout.property_meta == layout.entry_accessor
        && [
            layout.obj_props + layout.props_elems,
            layout.dense_packed,
            layout.dense_inline_len,
            layout.dense_inline_data,
            layout.dense_elems + layout.vec_ptr_off,
            layout.dense_elems + layout.vec_len_off,
            layout.dense_mirror + layout.vec_ptr_off,
            layout.dense_mirror + layout.vec_len_off,
            layout.obj_props + layout.props_mirror_flags,
        ]
        .into_iter()
        .all(|offset| offset <= i32::MAX as usize)
}

impl Lowering<'_, '_> {
    /// Return the Object base. This checks the side-table exotic flag
    /// before using any ordinary storage; checking only Exotic::None would accept proxies.
    pub(super) fn element_receiver(&mut self, word: Value, slow: Block) -> Value {
        let tag = self.b.ins().ushr_imm_u(word, 48);
        let object = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, tag, (PACK_OBJ >> 48) as i64);
        self.property_guard(object, slow);
        let owner = self.b.ins().band_imm_s(word, 0x0000_ffff_ffff_ffff);
        let object = self
            .b
            .ins()
            .iadd_imm_s(owner, self.values.obj_from_rc as i64);
        self.property_plain(object, slow);
        let exotic = self.property_load(types::I8, object, self.values.obj_exotic);
        let ordinary =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, exotic, self.values.exotic_none_tag as i64);
        let array =
            self.b
                .ins()
                .icmp_imm_s(IntCC::Equal, exotic, self.values.exotic_array_tag as i64);
        let supported = self.b.ins().bor(ordinary, array);
        self.property_guard(supported, slow);
        object
    }

    fn element_index(&mut self, key: Value, known_number: bool, slow: Block) -> Value {
        let string = (!known_number).then(|| self.b.create_block());
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I64);
        if let Some(string) = string {
            let numeric = self.number(key);
            self.property_guard(numeric, string);
        }
        let number = self.b.ins().bitcast(types::F64, MemFlags::new(), key);
        // Saturating conversion is non-trapping on both backends. Round-trip equality
        // excludes NaN, fractions, negative and out-of-range numbers, and accepts -0 as 0.
        let index = self.b.ins().fcvt_to_uint_sat(types::I32, number);
        let roundtrip = self.b.ins().fcvt_from_uint(types::F64, index);
        let exact = self.b.ins().fcmp(FloatCC::Equal, roundtrip, number);
        self.property_guard(exact, slow);
        let index = self.b.ins().uextend(types::I64, index);
        self.b.ins().jump(done, &[index.into()]);
        if let Some(string) = string {
            self.b.switch_to_block(string);
            if !self.owners_valid {
                self.b.ins().jump(slow, &[]);
            } else {
                let tag = self.b.ins().ushr_imm_u(key, 48);
                let is_string = self
                    .b
                    .ins()
                    .icmp_imm_s(IntCC::Equal, tag, (PACK_STR >> 48) as i64);
                self.property_guard(is_string, slow);
                // The packed-owner probe validates this exact LStr header identity. Its
                // repr(C) length/data offsets come from the string implementation, not Rc<str>.
                let header = self.b.ins().band_imm_s(key, 0x0000_ffff_ffff_ffff);
                let length = self.property_load(types::I32, header, crate::lstr::LEN_OFF);
                let range = self.b.ins().iadd_imm_s(length, -1);
                let bounded = self.b.ins().icmp_imm_s(IntCC::UnsignedLessThan, range, 10);
                self.property_guard(bounded, slow);
                let length = self.b.ins().uextend(types::I64, length);
                let byte = self.property_load(types::I8, header, crate::lstr::DATA_OFF);
                let byte = self.b.ins().uextend(types::I64, byte);
                let first = self.b.ins().iadd_imm_s(byte, -i64::from(b'0'));
                let digit = self.b.ins().icmp_imm_s(IntCC::UnsignedLessThan, first, 10);
                self.property_guard(digit, slow);
                let single = self.b.ins().icmp_imm_s(IntCC::Equal, length, 1);
                let nonzero = self.b.ins().icmp_imm_s(IntCC::NotEqual, first, 0);
                let canonical = self.b.ins().bor(single, nonzero);
                self.property_guard(canonical, slow);
                let scan = self.b.create_block();
                self.b.append_block_param(scan, types::I64); // byte cursor
                self.b.append_block_param(scan, types::I64); // accumulated index
                let one = self.b.ins().iconst(types::I64, 1);
                self.b.ins().brif(
                    single,
                    done,
                    &[first.into()],
                    scan,
                    &[one.into(), first.into()],
                );
                self.b.switch_to_block(scan);
                let cursor = self.b.block_params(scan)[0];
                let index = self.b.block_params(scan)[1];
                let address = self.b.ins().iadd(header, cursor);
                let byte = self.property_load(types::I8, address, crate::lstr::DATA_OFF);
                let byte = self.b.ins().uextend(types::I64, byte);
                let digit = self.b.ins().iadd_imm_s(byte, -i64::from(b'0'));
                let decimal = self.b.ins().icmp_imm_s(IntCC::UnsignedLessThan, digit, 10);
                self.property_guard(decimal, slow);
                let index = self.b.ins().imul_imm_s(index, 10);
                let index = self.b.ins().iadd(index, digit); // at most 9_999_999_999, no u64 wrap
                let cursor = self.b.ins().iadd_imm_s(cursor, 1);
                let more = self.b.ins().icmp(IntCC::UnsignedLessThan, cursor, length);
                self.b.ins().brif(
                    more,
                    scan,
                    &[cursor.into(), index.into()],
                    done,
                    &[index.into()],
                );
            }
        }
        self.b.switch_to_block(done);
        let index = self.b.block_params(done)[0];
        let valid = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedLessThan, index, u32::MAX as i64);
        self.property_guard(valid, slow);
        index
    }

    /// Locate a canonical, non-hole own Property in all three existing storage forms.
    /// No preparation, allocation, owner transfer, mirror mutation or prototype observation.
    pub(super) fn element_property(
        &mut self,
        object: Value,
        index: Value,
        slow: Block,
    ) -> (Value, Value) {
        let dense = self.load(
            object,
            (self.values.obj_props + self.values.props_elems) as i32,
        );
        self.property_guard(dense, slow);
        let packed = self.load(dense, self.values.dense_packed as i32);
        let vector = self.b.create_block();
        let inline_or_classic = self.b.create_block();
        let hit = self.b.create_block();
        self.b.append_block_param(hit, types::I64);
        self.b
            .ins()
            .brif(packed, vector, &[], inline_or_classic, &[]);
        self.b.switch_to_block(vector);
        let length = self.load(packed, self.values.vec_len_off as i32);
        let in_bounds = self.b.ins().icmp(IntCC::UnsignedLessThan, index, length);
        self.property_guard(in_bounds, slow);
        let data = self.load(packed, self.values.vec_ptr_off as i32);
        let offset = self
            .b
            .ins()
            .imul_imm_s(index, self.values.property_size as i64);
        let property = self.b.ins().iadd(data, offset);
        self.b.ins().jump(hit, &[property.into()]);

        self.b.switch_to_block(inline_or_classic);
        let length = self.property_load(types::I8, dense, self.values.dense_inline_len);
        let inline = self.b.create_block();
        let classic = self.b.create_block();
        self.b.ins().brif(length, inline, &[], classic, &[]);
        self.b.switch_to_block(inline);
        let length = self.b.ins().uextend(types::I64, length);
        let in_bounds = self.b.ins().icmp(IntCC::UnsignedLessThan, index, length);
        self.property_guard(in_bounds, slow);
        let data = self
            .b
            .ins()
            .iadd_imm_s(dense, self.values.dense_inline_data as i64);
        let offset = self
            .b
            .ins()
            .imul_imm_s(index, self.values.property_size as i64);
        let property = self.b.ins().iadd(data, offset);
        self.b.ins().jump(hit, &[property.into()]);

        self.b.switch_to_block(classic);
        let length = self.load(
            dense,
            (self.values.dense_elems + self.values.vec_len_off) as i32,
        );
        let in_bounds = self.b.ins().icmp(IntCC::UnsignedLessThan, index, length);
        self.property_guard(in_bounds, slow);
        let data = self.load(
            dense,
            (self.values.dense_elems + self.values.vec_ptr_off) as i32,
        );
        let offset = self.b.ins().ishl_imm_u(index, 2);
        let address = self.b.ins().iadd(data, offset);
        let slot = self.property_load(types::I32, address, 0);
        let present = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, slot, u32::MAX as i64);
        self.property_guard(present, slow);
        let slot = self.b.ins().uextend(types::I64, slot);
        let length = self.load(
            object,
            (self.values.obj_props + self.values.props_entries + self.values.vec_len_off) as i32,
        );
        let in_bounds = self.b.ins().icmp(IntCC::UnsignedLessThan, slot, length);
        self.property_guard(in_bounds, slow);
        let data = self.load(
            object,
            (self.values.obj_props + self.values.props_entries + self.values.vec_ptr_off) as i32,
        );
        let offset = self.b.ins().imul_imm_s(slot, self.values.entry_size as i64);
        let property = self.b.ins().iadd(data, offset);
        self.b.ins().jump(hit, &[property.into()]);

        self.b.switch_to_block(hit);
        let property = self.b.block_params(hit)[0];
        let value = self.load(property, self.values.property_value as i32);
        let present = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, value, PACK_EMPTY as i64);
        self.property_guard(present, slow);
        (property, value)
    }

    pub(super) fn property_data(&mut self, property: Value, writable: bool, slow: Block) {
        let flags = self.property_load(types::I8, property, self.values.property_meta);
        let mask = PROP_ACCESSOR | if writable { PROP_WRITABLE } else { 0 };
        let bits = self.b.ins().band_imm_s(flags, mask as i64);
        let expected = if writable { PROP_WRITABLE } else { 0 };
        let valid = self.b.ins().icmp_imm_s(IntCC::Equal, bits, expected as i64);
        self.property_guard(valid, slow);
    }

    /// Committed overwrite: all paths finish here, none fall back/reenter/collect. Preserve a
    /// coherent numeric mirror and its exact-i32 proof; invalidate only an incompatible value
    /// or unavailable mirror cell. An indexed write also clears a failed-preparation hint.
    pub(super) fn element_mirror_store(&mut self, object: Value, index: Value, word: Value) {
        let flag_offset = self.values.obj_props + self.values.props_mirror_flags;
        let flags = self.property_load(types::I8, object, flag_offset);
        let dense = self.load(
            object,
            (self.values.obj_props + self.values.props_elems) as i32,
        );
        let active = self.b.create_block();
        let inactive = self.b.create_block();
        let invalidate = self.b.create_block();
        let done = self.b.create_block();
        let coherent = self.b.ins().band_imm_s(flags, MIRROR_OK as i64);
        self.b.ins().brif(coherent, active, &[], inactive, &[]);
        self.b.switch_to_block(inactive);
        self.b.ins().brif(flags, invalidate, &[], done, &[]);
        self.b.switch_to_block(active);
        let numeric = self.number(word);
        let not_hole = self
            .b
            .ins()
            .icmp_imm_s(IntCC::NotEqual, word, MIRROR_HOLE as i64);
        let valid = self.b.ins().band(numeric, not_hole);
        let numeric_block = self.b.create_block();
        self.b
            .ins()
            .brif(valid, numeric_block, &[], invalidate, &[]);
        self.b.switch_to_block(numeric_block);
        let length = self.load(
            dense,
            (self.values.dense_mirror + self.values.vec_len_off) as i32,
        );
        let in_bounds = self.b.ins().icmp(IntCC::UnsignedLessThan, index, length);
        let store = self.b.create_block();
        self.b.ins().brif(in_bounds, store, &[], invalidate, &[]);
        self.b.switch_to_block(store);
        let number = self.b.ins().bitcast(types::F64, MemFlags::new(), word);
        let integer = self.b.ins().fcvt_to_sint_sat(types::I32, number);
        let roundtrip = self.b.ins().fcvt_from_sint(types::F64, integer);
        let bits = self.b.ins().bitcast(types::I64, MemFlags::new(), roundtrip);
        let exact = self.b.ins().icmp(IntCC::Equal, bits, word); // notably excludes -0
        let cleared = self.b.ins().band_imm_s(flags, !(MIRROR_ALL_I32 as i64));
        let flags = self.b.ins().select(exact, flags, cleared);
        self.b
            .ins()
            .store(MemFlags::trusted(), flags, object, flag_offset as i32);
        let data = self.load(
            dense,
            (self.values.dense_mirror + self.values.vec_ptr_off) as i32,
        );
        let offset = self.b.ins().ishl_imm_u(index, 3);
        let address = self.b.ins().iadd(data, offset);
        self.store(word, address, 0);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(invalidate);
        let zero = self.b.ins().iconst(types::I8, 0);
        self.b
            .ins()
            .store(MemFlags::trusted(), zero, object, flag_offset as i32);
        let zero = self.b.ins().iconst(types::I64, 0);
        self.store(
            zero,
            dense,
            (self.values.dense_mirror + self.values.vec_len_off) as i32,
        );
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    pub(super) fn element_read(&mut self, pc: usize, op: &Op, next: Block) {
        if !supported(self.values) {
            self.checked(get_element, pc as u32, next);
            return;
        }
        let slow = self.b.create_block();
        let receiver = if let Op::GetElemLocal(slot) = op {
            self.local(*slot)
        } else {
            self.stack_read(2)
        };
        let key = self.stack_read(1);
        let numeric_key = self.input_types().top.is_number();
        let index = self.element_index(key, numeric_key, slow);
        let object = self.element_receiver(receiver, slow);
        let (property, value) = self.element_property(object, index, slow);
        self.property_data(property, false, slow);
        let mut owners = vec![(value, true)];
        if !numeric_key {
            owners.push((key, false));
        }
        if matches!(op, Op::GetElem) {
            owners.push((receiver, false));
        }
        self.transfer_owners(&owners, slow);
        if matches!(op, Op::GetElem) {
            self.stack_write(2, value);
            self.change_top(-8);
        } else {
            self.stack_write(1, value); // GetMethodElem keeps the original owning receiver.
        }
        self.jump_normal(next);
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        self.stack_state = self.stack_plan.at[pc];
        self.checked(get_element, pc as u32, next);
    }

    pub(super) fn element_write(&mut self, pc: usize, op: &Op, next: Block) {
        if !supported(self.values) {
            self.checked(set_element, pc as u32, next);
            return;
        }
        let slow = self.b.create_block();
        let local = matches!(op, Op::SetElemLocal(_) | Op::SetElemLocalDrop(_));
        let keep = matches!(op, Op::SetElem | Op::SetElemLocal(_));
        let receiver = if let Op::SetElemLocal(slot) | Op::SetElemLocalDrop(slot) = op {
            self.local(*slot)
        } else {
            self.stack_read(3)
        };
        let key = self.stack_read(2);
        let incoming = self.stack_read(1);
        let numeric_key = self.input_types().second.is_number();
        let index = self.element_index(key, numeric_key, slow);
        let object = self.element_receiver(receiver, slow);
        let (property, old) = self.element_property(object, index, slow);
        self.property_data(property, true, slow);
        let mut owners = vec![(old, false)];
        if !numeric_key {
            owners.push((key, false));
        }
        if !local {
            owners.push((receiver, false));
        }
        if keep && !self.input_types().top.is_copyable() {
            owners.push((incoming, true));
        }
        self.transfer_owners(&owners, slow);
        self.store(incoming, property, self.values.property_value as i32);
        self.element_mirror_store(object, index, incoming);
        let consumed = if local { 2 } else { 3 };
        if keep {
            self.stack_write(consumed, incoming);
        }
        self.change_top(-8 * (consumed - usize::from(keep)) as i64);
        self.jump_normal(next);
        self.b.switch_to_block(slow);
        self.b.set_cold_block(slow);
        self.stack_state = self.stack_plan.at[pc];
        self.checked(set_element, pc as u32, next);
    }
}
