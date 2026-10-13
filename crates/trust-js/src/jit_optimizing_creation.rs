//! Native ordinary-property creation, using the runtime's live creation-cache proof.
//!
//! ECMA-262 snapshot e28783d5: OrdinarySetWithOwnDescriptor, CreateDataProperty,
//! ValidateAndApplyPropertyDescriptor and OrdinaryOwnPropertyKeys. A before-shape
//! proves absence, not permission to insert: extensibility, prototype identity and
//! the non-saturated prototype epoch must also agree with the checked chain walk.
//! Predicted keys are not properties until the live field prefix is extended.
//!
//! No allocation, observer or failing guard may follow the first owner-count write.
//! Private last-owned layouts, index/hash maintenance and prototype invalidation
//! remain checked. The key-only metadata allocation cannot alias a JavaScript owner.

use super::*;
use crate::bytecode::{IC_OFF_HOLDER_SHAPE, IC_OFF_MID2_SHAPE, IC_OFF_MID_SHAPE, IC_OFF_SLOT};
use crate::value::{JitLayout, PROP_CONFIGURABLE, PROP_ENUMERABLE, PROP_WRITABLE};

pub(super) fn supported(layout: &JitLayout, name: &str) -> bool {
    layout.valid
        && layout.key_probe_ok
        && name.len() <= i32::MAX as usize
        // The checked fill excludes digit-leading names. NUL-prefixed internal Symbol
        // keys additionally require the sidecar's SymbolData owner, not just an Rc<str>.
        && !name.as_bytes().first().is_some_and(|b| b.is_ascii_digit())
        && !name.starts_with('\0')
        && [
            layout.obj_extensible,
            layout.obj_props + layout.props_proto_flag,
            layout.obj_props + layout.props_elems,
            layout.obj_props + layout.props_layout,
            layout.obj_props + layout.props_entries + layout.vec_cap_off,
            layout.layout_data_off + layout.vec_len_off,
            layout.layout_data_off + layout.vec_ptr_off,
            layout.obj_heap,
            layout.heap_layouts,
            layout.shape_layout_entry_size,
            layout.shape_layout_entry_id,
            layout.shape_layout_entry_keys,
            layout.str_len_word,
            layout.str_ptr_word,
            layout.str_data_off,
        ]
        .into_iter()
        .all(|offset| offset <= i32::MAX as usize)
}

impl Lowering<'_, '_> {
    /// Compare a predicted next key without assuming that equal strings are interned.
    /// Short names use constant-sized loads; long names use a bounded loop, not IR
    /// proportional to source-name length. Every load stays inside the proven string.
    fn creation_key(&mut self, layout: Value, length: Value, name: &str, miss: Block) {
        let predicted = self.load(
            layout,
            (self.values.layout_data_off + self.values.vec_len_off) as i32,
        );
        let present = self
            .b
            .ins()
            .icmp(IntCC::UnsignedLessThan, length, predicted);
        self.property_guard(present, miss);
        let keys = self.load(
            layout,
            (self.values.layout_data_off + self.values.vec_ptr_off) as i32,
        );
        let offset = self
            .b
            .ins()
            .imul_imm_s(length, std::mem::size_of::<std::rc::Rc<str>>() as i64);
        let key = self.b.ins().iadd(keys, offset);
        let len = self.load(key, self.values.str_len_word as i32);
        let equal_length = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, len, name.len() as i64);
        self.property_guard(equal_length, miss);
        let owner = self.load(key, self.values.str_ptr_word as i32);
        let data = self
            .b
            .ins()
            .iadd_imm_s(owner, self.values.str_data_off as i64);
        let expected = self.b.ins().iconst(types::I64, name.as_ptr() as i64);
        let same = self.b.ins().icmp(IntCC::Equal, data, expected);
        let contents = self.b.create_block();
        let equal = self.b.create_block();
        self.b.ins().brif(same, equal, &[], contents, &[]);
        self.b.switch_to_block(contents);
        // Rc<str> bytes need not have the alignment of an integer load. Do not use
        // MemFlags::trusted(), which also promises alignment, for byte comparisons.
        let flags = MemFlags::new().with_notrap();
        let mut offset = 0;
        if name.len() > 32 {
            let full_bytes = name.len() & !7;
            let scan = self.b.create_block();
            self.b.append_block_param(scan, types::I64);
            let tail = self.b.create_block();
            let zero = self.b.ins().iconst(types::I64, 0);
            self.b.ins().jump(scan, &[zero.into()]);
            self.b.switch_to_block(scan);
            let index = self.b.block_params(scan)[0];
            let left_ptr = self.b.ins().iadd(data, index);
            let right_ptr = self.b.ins().iadd(expected, index);
            let left = self.b.ins().load(types::I64, flags, left_ptr, 0);
            let right = self.b.ins().load(types::I64, flags, right_ptr, 0);
            let matches = self.b.ins().icmp(IntCC::Equal, left, right);
            self.property_guard(matches, miss);
            let next = self.b.ins().iadd_imm_s(index, 8);
            let more = self
                .b
                .ins()
                .icmp_imm_s(IntCC::UnsignedLessThan, next, full_bytes as i64);
            self.b.ins().brif(more, scan, &[next.into()], tail, &[]);
            self.b.switch_to_block(tail);
            offset = full_bytes;
        }
        for (width, ty) in [
            (8, types::I64),
            (4, types::I32),
            (2, types::I16),
            (1, types::I8),
        ] {
            while name.len() - offset >= width {
                let bytes = &name.as_bytes()[offset..offset + width];
                let expected = match width {
                    8 => u64::from_ne_bytes(bytes.try_into().unwrap()),
                    4 => u64::from(u32::from_ne_bytes(bytes.try_into().unwrap())),
                    2 => u64::from(u16::from_ne_bytes(bytes.try_into().unwrap())),
                    1 => u64::from(bytes[0]),
                    _ => unreachable!(),
                };
                let word = self.b.ins().load(ty, flags, data, offset as i32);
                let expected = self.b.ins().iconst(ty, expected as i64);
                let matches = self.b.ins().icmp(IntCC::Equal, word, expected);
                self.property_guard(matches, miss);
                offset += width;
            }
        }
        self.b.ins().jump(equal, &[]);
        self.b.switch_to_block(equal);
    }

    /// Prepare key-layout ownership without effects. Returned words are replacement,
    /// previous owner, replacement's final count and previous owner's final count.
    /// A zero replacement means the receiver already owns a valid prediction.
    fn creation_layout(
        &mut self,
        object: Value,
        cache: Value,
        length: Value,
        name: &str,
        miss: Block,
    ) -> [Value; 4] {
        let old = self.load(
            object,
            (self.values.obj_props + self.values.props_layout) as i32,
        );
        let prediction = self.b.create_block();
        let transition = self.b.create_block();
        let ready = self.b.create_block();
        for _ in 0..4 {
            self.b.append_block_param(ready, types::I64);
        }
        let zero = self.b.ins().iconst(types::I64, 0);
        self.b.ins().brif(old, prediction, &[], transition, &[]);
        self.b.switch_to_block(prediction);
        self.creation_key(old, length, name, transition);
        self.b
            .ins()
            .jump(ready, &[zero.into(), zero.into(), zero.into(), zero.into()]);
        self.b.switch_to_block(transition);
        // The receiver's owning heap, not a compiler-time/thread-local heap, owns
        // these hints. Low ID bits choose a candidate; only the FULL ID proves it.
        let heap = self.load(object, self.values.obj_heap as i32);
        let shape = self.property_load(types::I32, cache, IC_OFF_HOLDER_SHAPE as usize);
        let wide = self.b.ins().uextend(types::I64, shape);
        let page_index = self.b.ins().ushr_imm_u(
            wide,
            crate::value::SHAPE_LAYOUT_PAGE_SIZE.trailing_zeros() as i64,
        );
        let page_index = self.b.ins().band_imm_s(
            page_index,
            (crate::value::SHAPE_LAYOUT_PAGE_COUNT - 1) as i64,
        );
        let page_offset = self
            .b
            .ins()
            .imul_imm_s(page_index, std::mem::size_of::<usize>() as i64);
        let page_ptr = self.b.ins().iadd(heap, page_offset);
        let page = self.load(page_ptr, self.values.heap_layouts as i32);
        self.property_guard(page, miss);
        let index = self
            .b
            .ins()
            .band_imm_s(wide, (crate::value::SHAPE_LAYOUT_PAGE_SIZE - 1) as i64);
        let offset = self
            .b
            .ins()
            .imul_imm_s(index, self.values.shape_layout_entry_size as i64);
        let entry = self.b.ins().iadd(page, offset);
        let actual = self.property_load(types::I32, entry, self.values.shape_layout_entry_id);
        let matches = self.b.ins().icmp(IntCC::Equal, shape, actual);
        self.property_guard(matches, miss);
        let new = self.load(entry, self.values.shape_layout_entry_keys as i32);
        self.property_guard(new, miss);
        let capacity = self.load(
            new,
            (self.values.layout_data_off + self.values.vec_len_off) as i32,
        );
        let complete = self.b.ins().icmp(IntCC::UnsignedLessThan, length, capacity);
        self.property_guard(complete, miss);
        let different = self.b.ins().icmp(IntCC::NotEqual, old, new);
        let replace = self.b.create_block();
        self.b.ins().brif(
            different,
            replace,
            &[],
            ready,
            &[zero.into(), zero.into(), zero.into(), zero.into()],
        );
        self.b.switch_to_block(replace);
        let count = self.load(new, self.values.rc_strong_off as i32);
        let retained = self.b.ins().iadd_imm_s(count, 1);
        let valid = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, retained, 1);
        self.property_guard(valid, miss);
        let release = self.b.create_block();
        self.b.ins().brif(
            old,
            release,
            &[],
            ready,
            &[new.into(), zero.into(), retained.into(), zero.into()],
        );
        self.b.switch_to_block(release);
        let count = self.load(old, self.values.rc_strong_off as i32);
        let shared = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, count, 1);
        self.property_guard(shared, miss);
        let released = self.b.ins().iadd_imm_s(count, -1);
        self.b.ins().jump(
            ready,
            &[new.into(), old.into(), retained.into(), released.into()],
        );
        self.b.switch_to_block(ready);
        self.b.block_params(ready).try_into().unwrap()
    }

    /// The caller already proved an ordinary receiver, a matching before-shape
    /// and IC_CREATE at this live polymorphic cache way. All failures keep the
    /// original operands/counts intact and may continue probing another way.
    pub(super) fn property_create(
        &mut self,
        op: &Op,
        object: Value,
        receiver: Value,
        incoming: Value,
        cache: Value,
        name: &str,
        miss: Block,
        next: Block,
    ) {
        let extensible = self.property_load(types::I8, object, self.values.obj_extensible);
        self.property_guard(extensible, miss);
        let prototype = self.property_load(
            types::I8,
            object,
            self.values.obj_props + self.values.props_proto_flag,
        );
        let not_prototype = self.b.ins().icmp_imm_s(IntCC::Equal, prototype, 0);
        self.property_guard(not_prototype, miss);
        let dense = self.load(
            object,
            (self.values.obj_props + self.values.props_elems) as i32,
        );
        let named_only = self.b.ins().icmp_imm_s(IntCC::Equal, dense, 0);
        self.property_guard(named_only, miss);
        let base = self.values.obj_props + self.values.props_entries;
        let length = self.load(object, (base + self.values.vec_len_off) as i32);
        // The ninth field would construct the hash sidecar; reserve/growth remains
        // checked too. This appends to the live prefix, never to predicted keys.
        let small = self.b.ins().icmp_imm_s(
            IntCC::UnsignedLessThan,
            length,
            crate::value::INDEX_THRESHOLD as i64,
        );
        self.property_guard(small, miss);
        let capacity = self.load(object, (base + self.values.vec_cap_off) as i32);
        let reserved = self.b.ins().icmp(IntCC::UnsignedLessThan, length, capacity);
        self.property_guard(reserved, miss);
        let epoch_address = self
            .b
            .ins()
            .iconst(types::I64, crate::value::proto_epoch_ptr() as i64);
        // This epoch is genuinely process-global AtomicU32. Keep its atomic nature
        // visible to Cranelift too; a plain load must not be CSE'd across native stores
        // by another Agent. Cranelift's seq-cst load is stronger than Rust's relaxed
        // reader, but both preserve the same invalidation semantics.
        let epoch = self
            .b
            .ins()
            .atomic_load(types::I32, MemFlags::trusted(), epoch_address);
        let cached_epoch = self.property_load(types::I32, cache, IC_OFF_MID_SHAPE as usize);
        let current = self.b.ins().icmp(IntCC::Equal, epoch, cached_epoch);
        self.property_guard(current, miss);
        let usable = self.b.ins().icmp_imm_s(IntCC::NotEqual, epoch, -1);
        self.property_guard(usable, miss);
        let low = self.property_load(types::I32, cache, IC_OFF_SLOT as usize);
        let low = self.b.ins().uextend(types::I64, low);
        let high = self.property_load(types::I32, cache, IC_OFF_MID2_SHAPE as usize);
        let high = self.b.ins().uextend(types::I64, high);
        let high = self.b.ins().ishl_imm_u(high, 32);
        let cached_proto = self.b.ins().bor(low, high);
        let proto = self.load(object, self.values.obj_proto as i32);
        let nonnull = self
            .b
            .ins()
            .iadd_imm_s(proto, self.values.gc_data_off as i64);
        let proto = self.b.ins().select(proto, nonnull, proto);
        let same_proto = self.b.ins().icmp(IntCC::Equal, cached_proto, proto);
        self.property_guard(same_proto, miss);
        let [replacement, previous, retained, released] =
            self.creation_layout(object, cache, length, name, miss);
        let consumed_receiver = matches!(op, Op::SetProp(..) | Op::SetPropDrop(..));
        let mut changes = Vec::with_capacity(2);
        if consumed_receiver {
            changes.push((receiver, false));
        }
        if matches!(op, Op::SetProp(..)) && !self.input_types().top.is_copyable() {
            changes.push((incoming, true));
        }
        if !changes.is_empty() {
            self.transfer_owners(&changes, miss);
        }
        // Commit: layout allocations contain keys only and are disjoint from all
        // packed JS owners above. All count/descriptor/identity guards are complete.
        let replace = self.b.create_block();
        let installed = self.b.create_block();
        self.b.ins().brif(replacement, replace, &[], installed, &[]);
        self.b.switch_to_block(replace);
        self.store(retained, replacement, self.values.rc_strong_off as i32);
        let release = self.b.create_block();
        let store_layout = self.b.create_block();
        self.b.ins().brif(previous, release, &[], store_layout, &[]);
        self.b.switch_to_block(release);
        self.store(released, previous, self.values.rc_strong_off as i32);
        self.b.ins().jump(store_layout, &[]);
        self.b.switch_to_block(store_layout);
        self.store(
            replacement,
            object,
            (self.values.obj_props + self.values.props_layout) as i32,
        );
        self.b.ins().jump(installed, &[]);
        self.b.switch_to_block(installed);
        let entries = self.load(object, (base + self.values.vec_ptr_off) as i32);
        let offset = self
            .b
            .ins()
            .imul_imm_s(length, self.values.entry_size as i64);
        let property = self.b.ins().iadd(entries, offset);
        self.store(incoming, property, self.values.entry_value as i32);
        let flags = self.b.ins().iconst(
            types::I64,
            (PROP_WRITABLE | PROP_ENUMERABLE | PROP_CONFIGURABLE) as i64,
        );
        self.store(flags, property, self.values.entry_accessor as i32);
        let shape = self.property_load(types::I32, cache, IC_OFF_HOLDER_SHAPE as usize);
        self.b.ins().store(
            MemFlags::trusted(),
            shape,
            object,
            (self.values.obj_props + self.values.props_shape) as i32,
        );
        let length = self.b.ins().iadd_imm_s(length, 1);
        self.store(length, object, (base + self.values.vec_len_off) as i32);
        self.finish_property_store(op, incoming, next);
    }
}
