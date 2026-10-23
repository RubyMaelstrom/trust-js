//! Target-neutral compiler metadata for logical frame slots and side exits.
//!
//! These records describe the compiler's baseline-frame contract. They are not
//! published collector maps or a second execution frame, and retain no GC values.
//! The current producer emits ordered CopySlot recipes only.

/// A bytecode identity, independent of native instruction addresses.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BytecodePc(u32);

impl BytecodePc {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SafepointId(u32);

impl SafepointId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DeoptId(u32);

impl DeoptId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameMapError {
    SlotBitmapTooLong,
    OverlappingRootClasses,
    SlotOutOfRange,
    OperandDepthOutOfRange,
}

/// Immutable compiler metadata at one exact bytecode point. The bitmaps address
/// logical slots; no value or machine address is embedded in this record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RootMap {
    pub(crate) safepoint: SafepointId,
    pub(crate) bytecode_pc: BytecodePc,
    pub(crate) slot_count: u16,
    pub(crate) operand_depth: u16,
    pub(crate) tagged_registers: u64,
    pub(crate) tagged_slots: Vec<u64>,
    pub(crate) handle_slots: Vec<u64>,
    pub(crate) environment_slots: Vec<u64>,
}

impl RootMap {
    fn bitmap_words(slot_count: u16) -> usize {
        usize::from(slot_count).div_ceil(64)
    }

    fn check_bitmap(bits: &[u64], slot_count: u16) -> Result<(), FrameMapError> {
        let expected = Self::bitmap_words(slot_count);
        if bits.len() != expected {
            return Err(FrameMapError::SlotBitmapTooLong);
        }
        let excess = expected
            .checked_mul(64)
            .and_then(|count| count.checked_sub(usize::from(slot_count)))
            .unwrap_or(0);
        if excess != 0 && bits.last().is_some_and(|word| word >> (64 - excess) != 0) {
            return Err(FrameMapError::SlotOutOfRange);
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), FrameMapError> {
        if self.operand_depth > self.slot_count {
            return Err(FrameMapError::OperandDepthOutOfRange);
        }
        Self::check_bitmap(&self.tagged_slots, self.slot_count)?;
        Self::check_bitmap(&self.handle_slots, self.slot_count)?;
        Self::check_bitmap(&self.environment_slots, self.slot_count)?;
        for ((tagged, handles), environments) in self
            .tagged_slots
            .iter()
            .zip(&self.handle_slots)
            .zip(&self.environment_slots)
        {
            if tagged & handles != 0 || tagged & environments != 0 || handles & environments != 0 {
                return Err(FrameMapError::OverlappingRootClasses);
            }
        }
        Ok(())
    }

    /// Inspect logical indices only. The compiler's current producer does not
    /// publish this map to a collector; tests exercise its bitmap contract.
    #[cfg(test)]
    fn try_for_each_root_slot<E>(
        &self,
        mut visit: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        for word in 0..self.tagged_slots.len() {
            let mut bits =
                self.tagged_slots[word] | self.handle_slots[word] | self.environment_slots[word];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                visit(word * 64 + bit)?;
                bits &= bits - 1;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MaterializationRecipe {
    CopySlot(u16),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeoptRecordError {
    InvalidRootMap(FrameMapError),
    ValueCountOverflow,
    RecipeSlotOutOfRange,
}

/// Ordered logical-slot copies. There are no embedded constants, opaque handles,
/// virtual objects or runtime reconstruction machinery in compiler metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeoptRecord {
    pub(crate) id: DeoptId,
    pub(crate) bytecode_pc: BytecodePc,
    pub(crate) root_map: RootMap,
    pub(crate) recipes: Vec<MaterializationRecipe>,
}

impl DeoptRecord {
    pub(crate) fn validate(&self) -> Result<(), DeoptRecordError> {
        self.root_map
            .validate()
            .map_err(DeoptRecordError::InvalidRootMap)?;
        if self.recipes.len() > usize::from(u16::MAX) {
            return Err(DeoptRecordError::ValueCountOverflow);
        }
        let slots = self.root_map.slot_count;
        for MaterializationRecipe::CopySlot(slot) in &self.recipes {
            if *slot >= slots {
                return Err(DeoptRecordError::RecipeSlotOutOfRange);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root_map(slots: u16) -> RootMap {
        let words = usize::from(slots).div_ceil(64);
        RootMap {
            safepoint: SafepointId::new(13),
            bytecode_pc: BytecodePc::new(17),
            slot_count: slots,
            operand_depth: 0,
            tagged_registers: 0,
            tagged_slots: vec![0; words],
            handle_slots: vec![0; words],
            environment_slots: vec![0; words],
        }
    }

    #[test]
    fn root_maps_validate_empty_and_64_65_slot_boundaries() {
        root_map(0).validate().unwrap();
        for count in [64, 65] {
            let mut map = root_map(count);
            map.tagged_slots[0] = 1 << 63;
            if count == 65 {
                map.environment_slots[1] = 1;
                map.operand_depth = 1;
            }
            map.validate().unwrap();
            let mut indices = Vec::new();
            map.try_for_each_root_slot::<std::convert::Infallible>(|index| {
                indices.push(index);
                Ok(())
            })
            .unwrap();
            assert_eq!(indices, if count == 65 { vec![63, 64] } else { vec![63] });

            let mut too_long = map.clone();
            too_long.handle_slots.push(0);
            assert_eq!(too_long.validate(), Err(FrameMapError::SlotBitmapTooLong));
            let mut too_short = map.clone();
            too_short.environment_slots.pop();
            assert_eq!(too_short.validate(), Err(FrameMapError::SlotBitmapTooLong));
        }
        let mut excess = root_map(65);
        excess.tagged_slots[1] = 2;
        assert_eq!(excess.validate(), Err(FrameMapError::SlotOutOfRange));
    }

    #[test]
    fn logical_root_indices_are_ordered_and_classes_cannot_overlap() {
        let mut map = root_map(65);
        map.tagged_slots = vec![0b1001, 0];
        map.handle_slots = vec![0b0010, 0];
        map.environment_slots = vec![0b0100, 1];
        map.validate().unwrap();
        let mut indices = Vec::new();
        map.try_for_each_root_slot::<std::convert::Infallible>(|index| {
            indices.push(index);
            Ok(())
        })
        .unwrap();
        assert_eq!(indices, vec![0, 1, 2, 3, 64]);
        map.handle_slots[0] |= 1;
        assert_eq!(map.validate(), Err(FrameMapError::OverlappingRootClasses));
        map.handle_slots[0] &= !1;
        map.environment_slots[0] |= 0b0010;
        assert_eq!(map.validate(), Err(FrameMapError::OverlappingRootClasses));
        map.environment_slots[0] &= !0b0010;
        map.environment_slots[0] |= 0b1000;
        assert_eq!(map.validate(), Err(FrameMapError::OverlappingRootClasses));
        map.environment_slots[0] &= !0b1000;
        map.operand_depth = 66;
        assert_eq!(map.validate(), Err(FrameMapError::OperandDepthOutOfRange));
    }

    #[test]
    fn copy_recipes_preserve_source_order_without_storing_values_in_metadata() {
        use crate::value::Value;
        let record = DeoptRecord {
            id: DeoptId::new(7),
            bytecode_pc: BytecodePc::new(17),
            root_map: root_map(3),
            recipes: vec![
                MaterializationRecipe::CopySlot(2),
                MaterializationRecipe::CopySlot(0),
                MaterializationRecipe::CopySlot(1),
                MaterializationRecipe::CopySlot(1),
            ],
        };
        record.validate().unwrap();
        let frame = [Value::Num(-0.0), Value::Bool(true), Value::Num(f64::NAN)];
        let copied: Vec<Value> = record
            .recipes
            .iter()
            .map(|MaterializationRecipe::CopySlot(slot)| frame[usize::from(*slot)])
            .collect();
        assert_eq!(
            copied.iter().map(|value| value.bits()).collect::<Vec<_>>(),
            [frame[2], frame[0], frame[1], frame[1]]
                .iter()
                .map(|value| value.bits())
                .collect::<Vec<_>>()
        );
        let mut malformed = record.clone();
        malformed.recipes[0] = MaterializationRecipe::CopySlot(3);
        assert_eq!(
            malformed.validate(),
            Err(DeoptRecordError::RecipeSlotOutOfRange)
        );
        malformed = record.clone();
        malformed.root_map.tagged_slots.clear();
        assert_eq!(
            malformed.validate(),
            Err(DeoptRecordError::InvalidRootMap(
                FrameMapError::SlotBitmapTooLong
            ))
        );
        malformed = record;
        malformed.recipes = vec![MaterializationRecipe::CopySlot(0); usize::from(u16::MAX) + 1];
        assert_eq!(
            malformed.validate(),
            Err(DeoptRecordError::ValueCountOverflow)
        );
    }
}
