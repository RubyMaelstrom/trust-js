//! Checked central-heap nucleus for the tagged-value migration.
//!
//! This handle-table prototype is deliberately not connected to `Value` or the live collector.
//! It provides the ownership and relocation invariants that later object-family migrations must
//! satisfy without exposing Rust allocation addresses to generated code.

#![allow(dead_code)]

use crate::tagged::{
    FrameMapError, HeapRef, NoGcError, NoGcScope, NoGcState, RootMap, RootSet, RootedTagged,
    TaggedFrame, TaggedValue,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeapGeneration {
    Young,
    Old,
    Large,
    Pinned,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct HeapStats {
    pub(crate) allocations: u64,
    pub(crate) promotions: u64,
    pub(crate) collections: u64,
    pub(crate) mark_objects_scanned: u64,
    pub(crate) mark_bytes_scanned: u64,
    pub(crate) sweep_objects_scanned: u64,
    pub(crate) sweep_bytes_scanned: u64,
    pub(crate) copied_bytes: u64,
    pub(crate) freed_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AllocationSiteStats {
    pub(crate) allocations: u64,
    pub(crate) promotions: u64,
    pub(crate) reclaimed: u64,
    pub(crate) live_allocations: u64,
    pub(crate) requested_bytes: u64,
    pub(crate) live_requested_bytes: u64,
    pub(crate) max_age: u8,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NurseryCollection {
    pub(crate) promoted: usize,
    pub(crate) reclaimed: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayoutId(u32);

impl LayoutId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ObjectHeader {
    pub(crate) size_units: u32,
    pub(crate) allocation_site: u32,
    pub(crate) age: u8,
    pub(crate) layout: LayoutId,
    pub(crate) generation: HeapGeneration,
    pub(crate) marked: bool,
    pub(crate) forwarding: Option<HeapRef>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HeapObject {
    header: ObjectHeader,
    storage: HeapStorage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum HeapStorage {
    Bytes(Box<[u8]>),
    Tagged(Box<[TaggedValue]>),
}

impl HeapStorage {
    fn requested_bytes(&self) -> usize {
        match self {
            Self::Bytes(payload) => payload.len(),
            Self::Tagged(fields) => fields
                .len()
                .saturating_mul(std::mem::size_of::<TaggedValue>()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeapError {
    InvalidReference,
    ForwardedReference,
    AlreadyForwarded,
    NotForwarded,
    PayloadKindMismatch,
    InvalidTaggedField,
    FieldOutOfBounds,
    HeaderMismatch,
    RememberedSetMismatch,
    InvalidFrameMap(FrameMapError),
    SafepointForbidden,
    RequestTooLarge,
    ReferenceSpaceExhausted,
}

/// A checked Agent-local handle table. Index zero is permanently reserved so a zero tagged
/// payload can never accidentally resolve to an object. Slots are reused only with generation
/// cookies; a slot whose cookie would wrap is retired permanently.
pub(crate) struct CentralHeap {
    objects: Vec<Option<HeapObject>>,
    slot_cookies: Vec<u16>,
    free_slots: Vec<usize>,
    requested_bytes: usize,
    no_gc: NoGcState,
    remembered: Vec<HeapRef>,
    stats: HeapStats,
    allocation_sites: crate::fasthash::FastMap<u32, AllocationSiteStats>,
}

impl Default for CentralHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl CentralHeap {
    pub(crate) fn new() -> Self {
        Self {
            objects: vec![None],
            slot_cookies: vec![0],
            free_slots: Vec::new(),
            requested_bytes: 0,
            no_gc: NoGcState::default(),
            remembered: Vec::new(),
            stats: HeapStats::default(),
            allocation_sites: Default::default(),
        }
    }

    fn record_site_allocation(&mut self, site: u32, bytes: usize, logical: bool) {
        let stats = self.allocation_sites.entry(site).or_default();
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        if logical {
            stats.allocations = stats.allocations.saturating_add(1);
            stats.requested_bytes = stats.requested_bytes.saturating_add(bytes);
        }
        stats.live_allocations = stats.live_allocations.saturating_add(1);
        stats.live_requested_bytes = stats.live_requested_bytes.saturating_add(bytes);
    }

    fn record_site_release(&mut self, site: u32, bytes: usize, age: u8) {
        let stats = self.allocation_sites.entry(site).or_default();
        stats.reclaimed = stats.reclaimed.saturating_add(1);
        stats.live_allocations = stats.live_allocations.saturating_sub(1);
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        stats.live_requested_bytes = stats.live_requested_bytes.saturating_sub(bytes);
        stats.max_age = stats.max_age.max(age);
    }

    pub(crate) fn allocation_site_stats(&self, site: u32) -> AllocationSiteStats {
        self.allocation_sites
            .get(&site)
            .copied()
            .unwrap_or_default()
    }

    /// Enter a short raw-pointer region. The returned guard borrows this heap, so mutable
    /// allocation, relocation, and reclamation cannot be called while the region is live.
    pub(crate) fn enter_no_gc(&self) -> Result<NoGcScope<'_>, NoGcError> {
        self.no_gc.enter()
    }

    pub(crate) fn require_safepoint(&self) -> Result<(), NoGcError> {
        self.no_gc.require_safepoint()
    }

    fn reserve_slot(&mut self) -> Result<(usize, HeapRef), HeapError> {
        if let Some(index) = self.free_slots.pop() {
            let cookie = self
                .slot_cookies
                .get(index)
                .copied()
                .ok_or(HeapError::ReferenceSpaceExhausted)?;
            let raw_index = u32::try_from(index).map_err(|_| HeapError::ReferenceSpaceExhausted)?;
            let reference =
                HeapRef::from_parts(raw_index, cookie).ok_or(HeapError::ReferenceSpaceExhausted)?;
            return Ok((index, reference));
        }
        let index = self.objects.len();
        let raw_index = u32::try_from(index).map_err(|_| HeapError::ReferenceSpaceExhausted)?;
        let reference =
            HeapRef::from_parts(raw_index, 0).ok_or(HeapError::ReferenceSpaceExhausted)?;
        self.objects.push(None);
        self.slot_cookies.push(0);
        Ok((index, reference))
    }

    fn release_slot(&mut self, index: usize) {
        let Some(cookie) = self.slot_cookies.get_mut(index) else {
            return;
        };
        // The 12-bit cookie is deliberately never wrapped. A slot that exhausts its cookie
        // space is retired instead of allowing an old tagged word to become valid again.
        if *cookie < 0x0fff {
            *cookie += 1;
            self.free_slots.push(index);
        }
    }

    fn resolve_slot(&self, reference: HeapRef) -> Result<usize, HeapError> {
        let index = usize::try_from(reference.index()).map_err(|_| HeapError::InvalidReference)?;
        let cookie = self
            .slot_cookies
            .get(index)
            .copied()
            .ok_or(HeapError::InvalidReference)?;
        if cookie != reference.cookie() {
            return Err(HeapError::InvalidReference);
        }
        Ok(index)
    }

    pub(crate) fn allocate(
        &mut self,
        layout: LayoutId,
        size_units: usize,
        generation: HeapGeneration,
    ) -> Result<HeapRef, HeapError> {
        self.allocate_at_site(layout, size_units, generation, 0)
    }

    pub(crate) fn allocate_at_site(
        &mut self,
        layout: LayoutId,
        size_units: usize,
        generation: HeapGeneration,
        allocation_site: u32,
    ) -> Result<HeapRef, HeapError> {
        let size_units = u32::try_from(size_units).map_err(|_| HeapError::RequestTooLarge)?;
        let (index, reference) = self.reserve_slot()?;
        let payload = vec![0; usize::try_from(size_units).unwrap()].into_boxed_slice();
        self.requested_bytes = self.requested_bytes.saturating_add(payload.len());
        self.objects[index] = Some(HeapObject {
            header: ObjectHeader {
                size_units,
                allocation_site,
                age: 0,
                layout,
                generation,
                marked: false,
                forwarding: None,
            },
            storage: HeapStorage::Bytes(payload),
        });
        self.record_site_allocation(allocation_site, size_units as usize, true);
        self.stats.allocations = self.stats.allocations.saturating_add(1);
        Ok(reference)
    }

    pub(crate) fn allocate_tagged_fields(
        &mut self,
        layout: LayoutId,
        fields: Vec<TaggedValue>,
        generation: HeapGeneration,
    ) -> Result<HeapRef, HeapError> {
        self.allocate_tagged_fields_at_site(layout, fields, generation, 0)
    }

    pub(crate) fn allocate_tagged_fields_at_site(
        &mut self,
        layout: LayoutId,
        fields: Vec<TaggedValue>,
        generation: HeapGeneration,
        allocation_site: u32,
    ) -> Result<HeapRef, HeapError> {
        let size_units = u32::try_from(fields.len()).map_err(|_| HeapError::RequestTooLarge)?;
        if fields.iter().any(|value| value.validate().is_err()) {
            return Err(HeapError::InvalidTaggedField);
        }
        for field in &fields {
            if let Some(child) = field.as_heap() {
                self.object(child)?;
            }
        }
        let storage = HeapStorage::Tagged(fields.into_boxed_slice());
        let (index, reference) = self.reserve_slot()?;
        self.requested_bytes = self
            .requested_bytes
            .saturating_add(storage.requested_bytes());
        self.objects[index] = Some(HeapObject {
            header: ObjectHeader {
                size_units,
                allocation_site,
                age: 0,
                layout,
                generation,
                marked: false,
                forwarding: None,
            },
            storage,
        });
        self.record_site_allocation(
            allocation_site,
            self.objects[index]
                .as_ref()
                .map_or(0, |object| object.storage.requested_bytes()),
            true,
        );
        self.stats.allocations = self.stats.allocations.saturating_add(1);
        self.refresh_remembered_reference(reference)?;
        Ok(reference)
    }

    fn object(&self, reference: HeapRef) -> Result<&HeapObject, HeapError> {
        let index = self.resolve_slot(reference)?;
        let object = self
            .objects
            .get(index)
            .and_then(Option::as_ref)
            .ok_or(HeapError::InvalidReference)?;
        if object.header.forwarding.is_some() {
            return Err(HeapError::ForwardedReference);
        }
        Ok(object)
    }

    fn object_mut(&mut self, reference: HeapRef) -> Result<&mut HeapObject, HeapError> {
        let index = self.resolve_slot(reference)?;
        let object = self
            .objects
            .get_mut(index)
            .and_then(Option::as_mut)
            .ok_or(HeapError::InvalidReference)?;
        if object.header.forwarding.is_some() {
            return Err(HeapError::ForwardedReference);
        }
        Ok(object)
    }

    pub(crate) fn header(&self, reference: HeapRef) -> Result<ObjectHeader, HeapError> {
        Ok(self.object(reference)?.header)
    }

    pub(crate) fn payload(&self, reference: HeapRef) -> Result<&[u8], HeapError> {
        match &self.object(reference)?.storage {
            HeapStorage::Bytes(payload) => Ok(payload),
            HeapStorage::Tagged(_) => Err(HeapError::PayloadKindMismatch),
        }
    }

    pub(crate) fn payload_mut(&mut self, reference: HeapRef) -> Result<&mut [u8], HeapError> {
        match &mut self.object_mut(reference)?.storage {
            HeapStorage::Bytes(payload) => Ok(payload),
            HeapStorage::Tagged(_) => Err(HeapError::PayloadKindMismatch),
        }
    }

    pub(crate) fn tagged_fields(&self, reference: HeapRef) -> Result<&[TaggedValue], HeapError> {
        match &self.object(reference)?.storage {
            HeapStorage::Tagged(fields) => Ok(fields),
            HeapStorage::Bytes(_) => Err(HeapError::PayloadKindMismatch),
        }
    }

    fn tagged_fields_mut(&mut self, reference: HeapRef) -> Result<&mut [TaggedValue], HeapError> {
        match &mut self.object_mut(reference)?.storage {
            HeapStorage::Tagged(fields) => Ok(fields),
            HeapStorage::Bytes(_) => Err(HeapError::PayloadKindMismatch),
        }
    }

    fn has_young_edge(&self, reference: HeapRef) -> Result<bool, HeapError> {
        let object = self.object(reference)?;
        if object.header.generation == HeapGeneration::Young {
            return Ok(false);
        }
        let HeapStorage::Tagged(fields) = &object.storage else {
            return Ok(false);
        };
        for field in fields {
            let Some(child) = field.as_heap() else {
                continue;
            };
            if self.object(child)?.header.generation == HeapGeneration::Young {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn refresh_remembered_reference(&mut self, reference: HeapRef) -> Result<(), HeapError> {
        self.remembered.retain(|entry| *entry != reference);
        if self.has_young_edge(reference)? {
            self.remembered.push(reference);
        }
        Ok(())
    }

    /// Store through the barriered tagged-field primitive. Direct mutable slices are retained for
    /// verifier fixtures, but migrated object families must use this method for every write.
    pub(crate) fn store_tagged_field(
        &mut self,
        reference: HeapRef,
        index: usize,
        value: TaggedValue,
    ) -> Result<(), HeapError> {
        value
            .validate()
            .map_err(|_| HeapError::InvalidTaggedField)?;
        if let Some(child) = value.as_heap() {
            self.object(child)?;
        }
        let fields = self.tagged_fields_mut(reference)?;
        let field = fields.get_mut(index).ok_or(HeapError::FieldOutOfBounds)?;
        *field = value;
        self.refresh_remembered_reference(reference)
    }

    pub(crate) fn remembered_handles(&self) -> Vec<HeapRef> {
        self.remembered.clone()
    }

    /// Recompute old-to-young edges independently of the incremental barrier state.
    pub(crate) fn recompute_remembered_set(&self) -> Result<Vec<HeapRef>, HeapError> {
        self.validate_all()?;
        let mut recomputed = Vec::new();
        for (index, slot) in self.objects.iter().enumerate().skip(1) {
            let Some(object) = slot else { continue };
            if object.header.forwarding.is_some()
                || object.header.generation == HeapGeneration::Young
            {
                continue;
            }
            let HeapStorage::Tagged(fields) = &object.storage else {
                continue;
            };
            let mut has_young_edge = false;
            for field in fields {
                let Some(child) = field.as_heap() else {
                    continue;
                };
                if self.object(child)?.header.generation == HeapGeneration::Young {
                    has_young_edge = true;
                    break;
                }
            }
            if has_young_edge {
                let index = u32::try_from(index).expect("heap index is bounded by the table");
                let cookie = self.slot_cookies[index as usize];
                recomputed.push(
                    HeapRef::from_parts(index, cookie).expect("live slot has a valid handle"),
                );
            }
        }
        Ok(recomputed)
    }

    pub(crate) fn verify_remembered_set(&self) -> Result<(), HeapError> {
        let mut recorded = self.remembered.clone();
        let mut recomputed = self.recompute_remembered_set()?;
        recorded.sort_by_key(|reference| reference.get());
        recomputed.sort_by_key(|reference| reference.get());
        if recorded == recomputed {
            Ok(())
        } else {
            Err(HeapError::RememberedSetMismatch)
        }
    }

    /// Rewrite every tagged field in the table after a relocation. The caller must rewrite its
    /// external roots through the corresponding `RootSet` operation before reclaiming from-space.
    pub(crate) fn rewrite_tagged_references(&mut self, from: HeapRef, to: HeapRef) -> usize {
        let replacement = TaggedValue::heap(to);
        let mut rewritten = 0;
        for object in self.objects.iter_mut().flatten() {
            let HeapStorage::Tagged(fields) = &mut object.storage else {
                continue;
            };
            for field in fields.iter_mut() {
                if field.as_heap() == Some(from) {
                    *field = replacement;
                    rewritten += 1;
                }
            }
        }
        rewritten
    }

    fn rewrite_tagged_references_many(
        &mut self,
        forwarding: &crate::fasthash::FastMap<HeapRef, HeapRef>,
    ) -> usize {
        let mut rewritten = 0;
        for object in self.objects.iter_mut().flatten() {
            let HeapStorage::Tagged(fields) = &mut object.storage else {
                continue;
            };
            for field in fields.iter_mut() {
                let Some(source) = field.as_heap() else {
                    continue;
                };
                let Some(target) = forwarding.get(&source) else {
                    continue;
                };
                *field = TaggedValue::heap(*target);
                rewritten += 1;
            }
        }
        rewritten
    }

    pub(crate) fn mark(&mut self, reference: HeapRef) -> Result<(), HeapError> {
        self.object_mut(reference)?.header.marked = true;
        Ok(())
    }

    /// Mark the transitive strong closure of tagged roots. The fixture heap has no weak or
    /// external families yet, so every tagged child is a strong edge; those families must add
    /// their own tracing policy before they are admitted to this walk.
    pub(crate) fn mark_roots(&mut self, roots: &RootSet) -> Result<usize, HeapError> {
        self.validate_all()?;
        let mut pending = roots
            .snapshot()
            .map_err(|_| HeapError::InvalidTaggedField)?
            .into_iter()
            .filter_map(TaggedValue::as_heap)
            .collect::<Vec<_>>();
        let mut marked = 0;
        while let Some(reference) = pending.pop() {
            let children = match &self.object(reference)?.storage {
                HeapStorage::Bytes(_) => Vec::new(),
                HeapStorage::Tagged(fields) => fields
                    .iter()
                    .filter_map(|field| field.as_heap())
                    .collect::<Vec<_>>(),
            };
            let object = self.object(reference)?;
            if object.header.marked {
                continue;
            }
            let scanned_bytes = object.storage.requested_bytes() as u64;
            let object = self.object_mut(reference)?;
            object.header.marked = true;
            self.stats.mark_objects_scanned = self.stats.mark_objects_scanned.saturating_add(1);
            self.stats.mark_bytes_scanned =
                self.stats.mark_bytes_scanned.saturating_add(scanned_bytes);
            marked += 1;
            pending.extend(children);
        }
        Ok(marked)
    }

    /// Evacuate the reachable young closure into the old generation. This is a deterministic
    /// fixture for the eventual nursery; it moves only the tagged-field family and leaves weak,
    /// external, and unsupported object families to their future policies.
    pub(crate) fn collect_nursery(
        &mut self,
        roots: &RootSet,
    ) -> Result<NurseryCollection, HeapError> {
        self.require_safepoint()
            .map_err(|_| HeapError::SafepointForbidden)?;
        self.mark_roots(roots)?;
        let young = self
            .live_handles()
            .into_iter()
            .filter(|reference| {
                self.header(*reference)
                    .is_ok_and(|header| header.generation == HeapGeneration::Young && header.marked)
            })
            .collect::<Vec<_>>();
        let mut promoted = 0;
        // The public relocation/reclamation methods independently validate the whole table. A
        // collection already has one validation at the mark boundary and one after all forwarding
        // edges are rewritten, so avoid turning an N-object nursery into an O(N²) verifier walk.
        self.validate_all()?;
        let mut forwarding = Vec::with_capacity(young.len());
        let mut forwarding_map =
            crate::fasthash::FastMap::with_capacity_and_hasher(young.len(), Default::default());
        for source in young {
            if self.header(source).is_err() {
                continue;
            }
            let target = self.relocate_unchecked(source)?;
            forwarding.push((source, target));
            forwarding_map.insert(source, target);
        }
        // Relocations are allocated in stable handle order above. Rewriting all fields in one
        // table walk avoids rescanning the entire heap once per moved object, while the ordered
        // list below preserves deterministic promotion and free-slot behavior.
        roots.rewrite_heap_references(&forwarding_map);
        self.rewrite_tagged_references_many(&forwarding_map);
        for (source, target) in forwarding {
            self.promote(target, HeapGeneration::Old)?;
            self.reclaim_forwarded_unchecked(source)?;
            promoted += 1;
        }
        self.remembered = self.recompute_remembered_set()?;
        let reclaimed = self.sweep_unmarked_young();
        self.validate_all()?;
        Ok(NurseryCollection {
            promoted,
            reclaimed,
        })
    }

    /// Collect the fixture nursery while a tagged shadow frame is published at an exact
    /// safepoint. Frame roots are temporarily held in the Agent root set, so relocation uses the
    /// same mark/forwarding path as ordinary roots; the frame is rewritten before those temporary
    /// handles are released. This is the bridge contract that a live execution tier will adopt.
    pub(crate) fn collect_nursery_with_frame(
        &mut self,
        roots: &RootSet,
        frame: &mut TaggedFrame,
        map: &RootMap,
    ) -> Result<NurseryCollection, HeapError> {
        map.validate().map_err(HeapError::InvalidFrameMap)?;
        let frame_values = frame.roots(map).map_err(HeapError::InvalidFrameMap)?;
        let frame_handles: Vec<RootedTagged> = frame_values
            .iter()
            .copied()
            .map(|value| roots.root(value).map_err(|_| HeapError::InvalidTaggedField))
            .collect::<Result<_, _>>()?;

        let result = self.collect_nursery(roots)?;

        for (old, handle) in frame_values.iter().zip(&frame_handles) {
            let Some(from) = old.as_heap() else {
                continue;
            };
            let Some(to) = handle.value().and_then(TaggedValue::as_heap) else {
                continue;
            };
            frame
                .rewrite_heap_reference(map, from, to)
                .map_err(HeapError::InvalidFrameMap)?;
        }
        Ok(result)
    }

    pub(crate) fn promote(
        &mut self,
        reference: HeapRef,
        generation: HeapGeneration,
    ) -> Result<(), HeapError> {
        let previous = self.object(reference)?.header.generation;
        let object = self.object_mut(reference)?;
        object.header.generation = generation;
        let mut promoted_site = None;
        if previous == HeapGeneration::Young && generation != HeapGeneration::Young {
            object.header.age = object.header.age.saturating_add(1);
            promoted_site = Some((object.header.allocation_site, object.header.age));
        }
        if previous != generation {
            self.stats.promotions = self.stats.promotions.saturating_add(1);
        }
        if let Some((site, age)) = promoted_site {
            let stats = self.allocation_sites.entry(site).or_default();
            stats.promotions = stats.promotions.saturating_add(1);
            stats.max_age = stats.max_age.max(age);
        }
        self.refresh_remembered_reference(reference)
    }

    pub(crate) fn requested_bytes(&self) -> usize {
        self.requested_bytes
    }

    pub(crate) fn stats(&self) -> HeapStats {
        self.stats
    }

    /// Verify every live slot, header/storage size, forwarding edge, and tagged child reference.
    /// This is intentionally an independent walk rather than a fast-path assertion so stress
    /// collectors can run it before and after every relocation.
    pub(crate) fn validate_all(&self) -> Result<(), HeapError> {
        for (index, object) in self.objects.iter().enumerate().skip(1) {
            let Some(object) = object else { continue };
            let expected_units = match &object.storage {
                HeapStorage::Bytes(payload) => payload.len(),
                HeapStorage::Tagged(fields) => fields.len(),
            };
            if object.header.size_units as usize != expected_units {
                return Err(HeapError::HeaderMismatch);
            }
            if let Some(target) = object.header.forwarding {
                if target.index() as usize == index {
                    return Err(HeapError::HeaderMismatch);
                }
                let target_index = self.resolve_slot(target)?;
                let target_object = self
                    .objects
                    .get(target_index)
                    .and_then(Option::as_ref)
                    .ok_or(HeapError::InvalidReference)?;
                if target_object.header.forwarding.is_some() {
                    return Err(HeapError::HeaderMismatch);
                }
            }
            let HeapStorage::Tagged(fields) = &object.storage else {
                continue;
            };
            for field in fields {
                field
                    .validate()
                    .map_err(|_| HeapError::InvalidTaggedField)?;
                let Some(child) = field.as_heap() else {
                    continue;
                };
                let child_object = self.object(child)?;
                if child_object.header.forwarding.is_some() {
                    return Err(HeapError::ForwardedReference);
                }
            }
        }
        Ok(())
    }

    fn sweep_matching<F>(&mut self, should_reclaim: F) -> usize
    where
        F: Fn(&HeapObject) -> bool,
    {
        let mut reclaimed = 0;
        let mut released_bytes: usize = 0;
        let mut scanned_objects = 0;
        let mut scanned_bytes: usize = 0;
        let mut released_slots = Vec::new();
        let mut released_sites = Vec::new();
        for (index, slot) in self.objects.iter_mut().enumerate().skip(1) {
            if let Some(object) = slot.as_ref() {
                scanned_objects += 1;
                scanned_bytes = scanned_bytes.saturating_add(object.storage.requested_bytes());
            }
            let remove = slot.as_ref().is_some_and(&should_reclaim);
            if remove {
                if let Some(object) = slot.take() {
                    let bytes = object.storage.requested_bytes();
                    released_bytes = released_bytes.saturating_add(bytes);
                    released_sites.push((object.header.allocation_site, bytes, object.header.age));
                    released_slots.push(index);
                    reclaimed += 1;
                }
            } else if let Some(object) = slot {
                object.header.marked = false;
            }
        }
        self.requested_bytes = self.requested_bytes.saturating_sub(released_bytes);
        self.stats.collections = self.stats.collections.saturating_add(1);
        self.stats.sweep_objects_scanned = self
            .stats
            .sweep_objects_scanned
            .saturating_add(scanned_objects);
        self.stats.sweep_bytes_scanned = self
            .stats
            .sweep_bytes_scanned
            .saturating_add(u64::try_from(scanned_bytes).unwrap_or(u64::MAX));
        self.stats.freed_bytes = self
            .stats
            .freed_bytes
            .saturating_add(u64::try_from(released_bytes).unwrap_or(u64::MAX));
        for (site, bytes, age) in released_sites {
            self.record_site_release(site, bytes, age);
        }
        for index in released_slots {
            self.release_slot(index);
        }
        let remembered = std::mem::take(&mut self.remembered);
        self.remembered = remembered
            .into_iter()
            .filter(|reference| self.object(*reference).is_ok())
            .collect();
        reclaimed
    }

    /// Deterministically reclaim unmarked, non-forwarded slots. Mark bits are cleared for the
    /// survivors so the next collection starts from a clean white set. Forwarded sources remain
    /// until their explicit relocation handshake calls `reclaim_forwarded`.
    pub(crate) fn sweep_unmarked(&mut self) -> usize {
        self.sweep_matching(|object| !object.header.marked && object.header.forwarding.is_none())
    }

    fn sweep_unmarked_young(&mut self) -> usize {
        self.sweep_matching(|object| {
            object.header.generation == HeapGeneration::Young
                && !object.header.marked
                && object.header.forwarding.is_none()
        })
    }

    pub(crate) fn live_handles(&self) -> Vec<HeapRef> {
        self.objects
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(index, object)| {
                object.as_ref().and_then(|_| {
                    let index = u32::try_from(index).ok()?;
                    let cookie = *self.slot_cookies.get(index as usize)?;
                    HeapRef::from_parts(index, cookie)
                })
            })
            .collect()
    }

    /// Copy one object to a new slot and publish a forwarding edge on the source. The source is
    /// inaccessible through ordinary accessors until `reclaim_forwarded` runs, forcing callers to
    /// update every root and traced field before releasing from-space.
    pub(crate) fn relocate(&mut self, source: HeapRef) -> Result<HeapRef, HeapError> {
        self.validate_all()?;
        self.relocate_unchecked(source)
    }

    fn relocate_unchecked(&mut self, source: HeapRef) -> Result<HeapRef, HeapError> {
        let source_index = self.resolve_slot(source)?;
        let source_object = self.objects[source_index]
            .as_ref()
            .ok_or(HeapError::InvalidReference)?;
        if source_object.header.forwarding.is_some() {
            return Err(HeapError::AlreadyForwarded);
        }
        let mut moved = source_object.clone();
        moved.header.forwarding = None;
        let copied_bytes = u64::try_from(moved.storage.requested_bytes()).unwrap_or(u64::MAX);
        let copied_payload_bytes = moved.storage.requested_bytes();
        let (target_index, target) = self.reserve_slot()?;
        self.requested_bytes = self
            .requested_bytes
            .saturating_add(moved.storage.requested_bytes());
        self.objects[target_index] = Some(moved);
        self.record_site_allocation(
            self.objects[target_index]
                .as_ref()
                .map_or(0, |object| object.header.allocation_site),
            copied_payload_bytes,
            false,
        );
        self.stats.copied_bytes = self.stats.copied_bytes.saturating_add(copied_bytes);
        self.refresh_remembered_reference(target)?;
        let source_object = self.objects[source_index]
            .as_mut()
            .ok_or(HeapError::InvalidReference)?;
        source_object.header.forwarding = Some(target);
        Ok(target)
    }

    /// Reclaim a source only after its forwarding target has been published and all external
    /// references have been rewritten. The old handle then fails closed as invalid.
    pub(crate) fn reclaim_forwarded(&mut self, source: HeapRef) -> Result<HeapRef, HeapError> {
        self.validate_all()?;
        self.reclaim_forwarded_unchecked(source)
    }

    fn reclaim_forwarded_unchecked(&mut self, source: HeapRef) -> Result<HeapRef, HeapError> {
        let index = self.resolve_slot(source)?;
        let object = self.objects[index]
            .as_ref()
            .ok_or(HeapError::InvalidReference)?;
        let target = object.header.forwarding.ok_or(HeapError::NotForwarded)?;
        let released = self.objects[index]
            .take()
            .ok_or(HeapError::InvalidReference)?;
        let released_bytes = released.storage.requested_bytes();
        self.requested_bytes = self.requested_bytes.saturating_sub(released_bytes);
        self.record_site_release(
            released.header.allocation_site,
            released_bytes,
            released.header.age,
        );
        self.stats.freed_bytes = self
            .stats
            .freed_bytes
            .saturating_add(u64::try_from(released_bytes).unwrap_or(u64::MAX));
        self.remembered.retain(|entry| *entry != source);
        self.release_slot(index);
        Ok(target)
    }

    /// Remove one unreachable object. This path is intentionally separate from relocation so a
    /// collector cannot accidentally reclaim an object while a forwarding edge is still active.
    pub(crate) fn free(&mut self, reference: HeapRef) -> Result<(), HeapError> {
        let index = self.resolve_slot(reference)?;
        let object = self.objects[index]
            .as_ref()
            .ok_or(HeapError::InvalidReference)?;
        if object.header.forwarding.is_some() {
            return Err(HeapError::ForwardedReference);
        }
        let released = self.objects[index]
            .take()
            .ok_or(HeapError::InvalidReference)?;
        let released_bytes = released.storage.requested_bytes();
        self.requested_bytes = self.requested_bytes.saturating_sub(released_bytes);
        self.record_site_release(
            released.header.allocation_site,
            released_bytes,
            released.header.age,
        );
        self.stats.freed_bytes = self
            .stats
            .freed_bytes
            .saturating_add(u64::try_from(released_bytes).unwrap_or(u64::MAX));
        self.remembered.retain(|entry| *entry != reference);
        self.release_slot(index);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagged::{BytecodePc, RootMap, RootSet, SafepointId, TaggedFrame};

    #[test]
    fn allocation_uses_checked_handles_and_separate_payload_accounting() {
        let mut heap = CentralHeap::new();
        let reference = heap
            .allocate(LayoutId::new(3), 12, HeapGeneration::Young)
            .unwrap();
        assert_eq!(reference.get(), 1);
        assert_eq!(heap.requested_bytes(), 12);
        assert_eq!(heap.header(reference).unwrap().size_units, 12);
        assert_eq!(
            heap.header(reference).unwrap().generation,
            HeapGeneration::Young
        );
        assert_eq!(heap.payload(reference).unwrap().len(), 12);
        assert_eq!(heap.stats().allocations, 1);
        assert_eq!(heap.stats().mark_objects_scanned, 0);
        heap.payload_mut(reference).unwrap()[4] = 0xa5;
        assert_eq!(heap.payload(reference).unwrap()[4], 0xa5);
        heap.mark(reference).unwrap();
        assert!(heap.header(reference).unwrap().marked);
    }

    #[test]
    fn no_gc_guard_is_owned_by_the_heap_and_blocks_nested_safepoints() {
        let heap = CentralHeap::new();
        let guard = heap.enter_no_gc().unwrap();
        guard.assert_active();
        assert_eq!(heap.require_safepoint(), Err(NoGcError::SafepointForbidden));
        assert!(matches!(heap.enter_no_gc(), Err(NoGcError::AlreadyActive)));
        drop(guard);
        assert_eq!(heap.require_safepoint(), Ok(()));
    }

    #[test]
    fn invalid_and_freed_handles_fail_closed() {
        let mut heap = CentralHeap::new();
        assert_eq!(HeapRef::new(0), None);
        let reference = heap
            .allocate(LayoutId::new(1), 1, HeapGeneration::Old)
            .unwrap();
        heap.free(reference).unwrap();
        assert_eq!(heap.payload(reference), Err(HeapError::InvalidReference));
        assert_eq!(heap.requested_bytes(), 0);
        let replacement = heap
            .allocate(LayoutId::new(2), 2, HeapGeneration::Young)
            .unwrap();
        assert_eq!(replacement.index(), reference.index());
        assert_ne!(replacement.cookie(), reference.cookie());
        assert_eq!(heap.payload(reference), Err(HeapError::InvalidReference));
        assert_eq!(heap.payload(replacement).unwrap().len(), 2);
    }

    #[test]
    fn exhausted_generation_cookie_retires_a_slot_instead_of_wrapping() {
        let mut heap = CentralHeap::new();
        let first = heap
            .allocate(LayoutId::new(6), 1, HeapGeneration::Young)
            .unwrap();
        let mut current = first;
        for _ in 0..0x0fff {
            heap.free(current).unwrap();
            current = heap
                .allocate(LayoutId::new(6), 1, HeapGeneration::Young)
                .unwrap();
            assert_eq!(current.index(), first.index());
        }
        heap.free(current).unwrap();
        let replacement = heap
            .allocate(LayoutId::new(7), 1, HeapGeneration::Young)
            .unwrap();
        assert_ne!(replacement.index(), first.index());
        assert_eq!(heap.payload(first), Err(HeapError::InvalidReference));
    }

    #[test]
    fn relocation_requires_rewriting_before_reclaiming_from_space() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate(LayoutId::new(9), 4, HeapGeneration::Young)
            .unwrap();
        heap.payload_mut(source)
            .unwrap()
            .copy_from_slice(&[1, 2, 3, 4]);
        let target = heap.relocate(source).unwrap();
        assert_eq!(heap.payload(source), Err(HeapError::ForwardedReference));
        assert_eq!(heap.payload(target).unwrap(), &[1, 2, 3, 4]);
        assert_eq!(heap.stats().copied_bytes, 4);
        assert_eq!(heap.relocate(source), Err(HeapError::AlreadyForwarded));
        assert_eq!(heap.reclaim_forwarded(source), Ok(target));
        assert_eq!(heap.payload(source), Err(HeapError::InvalidReference));
        assert_eq!(heap.requested_bytes(), 4);
        assert_eq!(heap.stats().freed_bytes, 4);
    }

    #[test]
    fn ordinary_free_does_not_accept_a_forwarded_source() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate(LayoutId::new(2), 2, HeapGeneration::Young)
            .unwrap();
        let _target = heap.relocate(source).unwrap();
        assert_eq!(heap.free(source), Err(HeapError::ForwardedReference));
    }

    #[test]
    fn tagged_leaf_rewrites_roots_and_self_references_before_reclaim() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate_tagged_fields(
                LayoutId::new(11),
                vec![TaggedValue::undefined(), TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        heap.store_tagged_field(source, 0, TaggedValue::heap(source))
            .unwrap();
        let roots = RootSet::new();
        let root = roots.root(TaggedValue::heap(source)).unwrap();

        let target = heap.relocate(source).unwrap();
        assert_eq!(
            heap.rewrite_tagged_references(source, target),
            2,
            "one self-edge in from-space and one copied self-edge must be rewritten"
        );
        assert_eq!(roots.rewrite_heap_reference(source, target), 1);
        assert_eq!(heap.reclaim_forwarded(source), Ok(target));
        assert_eq!(
            heap.tagged_fields(target).unwrap()[0].as_heap(),
            Some(target)
        );
        assert_eq!(root.value().and_then(TaggedValue::as_heap), Some(target));
    }

    #[test]
    fn tagged_leaf_rejects_invalid_fields_and_keeps_byte_payloads_distinct() {
        let mut heap = CentralHeap::new();
        assert_eq!(
            heap.allocate_tagged_fields(
                LayoutId::new(1),
                vec![TaggedValue::from_raw(0x7ffc_0000_0000_0002)],
                HeapGeneration::Young,
            ),
            Err(HeapError::InvalidTaggedField)
        );
        let bytes = heap
            .allocate(LayoutId::new(2), 1, HeapGeneration::Old)
            .unwrap();
        assert_eq!(
            heap.tagged_fields(bytes),
            Err(HeapError::PayloadKindMismatch)
        );
    }

    #[test]
    fn validation_and_sweep_preserve_marked_generation_and_accounting() {
        let mut heap = CentralHeap::new();
        let keep = heap
            .allocate_at_site(LayoutId::new(4), 3, HeapGeneration::Young, 77)
            .unwrap();
        let discard = heap
            .allocate(LayoutId::new(5), 7, HeapGeneration::Old)
            .unwrap();
        assert_eq!(heap.validate_all(), Ok(()));
        heap.promote(keep, HeapGeneration::Old).unwrap();
        heap.mark(keep).unwrap();
        assert_eq!(heap.header(keep).unwrap().generation, HeapGeneration::Old);
        assert_eq!(heap.header(keep).unwrap().allocation_site, 77);
        assert_eq!(heap.header(keep).unwrap().age, 1);
        assert_eq!(heap.stats().promotions, 1);
        heap.promote(keep, HeapGeneration::Old).unwrap();
        assert_eq!(heap.header(keep).unwrap().age, 1);
        assert_eq!(heap.stats().promotions, 1);
        assert_eq!(heap.sweep_unmarked(), 1);
        assert_eq!(heap.requested_bytes(), 3);
        assert_eq!(heap.live_handles(), vec![keep]);
        assert_eq!(heap.payload(discard), Err(HeapError::InvalidReference));
        assert!(!heap.header(keep).unwrap().marked);
        assert_eq!(heap.validate_all(), Ok(()));
    }

    #[test]
    fn allocation_site_stats_track_logical_bytes_age_and_relocation_without_double_counting() {
        let mut heap = CentralHeap::new();
        let keep = heap
            .allocate_at_site(LayoutId::new(17), 3, HeapGeneration::Young, 77)
            .unwrap();
        let discard = heap
            .allocate_at_site(LayoutId::new(18), 5, HeapGeneration::Young, 77)
            .unwrap();
        let initial = heap.allocation_site_stats(77);
        assert_eq!(initial.allocations, 2);
        assert_eq!(initial.requested_bytes, 8);
        assert_eq!(initial.live_allocations, 2);
        assert_eq!(initial.live_requested_bytes, 8);

        heap.promote(keep, HeapGeneration::Old).unwrap();
        heap.free(discard).unwrap();
        let after_free = heap.allocation_site_stats(77);
        assert_eq!(after_free.promotions, 1);
        assert_eq!(after_free.reclaimed, 1);
        assert_eq!(after_free.live_allocations, 1);
        assert_eq!(after_free.live_requested_bytes, 3);
        assert_eq!(after_free.max_age, 1);

        let target = heap.relocate(keep).unwrap();
        assert_eq!(heap.reclaim_forwarded(keep), Ok(target));
        let after_relocation = heap.allocation_site_stats(77);
        assert_eq!(after_relocation.allocations, 2);
        assert_eq!(after_relocation.requested_bytes, 8);
        assert_eq!(after_relocation.reclaimed, 2);
        assert_eq!(after_relocation.live_allocations, 1);
        assert_eq!(after_relocation.live_requested_bytes, 3);
    }

    #[test]
    fn root_mark_walk_reaches_tagged_children_before_sweeping() {
        let mut heap = CentralHeap::new();
        let child = heap
            .allocate_tagged_fields(
                LayoutId::new(20),
                vec![TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        let parent = heap
            .allocate_tagged_fields(
                LayoutId::new(21),
                vec![TaggedValue::heap(child)],
                HeapGeneration::Old,
            )
            .unwrap();
        let unreachable = heap
            .allocate(LayoutId::new(22), 5, HeapGeneration::Young)
            .unwrap();
        let roots = RootSet::new();
        let _root = roots.root(TaggedValue::heap(parent)).unwrap();

        assert_eq!(heap.remembered_handles(), vec![parent]);
        assert_eq!(heap.verify_remembered_set(), Ok(()));
        assert_eq!(heap.mark_roots(&roots), Ok(2));
        assert!(heap.header(parent).unwrap().marked);
        assert!(heap.header(child).unwrap().marked);
        assert_eq!(heap.sweep_unmarked(), 1);
        assert_eq!(heap.payload(unreachable), Err(HeapError::InvalidReference));
        assert_eq!(heap.verify_remembered_set(), Ok(()));
        let stats = heap.stats();
        assert_eq!(stats.allocations, 3);
        assert_eq!(stats.mark_objects_scanned, 2);
        assert_eq!(stats.mark_bytes_scanned, 16);
        assert_eq!(stats.collections, 1);
        assert_eq!(stats.sweep_objects_scanned, 3);
        assert_eq!(stats.sweep_bytes_scanned, 21);
        assert_eq!(stats.freed_bytes, 5);
        assert_eq!(heap.validate_all(), Ok(()));
    }

    #[test]
    fn old_to_young_barrier_tracks_and_recomputes_edges() {
        let mut heap = CentralHeap::new();
        let old = heap
            .allocate_tagged_fields(
                LayoutId::new(30),
                vec![TaggedValue::null()],
                HeapGeneration::Old,
            )
            .unwrap();
        let young = heap
            .allocate_tagged_fields(
                LayoutId::new(31),
                vec![TaggedValue::undefined()],
                HeapGeneration::Young,
            )
            .unwrap();
        assert!(heap.remembered_handles().is_empty());
        heap.store_tagged_field(old, 0, TaggedValue::heap(young))
            .unwrap();
        assert_eq!(heap.remembered_handles(), vec![old]);
        assert_eq!(heap.verify_remembered_set(), Ok(()));

        heap.store_tagged_field(old, 0, TaggedValue::null())
            .unwrap();
        assert!(heap.remembered_handles().is_empty());
        assert_eq!(heap.verify_remembered_set(), Ok(()));

        heap.store_tagged_field(old, 0, TaggedValue::heap(young))
            .unwrap();
        heap.promote(old, HeapGeneration::Young).unwrap();
        assert!(heap.remembered_handles().is_empty());
        assert_eq!(heap.verify_remembered_set(), Ok(()));
    }

    #[test]
    fn nursery_collection_promotes_reachable_closure_and_reclaims_garbage() {
        let mut heap = CentralHeap::new();
        let child = heap
            .allocate_tagged_fields(
                LayoutId::new(40),
                vec![TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        let parent = heap
            .allocate_tagged_fields(
                LayoutId::new(41),
                vec![TaggedValue::heap(child)],
                HeapGeneration::Young,
            )
            .unwrap();
        let garbage = heap
            .allocate(LayoutId::new(42), 5, HeapGeneration::Young)
            .unwrap();
        let roots = RootSet::new();
        let root = roots.root(TaggedValue::heap(parent)).unwrap();

        let result = heap.collect_nursery(&roots).unwrap();
        assert_eq!(
            result,
            NurseryCollection {
                promoted: 2,
                reclaimed: 1
            }
        );
        let moved_parent = root.value().and_then(TaggedValue::as_heap).unwrap();
        assert_ne!(moved_parent, parent);
        assert_eq!(heap.header(parent), Err(HeapError::InvalidReference));
        assert_eq!(heap.header(child), Err(HeapError::InvalidReference));
        assert_eq!(heap.payload(garbage), Err(HeapError::InvalidReference));
        assert_eq!(
            heap.header(moved_parent).unwrap().generation,
            HeapGeneration::Old
        );
        let moved_child = heap.tagged_fields(moved_parent).unwrap()[0]
            .as_heap()
            .unwrap();
        assert_eq!(
            heap.header(moved_child).unwrap().generation,
            HeapGeneration::Old
        );
        assert!(heap.remembered_handles().is_empty());
        assert_eq!(heap.verify_remembered_set(), Ok(()));
        assert_eq!(heap.requested_bytes(), 16);
    }

    #[test]
    fn validation_requires_relocation_edges_to_be_rewritten() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate_tagged_fields(
                LayoutId::new(12),
                vec![TaggedValue::undefined()],
                HeapGeneration::Young,
            )
            .unwrap();
        heap.store_tagged_field(source, 0, TaggedValue::heap(source))
            .unwrap();
        let target = heap.relocate(source).unwrap();
        assert_eq!(heap.validate_all(), Err(HeapError::ForwardedReference));
        assert_eq!(
            heap.reclaim_forwarded(source),
            Err(HeapError::ForwardedReference)
        );
        assert_eq!(heap.rewrite_tagged_references(source, target), 2);
        assert_eq!(heap.validate_all(), Ok(()));
        assert_eq!(heap.reclaim_forwarded(source), Ok(target));
        assert_eq!(heap.validate_all(), Ok(()));
    }

    #[test]
    fn forced_safepoint_relocates_a_migrated_frame_before_reclaim() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate_tagged_fields(
                LayoutId::new(13),
                vec![TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        let map = RootMap {
            safepoint: SafepointId::new(6),
            bytecode_pc: BytecodePc::new(31),
            slot_count: 2,
            operand_depth: 1,
            tagged_registers: 0,
            tagged_slots: vec![0b01],
            handle_slots: vec![0],
            environment_slots: vec![0],
        };
        let mut frame = TaggedFrame::new(vec![TaggedValue::heap(source), TaggedValue::number(8.0)]);
        let roots = RootSet::new();

        let no_gc = heap.enter_no_gc().unwrap();
        assert_eq!(heap.require_safepoint(), Err(NoGcError::SafepointForbidden));
        drop(no_gc);
        assert_eq!(heap.require_safepoint(), Ok(()));

        let result = heap
            .collect_nursery_with_frame(&roots, &mut frame, &map)
            .unwrap();
        assert_eq!(result.promoted, 1);
        assert_eq!(result.reclaimed, 0);
        let target = frame.roots(&map).unwrap()[0].as_heap().unwrap();
        assert_ne!(target, source);
        assert_eq!(roots.active_len(), 0);
        assert_eq!(heap.header(source), Err(HeapError::InvalidReference));
        assert_eq!(frame.roots(&map).unwrap()[0].as_heap(), Some(target));
        assert_eq!(heap.validate_all(), Ok(()));
    }

    #[test]
    fn frame_collection_ignores_poisoned_dead_slots() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate_tagged_fields(
                LayoutId::new(14),
                vec![TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        let map = RootMap {
            safepoint: SafepointId::new(7),
            bytecode_pc: BytecodePc::new(32),
            slot_count: 2,
            operand_depth: 1,
            tagged_registers: 0,
            tagged_slots: vec![0b01],
            handle_slots: vec![0],
            environment_slots: vec![0],
        };
        let mut frame = TaggedFrame::new(vec![
            TaggedValue::heap(source),
            TaggedValue::from_raw(0x7ffc_0000_0000_0002),
        ]);
        let roots = RootSet::new();
        heap.collect_nursery_with_frame(&roots, &mut frame, &map)
            .unwrap();
        assert!(frame.roots(&map).unwrap()[0].as_heap().is_some());
        assert_eq!(heap.validate_all(), Ok(()));
    }

    #[test]
    fn frame_collection_rejects_invalid_map_without_mutating_heap_or_frame() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate_tagged_fields(
                LayoutId::new(15),
                vec![TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        let map = RootMap {
            safepoint: SafepointId::new(8),
            bytecode_pc: BytecodePc::new(33),
            slot_count: 1,
            operand_depth: 2,
            tagged_registers: 0,
            tagged_slots: vec![1],
            handle_slots: vec![0],
            environment_slots: vec![0],
        };
        let mut frame = TaggedFrame::new(vec![TaggedValue::heap(source)]);
        let before = frame.clone();
        let roots = RootSet::new();
        assert_eq!(
            heap.collect_nursery_with_frame(&roots, &mut frame, &map),
            Err(HeapError::InvalidFrameMap(
                FrameMapError::OperandDepthOutOfRange
            ))
        );
        assert_eq!(frame, before);
        assert_eq!(
            heap.header(source).unwrap().generation,
            HeapGeneration::Young
        );
        assert_eq!(roots.active_len(), 0);
    }

    #[test]
    fn nursery_collection_rejects_a_forbidden_safepoint_before_marking() {
        let mut heap = CentralHeap::new();
        let source = heap
            .allocate_tagged_fields(
                LayoutId::new(16),
                vec![TaggedValue::null()],
                HeapGeneration::Young,
            )
            .unwrap();
        let roots = RootSet::new();
        let _root = roots.root(TaggedValue::heap(source)).unwrap();
        heap.no_gc.set_active_for_test(true);
        assert_eq!(
            heap.collect_nursery(&roots),
            Err(HeapError::SafepointForbidden)
        );
        heap.no_gc.set_active_for_test(false);
        assert_eq!(
            heap.header(source).unwrap().generation,
            HeapGeneration::Young
        );
        assert_eq!(heap.stats().collections, 0);
    }
}
