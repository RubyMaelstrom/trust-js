//! Observations of the production Rc heap, never a shadow ownership graph.
//! ECMA-262 e28783d5 #sec-weakref-processing-model: records contain integers and
//! static Rust allocation locations only. They cannot extend JavaScript liveness.
//! Feature-gated out of ordinary builds. Lifetimes use allocation ordinals and GC
//! generations, not a clock per allocation; initial families are explicitly labelled.
use super::{Callable, Exotic, GcHeap, Property, Props, INLINE_PACKED_CAPACITY, PACK_LAZY_PROTO};
use std::collections::BTreeMap;
use std::panic::Location;

const MAX_SITES: usize = 256;
const MAX_SLOTS: usize = 262_144;
const FAMILIES: [&str; 7] = [
    "ordinary",
    "callable",
    "array",
    "arguments",
    "error",
    "wrapper",
    "side_table",
];

#[derive(Clone, Copy)]
struct Birth {
    site: usize,
    ordinal: u64,
    generation: u64,
}
#[derive(Default)]
struct Site {
    allocations: u64,
    deaths: u64,
    deaths_before_collection: u64,
    requested_named_capacity: u64,
    lifetime_allocations: [u64; 8],
}
#[derive(Default)]
pub(super) struct State {
    sites: BTreeMap<(&'static str, u32, &'static str), usize>,
    counts: Vec<Site>,
    births: Vec<Option<Birth>>,
    generation: u64,
    ordinal: u64,
    untracked_allocations: u64,
}

fn initial_family(exotic: &Exotic) -> &'static str {
    match exotic {
        Exotic::None => "ordinary",
        Exotic::Array => "array",
        Exotic::Arguments => "arguments",
        Exotic::Error(_) => "error",
        _ => "wrapper",
    }
}

/// Descriptor counts for fields in one backing representation. Data-property mask bits are
/// W/E/C in bits 0/1/2; mask 7 is the default. Accessor attributes and getter/setter presence
/// are reported separately because Writable does not apply to accessor descriptors.
#[derive(Clone, Copy, Default)]
struct FieldInventory {
    slots: u64,
    data: u64,
    default_wec_data: u64,
    exceptional_data_masks: [u64; 8],
    accessors: u64,
    accessor_enumerable_configurable: [u64; 4],
    accessor_getter_setter: [u64; 4],
    holes: u64,
    lazy_prototype_values: u64,
}

impl FieldInventory {
    fn add_property(&mut self, property: &Property) {
        self.slots += 1;
        if property.is_empty() {
            self.holes += 1;
            return;
        }
        if property.packed.tag() == PACK_LAZY_PROTO {
            // Inspect only the packed tag: `value()` would materialize and change the object.
            self.lazy_prototype_values += 1;
        }
        if property.accessor() {
            self.accessors += 1;
            let attributes =
                (property.enumerable() as usize) | ((property.configurable() as usize) << 1);
            self.accessor_enumerable_configurable[attributes] += 1;
            let endpoints = (property.getter().is_some() as usize)
                | ((property.setter().is_some() as usize) << 1);
            self.accessor_getter_setter[endpoints] += 1;
            return;
        }
        self.data += 1;
        let mask = (property.writable() as usize)
            | ((property.enumerable() as usize) << 1)
            | ((property.configurable() as usize) << 2);
        if mask == 0b111 {
            self.default_wec_data += 1;
        } else {
            self.exceptional_data_masks[mask] += 1;
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.slots += other.slots;
        self.data += other.data;
        self.default_wec_data += other.default_wec_data;
        self.accessors += other.accessors;
        self.holes += other.holes;
        self.lazy_prototype_values += other.lazy_prototype_values;
        for (dst, src) in self
            .exceptional_data_masks
            .iter_mut()
            .zip(other.exceptional_data_masks)
        {
            *dst += src;
        }
        for (dst, src) in self
            .accessor_enumerable_configurable
            .iter_mut()
            .zip(other.accessor_enumerable_configurable)
        {
            *dst += src;
        }
        for (dst, src) in self
            .accessor_getter_setter
            .iter_mut()
            .zip(other.accessor_getter_setter)
        {
            *dst += src;
        }
    }

    fn all_default_data_or_holes(&self) -> bool {
        self.slots != 0
            && self.data == self.default_wec_data
            && self.accessors == 0
            && self.data + self.holes == self.slots
    }

    fn json(&self) -> String {
        format!(
            "{{\"slots\":{},\"data\":{},\"default_wec_data\":{},\"exceptional_data_mask_counts_0_to_7\":{:?},\"accessors\":{},\"accessor_enumerable_configurable_mask_counts_0_to_3\":{:?},\"accessor_getter_setter_mask_counts_0_to_3\":{:?},\"holes\":{},\"lazy_prototype_values\":{}}}",
            self.slots,
            self.data,
            self.default_wec_data,
            self.exceptional_data_masks,
            self.accessors,
            self.accessor_enumerable_configurable,
            self.accessor_getter_setter,
            self.holes,
            self.lazy_prototype_values,
        )
    }
}

#[derive(Clone, Copy, Default)]
struct LengthPopulation {
    populations: u64,
    live_length: u64,
    reserved_capacity: u64,
}

impl LengthPopulation {
    fn add_assign(&mut self, other: &Self) {
        self.populations += other.populations;
        self.live_length += other.live_length;
        self.reserved_capacity += other.reserved_capacity;
    }

    fn observe(&mut self, live_length: usize, reserved_capacity: usize) {
        self.populations += 1;
        self.live_length += live_length as u64;
        self.reserved_capacity += reserved_capacity as u64;
    }
}

const ELIGIBLE_ENTRY_LENGTH_BOUNDS: [(usize, Option<usize>); 8] = [
    (1, Some(1)),
    (2, Some(2)),
    (3, Some(3)),
    (4, Some(4)),
    (5, Some(8)),
    (9, Some(16)),
    (17, Some(32)),
    (33, None),
];

const INLINE_DENSE_LENGTH_BOUNDS: [(usize, Option<usize>); 7] = [
    (0, Some(0)),
    (1, Some(1)),
    (2, Some(2)),
    (3, Some(3)),
    (4, Some(4)),
    (5, Some(8)),
    (9, Some(INLINE_PACKED_CAPACITY)),
];

fn length_bucket_index(len: usize, bounds: &[(usize, Option<usize>)]) -> usize {
    bounds
        .iter()
        .position(|(min, max)| len >= *min && max.is_none_or(|max| len <= max))
        .expect("length bucket bounds cover the supported range")
}

fn length_population_json(
    populations: &[LengthPopulation],
    bounds: &[(usize, Option<usize>)],
) -> String {
    let bins = populations
        .iter()
        .zip(bounds)
        .map(|(population, (min_len, max_len))| {
            let max_len = max_len.map_or_else(|| "null".to_owned(), |n| n.to_string());
            format!(
                "{{\"min_len\":{min_len},\"max_len\":{max_len},\"populations\":{},\"live_length\":{},\"reserved_capacity\":{}}}",
                population.populations,
                population.live_length,
                population.reserved_capacity,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{bins}]")
}

/// Object-level cost surface for a compact descriptor representation. Empty entry vectors are
/// kept separate so vacuous eligibility cannot inflate the nonempty promotion population.
#[derive(Clone, Copy, Default)]
struct EntryVecEligibility {
    vecs: u64,
    empty_vecs: u64,
    empty_reserved_capacity: u64,
    empty_reserved_property_bytes: u64,
    nonempty_vecs: u64,
    nonempty_all_default_wec_data_vecs: u64,
    eligible_live_length: u64,
    eligible_reserved_capacity: u64,
    eligible_reserved_property_bytes: u64,
    eligible_length_histogram: [LengthPopulation; 8],
}

impl EntryVecEligibility {
    fn add_assign(&mut self, other: &Self) {
        self.vecs += other.vecs;
        self.empty_vecs += other.empty_vecs;
        self.empty_reserved_capacity += other.empty_reserved_capacity;
        self.empty_reserved_property_bytes += other.empty_reserved_property_bytes;
        self.nonempty_vecs += other.nonempty_vecs;
        self.nonempty_all_default_wec_data_vecs += other.nonempty_all_default_wec_data_vecs;
        self.eligible_live_length += other.eligible_live_length;
        self.eligible_reserved_capacity += other.eligible_reserved_capacity;
        self.eligible_reserved_property_bytes += other.eligible_reserved_property_bytes;
        for (dst, src) in self
            .eligible_length_histogram
            .iter_mut()
            .zip(other.eligible_length_histogram)
        {
            dst.add_assign(&src);
        }
    }

    fn json(&self) -> String {
        let histogram = length_population_json(
            &self.eligible_length_histogram,
            &ELIGIBLE_ENTRY_LENGTH_BOUNDS,
        );
        format!(
            "{{\"vecs\":{},\"empty_vecs\":{},\"empty_reserved_capacity\":{},\"empty_reserved_property_bytes\":{},\"nonempty_vecs\":{},\"nonempty_all_default_wec_data_vecs\":{},\"eligible_live_length\":{},\"eligible_reserved_capacity\":{},\"eligible_reserved_property_bytes\":{},\"eligible_length_histogram\":{histogram},\"eligibility_rule\":\"nonempty; every initialized entry must be non-hole default WEC data\"}}",
            self.vecs,
            self.empty_vecs,
            self.empty_reserved_capacity,
            self.empty_reserved_property_bytes,
            self.nonempty_vecs,
            self.nonempty_all_default_wec_data_vecs,
            self.eligible_live_length,
            self.eligible_reserved_capacity,
            self.eligible_reserved_property_bytes,
        )
    }
}

/// Dense populations can represent holes as the existing PACK_EMPTY value, so holes remain
/// eligible but are counted explicitly. Heap Vec capacities and inline reserved storage are
/// reported separately; eligible capacities are subsets of the corresponding physical totals.
#[derive(Clone, Copy, Default)]
struct DenseEligibility {
    inline_populations: u64,
    inline_all_default_data_or_holes: u64,
    inline_eligible_live_length: u64,
    inline_eligible_reserved_capacity: u64,
    inline_eligible_reserved_property_bytes: u64,
    inline_eligible_holes: u64,
    heap_vectors: u64,
    heap_empty_vectors: u64,
    heap_empty_reserved_capacity: u64,
    heap_empty_reserved_property_bytes: u64,
    heap_all_default_data_or_holes: u64,
    heap_eligible_live_length: u64,
    heap_eligible_reserved_capacity: u64,
    heap_eligible_reserved_property_bytes: u64,
    heap_eligible_holes: u64,
    inline_sidecar_active_length_histogram: [LengthPopulation; 7],
}

impl DenseEligibility {
    fn add_assign(&mut self, other: &Self) {
        self.inline_populations += other.inline_populations;
        self.inline_all_default_data_or_holes += other.inline_all_default_data_or_holes;
        self.inline_eligible_live_length += other.inline_eligible_live_length;
        self.inline_eligible_reserved_capacity += other.inline_eligible_reserved_capacity;
        self.inline_eligible_reserved_property_bytes +=
            other.inline_eligible_reserved_property_bytes;
        self.inline_eligible_holes += other.inline_eligible_holes;
        self.heap_vectors += other.heap_vectors;
        self.heap_empty_vectors += other.heap_empty_vectors;
        self.heap_empty_reserved_capacity += other.heap_empty_reserved_capacity;
        self.heap_empty_reserved_property_bytes += other.heap_empty_reserved_property_bytes;
        self.heap_all_default_data_or_holes += other.heap_all_default_data_or_holes;
        self.heap_eligible_live_length += other.heap_eligible_live_length;
        self.heap_eligible_reserved_capacity += other.heap_eligible_reserved_capacity;
        self.heap_eligible_reserved_property_bytes += other.heap_eligible_reserved_property_bytes;
        self.heap_eligible_holes += other.heap_eligible_holes;
        for (dst, src) in self
            .inline_sidecar_active_length_histogram
            .iter_mut()
            .zip(other.inline_sidecar_active_length_histogram)
        {
            dst.add_assign(&src);
        }
    }

    fn json(&self) -> String {
        let inline_histogram = length_population_json(
            &self.inline_sidecar_active_length_histogram,
            &INLINE_DENSE_LENGTH_BOUNDS,
        );
        format!(
            "{{\"inline_populations\":{},\"inline_all_default_data_or_holes\":{},\"inline_eligible_live_length\":{},\"inline_eligible_reserved_capacity\":{},\"inline_eligible_reserved_property_bytes\":{},\"inline_eligible_holes\":{},\"inline_sidecar_active_length_histogram\":{inline_histogram},\"heap_vectors\":{},\"heap_empty_vectors\":{},\"heap_empty_reserved_capacity\":{},\"heap_empty_reserved_property_bytes\":{},\"heap_all_default_data_or_holes\":{},\"heap_eligible_live_length\":{},\"heap_eligible_reserved_capacity\":{},\"heap_eligible_reserved_property_bytes\":{},\"heap_eligible_holes\":{},\"eligibility_rule\":\"nonempty; every initialized slot is default WEC data or PACK_EMPTY hole\"}}",
            self.inline_populations,
            self.inline_all_default_data_or_holes,
            self.inline_eligible_live_length,
            self.inline_eligible_reserved_capacity,
            self.inline_eligible_reserved_property_bytes,
            self.inline_eligible_holes,
            self.heap_vectors,
            self.heap_empty_vectors,
            self.heap_empty_reserved_capacity,
            self.heap_empty_reserved_property_bytes,
            self.heap_all_default_data_or_holes,
            self.heap_eligible_live_length,
            self.heap_eligible_reserved_capacity,
            self.heap_eligible_reserved_property_bytes,
            self.heap_eligible_holes,
        )
    }
}

/// A live post-collection census of the storage that currently backs one or more objects.
/// `entries_capacity` is shared by named keys and classic indexed entries, so the two
/// populations are counted separately while that physical Vec capacity is reported once.
#[derive(Clone, Copy, Default)]
struct PropertyInventory {
    live_objects: u64,
    entries_len: u64,
    entries_capacity: u64,
    entries_unused_capacity: u64,
    entries_capacity_bytes: u64,
    named_entries: FieldInventory,
    classic_index_entries: FieldInventory,
    dense_sidecars: u64,
    dense_sidecar_body_bytes_excluding_inline_property_storage: u64,
    dense_inline_reserved_property_slots: u64,
    dense_inline_reserved_unused_property_slots: u64,
    dense_inline_reserved_property_bytes: u64,
    dense_inline_packed_len: u64,
    dense_heap_packed_len: u64,
    dense_heap_packed_vectors: u64,
    dense_heap_packed_vector_header_bytes: u64,
    dense_heap_packed_capacity: u64,
    dense_heap_packed_unused_capacity: u64,
    dense_heap_packed_capacity_bytes: u64,
    packed_dense_fields: FieldInventory,
    dense_index_directory_len: u64,
    dense_index_directory_capacity: u64,
    dense_index_directory_capacity_bytes: u64,
    dense_numeric_mirror_len: u64,
    dense_numeric_mirror_capacity: u64,
    dense_numeric_mirror_capacity_bytes: u64,
    dense_string_index_capacity: u64,
    dense_symbol_sidecar_capacity: u64,
    entry_vec_eligibility: EntryVecEligibility,
    dense_eligibility: DenseEligibility,
}

impl PropertyInventory {
    fn from_props(props: &Props) -> Self {
        let mut result = Self {
            live_objects: 1,
            entries_len: props.entries.len() as u64,
            entries_capacity: props.entries.capacity() as u64,
            entries_unused_capacity: props.entries.capacity().saturating_sub(props.entries.len())
                as u64,
            entries_capacity_bytes: props
                .entries
                .capacity()
                .saturating_mul(std::mem::size_of::<Property>())
                as u64,
            ..Self::default()
        };
        result.entry_vec_eligibility.vecs = 1;
        if props.entries.is_empty() {
            result.entry_vec_eligibility.empty_vecs = 1;
            result.entry_vec_eligibility.empty_reserved_capacity = props.entries.capacity() as u64;
            result.entry_vec_eligibility.empty_reserved_property_bytes = props
                .entries
                .capacity()
                .saturating_mul(std::mem::size_of::<Property>())
                as u64;
        } else {
            result.entry_vec_eligibility.nonempty_vecs = 1;
        }
        let mut entry_vec_all_default_wec_data = !props.entries.is_empty();
        for (key, property) in props.entries.iter() {
            entry_vec_all_default_wec_data &= !property.is_empty()
                && !property.accessor()
                && property.writable()
                && property.enumerable()
                && property.configurable();
            // This dimension describes canonical-index key spelling, not packed storage:
            // ordinary objects also keep such keys in `entries` and the index sidecar.
            if super::canonical_index(key).is_some() {
                result.classic_index_entries.add_property(property);
            } else {
                result.named_entries.add_property(property);
            }
        }
        if entry_vec_all_default_wec_data {
            let eligibility = &mut result.entry_vec_eligibility;
            eligibility.nonempty_all_default_wec_data_vecs = 1;
            eligibility.eligible_live_length = props.entries.len() as u64;
            eligibility.eligible_reserved_capacity = props.entries.capacity() as u64;
            eligibility.eligible_reserved_property_bytes = props
                .entries
                .capacity()
                .saturating_mul(std::mem::size_of::<Property>())
                as u64;
            let bucket = length_bucket_index(props.entries.len(), &ELIGIBLE_ENTRY_LENGTH_BOUNDS);
            eligibility.eligible_length_histogram[bucket]
                .observe(props.entries.len(), props.entries.capacity());
        }

        if let Some(dense) = props.elems.0.as_deref() {
            result.dense_sidecars = 1;
            // Include every retained DenseBuffers allocation. A zero-length bin means its
            // embedded ten-slot buffer is reserved but empty; heap-spilled packed vectors also
            // have an empty inline prefix after ownership moves to the heap Vec.
            let inline_len = dense.inline_packed.len as usize;
            let bucket = length_bucket_index(inline_len, &INLINE_DENSE_LENGTH_BOUNDS);
            result
                .dense_eligibility
                .inline_sidecar_active_length_histogram[bucket]
                .observe(inline_len, INLINE_PACKED_CAPACITY);
            let inline_property_bytes =
                INLINE_PACKED_CAPACITY.saturating_mul(std::mem::size_of::<Property>());
            result.dense_sidecar_body_bytes_excluding_inline_property_storage =
                std::mem::size_of::<super::DenseBuffers>().saturating_sub(inline_property_bytes)
                    as u64;
            result.dense_inline_reserved_property_slots = INLINE_PACKED_CAPACITY as u64;
            result.dense_inline_reserved_unused_property_slots =
                INLINE_PACKED_CAPACITY.saturating_sub(dense.inline_packed.len as usize) as u64;
            result.dense_inline_reserved_property_bytes = inline_property_bytes as u64;
            result.dense_index_directory_len = dense.elems.len() as u64;
            result.dense_index_directory_capacity = dense.elems.capacity() as u64;
            result.dense_index_directory_capacity_bytes = dense
                .elems
                .capacity()
                .saturating_mul(std::mem::size_of::<u32>())
                as u64;
            result.dense_numeric_mirror_len = dense.mirror.len() as u64;
            result.dense_numeric_mirror_capacity = dense.mirror.capacity() as u64;
            result.dense_numeric_mirror_capacity_bytes = dense
                .mirror
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>())
                as u64;
            result.dense_string_index_capacity = dense
                .index
                .as_deref()
                .map_or(0, |index| index.capacity() as u64);
            result.dense_symbol_sidecar_capacity = dense
                .symbols
                .as_deref()
                .map_or(0, |symbols| symbols.capacity() as u64);

            if let Some(packed) = dense.packed.as_deref() {
                result.dense_heap_packed_len = packed.len() as u64;
                result.dense_eligibility.heap_vectors = 1;
                result.dense_heap_packed_vectors = 1;
                result.dense_heap_packed_vector_header_bytes =
                    std::mem::size_of::<Vec<Property>>() as u64;
                result.dense_heap_packed_capacity = packed.capacity() as u64;
                result.dense_heap_packed_unused_capacity =
                    packed.capacity().saturating_sub(packed.len()) as u64;
                result.dense_heap_packed_capacity_bytes = packed
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Property>())
                    as u64;
                for property in packed {
                    result.packed_dense_fields.add_property(property);
                }
                if packed.is_empty() {
                    result.dense_eligibility.heap_empty_vectors = 1;
                    result.dense_eligibility.heap_empty_reserved_capacity =
                        packed.capacity() as u64;
                    result.dense_eligibility.heap_empty_reserved_property_bytes = packed
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Property>())
                        as u64;
                } else if result.packed_dense_fields.all_default_data_or_holes() {
                    result.dense_eligibility.heap_all_default_data_or_holes = 1;
                    result.dense_eligibility.heap_eligible_live_length = packed.len() as u64;
                    result.dense_eligibility.heap_eligible_reserved_capacity =
                        packed.capacity() as u64;
                    result
                        .dense_eligibility
                        .heap_eligible_reserved_property_bytes = packed
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Property>())
                        as u64;
                    result.dense_eligibility.heap_eligible_holes = result.packed_dense_fields.holes;
                }
            } else {
                let packed = dense.inline_packed.as_slice();
                result.dense_inline_packed_len = packed.len() as u64;
                for property in packed {
                    result.packed_dense_fields.add_property(property);
                }
                if !packed.is_empty() {
                    result.dense_eligibility.inline_populations = 1;
                    if result.packed_dense_fields.all_default_data_or_holes() {
                        result.dense_eligibility.inline_all_default_data_or_holes = 1;
                        result.dense_eligibility.inline_eligible_live_length = packed.len() as u64;
                        result.dense_eligibility.inline_eligible_reserved_capacity =
                            INLINE_PACKED_CAPACITY as u64;
                        result
                            .dense_eligibility
                            .inline_eligible_reserved_property_bytes = INLINE_PACKED_CAPACITY
                            .saturating_mul(std::mem::size_of::<Property>())
                            as u64;
                        result.dense_eligibility.inline_eligible_holes =
                            result.packed_dense_fields.holes;
                    }
                }
            }
        }
        result
    }

    fn add_assign(&mut self, other: &Self) {
        self.live_objects += other.live_objects;
        self.entries_len += other.entries_len;
        self.entries_capacity += other.entries_capacity;
        self.entries_unused_capacity += other.entries_unused_capacity;
        self.entries_capacity_bytes += other.entries_capacity_bytes;
        self.named_entries.add_assign(&other.named_entries);
        self.classic_index_entries
            .add_assign(&other.classic_index_entries);
        self.dense_sidecars += other.dense_sidecars;
        self.dense_sidecar_body_bytes_excluding_inline_property_storage +=
            other.dense_sidecar_body_bytes_excluding_inline_property_storage;
        self.dense_inline_reserved_property_slots += other.dense_inline_reserved_property_slots;
        self.dense_inline_reserved_unused_property_slots +=
            other.dense_inline_reserved_unused_property_slots;
        self.dense_inline_reserved_property_bytes += other.dense_inline_reserved_property_bytes;
        self.dense_inline_packed_len += other.dense_inline_packed_len;
        self.dense_heap_packed_len += other.dense_heap_packed_len;
        self.dense_heap_packed_vectors += other.dense_heap_packed_vectors;
        self.dense_heap_packed_vector_header_bytes += other.dense_heap_packed_vector_header_bytes;
        self.dense_heap_packed_capacity += other.dense_heap_packed_capacity;
        self.dense_heap_packed_unused_capacity += other.dense_heap_packed_unused_capacity;
        self.dense_heap_packed_capacity_bytes += other.dense_heap_packed_capacity_bytes;
        self.packed_dense_fields
            .add_assign(&other.packed_dense_fields);
        self.dense_index_directory_len += other.dense_index_directory_len;
        self.dense_index_directory_capacity += other.dense_index_directory_capacity;
        self.dense_index_directory_capacity_bytes += other.dense_index_directory_capacity_bytes;
        self.dense_numeric_mirror_len += other.dense_numeric_mirror_len;
        self.dense_numeric_mirror_capacity += other.dense_numeric_mirror_capacity;
        self.dense_numeric_mirror_capacity_bytes += other.dense_numeric_mirror_capacity_bytes;
        self.dense_string_index_capacity += other.dense_string_index_capacity;
        self.dense_symbol_sidecar_capacity += other.dense_symbol_sidecar_capacity;
        self.entry_vec_eligibility
            .add_assign(&other.entry_vec_eligibility);
        self.dense_eligibility.add_assign(&other.dense_eligibility);
    }

    fn json(&self) -> String {
        format!(
            "{{\"live_objects\":{},\"entries_len\":{},\"entries_capacity\":{},\"entries_unused_capacity\":{},\"entries_capacity_bytes\":{},\"named_entries\":{},\"classic_index_entries\":{},\"entry_vec_eligibility\":{},\"dense_sidecars\":{},\"dense_sidecar_body_bytes_excluding_inline_property_storage\":{},\"dense_inline_reserved_property_slots\":{},\"dense_inline_reserved_unused_property_slots\":{},\"dense_inline_reserved_property_bytes\":{},\"dense_inline_packed_len\":{},\"dense_heap_packed_len\":{},\"dense_heap_packed_vectors\":{},\"dense_heap_packed_vector_header_bytes\":{},\"dense_heap_packed_capacity\":{},\"dense_heap_packed_unused_capacity\":{},\"dense_heap_packed_capacity_bytes\":{},\"packed_dense_fields\":{},\"dense_eligibility\":{},\"dense_index_directory_len\":{},\"dense_index_directory_capacity\":{},\"dense_index_directory_capacity_bytes\":{},\"dense_numeric_mirror_len\":{},\"dense_numeric_mirror_capacity\":{},\"dense_numeric_mirror_capacity_bytes\":{},\"dense_string_index_capacity\":{},\"dense_symbol_sidecar_capacity\":{}}}",
            self.live_objects,
            self.entries_len,
            self.entries_capacity,
            self.entries_unused_capacity,
            self.entries_capacity_bytes,
            self.named_entries.json(),
            self.classic_index_entries.json(),
            self.entry_vec_eligibility.json(),
            self.dense_sidecars,
            self.dense_sidecar_body_bytes_excluding_inline_property_storage,
            self.dense_inline_reserved_property_slots,
            self.dense_inline_reserved_unused_property_slots,
            self.dense_inline_reserved_property_bytes,
            self.dense_inline_packed_len,
            self.dense_heap_packed_len,
            self.dense_heap_packed_vectors,
            self.dense_heap_packed_vector_header_bytes,
            self.dense_heap_packed_capacity,
            self.dense_heap_packed_unused_capacity,
            self.dense_heap_packed_capacity_bytes,
            self.packed_dense_fields.json(),
            self.dense_eligibility.json(),
            self.dense_index_directory_len,
            self.dense_index_directory_capacity,
            self.dense_index_directory_capacity_bytes,
            self.dense_numeric_mirror_len,
            self.dense_numeric_mirror_capacity,
            self.dense_numeric_mirror_capacity_bytes,
            self.dense_string_index_capacity,
            self.dense_symbol_sidecar_capacity,
        )
    }
}

struct LivePropertyInventory {
    scanned_objects: u64,
    unavailable_borrows: u64,
    untracked_birth_sites: u64,
    total: PropertyInventory,
    families: [PropertyInventory; 7],
    sites: Vec<PropertyInventory>,
}

impl LivePropertyInventory {
    fn new(site_count: usize) -> Self {
        Self {
            scanned_objects: 0,
            unavailable_borrows: 0,
            untracked_birth_sites: 0,
            total: PropertyInventory::default(),
            families: [PropertyInventory::default(); 7],
            // Bounded by MAX_SITES; this stores counters only, never object or JS owners.
            sites: vec![PropertyInventory::default(); site_count],
        }
    }

    fn json(&self, state: &State, quoted: impl Fn(&str) -> String) -> String {
        let families = FAMILIES
            .iter()
            .zip(self.families)
            .map(|(name, inventory)| format!("\"{name}\":{}", inventory.json()))
            .collect::<Vec<_>>()
            .join(",");
        let sites = state
            .sites
            .iter()
            .filter_map(|((file, line, initial_family), index)| {
                let inventory = self.sites.get(*index)?;
                (inventory.live_objects != 0).then(|| {
                    format!(
                        "{{\"file\":{},\"line\":{line},\"initial_family\":{},\"inventory\":{}}}",
                        quoted(file),
                        quoted(initial_family),
                        inventory.json(),
                    )
                })
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"scope\":\"post_gc_live_sample\",\"descriptor_encoding\":{{\"data_wec_mask_bits\":\"writable=1,enumerable=2,configurable=4\",\"accessor_attribute_mask_bits\":\"enumerable=1,configurable=2\",\"accessor_endpoint_mask_bits\":\"getter=1,setter=2\"}},\"scanned_objects\":{},\"unavailable_borrows\":{},\"untracked_birth_sites\":{},\"total\":{},\"families\":{{{families}}},\"sites\":[{sites}]}}",
            self.scanned_objects,
            self.unavailable_borrows,
            self.untracked_birth_sites,
            self.total.json(),
        )
    }
}

impl State {
    pub(super) fn born(
        &mut self,
        slot: usize,
        caller: &'static Location<'static>,
        exotic: &Exotic,
        capacity: usize,
    ) {
        self.ordinal = self.ordinal.wrapping_add(1);
        let key = (caller.file(), caller.line(), initial_family(exotic));
        let index = self.sites.get(&key).copied().or_else(|| {
            if self.sites.len() == MAX_SITES {
                return None;
            }
            let index = self.counts.len();
            self.sites.insert(key, index);
            self.counts.push(Site::default());
            Some(index)
        });
        let Some(site) = index.filter(|_| slot < MAX_SLOTS) else {
            self.untracked_allocations += 1;
            return;
        };
        self.counts[site].allocations += 1;
        self.counts[site].requested_named_capacity += capacity as u64;
        self.births.resize(self.births.len().max(slot + 1), None);
        assert!(
            self.births[slot].is_none(),
            "registry slot is not already alive"
        );
        self.births[slot] = Some(Birth {
            site,
            ordinal: self.ordinal,
            generation: self.generation,
        });
    }

    pub(super) fn died(&mut self, slot: usize) {
        let Some(birth) = self.births.get_mut(slot).and_then(Option::take) else {
            return;
        };
        let age = self.ordinal.wrapping_sub(birth.ordinal);
        let bucket = [0, 1, 16, 256, 4096, 65536, 1_048_576].partition_point(|limit| age > *limit);
        let site = &mut self.counts[birth.site];
        site.deaths += 1;
        site.deaths_before_collection += u64::from(birth.generation == self.generation);
        site.lifetime_allocations[bucket] += 1;
    }

    pub(super) fn finish_generation(&mut self) {
        self.generation += 1;
    }

    fn json(
        &self,
        heap_id: u64,
        live: [u64; 7],
        borrowed: u64,
        inventory: &LivePropertyInventory,
    ) -> String {
        // Compiler-provided Rust locations contain no user data. Escape file paths for JSON
        // using the same small explicit routine on all hosts (Windows paths include '\\').
        fn quoted(value: &str) -> String {
            let mut output = String::from("\"");
            for ch in value.chars() {
                match ch {
                    '"' => output.push_str("\\\""),
                    '\\' => output.push_str("\\\\"),
                    ch if ch < ' ' => {
                        use std::fmt::Write;
                        let _ = write!(output, "\\u{:04x}", ch as u32);
                    }
                    ch => output.push(ch),
                }
            }
            output.push('"');
            output
        }
        let sites = self.sites.iter().map(|((file,line,family),index)| {
            let site=&self.counts[*index];
            format!("{{\"file\":{},\"line\":{line},\"initial_family\":{},\"allocations\":{},\"deaths\":{},\"deaths_before_completed_collection\":{},\"requested_named_capacity\":{},\"lifetime_allocations\":{:?}}}",
                quoted(file), quoted(family), site.allocations, site.deaths, site.deaths_before_collection, site.requested_named_capacity, site.lifetime_allocations)
        }).collect::<Vec<_>>().join(",");
        let live = FAMILIES
            .iter()
            .zip(live)
            .map(|(name, count)| format!("\"{name}\":{count}"))
            .collect::<Vec<_>>()
            .join(",");
        let inventory = inventory.json(self, quoted);
        format!("{{\"heap\":{heap_id},\"allocations\":{},\"collections\":{},\"untracked_allocations\":{},\"borrowed_live_objects\":{borrowed},\"live_families\":{{{live}}},\"lifetime_bucket_upper_allocations\":[0,1,16,256,4096,65536,1048576,null],\"sites\":[{sites}],\"post_gc_live_property_inventory\":{inventory}}}",self.ordinal,self.generation,self.untracked_allocations)
    }
}

pub(super) fn report(heap: &GcHeap) {
    let state = heap.diagnostics.borrow();
    let mut live = [0; 7];
    let mut borrowed = 0;
    let mut inventory = LivePropertyInventory::new(state.counts.len());
    let registry = heap.registry.borrow();
    for (slot, &object) in registry
        .entries
        .iter()
        .enumerate()
        .filter(|(_, object)| !object.is_null())
    {
        // SAFETY: a non-null registry entry names a live object; `Object::drop` clears it first.
        let object = unsafe { &*object };
        let Ok(object) = object.try_borrow() else {
            borrowed += 1;
            inventory.unavailable_borrows += 1;
            continue;
        };
        let family = if !object.ic_plain.get() {
            6
        } else if !matches!(object.call, Callable::None) {
            1
        } else {
            match object.exotic {
                Exotic::None => 0,
                Exotic::Array => 2,
                Exotic::Arguments => 3,
                Exotic::Error(_) => 4,
                _ => 5,
            }
        };
        live[family] += 1;
        let current = PropertyInventory::from_props(&object.props);
        inventory.scanned_objects += 1;
        inventory.total.add_assign(&current);
        inventory.families[family].add_assign(&current);
        if let Some(site) = state
            .births
            .get(slot)
            .and_then(Option::as_ref)
            .map(|birth| birth.site)
            .filter(|site| *site < inventory.sites.len())
        {
            inventory.sites[site].add_assign(&current);
        } else {
            inventory.untracked_birth_sites += 1;
        }
    }
    eprintln!(
        "[allocation-diagnostic] {}",
        state.json(heap.heap_id, live, borrowed, &inventory)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_reuse_and_collection_age_do_not_retain_objects() {
        let mut state = State::default();
        state.born(0, Location::caller(), &Exotic::None, 4);
        state.died(0);
        state.born(0, Location::caller(), &Exotic::Array, 0);
        state.finish_generation();
        state.died(0);
        assert_eq!(state.counts.iter().map(|s| s.allocations).sum::<u64>(), 2);
        assert_eq!(state.counts.iter().map(|s| s.deaths).sum::<u64>(), 2);
        assert_eq!(
            state
                .counts
                .iter()
                .map(|s| s.deaths_before_collection)
                .sum::<u64>(),
            1
        );
        assert!(state.births.iter().all(Option::is_none));
        state.born(MAX_SLOTS, Location::caller(), &Exotic::None, 0);
        assert_eq!(state.untracked_allocations, 1);
    }

    #[test]
    fn inventory_length_bins_cover_entry_and_dense_inline_boundaries() {
        for (len, expected_entry, expected_dense) in [
            (0, None, Some(0)),
            (1, Some(0), Some(1)),
            (2, Some(1), Some(2)),
            (3, Some(2), Some(3)),
            (4, Some(3), Some(4)),
            (5, Some(4), Some(5)),
            (8, Some(4), Some(5)),
            (9, Some(5), Some(6)),
            (10, Some(5), Some(6)),
            (16, Some(5), None),
            (17, Some(6), None),
            (32, Some(6), None),
            (33, Some(7), None),
        ] {
            assert_eq!(
                expected_entry,
                (len != 0).then(|| length_bucket_index(len, &ELIGIBLE_ENTRY_LENGTH_BOUNDS)),
                "entry vector length {len}"
            );
            assert_eq!(
                expected_dense,
                (len <= INLINE_PACKED_CAPACITY)
                    .then(|| length_bucket_index(len, &INLINE_DENSE_LENGTH_BOUNDS)),
                "dense inline length {len}"
            );
        }
    }

    #[test]
    fn inventory_separates_named_dense_descriptors_holes_and_shared_capacity() {
        let mut props = Props::new();
        props.insert("plain", Property::plain(super::super::Value::Num(1.0)));
        props.insert(
            "hidden",
            Property::data(super::super::Value::Num(2.0), true, false, true),
        );
        props.insert(
            "accessor",
            Property::accessor_prop(Some(super::super::Value::Undefined), None, true, false),
        );

        let mut array = Props::packed_array_properties(
            vec![
                Property::plain(super::super::Value::Num(3.0)),
                Property::plain(super::super::Value::Empty),
                Property::data(super::super::Value::Num(4.0), false, true, true),
                Property::accessor_prop(None, Some(super::super::Value::Undefined), false, true),
            ]
            .into_iter(),
        );
        array.insert("label", Property::plain(super::super::Value::Num(5.0)));

        let named = PropertyInventory::from_props(&props);
        assert_eq!(named.live_objects, 1);
        assert_eq!(named.named_entries.slots, 3);
        assert_eq!(named.named_entries.default_wec_data, 1);
        assert_eq!(named.named_entries.exceptional_data_masks[0b101], 1);
        assert_eq!(named.named_entries.accessors, 1);
        assert_eq!(
            named.named_entries.accessor_enumerable_configurable[0b01],
            1
        );
        assert_eq!(named.named_entries.accessor_getter_setter[0b01], 1);
        assert_eq!(named.classic_index_entries.slots, 0);
        assert_eq!(named.entry_vec_eligibility.nonempty_vecs, 1);
        assert_eq!(
            named
                .entry_vec_eligibility
                .nonempty_all_default_wec_data_vecs,
            0,
            "a hole, accessor, or nondefault descriptor disqualifies the entry Vec"
        );

        let mut plain_record = Props::new();
        plain_record.insert("x", Property::plain(super::super::Value::Num(7.0)));
        plain_record.insert("y", Property::plain(super::super::Value::Num(8.0)));
        let plain_record = PropertyInventory::from_props(&plain_record);
        assert_eq!(
            plain_record
                .entry_vec_eligibility
                .nonempty_all_default_wec_data_vecs,
            1
        );
        assert_eq!(plain_record.entry_vec_eligibility.eligible_live_length, 2);
        assert!(
            plain_record
                .entry_vec_eligibility
                .eligible_reserved_capacity
                >= 2
        );
        assert_eq!(
            plain_record
                .entry_vec_eligibility
                .eligible_reserved_property_bytes,
            plain_record
                .entry_vec_eligibility
                .eligible_reserved_capacity
                * std::mem::size_of::<Property>() as u64
        );
        assert_eq!(
            plain_record.entries_unused_capacity,
            plain_record.entries_capacity - plain_record.entries_len
        );
        let two_entry_bucket = plain_record.entry_vec_eligibility.eligible_length_histogram[1];
        assert_eq!(two_entry_bucket.populations, 1);
        assert_eq!(two_entry_bucket.live_length, 2);
        assert_eq!(
            two_entry_bucket.reserved_capacity,
            plain_record
                .entry_vec_eligibility
                .eligible_reserved_capacity
        );

        let empty_reserve = PropertyInventory::from_props(&Props::with_capacity(5));
        assert_eq!(empty_reserve.entry_vec_eligibility.empty_vecs, 1);
        assert_eq!(
            empty_reserve
                .entry_vec_eligibility
                .nonempty_all_default_wec_data_vecs,
            0,
            "empty maps are reported separately instead of vacuous eligibility"
        );
        assert!(empty_reserve.entry_vec_eligibility.empty_reserved_capacity >= 5);

        let mut empty_inline = Props::new();
        // Ordinary canonical indices populate the index sidecar but stay in `entries`;
        // this leaves the DenseBuffers inline-property prefix at length zero.
        empty_inline.insert("0", Property::plain(super::super::Value::Num(6.0)));
        let empty_inline = PropertyInventory::from_props(&empty_inline);
        assert_eq!(empty_inline.dense_sidecars, 1);
        assert_eq!(empty_inline.dense_inline_packed_len, 0);
        let empty_inline_bucket = empty_inline
            .dense_eligibility
            .inline_sidecar_active_length_histogram[0];
        assert_eq!(empty_inline_bucket.populations, 1);
        assert_eq!(empty_inline_bucket.live_length, 0);
        assert_eq!(
            empty_inline_bucket.reserved_capacity,
            INLINE_PACKED_CAPACITY as u64
        );

        let dense = PropertyInventory::from_props(&array);
        assert_eq!(
            dense.entries_len, 2,
            "array length and label share entries Vec"
        );
        assert_eq!(dense.named_entries.default_wec_data, 1);
        assert_eq!(dense.named_entries.exceptional_data_masks[0b001], 1);
        assert_eq!(dense.entry_vec_eligibility.nonempty_vecs, 1);
        assert_eq!(
            dense
                .entry_vec_eligibility
                .nonempty_all_default_wec_data_vecs,
            0,
            "the nondefault Array length stays in entries, separate from packed indices"
        );
        assert_eq!(dense.packed_dense_fields.slots, 4);
        assert_eq!(dense.packed_dense_fields.default_wec_data, 1);
        assert_eq!(dense.packed_dense_fields.exceptional_data_masks[0b110], 1);
        assert_eq!(dense.packed_dense_fields.accessors, 1);
        assert_eq!(
            dense.packed_dense_fields.accessor_enumerable_configurable[0b10],
            1
        );
        assert_eq!(dense.packed_dense_fields.accessor_getter_setter[0b10], 1);
        assert_eq!(dense.packed_dense_fields.holes, 1);
        assert_eq!(dense.dense_sidecars, 1);
        assert_eq!(
            dense.dense_inline_reserved_property_slots,
            INLINE_PACKED_CAPACITY as u64
        );
        assert_eq!(dense.dense_inline_packed_len, 4);
        assert_eq!(dense.dense_heap_packed_capacity, 0);
        assert_eq!(dense.dense_eligibility.inline_populations, 1);
        let four_slot_inline_bucket = dense
            .dense_eligibility
            .inline_sidecar_active_length_histogram[4];
        assert_eq!(four_slot_inline_bucket.populations, 1);
        assert_eq!(four_slot_inline_bucket.live_length, 4);
        assert_eq!(
            four_slot_inline_bucket.reserved_capacity,
            INLINE_PACKED_CAPACITY as u64
        );
        assert_eq!(dense.dense_eligibility.inline_all_default_data_or_holes, 0);
        assert_eq!(
            dense.dense_inline_reserved_unused_property_slots,
            dense.dense_inline_reserved_property_slots - dense.dense_inline_packed_len
        );

        let compact_dense = Props::packed_array_properties(
            vec![
                Property::plain(super::super::Value::Num(1.0)),
                Property::plain(super::super::Value::Empty),
                Property::plain(super::super::Value::Num(3.0)),
            ]
            .into_iter(),
        );
        let compact_dense = PropertyInventory::from_props(&compact_dense);
        assert_eq!(
            compact_dense
                .dense_eligibility
                .inline_all_default_data_or_holes,
            1
        );
        assert_eq!(
            compact_dense.dense_eligibility.inline_eligible_live_length,
            3
        );
        assert_eq!(compact_dense.dense_eligibility.inline_eligible_holes, 1);
        assert_eq!(
            compact_dense
                .dense_eligibility
                .inline_eligible_reserved_capacity,
            INLINE_PACKED_CAPACITY as u64
        );

        let compact_heap_dense = Props::packed_array_properties(
            (0..12).map(|index| Property::plain(super::super::Value::Num(index as f64))),
        );
        let compact_heap_dense = PropertyInventory::from_props(&compact_heap_dense);
        assert_eq!(compact_heap_dense.dense_eligibility.heap_vectors, 1);
        assert_eq!(
            compact_heap_dense
                .dense_eligibility
                .heap_all_default_data_or_holes,
            1
        );
        assert_eq!(
            compact_heap_dense
                .dense_eligibility
                .heap_eligible_live_length,
            12
        );
        assert!(
            compact_heap_dense
                .dense_eligibility
                .heap_eligible_reserved_capacity
                >= 12
        );
        assert_eq!(
            compact_heap_dense.dense_heap_packed_unused_capacity,
            compact_heap_dense.dense_heap_packed_capacity
                - compact_heap_dense.dense_heap_packed_len
        );
        assert_eq!(
            compact_heap_dense
                .dense_eligibility
                .heap_eligible_reserved_property_bytes,
            compact_heap_dense
                .dense_eligibility
                .heap_eligible_reserved_capacity
                * std::mem::size_of::<Property>() as u64
        );

        let mut ordinary_numeric_name = Props::new();
        ordinary_numeric_name.insert("0", Property::plain(super::super::Value::Num(6.0)));
        let ordinary_numeric_name = PropertyInventory::from_props(&ordinary_numeric_name);
        assert_eq!(ordinary_numeric_name.named_entries.slots, 0);
        assert_eq!(ordinary_numeric_name.classic_index_entries.slots, 1);

        let mut classic = Props::new();
        classic.mark_array();
        // A first write beyond the bounded dense frontier stays in the entries Vec and
        // records a map-only canonical index, the classic sparse-array representation.
        classic.insert("1000", Property::plain(super::super::Value::Num(6.0)));
        let classic = PropertyInventory::from_props(&classic);
        assert_eq!(classic.entries_len, 1);
        assert_eq!(classic.named_entries.slots, 0);
        assert_eq!(classic.classic_index_entries.slots, 1);
        assert_eq!(classic.packed_dense_fields.slots, 0);
        assert_eq!(
            classic
                .entry_vec_eligibility
                .nonempty_all_default_wec_data_vecs,
            1
        );
    }

    #[test]
    fn live_inventory_keeps_explicit_total_and_family_histograms() {
        let mut plain = Props::new();
        plain.insert("x", Property::plain(super::super::Value::Num(1.0)));
        let mut second = Props::new();
        second.insert("y", Property::plain(super::super::Value::Num(2.0)));

        let plain_inventory = PropertyInventory::from_props(&plain);
        let second_inventory = PropertyInventory::from_props(&second);
        let mut snapshot = LivePropertyInventory::new(0);
        snapshot.scanned_objects = 2;
        snapshot.total.add_assign(&plain_inventory);
        snapshot.total.add_assign(&second_inventory);
        snapshot.families[0].add_assign(&plain_inventory);
        snapshot.families[2].add_assign(&second_inventory);

        assert_eq!(snapshot.total.live_objects, snapshot.scanned_objects);
        assert_eq!(
            snapshot
                .total
                .entry_vec_eligibility
                .eligible_length_histogram[0]
                .populations,
            2
        );
        assert_eq!(
            snapshot.families[0]
                .entry_vec_eligibility
                .eligible_length_histogram[0]
                .populations,
            1
        );
        assert_eq!(
            snapshot.families[2]
                .entry_vec_eligibility
                .eligible_length_histogram[0]
                .populations,
            1
        );
        let serialized = snapshot.json(&State::default(), |s| format!("\"{s}\""));
        assert!(serialized.contains("\"total\":{\"live_objects\":2"));
        assert!(serialized.contains("eligible_length_histogram"));
    }

    #[test]
    fn inventory_of_lazy_function_prototype_does_not_materialize_it() {
        let mut engine = crate::Engine::new();
        engine.eval("function unobserved(){}", false).unwrap();
        let env = engine.interp.global_env.clone();
        let function = engine.interp.get_var("unobserved", &env).ok().unwrap();
        let object = function.as_obj().unwrap().borrow();
        let prototype = object.props.get("prototype").unwrap();
        assert_eq!(prototype.packed.tag(), PACK_LAZY_PROTO);
        let packed_before = prototype.packed.bits();

        let inventory = PropertyInventory::from_props(&object.props);
        assert_eq!(inventory.named_entries.lazy_prototype_values, 1);
        assert_eq!(prototype.packed.bits(), packed_before);
        assert_eq!(prototype.packed.tag(), PACK_LAZY_PROTO);
    }
}
