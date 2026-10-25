//! Process-wide structural identity and bounded, reconstructible key-layout hints.
//!
//! OrdinaryGet/OrdinarySet must select the requested property even when an embedder
//! shares objects between Engines. Heap-local counters are not identity tokens.
//! ECMA-262 snapshot e28783d5, #sec-ordinaryget and #sec-ordinarysetwithowndescriptor.

use super::PropertyLayout;
use std::sync::atomic::{AtomicU32, Ordering};

/// Exhaustion disables new shape proofs, never wraps or terminates the host. Every
/// cache proof producer rejects this token; a native comparison with a cached
/// cacheable token consequently misses without an extra hot-path guard.
pub(crate) const SHAPE_UNCACHEABLE: u32 = u32::MAX;

static NEXT_SHAPE: AtomicU32 = AtomicU32::new(1);

#[inline]
pub(crate) fn is_cacheable_shape(shape: u32) -> bool {
    shape != SHAPE_UNCACHEABLE
}

fn allocate_from(next: &AtomicU32) -> u32 {
    crate::atomic_compat::try_update_u32(next, Ordering::Relaxed, Ordering::Relaxed, |id| {
        id.checked_add(1)
    })
    .unwrap_or(SHAPE_UNCACHEABLE)
}

pub(super) fn fresh_shape() -> u32 {
    allocate_from(&NEXT_SHAPE)
}

pub(crate) const SHAPE_LAYOUT_PAGE_BITS: u32 = 6;
pub(crate) const SHAPE_LAYOUT_PAGE_SIZE: usize = 1 << SHAPE_LAYOUT_PAGE_BITS;
pub(crate) const SHAPE_LAYOUT_PAGE_COUNT: usize = 64;
pub(crate) const SHAPE_LAYOUT_CACHE_LIMIT: usize = SHAPE_LAYOUT_PAGE_SIZE * SHAPE_LAYOUT_PAGE_COUNT;

/// The exact identity is mandatory: low ID bits only select a cache slot. Eviction
/// releases key-only layout hints, never language values or an object's live keys.
#[repr(C)]
#[derive(Default)]
pub(super) struct LayoutEntry {
    pub(super) shape: u32,
    pub(super) keys: Option<PropertyLayout>,
}

pub(super) type LayoutPage = [LayoutEntry; SHAPE_LAYOUT_PAGE_SIZE];

pub(super) struct ShapeLayouts {
    pub(super) pages: [Option<Box<LayoutPage>>; SHAPE_LAYOUT_PAGE_COUNT],
    /// The shapes whose entries were written since a collection's marking last traced the
    /// table ([`ShapeLayouts::trace_roots`]): a minor collection traces these entries alone.
    /// `overflow` once more were written than the table holds, which makes the next one trace
    /// every entry.
    written: std::cell::Cell<Vec<u32>>,
    overflow: std::cell::Cell<bool>,
}

impl Default for ShapeLayouts {
    fn default() -> Self {
        Self {
            pages: std::array::from_fn(|_| None),
            written: std::cell::Cell::new(Vec::new()),
            overflow: std::cell::Cell::new(false),
        }
    }
}

impl ShapeLayouts {
    fn index(shape: u32) -> (usize, usize) {
        let index = shape as usize & (SHAPE_LAYOUT_CACHE_LIMIT - 1);
        (
            index >> SHAPE_LAYOUT_PAGE_BITS,
            index & (SHAPE_LAYOUT_PAGE_SIZE - 1),
        )
    }

    pub(super) fn get(&self, shape: u32) -> Option<&PropertyLayout> {
        if !is_cacheable_shape(shape) {
            return None;
        }
        let (page, slot) = Self::index(shape);
        let entry = &self.pages[page].as_ref()?[slot];
        (entry.shape == shape)
            .then_some(entry.keys.as_ref())
            .flatten()
    }

    /// The entry of `shape`, for its writer: the entry counts as written.
    pub(super) fn get_mut(&mut self, shape: u32) -> Option<&mut PropertyLayout> {
        if !is_cacheable_shape(shape) {
            return None;
        }
        let (page, slot) = Self::index(shape);
        let entry = &mut self.pages[page].as_mut()?[slot];
        if entry.shape != shape || entry.keys.is_none() {
            return None;
        }
        Self::note_written(&mut self.written, &mut self.overflow, shape);
        entry.keys.as_mut()
    }

    /// Record that the entry of `shape` was written, for the next minor collection.
    fn note_written(
        written: &mut std::cell::Cell<Vec<u32>>,
        overflow: &mut std::cell::Cell<bool>,
        shape: u32,
    ) {
        if *overflow.get_mut() {
            return;
        }
        let written = written.get_mut();
        if written.len() < SHAPE_LAYOUT_CACHE_LIMIT {
            written.push(shape);
        } else {
            *written = Vec::new();
            *overflow.get_mut() = true;
        }
    }

    pub(super) fn insert(&mut self, shape: u32, keys: PropertyLayout) {
        keys.mark_shared();
        if !is_cacheable_shape(shape) {
            return;
        }
        let (page, slot) = Self::index(shape);
        let page = self.pages[page]
            .get_or_insert_with(|| Box::new(std::array::from_fn(|_| LayoutEntry::default())));
        let entry = &mut page[slot];
        *entry = LayoutEntry {
            shape,
            keys: Some(keys),
        };
        Self::note_written(&mut self.written, &mut self.overflow, shape);
    }

    /// Report the layouts a collection of `kind` traces: every entry's for a major collection;
    /// for a minor one, the entries written since a collection's marking last called this (every
    /// entry once more were written than the table holds), the others' layouts being old.
    /// Either way the record of writes starts anew, so only a marking calls this.
    pub(super) fn trace_roots(
        &self,
        kind: crate::gc_core::CollectionKind,
        sink: &mut dyn crate::gc_trace::GcSink,
    ) {
        let mut written = self.written.take();
        if kind == crate::gc_core::CollectionKind::Minor && !self.overflow.replace(false) {
            for &shape in &written {
                if let Some(layout) = self.get(shape) {
                    sink.layout(layout);
                }
            }
        } else {
            self.overflow.set(false);
            crate::gc_trace::GcTrace::trace_gc(self, sink);
        }
        written.clear();
        self.written.set(written);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.iter().count()
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count() * std::mem::size_of::<LayoutPage>()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &PropertyLayout> {
        self.pages
            .iter()
            .filter_map(Option::as_ref)
            .flat_map(|page| page.iter().filter_map(|entry| entry.keys.as_ref()))
    }
}

impl crate::gc_trace::GcTrace for ShapeLayouts {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self {
            pages,
            written: _,
            overflow: _,
        } = self;
        for page in pages.iter().flatten() {
            for entry in page.iter() {
                let LayoutEntry { shape: _, keys } = entry;
                crate::gc_trace::GcTrace::trace_gc(keys, sink);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn minor_collections_trace_the_layouts_written_since_the_last_marking() {
        use crate::gc_core::CollectionKind;
        struct Words(Vec<u64>);
        impl crate::gc_trace::GcSink for Words {
            fn word(&mut self, word: u64) {
                self.0.push(word);
            }
        }
        let traced = |layouts: &ShapeLayouts, kind| {
            let mut words = Words(Vec::new());
            layouts.trace_roots(kind, &mut words);
            words.0.sort_unstable();
            words.0
        };
        // Layouts are cells of the active heap.
        let _engine = crate::Engine::new();
        let layout =
            |name: &str| super::super::new_shared_indexed_property_layout(vec![name.into()]);
        let mut layouts = ShapeLayouts::default();
        let (a, b, c) = (layout("a"), layout("b"), layout("c"));
        layouts.insert(1, a.clone());
        layouts.insert(2, b.clone());
        let mut both = vec![a.cell_word(), b.cell_word()];
        both.sort_unstable();
        assert_eq!(traced(&layouts, CollectionKind::Minor), both);
        assert!(
            traced(&layouts, CollectionKind::Minor).is_empty(),
            "nothing new"
        );
        // A prediction's update through `get_mut` is a write.
        *layouts.get_mut(1).expect("entry") = c.clone();
        assert_eq!(traced(&layouts, CollectionKind::Minor), [c.cell_word()]);
        let mut all = vec![b.cell_word(), c.cell_word()];
        all.sort_unstable();
        assert_eq!(traced(&layouts, CollectionKind::Major), all);
        assert!(traced(&layouts, CollectionKind::Minor).is_empty());
        // More writes than the table holds: the next minor collection traces every entry.
        for _ in 0..=SHAPE_LAYOUT_CACHE_LIMIT {
            *layouts.get_mut(2).expect("entry") = b.clone();
        }
        assert_eq!(traced(&layouts, CollectionKind::Minor), all);
        assert!(traced(&layouts, CollectionKind::Minor).is_empty());
    }

    #[test]
    fn shape_identity_exhaustion_disables_proofs_without_wrapping() {
        let next = AtomicU32::new(u32::MAX - 2);
        assert_eq!(allocate_from(&next), u32::MAX - 2);
        assert_eq!(allocate_from(&next), u32::MAX - 1);
        for _ in 0..8 {
            assert_eq!(allocate_from(&next), SHAPE_UNCACHEABLE);
        }
        assert_eq!(next.load(Ordering::Relaxed), SHAPE_UNCACHEABLE);
        assert!(is_cacheable_shape(0), "the empty key sequence is universal");
        assert!(!is_cacheable_shape(SHAPE_UNCACHEABLE));
    }

    #[test]
    fn shape_identity_allocation_is_unique_across_threads() {
        let next = AtomicU32::new(1);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| {
                    let next = &next;
                    scope.spawn(move || (0..1024).map(|_| allocate_from(next)).collect::<Vec<_>>())
                })
                .collect();
            let mut ids: Vec<_> = workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect();
            ids.sort_unstable();
            assert_eq!(ids, (1..=4096).collect::<Vec<_>>());
        });
    }

    #[test]
    fn shape_layout_cache_tags_collisions_and_releases_evicted_pins() {
        let mut layouts = ShapeLayouts::default();
        assert_eq!(layouts.allocated_bytes(), 0);
        let keys = crate::value::new_property_layout(vec![Rc::from("first")]);
        let weak = crate::gc_handle::WeakRef::to(keys.as_ptr() as usize);
        layouts.insert(17, keys);
        assert_eq!(layouts.allocated_bytes(), std::mem::size_of::<LayoutPage>());
        assert_eq!(&*layouts.get(17).unwrap()[0], "first");
        let collision = 17 + SHAPE_LAYOUT_CACHE_LIMIT as u32;
        assert!(layouts.get(collision).is_none());
        assert!(layouts.get_mut(collision).is_none());
        layouts.insert(
            collision,
            crate::value::new_property_layout(vec![Rc::from("second")]),
        );
        assert_eq!(layouts.len(), 1);
        assert!(layouts.get(17).is_none());
        crate::value::property_layout_cell_tests::collect(
            &crate::value::active_gc_heap(),
            &layouts,
            crate::gc_core::CollectionKind::Major,
        );
        assert!(weak.live_ptr().is_none());
        assert_eq!(&*layouts.get(collision).unwrap()[0], "second");
        layouts.insert(
            SHAPE_UNCACHEABLE,
            crate::value::new_property_layout(vec![Rc::from("untracked")]),
        );
        assert!(layouts.get(SHAPE_UNCACHEABLE).is_none());
        assert_eq!(layouts.len(), 1);
    }

    #[test]
    fn shape_layout_cache_has_a_fixed_live_storage_bound() {
        let mut layouts = ShapeLayouts::default();
        for id in 1..=(SHAPE_LAYOUT_CACHE_LIMIT as u32 * 3) {
            layouts.insert(id, crate::value::new_property_layout(vec![Rc::from("key")]));
        }
        assert_eq!(layouts.len(), SHAPE_LAYOUT_CACHE_LIMIT);
        assert_eq!(layouts.iter().count(), SHAPE_LAYOUT_CACHE_LIMIT);
        assert_eq!(
            layouts.allocated_bytes(),
            SHAPE_LAYOUT_PAGE_COUNT * std::mem::size_of::<LayoutPage>()
        );
        assert!(layouts.get(1).is_none());
        assert!(layouts.get(SHAPE_LAYOUT_CACHE_LIMIT as u32 * 3).is_some());
    }
}
