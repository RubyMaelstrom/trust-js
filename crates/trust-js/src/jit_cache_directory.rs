//! Private fixed-size pages: a bounded entry count is also a bounded allocated payload.
//! No VecDeque growth/shrink inference, and no O(capacity) move on registration.
use super::{Retirement, SlotState, MAX_RESIDENT_ENTRIES};
use std::rc::{Rc, Weak};

pub(super) const PAGE_ENTRIES: usize = 256;
pub(super) const PAGE_COUNT: usize = MAX_RESIDENT_ENTRIES / PAGE_ENTRIES;
const PAGE_WORDS: usize = PAGE_ENTRIES / 64;
const DIRECTORY_WORDS: usize = PAGE_COUNT / 64;

pub(super) struct Page {
    entries: [Option<Weak<SlotState>>; PAGE_ENTRIES],
    occupied: [u64; PAGE_WORDS],
}

impl Default for Page {
    fn default() -> Self {
        Self {
            entries: std::array::from_fn(|_| None),
            occupied: [0; PAGE_WORDS],
        }
    }
}

pub(super) struct Directory {
    pages: [Option<Box<Page>>; PAGE_COUNT],
    non_full: [u64; DIRECTORY_WORDS],
    pub(super) len: usize,
    pub(super) allocated_pages: usize,
}

impl Default for Directory {
    fn default() -> Self {
        Self {
            pages: std::array::from_fn(|_| None),
            non_full: [u64::MAX; DIRECTORY_WORDS],
            len: 0,
            allocated_pages: 0,
        }
    }
}

impl Directory {
    pub(super) fn insert(&mut self, state: &Rc<SlotState>) -> usize {
        let (word, available) = self
            .non_full
            .iter()
            .copied()
            .enumerate()
            .find(|(_, bits)| *bits != 0)
            .expect("admission reserved directory capacity");
        let page_index = word * 64 + available.trailing_zeros() as usize;
        if self.pages[page_index].is_none() {
            self.pages[page_index] = Some(Box::default());
            self.allocated_pages += 1;
        }
        let page = self.pages[page_index].as_mut().unwrap();
        let (lane, bits) = page
            .occupied
            .iter()
            .copied()
            .enumerate()
            .find(|(_, bits)| *bits != u64::MAX)
            .unwrap();
        let bit = (!bits).trailing_zeros() as usize;
        let slot = lane * 64 + bit;
        debug_assert!(page.entries[slot].is_none());
        page.entries[slot] = Some(Rc::downgrade(state));
        page.occupied[lane] |= 1 << bit;
        if page.occupied.iter().all(|bits| *bits == u64::MAX) {
            self.non_full[word] &= !(1 << (page_index % 64));
        }
        self.len += 1;
        page_index * PAGE_ENTRIES + slot
    }

    pub(super) fn get(&self, index: usize) -> &Weak<SlotState> {
        self.pages[index / PAGE_ENTRIES].as_ref().unwrap().entries[index % PAGE_ENTRIES]
            .as_ref()
            .unwrap()
    }

    /// No entry/page owner is dropped here: even a dead Weak and an empty page
    /// move to the retirement batch, dropped after the registry borrow ends.
    pub(super) fn remove(&mut self, index: usize, retired: &mut Retirement) {
        let page_index = index / PAGE_ENTRIES;
        let slot = index % PAGE_ENTRIES;
        let page = self.pages[page_index].as_mut().unwrap();
        retired.weak.push(page.entries[slot].take().unwrap());
        page.occupied[slot / 64] &= !(1 << (slot % 64));
        self.non_full[page_index / 64] |= 1 << (page_index % 64);
        self.len -= 1;
        if page.occupied.iter().all(|bits| *bits == 0) {
            retired.pages.push(self.pages[page_index].take().unwrap());
            self.allocated_pages -= 1;
        }
    }

    /// One bounded inspection: one absent page or one bitmap word. Empty-space
    /// traversal is charged to the same budget as an occupied entry visit.
    pub(super) fn inspect(&self, cursor: &mut usize) -> Option<usize> {
        let page_index = *cursor / PAGE_ENTRIES;
        let Some(page) = self.pages[page_index].as_ref() else {
            *cursor = ((page_index + 1) * PAGE_ENTRIES) % MAX_RESIDENT_ENTRIES;
            return None;
        };
        let slot = *cursor % PAGE_ENTRIES;
        let lane = slot / 64;
        let bits = page.occupied[lane] & (u64::MAX << (slot % 64));
        if bits == 0 {
            *cursor = (page_index * PAGE_ENTRIES + (lane + 1) * 64) % MAX_RESIDENT_ENTRIES;
            return None;
        }
        let index = page_index * PAGE_ENTRIES + lane * 64 + bits.trailing_zeros() as usize;
        *cursor = (index + 1) % MAX_RESIDENT_ENTRIES;
        Some(index)
    }

    /// Explicit diagnostic only: no upgrade/drop, no Realm or object-graph walk.
    pub(super) fn dead_entries(&self) -> usize {
        self.pages
            .iter()
            .flatten()
            .flat_map(|page| page.entries.iter().flatten())
            .filter(|weak| weak.strong_count() == 0)
            .count()
    }

    pub(super) fn allocated_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.allocated_pages * std::mem::size_of::<Page>()
    }
}
