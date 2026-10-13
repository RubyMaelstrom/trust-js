//! Byte-accounted LRU storage for non-observable, reconstructible engine artifacts.
//!
//! Entry-count limits alone allow one source string or compiled program to retain an arbitrary
//! amount of memory. This cache applies both a byte budget and a metadata-count budget. Its queue
//! uses generations, so a hit is one hash lookup plus an append; stale recency records are removed
//! incrementally during eviction and periodically compacted.

use std::collections::VecDeque;
use std::hash::Hash;
use std::rc::Rc;

use crate::fasthash::FastMap;

struct Entry<V> {
    value: V,
    bytes: usize,
    generation: u64,
}

pub(crate) struct ByteLru<K, V> {
    entries: FastMap<K, Entry<V>>,
    order: VecDeque<(K, u64)>,
    bytes: usize,
    max_bytes: usize,
    max_entries: usize,
    generation: u64,
}

struct RegexpEntry {
    source: Rc<str>,
    flags: Rc<str>,
    value: Rc<crate::regex::Regex>,
    bytes: usize,
    generation: u64,
}

/// Allocation-free-on-hit two-level index for compiled RegExp programs, with global byte/LRU
/// accounting across every `(source, flags)` pair.
pub(crate) struct RegexpProgramCache {
    programs: FastMap<Rc<str>, FastMap<Rc<str>, RegexpEntry>>,
    order: VecDeque<(Rc<str>, Rc<str>, u64)>,
    bytes: usize,
    max_bytes: usize,
    max_entries: usize,
    entries: usize,
    generation: u64,
}

impl RegexpProgramCache {
    pub(crate) fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            programs: FastMap::default(),
            order: VecDeque::new(),
            bytes: 0,
            max_bytes,
            max_entries,
            entries: 0,
            generation: 0,
        }
    }

    pub(crate) fn get(&mut self, source: &str, flags: &str) -> Option<Rc<crate::regex::Regex>> {
        let entry = self.programs.get(source)?.get(flags)?;
        if entry.generation == self.generation {
            return Some(entry.value.clone());
        }
        self.next_generation();
        let entry = self
            .programs
            .get_mut(source)
            .and_then(|by_flags| by_flags.get_mut(flags))
            .expect("RegExp LRU hit remains live");
        entry.generation = self.generation;
        let source = entry.source.clone();
        let flags = entry.flags.clone();
        let value = entry.value.clone();
        self.order.push_back((source, flags, self.generation));
        self.compact_order_if_needed();
        Some(value)
    }

    pub(crate) fn insert(
        &mut self,
        source: &str,
        flags: &str,
        value: Rc<crate::regex::Regex>,
        bytes: usize,
    ) {
        if bytes > self.max_bytes || self.max_entries == 0 {
            self.remove(source, flags);
            return;
        }
        self.next_generation();
        let source: Rc<str> = source.into();
        let flags: Rc<str> = flags.into();
        let entry = RegexpEntry {
            source: source.clone(),
            flags: flags.clone(),
            value,
            bytes,
            generation: self.generation,
        };
        if let Some(previous) = self
            .programs
            .entry(source.clone())
            .or_default()
            .insert(flags.clone(), entry)
        {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        } else {
            self.entries += 1;
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.order.push_back((source, flags, self.generation));
        self.evict_to_budget();
        self.compact_order_if_needed();
    }

    fn remove(&mut self, source: &str, flags: &str) {
        let (removed, empty) = match self.programs.get_mut(source) {
            Some(by_flags) => {
                let removed = by_flags.remove(flags);
                (removed, by_flags.is_empty())
            }
            None => return,
        };
        if let Some(entry) = removed {
            self.bytes = self.bytes.saturating_sub(entry.bytes);
            self.entries -= 1;
        }
        if empty {
            self.programs.remove(source);
        }
    }

    fn evict_to_budget(&mut self) {
        while self.bytes > self.max_bytes || self.entries > self.max_entries {
            let Some((source, flags, generation)) = self.order.pop_front() else {
                break;
            };
            if self
                .programs
                .get(source.as_ref())
                .and_then(|by_flags| by_flags.get(flags.as_ref()))
                .is_some_and(|entry| entry.generation == generation)
            {
                self.remove(source.as_ref(), flags.as_ref());
            }
        }
    }

    fn compact_order_if_needed(&mut self) {
        let limit = self.entries.saturating_mul(4).saturating_add(64);
        if self.order.len() <= limit {
            return;
        }
        let mut current: Vec<_> = self
            .programs
            .values()
            .flat_map(|by_flags| {
                by_flags
                    .values()
                    .map(|entry| (entry.source.clone(), entry.flags.clone(), entry.generation))
            })
            .collect();
        current.sort_unstable_by_key(|(_, _, generation)| *generation);
        self.order = current.into();
    }

    fn next_generation(&mut self) {
        if self.generation == u64::MAX {
            let mut current: Vec<_> = self
                .programs
                .values()
                .flat_map(|by_flags| {
                    by_flags
                        .values()
                        .map(|entry| (entry.source.clone(), entry.flags.clone(), entry.generation))
                })
                .collect();
            current.sort_unstable_by_key(|(_, _, generation)| *generation);
            self.order.clear();
            for (index, (source, flags, _)) in current.into_iter().enumerate() {
                let generation = index as u64 + 1;
                self.programs
                    .get_mut(source.as_ref())
                    .and_then(|by_flags| by_flags.get_mut(flags.as_ref()))
                    .expect("RegExp LRU key remains live")
                    .generation = generation;
                self.order.push_back((source, flags, generation));
            }
            self.generation = self.entries as u64;
        }
        self.generation += 1;
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.entries, self.bytes)
    }

    /// Visit every allocation retained by the cache while returning the directly-owned table and
    /// recency-queue storage. Standard-library HashMap bucket capacity is opaque, so the table
    /// contribution is deliberately a lower bound; payloads are attributed by their canonical
    /// allocation family through `Visitor` rather than to the cache that discovered them.
    pub(crate) fn scan_retained_memory(
        &self,
        visitor: &mut crate::memory::Visitor,
    ) -> (usize, bool) {
        let mut storage = self
            .programs
            .len()
            .saturating_mul(std::mem::size_of::<(Rc<str>, FastMap<Rc<str>, RegexpEntry>)>())
            .saturating_add(self.order.capacity().saturating_mul(std::mem::size_of::<(
                Rc<str>,
                Rc<str>,
                u64,
            )>()));
        for (source, by_flags) in &self.programs {
            visitor.rc_str(source);
            storage = storage.saturating_add(
                by_flags
                    .len()
                    .saturating_mul(std::mem::size_of::<(Rc<str>, RegexpEntry)>()),
            );
            for (flags, entry) in by_flags {
                visitor.rc_str(flags);
                visitor.rc_str(&entry.source);
                visitor.rc_str(&entry.flags);
                visitor.regex(&entry.value);
            }
        }
        // Stale recency records can outlive their table entry and therefore be the sole owner of
        // their string payload. The global Rc<str> identity set prevents duplicate attribution.
        for (source, flags, _) in &self.order {
            visitor.rc_str(source);
            visitor.rc_str(flags);
        }
        (storage, false)
    }
}

impl<K, V> ByteLru<K, V>
where
    K: Clone + Eq + Hash,
{
    pub(crate) fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            entries: FastMap::default(),
            order: VecDeque::new(),
            bytes: 0,
            max_bytes,
            max_entries,
            generation: 0,
        }
    }

    pub(crate) fn get_cloned(&mut self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get_mapped(key, Clone::clone)
    }

    /// Refresh recency but copy only the caller-selected result, not ownership pins or other
    /// metadata retained by the entry. The projection cannot borrow beyond this lookup.
    pub(crate) fn get_mapped<R>(&mut self, key: &K, project: impl FnOnce(&V) -> R) -> Option<R> {
        let entry = self.entries.get(key)?;
        if entry.generation == self.generation {
            return Some(project(&entry.value));
        }
        let generation = self.next_generation();
        let entry = self.entries.get_mut(key).expect("LRU hit remains live");
        entry.generation = generation;
        let value = project(&entry.value);
        self.order.push_back((key.clone(), generation));
        self.compact_order_if_needed();
        Some(value)
    }

    pub(crate) fn insert(&mut self, key: K, value: V, bytes: usize) {
        if bytes > self.max_bytes || self.max_entries == 0 {
            self.remove(&key);
            return;
        }
        let generation = self.next_generation();
        if let Some(previous) = self.entries.insert(
            key.clone(),
            Entry {
                value,
                bytes,
                generation,
            },
        ) {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.order.push_back((key, generation));
        self.evict_to_budget();
        self.compact_order_if_needed();
    }

    fn next_generation(&mut self) -> u64 {
        if self.generation == u64::MAX {
            self.renumber_generations();
        }
        self.generation += 1;
        self.generation
    }

    fn remove(&mut self, key: &K) {
        if let Some(entry) = self.entries.remove(key) {
            self.bytes = self.bytes.saturating_sub(entry.bytes);
        }
    }

    fn evict_to_budget(&mut self) {
        while self.bytes > self.max_bytes || self.entries.len() > self.max_entries {
            let Some((key, generation)) = self.order.pop_front() else {
                break;
            };
            if self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.generation == generation)
            {
                self.remove(&key);
            }
        }
    }

    fn compact_order_if_needed(&mut self) {
        let limit = self.entries.len().saturating_mul(4).saturating_add(64);
        if self.order.len() > limit {
            self.rebuild_order();
        }
    }

    fn rebuild_order(&mut self) {
        let mut current: Vec<_> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.generation))
            .collect();
        current.sort_unstable_by_key(|(_, generation)| *generation);
        self.order = current.into();
    }

    fn renumber_generations(&mut self) {
        let mut current: Vec<_> = self
            .entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.generation))
            .collect();
        current.sort_unstable_by_key(|(_, generation)| *generation);
        self.order.clear();
        for (index, (key, _)) in current.into_iter().enumerate() {
            let generation = index as u64 + 1;
            self.entries
                .get_mut(&key)
                .expect("LRU key remains live")
                .generation = generation;
            self.order.push_back((key, generation));
        }
        self.generation = self.entries.len() as u64;
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.entries.len(), self.bytes)
    }

    /// Visit live values and report storage owned by the table and recency queue. HashMap bucket
    /// capacity is not exposed by std, so the entry portion is a review-visible lower bound.
    pub(crate) fn scan_retained_memory(&self, mut visit: impl FnMut(&V)) -> (usize, bool) {
        for entry in self.entries.values() {
            visit(&entry.value);
        }
        let bytes = self
            .entries
            .len()
            .saturating_mul(std::mem::size_of::<(K, Entry<V>)>())
            .saturating_add(
                self.order
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(K, u64)>()),
            );
        (bytes, false)
    }
}

#[cfg(test)]
mod tests {
    use super::ByteLru;
    use std::rc::Rc;

    #[test]
    fn mapped_hits_do_not_clone_pins_and_stale_generations_do_not_evict_reused_keys() {
        struct Pinned {
            _pin: Rc<()>,
            result: usize,
        }
        let pin = Rc::new(());
        let weak = Rc::downgrade(&pin);
        let mut cache = ByteLru::new(2, 2);
        cache.insert(
            1,
            Pinned {
                _pin: pin,
                result: 10,
            },
            1,
        );
        cache.insert(
            2,
            Pinned {
                _pin: Rc::new(()),
                result: 20,
            },
            1,
        );
        assert_eq!(cache.get_mapped(&1, |entry| entry.result), Some(10));
        assert_eq!(weak.strong_count(), 1);
        assert_eq!(cache.get_mapped(&2, |entry| entry.result), Some(20));
        cache.remove(&1);
        assert!(
            weak.upgrade().is_none(),
            "stale recency key must not own entry's pins"
        );
        // Reused pointer-like identity with a different allocation and generation.
        cache.insert(
            1,
            Pinned {
                _pin: Rc::new(()),
                result: 30,
            },
            1,
        );
        cache.insert(
            3,
            Pinned {
                _pin: Rc::new(()),
                result: 40,
            },
            1,
        );
        assert_eq!(cache.get_mapped(&1, |entry| entry.result), Some(30));
        assert!(cache.get_mapped(&2, |entry| entry.result).is_none());
        cache.generation = u64::MAX;
        assert_eq!(cache.get_mapped(&3, |entry| entry.result), Some(40));
        assert_eq!(cache.get_mapped(&1, |entry| entry.result), Some(30));
    }

    #[test]
    fn evicts_by_bytes_then_recency_and_skips_oversized_entries() {
        let mut cache = ByteLru::new(10, 3);
        cache.insert("a", 1, 4);
        cache.insert("b", 2, 4);
        assert_eq!(cache.get_cloned(&"a"), Some(1));
        cache.insert("c", 3, 4);
        assert_eq!(cache.get_cloned(&"b"), None, "least-recent byte victim");
        assert_eq!(cache.get_cloned(&"a"), Some(1));
        assert_eq!(cache.get_cloned(&"c"), Some(3));
        assert_eq!(cache.stats(), (2, 8));

        cache.insert("huge", 4, 11);
        assert_eq!(cache.get_cloned(&"huge"), None);
        assert_eq!(cache.stats(), (2, 8));
    }

    #[test]
    fn replacement_updates_accounting_without_stale_queue_eviction() {
        let mut cache = ByteLru::new(8, 2);
        cache.insert(1, "old", 7);
        cache.insert(1, "new", 2);
        cache.insert(2, "other", 6);
        assert_eq!(cache.stats(), (2, 8));
        assert_eq!(cache.get_cloned(&1), Some("new"));
        assert_eq!(cache.get_cloned(&2), Some("other"));
    }

    #[test]
    fn regexp_cache_accounts_all_flag_variants_and_evicts_globally() {
        let mut cache = super::RegexpProgramCache::new(10, 3);
        let a = Rc::new(crate::regex::Regex::new("a", "").unwrap());
        let ai = Rc::new(crate::regex::Regex::new("a", "i").unwrap());
        let b = Rc::new(crate::regex::Regex::new("b", "").unwrap());
        cache.insert("a", "", a.clone(), 4);
        cache.insert("a", "i", ai, 4);
        assert!(Rc::ptr_eq(&cache.get("a", "").unwrap(), &a));
        cache.insert("b", "", b, 4);
        assert!(cache.get("a", "i").is_none());
        assert!(cache.get("a", "").is_some());
        assert!(cache.get("b", "").is_some());
        assert_eq!(cache.stats(), (2, 8));
    }
}
