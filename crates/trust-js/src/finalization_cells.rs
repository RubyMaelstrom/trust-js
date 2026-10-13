//! Indexed storage for FinalizationRegistry [[Cells]].
//!
//! ECMA-262 §26.2.3.2–3 requires unregister to remove every cell with a SameValue token;
//! §9.12 permits cleanup to choose *any* cleared cell and requires removing it before calling
//! user code. Neither operation requires preserving a physical registration-order vector.
//! https://tc39.es/ecma262/#sec-cleanup-finalization-registry
//!
//! Cells live in a dense vector. Intrusive lists index cells by weak target and unregister-token
//! identity and link cleared cells into a ready queue. Removing a cell updates at most six neighbours
//! and relocates the last entry, fixing its links without scanning other registrations. Thus
//! draining N callbacks takes O(N) storage work, and unregister takes O(matching cells), not
//! O(all registrations). GC scans only current cells, never historical tombstones.
//! No borrowed entry or index escapes this module: callback reentrancy cannot invalidate one.

use crate::fasthash::FastMap;
use crate::interpreter::{FinalizationCell, WeakKey, WeakTarget};

const NONE: usize = usize::MAX;
const MIN_CAPACITY: usize = 32;

struct Entry {
    cell: FinalizationCell,
    target_prev: usize,
    target_next: usize,
    token_prev: usize,
    token_next: usize,
    ready_prev: usize,
    ready_next: usize,
}

pub(crate) struct FinalizationCells {
    entries: Vec<Entry>,
    target_heads: FastMap<WeakKey, usize>,
    token_heads: FastMap<WeakKey, usize>,
    ready_head: usize,
    ready_tail: usize,
}

impl Default for FinalizationCells {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            target_heads: FastMap::default(),
            token_heads: FastMap::default(),
            ready_head: NONE,
            ready_tail: NONE,
        }
    }
}

impl FinalizationCells {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &FinalizationCell> {
        self.entries.iter().map(|entry| &entry.cell)
    }

    pub(crate) fn has_ready(&self) -> bool {
        self.ready_head != NONE
    }

    /// One actual weak allocation header per distinct live target. Physical cell indices never
    /// escape, so unregister and reentrant cleanup cannot invalidate reverse-index users.
    pub(crate) fn targets(&self) -> impl Iterator<Item = &WeakTarget> {
        self.target_heads.values().map(|&index| {
            self.entries[index]
                .cell
                .target
                .as_ref()
                .expect("target list has a live target")
        })
    }

    pub(crate) fn push(&mut self, cell: FinalizationCell) {
        let index = self.entries.len();
        let ready = cell.target.is_none();
        let target_next = cell
            .target
            .as_ref()
            .and_then(|target| self.target_heads.insert(target.key(), index))
            .unwrap_or(NONE);
        if target_next != NONE {
            self.entries[target_next].target_prev = index;
        }
        let token_next = cell
            .unregister_token
            .as_ref()
            .and_then(|token| self.token_heads.insert(token.key(), index))
            .unwrap_or(NONE);
        if token_next != NONE {
            self.entries[token_next].token_prev = index;
        }
        self.entries.push(Entry {
            cell,
            target_prev: NONE,
            target_next,
            token_prev: NONE,
            token_next,
            ready_prev: NONE,
            ready_next: NONE,
        });
        if ready {
            self.enqueue_ready(index);
        }
    }

    /// A WeakTarget retains the allocation's weak header, preventing address reuse while its
    /// identity is indexed. Object pointer / unique symbol id equality is precisely SameValue
    /// for CanBeHeldWeakly values; no weak edge is upgraded or made strong by this lookup.
    #[cfg(test)]
    pub(crate) fn unregister(&mut self, key: WeakKey) -> bool {
        self.unregister_with_targets(key, |_| {})
    }

    /// Report only targets whose final cell disappeared, for O(1) reverse unsubscription.
    pub(crate) fn unregister_with_targets(
        &mut self,
        key: WeakKey,
        mut last_target: impl FnMut(WeakKey),
    ) -> bool {
        let mut removed = false;
        while let Some(&index) = self.token_heads.get(&key) {
            let cell = self.remove(index);
            if let Some(target) = &cell.target {
                if !self.target_heads.contains_key(&target.key()) {
                    last_target(target.key());
                }
            }
            removed = true;
        }
        if removed {
            self.shrink_if_sparse();
        }
        removed
    }

    /// Called during ECMA-262 §9.9.3's atomic weak-target clearing, before sweeping strong
    /// edges. Each target transitions to ready at most once, including across repeated GCs.
    #[cfg(test)]
    pub(crate) fn clear_dead_targets(&mut self, mut is_dead: impl FnMut(&WeakTarget) -> bool) {
        let dead: Vec<_> = self
            .targets()
            .filter(|target| is_dead(target))
            .map(WeakTarget::key)
            .collect();
        for key in dead {
            self.clear_target(key);
        }
    }

    /// Clear only cells with this identity. Repeated notifications cannot enqueue a cell twice.
    /// The return value is a deterministic work count, independent of unrelated registrations.
    pub(crate) fn clear_target(&mut self, key: WeakKey) -> usize {
        let Some(mut index) = self.target_heads.remove(&key) else {
            return 0;
        };
        let mut cleared = 0;
        while index != NONE {
            let next = self.entries[index].target_next;
            self.entries[index].cell.target = None;
            self.entries[index].target_prev = NONE;
            self.entries[index].target_next = NONE;
            self.enqueue_ready(index);
            index = next;
            cleared += 1;
        }
        self.shrink_if_sparse();
        cleared
    }

    pub(crate) fn pop_ready(&mut self) -> Option<FinalizationCell> {
        if self.ready_head == NONE {
            return None;
        }
        let cell = self.remove(self.ready_head);
        self.shrink_if_sparse();
        Some(cell)
    }

    /// Allocation-accounting lower bound: std HashMap control/alignment overhead is opaque.
    pub(crate) fn allocated_bytes(&self) -> usize {
        self.entries
            .capacity()
            .saturating_mul(std::mem::size_of::<Entry>())
            .saturating_add(
                self.target_heads
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(WeakKey, usize)>()),
            )
            .saturating_add(
                self.token_heads
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(WeakKey, usize)>()),
            )
    }

    fn enqueue_ready(&mut self, index: usize) {
        self.entries[index].ready_prev = self.ready_tail;
        if self.ready_tail == NONE {
            self.ready_head = index;
        } else {
            self.entries[self.ready_tail].ready_next = index;
        }
        self.ready_tail = index;
    }

    fn remove(&mut self, index: usize) -> FinalizationCell {
        let entry = &self.entries[index];
        let target = entry.cell.target.as_ref().map(WeakTarget::key);
        let (target_prev, target_next) = (entry.target_prev, entry.target_next);
        let (prev, next) = (entry.token_prev, entry.token_next);
        let token = entry.cell.unregister_token.as_ref().map(WeakTarget::key);
        let ready = entry.cell.target.is_none();
        let (ready_prev, ready_next) = (entry.ready_prev, entry.ready_next);
        if let Some(target) = target {
            if target_prev == NONE {
                if target_next == NONE {
                    self.target_heads.remove(&target);
                } else {
                    self.target_heads.insert(target, target_next);
                }
            } else {
                self.entries[target_prev].target_next = target_next;
            }
            if target_next != NONE {
                self.entries[target_next].target_prev = target_prev;
            }
        }
        if let Some(token) = token {
            if prev == NONE {
                if next == NONE {
                    self.token_heads.remove(&token);
                } else {
                    self.token_heads.insert(token, next);
                }
            } else {
                self.entries[prev].token_next = next;
            }
            if next != NONE {
                self.entries[next].token_prev = prev;
            }
        }
        if ready {
            if ready_prev == NONE {
                self.ready_head = ready_next;
            } else {
                self.entries[ready_prev].ready_next = ready_next;
            }
            if ready_next == NONE {
                self.ready_tail = ready_prev;
            } else {
                self.entries[ready_next].ready_prev = ready_prev;
            }
        }
        let removed = self.entries.swap_remove(index);
        if index < self.entries.len() {
            // swap_remove relocated the last entry. All list links refer to stable *logical*
            // cells, so redirect the moved entry's neighbours and heads to its new index.
            let moved = &self.entries[index];
            let target = moved.cell.target.as_ref().map(WeakTarget::key);
            let (target_prev, target_next) = (moved.target_prev, moved.target_next);
            let (prev, next) = (moved.token_prev, moved.token_next);
            let token = moved.cell.unregister_token.as_ref().map(WeakTarget::key);
            let ready = moved.cell.target.is_none();
            let (ready_prev, ready_next) = (moved.ready_prev, moved.ready_next);
            if let Some(target) = target {
                if target_prev == NONE {
                    self.target_heads.insert(target, index);
                } else {
                    self.entries[target_prev].target_next = index;
                }
                if target_next != NONE {
                    self.entries[target_next].target_prev = index;
                }
            }
            if let Some(token) = token {
                if prev == NONE {
                    self.token_heads.insert(token, index);
                } else {
                    self.entries[prev].token_next = index;
                }
                if next != NONE {
                    self.entries[next].token_prev = index;
                }
            }
            if ready {
                if ready_prev == NONE {
                    self.ready_head = index;
                } else {
                    self.entries[ready_prev].ready_next = index;
                }
                if ready_next == NONE {
                    self.ready_tail = index;
                } else {
                    self.entries[ready_next].ready_prev = index;
                }
            }
        }
        removed.cell
    }

    fn shrink_if_sparse(&mut self) {
        // Hysteresis prevents register/unregister churn from repeatedly reallocating. Shrinking
        // only after at least 3/4 of capacity became unused is amortized linear over removals.
        if self.entries.capacity() > MIN_CAPACITY * 2
            && self.entries.len() < self.entries.capacity() / 4
        {
            self.entries
                .shrink_to((self.entries.len() * 2).max(MIN_CAPACITY));
        }
        if self.token_heads.capacity() > MIN_CAPACITY * 2
            && self.token_heads.len() < self.token_heads.capacity() / 4
        {
            self.token_heads
                .shrink_to((self.token_heads.len() * 2).max(MIN_CAPACITY));
        }
        if self.target_heads.capacity() > MIN_CAPACITY * 2
            && self.target_heads.len() < self.target_heads.capacity() / 4
        {
            self.target_heads
                .shrink_to((self.target_heads.len() * 2).max(MIN_CAPACITY));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Object, Value};

    fn cell(target: &Value, token: Option<&Value>, held: usize) -> FinalizationCell {
        FinalizationCell {
            target: WeakTarget::of(target),
            held_value: Value::Num(held as f64),
            unregister_token: token.and_then(WeakTarget::of),
        }
    }

    fn held(cell: &FinalizationCell) -> usize {
        match cell.held_value {
            Value::Num(n) => n as usize,
            _ => panic!("numeric held value"),
        }
    }

    fn validate(cells: &FinalizationCells) {
        let mut ready = vec![false; cells.len()];
        let (mut index, mut previous) = (cells.ready_head, NONE);
        while index != NONE {
            assert!(index < cells.len());
            assert!(!ready[index], "ready-list cycle");
            ready[index] = true;
            let entry = &cells.entries[index];
            assert!(entry.cell.target.is_none());
            assert_eq!(entry.ready_prev, previous);
            previous = index;
            index = entry.ready_next;
        }
        assert_eq!(previous, cells.ready_tail);
        let mut token_seen = vec![false; cells.len()];
        let mut target_seen = vec![false; cells.len()];
        for (key, &head) in &cells.target_heads {
            let (mut index, mut previous) = (head, NONE);
            while index != NONE {
                assert!(index < cells.len());
                assert!(
                    !target_seen[index],
                    "target-list cycle or duplicate membership"
                );
                target_seen[index] = true;
                let entry = &cells.entries[index];
                assert_eq!(entry.cell.target.as_ref().unwrap().key(), *key);
                assert_eq!(entry.target_prev, previous);
                previous = index;
                index = entry.target_next;
            }
        }
        for (key, &head) in &cells.token_heads {
            let (mut index, mut previous) = (head, NONE);
            while index != NONE {
                assert!(index < cells.len());
                assert!(
                    !token_seen[index],
                    "token-list cycle or duplicate membership"
                );
                token_seen[index] = true;
                let entry = &cells.entries[index];
                assert_eq!(entry.cell.unregister_token.as_ref().unwrap().key(), *key);
                assert_eq!(entry.token_prev, previous);
                previous = index;
                index = entry.token_next;
            }
        }
        for (index, entry) in cells.entries.iter().enumerate() {
            assert_eq!(ready[index], entry.cell.target.is_none());
            assert_eq!(token_seen[index], entry.cell.unregister_token.is_some());
            assert_eq!(target_seen[index], entry.cell.target.is_some());
        }
    }

    #[test]
    fn finalization_cells_relocation_and_reentrant_mutation_model() {
        let _engine = crate::Engine::new();
        let tokens: Vec<_> = (0..11).map(|_| Value::Obj(Object::new(None))).collect();
        let targets: Vec<_> = (0..7).map(|_| Value::Obj(Object::new(None))).collect();
        let mut cells = FinalizationCells::default();
        // A simple list model independently checks randomized registration, target clearing,
        // ready removal and unregister. We assert contents, not unspecified cleanup ordering.
        let mut model: Vec<(usize, Option<WeakKey>, Option<WeakKey>)> = Vec::new();
        let mut random = 0x71a4_019bu32;
        for serial in 0..12000 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            let target = &targets[(random as usize >> 8) % targets.len()];
            let token = &tokens[(random as usize >> 16) % tokens.len()];
            let target_key = WeakKey::of(target).unwrap();
            let token_key = WeakKey::of(token).unwrap();
            match random % 7 {
                0 => {
                    let expected = model.iter().any(|(_, _, key)| *key == Some(token_key));
                    assert_eq!(cells.unregister(token_key), expected);
                    model.retain(|(_, _, key)| *key != Some(token_key));
                }
                1 => {
                    cells.clear_dead_targets(|weak| weak.key() == target_key);
                    for (_, key, _) in &mut model {
                        if *key == Some(target_key) {
                            *key = None;
                        }
                    }
                }
                2 => {
                    if let Some(cell) = cells.pop_ready() {
                        let pos = model
                            .iter()
                            .position(|(id, _, _)| *id == held(&cell))
                            .unwrap();
                        assert!(model[pos].1.is_none());
                        model.swap_remove(pos);
                    } else {
                        assert!(model.iter().all(|(_, target, _)| target.is_some()));
                    }
                }
                n => {
                    let token = (n != 3).then_some(token);
                    cells.push(cell(target, token, serial));
                    model.push((serial, Some(target_key), token.and_then(WeakKey::of)));
                }
            }
            assert_eq!(cells.len(), model.len());
            validate(&cells);
            let mut actual: Vec<_> = cells.iter().map(held).collect();
            let mut expected: Vec<_> = model.iter().map(|entry| entry.0).collect();
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(actual, expected);
        }
        cells.clear_dead_targets(|_| true);
        let count = cells.len();
        for _ in 0..count {
            assert!(cells.pop_ready().is_some());
            validate(&cells);
        }
        assert!(cells.pop_ready().is_none());
        assert!(cells.token_heads.is_empty());
        assert!(cells.target_heads.is_empty());
    }

    #[test]
    fn finalization_cells_targeted_clear_and_last_subscription_removal() {
        let _engine = crate::Engine::new();
        let old: Vec<_> = (0..10000).map(|_| Value::Obj(Object::new(None))).collect();
        let target = Value::Obj(Object::new(None));
        let token = Value::Obj(Object::new(None));
        let mut cells = FinalizationCells::default();
        for (index, old) in old.iter().enumerate() {
            cells.push(cell(old, None, index));
        }
        for n in 0..3 {
            cells.push(cell(&target, Some(&token), n));
        }
        let key = WeakKey::of(&target).unwrap();
        assert_eq!(cells.clear_target(key), 3);
        assert_eq!(cells.clear_target(key), 0);
        assert_eq!(cells.targets().count(), old.len());
        let mut removed = Vec::new();
        assert!(
            cells.unregister_with_targets(WeakKey::of(&token).unwrap(), |key| removed.push(key))
        );
        assert!(
            removed.is_empty(),
            "already cleared targets have no subscription"
        );
        cells.push(cell(&target, Some(&token), 10));
        cells.push(cell(&target, Some(&token), 11));
        assert!(
            cells.unregister_with_targets(WeakKey::of(&token).unwrap(), |key| removed.push(key))
        );
        assert_eq!(removed, vec![key]);
        validate(&cells);
    }

    #[test]
    fn finalization_cells_large_drains_and_churn_bound_storage() {
        let _engine = crate::Engine::new();
        let target = Value::Obj(Object::new(None));
        let token = Value::Obj(Object::new(None));
        let mut cells = FinalizationCells::default();
        for n in 0..100000 {
            cells.push(cell(&target, Some(&token), n));
        }
        cells.clear_dead_targets(|_| true);
        // Repeated collections do not enqueue a ready entry twice.
        cells.clear_dead_targets(|_| true);
        let mut cleaned: Vec<_> = (0..100000)
            .map(|_| held(&cells.pop_ready().unwrap()))
            .collect();
        cleaned.sort_unstable();
        assert_eq!(cleaned, (0..100000).collect::<Vec<_>>());
        assert!(cells.pop_ready().is_none());
        assert!(cells.entries.capacity() <= MIN_CAPACITY * 2);
        assert!(cells.token_heads.capacity() <= MIN_CAPACITY * 2);
        for n in 0..10000 {
            cells.push(cell(&target, Some(&token), n));
            assert!(cells.unregister(WeakKey::of(&token).unwrap()));
        }
        assert!(cells.entries.capacity() <= MIN_CAPACITY * 2);
        validate(&cells);
    }

    fn eval(engine: &mut crate::Engine, source: &str, expected: &str) {
        match engine
            .eval(source, false)
            .expect("finalization fixture parses")
        {
            crate::Completion::Value(result) => assert_eq!(result, expected),
            crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    #[test]
    fn finalization_callbacks_unregister_ready_cells_and_register_new_targets_all_tiers() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = crate::Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                var cleaned = [], token = {}, freshTarget = {}, removed;
                var registry = new FinalizationRegistry(function(value) {
                    cleaned.push(value);
                    if (cleaned.length === 1) {
                        removed = registry.unregister(token);
                        registry.register(freshTarget, 'fresh');
                    }
                });
                (function() {
                    for (var i = 0; i < 100; i++) registry.register({}, i, token);
                })();
                true
            "#,
                "true",
            );
            // Collection queues cleanup only; its first callback removes all remaining ready
            // cells. It must also observe that its own cell was removed before invocation.
            engine.interp.gc_collect();
            eval(&mut engine, "true", "true");
            eval(
                &mut engine,
                "[cleaned.length, removed, registry.unregister(token)].join(',')",
                "1,true,false",
            );
            eval(&mut engine, "freshTarget = null; true", "true");
            engine.interp.gc_collect();
            eval(&mut engine, "true", "true");
            eval(
                &mut engine,
                "[cleaned.length, cleaned[1]].join(',')",
                "2,fresh",
            );
        }
    }

    #[test]
    fn finalization_callback_throw_removes_only_chosen_cell_and_preserves_ready_queue() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = crate::Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                var cleaned = [], fail = true, token = {};
                var registry = new FinalizationRegistry(function(value) {
                    cleaned.push(value);
                    if (fail) { fail = false; throw new Error('cleanup stopped'); }
                });
                (function() { for (var i=0; i<8; i++) registry.register({}, i, token); })();
                true
            "#,
                "true",
            );
            engine.interp.gc_collect();
            eval(&mut engine, "true", "true");
            eval(&mut engine, "cleaned.length", "1");
            engine.interp.gc_collect();
            eval(&mut engine, "true", "true");
            eval(
                &mut engine,
                "[cleaned.length, cleaned.slice().sort().join(''), registry.unregister(token)].join(',')",
                "8,01234567,false",
            );
        }
    }
}
