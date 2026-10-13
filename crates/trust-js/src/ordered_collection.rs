//! Ordered, indexed [[MapData]] / [[SetData]] storage.
//!
//! ECMA-262 (snapshot e28783d5), #sec-createmapiterator, #sec-createsetiterator,
//! and Map/Set.prototype.forEach require a live insertion-order walk: deletion skips an entry,
//! reinsertion appends it, and clear must not disconnect suspended cursors from future insertions.
//! The specification's unbounded list of empty entries is not an implementation requirement.
//! Compact holes geometrically and translate every live cursor to its new next-entry offset.
//! Weak cursor registrations never retain an iterator or a JavaScript object.

use crate::fasthash::{FastMap, FxHasher};
use crate::value::{Gc, Value};
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::rc::{Rc, Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollectionKind {
    Map,
    Set,
}

impl CollectionKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Map => "Map",
            Self::Set => "Set",
        }
    }
}

/// Most hash buckets have one entry, so the collision-free case needs no separate allocation.
#[derive(Debug)]
enum Bucket {
    One(usize),
    Many(Vec<usize>),
}

impl Bucket {
    /// Hash equality is only an index hint; every hit still proves SameValueZero against
    /// the authoritative ordered entry. No key conversion or author code runs here.
    fn find(&self, entries: &[Option<(Value, Value)>], key: &Value) -> Option<usize> {
        let matches = |offset: usize| {
            entries[offset]
                .as_ref()
                .is_some_and(|(candidate, _)| same_key(candidate, key))
        };
        match self {
            Self::One(offset) => matches(*offset).then_some(*offset),
            Self::Many(offsets) => offsets.iter().copied().find(|&offset| matches(offset)),
        }
    }

    fn append(&mut self, offset: usize) {
        match self {
            Self::One(previous) => *self = Self::Many(vec![*previous, offset]),
            Self::Many(offsets) => offsets.push(offset),
        }
    }
}

#[derive(Clone)]
pub(crate) struct CollectionCursor(Rc<CursorState>);

type CursorRegistry = RefCell<Vec<Weak<CursorState>>>;

struct CursorState {
    position: Cell<usize>,
    slot: Cell<usize>,
    registry: Weak<CursorRegistry>,
}

impl Drop for CursorState {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut cursors = registry.borrow_mut();
        let slot = self.slot.get();
        cursors.swap_remove(slot);
        if let Some(moved) = cursors.get(slot).and_then(Weak::upgrade) {
            moved.slot.set(slot);
        }
        // Final cursor destruction removes its weak header immediately. No GC or future cursor
        // allocation has to scan historical registrations, including after a one-off large burst.
        if cursors.capacity() > cursors.len().saturating_mul(4).max(32) {
            let capacity = cursors.len().saturating_mul(2).max(16);
            cursors.shrink_to(capacity);
        }
    }
}

/// Actual internal slots, not forgeable properties. An exhausted iterator keeps its brand but
/// releases both the target collection and its cursor, as a completed generator does.
pub(crate) struct CollectionIterator {
    pub(crate) target: Option<Gc>,
    pub(crate) cursor: Option<CollectionCursor>,
    pub(crate) kind: u8,
    pub(crate) brand: CollectionKind,
}

pub(crate) struct OrderedCollection {
    kind: CollectionKind,
    entries: Vec<Option<(Value, Value)>>,
    index: FastMap<u64, Bucket>,
    live_len: usize,
    // Lazy: collections used only for lookup need no registry allocation.
    cursors: Option<Rc<CursorRegistry>>,
}

impl OrderedCollection {
    pub(crate) fn new(kind: CollectionKind) -> Self {
        Self::with_capacity(kind, 0)
    }

    pub(crate) fn with_capacity(kind: CollectionKind, capacity: usize) -> Self {
        Self {
            kind,
            entries: Vec::with_capacity(capacity),
            index: FastMap::with_capacity_and_hasher(capacity, Default::default()),
            live_len: 0,
            cursors: None,
        }
    }

    pub(crate) fn kind(&self) -> CollectionKind {
        self.kind
    }

    pub(crate) fn len(&self) -> usize {
        self.live_len
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &(Value, Value)> {
        self.entries.iter().filter_map(Option::as_ref)
    }

    fn entry_index(&self, key: &Value) -> Option<usize> {
        self.index.get(&key_hash(key))?.find(&self.entries, key)
    }

    pub(crate) fn get(&self, key: &Value) -> Option<&Value> {
        let offset = self.entry_index(key)?;
        self.entries[offset].as_ref().map(|(_, value)| value)
    }

    pub(crate) fn has(&self, key: &Value) -> bool {
        self.entry_index(key).is_some()
    }

    fn index_insert(&mut self, hash: u64, offset: usize) {
        use std::collections::hash_map::Entry;
        match self.index.entry(hash) {
            Entry::Vacant(entry) => {
                entry.insert(Bucket::One(offset));
            }
            Entry::Occupied(mut entry) => entry.get_mut().append(offset),
        }
    }

    pub(crate) fn insert(&mut self, key: Value, value: Value) {
        use std::collections::hash_map::Entry;
        // Map.prototype.set / Set.prototype.add canonicalize -0, then update in place or
        // append. Hold the one index lookup across that decision: hashing a long String or
        // BigInt again, or probing the table again, has no semantic purpose. The ordered
        // entries and live cursors remain authoritative (ECMA-262 §24.1.3.9 / §24.2.4.1).
        let key = match key {
            Value::Num(n) => Value::Num(if n == 0.0 { 0.0 } else { n }),
            key => key,
        };
        let hash = key_hash(&key);
        match self.index.entry(hash) {
            Entry::Occupied(mut entry) => {
                if let Some(offset) = entry.get().find(&self.entries, &key) {
                    self.entries[offset].as_mut().unwrap().1 = value;
                    return;
                }
                let offset = self.entries.len();
                self.entries.push(Some((key, value)));
                entry.get_mut().append(offset);
            }
            Entry::Vacant(entry) => {
                let offset = self.entries.len();
                self.entries.push(Some((key, value)));
                entry.insert(Bucket::One(offset));
            }
        }
        self.live_len += 1;
    }

    pub(crate) fn delete(&mut self, key: &Value) -> bool {
        use std::collections::hash_map::Entry;
        let Entry::Occupied(mut entry) = self.index.entry(key_hash(key)) else {
            return false;
        };
        let Some(offset) = entry.get().find(&self.entries, key) else {
            return false;
        };
        match entry.get_mut() {
            Bucket::One(_) => {
                entry.remove();
            }
            Bucket::Many(offsets) => {
                offsets.retain(|&candidate| candidate != offset);
                if offsets.len() == 1 {
                    *entry.get_mut() = Bucket::One(offsets[0]);
                } else if offsets.capacity() > offsets.len().saturating_mul(4).max(32) {
                    offsets.shrink_to(offsets.len().saturating_mul(2));
                }
            }
        }
        self.entries[offset] = None;
        self.live_len -= 1;
        if self.live_len == 0 {
            self.clear();
        } else if self.entries.len() - self.live_len > self.live_len.max(32) {
            self.compact();
        }
        true
    }

    fn visit_cursors(&self, mut visit: impl FnMut(&Cell<usize>)) {
        let Some(registry) = &self.cursors else {
            return;
        };
        // Only private position translations run here: never author code or cursor destruction.
        for weak in registry.borrow().iter() {
            let cursor = weak.upgrade().expect("live cursor owns its registration");
            visit(&cursor.position);
        }
    }

    pub(crate) fn cursor(&mut self) -> CollectionCursor {
        let registry = self.cursors.get_or_insert_with(Default::default);
        let mut cursors = registry.borrow_mut();
        let cursor = Rc::new(CursorState {
            position: Cell::new(0),
            slot: Cell::new(cursors.len()),
            registry: Rc::downgrade(registry),
        });
        cursors.push(Rc::downgrade(&cursor));
        CollectionCursor(cursor)
    }

    /// Clone the next live entry before any JavaScript callback executes. The cursor is advanced
    /// first, so a reentrant delete/reinsert or clear sees precisely the suspended next position.
    pub(crate) fn next(&self, cursor: &CollectionCursor) -> Option<(Value, Value)> {
        let mut offset = cursor.0.position.get();
        while offset < self.entries.len() {
            let entry = &self.entries[offset];
            offset += 1;
            cursor.0.position.set(offset);
            if let Some(entry) = entry {
                return Some(entry.clone());
            }
        }
        None
    }

    pub(crate) fn clear(&mut self) {
        // Clear changes no observable iteration state: all old entries disappeared, and every
        // still-live cursor must see later appends. Completed JS iterators already dropped theirs.
        self.entries.clear();
        self.index.clear();
        self.live_len = 0;
        self.visit_cursors(|cursor| cursor.set(0));
        // Keep only small reusable allocations, not the collection's historical high-water mark.
        if self.entries.capacity() > 64 {
            self.entries = Vec::new();
        }
        if self.index.capacity() > 64 {
            self.index = FastMap::default();
        }
    }

    fn compact(&mut self) {
        // prefix[n] is the number of surviving entries before old position n. Translate cursors
        // before moving entries; this works for any number of nested and dormant live walks.
        if self
            .cursors
            .as_ref()
            .is_some_and(|cursors| !cursors.borrow().is_empty())
        {
            let mut prefix = Vec::with_capacity(self.entries.len() + 1);
            prefix.push(0usize);
            for entry in &self.entries {
                prefix.push(prefix.last().unwrap() + usize::from(entry.is_some()));
            }
            self.visit_cursors(|cursor| cursor.set(prefix[cursor.get()]));
        }
        self.entries.retain(Option::is_some);
        if self.entries.capacity() > self.live_len.saturating_mul(4).max(64) {
            self.entries.shrink_to(self.live_len.saturating_mul(2));
        }
        self.index.clear();
        if self.index.capacity() > self.live_len.saturating_mul(4).max(64) {
            self.index.shrink_to(self.live_len.saturating_mul(2));
        }
        for offset in 0..self.entries.len() {
            let hash = key_hash(&self.entries[offset].as_ref().unwrap().0);
            self.index_insert(hash, offset);
        }
    }

    /// Owned allocation bytes; HashMap control bytes and spare buckets remain a documented lower
    /// bound. Each live cursor Rc allocation is counted once via its sole weak registration.
    pub(crate) fn storage_bytes_lower_bound(&self) -> usize {
        let mut bytes = self
            .entries
            .capacity()
            .saturating_mul(size_of::<Option<(Value, Value)>>())
            .saturating_add(self.index.len().saturating_mul(size_of::<(u64, Bucket)>()));
        if let Some(registry) = &self.cursors {
            let cursors = registry.borrow();
            bytes = bytes
                .saturating_add(size_of::<CursorRegistry>() + 2 * size_of::<usize>())
                .saturating_add(cursors.capacity() * size_of::<Weak<CursorState>>())
                .saturating_add(
                    cursors.len() * (size_of::<CursorState>() + 2 * size_of::<usize>()),
                );
        }
        for bucket in self.index.values() {
            if let Bucket::Many(offsets) = bucket {
                bytes = bytes.saturating_add(offsets.capacity().saturating_mul(size_of::<usize>()));
            }
        }
        bytes
    }
}

fn same_key(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Num(a), Value::Num(b)) => a == b || (a.is_nan() && b.is_nan()),
        _ => crate::builtins::same_value_pub(left, right),
    }
}

/// SameValueZero-compatible hash. Always verify equality after a bucket hit: the key count is
/// independent of the number of distinct hashes, and strings/BigInts have no second owning copy.
pub(crate) fn key_hash(value: &Value) -> u64 {
    #[cfg(test)]
    KEY_HASH_CALLS.with(|count| count.set(count.get() + 1));
    let mut state = FxHasher::default();
    match value {
        Value::Undefined => state.write_u8(0),
        Value::Empty => state.write_u8(1),
        Value::Null => state.write_u8(2),
        Value::Bool(value) => {
            state.write_u8(3);
            value.hash(&mut state);
        }
        Value::Num(value) => {
            state.write_u8(4);
            state.write_u64(if *value == 0.0 {
                0.0f64.to_bits()
            } else if value.is_nan() {
                f64::NAN.to_bits()
            } else {
                value.to_bits()
            });
        }
        Value::BigInt(value) => {
            state.write_u8(5);
            value.hash(&mut state);
        }
        Value::Str(value) => {
            state.write_u8(6);
            value.hash(&mut state);
            // Fx's byte loop still leaves clustered low bits for fixed-width suffixes.
            // Fold the high half into the low half before and after an odd multiply.
            // This bijection retains collisions among strings while using one multiply
            // for a digest that already accumulated all string bytes and its delimiter.
            let hash = state.finish();
            let hash = (hash ^ (hash >> 32)).wrapping_mul(0xd6e8_feb8_6659_fd93);
            return hash ^ (hash >> 32);
        }
        Value::Sym(value) => {
            state.write_u8(7);
            state.write_u64(value.id);
        }
        Value::Obj(value) => {
            state.write_u8(8);
            state.write_usize(Rc::as_ptr(value) as usize);
        }
    }
    // The table masks the LOW hash bits to choose its initial bucket. Fx's
    // multiply/rotate accumulation does not move the high bits of a final word
    // down: thousands of ordinary
    // integer-valued f64 keys otherwise all start at the same table bucket.
    // Avalanche fixed-width/BigInt tagged digests so every input bit reaches both
    // bucket selection and the high-bit control-byte fingerprint. This
    // bijective 64-bit finalizer changes neither equality nor true collisions;
    // Bucket::find still proves SameValueZero, and ordered entries/cursors are
    // unchanged. ECMA-262 #sec-map-objects / #sec-set-objects (e28783d5)
    // require average sublinear access, not a linear probe through this cluster.
    let mut hash = state.finish();
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

#[cfg(test)]
thread_local! {
    static KEY_HASH_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
impl From<Vec<(Value, Value)>> for OrderedCollection {
    fn from(entries: Vec<(Value, Value)>) -> Self {
        let mut collection = Self::with_capacity(CollectionKind::Map, entries.len());
        for (key, value) in entries {
            collection.insert(key, value);
        }
        collection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(value: &Value) -> u32 {
        match value {
            Value::Num(value) => *value as u32,
            _ => panic!("expected numeric model value"),
        }
    }

    #[test]
    fn ordered_collection_hash_spreads_numeric_key_families_across_buckets() {
        use std::hash::BuildHasher;

        // ECMA-262 #sec-map-objects / #sec-set-objects require average
        // sublinear access. Distinct 64-bit hashes are insufficient if a
        // power-of-two table sends all ordinary numeric keys to one bucket.
        // Inspect the actual index hasher, not only the intermediate digest.
        let data = OrderedCollection::new(CollectionKind::Map);
        for count in [128usize, 1024, 4096] {
            for scale in [
                f64::from_bits(1),
                2.0f64.powi(-1022),
                0.125,
                1.0,
                2.0f64.powi(900),
            ] {
                for sign in [-1.0, 1.0] {
                    let mut occupancy = vec![0usize; count * 2];
                    for index in 0..count {
                        let key = Value::Num(sign * ((index + 1) as f64) * scale);
                        let bucket = data.index.hasher().hash_one(key_hash(&key)) as usize
                            & (occupancy.len() - 1);
                        occupancy[bucket] += 1;
                    }
                    let occupied = occupancy.iter().filter(|&&length| length != 0).count();
                    let longest = occupancy.iter().copied().max().unwrap();
                    assert!(
                        occupied > count / 2 && longest < 16,
                        "count={count} scale={scale} sign={sign}: {occupied} occupied buckets, largest={longest}"
                    );
                }
            }
        }
    }

    #[test]
    fn ordered_collection_string_hash_distributes_common_prefixes_and_suffixes() {
        use std::hash::BuildHasher;

        // Check the actual bucket mapping: retaining the byte-hash path must not
        // trade string lookup cost for the numeric clustering defect repaired above.
        let data = OrderedCollection::new(CollectionKind::Map);
        for count in [128usize, 1024, 4096] {
            for family in 0..4 {
                let mut occupancy = vec![0usize; count * 2];
                for index in 0..count {
                    let text = match family {
                        0 => format!("key-{index:08}"),
                        1 => format!("common prefix for a collection key {index:08}"),
                        2 => format!("{index:08} common suffix for a collection key"),
                        _ => format!("é中🦀-{index:08}-é中🦀"),
                    };
                    let hash = key_hash(&Value::lstr(text));
                    let bucket =
                        data.index.hasher().hash_one(hash) as usize & (occupancy.len() - 1);
                    occupancy[bucket] += 1;
                }
                let occupied = occupancy.iter().filter(|&&length| length != 0).count();
                let longest = occupancy.iter().copied().max().unwrap();
                assert!(
                    occupied > count / 2 && longest < 16,
                    "count={count} family={family}: {occupied} occupied buckets, largest={longest}"
                );
            }
        }
    }

    #[test]
    fn ordered_collection_hash_preserves_equal_keys_and_identity() {
        let mut data = OrderedCollection::new(CollectionKind::Map);
        let equal_pairs = [
            (Value::Num(-0.0), Value::Num(0.0)),
            (
                Value::Num(f64::from_bits(0x7ff0_0000_0000_0001)),
                Value::Num(f64::from_bits(0xfff8_ffff_ffff_ffff)),
            ),
            (
                Value::lstr("equal separate string allocations".to_owned()),
                Value::lstr("equal separate string allocations".to_owned()),
            ),
            (
                Value::BigInt(crate::bigint::JsBigInt::from(123456789i64)),
                Value::BigInt(crate::bigint::JsBigInt::from(123456789i64)),
            ),
        ];
        for (index, (first, second)) in equal_pairs.into_iter().enumerate() {
            assert_eq!(key_hash(&first), key_hash(&second));
            data.insert(first, Value::Num(index as f64));
            assert_eq!(number(data.get(&second).unwrap()), index as u32);
            data.insert(second, Value::Num((index + 10) as f64));
            assert_eq!(data.len(), index + 1);
        }
        let object = Value::Obj(crate::value::Object::new(None));
        let different = Value::Obj(crate::value::Object::new(None));
        data.insert(object.clone(), Value::Num(100.0));
        assert_eq!(number(data.get(&object).unwrap()), 100);
        assert!(!data.has(&different));
        // Tagging must preserve semantic distinctions even when payloads look
        // alike. Hash collisions are allowed but may never merge these keys.
        for key in [
            Value::Undefined,
            Value::Null,
            Value::Bool(false),
            Value::Num(123456789.0),
            Value::lstr("123456789".to_owned()),
        ] {
            let previous = data.len();
            data.insert(key, Value::Undefined);
            assert_eq!(data.len(), previous + 1);
        }
    }

    #[test]
    fn ordered_collection_collision_size_and_zero_nan() {
        let mut data = OrderedCollection::new(CollectionKind::Map);
        let collision = Value::Num(f64::from_bits(0xbe60_db93_9105_4a88));
        assert_eq!(key_hash(&Value::Undefined), key_hash(&collision));
        data.insert(Value::Undefined, Value::Num(1.0));
        data.insert(collision.clone(), Value::Num(2.0));
        assert_eq!(data.len(), 2);
        assert_eq!(number(data.get(&collision).unwrap()), 2);
        assert!(data.delete(&Value::Undefined));
        assert_eq!(data.len(), 1);
        assert!(data.has(&collision));
        data.insert(Value::Num(-0.0), Value::Num(3.0));
        data.insert(Value::Num(0.0), Value::Num(4.0));
        data.insert(Value::Num(f64::NAN), Value::Num(5.0));
        data.insert(
            Value::Num(f64::from_bits(0x7ff8_0000_0000_0001)),
            Value::Num(6.0),
        );
        assert_eq!(data.len(), 3);
        assert_eq!(number(data.get(&Value::Num(f64::NAN)).unwrap()), 6);
        let zero = data
            .iter()
            .find(|(key, _)| same_key(key, &Value::Num(0.0)))
            .unwrap();
        assert!(matches!(zero.0, Value::Num(n) if !n.is_sign_negative()));
    }

    #[test]
    fn ordered_collection_mutations_hash_once_and_keep_collision_order() {
        let mut data = OrderedCollection::new(CollectionKind::Map);
        let collision = Value::Num(f64::from_bits(0xbe60_db93_9105_4a88));
        assert_eq!(key_hash(&Value::Undefined), key_hash(&collision));
        let once = |operation: &mut dyn FnMut()| {
            let before = KEY_HASH_CALLS.with(Cell::get);
            operation();
            assert_eq!(KEY_HASH_CALLS.with(Cell::get) - before, 1);
        };
        once(&mut || data.insert(Value::Undefined, Value::Num(1.0)));
        once(&mut || data.insert(collision.clone(), Value::Num(2.0)));
        let cursor = data.cursor();
        assert!(matches!(data.next(&cursor).unwrap().0, Value::Undefined));
        // Updating a collision hit must neither append nor advance/rewind live cursors.
        once(&mut || data.insert(collision.clone(), Value::Num(3.0)));
        once(&mut || data.insert(Value::Undefined, Value::Num(4.0)));
        assert_eq!(data.len(), 2);
        assert_eq!(number(&data.next(&cursor).unwrap().1), 3);
        assert!(data.next(&cursor).is_none());
        once(&mut || assert!(data.delete(&Value::Undefined)));
        once(&mut || assert!(!data.delete(&Value::Undefined)));
        once(&mut || data.insert(Value::Undefined, Value::Num(5.0)));
        assert!(matches!(data.next(&cursor).unwrap().0, Value::Undefined));
        assert!(data.next(&cursor).is_none());
        // Collapse Many to One through the other key, then remove the final bucket.
        once(&mut || assert!(data.delete(&collision)));
        assert_eq!(number(data.get(&Value::Undefined).unwrap()), 5);
        once(&mut || assert!(data.delete(&Value::Undefined)));
        once(&mut || assert!(!data.delete(&Value::Undefined)));
        assert_eq!(data.len(), 0);
        assert!(data.index.is_empty());
    }

    #[test]
    fn ordered_collection_single_probe_preserves_key_and_releases_replaced_values() {
        let key = crate::value::Object::new(None);
        let first = crate::value::Object::new(None);
        let second = crate::value::Object::new(None);
        let first_weak = Rc::downgrade(&first);
        let second_weak = Rc::downgrade(&second);
        let mut data = OrderedCollection::new(CollectionKind::Map);
        data.insert(Value::Obj(key.clone()), Value::Obj(first));
        assert_eq!(Rc::strong_count(&key), 2);
        data.insert(Value::Obj(key.clone()), Value::Obj(second));
        assert_eq!(Rc::strong_count(&key), 2, "index never owns another key");
        assert!(
            first_weak.upgrade().is_none(),
            "replaced payload drops once"
        );
        assert!(second_weak.upgrade().is_some());
        assert_eq!(data.len(), 1);
        assert!(data.delete(&Value::Obj(key.clone())));
        assert_eq!(Rc::strong_count(&key), 1);
        assert!(
            second_weak.upgrade().is_none(),
            "deleted payload drops once"
        );

        // Equal separately allocated strings hit the original entry, not a second bucket.
        let text = "wide string key ".repeat(128);
        let first_key = Value::lstr(text.clone());
        let another_key = Value::lstr(text);
        data.insert(first_key, Value::Num(7.0));
        let before = KEY_HASH_CALLS.with(Cell::get);
        data.insert(another_key.clone(), Value::Num(8.0));
        assert_eq!(KEY_HASH_CALLS.with(Cell::get) - before, 1);
        assert_eq!(data.len(), 1);
        assert_eq!(number(data.get(&another_key).unwrap()), 8);
        assert!(data.delete(&another_key));
    }

    #[test]
    fn ordered_collection_storage_bounded_with_dormant_cursors() {
        let mut data = OrderedCollection::new(CollectionKind::Map);
        let dormant = data.cursor();
        let advancing = data.cursor();
        for round in 0..500 {
            for key in 0..256 {
                data.insert(Value::Num(key as f64), Value::Num(round as f64));
            }
            let _ = data.next(&advancing);
            data.clear();
            assert!(data.entries.capacity() <= 64);
            assert!(data.index.capacity() <= 64);
            assert_eq!(dormant.0.position.get(), 0);
        }
        for key in 0..10_000 {
            data.insert(Value::Num(key as f64), Value::Num(key as f64));
        }
        for key in 0..9_997 {
            assert!(data.delete(&Value::Num(key as f64)));
            assert!(data.entries.len() <= 2 * data.len() + 32);
        }
        assert!(data.entries.capacity() <= 256);
        assert!(data.index.capacity() <= 256);
        assert_eq!(number(&data.next(&dormant).unwrap().0), 9_997);
        for key in 10_000..30_000 {
            assert!(data.delete(&Value::Num((key - 3) as f64)));
            data.insert(Value::Num(key as f64), Value::Num(key as f64));
            assert!(data.entries.len() <= 2 * data.len() + 32);
        }
        assert_eq!(number(&data.next(&dormant).unwrap().0), 29_997);
    }

    #[test]
    fn ordered_collection_cursor_registry_bounded() {
        let mut data = OrderedCollection::new(CollectionKind::Set);
        let mut live = Vec::new();
        for _ in 0..1000 {
            live.push(data.cursor());
        }
        for _ in 0..10_000 {
            drop(data.cursor());
            assert_eq!(data.cursors.as_ref().unwrap().borrow().len(), live.len());
        }
        live.clear();
        assert!(data.cursors.as_ref().unwrap().borrow().is_empty());
        assert!(data.cursors.as_ref().unwrap().borrow().capacity() <= 32);
        for _ in 0..10_000 {
            drop(data.cursor());
            assert!(data.cursors.as_ref().unwrap().borrow().is_empty());
        }
    }

    #[test]
    fn ordered_collection_cursor_clones_unregister_once_and_outlive_collection() {
        let mut data = OrderedCollection::new(CollectionKind::Map);
        let first = data.cursor();
        let middle = data.cursor();
        let last = data.cursor();
        let cloned = middle.clone();
        drop(middle);
        assert_eq!(data.cursors.as_ref().unwrap().borrow().len(), 3);
        drop(first); // swap-removes last and repairs its slot
        assert_eq!(last.0.slot.get(), 0);
        drop(cloned);
        assert_eq!(data.cursors.as_ref().unwrap().borrow().len(), 1);
        drop(data);
        drop(last); // weak backlink must not retain or reborrow a dead registry
    }

    #[test]
    fn ordered_collection_randomized_live_cursor_model() {
        // The reference deliberately retains every deleted historical slot, exactly as the
        // specification List does. Implementation storage must stay bounded despite that history.
        let mut data = OrderedCollection::new(CollectionKind::Map);
        let mut model: Vec<Option<(u32, u32)>> = Vec::new();
        let mut cursors: Vec<_> = (0..8).map(|_| (data.cursor(), 0usize)).collect();
        let mut seed = 0x93a6_27c1u32;
        for turn in 0..40_000 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let key = (seed >> 8) % 64;
            match seed % 20 {
                0 => {
                    data.clear();
                    model.fill(None);
                }
                1..=7 => {
                    let old = model
                        .iter()
                        .position(|entry| entry.is_some_and(|entry| entry.0 == key));
                    assert_eq!(data.delete(&Value::Num(key as f64)), old.is_some());
                    if let Some(old) = old {
                        model[old] = None;
                    }
                }
                8..=13 => {
                    data.insert(Value::Num(key as f64), Value::Num(turn as f64));
                    if let Some(entry) = model
                        .iter_mut()
                        .find(|entry| entry.is_some_and(|entry| entry.0 == key))
                    {
                        *entry = Some((key, turn));
                    } else {
                        model.push(Some((key, turn)));
                    }
                }
                14 => {
                    cursors[(key % 8) as usize] = (data.cursor(), 0);
                }
                _ => {
                    let (cursor, model_cursor) = &mut cursors[(key % 8) as usize];
                    let mut expected = None;
                    while *model_cursor < model.len() {
                        let entry = model[*model_cursor];
                        *model_cursor += 1;
                        if entry.is_some() {
                            expected = entry;
                            break;
                        }
                    }
                    let actual = data
                        .next(cursor)
                        .map(|(key, value)| (number(&key), number(&value)));
                    assert_eq!(actual, expected, "model step {turn}");
                }
            }
            let expected: Vec<_> = model.iter().copied().flatten().collect();
            let actual: Vec<_> = data
                .iter()
                .map(|(key, value)| (number(key), number(value)))
                .collect();
            assert_eq!(actual, expected);
            assert_eq!(data.len(), expected.len());
            assert!(data.entries.len() <= 2 * data.len() + 32);
        }
    }
}
