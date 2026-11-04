//! An element block of 8-byte words (W2c step 2b; the plan's "W2c step 2: words"), as
//! SpiderMonkey keeps a native object's elements (vm/NativeObject.h, `ObjectElements`): one
//! storage block of a 16-byte header and `capacity` words of Value bits, a hole being the
//! [`PACK_EMPTY`] word. A map holds a pointer to the first word, with the header just before it,
//! and never a null one: a map without elements points just past the static empty block's
//! header, so generated code reads a header and bounds-checks without testing for null.
//!
//! Shifts leave front room before the header (`shifted`, SpiderMonkey's `numShiftedElements`):
//! `shift` moves the header one word forward instead of copying the words, and `unshift` takes
//! that room back first. The header's `flags` hold the kind bits, facts about the used words
//! that the block's own operations keep exact (W2c's design 2), and that generated code and the
//! builtins may rely on.
//!
//! A block is plain data in the heap's storage pages (`gc_storage`): dropping a [`Words`] reads
//! nothing, the owner reports the block and traces its words ([`Words::trace`]), and growth just
//! drops the old block's address for the collector to free.
//!
//! A block of more than [`REMEMBER_MIN`] words keeps a remembered record after its last word: the
//! length its last full mark traced, and a dirty bit for each [`DIRTY_BLOCK`] words below it that
//! an edge has been written to since. A remembered (minor) trace of an old array then reports the
//! dirty runs and the words past the traced length, not every word, as P1's PackedBuffer did
//! (S6). Every write of an edge goes through [`Words::set`], growth that keeps the words' indices
//! keeps the record, and every reordering forces the next trace to report all words.
//!
//! A constant array literal's copies share one block (W2d step 4; V8's fixed_cow_array_map, JSC's
//! CopyOnWrite indexing type): the site makes it once, marked [`KIND_COW`], and every copy points
//! at it ([`Words::share`]). No owner writes a shared block, not a word, the header or a record
//! (it keeps none): a map that changes its elements takes a private copy first
//! ([`Words::unshare`]), and generated code tests the bit with FROZEN's, so its stores send a
//! shared block to the helpers, which copy.
use super::*;
use std::ptr::NonNull;

/// No used word is a hole.
pub(crate) const KIND_NO_HOLES: u32 = 1 << 0;
/// Every used word that is not a hole is a Number.
pub(crate) const KIND_NUMBERS: u32 = 1 << 1;
/// Every used word that is not a hole is an exact int32 (bit-identical through an i32, so not
/// -0).
pub(crate) const KIND_INT32: u32 = 1 << 2;
/// The facts of a block with no words.
pub(crate) const KIND_ALL: u32 = KIND_NO_HOLES | KIND_NUMBERS | KIND_INT32;
/// The integrity levels (W2c step 7a; SpiderMonkey's elements flags, V8's elements kinds), not
/// facts about the words' values but their elements' attributes: every element is
/// non-configurable (SEALED), and also non-writable (FROZEN, which comes with SEALED). Set once
/// on a block the object owns (never the static one), by SetIntegrityLevel; never cleared. A
/// sealed block's words take overwrites only; a frozen one's take no write.
///
/// FROZEN and COW lie between INT32 and SCANNED so that `INT32 | FROZEN | COW` and
/// `FROZEN | COW | SCANNED` are each one run of bits, a single logical immediate: a generated
/// Number store tests both within its kind update's own tests
/// (`emit_word_number_kind_unfrozen` in jit.rs), and every other generated write tests
/// [`KIND_UNWRITABLE`], one run too.
pub(crate) const KIND_FROZEN: u32 = 1 << 3;
/// Not a fact about the words or their attributes: the block is a constant literal's, shared by
/// the literal's site and every copy (see the module comment), so no owner writes it. A shared
/// block is exact (`capacity` = `used`), holds no hole, has no front room and no remembered
/// record, and its kind bits are exact and [`KIND_SCANNED`] for good. Set only by
/// [`Words::shared_literal`]; a private copy ([`Words::unshare`]) drops it.
pub(crate) const KIND_COW: u32 = 1 << 4;
/// The words no generated code writes in line: frozen elements, or a shared block (the ARM64
/// templates' test; the x86-64 backend writes no element word in line).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) const KIND_UNWRITABLE: u32 = KIND_FROZEN | KIND_COW;
/// Not a fact about the words: a scan proved the kind bits ([`Words::reprove`]) and no write
/// since could have made a cleared fact true again (a write that keeps a fact can only clear
/// it), so a cleared kind bit is exact and another proof need not scan. Every write through
/// [`Words::set`] and every removal clears it.
pub(crate) const KIND_SCANNED: u32 = 1 << 5;
pub(crate) const KIND_SEALED: u32 = 1 << 6;
pub(crate) const KIND_INTEGRITY: u32 = KIND_SEALED | KIND_FROZEN;

/// The 16 bytes before a block's first word. Generated code reads `used` at -16, `capacity` at
/// -12 and `flags` at -4 from the words' address.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WordHeader {
    /// The words in use: an array's elements are words `0..used`.
    pub(crate) used: u32,
    /// The words the block has room for from the first one.
    pub(crate) capacity: u32,
    /// The words of front room before the header, which shifts left.
    pub(crate) shifted: u32,
    /// The kind bits.
    pub(crate) flags: u32,
}

const HEADER_BYTES: usize = std::mem::size_of::<WordHeader>();
const _: () = assert!(HEADER_BYTES == 16 && std::mem::align_of::<WordHeader>() <= 8);

#[repr(C, align(16))]
struct Empty {
    header: WordHeader,
}

static EMPTY: Empty = Empty {
    header: WordHeader {
        used: 0,
        capacity: 0,
        shifted: 0,
        flags: KIND_ALL,
    },
};

/// The largest block: its `used` and `capacity` must fit their u32 fields.
const MAX_WORDS: usize = u32::MAX as usize >> 1;

/// A block with room for more words than this, front room included, keeps a remembered record.
pub(crate) const REMEMBER_MIN: usize = 64;
/// The words one dirty bit covers.
pub(crate) const DIRTY_BLOCK: usize = 64;
/// A record's traced length when the next trace must report every word.
const TRACED_NONE: u64 = u64::MAX;

/// The record's words after a block with room for `words` words: none for a small block, else
/// the traced length and the dirty bitmap.
pub(crate) fn record_words(words: usize) -> usize {
    if words <= REMEMBER_MIN {
        0
    } else {
        1 + words.div_ceil(DIRTY_BLOCK).div_ceil(64)
    }
}

#[cfg(test)]
thread_local! {
    /// The words a marker's trace visited, full and remembered.
    pub(crate) static MARK_VISITS: std::cell::Cell<[usize; 2]> =
        const { std::cell::Cell::new([0; 2]) };
}

/// A map's element words (see the module comment).
pub(crate) struct Words(NonNull<u64>);

impl Words {
    /// The static empty block.
    #[inline]
    pub(crate) fn new() -> Self {
        // SAFETY: one past the static header, the address of a zero-length run of words.
        Words(unsafe { NonNull::from(&EMPTY.header).add(1).cast() })
    }

    /// A block with room for `capacity` words, or the static empty one for none.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Words::new();
        }
        Words::fresh(capacity)
    }

    /// A new block with room for `capacity` words and none used: its header, with every kind
    /// bit (no word breaks one), and past 64 words a record whose next trace reports every
    /// word. Literals and host lists start here, without the growth path's copy.
    fn fresh(capacity: usize) -> Self {
        assert!(capacity <= MAX_WORDS, "element block of {capacity} words");
        let record = record_words(capacity);
        let block = crate::gc_storage::alloc_block(block_size(capacity + record));
        // SAFETY: a fresh block of a header, `capacity` words and the record.
        unsafe {
            let words = block.as_ptr().add(HEADER_BYTES).cast::<u64>();
            words.cast::<WordHeader>().sub(1).write(WordHeader {
                used: 0,
                capacity: capacity as u32,
                shifted: 0,
                flags: KIND_ALL,
            });
            if record != 0 {
                words.add(capacity).write(TRACED_NONE);
                std::ptr::write_bytes(words.add(capacity + 1), 0, record - 1);
            }
            Words(NonNull::new_unchecked(words))
        }
    }

    /// An exact block of the words of `values` (an array literal's evaluated window, a host
    /// list) with their kind bits. Unlike [`Words::from_values`], each loop has a single bound,
    /// the slice's, however the caller is inlined.
    #[inline]
    pub(crate) fn from_words(values: &[u64]) -> Self {
        let capacity = values.len();
        if capacity > SHORT_WORDS {
            return Self::from_long_words(values);
        }
        let mut words = Words::with_capacity(capacity);
        if capacity == 0 {
            return words;
        }
        let mut flags = KIND_ALL;
        let first = words.0.as_ptr();
        for (index, &word) in values.iter().enumerate() {
            flags &= word_kind(word);
            // SAFETY: a fresh heap block with room for `capacity` words. The store is volatile
            // so that the loop stays one short scalar loop: LLVM would split it into a memcpy
            // call and a vectorized fold, whose fixed cost exceeds a short literal's copy.
            unsafe { first.add(index).write_volatile(word) };
        }
        let header = words.header_mut();
        header.used = capacity as u32;
        header.flags = flags;
        words
    }

    /// [`Words::from_words`] past [`SHORT_WORDS`]: one copy, then the kind bits folded over it
    /// (vectorized), out of line.
    #[inline(never)]
    fn from_long_words(values: &[u64]) -> Self {
        let capacity = values.len();
        let mut words = Words::with_capacity(capacity);
        // SAFETY: a fresh heap block with room for `capacity` words, which `values` (a slice
        // elsewhere) does not overlap.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), words.0.as_ptr(), capacity) };
        let flags = values
            .iter()
            .fold(KIND_ALL, |flags, &word| flags & word_kind(word));
        let header = words.header_mut();
        header.used = capacity as u32;
        header.flags = flags;
        words
    }

    /// The shared block of a constant array literal's `values` (see [`KIND_COW`]): exact, with
    /// their kind bits, scanned, and no remembered record. `values` is not empty and holds no
    /// hole.
    pub(crate) fn shared_literal(values: &[u64]) -> Self {
        debug_assert!(!values.is_empty() && !values.contains(&PACK_EMPTY));
        let capacity = values.len();
        assert!(capacity <= MAX_WORDS, "element block of {capacity} words");
        let flags = values
            .iter()
            .fold(KIND_ALL, |flags, &word| flags & word_kind(word));
        let block = crate::gc_storage::alloc_block(block_size(capacity));
        // SAFETY: a fresh block of a header and `capacity` words, which `values` (a slice
        // elsewhere) does not overlap; a shared block keeps no record.
        unsafe {
            let words = block.as_ptr().add(HEADER_BYTES).cast::<u64>();
            std::ptr::copy_nonoverlapping(values.as_ptr(), words, capacity);
            words.cast::<WordHeader>().sub(1).write(WordHeader {
                used: capacity as u32,
                capacity: capacity as u32,
                shifted: 0,
                flags: flags | KIND_SCANNED | KIND_COW,
            });
            Words(NonNull::new_unchecked(words))
        }
    }

    /// Another owner of this shared block (see [`KIND_COW`]).
    #[inline]
    pub(crate) fn share(&self) -> Self {
        debug_assert!(self.is_shared(), "only a shared block has two owners");
        Words(self.0)
    }

    /// Whether the block is a constant literal's shared one ([`KIND_COW`]), which no owner
    /// writes.
    #[inline]
    pub(crate) fn is_shared(&self) -> bool {
        self.header().flags & KIND_COW != 0
    }

    /// Make the block this owner's own before a write in place: a shared block's words, kind
    /// bits and scan go into an exact private block, whose next trace reports every word (V8's
    /// CopyElementsOnWrite, JSC's convertFromCopyOnWrite). A private block stays. Every write
    /// that keeps the words where they are calls it first; an append needs no call, since a
    /// shared block is exact and its growth makes a private block (V8's kGrowAndHandleCOW).
    #[inline]
    fn unshare(&mut self) {
        if self.is_shared() {
            self.copy_shared();
        }
    }

    #[inline(never)]
    fn copy_shared(&mut self) {
        let len = self.len();
        let flags = self.header().flags & !KIND_COW;
        let mut copy = Words::fresh(len);
        // SAFETY: a fresh block with room for the `len` used words of another block.
        unsafe { std::ptr::copy_nonoverlapping(self.0.as_ptr(), copy.0.as_ptr(), len) };
        let header = copy.header_mut();
        header.used = len as u32;
        header.flags = flags;
        *self = copy;
    }

    /// The words a shared block keeps after a removal (pop, shift, truncate), in an exact
    /// private block, or the static empty one for none: one copy, not a copy then a removal.
    fn kept(words: &[u64]) -> Self {
        Words::from_words(words)
    }

    /// An exact block of `values`, written in one pass with their kind bits (array literals
    /// and host lists build their elements here).
    pub(crate) fn from_values(values: impl ExactSizeIterator<Item = Value>) -> Self {
        let capacity = values.len();
        let mut words = Words::with_capacity(capacity);
        if capacity == 0 {
            return words;
        }
        let mut flags = KIND_ALL;
        let mut used = 0;
        // SAFETY: a heap block with room for `capacity` words; no more are written, whatever
        // the iterator's length claimed.
        unsafe {
            let first = words.0.as_ptr();
            for value in values.take(capacity) {
                let word = value.bits();
                flags &= word_kind(word);
                first.add(used).write(word);
                used += 1;
            }
        }
        let header = words.header_mut();
        header.used = used as u32;
        header.flags = flags;
        words
    }

    #[inline]
    fn header(&self) -> &WordHeader {
        // SAFETY: every Words points just past a header, its block's or the static one's.
        unsafe { &*self.0.as_ptr().cast::<WordHeader>().sub(1) }
    }

    #[inline]
    fn header_mut(&mut self) -> &mut WordHeader {
        debug_assert!(
            !self.is_static_empty(),
            "the static empty block is never written"
        );
        debug_assert!(!self.is_shared(), "a shared block is written");
        // SAFETY: a heap block's header, which `&mut self` holds uniquely.
        unsafe { &mut *self.0.as_ptr().cast::<WordHeader>().sub(1) }
    }

    /// The address of the first word (the static empty block's for none), as a map's word
    /// holds it (W2e step 5's array images).
    #[inline]
    pub(crate) fn words_address(&self) -> usize {
        self.0.as_ptr() as usize
    }

    /// Whether this is the static empty block, which holds no words and is never written.
    #[inline]
    pub(crate) fn is_static_empty(&self) -> bool {
        std::ptr::eq(self.header(), &EMPTY.header)
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.header().used as usize
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.header().capacity as usize
    }

    /// The words of front room before the header, which shifts left.
    #[inline]
    pub(crate) fn front_room(&self) -> usize {
        self.header().shifted as usize
    }

    /// The used words that are no hole: the used length when NO_HOLES holds, else a count.
    pub(crate) fn present(&self) -> usize {
        if self.kind() & KIND_NO_HOLES != 0 {
            self.len()
        } else {
            self.as_slice()
                .iter()
                .filter(|&&word| word != PACK_EMPTY)
                .count()
        }
    }

    /// The integrity bits ([`KIND_SEALED`], [`KIND_FROZEN`]): every element's attributes.
    #[inline]
    pub(crate) fn integrity(&self) -> u32 {
        self.header().flags & KIND_INTEGRITY
    }

    /// Whether the block has an integrity level: every level sets [`KIND_SEALED`] (FROZEN
    /// comes with it), so one bit answers, a single test-bit branch.
    #[inline]
    pub(crate) fn sealed(&self) -> bool {
        self.header().flags & KIND_SEALED != 0
    }

    /// Give every element the integrity level `bits` (sealed, or frozen and sealed). The
    /// static block has no element to describe and is never written.
    pub(crate) fn set_integrity(&mut self, bits: u32) {
        self.unshare();
        debug_assert!(bits == KIND_SEALED || bits == KIND_INTEGRITY);
        if !self.is_static_empty() {
            self.header_mut().flags |= bits | KIND_SEALED;
        }
    }

    /// The kind bits of the used words.
    #[inline]
    pub(crate) fn kind(&self) -> u32 {
        self.header().flags & KIND_ALL
    }

    /// The kind and integrity bits and the shared-literal bit, for the element record (W4 step 4).
    #[inline]
    pub(crate) fn facts(&self) -> u32 {
        self.header().flags & (KIND_ALL | KIND_FROZEN | KIND_COW)
    }

    /// Whether a scan made the kind bits exact ([`KIND_SCANNED`]).
    #[inline]
    pub(crate) fn scanned(&self) -> bool {
        self.header().flags & KIND_SCANNED != 0
    }

    /// The used words.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u64] {
        // SAFETY: `used` initialized words follow the header.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), self.len()) }
    }

    /// The element at `index`: `None` past the used words or for a hole.
    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<Value> {
        let word = *self.as_slice().get(index)?;
        // SAFETY: a used word is a Value's bits, written by `set`, `push` or `unshift`.
        (word != PACK_EMPTY).then(|| unsafe { Value::from_bits(word) })
    }

    /// Store `value` at `index`, below the used length, keeping the kind bits and the record.
    pub(crate) fn set(&mut self, index: usize, value: Value) {
        self.unshare();
        assert!(index < self.len(), "word index {index} past {}", self.len());
        debug_assert!(
            self.integrity() & KIND_FROZEN == 0,
            "a frozen element is written"
        );
        let word = value.bits();
        self.note(word);
        self.remember(index, word);
        // SAFETY: a used word of a heap block (the static block has none).
        unsafe { self.0.as_ptr().add(index).write(word) };
    }

    /// Leave a hole at `index`, below the used length, as its element moves to a named entry
    /// with attributes a word cannot keep (a define, or a descriptor escape). Not a write of the
    /// element: a sealed or frozen block allows it (W2c step 7a).
    pub(crate) fn move_out(&mut self, index: usize) {
        self.unshare();
        assert!(index < self.len(), "word index {index} past {}", self.len());
        self.note(PACK_EMPTY);
        // SAFETY: a used word of a heap block (the static block has none).
        unsafe { self.0.as_ptr().add(index).write(PACK_EMPTY) };
    }

    /// Append `value`, growing the block when it is full (a shared block always is).
    pub(crate) fn push(&mut self, value: Value) {
        debug_assert!(self.integrity() == 0, "a sealed block changes its elements");
        let len = self.len();
        if len == self.capacity() {
            self.reallocate(grown(len, 1), 0);
        }
        debug_assert!(!self.is_shared(), "a shared block is written");
        let word = value.bits();
        self.note(word);
        // SAFETY: below capacity, so inside the block.
        unsafe { self.0.as_ptr().add(len).write(word) };
        self.header_mut().used += 1;
    }

    /// Remove and return the last word (a hole's too).
    pub(crate) fn pop(&mut self) -> Option<u64> {
        debug_assert!(self.integrity() == 0, "a sealed block changes its elements");
        let len = self.len().checked_sub(1)?;
        let word = self.as_slice()[len];
        if self.is_shared() {
            // The words kept, in a private block of their own.
            *self = Words::kept(&self.as_slice()[..len]);
            return Some(word);
        }
        let header = self.header_mut();
        header.used = len as u32;
        header.flags &= !KIND_SCANNED;
        self.lower_traced(len);
        Some(word)
    }

    /// Remove and return the first word, leaving its room in front of the header.
    pub(crate) fn shift(&mut self) -> Option<u64> {
        debug_assert!(self.integrity() == 0, "a sealed block changes its elements");
        if self.is_empty() {
            return None;
        }
        let first = self.as_slice()[0];
        if self.is_shared() {
            *self = Words::kept(&self.as_slice()[1..]);
            return Some(first);
        }
        let header = *self.header();
        // Every word's index moves: the next trace reports them all.
        self.force_full();
        // SAFETY: the header moves one word forward, onto the word just read; the words after
        // it stay where they are.
        unsafe {
            let words = self.0.as_ptr().add(1);
            words.cast::<WordHeader>().sub(1).write(WordHeader {
                used: header.used - 1,
                capacity: header.capacity - 1,
                shifted: header.shifted + 1,
                flags: header.flags & !KIND_SCANNED,
            });
            self.0 = NonNull::new_unchecked(words);
        }
        Some(first)
    }

    /// Insert `values` before the first word, in front room when there is enough.
    pub(crate) fn unshift(&mut self, values: &[Value]) {
        debug_assert!(self.integrity() == 0, "a sealed block changes its elements");
        let n = values.len();
        if n == 0 {
            return;
        }
        // A shared block has no front room, so it takes the reallocation below, whose block is
        // private.
        let header = *self.header();
        let flags = values
            .iter()
            .fold(header.flags & !KIND_COW, |flags, value| {
                flags & word_kind(value.bits())
            });
        if !self.is_static_empty() && header.shifted as usize >= n {
            // Every word's index moves: the next trace reports them all.
            self.force_full();
            // SAFETY: the front room holds `n` words and the header before them; the new
            // header is written before the words that may cover the old one.
            unsafe {
                let words = self.0.as_ptr().sub(n);
                words.cast::<WordHeader>().sub(1).write(WordHeader {
                    used: header.used + n as u32,
                    capacity: header.capacity + n as u32,
                    shifted: header.shifted - n as u32,
                    flags,
                });
                for (i, value) in values.iter().enumerate() {
                    words.add(i).write(value.bits());
                }
                self.0 = NonNull::new_unchecked(words);
            }
            return;
        }
        // Front room of at least the used words again, as P1's deque prepend kept, so repeated
        // unshifts move the words once per doubling (SpiderMonkey's tryUnshiftDenseElements
        // reserves room ahead the same way).
        let len = self.len();
        self.reallocate_with_room(grown(len, n), n, len.max(8));
        // SAFETY: the new block has `n` words of room before the copied ones.
        unsafe {
            for (i, value) in values.iter().enumerate() {
                self.0.as_ptr().add(i).write(value.bits());
            }
        }
        let header = self.header_mut();
        header.used = (len + n) as u32;
        header.flags = flags;
    }

    /// Reverse the used words in place; their kind bits stay true.
    pub(crate) fn reverse(&mut self) {
        self.unshare();
        debug_assert!(
            self.integrity() & KIND_FROZEN == 0,
            "a frozen element is written"
        );
        let len = self.len();
        if len > 1 {
            // Words change places: the next trace reports them all.
            self.force_full();
            // SAFETY: the used words of a heap block, which `&mut self` holds uniquely.
            unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr(), len) }.reverse();
        }
    }

    /// Replace the words in `range` by `values`, returning the values removed (holes as
    /// `None`), as `Vec::splice` does: the words after `range` move once, in place. The words
    /// from `range.start` on count as untraced for the remembered record; the kind bits keep
    /// what the inserted values keep, and a removal may make a cleared fact true, so a scan's
    /// exactness goes.
    pub(crate) fn splice(
        &mut self,
        range: std::ops::Range<usize>,
        values: &[Value],
    ) -> Vec<Option<Value>> {
        debug_assert!(self.integrity() == 0, "a sealed block changes its elements");
        let len = self.len();
        assert!(
            range.start <= range.end && range.end <= len,
            "splice {range:?} of {len}"
        );
        let removed = self.as_slice()[range.clone()]
            .iter()
            // SAFETY: a used word is a Value's bits.
            .map(|&word| (word != PACK_EMPTY).then(|| unsafe { Value::from_bits(word) }))
            .collect();
        let new_len = len - range.len() + values.len();
        // A shared block (exact) grows into a private one; one that does not grow is copied.
        if new_len > self.capacity() {
            self.reserve(new_len - len);
        }
        self.unshare();
        if self.is_static_empty() {
            return removed;
        }
        let flags = values
            .iter()
            .fold(self.header().flags & !KIND_SCANNED, |flags, value| {
                flags & (word_kind(value.bits()) | !KIND_ALL)
            })
            & !KIND_SCANNED;
        // SAFETY: the block has room for `new_len` words; the suffix moves within it (the
        // ranges may overlap), then the values fill the gap before it.
        unsafe {
            let words = self.0.as_ptr();
            std::ptr::copy(
                words.add(range.end),
                words.add(range.start + values.len()),
                len - range.end,
            );
            for (i, value) in values.iter().enumerate() {
                words.add(range.start + i).write(value.bits());
            }
        }
        let header = self.header_mut();
        header.used = new_len as u32;
        header.flags = flags;
        self.lower_traced(range.start);
        removed
    }

    /// Keep the first `len` words.
    pub(crate) fn truncate(&mut self, len: usize) {
        if self.is_shared() {
            if len < self.len() {
                *self = Words::kept(&self.as_slice()[..len]);
            }
            return;
        }
        debug_assert!(
            self.integrity() == 0
                || self
                    .as_slice()
                    .get(len..)
                    .is_none_or(|tail| tail.iter().all(|&w| w == PACK_EMPTY)),
            "a sealed block loses an element"
        );
        if len < self.len() {
            let header = self.header_mut();
            header.used = len as u32;
            header.flags &= !KIND_SCANNED;
            self.lower_traced(len);
        }
    }

    /// Lengthen to `len` words with holes after the used ones.
    pub(crate) fn resize_with_holes(&mut self, len: usize) {
        debug_assert!(self.integrity() == 0, "a sealed block changes its elements");
        let used = self.len();
        if len <= used {
            return;
        }
        // A shared block is exact, so it grows into a private one here.
        if len > self.capacity() {
            self.reallocate(grown(used, len - used), 0);
        }
        debug_assert!(!self.is_shared(), "a shared block is written");

        // SAFETY: below capacity.
        unsafe {
            for i in used..len {
                self.0.as_ptr().add(i).write(PACK_EMPTY);
            }
        }
        let header = self.header_mut();
        header.used = len as u32;
        header.flags &= !KIND_NO_HOLES;
    }

    /// Room for exactly `additional` more words, when there is less.
    pub(crate) fn reserve_exact(&mut self, additional: usize) {
        let len = self.len();
        if self.capacity() - len < additional {
            self.reallocate(len + additional, 0);
        }
    }

    /// Room for `additional` more words without growing.
    pub(crate) fn reserve(&mut self, additional: usize) {
        let len = self.len();
        if self.capacity() - len < additional {
            self.reallocate(grown(len, additional), 0);
        }
    }

    /// The kind bits of the used words as they are now, by a scan.
    pub(crate) fn computed_kind(&self) -> u32 {
        self.as_slice()
            .iter()
            .fold(KIND_ALL, |flags, &word| flags & word_kind(word))
    }

    /// Prove the kind bits again from the words, which a cleared bit needs to come back; the
    /// block is then [`KIND_SCANNED`].
    pub(crate) fn reprove(&mut self) -> u32 {
        let flags = self.computed_kind();
        // A shared block's bits are exact and scanned for good, and it is never written.
        if !self.is_static_empty() && !self.is_shared() {
            let integrity = self.integrity();
            self.header_mut().flags = flags | KIND_SCANNED | integrity;
        }
        flags
    }

    /// The block's address for the collector, front room included; `None` for the static one.
    #[inline]
    pub(crate) fn block_start(&self) -> Option<*const u8> {
        if self.is_static_empty() {
            return None;
        }
        let front = HEADER_BYTES + 8 * self.header().shifted as usize;
        // SAFETY: the block begins `front` bytes before the first word.
        Some(unsafe { self.0.as_ptr().cast::<u8>().sub(front) })
    }

    /// Report the block and the edges of its words (a shared literal's site traces its block
    /// so; maps report the block and trace the words apart).
    pub(crate) fn trace<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(start) = self.block_start() {
            crate::gc_storage::trace_block(start, sink);
        }
        self.trace_values(None, sink);
    }

    /// Report the edges of the words, not the block. A Numbers block has none. A marker's
    /// remembered trace reports only the dirty runs below the traced length and the words past
    /// it; every other trace (a marker's full one, or `None`'s, the verifiers' and the
    /// heap walks') reports every word. The kind bits are proofs that this trace (NUMBERS: no
    /// edge) and generated code trust, kept by every write in Rust and in generated code, so
    /// debug builds check at every trace that each one set holds of the words (W2c step 5): a
    /// NUMBERS bit left set over a reference would hide it from the collector.
    pub(crate) fn trace_values<S: crate::gc_trace::GcSink + ?Sized>(
        &self,
        purpose: Option<crate::gc_core::TracePurpose>,
        sink: &mut S,
    ) {
        debug_assert_eq!(
            self.kind() & KIND_ALL & !self.computed_kind(),
            0,
            "element kind bits {:#x} claim more than the words hold ({:#x})",
            self.kind(),
            self.computed_kind()
        );
        if self.kind() & KIND_NUMBERS != 0 {
            return;
        }
        let words = self.as_slice();
        let mut report = |range: std::ops::Range<usize>| {
            #[cfg(test)]
            if let Some(purpose) = purpose {
                MARK_VISITS.with(|visits| {
                    let mut counts = visits.get();
                    counts[usize::from(purpose == crate::gc_core::TracePurpose::Remembered)] +=
                        range.len();
                    visits.set(counts);
                });
            }
            for &word in &words[range] {
                if word != PACK_EMPTY {
                    // SAFETY: a used word is a Value's bits.
                    sink.value(&unsafe { Value::from_bits(word) });
                }
            }
        };
        let traced = match (purpose, self.record()) {
            // SAFETY: a record's first word is its traced length.
            (Some(crate::gc_core::TracePurpose::Remembered), Some(record)) => unsafe {
                record.read()
            },
            _ => TRACED_NONE,
        };
        let len = words.len();
        if traced == TRACED_NONE {
            report(0..len);
            return;
        }
        let prefix = (traced as usize).min(len);
        let record = self.record().unwrap();
        let bitmap = self.bitmap_words();
        // The dirty blocks below the prefix, adjacent ones merged into one run.
        let mut run: Option<std::ops::Range<usize>> = None;
        'bits: for bit_word in 0..bitmap {
            // SAFETY: the bitmap follows the traced length, `bitmap` words of it.
            let mut bits = unsafe { record.add(1 + bit_word).read() };
            while bits != 0 {
                let block = bit_word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let start = block * DIRTY_BLOCK;
                if start >= prefix {
                    break 'bits;
                }
                let end = (start + DIRTY_BLOCK).min(prefix);
                match &mut run {
                    Some(open) if open.end == start => open.end = end,
                    Some(open) => {
                        report(open.clone());
                        *open = start..end;
                    }
                    None => run = Some(start..end),
                }
            }
        }
        match run {
            Some(open) if open.end == prefix => report(open.start..len),
            Some(open) => {
                report(open);
                report(prefix..len);
            }
            None => report(prefix..len),
        }
    }

    /// After a marker reported every edge of the owner: the used words are traced and clean.
    #[inline]
    pub(crate) fn finish_mark_trace(&self) {
        if self.is_static_empty() {
            return;
        }
        let Some(record) = self.record() else {
            return;
        };
        // SAFETY: the record's words, which no reference covers (see `record`).
        unsafe {
            record.write(self.len() as u64);
            std::ptr::write_bytes(record.add(1).as_ptr(), 0, self.bitmap_words());
        }
    }

    /// Make the next trace report every word: the block's words were moved to another owner,
    /// or changed places.
    #[inline]
    pub(crate) fn force_full(&self) {
        if self.is_static_empty() {
            return;
        }
        if let Some(record) = self.record() {
            // SAFETY: as in `finish_mark_trace`.
            unsafe { record.write(TRACED_NONE) };
        }
    }

    /// The traced length (`None` for every word next) and the dirty blocks, for tests.
    #[cfg(test)]
    pub(crate) fn remembered(&self) -> Option<(Option<usize>, Vec<usize>)> {
        let record = self.record()?;
        // SAFETY: as in `trace_values`.
        let traced = unsafe { record.read() };
        let blocks = (0..self.bitmap_words() * 64)
            .filter(
                |&block| unsafe { record.add(1 + block / 64).read() } & (1 << (block % 64)) != 0,
            )
            .collect();
        Some(((traced != TRACED_NONE).then_some(traced as usize), blocks))
    }

    /// The bytes of the block: front room, header, words and record.
    pub(crate) fn block_bytes(&self) -> usize {
        if self.is_static_empty() {
            return 0;
        }
        let words = self.block_words();
        let record = if self.is_shared() {
            0
        } else {
            record_words(words)
        };
        HEADER_BYTES + 8 * (words + record)
    }

    /// The words the block has room for, front room included (a shift moves the first word and
    /// the header, never the block's end).
    #[inline]
    fn block_words(&self) -> usize {
        let header = self.header();
        header.capacity as usize + header.shifted as usize
    }

    /// The remembered record, just past the block's last word, for a block that keeps one. It
    /// is read and written only through this pointer, never a reference, so a shared `Words`
    /// may update it (the marker's `finish_mark_trace`).
    #[inline]
    fn record(&self) -> Option<NonNull<u64>> {
        // SAFETY: `capacity` words on from the first word is the block's end, where the record
        // begins (`reallocate`); the static block and a shared one keep none.
        (self.block_words() > REMEMBER_MIN && !self.is_shared())
            .then(|| unsafe { self.0.add(self.header().capacity as usize) })
    }

    /// The dirty bitmap's words in the record.
    #[inline]
    fn bitmap_words(&self) -> usize {
        record_words(self.block_words()).saturating_sub(1)
    }

    /// Record a write of `word` at `index` for the next remembered trace: an edge (not a Number
    /// or a hole) below the traced length marks its dirty block.
    #[inline]
    fn remember(&mut self, index: usize, word: u64) {
        if word_kind(word) & KIND_NUMBERS != 0 {
            return;
        }
        let Some(record) = self.record() else {
            return;
        };
        // SAFETY: as in `finish_mark_trace`; the block's bit is in the bitmap, as `index` is
        // below the used length.
        unsafe {
            let traced = record.read();
            if traced == TRACED_NONE || index as u64 >= traced {
                return;
            }
            let block = index / DIRTY_BLOCK;
            let bits = record.add(1 + block / 64).as_ptr();
            *bits |= 1 << (block % 64);
        }
    }

    /// Words at `len` and past it are no longer traced ones: a later push there is past the
    /// traced length.
    #[inline]
    fn lower_traced(&self, len: usize) {
        if let Some(record) = self.record() {
            // SAFETY: as in `finish_mark_trace`.
            unsafe {
                let traced = record.read();
                if traced != TRACED_NONE && traced > len as u64 {
                    record.write(len as u64);
                }
            }
        }
    }

    /// Clear the kind bits that `word` breaks, and [`KIND_SCANNED`] (`word` may replace the
    /// one word that broke a fact).
    #[inline]
    fn note(&mut self, word: u64) {
        let flags = self.header().flags;
        let kept = flags & (word_kind(word) | KIND_INTEGRITY);
        if kept != flags {
            self.header_mut().flags = kept;
        }
    }

    /// Move the used words into a fresh block with room for `capacity` words and `front` free
    /// words before them (which the caller fills), dropping any front room.
    fn reallocate(&mut self, capacity: usize, front: usize) {
        self.reallocate_with_room(capacity, front, 0);
    }

    /// [`Words::reallocate`], with `shifted` words of front room before the header.
    fn reallocate_with_room(&mut self, capacity: usize, front: usize, shifted: usize) {
        let len = self.len();
        assert!(
            capacity <= MAX_WORDS && shifted <= MAX_WORDS && len + front <= capacity,
            "element block of {capacity} words"
        );
        // The new block is this owner's own, whoever else shares the old one.
        let flags = self.header().flags & !KIND_COW;
        let total = shifted + capacity;
        let record = record_words(total);
        // Words that keep their indices keep the old record's traced length and dirty blocks
        // (P1's PackedBuffer kept them across its deque's growth); otherwise, and for a block
        // that kept none, the next trace reports every word.
        let kept = self
            .record()
            .filter(|_| front == 0 && record != 0)
            .map(|old| (old, self.bitmap_words().min(record - 1)));
        let block = crate::gc_storage::alloc_block(block_size(
            total.checked_add(record).expect("element block size"),
        ));
        // SAFETY: a fresh block of the front room, a header, `capacity` words and the record;
        // the used words are copied after `front` words of room, and a kept record's words
        // into the new one's (its bitmap covers the used words, as the old one's did).
        unsafe {
            let words = block.as_ptr().add(HEADER_BYTES).cast::<u64>().add(shifted);
            words.cast::<WordHeader>().sub(1).write(WordHeader {
                used: len as u32,
                capacity: capacity as u32,
                shifted: shifted as u32,
                flags,
            });
            std::ptr::copy_nonoverlapping(self.0.as_ptr(), words.add(front), len);
            if record != 0 {
                let bitmap = words.add(capacity + 1);
                std::ptr::write_bytes(bitmap, 0, record - 1);
                match kept {
                    Some((old, bits)) => {
                        words.add(capacity).write(old.read());
                        std::ptr::copy_nonoverlapping(old.add(1).as_ptr(), bitmap, bits);
                    }
                    None => words.add(capacity).write(TRACED_NONE),
                }
            }
            self.0 = NonNull::new_unchecked(words);
        }
    }
}

/// The bytes of a block of a header and `words` words, checked: on a 32-bit target the count of
/// a large reservation can pass usize.
#[inline]
fn block_size(words: usize) -> usize {
    words
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(HEADER_BYTES))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .expect("element block size")
}

/// The longest list [`Words::from_words`] copies in its scalar loop.
const SHORT_WORDS: usize = 8;

/// The kind bits a block keeps after holding `word`.
#[inline]
fn word_kind(word: u64) -> u32 {
    // Without branches: a hole breaks only NO_HOLES, any other word that is not a Number breaks
    // NUMBERS and INT32, and a Number that is not an exact int32 breaks INT32. A tag word is a
    // NaN pattern, never an exact int32 (an i32 round trip ends at an integral double).
    let hole = word == PACK_EMPTY;
    let number = is_number_word(word);
    let int32 = f64_exact_i32(f64::from_bits(word));
    (u32::from(!hole) * KIND_NO_HOLES)
        | (u32::from(number | hole) * KIND_NUMBERS)
        | (u32::from(int32 | hole) * KIND_INT32)
}

/// Whether `word` is a Number's bits: its tag is none of the other kinds' (`Value::number`'s
/// test), which are 0x7ff9-0x7fff (undefined, the hole, null, Booleans, BigInts, Strings,
/// Symbols) and 0xfff9-0xfffa (Objects, the deferred prototype). Every Number constructor
/// canonicalizes NaN, so no Number has one of them.
#[inline]
fn is_number_word(word: u64) -> bool {
    let tag = word >> 48;
    tag.wrapping_sub(0x7ff9) >= 7 && tag.wrapping_sub(0xfff9) >= 2
}

/// The words a block grows to from none (W2d step 4b): SpiderMonkey's least element capacity,
/// ELEMENT_CAPACITY_MIN (vm/NativeObject.h:1100-1103), which is also JSC's empty-array vector of
/// 5 (runtime/ArrayConventions.h:80-81) filled to its storage class, here the 64-byte one; not
/// V8's 16, whose 144-byte block takes the 192-byte class. The census of step 4b: most arrays
/// born empty take one to four elements (Vue 36,813 of 37,497, DeltaBlue 4,111 of 4,172).
const FIRST_WORDS: usize = 6;

/// V8's first capacity (JSObject::NewElementsCapacity of 0): the second step of the ladder,
/// which a block of fewer words grows to.
const V8_FIRST_WORDS: usize = 16;

/// The capacity to grow a block of `len` used words to for `additional` more, at least what is
/// needed: [`FIRST_WORDS`] from none, then V8's ladder (JSObject::NewElementsCapacity, `old + old
/// / 2 + 16` from its first 16), which a smaller block joins at 16, so that every block past
/// [`FIRST_WORDS`] takes exactly the capacities it took before (W2d step 4b: starting the ladder
/// at 6 shifted every later step, and Gameboy's 64K-word arrays ended 30% larger).
fn grown(len: usize, additional: usize) -> usize {
    let needed = len + additional;
    let rule = if len == 0 {
        FIRST_WORDS
    } else if len < V8_FIRST_WORDS {
        V8_FIRST_WORDS
    } else {
        len + len / 2 + 16
    };
    rule.max(needed).min(MAX_WORDS.max(needed))
}

/// A shared literal's block as a code root's late fill reports it
/// (`value::remember_code_edges`).
impl crate::gc_trace::GcTrace for Words {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        self.trace(sink);
    }
}

impl Default for Words {
    fn default() -> Self {
        Words::new()
    }
}

impl Clone for Words {
    /// An exact copy of the used words and their kind bits; a shared block is shared again. A
    /// map without words (an object literal's template: every copy of it) keeps the static
    /// block, in line.
    #[inline]
    fn clone(&self) -> Self {
        if self.is_empty() {
            return Words::new();
        }
        self.clone_words()
    }
}

impl Words {
    /// [`Clone::clone`] of a block with words: shared again, or an exact copy.
    #[inline(never)]
    fn clone_words(&self) -> Self {
        if self.is_shared() {
            return self.share();
        }
        let mut copy = Words::fresh(self.len());
        // SAFETY: a fresh exact block; the used words are written now and `used` set after.
        unsafe {
            std::ptr::copy_nonoverlapping(self.0.as_ptr(), copy.0.as_ptr(), self.len());
        }
        let header = copy.header_mut();
        header.used = self.len() as u32;
        header.flags = self.kind();
        copy
    }
}

impl std::fmt::Debug for Words {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Words")
            .field("header", self.header())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(n: f64) -> Value {
        Value::Num(n)
    }

    /// Every operation keeps the header's kind bits equal to a scan of the words.
    fn exact(words: &Words) {
        assert_eq!(words.kind(), words.kind() & words.computed_kind());
        assert!(words.len() <= words.capacity() || words.is_static_empty());
    }

    /// The kind bits as `Value::number`'s tag table states them (a reference for `word_kind`):
    /// Value::from_bits itself refuses the synthetic words the test also covers.
    fn decoded_kind(word: u64) -> u32 {
        if word == PACK_EMPTY {
            return KIND_NUMBERS | KIND_INT32;
        }
        let number = !matches!(
            word & !PACK_PAYLOAD,
            PACK_UNDEFINED
                | PACK_EMPTY
                | PACK_NULL
                | PACK_BOOL
                | PACK_BIGINT
                | PACK_STR
                | PACK_SYM
                | PACK_OBJ
                | PACK_LAZY_PROTO
        );
        match number.then(|| f64::from_bits(word)) {
            Some(n) if f64_exact_i32(n) => KIND_ALL,
            Some(_) => KIND_NO_HOLES | KIND_NUMBERS,
            None => KIND_NO_HOLES,
        }
    }

    #[test]
    fn word_kinds_match_the_value_decoding_for_every_tag() {
        let mut words = vec![
            PACK_EMPTY,
            PACK_CANON_NAN,
            0.0f64.to_bits(),
            (-0.0f64).to_bits(),
            1.0f64.to_bits(),
            (-1.0f64).to_bits(),
            0.5f64.to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            f64::MAX.to_bits(),
            f64::MIN_POSITIVE.to_bits(),
            (i32::MAX as f64).to_bits(),
            (i32::MIN as f64).to_bits(),
            (i32::MAX as f64 + 1.0).to_bits(),
            4294967295.0f64.to_bits(),
        ];
        // Every 16-bit tag, each with a few payloads: the other kinds' tags and the doubles
        // around them (0x7ff8 and 0xfff8 are NaNs; 0xfffb and up negative NaN patterns).
        for tag in 0..=0xffffu64 {
            for payload in [0, 1, 0x1234_5678, PACK_PAYLOAD] {
                words.push(tag << 48 | payload);
            }
        }
        for word in words {
            assert_eq!(word_kind(word), decoded_kind(word), "{word:#018x}");
        }
    }

    #[test]
    fn the_empty_block_is_static_and_never_written() {
        let _engine = crate::Engine::new();
        let words = Words::new();
        assert!(words.is_static_empty() && words.is_empty() && words.capacity() == 0);
        assert_eq!(words.kind(), KIND_ALL);
        assert_eq!(words.block_start(), None);
        assert!(words.get(0).is_none());
        assert!(Words::with_capacity(0).is_static_empty());
        assert!(words.clone().is_static_empty());
    }

    #[test]
    fn words_keep_values_holes_and_their_kind() {
        let _engine = crate::Engine::new();
        let mut words = Words::new();
        for n in 0..40 {
            words.push(num(n as f64));
            exact(&words);
        }
        assert_eq!(words.len(), 40);
        assert_eq!(words.kind(), KIND_ALL);
        assert_eq!(words.get(7).and_then(|v| v.number()), Some(7.0));
        words.set(3, num(0.5));
        assert_eq!(words.kind(), KIND_NO_HOLES | KIND_NUMBERS);
        words.set(4, Value::Null);
        assert_eq!(words.kind(), KIND_NO_HOLES);
        words.resize_with_holes(45);
        assert_eq!(words.kind(), 0);
        assert!(words.get(44).is_none() && words.get(45).is_none());
        exact(&words);
        assert_eq!(words.pop(), Some(PACK_EMPTY));
        words.truncate(3);
        assert_eq!(words.len(), 3);
        // A cleared fact comes back only by proof.
        assert_eq!(words.kind(), 0);
        assert_eq!(words.reprove(), KIND_ALL);
        assert_eq!(words.kind(), KIND_ALL);
        assert_eq!(num(-0.0).number(), Some(-0.0));
        words.push(num(-0.0));
        assert_eq!(words.kind(), KIND_NO_HOLES | KIND_NUMBERS);
    }

    #[test]
    fn shifts_move_the_header_and_unshifts_take_the_room_back() {
        let _engine = crate::Engine::new();
        let mut words = Words::from_values((0..10).map(|n| num(n as f64)));
        let start = words.block_start().unwrap();
        for n in 0..4 {
            assert_eq!(words.shift(), Some(num(n as f64).bits()));
            exact(&words);
            // The block does not move; its first word does.
            assert_eq!(words.block_start(), Some(start));
        }
        assert_eq!(words.len(), 6);
        assert_eq!(words.header().shifted, 4);
        assert_eq!(words.get(0).and_then(|v| v.number()), Some(4.0));
        words.unshift(&[num(2.0), num(3.0)]);
        assert_eq!(
            words.block_start(),
            Some(start),
            "front room taken back in place"
        );
        assert_eq!(words.header().shifted, 2);
        words.unshift(&[num(-1.0), num(0.0), num(1.0)]);
        assert_eq!(
            words.header().shifted,
            8,
            "a fresh block keeps front room of the used words"
        );
        let values: Vec<f64> = (0..words.len())
            .map(|i| words.get(i).unwrap().number().unwrap())
            .collect();
        assert_eq!(
            values,
            [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        exact(&words);
        while words.shift().is_some() {}
        assert!(words.is_empty() && words.shift().is_none());
        assert!(Words::new().clone().shift().is_none());
    }

    #[test]
    fn copies_are_exact_and_growth_follows_the_exemplars() {
        let _engine = crate::Engine::new();
        let mut words = Words::new();
        words.push(Value::Undefined);
        assert_eq!(words.capacity(), 6, "FIRST_WORDS: the 64-byte class");
        for _ in 0..5 {
            words.push(num(1.0));
        }
        assert_eq!(words.capacity(), 6);
        words.push(num(2.0));
        assert_eq!(
            words.capacity(),
            16,
            "V8's first capacity, joining its ladder"
        );
        for _ in 0..9 {
            words.push(num(3.0));
        }
        assert_eq!(words.capacity(), 16);
        words.push(num(4.0));
        assert_eq!(words.capacity(), 16 + 8 + 16, "then V8's ladder");
        let mut exact = Words::from_values((0..10).map(|n| num(n as f64)));
        exact.push(num(10.0));
        assert_eq!(
            exact.capacity(),
            16,
            "a smaller block joins the ladder at 16"
        );
        let copy = words.clone();
        assert_eq!(copy.as_slice(), words.as_slice());
        assert_eq!((copy.capacity(), copy.kind()), (words.len(), words.kind()));
        assert_ne!(copy.block_start(), words.block_start());
        words.reserve(1000);
        assert!(words.capacity() >= words.len() + 1000);
        assert_eq!(words.as_slice(), copy.as_slice());
    }

    #[test]
    fn random_operations_match_a_vector_and_never_overclaim_a_fact() {
        let _engine = crate::Engine::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        let mut words = Words::new();
        let mut model: Vec<u64> = Vec::new();
        let pick = |n: u64| match n % 5 {
            0 => num((n / 5) as f64),
            1 => num(n as f64 + 0.5),
            2 => Value::Null,
            3 => num(-0.0),
            _ => Value::Undefined,
        };
        for _ in 0..20_000 {
            match next(9) {
                0 | 1 => {
                    let value = pick(next(1000));
                    words.push(value);
                    model.push(value.bits());
                }
                2 => assert_eq!(words.pop(), model.pop()),
                3 => assert_eq!(words.shift(), (!model.is_empty()).then(|| model.remove(0))),
                4 => {
                    let values: Vec<Value> = (0..1 + next(3)).map(|_| pick(next(1000))).collect();
                    words.unshift(&values);
                    model.splice(0..0, values.iter().map(|v| v.bits()));
                }
                5 if !model.is_empty() => {
                    let index = next(model.len() as u64) as usize;
                    let value = pick(next(1000));
                    words.set(index, value);
                    model[index] = value.bits();
                }
                6 => {
                    let len = next(model.len() as u64 + 1) as usize;
                    words.truncate(len);
                    model.truncate(len);
                }
                7 => {
                    let len = model.len() + next(4) as usize;
                    words.resize_with_holes(len);
                    model.resize(len, PACK_EMPTY);
                }
                _ => {
                    if next(4) == 0 {
                        words.reprove();
                    } else {
                        words = words.clone();
                    }
                }
            }
            assert_eq!(words.as_slice(), &model[..]);
            assert_eq!(
                words.kind() & !words.computed_kind(),
                0,
                "a fact the words break"
            );
            if model.len() > 200 {
                words.truncate(50);
                model.truncate(50);
            }
        }
    }

    #[test]
    fn traces_report_the_block_and_its_cells_but_never_a_numbers_block() {
        let mut engine = crate::Engine::new();
        let object = engine.interp.new_object();
        let _ = &mut engine;
        struct Seen(Vec<u64>);
        impl crate::gc_trace::GcSink for Seen {
            fn word(&mut self, word: u64) {
                self.0.push(word);
            }
        }
        let mut numbers = Words::from_values((0..5).map(|n| num(n as f64)));
        let mut seen = Seen(Vec::new());
        numbers.trace(&mut seen);
        assert_eq!(seen.0, [numbers.block_start().unwrap() as u64]);
        numbers.set(2, Value::Obj(object));
        numbers.resize_with_holes(8);
        let mut seen = Seen(Vec::new());
        numbers.trace(&mut seen);
        assert_eq!(
            seen.0.len(),
            2,
            "the block and the object, no hole or number: {:x?}",
            seen.0
        );
        assert_eq!(seen.0[0], numbers.block_start().unwrap() as u64);
        let mut seen = Seen(Vec::new());
        Words::new().trace(&mut seen);
        assert!(seen.0.is_empty());
    }
}
