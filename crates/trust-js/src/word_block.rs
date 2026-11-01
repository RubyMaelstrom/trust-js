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
use super::*;
use std::ptr::NonNull;

/// No used word is a hole.
pub(crate) const KIND_NO_HOLES: u32 = 1 << 0;
/// Every used word that is not a hole is a Number.
pub(crate) const KIND_NUMBERS: u32 = 1 << 1;
/// Every used word that is not a hole is an exact int32 (bit-identical through an i32, so not
/// -0; `MIRROR_ALL_I32`'s meaning).
pub(crate) const KIND_INT32: u32 = 1 << 2;
/// The facts of a block with no words.
pub(crate) const KIND_ALL: u32 = KIND_NO_HOLES | KIND_NUMBERS | KIND_INT32;

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
        let mut words = Words::new();
        if capacity != 0 {
            words.reallocate(capacity, 0);
        }
        words
    }

    /// An exact block of `values`.
    pub(crate) fn from_values(values: impl ExactSizeIterator<Item = Value>) -> Self {
        let mut words = Words::with_capacity(values.len());
        for value in values {
            words.push(value);
        }
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
        // SAFETY: a heap block's header, which `&mut self` holds uniquely.
        unsafe { &mut *self.0.as_ptr().cast::<WordHeader>().sub(1) }
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

    /// The kind bits of the used words.
    #[inline]
    pub(crate) fn kind(&self) -> u32 {
        self.header().flags
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
        assert!(index < self.len(), "word index {index} past {}", self.len());
        let word = value.bits();
        self.note(word);
        self.remember(index, word);
        // SAFETY: a used word of a heap block (the static block has none).
        unsafe { self.0.as_ptr().add(index).write(word) };
    }

    /// Append `value`, growing the block when it is full.
    pub(crate) fn push(&mut self, value: Value) {
        let len = self.len();
        if len == self.capacity() {
            self.reallocate(grown(len, 1), 0);
        }
        let word = value.bits();
        self.note(word);
        // SAFETY: below capacity, so inside the block.
        unsafe { self.0.as_ptr().add(len).write(word) };
        self.header_mut().used += 1;
    }

    /// Remove and return the last word (a hole's too).
    pub(crate) fn pop(&mut self) -> Option<u64> {
        let len = self.len().checked_sub(1)?;
        let word = self.as_slice()[len];
        self.header_mut().used = len as u32;
        self.lower_traced(len);
        Some(word)
    }

    /// Remove and return the first word, leaving its room in front of the header.
    pub(crate) fn shift(&mut self) -> Option<u64> {
        if self.is_empty() {
            return None;
        }
        let header = *self.header();
        let first = self.as_slice()[0];
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
                flags: header.flags,
            });
            self.0 = NonNull::new_unchecked(words);
        }
        Some(first)
    }

    /// Insert `values` before the first word, in front room when there is enough.
    pub(crate) fn unshift(&mut self, values: &[Value]) {
        let n = values.len();
        if n == 0 {
            return;
        }
        let header = *self.header();
        let flags = values
            .iter()
            .fold(header.flags, |flags, value| flags & word_kind(value.bits()));
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
        let len = self.len();
        if len > 1 {
            // Words change places: the next trace reports them all.
            self.force_full();
            // SAFETY: the used words of a heap block, which `&mut self` holds uniquely.
            unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr(), len) }.reverse();
        }
    }

    /// Replace the words in `range` by `values`, returning the values removed (holes as
    /// `None`), as `Vec::splice` does.
    pub(crate) fn splice(
        &mut self,
        range: std::ops::Range<usize>,
        values: &[Value],
    ) -> Vec<Option<Value>> {
        let words = self.as_slice();
        let removed = words[range.clone()]
            .iter()
            // SAFETY: a used word is a Value's bits.
            .map(|&word| (word != PACK_EMPTY).then(|| unsafe { Value::from_bits(word) }))
            .collect();
        let suffix = words[range.end..].to_vec();
        self.truncate(range.start);
        self.reserve(values.len() + suffix.len());
        for &value in values {
            self.push(value);
        }
        for word in suffix {
            // SAFETY: as above.
            self.push(unsafe { Value::from_bits(word) });
        }
        removed
    }

    /// Keep the first `len` words.
    pub(crate) fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.header_mut().used = len as u32;
            self.lower_traced(len);
        }
    }

    /// Lengthen to `len` words with holes after the used ones.
    pub(crate) fn resize_with_holes(&mut self, len: usize) {
        let used = self.len();
        if len <= used {
            return;
        }
        if len > self.capacity() {
            self.reallocate(grown(used, len - used), 0);
        }
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

    /// Prove the kind bits again from the words, which a cleared bit needs to come back.
    pub(crate) fn reprove(&mut self) -> u32 {
        let flags = self.computed_kind();
        if !self.is_static_empty() {
            self.header_mut().flags = flags;
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

    /// Report the block and the edges of its words.
    #[cfg(test)]
    pub(crate) fn trace<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(start) = self.block_start() {
            crate::gc_storage::trace_block(start, sink);
        }
        self.trace_values(None, sink);
    }

    /// Report the edges of the words, not the block. A Numbers block has none. A marker's
    /// remembered trace reports only the dirty runs below the traced length and the words past
    /// it; every other trace (a marker's full one, or `None`'s, the verifiers' and the
    /// heap walks') reports every word.
    pub(crate) fn trace_values<S: crate::gc_trace::GcSink + ?Sized>(
        &self,
        purpose: Option<crate::gc_core::TracePurpose>,
        sink: &mut S,
    ) {
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
    pub(crate) fn finish_mark_trace(&self) {
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
    pub(crate) fn force_full(&self) {
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
        HEADER_BYTES + 8 * (words + record_words(words))
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
        // begins (`reallocate`); the static block keeps none.
        (self.block_words() > REMEMBER_MIN)
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

    /// Clear the kind bits that `word` breaks.
    #[inline]
    fn note(&mut self, word: u64) {
        let flags = self.header().flags;
        let kept = flags & word_kind(word);
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
        let flags = self.kind();
        let total = shifted + capacity;
        let record = record_words(total);
        // Words that keep their indices keep the old record's traced length and dirty blocks
        // (P1's PackedBuffer kept them across its deque's growth); otherwise, and for a block
        // that kept none, the next trace reports every word.
        let kept = self
            .record()
            .filter(|_| front == 0 && record != 0)
            .map(|old| (old, self.bitmap_words().min(record - 1)));
        let block = crate::gc_storage::alloc_block(HEADER_BYTES + 8 * (total + record));
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

/// The kind bits a block keeps after holding `word`.
#[inline]
fn word_kind(word: u64) -> u32 {
    if word == PACK_EMPTY {
        return KIND_NUMBERS | KIND_INT32;
    }
    // SAFETY: a Value's bits.
    match unsafe { Value::from_bits(word) }.number() {
        Some(n) if f64_exact_i32(n) => KIND_ALL,
        Some(_) => KIND_NO_HOLES | KIND_NUMBERS,
        None => KIND_NO_HOLES,
    }
}

/// The capacity to grow a block of `len` used words to for `additional` more: V8's
/// (JSObject::NewElementsCapacity, `old + old / 2 + 16`), at least what is needed.
fn grown(len: usize, additional: usize) -> usize {
    let needed = len + additional;
    (len + len / 2 + 16).max(needed).min(MAX_WORDS.max(needed))
}

impl Default for Words {
    fn default() -> Self {
        Words::new()
    }
}

impl Clone for Words {
    /// An exact copy of the used words and their kind bits.
    fn clone(&self) -> Self {
        let mut copy = Words::new();
        if !self.is_empty() {
            copy.reallocate(self.len(), 0);
            // SAFETY: `reallocate` copied from the static block's empty words; the used words
            // are written now and `used` set after.
            unsafe {
                std::ptr::copy_nonoverlapping(self.0.as_ptr(), copy.0.as_ptr(), self.len());
            }
            let header = copy.header_mut();
            header.used = self.len() as u32;
            header.flags = self.kind();
        }
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
    fn copies_are_exact_and_growth_follows_v8() {
        let _engine = crate::Engine::new();
        let mut words = Words::new();
        words.push(Value::Undefined);
        assert_eq!(words.capacity(), 16, "0 + 0 / 2 + 16");
        for _ in 0..15 {
            words.push(num(1.0));
        }
        assert_eq!(words.capacity(), 16);
        words.push(num(2.0));
        assert_eq!(words.capacity(), 16 + 8 + 16);
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
