//! Contiguous dense storage with spare space at both ends. The first three words have
//! the Vec header layout accepted by the native-layout probes; the allocation itself
//! always retains its original base pointer. Generated append paths may update `len`
//! directly after initializing a slot, so that header is the authoritative live range.
//!
//! P1 S6: the storage is a block of the active Agent heap (`gc_storage`), not a Rust-heap
//! allocation. A deque belongs to a cell (or a traced template) whose trace reports
//! [`DenseDeque::block_ptr`]; growth moves the elements to a new block and leaves the old one
//! to the collector. Its elements are plain words (no drop glue, which constructors check), so
//! dropping a deque reads nothing: its block may already have gone with its heap.
use std::ops::{Deref, DerefMut, Range};
use std::ptr::NonNull;

#[repr(C)]
pub(crate) struct DenseDeque<T> {
    // Vec's order is not guaranteed. jit_layout verifies these offsets against every
    // vector representation before native code is admitted; a mismatch fails closed.
    pub(super) cap: usize,
    pub(super) ptr: NonNull<T>,
    pub(super) len: usize,
    // The logical pointer retains its allocation provenance. Native append paths
    // only publish `len`; total capacity is always `cap + front`, and subtracting
    // `front` recovers the original allocator base without another Vec header.
    front: usize,
}

impl<T> Default for DenseDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DenseDeque<T> {
    pub(super) const fn new() -> Self {
        assert!(std::mem::size_of::<T>() != 0);
        const {
            assert!(
                !std::mem::needs_drop::<T>(),
                "dense deque elements own nothing"
            )
        };
        Self {
            cap: 0,
            ptr: NonNull::dangling(),
            len: 0,
            front: 0,
        }
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        let mut deque = Self::new();
        if capacity != 0 {
            deque
                .reallocate(0, capacity, true)
                .expect("dense allocation failed");
        }
        deque
    }

    /// Total retained storage, including unused space before the live range.
    pub(super) fn capacity(&self) -> usize {
        self.cap + self.front
    }

    #[cfg(test)]
    pub(super) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub(super) fn as_slice(&self) -> &[T] {
        // The logical header owns precisely this initialized range. Spare slots and
        // consumed prefixes contain no owners; Drop frees the original allocation.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn front_capacity(&self) -> usize {
        self.front
    }

    fn allocation_ptr(&self) -> *mut T {
        // Includes the empty/dangling case, where `front` is zero. A completely
        // consumed list may have a one-past logical pointer, still with the same
        // provenance; its original block remains owned until the deque moves on.
        unsafe { self.ptr.sub(self.front).as_ptr() }
    }

    /// The storage block's address, while the deque has one: the edge its owner reports
    /// (`gc_storage::trace_block`).
    #[inline]
    pub(crate) fn block_ptr(&self) -> Option<*const u8> {
        (self.capacity() != 0).then(|| self.allocation_ptr() as *const u8)
    }

    fn reallocate(
        &mut self,
        front: usize,
        back: usize,
        exact: bool,
    ) -> Result<(), std::collections::TryReserveError> {
        let required = self.len.saturating_add(front).saturating_add(back);
        let capacity = if exact {
            required
        } else {
            required.max(self.len.saturating_mul(2)).max(8)
        };
        let block = capacity
            .checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| bytes != 0 && bytes <= isize::MAX as usize)
            .and_then(crate::gc_storage::try_alloc_block)
            .ok_or_else(capacity_overflow)?;
        const { assert!(std::mem::align_of::<T>() <= crate::gc_storage::BLOCK_ALIGN) };
        // The block is non-null and aligned for T; so is every slot pointer derived from it.
        let ptr = unsafe { NonNull::new_unchecked(block.as_ptr().cast::<T>().add(front)) };
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), ptr.as_ptr(), self.len) };
        // The old block's slots have relinquished their owners; the collector frees it.
        self.ptr = ptr;
        self.cap = capacity - front;
        self.front = front;
        Ok(())
    }

    pub(super) fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        if additional > self.cap - self.len {
            self.reallocate(0, additional, true)?;
        }
        Ok(())
    }

    pub(super) fn reserve_exact(&mut self, additional: usize) {
        self.try_reserve_exact(additional)
            .expect("dense allocation failed");
    }

    pub(super) fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.reallocate(0, 1, false)
                .expect("dense allocation failed");
        }
        unsafe { self.ptr.as_ptr().add(self.len).write(value) };
        self.len += 1;
    }

    pub(super) fn pop(&mut self) -> Option<T> {
        self.len = self.len.checked_sub(1)?;
        Some(unsafe { self.ptr.as_ptr().add(self.len).read() })
    }

    pub(super) fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let first = unsafe { self.ptr.as_ptr().read() };
        self.ptr = unsafe { self.ptr.add(1) };
        self.len -= 1;
        self.cap -= 1;
        self.front += 1;
        Some(first)
    }

    pub(super) fn prepend(
        &mut self,
        items: impl ExactSizeIterator<Item = T> + DoubleEndedIterator,
    ) {
        let count = items.len();
        if count == 0 {
            return;
        }
        if self.front_capacity() < count {
            // Reserve a geometric prefix: repeated single-element prepends move
            // existing values only once per doubling, while ordinary reads stay
            // one pointer/index operation and never need ring arithmetic.
            self.reallocate(count.max(self.len).max(8), 0, false)
                .expect("dense allocation failed");
        }
        // Fill from the back so even a panicking iterator leaves a valid live
        // range. Check the reserve independently of the iterator's size claim.
        for value in items.rev() {
            assert!(self.front_capacity() != 0);
            self.ptr = unsafe { self.ptr.sub(1) };
            self.cap += 1;
            self.front -= 1;
            unsafe { self.ptr.as_ptr().write(value) };
            self.len += 1;
        }
    }

    pub(super) fn truncate(&mut self, length: usize) {
        // The removed elements own nothing (see the module comment).
        self.len = self.len.min(length);
    }

    pub(super) fn clear(&mut self) {
        self.truncate(0);
        self.ptr = unsafe { self.ptr.sub(self.front) };
        self.cap += self.front;
        self.front = 0;
    }

    pub(super) fn resize_with(&mut self, length: usize, mut make: impl FnMut() -> T) {
        self.truncate(length);
        self.reserve_exact(length.saturating_sub(self.len));
        while self.len < length {
            self.push(make());
        }
    }

    fn into_vec(mut self) -> Vec<T> {
        let length = self.len;
        let mut vector = Vec::with_capacity(self.capacity());
        // The live owners move bitwise; the deque then holds none and drops nothing.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), vector.as_mut_ptr(), length);
            vector.set_len(length);
        }
        self.len = 0;
        vector
    }

    /// General splices are linear as specified; shift/unshift use the end operations.
    pub(crate) fn splice(
        &mut self,
        range: Range<usize>,
        items: impl IntoIterator<Item = T>,
    ) -> std::vec::IntoIter<T> {
        let mut vector = std::mem::take(self).into_vec();
        let removed: Vec<T> = vector.splice(range, items).collect();
        *self = vector.into();
        removed.into_iter()
    }
}

impl<T> From<Vec<T>> for DenseDeque<T> {
    fn from(mut vector: Vec<T>) -> Self {
        assert_ne!(std::mem::size_of::<T>(), 0);
        let mut deque = Self::new();
        let length = vector.len();
        if vector.capacity() != 0 {
            deque
                .reallocate(0, vector.capacity(), true)
                .expect("dense allocation failed");
        }
        // The owners move bitwise into the block; the vector keeps its buffer and frees it
        // without dropping them.
        unsafe {
            std::ptr::copy_nonoverlapping(vector.as_ptr(), deque.ptr.as_ptr(), length);
            vector.set_len(0);
        }
        deque.len = length;
        deque
    }
}

/// The error of a request no block can satisfy.
fn capacity_overflow() -> std::collections::TryReserveError {
    Vec::<u8>::new()
        .try_reserve_exact(usize::MAX)
        .expect_err("a usize::MAX-byte reservation overflows")
}

impl<T: Clone> Clone for DenseDeque<T> {
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

impl<T> Deref for DenseDeque<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for DenseDeque<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Extend<T> for DenseDeque<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<T> FromIterator<T> for DenseDeque<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl<'a, T> IntoIterator for &'a DenseDeque<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn mixed_end_edits_reallocation_and_clone_match_a_reference_deque() {
        let mut dense = DenseDeque::new();
        let mut reference = VecDeque::new();
        let mut state = 9182_u32;
        for step in 0..20_000_u64 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            match state >> 29 {
                0 | 1 => {
                    dense.push(step);
                    reference.push_back(step);
                }
                2 | 3 => {
                    dense.prepend(std::iter::once(step));
                    reference.push_front(step);
                }
                4 => assert_eq!(dense.pop(), reference.pop_back()),
                5 => assert_eq!(dense.pop_front(), reference.pop_front()),
                6 => {
                    let length = dense.len() / 2;
                    dense.truncate(length);
                    reference.truncate(length);
                }
                _ => {
                    let clone = dense.clone();
                    assert!(clone.iter().eq(reference.iter()));
                }
            }
            assert!(dense.iter().eq(reference.iter()));
        }
        dense.clear();
        assert!(dense.is_empty());
        dense.push(999);
        assert_eq!(dense.pop_front(), Some(999));
    }

    #[test]
    fn fifo_and_lifo_front_edits_amortize_relocations() {
        for prepend in [false, true] {
            let mut dense: DenseDeque<usize> = (0..1024).collect();
            let mut moves = 0;
            for value in 1024..21_024 {
                let pointer = dense.allocation_ptr();
                if prepend {
                    dense.prepend(std::iter::once(value));
                    dense.pop();
                } else {
                    assert_eq!(dense.pop_front(), Some(value - 1024));
                    dense.push(value);
                }
                moves += usize::from(pointer != dense.allocation_ptr());
                assert_eq!(dense.len(), 1024);
                assert!(dense.capacity() <= 2049);
            }
            assert!(
                moves <= 22,
                "each edit must not relocate the live list: {moves}"
            );
        }
    }

    #[test]
    fn native_append_header_updates_remain_in_the_live_range_after_front_edits() {
        let mut dense = DenseDeque::with_capacity(16);
        dense.push(1_u64);
        dense.push(2);
        assert_eq!(dense.pop_front(), Some(1));
        // Exactly the native append protocol: initialize into admitted capacity,
        // then publish the new logical length.
        unsafe { dense.ptr.as_ptr().add(dense.len).write(3) };
        dense.len += 1;
        assert_eq!(dense.as_slice(), &[2, 3]);
        let removed: Vec<_> = dense.splice(0..1, std::iter::empty()).collect();
        assert_eq!(removed, [2]);
        assert_eq!(dense.as_slice(), &[3]);
    }

    #[test]
    fn empty_and_fully_drained_lists_convert_and_reuse_their_block() {
        let empty = DenseDeque::<u64>::new().into_vec();
        assert!(empty.is_empty());
        assert_eq!(empty.capacity(), 0);
        let mut dense: DenseDeque<u64> = (0..16).collect();
        for value in 0..16 {
            assert_eq!(dense.pop_front(), Some(value));
        }
        assert_eq!((dense.len, dense.front, dense.cap), (0, 16, 0));
        let vector = dense.into_vec();
        assert!(vector.is_empty());
        assert!(vector.capacity() >= 16);

        let mut dense: DenseDeque<u64> = (0..16).collect();
        for _ in 0..16 {
            dense.pop_front();
        }
        dense.prepend(std::iter::once(7));
        assert_eq!((dense.len, dense.front, dense.cap), (1, 15, 1));
        assert_eq!(dense.pop_front(), Some(7));
        dense.push(8);
        assert_eq!(dense.front, 0);
        assert_eq!(dense.as_slice(), &[8]);
    }

    #[test]
    fn reserve_overflow_preserves_the_shifted_header_and_every_element() {
        let mut dense = DenseDeque::with_capacity(16);
        dense.push(1_u64);
        dense.push(2);
        assert_eq!(dense.pop_front(), Some(1));
        let before = (dense.ptr, dense.len, dense.cap, dense.front);
        assert!(dense.try_reserve_exact(usize::MAX).is_err());
        assert_eq!((dense.ptr, dense.len, dense.cap, dense.front), before);
        assert_eq!(dense.as_slice(), &[2]);
    }
}
