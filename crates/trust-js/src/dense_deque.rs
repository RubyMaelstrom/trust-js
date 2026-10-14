//! Contiguous dense storage with spare space at both ends. The first three words have
//! the Vec header layout accepted by the native-layout probes; the allocation itself
//! always retains its original base pointer. Generated append paths may update `len`
//! directly after initializing a slot, so that header is the authoritative live range.
use std::mem::MaybeUninit;
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
        Self {
            cap: 0,
            ptr: NonNull::dangling(),
            len: 0,
            front: 0,
        }
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
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
        // provenance; its original allocation remains owned until Drop.
        unsafe { self.ptr.sub(self.front).as_ptr() }
    }

    unsafe fn allocation_owner(&self) -> Vec<MaybeUninit<T>> {
        // Callers transfer this unique allocation owner exactly once and publish
        // new metadata or suppress DenseDeque::drop before it can be freed again.
        unsafe {
            Vec::from_raw_parts(
                self.allocation_ptr().cast::<MaybeUninit<T>>(),
                0,
                self.capacity(),
            )
        }
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
        let mut allocation = Vec::<MaybeUninit<T>>::new();
        allocation.try_reserve_exact(capacity)?;
        let total = allocation.capacity();
        // Vec's pointer is non-null even at zero capacity. This retained fact also
        // lets ordinary dense reads use the same ptr/len loads as a Vec header.
        let ptr = unsafe { NonNull::new_unchecked(allocation.as_mut_ptr().add(front).cast::<T>()) };
        let previous = unsafe { self.allocation_owner() };
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), ptr.as_ptr(), self.len) };
        // Old slots have relinquished their owners; MaybeUninit never drops them.
        std::mem::forget(allocation);
        self.ptr = ptr;
        self.cap = total - front;
        self.front = front;
        drop(previous);
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
        if length < self.len {
            let removed = self.len - length;
            self.len = length;
            // Slice drop glue also cleans up later elements if a destructor
            // panics. Publish the shorter range first, matching Vec::truncate.
            unsafe {
                std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                    self.ptr.as_ptr().add(length),
                    removed,
                ));
            }
        }
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

    fn into_vec(self) -> Vec<T> {
        let length = self.len;
        let base = self.allocation_ptr();
        let capacity = self.capacity();
        unsafe { std::ptr::copy(self.ptr.as_ptr(), base, length) };
        // T and MaybeUninit<T> have identical allocation layouts. All initialized
        // owners now occupy the prefix and the base pointer is unchanged.
        std::mem::forget(self);
        unsafe { Vec::from_raw_parts(base, length, capacity) }
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
        let (ptr, len, cap) = (
            unsafe { NonNull::new_unchecked(vector.as_mut_ptr()) },
            vector.len(),
            vector.capacity(),
        );
        std::mem::forget(vector);
        Self {
            cap,
            ptr,
            len,
            front: 0,
        }
    }
}

impl<T: Clone> Clone for DenseDeque<T> {
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

impl<T> Drop for DenseDeque<T> {
    fn drop(&mut self) {
        // Acquire the base-allocation guard before running any T destructor. Its
        // empty initialized prefix owns no T values, and unwind cleanup releases
        // the allocation even if dropping a live element panics.
        let allocation = unsafe { self.allocation_owner() };
        self.truncate(0);
        drop(allocation);
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
    use std::rc::Rc;

    #[test]
    fn mixed_end_edits_reallocation_and_clone_keep_reference_owners() {
        let mut dense = DenseDeque::new();
        let mut reference = VecDeque::new();
        let owner = Rc::new(17);
        let mut state = 9182_u32;
        for step in 0..20_000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            match state >> 29 {
                0 | 1 => {
                    dense.push((step, owner.clone()));
                    reference.push_back((step, owner.clone()));
                }
                2 | 3 => {
                    dense.prepend(std::iter::once((step, owner.clone())));
                    reference.push_front((step, owner.clone()));
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
                    assert_eq!(Rc::strong_count(&owner), 1 + dense.len() * 3);
                }
            }
            assert!(dense.iter().eq(reference.iter()));
            assert_eq!(Rc::strong_count(&owner), 1 + dense.len() * 2);
        }
        dense.clear();
        reference.clear();
        assert_eq!(Rc::strong_count(&owner), 1);
        dense.push((999, owner.clone()));
        assert_eq!(dense.pop_front().unwrap().0, 999);
        assert_eq!(Rc::strong_count(&owner), 1);
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
    fn native_append_header_updates_remain_owned_after_front_edits() {
        let owner = Rc::new(23);
        let mut dense = DenseDeque::with_capacity(16);
        dense.push(owner.clone());
        dense.push(owner.clone());
        drop(dense.pop_front());
        // Exactly the native append protocol: initialize into admitted capacity,
        // then publish the new logical length. Spare slots contain no T owners.
        unsafe { dense.ptr.as_ptr().add(dense.len).write(owner.clone()) };
        dense.len += 1;
        assert_eq!(Rc::strong_count(&owner), 3);
        assert!(dense.iter().all(|item| Rc::ptr_eq(item, &owner)));
        let removed: Vec<_> = dense.splice(0..1, std::iter::empty()).collect();
        assert_eq!(removed.len(), 1);
        assert_eq!(dense.len(), 1);
        drop(removed);
        drop(dense);
        assert_eq!(Rc::strong_count(&owner), 1);
    }

    #[test]
    fn empty_and_fully_drained_lists_transfer_and_reuse_the_original_allocation() {
        let empty = DenseDeque::<Rc<usize>>::new().into_vec();
        assert!(empty.is_empty());
        assert_eq!(empty.capacity(), 0);
        let owner = Rc::new(7);
        let mut dense: DenseDeque<_> = (0..16).map(|_| owner.clone()).collect();
        for _ in 0..16 {
            drop(dense.pop_front().unwrap());
        }
        assert_eq!((dense.len, dense.front, dense.cap), (0, 16, 0));
        let mut vector = dense.into_vec();
        assert_eq!(vector.capacity(), 16);
        vector.push(owner.clone());
        assert_eq!(Rc::strong_count(&owner), 2);
        drop(vector);
        assert_eq!(Rc::strong_count(&owner), 1);

        let mut dense: DenseDeque<_> = (0..16).map(|_| owner.clone()).collect();
        for _ in 0..16 {
            drop(dense.pop_front().unwrap());
        }
        dense.prepend(std::iter::once(owner.clone()));
        assert_eq!((dense.len, dense.front, dense.cap), (1, 15, 1));
        drop(dense.pop_front());
        dense.push(owner.clone());
        assert_eq!(dense.front, 0);
        assert_eq!(dense.len, 1);
        assert_eq!(Rc::strong_count(&owner), 2);
        drop(dense);
        assert_eq!(Rc::strong_count(&owner), 1);
    }

    #[test]
    fn reserve_overflow_preserves_the_shifted_header_and_every_owner() {
        let owner = Rc::new(7);
        let mut dense = DenseDeque::with_capacity(16);
        dense.push(owner.clone());
        dense.push(owner.clone());
        drop(dense.pop_front());
        let before = (dense.ptr, dense.len, dense.cap, dense.front);
        assert!(dense.try_reserve_exact(usize::MAX).is_err());
        assert_eq!((dense.ptr, dense.len, dense.cap, dense.front), before);
        assert_eq!(Rc::strong_count(&owner), 2);
        assert!(Rc::ptr_eq(&dense[0], &owner));
        drop(dense);
        assert_eq!(Rc::strong_count(&owner), 1);
    }

    #[test]
    fn destructor_panic_still_drops_every_live_owner() {
        struct Item {
            panic: bool,
            dropped: Rc<std::cell::Cell<usize>>,
        }
        impl Drop for Item {
            fn drop(&mut self) {
                self.dropped.set(self.dropped.get() + 1);
                if self.panic {
                    panic!("one destructor failed");
                }
            }
        }
        let dropped = Rc::new(std::cell::Cell::new(0));
        let mut dense: DenseDeque<Item> = (0..16)
            .map(|index| Item {
                panic: index == 7,
                dropped: dropped.clone(),
            })
            .collect();
        drop(dense.pop_front());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(dense))).is_err());
        assert_eq!(dropped.get(), 16);
        assert_eq!(Rc::strong_count(&dropped), 1);
    }
}
