//! P1 S3: a vector that keeps its first `N` elements in place (Decision 3's inline slots).
//!
//! Up to `N` elements live inside the owning value, so an object cell holding one needs no
//! second allocation for a few properties. The `N + 1`th element moves all of them to one heap
//! buffer, which is kept when the length shrinks again: a map that grew past `N` tends to stay
//! large, and moving back would make every shrink and regrow copy.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::mem::MaybeUninit;
use std::ptr;

pub(crate) struct InlineVec<T, const N: usize> {
    len: u32,
    /// Heap capacity once spilled; 0 while the elements are in place.
    cap: u32,
    heap: *mut T,
    inline: [MaybeUninit<T>; N],
}

impl<T, const N: usize> InlineVec<T, N> {
    pub(crate) const fn new() -> Self {
        Self {
            len: 0,
            cap: 0,
            heap: ptr::null_mut(),
            inline: [const { MaybeUninit::uninit() }; N],
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        if capacity > N {
            vec.grow_to(capacity);
        }
        vec
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        if self.cap == 0 {
            N
        } else {
            self.cap as usize
        }
    }

    /// Whether the elements live in a heap buffer.
    #[inline]
    pub(crate) fn spilled(&self) -> bool {
        self.cap != 0
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        if self.cap == 0 {
            self.inline.as_ptr().cast()
        } else {
            self.heap
        }
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        if self.cap == 0 {
            self.inline.as_mut_ptr().cast()
        } else {
            self.heap
        }
    }

    /// Move the elements to a heap buffer of exactly `capacity` (at least the length).
    fn grow_to(&mut self, capacity: usize) {
        debug_assert!(capacity >= self.len() && capacity > N);
        let capacity = u32::try_from(capacity).expect("InlineVec capacity overflow");
        let layout = Layout::array::<T>(capacity as usize).expect("InlineVec capacity overflow");
        assert!(layout.size() != 0, "InlineVec of zero-sized elements");
        // SAFETY: `layout` has a non-zero size.
        let heap = unsafe { alloc(layout) }.cast::<T>();
        if heap.is_null() {
            handle_alloc_error(layout);
        }
        // SAFETY: the old storage holds `len` initialized elements; the new buffer has room for
        // them and does not overlap the old one. They move bitwise; the old copies are forgotten.
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), heap, self.len());
            self.free_heap();
        }
        self.heap = heap;
        self.cap = capacity;
    }

    /// Release the heap buffer without touching its elements.
    ///
    /// # Safety
    /// The elements must have been moved out or dropped already.
    unsafe fn free_heap(&mut self) {
        if self.cap != 0 {
            let layout = Layout::array::<T>(self.cap as usize).expect("valid InlineVec layout");
            // SAFETY: `heap` was allocated by `grow_to` with this layout.
            unsafe { dealloc(self.heap.cast(), layout) };
            self.heap = ptr::null_mut();
            self.cap = 0;
        }
    }

    pub(crate) fn reserve_exact(&mut self, additional: usize) {
        let needed = self
            .len()
            .checked_add(additional)
            .expect("InlineVec capacity overflow");
        if needed > self.capacity() {
            self.grow_to(needed);
        }
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        let needed = self
            .len()
            .checked_add(additional)
            .expect("InlineVec capacity overflow");
        if needed > self.capacity() {
            self.grow_to(needed.max(self.capacity() * 2));
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, value: T) {
        if self.len() == self.capacity() {
            self.reserve(1);
        }
        // SAFETY: there is room at index `len`.
        unsafe { self.as_mut_ptr().add(self.len()).write(value) };
        self.len += 1;
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: the element at the old last index is initialized and now outside the length.
        Some(unsafe { self.as_ptr().add(self.len()).read() })
    }

    pub(crate) fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(
            index < len,
            "InlineVec::remove index {index} out of bounds ({len})"
        );
        // SAFETY: `index` is in bounds; the tail shifts down over the element read out.
        unsafe {
            let base = self.as_mut_ptr();
            let value = base.add(index).read();
            ptr::copy(base.add(index + 1), base.add(index), len - index - 1);
            self.len -= 1;
            value
        }
    }

    /// Keep the elements `keep` accepts, in order; drop the others.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let len = self.len();
        // A panicking `keep` leaks the unvisited tail rather than dropping anything twice.
        self.len = 0;
        let base = self.as_mut_ptr();
        let mut kept = 0;
        for index in 0..len {
            // SAFETY: indices `kept..len` hold initialized elements not yet visited or moved;
            // `kept <= index`, so the move target is free.
            unsafe {
                let element = base.add(index);
                if keep(&*element) {
                    if kept != index {
                        ptr::copy_nonoverlapping(element, base.add(kept), 1);
                    }
                    kept += 1;
                    self.len = kept as u32;
                } else {
                    ptr::drop_in_place(element);
                }
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        let len = self.len();
        self.len = 0;
        // SAFETY: the first `len` elements are initialized and now outside the length.
        unsafe { ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.as_mut_ptr(), len)) };
    }
}

impl<T, const N: usize> Drop for InlineVec<T, N> {
    fn drop(&mut self) {
        self.clear();
        // SAFETY: every element was dropped above.
        unsafe { self.free_heap() };
    }
}

impl<T, const N: usize> std::ops::Deref for InlineVec<T, N> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: the first `len` elements are initialized.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T, const N: usize> std::ops::DerefMut for InlineVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let len = self.len();
        // SAFETY: the first `len` elements are initialized and uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
    }
}

impl<T, const N: usize> Default for InlineVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const N: usize> Clone for InlineVec<T, N> {
    fn clone(&self) -> Self {
        let mut copy = Self::with_capacity(self.len());
        for element in self.iter() {
            copy.push(element.clone());
        }
        copy
    }
}

impl<T, const N: usize> Extend<T> for InlineVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for element in iter {
            self.push(element);
        }
    }
}

impl<T, const N: usize> FromIterator<T> for InlineVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Self::new();
        vec.extend(iter);
        vec
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a InlineVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut InlineVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::InlineVec;
    use std::cell::Cell;
    use std::rc::Rc;

    /// Counts its drops, so every element is shown dropped exactly once.
    #[derive(Clone)]
    struct Counted(u32, Rc<Cell<usize>>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.1.set(self.1.get() + 1);
        }
    }

    fn values(vec: &InlineVec<Counted, 4>) -> Vec<u32> {
        vec.iter().map(|c| c.0).collect()
    }

    #[test]
    fn elements_stay_in_place_up_to_n_then_spill_once() {
        let drops = Rc::new(Cell::new(0));
        let mut vec: InlineVec<Counted, 4> = InlineVec::new();
        for i in 0..4 {
            vec.push(Counted(i, drops.clone()));
        }
        assert!(!vec.spilled());
        assert_eq!(vec.capacity(), 4);
        vec.push(Counted(4, drops.clone()));
        assert!(vec.spilled());
        assert_eq!(values(&vec), [0, 1, 2, 3, 4]);
        assert_eq!(
            drops.get(),
            0,
            "spilling moves elements without dropping them"
        );
        // Shrinking keeps the heap buffer.
        assert_eq!(vec.pop().map(|c| c.0), Some(4));
        assert_eq!(drops.get(), 1);
        assert!(vec.spilled());
        drop(vec);
        assert_eq!(drops.get(), 5);
    }

    #[test]
    fn remove_retain_and_clear_drop_each_element_once() {
        for count in [3u32, 9] {
            let drops = Rc::new(Cell::new(0));
            let mut vec: InlineVec<Counted, 4> =
                (0..count).map(|i| Counted(i, drops.clone())).collect();
            assert_eq!(vec.remove(1).0, 1);
            assert_eq!(drops.get(), 1);
            vec.retain(|c| c.0 % 2 == 0);
            let expected: Vec<u32> = (0..count).filter(|i| *i != 1 && i % 2 == 0).collect();
            assert_eq!(values(&vec), expected);
            let dropped = drops.get();
            assert_eq!(dropped as u32, count - expected.len() as u32);
            vec.clear();
            assert!(vec.is_empty());
            assert_eq!(drops.get(), count as usize);
            vec.push(Counted(7, drops.clone()));
            drop(vec);
            assert_eq!(drops.get(), count as usize + 1);
        }
    }

    #[test]
    fn clone_extend_reserve_and_indexing_follow_vec() {
        let drops = Rc::new(Cell::new(0));
        let mut vec: InlineVec<Counted, 4> = InlineVec::with_capacity(2);
        assert!(!vec.spilled());
        vec.extend((0..3).map(|i| Counted(i, drops.clone())));
        let copy = vec.clone();
        assert_eq!(values(&copy), [0, 1, 2]);
        vec.reserve_exact(10);
        assert!(vec.spilled() && vec.capacity() >= 13);
        vec[1] = Counted(10, drops.clone());
        assert_eq!(
            drops.get(),
            1,
            "assigning through IndexMut drops the old element"
        );
        vec.swap(0, 2);
        assert_eq!(values(&vec), [2, 10, 0]);
        let spilled_copy = InlineVec::<Counted, 4>::with_capacity(8);
        assert!(spilled_copy.spilled());
        drop((vec, copy, spilled_copy));
        assert_eq!(drops.get(), 1 + 3 + 3);
    }
}
