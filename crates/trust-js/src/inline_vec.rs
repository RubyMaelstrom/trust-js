//! P1 S3: a vector that keeps its first `N` elements in place (Decision 3's inline slots).
//!
//! Up to `N` elements live inside the owning value, so an object cell holding one needs no
//! second allocation for a few properties. The `N + 1`th element moves all of them to one heap
//! buffer, which is kept when the length shrinks again: a map that grew past `N` tends to stay
//! large, and moving back would make every shrink and regrow copy. The buffer's pointer takes
//! the place of the in-place elements (P1 S4), so the header is two 32-bit words.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr;

/// Where an [`InlineVec`] keeps its elements once they no longer fit in place.
pub(crate) trait Spill {
    /// A buffer for `layout` (of non-zero size); never null.
    fn alloc(layout: Layout) -> *mut u8;

    /// Give back a buffer [`Spill::alloc`] returned for `layout`.
    ///
    /// # Safety
    /// `ptr` came from `alloc(layout)` and is not used again; its elements were moved out or
    /// dropped.
    unsafe fn free(ptr: *mut u8, layout: Layout);
}

/// The Rust heap, for a vector outside the Agent heap's cells. The engine's cells spill into
/// storage blocks instead (`gc_storage::GcSpill`), which their collection frees and counts.
pub(crate) struct RustHeap;

impl Spill for RustHeap {
    fn alloc(layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout has a non-zero size.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        ptr
    }

    unsafe fn free(ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract: `alloc(layout)` returned `ptr`.
        unsafe { dealloc(ptr, layout) };
    }
}

pub(crate) struct InlineVec<T, const N: usize, S: Spill = RustHeap> {
    len: u32,
    /// Heap capacity once spilled; 0 while the elements are in place.
    cap: u32,
    data: Data<T, N>,
    spill: PhantomData<S>,
}

/// The in-place elements while `cap` is 0, else the heap buffer.
union Data<T, const N: usize> {
    inline: ManuallyDrop<[MaybeUninit<T>; N]>,
    heap: *mut T,
}

impl<T, const N: usize, S: Spill> InlineVec<T, N, S> {
    pub(crate) const fn new() -> Self {
        Self {
            len: 0,
            cap: 0,
            data: Data {
                inline: ManuallyDrop::new([const { MaybeUninit::uninit() }; N]),
            },
            spill: PhantomData,
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

    /// The heap buffer's capacity: 0 while the elements are in place.
    #[inline]
    pub(crate) fn heap_capacity(&self) -> usize {
        self.cap as usize
    }

    /// The spill buffer's address while the elements live in one: with [`Spill`] storage the
    /// collector frees (`gc_storage::GcSpill`), the edge the owner reports.
    #[inline]
    pub(crate) fn spill_ptr(&self) -> Option<*const u8> {
        if self.cap == 0 {
            return None;
        }
        // SAFETY: `cap` says which field of `data` is in use. Read only when spilled: the
        // inline field's bytes may be uninitialized.
        Some(unsafe { self.data.heap } as *const u8)
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        // SAFETY: `cap` says which field of `data` is in use.
        unsafe {
            if self.cap == 0 {
                self.data.inline.as_ptr().cast()
            } else {
                self.data.heap
            }
        }
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        // SAFETY: as in `as_ptr`.
        unsafe {
            if self.cap == 0 {
                (*self.data.inline).as_mut_ptr().cast()
            } else {
                self.data.heap
            }
        }
    }

    /// Move the elements to a heap buffer of exactly `capacity` (at least the length).
    fn grow_to(&mut self, capacity: usize) {
        debug_assert!(capacity >= self.len() && capacity > N);
        let capacity = u32::try_from(capacity).expect("InlineVec capacity overflow");
        let layout = Layout::array::<T>(capacity as usize).expect("InlineVec capacity overflow");
        assert!(layout.size() != 0, "InlineVec of zero-sized elements");
        let heap = S::alloc(layout).cast::<T>();
        // SAFETY: the old storage holds `len` initialized elements; the new buffer has room for
        // them and does not overlap the old one. They move bitwise; the old copies are forgotten
        // (in-place storage is then overwritten by the buffer's pointer).
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), heap, self.len());
            self.free_heap();
        }
        self.data.heap = heap;
        self.cap = capacity;
    }

    /// Release the heap buffer without touching its elements.
    ///
    /// # Safety
    /// The elements must have been moved out or dropped already.
    unsafe fn free_heap(&mut self) {
        if self.cap != 0 {
            let layout = Layout::array::<T>(self.cap as usize).expect("valid InlineVec layout");
            // SAFETY: the buffer was allocated by `grow_to` with this layout.
            unsafe { S::free(self.data.heap.cast(), layout) };
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

    /// Remove the element at `index`, moving the last element into its place.
    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        assert!(
            index < len,
            "InlineVec::swap_remove index {index} out of bounds ({len})"
        );
        // SAFETY: `index` and `len - 1` are in bounds; the element read out is replaced by the
        // last one, which then falls outside the length.
        unsafe {
            let base = self.as_mut_ptr();
            let value = base.add(index).read();
            ptr::copy(base.add(len - 1), base.add(index), 1);
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

    /// Where generated code finds this vector's parts ([`crate::value::JitVecLayout::Inline`]):
    /// the field offsets of the type, checked against live vectors in both states — elements in
    /// place and spilled to a heap buffer — by reading every word through them and comparing it
    /// with the safe accessors. `element` makes the probe's elements. `None` if any disagrees.
    pub(crate) fn probe_jit_layout(
        mut element: impl FnMut() -> T,
    ) -> Option<crate::value::JitVecLayout> {
        use std::mem::offset_of;
        let data = offset_of!(Self, data);
        let (len, cap) = (offset_of!(Self, len), offset_of!(Self, cap));
        let inline = data + offset_of!(Data<T, N>, inline);
        let heap = data + offset_of!(Data<T, N>, heap);
        let at = |vec: &Self, off: usize| (vec as *const Self).cast::<u8>().wrapping_add(off);
        // SAFETY (both reads): `off` is a measured offset of a field of that type within `vec`.
        let word = |vec: &Self, off: usize| unsafe { at(vec, off).cast::<u32>().read() } as usize;
        let pointer = |vec: &Self, off: usize| unsafe { at(vec, off).cast::<usize>().read() };
        let mut in_place = Self::new();
        in_place.push(element());
        // Length N + 1 below capacity N + 2, so every word read is distinct.
        let mut spilled = Self::with_capacity(N + 2);
        for _ in 0..=N {
            spilled.push(element());
        }
        let in_place_ok = !in_place.spilled()
            && word(&in_place, len) == in_place.len()
            && word(&in_place, cap) == 0
            && std::ptr::eq(at(&in_place, inline), in_place.as_ptr().cast::<u8>());
        let spilled_ok = spilled.spilled()
            && word(&spilled, len) == spilled.len()
            && word(&spilled, cap) == spilled.capacity()
            && spilled.len() != spilled.capacity()
            && pointer(&spilled, heap) == spilled.as_ptr() as usize;
        (in_place_ok && spilled_ok && heap.is_multiple_of(8)).then_some(
            crate::value::JitVecLayout::Inline {
                len,
                cap,
                inline,
                heap,
                inline_capacity: N,
            },
        )
    }
}

impl<T, const N: usize, S: Spill> Drop for InlineVec<T, N, S> {
    fn drop(&mut self) {
        self.clear();
        // SAFETY: every element was dropped above.
        unsafe { self.free_heap() };
    }
}

impl<T, const N: usize, S: Spill> std::ops::Deref for InlineVec<T, N, S> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: the first `len` elements are initialized.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T, const N: usize, S: Spill> std::ops::DerefMut for InlineVec<T, N, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        let len = self.len();
        // SAFETY: the first `len` elements are initialized and uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
    }
}

impl<T, const N: usize, S: Spill> Default for InlineVec<T, N, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, const N: usize, S: Spill> Clone for InlineVec<T, N, S> {
    fn clone(&self) -> Self {
        let mut copy = Self::with_capacity(self.len());
        for element in self.iter() {
            copy.push(element.clone());
        }
        copy
    }
}

impl<T, const N: usize, S: Spill> Extend<T> for InlineVec<T, N, S> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for element in iter {
            self.push(element);
        }
    }
}

impl<T, const N: usize, S: Spill> FromIterator<T> for InlineVec<T, N, S> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Self::new();
        vec.extend(iter);
        vec
    }
}

/// The elements of an [`InlineVec`], by value.
pub(crate) struct IntoIter<T, const N: usize, S: Spill = RustHeap> {
    vec: InlineVec<T, N, S>,
    next: usize,
}

impl<T, const N: usize, S: Spill> Iterator for IntoIter<T, N, S> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.next == self.vec.len() {
            return None;
        }
        // SAFETY: elements `next..len` are initialized and not yet moved out.
        let element = unsafe { self.vec.as_ptr().add(self.next).read() };
        self.next += 1;
        Some(element)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.vec.len() - self.next;
        (left, Some(left))
    }
}

impl<T, const N: usize, S: Spill> Drop for IntoIter<T, N, S> {
    fn drop(&mut self) {
        let (next, len) = (self.next, self.vec.len());
        // The vector's own drop then only frees the buffer; elements before `next` were moved
        // out, and the rest are dropped here.
        self.vec.len = 0;
        // SAFETY: elements `next..len` are initialized and not yet moved out.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.vec.as_mut_ptr().add(next),
                len - next,
            ))
        };
    }
}

impl<T, const N: usize, S: Spill> IntoIterator for InlineVec<T, N, S> {
    type Item = T;
    type IntoIter = IntoIter<T, N, S>;

    fn into_iter(self) -> IntoIter<T, N, S> {
        IntoIter { vec: self, next: 0 }
    }
}

impl<'a, T, const N: usize, S: Spill> IntoIterator for &'a InlineVec<T, N, S> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize, S: Spill> IntoIterator for &'a mut InlineVec<T, N, S> {
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
    fn swap_remove_and_owning_iteration_drop_each_element_once() {
        for count in [1u32, 3, 9] {
            let drops = Rc::new(Cell::new(0));
            let mut vec: InlineVec<Counted, 4> =
                (0..count).map(|i| Counted(i, drops.clone())).collect();
            assert_eq!(vec.swap_remove(0).0, 0);
            assert_eq!(drops.get(), 1);
            let mut expected: Vec<u32> = (1..count).collect();
            if let Some(last) = expected.pop() {
                expected.insert(0, last);
            }
            assert_eq!(values(&vec), expected);
            assert_eq!(vec.heap_capacity() != 0, count > 4);
            // Take some elements by value, then drop the iterator with the rest unread.
            let mut iter = vec.into_iter();
            let taken: Vec<u32> = iter.by_ref().take(1).map(|c| c.0).collect();
            assert_eq!(taken, expected[..taken.len()]);
            assert_eq!(drops.get(), 1 + taken.len());
            drop(iter);
            assert_eq!(drops.get(), count as usize);
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
