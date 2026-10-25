//! I1 rooting for values held in Rust-heap buffers across safepoints.
//!
//! The conservative stack scan (Decision 0) finds handles in native frames and registers, never
//! inside the Rust-heap buffers those frames point to. A buffer of values that must survive a
//! safepoint — an argument list under evaluation, a spread's accumulated elements, a built-in's
//! working copy — is a [`RootedVec`] (owned, growable), is registered for a scope with
//! [`root_slice`] (borrowed, fixed), or lives in a [`Rooted`] (any traceable state); values
//! entering a structure that cannot be traced are kept in [`ValueRoots`].

use crate::value::Value;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

/// Raw ownership, without Box's pointee retagging when the root wrapper moves. Registry
/// pointers are derived directly from this capability, before any temporary shared reference.
struct Storage<T>(NonNull<T>);
impl<T> Storage<T> {
    fn new(value: T) -> Self {
        Self(NonNull::new(Box::into_raw(Box::new(value))).expect("Box is non-null"))
    }
    fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
    fn into_inner(self) -> T {
        let this = ManuallyDrop::new(self);
        // SAFETY: the sole owner, after its registry entry was removed.
        unsafe { *Box::from_raw(this.0.as_ptr()) }
    }
}
impl<T> Drop for Storage<T> {
    fn drop(&mut self) {
        // SAFETY: sole raw ownership; the containing root unregisters before this field drops.
        unsafe {
            drop(Box::from_raw(self.0.as_ptr()));
        }
    }
}

/// A `Vec<Value>` whose elements are GC roots while it lives.
pub(crate) struct RootedVec {
    values: Storage<Vec<Value>>,
    slot: registry::Slot,
}

impl RootedVec {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    #[inline]
    pub(crate) fn from_vec(values: Vec<Value>) -> Self {
        let values = Storage::new(values);
        let slot = registry::register(registry::Entry::Vec(values.as_ptr().cast_const()));
        RootedVec { values, slot }
    }

    /// The values, no longer rooted: the caller roots them again before the next safepoint
    /// (a call roots its argument list for its duration).
    #[inline]
    pub(crate) fn into_vec(self) -> Vec<Value> {
        let this = ManuallyDrop::new(self);
        registry::unregister(this.slot);
        // SAFETY: this no longer runs Drop; transfer its sole storage owner after unregistering.
        unsafe { std::ptr::read(&this.values) }.into_inner()
    }

    // These are inherent methods: evaluating push(callback()) borrows only the raw-owner
    // wrapper, not the Vec payload. Every payload loan ends before iterator/predicate code.
    pub(crate) fn push(&mut self, value: Value) {
        unsafe {
            (&mut *self.values.as_ptr()).push(value);
        }
    }
    pub(crate) fn pop(&mut self) -> Option<Value> {
        unsafe { (&mut *self.values.as_ptr()).pop() }
    }
    pub(crate) fn set(&mut self, index: usize, value: Value) {
        unsafe {
            (&mut *self.values.as_ptr())[index] = value;
        }
    }
    pub(crate) fn insert(&mut self, index: usize, value: Value) {
        unsafe {
            (&mut *self.values.as_ptr()).insert(index, value);
        }
    }
    pub(crate) fn remove(&mut self, index: usize) -> Value {
        unsafe { (&mut *self.values.as_ptr()).remove(index) }
    }
    pub(crate) fn truncate(&mut self, len: usize) {
        unsafe {
            (&mut *self.values.as_ptr()).truncate(len);
        }
    }
    pub(crate) fn clear(&mut self) {
        self.truncate(0);
    }
    #[cfg(test)]
    #[allow(dead_code)] // Standalone actual-module Miri consumer exercises reallocation.
    pub(crate) fn reserve(&mut self, additional: usize) {
        unsafe {
            (&mut *self.values.as_ptr()).reserve(additional);
        }
    }
    pub(crate) fn extend(&mut self, values: impl IntoIterator<Item = Value>) {
        for value in values {
            self.push(value);
        }
    }
    pub(crate) fn append(&mut self, values: &mut Vec<Value>) {
        self.extend(std::mem::take(values));
    }
    /// The caller uses raw words only, ends each writable loan before a safepoint, and retains
    /// this owner without reallocating its buffer. No mutable Rust slice/reference to the
    /// payload may span registry tracing.
    #[cfg(test)]
    #[allow(dead_code)] // Standalone Miri consumer models the canonical raw native-word loan.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut Value {
        unsafe { (&mut *self.values.as_ptr()).as_mut_ptr() }
    }
}

impl Default for RootedVec {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for RootedVec {
    type Target = Vec<Value>;
    #[inline]
    fn deref(&self) -> &Vec<Value> {
        // SAFETY: the stable allocation is owned and no mutable payload reference escapes.
        unsafe { &*self.values.as_ptr() }
    }
}

/// A copy rooted on its own.
impl Clone for RootedVec {
    fn clone(&self) -> Self {
        RootedVec::from_vec(self.to_vec())
    }
}

impl std::fmt::Debug for RootedVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RootedVec({} values)", self.len())
    }
}

impl<'a> IntoIterator for &'a RootedVec {
    type Item = &'a Value;
    type IntoIter = std::slice::Iter<'a, Value>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterating a `RootedVec` by value keeps the values not yet taken rooted, so a loop over a key
/// list may run JavaScript between elements.
impl IntoIterator for RootedVec {
    type Item = Value;
    type IntoIter = RootedIntoIter;
    #[inline]
    fn into_iter(self) -> RootedIntoIter {
        RootedIntoIter {
            values: self,
            next: 0,
        }
    }
}

/// See [`RootedVec`]'s by-value `IntoIterator`.
pub(crate) struct RootedIntoIter {
    values: RootedVec,
    next: usize,
}

impl Iterator for RootedIntoIter {
    type Item = Value;
    #[inline]
    fn next(&mut self) -> Option<Value> {
        let value = *self.values.get(self.next)?;
        self.next += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.values.len() - self.next;
        (left, Some(left))
    }
}

impl ExactSizeIterator for RootedIntoIter {}

impl FromIterator<Value> for RootedVec {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        let mut rooted = RootedVec::new();
        rooted.extend(iter);
        rooted
    }
}

impl Drop for RootedVec {
    #[inline]
    fn drop(&mut self) {
        registry::unregister(self.slot);
    }
}

/// What [`Rooted`] can hold: everything traceable.
pub(crate) use crate::gc_trace::GcTrace as Traceable;

/// A value whose GC edges are roots while it lives: engine state built up in a Rust structure
/// across safepoints, such as a class evaluation's decorators. Mutable generic state uses
/// checked interior loans (e.g. RefCell) that end before callbacks/tracing; no &mut T escapes.
/// T's destructor must not execute JavaScript or collect: it runs after unregistering. Current
/// payloads, including JsonRecord's iterative teardown, satisfy this internal contract.
pub(crate) struct Rooted<T: Traceable + 'static> {
    value: Storage<T>,
    slot: registry::Slot,
}

impl<T: Traceable + 'static> Rooted<T> {
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        let value = Storage::new(value);
        let traced = value.as_ptr() as *const dyn crate::gc_trace::GcTrace;
        let slot = registry::register(registry::Entry::Traced(traced));
        Rooted { value, slot }
    }

    /// Transfer unrooted state. Its GC edges need a new owner/root before the next safepoint.
    pub(crate) fn into_inner(self) -> T {
        let this = ManuallyDrop::new(self);
        registry::unregister(this.slot);
        unsafe { std::ptr::read(&this.value) }.into_inner()
    }

    /// Return the previous, now-unrooted state; root/publish it before another safepoint.
    pub(crate) fn replace(&mut self, value: T) -> T {
        // Argument evaluation has ended. No mutable payload reference escapes this operation.
        unsafe { std::ptr::replace(self.value.as_ptr(), value) }
    }
}

impl<T: Traceable + 'static> std::ops::Deref for Rooted<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.value.as_ptr() }
    }
}

impl<T: Traceable + 'static> Drop for Rooted<T> {
    #[inline]
    fn drop(&mut self) {
        registry::unregister(self.slot);
    }
}

/// Keeps values alive while it lives, beside a Rust-heap structure the scan cannot see (a
/// built-in's groups or worklist): `keep` each value as it enters the structure.
#[derive(Default)]
pub(crate) struct ValueRoots {
    values: RootedVec,
}

impl ValueRoots {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn keep(&mut self, value: &Value) {
        self.values.push(*value);
    }
}

/// Roots a borrowed slice of values while it lives; see [`root_slice`].
#[must_use]
pub(crate) struct SliceRoot<'a> {
    slot: registry::Slot,
    _values: std::marker::PhantomData<&'a [Value]>,
}

/// Root `values` until the returned guard drops. The borrow keeps the slice from being changed
/// or freed meanwhile.
#[inline]
pub(crate) fn root_slice(values: &[Value]) -> SliceRoot<'_> {
    SliceRoot {
        slot: registry::register(registry::Entry::Slice(std::ptr::from_ref(values))),
        _values: std::marker::PhantomData,
    }
}

impl Drop for SliceRoot<'_> {
    #[inline]
    fn drop(&mut self) {
        registry::unregister(self.slot);
    }
}

/// Trace every value rooted on this thread.
pub(crate) fn trace_rooted(sink: &mut dyn crate::gc_trace::GcSink) {
    registry::trace(sink);
}

mod registry {
    use crate::gc_trace::GcTrace;
    use crate::value::Value;
    use std::cell::RefCell;

    #[derive(Clone, Copy)]
    pub(super) enum Entry {
        Vec(*const Vec<Value>),
        Slice(*const [Value]),
        Traced(*const dyn GcTrace),
    }

    /// An index into this thread's table.
    #[derive(Clone, Copy)]
    pub(super) struct Slot(usize);

    #[derive(Default)]
    struct Table {
        entries: Vec<Option<Entry>>,
        free: Vec<usize>,
    }

    thread_local! {
        static TABLE: RefCell<Table> = RefCell::default();
    }

    pub(super) fn register(entry: Entry) -> Slot {
        TABLE.with(|table| {
            let mut table = table.borrow_mut();
            // The free list may name slots the table has since dropped or refilled (see
            // `unregister`): only an empty slot below the length is reused.
            while let Some(index) = table.free.pop() {
                if table.entries.get(index).is_some_and(Option::is_none) {
                    table.entries[index] = Some(entry);
                    return Slot(index);
                }
            }
            table.entries.push(Some(entry));
            Slot(table.entries.len() - 1)
        })
    }

    pub(super) fn unregister(slot: Slot) {
        // A thread-local destructor may run after the table's own.
        let _ = TABLE.try_with(|table| {
            let mut table = table.borrow_mut();
            table.entries[slot.0] = None;
            if slot.0 + 1 == table.entries.len() {
                // The common, stack-ordered case: drop the empty tail. Free-list entries beyond
                // it go stale and are skipped when they come up (`register`), so this stays
                // constant-time amortized instead of filtering the list each time.
                table.entries.pop();
                while let Some(None) = table.entries.last() {
                    table.entries.pop();
                }
            } else {
                table.free.push(slot.0);
            }
        });
    }

    pub(super) fn trace(sink: &mut dyn crate::gc_trace::GcSink) {
        TABLE.with(|table| {
            for entry in table.borrow().entries.iter().flatten() {
                // SAFETY: an entry is registered only while its owner (the raw-owned storage of a `RootedVec`
                // or `Rooted`, or a `SliceRoot`'s borrow) keeps it alive and unmoved.
                unsafe {
                    match *entry {
                        Entry::Vec(values) => (*values).trace_gc(sink),
                        Entry::Slice(values) => (*values).trace_gc(sink),
                        Entry::Traced(value) => (*value).trace_gc(sink),
                    }
                }
            }
        });
    }
}
