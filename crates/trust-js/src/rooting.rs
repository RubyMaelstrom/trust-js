//! I1 rooting for values held in Rust-heap buffers across safepoints.
//!
//! The conservative stack scan (Decision 0) finds handles in native frames and registers, never
//! inside the Rust-heap buffers those frames point to. A buffer of values that must survive a
//! safepoint — an argument list under evaluation, a spread's accumulated elements, a built-in's
//! working copy — is a [`RootedVec`] (owned, growable), is registered for a scope with
//! [`root_slice`] (borrowed, fixed), or lives in a [`Rooted`] (any traceable state); values
//! entering a structure that cannot be traced are kept in [`ValueRoots`]. Without `gc-heap` all
//! of these are plain buffers: reference counts keep the values alive.

use crate::value::Value;

/// A `Vec<Value>` whose elements are GC roots while it lives (feature `gc-heap`).
pub(crate) struct RootedVec {
    /// Boxed so the registered address stays put when the `RootedVec` moves.
    #[cfg(feature = "gc-heap")]
    #[allow(clippy::box_collection)]
    values: Box<Vec<Value>>,
    #[cfg(feature = "gc-heap")]
    slot: registry::Slot,
    #[cfg(not(feature = "gc-heap"))]
    values: Vec<Value>,
}

impl RootedVec {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    #[inline]
    pub(crate) fn from_vec(values: Vec<Value>) -> Self {
        #[cfg(feature = "gc-heap")]
        {
            let values = Box::new(values);
            let slot = registry::register(registry::Entry::Vec(std::ptr::from_ref(&*values)));
            RootedVec { values, slot }
        }
        #[cfg(not(feature = "gc-heap"))]
        RootedVec { values }
    }

    /// The values, no longer rooted: the caller roots them again before the next safepoint
    /// (a call roots its argument list for its duration).
    #[inline]
    pub(crate) fn into_vec(mut self) -> Vec<Value> {
        let values: &mut Vec<Value> = &mut self.values;
        std::mem::take(values)
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
        &self.values
    }
}

impl std::ops::DerefMut for RootedVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<Value> {
        &mut self.values
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
        let value = self.values.get(self.next)?.clone();
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

#[cfg(feature = "gc-heap")]
impl Drop for RootedVec {
    #[inline]
    fn drop(&mut self) {
        registry::unregister(self.slot);
    }
}

/// What [`Rooted`] can hold: everything traceable (gc-heap), anything otherwise.
#[cfg(feature = "gc-heap")]
pub(crate) use crate::gc_trace::GcTrace as Traceable;
#[cfg(not(feature = "gc-heap"))]
pub(crate) trait Traceable {}
#[cfg(not(feature = "gc-heap"))]
impl<T: ?Sized> Traceable for T {}

/// A value whose GC edges are roots while it lives (feature `gc-heap`): engine state built up
/// in a Rust structure across safepoints, such as a class evaluation's decorators.
pub(crate) struct Rooted<T: Traceable + 'static> {
    /// Boxed so the registered address stays put.
    #[cfg(feature = "gc-heap")]
    value: Box<T>,
    #[cfg(feature = "gc-heap")]
    slot: registry::Slot,
    #[cfg(not(feature = "gc-heap"))]
    value: T,
}

impl<T: Traceable + 'static> Rooted<T> {
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        #[cfg(feature = "gc-heap")]
        {
            let value = Box::new(value);
            let traced: &dyn crate::gc_trace::GcTrace = &*value;
            let slot = registry::register(registry::Entry::Traced(std::ptr::from_ref(traced)));
            Rooted { value, slot }
        }
        #[cfg(not(feature = "gc-heap"))]
        Rooted { value }
    }
}

impl<T: Traceable + 'static> std::ops::Deref for Rooted<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T: Traceable + 'static> std::ops::DerefMut for Rooted<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

#[cfg(feature = "gc-heap")]
impl<T: Traceable + 'static> Drop for Rooted<T> {
    #[inline]
    fn drop(&mut self) {
        registry::unregister(self.slot);
    }
}

/// Keeps values alive while it lives, beside a Rust-heap structure the scan cannot see (a
/// built-in's groups or worklist): `keep` each value as it enters the structure. Without
/// `gc-heap` it holds nothing and costs nothing.
#[derive(Default)]
pub(crate) struct ValueRoots {
    #[cfg(feature = "gc-heap")]
    values: RootedVec,
}

impl ValueRoots {
    #[inline]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn keep(&mut self, value: &Value) {
        #[cfg(feature = "gc-heap")]
        self.values.push(value.clone());
        #[cfg(not(feature = "gc-heap"))]
        let _ = value;
    }
}

/// Roots a borrowed slice of values while it lives; see [`root_slice`].
#[must_use]
pub(crate) struct SliceRoot<'a> {
    #[cfg(feature = "gc-heap")]
    slot: registry::Slot,
    _values: std::marker::PhantomData<&'a [Value]>,
}

/// Root `values` until the returned guard drops (feature `gc-heap`; free otherwise). The
/// borrow keeps the slice from being changed or freed meanwhile.
#[inline]
pub(crate) fn root_slice(values: &[Value]) -> SliceRoot<'_> {
    #[cfg(not(feature = "gc-heap"))]
    let _ = values;
    SliceRoot {
        #[cfg(feature = "gc-heap")]
        slot: registry::register(registry::Entry::Slice(std::ptr::from_ref(values))),
        _values: std::marker::PhantomData,
    }
}

#[cfg(feature = "gc-heap")]
impl Drop for SliceRoot<'_> {
    #[inline]
    fn drop(&mut self) {
        registry::unregister(self.slot);
    }
}

/// Trace every value rooted on this thread.
#[cfg(feature = "gc-heap")]
pub(crate) fn trace_rooted(sink: &mut dyn crate::gc_trace::GcSink) {
    registry::trace(sink);
}

#[cfg(feature = "gc-heap")]
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
            match table.free.pop() {
                Some(index) => {
                    table.entries[index] = Some(entry);
                    Slot(index)
                }
                None => {
                    table.entries.push(Some(entry));
                    Slot(table.entries.len() - 1)
                }
            }
        })
    }

    pub(super) fn unregister(slot: Slot) {
        // A thread-local destructor may run after the table's own.
        let _ = TABLE.try_with(|table| {
            let mut table = table.borrow_mut();
            table.entries[slot.0] = None;
            if slot.0 + 1 == table.entries.len() {
                table.entries.pop();
                while let Some(None) = table.entries.last() {
                    table.entries.pop();
                }
                let len = table.entries.len();
                table.free.retain(|&index| index < len);
            } else {
                table.free.push(slot.0);
            }
        });
    }

    pub(super) fn trace(sink: &mut dyn crate::gc_trace::GcSink) {
        TABLE.with(|table| {
            for entry in table.borrow().entries.iter().flatten() {
                // SAFETY: an entry is registered only while its owner (the box of a `RootedVec`
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
