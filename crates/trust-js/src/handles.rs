//! S6 step 11a: rooted handles.
//!
//! Rust code that keeps an engine value outside the engine (host state, a resource, an
//! embedder's own structures) keeps it in a handle: a counted entry of its heap's handle table
//! ([`HandleTable`]), which every collection reads. A strong entry is a root. A weak entry is
//! cleared between a collection's phases once its target is found dead, before any cell is
//! freed, so a weak handle never yields a freed cell or a new one at a reused address. Each
//! Agent heap has one table, which a ShadowRealm tree shares. The table knows its heap's
//! identity and whether the heap still exists: a handle read through another heap's context, or
//! after its own Engine has dropped, is refused instead of naming a freed cell. Handles are made
//! from a scope's values and read back into a scope (S6 step 11b, `scope`).
//!
//! Collections read the table by kind ([`CollectionKind`]). A major collection reports every
//! strong entry as a root and clears every weak entry whose target died. A minor collection
//! reads only the entries written since the last collection (made, given a new value, or made
//! strong again), which the table records as they are written and each collection's marking
//! consumes beside the side tables' records (`gc_collect::ClientRecords`): an entry not written
//! since names an old cell, one the last collection marked, which no minor collection unmarks
//! or frees, or, if weak, nothing. Debug builds check that as each collection starts
//! ([`HandleTable::check_records`]). Handles write no cell, so the write barrier is not
//! involved.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::gc_core::{CellAge, CollectionKind};
use crate::value::Value;

/// One heap's handle table: see the module documentation.
pub(crate) struct HandleTable {
    /// The heap's identity, never reused (`value::heap_id`).
    heap_id: u64,
    /// Cleared by the heap's release (`value::HeapRelease`), before its cells go: every handle
    /// is inert from then on.
    alive: Cell<bool>,
    /// Set while a collection of the heap runs, from its marking until its sweep has freed the
    /// dead (`gc_collect`): no entry may be made, given a value or made strong then.
    collecting: Cell<bool>,
    slots: RefCell<Slots>,
}

/// The table's entries.
#[derive(Default)]
struct Slots {
    entries: Vec<Slot>,
    /// The free entries, reused last first. (Only the embedder's API makes entries.)
    #[cfg_attr(not(feature = "embed"), allow(dead_code))]
    free: Vec<u32>,
    /// The live strong entries.
    strong: usize,
    /// The live weak entries, cleared or not.
    weak: usize,
    /// The entries written since the last collection's marking consumed the list, each once.
    written: Vec<u32>,
    /// `written` as the collection in progress consumed it: the weak entries a minor collection
    /// examines.
    written_swept: Vec<u32>,
}

/// One entry.
#[derive(Default)]
struct Slot {
    /// `None` in a free entry, and in a weak entry whose target died.
    value: Option<Value>,
    /// The handles that name the entry; 0 in a free entry.
    count: u32,
    strong: bool,
    /// Listed in [`Slots::written`].
    recorded: bool,
}

impl Slots {
    /// Record that entry `index` was written since the last collection.
    #[cfg_attr(not(feature = "embed"), allow(dead_code))]
    fn record(&mut self, index: u32) {
        let slot = &mut self.entries[index as usize];
        if !slot.recorded {
            slot.recorded = true;
            self.written.push(index);
        }
    }
}

impl HandleTable {
    pub(crate) fn new(heap_id: u64) -> Rc<HandleTable> {
        Rc::new(HandleTable {
            heap_id,
            alive: Cell::new(true),
            collecting: Cell::new(false),
            slots: RefCell::default(),
        })
    }

    /// Lifetime metadata only: safe even after the heap's cells have been released.
    pub(crate) fn is_alive(&self) -> bool {
        self.alive.get()
    }

    /// A collection of the heap starts (`true`, before its marking) or ends (`false`, once its
    /// sweep has freed the dead, or when the heap deferred it).
    pub(crate) fn set_collecting(&self, collecting: bool) {
        self.collecting.set(collecting);
        if !collecting {
            self.slots.borrow_mut().written_swept.clear();
        }
    }

    /// The roots of a collection of `kind`, each with its entry's index: in a major collection
    /// every strong entry's value; in a minor one those of the strong entries written since the
    /// last collection (an entry not written since names an old cell, which the last collection
    /// marked).
    pub(crate) fn trace_roots(&self, kind: CollectionKind, mut root: impl FnMut(u32, &Value)) {
        let slots = self.slots.borrow();
        let mut visit = |index: u32| {
            let slot = &slots.entries[index as usize];
            if slot.strong && slot.count > 0 {
                if let Some(value) = &slot.value {
                    root(index, value);
                }
            }
        };
        match kind {
            CollectionKind::Major => (0..slots.entries.len() as u32).for_each(&mut visit),
            CollectionKind::Minor => slots.written.iter().copied().for_each(&mut visit),
        }
    }

    /// The collection in progress has marked: start recording for the next one, keeping what
    /// was recorded for this one's weak clearing. A deferred collection consumes nothing.
    pub(crate) fn consume(&self) {
        let mut slots = self.slots.borrow_mut();
        let Slots {
            entries,
            written,
            written_swept,
            ..
        } = &mut *slots;
        for &index in written.iter() {
            entries[index as usize].recorded = false;
        }
        written_swept.clear();
        std::mem::swap(written, written_swept);
        if written.capacity() > 4096 {
            written.shrink_to(64);
        }
    }

    /// Between a collection's phases, before any cell is freed: clear the weak entries whose
    /// targets `is_dead` says the marking left unmarked. A major collection examines every weak
    /// entry, a minor one those written since the last collection: an entry not written since
    /// names an old cell, which no minor collection frees. Returns how many it cleared.
    pub(crate) fn clear_dead_weak(
        &self,
        kind: CollectionKind,
        is_dead: impl Fn(u64) -> bool,
    ) -> usize {
        let mut slots = self.slots.borrow_mut();
        let Slots {
            entries,
            written_swept,
            ..
        } = &mut *slots;
        let mut cleared = 0;
        let mut clear = |slot: &mut Slot| {
            if slot.strong || slot.count == 0 {
                return;
            }
            if slot
                .value
                .as_ref()
                .and_then(Value::cell_word)
                .is_some_and(&is_dead)
            {
                slot.value = None;
                cleared += 1;
            }
        };
        match kind {
            CollectionKind::Major => entries.iter_mut().for_each(&mut clear),
            CollectionKind::Minor => {
                for &index in written_swept.iter() {
                    clear(&mut entries[index as usize]);
                }
            }
        }
        cleared
    }

    /// Debug builds, as each collection starts, beside the side tables' check
    /// (`MarkClient::check_records`), against the last collection's marks: an entry not written
    /// since the last collection names an old cell or, if weak, nothing, since a minor collection
    /// reads only the entries written since. Panics naming the entry, its strength and the cell.
    pub(crate) fn check_records(&self, heap: &crate::gc_core::Heap) {
        let slots = self.slots.borrow();
        for (index, slot) in slots.entries.iter().enumerate() {
            if slot.count == 0 || slot.recorded {
                continue;
            }
            let Some(word) = slot.value.as_ref().and_then(Value::cell_word) else {
                continue;
            };
            let age = heap.cell_age(word);
            if age != Some(CellAge::Old) {
                panic!(
                    "handle table of heap {}: {} entry {index}, not written since the last \
                     collection, names {} cell {word:#x}",
                    self.heap_id,
                    if slot.strong { "strong" } else { "weak" },
                    match age {
                        Some(CellAge::Young) => "a young (unmarked)",
                        Some(CellAge::Dead) => "a dead",
                        _ => "no allocated",
                    }
                );
            }
        }
    }

    /// The heap is being released (`value::HeapRelease`): before its cells go, the table drops
    /// every entry and every handle becomes inert. Its `get` errs, and dropping it frees only
    /// its share of the table's allocation.
    pub(crate) fn release(&self) {
        self.alive.set(false);
        *self.slots.borrow_mut() = Slots::default();
    }

    /// Each value the table's entries hold, strong or weak (the memory snapshot: the JavaScript
    /// values a host keeps are its handles').
    pub(crate) fn for_each_value(&self, mut f: impl FnMut(&Value)) {
        for slot in self.slots.borrow().entries.iter() {
            if slot.count > 0 {
                if let Some(value) = &slot.value {
                    f(value);
                }
            }
        }
    }

    /// The live entries, strong and weak (the memory snapshot).
    pub(crate) fn counts(&self) -> (usize, usize) {
        let slots = self.slots.borrow();
        (slots.strong, slots.weak)
    }

    /// Forget that entry `index` was written (tests: a write the record missed).
    #[cfg(all(test, feature = "embed", debug_assertions))]
    pub(crate) fn forget_written(&self, index: u32) {
        let mut slots = self.slots.borrow_mut();
        slots.written.retain(|&written| written != index);
        slots.entries[index as usize].recorded = false;
    }
}

#[cfg(feature = "embed")]
pub use api::{CountedHandle, Handle, HandleError, WeakHandle};

/// The embedder's API over the table (`trust_js::embed`).
#[cfg(feature = "embed")]
mod api {
    use super::{HandleTable, Slot};
    use crate::gc_core::CellAge;
    use crate::scope::{Ctx, Value};
    use crate::value::GcHeap;
    use std::rc::Rc;

    /// The heap of a scope's context.
    fn heap_of<'a>(cx: &'a Ctx<'_>) -> &'a GcHeap {
        &cx.interp().gc_heap
    }

    /// Why a handle could not be made or read.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum HandleError {
        /// The handle belongs to another Engine's heap than the context it was read through,
        /// or the value to another heap than the context's.
        Foreign,
        /// The handle's heap has been released: its Engine has dropped.
        Released,
        /// A collection of the heap is running (a host sweep or a finalizer): no handle can be
        /// made, given a value or made strong then. Debug builds panic instead.
        Collecting,
    }

    impl std::fmt::Display for HandleError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(match self {
                HandleError::Foreign => "the handle or value belongs to another engine's heap",
                HandleError::Released => "the handle's engine has been dropped",
                HandleError::Collecting => "a handle cannot be written while a collection runs",
            })
        }
    }

    impl std::error::Error for HandleError {}

    impl HandleTable {
        /// Whether a handle of this table may be read through a context of the heap `heap_id`.
        fn check_owner(&self, heap_id: u64) -> Result<(), HandleError> {
            if !self.alive.get() {
                Err(HandleError::Released)
            } else if self.heap_id != heap_id {
                Err(HandleError::Foreign)
            } else {
                Ok(())
            }
        }

        /// Whether an entry may be written: the heap lives, and no collection runs (debug
        /// builds panic if one does).
        fn check_writable(&self) -> Result<(), HandleError> {
            if !self.alive.get() {
                return Err(HandleError::Released);
            }
            if self.collecting.get() {
                if cfg!(debug_assertions) {
                    panic!("a handle was written while a collection of its heap runs");
                }
                return Err(HandleError::Collecting);
            }
            Ok(())
        }

        /// A new entry holding `value`, strong or weak, named by one handle; recorded.
        fn make(&self, value: crate::value::Value, strong: bool) -> Result<u32, HandleError> {
            self.check_writable()?;
            let mut slots = self.slots.borrow_mut();
            let index = match slots.free.pop() {
                Some(index) => index,
                None => {
                    let index = u32::try_from(slots.entries.len())
                        .expect("handle table index space exhausted");
                    slots.entries.push(Slot::default());
                    index
                }
            };
            {
                let slot = &mut slots.entries[index as usize];
                slot.value = Some(value);
                slot.count = 1;
                slot.strong = strong;
            }
            if strong {
                slots.strong += 1;
            } else {
                slots.weak += 1;
            }
            slots.record(index);
            Ok(index)
        }

        /// Another handle names entry `index` (`Clone`).
        fn retain(&self, index: u32) {
            if self.alive.get() {
                let mut slots = self.slots.borrow_mut();
                let slot = &mut slots.entries[index as usize];
                slot.count = slot.count.checked_add(1).expect("handle count overflow");
            }
        }

        /// A handle naming entry `index` drops; the last frees it. Once the heap is released,
        /// nothing is left to free.
        fn drop_handle(&self, index: u32) {
            if !self.alive.get() {
                return;
            }
            let mut slots = self.slots.borrow_mut();
            let strong = {
                let slot = &mut slots.entries[index as usize];
                slot.count -= 1;
                if slot.count > 0 {
                    return;
                }
                slot.value = None;
                slot.strong
            };
            if strong {
                slots.strong -= 1;
            } else {
                slots.weak -= 1;
            }
            slots.free.push(index);
        }

        /// Entry `index`'s value (`None` once a weak entry's target has died).
        fn value(&self, index: u32) -> Option<crate::value::Value> {
            self.slots.borrow().entries[index as usize].value
        }

        /// Give entry `index` a new value (`value` is the heap's); recorded.
        fn set_value(&self, index: u32, value: crate::value::Value) -> Result<(), HandleError> {
            self.check_writable()?;
            let mut slots = self.slots.borrow_mut();
            slots.entries[index as usize].value = Some(value);
            slots.record(index);
            Ok(())
        }

        /// Make entry `index` strong or weak in place. Made strong it is recorded: its target
        /// may be young.
        fn set_strong(&self, index: u32, strong: bool) -> Result<(), HandleError> {
            if strong {
                self.check_writable()?;
            } else if !self.alive.get() {
                return Err(HandleError::Released);
            }
            let mut slots = self.slots.borrow_mut();
            if slots.entries[index as usize].strong == strong {
                return Ok(());
            }
            slots.entries[index as usize].strong = strong;
            if strong {
                slots.weak -= 1;
                slots.strong += 1;
                slots.record(index);
            } else {
                slots.strong -= 1;
                slots.weak += 1;
            }
            Ok(())
        }
    }

    /// Whether `value` may enter `heap`'s table: a cell value must be one of `heap`'s live cells.
    fn check_value(heap: &GcHeap, value: &crate::value::Value) -> Result<(), HandleError> {
        heap.handles.check_writable()?;
        if let Some(word) = value.cell_word() {
            let core = heap
                .core
                .try_borrow()
                .map_err(|_| HandleError::Collecting)?;
            if !matches!(core.cell_age(word), Some(CellAge::Old | CellAge::Young)) {
                return Err(HandleError::Foreign);
            }
        }
        Ok(())
    }

    /// A new entry of `cx`'s heap holding `value`.
    fn make(
        cx: &Ctx<'_>,
        value: &Value<'_>,
        strong: bool,
    ) -> Result<(Rc<HandleTable>, u32), HandleError> {
        let heap = heap_of(cx);
        let value = value.into_internal();
        check_value(heap, &value)?;
        let slot = heap.handles.make(value, strong)?;
        Ok((heap.handles.clone(), slot))
    }

    /// A strong handle: keeps an engine value alive, wherever the handle is stored, until the
    /// handle drops. Clones share one entry of the heap's handle table, and the last to drop
    /// frees it. A handle is made from a scope's value and read into a scope of its own Engine;
    /// with another Engine's context, or once its Engine has dropped (it is then inert),
    /// reading it errs. It is neither `Send` nor `Sync`.
    pub struct Handle {
        table: Rc<HandleTable>,
        slot: u32,
    }

    impl Handle {
        /// Hold `value`, one of `cx`'s values.
        pub fn new<'s>(cx: &Ctx<'s>, value: &Value<'s>) -> Result<Handle, HandleError> {
            let (table, slot) = make(cx, value, true)?;
            Ok(Handle { table, slot })
        }

        /// The value, read into the scope `cx`, a scope of the handle's Engine.
        pub fn get<'s>(&self, cx: &Ctx<'s>) -> Result<Value<'s>, HandleError> {
            self.table.check_owner(heap_of(cx).handles.heap_id)?;
            Ok(Value::from_internal(
                self.table
                    .value(self.slot)
                    .expect("a strong handle's entry keeps its value"),
            ))
        }

        /// Hold `value` instead, one of `cx`'s values; every clone of this handle sees it.
        pub fn set<'s>(&self, cx: &Ctx<'s>, value: &Value<'s>) -> Result<(), HandleError> {
            let heap = heap_of(cx);
            self.table.check_owner(heap.handles.heap_id)?;
            let value = value.into_internal();
            check_value(heap, &value)?;
            self.table.set_value(self.slot, value)
        }

        /// The handle's entry (tests).
        #[cfg(test)]
        pub(crate) fn slot(&self) -> u32 {
            self.slot
        }

        /// A weak handle to the same value, in an entry of its own.
        pub fn downgrade(&self) -> Result<WeakHandle, HandleError> {
            if !self.table.alive.get() {
                return Err(HandleError::Released);
            }
            let value = self
                .table
                .value(self.slot)
                .expect("a strong handle's entry keeps its value");
            let slot = self.table.make(value, false)?;
            Ok(WeakHandle {
                table: self.table.clone(),
                slot,
            })
        }
    }

    impl Clone for Handle {
        fn clone(&self) -> Handle {
            self.table.retain(self.slot);
            Handle {
                table: self.table.clone(),
                slot: self.slot,
            }
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            self.table.drop_handle(self.slot);
        }
    }

    impl std::fmt::Debug for Handle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Handle(heap {}, entry {})",
                self.table.heap_id, self.slot
            )
        }
    }

    /// A weak handle: names an engine value without keeping it alive. Once a collection finds
    /// the value dead, the handle is cleared, between the collection's phases and before the
    /// value's cell is freed, so it never yields a freed value or a new one at the same address.
    /// Primitive values are never cleared. Clones share one entry; as [`Handle`] otherwise. It is
    /// also how a [`HostTrace`](crate::embed::HostTrace) reports the values of its graph.
    pub struct WeakHandle {
        table: Rc<HandleTable>,
        slot: u32,
    }

    impl WeakHandle {
        /// Name `value`, one of `cx`'s values, weakly.
        pub fn new<'s>(cx: &Ctx<'s>, value: &Value<'s>) -> Result<WeakHandle, HandleError> {
            let (table, slot) = make(cx, value, false)?;
            Ok(WeakHandle { table, slot })
        }

        /// The value while it lives (`None` once a collection has found it dead), read into the
        /// scope `cx`, a scope of the handle's Engine.
        pub fn upgrade<'s>(&self, cx: &Ctx<'s>) -> Result<Option<Value<'s>>, HandleError> {
            self.table.check_owner(heap_of(cx).handles.heap_id)?;
            Ok(self.table.value(self.slot).map(Value::from_internal))
        }

        /// Whether the handle will never yield its value again: a collection found it dead, or
        /// the handle's Engine has dropped. It reads no value, so it needs no context: a host's
        /// sweep ([`HostGc`](crate::embed::HostGc)), which runs between a collection's phases
        /// after the weak handles to dead values have been cleared, can ask it.
        pub fn is_cleared(&self) -> bool {
            !self.table.alive.get() || self.table.value(self.slot).is_none()
        }

        /// The handle's value for a collection of the heap `heap_id` reading a host's report
        /// (`HostGcVisitor`): `None` once cleared, or when the handle is another heap's.
        pub(crate) fn reported(&self, heap_id: u64) -> Option<crate::value::Value> {
            if !self.table.alive.get() || self.table.heap_id != heap_id {
                return None;
            }
            self.table.value(self.slot)
        }

        /// The handle's entry (tests).
        #[cfg(all(test, debug_assertions))]
        pub(crate) fn slot(&self) -> u32 {
            self.slot
        }
    }

    impl Clone for WeakHandle {
        fn clone(&self) -> WeakHandle {
            self.table.retain(self.slot);
            WeakHandle {
                table: self.table.clone(),
                slot: self.slot,
            }
        }
    }

    impl Drop for WeakHandle {
        fn drop(&mut self) {
            self.table.drop_handle(self.slot);
        }
    }

    impl std::fmt::Debug for WeakHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "WeakHandle(heap {}, entry {})",
                self.table.heap_id, self.slot
            )
        }
    }

    /// A counted handle, for references with Node-API's semantics: its entry is strong while
    /// the count is above 0 and weak at 0, flipping in place as the count crosses 0, so the
    /// value can die while no count holds it. Once its value has died, the handle stays
    /// cleared whatever the count. Not `Clone`: one handle per count.
    pub struct CountedHandle {
        table: Rc<HandleTable>,
        slot: u32,
        count: u32,
    }

    impl CountedHandle {
        /// Name `value`, one of `cx`'s values, with `count` (strongly when it is above 0).
        pub fn new<'s>(
            cx: &Ctx<'s>,
            value: &Value<'s>,
            count: u32,
        ) -> Result<CountedHandle, HandleError> {
            let (table, slot) = make(cx, value, count > 0)?;
            Ok(CountedHandle { table, slot, count })
        }

        /// The count.
        pub fn count(&self) -> u32 {
            self.count
        }

        /// The handle's entry (tests).
        #[cfg(test)]
        pub(crate) fn slot(&self) -> u32 {
            self.slot
        }

        /// Add one to the count, making the entry strong at 1; the new count.
        pub fn increment(&mut self) -> Result<u32, HandleError> {
            let count = self.count.checked_add(1).expect("handle count overflow");
            if self.count == 0 {
                self.table.set_strong(self.slot, true)?;
            }
            self.count = count;
            Ok(count)
        }

        /// Take one from the count, making the entry weak at 0 (a count of 0 stays 0); the new
        /// count.
        pub fn decrement(&mut self) -> Result<u32, HandleError> {
            if self.count == 1 {
                self.table.set_strong(self.slot, false)?;
            }
            self.count = self.count.saturating_sub(1);
            Ok(self.count)
        }

        /// The value while it lives (`None` once a collection has found it dead at a count of 0),
        /// read into the scope `cx`, a scope of the handle's Engine.
        pub fn get<'s>(&self, cx: &Ctx<'s>) -> Result<Option<Value<'s>>, HandleError> {
            self.table.check_owner(heap_of(cx).handles.heap_id)?;
            Ok(self.table.value(self.slot).map(Value::from_internal))
        }
    }

    impl Drop for CountedHandle {
        fn drop(&mut self) {
            self.table.drop_handle(self.slot);
        }
    }

    impl std::fmt::Debug for CountedHandle {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "CountedHandle(heap {}, entry {}, count {})",
                self.table.heap_id, self.slot, self.count
            )
        }
    }
}
