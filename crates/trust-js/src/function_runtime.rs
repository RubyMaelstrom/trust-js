//! Heap-bound runtime caches of neutral parsed Functions. No entry retains its heap.
//! Primary access borrows its stable entry; only foreign entries need an Rc lease. Eviction
//! retires the cache's lease, while active frames and code leases retain immutable chunks.

use crate::ast::FunctionMaps;
use crate::bytecode::Chunk;
use crate::value::{active_heap_id, active_heap_lifetime, HeapLifetime};
use std::cell::{Cell, OnceCell, RefCell};
use std::ops::Deref;
use std::rc::Rc;

const FOREIGN_CACHE_LIMIT: usize = 4;

pub(crate) struct FunctionRuntime {
    pub(crate) heap: HeapLifetime,
    pub(crate) calls: Cell<u32>,
    pub(crate) code: OnceCell<Option<Rc<Chunk>>>,
    pub(crate) code2: OnceCell<Option<Rc<Chunk>>>,
    pub(crate) fn_maps: OnceCell<Rc<FunctionMaps>>,
    /// The runtime ScriptOrModule this heap's first closure of the function had. Closures with
    /// the same record share it instead of each boxing a copy (`UserCallableCache::own_origin`).
    /// Those closures trace it; it is compared by identity with a live active record before a
    /// new closure shares it, so its GC edges are never followed after its sharers die.
    pub(crate) origin: OnceCell<crate::agent_control::RuntimeOrigin>,
}

impl FunctionRuntime {
    pub(crate) fn new() -> Self {
        Self {
            heap: active_heap_lifetime(),
            calls: Cell::new(0),
            code: OnceCell::new(),
            code2: OnceCell::new(),
            fn_maps: OnceCell::new(),
            origin: OnceCell::new(),
        }
    }

    /// Realm specialization remains subordinate to heap ownership. Active frames keep their
    /// selected immutable chunk, including when later calls compile a second stage.
    pub(crate) fn execution_code_for(&self, genv: usize) -> Option<&Option<Rc<Chunk>>> {
        match self.code2.get() {
            Some(Some(chunk)) if !chunk.inline_realm_compatible(genv) => self.code.get(),
            Some(_) => self.code2.get(),
            None => self.code.get(),
        }
    }

    #[cfg(test)]
    pub(crate) fn execution_code(&self) -> Option<&Option<Rc<Chunk>>> {
        self.code2.get().or_else(|| self.code.get())
    }
}

impl Drop for FunctionRuntime {
    fn drop(&mut self) {
        if self.code.get().is_some_and(Option::is_some)
            || self.code2.get().is_some_and(Option::is_some)
        {
            // Call ICs keep raw addresses inside these set-once slots, including code-keyed
            // entries that can match a later closure of the same AST. Fence those entries
            // BEFORE Rust drops either slot or its chunks. Active frames retain chunk leases.
            crate::bytecode::invalidate_call_caches();
        }
    }
}

impl std::fmt::Debug for FunctionRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Old entries can survive heap release. Never format their cell-backed constants.
        f.debug_struct("FunctionRuntime")
            .field("heap", &self.heap)
            .field("calls", &self.calls.get())
            .field("compiled", &self.code.get().is_some())
            .field("inlined", &self.code2.get().is_some())
            .finish()
    }
}

#[derive(Debug, Default)]
struct ForeignRuntimes {
    entries: Vec<Rc<FunctionRuntime>>,
}

#[derive(Debug)]
pub(crate) struct FunctionRuntimeCache {
    identity: u64,
    primary: FunctionRuntime,
    // Foreign reuse is uncommon. Keep its Vec metadata out of each single-heap cache too.
    foreign: RefCell<Option<Box<ForeignRuntimes>>>,
}

pub(crate) enum RuntimeLease<'a> {
    Primary(&'a FunctionRuntime),
    Foreign(Rc<FunctionRuntime>),
}

impl Deref for RuntimeLease<'_> {
    type Target = FunctionRuntime;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Primary(entry) => entry,
            Self::Foreign(entry) => entry,
        }
    }
}

impl FunctionRuntimeCache {
    pub(crate) fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            identity: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            primary: FunctionRuntime::new(),
            foreign: RefCell::default(),
        }
    }

    #[inline]
    pub(crate) fn current(&self) -> RuntimeLease<'_> {
        let id = active_heap_id();
        if self.primary.heap.id == id {
            return RuntimeLease::Primary(&self.primary);
        }
        let mut foreign = self.foreign.borrow_mut();
        let foreign = &mut foreign.get_or_insert_with(Box::default).entries;
        if let Some(entry) = foreign.iter().find(|entry| entry.heap.id == id) {
            return RuntimeLease::Foreign(entry.clone());
        }
        // Prune only on a foreign miss, never on the primary hot path. Dropping an entry is
        // safe after heap release: its handles have no Drop and no cell data is inspected.
        foreign.retain(|entry| entry.heap.is_alive());
        if foreign.len() == FOREIGN_CACHE_LIMIT {
            foreign.remove(0);
        }
        let entry = crate::value::foreign_function_runtime(self.identity);
        foreign.push(entry.clone());
        RuntimeLease::Foreign(entry)
    }

    /// The record shared by closures created under the primary entry (see `FunctionRuntime`).
    pub(crate) fn primary_origin(&self) -> Option<&crate::agent_control::RuntimeOrigin> {
        self.primary.origin.get()
    }

    /// Only Rust metadata is safe to inspect on a dead entry. The caller chooses which live
    /// heaps belong to its census before following any cell-backed field.
    pub(crate) fn for_each(&self, mut visit: impl FnMut(&FunctionRuntime, bool)) {
        visit(&self.primary, false);
        if let Some(foreign) = self.foreign.borrow().as_deref() {
            for entry in &foreign.entries {
                visit(entry, true);
            }
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.foreign.borrow().as_deref().map_or(0, |foreign| {
                std::mem::size_of::<ForeignRuntimes>()
                    + foreign.entries.capacity() * std::mem::size_of::<Rc<FunctionRuntime>>()
            })
    }
}
