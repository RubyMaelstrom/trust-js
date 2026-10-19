//! Opt-in, post-collection managed-memory accounting.
//!
//! This deliberately reports requested payload/capacity bytes rather than guessing allocator
//! rounding or Rust's private `RcBox`/`HashMap` layouts. Every partial category says so in the
//! emitted record; unavailable owners are never represented by a misleading zero.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::rc::Rc;

use crate::interpreter::{Env, Interp, Scope};
use crate::value::{Callable, Exotic, Gc, Object, SymbolData, Value};

#[derive(Clone, Copy)]
enum Quality {
    Exact,
    LowerBound,
}

impl Quality {
    fn json(self) -> &'static str {
        match self {
            Quality::Exact => "exact",
            Quality::LowerBound => "lower_bound",
        }
    }
}

#[derive(Clone, Copy)]
struct Category {
    bytes: usize,
    quality: Quality,
    reason: Option<&'static str>,
    coverage_complete: bool,
}

#[derive(Clone, Copy)]
struct HostCategory {
    reported_bytes: usize,
    unavailable_entries: usize,
    identity_conflict: bool,
    opaque_storage: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SharedBackingAllocation {
    id: u64,
    bytes: usize,
}

#[derive(Clone, Default)]
struct SharedBackingCategory {
    allocations: Vec<SharedBackingAllocation>,
    unavailable_ids: usize,
}

impl SharedBackingCategory {
    fn bytes(&self) -> usize {
        self.allocations
            .iter()
            .map(|allocation| allocation.bytes)
            .fold(0usize, usize::saturating_add)
    }

    fn json(&self) -> String {
        if self.unavailable_ids != 0 {
            return format!(
                concat!(
                    "{{\"bytes\":null,\"quality\":\"unavailable\",",
                    "\"externally_shared\":true,",
                    "\"reason\":\"{} referenced Shared Data Block identities were absent from the registry\"}}"
                ),
                self.unavailable_ids
            );
        }
        let allocations = self
            .allocations
            .iter()
            .map(|allocation| {
                format!(
                    concat!(
                        "{{\"allocation_id\":\"shared-data-block:{}\",",
                        "\"bytes\":{},\"externally_shared\":true}}"
                    ),
                    allocation.id, allocation.bytes
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{\"bytes\":{},\"quality\":\"exact\",",
                "\"externally_shared\":true,\"allocations\":[{}]}}"
            ),
            self.bytes(),
            allocations
        )
    }
}

#[derive(Clone, Copy)]
struct WasmBackingCategory {
    bytes: usize,
    identity_conflict: bool,
    unclassified_external_entries: usize,
}

impl WasmBackingCategory {
    fn json(self) -> String {
        if self.identity_conflict {
            return format!(
                concat!(
                    "{{\"bytes\":{},\"quality\":\"lower_bound\",",
                    "\"reason\":\"conflicting byte counts were reported for one Wasm allocation identity\"}}"
                ),
                self.bytes
            );
        }
        if self.unclassified_external_entries != 0 {
            return format!(
                concat!(
                    "{{\"bytes\":{},\"quality\":\"lower_bound\",",
                    "\"reason\":\"{} live embedder entries have neither retained nor external-memory classification\"}}"
                ),
                self.bytes, self.unclassified_external_entries
            );
        }
        format!("{{\"bytes\":{},\"quality\":\"exact\"}}", self.bytes)
    }

    fn coverage_complete(self) -> bool {
        !self.identity_conflict && self.unclassified_external_entries == 0
    }
}

impl HostCategory {
    fn json(self) -> String {
        if self.unavailable_entries != 0 {
            return format!(
                concat!(
                    "{{\"bytes\":null,\"quality\":\"unavailable\",",
                    "\"reason\":\"{} reachable host owners lack complete retained-memory traversal\"}}"
                ),
                self.unavailable_entries
            );
        }
        if self.identity_conflict {
            return format!(
                concat!(
                    "{{\"bytes\":{},\"quality\":\"lower_bound\",",
                    "\"reason\":\"conflicting byte counts were reported for one host allocation identity\"}}"
                ),
                self.reported_bytes
            );
        }
        if self.opaque_storage {
            return format!(
                concat!(
                    "{{\"bytes\":{},\"quality\":\"lower_bound\",",
                    "\"reason\":\"opaque standard-library map or channel storage\"}}"
                ),
                self.reported_bytes
            );
        }
        format!(
            "{{\"bytes\":{},\"quality\":\"exact\"}}",
            self.reported_bytes
        )
    }
}

impl From<crate::host::HostRetainedMemory> for HostCategory {
    fn from(memory: crate::host::HostRetainedMemory) -> Self {
        Self {
            reported_bytes: memory.reported_bytes,
            unavailable_entries: memory.unavailable_entries,
            identity_conflict: memory.identity_conflict,
            opaque_storage: memory.opaque_storage,
        }
    }
}

impl Category {
    fn exact(bytes: usize) -> Self {
        Self {
            bytes,
            quality: Quality::Exact,
            reason: None,
            coverage_complete: true,
        }
    }

    fn lower_bound(bytes: usize, reason: &'static str) -> Self {
        Self {
            bytes,
            quality: Quality::LowerBound,
            reason: Some(reason),
            coverage_complete: true,
        }
    }

    fn incomplete_lower_bound(bytes: usize, reason: &'static str) -> Self {
        Self {
            bytes,
            quality: Quality::LowerBound,
            reason: Some(reason),
            coverage_complete: false,
        }
    }

    fn add(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_add(bytes);
    }

    fn make_lower_bound(&mut self, reason: &'static str) {
        self.quality = Quality::LowerBound;
        self.reason = Some(reason);
    }

    fn json(self) -> String {
        match self.reason {
            Some(reason) => format!(
                "{{\"bytes\":{},\"quality\":\"{}\",\"reason\":\"{}\"}}",
                self.bytes,
                self.quality.json(),
                reason
            ),
            None => format!(
                "{{\"bytes\":{},\"quality\":\"{}\"}}",
                self.bytes,
                self.quality.json()
            ),
        }
    }

    fn is_exact(self) -> bool {
        matches!(self.quality, Quality::Exact)
    }
}

#[derive(Clone)]
pub(crate) struct Snapshot {
    object_bodies: Category,
    property_storage: Category,
    scope_bodies: Category,
    scope_storage: Category,
    strings_symbols_bigints: Category,
    callable_metadata: Category,
    function_bytecode_metadata: Category,
    jit_heap_metadata: Category,
    regexp_metadata: Category,
    engine_caches: Category,
    interpreter_side_tables: Category,
    array_buffer_backing: Category,
    shared_array_buffer_backing: SharedBackingCategory,
    wasm_backing: WasmBackingCategory,
    host_resources: HostCategory,
}

impl Snapshot {
    fn managed_requested_bytes(&self) -> usize {
        self.requested_categories()
            .into_iter()
            .map(|category| category.bytes)
            .fold(0usize, usize::saturating_add)
    }

    fn requested_categories(&self) -> [Category; 11] {
        [
            self.object_bodies,
            self.property_storage,
            self.scope_bodies,
            self.scope_storage,
            self.strings_symbols_bigints,
            self.callable_metadata,
            self.function_bytecode_metadata,
            self.jit_heap_metadata,
            self.regexp_metadata,
            self.engine_caches,
            self.interpreter_side_tables,
        ]
    }

    fn managed_requested_category(&self) -> Category {
        let bytes = self.managed_requested_bytes();
        if self
            .requested_categories()
            .into_iter()
            .all(Category::is_exact)
        {
            Category::exact(bytes)
        } else if self
            .requested_categories()
            .into_iter()
            .all(|category| category.coverage_complete)
        {
            Category::lower_bound(
                bytes,
                "one or more component categories exclude opaque storage",
            )
        } else {
            Category::incomplete_lower_bound(
                bytes,
                "one or more requested-payload allocation families lack retained-memory traversal",
            )
        }
    }

    fn managed_external_category(&self) -> Category {
        let bytes = self
            .array_buffer_backing
            .bytes
            .saturating_add(self.shared_array_buffer_backing.bytes())
            .saturating_add(self.wasm_backing.bytes);
        if self.shared_array_buffer_backing.unavailable_ids != 0 {
            return Category::incomplete_lower_bound(
                bytes,
                "referenced Shared Data Block identities unavailable from the backing registry",
            );
        }
        if !self.wasm_backing.coverage_complete() {
            return Category::incomplete_lower_bound(
                bytes,
                "Wasm backing identities or embedder external-memory classification are incomplete",
            );
        }
        Category::exact(bytes)
    }

    fn complete(&self) -> bool {
        [
            self.object_bodies,
            self.property_storage,
            self.scope_bodies,
            self.scope_storage,
            self.strings_symbols_bigints,
            self.callable_metadata,
            self.function_bytecode_metadata,
            self.jit_heap_metadata,
            self.regexp_metadata,
            self.engine_caches,
            self.interpreter_side_tables,
            self.array_buffer_backing,
        ]
        .into_iter()
        .all(|category| {
            category.coverage_complete
                && match category.quality {
                    Quality::Exact => category.reason.is_none(),
                    Quality::LowerBound => category.reason.is_some_and(|reason| !reason.is_empty()),
                }
        }) && self.shared_array_buffer_backing.unavailable_ids == 0
            && self.wasm_backing.coverage_complete()
            && self.host_resources.unavailable_entries == 0
            && !self.host_resources.identity_conflict
    }

    fn json(&self, agent_id: u64, heap_id: u64) -> String {
        let managed_requested = self.managed_requested_category();
        let managed_external = self.managed_external_category();
        format!(
            concat!(
                "{{\"schema_version\":1,\"agent_id\":{},\"heap_id\":{},",
                "\"safepoint\":\"post_gc\",\"complete\":{},",
                "\"managed_requested_bytes\":{},",
                "\"managed_external_bytes\":{},",
                "\"categories\":{{",
                "\"object_bodies\":{},\"property_storage\":{},",
                "\"scope_bodies\":{},\"scope_storage\":{},",
                "\"strings_symbols_bigints\":{},\"callable_metadata\":{},",
                "\"function_bytecode_metadata\":{},\"jit_heap_metadata\":{},",
                "\"regexp_metadata\":{},\"engine_caches\":{},",
                "\"interpreter_side_tables\":{},\"array_buffer_backing\":{},",
                "\"shared_array_buffer_backing\":{},",
                "\"wasm_backing\":{},",
                "\"host_resources\":{}",
                "}}}}"
            ),
            agent_id,
            heap_id,
            self.complete(),
            managed_requested.json(),
            managed_external.json(),
            self.object_bodies.json(),
            self.property_storage.json(),
            self.scope_bodies.json(),
            self.scope_storage.json(),
            self.strings_symbols_bigints.json(),
            self.callable_metadata.json(),
            self.function_bytecode_metadata.json(),
            self.jit_heap_metadata.json(),
            self.regexp_metadata.json(),
            self.engine_caches.json(),
            self.interpreter_side_tables.json(),
            self.array_buffer_backing.json(),
            self.shared_array_buffer_backing.json(),
            self.wasm_backing.json(),
            self.host_resources.json(),
        )
    }
}

#[derive(Default)]
pub(crate) struct Visitor {
    /// Cache admission accounts one plan, never traverses independent source/callee graphs.
    fragment_plan_root: Option<usize>,
    /// One-time immutable source charge: skip mutable Function code/hoist/object-map sidecars.
    fragment_source_only: bool,
    loop_site_tokens: HashSet<usize>,
    gc_heaps: HashSet<usize>,
    lstrs: HashSet<usize>,
    rc_strs: HashSet<usize>,
    property_layouts: HashSet<usize>,
    symbols: HashSet<usize>,
    bigints: HashSet<usize>,
    callable_allocations: HashSet<usize>,
    native_closure_allocations: HashSet<usize>,
    native_closure_reporters: HashSet<usize>,
    reported_native_closures: HashSet<usize>,
    unreported_native_closures: HashSet<usize>,
    native_managed_allocations: HashMap<(&'static str, usize), usize>,
    native_managed_identity_conflict: bool,
    host_managed_allocations: HashMap<(&'static str, usize), usize>,
    functions: HashSet<usize>,
    classes: HashSet<usize>,
    chunks: HashSet<usize>,
    inline_site_allocations: HashSet<usize>,
    jit_codes: HashSet<usize>,
    hoist_plans: HashSet<usize>,
    binding_layouts: HashSet<usize>,
    stmt_bodies: HashSet<usize>,
    global_var_name_sets: HashSet<usize>,
    re_texts: HashSet<usize>,
    regexes: HashSet<usize>,
    regexp_programs: HashSet<usize>,
    regexp_char_classes: HashSet<usize>,
    regexp_string_sets: HashSet<usize>,
    regexp_backref_groups: HashSet<usize>,
    rc_u16_slices: HashSet<usize>,
    string_utf8_indexes: HashSet<usize>,
    rc_value_slices: HashSet<usize>,
    array_buffers: HashMap<usize, usize>,
    native_buffers: HashSet<usize>,
    shared_array_buffers: HashSet<u64>,
    shared_array_buffer_allocations: Vec<SharedBackingAllocation>,
    unavailable_shared_array_buffers: usize,
    external_allocations:
        HashMap<crate::host::RetainedExternalIdentity, crate::host::RetainedExternalAllocation>,
    external_identity_conflict: bool,
    strings_symbols_bigints: usize,
    callable_metadata: usize,
    function_bytecode_metadata: usize,
    function_bytecode_opaque_storage: bool,
    jit_heap_metadata: usize,
    regexp_metadata: usize,
    detached_property_storage: usize,
    detached_property_storage_opaque: bool,
}

/// Known requested payload directly reachable from this plan's own metadata. Independent
/// Rc<Function>/Class/callee-Chunk graphs are deliberately not traversed. This is an eviction
/// charge, not an exact total-retained-heap claim (Rc headers/opaque table buckets excluded).
pub(crate) fn fragment_plan_charge(chunk: &Rc<crate::bytecode::Chunk>) -> usize {
    let mut visitor = Visitor {
        fragment_plan_root: Some(Rc::as_ptr(chunk) as usize),
        ..Visitor::default()
    };
    visitor.chunk(chunk);
    visitor.fragment_charge()
}

pub(crate) fn fragment_plan_source_charge(chunk: &Rc<crate::bytecode::Chunk>) -> usize {
    let direct = fragment_plan_charge(chunk);
    let mut visitor = Visitor {
        fragment_source_only: true,
        ..Visitor::default()
    };
    visitor.chunk(chunk);
    visitor.fragment_charge().saturating_sub(direct)
}

impl Visitor {
    fn fragment_charge(&self) -> usize {
        self.function_bytecode_metadata
            .saturating_add(self.strings_symbols_bigints)
            .saturating_add(self.jit_heap_metadata)
            .saturating_add(self.regexp_metadata)
            .saturating_add(self.detached_property_storage)
    }
    pub(crate) fn loop_site_token(&mut self, token: &Rc<crate::ast::LoopSiteToken>) {
        if self.loop_site_tokens.insert(Rc::as_ptr(token) as usize) {
            self.add_function_bytecode_bytes(size_of::<crate::ast::LoopSiteToken>());
        }
    }

    fn gc_heap(&mut self, heap: &crate::value::GcHeap) -> (usize, bool) {
        let identity = Rc::as_ptr(heap) as usize;
        if !self.gc_heaps.insert(identity) {
            return (0, true);
        }
        crate::value::scan_gc_heap_retained_memory(heap, self)
    }

    pub(crate) fn lstr(&mut self, value: &crate::lstr::LStr) {
        let identity = value.as_ptr() as usize;
        if self.lstrs.insert(identity) {
            self.strings_symbols_bigints = self
                .strings_symbols_bigints
                .saturating_add(value.allocation_requested_bytes());
            if let Some(owner) = value.view_owner() {
                self.lstr(owner);
            }
        }
    }

    pub(crate) fn bigint(&mut self, value: &crate::bigint::JsBigInt) {
        if self.bigints.insert(value.allocation_identity()) {
            self.strings_symbols_bigints = self
                .strings_symbols_bigints
                .saturating_add(value.retained_requested_bytes());
        }
    }

    /// A BigInt literal's digits, held by source code.
    pub(crate) fn bigint_literal(&mut self, value: &crate::bigint::BigIntLiteral) {
        if self.bigints.insert(value.allocation_identity()) {
            self.strings_symbols_bigints = self
                .strings_symbols_bigints
                .saturating_add(value.retained_requested_bytes());
        }
    }

    /// A shared source record: its own bytes and the text it keeps alive, once per record.
    pub(crate) fn script_source(&mut self, source: &Rc<crate::stack_trace::ScriptSource>) {
        let identity = Rc::as_ptr(source) as usize;
        if self.rc_strs.insert(identity) {
            self.strings_symbols_bigints = self
                .strings_symbols_bigints
                .saturating_add(source.owned_bytes());
            self.rc_str(source.text());
        }
    }

    pub(crate) fn rc_str(&mut self, value: &Rc<str>) {
        let identity = Rc::as_ptr(value) as *const () as usize;
        if self.rc_strs.insert(identity) {
            // `RcBox` counters are a private standard-library layout and are excluded.
            self.strings_symbols_bigints = self.strings_symbols_bigints.saturating_add(value.len());
        }
    }

    pub(crate) fn symbol(&mut self, value: &crate::value::SymbolRef) {
        let identity = crate::value::symbol_identity(value);
        if self.symbols.insert(identity) {
            self.strings_symbols_bigints = self
                .strings_symbols_bigints
                .saturating_add(size_of::<SymbolData>());
            if let Some(description) = &value.description {
                self.rc_str(description);
            }
            if let Some(observers) = value.weak_observers.borrow().as_ref() {
                self.strings_symbols_bigints = self
                    .strings_symbols_bigints
                    .saturating_add(size_of::<crate::weak_metadata::DeathObservers>())
                    .saturating_add(observers.allocated_bytes());
            }
        }
    }

    pub(crate) fn value(&mut self, value: &Value) {
        match value {
            Value::BigInt(value) => {
                self.bigint(value);
            }
            Value::Str(value) => {
                self.lstr(value);
            }
            Value::Sym(value) => self.symbol(value),
            Value::Undefined
            | Value::Empty
            | Value::Null
            | Value::Bool(_)
            | Value::Num(_)
            | Value::Obj(_) => {}
        }
    }

    fn callable(&mut self, callable: &Callable) {
        match callable {
            Callable::None | Callable::Native(_) => {}
            Callable::NativeData(value) => {
                let identity = Rc::as_ptr(value) as usize;
                let first = self.callable_allocations.insert(identity);
                if first {
                    self.callable_metadata = self
                        .callable_metadata
                        .saturating_add(size_of_val(value.as_ref()));
                }
                // Keep the immutable registration label in the canonical string family.
                self.rc_str(&value.identity);
                let crate::value::NativeCallableBody::Opaque(function) = &value.body else {
                    if let crate::value::NativeCallableBody::Captured { captures, .. } = &value.body
                    {
                        if first {
                            self.callable_metadata = self
                                .callable_metadata
                                .saturating_add(size_of_val(captures.as_ref()));
                        }
                        for capture in captures {
                            self.value(capture);
                        }
                    }
                    return;
                };
                let closure_identity = Rc::as_ptr(function) as *const () as usize;
                if self.native_closure_allocations.insert(closure_identity) {
                    self.callable_metadata = self
                        .callable_metadata
                        .saturating_add(size_of_val(function.as_ref()));
                }
                if let Some(reporter) = &value.retained {
                    self.reported_native_closures.insert(closure_identity);
                    let reporter_identity = Rc::as_ptr(reporter) as *const () as usize;
                    if self.native_closure_reporters.insert(reporter_identity) {
                        self.callable_metadata = self
                            .callable_metadata
                            .saturating_add(size_of_val(reporter.as_ref()));
                        reporter.scan_retained_memory(self);
                    }
                } else {
                    self.unreported_native_closures.insert(closure_identity);
                }
            }
            Callable::User(value) => {
                let identity = crate::value::user_fn_ptr(value) as usize;
                if self.callable_allocations.insert(identity) {
                    self.callable_metadata = self
                        .callable_metadata
                        .saturating_add(size_of_val(value.as_ref()));
                    if let Some(cache) = value.binding_getter_cache.take() {
                        self.callable_metadata = self
                            .callable_metadata
                            .saturating_add(cache.retained_requested_storage_bytes());
                        self.rc_str(&cache.name);
                        value.binding_getter_cache.set(Some(cache));
                    }
                }
                self.function(&value.func);
            }
            Callable::Bound(value) => {
                // Box-backed variants have one allocation per containing Object, and the object
                // snapshot is identity-unique, so they need no shared-allocation identity set.
                self.callable_metadata = self
                    .callable_metadata
                    .saturating_add(size_of_val(value.as_ref()))
                    .saturating_add(value.args.capacity().saturating_mul(size_of::<Value>()));
                self.value(&value.this);
                for argument in &value.args {
                    self.value(argument);
                }
            }
            Callable::WrappedShadow(value) => {
                let identity = Rc::as_ptr(value) as usize;
                if self.callable_allocations.insert(identity) {
                    self.callable_metadata = self
                        .callable_metadata
                        .saturating_add(size_of_val(value.as_ref()))
                        .saturating_add(size_of::<Value>());
                    self.value(&value.target);
                }
            }
            Callable::WrappedCross(value) => {
                // See Bound above: this Box cannot be shared between object holders.
                self.callable_metadata = self
                    .callable_metadata
                    .saturating_add(size_of_val(value.as_ref()));
                self.value(&value.target);
            }
            Callable::AccessorGet(name)
            | Callable::AccessorSet(name)
            | Callable::PropGet(name)
            | Callable::PropSet(name)
            | Callable::PrivatePropGet(name)
            | Callable::PrivatePropSet(name) => {
                let outer = Rc::as_ptr(name) as usize;
                if self.callable_allocations.insert(outer) {
                    self.callable_metadata =
                        self.callable_metadata.saturating_add(size_of::<Rc<str>>());
                }
                self.rc_str(name.as_ref());
            }
        }
    }

    pub(crate) fn add_function_bytecode_bytes(&mut self, bytes: usize) {
        self.function_bytecode_metadata = self.function_bytecode_metadata.saturating_add(bytes);
    }

    pub(crate) fn inline_sites_bytes(&mut self, identity: usize, bytes: usize) -> usize {
        if self.inline_site_allocations.insert(identity) {
            bytes
        } else {
            0
        }
    }

    pub(crate) fn mark_function_bytecode_opaque_storage(&mut self) {
        self.function_bytecode_opaque_storage = true;
    }

    pub(crate) fn add_jit_heap_metadata_bytes(&mut self, bytes: usize) {
        self.jit_heap_metadata = self.jit_heap_metadata.saturating_add(bytes);
    }

    pub(crate) fn str_units(&mut self, units: &crate::interpreter::StrUnits) {
        if let crate::interpreter::StrUnits::Units(units) = units {
            let identity = Rc::as_ptr(units) as *const () as usize;
            if self.rc_u16_slices.insert(identity) {
                self.strings_symbols_bigints = self
                    .strings_symbols_bigints
                    .saturating_add(units.len().saturating_mul(std::mem::size_of::<u16>()));
            }
        }
    }

    fn value_slice(&mut self, values: &Rc<[Value]>) -> usize {
        let identity = Rc::as_ptr(values) as *const () as usize;
        if !self.rc_value_slices.insert(identity) {
            return 0;
        }
        for value in values.iter() {
            self.value(value);
        }
        values.len().saturating_mul(size_of::<Value>())
    }

    pub(crate) fn re_text(&mut self, text: &Rc<crate::regex::ReText>) {
        let identity = Rc::as_ptr(text) as usize;
        if self.re_texts.insert(identity) {
            let bytes = text.scan_retained_memory(self);
            self.regexp_metadata = self.regexp_metadata.saturating_add(bytes);
        }
    }

    pub(crate) fn regex(&mut self, regex: &Rc<crate::regex::Regex>) {
        let identity = Rc::as_ptr(regex) as usize;
        if self.regexes.insert(identity) {
            self.regexp_metadata = self
                .regexp_metadata
                .saturating_add(regex.scan_retained_memory(self));
        }
    }

    pub(crate) fn regexp_program_allocation(&mut self, identity: usize) -> bool {
        self.regexp_programs.insert(identity)
    }

    pub(crate) fn regexp_char_class_allocation(&mut self, identity: usize) -> bool {
        self.regexp_char_classes.insert(identity)
    }

    pub(crate) fn regexp_string_set_allocation(&mut self, identity: usize) -> bool {
        self.regexp_string_sets.insert(identity)
    }

    pub(crate) fn regexp_backref_groups_allocation(&mut self, identity: usize) -> bool {
        self.regexp_backref_groups.insert(identity)
    }

    fn array_buffer(&mut self, buffer: &crate::interpreter::ArrayBufferBytes) {
        let identity = Rc::as_ptr(buffer) as usize;
        self.array_buffers
            .entry(identity)
            .or_insert_with(|| buffer.borrow().capacity());
    }

    fn array_buffer_bytes(&self) -> usize {
        self.array_buffers
            .iter()
            .filter(|(identity, _)| {
                !self.external_allocations.contains_key(
                    &crate::host::RetainedExternalIdentity::ArrayBufferBytes(**identity),
                )
            })
            .map(|(_, bytes)| *bytes)
            .fold(0usize, usize::saturating_add)
    }

    fn external_allocation(&mut self, allocation: crate::host::RetainedExternalAllocation) {
        match self.external_allocations.entry(allocation.identity) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(allocation);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                if entry.get().bytes != allocation.bytes || entry.get().kind != allocation.kind {
                    self.external_identity_conflict = true;
                    if allocation.bytes > entry.get().bytes {
                        entry.insert(allocation);
                    }
                }
            }
        }
    }

    fn shared_array_buffer(&mut self, id: u64) {
        if !self.shared_array_buffers.insert(id) {
            return;
        }
        let Some(backing) = crate::interpreter::shared_mem_get(id) else {
            self.unavailable_shared_array_buffers =
                self.unavailable_shared_array_buffers.saturating_add(1);
            return;
        };
        let bytes = backing.lock().unwrap().capacity();
        self.shared_array_buffer_allocations
            .push(SharedBackingAllocation { id, bytes });
    }

    pub(crate) fn function(&mut self, function: &Rc<crate::ast::Function>) {
        if self.fragment_plan_root.is_some() {
            return;
        }
        let identity = Rc::as_ptr(function) as usize;
        if !self.functions.insert(identity) {
            return;
        }
        let mut bytes = crate::ast::scan_function_retained_memory(function, self);
        if self.fragment_source_only {
            self.add_function_bytecode_bytes(bytes);
            return;
        }
        if let Some((_, hoist)) = function.hoist.get() {
            let identity = Rc::as_ptr(hoist) as usize;
            if self.hoist_plans.insert(identity) {
                bytes = bytes
                    .saturating_add(size_of::<Vec<crate::ast::HoistOp>>())
                    .saturating_add(
                        hoist
                            .capacity()
                            .saturating_mul(size_of::<crate::ast::HoistOp>()),
                    );
                for op in hoist.iter() {
                    match op {
                        crate::ast::HoistOp::Var(name) => {
                            bytes = bytes.saturating_add(name.capacity());
                        }
                        crate::ast::HoistOp::Fn(name, nested)
                        | crate::ast::HoistOp::AnnexB(name, nested) => {
                            bytes = bytes.saturating_add(name.capacity());
                            self.function(nested);
                        }
                    }
                }
            }
        }
        if let Some(chunk) = function.code.get().and_then(Option::as_ref) {
            self.chunk(chunk);
        }
        if let Some(chunk) = function.code2.get().and_then(Option::as_ref) {
            self.chunk(chunk);
        }
        if let Some(maps) = function.fn_maps.get() {
            bytes = bytes.saturating_add(size_of::<crate::ast::FunctionMaps>());
            bytes = bytes.saturating_add(maps.roots.heap_bytes());
            self.props(&maps.function);
            if let Some(prototype_map) = &maps.prototype {
                self.props(prototype_map);
            }
            if let Some((name, named)) = &*maps.named.borrow() {
                bytes = bytes.saturating_add(name.len());
                self.props(named);
            }
        }
        if let Some(layouts) = function.env_layouts.get() {
            bytes = bytes.saturating_add(size_of::<
                [std::cell::OnceCell<Rc<crate::interpreter::BindingLayout>>; 2],
            >());
            for layout in layouts.iter().filter_map(std::cell::OnceCell::get) {
                self.binding_layout(layout);
            }
        }
        self.add_function_bytecode_bytes(bytes);
    }

    pub(crate) fn binding_layout(&mut self, layout: &Rc<crate::interpreter::BindingLayout>) {
        if self.binding_layouts.insert(Rc::as_ptr(layout) as usize) {
            let bytes = layout.retained_memory(self);
            self.add_function_bytecode_bytes(bytes);
            self.function_bytecode_opaque_storage = true;
        }
    }

    pub(crate) fn class(&mut self, class: &Rc<crate::ast::Class>) {
        if self.fragment_plan_root.is_some() {
            return;
        }
        let identity = Rc::as_ptr(class) as usize;
        if self.classes.insert(identity) {
            let bytes = crate::ast::scan_class_retained_memory(class, self);
            self.add_function_bytecode_bytes(bytes);
        }
    }

    pub(crate) fn stmt_body(&mut self, body: &Rc<Vec<crate::ast::Stmt>>) {
        if self.fragment_plan_root.is_some() {
            return;
        }
        let identity = Rc::as_ptr(body) as usize;
        if self.stmt_bodies.insert(identity) {
            let bytes = crate::ast::scan_stmt_body_retained_memory(body, self);
            self.add_function_bytecode_bytes(bytes);
        }
    }

    pub(crate) fn global_var_names(
        &mut self,
        names: &Rc<RefCell<std::collections::HashSet<String>>>,
    ) -> (usize, bool) {
        let identity = Rc::as_ptr(names) as usize;
        if !self.global_var_name_sets.insert(identity) {
            return (0, true);
        }
        let names = names.borrow();
        let bytes = size_of::<RefCell<std::collections::HashSet<String>>>()
            .saturating_add(names.len().saturating_mul(size_of::<String>()))
            .saturating_add(
                names
                    .iter()
                    .map(String::capacity)
                    .fold(0usize, usize::saturating_add),
            );
        (bytes, names.is_empty())
    }

    pub(crate) fn chunk(&mut self, chunk: &Rc<crate::bytecode::Chunk>) {
        let identity = Rc::as_ptr(chunk) as usize;
        if self.fragment_plan_root.is_some_and(|root| root != identity) {
            return;
        }
        if self.chunks.insert(identity) {
            chunk.scan_retained_memory(self);
        }
    }

    pub(crate) fn jit_code(&mut self, code: &Rc<crate::jit::JitCode>) {
        let identity = Rc::as_ptr(code) as usize;
        if self.jit_codes.insert(identity) {
            self.add_jit_heap_metadata_bytes(code.retained_heap_metadata_bytes());
        }
    }

    pub(crate) fn property_layout(&mut self, layout: &crate::value::PropertyLayout) {
        if self.property_layouts.insert(Rc::as_ptr(layout) as usize) {
            // Requested payload, like the other Rc-backed categories: one layout body and key
            // buffer per shared layout, not private allocator/refcount headers or one copy per
            // instance. The optional standard HashMap has a known entry-capacity lower bound;
            // Rust does not expose its exact bucket allocation.
            self.detached_property_storage = self.detached_property_storage.saturating_add(
                size_of::<crate::value::PropertyLayoutData>()
                    + layout.capacity() * size_of::<Rc<str>>(),
            );
            if let Some(capacity) = layout.index_capacity() {
                self.detached_property_storage = self
                    .detached_property_storage
                    .saturating_add(size_of::<crate::fasthash::FastMap<Rc<str>, usize>>())
                    .saturating_add(capacity * size_of::<(Rc<str>, usize)>());
                self.detached_property_storage_opaque = true;
            }
            for key in layout.iter() {
                self.rc_str(key);
            }
        }
    }

    pub(crate) fn props(&mut self, props: &crate::value::Props) {
        if let Some(layout) = props.shared_layout() {
            self.property_layout(layout);
        }
        let (bytes, exact) = props.retained_requested_storage_bytes();
        self.detached_property_storage = self.detached_property_storage.saturating_add(bytes);
        self.detached_property_storage_opaque |= !exact;
        for (_, property) in props.iter() {
            property.visit_retained_value(self);
            if let Some(getter) = property.getter() {
                self.value(getter);
            }
            if let Some(setter) = property.setter() {
                self.value(setter);
            }
        }
        for property in props.packed_values() {
            property.visit_retained_value(self);
        }
    }

    fn object(&mut self, object: &Object) -> (usize, bool) {
        if let Some(layout) = object.props.shared_layout() {
            self.property_layout(layout);
        }
        for (key, _) in object.internal_slots().iter() {
            self.rc_str(key);
        }
        for (_, property) in object.props.iter().chain(object.internal_slots().iter()) {
            property.visit_retained_value(self);
            if let Some(getter) = property.getter() {
                self.value(getter);
            }
            if let Some(setter) = property.setter() {
                self.value(setter);
            }
        }
        // Packed array elements have no key entry and must be visited separately.
        for property in object.props.packed_values() {
            property.visit_retained_value(self);
            if let Some(getter) = property.getter() {
                self.value(getter);
            }
            if let Some(setter) = property.setter() {
                self.value(setter);
            }
        }
        self.callable(&object.call);
        let mut internal_bytes = object.internal_slots().retained_storage_bytes();
        if let Some(view) = &object.native_typed_array {
            internal_bytes += size_of::<crate::native_typed_array::NativeTypedArray>();
            internal_bytes += self.native_buffer(&view.buffer);
        }
        match &object.exotic {
            Exotic::StrWrap(value) => self.value(&Value::Str((**value).clone())),
            Exotic::SymWrap(value) => self.symbol(value),
            Exotic::BigIntWrap(value) => self.value(&Value::BigInt((**value).clone())),
            Exotic::Error(trace) => {
                internal_bytes += trace
                    .as_deref()
                    .map_or(0, crate::stack_trace::StackTrace::retained_bytes);
            }
            Exotic::ArrayIterator(state) => {
                internal_bytes += std::mem::size_of_val(&**state);
                self.value(&state.target);
            }
            Exotic::StringIterator(state) => {
                internal_bytes += std::mem::size_of_val(&**state);
                if let Some(string) = &state.string {
                    self.value(&Value::Str(string.clone()));
                }
            }
            Exotic::Collection(entries) => {
                internal_bytes += std::mem::size_of_val(&**entries)
                    .saturating_add(entries.storage_bytes_lower_bound());
                for (key, value) in entries.iter() {
                    self.value(key);
                    self.value(value);
                }
            }
            Exotic::None
            | Exotic::Array
            | Exotic::BoolWrap(_)
            | Exotic::NumWrap(_)
            | Exotic::Arguments => {}
        }
        let (bytes, exact) = object.props.retained_requested_storage_bytes();
        (bytes.saturating_add(internal_bytes), exact)
    }

    fn native_buffer(
        &mut self,
        buffer: &std::rc::Rc<crate::native_typed_array::NativeBuffer>,
    ) -> usize {
        if !self
            .native_buffers
            .insert(std::rc::Rc::as_ptr(buffer) as usize)
        {
            return 0;
        }
        if let Some(storage) = buffer.storage.borrow().as_ref() {
            self.array_buffer(storage);
        }
        size_of::<crate::native_typed_array::NativeBuffer>() + 2 * size_of::<usize>()
    }

    fn scope(&mut self, scope: &Scope) -> (usize, bool) {
        if let Some(layout) = scope.vars.shared_layout() {
            self.binding_layout(layout);
        }
        let (mut bytes, exact) = scope.vars.retained_requested_storage_bytes();
        bytes = bytes.saturating_add(scope.lexical_names.retained_requested_storage_bytes());
        for (name, binding) in scope.vars.iter() {
            self.rc_str(name);
            self.value(&binding.value);
            if let Some((_, import_name)) = binding.import_ref.as_deref() {
                bytes =
                    bytes.saturating_add(size_of::<(crate::interpreter::Env, std::rc::Rc<str>)>());
                self.rc_str(import_name);
            }
        }
        (bytes, exact)
    }
}

impl crate::value::NativeRetainedMemoryVisitor for Visitor {
    fn allocation(&mut self, allocation: crate::value::RetainedManagedAllocation) {
        let identity = (allocation.identity_domain, allocation.identity);
        match self.native_managed_allocations.entry(identity) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(allocation.requested_bytes);
                self.callable_metadata = self
                    .callable_metadata
                    .saturating_add(allocation.requested_bytes);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                let previous = *entry.get();
                if previous != allocation.requested_bytes {
                    self.native_managed_identity_conflict = true;
                    if allocation.requested_bytes > previous {
                        self.callable_metadata = self
                            .callable_metadata
                            .saturating_add(allocation.requested_bytes - previous);
                        *entry.into_mut() = allocation.requested_bytes;
                    }
                }
            }
        }
    }

    fn value(&mut self, value: &Value) {
        Visitor::value(self, value);
    }
}

struct HostMemoryAdapter<'a> {
    visitor: &'a mut Visitor,
    memory: &'a mut crate::host::HostRetainedMemory,
}

impl crate::host::HostRetainedMemoryVisitor for HostMemoryAdapter<'_> {
    fn allocation(&mut self, allocation: crate::value::RetainedManagedAllocation) {
        let identity = (allocation.identity_domain, allocation.identity);
        match self.visitor.host_managed_allocations.entry(identity) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(allocation.requested_bytes);
                self.memory.reported_bytes = self
                    .memory
                    .reported_bytes
                    .saturating_add(allocation.requested_bytes);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                let previous = *entry.get();
                if previous != allocation.requested_bytes {
                    self.memory.identity_conflict = true;
                    if allocation.requested_bytes > previous {
                        self.memory.reported_bytes = self
                            .memory
                            .reported_bytes
                            .saturating_add(allocation.requested_bytes - previous);
                        *entry.into_mut() = allocation.requested_bytes;
                    }
                }
            }
        }
    }

    fn value(&mut self, value: &Value) {
        self.visitor.value(value);
    }

    fn opaque_storage(&mut self) {
        self.memory.opaque_storage = true;
    }

    fn unavailable(&mut self) {
        self.memory.unavailable_entries = self.memory.unavailable_entries.saturating_add(1);
    }
}

struct DirectTotals {
    object_count: usize,
    scope_count: usize,
    property_storage: Category,
    scope_storage: Category,
    engine_caches: Category,
    interpreter_side_tables: Category,
    host_resources: crate::host::HostRetainedMemory,
}

impl Default for DirectTotals {
    fn default() -> Self {
        Self {
            object_count: 0,
            scope_count: 0,
            property_storage: Category::exact(0),
            scope_storage: Category::exact(0),
            engine_caches: Category::exact(0),
            interpreter_side_tables: Category::exact(0),
            host_resources: crate::host::HostRetainedMemory::default(),
        }
    }
}

fn scan_realm(
    interp: &Interp,
    objects: &[Gc],
    scopes: &[Env],
    visitor: &mut Visitor,
    totals: &mut DirectTotals,
) {
    totals.object_count = totals.object_count.saturating_add(objects.len());
    totals.scope_count = totals.scope_count.saturating_add(scopes.len());
    totals.interpreter_side_tables.add(size_of::<Interp>());
    let mut host_memory = interp.host_state.retained_memory();
    for allocation in &host_memory.external_allocations {
        visitor.external_allocation(*allocation);
    }
    interp
        .host_state
        .scan_retained_memory(&mut HostMemoryAdapter {
            visitor,
            memory: &mut host_memory,
        });
    totals.host_resources.add(host_memory);
    let (gc_heap_bytes, gc_heap_exact) = visitor.gc_heap(&interp.gc_heap);
    totals.interpreter_side_tables.add(gc_heap_bytes);
    if !gc_heap_exact {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque object-shape HashMap bucket storage");
    }
    for object in objects {
        let (bytes, exact) = visitor.object(&object.borrow());
        totals.property_storage.add(bytes);
        if !exact {
            totals
                .property_storage
                .make_lower_bound("opaque standard-library HashMap bucket storage");
        }
    }
    for scope in scopes {
        let (bytes, exact) = visitor.scope(&scope.borrow());
        totals.scope_storage.add(bytes);
        if !exact {
            totals
                .scope_storage
                .make_lower_bound("opaque standard-library HashMap bucket storage");
        }
    }

    // Candidate backings are registered GC objects. The heap census above counts their
    // property storage and strings once, including backings pinned by active iterators after
    // cache eviction. Cache entries additionally pin key-only layouts; these can outlive their
    // originating foreign heap. Visit them through the same layout/string identity census as
    // ordinary properties and constructor hints. Stale recency keys own no allocations.
    let (bytes, exact) = interp.enumeration_keys.scan_retained_memory(|entry| {
        entry.scan_retained_layouts(visitor);
    });
    totals.engine_caches.add(bytes);
    if !exact {
        totals
            .engine_caches
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    let (bytes, exact) = interp
        .property_key_strings
        .scan_retained_memory(|(key, string)| {
            visitor.rc_str(key);
            visitor.lstr(string);
        });
    totals.engine_caches.add(bytes);
    if !exact {
        totals
            .engine_caches
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    let (bytes, exact) = interp
        .str_units
        .scan_retained_memory(|(string, units, index)| {
            visitor.lstr(string);
            visitor.str_units(units);
            if let Some(index) = index {
                if visitor
                    .string_utf8_indexes
                    .insert(Rc::as_ptr(index) as usize)
                {
                    visitor.strings_symbols_bigints = visitor
                        .strings_symbols_bigints
                        .saturating_add(index.heap_bytes());
                }
            }
        });
    totals.engine_caches.add(bytes);
    if !exact {
        totals
            .engine_caches
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    let (bytes, exact) = interp.re_texts.scan_retained_memory(|(string, text)| {
        visitor.lstr(string);
        visitor.re_text(text);
    });
    totals.engine_caches.add(bytes);
    if !exact {
        totals
            .engine_caches
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    if let Some((string, _, text)) = &interp.re_text_ascii_hot {
        visitor.lstr(string);
        visitor.re_text(text);
    }
    let (bytes, exact) = interp.regexp_programs.scan_retained_memory(visitor);
    totals.engine_caches.add(bytes);
    if !exact {
        totals
            .engine_caches
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    totals.engine_caches.add(
        interp
            .vm_pool
            .capacity()
            .saturating_mul(size_of::<crate::execution_storage::CompactFrame>())
            .saturating_add(
                interp
                    .stub_cache
                    .capacity()
                    .saturating_mul(size_of::<std::cell::Cell<crate::interpreter::StubEntry>>()),
            )
            .saturating_add(
                interp
                    .stub_cache_names
                    .borrow()
                    .capacity()
                    .saturating_mul(size_of::<Option<Rc<str>>>()),
            )
            .saturating_add(
                interp
                    .frame_pool
                    .capacity()
                    .saturating_mul(size_of::<std::ptr::NonNull<crate::jit::JitFrame>>()),
            )
            .saturating_add(
                interp
                    .frame_pool
                    .len()
                    .saturating_mul(size_of::<crate::jit::JitFrame>()),
            )
            .saturating_add(
                interp
                    .creation_pins
                    .len()
                    .saturating_mul(size_of::<(usize, crate::value::WeakGc)>()),
            )
            .saturating_add(
                interp
                    .global_env_pins
                    .capacity()
                    .saturating_mul(size_of::<crate::interpreter::WeakEnv>()),
            ),
    );
    totals.engine_caches.add(interp.frame_pool.registry_bytes());
    for (slots, stack) in &interp.vm_pool {
        totals.engine_caches.add(
            slots
                .capacity()
                .saturating_add(stack.capacity())
                .saturating_mul(size_of::<crate::value::PackedValue>()),
        );
        debug_assert!(slots.is_empty() && stack.is_empty());
    }
    totals.engine_caches.add(
        interp
            .native_arg_pool
            .capacity()
            .saturating_mul(size_of::<Vec<Value>>()),
    );
    for buf in &interp.native_arg_pool {
        totals
            .engine_caches
            .add(buf.capacity().saturating_mul(size_of::<Value>()));
        debug_assert!(buf.is_empty());
    }
    for name in interp.stub_cache_names.borrow().iter().flatten() {
        visitor.rc_str(name);
    }
    if let Some((keys, ..)) = &interp.arguments_shapes {
        for key in keys {
            visitor.rc_str(key);
        }
    }
    for map in interp
        .descriptor_maps
        .iter()
        .chain(std::iter::once(&interp.iter_result_map))
    {
        if let Some(map) = map.get() {
            visitor.props(map);
        }
    }
    if !interp.creation_pins.is_empty() {
        totals
            .engine_caches
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .regexps
            .len()
            .saturating_mul(size_of::<(usize, Rc<crate::regex::Regex>)>()),
    );
    if !interp.regexps.is_empty() {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    for regex in interp.regexps.values() {
        visitor.regex(regex);
    }
    if let Some(last) = &interp.regexp_last {
        totals.interpreter_side_tables.add(
            last.caps
                .capacity()
                .saturating_mul(size_of::<Option<(usize, usize)>>()),
        );
        visitor.lstr(&last.input);
        visitor.re_text(&last.text);
        if let Some((regex, _)) = &last.lazy_captures {
            visitor.regex(regex);
        }
    }
    if let Some(symbol) = &interp.iterator_sym {
        visitor.symbol(symbol);
    }
    totals
        .interpreter_side_tables
        .add(
            interp
                .wk_syms
                .capacity()
                .saturating_mul(size_of::<(&'static str, Value, Rc<str>)>()),
        );
    for (_, symbol, key) in &interp.wk_syms {
        visitor.value(symbol);
        visitor.rc_str(key);
    }
    totals.interpreter_side_tables.add(
        interp
            .error_protos
            .len()
            .saturating_mul(size_of::<(&'static str, Gc)>())
            .saturating_add(
                interp
                    .extra_protos
                    .len()
                    .saturating_mul(size_of::<(&'static str, Gc)>()),
            )
            .saturating_add(
                interp
                    .console
                    .capacity()
                    .saturating_mul(size_of::<String>()),
            )
            .saturating_add(
                interp
                    .eval_realm_fns
                    .len()
                    .saturating_mul(size_of::<usize>()),
            )
            .saturating_add(interp.import_base.capacity())
            .saturating_add(
                interp
                    .native_function_realms
                    .borrow()
                    .len()
                    .saturating_mul(size_of::<(usize, (crate::value::WeakGc, usize))>()),
            )
            .saturating_add(
                interp
                    .construct_capacity_hints
                    .len()
                    .saturating_mul(size_of::<(
                        usize,
                        (crate::value::WeakGc, crate::value::PropertyLayout),
                    )>()),
            )
            .saturating_add(
                interp
                    .htmldda
                    .len()
                    .saturating_mul(size_of::<(usize, crate::value::WeakGc)>()),
            ),
    );
    for line in &interp.console {
        totals.interpreter_side_tables.add(line.capacity());
    }
    for (_, layout) in interp.construct_capacity_hints.values() {
        visitor.property_layout(layout);
    }
    if let Some(meta) = &interp.import_meta {
        visitor.value(meta);
    }
    if !interp.error_protos.is_empty()
        || !interp.extra_protos.is_empty()
        || !interp.eval_realm_fns.is_empty()
        || !interp.native_function_realms.borrow().is_empty()
        || !interp.construct_capacity_hints.is_empty()
        || !interp.htmldda.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap/HashSet bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .modules
            .len()
            .saturating_mul(size_of::<(String, Value)>())
            .saturating_add(
                interp
                    .module_recs
                    .len()
                    .saturating_mul(size_of::<(String, crate::modules::ModuleRec)>()),
            )
            .saturating_add(
                interp
                    .pending_dynamic_imports
                    .len()
                    .saturating_mul(size_of::<(u64, crate::modules::PendingDynamicImport)>()),
            )
            .saturating_add(interp.module_ns.len().saturating_mul(size_of::<(
                usize,
                crate::fasthash::FastMap<String, crate::modules::NsBinding>,
            )>())),
    );
    for (key, value) in &interp.modules {
        totals.interpreter_side_tables.add(key.capacity());
        visitor.value(value);
    }
    for (key, record) in &interp.module_recs {
        totals.interpreter_side_tables.add(
            key.capacity()
                .saturating_add(record.scan_retained_memory(visitor)),
        );
    }
    for pending in interp.pending_dynamic_imports.values() {
        totals
            .interpreter_side_tables
            .add(pending.scan_retained_memory(visitor));
    }
    for bindings in interp.module_ns.values() {
        totals.interpreter_side_tables.add(
            bindings
                .len()
                .saturating_mul(size_of::<(String, crate::modules::NsBinding)>()),
        );
        for (name, binding) in bindings {
            totals.interpreter_side_tables.add(
                name.capacity()
                    .saturating_add(binding.scan_retained_memory(visitor)),
            );
        }
    }
    if !interp.modules.is_empty()
        || !interp.module_recs.is_empty()
        || !interp.pending_dynamic_imports.is_empty()
        || !interp.module_ns.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .promises
            .len()
            .saturating_mul(size_of::<(usize, crate::interpreter::PromiseState)>())
            .saturating_add(
                interp
                    .unhandled_rejections
                    .len()
                    .saturating_mul(size_of::<(usize, (u64, Value, Value, u64))>()),
            )
            .saturating_add(
                interp
                    .handled_rejections
                    .capacity()
                    .saturating_mul(size_of::<(Value, Value)>()),
            )
            .saturating_add(
                interp
                    .microtasks
                    .capacity()
                    .saturating_mul(size_of::<crate::interpreter::Job>()),
            )
            .saturating_add(interp.host_settings_states.len().saturating_mul(size_of::<(
                (usize, u64),
                crate::interpreter::HostSettingsState,
            )>()))
            .saturating_add(
                interp
                    .retired_host_job_contexts
                    .len()
                    .saturating_mul(size_of::<u64>()),
            )
            .saturating_add(
                interp
                    .kept_alive
                    .capacity()
                    .saturating_mul(size_of::<Value>()),
            )
            .saturating_add(
                interp
                    .promise_forward
                    .len()
                    .saturating_mul(size_of::<(usize, Value)>()),
            ),
    );
    for state in interp.promises.values() {
        visitor.value(&state.value);
        totals
            .interpreter_side_tables
            .add(state.reactions.capacity().saturating_mul(size_of::<(
                crate::interpreter::JobCallback,
                crate::interpreter::JobCallback,
                Value,
                u64,
            )>()));
        for (on_fulfilled, on_rejected, result, _) in &state.reactions {
            visitor.value(&on_fulfilled.callback);
            visitor.value(&on_rejected.callback);
            if let Some(global) = &on_fulfilled.script_caller {
                visitor.value(&Value::Obj(global.clone()));
            }
            if let Some(global) = &on_rejected.script_caller {
                visitor.value(&Value::Obj(global.clone()));
            }
            visitor.value(result);
        }
    }
    for (promise, reason) in interp
        .unhandled_rejections
        .values()
        .map(|(_, promise, reason, _)| (promise, reason))
        .chain(
            interp
                .handled_rejections
                .iter()
                .map(|(promise, reason, _)| (promise, reason)),
        )
    {
        visitor.value(promise);
        visitor.value(reason);
    }
    for job in &interp.microtasks {
        visitor.value(&job.handler.callback);
        if let Some(global) = &job.handler.script_caller {
            visitor.value(&Value::Obj(global.clone()));
        }
        visitor.value(&job.result);
        visitor.value(&job.value);
    }
    for state in interp.host_settings_states.values() {
        // Environments are already canonical to the collector's scope snapshot. The parallel
        // global-name set is not part of an Env and therefore needs its own identity registry.
        let (bytes, exact) = visitor.global_var_names(&state.global_var_names);
        totals.interpreter_side_tables.add(bytes);
        if !exact {
            totals
                .interpreter_side_tables
                .make_lower_bound("opaque standard-library HashMap/HashSet bucket storage");
        }
    }
    let (bytes, exact) = visitor.global_var_names(&interp.global_var_names);
    totals.interpreter_side_tables.add(bytes);
    if !exact {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap/HashSet bucket storage");
    }
    for value in &interp.kept_alive {
        visitor.value(value);
    }
    for value in interp.promise_forward.values() {
        visitor.value(value);
    }
    if !interp.promises.is_empty()
        || !interp.unhandled_rejections.is_empty()
        || !interp.host_settings_states.is_empty()
        || !interp.retired_host_job_contexts.is_empty()
        || !interp.promise_forward.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap/HashSet bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .generators
            .len()
            .saturating_mul(size_of::<(usize, crate::coroutine::Coroutine)>())
            .saturating_add(interp.async_gens.len().saturating_mul(size_of::<usize>()))
            .saturating_add(
                interp
                    .async_gen_busy
                    .len()
                    .saturating_mul(size_of::<usize>()),
            )
            .saturating_add(interp.async_gen_queue.len().saturating_mul(size_of::<(
                usize,
                std::collections::VecDeque<(Value, crate::coroutine::Resume)>,
            )>())),
    );
    for coroutine in interp.generators.values() {
        totals
            .interpreter_side_tables
            .add(coroutine.scan_retained_memory(visitor));
    }
    for queue in interp.async_gen_queue.values() {
        totals.interpreter_side_tables.add(
            queue
                .capacity()
                .saturating_mul(size_of::<(Value, crate::coroutine::Resume)>()),
        );
        for (promise, signal) in queue {
            visitor.value(promise);
            signal.scan_retained_memory(visitor);
        }
    }
    if !interp.generators.is_empty()
        || !interp.async_gens.is_empty()
        || !interp.async_gen_busy.is_empty()
        || !interp.async_gen_queue.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap/HashSet bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .weak_refs
            .len()
            .saturating_mul(size_of::<(usize, Option<crate::interpreter::WeakTarget>)>())
            .saturating_add(
                interp
                    .finalization_registries
                    .len()
                    .saturating_mul(size_of::<(usize, crate::interpreter::FinalizationState)>()),
            )
            .saturating_add(
                interp
                    .pending_finalization_cleanup
                    .capacity()
                    .saturating_mul(size_of::<Value>()),
            ),
    );
    totals
        .interpreter_side_tables
        .add(interp.weak_metadata.allocated_bytes());
    for registry in interp.finalization_registries.values() {
        visitor.value(&registry.cleanup_callback.callback);
        if let Some(global) = &registry.cleanup_callback.script_caller {
            visitor.value(&Value::Obj(global.clone()));
        }
        totals
            .interpreter_side_tables
            .add(registry.cells.allocated_bytes());
        for cell in registry.cells.iter() {
            // `target` and `unregister_token` are deliberately not upgraded: diagnostics must
            // never turn an ECMA-262 weak edge into a strong root. The cell buffer already
            // includes their inline Weak handles; only [[HeldValue]] is strongly retained.
            visitor.value(&cell.held_value);
        }
    }
    for registry in &interp.pending_finalization_cleanup {
        visitor.value(registry);
    }
    if !interp.weak_refs.is_empty() || !interp.finalization_registries.is_empty() {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .fn_frames
            .capacity()
            .saturating_mul(size_of::<crate::interpreter::FnFrame>())
            .saturating_add(
                interp
                    .code_frames
                    .capacity()
                    .saturating_mul(size_of::<crate::stack_trace::CodeFrame>()),
            )
            .saturating_add(
                interp
                    .job_script_contexts
                    .capacity()
                    .saturating_mul(size_of::<(
                        u32,
                        std::rc::Rc<crate::stack_trace::ScriptSource>,
                    )>()),
            )
            .saturating_add(interp.pending_fn_name.as_ref().map_or(0, String::capacity))
            .saturating_add(
                interp
                    .using_stack
                    .capacity()
                    .saturating_mul(size_of::<Vec<crate::interpreter::Disposable>>()),
            )
            .saturating_add(
                interp
                    .decorator_initializers
                    .capacity()
                    .saturating_mul(size_of::<Value>()),
            ),
    );
    visitor.value(&interp.new_target);
    visitor.value(&interp.pending_new_target);
    for frame in &interp.fn_frames {
        if let Some(extra) = &frame.extra {
            totals
                .interpreter_side_tables
                .add(size_of::<crate::interpreter::FrameExtra>());
            visitor.value(&extra.args_obj);
            if let Some((function, arguments, _environment)) = &extra.lazy {
                visitor.function(function);
                totals
                    .interpreter_side_tables
                    .add(visitor.value_slice(arguments));
            }
        }
    }
    if let Some(pending) = &interp.pending_tail {
        totals
            .interpreter_side_tables
            .add(size_of::<(Value, Value, Vec<Value>)>());
        visitor.value(&pending.0);
        visitor.value(&pending.1);
        totals
            .interpreter_side_tables
            .add(pending.2.capacity().saturating_mul(size_of::<Value>()));
        for argument in &pending.2 {
            visitor.value(argument);
        }
    }
    for frame in &interp.using_stack {
        totals.interpreter_side_tables.add(
            frame
                .capacity()
                .saturating_mul(size_of::<crate::interpreter::Disposable>()),
        );
        for resource in frame {
            visitor.value(&resource.value);
            visitor.value(&resource.method);
        }
    }
    for initializer in &interp.decorator_initializers {
        visitor.value(initializer);
    }

    totals.interpreter_side_tables.add(
        interp
            .gc_pins
            .len()
            .saturating_mul(size_of::<(usize, Gc)>())
            .saturating_add(
                interp
                    .proxies
                    .len()
                    .saturating_mul(size_of::<(usize, (Value, Value))>()),
            )
            .saturating_add(
                interp
                    .host_indexed
                    .len()
                    .saturating_mul(
                        size_of::<(usize, crate::interpreter::HostIndexedProperties)>(),
                    ),
            )
            .saturating_add(
                interp
                    .template_cache
                    .len()
                    .saturating_mul(size_of::<((usize, u64), Value)>()),
            )
            .saturating_add(
                interp
                    .annexb_fn_sync
                    .len()
                    .saturating_mul(size_of::<(usize, Rc<crate::ast::Function>)>()),
            )
            .saturating_add(
                interp
                    .deferred_ns
                    .len()
                    .saturating_mul(size_of::<(usize, String)>()),
            )
            .saturating_add(
                interp
                    .deferred_ns_objs
                    .len()
                    .saturating_mul(size_of::<(String, Value)>()),
            )
            .saturating_add(
                interp
                    .mapped_arguments
                    .len()
                    .saturating_mul(size_of::<(usize, (Env, Vec<Option<String>>))>()),
            )
            .saturating_add(
                interp
                    .module_source_objs
                    .len()
                    .saturating_mul(size_of::<(String, Value)>()),
            ),
    );
    for (target, handler) in interp.proxies.values() {
        visitor.value(target);
        visitor.value(handler);
    }
    for properties in interp.host_indexed.values() {
        visitor.value(&properties.getter);
        if let Some(live) = &properties.live {
            totals
                .interpreter_side_tables
                .add(size_of::<crate::interpreter::HostIndexedLive>() + 2 * size_of::<usize>());
            visitor.value(&live.state);
            if let Some(getter) = &live.named_getter {
                visitor.value(getter);
            }
        }
    }
    for value in interp.template_cache.values() {
        visitor.value(value);
    }
    for function in interp.annexb_fn_sync.values() {
        visitor.function(function);
    }
    for module in interp.deferred_ns.values() {
        totals.interpreter_side_tables.add(module.capacity());
    }
    for (module, namespace) in &interp.deferred_ns_objs {
        totals.interpreter_side_tables.add(module.capacity());
        visitor.value(namespace);
    }
    for (_environment, names) in interp.mapped_arguments.values() {
        totals
            .interpreter_side_tables
            .add(names.capacity().saturating_mul(size_of::<Option<String>>()));
        for name in names.iter().flatten() {
            totals.interpreter_side_tables.add(name.capacity());
        }
    }
    if let Some(registry) = &interp.borrowed_parameters {
        totals
            .interpreter_side_tables
            .add(registry.scan_retained_memory(visitor));
    }
    for (module, source) in &interp.module_source_objs {
        totals.interpreter_side_tables.add(module.capacity());
        visitor.value(source);
    }
    if !interp.gc_pins.is_empty()
        || !interp.proxies.is_empty()
        || !interp.host_indexed.is_empty()
        || !interp.template_cache.is_empty()
        || !interp.annexb_fn_sync.is_empty()
        || !interp.deferred_ns.is_empty()
        || !interp.deferred_ns_objs.is_empty()
        || !interp.mapped_arguments.is_empty()
        || !interp.module_source_objs.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .class_info
            .len()
            .saturating_mul(size_of::<(usize, crate::interpreter::ClassInfo)>()),
    );
    for class in interp.class_info.values() {
        totals
            .interpreter_side_tables
            .add(class.scan_retained_memory(visitor));
    }
    let (construct_ic_bytes, construct_ics_exact) = interp.construct_ic_retained_memory();
    totals.interpreter_side_tables.add(construct_ic_bytes);
    totals
        .interpreter_side_tables
        .add(interp.call_overflow.retained_bytes());
    if let Some(cache) = &interp.native_callback_cache {
        totals.engine_caches.add(cache.retained_bytes());
        totals
            .engine_caches
            .make_lower_bound("opaque callback HashMap bucket and Weak allocation storage");
    }
    if let Some(caches) = &interp.proxy_trap_caches {
        totals.engine_caches.add(caches.retained_bytes());
        totals
            .engine_caches
            .make_lower_bound("opaque callback HashMap bucket and Weak allocation storage");
    }
    if let Some(source) = interp.self_hosted.get() {
        totals
            .engine_caches
            .add(source.scan_retained_memory(visitor));
    }
    totals.engine_caches.add(
        interp
            .shared_snapshots
            .capacity()
            .saturating_mul(std::mem::size_of::<crate::interpreter::SharedSnapshot>()),
    );
    for snapshot in &interp.shared_snapshots {
        // Identity-deduplicated with the Function/Class allocations closures retain.
        visitor.stmt_body(&snapshot.body);
    }
    totals
        .engine_caches
        .add(interp.fragment_cache.scan_retained_memory(visitor));
    totals
        .interpreter_side_tables
        .add(interp.computed_reads.scan_retained_memory(visitor));
    if !interp.class_info.is_empty() || !construct_ics_exact {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .realms
            .len()
            .saturating_mul(size_of::<(usize, crate::interpreter::RealmState)>()),
    );
    for realm in interp.realms.values() {
        let (bytes, exact) = realm.scan_retained_memory(visitor);
        totals.interpreter_side_tables.add(bytes);
        if !exact {
            totals
                .interpreter_side_tables
                .make_lower_bound("opaque standard-library HashMap bucket storage");
        }
    }
    if let Some(realm) = &interp.ctor_caller_realm {
        let (bytes, exact) = realm.scan_retained_memory(visitor);
        totals.interpreter_side_tables.add(bytes);
        if !exact {
            totals
                .interpreter_side_tables
                .make_lower_bound("opaque standard-library HashMap bucket storage");
        }
    }
    if !interp.realms.is_empty() {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .temporal
            .len()
            .saturating_mul(size_of::<(usize, crate::temporal::Temporal)>())
            .saturating_add(
                interp
                    .temporal_cal
                    .len()
                    .saturating_mul(size_of::<(usize, Rc<str>)>()),
            ),
    );
    for temporal in interp.temporal.values() {
        temporal.scan_retained_memory(visitor);
    }
    for calendar in interp.temporal_cal.values() {
        visitor.rc_str(calendar);
    }
    if !interp.temporal.is_empty() || !interp.temporal_cal.is_empty() {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .pending_async_waits
            .capacity()
            .saturating_mul(size_of::<crate::interpreter::AsyncWait>())
            .saturating_add(
                interp
                    .async_wait_inbox
                    .as_ref()
                    .map_or(0, |_| size_of::<crate::interpreter::AsyncWaitInbox>()),
            )
            .saturating_add(
                interp
                    .pending_timers
                    .capacity()
                    .saturating_mul(size_of::<(Value, std::time::Instant)>()),
            ),
    );
    for wait in &interp.pending_async_waits {
        visitor.value(&wait.promise);
        visitor.shared_array_buffer(wait.block.id());
    }
    for (callback, _) in &interp.pending_timers {
        visitor.value(callback);
    }
    if let Some(agent) = &interp.agent {
        totals.interpreter_side_tables.add(
            size_of::<crate::interpreter::AgentChannels>().saturating_add(
                agent
                    .agent_broadcast_txs
                    .capacity()
                    .saturating_mul(size_of::<std::sync::mpsc::Sender<(u64, usize)>>()),
            ),
        );
    }
    if interp.async_wait_inbox.is_some() || interp.agent.is_some() {
        totals.interpreter_side_tables.make_lower_bound(
            "standard-library channel backing and queued message storage are opaque",
        );
    }

    totals
        .interpreter_side_tables
        .add(interp.collection_iterators.len().saturating_mul(size_of::<(
            usize,
            crate::ordered_collection::CollectionIterator,
        )>()));
    for state in interp.collection_iterators.values() {
        if let Some(target) = &state.target {
            visitor.value(&Value::Obj(target.clone()));
        }
    }
    totals
        .interpreter_side_tables
        .add(interp.weak_collection_data.len().saturating_mul(size_of::<(
            usize,
            Vec<(crate::interpreter::WeakTarget, Value)>,
        )>()));
    for entries in interp.weak_collection_data.values() {
        totals.interpreter_side_tables.add(
            entries
                .capacity()
                .saturating_mul(size_of::<(crate::interpreter::WeakTarget, Value)>()),
        );
        // Weak keys must not become strong merely because diagnostics are enabled. WeakMap values
        // are visited because the collector has already resolved ephemeron liveness.
        for (_, value) in entries {
            visitor.value(value);
        }
    }
    totals.interpreter_side_tables.add(
        interp
            .weak_collection_index
            .capacity()
            .saturating_mul(size_of::<(usize, crate::weak_metadata::WeakCollectionIndex)>()),
    );
    for index in interp.weak_collection_index.values() {
        totals.interpreter_side_tables.add(
            index
                .entries
                .capacity()
                .saturating_mul(size_of::<(crate::interpreter::WeakKey, usize)>()),
        );
    }
    if !interp.collection_iterators.is_empty()
        || !interp.weak_collection_data.is_empty()
        || !interp.weak_collection_index.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    totals.interpreter_side_tables.add(
        interp
            .array_buffers
            .len()
            .saturating_mul(size_of::<(usize, crate::interpreter::BufferData)>())
            .saturating_add(
                interp
                    .array_buffer_versions
                    .len()
                    .saturating_mul(size_of::<(usize, u64)>()),
            )
            .saturating_add(
                interp
                    .array_buffer_dirty_ranges
                    .len()
                    .saturating_mul(size_of::<(usize, Vec<std::ops::Range<usize>>)>()),
            )
            .saturating_add(
                interp
                    .immutable_buffers
                    .len()
                    .saturating_mul(size_of::<usize>()),
            )
            .saturating_add(
                interp
                    .host_keyed_buffers
                    .len()
                    .saturating_mul(size_of::<usize>()),
            )
            .saturating_add(
                interp
                    .typed_arrays
                    .len()
                    .saturating_mul(size_of::<(usize, crate::value::TaInfo)>()),
            )
            .saturating_add(
                interp
                    .ta_buffer
                    .len()
                    .saturating_mul(size_of::<(usize, Value)>()),
            )
            .saturating_add(
                interp
                    .data_views
                    .len()
                    .saturating_mul(size_of::<(usize, (usize, usize, usize, bool))>()),
            ),
    );
    for ranges in interp.array_buffer_dirty_ranges.values() {
        totals.interpreter_side_tables.add(
            ranges
                .capacity()
                .saturating_mul(size_of::<std::ops::Range<usize>>()),
        );
    }
    totals
        .interpreter_side_tables
        .add(interp.native_buffers.len().saturating_mul(size_of::<(
            usize,
            std::rc::Rc<crate::native_typed_array::NativeBuffer>,
        )>()));
    for buffer in interp.native_buffers.values() {
        totals
            .interpreter_side_tables
            .add(visitor.native_buffer(buffer));
    }
    for buffer in interp.ta_buffer.values() {
        visitor.value(buffer);
    }
    if !interp.array_buffers.is_empty()
        || !interp.array_buffer_versions.is_empty()
        || !interp.array_buffer_dirty_ranges.is_empty()
        || !interp.immutable_buffers.is_empty()
        || !interp.host_keyed_buffers.is_empty()
        || !interp.typed_arrays.is_empty()
        || !interp.ta_buffer.is_empty()
        || !interp.data_views.is_empty()
    {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap/HashSet bucket storage");
    }
    for data in interp.array_buffers.values() {
        match data {
            crate::interpreter::BufferData::Local(bytes) => visitor.array_buffer(bytes),
            crate::interpreter::BufferData::Shared(block) => {
                visitor.shared_array_buffer(block.id())
            }
        }
    }

    totals.interpreter_side_tables.add(
        interp
            .shadow_realms
            .len()
            .saturating_mul(size_of::<(usize, Box<Interp>)>()),
    );
    if !interp.shadow_realms.is_empty() {
        totals
            .interpreter_side_tables
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }
    for sub in interp.shadow_realms.values() {
        let objects = crate::value::heap_gc_snapshot(&sub.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&sub.gc_heap);
        scan_realm(sub, &objects, &scopes, visitor, totals);
    }
}

fn scan_symbol_agent(interp: &Interp, visitor: &mut Visitor, totals: &mut DirectTotals) {
    let agent = interp.symbol_agent.borrow();
    totals
        .interpreter_side_tables
        .add(size_of::<crate::value::SymbolAgentState>());
    totals.interpreter_side_tables.add(
        agent
            .symbols
            .len()
            .saturating_mul(size_of::<(u64, crate::value::WeakSymbol)>())
            .saturating_add(
                agent
                    .global_by_key
                    .len()
                    .saturating_mul(size_of::<(Rc<str>, crate::value::SymbolRef)>()),
            )
            .saturating_add(
                agent
                    .global_key_by_id
                    .len()
                    .saturating_mul(size_of::<(u64, Rc<str>)>()),
            )
            .saturating_add(
                agent
                    .well_known
                    .len()
                    .saturating_mul(size_of::<(&'static str, crate::value::SymbolRef)>()),
            ),
    );
    totals
        .interpreter_side_tables
        .make_lower_bound("opaque standard-library HashMap bucket storage");
    for (key, symbol) in &agent.global_by_key {
        visitor.rc_str(key);
        visitor.symbol(symbol);
    }
    for key in agent.global_key_by_id.values() {
        visitor.rc_str(key);
    }
    for symbol in agent.well_known.values() {
        visitor.symbol(symbol);
    }
    // The weak identity table owns no Symbol payload. `memory_snapshots` is diagnostic output
    // produced by this visitor and is excluded so measurement cannot inflate the workload.
}

fn measure(interp: &Interp, objects: &[Gc], scopes: &[Env]) -> Snapshot {
    let mut visitor = Visitor::default();
    let mut totals = DirectTotals::default();
    scan_realm(interp, objects, scopes, &mut visitor, &mut totals);
    scan_symbol_agent(interp, &mut visitor, &mut totals);
    visitor
        .shared_array_buffer_allocations
        .sort_unstable_by_key(|allocation| allocation.id);
    let wasm_backing_bytes = visitor
        .external_allocations
        .values()
        .filter(|allocation| allocation.kind == crate::host::RetainedExternalKind::WasmMemory)
        .map(|allocation| allocation.bytes)
        .fold(0usize, usize::saturating_add);

    totals
        .property_storage
        .add(visitor.detached_property_storage);
    if visitor.detached_property_storage_opaque {
        totals
            .property_storage
            .make_lower_bound("opaque standard-library HashMap bucket storage");
    }

    Snapshot {
        object_bodies: Category::exact(
            totals
                .object_count
                .saturating_mul(size_of::<RefCell<Object>>()),
        ),
        property_storage: totals.property_storage,
        scope_bodies: Category::exact(
            totals
                .scope_count
                .saturating_mul(size_of::<RefCell<Scope>>()),
        ),
        scope_storage: totals.scope_storage,
        strings_symbols_bigints: Category::exact(visitor.strings_symbols_bigints),
        callable_metadata: if visitor
            .unreported_native_closures
            .is_subset(&visitor.reported_native_closures)
            && !visitor.native_managed_identity_conflict
        {
            Category::exact(visitor.callable_metadata)
        } else {
            Category::incomplete_lower_bound(
                visitor.callable_metadata,
                "native closure captured allocations are unreported or have conflicting identities",
            )
        },
        function_bytecode_metadata: if visitor.function_bytecode_opaque_storage {
            Category::lower_bound(
                visitor.function_bytecode_metadata,
                "Chunk call-pin HashMap bucket storage is opaque",
            )
        } else {
            Category::exact(visitor.function_bytecode_metadata)
        },
        jit_heap_metadata: Category::exact(visitor.jit_heap_metadata),
        regexp_metadata: Category::exact(visitor.regexp_metadata),
        engine_caches: totals.engine_caches,
        interpreter_side_tables: totals.interpreter_side_tables,
        array_buffer_backing: Category::exact(visitor.array_buffer_bytes()),
        shared_array_buffer_backing: SharedBackingCategory {
            allocations: visitor.shared_array_buffer_allocations,
            unavailable_ids: visitor.unavailable_shared_array_buffers,
        },
        wasm_backing: WasmBackingCategory {
            bytes: wasm_backing_bytes,
            identity_conflict: visitor.external_identity_conflict,
            unclassified_external_entries: totals.host_resources.unclassified_external_entries,
        },
        host_resources: totals.host_resources.into(),
    }
}

/// Record the managed-memory snapshot of a finished collection, for `TRUST_JS_PERF_METRICS`.
/// Only the reference-counted collectors called it: under the GC heap the record says
/// "unavailable" (open since P1 S2; see docs/v8-chase-2026-10-15.txt, S6 step 4).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn record_post_gc(interp: &Interp, objects: &[Gc], scopes: &[Env]) {
    let snapshot = measure(interp, objects, scopes);
    let heap_id = crate::value::heap_id(&interp.gc_heap);
    interp
        .symbol_agent
        .borrow_mut()
        .memory_snapshots
        .insert(heap_id, snapshot);
}

pub(crate) fn json(interp: &Interp) -> String {
    let heap_id = crate::value::heap_id(&interp.gc_heap);
    let agent = interp.symbol_agent.borrow();
    match agent.memory_snapshots.get(&heap_id) {
        Some(snapshot) => snapshot.json(agent.agent_id, heap_id),
        None => format!(
            concat!(
                "{{\"schema_version\":1,\"agent_id\":{},\"heap_id\":{},",
                "\"safepoint\":null,\"complete\":false,",
                "\"managed_requested_bytes\":{{\"bytes\":null,\"quality\":\"unavailable\",\"reason\":\"no post-collection safepoint has been recorded\"}},",
                "\"managed_external_bytes\":{{\"bytes\":null,\"quality\":\"unavailable\",\"reason\":\"no post-collection safepoint has been recorded\"}}}}"
            ),
            agent.agent_id, heap_id
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{
        HostRetainedMemoryVisitor, RetainedBytes, RetainedExternalAllocation,
        RetainedExternalMemory, RetainedMemory,
    };
    use crate::lstr::LStr;
    use crate::value::Props;

    struct ReportedHostState(Vec<u8>);

    struct ExternalWasmState(crate::interpreter::ArrayBufferBytes);

    struct ReportedNativeClosure {
        allocation: Rc<RefCell<Vec<u8>>>,
        value: Value,
    }

    struct ReportedManagedHostState {
        allocation: Rc<RefCell<Vec<u8>>>,
        value: Value,
        requested_bytes: usize,
    }

    crate::value_free!(ReportedHostState, ExternalWasmState);

    impl crate::host::HostTrace for ReportedManagedHostState {
        fn trace_gc(&self, visitor: &mut dyn crate::host::HostGcVisitor) {
            visitor.root(&self.value);
        }
    }

    impl RetainedBytes for ReportedHostState {
        fn retained_bytes(&self) -> usize {
            self.0.capacity()
        }
    }

    impl RetainedExternalMemory for ExternalWasmState {
        fn retained_external_memory(&self, visit: &mut dyn FnMut(RetainedExternalAllocation)) {
            visit(RetainedExternalAllocation::wasm_array_buffer(&self.0));
        }
    }

    impl RetainedMemory for ReportedManagedHostState {
        fn scan_retained_memory(&self, visitor: &mut dyn HostRetainedMemoryVisitor) {
            visitor.allocation(crate::value::RetainedManagedAllocation::rc(
                "trust-js-test.host-shared-buffer",
                &self.allocation,
                self.requested_bytes,
            ));
            visitor.value(&self.value);
        }
    }

    impl crate::value::NativeCallableRetained for ReportedNativeClosure {
        fn scan_retained_memory(
            &self,
            visitor: &mut dyn crate::value::NativeRetainedMemoryVisitor,
        ) {
            let requested_bytes =
                size_of::<RefCell<Vec<u8>>>().saturating_add(self.allocation.borrow().capacity());
            visitor.allocation(crate::value::RetainedManagedAllocation::rc(
                "trust-js-test.native-shared-buffer",
                &self.allocation,
                requested_bytes,
            ));
            visitor.value(&self.value);
        }
    }

    #[test]
    fn property_storage_counts_capacity_not_length() {
        let props = Props::with_capacity(17);
        let (bytes, exact) = props.retained_requested_storage_bytes();
        assert!(exact);
        assert!(bytes >= 17 * size_of::<crate::value::Property>());
    }

    #[test]
    fn shared_property_layouts_are_counted_once_including_unused_predictions() {
        let layout = crate::value::new_property_layout(vec![
            Rc::from("a"),
            Rc::from("b"),
            Rc::from("unused"),
        ]);
        let mut a = Props::with_layout(1, Some(layout.clone()));
        let mut b = Props::with_layout(2, Some(layout.clone()));
        a.insert("a", crate::value::Property::plain(Value::Undefined));
        b.insert("a", crate::value::Property::plain(Value::Undefined));
        b.insert("b", crate::value::Property::plain(Value::Undefined));
        let mut visitor = Visitor::default();
        visitor.props(&a);
        visitor.props(&b);
        visitor.property_layout(&layout); // also retained by a hypothetical constructor hint
        assert_eq!(visitor.property_layouts.len(), 1);
        assert_eq!(visitor.rc_strs.len(), 3);
        assert_eq!(
            visitor.detached_property_storage,
            crate::value::detached_field_bytes(1)
                + crate::value::detached_field_bytes(2)
                + size_of::<crate::value::PropertyLayoutData>()
                + layout.capacity() * size_of::<Rc<str>>()
        );
    }

    #[test]
    fn shared_ordinary_lookup_index_is_counted_once_per_layout() {
        let mut engine = crate::Engine::new();
        engine.interp.activate_gc_heap();
        engine
            .eval(
                "function make(){var o={};for(var i=0;i<12;i++)o['field'+i]=i;return o}var a=make(),b=make();",
                false,
            )
            .expect("layout fixture parses");
        let env = engine.interp.global_env.clone();
        let Value::Obj(a) = engine
            .interp
            .get_var("a", &env)
            .ok()
            .expect("a is initialized")
        else {
            panic!("a is an object")
        };
        let Value::Obj(b) = engine
            .interp
            .get_var("b", &env)
            .ok()
            .expect("b is initialized")
        else {
            panic!("b is an object")
        };
        let a = a.borrow().props.clone();
        let b = b.borrow().props.clone();
        let layout = a.shared_layout().expect("shared layout");
        assert!(layout.has_index());
        assert!(Rc::ptr_eq(layout, b.shared_layout().expect("b layout")));

        let expected_layout = size_of::<crate::value::PropertyLayoutData>()
            + layout.capacity() * size_of::<Rc<str>>()
            + size_of::<crate::fasthash::FastMap<Rc<str>, usize>>()
            + layout.index_capacity().unwrap() * size_of::<(Rc<str>, usize)>();
        let mut visitor = Visitor::default();
        visitor.props(&a);
        visitor.props(&b);
        assert_eq!(visitor.property_layouts.len(), 1);
        assert_eq!(
            visitor.detached_property_storage,
            24 * size_of::<crate::value::Property>() + expected_layout
        );
        assert!(
            visitor.detached_property_storage_opaque,
            "HashMap bucket bytes remain a documented lower bound"
        );
    }

    #[test]
    fn for_in_cache_layout_pins_are_censused_once_after_source_dies() {
        let mut engine = crate::Engine::new();
        engine.interp.activate_gc_heap();
        let layout = crate::value::new_property_layout(vec![
            Rc::from("live"),
            Rc::from("unused prediction"),
        ]);
        let weak_layout = Rc::downgrade(&layout);
        let expected_layout_bytes = size_of::<crate::value::PropertyLayoutData>()
            + layout.capacity() * size_of::<Rc<str>>();
        // The source lives only in this helper frame (the collector scans the stack).
        let (weak_source, _keys) = crate::gc_test::in_frame(|| {
            let mut props = Props::with_layout(1, Some(layout.clone()));
            props.append_initialized_field(
                &layout[0],
                crate::value::Property::plain(Value::Undefined),
            );
            let source = Value::Obj(crate::value::Object::new_with_parts(
                None,
                props,
                Exotic::None,
            ));
            let weak_source = Gc::downgrade(source.as_obj().unwrap());
            let keys = engine
                .interp
                .for_in_keys(&source)
                .ok()
                .expect("ordinary keys");
            (weak_source, keys)
        });
        drop(layout);
        // A collection reclaims the dropped source.
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(&weak_source));
        assert_eq!(
            weak_layout.strong_count(),
            1,
            "cache is the only layout owner"
        );
        let mut visitor = Visitor::default();
        for _ in 0..2 {
            engine
                .interp
                .enumeration_keys
                .scan_retained_memory(|entry| {
                    entry.scan_retained_layouts(&mut visitor);
                });
        }
        assert_eq!(visitor.property_layouts.len(), 1);
        assert_eq!(
            visitor.rc_strs.len(),
            2,
            "unused predictions are retained too"
        );
        assert_eq!(visitor.detached_property_storage, expected_layout_bytes);
    }

    #[test]
    fn unavailable_snapshot_is_not_reported_as_zero() {
        let json = Snapshot {
            object_bodies: Category::exact(1),
            property_storage: Category::lower_bound(2, "test lower bound"),
            scope_bodies: Category::exact(3),
            scope_storage: Category::exact(4),
            strings_symbols_bigints: Category::lower_bound(5, "test lower bound"),
            callable_metadata: Category::lower_bound(6, "test lower bound"),
            function_bytecode_metadata: Category::lower_bound(8, "test lower bound"),
            jit_heap_metadata: Category::lower_bound(9, "test lower bound"),
            regexp_metadata: Category::lower_bound(10, "test lower bound"),
            engine_caches: Category::lower_bound(11, "test lower bound"),
            interpreter_side_tables: Category::lower_bound(12, "test lower bound"),
            array_buffer_backing: Category::lower_bound(7, "test lower bound"),
            shared_array_buffer_backing: SharedBackingCategory::default(),
            wasm_backing: WasmBackingCategory {
                bytes: 0,
                identity_conflict: false,
                unclassified_external_entries: 0,
            },
            host_resources: HostCategory {
                reported_bytes: 0,
                unavailable_entries: 1,
                identity_conflict: false,
                opaque_storage: false,
            },
        }
        .json(1, 1);
        assert!(json.contains("\"interpreter_side_tables\":{\"bytes\":12"));
        assert!(json.contains("\"managed_requested_bytes\":{\"bytes\":71"));
        assert!(json.contains("\"host_resources\":{\"bytes\":null"));
    }

    #[test]
    fn host_snapshot_requires_every_live_entry_to_report_retained_bytes() {
        let empty = Interp::new();
        let empty_snapshot = measure(&empty, &[], &[]);
        assert_eq!(empty_snapshot.host_resources.reported_bytes, 0);
        assert_eq!(empty_snapshot.host_resources.unavailable_entries, 0);
        assert!(!empty_snapshot.host_resources.opaque_storage);
        assert!(empty_snapshot
            .json(1, 1)
            .contains("\"host_resources\":{\"bytes\":0,\"quality\":\"exact\"}"));
        assert!(empty_snapshot.complete());
        assert!(empty_snapshot.json(1, 1).contains("\"complete\":true"));

        let mut reported = Interp::new();
        reported
            .host_state
            .put_retained(ReportedHostState(Vec::with_capacity(47)));
        let reported_snapshot = measure(&reported, &[], &[]);
        assert_eq!(reported_snapshot.host_resources.unavailable_entries, 0);
        assert!(reported_snapshot.host_resources.reported_bytes >= 47);
        assert!(reported_snapshot.host_resources.opaque_storage);
        assert!(reported_snapshot.complete());

        reported.host_state.put(String::from("opaque host state"));
        let unavailable_snapshot = measure(&reported, &[], &[]);
        assert_eq!(unavailable_snapshot.host_resources.unavailable_entries, 1);
        assert!(!unavailable_snapshot.complete());
        let json = unavailable_snapshot.json(1, 1);
        assert!(json.contains("\"host_resources\":{\"bytes\":null"));
        assert!(json.contains("reachable host owners lack complete retained-memory traversal"));
    }

    #[test]
    fn string_view_census_deduplicates_flat_owner() {
        let source: LStr = "x".repeat(1024).into();
        let a = source.slice_bytes(0, 900);
        let b = a.slice_bytes(20, 880);
        let mut visitor = Visitor::default();
        visitor.lstr(&a);
        visitor.lstr(&b);
        visitor.lstr(&source);
        visitor.lstr(&a);
        assert_eq!(
            visitor.strings_symbols_bigints,
            source.allocation_requested_bytes()
                + a.allocation_requested_bytes()
                + b.allocation_requested_bytes()
        );
        assert_eq!(
            a.retained_requested_bytes(),
            source.allocation_requested_bytes() + a.allocation_requested_bytes()
        );
    }

    #[test]
    fn prepared_window_census_retains_and_deduplicates_shared_tables() {
        let source: LStr = format!("{}é😀", "x".repeat(1024)).into();
        let parent = Rc::new(crate::regex::ReText::new_rc(true, &source));
        let a = source.slice_bytes(100, source.len());
        let b = source.slice_bytes(200, source.len());
        let first = Rc::new(crate::regex::ReText::window(parent.clone(), &a, 100, 1027).unwrap());
        let second = Rc::new(crate::regex::ReText::window(parent.clone(), &b, 200, 1027).unwrap());
        let mut visitor = Visitor::default();
        visitor.re_text(&first);
        visitor.re_text(&second);
        visitor.re_text(&parent);
        visitor.re_text(&first);
        assert_eq!(
            visitor.regexp_metadata,
            parent.heap_bytes() + 2 * size_of::<crate::regex::ReText>()
        );
    }

    #[test]
    fn identity_aware_host_allocations_and_values_are_deduplicated() {
        let mut interp = Interp::new();
        let baseline_strings = measure(&interp, &[], &[]).strings_symbols_bigints.bytes;
        let allocation = Rc::new(RefCell::new(Vec::<u8>::with_capacity(83)));
        let string = LStr::from("host-retained string");
        let requested_bytes = size_of::<RefCell<Vec<u8>>>() + allocation.borrow().capacity();
        let make_reporter = || ReportedManagedHostState {
            allocation: allocation.clone(),
            value: Value::Str(string.clone()),
            requested_bytes,
        };
        interp.host_state.put_retained_memory(make_reporter());
        interp
            .host_state
            .resources
            .add_retained_memory(make_reporter());

        let snapshot = measure(&interp, &[], &[]);
        assert_eq!(snapshot.host_resources.unavailable_entries, 0);
        assert!(!snapshot.host_resources.identity_conflict);
        assert!(snapshot.host_resources.reported_bytes >= requested_bytes);
        assert_eq!(
            snapshot
                .strings_symbols_bigints
                .bytes
                .saturating_sub(baseline_strings),
            string.retained_requested_bytes()
        );
        assert!(snapshot.complete());

        interp
            .host_state
            .put_retained_memory(ReportedManagedHostState {
                allocation,
                value: Value::Undefined,
                requested_bytes: requested_bytes + 1,
            });
        let conflicting = measure(&interp, &[], &[]);
        assert!(conflicting.host_resources.identity_conflict);
        assert!(!conflicting.complete());
        assert!(conflicting
            .json(1, 1)
            .contains("conflicting byte counts were reported for one host allocation identity"));
    }

    #[test]
    fn wasm_backing_alias_is_not_credited_again_as_an_array_buffer() {
        let mut interp = Interp::new();
        let storage = Rc::new(RefCell::new(Vec::with_capacity(79)));
        interp
            .array_buffers
            .insert(1, crate::interpreter::BufferData::Local(storage.clone()));
        interp
            .host_state
            .put_external_memory(ExternalWasmState(storage.clone()));
        interp
            .host_state
            .resources
            .add_external_memory(ExternalWasmState(storage));

        let snapshot = measure(&interp, &[], &[]);
        assert_eq!(snapshot.array_buffer_backing.bytes, 0);
        assert_eq!(snapshot.wasm_backing.bytes, 79);
        assert!(!snapshot.wasm_backing.identity_conflict);
        assert_eq!(snapshot.host_resources.unavailable_entries, 2);
        assert_eq!(snapshot.wasm_backing.unclassified_external_entries, 0);
        assert!(snapshot.wasm_backing.coverage_complete());
        let json = snapshot.json(1, 1);
        assert!(json.contains("\"wasm_backing\":{\"bytes\":79"));
        assert!(json.contains("\"managed_external_bytes\":{\"bytes\":79,\"quality\":\"exact\"}"));
    }

    #[test]
    fn shared_string_allocations_are_counted_once() {
        let string = LStr::from("shared allocation");
        let expected = string.retained_requested_bytes();
        let mut visitor = Visitor::default();
        visitor.value(&Value::Str(string.clone()));
        visitor.value(&Value::Str(string));
        assert_eq!(visitor.strings_symbols_bigints, expected);
    }

    #[test]
    fn cache_pins_are_scanned_into_canonical_allocation_families() {
        let mut interp = Interp::new();
        let non_ascii = LStr::from("éééé");
        interp.units_full(&non_ascii);
        interp.re_text(true, &non_ascii);
        let ascii = LStr::from("cache-hot-ascii");
        interp.re_text(false, &ascii);
        interp
            .compiled_regexp("(?:cache)+", "gi")
            .unwrap_or_else(|_| panic!("regexp compiles"));

        let objects = crate::value::heap_gc_snapshot(&interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&interp.gc_heap);
        let first = measure(&interp, &objects, &scopes);
        let second = measure(&interp, &objects, &scopes);

        assert!(first.engine_caches.bytes > 0);
        assert!(first.regexp_metadata.bytes > 0);
        assert!(first.strings_symbols_bigints.bytes >= non_ascii.len() + ascii.len());
        assert_eq!(first.json(9, 12), second.json(9, 12));
    }

    #[test]
    fn regexp_side_tables_scan_match_state_and_program_pins() {
        let mut engine = crate::Engine::new();
        engine
            .eval("/(cache)(?<tail>.*)/giu.exec('cache-state')", false)
            .expect("evaluates");

        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let snapshot = measure(&engine.interp, &objects, &scopes);

        assert!(snapshot.interpreter_side_tables.bytes > 0);
        assert!(snapshot.regexp_metadata.bytes > 0);
    }

    #[test]
    fn regexp_allocations_are_exact_and_identity_deduplicated() {
        let regexes = [
            Rc::new(crate::regex::Regex::new("[a-z]+(?=[A-Z])", "").unwrap()),
            Rc::new(crate::regex::Regex::new(r"\p{Basic_Emoji}", "v").unwrap()),
            Rc::new(
                crate::regex::Regex::new(r"(?:(?<letter>a)|(?<letter>b))\k<letter>", "u").unwrap(),
            ),
        ];
        let mut visitor = Visitor::default();
        for regex in &regexes {
            visitor.regex(regex);
        }
        let first_bytes = visitor.regexp_metadata;
        assert!(first_bytes > 0);
        assert!(!visitor.regexp_programs.is_empty());
        assert!(!visitor.regexp_char_classes.is_empty());
        assert!(!visitor.regexp_string_sets.is_empty());
        assert!(!visitor.regexp_backref_groups.is_empty());

        for regex in &regexes {
            visitor.regex(regex);
        }
        assert_eq!(visitor.regexp_metadata, first_bytes);
    }

    #[test]
    fn requested_payload_categories_do_not_blame_excluded_allocator_metadata() {
        let interp = Interp::new();
        let snapshot = measure(&interp, &[], &[]);

        assert!(matches!(
            snapshot.strings_symbols_bigints.quality,
            Quality::Exact
        ));
        assert!(matches!(snapshot.jit_heap_metadata.quality, Quality::Exact));
        assert!(matches!(
            snapshot.array_buffer_backing.quality,
            Quality::Exact
        ));
        assert!(matches!(snapshot.regexp_metadata.quality, Quality::Exact));
    }

    #[test]
    fn native_closure_captures_keep_the_snapshot_incomplete() {
        let interp = Interp::new();
        let retained = Vec::<u8>::with_capacity(101);
        let closure: Rc<crate::value::NativeClosure> = Rc::new(
            move |_: &mut Interp, _: Value, _: &[Value]| -> Result<Value, Value> {
                std::hint::black_box(retained.len());
                Ok(Value::Undefined)
            },
        );
        let object =
            interp.make_native_closure("capturing", 0, closure, crate::value::NativeCaptures::None);
        let snapshot = measure(&interp, &[object], &[]);

        assert!(!snapshot.callable_metadata.coverage_complete);
        assert!(matches!(
            snapshot.callable_metadata.quality,
            Quality::LowerBound
        ));
        assert!(!snapshot.complete());
        assert!(snapshot.json(1, 1).contains(
            "native closure captured allocations are unreported or have conflicting identities"
        ));
    }

    #[test]
    fn explicit_native_captures_are_complete_and_shared_storage_is_counted_once() {
        let string = LStr::from("explicit native capture");
        let callable = Callable::NativeData(Rc::new(crate::value::NativeCallable {
            body: crate::value::NativeCallableBody::Captured {
                function: |_, _, _, captures| Ok(captures[0].clone()),
                captures: vec![Value::Str(string.clone()), Value::Str(string.clone())]
                    .into_boxed_slice(),
            },
            retained: None,
            identity: Rc::from("captured"),
        }));
        let mut visitor = Visitor::default();
        visitor.callable(&callable);
        visitor.callable(&callable.clone());
        assert_eq!(
            visitor.callable_metadata,
            size_of::<crate::value::NativeCallable>() + 2 * size_of::<Value>()
        );
        assert_eq!(
            visitor.strings_symbols_bigints,
            string.retained_requested_bytes() + "captured".len()
        );
        assert!(visitor.native_closure_allocations.is_empty());
        assert!(visitor.unreported_native_closures.is_empty());

        let interp = Interp::new();
        let first = Object::new(None);
        let second = Object::new(None);
        first.borrow_mut().call = callable.clone();
        second.borrow_mut().call = callable;
        let snapshot = measure(&interp, &[first, second], &[]);
        assert!(matches!(snapshot.callable_metadata.quality, Quality::Exact));
        assert!(snapshot.callable_metadata.coverage_complete);
        assert!(snapshot.complete());
    }

    #[test]
    fn reported_native_closure_allocations_and_values_are_deduplicated() {
        let interp = Interp::new();
        let allocation = Rc::new(RefCell::new(Vec::<u8>::with_capacity(137)));
        let string = LStr::from("shared native capture");
        let make_closure = || {
            let state = Rc::new(ReportedNativeClosure {
                allocation: allocation.clone(),
                value: Value::Str(string.clone()),
            });
            let callable_state = state.clone();
            let callable: Rc<crate::value::NativeClosure> = Rc::new(
                move |_: &mut Interp, _: Value, _: &[Value]| -> Result<Value, Value> {
                    std::hint::black_box(&callable_state);
                    Ok(Value::Undefined)
                },
            );
            (
                callable,
                state as Rc<dyn crate::value::NativeCallableRetained>,
            )
        };
        let (first_callable, first_reporter) = make_closure();
        let first = interp.make_native_closure(
            "first",
            0,
            first_callable,
            crate::value::NativeCaptures::Reported(first_reporter),
        );
        let (second_callable, second_reporter) = make_closure();
        let second = interp.make_native_closure(
            "second",
            0,
            second_callable,
            crate::value::NativeCaptures::Reported(second_reporter),
        );
        let snapshot = measure(&interp, &[first, second], &[]);

        assert!(matches!(snapshot.callable_metadata.quality, Quality::Exact));
        assert!(snapshot.callable_metadata.coverage_complete);
        assert!(snapshot.complete());

        let mut visitor = Visitor::default();
        let (first_func, first_retained) = make_closure();
        let first_callable =
            crate::value::Callable::NativeData(Rc::new(crate::value::NativeCallable {
                body: crate::value::NativeCallableBody::Opaque(first_func),
                retained: Some(first_retained),
                identity: Rc::from("first"),
            }));
        let (second_func, second_retained) = make_closure();
        let second_callable =
            crate::value::Callable::NativeData(Rc::new(crate::value::NativeCallable {
                body: crate::value::NativeCallableBody::Opaque(second_func),
                retained: Some(second_retained),
                identity: Rc::from("second"),
            }));
        visitor.callable(&first_callable);
        visitor.callable(&second_callable);
        assert_eq!(visitor.native_managed_allocations.len(), 1);
        assert_eq!(
            visitor.strings_symbols_bigints,
            string.retained_requested_bytes() + "first".len() + "second".len()
        );
        assert!(visitor.unreported_native_closures.is_empty());
    }

    #[test]
    fn native_diagnostic_identity_is_not_the_mutable_name_property() {
        let interp = Interp::new();
        let callable = interp.make_native_closure(
            "registeredOperation",
            0,
            Rc::new(|_i, _this, _args| Ok(Value::Undefined)),
            crate::value::NativeCaptures::None,
        );
        callable.borrow_mut().props.insert(
            "name",
            crate::value::Property::data(
                Value::lstr("authorRenamed".to_string()),
                true,
                false,
                true,
            ),
        );
        let identity = match &callable.borrow().call {
            crate::value::Callable::NativeData(native) => native.identity.to_string(),
            _ => panic!("expected data-carrying native"),
        };
        assert_eq!(identity, "registeredOperation");
    }

    #[test]
    fn realm_symbol_caches_credit_storage_and_shared_payload_once() {
        let interp = Interp::new();
        let objects = crate::value::heap_gc_snapshot(&interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&interp.gc_heap);
        let snapshot = measure(&interp, &objects, &scopes);
        let wk_storage =
            interp
                .wk_syms
                .capacity()
                .saturating_mul(size_of::<(&'static str, Value, Rc<str>)>());

        assert!(!interp.wk_syms.is_empty());
        assert!(snapshot.interpreter_side_tables.bytes >= wk_storage);
        assert!(snapshot.strings_symbols_bigints.bytes > 0);
    }

    #[test]
    fn agent_snapshot_aggregates_shadow_heaps_with_global_deduplication() {
        use crate::interpreter::BufferData;
        let mut root = Interp::new();
        let mut shadow = Interp::new_with_symbol_agent(root.symbol_agent.clone());
        let shared_buffer = Rc::new(RefCell::new(Vec::with_capacity(333)));
        root.array_buffers
            .insert(1, BufferData::Local(shared_buffer.clone()));
        shadow
            .array_buffers
            .insert(2, BufferData::Local(shared_buffer));
        let shared_data_block = crate::interpreter::alloc_shared_mem(337).unwrap();
        let shared_id = shared_data_block.id();
        root.array_buffers
            .insert(3, BufferData::Shared(shared_data_block.clone()));
        shadow
            .array_buffers
            .insert(4, BufferData::Shared(shared_data_block));

        let root_objects = crate::value::heap_gc_snapshot(&root.gc_heap);
        let root_scopes = crate::value::gc_scope_snapshot(&root.gc_heap);
        let shadow_objects = crate::value::heap_gc_snapshot(&shadow.gc_heap);
        let shadow_scopes = crate::value::gc_scope_snapshot(&shadow.gc_heap);
        let root_only = measure(&root, &root_objects, &root_scopes);
        let shadow_only = measure(&shadow, &shadow_objects, &shadow_scopes);
        let expected_objects = root_objects.len().saturating_add(shadow_objects.len());
        let expected_scopes = root_scopes.len().saturating_add(shadow_scopes.len());

        root.shadow_realms.insert(7, Box::new(shadow));
        let aggregate = measure(&root, &root_objects, &root_scopes);

        assert_eq!(
            aggregate.object_bodies.bytes,
            expected_objects.saturating_mul(size_of::<RefCell<Object>>())
        );
        assert_eq!(
            aggregate.scope_bodies.bytes,
            expected_scopes.saturating_mul(size_of::<RefCell<Scope>>())
        );
        assert_eq!(aggregate.array_buffer_backing.bytes, 333);
        assert_eq!(aggregate.shared_array_buffer_backing.allocations.len(), 1);
        assert_eq!(
            aggregate.shared_array_buffer_backing.allocations[0].id,
            shared_id
        );
        assert!(aggregate.shared_array_buffer_backing.bytes() >= 337);
        let json = aggregate.json(5, 7);
        assert!(json.contains(&format!(
            "\"allocation_id\":\"shared-data-block:{shared_id}\""
        )));
        assert!(json.contains("\"externally_shared\":true"));
        assert!(
            aggregate.strings_symbols_bigints.bytes
                < root_only
                    .strings_symbols_bigints
                    .bytes
                    .saturating_add(shadow_only.strings_symbols_bigints.bytes),
            "Agent-shared symbol/string payload must be credited once"
        );
        assert!(aggregate.interpreter_side_tables.bytes >= 2 * size_of::<Interp>());
    }

    #[test]
    fn absent_shared_backing_identity_is_unavailable_not_zero() {
        let absent_id = crate::interpreter::next_shared_id();
        let mut visitor = Visitor::default();
        visitor.shared_array_buffer(absent_id);
        let category = SharedBackingCategory {
            allocations: visitor.shared_array_buffer_allocations,
            unavailable_ids: visitor.unavailable_shared_array_buffers,
        };
        assert_eq!(category.unavailable_ids, 1);
        let json = format!("\"shared_array_buffer_backing\":{}", category.json());
        assert!(json.contains(
            "\"shared_array_buffer_backing\":{\"bytes\":null,\"quality\":\"unavailable\""
        ));
        assert!(json.contains("referenced Shared Data Block identities were absent"));
    }

    #[test]
    fn collection_side_tables_account_vectors_indexes_and_values() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);
        engine
            .eval(
                r#"
                    globalThis.strong = new Map([["a", 1], ["b", 2], ["c", 3]]);
                    globalThis.strongIterator = strong.entries();
                    globalThis.weakKey = {};
                    globalThis.weak = new WeakMap([[weakKey, { held: "value" }]]);
                "#,
                false,
            )
            .expect("collections evaluate");
        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &objects, &scopes);

        assert!(!engine.interp.collection_iterators.is_empty());
        assert!(!engine.interp.weak_collection_data.is_empty());
        assert!(!engine.interp.weak_collection_index.is_empty());
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
    }

    #[test]
    fn array_buffer_view_metadata_is_accounted_separately_from_backing() {
        let mut engine = crate::Engine::new();
        engine
            .eval(
                r#"
                    globalThis.buffer = new ArrayBuffer(64, { maxByteLength: 128 });
                    globalThis.typed = new Uint8Array(buffer);
                    globalThis.view = new DataView(buffer, 4, 16);
                    typed[2] = 7;
                "#,
                false,
            )
            .expect("buffer views evaluate");
        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let snapshot = measure(&engine.interp, &objects, &scopes);

        assert!(!engine.interp.array_buffers.is_empty());
        assert!(!engine.interp.typed_arrays.is_empty());
        assert!(!engine.interp.ta_buffer.is_empty());
        assert!(!engine.interp.data_views.is_empty());
        assert!(snapshot.interpreter_side_tables.bytes > size_of::<Interp>());
        assert!(snapshot.array_buffer_backing.bytes >= 64);
    }

    #[test]
    fn reusable_execution_pools_report_retained_capacity() {
        let mut engine = crate::Engine::new();
        engine.set_tier(crate::bytecode::Tier::Bytecode);
        engine.set_tier_threshold(0);
        engine
            .eval(
                "function pooled(a, b) { return a + b; } pooled(20, 22);",
                false,
            )
            .expect("bytecode function evaluates");
        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let snapshot = measure(&engine.interp, &objects, &scopes);
        let stub_bytes = engine
            .interp
            .stub_cache
            .capacity()
            .saturating_mul(size_of::<std::cell::Cell<crate::interpreter::StubEntry>>());

        assert!(!engine.interp.vm_pool.is_empty());
        assert!(snapshot.engine_caches.bytes >= stub_bytes);
    }

    /// A function's deferred prototype counts once, in property storage, as its own
    /// `LazyFunctionPrototype` payload: 48 requested bytes, as before the payload became a
    /// cell (no report counts cells by bytes, so the cell is not counted twice). Measuring
    /// does not materialize it. Of functions made by one expression (their prototype
    /// template's key layout is counted with the template), replacing one's `prototype`, after
    /// a warm-up replacement and with the garbage collected, lowers property storage by exactly
    /// that payload.
    #[test]
    fn deferred_prototype_payload_is_counted_once_in_property_storage() {
        let mut engine = crate::Engine::new();
        let run = |engine: &mut crate::Engine, source: &str| {
            engine
                .eval(source, false)
                .expect("memory fixture evaluates");
            engine.interp.gc_collect();
        };
        let property_storage = |engine: &crate::Engine| {
            let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
            let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
            measure(&engine.interp, &objects, &scopes)
                .property_storage
                .bytes
        };
        run(
            &mut engine,
            "function make() { return function () {}; } \
             var F = make(), G = make(), H = make(); \
             Object.defineProperty(H, 'prototype', { value: null }); 0",
        );
        let materialized = crate::value::prototypes_materialized();
        let before = property_storage(&engine);
        run(
            &mut engine,
            "Object.defineProperty(F, 'prototype', { value: null }); 0",
        );
        let after = property_storage(&engine);
        assert_eq!(crate::value::DEFERRED_PROTOTYPE_BYTES, 48);
        assert_eq!(before - after, crate::value::DEFERRED_PROTOTYPE_BYTES);
        assert_eq!(
            crate::value::prototypes_materialized() - materialized,
            1,
            "only the store materialized (F's prototype, through its descriptor snapshot)"
        );
    }

    #[test]
    fn realm_metadata_accounts_console_and_import_string_capacity() {
        let mut interp = Interp::new();
        let objects = crate::value::heap_gc_snapshot(&interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&interp.gc_heap);
        let before = measure(&interp, &objects, &scopes);
        let mut line = String::with_capacity(257);
        line.push_str("diagnostic console line");
        interp.console.push(line);
        interp.import_base = String::with_capacity(193);
        interp.import_base.push_str("file:///realm/");
        let after = measure(&interp, &objects, &scopes);

        assert!(
            after.interpreter_side_tables.bytes
                >= before
                    .interpreter_side_tables
                    .bytes
                    .saturating_add(257)
                    .saturating_add(193)
        );
    }

    #[test]
    fn module_records_account_ast_exports_and_namespace_indexes() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);
        engine
            .eval_module(
                r#"
                    import { value as dependency } from "./dependency.js";
                    export const answer = dependency + 1;
                "#,
                "file:///main.js",
                |specifier, _| {
                    (specifier == "./dependency.js").then(|| {
                        (
                            "file:///dependency.js".to_string(),
                            "export const value = 41;".to_string(),
                        )
                    })
                },
            )
            .expect("module graph evaluates");
        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &objects, &scopes);

        assert_eq!(engine.interp.module_recs.len(), 2);
        assert!(!engine.interp.module_ns.is_empty());
        assert!(after.function_bytecode_metadata.bytes > before.function_bytecode_metadata.bytes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
    }

    #[test]
    fn promise_reactions_and_queued_jobs_retain_their_storage() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        let source = engine.interp.new_promise();
        let result = engine.interp.new_promise();
        engine
            .interp
            .promise_then_into(&source, Value::Undefined, Value::Undefined, result);
        let pending_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let pending_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let pending = measure(&engine.interp, &pending_objects, &pending_scopes);
        assert_eq!(engine.interp.promises.len(), 2);
        assert_eq!(engine.interp.microtasks.len(), 0);
        assert!(engine
            .interp
            .promises
            .values()
            .any(|state| state.reactions.capacity() > 0));
        assert!(pending.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);

        engine
            .interp
            .resolve_promise(&source, Value::lstr("settled payload"));
        let queued_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let queued_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let queued = measure(&engine.interp, &queued_objects, &queued_scopes);
        assert_eq!(engine.interp.microtasks.len(), 1);
        assert!(queued.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(queued.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
    }

    #[test]
    fn host_settings_global_names_are_identity_deduplicated() {
        let mut engine = crate::Engine::new();
        engine
            .interp
            .global_var_names
            .borrow_mut()
            .insert("rootGlobalWithRetainedCapacity".to_string());
        engine.interp.switch_host_job_context(41);
        engine
            .interp
            .global_var_names
            .borrow_mut()
            .insert("contextGlobalWithRetainedCapacity".to_string());
        let shared_names = engine.interp.global_var_names.clone();
        let saved_names = &engine
            .interp
            .host_settings_states
            .values()
            .next()
            .expect("switched host context is saved")
            .global_var_names;
        assert!(Rc::ptr_eq(&shared_names, saved_names));
        let mut visitor = Visitor::default();
        let (first_names, _) = visitor.global_var_names(&shared_names);
        let (duplicate_names, duplicate_exact) = visitor.global_var_names(saved_names);
        assert!(first_names > 0);
        assert_eq!(duplicate_names, 0);
        assert!(duplicate_exact);
        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);

        let first = measure(&engine.interp, &objects, &scopes);
        let second = measure(&engine.interp, &objects, &scopes);

        assert_eq!(engine.interp.host_settings_states.len(), 1);
        assert_eq!(
            first.interpreter_side_tables.bytes,
            second.interpreter_side_tables.bytes
        );
        assert!(first.interpreter_side_tables.bytes >= size_of::<Interp>());
    }

    #[test]
    fn suspended_coroutines_and_async_requests_retain_their_storage() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine
            .eval(
                r#"
                    globalThis.memoryGenerator = (function* (value) {
                        yield value;
                    })("retained generator payload");
                "#,
                false,
            )
            .expect("generator setup parses");
        assert_eq!(engine.interp.generators.len(), 1);
        assert!(engine
            .interp
            .generators
            .values()
            .all(|coroutine| matches!(coroutine, crate::coroutine::Coroutine::Vm(_))));

        let request_key = *engine
            .interp
            .generators
            .keys()
            .next()
            .expect("generator has an identity");
        engine.interp.async_gen_busy.insert(request_key);
        engine
            .interp
            .async_gen_queue
            .entry(request_key)
            .or_default()
            .push_back((
                Value::lstr("queued request promise"),
                crate::coroutine::Resume::Throw(Value::lstr("queued request signal")),
            ));
        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);

        assert!(after.function_bytecode_metadata.bytes > before.function_bytecode_metadata.bytes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
    }

    #[test]
    fn weak_targets_stay_weak_while_finalization_payloads_are_scanned() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine
            .eval(
                r#"
                    globalThis.memoryWeakTarget = {};
                    globalThis.memoryWeakToken = {};
                    globalThis.memoryWeakRef = new WeakRef(memoryWeakTarget);
                    globalThis.memoryFinalizer = new FinalizationRegistry(function () {});
                    memoryFinalizer.register(
                        memoryWeakTarget,
                        "held finalization payload",
                        memoryWeakToken
                    );
                "#,
                false,
            )
            .expect("weak-owner setup parses");
        assert_eq!(engine.interp.weak_refs.len(), 1);
        assert_eq!(engine.interp.finalization_registries.len(), 1);
        assert_eq!(
            engine
                .interp
                .finalization_registries
                .values()
                .next()
                .expect("registry state exists")
                .cells
                .len(),
            1
        );
        let registry = match engine
            .interp
            .get_member_key(&Value::Obj(engine.interp.global.clone()), "memoryFinalizer")
        {
            Ok(registry) => registry,
            Err(_) => panic!("registry global is readable"),
        };
        engine
            .interp
            .pending_finalization_cleanup
            .push_back(registry);

        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
    }

    #[test]
    fn active_execution_scratch_accounts_nested_buffers_and_values() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine.interp.fn_frames.reserve(3);
        engine.interp.fn_frames.push(crate::interpreter::FnFrame {
            fn_ptr: 0,
            call_site: 0,
            strict: false,
            construct: false,
            extra: Some(Box::new(crate::interpreter::FrameExtra {
                args_obj: Value::lstr("materialized frame arguments"),
                lazy: None,
            })),
        });
        engine.interp.pending_fn_name = Some("pending inferred function name".to_string());
        engine.interp.pending_tail = Some(Box::new((
            Value::lstr("tail callee"),
            Value::lstr("tail receiver"),
            vec![Value::lstr("tail argument")],
        )));
        let mut resources = Vec::with_capacity(4);
        resources.push(crate::interpreter::Disposable {
            value: Value::lstr("disposable value"),
            method: Value::lstr("dispose method"),
            kind_is_async: false,
            method_is_async: false,
        });
        engine.interp.using_stack = Vec::with_capacity(3);
        engine.interp.using_stack.push(resources);
        engine.interp.decorator_initializers.reserve(5);
        engine
            .interp
            .decorator_initializers
            .push(Value::lstr("decorator initializer"));
        engine.interp.new_target = Value::lstr("active new target");
        engine.interp.pending_new_target = Value::lstr("pending new target");

        let shared_values: Rc<[Value]> = vec![Value::lstr("lazy frame argument")].into();
        let mut visitor = Visitor::default();
        assert!(visitor.value_slice(&shared_values) > 0);
        assert_eq!(visitor.value_slice(&shared_values), 0);

        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
    }

    #[test]
    fn object_side_tables_account_keys_vectors_and_payload_values() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);
        let object = engine.interp.global.clone();
        let identity = Gc::as_ptr(&object) as usize;

        engine.interp.gc_pins.insert(identity, object.clone());
        engine.interp.proxies.insert(
            identity,
            (Value::Obj(object.clone()), Value::Obj(object.clone())),
        );
        engine.interp.host_indexed.insert(
            identity,
            crate::interpreter::HostIndexedProperties {
                length: 1,
                getter: Value::lstr("host indexed getter payload"),
                live: None,
            },
        );
        engine
            .interp
            .template_cache
            .insert((identity, 7), Value::lstr("template cache payload"));
        engine
            .interp
            .deferred_ns
            .insert(identity, "deferred/module.js".to_string());
        engine.interp.deferred_ns_objs.insert(
            "deferred/module.js".to_string(),
            Value::lstr("deferred namespace payload"),
        );
        let mut mapped_names = Vec::with_capacity(5);
        mapped_names.push(Some("mappedParameterWithCapacity".to_string()));
        engine
            .interp
            .mapped_arguments
            .insert(identity, (engine.interp.global_env.clone(), mapped_names));
        engine.interp.module_source_objs.insert(
            "source/module.js".to_string(),
            Value::lstr("module source payload"),
        );

        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
    }

    #[test]
    fn class_metadata_routes_initializer_ast_to_bytecode_accounting() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine
            .eval(
                r#"
                    globalThis.MemoryAccountingClass = class {
                        retainedField = "class initializer payload";
                        #retainedMethod() { return this.retainedField; }
                    };
                "#,
                false,
            )
            .expect("class metadata setup parses");
        assert_eq!(engine.interp.class_info.len(), 1);
        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);

        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.function_bytecode_metadata.bytes > before.function_bytecode_metadata.bytes);
    }

    #[test]
    fn additional_realms_account_only_their_direct_metadata() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine
            .eval("globalThis.memoryRealm = $262.createRealm().global;", false)
            .expect("realm setup parses");
        assert_eq!(engine.interp.realms.len(), 2);
        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);

        assert!(after.object_bodies.bytes > before.object_bodies.bytes);
        assert!(after.scope_bodies.bytes > before.scope_bodies.bytes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
    }

    #[test]
    fn temporal_records_account_zone_and_calendar_strings_once() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine
            .eval(
                r#"
                    globalThis.memoryDate = new Temporal.PlainDate(2024, 1, 2, "gregory");
                    globalThis.memoryZoned = new Temporal.ZonedDateTime(0n, "UTC", "iso8601");
                "#,
                false,
            )
            .expect("Temporal setup parses");
        assert_eq!(engine.interp.temporal.len(), 2);
        assert_eq!(engine.interp.temporal_cal.len(), 2);
        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);

        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
    }

    #[test]
    fn async_wait_timer_and_agent_handles_report_visible_storage() {
        let mut engine = crate::Engine::new();
        let before_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let before_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let before = measure(&engine.interp, &before_objects, &before_scopes);

        engine.interp.pending_async_waits.reserve(3);
        engine
            .interp
            .pending_async_waits
            .push(crate::interpreter::AsyncWait {
                promise: Value::lstr("wait promise payload"),
                token: 0,
                id: 0,
                block: crate::interpreter::alloc_shared_mem(8).unwrap(),
                index: 0,
                deadline: None,
            });
        engine.interp.async_wait_inbox = Some(Default::default());
        engine.interp.pending_timers.reserve(4);
        engine.interp.pending_timers.push((
            Value::lstr("timer callback payload"),
            std::time::Instant::now(),
        ));

        let (broadcast_tx, broadcast_rx) = std::sync::mpsc::channel();
        let (report_tx, report_rx) = std::sync::mpsc::channel();
        report_tx
            .send("opaque queued report".to_string())
            .expect("test receiver remains connected");
        engine.interp.agent = Some(Box::new(crate::interpreter::AgentChannels {
            agent_broadcast_txs: vec![broadcast_tx],
            report_rx: Some(report_rx),
            report_tx,
            broadcast_rx: Some(broadcast_rx),
        }));

        let after_objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let after_scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let after = measure(&engine.interp, &after_objects, &after_scopes);
        assert!(after.interpreter_side_tables.bytes > before.interpreter_side_tables.bytes);
        assert!(after.strings_symbols_bigints.bytes > before.strings_symbols_bigints.bytes);
        assert!(matches!(
            after.interpreter_side_tables.quality,
            Quality::LowerBound
        ));
    }

    #[test]
    fn gc_heap_registries_are_identity_deduplicated() {
        let engine = crate::Engine::new();
        let mut visitor = Visitor::default();
        let (first, _) = visitor.gc_heap(&engine.interp.gc_heap);
        let (duplicate, duplicate_exact) = visitor.gc_heap(&engine.interp.gc_heap);

        assert!(first > size_of::<crate::value::GcHeap>());
        assert_eq!(duplicate, 0);
        assert!(duplicate_exact);
    }

    #[test]
    fn aliased_array_buffer_backing_is_counted_once() {
        let buffer = Rc::new(RefCell::new(Vec::with_capacity(257)));
        let aliases = [buffer.clone(), buffer];
        let mut visitor = Visitor::default();
        for buffer in &aliases {
            visitor.array_buffer(buffer);
        }
        assert_eq!(visitor.array_buffer_bytes(), aliases[0].borrow().capacity());
    }

    #[test]
    fn repeated_scan_at_one_safepoint_is_byte_deterministic() {
        let interp = Interp::new();
        let objects = crate::value::heap_gc_snapshot(&interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&interp.gc_heap);
        assert_eq!(
            measure(&interp, &objects, &scopes).json(7, 11),
            measure(&interp, &objects, &scopes).json(7, 11)
        );
    }

    #[test]
    fn independent_agents_cannot_overwrite_each_others_snapshots() {
        let first = Interp::new();
        let first_objects = crate::value::heap_gc_snapshot(&first.gc_heap);
        let first_scopes = crate::value::gc_scope_snapshot(&first.gc_heap);
        record_post_gc(&first, &first_objects, &first_scopes);
        let first_before = json(&first);

        let second = Interp::new();
        let second_objects = crate::value::heap_gc_snapshot(&second.gc_heap);
        let second_scopes = crate::value::gc_scope_snapshot(&second.gc_heap);
        record_post_gc(&second, &second_objects, &second_scopes);

        assert_eq!(json(&first), first_before);
        assert_ne!(
            first.symbol_agent.borrow().agent_id,
            second.symbol_agent.borrow().agent_id
        );
        assert_ne!(
            crate::value::heap_id(&first.gc_heap),
            crate::value::heap_id(&second.gc_heap)
        );
    }

    #[test]
    fn compiled_function_metadata_is_retained_in_its_canonical_categories() {
        let mut engine = crate::Engine::new();
        engine.set_tier(crate::bytecode::Tier::Jit);
        engine.set_tier_threshold(0);
        engine
            .eval(
                "function f(o) { let n = 0; for (let i=0;i<20;i++) n += o.x+i; return n; } globalThis.keep=f; f({x:1});",
                false,
            )
            .expect("metadata fixture parses");
        let objects = crate::value::heap_gc_snapshot(&engine.interp.gc_heap);
        let scopes = crate::value::gc_scope_snapshot(&engine.interp.gc_heap);
        let snapshot = measure(&engine.interp, &objects, &scopes);
        assert!(snapshot.function_bytecode_metadata.bytes > 0);
        assert!(snapshot.callable_metadata.bytes > 0);
    }

    #[test]
    fn recursive_ast_allocations_are_identity_deduplicated() {
        let source = r#"
            function outer({ first: [head = "default", ...tail] }, ...items) {
                class Nested extends Base {
                    ["method"]({ value = 123456789012345678901234567890n }) {
                        return `${value}:${head}`;
                    }
                }
                function inner(arg) { return { [arg]: /x+/gi, ...items }; }
                try {
                    for (let item of items) { if (item) continue; }
                } catch ({ message }) {
                    throw message;
                }
                return [Nested, inner];
            }
        "#;
        let statements = crate::parser::parse_script(source, false)
            .ok()
            .expect("recursive AST parses");
        let function = statements
            .iter()
            .find_map(|statement| match statement {
                crate::ast::Stmt::FuncDecl(function) => Some(function.clone()),
                _ => None,
            })
            .expect("outer function declaration");

        let mut visitor = Visitor::default();
        visitor.function(&function);
        let first_ast_bytes = visitor.function_bytecode_metadata;
        let first_string_bytes = visitor.strings_symbols_bigints;
        visitor.function(&function);

        assert!(first_ast_bytes > size_of::<crate::ast::Function>());
        assert!(first_string_bytes > 0);
        assert_eq!(visitor.function_bytecode_metadata, first_ast_bytes);
        assert_eq!(visitor.strings_symbols_bigints, first_string_bytes);
        assert!(visitor.functions.len() >= 3, "outer, method, and inner");
        assert_eq!(visitor.classes.len(), 1);
        assert_eq!(visitor.bigints.len(), 1);
    }

    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    #[test]
    fn borrowed_osr_cache_metadata_is_counted_once_and_released_with_chunk() {
        let mut engine = crate::Engine::new();
        let body = crate::parser::parse_script("for(let n=0;n<1000;n++){}", false)
            .ok()
            .expect("OSR memory fixture parses");
        let chunk = crate::bytecode::compile_script(&body, false).expect("Script compiles");
        let mut before = Visitor::default();
        before.chunk(&chunk);
        let state = chunk
            .osr
            .get_or_init(|| Box::new(crate::tiering::OsrCodeState::default()));
        let code = state
            .code(&mut engine.interp, &chunk)
            .expect("borrowed code compiles");
        let weak = Rc::downgrade(&code);
        let mut after = Visitor::default();
        after.chunk(&chunk);
        // The reclaimable native slot now owns a lazy state allocation in
        // addition to the OSR sidecar itself. It is not executable/JitCode data.
        assert!(state.retained_metadata_bytes() > size_of::<crate::tiering::OsrCodeState>());
        assert_eq!(
            after.function_bytecode_metadata,
            before.function_bytecode_metadata + state.retained_metadata_bytes()
        );
        assert_eq!(after.jit_codes.len(), 1);
        let retained = after.jit_heap_metadata;
        assert!(retained > 0);
        after.chunk(&chunk);
        after.jit_code(&code);
        assert_eq!(after.jit_heap_metadata, retained);
        drop(code);
        assert!(weak.upgrade().is_some());
        drop(chunk);
        assert!(weak.upgrade().is_none());
    }
}
