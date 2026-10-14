//! Opt-in observations of the actual optimizing path, without detailed-feedback rerouting.
//!
//! ECMA-262 e28783d5 #sec-execution-contexts: telemetry never changes an activation,
//! exposes an object, or keeps a JS/code owner alive. Counters belong to their native
//! code, and the bounded directory holds Weak references only. Disabled emission adds
//! no instructions or execution clocks. Diagnostic runs are never acceptance timing.

use crate::bytecode::{Chunk, Op};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

const MAX_LIVE_RECORDS: usize = 8192;
const LEGACY_COUNTERS: usize = 16;
pub(super) const NORMAL_EXIT: usize = 16;
pub(super) const ERROR_EXIT: usize = 17;
pub(super) const VM_RECOVERY: usize = 18;
pub(super) const VM_RECOVERY_NORMAL: usize = 19;
pub(super) const RETIREMENT_REQUEST: usize = 20;
const SITE_BASE: usize = 21;
const MAX_SITES: usize = 8;
const MAX_OMITTED_PCS: usize = 32;
pub(super) const OWNER_LEAF: usize = SITE_BASE + MAX_SITES;
pub(super) const OWNER_LEAF_GUARDS: usize = OWNER_LEAF + 1;
pub(super) const OBSERVED_REGION_EXIT: usize = OWNER_LEAF_GUARDS + 1;
pub(super) const COUNTERS: usize = OBSERVED_REGION_EXIT + 1;
pub(super) const SHARED_CALL: usize = 10;
pub(super) const RETAIN: usize = 11;
pub(super) const RELEASE: usize = 12;
pub(super) const PUBLICATION: usize = 13;
pub(super) const HEAP_GUARD: usize = 14;
pub(super) const SPECIALIZATION_BAILOUT: usize = 15;
const LABELS: [&str; SITE_BASE] = [
    "entries",
    "ownership_helper",
    "property_read_helper",
    "property_write_helper",
    "element_read_helper",
    "element_write_helper",
    "allocation_helper",
    "call_helper",
    "numeric_helper",
    "other_helper",
    "shared_call_stub",
    "native_retains",
    "native_releases",
    "published_words",
    "heap_guards",
    "specialization_bailouts",
    "normal_exits",
    "error_exits",
    "terminal_vm_recoveries",
    "terminal_vm_normal_completions",
    "specialization_retirement_requests",
];

fn labels() -> impl Iterator<Item = (usize, &'static str)> {
    LABELS.iter().copied().enumerate().chain([
        (OWNER_LEAF, "shared_owner_leaf"),
        (OWNER_LEAF_GUARDS, "owner_leaf_guards"),
        (OBSERVED_REGION_EXIT, "observed_region_vm_exits"),
    ])
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compile,
    Full,
    Execution,
}
static CONFIG: OnceLock<Option<Mode>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static LIVE: RefCell<Vec<Weak<Code>>> = const { RefCell::new(Vec::new()) };
    static DROPPED: Cell<u64> = const { Cell::new(0) };
}

pub(super) fn enabled() -> bool {
    configuration().is_some()
}

#[cfg(test)]
thread_local! { static TEST_MODE: Cell<Option<Option<Mode>>> = const { Cell::new(None) }; }

#[cfg(test)]
pub(super) fn with_test_mode<T>(setting: &str, run: impl FnOnce() -> T) -> T {
    struct Reset(Option<Option<Mode>>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_MODE.with(|mode| mode.set(self.0));
        }
    }
    let mode = match setting {
        "compile" => Some(Mode::Compile),
        "execution" => Some(Mode::Execution),
        "1" => Some(Mode::Full),
        _ => None,
    };
    let _reset = Reset(TEST_MODE.with(|current| current.replace(Some(mode))));
    run()
}

fn configuration() -> Option<Mode> {
    #[cfg(test)]
    if let Some(mode) = TEST_MODE.with(Cell::get) {
        return mode;
    }
    *CONFIG.get_or_init(
        || match std::env::var("TRUST_JS_OPT_JIT_DIAGNOSTICS").as_deref() {
            Ok("1") => Some(Mode::Full),
            Ok("compile") => Some(Mode::Compile),
            Ok("execution") => Some(Mode::Execution),
            _ => None,
        },
    )
}

pub(super) fn category(op: &Op) -> usize {
    match op {
        Op::LoadLocal(_) | Op::StoreLocal(_) | Op::Dup | Op::Dup2 | Op::Pop => 1,
        Op::GetProp(..) | Op::GetPropLocal(..) | Op::GetPropThis(..) | Op::GetMethod(..) => 2,
        Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropLocalDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::AppendProp(..) => 3,
        Op::GetElem | Op::GetElemLocal(_) | Op::GetMethodElem => 4,
        Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(_) | Op::SetElemLocalDrop(_) => 5,
        Op::MakeObject(..) | Op::MakeArray(..) | Op::NewObject | Op::MakeClosure(..) => 6,
        Op::Call(..)
        | Op::CallWithThis(..)
        | Op::CallSpread(..)
        | Op::CallSpreadThis(..)
        | Op::TailCall(..)
        | Op::New(..) => 7,
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Mod
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::EqEq
        | Op::NotEq
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::UShr
        | Op::Neg
        | Op::Plus
        | Op::BitNot
        | Op::UpdateLocal(..) => 8,
        _ => 9,
    }
}

/// Owns stable counter addresses embedded in the executable mapping. The referring JitCode
/// owns this Rc until its mapping is unmapped; Weak records never pin executable or JS state.
struct Site {
    pc: usize,
    position: u32,
    op: String,
    key: String,
    key_truncated: bool,
    kind: &'static str,
}

pub(super) struct Code {
    id: u64,
    chunk: usize,
    created: Instant,
    ops: usize,
    pub runtime_counters: bool,
    pub execution_counters: bool,
    source_origin: (u32, u32),
    slots: Vec<String>,
    slot_truncated: Vec<bool>,
    continuation: Cell<bool>,
    region_source_version: Cell<u64>,
    region_native_ops: Cell<usize>,
    region_cut_blocks: Cell<usize>,
    sites: RefCell<Vec<Site>>,
    omitted_pcs: RefCell<Vec<usize>>,
    omitted_sites: Cell<usize>,
    pub counters: [Cell<u64>; COUNTERS],
    pub emitted_helpers: Cell<u64>,
    pub emitted_owner_leaf_calls: Cell<u64>,
    pub emitted_read_guard_calls: Cell<u64>,
    pub emitted_publications: Cell<u64>,
    pub emitted_guards: Cell<u64>,
    pub ir_instructions: Cell<usize>,
    pub ir_blocks: Cell<usize>,
    pub native_bytes: Cell<usize>,
    pub analysis_ns: Cell<u64>,
    pub lowering_ns: Cell<u64>,
    pub codegen_ns: Cell<u64>,
    pub finish_ns: Cell<u64>,
}

impl Code {
    pub(super) fn id(&self) -> u64 {
        self.id
    }
    pub(super) fn new(chunk: &Chunk) -> Option<Rc<Self>> {
        let mode = configuration()?;
        LIVE.with(|live| {
            let mut live = live.borrow_mut();
            if live.len() == MAX_LIVE_RECORDS {
                live.retain(|record| record.strong_count() != 0);
            }
            if live.len() == MAX_LIVE_RECORDS {
                DROPPED.with(|count| count.set(count.get().saturating_add(1)));
                return None;
            }
            let result = Rc::new(Self {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                chunk: chunk as *const Chunk as usize,
                created: Instant::now(),
                ops: chunk.jit_ops().len(),
                runtime_counters: mode == Mode::Full,
                execution_counters: mode == Mode::Execution,
                source_origin: chunk.debug_origin,
                slots: chunk
                    .jit_slot_names()
                    .iter()
                    .take(4)
                    .map(|name| name.chars().take(32).collect())
                    .collect(),
                slot_truncated: chunk
                    .jit_slot_names()
                    .iter()
                    .take(4)
                    .map(|name| name.chars().count() > 32)
                    .collect(),
                continuation: Cell::new(false),
                region_source_version: Cell::new(0),
                region_native_ops: Cell::new(chunk.jit_ops().len()),
                region_cut_blocks: Cell::new(0),
                sites: RefCell::new(Vec::new()),
                omitted_pcs: RefCell::new(Vec::new()),
                omitted_sites: Cell::new(0),
                counters: std::array::from_fn(|_| Cell::new(0)),
                emitted_helpers: Cell::new(0),
                emitted_owner_leaf_calls: Cell::new(0),
                emitted_read_guard_calls: Cell::new(0),
                emitted_publications: Cell::new(0),
                emitted_guards: Cell::new(0),
                ir_instructions: Cell::new(0),
                ir_blocks: Cell::new(0),
                native_bytes: Cell::new(0),
                analysis_ns: Cell::new(0),
                lowering_ns: Cell::new(0),
                codegen_ns: Cell::new(0),
                finish_ns: Cell::new(0),
            });
            live.push(Rc::downgrade(&result));
            Some(result)
        })
    }

    pub(super) fn set_continuation(&self, continuation: bool) {
        self.continuation.set(continuation);
    }

    pub(super) fn set_regions(&self, source_version: u64, native_ops: usize, cut_blocks: usize) {
        self.region_source_version.set(source_version);
        self.region_native_ops.set(native_ops);
        self.region_cut_blocks.set(cut_blocks);
    }

    pub(super) fn counts(&self, index: usize) -> bool {
        (self.runtime_counters
            && (index < LEGACY_COUNTERS
                || matches!(index, OWNER_LEAF | OWNER_LEAF_GUARDS | OBSERVED_REGION_EXIT)))
            || (self.execution_counters
                && (matches!(
                    index,
                    0 | SPECIALIZATION_BAILOUT
                        | NORMAL_EXIT
                        | ERROR_EXIT
                        | VM_RECOVERY
                        | VM_RECOVERY_NORMAL
                        | RETIREMENT_REQUEST
                        | OBSERVED_REGION_EXIT
                ) || (SITE_BASE..SITE_BASE + MAX_SITES).contains(&index)))
    }

    /// Compiler-only selection. Runtime sites embed stable Cell addresses, never a lookup.
    pub(super) fn select_site(
        &self,
        chunk: &Chunk,
        pc: usize,
        kind: &'static str,
    ) -> Option<usize> {
        if !self.execution_counters {
            return None;
        }
        let mut sites = self.sites.borrow_mut();
        if let Some(index) = sites.iter().position(|site| site.pc == pc) {
            return Some(SITE_BASE + index);
        }
        if sites.len() == MAX_SITES {
            self.omitted_sites
                .set(self.omitted_sites.get().saturating_add(1));
            let mut omitted = self.omitted_pcs.borrow_mut();
            if omitted.len() < MAX_OMITTED_PCS {
                omitted.push(pc);
            }
            return None;
        }
        let op = &chunk.jit_ops()[pc];
        let name = match op {
            Op::GetProp(name, _)
            | Op::GetMethod(name, _)
            | Op::GetPropLocal(_, name, _)
            | Op::GetPropThis(name, _)
            | Op::SetProp(name, _)
            | Op::SetPropDrop(name, _)
            | Op::SetPropLocalDrop(_, name, _)
            | Op::SetPropThisDrop(name, _) => Some(*name),
            _ => None,
        };
        sites.push(Site {
            pc,
            position: chunk.position(pc),
            op: format!("{op:?}"),
            key: name.map_or_else(String::new, |name| {
                chunk.jit_name(name).chars().take(64).collect()
            }),
            key_truncated: name.is_some_and(|name| chunk.jit_name(name).chars().count() > 64),
            kind,
        });
        Some(SITE_BASE + sites.len() - 1)
    }

    #[cfg(test)]
    pub(super) fn site_stats(&self) -> Vec<(usize, &'static str, u64)> {
        self.sites
            .borrow()
            .iter()
            .enumerate()
            .map(|(index, site)| (site.pc, site.kind, self.counters[SITE_BASE + index].get()))
            .collect()
    }

    /// Heap-requested payloads owned by this record; allocator rounding is excluded.
    /// Chunk/cache/JS roots are borrowed and are never included here.
    pub(super) fn retained_heap_metadata_bytes(&self) -> usize {
        let sites = self.sites.borrow();
        let mut bytes =
            std::mem::size_of::<Self>().saturating_add(2 * std::mem::size_of::<usize>());
        bytes = bytes.saturating_add(
            self.slots
                .capacity()
                .saturating_mul(std::mem::size_of::<String>()),
        );
        for slot in &self.slots {
            bytes = bytes.saturating_add(slot.capacity());
        }
        bytes = bytes.saturating_add(
            self.slot_truncated
                .capacity()
                .saturating_mul(std::mem::size_of::<bool>()),
        );
        bytes = bytes.saturating_add(sites.capacity().saturating_mul(std::mem::size_of::<Site>()));
        for site in sites.iter() {
            bytes = bytes
                .saturating_add(site.op.capacity())
                .saturating_add(site.key.capacity());
        }
        bytes.saturating_add(
            self.omitted_pcs
                .borrow()
                .capacity()
                .saturating_mul(std::mem::size_of::<usize>()),
        )
    }

    pub(super) fn counter(&self, index: usize) -> *const Cell<u64> {
        &self.counters[index]
    }

    pub(super) fn json(&self) -> String {
        use std::fmt::Write;
        let mut result = format!(
            "{{\"id\":{},\"chunk\":{},\"ops\":{},\"age_ns\":{},\"analysis_ns\":{},\"lowering_ns\":{},\"codegen_ns\":{},\"finish_ns\":{},\"ir_instructions\":{},\"ir_blocks\":{},\"native_bytes\":{},\"emitted_helpers\":{},\"emitted_owner_leaf_calls\":{},\"emitted_read_guard_calls\":{},\"emitted_publication_words\":{},\"emitted_heap_guards\":{},\"observed_regions\":{{\"source_version\":{},\"native_selected_ops\":{},\"vm_cut_blocks\":{}}},\"counters\":{{",
            self.id, self.chunk, self.ops, nanos(self.created), self.analysis_ns.get(),
            self.lowering_ns.get(), self.codegen_ns.get(), self.finish_ns.get(),
            self.ir_instructions.get(), self.ir_blocks.get(), self.native_bytes.get(),
            self.emitted_helpers.get(), self.emitted_owner_leaf_calls.get(), self.emitted_read_guard_calls.get(), self.emitted_publications.get(), self.emitted_guards.get(),
            self.region_source_version.get(), self.region_native_ops.get(), self.region_cut_blocks.get(),
        );
        for (index, label) in labels() {
            if index != 0 {
                result.push(',');
            }
            let _ = write!(result, "\"{label}\":{}", self.counters[index].get());
        }
        let mode = if self.execution_counters {
            "execution"
        } else if self.runtime_counters {
            "full"
        } else {
            "compile"
        };
        let quote = |text: &str| {
            let mut out = String::from("\"");
            super::json_escape_into(&mut out, text);
            out.push('"');
            out
        };
        let tracked = labels()
            .filter(|(index, _)| self.counts(*index))
            .map(|(_, label)| quote(label))
            .collect::<Vec<_>>()
            .join(",");
        let saturated = labels()
            .filter(|(index, _)| self.counts(*index) && self.counters[*index].get() == u64::MAX)
            .map(|(_, label)| quote(label))
            .collect::<Vec<_>>()
            .join(",");
        let sites = self.sites.borrow();
        let rows = sites.iter().enumerate().map(|(index, site)| format!("{{\"pc\":{},\"position_token\":{},\"source_byte_offset\":{},\"op\":{},\"key\":{},\"key_truncated\":{},\"kind\":{},\"native_fast_successes\":{},\"saturated\":{}}}", site.pc, site.position, if site.position == crate::stack_trace::NO_POSITION || crate::stack_trace::is_inline_position(site.position) {"null".into()} else {site.position.to_string()}, quote(&site.op), quote(&site.key), site.key_truncated, quote(site.kind), self.counters[SITE_BASE+index].get(), self.counters[SITE_BASE+index].get()==u64::MAX)).collect::<Vec<_>>().join(",");
        let _ = write!(result, "}},\"runtime_counters\":{},\"counter_mode\":{},\"tracked_counter_labels\":[{}],\"saturated_counter_labels\":[{}],\"source_start\":{},\"source_bytes\":{},\"slots\":[{}],\"slot_label_limit_chars\":32,\"slot_label_count_limit\":4,\"slot_labels_truncated\":{:?},\"key_label_limit_chars\":64,\"entry_kind\":{},\"execution_scope\":\"physical optimizing-native entries; normal exits include terminal VM completions; native-control-flow normal completions = normal_exits - terminal_vm_normal_completions, except saturated counters; checked helpers and author calls may execute within native control flow\",\"execution_counters_saturate\":{},\"sites\":[{}],\"site_selection\":\"first eight emitted committed data-read/existing-property-overwrite fast-success PCs; creation, loop-scalar field operations, element accesses and other operation families excluded; no estimate for omitted sites\",\"site_limit\":{},\"omitted_site_count\":{},\"omitted_pcs\":{:?},\"omitted_pc_list_truncated\":{}}}", self.runtime_counters || self.execution_counters, quote(mode), tracked, saturated, self.source_origin.0, self.source_origin.1, self.slots.iter().map(|slot|quote(slot)).collect::<Vec<_>>().join(","), self.slot_truncated, quote(if self.continuation.get() {"loop_continuation"} else {"whole_function"}), self.execution_counters, rows, MAX_SITES, self.omitted_sites.get(), self.omitted_pcs.borrow(), self.omitted_sites.get()>self.omitted_pcs.borrow().len());
        result
    }

    pub(super) fn compiled(&self) {
        eprintln!(
            "[optimizing-diagnostic] {{\"event\":\"compiled\",\"code\":{}}}",
            self.json()
        );
    }
}

impl Drop for Code {
    fn drop(&mut self) {
        eprintln!(
            "[optimizing-diagnostic] {{\"event\":\"retired\",\"code\":{}}}",
            self.json()
        );
    }
}

pub(super) fn nanos(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

pub(super) fn snapshot_json() -> String {
    if !enabled() {
        return String::from("null");
    }
    LIVE.with(|live| {
        let records = live
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .map(|code| code.json())
            .collect::<Vec<_>>()
            .join(",");
        let dropped = DROPPED.with(Cell::get);
        format!("{{\"dropped_records\":{dropped},\"live\":[{records}]}}")
    })
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use crate::{bytecode::Tier, value::Callable, Engine};

    #[test]
    fn retained_metadata_counts_capacity_and_releases_string_payloads() {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine
            .eval("function subject(o){return o.x;}subject({x:1})", false)
            .unwrap();
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let chunk = {
            let object = value.as_obj().unwrap().borrow();
            let Callable::User(user) = &object.call else {
                panic!("user function")
            };
            user.func.code.get().unwrap().as_ref().unwrap().clone()
        };
        let record = with_test_mode("execution", || Code::new(&chunk).unwrap());
        let mut record = Rc::try_unwrap(record).ok().unwrap();
        assert!(!record.counts(1), "execution mode excludes owner helpers");
        assert!(record.counts(SITE_BASE + MAX_SITES - 1));
        assert!(!record.counts(OWNER_LEAF));
        assert!(!record.counts(OWNER_LEAF_GUARDS));
        for (mode, expected) in [("compile", false), ("1", true)] {
            let owner_record = with_test_mode(mode, || Code::new(&chunk).unwrap());
            assert_eq!(owner_record.counts(OWNER_LEAF), expected);
            assert_eq!(owner_record.counts(OWNER_LEAF_GUARDS), expected);
            assert!(!owner_record.counts(SITE_BASE));
            assert!(!owner_record.counts(COUNTERS));
        }
        assert!(
            !record.counts(COUNTERS),
            "future counters require explicit admission"
        );
        record.slots.clear();
        record.slots.shrink_to_fit();
        record.slot_truncated.clear();
        record.slot_truncated.shrink_to_fit();
        let base = std::mem::size_of::<Code>() + 2 * std::mem::size_of::<usize>();
        assert_eq!(record.retained_heap_metadata_bytes(), base);

        // Reserved-but-unused backing storage is retained too. Payload strings use
        // deliberately larger capacities than lengths, independently of labels' bounds.
        record.slots.reserve_exact(7);
        let mut slot = String::from("局部");
        slot.reserve_exact(91);
        let slot_capacity = slot.capacity();
        record.slots.push(slot);
        record.slot_truncated.reserve_exact(5);
        record.slot_truncated.push(false);
        let mut op = String::from("GetMethod");
        op.reserve_exact(113);
        let mut key = String::from("界");
        key.reserve_exact(127);
        let site_strings = op.capacity() + key.capacity();
        record.sites.get_mut().reserve_exact(13);
        record.sites.get_mut().push(Site {
            pc: 1,
            position: crate::stack_trace::NO_POSITION,
            op,
            key,
            key_truncated: false,
            kind: "data_property_read",
        });
        record.omitted_pcs.get_mut().reserve_exact(23);
        let backing = record.slots.capacity() * std::mem::size_of::<String>()
            + record.slot_truncated.capacity() * std::mem::size_of::<bool>()
            + record.sites.get_mut().capacity() * std::mem::size_of::<Site>()
            + record.omitted_pcs.get_mut().capacity() * std::mem::size_of::<usize>();
        let chunk_owners = Rc::strong_count(&chunk);
        assert_eq!(
            record.retained_heap_metadata_bytes(),
            base + backing + slot_capacity + site_strings
        );
        assert_eq!(Rc::strong_count(&chunk), chunk_owners);
        record.slots.clear();
        record.slot_truncated.clear();
        record.sites.get_mut().clear();
        assert_eq!(record.retained_heap_metadata_bytes(), base + backing);
        record.slots.shrink_to_fit();
        record.slot_truncated.shrink_to_fit();
        record.sites.get_mut().shrink_to_fit();
        record.omitted_pcs.get_mut().shrink_to_fit();
        assert_eq!(record.retained_heap_metadata_bytes(), base);
    }
}

// Cranelift's default features already enable its pass-timing API. Override its thread-local
// profiler only for this compilation and restore it on every exit, including failed codegen.
// This does not clear or absorb the host's Wasm compiler timings.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
pub(super) mod passes {
    use super::*;
    use cranelift_codegen::timing::{self, Pass, Profiler, NUM_PASSES};
    use std::any::Any;

    #[derive(Default, Clone, Copy)]
    struct Cost {
        inclusive: u64,
        exclusive: u64,
        calls: u64,
    }
    struct State {
        costs: [Cost; NUM_PASSES],
        children: Vec<u64>,
    }
    struct Recorder(Rc<RefCell<State>>);
    struct Token {
        state: Rc<RefCell<State>>,
        pass: usize,
        started: Instant,
    }

    impl Profiler for Recorder {
        fn start_pass(&self, pass: Pass) -> Box<dyn Any> {
            self.0.borrow_mut().children.push(0);
            Box::new(Token {
                state: self.0.clone(),
                pass: pass as usize,
                started: Instant::now(),
            })
        }
    }
    impl Drop for Token {
        fn drop(&mut self) {
            let elapsed = nanos(self.started);
            let mut state = self.state.borrow_mut();
            let children = state.children.pop().expect("Cranelift pass nesting");
            let cost = &mut state.costs[self.pass];
            cost.inclusive = cost.inclusive.saturating_add(elapsed);
            cost.exclusive = cost
                .exclusive
                .saturating_add(elapsed.saturating_sub(children));
            cost.calls = cost.calls.saturating_add(1);
            if let Some(parent) = state.children.last_mut() {
                *parent = parent.saturating_add(elapsed);
            }
        }
    }

    pub(in crate::jit) struct Scope {
        previous: Option<Box<dyn Profiler>>,
        state: Rc<RefCell<State>>,
        id: u64,
    }
    impl Scope {
        pub(in crate::jit) fn start(code: &Code) -> Self {
            Self::start_id(code.id)
        }
        pub(in crate::jit) fn start_id(id: u64) -> Self {
            let state = Rc::new(RefCell::new(State {
                costs: [Cost::default(); NUM_PASSES],
                children: Vec::new(),
            }));
            let previous = timing::set_thread_profiler(Box::new(Recorder(state.clone())));
            Self {
                previous: Some(previous),
                state,
                id,
            }
        }
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            let _ =
                timing::set_thread_profiler(self.previous.take().expect("profiler restored once"));
            let state = self.state.borrow();
            // Numeric pass ids are the pinned Cranelift 0.135.2 timing::Pass discriminants.
            // The manifest records this version; no unstable enum layout crosses a native ABI.
            let rows = state
                .costs
                .iter()
                .enumerate()
                .filter(|(_, cost)| cost.calls != 0)
                .map(|(pass, c)| {
                    format!(
                        "{{\"pass\":{pass},\"inclusive_ns\":{},\"exclusive_ns\":{},\"calls\":{}}}",
                        c.inclusive, c.exclusive, c.calls
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            eprintln!("[optimizing-diagnostic] {{\"event\":\"compiler_passes\",\"id\":{},\"passes\":[{rows}]}}", self.id);
        }
    }
}
