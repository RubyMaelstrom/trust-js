//! Tree-walking interpreter: lexical environments, the prototype-based object model, and the
//! ECMAScript abstract operations (ToNumber/ToString/ToBoolean/ToPrimitive, equality, etc.).
//!
//! Control flow uses [`Abrupt`] threaded through `Result`: expressions can only ever raise
//! `Throw`, while statements additionally produce `Return`/`Break`/`Continue` completions.

use crate::ast::*;
use crate::builtins::TaAccessor;
use crate::value::ValueView;
use crate::value::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[path = "borrowed_parameters.rs"]
pub(crate) mod borrowed_parameters;

/// Dirty metadata is a host synchronization aid, not part of the ECMAScript Data Block.
/// SetValueInBuffer must store the bytes immediately (ECMA-262, local snapshot
/// e28783d5fc9d, https://tc39.es/ecma262/#sec-setvalueinbuffer); retaining every distinct byte
/// range is unnecessary. Bound fragmentation so emulator heaps cannot turn every store into
/// a scan/memmove across hundreds of thousands of ranges. Precision resets on the next take.
const MAX_BUFFER_DIRTY_RANGES: usize = 64;

pub(crate) fn record_buffer_write(
    ranges: &mut Vec<Range<usize>>,
    mut write: Range<usize>,
    buffer_len: usize,
) {
    let first = ranges.partition_point(|range| range.end < write.start);
    if let Some(range) = ranges.get(first) {
        if range.start <= write.start && write.end <= range.end {
            return; // Already covered: do not remove/reinsert and shift the vector twice.
        }
    }
    let mut last = first;
    while last < ranges.len() && ranges[last].start <= write.end {
        write.start = write.start.min(ranges[last].start);
        write.end = write.end.max(ranges[last].end);
        last += 1;
    }
    if first == last {
        if ranges.len() == MAX_BUFFER_DIRTY_RANGES {
            ranges.clear();
            ranges.push(0..buffer_len);
        } else {
            ranges.insert(first, write);
        }
    } else {
        ranges[first] = write;
        ranges.drain(first + 1..last);
    }
}

#[cfg(test)]
mod buffer_dirty_range_tests {
    use super::{record_buffer_write, MAX_BUFFER_DIRTY_RANGES};

    #[test]
    fn writes_coalesce_overlap_adjacency_and_bridges() {
        let mut ranges = vec![4..8, 12..16, 20..24];
        record_buffer_write(&mut ranges, 5..7, 32);
        assert_eq!(ranges, [4..8, 12..16, 20..24]);
        record_buffer_write(&mut ranges, 8..20, 32);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 4..24);
        record_buffer_write(&mut ranges, 0..4, 32);
        record_buffer_write(&mut ranges, 24..32, 32);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..32);
    }

    #[test]
    fn fragmented_writes_are_bounded_and_never_lost() {
        let mut ranges = Vec::new();
        let mut written = [false; 1024];
        for index in 0..1024 {
            // An odd multiplier permutes the buffer: scattered insertions and rewrites.
            let start = (index * 397) % written.len();
            written[start] = true;
            record_buffer_write(&mut ranges, start..start + 1, written.len());
            assert!(ranges.len() <= MAX_BUFFER_DIRTY_RANGES);
            assert!(ranges.windows(2).all(|pair| pair[0].end < pair[1].start));
            for (byte, &dirty) in written.iter().enumerate() {
                assert!(!dirty || ranges.iter().any(|range| range.contains(&byte)));
            }
        }
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..1024);
        ranges.clear(); // Taking the dirty set restores exact tracking for the next batch.
        record_buffer_write(&mut ranges, 100..104, 1024);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 100..104);
    }
}

#[derive(Default)]
struct FunctionProfileEntry {
    description: String,
    execution_state: String,
    source: Option<Rc<str>>,
    calls: u64,
    inclusive_ns: u128,
    self_ns: u128,
    callers: crate::fasthash::FastMap<(usize, usize), u64>,
}

struct FunctionProfileFrame {
    key: (usize, usize),
    started: std::time::Instant,
    child_ns: u128,
}

fn function_profile_description(
    key: (usize, usize),
    name: Option<&str>,
    source: Option<&str>,
    include_source: bool,
) -> String {
    let name = name.unwrap_or("<anon>");
    let mut description = format!("[realm={:#x} function={:#x}] {name}", key.1, key.0);
    // Profiling is metadata-only unless source output is explicitly requested. In
    // particular, the ordinary one-line description must not leak a source prefix
    // while the separately gated full-source record is disabled.
    if include_source {
        description.push_str(": ");
        description.extend(source.unwrap_or("").chars().take(100).map(|c| {
            if c == '\n' || c == '\r' {
                ' '
            } else {
                c
            }
        }));
    }
    description
}

#[cfg(test)]
mod function_profile_tests {
    use super::function_profile_description;

    #[test]
    fn source_output_is_opt_in_including_description_prefixes() {
        let source = "function measured() { return 'private-payload'; }";
        let metadata = function_profile_description((1, 2), Some("measured"), Some(source), false);
        assert_eq!(metadata, "[realm=0x2 function=0x1] measured");
        assert!(!metadata.contains("private-payload"));
        let detailed = function_profile_description((1, 2), Some("measured"), Some(source), true);
        assert!(detailed.ends_with(source));
        let source = "é\r\n".repeat(1000);
        let detailed = function_profile_description((3, 2), None, Some(&source), true);
        let prefix = "[realm=0x2 function=0x3] <anon>: ";
        assert!(detailed.starts_with(prefix));
        assert_eq!(detailed[prefix.len()..].chars().count(), 100);
        assert!(!detailed.contains(['\r', '\n']));
    }
}

#[derive(Default)]
struct FunctionProfileState {
    entries: crate::fasthash::FastMap<(usize, usize), FunctionProfileEntry>,
    stack: Vec<FunctionProfileFrame>,
}

impl FunctionProfileState {
    fn dump(&mut self) {
        let mut entries = self.entries.values().collect::<Vec<_>>();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.self_ns));
        for entry in entries.into_iter().take(32) {
            let (caller_calls, caller) = entry
                .callers
                .iter()
                .max_by_key(|(_, calls)| *calls)
                .map(|(key, calls)| {
                    (
                        *calls,
                        self.entries
                            .get(key)
                            .map(|entry| entry.description.as_str())
                            .unwrap_or("<native or previous profile batch>"),
                    )
                })
                .unwrap_or((0, "<root>"));
            eprintln!(
                "[fn-profile] self={:.3}s inclusive={:.3}s calls={} {} top_caller_calls={} top_caller={}",
                entry.self_ns as f64 / 1_000_000_000.0,
                entry.inclusive_ns as f64 / 1_000_000_000.0,
                entry.calls,
                entry.description,
                caller_calls,
                caller,
            );
            eprintln!(
                "[fn-profile-state] description={:?} {}",
                entry.description, entry.execution_state,
            );
            if let Some(source) = &entry.source {
                eprintln!(
                    "[fn-profile-source] description={:?} source={:?}",
                    entry.description,
                    source.chars().take(16_384).collect::<String>()
                );
            }
        }
        self.entries.clear();
    }
}

impl Drop for FunctionProfileState {
    fn drop(&mut self) {
        self.dump();
    }
}

thread_local! {
    static FUNCTION_PROFILE: RefCell<FunctionProfileState> = RefCell::new(Default::default());
    static MISSING_PROPERTY_TRACE: RefCell<std::collections::VecDeque<String>> =
        const { RefCell::new(std::collections::VecDeque::new()) };
    static BOUND_CALL_TRACE: RefCell<BoundCallTrace> = RefCell::new(BoundCallTrace::default());
}

#[derive(Default)]
struct BoundCallTrace {
    next_id: u64,
    stack: Vec<u64>,
    history: std::collections::VecDeque<String>,
    arrays: std::collections::VecDeque<(usize, Vec<String>)>,
}

impl BoundCallTrace {
    fn record(&mut self, entry: String) {
        if self.history.len() == 256 {
            self.history.pop_front();
        }
        self.history.push_back(entry);
    }
}

// Native descriptors only: diagnostics must not call an author's getter, Proxy
// trap or conversion method, or keep a returned object alive after its call.
fn bound_trace_preview(value: &Value, depth: usize) -> String {
    match &(value).view() {
        ValueView::Str(text) => format!(
            "string(bytes={},prefix={:?})",
            text.text().len(),
            text.text().chars().take(48).collect::<String>()
        ),
        ValueView::Num(number) => format!("{number}"),
        ValueView::Bool(boolean) => format!("{boolean}"),
        ValueView::Obj(object) => {
            let id = Gc::as_ptr(object) as usize;
            let Ok(object) = object.try_borrow() else {
                return format!("object@{id:x}(borrowed)");
            };
            let name = match object.props.get("name") {
                Some(property) if !property.accessor() => match (property.value()).view() {
                    ValueView::Str(name) => name.text().chars().take(48).collect::<String>(),
                    _ => String::new(),
                },
                _ => String::new(),
            };
            if let Callable::Bound(bound) = &object.call {
                let metadata = if depth < 2 {
                    bound
                        .args
                        .iter()
                        .take(2)
                        .map(|arg| bound_trace_preview(arg, depth + 1))
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                return format!("bound@{id:x}({name:?},metadata={metadata:?})");
            }
            if !matches!(object.call, Callable::None) {
                return format!("function@{id:x}({name:?})");
            }
            if matches!(object.exotic, Exotic::Array) && depth < 2 {
                let items = (0..8)
                    .map(|index| match object.props.get(&index.to_string()) {
                        Some(property) if !property.accessor() => {
                            bound_trace_preview(&property.value(), depth + 1)
                        }
                        Some(_) => "<accessor>".to_string(),
                        None => "<hole>".to_string(),
                    })
                    .collect::<Vec<_>>();
                return format!("array@{id:x}{items:?}");
            }
            format!("object@{id:x}")
        }
        _ => type_name(value).to_string(),
    }
}

fn trace_bound_call_enter(bound: &BoundCallable, args: &[Value]) -> Option<u64> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("TRUST_JS_TRACE_BOUND_CALLS").is_some()) {
        return None;
    }
    let target = bound_trace_preview(&Value::Obj(bound.target), 0);
    let saved = bound
        .args
        .iter()
        .take(4)
        .map(|arg| bound_trace_preview(arg, 0))
        .collect::<Vec<_>>();
    let args = args
        .iter()
        .take(8)
        .map(|arg| bound_trace_preview(arg, 0))
        .collect::<Vec<_>>();
    Some(BOUND_CALL_TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        trace.next_id = trace.next_id.wrapping_add(1);
        let id = trace.next_id;
        let parent = trace.stack.last().copied();
        trace.stack.push(id);
        trace.record(format!(
            "enter id={id} parent={parent:?} target={target} saved={saved:?} args={args:?}"
        ));
        id
    }))
}

fn trace_bound_call_exit(id: u64, result: &Result<Value, Abrupt>) {
    let result = match result {
        Ok(value) => format!("return {}", bound_trace_preview(value, 0)),
        Err(Abrupt::Throw(value)) => format!("throw {}", bound_trace_preview(value, 0)),
        Err(_) => "abrupt completion".to_string(),
    };
    BOUND_CALL_TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        trace.stack.pop();
        trace.record(format!("exit id={id} {result}"));
    });
}

fn bound_trace_arrays(value: &Value) -> Vec<(usize, String, Vec<String>)> {
    let ValueView::Obj(owner) = &(value).view() else {
        return Vec::new();
    };
    let Ok(owner) = owner.try_borrow() else {
        return Vec::new();
    };
    let mut arrays = Vec::new();
    for (key, property) in owner.props.iter().take(16) {
        if property.accessor() {
            continue;
        }
        let ValueView::Obj(array) = (property.value()).view() else {
            continue;
        };
        let id = Gc::as_ptr(&array) as usize;
        let Ok(array) = array.try_borrow() else {
            continue;
        };
        if !matches!(array.exotic, Exotic::Array) {
            continue;
        }
        let Some(length) = array.props.get("length").filter(|p| !p.accessor()) else {
            continue;
        };
        let ValueView::Num(length) = (length.value()).view() else {
            continue;
        };
        if !(16.0..=512.0).contains(&length) {
            continue;
        }
        let values = (0..length as usize)
            .map(|index| match array.props.get(&index.to_string()) {
                Some(property) if !property.accessor() => bound_trace_preview(&property.value(), 0),
                Some(_) => "<accessor>".to_string(),
                None => "<hole>".to_string(),
            })
            .collect();
        arrays.push((id, key.text().chars().take(48).collect(), values));
        if arrays.len() == 4 {
            break;
        }
    }
    arrays
}

fn trace_bound_array_state(call_id: u64, phase: &str, value: Option<&Value>) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("TRUST_JS_TRACE_BOUND_ARRAYS").is_some()) {
        return;
    }
    let Some(value) = value else {
        return;
    };
    let arrays = bound_trace_arrays(value);
    BOUND_CALL_TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        for (array_id, key, values) in arrays {
            let old = trace
                .arrays
                .iter()
                .position(|(id, _)| *id == array_id)
                .and_then(|index| trace.arrays.remove(index));
            let changes = values
                .iter()
                .enumerate()
                .filter(|(index, value)| {
                    old.as_ref().and_then(|(_, old)| old.get(*index)) != Some(*value)
                })
                .map(|(index, value)| (index, value.as_str()))
                .collect::<Vec<_>>();
            if !changes.is_empty() {
                trace.record(format!(
                    "array id={call_id} phase={phase} object={array_id:x} field={key:?} changes={changes:?}"
                ));
            }
            if trace.arrays.len() == 16 {
                trace.arrays.pop_front();
            }
            trace.arrays.push_back((array_id, values));
        }
    });
}

#[cfg(test)]
mod bound_trace_tests {
    use super::*;

    /// A preview names an accessor without reading it, notes a borrowed object instead of
    /// borrowing it, and stays short for a long string.
    #[test]
    fn native_previews_are_bounded_and_skip_accessors() {
        let object = Object::new(None);
        object.borrow_mut().exotic = Exotic::Array;
        object
            .borrow_mut()
            .props
            .insert("0", Property::accessor_prop(None, None, true, true));
        let value = Value::Obj(object);
        let preview = bound_trace_preview(&value, 0);
        assert!(preview.contains("<accessor>"));
        let borrowed = object.borrow_mut();
        assert!(bound_trace_preview(&value, 0).contains("(borrowed)"));
        drop(borrowed);
        let text = Value::lstr("é".repeat(100_000));
        assert!(bound_trace_preview(&text, 0).len() < 200);
    }

    #[test]
    fn history_keeps_only_the_last_256_events() {
        let mut trace = BoundCallTrace::default();
        for n in 0..400 {
            trace.record(n.to_string());
        }
        assert_eq!(trace.history.len(), 256);
        assert_eq!(trace.history.front().map(String::as_str), Some("144"));
        assert_eq!(trace.history.back().map(String::as_str), Some("399"));
    }

    /// An array-state preview reads every element up to the length bound, naming accessors and
    /// holes, and skips an array it cannot borrow or one beyond the bound.
    #[test]
    fn array_state_previews_skip_accessors_and_bound_their_arrays() {
        let owner = Object::new(None);
        let array = Object::new(None);
        array.borrow_mut().exotic = Exotic::Array;
        array
            .borrow_mut()
            .props
            .insert("length", Property::builtin(Value::Num(256.0)));
        array
            .borrow_mut()
            .props
            .insert("0", Property::accessor_prop(None, None, true, true));
        array
            .borrow_mut()
            .props
            .insert("255", Property::builtin(Value::Bool(false)));
        owner
            .borrow_mut()
            .props
            .insert("bank", Property::builtin(Value::Obj(array)));
        let values = bound_trace_arrays(&Value::Obj(owner));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].2[0], "<accessor>");
        assert_eq!(values[0].2[1], "<hole>");
        assert_eq!(values[0].2[255], "false");
        let borrowed = array.borrow_mut();
        assert!(bound_trace_arrays(&Value::Obj(owner)).is_empty());
        drop(borrowed);
        array
            .borrow_mut()
            .props
            .insert("length", Property::builtin(Value::Num(513.0)));
        assert!(bound_trace_arrays(&Value::Obj(owner)).is_empty());
    }
}

struct FunctionProfileGuard {
    dump_on_exit: bool,
    dump_after_ns: Option<u128>,
}

impl Drop for FunctionProfileGuard {
    fn drop(&mut self) {
        let _ = FUNCTION_PROFILE.try_with(|profile| {
            let elapsed = profile
                .borrow()
                .stack
                .last()
                .map(|frame| frame.started.elapsed().as_nanos())
                .unwrap_or(0);
            let mut profile = profile.borrow_mut();
            let Some(frame) = profile.stack.pop() else {
                return;
            };
            let entry = profile
                .entries
                .get_mut(&frame.key)
                .expect("profiled function has an entry");
            entry.calls += 1;
            entry.inclusive_ns += elapsed;
            entry.self_ns += elapsed.saturating_sub(frame.child_ns);
            if let Some(parent) = profile.stack.last_mut() {
                parent.child_ns += elapsed;
            }
            if profile.stack.is_empty()
                && (self.dump_on_exit
                    || self
                        .dump_after_ns
                        .is_some_and(|threshold| elapsed >= threshold))
            {
                profile.dump();
            }
        });
    }
}

fn function_profile_enter(
    func: &Rc<Function>,
    closure: &Env,
    fn_obj: &Gc,
    realm_key: usize,
    genv: usize,
) -> Option<FunctionProfileGuard> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static SOURCE: OnceLock<bool> = OnceLock::new();
    static FLUSH_NAME: OnceLock<Option<String>> = OnceLock::new();
    static FLUSH_AFTER_NS: OnceLock<Option<u128>> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("TRUST_JS_FN_PROFILE").is_some()) {
        return None;
    }
    let dump_on_exit = FLUSH_NAME
        .get_or_init(|| std::env::var("TRUST_JS_FN_PROFILE_FLUSH").ok())
        .as_deref()
        .is_some_and(|name| func.name.as_deref() == Some(name));
    let dump_after_ns = *FLUSH_AFTER_NS.get_or_init(|| {
        std::env::var("TRUST_JS_FN_PROFILE_FLUSH_SECONDS")
            .ok()
            .and_then(|seconds| seconds.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .map(|seconds| (seconds * 1_000_000_000.0) as u128)
    });
    let key = (Rc::as_ptr(func) as usize, realm_key);
    FUNCTION_PROFILE.with(|profile| {
        let mut profile = profile.borrow_mut();
        let caller = profile.stack.last().map(|frame| frame.key);
        profile.entries.entry(key).or_insert_with(|| {
            let include_source = *SOURCE
                .get_or_init(|| std::env::var_os("TRUST_JS_FN_PROFILE_SOURCE").is_some());
            // Only inspect internal state, once per function per profile batch.
            // In particular, do not call an author's getters or keep extra JS
            // roots alive. This distinguishes a slow caller from a callee whose
            // compiled entry is unavailable or ineligible for cached dispatch.
            let runtime = func.runtime();
            let compiled = runtime.execution_code_for(genv);
            let chunk = compiled.and_then(Option::as_ref);
            let code = chunk.and_then(|chunk| chunk.jit.get());
            let tier = match (compiled, chunk, code) {
                (None, _, _) => "cold",
                (Some(None), _, _) => "bytecode-bail",
                (_, _, Some(Some(_))) => "jit",
                (_, _, Some(None)) => "jit-bail",
                (_, Some(chunk), None) if chunk.jit_budget_wait_bytes.get() > 0 => "jit-budget-wait",
                _ => "bytecode",
            };
            let mut depth = 0;
            let mut scope = Some(*closure);
            while let Some(env) = scope {
                depth += 1;
                if depth >= 64 {
                    break;
                }
                scope = env.borrow().parent;
            }
            let same_realm = match &fn_obj.borrow().call {
                Callable::User(user) => user.realm == realm_key,
                _ => false,
            };
            let (code_used, code_limit, code_denials) = crate::jit::executable_code_stats();
            FunctionProfileEntry {
                description: function_profile_description(
                    key,
                    func.name.as_deref(),
                    func.source.as_deref(),
                    include_source,
                ),
                execution_state: format!(
                    "tier={tier} strict={} arrow={} with={} env_depth={depth} same_realm={same_realm} prepared={} activation={} code_used={code_used} code_limit={code_limit} code_denials={code_denials}",
                    func.is_strict,
                    func.is_arrow,
                    closure.borrow().under_with,
                    chunk.is_some_and(|chunk| chunk.prepared_entry),
                    chunk.is_some_and(|chunk| !chunk.jit_no_activation()),
                ),
                source: if include_source {
                    func.source.clone()
                } else {
                    None
                },
                ..Default::default()
            }
        });
        if let Some(caller) = caller {
            *profile
                .entries
                .get_mut(&key)
                .expect("profiled function has an entry")
                .callers
                .entry(caller)
                .or_default() += 1;
        }
        profile.stack.push(FunctionProfileFrame {
            key,
            started: std::time::Instant::now(),
            child_ns: 0,
        });
    });
    Some(FunctionProfileGuard {
        dump_on_exit,
        dump_after_ns,
    })
}

fn function_profile_dump() {
    let _ = FUNCTION_PROFILE.try_with(|profile| profile.borrow_mut().dump());
}

/// Agent-local byte storage shared by an ECMAScript `ArrayBuffer` and an embedder resource.
///
/// The single-threaded interpreter uses `RefCell` rather than a lock. WebAssembly keeps another
/// `Rc` to the same storage, so JS typed-array accesses and wasm loads/stores identify one Data
/// Block as required by the WebAssembly JavaScript Interface, without copying at call boundaries.
pub type ArrayBufferBytes = Rc<RefCell<Vec<u8>>>;

/// The Data Block behind an ArrayBuffer or SharedArrayBuffer object (its [[ArrayBufferData]]
/// slot): the values of `Interp::array_buffers`. A detached ArrayBuffer has no entry, and a
/// SharedArrayBuffer is never detached.
#[derive(Clone)]
pub(crate) enum BufferData {
    /// An ArrayBuffer's own bytes, which an embedder resource such as wasm memory may share.
    Local(ArrayBufferBytes),
    /// A SharedArrayBuffer's owning reference to its Shared Data Block. Every agent sharing
    /// the block observes the same bytes and current length.
    Shared(SharedMem),
}

impl BufferData {
    /// The block's current byte length.
    pub(crate) fn byte_len(&self) -> Option<usize> {
        match self {
            BufferData::Local(bytes) => Some(bytes.borrow().len()),
            BufferData::Shared(block) => Some(block.lock().unwrap().len()),
        }
    }

    /// Run `f` over the block's bytes: borrowed for an ArrayBuffer's, under the block's lock for
    /// a Shared Data Block.
    pub(crate) fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        match self {
            BufferData::Local(bytes) => Some(f(&bytes.borrow())),
            BufferData::Shared(block) => Some(f(&block.lock().unwrap())),
        }
    }

    /// [`Self::with_bytes`] with the bytes writable. Writes through it are not recorded as
    /// ArrayBuffer mutations; callers mark them (`Interp::mark_array_buffer_dirty_range`).
    pub(crate) fn with_bytes_mut<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> Option<R> {
        match self {
            BufferData::Local(bytes) => Some(f(&mut bytes.borrow_mut())),
            BufferData::Shared(block) => Some(f(&mut block.lock().unwrap())),
        }
    }
}

/// A weak reference to a JavaScript object, for tests that watch an object die (embedders use
/// `embed::WeakHandle`).
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct WeakValue(crate::value::WeakGc);

#[cfg(test)]
impl WeakValue {
    pub(crate) fn upgrade(&self) -> Option<Value> {
        self.0.upgrade().map(Value::Obj)
    }
}

/// `$262.agent` wiring. The main agent holds a broadcast sender per spawned agent plus the report
/// receiver; a spawned agent holds its broadcast receiver and a clone of the report sender.
pub struct AgentChannels {
    pub agent_broadcast_txs: Vec<std::sync::mpsc::Sender<(SharedMem, usize)>>,
    pub report_rx: Option<std::sync::mpsc::Receiver<String>>,
    pub report_tx: std::sync::mpsc::Sender<String>,
    pub broadcast_rx: Option<std::sync::mpsc::Receiver<(SharedMem, usize)>>,
}

/// A contiguous, getter-backed set of read-only indexed properties installed by an embedder.
///
/// This is the native representation of Web IDL's legacy platform-object indexed properties for
/// collection interfaces such as `NodeList`. The values remain lazy: the getter is called only
/// when Web IDL's `LegacyPlatformObjectGetOwnProperty` algorithm observes a supported index.
#[derive(Clone)]
pub(crate) struct HostIndexedProperties {
    pub(crate) length: u32,
    pub(crate) getter: Value,
    pub(crate) live: Option<Rc<HostIndexedLive>>,
}

/// Native membership queries are pure reads of host state: they must not run
/// JavaScript, collect, or mutate the collection. The explicit state value is
/// an ordinary traced edge, never an opaque Rust closure capturing JS roots.
pub(crate) struct HostIndexedLive {
    pub(crate) state: Value,
    pub(crate) length: fn(&Interp, &Value) -> u32,
    pub(crate) names: Option<fn(&Interp, &Value) -> Vec<String>>,
    pub(crate) named_getter: Option<Value>,
}

/// An owning reference to a SharedArrayBuffer's Shared Data Block. Buffer wrappers, queued
/// broadcasts and pending asynchronous waits retain it across agents; the identity directory
/// keeps only a weak reference, so the last owner releases the allocation.
pub type SharedMem = Arc<SharedDataBlock>;

pub struct SharedDataBlock {
    id: u64,
    bytes: Mutex<Vec<u8>>,
}

impl SharedDataBlock {
    pub(crate) fn id(&self) -> u64 {
        self.id
    }
}

impl std::ops::Deref for SharedDataBlock {
    type Target = Mutex<Vec<u8>>;

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

impl Drop for SharedDataBlock {
    fn drop(&mut self) {
        shared_mem_registry().lock().unwrap().remove(&self.id);
    }
}

/// Hashable language identity for values accepted by CanBeHeldWeakly. ECMA-262 WeakMap and WeakSet
/// operations compare keys with SameValue; for objects and symbols that is exactly pointer/id
/// identity. Keeping this separate from `Value` avoids inventing hashes for unrelated primitives.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum WeakKey {
    Object(usize),
    Symbol(u64),
}

impl WeakKey {
    pub(crate) fn of(value: &Value) -> Option<Self> {
        match &(value).view() {
            ValueView::Obj(object) => Some(Self::Object(Gc::as_ptr(object) as usize)),
            ValueView::Sym(symbol) => Some(Self::Symbol(symbol.id)),
            _ => None,
        }
    }
}

/// A weakly held ECMAScript value. `Weak` keeps an object's allocation header alive, so the raw
/// identity used by the O(1) weak-collection index cannot be recycled while an entry still exists.
/// ECMA-262 §9.9 requires these edges not to contribute to liveness.
#[derive(Clone)]
pub(crate) enum WeakTarget {
    Object(crate::value::WeakGc),
    Symbol(crate::value::WeakSymbol, u64),
}

impl WeakTarget {
    pub(crate) fn of(value: &Value) -> Option<Self> {
        match &(value).view() {
            ValueView::Obj(object) => Some(Self::Object(Gc::downgrade(object))),
            ValueView::Sym(symbol) => Some(Self::Symbol(
                crate::value::symbol_downgrade(symbol),
                symbol.id,
            )),
            _ => None,
        }
    }

    pub(crate) fn key(&self) -> WeakKey {
        match self {
            Self::Object(object) => WeakKey::Object(object.as_ptr() as usize),
            Self::Symbol(_, id) => WeakKey::Symbol(*id),
        }
    }

    pub(crate) fn upgrade(&self) -> Option<Value> {
        match self {
            Self::Object(object) => object.upgrade().map(Value::Obj),
            Self::Symbol(symbol, _) => symbol.upgrade().map(Value::Sym),
        }
    }
}

/// One FinalizationRegistry Cell record. The held value is the cell's sole strong edge; target and
/// unregister token use ECMA-262's weak retention semantics.
pub(crate) struct FinalizationCell {
    pub(crate) target: Option<WeakTarget>,
    pub(crate) held_value: Value,
    pub(crate) unregister_token: Option<WeakTarget>,
}

/// Internal slots of a FinalizationRegistry instance (ECMA-262 §26.2).
pub(crate) struct FinalizationState {
    pub(crate) cleanup_callback: JobCallback,
    pub(crate) cells: crate::finalization_cells::FinalizationCells,
    pub(crate) cleanup_scheduled: bool,
    pub(crate) registry_realm: Gc,
    pub(crate) registry_context: u64,
    pub(crate) callback_realm: Option<usize>,
    pub(crate) callback_context: u64,
}

fn shared_mem_registry() -> &'static Mutex<HashMap<u64, std::sync::Weak<SharedDataBlock>>> {
    static R: OnceLock<Mutex<HashMap<u64, std::sync::Weak<SharedDataBlock>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Default::default()))
}
pub fn next_shared_id() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, AtomicOrdering::SeqCst)
}
/// `len` zero bytes with a maximum capacity reservation, or `None` when allocation fails.
/// Fixed blocks use the allocator's zeroed path (system calloc supplies lazy zero pages for
/// large asm.js heaps). Resizable/growable blocks initialize only the logical prefix: a host
/// allocator's default `alloc_zeroed` may otherwise touch every reserved byte. Spare capacity
/// is inaccessible until `Vec::resize` initializes it. An impossible maximum must fail even
/// when `len` is zero (ECMA-262 AllocateArrayBuffer/AllocateSharedArrayBuffer).
pub(crate) fn try_zeroed_bytes_with_capacity(len: usize, capacity: usize) -> Option<Vec<u8>> {
    if len > capacity {
        return None;
    }
    if capacity == 0 {
        return Some(Vec::new());
    }
    let layout = std::alloc::Layout::array::<u8>(capacity).ok()?;
    // SAFETY: `layout` has a non-zero size. The nonzero logical prefix, if any, lies entirely
    // within the allocation; only that prefix needs initialization before the Vec owns it.
    let ptr = unsafe {
        if len == capacity {
            std::alloc::alloc_zeroed(layout)
        } else {
            let ptr = std::alloc::alloc(layout);
            if !ptr.is_null() && len != 0 {
                ptr.write_bytes(0, len);
            }
            ptr
        }
    };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` is a live allocation from the global allocator with exactly the layout a
    // `Vec<u8>` of capacity `capacity` uses (`capacity` bytes, alignment 1), and its `len` live
    // bytes are initialized (to zero); the Vec owns and frees the allocation with that layout.
    Some(unsafe { Vec::from_raw_parts(ptr, len, capacity) })
}

/// Allocate a fresh shared-memory block of `len` zero bytes; returns its owner, or `None`
/// when the memory is not available.
#[cfg(test)]
pub fn alloc_shared_mem(len: usize) -> Option<SharedMem> {
    alloc_shared_mem_with_capacity(len, len)
}

pub(crate) fn alloc_shared_mem_with_capacity(len: usize, capacity: usize) -> Option<SharedMem> {
    let block = Arc::new(SharedDataBlock {
        id: next_shared_id(),
        bytes: Mutex::new(try_zeroed_bytes_with_capacity(len, capacity)?),
    });
    shared_mem_registry()
        .lock()
        .unwrap()
        .insert(block.id(), Arc::downgrade(&block));
    Some(block)
}
pub fn shared_mem_get(id: u64) -> Option<SharedMem> {
    shared_mem_registry().lock().unwrap().get(&id)?.upgrade()
}
/// Milliseconds since a fixed process-global instant (for `$262.agent.monotonicNow`).
pub fn monotonic_now_ms() -> f64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_secs_f64() * 1000.0
}

/// A futex-like store of the WaiterList Records of `Atomics.wait`/`waitAsync`/`notify`, keyed by
/// `(shared-memory id, byte index)`. Its one lock is the critical section of every list (ECMA-262
/// EnterCriticalSection): DoWait compares the element and adds its waiter inside it, and notify
/// removes waiters inside it, so a notify can never fall between the comparison and the enqueue.
/// Each waiter has its own wake handle so `notify` can wake an exact count.
fn wait_table() -> &'static Mutex<HashMap<(u64, usize), Vec<Waiter>>> {
    static T: OnceLock<Mutex<HashMap<(u64, usize), Vec<Waiter>>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(Default::default()))
}
/// Run `f` on the WaiterList of `(id, index)` inside its critical section.
pub fn with_waiter_list<R>(id: u64, index: usize, f: impl FnOnce(&mut Vec<Waiter>) -> R) -> R {
    let mut lists = wait_table().lock().unwrap();
    let list = lists.entry((id, index)).or_default();
    let result = f(list);
    if list.is_empty() {
        lists.remove(&(id, index));
    }
    result
}
/// A Waiter Record of a WaiterList.
#[derive(Clone)]
pub enum Waiter {
    /// An agent suspended in `Atomics.wait`.
    Blocked(BlockedAgent),
    /// An `Atomics.waitAsync` promise: the token names it among the pending waits of the agent
    /// that receives its notification from other agents through the inbox.
    Async(Arc<AsyncWaitInbox>, u64),
}
/// The wake handle of an agent suspended in `Atomics.wait`: whether a notify woke it.
pub type BlockedAgent = Arc<(Mutex<bool>, Condvar)>;

/// The notifications of an agent's `Atomics.waitAsync` waiters that other agents delivered
/// (ECMA-262 EnqueueResolveInAgentJob), kept until that agent's next job checkpoint resolves
/// their promises.
#[derive(Default)]
pub struct AsyncWaitInbox {
    notified: Mutex<Vec<u64>>,
    arrived: Condvar,
}

impl AsyncWaitInbox {
    fn deliver(&self, token: u64) {
        self.notified.lock().unwrap().push(token);
        self.arrived.notify_all();
    }
    /// The tokens delivered since the last call.
    pub(crate) fn take(&self) -> Vec<u64> {
        std::mem::take(&mut self.notified.lock().unwrap())
    }
    pub(crate) fn has_notifications(&self) -> bool {
        !self.notified.lock().unwrap().is_empty()
    }
    /// Block until a notification arrives, `deadline` passes or the host interrupts the agent.
    pub(crate) fn wait_until(
        self: &Arc<Self>,
        deadline: std::time::Instant,
        interrupt: &crate::RuntimeInterrupt,
    ) {
        let inbox = self.clone();
        let wake = crate::interrupt::BlockedWake(Arc::new(move || {
            let _held = inbox.notified.lock().unwrap();
            inbox.arrived.notify_all();
        }));
        interrupt.while_blocked(wake, || {
            let notified = self.notified.lock().unwrap();
            if !notified.is_empty() || interrupt.current_reason().is_some() {
                return;
            }
            let wake_at = interrupt
                .armed_deadline()
                .map_or(deadline, |armed| armed.min(deadline));
            let now = std::time::Instant::now();
            if wake_at > now {
                drop(self.arrived.wait_timeout(notified, wake_at - now).unwrap());
            }
        });
    }
}

/// How an agent suspended in `Atomics.wait` resumed.
pub enum WaitOutcome {
    /// A notify woke it ("ok").
    Woken,
    /// Its timeout elapsed ("timed-out").
    TimedOut,
    /// The host interrupted the agent (see [`crate::RuntimeInterrupt`]).
    Interrupted,
}

/// An `Atomics.waitAsync` that went async (ECMA-262 DoWait): its promise, and the Waiter Record
/// it keeps in the list of `(id, index)` until a notify or its timeout.
pub(crate) struct AsyncWait {
    pub(crate) promise: Value,
    pub(crate) token: u64,
    pub(crate) id: u64,
    /// The waiter retains its Shared Data Block even after every JavaScript wrapper dies.
    pub(crate) block: SharedMem,
    pub(crate) index: usize,
    /// When its timeout job runs (EnqueueAtomicsWaitAsyncTimeoutJob); `None` never times out.
    pub(crate) deadline: Option<std::time::Instant>,
}

/// A token for a new `Atomics.waitAsync` waiter, unique in the process.
pub(crate) fn next_async_wait_token() -> u64 {
    static N: AtomicU64 = AtomicU64::new(1);
    N.fetch_add(1, AtomicOrdering::Relaxed)
}

/// Take async waiter `token` from the list of `(id, index)` for its timeout; `false` when a
/// notify already took it.
pub(crate) fn futex_remove_async(id: u64, index: usize, token: u64) -> bool {
    with_waiter_list(id, index, |list| {
        let Some(at) = list
            .iter()
            .position(|w| matches!(w, Waiter::Async(_, t) if *t == token))
        else {
            return false;
        };
        list.remove(at);
        true
    })
}

/// Block an agent waiting in the list of `(id, index)` until a notify wakes it, `timeout`
/// elapses, or the host interrupts it: a cancellation or navigation request wakes it at once,
/// and an armed deadline when it passes. A waiter that stops for a timeout or an interruption
/// leaves its list in the critical section, unless a notify removed it first: that notify
/// counted it as woken, so it reports "ok".
pub fn futex_block(
    agent: &BlockedAgent,
    id: u64,
    index: usize,
    timeout: Option<std::time::Duration>,
    interrupt: &crate::RuntimeInterrupt,
) -> WaitOutcome {
    let (lock, cvar) = &**agent;
    // A timeout too far away to represent never elapses.
    let deadline = timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout));
    let wake = {
        let agent = agent.clone();
        crate::interrupt::BlockedWake(Arc::new(move || {
            let _held = agent.0.lock().unwrap();
            agent.1.notify_all();
        }))
    };
    let stopped = interrupt.while_blocked(wake, || {
        let mut woken = lock.lock().unwrap();
        loop {
            if *woken {
                return None;
            }
            if interrupt.current_reason().is_some() {
                return Some(WaitOutcome::Interrupted);
            }
            let now = std::time::Instant::now();
            if deadline.is_some_and(|deadline| now >= deadline) {
                return Some(WaitOutcome::TimedOut);
            }
            woken = match [deadline, interrupt.armed_deadline()]
                .into_iter()
                .flatten()
                .min()
            {
                Some(wake_at) => {
                    let wait = wake_at.saturating_duration_since(now);
                    cvar.wait_timeout(woken, wait).unwrap().0
                }
                None => cvar.wait(woken).unwrap(),
            };
        }
    });
    let Some(outcome) = stopped else {
        return WaitOutcome::Woken;
    };
    with_waiter_list(id, index, |list| {
        let Some(at) = list
            .iter()
            .position(|w| matches!(w, Waiter::Blocked(b) if Arc::ptr_eq(b, agent)))
        else {
            return WaitOutcome::Woken;
        };
        list.remove(at);
        outcome
    })
}
/// Atomics.notify: take up to `max` (`<0` ⇒ all) waiters of `(id, index)` in list order and
/// notify each (ECMA-262 NotifyWaiter): wake a blocked agent, or deliver an async waiter's token
/// to its agent's inbox — except the calling agent's own (inbox `own`), whose tokens are
/// returned for the caller to resolve at once. Returns the count and those tokens.
pub fn futex_notify(
    id: u64,
    index: usize,
    max: i64,
    own: Option<&Arc<AsyncWaitInbox>>,
) -> (u64, Vec<u64>) {
    let taken: Vec<Waiter> = with_waiter_list(id, index, |list| {
        let n = if max < 0 {
            list.len()
        } else {
            (max as usize).min(list.len())
        };
        list.drain(..n).collect()
    });
    let mut own_tokens = Vec::new();
    for waiter in &taken {
        match waiter {
            Waiter::Blocked(agent) => {
                let (lock, cvar) = &**agent;
                *lock.lock().unwrap() = true;
                cvar.notify_all();
            }
            Waiter::Async(inbox, token) if own.is_some_and(|own| Arc::ptr_eq(own, inbox)) => {
                own_tokens.push(*token);
            }
            Waiter::Async(inbox, token) => inbox.deliver(*token),
        }
    }
    (taken.len() as u64, own_tokens)
}

pub use crate::env_handle::{Env, WeakEnv};

/// How [`Interp::resolve_cached_call`] found a call target: a live way or overflow entry to copy
/// as it is, or a code-sharing copy already adjusted to the live closure.
enum CachedCall {
    Miss,
    Entry(*const crate::bytecode::CallIc),
    Shared(crate::bytecode::CallIc),
}

/// A callee resolved once for a run of calls from one native algorithm; see
/// [`Interp::prepare_call`].
pub(crate) struct PreparedCall {
    ic: crate::bytecode::CallIc,
    /// Machine-code lease for a compiled user entry; `None` for a native entry.
    code: Option<Rc<crate::jit::JitCode>>,
}

pub(crate) const PROXY_TRAP_GET: usize = 0;
pub(crate) const PROXY_TRAP_SET: usize = 1;
pub(crate) const PROXY_TRAP_HAS: usize = 2;
pub(crate) const PROXY_TRAP_DELETE: usize = 3;
pub(crate) const PROXY_TRAP_GET_OWN_PROPERTY_DESCRIPTOR: usize = 4;
pub(crate) const PROXY_TRAP_DEFINE_PROPERTY: usize = 5;
pub(crate) const PROXY_TRAP_GET_PROTOTYPE_OF: usize = 6;
pub(crate) const PROXY_TRAP_SET_PROTOTYPE_OF: usize = 7;
pub(crate) const PROXY_TRAP_IS_EXTENSIBLE: usize = 8;
pub(crate) const PROXY_TRAP_PREVENT_EXTENSIONS: usize = 9;
pub(crate) const PROXY_TRAP_OWN_KEYS: usize = 10;
pub(crate) const PROXY_TRAP_APPLY: usize = 11;
pub(crate) const PROXY_TRAP_CONSTRUCT: usize = 12;
const PROXY_TRAP_KINDS: usize = 13;
/// The traps' names, by `PROXY_TRAP_*` kind.
const PROXY_TRAP_NAMES: [crate::atom::StaticName; PROXY_TRAP_KINDS] = {
    use crate::atom::names;
    [
        names::GET,
        names::SET,
        names::HAS,
        names::DELETE_PROPERTY,
        names::GET_OWN_PROPERTY_DESCRIPTOR,
        names::DEFINE_PROPERTY,
        names::GET_PROTOTYPE_OF,
        names::SET_PROTOTYPE_OF,
        names::IS_EXTENSIBLE,
        names::PREVENT_EXTENSIONS,
        names::OWN_KEYS,
        names::APPLY,
        names::CONSTRUCT,
    ]
};

/// Caches for the Proxy internal methods (ECMA-262 §10.5), one per trap: each trap's GetMethod
/// on the handler goes through a property inline cache of its own, and its Call through a
/// callback cache of its own, so handlers shared by many proxies (framework reactivity) stay
/// monomorphic. Neither caches a result: every operation still performs GetMethod and Call.
pub(crate) struct ProxyTrapCaches {
    lookups: [[std::cell::Cell<crate::bytecode::IcState>; crate::bytecode::PROP_IC_WAYS];
        PROXY_TRAP_KINDS],
    calls: [crate::callback::CallbackCache; PROXY_TRAP_KINDS],
}

impl ProxyTrapCaches {
    fn new() -> Self {
        Self {
            lookups: std::array::from_fn(|_| {
                [const { std::cell::Cell::new(crate::bytecode::IcState::EMPTY) };
                    crate::bytecode::PROP_IC_WAYS]
            }),
            calls: std::array::from_fn(|_| crate::callback::CallbackCache::new()),
        }
    }

    pub(crate) fn prune_dead(&self) {
        for cache in &self.calls {
            cache.prune_dead();
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.calls
            .iter()
            .map(crate::callback::CallbackCache::retained_bytes)
            .fold(std::mem::size_of::<Self>(), usize::saturating_add)
    }
}

/// One entry of the legacy `fn.caller`/`fn.arguments` reflection stack (see `call_user`). The
/// arguments object materializes lazily: a body that never names `arguments` skips building it,
/// and `lazy` keeps what a later reflective read needs to conjure it on demand.
/// `repr(C)`: the asm call sequence (arc 3b) pushes frames from machine code — fn_ptr@0,
/// call_site@8, strict@12, construct@13, extra@16, size 24 (asserted in jit.rs).
#[repr(C)]
pub struct FnFrame {
    /// [`Gc::as_ptr`] of the callee. The frame holds no handle: every frame is pushed while its
    /// caller keeps the callee reachable (the callee `Value` sits on the caller's operand stack
    /// or in the dispatch chain for the whole call), so the rare reflective reads reconstruct
    /// one via [`FnFrame::callee`].
    pub fn_ptr: usize,
    /// Where the caller was: [`Interp::call_site`] when this frame was pushed, the byte
    /// position of the call in the caller's source (see `crate::stack_trace`). Popping the frame
    /// restores it.
    pub call_site: u32,
    pub strict: bool,
    /// The frame runs a `[[Construct]]` (stack traces print `new name`).
    pub construct: bool,
    /// The rare per-frame state (a live `arguments` object, or what a reflective `fn.arguments`
    /// read needs to conjure one). Boxed so the common frame stays 24 bytes — frames are pushed
    /// and popped on EVERY call, and the pop's copy-out and drop-check of a fat frame was a
    /// measurable slice of the call path.
    pub extra: Option<Box<FrameExtra>>,
}

/// See [`FnFrame::extra`].
pub struct FrameExtra {
    pub args_obj: Value,
    pub lazy: Option<(Rc<crate::ast::Function>, Rc<[Value]>, Env)>,
}

impl Default for FrameExtra {
    fn default() -> FrameExtra {
        FrameExtra {
            args_obj: Value::Null,
            lazy: None,
        }
    }
}

/// The youngest activation of a function, for the legacy `caller` / `arguments` reflection.
#[derive(Clone, Copy)]
enum LegacyActivation {
    /// Function frame `k`.
    Frame(usize),
    /// Entry `j` of the spliced callees (innermost first) that function frame `k` was called
    /// from: they run in the context below frame `k` (see `crate::stack_trace::inline_chain`).
    Inline(usize, usize),
}

#[cfg(test)]
thread_local! {
    /// Natives [`Interp::call_native_light`] ran without the native-call boundary, for tests.
    pub(crate) static TEST_LIGHT_NATIVE_CALLS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

impl Interp {
    /// The youngest activation of the function with `Gc::as_ptr` identity `function`. Callees
    /// the compiler spliced into their caller have no frame; those that made the call to a
    /// frame above them are named by that frame's call position.
    fn legacy_activation(&self, function: usize) -> Option<LegacyActivation> {
        for k in (0..self.fn_frames.len()).rev() {
            if self.fn_frames[k].fn_ptr == function {
                return Some(LegacyActivation::Frame(k));
            }
            let (chain, _) = crate::stack_trace::inline_chain(self.fn_frames[k].call_site);
            if let Some(j) = chain
                .iter()
                .position(|(callee, _)| Gc::as_ptr(callee) as usize == function)
            {
                return Some(LegacyActivation::Inline(k, j));
            }
        }
        None
    }

    /// The legacy `caller` of `activation`: the function whose code made its call (null for
    /// none, or for a strict caller).
    fn legacy_caller(&self, activation: LegacyActivation) -> Value {
        let (k, next) = match activation {
            LegacyActivation::Frame(k) => (k, 0),
            LegacyActivation::Inline(k, j) => (k, j + 1),
        };
        let (chain, _) = crate::stack_trace::inline_chain(self.fn_frames[k].call_site);
        if let Some((callee, _)) = chain.into_iter().nth(next) {
            let strict = match &callee.borrow().call {
                Callable::User(user) => user.func.is_strict,
                _ => false,
            };
            return if strict {
                Value::Null
            } else {
                Value::Obj(callee)
            };
        }
        match self.fn_frames[..k].last() {
            None => Value::Null,
            Some(fr) if fr.strict => Value::Null,
            Some(fr) => Value::Obj(fr.callee()),
        }
    }
}

impl FnFrame {
    /// A strong handle to the callee, reconstructed from `fn_ptr` (see its aliveness invariant).
    pub(crate) fn callee(&self) -> Gc {
        let p = self.fn_ptr as *const RefCell<crate::value::Object>;
        unsafe { Gc::from_raw(p) }
    }
}

/// Diagnose whether the existing legacy reflection extension is actually used. This does not
/// expose source, function identities, property payloads or arguments, and changes no result.
/// Each of six fixed states is printed at most once per thread. Ordinary execution is unchanged.
fn trace_legacy_reflection(key: &str, frame: Option<&FnFrame>) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("TRUST_JS_TRACE_LEGACY_REFLECTION").is_some()) {
        return;
    }
    let state = match (key, frame) {
        ("arguments", None) => 0,
        ("arguments", Some(frame)) => match frame.extra.as_deref() {
            Some(extra) if !matches!((extra.args_obj).view(), ValueView::Null) => 1,
            Some(extra) if extra.lazy.is_some() => 2,
            _ => 3,
        },
        ("caller", None) => 4,
        ("caller", Some(_)) => 5,
        _ => return,
    };
    thread_local! { static SEEN: std::cell::Cell<u8> = const { std::cell::Cell::new(0) }; }
    SEEN.with(|seen| {
        let bit = 1 << state;
        if seen.get() & bit == 0 {
            seen.set(seen.get() | bit);
            let label = [
                "arguments: inactive",
                "arguments: active, materialized",
                "arguments: active, lazily available",
                "arguments: active, unavailable",
                "caller: inactive",
                "caller: active",
            ][state];
            eprintln!("[legacy-reflection] {label}");
        }
    });
}

/// An interpreter's JIT activation records (see `Interp::frame_pool` and
/// [`crate::jit::JitFrame`]). `pool` is the freelist (dereferenced to): Rust entries pop and push
/// it, and so does generated code, through the Vec's words at their probed offsets
/// (`InterpLayout::frame_pool` is the address of `pool`). A pooled record's frame contents are
/// already released. Teardown frees the pooled records.
///
/// `records` knows every record allocated for this interpreter and not yet freed:
/// [`crate::jit::JitFrame::alloc`] inserts it, and freeing it (`JitFrame::release` past
/// [`crate::jit::FRAME_POOL_LIMIT`], or teardown here) removes it. A record not in
/// `pool[..len]` is active: a Rust entry (`run_moved_inner`) or a direct call popped it for an
/// activation that has not finished yet (generated code pops with `len--` and pushes back in the
/// finish stub; it never allocates or frees one). The collector traces every active record (see
/// `GcTrace for FramePool`), so the registry costs nothing per call.
pub(crate) struct FramePool {
    pub(crate) pool: Vec<std::ptr::NonNull<crate::jit::JitFrame>>,
    records: crate::fasthash::FastSet<std::ptr::NonNull<crate::jit::JitFrame>>,
}

/// The freelist reserves its whole capacity at once: `JitFrame::release` pushes only below
/// [`crate::jit::FRAME_POOL_LIMIT`], and the direct-call finish stub's push relies on it never
/// reallocating.
impl Default for FramePool {
    fn default() -> Self {
        Self {
            pool: Vec::with_capacity(crate::jit::FRAME_POOL_LIMIT),
            records: Default::default(),
        }
    }
}

impl std::ops::Deref for FramePool {
    type Target = Vec<std::ptr::NonNull<crate::jit::JitFrame>>;
    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}
impl std::ops::DerefMut for FramePool {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.pool
    }
}
impl Drop for FramePool {
    fn drop(&mut self) {
        for p in self.pool.drain(..) {
            self.records.remove(&p);
            crate::jit::note_record(p, false);
            // SAFETY: pooled records come from `JitFrame::alloc` and are not executing.
            drop(unsafe { Box::from_raw(p.as_ptr()) });
        }
    }
}

impl FramePool {
    /// Note a record [`crate::jit::JitFrame::alloc`] just created for this interpreter.
    pub(crate) fn register(&mut self, record: std::ptr::NonNull<crate::jit::JitFrame>) {
        self.records.insert(record);
        crate::jit::note_record(record, true);
    }

    /// Forget a record about to be freed.
    pub(crate) fn unregister(&mut self, record: std::ptr::NonNull<crate::jit::JitFrame>) {
        self.records.remove(&record);
        crate::jit::note_record(record, false);
    }

    /// How many records are registered, pooled or active (tests).
    #[cfg(test)]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn registered(&self) -> usize {
        self.records.len()
    }

    /// The registry's own storage (memory reports; it keeps the capacity a deep recursion
    /// reached).
    pub(crate) fn registry_bytes(&self) -> usize {
        self.records
            .capacity()
            .saturating_mul(size_of::<std::ptr::NonNull<crate::jit::JitFrame>>())
    }

    /// The records of this interpreter that are not pooled: the running activations'.
    pub(crate) fn active_records(&self) -> Vec<std::ptr::NonNull<crate::jit::JitFrame>> {
        // Outside JavaScript every record is pooled; the pool holds at most FRAME_POOL_LIMIT.
        if self.records.len() == self.pool.len() {
            return Vec::new();
        }
        let pooled: crate::fasthash::FastSet<_> = self.pool.iter().copied().collect();
        self.records
            .iter()
            .copied()
            .filter(|record| !pooled.contains(record))
            .collect()
    }
}

/// Every active record: its context's values precisely and its words conservatively (see
/// `GcTrace for JitFrame`). Pooled records own nothing and are not traced, so the stale words a
/// finished activation leaves in its record retain nothing.
impl crate::gc_trace::GcTrace for FramePool {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        for record in self.active_records() {
            // SAFETY: a registered record is allocated until it is freed, which happens only
            // to pooled records (or past the pool limit, after its activation finished).
            unsafe { record.as_ref() }.trace_gc(sink);
        }
    }
}

/// A string's code units as the UTF-16 view readers take them (see [`crate::lstr::Units`]).
#[derive(Clone, Copy)]
pub(crate) enum StrUnits<'a> {
    /// One-byte: unit `i` is byte `i`.
    Ascii,
    /// Two-byte: the stored units.
    Units(&'a [u16]),
}

/// Raw state of the last successful regex match, deferred for the legacy `RegExp.$1` statics.
/// `ctor` is the %RegExp% constructor the statics belong to (the realm active at match time).
/// It is weak: a pending match must not keep that constructor's Realm alive, and once the
/// Realm is gone its statics can no longer be observed.
#[derive(Clone)]
pub(crate) struct RegexpLastMatch {
    pub ctor: crate::value::WeakGc,
    pub input: crate::lstr::LStr,
    pub caps: Vec<Option<(usize, usize)>>,
    pub ngroups: usize,
    /// Dead-result regexp operations record only the whole match and defer capture extraction
    /// until a legacy `RegExp.$1`-style accessor actually observes it.
    pub lazy_captures: Option<(Rc<crate::regex::Regex>, usize)>,
}

/// One scope's bindings: their names ([`Names`]) and, slot for slot, their values. Every
/// *structural* mutation, anything that can move a binding or change what a name resolves to
/// (insert, remove, clear), bumps a generation counter. The bytecode tier's per-site name caches
/// hold a raw `&Binding` pointer plus the generation they resolved it at (see
/// `bytecode::NameIc`): a matching generation proves the map hasn't changed shape since, so the
/// pointer is still valid *and* still the right resolution. In-place binding writes (`get_mut`)
/// intentionally don't bump: they can't move a binding, and a cache read-through observes the
/// new value, which is exactly correct. Mutations only exist as the inherent methods below, so
/// a new mutation site can't forget the bump (it won't compile).
pub struct VarMap {
    names: Names,
    values: BindingValues,
    generation: std::cell::Cell<u32>,
    /// Nonzero after a fully instantiated activation publishes its ordered key
    /// layout. Changes to that key set or storage invalidate the proof.
    /// Binding values/TDZ flags remain live and are never part of this identity.
    layout_id: u32,
}

static NEXT_BINDING_LAYOUT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Process-unique, never-recycled binding-layout identity. Exhaustion disables
/// new layout proofs rather than wrapping into an old cache's identity.
pub(crate) fn new_binding_layout_id() -> u32 {
    allocate_binding_layout_id(&NEXT_BINDING_LAYOUT)
}

fn allocate_binding_layout_id(next: &std::sync::atomic::AtomicU32) -> u32 {
    use std::sync::atomic::Ordering::Relaxed;
    crate::atomic_compat::try_update_u32(next, Relaxed, Relaxed, |id| id.checked_add(1))
        .unwrap_or(0)
}

/// Past this many names a descriptor or a record's own list keeps an index of them.
const SMALL_VAR_MAP_CAPACITY: usize = 8;

/// Shared names and slot positions, as text, with no binding values or environment owners:
/// the shape a tree-walked function's activations share, which its AST, shared by every heap,
/// caches. Each heap turns it into a [`ScopeDesc`] of its atoms ([`crate::value::scope_desc_for`]).
/// Created from a completed FunctionDeclarationInstantiation, so parameter eval
/// and conditional constructor metadata cannot masquerade as a static key set.
#[derive(Debug)]
pub(crate) struct BindingLayout {
    id: u32,
    slots: crate::fasthash::FastMap<Rc<str>, usize>,
}

impl BindingLayout {
    pub(crate) fn retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        for name in self.slots.keys() {
            visitor.rc_str(name);
        }
        // HashMap does not expose its bucket allocation; a lower bound.
        std::mem::size_of::<Self>() + self.slots.len() * std::mem::size_of::<(Rc<str>, usize)>()
    }

    /// The slot of the binding named by `name`, if the layout has one.
    fn slot_of_atom(&self, name: crate::atom::Atom) -> Option<usize> {
        self.slots.get(&*name.text()).copied()
    }

    /// The names, by slot.
    pub(crate) fn ordered_names(&self) -> Vec<&str> {
        let mut names = vec![""; self.slots.len()];
        for (name, &slot) in &self.slots {
            names[slot] = name;
        }
        names
    }
}

/// The names of a scope's bindings in slot order, shared by every record the scope makes (a
/// compiled scope's, or a tree-walked function's activation shape). Its heap keeps it for as
/// long as the heap lives ([`crate::value::keep_scope_desc`]), and its records name it without
/// owning it ([`DescRef`]). Making a record then copies binding values, not names.
pub(crate) struct ScopeDesc {
    names: Box<[crate::atom::Atom]>,
    /// Name → slot, past [`SMALL_VAR_MAP_CAPACITY`] names.
    index: Option<crate::fasthash::FastMap<crate::atom::Atom, u32>>,
    /// Text → slot, past [`SMALL_VAR_MAP_CAPACITY`] names: the tree-walker and the runtime's
    /// name references look names up as text.
    text: Option<Rc<BindingLayout>>,
}

impl ScopeDesc {
    pub(crate) fn new(names: Vec<crate::atom::Atom>) -> Rc<Self> {
        let large = names.len() > SMALL_VAR_MAP_CAPACITY;
        let text = large.then(|| {
            Rc::new(BindingLayout {
                id: 0,
                slots: names
                    .iter()
                    .enumerate()
                    .map(|(slot, name)| (Rc::from(&*name.text()), slot))
                    .collect(),
            })
        });
        Self::with_text(names, text)
    }

    /// A descriptor of `names` whose text index, if it needs one, is `text`'s.
    fn with_text(names: Vec<crate::atom::Atom>, text: Option<Rc<BindingLayout>>) -> Rc<Self> {
        let large = names.len() > SMALL_VAR_MAP_CAPACITY;
        let index = large.then(|| {
            names
                .iter()
                .enumerate()
                .map(|(slot, &name)| (name, slot as u32))
                .collect()
        });
        Rc::new(Self {
            names: names.into_boxed_slice(),
            index,
            text: text.filter(|_| large),
        })
    }

    /// The descriptor of a tree-walked function's activation layout, in the active heap's
    /// atoms; the layout serves as its text index.
    pub(crate) fn of_layout(layout: &Rc<BindingLayout>) -> Rc<Self> {
        let names = layout
            .ordered_names()
            .into_iter()
            .map(crate::atom::atomize)
            .collect();
        Self::with_text(names, Some(layout.clone()))
    }

    #[inline]
    fn slot_of(&self, name: crate::atom::Atom) -> Option<usize> {
        match &self.index {
            Some(index) => index.get(&name).map(|&slot| slot as usize),
            None => self.names.iter().position(|&known| known == name),
        }
    }

    pub(crate) fn retained_memory(&self) -> usize {
        // The names are atoms, which their heap's table accounts, and a text index a layout's,
        // counted with the layout. HashMap does not expose its bucket allocation; a lower bound.
        std::mem::size_of::<Self>()
            + self.names.len() * std::mem::size_of::<crate::atom::Atom>()
            + self.index.as_ref().map_or(0, |index| {
                index.len() * std::mem::size_of::<(crate::atom::Atom, u32)>()
            })
    }
}

/// A scope descriptor its heap keeps for as long as the heap lives
/// ([`crate::value::keep_scope_desc`]), named by the records that share it without owning it.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(crate) struct DescRef(std::ptr::NonNull<ScopeDesc>);

impl DescRef {
    /// # Safety
    /// `desc` outlives every record that names it: the active heap keeps it, or it is static.
    pub(crate) unsafe fn kept(desc: &ScopeDesc) -> Self {
        Self(std::ptr::NonNull::from(desc))
    }

    /// The descriptor of no names, which every heap shares.
    pub(crate) fn empty() -> Self {
        struct Empty(ScopeDesc);
        // SAFETY: the descriptor is immutable and holds no atom.
        unsafe impl Sync for Empty {}
        // SAFETY: as for `Sync`.
        unsafe impl Send for Empty {}
        static EMPTY: std::sync::OnceLock<Empty> = std::sync::OnceLock::new();
        let empty = EMPTY.get_or_init(|| {
            Empty(ScopeDesc {
                names: Box::new([]),
                index: None,
                text: None,
            })
        });
        // SAFETY: the empty descriptor is static.
        unsafe { Self::kept(&empty.0) }
    }
}

impl std::ops::Deref for DescRef {
    type Target = ScopeDesc;
    #[inline]
    fn deref(&self) -> &ScopeDesc {
        // SAFETY: the heap of the record that names the descriptor keeps it (or it is static),
        // and the record lives no longer than its heap.
        unsafe { self.0.as_ref() }
    }
}

/// A record's own names: those of a record built name by name (the tree-walker's, eval's) or
/// changed after a descriptor's (a sloppy eval's `var`), in a storage block the record reports.
/// It is never dropped: the names are atoms, the spill is a block, and a large list's index
/// lives in a payload cell whose finalizer drops it.
struct OwnNames {
    names: std::mem::ManuallyDrop<
        crate::inline_vec::InlineVec<crate::atom::Atom, 2, crate::gc_storage::GcSpill>,
    >,
    index: Option<crate::gc_storage::PayloadBox<NameIndex>>,
}

/// An [`OwnNames`]' name → slot index, past [`SMALL_VAR_MAP_CAPACITY`] names.
#[derive(Default)]
pub(crate) struct NameIndex(crate::fasthash::FastMap<crate::atom::Atom, u32>);

impl std::ops::Deref for NameIndex {
    type Target = crate::fasthash::FastMap<crate::atom::Atom, u32>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for NameIndex {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A record's binding names, by slot, in one word: a descriptor records share ([`DescRef`]), the
/// record's own list ([`OwnNames`], in a storage block), or, while a record that owns its names
/// has one binding, that name itself, which needs no block. Cells are 16-byte aligned, so the
/// low bits tag the kind.
#[derive(Clone, Copy)]
struct Names(std::num::NonZeroUsize);

impl Names {
    const SHARED: usize = 0;
    const OWN: usize = 1;
    const ONE: usize = 2;
    const TAG: usize = 3;

    #[inline]
    fn shared(desc: DescRef) -> Self {
        Self(std::num::NonZeroUsize::new(desc.0.as_ptr() as usize).expect("non-null"))
    }

    #[inline]
    fn one(name: crate::atom::Atom) -> Self {
        let word = name.cell_word() as usize;
        debug_assert_eq!(word & Self::TAG, 0, "string cells are aligned");
        Self(std::num::NonZeroUsize::new(word | Self::ONE).expect("non-null"))
    }

    fn own(names: crate::gc_storage::StorageBox<OwnNames>) -> Self {
        let word = crate::gc_storage::StorageBox::into_raw(names) as usize;
        debug_assert_eq!(word & Self::TAG, 0, "blocks are aligned");
        Self(std::num::NonZeroUsize::new(word | Self::OWN).expect("non-null"))
    }

    #[inline]
    fn kind(self) -> usize {
        self.0.get() & Self::TAG
    }

    #[inline]
    fn untagged(self) -> usize {
        self.0.get() & !Self::TAG
    }

    #[inline]
    fn desc(&self) -> Option<&ScopeDesc> {
        // SAFETY: a shared word is a kept descriptor's address (see `DescRef`).
        (self.kind() == Self::SHARED).then(|| unsafe { &*(self.untagged() as *const ScopeDesc) })
    }

    #[inline]
    fn own_names(&self) -> Option<&OwnNames> {
        // SAFETY: an own word is the address of the record's block, which the record reports.
        (self.kind() == Self::OWN).then(|| unsafe { &*(self.untagged() as *const OwnNames) })
    }

    #[inline]
    fn own_names_mut(&mut self) -> Option<&mut OwnNames> {
        // SAFETY: as in `own_names`; the record, borrowed mutably, is the block's one owner.
        (self.kind() == Self::OWN).then(|| unsafe { &mut *(self.untagged() as *mut OwnNames) })
    }

    #[inline]
    fn single(self) -> Option<crate::atom::Atom> {
        // SAFETY: a one word is an atom's address (see `Names::one`).
        (self.kind() == Self::ONE)
            .then(|| unsafe { crate::atom::Atom::from_cell_word(self.untagged() as u64) })
    }

    fn len(&self) -> usize {
        match self.kind() {
            Self::SHARED => self.desc().expect("shared").names.len(),
            Self::OWN => self.own_names().expect("own").names.len(),
            _ => 1,
        }
    }

    /// The name at `slot`.
    #[inline]
    fn get(&self, slot: usize) -> Option<crate::atom::Atom> {
        match self.kind() {
            Self::SHARED => self.desc().expect("shared").names.get(slot).copied(),
            Self::OWN => self.own_names().expect("own").names.get(slot).copied(),
            _ => (slot == 0).then(|| self.single().expect("one")),
        }
    }

    /// The slot of `name`.
    #[inline]
    fn slot_of(&self, name: crate::atom::Atom) -> Option<usize> {
        match self.kind() {
            Self::SHARED => self.desc().expect("shared").slot_of(name),
            Self::OWN => {
                let own = self.own_names().expect("own");
                match &own.index {
                    Some(index) => index.get(&name).map(|&slot| slot as usize),
                    None => own.names.iter().position(|&known| known == name),
                }
            }
            _ => (self.single() == Some(name)).then_some(0),
        }
    }

    /// `f` on the names, in slot order.
    #[inline]
    fn with_slice<R>(&self, f: impl FnOnce(&[crate::atom::Atom]) -> R) -> R {
        match self.kind() {
            Self::SHARED => f(&self.desc().expect("shared").names),
            Self::OWN => f(&self.own_names().expect("own").names),
            _ => f(std::slice::from_ref(&self.single().expect("one"))),
        }
    }

    /// The slot of the name whose text is `text`.
    fn slot_of_text(&self, text: &str) -> Option<usize> {
        if let Some(desc) = self.desc() {
            if let Some(layout) = &desc.text {
                return layout.slots.get(text).copied();
            }
        } else if self.own_names().is_some_and(|own| own.index.is_some()) {
            return self.slot_of(crate::atom::lookup(text)?);
        }
        self.with_slice(|names| names.iter().position(|name| name.is(text)))
    }

    /// Exchange the names at `a` and `b`, owning the names first, without updating an own
    /// list's index: for a permutation whose names a descriptor then replaces.
    fn swap_unindexed(&mut self, a: usize, b: usize) {
        if a != b {
            self.own_mut().names.swap(a, b);
        }
    }

    /// The names as the record's own list, copied out of a descriptor or the one-name word
    /// first (the generation's bump is the caller's).
    fn own_mut(&mut self) -> &mut OwnNames {
        if self.kind() != Self::OWN {
            let mut names = crate::inline_vec::InlineVec::new();
            for slot in 0..self.len() {
                names.push(self.get(slot).expect("in range"));
            }
            *self = Self::own(crate::gc_storage::StorageBox::new(OwnNames {
                names: std::mem::ManuallyDrop::new(names),
                index: None,
            }));
            self.reindex();
        }
        self.own_names_mut().expect("own")
    }

    /// Rebuild an own list's index after its names changed: one past
    /// [`SMALL_VAR_MAP_CAPACITY`] names, none below.
    fn reindex(&mut self) {
        let own = self.own_names_mut().expect("own");
        if own.names.len() <= SMALL_VAR_MAP_CAPACITY {
            own.index = None;
            return;
        }
        let index: crate::fasthash::FastMap<crate::atom::Atom, u32> = own
            .names
            .iter()
            .enumerate()
            .map(|(slot, &name)| (name, slot as u32))
            .collect();
        match &mut own.index {
            Some(existing) => (**existing).0 = index,
            None => own.index = Some(crate::gc_storage::PayloadBox::new(NameIndex(index))),
        }
    }

    /// Append `name` at the next slot, owning the names first.
    fn push(&mut self, name: crate::atom::Atom) {
        if self.kind() == Self::SHARED && self.len() == 0 {
            *self = Self::one(name);
            return;
        }
        let own = self.own_mut();
        own.names.push(name);
        let slot = own.names.len() - 1;
        match &mut own.index {
            Some(index) => {
                index.insert(name, slot as u32);
            }
            None if own.names.len() > SMALL_VAR_MAP_CAPACITY => self.reindex(),
            None => {}
        }
    }

    /// Remove the name at `slot`, moving the last into its place (as `swap_remove`).
    fn swap_remove(&mut self, slot: usize) {
        if self.kind() == Self::ONE {
            *self = Self::shared(DescRef::empty());
            return;
        }
        let own = self.own_mut();
        own.names.swap_remove(slot);
        self.reindex();
    }

    /// Report an own list's blocks and cells.
    fn trace<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(own) = self.own_names() {
            crate::gc_storage::trace_block(self.untagged() as *const u8, sink);
            if let Some(block) = own.names.spill_ptr() {
                crate::gc_storage::trace_block(block, sink);
            }
            if let Some(index) = &own.index {
                crate::gc_storage::PayloadBox::trace(index, sink);
            }
        }
    }

    /// The bytes the names own beside the record, and whether the count is exact.
    fn retained_requested_storage_bytes(&self) -> (usize, bool) {
        match self.own_names() {
            Some(own) => (
                std::mem::size_of::<OwnNames>()
                    + own.names.heap_capacity() * std::mem::size_of::<crate::atom::Atom>()
                    + own.index.as_ref().map_or(0, |index| {
                        index.len() * std::mem::size_of::<(crate::atom::Atom, u32)>()
                    }),
                own.index.is_none(),
            ),
            None => (0, true),
        }
    }
}

/// A record's binding values in slot order. P1 S4: the first lives in the environment cell, so
/// a record with one binding (most block and many function records) allocates nothing beside
/// it; more move the values to one storage block (`gc_storage`), which the record reports
/// ([`VarMap::trace_storage`]). The values are never dropped: bindings own nothing and the
/// collector frees their block, so an environment carries no drop glue and needs no finalizer.
#[repr(transparent)]
pub(crate) struct BindingValues(std::mem::ManuallyDrop<BindingVec>);

type BindingVec = crate::inline_vec::InlineVec<Binding, 1, crate::gc_storage::GcSpill>;

impl BindingValues {
    pub(crate) fn new() -> Self {
        Self(std::mem::ManuallyDrop::new(BindingVec::new()))
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self(std::mem::ManuallyDrop::new(BindingVec::with_capacity(
            capacity,
        )))
    }
}

impl Default for BindingValues {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for BindingValues {
    type Target = BindingVec;
    #[inline]
    fn deref(&self) -> &BindingVec {
        &self.0
    }
}

impl std::ops::DerefMut for BindingValues {
    #[inline]
    fn deref_mut(&mut self) -> &mut BindingVec {
        &mut self.0
    }
}

impl FromIterator<Binding> for BindingValues {
    fn from_iter<I: IntoIterator<Item = Binding>>(iter: I) -> Self {
        Self(std::mem::ManuallyDrop::new(iter.into_iter().collect()))
    }
}

impl Default for VarMap {
    fn default() -> Self {
        Self {
            names: Names::shared(DescRef::empty()),
            values: BindingValues::new(),
            generation: std::cell::Cell::new(0),
            layout_id: 0,
        }
    }
}

/// The bindings of a [`VarMap`] with their names, in slot order.
pub(crate) struct VarIter<'a> {
    map: &'a VarMap,
    slot: usize,
}

impl<'a> Iterator for VarIter<'a> {
    type Item = (crate::atom::Atom, &'a Binding);
    fn next(&mut self) -> Option<Self::Item> {
        let binding = self.map.values.get(self.slot)?;
        let name = self.map.names.get(self.slot).expect("a name per value");
        self.slot += 1;
        Some((name, binding))
    }
}

impl VarMap {
    /// Report the record's blocks: the spilled values' and the names'.
    #[inline]
    pub(crate) fn trace_storage<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(block) = self.values.spill_ptr() {
            crate::gc_storage::trace_block(block, sink);
        }
        self.names.trace(sink);
    }

    /// Publish actual instantiated names, never assumptions about syntax. A
    /// different eval-created key set retains the dynamic names. Moving bindings
    /// invalidates raw-pointer caches before any address can change.
    pub(crate) fn share_layout(&mut self, cached: &std::cell::OnceCell<Rc<BindingLayout>>) {
        if self.layout_id != 0 {
            return;
        }
        let layout = cached.get_or_init(|| {
            Rc::new(BindingLayout {
                id: new_binding_layout_id(),
                slots: self
                    .keys()
                    .enumerate()
                    .map(|(slot, name)| (Rc::from(&*name.text()), slot))
                    .collect(),
            })
        });
        if layout.id == 0
            || self.values.len() != layout.slots.len()
            || self.keys().any(|name| layout.slot_of_atom(name).is_none())
        {
            return;
        }
        let desc = crate::value::scope_desc_for(layout);
        self.bump();
        // A linear permutation into the shared ordering, without cloning values
        // or retaining an old activation. All names were validated above.
        for slot in 0..self.values.len() {
            loop {
                let name = self.names.get(slot).expect("a name per value");
                let target = desc.slot_of(name).expect("validated");
                if target == slot {
                    break;
                }
                self.names.swap_unindexed(slot, target);
                self.values.swap(slot, target);
            }
        }
        self.names = Names::shared(desc);
        self.layout_id = layout.id;
    }

    /// The descriptor the record shares, if it does.
    pub(crate) fn shared_desc(&self) -> Option<&ScopeDesc> {
        self.names.desc().filter(|desc| !desc.names.is_empty())
    }

    /// A compiled scope's record: `values` named by `desc`'s names, slot for slot.
    /// Structural writes still use `insert`/`remove` and invalidate the published
    /// layout before changing or replacing this storage.
    pub(crate) fn from_fixed_bindings(values: BindingValues, desc: DescRef) -> Self {
        debug_assert!(values.iter().all(|binding| binding.import_ref.is_none()));
        debug_assert_eq!(values.len(), desc.names.len());
        Self {
            names: Names::shared(desc),
            values,
            generation: std::cell::Cell::new(0),
            layout_id: 0,
        }
    }

    /// Before publication only: the initialization plan owns this exact slot.
    pub(crate) fn initialize_fixed_value(&mut self, slot: usize, value: Value) {
        debug_assert_eq!(self.layout_id, 0);
        self.values[slot].set_value(value);
    }

    /// Requested bytes owned directly by this binding map, plus whether the result is exact.
    /// `HashMap` bucket allocation sizes are intentionally reported as an incomplete lower bound
    /// because the standard library does not expose their requested layout.
    pub(crate) fn retained_requested_storage_bytes(&self) -> (usize, bool) {
        let (names, exact) = self.names.retained_requested_storage_bytes();
        (
            names.saturating_add(
                self.values
                    .heap_capacity()
                    .saturating_mul(std::mem::size_of::<Binding>()),
            ),
            exact,
        )
    }

    pub(crate) fn with_capacity(capacity: usize) -> VarMap {
        VarMap {
            values: BindingValues::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// The structural generation (name-cache validation token).
    #[inline]
    pub(crate) fn generation(&self) -> u32 {
        self.generation.get()
    }
    /// Exhaustion permanently disables identity/generation pointer proofs for this map.
    /// Published fixed-layout identities are independent, non-recycled key-set proofs.
    #[inline]
    pub(crate) fn matches_generation(&self, generation: u32) -> bool {
        generation != u32::MAX && self.generation.get() == generation
    }
    #[cfg(test)]
    pub(crate) fn set_generation_for_test(&mut self, generation: u32) {
        self.generation.set(generation);
        self.layout_id = 0;
    }
    #[inline]
    fn bump(&mut self) {
        self.generation.set(self.generation.get().saturating_add(1));
        self.layout_id = 0;
    }

    #[inline]
    pub(crate) fn layout_id(&self) -> u32 {
        self.layout_id
    }

    /// Stable ordered slot within an activation.
    pub(crate) fn binding_slot(&self, name: &str) -> Option<usize> {
        self.names.slot_of_text(name)
    }

    pub(crate) fn binding_at(&self, slot: usize, name: &str) -> Option<&Binding> {
        self.names
            .get(slot)
            .filter(|key| key.is(name))
            .and_then(|_| self.values.get(slot))
    }

    pub(crate) fn binding_at_mut(&mut self, slot: usize, name: &str) -> Option<&mut Binding> {
        if !self.names.get(slot).is_some_and(|key| key.is(name)) {
            return None;
        }
        self.values.get_mut(slot)
    }

    /// The caller has checked the same nonzero published ordered-layout ID.
    pub(crate) fn fixed_binding_at(&self, slot: usize) -> Option<&Binding> {
        self.values.get(slot)
    }

    pub(crate) fn fixed_binding_at_mut(&mut self, slot: usize) -> Option<&mut Binding> {
        self.values.get_mut(slot)
    }

    /// Only the creator of a fresh, fully instantiated fixed-key activation may
    /// publish its plan's identity. Never publish on a reused/dynamic environment.
    pub(crate) fn publish_layout(&mut self, layout_id: u32) {
        self.layout_id = layout_id;
    }

    pub(crate) fn insert(
        &mut self,
        k: impl crate::atom::IntoAtom,
        mut v: Binding,
    ) -> Option<Binding> {
        v.imported = v.import_ref.is_some();
        let k = k.into_atom();
        if let Some(slot) = self.names.slot_of(k) {
            // InitializeBinding and same-name declaration replacement keep
            // the key set and addresses. Native users read flags/values live.
            self.generation.set(self.generation.get().saturating_add(1));
            if self.names.kind() != Names::SHARED {
                self.layout_id = 0;
            }
            return Some(std::mem::replace(&mut self.values[slot], v));
        }
        self.bump();
        self.names.push(k);
        self.values.push(v);
        None
    }

    pub fn remove(&mut self, k: &str) -> Option<Binding> {
        self.bump();
        let slot = self.names.slot_of_text(k)?;
        self.names.swap_remove(slot);
        Some(self.values.swap_remove(slot))
    }

    pub fn clear(&mut self) {
        self.bump();
        self.names = Names::shared(DescRef::empty());
        self.values.clear();
    }

    /// In-place binding write: bindings don't move, so the generation stays (see the type docs).
    pub fn get_mut(&mut self, k: &str) -> Option<&mut Binding> {
        let slot = self.names.slot_of_text(k)?;
        self.values.get_mut(slot)
    }

    pub fn get(&self, k: &str) -> Option<&Binding> {
        let slot = self.names.slot_of_text(k)?;
        self.values.get(slot)
    }

    pub fn contains_key(&self, k: &str) -> bool {
        self.names.slot_of_text(k).is_some()
    }

    /// The binding named by `name`, an atom of this environment's heap.
    pub(crate) fn get_atom(&self, name: crate::atom::Atom) -> Option<&Binding> {
        let slot = self.names.slot_of(name)?;
        self.values.get(slot)
    }

    /// In-place binding write by atom (see [`VarMap::get_mut`]).
    pub(crate) fn get_atom_mut(&mut self, name: crate::atom::Atom) -> Option<&mut Binding> {
        let slot = self.names.slot_of(name)?;
        self.values.get_mut(slot)
    }

    pub(crate) fn iter(&self) -> VarIter<'_> {
        VarIter { map: self, slot: 0 }
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = crate::atom::Atom> + '_ {
        (0..self.values.len()).map(|slot| self.names.get(slot).expect("a name per value"))
    }

    pub(crate) fn values(&self) -> std::slice::Iter<'_, Binding> {
        self.values.iter()
    }

    /// Byte offset of the generation counter within a `VarMap` (for the JIT's inline template).
    pub(crate) fn generation_offset() -> usize {
        std::mem::offset_of!(VarMap, generation)
    }

    pub(crate) fn layout_id_offset() -> usize {
        std::mem::offset_of!(VarMap, layout_id)
    }

    /// Where generated code finds a record's binding values: an `InlineVec` whose first value
    /// stays in the environment cell, probed in place and spilled. A native fresh-activation
    /// hit additionally requires the same nonzero ordered layout identity. Unknown compiler
    /// layouts simply retain the checked map path.
    pub(crate) fn jit_small_storage_layout() -> Option<JitBindingStorage> {
        let storage = BindingVec::probe_jit_layout(|| Binding::data(Value::Num(7.0), true, true))?;
        Some(JitBindingStorage {
            entries: std::mem::offset_of!(VarMap, values),
            storage,
            stride: std::mem::size_of::<Binding>(),
            binding: 0,
        })
    }
}

/// Where generated code finds a binding map's values, relative to its [`VarMap`]
/// (see [`VarMap::jit_small_storage_layout`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JitBindingStorage {
    /// The values' vector.
    pub(crate) entries: usize,
    /// How the vector holds its `Binding` elements.
    pub(crate) storage: crate::value::JitVecLayout,
    /// The element stride, and the `Binding` within an element.
    pub(crate) stride: usize,
    pub(crate) binding: usize,
}

#[cfg(test)]
mod binding_layout_tests {
    use super::*;
    use crate::value::ValueView;

    #[test]
    fn cold_import_cells_trace_exporters_and_keep_cached_reads_live() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = crate::Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let global = engine.interp.global_env;
            // Strong handles live only in helper frames (the collector scans the stack).
            let weak = crate::gc_test::in_frame(|| {
                let exporter = new_scope(Some(global));
                let object = engine.interp.new_object();
                object
                    .borrow_mut()
                    .props
                    .insert("n", crate::value::Property::plain(Value::Num(7.)));
                exporter
                    .borrow_mut()
                    .vars
                    .insert("source", Binding::data(Value::Obj(object), true, true));
                let mut import = Binding::data(Value::Undefined, false, true);
                import.set_import_reference(Some((exporter, crate::atom::atomize("source"))));
                global.borrow_mut().vars.insert("coldLeft", import.clone());
                global.borrow_mut().vars.insert("coldRight", import);
                Env::downgrade(&exporter)
            });
            assert_eq!(evaluate(&mut engine, "function readCold(){return coldLeft.n+coldRight.n;}for(var k=0;k<40;k++)readCold();readCold()"), "14");
            engine.interp.gc_collect();
            crate::gc_test::in_frame(|| {
                let exporter = weak
                    .upgrade()
                    .expect("rooted imports retain their exporter");
                let value = exporter.borrow().vars.get("source").unwrap().value();
                value
                    .as_obj()
                    .unwrap()
                    .borrow_mut()
                    .props
                    .insert("n", crate::value::Property::plain(Value::Num(9.)));
            });
            assert_eq!(evaluate(&mut engine, "readCold()"), "18", "{tier:?}");
            global.borrow_mut().vars.remove("coldLeft");
            engine.interp.gc_collect();
            assert_eq!(evaluate(&mut engine, "coldRight.n"), "9", "{tier:?}");
            global.borrow_mut().vars.remove("coldRight");
            engine.interp.gc_collect();
            assert!(
                !crate::gc_test::alive(&weak),
                "removed imports release their exporter"
            );
        }
    }

    #[test]
    fn cold_environment_layout_preserves_values_across_order_and_structural_changes() {
        let cached = std::cell::OnceCell::new();
        let mut first = VarMap::default();
        let mut second = VarMap::default();
        for n in 0..24 {
            first.insert(
                format!("v{n}"),
                Binding::data(Value::Num(n as f64), true, true),
            );
            second.insert(
                format!("v{}", 23 - n),
                Binding::data(Value::Num((123 - n) as f64), true, true),
            );
        }
        let before = first.generation();
        first.share_layout(&cached);
        second.share_layout(&cached);
        assert!(
            first.generation() > before,
            "old raw pointers must fail before entries move"
        );
        let layout = first.layout_id();
        assert_ne!(layout, 0);
        assert_eq!(second.layout_id(), layout);
        for n in 0..24 {
            let name = format!("v{n}");
            let slot = first.binding_slot(&name).unwrap();
            assert_eq!(second.binding_slot(&name), Some(slot));
            assert!(
                matches!((second.fixed_binding_at(slot).unwrap().value()).view(), ValueView::Num(v) if v == (100+n) as f64)
            );
        }
        let slot = first.binding_slot("v23").unwrap();
        first.insert("v23", Binding::data(Value::Num(999.), false, false));
        assert_eq!(
            first.layout_id(),
            layout,
            "same names preserve the key proof"
        );
        let binding = first.fixed_binding_at(slot).unwrap();
        assert!(!binding.mutable && !binding.initialized);
        assert!(matches!((binding.value()).view(), ValueView::Num(999.)));
        first.insert("dynamic", Binding::data(Value::Num(7.), true, true));
        assert_eq!(first.layout_id(), 0);
        first.share_layout(&cached);
        assert_eq!(
            first.layout_id(),
            0,
            "a different key set must not adopt the layout"
        );
        assert!(first.get("v23").is_some());
        assert!(second.remove("v0").is_some());
        assert_eq!(second.layout_id(), 0);
        assert!(second.get("v0").is_none());
        second.clear();
        assert_eq!(second.keys().count(), 0);
    }

    fn evaluate(engine: &mut crate::Engine, source: &str) -> String {
        match engine
            .eval(source, false)
            .expect("environment fixture parses")
        {
            crate::Completion::Value(value) => value,
            crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn closure_env(engine: &mut crate::Engine, name: &str) -> Env {
        let global = engine.interp.global_env;
        let ValueView::Obj(object) = (engine
            .interp
            .get_var(name, &global)
            .ok()
            .expect("closure exists"))
        .view() else {
            panic!("expected function object");
        };
        let result = match &object.borrow().call {
            crate::value::Callable::User(user) => user.env,
            _ => panic!("expected user closure"),
        };
        result
    }

    #[test]
    fn cold_environment_closures_compile_later_without_recreating_bindings() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = crate::Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(u32::MAX);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                function make(seed) {
                    let a=seed,b=seed+1,c=seed+2,d=seed+3,e=seed+4,f=seed+5;
                    let g=seed+6,h=seed+7,j=seed+8,k=seed+9,l=seed+10,m=seed+11;
                    return {read:()=>a+b+c+d+e+f+g+h+j+k+l+m,write:()=>++a};
                }
                var first=make(1),second=make(10),one=first.read,two=second.read;
                one()===78 && two()===186
            "#
                ),
                "true"
            );
            let first = closure_env(&mut engine, "one");
            let second = closure_env(&mut engine, "two");
            if cold_environment_layouts_enabled() {
                assert_ne!(first.borrow().vars.layout_id(), 0);
                assert_eq!(
                    first.borrow().vars.layout_id(),
                    second.borrow().vars.layout_id()
                );
                assert!(first.borrow().vars.binding_slot("m").is_some());
            }
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                for(let n=0;n<40;n++) if((n&1?one:two)()!==(n&1?78:186)) throw 'wrong instance';
                first.write(); one()===79 && two()===186
            "#
                ),
                "true"
            );
            assert!(Env::ptr_eq(&first, &closure_env(&mut engine, "one")));
            first
                .borrow_mut()
                .vars
                .insert("injected", Binding::data(Value::Num(7.), true, true));
            assert_eq!(first.borrow().vars.layout_id(), 0);
            assert_eq!(evaluate(&mut engine, "one()===79 && two()===186"), "true");
            first.borrow_mut().vars.get_mut("m").unwrap().initialized = false;
            assert_eq!(
                evaluate(&mut engine, "try {one(); 'bad'} catch(e) {e.name}"),
                "ReferenceError"
            );
        }
    }

    #[test]
    fn cold_environment_layouts_preserve_parameter_eval_arguments_and_tdz() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = crate::Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                evaluate(
                    &mut engine,
                    r#"
                function defaults(a, read=()=>a) { var a=9; return [read,()=>a]; }
                function mapped(a) { var get=()=>a; arguments[0]=7; return [get,()=>++a,arguments]; }
                function dynamic(a) { var get=()=>a+extra; eval('var extra=5'); return [get,()=>eval('extra=8')]; }
                function early() { var read=()=>late; try { read(); } catch(e) { return e.name; } let late=1; }
                function immutable() { const value=3; return ()=>{try {value=4} catch(e) {return e.name;}}; }
                var d=defaults(2),m=mapped(1),e=dynamic(4),trace=[d[0](),d[1](),m[0]()];
                m[1](); trace.push(m[2][0],e[0]()); e[1](); trace.push(e[0](),early(),immutable()());
                var obj={x:5},read; with(obj) {read=()=>x;} obj.x=6; trace.push(read());
                trace.join('|')
            "#
                ),
                "2|9|7|8|9|12|ReferenceError|TypeError|6",
                "{tier:?}"
            );
        }
    }

    #[test]
    fn binding_layout_invalidates_every_structural_mutation_but_not_live_writes() {
        let mut vars = VarMap::default();
        let id = new_binding_layout_id();
        assert_ne!(id, 0);
        vars.insert("value", Binding::data(Value::Num(1.0), true, true));
        vars.publish_layout(id);
        vars.get_mut("value").unwrap().set_value(Value::Num(2.0));
        assert_eq!(vars.layout_id(), id);
        vars.get_mut("value").unwrap().initialized = false;
        assert_eq!(vars.layout_id(), id, "TDZ is checked live");
        vars.insert("value", Binding::data(Value::Num(3.0), true, true));
        assert_eq!(vars.layout_id(), 0, "even same-key replacement invalidates");
        vars.publish_layout(id);
        vars.remove("missing");
        assert_eq!(vars.layout_id(), 0);
        vars.publish_layout(id);
        vars.clear();
        assert_eq!(vars.layout_id(), 0);
        vars.publish_layout(id);
        for n in 0..20 {
            vars.insert(
                n.to_string(),
                Binding::data(Value::Num(n as f64), true, true),
            );
        }
        assert_eq!(
            vars.layout_id(),
            0,
            "small-to-large growth also invalidates"
        );
    }

    #[test]
    fn binding_layout_id_exhaustion_never_reuses_old_id() {
        let next = std::sync::atomic::AtomicU32::new(u32::MAX - 1);
        assert_eq!(allocate_binding_layout_id(&next), u32::MAX - 1);
        assert_eq!(allocate_binding_layout_id(&next), 0);
        assert_eq!(allocate_binding_layout_id(&next), 0);
    }
}

pub struct Scope {
    pub vars: VarMap,
    pub parent: Option<Env>,
    /// For a `with (obj)` block: identifier resolution checks `obj`'s properties before the parent.
    with_obj: Value,
    /// `true` when this scope or ANY ancestor is a `with` scope (propagated at construction).
    /// Compiled tiers refuse to run a body whose closure chain is under a `with`: the VM's
    /// free-name ops re-resolve at execution time, but a with-object binding can appear or
    /// vanish between the spec's reference resolution and the store (e.g. `x = (delete env.x,
    /// 0)`) — the tree-walker models that; the ops don't.
    pub under_with: bool,
    /// `true` if this is a *variable* environment (a function body scope, the global scope, or an
    /// `eval` scope) — the target for `var`/function hoisting. Block, `with`, and function *parameter*
    /// scopes are `false`, so a sloppy direct `eval` hoists its vars past them into the nearest
    /// enclosing variable environment (see EvalDeclarationInstantiation).
    pub var_boundary: bool,
    /// `true` for a `catch` clause's parameter environment. A sloppy direct `eval`'s
    /// EvalDeclarationInstantiation walk skips it, so `eval("var e")` inside `catch (e) { … }` is
    /// allowed (the web-compatibility carve-out for VariableStatements in catch blocks).
    pub catch_param: bool,
    /// Names declared *lexically* at this scope's own level (let/const/using/class). A function
    /// body scope holds both hoisted vars and body-level lexicals; a sloppy direct eval's
    /// var/lexical conflict check needs to tell them apart.
    pub lexical_names: ScopeNames,
}

impl Scope {
    /// A with record is present exactly when its word is not EMPTY. Presence is independent
    /// of the held value's truthiness or kind (including synthetic Null records in tests).
    #[inline]
    pub(crate) fn has_with_object(&self) -> bool {
        !self.with_obj.is_empty()
    }

    #[inline]
    pub(crate) fn with_object(&self) -> Option<Value> {
        self.has_with_object().then_some(self.with_obj)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn set_with_object(&mut self, object: Option<Value>) {
        debug_assert!(!object.is_some_and(|value| matches!(value.view(), ValueView::Empty)));
        self.with_obj = object.unwrap_or(Value::Empty);
    }

    pub(crate) const fn with_object_offset() -> usize {
        std::mem::offset_of!(Self, with_obj)
    }
}

impl crate::gc_trace::GcTrace for Scope {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        self.trace_fields(sink);
    }
}

impl Scope {
    /// [`GcTrace`](crate::gc_trace::GcTrace)'s edges, for any sink (the marker's among them).
    #[inline]
    pub(crate) fn trace_fields<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        let Self {
            vars,
            parent,
            with_obj,
            under_with: _,
            var_boundary: _,
            catch_param: _,
            lexical_names,
        } = self;
        vars.trace_storage(sink);
        lexical_names.trace(sink);
        for binding in vars.values() {
            binding.trace_edges(sink);
        }
        if let Some(parent) = parent {
            sink.env(parent);
        }
        sink.value(with_obj);
    }
}

/// P2 W2b: an environment cell, with its first binding's value in place and its names in one
/// word, fits the heap's 96-byte size class (P1 S4's, with the name in the entry, was 128).
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<std::cell::RefCell<Scope>>() <= 96);

/// Lexical-name bookkeeping is only populated by direct eval, but every call allocates a scope.
/// Keep the common empty case to one nullable pointer and materialize the list on first use, in
/// a storage block: the names are atoms, the list is never dropped, and the record reports its
/// blocks ([`ScopeNames::trace`]), as it does its entries'.
#[derive(Default)]
#[repr(transparent)]
pub struct ScopeNames(Option<crate::gc_storage::StorageBox<ScopeNameList>>);

type ScopeNameList = std::mem::ManuallyDrop<
    crate::inline_vec::InlineVec<crate::atom::Atom, 1, crate::gc_storage::GcSpill>,
>;

impl ScopeNames {
    /// Record a lexically declared name.
    pub(crate) fn push(&mut self, name: impl crate::atom::IntoAtom) {
        self.0
            .get_or_insert_with(|| crate::gc_storage::StorageBox::new(ScopeNameList::default()))
            .push(name.into_atom());
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = crate::atom::Atom> + '_ {
        self.0
            .as_deref()
            .map_or(&[][..], |names| &names[..])
            .iter()
            .copied()
    }

    /// Report the list's blocks.
    pub(crate) fn trace<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(names) = &self.0 {
            crate::gc_storage::StorageBox::trace(names, sink);
            if let Some(block) = names.spill_ptr() {
                crate::gc_storage::trace_block(block, sink);
            }
        }
    }

    pub(crate) fn retained_requested_storage_bytes(&self) -> usize {
        let Some(names) = self.0.as_deref() else {
            return 0;
        };
        // The names are atoms, which their heap's table accounts.
        std::mem::size_of::<ScopeNameList>().saturating_add(
            names
                .heap_capacity()
                .saturating_mul(std::mem::size_of::<crate::atom::Atom>()),
        )
    }
}

#[derive(Clone)]
pub struct Binding {
    /// Execution word only; initialization remains an independent flag.
    value: Value,
    pub mutable: bool,
    /// `false` while a `let`/`const` is in its temporal dead zone.
    pub initialized: bool,
    /// A live module import: reads redirect to `(exporter scope, local name)`, kept in a
    /// storage block of its own, which the binding's record reports: ordinary bindings need
    /// only a nullable pointer, and the record nothing to drop.
    pub(crate) import_ref: Option<crate::gc_storage::StorageBox<(Env, crate::atom::Atom)>>,
    /// Native-visible kind bit, occupying existing flag padding. All structural insertion
    /// paths normalize it; in-place import changes must use set_import_reference.
    pub(crate) imported: bool,
    /// `true` for a `var`/function binding created by a sloppy `eval` (CreateMutableBinding with
    /// `deletable` set): `delete <name>` may remove it, unlike ordinary declarations.
    pub deletable: bool,
    /// For an *immutable* binding (`mutable == false`): whether it was created strict. A `const`
    /// (CreateImmutableBinding(_, true)) always throws on reassignment; a named function
    /// expression's own name (CreateImmutableBinding(_, false)) is a silent no-op in sloppy code
    /// and throws only under strict mode. Irrelevant when `mutable` is true.
    pub strict_immutable: bool,
}

/// A bounded, environment-specific proof used by trivial accessor functions whose whole body is
/// `return <identifier>`. Weak pins describe the live path from the function's captured
/// environment through the binding's scope, prevent scope-allocation ABA, and retain no scope or
/// value. Each structural generation validates the cached Binding pointer.
#[derive(Clone)]
pub(crate) struct BindingGetterScopeGuard {
    pub(crate) pin: crate::interpreter::WeakEnv,
    pub(crate) generation: u32,
}

#[derive(Clone)]
pub(crate) struct BindingGetterCache {
    pub(crate) name: Rc<str>,
    pub(crate) scopes: Vec<BindingGetterScopeGuard>,
    pub(crate) binding: *const Binding,
}

impl BindingGetterCache {
    pub(crate) fn retained_requested_storage_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(
            self.scopes
                .capacity()
                .saturating_mul(std::mem::size_of::<BindingGetterScopeGuard>()),
        )
    }
}

/// Keep getter binding caches shallow and deterministic in size; deeper lexical chains retain
/// the ordinary Call path.
pub(crate) const MAX_BINDING_GETTER_SCOPES: usize = 16;

impl Binding {
    #[inline]
    pub(crate) fn value(&self) -> Value {
        self.value
    }

    #[inline]
    pub(crate) fn set_value(&mut self, value: Value) {
        self.value = value;
    }

    pub(crate) const fn value_offset() -> usize {
        std::mem::offset_of!(Self, value)
    }

    #[cfg(test)]
    pub(crate) fn set_import_reference(&mut self, reference: Option<(Env, crate::atom::Atom)>) {
        self.imported = reference.is_some();
        self.import_ref = reference.map(crate::gc_storage::StorageBox::new);
    }
    pub(crate) fn data(value: Value, mutable: bool, initialized: bool) -> Binding {
        Binding {
            value,
            mutable,
            initialized,
            import_ref: None,
            imported: false,
            deletable: false,
            // An immutable binding created through this helper (const/TDZ) is strict by default.
            strict_immutable: !mutable,
        }
    }

    pub(crate) fn with_strict_immutable(mut self, strict: bool) -> Self {
        self.strict_immutable = strict;
        self
    }

    pub(crate) fn live_import(environment: Env, name: crate::atom::Atom) -> Self {
        let mut binding = Self::data(Value::Undefined, false, true);
        binding.import_ref = Some(crate::gc_storage::StorageBox::new((environment, name)));
        binding.imported = true;
        binding
    }
}

impl crate::gc_trace::GcTrace for Binding {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        self.trace_edges(sink);
    }
}

impl Binding {
    /// [`GcTrace`](crate::gc_trace::GcTrace)'s edges, for any sink.
    #[inline]
    fn trace_edges<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        let Self {
            value,
            mutable: _,
            initialized: _,
            import_ref,
            imported: _,
            deletable: _,
            strict_immutable: _,
        } = self;
        sink.value(value);
        if let Some(import) = import_ref {
            crate::gc_storage::StorageBox::trace(import, sink);
            sink.env(&import.0);
        }
    }
}

/// A new environment record.
#[inline]
fn new_env(
    vars: VarMap,
    parent: Option<Env>,
    with_obj: Option<Value>,
    under_with: bool,
    var_boundary: bool,
    catch_param: bool,
) -> Env {
    Env::new(Scope {
        vars,
        parent,
        with_obj: (with_obj.unwrap_or(Value::Empty)),
        under_with,
        var_boundary,
        catch_param,
        lexical_names: ScopeNames::default(),
    })
}

pub fn new_scope(parent: Option<Env>) -> Env {
    let under_with = parent.as_ref().is_some_and(|p| p.borrow().under_with);
    new_env(Default::default(), parent, None, under_with, false, false)
}

/// A *variable* environment: the hoisting target for `var`/function declarations (function body,
/// global, or `eval` scope). See [`Scope::var_boundary`].
pub fn new_var_scope(parent: Option<Env>) -> Env {
    new_var_scope_with_capacity(parent, 0)
}

pub(crate) fn new_var_scope_with_capacity(parent: Option<Env>, capacity: usize) -> Env {
    new_var_scope_with_bindings(parent, VarMap::with_capacity(capacity))
}

pub(crate) fn new_var_scope_with_bindings(parent: Option<Env>, vars: VarMap) -> Env {
    let under_with = parent.as_ref().is_some_and(|p| p.borrow().under_with);
    new_env(vars, parent, None, under_with, true, false)
}

/// A `catch (e)` parameter environment: like a block scope, but flagged so a sloppy direct `eval`'s
/// var-hoisting walk skips it (see [`Scope::catch_param`]).
pub fn new_catch_scope(parent: Env) -> Env {
    let under_with = parent.borrow().under_with;
    new_env(
        Default::default(),
        Some(parent),
        None,
        under_with,
        false,
        true,
    )
}

/// A `with (obj)` environment: identifier lookups consult `obj` before the enclosing scope.
pub fn new_with_scope(parent: Env, obj: Value) -> Env {
    new_env(
        Default::default(),
        Some(parent),
        Some(obj),
        true,
        false,
        false,
    )
}

/// Walk up from `env` to the nearest variable environment (function body, global, or `eval` scope) —
/// the scope a `var`/function declaration hoists into. Falls back to the outermost scope.
pub fn nearest_var_env(env: &Env) -> Env {
    let mut cur = *env;
    loop {
        if cur.borrow().var_boundary {
            return cur;
        }
        let parent = cur.borrow().parent;
        match parent {
            Some(p) => cur = p,
            None => return cur,
        }
    }
}

/// A non-local completion. Expressions only raise `Throw`; the rest flow out of statements.
pub(crate) enum Abrupt {
    Throw(Value),
    /// Host control completion. It is deliberately not a thrown Value: HTML script termination
    /// bypasses author `catch` and `finally` blocks.
    Interrupt(crate::InterruptReason),
    Return(Value),
    /// Break/Continue carry the completion value threaded so far by the enclosing statement list
    /// (`Value::Empty` when none), per the spec's UpdateEmpty bookkeeping.
    Break(Option<String>, Value),
    Continue(Option<String>, Value),
}

pub(crate) type Completion = Result<Value, Abrupt>;

/// UpdateEmpty: replace an EMPTY statement-completion value with `undefined`.
pub(crate) fn update_empty(v: Value) -> Value {
    match v.view() {
        ValueView::Empty => Value::Undefined,
        _ => v,
    }
}

/// UpdateEmpty for an abrupt completion: fill an empty break/continue value with `v`.
pub(crate) fn update_abrupt_empty(a: Abrupt, v: Value) -> Abrupt {
    match a {
        Abrupt::Break(l, old) if matches!(old.view(), ValueView::Empty) => Abrupt::Break(l, v),
        Abrupt::Continue(l, old) if matches!(old.view(), ValueView::Empty) => {
            Abrupt::Continue(l, v)
        }
        other => other,
    }
}

/// Extract the thrown value from an abrupt completion (non-throw completions surface as undefined).
pub(crate) fn abrupt_value(a: Abrupt) -> Value {
    match a {
        Abrupt::Throw(v) => v,
        _ => Value::Undefined,
    }
}

/// How [`Interp::bind_pattern`] should bind the identifiers it reaches.
#[derive(Clone, Copy)]
pub enum BindMode {
    /// `var` — assign to the (already-hoisted) function-scoped binding.
    Var,
    /// `let`/`const` — create a fresh lexical binding (`true` = const).
    Lexical(bool),
}

/// Top-level lexically-declared names of a block body (`let`/`const`/`class`) — used by Annex B.3.3
/// to decide whether a synthesized block-function var binding would conflict.
fn block_lexical_names(stmts: &[Stmt]) -> Vec<String> {
    let mut out = Vec::new();
    for s in stmts {
        match s {
            Stmt::VarDecl {
                kind: DeclKind::Let | DeclKind::Const,
                decls,
            } => {
                for (pat, _) in decls {
                    pattern_idents(pat, &mut out);
                }
            }
            Stmt::ClassDecl(class) => {
                if let Some(n) = &class.name {
                    out.push(n.clone());
                }
            }
            _ => {}
        }
    }
    out
}

/// BoundNames of a formal parameter list.
pub(crate) fn param_bound_names(params: &[Param]) -> Vec<String> {
    let mut out = Vec::new();
    for p in params {
        pattern_idents(&p.pattern, &mut out);
    }
    out
}

/// Push `added` names onto the Annex B blocked list (dedup'd); returns how many were pushed so
/// the caller can truncate back when leaving the scope.
fn push_blocked(blocked: &mut Vec<String>, added: Vec<String>) -> usize {
    let mut pushed = 0;
    for x in added {
        if !blocked.iter().any(|b| b == &x) {
            blocked.push(x);
            pushed += 1;
        }
    }
    pushed
}

/// Unwrap an `export <decl>` / `export default <decl>` to the declaration it wraps (so the hoisting
/// and lexical-declaration passes treat them like ordinary declarations).
pub fn unwrap_export(stmt: &Stmt) -> &Stmt {
    match stmt {
        Stmt::ExportDecl(inner) | Stmt::ExportDefault(inner) => inner,
        other => other,
    }
}

/// Collect every identifier bound by a pattern (for `var` hoisting and TDZ pre-declaration).
/// [`pattern_idents`] borrowing the AST's names: text at a stable address, which the atom
/// table's recent lookups serve.
pub(crate) fn pattern_ident_strs<'a>(pat: &'a Pattern, out: &mut Vec<&'a str>) {
    match pat {
        Pattern::Ident(n) => out.push(n),
        Pattern::Array(elems) => {
            for e in elems {
                match e {
                    ArrayPatElem::Hole => {}
                    ArrayPatElem::Elem { pattern, .. } => pattern_ident_strs(pattern, out),
                    ArrayPatElem::Rest(p) => pattern_ident_strs(p, out),
                }
            }
        }
        Pattern::Object(o) => {
            for p in &o.props {
                pattern_ident_strs(&p.value, out);
            }
            if let Some(r) = &o.rest {
                out.push(r);
            }
        }
        Pattern::Member(_) => {}
    }
}

pub fn pattern_idents(pat: &Pattern, out: &mut Vec<String>) {
    match pat {
        Pattern::Ident(n) => out.push(n.clone()),
        Pattern::Array(elems) => {
            for e in elems {
                match e {
                    ArrayPatElem::Hole => {}
                    ArrayPatElem::Elem { pattern, .. } => pattern_idents(pattern, out),
                    ArrayPatElem::Rest(p) => pattern_idents(p, out),
                }
            }
        }
        Pattern::Object(o) => {
            for p in &o.props {
                pattern_idents(&p.value, out);
            }
            if let Some(r) = &o.rest {
                out.push(r.clone());
            }
        }
        Pattern::Member(_) => {}
    }
}

/// Whether a formal parameter list "contains an expression" (ECMAScript ContainsExpression): a
/// default initializer or a destructuring pattern with a default/computed key. When true, the callee
/// gets a separate parameter Environment Record distinct from the body's variable environment, so a
/// direct `eval` in a parameter default cannot leak bindings into the body (and `arguments`/params
/// live in a scope the body's `var` hoisting sits below).
pub fn params_have_expr(params: &[Param]) -> bool {
    params
        .iter()
        .any(|p| p.default.is_some() || pattern_has_expr(&p.pattern))
}

fn pattern_has_expr(pat: &Pattern) -> bool {
    match pat {
        Pattern::Ident(_) | Pattern::Member(_) => false,
        Pattern::Array(elems) => elems.iter().any(|e| match e {
            ArrayPatElem::Hole => false,
            ArrayPatElem::Elem { pattern, default } => {
                default.is_some() || pattern_has_expr(pattern)
            }
            ArrayPatElem::Rest(p) => pattern_has_expr(p),
        }),
        Pattern::Object(o) => o.props.iter().any(|p| {
            p.default.is_some()
                || matches!(p.key, PropKey::Computed(_))
                || pattern_has_expr(&p.value)
        }),
    }
}

/// One megamorphic-stub-cache entry (see [`Interp::stub_cache`]): the derived resolution for a
/// (receiver shape, site key) pair. The key is kept separately in `stub_cache_names`.
#[derive(Clone, Copy)]
pub(crate) struct StubEntry {
    pub(crate) st: crate::bytecode::IcState,
}

impl Default for StubEntry {
    fn default() -> StubEntry {
        StubEntry {
            st: crate::bytecode::IcState {
                recv_shape: 0,
                holder_shape: 0,
                slot: 0,
                depth: crate::bytecode::IC_EMPTY,
                mid_ok: 0,
                mid_shape: 0,
                mid2_shape: 0,
                mid3_shape: 0,
                mid4_shape: 0,
            },
        }
    }
}

/// Stub-cache capacity (power of two). 4096 × 24-byte entries = 96 KiB.
const STUB_CACHE_SIZE: usize = 4096;
const COMPUTED_STUB_KEY_MAX_BYTES: usize = 256;

#[path = "property_read.rs"]
mod property_read;
use property_read::CachedGet;

#[cfg(test)]
mod deep_property_ic_tests {
    use super::*;
    use crate::bytecode::{IcState, IC_ABSENT, PROP_IC_WAYS};
    use std::cell::Cell;

    #[test]
    fn deep_property_cache_validates_every_live_prototype() {
        let mut interp = Interp::new();
        let base = Object::new(None);
        let mut receiver = base;
        let mut levels = vec![base];
        for _ in 0..5 {
            receiver = Object::new(Some(receiver));
            levels.push(receiver);
        }
        let object = Value::Obj(receiver);
        let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        let read = |interp: &mut Interp| {
            interp
                .get_prop_ic(
                    &object,
                    "optional",
                    crate::atom::PropKey::from_text("optional"),
                    &cache[0],
                )
                .unwrap_or_else(|_| panic!("ordinary read threw"))
        };
        assert!(matches!(
            (read(&mut interp)).view(),
            crate::value::ValueView::Undefined
        ));
        assert_eq!(cache[0].get().depth, IC_ABSENT);
        assert_eq!(cache[0].get().slot, 6);
        assert!(matches!(
            (read(&mut interp)).view(),
            crate::value::ValueView::Undefined
        ));
        base.borrow_mut()
            .props
            .insert("optional", Property::plain(Value::Num(7.0)));
        assert!(matches!(
            (read(&mut interp)).view(),
            crate::value::ValueView::Num(7.0)
        ));
        assert_eq!(cache[0].get().depth, 5);
        assert!(matches!(
            (read(&mut interp)).view(),
            crate::value::ValueView::Num(7.0)
        ));
        levels[2]
            .borrow_mut()
            .props
            .insert("optional", Property::plain(Value::Num(11.0)));
        assert!(matches!(
            (read(&mut interp)).view(),
            crate::value::ValueView::Num(11.0)
        ));
        // Identical layout, different prototype identity/value: the IC must follow it live.
        let replacement = Object::new(None);
        replacement
            .borrow_mut()
            .props
            .insert("optional", Property::plain(Value::Num(19.0)));
        levels[3].borrow_mut().proto = Some(replacement);
        assert!(matches!(
            (read(&mut interp)).view(),
            crate::value::ValueView::Num(19.0)
        ));
        assert_eq!(std::mem::size_of::<IcState>(), 32);
        assert_eq!(
            std::mem::offset_of!(IcState, mid3_shape),
            crate::bytecode::IC_OFF_MID3_SHAPE as usize
        );
        assert_eq!(
            std::mem::offset_of!(IcState, mid4_shape),
            crate::bytecode::IC_OFF_MID4_SHAPE as usize
        );
    }
}

/// The table index for a (receiver shape, key) pair. An atom key's low bits are alignment
/// zeros; shift them off before mixing so they contribute entropy.
#[inline(always)]
fn stub_slot(shape: u32, key: crate::atom::PropKey) -> usize {
    let n = key.word() as usize >> 3;
    (shape as usize ^ n ^ (n >> 7)) & (STUB_CACHE_SIZE - 1)
}

impl Interp {
    /// Probe the megamorphic property cache. An entry holds its key, a word that names one
    /// property for the heap's whole life.
    #[inline]
    fn stub_cache_get(&self, shape: u32, key: crate::atom::PropKey) -> Option<StubEntry> {
        let slot = stub_slot(shape, key);
        let same_key = self.stub_cache_names.borrow().get(slot) == Some(&Some(key));
        same_key.then(|| self.stub_cache[slot].get())
    }

    /// Store a stub resolution and its key.
    #[inline]
    fn stub_cache_set(&self, shape: u32, key: crate::atom::PropKey, st: crate::bytecode::IcState) {
        if !crate::value::is_cacheable_shape(shape) || !st.has_cacheable_shapes() {
            return;
        }
        let slot = stub_slot(shape, key);
        self.stub_cache_names.borrow_mut()[slot] = Some(key);
        self.stub_cache[slot].set(StubEntry { st });
    }
}

#[cfg(test)]
#[path = "uncacheable_shape_tests.rs"]
mod uncacheable_shape_tests;

#[cfg(test)]
#[path = "ordinary_data_read_tests.rs"]
mod ordinary_data_read_tests;

#[cfg(test)]
#[path = "property_read_tests.rs"]
mod property_read_tests;

#[cfg(test)]
#[path = "constructor_transfer_tests.rs"]
mod constructor_transfer_tests;

#[cfg(test)]
mod computed_key_cache_tests {
    use super::*;
    use crate::value::ValueView;

    /// The stub cache keys computed resolutions by their name text, not the key's address: a key
    /// allocation reused for another name misses. Only the text and validated layout metadata
    /// are kept, and a key too large to keep is not cached.
    #[test]
    fn computed_key_stub_cache_owns_text_and_rejects_reused_addresses() {
        let mut engine = crate::Engine::new();
        let i = &mut engine.interp;
        let object = Object::new(None);
        object
            .borrow_mut()
            .props
            .insert("alpha", Property::plain(Value::Num(7.0)));
        object
            .borrow_mut()
            .props
            .insert("brava", Property::plain(Value::Num(9.0)));
        let shape = object.borrow().props.shape();
        let base = Value::Obj(object);
        let cells = [const { std::cell::Cell::new(crate::bytecode::IcState::EMPTY) };
            crate::bytecode::PROP_IC_WAYS];
        let mut key = String::from("alpha");
        let address = key.as_ptr();
        let read = i.get_prop_keyed(&base, &key, &cells[0]);
        assert!(matches!(
            (read).map(|__value| (__value).view()),
            Ok(ValueView::Num(7.0))
        ));
        let atom = |text: &str| crate::atom::PropKey::from_text(text);
        assert!(
            i.stub_cache_get(shape, atom(&key)).is_some(),
            "computed reads must populate the stub cache"
        );

        // Reuse exactly the same key allocation for a different property: the stub cache is
        // keyed by the property's key, not the text's address, so alpha's resolution stays put.
        key.replace_range(.., "brava");
        assert_eq!(address, key.as_ptr());
        assert!(i.stub_cache_get(shape, atom(&key)).is_none());
        for cell in &cells {
            cell.set(crate::bytecode::IcState::EMPTY);
        }
        let read = i.get_prop_keyed(&base, &key, &cells[0]);
        assert!(matches!(
            (read).map(|__value| (__value).view()),
            Ok(ValueView::Num(9.0))
        ));
        assert!(i.stub_cache_get(shape, atom(&key)).is_some());
        let computed_key = crate::lstr::LStr::from("alpha");
        assert!(matches!(
            (i.get_computed_property(&base, &computed_key)).map(|__value| (__value).view()),
            Ok(ValueView::Num(7.0))
        ));

        let long_key = "k".repeat(COMPUTED_STUB_KEY_MAX_BYTES + 1);
        object
            .borrow_mut()
            .props
            .insert(long_key.as_str(), Property::plain(Value::Num(11.0)));
        for cell in &cells {
            cell.set(crate::bytecode::IcState::EMPTY);
        }
        let read = i.get_prop_keyed(&base, &long_key, &cells[0]);
        assert!(matches!(
            (read).map(|__value| (__value).view()),
            Ok(ValueView::Num(11.0))
        ));
        assert!(
            i.stub_cache_get(object.borrow().props.shape(), atom(&long_key))
                .is_none(),
            "large transient keys must not accumulate in the stub table"
        );
    }
}

/// Runtime-probed byte offsets from an `Interp`'s base address to the fields the asm call
/// thunk touches (see the asm-call-frames arc). `Interp` is not `repr(C)`, so nothing here is
/// compile-time stable — every offset is measured against a live instance, and `valid` fails
/// closed. Vec word offsets (`ptr`/`len`/`cap` order) are probed per instantiation, same as
/// [`crate::value::JitLayout`]'s.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[allow(dead_code)] // consumed incrementally as the asm thunk lands
pub(crate) struct InterpLayout {
    pub depth: usize,
    pub direct_call_depth: usize,
    pub gc_tick: usize,
    pub gc_next: usize,
    pub interrupt_poll_tick: usize,
    pub call_site: usize,
    pub constructing: usize,
    /// Ambient mode used by semantic helpers; every native function body saves/restores it.
    pub strict: usize,
    pub new_target: usize,
    pub pending_tail: usize,
    /// `fn_frames` Vec base, plus its ptr/len/cap word offsets within the Vec header.
    pub fn_frames: usize,
    pub fnf_ptr_word: usize,
    pub fnf_len_word: usize,
    pub fnf_cap_word: usize,
    /// The `frame_pool` freelist (`FramePool::pool`, a Vec<NonNull<JitFrame>>) base + its word
    /// offsets.
    pub frame_pool: usize,
    pub fp_ptr_word: usize,
    pub fp_len_word: usize,
    pub fp_cap_word: usize,
    /// `string_proto` (a `Gc`: one stored cell address): the GetMethod template resolves
    /// string-primitive receivers against the ACTIVE realm's String.prototype through this.
    pub string_proto: usize,
    /// `number_proto` / `boolean_proto`: Number/Boolean-primitive receivers in the
    /// non-consuming property-load forms resolve against the ACTIVE realm's wrapper
    /// prototypes through these (both fields are realm-swapped with `string_proto`).
    pub number_proto: usize,
    pub boolean_proto: usize,
    /// Canonical Function.prototype, used to validate the ordinary `instanceof` fast path.
    pub function_proto: usize,
    pub call_overflow_sets: usize,
    pub call_overflow_shift: usize,
    pub computed_read_sets: usize,
    pub computed_read_mask: usize,
    pub valid: bool,
}

/// Probe the ptr/len/cap word order of a `Vec<T>` header holding at least one element whose
/// data pointer and length are known. Returns (ptr_word, len_word, cap_word) or None.
fn probe_vec_words<T>(v: &Vec<T>, len: usize, cap: usize) -> Option<(usize, usize, usize)> {
    let data = v.as_ptr() as usize;
    let words: [usize; 3] = unsafe { std::mem::transmute_copy::<Vec<T>, [usize; 3]>(v) };
    let mut ptr_w = None;
    let mut len_w = None;
    let mut cap_w = None;
    for (k, w) in words.iter().enumerate() {
        if *w == data && ptr_w.is_none() {
            ptr_w = Some(k * 8);
        } else if *w == len && len_w.is_none() {
            len_w = Some(k * 8);
        } else if *w == cap && cap_w.is_none() {
            cap_w = Some(k * 8);
        }
    }
    Some((ptr_w?, len_w?, cap_w?))
}

pub(crate) fn interp_layout(i: &mut Interp) -> InterpLayout {
    let base = i as *mut Interp as usize;
    let off = |p: usize| p - base;
    // Distinguishable len/cap for the Vec probes: len 3, cap ≥ 4 and != 3.
    let saved_frames = std::mem::take(&mut i.fn_frames);
    i.fn_frames = Vec::with_capacity(7);
    for k in 0..3 {
        i.fn_frames.push(FnFrame {
            fn_ptr: 0x1000 + k,
            call_site: 0,
            strict: false,
            construct: false,
            extra: None,
        });
    }
    let fnf = probe_vec_words(&i.fn_frames, 3, i.fn_frames.capacity());
    i.fn_frames = saved_frames;
    let saved_pool = std::mem::replace(&mut i.frame_pool.pool, Vec::with_capacity(7));
    for _ in 0..3 {
        i.frame_pool
            .pool
            .push(std::ptr::NonNull::new(0x2000 as *mut crate::jit::JitFrame).unwrap());
    }
    let fp = probe_vec_words(&i.frame_pool.pool, 3, i.frame_pool.pool.capacity());
    i.frame_pool.pool.clear();
    i.frame_pool.pool = saved_pool;
    let (Some(fnf), Some(fp)) = (fnf, fp) else {
        return InterpLayout::default(); // valid: false
    };
    InterpLayout {
        depth: off(&i.depth as *const _ as usize),
        direct_call_depth: off(&i.direct_call_depth as *const _ as usize),
        gc_tick: off(&i.gc_tick as *const _ as usize),
        gc_next: off(&i.gc_next as *const _ as usize),
        interrupt_poll_tick: off(&i.interrupt_poll_tick as *const _ as usize),
        call_site: off(&i.call_site as *const _ as usize),
        constructing: off(&i.constructing as *const _ as usize),
        strict: off(&i.strict as *const _ as usize),
        new_target: off(&i.new_target as *const _ as usize),
        pending_tail: off(&i.pending_tail as *const _ as usize),
        fn_frames: off(&i.fn_frames as *const _ as usize),
        fnf_ptr_word: fnf.0,
        fnf_len_word: fnf.1,
        fnf_cap_word: fnf.2,
        frame_pool: off(&i.frame_pool.pool as *const _ as usize),
        fp_ptr_word: fp.0,
        fp_len_word: fp.1,
        fp_cap_word: fp.2,
        string_proto: off(&i.string_proto as *const _ as usize),
        number_proto: off(&i.number_proto as *const _ as usize),
        boolean_proto: off(&i.boolean_proto as *const _ as usize),
        function_proto: off(&i.function_proto as *const _ as usize),
        call_overflow_sets: off(&i.call_overflow.raw_sets as *const _ as usize),
        call_overflow_shift: off(&i.call_overflow.hash_shift as *const _ as usize),
        computed_read_sets: off(&i.computed_reads.raw_sets as *const _ as usize),
        computed_read_mask: off(&i.computed_reads.mask as *const _ as usize),
        valid: true,
    }
}

struct ConstructIc {
    call: crate::bytecode::CallIc,
    pin: crate::value::WeakGc,
    /// Shape and own-entry slot of the constructor's `prototype` data property. The slot's
    /// value is deliberately read live on every `new`, so `F.prototype = other` remains visible
    /// without invalidation; the shape guard only removes the repeated string-map lookup.
    prototype_shape: u32,
    prototype_slot: u32,
    /// The constructor's immutable chunk has the exact Prototype.js
    /// `this.initialize.apply(this, arguments)` forwarding body.
    arguments_apply_forwarder: bool,
}

pub struct Interp {
    /// Collector ownership follows the ECMAScript Agent, not the native thread currently running
    /// one of its suspended execution contexts. All realms in this interpreter share this heap.
    pub(crate) gc_heap: crate::value::GcHeap,
    pub(crate) realm_id: crate::agent_access::RealmId,
    /// ECMA-262 Agent Record symbol state, shared by all realms including ShadowRealms.
    pub(crate) symbol_agent: crate::value::SymbolAgent,
    pub(crate) global: Gc,
    pub(crate) global_env: Env,
    pub(crate) object_proto: Gc,
    pub(crate) function_proto: Gc,
    pub(crate) array_proto: Gc,
    /// This realm's intrinsic `%Array%` constructor. Kept separately from mutable public
    /// properties so cross-realm fast paths can prove the constructor's actual realm identity.
    pub(crate) array_ctor: Option<Gc>,
    pub(crate) string_proto: Gc,
    pub(crate) number_proto: Gc,
    pub(crate) boolean_proto: Gc,
    pub(crate) symbol_proto: Gc,
    pub(crate) error_protos: IntrinsicMap,
    pub(crate) console: Vec<String>,
    /// Realm-local implementation of HTML `HostSystemUTCEpochNanoseconds(global)`, expressed in
    /// milliseconds for ECMAScript `Date`. Each Window/Worker realm owns its own time origin and
    /// virtual-time anchor; a process-global callback cannot model that.
    pub(crate) wall_clock: Option<Rc<dyn Fn() -> f64>>,
    pub(crate) runtime_interrupt: std::sync::Arc<crate::RuntimeInterrupt>,
    /// Hot-loop poll divider. Execution tiers increment this cheaply and consult the shared
    /// atomics/deadline only once per bounded batch.
    pub(crate) interrupt_poll_tick: u32,
    /// The tree-walking evaluator's stack-gateway divider: it queries the native stack once per
    /// eight evaluations, so at least once every eight levels of nesting (see `Interp::eval`).
    pub(crate) ast_tick: u32,
    /// Current strict-mode flag (pushed/popped around function bodies).
    pub(crate) strict: bool,
    /// Execution tier (env `TRUST_JS_TIER`, CLI `--tier`, [`Engine::set_tier`]). `Jit` is the
    /// default; `Interp` is the reference tree-walker, opt-in via `TRUST_JS_TIER=interp`.
    pub(crate) tier: crate::bytecode::Tier,
    /// Calls before an eligible function tier-ups to bytecode (env `TRUST_JS_TIER_THRESHOLD`).
    pub(crate) tier_threshold: u32,
    /// Bytecode VM runs a loop-free function makes before it is compiled to native code (env
    /// `TRUST_JS_JIT_ADMIT_RUNS`). A body with a loop is compiled on its first run: one call can
    /// spend its whole time in the loop. Off by default (0): waiting 64 runs compiled 25-30%
    /// fewer functions and 12% less machine code on Speedometer's NewsSite pages, but cost
    /// React and Vue interactions 7% and 2% (16 runs: about 2% each).
    pub(crate) jit_admit_runs: u32,
    /// Recycled (slots, operand stack) buffers for bytecode-VM activations, so a hot call tree
    /// doesn't allocate two `Vec`s per call (see `bytecode::run`).
    pub(crate) vm_pool: Vec<crate::execution_storage::ExecutionFrame>,
    /// Recycled argument buffers for `Function.prototype.apply` list assembly, so a hot
    /// `f.apply(null, denseArray)` loop doesn't allocate a fresh `Vec` per call. Buffers are
    /// cleared before recycling; the memory walker counts their payload exactly like `vm_pool`.
    pub(crate) native_arg_pool: Vec<Vec<Value>>,
    /// Megamorphic stub cache: a global open-addressed table keyed by (receiver shape, site name)
    /// holding the last derived [`crate::bytecode::IcState`] for that pair. Probed when
    /// both of a site's ways miss, so a site rotating through more receiver shapes than it has
    /// ways (constraint hierarchies, visitor patterns) still resolves by shape compares instead
    /// of re-walking the chain with hashed key lookups. Entries are validated by the same live
    /// shape probe as the per-site ways, so stale entries are misses, never wrong hits — the
    /// table needs no invalidation. `stub_cache_names` holds each entry's key: an atom lives as
    /// long as its heap, so a key word is never reused for another name.
    pub(crate) stub_cache: Vec<std::cell::Cell<StubEntry>>,
    pub(crate) stub_cache_names: std::cell::RefCell<Vec<Option<crate::atom::PropKey>>>,
    /// The shape chain of an arguments object's named properties (`length`, @@iterator,
    /// `callee`) appended to empty or packed element storage, derived once by ordinary insertion
    /// into this interpreter's heap. Holds only keys and shape ids: no JS value or Realm.
    pub(crate) arguments_shapes: Option<([crate::atom::PropKey; 3], [u32; 3], u32)>,
    /// Pre-shaped property maps of FromPropertyDescriptor results: `[data, accessor]`, built by
    /// ordinary insertion on first use. Keys and shape only; every result object is fresh and
    /// takes the active Realm's %Object.prototype%.
    pub(crate) descriptor_maps: [std::cell::OnceCell<Props>; 2],
    /// Pre-shaped property map of CreateIterResultObject results (`value`, `done`), built by
    /// ordinary insertion on first use. Keys and shape only, like `descriptor_maps`.
    pub(crate) iter_result_map: std::cell::OnceCell<Props>,
    /// Pre-shaped property maps of the arrays RegExpBuiltinExec returns, indexed by
    /// `builtins::RegexpArray`: an empty array's map with the named keys a match array gains,
    /// built by ordinary insertion on first use. Keys and shape only, like `descriptor_maps`;
    /// every match array is fresh and takes the active Realm's %Array.prototype%.
    pub(crate) regexp_array_maps: [std::cell::OnceCell<Props>; 3],
    pub(crate) computed_reads: crate::bytecode::ComputedReadCache,
    /// Quiescent source-scoped loop plans; executing plans have only weak cache entries.
    pub(crate) fragment_cache: crate::bytecode::fragment_cache::Cache,
    pub(crate) enumeration_keys:
        crate::cache::ByteLru<crate::eval::EnumerationShape, crate::eval::CachedForInKeys>,
    /// The Strings of property keys that are not heap Strings themselves (see
    /// [`Interp::property_key_string`]), by key word: keys are never reused.
    pub(crate) property_key_strings: crate::cache::ByteLru<u64, crate::lstr::LStr>,
    /// Freelist of the JIT's activation records ([`crate::jit::JitFrame`]: a context and
    /// [`crate::jit::FRAME_BUF`] slot and operand words each) — a pop and pointer math per call
    /// instead of `Vec` bookkeeping. Records hold no live values while pooled. It also
    /// registers every record, so the collector finds the active ones (see [`FramePool`]).
    pub(crate) frame_pool: FramePool,
    /// Prototype pins for property-*creation* inline caches (see
    /// [`crate::bytecode::IC_CREATE`]), keyed by prototype identity. Creation states rotate
    /// through a site's normal polymorphic ways, so tying a pin to a physical cache cell would
    /// break when `ic_insert` demotes that state. The Weak prevents allocator ABA while any
    /// cached state compares the encoded raw prototype address.
    pub(crate) creation_pins: crate::fasthash::FastMap<usize, crate::value::WeakGc>,
    /// Weak allocation pins for every global scope a call cache has recorded (see
    /// [`crate::bytecode::CallIc::global_env`]): the cached same-realm proof compares the scope
    /// address raw, so its allocation header must not be recycled while the cache survives. A
    /// strong `Env` here would make every realm touched by JIT code immortal.
    pub(crate) global_env_pins: Vec<crate::interpreter::WeakEnv>,
    /// The byte position of the explicit call (or `new`, `super()`, tagged template) the
    /// innermost execution context is making, or [`crate::stack_trace::NO_POSITION`] outside one.
    /// Pushing a function frame moves it into [`FnFrame::call_site`]; popping restores it (see
    /// `crate::stack_trace`).
    pub(crate) call_site: u32,
    /// Script, module and eval contexts, interleaved with `fn_frames` for stack traces.
    pub(crate) code_frames: Vec<crate::stack_trace::CodeFrame>,
    /// Script execution contexts pushed for job callbacks (HTML HostCallJobCallback), each with
    /// the context depth it was pushed at; they appear in no stack trace.
    pub(crate) job_script_contexts: Vec<(u32, crate::agent_control::RuntimeOrigin)>,
    /// The prototype of the [[AlreadyResolved]] record shared by the resolving functions of an
    /// embedder's platform resolution ([`Interp::promise_resolved_with`]). Never exposed to
    /// script; it identifies those functions as platform steps (see [`InternalJob::HostReaction`]).
    pub(crate) host_resolution_marker: Gc,
    /// The JIT's helper function table, built once (a stable address the machine code indexes
    /// through x21) instead of re-materialized on every call.
    pub(crate) jit_helpers: [usize; crate::jit::N_HELPERS],
    /// Object-graph byte offsets the JIT's inline property-cache templates bake in (measured once;
    /// see [`crate::value::jit_layout`]). Lazily computed on first JIT compile.
    pub(crate) jit_layout: std::cell::OnceCell<crate::value::JitLayout>,
    /// Runtime-probed `Interp` field offsets for the asm call thunk (see [`InterpLayout`]).
    /// Filled by [`Interp::interp_layout_init`] on first JIT compile, like `jit_layout`.
    pub(crate) interp_layout: std::cell::Cell<InterpLayout>,
    /// Realm-wide proof that no Proxy has been created. Raw prototype-chain walks read it live;
    /// side-table exotics are guarded per receiver by `Object::ic_plain`.
    pub(crate) inline_ic_safe: std::cell::Cell<bool>,
    /// The RegExp protector (see [`crate::builtins::RegexpProof`]): the current Realm's RegExp
    /// algorithms read the intrinsic methods, flag getters and species.
    pub(crate) regexp_protector: RefCell<crate::builtins::RegexpProof>,
    /// The last successful `exec` match, kept raw for the legacy `RegExp.$1`-style statics:
    /// the 14 strings materialize into the constructor's internal slots only when an accessor
    /// actually reads them (see `builtins::flush_regexp_legacy`), not on every match.
    pub(crate) regexp_last: Option<RegexpLastMatch>,
    /// Live ECMAScript execution depth. Native builds use it to decide when direct JIT calls must
    /// return to a stack-growth checkpoint; it is bookkeeping, not a language-visible budget.
    pub(crate) depth: u32,
    /// Deep direct JIT calls return to Rust before this point so the next committed call can move
    /// to a heap-backed native stack segment when necessary.
    pub(crate) direct_call_depth: u32,
    /// Trapless Proxy forwards currently nested (`proxy_forward`), bounded by
    /// `MAX_PROXY_FORWARD_DEPTH`.
    pub(crate) proxy_forward_depth: u32,
    /// Per-class metadata (instance fields + whether the class extends another), keyed by the
    /// constructor object's pointer (`Gc::as_ptr(..) as usize`). Lets `construct`/`super` run field
    /// initializers without attaching engine data to the `Object` itself.
    pub(crate) class_info: gc_collect::SideSlotTable<ClassInfo>,
    /// The global `eval` function object, so a *direct* eval call (`eval(src)` by that name) can be
    /// distinguished from an indirect one and run in the caller's scope.
    pub(crate) eval_fn: Option<Gc>,
    /// Object pointers of every function tagged `__eval_realm` (each realm's `eval`): the
    /// pre-filter `call_dispatch` checks before the property lookup, so ordinary calls skip it.
    pub(crate) eval_realm_fns: crate::fasthash::FastSet<usize>,
    /// Builtin functions' immutable [[Realm]] slots (ECMA-262 CreateBuiltinFunction).
    /// Weak identity pins prevent address reuse without rooting either functions or realms.
    /// The collector activates the owning realm when a registered function is reachable.
    pub(crate) native_function_realms: RefCell<gc_collect::NativeFunctionRealms>,
    /// The "no-elements-on-array-protos" protector: `(epoch, clean)` — while `epoch` matches the
    /// live [`crate::value::proto_epoch`], `clean` says the canonical `Array.prototype` →
    /// `Object.prototype` chain has no canonical-index own keys, no exotic hops, and terminates
    /// at `Object.prototype`. The dense `push` fast path needs this proof: an indexed accessor
    /// on the chain must intercept the append (OrdinarySet), so the fast path stands down when
    /// the proof lapses. Every defineProperty / proto swap / structural change to a marked
    /// prototype bumps the epoch, forcing a re-verify.
    pub(crate) elems_protector: std::cell::Cell<(u32, bool)>,
    /// The ArraySpeciesCreate protector (see `builtins::array_species_is_intrinsic`):
    /// `(epoch, slot)`. While `epoch` matches the live [`crate::value::proto_epoch`], the
    /// current Realm's %Array% has its intrinsic `@@species` getter and `slot` is the `entries`
    /// slot of %Array.prototype%'s own "constructor" data property (`u32::MAX`: the proof
    /// failed). Both objects are marked prototypes, so every defineProperty and structural
    /// change to either bumps the epoch; an assignment can still replace the constructor's
    /// value, which each use re-reads.
    pub(crate) species_protector: std::cell::Cell<(u32, u32)>,
    /// The TypedArray accessor protector (see `Interp::ta_accessor_reached`).
    pub(crate) ta_accessor_protector: RefCell<TaAccessorProof>,
    /// The intrinsic Array iteration protector (see [`crate::builtins::ArrayIterationProof`]):
    /// proves without running author code that GetIterator on an ordinary Array, its steps and
    /// IteratorClose follow the built-in algorithms.
    pub(crate) array_iteration_protector: RefCell<crate::builtins::ArrayIterationProof>,
    /// Per-CONSTRUCTOR construct cache (`new F(...)` — keyed by callee identity, since a
    /// constructor's derived state doesn't vary per site): the same raw pointers a [`CallIc`]
    /// caches, validated by the same epoch. The `Weak` pins the callee's address against
    /// recycling (ABA), exactly like `Chunk::call_pins`.
    construct_ics: crate::fasthash::FastMap<usize, ConstructIc>,
    /// Shared, lazily allocated secondary cache for evictions from four-way call sites.
    pub(crate) call_overflow: crate::bytecode::CallOverflow,
    /// Lazily allocated reusable entry for already-resolved accessor/proxy/host callbacks.
    /// All JS identity and AST pins are Weak; this cache cannot keep a closure or realm alive.
    pub(crate) native_callback_cache: Option<Rc<crate::callback::CallbackCache>>,
    pub(crate) proxy_trap_caches: Option<Rc<ProxyTrapCaches>>,
    /// The parsed self-hosted built-in source, shared by every Realm this interpreter creates
    /// (see `crate::self_hosted`).
    pub(crate) self_hosted: std::cell::OnceCell<Rc<crate::self_hosted::SelfHostedSource>>,
    /// Programs decoded from static host snapshots, shared by every Realm of this Agent that
    /// evaluates the same snapshot (see `Interp::eval_shared_classic_snapshot_interruptible`).
    /// Keyed by the `&'static` snapshot's address and length: static bytes are immutable and
    /// never reused, so the key identifies the content. The AST and the compiled code its
    /// functions cache belong to this Agent's heap and never cross to another interpreter.
    pub(crate) shared_snapshots: Vec<SharedSnapshot>,
    /// Small ordered key layouts learned from successful ordinary construction, keyed and
    /// weak-pinned by constructor identity. Unlike `construct_ics`, this also covers constructors
    /// that need an activation (notably Prototype-style `initialize.apply(this, arguments)`
    /// wrappers), so fresh instances share keys and reserve slots for creation-IC appends.
    pub(crate) construct_capacity_hints:
        gc_collect::SideSlotTable<(crate::value::WeakGc, crate::value::PropertyLayout)>,
    /// `Symbol.iterator`, cached so the iterator protocol can look up `obj[@@iterator]` cheaply.
    pub(crate) iterator_sym: Option<crate::value::SymbolRef>,
    /// Well-known symbols, minted once per Interp — additional realms (`$262.createRealm()`) reuse
    /// them so `@@iterator` etc. have the same identity cross-realm, as the spec requires.
    /// Well-known symbol descriptors and their internal property-map keys. The key is formatted
    /// once when the symbol is minted; protocol dispatch clones this `Rc<str>` instead of
    /// repeatedly walking `%Symbol%` and allocating `"\0{id}"` strings.
    pub(crate) wk_syms: Vec<(&'static str, Value, Rc<str>)>,
    /// Set while a `?.` link in the current optional chain saw a nullish base, so the rest of the
    /// chain short-circuits to `undefined`. Reset at each `OptionalChain` boundary.
    pub(crate) short_circuit: bool,
    /// The `import.meta` object for the module currently executing (None in script code).
    pub(crate) import_meta: Option<Value>,
    /// Default referrer for a bare `import()` in script code (so relative specifiers resolve).
    pub(crate) import_base: String,
    /// Loaded module namespace objects, keyed by canonical specifier (for `import()` + caching).
    pub(crate) modules: std::collections::HashMap<String, Value>,
    /// Full module records (parsed body, environment, resolved export tables, evaluation status),
    /// keyed by canonical specifier. Drives the two-phase Instantiate/Evaluate module linking.
    pub(crate) module_recs: std::collections::HashMap<String, crate::modules::ModuleRec>,
    /// Host module loader: `(specifier, referrer, attr_type)` → `(canonical_key, source)`.
    /// `attr_type` is the import's `with { type: ... }` attribute — an attribute-aware host
    /// returns RAW file contents for `json`/`text`/`bytes` (binary latin-1-decoded, one char per
    /// byte) and the engine synthesizes the wrapper module; attribute-blind hosts ignore it.
    #[allow(clippy::type_complexity)]
    pub(crate) module_loader:
        Option<Rc<dyn Fn(&str, &str, Option<&str>) -> Option<(String, String)>>>,
    /// Optional asynchronous host hook for dynamic `import()`. The hook accepts a stable request
    /// id and starts HostLoadImportedModule; the embedder later calls
    /// `Engine::finish_dynamic_module_load` with the fetched source or failure.
    #[allow(clippy::type_complexity)]
    pub(crate) dynamic_module_loader:
        Option<Rc<dyn Fn(u64, u64, &str, &str, Option<&str>) -> bool>>,
    /// Conditional intrinsic importValue bridges, owned by the requester promise. They are
    /// not Agent work roots until the child import actually queues a runnable reaction.
    pub(crate) routed_imports: gc_collect::SideSlotTable<crate::agent_control::RoutedImport>,
    /// Live module-namespace state keyed by the namespace object's pointer: for each exported name,
    /// how to read its current value (a live binding in some module scope, or a static value for a
    /// star-as namespace re-export). Namespace property reads consult this so they stay live.
    pub(crate) module_ns:
        gc_collect::SideSlotTable<crate::fasthash::FastMap<String, crate::modules::NsBinding>>,
    /// WeakMap/WeakSet cells. Keys are real weak handles; only WeakMap values are strong, and then
    /// only ephemerally while both the collection and key are live.
    pub(crate) weak_collection_data: gc_collect::WeakCollectionData,
    /// Object/symbol identity to entry offset for WeakMap/WeakSet. Their order is unobservable, and
    /// ECMA-262 requires average access time sublinear in the collection size; the parallel index
    /// makes get/set/has/delete average O(1) while `weak_collection_data` retains the payload.
    pub(crate) weak_collection_index:
        crate::fasthash::FastMap<usize, crate::weak_metadata::WeakCollectionIndex>,
    /// Prototypes for builtins created after `new()` (Map/Set/Date/...), looked up by name so their
    /// native constructors can stamp the right `[[Prototype]]`.
    pub(crate) extra_protos: IntrinsicMap,
    /// The Data Block of every attached ArrayBuffer and every SharedArrayBuffer, keyed by the
    /// buffer object's pointer: an ArrayBuffer's own bytes, whose indirection lets an embedder
    /// identify them with an external resource such as wasm linear memory, or a
    /// SharedArrayBuffer's owning Shared Data Block reference. Detaching an ArrayBuffer removes
    /// its entry; sweeping the last shared owner releases that block's allocation.
    pub(crate) array_buffers: crate::fasthash::FastMap<usize, BufferData>,
    pub(crate) native_buffers:
        crate::fasthash::FastMap<usize, Rc<crate::native_typed_array::NativeBuffer>>,
    /// Monotonic mutation generations for ArrayBuffer byte storage. Host embedders use these
    /// generations to avoid copying an unchanged external mirror across a synchronous boundary.
    pub(crate) array_buffer_versions: crate::fasthash::FastMap<usize, u64>,
    /// Bounded, conservative byte ranges covering writes since the last host synchronization.
    pub(crate) array_buffer_dirty_ranges: crate::fasthash::FastMap<usize, Vec<Range<usize>>>,
    /// Immutable ArrayBuffer pointers (created via `transferToImmutable`/`sliceToImmutable`): their
    /// bytes can be read but never written, resized, detached, or transferred.
    pub(crate) immutable_buffers: std::collections::HashSet<usize>,
    /// ArrayBuffers carrying an embedder detach key. ECMAScript transfer operations do not know
    /// that key and therefore cannot detach them; the embedder can still detach them explicitly
    /// when the external resource changes identity (notably WebAssembly memory growth).
    pub(crate) host_keyed_buffers: std::collections::HashSet<usize>,
    /// Whether this agent may block in `Atomics.wait` (false for the main agent, true for the
    /// worker agents spawned by `$262.agent.start`).
    pub(crate) can_block: bool,
    /// Agent-harness wiring (present only in spawned agents / a main with agents).
    pub(crate) agent: Option<Box<AgentChannels>>,
    /// TypedArray view state, keyed by the typed-array object's pointer.
    pub(crate) typed_arrays: crate::fasthash::FastMap<usize, TaInfo>,
    /// Object pointers of async generator instances (the AsyncGenerator brand).
    pub(crate) async_gens: std::collections::HashSet<usize>,
    /// The global environment's [[VarNames]]: names declared by `var`/function in global code, for
    /// GlobalDeclarationInstantiation's cross-script clash checks.
    pub(crate) global_var_names: Rc<RefCell<std::collections::HashSet<String>>>,

    /// GC pins: a `Gc` for each object that has an entry in a pointer-keyed side table
    /// (typed_arrays, promises, array_buffers, …), keyed by its address. A pin keeps nothing
    /// alive (the collector does not trace pins): such an object dies in a collection like any
    /// other, whose sweep evicts its table entries and its pin before a later allocation can
    /// reuse the address.
    pub(crate) gc_pins: crate::fasthash::FastMap<usize, Gc>,
    /// The backing ArrayBuffer *object* for each TypedArray (so the `buffer` getter can return it
    /// without storing it as an observable own property). Keyed by the TypedArray's pointer.
    pub(crate) ta_buffer: gc_collect::SideSlotTable<Value>,
    /// Each `ShadowRealm` instance owns an isolated realm (a full sub-interpreter), keyed by the
    /// ShadowRealm object's pointer. Only primitive completion values cross the boundary.
    pub(crate) shadow_realms: crate::fasthash::FastMap<usize, crate::agent_access::Owned<Interp>>,
    /// A ShadowRealm's interpreter: its place in its parent's tree (None for an Agent's root).
    pub(crate) shadow_home: Option<ShadowHome>,
    /// DataView state `(buffer ptr, byteOffset, byteLength)`, keyed by the DataView's pointer.
    /// DataView state: (buffer ptr, byteOffset, byteLength, is-length-tracking).
    pub(crate) data_views: crate::fasthash::FastMap<usize, (usize, usize, usize, bool)>,
    /// Compiled regular expressions, keyed by the RegExp object's pointer.
    pub(crate) regexps: crate::fasthash::FastMap<usize, Rc<crate::regex::Regex>>,
    /// Immutable matcher programs shared by RegExp instances with the same source and flags.
    /// A literal expression must still produce a fresh object (and therefore a fresh
    /// `lastIndex`) each time it is evaluated, but recompiling its parser/bytecode and first-set
    /// table is pure duplicate work. The outer map is bounded in `compiled_regexp` so code that
    /// constructs unbounded dynamic patterns cannot retain them forever.
    pub(crate) regexp_programs: crate::cache::RegexpProgramCache,
    /// Proxy `(target, handler)` pairs, keyed by the proxy object's pointer.
    pub(crate) proxies: gc_collect::SideSlotTable<(Value, Value)>,
    /// Embedder-installed Web IDL indexed properties, keyed by the platform object's pointer.
    /// The object is pinned in `gc_pins`; `getter` is an internal GC edge owned by that object.
    pub(crate) host_indexed: gc_collect::SideSlotTable<HostIndexedProperties>,
    /// The active `new.target` for the function currently executing (Undefined outside a `new`).
    pub(crate) new_target: Value,
    /// The `new.target` to install for the next constructor invocation (set by `construct`).
    pub(crate) pending_new_target: Value,
    /// The `$262.IsHTMLDDA` objects (emulate `document.all`), indexed weakly by identity: typeof
    /// "undefined", falsy, and loosely equal to undefined/null, despite being callable Objects.
    /// The realm/global property is the real owner; this language-level brand must not retain it.
    pub(crate) htmldda: crate::fasthash::FastMap<usize, crate::value::WeakGc>,
    /// The caller's realm during a cross-realm [[Construct]]: the spec pops the callee context
    /// before throwing a derived constructor's return/`this` validation errors, so those errors
    /// belong to the caller's realm.
    pub(crate) ctor_caller_realm: Option<RealmState>,
    /// Additional realms created via `$262.createRealm()`, keyed by the realm's global-object pointer.
    /// Each holds its own intrinsics; the shared side tables (proxies, buffers, …) and well-known
    /// symbols are common, so objects cross realm boundaries freely.
    pub(crate) realms: crate::fasthash::FastMap<usize, RealmState>,
    /// Promise state keyed by the promise object's pointer.
    pub(crate) promises: gc_collect::SideSlotTable<PromiseState>,
    /// Rejected promises with no handler attached at rejection time (ptr -> reason). Removed when a
    /// handler is later attached; whatever remains after a microtask checkpoint is a genuine
    /// unhandled rejection the embedder can report (see [`Engine::take_unhandled_rejections`]).
    /// Keyed by the promise object's pointer; holds `(order, promise, reason, context)` so the
    /// runtime can hand both to a global `unhandledrejection` handler in rejection order (HTML's
    /// about-to-be-notified rejected promises list). `context` is the host settings context
    /// active when the promise was rejected: HTML's HostPromiseRejectionTracker appends the
    /// promise to that settings object's global's list, not to the top-level one's.
    pub(crate) unhandled_rejections: crate::fasthash::FastMap<usize, (u64, Value, Value, u64)>,
    /// The order key of the next unhandled rejection.
    pub(crate) next_unhandled_rejection: u64,
    /// HostPromiseRejectionTracker "handle" operations on promises of the host's outstanding
    /// rejected promises weak set, as `(promise, reason, context)` with the host settings context
    /// whose global's set held the promise, until the host takes them (see
    /// [`crate::Engine::take_handled_rejections`]) to fire `rejectionhandled` at that global.
    pub(crate) handled_rejections: Vec<(Value, Value, u64)>,
    /// Temporal object internal slots, keyed by the object's pointer.
    pub(crate) temporal: crate::fasthash::FastMap<usize, crate::temporal::Temporal>,
    /// The calendar id of a Temporal date-bearing object (default "iso8601"), keyed by object ptr.
    pub(crate) temporal_cal: crate::fasthash::FastMap<usize, std::rc::Rc<str>>,
    /// Opaque embedder settings token associated with the currently-running script/callback.
    /// Promise reactions retain this alongside their ECMA-262 handler Realm so a browser host
    /// that temporarily multiplexes several Window settings objects through one engine Realm can
    /// restore the relevant settings while the job runs. Zero is the embedder's default context.
    pub(crate) host_job_context: u64,
    /// Script/Eval entry below the currently active function frames. Realm keys are
    /// non-owning: the realm registry pins every live global. This is not the HTML
    /// backup incumbent stack; the embedder supplies that for native callbacks.
    pub(crate) script_entry: Option<(usize, usize)>,
    /// Embedder entry context, distinct from the innermost Script/Eval caller.
    /// HTML's entry Realm survives ordinary cross-Realm calls and indirect eval.
    /// Realm keys are pinned by the realm registry.
    pub(crate) host_entry_realm: Option<usize>,
    /// Opt-in browser incumbent-source propagation for JobCallback records. Non-browser hosts
    /// retain ECMA-262's default HostMakeJobCallback/HostCallJobCallback behavior.
    pub(crate) capture_job_script_caller: bool,
    /// Actual caller source above the suspended local context floor at a cross-Interp entry.
    pub(crate) caller_origin: Option<crate::agent_control::CallerOrigin>,
    /// Per-(ECMAScript realm, host settings object) global lexical state.
    /// HTML normally has a 1:1 realm/settings mapping; browser embedders that
    /// multiplex logical Window realms select the corresponding state through
    /// the opaque context token above.
    pub(crate) host_settings_states: crate::fasthash::FastMap<(usize, u64), HostSettingsState>,
    /// Settings tokens whose Document has been destroyed by the embedder.
    /// Tokens are opaque and monotonic, so retaining the small numeric set
    /// prevents a late host completion from recreating a retired environment.
    pub(crate) retired_host_job_contexts: std::collections::HashSet<u64>,
    pub(crate) host_job_context_enter: Option<fn(&mut Interp, u64)>,
    pub(crate) host_job_context_leave: Option<fn(&mut Interp)>,
    /// Embedder host state (typed slots + resource table); see [`crate::host`]. Reached from
    /// native fns via [`Interp::op_state`] — the only way, since `NativeFn` cannot capture.
    pub(crate) host_state: crate::host::OpState,
    /// Live generator coroutines, keyed by the generator object's pointer. Each owns an explicit
    /// heap VM continuation that parks at `yield`/`await` without reserving an OS thread stack.
    pub(crate) generators: gc_collect::SideSlotTable<crate::coroutine::Coroutine>,
    /// The live-object ceiling: a safepoint (`gc_check`, the JIT's entry and loop polls) that
    /// finds the live count above it collects, and throws the limit's RangeError if more than
    /// the Agent limit’s objects stay alive; the record-getter fast path then leaves the call
    /// to the ordinary path. It mirrors the Agent limit, or, after that RangeError, the live
    /// count at the throw plus [`LIMIT_ERROR_HANDLER_ROOM`]. It is not a growth trigger:
    /// allocation makes a collection due through the heap's byte budget (see
    /// `gc_collect::GC_MIN_HEAP_BYTES`).
    pub(crate) gc_next: i64,
    /// Call counter for [`Interp::gc_check_amortized`]: layered calls poll every 256 calls. Direct
    /// JIT calls leave it alone; the callee's entry poll is their safepoint.
    pub(crate) gc_tick: u32,
    /// True while a native constructor is being invoked via `new` (lets e.g. `Number`/`String`
    /// build a wrapper object instead of returning a primitive).
    pub(crate) constructing: bool,
    /// True only while a *derived* class constructor body is executing — the one context where a
    /// `super(...)` call is legal. Field initializers, methods, plain functions and global code all
    /// clear it, so a stray `super()` (including one reached through a direct `eval`) is rejected
    /// instead of re-entering instance-field initialization unboundedly.
    pub(crate) super_call_ok: bool,
    /// Live user-function activations, oldest first: (fn ptr, fn object, arguments object,
    /// strict). Backs the legacy SpiderMonkey `fn.caller` / `fn.arguments` reflection.
    pub(crate) fn_frames: Vec<FnFrame>,
    /// Monotonic [[AsyncEvaluationOrder]] source for module evaluation.
    pub(crate) module_async_seq: u64,
    /// Set by `yield*` just before parking: the yielded value is the inner iterator's result
    /// object and must pass through the generator driver unwrapped.
    pub(crate) yield_raw_result: bool,
    /// True while statements *directly* in an async generator's body run (nested ordinary
    /// functions clear it; arrows inherit): `return <expr>` awaits its value only there.
    pub(crate) in_async_gen_body: bool,
    /// NamedEvaluation: the binding/property name an anonymous class expression being evaluated
    /// should take — applied *before* its static initializers run.
    pub(crate) pending_fn_name: Option<String>,
    /// A pending proper tail call: set by `return f(...)` in a strict ordinary function; the
    /// nearest `Interp::call` frame re-dispatches it (a trampoline), keeping the stack flat.
    pub(crate) pending_tail: Option<Box<(Value, Value, Vec<Value>)>>,
    /// True while executing a body where `return f(...)` is a proper tail call (a strict,
    /// ordinary, non-constructor function).
    pub(crate) tco_ok: bool,
    /// GetTemplateObject cache: one frozen strings array per tagged-template *site* (AST node)
    /// within its Realm, so the same site always passes the identical object. The Realm global's
    /// identity is part of the key because HTML/ECMA-262 give each Realm its own [[TemplateMap]];
    /// this also lets a collection discard all entries when a browsing-context Realm dies.
    pub(crate) template_cache: std::collections::HashMap<(usize, u64), Value>,
    /// True while class field initializer code runs: a direct eval from there may not contain an
    /// `arguments` reference (arrows inherit the context; ordinary functions clear it).
    pub(crate) in_field_init_code: bool,
    /// A stack of disposal scopes — one frame per block/function body that can hold `using`
    /// declarations. Resources are disposed in reverse on scope exit (see `dispose_frame`).
    pub(crate) using_stack: Vec<Vec<Disposable>>,
    /// Monotonic counter minting globally-unique private backing keys for auto-accessors (so a
    /// subclass's accessor never collides with a superclass's on the same instance).
    pub(crate) accessor_seq: u64,
    /// Scratch collector for `context.addInitializer(fn)` calls during decorator application; drained
    /// after each decorator runs.
    pub(crate) decorator_initializers: Vec<Value>,
    /// Annex B.3.3 web-compat function hoisting: block/if-position function declarations (in sloppy
    /// code) whose names got a function-scope `var` binding, keyed by AST pointer. When such a
    /// declaration is *evaluated*, the block binding's value is copied into the var binding. The
    /// `Rc` value keeps the AST alive so a pointer is never reused by a different function.
    pub(crate) annexb_fn_sync: crate::fasthash::FastMap<usize, Rc<Function>>,
    /// `import defer` namespaces awaiting first access: namespace object ptr → module key.
    /// Accessing a property of one evaluates the module (then the entry is removed).
    pub(crate) deferred_ns: crate::fasthash::FastMap<usize, String>,
    /// module key → its (single) deferred namespace object.
    pub(crate) deferred_ns_objs: crate::fasthash::FastMap<String, Value>,
    /// Promise-state forwarding: when a native super() grafts a fresh promise's state onto the
    /// subclass `this`, the original object's resolvers redirect here.
    pub(crate) promise_forward: gc_collect::SideSlotTable<Value>,
    /// Mapped `arguments` objects: object ptr → (function scope, per-index parameter name — None
    /// once unmapped by delete/defineProperty). Reads/writes of still-mapped indices alias the
    /// parameter bindings.
    pub(crate) mapped_arguments: gc_collect::SideSlotTable<(Env, Vec<Option<String>>)>,
    /// Live AST-to-VM formal aliases. Allocated only while a sloppy activation needs them.
    pub(crate) borrowed_parameters: Option<borrowed_parameters::Registry>,
    /// Source-phase imports: canonical module key → its (cached) ModuleSource object.
    pub(crate) module_source_objs: crate::fasthash::FastMap<String, Value>,
    /// WeakRef [[WeakRefTarget]] slots, keyed by the WeakRef instance.
    pub(crate) weak_refs: crate::fasthash::FastMap<usize, Option<WeakTarget>>,
    /// Weak-target reverse subscriptions and pinned acyclic-death notifications; no strong edges.
    pub(crate) weak_metadata: crate::weak_metadata::WeakMetadata,
    /// FinalizationRegistry internal slots, keyed by the registry instance.
    pub(crate) finalization_registries: gc_collect::SideSlotTable<FinalizationState>,
    /// Async generators mid-step (running or parked at an `await`): further next/return/throw
    /// requests queue here (the spec's AsyncGeneratorRequest queue) until the step completes.
    pub(crate) async_gen_busy: std::collections::HashSet<usize>,
    pub(crate) async_gen_queue: gc_collect::SideSlotTable<
        std::collections::VecDeque<(Value, crate::coroutine::Resume)>,
        std::collections::hash_map::RandomState,
    >,
    /// P1 S2: frees this Agent's object cells. Keep it the LAST field: fields drop in
    /// declaration order, and every earlier one may still read objects as it drops.
    #[allow(dead_code)] // held for its Drop
    heap_release: crate::value::HeapRelease,
}

// Phase 0 managed-memory ownership inventory. This macro deliberately generates both the golden
// classification list and an exhaustive (no `..`) struct pattern from the same field tokens. A
// newly added `Interp` field therefore makes test builds fail until its ownership class is chosen
// in a review-visible diff. `unaccounted` is permitted while the visitor is incomplete, but must
// reach zero before the managed-memory record may set `complete: true`.
#[cfg(test)]
macro_rules! interp_memory_inventory {
    ($($(#[$attribute:meta])* $field:ident => $class:literal,)*) => {
        const INTERP_MEMORY_INVENTORY: &[(&str, &str)] = &[
            $($(#[$attribute])* (stringify!($field), $class),)*
        ];

        fn interp_memory_inventory_exhaustive_pattern(interp: &Interp) {
            let Interp { $($(#[$attribute])* $field: _,)* } = interp;
        }
    };
}

#[cfg(test)]
interp_memory_inventory! {
    gc_heap => "measured",
    realm_id => "non_owning",
    symbol_agent => "measured",
    global => "measured",
    global_env => "measured",
    object_proto => "measured",
    function_proto => "measured",
    array_proto => "measured",
    array_ctor => "measured",
    string_proto => "measured",
    number_proto => "measured",
    boolean_proto => "measured",
    symbol_proto => "measured",
    error_protos => "measured",
    console => "measured",
    wall_clock => "external",
    runtime_interrupt => "external",
    interrupt_poll_tick => "non_owning",
    ast_tick => "non_owning",
    strict => "non_owning",
    tier => "non_owning",
    tier_threshold => "non_owning",
    jit_admit_runs => "non_owning",
    vm_pool => "measured",
    native_arg_pool => "measured",
    stub_cache => "measured",
    arguments_shapes => "measured",
    descriptor_maps => "measured",
    iter_result_map => "measured",
    regexp_array_maps => "measured",
    stub_cache_names => "measured",
    computed_reads => "measured",
    fragment_cache => "measured",
    enumeration_keys => "measured",
    property_key_strings => "measured",
    frame_pool => "measured",
    creation_pins => "measured",
    global_env_pins => "measured",
    call_site => "non_owning",
    code_frames => "measured",
    job_script_contexts => "measured",
    host_resolution_marker => "measured",
    jit_helpers => "non_owning",
    jit_layout => "non_owning",
    interp_layout => "non_owning",
    inline_ic_safe => "non_owning",
    regexp_protector => "non_owning",
    regexp_last => "measured",
    depth => "non_owning",
    direct_call_depth => "non_owning",
    proxy_forward_depth => "non_owning",
    class_info => "measured",
    eval_fn => "measured",
    eval_realm_fns => "measured",
    native_function_realms => "measured",
    elems_protector => "non_owning",
    species_protector => "non_owning",
    ta_accessor_protector => "non_owning",
    array_iteration_protector => "non_owning",
    construct_ics => "measured",
    call_overflow => "measured",
    native_callback_cache => "measured",
    proxy_trap_caches => "measured",
    self_hosted => "measured",
    shared_snapshots => "measured",
    construct_capacity_hints => "measured",
    iterator_sym => "measured",
    wk_syms => "measured",
    short_circuit => "non_owning",
    import_meta => "measured",
    import_base => "measured",
    modules => "measured",
    module_recs => "measured",
    module_loader => "external",
    dynamic_module_loader => "external",
    routed_imports => "measured",
    module_ns => "measured",
    weak_collection_data => "measured",
    weak_collection_index => "measured",
    extra_protos => "measured",
    array_buffers => "external",
    native_buffers => "measured",
    array_buffer_versions => "measured",
    array_buffer_dirty_ranges => "measured",
    immutable_buffers => "measured",
    host_keyed_buffers => "measured",
    can_block => "non_owning",
    agent => "measured",
    typed_arrays => "measured",
    async_gens => "measured",
    global_var_names => "measured",
    gc_pins => "measured",
    ta_buffer => "measured",
    shadow_realms => "measured",
    shadow_home => "non_owning",
    data_views => "measured",
    regexps => "measured",
    regexp_programs => "measured",
    proxies => "measured",
    host_indexed => "measured",
    new_target => "measured",
    pending_new_target => "measured",
    htmldda => "measured",
    ctor_caller_realm => "measured",
    realms => "measured",
    promises => "measured",
    unhandled_rejections => "measured",
    next_unhandled_rejection => "non_owning",
    handled_rejections => "measured",
    temporal => "measured",
    temporal_cal => "measured",
    host_job_context => "non_owning",
    script_entry => "non_owning",
    host_entry_realm => "non_owning",
    capture_job_script_caller => "non_owning",
    caller_origin => "measured",
    host_settings_states => "measured",
    retired_host_job_contexts => "measured",
    host_job_context_enter => "non_owning",
    host_job_context_leave => "non_owning",
    host_state => "external",
    generators => "measured",
    gc_next => "non_owning",
    gc_tick => "non_owning",
    constructing => "non_owning",
    super_call_ok => "non_owning",
    fn_frames => "measured",
    module_async_seq => "non_owning",
    yield_raw_result => "non_owning",
    in_async_gen_body => "non_owning",
    pending_fn_name => "measured",
    pending_tail => "measured",
    tco_ok => "non_owning",
    template_cache => "measured",
    in_field_init_code => "non_owning",
    using_stack => "measured",
    accessor_seq => "non_owning",
    decorator_initializers => "measured",
    annexb_fn_sync => "measured",
    deferred_ns => "measured",
    deferred_ns_objs => "measured",
    promise_forward => "measured",
    mapped_arguments => "measured",
    borrowed_parameters => "measured",
    module_source_objs => "measured",
    weak_refs => "measured",
    weak_metadata => "measured",
    finalization_registries => "measured",
    async_gen_busy => "measured",
    async_gen_queue => "measured",
    heap_release => "non_owning",
}

#[cfg(test)]
#[test]
fn interp_managed_memory_inventory_is_exhaustive_and_classified() {
    use std::collections::HashSet;

    let mut names = HashSet::new();
    for &(name, class) in INTERP_MEMORY_INVENTORY {
        assert!(
            names.insert(name),
            "duplicate Interp inventory field {name}"
        );
        assert!(
            matches!(
                class,
                "measured" | "external" | "non_owning" | "unaccounted"
            ),
            "invalid Interp memory classification for {name}: {class}"
        );
    }
    assert_eq!(names.len(), 151);
    assert!(
        INTERP_MEMORY_INVENTORY
            .iter()
            .all(|(_, class)| *class != "unaccounted"),
        "all Interp fields must have a retained-memory classification"
    );
    let _tripwire: fn(&Interp) = interp_memory_inventory_exhaustive_pattern;
}

/// Stable topology and a cached retained path. The path is traced only by active/queued work.
pub(crate) struct ShadowHome {
    pub(crate) owner: crate::value::WeakGc,
    pub(crate) parent: crate::agent_access::RealmId,
    pub(crate) route: Rc<crate::agent_control::Route>,
}

/// A detached subtree is destroyed only after collection visitors have ended. Coroutines
/// terminate as memory; collection never resumes author code during retirement.
fn drop_collected_realm(mut realm: crate::agent_access::Owned<Interp>) {
    // SAFETY: retirement removed this inactive subtree from all collection snapshots. The
    // owning capability is now exclusive, and no child lease exists (Owned::drop checks).
    let mut pointer = realm.retirement_pointer();
    let interp = unsafe { pointer.as_mut() };
    for (_, nested) in std::mem::take(&mut interp.shadow_realms) {
        drop_collected_realm(nested);
    }
    drop(std::mem::take(&mut interp.generators));
}

impl Drop for Interp {
    fn drop(&mut self) {
        // libtest exits the process before persistent runtime-worker TLS destructors run. Flush
        // the opt-in function profile when the owning Agent is destroyed so browser-workload
        // diagnostics retain their measurements; `dump` clears the TLS aggregate, preventing a
        // later Agent on the same worker from reporting these calls twice. A ShadowRealm's
        // interpreter is not an Agent: its calls stay in its Agent's profile.
        if self.shadow_home.is_none() {
            function_profile_dump();
        } else {
            let children = &self.gc_heap.agent.children;
            children.set(children.get() - 1);
            if children.get() == 0 {
                // The root is alone again; its own tables are authoritative.
                if let Ok(mut index) = self.gc_heap.agent.metadata_owners.try_borrow_mut() {
                    index.clear();
                }
            }
        }
        // A suspended coroutine owns non-Send `Rc` values on its worker stack and retains a raw
        // pointer back to this interpreter. Wake and unwind every such body while `self` is still
        // valid, waiting for acknowledgement before normal field destruction begins.
        let mut coroutines = std::mem::take(&mut self.generators);
        for coroutine in coroutines.values_mut(&self.gc_heap) {
            coroutine.terminate(self);
        }
        // The whole Agent is ending, not merely reaching a GC safepoint: the heap's owner
        // (`heap_release`) finalizes and frees every cell once all other fields have dropped,
        // and a ShadowRealm's interpreter must not touch the heap it shares with its parent.
        // Do not let the driver thread retain this Agent after its interpreter dies. Heap
        // release runs after the ensuing field drops, while the heap is still available.
        // A ShadowRealm sharing its parent's heap and symbols has the parent's Agent identity.
        if !self.gc_heap_root() {
            return;
        }
        crate::value::deactivate_agent_if(&self.gc_heap, &self.symbol_agent);
    }
}

/// A `using x = v` resource: the value plus its captured dispose method.
pub struct Disposable {
    pub value: Value,
    pub method: Value,
    /// The declaration/resource kind is async-dispose. This remains true for the synthesized
    /// async wrapper around a synchronous `@@dispose` method and for null/undefined markers.
    pub kind_is_async: bool,
    /// The captured method itself came from `@@asyncDispose`. When false but `kind_is_async` is
    /// true, DisposeResources must translate the sync call into a promise and await that wrapper.
    pub method_is_async: bool,
}

/// ECMA-262 JobCallback plus the optional browser host's incumbent global. Keep the global as
/// an owned GC edge, not a permanently rooted host registry or an untraced Realm pointer.
/// This source context is independent of the callable's execution Realm / job settings.
#[derive(Clone)]
pub(crate) struct JobCallback {
    pub(crate) callback: Value,
    pub(crate) script_caller: Option<Gc>,
    /// HTML HostMakeJobCallback's [[ActiveScriptContext]]: the active ScriptOrModule at
    /// registration, which HostCallJobCallback pushes while calling the callback (browser hosts
    /// only; see [`Interp::make_job_callback`]).
    pub(crate) active_script: Option<crate::agent_control::RuntimeOrigin>,
    /// A reaction step the specification models as an unobservable built-in function. When set,
    /// `callback` is not called; see [`InternalJob`].
    pub(crate) internal: InternalJob,
}

/// Promise reaction steps that ECMA-262 defines as built-in functions no script can observe.
/// Running them directly avoids allocating those function objects for every reaction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum InternalJob {
    #[default]
    None,
    /// Await's onFulfilled (ECMA-262 `#await` steps 3-4, local snapshot e28783d5): resume the
    /// suspended async function with a normal completion. The reaction's `callback` is that
    /// function's result promise, which also keys its coroutine. `realm` is the Realm
    /// CreateBuiltinFunction would have captured: the one running at the `await`.
    AwaitFulfilled { realm: usize },
    /// Await's onRejected (steps 5-6): resume the suspended async function with a throw
    /// completion.
    AwaitRejected { realm: usize },
    /// An embedder's own reaction steps (Web IDL "react to a promise"; see
    /// [`Interp::react_to_promise`]): `callback` is called like a handler, but the job belongs
    /// to no Realm or settings object, so HTML HostEnqueuePromiseJob gives it null job settings
    /// and it runs even after the settings object it was registered in is gone.
    HostReaction,
    /// The callback word keys the requester promise's conditional ShadowRealm import slots.
    /// It is never called. The dispatch route and language Realm are independent identities.
    RoutedImportFulfilled {
        requester: crate::agent_access::RealmId,
        realm: usize,
        context: u64,
    },
    RoutedImportRejected {
        requester: crate::agent_access::RealmId,
        realm: usize,
        context: u64,
    },
}

impl InternalJob {
    /// The Realm of the built-in function this step stands for.
    pub(crate) fn realm(self) -> Option<usize> {
        match self {
            InternalJob::None | InternalJob::HostReaction => None,
            InternalJob::AwaitFulfilled { realm }
            | InternalJob::AwaitRejected { realm }
            | InternalJob::RoutedImportFulfilled { realm, .. }
            | InternalJob::RoutedImportRejected { realm, .. } => Some(realm),
        }
    }

    pub(crate) fn routed_import(self) -> Option<(crate::agent_access::RealmId, usize, u64, bool)> {
        match self {
            Self::RoutedImportFulfilled {
                requester,
                realm,
                context,
            } => Some((requester, realm, context, true)),
            Self::RoutedImportRejected {
                requester,
                realm,
                context,
            } => Some((requester, realm, context, false)),
            _ => None,
        }
    }

    /// Await's resumption steps, which run instead of calling `callback`.
    pub(crate) fn resumes_await(self) -> bool {
        matches!(
            self,
            InternalJob::AwaitFulfilled { .. } | InternalJob::AwaitRejected { .. }
        )
    }
}

impl JobCallback {
    pub(crate) fn plain(callback: Value) -> Self {
        Self {
            callback,
            script_caller: None,
            active_script: None,
            internal: InternalJob::None,
        }
    }
}

/// A queued microtask: running one promise reaction.
pub struct Job {
    pub(crate) handler: JobCallback,
    pub result: Value,
    pub value: Value,
    pub fulfilled: bool,
    /// Realm supplied to HostEnqueuePromiseJob. `None` is the spec's null Realm for a reaction
    /// without a callable handler; otherwise this is the Realm-global pointer.
    pub realm: Option<usize>,
    pub host_context: u64,
}

/// The host settings a promise reaction was registered with. The token is meaningful only in
/// its registration Realm, and a pending reaction can settle from another Realm, so both are
/// kept until NewPromiseReactionJob chooses the handler Realm at enqueue time.
#[derive(Clone, Copy)]
pub(crate) struct PromiseReactionSettings {
    /// The registration Realm's global (the key of `Job::realm`).
    pub realm: usize,
    pub context: u64,
}

#[derive(Default)]
pub struct PromiseState {
    /// 0 = pending, 1 = fulfilled, 2 = rejected.
    pub status: u8,

    pub value: Value,
    /// Pending reactions: `(onFulfilled, onRejected, resultPromise, registrationSettings)`.
    pub(crate) reactions: Vec<(JobCallback, JobCallback, Value, PromiseReactionSettings)>,
    /// [[PromiseIsHandled]]: a reaction was ever registered (PerformPromiseThen), or the host
    /// marked the promise handled (Web IDL "mark as handled").
    pub(crate) handled: bool,
    /// Membership of a global's outstanding rejected promises weak set (HTML #unhandled-
    /// promise-rejections), as the host settings context of that global: the host reported this
    /// rejection as unhandled there, so a later handler is reported back to it
    /// ([`Interp::add_outstanding_rejection`]). Dies with the promise.
    pub(crate) outstanding: Option<u64>,
}

/// The pointer-keyed `Interp` tables whose values are strong internal-slot edges of their key
/// object, traced once that object is marked (`Interp::trace_owner_slots`). Expands
/// `$callback!(table, ...)`. A Map's or Set's [[MapData]]/[[SetData]] is in its cell.
macro_rules! gc_side_slot_tables {
    ($callback:ident) => {
        $callback!(
            class_info,
            module_ns,
            ta_buffer,
            proxies,
            host_indexed,
            promises,
            promise_forward,
            routed_imports,
            async_gen_queue,
            finalization_registries,
            mapped_arguments,
            construct_capacity_hints,
        )
    };
}

#[path = "gc_collect.rs"]
pub(crate) mod gc_collect;

/// A Realm's named intrinsic table. Shared, not copied, between the active Realm fields and
/// its saved `RealmState`: a cross-Realm call swaps Realms by reference count instead of
/// cloning these tables several times. Realm setup mutates through `Rc::make_mut`, which
/// copies only a table that another snapshot still shares.
pub(crate) type IntrinsicMap = Rc<crate::fasthash::FastMap<&'static str, Gc>>;

/// Engine-side metadata for a class constructor (see [`Interp::class_info`]).
pub struct ClassInfo {
    /// Instance fields (and auto-accessor backing fields), in declaration order. Shared so each
    /// construction reads the list without copying keys, initializer ASTs or transforms.
    pub fields: Rc<[FieldInit]>,
    /// The environment field initializers evaluate in (carries the class's super bindings).
    pub field_env: Env,
    /// Instance initializers registered by decorators via `context.addInitializer`, run on each new
    /// instance after its fields are set, with `this` = the instance.
    pub instance_initializers: Vec<Value>,
    /// Instance private methods and accessors, stamped as private elements on each instance when
    /// `this` is created (before the fields run). Stamping twice is a TypeError.
    pub private_members: Vec<(String, crate::value::Property)>,
}

impl ClassInfo {
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        let mut bytes = self
            .fields
            .len()
            .saturating_mul(std::mem::size_of::<FieldInit>())
            .saturating_add(
                self.instance_initializers
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Value>()),
            )
            .saturating_add(
                self.private_members
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(String, crate::value::Property)>()),
            );
        for field in self.fields.iter() {
            bytes = bytes.saturating_add(field.key.scan_retained_memory(visitor));
            if let Some(initializer) = &field.init {
                let ast_bytes = crate::ast::scan_expr_retained_memory(initializer, visitor);
                visitor.add_function_bytecode_bytes(ast_bytes);
            }
            bytes = bytes.saturating_add(
                field
                    .transforms
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Value>()),
            );
            for transform in &field.transforms {
                visitor.value(transform);
            }
        }
        for initializer in &self.instance_initializers {
            visitor.value(initializer);
        }
        for (name, property) in &self.private_members {
            bytes = bytes
                .saturating_add(name.capacity())
                .saturating_add(property.scan_retained_memory(visitor));
        }
        // `field_env` is a registered scope and remains canonical to the collector snapshot.
        bytes
    }
}

/// One instance field: its key, optional initializer expression, and any decorator-supplied
/// initializer functions (each maps the current value to a new one during construction).
pub struct FieldInit {
    pub key: crate::value::PropertyKey,
    pub init: Option<Expr>,
    pub transforms: Vec<Value>,
}

/// Direct machine-code calls return to the committed Rust path at this depth. That path is a
/// stack-growth checkpoint, so this threshold changes dispatch strategy without rejecting work.
const MAX_DIRECT_CALL_DEPTH: u32 = 128;

/// wasm32 cannot switch native stacks. This temporary containment guard remains until ordinary
/// calls join coroutine execution on heap-owned VM frames; it must not be confused with the
/// native engine's execution semantics.
#[cfg(target_arch = "wasm32")]
pub(crate) const WASM_EXECUTION_DEPTH_GUARD: u32 = 128;

/// Native execution is allowed to use the current thread stack while it has comfortable
/// headroom. Once a call chain becomes deep, `stacker` moves the next Rust continuation to a
/// heap-backed stack segment. Execution contexts are not native frames (ECMA-262 §9.4),
/// but their native storage must still be finite. Bound simultaneous segment storage rather
/// than JS call count; report exhaustion as a catchable RangeError at an owning call boundary.
/// Proper tail calls still reuse their storage (ECMA-262 §15.10.3).
/// https://tc39.es/ecma262/multipage/executable-code-and-execution-contexts.html#sec-execution-contexts
/// https://tc39.es/ecma262/multipage/ecmascript-language-functions-and-classes.html#sec-preparefortailcall
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const EXECUTION_STACK_RED_ZONE: usize = 1024 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const EXECUTION_STACK_SEGMENT: usize = 8 * 1024 * 1024;
// Host resource policy, not a language recursion-depth limit. The original thread stack is
// owned by the embedder; TRust JS may add at most 512 MiB of simultaneously live native segments.
// This also accommodates the existing 4096-call stress case with large unoptimized VM frames.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const MAX_EXECUTION_STACK_SEGMENTS: usize = 64;
/// A call path checks for exhaustion ([`execution_stack_exhausted`]) and then reaches its growth
/// checkpoint ([`with_execution_stack`]) at the same depth but a little deeper on the stack: in
/// the frame that holds the call's continuation, inside the closures that borrow its arguments. At
/// the segment cap the check keeps this margin above the red zone, which bounds those frames,
/// so a path that passes the check still finds the red zone free at its checkpoint and needs no
/// segment there.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const EXECUTION_STACK_CHECK_MARGIN: usize = 256 * 1024;
/// The most trapless Proxy forwards that may nest. Each forwards natively, without a JavaScript
/// frame, so the whole native stack budget would allow hundreds of thousands, and an unbounded
/// chain (a Proxy on its own target's prototype chain, which fingerprinting scripts probe for
/// every API they inspect) would take a third of a second to reach its RangeError. Engines stop
/// near 4,000 (V8), so no working page nests more.
const MAX_PROXY_FORWARD_DEPTH: u32 = 10_000;
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    // Thread-local so re-entrant Engines/Realms cannot reset the native storage budget.
    static EXECUTION_STACK_SEGMENTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(not(target_arch = "wasm32"))]
// A bytecode execution context can carry substantially more native machinery than a small
// arithmetic recursion frame (accessors and host-native calls are common examples). Checking at
// entry and then every eight contexts leaves the 1 MiB red zone enough room for the measured
// 96 KiB large-frame stress case while removing a platform/TLS stack query from ordinary calls.
// `stacker` still switches segments whenever a checkpoint finds insufficient headroom.
const EXECUTION_STACK_CHECK_MASK: u32 = 7;

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static EXECUTION_STACK_CHECKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Checkpoints that found the red zone reached with the segment cap reached (see
    /// [`with_execution_stack`]).
    static EXECUTION_STACK_CAPPED_CHECKPOINTS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn execution_stack_checks() -> u64 {
    EXECUTION_STACK_CHECKS.with(std::cell::Cell::get)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn execution_stack_segments() -> usize {
    EXECUTION_STACK_SEGMENTS.with(std::cell::Cell::get)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn execution_stack_capped_checkpoints() -> u64 {
    EXECUTION_STACK_CAPPED_CHECKPOINTS.with(std::cell::Cell::get)
}

/// Run `f` with this thread's live segment count set to `segments`, restored afterwards: the
/// accounting at and below the cap, without allocating the segments.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn with_execution_stack_segments<R>(segments: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            EXECUTION_STACK_SEGMENTS.with(|live| live.set(self.0));
        }
    }
    let _restore = Restore(EXECUTION_STACK_SEGMENTS.with(|live| live.replace(segments)));
    f()
}

/// Check before entering a call (and before moving its owned JIT arguments). Leave the red
/// zone available for error construction and unwinding, and at the segment cap the margin above
/// it for the frames between this check and the call's growth checkpoint
/// ([`EXECUTION_STACK_CHECK_MARGIN`]). The caller must preserve normal argument/frame cleanup on
/// failure, just as for any other abrupt completion.
#[inline]
pub(crate) fn execution_stack_exhausted(depth: u32) -> bool {
    #[cfg(target_arch = "wasm32")]
    return depth > WASM_EXECUTION_DEPTH_GUARD;
    #[cfg(not(target_arch = "wasm32"))]
    {
        (depth == 1 || depth & EXECUTION_STACK_CHECK_MASK == 0)
            && EXECUTION_STACK_SEGMENTS.with(|segments| {
                segments.get() >= MAX_EXECUTION_STACK_SEGMENTS
                    && stacker::remaining_stack().is_none_or(|remaining| {
                        remaining < EXECUTION_STACK_RED_ZONE + EXECUTION_STACK_CHECK_MARGIN
                    })
            })
    }
}

/// `stacker::grow` for paths that run JavaScript. The new segment is registered, with exact
/// bounds, for conservative scanning (Decision 0, option (a)); a collection requested on an
/// unregistered segment would defer instead.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn grow_stack<R>(stack_size: usize, f: impl FnOnce() -> R) -> R {
    crate::gc_core::grow(stack_size, f)
}

#[inline]
pub(crate) fn with_execution_stack<R>(depth: u32, f: impl FnOnce() -> R) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if depth == 1 || depth & EXECUTION_STACK_CHECK_MASK == 0 {
            #[cfg(test)]
            EXECUTION_STACK_CHECKS.with(|checks| checks.set(checks.get().wrapping_add(1)));
            if stacker::remaining_stack()
                .is_none_or(|remaining| remaining < EXECUTION_STACK_RED_ZONE)
            {
                if EXECUTION_STACK_SEGMENTS.with(std::cell::Cell::get)
                    >= MAX_EXECUTION_STACK_SEGMENTS
                {
                    // At the cap no segment is added: the call runs on this stack, whose red
                    // zone holds the contexts up to the next checkpoint, where the exhaustion
                    // check reports the RangeError. The check's margin keeps a path that checks
                    // first from getting here; tests count the checkpoints that do.
                    #[cfg(test)]
                    EXECUTION_STACK_CAPPED_CHECKPOINTS.with(|capped| capped.set(capped.get() + 1));
                    return f();
                }
                // Pair native segment allocation with its lifetime, including Rust unwinding.
                // All execution entry points check exhaustion before moving frame ownership.
                struct SegmentGuard;
                impl Drop for SegmentGuard {
                    fn drop(&mut self) {
                        EXECUTION_STACK_SEGMENTS.with(|segments| segments.set(segments.get() - 1));
                    }
                }
                EXECUTION_STACK_SEGMENTS.with(|segments| segments.set(segments.get() + 1));
                let _guard = SegmentGuard;
                return grow_stack(EXECUTION_STACK_SEGMENT, f);
            }
        }
    }
    f()
}

/// Walks over parsed source (the parser, the tree-walking evaluator, the bytecode compiler and
/// the other AST passes) recurse once per nesting level, and the parser bounds that depth
/// (`MAX_PARSE_DEPTH`) rather than the native stack. Each recursive gateway first checks
/// [`ast_stack_low`] and, when it is, re-enters itself through [`with_ast_stack`], so a deep tree
/// continues on fresh heap-allocated segments instead of overflowing whatever stack the walk
/// started on:
///
/// ```ignore
/// if crate::interpreter::ast_stack_low() {
///     return crate::interpreter::with_ast_stack(|| self.walk(node));
/// }
/// ```
///
/// The red zone covers one level's frames (large in debug builds) plus error construction;
/// segments are released when the level returns. The parser bounds how deep these walks go, so
/// their segments are not counted against the execution budget (see [`with_data_stack`] for
/// recursion over author data).
#[cfg(not(target_arch = "wasm32"))]
const AST_STACK_RED_ZONE: usize = 512 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const AST_STACK_SEGMENT: usize = 2 * 1024 * 1024;

/// Whether an AST walk should move to a fresh stack segment before its next level.
#[inline]
pub(crate) fn ast_stack_low() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    return stacker::remaining_stack().is_none_or(|remaining| remaining < AST_STACK_RED_ZONE);
    #[cfg(target_arch = "wasm32")]
    false
}

/// Run one level of an AST walk, on a fresh segment when the native stack is low.
#[inline]
pub(crate) fn with_ast_stack<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    if ast_stack_low() {
        return grow_stack(AST_STACK_SEGMENT, f);
    }
    f()
}

/// The native stack dropping a whole parsed tree may need: drop glue recurses through every
/// nesting level without a gateway, under 100 bytes a level in release builds, up to
/// `MAX_PARSE_DEPTH` levels. (Debug frames are larger; test threads run on large stacks.)
#[cfg(not(target_arch = "wasm32"))]
const AST_DROP_STACK: usize = 512 * 1024;

/// Drop a parsed tree (a function body's statements) where its deepest nesting fits.
pub(crate) fn drop_ast<T>(tree: T) {
    #[cfg(not(target_arch = "wasm32"))]
    if stacker::remaining_stack().is_none_or(|remaining| remaining < AST_DROP_STACK) {
        return stacker::grow(AST_STACK_SEGMENT, move || drop(tree));
    }
    drop(tree)
}

/// The deepest nesting of a native recursion over author data that wasm32, which cannot switch
/// stacks, attempts before throwing.
#[cfg(target_arch = "wasm32")]
const WASM_DATA_DEPTH_GUARD: u32 = 1000;
#[cfg(target_arch = "wasm32")]
thread_local! {
    static DATA_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Run one level of a native recursion whose depth author data controls (JSON.parse and
/// stringify, `flat`): on a fresh segment when the stack runs low, sharing the execution
/// budget of [`with_execution_stack`], so the recursion either completes or reports exhaustion.
/// `None` means the budget is spent and the caller throws the RangeError a JavaScript recursion
/// would ("Maximum call stack size exceeded").
#[inline]
pub(crate) fn with_data_stack<R>(f: impl FnOnce() -> R) -> Option<R> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if stacker::remaining_stack().is_none_or(|remaining| remaining < EXECUTION_STACK_RED_ZONE) {
            if EXECUTION_STACK_SEGMENTS.with(|segments| segments.get())
                >= MAX_EXECUTION_STACK_SEGMENTS
            {
                return None;
            }
            struct SegmentGuard;
            impl Drop for SegmentGuard {
                fn drop(&mut self) {
                    EXECUTION_STACK_SEGMENTS.with(|segments| segments.set(segments.get() - 1));
                }
            }
            EXECUTION_STACK_SEGMENTS.with(|segments| segments.set(segments.get() + 1));
            let _guard = SegmentGuard;
            return Some(grow_stack(EXECUTION_STACK_SEGMENT, f));
        }
        Some(f())
    }
    #[cfg(target_arch = "wasm32")]
    {
        struct DepthGuard;
        impl Drop for DepthGuard {
            fn drop(&mut self) {
                DATA_DEPTH.with(|depth| depth.set(depth.get() - 1));
            }
        }
        if DATA_DEPTH.with(|depth| depth.get()) >= WASM_DATA_DEPTH_GUARD {
            return None;
        }
        DATA_DEPTH.with(|depth| depth.set(depth.get() + 1));
        let _guard = DepthGuard;
        Some(f())
    }
}

/// Smallest automatically derived live-object ceiling (the former fixed limit). When a safe point
/// sees more *live* objects than the engine's ceiling (see [`crate::Engine::live_object_limit`]), a
/// collection runs; if it can't get back under, a RangeError is thrown rather than exhausting
/// RAM. This bounds genuine retention; transient garbage is reclaimed and doesn't count.
pub const MIN_LIVE_OBJECT_LIMIT: i64 = 3_000_000;
/// Largest automatically derived ceiling. An embedder may configure a larger explicit value.
const MAX_DEFAULT_LIVE_OBJECT_LIMIT: i64 = 1 << 28;
/// Ceiling used where the platform cannot report its physical memory.
const FALLBACK_LIVE_OBJECT_LIMIT: i64 = 16_000_000;
/// Conservative retained cost of one ordinary object, including its property storage and
/// collector bookkeeping, used only to translate a memory budget into an object ceiling.
const ESTIMATED_LIVE_OBJECT_BYTES: u64 = 384;

/// The default live-object ceiling: a quarter of the memory available to this process, as
/// objects. Like other engines' default heap limits it scales with the machine instead of
/// capping large applications at a fixed count; embedders override it per engine.
pub(crate) fn default_live_object_limit() -> i64 {
    static LIMIT: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| match crate::host_memory::physical_memory_bytes() {
        Some(bytes) => {
            let objects = bytes / 4 / ESTIMATED_LIVE_OBJECT_BYTES;
            (objects.min(i64::MAX as u64) as i64)
                .clamp(MIN_LIVE_OBJECT_LIMIT, MAX_DEFAULT_LIVE_OBJECT_LIMIT)
        }
        None => FALLBACK_LIVE_OBJECT_LIMIT,
    })
}

/// Smallest live-object limit an embedder can set: [`crate::Engine::set_live_object_limit`]
/// raises smaller values to it.
pub(crate) const MIN_CONFIGURED_LIVE_OBJECT_LIMIT: i64 = 100_000;
/// Objects the handler of the live-object limit's RangeError may allocate before a safepoint
/// collects again: the throwing safepoint sets `Interp::gc_next` this far above the live count
/// (see `gc_check`).
pub(crate) const LIMIT_ERROR_HANDLER_ROOM: i64 = 100_000;
/// Layered calls check allocation counters once per 256 calls (see `gc_check_amortized`).
pub(crate) const GC_CALL_POLL_MASK: u32 = 255;

/// ArrayBuffer data allocated since the last collection that triggers another, at least. A
/// backing store is freed only when the collector finds its buffer unreachable, and a buffer is
/// one object however many bytes it holds, so object pressure alone let discarded buffers pile
/// up (a loop allocating 16 MiB arrays and keeping none reached 16 GB). Like V8's external
/// memory limit, the trigger grows with the buffer data that survives a collection.
const BUFFER_GC_TRIGGER: usize = 64 * 1024 * 1024;

/// One Agent's ArrayBuffer pressure. Every interpreter sharing the heap adds to the same
/// count: a ShadowRealm's buffers die in the same collections as its parent's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BufferPressure {
    pub(crate) since_gc: usize,
    pub(crate) trigger: usize,
}

impl Default for BufferPressure {
    fn default() -> Self {
        BufferPressure {
            since_gc: 0,
            trigger: BUFFER_GC_TRIGGER,
        }
    }
}

/// The trigger grows with the buffer data that survives a collection, whether an ArrayBuffer's
/// own bytes or a SharedArrayBuffer's Shared Data Block. Four 40 MiB buffers are kept: the
/// second allocation collects (one block retained, below the floor) and the fourth collects
/// with three retained. Shared blocks used to count as empty, leaving the trigger at its floor,
/// so programs holding large SharedArrayBuffers collected after every 64 MiB of new buffers.
#[cfg(test)]
#[test]
fn buffer_gc_trigger_grows_with_retained_shared_data_blocks() {
    // The expected trigger assumes exactly the two collections described above.
    if crate::gc_test::stress_active() {
        return;
    }
    for constructor in ["ArrayBuffer", "SharedArrayBuffer"] {
        let mut engine = crate::Engine::new();
        let source = format!(
            "globalThis.kept = []; for (let k = 0; k < 4; k++) kept.push(new {constructor}(40 << 20)); kept.length"
        );
        assert!(matches!(
            engine.eval(&source, false).expect("source parses"),
            crate::Completion::Value(ref value) if value == "4"
        ));
        assert_eq!(
            engine.interp.gc_heap.agent.buffers.get().trigger,
            120 << 20,
            "{constructor} survivors must raise the trigger"
        );
    }
}

/// Minimum pressure that triggers an end-of-task collection. Larger retained heaps scale
/// this floor, so short-lived allocation churn alone does not force a full scan every frame.
const GC_TASK_ALLOCATION_TRIGGER: u64 = 10_000;

/// Memory safety valves. Several built-ins iterate/allocate in proportion to a user-controlled
/// `length`, so without these a single adversarial test (e.g. `Array(4e9).join()` or `s += s`
/// doubling a string) can exhaust all RAM. Operations that would materialize more than these
/// bounds raise a RangeError instead.
///
/// The most elements one operation materializes (`[...a]`, `join`, `sort`, `fill`, `split`,
/// iteration into a list). It matches V8's largest elements backing store (2^27 slots), so the
/// arrays real programs build (CSV rows, large datasets) fit, while an absurd array-like
/// `length` still throws before anything is allocated.
pub const MAX_ARRAY_OP_LEN: usize = 1 << 27; // ~134M elements
/// The most arguments one call receives from a spread or an `apply`/`Reflect.apply` list. Each
/// is copied into the callee's frame and `arguments`; V8 runs out of stack near 120,000 of them
/// and SpiderMonkey stops at 500,000.
pub const MAX_ARGUMENTS_LEN: usize = 1 << 20; // ~1M arguments

// A string splits or spreads into at most one piece per byte (plus one), so string operations
// never reach the array limit.
const _: () = assert!(MAX_STR_LEN < MAX_ARRAY_OP_LEN);

/// Match Node/V8's non-sandboxed ArrayBuffer ceiling: the maximum safe integer on 64-bit
/// targets, or 2 GiB minus one on 32-bit targets. Allocation is fallible; unavailable virtual
/// memory still produces a RangeError. Node v24.21.0: deps/v8/include/v8-array-buffer.h.
#[cfg(target_pointer_width = "64")]
pub const MAX_BUFFER_BYTES: usize = ((1u64 << 53) - 1) as usize;
#[cfg(target_pointer_width = "32")]
pub const MAX_BUFFER_BYTES: usize = i32::MAX as usize;
// Large enough for suite tests that build 2^24-char escape strings, small enough that runaway
// string growth still dies as a RangeError rather than an OOM.
pub const MAX_STR_LEN: usize = 1 << 26; // ~67M

// Builders check MAX_STR_LEN as they accumulate. A few (escaping, case mapping, normalization)
// only bound their result by a fixed expansion of an input that is within the limit (at most
// 12x, for NFKD); keep that headroom below what an engine string can represent.
const _: () = assert!(MAX_STR_LEN * 15 <= crate::lstr::MAX_LEN);

#[derive(Clone)]
pub(crate) struct HostSettingsState {
    pub(crate) global_env: Env,
    pub(crate) global_var_names: Rc<RefCell<std::collections::HashSet<String>>>,
}

/// See [`Interp::jit_admit_runs`].
pub(crate) const DEFAULT_JIT_ADMIT_RUNS: u32 = 0;

/// One decoded static host snapshot (see `Interp::shared_snapshots`). Every evaluation runs a
/// fresh ScriptEvaluation of the same Script body in the then-active Realm: function objects,
/// environments and template objects are created anew per Realm, while the immutable AST and
/// the bytecode/machine code cached on its functions are shared. Compiled code holds no object
/// owners; its caches are validated against live receivers, environments and Realms.
#[cfg_attr(not(any(feature = "embed", test)), allow(dead_code))]
pub(crate) struct SharedSnapshot {
    pub(crate) address: usize,
    pub(crate) len: usize,
    pub(crate) body: Rc<Vec<Stmt>>,
}

/// The proof behind [`Interp::ta_accessor_protector`]. While `epoch` matches the live
/// [`crate::value::proto_epoch`], `holder` (normally a Realm's %TypedArray.prototype%) is an
/// ordinary object whose own `length`, `byteLength`, `byteOffset` and `buffer` are the matching
/// intrinsic accessors, and each set `concrete` entry an ordinary object without those keys whose
/// [[Prototype]] is `holder` (a concrete TypedArray prototype). All are marked prototypes, so a
/// new or deleted key, a redefinition or a [[SetPrototypeOf]] moves the epoch on; the weak
/// references keep their addresses from reuse.
#[derive(Default)]
pub(crate) struct TaAccessorProof {
    epoch: u32,
    holder: crate::value::WeakGc,
    concrete: [crate::value::WeakGc; 4],
    next_concrete: usize,
}

/// A realm's intrinsics: the global object, its environment, and the per-realm prototypes/constructors
/// installed by `builtins::install`. Well-known symbols and the engine side tables are shared, not here.
pub struct RealmState {
    pub global: Gc,
    pub global_env: Env,
    pub global_var_names: Rc<RefCell<std::collections::HashSet<String>>>,
    /// HTML's environment-settings object associated with this Realm. This is
    /// opaque to ECMAScript, but restoring it with the Realm is required for
    /// synchronous cross-Realm calls as well as Promise jobs.
    pub host_job_context: u64,
    pub object_proto: Gc,
    pub function_proto: Gc,
    pub array_proto: Gc,
    pub array_ctor: Option<Gc>,
    pub string_proto: Gc,
    pub number_proto: Gc,
    pub boolean_proto: Gc,
    pub symbol_proto: Gc,
    pub error_protos: IntrinsicMap,
    pub eval_fn: Option<Gc>,
    pub extra_protos: IntrinsicMap,
}

impl RealmState {
    pub(crate) fn scan_retained_memory(
        &self,
        visitor: &mut crate::memory::Visitor,
    ) -> (usize, bool) {
        let (global_names, global_names_exact) = visitor.global_var_names(&self.global_var_names);
        let bytes = global_names
            .saturating_add(
                self.error_protos
                    .len()
                    .saturating_mul(std::mem::size_of::<(&'static str, Gc)>()),
            )
            .saturating_add(
                self.extra_protos
                    .len()
                    .saturating_mul(std::mem::size_of::<(&'static str, Gc)>()),
            );
        // Global/intrinsic objects are canonical to the object snapshot, and `global_env` is
        // canonical to the scope snapshot. Only the two map allocations and global-name set are
        // owned below this record.
        (
            bytes,
            global_names_exact && self.error_protos.is_empty() && self.extra_protos.is_empty(),
        )
    }
}

impl Interp {
    pub(crate) fn construct_ic_retained_memory(&self) -> (usize, bool) {
        (
            self.construct_ics
                .len()
                .saturating_mul(std::mem::size_of::<(usize, ConstructIc)>()),
            self.construct_ics.is_empty(),
        )
    }

    /// Snapshot the current realm's intrinsics so they can be swapped out and back.
    pub(crate) fn snapshot_realm(&self) -> RealmState {
        RealmState {
            global: self.global,
            global_env: self.global_env,
            global_var_names: self.global_var_names.clone(),
            host_job_context: self.host_job_context,
            object_proto: self.object_proto,
            function_proto: self.function_proto,
            array_proto: self.array_proto,
            array_ctor: self.array_ctor,
            string_proto: self.string_proto,
            number_proto: self.number_proto,
            boolean_proto: self.boolean_proto,
            symbol_proto: self.symbol_proto,
            error_protos: self.error_protos.clone(),
            eval_fn: self.eval_fn,
            extra_protos: self.extra_protos.clone(),
        }
    }

    /// Install a realm's intrinsics as the active ones.
    pub(crate) fn restore_realm(&mut self, r: &RealmState) {
        self.global = r.global;
        self.global_env = r.global_env;
        self.global_var_names = r.global_var_names.clone();
        self.host_job_context = r.host_job_context;
        self.object_proto = r.object_proto;
        self.function_proto = r.function_proto;
        self.array_proto = r.array_proto;
        self.array_ctor = r.array_ctor;
        self.string_proto = r.string_proto;
        self.number_proto = r.number_proto;
        self.boolean_proto = r.boolean_proto;
        self.symbol_proto = r.symbol_proto;
        self.error_protos = r.error_protos.clone();
        self.eval_fn = r.eval_fn;
        self.extra_protos = r.extra_protos.clone();
        // The protectors prove facts about the active Realm's intrinsics; epoch 0 is never
        // live, so the next use re-verifies against the restored ones.
        self.elems_protector.set((0, false));
        self.species_protector.set((0, u32::MAX));
        *self.array_iteration_protector.borrow_mut() = Default::default();
        self.regexp_protector.borrow_mut().invalidate();
    }

    /// Exchange an owned Realm snapshot with the active one. Cross-Realm calls
    /// retain the caller's owners in `saved` until return, so transferring those
    /// owners avoids cloning and dropping every intrinsic twice per boundary.
    /// The registry still owns the target snapshot throughout author execution.
    pub(crate) fn swap_realm(&mut self, saved: &mut RealmState) {
        std::mem::swap(&mut self.global, &mut saved.global);
        std::mem::swap(&mut self.global_env, &mut saved.global_env);
        std::mem::swap(&mut self.global_var_names, &mut saved.global_var_names);
        std::mem::swap(&mut self.host_job_context, &mut saved.host_job_context);
        std::mem::swap(&mut self.object_proto, &mut saved.object_proto);
        std::mem::swap(&mut self.function_proto, &mut saved.function_proto);
        std::mem::swap(&mut self.array_proto, &mut saved.array_proto);
        std::mem::swap(&mut self.array_ctor, &mut saved.array_ctor);
        std::mem::swap(&mut self.string_proto, &mut saved.string_proto);
        std::mem::swap(&mut self.number_proto, &mut saved.number_proto);
        std::mem::swap(&mut self.boolean_proto, &mut saved.boolean_proto);
        std::mem::swap(&mut self.symbol_proto, &mut saved.symbol_proto);
        std::mem::swap(&mut self.error_protos, &mut saved.error_protos);
        std::mem::swap(&mut self.eval_fn, &mut saved.eval_fn);
        std::mem::swap(&mut self.extra_protos, &mut saved.extra_protos);
        self.elems_protector.set((0, false));
        self.species_protector.set((0, u32::MAX));
        *self.array_iteration_protector.borrow_mut() = Default::default();
        self.regexp_protector.borrow_mut().invalidate();
    }

    /// Select the global lexical state associated with an embedder settings
    /// object while retaining this ECMAScript realm's global object and
    /// intrinsics. Context zero is the default standalone state.
    pub(crate) fn switch_host_job_context(&mut self, context: u64) {
        let realm = Gc::as_ptr(&self.global) as usize;
        if context == self.host_job_context
            && self
                .host_settings_states
                .get(&(realm, context))
                .is_some_and(|state| {
                    Env::ptr_eq(&state.global_env, &self.global_env)
                        && Rc::ptr_eq(&state.global_var_names, &self.global_var_names)
                })
        {
            return;
        }
        if self.retired_host_job_contexts.contains(&context) {
            return;
        }
        // A freshly-created embedder Realm starts with the creator's context
        // token, but its initial `RealmState` already owns that global
        // environment.  When Window bootstrap immediately switches it to its
        // first nonzero settings context, retaining a second `(realm, 0)`
        // state would make that otherwise replaced environment an external
        // Rust root forever.  Preserve context zero only after this Realm has
        // actually been multiplexed before; genuine context switches still
        // need the old state for restoration.
        let has_saved_state = self
            .host_settings_states
            .contains_key(&(realm, self.host_job_context));
        if !self
            .retired_host_job_contexts
            .contains(&self.host_job_context)
            && (self.host_job_context != 0 || has_saved_state)
        {
            self.host_settings_states.insert(
                (realm, self.host_job_context),
                HostSettingsState {
                    global_env: self.global_env,
                    global_var_names: self.global_var_names.clone(),
                },
            );
        }
        let state = if let Some(state) = self.host_settings_states.get(&(realm, context)) {
            state.clone()
        } else {
            let global_env = new_var_scope(None);
            global_env.borrow_mut().vars.insert(
                crate::atom::names::THIS.atom(),
                Binding::data(Value::Obj(self.global), false, true),
            );
            let state = HostSettingsState {
                global_env,
                global_var_names: Rc::new(RefCell::new(std::collections::HashSet::new())),
            };
            self.host_settings_states
                .insert((realm, context), state.clone());
            state
        };
        self.global_env = state.global_env;
        self.global_var_names = state.global_var_names;
        self.host_job_context = context;
    }

    /// Retire one embedder settings object after its owning Document is
    /// destroyed. HTML removes tasks belonging to the destroyed Document; the
    /// environment settings object and its per-Document module map must no
    /// longer be permanent engine roots either. The currently-running
    /// environment is allowed to retire itself; its active references are
    /// released on the next context switch and are not snapshotted again.
    pub fn release_host_job_context(&mut self, context: u64) -> bool {
        if context == 0 || !self.retired_host_job_contexts.insert(context) {
            return false;
        }
        // The Realms this settings object belongs to. A pending reaction registered while it was
        // current goes with it, unless a handler is a function of another Realm: HTML
        // HostEnqueuePromiseJob takes the job settings from the handler's Realm, so a reaction
        // that author code of another Window attached through this Realm's `then` survives.
        let mut owners: std::collections::HashSet<usize> = self
            .host_settings_states
            .keys()
            .filter(|(_, candidate)| *candidate == context)
            .map(|(realm, _)| *realm)
            .collect();
        owners.extend(
            self.realms
                .iter()
                .filter(|(_, state)| state.host_job_context == context)
                .map(|(realm, _)| *realm),
        );
        if self.host_job_context == context {
            owners.insert(Gc::as_ptr(&self.global) as usize);
        }
        self.host_settings_states
            .retain(|(_, candidate), _| *candidate != context);
        {
            let id = self.realm_id;
            let mut work = self.gc_heap.agent.work.borrow_mut();
            work.promises
                .retain(|job| job.route.target != id || job.payload.host_context != context);
            work.cleanup
                .retain(|job| job.route.target != id || job.payload.registry_context != context);
        }
        let cancelled: Vec<_> = self
            .routed_imports
            .iter()
            .filter(|(_, pending)| pending.context == context)
            .map(|(&owner, _)| owner)
            .collect();
        for owner in cancelled {
            self.routed_imports.remove(&owner);
        }
        for promise in self.promises.values_mut(&self.gc_heap) {
            promise
                .reactions
                .retain(|(fulfilled, rejected, _, candidate)| {
                    candidate.context != context
                        || [fulfilled, rejected].iter().any(|handler| {
                            function_realm_key(&handler.callback)
                                .is_some_and(|realm| !owners.contains(&realm))
                        })
                });
        }
        self.cancel_load_authority(crate::agent_control::LoadAuthority {
            agent: crate::value::heap_id(&self.gc_heap),
            owner: self.realm_id,
            context,
        });
        // HTML #discard-a-document releases settings/module roots independently
        // of allocation pressure. Revisit that graph at the next safe boundary,
        // rather than charging its eventual destruction to unrelated new work.
        // ECMA-262 #sec-liveness / #sec-weakref-invariants still protect retained
        // objects and the current job: this only coalesces a collection request,
        // honoring the host's existing task deferral and ClearKeptObjects order.
        // The released graph is mostly old, which a minor collection keeps (S6 step
        // 10c): the boundary's collection is major. A collection in progress (a host
        // sweep retiring a context) leaves the request to the boundary's minor one
        // and the graph to the next major.
        self.gc_heap.agent.task_pending.set(true);
        if let Ok(mut core) = self.gc_heap.core.try_borrow_mut() {
            core.request_major();
        }
        true
    }

    /// `$262.createRealm()`: build a fresh realm (its own global + intrinsics) and register it. The
    /// well-known symbols are shared with the creating realm so `@@iterator` etc. match cross-realm.
    /// Whether a SECOND realm exists (`$262.createRealm`). `Interp::new` always registers the
    /// main realm, so `realms.is_empty()` never distinguishes anything — cross-realm dispatch
    /// only becomes possible once another live realm joins. The collector removes an unreachable
    /// realm as one intrinsic group, so single-realm fast paths recover after temporary realms die.
    #[inline]
    pub(crate) fn multi_realm(&self) -> bool {
        self.realms.len() > 1
    }

    pub(crate) fn create_realm(&mut self) -> Value {
        // Register the creating (main) realm too, so cross-realm dispatch can switch BACK to it
        // and resolve its intrinsics like any other realm's.
        if self.realms.is_empty() {
            let main = self.snapshot_realm();
            self.gc_pin(&main.global);
            self.realms.insert(Gc::as_ptr(&main.global) as usize, main);
        }
        let saved = crate::rooting::Rooted::new(self.snapshot_realm());

        // Fresh intrinsics, mirroring `Interp::new`. The new realm's Symbol constructor gets the
        // Agent's well-known Symbols (`Interp::well_known_symbol`), so `@@iterator` etc. match
        // cross-realm whatever the creating realm's global `Symbol` now holds.
        let object_proto = Object::new(None);
        let function_proto = Object::new(Some(object_proto));
        let array_proto = Object::new(Some(object_proto));
        let string_proto = Object::new(Some(object_proto));
        let number_proto = Object::new(Some(object_proto));
        let boolean_proto = Object::new(Some(object_proto));
        string_proto.borrow_mut().exotic = Exotic::str_wrap("".into());
        number_proto.borrow_mut().exotic = Exotic::NumWrap(0.0);
        boolean_proto.borrow_mut().exotic = Exotic::BoolWrap(false);
        let symbol_proto = Object::new(Some(object_proto));
        let global = Object::new(Some(object_proto));
        self.object_proto = object_proto;
        self.function_proto = function_proto;
        self.array_proto = array_proto;
        self.array_ctor = None;
        self.string_proto = string_proto;
        self.number_proto = number_proto;
        self.boolean_proto = boolean_proto;
        self.symbol_proto = symbol_proto;
        self.global = global;
        self.global_env = new_var_scope(None);
        self.error_protos = Default::default();
        self.extra_protos = Default::default();
        self.eval_fn = None;
        crate::builtins::install(self);
        // Top-level `this` is the new global.
        let g = Value::Obj(self.global);
        self.global_env
            .borrow_mut()
            .vars
            .insert("this".to_string(), Binding::data(g, false, true));
        // Tag this realm's `eval` so an *indirect* call to it (`otherRealm.eval(code)`) runs the code
        // in this realm's global scope, not the caller's (see `call_inner`).
        if let Some(ef) = &self.eval_fn {
            ef.borrow_mut().internal_slots_mut().insert(
                "#\u{0}eval_realm",
                Property::data(Value::Obj(self.global), false, false, false),
            );
            self.eval_realm_fns.insert(Gc::as_ptr(ef) as usize);
        }

        let realm_state = self.snapshot_realm();
        let ptr = Gc::as_ptr(&global) as usize;
        self.gc_pin(&global);
        self.realms.insert(ptr, realm_state);
        self.restore_realm(&saved);
        Value::Obj(global)
    }

    /// Run the Rust text `src` as a Script in the Realm whose global is `realm_global`:
    /// ParseScript then ScriptEvaluation (ECMA-262 #sec-scriptevaluation), so top-level
    /// `let`/`const`/`class` declarations join that Realm's global lexical environment, unlike
    /// an indirect `eval`. A parse failure is that Realm's SyntaxError.
    pub(crate) fn eval_in_realm(
        &mut self,
        realm_global: &Value,
        src: &str,
    ) -> Result<Value, Abrupt> {
        let text = crate::jstr::from_text(src);
        self.eval_script_in_realm(realm_global, &text)
    }

    /// [`Self::eval_in_realm`] for source already in the engine's string representation.
    pub(crate) fn eval_script_in_realm(
        &mut self,
        realm_global: &Value,
        src: &str,
    ) -> Result<Value, Abrupt> {
        self.in_realm(
            realm_global,
            |interp| match crate::parser::parse_script_from(
                src,
                false,
                crate::stack_trace::SourceKind::Script,
                None,
            ) {
                Ok((body, source)) => interp.run_program_from(&body, Some(source)),
                Err(error) => Err(interp.throw("SyntaxError", error.message)),
            },
        )
    }

    /// A call of another Realm's %eval%: PerformEval(x, strictCaller = false, direct = false)
    /// in that function's Realm (ECMA-262 #sec-eval-x), so its `var`s and sloppy assignments
    /// reach that Realm's global object.
    pub(crate) fn indirect_eval_in_realm(
        &mut self,
        realm_global: &Value,
        src: &str,
    ) -> Result<Value, Abrupt> {
        self.in_realm(realm_global, |interp| {
            let env = interp.global_env;
            interp.perform_eval(src, &env, false)
        })
    }

    /// Run `body` with the Realm whose global is `realm_global` current, keeping that Realm's
    /// updated state afterwards.
    fn in_realm(
        &mut self,
        realm_global: &Value,
        body: impl FnOnce(&mut Self) -> Result<Value, Abrupt>,
    ) -> Result<Value, Abrupt> {
        let ptr = match &(realm_global).view() {
            ValueView::Obj(o) => Gc::as_ptr(o) as usize,
            _ => return Err(self.throw("TypeError", "not a realm global")),
        };
        let target = match self.realms.get(&ptr) {
            Some(realm) => realm.snapshot_clone(),
            None => return Err(self.throw("TypeError", "unknown realm")),
        };
        self.with_realm_snapshot(ptr, target, body)
    }
}

impl RealmState {
    pub(crate) fn snapshot_clone(&self) -> RealmState {
        RealmState {
            global: self.global,
            global_env: self.global_env,
            global_var_names: self.global_var_names.clone(),
            host_job_context: self.host_job_context,
            object_proto: self.object_proto,
            function_proto: self.function_proto,
            array_proto: self.array_proto,
            array_ctor: self.array_ctor,
            string_proto: self.string_proto,
            number_proto: self.number_proto,
            boolean_proto: self.boolean_proto,
            symbol_proto: self.symbol_proto,
            error_protos: self.error_protos.clone(),
            eval_fn: self.eval_fn,
            extra_protos: self.extra_protos.clone(),
        }
    }
}

#[cfg(feature = "embed")]
impl Interp {
    /// Create a fresh ECMAScript Realm in this Agent and return its global
    /// object. HTML embedders use this for Window globals; the Realm shares
    /// Agent-wide well-known symbols and the GC heap but owns independent
    /// intrinsics and a Global Environment Record.
    pub fn create_embed_realm(&mut self) -> Value {
        self.create_realm()
    }

    pub(crate) fn enter_embed_realm(
        &mut self,
        realm_global: &Value,
    ) -> Result<(usize, RealmState), Value> {
        let key = match &(realm_global).view() {
            ValueView::Obj(global) => Gc::as_ptr(global) as usize,
            _ => return Err(self.make_error("TypeError", "not a realm global")),
        };
        let mut saved = match self.realms.get(&key) {
            Some(realm) => realm.snapshot_clone(),
            None => return Err(self.make_error("TypeError", "unknown realm")),
        };
        self.swap_realm(&mut saved);
        Ok((key, saved))
    }

    pub(crate) fn leave_embed_realm(&mut self, key: usize, mut saved: RealmState) {
        self.swap_realm(&mut saved);
        self.realms.insert(key, saved);
    }

    /// Run one embedder operation with `realm_global` installed as the active
    /// Realm, then restore the caller's Realm even when the operation returns
    /// an error value. Realm-owned mutations are retained in the registry.
    pub fn with_embed_realm<R>(
        &mut self,
        realm_global: &Value,
        operation: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, Value> {
        let key = realm_global
            .as_obj()
            .map(|global| Gc::as_ptr(&global) as usize)
            .ok_or_else(|| self.make_error("TypeError", "not a realm global"))?;
        let target = self
            .realms
            .get(&key)
            .map(RealmState::snapshot_clone)
            .ok_or_else(|| self.make_error("TypeError", "unknown realm"))?;
        Ok(self.with_realm_snapshot(key, target, operation))
    }

    /// The opaque HTML environment-settings token currently associated with
    /// jobs created in the active host operation.
    pub fn host_job_context(&self) -> u64 {
        self.host_job_context
    }

    /// The promise state of `value`, when it is a promise.
    fn promise_state_mut(&mut self, value: &Value) -> Option<&mut PromiseState> {
        let ValueView::Obj(object) = &(value).view() else {
            return None;
        };
        self.promises
            .get_mut(&self.gc_heap, &(Gc::as_ptr(object) as usize))
    }

    /// A promise's [[PromiseIsHandled]] (None when `promise` is not a promise): whether a
    /// handler was ever attached, or the host marked it handled. HTML's "notify about rejected
    /// promises" skips a handled promise and keeps an unhandled one as outstanding.
    pub fn promise_is_handled(&mut self, promise: &Value) -> Option<bool> {
        self.promise_state_mut(promise).map(|state| state.handled)
    }

    /// Web IDL "mark as handled": set [[PromiseIsHandled]] to true, so the rejection is never
    /// reported as unhandled. False when `promise` is not a promise.
    pub fn mark_promise_handled(&mut self, promise: &Value) -> bool {
        let Some(state) = self.promise_state_mut(promise) else {
            return false;
        };
        state.handled = true;
        state.outstanding = None;
        if let ValueView::Obj(object) = &(promise).view() {
            self.unhandled_rejections
                .remove(&(Gc::as_ptr(object) as usize));
        }
        true
    }

    /// Add a rejected promise whose `unhandledrejection` the host fired and which is still not
    /// handled to the outstanding rejected promises weak set (HTML "notify about rejected
    /// promises") of the global of the active host settings context, the one that fired the
    /// event. The set holds the promise weakly; when a handler is attached later,
    /// HostPromiseRejectionTracker's "handle" removes it and reports it through
    /// [`crate::Engine::take_handled_rejections`]. False (and no change) for anything else.
    pub fn add_outstanding_rejection(&mut self, promise: &Value) -> bool {
        let context = self.host_job_context;
        match self.promise_state_mut(promise) {
            Some(state) if state.status == 2 && !state.handled => {
                state.outstanding = Some(context);
                true
            }
            _ => false,
        }
    }

    /// The HostPromiseRejectionTracker "handle" operations since the last call on promises of
    /// an outstanding rejected promises weak set, as `(context, promise, reason)` in order: each
    /// is a `rejectionhandled` event the host queues (HTML #unhandled-promise-rejections) at the
    /// global of the host settings context whose weak set held the promise.
    pub fn take_handled_rejections_in_contexts(&mut self) -> Vec<(u64, Value, Value)> {
        std::mem::take(&mut self.handled_rejections)
            .into_iter()
            .map(|(promise, reason, context)| (context, promise, reason))
            .collect()
    }

    /// Web IDL "a promise resolved with" `value` for an embedder's platform steps: a new
    /// promise of the current Realm, resolved with `value`. When `value` is a thenable, adopting
    /// its state is platform work as well: the NewPromiseResolveThenableJob and the reactions of
    /// the promise's resolving functions get null job settings, like the jobs of
    /// [`Self::react_to_promise`], so they survive [`Self::release_host_job_context`] of the
    /// settings object that created the promise. A [`Self::react_to_promise`] step that returns
    /// a thenable resolves its result promise the same way.
    pub fn promise_resolved_with(&mut self, value: Value) -> Value {
        let promise = self.new_promise();
        self.resolve_promise_from(&promise, value, true);
        promise
    }

    /// Web IDL "react to a promise" with embedder steps: PerformPromiseThen(`promise`,
    /// `on_fulfilled`, `on_rejected`) into a new promise of the current Realm, which is
    /// returned. The steps are platform-internal, not author callbacks: their jobs get null job
    /// settings (HTML HostEnqueuePromiseJob with a null Realm), so they run in whatever settings
    /// are current and survive [`Self::release_host_job_context`] of the settings object that
    /// registered them. A missing step (`undefined`) passes the settlement through. Like any
    /// PerformPromiseThen this marks `promise` handled. TypeError when `promise` is not a promise.
    pub fn react_to_promise(
        &mut self,
        promise: &Value,
        on_fulfilled: Value,
        on_rejected: Value,
    ) -> Result<Value, Value> {
        if self.promise_state_mut(promise).is_none() {
            return Err(self.make_error("TypeError", "react to a promise: not a promise"));
        }
        let step = |callback: Value| JobCallback {
            internal: InternalJob::HostReaction,
            ..JobCallback::plain(if callback.is_callable() {
                callback
            } else {
                Value::Undefined
            })
        };
        let result = self.new_promise();
        self.perform_promise_then(promise, step(on_fulfilled), step(on_rejected), result);
        Ok(result)
    }
}

/// The Realm key of an ECMAScript function or bound function's target (None for built-ins
/// and other callables, whose Realm is not recorded on the object).
fn function_realm_key(callback: &Value) -> Option<usize> {
    let ValueView::Obj(function) = &(callback).view() else {
        return None;
    };
    let function = function.borrow();
    match &function.call {
        Callable::User(user) => Some(user.realm),
        Callable::Bound(bound) => function_realm_key(&Value::Obj(bound.target)),
        _ => None,
    }
}

/// The intrinsic %TypedArray.prototype% getter that OrdinaryGet of `key` finds starting at
/// `start`, and the object holding it, when every object it consults is an ordinary one whose
/// properties all live in its property map (no Proxy, TypedArray, namespace, host or wrapper
/// exotic).
fn ta_accessor_on_chain(start: &Gc, key: &str) -> Option<(TaAccessor, Gc)> {
    let mut cur = Some(*start);
    while let Some(object) = cur {
        let next = {
            let b = object.borrow();
            if !b.ic_plain.get() || !b.exotic.is_ordinary() {
                return None;
            }
            if let Some(property) = b.props.get(key) {
                let found = property
                    .getter()
                    .filter(|_| property.accessor())
                    .and_then(crate::builtins::ta_accessor_of)?;
                drop(b);
                return Some((found, object));
            }
            b.proto
        };
        cur = next;
    }
    None
}

/// Whether `holder` is an ordinary object whose own `length`, `byteLength`, `byteOffset` and
/// `buffer` are each the matching intrinsic %TypedArray.prototype% accessor.
fn ta_accessor_holder_is_pristine(holder: &Gc) -> bool {
    let holder = holder.borrow();
    holder.ic_plain.get()
        && holder.exotic.is_ordinary()
        && [
            TaAccessor::Length,
            TaAccessor::ByteLength,
            TaAccessor::ByteOffset,
            TaAccessor::Buffer,
        ]
        .into_iter()
        .all(|accessor| {
            holder.props.get(accessor.name()).is_some_and(|property| {
                property.accessor()
                    && property.getter().and_then(crate::builtins::ta_accessor_of) == Some(accessor)
            })
        })
}

impl Interp {
    pub(crate) fn new() -> Interp {
        Self::new_with_symbol_agent(crate::value::new_symbol_agent())
    }

    /// Construct another realm implementation within the same ECMAScript Agent. ShadowRealm has
    /// an isolated global/object heap, but ECMA-262 §20.4 requires its global and well-known
    /// Symbols to retain Agent-wide identity.
    pub(crate) fn new_with_symbol_agent(symbol_agent: crate::value::SymbolAgent) -> Interp {
        Self::new_in_heap(symbol_agent, crate::value::new_gc_heap(), true)
    }

    /// A ShadowRealm's implementation interpreter, owned by the ShadowRealm object `owner`:
    /// same Agent (Symbols), isolated globals and intrinsics. It allocates in its parent's heap
    /// — one heap per Agent, so no edge between the two realms' objects ever crosses heaps (I6)
    /// — and the root of the ShadowRealm tree collects for all of it.
    pub(crate) fn new_shadow(parent: &Interp, owner: &Gc) -> crate::agent_access::Owned<Interp> {
        let _unpublished = crate::agent_control::Suppression::new(&parent.gc_heap);
        let access = parent.gc_heap.agent.access.clone();
        let id = access.fresh_id();
        let mut realm =
            Self::new_in_heap(parent.symbol_agent.clone(), parent.gc_heap.clone(), false);
        realm.realm_id = id;
        let mut owners = parent.realm_route().owners.to_vec();
        owners.push(*owner);
        realm.shadow_home = Some(ShadowHome {
            owner: Gc::downgrade(owner),
            parent: parent.realm_id,
            route: Rc::new(crate::agent_control::Route {
                agent: crate::value::heap_id(&parent.gc_heap),
                target: id,
                owners: owners.into_boxed_slice(),
            }),
        });
        // Counted while `shadow_home` is set; `Drop` uncounts it.
        let children = &parent.gc_heap.agent.children;
        children.set(children.get() + 1);
        if children.get() == 1 {
            // A lone interpreter kept no owner index. Its metadata is foreign from now on.
            for identity in parent
                .gc_pins
                .keys()
                .copied()
                .chain(parent.native_function_realms.borrow().keys().copied())
            {
                parent.remember_metadata_owner(identity);
            }
        }
        // No callback/collection occurs between assigning identity and adoption. Include
        // native callables as well as gc_pin owners created during unpublished bootstrap.
        for identity in realm
            .gc_pins
            .keys()
            .copied()
            .chain(realm.native_function_realms.borrow().keys().copied())
        {
            realm.remember_metadata_owner(identity);
        }
        // Registration completes before suppression ends; the caller's table publication has
        // no intervening safepoint. Allocation itself never collects.
        let owner = access.adopt(id, parent.realm_id, realm);
        debug_assert_eq!(access.parent(id), Some(parent.realm_id));
        owner
    }

    /// Suspend this current borrow and run in a raw-owned child. The closure cannot export
    /// an Interp reference; all owner/table borrows end before entering author code.
    pub(crate) fn with_shadow<R>(
        &mut self,
        realm: &Gc,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> Option<R> {
        let id = self.shadow_realms.get(&(Gc::as_ptr(realm) as usize))?.id();
        Some(self.with_route(id, operation))
    }

    pub(crate) fn shadow_entered(&self) -> bool {
        self.gc_heap.agent.access.active(self.realm_id)
    }

    fn new_in_heap(
        symbol_agent: crate::value::SymbolAgent,
        gc_heap: crate::value::GcHeap,
        owns_heap: bool,
    ) -> Interp {
        let _unpublished = crate::agent_control::Suppression::new(&gc_heap);
        // Decision 0: every engine frame on this thread lies below the scanned stack top.
        crate::gc_core::register_thread_native_stack();
        #[cfg(all(test, target_os = "linux"))]
        crate::gc_test::install_crash_backtrace();
        crate::value::activate_agent(&gc_heap, &symbol_agent);
        let object_proto = Object::new(None);
        let function_proto = Object::new(Some(object_proto));
        let array_proto = Object::new(Some(object_proto));
        let string_proto = Object::new(Some(object_proto));
        let number_proto = Object::new(Some(object_proto));
        let boolean_proto = Object::new(Some(object_proto));
        // These prototypes are themselves wrapper exotics with default primitive data, so e.g.
        // `Number.prototype.valueOf()` / `Number.prototype == 0` work.
        string_proto.borrow_mut().exotic = Exotic::str_wrap("".into());
        number_proto.borrow_mut().exotic = Exotic::NumWrap(0.0);
        boolean_proto.borrow_mut().exotic = Exotic::BoolWrap(false);
        let symbol_proto = Object::new(Some(object_proto));
        let global = Object::new(Some(object_proto));
        let global_env = new_var_scope(None);
        let live_object_limit = gc_heap.agent.next.get();
        let mut interp = Interp {
            gc_heap: gc_heap.clone(),
            realm_id: if owns_heap {
                crate::agent_access::RealmId::ROOT
            } else {
                crate::agent_access::RealmId::UNPUBLISHED
            },
            symbol_agent,
            global,
            global_env,
            object_proto,
            function_proto,
            array_proto,
            array_ctor: None,
            string_proto,
            number_proto,
            boolean_proto,
            symbol_proto,
            error_protos: Default::default(),
            console: Vec::new(),
            wall_clock: None,
            runtime_interrupt: Default::default(),
            interrupt_poll_tick: 0,
            ast_tick: 0,
            strict: false,
            depth: 0,
            direct_call_depth: MAX_DIRECT_CALL_DEPTH,
            proxy_forward_depth: 0,
            class_info: Default::default(),
            eval_fn: None,
            eval_realm_fns: Default::default(),
            native_function_realms: Default::default(),
            elems_protector: std::cell::Cell::new((0, false)),
            species_protector: std::cell::Cell::new((0, u32::MAX)),
            ta_accessor_protector: RefCell::default(),
            array_iteration_protector: Default::default(),
            construct_ics: Default::default(),
            call_overflow: Default::default(),
            native_callback_cache: None,
            proxy_trap_caches: None,
            self_hosted: std::cell::OnceCell::new(),
            shared_snapshots: Vec::new(),
            construct_capacity_hints: Default::default(),
            iterator_sym: None,
            wk_syms: Vec::new(),
            short_circuit: false,
            import_meta: None,
            import_base: String::new(),
            modules: Default::default(),
            module_recs: Default::default(),
            module_loader: None,
            dynamic_module_loader: None,
            routed_imports: Default::default(),
            module_ns: Default::default(),
            tier: match std::env::var("TRUST_JS_TIER").as_deref() {
                Ok("interp") => crate::bytecode::Tier::Interp,
                Ok("bytecode") => crate::bytecode::Tier::Bytecode,
                _ => crate::bytecode::Tier::Jit,
            },
            tier_threshold: std::env::var("TRUST_JS_TIER_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            jit_admit_runs: std::env::var("TRUST_JS_JIT_ADMIT_RUNS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_JIT_ADMIT_RUNS),
            vm_pool: Vec::new(),
            native_arg_pool: Vec::new(),
            stub_cache: vec![std::cell::Cell::new(StubEntry::default()); STUB_CACHE_SIZE],
            arguments_shapes: None,
            descriptor_maps: Default::default(),
            iter_result_map: Default::default(),
            regexp_array_maps: Default::default(),
            stub_cache_names: std::cell::RefCell::new(vec![None; STUB_CACHE_SIZE]),
            computed_reads: Default::default(),
            fragment_cache: Default::default(),
            enumeration_keys: crate::cache::ByteLru::new(1 << 20, 256),
            property_key_strings: crate::cache::ByteLru::new(256 << 10, 512),
            frame_pool: FramePool::default(),
            creation_pins: Default::default(),
            global_env_pins: Vec::new(),
            call_site: crate::stack_trace::NO_POSITION,
            code_frames: Vec::new(),
            job_script_contexts: Vec::new(),
            host_resolution_marker: Object::new(None),
            jit_helpers: crate::jit::helper_table(),
            jit_layout: std::cell::OnceCell::new(),
            interp_layout: std::cell::Cell::new(InterpLayout::default()),
            inline_ic_safe: std::cell::Cell::new(true),
            regexp_protector: RefCell::default(),
            regexp_last: None,
            weak_collection_data: Default::default(),
            weak_collection_index: Default::default(),
            extra_protos: Default::default(),
            array_buffers: Default::default(),
            native_buffers: Default::default(),
            array_buffer_versions: Default::default(),
            array_buffer_dirty_ranges: Default::default(),
            immutable_buffers: std::collections::HashSet::new(),
            host_keyed_buffers: std::collections::HashSet::new(),
            can_block: true,
            agent: None,
            typed_arrays: Default::default(),
            async_gens: std::collections::HashSet::new(),
            global_var_names: Rc::new(RefCell::new(std::collections::HashSet::new())),
            gc_pins: Default::default(),
            ta_buffer: Default::default(),
            shadow_realms: Default::default(),
            shadow_home: None,
            data_views: Default::default(),
            regexps: Default::default(),
            regexp_programs: crate::cache::RegexpProgramCache::new(16 << 20, 512),
            proxies: Default::default(),
            host_indexed: Default::default(),
            new_target: (Value::Undefined),
            pending_new_target: Value::Undefined,
            htmldda: Default::default(),
            ctor_caller_realm: None,
            realms: Default::default(),
            promises: Default::default(),
            unhandled_rejections: Default::default(),
            next_unhandled_rejection: 0,
            handled_rejections: Vec::new(),
            temporal: Default::default(),
            temporal_cal: Default::default(),
            host_job_context: 0,
            script_entry: None,
            host_entry_realm: None,
            capture_job_script_caller: false,
            caller_origin: None,
            host_settings_states: Default::default(),
            retired_host_job_contexts: Default::default(),
            host_job_context_enter: None,
            host_job_context_leave: None,
            host_state: Default::default(),
            generators: Default::default(),
            gc_next: live_object_limit,
            gc_tick: 0,
            constructing: false,
            super_call_ok: false,
            fn_frames: Vec::new(),
            module_async_seq: 0,
            in_field_init_code: false,
            yield_raw_result: false,
            in_async_gen_body: false,
            pending_fn_name: None,
            pending_tail: None,
            tco_ok: false,
            template_cache: Default::default(),
            using_stack: Vec::new(),
            accessor_seq: 0,
            decorator_initializers: Vec::new(),
            annexb_fn_sync: Default::default(),
            deferred_ns: Default::default(),
            deferred_ns_objs: Default::default(),
            promise_forward: Default::default(),
            mapped_arguments: Default::default(),
            borrowed_parameters: None,
            module_source_objs: Default::default(),
            weak_refs: Default::default(),
            weak_metadata: Default::default(),
            finalization_registries: Default::default(),
            async_gen_busy: std::collections::HashSet::new(),
            async_gen_queue: Default::default(),
            heap_release: if owns_heap {
                crate::value::HeapRelease::new(&gc_heap)
            } else {
                crate::value::HeapRelease::shared()
            },
        };
        crate::builtins::install(&mut interp);
        // `this` at the top level is the global object (sloppy mode).
        let g = Value::Obj(interp.global);
        interp
            .global_env
            .borrow_mut()
            .vars
            .insert("this".to_string(), Binding::data(g, false, true));
        // Register the main realm so a call back into main-realm code from inside another realm
        // swaps the main intrinsics back in (see `callee_realm_global`).
        let main = interp.snapshot_realm();
        let main_global = interp.global;
        interp.gc_pin(&main_global);
        interp
            .realms
            .insert(Gc::as_ptr(&interp.global) as usize, main);
        interp
    }

    /// Make this interpreter's Agent the surrounding Agent on the current native thread. This is
    /// required at every execution entry because generator and async bodies may resume on a pooled
    /// worker, while embedders may alternate independent engines on one driver thread.
    #[inline]
    pub(crate) fn activate_gc_heap(&self) {
        crate::value::activate_agent(&self.gc_heap, &self.symbol_agent);
    }

    /// Temporarily make this interpreter's heap and Symbol state current, restoring the
    /// surrounding Agent when the returned guard leaves scope.
    #[inline]
    pub(crate) fn enter_agent(&self) -> crate::value::AgentActivationGuard {
        crate::value::enter_agent(&self.gc_heap, &self.symbol_agent)
    }

    /// Run one host callback, then restore this interpreter's Agent before returning or
    /// unwinding. An Engine's public entry leaves its Agent active, so nested host execution
    /// must not redirect subsequent allocations. No Rc clone occurs when the Agent stayed
    /// active: restoration first compares its identities.
    #[inline]
    pub(crate) fn with_host_callback<R>(&mut self, callback: impl FnOnce(&mut Self) -> R) -> R {
        crate::agent_access::suspend_local(self, |pointer| {
            struct RestoreAgent(std::ptr::NonNull<Interp>);
            impl Drop for RestoreAgent {
                fn drop(&mut self) {
                    // SAFETY: callback descends from this same supplied raw capability, and its
                    // mutable reference ended before Drop, including on unwind. No sibling
                    // field pointer or older caller reference is resumed in this scope.
                    unsafe {
                        (&*self.0.as_ptr()).activate_gc_heap();
                    }
                }
            }
            let _restore = RestoreAgent(pointer);
            callback(unsafe { &mut *pointer.as_ptr() })
        })
    }

    /// The read-only form of `with_host_callback`, requiring no raw pointers: the shared
    /// interpreter borrow stays alive through the query and its restoration guard.
    #[inline]
    pub(crate) fn with_host_query<R>(&self, query: impl FnOnce(&Self) -> R) -> R {
        struct RestoreAgent<'a>(&'a Interp);
        impl Drop for RestoreAgent<'_> {
            fn drop(&mut self) {
                self.0.activate_gc_heap();
            }
        }
        let _restore = RestoreAgent(self);
        query(self)
    }

    // ----- error helpers ----------------------------------------------------------------------

    pub fn make_error(&self, kind: &str, message: impl Into<String>) -> Value {
        self.make_error_skipping(kind, message, crate::stack_trace::Skip::None)
    }

    /// [`Interp::make_error`] whose stack trace leaves out the frames `skip` names (an Error
    /// constructor run for a subclass leaves out the subclass constructors, as V8 does).
    pub(crate) fn make_error_skipping(
        &self,
        kind: &str,
        message: impl Into<String>,
        skip: crate::stack_trace::Skip,
    ) -> Value {
        crate::jit::perf_error_construction();
        let proto = self
            .error_protos
            .get(kind)
            .cloned()
            .unwrap_or_else(|| self.error_protos["Error"]);
        let object_started = crate::jit::perf_stage_start();
        let obj = Object::new(Some(proto));
        crate::jit::perf_error_object_end(object_started);
        let stack_started = crate::jit::perf_stage_start();
        let stack = self.capture_stack_trace(skip);
        crate::jit::perf_error_stack_capture_end(stack_started);
        obj.borrow_mut().exotic = Exotic::error(stack);
        let message_started = crate::jit::perf_stage_start();
        let msg = message.into();
        crate::jit::perf_error_message_end(message_started);
        if !msg.is_empty() {
            obj.borrow_mut()
                .props
                .insert("message", Property::builtin(Value::lstr(msg)));
        }
        let value = Value::Obj(obj);
        // Opt-in, bounded diagnostics include caught native errors without
        // invoking author accessors or changing exception propagation.
        static TRACE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        static TRACED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if let Some(filter) = TRACE.get_or_init(|| std::env::var("TRUST_JS_TRACE_ERRORS").ok()) {
            if let Some(diagnostic) = self.error_diagnostic(&value) {
                if (filter == "1" || diagnostic.contains(filter))
                    && TRACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 1024
                {
                    eprintln!("trust-js-error: {kind}: {diagnostic}");
                    MISSING_PROPERTY_TRACE.with(|trace| {
                        for entry in trace.borrow().iter() {
                            eprintln!("trust-js-error-undefined-property: {entry}");
                        }
                    });
                    BOUND_CALL_TRACE.with(|trace| {
                        for entry in &trace.borrow().history {
                            eprintln!("trust-js-error-bound-call: {entry}");
                        }
                    });
                    static SOURCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    if *SOURCE
                        .get_or_init(|| std::env::var_os("TRUST_JS_TRACE_ERROR_SOURCE").is_some())
                    {
                        static SOURCE_LIMIT: std::sync::OnceLock<usize> =
                            std::sync::OnceLock::new();
                        let limit = *SOURCE_LIMIT.get_or_init(|| {
                            std::env::var("TRUST_JS_TRACE_ERROR_SOURCE")
                                .ok()
                                .and_then(|value| value.parse::<usize>().ok())
                                .filter(|limit| *limit > 1)
                                .unwrap_or(800)
                                .min(16_384)
                        });
                        for frame in self.fn_frames.iter().rev().take(4) {
                            let callee = frame.callee();
                            let object = callee.borrow();
                            if let Callable::User(user) = &object.call {
                                if let Some(source) = &user.func.source {
                                    eprintln!(
                                        "trust-js-error-source: {:?}",
                                        source.chars().take(limit).collect::<String>()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        value
    }

    pub(crate) fn throw(&self, kind: &str, message: impl Into<String>) -> Abrupt {
        Abrupt::Throw(self.make_error(kind, message))
    }
    #[allow(dead_code)]
    pub(crate) fn type_err<T>(&self, message: impl Into<String>) -> Result<T, Abrupt> {
        Err(self.throw("TypeError", message))
    }

    // ----- typed arrays -----------------------------------------------------------------------

    /// Read element `idx` of a TypedArray as a Number (or undefined if out of range / detached).
    pub(crate) fn ta_read(&self, info: &TaInfo, idx: usize) -> Value {
        let es = info.kind.elsize();
        let decode = |buf: &[u8]| -> Value {
            if idx < info.length_for_buffer(buf.len()).unwrap_or(0) {
                let start = info.offset + idx * es;
                let bytes = &buf[start..start + es];
                if info.kind.is_bigint() {
                    Value::BigInt(info.kind.read_bigint(bytes).into())
                } else {
                    Value::Num(info.kind.read(bytes))
                }
            } else {
                Value::Undefined
            }
        };
        match self.array_buffers.get(&info.buffer) {
            // Ordinary buffers validate and decode under one borrow, with just one backing-store
            // lookup.
            Some(BufferData::Local(buf)) => decode(&buf.borrow()),
            // Keep the shared length witness and synchronized data read.
            Some(BufferData::Shared(mem)) => {
                if idx >= self.ta_len(info).unwrap_or(0) {
                    return Value::Undefined;
                }
                decode(&mem.lock().unwrap())
            }
            None => Value::Undefined,
        }
    }

    /// The raw bytes of `nelems` elements starting at element `elem_start` (None when the view
    /// is detached/out-of-bounds). Used for bitwise same-type copies that must preserve NaN bits.
    pub(crate) fn ta_read_bytes(
        &self,
        info: &TaInfo,
        elem_start: usize,
        nelems: usize,
    ) -> Option<Vec<u8>> {
        let es = info.kind.elsize();
        let start = info.offset + elem_start * es;
        let end = start + nelems * es;
        let take = |buf: &[u8]| {
            if end <= buf.len() {
                Some(buf[start..end].to_vec())
            } else {
                None
            }
        };
        self.array_buffers.get(&info.buffer)?.with_bytes(take)?
    }

    /// Write raw bytes at element `elem_start` (bounds-checked; silently drops what doesn't fit).
    pub(crate) fn ta_write_bytes(&mut self, info: &TaInfo, elem_start: usize, bytes: &[u8]) {
        let start = info.offset + elem_start * info.kind.elsize();
        let put = |buf: &mut [u8]| {
            if start + bytes.len() <= buf.len() {
                buf[start..start + bytes.len()].copy_from_slice(bytes);
            }
        };
        match self.array_buffers.get(&info.buffer) {
            Some(BufferData::Local(storage)) => {
                let mut b = storage.borrow_mut();
                if start
                    .checked_add(bytes.len())
                    .is_some_and(|end| end <= b.len())
                {
                    put(&mut b);
                    drop(b);
                    self.mark_array_buffer_dirty_range(info.buffer, start, bytes.len());
                }
            }
            // A Shared Data Block has no mutation generation or dirty ranges.
            Some(BufferData::Shared(mem)) => put(&mut mem.lock().unwrap()),
            None => {}
        }
    }

    /// Atomically read-modify-write an integer TypedArray element: `f` maps the old raw value to
    /// the new one (or `None` for no write, as in a failed compareExchange). For a shared buffer
    /// the whole operation happens under one lock hold, so concurrent agents can't interleave
    /// (a lost `Atomics.add` increment would spin another agent's waitUntil loop forever).
    /// Returns the old value, or `None` when the index is out of range.
    pub(crate) fn ta_modify(
        &mut self,
        info: &TaInfo,
        idx: usize,
        f: impl FnOnce(i128) -> Option<i128>,
    ) -> Option<i128> {
        if idx >= self.ta_len(info).unwrap_or(0) {
            return None;
        }
        let es = info.kind.elsize();
        let start = info.offset + idx * es;
        self.mark_array_buffer_dirty_range(info.buffer, start, es);
        let apply = |buf: &mut [u8]| -> Option<i128> {
            if start + es > buf.len() {
                return None;
            }
            let bytes = &mut buf[start..start + es];
            let old = if info.kind.is_bigint() {
                info.kind.read_bigint(bytes)
            } else {
                info.kind.read(bytes) as i128
            };
            if let Some(new) = f(old) {
                if info.kind.is_bigint() {
                    bytes.copy_from_slice(&(new as u64).to_le_bytes());
                } else {
                    info.kind.write_into(new as f64, bytes);
                }
            }
            Some(old)
        };
        self.array_buffers
            .get(&info.buffer)?
            .with_bytes_mut(apply)?
    }

    /// Store a JS value into a TypedArray element, coercing per the element type. BigInt arrays
    /// require a BigInt (TypeError otherwise); numeric arrays coerce with ToNumber.
    pub(crate) fn ta_store(&mut self, info: &TaInfo, idx: usize, v: &Value) -> Result<(), Abrupt> {
        // IntegerIndexedElementSet: the value coerces FIRST (observable, spec order); a write
        // into an immutable buffer then fails like a non-writable data property — a TypeError in
        // strict mode, a silent no-op in sloppy.
        let immutable =
            !self.immutable_buffers.is_empty() && self.immutable_buffers.contains(&info.buffer);
        if info.kind.is_bigint() {
            let n = self.to_bigint(v)?;
            if !immutable {
                self.ta_write_bigint(info, idx, n.to_i128_wrapping());
            }
        } else {
            let n = self.to_number(v)?;
            if !immutable {
                self.ta_write(info, idx, n);
            }
        }
        if immutable && self.strict {
            return Err(self.throw(
                "TypeError",
                "Cannot write to a TypedArray over an immutable ArrayBuffer",
            ));
        }
        Ok(())
    }

    /// Write a BigInt (i128) into element `idx` (out-of-range writes are ignored).
    pub(crate) fn ta_write_bigint(&mut self, info: &TaInfo, idx: usize, n: i128) {
        if idx >= self.ta_len(info).unwrap_or(0) {
            return;
        }
        let es = info.kind.elsize();
        let start = info.offset + idx * es;
        // Element encoding of a BigInt: wrapped modulo 2^64, little-endian.
        let bytes = (n as u64).to_le_bytes();
        let wrote = match self.array_buffers.get(&info.buffer) {
            Some(BufferData::Local(storage)) => {
                let mut buf = storage.borrow_mut();
                let fits = start + es <= buf.len();
                if fits {
                    buf[start..start + es].copy_from_slice(&bytes);
                }
                fits
            }
            // A Shared Data Block has no mutation generation or dirty ranges.
            Some(BufferData::Shared(mem)) => {
                let mut buf = mem.lock().unwrap();
                if start + es <= buf.len() {
                    buf[start..start + es].copy_from_slice(&bytes);
                }
                false
            }
            None => false,
        };
        if wrote {
            self.mark_array_buffer_dirty_range(info.buffer, start, es);
        }
    }

    /// Write Number `n` into element `idx` of a TypedArray (out-of-range writes are ignored).
    pub(crate) fn ta_write(&mut self, info: &TaInfo, idx: usize, n: f64) {
        let es = info.kind.elsize();
        let wrote = match self.array_buffers.get(&info.buffer) {
            Some(BufferData::Local(storage)) => {
                let mut buf = storage.borrow_mut();
                let in_bounds = idx < info.length_for_buffer(buf.len()).unwrap_or(0);
                if in_bounds {
                    let start = info.offset + idx * es;
                    info.kind.write_into(n, &mut buf[start..start + es]);
                }
                in_bounds
            }
            // A Shared Data Block has no mutation generation or dirty ranges.
            Some(BufferData::Shared(mem)) => {
                if idx >= self.ta_len(info).unwrap_or(0) {
                    return;
                }
                let start = info.offset + idx * es;
                let mut buf = mem.lock().unwrap();
                if start + es <= buf.len() {
                    info.kind.write_into(n, &mut buf[start..start + es]);
                }
                false
            }
            None => false,
        };
        if wrote {
            self.mark_array_buffer_dirty_range(info.buffer, info.offset + idx * es, es);
        }
    }

    /// SetTypedArrayFromArrayLike, specialized only after every source element is proven to be
    /// an own Number data property of an ordinary Array. No getters, coercions, prototype reads,
    /// or shared-memory events can occur, so copying directly is observationally equivalent to
    /// the specification's ordered Get/TypedArraySetElement loop. A failed proof writes nothing.
    pub(crate) fn ta_set_numeric_array(
        &mut self,
        info: &TaInfo,
        offset: usize,
        source: &Gc,
        len: usize,
    ) -> bool {
        if info.kind.is_bigint()
            || self.is_shared_buffer(info.buffer)
            || self.immutable_buffers.contains(&info.buffer)
            || !self.ordinary_get_ptr(Gc::as_ptr(source) as usize)
        {
            return false;
        }
        let source = source.borrow();
        if !matches!(source.exotic, Exotic::Array)
            || !(0..len).all(|index| {
                source
                    .props
                    .get_index(index as u32)
                    .is_some_and(|property| property.number_value().is_some())
            })
        {
            return false;
        }
        let Some(storage) = self.local_buffer(info.buffer) else {
            return false;
        };
        let mut buffer = storage.borrow_mut();
        let Some(length) = info.length_for_buffer(buffer.len()) else {
            return false;
        };
        if offset > length || len > length - offset {
            return false;
        }
        let size = info.kind.elsize();
        let start = info.offset + offset * size;
        for (index, bytes) in buffer[start..start + len * size]
            .chunks_exact_mut(size)
            .enumerate()
        {
            let number = source
                .props
                .get_index(index as u32)
                .and_then(Property::number_value)
                .expect("numeric array validated without running author code");
            info.kind.write_into(number, bytes);
        }
        drop(buffer);
        drop(source);
        self.mark_array_buffer_dirty_range(info.buffer, start, len * size);
        true
    }

    // ----- symbols ----------------------------------------------------------------------------

    /// Mint a fresh Symbol in the surrounding Agent. The identity lookup is weak: an ordinary
    /// Symbol is retained by actual Values and Symbol-valued property keys, never by this index.
    pub fn new_symbol(&mut self, description: Option<Rc<str>>) -> Value {
        let mut agent = self.symbol_agent.borrow_mut();
        agent.next_id = agent
            .next_id
            .checked_add(1)
            .expect("ECMAScript Agent exhausted its Symbol identity space");
        let data = crate::value::new_symbol_ref(SymbolData {
            id: agent.next_id,
            description,
            weak_observers: RefCell::new(None),
            agent_held: std::cell::Cell::new(false),
        });
        agent
            .symbols
            .insert(data.id, crate::value::symbol_downgrade(&data));
        // Symbol-only churn does not advance the object collector's allocation trigger. Prune
        // expired identity entries geometrically so a script that repeatedly calls Symbol()
        // remains bounded without making an all-live workload quadratic.
        if agent.symbols.len() >= 4096 && agent.symbols.len().is_power_of_two() {
            agent.symbols.retain(|_, symbol| symbol.strong_count() != 0);
        }
        Value::Sym(data)
    }

    /// ECMA-262 §20.4.2.2 Symbol.for, backed by the surrounding Agent Record.
    pub(crate) fn symbol_for(&mut self, key: String) -> crate::value::SymbolRef {
        if let Some(symbol) = self
            .symbol_agent
            .borrow()
            .global_by_key
            .get(key.as_str())
            .cloned()
        {
            return symbol;
        }
        let key: Rc<str> = Rc::from(key);
        let ValueView::Sym(symbol) = (self.new_symbol(Some(key.clone()))).view() else {
            unreachable!("new_symbol must return a symbol")
        };
        let mut agent = self.symbol_agent.borrow_mut();
        agent.global_key_by_id.insert(symbol.id, key.clone());
        symbol.agent_held.set(true);
        agent.global_by_key.insert(key, symbol);
        symbol
    }

    /// ECMA-262 §20.4.5.2 KeyForSymbol. The reverse Agent index makes the normative registry
    /// identity lookup O(1), with a pointer check guarding values from an unrelated Agent.
    pub(crate) fn symbol_key_for(&self, symbol: &crate::value::SymbolRef) -> Option<String> {
        let agent = self.symbol_agent.borrow();
        let key = agent.global_key_by_id.get(&symbol.id)?;
        agent
            .global_by_key
            .get(key)
            .is_some_and(|registered| crate::value::symbol_ptr_eq(registered, symbol))
            .then(|| key.to_string())
    }

    pub(crate) fn symbol_is_registered(&self, symbol: &crate::value::SymbolRef) -> bool {
        self.symbol_key_for(symbol).is_some()
    }

    /// Return one Agent-wide well-known Symbol and populate this realm implementation's compact
    /// name/key cache. ECMA-262 §6.1.5.1 requires these identities to be shared by all realms.
    pub(crate) fn well_known_symbol(&mut self, name: &'static str) -> Value {
        let existing = self.symbol_agent.borrow().well_known.get(name).cloned();
        let symbol = match existing {
            Some(symbol) => symbol,
            None => {
                let ValueView::Sym(symbol) =
                    (self.new_symbol(Some(Rc::from(format!("Symbol.{name}").as_str())))).view()
                else {
                    unreachable!("new_symbol must return a symbol")
                };
                symbol.agent_held.set(true);
                self.symbol_agent
                    .borrow_mut()
                    .well_known
                    .insert(name, symbol);
                symbol
            }
        };
        if !self.wk_syms.iter().any(|(cached, _, _)| *cached == name) {
            let value = Value::Sym(symbol);
            let key: Rc<str> = Rc::from(Self::sym_key(&symbol));
            self.wk_syms.push((name, value, key));
        }
        Value::Sym(symbol)
    }

    /// Internal Symbol key. Leading-NUL String keys are escaped by ToPropertyKey and
    /// static property-name compilation, so no author String can alias this identity.
    pub(crate) fn sym_key(data: &SymbolData) -> String {
        format!("\u{0}{}", data.id)
    }
    pub(crate) fn is_sym_key(key: &str) -> bool {
        key.strip_prefix('\0').is_some_and(|digits| {
            !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
        })
    }

    /// Classify an element receiver without exposing the current object/exotic representation to
    /// feedback consumers. This is deliberately a semantic family (ordinary, Array, TypedArray,
    /// String, primitive, or other exotic), not an engine enum or object identity.
    pub(crate) fn element_receiver_kind(
        &self,
        base: &Value,
    ) -> crate::feedback::ElementReceiverKind {
        use crate::feedback::ElementReceiverKind;
        match &(base).view() {
            ValueView::Obj(object) => {
                let ptr = Gc::as_ptr(object) as usize;
                if self.typed_arrays.contains_key(&ptr) {
                    return ElementReceiverKind::TypedArray;
                }
                if self.proxies.contains_key(&ptr)
                    || self.module_ns.contains_key(&ptr)
                    || self.deferred_ns.contains_key(&ptr)
                    || self.data_views.contains_key(&ptr)
                    || self.buffer_attached(ptr)
                    || self.host_indexed.contains_key(&ptr)
                {
                    return ElementReceiverKind::Exotic;
                }
                match &object.borrow().exotic {
                    crate::value::Exotic::Array => ElementReceiverKind::Array,
                    crate::value::Exotic::StrWrap(_) => ElementReceiverKind::String,
                    crate::value::Exotic::None => ElementReceiverKind::Ordinary,
                    crate::value::Exotic::Collection(_) => ElementReceiverKind::Ordinary,
                    _ => ElementReceiverKind::Exotic,
                }
            }
            ValueView::Str(_) => ElementReceiverKind::String,
            ValueView::Num(_) | ValueView::Bool(_) | ValueView::Sym(_) | ValueView::BigInt(_) => {
                ElementReceiverKind::Primitive
            }
            ValueView::Undefined | ValueView::Empty | ValueView::Null => {
                ElementReceiverKind::Exotic
            }
        }
    }

    /// Classify a post-`ToPropertyKey` string for indexed feedback. The canonical numeric
    /// category is kept separate from array indices because Integer-Indexed exotic objects treat
    /// canonical numeric non-indices (for example `-0` or `1.5`) as inert rather than ordinary
    /// properties (ECMA-262 §§7.1.19, 10.4.5.8-10).
    pub(crate) fn element_key_kind(&self, key: &str) -> crate::feedback::ElementKeyKind {
        if Self::is_sym_key(key) {
            crate::feedback::ElementKeyKind::Symbol
        } else if crate::value::canonical_index(key).is_some() {
            crate::feedback::ElementKeyKind::Index
        } else if self.canonical_numeric_index(key).is_some() {
            crate::feedback::ElementKeyKind::CanonicalNumeric
        } else {
            crate::feedback::ElementKeyKind::String
        }
    }

    /// Classify a call target by its observable [[Call]] family. This intentionally records no
    /// pointer or function address; identity-sensitive caches remain private execution machinery.
    pub(crate) fn call_target_kind(&self, callee: &Value) -> crate::feedback::CallTargetKind {
        use crate::feedback::CallTargetKind;
        let ValueView::Obj(object) = (callee).view() else {
            return CallTargetKind::NonCallable;
        };
        let ptr = Gc::as_ptr(&object) as usize;
        if self.proxies.contains_key(&ptr) {
            return CallTargetKind::Proxy;
        }
        let body = object.borrow();
        match &body.call {
            Callable::User(_) => CallTargetKind::UserFunction,
            native if native.is_native() => CallTargetKind::NativeFunction,
            Callable::Bound(_) => CallTargetKind::BoundFunction,
            Callable::WrappedShadow(_) | Callable::WrappedCross(_) => {
                CallTargetKind::WrappedFunction
            }
            Callable::None => CallTargetKind::NonCallable,
            _ => CallTargetKind::CallableObject,
        }
    }

    /// Classify the activation requirement of a call target. User functions use the compiled
    /// chunk's exact activation predicate when available; an uncompiled function remains
    /// `Unknown` rather than being mistaken for a no-environment fast path.
    pub(crate) fn call_environment_kind(
        &self,
        callee: &Value,
    ) -> crate::feedback::CallEnvironmentKind {
        use crate::feedback::{CallEnvironmentKind, CallTargetKind};
        match self.call_target_kind(callee) {
            CallTargetKind::NativeFunction | CallTargetKind::CallableObject => {
                CallEnvironmentKind::None
            }
            CallTargetKind::BoundFunction
            | CallTargetKind::Proxy
            | CallTargetKind::WrappedFunction => CallEnvironmentKind::Dynamic,
            CallTargetKind::UserFunction => {
                let ValueView::Obj(object) = (callee).view() else {
                    return CallEnvironmentKind::Unknown;
                };
                let Callable::User(user) = &object.borrow().call else {
                    return CallEnvironmentKind::Unknown;
                };
                match user.func.runtime().code.get() {
                    Some(Some(chunk)) if chunk.requires_activation_environment() => {
                        CallEnvironmentKind::Captured
                    }
                    Some(Some(_)) => CallEnvironmentKind::None,
                    _ => CallEnvironmentKind::Unknown,
                }
            }
            CallTargetKind::NonCallable => CallEnvironmentKind::Unknown,
        }
    }
    /// Recover the symbol `Value` behind an internal symbol key (for `getOwnPropertySymbols`).
    pub(crate) fn sym_from_key(&self, key: &str) -> Option<Value> {
        if !Self::is_sym_key(key) {
            return None;
        }
        let id: u64 = key.strip_prefix('\0')?.parse().ok()?;
        self.symbol_agent
            .borrow()
            .symbols
            .get(&id)?
            .upgrade()
            .map(Value::Sym)
    }

    // ----- object construction ----------------------------------------------------------------

    pub(crate) fn new_object(&self) -> Gc {
        Object::new(Some(self.object_proto))
    }

    /// The global object (`globalThis`) as a value — for embedders that must expose it
    /// (N-API's `napi_get_global`).
    pub fn global_this(&self) -> Value {
        Value::Obj(self.global)
    }

    /// Global of the topmost executing Script/Module/Eval or ECMAScript function,
    /// ignoring intervening native and bound-function calls. A native operation's
    /// own Realm is its *receiver* context, not necessarily its script caller.
    /// ECMA-262 PrepareForOrdinaryCall / BuiltinCallOrConstruct / ScriptEvaluation.
    ///
    /// Host-written JavaScript is still JavaScript here. Embedders implementing
    /// HTML incumbent settings must additionally handle their platform shims and
    /// the backup incumbent context of Web IDL callbacks.
    pub fn script_caller_global(&self) -> Value {
        let floor = self.script_entry.map_or(0, |(_, depth)| depth);
        let key = self
            .fn_frames
            .get(floor..)
            .and_then(|frames| frames.last())
            .and_then(|frame| match &frame.callee().borrow().call {
                Callable::User(user) => Some(user.realm),
                _ => None,
            })
            .or_else(|| self.script_entry.map(|(realm, _)| realm));
        key.and_then(|key| self.realms.get(&key))
            .map_or_else(|| self.global_this(), |realm| Value::Obj(realm.global))
    }

    /// Global of the embedder's most recent script/callback entry. HTML
    /// #entry-settings-object / #prepare-to-run-script distinguishes this from
    /// the incumbent (innermost author function) and the native receiver Realm.
    pub fn script_entry_global(&self) -> Value {
        self.host_entry_realm
            .and_then(|key| self.realms.get(&key))
            .map_or_else(|| self.global_this(), |realm| Value::Obj(realm.global))
    }

    /// Enter an author callback through a host boundary, using GetFunctionRealm
    /// (including bound functions and proxies), then restore the prior entry on
    /// every completion. Ordinary `invoke` deliberately does not enter anew.
    pub fn invoke_callback_entry(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Value> {
        let key = if let ValueView::Obj(ref object) = (callee).view() {
            self.get_function_realm_global(object)
                .map_err(abrupt_value)?
        } else {
            None
        }
        .unwrap_or(Gc::as_ptr(&self.global) as usize);
        self.with_entry_context(self.script_entry, Some(key), |interp| {
            interp.invoke(callee, this, args)
        })
    }

    pub(crate) fn with_script_entry<R>(&mut self, operation: impl FnOnce(&mut Self) -> R) -> R {
        let host = self
            .host_entry_realm
            .or(Some(Gc::as_ptr(&self.global) as usize));
        let script = Some((Gc::as_ptr(&self.global) as usize, self.fn_frames.len()));
        self.with_entry_context(script, host, operation)
    }

    /// Invoke a host callback with a previously captured incumbent global as its fallback
    /// script caller. Frames belonging to self-hosted platform plumbing below this boundary
    /// are ignored; new author function frames above it still take precedence. This does NOT
    /// switch the callable's Realm or settings, nor change its Call/throw/interrupt behavior.
    /// The host must capture the global before entering its JavaScript API implementation.
    /// HTML prepare/clean up after running a callback; Web IDL invoke a callback function.
    pub fn with_callback_script_caller<R>(
        &mut self,
        global: &Value,
        operation: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let ValueView::Obj(global) = &(global).view() else {
            return operation(self);
        };
        let key = Gc::as_ptr(global) as usize;
        if !self.realms.contains_key(&key) {
            return operation(self);
        }
        self.with_entry_context(
            Some((key, self.fn_frames.len())),
            self.host_entry_realm,
            operation,
        )
    }

    /// HostMakeJobCallback. A browser host (see [`Self::capture_job_script_caller`]) saves the
    /// incumbent script caller and, per HTML, the active script as a script execution context;
    /// other hosts keep ECMA-262's default (§9.5.2), which saves neither.
    pub(crate) fn make_job_callback(&self, callback: Value) -> JobCallback {
        // The resolving functions of a platform resolution are platform steps themselves.
        if self.is_host_resolver(&callback) {
            return JobCallback {
                internal: InternalJob::HostReaction,
                ..JobCallback::plain(callback)
            };
        }
        let (script_caller, active_script) =
            if self.capture_job_script_caller && callback.is_callable() {
                let script_caller = match (self.script_caller_global()).view() {
                    ValueView::Obj(global) => Some(global),
                    _ => None,
                };
                (script_caller, self.job_active_script(&callback))
            } else {
                (None, None)
            };
        JobCallback {
            callback,
            script_caller,
            active_script,
            internal: InternalJob::None,
        }
    }

    /// The script execution context HostMakeJobCallback saves for `callback`: the active
    /// ScriptOrModule, unless the callback is an ECMAScript function with a ScriptOrModule of
    /// its own, whose function context would always be found first.
    fn job_active_script(&self, callback: &Value) -> Option<crate::agent_control::RuntimeOrigin> {
        if let ValueView::Obj(function) = &(callback).view() {
            if let Callable::User(user) = &function.borrow().call {
                if !user.func.self_hosted
                    && user
                        .func
                        .script
                        .as_ref()
                        .is_some_and(|script| script.referrer().is_some())
                {
                    return None;
                }
            }
        }
        self.active_runtime_origin()
    }

    /// HostMakeJobCallback for one of Await's built-in reaction steps (see [`InternalJob`]).
    /// Those built-ins are callable, so the incumbent script caller is captured exactly as for
    /// an author callback.
    pub(crate) fn make_await_job_callback(
        &self,
        async_promise: Value,
        fulfilled: bool,
    ) -> JobCallback {
        let (script_caller, active_script) = if self.capture_job_script_caller {
            let script_caller = match (self.script_caller_global()).view() {
                ValueView::Obj(global) => Some(global),
                _ => None,
            };
            (script_caller, self.active_runtime_origin())
        } else {
            (None, None)
        };
        let realm = Gc::as_ptr(&self.global) as usize;
        JobCallback {
            callback: async_promise,
            script_caller,
            active_script,
            internal: if fulfilled {
                InternalJob::AwaitFulfilled { realm }
            } else {
                InternalJob::AwaitRejected { realm }
            },
        }
    }

    pub(crate) fn call_job_callback(
        &mut self,
        callback: &JobCallback,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        let entry = if self.capture_job_script_caller {
            let key = if let ValueView::Obj(ref object) = callback.callback.view() {
                self.get_function_realm_global(object)?
            } else {
                None
            }
            .unwrap_or(Gc::as_ptr(&self.global) as usize);
            Some(key)
        } else {
            self.host_entry_realm
        };
        self.with_entry_context(self.script_entry, entry, |interp| {
            interp.with_job_script_context(callback.active_script.as_ref(), |ctx| {
                if let Some(global) = &callback.script_caller {
                    ctx.with_callback_script_caller(&Value::Obj(*global), |ctx| {
                        ctx.call_callback(callback.callback, this, args)
                    })
                } else {
                    ctx.call_callback(callback.callback, this, args)
                }
            })
        })
    }

    /// ECMA-262 RunSuspendedContext: the source continuation becomes the topmost
    /// script context until it parks/completes, even when a foreign native
    /// Promise reaction or borrowed generator method resumes it. Ordinary calls
    /// made by the body still take precedence above this entry's frame floor.
    pub(crate) fn with_suspended_context<R: crate::gc_trace::GcTrace + 'static>(
        &mut self,
        global: &Gc,
        context: u64,
        operation: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let key = Gc::as_ptr(global) as usize;
        if !Gc::ptr_eq(global, &self.global) {
            let target = self
                .realms
                .get(&key)
                .expect("a live source continuation retains its owning Realm")
                .snapshot_clone();
            self.with_realm_snapshot(key, target, |interp| {
                interp.with_settings_context(context, |interp| interp.with_script_entry(operation))
            })
        } else {
            self.with_settings_context(context, |interp| interp.with_script_entry(operation))
        }
    }

    /// [[Get]] for embedders: like [`Self::get_member`] but surfaces a thrown value rather than
    /// the crate-internal `Abrupt` completion, so op crates can propagate it as `Err(Value)`.
    pub fn member_get(&mut self, base: &Value, key: &str) -> Result<Value, Value> {
        self.get_member_key(base, &crate::value::encode_string_key(key))
            .map_err(|a| match a {
                Abrupt::Throw(v) => v,
                _ => self.make_error(
                    "Error",
                    "unexpected non-throw completion during property get",
                ),
            })
    }

    /// [[Set]] for embedders (see [`Self::member_get`]).
    pub fn member_set(&mut self, base: &Value, key: &str, value: Value) -> Result<(), Value> {
        self.set_member_key(base, &crate::value::encode_string_key(key), value)
            .map_err(|a| match a {
                Abrupt::Throw(v) => v,
                _ => self.make_error(
                    "Error",
                    "unexpected non-throw completion during property set",
                ),
            })
    }

    // ----- embed helpers for host addons (N-API) ---------------------------------------------

    /// A stable identity for an object value (its heap address), or `None` for a non-object.
    /// Used by host addons to key per-object native state (`napi_wrap`) and to implement `===`
    /// on objects without exposing the object handle.
    pub fn object_addr(&self, v: &Value) -> Option<usize> {
        v.as_obj().map(|o| Gc::as_ptr(&o) as usize)
    }

    /// Downgrade an object Value (tests watching it die).
    #[cfg(test)]
    pub(crate) fn downgrade_object_value(&self, v: &Value) -> Option<WeakValue> {
        v.as_obj().map(|object| WeakValue(Gc::downgrade(&object)))
    }

    /// Whether `v` has Proxy exotic behavior. Hosts need this for Node's
    /// `util.types.isProxy()`; exposing the predicate avoids leaking proxy targets or handlers.
    pub fn is_proxy_value(&self, v: &Value) -> bool {
        self.object_addr(v)
            .is_some_and(|ptr| self.proxies.contains_key(&ptr))
    }

    /// Run a full (major) collection and return the number of reclaimed heap objects.
    /// Inlined: the stack below the host's frame is zeroed before the collection's frames are
    /// built (see `gc_collect`).
    #[inline(always)]
    pub fn collect_garbage_for_host(&mut self) -> i64 {
        #[cfg(not(target_arch = "wasm32"))]
        crate::gc_core::scrub_stack(crate::gc_core::ENTRY_SCRUB_BYTES);
        self.activate_gc_heap();
        let before = crate::value::heap_live_objects(&self.gc_heap);
        let outcome = self.gc_collect_with_cause(
            crate::value::GcCause::Explicit,
            Some(crate::gc_core::CollectionKind::Major),
        );
        let live = crate::value::heap_live_objects(&self.gc_heap);
        if outcome
            == crate::agent_control::CollectionOutcome::Collected(
                crate::gc_core::CollectionKind::Major,
            )
        {
            self.set_agent_next(self.gc_heap.agent.limit.get());
            self.gc_heap
                .agent
                .task_allocated
                .set(crate::value::heap_allocated_objects(&self.gc_heap));
        }
        before.saturating_sub(live)
    }

    /// When the memory reducer's timer should next run (`Engine::memory_reducer_deadline`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn memory_reducer_deadline(&self) -> Option<std::time::Instant> {
        self.gc_heap
            .core
            .try_borrow()
            .ok()
            .and_then(|core| core.memory_reducer_deadline())
    }

    /// Run the memory reducer's timer at `now` (`Engine::run_memory_reducer`): a collection can
    /// start outside JS execution and GC suppression only. Returns whether it collected.
    #[cfg(not(target_arch = "wasm32"))]
    #[inline(always)]
    pub fn run_memory_reducer(&mut self, now: std::time::Instant) -> bool {
        let agent = &self.gc_heap.agent;
        let can_start = agent.suppressed.get() == 0 && agent.execution_depth.get() == 0;
        let start = self
            .gc_heap
            .core
            .try_borrow_mut()
            .is_ok_and(|mut core| core.memory_reducer_timer(now, can_start));
        if start {
            self.collect_garbage_for_host();
        }
        start
    }

    /// `Engine::notify_possible_garbage`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn notify_possible_garbage(&mut self) {
        if let Ok(mut core) = self.gc_heap.core.try_borrow_mut() {
            core.notify_possible_garbage(std::time::Instant::now());
        }
    }

    /// Run an explicit minor collection (a major one when a major request is pending) and
    /// return the number of reclaimed heap objects: the test hook behind
    /// `Ctx::unstable_collect_young_for_host_tests`.
    #[inline(always)]
    pub(crate) fn unstable_collect_young_for_host_tests(&mut self) -> i64 {
        #[cfg(not(target_arch = "wasm32"))]
        crate::gc_core::scrub_stack(crate::gc_core::ENTRY_SCRUB_BYTES);
        self.activate_gc_heap();
        let before = crate::value::heap_live_objects(&self.gc_heap);
        self.gc_collect_with_cause(
            crate::value::GcCause::Explicit,
            Some(crate::gc_core::CollectionKind::Minor),
        );
        before.saturating_sub(crate::value::heap_live_objects(&self.gc_heap))
    }

    /// Create a pending intrinsic Promise and its resolving functions for a host operation.
    ///
    /// This is the engine-facing form of ECMA-262 `CreateResolvingFunctions`: `resolve` and
    /// `reject` share one `[[AlreadyResolved]]` record, so only the first call can settle the
    /// promise. Hosts retain these values on the engine thread, call one when an external task
    /// completes, and then perform their normal microtask checkpoint.
    pub fn new_promise_with_resolvers(&mut self) -> (Value, Value, Value) {
        let promise = self.new_promise();
        let (resolve, reject) = self.make_resolver_pair(&promise);
        (promise, resolve, reject)
    }

    /// Current number of heap objects tracked by the engine.
    pub fn live_object_count(&self) -> i64 {
        crate::value::heap_live_objects(&self.gc_heap)
    }

    /// Current UTC epoch time for this realm, with the legacy process callback and system clock
    /// as fallbacks. Consumers apply their own precision rules (`Date` truncates to milliseconds).
    pub(crate) fn wall_now_ms(&self) -> f64 {
        if let Some(clock) = &self.wall_clock {
            return self.with_host_query(|_| clock());
        }
        let host_now = self.with_host_query(|_| crate::host_now_ms());
        if let Some(ms) = host_now {
            return ms;
        }
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    /// Poll the shared host control at a bounded hot-loop cadence.
    #[inline]
    pub(crate) fn interrupt_poll(&mut self) -> Result<(), Abrupt> {
        self.interrupt_poll_tick = self.interrupt_poll_tick.wrapping_add(1);
        if self.interrupt_poll_tick & 0x3fff != 0 {
            return Ok(());
        }
        self.interrupt_poll_force()
    }

    /// Poll without the cadence divider (script/callback entry and JIT slow poll).
    pub(crate) fn interrupt_poll_force(&self) -> Result<(), Abrupt> {
        match self.runtime_interrupt.current_reason() {
            Some(reason) => Err(Abrupt::Interrupt(reason)),
            None => Ok(()),
        }
    }

    /// Execute all currently queued promise reaction jobs synchronously.
    pub fn drain_microtasks_for_host(&mut self) {
        self.drain_promise_jobs_for_host();
    }

    /// The `[[Prototype]]` slot of `ToObject(v)` without running author code: an object's stored
    /// prototype (a Proxy's own slot; its `getPrototypeOf` trap is not called), the current
    /// Realm's wrapper prototype for a primitive (`%Number.prototype%` for a Number, …), and
    /// `null` for `null`/`undefined` or an object without a prototype. Use
    /// [`Self::get_prototype_of`] for the full `Object.getPrototypeOf` semantics.
    pub fn prototype_of(&self, v: &Value) -> Value {
        match &(v).view() {
            ValueView::Obj(o) => o.borrow().proto.map_or(Value::Null, Value::Obj),
            ValueView::Str(_) => Value::Obj(self.string_proto),
            ValueView::Num(_) => Value::Obj(self.number_proto),
            ValueView::Bool(_) => Value::Obj(self.boolean_proto),
            ValueView::Sym(_) => Value::Obj(self.symbol_proto),
            ValueView::BigInt(_) => self
                .extra_protos
                .get("BigInt")
                .cloned()
                .map_or(Value::Null, Value::Obj),
            _ => Value::Null,
        }
    }

    /// `Object.getPrototypeOf(v)` (ECMA-262 #sec-object.getprototypeof): ToObject (a TypeError
    /// for `null`/`undefined`), then `[[GetPrototypeOf]]`, which runs a Proxy's trap and its
    /// invariant checks. `Err` carries the thrown value.
    pub fn get_prototype_of(&mut self, v: &Value) -> Result<Value, Value> {
        {
            let __value_match_original = v;
            match &(__value_match_original).view() {
                ValueView::Obj(_) => crate::builtins::js_get_prototype_of(self, v),
                ValueView::Undefined | ValueView::Empty | ValueView::Null => {
                    Err(self.make_error("TypeError", "Cannot convert undefined or null to object"))
                }
                _ => {
                    let primitive = __value_match_original;
                    Ok(self.prototype_of(primitive))
                }
            }
        }
    }

    /// Whether `v` is an Error object (has the error exotic on itself or anywhere up its
    /// prototype chain — so subclass instances count too).
    pub fn is_error_value(&self, v: &Value) -> bool {
        let mut cur = v.as_obj();
        while let Some(o) = cur {
            if matches!(o.borrow().exotic, Exotic::Error(_)) {
                return true;
            }
            cur = o.borrow().proto;
        }
        false
    }

    /// A bounded, side-effect-free diagnostic for an actual Error exotic.
    ///
    /// Unlike Error.prototype.toString or the JS `stack` accessor, this never
    /// performs [[Get]], follows a prototype, invokes a proxy, or coerces a
    /// value. Only an own string data message and the engine-captured stack are
    /// inspected. Non-Errors (including proxies around Errors) return None.
    pub fn error_diagnostic(&self, value: &Value) -> Option<String> {
        let handle = value.as_obj()?;
        let object = handle.try_borrow().ok()?;
        let Exotic::Error(trace) = &object.exotic else {
            return None;
        };
        let message = match object.props.get("message") {
            Some(property) if !property.accessor() => match (property.value()).view() {
                ValueView::Str(message) => message.text().chars().take(512).collect::<String>(),
                _ => "<non-string message>".to_owned(),
            },
            Some(_) => "<accessor message>".to_owned(),
            None => "<no own message>".to_owned(),
        };
        let stack: String = trace
            .as_deref()
            .map(crate::stack_trace::StackTrace::formatted)
            .unwrap_or_default()
            .chars()
            .take(2048)
            .collect();
        Some(format!("Error: {message}{stack}"))
    }

    /// The `===` (SameValueNonNumeric / strict-equality) predicate, for host addons.
    pub fn values_strict_equal(&self, a: &Value, b: &Value) -> bool {
        self.strict_equals(a, b)
    }

    /// Whether the current native call is a construct (`new`) — a native constructor uses this to
    /// build its instance only under `new`.
    pub fn is_constructing(&self) -> bool {
        self.constructing
    }

    /// The active native constructor's NewTarget, or Undefined for an ordinary
    /// call. ECMA-262 BuiltinCallOrConstruct passes this independently of `this`.
    /// Returning an owned handle preserves it across reentrant conversions/GC.
    pub fn constructor_new_target(&self) -> Value {
        self.new_target
    }

    /// GetFunctionRealm for host bindings which implement constructor prototype
    /// fallback (for example Web IDL's interface object creation). This follows
    /// bound/proxy targets, checks revocation, and does not read author properties.
    pub fn callable_realm_global(&mut self, function: &Value) -> Result<Value, Value> {
        let Some(object) = function.as_obj().filter(|_| function.is_callable()) else {
            return Err(self.make_error("TypeError", "value is not callable"));
        };
        let key = self
            .get_function_realm_global(&object)
            .map_err(|error| match error {
                Abrupt::Throw(value) => value,
                _ => self.make_error("Error", "unexpected GetFunctionRealm completion"),
            })?;
        Ok(key
            .and_then(|key| self.realms.get(&key))
            .map_or_else(|| self.global_this(), |realm| Value::Obj(realm.global)))
    }

    /// A fresh object with an explicit `[[Prototype]]` (`null` → no prototype). For host addons
    /// creating instances with a class's prototype.
    pub fn new_object_with_proto(&self, proto: &Value) -> Value {
        Value::Obj(Object::new(proto.as_obj()))
    }

    /// Install a contiguous set of lazy, read-only indexed properties on a host-created object.
    ///
    /// The resulting object implements the indexed portions of Web IDL's legacy platform-object
    /// internal methods. `getter` is called as `getter.call(target, index)` only when a supported
    /// property is observed. The target must be a fresh ordinary, extensible, non-callable object
    /// with no existing array-index properties; named expandos remain ordinary properties.
    pub fn install_readonly_indexed_properties(
        &mut self,
        target: &Value,
        length: u32,
        getter: Value,
    ) -> Result<(), Value> {
        let Some(object) = target.as_obj() else {
            return Err(self.make_error("TypeError", "indexed-property target must be an object"));
        };
        if !getter.is_callable() {
            return Err(self.make_error("TypeError", "indexed-property getter must be callable"));
        }
        let ptr = Gc::as_ptr(&object) as usize;
        if self.host_indexed.contains_key(&ptr) {
            return Err(self.make_error(
                "TypeError",
                "indexed properties are already installed on this object",
            ));
        }
        {
            let borrowed = object.borrow();
            let ordinary = borrowed.exotic.is_ordinary()
                && matches!(borrowed.call, Callable::None)
                && borrowed.extensible
                && self.ordinary_get_ptr(ptr);
            let has_index = borrowed
                .props
                .keys()
                .iter()
                .any(|key| key.as_index().is_some());
            if !ordinary || has_index {
                return Err(self.make_error(
                    "TypeError",
                    "indexed-property target must be a fresh ordinary extensible object",
                ));
            }
        }
        self.gc_pin(&object);
        object.borrow().ic_plain.set(false);
        self.host_indexed.insert(
            &self.gc_heap,
            ptr,
            HostIndexedProperties {
                length,
                getter,
                live: None,
            },
        );
        Ok(())
    }

    /// Install a live Web IDL collection. Membership comes directly from a
    /// native view, while values use the ordinary callable/realm/GC boundary.
    /// Optional named properties are read-only and unenumerable, with ordinary
    /// prototype/expando precedence (no LegacyOverrideBuiltIns).
    ///
    /// The membership functions must perform native reads only: no JavaScript
    /// calls, JS allocations, collection, or observable state changes.
    pub fn install_live_readonly_indexed_properties(
        &mut self,
        target: &Value,
        state: Value,
        length: fn(&Interp, &Value) -> u32,
        getter: Value,
        named: Option<(fn(&Interp, &Value) -> Vec<String>, Value)>,
    ) -> Result<(), Value> {
        if named
            .as_ref()
            .is_some_and(|(_, getter)| !getter.is_callable())
        {
            return Err(self.make_error("TypeError", "named-property getter must be callable"));
        }
        self.install_readonly_indexed_properties(target, 0, getter)?;
        let (names, named_getter) = match named {
            Some((names, getter)) => (Some(names), Some(getter)),
            None => (None, None),
        };
        self.host_indexed
            .get_mut(
                &self.gc_heap,
                &(Gc::as_ptr(&target.as_obj().unwrap()) as usize),
            )
            .unwrap()
            .live = Some(Rc::new(HostIndexedLive {
            state,
            length,
            names,
            named_getter,
        }));
        Ok(())
    }

    /// Mark a native function as a constructor and wire up its `prototype`/`constructor` link, so
    /// `new ctor()` is legal and instances inherit from `proto`. Both links are non-enumerable,
    /// as for a JS `class`.
    pub fn set_constructor_prototype(&self, ctor: &Value, proto: &Value) {
        if let Some(c) = ctor.as_obj() {
            c.borrow_mut().is_constructor = true;
            c.borrow_mut()
                .props
                .insert("prototype", Property::data(*proto, false, false, false));
        }
        if let Some(p) = proto.as_obj() {
            p.borrow_mut()
                .props
                .insert("constructor", Property::data(*ctor, true, false, true));
        }
    }

    /// Define an accessor (getter/setter) property on `target`. For host addons registering class
    /// accessors (`napi_define_class` with getter/setter descriptors).
    pub fn define_accessor_value(
        &self,
        target: &Value,
        name: &str,
        get: Option<Value>,
        set: Option<Value>,
        enumerable: bool,
    ) {
        if let Some(o) = target.as_obj() {
            o.borrow_mut().props.insert(
                crate::value::encode_string_key(name).as_ref(),
                Property::accessor_prop(get, set, enumerable, true),
            );
        }
    }

    /// A TypedArray's `(napi element-type code, byte length, data pointer)`, or `None` for a
    /// non-typed-array. The pointer aliases the live backing store; it is valid until the buffer
    /// is resized or detached. For `napi_get_typedarray_info`.
    pub fn typed_array_raw(&mut self, v: &Value) -> Option<(u8, usize, *mut u8)> {
        let obj = v.as_obj()?;
        let info = self
            .typed_arrays
            .get(&(Gc::as_ptr(&obj) as usize))
            .copied()?;
        let len = self.ta_len(&info)?;
        let (code, elem) = match info.kind {
            crate::value::TaKind::I8 => (0u8, 1usize),
            crate::value::TaKind::U8 => (1, 1),
            crate::value::TaKind::U8Clamped => (2, 1),
            crate::value::TaKind::I16 => (3, 2),
            crate::value::TaKind::U16 => (4, 2),
            crate::value::TaKind::I32 => (5, 4),
            crate::value::TaKind::U32 => (6, 4),
            crate::value::TaKind::F32 => (7, 4),
            crate::value::TaKind::F64 => (8, 8),
            crate::value::TaKind::I64 => (9, 8),
            crate::value::TaKind::U64 => (10, 8),
            crate::value::TaKind::F16 => (1, 2), // no N-API code for float16; report bytes
        };
        let byte_len = len * elem;
        let ptr = match self.array_buffers.get(&info.buffer)? {
            BufferData::Local(storage) => {
                let ptr = storage.borrow_mut().as_mut_ptr();
                // N-API receives a live mutable pointer and may write through it after this call
                // returns.
                self.mark_array_buffer_dirty(info.buffer);
                ptr
            }
            // The backing buffer owns the Shared Data Block; only a grow moves its bytes.
            BufferData::Shared(block) => block.lock().unwrap().as_mut_ptr(),
        };
        // SAFETY: `ta_len` placed the view's `byte_len` bytes from `info.offset` inside the
        // block, so the offset stays within (or one past the end of) its allocation.
        Some((code, byte_len, unsafe { ptr.add(info.offset) }))
    }

    pub fn make_array(&self, items: Vec<Value>) -> Value {
        if items.len() <= 32 || crate::value::dense_elements_enabled() {
            return Value::Obj(Object::new_with_parts(
                Some(self.array_proto),
                Props::packed_array(items.into_iter()),
                Exotic::Array,
            ));
        }
        let obj = Object::new(Some(self.array_proto));
        let len = items.len();
        let numeric = items
            .iter()
            .all(|v| matches!(&(v).view(), ValueView::Num(_)));
        {
            let mut b = obj.borrow_mut();
            b.props.mark_array();
            b.props.reserve_dense_exact(len, numeric);
        }
        obj.borrow_mut().exotic = Exotic::Array;
        {
            let mut b = obj.borrow_mut();
            for v in items {
                b.props.push_dense(Property::plain(v));
            }
            b.props.insert(
                "length",
                Property::data(Value::Num(len as f64), true, false, false),
            );
        }
        Value::Obj(obj)
    }

    /// CreateArrayFromList for an entry pair: copy the canonical word straight into
    /// array storage without an intermediate Vec.
    pub(crate) fn make_entry_pair(&self, key: crate::lstr::LStr, value: Value) -> Value {
        Value::Obj(Object::new_with_parts(
            Some(self.array_proto),
            Props::packed_array([(Value::Str(key)), value].into_iter()),
            Exotic::Array,
        ))
    }

    /// The String a (String) property key names, as enumeration produces it. A heap atom is
    /// that String itself; a static name (no heap cell), an index or an escaped key
    /// ([`crate::value::encode_string_key`]) gets a heap String, cached by key. Live property
    /// attributes and values are always read by the enumeration caller.
    /// A property key, as engine key text, as the value author code receives (a Proxy trap's
    /// key argument): its Symbol, or a String, which for a key with an atom is that atom's own
    /// cell or a static name's cached String (see [`Interp::property_key_string`]) rather than a
    /// fresh copy. A String's identity is unobservable, and a trap that reads the key back finds
    /// its atom at once.
    pub(crate) fn key_value(&mut self, key: &str) -> Value {
        if let Some(symbol) = self.sym_from_key(key) {
            return symbol;
        }
        match crate::atom::PropKey::lookup_text(key) {
            Some(key) => Value::Str(self.property_key_string(key)),
            None => crate::value::string_key_value(key),
        }
    }

    pub(crate) fn property_key_string(&mut self, key: crate::atom::PropKey) -> crate::lstr::LStr {
        use crate::lstr::Units;
        if let Some(atom) = key.as_atom() {
            let escaped = match key.units() {
                Some(Units::Ascii(units)) => units.starts_with(&[0, 0]),
                Some(Units::Utf16(units)) => units.starts_with(&[0, 0]),
                None => false,
            };
            if !escaped && !atom.is_static() {
                return atom.as_lstr();
            }
        }
        if let Some(string) = self
            .property_key_strings
            .get_mapped(&key.word(), |string| *string)
        {
            return string;
        }
        let string = crate::lstr::LStr::from(crate::value::decode_string_key(&key.text()));
        let bytes = string.retained_requested_bytes();
        self.property_key_strings.insert(key.word(), string, bytes);
        string
    }

    /// Copy a bounded ordinary Array containing only own numeric data properties.
    /// Returns None without executing author code for proxies, holes, accessors, or
    /// non-numbers; callers must then use normal property reads and conversions.
    /// ECMA-262 OrdinaryGet returns these own data values before consulting prototypes.
    pub fn copy_numeric_array(&self, value: &Value, limit: usize) -> Option<Vec<f64>> {
        let object = value.as_obj()?;
        if !self.ordinary_get_ptr(Gc::as_ptr(&object) as usize) {
            return None;
        }
        let object = object.borrow();
        if !matches!(object.exotic, Exotic::Array) {
            return None;
        }
        let length = object.props.get("length")?;
        if length.accessor() {
            return None;
        }
        let length = length.value().as_num_opt()?;
        if length > limit as f64 {
            return None;
        }
        // Reserve only what a bounded prefix needs: a large sparse `length` must not allocate
        // before its first hole ends the copy.
        let mut values = Vec::with_capacity((length as usize).min(4096));
        for index in 0..length as u32 {
            let property = object.props.get_index(index)?;
            if property.accessor() {
                return None;
            }
            values.push(property.value().as_num_opt()?);
        }
        Some(values)
    }

    /// Intrinsic WeakMap.prototype.get for host-owned private-slot tables. Uses
    /// the same brand check and ephemeron storage as JS, without an overridable
    /// property lookup or a temporary Reflect.apply argument array.
    pub fn weak_map_get(&mut self, map: &Value, key: &Value) -> Result<Value, Value> {
        crate::builtins::weak_map_get(self, map, key)
    }

    /// Intrinsic Map lookup for private host state; does not invoke overridden methods.
    pub fn map_get(&mut self, map: &Value, key: &Value) -> Result<Value, Value> {
        crate::builtins::map_get(self, map, key)
    }

    /// Intrinsic Map assignment for private host state, with normal SameValueZero key semantics.
    pub fn map_set(&mut self, map: &Value, key: Value, value: Value) -> Result<(), Value> {
        crate::builtins::map_set(self, map, key, value)
    }

    /// Build an array by copying `len` canonical words from an immutable evaluated window.
    ///
    /// # Safety
    /// `items` points to `len` initialized execution words in stable immutable storage retained
    /// through this call. The caller retires its window only after the copy completes.
    pub(crate) unsafe fn make_array_from_raw(&self, items: *const Value, len: usize) -> Value {
        if len <= 32 || crate::value::dense_elements_enabled() {
            let obj = Object::new_with_parts(
                Some(self.array_proto),
                unsafe { Props::packed_array_from_raw(items, len) },
                Exotic::Array,
            );
            return Value::Obj(obj);
        }
        let obj = Object::new(Some(self.array_proto));
        let numeric =
            (0..len).all(|k| matches!((unsafe { *items.add(k) }).view(), ValueView::Num(_)));
        {
            let mut b = obj.borrow_mut();
            b.props.mark_array();
            b.props.reserve_dense_exact(len, numeric);
            b.exotic = Exotic::Array;
            for k in 0..len {
                b.props
                    .push_dense(Property::plain(unsafe { items.add(k).read() }));
            }
            b.props.insert(
                "length",
                Property::data(Value::Num(len as f64), true, false, false),
            );
        }
        Value::Obj(obj)
    }

    /// Build the generator/iterator object whose `next`/`return`/`throw` drive its coroutine (stored
    /// separately in `self.generators`).
    fn make_generator(&mut self, is_async: bool, gen_proto: Option<Value>) -> Value {
        // GetPrototypeFromConstructor: the instance's [[Prototype]] is the generator function's
        // `.prototype` (which chains to %GeneratorPrototype% / %AsyncGeneratorPrototype%, where
        // next/return/throw and @@asyncIterator live), else the intrinsic directly.
        let intrinsic = if is_async {
            "%AsyncGeneratorPrototype%"
        } else {
            "%GeneratorPrototype%"
        };
        let proto = match (gen_proto).map(|__value| (__value).view()) {
            Some(ValueView::Obj(o)) => Some(o),
            _ => self.extra_protos.get(intrinsic).cloned(),
        }
        .or(Some(self.object_proto));
        Value::Obj(Object::new(proto))
    }

    /// Invoke a native callable (bare `fn` or data-carrying closure) — the shared body for the
    /// call/construct/super dispatch paths.
    pub(crate) fn dispatch_native(
        &mut self,
        call: &Callable,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        self.interrupt_poll_force()?;
        let perf_started = crate::jit::perf_stage_start();
        // I1: the arguments often live in the caller's Rust-heap buffer.
        let _arguments = crate::rooting::root_slice(args);
        let result = self.with_host_callback(|interp| {
            match call {
                Callable::Native(f) => f(interp, this, args),
                #[cfg(feature = "embed")]
                Callable::ScopedNative(f) => crate::scope::invoke_native(*f, interp, this, args),
                // The call's own handle: the function object, and with it the payload cell that
                // `call` names, may become unreachable while the function runs.
                Callable::NativeData(data) => match &Rc::clone(data).body {
                    crate::value::NativeCallableBody::Opaque(f) => {
                        let f = f.clone();
                        f(interp, this, args)
                    }
                    crate::value::NativeCallableBody::Captured { function, captures } => {
                        // An active invocation keeps its captures alive (I1): its function object
                        // may become unreachable while it runs.
                        let _captures = crate::rooting::root_slice(captures);
                        function(interp, this, args, captures)
                    }
                    #[cfg(feature = "embed")]
                    crate::value::NativeCallableBody::ScopedCaptured { function, captures } => {
                        let _captures = crate::rooting::root_slice(captures);
                        crate::scope::invoke_captured_native(
                            *function, interp, this, args, captures,
                        )
                    }
                    crate::value::NativeCallableBody::CapturedClosure { function, captures } => {
                        // As above; the closure lives while this call holds it.
                        let _captures = crate::rooting::root_slice(captures);
                        let function = function.clone();
                        function(interp, this, args, captures)
                    }
                },
                _ => unreachable!("dispatch_native on a non-native callable"),
            }
        });
        crate::jit::perf_native_end(
            perf_started,
            result.is_ok(),
            crate::jit::NativeLabelSrc::Call(call),
        );
        // A native may run nested JavaScript (for example `$262.evalScript`). Host interruption
        // from that execution must override its Value-shaped native error before author `catch`.
        self.interrupt_poll_force()?;
        result.map_err(Abrupt::Throw)
    }

    /// A function `Value` backed by a data-carrying native closure — the embedder API for host
    /// functions that must capture state (N-API callbacks carrying a C fn pointer + `void*`).
    /// `captures` declares the closure's JavaScript values (none, or a reporter for them, which
    /// also takes part in managed-memory diagnostics and may share the closure's state).
    pub fn new_native_fn(
        &self,
        name: &str,
        len: usize,
        f: std::rc::Rc<crate::value::NativeClosure>,
        captures: crate::value::NativeCaptures,
    ) -> Value {
        Value::Obj(self.make_native_closure(name, len, f, captures))
    }

    /// A native function whose JavaScript captures are owned and traced by the
    /// engine, including cycles through the function itself. The operation is a
    /// function pointer, so it cannot hide additional JavaScript handles in a
    /// Rust closure. Use `op_state` for independent native host state.
    ///
    /// Captures are immutable as a list; captured objects retain normal mutable
    /// JavaScript behavior. A live function or an active invocation keeps its captures
    /// alive; a Rust callable lease alone does not: hold the function. Unreachable
    /// captured cycles can be reclaimed.
    /// This does not change the ordinary function's receiver or argument list.
    pub fn new_native_fn_with_captures(
        &self,
        name: &str,
        len: usize,
        function: crate::value::NativeCaptureFn,
        captures: Vec<Value>,
    ) -> Value {
        Value::Obj(self.make_native_data_callable(
            name,
            len,
            crate::value::NativeCallableBody::Captured {
                function,
                captures: captures.into_boxed_slice(),
            },
            None,
        ))
    }

    /// A function object backed by a data-carrying native closure (see [`NativeClosure`]). Like
    /// [`make_native`], but the callable can capture host state (used for N-API functions);
    /// `captures` declares its JavaScript values (see [`Self::new_native_fn`]).
    pub fn make_native_closure(
        &self,
        name: &str,
        len: usize,
        f: std::rc::Rc<crate::value::NativeClosure>,
        captures: crate::value::NativeCaptures,
    ) -> Gc {
        self.make_native_data_callable(
            name,
            len,
            crate::value::NativeCallableBody::Opaque(f),
            captures.into_reporter(),
        )
    }

    /// A function object backed by a native closure that carries Rust state and JavaScript
    /// captures both (S6 step 11b): the captures are owned and traced by the engine, as
    /// [`Self::new_native_fn_with_captures`]'s, and passed to the closure as its last argument.
    pub(crate) fn make_native_capture_closure(
        &self,
        name: &str,
        len: usize,
        function: Rc<crate::value::CaptureClosure>,
        captures: Vec<Value>,
        retained: crate::value::NativeCaptures,
    ) -> Gc {
        self.make_native_data_callable(
            name,
            len,
            crate::value::NativeCallableBody::CapturedClosure {
                function,
                captures: captures.into_boxed_slice(),
            },
            retained.into_reporter(),
        )
    }

    fn make_native_data_callable(
        &self,
        name: &str,
        len: usize,
        body: crate::value::NativeCallableBody,
        retained: Option<std::rc::Rc<dyn crate::value::NativeCallableRetained>>,
    ) -> Gc {
        self.make_callable_object(
            name,
            len,
            Callable::native_data(Rc::new(crate::value::NativeCallable {
                body,
                retained,
                identity: Rc::from(name),
            })),
        )
    }

    /// Install the common ordinary-function properties without adapting the callback type.
    fn make_callable_object(&self, name: &str, len: usize, call: Callable) -> Gc {
        let obj = Object::new(Some(self.function_proto));
        {
            let mut b = obj.borrow_mut();
            b.call = call;
            b.props.insert(
                "length",
                Property::data(Value::Num(len as f64), false, false, true),
            );
            b.props.insert(
                "name",
                Property::data(Value::lstr(name.to_string()), false, false, true),
            );
        }
        self.register_native_realm(&obj);
        obj
    }

    pub(crate) fn make_native(&self, name: &str, len: usize, f: NativeFn) -> Gc {
        crate::jit::perf_native_register(f as usize, name);
        self.make_callable_object(name, len, Callable::Native(f))
    }

    #[cfg(feature = "embed")]
    pub(crate) fn make_scoped_native(
        &self,
        name: &str,
        len: usize,
        f: crate::scope::NativeFn,
    ) -> Gc {
        crate::jit::perf_native_register(f as usize, name);
        self.make_callable_object(name, len, Callable::ScopedNative(f))
    }

    #[cfg(feature = "embed")]
    pub(crate) fn new_scoped_native_with_captures(
        &self,
        name: &str,
        len: usize,
        function: crate::scope::NativeCaptureFn,
        captures: Vec<Value>,
    ) -> Value {
        Value::Obj(self.make_native_data_callable(
            name,
            len,
            crate::value::NativeCallableBody::ScopedCaptured {
                function,
                captures: captures.into_boxed_slice(),
            },
            None,
        ))
    }

    pub(crate) fn register_native_realm(&self, function: &Gc) {
        self.remember_metadata_owner(Gc::as_ptr(function) as usize);
        self.native_function_realms.borrow_mut().register(
            &self.gc_heap,
            function,
            Gc::as_ptr(&self.global) as usize,
        );
    }

    /// Same-realm native calls need no context switch, even in a multi-Window Agent.
    /// Never infer [[Realm]] from the mutable prototype chain. Indirect eval retains
    /// its special dispatch, and unregistered native sentinels take the generic path.
    pub(crate) fn native_call_in_current_realm(&self, function: &Gc) -> bool {
        let key = Gc::as_ptr(function) as usize;
        if self.eval_realm_fns.contains(&key) {
            return false;
        }
        self.native_function_realms
            .borrow()
            .get(&key)
            .is_some_and(|(_, realm)| *realm == Gc::as_ptr(&self.global) as usize)
    }

    /// Create a minimal callable `[[IsHTMLDDA]]` exotic (ECMA-262 Annex B.3.6): falsy, `typeof`
    /// "undefined" and loosely equal to `null`/`undefined`, with a [[Call]] that returns `null`, as
    /// test262's `$262.IsHTMLDDA` specifies (INTERPRETING.md). A browser's `document.all` is the
    /// complete [`Self::make_html_all_collection`]. The engine mints both because these are
    /// language-level exceptions with dedicated interpreter and JIT handling.
    pub fn make_html_dda(&mut self) -> Value {
        let dda = self.make_native("IsHTMLDDA", 0, |_i, _this, _args| Ok(Value::Null));
        self.htmldda
            .insert(Gc::as_ptr(&dda) as usize, Gc::downgrade(&dda));
        // Every JIT object fast path assumes an ordinary object is truthy and not loosely equal to
        // null. Force this exotic through the checked helpers that implement Annex B.3.6.
        dda.borrow().ic_plain.set(false);
        Value::Obj(dda)
    }

    /// The web platform's `HTMLAllCollection` (HTML #the-htmlallcollection-interface): the
    /// `[[IsHTMLDDA]]` object (Annex B.3.6) that a Document's `all` attribute returns, as a
    /// complete legacy platform object rather than a bare falsy callable.
    ///
    /// - Its [[Prototype]] is `prototype` (the embedder's `HTMLAllCollection.prototype`); it has
    ///   no own `length` or `name`, so the interface's attributes apply.
    /// - Its read-only indexed and unenumerable named properties are live, read through `state`
    ///   by the pure native `length`/`names` views and produced by `getter`/the named getter,
    ///   exactly as for [`Self::install_live_readonly_indexed_properties`].
    /// - Its [[Call]], HTML's legacy caller operation, invokes `call` with the call's receiver
    ///   and arguments.
    pub fn make_html_all_collection(
        &mut self,
        prototype: &Value,
        state: Value,
        length: fn(&Interp, &Value) -> u32,
        getter: Value,
        named: (fn(&Interp, &Value) -> Vec<String>, Value),
        call: Value,
    ) -> Result<Value, Value> {
        let Some(prototype) = prototype.as_obj() else {
            return Err(
                self.make_error("TypeError", "HTMLAllCollection prototype must be an object")
            );
        };
        let (names, named_getter) = named;
        if !getter.is_callable() || !named_getter.is_callable() || !call.is_callable() {
            return Err(self.make_error(
                "TypeError",
                "HTMLAllCollection getters and legacy caller must be callable",
            ));
        }
        fn legacy_caller(
            interp: &mut Interp,
            this: Value,
            args: &[Value],
            captures: &[Value],
        ) -> Result<Value, Value> {
            interp
                .call_callback(captures[0], this, args)
                .map_err(abrupt_value)
        }
        let collection = self.new_native_fn_with_captures("", 0, legacy_caller, vec![call]);
        let object = collection.as_obj().expect("native function object");
        {
            let mut borrowed = object.borrow_mut();
            borrowed.props.remove("length");
            borrowed.props.remove("name");
            borrowed.proto = Some(prototype);
        }
        let ptr = Gc::as_ptr(&object) as usize;
        self.htmldda.insert(ptr, Gc::downgrade(&object));
        // Every JIT object fast path assumes an ordinary, truthy object without platform
        // internal methods. Route this one through the checked helpers.
        object.borrow().ic_plain.set(false);
        self.gc_pin(&object);
        self.host_indexed.insert(
            &self.gc_heap,
            ptr,
            HostIndexedProperties {
                length: 0,
                getter,
                live: Some(Rc::new(HostIndexedLive {
                    state,
                    length,
                    names: Some(names),
                    named_getter: Some(named_getter),
                })),
            },
        );
        Ok(collection)
    }

    /// Define a native method on `target` (non-enumerable, as built-ins are).
    pub(crate) fn def_method(&self, target: &Gc, name: &str, len: usize, f: NativeFn) {
        let func = self.make_native(name, len, f);
        target.borrow_mut().props.insert(
            crate::value::encode_string_key(name).as_ref(),
            Property::builtin(Value::Obj(func)),
        );
    }

    #[cfg(feature = "embed")]
    pub(crate) fn def_scoped_method(
        &self,
        target: &Gc,
        name: &str,
        len: usize,
        f: crate::scope::NativeFn,
    ) {
        let function = self.make_scoped_native(name, len, f);
        target.borrow_mut().props.insert(
            crate::value::encode_string_key(name).as_ref(),
            Property::builtin(Value::Obj(function)),
        );
    }

    #[cfg(feature = "embed")]
    pub(crate) fn define_scoped_global(
        &self,
        name: &str,
        len: usize,
        function: crate::scope::NativeFn,
    ) {
        self.def_scoped_method(&self.global, name, len, function);
    }

    /// Embedder host state (typed per-subsystem slots + the resource table). This is how a
    /// native fn reaches Rust state: `NativeFn` is a bare `fn` pointer and cannot capture.
    pub fn op_state(&mut self) -> &mut crate::host::OpState {
        &mut self.host_state
    }

    /// Shorthand for `op_state().get_mut::<T>()`.
    pub fn host_mut<T: std::any::Any>(&mut self) -> Option<&mut T> {
        self.host_state.get_mut()
    }

    /// Read-only native state access for non-reentrant platform membership queries.
    pub fn host<T: std::any::Any>(&self) -> Option<&T> {
        self.host_state.get()
    }

    /// Shorthand for `op_state().resources`.
    pub fn resource_table(&mut self) -> &mut crate::host::ResourceTable {
        &mut self.host_state.resources
    }

    /// [`call`](Interp::call) with an abrupt completion lowered to the thrown value — the
    /// shape a [`NativeFn`] returns, for native fns that call back into JS.
    ///
    /// A host interruption (`RuntimeInterrupt`) is not a JavaScript exception and cannot be a
    /// thrown value: it surfaces here as `Err(undefined)`. Inside a [`NativeFn`] that is enough,
    /// because the engine resumes the interruption when the native function returns. Host code
    /// calling outside a native function should use [`Self::invoke_interruptible`] (or check
    /// [`Self::pending_interrupt`]); the same holds for the other lowering helpers below.
    pub fn invoke(&mut self, callee: Value, this: Value, args: &[Value]) -> Result<Value, Value> {
        self.call_callback(callee, this, args).map_err(abrupt_value)
    }

    /// [`Self::invoke`] keeping a host interruption distinct from a JavaScript throw.
    #[cfg(feature = "embed")]
    pub(crate) fn invoke_interruptible(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, crate::EvalFailure> {
        // Callback entry polls without the cadence divider, as the engine-level entry points do.
        self.interrupt_poll_force()
            .and_then(|()| self.call_callback(callee, this, args))
            .map_err(|abrupt| match abrupt {
                Abrupt::Interrupt(reason) => crate::EvalFailure::Interrupted(reason),
                other => crate::EvalFailure::Throw(abrupt_value(other)),
            })
    }

    /// The host interruption currently in force, if any: the reason a lowered `Err(undefined)`
    /// from [`Self::invoke`] or a coercion helper was an interruption rather than a throw.
    pub fn pending_interrupt(&self) -> Option<crate::InterruptReason> {
        self.runtime_interrupt.current_reason()
    }

    /// The `typeof` operator's result for `v` (ECMA-262 #sec-typeof-operator), including the
    /// Annex B.3.6 [[IsHTMLDDA]] exception ("undefined"), which [`Value::type_of`] cannot see.
    pub fn type_of(&self, v: &Value) -> &'static str {
        if self.is_htmldda(v) {
            "undefined"
        } else {
            v.type_of()
        }
    }

    /// [`construct`](Interp::construct) with an abrupt completion lowered to the thrown value —
    /// the embed-friendly `new callee(...args)` (used by N-API's `napi_new_instance` /
    /// `napi_create_promise`).
    pub fn construct_value(&mut self, callee: Value, args: &[Value]) -> Result<Value, Value> {
        self.construct(callee, args).map_err(abrupt_value)
    }

    /// ToNumber with the abrupt completion lowered to the thrown value (see [`invoke`]).
    /// Conversions can throw: they may run user `valueOf`/`toString`.
    pub fn coerce_number(&mut self, v: &Value) -> Result<f64, Value> {
        self.to_number(v).map_err(abrupt_value)
    }

    /// ToBigInt followed by wrapping conversion to a signed 64-bit integer. This is the
    /// embedder-facing form required by the WebAssembly JavaScript interface for `i64`: Numbers
    /// are rejected, while booleans, strings, and objects run the language's observable ToBigInt
    /// coercion before the low 64 bits are selected.
    pub fn coerce_bigint_i64(&mut self, v: &Value) -> Result<i64, Value> {
        self.to_bigint(v)
            .map(|value| value.to_i128_wrapping() as i64)
            .map_err(abrupt_value)
    }

    /// ToString as Rust text, with the abrupt completion lowered to the thrown value (see
    /// [`invoke`]). A Rust string holds only Unicode scalar values, so this is the Web IDL
    /// `USVString` conversion: surrogate pairs become their characters and every lone surrogate
    /// becomes U+FFFD. Use [`Interp::coerce_utf16`] where lone surrogates must survive.
    pub fn coerce_string(&mut self, v: &Value) -> Result<Rc<str>, Value> {
        self.to_string(v)
            .map(|s| match crate::jstr::to_text(&s.text()) {
                std::borrow::Cow::Borrowed(text) => Rc::from(text),
                std::borrow::Cow::Owned(text) => Rc::from(text),
            })
            .map_err(abrupt_value)
    }

    /// Web IDL `USVString`: [`Interp::coerce_string`] as an owned `String`. WHATWG Encoding's
    /// `TextEncoder` consumes exactly this scalar-value sequence.
    pub fn coerce_usv_string(&mut self, v: &Value) -> Result<String, Value> {
        let value = self.to_string(v).map_err(abrupt_value)?;
        Ok(crate::jstr::to_text(&value.text()).into_owned())
    }

    /// ToString as UTF-16 code units, lone surrogates included (a Web IDL `DOMString`).
    pub fn coerce_utf16(&mut self, v: &Value) -> Result<Vec<u16>, Value> {
        let value = self.to_string(v).map_err(abrupt_value)?;
        Ok(value.units().to_utf16().into_owned())
    }

    /// ToString as engine text, lone surrogates included (a Web IDL `DOMString` kept as a Rust
    /// string): [`Value::from_engine_text`] rebuilds exactly the same JS string. This is not
    /// Rust text; see [`Value::as_engine_text`].
    pub fn coerce_engine_text(&mut self, v: &Value) -> Result<Rc<str>, Value> {
        self.to_string(v)
            .map(|s| Rc::<str>::from(&s))
            .map_err(abrupt_value)
    }

    /// The bytes a TypedArray view covers (`None` when `v` isn't a typed array or its buffer
    /// is detached). The embedder's binary bridge: encoders/crypto/fetch move bytes through
    /// this.
    pub fn typed_array_bytes(&self, v: &Value) -> Option<Vec<u8>> {
        let obj = v.as_obj()?;
        let info = self
            .typed_arrays
            .get(&(Gc::as_ptr(&obj) as usize))
            .copied()?;
        let len = self.ta_len(&info)?;
        self.ta_read_bytes(&info, 0, len)
    }

    /// Run `f` over the bytes a TypedArray view covers, in place: the same view, bounds and
    /// detached rules as [`Interp::typed_array_bytes`] without copying them. The bytes stay
    /// borrowed while `f` runs, so `f` must not call back into the engine (it may use the
    /// embedder's own state).
    pub fn with_typed_array_bytes<R>(&self, v: &Value, f: impl FnOnce(&[u8]) -> R) -> Option<R> {
        let obj = v.as_obj()?;
        let info = self
            .typed_arrays
            .get(&(Gc::as_ptr(&obj) as usize))
            .copied()?;
        let len = self.ta_len(&info)?;
        let es = info.kind.elsize();
        let (start, end) = (info.offset, info.offset + len * es);
        self.array_buffers
            .get(&info.buffer)?
            .with_bytes(|buf| (end <= buf.len()).then(|| f(&buf[start..end])))?
    }

    /// Snapshot a matching TypedArray over fixed backing storage for Web IDL unions.
    /// Brand, view bounds, detached state and resizability come from intrinsic slots;
    /// user properties and @@iterator are never read. None requires normal conversion
    /// or rejection by the host, never treating an unmatched value as an empty list.
    pub fn fixed_typed_array_bytes(
        &self,
        value: &Value,
        name: &str,
        allow_shared: bool,
    ) -> Option<Vec<u8>> {
        let ptr = self.object_addr(value)?;
        let info = self.typed_arrays.get(&ptr)?;
        if info.kind.name() != name || (!allow_shared && self.is_shared_buffer(info.buffer)) {
            return None;
        }
        let handle = self.ta_buffer.get(&ptr)?.as_obj()?;
        let buffer = handle.borrow();
        if !matches!(
            (buffer
                .internal_slots()
                .get("#\u{0}abResizable")
                .map(|p| p.value()))
            .map(|__value| (__value).view()),
            Some(ValueView::Bool(false))
        ) {
            return None;
        }
        let len = self.ta_len(info)?;
        self.ta_read_bytes(info, 0, len)
    }

    /// Copy the bytes held by a Web IDL `BufferSource` (`ArrayBuffer`, `DataView`, or any
    /// `TypedArray`). View offsets and lengths are honored; detached and out-of-bounds buffers are
    /// rejected. `allow_shared` must be true only for an IDL argument carrying `[AllowShared]`.
    ///
    /// Keeping this union conversion inside the engine prevents embedders from approximating a
    /// `BufferSource` by enumerating JS properties, which loses byte offsets, detached state, and
    /// the distinction between ordinary and shared backing stores.
    pub fn buffer_source_bytes(&self, v: &Value, allow_shared: bool) -> Option<Vec<u8>> {
        let obj = v.as_obj()?;
        let ptr = Gc::as_ptr(&obj) as usize;

        if let Some(info) = self.typed_arrays.get(&ptr).copied() {
            if !allow_shared && self.is_shared_buffer(info.buffer) {
                return None;
            }
            let len = self.ta_len(&info)?;
            return self.ta_read_bytes(&info, 0, len);
        }

        if let Some((buffer, offset, fixed_len, length_tracking)) =
            self.data_views.get(&ptr).copied()
        {
            let data = self.array_buffers.get(&buffer)?;
            if matches!(data, BufferData::Shared(_)) && !allow_shared {
                return None;
            }
            // Copy only the view's bytes, not the whole backing store.
            let view = |bytes: &[u8]| {
                let len = if length_tracking {
                    bytes.len().checked_sub(offset)?
                } else {
                    let end = offset.checked_add(fixed_len)?;
                    (end <= bytes.len()).then_some(fixed_len)?
                };
                Some(bytes[offset..offset + len].to_vec())
            };
            return data.with_bytes(view)?;
        }

        match self.array_buffers.get(&ptr)? {
            BufferData::Shared(_) if !allow_shared => None,
            data => data.with_bytes(<[u8]>::to_vec),
        }
    }

    /// A fresh fixed-length `ArrayBuffer` initialized with `bytes`, with the realm's intrinsic
    /// %ArrayBuffer.prototype%. Creating it runs no author code: a replaced `globalThis.ArrayBuffer`
    /// cannot observe or substitute host-created buffers. The buffer has ordinary ECMAScript
    /// storage; embedders that mirror external memory can update it with
    /// [`Self::array_buffer_set_bytes`].
    pub fn make_array_buffer(&mut self, bytes: &[u8]) -> Result<Value, Value> {
        self.note_buffer_allocation_deferred(bytes.len());
        Ok(
            crate::builtins::array_buffer_with_storage(self, Rc::new(RefCell::new(bytes.to_vec())))
                .0,
        )
    }

    /// A fresh fixed-length `ArrayBuffer` with an embedder detach key. Author JavaScript cannot
    /// transfer/detach it, while [`Self::detach_array_buffer`] retains the key-aware host path.
    /// This models the `"WebAssembly.Memory"` key required for `Memory.buffer`.
    pub fn make_host_keyed_array_buffer(&mut self, bytes: &[u8]) -> Result<Value, Value> {
        self.note_buffer_allocation_deferred(bytes.len());
        self.make_host_keyed_array_buffer_from_storage(Rc::new(RefCell::new(bytes.to_vec())))
    }

    /// Create a fixed-length host-keyed `ArrayBuffer` identified with an existing Data Block.
    /// The WebAssembly JS API §4.1 requires writes through either side to update the other side;
    /// retaining the `Rc` here implements that identity without a mirror or synchronization pass.
    /// Like [`Self::make_array_buffer`] it uses the intrinsic prototype ("create a fixed length
    /// memory buffer" makes a new ArrayBuffer directly), so no author code runs and no
    /// placeholder Data Block is allocated.
    pub fn make_host_keyed_array_buffer_from_storage(
        &mut self,
        storage: ArrayBufferBytes,
    ) -> Result<Value, Value> {
        let (buffer, ptr) = crate::builtins::array_buffer_with_storage(self, storage);
        self.host_keyed_buffers.insert(ptr);
        Ok(buffer)
    }

    /// Replace an ordinary, attached `ArrayBuffer`'s bytes without changing its identity. Returns
    /// `false` for another value, a shared/detached buffer, or a length mismatch. Requiring an
    /// exact length keeps existing DataView and TypedArray bounds invariant.
    pub fn array_buffer_set_bytes(&mut self, v: &Value, bytes: &[u8]) -> bool {
        let Some(obj) = v.as_obj() else {
            return false;
        };
        let ptr = Gc::as_ptr(&obj) as usize;
        let Some(storage) = self.local_buffer(ptr) else {
            return false;
        };
        let mut buffer = storage.borrow_mut();
        if buffer.len() != bytes.len() {
            return false;
        }
        buffer.copy_from_slice(bytes);
        // This is an embedder-side replacement, not a JavaScript mutation.
        // Keep the mutation generation reserved for writes performed by the
        // JS engine so host synchronization does not immediately echo the
        // same bytes back to the embedder on the next callback.
        true
    }

    /// Replace a range of an ordinary, attached `ArrayBuffer` without changing its identity or
    /// JavaScript mutation generation. This is the range-aware counterpart to
    /// [`Self::array_buffer_set_bytes`] for embedders mirroring a host-owned data block. Returns
    /// `false` for another value, a shared/detached buffer, or a range outside the buffer.
    pub fn array_buffer_set_range(&mut self, v: &Value, start: usize, bytes: &[u8]) -> bool {
        let Some(obj) = v.as_obj() else {
            return false;
        };
        let ptr = Gc::as_ptr(&obj) as usize;
        let Some(storage) = self.local_buffer(ptr) else {
            return false;
        };
        let mut buffer = storage.borrow_mut();
        let Some(end) = start.checked_add(bytes.len()) else {
            return false;
        };
        if end > buffer.len() {
            return false;
        }
        buffer[start..end].copy_from_slice(bytes);
        true
    }

    /// Copies a range from an ordinary, attached `ArrayBuffer` into an embedder-owned buffer.
    /// Returns `false` for another value, a shared/detached buffer, or a range outside either.
    pub fn array_buffer_copy_range(
        &self,
        v: &Value,
        start: usize,
        len: usize,
        destination: &mut [u8],
    ) -> bool {
        let Some(obj) = v.as_obj() else {
            return false;
        };
        let ptr = Gc::as_ptr(&obj) as usize;
        let Some(storage) = self.local_buffer(ptr) else {
            return false;
        };
        let buffer = storage.borrow();
        let Some(end) = start.checked_add(len) else {
            return false;
        };
        if destination.len() != len || end > buffer.len() {
            return false;
        }
        destination.copy_from_slice(&buffer[start..end]);
        true
    }

    /// Takes coalesced byte ranges covering JavaScript writes since the previous call. Highly
    /// fragmented writes collapse to the whole buffer, so ranges may include unchanged bytes.
    /// As with a full-buffer copy, the embedder must synchronize host writes before allowing JS
    /// to modify the mirror. Taking the ranges does not change the mutation generation. `None`
    /// for another value, a detached buffer or a SharedArrayBuffer, which has no dirty ranges.
    pub fn take_array_buffer_dirty_ranges(&mut self, v: &Value) -> Option<Vec<Range<usize>>> {
        let ptr = v.as_obj().map(|obj| Gc::as_ptr(&obj) as usize)?;
        self.flush_native_buffer_writes(ptr);
        self.local_buffer(ptr).is_some().then(|| {
            self.array_buffer_dirty_ranges
                .remove(&ptr)
                .unwrap_or_default()
        })
    }

    /// Returns the mutation generation of an attached ordinary ArrayBuffer. The generation is
    /// intended for embedders that mirror an ArrayBuffer into another storage; it is not an
    /// observable JavaScript value and may wrap after `u64::MAX` mutations. `None` for another
    /// value, a detached buffer or a SharedArrayBuffer: other agents write a Shared Data Block
    /// without any generation this agent could observe.
    pub fn array_buffer_version(&self, v: &Value) -> Option<u64> {
        let ptr = v.as_obj().map(|obj| Gc::as_ptr(&obj) as usize)?;
        self.local_buffer(ptr).is_some().then(|| {
            self.array_buffer_versions
                .get(&ptr)
                .copied()
                .unwrap_or(0)
                .wrapping_add(
                    self.native_buffers
                        .get(&ptr)
                        .map_or(0, |buffer| buffer.writes.get()),
                )
        })
    }

    /// Marks bytes in an ArrayBuffer as changed by an engine-owned view or host pointer. This is
    /// deliberately conservative: callers may mark a no-op or failed write, causing one extra
    /// mirror check but never allowing a stale external view to remain visible.
    pub(crate) fn mark_array_buffer_dirty(&mut self, ptr: usize) {
        let len = self
            .local_buffer(ptr)
            .map_or(0, |buffer| buffer.borrow().len());
        self.mark_array_buffer_dirty_range(ptr, 0, len);
    }

    /// Marks a byte range as changed by an engine-owned view or host pointer. Only an
    /// ArrayBuffer's own bytes are tracked: a Shared Data Block, which other agents write too,
    /// has no mutation generation or dirty ranges.
    pub(crate) fn mark_array_buffer_dirty_range(&mut self, ptr: usize, start: usize, len: usize) {
        let Some(buffer_len) = self.local_buffer(ptr).map(|buffer| buffer.borrow().len()) else {
            return;
        };
        let Some(end) = start.checked_add(len) else {
            return;
        };
        if len == 0 || end > buffer_len {
            return;
        }
        let ranges = self.array_buffer_dirty_ranges.entry(ptr).or_default();
        record_buffer_write(ranges, start..end, buffer_len);
        let version = self.array_buffer_versions.entry(ptr).or_default();
        *version = version.wrapping_add(1);
    }

    /// Detach an ordinary `ArrayBuffer`, making its byte length zero and invalidating its views.
    /// Returns `false` for another value, a shared buffer, or an already-detached buffer.
    pub fn detach_array_buffer(&mut self, v: &Value) -> bool {
        let Some(obj) = v.as_obj() else {
            return false;
        };
        let ptr = Gc::as_ptr(&obj) as usize;
        // A Shared Data Block cannot be detached.
        if self.is_shared_buffer(ptr) {
            return false;
        }
        self.host_keyed_buffers.remove(&ptr);
        self.invalidate_native_buffer(ptr);
        let removed = self.array_buffers.remove(&ptr).is_some();
        self.array_buffer_versions.remove(&ptr);
        self.array_buffer_dirty_ranges.remove(&ptr);
        removed
    }

    /// Overwrite a TypedArray's leading bytes. The write must fit the view's current byte length:
    /// `false` (and no write) when `v` isn't a typed array, its buffer is detached or out of
    /// bounds, or `bytes` is longer than the view; bytes beyond the view are never touched.
    pub fn typed_array_set_bytes(&mut self, v: &Value, bytes: &[u8]) -> bool {
        let Some(obj) = v.as_obj() else {
            return false;
        };
        let Some(info) = self.typed_arrays.get(&(Gc::as_ptr(&obj) as usize)).copied() else {
            return false;
        };
        let Some(length) = self.ta_len(&info) else {
            return false;
        };
        if bytes.len() > length * info.kind.elsize() {
            return false;
        }
        self.ta_write_bytes(&info, 0, bytes);
        true
    }

    /// A fresh `Uint8Array` holding `bytes`: AllocateTypedArray with the current Realm's
    /// intrinsic %Uint8Array.prototype% over a new ArrayBuffer. No author code runs, so a replaced
    /// `globalThis.Uint8Array` cannot observe or substitute the result.
    pub fn make_uint8array(&mut self, bytes: &[u8]) -> Result<Value, Value> {
        self.note_buffer_allocation_deferred(bytes.len());
        Ok(
            crate::builtins::typed_array_from_bytes(self, crate::value::TaKind::U8, bytes.to_vec())
                .0,
        )
    }

    pub(crate) fn make_function(&self, func: Rc<Function>, env: Env) -> Value {
        self.make_function_named(func, env, None)
    }

    /// [`Interp::make_function`] followed by SetFunctionName(F, `name`) (ECMA-262 §10.2.9) for an
    /// anonymous function expression's NamedEvaluation or a concise method's key: the name is
    /// part of the cloned property template.
    pub(crate) fn make_function_named(
        &self,
        func: Rc<Function>,
        env: Env,
        name: Option<&Rc<str>>,
    ) -> Value {
        let is_arrow = func.is_arrow;
        let is_method = func.is_method;
        let is_generator = func.is_generator;
        let is_async = func.is_async;
        // A generator / async / async-generator function object's [[Prototype]] is the matching
        // intrinsic (%GeneratorFunction.prototype% etc.), not %Function.prototype%. Async arrows
        // use %AsyncFunction.prototype% too.
        let fn_proto = match (is_arrow, is_async, is_generator) {
            (false, false, true) => self.extra_protos.get("%GeneratorFunction.prototype%"),
            (_, true, false) => self.extra_protos.get("%AsyncFunction.prototype%"),
            (false, true, true) => self.extra_protos.get("%AsyncGeneratorFunction.prototype%"),
            _ => None,
        }
        .cloned()
        .unwrap_or(self.function_proto);
        // A `prototype` is present on ordinary functions and on generators — sync OR async (even
        // generator methods); arrows, plain async functions, and concise methods/getters/setters have
        // none. A generator's `.prototype` chains to %GeneratorPrototype% / %AsyncGeneratorPrototype%
        // (an ordinary function's is a plain object). Only ordinary functions are constructors.
        let has_prototype = !is_arrow && (is_generator || (!is_async && !is_method));
        crate::workload_metrics::function_created(&func, has_prototype);
        // Per-function, heap-local map templates (see `FunctionRuntime::fn_maps`) pay the shape
        // transitions, key hashing and name string once. Each closure copies the field words,
        // shares the layout, and patches the two identity-dependent values in place; value
        // writes never touch the shape.
        let runtime = func.runtime();
        let maps = runtime.fn_maps.get_or_init(|| {
            let arity = func
                .params
                .iter()
                .take_while(|p| p.default.is_none() && !p.rest)
                .count();
            let name = Value::lstr(func.name.clone().unwrap_or_default());
            let mut p = crate::value::Props::new();
            p.insert(
                crate::value::fn_key(0), // "length"
                Property::data(Value::Num(arity as f64), false, false, true),
            );
            p.insert(
                crate::value::fn_key(1), // "name"
                Property::data(name, false, false, true),
            );
            if has_prototype {
                p.insert(
                    crate::value::fn_key(2), // "prototype" (placeholder value; patched per clone)
                    Property::data(Value::Undefined, true, false, false),
                );
            }
            let pp = if has_prototype {
                let mut q = crate::value::Props::new();
                // A generator's `.prototype` has no own `constructor` (the methods live on the
                // intrinsic).
                if !is_generator {
                    q.insert(
                        crate::value::fn_key(3), // "constructor" (placeholder; patched per clone)
                        Property::builtin(Value::Undefined),
                    );
                }
                Some(q)
            } else {
                None
            };
            p.mark_layout_shared();
            if let Some(props) = &pp {
                props.mark_layout_shared();
            }
            let maps = Rc::new(crate::ast::FunctionMaps {
                function: p,
                prototype: pp,
                named: std::cell::RefCell::new(None),
                heap: crate::value::active_heap_id(),
            });
            crate::value::register_code_roots(&maps);
            maps
        });
        let proto_map = &maps.prototype;
        // SetFunctionName: NamedEvaluation of an anonymous function expression, or a concise
        // method's key (MethodDefinitionEvaluation). A function with its own name keeps it; the
        // template's `name` slot is entry 1 with the "" value.
        let named = name.filter(|_| func.name.is_none());
        debug_assert_eq!(maps.heap, crate::value::active_heap_id());
        let fn_map = match named {
            None => maps.function.clone(),
            Some(name) => {
                let mut named = maps.named.borrow_mut();
                match &*named {
                    // A creation site passes its chunk's interned name, so identity is the common
                    // proof; equal text from another site shares the template too.
                    Some((cached, props)) if Rc::ptr_eq(cached, name) || **cached == **name => {
                        props.clone()
                    }
                    _ => {
                        let mut props = maps.function.clone();
                        let name_value = Value::lstr(name.to_string());
                        assert!(props.set_value_at(1, name_value), "function name slot");
                        if named.is_none() {
                            *named = Some((name.clone(), props.clone()));
                            // A late fill of the registered maps (see
                            // `value::remember_code_edges`): the stored copy's own blocks.
                            if let Some((_, stored)) = &*named {
                                crate::value::remember_code_edges(stored);
                            }
                            props.mark_layout_shared();
                        }
                        props
                    }
                }
            }
        };
        // Instantiate{Ordinary,Generator,Async}FunctionExpression creates the immutable
        // self-name environment ONCE, outside the activation, when the closure is created.
        // Retain it in [[Environment]] so all execution tiers and recursive calls share the
        // correct binding without allocating an extra scope on every call.
        let self_env = if func.needs_self_environment() {
            Some(new_scope(Some(env)))
        } else {
            None
        };
        let obj = Object::new(Some(fn_proto));
        {
            let mut b = obj.borrow_mut();
            b.call = Callable::user(
                func.clone(),
                self_env.as_ref().cloned().unwrap_or(env),
                Gc::as_ptr(&self.global) as usize,
                if func.self_hosted {
                    None
                } else {
                    self.active_runtime_origin()
                },
                self.realm_id,
            );
            b.props = fn_map;
        }
        if has_prototype {
            let proto_parent = match (is_generator, is_async) {
                (true, false) => self.extra_protos.get("%GeneratorPrototype%").cloned(),
                (true, true) => self.extra_protos.get("%AsyncGeneratorPrototype%").cloned(),
                _ => None,
            }
            .or(Some(self.object_proto));
            assert!(
                obj.borrow_mut().props.defer_function_prototype_at(
                    2,
                    &obj,
                    proto_parent.expect("function prototype intrinsic parent"),
                    proto_map.as_ref().expect("prototype template"),
                ),
                "prototype slot"
            );
        }
        if !is_arrow && !is_method && !is_async && !is_generator {
            obj.borrow_mut().is_constructor = true;
        }
        if let Some(self_env) = self_env {
            self_env.borrow_mut().vars.insert(
                func.name.as_ref().expect("named expression").clone(),
                // CreateImmutableBinding(name, false): sloppy writes are silent no-ops.
                Binding::data(Value::Obj(obj), false, true).with_strict_immutable(false),
            );
        }
        Value::Obj(obj)
    }

    // ----- property access --------------------------------------------------------------------

    /// Get `base[key]`, walking the prototype chain and invoking getters. Primitive bases are
    /// handled by routing to their wrapper prototype (and string index/length specially). The
    /// receiver defaults to `base`.
    /// Read engine-only state from this object. Internal slots are never inherited and no
    /// Proxy, accessor or host property callback participates in this lookup.
    pub(crate) fn get_internal_slot(&self, base: &Value, key: &str) -> Result<Value, Abrupt> {
        Ok(base
            .as_obj()
            .and_then(|object| {
                object
                    .borrow()
                    .internal_slots()
                    .get(key)
                    .map(Property::value)
            })
            .unwrap_or(Value::Undefined))
    }

    pub(crate) fn set_internal_slot(
        &self,
        base: &Value,
        key: &str,
        value: Value,
    ) -> Result<(), Abrupt> {
        if let Some(object) = base.as_obj() {
            object
                .borrow_mut()
                .internal_slots_mut()
                .insert(key, Property::plain(value));
        }
        Ok(())
    }

    /// Read an author String key. Every String, including a leading-NUL name, is
    /// distinct from every Symbol key; hosts cannot forge an encoded Symbol identity.
    pub(crate) fn get_member(&mut self, base: &Value, key: &str) -> Result<Value, Abrupt> {
        self.get_member_key(base, &crate::value::encode_string_key(key))
    }

    /// Engine [[Get]] for an already encoded String-or-Symbol property key.
    pub(crate) fn get_member_key(&mut self, base: &Value, key: &str) -> Result<Value, Abrupt> {
        let result = self.get_member_recv(base, key, *base);
        if matches!(
            (result).as_ref().map(|__value| (__value).view()),
            Ok(ValueView::Undefined)
        ) {
            self.trace_undefined_property(base, key);
        }
        result
    }

    // Diagnostic history of completed slow-path reads, not an absence proof:
    // an accessor/Proxy can also deliberately return undefined. OrdinaryGet
    // has already executed exactly once; never replay author code to inspect it.
    fn trace_undefined_property(&self, base: &Value, key: &str) {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED
            .get_or_init(|| std::env::var_os("TRUST_JS_TRACE_UNDEFINED_PROPERTIES").is_some())
        {
            return;
        }
        let function = self
            .fn_frames
            .last()
            .and_then(|frame| {
                let callee = frame.callee();
                let object = callee.borrow();
                match (object.props.get("name").map(|property| property.value()))
                    .map(|__value| (__value).view())
                {
                    Some(ValueView::Str(name)) => {
                        Some(name.text().chars().take(80).collect::<String>())
                    }
                    _ => None,
                }
            })
            .unwrap_or_default();
        let key = key.chars().take(120).collect::<String>();
        let mut entry = format!("function={function:?} base={} key={key:?}", type_name(base));
        if canonical_index(&key).is_none() {
            if let ValueView::Obj(object) = &(base).view() {
                let id = Gc::as_ptr(object) as usize;
                let object = object.borrow();
                let kind = if let Some(data) = self.buffer_data(id) {
                    match data {
                        BufferData::Local(_) => "ArrayBuffer",
                        BufferData::Shared(_) => "SharedArrayBuffer",
                    }
                } else if self.typed_arrays.contains_key(&id) {
                    "TypedArray"
                } else {
                    match &object.exotic {
                        Exotic::Array => "Array",
                        Exotic::StrWrap(_) => "String object",
                        Exotic::Arguments => "Arguments",
                        Exotic::None => "ordinary/host object",
                        _ => "other exotic",
                    }
                };
                // Read only native descriptors. Do not enumerate a Proxy,
                // call a constructor/tag getter, or retain the receiver.
                let names = object
                    .props
                    .iter()
                    .take(8)
                    .map(|(name, _)| name.text().chars().take(60).collect::<String>())
                    .collect::<Vec<_>>();
                let constructor = object.proto.as_ref().and_then(|prototype| {
                    let prototype = prototype.borrow();
                    match (prototype
                        .props
                        .get("constructor")
                        .map(|property| property.value()))
                    .map(|__value| (__value).view())
                    {
                        Some(ValueView::Obj(function)) => {
                            let function = function.borrow();
                            match (function.props.get("name").map(|property| property.value()))
                                .map(|__value| (__value).view())
                            {
                                Some(ValueView::Str(name)) => {
                                    Some(name.text().chars().take(80).collect::<String>())
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                });
                entry.push_str(&format!(
                    " kind={kind:?} constructor={constructor:?} own={names:?}"
                ));
            }
        }
        MISSING_PROPERTY_TRACE.with(|trace| {
            let mut trace = trace.borrow_mut();
            if trace.len() == 32 {
                trace.pop_front();
            }
            trace.push_back(entry);
        });
    }

    /// Profile-enabled `[[Get]]` entry used by computed element operations. It executes the
    /// canonical property algorithm once and records the branch that determined the result;
    /// getters, proxy traps, and conversions are never replayed for diagnostics.
    pub(crate) fn get_member_profiled(
        &mut self,
        base: &Value,
        key: &str,
        trace: &mut crate::feedback::CurrentPropertyTrace,
    ) -> Result<Value, Abrupt> {
        trace.receiver_shape = self.property_receiver_shape(base);
        self.get_member_recv_impl(base, key, *base, Some(trace), None)
    }

    /// Whether `o` may take the dense element fast paths: a plain object or array — not an
    /// exotic (wrapper/arguments), module namespace, or deferred namespace, whose reads/writes
    /// have extra semantics the fast paths would skip. (Typed arrays and proxies keep their
    /// indexed data out of `props`, so a dense hit on them is impossible and they need no guard.)
    #[inline]
    /// Whether appending a NEW element to array `o` can skip the OrdinarySet prototype walk:
    /// `o`'s prototype is exactly the realm's `Array.prototype` and the protector proof holds
    /// (see [`Interp::elems_protector`]). The caller has already verified `o` itself.
    pub(crate) fn array_append_unshadowed(&self, o: &Gc) -> bool {
        {
            let b = o.borrow();
            match b.proto.as_ref() {
                Some(p) if Gc::ptr_eq(p, &self.array_proto) => {}
                _ => return false,
            }
        }
        self.array_prototypes_unshadowed()
    }

    /// Realm-wide half of [`Self::array_append_unshadowed`]. Callers that already hold and have
    /// validated an array receiver can use this without releasing and re-borrowing it merely to
    /// repeat the prototype identity check.
    #[inline]
    fn array_prototypes_unshadowed(&self) -> bool {
        let epoch = crate::value::proto_epoch();
        if epoch == u32::MAX {
            return false; // permanently invalidated — never trust the cache
        }
        let (e, ok) = self.elems_protector.get();
        if e == epoch {
            return ok;
        }
        let no_index_keys = |g: &Gc| {
            // OrdinarySet must observe inherited indexed setters/data descriptors
            // whether that prototype uses named entries or contiguous elements.
            g.borrow().props.indexed_properties().next().is_none()
        };
        let ap = &self.array_proto;
        let op = &self.object_proto;
        let ok = self.ordinary_get_ptr(Gc::as_ptr(ap) as usize)
            && self.ordinary_get_ptr(Gc::as_ptr(op) as usize)
            && matches!(ap.borrow().proto.as_ref(), Some(p) if Gc::ptr_eq(p, op))
            && op.borrow().proto.is_none()
            && op.borrow().exotic.is_ordinary()
            && no_index_keys(ap)
            && no_index_keys(op)
            && {
                // Future structural inserts on either proto must bump the epoch this proof is
                // keyed on (plain `Array.prototype[0] = v` isn't a defineProperty).
                ap.borrow().props.mark_proto();
                op.borrow().props.mark_proto();
                true
            };
        self.elems_protector.set((epoch, ok));
        ok
    }

    fn plain_non_typed_for_elems(&self, o: &Gc) -> bool {
        // Numeric indexed access is an ordinary [[Get]] shortcut.  Proxies, typed arrays,
        // module/deferred namespaces, and Web IDL indexed objects all have independent internal
        // methods or backing stores; even an own-looking entry must not bypass those semantics
        // (ECMA-262 §10.5.8 and §10.4.5).
        // The caller already looked up TypedArray identity while selecting its direct path.
        if !self.non_typed_ordinary_get_ptr(Gc::as_ptr(o) as usize) {
            return false;
        }
        if !(matches!(o.borrow().exotic, Exotic::Array) || o.borrow().exotic.is_ordinary()) {
            return false;
        }
        true
    }

    /// HasOwnProperty of element `index` for an ordinary Array or object whose element storage
    /// holds it (a data or accessor property). A set `ic_plain` byte excludes every side-table
    /// internal method (Proxy, typed array, namespace, Web IDL indexed object), and the `exotic`
    /// check excludes arguments and wrapper objects. `false` means "use the general algorithm"
    /// (holes and far keys consult the string-keyed map and the prototype chain there), never
    /// "absent".
    #[inline]
    pub(crate) fn plain_own_element_present(object: &Gc, index: u32) -> bool {
        let body = object.borrow();
        body.ic_plain.get()
            && (matches!(body.exotic, Exotic::Array) || body.exotic.is_ordinary())
            && body.props.get_index(index).is_some()
    }

    /// `o[n]` read fast path: a dense data element or a Number-content TypedArray index, without
    /// stringifying the index. `None` means "take the generic path", never "absent".
    #[inline]
    pub(crate) fn fast_get_elem(&mut self, o: &Gc, n: f64) -> Option<Value> {
        if n.trunc() != n || !(0.0..u32::MAX as f64).contains(&n) {
            return None;
        }
        let mut b = o.borrow();
        // A set `ic_plain` byte proves no side-table internal methods (see `object_ic_plain`).
        if !b.ic_plain.get() {
            drop(b);
            if let Some(info) = self.typed_arrays.get(&(Gc::as_ptr(o) as usize)) {
                // ECMA-262 ToPropertyKey + TypedArrayGetElement (snapshot e28783d5fc9d): a
                // non-negative integral Number already identifies the element. Numeric -0
                // becomes the key "0"; the distinct string "-0" still goes through the generic
                // path. The identity map cannot match a Proxy wrapping a TypedArray. ta_read
                // retains detach/resize bounds and shared-memory synchronization. BigInt views
                // stay on the general path: speculative numeric UpdateElem callers would discard
                // that result and read again, which must not add a shared-memory read event.
                return (!info.kind.is_bigint()).then(|| self.ta_read(info, n as usize));
            }
            if !self.plain_non_typed_for_elems(o) {
                return None;
            }
            b = o.borrow();
        } else if !(matches!(b.exotic, Exotic::Array) || b.exotic.is_ordinary()) {
            return None;
        }
        // One-load mirror hit (see `Props::mirror`); a miss only means "answer classically".
        if let Some(f) = b.props.mirror_get(n as u32) {
            return Some(Value::Num(f));
        }
        let p = b.props.get_index(n as u32)?;
        if p.accessor() {
            return None;
        }
        Some(p.value())
    }

    /// `o[n] = v` write fast path: a writable dense data element or numeric TypedArray store.
    /// Correct regardless of the prototype chain — an own writable data property always wins
    /// OrdinarySet. Returns the value back on miss so the caller runs the generic path.
    #[inline]
    pub(crate) fn fast_set_elem(&mut self, o: &Gc, n: f64, mut v: Value) -> Result<(), Value> {
        if n.trunc() != n || !(0.0..u32::MAX as f64).contains(&n) {
            return Err(v);
        }
        // A Proxy/typed array/module/host object owns a different [[Set]] algorithm.  Do this
        // identity check before probing the ordinary property storage so a future side-table
        // representation cannot accidentally bypass its trap or backing store. A set
        // `ic_plain` byte proves the object is in none of those tables.
        if !o.borrow().ic_plain.get() {
            if let Some(info) = self.typed_arrays.get(&(Gc::as_ptr(o) as usize)).copied() {
                // TypedArraySetElement converts the value before checking current bounds. Only
                // a value already of the view's content type (a Number, or a BigInt for BigInt
                // content) can bypass that observable step; coercions, mismatched content and
                // immutable-buffer errors retain the general path.
                match &(v).view() {
                    ValueView::Num(value)
                        if !info.kind.is_bigint()
                            && !self.immutable_buffers.contains(&info.buffer) =>
                    {
                        self.ta_write(&info, n as usize, *value);
                        return Ok(());
                    }
                    ValueView::BigInt(value)
                        if info.kind.is_bigint()
                            && !self.immutable_buffers.contains(&info.buffer) =>
                    {
                        self.ta_write_bigint(&info, n as usize, value.to_i128_wrapping());
                        return Ok(());
                    }
                    _ => {}
                }
                return Err(v);
            }
            if !self.non_typed_ordinary_get_ptr(Gc::as_ptr(o) as usize) {
                return Err(v);
            }
        }
        let n = n as u32;
        let mut b = o.borrow_mut();
        if !(matches!(b.exotic, Exotic::Array) || b.exotic.is_ordinary()) {
            return Err(v);
        }
        // Mirror-coherent overwrite (does NOT invalidate the mirror like a raw get_index_mut).
        match b.props.set_index_value(n, v) {
            Ok(()) => return Ok(()),
            Err(back) => v = back,
        }
        // A fresh dense element is the other overwhelmingly common array write. The protector
        // proof executes no JS, so all receiver validation and the eventual commit can safely
        // remain under this single borrow.
        if !matches!(b.exotic, Exotic::Array)
            || !b.extensible
            || !matches!(b.proto.as_ref(), Some(p) if Gc::ptr_eq(p, &self.array_proto))
            || !self.array_prototypes_unshadowed()
        {
            return Err(v);
        }
        let Some((old_len, length_writable)) =
            b.props
                .length_property()
                .and_then(|p| match (p.value()).view() {
                    ValueView::Num(len) => Some((len, p.writable())),
                    _ => None,
                })
        else {
            return Err(v);
        };
        if n as f64 >= old_len && !length_writable {
            return Err(v);
        }
        // Sequential growth is the common case and `try_append_element` only needs to
        // validate the dense frontier. Keep the more general hole/padding machinery as
        // the fallback for out-of-order writes.
        let prop = b.props.try_append_element(n, Property::plain(v)).err();
        let result = match prop {
            None => Ok(()),
            Some(prop) => b.props.try_define_dense_element(n, prop),
        };
        match result {
            Ok(()) => {
                if n as f64 >= old_len {
                    b.props.set_array_length_value(n as f64 + 1.0);
                }
                Ok(())
            }
            Err(prop) => Err(prop.into_value()),
        }
    }

    /// `obj.name` read with a per-site inline cache (bytecode `GetProp`/`GetMethod`). The cache
    /// holds where `name` was last found at this site: an own `entries` slot, or a slot on the
    /// object `depth` prototype hops up (methods!). A hit re-validates everything it relies on —
    /// each hop below the holder is hash-checked to still lack an own `name`, the holder slot is
    /// key-compared — so staleness costs a re-derivation, never a wrong answer. Anything not a
    /// ordinary descriptor on an ordinary (or array) chain falls through to full `[[Get]]`.
    /// Accessor hits snapshot the live getter and perform Call after all lookup borrows end.
    pub(crate) fn get_prop_ic(
        &mut self,
        base: &Value,
        name: &str,
        key: crate::atom::PropKey,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> Result<Value, Abrupt> {
        self.get_prop_ic_impl(base, name, key, cache, true, None)
    }

    /// Compact execution-storage form of OrdinaryGet. Cache resolution borrows
    /// the receiver only while no author code can run, and snapshots the live
    /// descriptor's owned value/getter. Data hits never widen either owner. A
    /// getter or exotic path first roots a separate receiver, since it may replace
    /// the caller's local/operand slot while running.
    ///
    /// # Safety
    /// `base` must point to an initialized, rooted execution slot at entry. Its
    /// borrow ends before any JS call: keep the pointer raw at this boundary so
    /// reentry can replace the slot without a protected Rust reference to it.
    pub(crate) unsafe fn get_prop_ic_from_slot(
        &mut self,
        base: *const Value,
        name: &str,
        key: crate::atom::PropKey,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> Result<Value, Abrupt> {
        // Preserve the named-read String length shortcut without widening or
        // retaining a temporary string, including when GetProp moved its operand.
        if name == "length" {
            if let Some(length) = unsafe { &*base }.with_string(|text| self.str_len(text)) {
                return Ok(Value::Num(length as f64));
            }
        }
        let read =
            unsafe { &*base }.with_object(|object| self.try_ic_get(object, name, key, cache, true));
        match read {
            Some(Some(read)) => match read.into_data() {
                Ok(value) => Ok(value),
                Err(read) => {
                    let receiver = *unsafe { &*base };
                    self.finish_cached_get(read, &receiver, None)
                }
            },
            Some(None) => {
                // The object IC already declined. Do not repeat its probes before
                // the full exotic/prototype algorithm.
                let receiver = *unsafe { &*base };
                self.get_member_recv_impl(&receiver, name, receiver, None, Some(key))
            }
            None => {
                let receiver = *unsafe { &*base };
                self.get_prop_ic(&receiver, name, key, cache)
            }
        }
    }

    /// Profile-enabled form of [`Self::get_prop_ic`]. Slow paths fill `trace` at the exact
    /// OrdinaryGet/accessor/exotic branch that determines the result (ECMA-262 §10.1.8.1 and
    /// §10.5.8). IC hits need no duplicate work: the chunk's existing IC adapter records them at
    /// the diagnostic safepoint.
    pub(crate) fn get_prop_ic_profiled(
        &mut self,
        base: &Value,
        name: &str,
        key: crate::atom::PropKey,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
        trace: &mut crate::feedback::CurrentPropertyTrace,
    ) -> Result<Value, Abrupt> {
        trace.receiver_shape = self.property_receiver_shape(base);
        self.get_prop_ic_impl(base, name, key, cache, true, Some(trace))
    }

    /// Computed string-key property read. The caller clears its scratch ways for each key:
    /// unlike a named site, those cells alone do not establish the property's name. The
    /// bounded stub cache compares its entries' keys (`stub_cache_get`), so it is safe to
    /// reuse for transient keys. Long transient keys stay out of it. Text with no atom names no
    /// property: the full [[Get]] answers it.
    #[cfg(test)]
    fn get_prop_keyed(
        &mut self,
        base: &Value,
        name: &str,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> Result<Value, Abrupt> {
        let Some(key) = crate::atom::PropKey::lookup_text(name) else {
            return self.get_member_key(base, name);
        };
        self.get_prop_ic_impl(
            base,
            name,
            key,
            cache,
            name.len() <= COMPUTED_STUB_KEY_MAX_BYTES,
            None,
        )
    }

    /// GetValue / OrdinaryGet for an already-string computed key. Pointer-keyed resolutions
    /// are only hints: live shapes, prototype links, exotic gates and descriptors are checked
    /// on every hit. Accessors retain the original primitive/object receiver and run once.
    pub(crate) fn get_computed_property(
        &mut self,
        base: &Value,
        key: &crate::lstr::LStr,
    ) -> Result<Value, Abrupt> {
        let text = key.text();
        if text.starts_with('\0') {
            return self.get_member_key(base, &crate::value::encode_string_key(&text));
        }
        let cacheable = text.len() <= COMPUTED_STUB_KEY_MAX_BYTES
            && &*text != "length"
            && &*text != "description"
            && !text.as_bytes().first().is_some_and(u8::is_ascii_digit);
        if cacheable {
            let object = match (base).view() {
                ValueView::Obj(object) => Some(object),
                ValueView::Str(_) => Some(self.string_proto),
                ValueView::Num(_) => Some(self.number_proto),
                ValueView::Bool(_) => Some(self.boolean_proto),
                _ => None,
            };
            if let Some(object) = object {
                let head = Gc::as_ptr(&object);
                let shape = object.borrow().props.shape();
                if let Some(state) = self.computed_reads.lookup(key.as_ptr() as usize, shape) {
                    if let Some(value) = self.ic_shape_probe(head, &text, None, state) {
                        return self.finish_cached_get(value, base, None);
                    }
                }
                // This probe/derivation never executes author code. Do not harvest scratch
                // IC state after a getter or Proxy callback: reentrancy can change its meaning.
                let scratch = [const { std::cell::Cell::new(crate::bytecode::IcState::EMPTY) };
                    crate::bytecode::PROP_IC_WAYS];
                // A computed-cache miss already rejected the identity/shape resolution. The
                // scratch ways are empty; probing them and copying the key into a second
                // pointer-indexed stub table adds work on high-cardinality/transient inputs.
                // Derive once and retain the key only in the bounded computed cache.
                let prop_key = crate::atom::PropKey::lookup_text(&text);
                if let Some(value) =
                    self.ic_get_rederive(&object, &text, prop_key, &scratch[0], false)
                {
                    let young = self.gc_heap.is_young(key.cell_word());
                    self.computed_reads.insert(
                        key,
                        scratch[0].get(),
                        young,
                        self.gc_heap.markings(),
                    );
                    return self.finish_cached_get(value, base, None);
                }
            }
        }
        self.get_member_key(base, &text)
    }

    /// [[Get]](P, Receiver) of `object` for a String key P whose engine text is `name`, resolved
    /// as a named read's helper resolves it when every object on the chain is one the property
    /// ICs read: the key's atom (text with none names no property) probes the megamorphic stub
    /// cache, and a miss re-derives the read along the chain and fills it. A getter is called
    /// on `receiver`. `None` declines, having run no code, for an exotic or side-table object on
    /// the chain or a name the ICs leave to the general [[Get]] (index-like or NUL-led text,
    /// `length`, `description`); the caller then runs that [[Get]].
    pub(crate) fn get_named_cached(
        &mut self,
        object: &Gc,
        name: &str,
        receiver: &Value,
    ) -> Option<Result<Value, Abrupt>> {
        if name.len() > COMPUTED_STUB_KEY_MAX_BYTES
            || name == "length"
            || name == "description"
            || name
                .as_bytes()
                .first()
                .is_some_and(|&byte| byte == 0 || byte.is_ascii_digit())
        {
            return None;
        }
        let key = crate::atom::PropKey::lookup_text(name);
        let read = self.ic_read_by_key(object, name, key)?;
        Some(self.finish_cached_get(read, receiver, None))
    }

    /// The property ICs' resolution of the read of `key`, whose text is `name`, on `object`'s
    /// chain (see [`Interp::get_named_cached`]): the stub cache's entry for the receiver's
    /// shape, else a re-derivation that fills it. `None` for text with no atom proves the
    /// read absent along a chain of ordinary objects.
    fn ic_read_by_key(
        &self,
        object: &Gc,
        name: &str,
        key: Option<crate::atom::PropKey>,
    ) -> Option<CachedGet> {
        if let Some(key) = key {
            let shape = object.borrow().props.shape();
            if let Some(entry) = self.stub_cache_get(shape, key) {
                if entry.st.recv_shape == shape {
                    if let Some(read) =
                        self.ic_shape_probe(Gc::as_ptr(object), name, Some(key), entry.st)
                    {
                        return Some(read);
                    }
                }
            }
        }
        let scratch = [const { std::cell::Cell::new(crate::bytecode::IcState::EMPTY) };
            crate::bytecode::PROP_IC_WAYS];
        self.ic_get_rederive(object, name, key, &scratch[0], key.is_some())
    }

    fn get_prop_ic_impl(
        &mut self,
        base: &Value,
        name: &str,
        key: crate::atom::PropKey,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
        stable_name: bool,
        trace: Option<&mut crate::feedback::CurrentPropertyTrace>,
    ) -> Result<Value, Abrupt> {
        match &(base).view() {
            ValueView::Obj(o) => {
                if let Some(v) = self.try_ic_get(o, name, key, cache, stable_name) {
                    return self.finish_cached_get(v, base, trace);
                }
            }
            // Primitive receivers: named non-index reads (`"x".replace`, `(1).toFixed`) resolve
            // on the wrapper prototype chain — cache that walk too. `length` and numeric keys
            // have primitive-specific handling in get_member; symbols get description handling.
            ValueView::Str(_) | ValueView::Num(_) | ValueView::Bool(_)
                if name != "length"
                    && name != "description"
                    && !name.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) =>
            {
                let proto = match &(base).view() {
                    ValueView::Str(_) => self.string_proto,
                    ValueView::Num(_) => self.number_proto,
                    _ => self.boolean_proto,
                };
                if let Some(v) = self.try_ic_get(&proto, name, key, cache, stable_name) {
                    return self.finish_cached_get(v, base, trace);
                }
            }
            _ => {}
        }
        self.get_member_recv_impl(base, name, *base, trace, Some(key))
    }

    fn property_receiver_shape(&self, base: &Value) -> Option<u32> {
        let object = match (base).view() {
            ValueView::Obj(object) => Some(object),
            ValueView::Str(_) => Some(self.string_proto),
            ValueView::Num(_) => Some(self.number_proto),
            ValueView::Bool(_) => Some(self.boolean_proto),
            ValueView::Sym(_) => Some(self.symbol_proto),
            ValueView::BigInt(_) => self.extra_protos.get("BigInt").copied(),
            _ => None,
        }?;
        let shape = object.borrow().props.shape();
        Some(shape)
    }

    /// Whether object pointer `ptr` (holding `b`) reads like an ordinary object for IC purposes.
    /// `Exotic::Array` is allowed: array *named* reads are ordinary (`length` is a real stored
    /// property); the primitive wrappers are allowed as justified below; everything else exotic
    /// (arguments, …) and anything in a side table (proxy, namespace, typed array) is not.
    #[inline]
    fn ic_plain_ptr(&self, _ptr: usize, b: &Object) -> bool {
        // StrWrap (String.prototype, string wrapper objects): index and `length` reads are
        // exotic-provided and never live in `entries`, and StrWrap maps are NOT `elem_mode`,
        // so named entries stay shape-pinned — a named non-index IC over one is as sound as
        // over an ordinary object. NumWrap/BoolWrap (Number/Boolean prototypes and their
        // wrappers) are stricter still: no name is exotic-provided at all, and they are never
        // `elem_mode` either (`mark_array` fires only on becoming `Exotic::Array`).
        (matches!(
            b.exotic,
            Exotic::None
                | Exotic::Array
                | Exotic::StrWrap(_)
                | Exotic::NumWrap(_)
                | Exotic::BoolWrap(_)
        ) || b.exotic.is_collection())
            && b.ic_plain.get()
    }

    /// The `GetProp`/`GetMethod` inline-cache fast path; `None` means "take the slow path".
    ///
    /// The prototype chain is walked by raw pointer with no `Gc` clones: every object on it is
    /// transitively kept alive by `o` (which the caller holds) for the duration of this call, and
    /// nothing mutates the chain here, so the pointers stay valid.
    #[inline(always)]
    fn try_ic_get(
        &self,
        o: &Gc,
        name: &str,
        key: crate::atom::PropKey,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
        stable_name: bool,
    ) -> Option<CachedGet> {
        let head = Gc::as_ptr(o);
        // PROP_IC_WAYS ways per site (polymorphic receivers — subclass hierarchies rotating
        // through one access site — miss fewer ways constantly): probe all, then the global
        // stub cache (unbounded polymorphism), then re-derive into way 1 with the older ways
        // demoted.
        for k in 0..crate::bytecode::PROP_IC_WAYS {
            if let Some(v) = self.ic_shape_probe(head, name, Some(key), self.ic_way(cache, k).get())
            {
                return Some(v);
            }
        }
        if stable_name {
            let shape = unsafe { (*head).borrow().props.shape() };
            let Some(e) = self.stub_cache_get(shape, key) else {
                return self.ic_get_rederive(o, name, Some(key), cache, stable_name);
            };
            if e.st.recv_shape == shape {
                if let Some(v) = self.ic_shape_probe(head, name, Some(key), e.st) {
                    // Promote into the site (demoting the other ways): the next access hits
                    // way-first.
                    self.ic_insert(cache, e.st);
                    return Some(v);
                }
            }
        }
        self.ic_get_rederive(o, name, Some(key), cache, stable_name)
    }

    /// Cache way `k` of a property site (sites allocate `PROP_IC_WAYS` consecutive cells — see
    /// `Compiler::new_cache`, which keeps this pointer step in bounds).
    #[inline]
    pub(crate) fn ic_way<'a>(
        &self,
        cache: &'a std::cell::Cell<crate::bytecode::IcState>,
        k: usize,
    ) -> &'a std::cell::Cell<crate::bytecode::IcState> {
        debug_assert!(k < crate::bytecode::PROP_IC_WAYS);
        unsafe { &*(cache as *const std::cell::Cell<crate::bytecode::IcState>).add(k) }
    }

    /// Install `st` as way 1 of the site, demoting the existing ways one step (LRU-ish: a site
    /// cycling through up to `PROP_IC_WAYS` shapes keeps them all resident).
    #[inline]
    pub(crate) fn ic_insert(
        &self,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
        st: crate::bytecode::IcState,
    ) {
        if !st.has_cacheable_shapes() {
            return;
        }
        for k in (1..crate::bytecode::PROP_IC_WAYS).rev() {
            self.ic_way(cache, k).set(self.ic_way(cache, k - 1).get());
        }
        cache.set(st);
    }

    /// One way's shape fast path: a `depth == 0` (own), `depth == 1` (immediate-prototype, e.g.
    /// a method), or `depth == 2` (a subclass hierarchy hop — needs the recorded `mid_shape`)
    /// hit on non-exotic ordinary objects validates by shape-id compares alone — no per-hop
    /// key or hash checks. Shapes are shared across structurally-identical objects, so an id
    /// recorded from one instance validates the same slot on any sibling instance; a shape
    /// match on a non-holder hop proves it still *lacks* the name. Protos are re-followed
    /// live and every hop's shape re-checked, so a proto swap or hop mutation is caught.
    /// `None` = miss (no side effects). Accessors return an owned descriptor snapshot,
    /// never execute JavaScript during the borrowed chain walk.
    #[inline(always)]
    fn ic_shape_probe(
        &self,
        head: *const std::cell::RefCell<Object>,
        name: &str,
        key: Option<crate::atom::PropKey>,
        st: crate::bytecode::IcState,
    ) -> Option<CachedGet> {
        // A key-checked state compares the entry's key with the site's: by word, or by text
        // where the caller has looked no key up (a computed read).
        let is_key = |entry: crate::atom::PropKey| match key {
            Some(key) => entry == key,
            None => entry.is(name),
        };
        use crate::bytecode::{IC_ABSENT, IC_ACCESSOR, IC_ARR_KEYCHK, IC_EMPTY};
        let _ = IC_EMPTY;
        // Cached absence: re-walk the live chain, validating each level's shape (and the same
        // plainness/exotic gates a rederive would demand). All shapes matching proves the key
        // is still absent everywhere — the read is `undefined` with no entry scan at all.
        if st.depth == IC_ABSENT {
            let shapes = [
                st.recv_shape,
                st.mid_shape,
                st.mid2_shape,
                st.mid3_shape,
                st.mid4_shape,
                st.holder_shape,
            ];
            let levels = st.slot as usize;
            if levels == 0 || levels > shapes.len() {
                return None;
            }
            let mut cur = head;
            for (k, &shape) in shapes.iter().take(levels).enumerate() {
                let b = unsafe { (*cur).borrow() };
                // Absence-of-entry proves OrdinaryGet absence only where no name is
                // exotic-provided: `None` plus NumWrap/BoolWrap (never StrWrap's
                // `length`/indices, never Array elements), mirrored in the fill.
                if !((matches!(
                    b.exotic,
                    Exotic::None | Exotic::NumWrap(_) | Exotic::BoolWrap(_)
                ) || b.exotic.is_collection())
                    && b.ic_plain.get()
                    && b.props.shape() == shape)
                {
                    return None;
                }
                match b.proto.as_ref() {
                    Some(p) => cur = Gc::as_ptr(p),
                    None => {
                        return if k + 1 == levels {
                            Some(CachedGet::undefined())
                        } else {
                            None
                        }
                    }
                }
            }
            // Chain grew a level (a proto was attached where the fill saw the end).
            return None;
        }
        // Key-checked entries (array holder — see `IC_ARR_KEYCHK`): decode the flag off the
        // depth. The `< 0x80` range check filters `IC_CREATE`/`IC_EMPTY`.
        if st.depth >= 0x80 {
            return None;
        }
        let keychk = st.depth & IC_ARR_KEYCHK != 0;
        let accessor = st.depth & IC_ACCESSOR != 0;
        let depth = st.depth & !(IC_ARR_KEYCHK | IC_ACCESSOR);
        let required_mids = (1u8 << depth.saturating_sub(1).min(7)) - 1;
        if depth > crate::bytecode::IC_MAX_DEPTH || st.mid_ok & required_mids != required_mids {
            return None;
        }
        unsafe {
            let rb = (*head).borrow();
            // An Array receiver may shape-validate only as a NON-holder (depth ≥ 1): its
            // shape proves which NAMED keys it has — so a non-digit name is provably absent
            // (element keys are all canonical indices) — but NOT where a named entry sits
            // (element entries occupy slots without transitioning the shape, so two
            // same-shape arrays can hold `length` at different slots).
            let non_digit = !name.as_bytes().first().is_some_and(|b| b.is_ascii_digit());
            let recv_shape_ok = rb.exotic.is_ordinary()
                || (matches!(rb.exotic, Exotic::Array) && (depth >= 1 || keychk) && non_digit)
                // Wrapper named entries are shape-pinned (not elem_mode); StrWrap index/length
                // reads are exotic-provided and never fill states at this site's name, and the
                // digit gate is kept for Num/Bool wrappers too, mirroring StrWrap.
                || (matches!(
                    rb.exotic,
                    Exotic::StrWrap(_) | Exotic::NumWrap(_) | Exotic::BoolWrap(_)
                ) && non_digit);
            if recv_shape_ok && rb.ic_plain.get() && rb.props.shape() == st.recv_shape {
                if depth == 0 {
                    let property = if keychk {
                        rb.props
                            .entry_at(st.slot as usize)
                            .and_then(|(entry, p)| is_key(entry).then_some(p))
                    } else {
                        rb.props.property_at(st.slot as usize)
                    };
                    if let Some(p) = property {
                        // Shapes encode descriptor class and every production mutation
                        // path demotes on class change, so a shape-matched slot's kind
                        // is the recorded one; only key-checked (array-holder) states,
                        // whose slot is not shape-proven, still compare live.
                        if p.accessor() == accessor || !keychk {
                            return self.snapshot_cached_get(
                                p,
                                name,
                                rb.props.shape(),
                                depth,
                                keychk,
                            );
                        }
                    }
                } else if let Some(pr) = rb.proto.as_ref() {
                    let mut hp = Gc::as_ptr(pr);
                    drop(rb); // the next hop is a different object; release the borrow
                              // Validate each intermediate hop's shape (a match proves it still lacks
                              // the name), following live protos.
                    for mid_shape in [st.mid_shape, st.mid2_shape, st.mid3_shape, st.mid4_shape]
                        .into_iter()
                        .take(depth.saturating_sub(1) as usize)
                    {
                        let mb = (*hp).borrow();
                        if !((matches!(
                            mb.exotic,
                            Exotic::None
                                | Exotic::StrWrap(_)
                                | Exotic::NumWrap(_)
                                | Exotic::BoolWrap(_)
                        ) || mb.exotic.is_collection())
                            && mb.ic_plain.get()
                            && mb.props.shape() == mid_shape)
                        {
                            return None;
                        }
                        hp = Gc::as_ptr(mb.proto.as_ref()?);
                    }
                    let hb = (*hp).borrow();
                    let holder_exotic_ok = (matches!(
                        hb.exotic,
                        Exotic::None
                            | Exotic::StrWrap(_)
                            | Exotic::NumWrap(_)
                            | Exotic::BoolWrap(_)
                    ) || hb.exotic.is_collection())
                        || (keychk && matches!(hb.exotic, Exotic::Array));
                    if holder_exotic_ok && hb.ic_plain.get() && hb.props.shape() == st.holder_shape
                    {
                        let property = if keychk {
                            hb.props
                                .entry_at(st.slot as usize)
                                .and_then(|(entry, p)| is_key(entry).then_some(p))
                        } else {
                            hb.props.property_at(st.slot as usize)
                        };
                        if let Some(p) = property {
                            if p.accessor() == accessor || !keychk {
                                return self.snapshot_cached_get(
                                    p,
                                    name,
                                    hb.props.shape(),
                                    depth,
                                    keychk,
                                );
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// [`try_ic_get`]'s miss path: re-derive the resolution by a raw-pointer chain walk and
    /// refill the cache. Split out so the shape fast path's early bail-outs can tail into it.
    /// `key` is the name's key, none where text with no atom names no property.
    fn ic_get_rederive(
        &self,
        o: &Gc,
        name: &str,
        key: Option<crate::atom::PropKey>,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
        stable_name: bool,
    ) -> Option<CachedGet> {
        use crate::bytecode::{IcState, IC_MAX_DEPTH};
        type ObjCell = std::cell::RefCell<Object>;
        let head = Gc::as_ptr(o);
        // Re-derive: walk the chain by raw pointer (every object is kept alive transitively by
        // `o`, nothing mutates during the read). On an ordinary descriptor within reach record
        // (depth, slot, receiver shape, holder shape); a non-plain level defers to full
        // `[[Get]]`. An ordinary accessor is invoked only after this lookup returns.
        let recv_shape = unsafe { (*head).borrow().props.shape() };
        let mut cur: *const ObjCell = head;
        // The depth-1 hop's shape, recorded when that hop is a plain (non-array) ordinary object:
        // a depth-2 fill stores it as `mid_shape` so later hits can shape-validate the whole
        // two-hop chain. An Array mid hop stays ineligible (its shape doesn't track elements, so
        // a shape match couldn't prove an index-like name is still absent).
        let mut mids = [None; 4];
        // Level shapes for an absent-property fill (see `IC_ABSENT`): valid only while every
        // level so far answers no name outside its entries — plain `Exotic::None` objects and
        // NumWrap/BoolWrap wrappers. StrWrap (`length`/indices) and Array (elements) answer
        // names exotically, so absence-of-entry proves nothing across them.
        let mut absent_shapes = [0; 6];
        let mut absent_ok = true;
        unsafe {
            for depth in 0..=IC_MAX_DEPTH {
                let b = (*cur).borrow();
                if !self.ic_plain_ptr(cur as usize, &b) {
                    return None;
                }
                absent_shapes[depth as usize] = b.props.shape();
                absent_ok = absent_ok
                    && (matches!(
                        b.exotic,
                        Exotic::None | Exotic::NumWrap(_) | Exotic::BoolWrap(_)
                    ) || b.exotic.is_collection());
                if let Some(slot) = key.and_then(|key| b.props.slot_of_key(key)) {
                    let p = b.props.property_at(slot).unwrap();
                    let v = self.snapshot_cached_get(
                        p,
                        name,
                        b.props.shape(),
                        depth,
                        matches!(b.exotic, Exotic::Array),
                    )?;
                    // An Array HOLDER's slot can't be trusted by shape alone: element entries
                    // occupy slots without transitioning the shape, so two same-shape arrays
                    // map the name at DIFFERENT slots. Cache it as a key-checked entry
                    // (`IC_ARR_KEYCHK`): hits re-verify the entry's key, and the JIT templates
                    // (exact depth compares, no bounds check) route it to the helper. Digit
                    // names stay uncached (element reads have their own paths).
                    let keychk = !(matches!(
                        b.exotic,
                        Exotic::None
                            | Exotic::StrWrap(_)
                            | Exotic::NumWrap(_)
                            | Exotic::BoolWrap(_)
                    ) || b.exotic.is_collection());
                    if keychk && name.as_bytes().first().is_some_and(|b| b.is_ascii_digit()) {
                        return Some(v);
                    }
                    let mid_ok = mids
                        .iter()
                        .take(depth.saturating_sub(1) as usize)
                        .enumerate()
                        .fold(0, |mask, (index, shape)| {
                            mask | ((shape.is_some() as u8) << index)
                        });
                    // Demote the previous ways before refilling: a site rotating through up
                    // to PROP_IC_WAYS shapes stabilizes with one shape per way instead of
                    // thrashing a single cell.
                    let st = IcState {
                        depth: depth
                            | if p.accessor() {
                                crate::bytecode::IC_ACCESSOR
                            } else {
                                0
                            }
                            | if keychk {
                                crate::bytecode::IC_ARR_KEYCHK
                            } else {
                                0
                            },
                        slot: slot as u32,
                        recv_shape,
                        holder_shape: b.props.shape(),
                        mid_ok,
                        mid_shape: mids[0].unwrap_or(0),
                        mid2_shape: mids[1].unwrap_or(0),
                        mid3_shape: mids[2].unwrap_or(0),
                        mid4_shape: mids[3].unwrap_or(0),
                    };
                    self.ic_insert(cache, st);
                    // Mirror into the owned-text stub cache so OTHER shapes rotating through
                    // this site don't evict this resolution for good.
                    if let Some(key) = key.filter(|_| stable_name) {
                        self.stub_cache_set(recv_shape, key, st);
                    }
                    return Some(v);
                }
                if (1..=4).contains(&depth)
                    && (matches!(
                        b.exotic,
                        Exotic::None
                            | Exotic::StrWrap(_)
                            | Exotic::NumWrap(_)
                            | Exotic::BoolWrap(_)
                    ) || b.exotic.is_collection())
                {
                    mids[depth as usize - 1] = Some(b.props.shape());
                }
                match b.proto.as_ref() {
                    Some(p) => cur = Gc::as_ptr(p),
                    None => {
                        // Chain ended: the property is absent. Cache that (the cheapest read
                        // there is) when every level answered no name outside its entries
                        // and the chain fits the recorded shape fields. OrdinaryGet must observe
                        // additions and prototype swaps at every level, including deep bases.
                        let levels = depth as usize + 1;
                        if absent_ok && levels <= absent_shapes.len() {
                            let st = IcState {
                                depth: crate::bytecode::IC_ABSENT,
                                slot: levels as u32,
                                recv_shape: absent_shapes[0],
                                holder_shape: absent_shapes[5],
                                mid_ok: 0,
                                mid_shape: absent_shapes[1],
                                mid2_shape: absent_shapes[2],
                                mid3_shape: absent_shapes[3],
                                mid4_shape: absent_shapes[4],
                            };
                            self.ic_insert(cache, st);
                            if let Some(key) = key.filter(|_| stable_name) {
                                self.stub_cache_set(recv_shape, key, st);
                            }
                            return Some(CachedGet::undefined());
                        }
                        return None;
                    }
                }
            }
        }
        None
    }

    /// Diagnostic (`TRUST_JS_JIT_PROPSTAT`): classify why a named read that reached a JIT
    /// property helper was not served by the inline template. Read-only: it walks live
    /// descriptors like `ic_get_rederive` but fills nothing and runs no JavaScript, so it
    /// observes OrdinaryGet's resolution (ECMA-262 e28783d5, #sec-ordinaryget) without
    /// performing it. The buckets mirror the template guards in `emit_prop_way_probe`.
    pub(crate) fn propstat_classify(
        &self,
        base: &Value,
        name: &str,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> &'static str {
        let ValueView::Obj(o) = &(base).view() else {
            return match &(base).view() {
                ValueView::Str(_) | ValueView::Num(_) | ValueView::Bool(_) => "receiver: primitive",
                _ => "receiver: non-object",
            };
        };
        if self.is_proxy_value(base) {
            return "receiver: proxy";
        }
        let head = Gc::as_ptr(o);
        let recv_shape = match unsafe { (*head).try_borrow() } {
            Ok(b) => b.props.shape(),
            Err(_) => return "receiver: borrowed (unclassified)",
        };
        let cached = (0..crate::bytecode::PROP_IC_WAYS).any(|k| {
            let st = self.ic_way(cache, k).get();
            st.depth != crate::bytecode::IC_EMPTY && st.recv_shape == recv_shape
        });
        let mut cur = head;
        for depth in 0..=8u32 {
            let b = match unsafe { (*cur).try_borrow() } {
                Ok(b) => b,
                Err(_) => return "chain: borrowed (unclassified)",
            };
            if !self.ic_plain_ptr(cur as usize, &b) {
                return if depth == 0 {
                    "receiver: exotic/side-table"
                } else {
                    "chain: non-plain hop"
                };
            }
            if let Some(slot) = b.props.slot_of(name) {
                let Some(p) = b.props.property_at(slot) else {
                    return "holder: slot vanished";
                };
                if p.accessor() {
                    return "accessor descriptor";
                }
                if matches!(b.exotic, Exotic::Array) {
                    return "array holder (key-checked)";
                }
                if depth > 3 {
                    return "data at depth > 3";
                }
                return if cached {
                    "data cached (other template guard)"
                } else {
                    "data uncached (shape churn)"
                };
            }
            match b.proto.as_ref() {
                Some(p) => cur = Gc::as_ptr(p),
                None => {
                    return if cached {
                        "absent cached (other template guard)"
                    } else {
                        "absent uncached"
                    }
                }
            }
        }
        "chain: deeper than 8"
    }

    /// `obj.name = v` write with a per-site inline cache (bytecode `SetProp`/`SetPropDrop`). The
    /// fast path overwrites an existing own writable data property (OrdinarySet's winning case,
    /// correct regardless of the prototype chain); everything else defers to full `[[Set]]`.
    pub(crate) fn set_prop_ic(
        &mut self,
        base: &Value,
        name: &Rc<str>,
        key: crate::atom::PropKey,
        v: Value,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> Result<(), Abrupt> {
        if let ValueView::Obj(o) = &(base).view() {
            if self.try_ic_set(o, key, &v, cache) {
                return Ok(());
            }
        }
        self.set_member_key(base, name, v)
    }

    /// Profile-enabled form of [`Self::set_prop_ic`]. Ordinary IC hits are retained by the
    /// existing adapter; slow accessor/exotic/rejection paths annotate the canonical [[Set]]
    /// branch from ECMA-262 §10.1.9.1-2 / §10.5.9 without replaying any observable operation.
    pub(crate) fn set_prop_ic_profiled(
        &mut self,
        base: &Value,
        name: &Rc<str>,
        key: crate::atom::PropKey,
        v: Value,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
        trace: &mut crate::feedback::CurrentPropertyTrace,
    ) -> Result<(), Abrupt> {
        trace.receiver_shape = self.property_receiver_shape(base);
        if let ValueView::Obj(o) = &(base).view() {
            if self.try_ic_set(o, key, &v, cache) {
                return Ok(());
            }
        }
        let strict = self.strict;
        let success = self.set_member_recv_impl(base, name, v, *base, Some(trace))?;
        self.finish_assignment_set(name, strict, success)
    }

    /// The `SetProp` inline-cache fast path; `false` means "take the slow path". Writes cache
    /// own slots only (`depth` 0): an own writable data property wins OrdinarySet regardless of
    /// the prototype chain.
    /// The fused `obj.k += v` fast path (`bytecode::Op::AppendProp`): when the property still
    /// holds the exact string the compound read produced (`lval` — nothing replaced it while the
    /// RHS evaluated), both sides are strings, and the slot is an own writable data property on a
    /// plain object, append **in place**: an accumulated string ends at its buffer's frontier,
    /// so `LStr::concat_owned` writes past it (or rebuilds once with doubled capacity) — the
    /// amortized-O(append) accumulator loop, instead of copying the whole accumulation per step.
    /// `Err(lval)` = not applicable, nothing touched, caller runs the generic Add + set.
    pub(crate) fn append_prop_fast(
        &mut self,
        o: &Gc,
        name: &str,
        lval: Value,
        x: &crate::lstr::LStr,
    ) -> Result<(), Value> {
        let ValueView::Str(lv) = &(lval).view() else {
            return Err(lval);
        };
        if lv.unit_len() + x.unit_len() > MAX_STR_LEN {
            return Err(lval); // the generic path raises the RangeError
        }
        let mut b = o.borrow_mut();
        if !b.ic_plain.get() || !b.exotic.is_ordinary() {
            return Err(lval);
        }
        let Some(p) = b.props.get(name) else {
            return Err(lval);
        };
        if p.accessor() || !p.writable() {
            return Err(lval);
        }
        let same = matches!((p.value()).view(), ValueView::Str(cur) if crate::lstr::LStr::ptr_eq(&cur, lv));
        if !same {
            return Err(lval);
        }
        // Take the slot's handle and append to it. The object stays borrowed throughout, so
        // the checked descriptor cannot change between the read above and the value-only
        // take/store below.
        let cur = match (b.props.take_value(name)).map(|__value| (__value).view()) {
            Some(ValueView::Str(s)) => s,
            _ => unreachable!("checked above"),
        };
        drop(lval);
        b.props.set_value(name, Value::Str(cur.concat_owned(x)));
        Ok(())
    }

    fn try_ic_set(
        &mut self,
        o: &Gc,
        key: crate::atom::PropKey,
        v: &Value,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> bool {
        let mut b = o.borrow_mut();
        if !b.exotic.is_ordinary() {
            return false;
        }
        if !b.ic_plain.get() {
            return false;
        }
        // Shape fast path (own writable data property), every way: a shape match means `slot`
        // still maps `name`; skip the key compare. `accessor`/`writable` are re-checked (an
        // in-place defineProperty could have flipped them without changing the shape).
        for k in 0..crate::bytecode::PROP_IC_WAYS {
            let st = self.ic_way(cache, k).get();
            if st.depth == 0 && b.props.shape() == st.recv_shape {
                if let Some(p) = b.props.property_at(st.slot as usize) {
                    if !p.accessor() && p.writable() {
                        b.props.set_value_at(st.slot as usize, *v);
                        return true;
                    }
                }
            }
        }
        // Stub-cache probe (shared with the get path — a depth-0 entry means "this shape maps
        // this name at this slot", and accessor/writable are re-checked live either way): a
        // store site rotating through more shapes than its ways still resolves without the
        // hashed existence scan below. Promote into way 1 like the get path does.
        {
            let shape = b.props.shape();
            if let Some(e) = self.stub_cache_get(shape, key) {
                if e.st.recv_shape == shape && e.st.depth == 0 {
                    if let Some(p) = b.props.property_at(e.st.slot as usize) {
                        if !p.accessor() && p.writable() {
                            b.props.set_value_at(e.st.slot as usize, *v);
                            self.ic_insert(cache, e.st);
                            return true;
                        }
                    }
                }
            }
        }
        // Creation fast path (see IC_CREATE), every polymorphic way: a shared base constructor
        // commonly initializes subclass instances that have the same empty shape but distinct
        // prototypes. Shape + epoch + prototype identity re-prove the fill-time chain walk, and
        // the recorded child shape makes the insert a plain append.
        if b.extensible {
            let shape = b.props.shape();
            let epoch = crate::value::proto_epoch();
            let proto_ptr = b.proto.as_ref().map_or(0, |p| Gc::as_ptr(p) as usize);
            for way in 0..crate::bytecode::PROP_IC_WAYS {
                let st = self.ic_way(cache, way).get();
                if st.depth == crate::bytecode::IC_CREATE
                    && st.recv_shape == shape
                    && st.mid_shape == epoch
                    && epoch != u32::MAX
                {
                    let cached_proto = (((st.mid2_shape as u64) << 32) | st.slot as u64) as usize;
                    if cached_proto == proto_ptr {
                        b.props
                            .append_new(key, Property::plain(*v), st.holder_shape);
                        return true;
                    }
                }
            }
        }
        match b.props.slot_of_key(key) {
            Some(slot) => {
                let p = b.props.property_at(slot).unwrap();
                if p.accessor() || !p.writable() {
                    return false; // setter, or non-writable (strict-throw) — slow path
                }
                let shape = b.props.shape();
                b.props.set_value_at(slot, *v);
                let st = crate::bytecode::IcState {
                    depth: 0,
                    slot: slot as u32,
                    recv_shape: shape,
                    holder_shape: shape,
                    mid_ok: 0,
                    mid_shape: 0,
                    mid2_shape: 0,
                    mid3_shape: 0,
                    mid4_shape: 0,
                };
                self.ic_insert(cache, st);
                // Mirror into the stub cache (see the probe above).
                self.stub_cache_set(shape, key, st);
                true
            }
            None => self.try_ic_create(o, b, key, v, cache),
        }
    }

    /// Property *creation* through the set IC (see [`crate::bytecode::IC_CREATE`]): the hot
    /// constructor pattern `this.x = v` inserts a fresh own property on every call. A cache hit
    /// (same before-shape + same prototype identity + unchanged proto epoch + extensible)
    /// re-proves the fill-time chain walk, so the insert skips `OrdinarySet` entirely. `b` is the
    /// receiver's live borrow (the receiver is already known ordinary, `Exotic::None`).
    fn try_ic_create(
        &mut self,
        _o: &Gc,
        mut b: std::cell::RefMut<'_, crate::value::Object>,
        key: crate::atom::PropKey,
        v: &Value,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> bool {
        use crate::bytecode::{IcState, IC_CREATE};
        // Index-like names belong to the dense-element machinery — generic path.
        if key.starts_with_digit() {
            return false;
        }
        if !b.extensible {
            return false;
        }
        let epoch = crate::value::proto_epoch();
        // A saturated epoch can never validate a hit — filling would only waste walks.
        if epoch == u32::MAX {
            return false;
        }
        let proto_ptr = b.proto.as_ref().map_or(0, |p| Gc::as_ptr(p) as usize);
        // Bound the pin map for cache-churn-heavy embedders (per-request eval): existing sites
        // keep hitting; new sites just stop caching once the budget is spent.
        if self.creation_pins.len() >= 65536 && !self.creation_pins.contains_key(&proto_ptr) {
            return false;
        }
        // Fill: prove the whole chain compatible with creating an own property. An inherited
        // writable data property is compatible too: OrdinarySet creates on the receiver in that
        // case. Accessors and non-writable data still reject. Each visited hop is marked as a
        // live prototype so any later structural/descriptor change bumps the epoch; proto swaps
        // and defineProperty bump it globally.
        let recv_shape = b.props.shape();
        let mut cur = b.proto;
        {
            let mut walk = cur;
            let mut hops = 0;
            while let Some(h) = walk {
                hops += 1;
                if hops > 8 {
                    return false;
                }
                let hb = h.borrow();
                if !(hb.exotic.is_ordinary() && hb.ic_plain.get()) {
                    return false;
                }
                if let Some(slot) = hb.props.slot_of_key(key) {
                    let inherited = hb.props.property_at(slot).unwrap();
                    if inherited.accessor() || !inherited.writable() {
                        return false;
                    }
                    hb.props.mark_proto();
                    break;
                }
                hb.props.mark_proto();
                let next = hb.proto;
                drop(hb);
                walk = next;
            }
        }
        b.props.insert(key, Property::plain(*v));
        self.ic_insert(
            cache,
            IcState {
                depth: IC_CREATE,
                slot: proto_ptr as u32,
                recv_shape,
                // The memoized child shape the insert just landed on: hits append straight to it.
                holder_shape: b.props.shape(),
                mid_ok: 0,
                mid_shape: crate::value::proto_epoch(),
                mid2_shape: ((proto_ptr as u64) >> 32) as u32,
                mid3_shape: 0,
                mid4_shape: 0,
            },
        );
        let pin = cur.take().map(|p| Gc::downgrade(&p)).unwrap_or_default();
        self.creation_pins.insert(proto_ptr, pin);
        true
    }

    /// Whether object pointer `ptr` reads/writes like an ordinary object — i.e. it is not
    /// registered in any of the exotic side tables (proxy, module namespace, typed array,
    /// deferred namespace). Cheap: each empty table is skipped without hashing, which is the
    /// common case in a hot loop.
    #[inline]
    pub(crate) fn ordinary_get_ptr(&self, ptr: usize) -> bool {
        if Self::object_ic_plain(ptr) {
            debug_assert!(self.side_table_ordinary_get_ptr(ptr));
            return true;
        }
        self.side_table_ordinary_get_ptr(ptr)
    }

    #[inline]
    fn side_table_ordinary_get_ptr(&self, ptr: usize) -> bool {
        self.typed_array_at(ptr).is_none() && self.side_table_non_typed_ordinary_get_ptr(ptr)
    }

    /// An object whose [[Get]]/[[Set]] behavior lives in an interpreter side table (Proxy,
    /// typed array, module namespace, Web IDL indexed host object) always has its `ic_plain`
    /// byte cleared at registration, and that byte is never set again. A set byte therefore
    /// proves the object is in none of those tables with one load instead of several hash
    /// probes. A clear byte (including an object currently borrowed mutably) is only a hint.
    ///
    /// `ptr` must be the address of a live object (`Gc::as_ptr` of a held `Gc`).
    #[inline(always)]
    fn object_ic_plain(ptr: usize) -> bool {
        let cell = unsafe { &*(ptr as *const RefCell<crate::value::Object>) };
        // The shared reference ends before this returns, and a mutable borrow elsewhere makes
        // this conservatively report "not proven plain".
        matches!(unsafe { cell.try_borrow_unguarded() }, Ok(object) if object.ic_plain.get())
    }

    #[inline]
    fn non_typed_ordinary_get_ptr(&self, ptr: usize) -> bool {
        if Self::object_ic_plain(ptr) {
            debug_assert!(self.side_table_non_typed_ordinary_get_ptr(ptr));
            return true;
        }
        self.side_table_non_typed_ordinary_get_ptr(ptr)
    }

    #[inline]
    fn side_table_non_typed_ordinary_get_ptr(&self, ptr: usize) -> bool {
        if !self.single_interpreter() {
            return self.agent_side_table_non_typed_ordinary_get_ptr(ptr);
        }
        (self.proxies.is_empty() || !self.proxies.contains_key(&ptr))
            && (self.module_ns.is_empty() || !self.module_ns.contains_key(&ptr))
            && (self.deferred_ns.is_empty() || !self.deferred_ns.contains_key(&ptr))
            && (self.host_indexed.is_empty() || !self.host_indexed.contains_key(&ptr))
    }

    /// The same probe when another interpreter may own the side-table entry.
    #[inline(never)]
    fn agent_side_table_non_typed_ordinary_get_ptr(&self, ptr: usize) -> bool {
        self.proxy_at(ptr).is_none()
            && !self.is_namespace(ptr)
            && !self.is_deferred_namespace(ptr)
            && self
                .metadata_owner(ptr, |interp, key| interp.host_indexed.contains_key(&key))
                .is_none()
    }

    /// Number of contiguous Web IDL supported property indices on `object`, if installed.
    #[inline(always)]
    pub(crate) fn host_indexed_len(&self, object: &Gc) -> Option<u32> {
        if self.host_indexed.is_empty() {
            return None;
        }
        self.host_indexed
            .get(&(Gc::as_ptr(object) as usize))
            .map(|properties| {
                properties.live.as_ref().map_or(properties.length, |live| {
                    self.with_host_query(|interp| (live.length)(interp, &live.state))
                })
            })
    }

    /// Whether `key` is one of Web IDL's array-index property names on an indexed host object.
    /// This deliberately includes unsupported indices: read-only collections reject defining any
    /// array-index property, not only the indices currently exposed by the getter.
    #[inline(always)]
    pub(crate) fn host_indexed_array_key(&self, object: &Gc, key: &str) -> bool {
        !self.host_indexed.is_empty()
            && self
                .host_indexed
                .contains_key(&(Gc::as_ptr(object) as usize))
            && crate::value::canonical_index(key).is_some()
    }

    /// Run Web IDL's indexed property getter for a supported own index. `Ok(None)` means the key
    /// is not a supported indexed own property (including an out-of-range array-index key).
    pub(crate) fn host_indexed_own_value(
        &mut self,
        object: &Gc,
        key: &str,
    ) -> Result<Option<Value>, Abrupt> {
        // Installing platform-object internal methods clears this one-way
        // ordinary-object proof. Avoid side-table/name probes for ordinary
        // receivers, including every hop of their common property paths.
        if object.borrow().ic_plain.get() {
            return Ok(None);
        }
        let Some(index) = crate::value::canonical_index(key) else {
            return self.host_named_own_value(object, key);
        };
        let ptr = Gc::as_ptr(object) as usize;
        let Some(properties) = self.host_indexed.get(&ptr) else {
            return Ok(None);
        };
        if index >= self.host_indexed_len(object).unwrap_or(0) {
            return Ok(None);
        }
        let getter = properties.getter;
        self.call_callback(getter, Value::Obj(*object), &[Value::Num(index as f64)])
            .map(Some)
    }

    /// [[Get]](P, Receiver): like [`Self::get_member_key`] but with an explicit `receiver` — the `this` a
    /// getter is invoked with, the proxy `get` trap's Receiver argument, and what a forwarded
    /// `[[Get]]` carries through a proxy target chain.
    /// The deferred-namespace evaluation trigger: most operations with a string key (and all
    /// key-less ones like [[OwnPropertyKeys]]) evaluate the module; symbol keys and the "then"
    /// property never do. Accessing a namespace whose module is mid-evaluation is a TypeError.
    /// The proxy record for `ptr`, skipping the table probe entirely when no proxies exist
    /// (every prototype-chain hop asks).
    #[inline(always)]
    pub(crate) fn proxy_at(&self, ptr: usize) -> Option<(Value, Value)> {
        if !self.proxies.is_empty() {
            if let Some(proxy) = self.proxies.get(&ptr) {
                return Some(*proxy);
            }
        }
        let owner =
            self.foreign_metadata_owner(ptr, |interp, key| interp.proxies.contains_key(&key))?;
        self.with_realm_view(owner, |interp| interp.proxies.get(&ptr).cloned())
    }

    /// The TypedArray state of `ptr`, skipping the table probe when no TypedArray exists: the
    /// [[Get]]/[[Set]] slow paths ask for every object.
    #[inline(always)]
    pub(crate) fn typed_array_at(&self, ptr: usize) -> Option<TaInfo> {
        if !self.typed_arrays.is_empty() {
            if let Some(info) = self.typed_arrays.get(&ptr) {
                return Some(*info);
            }
        }
        let owner =
            self.foreign_metadata_owner(ptr, |interp, key| interp.typed_arrays.contains_key(&key))?;
        self.with_realm_view(owner, |interp| interp.typed_arrays.get(&ptr).copied())
    }

    #[inline(always)]
    pub(crate) fn defer_trigger(&mut self, o: &Gc, key: Option<&str>) -> Result<(), Abrupt> {
        if self.deferred_ns.is_empty() && self.single_interpreter() {
            return Ok(());
        }
        self.defer_trigger_slow(o, key)
    }

    fn defer_trigger_slow(&mut self, o: &Gc, key: Option<&str>) -> Result<(), Abrupt> {
        if matches!(key, Some(k) if Interp::is_sym_key(k) || k == "then") {
            return Ok(());
        }
        let ptr = Gc::as_ptr(o) as usize;
        let Some(owner) =
            self.metadata_owner(ptr, |interp, ptr| interp.deferred_ns.contains_key(&ptr))
        else {
            return Ok(());
        };
        let module_key = self.with_realm_view(owner, |interp| interp.deferred_ns[&ptr].clone());
        self.with_route(owner, |owner| owner.evaluate_deferred(&module_key))
    }

    pub(crate) fn get_member_recv(
        &mut self,
        base: &Value,
        key: &str,
        receiver: Value,
    ) -> Result<Value, Abrupt> {
        self.get_member_recv_impl(base, key, receiver, None, None)
    }

    /// `atom` is `key`'s key where the caller has it (see [`Interp::get_from_chain`]).
    fn get_member_recv_impl(
        &mut self,
        base: &Value,
        key: &str,
        receiver: Value,
        mut trace: Option<&mut crate::feedback::CurrentPropertyTrace>,
        atom: Option<crate::atom::PropKey>,
    ) -> Result<Value, Abrupt> {
        // An `import defer` namespace evaluates its module on string-keyed access.
        if let ValueView::Obj(o) = &(base).view() {
            if self.is_deferred_namespace(Gc::as_ptr(o) as usize) {
                let shape = o.borrow().props.shape();
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Exotic,
                        Some(shape),
                        0,
                        None,
                    );
                }
            }
            self.defer_trigger(o, Some(key))?;
        }
        match &(base).view() {
            ValueView::Undefined | ValueView::Empty | ValueView::Null => {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(crate::feedback::PropertyOutcome::Rejected, None, 0, None);
                }
                Err(self.throw(
                    "TypeError",
                    format!(
                        "cannot read property '{}' of {}",
                        crate::value::decode_string_key(key),
                        type_name(base)
                    ),
                ))
            }
            ValueView::Str(s) => {
                // `length` and indices are in UTF-16 code units (cached — a scanner loop reads
                // them per character of a megabytes-long source).
                if key == "length" {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(crate::feedback::PropertyOutcome::Exotic, None, 0, None);
                    }
                    let s = *s;
                    return Ok(Value::Num(self.str_len(&s) as f64));
                }
                // StringGetOwnProperty: only a canonical numeric index below the length is own
                // ("01", "+1" and indices past the end are read from the prototype chain).
                if let Some(i) = crate::value::canonical_index(key) {
                    let s = *s;
                    if let Some(u) = self.unit_at(&s, i as usize) {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(crate::feedback::PropertyOutcome::Exotic, None, 0, None);
                        }
                        return Ok(Value::Str(crate::jstr::unit_lstr(u)));
                    }
                }
                let proto = self.string_proto;
                self.get_from_chain(&proto, key, &receiver, trace, atom)
            }
            ValueView::Num(_) => {
                let proto = self.number_proto;
                self.get_from_chain(&proto, key, &receiver, trace, atom)
            }
            ValueView::Bool(_) => {
                let proto = self.boolean_proto;
                self.get_from_chain(&proto, key, &receiver, trace, atom)
            }
            ValueView::Sym(s) => {
                if key == "description" {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(crate::feedback::PropertyOutcome::Exotic, None, 0, None);
                    }
                    return Ok(s
                        .description
                        .clone()
                        .map(|k| Value::Str(k.into()))
                        .unwrap_or(Value::Undefined));
                }
                let proto = self.symbol_proto;
                self.get_from_chain(&proto, key, &receiver, trace, atom)
            }
            ValueView::BigInt(_) => match self.extra_protos.get("BigInt").cloned() {
                Some(proto) => self.get_from_chain(&proto, key, &receiver, trace, atom),
                None => {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(crate::feedback::PropertyOutcome::Absent, None, 0, None);
                    }
                    Ok(Value::Undefined)
                }
            },
            ValueView::Obj(o) => {
                let o = *o;
                let ptr = Gc::as_ptr(&o) as usize;
                // A mapped `arguments` index aliases its parameter binding.
                if let Some(v) = self.mapped_arg_value(ptr, key) {
                    let shape = o.borrow().props.shape();
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(shape),
                            0,
                            None,
                        );
                    }
                    return Ok(v);
                }
                // String wrapper (`new String(...)`/`Object("...")`): own indexed chars + `length`.
                let wrapped_string = match &o.borrow().exotic {
                    Exotic::StrWrap(s) => Some(**s),
                    _ => None,
                };
                if let Some(s) = wrapped_string {
                    if key == "length" {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(
                                crate::feedback::PropertyOutcome::Exotic,
                                Some(o.borrow().props.shape()),
                                0,
                                None,
                            );
                        }
                        return Ok(Value::Num(self.str_len(&s) as f64));
                    }
                    if let Some(i) = crate::value::canonical_index(key) {
                        if let Some(u) = self.unit_at(&s, i as usize) {
                            if let Some(trace) = trace.as_deref_mut() {
                                trace.record(
                                    crate::feedback::PropertyOutcome::Exotic,
                                    Some(o.borrow().props.shape()),
                                    0,
                                    None,
                                );
                            }
                            return Ok(Value::Str(crate::jstr::unit_lstr(u)));
                        }
                    }
                }
                // Copy only the actual live binding descriptor from its metadata owner.
                // Binding reads and their errors still belong to this executing algorithm.
                if let Some(property) = self.namespace_own_property(ptr, key) {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(o.borrow().props.shape()),
                            0,
                            None,
                        );
                    }
                    return property.map(|property| property.value());
                }
                // Proxy: invoke the `get` trap, or forward to the target.
                {
                    if let Some((target, handler)) = self.proxy_at(ptr) {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(
                                crate::feedback::PropertyOutcome::Exotic,
                                Some(o.borrow().props.shape()),
                                0,
                                None,
                            );
                        }
                        return self.proxy_get(target, handler, key, None, receiver);
                    }
                }
                // TypedArray integer-index reads come from the backing buffer, not the property map.
                // length/byteLength/byteOffset are computed (and 0 once the buffer is detached).
                if let Some(info) = self.typed_array_at(ptr) {
                    match self.ta_index_kind(&info, key) {
                        TaIndex::Element(idx) => {
                            if let Some(trace) = trace.as_deref_mut() {
                                trace.record(
                                    crate::feedback::PropertyOutcome::Exotic,
                                    Some(o.borrow().props.shape()),
                                    0,
                                    None,
                                );
                            }
                            return Ok(self.ta_read(&info, idx));
                        }
                        // A canonical-numeric non-index never reaches the prototype: it reads undefined.
                        TaIndex::Exotic => {
                            if let Some(trace) = trace.as_deref_mut() {
                                trace.record(
                                    crate::feedback::PropertyOutcome::Exotic,
                                    Some(o.borrow().props.shape()),
                                    0,
                                    None,
                                );
                            }
                            return Ok(Value::Undefined);
                        }
                        TaIndex::Ordinary => {}
                    }
                    // `length`, `byteLength`, `byteOffset` and `buffer` are accessors of
                    // %TypedArray.prototype%. When OrdinaryGet with this receiver would call one
                    // of those intrinsic getters, compute its result instead; anything else on
                    // the way (an own property, a subclass or replaced accessor, a Proxy, a null
                    // prototype) takes the ordinary lookup below.
                    let accessor = match &(receiver).view() {
                        ValueView::Obj(r) if Gc::ptr_eq(r, &o) => self.ta_accessor_reached(&o, key),
                        _ => None,
                    };
                    if let Some(accessor) = accessor {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(
                                crate::feedback::PropertyOutcome::Exotic,
                                Some(o.borrow().props.shape()),
                                0,
                                None,
                            );
                        }
                        let cur = self.ta_len(&info);
                        return Ok(match accessor {
                            TaAccessor::Length => Value::Num(cur.unwrap_or(0) as f64),
                            TaAccessor::ByteLength => {
                                Value::Num((cur.unwrap_or(0) * info.kind.elsize()) as f64)
                            }
                            TaAccessor::ByteOffset => Value::Num(if cur.is_none() {
                                0.0
                            } else {
                                info.offset as f64
                            }),
                            TaAccessor::Buffer => self
                                .ta_buffer
                                .get(&ptr)
                                .cloned()
                                .unwrap_or(Value::Undefined),
                        });
                    }
                }
                // Web IDL legacy platform object [[GetOwnProperty]]: a supported indexed
                // property is a lazy, enumerable/configurable, read-only data property.
                if let Some(value) = self.host_indexed_own_value(&o, key)? {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(o.borrow().props.shape()),
                            0,
                            None,
                        );
                    }
                    return Ok(value);
                }
                self.get_from_chain(&o, key, &receiver, trace, atom)
            }
        }
    }

    /// A Proxy's [[Get]] (ECMA-262 §10.5.8) of `key` with `receiver`: GetMethod(handler, "get");
    /// without a trap the target's [[Get]], else Call(trap, handler, «target, P, Receiver») and
    /// the invariant checks. `key_value` is P as an ECMAScript value when the caller already
    /// has it; otherwise it is built from the internal key text.
    fn proxy_get(
        &mut self,
        target: Value,
        handler: Value,
        key: &str,
        key_value: Option<Value>,
        receiver: Value,
    ) -> Result<Value, Abrupt> {
        let Some(trap) = self.proxy_get_trap(&handler, PROXY_TRAP_GET, "get")? else {
            // Forward to the target's [[Get]], preserving the original Receiver.
            return self.proxy_forward(|me| me.get_member_recv(&target, key, receiver));
        };
        let key_value = key_value.unwrap_or_else(|| self.key_value(key));
        let res = self.proxy_call_trap(
            PROXY_TRAP_GET,
            &trap,
            handler,
            [target, key_value, receiver],
        )?;
        self.proxy_get_invariant(&target, key, &res)?;
        Ok(res)
    }

    /// `object[index]` when `object` is a Proxy, for an integral Number key whose
    /// ToPropertyKey is `key` (GetValue → [[Get]](P, object)); `None` for any other object. A
    /// Proxy is never a module or deferred namespace, mapped arguments object or String
    /// wrapper, so its [[Get]] is exactly the trap path `get_member` would reach. Small index
    /// keys reach the trap as shared canonical strings.
    pub(crate) fn proxy_get_integer_key(
        &mut self,
        object: &Gc,
        index: f64,
        key: &str,
    ) -> Option<Result<Value, Abrupt>> {
        if object.borrow().ic_plain.get() {
            return None;
        }
        let (target, handler) = self.proxy_at(Gc::as_ptr(object) as usize)?;
        let key_value = if index >= 0.0 && index < crate::jstr::INDEX_STRINGS as f64 {
            // -0 is the index 0: ToString(-0) is "0".
            Value::Str(crate::jstr::index_lstr(index as u32))
        } else {
            crate::value::string_key_value(key)
        };
        Some(self.proxy_get(target, handler, key, Some(key_value), Value::Obj(*object)))
    }

    fn get_from_chain(
        &mut self,
        start: &Gc,
        key: &str,
        receiver: &Value,
        mut trace: Option<&mut crate::feedback::CurrentPropertyTrace>,
        atom: Option<crate::atom::PropKey>,
    ) -> Result<Value, Abrupt> {
        let mut cur = Some(*start);
        let mut depth = 0_u8;
        // The key's atom: the caller's, or once a map on the chain has looked it up.
        let mut walk_atom = atom;
        while let Some(obj) = cur {
            // A deferred namespace anywhere on the chain evaluates its module first.
            self.defer_trigger(&obj, Some(key))?;
            // A proxy anywhere on the chain handles the read itself (with the original receiver).
            let ptr = Gc::as_ptr(&obj) as usize;
            if let Some((target, handler)) = self.proxy_at(ptr) {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Exotic,
                        Some(obj.borrow().props.shape()),
                        depth,
                        None,
                    );
                }
                if matches!((handler).view(), ValueView::Null) {
                    return Err(self.throw("TypeError", "cannot perform 'get' on a revoked proxy"));
                }
                let Some(trap) = self.proxy_get_trap(&handler, PROXY_TRAP_GET, "get")? else {
                    return self.proxy_forward(|me| me.get_member_recv(&target, key, *receiver));
                };
                let key_value = self.key_value(key);
                let res = self.proxy_call_trap(
                    PROXY_TRAP_GET,
                    &trap,
                    handler,
                    [target, key_value, *receiver],
                )?;
                self.proxy_get_invariant(&target, key, &res)?;
                return Ok(res);
            }
            if let Some(value) = self.host_indexed_own_value(&obj, key)? {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Exotic,
                        Some(obj.borrow().props.shape()),
                        depth,
                        None,
                    );
                }
                return Ok(value);
            }
            // A TypedArray on the chain (`Object.create(typedArray)[1]`) answers with its own
            // [[Get]] (ECMA-262 §10.4.5.4): a canonical numeric key reads TypedArrayGetElement,
            // which is undefined for an invalid index and never consults its prototype; any
            // other key continues with OrdinaryGet. Its elements live in the buffer, not in the
            // property map read below.
            if !self.typed_arrays.is_empty() {
                if let Some(info) = self.typed_arrays.get(&ptr).copied() {
                    let element = match self.ta_index_kind(&info, key) {
                        TaIndex::Element(index) => Some(self.ta_read(&info, index)),
                        TaIndex::Exotic => Some(Value::Undefined),
                        TaIndex::Ordinary => None,
                    };
                    if let Some(value) = element {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(
                                crate::feedback::PropertyOutcome::Exotic,
                                Some(obj.borrow().props.shape()),
                                depth,
                                None,
                            );
                        }
                        return Ok(value);
                    }
                }
            }
            // OrdinaryGet (ECMA-262): a data descriptor returns its owned value directly.
            // Only getters need a snapshot across the author-code boundary. Keeping data out
            // of that snapshot also avoids copying an optional descriptor/trace tuple.
            let getter = {
                let b = obj.borrow();
                if let Some(trace) = trace.as_deref_mut() {
                    match b.props.get_with_slot_walking(key, &mut walk_atom) {
                        Some((p, slot)) => {
                            let holder_shape = b.props.shape();
                            if !p.accessor() {
                                let slot = slot.map(|slot| slot as u32);
                                let value = p.value();
                                drop(b);
                                trace.record(
                                    crate::feedback::PropertyOutcome::Data,
                                    Some(holder_shape),
                                    depth,
                                    slot,
                                );
                                return Ok(value);
                            }
                            let getter = p.getter().cloned();
                            drop(b);
                            trace.record(
                                crate::feedback::PropertyOutcome::Accessor,
                                Some(holder_shape),
                                depth,
                                None,
                            );
                            Some(getter)
                        }
                        None => None,
                    }
                } else {
                    // `Props::get` is the semantic lookup and also covers dense/indexed storage;
                    // an untraced data read needs neither the named slot nor the holder shape.
                    match b.props.get_walking(key, &mut walk_atom) {
                        Some(p) if !p.accessor() => return Ok(p.value()),
                        Some(p) => Some(p.getter().cloned()),
                        None => None,
                    }
                }
            };
            if let Some(getter) = getter {
                // Legacy `fn.caller` / `fn.arguments`: reading the poisoned
                // %Function.prototype% accessor through an ordinary sloppy function
                // yields undefined instead of throwing.
                if matches!(key, "caller" | "arguments")
                    && self.is_throw_type_error(&getter)
                    && !Gc::ptr_eq(
                        &obj,
                        match &(receiver).view() {
                            ValueView::Obj(r) => r,
                            _ => &obj,
                        },
                    )
                {
                    if let ValueView::Obj(r) = &(receiver).view() {
                        let plain = match &r.borrow().call {
                            Callable::User(user) => {
                                !user.func.is_strict
                                    && !user.func.is_arrow
                                    && !user.func.is_generator
                                    && !user.func.is_async
                                    && !user.func.is_method
                            }
                            _ => false,
                        };
                        if plain {
                            // Legacy SpiderMonkey reflection: the youngest activation's
                            // arguments object / calling function (null when inactive; a
                            // strict caller is censored to null). Eval runs inline, so eval
                            // frames are naturally skipped.
                            let rptr = Gc::as_ptr(r) as usize;
                            let activation = self.legacy_activation(rptr);
                            let top = match activation {
                                Some(LegacyActivation::Frame(k)) => Some(k),
                                _ => None,
                            };
                            trace_legacy_reflection(key, top.map(|k| &self.fn_frames[k]));
                            if let Some(activation) = activation.filter(|_| key == "caller") {
                                return Ok(self.legacy_caller(activation));
                            }
                            return Ok(match (key, top) {
                                // A spliced activation keeps no arguments object.
                                (_, None) => Value::Null,
                                ("arguments", Some(k)) => {
                                    // The activation skipped building its arguments object
                                    // (the body never names it) — conjure it now.
                                    let lazy = match self.fn_frames[k].extra.as_deref() {
                                        Some(x)
                                            if matches!((x.args_obj).view(), ValueView::Null) =>
                                        {
                                            x.lazy.clone()
                                        }
                                        _ => None,
                                    };
                                    if let Some((func, args, scope)) = lazy {
                                        let ao =
                                            self.make_arguments_object(&func, &args, &scope, r);
                                        if let Some(x) = self.fn_frames[k].extra.as_deref_mut() {
                                            x.args_obj = Value::Obj(ao);
                                        }
                                    }
                                    match self.fn_frames[k].extra.as_deref() {
                                        Some(x) => x.args_obj,
                                        None => Value::Null,
                                    }
                                }
                                (_, Some(_)) => unreachable!("caller is answered above"),
                            });
                        }
                    }
                }
                return match getter {
                    Some(getter) => match self.try_binding_getter(&getter) {
                        Some(value) => Ok(value),
                        None => self.call_callback(getter, *receiver, &[]),
                    },
                    None => Ok(Value::Undefined),
                };
            }
            cur = obj.borrow().proto;
            depth = depth.saturating_add(1);
        }
        if let Some(trace) = trace {
            trace.record(crate::feedback::PropertyOutcome::Absent, None, depth, None);
        }
        Ok(Value::Undefined)
    }

    /// Throw with the constructing caller's realm's error intrinsics (see `ctor_caller_realm`).
    fn throw_in_caller_realm(&mut self, kind: &str, msg: &str) -> Abrupt {
        match self
            .ctor_caller_realm
            .as_ref()
            .map(RealmState::snapshot_clone)
        {
            Some(caller) => {
                let key = Gc::as_ptr(&caller.global) as usize;
                self.with_realm_snapshot(key, caller, |interp| interp.throw(kind, msg))
            }
            None => self.throw(kind, msg),
        }
    }

    /// Whether `v` is the [[IsHTMLDDA]] object.
    pub(crate) fn is_htmldda(&self, v: &Value) -> bool {
        match &(v).view() {
            ValueView::Obj(o) => self
                .htmldda
                .get(&(Gc::as_ptr(o) as usize))
                .is_some_and(|dda| dda.ptr_eq(&Gc::downgrade(o))),
            _ => false,
        }
    }

    /// Both operands of a literal typeof comparison are strings. Their equality
    /// needs no conversion, callback or allocated typeof result.
    pub(crate) fn typeof_matches(&self, value: &Value, test: crate::value::TypeofTest) -> bool {
        use crate::value::TypeofTest;
        if self.is_htmldda(value) {
            return test == TypeofTest::Undefined;
        }
        match test {
            TypeofTest::Undefined => {
                matches!(&(value).view(), ValueView::Undefined | ValueView::Empty)
            }
            TypeofTest::Boolean => matches!(&(value).view(), ValueView::Bool(_)),
            TypeofTest::Number => matches!(&(value).view(), ValueView::Num(_)),
            TypeofTest::BigInt => matches!(&(value).view(), ValueView::BigInt(_)),
            TypeofTest::String => matches!(&(value).view(), ValueView::Str(_)),
            TypeofTest::Symbol => matches!(&(value).view(), ValueView::Sym(_)),
            TypeofTest::Object => match &(value).view() {
                ValueView::Null => true,
                ValueView::Obj(object) => {
                    matches!(object.borrow().call, Callable::None)
                }
                _ => false,
            },
            TypeofTest::Function => value.is_callable(),
            TypeofTest::Never => false,
        }
    }

    /// Whether `getter` is the %ThrowTypeError% intrinsic.
    fn is_throw_type_error(&self, getter: &Option<Value>) -> bool {
        match (
            (getter).map(|__value| (__value).view()),
            self.extra_protos.get("%ThrowTypeError%"),
        ) {
            (Some(ValueView::Obj(g)), Some(tte)) => Gc::ptr_eq(&g, tte),
            _ => false,
        }
    }

    /// Set `base[key] = value`, honouring setters, accessor-only properties, read-only data
    /// properties, and array `length`/index bookkeeping. The receiver defaults to `base`.
    /// The parameter name a still-mapped `arguments[key]` aliases, if any.
    pub(crate) fn mapped_arg_name(&self, ptr: usize, key: &str) -> Option<String> {
        let (_, names) = self.mapped_arguments.get(&ptr)?;
        // ParameterMap keys are exact array-index strings, not every spelling
        // accepted by an unsigned integer parser (for example, "00" or "+0").
        let idx = crate::value::canonical_index(key)? as usize;
        names.get(idx)?.clone()
    }

    /// The current parameter value a mapped `arguments[key]` aliases, if the map is live.
    pub(crate) fn mapped_arg_value(&self, ptr: usize, key: &str) -> Option<Value> {
        let name = self.mapped_arg_name(ptr, key)?;
        let (env, _) = self.mapped_arguments.get(&ptr)?;
        if let Some(value) = self
            .borrowed_parameters
            .as_ref()
            .and_then(|registry| registry.read(env, &name))
        {
            return Some(value);
        }
        env.borrow().vars.get(name.as_str()).map(|b| b.value())
    }

    /// Write through a mapped `arguments[key]` alias into its parameter binding.
    pub(crate) fn mapped_arg_write(&mut self, ptr: usize, key: &str, v: &Value) -> bool {
        let Some(name) = self.mapped_arg_name(ptr, key) else {
            return false;
        };
        let Some((env, _)) = self.mapped_arguments.get(&ptr) else {
            return false;
        };
        if self
            .borrowed_parameters
            .as_ref()
            .is_some_and(|registry| registry.write(env, &name, v))
        {
            return true;
        }
        if let Some(b) = env.borrow_mut().vars.get_mut(&name) {
            b.set_value(*v);
            return true;
        }
        false
    }

    /// Unmap an `arguments` index (delete / defineProperty severs the parameter alias).
    pub(crate) fn unmap_argument(&mut self, ptr: usize, key: &str) {
        if let Some((_, names)) = self.mapped_arguments.get_mut(&self.gc_heap, &ptr) {
            if let Some(idx) = crate::value::canonical_index(key) {
                if let Some(slot) = names.get_mut(idx as usize) {
                    *slot = None;
                }
            }
        }
    }

    /// Resolve a module-namespace property before rejecting a write: an exported binding still in
    /// its temporal dead zone throws ReferenceError (per the namespace [[GetOwnProperty]] step).
    fn ns_probe_tdz(&mut self, ptr: usize, key: &str) -> Result<(), Abrupt> {
        let live = self
            .namespace_binding(ptr, key)
            .and_then(|binding| match binding {
                crate::modules::NsBinding::Live(env, local) => Some((env, local)),
                _ => None,
            });
        if let Some((env, local)) = live {
            self.get_var(&local, &env)?;
        }
        Ok(())
    }

    /// Write an author String key; see [`Self::get_member`] for its key boundary.
    pub(crate) fn set_member(
        &mut self,
        base: &Value,
        key: &str,
        value: Value,
    ) -> Result<(), Abrupt> {
        self.set_member_key(base, &crate::value::encode_string_key(key), value)
    }

    /// Engine [[Set]] for an already encoded String-or-Symbol property key.
    pub(crate) fn set_member_key(
        &mut self,
        base: &Value,
        key: &str,
        value: Value,
    ) -> Result<(), Abrupt> {
        let strict = self.strict;
        let success = self.set_member_recv(base, key, value, *base)?;
        self.finish_assignment_set(key, strict, success)
    }

    /// PutValue tests the Boolean [[Set]] result using the reference's strictness.
    /// Proxy traps can reject a write normally; Reflect.set still receives false
    /// from the underlying internal method instead of an assignment exception.
    pub(crate) fn finish_assignment_set(
        &mut self,
        key: &str,
        strict: bool,
        success: bool,
    ) -> Result<(), Abrupt> {
        if !success && strict {
            Err(self.throw(
                "TypeError",
                format!(
                    "Cannot assign to property '{}'",
                    crate::value::decode_string_key(key)
                ),
            ))
        } else {
            Ok(())
        }
    }

    /// Profile-enabled `[[Set]]` entry used by computed element operations. The trace is filled at
    /// the semantic setter/proxy/rejection branch, while the operation itself remains single-shot.
    pub(crate) fn set_member_profiled(
        &mut self,
        base: &Value,
        key: &str,
        value: Value,
        trace: &mut crate::feedback::CurrentPropertyTrace,
    ) -> Result<(), Abrupt> {
        trace.receiver_shape = self.property_receiver_shape(base);
        let strict = self.strict;
        let success = self.set_member_recv_impl(base, key, value, *base, Some(trace))?;
        self.finish_assignment_set(key, strict, success)
    }

    /// [[Set]](P, V, Receiver): like [`Self::set_member_key`] but with an explicit `receiver` and returning the
    /// [[Set]] success boolean. The receiver is the object a setter is invoked on, the proxy `set`
    /// trap's Receiver argument, and what a forwarded `[[Set]]` carries through a proxy target chain.
    pub(crate) fn set_member_recv(
        &mut self,
        base: &Value,
        key: &str,
        value: Value,
        receiver: Value,
    ) -> Result<bool, Abrupt> {
        self.set_member_recv_impl(base, key, value, receiver, None)
    }

    fn set_member_recv_impl(
        &mut self,
        base: &Value,
        key: &str,
        value: Value,
        receiver: Value,
        mut trace: Option<&mut crate::feedback::CurrentPropertyTrace>,
    ) -> Result<bool, Abrupt> {
        // Arguments [[Set]] updates its ParameterMap only when the arguments object
        // itself is Receiver. A proxy forwarding to it must preserve a distinct Receiver.
        if !self.mapped_arguments.is_empty() {
            if let (ValueView::Obj(o), ValueView::Obj(r)) = (&(base).view(), &(receiver).view()) {
                let ptr = Gc::as_ptr(o) as usize;
                if Gc::ptr_eq(o, r) && self.mapped_arg_write(ptr, key, &value) {
                    // Update the own property's value in place — its attributes (e.g. a
                    // configurable:false from defineProperty) must survive.
                    let mut b = o.borrow_mut();
                    if !b.props.set_value(key, value) {
                        b.props
                            .insert(key, crate::value::Property::data(value, true, true, true));
                    }
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(b.props.shape()),
                            0,
                            None,
                        );
                    }
                    return Ok(true);
                }
            }
        }
        let obj = match &(base).view() {
            ValueView::Obj(o) => *o,
            ValueView::Undefined | ValueView::Null => {
                if let Some(trace) = trace {
                    trace.record(crate::feedback::PropertyOutcome::Rejected, None, 0, None);
                }
                return Err(self.throw(
                    "TypeError",
                    format!(
                        "cannot set property '{}' of {}",
                        crate::value::decode_string_key(key),
                        type_name(base)
                    ),
                ));
            }
            // Setting a property on a primitive: an accessor (or proxy) on the wrapper
            // prototype chain still handles the write, with the primitive as receiver; otherwise
            // the write fails — a silent no-op in sloppy mode, a TypeError in strict mode.
            _ => {
                let proto = match &(base).view() {
                    ValueView::Str(_) => Some(self.string_proto),
                    ValueView::Num(_) => Some(self.number_proto),
                    ValueView::Bool(_) => Some(self.boolean_proto),
                    ValueView::Sym(_) => Some(self.symbol_proto),
                    ValueView::BigInt(_) => self.extra_protos.get("BigInt").cloned(),
                    _ => None,
                };
                let mut cur = proto;
                let mut depth = 0_u8;
                while let Some(o) = cur {
                    // A proxy on the chain (or any accessor) takes over with the original receiver.
                    if self.proxies.contains_key(&(Gc::as_ptr(&o) as usize)) {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(
                                crate::feedback::PropertyOutcome::Exotic,
                                Some(o.borrow().props.shape()),
                                depth,
                                None,
                            );
                        }
                        return self.set_member_recv(&Value::Obj(o), key, value, receiver);
                    }
                    let prop = o.borrow().props.get(key).cloned();
                    if let Some(p) = prop {
                        if p.accessor() {
                            if let Some(trace) = trace.as_deref_mut() {
                                trace.record(
                                    crate::feedback::PropertyOutcome::Accessor,
                                    Some(o.borrow().props.shape()),
                                    depth,
                                    None,
                                );
                            }
                            return match p.setter().cloned() {
                                Some(setter) => {
                                    self.call_callback(setter, receiver, &[value])?;
                                    Ok(true)
                                }
                                None => {
                                    if self.strict {
                                        Err(self.throw(
                                            "TypeError",
                                            format!(
                                                "cannot set getter-only property '{}'",
                                                crate::value::decode_string_key(key)
                                            ),
                                        ))
                                    } else {
                                        Ok(false)
                                    }
                                }
                            };
                        }
                        break;
                    }
                    let parent = o.borrow().proto;
                    cur = parent;
                    depth = depth.saturating_add(1);
                }
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Rejected,
                        None,
                        depth,
                        None,
                    );
                }
                if self.strict {
                    return Err(self.throw(
                        "TypeError",
                        format!(
                            "Cannot create property '{key}' on {} value",
                            type_name(base)
                        ),
                    ));
                }
                return Ok(false);
            }
        };

        let ptr = Gc::as_ptr(&obj) as usize;
        if matches!(obj.borrow().exotic, Exotic::Arguments) {
            if let Some(trace) = trace.as_deref_mut() {
                trace.record(
                    crate::feedback::PropertyOutcome::Exotic,
                    Some(obj.borrow().props.shape()),
                    0,
                    None,
                );
            }
            // The mapped same-receiver case above keeps its direct write. Every
            // other Arguments [[Set]] uses OrdinarySet with the original Receiver,
            // including proxy forwarding and primitive or rejecting receivers.
            return crate::builtins::reflect_ordinary_set(self, base, key, value, &receiver)
                .map_err(Abrupt::Throw);
        }
        // A module namespace exotic object's [[Set]] always fails: in strict code (all module code
        // is strict) the assignment throws a TypeError; a sloppy caller sees an inert no-op. The
        // property is resolved first, so a binding still in its TDZ throws ReferenceError instead.
        if self.is_namespace(ptr) {
            if let Some(trace) = trace.as_deref_mut() {
                trace.record(
                    crate::feedback::PropertyOutcome::Exotic,
                    Some(obj.borrow().props.shape()),
                    0,
                    None,
                );
            }
            self.ns_probe_tdz(ptr, key)?;
            if self.strict {
                return Err(self.throw(
                    "TypeError",
                    format!(
                        "Cannot assign to read only property '{key}' of module namespace object"
                    ),
                ));
            }
            return Ok(false);
        }
        // Proxy: invoke the `set` trap, or forward to the target.
        {
            if let Some((target, handler)) = self.proxy_at(ptr) {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Exotic,
                        Some(obj.borrow().props.shape()),
                        0,
                        None,
                    );
                }
                let Some(trap) = self.proxy_get_trap(&handler, PROXY_TRAP_SET, "set")? else {
                    // Forward to the target's [[Set]], preserving the original Receiver.
                    return self
                        .proxy_forward(|me| me.set_member_recv(&target, key, value, receiver));
                };
                let key_value = self.key_value(key);
                let ok = self.proxy_call_trap(
                    PROXY_TRAP_SET,
                    &trap,
                    handler,
                    [target, key_value, value, receiver],
                )?;
                // A successful `set` can't contradict a non-configurable property on the target.
                let success = self.to_boolean(&ok);
                if success {
                    self.proxy_set_invariant(&target, key, &value)?;
                }
                return Ok(success);
            }
        }
        // TypedArray integer-index writes go straight to the backing buffer; a canonical-numeric key
        // that isn't a valid index is inert (the value coercion still runs, then is discarded).
        if let Some(info) = self.typed_array_at(ptr) {
            match self.ta_index_kind(&info, key) {
                TaIndex::Element(_) | TaIndex::Exotic => {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(obj.borrow().props.shape()),
                            0,
                            None,
                        );
                    }
                    // IntegerIndexedElementSet coerces the value first — the coercion can resize
                    // the underlying buffer, changing which indices are valid — then re-checks the
                    // index and silently discards an out-of-bounds write.
                    let num = if info.kind.is_bigint() {
                        Value::BigInt(self.to_bigint(&value)?)
                    } else {
                        Value::Num(self.to_number(&value)?)
                    };
                    if let TaIndex::Element(idx) = self.ta_index_kind(&info, key) {
                        self.ta_store(&info, idx, &num)?;
                    }
                    return Ok(true);
                }
                TaIndex::Ordinary => {}
            }
        }

        // Walk the chain for an accessor or read-only data property.
        let mut cur = Some(obj);
        let mut depth = 0_u8;
        while let Some(o) = cur {
            // A deferred namespace anywhere on the chain evaluates its module first.
            self.defer_trigger(&o, Some(key))?;
            // A proxy on the chain handles the write itself (with the original receiver).
            let optr = Gc::as_ptr(&o) as usize;
            if let Some((target, handler)) = self.proxy_at(optr) {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Exotic,
                        Some(o.borrow().props.shape()),
                        depth,
                        None,
                    );
                }
                if matches!((handler).view(), ValueView::Null) {
                    return Err(self.throw("TypeError", "cannot perform 'set' on a revoked proxy"));
                }
                let Some(trap) = self.proxy_get_trap(&handler, PROXY_TRAP_SET, "set")? else {
                    return self
                        .proxy_forward(|me| me.set_member_recv(&target, key, value, receiver));
                };
                let key_value = self.key_value(key);
                let ok = self.proxy_call_trap(
                    PROXY_TRAP_SET,
                    &trap,
                    handler,
                    [target, key_value, value, receiver],
                )?;
                let success = self.to_boolean(&ok);
                if success {
                    self.proxy_set_invariant(&target, key, &value)?;
                }
                return Ok(success);
            }
            // A TypedArray reached via the prototype chain: its integer-indexed elements are its own
            // properties, so the search must stop here (never consulting the TA prototype's
            // accessors). A valid index is a writable data property → create one on the receiver; a
            // canonical-numeric non-index is an inert success.
            if !Gc::ptr_eq(&o, &obj) {
                if let Some(info) = self.typed_arrays.get(&optr).copied() {
                    match self.ta_index_kind(&info, key) {
                        TaIndex::Element(_) | TaIndex::Exotic if matches!(&(receiver).view(), ValueView::Obj(r) if Gc::ptr_eq(r, &o)) =>
                        {
                            if let Some(trace) = trace.as_deref_mut() {
                                trace.record(
                                    crate::feedback::PropertyOutcome::Exotic,
                                    Some(o.borrow().props.shape()),
                                    depth,
                                    None,
                                );
                            }
                            // The receiver IS this TypedArray: IntegerIndexedElementSet applies
                            // (coerce, then store or silently drop an out-of-range write).
                            let num = if info.kind.is_bigint() {
                                Value::BigInt(self.to_bigint(&value)?)
                            } else {
                                Value::Num(self.to_number(&value)?)
                            };
                            if let TaIndex::Element(idx) = self.ta_index_kind(&info, key) {
                                self.ta_store(&info, idx, &num)?;
                            }
                            return Ok(true);
                        }
                        // Foreign receiver: a valid element behaves as a writable data property
                        // (create it on the receiver, uncoerced); a numeric non-index is inert.
                        TaIndex::Element(_) => break,
                        TaIndex::Exotic => {
                            if let Some(trace) = trace.as_deref_mut() {
                                trace.record(
                                    crate::feedback::PropertyOutcome::Exotic,
                                    Some(o.borrow().props.shape()),
                                    depth,
                                    None,
                                );
                            }
                            return Ok(true);
                        }
                        TaIndex::Ordinary => {}
                    }
                }
            }
            // Web IDL legacy platform object [[Set]] first obtains the indexed own descriptor.
            // A supported read-only index therefore runs its getter and then rejects the write.
            if self.host_indexed_array_key(&o, key)
                && self.host_indexed_own_value(&o, key)?.is_some()
            {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.record(
                        crate::feedback::PropertyOutcome::Exotic,
                        Some(o.borrow().props.shape()),
                        depth,
                        None,
                    );
                }
                if self.strict {
                    return Err(self.throw(
                        "TypeError",
                        format!(
                            "cannot assign to read-only indexed property '{}'",
                            crate::value::decode_string_key(key)
                        ),
                    ));
                }
                return Ok(false);
            }
            let prop = o.borrow().props.get(key).cloned();
            if let Some(p) = prop {
                if p.accessor() {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Accessor,
                            Some(o.borrow().props.shape()),
                            depth,
                            None,
                        );
                    }
                    return match p.setter().cloned() {
                        Some(setter) => {
                            self.call_callback(setter, receiver, &[value])?;
                            Ok(true)
                        }
                        None => {
                            if self.strict {
                                Err(self.throw(
                                    "TypeError",
                                    format!(
                                        "cannot set getter-only property '{}'",
                                        crate::value::decode_string_key(key)
                                    ),
                                ))
                            } else {
                                Ok(false)
                            }
                        }
                    };
                }
                if Gc::ptr_eq(&o, &obj) {
                    if !p.writable() {
                        if let Some(trace) = trace.as_deref_mut() {
                            trace.record(
                                crate::feedback::PropertyOutcome::Rejected,
                                Some(o.borrow().props.shape()),
                                depth,
                                None,
                            );
                        }
                        if self.strict {
                            return Err(self.throw(
                                "TypeError",
                                format!(
                                    "cannot assign to read-only property '{}'",
                                    crate::value::decode_string_key(key)
                                ),
                            ));
                        }
                        return Ok(false);
                    }
                    break; // own writable data property — update below
                }
                if !p.writable() {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Rejected,
                            Some(o.borrow().props.shape()),
                            depth,
                            None,
                        );
                    }
                    if self.strict {
                        return Err(self.throw(
                            "TypeError",
                            format!(
                                "cannot assign to read-only property '{}'",
                                crate::value::decode_string_key(key)
                            ),
                        ));
                    }
                    return Ok(false);
                }
                break; // inherited writable data property — create own on receiver
            }
            cur = o.borrow().proto;
            depth = depth.saturating_add(1);
        }

        // OrdinarySet: the write lands on the *receiver* (they differ for `super.x = v` and
        // Reflect.set with an explicit receiver). A namespace receiver rejects the define,
        // probing the binding first so a TDZ export surfaces ReferenceError.
        let obj = match &(receiver).view() {
            ValueView::Obj(r) if !Gc::ptr_eq(r, &obj) => {
                // A deferred-namespace receiver evaluates before its own properties are consulted.
                self.defer_trigger(r, Some(key))?;
                let rptr = Gc::as_ptr(r) as usize;
                if self.is_namespace(rptr) {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(r.borrow().props.shape()),
                            depth,
                            None,
                        );
                    }
                    self.ns_probe_tdz(rptr, key)?;
                    if self.strict {
                        return Err(self.throw(
                            "TypeError",
                            format!(
                                "Cannot assign to read only property '{key}' of module namespace object"
                            ),
                        ));
                    }
                    return Ok(false);
                }
                // Proxy and arguments receivers require their [[DefineOwnProperty]]:
                // observable traps and live parameter mappings must not be bypassed.
                let arguments_receiver = matches!(r.borrow().exotic, Exotic::Arguments);
                if arguments_receiver || self.proxies.contains_key(&rptr) {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Exotic,
                            Some(r.borrow().props.shape()),
                            depth,
                            None,
                        );
                    }
                    return match crate::builtins::reflect_define_on_receiver(
                        self, &receiver, key, value,
                    ) {
                        Ok(false) if self.strict => Err(self.throw(
                            "TypeError",
                            format!(
                                "cannot create property '{}' on the receiver",
                                crate::value::decode_string_key(key)
                            ),
                        )),
                        Ok(b) => Ok(b),
                        Err(v) => Err(Abrupt::Throw(v)),
                    };
                }
                // OrdinarySetWithOwnDescriptor checks a distinct receiver's existing own
                // descriptor before calling its [[DefineOwnProperty]]. In particular, an
                // Array's read-only length rejects *before* either length conversion.
                let rejected_array_descriptor = {
                    let receiver = r.borrow();
                    matches!(receiver.exotic, Exotic::Array)
                        && receiver
                            .props
                            .get(key)
                            .is_some_and(|p| p.accessor() || !p.writable())
                };
                if rejected_array_descriptor {
                    if self.strict {
                        return Err(self.throw(
                            "TypeError",
                            format!(
                                "cannot assign to read-only property '{}'",
                                crate::value::decode_string_key(key)
                            ),
                        ));
                    }
                    return Ok(false);
                }
                *r
            }
            _ => obj,
        };
        // OrdinarySet ultimately creates an own property through the receiver's
        // [[DefineOwnProperty]]. A read-only Web IDL collection rejects every array-index key,
        // including currently unsupported indices.
        if self.host_indexed_array_key(&obj, key) || self.host_named_reject_define(&obj, key) {
            if let Some(trace) = trace.as_deref_mut() {
                trace.record(
                    crate::feedback::PropertyOutcome::Exotic,
                    Some(obj.borrow().props.shape()),
                    depth,
                    None,
                );
            }
            if self.strict {
                return Err(self.throw(
                    "TypeError",
                    format!(
                        "cannot define indexed property '{}'",
                        crate::value::decode_string_key(key)
                    ),
                ));
            }
            return Ok(false);
        }
        let is_array = matches!(obj.borrow().exotic, Exotic::Array);
        if is_array {
            if let Some(trace) = trace.as_deref_mut() {
                trace.record(
                    crate::feedback::PropertyOutcome::Exotic,
                    Some(obj.borrow().props.shape()),
                    depth,
                    None,
                );
            }
            return self.array_set(&obj, key, value);
        } else {
            let existed = obj.borrow().props.contains(key);
            if existed {
                // OrdinarySetWithOwnDescriptor 4.d: an existing receiver property must be a
                // writable data property (super.x = v / Reflect.set can reach here with a
                // receiver whose own descriptor differs from the base chain's).
                let (accessor, writable) = {
                    let b = obj.borrow();
                    let pr = b.props.get(key).unwrap();
                    (pr.accessor(), pr.writable())
                };
                if accessor || !writable {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Rejected,
                            Some(obj.borrow().props.shape()),
                            depth,
                            None,
                        );
                    }
                    if self.strict {
                        return Err(self.throw(
                            "TypeError",
                            format!(
                                "cannot assign to read only property '{}'",
                                crate::value::decode_string_key(key)
                            ),
                        ));
                    }
                    return Ok(false);
                }
                obj.borrow_mut().props.set_value(key, value);
                if let Some(trace) = trace.as_deref_mut() {
                    let b = obj.borrow();
                    trace.record(
                        crate::feedback::PropertyOutcome::Data,
                        Some(b.props.shape()),
                        0,
                        b.props.slot_of(key).map(|slot| slot as u32),
                    );
                }
            } else {
                if !obj.borrow().extensible {
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.record(
                            crate::feedback::PropertyOutcome::Rejected,
                            None,
                            depth,
                            None,
                        );
                    }
                    if self.strict {
                        return Err(self
                            .throw("TypeError", "cannot add property, object is not extensible"));
                    }
                    return Ok(false);
                }
                obj.borrow_mut().props.insert(key, Property::plain(value));
                if let Some(trace) = trace {
                    trace.record(
                        crate::feedback::PropertyOutcome::Created,
                        Some(obj.borrow().props.shape()),
                        depth,
                        None,
                    );
                }
            }
        }
        Ok(true)
    }

    fn array_set(&mut self, obj: &Gc, key: &str, value: Value) -> Result<bool, Abrupt> {
        if key == "length" {
            let success = crate::builtins::array_set_length_value(self, obj, value)?;
            if !success && self.strict {
                return Err(self.throw("TypeError", "cannot assign array length"));
            }
            return Ok(success);
        }
        // Adding a new index to a non-extensible (sealed/frozen) array is rejected.
        if !obj.borrow().props.contains(key) && !obj.borrow().extensible {
            if self.strict {
                return Err(
                    self.throw("TypeError", "cannot add property, object is not extensible")
                );
            }
            return Ok(false);
        }
        // Appending past a non-writable `length` is rejected.
        if let Some(idx) = crate::value::canonical_index(key) {
            let len = self.array_length(obj);
            let len_writable = obj
                .borrow()
                .props
                .get("length")
                .map(|p| p.writable())
                .unwrap_or(true);
            if idx as usize >= len && !len_writable {
                if self.strict {
                    return Err(self.throw(
                        "TypeError",
                        "cannot add an element past a read-only array length",
                    ));
                }
                return Ok(false);
            }
        }
        // An existing element keeps its attributes; [[Set]] only updates the value.
        if obj.borrow().props.contains(key) {
            obj.borrow_mut().props.set_value(key, value);
        } else {
            obj.borrow_mut().props.insert(key, Property::plain(value));
        }
        // Only a canonical array index (< 2^32 - 1) updates `length`; larger numeric keys are
        // ordinary properties.
        if let Some(i) = crate::value::canonical_index(key) {
            let i = i as usize;
            let len = self.array_length(obj);
            if i >= len {
                obj.borrow_mut().props.insert(
                    "length",
                    Property::data(Value::Num((i + 1) as f64), true, false, false),
                );
            }
        }
        Ok(true)
    }

    /// A TypedArray's *current* element length, or `None` if it's out of bounds (a fixed-length view
    /// whose resizable buffer shrank below its range) or its buffer is detached. A length-tracking
    /// view recomputes its length from the buffer's current size.
    pub(crate) fn ta_len(&self, info: &TaInfo) -> Option<usize> {
        info.length_for_buffer(self.array_buffer_byte_length(info.buffer)?)
    }

    /// ArrayBufferByteLength (ECMA-262 #sec-arraybufferbytelength) of the buffer object at `ptr`,
    /// or `None` when it is detached. A SharedArrayBuffer's length is that of its Shared Data
    /// Block, which any agent sharing it may grow.
    pub(crate) fn array_buffer_byte_length(&self, ptr: usize) -> Option<usize> {
        self.array_buffers.get(&ptr)?.byte_len()
    }

    /// The Data Block of the ArrayBuffer or SharedArrayBuffer object at `ptr`; `None` for a
    /// detached ArrayBuffer or any other object.
    pub(crate) fn buffer_data(&self, ptr: usize) -> Option<&BufferData> {
        self.array_buffers.get(&ptr)
    }

    /// Whether `ptr` is a buffer object with a Data Block: an attached ArrayBuffer or a
    /// SharedArrayBuffer (`!IsDetachedBuffer`).
    pub(crate) fn buffer_attached(&self, ptr: usize) -> bool {
        self.array_buffers.contains_key(&ptr)
    }

    /// The Shared Data Block id of the SharedArrayBuffer at `ptr`; `None` for an ArrayBuffer,
    /// attached or detached, and any other object.
    pub(crate) fn shared_block_id(&self, ptr: usize) -> Option<u64> {
        self.shared_block(ptr).map(|block| block.id())
    }

    /// An owning Shared Data Block reference that can outlive a JavaScript buffer wrapper.
    pub(crate) fn shared_block(&self, ptr: usize) -> Option<&SharedMem> {
        match self.array_buffers.get(&ptr)? {
            BufferData::Shared(block) => Some(block),
            BufferData::Local(_) => None,
        }
    }

    /// IsSharedArrayBuffer (ECMA-262 #sec-issharedarraybuffer) for the buffer object at `ptr`.
    pub(crate) fn is_shared_buffer(&self, ptr: usize) -> bool {
        self.shared_block_id(ptr).is_some()
    }

    /// The bytes of the attached ArrayBuffer at `ptr`; `None` for a SharedArrayBuffer, whose
    /// bytes are its Shared Data Block's, a detached ArrayBuffer or any other object.
    pub(crate) fn local_buffer(&self, ptr: usize) -> Option<&ArrayBufferBytes> {
        match self.array_buffers.get(&ptr)? {
            BufferData::Local(bytes) => Some(bytes),
            BufferData::Shared(_) => None,
        }
    }

    /// The intrinsic %TypedArray.prototype% getter that OrdinaryGet of `key` on the TypedArray
    /// `o` (ECMA-262 §10.4.5.5 [[Get]] for a non-numeric key, then §10.1.8.1) would call, when
    /// `o` has no own `key` and every prototype up to that accessor is an ordinary object whose
    /// properties all live in its property map. None for any other outcome.
    fn ta_accessor_reached(&self, o: &Gc, key: &str) -> Option<TaAccessor> {
        let accessor = match key {
            "length" => TaAccessor::Length,
            "byteLength" => TaAccessor::ByteLength,
            "byteOffset" => TaAccessor::ByteOffset,
            "buffer" => TaAccessor::Buffer,
            _ => return None,
        };
        let proto = {
            let o = o.borrow();
            if o.props.contains(key) {
                return None;
            }
            o.proto?
        };
        // The usual chain is a concrete prototype without such a key, then the holder of all
        // four intrinsic accessors: both proven once per epoch (see `TaAccessorProof`).
        let epoch = crate::value::proto_epoch();
        {
            let proof = self.ta_accessor_protector.borrow();
            let proto = Gc::as_ptr(&proto);
            if proof.epoch == epoch
                && epoch != u32::MAX
                && (std::ptr::eq(proof.holder.as_ptr(), proto)
                    || proof
                        .concrete
                        .iter()
                        .any(|c| std::ptr::eq(c.as_ptr(), proto)))
            {
                return Some(accessor);
            }
        }
        let (found, holder) = ta_accessor_on_chain(&proto, key)?;
        let mut proof = self.ta_accessor_protector.borrow_mut();
        if proof.epoch != epoch || !std::ptr::eq(proof.holder.as_ptr(), Gc::as_ptr(&holder)) {
            if !ta_accessor_holder_is_pristine(&holder) {
                return Some(found);
            }
            holder.borrow().props.mark_proto();
            *proof = TaAccessorProof {
                epoch,
                holder: Gc::downgrade(&holder),
                ..Default::default()
            };
        }
        let concrete = {
            let concrete = proto.borrow();
            concrete.ic_plain.get()
                && concrete.exotic.is_ordinary()
                && concrete
                    .proto
                    .as_ref()
                    .is_some_and(|p| Gc::ptr_eq(p, &holder))
                && ["length", "byteLength", "byteOffset", "buffer"]
                    .iter()
                    .all(|key| !concrete.props.contains(key))
        };
        if concrete {
            proto.borrow().props.mark_proto();
            let slot = proof.next_concrete;
            proof.concrete[slot] = Gc::downgrade(&proto);
            proof.next_concrete = (slot + 1) % proof.concrete.len();
        }
        Some(found)
    }

    /// Classify a property key against a TypedArray's integer-index exotic behavior: a valid
    /// in-range element index, a *canonical numeric* key that isn't one (e.g. "1.5", "-0", "-1",
    /// out of range — these are inert: never stored, never reach the prototype), or an ordinary key.
    pub(crate) fn ta_index_kind(&self, info: &TaInfo, key: &str) -> TaIndex {
        let n = match self.canonical_numeric_index(key) {
            Some(n) => n,
            None => return TaIndex::Ordinary,
        };
        // Valid index: a non-negative integer (not -0) below the current length.
        if n >= 0.0
            && n.fract() == 0.0
            && !(n == 0.0 && n.is_sign_negative())
            && (n as usize) < self.ta_len(info).unwrap_or(0)
        {
            TaIndex::Element(n as usize)
        } else {
            TaIndex::Exotic
        }
    }

    /// CanonicalNumericIndexString: the numeric value when `key` is the canonical string form of a
    /// Number ("0", "1.5", "-0", "Infinity", "NaN"…), else None.
    pub(crate) fn canonical_numeric_index(&self, key: &str) -> Option<f64> {
        // Number::toString output starts with a digit, `-`, `Infinity` or `NaN`: any other key
        // (every method or accessor name) is rejected before a float parse.
        if !matches!(
            key.as_bytes().first(),
            Some(b'0'..=b'9' | b'-' | b'I' | b'N')
        ) {
            return None;
        }
        if key == "-0" {
            return Some(-0.0);
        }
        let n = match key {
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            "NaN" => f64::NAN,
            _ => key.parse::<f64>().ok()?,
        };
        if self.num_to_str(n) == key {
            Some(n)
        } else {
            None
        }
    }

    pub(crate) fn array_length(&self, obj: &Gc) -> usize {
        // A TypedArray's length lives in its info slot, not an own `length` property.
        if let Some(info) = self.typed_arrays.get(&(Gc::as_ptr(obj) as usize)) {
            return self.ta_len(info).unwrap_or(0);
        }
        match (obj.borrow().props.length_property().map(|p| p.value()))
            .map(|__value| (__value).view())
        {
            Some(ValueView::Num(n)) => n as usize,
            _ => 0,
        }
    }

    /// Array length for an operation that will iterate/allocate proportional to it. Errors with a
    /// RangeError past [`MAX_ARRAY_OP_LEN`] so a huge `.length` cannot overwhelm memory.
    ///
    /// Take a recycled native-argument buffer (or allocate one) sized for `len` entries; callers
    /// must clear and return it through [`Interp::recycle_native_arg_buf`] after the call finishes.
    pub(crate) fn take_native_arg_buf(&mut self, len: usize) -> Vec<Value> {
        let mut buf = self.native_arg_pool.pop().unwrap_or_default();
        buf.clear();
        if buf.capacity() < len {
            buf.reserve(len.saturating_sub(buf.capacity()));
        }
        buf
    }

    /// Return a cleared native-argument buffer to the bounded pool (same cap as `vm_pool`).
    pub(crate) fn recycle_native_arg_buf(&mut self, mut buf: Vec<Value>) {
        buf.clear();
        if self.native_arg_pool.len() < 64 {
            self.native_arg_pool.push(buf);
        }
    }

    /// Array length for an operation that will iterate/allocate proportional to it. Errors with a
    /// RangeError past [`MAX_ARRAY_OP_LEN`] so a huge `.length` cannot exhaust memory.
    pub(crate) fn checked_array_len(&mut self, obj: &Gc) -> Result<usize, Abrupt> {
        let len = self.to_length(obj)?;
        if len > MAX_ARRAY_OP_LEN {
            return Err(self.throw("RangeError", "array length exceeds engine limit"));
        }
        Ok(len)
    }

    /// ToLength of an array-like's `length` property (coercing string/object lengths), clamped to
    /// the 2^53-1 spec maximum.
    pub(crate) fn to_length(&mut self, obj: &Gc) -> Result<usize, Abrupt> {
        let v = self.get_member_key(&Value::Obj(*obj), "length")?;
        let n = self.to_number(&v)?;
        Ok(if n.is_nan() || n <= 0.0 {
            0
        } else {
            n.trunc().min(9007199254740991.0) as usize
        })
    }

    // ----- garbage collection -----------------------------------------------------------------

    /// `s`'s UTF-16 view (see [`StrUnits`]), read from its storage.
    #[inline]
    pub(crate) fn units_of<'a>(&mut self, s: &'a crate::lstr::LStr) -> StrUnits<'a> {
        match s.units() {
            crate::lstr::Units::Ascii(_) => StrUnits::Ascii,
            crate::lstr::Units::Utf16(units) => StrUnits::Units(units),
        }
    }

    /// Code unit `idx` of `s`. `None` = out of range.
    #[inline]
    pub(crate) fn unit_at(&mut self, s: &crate::lstr::LStr, idx: usize) -> Option<u16> {
        s.unit_at(idx)
    }

    /// CodePointAt after receiver/index coercions.
    #[inline]
    pub(crate) fn code_point_at(&mut self, s: &crate::lstr::LStr, idx: usize) -> Option<u32> {
        s.code_point_at(idx)
    }

    /// The units as one UTF-16 slice (for range/search operations): a two-byte string's own,
    /// a one-byte string's widened.
    #[inline]
    pub(crate) fn units_full<'a>(
        &mut self,
        s: &'a crate::lstr::LStr,
    ) -> std::borrow::Cow<'a, [u16]> {
        s.units().to_utf16()
    }

    /// `s.length` (UTF-16 units).
    #[inline]
    pub(crate) fn str_len(&mut self, s: &crate::lstr::LStr) -> usize {
        s.unit_len()
    }

    /// [`gc_check`] amortized for layered calls. A direct JIT call does not come here: its
    /// callee's entry poll is the call's safepoint (see `jit::emit_interrupt_poll`).
    #[inline]
    pub(crate) fn gc_check_amortized(&mut self) -> Result<(), Abrupt> {
        self.gc_tick = self.gc_tick.wrapping_add(1);
        if gc_collect::gc_stress() != 0 {
            return self.gc_check();
        }
        if self.gc_tick & GC_CALL_POLL_MASK == 0 {
            self.gc_check()
        } else {
            Ok(())
        }
    }

    /// Allocation safe point. Collects when the heap's nursery or byte budget is spent, its
    /// stress interval reached or a deferred request waits (`gc_heap_due`): minorly, unless a
    /// major collection is due. Collects majorly when the live-object count is above its
    /// ceiling (`gc_next`), counting old objects that died since a collection last marked them;
    /// if that leaves more than the engine's live-object limit alive, throws a RangeError rather
    /// than exhaust RAM.
    pub(crate) fn gc_check(&mut self) -> Result<(), Abrupt> {
        // The due word JIT polls read is the core's `collection_due` (debug builds).
        crate::value::debug_assert_due_word(&self.gc_heap);
        if self.gc_heap.agent.suppressed.get() != 0 {
            return Ok(());
        }
        self.gc_weak_metadata_safepoint(false);
        // Only a collection the ceiling makes due enforces the limit: GC stress and the cell
        // heap's budgets also collect here, and must not throw again inside the handler that
        // the room set below lets allocate.
        let over_ceiling =
            crate::value::heap_live_objects(&self.gc_heap) > self.gc_heap.agent.next.get();
        if !over_ceiling && !self.gc_heap_due() {
            return Ok(());
        }
        // A minor collection would leave the old dead counted: the limit is enforced after a
        // major one.
        let kind = over_ceiling.then_some(crate::gc_core::CollectionKind::Major);
        let outcome = self.gc_collect_with_cause(crate::value::GcCause::AllocationThreshold, kind);
        if outcome == crate::agent_control::CollectionOutcome::Deferred {
            return Ok(());
        }
        let live = crate::value::heap_live_objects(&self.gc_heap);
        if std::env::var_os("TRUST_JS_GC_LOG").is_some() {
            eprintln!(
                "[gc] live={live} realms={} templates={} pins={} host_settings={} coroutines={}",
                self.realms.len(),
                self.template_cache.len(),
                self.gc_pins.len(),
                self.host_settings_states.len(),
                self.generators.len()
            );
        }
        if over_ceiling && live > self.gc_heap.agent.limit.get() {
            // Leave the handler that catches this error room to allocate (its error object,
            // messages, cleanup) before the next safepoint collects again. Without the room,
            // the very first allocation in the `catch` block would rerun the full collection
            // and throw again, so the error could never be handled in practice.
            self.set_agent_next(live.saturating_add(LIMIT_ERROR_HANDLER_ROOM));
            return Err(self.throw("RangeError", "allocation limit exceeded"));
        }
        // A completed collection below the limit restores the shared ceiling. Collections
        // not caused by the ceiling preserve the Agent's existing handler allowance.
        if over_ceiling {
            self.set_agent_next(self.gc_heap.agent.limit.get());
        }
        Ok(())
    }

    /// Optional collection after a completed outer checkpoint. ClearKeptObjects/host leases
    /// are released by the checkpoint independently of this optional scheduling policy.
    pub(crate) fn gc_task_boundary(&mut self) -> i64 {
        let heap = self.gc_heap.clone();
        if heap.agent.suppressed.get() != 0 || heap.agent.execution_depth.get() != 0 {
            return 0;
        }
        self.gc_weak_metadata_safepoint(true);
        let allocated = crate::value::heap_allocated_objects(&heap);
        let churn = allocated.wrapping_sub(heap.agent.task_allocated.replace(allocated));
        let before = crate::value::heap_live_objects(&heap);
        let previous = heap.agent.task_live.get();
        let floor = GC_TASK_ALLOCATION_TRIGGER as i64;
        let growth = before.saturating_sub(previous);
        heap.agent.task_pending.set(
            heap.agent.task_pending.get()
                || growth >= (previous / 8).max(floor)
                || churn >= (previous / 2).max(floor) as u64,
        );
        if !heap.agent.task_pending.get() || heap.agent.task_deferred.get() {
            return 0;
        }
        self.gc_collect_with_cause(crate::value::GcCause::TaskBoundary, None);
        let live = crate::value::heap_live_objects(&heap);
        if std::env::var_os("TRUST_JS_GC_LOG").is_some() {
            eprintln!("[gc-task] live={live} reclaimed={} realms={} templates={} pins={} host_settings={} coroutines={}",
                before.saturating_sub(live), self.realms.len(), self.template_cache.len(),
                self.gc_pins.len(), self.host_settings_states.len(), self.generators.len());
        }
        before.saturating_sub(live)
    }

    pub fn suspend_gc(&mut self) {
        self.gc_heap
            .agent
            .suppressed
            .set(self.gc_heap.agent.suppressed.get().saturating_add(1));
    }

    pub fn resume_gc(&mut self) {
        self.gc_heap
            .agent
            .suppressed
            .set(self.gc_heap.agent.suppressed.get().saturating_sub(1));
    }

    /// Pin `o` for the lifetime of its side-table entries (see `gc_pins`).
    pub(crate) fn gc_pin(&mut self, o: &Gc) {
        let owner = Gc::as_ptr(o) as usize;
        // All callers install native metadata, a Realm group, or a template/namespace owner;
        // cache keys and weak targets use separate registries. A ShadowRealm interpreter's pin
        // is a conditional edge from its owner to the interpreter's place in the tree
        // (`Marking::trace_side_slots`), which a minor collection follows for the owners
        // recorded since the last one. Attaching new ownership to an OLD object is still a
        // changed conditional edge, including before child publication. A root interpreter has
        // no place to keep, so its pins add no edge and need no record: every RegExp object is
        // pinned, and recording each made every minor collection visit every one allocated
        // since the last. The side-slot tables' writes record their owners themselves.
        if !self.gc_heap_root() {
            self.gc_heap.records.borrow_mut().remember_owner(owner);
        }
        self.remember_metadata_owner(owner);
        self.gc_pins.insert(owner, *o);
    }

    /// Sweep one already-proven non-live set: atomic weak clearing and internal-slot eviction
    /// (ECMA-262 #sec-weakref-execution).
    pub(crate) fn gc_sweep_dead_objects(
        &mut self,
        garbage_objects: &crate::gc_sweep::DeadObjects<'_>,
        native: &crate::gc_native::NativeGcGraph,
        kind: crate::gc_core::CollectionKind,
        retired: &mut Vec<crate::agent_access::Owned<Interp>>,
    ) {
        self.host_state
            .sweep_gc_with_native(&|id| native.is_live(id));

        // Drop template entries owned by a dead Realm before the sweep releases its global.
        // Checking the Realm table as well as the dead objects handles stale pointer-keyed entries
        // without allowing an address-reused global to inherit an old template object.
        self.template_cache.retain(|(realm_key, _), value| {
            self.realms.contains_key(realm_key)
                && !garbage_objects.contains(*realm_key)
                && match (value).view() {
                    ValueView::Obj(object) => {
                        !garbage_objects.contains(Gc::as_ptr(&object) as usize)
                    }
                    _ => true,
                }
        });

        // ECMA-262 §9.9 performs all weak-target clearing atomically for the chosen non-live set.
        // Do this before any strong side-table edge is released, while every identity is stable.
        self.gc_clear_weak_targets(garbage_objects);
        self.gc_forget_weak_owners(garbage_objects);
        self.gc_drain_weak_deaths();
        self.gc_schedule_finalization_cleanup(Some(garbage_objects));

        // Evict the dead from pointer-keyed side tables, so that a future object reusing an
        // address can't inherit stale metadata.
        self.gc_sweep_native_realms(garbage_objects, kind);
        // Evict the chosen dead set in batches. Probing all internal-slot tables for each
        // ordinary function/prototype makes closure churn pay for unrelated platform state.
        // Small dead sets keep directed removal; dense churn scans each table only once.
        // Dead cells stay intact until the heap frees them after this sweep, and weak targets
        // were cleared above.
        self.gc_evict_dead_owners(garbage_objects, retired);
    }

    /// I8: every interpreter field, classified by what it keeps about objects that may die, so
    /// a new field does not compile until it is classified. A table keyed by an object's
    /// address evicts a dead object's entries here: a later object at the same address must
    /// not inherit them. Every other field keeps its keys alive, pins them weakly, is swept by
    /// an earlier step, or is not keyed by object addresses.
    fn gc_evict_dead_owners(
        &mut self,
        dead: &crate::gc_sweep::DeadObjects<'_>,
        retired: &mut Vec<crate::agent_access::Owned<Interp>>,
    ) {
        let _evicting = crate::gc_sweep::Evicting::enter();
        let Interp {
            // Keyed by an object's address: evicted below.
            class_info,
            eval_realm_fns,
            module_ns,
            realms,
            typed_arrays,
            data_views,
            regexps,
            proxies,
            host_indexed,
            promises,
            temporal,
            array_buffers,
            native_buffers,
            array_buffer_versions,
            array_buffer_dirty_ranges,
            ta_buffer,
            shadow_realms,
            immutable_buffers,
            host_keyed_buffers,
            generators,
            async_gens,
            async_gen_busy,
            async_gen_queue,
            mapped_arguments,
            deferred_ns,
            promise_forward,
            temporal_cal,
            gc_pins,
            // Swept by the steps above: host caches, Realm templates, [[Realm]] pins, weak targets
            // and weak owners.
            host_state: _,
            template_cache: _,
            native_function_realms: _,
            weak_refs: _,
            weak_collection_data: _,
            weak_collection_index: _,
            weak_metadata: _,
            finalization_registries: _,
            // Weak pins, pruned after the collection (gc_prune_dead_pins). A dead address a pin
            // still names stays reserved, so no new object takes it meanwhile.
            creation_pins: _,
            construct_ics: _,
            call_overflow: _,
            native_callback_cache: _,
            proxy_trap_caches: _,
            construct_capacity_hints,
            global_env_pins: _,
            htmldda: _,
            symbol_agent: _,
            shadow_home: _, // a weak owner
            // Proofs naming addresses through weak pins (reserved, as above).
            ta_accessor_protector: _,
            array_iteration_protector: _,
            // Keyed by values they keep alive (roots, string pins): no key dies while present.
            unhandled_rejections: _,
            host_settings_states: _,
            computed_reads: _,
            property_key_strings: _,
            // Keyed weakly; a dead key misses on lookup.
            fragment_cache: _,
            // Not keyed by object addresses: intrinsics and roots, execution state, counters, and
            // tables keyed by shapes, names, ids, source text or AST nodes.
            gc_heap: _,
            realm_id: _,
            global: _,
            global_env: _,
            object_proto: _,
            function_proto: _,
            array_proto: _,
            array_ctor: _,
            string_proto: _,
            number_proto: _,
            boolean_proto: _,
            symbol_proto: _,
            error_protos: _,
            console: _,
            wall_clock: _,
            runtime_interrupt: _,
            interrupt_poll_tick: _,
            ast_tick: _,
            strict: _,
            tier: _,
            tier_threshold: _,
            jit_admit_runs: _,
            vm_pool: _,
            native_arg_pool: _,
            stub_cache: _,
            stub_cache_names: _,
            arguments_shapes: _,
            descriptor_maps: _,
            iter_result_map: _,
            regexp_array_maps: _,
            enumeration_keys: _,
            frame_pool: _,
            call_site: _,
            code_frames: _,
            job_script_contexts: _,
            host_resolution_marker: _,
            jit_helpers: _,
            jit_layout: _,
            interp_layout: _,
            inline_ic_safe: _,
            regexp_protector: _,
            regexp_last: _,
            depth: _,
            direct_call_depth: _,
            proxy_forward_depth: _,
            eval_fn: _,
            elems_protector: _,
            species_protector: _,
            self_hosted: _,
            shared_snapshots: _,
            iterator_sym: _,
            wk_syms: _,
            short_circuit: _,
            import_meta: _,
            import_base: _,
            modules: _,
            module_recs: _,
            module_loader: _,
            dynamic_module_loader: _,
            routed_imports,
            extra_protos: _,
            can_block: _,
            agent: _,
            global_var_names: _,
            regexp_programs: _,
            new_target: _,
            pending_new_target: _,
            ctor_caller_realm: _,
            next_unhandled_rejection: _,
            handled_rejections: _,
            host_job_context: _,
            script_entry: _,
            host_entry_realm: _,
            capture_job_script_caller: _,
            caller_origin: _,
            retired_host_job_contexts: _,
            host_job_context_enter: _,
            host_job_context_leave: _,
            gc_next: _,
            gc_tick: _,
            constructing: _,
            super_call_ok: _,
            fn_frames: _,
            module_async_seq: _,
            yield_raw_result: _,
            in_async_gen_body: _,
            pending_fn_name: _,
            pending_tail: _,
            tco_ok: _,
            in_field_init_code: _,
            using_stack: _,
            accessor_seq: _,
            decorator_initializers: _,
            annexb_fn_sync: _,
            deferred_ns_objs: _,
            borrowed_parameters: _,
            module_source_objs: _,
            heap_release: _,
        } = self;
        class_info.evict_dead(dead);
        crate::gc_sweep::set(eval_realm_fns, dead);
        module_ns.evict_dead(dead);
        crate::gc_sweep::map(realms, dead);
        crate::gc_sweep::map(typed_arrays, dead);
        crate::gc_sweep::map(data_views, dead);
        crate::gc_sweep::map(regexps, dead);
        proxies.evict_dead(dead);
        host_indexed.evict_dead(dead);
        promises.evict_dead(dead);
        crate::gc_sweep::map(temporal, dead);
        crate::gc_sweep::map(array_buffers, dead);
        crate::gc_sweep::map(native_buffers, dead);
        crate::gc_sweep::map(array_buffer_versions, dead);
        crate::gc_sweep::map(array_buffer_dirty_ranges, dead);
        ta_buffer.evict_dead(dead);
        if !shadow_realms.is_empty() {
            let collected: Vec<usize> = shadow_realms
                .keys()
                .filter(|&&owner| dead.contains(owner))
                .copied()
                .collect();
            for owner in collected {
                if let Some(realm) = shadow_realms.remove(&owner) {
                    retired.push(realm);
                }
            }
        }
        crate::gc_sweep::set(immutable_buffers, dead);
        crate::gc_sweep::set(host_keyed_buffers, dead);
        generators.evict_dead(dead);
        crate::gc_sweep::set(async_gens, dead);
        crate::gc_sweep::set(async_gen_busy, dead);
        async_gen_queue.evict_dead(dead);
        routed_imports.evict_dead(dead);
        mapped_arguments.evict_dead(dead);
        construct_capacity_hints.evict_dead(dead);
        crate::gc_sweep::map(deferred_ns, dead);
        promise_forward.evict_dead(dead);
        crate::gc_sweep::map(temporal_cal, dead);
        crate::gc_sweep::map(gc_pins, dead);
    }

    /// Raw-address inline-cache maps carry weak allocation pins. Prune expired entries so the
    /// pin itself does not keep a dead cell reserved forever; a live cache identity remains
    /// ABA-safe because its pin still keeps that cell reserved.
    pub(crate) fn gc_prune_dead_pins(&mut self) {
        let creation_pins = self.creation_pins.len();
        self.creation_pins.retain(|_, pin| pin.strong_count() != 0);
        if self.creation_pins.len() != creation_pins {
            // A property-creation cache names its prototype by address and outlives the pin:
            // once the pin goes, a new object can take that address. Fence every creation
            // cache before the address can be reused.
            crate::value::bump_proto_epoch();
        }
        self.construct_ics
            .retain(|_, entry| entry.pin.strong_count() != 0);
        self.call_overflow.prune_dead();
        if let Some(cache) = &self.native_callback_cache {
            cache.prune_dead();
        }
        if let Some(caches) = &self.proxy_trap_caches {
            caches.prune_dead();
        }

        if self
            .global_env_pins
            .iter()
            .any(|pin| pin.strong_count() == 0)
        {
            // Call/construct/overflow entries can outlive a host settings environment,
            // especially when their callee is native or their AST serves another Realm.
            // Fence every raw genv comparison BEFORE releasing allocation-header pins.
            crate::bytecode::invalidate_call_caches();
            self.global_env_pins.retain(|pin| pin.strong_count() != 0);
        }
        self.htmldda.retain(|_, pin| pin.strong_count() != 0);
        self.symbol_agent
            .borrow_mut()
            .symbols
            .retain(|_, symbol| symbol.strong_count() != 0);
    }

    /// An explicit collection (a test or host request). Inlined, like `gc_heap_collect_explicit`,
    /// so that its stack scrub runs from the caller's frame: the collection's frames are then
    /// built on zeroed stack. A frame of their own, built first, keeps any slot it leaves
    /// unwritten as it was, and on x86-64 such slots held the address of a cell the caller's
    /// finished callee had used, which the scan took for a root.
    #[inline(always)]
    pub(crate) fn gc_collect(&mut self) {
        self.gc_heap_collect_explicit(crate::gc_core::CollectionKind::Major);
    }

    /// A minor collection, explicitly: for tests of what one keeps and frees (explicit requests
    /// are otherwise major). Inlined like `gc_collect`.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn gc_collect_minor(&mut self) {
        self.gc_heap_collect_explicit(crate::gc_core::CollectionKind::Minor);
    }

    /// Account for `len` bytes of new ArrayBuffer data, first collecting (majorly) once enough
    /// has been allocated since the last major collection that discarded buffers may be holding
    /// as much again. Called before the data block is allocated, where a collection is a
    /// safepoint like any allocation of the buffer object itself. A deferred collection keeps
    /// the debt for the next safepoint.
    pub(crate) fn note_buffer_allocation(&mut self, len: usize) {
        if !self.count_buffer_bytes(len) || self.gc_heap.agent.suppressed.get() != 0 {
            return;
        }
        self.gc_collect_with_cause(
            crate::value::GcCause::AllocationThreshold,
            Some(crate::gc_core::CollectionKind::Major),
        );
    }

    /// [`Interp::note_buffer_allocation`] for data made where a collection must not run: a
    /// host's bytes copied into a new buffer (an embedder may hold values outside its stack
    /// there), or a copy a built-in makes from bytes it has already read. Past the trigger it
    /// makes a major collection due at the next safepoint.
    pub(crate) fn note_buffer_allocation_deferred(&mut self, len: usize) {
        if self.count_buffer_bytes(len) {
            self.gc_heap
                .agent
                .request(&self.gc_heap, Some(crate::gc_core::CollectionKind::Major));
        }
    }

    /// Add `len` bytes to the Agent's buffer data since the last major collection: whether
    /// they passed the trigger.
    fn count_buffer_bytes(&self, len: usize) -> bool {
        let agent = &self.gc_heap.agent;
        let mut pressure = agent.buffers.get();
        pressure.since_gc = pressure.since_gc.saturating_add(len);
        agent.buffers.set(pressure);
        pressure.since_gc > pressure.trigger
    }

    /// After a major collection, start counting buffer data again, against a trigger that
    /// grows with the buffer data that survived it (whatever started the collection).
    pub(crate) fn reset_buffer_pressure(&mut self) {
        let retained = self.agent_retained_buffer_bytes();
        self.gc_heap.agent.buffers.set(BufferPressure {
            since_gc: 0,
            trigger: retained.max(BUFFER_GC_TRIGGER),
        });
    }

    /// Buffer data every interpreter of this Agent still holds. A surviving SharedArrayBuffer
    /// retains its Shared Data Block; each block counts once, however many buffers (in however
    /// many Realms) alias it.
    fn agent_retained_buffer_bytes(&mut self) -> usize {
        let mut local = 0usize;
        let mut blocks = crate::fasthash::FastSet::default();
        let access = self.gc_heap.agent.access.clone();
        self.with_raw_access(|_| {
            for (_, pointer) in access.snapshot() {
                // SAFETY: a shared visit after the requester published its current lineage;
                // nothing here runs JavaScript, collects or keeps a table reference.
                let interp = unsafe { &*pointer.as_ptr() };
                for data in interp.array_buffers.values() {
                    match data {
                        BufferData::Local(bytes) => {
                            local += bytes.try_borrow().map_or(0, |bytes| bytes.len());
                        }
                        BufferData::Shared(block) => {
                            blocks.insert(block.id());
                        }
                    }
                }
            }
        });
        let shared: usize = blocks
            .into_iter()
            .filter_map(shared_mem_get)
            .map(|block| block.lock().unwrap().len())
            .sum();
        local.saturating_add(shared)
    }

    /// Account for host-allocated data an ArrayBuffer can expose (a WebAssembly memory's
    /// initial or grown pages), exactly like an engine allocation of that many buffer bytes.
    pub(crate) fn note_external_buffer_growth(&mut self, len: usize) {
        if len != 0 {
            self.note_buffer_allocation(len);
        }
    }

    /// A data block could not be allocated: collect everything unreachable (discarded buffers
    /// included) before the caller retries once.
    pub(crate) fn collect_for_failed_allocation(&mut self) {
        self.gc_collect();
    }

    /// Collect for `cause`: a collection of `kind`, or with `None` of the kind the heap asks for
    /// (minor unless a major collection is due; see `Interp::gc_heap_collect`).
    pub(crate) fn gc_collect_with_cause(
        &mut self,
        cause: crate::value::GcCause,
        kind: Option<crate::gc_core::CollectionKind>,
    ) -> crate::agent_control::CollectionOutcome {
        match (cause, kind) {
            (crate::value::GcCause::Explicit, Some(kind)) => self.gc_heap_collect_explicit(kind),
            _ => self.gc_heap_collect(cause, kind),
        }
    }

    // ----- calling ----------------------------------------------------------------------------

    /// Call an already-selected callback. ECMA-262 OrdinaryGet/OrdinarySet and proxy [[Get]],
    /// [[Set]], [[Call]] still perform every descriptor/trap lookup before entering here. Only
    /// the immutable function identity/code proof is reusable; replacement/revocation, realm
    /// switching and parameter-instantiation fallbacks remain in the ordinary Call machinery.
    fn proxy_trap_caches(&mut self) -> Rc<ProxyTrapCaches> {
        self.proxy_trap_caches
            .get_or_insert_with(|| Rc::new(ProxyTrapCaches::new()))
            .clone()
    }

    /// GetMethod(handler, name) (ECMA-262 §7.3.10) for Proxy trap `kind`, through that trap's
    /// inline cache: `None` for an undefined/null trap, TypeError for a non-callable one.
    pub(crate) fn proxy_get_trap(
        &mut self,
        handler: &Value,
        kind: usize,
        name: &'static str,
    ) -> Result<Option<Value>, Abrupt> {
        let caches = self.proxy_trap_caches();
        {
            let key = PROXY_TRAP_NAMES[kind];
            debug_assert!(key.atom().is(name));
            let __value_match_original =
                self.get_prop_ic(handler, name, key.key(), &caches.lookups[kind][0])?;
            match (__value_match_original).view() {
                ValueView::Undefined | ValueView::Null => Ok(None),
                _ if {
                    let trap = __value_match_original;
                    trap.is_callable()
                } =>
                {
                    let trap = __value_match_original;
                    Ok(Some(trap))
                }
                _ => Err(self.throw("TypeError", format!("proxy '{name}' trap is not callable"))),
            }
        }
    }

    /// Call(trap, handler, args) for Proxy trap `kind`, through that trap's callback cache.
    pub(crate) fn proxy_call_trap<const N: usize>(
        &mut self,
        kind: usize,
        trap: &Value,
        handler: Value,
        args: [Value; N],
    ) -> Result<Value, Abrupt> {
        let caches = self.proxy_trap_caches();
        caches.calls[kind].call(self, trap, handler, args)
    }

    /// Whether an ordinary target (set `ic_plain`, no exotic tag) proves a Proxy invariant check
    /// for `key` vacuous: it has no own `key`, or only a configurable one (ECMA-262 §10.5.8 step
    /// 9 and §10.5.9 step 10 constrain non-configurable properties only). No descriptor copy.
    fn proxy_invariant_vacuous(target: &Value, key: &str) -> bool {
        let ValueView::Obj(target) = &(target).view() else {
            return false;
        };
        let body = target.borrow();
        body.ic_plain.get()
            && (body.exotic.is_ordinary() || matches!(body.exotic, Exotic::Array))
            && body.props.get(key).is_none_or(Property::configurable)
    }

    pub(crate) fn call_callback(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        if self.tier != crate::bytecode::Tier::Jit
            || !matches!(&(callee).view(), ValueView::Obj(o) if matches!(&o.borrow().call, Callable::User(_)))
        {
            return self.call(callee, this, args);
        }
        let cache = self
            .native_callback_cache
            .get_or_insert_with(|| Rc::new(crate::callback::CallbackCache::new()))
            .clone();
        cache.call_values(self, &callee, this, args)
    }

    /// Closed intrinsic iterator bodies cannot produce a proper-tail transfer of
    /// their own; any user getter they invoke goes through call() and drains it.
    /// Preserve the ordinary native-call activation, GC/interrupt and depth guards
    /// while returning the unmaterialized IteratorStepValue continuation directly.
    pub(crate) fn try_intrinsic_iterator_step(
        &mut self,
        iterator: &Value,
        next: &Value,
    ) -> Option<Result<Option<Value>, Abrupt>> {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED
            .get_or_init(|| std::env::var("TRUST_JS_ITERATOR_RESULTS").as_deref() != Ok("0"))
            || self.pending_tail.is_some()
        {
            return None;
        }
        let object = next.as_obj()?;
        let native = match object.borrow().call {
            Callable::Native(native) => native,
            _ => return None,
        };
        let step = crate::builtins::intrinsic_iterator_step(native)?;
        // `ic_plain` excludes a Proxy without a table lookup.
        if !self.ordinary_get_ptr(Gc::as_ptr(&object) as usize)
            || self.callee_realm_global(&object).is_some()
        {
            return None;
        }
        // A step that executes no ECMAScript code needs no activation of its own. It keeps the
        // amortized safepoints of the call it replaces: a drain loop may consist of nothing else.
        if let Err(error) = self
            .interrupt_poll()
            .and_then(|()| self.gc_check_amortized())
        {
            return Some(Err(error));
        }
        if let Some(value) = crate::builtins::inert_iterator_step(self, native, iterator) {
            return Some(Ok(value));
        }
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Some(Err(
                self.throw("RangeError", "Maximum call stack size exceeded")
            ));
        }
        if let Err(error) = self.gc_check_amortized() {
            self.depth -= 1;
            return Some(Err(error));
        }
        let saved_ctor = std::mem::replace(&mut self.constructing, false);
        let saved_target = std::mem::replace(&mut self.new_target, Value::Undefined);
        let result = with_execution_stack(self.depth, || {
            self.interrupt_poll_force()?;
            let started = crate::jit::perf_stage_start();
            let result = step(self, *iterator);
            crate::jit::perf_native_end(
                started,
                result.is_ok(),
                crate::jit::NativeLabelSrc::Addr(native as usize),
            );
            self.interrupt_poll_force()?;
            result.map_err(Abrupt::Throw)
        });
        self.constructing = saved_ctor;
        self.new_target = saved_target;
        self.depth -= 1;
        debug_assert!(self.pending_tail.is_none());
        Some(result)
    }

    pub(crate) fn call(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        crate::bytecode::note_call_path(crate::bytecode::CallPath::Uncached);
        // I1: callers often hold the argument list in a Rust-heap buffer.
        let _arguments = crate::rooting::root_slice(args);
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        if let Err(e) = self.gc_check_amortized() {
            self.depth -= 1;
            return Err(e);
        }
        let mut r = with_execution_stack(self.depth, || self.call_inner(callee, this, args));
        // Trampoline: a proper tail call unwound out of the callee re-dispatches here, in the
        // same stack frame, so mutual tail recursion runs in constant stack space.
        while r.is_ok() {
            match self.run_pending_tail(true) {
                Some(next) => r = next,
                None => break,
            }
        }
        self.depth -= 1;
        r
    }

    /// One step of a proper-tail-call trampoline: take the call a callee left pending, if any,
    /// and run it after a collection check (amortized or not, as the trampoline's call path
    /// checks). The argument list leaves `pending_tail`, which the collector traces, for a
    /// Rust-heap buffer the stack scan cannot see, so it is rooted from here through the check
    /// and the call (I1): a Proxy callee, say, runs its handler's `apply` getter before it copies
    /// the arguments out. The callee and receiver are locals of this frame.
    #[inline]
    pub(crate) fn run_pending_tail(&mut self, amortized: bool) -> Option<Result<Value, Abrupt>> {
        let (f, t, a) = *self.pending_tail.take()?;
        let _tail_arguments = crate::rooting::root_slice(&a);
        let checked = if amortized {
            self.gc_check_amortized()
        } else {
            self.gc_check()
        };
        Some(checked.and_then(|()| self.call_tail(f, t, &a)))
    }

    /// GetFunctionRealm: the realm-global pointer of `obj`, unwrapping bound functions and
    /// proxies (a revoked proxy is a TypeError). `None` means the active realm.
    pub(crate) fn get_function_realm_global(&mut self, obj: &Gc) -> Result<Option<usize>, Abrupt> {
        let mut cur = *obj;
        // ECMA-262 #sec-getfunctionrealm has no depth cutoff. Immutable bound
        // and proxy target links always lead toward an already-created object,
        // so this iterative traversal cannot form a target cycle.
        loop {
            if let Some((target, handler)) = self.proxy_at(Gc::as_ptr(&cur) as usize) {
                if matches!((handler).view(), ValueView::Null) {
                    return Err(self.throw("TypeError", "proxy has been revoked"));
                }
                match target.view() {
                    ValueView::Obj(t) => {
                        cur = t;
                        continue;
                    }
                    _ => return Ok(None),
                }
            }
            let bound = match &cur.borrow().call {
                Callable::Bound(bound) => Some(bound.target),
                _ => None,
            };
            match bound {
                Some(t) => cur = t,
                None => return Ok(self.callee_realm_global(&cur)),
            }
        }
    }

    /// Resume a proper tail call at the existing logical depth, but re-enter
    /// the native storage guard. The previous callee's stack segment has
    /// already been released; its continuation may make ordinary nested calls.
    /// ECMA-262 #sec-preparefortailcall reuses execution-context resources, not
    /// permission to execute the next callee on an exhausted native stack.
    pub(crate) fn call_tail(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        if execution_stack_exhausted(self.depth) {
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        with_execution_stack(self.depth, || self.call_inner(callee, this, args))
    }

    pub(crate) fn call_inner(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        // With one Realm and one interpreter, every callee already runs in the active Realm.
        let foreign_possible = self.multi_realm() || !self.single_interpreter();
        if let Some(object) = callee.as_obj().filter(|_| foreign_possible) {
            // Proxy and Bound abstract-operation ordering stays in the caller. Their eventual
            // trap/target call goes through this same actual callable-Realm funnel.
            if self.proxy_at(Gc::as_ptr(&object) as usize).is_none() {
                if let Some((owner, realm)) = self.callable_home(&object) {
                    if owner != self.realm_id {
                        return self
                            .with_route(owner, |target| target.call_inner(callee, this, args));
                    }
                    if realm != Gc::as_ptr(&self.global) as usize {
                        let state = self.realms[&realm].snapshot_clone();
                        return self.with_realm_snapshot(realm, state, |target| {
                            target.call_dispatch(callee, this, args)
                        });
                    }
                }
            }
        }
        self.call_dispatch(callee, this, args)
    }

    /// Only actual User/native and wrapped callables own a direct [[Realm]]. Bound/Proxy target
    /// unwrapping belongs to their abstract operations, not an eager owner switch.
    pub(crate) fn callable_home(
        &self,
        object: &Gc,
    ) -> Option<(crate::agent_access::RealmId, usize)> {
        let realm = {
            let body = object.borrow();
            match &body.call {
                Callable::User(user) => user.realm,
                // ShadowRealm WrappedFunctionCreate gives a wrapper its caller Realm's
                // %Function.prototype%; that Realm is the wrapper's [[Realm]] (one of ours).
                Callable::WrappedShadow(_) | Callable::WrappedCross(_) => {
                    let realm = self
                        .wrapper_prototype_realm(body.proto)
                        .unwrap_or(Gc::as_ptr(&self.global) as usize);
                    return Some((self.realm_id, realm));
                }
                call if call.is_native() => {
                    let key = Gc::as_ptr(object) as usize;
                    let owner = self.metadata_owner(key, |interp, key| {
                        interp.native_function_realms.borrow().contains_key(&key)
                    })?;
                    self.with_realm_view(owner, |interp| {
                        interp
                            .native_function_realms
                            .borrow()
                            .get(&key)
                            .map(|(_, realm)| *realm)
                    })?
                }
                _ => return None,
            }
        };
        // The active Realm is this interpreter's own: no table probe on the common path.
        if realm == Gc::as_ptr(&self.global) as usize {
            return Some((self.realm_id, realm));
        }
        let owner = self.metadata_owner(realm, |interp, key| interp.realms.contains_key(&key))?;
        Some((owner, realm))
    }

    /// The realm (keyed by its global's pointer) a function belongs to, when it is NOT the active
    /// realm. A user function's immutable [[Realm]] is recorded at creation; its lexical
    /// environment and mutable [[Prototype]] do not determine that identity. Other callables
    /// retain their existing native registration/prototype lookup.
    pub(crate) fn callee_realm_global(&self, obj: &Gc) -> Option<usize> {
        // One Realm and one interpreter: every callable belongs to the active Realm.
        if !self.multi_realm() && self.single_interpreter() {
            return None;
        }
        if let Some((_, realm)) = self.callable_home(obj) {
            return (realm != Gc::as_ptr(&self.global) as usize).then_some(realm);
        }
        self.wrapper_prototype_realm(obj.borrow().proto)
            .filter(|&realm| realm != Gc::as_ptr(&self.global) as usize)
    }

    /// The Realm whose %Function.prototype% a callable's prototype chain reaches first.
    fn wrapper_prototype_realm(&self, proto: Option<Gc>) -> Option<usize> {
        let mut cur = proto;
        let mut hops = 0;
        while let Some(p) = cur {
            if Gc::ptr_eq(&p, &self.function_proto) {
                return Some(Gc::as_ptr(&self.global) as usize);
            }
            for (g, rs) in &self.realms {
                if Gc::ptr_eq(&p, &rs.function_proto) {
                    return Some(*g);
                }
            }
            hops += 1;
            if hops > 8 {
                break;
            }
            cur = p.borrow().proto;
        }
        None
    }

    fn call_dispatch(
        &mut self,
        callee: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        let obj = match &(callee).view() {
            ValueView::Obj(o) => *o,
            _ => {
                // Optional call-boundary context complements the error stack
                // when an obfuscated interpreter attempts to call a primitive.
                // Inspect only native primitive data: no author coercion/hooks.
                static TRACE_CALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                static TRACE_COUNT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                if *TRACE_CALL.get_or_init(|| {
                    std::env::var_os("TRUST_JS_TRACE_ERROR_SOURCE").is_some()
                        && std::env::var("TRUST_JS_TRACE_ERRORS")
                            .is_ok_and(|filter| filter == "1" || filter.contains("not a function"))
                }) && TRACE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32
                {
                    let preview = match &(callee).view() {
                        ValueView::Str(text) => text.text().chars().take(160).collect::<String>(),
                        _ => type_name(&callee).to_string(),
                    };
                    eprintln!(
                        "trust-js-call-error: callee={preview:?} this-type={} argument-types={:?}",
                        type_name(&this),
                        args.iter().take(8).map(type_name).collect::<Vec<_>>()
                    );
                }
                return Err(self.throw(
                    "TypeError",
                    format!("{} is not a function", type_name(&callee)),
                ));
            }
        };
        // A call is a safepoint on the shared amortized interruption cadence, so loop-free
        // recursion in the interpreter and bytecode tiers stays abortable (HTML §8.1.4.5).
        self.interrupt_poll()?;
        // Proxy with an `apply` trap (or forward to the target).
        {
            if let Some((target, handler)) = self.proxy_at(Gc::as_ptr(&obj) as usize) {
                // ProxyCreate step 5 gives a proxy [[Call]] only when its target is callable,
                // and Call step 2 throws before any trap is read (ECMA-262, local e28783d).
                if matches!(obj.borrow().call, Callable::None) {
                    return Err(self.throw("TypeError", "value is not a function"));
                }
                if matches!((handler).view(), ValueView::Null) {
                    return Err(self.throw("TypeError", "proxy is revoked"));
                }
                let Some(trap) = self.proxy_get_trap(&handler, PROXY_TRAP_APPLY, "apply")? else {
                    return self.call(target, this, args);
                };
                let arr = self.make_array(args.to_vec());
                return self.proxy_call_trap(PROXY_TRAP_APPLY, &trap, handler, [target, this, arr]);
            }
        }
        // An indirect call to another realm's `eval` runs in that realm's global scope. (A direct
        // `eval(...)` of the current realm is intercepted earlier, in `eval_call`.) Only a
        // native function can be a realm's `eval` — skip the property lookup for user calls,
        // which is every hot call (the main realm always registers, so the map is never empty).
        if self.multi_realm() && self.eval_realm_fns.contains(&(Gc::as_ptr(&obj) as usize)) {
            let realm_g = obj
                .borrow()
                .internal_slots()
                .get("#\u{0}eval_realm")
                .map(|p| p.value());
            if let Some(realm_g) = realm_g.filter(|value| value.as_obj().is_some()) {
                return match args.first() {
                    Some(value) => match value.view() {
                        ValueView::Str(s) => self.indirect_eval_in_realm(&realm_g, &s.text()),
                        _ => Ok(*value),
                    },
                    None => Ok(Value::Undefined),
                };
            }
        }

        let call = obj.borrow().call.clone();
        // A plain call is never constructing (only `new` sets the flag). Clearing it here keeps a
        // wrapper constructor invoked as a function — `Number(x)` — from boxing. `new.target` is
        // likewise cleared so a native constructor called *as a function* during an outer `new`
        // (e.g. `new F(){ Function('a','b') }`) doesn't inherit the outer new.target.
        let saved_ctor = self.constructing;
        let saved_nt = self.new_target;
        self.constructing = false;
        // An arrow inherits the enclosing new.target lexically — don't clear it for one.
        if !matches!(&call, Callable::User(user) if user.func.is_arrow) {
            self.new_target = Value::Undefined;
        }
        let r = match call {
            Callable::None => Err(self.throw("TypeError", "value is not a function")),
            Callable::Native(_) | Callable::NativeData(_) => {
                self.dispatch_native(&call, this, args)
            }
            #[cfg(feature = "embed")]
            Callable::ScopedNative(_) => self.dispatch_native(&call, this, args),
            Callable::User(user) => {
                // A class constructor cannot be [[Call]]ed. (Empty-map guard: this runs on every
                // single call, and most programs define no classes.)
                if obj.borrow().class_kind != 0 {
                    return Err(self.throw(
                        "TypeError",
                        "Class constructor cannot be invoked without 'new'",
                    ));
                }
                self.call_user(&user.func, user.env, this, args, false, &obj)
            }
            Callable::Bound(bound) => {
                let mut all = bound.args.clone();
                all.extend_from_slice(args);
                // ECMA-262 BoundFunction [[Call]] still forwards the original
                // concatenated argument list once. The optional history records
                // only bounded native previews, without changing author APIs.
                let trace = trace_bound_call_enter(&bound, args);
                if let Some(id) = trace {
                    trace_bound_array_state(id, "enter", bound.args.first());
                }
                let result = self.call(Value::Obj(bound.target), bound.this, &all);
                if let Some(id) = trace {
                    trace_bound_array_state(id, "exit", bound.args.first());
                    trace_bound_call_exit(id, &result);
                }
                result
            }
            Callable::WrappedShadow(shadow) => {
                // The call's own handle, as for native data (`dispatch_native`).
                let shadow = Rc::clone(&shadow);
                self.call_wrapped_shadow(&shadow.realm, *shadow.target, args)
            }
            Callable::WrappedCross(cross) => {
                let target = cross.target;
                // A cross wrapper runs only in its own realm, entered from its parent.
                let (realm, parent) = {
                    assert!(
                        self.shadow_entered(),
                        "a cross-realm wrapper runs in its entered ShadowRealm"
                    );
                    let home = self
                        .shadow_home
                        .as_ref()
                        .expect("a ShadowRealm knows its parent");
                    let realm = home
                        .owner
                        .upgrade()
                        .expect("an entered ShadowRealm's object is alive");
                    (realm, home.parent)
                };
                if std::env::var("TRUST_JS_DBG").is_ok() {
                    eprintln!(
                        "DBG cross enter realm={:p} parent={parent:?} self={:p} tkind={}",
                        Gc::as_ptr(&realm),
                        self as *const Interp,
                        match &(target).view() {
                            ValueView::Obj(o) => match &o.borrow().call {
                                Callable::User(_) => "user",
                                Callable::WrappedShadow(_) => "wshadow",
                                Callable::WrappedCross(_) => "wcross",
                                native if native.is_native() => "native",
                                _ => "other",
                            },
                            _ => "nonobj",
                        }
                    );
                }
                for a in args {
                    if matches!(&(a).view(), ValueView::Obj(_)) && !a.is_callable() {
                        return Err(self.throw(
                            "TypeError",
                            "wrapped function arguments must be primitives or callables",
                        ));
                    }
                }
                let host_result =
                    self.with_route(parent, |host| -> Result<Result<Value, Abrupt>, ()> {
                        // Wrapping copies name/length, which can run getters (I1).
                        let mut out_args = crate::rooting::RootedVec::new();
                        for a in args {
                            if a.is_callable() {
                                out_args
                                    .push(host.make_wrapped_shadow(&realm, *a).map_err(|_| ())?);
                            } else {
                                out_args.push(*a);
                            }
                        }
                        Ok(host.call(target, Value::Undefined, &out_args))
                    });
                let host_result = match host_result {
                    Ok(result) => result,
                    Err(()) => {
                        return Err(self.throw(
                            "TypeError",
                            "cannot wrap the callable argument for the host realm",
                        ))
                    }
                };
                match host_result {
                    Ok(v) if !matches!((v).view(), ValueView::Obj(_)) => Ok(v),
                    Ok(v) if v.is_callable() => Ok(self.make_wrapped_cross(v)),
                    Ok(_) => {
                        Err(self.throw("TypeError", "a wrapped function returned a non-primitive"))
                    }
                    Err(e) => {
                        if std::env::var("TRUST_JS_DBG").is_ok() {
                            eprintln!(
                                "DBG cross err: {:?}",
                                match &e {
                                    Abrupt::Throw(value) if value.as_engine_text().is_some() =>
                                        format!("str {}", value.as_engine_text().unwrap()),
                                    Abrupt::Throw(value) if value.as_obj().is_some() => format!(
                                        "obj {:?}",
                                        value
                                            .as_obj()
                                            .unwrap()
                                            .borrow()
                                            .props
                                            .get("message")
                                            .map(|p| p.value())
                                            .and_then(|v| match (v).view() {
                                                ValueView::Str(s) => Some(s.to_string()),
                                                _ => None,
                                            })
                                    ),
                                    _ => "non-throw".to_string(),
                                }
                            );
                        }
                        Err(self.throw("TypeError", "a wrapped function threw in the host realm"))
                    }
                }
            }
            // Auto-accessor get/set: the receiver must *own* the private backing field (so a static
            // accessor reached through a subclass, which doesn't carry the slot, throws).
            Callable::AccessorGet(key) => self.accessor_load(&this, &key),
            Callable::AccessorSet(key) => self.accessor_store(
                &this,
                &key,
                args.first().cloned().unwrap_or(Value::Undefined),
            ),
            // Decorator `context.access` get/set: read/write a named property on the first argument.
            // The key lives in the callee's payload cell; a getter or setter may let the callee
            // die, so each holds its own count while it runs.
            Callable::PropGet(key) => {
                let key = Rc::clone(&key);
                let recv = args.first().cloned().unwrap_or(Value::Undefined);
                self.get_member_key(&recv, &key)
            }
            Callable::PropSet(key) => {
                let key = Rc::clone(&key);
                let recv = args.first().cloned().unwrap_or(Value::Undefined);
                let val = args.get(1).cloned().unwrap_or(Value::Undefined);
                self.set_member_key(&recv, &key, val)
                    .map(|_| Value::Undefined)
            }
            Callable::PrivatePropGet(key) => {
                let key = Rc::clone(&key);
                let recv = args.first().cloned().unwrap_or(Value::Undefined);
                self.get_private_member(&recv, &key)
            }
            Callable::PrivatePropSet(key) => {
                let key = Rc::clone(&key);
                let recv = args.first().cloned().unwrap_or(Value::Undefined);
                let val = args.get(1).cloned().unwrap_or(Value::Undefined);
                self.set_private_member(&recv, &key, val)
                    .map(|_| Value::Undefined)
            }
        };
        self.constructing = saved_ctor;
        self.new_target = saved_nt;
        r
    }

    /// Read an auto-accessor's backing field off `this`, throwing if `this` lacks the slot.
    fn accessor_load(&mut self, this: &Value, key: &str) -> Result<Value, Abrupt> {
        match &(this).view() {
            ValueView::Obj(o) if o.borrow().internal_slots().contains(key) => Ok(o
                .borrow()
                .internal_slots()
                .get(key)
                .map(|p| p.value())
                .unwrap()),
            _ => Err(self.throw(
                "TypeError",
                "cannot read auto-accessor backing field from an unrelated object",
            )),
        }
    }

    /// Write an auto-accessor's backing field on `this`, throwing if `this` lacks the slot.
    fn accessor_store(&mut self, this: &Value, key: &str, value: Value) -> Result<Value, Abrupt> {
        match &(this).view() {
            ValueView::Obj(o) if o.borrow().internal_slots().contains(key) => {
                if let Some(p) = o.borrow_mut().internal_slots_mut().get_mut(key) {
                    p.set_value(value);
                }
                Ok(Value::Undefined)
            }
            _ => Err(self.throw(
                "TypeError",
                "cannot write auto-accessor backing field on an unrelated object",
            )),
        }
    }

    /// Marshal an evaluate/importValue result out of the realm of the ShadowRealm object `realm`.
    pub(crate) fn make_shadow_result(&mut self, realm: &Gc, v: Value) -> Result<Value, Abrupt> {
        self.marshal_from_shadow(realm, v)
    }

    /// Marshal one value crossing OUT of a shadow realm into this (host) realm.
    fn marshal_from_shadow(&mut self, realm: &Gc, v: Value) -> Result<Value, Abrupt> {
        if !matches!((v).view(), ValueView::Obj(_)) {
            return Ok(v);
        }
        if v.is_callable() {
            return self.make_wrapped_shadow(realm, v);
        }
        Err(self.throw(
            "TypeError",
            "only primitives and callables may cross a ShadowRealm boundary",
        ))
    }

    /// Make a ShadowRealm wrapped function: a caller-realm callable around `target` in the realm
    /// of the ShadowRealm object `realm`. CopyNameAndLength copies the target's `length` (a
    /// non-negative integer) and `name` (a string, else "") — a throwing `length`/`name` getter
    /// propagates.
    pub(crate) fn make_wrapped_shadow(
        &mut self,
        realm: &Gc,
        target: Value,
    ) -> Result<Value, Abrupt> {
        let f = Object::new(Some(self.function_proto));
        f.borrow_mut().call = Callable::wrapped_shadow(*realm, target);
        // CopyNameAndLength: the Gets run in the realm that owns `target` (its proxy state
        // lives there); an abrupt Get is a TypeError of the calling realm.
        let bad = |i: &mut Interp| {
            i.throw(
                "TypeError",
                "wrapped function name/length are not accessible",
            )
        };
        let (length_r, name_r) = self
            .with_shadow(realm, |sub| {
                let l = match crate::builtins::has_own_property_trapped(sub, &target, "length") {
                    Ok(true) => sub.get_member_key(&target, "length"),
                    Ok(false) => Ok(Value::Num(0.0)),
                    Err(e) => Err(Abrupt::Throw(e)),
                };
                let n = sub.get_member_key(&target, "name");
                (l, n)
            })
            .unwrap_or_else(|| {
                (
                    self.get_member_key(&target, "length"),
                    self.get_member_key(&target, "name"),
                )
            });
        let length = match (length_r).map(|__value| (__value).view()) {
            Ok(ValueView::Num(n)) if n.is_finite() => n.trunc().max(0.0),
            Ok(ValueView::Num(n)) if n == f64::INFINITY => f64::INFINITY,
            Ok(_) => 0.0,
            Err(_) => return Err(bad(self)),
        };
        let name = match (name_r).map(|__value| (__value).view()) {
            Ok(ValueView::Str(s)) => s.to_string(),
            Ok(_) => String::new(),
            Err(_) => return Err(bad(self)),
        };
        {
            let mut b = f.borrow_mut();
            b.props.insert(
                "length",
                Property::data(Value::Num(length), false, false, true),
            );
            b.props.insert(
                "name",
                Property::data(Value::lstr(name), false, false, true),
            );
        }
        Ok(Value::Obj(f))
    }

    /// Call a ShadowRealm wrapped function: marshal primitive args into the sub-realm, call `target`
    /// there, and marshal the primitive (or further-wrapped callable) result back.
    fn call_wrapped_shadow(
        &mut self,
        realm: &Gc,
        target: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        for a in args {
            if matches!(&(a).view(), ValueView::Obj(_)) && !a.is_callable() {
                return Err(self.throw(
                    "TypeError",
                    "ShadowRealm wrapped function: arguments must be primitives or callables",
                ));
            }
        }
        // Primitive arguments cross directly; callables wrap as sub-realm functions that
        // re-enter this realm; other objects throw.
        let Some(result) = self.with_shadow(realm, |sub| {
            let mut inner_args = crate::rooting::RootedVec::new();
            for a in args {
                inner_args.push(if a.is_callable() {
                    sub.make_wrapped_cross(*a)
                } else {
                    *a
                });
            }
            sub.call(target, Value::Undefined, &inner_args)
        }) else {
            return Err(self.throw("TypeError", "the ShadowRealm is no longer available"));
        };
        match result {
            Ok(v) => self
                .marshal_from_shadow(realm, v)
                .map_err(|_| {
                    abrupt_value(
                        self.throw("TypeError", "a wrapped function returned a non-primitive"),
                    )
                })
                .map_err(Abrupt::Throw),
            Err(e) => {
                if std::env::var("TRUST_JS_DBG").is_ok() {
                    eprintln!(
                        "DBG cws err: {:?}",
                        match &e {
                            Abrupt::Throw(value) if value.as_engine_text().is_some() =>
                                format!("str {}", value.as_engine_text().unwrap()),
                            Abrupt::Throw(value) if value.as_obj().is_some() => format!(
                                "obj {:?} {:?}",
                                value
                                    .as_obj()
                                    .unwrap()
                                    .borrow()
                                    .props
                                    .get("name")
                                    .map(|p| p.value())
                                    .and_then(|v| match (v).view() {
                                        ValueView::Str(s) => Some(s.to_string()),
                                        _ => None,
                                    }),
                                value
                                    .as_obj()
                                    .unwrap()
                                    .borrow()
                                    .props
                                    .get("message")
                                    .map(|p| p.value())
                                    .and_then(|v| match (v).view() {
                                        ValueView::Str(s) => Some(s.to_string()),
                                        _ => None,
                                    })
                            ),
                            _ => "non-throw".to_string(),
                        }
                    );
                }
                Err(self.throw(
                    "TypeError",
                    "a wrapped function threw inside the ShadowRealm",
                ))
            }
        }
    }

    /// Create the sub-realm-side wrapper for a host callable.
    pub(crate) fn make_wrapped_cross(&mut self, target: Value) -> Value {
        let f = Object::new(Some(self.function_proto));
        f.borrow_mut().call = Callable::wrapped_cross(target);
        f.borrow_mut().props.insert(
            "length",
            Property::data(Value::Num(0.0), false, false, true),
        );
        f.borrow_mut()
            .props
            .insert("name", Property::data(Value::lstr(""), false, false, true));
        Value::Obj(f)
    }

    /// Build the `arguments` exotic object for an activation: indexed own props + configurable
    /// `length`, `@@iterator` = Array.prototype.values. An unmapped one (strict function OR
    /// non-simple parameter list) exposes `callee` as the %ThrowTypeError% poison accessor; a
    /// mapped one aliases still-mapped indices to the parameter bindings in `scope` (see
    /// `mapped_arguments`) and carries the function as `callee`.
    pub(crate) fn make_arguments_object(
        &mut self,
        func: &Rc<Function>,
        args: &[Value],
        scope: &Env,
        fn_obj: &Gc,
    ) -> Gc {
        // ECMA-262 CreateUnmappedArgumentsObject / CreateMappedArgumentsObject (local snapshot
        // e28783d5, spec.html:15059): "length", the indices, then @@iterator as the intrinsic
        // %Array.prototype.values% (not the current property value) and "callee".
        let ao = Object::new(Some(self.object_proto));
        let packed = {
            let mut object = ao.borrow_mut();
            object.exotic = crate::value::Exotic::Arguments;
            // Create{Unmapped,Mapped}ArgumentsObject defines a contiguous run of
            // ordinary indexed data properties. Use the same packed storage as
            // fresh arrays without changing the arguments object's exotic kind,
            // length descriptor, or (for sloppy calls) ParameterMap semantics.
            object.props.reserve_dense_exact(args.len(), false);
            for value in args {
                object.props.push_dense(Property::plain(*value));
            }
            args.is_empty() || object.props.has_packed_elements()
        };
        let simple_params = func
            .params
            .iter()
            .all(|p| !p.rest && p.default.is_none() && matches!(p.pattern, Pattern::Ident(_)));
        let unmapped = func.is_strict || !simple_params;
        let length = Property::data(Value::Num(args.len() as f64), true, false, true);
        let iterator = self
            .extra_protos
            .get("%Array.prototype.values%")
            .cloned()
            .map(|values| Property::builtin(Value::Obj(values)));
        let callee = if unmapped {
            self.extra_protos
                .get("%ThrowTypeError%")
                .cloned()
                .map(|thrower| {
                    Property::accessor_prop(
                        Some(Value::Obj(thrower)),
                        Some(Value::Obj(thrower)),
                        false,
                        false,
                    )
                })
        } else {
            Some(Property::data(Value::Obj(*fn_obj), true, false, true))
        };
        let shapes = if packed && iterator.is_some() && callee.is_some() {
            self.arguments_named_shapes()
        } else {
            None
        };
        {
            let mut object = ao.borrow_mut();
            match (shapes, iterator, callee) {
                (Some((keys, shapes, accessor_callee_shape)), Some(iterator), Some(callee)) => {
                    // Known-absent keys landing on the recorded child shapes: no existence scan,
                    // index parse, or transition lookup. The callee child shape records its
                    // descriptor class, so the poison accessor and the data callee diverge.
                    let callee_shape = if callee.accessor() {
                        accessor_callee_shape
                    } else {
                        shapes[2]
                    };
                    let [length_key, iterator_key, callee_key] = keys;
                    object.props.append_new(length_key, length, shapes[0]);
                    object.props.append_new(iterator_key, iterator, shapes[1]);
                    object.props.append_new(callee_key, callee, callee_shape);
                }
                (_, iterator, callee) => {
                    object.props.insert("length", length);
                    if let (Some(sym), Some(iterator)) = (self.iterator_sym, iterator) {
                        object.props.insert(Self::sym_key(&sym), iterator);
                    }
                    if let Some(callee) = callee {
                        object.props.insert("callee", callee);
                    }
                }
            }
        }
        if !unmapped {
            // Mapped: indices alias the parameter bindings (which live in `scope`). With
            // duplicate parameter names only the LAST occurrence is mapped — earlier indices
            // stay plain data properties holding the original argument values.
            let mut names = vec![None; func.params.len().min(args.len())];
            let mut seen: Vec<String> = Vec::new();
            // CreateMappedArgumentsObject walks ALL formals before checking index < argc.
            // A missing final duplicate still prevents an earlier supplied index from mapping.
            for (index, parameter) in func.params.iter().enumerate().rev() {
                let Pattern::Ident(name) = &parameter.pattern else {
                    unreachable!("mapped arguments require simple formals")
                };
                if !seen.contains(name) {
                    seen.push(name.clone());
                    if let Some(slot) = names.get_mut(index) {
                        *slot = Some(name.clone());
                    }
                }
            }
            if names.iter().any(Option::is_some) {
                self.gc_pin(&ao);
                self.mapped_arguments.insert(
                    &self.gc_heap,
                    Gc::as_ptr(&ao) as usize,
                    (*scope, names),
                );
            }
        }
        ao
    }

    /// The recorded shape chain for an arguments object's `length`, @@iterator and `callee`
    /// properties (see `Interp::arguments_shapes`), derived on first use. The third return
    /// value is the callee child shape when `callee` is the strict poison ACCESSOR: since
    /// shapes encode descriptor class, the data probe's chain must not be reused for it.
    fn arguments_named_shapes(&mut self) -> Option<([crate::atom::PropKey; 3], [u32; 3], u32)> {
        if self.arguments_shapes.is_none() {
            let iterator = self.iterator_sym?;
            let keys = [
                crate::atom::names::LENGTH.key(),
                crate::atom::PropKey::from_text(&Self::sym_key(&iterator)),
                crate::atom::names::CALLEE.key(),
            ];
            let mut probe = Props::new();
            let mut shapes = [0u32; 3];
            for (index, &key) in keys.iter().enumerate() {
                probe.insert(key, Property::data(Value::Undefined, true, false, true));
                shapes[index] = probe.shape();
            }
            // Replay the shared prefix, then take the accessor transition for `callee`.
            let mut strict_probe = Props::new();
            for &key in &keys[..2] {
                strict_probe.insert(key, Property::data(Value::Undefined, true, false, true));
            }
            strict_probe.insert(keys[2], Property::accessor_prop(None, None, false, false));
            let accessor_callee_shape = strict_probe.shape();
            if !shapes
                .iter()
                .chain([&accessor_callee_shape])
                .all(|&shape| crate::value::is_cacheable_shape(shape))
            {
                return None;
            }
            self.arguments_shapes = Some((keys, shapes, accessor_callee_shape));
        }
        self.arguments_shapes
    }

    /// Materialize the dedicated unmapped arguments slot of a compiled function. The active
    /// [`FnFrame`] supplies the observable `callee` identity for both ordinary and activation-aware
    /// moved-frame entries.
    pub(crate) fn make_compiled_arguments_object(&mut self, args: &[Value], scope: &Env) -> Gc {
        let fn_obj = self
            .fn_frames
            .last()
            .expect("compiled arguments object requires an active function frame")
            .callee();
        let func = match &fn_obj.borrow().call {
            Callable::User(user) => user.func.clone(),
            _ => unreachable!("compiled arguments object belongs to a user function"),
        };
        self.make_arguments_object(&func, args, scope, &fn_obj)
    }

    /// OrdinaryCallBindThis for a compiled body, computed only when the body reads `this`. A
    /// construct's `this` is the fresh instance — bound directly, never coerced; a sloppy plain
    /// call boxes a primitive / substitutes the global for undefined/null.
    fn bind_compiled_this(
        &mut self,
        func: &Rc<Function>,
        chunk: &crate::bytecode::Chunk,
        this: Value,
        is_construct: bool,
    ) -> Value {
        if !chunk.needs_frame_this() {
            return Value::Undefined;
        }
        if func.is_strict || is_construct {
            return this;
        }
        match this.view() {
            ValueView::Undefined | ValueView::Null => Value::Obj(self.global),
            ValueView::Obj(_) => this,
            _ => crate::builtins::box_primitive_pub(self, this),
        }
    }

    /// OrdinaryCallBindThis on a moved operand word, for a compiled callee that a
    /// moved entry runs (`jit::run_moved` and its forms take a `Value`): Undefined when the
    /// body never reads `this` (`uses_this`), the word itself for a strict callee, and for a
    /// sloppy one the global object for undefined or null, an object word unchanged, and a
    /// primitive boxed into an object value.
    fn bind_this_word(&mut self, this: Value, uses_this: bool, strict: bool) -> Value {
        if !uses_this {
            Value::Undefined
        } else if strict || this.is_object() {
            this
        } else if this.is_nullish() {
            Value::Obj(self.global)
        } else {
            crate::builtins::box_primitive_pub(self, this)
        }
    }

    /// Whether `chunk` may run (or be compiled to) native code on this run. A loop-free body
    /// first runs `jit_admit_runs` times on the bytecode VM (see [`Interp::jit_admit_runs`]);
    /// a forced tier (`tier_threshold` 0) admits at once.
    fn native_admitted(&self, func: &Function, chunk: &crate::bytecode::Chunk) -> bool {
        if !chunk.jit.untried()
            || self.tier_threshold == 0
            || func.scan_flags() & crate::ast::SCAN_HAS_LOOP != 0
        {
            return true;
        }
        let runs = chunk.vm_runs.get();
        if runs >= self.jit_admit_runs {
            return true;
        }
        chunk.vm_runs.set(runs + 1);
        false
    }

    /// Run an already-compiled chunk (JIT or bytecode VM), saving/restoring the per-body engine
    /// flags (`strict`, `tco_ok`, field-init / async-gen-body markers). Shared by the lean
    /// [`call_user_inner`] path and the [`call_compiled_fast`] shortcut so they can't diverge.
    /// `this_val` is already OrdinaryCallBindThis-bound; `new_target`/`constructing`/frame
    /// bookkeeping is the caller's job.
    fn run_compiled_chunk(
        &mut self,
        func: &Rc<Function>,
        chunk: &Rc<crate::bytecode::Chunk>,
        closure: &Env,
        this_val: Value,
        args: &[Value],
        is_construct: bool,
    ) -> Result<Value, Abrupt> {
        let saved_strict = std::mem::replace(&mut self.strict, func.is_strict);
        let saved_tco = std::mem::replace(&mut self.tco_ok, func.is_strict && !is_construct);
        let saved_field_init = self.in_field_init_code;
        let saved_agb = self.in_async_gen_body;
        if !func.is_arrow {
            self.in_field_init_code = false;
            self.in_async_gen_body = false;
        }
        // Capacity pressure is retryable. Reclamation invalidates raw ICs first
        // and preserves every active native frame and owned Rust code lease.
        let mut jit_code = None;
        // A static snapshot's run-once bootstrap function (`Function::run_once`) runs once per
        // Realm: the bytecode VM runs it about as fast as native code would, without making
        // every Agent keep its executable code.
        if matches!(self.tier, crate::bytecode::Tier::Jit)
            && !func.run_once
            && self.native_admitted(func, chunk)
        {
            let permit = if chunk.jit.ready_to_compile()
                && crate::jit::ensure_executable_capacity(chunk.jit_budget_wait_bytes.get())
            {
                chunk.jit.begin_compile()
            } else {
                None
            };
            if let Some(permit) = permit {
                let layout = *self
                    .jit_layout
                    .get_or_init(|| crate::value::jit_layout(&self.object_proto));
                if !self.interp_layout.get().valid {
                    let l = interp_layout(self);
                    self.interp_layout.set(l);
                }
                match crate::jit::compile_profiled(chunk, &layout, &self.interp_layout.get()) {
                    crate::jit::JitCompileOutcome::Compiled(code) => {
                        if env_flag!("TRUST_JS_TIER_LOG") {
                            eprintln!(
                                "[tier] native: {} ops, {} code bytes, loop {}, after {} VM runs",
                                chunk.jit_ops().len(),
                                code.code_bytes(),
                                func.scan_flags() & crate::ast::SCAN_HAS_LOOP != 0,
                                chunk.vm_runs.get()
                            );
                        }
                        chunk.jit_budget_wait_bytes.set(0);
                        permit.commit(Some(std::rc::Rc::new(code)));
                    }
                    crate::jit::JitCompileOutcome::Deferred { required_bytes } => {
                        chunk.jit_budget_wait_bytes.set(required_bytes);
                    }
                    crate::jit::JitCompileOutcome::Unavailable => {
                        permit.commit(None);
                    }
                }
            }
            if let Some(Some(code)) = chunk.jit.get() {
                jit_code = Some(code);
            }
        }
        let r = match jit_code {
            Some(code) => {
                // A tree-walker caller can repeatedly enter a hot ordinary native callee
                // before its own frame tiers up. These are real machine-code runs too:
                // ignoring them delays the callee's inline checkpoint until a cached native
                // caller appears. The selected chunk/code lease stays pinned for this call;
                // publication changes only later dispatch, never a live borrowed frame.
                self.note_native_entry(
                    Rc::as_ptr(func),
                    chunk,
                    Gc::as_ptr(&self.global) as usize,
                    Env::as_ptr(&self.global_env) as usize,
                );
                crate::jit::run(self, chunk, &code, closure, this_val, args)
            }
            None => crate::bytecode::run(self, chunk, closure, this_val, args),
        };
        self.strict = saved_strict;
        self.tco_ok = saved_tco;
        self.in_field_init_code = saved_field_init;
        self.in_async_gen_body = saved_agb;
        r
    }

    /// Run a compiled derived constructor under the Function Environment Record required by
    /// ECMA-262 §§10.2.2 and 13.3.7.1. Its `this` binding begins uninitialized; `super()`
    /// constructs the parent, binds that exact record, and initializes this class's instance
    /// elements before the following bytecode executes.
    fn run_compiled_derived_constructor(
        &mut self,
        func: &Rc<Function>,
        chunk: &Rc<crate::bytecode::Chunk>,
        closure: &Env,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        thread_local! {
            /// The binding names of every derived constructor's Function Environment Record.
            static NAMES: [Rc<str>; 4] = [
                Rc::from("%newtarget%"),
                Rc::from("this"),
                Rc::from("%supercallok%"),
                Rc::from("%superpropok%"),
            ];
        }
        let [new_target_name, this_name, super_call_name, super_prop_name] =
            NAMES.with(|names| names.clone());
        let env = new_var_scope(Some(*closure));
        {
            let mut scope = env.borrow_mut();
            scope.vars.insert(
                new_target_name,
                Binding::data(self.new_target, false, true).with_strict_immutable(false),
            );
            scope
                .vars
                .insert(this_name, Binding::data(Value::Undefined, false, false));
            scope.vars.insert(
                super_call_name,
                Binding::data(Value::Bool(true), false, true),
            );
            scope.vars.insert(
                super_prop_name,
                Binding::data(Value::Bool(true), false, true),
            );
        }

        let saved_strict = std::mem::replace(&mut self.strict, func.is_strict);
        let saved_tco = std::mem::replace(&mut self.tco_ok, false);
        let saved_field_init = self.in_field_init_code;
        let saved_agb = self.in_async_gen_body;
        self.in_field_init_code = false;
        self.in_async_gen_body = false;
        // SuperCall/NewTarget/lexical-this ops intentionally keep derived constructors on the
        // heap bytecode VM until the machine-code tier can represent this live Environment Record.
        let result = crate::bytecode::run(self, chunk, &env, Value::Undefined, args);
        self.strict = saved_strict;
        self.tco_ok = saved_tco;
        self.in_field_init_code = saved_field_init;
        self.in_async_gen_body = saved_agb;

        let value = result?;
        if matches!((value).view(), ValueView::Obj(_)) {
            return Ok(value);
        }
        if !matches!((value).view(), ValueView::Undefined | ValueView::Empty) {
            return Err(self.throw_in_caller_realm(
                "TypeError",
                "a derived constructor may only return an object or undefined",
            ));
        }
        let binding = env
            .borrow()
            .vars
            .get("this")
            .map(|binding| (binding.value(), binding.initialized));
        match binding {
            Some((this, true)) => Ok(this),
            _ => Err(self.throw_in_caller_realm(
                "ReferenceError",
                "derived constructor returned without calling super()",
            )),
        }
    }

    pub(crate) fn call_user(
        &mut self,
        func: &Rc<Function>,
        closure: Env,
        this: Value,
        args: &[Value],
        is_construct: bool,
        fn_obj: &Gc,
    ) -> Result<Value, Abrupt> {
        // ECMA-262 functions carry the Realm in which they were created. A
        // same-Agent call from another Window must activate that Realm before
        // resolving intrinsics, global declarations, and host settings; the
        // caller's Realm is restored after completion (including throws).
        let target_key = match &fn_obj.borrow().call {
            Callable::User(user) => user.realm,
            _ => 0,
        };
        let current_key = Gc::as_ptr(&self.global) as usize;
        if self.realms.len() > 1 && target_key != 0 && target_key != current_key {
            if let Some(target) = self.realms.get(&target_key).map(RealmState::snapshot_clone) {
                return self.with_realm_snapshot(target_key, target, |interp| {
                    interp.call_user_active(func, closure, this, args, is_construct, fn_obj)
                });
            }
        }
        self.call_user_active(func, closure, this, args, is_construct, fn_obj)
    }

    fn call_user_active(
        &mut self,
        func: &Rc<Function>,
        closure: Env,
        this: Value,
        args: &[Value],
        is_construct: bool,
        fn_obj: &Gc,
    ) -> Result<Value, Abrupt> {
        self.fn_frames.push(FnFrame {
            fn_ptr: Gc::as_ptr(fn_obj) as usize,
            call_site: std::mem::replace(&mut self.call_site, crate::stack_trace::NO_POSITION),
            strict: func.is_strict,
            construct: is_construct,
            extra: None,
        });
        let r = self.call_user_inner(func, closure, this, args, is_construct, fn_obj);
        self.pop_fn_frame();
        r
    }

    /// The JIT→JIT fast call (from `bytecode::jit_exec`'s Call/CallWithThis arms): a plain,
    /// same-realm, already-JIT-compiled user function with no activation environment runs with
    /// exactly the observable effects of the layered path — recursion depth, gc_check, the
    /// `FnFrame` for `f.caller` reflection, constructing/new.target save-clear-restore, and the
    /// proper-tail-call trampoline — but skips the dispatch layers' proxy/realm/eval-marker
    /// re-checks (guarded up front). It copies canonical words from the caller's immutable
    /// evaluated argument window into the callee's separate slots.
    ///
    /// `None` leaves the argument window untouched; `Some` completes the call while retaining
    /// that window and receiver through every nested call, post-call poll and tail drain.
    ///
    /// The identity-cached JIT→JIT call (see [`crate::bytecode::CallIc`]): on a per-site hit,
    /// the entire dispatch guard set collapses into two pointer compares (callee identity +
    /// active-realm global), and the frame's derived state (env, chunk, machine code, strict)
    /// reads through the cached raw pointers.
    /// `None` = miss (empty cache / different callee / realm switched), with NO side effects —
    /// the caller falls into [`Interp::call_jit_fast`], which revalidates and refills.
    ///
    /// # Safety
    /// `args` is aligned and non-null even for zero arity. `args..args+argc` and `*this_slot`
    /// are initialized immutable execution words, stable
    /// through the complete call including post-call polls and pending tails. The caller
    /// roots the words before entry, using its live native/JIT frame or an explicit root for
    /// Rust-heap storage, because the first poll precedes copying into the callee's frame.
    /// It retires the argument window only after `Some` returns; `None` leaves it untouched.
    pub(crate) unsafe fn call_jit_cached(
        &mut self,
        site: &crate::bytecode::CallSite,
        callee: &Value,
        this_slot: *const crate::value::Value,
        args: *const crate::value::Value,
        argc: usize,
    ) -> Option<Result<Value, Abrupt>> {
        let ic = match self.resolve_cached_call_entry(site, callee) {
            CachedCall::Miss => return None,
            CachedCall::Entry(entry) => {
                crate::bytecode::note_call_path(crate::bytecode::CallPath::Cached);
                // A native entry needs only its builtin: read it in place rather than from a
                // copy whose stores are still in flight.
                // SAFETY: `entry` is live and nothing re-entrant has run since the probe.
                let native = unsafe { (*entry).native };
                if native != 0 {
                    let signature = unsafe { (*entry).direct };
                    let intrinsic = unsafe { (*entry).intrinsic };
                    let result = unsafe {
                        self.call_native_entry(native, signature, intrinsic, this_slot, args, argc)
                    };
                    #[cfg(all(test, feature = "embed"))]
                    if signature & crate::bytecode::CALL_IC_SCOPED_NATIVE != 0 {
                        // A completed scoped cached-native boundary, through the same production
                        // helper in tests and release. The fixture checks each exact Ok/Err
                        // payload so an early interruption/RangeError cannot satisfy its count.
                        crate::bytecode::TEST_SCOPED_NATIVE_CACHED_COMMITS
                            .with(|count| count.set(count.get() + 1));
                    }
                    return Some(result);
                }
                // The callee may refill this way; run from a copy.
                unsafe { entry.read() }
            }
            CachedCall::Shared(ic) => {
                crate::bytecode::note_call_path(crate::bytecode::CallPath::Cached);
                ic
            }
        };
        let code = unsafe { crate::jit::cache::lease_raw(ic.code) };
        self.note_compiled_call(&ic);
        Some(unsafe { self.call_jit_committed(&ic, &code, this_slot, args, argc) })
    }

    /// The cache resolution of [`Interp::call_jit_cached`]: copy the entry that would run
    /// `callee` from `site` now into `out` and answer `true`, or answer `false` (with no side
    /// effects) when the site must revalidate. Inlined so the hit's entry copy lands directly
    /// in the caller's frame, as one plain copy: returning the entry inside an `Option` made the
    /// compiler move it around the `strict` niche in overlapping, misaligned pieces that each
    /// waited on the previous stores.
    #[inline(always)]
    fn resolve_cached_call(
        &self,
        site: &crate::bytecode::CallSite,
        callee: &Value,
        out: &mut std::mem::MaybeUninit<crate::bytecode::CallIc>,
    ) -> bool {
        match self.resolve_cached_call_entry(site, callee) {
            CachedCall::Miss => false,
            CachedCall::Entry(entry) => {
                // SAFETY: `entry` points at a live way or overflow entry; nothing re-entrant
                // runs between the probe and this copy.
                unsafe { std::ptr::copy_nonoverlapping(entry, out.as_mut_ptr(), 1) };
                true
            }
            CachedCall::Shared(ic) => {
                out.write(ic);
                true
            }
        }
    }

    #[inline(always)]
    fn resolve_cached_call_entry(
        &self,
        site: &crate::bytecode::CallSite,
        callee: &Value,
    ) -> CachedCall {
        #[cfg(feature = "architecture-diagnostics")]
        crate::bytecode::call_cache_diagnostics::probe(site, callee);
        let ValueView::Obj(o) = &(callee).view() else {
            return CachedCall::Miss;
        };
        let key = Gc::as_ptr(o) as usize;
        let genv = Env::as_ptr(&self.global_env) as usize;
        // Probe the identity fields through the Cell without copying whole entries; only the
        // hit is copied out (nothing re-entrant runs between the probe and the copy).
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if epoch == u32::MAX {
            return CachedCall::Miss;
        }
        for e in &site.entries {
            let p = e.as_ptr();
            unsafe {
                if (*p).callee == key && (*p).global_env == genv && (*p).epoch == epoch {
                    return CachedCall::Entry(p);
                }
            }
        }
        if let Some(entry) = self.call_overflow.lookup_entry(key, genv, epoch) {
            return CachedCall::Entry(entry);
        }
        {
            {
                // OrdinaryFunctionCreate gives each closure its own identity, [[Environment]]
                // and [[Realm]], but closures from the same expression share immutable code.
                // Reuse that code for BOTH activation-homed and slot-only functions. Restricting
                // this retry to needs-env entries made tiny freshly allocated callbacks refill
                // the cache on every invocation, eventually saturating the weak-pin budget.
                let has_user_entry = site.entries.iter().any(|e| {
                    let p = e.as_ptr();
                    unsafe {
                        !(*p).func.is_null() && (*p).global_env == genv && (*p).epoch == epoch
                    }
                });
                if !has_user_entry {
                    return CachedCall::Miss;
                }
                let object = o.borrow();
                let (fp, ep) = match &object.call {
                    // `under_with`: see the refusal in `call_jit_fast` — this retry runs a
                    // FRESH closure instance whose environment/realm was never vetted there.
                    // The immutable [[Realm]] slot proves same-realm execution independently
                    // of user mutations to the function object's prototype chain.
                    Callable::User(user)
                        if object.ic_plain.get()
                            && user.realm == Gc::as_ptr(&self.global) as usize
                            && !user.env.borrow().under_with =>
                    {
                        (Rc::as_ptr(&user.func), Env::as_ptr(&user.env))
                    }
                    _ => return CachedCall::Miss,
                };
                let class_constructor = object.class_kind != 0;
                drop(object);
                if class_constructor {
                    return CachedCall::Miss;
                }
                let mut found = None;
                for e in &site.entries {
                    let p = e.as_ptr();
                    unsafe {
                        if (*p).func == fp && (*p).global_env == genv && (*p).epoch == epoch {
                            found = Some(*p);
                            break;
                        }
                    }
                }
                let Some(mut ic) = found else {
                    return CachedCall::Miss;
                };
                // PrepareForOrdinaryCall's Function is the LIVE closure, not the
                // instance that populated this code-sharing entry. FnFrame borrows
                // its identity from the caller's callee Value; the old instance may
                // already be dead (the cache only weakly pins its address). Keeping
                // ic.callee here caused reflection/error stack capture to read freed
                // object storage. Change the local copy, not the pinned cache entry.
                ic.callee = key;
                ic.env = ep;
                CachedCall::Shared(ic)
            }
        }
    }

    /// Inline-recompile trigger shared by every compiled-entry call: a chunk that keeps running
    /// in machine code gets one shot at splicing its own hot monomorphic callees (see
    /// `bytecode::plan_inlines`).
    #[inline]
    fn note_compiled_call(&mut self, ic: &crate::bytecode::CallIc) {
        self.note_native_entry(ic.func, unsafe { &*ic.chunk }, ic.realm, ic.global_env);
    }

    /// Account a committed, leased native entry regardless of how its caller dispatched it.
    /// The selected chunk/code remain pinned for this call if a splice is published here.
    #[inline]
    fn note_native_entry(
        &mut self,
        func: *const crate::ast::Function,
        chunk: &crate::bytecode::Chunk,
        realm: usize,
        global_env: usize,
    ) {
        let runs = chunk.jit_runs.get().saturating_add(1);
        chunk.jit_runs.set(runs);
        // A prepared body retains its original FunctionDeclarationInstantiation even when
        // its canonical caller chunk is recompiled with frame-independent callee splices.
        if chunk.inline_retry_due(runs) {
            self.try_inline_recompile(func, chunk, realm, global_env);
        }
    }

    /// Resolve `callee` once for a run of calls made by one native algorithm (an Array
    /// iteration method, a sort comparator, ...). The result carries the machine-code lease, so
    /// each [`Interp::call_prepared`] skips the identity probe, the entry copy and the lease.
    /// The caller must keep `callee` alive while the preparation is in use.
    pub(crate) fn prepare_call(
        &self,
        site: &crate::bytecode::CallSite,
        callee: &Value,
    ) -> Option<PreparedCall> {
        let mut entry = std::mem::MaybeUninit::uninit();
        if !self.resolve_cached_call(site, callee, &mut entry) {
            return None;
        }
        // SAFETY: a `true` resolution initialized the entry.
        let ic = unsafe { entry.assume_init() };
        let code = (ic.native == 0).then(|| unsafe { crate::jit::cache::lease_raw(ic.code) });
        Some(PreparedCall { ic, code })
    }

    /// Whether a preparation still describes the entry [`Interp::call_jit_cached`] would take:
    /// the global call-cache epoch (recompiles, invalidations) and the active realm are
    /// unchanged. Callee identity is fixed by the preparation's owner.
    #[inline]
    pub(crate) fn prepared_call_current(&self, prepared: &PreparedCall) -> bool {
        prepared.ic.epoch
            == crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
            && prepared.ic.global_env == Env::as_ptr(&self.global_env) as usize
    }

    /// One call through a current [`PreparedCall`], with exactly the effects of the
    /// corresponding [`Interp::call_jit_cached`] hit.
    ///
    /// # Safety
    /// Same immutable-window loan as `call_jit_cached`; `prepared` must be current
    /// (see [`Interp::prepared_call_current`]) for a callee the caller keeps alive.
    pub(crate) unsafe fn call_prepared(
        &mut self,
        prepared: &PreparedCall,
        this_slot: *const crate::value::Value,
        args: *const crate::value::Value,
        argc: usize,
    ) -> Result<Value, Abrupt> {
        let ic = &prepared.ic;
        match &prepared.code {
            None => unsafe {
                self.call_native_entry(ic.native, ic.direct, ic.intrinsic, this_slot, args, argc)
            },
            Some(code) => {
                self.note_compiled_call(ic);
                unsafe { self.call_jit_committed(ic, code, this_slot, args, argc) }
            }
        }
    }

    /// A needs-env IC hit ([`crate::bytecode::CALL_IC_NEEDS_ENV`]): same guards and frame
    /// bookkeeping as [`Interp::call_jit_committed`]. The activation is built before the owned
    /// arguments move into a fixed JIT frame, so captured values are cloned only where required
    /// and the ordinary frame slots do not duplicate the caller's arguments.
    ///
    /// # Safety
    /// Same contract as `call_jit_committed`.
    pub(crate) unsafe fn call_jit_env_committed(
        &mut self,
        ic: &crate::bytecode::CallIc,
        code: &crate::jit::JitCode,
        env_ptr: *const RefCell<Scope>,
        this_slot: *const crate::value::Value,
        args: *const crate::value::Value,
        argc: usize,
    ) -> Result<Value, Abrupt> {
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        if let Err(e) = self.gc_check_amortized() {
            self.depth -= 1;
            return Err(e);
        }
        let saved_ctor = std::mem::replace(&mut self.constructing, false);
        // ECMA-262 OrdinaryCallBindThis / arrow FunctionDeclarationInstantiation:
        // lexical calls do not introduce this/new.target bindings of their own.
        let saved_nt = if ic.direct & crate::bytecode::CALL_IC_LEXICAL_THIS != 0 {
            self.new_target
        } else {
            std::mem::replace(&mut self.new_target, Value::Undefined)
        };
        self.fn_frames.push(FnFrame {
            fn_ptr: ic.callee,
            call_site: std::mem::replace(&mut self.call_site, crate::stack_trace::NO_POSITION),
            strict: ic.strict,
            construct: false,
            extra: None,
        });
        // Same non-owning env borrow as the moved path: the callee object on the caller's
        // operand stack keeps it alive through its never-reassigned `call` field. `env_ptr`
        // is the LIVE callee's env — for a func-keyed hit it differs from `ic.env` (which
        // belongs to whichever closure instance filled the entry).
        let env = std::mem::ManuallyDrop::new(unsafe { Env::from_raw(env_ptr) });
        let chunk = unsafe { &*ic.chunk };
        let this_val = self.bind_this_word(unsafe { this_slot.read() }, ic.uses_this, ic.strict);
        let mut r = with_execution_stack(self.depth, || unsafe {
            crate::jit::run_moved_env(
                self,
                chunk,
                code,
                &*env as *const Env,
                this_val,
                args,
                argc,
                (ic.n_params as usize, ic.n_slots as usize),
            )
        });
        self.pop_fn_frame();
        self.constructing = saved_ctor;
        self.new_target = saved_nt;
        while r.is_ok() {
            match self.run_pending_tail(false) {
                Some(next) => r = next,
                None => break,
            }
        }
        self.depth -= 1;
        r
    }

    /// Invoke a cached bare native using the original callback signature. Inspect the signature
    /// before reconstructing a typed pointer; an address alone cannot distinguish Rust Results.
    ///
    /// # Safety
    /// `native` must be the original function address selected by `signature`, from a validated
    /// live cache entry. The receiver and argument window obey `call_native_committed`'s loan.
    #[inline]
    pub(crate) unsafe fn call_native_address_committed(
        &mut self,
        native: usize,
        signature: u8,
        this_slot: *const Value,
        args: *const Value,
        argc: usize,
    ) -> Result<Value, Abrupt> {
        #[cfg(feature = "embed")]
        if signature & crate::bytecode::CALL_IC_SCOPED_NATIVE != 0 {
            // SAFETY: the signature bit records this exact declared type at publication.
            let function: crate::scope::NativeFn =
                unsafe { std::mem::transmute(native as *const ()) };
            return unsafe {
                self.call_native_committed_with(
                    native,
                    this_slot,
                    args,
                    argc,
                    |interp, this, arguments| {
                        crate::scope::invoke_native(function, interp, this, arguments)
                    },
                )
            };
        }
        debug_assert_eq!(signature & crate::bytecode::CALL_IC_SCOPED_NATIVE, 0);
        let function: crate::value::NativeFn = unsafe { std::mem::transmute(native as *const ()) };
        unsafe { self.call_native_committed(function, this_slot, args, argc) }
    }

    /// A native entry's call: on the caller's window when the entry's class proves the native
    /// runs no author code for these operands ([`Interp::call_native_light`]), else through the
    /// boundary ([`Interp::call_native_address_committed`]).
    ///
    /// # Safety
    /// As [`Interp::call_native_address_committed`]; `intrinsic` is the entry's.
    #[inline]
    pub(crate) unsafe fn call_native_entry(
        &mut self,
        native: usize,
        signature: u8,
        intrinsic: u8,
        this_slot: *const Value,
        args: *const Value,
        argc: usize,
    ) -> Result<Value, Abrupt> {
        if signature & crate::bytecode::CALL_IC_SCOPED_NATIVE == 0 {
            // SAFETY: without the scoped bit the entry's address is an engine NativeFn.
            let function: crate::value::NativeFn =
                unsafe { std::mem::transmute(native as *const ()) };
            if let Some(result) =
                unsafe { self.call_native_light(function, intrinsic, this_slot, args, argc) }
            {
                return result;
            }
        }
        unsafe { self.call_native_address_committed(native, signature, this_slot, args, argc) }
    }

    /// W3's item 2a: a native-entry hit whose class (`intrinsic`, see
    /// [`crate::bytecode::native_light_operands`]) proves that the native runs no author code
    /// for these operands, makes no tail call and reaches no safepoint, run on the caller's
    /// window without [`Interp::call_native_committed_with`]'s bookkeeping. Nothing collects
    /// before it returns, so the window needs no root of its own, and nothing in it can be
    /// interrupted, run a host callback or leave a tail call. The construct state is cleared
    /// around it, as the boundary does: natives that are also constructors read it. `None`: these
    /// operands need the boundary.
    ///
    /// # Safety
    /// As [`Interp::call_native_committed`]; `function` is the entry's proven native.
    #[inline(always)]
    pub(crate) unsafe fn call_native_light(
        &mut self,
        function: crate::value::NativeFn,
        intrinsic: u8,
        this_slot: *const Value,
        args: *const Value,
        argc: usize,
    ) -> Option<Result<Value, Abrupt>> {
        if !unsafe { crate::bytecode::native_light_operands(intrinsic, this_slot, args, argc) } {
            return None;
        }
        #[cfg(test)]
        TEST_LIGHT_NATIVE_CALLS.with(|count| count.set(count.get() + 1));
        // The native runs with a call's construct state: not constructing, new.target Undefined.
        // An ordinary caller already holds it (the ARM64 direct sequence relies on the same), so
        // only a constructing caller's state is saved, cleared and restored around the call.
        let ordinary = !self.constructing && matches!(self.new_target.view(), ValueView::Undefined);
        let saved = (!ordinary).then(|| {
            (
                std::mem::replace(&mut self.constructing, false),
                std::mem::replace(&mut self.new_target, Value::Undefined),
            )
        });
        // SAFETY: the caller's window is initialized and outlives this call.
        let result = unsafe {
            crate::execution_storage::with_borrowed_values(args, argc, |arguments| {
                function(self, *this_slot, arguments)
            })
        };
        if let Some((constructing, new_target)) = saved {
            self.constructing = constructing;
            self.new_target = new_target;
        }
        Some(result.map_err(Abrupt::Throw))
    }

    /// A native-entry IC hit, borrowing the caller's canonical evaluated argument window.
    /// Roots and loans last through post-call interruption and the pending-tail drain. The
    /// caller retires its operands only after this returns; the callee never clears or pools them.
    ///
    /// # Safety
    /// `args` is aligned and non-null even for zero arity. `args..args+argc` and `this_slot`
    /// are initialized execution words in stable immutable
    /// storage, retained by the caller through this entire operation and all nested reentry.
    #[inline]
    pub(crate) unsafe fn call_native_committed(
        &mut self,
        function: crate::value::NativeFn,
        this_slot: *const Value,
        args: *const Value,
        argc: usize,
    ) -> Result<Value, Abrupt> {
        unsafe {
            self.call_native_committed_with(function as usize, this_slot, args, argc, function)
        }
    }

    /// Share native bookkeeping while Rust monomorphizes each actual callback signature.
    ///
    /// # Safety
    /// The receiver/arguments obey `call_native_committed`'s stable immutable-window contract.
    /// `invoke` calls its function at the actual declared signature, mapping its Result explicitly.
    pub(crate) unsafe fn call_native_committed_with(
        &mut self,
        address: usize,
        this_slot: *const Value,
        args: *const Value,
        argc: usize,
        invoke: impl FnOnce(&mut Self, Value, &[Value]) -> Result<Value, Value>,
    ) -> Result<Value, Abrupt> {
        // SAFETY: the caller owns a stable immutable window until this helper returns. Reentry
        // may churn other frames and pools, but cannot retire or reuse this caller's allocation.
        unsafe {
            crate::execution_storage::with_borrowed_values(args, argc, |arguments| {
                let _arguments = crate::rooting::root_slice(arguments);
                self.depth += 1;
                if execution_stack_exhausted(self.depth) {
                    self.depth -= 1;
                    return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
                }
                if let Err(error) = self.gc_check_amortized() {
                    self.depth -= 1;
                    return Err(error);
                }
                let saved_ctor = std::mem::replace(&mut self.constructing, false);
                let saved_nt = std::mem::replace(&mut self.new_target, Value::Undefined);
                let mut result = match self.interrupt_poll_force() {
                    Ok(()) => {
                        let native_result = with_execution_stack(self.depth, || {
                            let perf_started = crate::jit::perf_stage_start();
                            let result = self
                                .with_host_callback(|interp| invoke(interp, *this_slot, arguments));
                            crate::jit::perf_native_end(
                                perf_started,
                                result.is_ok(),
                                crate::jit::NativeLabelSrc::Addr(address),
                            );
                            result
                        });
                        match self.interrupt_poll_force() {
                            Ok(()) => native_result.map_err(Abrupt::Throw),
                            Err(interrupt) => Err(interrupt),
                        }
                    }
                    Err(interrupt) => Err(interrupt),
                };
                self.constructing = saved_ctor;
                self.new_target = saved_nt;
                while result.is_ok() {
                    match self.run_pending_tail(true) {
                        Some(next) => result = next,
                        None => break,
                    }
                }
                self.depth -= 1;
                result
            })
        }
    }

    /// [`Interp::call_jit_cached`]'s committed tail, split out so the asm call thunk can enter
    /// it with an already-validated `CallIc` (the probe having run in machine code). From here
    /// the arguments and `*this_slot` remain borrowed until the whole call returns.
    ///
    /// # Safety
    /// `ic` must be a hit validated THIS turn (callee identity + epoch + realm), with the callee
    /// object alive in the caller's hands; `args..args+argc` and `*this_slot` are live
    /// immutable argument words in storage the caller retains and roots through this whole
    /// call. A live native/JIT frame supplies that root; Rust-heap windows require an explicit
    /// root before the initial poll, which precedes frame seeding. `code` is the owned lease of
    /// `ic.code`, acquired immediately after validation and retained by the caller.
    pub(crate) unsafe fn call_jit_committed(
        &mut self,
        ic: &crate::bytecode::CallIc,
        code: &crate::jit::JitCode,
        this_slot: *const crate::value::Value,
        args: *const crate::value::Value,
        argc: usize,
    ) -> Result<Value, Abrupt> {
        if ic.direct & crate::bytecode::CALL_IC_NEEDS_ENV != 0 {
            return unsafe { self.call_jit_env_committed(ic, code, ic.env, this_slot, args, argc) };
        }
        // --- committed: identical to call_jit_fast's committed path ---
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        if let Err(e) = self.gc_check_amortized() {
            self.depth -= 1;
            return Err(e);
        }
        let saved_ctor = std::mem::replace(&mut self.constructing, false);
        let saved_nt = if ic.direct & crate::bytecode::CALL_IC_LEXICAL_THIS != 0 {
            self.new_target
        } else {
            std::mem::replace(&mut self.new_target, Value::Undefined)
        };
        self.fn_frames.push(FnFrame {
            fn_ptr: ic.callee,
            call_site: std::mem::replace(&mut self.call_site, crate::stack_trace::NO_POSITION),
            strict: ic.strict,
            construct: false,
            extra: None,
        });
        // The caller retains the callee, whose UserCallable traces this environment through
        // the call. Reconstructing this Copy Env handle registers no root or owner.
        let env = std::mem::ManuallyDrop::new(unsafe { Env::from_raw(ic.env) });
        let chunk = unsafe { &*ic.chunk };
        // OrdinaryCallBindThis on the copied receiver word, over `ic.strict` and the fill-time
        // `uses_this`: `bind_this_word` binds it on its tag.
        let this_val = self.bind_this_word(unsafe { this_slot.read() }, ic.uses_this, ic.strict);
        let mut r = with_execution_stack(self.depth, || unsafe {
            crate::jit::run_moved(
                self,
                chunk,
                code,
                &*env as *const Env,
                this_val,
                args,
                argc,
                (ic.n_params as usize, ic.n_slots as usize),
            )
        });
        self.pop_fn_frame();
        self.constructing = saved_ctor;
        self.new_target = saved_nt;
        // Proper-tail-call trampoline, exactly like `call`.
        while r.is_ok() {
            match self.run_pending_tail(false) {
                Some(next) => r = next,
                None => break,
            }
        }
        self.depth -= 1;
        r
    }

    /// Bounded, feedback-adaptive second-stage compile of a hot chunk's function with its callees
    /// spliced inline. On success the new chunk lands in `FunctionRuntime::code2` and the global call-IC
    /// epoch bumps, so every cached caller re-resolves and picks it up; the old chunk stays
    /// alive (and correct) beneath any pointers already handed out.
    pub(crate) fn try_inline_recompile(
        &mut self,
        func: *const crate::ast::Function,
        chunk: &crate::bytecode::Chunk,
        // The function's [[Realm]] (`UserCallable::realm`), as its call cache recorded it.
        realm: usize,
        // That Realm's global environment, the call cache's fill-time same-Realm proof.
        genv: usize,
    ) {
        if func.is_null() {
            crate::jit::perf_inline_suppressed();
            return;
        }
        if !chunk.advance_inline_feedback() {
            crate::jit::perf_inline_suppressed();
            return;
        }
        // Alive because the callee object that carried this IC is alive and its `call` field is
        // never reassigned (the same argument that makes the rest of the IC readable).
        let func = unsafe { &*func };
        let runtime = func.runtime();
        if runtime.code2.get().is_some() {
            chunk.inline_retry_at.set(0);
            crate::jit::perf_inline_suppressed();
            return;
        }
        crate::jit::perf_inline_attempt();
        // A cached native checkpoint must still describe the active Realm. Borrow its live
        // environment for planning; raw addresses cannot manufacture a lifetime pin.
        if genv != Env::as_ptr(&self.global_env) as usize {
            crate::jit::perf_inline_suppressed();
            return;
        }
        let plan = crate::bytecode::plan_inlines(chunk, func, realm, &self.global_env);
        if plan.is_empty() {
            crate::jit::perf_inline_empty_plan();
            if std::env::var_os("TRUST_JS_TIER_LOG").is_some() {
                let src = func.source.as_deref().unwrap_or("<no source>");
                let head: String = src.chars().take(60).collect();
                eprintln!("[tier] inline: empty plan for: {}", head.replace('\n', " "));
            }
            return;
        }
        crate::jit::perf_inline_plan_sites(plan.len());
        if let Some(chunk2) = crate::bytecode::compile_with_inlines(func, &plan, chunk) {
            crate::jit::perf_inline_success();
            if std::env::var_os("TRUST_JS_TIER_LOG").is_some() {
                let src = func.source.as_deref().unwrap_or("<no source>");
                let head: String = src.chars().take(60).collect();
                eprintln!(
                    "[tier] inlined {} site(s) into: {}",
                    plan.len(),
                    head.replace('\n', " ")
                );
            }
            let _ = runtime.code2.set(Some(chunk2));
            chunk.inline_retry_at.set(0);
            crate::bytecode::invalidate_call_caches();
        } else {
            crate::jit::perf_inline_failure();
        }
    }

    fn learned_construct_layout(&self, constructor: &Gc) -> Option<crate::value::PropertyLayout> {
        let key = Gc::as_ptr(constructor) as usize;
        self.construct_capacity_hints
            .get(&key)
            .and_then(|(pin, layout)| {
                (pin.as_ptr() == Gc::as_ptr(constructor)).then(|| layout.clone())
            })
    }

    fn observe_construct_capacity(&mut self, constructor: &Gc, instance: &Gc) {
        let instance = instance.borrow();
        let observed = instance.props.observed_instance_capacity();
        if observed == 0 {
            return;
        }
        let key = Gc::as_ptr(constructor) as usize;
        let layout = instance
            .props
            .shared_layout()
            .expect("non-empty instance layout");
        if let Some((_, cached)) = self.construct_capacity_hints.get(&key) {
            // Written (and recorded for the next minor collection) only when the hint grows.
            if cached.len() < observed {
                if let Some((_, cached)) =
                    self.construct_capacity_hints.get_mut(&self.gc_heap, &key)
                {
                    *cached = layout.clone();
                }
            }
        } else if self.construct_capacity_hints.len() < 65536 {
            self.construct_capacity_hints.insert(
                &self.gc_heap,
                key,
                (Gc::downgrade(constructor), layout.clone()),
            );
        }
    }

    /// Execute `this.<initializer>.apply(this, arguments);` by forwarding the constructor's
    /// still-owned argument frame directly. The synthetic depth/gc step represents the elided
    /// native `Function.prototype.apply` call; the initializer's own call performs its normal
    /// step as well. `None` is a guard miss and has not consumed the arguments.
    #[inline(never)]
    unsafe fn construct_arguments_apply_forwarder(
        &mut self,
        chunk: &crate::bytecode::Chunk,
        this: &Gc,
        this_val: &Value,
        args: *const crate::value::Value,
        argc: usize,
    ) -> Option<Result<Value, Abrupt>> {
        let (initializer_name, initializer_key, initializer_cache, apply_cache) =
            chunk.jit_arguments_apply_forwarder()?;
        // Reuse the exact property feedback owned by the elided bytecode body. The IC probe
        // walks prototype links by borrowed raw pointer, validates every live shape/descriptor,
        // and clones only the final value. A miss re-derives the same side-effect-free ordinary
        // data-property proof; accessors/exotics still decline before anything is consumed.
        let initializer = self
            .try_ic_get(
                this,
                initializer_name,
                initializer_key,
                initializer_cache,
                true,
            )?
            .data()?;
        let ValueView::Obj(initializer_obj) = &(initializer).view() else {
            return None;
        };
        // Reuse the body's own `GetMethod("apply")` feedback too. Its shape guards prove there
        // is no own shadow, validate the live prototype link, and read the inherited slot by
        // index; an override/prototype swap/accessor/exotic still declines before arguments move.
        let apply = self
            .try_ic_get(
                initializer_obj,
                "apply",
                crate::atom::names::APPLY.key(),
                apply_cache,
                true,
            )?
            .data()?;
        let ValueView::Obj(apply_obj) = &(apply).view() else {
            return None;
        };
        let apply_ptr = Gc::as_ptr(apply_obj) as usize;

        // A large family of Prototype.js-era constructors forwards directly into a tiny
        // `initialize` body. Once that body has compiled and every property-creation site has
        // warmed its live IC, execute its structurally proven parameter/default-to-field plan
        // without entering a second JIT frame. This is deliberately derived from bytecode and
        // cache state only: function/source identity and observed argument/result values never
        // participate.
        if !self.native_call_in_current_realm(apply_obj)
            || !matches!(&initializer_obj.borrow().call,
                Callable::User(user) if user.realm == Gc::as_ptr(&self.global) as usize)
        {
            return None;
        }
        let initializer_plan =
            if let Some(runtime) = chunk.jit_forwarder_runtime(initializer_obj, apply_ptr) {
                Some(runtime)
            } else {
                if !matches!(
                    apply_obj.borrow().call,
                    Callable::Native(native)
                        if native as usize
                            == crate::builtins::nf_function_apply as *const () as usize
                ) {
                    return None;
                }
                let callable = initializer_obj.borrow();
                let Callable::User(user) = &callable.call else {
                    return None;
                };
                if user.func.is_arrow || user.env.borrow().under_with || callable.class_kind != 0 {
                    None
                } else {
                    let same_realm = user.realm == Gc::as_ptr(&self.global) as usize;
                    if same_realm {
                        let runtime = user
                            .func
                            .runtime()
                            .execution_code_for(Env::as_ptr(&self.global_env) as usize)
                            .and_then(Option::as_ref)
                            .filter(|initializer_chunk| {
                                initializer_chunk
                                    .jit
                                    .get()
                                    .is_some_and(|code| code.is_some())
                                    && initializer_chunk.jit_no_activation()
                                    && initializer_chunk.jit_initializer_plan().is_some()
                            })
                            .cloned();
                        if let Some(runtime) = &runtime {
                            chunk.cache_jit_forwarder_runtime(
                                initializer_obj,
                                apply_ptr,
                                runtime.clone(),
                            );
                        }
                        runtime
                    } else {
                        None
                    }
                }
            };

        let plan_shapes = initializer_plan.as_ref().and_then(|initializer_chunk| {
            let plan = initializer_chunk
                .jit_initializer_plan()
                .expect("initializer plan checked above");
            let epoch = crate::value::proto_epoch();
            let object = this.borrow();
            let proto_ptr = object
                .proto
                .as_ref()
                .map_or(0, |proto| Gc::as_ptr(proto) as usize);
            let start_shape = object.props.shape();
            drop(object);
            if epoch == u32::MAX || !crate::value::is_cacheable_shape(start_shape) {
                return None;
            }
            if let Some(shapes) = plan.cached_shapes(epoch, proto_ptr, start_shape) {
                return Some((epoch, proto_ptr, shapes));
            }
            let mut shape = start_shape;
            let mut shapes = [0u32; 16];
            for field in 0..plan.len() {
                let (_, _, _, cache) = initializer_chunk.jit_initializer_field(plan, field);
                let transition = (0..crate::bytecode::PROP_IC_WAYS).find_map(|way| {
                    let state = self.ic_way(cache, way).get();
                    (state.depth == crate::bytecode::IC_CREATE
                        && state.has_cacheable_shapes()
                        && state.recv_shape == shape
                        && state.mid_shape == epoch
                        && (((state.mid2_shape as u64) << 32) | state.slot as u64) as usize
                            == proto_ptr)
                        .then_some(state.holder_shape)
                });
                match transition {
                    Some(next) => {
                        shape = next;
                        shapes[field] = next;
                    }
                    None => return None,
                }
            }
            plan.cache_shapes(epoch, proto_ptr, start_shape, &shapes[..plan.len()]);
            Some((epoch, proto_ptr, shapes))
        });

        if let (Some(initializer_chunk), Some((_epoch, _proto_ptr, shapes))) =
            (initializer_plan.as_ref(), plan_shapes)
        {
            // Account for both elided calls in the recursion limit. The plan contains no calls,
            // allocation, coercion, or other user-code boundary, so it needs neither physical
            // reflection frames nor separate GC polls.
            #[cfg(target_arch = "wasm32")]
            if self.depth > WASM_EXECUTION_DEPTH_GUARD.saturating_sub(2) {
                return Some(Err(
                    self.throw("RangeError", "Maximum call stack size exceeded")
                ));
            }

            let plan = initializer_chunk
                .jit_initializer_plan()
                .expect("initializer plan checked above");
            chunk.note_forwarded_capacity(plan.len());
            let mut object = this.borrow_mut();
            let empty_receiver = object.props.shape() == 0;
            let initialized_layout =
                empty_receiver && initializer_chunk.initializer_layout().is_some();
            if empty_receiver {
                if let Some(layout) = initializer_chunk.instance_layout() {
                    object.props.predict_empty_layout(layout);
                }
            }
            for field in 0..plan.len() {
                let (slot, default, name, _) = initializer_chunk.jit_initializer_field(plan, field);
                let value = if slot < argc {
                    let arg = unsafe { args.add(slot) };
                    if let Some(default) =
                        default.filter(|_| !self.to_boolean(&unsafe { &*arg }.clone()))
                    {
                        *default
                    } else {
                        // The caller retains the immutable evaluated argument window through
                        // this operation. Copy its canonical word into the proven data field.
                        unsafe { arg.read() }
                    }
                } else {
                    default.cloned().unwrap_or(Value::Undefined)
                };
                if initialized_layout {
                    object
                        .props
                        .append_initialized_field(name, Property::plain(value));
                } else {
                    object
                        .props
                        .append_proven_plain(name, Property::plain(value));
                }
            }
            #[cfg(test)]
            constructor_transfer_tests::record_forwarded();
            object
                .props
                .finish_proven_plain_shape(shapes[plan.len() - 1]);
            if empty_receiver {
                initializer_chunk.note_initializer_layout(
                    object
                        .props
                        .shared_layout()
                        .expect("created initializer fields"),
                );
            }
            drop(object);
            return Some(Ok(Value::Undefined));
        }

        // The original body calls the native apply function here. Preserve its depth and GC
        // boundary even though the argument-list object and native dispatch are elided.
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Some(Err(
                self.throw("RangeError", "Maximum call stack size exceeded")
            ));
        }
        if let Err(e) = self.gc_check_amortized() {
            self.depth -= 1;
            return Some(Err(e));
        }

        let mut forwarded_this = std::mem::ManuallyDrop::new(*this_val);
        let fast = unsafe {
            self.call_jit_fast(
                &initializer,
                &*forwarded_this as *const Value,
                args,
                argc,
                None,
            )
        };
        let result = match fast {
            Some(r) => r,
            None => {
                unsafe { std::mem::ManuallyDrop::drop(&mut forwarded_this) };
                let args_ref = unsafe { std::slice::from_raw_parts(args, argc) };
                self.call(initializer, *this_val, args_ref)
            }
        };
        self.depth -= 1;
        Some(result.map(|_| Value::Undefined))
    }

    /// The identity-cached JIT construct (`Op::New` with a plain, same-realm compiled function
    /// constructor): replicates `construct` → `construct_dispatch`'s
    /// observable effects for that case — IsConstructor, prototype read (LIVE, each time:
    /// `F.prototype = other` is a plain value write no epoch sees), OrdinaryCreateFromConstructor,
    /// depth guard, gc tick, `FnFrame`, new.target = the callee for the body's duration,
    /// return-override — while skipping the dispatch layering and borrowing the caller's
    /// stable evaluated argument window, just as cached calls do.
    ///
    /// `None` = not applicable, no side effects, arguments untouched (caller runs the generic
    /// path). `Some(r)` completes the operation; the caller may then retire its argument window.
    ///
    /// # Safety
    /// `args` is aligned/non-null even at zero arity and names initialized immutable
    /// execution words. The caller retains and roots the complete window through all
    /// allocations, polls, nested execution and tail drains, as for `call_jit_cached`.
    pub(crate) unsafe fn construct_jit_fast(
        &mut self,
        callee: &Value,
        args: *const crate::value::Value,
        argc: usize,
        caller_ctx: Option<*mut crate::jit::JitCtx>,
        site: Option<(&crate::bytecode::Chunk, u32)>,
    ) -> Option<Result<Value, Abrupt>> {
        let ValueView::Obj(o) = &(callee).view() else {
            return None;
        };
        let key = Gc::as_ptr(o) as usize;
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if epoch == u32::MAX {
            return None;
        }
        let genv = Env::as_ptr(&self.global_env) as usize;
        // A live site entry can only have been installed by `construct_ic_fill`, hence it is
        // already known to be a same-realm ordinary user constructor. Probe it before classifying
        // natives: hot `new C()` sites otherwise borrow C and look up `.prototype` twice per
        // allocation (once to ask whether C is native, then again for the live instance proto).
        let cached_site = site
            .map(|(chunk, cache)| chunk.construct_cache(cache))
            .filter(|entry| {
                entry.call.callee == key
                    && entry.call.epoch == epoch
                    && entry.call.global_env == genv
            });
        // Native constructors do not need an ordinary instance or a JS activation: they build
        // their result in the native body. Keep them on the same read-only argument-window
        // contract, so hot `new Array()` / `new Date()` sites avoid the full IsConstructor → realm →
        // construct-dispatch stack on every iteration.  Cross-realm calls retain the layered
        // path because the callee's intrinsics must be installed before its native body runs.
        let native = if cached_site.is_none() && self.native_call_in_current_realm(o) {
            let b = o.borrow();
            // Proxy callability uses a native sentinel, not its target's body. Only the
            // actual callee's internal methods matter; unrelated proxies must not disable
            // ordinary native construction throughout the Agent.
            if b.ic_plain.get() && (b.is_constructor || b.props.contains("prototype")) {
                match &b.call {
                    native if native.is_native() => Some(native.clone()),
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some(call) = native {
            self.depth += 1;
            if execution_stack_exhausted(self.depth) {
                self.depth -= 1;
                return Some(Err(
                    self.throw("RangeError", "Maximum call stack size exceeded")
                ));
            }
            let saved_ctor = std::mem::replace(&mut self.constructing, true);
            let saved_nt = std::mem::replace(&mut self.new_target, *callee);
            let args_ref = unsafe { std::slice::from_raw_parts(args, argc) };
            let r = with_execution_stack(self.depth, || {
                self.dispatch_native(&call, Value::Undefined, args_ref)
            });
            self.constructing = saved_ctor;
            self.new_target = saved_nt;
            self.depth -= 1;
            return Some(r);
        }
        let (ic, prototype_shape, prototype_slot, arguments_apply_forwarder) = match cached_site {
            Some(entry) => (
                entry.call,
                entry.prototype_shape,
                entry.prototype_slot,
                entry.arguments_apply_forwarder,
            ),
            None => match self.construct_ics.get(&key) {
                Some(entry)
                    if entry.pin.as_ptr() == Gc::as_ptr(o)
                        && entry.call.epoch == epoch
                        && entry.call.global_env == genv =>
                {
                    (
                        entry.call,
                        entry.prototype_shape,
                        entry.prototype_slot,
                        entry.arguments_apply_forwarder,
                    )
                }
                _ => self.construct_ic_fill(o, key, epoch, genv)?,
            },
        };
        // Before GC, instance preparation or a forwarding initializer can
        // compile/reclaim code, turn the validated raw pointer into a lease.
        let code = unsafe { crate::jit::cache::lease_raw(ic.code) };
        if cached_site.is_none() {
            if let Some((chunk, cache)) = site {
                chunk.fill_construct_cache(
                    cache,
                    crate::bytecode::ConstructSite {
                        call: ic,
                        prototype_shape,
                        prototype_slot,
                        arguments_apply_forwarder,
                    },
                    o,
                );
            }
        }
        // OrdinaryCreateFromConstructor: the instance prototype is the constructor's LIVE
        // `prototype` own data property; anything else (accessor, non-object, absent) → generic.
        let proto = {
            let b = o.borrow();
            let property = if b.props.shape() == prototype_shape {
                b.props.property_at(prototype_slot as usize)
            } else {
                b.props.get("prototype")
            };
            match property {
                Some(p) if !p.accessor() => match (p.value()).view() {
                    ValueView::Obj(pp) => pp,
                    _ => return None,
                },
                _ => return None,
            }
        };
        let chunk = unsafe { &*ic.chunk };
        let static_capacity = chunk.instance_capacity_hint();
        // A non-zero chunk hint is already the exact straight-line `this.x = ...` prefix that
        // dominates ordinary constructors. Avoid two constructor-identity hash probes per
        // allocation (read learned hint now, observe/write it after return) in that common case.
        // Dynamic/forwarding constructors without a static key layout retain the learned path.
        // Guarded forwarding plans select their initializer's layout after resolving the live
        // method. The wrapper chunk may be shared by closures with different initializers.
        // Its existing capacity hint is enough here; avoid both identity-table probes per new.
        let forwarded_reservation = arguments_apply_forwarder && chunk.has_forwarded_capacity();
        let instance_layout = if forwarded_reservation {
            None
        } else {
            chunk
                .instance_layout()
                .cloned()
                .or_else(|| self.learned_construct_layout(o))
        };
        let instance_capacity =
            static_capacity.max(instance_layout.as_ref().map_or(0, |keys| keys.len()));
        let proto_ptr = Gc::as_ptr(&proto) as usize;
        let this = crate::value::Object::new_with_parts(
            Some(proto),
            crate::value::Props::with_layout(instance_capacity, instance_layout),
            crate::value::Exotic::None,
        );
        let this_val = Value::Obj(this);
        // --- committed: identical shape to call_jit_cached's committed path ---
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Some(Err(
                self.throw("RangeError", "Maximum call stack size exceeded")
            ));
        }
        if let Err(e) = self.gc_check_amortized() {
            self.depth -= 1;
            return Some(Err(e));
        }
        // Once every field site's creation IC proves the complete fresh-object shape chain, the
        // body has no remaining observable operation: append all fields under one borrow,
        // copying parameter words from the caller's immutable evaluated window. No function
        // identity or source name participates; a cold/polymorphic/accessor/prototype-mutated site executes
        // the ordinary compiled constructor and refreshes its live ICs.
        if let Some(simple) = chunk.jit_simple_constructor() {
            let epoch = crate::value::proto_epoch();
            // A freshly allocated ordinary object always starts at the shared empty shape, and
            // its prototype is the live object just read above. Avoid re-borrowing the new
            // RefCell twice merely to rediscover those construction invariants.
            let start_shape = 0;
            let shapes = if epoch == u32::MAX {
                None
            } else if let Some(cached) = simple.cached_shapes(epoch, proto_ptr, start_shape) {
                Some(cached)
            } else {
                let mut shape = start_shape;
                let mut transitions = [0u32; 16];
                let mut valid = true;
                for field in 0..simple.len() {
                    let (_, _, cache) = simple.field(field);
                    let transition = (0..crate::bytecode::PROP_IC_WAYS).find_map(|way| {
                        let state = self.ic_way(cache, way).get();
                        (state.depth == crate::bytecode::IC_CREATE
                            && state.has_cacheable_shapes()
                            && state.recv_shape == shape
                            && state.mid_shape == epoch
                            && (((state.mid2_shape as u64) << 32) | state.slot as u64) as usize
                                == proto_ptr)
                            .then_some(state.holder_shape)
                    });
                    match transition {
                        Some(next) => {
                            shape = next;
                            transitions[field] = next;
                        }
                        None => {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid {
                    simple.cache_shapes(
                        epoch,
                        proto_ptr,
                        start_shape,
                        &transitions[..simple.len()],
                    );
                    Some(transitions)
                } else {
                    None
                }
            };
            if let Some(shapes) = shapes {
                let mut object = this.borrow_mut();
                let initialized_layout = chunk.initializer_layout().is_some();
                for field in 0..simple.len() {
                    let (slot, name, _) = simple.field(field);
                    let value = if slot < argc {
                        let arg = unsafe { args.add(slot) };
                        // Copy from the immutable evaluated argument window; the caller
                        // retires its storage after construction completes.
                        unsafe { arg.read() }
                    } else {
                        Value::Undefined
                    };
                    if initialized_layout {
                        object
                            .props
                            .append_initialized_field(name, Property::plain(value));
                    } else {
                        object
                            .props
                            .append_proven_plain(name, Property::plain(value));
                    }
                }
                #[cfg(test)]
                constructor_transfer_tests::record_simple();
                object
                    .props
                    .finish_proven_plain_shape(shapes[simple.len() - 1]);
                chunk
                    .note_initializer_layout(object.props.shared_layout().expect("created fields"));
                drop(object);
                self.depth -= 1;
                return Some(Ok(this_val));
            }
        }
        let saved_ctor = std::mem::replace(&mut self.constructing, false);
        // The body runs with new.target = the callee (what the layered path's pending-new-target
        // handoff produces for `new F()`); compiled chunks can't read it directly, but natives
        // the body invokes could observe it ambiently.
        let saved_nt = std::mem::replace(&mut self.new_target, *callee);
        self.fn_frames.push(FnFrame {
            fn_ptr: key,
            call_site: std::mem::replace(&mut self.call_site, crate::stack_trace::NO_POSITION),
            strict: ic.strict,
            construct: true,
            extra: None,
        });
        let env = std::mem::ManuallyDrop::new(unsafe { Env::from_raw(ic.env) });
        let tv = if ic.uses_this {
            this_val
        } else {
            Value::Undefined
        };
        // The moved entries take `this` packed.
        let mut r = with_execution_stack(self.depth, || {
            let forwarded = arguments_apply_forwarder
                .then(|| unsafe {
                    self.construct_arguments_apply_forwarder(chunk, &this, &this_val, args, argc)
                })
                .flatten();
            if let Some(r) = forwarded {
                r
            } else if ic.direct & crate::bytecode::CALL_IC_NEEDS_ENV != 0 {
                unsafe {
                    crate::jit::run_moved_env(
                        self,
                        chunk,
                        &code,
                        &*env as *const Env,
                        tv,
                        args,
                        argc,
                        (ic.n_params as usize, ic.n_slots as usize),
                    )
                }
            } else {
                match caller_ctx {
                    Some(caller_ctx) => unsafe {
                        crate::jit::run_moved_shared(
                            self,
                            &mut *caller_ctx,
                            chunk,
                            &code,
                            &*env as *const Env,
                            tv,
                            args,
                            argc,
                            (ic.n_params as usize, ic.n_slots as usize),
                        )
                    },
                    None => unsafe {
                        crate::jit::run_moved(
                            self,
                            chunk,
                            &code,
                            &*env as *const Env,
                            tv,
                            args,
                            argc,
                            (ic.n_params as usize, ic.n_slots as usize),
                        )
                    },
                }
            }
        });
        self.pop_fn_frame();
        self.constructing = saved_ctor;
        self.new_target = saved_nt;
        while r.is_ok() {
            match self.run_pending_tail(false) {
                Some(next) => r = next,
                None => break,
            }
        }
        self.depth -= 1;
        if r.is_ok() && chunk.instance_layout().is_none() && !forwarded_reservation {
            self.observe_construct_capacity(o, &this);
        }
        // A constructor explicitly returning an object overrides the instance.
        Some(r.map(|ret| match ret.view() {
            ValueView::Obj(_) => ret,
            _ => this_val,
        }))
    }

    /// [`Interp::construct_jit_fast`]'s miss path: revalidate the full guard set (the same one
    /// `call_jit_fast` uses, plus [[Construct]]'s own arrow/method/generator/async and
    /// class-constructor exclusions) and cache the derived state under the callee's identity.
    fn construct_ic_fill(
        &mut self,
        o: &Gc,
        key: usize,
        epoch: u32,
        genv: usize,
    ) -> Option<(crate::bytecode::CallIc, u32, u32, bool)> {
        let object = o.borrow();
        let (func, env) = match &object.call {
            Callable::User(user)
                if object.ic_plain.get() && user.realm == Gc::as_ptr(&self.global) as usize =>
            {
                (user.func.clone(), user.env)
            }
            _ => return None,
        };
        let class_constructor = object.class_kind != 0;
        drop(object);
        if func.is_arrow || func.is_method || func.is_generator || func.is_async {
            return None;
        }
        if env.borrow().under_with {
            return None;
        }
        if class_constructor {
            return None;
        }
        let (prototype_shape, prototype_slot) = {
            let b = o.borrow();
            if !crate::value::is_cacheable_shape(b.props.shape()) {
                return None;
            }
            let slot = b.props.slot_of("prototype")?;
            let property = b.props.property_at(slot)?;
            if property.accessor() || !matches!((property.value()).view(), ValueView::Obj(_)) {
                return None;
            }
            (b.props.shape(), slot as u32)
        };
        let runtime = func.runtime();
        let chunk = match runtime.execution_code_for(Env::as_ptr(&self.global_env) as usize) {
            Some(Some(c)) => c,
            _ => return None,
        };
        let code = match chunk.jit.get() {
            Some(Some(c)) => c,
            _ => return None,
        };
        let arguments_apply_forwarder = chunk.jit_arguments_apply_forwarder().is_some();
        // The general [[Construct]] continuation still owns return-value/derived-this handling.
        // Do not enter a call-only tail-transfer body through its lean constructor shortcut.
        if chunk.has_tail_calls() || chunk.prepared_entry {
            return None;
        }
        // Activation-requiring chunks use the same fresh-frame entry as cached calls,
        // materializing captured bindings/arguments while retaining the caller's argument loan.
        let needs_env = !chunk.jit_no_activation();
        let (n_params, n_slots) = chunk.jit_frame();
        if n_params > u16::MAX as usize || n_slots > u16::MAX as usize {
            return None;
        }
        if self.construct_ics.len() >= 65536 {
            return None; // runaway-ctor backstop; identity caching should never get here
        }
        if !self
            .global_env_pins
            .iter()
            .any(|g| g.as_ptr() == Env::as_ptr(&self.global_env))
        {
            let g = Env::downgrade(&self.global_env);
            self.global_env_pins.push(g);
        }
        let ic = crate::bytecode::CallIc {
            callee: key,
            env: Env::as_ptr(&env),
            chunk: chunk as *const Rc<crate::bytecode::Chunk>,
            code: Rc::as_ptr(&code),
            global_env: genv,
            strict: func.is_strict,
            uses_this: chunk.needs_frame_this(),
            n_params: n_params as u16,
            n_slots: n_slots as u16,
            direct: if needs_env {
                crate::bytecode::CALL_IC_NEEDS_ENV
            } else {
                chunk.jit_direct_flags(&code)
                    | (((chunk.inline_attempted.get() || func.runtime().code2.get().is_some())
                        as u8)
                        << 2)
            },
            func: Rc::as_ptr(&func),
            epoch,
            chunk_raw: Rc::as_ptr(chunk),
            code_mem: code.mem_ptr(),
            pc_offs_ptr: code.pc_offsets_ptr(),
            native: 0,
            intrinsic: 0,
            // Checked above: the constructor belongs to the active Realm.
            realm: Gc::as_ptr(&self.global) as usize,
        };
        self.construct_ics.insert(
            key,
            ConstructIc {
                call: ic,
                pin: Gc::downgrade(o),
                prototype_shape,
                prototype_slot,
                arguments_apply_forwarder,
            },
        );
        Some((
            ic,
            prototype_shape,
            prototype_slot,
            arguments_apply_forwarder,
        ))
    }

    /// A guarded cold compiled call; `None` leaves the caller's window untouched.
    ///
    /// # Safety
    /// The initialized immutable receiver/argument window is aligned/non-null (also at zero
    /// arity), retained and rooted for this complete operation and its polls/tail drain.
    /// Native/JIT stack windows are roots under Decision 0; Rust-heap windows need an explicit
    /// root before entry because the first collection precedes copying into the callee frame.
    pub(crate) unsafe fn call_jit_fast(
        &mut self,
        callee: &Value,
        this_slot: *const crate::value::Value,
        args: *const crate::value::Value,
        argc: usize,
        // The caller's per-site cache + pin map: filled once the full guard set passes, so
        // subsequent calls take [`Interp::call_jit_cached`]'s pointer-compare path.
        site: Option<(
            &crate::bytecode::CallSite,
            &RefCell<crate::fasthash::FastMap<usize, crate::value::WeakGc>>,
        )>,
    ) -> Option<Result<Value, Abrupt>> {
        // Eligibility belongs to the callee, not to all objects in the Agent. Proxy
        // [[Call]] must dispatch its traps, but a separate ordinary function still has the
        // same [[Call]] after any number of unrelated proxies have been created/revoked.
        // The immutable Callable and identity pin preserve this proof on cached hits.
        let ValueView::Obj(o) = &(callee).view() else {
            return None;
        };
        let object = o.borrow();
        let (func, env, realm) = match &object.call {
            Callable::User(user) if object.ic_plain.get() => {
                (user.func.clone(), user.env, user.realm)
            }
            _ => return None,
        };
        drop(object);
        // A closure under a `with` scope stays on the tree-walker (see `Scope::under_with`):
        // the VM's free-name ops re-resolve at execution time, which is observably wrong when
        // a with-object binding changes between the spec's reference resolution and its use.
        if env.borrow().under_with {
            return None;
        }
        // PrepareForOrdinaryCall selects immutable [[Realm]], independently of
        // [[Environment]] and the function object's mutable prototype. Private
        // self-hosted environments have no global parent, and host settings may
        // replace a Realm's global environment while ordinary closures survive.
        // Cache fills still pin and guard the currently active global environment.
        if realm != Gc::as_ptr(&self.global) as usize {
            return None;
        }
        if o.borrow().class_kind != 0 {
            return None;
        }
        // Not yet tiered / didn't compile / no machine code → generic (which counts calls up).
        // Function's chunk handles remain set-once, but native residency is
        // reclaimable. Hold the owned code lease through GC and the whole call.
        let runtime = func.runtime();
        let chunk = match runtime.execution_code_for(Env::as_ptr(&self.global_env) as usize) {
            Some(Some(c)) => c,
            _ => return None,
        };
        let code = match chunk.jit.get() {
            Some(Some(c)) => c,
            _ => return None,
        };
        // A prepared body must first run FunctionDeclarationInstantiation. Its environment is
        // not the closure environment supplied by this moved-frame shortcut.
        if chunk.prepared_entry || chunk.jit_is_resumable() {
            return None;
        }
        // A callee that needs an activation env (captured locals / lexical this) is cacheable
        // too: its committed path enters through `jit::run` (which builds the activation)
        // instead of `run_moved`. IC entries carry direct bit 4; bits 0-3 stay clear, so the
        // machine-code direct sequence's first gate routes them to the helper.
        let needs_env = !chunk.jit_no_activation();
        // Every guard passed: remember the callee at this site so the next call takes the
        // one-compare cached path. The Weak pin makes the raw identity compare ABA-safe (the
        // address can't be recycled while pinned); a megamorphic site stops refilling at the cap
        // instead of growing the pin list without bound.
        if let Some((cs, pins)) = site {
            let key = Gc::as_ptr(o) as usize;
            let mut p = pins.borrow_mut();
            if p.len() < 4096 || p.contains_key(&key) {
                p.entry(key).or_insert_with(|| Gc::downgrade(o));
                // Pin the active global scope's address too: the cached same-realm proof
                // compares it raw, so it must never be recycled while cached (realms already
                // outlive their caches in practice; this makes it airtight). Fills are rare, so
                // the linear dedup scan (bounded by realm count) doesn't matter.
                if !self
                    .global_env_pins
                    .iter()
                    .any(|g| g.as_ptr() == Env::as_ptr(&self.global_env))
                {
                    let genv = Env::downgrade(&self.global_env);
                    self.global_env_pins.push(genv);
                }
                let (n_params, n_slots) = chunk.jit_frame();
                if n_params <= u16::MAX as usize && n_slots <= u16::MAX as usize {
                    let evicted = cs.fill(
                        crate::bytecode::CallIc {
                            callee: key,
                            env: Env::as_ptr(&env),
                            chunk: chunk as *const Rc<crate::bytecode::Chunk>,
                            code: Rc::as_ptr(&code),
                            global_env: Env::as_ptr(&self.global_env) as usize,
                            strict: func.is_strict,
                            uses_this: chunk.needs_frame_this(),
                            n_params: n_params as u16,
                            n_slots: n_slots as u16,
                            direct: (if needs_env {
                                crate::bytecode::CALL_IC_NEEDS_ENV
                            } else {
                                chunk.jit_direct_flags(&code)
                                    | (((chunk.inline_attempted.get()
                                        || func.runtime().code2.get().is_some())
                                        as u8)
                                        << 2)
                            }) | if func.is_arrow {
                                crate::bytecode::CALL_IC_LEXICAL_THIS
                            } else {
                                0
                            },
                            func: Rc::as_ptr(&func),
                            epoch: crate::bytecode::CALL_IC_EPOCH
                                .load(std::sync::atomic::Ordering::Relaxed),
                            chunk_raw: Rc::as_ptr(chunk),
                            code_mem: code.mem_ptr(),
                            pc_offs_ptr: code.pc_offsets_ptr(),
                            native: 0,
                            intrinsic: 0,
                            realm,
                        },
                        Some(&func),
                    );
                    if let Some(evicted) = evicted {
                        let pin = p.get(&evicted.callee).expect("call entry has an ABA pin");
                        self.call_overflow.insert(evicted, pin.clone());
                    }
                }
            }
        }
        // Committed: retain the caller's immutable argument/receiver window through the call.
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Some(Err(
                self.throw("RangeError", "Maximum call stack size exceeded")
            ));
        }
        if let Err(e) = self.gc_check_amortized() {
            self.depth -= 1;
            return Some(Err(e));
        }
        let saved_ctor = std::mem::replace(&mut self.constructing, false);
        let saved_nt = if func.is_arrow {
            self.new_target
        } else {
            std::mem::replace(&mut self.new_target, Value::Undefined)
        };
        self.fn_frames.push(FnFrame {
            fn_ptr: Gc::as_ptr(o) as usize,
            call_site: std::mem::replace(&mut self.call_site, crate::stack_trace::NO_POSITION),
            strict: func.is_strict,
            construct: false,
            extra: None,
        });
        let this_val = self.bind_this_word(
            unsafe { this_slot.read() },
            chunk.needs_frame_this(),
            func.is_strict,
        );
        // A successful cache fill is a native entry just like the next cache hit. Its
        // original code lease stays live even if the checkpoint invalidates the fresh IC.
        self.note_native_entry(
            Rc::as_ptr(&func),
            chunk,
            realm,
            Env::as_ptr(&self.global_env) as usize,
        );
        let mut r = with_execution_stack(self.depth, || {
            if needs_env {
                unsafe {
                    crate::jit::run_moved_env(
                        self,
                        chunk,
                        &code,
                        &env as *const Env,
                        this_val,
                        args,
                        argc,
                        chunk.jit_frame(),
                    )
                }
            } else {
                unsafe {
                    crate::jit::run_moved(
                        self,
                        chunk,
                        &code,
                        &env as *const Env,
                        this_val,
                        args,
                        argc,
                        chunk.jit_frame(),
                    )
                }
            }
        });
        self.pop_fn_frame();
        self.constructing = saved_ctor;
        self.new_target = saved_nt;
        // Proper-tail-call trampoline, exactly like `call`.
        while r.is_ok() {
            match self.run_pending_tail(false) {
                Some(next) => r = next,
                None => break,
            }
        }
        self.depth -= 1;
        Some(r)
    }

    fn call_user_inner(
        &mut self,
        func: &Rc<Function>,
        closure: Env,
        this: Value,
        args: &[Value],
        is_construct: bool,
        fn_obj: &Gc,
    ) -> Result<Value, Abrupt> {
        let realm_key = Gc::as_ptr(&self.global) as usize;
        let _profile = function_profile_enter(
            func,
            &closure,
            fn_obj,
            realm_key,
            Env::as_ptr(&self.global_env) as usize,
        );
        // The callee runs (or compiles) now: read a snapshot-encoded body once, ahead of the
        // body-fact scan below.
        func.body.prepare_to_run();
        // Bytecode fast call: an eligible sync callee runs on the VM. Ordinary and base-class
        // bodies use the lean activation chosen by the compiler; a derived constructor uses the
        // dedicated Function Environment Record path below so `super()` can initialize `this`.
        // Constructs qualify when their `this` is already initialized. For an ordinary function
        // the fresh object came from `construct_nt`; for a base class, `run_constructor_on`
        // already performed ECMA-262 §10.2.2 InitializeInstanceElements before entering this
        // body. In either case a VM body cannot rebind `this`, and the outer construct path maps a
        // non-object return back to that same object. A derived class's bytecode instead retains
        // the Function Environment Record whose `this` begins uninitialized and is later bound by
        // `super()`; it deliberately stays off the lean/JIT constructor shortcuts. Generators and
        // async functions use their coroutine paths. For plain calls `call_dispatch`
        // already saved/cleared `new_target`/`constructing`. One divergence: no `lazy` args stash,
        // so legacy `f.arguments` reflection during an active VM frame reads null (the VM's
        // slot-based locals never aliased it faithfully anyway).
        let allow_compiled_body = !closure.borrow().under_with;
        let derived_class_construct =
            is_construct && fn_obj.borrow().class_kind == crate::value::CLASS_DERIVED;
        if !matches!(self.tier, crate::bytecode::Tier::Interp)
            && !func.is_generator
            && !func.is_async
            // Closures under a `with` scope stay on the tree-walker (see `Scope::under_with`).
            && allow_compiled_body
        {
            if func.runtime().code.get().is_none() {
                let n = func.runtime().calls.get().saturating_add(1);
                func.runtime().calls.set(n);
                // Cold loop-containing functions obey the same call threshold as other
                // bodies. A long-running first invocation transfers its already-instantiated
                // function frame at a settled hot backedge instead of replaying its prologue.
                // Self-hosted built-ins still compile immediately.
                if n > self.tier_threshold || func.self_hosted {
                    let compiled = if derived_class_construct {
                        crate::bytecode::compile_derived_constructor(func)
                    } else {
                        crate::bytecode::compile(func)
                    };
                    if compiled.is_none() && std::env::var_os("TRUST_JS_TIER_LOG").is_some() {
                        let src = func.source.as_deref().unwrap_or("<no source>");
                        let head: String = src.chars().take(70).collect();
                        eprintln!("[tier] bail: {}", head.replace('\n', " "));
                    }
                    let _ = func.runtime().code.set(compiled);
                }
            }
            if let Some(chunk) = func
                .runtime()
                .execution_code_for(Env::as_ptr(&self.global_env) as usize)
                .and_then(Option::as_ref)
                .filter(|chunk| !chunk.prepared_entry && (!is_construct || !chunk.has_tail_calls()))
            {
                let chunk = chunk.clone();
                let this_val = if derived_class_construct {
                    None
                } else {
                    Some(self.bind_compiled_this(func, &chunk, this, is_construct))
                };
                // A construct consumes the pending new.target exactly like the slow path, so a
                // stale value can never be observed later; restored after the run.
                let saved_new_target = if is_construct {
                    Some(std::mem::replace(
                        &mut self.new_target,
                        std::mem::replace(&mut self.pending_new_target, Value::Undefined),
                    ))
                } else {
                    None
                };
                let r = if derived_class_construct {
                    self.run_compiled_derived_constructor(func, &chunk, &closure, args)
                } else {
                    self.run_compiled_chunk(
                        func,
                        &chunk,
                        &closure,
                        this_val.expect("ordinary compiled call has a bound this"),
                        args,
                        is_construct,
                    )
                };
                if let Some(nt) = saved_new_target {
                    self.new_target = nt;
                }
                return r;
            }
        }
        // Debug: `TRUST_JS_AST_HOT=1` reports functions whose bodies keep executing on the
        // tree-walker (each time the per-function call count crosses a power of ten).
        static AST_HOT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !func
            .runtime()
            .code
            .get()
            .and_then(Option::as_ref)
            .is_some_and(|chunk| chunk.prepared_entry)
            && *AST_HOT.get_or_init(|| std::env::var_os("TRUST_JS_AST_HOT").is_some())
        {
            let n = func.runtime().calls.get().saturating_add(1);
            func.runtime().calls.set(n);
            if n >= 1000 && (n == 1000 || n == 10_000 || n == 100_000 || n == 1_000_000) {
                let name = func.name.as_deref().unwrap_or("<anon>");
                let src: String = func
                    .source
                    .as_deref()
                    .unwrap_or("")
                    .chars()
                    .take(60)
                    .collect();
                eprintln!("[ast-hot] {name} ×{n}: {}", src.replace('\n', " "));
            }
        }
        // A function with parameter expressions (default values or destructuring with defaults) gets
        // a separate parameter Environment Record — not a variable environment — so its body's `var`
        // hoisting sits in a distinct scope below it (and a direct `eval` in a parameter default
        // cannot leak declarations into the body). Otherwise a single variable environment suffices.
        // A named expression's self-binding is already in its closure environment; body vars
        // and parameters get nearer bindings here, just as they do for every other function.
        let has_param_exprs = params_have_expr(&func.params);
        let scope = if has_param_exprs {
            // Chain: callee base (variable env) → parameter env → body variable env. A direct `eval`
            // in a parameter default hoists its `var`s into the callee base, not the enclosing
            // scope, and its walk passes through the parameter env (where `arguments`/params live).
            let callee_base = new_var_scope(Some(closure));
            new_scope(Some(callee_base))
        } else {
            new_var_scope(Some(closure))
        };

        // `new.target`: an ordinary call clears it, a construct installs the pending target; an arrow
        // inherits the enclosing value. Saved and restored around the body.
        let saved_new_target = self.new_target;
        let saved_field_init = self.in_field_init_code;
        let saved_agb = self.in_async_gen_body;
        if !func.is_arrow {
            self.new_target = if is_construct {
                std::mem::replace(&mut self.pending_new_target, Value::Undefined)
            } else {
                Value::Undefined
            };
            // An ordinary function body is not class-field-initializer code, nor an async
            // generator's own body (an arrow inherits both).
            self.in_field_init_code = false;
            self.in_async_gen_body = false;
        }

        if !func.is_arrow {
            let scan = func.scan_flags();
            // `new.target` resolves lexically (arrows and closures created here keep seeing this
            // function's value, even after it returns). Skipped when nothing in the activation
            // can name it — Expr::NewTarget peeks up the scope chain, so an unused binding here
            // would only shadow an (equally unread) outer one.
            if scan & crate::ast::SCAN_NEW_TARGET != 0 {
                scope.borrow_mut().vars.insert(
                    crate::atom::names::NEW_TARGET.atom(),
                    Binding::data(self.new_target, false, true).with_strict_immutable(false),
                );
            }
            // Class machinery and every [[Construct]] need the `this` binding regardless of the
            // body: a derived constructor's stays in TDZ until `super()` initializes it in place,
            // `super.x` resolves its receiver through it, and a constructor's non-object return
            // reads the *current* binding back out of the scope. Otherwise it exists only for
            // bodies that can name `this`.
            let class_kind = fn_obj.borrow().class_kind;
            let is_class_fn = class_kind != 0;
            let derived_tdz = is_construct && class_kind == crate::value::CLASS_DERIVED;
            if scan & crate::ast::SCAN_THIS != 0 || func.is_method || is_class_fn || is_construct {
                // Strict: pass through. Sloppy: undefined/null → global; primitive → box.
                let this_val = if func.is_strict || is_construct {
                    this
                } else {
                    match this.view() {
                        ValueView::Undefined | ValueView::Null => Value::Obj(self.global),
                        ValueView::Obj(_) => this,
                        // Sloppy mode: a primitive `this` is boxed to its wrapper (ToObject).
                        _ => crate::builtins::box_primitive_pub(self, this),
                    }
                };
                scope.borrow_mut().vars.insert(
                    crate::atom::names::THIS.atom(),
                    Binding::data(this_val, false, !derived_tdz),
                );
            }
            // A SuperCall is syntactically valid in a derived constructor and arrows nested
            // within it. The Function Environment Record outlives the call when such an arrow
            // escapes, so retain that lexical capability explicitly instead of relying only on
            // the interpreter's transient `super_call_ok` execution flag.
            if derived_tdz {
                scope.borrow_mut().vars.insert(
                    crate::atom::names::SUPER_CALL_OK.atom(),
                    Binding::data(Value::Bool(true), false, true),
                );
            }
            // The `arguments` exotic object is built only when the body (or a nested arrow, or a
            // possible direct eval) can name it. Otherwise a plain sloppy function stashes what a
            // reflective `fn.arguments` read would need to materialize it on demand.
            if scan & crate::ast::SCAN_ARGUMENTS != 0 {
                let ao = self.make_arguments_object(func, args, &scope, fn_obj);
                if let Some(frame) = self.fn_frames.last_mut() {
                    frame.extra.get_or_insert_with(Default::default).args_obj = Value::Obj(ao);
                }
                scope.borrow_mut().vars.insert(
                    crate::atom::names::ARGUMENTS.atom(),
                    Binding::data(Value::Obj(ao), true, true),
                );
            } else if !func.is_strict && !func.is_generator && !func.is_async && !func.is_method {
                if let Some(frame) = self.fn_frames.last_mut() {
                    frame.extra.get_or_insert_with(Default::default).lazy =
                        Some((func.clone(), Rc::from(args), scope));
                }
            }
            // Methods and class constructors may reference `super.x` — including from a direct
            // eval in a nested arrow, which resolves this marker lexically long after the call.
            if func.is_method || fn_obj.borrow().class_kind != 0 {
                scope.borrow_mut().vars.insert(
                    crate::atom::names::SUPER_PROP_OK.atom(),
                    Binding::data(Value::Bool(true), false, true),
                );
            }
            // A named expression's self-name was bound at closure creation, outside this scope.
        }

        // Parameter binding may throw (a default initializer, a destructuring mismatch, or an
        // EvalDeclarationInstantiation conflict). For an ordinary async function this abrupt
        // completion rejects the returned promise rather than throwing synchronously; every other
        // kind of function (including async *generators*) throws synchronously.
        let bind_result = self.bind_params(&func.params, args, &scope);

        // With parameter expressions, the body runs in a fresh variable environment below the
        // parameter scope; a `var` sharing a parameter's name starts with that parameter's value.
        let body = if has_param_exprs {
            new_var_scope(Some(scope))
        } else {
            scope
        };
        let param_seed = if has_param_exprs { Some(scope) } else { None };

        let saved_strict = self.strict;
        self.strict = func.is_strict;

        if func.is_async && !func.is_generator {
            if let Err(e) = bind_result {
                self.strict = saved_strict;
                self.new_target = saved_new_target;
                self.in_field_init_code = saved_field_init;
                self.in_async_gen_body = saved_agb;
                let reason = abrupt_value(e);
                let promise = self.new_promise();
                self.reject_promise(&promise, reason);
                return Ok(promise);
            }
            let r = self.run_async(func, &body, args);
            self.strict = saved_strict;
            self.new_target = saved_new_target;
            self.in_field_init_code = saved_field_init;
            self.in_async_gen_body = saved_agb;
            return r;
        }

        if let Err(e) = bind_result {
            self.strict = saved_strict;
            self.new_target = saved_new_target;
            self.in_field_init_code = saved_field_init;
            self.in_async_gen_body = saved_agb;
            return Err(e);
        }

        // Generators (sync and async) suspend at each yield on their own coroutine; see run_generator.
        if func.is_generator {
            // The generator object's [[Prototype]] comes from the function's own `.prototype`.
            let gen_proto = fn_obj.borrow().props.get("prototype").map(|p| p.value());
            let gen = self.run_generator(func, &body, gen_proto, args);
            self.strict = saved_strict;
            self.new_target = saved_new_target;
            self.in_field_init_code = saved_field_init;
            self.in_async_gen_body = saved_agb;
            return gen;
        }

        // Hoist `var`/function declarations into the function scope before executing the body.
        // Only the formal parameter names block Annex B promotions (see `hoist_fn_body` for a
        // block function named `arguments`).
        let param_names = param_bound_names(&func.params);
        self.hoist_fn_body(func, &body, &param_names);
        if let Some(ps) = &param_seed {
            self.seed_param_vars(ps, &body);
        }
        // Pre-declare body-level `let`/`const` in their temporal dead zone.
        self.declare_block_lexicals(&func.body, &body, false);

        // ECMA-262 #sec-functiondeclarationinstantiation: only representation
        // changes here. Default expressions/hoisting/TDZ creation ran exactly
        // once; closure objects and mapped arguments keep these same Envs.
        // Sharing applies before tier-up as well as to prepared compiled bodies.
        if cold_environment_layouts_enabled() && !body.borrow().under_with {
            let layouts = func.env_layouts.get_or_init(Box::default);
            if has_param_exprs {
                scope.borrow_mut().vars.share_layout(&layouts[0]);
            }
            body.borrow_mut().vars.share_layout(&layouts[1]);
        }

        // The general compiled entry shares the normative prologue above with the interpreter:
        // parameters/defaults/destructuring, mapped arguments, self-name and hoisted closures
        // execute once. Only body evaluation changes tier. Its chunk reuses these live bindings
        // rather than constructing a second activation or replaying initializers.
        if !matches!(self.tier, crate::bytecode::Tier::Interp)
            && !is_construct
            && allow_compiled_body
        {
            if let Some(chunk) = func
                .runtime()
                .execution_code_for(Env::as_ptr(&self.global_env) as usize)
                .and_then(Option::as_ref)
                .filter(|chunk| chunk.prepared_entry)
            {
                let this_value = if chunk.needs_frame_this() {
                    self.get_var("this", &body)
                } else {
                    Ok(Value::Undefined)
                };
                let result = this_value.and_then(|this_value| {
                    self.run_compiled_chunk(func, chunk, &body, this_value, &[], false)
                });
                self.strict = saved_strict;
                self.new_target = saved_new_target;
                self.in_field_init_code = saved_field_init;
                self.in_async_gen_body = saved_agb;
                return result;
            }
        }

        // The function body is a disposal boundary for its `using` declarations.
        let has_using = func.body.iter().any(crate::eval::stmt_declares_using);
        if has_using {
            self.using_stack.push(Vec::new());
        }
        // `return f(...)` is a proper tail call only in a strict, ordinary, non-constructor body.
        let saved_tco = std::mem::replace(
            &mut self.tco_ok,
            func.is_strict && !func.is_generator && !func.is_async && !is_construct,
        );
        // (The bytecode tier intercepted eligible calls at the top of this function; anything
        // reaching here — construct calls, uncompilable bodies — runs on the tree-walker.)
        let mut result = Ok(Value::Undefined);
        for stmt in &func.body {
            match self.exec_stmt(stmt, &body) {
                Ok(_) => {}
                Err(Abrupt::Return(v)) => {
                    result = Ok(v);
                    break;
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.tco_ok = saved_tco;
        if has_using {
            let frame = self.using_stack.pop().unwrap_or_default();
            result = self.dispose_frame(frame, result);
        }
        // [[Construct]]: a non-object return yields the *current* `this` binding — which a
        // derived constructor's super() may have rebound to a base constructor's returned object.
        // A derived constructor may only return an object or undefined (TypeError otherwise),
        // and its `this` must have been initialized by super() (ReferenceError otherwise); both
        // errors are created in the caller's realm (the callee context has conceptually popped).
        if is_construct && !func.is_arrow {
            let derived = fn_obj.borrow().class_kind == crate::value::CLASS_DERIVED;
            if let Ok(v) = &result {
                if !matches!((v).view(), ValueView::Obj(_)) {
                    if derived && !matches!((v).view(), ValueView::Undefined | ValueView::Empty) {
                        result = Err(self.throw_in_caller_realm(
                            "TypeError",
                            "a derived constructor may only return an object or undefined",
                        ));
                    } else {
                        let mut cur = Some(body);
                        while let Some(scope) = cur {
                            let found = scope
                                .borrow()
                                .vars
                                .get("this")
                                .map(|b| (b.value(), b.initialized));
                            if let Some((t, init)) = found {
                                result = if derived && !init {
                                    Err(self.throw_in_caller_realm(
                                        "ReferenceError",
                                        "derived constructor returned without calling super()",
                                    ))
                                } else {
                                    Ok(t)
                                };
                                break;
                            }
                            let parent = scope.borrow().parent;
                            cur = parent;
                        }
                    }
                }
            }
        }
        self.strict = saved_strict;
        self.new_target = saved_new_target;
        self.in_field_init_code = saved_field_init;
        self.in_async_gen_body = saved_agb;
        result
    }

    /// Start a generator in suspended-start and return its object. The heap VM continuation owns
    /// the execution context without reserving a native stack.
    fn run_generator(
        &mut self,
        func: &Rc<Function>,
        scope: &Env,
        gen_proto: Option<Value>,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        if func.runtime().code.get().is_none() {
            let _ = func.runtime().code.set(crate::bytecode::compile(func));
        }
        let continuation = if let Some(Some(chunk)) = func.runtime().code.get() {
            let this_val = if chunk.needs_frame_this() {
                self.get_var("this", scope)?
            } else {
                Value::Undefined
            };
            let params = self.coroutine_parameter_values(func, scope)?;
            let mut vm = if func.is_async {
                crate::bytecode::VmCoro::new_async_generator(
                    self,
                    chunk.clone(),
                    *scope,
                    this_val,
                    &params,
                    args,
                )
            } else {
                crate::bytecode::VmCoro::new_generator(
                    self,
                    chunk.clone(),
                    *scope,
                    this_val,
                    &params,
                    args,
                )
            };
            vm.frame = self.resumed_frame(func);
            crate::coroutine::Coroutine::Vm(Box::new(vm))
        } else {
            if std::env::var_os("TRUST_JS_TIER_LOG").is_some() {
                let source = func.source.as_deref().unwrap_or("<no source>");
                let head: String = source.chars().take(90).collect();
                eprintln!(
                    "[tier] unavailable VM coroutine: {}",
                    head.replace('\n', " ")
                );
            }
            let error = self.make_error("Error", crate::coroutine::VM_UNAVAILABLE_MSG);
            crate::coroutine::unavailable(error)
        };
        let object = self.make_generator(func.is_async, gen_proto);
        if let ValueView::Obj(value) = &(object).view() {
            self.gc_pin(value);
            self.generators
                .insert(&self.gc_heap, Gc::as_ptr(value) as usize, continuation);
            if func.is_async {
                self.async_gens.insert(Gc::as_ptr(value) as usize);
            }
        }
        Ok(object)
    }

    /// Read the already-instantiated formal parameter bindings for a generator/async VM frame.
    /// FunctionDeclarationInstantiation ran before the resumable execution context was created;
    /// seeding raw call arguments would make the chunk repeat default initialization on resume.
    fn coroutine_parameter_values(
        &mut self,
        func: &Function,
        scope: &Env,
    ) -> Result<Vec<Value>, Abrupt> {
        param_bound_names(&func.params)
            .into_iter()
            .map(|name| self.get_var(&name, scope))
            .collect()
    }

    /// Start an async function: spawn its coroutine, return a promise that settles when the body
    /// finishes. Each `await` parks the coroutine; a microtask resumes it once the awaited value
    /// settles.
    /// The compiled chunk for an async function if it fits the VM subset. Coroutine bytecode is
    /// also the continuation representation, not merely an optimization tier, so compile on the
    /// first call instead of reserving a native worker until an arbitrary hotness threshold.
    fn async_vm_chunk(&self, func: &Rc<Function>) -> Option<Rc<crate::bytecode::Chunk>> {
        // `Tier::Interp` disables bytecode as an ordinary optimization, but an async function's
        // chunk is also its heap-owned execution context. Use it whenever available so selecting
        // the reference tier never silently reintroduces an OS-thread continuation.
        if func.runtime().code.get().is_none() {
            let n = func.runtime().calls.get().saturating_add(1);
            func.runtime().calls.set(n);
            let _ = func.runtime().code.set(crate::bytecode::compile(func));
        }
        match func
            .runtime()
            .execution_code_for(Env::as_ptr(&self.global_env) as usize)
        {
            Some(Some(chunk)) => Some(chunk.clone()),
            _ => None,
        }
    }

    fn run_async(
        &mut self,
        func: &Rc<Function>,
        scope: &Env,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        // The bytecode VM suspends at each `await` with already-instantiated parameter values in
        // its slots; the original `args` are retained only for the arguments object.
        let coro = if let Some(chunk) = self.async_vm_chunk(func) {
            let this_val = if chunk.needs_frame_this() {
                self.get_var("this", scope)?
            } else {
                Value::Undefined
            };
            let params = self.coroutine_parameter_values(func, scope)?;
            let mut vm = crate::bytecode::VmCoro::new(self, chunk, *scope, this_val, &params, args);
            vm.frame = self.resumed_frame(func);
            crate::coroutine::Coroutine::Vm(Box::new(vm))
        } else {
            if std::env::var_os("TRUST_JS_TIER_LOG").is_some() {
                let source = func.source.as_deref().unwrap_or("<no source>");
                let head: String = source.chars().take(90).collect();
                eprintln!(
                    "[tier] unavailable VM coroutine: {}",
                    head.replace('\n', " ")
                );
            }
            let error = self.make_error("Error", crate::coroutine::VM_UNAVAILABLE_MSG);
            crate::coroutine::unavailable(error)
        };
        let promise = self.new_promise();
        if let ValueView::Obj(o) = &(promise).view() {
            self.gc_pin(o);
            self.generators
                .insert(&self.gc_heap, Gc::as_ptr(o) as usize, coro);
        }
        let key = match &(promise).view() {
            ValueView::Obj(o) => Gc::as_ptr(o) as usize,
            _ => unreachable!(),
        };
        self.drive_async(
            key,
            promise,
            crate::coroutine::Resume::Next(Value::Undefined),
        );
        Ok(promise)
    }

    /// Resume an async coroutine and react to how it parks: an `await` (Yield) attaches a microtask
    /// that re-drives it once the awaited value settles; completion settles the result promise.
    pub(crate) fn drive_async(
        &mut self,
        key: usize,
        promise: Value,
        signal: crate::coroutine::Resume,
    ) {
        use crate::coroutine::Suspend;
        let mut coro = match self.generators.remove(&key) {
            Some(c) => c,
            None => return,
        };
        let suspend = coro.resume(self, signal);
        match suspend {
            Suspend::Await(awaited) => {
                self.generators.insert(&self.gc_heap, key, coro); // still running
                                                                  // Await → PromiseResolve(%Promise%, value); its abrupt (a poisoned `constructor`
                                                                  // getter) throws at the `await` itself.
                let px = match self.promise_resolve_checked(awaited) {
                    Ok(p) => p,
                    Err(e) => {
                        return self.drive_async(key, promise, crate::coroutine::Resume::Throw(e))
                    }
                };
                self.perform_await_then(&px, &promise);
            }
            Suspend::Yield(v) => self.resolve_promise(&promise, v),
            Suspend::Done(v) => self.resolve_promise(&promise, v),
            Suspend::Throw(e) => self.reject_promise(&promise, e),
        }
    }

    /// Drive an async generator's coroutine, settling the `next()`/`return()`/`throw()` result
    /// promise `r` with `{value, done}`. An `await` parks the generator (the promise stays pending
    /// until a later `yield`/return); a `yield` fulfils the promise.
    pub(crate) fn drive_async_gen(
        &mut self,
        key: usize,
        r: Value,
        signal: crate::coroutine::Resume,
    ) {
        // A request while a step is in progress (running or awaiting) queues behind it.
        if self.async_gen_busy.contains(&key) {
            self.async_gen_queue
                .entry(&self.gc_heap, key)
                .or_default()
                .push_back((r, signal));
            return;
        }
        self.async_gen_busy.insert(key);
        self.drive_async_gen_inner(key, r, signal);
    }

    /// The step just completed (resolved or rejected its promise): pump the next queued request,
    /// or clear the busy flag.
    pub(crate) fn finish_async_gen_step(&mut self, key: usize) {
        match self
            .async_gen_queue
            .get_mut(&self.gc_heap, &key)
            .and_then(|q| q.pop_front())
        {
            Some((r, signal)) => self.drive_async_gen_inner(key, r, signal),
            None => {
                self.async_gen_busy.remove(&key);
            }
        }
    }

    pub(crate) fn drive_async_gen_inner(
        &mut self,
        key: usize,
        r: Value,
        signal: crate::coroutine::Resume,
    ) {
        use crate::coroutine::{Resume, Suspend};
        let mut coro = match self.generators.remove(&key) {
            Some(c) => c,
            None => {
                let res = self.iter_result_obj(Value::Undefined, true);
                self.resolve_promise(&r, res);
                return;
            }
        };
        if coro.done() {
            self.generators.insert(&self.gc_heap, key, coro);
            match signal {
                Resume::Throw(e) => {
                    self.reject_promise(&r, e);
                    self.finish_async_gen_step(key);
                }
                // AsyncGeneratorAwaitReturn: the returned value is awaited before the request
                // promise settles with { value, done: true }.
                Resume::Return(v) => self.async_gen_await_return(key, r, v),
                Resume::Next(_) => {
                    let res = self.iter_result_obj(Value::Undefined, true);
                    self.resolve_promise(&r, res);
                    self.finish_async_gen_step(key);
                }
                Resume::Terminate => {
                    let res = self.iter_result_obj(Value::Undefined, true);
                    self.resolve_promise(&r, res);
                    self.finish_async_gen_step(key);
                }
            }
            return;
        }
        // suspendedStart: a return() before the first next() awaits its value without ever
        // running the body.
        if !coro.started() {
            if let Resume::Return(v) = signal {
                self.generators.insert(&self.gc_heap, key, coro);
                return self.async_gen_await_return(key, r, v);
            }
        }
        let suspend = coro.resume(self, signal);
        self.generators.insert(&self.gc_heap, key, coro);
        match suspend {
            Suspend::Yield(v) => {
                let res = self.iter_result_obj(v, false);
                self.resolve_promise(&r, res);
                self.finish_async_gen_step(key);
            }
            Suspend::Await(x) => {
                // Still mid-step: the reaction resumes via drive_async_gen_inner, keeping the
                // request queue intact.
                let px = match self.promise_resolve_checked(x) {
                    Ok(p) => p,
                    Err(e) => return self.drive_async_gen_inner(key, r, Resume::Throw(e)),
                };
                let on_f = self.make_async_gen_reaction(key, &r, true);
                let on_r = self.make_async_gen_reaction(key, &r, false);
                self.promise_then(&px, on_f, on_r);
            }
            Suspend::Done(v) => {
                let res = self.iter_result_obj(v, true);
                self.resolve_promise(&r, res);
                self.finish_async_gen_step(key);
            }
            Suspend::Throw(e) => {
                self.reject_promise(&r, e);
                self.finish_async_gen_step(key);
            }
        }
    }

    /// AsyncGeneratorAwaitReturn: await the `return()` value; fulfilment completes the generator
    /// and settles the request with `{ value: awaited, done: true }`, rejection rejects it.
    fn async_gen_await_return(&mut self, key: usize, r: Value, v: Value) {
        let px = match self.promise_resolve_checked(v) {
            Ok(p) => p,
            Err(e) => {
                self.reject_promise(&r, e);
                self.finish_async_gen_step(key);
                return;
            }
        };
        let on_f = self.make_async_gen_return_reaction(key, &r, true);
        let on_r = self.make_async_gen_return_reaction(key, &r, false);
        self.promise_then(&px, on_f, on_r);
    }

    fn make_async_gen_return_reaction(&mut self, key: usize, r: &Value, fulfil: bool) -> Value {
        // AsyncGeneratorAwaitReturn's abstract closures capture the generator itself.
        // A numeric address is not an owning edge and can become stale across collection.
        let generator = Value::Obj(
            *self
                .gc_pins
                .get(&key)
                .expect("a pending async-generator return has a pinned owner"),
        );
        let target = self.make_native(
            "",
            1,
            if fulfil {
                crate::builtins::async_gen_return_fulfil
            } else {
                crate::builtins::async_gen_return_reject
            },
        );
        let bound = Object::new(Some(self.function_proto));
        bound.borrow_mut().call = Callable::bound(target, *r, vec![generator, *r]);
        Value::Obj(bound)
    }

    /// A `{ value, done }` iterator-result object.
    pub(crate) fn iter_result_obj(&mut self, value: Value, done: bool) -> Value {
        let o = self.new_object();
        {
            let mut b = o.borrow_mut();
            b.props
                .insert("value", Property::data(value, true, true, true));
            b.props
                .insert("done", Property::data(Value::Bool(done), true, true, true));
        }
        Value::Obj(o)
    }

    fn make_async_gen_reaction(&mut self, key: usize, r: &Value, fulfil: bool) -> Value {
        let generator = Value::Obj(
            *self
                .gc_pins
                .get(&key)
                .expect("a pending async-generator reaction has a pinned owner"),
        );
        let target = self.make_native(
            "",
            1,
            if fulfil {
                crate::builtins::async_gen_react_fulfil
            } else {
                crate::builtins::async_gen_react_reject
            },
        );
        let bound = Object::new(Some(self.function_proto));
        // Await's continuation owns the generator, not just its recyclable address.
        bound.borrow_mut().call = Callable::bound(target, *r, vec![generator, *r]);
        Value::Obj(bound)
    }

    /// PromiseResolve with the observable `Get(x, "constructor")` on a native promise — a
    /// poisoned getter surfaces as the returned error.
    pub(crate) fn promise_resolve_checked(&mut self, v: Value) -> Result<Value, Value> {
        if let ValueView::Obj(o) = &(v).view() {
            if self.promises.contains_key(&(Gc::as_ptr(o) as usize)) {
                return match self.get_member_key(&v, "constructor") {
                    Ok(_) => Ok(v),
                    Err(Abrupt::Throw(e)) => Err(e),
                    Err(_) => Err(Value::Undefined),
                };
            }
        }
        let p = self.new_promise();
        self.resolve_promise(&p, v);
        Ok(p)
    }

    fn bind_params(&mut self, params: &[Param], args: &[Value], scope: &Env) -> Result<(), Abrupt> {
        // With parameter expressions, every parameter's binding is created (uninitialized) up front,
        // so a default initializer naming a later parameter is a temporal-dead-zone reference and a
        // direct `eval` observes the parameter bindings (matching FunctionDeclarationInstantiation).
        if params_have_expr(params) {
            let mut names = Vec::new();
            for p in params {
                pattern_idents(&p.pattern, &mut names);
            }
            for name in names {
                scope
                    .borrow_mut()
                    .vars
                    .insert(name, Binding::data(Value::Undefined, true, false));
            }
        }
        for (i, p) in params.iter().enumerate() {
            let value = if p.rest {
                let rest: Vec<Value> = args.iter().skip(i).cloned().collect();
                self.make_array(rest)
            } else {
                let mut v = args.get(i).cloned().unwrap_or(Value::Undefined);
                if matches!((v).view(), ValueView::Undefined) {
                    if let Some(d) = &p.default {
                        v = self.eval(d, scope)?;
                        if let (crate::ast::Pattern::Ident(n), true) =
                            (&p.pattern, crate::eval::is_anonymous_fn(d))
                        {
                            self.set_fn_name(&v, n);
                        }
                    }
                }
                v
            };
            self.bind_pattern(&p.pattern, value, scope, BindMode::Lexical(false))?;
            if p.rest {
                break;
            }
        }
        Ok(())
    }

    /// When a function has parameter expressions, a body `var` sharing a parameter's name starts
    /// with that parameter's value (FunctionDeclarationInstantiation step for the separate variable
    /// environment). `hoist` has just created the body's `var` bindings as `undefined`; fill in the
    /// initial value from the parameter scope for names it declares.
    fn seed_param_vars(&self, param_scope: &Env, body_scope: &Env) {
        let names: Vec<String> = body_scope
            .borrow()
            .vars
            .keys()
            .map(|k| k.to_string())
            .collect();
        for name in names {
            if name == "this" {
                continue;
            }
            let seeded = param_scope
                .borrow()
                .vars
                .get(name.as_str())
                .filter(|b| b.initialized)
                .map(|b| b.value());
            if let Some(v) = seeded {
                let mut bs = body_scope.borrow_mut();
                if let Some(b) = bs.vars.get_mut(&name) {
                    // Only a plain `var` slot (still `undefined`) inherits the parameter value; a
                    // function declaration keeps its function object.
                    if matches!((b.value()).view(), ValueView::Undefined) {
                        b.set_value(v);
                    }
                }
            }
        }
    }

    pub(crate) fn construct(&mut self, callee: Value, args: &[Value]) -> Result<Value, Abrupt> {
        let nt = callee;
        self.construct_nt(callee, args, nt)
    }

    /// A trapless Proxy forwards the operation to its target (ECMA-262 §10.5), and the target's
    /// prototype chain may lead back to the same Proxy: [[SetPrototypeOf]]'s cycle check stops
    /// at Proxies. Account each forward like a call, and bound how many forwards may nest, so an
    /// unbounded chain throws RangeError promptly instead of exhausting the native stack.
    pub(crate) fn proxy_forward<R>(
        &mut self,
        forward: impl FnOnce(&mut Self) -> Result<R, Abrupt>,
    ) -> Result<R, Abrupt> {
        if self.proxy_forward_depth >= MAX_PROXY_FORWARD_DEPTH {
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        self.proxy_forward_depth += 1;
        let result = with_execution_stack(self.depth, || forward(self));
        self.proxy_forward_depth -= 1;
        self.depth -= 1;
        result
    }

    /// Like `construct`, but with an explicit `new.target` (for `Reflect.construct`'s third argument
    /// and a proxy's `[[Construct]]` forwarding, where new.target differs from the callee).
    pub(crate) fn construct_nt(
        &mut self,
        callee: Value,
        args: &[Value],
        new_target: Value,
    ) -> Result<Value, Abrupt> {
        // I1: callers often hold the argument list in a Rust-heap buffer.
        let _arguments = crate::rooting::root_slice(args);
        self.depth += 1;
        if execution_stack_exhausted(self.depth) {
            self.depth -= 1;
            return Err(self.throw("RangeError", "Maximum call stack size exceeded"));
        }
        let r = with_execution_stack(self.depth, || {
            self.construct_inner(callee, args, new_target)
        });
        self.depth -= 1;
        r
    }

    fn construct_inner(
        &mut self,
        callee: Value,
        args: &[Value],
        new_target: Value,
    ) -> Result<Value, Abrupt> {
        // IsConstructor is checked in the *caller's* realm, before any realm swap — the
        // TypeError for `new nonCtor` belongs to the code doing the `new`.
        if !matches!(&(callee).view(), ValueView::Obj(_)) || !self.value_is_constructor(&callee) {
            return Err(self.throw(
                "TypeError",
                format!("{} is not a constructor", type_name(&callee)),
            ));
        }
        // With one Realm and one interpreter, every constructor already runs in the active Realm.
        let foreign_possible = self.multi_realm() || !self.single_interpreter();
        if let Some(object) = callee.as_obj().filter(|_| foreign_possible) {
            if self.proxy_at(Gc::as_ptr(&object) as usize).is_none() {
                if let Some((owner, realm)) = self.callable_home(&object) {
                    if owner != self.realm_id {
                        let caller = crate::rooting::Rooted::new(self.snapshot_realm());
                        return self.with_route(owner, |target| {
                            target.with_constructor_caller(
                                Some(caller.snapshot_clone()),
                                |target| {
                                    if realm == Gc::as_ptr(&target.global) as usize {
                                        target.construct_dispatch(callee, args, new_target)
                                    } else {
                                        let state = target.realms[&realm].snapshot_clone();
                                        target.with_realm_snapshot(realm, state, |target| {
                                            target.construct_dispatch(callee, args, new_target)
                                        })
                                    }
                                },
                            )
                        });
                    }
                    if realm != Gc::as_ptr(&self.global) as usize {
                        let caller = self.snapshot_realm();
                        let state = self.realms[&realm].snapshot_clone();
                        return self.with_realm_snapshot(realm, state, |target| {
                            target.with_constructor_caller(Some(caller), |target| {
                                target.construct_dispatch(callee, args, new_target)
                            })
                        });
                    }
                }
            }
        }
        self.construct_dispatch(callee, args, new_target)
    }

    fn construct_native_dispatch(
        &mut self,
        obj: &Gc,
        call: &Callable,
        args: &[Value],
        new_target: Value,
    ) -> Result<Value, Abrupt> {
        // A native non-constructor (a method, global function, Math fn) has no own
        // `prototype` property; only real built-in constructors do. Reject `new` on the rest.
        let constructable = obj.borrow().is_constructor || obj.borrow().props.contains("prototype");

        if !constructable {
            return Err(self.throw("TypeError", "function is not a constructor"));
        }
        // Built-in constructors build and return their own object. The `constructing` flag
        // lets wrapper constructors (Number/String/...) distinguish `new X()` from `X()`.
        // `new_target` is exposed so a native constructor can derive the instance prototype
        // from it (OrdinaryCreateFromConstructor / subclassing / Reflect.construct).
        let saved = self.constructing;
        let saved_nt = self.new_target;
        self.constructing = true;
        self.new_target = new_target;
        let r = self.dispatch_native(call, Value::Undefined, args);
        self.constructing = saved;
        self.new_target = saved_nt;
        r
    }

    fn construct_dispatch(
        &mut self,
        callee: Value,
        args: &[Value],
        new_target: Value,
    ) -> Result<Value, Abrupt> {
        let obj = match &(callee).view() {
            ValueView::Obj(o) => *o,
            _ => return Err(self.throw("TypeError", "value is not a constructor")),
        };
        // Proxy with a `construct` trap (or forward to the target).
        {
            if let Some((target, handler)) = self.proxy_at(Gc::as_ptr(&obj) as usize) {
                if matches!((handler).view(), ValueView::Null) {
                    return Err(self.throw("TypeError", "proxy is revoked"));
                }
                let Some(trap) =
                    self.proxy_get_trap(&handler, PROXY_TRAP_CONSTRUCT, "construct")?
                else {
                    // Forward to the target's [[Construct]] with the original new.target.
                    return self.construct_inner(target, args, new_target);
                };
                let arr = self.make_array(args.to_vec());
                let result = self.proxy_call_trap(
                    PROXY_TRAP_CONSTRUCT,
                    &trap,
                    handler,
                    [target, arr, new_target],
                )?;
                // [[Construct]] invariant: the trap must return an Object.
                if !matches!((result).view(), ValueView::Obj(_)) {
                    return Err(
                        self.throw("TypeError", "proxy construct trap returned a non-object")
                    );
                }
                return Ok(result);
            }
        }
        let call = obj.borrow().call.clone();
        match call {
            Callable::Native(_) | Callable::NativeData(_) => {
                self.construct_native_dispatch(&obj, &call, args, new_target)
            }
            #[cfg(feature = "embed")]
            Callable::ScopedNative(_) => {
                self.construct_native_dispatch(&obj, &call, args, new_target)
            }
            Callable::User(user) => {
                // Arrows, concise methods, getters/setters, generators and async functions have no
                // [[Construct]].
                if user.func.is_arrow
                    || user.func.is_method
                    || user.func.is_generator
                    || user.func.is_async
                {
                    return Err(self.throw("TypeError", "this function is not a constructor"));
                }
                // ECMA-262 §10.2.2 [[Construct]] step 3 allocates only for a base constructor. A
                // derived constructor's `this` is bound by super(), whose base constructor reads
                // new.target's `prototype` exactly once for the whole chain.
                let class_kind = obj.borrow().class_kind;
                if class_kind == crate::value::CLASS_DERIVED {
                    if user.func.default_ctor {
                        return self.construct_default_derived(&obj, args, new_target);
                    }
                    self.pending_new_target = new_target;
                    return self.run_constructor_on(&callee, &Value::Undefined, args);
                }
                // OrdinaryCreateFromConstructor: the new instance's prototype comes from
                // new.target; a non-object `prototype` falls back to new.target's realm's
                // %Object.prototype% (GetFunctionRealm).
                let proto = match (self.get_member_key(&new_target, "prototype")?).view() {
                    ValueView::Obj(p) => Some(p),
                    _ => {
                        let realm = match &(new_target).view() {
                            ValueView::Obj(nt) => self.get_function_realm_global(nt)?,
                            _ => None,
                        };
                        realm
                            .and_then(|g| self.realms.get(&g).map(|rs| rs.object_proto))
                            .or(Some(self.object_proto))
                    }
                };
                let layout = self.learned_construct_layout(&obj);
                let capacity = layout.as_ref().map_or(0, |keys| keys.len());
                let this = Object::new_with_parts(
                    proto,
                    crate::value::Props::with_layout(capacity, layout),
                    crate::value::Exotic::None,
                );
                let this_val = Value::Obj(this);
                self.pending_new_target = new_target;
                // Base class constructors run their field initializers before the body; plain
                // function constructors just run their body.
                let ret = if class_kind != 0 {
                    self.run_constructor_on(&callee, &this_val, args)?
                } else {
                    self.call_user(&user.func, user.env, this_val, args, true, &obj)?
                };
                // A forwarding constructor's own body may contain no direct field stores. Record
                // the small final named-map size after a successful call so its next instance has
                // spare capacity for the machine-code creation caches. The hint changes only
                // allocation size; all writes still perform their normal live semantic guards.
                self.observe_construct_capacity(&obj, &this);
                // A constructor explicitly returning an object overrides the instance.
                Ok(match (ret).view() {
                    ValueView::Obj(_) => ret,
                    _ => this_val,
                })
            }
            Callable::Bound(bound) => {
                let mut all = bound.args.clone();
                all.extend_from_slice(args);
                // BoundFunction [[Construct]]: if new.target is the bound function itself, use the
                // bound target as new.target instead.
                let nt = if self.strict_equals(&new_target, &callee) {
                    Value::Obj(bound.target)
                } else {
                    new_target
                };
                self.construct_nt(Value::Obj(bound.target), &all, nt)
            }
            Callable::None
            | Callable::WrappedShadow(_)
            | Callable::WrappedCross(_)
            | Callable::AccessorGet(_)
            | Callable::AccessorSet(_)
            | Callable::PropGet(_)
            | Callable::PropSet(_)
            | Callable::PrivatePropGet(_)
            | Callable::PrivatePropSet(_) => {
                Err(self.throw("TypeError", "value is not a constructor"))
            }
        }
    }

    /// The default constructor of a derived class (ECMA-262 ClassDefinitionEvaluation, step
    /// 14.a): Construct(F.[[GetPrototypeOf]](), args, NewTarget), then
    /// InitializeInstanceElements(result, F). It is a built-in abstract closure, so it creates
    /// no Function Environment Record and its argument forwarding iterates nothing. The frame
    /// remains visible to stack capture like the synthesized body it replaces.
    fn construct_default_derived(
        &mut self,
        ctor: &Gc,
        args: &[Value],
        new_target: Value,
    ) -> Result<Value, Abrupt> {
        self.fn_frames.push(FnFrame {
            fn_ptr: Gc::as_ptr(ctor) as usize,
            call_site: std::mem::replace(&mut self.call_site, crate::stack_trace::NO_POSITION),
            strict: true,
            construct: true,
            extra: None,
        });
        let result = (|| {
            // GetSuperConstructor, exactly as a written super() call performs it.
            let parent = crate::builtins::js_get_prototype_of(self, &Value::Obj(*ctor))
                .map_err(Abrupt::Throw)?;
            if !self.value_is_constructor(&parent) {
                return Err(self.throw("TypeError", "super constructor is not a constructor"));
            }
            let result = self.construct_nt(parent, args, new_target)?;
            self.init_instance_fields(&Value::Obj(*ctor), &result)?;
            Ok(result)
        })();
        self.pop_fn_frame();
        result
    }

    // ----- program / statement execution ------------------------------------------------------

    /// Proxy `[[Get]]` invariant: a non-configurable non-writable data property on the target must be
    /// reported with its actual value; a non-configurable accessor with no getter must report
    /// undefined. (`Abrupt` carries the thrown TypeError.)
    /// ECMA-262 §10.5.8 steps 8-9: validate a `get` trap result against
    /// target.[[GetOwnProperty]](P), which may itself run a Proxy target's traps.
    fn proxy_get_invariant(
        &mut self,
        target: &Value,
        key: &str,
        result: &Value,
    ) -> Result<(), Abrupt> {
        if Self::proxy_invariant_vacuous(target, key) {
            return Ok(());
        }
        let prop = match &(target).view() {
            ValueView::Obj(t) => {
                crate::builtins::object_get_own_property(self, t, key).map_err(Abrupt::Throw)?
            }
            _ => None,
        };
        if let Some(p) = prop {
            if !p.configurable() {
                let bad = if p.accessor() {
                    p.getter()
                        .is_none_or(|value| matches!(value.view(), ValueView::Undefined))
                        && !matches!(&(result).view(), ValueView::Undefined)
                } else {
                    !p.writable() && !crate::builtins::same_value_pub(result, &p.value())
                };
                if bad {
                    return Err(self.throw(
                        "TypeError",
                        "proxy 'get' trap violated an invariant for a non-configurable property",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Proxy `[[Set]]` invariant: a `true` result can't contradict a non-configurable non-writable
    /// data property (value must match) or a non-configurable accessor with no setter.
    pub(crate) fn proxy_set_invariant(
        &mut self,
        target: &Value,
        key: &str,
        value: &Value,
    ) -> Result<(), Abrupt> {
        // ECMA-262 §10.5.9 steps 9-10: target.[[GetOwnProperty]](P), which may itself run a
        // Proxy target's traps.
        if Self::proxy_invariant_vacuous(target, key) {
            return Ok(());
        }
        let prop = match &(target).view() {
            ValueView::Obj(t) => {
                crate::builtins::object_get_own_property(self, t, key).map_err(Abrupt::Throw)?
            }
            _ => None,
        };
        if let Some(p) = prop {
            if !p.configurable() {
                let bad = if p.accessor() {
                    p.setter()
                        .is_none_or(|value| matches!(value.view(), ValueView::Undefined))
                } else {
                    !p.writable() && !crate::builtins::same_value_pub(value, &p.value())
                };
                if bad {
                    return Err(self.throw(
                        "TypeError",
                        "proxy 'set' trap violated an invariant for a non-configurable property",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn run_program(&mut self, body: &[Stmt]) -> Result<Value, Abrupt> {
        self.run_program_from(body, None)
    }

    /// [`Interp::run_program`] for a body parsed from `source`, whose top-level code then shows
    /// in stack traces (code without a source, such as a host snapshot, does not).
    pub(crate) fn run_program_from(
        &mut self,
        body: &[Stmt],
        source: Option<Rc<crate::stack_trace::ScriptSource>>,
    ) -> Result<Value, Abrupt> {
        // The program allocates in this Agent's heap, whichever was active when it was called;
        // this Agent stays the surrounding one afterwards, as the engine's entry points leave it.
        self.activate_gc_heap();
        // ScriptEvaluation: the Script Record is the ScriptOrModule of its code; its host-
        // defined base is the host's current script base (HTML: the classic script's base URL).
        if let Some(source) = &source {
            if source.kind() == crate::stack_trace::SourceKind::Script {
                source.set_referrer(Rc::from(self.import_base.as_str()));
            }
        }
        self.in_code(source, |this| {
            this.with_script_entry(|this| this.run_program_body(body))
        })
    }

    /// ScriptEvaluation of a classic Script body in the active Realm: its own directive
    /// prologue decides strictness, and the caller's mode is restored afterwards.
    #[cfg(any(feature = "embed", test))]
    pub(crate) fn run_classic_program(
        &mut self,
        body: &[Stmt],
        source: Option<Rc<crate::stack_trace::ScriptSource>>,
    ) -> Result<Value, Abrupt> {
        let directive_strict = matches!(
            body.first(),
            Some(Stmt::Expr(Expr::Str(value))) if &**value == "use strict"
        );
        let previous_strict = std::mem::replace(&mut self.strict, directive_strict);
        let result = self.run_program_from(body, source);
        self.strict = previous_strict;
        result
    }

    /// The Script body of a static host snapshot, decoded on first use and then shared by every
    /// Realm of this Agent that evaluates it (see [`Interp::shared_snapshots`]).
    #[cfg(any(feature = "embed", test))]
    pub(crate) fn shared_snapshot_program(
        &mut self,
        bytes: &'static [u8],
    ) -> Result<Rc<Vec<Stmt>>, String> {
        let (address, len) = (bytes.as_ptr() as usize, bytes.len());
        if let Some(entry) = self
            .shared_snapshots
            .iter()
            .find(|entry| entry.address == address && entry.len == len)
        {
            return Ok(entry.body.clone());
        }
        // Function bodies stay encoded in the static bytes until each function first needs its
        // statements (see `ast::FunctionBody`): platform code this Agent never runs costs it
        // nothing beyond the bytes the process already holds.
        let body = Rc::new(crate::snapshot::decode_static(bytes)?);
        // Embedders install a handful of distinct bootstraps; keep the table bounded if one
        // instead passes many different static snapshots.
        const MAX_SHARED_SNAPSHOTS: usize = 8;
        if self.shared_snapshots.len() == MAX_SHARED_SNAPSHOTS {
            self.shared_snapshots.remove(0);
        }
        self.shared_snapshots.push(SharedSnapshot {
            address,
            len,
            body: body.clone(),
        });
        Ok(body)
    }

    fn run_program_body(&mut self, body: &[Stmt]) -> Result<Value, Abrupt> {
        self.activate_gc_heap();
        self.interrupt_poll_force()?;
        // GlobalDeclarationInstantiation validates declarations before creating
        // bindings or function objects. The hoist plan supplies names and kinds;
        // no throwaway scope or duplicate function/prototype objects are needed.
        let hoist_ops = collect_hoist_ops(body, self.strict, &[]);
        let new_vars = ordered_global_hoist_bindings(&hoist_ops);
        let mut lex_names: Vec<String> = Vec::new();
        for s in body {
            match unwrap_export(s) {
                Stmt::VarDecl {
                    kind:
                        crate::ast::DeclKind::Let
                        | crate::ast::DeclKind::Const
                        | crate::ast::DeclKind::Using
                        | crate::ast::DeclKind::AwaitUsing,
                    decls,
                } => {
                    for (pat, _) in decls {
                        pattern_idents(pat, &mut lex_names);
                    }
                }
                Stmt::ClassDecl(c) => lex_names.extend(c.name.clone()),
                _ => {}
            }
        }
        for n in &lex_names {
            // HasVarDeclaration / HasLexicalDeclaration / HasRestrictedGlobalProperty.
            let restricted = self
                .global
                .borrow()
                .props
                .get(n.as_str())
                .is_some_and(|p| !p.configurable());
            if self.global_env.borrow().vars.contains_key(n.as_str())
                || self.global_var_names.borrow().contains(n)
                || restricted
            {
                return Err(Abrupt::Throw(self.make_error(
                    "SyntaxError",
                    format!("Identifier '{n}' has already been declared"),
                )));
            }
        }
        let extensible = self.global.borrow().extensible;
        for &(name, is_func) in &new_vars {
            if self.global_env.borrow().vars.contains_key(name) {
                return Err(Abrupt::Throw(self.make_error(
                    "SyntaxError",
                    format!("Identifier '{name}' has already been declared"),
                )));
            }
            let existing = self
                .global
                .borrow()
                .props
                .get(name)
                .map(|p| (p.configurable(), p.accessor(), p.writable(), p.enumerable()));
            if is_func {
                // CanDeclareGlobalFunction.
                let ok = match existing {
                    Some((true, ..)) => true,
                    Some((false, accessor, writable, enumerable)) => {
                        !accessor && writable && enumerable
                    }
                    None => extensible,
                };
                if !ok {
                    return Err(Abrupt::Throw(self.make_error(
                        "TypeError",
                        format!("cannot declare global function '{name}'"),
                    )));
                }
            } else if existing.is_none() && !extensible {
                // CanDeclareGlobalVar.
                return Err(Abrupt::Throw(self.make_error(
                    "TypeError",
                    format!("cannot declare global variable '{name}'"),
                )));
            }
        }
        self.global_var_names
            .borrow_mut()
            .extend(new_vars.iter().map(|(name, _)| (*name).to_owned()));
        self.apply_hoist_ops(&hoist_ops, &self.global_env.clone(), false);
        // The global Environment Record is object-backed: this script's `var`/`function` bindings
        // (which `hoist` just placed in `global_env`) become properties of the global object, so
        // they are visible as `globalThis.<name>` and writes stay in sync. Only the plan-computed
        // names move — lexical bindings from earlier scripts stay in the environment.
        for (name, is_func) in new_vars {
            let Some(binding) = self.global_env.borrow_mut().vars.remove(name) else {
                continue;
            };
            let existing = self
                .global
                .borrow()
                .props
                .get(name)
                .map(|p| (p.writable(), p.configurable()));
            match existing {
                None => {
                    self.global
                        .borrow_mut()
                        .props
                        .insert(name, Property::data(binding.value(), true, true, false));
                }
                // CreateGlobalFunctionBinding: a configurable property is fully redefined; a
                // non-configurable one (already validated writable) just takes the new value.
                Some((_, true)) if is_func => {
                    self.global
                        .borrow_mut()
                        .props
                        .insert(name, Property::data(binding.value(), true, true, false));
                }
                Some((true, _)) if is_func => {
                    self.global
                        .borrow_mut()
                        .props
                        .set_value(name, binding.value());
                }
                _ => {}
            }
        }
        drop(hoist_ops);
        // Top-level `let`/`const` are pre-declared in their temporal dead zone.
        self.declare_block_lexicals(body, &self.global_env.clone(), false);
        let env = self.global_env;
        // Cold Script bodies retain the cheap reference path, including short/zero loops.
        // Settled hot AST loop backedges admit a borrowed fragment, then that exact VM frame
        // may enter native code. Threshold zero deliberately forces whole-Script coverage.
        let forced_tier = self.tier_threshold == 0;
        if !matches!(self.tier, crate::bytecode::Tier::Interp) && forced_tier {
            if let Some(chunk) = crate::bytecode::compile_script(body, self.strict) {
                // ScriptEvaluation has already installed the Script/Realm context and completed
                // GDI. Enter the exact global Environment Record, without a Function frame or
                // FunctionDeclarationInstantiation. A failed compile falls back before any body
                // operation executes; a runtime abrupt completion is never replayed.
                let mut code = None;
                if forced_tier && matches!(self.tier, crate::bytecode::Tier::Jit) {
                    let layout = *self
                        .jit_layout
                        .get_or_init(|| crate::value::jit_layout(&self.object_proto));
                    if !self.interp_layout.get().valid {
                        let interp_offsets = interp_layout(self);
                        self.interp_layout.set(interp_offsets);
                    }
                    if let crate::jit::JitCompileOutcome::Compiled(native) =
                        crate::jit::compile_profiled(&chunk, &layout, &self.interp_layout.get())
                    {
                        code = Some(Rc::new(native));
                    }
                }
                // The Environment Record owns [[GlobalThisValue]], which an embedder may
                // distinguish from the object backing global properties (for example a proxy).
                let this = self.get_var("this", &env)?;
                #[cfg(test)]
                crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| {
                    let (vm, native) = counts.get();
                    counts.set((
                        vm + usize::from(code.is_none()),
                        native + usize::from(code.is_some()),
                    ));
                });
                return if let Some(code) = code {
                    crate::jit::run(self, &chunk, &code, &env, this, &[])
                } else {
                    crate::bytecode::run(self, &chunk, &env, this, &[])
                };
            }
        }
        let mut last = Value::Undefined;
        for stmt in body {
            match self.exec_stmt(stmt, &env) {
                Ok(v) => {
                    if !matches!((v).view(), ValueView::Empty) {
                        last = v;
                    }
                }
                Err(Abrupt::Throw(v)) => return Err(Abrupt::Throw(v)),
                Err(Abrupt::Interrupt(reason)) => return Err(Abrupt::Interrupt(reason)),
                Err(_) => return Ok(last), // stray break/continue/return at top level: stop
            }
        }
        Ok(last)
    }

    /// Hoist `var` and function declarations into `scope`. `let`/`const` get TDZ bindings created
    /// at block entry instead (see [`Self::exec_block`]). `param_blocked` carries the formal
    /// parameter names for *function code* — per B.3.3.1 they block the Annex B var-promotion of
    /// same-named block functions (eval and global code pass none; B.3.3.2/3 have no such clause).
    pub(crate) fn hoist(&mut self, stmts: &[Stmt], scope: &Env, param_blocked: &[String]) {
        let ops = collect_hoist_ops(stmts, self.strict, param_blocked);
        self.apply_hoist_ops(&ops, scope, false);
    }

    /// [`Self::hoist`] for a function body, with the hoisting plan cached on the [`Function`]
    /// (the body and parameter names never change, so the plan is computed once per function).
    pub(crate) fn hoist_fn_body(
        &mut self,
        func: &Rc<Function>,
        scope: &Env,
        param_blocked: &[String],
    ) {
        let strict = self.strict;
        let cached = func.hoist.get_or_init(|| {
            (
                strict,
                Rc::new(collect_hoist_ops(&func.body, strict, param_blocked)),
            )
        });
        if cached.0 == strict {
            let ops = cached.1.clone();
            self.apply_hoist_ops(&ops, scope, true);
        } else {
            // Strictness differs from the cached computation (shouldn't happen — a function's
            // strictness is fixed at parse) — fall back to an uncached walk.
            let ops = collect_hoist_ops(&func.body, strict, param_blocked);
            self.apply_hoist_ops(&ops, scope, true);
        }
    }

    /// Replay a hoisting plan against `scope` (see [`HoistOp`] for the op semantics).
    /// `function_code` selects FunctionDeclarationInstantiation's Annex B rule (ECMA-262
    /// §10.2.11, web-compat insertion point): a block function named `arguments` gets no var
    /// binding there. Evaluating its declaration sets the VariableEnvironment's `arguments`
    /// binding (the arguments object's, when there is one) or creates it.
    pub(crate) fn apply_hoist_ops(&mut self, ops: &[HoistOp], scope: &Env, function_code: bool) {
        for op in ops {
            match op {
                HoistOp::Var(name) => {
                    if !scope.borrow().vars.contains_key(name.as_str()) {
                        scope
                            .borrow_mut()
                            .vars
                            .insert(name.clone(), undef_var_binding());
                    }
                }
                HoistOp::Fn(name, func) => {
                    let f = self.make_function(func.clone(), *scope);
                    if name == "*default*" {
                        self.set_fn_name(&f, "default");
                    }
                    scope
                        .borrow_mut()
                        .vars
                        .insert(name.clone(), Binding::data(f, true, true));
                }
                HoistOp::AnnexB(name, func) => {
                    if !(function_code && name == "arguments")
                        && !scope.borrow().vars.contains_key(name.as_str())
                    {
                        scope
                            .borrow_mut()
                            .vars
                            .insert(name.clone(), undef_var_binding());
                    }
                    self.annexb_fn_sync
                        .insert(Rc::as_ptr(func) as usize, func.clone());
                }
            }
        }
    }
}

/// The undefined, mutable, function-scoped binding hoisting creates for a `var` name.
fn cold_environment_layouts_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TRUST_JS_COLD_ENV_LAYOUTS").as_deref() != Ok("0"))
}

fn undef_var_binding() -> Binding {
    Binding::data(Value::Undefined, true, true)
}

/// Build the hoisting plan for a statement list: `var` names in traversal order, then function
/// declarations in source order, then (sloppy mode) Annex B block-function promotions. Pure — the
/// plan for a function body is cached on its [`Function`] and replayed each call.
pub(crate) fn collect_hoist_ops(
    stmts: &[Stmt],
    strict: bool,
    param_blocked: &[String],
) -> Vec<HoistOp> {
    let mut out = Vec::new();
    for stmt in stmts {
        collect_hoist_stmt(stmt, &mut out);
    }
    // Function declarations are also initialised eagerly (in source order, after var names). An
    // anonymous `export default function(){}` is a HoistableDeclaration too, bound to
    // `*default*`. Annex B: labels over a top-level function declaration are transparent.
    for stmt in stmts {
        let mut inner = unwrap_export(stmt);
        while let (false, Stmt::Labeled { body, .. }) = (strict, inner) {
            inner = body;
        }
        if let Stmt::FuncDecl(func) = inner {
            let name = match &func.name {
                Some(n) => n.clone(),
                None if matches!(stmt, Stmt::ExportDefault(_)) => "*default*".to_string(),
                None => continue,
            };
            out.push(HoistOp::Fn(name, func.clone()));
        }
    }
    // Annex B.3.3: in sloppy mode, a function declared inside a block (or an if/label
    // substatement position) also gets a `var`-style binding in the enclosing function/global
    // scope — initialized to undefined, synced with the block binding when the declaration is
    // evaluated — unless a lexical declaration between here and the block makes the equivalent
    // `var` an early error.
    if !strict {
        let mut blocked: Vec<String> = block_lexical_names(stmts);
        blocked.extend(param_blocked.iter().cloned());
        for stmt in stmts {
            collect_annexb_funcs(stmt, false, &mut blocked, &mut out);
        }
    }
    out
}

/// Property creation order for GlobalDeclarationInstantiation and global
/// EvalDeclarationInstantiation. Scope bindings are hashed; iterating that table
/// would make the observable global object's own-key order depend on its buckets.
/// Annex B-only vars precede surviving function declarations (last declaration
/// wins), which precede ordinary vars in their first-occurrence order.
pub(crate) fn ordered_global_hoist_bindings(ops: &[HoistOp]) -> Vec<(&str, bool)> {
    // Borrow plan-owned names. The flag identifies a function binding without
    // allocating a function merely to infer its declaration kind from a value.
    let mut seen: crate::fasthash::FastSet<&str> = Default::default();
    let mut functions_and_vars = Vec::new();
    for op in ops.iter().rev() {
        if let HoistOp::Fn(name, _) = op {
            if seen.insert(name.as_str()) {
                functions_and_vars.push((name.as_str(), true));
            }
        }
    }
    functions_and_vars.reverse();
    for op in ops {
        if let HoistOp::Var(name) = op {
            if seen.insert(name.as_str()) {
                functions_and_vars.push((name.as_str(), false));
            }
        }
    }
    let mut ordered = Vec::with_capacity(seen.len());
    for op in ops {
        if let HoistOp::AnnexB(name, _) = op {
            if seen.insert(name.as_str()) {
                ordered.push((name.as_str(), false));
            }
        }
    }
    ordered.extend(functions_and_vars);
    ordered
}

/// Hoist `var` declarations from one statement (function declarations and Annex B promotions are
/// collected by the later phases). ECMA-262 §10.2.11 FunctionDeclarationInstantiation creates a
/// var binding only when the name is not already a parameter/var binding; a `for` head does not
/// get different instantiation semantics.
fn collect_hoist_stmt(stmt: &Stmt, out: &mut Vec<HoistOp>) {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| collect_hoist_stmt(stmt, out));
    }
    match stmt {
        Stmt::ExportDecl(inner) | Stmt::ExportDefault(inner) => collect_hoist_stmt(inner, out),
        Stmt::VarDecl {
            kind: DeclKind::Var,
            decls,
        } => {
            for (pat, _) in decls {
                let mut names = Vec::new();
                pattern_idents(pat, &mut names);
                out.extend(names.into_iter().map(HoistOp::Var));
            }
        }
        Stmt::If { cons, alt, .. } => {
            collect_hoist_stmt(cons, out);
            if let Some(a) = alt {
                collect_hoist_stmt(a, out);
            }
        }
        Stmt::Block(body) => {
            for s in body {
                collect_hoist_stmt(s, out);
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_hoist_stmt(body, out),
        Stmt::Labeled { body, .. } | Stmt::With { body, .. } => collect_hoist_stmt(body, out),
        Stmt::For { init, body, .. } => {
            if let Some(init) = init {
                if let ForInit::VarDecl {
                    kind: DeclKind::Var,
                    decls,
                } = init.as_ref()
                {
                    for (pat, _) in decls {
                        let mut names = Vec::new();
                        pattern_idents(pat, &mut names);
                        out.extend(names.into_iter().map(HoistOp::Var));
                    }
                }
            }
            collect_hoist_stmt(body, out);
        }
        Stmt::ForInOf {
            decl: Some(DeclKind::Var),
            left,
            body,
            ..
        } => {
            let mut names = Vec::new();
            pattern_idents(left, &mut names);
            out.extend(names.into_iter().map(HoistOp::Var));
            collect_hoist_stmt(body, out);
        }
        Stmt::ForInOf { body, .. } => collect_hoist_stmt(body, out),
        // A switch is a single block scope, but `var`s in its case bodies still hoist to the
        // enclosing function/script (only lexical bindings are switch-block-scoped).
        Stmt::Switch { cases, .. } => {
            for case in cases {
                for s in &case.body {
                    collect_hoist_stmt(s, out);
                }
            }
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            for s in block {
                collect_hoist_stmt(s, out);
            }
            if let Some((_, h)) = handler {
                for s in h {
                    collect_hoist_stmt(s, out);
                }
            }
            if let Some(f) = finalizer {
                for s in f {
                    collect_hoist_stmt(s, out);
                }
            }
        }
        _ => {}
    }
}

/// Annex B.3.3 collection (sloppy mode): a *plain* function declaration in a block/if/label
/// substatement position promotes to a var binding unless a lexical name in between blocks it.
fn collect_annexb_funcs(
    stmt: &Stmt,
    in_block: bool,
    blocked: &mut Vec<String>,
    out: &mut Vec<HoistOp>,
) {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| {
            collect_annexb_funcs(stmt, in_block, blocked, out)
        });
    }
    match stmt {
        Stmt::FuncDecl(func) if in_block => {
            // Annex B.3.3 promotes only *plain* function declarations; a block-scoped
            // generator or async function stays lexical.
            if func.is_generator || func.is_async {
                return;
            }
            if let Some(name) = &func.name {
                // Skip entirely if an intervening lexical declaration with the same name
                // would make the equivalent `var` an early error.
                if blocked.iter().any(|b| b == name) {
                    return;
                }
                out.push(HoistOp::AnnexB(name.clone(), func.clone()));
            }
        }
        Stmt::Block(body) => {
            let added = block_lexical_names(body);
            let pushed = push_blocked(blocked, added);
            // The block's DIRECT function declarations hoist first (their own/sibling names
            // don't block them)...
            for s in body {
                if matches!(s, Stmt::FuncDecl(_)) {
                    collect_annexb_funcs(s, true, blocked, out);
                }
            }
            // ...then nested statements see those names as lexical (a same-named function in
            // an inner block would clash when var-hoisted through this one).
            let fn_names: Vec<String> = body
                .iter()
                .filter_map(|s| match s {
                    Stmt::FuncDecl(f) => f.name.clone(),
                    _ => None,
                })
                .collect();
            let pushed_fns = push_blocked(blocked, fn_names);
            for s in body {
                if !matches!(s, Stmt::FuncDecl(_)) {
                    collect_annexb_funcs(s, true, blocked, out);
                }
            }
            blocked.truncate(blocked.len() - pushed_fns);
            blocked.truncate(blocked.len() - pushed);
        }
        Stmt::If { cons, alt, .. } => {
            collect_annexb_funcs(cons, true, blocked, out);
            if let Some(a) = alt {
                collect_annexb_funcs(a, true, blocked, out);
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
            collect_annexb_funcs(body, true, blocked, out)
        }
        // A `for` head's lexical bindings shadow the body: an equivalent `var` inside would
        // be an early error, so same-named block functions there are skipped.
        Stmt::For { init, body, .. } => {
            let mut added = Vec::new();
            if let Some(init) = init {
                if let ForInit::VarDecl {
                    kind: DeclKind::Let | DeclKind::Const,
                    decls,
                } = init.as_ref()
                {
                    for (pat, _) in decls {
                        pattern_idents(pat, &mut added);
                    }
                }
            }
            let pushed = push_blocked(blocked, added);
            collect_annexb_funcs(body, true, blocked, out);
            blocked.truncate(blocked.len() - pushed);
        }
        Stmt::ForInOf {
            decl, left, body, ..
        } => {
            let mut added = Vec::new();
            if matches!(decl, Some(DeclKind::Let | DeclKind::Const)) {
                pattern_idents(left, &mut added);
            }
            let pushed = push_blocked(blocked, added);
            collect_annexb_funcs(body, true, blocked, out);
            blocked.truncate(blocked.len() - pushed);
        }
        Stmt::Labeled { body, .. } | Stmt::With { body, .. } => {
            collect_annexb_funcs(body, true, blocked, out)
        }
        Stmt::Switch { cases, .. } => {
            // The switch block's lexicals are shared across all cases.
            let mut added = Vec::new();
            for c in cases {
                added.extend(block_lexical_names(&c.body));
            }
            let pushed = push_blocked(blocked, added);
            for c in cases {
                for s in &c.body {
                    collect_annexb_funcs(s, true, blocked, out);
                }
            }
            blocked.truncate(blocked.len() - pushed);
        }
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            {
                let added = block_lexical_names(block);
                let pushed = push_blocked(blocked, added);
                for s in block {
                    collect_annexb_funcs(s, true, blocked, out);
                }
                blocked.truncate(blocked.len() - pushed);
            }
            if let Some((param, h)) = handler {
                // A destructuring catch parameter blocks same-named function hoisting out of
                // the handler; a simple `catch (f)` does not (the B.3.5 legacy var exemption).
                let mut added = block_lexical_names(h);
                if let Some(pat) = param {
                    if !matches!(pat, Pattern::Ident(_)) {
                        pattern_idents(pat, &mut added);
                    }
                }
                let pushed = push_blocked(blocked, added);
                for s in h {
                    collect_annexb_funcs(s, true, blocked, out);
                }
                blocked.truncate(blocked.len() - pushed);
            }
            if let Some(f) = finalizer {
                let added = block_lexical_names(f);
                let pushed = push_blocked(blocked, added);
                for s in f {
                    collect_annexb_funcs(s, true, blocked, out);
                }
                blocked.truncate(blocked.len() - pushed);
            }
        }
        _ => {}
    }
}

pub(crate) fn type_name(v: &Value) -> &'static str {
    match &(v).view() {
        ValueView::Undefined | ValueView::Empty => "undefined",
        ValueView::Null => "null",
        ValueView::Bool(_) => "boolean",
        ValueView::Num(_) => "number",
        ValueView::BigInt(_) => "bigint",
        ValueView::Str(_) => "string",
        ValueView::Sym(_) => "symbol",
        ValueView::Obj(_) => "object",
    }
}

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}
