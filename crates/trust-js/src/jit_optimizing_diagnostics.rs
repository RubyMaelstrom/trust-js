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
pub(super) const COUNTERS: usize = 16;
pub(super) const SHARED_CALL: usize = 10;
pub(super) const RETAIN: usize = 11;
pub(super) const RELEASE: usize = 12;
pub(super) const PUBLICATION: usize = 13;
pub(super) const HEAP_GUARD: usize = 14;
pub(super) const SPECIALIZATION_BAILOUT: usize = 15;
const LABELS: [&str; COUNTERS] = [
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
];

static CONFIG: OnceLock<Option<bool>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static LIVE: RefCell<Vec<Weak<Code>>> = const { RefCell::new(Vec::new()) };
    static DROPPED: Cell<u64> = const { Cell::new(0) };
}

pub(super) fn enabled() -> bool {
    configuration().is_some()
}

fn configuration() -> Option<bool> {
    *CONFIG.get_or_init(
        || match std::env::var("TRUST_JS_OPT_JIT_DIAGNOSTICS").as_deref() {
            Ok("1") => Some(true),
            Ok("compile") => Some(false),
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
pub(super) struct Code {
    id: u64,
    chunk: usize,
    created: Instant,
    ops: usize,
    pub runtime_counters: bool,
    pub counters: [Cell<u64>; COUNTERS],
    pub emitted_helpers: Cell<u64>,
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
    pub(super) fn new(chunk: &Chunk) -> Option<Rc<Self>> {
        let runtime_counters = configuration()?;
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
                runtime_counters,
                counters: std::array::from_fn(|_| Cell::new(0)),
                emitted_helpers: Cell::new(0),
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

    pub(super) fn counter(&self, index: usize) -> *const Cell<u64> {
        &self.counters[index]
    }

    pub(super) fn json(&self) -> String {
        use std::fmt::Write;
        let mut result = format!(
            "{{\"id\":{},\"chunk\":{},\"ops\":{},\"age_ns\":{},\"analysis_ns\":{},\"lowering_ns\":{},\"codegen_ns\":{},\"finish_ns\":{},\"ir_instructions\":{},\"ir_blocks\":{},\"native_bytes\":{},\"emitted_helpers\":{},\"emitted_publication_words\":{},\"emitted_heap_guards\":{},\"counters\":{{",
            self.id, self.chunk, self.ops, nanos(self.created), self.analysis_ns.get(),
            self.lowering_ns.get(), self.codegen_ns.get(), self.finish_ns.get(),
            self.ir_instructions.get(), self.ir_blocks.get(), self.native_bytes.get(),
            self.emitted_helpers.get(), self.emitted_publications.get(), self.emitted_guards.get(),
        );
        for (index, label) in LABELS.iter().enumerate() {
            if index != 0 {
                result.push(',');
            }
            let _ = write!(result, "\"{label}\":{}", self.counters[index].get());
        }
        let _ = write!(
            result,
            "}},\"runtime_counters\":{}}}",
            self.runtime_counters
        );
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
            let state = Rc::new(RefCell::new(State {
                costs: [Cost::default(); NUM_PASSES],
                children: Vec::new(),
            }));
            let previous = timing::set_thread_profiler(Box::new(Recorder(state.clone())));
            Self {
                previous: Some(previous),
                state,
                id: code.id,
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
