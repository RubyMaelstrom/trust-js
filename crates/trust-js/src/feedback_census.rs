//! The feedback census (W4 step 2), compiled only with the `architecture-diagnostics` feature.
//!
//! With `TRUST_JS_FEEDBACK_CENSUS=<path>`, the VM's loop counts the operations it dispatches by
//! bytecode pc, and each chunk's production feedback (the property, call, construct and name
//! caches, the allocation images) is snapshotted when the chunk drops and when the shell exits,
//! which writes everything to `<path>`. A `--tier=bytecode` run puts every compiled-code
//! operation through the VM's loop, so its counts are each site's executions; the production
//! `--tier=jit` run's snapshot is each site's feedback at exit. `scripts/feedback-census.py` joins
//! the two runs' files into feedback-schema.txt's coverage (section 4). Chunks are keyed by a hash
//! of their operations and their function's origin, which the two runs share; an inlined
//! (transformed) chunk is keyed apart, since its pcs are its own.
//!
//! The file: a `#` header line naming the records the engine writes; per chunk key a `C` line
//! (key, inlined, chunks merged into it, chunks with native code, label); per pc with anything to
//! say an `O` line (key, inlined, pc, census kind, VM dispatches, state word in hex, the
//! always-on record's writes, its word in hex, operation). The state word's meaning per kind is
//! `Chunk::feedback_census_snapshot`'s; the record's is feedback-schema.txt's section 5.
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::Write;
use std::rc::Weak;
use std::sync::OnceLock;

static PATH: OnceLock<Option<String>> = OnceLock::new();

/// Whether the census runs in this process (`TRUST_JS_FEEDBACK_CENSUS`, read once), or, in a
/// test, on this thread.
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if TEST_ON.with(Cell::get) {
        return true;
    }
    PATH.get_or_init(|| std::env::var("TRUST_JS_FEEDBACK_CENSUS").ok())
        .is_some()
}

#[cfg(test)]
thread_local! {
    static TEST_ON: Cell<bool> = const { Cell::new(false) };
}

/// Run the census on this thread (tests: one engine a thread, no file).
#[cfg(test)]
pub(crate) fn enable_for_test() {
    TEST_ON.with(|on| on.set(true));
}

/// The census text so far, the live chunks snapshotted into it, and the census reset (tests).
#[cfg(test)]
pub(crate) fn take_for_test() -> String {
    snapshot_live();
    let text = render();
    STATE.with(|state| *state.borrow_mut() = State::default());
    text
}

/// A chunk's census state, which its feedback vector holds while the census runs.
pub(crate) struct ChunkCensus {
    pub(crate) key: u64,
    pub(crate) inlined: bool,
    pub(crate) label: String,
    counts: Box<[Cell<u64>]>,
    /// The always-on record's writes per pc (W4 step 3): the VM's executions, and generated
    /// code's misses outside their in-line sets.
    records: Box<[Cell<u64>]>,
    /// Snapshotted already (at exit or by a test), so its drop adds nothing.
    snapped: Cell<bool>,
}

impl ChunkCensus {
    pub(crate) fn new(key: u64, inlined: bool, label: String, ops: usize) -> Self {
        Self {
            key,
            inlined,
            label,
            counts: (0..ops).map(|_| Cell::new(0)).collect(),
            records: (0..ops).map(|_| Cell::new(0)).collect(),
            snapped: Cell::new(false),
        }
    }

    /// Count one dispatch of the operation at `pc` by the VM's loop.
    #[inline(always)]
    pub(crate) fn count(&self, pc: usize) {
        if let Some(cell) = self.counts.get(pc) {
            cell.set(cell.get() + 1);
        }
    }

    /// Count one write of the always-on record at `pc`.
    #[inline(always)]
    pub(crate) fn count_record(&self, pc: usize) {
        if let Some(cell) = self.records.get(pc) {
            cell.set(cell.get() + 1);
        }
    }
}

/// The census key of a chunk: FNV-1a over its operations' text and its function's origin.
pub(crate) fn chunk_key(ops_text: &str, start: u32, len: u32, url: &str) -> u64 {
    let mut hash = fnv(0xcbf2_9ce4_8422_2325, ops_text.as_bytes());
    hash = fnv(hash, &start.to_le_bytes());
    hash = fnv(hash, &len.to_le_bytes());
    fnv(hash, url.as_bytes())
}

fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// One pc's snapshot: dispatches summed, state words ORed over the chunks of one key.
#[derive(Default, Clone)]
struct OpRecord {
    op: String,
    kind: u8,
    count: u64,
    records: u64,
    state: u32,
    record: u32,
}

#[derive(Default)]
struct ChunkRecord {
    label: String,
    chunks: u32,
    native: u32,
    ops: Vec<OpRecord>,
}

#[derive(Default)]
struct State {
    registry: Vec<Weak<crate::bytecode::Chunk>>,
    chunks: HashMap<(u64, bool), ChunkRecord>,
    dumped: bool,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Register a chunk the census follows, for the snapshot at exit.
pub(crate) fn register(chunk: &std::rc::Rc<crate::bytecode::Chunk>) {
    STATE.with(|state| {
        state
            .borrow_mut()
            .registry
            .push(std::rc::Rc::downgrade(chunk))
    });
}

/// One operation of a chunk's snapshot: its text, census kind, state word and always-on record.
pub(crate) struct OpSnapshot {
    pub(crate) op: String,
    pub(crate) kind: u8,
    pub(crate) state: u32,
    pub(crate) record: u32,
}

/// Merge a chunk's snapshot with its dispatch counts (at its drop, or at exit for a live one),
/// once a chunk.
pub(crate) fn merge(census: &ChunkCensus, ops: Vec<OpSnapshot>, native: bool) {
    if census.snapped.replace(true) {
        return;
    }
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.dumped {
            return;
        }
        let record = state
            .chunks
            .entry((census.key, census.inlined))
            .or_default();
        if record.ops.is_empty() {
            record.label = census.label.clone();
            record.ops = vec![OpRecord::default(); ops.len()];
        }
        record.chunks += 1;
        record.native += u32::from(native);
        for (pc, snapshot) in ops.into_iter().enumerate() {
            let Some(slot) = record.ops.get_mut(pc) else {
                continue;
            };
            if slot.op.is_empty() {
                slot.op = snapshot.op;
                slot.kind = snapshot.kind;
            }
            slot.state |= snapshot.state;
            slot.record |= snapshot.record;
            slot.count += census.counts.get(pc).map_or(0, Cell::get);
            slot.records += census.records.get(pc).map_or(0, Cell::get);
        }
    });
}

/// Snapshot every live chunk the census follows. Each chunk is snapshotted once: at this call,
/// or at its drop if it drops first; the registry then forgets it.
fn snapshot_live() -> Vec<std::rc::Rc<crate::bytecode::Chunk>> {
    let live: Vec<_> = STATE.with(|state| {
        std::mem::take(&mut state.borrow_mut().registry)
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    });
    for chunk in &live {
        chunk.feedback_census_snapshot();
    }
    live
}

/// The shell's exit: snapshot the live chunks and write the census file.
pub(crate) fn dump() {
    let Some(Some(path)) = PATH.get() else {
        return;
    };
    let live = snapshot_live();
    let out = render();
    STATE.with(|state| state.borrow_mut().dumped = true);
    if let Ok(mut file) = std::fs::File::create(path) {
        let _ = file.write_all(out.as_bytes());
    }
    // The engine's teardown drops these chunks after the file is written; `dumped` refuses their
    // snapshots, so nothing is gained by dropping them here.
    std::mem::forget(live);
}

/// The census file's text.
fn render() -> String {
    // `records=` names the records the engine writes, whose readings the script applies.
    let mut out = String::from("# feedback-census 1 records=values,elements,generic\n");
    STATE.with(|state| {
        let state = state.borrow();
        let mut keys: Vec<_> = state.chunks.keys().copied().collect();
        keys.sort_unstable();
        for key in keys {
            let record = &state.chunks[&key];
            out.push_str(&format!(
                "C\t{:016x}\t{}\t{}\t{}\t{}\n",
                key.0,
                u8::from(key.1),
                record.chunks,
                record.native,
                record.label.replace(['\t', '\n'], " ")
            ));
            for (pc, op) in record.ops.iter().enumerate() {
                if op.count == 0 && op.state == 0 && op.record == 0 && op.records == 0 {
                    continue;
                }
                out.push_str(&format!(
                    "O\t{:016x}\t{}\t{}\t{}\t{}\t{:x}\t{}\t{:x}\t{}\n",
                    key.0,
                    u8::from(key.1),
                    pc,
                    op.kind,
                    op.count,
                    op.state,
                    op.records,
                    op.record,
                    op.op.replace(['\t', '\n'], " ")
                ));
            }
        }
    });
    out
}
