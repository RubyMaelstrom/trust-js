//! Bounded, scalar baseline CFG-region entry provenance.
//!
//! Entered means before effects, and native fusions may conservatively enter a superset.
//! Unseen is limited to this exact immutable bytecode version's collection window. Unknown
//! routes never manufacture a negative observation. This is workload feedback, not a language
//! dead-code proof (ECMA-262 e28783d5: GetValue, Execution Contexts and TryStatement Evaluation).

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::OnceLock;

use crate::bytecode::Chunk;

const MAX_OPS: usize = 65_536;
const MAX_BLOCKS: usize = 8_192;
const MAX_VERSIONS: usize = 1_024;
const MAX_BYTES: usize = 16 * 1024 * 1024;
const RC_HEADER: usize = 2 * std::mem::size_of::<usize>();
static ENABLED: OnceLock<bool> = OnceLock::new();

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
    static VERSIONS: Cell<usize> = const { Cell::new(0) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
    static DENIALS: Cell<u64> = const { Cell::new(0) };
    static DIRECTORY: RefCell<Option<Directory>> = const { RefCell::new(None) };
    #[cfg(test)]
    static TEST_ENABLED: Cell<Option<bool>> = const { Cell::new(None) };
    #[cfg(test)]
    static TEST_VM_MARKS: RefCell<std::collections::HashMap<u64, u64>> = RefCell::new(std::collections::HashMap::new());
}

#[cfg(test)]
pub(crate) fn record_vm_marker(source: &Coverage) {
    TEST_VM_MARKS.with(|marks| {
        let mut marks = marks.borrow_mut();
        let count = marks.entry(source.id()).or_default();
        *count = count.saturating_add(1);
    });
}

#[cfg(test)]
pub(crate) fn test_vm_marks(chunk: &Chunk) -> u64 {
    TEST_VM_MARKS.with(|marks| {
        chunk
            .execution_coverage
            .source_version()
            .and_then(|id| marks.borrow().get(&id).copied())
            .unwrap_or(0)
    })
}

fn denial(reason: Reason) -> Reason {
    DENIALS.with(|count| count.set(count.get().saturating_add(1)));
    reason
}

#[inline]
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_ENABLED.with(Cell::get) {
        return enabled;
    }
    *ENABLED.get_or_init(|| std::env::var("TRUST_JS_OPT_JIT_COVERAGE").as_deref() == Ok("1"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub(crate) enum Reason {
    Disabled = 1,
    Layout = 1 << 1,
    Quota = 1 << 2,
    VersionLimit = 1 << 3,
    IdOverflow = 1 << 4,
    InvalidMapping = 1 << 5,
    AstBody = 1 << 6,
    GetterShortcut = 1 << 7,
    ConstructorShortcut = 1 << 8,
    ForwarderShortcut = 1 << 9,
    OptimizedNative = 1 << 10,
    BindingGetterShortcut = 1 << 11,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum State {
    Entered,
    UnseenInWindow,
    Unknown,
}

/// Source versions own mutable scalar feedback; optimizing clones own frozen scalar feedback.
/// Neither variant keeps an AST, JavaScript object, environment or executable allocation alive.
pub(crate) enum Handle {
    Unavailable(Reason),
    Source(Rc<Coverage>),
    Frozen(Rc<Snapshot>),
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Self::Source(source) = self {
            source.chunk_handles.set(source.chunk_handles.get() - 1);
        }
    }
}

/// Native/prepared bodies pin scalar addresses only; no JavaScript or code owner is added.
pub(crate) struct Lease(Rc<Coverage>);

impl Lease {
    pub(crate) fn new(source: &Rc<Coverage>) -> Self {
        source.native_leases.set(source.native_leases.get() + 1);
        Self(source.clone())
    }

    pub(crate) fn allocation_metadata(&self) -> (usize, usize) {
        (
            Rc::as_ptr(&self.0) as usize,
            self.0.payload.bytes + std::mem::size_of::<Coverage>(),
        )
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.0.native_leases.set(self.0.native_leases.get() - 1);
    }
}

impl Default for Handle {
    fn default() -> Self {
        Self::Unavailable(Reason::Disabled)
    }
}

impl Handle {
    pub(crate) fn new(chunk: &Chunk) -> Self {
        if !enabled() {
            return Self::default();
        }
        if chunk.jit_ops().is_empty() || chunk.jit_ops().len() > MAX_OPS {
            return Self::Unavailable(denial(Reason::Layout));
        }
        let Ok(starts) = crate::jit_ir::semantic_block_starts(chunk.jit_ops()) else {
            return Self::Unavailable(denial(Reason::Layout));
        };
        match Coverage::new(chunk, &starts) {
            Ok(coverage) => Self::Source(coverage),
            Err(reason) => Self::Unavailable(denial(reason)),
        }
    }

    pub(crate) fn source(&self) -> Option<&Rc<Coverage>> {
        match self {
            Self::Source(source) => Some(source),
            Self::Unavailable(_) | Self::Frozen(_) => None,
        }
    }

    pub(crate) fn freeze(&self) -> Self {
        match self {
            Self::Source(source) => match source.freeze() {
                Ok(snapshot) => Self::Frozen(snapshot),
                Err(reason) => Self::Unavailable(reason),
            },
            Self::Frozen(snapshot) => Self::Frozen(snapshot.clone()),
            Self::Unavailable(reason) => Self::Unavailable(*reason),
        }
    }

    pub(crate) fn invalidate(&self, reason: Reason) {
        if let Some(source) = self.source() {
            source.invalidate(reason);
        }
    }

    #[allow(dead_code)] // Typed compiler consumer; separately integrated native region selection.
    pub(crate) fn source_version(&self) -> Option<u64> {
        match self {
            Self::Source(source) => Some(source.id),
            Self::Frozen(snapshot) => Some(snapshot.source_version),
            Self::Unavailable(_) => None,
        }
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) {
        let allocation = match self {
            Self::Source(source) => Some((
                Rc::as_ptr(source) as usize,
                source.payload.bytes + std::mem::size_of::<Coverage>(),
            )),
            Self::Frozen(snapshot) => Some((
                Rc::as_ptr(snapshot) as usize,
                snapshot.reservation.bytes - RC_HEADER,
            )),
            Self::Unavailable(_) => None,
        };
        if let Some((identity, bytes)) = allocation {
            visitor.execution_coverage_allocation(identity, bytes);
        }
    }

    #[allow(dead_code)] // Typed compiler consumer; separately integrated native region selection.
    pub(crate) fn frozen_snapshot(&self) -> Option<&Snapshot> {
        match self {
            Self::Frozen(snapshot) => Some(snapshot),
            _ => None,
        }
    }
}

/// The reservation covers Rust-requested container capacities and owned payload/header bytes.
/// Allocator bookkeeping/rounding is outside this metric. Native code never accesses it.
struct Reservation {
    bytes: usize,
    version: bool,
}

impl Reservation {
    fn new(bytes: usize, version: bool) -> Result<Self, Reason> {
        let fits = BYTES.with(|current| {
            let Some(next) = current.get().checked_add(bytes).filter(|&n| n <= MAX_BYTES) else {
                return false;
            };
            current.set(next);
            true
        });
        if !fits {
            return Err(Reason::Quota);
        }
        if version {
            let fits = VERSIONS.with(|current| {
                if current.get() == MAX_VERSIONS {
                    false
                } else {
                    current.set(current.get() + 1);
                    true
                }
            });
            if !fits {
                BYTES.with(|current| current.set(current.get() - bytes));
                return Err(Reason::VersionLimit);
            }
        }
        Ok(Self { bytes, version })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let _ = BYTES.try_with(|current| current.set(current.get() - self.bytes));
        if self.version {
            let _ = VERSIONS.try_with(|current| current.set(current.get() - 1));
        }
    }
}

struct DirectoryEntry {
    source: Weak<Coverage>,
    // A Weak retains the Rc allocation header after its last strong owner releases payload.
    // Keep that storage charged until this directory entry releases the final weak handle.
    _header: Reservation,
}

struct Directory {
    entries: Vec<DirectoryEntry>,
    _reservation: Reservation,
}

fn with_directory<T>(f: impl FnOnce(&mut Directory) -> Result<T, Reason>) -> Result<T, Reason> {
    DIRECTORY.with(|directory| {
        let mut directory = directory.borrow_mut();
        if directory.is_none() {
            let estimate = std::mem::size_of::<Directory>()
                + MAX_VERSIONS * std::mem::size_of::<DirectoryEntry>();
            let reservation = Reservation::new(estimate, false)?;
            let mut entries = Vec::new();
            entries
                .try_reserve_exact(MAX_VERSIONS)
                .map_err(|_| Reason::Quota)?;
            let actual = std::mem::size_of::<Directory>()
                + entries.capacity() * std::mem::size_of::<DirectoryEntry>();
            // Never retain a container larger than the charged reservation.
            if actual != estimate {
                drop(reservation);
                let reservation = Reservation::new(actual, false)?;
                *directory = Some(Directory {
                    entries,
                    _reservation: reservation,
                });
            } else {
                *directory = Some(Directory {
                    entries,
                    _reservation: reservation,
                });
            }
        }
        let directory = directory.as_mut().unwrap();
        directory
            .entries
            .retain(|entry| entry.source.strong_count() != 0);
        f(directory)
    })
}

pub(crate) struct Coverage {
    id: u64,
    ops: usize,
    slots: usize,
    origin: (u32, u32),
    pc_blocks: Box<[u16]>,
    block_starts: Box<[u32]>,
    entered: Box<[Cell<u8>]>,
    unknown: Cell<u32>,
    chunk_handles: Cell<usize>,
    native_leases: Cell<usize>,
    payload: Reservation,
}

impl Coverage {
    fn new(chunk: &Chunk, boundaries: &[usize]) -> Result<Rc<Self>, Reason> {
        let ops = chunk.jit_ops().len();
        let blocks = boundaries.len().saturating_sub(1);
        if ops == 0 || ops > MAX_OPS || blocks == 0 || blocks > MAX_BLOCKS {
            return Err(Reason::Layout);
        }
        with_directory(|directory| {
            if directory.entries.len() == MAX_VERSIONS {
                return Err(Reason::VersionLimit);
            }
            let bytes = ops * std::mem::size_of::<u16>()
                + blocks * std::mem::size_of::<u32>()
                + blocks.div_ceil(8) * std::mem::size_of::<Cell<u8>>();
            let payload = Reservation::new(bytes, true)?;
            let header = Reservation::new(std::mem::size_of::<Self>() + RC_HEADER, false)?;
            let id = NEXT_ID.with(|next| {
                let id = next.get();
                let after = id.checked_add(1).ok_or(Reason::IdOverflow)?;
                next.set(after);
                Ok(id)
            })?;
            let mut pc_blocks = vec![u16::MAX; ops];
            let mut starts = Vec::with_capacity(blocks);
            for (id, pair) in boundaries.windows(2).enumerate() {
                let (start, end) = (pair[0], pair[1]);
                if start >= end || end > ops {
                    return Err(Reason::InvalidMapping);
                }
                starts.push(start as u32);
                pc_blocks[start..end].fill(id as u16);
            }
            if starts.first() != Some(&0)
                || starts.windows(2).any(|pair| pair[0] >= pair[1])
                || pc_blocks.contains(&u16::MAX)
            {
                return Err(Reason::InvalidMapping);
            }
            let source = Rc::new(Self {
                id,
                ops,
                slots: chunk.jit_frame().1,
                origin: chunk.jit_debug_origin(),
                pc_blocks: pc_blocks.into_boxed_slice(),
                block_starts: starts.into_boxed_slice(),
                entered: (0..blocks.div_ceil(8)).map(|_| Cell::new(0)).collect(),
                unknown: Cell::new(0),
                chunk_handles: Cell::new(1),
                native_leases: Cell::new(0),
                payload,
            });
            directory.entries.push(DirectoryEntry {
                source: Rc::downgrade(&source),
                _header: header,
            });
            Ok(source)
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn unknown_reasons(&self) -> u32 {
        self.unknown.get()
    }

    pub(crate) fn invalidate(&self, reason: Reason) {
        self.unknown.set(self.unknown.get() | reason as u32);
    }

    pub(crate) fn enter_pc(&self, pc: usize) {
        let Some(&block) = self.pc_blocks.get(pc) else {
            self.invalidate(Reason::InvalidMapping);
            return;
        };
        let byte = &self.entered[block as usize / 8];
        byte.set(byte.get() | (1 << (block as usize % 8)));
    }

    pub(crate) fn is_leader(&self, pc: usize) -> bool {
        self.pc_blocks
            .get(pc)
            .is_some_and(|&block| self.block_starts[block as usize] as usize == pc)
    }

    /// Compiler-only immutable mapping. A failed range returns the sticky-Unknown marker
    /// rather than silently excluding possibly executed operations.
    pub(crate) fn markers(&self, ranges: &[(usize, usize)]) -> Result<Vec<(*mut u8, u8)>, Reason> {
        let mut masks = vec![0u8; self.entered.len()];
        for &(start, end) in ranges {
            if start >= end || end > self.ops {
                return Err(Reason::InvalidMapping);
            }
            for &block in &self.pc_blocks[start..end] {
                masks[block as usize / 8] |= 1 << (block as usize % 8);
            }
        }
        Ok(masks
            .into_iter()
            .enumerate()
            .filter_map(|(byte, mask)| (mask != 0).then_some((self.entered[byte].as_ptr(), mask)))
            .collect())
    }

    pub(crate) fn unknown_ptr(&self) -> *mut u32 {
        self.unknown.as_ptr()
    }

    fn freeze(&self) -> Result<Rc<Snapshot>, Reason> {
        let bytes = std::mem::size_of::<Snapshot>()
            + RC_HEADER
            + std::mem::size_of_val(self.block_starts.as_ref())
            + self.entered.len();
        let reservation = Reservation::new(bytes, false).map_err(denial)?;
        Ok(Rc::new(Snapshot {
            source_version: self.id(),
            ops: self.ops,
            slots: self.slots,
            origin: self.origin,
            block_starts: self.block_starts.clone(),
            entered: self.entered.iter().map(Cell::get).collect(),
            unknown: self.unknown_reasons(),
            reservation,
        }))
    }
}

pub(crate) struct Snapshot {
    pub(crate) source_version: u64,
    ops: usize,
    slots: usize,
    origin: (u32, u32),
    block_starts: Box<[u32]>,
    entered: Box<[u8]>,
    unknown: u32,
    reservation: Reservation,
}

impl Snapshot {
    pub(crate) fn state(&self, source_version: u64, pc: usize) -> State {
        if source_version != self.source_version || pc >= self.ops {
            return State::Unknown;
        }
        let block = self
            .block_starts
            .partition_point(|&start| start as usize <= pc)
            - 1;
        if self.entered[block / 8] & (1 << (block % 8)) != 0 {
            State::Entered
        } else if self.unknown != 0 || self.entered.iter().all(|&byte| byte == 0) {
            State::Unknown
        } else {
            State::UnseenInWindow
        }
    }

    /// Validate the canonical Chunk/snapshot relation before consuming negative feedback.
    /// Source-version identity names the entire immutable graph, not merely a source span.
    #[allow(dead_code)] // Typed compiler consumer; separately integrated native region selection.
    pub(crate) fn state_for(&self, chunk: &Chunk, pc: usize) -> State {
        if chunk.execution_coverage.source_version() != Some(self.source_version)
            || chunk.jit_ops().len() != self.ops
            || chunk.jit_frame().1 != self.slots
            || chunk.jit_debug_origin() != self.origin
        {
            return State::Unknown;
        }
        self.state(self.source_version, pc)
    }

    fn json(&self) -> String {
        use std::fmt::Write;
        let mut result = format!("{{\"source_version\":{},\"ops\":{},\"slots\":{},\"origin\":[{},{}],\"unknown_bits\":{},\"block_starts\":[",
            self.source_version, self.ops, self.slots, self.origin.0, self.origin.1, self.unknown);
        for (index, start) in self.block_starts.iter().enumerate() {
            if index != 0 {
                result.push(',');
            }
            write!(result, "{start}").unwrap();
        }
        let mut entered = 0usize;
        let mut unseen = 0usize;
        let mut unknown = 0usize;
        for &pc in &self.block_starts {
            match self.state(self.source_version, pc as usize) {
                State::Entered => entered += 1,
                State::UnseenInWindow => unseen += 1,
                State::Unknown => unknown += 1,
            }
        }
        write!(result, "],\"entered_blocks\":{entered},\"unseen_blocks\":{unseen},\"unknown_blocks\":{unknown},\"entered_hex\":\"").unwrap();
        for &byte in &self.entered {
            write!(result, "{byte:02x}").unwrap();
        }
        result.push_str("\"}");
        result
    }
}

/// Explicit cold diagnostic consumer. Freezes no new optimizer facts and observes no JS state.
/// Temporary returned strings are caller-owned and are not retained profile storage.
pub(crate) fn snapshot_json() -> Option<String> {
    if !enabled() {
        return None;
    }
    let result = with_directory(|directory| {
        let mut records = Vec::new();
        let mut omitted = 0usize;
        for entry in &directory.entries {
            if let Some(source) = entry.source.upgrade() {
                match source.freeze() {
                    Ok(snapshot) => {
                        let mut record = snapshot.json();
                        record.pop();
                        records.push(format!("{record},\"live_chunk_handles\":{},\"native_or_prepared_scalar_leases\":{}}}",
                            source.chunk_handles.get(), source.native_leases.get()));
                    }
                    Err(_) => omitted += 1,
                }
            }
        }
        Ok(format!("{{\"schema_version\":1,\"scope\":\"owning-thread-exact-baseline-version-collection-window\",\"positive\":\"entered-before-effects-conservative-CFG-region\",\"separate_optimizer_clone_execution_included\":false,\"allocator_bookkeeping_included\":false,\"refcount_header_model_bytes\":{RC_HEADER},\"retained_requested_bytes\":{},\"byte_limit\":{MAX_BYTES},\"live_versions\":{},\"version_limit\":{MAX_VERSIONS},\"denials\":{},\"omitted_snapshots\":{omitted},\"records\":[{}]}}",
            BYTES.with(Cell::get), VERSIONS.with(Cell::get), DENIALS.with(Cell::get), records.join(",")))
    });
    Some(result.unwrap_or_else(|reason| {
        format!(
            "{{\"schema_version\":1,\"unknown_reason\":{}}}",
            reason as u32
        )
    }))
}

#[cfg(test)]
pub(crate) struct TestControl(Option<bool>);

#[cfg(test)]
impl TestControl {
    pub(crate) fn new(enabled: bool) -> Self {
        Self(TEST_ENABLED.with(|current| current.replace(Some(enabled))))
    }
}

#[cfg(test)]
impl Drop for TestControl {
    fn drop(&mut self) {
        TEST_ENABLED.with(|current| current.set(self.0));
    }
}

/// Borrowed native slices may resume at a nonleader. The emitter calls this before loading
/// the destination/flags, with established callee-saved canonical frame registers. This
/// scalar leaf allocates nothing and never observes or enters JavaScript.
pub(crate) unsafe extern "C" fn mark_native_resume(ctx: *mut crate::jit::JitCtx) {
    let ctx = unsafe { &*ctx };
    let chunk = unsafe { &*ctx.chunk };
    if let Some(source) = chunk.execution_coverage.source() {
        source.enter_pc(ctx.resume_pc);
    }
}

/// AST evaluation has no exact transformed-PC relation. Invalidate every retained compiled
/// version of this source Function; reading cached slots never materializes encoded bodies.
#[inline]
pub(crate) fn invalidate_function(function: &crate::ast::Function, reason: Reason) {
    if !enabled() {
        return;
    }
    for slot in [&function.code, &function.code2, &function.optimizing_code] {
        if let Some(Some(chunk)) = slot.get() {
            chunk.execution_coverage.invalidate(reason);
        }
    }
}

#[cfg(test)]
#[path = "execution_coverage_tests.rs"]
mod tests;
