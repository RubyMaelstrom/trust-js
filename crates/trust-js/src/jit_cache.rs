//! Bounded, thread-owned native-code residency. ECMAScript execution contexts
//! (ECMA-262 §9.4) and resumptions remain canonical VM state, not code-cache state.
//!
//! A resident slot owns code; Rust dispatch owns a temporary Rc lease; generated
//! entry/exit marks protect direct-call frames, including suspended return PCs.
//! Reclamation never changes a live activation or calls into author code.

use super::JitCode;
use std::cell::{Cell, OnceCell, RefCell};
use std::rc::{Rc, Weak};

#[path = "jit_cache_directory.rs"]
mod directory;
use directory::{Directory, Page};

// Fixed page/directory payload plus at most this many Weak-held SlotState
// allocations. shared_metadata_json reports the exact requested payload pieces;
// private Rc headers and allocator rounding remain excluded. No Realm is pinned.
const MAX_RESIDENT_ENTRIES: usize = 131_072;
const MAX_CLOCK_VISITS: usize = 4096;
const COLD_MAINTENANCE_STEPS: usize = 8;
const NO_REGISTRATION: u32 = u32::MAX;

/// Stable allocation embedded by address in each native prologue/epilogue.
/// Native recursion is already bounded by the engine's checked stack limit.
#[repr(C)]
#[derive(Default)]
pub(super) struct CodeResidency {
    pub active: Cell<u64>,
    pub referenced: Cell<u8>,
}

#[derive(Default)]
enum CodeState {
    #[default]
    Cold,
    Pending,
    Unavailable,
    Resident(Rc<JitCode>),
    #[cfg(feature = "optimizing-jit")]
    Retiring(Rc<JitCode>),
}

struct SlotState {
    code: RefCell<CodeState>,
    registration: Cell<u32>,
    evictions: Cell<u8>,
    cooldown: Cell<u16>,
}

impl Default for SlotState {
    fn default() -> Self {
        Self {
            code: RefCell::new(CodeState::Cold),
            registration: Cell::new(NO_REGISTRATION),
            evictions: Cell::new(0),
            cooldown: Cell::new(0),
        }
    }
}

/// Lazy stable sidecar: cold bytecode retains only one empty OnceCell pointer.
#[derive(Default)]
pub(crate) struct NativeCodeSlot {
    state: OnceCell<Rc<SlotState>>,
}

impl NativeCodeSlot {
    pub(crate) fn retained_metadata_bytes(&self) -> usize {
        self.state
            .get()
            .map_or(0, |_| std::mem::size_of::<SlotState>())
    }

    pub(crate) const fn new() -> Self {
        Self {
            state: OnceCell::new(),
        }
    }

    /// Outer None is retryable (cold, evicted, or capacity-deferred); inner None
    /// is structural unavailability. Every successful read owns a code lease.
    pub(crate) fn get(&self) -> Option<Option<Rc<JitCode>>> {
        let state = self.state.get()?;
        #[cfg(feature = "optimizing-jit")]
        let mut retiring = false;
        let result = match &*state.code.borrow() {
            CodeState::Cold | CodeState::Pending => None,
            CodeState::Unavailable => Some(None),
            CodeState::Resident(code) => Some(Some(code.clone())),
            #[cfg(feature = "optimizing-jit")]
            CodeState::Retiring(_) => {
                retiring = true;
                None
            }
        };
        #[cfg(feature = "optimizing-jit")]
        if retiring {
            self.finish_retirement(state);
        }
        result
    }

    /// Stop publishing this version immediately, but retain every active return PC and
    /// outstanding Rust lease. A later dispatch reclaims it and may compile the template.
    /// No successful native entry gains a retirement poll or counter.
    #[cfg(feature = "optimizing-jit")]
    pub(crate) fn request_retirement(&self) {
        let Some(state) = self.state.get() else {
            return;
        };
        let mut contents = state.code.borrow_mut();
        if !matches!(*contents, CodeState::Resident(_)) {
            return;
        }
        let CodeState::Resident(code) = std::mem::take(&mut *contents) else {
            unreachable!()
        };
        *contents = CodeState::Retiring(code);
        drop(contents);
        // Cached raw entrypoints must miss BEFORE another call can acquire this version.
        crate::bytecode::invalidate_call_caches();
    }

    #[cfg(feature = "optimizing-jit")]
    #[cold]
    fn finish_retirement(&self, state: &Rc<SlotState>) {
        let mut contents = state.code.borrow_mut();
        if !matches!(&*contents, CodeState::Retiring(code)
            if Rc::strong_count(code) == 1 && code.residency.active.get() == 0)
        {
            return;
        }
        let CodeState::Retiring(code) = std::mem::take(&mut *contents) else {
            unreachable!()
        };
        drop(contents);
        let mut retired = Retirement::default();
        retired.push_code(code);
        REGISTRY.with(|registry| {
            let mut registry = registry.borrow_mut();
            let index = state.registration.replace(NO_REGISTRATION);
            assert_ne!(
                index, NO_REGISTRATION,
                "resident version owns its directory cell"
            );
            registry.remove(index as usize, &mut retired);
        });
        retired.finish();
    }

    /// No compile was ever attempted for this slot (nothing resident, pending or reclaimed).
    pub(crate) fn untried(&self) -> bool {
        self.state.get().is_none()
    }

    /// Bounded re-use evidence avoids immediate re-emission after each eviction.
    /// This is separate from initial function/OSR hotness and does not change it.
    pub(crate) fn ready_to_compile(&self) -> bool {
        let Some(state) = self.state.get() else {
            return true;
        };
        if !matches!(*state.code.borrow(), CodeState::Cold) {
            return false;
        }
        let remaining = state.cooldown.get();
        state.cooldown.set(remaining.saturating_sub(1));
        remaining == 0
    }

    /// Cold-only admission, BEFORE emission/allocation of an executable mapping.
    /// ECMA-262 §9.4/RunSuspendedContext and [[Call]] (local e28783d5): a denied
    /// cache admission leaves the canonical activation and completion untouched.
    /// No registry access is added to a resident call or native entry/exit.
    pub(crate) fn begin_compile(&self) -> Option<CompilePermit> {
        let state = self.state.get_or_init(|| Rc::new(SlotState::default()));
        if !matches!(*state.code.borrow(), CodeState::Cold) {
            return None;
        }
        let mut retired = Retirement::default();
        let admitted = REGISTRY.with(|registry| registry.borrow_mut().admit(state, &mut retired));
        // Arm rollback BEFORE any retired mapping/profiler destructor can
        // unwind or reenter; the slot is already Pending inside the registry.
        let permit = admitted.then(|| CompilePermit {
            state: Some(state.clone()),
        });
        retired.finish();
        if !admitted {
            // Preserve the existing metadata-denial cooldown. Shared retry
            // scheduling additionally bounds work across DISTINCT new slots.
            state.cooldown.set(256);
            return None;
        }
        permit
    }
}

/// Reserves an ACTUAL directory cell, not a racy promise of later capacity.
/// The owning state is Pending until commit/rollback. Compiler allocation may
/// reenter reclamation; Pending entries cannot be evicted or double-admitted.
pub(crate) struct CompilePermit {
    state: Option<Rc<SlotState>>,
}

impl CompilePermit {
    pub(crate) fn commit(mut self, code: Option<Rc<JitCode>>) {
        let state = self.state.take().expect("live compilation permit");
        debug_assert!(matches!(*state.code.borrow(), CodeState::Pending));
        if let Some(code) = code {
            code.residency.referenced.set(1);
            *state.code.borrow_mut() = CodeState::Resident(code);
            REGISTRY.with(|registry| registry.borrow_mut().pending -= 1);
        } else {
            // Structural unavailability is permanent; allocation/pressure
            // failures instead leave the permit uncommitted and retryable.
            *state.code.borrow_mut() = CodeState::Unavailable;
            release_reservation(&state);
        }
    }
}

impl Drop for CompilePermit {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            debug_assert!(matches!(*state.code.borrow(), CodeState::Pending));
            *state.code.borrow_mut() = CodeState::Cold;
            release_reservation(&state);
        }
    }
}

fn release_reservation(state: &Rc<SlotState>) {
    let mut retired = Retirement::default();
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry.pending -= 1;
        let index = state.registration.replace(NO_REGISTRATION);
        assert_ne!(index, NO_REGISTRATION, "permit released exactly once");
        let directory = registry.entries.as_ref().unwrap();
        assert!(std::ptr::eq(
            directory.get(index as usize).as_ptr(),
            Rc::as_ptr(state)
        ));
        registry.remove(index as usize, &mut retired);
    });
    retired.finish();
}

/// Everything potentially deallocating leaves the Registry borrow in this
/// batch. Invalidation precedes mapping/profiler destruction, including when
/// destructors synchronously reenter the engine or cache.
#[derive(Default)]
struct Retirement {
    code: Vec<Rc<JitCode>>,
    #[cfg(feature = "optimizing-jit")]
    call_stub_owners: std::collections::HashMap<usize, usize>,
    states: Vec<Rc<SlotState>>,
    weak: Vec<Weak<SlotState>>,
    // Keep each allocation boxed until after the Registry borrow ends; moving
    // Page out of its Box here would deallocate inside that reentrant boundary.
    #[allow(clippy::vec_box)]
    pages: Vec<Box<Page>>,
    directories: Vec<Box<Directory>>,
}

impl Retirement {
    /// Expected executable bytes released by this batch. A shared stub contributes only
    /// when ALL of its strong owners have entered retirement; counting it for every caller
    /// overstates capacity, while ignoring it can discard additional bodies unnecessarily.
    /// The executable reservation still checks actual free capacity after drops/reentrancy.
    fn push_code(&mut self, code: Rc<JitCode>) -> usize {
        let bytes = code.len;
        #[cfg(feature = "optimizing-jit")]
        let bytes = {
            let mut bytes = bytes;
            for stub in &code.call_stubs {
                let count = self
                    .call_stub_owners
                    .entry(Rc::as_ptr(stub) as usize)
                    .or_default();
                *count += 1;
                if *count == Rc::strong_count(stub) {
                    bytes = bytes.saturating_add(stub.executable_bytes());
                }
            }
            bytes
        };
        self.code.push(code);
        bytes
    }

    fn finish(self) {
        if !self.code.is_empty() {
            crate::bytecode::invalidate_call_caches();
        }
        #[cfg(test)]
        tests::before_retirement_drop();
        drop(self);
    }
}

#[derive(Clone, Copy)]
enum Goal {
    DeadOnly,
    Metadata,
    Bytes(usize),
}

#[derive(Default)]
struct Registry {
    entries: Option<Box<Directory>>,
    cursor: usize,
    pending: usize,
    evictions: u64,
    pressure_miss: Option<(usize, u8)>,
    metadata_retry: u8,
    #[cfg(test)]
    entry_limit: Option<usize>,
    #[cfg(test)]
    steps: usize,
}

impl Registry {
    fn len(&self) -> usize {
        self.entries.as_ref().map_or(0, |entries| entries.len)
    }

    fn limit(&self) -> usize {
        #[cfg(test)]
        if let Some(limit) = self.entry_limit {
            return limit;
        }
        MAX_RESIDENT_ENTRIES
    }

    fn remove(&mut self, index: usize, retired: &mut Retirement) {
        let directory = self.entries.as_mut().unwrap();
        directory.remove(index, retired);
        if directory.len == 0 {
            retired.directories.push(self.entries.take().unwrap());
            self.cursor = 0;
        }
        self.metadata_retry = 0;
    }

    fn admit(&mut self, state: &Rc<SlotState>, retired: &mut Retirement) -> bool {
        debug_assert_eq!(state.registration.get(), NO_REGISTRATION);
        debug_assert!(matches!(*state.code.borrow(), CodeState::Cold));
        self.collect(Goal::DeadOnly, COLD_MAINTENANCE_STEPS, retired);
        if self.len() >= self.limit() {
            if self.metadata_retry != 0 {
                self.metadata_retry -= 1;
                return false;
            }
            self.collect(Goal::Metadata, MAX_CLOCK_VISITS, retired);
        }
        if self.len() >= self.limit() {
            self.metadata_retry = 31;
            return false;
        }
        let index = self.entries.get_or_insert_with(Box::default).insert(state);
        state.registration.set(index as u32);
        *state.code.borrow_mut() = CodeState::Pending;
        self.pending += 1;
        true
    }

    fn collect(&mut self, goal: Goal, max_steps: usize, retired: &mut Retirement) {
        let mut bytes = 0usize;
        for _ in 0..max_steps {
            if self.len() == 0
                || match goal {
                    Goal::Bytes(required) => bytes >= required,
                    Goal::Metadata => self.len() < self.limit(),
                    Goal::DeadOnly => false,
                }
            {
                break;
            }
            #[cfg(test)]
            {
                self.steps += 1;
            }
            let directory = self.entries.as_ref().unwrap();
            let Some(index) = directory.inspect(&mut self.cursor) else {
                continue;
            };
            if matches!(goal, Goal::DeadOnly) {
                // Every live registration is Pending, Resident, or explicitly
                // Retiring. Admission publishes Pending under this borrow;
                // commit(None)/rollback remove synchronously before callbacks;
                // eviction changes/removes under this same borrow. Therefore
                // cheap cold maintenance needs neither an upgraded owner nor
                // a CodeState borrow/retirement allocation for LIVE entries.
                if directory.get(index).strong_count() == 0 {
                    self.remove(index, retired);
                }
                continue;
            }
            let Some(state) = directory.get(index).upgrade() else {
                self.remove(index, retired);
                continue;
            };
            let mut contents = state.code.borrow_mut();
            let remove = match &*contents {
                CodeState::Pending => false,
                CodeState::Cold | CodeState::Unavailable => true,
                #[cfg(feature = "optimizing-jit")]
                CodeState::Retiring(code) => {
                    if Rc::strong_count(code) != 1 || code.residency.active.get() != 0 {
                        false
                    } else {
                        let CodeState::Retiring(code) = std::mem::take(&mut *contents) else {
                            unreachable!()
                        };
                        // An unstable function may never be called again. Pressure must
                        // still reclaim it without waiting for a future dispatch poll.
                        bytes = bytes.saturating_add(retired.push_code(code));
                        true
                    }
                }
                CodeState::Resident(code) => {
                    if Rc::strong_count(code) != 1
                        || code.residency.active.get() != 0
                        || code.residency.referenced.replace(0) != 0
                    {
                        false
                    } else {
                        let CodeState::Resident(code) = std::mem::take(&mut *contents) else {
                            unreachable!()
                        };
                        bytes = bytes.saturating_add(retired.push_code(code));
                        let count = state.evictions.get().saturating_add(1);
                        state.evictions.set(count);
                        state.cooldown.set(8 << count.saturating_sub(1).min(5));
                        self.evictions = self.evictions.saturating_add(1);
                        true
                    }
                }
            };
            drop(contents);
            if remove {
                state.registration.set(NO_REGISTRATION);
                self.remove(index, retired);
            }
            retired.states.push(state);
        }
    }
}

thread_local! {
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry::default());
}

pub(super) fn reclaim(required: usize) {
    if required == 0 {
        return;
    }
    let mut retired = Retirement::default();
    REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .collect(Goal::Bytes(required), MAX_CLOCK_VISITS, &mut retired)
    });
    retired.finish();
}

/// A fully active/foreign-thread working set must not turn every VM fallback
/// into a scan of the code cache. Retry after bounded execution evidence, or
/// immediately when released mapping capacity changes. This is scheduling
/// information only, not an address-validity generation or a safety proof.
pub(super) fn reclaim_for_capacity(bytes: usize, available: usize) {
    let retry = REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        if let Some((previous, remaining)) = &mut registry.pressure_miss {
            if *previous == available && *remaining != 0 {
                *remaining -= 1;
                return false;
            }
        }
        registry.pressure_miss = None;
        true
    });
    if !retry {
        return;
    }
    reclaim(bytes.saturating_sub(available));
    let available = super::executable_code_budget()
        .remaining
        .load(std::sync::atomic::Ordering::Relaxed);
    if available < bytes {
        REGISTRY.with(|registry| registry.borrow_mut().pressure_miss = Some((available, 31)));
    }
}

/// Opt-in diagnostics only. This is shared THREAD bookkeeping, not an Engine's
/// reachable heap: live SlotState values are already counted through their
/// NativeCodeSlot owners. Dead Weak backing is additional requested payload.
/// Rc allocation headers, allocator rounding and TLS runtime metadata are not
/// portable Rust layout and are expressly excluded from this lower bound.
pub(super) fn shared_metadata_json() -> String {
    REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let directory_bytes = registry.entries.as_ref().map_or(0, |entries| entries.allocated_bytes());
        let dead = registry.entries.as_ref().map_or(0, |entries| entries.dead_entries());
        let pages = registry.entries.as_ref().map_or(0, |entries| entries.allocated_pages);
        let slots = registry.len();
        let registry_bytes = std::mem::size_of::<Registry>();
        let dead_payload = dead * std::mem::size_of::<SlotState>();
        let live_payload = (slots - dead) * std::mem::size_of::<SlotState>();
        let directory_bound = std::mem::size_of::<Directory>()
            + directory::PAGE_COUNT * std::mem::size_of::<Page>();
        let dead_bound = MAX_RESIDENT_ENTRIES * std::mem::size_of::<SlotState>();
        let total = registry_bytes + directory_bytes + dead_payload;
        format!(concat!("{{\"scope\":\"current_thread_shared_cache\",",
            "\"quality\":\"lower_bound\",\"included_in_managed_memory\":false,",
            "\"known_requested_bytes\":{},\"registry_payload_bytes\":{},",
            "\"directory_and_page_payload_bytes\":{},\"allocated_pages\":{},",
            "\"registered_entries\":{},\"pending_entries\":{},\"evictions\":{},\"dead_entries\":{},",
            "\"dead_slot_value_payload_bytes\":{},\"entry_limit\":{},",
            "\"live_slot_value_payload_bytes_excluded\":{},",
            "\"maximum_directory_and_page_payload_bytes\":{},\"maximum_dead_slot_value_payload_bytes\":{},",
            "\"excludes\":\"live SlotState values already attributed to owners; Rc headers; allocator rounding; TLS runtime; in-flight retirement batches\"}}"),
            total, registry_bytes, directory_bytes, pages, slots, registry.pending, registry.evictions,
            dead, dead_payload, registry.limit(), live_payload, directory_bound, dead_bound)
    })
}

/// Acquire immediately after validating a raw CallIc, before any reentrant work
/// (including an inline compilation that can reclaim the selected old code).
///
/// # Safety
/// `code` is a live Rc allocation established by a successful identity/realm/
/// non-sentinel epoch check; no reentrant operation occurred after that check.
pub(crate) unsafe fn lease_raw(code: *const JitCode) -> Rc<JitCode> {
    unsafe {
        Rc::increment_strong_count(code);
        Rc::from_raw(code)
    }
}

#[cfg(test)]
pub(super) fn eviction_count() -> u64 {
    REGISTRY.with(|registry| registry.borrow().evictions)
}

#[cfg(test)]
#[path = "jit_cache_admission_tests.rs"]
mod tests;
