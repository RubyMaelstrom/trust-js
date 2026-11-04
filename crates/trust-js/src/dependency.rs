//! The dependency registry (W4 step 10; feedback-schema.txt section 5.7): JSC's watchpoint sets
//! (Source/JavaScriptCore/bytecode/Watchpoint.h, dfg/DFGDesiredWatchpoints.h). A set guards one
//! fact compiled code may assume. It starts clear (the fact holds and nothing relies on it), is
//! watched once a compilation installs itself in it, and is invalidated for good when the fact
//! breaks: firing marks every code installed in it invalid and unlinks the call caches that could
//! enter that code. A compilation collects the sets it relies on ([`DesiredWatchpoints`]) and
//! installs itself in all of them at its end, failing if one has fired meanwhile, as a DFG plan's
//! finalization does.
//!
//! The first producers: a prototype's shape (its set fires at any structural change of the
//! prototype, Props::note_structural's) and the Array elements protector
//! (`Interp::array_prototypes_unshadowed`'s fact, which fires at any structural change of
//! Array.prototype or Object.prototype: a conservative superset of an indexed property's
//! appearance). The speculative tier (Phase 3) is the first real consumer; until then the tests
//! are. PROTO_EPOCH and CALL_IC_EPOCH stay for the baseline's caches.
//!
//! Sets are shared across threads (generator and async bodies run on pooled threads that share
//! the engine, as PROTO_EPOCH's note says), so they are atomic and locked; firing is rare.
// The speculative tier is the registry's consumer; until it exists, the tests read it.
#![cfg_attr(not(test), allow(dead_code))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// A set's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WatchState {
    /// The fact holds; no code relies on it.
    Clear = 0,
    /// The fact holds; some code relies on it.
    Watched = 1,
    /// The fact broke; no code may rely on it again.
    Invalidated = 2,
}

/// One fact's watchpoint set.
pub(crate) struct WatchpointSet {
    state: AtomicU8,
    /// What the fact is, for a code's invalidation reason.
    what: &'static str,
    codes: Mutex<Vec<Weak<CodeDependencies>>>,
}

impl WatchpointSet {
    pub(crate) fn new(what: &'static str) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(WatchState::Clear as u8),
            what,
            codes: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn state(&self) -> WatchState {
        match self.state.load(Ordering::Acquire) {
            0 => WatchState::Clear,
            1 => WatchState::Watched,
            _ => WatchState::Invalidated,
        }
    }

    pub(crate) fn is_invalidated(&self) -> bool {
        self.state() == WatchState::Invalidated
    }

    pub(crate) fn what(&self) -> &'static str {
        self.what
    }

    /// The fact broke: invalidate the set and every code installed in it, and unlink the call
    /// caches that could enter such code (CALL_IC_EPOCH's bump: every call site re-resolves).
    pub(crate) fn fire(&self) {
        if self
            .state
            .swap(WatchState::Invalidated as u8, Ordering::AcqRel)
            == WatchState::Invalidated as u8
        {
            return;
        }
        let codes = std::mem::take(&mut *self.codes.lock().unwrap_or_else(|e| e.into_inner()));
        let mut invalidated = false;
        for code in codes.iter().filter_map(Weak::upgrade) {
            invalidated |= code.invalidate(self.what);
        }
        if invalidated {
            crate::bytecode::invalidate_call_caches();
        }
    }

    /// Install `code`, unless the set has fired.
    fn install(&self, code: &Arc<CodeDependencies>) -> bool {
        let mut codes = self.codes.lock().unwrap_or_else(|e| e.into_inner());
        if self.is_invalidated() {
            return false;
        }
        codes.retain(|weak| weak.strong_count() != 0);
        codes.push(Arc::downgrade(code));
        let _ = self.state.compare_exchange(
            WatchState::Clear as u8,
            WatchState::Watched as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        true
    }
}

/// A compiled code's side of its dependencies: whether it is still valid, why it was not, and
/// the sets it is installed in (which it keeps alive).
pub(crate) struct CodeDependencies {
    valid: AtomicBool,
    reason: Mutex<Option<&'static str>>,
    sets: Mutex<Vec<Arc<WatchpointSet>>>,
}

impl CodeDependencies {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            valid: AtomicBool::new(true),
            reason: Mutex::new(None),
            sets: Mutex::new(Vec::new()),
        })
    }

    /// Whether every fact the code relies on still holds.
    pub(crate) fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }

    /// The fact whose set invalidated the code.
    pub(crate) fn invalidated_by(&self) -> Option<&'static str> {
        *self.reason.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn invalidate(&self, what: &'static str) -> bool {
        if !self.valid.swap(false, Ordering::AcqRel) {
            return false;
        }
        *self.reason.lock().unwrap_or_else(|e| e.into_inner()) = Some(what);
        true
    }
}

/// The sets a compilation relies on, gathered while it compiles.
#[derive(Default)]
pub(crate) struct DesiredWatchpoints {
    sets: Vec<Arc<WatchpointSet>>,
}

impl DesiredWatchpoints {
    pub(crate) fn add(&mut self, set: Arc<WatchpointSet>) {
        if !self.sets.iter().any(|held| Arc::ptr_eq(held, &set)) {
            self.sets.push(set);
        }
    }

    /// At the compilation's end: install `code` in every set, or fail with the fact that broke
    /// (a set fired while it compiled). A failed install leaves `code` invalid.
    pub(crate) fn install(self, code: &Arc<CodeDependencies>) -> Result<(), &'static str> {
        for set in &self.sets {
            if !set.install(code) {
                code.invalidate(set.what);
                return Err(set.what);
            }
        }
        *code.sets.lock().unwrap_or_else(|e| e.into_inner()) = self.sets;
        Ok(())
    }
}

/// The prototypes' sets by their property map's address (a cell does not move), and how many
/// there are, which keeps Props::note_structural's report to one load while none is watched.
struct Prototypes {
    sets: Mutex<HashMap<usize, Vec<Arc<WatchpointSet>>>>,
}

static WATCHED_PROTOTYPES: AtomicUsize = AtomicUsize::new(0);

fn prototypes() -> &'static Prototypes {
    static PROTOTYPES: OnceLock<Prototypes> = OnceLock::new();
    PROTOTYPES.get_or_init(|| Prototypes {
        sets: Mutex::new(HashMap::new()),
    })
}

/// A prototype's shape, the fact [`prototype_shape_set`] guards.
pub(crate) const PROTOTYPE_SHAPE: &str = "a prototype's shape";
/// The Array elements protector, the fact `Interp::array_protector_set` guards.
pub(crate) const ARRAY_PROTECTOR: &str = "the Array elements protector";

/// Have `set` fire at the next structural change of the prototype whose property map is at
/// `props`. The caller marks the map a prototype's, so that the change reports
/// (Props::mark_proto).
pub(crate) fn watch_prototype(props: usize, set: &Arc<WatchpointSet>) {
    let mut sets = prototypes().sets.lock().unwrap_or_else(|e| e.into_inner());
    let entry = sets.entry(props).or_default();
    if !entry.iter().any(|held| Arc::ptr_eq(held, set)) {
        entry.push(Arc::clone(set));
        WATCHED_PROTOTYPES.fetch_add(1, Ordering::AcqRel);
    }
}

/// The shape set of the prototype whose property map is at `props`, shared by every compilation
/// that relies on it until it fires; the caller marks the map a prototype's.
pub(crate) fn prototype_shape_set(props: usize) -> Arc<WatchpointSet> {
    let mut sets = prototypes().sets.lock().unwrap_or_else(|e| e.into_inner());
    let entry = sets.entry(props).or_default();
    if let Some(set) = entry
        .iter()
        .find(|set| set.what == PROTOTYPE_SHAPE && !set.is_invalidated())
    {
        return Arc::clone(set);
    }
    let set = WatchpointSet::new(PROTOTYPE_SHAPE);
    entry.push(Arc::clone(&set));
    WATCHED_PROTOTYPES.fetch_add(1, Ordering::AcqRel);
    set
}

/// A marked prototype's map at `props` changed structurally (Props::note_structural): fire its
/// sets, which then leave the registry.
#[inline]
pub(crate) fn note_prototype_changed(props: usize) {
    // Relaxed: the count only gates the locked path, whose lock orders the registry.
    if WATCHED_PROTOTYPES.load(Ordering::Relaxed) != 0 {
        fire_prototype(props);
    }
}

#[cold]
#[inline(never)]
fn fire_prototype(props: usize) {
    let fired = {
        let mut sets = prototypes().sets.lock().unwrap_or_else(|e| e.into_inner());
        sets.remove(&props)
    };
    if let Some(fired) = fired {
        WATCHED_PROTOTYPES.fetch_sub(fired.len(), Ordering::AcqRel);
        // Fired outside the registry's lock: a set's codes may be watched elsewhere too.
        for set in fired {
            set.fire();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_invalidates_its_codes_once_and_refuses_later_installs() {
        let set = WatchpointSet::new("a test fact");
        assert_eq!((set.state(), set.what()), (WatchState::Clear, "a test fact"));
        let code = CodeDependencies::new();
        let mut desired = DesiredWatchpoints::default();
        desired.add(Arc::clone(&set));
        desired.add(Arc::clone(&set));
        assert_eq!(desired.install(&code), Ok(()));
        assert_eq!(set.state(), WatchState::Watched);
        assert!(code.is_valid());
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(Ordering::Relaxed);
        set.fire();
        assert!(set.is_invalidated());
        assert!(!code.is_valid());
        assert_eq!(code.invalidated_by(), Some("a test fact"));
        assert_ne!(
            crate::bytecode::CALL_IC_EPOCH.load(Ordering::Relaxed),
            epoch,
            "firing an installed code unlinks the call caches"
        );
        set.fire();
        let late = CodeDependencies::new();
        let mut desired = DesiredWatchpoints::default();
        desired.add(set);
        assert_eq!(desired.install(&late), Err("a test fact"));
        assert!(!late.is_valid());
    }
}
