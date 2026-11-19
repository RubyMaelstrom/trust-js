//! A global property's value as a constant of the tier's code, under a dependency (F4 part 2),
//! as V8 (compiler/compilation-dependencies.cc's GlobalPropertyDependency, a PropertyCell's
//! DependentCode), SpiderMonkey (vm/ObjectFuse.h's DepMap of dependent IonScripts) and JSC
//! (a WatchpointSet per global variable) fold a global that is stored once.
//!
//! - The snapshot's fold (`prepare`, main thread): a global-object read the snapshot found
//!   (`NameView::Global`) whose property the realm's global fuse makes Constant
//!   (`ObjectFuse::try_optimize_constant`) takes its value as `NameView::Global::folded`, with
//!   the generation it was made Constant under; the code's validity word is made with the first.
//! - The builder reads such a site as a `FoldedGlobal` node after a `CheckCodeValid` of its
//!   effect epoch (`build_names`), which `verify::folded_reads` proves.
//! - The install (`driver::dropped`, `driver::finish`): a job whose folded property moved since its
//!   snapshot is dropped; else the code's word goes on each folded property's list
//!   (`ObjectFuse::add_dependent`), which the property's change, removal or shadowing fires.
//! - A function discarded at `FOLD_LIMIT` Invalidated exits folds nothing again (R-3,
//!   `invalidation_discard`): the exemplars count a jettison as a reoptimization; this engine's
//!   profile is the teacher, so a hot function under many globals that each change once ends
//!   in the tier without folds rather than in template code.
//!
//! Layer 2 folds only under a test's switch (`TEST_FOLDS`): the lists, the word, the exit and
//! the rules, driven by tests; release builds fold nothing.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::profile::ChunkProfile;
use super::snapshot::{Folded, NameView, Raw};
use crate::bytecode::Chunk;
use crate::interpreter::Interp;
use crate::object_fuse::ObjectFuse;

/// The Invalidated discards of one function after which its compiles fold no global (R-3).
pub(crate) const FOLD_LIMIT: u8 = 3;

#[cfg(test)]
thread_local! {
    /// Whether this thread's compiles fold Number globals (tests; layer 2's only fold).
    pub(crate) static TEST_FOLDS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether an Invalidated discard counts the function's R as other discards do (tests:
    /// R-3's two forms, measured by the e4 micro). Default: counted.
    pub(crate) static TEST_INVALIDATION_COUNTS_R: std::cell::Cell<bool> =
        const { std::cell::Cell::new(true) };
}

/// Whether compiles fold globals now.
fn folds_enabled() -> bool {
    #[cfg(test)]
    {
        TEST_FOLDS.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// What a compile folded (see the module's doc): the running realm's global fuse, each folded
/// property's (slot, generation), and the code's validity word, made with the first fold.
#[derive(Default)]
pub(crate) struct Folds {
    fuse: Option<Arc<ObjectFuse>>,
    deps: Vec<(u32, u32)>,
    valid: Option<Arc<AtomicU32>>,
}

impl Folds {
    /// The code's validity word, where the compile folds.
    pub(crate) fn valid(&self) -> Option<&Arc<AtomicU32>> {
        self.valid.as_ref()
    }

    /// The validity word's address, 0 where the compile folds nothing (`a64::Switches`).
    pub(crate) fn valid_word(&self) -> usize {
        self.valid
            .as_ref()
            .map_or(0, |word| Arc::as_ptr(word) as usize)
    }

    /// Whether a folded property is no longer Constant under the generation its fold recorded:
    /// the install's check (`driver::dropped`).
    pub(crate) fn changed(&self) -> bool {
        let Some(fuse) = &self.fuse else {
            return false;
        };
        self.deps
            .iter()
            .any(|&(slot, generation)| !fuse.still_constant(slot as usize, generation))
    }

    /// Put the installed code's validity word on each folded property's list (the install,
    /// `driver::finish`, before the code can run). A property that moved meanwhile sets the
    /// word at once: the code then exits at its first test and is discarded.
    pub(crate) fn register(&self) {
        let (Some(fuse), Some(valid)) = (&self.fuse, &self.valid) else {
            return;
        };
        for &(slot, generation) in &self.deps {
            if !fuse.add_dependent(slot as usize, generation, valid) {
                valid.store(1, Ordering::Relaxed);
            }
        }
    }
}

/// The fold of a compile of `chunk` in `i`'s running realm (see the module's doc), over the
/// global-object reads of `raws` (the compiled chunk's snapshot and its inlined callees'),
/// on the main thread after the snapshot and before the job leaves it.
pub(crate) fn prepare<'a>(
    i: &Interp,
    chunk: &Chunk,
    raws: impl Iterator<Item = &'a mut Raw>,
) -> Folds {
    let mut folds = Folds::default();
    if !folds_enabled() || !folds_globals(chunk) {
        return folds;
    }
    let Some(fuse) = i.global_fuse() else {
        return folds;
    };
    let Ok(global) = i.global.try_borrow() else {
        return folds;
    };
    let global_cell = crate::value::Gc::as_ptr(&i.global) as usize;
    for raw in raws {
        for view in raw.global_reads_mut() {
            let NameView::Global {
                global: of,
                slot,
                folded,
                ..
            } = view
            else {
                continue;
            };
            if *of != global_cell {
                continue;
            }
            let Some(value) = global
                .props
                .slot_at(*slot as usize)
                .filter(|value| !value.accessor())
                .and_then(|value| value.number_value().map(|_| value.value()))
            else {
                continue;
            };
            let Some(guard) = fuse.try_optimize_constant(*slot as usize) else {
                continue;
            };
            *folded = Some(Folded {
                word: value.bits(),
                generation: guard.generation,
            });
            if !folds.deps.contains(&(*slot, guard.generation)) {
                folds.deps.push((*slot, guard.generation));
            }
        }
    }
    if !folds.deps.is_empty() {
        folds.fuse = Some(fuse);
        folds.valid = Some(Arc::new(AtomicU32::new(0)));
    }
    folds
}

/// Whether `chunk`'s function still folds globals: fewer than [`FOLD_LIMIT`] of its codes were
/// discarded at an Invalidated exit (R-3).
fn folds_globals(chunk: &Chunk) -> bool {
    let first = super::profile::first_stage(chunk);
    ChunkProfile::get(first).is_none_or(|profile| profile.invalidations() < FOLD_LIMIT)
}

/// A discard at an Invalidated exit of a code of the function whose first stage is `first`
/// (`driver::discard`): the new R, which the discard counts as any other's (JSC's watchpoint
/// jettison counts a reoptimization, bytecode/CodeBlockJettisoningWatchpoint.cpp:46), and
/// whether this discard reached [`FOLD_LIMIT`], from which the function folds nothing.
pub(crate) fn invalidation_discard(first: &Chunk) -> (u8, bool) {
    let profile = ChunkProfile::of(first);
    let limited = profile.add_invalidation() == FOLD_LIMIT;
    #[cfg(test)]
    if !TEST_INVALIDATION_COUNTS_R.with(std::cell::Cell::get) {
        return (profile.discards(), limited);
    }
    (profile.add_discard(), limited)
}
