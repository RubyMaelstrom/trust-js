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
//! Folded: Numbers (layer 3), whose word is a constant of the code (no object, nothing to
//! root); typed arrays (layer 4a), whose word is an ordinary value with a home (the review's
//! R-2: no root in the code, the global holds the object while the code's word is clear),
//! loaded from the code's literal pool as V8's arm64 loads an embedded object
//! (MacroAssembler::Mov, codegen/arm64/macro-assembler-arm64.cc:449), with the facts its
//! identity fixes: an object, a typed array, its view's element kind (no CheckObject, no
//! CheckElements); its length and data stay loads (TypedLength, TypedData).
//! TRUST_JS_OPT_FOLD_GLOBALS=0 turns the fold off, for measurement.

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
    /// Whether this thread's compiles fold globals, whatever the environment says (tests: a test
    /// of the unfolded reads pins it off); None: as the environment says.
    pub(crate) static TEST_FOLDS: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
    /// Whether an Invalidated discard counts the function's R as other discards do (tests:
    /// R-3's two forms, measured by the e4 micro). Default: counted.
    pub(crate) static TEST_INVALIDATION_COUNTS_R: std::cell::Cell<bool> =
        const { std::cell::Cell::new(true) };
}

/// Whether compiles fold globals: unless TRUST_JS_OPT_FOLD_GLOBALS=0 (measurement only), as
/// the owner decided (the design's j3).
fn folds_enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_FOLDS.with(std::cell::Cell::get) {
        return on;
    }
    env_value!("TRUST_JS_OPT_FOLD_GLOBALS") != Some("0")
}

/// Pin the fold on or off on this thread, whatever the environment says (tests).
#[cfg(test)]
pub(crate) fn pin_folds_for_test(on: bool) {
    TEST_FOLDS.with(|folds| folds.set(Some(on)));
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

    /// The folded properties (the log's).
    pub(crate) fn count(&self) -> usize {
        self.deps.len()
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
                .map(|value| value.value())
            else {
                continue;
            };
            // A Number (layer 3), or a typed array of a kind the tier's element sites take
            // (layer 4a); nothing else folds.
            let typed = typed_kind(&value);
            let function = callable(&value);
            let number = matches!(value.view(), crate::value::ValueView::Num(_));
            if !number && typed.is_none() && !function {
                continue;
            }
            let Some(guard) = fuse.try_optimize_constant(*slot as usize) else {
                continue;
            };
            *folded = Some(Folded {
                word: value.bits(),
                generation: guard.generation,
                typed,
                function,
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

/// The element kind of `value` where it is a typed array whose view the tier's element sites take
/// (a kind of `TypedKind::ALL`; a declined view's kind byte is none of them): what a fold of it
/// may assume for the object's life. Its being a typed array (the object's kind) and its view's
/// kind are written once, at its construction (`native_typed_array.rs`); its length and its data
/// are not facts of the object (a detach, a resize, a transfer change them while the property
/// holds the same object), so the fold leaves them to the loads they are (layer 4a; 4b's
/// detaching dependency is a separate piece).
fn typed_kind(value: &crate::value::Value) -> Option<super::ir::TypedKind> {
    let crate::value::ValueView::Obj(object) = value.view() else {
        return None;
    };
    let object = object.try_borrow().ok()?;
    let crate::value::ObjectKind::TypedArray(view) = &object.kind else {
        return None;
    };
    super::ir::TypedKind::ALL
        .get(usize::from(view.kind))
        .copied()
}

/// Whether `value` is a function: its identity fixes what a callee guard tests of it, its
/// [[Environment]], its Realm and its code, all written at its creation (functions piece).
fn callable(value: &crate::value::Value) -> bool {
    let crate::value::ValueView::Obj(object) = value.view() else {
        return false;
    };
    object
        .try_borrow()
        .is_ok_and(|object| object.kind.is_callable())
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
