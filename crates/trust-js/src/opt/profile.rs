//! The optimizing tier's record of a chunk (P3 stage 2, step 3; the plan's point 12 and the
//! stage's "Step 3, the exit profile and reoptimization"): what a chunk carries for the tier, made
//! at its first use and dropped with the chunk, so that no record outlives its chunk or passes to
//! a later chunk at the same address. It holds integers only: no GC edge.
//!
//! - The feedback fingerprint the settled-feedback rule recorded at the chunk's last checkpoint
//!   (`opt::settled`): a first stage's before its second stage is built, a discarded second
//!   stage's during its warm-up.
//! - The exit profile: the sites whose exits a discarded code took, each a (pc, reason, inlined)
//!   key, as JSC's DFG::ExitProfile keys a site by its bytecode index, its exit kind and its
//!   inline kind (bytecode/DFGExitProfile.h:45-145). A key lives on the baseline chunk the
//!   discarded second stage's record map names for the exit's pc, so that every later compile of
//!   that site finds it (the second stage again, another caller's splice, the callee's own second
//!   stage), as JSC adds a site to the code block of the exit's semantic origin
//!   (bytecode/DFGExitProfile.cpp:44-75); a pc the map names nothing for keys on the second stage
//!   itself. A WrongShape key keeps the most shapes a discarded check at its site expected, which
//!   tells a recompile whether the site's ways grew since.
//! - R, the function's count of discards, on its first stage, as JSC keeps its reoptimization
//!   retry counter on the baseline code block (bytecode/CodeBlock.cpp:2478-2483).
//! - A discarded second stage's warm-up: its next checkpoint toward the recompile and the native
//!   run count it is due at.
//!
//! Only the tier's back end reads and writes the record (`snapshot`, `driver`): where it has none
//! (x86-64 until the plan's stage 6) a chunk carries an empty cell.
#![cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use crate::bytecode::Chunk;
use std::cell::{Cell, RefCell};

/// The most discards R counts: the warm-up's longest distance, the 6,400 invocations << R to its
/// fourth checkpoint, 409,600 << R native runs at K = 64 an invocation (`opt::ENTRY_WEIGHT`;
/// stage 2, step 11), then still fits the u32 checkpoint word (409,600 << 13 is 3,355,443,200),
/// as JSC's cap comes from its counter's range (runtime/Options.cpp:940-947). Optimization is
/// never disabled for good.
pub(crate) const MAX_DISCARDS: u8 = 13;

/// A chunk's record (see the module's doc).
#[derive(Debug, Default)]
pub(crate) struct ChunkProfile {
    settling: Cell<Option<u64>>,
    /// The exit sites, ascending by key: `pc << 8 | reason << 1 | inlined`, and for a WrongShape
    /// key the most shapes a discarded check there expected (0 otherwise).
    sites: RefCell<Vec<(u32, u8)>>,
    discards: Cell<u8>,
    /// The warm-up's next checkpoint (from 1, saturating at 255), 0 while none is armed.
    warm_up: Cell<u8>,
    /// The native run count the warm-up's next checkpoint is due at.
    warm_up_at: Cell<u32>,
    /// Under TRUST_JS_OPT_LOG only: the discard's compile of the template JIT's fallback was
    /// deferred for executable capacity, so the chunk's next compile is the fallback's
    /// (`opt::fallback_timer`).
    fallback_deferred: Cell<bool>,
    /// A compile this chunk's checkpoint started runs off the main thread (P3 stage 2, step 12):
    /// no checkpoint of the chunk is armed meanwhile, and none compiles it again until the
    /// install, as V8 marks a function tiering_in_progress (objects/feedback-vector.h:417-421)
    /// and SpiderMonkey a script isIonCompilingOffThread (jit/Ion.cpp:1918).
    compiling: Cell<bool>,
    /// Counts the changes of what a snapshot reads of the record (a site added or its shapes or
    /// proof changed, a discard counted), so that an install can tell that a discard flagged
    /// sites since its snapshot (stage 2, step 12).
    revision: Cell<u32>,
}

/// The key of an exit site.
fn key(pc: u32, reason: u8, inlined: bool) -> u32 {
    debug_assert!(pc <= crate::feedback::RecordMap::MAX_PC as u32 && reason < 0x80);
    pc << 8 | u32::from(reason) << 1 | u32::from(inlined)
}

impl ChunkProfile {
    /// `chunk`'s record, made at its first use.
    pub(crate) fn of(chunk: &Chunk) -> &ChunkProfile {
        chunk.opt.get_or_init(Box::default)
    }

    /// `chunk`'s record, if it has one.
    pub(crate) fn get(chunk: &Chunk) -> Option<&ChunkProfile> {
        chunk.opt.get().map(|profile| &**profile)
    }

    /// Record the fingerprint of this checkpoint; the previous checkpoint's.
    pub(crate) fn settle(&self, print: u64) -> Option<u64> {
        self.settling.replace(Some(print))
    }

    /// The chunk's checkpoints ended, or its wait starts afresh.
    pub(crate) fn unsettle(&self) {
        self.settling.set(None);
    }

    /// Add the exit site (`pc`, the reason's code, `inlined`), with the shapes its check expected
    /// for a WrongShape site, the most a discarded check there expected, or, `latest`, the proof's
    /// byte its check expected for a WrongName site, the last discard's (stage 2, step 4):
    /// whether the site is new.
    pub(crate) fn add_site(
        &self,
        pc: u32,
        reason: u8,
        inlined: bool,
        ways: u8,
        latest: bool,
    ) -> bool {
        let key = key(pc, reason, inlined);
        let mut sites = self.sites.borrow_mut();
        match sites.binary_search_by_key(&key, |&(key, _)| key) {
            Ok(at) => {
                let next = if latest { ways } else { sites[at].1.max(ways) };
                if next != sites[at].1 {
                    sites[at].1 = next;
                    self.revise();
                }
                false
            }
            Err(at) => {
                sites.insert(at, (key, ways));
                self.revise();
                true
            }
        }
    }

    fn revise(&self) {
        self.revision.set(self.revision.get().wrapping_add(1));
    }

    /// The record's revision (see the field), which an install compares with its snapshot's.
    pub(crate) fn revision(&self) -> u32 {
        self.revision.get()
    }

    /// Whether a compile the chunk's checkpoint started runs off the main thread.
    pub(crate) fn compiling(&self) -> bool {
        self.compiling.get()
    }

    /// Mark a compile of the chunk running off the main thread, or its end.
    pub(crate) fn set_compiling(&self, compiling: bool) {
        self.compiling.set(compiling);
    }

    /// Whether the record holds exit sites.
    pub(crate) fn has_sites(&self) -> bool {
        !self.sites.borrow().is_empty()
    }

    /// The exit sites at `pc`: each one's reason code, whether it is inlined, and its shapes.
    pub(crate) fn sites_at(&self, pc: u32, mut each: impl FnMut(u8, bool, u8)) {
        let sites = self.sites.borrow();
        let first = sites.partition_point(|&(key, _)| key < pc << 8);
        for &(key, ways) in sites[first..]
            .iter()
            .take_while(|&&(key, _)| key >> 8 == pc)
        {
            each((key >> 1 & 0x7f) as u8, key & 1 != 0, ways);
        }
    }

    /// R, the function's discards so far (on its first stage).
    pub(crate) fn discards(&self) -> u8 {
        self.discards.get()
    }

    /// Count a discard; the new R.
    pub(crate) fn add_discard(&self) -> u8 {
        let r = self.discards.get().saturating_add(1).min(MAX_DISCARDS);
        self.discards.set(r);
        self.revise();
        r
    }

    /// The warm-up's next checkpoint and the run count it is due at, while one is armed.
    pub(crate) fn warm_up(&self) -> Option<(u8, u32)> {
        let checkpoint = self.warm_up.get();
        (checkpoint != 0).then(|| (checkpoint, self.warm_up_at.get()))
    }

    /// Arm the warm-up's `checkpoint`, due at `at` native runs.
    pub(crate) fn arm(&self, checkpoint: u8, at: u32) {
        self.warm_up.set(checkpoint);
        self.warm_up_at.set(at);
    }

    /// The warm-up ends: its recompile was made, or refused, or none follows.
    pub(crate) fn disarm(&self) {
        self.warm_up.set(0);
    }

    /// Whether the fallback's compile is deferred (see the field; the log's only).
    pub(crate) fn fallback_deferred(&self) -> bool {
        self.fallback_deferred.get()
    }

    /// Mark the fallback's compile deferred, or clear the mark (the log's only).
    pub(crate) fn set_fallback_deferred(&self, deferred: bool) {
        self.fallback_deferred.set(deferred);
    }

    /// Heap bytes the record holds (the memory census).
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.sites.borrow().capacity() * 8
    }
}

/// The function's first stage, which holds R and the keys of its own sites: the baseline chunk
/// a recompile's record map names first, or the chunk itself, which has no map.
pub(crate) fn first_stage(chunk: &Chunk) -> &Chunk {
    chunk
        .feedback_vector()
        .record_map()
        .and_then(|map| map.targets().first())
        .map_or(chunk, |first| &**first)
}

/// The baseline chunk and pc a recompile's record map names for its `pc`, and whether the site
/// is a splice's; None for a pc the map names nothing for (guards, calls, block starts that are
/// no value or named site), and for a chunk without a map.
pub(crate) fn baseline_site(chunk: &Chunk, pc: usize) -> Option<(&Chunk, usize, bool)> {
    chunk.feedback_vector().record_map()?.site(pc)
}
