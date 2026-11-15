//! The optimizing tier's record of a chunk (P3 stage 2, step 3; the plan's point 12 and the
//! stage's "Step 3, the exit profile and reoptimization"): what a chunk carries for the tier, made
//! at its first use and dropped with the chunk, so that no record outlives its chunk or passes to
//! a later chunk at the same address. It holds integers and the template JIT's kept code: no GC
//! edge.
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
//! - The tier's code in place (P3 stage 2, step 14): whether an install put the tier's code in
//!   the chunk's own slot, over its template JIT's code (a first stage whose plan splices
//!   nothing), and that template code, kept as the fallback a discard returns to.
//! - The tier's loop limit (P3 stage 2, step 13): whether its warm-up is armed on the chunk's
//!   loop word.
//! - Template code's last-seen element receivers (P3 stage 3, rule U): two bytes per pc, which
//!   the template JIT's in-line element hits store (a typed array's native kind, an Array's or an
//!   ordinary object's kind tag) and the tier's readers merge into the element records
//!   ([`merge_element_hits`]), as JSC's baseline stores a last-seen structure into an access's
//!   ArrayProfile (jit/JITInlines.h:328-348) and computeUpdatedPrediction merges it
//!   (bytecode/ArrayProfile.cpp:126-136).
//!
//! Only the tier reads and writes the record (`snapshot`, `driver`; the template JIT's code, with
//! the tier on, the last-seen bytes): where it has no back end (x86-64 until the plan's stage 6) a
//! chunk carries an empty cell.
#![cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use crate::bytecode::Chunk;
use std::cell::{Cell, OnceCell, RefCell};

/// A last-seen byte no in-line access stored since the last merge (rule U): no native kind is
/// 0xff (native_typed_array.rs: Uint8 0 .. Uint8Clamped 8), and no kind tag (value.rs: below 64).
pub(crate) const UNSEEN: u8 = 0xff;

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
    /// An install put the tier's code in the chunk's own slot, over the template JIT's code (P3
    /// stage 2, step 14: a first stage whose plan splices nothing, compiled as V8 and
    /// SpiderMonkey compile a function's one bytecode): the function's tier-up is this chunk's
    /// from then on (its discards, warm-ups and reoptimizations), and no checkpoint of it builds
    /// a second stage (`Interp::try_inline_recompile`'s done test).
    in_place: Cell<bool>,
    /// The template JIT's code an install of the tier's code replaced in the chunk's slot, kept
    /// as the fallback a discard returns to (P3 stage 2, step 14), as SpiderMonkey's JitScript
    /// keeps the BaselineScript beside the IonScript (jit/JitScript.h:295-400) and JSC's DFG code
    /// block its baseline alternative (bytecode/CodeBlock.cpp:2400-2434): a slot of the code
    /// cache's own, whose CLOCK may evict it as any code (a discard whose kept code is gone
    /// compiles one, as before step 14).
    kept: crate::jit::cache::NativeCodeSlot,
    /// The tier's loop limit is armed on the chunk (P3 stage 2, step 13; `opt::arm_loop_limit`):
    /// its loop word holds the warm-up and its checkpoint word idles, or holds the request of a
    /// loop's crossing, until the words' other uses set them (`Chunk::set_inline_retry_at`) or an
    /// entry takes the check (`opt::loop_limit_due`).
    loop_armed: Cell<bool>,
    /// The tier refused a compile of the chunk that inlined in its graph (TRUST_JS_OPT_INLINE=
    /// graph; P3 stage 3, step 2): its function's next checkpoint splices, as a function the tier
    /// refuses keeps a second stage's splices for its template code.
    graph_refused: Cell<bool>,
    /// The ways the graph inliner took at the chunk's first compile under TRUST_JS_OPT_INLINE=
    /// graph (P3 stage 3, step 2), which its reoptimizations take again (`opt::oracle`), as a
    /// second stage keeps the splices its recompile made: weak pins, no GC edge.
    inline_ways: RefCell<Vec<KeptWay>>,
    /// Template code's last-seen element receivers (P3 stage 3, rule U), two bytes per pc (`2 pc`
    /// and `2 pc + 1`), made by the first template compile that records into them
    /// ([`ChunkProfile::element_byte`]), JSC's ArrayProfile::m_lastSeenStructureID
    /// (bytecode/ArrayProfile.h:245-246): the first [`UNSEEN`] or the native kind of the
    /// typed-array view an in-line access of the site served last (`jit::typed_array::emit`), the
    /// second [`UNSEEN`] or the kind tag of the Array or other ordinary object whose element words
    /// an in-line access of the site took last (`jit::emit_family_record`). Kinds and tags, never
    /// a pointer: nothing the collector reads. The bytes never move: generated code stores to
    /// their addresses.
    elements: OnceCell<Box<[Cell<u8>]>>,
    /// Template code's last-seen equality pairs (P3 stage 3, step L1b): a group of
    /// `feedback::EQ_CLASSES` bytes for each equality of the chunk that keeps a record (all but
    /// those a constant operand decides, `feedback_read::records_equality`), made
    /// by the first template compile that records into them ([`ChunkProfile::equality_bytes`]):
    /// byte k of a group is
    /// [`UNSEEN`], or 0 where an in-line path of the site met a pair of class k
    /// (`feedback::eq_class_bit`) since the last merge (`jit::emit_equality_numbers_record`,
    /// `jit::emit_equality_tags_record`). Classes, never a pointer; the bytes never move.
    equalities: OnceCell<EqualityBytes>,
}

/// The equality groups of a chunk ([`ChunkProfile::equalities`]): the pcs of the equalities that
/// keep a record, ascending, and their groups of `feedback::EQ_CLASSES` bytes in that order.
#[derive(Debug)]
struct EqualityBytes {
    pcs: Box<[u32]>,
    bytes: Box<[Cell<u8>]>,
}

impl EqualityBytes {
    fn new(chunk: &Chunk) -> Self {
        let targets = crate::bytecode::jump_targets(chunk.jit_ops());
        let pcs: Box<[u32]> = (0..chunk.jit_ops().len())
            .filter(|&pc| crate::bytecode::feedback_read::records_equality(chunk, &targets, pc))
            .map(|pc| pc as u32)
            .collect();
        let bytes = (0..pcs.len() * crate::feedback::EQ_CLASSES)
            .map(|_| Cell::new(UNSEEN))
            .collect();
        EqualityBytes { pcs, bytes }
    }

    /// The group of the equality at `pc`, None for a pc that is no equality or one that keeps no
    /// record.
    fn group(&self, pc: usize) -> Option<&[Cell<u8>]> {
        let k = self.pcs.binary_search(&u32::try_from(pc).ok()?).ok()?;
        let n = crate::feedback::EQ_CLASSES;
        Some(&self.bytes[n * k..n * (k + 1)])
    }
}

/// A way the graph inliner took (`ChunkProfile::inline_ways`; P3 stage 3, step 2): its call
/// site, its callee's pin and the planner's facts of it, from which a reoptimization makes the way
/// again while the callee lives.
#[derive(Clone)]
pub(crate) struct KeptWay {
    pub(crate) cache: u32,
    pub(crate) callee: crate::value::WeakGc,
    pub(crate) check_this: bool,
    pub(crate) uses_this: bool,
    pub(crate) expected_genv: usize,
    pub(crate) genv_pin: crate::interpreter::WeakEnv,
}

impl std::fmt::Debug for KeptWay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KeptWay(site {})", self.cache)
    }
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

    /// Whether an install put the tier's code in the chunk's own slot (see the field).
    pub(crate) fn in_place(&self) -> bool {
        self.in_place.get()
    }

    /// Mark an install of the tier's code in the chunk's own slot (see the field).
    pub(crate) fn set_in_place(&self) {
        self.in_place.set(true);
    }

    /// The template JIT's code an install kept (see the field), whose slot the caller takes it
    /// from or puts it in.
    pub(crate) fn kept(&self) -> &crate::jit::cache::NativeCodeSlot {
        &self.kept
    }

    /// Whether the tier's loop limit is armed on the chunk (see the field).
    pub(crate) fn loop_armed(&self) -> bool {
        self.loop_armed.get()
    }

    /// Arm the loop limit (`opt::arm_loop_limit`).
    pub(crate) fn set_loop_armed(&self, armed: bool) {
        self.loop_armed.set(armed);
    }

    /// Whether the tier refused a compile of the chunk that inlined in its graph (see the
    /// field).
    pub(crate) fn graph_refused(&self) -> bool {
        self.graph_refused.get()
    }

    pub(crate) fn set_graph_refused(&self) {
        self.graph_refused.set(true);
    }

    /// The ways the graph inliner took at the chunk's first compile (see the field).
    pub(crate) fn inline_ways(&self) -> Vec<KeptWay> {
        self.inline_ways.borrow().clone()
    }

    pub(crate) fn set_inline_ways(&self, ways: Vec<KeptWay>) {
        *self.inline_ways.borrow_mut() = ways;
    }

    /// The address of `pc`'s last-seen typed-kind byte in a chunk of `len` operations, its family
    /// byte the next (see the `elements` field), the bytes made [`UNSEEN`] at the first ask (main
    /// thread: a template compile that records, rule U); None past the chunk. Generated code
    /// stores a kind there (`jit::typed_array::emit`) and a tag in the next
    /// (`jit::emit_family_record`).
    pub(crate) fn element_byte(&self, len: usize, pc: usize) -> Option<u64> {
        let bytes = self
            .elements
            .get_or_init(|| (0..2 * len).map(|_| Cell::new(UNSEEN)).collect());
        bytes.get(2 * pc).map(|byte| byte.as_ptr() as u64)
    }

    /// `pc`'s last-seen typed-kind byte ([`UNSEEN`] where none was made: tests).
    #[cfg(test)]
    pub(crate) fn element_seen(&self, pc: usize) -> u8 {
        self.seen_byte(2 * pc)
    }

    /// `pc`'s last-seen family byte, a kind tag ([`UNSEEN`] where none was made: tests).
    #[cfg(test)]
    pub(crate) fn family_seen(&self, pc: usize) -> u8 {
        self.seen_byte(2 * pc + 1)
    }

    #[cfg(test)]
    fn seen_byte(&self, at: usize) -> u8 {
        self.elements
            .get()
            .and_then(|bytes| bytes.get(at))
            .map_or(UNSEEN, Cell::get)
    }

    /// Store `kind` in `pc`'s last-seen typed-kind byte of a chunk of `len` operations, as
    /// template code's hit does (tests: the builder's, where no template code runs).
    #[cfg(test)]
    pub(crate) fn see_element(&self, len: usize, pc: usize, kind: u8) {
        self.element_byte(len, pc).expect("a pc of the chunk");
        self.elements.get().expect("made")[2 * pc].set(kind);
    }

    /// The address of the equality at `pc`'s group of last-seen class bytes in `chunk`, this
    /// profile's (see the `equalities` field; P3 stage 3, step L1b), the groups made [`UNSEEN`]
    /// at the first ask, one for each equality of the chunk that keeps a record (main thread: a
    /// template compile that records); None for a pc that is no such equality (steps L1b D and
    /// L1c: one a constant operand decides, which then records nothing). Generated code
    /// stores zero into the byte of the class of each pair it meets.
    pub(crate) fn equality_bytes(&self, chunk: &Chunk, pc: usize) -> Option<u64> {
        let group = self
            .equalities
            .get_or_init(|| EqualityBytes::new(chunk))
            .group(pc)?;
        Some(group[0].as_ptr() as u64)
    }

    /// Mark the pair class `class` met at the equality at `pc` of `chunk`, as template code's
    /// store does (tests, as [`see_element`]; P3 stage 3, step L1b).
    #[cfg(test)]
    pub(crate) fn see_equality(&self, chunk: &Chunk, pc: usize, class: u8) {
        self.equality_bytes(chunk, pc).expect("an equality's pc");
        let group = self
            .equalities
            .get()
            .expect("made")
            .group(pc)
            .expect("made");
        group[usize::from(class)].set(0);
    }

    /// The pair classes met at the equality at `pc` since the last merge, bit k class k (0 where
    /// none was made: tests).
    #[cfg(test)]
    pub(crate) fn equality_seen(&self, pc: usize) -> u32 {
        self.equalities
            .get()
            .and_then(|eq| eq.group(pc))
            .map_or(0, |group| {
                group
                    .iter()
                    .enumerate()
                    .filter(|(_, byte)| byte.get() != UNSEEN)
                    .fold(0, |bits, (k, _)| bits | 1 << k)
            })
    }

    /// Store the kind tag `tag` in `pc`'s last-seen family byte (tests, as [`see_element`]).
    #[cfg(test)]
    pub(crate) fn see_family(&self, len: usize, pc: usize, tag: u8) {
        self.element_byte(len, pc).expect("a pc of the chunk");
        self.elements.get().expect("made")[2 * pc + 1].set(tag);
    }

    /// Heap bytes the record holds (the memory census), the kept code's own bytes left to
    /// `scan_kept`.
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.sites.borrow().capacity() * 8
            + self.elements.get().map_or(0, |bytes| bytes.len())
            + self
                .equalities
                .get()
                .map_or(0, |eq| eq.bytes.len() + 4 * eq.pcs.len())
            + self.inline_ways.borrow().capacity() * std::mem::size_of::<KeptWay>()
            + self.kept.retained_metadata_bytes()
    }

    /// The kept code, for the memory census.
    pub(crate) fn kept_code(&self) -> Option<std::rc::Rc<crate::jit::JitCode>> {
        self.kept.get().flatten()
    }
}

/// Merge template code's last-seen element receivers into the element records the tier reads of
/// `chunk` (P3 stage 3, rule U): its own, and for a transformed chunk those of each baseline
/// chunk its record map names, where its code's hits stored (`opt::element_records`). A typed
/// byte that holds a kind ORs a typed array of that kind, a family byte that holds a tag an Array
/// (`KIND_ARRAY_TAG`) or another ordinary object, each with an index key (the in-line access's),
/// into its site's record through `FeedbackVector::record`, the record's one writer, and is reset
/// to [`UNSEEN`], as JSC's ArrayProfile::computeUpdatedPrediction ORs its last-seen structure into
/// the observed array modes and clears it (bytecode/ArrayProfile.cpp:126-136): a record that holds
/// more (another kind, other receivers, a String key, the generic mark) keeps all of it. JSC's
/// first-run pruning (:137-140, the first merge's modes cut to the last-seen one) is not taken:
/// the record only grows (feedback-schema.txt). Each equality's group of class bytes ORs every
/// class met since the last merge into its record and is reset (P3 stage 3, step L1b). Main
/// thread: the tier's preparation and its
/// checkpoints' fingerprints (`opt::snapshot`), as JSC merges at each tier-up check and at the
/// DFG's read (jit/JITOperations.cpp:2826-2829, 2881, 2940; dfg/DFGByteCodeParser.cpp:
/// 1354-1359). One load where no template code of the chunks recorded.
pub(crate) fn merge_element_hits(chunk: &Chunk) {
    merge_own_element_hits(chunk);
    if let Some(map) = chunk.feedback_vector().record_map() {
        for target in map.targets() {
            merge_own_element_hits(target);
        }
    }
}

/// [`merge_element_hits`] of `chunk`'s own bytes.
fn merge_own_element_hits(chunk: &Chunk) {
    use crate::feedback::{
        elem_typed_kind, ELEM_ARRAY, ELEM_KEY_INDEX, ELEM_ORDINARY, ELEM_TYPED_ARRAY,
    };
    let Some(profile) = ChunkProfile::get(chunk) else {
        return;
    };
    let feedback = chunk.feedback_vector();
    if let Some(eq) = profile.equalities.get() {
        // Each equality's classes met since the last merge (P3 stage 3, step L1b;
        // feedback-schema.txt 5.8), every one of them, as V8 ORs a compare's feedback at every
        // execution (codegen/code-stub-assembler.cc:13847-13866).
        let n = crate::feedback::EQ_CLASSES;
        for (&pc, group) in eq.pcs.iter().zip(eq.bytes.chunks_exact(n)) {
            let mut pairs = 0;
            for (class, byte) in group.iter().enumerate() {
                if byte.replace(UNSEEN) != UNSEEN {
                    pairs |= crate::feedback::eq_class_bit(class as u8);
                }
            }
            if pairs != 0 {
                feedback.record(pc as usize, pairs);
            }
        }
    }
    let Some(bytes) = profile.elements.get() else {
        return;
    };
    for (pc, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let kind = pair[0].replace(UNSEEN);
        if kind != UNSEEN {
            feedback.record(
                pc,
                ELEM_TYPED_ARRAY | ELEM_KEY_INDEX | elem_typed_kind(kind),
            );
        }
        let tag = pair[1].replace(UNSEEN);
        if tag != UNSEEN {
            let family = if tag == crate::value::KIND_ARRAY_TAG {
                ELEM_ARRAY
            } else {
                ELEM_ORDINARY
            };
            feedback.record(pc, family | ELEM_KEY_INDEX);
        }
    }
}

/// End the tier's loop limit on `chunk` where it is armed (the words' other uses come first; P3
/// stage 2, step 13): whether it was. One load where the chunk has no record.
#[inline]
pub(crate) fn disarm_loop_limit(chunk: &Chunk) -> bool {
    ChunkProfile::get(chunk).is_some_and(|profile| profile.loop_armed.replace(false))
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
