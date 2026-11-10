//! The tier's summary of a chunk's feedback (the plan's Design point 3 and the Stage 1 record's
//! front end): one view of the read API (`crate::feedback::read`) per pc, copied once when a
//! compile starts, so that the builder reads only this snapshot and the chunk's immutable
//! operations, as V8's ProcessedFeedback, JSC's status objects and SpiderMonkey's WarpSnapshot
//! serve their builders. Taking it allocates the per-pc arrays (the third only where an exit
//! profile has sites) and nothing else; it interns nothing (the views hold process shape ids and
//! cell addresses).
//!
//! A transformed chunk (a second stage, its hot callees spliced in) keeps its own property and
//! call caches, which the second stage seeded from the first stage and goes on filling, but its
//! value and element records are its baseline sites' (W4 step 7: its own record words only
//! filter what it forwards): the snapshot reads those through the chunk's record map.
//!
//! The evidence that code ran comes from named and call sites only: their first execution fills a
//! way or sets the generic or megamorphic mark, and a second stage's seeded caches count. Value
//! and element records are no evidence, since in-line hits record nothing (rule U).
//!
//! A free name's site (stage 2, step 4) has a view of its own ([`NameView`]): its static address,
//! or its name cache's class, a global property resolved again in the realm's global object as
//! V8's broker reads a property cell and Warp's oracle a stub. An empty cache is evidence that the
//! site never ran, and the refused-fill mark (`bytecode::REFUSED_NAME_IC`) that it ran; a static
//! address is none, since the templates' static reads fill no cache.
//!
//! The exit profile (stage 2; `opt::profile`) joins the feedback as each pc's declined reasons:
//! the chunk's own sites at the pc and those of the baseline site its record map names, a value
//! or coverage site from any inline kind and a named or call site only from a site of the pc's
//! own kind (spliced or not), as JSC's parser takes its decisions' exit sites at any inline kind
//! and its IC statuses' at the matching one (bytecode/DFGExitProfile.h:195-232,
//! GetByStatus.cpp:462-475, CallLinkStatus.cpp:116-119).

use super::ir::ExitReason;
use super::profile::ChunkProfile;
use crate::bytecode::{Chunk, NameIc, Op, StaticName};
use crate::feedback::{self, CallState, NamedView, RecordMap, SiteView, ValuesView};
use crate::interpreter::{Env, Interp};

/// Whether a pc's feedback shows that its operation ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ran {
    /// The feedback says nothing either way: an operation without a property or call cache, or a
    /// `length` read without a way (a String's length is read by its tag and fills none).
    Unknown,
    /// A named site with a filled way or its generic mark, or a call site that is not empty.
    Yes,
    /// A named or call site holding neither: it never ran.
    No,
}

/// The per-pc feedback one compile reads (see the module's doc).
pub(crate) struct Snapshot {
    views: Box<[SiteView]>,
    ran: Box<[Ran]>,
    /// Each pc's declined reasons (bits by discriminant) and, for WrongShape, the most shapes a
    /// discarded check there expected, for WrongName the proof byte a discarded check there
    /// expected; empty where no chunk the compile reads has exit sites.
    declined: Box<[(u32, u8)]>,
    /// Each free-name site's view (see [`NameView`]); empty for a chunk without one.
    names: Box<[NameView]>,
}

// Every reason has its bit in a declined mask: the discriminants run from 0 to the last reason's
// (ExitReason's last variant), which the mask's width bounds.
const _: () = assert!((ExitReason::PhiNotInt32 as u32) < u32::BITS);

/// Where a free-name read's search starts (stage 2, step 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Search {
    /// The record this many parent links from the running environment: the realm's global
    /// environment for a static global (`Chunk::jit_name_global`), else the running one (0).
    Running(u16),
    /// A spliced callee's [[Environment]] (`Op::LoadNameIn`), a compile-time constant while the
    /// splice's guard holds, which proved the realm too.
    Pinned,
}

/// The record a static address counts its hops from (stage 2, step 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Base {
    /// The running environment.
    Running,
    /// For a splice's read (`Op::LoadNameIn`): the record its hops reach from the pinned
    /// [[Environment]], folded at the compile (JSC's SkipScope on a constant base): parent links
    /// do not change once code can read through them (the plan's Q3).
    Holder(usize),
    /// A splice whose pinned function is dead or whose chain is shorter than its hops: its guard
    /// never passes, and the read stays the effect.
    Unknown,
}

/// A free-name operation's site as one compile reads it (stage 2, step 4; the plan's "Step 4"):
/// LoadName, LoadNameForCall and their splices' forms, StoreNameCached, UpdateNameCached, and a
/// fixed activation's captured bindings (LoadCap, StoreCap, StoreCapInit, UpdateCap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NameView {
    /// No free-name site: another operation, or a captured binding outside a fixed activation.
    None,
    /// Decision 21's static address (a captured binding's at hops 0): no guard, as V8's context
    /// slots, SpiderMonkey's GetAliasedVar and JSC's GetClosureVar read theirs.
    Static { address: StaticName, base: Base },
    /// A data property of the realm's global object (the cache in global-object mode on the
    /// realm's global environment), found again now: its slot and location word
    /// (`value::location_word`) and the global property epoch (`Props::slot_epoch`); the search's
    /// environment and the global object the code checks are the realm's (`genv`, and `global`,
    /// the object's cell). SpiderMonkey's GetGName (jit/CacheIR.cpp:3467-3482).
    Global {
        search: Search,
        genv: usize,
        global: usize,
        slot: u32,
        location: u16,
        epoch: u16,
    },
    /// A binding of the realm's global declarative record, a script-level `let`, `const` or
    /// `class` (the cache in scope mode on the realm's global environment), under the cache's
    /// own proof: the record's generation and the binding's address. Not unguarded, as
    /// SpiderMonkey reads its global lexical slots: the record does not only append (each
    /// script's GlobalDeclarationInstantiation adds its var names and removes them again, which
    /// moves the last binding into the gap, `Interp::run_program_body`; a sloppy eval's
    /// deletable bindings go too, `Interp::delete_ident`) and its slots move when it outgrows
    /// its cell (the plan's Q4). No TDZ test: the cache filled only an initialized binding, and
    /// a global lexical never returns to its TDZ (SpiderMonkey's rule, CacheIR.cpp:3450-3466).
    GlobalLexical {
        search: Search,
        genv: usize,
        generation: u32,
        binding: usize,
    },
    /// A cache in a mode the tier keeps the effect for (scope, depth-1, layout, deep, another
    /// realm's), or a store's or update's that has no static address.
    Dynamic,
    /// An empty cache: the site never ran.
    Empty,
    /// The refused-fill mark: the site ran, and no fill could cache it.
    Refused,
}

impl NameView {
    /// The byte of the proof a site's checks expect, which a WrongName exit keeps in the exit
    /// profile as WrongShape's keeps its shapes: a recompile whose proof's byte differs (the
    /// epoch moved, another realm's cache) checks again (stage 2, step 4).
    pub(crate) fn proof(&self) -> u8 {
        match *self {
            NameView::Global { genv, epoch, .. } => (genv >> 4) as u8 ^ epoch as u8,
            NameView::GlobalLexical {
                genv, generation, ..
            } => (genv >> 4) as u8 ^ generation as u8,
            _ => 0,
        }
    }
}

impl Snapshot {
    /// The feedback of every operation of `chunk` (its `jit_ops`), as it stands now, its global
    /// names resolved in no realm (they stay effects): the tests' view.
    #[cfg(test)]
    pub(crate) fn take(chunk: &Chunk) -> Snapshot {
        Snapshot::take_in(chunk, None)
    }

    /// The feedback of every operation of `chunk` (its `jit_ops`), as it stands now, its global
    /// names resolved in `realm`'s running realm, the one the compile runs in, whose global
    /// environment and object its code checks (in none, they stay effects).
    pub(crate) fn take_in(chunk: &Chunk, realm: Option<&Interp>) -> Snapshot {
        let ops = chunk.jit_ops();
        let map = chunk.feedback_vector().record_map();
        let mut views = Vec::with_capacity(ops.len());
        let mut ran = Vec::with_capacity(ops.len());
        let mut names = Vec::new();
        for (pc, op) in ops.iter().enumerate() {
            let view = view(chunk, map, pc);
            let name = name_view(chunk, op, realm);
            if name == NameView::None {
                ran.push(evidence(chunk, op, &view));
            } else {
                if names.is_empty() {
                    names = vec![NameView::None; ops.len()];
                }
                ran.push(name_evidence(&name));
                names[pc] = name;
            }
            views.push(view);
        }
        Snapshot {
            views: views.into_boxed_slice(),
            ran: ran.into_boxed_slice(),
            declined: declined(chunk, map, ops.len()),
            names: names.into_boxed_slice(),
        }
    }

    /// Whether the exit profile flagged `reason` at `pc`: the builder declines what it names.
    pub(crate) fn declines(&self, pc: usize, reason: ExitReason) -> bool {
        self.declined
            .get(pc)
            .is_some_and(|&(reasons, _)| reasons & bit(reason) != 0)
    }

    /// The most shapes a discarded check at `pc` expected, where WrongShape is flagged there.
    pub(crate) fn discarded_ways(&self, pc: usize) -> usize {
        self.declined
            .get(pc)
            .map_or(0, |&(_, ways)| usize::from(ways))
    }

    /// The proof byte a discarded check at `pc` expected, where WrongName is flagged there
    /// ([`NameView::proof`]).
    pub(crate) fn discarded_proof(&self, pc: usize) -> u8 {
        self.declined.get(pc).map_or(0, |&(_, proof)| proof)
    }

    /// Whether a free-name site's view reads the realm whose global environment is `genv` (a
    /// global property's or a global lexical's), which the code may then check by identity: the
    /// compile pins it (`driver::prepare`).
    #[cfg_attr(
        not(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        )),
        allow(dead_code)
    )]
    pub(crate) fn reads_global_names_of(&self, genv: usize) -> bool {
        self.names.iter().any(|name| {
            matches!(*name, NameView::Global { genv: g, .. } | NameView::GlobalLexical { genv: g, .. } if g == genv)
        })
    }

    /// The view of the free-name site at `pc` ([`NameView::None`] for any other operation).
    pub(crate) fn name(&self, pc: usize) -> NameView {
        self.names.get(pc).copied().unwrap_or(NameView::None)
    }

    /// The operation count the snapshot covers.
    pub(crate) fn len(&self) -> usize {
        self.views.len()
    }

    /// The view of the operation at `pc` (None past the end).
    pub(crate) fn site(&self, pc: usize) -> SiteView {
        self.views.get(pc).copied().unwrap_or(SiteView::None)
    }

    /// Whether the operation at `pc` ran, by its feedback.
    pub(crate) fn ran(&self, pc: usize) -> Ran {
        self.ran.get(pc).copied().unwrap_or(Ran::Unknown)
    }

    /// The value classes of an arithmetic, unary, update or relational operation (an update of a
    /// named property's too).
    pub(crate) fn values(&self, pc: usize) -> Option<ValuesView> {
        match self.site(pc) {
            SiteView::Values(values) | SiteView::NamedUpdate(_, values) => Some(values),
            _ => None,
        }
    }

    /// The ways of a named load, store or update.
    pub(crate) fn named(&self, pc: usize) -> Option<NamedView> {
        match self.site(pc) {
            SiteView::Named(named) | SiteView::NamedUpdate(named, _) => Some(named),
            _ => None,
        }
    }

    /// The element record of an element load or store (P3 stage 2 step 2): a transformed chunk's
    /// is its baseline site's, as values() reads it. An update of an element (UpdateElem) has a
    /// value record instead (feedback-schema.txt 5.2).
    pub(crate) fn element(&self, pc: usize) -> Option<crate::feedback::ElementView> {
        match self.site(pc) {
            SiteView::Element(element) => Some(element),
            _ => None,
        }
    }
}

/// A reason's bit in a declined mask.
fn bit(reason: ExitReason) -> u32 {
    1 << reason as u8
}

/// Whether a reason's sites apply only where the inline kind matches: a named access's shapes,
/// a call's callees and a free name's proof are a site's in its context, as an IC status is,
/// where a value's type, a binding's TDZ or a block's coverage is the operation's anywhere.
fn kind_bound(reason: u8) -> bool {
    reason == ExitReason::WrongShape as u8
        || reason == ExitReason::WrongCallee as u8
        || reason == ExitReason::WrongName as u8
}

/// The exit profile's declined reasons per pc of `chunk` (see the module's doc): empty unless the
/// chunk or a baseline chunk its map names holds exit sites.
fn declined(chunk: &Chunk, map: Option<&RecordMap>, len: usize) -> Box<[(u32, u8)]> {
    let has_sites = |chunk: &Chunk| ChunkProfile::get(chunk).is_some_and(ChunkProfile::has_sites);
    let own = ChunkProfile::get(chunk).filter(|profile| profile.has_sites());
    let baselines = map.is_some_and(|map| map.targets().iter().any(|target| has_sites(target)));
    if own.is_none() && !baselines {
        return Box::new([]);
    }
    let mut declined = vec![(0u32, 0u8); len];
    for (pc, (reasons, ways)) in declined.iter_mut().enumerate() {
        let mut add = |reason: u8, shapes: u8| {
            *reasons |= 1u32 << reason;
            if reason == ExitReason::WrongShape as u8 {
                *ways = (*ways).max(shapes);
            } else if reason == ExitReason::WrongName as u8 {
                // A name site's byte is the proof its discarded check expected, as it is.
                *ways = shapes;
            }
        };
        // The chunk's own sites, keyed at its pcs: a recompile's where its map names nothing,
        // or a baseline chunk's own (its spliced sites are its callers' context).
        if let Some(profile) = own {
            profile.sites_at(pc as u32, |reason, inlined, shapes| {
                if !kind_bound(reason) || !inlined {
                    add(reason, shapes);
                }
            });
        }
        let Some((baseline, baseline_pc, spliced)) = map.and_then(|map| map.site(pc)) else {
            continue;
        };
        if let Some(profile) = ChunkProfile::get(baseline) {
            profile.sites_at(baseline_pc as u32, |reason, inlined, shapes| {
                if !kind_bound(reason) || inlined == spliced {
                    add(reason, shapes);
                }
            });
        }
    }
    declined.into_boxed_slice()
}

/// The view of `pc` the tier reads: the chunk's own, with a transformed chunk's value and element
/// records its baseline site's (`map` is the chunk's record map).
fn view(chunk: &Chunk, map: Option<&RecordMap>, pc: usize) -> SiteView {
    let own = feedback::read(chunk, pc);
    match map.and_then(|map| map.target(pc)) {
        Some((baseline, baseline_pc)) => {
            with_baseline_records(own, feedback::read(baseline, baseline_pc))
        }
        None => own,
    }
}

/// A transformed chunk's view of a site with its baseline site's value or element record in place
/// of its own (which holds only the in-line set: a transformed chunk reads no record of its own).
fn with_baseline_records(own: SiteView, baseline: SiteView) -> SiteView {
    match (own, baseline) {
        (SiteView::Values(_), SiteView::Values(values)) => SiteView::Values(values),
        (SiteView::Element(_), SiteView::Element(element)) => SiteView::Element(element),
        (SiteView::NamedUpdate(named, _), SiteView::NamedUpdate(_, values)) => {
            SiteView::NamedUpdate(named, values)
        }
        (own, _) => own,
    }
}

/// The evidence the view of `op` gives (see [`Ran`]).
fn evidence(chunk: &Chunk, op: &Op, view: &SiteView) -> Ran {
    match view {
        SiteView::Named(named) | SiteView::NamedUpdate(named, _) => {
            if named.generic || named.filled().next().is_some() {
                Ran::Yes
            } else if is_length_read(chunk, op) {
                Ran::Unknown
            } else {
                Ran::No
            }
        }
        SiteView::Call(call) if call.state == CallState::Empty => Ran::No,
        SiteView::Call(_) => Ran::Yes,
        _ => Ran::Unknown,
    }
}

/// The evidence a free-name site's view gives (see [`Ran`]): none from a static address, which
/// the templates read without their cache; an empty cache says the site never ran, any other
/// cache (the refused-fill mark among them) that it ran.
fn name_evidence(name: &NameView) -> Ran {
    match name {
        NameView::None | NameView::Static { .. } => Ran::Unknown,
        NameView::Empty => Ran::No,
        NameView::Global { .. }
        | NameView::GlobalLexical { .. }
        | NameView::Dynamic
        | NameView::Refused => Ran::Yes,
    }
}

/// The free-name operation `op`'s name and cache, and its splice's inline target.
fn name_site(op: &Op) -> Option<(u32, u32, Option<u32>)> {
    Some(match *op {
        Op::LoadName(name, cache)
        | Op::LoadNameForCall(name, cache)
        | Op::StoreNameCached(name, cache)
        | Op::UpdateNameCached(name, cache, _) => (name, cache, None),
        Op::LoadNameIn(name, cache, target) | Op::LoadNameForCallIn(name, cache, target) => {
            (name, cache, Some(target))
        }
        _ => return None,
    })
}

/// The view of the free-name operation `op` of `chunk` (see [`NameView`]), its global names
/// resolved in `realm`'s running realm.
fn name_view(chunk: &Chunk, op: &Op, realm: Option<&Interp>) -> NameView {
    if let Op::LoadCap(name)
    | Op::StoreCap(name)
    | Op::StoreCapInit(name)
    | Op::UpdateCap(name, _) = *op
    {
        // A captured binding of a fixed activation, at its slot of the running record.
        return match chunk.jit_cap_binding(name) {
            Some((slot, attrs)) => match u16::try_from(slot) {
                Ok(slot) => NameView::Static {
                    address: StaticName {
                        hops: 0,
                        slot,
                        lexical: attrs.lexical(),
                        mutable: attrs.mutable(),
                    },
                    base: Base::Running,
                },
                Err(_) => NameView::None,
            },
            None => NameView::None,
        };
    }
    let Some((name, cache, target)) = name_site(op) else {
        return NameView::None;
    };
    if let Some(address) = chunk.jit_name_static(cache) {
        let base = match target {
            None => Base::Running,
            Some(target) => holder(chunk, target, address.hops),
        };
        return NameView::Static { address, base };
    }
    let ic = chunk.jit_name_ic(cache);
    let read = !matches!(op, Op::StoreNameCached(..) | Op::UpdateNameCached(..));
    match realm {
        Some(realm) if read && !ic.unfilled() => {
            global_view(chunk, realm, name, cache, target, &ic).unwrap_or(NameView::Dynamic)
        }
        _ => cache_view(&ic),
    }
}

/// A global read's view (see [`NameView::Global`]): its cache in global-object mode on `realm`'s
/// global environment, the name found again now as a data property of the realm's global object,
/// ordinary and IC_PLAIN, which no global lexical shadows, at a slot a location word reaches; or
/// its cache in scope mode there, a global lexical's (see [`NameView::GlobalLexical`]), whose
/// proof still holds. A splice's only where its pinned [[Environment]] is that environment and
/// its guard proves the realm. None otherwise.
fn global_view(
    chunk: &Chunk,
    realm: &Interp,
    name: u32,
    cache: u32,
    target: Option<u32>,
    ic: &NameIc,
) -> Option<NameView> {
    let genv = Env::as_ptr(&realm.global_env) as usize;
    let search = match target {
        Some(target) => {
            let pinned = chunk.inline_target_env(target)?;
            if Env::as_ptr(&pinned) as usize != genv
                || chunk.jit_inline_target(target).expected_genv != genv
            {
                return None;
            }
            Search::Pinned
        }
        None => Search::Running(chunk.jit_name_global(cache).unwrap_or(0)),
    };
    if ic.env == genv {
        // Scope mode on the global environment: a global lexical, under its cache's proof.
        let record = realm.global_env.try_borrow().ok()?;
        return (ic.gen != u32::MAX && record.vars.matches_generation(ic.gen)).then_some(
            NameView::GlobalLexical {
                search,
                genv,
                generation: ic.gen,
                binding: ic.binding as usize,
            },
        );
    }
    if ic.env != genv | 1 {
        return None;
    }
    let name = chunk.jit_name(name);
    let global = realm.global.try_borrow().ok()?;
    if !global.kind.exotic_is_none() || !global.ic_plain() {
        return None;
    }
    if realm.global_env.try_borrow().ok()?.vars.contains_key(name) {
        return None;
    }
    let slot = global.props.slot_of(name)?;
    if global.props.property_at(slot)?.accessor() {
        return None;
    }
    let location = global.props.slot_location(slot)?;
    let epoch = global.props.slot_epoch();
    (epoch != u16::MAX).then(|| NameView::Global {
        search,
        genv,
        global: crate::value::Gc::as_ptr(&realm.global) as usize,
        slot: slot as u32,
        location,
        epoch,
    })
}

/// A name cache's class, for a site the tier keeps the effect at.
fn cache_view(ic: &NameIc) -> NameView {
    match ic.env {
        0 => NameView::Empty,
        crate::bytecode::REFUSED_NAME_IC => NameView::Refused,
        _ => NameView::Dynamic,
    }
}

/// The record `hops` parent links from inline target `target`'s pinned [[Environment]], the
/// holder a splice's static read folds (see [`Base::Holder`]).
fn holder(chunk: &Chunk, target: u32, hops: u16) -> Base {
    let Some(mut record) = chunk.inline_target_env(target) else {
        return Base::Unknown;
    };
    for _ in 0..hops {
        let Some(parent) = record.try_borrow().ok().and_then(|scope| scope.parent) else {
            return Base::Unknown;
        };
        record = parent;
    }
    Base::Holder(Env::as_ptr(&record) as usize)
}

/// A free-name cache's class for the fingerprint: its mode, which a fill or a refusal moves, and
/// not its proof, which a global object's every addition moves (a shape).
fn cache_class(ic: &NameIc) -> u8 {
    match ic.env {
        0 => 0,
        crate::bytecode::REFUSED_NAME_IC => 1,
        crate::bytecode::FIXED_NAME_IC => 2,
        crate::bytecode::lexical_cache::DEEP_NAME_IC => 3,
        env if env & 1 != 0 => 4,
        env if env & 2 != 0 => 5,
        _ => 6,
    }
}

/// Whether `op` reads a property named `length` (which a String receiver reads without a way).
fn is_length_read(chunk: &Chunk, op: &Op) -> bool {
    match *op {
        Op::GetProp(n, _)
        | Op::GetPropThis(n, _)
        | Op::GetPropLocal(_, n, _)
        | Op::GetMethod(n, _) => chunk.jit_name(n) == "length",
        _ => false,
    }
}

/// The feedback fingerprint the settled-feedback rule compares between a chunk's checkpoints (the
/// Stage 1 record's "When", V8's wait for a quiet IC): its named sites' ways and generic marks, its
/// call sites' ways and megamorphic states, its value and element records, read as the snapshot
/// reads them (a transformed chunk's records are its baseline sites', which its misses go on
/// filling; its own never change), and its free-name caches' classes (stage 2, step 4). A call
/// way's epoch (`CallTarget::current`) is left out: an epoch moved by another function's code
/// changes no feedback of this one. So are a generic named site's ways and a megamorphic call
/// site's (stage 2, step 11): past their capacity their refills are churn, which the snapshot
/// reads as the generic access and the megamorphic call, as V8's megamorphic IC updates only its
/// stub cache and notifies no IC change (ic/ic.cc:1061-1072), where the transition to the state
/// does (:377-390).
pub(crate) fn fingerprint(chunk: &Chunk) -> u64 {
    use std::hash::Hasher;
    let mut hash = crate::fasthash::FxHasher::default();
    let map = chunk.feedback_vector().record_map();
    for (pc, op) in chunk.jit_ops().iter().enumerate() {
        hash.write_usize(pc);
        if let Some((_, cache, _)) = name_site(op) {
            // A static address is the compiler's: its cache is the template's business.
            if chunk.jit_name_static(cache).is_none() {
                hash.write_u8(cache_class(&chunk.jit_name_ic(cache)));
            }
        }
        match view(chunk, map, pc) {
            SiteView::Named(named) => hash_named(&mut hash, &named),
            SiteView::NamedUpdate(named, values) => {
                hash_named(&mut hash, &named);
                hash_values(&mut hash, &values);
            }
            SiteView::Values(values) => hash_values(&mut hash, &values),
            SiteView::Element(element) => {
                hash.write_u32(element.recorded_receivers);
                hash.write_u32(element.recorded_keys);
                hash.write_u32(element.facts);
                hash.write_u8(u8::from(element.generic));
                hash.write_u32(element.typed_kinds);
            }
            SiteView::Call(call) => {
                hash.write_u8(call.state as u8);
                if call.state != CallState::Megamorphic {
                    for target in call.ways.iter().flatten() {
                        hash.write_usize(target.callee);
                        hash.write_usize(target.function);
                        hash.write_u32(u32::from(target.builtin));
                    }
                }
            }
            SiteView::Construct(construct) => {
                if let Some(target) = construct.target {
                    hash.write_usize(target.callee);
                }
                hash.write_u32(construct.prototype_shape);
            }
            SiteView::Allocation(_) | SiteView::None => {}
        }
    }
    hash.finish()
}

/// A named site's part of the fingerprint: its generic mark, and while it is not generic its
/// filled ways.
fn hash_named(hash: &mut impl std::hash::Hasher, named: &NamedView) {
    hash.write_u8(u8::from(named.generic));
    if named.generic {
        return;
    }
    for way in named.filled() {
        hash.write_u32(way.receiver_shape);
        hash.write_u32(way.holder_shape);
        hash.write_u8(way.primitive);
        hash.write_u32(way.slot);
        hash.write_u8(way.depth);
        hash.write_u8(way.kind as u8);
    }
}

fn hash_values(hash: &mut impl std::hash::Hasher, values: &ValuesView) {
    for role in values.recorded {
        hash.write_u32(role);
    }
}
