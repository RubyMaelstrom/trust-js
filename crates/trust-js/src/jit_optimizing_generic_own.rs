//! Experimental ordinary own-data metadata proofs for arbitrary packed Values.
//!
//! Every designated read/write guard must terminate native execution on a miss,
//! before that opcode. Generic results still acquire their real packed owners;
//! stores retain the existing alias-aware transaction. No value, owner count or
//! storage pointer is cached. All CFG blocks start empty, and every unselected
//! helper, local write, poll and cleanup boundary clears the facts. A private
//! slot therefore identifies one unchanged owning word only within that span.
//! ECMA-262 e28783d5: OrdinaryGet/OrdinarySetWithOwnDescriptor and WeakRef liveness.

use super::{frame, heap_proofs, stack};
use crate::{bytecode::Chunk, bytecode::IcState, bytecode::Op, jit_ir::Cfg};
use heap_proofs::{Block, Budget, Event, Graph, HeapEffect, Metadata, Proof, Receiver};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(super) static TEST_SITES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    pub(super) static TEST_ELISIONS: std::cell::Cell<[usize; 8]> = const { std::cell::Cell::new([0; 8]) };
}

#[cfg(test)]
pub(super) fn record(write: bool, removed: heap_proofs::ReadElisions) {
    TEST_SITES.with(|cell| {
        let mut counts = cell.get();
        counts[usize::from(write)] += 1;
        cell.set(counts);
    });
    TEST_ELISIONS.with(|cell| {
        let mut counts = cell.get();
        for (index, removed) in [
            removed.ordinary,
            removed.shape,
            removed.bounds,
            removed.data,
        ]
        .into_iter()
        .enumerate()
        {
            counts[usize::from(write) * 4 + index] += usize::from(removed);
        }
        cell.set(counts);
    });
}

fn requested() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_ENABLED.with(std::cell::Cell::get) {
        return enabled;
    }
    std::env::var("TRUST_JS_OPT_JIT_GENERIC_OWN").as_deref() == Ok("1")
}

fn push_event(events: &mut Vec<Event>, count: &mut usize, event: Event) -> Option<()> {
    // Bound construction before growing a Vec, not just the eventual solver.
    if *count >= Budget::default().events {
        return None;
    }
    *count += 1;
    events.push(event);
    Some(())
}

fn same_state(first: IcState, second: IcState) -> bool {
    // Compare initialized scalar fields, never repr(C) padding bytes.
    let fields = |state: IcState| {
        (
            state.recv_shape,
            state.holder_shape,
            state.slot,
            state.depth,
            state.mid_ok,
            state.mid_shape,
            state.mid2_shape,
            state.mid3_shape,
            state.mid4_shape,
        )
    };
    fields(first) == fields(second)
}

fn candidate(
    chunk: &Chunk,
    frame: &frame::FramePlan,
    stack: &stack::StackPlan,
    pc: usize,
) -> Option<(Receiver, u32, u32, bool)> {
    let result = match *chunk.jit_ops().get(pc)? {
        Op::GetPropLocal(slot, name, cache) if frame.tracked.contains(&slot) => {
            (Receiver(u32::from(slot)), name, cache, false)
        }
        Op::SetPropLocalDrop(slot, name, cache)
            if frame.tracked.contains(&slot) && stack.at[pc].known() >= 1 =>
        {
            (Receiver(u32::from(slot)), name, cache, true)
        }
        Op::GetPropThis(name, cache) => (Receiver(u32::MAX), name, cache, false),
        Op::SetPropThisDrop(name, cache) if stack.at[pc].known() >= 1 => {
            (Receiver(u32::MAX), name, cache, true)
        }
        _ => return None,
    };
    crate::value::canonical_index(chunk.jit_name(result.1))
        .is_none()
        .then_some(result)
}

#[derive(Clone, Copy)]
pub(super) struct Site {
    /// The same scalar cache snapshot used to derive the proofs, not a later IC.
    pub(super) state: IcState,
    pub(super) removed: heap_proofs::ReadElisions,
    write: bool,
    /// All guards in this complete observer-free span are retained together when
    /// any of them supports a real later metadata elision. Zero-reuse spans do
    /// not justify turning their independent property operations into exits.
    pub(super) reuse_span_profitable: bool,
}

pub(super) struct Plan {
    sites: Vec<Option<Site>>,
}

impl Plan {
    pub(super) fn read(&self, pc: usize) -> Option<Site> {
        self.sites
            .get(pc)
            .copied()
            .flatten()
            .filter(|site| !site.write)
    }

    pub(super) fn write(&self, pc: usize) -> Option<Site> {
        self.sites
            .get(pc)
            .copied()
            .flatten()
            .filter(|site| site.write)
    }

    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        frame: &frame::FramePlan,
        stack: &stack::StackPlan,
    ) -> Option<Self> {
        if chunk.optimizing_candidate {
            return None;
        }
        Self::build_selected(chunk, cfg, frame, stack, None, None)
    }

    /// Compilation consumes exactly admission's frozen scalar selection, after
    /// the caller has constructed its canonical optimizer candidate.
    #[cfg(test)]
    pub(super) fn build_hot(
        chunk: &Chunk,
        cfg: &Cfg,
        frame: &frame::FramePlan,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
    ) -> Option<Self> {
        chunk
            .optimizing_candidate
            .then(|| Self::build_hot_selection(chunk, cfg, frame, stack, selected))
            .flatten()
    }

    /// Admission can analyze its still-canonical baseline body with precisely
    /// the same selection that compilation later uses. Validate the entire set
    /// before graph allocation: dropping a stale site or choosing a newer way
    /// would disconnect emitted guards from admission's factual removed work.
    #[cfg(test)]
    pub(super) fn build_hot_selection(
        chunk: &Chunk,
        cfg: &Cfg,
        frame: &frame::FramePlan,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
    ) -> Option<Self> {
        Self::build_hot_selection_with_regions(chunk, cfg, frame, stack, selected, None)
    }

    pub(super) fn build_hot_selection_with_regions(
        chunk: &Chunk,
        cfg: &Cfg,
        frame: &frame::FramePlan,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
        regions: Option<&super::observed_regions::Plan>,
    ) -> Option<Self> {
        if !requested() || selected.is_empty() || selected.len() > 32 {
            return None;
        }
        let mut previous = None;
        for &(pc, state) in selected {
            if previous.is_some_and(|previous| pc <= previous)
                || state.depth != 0
                || !state.has_cacheable_shapes()
            {
                return None;
            }
            let (_, _, cache, _) = candidate(chunk, frame, stack, pc as usize)?;
            if !same_state(chunk.jit_cache_preferred(cache)?, state) {
                return None;
            }
            previous = Some(pc);
        }
        Self::build_selected(chunk, cfg, frame, stack, Some(selected), regions)
    }

    fn build_selected(
        chunk: &Chunk,
        cfg: &Cfg,
        frame: &frame::FramePlan,
        stack: &stack::StackPlan,
        selected: Option<&[(u32, IcState)]>,
        regions: Option<&super::observed_regions::Plan>,
    ) -> Option<Self> {
        // The compiler's borrowed Script completion home distinguishes Script
        // entries from ordinary function frames that resume_before can finish.
        let large = super::large_regions::certificate(chunk, cfg, regions);
        if !requested()
            || cfg.blocks().len()
                > if large {
                    super::large_regions::MAX_BLOCKS
                } else {
                    Budget::default().blocks
                }
            || chunk.jit_ops().len() > Budget::default().events
            || !crate::bytecode::native_deopt::supported(chunk)
            || chunk
                .jit_slot_names()
                .iter()
                .any(|name| &**name == "%script-completion%")
            || std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0")
            || std::env::var("TRUST_JS_OPT_JIT_COMPACT_STORES").as_deref() == Ok("0")
        {
            return None;
        }
        let ops = chunk.jit_ops();
        let mut poll = vec![false; ops.len()];
        for (pc, op) in ops.iter().enumerate() {
            if let Some(target) = crate::jit_ir::jump_target(op).filter(|&target| target <= pc) {
                poll[target] = true;
            }
        }
        let mut graph = Graph {
            blocks: Vec::with_capacity(cfg.blocks().len()),
            // No CFG edge imports a receiver identity, including a catch, OSR,
            // loop backedge or inline fallback. Every block is an empty root.
            roots: (0..cfg.blocks().len()).collect(),
        };
        let mut sites = Vec::new();
        let mut event_count = 0;
        let mut span = 0usize;
        for (block_index, block) in cfg.blocks().iter().enumerate() {
            span = span.checked_add(1)?;
            let mut events = Vec::new();
            for pc in block.start..block.end {
                if regions.is_some_and(|plan| {
                    plan.emission(pc) != super::observed_regions::Emission::Native
                }) {
                    continue;
                }
                if poll[pc] {
                    span = span.checked_add(1)?;
                    push_event(
                        &mut events,
                        &mut event_count,
                        Event::Effect(HeapEffect::Observer),
                    )?;
                }
                let candidate =
                    candidate(chunk, frame, stack, pc).and_then(|(receiver, _, cache, write)| {
                        // Named ordinary storage only. Indexed/array/creation and
                        // prototype sites retain their original complete lowering.
                        let state = if let Some(selected) = selected {
                            selected
                                .binary_search_by_key(&(pc as u32), |&(pc, _)| pc)
                                .ok()
                                .map(|index| selected[index].1)
                        } else {
                            chunk
                                .jit_cache_preferred(cache)
                                .filter(|state| state.depth == 0)
                        }?;
                        Some((receiver, state, write))
                    });
                if let Some((receiver, state, write)) = candidate {
                    let first = events.len();
                    for metadata in [
                        Metadata::Ordinary,
                        Metadata::NamedShape(state.recv_shape),
                        Metadata::NamedSlotInBounds(state.slot),
                        Metadata::OwnData {
                            slot: state.slot,
                            writable: write,
                        },
                    ] {
                        push_event(
                            &mut events,
                            &mut event_count,
                            Event::Guard(Proof { receiver, metadata }),
                        )?;
                    }
                    sites.push((pc, block_index, first, state, write, span));
                    if write {
                        // This event describes ONLY the selected terminal-miss
                        // path's successful existing-data overwrite. Values and
                        // alias contents are not facts in this analysis.
                        push_event(
                            &mut events,
                            &mut event_count,
                            Event::Effect(HeapEffect::ExistingOwnDataOverwrite),
                        )?;
                    }
                } else if !matches!(ops[pc], Op::Undef)
                    && !matches!(ops[pc], Op::Const(index) if chunk.jit_const_copyable(index))
                {
                    // Deliberately do not infer heap purity from checked_writes.
                    // Even an owner operation may take a rejoining destructor
                    // helper; a local rebind changes this span's receiver word.
                    span = span.checked_add(1)?;
                    push_event(
                        &mut events,
                        &mut event_count,
                        Event::Effect(HeapEffect::Observer),
                    )?;
                }
            }
            graph.blocks.push(Block {
                events,
                successors: Vec::new(),
            });
        }
        if sites.is_empty() {
            return None;
        }
        let solved = if large {
            heap_proofs::analyze_independent(
                &graph,
                Budget {
                    blocks: super::large_regions::MAX_BLOCKS,
                    ..Budget::default()
                },
            )?
        } else {
            heap_proofs::analyze(&graph, Budget::default())?
        };
        let mut result = vec![None; ops.len()];
        // Sites are emitted in PC order, with monotonically increasing span IDs.
        // Use the exact purity/observer events above rather than a second opcode
        // whitelist. Retain every producer within a useful span, including its
        // first guard with zero immediate credit; remove only complete zero spans.
        let mut first_site = 0;
        while first_site < sites.len() {
            let span = sites[first_site].5;
            let mut end = first_site + 1;
            while end < sites.len() && sites[end].5 == span {
                end += 1;
            }
            let useful = sites[first_site..end]
                .iter()
                .any(|&(_, block, first, _, _, _)| {
                    (0..4).any(|offset| solved.redundant(block, first + offset))
                });
            for &(pc, block, first, state, write, _) in &sites[first_site..end] {
                result[pc] = Some(Site {
                    state,
                    removed: heap_proofs::ReadElisions {
                        ordinary: solved.redundant(block, first),
                        shape: solved.redundant(block, first + 1),
                        bounds: solved.redundant(block, first + 2),
                        data: solved.redundant(block, first + 3),
                    },
                    write,
                    reuse_span_profitable: useful,
                });
            }
            first_site = end;
        }
        if selected.is_some_and(|selected| {
            selected
                .iter()
                .any(|&(pc, _)| result.get(pc as usize).is_none_or(Option::is_none))
        }) {
            return None;
        }
        Some(Self { sites: result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writable_own_data_implies_readable_but_readable_does_not_imply_writable() {
        for (first, second, redundant) in [(true, false, true), (false, true, false)] {
            let proof = |writable| Proof {
                receiver: Receiver(0),
                metadata: Metadata::OwnData { slot: 1, writable },
            };
            let graph = Graph {
                blocks: vec![Block {
                    events: vec![Event::Guard(proof(first)), Event::Guard(proof(second))],
                    successors: Vec::new(),
                }],
                roots: vec![0],
            };
            let plan = heap_proofs::analyze(&graph, Budget::default()).unwrap();
            assert_eq!(plan.redundant(0, 1), redundant);
        }
    }

    #[test]
    fn construction_event_budget_is_checked_before_the_next_allocation() {
        let mut events = Vec::new();
        let mut count = Budget::default().events - 1;
        push_event(&mut events, &mut count, Event::Effect(HeapEffect::Observer)).unwrap();
        assert_eq!(events.len(), 1);
        assert!(push_event(&mut events, &mut count, Event::Effect(HeapEffect::Observer)).is_none());
        assert_eq!(events.len(), 1);
        assert_eq!(count, Budget::default().events);
    }
}
