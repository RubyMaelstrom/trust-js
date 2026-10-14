//! Must-proofs for ordinary named-property metadata, independent of heap contents.
//!
//! A receiver names a semantic SSA value, never a physical local or a heap address.
//! Guards in this graph must exit native execution on failure. A checked helper
//! that can rejoin is an explicit observer edge; its fast successor can establish
//! facts separately, but the join cannot inherit facts from that successor alone.
//! No proof licenses caching a property value, owner count or storage pointer.
//!
//! ECMA-262 e28783d5 OrdinaryGet/OrdinarySetWithOwnDescriptor: key-only TRust shapes
//! do not prove descriptor attributes. Reads therefore need independent own-data
//! facts. Existing own-data overwrites preserve metadata; descriptor/prototype
//! mutation, unknown effects and observing helpers invalidate all facts because
//! distinct SSA receivers may alias. Catch/OSR entries supply empty fact sets.

use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Receiver(pub(super) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Metadata {
    Ordinary,
    NamedShape(u32),
    NamedSlotInBounds(u32),
    OwnData { slot: u32, writable: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Proof {
    pub(super) receiver: Receiver,
    pub(super) metadata: Metadata,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HeapEffect {
    /// Only an already guarded native own-data overwrite, including its guarded
    /// owner transfer. It neither changes metadata nor runs an observer.
    ExistingOwnDataOverwrite,
    /// Includes deletion, descriptor/prototype changes, creation/storage changes
    /// and operations with unknown alias/write effects.
    /// The production bridge conservatively treats these paths as observers because
    /// their checked helpers may enter author code; the solver tests this distinction.
    #[cfg(test)]
    MetadataMutation,
    /// Includes calls, getters, proxies, coercion and canonical helper/poll entry.
    Observer,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Event {
    Guard(Proof),
    Effect(HeapEffect),
}

#[derive(Default)]
pub(super) struct Edge {
    pub(super) target: usize,
    /// (target phi receiver, incoming receiver). These are exact SSA phi inputs,
    /// not guessed equivalence between locals or observations of equal shapes.
    pub(super) aliases: Vec<(Receiver, Receiver)>,
    /// Assertions established only on this native fast-success edge. A generic
    /// fallback that rejoins must be modeled as a separate observer successor.
    pub(super) proven: Vec<Proof>,
}

#[derive(Default)]
pub(super) struct Block {
    pub(super) events: Vec<Event>,
    pub(super) successors: Vec<Edge>,
}

pub(super) struct Graph {
    pub(super) blocks: Vec<Block>,
    /// Function, catch and OSR entries; each has an independent empty input.
    pub(super) roots: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(super) struct Budget {
    pub(super) blocks: usize,
    pub(super) events: usize,
    pub(super) facts_per_block: usize,
    pub(super) work: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            blocks: 128,
            events: 4096,
            facts_per_block: 64,
            work: 262_144,
        }
    }
}

pub(super) struct Plan {
    redundant: Vec<Vec<bool>>,
}

impl Plan {
    pub(super) fn redundant(&self, block: usize, event: usize) -> bool {
        self.redundant
            .get(block)
            .and_then(|events| events.get(event))
            .copied()
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Facts(Vec<Proof>);

impl Facts {
    fn contains(&self, proof: Proof) -> bool {
        self.0.binary_search(&proof).is_ok()
    }

    fn insert_one(&mut self, proof: Proof, limit: usize) {
        if let Err(index) = self.0.binary_search(&proof) {
            if self.0.len() < limit {
                self.0.insert(index, proof);
            }
        }
    }

    fn insert(&mut self, proof: Proof, limit: usize) {
        // A writable own-data descriptor also proves the read-only check. Keep
        // that weaker fact explicit so intersections preserve it on joins.
        if let Metadata::OwnData {
            slot,
            writable: true,
        } = proof.metadata
        {
            self.insert_one(
                Proof {
                    metadata: Metadata::OwnData {
                        slot,
                        writable: false,
                    },
                    ..proof
                },
                limit,
            );
        }
        self.insert_one(proof, limit);
    }

    fn effect(&mut self, effect: HeapEffect) {
        if effect != HeapEffect::ExistingOwnDataOverwrite {
            self.0.clear();
        }
    }

    fn edge(&self, edge: &Edge, limit: usize) -> Self {
        let mut result = self.clone();
        // Phi aliases are simultaneous: derive every one from the incoming
        // state, and first erase stale facts for the target phi identities.
        result.0.retain(|proof| {
            !edge
                .aliases
                .iter()
                .any(|(target, _)| *target == proof.receiver)
        });
        for &(target, incoming) in &edge.aliases {
            for proof in &self.0 {
                if proof.receiver == incoming {
                    result.insert(
                        Proof {
                            receiver: target,
                            ..*proof
                        },
                        limit,
                    );
                }
            }
        }
        for &proof in &edge.proven {
            result.insert(proof, limit);
        }
        result
    }
}

fn spend(work: &mut usize, amount: usize) -> Option<()> {
    *work = work.checked_sub(amount)?;
    Some(())
}

/// Solve before choosing any elision. Unreached predecessors have no executable
/// edge yet; revisiting after their first output removes any optimistic facts.
/// Exhaustion/malformed graphs disable the whole plan, retaining every guard.
pub(super) fn analyze(graph: &Graph, budget: Budget) -> Option<Plan> {
    let count = graph.blocks.len();
    if count == 0
        || count > budget.blocks
        || graph.roots.is_empty()
        || graph.roots.len() > budget.blocks
    {
        return None;
    }
    let mut work = budget.work;
    let mut size = 0usize;
    let mut incoming = vec![Vec::new(); count];
    for (source, block) in graph.blocks.iter().enumerate() {
        size = size.checked_add(block.events.len())?;
        if size > budget.events {
            return None;
        }
        spend(&mut work, block.events.len())?;
        for (edge_index, edge) in block.successors.iter().enumerate() {
            size = size.checked_add(1 + edge.aliases.len() + edge.proven.len())?;
            if size > budget.events {
                return None;
            }
            spend(
                &mut work,
                1usize.checked_add(edge.aliases.len().checked_mul(edge.aliases.len())?)?,
            )?;
            incoming.get_mut(edge.target)?.push((source, edge_index));
            // A phi has one incoming value per edge. Conflicting entries could
            // otherwise manufacture a fact from only one possible receiver.
            for (index, (target, _)) in edge.aliases.iter().enumerate() {
                if edge.aliases[..index]
                    .iter()
                    .any(|(other, _)| other == target)
                {
                    return None;
                }
            }
        }
    }
    // Every phi mentioned by a predecessor must have an explicit incoming
    // mapping on all predecessor edges, including an unchanged self mapping.
    // Otherwise a loop could retain the previous iteration's receiver fact.
    for predecessors in &incoming {
        let mut targets = Vec::new();
        for &(source, edge_index) in predecessors {
            for &(target, _) in &graph.blocks[source].successors[edge_index].aliases {
                spend(&mut work, 1 + targets.len())?;
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
        }
        for &(source, edge_index) in predecessors {
            let aliases = &graph.blocks[source].successors[edge_index].aliases;
            spend(&mut work, targets.len().checked_mul(aliases.len().max(1))?)?;
            if targets
                .iter()
                .any(|target| !aliases.iter().any(|(present, _)| present == target))
            {
                return None;
            }
        }
    }
    let mut roots = vec![false; count];
    let mut queued = vec![false; count];
    let mut pending = VecDeque::new();
    for &root in &graph.roots {
        *roots.get_mut(root)? = true;
        if !std::mem::replace(&mut queued[root], true) {
            pending.push_back(root);
        }
    }
    let mut entries: Vec<Option<Facts>> = vec![None; count];
    let mut outputs: Vec<Option<Facts>> = vec![None; count];
    while let Some(block) = pending.pop_front() {
        queued[block] = false;
        spend(&mut work, 1)?;
        let mut facts = roots[block].then(Facts::default);
        for &(source, edge_index) in &incoming[block] {
            if let Some(output) = &outputs[source] {
                let edge = &graph.blocks[source].successors[edge_index];
                let operations = output
                    .0
                    .len()
                    .checked_mul(1 + edge.aliases.len())?
                    .checked_add(edge.aliases.len())?
                    .checked_add(edge.proven.len())?;
                let cost = operations
                    .checked_mul(budget.facts_per_block.max(1))?
                    .checked_add(budget.facts_per_block)?
                    .checked_add(1)?;
                spend(&mut work, cost)?;
                let other = output.edge(edge, budget.facts_per_block);
                if let Some(current) = &mut facts {
                    current.0.retain(|proof| other.contains(*proof));
                } else {
                    facts = Some(other);
                }
            }
        }
        let Some(mut facts) = facts else { continue };
        entries[block] = Some(facts.clone());
        for event in &graph.blocks[block].events {
            spend(&mut work, 1 + facts.0.len())?;
            match *event {
                Event::Guard(proof) => facts.insert(proof, budget.facts_per_block),
                Event::Effect(effect) => facts.effect(effect),
            }
        }
        if outputs[block].as_ref() != Some(&facts) {
            outputs[block] = Some(facts);
            for edge in &graph.blocks[block].successors {
                if !std::mem::replace(&mut queued[edge.target], true) {
                    pending.push_back(edge.target);
                }
            }
        }
    }
    let mut redundant = Vec::with_capacity(count);
    for (block, entry) in graph.blocks.iter().zip(entries) {
        let mut decisions = vec![false; block.events.len()];
        if let Some(mut facts) = entry {
            for (index, event) in block.events.iter().enumerate() {
                spend(&mut work, 1 + facts.0.len())?;
                match *event {
                    Event::Guard(proof) => {
                        decisions[index] = facts.contains(proof);
                        facts.insert(proof, budget.facts_per_block);
                    }
                    Event::Effect(effect) => facts.effect(effect),
                }
            }
        }
        redundant.push(decisions);
    }
    Some(Plan { redundant })
}

#[derive(Clone, Copy, Default)]
pub(super) struct ReadElisions {
    pub(super) ordinary: bool,
    pub(super) shape: bool,
    pub(super) bounds: bool,
    pub(super) data: bool,
}

pub(super) struct MetadataPlan {
    reads: Vec<Option<(crate::bytecode::IcState, ReadElisions)>>,
}

impl MetadataPlan {
    /// Emit the exact scalar cache snapshot used by the solver. Per-site
    /// substitution of a newer cache state could invalidate downstream proofs.
    /// IcState carries shape/slot numbers only, never a JavaScript object owner.
    pub(super) fn read(&self, pc: usize) -> Option<(crate::bytecode::IcState, ReadElisions)> {
        self.reads.get(pc).copied().flatten()
    }

    /// Initial bridge uses only never-written private receiver slots and This.
    /// That restricted set has stable semantic identities across the whole CFG;
    /// reassigned slots receive no optimization, rather than guessed SSA origins.
    pub(super) fn build(
        chunk: &crate::bytecode::Chunk,
        cfg: &crate::jit_ir::Cfg,
        frame: &super::frame::FramePlan,
        stack: &super::stack::StackPlan,
        values: Option<&super::value_facts::ValuePlan>,
        properties: &[(u32, super::value_facts::Types)],
    ) -> Option<Self> {
        use crate::bytecode::Op;
        if std::env::var("TRUST_JS_OPT_JIT_HEAP_PROOFS").as_deref() == Ok("0")
            || properties.is_empty()
            || cfg.blocks().len() > Budget::default().blocks
        {
            return None;
        }
        let ops = chunk.jit_ops();
        let immutable: Vec<_> = frame
            .tracked
            .iter()
            .copied()
            .filter(|&slot| {
                !ops.iter().any(|op| {
                    // Completion routing can replace canonical frame words,
                    // but cannot continue along an ordinary CFG edge. Its
                    // cleanup/catch destinations are independent empty roots.
                    !matches!(
                        op,
                        Op::Return | Op::ReturnBare | Op::ReturnUndef | Op::Throw
                    ) && super::effects::checked_writes(op).contains(slot)
                })
            })
            .collect();
        let mut poll = vec![false; ops.len()];
        for (pc, op) in ops.iter().enumerate() {
            if let Some(target) = crate::jit_ir::jump_target(op).filter(|&target| target <= pc) {
                poll[target] = true;
            }
        }
        let mut graph = Graph {
            blocks: Vec::new(),
            roots: vec![cfg.block_at(0)?.0 as usize],
        };
        for root in cfg.handler_roots().iter().chain(cfg.resume_roots()) {
            graph.roots.push(root.target.0 as usize);
        }
        for op in ops {
            if let Op::AbruptJump(target, _) = op {
                graph.roots.push(cfg.block_at(*target as usize)?.0 as usize);
            }
        }
        graph.roots.sort_unstable();
        graph.roots.dedup();
        let mut sites = Vec::new();
        for (block_index, block) in cfg.blocks().iter().enumerate() {
            let mut events = Vec::new();
            for pc in block.start..block.end {
                // Lowering::poll has a conditional observing helper that rejoins.
                // Clear at its PC even when the usual runtime edge is the fast one.
                if poll[pc] {
                    events.push(Event::Effect(HeapEffect::Observer));
                }
                let receiver_cache = match ops[pc] {
                    Op::GetPropLocal(slot, _, cache) if immutable.contains(&slot) => {
                        Some((Receiver(u32::from(slot)), cache))
                    }
                    Op::GetPropThis(_, cache) => Some((Receiver(u32::MAX), cache)),
                    _ => None,
                };
                let numeric = properties
                    .iter()
                    .any(|&(site, ty)| site as usize == pc && ty.is_number());
                if let Some((receiver, state)) =
                    receiver_cache
                        .filter(|_| numeric)
                        .and_then(|(receiver, cache)| {
                            chunk
                                .jit_cache_preferred(cache)
                                .filter(|state| state.depth == 0)
                                .map(|state| (receiver, state))
                        })
                {
                    let first = events.len();
                    for metadata in [
                        Metadata::Ordinary,
                        Metadata::NamedShape(state.recv_shape),
                        Metadata::NamedSlotInBounds(state.slot),
                        Metadata::OwnData {
                            slot: state.slot,
                            writable: false,
                        },
                    ] {
                        events.push(Event::Guard(Proof { receiver, metadata }));
                    }
                    sites.push((pc, block_index, first, state));
                } else if !preserves_metadata(chunk, pc, stack, values) {
                    // Generic reads/writes/owner releases can reach a checked helper.
                    // The private-slot write table cannot prove heap purity.
                    events.push(Event::Effect(HeapEffect::Observer));
                }
            }
            graph.blocks.push(Block {
                events,
                successors: block
                    .successors
                    .iter()
                    .map(|target| Edge {
                        target: target.0 as usize,
                        ..Edge::default()
                    })
                    .collect(),
            });
        }
        let plan = analyze(&graph, Budget::default())?;
        let mut reads = vec![None; ops.len()];
        for (pc, block, event, state) in sites {
            reads[pc] = Some((
                state,
                ReadElisions {
                    ordinary: plan.redundant(block, event),
                    shape: plan.redundant(block, event + 1),
                    bounds: plan.redundant(block, event + 2),
                    data: plan.redundant(block, event + 3),
                },
            ));
        }
        Some(Self { reads })
    }
}

/// Describes the fast AND slow paths selected by this backend, not generic
/// opcode purity. Only proven immediate/Number operations cannot reach an
/// observing fallback. Unknown operations conservatively invalidate all facts.
fn preserves_metadata(
    chunk: &crate::bytecode::Chunk,
    pc: usize,
    stack: &super::stack::StackPlan,
    values: Option<&super::value_facts::ValuePlan>,
) -> bool {
    use crate::bytecode::Op;
    let inputs = values
        .and_then(|values| values.at.get(pc))
        .copied()
        .unwrap_or_default();
    let known = stack.at[pc].known();
    match chunk.jit_ops()[pc] {
        Op::Const(index) => chunk.jit_const_copyable(index),
        Op::Undef | Op::Jump(_) => true,
        Op::LoadLocal(_) => inputs.local.is_copyable() && inputs.local.initialized(),
        Op::Dup | Op::Pop => known >= 1 && inputs.top.is_copyable(),
        Op::StoreLocal(_) => known >= 1 && inputs.top.is_copyable() && inputs.local.is_copyable(),
        Op::UpdateLocal(..) => inputs.local.is_number(),
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::UShr
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::StrictEq
        | Op::StrictNotEq => known >= 2 && inputs.top.is_number() && inputs.second.is_number(),
        Op::JumpIfFalse(_) | Op::JumpIfFalsePeek(_) | Op::JumpIfTruePeek(_) => {
            known >= 1 && (inputs.top.is_number() || inputs.top.is_boolean())
        }
        Op::JumpIfNotNullishPeek(_) => known >= 1,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proof(receiver: u32) -> Proof {
        Proof {
            receiver: Receiver(receiver),
            metadata: Metadata::NamedShape(7),
        }
    }
    fn edge(target: usize) -> Edge {
        Edge {
            target,
            ..Edge::default()
        }
    }
    fn guard(receiver: u32) -> Event {
        Event::Guard(proof(receiver))
    }
    fn solve(blocks: Vec<Block>, roots: Vec<usize>) -> Plan {
        analyze(&Graph { blocks, roots }, Budget::default()).expect("bounded graph")
    }

    #[test]
    fn own_value_writes_preserve_metadata_but_descriptor_mutation_does_not() {
        let writable = Proof {
            receiver: Receiver(0),
            metadata: Metadata::OwnData {
                slot: 2,
                writable: true,
            },
        };
        let readable = Proof {
            metadata: Metadata::OwnData {
                slot: 2,
                writable: false,
            },
            ..writable
        };
        let plan = solve(
            vec![Block {
                events: vec![
                    Event::Guard(writable),
                    Event::Effect(HeapEffect::ExistingOwnDataOverwrite),
                    Event::Guard(readable),
                    Event::Effect(HeapEffect::MetadataMutation),
                    Event::Guard(readable),
                ],
                ..Block::default()
            }],
            vec![0],
        );
        assert!(!plan.redundant(0, 0));
        assert!(plan.redundant(0, 2));
        assert!(!plan.redundant(0, 4));
    }

    #[test]
    fn diamond_requires_every_executable_predecessor() {
        for right_guarded in [false, true] {
            let plan = solve(
                vec![
                    Block {
                        successors: vec![edge(1), edge(2)],
                        ..Block::default()
                    },
                    Block {
                        events: vec![guard(0)],
                        successors: vec![edge(3)],
                    },
                    Block {
                        events: if right_guarded {
                            vec![guard(0)]
                        } else {
                            vec![]
                        },
                        successors: vec![edge(3)],
                    },
                    Block {
                        events: vec![guard(0)],
                        ..Block::default()
                    },
                ],
                vec![0],
            );
            assert_eq!(plan.redundant(3, 0), right_guarded);
        }
    }

    #[test]
    fn fast_success_does_not_hide_a_rejoining_observing_helper() {
        for slow_rejoins in [false, true] {
            let plan = solve(
                vec![
                    Block {
                        successors: vec![
                            Edge {
                                target: 1,
                                proven: vec![proof(0)],
                                ..Edge::default()
                            },
                            edge(2),
                        ],
                        ..Block::default()
                    },
                    Block {
                        events: vec![guard(0)],
                        successors: vec![edge(3)],
                    },
                    Block {
                        events: vec![Event::Effect(HeapEffect::Observer)],
                        successors: if slow_rejoins { vec![edge(3)] } else { vec![] },
                    },
                    Block {
                        events: vec![guard(0)],
                        ..Block::default()
                    },
                ],
                vec![0],
            );
            assert!(plan.redundant(1, 0));
            assert_eq!(plan.redundant(3, 0), !slow_rejoins);
        }
    }

    #[test]
    fn loop_backedges_can_remove_optimistic_entry_facts() {
        for effect in [
            HeapEffect::ExistingOwnDataOverwrite,
            HeapEffect::MetadataMutation,
            HeapEffect::Observer,
        ] {
            let plan = solve(
                vec![
                    Block {
                        events: vec![guard(0)],
                        successors: vec![edge(1)],
                    },
                    Block {
                        events: vec![guard(0), guard(0)],
                        successors: vec![edge(2)],
                    },
                    Block {
                        events: vec![Event::Effect(effect)],
                        successors: vec![edge(1)],
                    },
                ],
                vec![0],
            );
            assert_eq!(
                plan.redundant(1, 0),
                effect == HeapEffect::ExistingOwnDataOverwrite
            );
            assert!(plan.redundant(1, 1));
        }
    }

    #[test]
    fn catch_and_osr_entries_do_not_inherit_preheader_checks() {
        let plan = solve(
            vec![
                Block {
                    events: vec![guard(0)],
                    successors: vec![edge(1), edge(2)],
                },
                Block {
                    events: vec![guard(0), guard(0)],
                    successors: vec![edge(1)],
                },
                Block {
                    events: vec![guard(0), guard(0)],
                    ..Block::default()
                },
            ],
            vec![0, 1, 2],
        );
        for block in [1, 2] {
            assert!(!plan.redundant(block, 0));
            assert!(plan.redundant(block, 1));
        }
    }

    #[test]
    fn phi_facts_follow_actual_receiver_inputs_and_reassignment() {
        for second_guarded in [false, true] {
            let mut left = edge(3);
            left.aliases.push((Receiver(2), Receiver(0)));
            let mut right = edge(3);
            right.aliases.push((Receiver(2), Receiver(1)));
            let plan = solve(
                vec![
                    Block {
                        events: vec![guard(0)],
                        successors: vec![edge(1), edge(2)],
                    },
                    Block {
                        successors: vec![left],
                        ..Block::default()
                    },
                    Block {
                        events: if second_guarded {
                            vec![guard(1)]
                        } else {
                            vec![]
                        },
                        successors: vec![right],
                    },
                    Block {
                        events: vec![guard(2), guard(3)],
                        ..Block::default()
                    },
                ],
                vec![0],
            );
            assert_eq!(plan.redundant(3, 0), second_guarded);
            assert!(!plan.redundant(3, 1));
        }
    }

    #[test]
    fn budget_exhaustion_and_malformed_phi_keep_complete_fallback() {
        let mut graph = Graph {
            blocks: vec![Block {
                events: vec![guard(0), guard(0)],
                successors: vec![edge(0)],
            }],
            roots: vec![0],
        };
        assert!(analyze(
            &graph,
            Budget {
                work: 1,
                ..Budget::default()
            }
        )
        .is_none());
        graph.blocks[0].successors[0].aliases =
            vec![(Receiver(2), Receiver(0)), (Receiver(2), Receiver(1))];
        assert!(analyze(&graph, Budget::default()).is_none());
    }

    #[test]
    fn missing_backedge_phi_mapping_cannot_reuse_a_previous_receiver() {
        let graph = Graph {
            blocks: vec![
                Block {
                    events: vec![guard(0)],
                    successors: vec![Edge {
                        target: 1,
                        aliases: vec![(Receiver(2), Receiver(0))],
                        ..Edge::default()
                    }],
                },
                Block {
                    events: vec![guard(2)],
                    successors: vec![edge(2)],
                },
                Block {
                    successors: vec![edge(1)],
                    ..Block::default()
                },
            ],
            roots: vec![0],
        };
        assert!(analyze(&graph, Budget::default()).is_none());
    }

    #[test]
    fn receiver_phi_swaps_are_simultaneous() {
        let seven = proof(0);
        let eight = Proof {
            receiver: Receiver(1),
            metadata: Metadata::NamedShape(8),
        };
        let plan = solve(
            vec![
                Block {
                    events: vec![Event::Guard(seven), Event::Guard(eight)],
                    successors: vec![Edge {
                        target: 1,
                        aliases: vec![(Receiver(0), Receiver(1)), (Receiver(1), Receiver(0))],
                        ..Edge::default()
                    }],
                },
                Block {
                    events: vec![
                        Event::Guard(Proof {
                            receiver: Receiver(0),
                            ..eight
                        }),
                        Event::Guard(Proof {
                            receiver: Receiver(1),
                            ..seven
                        }),
                    ],
                    ..Block::default()
                },
            ],
            vec![0],
        );
        assert!(plan.redundant(1, 0));
        assert!(plan.redundant(1, 1));
    }
}
