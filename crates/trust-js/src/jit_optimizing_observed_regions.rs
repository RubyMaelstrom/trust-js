//! Select native emission from frozen, exact-version baseline execution provenance.
//!
//! ECMA-262 e28783d5 Execution Contexts, GetValue/PutValue and TryStatement:
//! an unseen region is workload feedback, never unreachable JavaScript. Its first
//! original PC finishes the same invocation in the VM, before effects. The full
//! Chunk, CFG, frame/stack analyses and exception/completion destinations survive.
//! This opt-in emission policy neither increases admission limits nor adds OSR.

use crate::{
    bytecode::Chunk,
    execution_coverage::{Handle, State},
    jit_ir::Cfg,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Emission {
    Native,
    Cut,
    OmittedInterior,
}

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_ENABLED.with(std::cell::Cell::get) {
        return enabled;
    }
    std::env::var("TRUST_JS_OPT_JIT_OBSERVED_REGIONS").as_deref() == Ok("1")
}

/// Compile-time scalar selection only. No source feedback, JS object, environment
/// or executable owner is retained in this plan or passed to the backend worker.
pub(super) struct Plan {
    emission: Box<[Emission]>,
    cuts: Box<[bool]>,
    pub(super) source_version: u64,
    pub(super) native_ops: usize,
    pub(super) cut_blocks: usize,
}

impl Plan {
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &super::stack::StackPlan,
        frozen: &Handle,
    ) -> Option<Self> {
        if !enabled() {
            return None;
        }
        let snapshot = frozen.frozen_snapshot()?;
        // A positive entry is required even if all other observations are unknown.
        // state_for validates the whole immutable source-version relation.
        if snapshot.state_for(chunk, 0) != State::Entered {
            return None;
        }
        let mut emission = vec![Emission::Native; chunk.jit_ops().len()];
        let mut cuts = vec![false; cfg.blocks().len()];
        let mut native_ops = emission.len();
        let mut cut_blocks = 0;
        for (id, block) in cfg.blocks().iter().enumerate() {
            if stack.reachable[block.start]
                && snapshot.state_for(chunk, block.start) == State::UnseenInWindow
            {
                // Only CFG leaders are runtime exits. All branches, handler roots
                // and completion pads are leaders in this same original graph.
                emission[block.start] = Emission::Cut;
                emission[block.start + 1..block.end].fill(Emission::OmittedInterior);
                cuts[id] = true;
                native_ops -= block.end - block.start;
                cut_blocks += 1;
            }
        }
        (cut_blocks != 0).then(|| Self {
            emission: emission.into_boxed_slice(),
            cuts: cuts.into_boxed_slice(),
            source_version: snapshot.source_version,
            native_ops,
            cut_blocks,
        })
    }

    pub(super) fn emission(&self, pc: usize) -> Emission {
        self.emission[pc]
    }

    pub(super) fn cuts_block(&self, block: usize) -> bool {
        self.cuts[block]
    }

    /// Minimum saved checks before any native normal/abrupt completion or VM cut.
    /// Keep the old <=256-op/no-handler accounting bound; this does not authorize
    /// large bodies. Numeric guards are charged conservatively even when their
    /// field sites were omitted. Entry/field specialization misses remain separate
    /// outcomes, just as in existing hot accounting.
    pub(super) fn minimum_saved_tests(
        &self,
        chunk: &Chunk,
        cfg: &Cfg,
        numeric: Option<&super::specialization::Plan>,
        structural: Option<&super::generic_admission::Plan>,
        objects: Option<&super::object_facts::Plan>,
    ) -> Option<u32> {
        if chunk.jit_ops().len() > 256 {
            return None;
        }
        let mut structural_pc = vec![None; self.emission.len()];
        if let Some(plan) = structural {
            structural_pc.copy_from_slice(&plan.per_pc);
        }
        let mut numeric_pc = vec![0u32; self.emission.len()];
        if let Some(plan) = numeric {
            for opportunity in &plan.opportunities {
                if self.emission(opportunity.pc) == Emission::Native {
                    numeric_pc[opportunity.pc] =
                        numeric_pc[opportunity.pc].checked_add(opportunity.removed_tests)?;
                }
            }
        }
        let mut weights = vec![0u32; cfg.blocks().len()];
        for (id, block) in cfg.blocks().iter().enumerate() {
            for pc in block.start..block.end {
                if self.emission(pc) == Emission::Native {
                    weights[id] = weights[id]
                        .checked_add(structural_pc[pc].unwrap_or(0))?
                        .checked_add(numeric_pc[pc])?;
                } else {
                    structural_pc[pc] = None;
                }
            }
        }
        let minimum = super::generic_admission::minimum_path(
            cfg,
            chunk,
            &weights,
            &structural_pc,
            &numeric_pc,
            objects,
            Some(self),
        )?;
        let guards = numeric.map_or(0, |plan| plan.inputs.len() + plan.properties.len());
        // Structural per-PC credit is raw; do not subtract Plan::minimum_tests
        // again. Object result guards and numeric category guards are distinct
        // emitted checks, and each is charged once across the combined path.
        let guards = u32::try_from(guards)
            .ok()?
            .checked_add(objects.map_or(0, |plan| plan.guard_cost))?;
        let saved = minimum.checked_sub(guards)?;
        (saved != 0).then_some(saved)
    }
}

#[cfg(test)]
thread_local! {
    static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    // Capture calls to the actual opcode lowerer, independently of selection metadata.
    pub(super) static TEST_LOWERED: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    pub(super) static TEST_CAPTURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) struct TestControl(Option<bool>, bool);

#[cfg(test)]
impl TestControl {
    pub(super) fn new(enabled: bool) -> Self {
        TEST_LOWERED.with(|pcs| pcs.borrow_mut().clear());
        Self(
            TEST_ENABLED.with(|current| current.replace(Some(enabled))),
            TEST_CAPTURE.with(|current| current.replace(true)),
        )
    }
}

#[cfg(test)]
impl Drop for TestControl {
    fn drop(&mut self) {
        TEST_ENABLED.with(|current| current.set(self.0));
        TEST_CAPTURE.with(|current| current.set(self.1));
    }
}
