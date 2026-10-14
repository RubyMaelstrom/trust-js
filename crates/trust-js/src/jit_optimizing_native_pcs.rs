//! Sparse physical native labels with the original semantic graph intact.
//!
//! ECMA-262 e28783d5 Execution Contexts / TryStatement: an unseen block is
//! recovered at its original leader before effects. Only its certified interior
//! lacks machine labels. Every native successor and dynamic completion landing
//! remains present; the immutable Chunk and full frame/stack analyses are unchanged.

use super::{observed_regions, Cfg, CompileError, Op};

pub(super) struct Plan {
    present: Box<[bool]>,
}

impl Plan {
    pub(super) fn build(
        ops: &[Op],
        cfg: &Cfg,
        entry: usize,
        regions: Option<&observed_regions::Plan>,
    ) -> Result<Self, CompileError> {
        #[cfg(test)]
        let sparse = TEST_ENABLED.with(std::cell::Cell::get).unwrap_or(true);
        #[cfg(not(test))]
        let sparse = true;
        let mut present = vec![true; ops.len() + 1];
        if let Some(regions) = regions.filter(|_| sparse) {
            for (pc, label) in present[..ops.len()].iter_mut().enumerate() {
                if regions.emission(pc) == observed_regions::Emission::OmittedInterior {
                    // Validate the cut certificate against this original graph,
                    // independently of the absence of an emitted instruction.
                    let block = cfg.block_at(pc).ok_or(CompileError::Unsupported)?;
                    let start = cfg.blocks()[block.0 as usize].start;
                    if start == pc || regions.emission(start) != observed_regions::Emission::Cut {
                        return Err(CompileError::Unsupported);
                    }
                    *label = false;
                }
            }
        }
        Self::validate(&present, ops, cfg, entry, regions)?;
        #[cfg(test)]
        TEST_COUNTS.with(|counts| {
            counts.set([
                present.iter().filter(|&&label| label).count(),
                present.iter().filter(|&&label| !label).count(),
            ]);
        });
        Ok(Self {
            present: present.into_boxed_slice(),
        })
    }

    fn validate(
        present: &[bool],
        ops: &[Op],
        cfg: &Cfg,
        entry: usize,
        regions: Option<&observed_regions::Plan>,
    ) -> Result<(), CompileError> {
        let has = |pc| present.get(pc).copied().unwrap_or(false);
        if present.len() != ops.len() + 1 || !has(entry) || !has(ops.len()) {
            return Err(CompileError::Unsupported);
        }
        // Original leaders include ordinary branches, all PushFinally/iterator
        // modes, cleanup pads, inline fallback and abrupt-jump destinations.
        // Keep even cold or unreachable roots: a checked helper may route there.
        for block in cfg.blocks() {
            if !has(block.start)
                || block
                    .successors
                    .iter()
                    .any(|id| !has(cfg.blocks()[id.0 as usize].start))
            {
                return Err(CompileError::Unsupported);
            }
        }
        for root in cfg.handler_roots().iter().chain(cfg.resume_roots()) {
            if !has(cfg.blocks()[root.target.0 as usize].start) {
                return Err(CompileError::Unsupported);
            }
        }
        for (pc, op) in ops.iter().enumerate() {
            if crate::jit_ir::jump_target(op).is_some_and(|target| !has(target)) {
                return Err(CompileError::Unsupported);
            }
            match regions.map(|regions| regions.emission(pc)) {
                Some(observed_regions::Emission::OmittedInterior) => {}
                Some(observed_regions::Emission::Cut) if has(pc) => {}
                Some(observed_regions::Emission::Cut) => {
                    return Err(CompileError::Unsupported);
                }
                // lower() obtains next even for operations that terminate, and
                // normal checked paths must retain this exact successor label.
                _ if has(pc) && has(pc + 1) => {}
                _ => return Err(CompileError::Unsupported),
            }
        }
        Ok(())
    }

    pub(super) fn labels(&self) -> &[bool] {
        &self.present
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    // Original PC labels allocated / certified omitted interiors with no label.
    pub(super) static TEST_COUNTS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

#[cfg(test)]
pub(super) struct TestControl(Option<bool>);

#[cfg(test)]
impl TestControl {
    pub(super) fn new(enabled: bool) -> Self {
        Self(TEST_ENABLED.with(|setting| setting.replace(Some(enabled))))
    }
}

#[cfg(test)]
impl Drop for TestControl {
    fn drop(&mut self) {
        TEST_ENABLED.with(|setting| setting.set(self.0));
    }
}

#[cfg(test)]
#[path = "jit_optimizing_native_pc_tests.rs"]
mod tests;
