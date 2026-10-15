//! Physical native reachability over certified lowering, not JavaScript reachability.
//!
//! ECMA-262 e28783d5 Completion Records / TryStatement: before-effect cuts finish
//! the same invocation in the VM. They never rejoin this native graph. Keep the
//! immutable Chunk and full semantic CFG/stack/frame tables for that continuation.
//! Fused operations have exact original-PC successors; only their independent
//! labels disappear. Dynamic completion entries remain independent native roots.

use super::{
    borrowed_operands, observed_regions, stack, string_dispatch, string_literals, Cfg,
    CompileError, Op,
};

/// Full-source scans, ownership/edge/root maps and bounded worklist traversal,
/// including declined certificates and the bounded dynamic-root ordered set.
/// Price preparation without taking semantic credit for labels. The coefficient is a conservative
/// work estimate, not a measured instruction count or a lowered IR allowance.
pub(super) fn analysis_work(ops: usize) -> Option<u64> {
    u64::try_from(ops.checked_add(1)?).ok()?.checked_mul(64)
}

#[derive(Clone, Copy, Default)]
pub(super) struct Selected<'a> {
    pub(super) borrowed: Option<&'a borrowed_operands::Plan>,
    pub(super) dispatch: Option<&'a string_dispatch::Plan>,
    pub(super) literals: Option<&'a string_literals::Plan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Kind {
    Ordinary,
    Borrowed,
    Dispatch,
    Literal,
    Interior,
}

pub(super) struct Plan {
    present: Box<[bool]>,
    reachable: Box<[bool]>,
    kinds: Box<[Kind]>,
    sparse: bool,
}

/// The very same whitelist drives both preflight and generated unwind dispatch.
/// Resume roots also remain labels, but retain their separate entry contract.
pub(super) fn dispatch_destinations(ops: &[Op], cfg: &Cfg) -> Vec<usize> {
    let mut destinations = std::collections::BTreeSet::new();
    for root in cfg.handler_roots() {
        destinations.insert(cfg.blocks()[root.target.0 as usize].start);
    }
    for op in ops {
        if let Op::AbruptJump(target, _) = op {
            destinations.insert(*target as usize);
        }
    }
    destinations.into_iter().collect()
}

/// Revalidate original operation identities, not merely co-owned target labels.
/// The private immutable plan remains the language/stack proof, but a stale or
/// falsified structural certificate must never authorize a different internal edge.
fn literal_matches(ops: &[Op], pc: usize, site: string_literals::Site) -> bool {
    let (slot, literal, negate, mismatch) = site.identity();
    let Some(window) = pc.checked_add(4).and_then(|end| ops.get(pc..end)) else {
        return false;
    };
    let (actual_slot, actual_literal, comparison, target) = match window {
        [Op::LoadLocal(s), Op::Const(k), comparison, Op::JumpIfFalse(t)]
        | [Op::Const(k), Op::LoadLocal(s), comparison, Op::JumpIfFalse(t)] => {
            (*s, *k, comparison, *t as usize)
        }
        _ => return false,
    };
    actual_slot == slot
        && actual_literal == literal
        && target == mismatch
        && matches!(
            (negate, comparison),
            (false, Op::StrictEq) | (true, Op::StrictNotEq)
        )
}

impl Plan {
    #[allow(clippy::too_many_arguments)] // Original graph, exact entry/selection and stress certificates.
    pub(super) fn build(
        ops: &[Op],
        cfg: &Cfg,
        stack: &stack::StackPlan,
        entry: usize,
        regions: Option<&observed_regions::Plan>,
        deopt: Option<usize>,
        selected: Selected<'_>,
    ) -> Result<Self, CompileError> {
        let n = ops.len();
        if entry > n || deopt.is_some_and(|pc| pc > n) || stack.reachable.len() != n + 1 {
            return Err(CompileError::Unsupported);
        }
        // This owner map describes exactly the emitter's priority. Individual
        // literal sites contained in a group are its only permitted overlap.
        let mut owner = vec![None; n];
        let mut kinds = vec![Kind::Ordinary; n];
        for pc in 0..n {
            if owner[pc].is_some() {
                continue;
            }
            if let Some(site) = selected.borrowed.and_then(|p| p.site(pc)) {
                let end = site.native_end();
                if !site.native_matches(ops, pc) {
                    return Err(CompileError::Unsupported);
                }
                for at in pc..=end {
                    owner[at] = Some(pc);
                    kinds[at] = Kind::Interior;
                }
                kinds[pc] = Kind::Borrowed;
            } else if let Some(site) = selected.dispatch.and_then(|p| p.site(pc)) {
                let (start, slot, first_load, count) = site.native_identity();
                if start != pc
                    || first_load != pc + usize::from(!matches!(ops[pc], Op::LoadLocal(_)))
                    || !(2..=128).contains(&count)
                    || site.end() > n
                    || pc.checked_add(count.checked_mul(5).ok_or(CompileError::Unsupported)?)
                        != Some(site.end())
                {
                    return Err(CompileError::Unsupported);
                }
                for index in 0..count {
                    let at = pc + index * 5;
                    let literal = selected
                        .literals
                        .and_then(|p| p.site(at))
                        .ok_or(CompileError::Unsupported)?;
                    let (actual_slot, _, negate, mismatch) = literal.identity();
                    if !literal_matches(ops, at, literal)
                        || actual_slot != slot
                        || negate
                        || mismatch != at + 5
                        || !matches!(ops[at + 4], Op::Jump(_))
                    {
                        return Err(CompileError::Unsupported);
                    }
                    for part in at..at + 4 {
                        owner[part] = Some(pc);
                        kinds[part] = Kind::Interior;
                    }
                }
                kinds[pc] = Kind::Dispatch;
            } else if let Some(site) = selected.literals.and_then(|p| p.site(pc)) {
                if !literal_matches(ops, pc, site) || site.identity().3 > n {
                    return Err(CompileError::Unsupported);
                }
                for at in pc..pc + 4 {
                    owner[at] = Some(pc);
                    kinds[at] = Kind::Interior;
                }
                kinds[pc] = Kind::Literal;
            }
        }
        // Independently validate actual certificates, including contained sites
        // and every original incoming edge. A corrupt overlap never becomes a
        // missing label, and a future planner cannot silently waive this check.
        for pc in 0..n {
            let borrowed_here = selected
                .borrowed
                .is_some_and(|p| p.site(pc).is_some() || p.interior(pc));
            let dispatch_here = selected
                .dispatch
                .is_some_and(|p| p.site(pc).is_some() || p.interior(pc));
            let literal_here = selected
                .literals
                .is_some_and(|p| p.site(pc).is_some() || p.interior(pc));
            match owner[pc].map(|head| kinds[head]) {
                Some(Kind::Borrowed) if borrowed_here && !dispatch_here && !literal_here => {}
                Some(Kind::Dispatch) if dispatch_here && literal_here && !borrowed_here => {}
                Some(Kind::Literal) if literal_here && !dispatch_here && !borrowed_here => {}
                None if !borrowed_here && !dispatch_here && !literal_here => {}
                _ => return Err(CompileError::Unsupported),
            }
            if owner[pc].is_some()
                && (!stack.reachable[pc]
                    || regions
                        .is_some_and(|p| p.emission(pc) != observed_regions::Emission::Native))
            {
                return Err(CompileError::Unsupported);
            }
            let (targets, count) = stack::successors(&ops[pc], pc);
            for &target in &targets[..count] {
                if target > n {
                    return Err(CompileError::Unsupported);
                }
                if target < n
                    && kinds[target] == Kind::Interior
                    && (owner[pc].is_none() || owner[pc] != owner[target])
                {
                    return Err(CompileError::Unsupported);
                }
            }
            if crate::jit_ir::jump_target(&ops[pc])
                .is_some_and(|target| target <= pc && target < n && kinds[target] == Kind::Interior)
            {
                return Err(CompileError::Unsupported);
            }
            if regions
                .is_some_and(|p| p.emission(pc) == observed_regions::Emission::OmittedInterior)
            {
                let block = cfg.block_at(pc).ok_or(CompileError::Unsupported)?;
                let start = cfg.blocks()[block.0 as usize].start;
                if start == pc
                    || regions.unwrap().emission(start) != observed_regions::Emission::Cut
                {
                    return Err(CompileError::Unsupported);
                }
            }
        }
        let mut roots = vec![false; n + 1];
        roots[entry] = true;
        for root in cfg.handler_roots().iter().chain(cfg.resume_roots()) {
            roots[cfg.blocks()[root.target.0 as usize].start] = true;
        }
        for pc in dispatch_destinations(ops, cfg) {
            *roots.get_mut(pc).ok_or(CompileError::Unsupported)? = true;
        }
        if let Some(pc) = deopt {
            roots[pc] = true;
        }
        for (pc, &root) in roots.iter().enumerate().take(n) {
            if root
                && (kinds[pc] == Kind::Interior
                    || regions.is_some_and(|p| {
                        p.emission(pc) == observed_regions::Emission::OmittedInterior
                    }))
            {
                return Err(CompileError::Unsupported);
            }
        }
        let mut reachable = vec![false; n + 1];
        let mut queue: std::collections::VecDeque<_> = roots
            .iter()
            .enumerate()
            .filter_map(|(pc, &root)| root.then_some(pc))
            .collect();
        while let Some(pc) = queue.pop_front() {
            if reachable[pc] {
                continue;
            }
            reachable[pc] = true;
            if pc == n
                || !stack.reachable[pc]
                || deopt == Some(pc)
                || regions.is_some_and(|p| p.emission(pc) == observed_regions::Emission::Cut)
            {
                continue;
            }
            if kinds[pc] == Kind::Interior
                || regions
                    .is_some_and(|p| p.emission(pc) == observed_regions::Emission::OmittedInterior)
            {
                return Err(CompileError::Unsupported);
            }
            Self::successors(ops, pc, kinds[pc], selected, |target| {
                if target > n || target < n && kinds[target] == Kind::Interior {
                    return Err(CompileError::Unsupported);
                }
                queue.push_back(target);
                Ok(())
            })?;
        }
        #[cfg(test)]
        let sparse = TEST_ENABLED.with(std::cell::Cell::get).unwrap_or(true);
        #[cfg(not(test))]
        let sparse = true;
        let mut present = if sparse {
            reachable.clone()
        } else {
            vec![true; n + 1]
        };
        // Keep the real end address even if it has no native predecessor. It
        // remains uninserted unless an actual emitted edge or dynamic root uses it.
        present[n] = true;
        Self::validate(&present, &reachable, &roots)?;
        #[cfg(test)]
        TEST_COUNTS.with(|counts| {
            counts.set([
                present.iter().filter(|&&label| label).count(),
                present.iter().filter(|&&label| !label).count(),
            ])
        });
        Ok(Self {
            present: present.into_boxed_slice(),
            reachable: reachable.into_boxed_slice(),
            kinds: kinds.into_boxed_slice(),
            sparse,
        })
    }

    fn successors(
        ops: &[Op],
        pc: usize,
        kind: Kind,
        selected: Selected<'_>,
        mut edge: impl FnMut(usize) -> Result<(), CompileError>,
    ) -> Result<(), CompileError> {
        match kind {
            Kind::Borrowed => edge(
                selected
                    .borrowed
                    .and_then(|p| p.site(pc))
                    .ok_or(CompileError::Unsupported)?
                    .native_end()
                    + 1,
            )?,
            Kind::Dispatch => {
                let site = selected
                    .dispatch
                    .and_then(|p| p.site(pc))
                    .ok_or(CompileError::Unsupported)?;
                for index in 0..site.count() {
                    edge(site.branch(index) + 1)?;
                }
                edge(site.end())?;
            }
            Kind::Literal => {
                edge(pc + 4)?;
                edge(
                    selected
                        .literals
                        .and_then(|p| p.site(pc))
                        .ok_or(CompileError::Unsupported)?
                        .identity()
                        .3,
                )?;
            }
            Kind::Ordinary => {
                let (targets, count) = stack::successors(&ops[pc], pc);
                for &target in &targets[..count] {
                    edge(target)?;
                }
            }
            Kind::Interior => return Err(CompileError::Unsupported),
        }
        Ok(())
    }

    fn validate(present: &[bool], reachable: &[bool], roots: &[bool]) -> Result<(), CompileError> {
        if present.len() != reachable.len()
            || present.len() != roots.len()
            || present.last() != Some(&true)
            || present
                .iter()
                .zip(reachable.iter().zip(roots))
                .any(|(&has, (&live, &root))| !has && (live || root))
        {
            return Err(CompileError::Unsupported);
        }
        Ok(())
    }

    pub(super) fn labels(&self) -> &[bool] {
        &self.present
    }
    pub(super) fn emits_end(&self) -> bool {
        !self.sparse || self.reachable.last() == Some(&true)
    }
    pub(super) fn kind(&self, pc: usize) -> Kind {
        self.kinds[pc]
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    // Allocated original PC labels / certified absent independent native labels.
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
