//! Same-activation Script tiering. This is execution policy, not ScriptEvaluation:
//! declarations and Script/Realm context are established once by the interpreter.
use crate::bytecode::Chunk;
use crate::interpreter::Interp;
use crate::jit::{JitCode, JitCompileOutcome};
use std::cell::Cell;
use std::rc::Rc;

/// Cold loops collect bounded work evidence at already-required backedge safepoints.
/// The capped span is a cheap size weight, not a measured instruction count. Requiring 128
/// executed edges keeps short one-shot loops off the native compiler regardless of size.
pub(crate) struct VmTiering {
    edges: u32,
    work: u32,
    next_work: u32,
    attempts: u8,
    disabled: bool,
}

impl VmTiering {
    pub(crate) fn new() -> Self {
        Self {
            edges: 0,
            work: 0,
            next_work: 4096,
            attempts: 0,
            disabled: false,
        }
    }

    pub(crate) fn backedge(&mut self, from: usize, to: usize) -> bool {
        if self.disabled || to > from {
            return false;
        }
        self.edges = self.edges.saturating_add(1);
        self.work = self.work.saturating_add((from - to + 1).min(64) as u32);
        if self.edges < 128 || self.work < self.next_work {
            return false;
        }
        self.attempts += 1;
        self.next_work = self.work.saturating_add(4096u32 << self.attempts.min(7));
        self.disabled = self.attempts == 8;
        true
    }

    pub(crate) fn disable(&mut self) {
        self.disabled = true;
    }
}

/// Borrowed-frame code must never enter the ordinary function CallIc cache.
/// Allocate this sidecar only after an activation demonstrates loop hotness.
#[derive(Default)]
pub(crate) struct OsrCodeState {
    code: crate::jit::cache::NativeCodeSlot,
    budget_wait_bytes: Cell<usize>,
}

impl OsrCodeState {
    pub(crate) fn code(&self, i: &mut Interp, chunk: &Chunk) -> Option<Rc<JitCode>> {
        let permit = if self.code.ready_to_compile()
            && crate::jit::ensure_executable_capacity(self.budget_wait_bytes.get())
        {
            self.code.begin_compile()
        } else {
            None
        };
        if let Some(permit) = permit {
            let layout = *i
                .jit_layout
                .get_or_init(|| crate::value::jit_layout(&i.object_proto));
            if !i.interp_layout.get().valid {
                let layout = crate::interpreter::interp_layout(i);
                i.interp_layout.set(layout);
            }
            match crate::jit::compile_borrowed_profiled(chunk, &layout, &i.interp_layout.get()) {
                JitCompileOutcome::Compiled(code) => {
                    self.budget_wait_bytes.set(0);
                    permit.commit(Some(Rc::new(code)));
                }
                JitCompileOutcome::Deferred { required_bytes } => {
                    self.budget_wait_bytes.set(required_bytes)
                }
                JitCompileOutcome::Unavailable => {
                    permit.commit(None);
                }
            }
        }
        self.code.get().flatten()
    }

    pub(crate) fn unavailable(&self) -> bool {
        matches!(self.code.get(), Some(None))
    }

    pub(crate) fn retained_metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.code.retained_metadata_bytes()
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) {
        visitor.add_function_bytecode_bytes(self.retained_metadata_bytes());
        if let Some(Some(code)) = self.code.get() {
            visitor.jit_code(&code);
        }
    }
}

/// One bounded optimizing continuation per Chunk. Counters contain no JS owners;
/// their immutable vector is pinned by the same Chunk as the generated loads.
#[cfg(feature = "optimizing-jit")]
pub(crate) struct OptimizingLoopState {
    pub(crate) counters: Vec<(usize, Cell<u32>)>,
    pub(crate) selected: Cell<Option<usize>>,
    pub(crate) attempted: Cell<bool>,
    pub(crate) code: crate::jit::cache::NativeCodeSlot,
    pub(crate) misses: Cell<u8>,
    #[cfg(test)]
    pub(crate) entries: Cell<usize>,
}

#[cfg(feature = "optimizing-jit")]
impl OptimizingLoopState {
    pub(crate) fn new(headers: Vec<usize>, threshold: u32) -> Self {
        Self {
            counters: headers
                .into_iter()
                .map(|pc| (pc, Cell::new(threshold)))
                .collect(),
            selected: Cell::new(None),
            attempted: Cell::new(false),
            code: crate::jit::cache::NativeCodeSlot::new(),
            misses: Cell::new(0),
            #[cfg(test)]
            entries: Cell::new(0),
        }
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) {
        visitor.add_function_bytecode_bytes(
            std::mem::size_of::<Self>()
                + self.counters.capacity() * std::mem::size_of::<(usize, Cell<u32>)>()
                + self.code.retained_metadata_bytes(),
        );
        if let Some(Some(code)) = self.code.get() {
            visitor.jit_code(&code);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotness_requires_executed_edges_and_has_bounded_retries() {
        let mut tier = VmTiering::new();
        for _ in 0..100 {
            assert!(!tier.backedge(1000, 0));
        }
        let mut requests = 0;
        for _ in 0..200_000 {
            requests += usize::from(tier.backedge(1000, 0));
        }
        assert_eq!(requests, 8);
        assert!(!tier.backedge(0, 1));
    }
}
