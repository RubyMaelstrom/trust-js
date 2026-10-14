//! Share terminal recovery only across identical canonical publication shapes.
//!
//! ECMA-262 e28783d5 Execution Contexts, Completion Records and WeakRef liveness:
//! recovery continues the same invocation at its exact before-effect PC. Dirty
//! packed locals and the proven owning operand suffix are published before the
//! existing VM helper can observe them. Clean/dead SSA locals may be stale after
//! sparse observer reloads; neither they nor an opaque stack prefix are spilled.

use super::*;
use crate::fasthash::FastMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Kind {
    BeforeEffect,
    Specialization,
    LoopEntry,
}

impl Kind {
    fn helper(self) -> Helper {
        match self {
            Self::BeforeEffect => bytecode::native_deopt::resume_before,
            Self::Specialization => specialization_miss,
            Self::LoopEntry => loop_entry::miss,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key<'a> {
    kind: Kind,
    error: Block,
    dirty: &'a [u16],
    depth: usize,
    floor: usize,
}

struct Group<'a> {
    key: Key<'a>,
    block: Block,
}

pub(super) struct Plan<'a> {
    enabled: bool,
    index: FastMap<Key<'a>, usize>,
    groups: Vec<Group<'a>>,
}

impl Plan<'_> {
    pub(super) fn new() -> Self {
        let configured = || std::env::var("TRUST_JS_OPT_JIT_SHARED_RECOVERY").as_deref() != Ok("0");
        #[cfg(test)]
        let enabled = TEST_ENABLED
            .with(std::cell::Cell::get)
            .unwrap_or_else(configured);
        #[cfg(not(test))]
        let enabled = configured();
        Self {
            enabled,
            index: FastMap::default(),
            groups: Vec::new(),
        }
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    // Originating sites, physical groups/calls, original and emitted publication words.
    pub(super) static TEST_COUNTS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
    // Instructions, blocks, block parameters and actual branch arguments after SSA sealing.
    pub(super) static TEST_IR_SHAPE: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

#[cfg(test)]
pub(super) fn capture_ir_shape(function: &cranelift_codegen::ir::Function) {
    let parameters = function
        .layout
        .blocks()
        .map(|block| function.dfg.block_params(block).len())
        .sum();
    let arguments = function
        .layout
        .blocks()
        .flat_map(|block| function.layout.block_insts(block))
        .flat_map(|inst| {
            function.dfg.insts[inst]
                .branch_destination(&function.dfg.jump_tables, &function.dfg.exception_tables)
        })
        .map(|destination| destination.args(&function.dfg.value_lists).len())
        .sum();
    TEST_IR_SHAPE.with(|shape| {
        shape.set([
            function.dfg.num_insts(),
            function.dfg.num_blocks(),
            parameters,
            arguments,
        ]);
    });
}

impl Lowering<'_, '_> {
    pub(super) fn terminal_recovery(&mut self, kind: Kind, pc: usize, error: Block) {
        #[cfg(test)]
        TEST_COUNTS.with(|counts| {
            let mut counts_now = counts.get();
            counts_now[0] += 1;
            counts_now[2] += self.dirty.len() + self.stack_state.known();
            counts.set(counts_now);
        });
        let helper = kind.helper();
        if !self.terminal_recoveries.enabled {
            self.publish_locals();
            self.publish_stack(self.stack_state.depth);
            self.count_terminal_recovery_start();
            let success = self.canonical_helper(helper, pc as u32);
            self.count_terminal_recovery_result(success);
            self.b.ins().brif(success, self.ok, &[], error, &[]);
            #[cfg(test)]
            TEST_COUNTS.with(|counts| {
                let mut counts_now = counts.get();
                counts_now[1] += 1;
                counts_now[3] += self.dirty.len() + self.stack_state.known();
                counts.set(counts_now);
            });
            return;
        }
        let key = Key {
            kind,
            error,
            dirty: self.dirty,
            depth: self.stack_state.depth,
            floor: self.stack_state.floor,
        };
        let group = match self.terminal_recoveries.index.get(&key).copied() {
            Some(group) => group,
            None => {
                let block = self.b.create_block();
                self.b.append_block_param(block, types::I32);
                let group = self.terminal_recoveries.groups.len();
                self.terminal_recoveries.groups.push(Group { key, block });
                self.terminal_recoveries.index.insert(key, group);
                group
            }
        };
        // No observer occurs on this edge. SSA joins carry the same packed owners
        // the original site spilled, without retaining, dropping or inventing any.
        self.count_helper_runtime(helper);
        let pc = self.b.ins().iconst(types::I32, pc as i64);
        self.b
            .ins()
            .jump(self.terminal_recoveries.groups[group].block, &[pc.into()]);
    }

    pub(super) fn emit_terminal_recoveries(&mut self) {
        let groups = std::mem::take(&mut self.terminal_recoveries.groups);
        for Group { key, block } in groups {
            self.b.switch_to_block(block);
            self.b.set_cold_block(block);
            self.dirty = key.dirty;
            self.stack_state = stack::StackState {
                depth: key.depth,
                floor: key.floor,
            };
            self.publish_locals();
            self.publish_stack(key.depth);
            let pc = self.b.block_params(block)[0];
            self.count_terminal_recovery_start();
            let success = self.canonical_helper_call(key.kind.helper(), pc);
            self.count_terminal_recovery_result(success);
            // The VM consumed this invocation's completion/handlers already. Go
            // straight to its native epilogue, never unwind or run the old successor.
            self.b.ins().brif(success, self.ok, &[], key.error, &[]);
            #[cfg(test)]
            TEST_COUNTS.with(|counts| {
                let mut counts_now = counts.get();
                counts_now[1] += 1;
                counts_now[3] += key.dirty.len() + key.depth - key.floor;
                counts.set(counts_now);
            });
        }
    }
}

#[cfg(test)]
#[path = "jit_optimizing_recovery_tests.rs"]
mod tests;
