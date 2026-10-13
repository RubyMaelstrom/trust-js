//! Operand SSA publication and reloads; no reference-count operation is elided here.
use super::*;

impl Lowering<'_, '_> {
    pub(super) fn stack_read(&mut self, back: usize) -> Value {
        assert!(
            back > 0 && back <= self.stack_state.known(),
            "proven live operand"
        );
        self.b
            .use_var(self.stack_vars[self.stack_state.depth - back])
    }

    pub(super) fn stack_write(&mut self, back: usize, value: Value) {
        assert!(
            back > 0 && back <= self.stack_state.known(),
            "proven live operand"
        );
        self.b
            .def_var(self.stack_vars[self.stack_state.depth - back], value);
        let number = self.b.ins().bitcast(types::F64, MemFlags::new(), value);
        self.b
            .def_var(self.stack_numbers[self.stack_state.depth - back], number);
    }

    /// ECMA-262 e28783d5, Number Type and Number arithmetic: intermediate Number
    /// values need IEEE rounding/order, but not an observable NaN payload. Keep
    /// the unboxed result available across arithmetic while its canonical word
    /// remains the sole representation used for guards, owners and recovery.
    /// Both SSA variables are defined at every write/reload and merged together;
    /// an observer or a completion landing therefore cannot revive a stale value.
    /// Arbitrary packed bits may inhabit this variable, but only a proven/guarded
    /// Number may be consumed by floating-point instructions. These copies own
    /// nothing and are never published as packed words.
    pub(super) fn stack_number(&mut self, back: usize) -> Value {
        assert!(back > 0 && back <= self.stack_state.known());
        self.b
            .use_var(self.stack_numbers[self.stack_state.depth - back])
    }

    /// Materialize precisely the proven owning suffix. Opaque prefix cells are already
    /// canonical; writing them from static CFG depths could resurrect consumed owners.
    pub(super) fn publish_stack(&mut self, until: usize) {
        assert!(until <= self.stack_state.depth);
        self.count_publication(until.saturating_sub(self.stack_state.floor));
        let sp = self.top();
        for slot in self.stack_state.floor..until {
            let value = self.b.use_var(self.stack_vars[slot]);
            let offset = ((slot as i64 - self.stack_state.depth as i64) * 8) as i32;
            self.store(value, sp, offset);
        }
    }

    pub(super) fn reload_stack(&mut self, state: stack::StackState) {
        let sp = self.top();
        for slot in state.floor..state.depth {
            let offset = ((slot as i64 - state.depth as i64) * 8) as i32;
            let value = self.load(sp, offset);
            self.b.def_var(self.stack_vars[slot], value);
            let number = self.b.ins().bitcast(types::F64, MemFlags::new(), value);
            self.b.def_var(self.stack_numbers[slot], number);
        }
        self.stack_state = state;
    }

    pub(super) fn finish_normal(&mut self) {
        assert_eq!(
            self.stack_state, self.stack_plan.after[self.pc],
            "native stack transfer at {}",
            self.pc
        );
        let until = self.stack_plan.forget_to[self.pc];
        if until > self.stack_state.floor {
            self.publish_stack(until);
        }
    }

    pub(super) fn jump_normal(&mut self, next: Block) {
        self.finish_normal();
        if self
            .loop_facts
            .is_some_and(|plan| plan.native_write(self.pc))
        {
            // Both native hits and checked misses may change aliased fields or
            // array length. The completed write must never be replayed on a miss.
            self.validate_loop_fields(self.pc + 1);
        }
        let next = self.loop_exit(next);
        self.b.ins().jump(next, &[]);
    }
}
