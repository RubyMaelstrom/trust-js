//! Native guards for the bytecode compiler's frame-independent inline splices.
//!
//! ECMA-262 e28783d5 PrepareForOrdinaryCall, OrdinaryCallBindThis and GetValue:
//! callee identity, closure context and running Realm must match before binding.
//! Both guard exits retain every operand owner; a miss executes the original Call.
//! The canonical flattened Chunk supplies remapped slots and logical source frames.

use super::*;

impl Lowering<'_, '_> {
    /// Read an operand without retaining or transferring it. An opaque prefix is already
    /// canonical; a proven owning suffix may still have its current word only in SSA.
    fn inline_operand(&mut self, back: usize) -> Value {
        assert!(back > 0 && back <= self.stack_state.depth);
        if back <= self.stack_state.known() {
            self.stack_read(back)
        } else {
            let sp = self.top();
            self.load(sp, -(back as i32) * 8)
        }
    }

    pub(super) fn inline_guard(&mut self, target: u32, fallback: Block, next: Block) {
        // Packing a temporary owner probes the actual stored Rc word, rather than assuming
        // that Rc::as_ptr is the packed payload. The code embeds only bits: InlineTarget's
        // Weak keeps the allocation header from reuse, and live operands own the callee.
        let (argc, expected_env, expected_genv, check_this, expected) = {
            let target = self.chunk.jit_inline_target(target);
            let expected = self
                .values
                .valid
                .then(|| target.pin.upgrade())
                .flatten()
                .map(|object| PackedValue::pack(crate::value::Value::Obj(object.into())).bits());
            (
                target.argc,
                target.expected_env,
                target.expected_genv,
                target.check_this,
                expected,
            )
        };
        let Some(expected) = expected else {
            self.jump_normal(fallback);
            return;
        };
        let callee_back = argc as usize + 1;
        let callee = self.inline_operand(callee_back);
        let mut matches = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, callee, expected as i64);
        if expected_env != 0 {
            let env = self.load(self.ctx, offset_of!(JitCtx, env_raw) as i32);
            let same = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, env, expected_env as i64);
            matches = self.b.ins().band(matches, same);
        }
        let genv = self.load(self.ctx, offset_of!(JitCtx, genv) as i32);
        let same = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, genv, expected_genv as i64);
        matches = self.b.ins().band(matches, same);
        if check_this {
            let receiver = self.inline_operand(callee_back + 1);
            let tag = self
                .b
                .ins()
                .band_imm_s(receiver, 0xffff_0000_0000_0000u64 as i64);
            let object = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, crate::value::PACK_OBJ as i64);
            matches = self.b.ins().band(matches, object);
        }
        self.finish_normal();
        self.b.ins().brif(matches, next, &[], fallback, &[]);
    }
}
