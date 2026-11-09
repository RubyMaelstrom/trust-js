//! The free-name nodes' ARM64 code (P3 stage 2, step 4; the plan's "Step 4"): Decision 21's static
//! addresses as the templates read them (`operations::emit_static_name_load`,
//! `emit_cap_store`), a walk of parent links and a slot's load or store, and the checks of the
//! global names' proofs.

use super::*;

impl Gen<'_> {
    /// The parent field's offset in a record, which a load reaches in one instruction (the
    /// templates' gate, `operations::static_name`).
    fn scope_parent(&self, pc: u32) -> Result<u32, Refusal> {
        let layout = self.layout;
        if layout.scope_parent_valid
            && layout.scope_parent.is_multiple_of(8)
            && layout.scope_parent / 8 < 4096
        {
            Ok(layout.scope_parent as u32)
        } else {
            Err(Refusal::at(
                pc as usize,
                "a record layout whose parent no load reaches",
            ))
        }
    }

    /// A free-name node (see ir.rs's "Free names").
    pub(super) fn lower_name(&mut self, n: NodeIx) -> Result<(), Refusal> {
        let graph = self.graph;
        let node = graph.node(n);
        let unused = self.home(n) == Home::None;
        match node.op {
            NodeOp::Env => {
                if !unused {
                    let dst = self.gp_dst(n);
                    self.a
                        .ldr_imm(dst, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::EnvParent => {
                let offset = self.scope_parent(node.pc)?;
                if !unused {
                    let record = self.gp(node.inputs[0], X16);
                    let dst = self.gp_dst(n);
                    self.a.ldr_imm(dst, record, offset);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::LoadBinding { slot } => {
                if !unused {
                    let record = self.gp(node.inputs[0], X16);
                    let dst = self.gp_dst(n);
                    self.a
                        .ldr_imm(dst, record, operations::cap_slot_offset(u32::from(slot)));
                    self.gp_done(n, dst);
                }
            }
            NodeOp::CheckInitialized => {
                let site = self.check_exit(n)?;
                let word = self.gp(node.inputs[0], X16);
                self.a.movz(X17, (crate::value::PACK_EMPTY >> 48) as u32, 3);
                self.a.cmp_reg_x(word, X17);
                self.a.b_cond(EQ, site);
            }
            _ => unreachable!("a free-name node"),
        }
        Ok(())
    }
}
