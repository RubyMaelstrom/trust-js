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
            NodeOp::StoreBinding { slot } => {
                let record = self.gp(node.inputs[0], X16);
                let value = self.gp(node.inputs[1], X17);
                self.a
                    .str_imm(value, record, operations::cap_slot_offset(u32::from(slot)));
                if !self.scalar_word(node.inputs[1]) {
                    // The record's card, after the store and before any safepoint, as
                    // `emit_cap_store` and `Chunk::store_cap_ic` mark it: a minor collection
                    // then finds an old record's edge to a young value.
                    emit_card_mark(&mut self.a, record, X9, X30);
                }
            }
            NodeOp::GlobalObject => {
                if !unused {
                    let dst = self.gp_dst(n);
                    self.a
                        .ldr_imm(dst, 19, std::mem::offset_of!(JitCtx, global_body) as u32);
                    self.a.sub_imm(dst, dst, self.layout.obj_from_cell as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::CheckEnvIs { env: expected, .. }
            | NodeOp::CheckCellIs { cell: expected, .. } => {
                // A record's or a cell's address the compile found, embedded only in this code,
                // never in a shared stub; the global environment's is pinned against reuse
                // (`Interp::global_env_pins`) while the heap lives.
                let site = self.check_exit(n)?;
                let value = self.gp(node.inputs[0], X16);
                self.a.mov_imm64(X17, expected);
                self.a.cmp_reg_x(value, X17);
                self.a.b_cond(NE, site);
            }
            NodeOp::CheckEnvGeneration { generation, .. } => {
                let site = self.check_exit(n)?;
                let record = self.gp(node.inputs[0], X16);
                self.a.ldr_w_imm(X17, record, self.layout.scope_gen as u32);
                if generation < 4096 {
                    self.a.cmp_imm_w(X17, generation);
                } else {
                    self.a.mov_imm64(X9, u64::from(generation));
                    self.a.cmp_reg_w(X17, X9);
                }
                self.a.b_cond(NE, site);
            }
            NodeOp::LoadGlobalLexical { binding } => {
                // The binding's address, which the record's generation keeps: embedded only in
                // this code, after the checks that proved the record.
                if !unused {
                    let dst = self.gp_dst(n);
                    self.a.mov_imm64(dst, binding);
                    self.a.ldr_imm(dst, dst, self.layout.binding_value as u32);
                    self.gp_done(n, dst);
                }
            }
            NodeOp::CheckGlobalEpoch { epoch, .. } => {
                // The epoch through the global object, which the realm's checks proved.
                let site = self.check_exit(n)?;
                let cell = self.gp(node.inputs[0], X16);
                let layout = self.layout;
                let offset =
                    layout.obj_from_cell + layout.obj_props + crate::value::PROPS_EPOCH_OFF;
                self.a.ldrh_imm(X17, cell, offset as u32);
                if epoch < 4096 {
                    self.a.cmp_imm_w(X17, u32::from(epoch));
                } else {
                    self.a.mov_imm64(X9, u64::from(epoch));
                    self.a.cmp_reg_w(X17, X9);
                }
                self.a.b_cond(NE, site);
            }
            _ => unreachable!("a free-name node"),
        }
        Ok(())
    }
}
