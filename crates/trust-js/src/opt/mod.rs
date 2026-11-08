//! The optimizing tier (P3; docs/v8-chase-p3-plan-2026-11-08.txt, "Design: the tier" and the
//! "Stage 1" record).
//!
//! Stage 1 makes the tier the code generator of second-stage chunks on ARM64 under TRUST_JS_OPT=1:
//! where the tier accepts the chunk the second stage built (its hot callees spliced in, seeded from
//! the first stage), it generates that chunk's native code in place of the template JIT.
//!
//! The pipeline, by module:
//! - `snapshot`: the immutable per-compile summary of the chunk's feedback (one view per pc), and
//!   the feedback fingerprint the settled-feedback rule compares between checkpoints;
//! - `build`: the graph from the chunk's operations and the snapshot (one forward pass, known
//!   facts, never-run blocks as exits), or a refusal;
//! - `verify`: the graph's invariants (SSA dominance, representations, states, terminators);
//! - `regalloc` and `a64`: registers and ARM64 code, the effects through the baseline templates,
//!   the exits;
//! - `exit`: the recipes and the deoptimizer that rebuilds the canonical frame and resumes the
//!   chunk in the VM.
//! x86-64 builds none of the back end until the plan's stage 6.
#![cfg_attr(not(test), allow(dead_code))]

pub(crate) mod build;
pub(crate) mod exit;
pub(crate) mod ir;
pub(crate) mod regalloc;
pub(crate) mod snapshot;
pub(crate) mod verify;

#[cfg(test)]
mod build_tests;

/// TRUST_JS_OPT: the tier is off unless set to a value other than `0` (the plan's Stage 1: off
/// by default until a stage exit measures a net win).
pub(crate) fn enabled() -> bool {
    matches!(env_value!("TRUST_JS_OPT"), Some(v) if v != "0")
}

/// TRUST_JS_OPT_LOG: one stderr line per compile, refusal (with its reason and pc), exit and
/// compile time.
pub(crate) fn log_enabled() -> bool {
    env_flag!("TRUST_JS_OPT_LOG")
}

/// TRUST_JS_OPT_EAGER: compile at the first checkpoint, without waiting for settled feedback (the
/// eager gate; V8's --stress-maglev, SpiderMonkey's eager Ion compilation).
pub(crate) fn eager() -> bool {
    env_flag!("TRUST_JS_OPT_EAGER")
}

/// TRUST_JS_OPT_EXIT_EVERY=N: fail every Nth check passed (the forced-exit gate; V8's
/// --deopt-every-n-times, SpiderMonkey's bailAfter). 0 or unset: off.
pub(crate) fn exit_every() -> u32 {
    env_value!("TRUST_JS_OPT_EXIT_EVERY")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::ir::*;

    #[test]
    fn a_graph_records_blocks_nodes_edges_and_states() {
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, exit);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let n = g.add_node(entry, NodeOp::CheckNumber, &[a], Rep::F64, 1);
        let state = g.add_state(FrameState {
            pc: 1,
            slots: vec![SlotSource::Frame, SlotSource::Dead].into(),
            stack: vec![a].into(),
            handlers: Box::new([]),
        });
        g.node_mut(n).state = Some(state);
        g.node_mut(n).reason = Some(ExitReason::NotNumber);
        g.add_node(entry, NodeOp::Jump(exit), &[], Rep::None, 2);
        let phi = g.add_node(exit, NodeOp::Phi, &[n], Rep::F64, 3);
        let boxed = g.add_node(exit, NodeOp::F64ToTagged, &[phi], Rep::Tagged, 3);
        g.add_node(exit, NodeOp::Return, &[boxed], Rep::None, 3);

        assert_eq!(g.block(entry).succs, vec![exit]);
        assert_eq!(g.block(exit).preds, vec![entry]);
        assert_eq!(g.block(exit).phis, vec![phi]);
        assert_eq!(g.block(exit).body.len(), 2);
        assert!(g.node(g.block(exit).body[1]).op.is_terminator());
        assert!(g.node(n).op.can_exit());
        assert_eq!(g.state(state).stack[0], a);
        assert!(!NodeOp::I32And.can_exit());
        assert!(NodeOp::Effect {
            pc: 4,
            pops: 2,
            pushes: 1
        }
        .is_effectful());
    }
}
