//! The optimizing tier (P3; docs/v8-chase-p3-plan-2026-11-08.txt, "Design: the tier" and the
//! "Stage 1" record).
//!
//! Stage 1 makes the tier the code generator of second-stage chunks on ARM64 under TRUST_JS_OPT=1:
//! where the tier accepts the chunk the second stage built (its hot callees spliced in, seeded from
//! the first stage), it generates that chunk's native code in place of the template JIT. A hot
//! function with nothing to splice gets such a chunk too, published only if the tier accepts it
//! (`Interp::try_inline_recompile`, at the checkpoints the settled-feedback rule picks).
//!
//! The pipeline, by module:
//! - `snapshot`: the immutable per-compile summary of the chunk's feedback (one view per pc), and
//!   the feedback fingerprint the settled-feedback rule compares between checkpoints;
//! - `build`: the graph from the chunk's operations and the snapshot (one forward pass, known
//!   facts, never-run blocks as exits), or a refusal, with `liveness`, the slots each pc may read,
//!   and `phis`, the selection of each phi's representation after the build (stage 2 step 6);
//! - `verify`: the graph's invariants (SSA dominance, representations, states, terminators);
//! - `regalloc` and `a64`: registers and ARM64 code, the effects through the baseline templates,
//!   the exits;
//! - `exit`: the recipes and the deoptimizer that rebuilds the canonical frame and resumes the
//!   chunk in the VM, and the exits' counts;
//! - `profile`: what a chunk carries for the tier (its settled-feedback fingerprint, the exit
//!   profile, R, a discarded second stage's warm-up);
//! - `driver`: the switches, the settled-feedback rule, the pipeline's run, the discard of code
//!   that exits too often and its reoptimization (stage 2, step 3).
//!
//! x86-64 builds none of the back end until the plan's stage 6: there the tier is off, and the
//! front end is built only for its tests.

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) use crate::jit::opt_a64 as a64;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod build;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod driver;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) mod exit;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod ir;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) mod job;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod liveness;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod phis;
// Every chunk carries the record's cell (`Chunk::opt`), wherever the tier has a back end or not.
pub(crate) mod profile;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) mod regalloc;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod snapshot;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod verify;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) mod view;

#[cfg(test)]
mod build_tests;
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod tier_tests;

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) use driver::*;
#[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
)))]
pub(crate) use off::*;

/// The tier where it has no back end (x86-64 until the plan's stage 6): off, whatever TRUST_JS_OPT
/// says, so that no checkpoint reaches it, no exit countdown runs and no code has exits.
#[cfg(not(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
)))]
mod off {
    use crate::bytecode::Chunk;
    use crate::interpreter::Interp;

    pub(crate) fn enabled() -> bool {
        false
    }

    pub(crate) fn exit_every() -> u32 {
        0
    }

    pub(crate) fn waits(_: &Chunk) -> bool {
        false
    }

    pub(crate) fn refuses_unspliced(_: &Chunk) -> bool {
        false
    }

    pub(crate) fn main_thread_timer() -> Option<std::time::Instant> {
        None
    }

    pub(crate) fn log_main_thread(_: &str, _: &Chunk, _: Option<std::time::Instant>) {}

    pub(crate) fn fallback_timer(_: &Chunk) -> Option<std::time::Instant> {
        None
    }

    pub(crate) fn log_fallback(_: &Chunk, _: Option<std::time::Instant>, _: bool) {}

    pub(crate) fn forget(_: &Chunk) {}

    pub(crate) fn note_template(_: &Chunk, _: *mut u8, _: usize) {}

    pub(crate) fn template_class(_: &Chunk) -> Option<&'static str> {
        None
    }

    pub(crate) fn compile_into(_: &mut Interp, _: &Chunk) -> bool {
        false
    }

    pub(crate) fn reoptimizing(_: &Chunk) -> bool {
        false
    }

    pub(crate) fn reoptimize(_: &mut Interp, _: &Chunk) {}

    /// No code has an exit table.
    pub(crate) mod exit {
        pub(crate) fn release(_: *const u8) {}

        pub(crate) fn table_bytes(_: *const u8) -> usize {
            0
        }
    }
}

/// K, the native runs an entry counts toward its chunk's checkpoints with the tier on (stage 2,
/// step 11; the plan's "Step 11, the tier-up policy of point 2"): the checkpoint word counts in
/// 64ths of an invocation, so that a loop's back edge can charge its span's share of the chunk
/// (`jit::emit_loop_count`), as V8's interrupt budget counts bytecode bytes: Ignition charges the
/// whole bytecode length at an entry (InterpreterEntryTrampoline, builtins/arm64/
/// builtins-arm64.cc:1427-1433) and a back edge its loop's length (InterpreterAssembler::
/// JumpBackward, interpreter/interpreter-assembler.cc:1252-1256), against invocations times the
/// length (TieringManager's budget, execution/tiering-manager.cc:187-195, 264-290). Every distance
/// the checkpoints keep is K times as long, so that a loop-free function keeps its schedule.
pub(crate) const ENTRY_WEIGHT: u32 = 64;

/// The invocations between two checkpoints past the fourth with the tier on (stage 2, step 11),
/// a discard's warm-up's at R = 0: V8's minimum_invocations_after_ic_update (500;
/// flags/flag-definitions.h:1150), the budget an IC change raises a function's to
/// (TieringManager::NotifyICChanged, execution/tiering-manager.cc:497-576). Nothing forces a
/// compile: a function compiles at the first such window its feedback does not change in.
pub(crate) const SETTLED_WINDOW: u32 = 500;

/// The native runs an entry adds to its chunk's checkpoint count (`Chunk::jit_runs`): K
/// ([`ENTRY_WEIGHT`]) with the tier on, one with it off (P2's count). Every path that counts an
/// entry takes it: the direct sequence's gate and the construct sequence's take-back
/// (`jit::emit_direct_sequence`, whose shared stubs hold it as a process constant), and through
/// the engine's copy (`Interp::opt_entry_weight`) the call helper's hit,
/// `Interp::note_native_entry` and a borrowed frame's entry.
#[inline]
pub(crate) fn entry_weight() -> u32 {
    #[cfg(test)]
    {
        if enabled() {
            ENTRY_WEIGHT
        } else {
            1
        }
    }
    #[cfg(not(test))]
    {
        // The tier's switch is read once a process (`enabled`): so is the weight.
        static WEIGHT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        *WEIGHT.get_or_init(|| if enabled() { ENTRY_WEIGHT } else { 1 })
    }
}

/// Pin the tier off on this thread, whatever TRUST_JS_OPT, TRUST_JS_OPT_EAGER and
/// TRUST_JS_OPT_EXIT_EVERY say, for a test whose subject is the tier-off pipeline: the template
/// JIT's second stages and their mechanics, or the second stage the first checkpoint builds (the
/// suite also runs with the tier on, eager and forcing exits; V8's per-variant flags and status
/// file skips). Called first in the test, before it makes an engine, so that a child process the
/// test runs itself in pins itself as well. Where the tier has no back end it is off already.
#[cfg(test)]
pub(crate) fn pin_tier_off_for_test() {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    TestSwitches::default().set();
}

/// Keep the tier as the environment sets it, without its forced exits (TRUST_JS_OPT_EXIT_EVERY),
/// on this thread: for a test whose subject holds in the tier's code but which counts work that
/// generated code completes, some of which a forced exit hands to the VM (V8's status-file skips
/// of the deopt fuzzer). Called first in the test, as `pin_tier_off_for_test` is.
#[cfg(test)]
pub(crate) fn pin_forced_exits_off_for_test() {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    TestSwitches {
        on: enabled(),
        eager: eager(),
        exit_every: 0,
        no_reopt: no_reopt(),
        no_exit_profile: no_exit_profile(),
        exit_limit: exit_limit_base(),
        no_loop_count: no_loop_count(),
    }
    .set();
}

/// Keep the tier as the environment sets it, its template code counting no loop iterations
/// toward the checkpoints (TRUST_JS_OPT_NO_LOOP_COUNT), on this thread: for a test whose subject
/// needs its callees to reach their checkpoints before a caller whose loop calls them (stage 2,
/// step 8: the caller, which counts each call its loop makes, otherwise reaches its own first).
/// Called first in the test, as `pin_tier_off_for_test` is.
#[cfg(test)]
pub(crate) fn pin_loop_count_off_for_test() {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    TestSwitches {
        on: enabled(),
        eager: eager(),
        exit_every: exit_every(),
        no_reopt: no_reopt(),
        no_exit_profile: no_exit_profile(),
        exit_limit: exit_limit_base(),
        no_loop_count: true,
    }
    .set();
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
