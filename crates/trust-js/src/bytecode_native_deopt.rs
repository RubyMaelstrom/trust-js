//! Before-effect transfer of one ordinary native frame to its canonical VM continuation.
//!
//! ECMA-262 e28783d5: Execution Contexts, GetValue/PutValue, Completion Records,
//! TryStatement Evaluation, IteratorClose and WeakRef liveness. This is continuation,
//! not a new Call: never repeat argument binding, getters, coercions or completed cleanup.
//! Logical inlined frames and suspended entries need additional reconstruction and are
//! deliberately outside this entry contract.

use super::{drive_vm, new_native_activation, Op, PackedValue, Value, VmStep};
use crate::jit::JitCtx;

#[cfg(test)]
thread_local! {
    pub(crate) static ENTRIES: std::cell::RefCell<Vec<(usize, usize, usize)>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    pub(crate) static FOREIGN_SIDECAR_ENTRIES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

/// Continue the current ordinary invocation at `pc`, before that operation's effects.
/// Returns the ordinary native BODY ABI: one for success, zero for an escaping error.
/// The generated caller must immediately take its function epilogue, never its unwinder
/// or the original operation's successor. The VM already consumed all callee handlers.
///
/// # Safety
/// `ctx` owns the live frame for its pinned immutable Chunk. All dirty locals and all
/// operand owners in `[stack_base, sp)` have been published with their exact ownership. No owner is
/// represented only in a register; no borrowed duplicate masquerades as an owning word.
/// `pc` is a verified bytecode boundary with its actual incoming stack/lexical state.
/// The entry is non-resumable and contains no speculative inline bytecode.
pub(crate) unsafe extern "C" fn resume_before(
    raw: *mut JitCtx,
    pc: u32,
    sp: *mut PackedValue,
) -> u64 {
    let ctx = &mut *raw;
    let chunk = &*ctx.chunk;
    assert!(
        !chunk.resumable,
        "deoptimization requires an ordinary frame"
    );
    assert!((pc as usize) <= chunk.ops.len(), "deoptimization PC");
    debug_assert!(
        chunk.inline_targets.is_empty(),
        "logical inline frames need separate reconstruction"
    );
    debug_assert!(
        ctx.resume_activation.is_null() || (*ctx.resume_activation).chunk != ctx.chunk,
        "ordinary deoptimization cannot consume a continuation's borrowed allocation"
    );
    debug_assert!(
        ctx.error.is_none(),
        "before-effect state has no pending error"
    );
    let depth = sp.offset_from(ctx.stack_base) as usize;
    #[cfg(test)]
    ENTRIES.with(|entries| {
        entries
            .borrow_mut()
            .push((pc as usize, depth, ctx.handler_floor))
    });

    // Compiled calls give each callee a frame record of its own, so the sidecar found here is
    // this chunk's. Remain defensive regardless: reuse only this chunk's state. A stateless
    // callee instead owns a temporary sidecar, without replacing or clearing any other frame's
    // prepared References / lexical environment.
    let same_activation = ctx
        .activation
        .as_ref()
        .is_some_and(|state| state.chunk == ctx.chunk);
    #[cfg(test)]
    if !same_activation && ctx.activation.is_some() {
        FOREIGN_SIDECAR_ENTRIES.with(|count| count.set(count.get() + 1));
    }
    debug_assert!(
        same_activation || ctx.activation.is_none() || !chunk.jit_needs_activation_state(),
        "a stateful callee cannot use a shared context"
    );
    let mut temporary = (!same_activation).then(|| new_native_activation(ctx));
    let state = if same_activation {
        ctx.activation.as_mut().unwrap()
    } else {
        temporary.as_mut().unwrap()
    };
    assert!(
        state.stack.is_empty(),
        "native operand owners already moved"
    );
    state.stack.reserve(depth);

    // Move, do not clone: the native buffer's bytes cease to own these words. The VM may
    // resize its independent stack. No native instruction resumes with the old sp afterwards.
    std::ptr::copy_nonoverlapping(ctx.stack_base, state.stack.as_mut_ptr(), depth);
    state.stack.set_len(depth);
    ctx.final_sp = ctx.stack_base;

    // Handler PCs/depths are CALLEE-relative. Never let drive_vm see caller handlers in a
    // shared-context call: it must return an escaping throw to the real caller's dispatcher.
    let mut handlers = ctx.handlers.split_off(ctx.handler_floor);
    let mut next = pc as usize;
    let result = drive_vm(
        &mut *ctx.interp,
        chunk,
        &mut state.env,
        &state.cap_env,
        &mut state.references,
        std::slice::from_raw_parts_mut(ctx.slots, ctx.n_slots),
        &mut state.stack,
        &mut next,
        &ctx.this_val,
        &mut handlers,
        &mut state.disposal_frames,
        &mut state.class_states,
        None,
        false,
        None,
    );
    // Exact remaining owners on both normal and exceptional completion. Locals still belong
    // to the native frame and its existing teardown; never drop or reinitialize them here.
    state.stack.clear();
    if same_activation {
        ctx.env_raw = std::rc::Rc::as_ptr(&state.env) as *const u8;
        ctx.env_parent_raw = state
            .env
            .borrow()
            .parent
            .as_ref()
            .map_or(std::ptr::null(), |parent| {
                std::rc::Rc::as_ptr(parent) as *const u8
            });
    }
    match result {
        Ok(VmStep::Done(value) | VmStep::Return(value) | VmStep::ResumeReturn(value)) => {
            ctx.ret = PackedValue::pack(value);
            1
        }
        Ok(VmStep::BareReturn) => {
            ctx.ret = PackedValue::pack(Value::Undefined);
            1
        }
        Err(error) => {
            ctx.error = Some(error);
            0
        }
        Ok(_) => unreachable!("an ordinary deoptimized frame cannot suspend or escape a loop"),
    }
}

pub(crate) fn supported(chunk: &super::Chunk) -> bool {
    !chunk.resumable
        && chunk.inline_targets.is_empty()
        && !chunk.ops.iter().any(|op| matches!(op, Op::FragmentExit(_)))
}
