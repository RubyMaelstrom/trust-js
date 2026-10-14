//! Reusable native call transitions for whole-function code.
//!
//! ECMA-262 e28783d5: EvaluateCall, [[Call]], PrepareForOrdinaryCall,
//! OrdinaryCallBindThis and PrepareForTailCall. The template tier's identity/realm/epoch
//! probes, shared activation transition and completion teardown stay authoritative.
//! This module outlines those SAME emitters behind a scalar C ABI. It does not invent a
//! second binding algorithm, bypass proxies or move an operand before the last guard.
//!
//! A caller publishes all owning words before entry. The stub returns 0 normally / 1 on
//! throw and writes the exact remaining top to JitCtx::final_sp. The caller must check the
//! flag before loading a normal result, and retain its own code lease through the call.
//! Active JitCode owners pin their stubs; the bounded directory holds only Weak handles.
//! No compiled address captures a realm, closure, caller Chunk, stack or TLS heap pointer.

use super::ExecutableBuffer;
use crate::bytecode::Chunk;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use crate::bytecode::Op;
use crate::interpreter::InterpLayout;
use crate::value::JitLayout;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static GENERATED_COUNT: AtomicU64 = AtomicU64::new(0);
static GENERATED_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) fn metrics_json() -> String {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    let (directory, dead) = native::directory_metadata();
    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    )))]
    let (directory, dead) = (0, 0);
    format!(concat!("{{\"generated_mappings\":{},\"generated_code_bytes\":{},",
        "\"generation_scope\":\"process; only when performance counters enabled\",",
        "\"directory_scope\":\"current_thread_shared_cache\",",
        "\"directory_and_dead_payload_bytes\":{},\"dead_entries\":{},",
        "\"directory_quality\":\"lower_bound\",\"directory_in_managed_memory\":false,",
        "\"excludes\":\"Rc headers and TLS runtime; live stub metadata belongs to referring JitCode owners\"}}"),
        GENERATED_COUNT.load(Relaxed), GENERATED_BYTES.load(Relaxed), directory, dead)
}

pub(super) struct SharedCallStub {
    argc: u16,
    with_this: bool,
    // Exact emission bytes are the cache key: all probed layouts and helper entry addresses
    // participate, without assuming that a hand-maintained layout fingerprint is complete.
    words: Box<[u32]>,
    executable: ExecutableBuffer,
}

impl SharedCallStub {
    pub(super) fn matches(&self, argc: u16, with_this: bool) -> bool {
        self.argc == argc && self.with_this == with_this
    }

    pub(super) fn entry(&self) -> usize {
        self.executable.as_ptr() as usize
    }

    pub(super) fn executable_bytes(&self) -> usize {
        self.executable.len()
    }

    pub(super) fn retained_metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + std::mem::size_of_val(&*self.words)
    }
}

impl Drop for SharedCallStub {
    fn drop(&mut self) {
        super::profiler::unregister(self.executable.as_ptr() as *mut u8);
    }
}

/// Unique arities only: neither the directory nor a caller's owned list grows per call site.
/// Unsupported layouts/platforms retain the complete ordinary checked-call implementation.
pub(super) fn prepare(
    chunk: &Chunk,
    values: &JitLayout,
    interp: &InterpLayout,
) -> Vec<Rc<SharedCallStub>> {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    {
        let mut result = Vec::<Rc<SharedCallStub>>::new();
        if !crate::bytecode::direct_shared_context_enabled()
            || std::env::var("TRUST_JS_OPT_JIT_CALL_STUBS").as_deref() == Ok("0")
            || !crate::value::jit_packed_owners_supported(values)
            || values.rc_strong_off != 0
            || chunk.jit_detailed_feedback_enabled()
        {
            return result;
        }
        for op in chunk.jit_ops() {
            let (Op::Call(argc, _) | Op::CallWithThis(argc, _)) = op else {
                continue;
            };
            let with_this = matches!(op, Op::CallWithThis(..));
            if *argc > 64 || result.iter().any(|stub| stub.matches(*argc, with_this)) {
                continue;
            }
            if let Some(stub) = native::get(chunk, values, interp, *argc, with_this) {
                result.push(stub);
            }
        }
        result
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    )))]
    {
        let _ = (chunk, values, interp);
        Vec::new()
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
mod native {
    use super::*;
    use crate::jit::{self, asm, JitCtx};
    use std::cell::RefCell;
    use std::mem::{offset_of, size_of, size_of_val};
    use std::rc::Weak;

    const ENTRIES: usize = 65 * 2;
    thread_local! {
        // Dead Weak backing is bounded by 130 payloads; no executable reservation, JavaScript
        // value or realm stays alive solely because this directory remembers an old arity.
        static DIRECTORY: RefCell<[Weak<SharedCallStub>; ENTRIES]> =
            RefCell::new(std::array::from_fn(|_| Weak::new()));
    }

    pub(super) fn directory_metadata() -> (usize, usize) {
        let dead = DIRECTORY.with(|directory| {
            directory
                .borrow()
                .iter()
                .filter(|entry| {
                    entry.strong_count() == 0
                        && entry.as_ptr() != Weak::<SharedCallStub>::new().as_ptr()
                })
                .count()
        });
        (
            size_of::<RefCell<[Weak<SharedCallStub>; ENTRIES]>>()
                + dead * size_of::<SharedCallStub>(),
            dead,
        )
    }

    pub(super) fn get(
        chunk: &Chunk,
        values: &JitLayout,
        interp: &InterpLayout,
        argc: u16,
        with_this: bool,
    ) -> Option<Rc<SharedCallStub>> {
        let words = emit(chunk, values, interp, argc, with_this)?;
        let index = usize::from(argc) * 2 + usize::from(with_this);
        let old = DIRECTORY.with(|directory| directory.borrow()[index].upgrade());
        if let Some(old) = old.filter(|old| old.words.as_ref() == words.as_slice()) {
            return Some(old);
        }
        // Allocation can retire unrelated code. Never hold a directory borrow across it;
        // referring bodies, not this Weak table, own the old stub and its live return PCs.
        let bytes = unsafe {
            std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), size_of_val(words.as_slice()))
        };
        let executable = ExecutableBuffer::from_bytes(bytes)?;
        if jit::perf_metrics_enabled() {
            GENERATED_COUNT.fetch_add(1, Relaxed);
            GENERATED_BYTES.fetch_add(executable.len() as u64, Relaxed);
            // One physical mapping, not one charge per referring body. This is a subset
            // of total generated native bytes; its time is already inside compile latency.
            jit::PERF_GENERATED_CODE_BYTES.fetch_add(executable.len() as u64, Relaxed);
        }
        jit::profiler::register_stub(
            argc,
            with_this,
            executable.as_ptr() as *mut u8,
            executable.len(),
        );
        let stub = Rc::new(SharedCallStub {
            argc,
            with_this,
            words: words.into_boxed_slice(),
            executable,
        });
        DIRECTORY.with(|directory| directory.borrow_mut()[index] = Rc::downgrade(&stub));
        Some(stub)
    }

    fn emit(
        chunk: &Chunk,
        values: &JitLayout,
        interp: &InterpLayout,
        argc: u16,
        with_this: bool,
    ) -> Option<Vec<u32>> {
        let mut a = asm::Asm::new();
        let slow = a.new_label();
        let hit_slow = a.new_label();
        let hit = a.new_label();
        let primary_hit = a.new_label();
        let normal = a.new_label();
        let throw = a.new_label();
        let exit = a.new_label();
        let finish = a.new_label();

        // C ABI: (ctx, bytecode_pc, owning_sp, primary_call_cache) -> completion flag.
        // Only x19..x22 are used as callee-saved scratch; every nested entry obeys that ABI.
        a.stp_pre(29, 30, -48);
        a.stp_off(19, 20, 16);
        a.stp_off(21, 22, 32);
        a.add_imm(29, 31, 0); // AAPCS64 frame chain for native profiling/unwinding.
        a.mov(19, 0);
        a.mov_w(22, 1); // C u32 arguments do not promise meaningful upper register bits.
        a.mov(20, 2);
        a.mov(12, 3);
        a.ldr_imm(21, 19, offset_of!(JitCtx, helpers) as u32);

        a.sub_imm(9, 20, (u32::from(argc) + 1) * 8);
        a.ldr_imm(9, 9, 0);
        jit::emit_exec_tag_guard(&mut a, 9, crate::value::PACK_OBJ, 16, slow);
        jit::emit_exec_payload(&mut a, 9, 10);
        if values.gc_data_off >= 4096 {
            return None;
        }
        a.add_imm(13, 10, values.gc_data_off as u32);
        a.mov_imm64(11, &crate::bytecode::CALL_IC_EPOCH as *const _ as u64);
        a.ldr_w_imm(15, 11, 0);
        a.cmn_imm_w(15, 1);
        a.b_cond(jit::C_EQ, slow); // exhausted epoch is never a raw-pointer proof
        a.ldr_imm(17, 19, offset_of!(JitCtx, genv) as u32);
        a.movz(14, crate::bytecode::CALL_IC_WAYS as u32, 0);
        let probe = a.new_label();
        let next = a.new_label();
        a.bind(probe);
        a.ldur(11, 12, 0);
        a.cmp_reg_x(13, 11);
        a.b_cond(jit::C_NE, next);
        a.ldr_w_imm(11, 12, 56);
        a.cmp_reg_w(11, 15);
        a.b_cond(jit::C_NE, next);
        a.ldr_imm(11, 12, 32);
        a.cmp_reg_x(11, 17);
        a.b_cond(jit::C_EQ, primary_hit);
        a.bind(next);
        a.add_imm(
            12,
            12,
            size_of::<std::cell::Cell<crate::bytecode::CallIc>>() as u32,
        );
        a.sub_imm(14, 14, 1);
        a.cbnz(14, false, probe);
        jit::emit_call_overflow_probe(&mut a, interp, hit, slow);
        a.bind(primary_hit);
        a.movz(15, crate::bytecode::CALL_IC_WAYS as u32, 0);
        a.sub_reg(15, 15, 14);
        a.bind(hit);
        if !jit::emit_direct_call(
            &mut a,
            interp,
            values,
            chunk.jit_inline_attempted_off(),
            chunk.jit_runs_off(),
            chunk.jit_inline_retry_at_off(),
            usize::from(argc),
            with_this,
            hit_slow,
            throw,
            normal,
            finish,
            // Shared by every site of an arity: the caller's frame shows its function start.
            jit::DirectCallSite::Position(crate::stack_trace::NO_POSITION),
        ) {
            return None;
        }
        a.bind(hit_slow);
        // An overflow hit has no primary-way ordinal. The checked path must re-probe it.
        a.cmp_imm_w(15, crate::bytecode::CALL_IC_WAYS as u32);
        a.b_cond(jit::C_HS, slow);
        a.mov(0, 19);
        a.add_shifted(1, 22, 15, 16);
        a.mov(2, 20);
        a.ldr_imm(16, 21, (jit::H_CALL_HIT * 8) as u32);
        a.blr(16);
        a.mov(20, 0);
        a.cbnz(1, false, throw);
        a.b(normal);
        a.bind(slow);
        a.mov(0, 19);
        a.mov(1, 22);
        a.mov(2, 20);
        a.ldr_imm(16, 21, (jit::H_CALL * 8) as u32);
        a.blr(16);
        a.mov(20, 0);
        a.cbnz(1, false, throw);
        a.bind(normal);
        a.movz(0, 0, 0);
        a.b(exit);
        a.bind(throw);
        a.movz(0, 1, 0);
        a.bind(exit);
        a.str_imm(20, 19, offset_of!(JitCtx, final_sp) as u32);
        a.ldp_off(21, 22, 32);
        a.ldp_off(19, 20, 16);
        a.ldp_post(29, 30, 48);
        a.ret();
        a.bind(finish);
        jit::emit_direct_finish_stub(&mut a, interp, true);
        Some(a.finish_with_offsets(&[]).0)
    }
}
