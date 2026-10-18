//! Exact native call-cache exhaustion proof. ECMA-262 e28783d5 §9.4/[[Call]]:
//! changing a predicate must preserve the live callee/Realm and safe fallback.
use super::*;
use crate::bytecode::{CallIc, CallOverflow, Tier, CALL_IC_EPOCH, CALL_IC_WAYS};
use crate::value::{Callable, Object};
use std::cell::Cell;
use std::mem::{offset_of, size_of};
use std::sync::atomic::Ordering::Relaxed;

fn executable(words: Vec<u32>) -> ExecutableBuffer {
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    ExecutableBuffer::from_bytes(&bytes).expect("native epoch probe")
}

#[test]
fn call_epoch_cmn32_preserves_full_epoch_register_at_boundaries() {
    let mut a = asm::Asm::new();
    a.mov(15, 0);
    a.cmn_imm_w(15, 1);
    a.cset_w(0, C_EQ);
    a.str_imm(15, 1, 0);
    a.ret();
    let words = a.finish();
    assert_eq!(words[1], 0x3100_05ff, "ADDS WZR,W15,#1");
    let code = executable(words);
    let run: unsafe extern "C" fn(u64, *mut u64) -> u32 =
        unsafe { std::mem::transmute(code.as_ptr()) };
    for upper in [0, 1, 0x1234_abcd, u32::MAX] {
        for lower in [0, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX - 1, u32::MAX] {
            let value = (u64::from(upper) << 32) | u64::from(lower);
            let mut saved = 0;
            assert_eq!(
                unsafe { run(value, &mut saved) },
                u32::from(lower == u32::MAX)
            );
            assert_eq!(saved, value, "WZR destination must not change X15");
        }
    }
}

#[test]
fn call_epoch_cmn32_guard_removes_exactly_two_instructions() {
    fn sequence(compact: bool) -> Vec<u32> {
        let mut a = asm::Asm::new();
        let miss = a.new_label();
        if compact {
            a.cmn_imm_w(15, 1);
        } else {
            a.mov_imm64(11, u32::MAX as u64);
            a.cmp_reg_w(15, 11);
        }
        a.b_cond(C_EQ, miss);
        a.bind(miss);
        a.finish()
    }
    let before = sequence(false);
    let after = sequence(true);
    assert_eq!(before.len(), 4);
    assert_eq!(after.len(), 2);
    assert_eq!(after[0], 0x3100_05ff);
    assert_eq!(
        before.last(),
        after.last(),
        "unchanged BEQ target/condition"
    );
}

// These aligned words are exclusively a machine-code input, never an invalid Rust JitCtx.
// The helper stubs record their route and the incoming W15, then preserve the operand top.
#[repr(align(16))]
struct ProbeContext([usize; size_of::<JitCtx>().div_ceil(size_of::<usize>())]);

fn helper(route: u32) -> ExecutableBuffer {
    let mut a = asm::Asm::new();
    a.str_imm(15, 0, 8);
    a.movz(3, route, 0);
    a.str_imm(3, 0, 0);
    a.mov(0, 2);
    a.movz(1, 0, 0);
    a.ret();
    executable(a.finish())
}

fn caller_chunk(engine: &mut crate::Engine) -> Rc<Chunk> {
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    assert!(matches!(
        engine.eval(
            "function epochCaller(f){return f();}epochCaller(function(){return 1;});",
            false
        ),
        Ok(crate::Completion::Value(_))
    ));
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var("epochCaller", &env)
        .ok()
        .expect("caller");
    let object = value.as_obj().expect("function").borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func
        .code
        .get()
        .and_then(Option::as_ref)
        .expect("bytecode caller")
        .clone()
}

fn call_probe(engine: &mut crate::Engine, chunk: &Chunk) -> (ExecutableBuffer, u32) {
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let (pc, cache) = chunk
        .jit_ops()
        .iter()
        .enumerate()
        .find_map(|(pc, op)| match op {
            crate::bytecode::Op::Call(0, cache) => Some((pc, *cache)),
            _ => None,
        })
        .expect("zero-argument call");
    let mut a = asm::Asm::new();
    a.stp_pre(19, 20, -16);
    a.stp_pre(21, 30, -16);
    a.mov(19, 0);
    a.mov(20, 1);
    a.mov(21, 2);
    let unwind = a.new_label();
    let finish = a.new_label();
    // Exercise the REAL shared primary/overflow emitter, without entering a callee.
    // The raw helper stubs below observe which checked dispatch path was selected.
    emit_call_inline(
        &mut a, chunk, &layout, &ilayout, pc, 524288, false, false, unwind, finish,
    );
    a.bind(unwind);
    a.bind(finish);
    a.mov(0, 20);
    a.ldp_post(21, 30, 16);
    a.ldp_post(19, 20, 16);
    a.ret();
    // The secondary probes may be the chunk's shared stub (reached with `bl`; lr is saved).
    emit_shared_stubs(
        &mut a,
        &StubContext {
            layout: &layout,
            ilayout: Some(&ilayout),
            direct: Some(DirectCallContext {
                attempted_off: chunk.jit_inline_attempted_off(),
                runs_off: chunk.jit_runs_off(),
                retry_off: chunk.jit_inline_retry_at_off(),
                finish_stub: finish,
            }),
        },
    );
    let words = a.finish();
    assert_eq!(
        words.iter().filter(|&&word| word == 0x3100_05ff).count(),
        1,
        "actual call emitter must contain exactly one CMN W15 guard"
    );
    (executable(words), cache)
}

#[test]
fn call_epoch_native_primary_and_overflow_permanently_miss_at_max() {
    // The epoch is process-global and may not be restored after exhaustion. Isolate
    // this test so parallel GC/cache tests cannot change its boundary observations.
    const CHILD: &str = "TRUST_JS_TEST_CALL_EPOCH_CMN_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "jit::call_epoch_tests::call_epoch_native_primary_and_overflow_permanently_miss_at_max", "--test-threads=1"])
            .env(CHILD, "1").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let mut engine = crate::Engine::new();
    let chunk = caller_chunk(&mut engine);
    let (code, cache) = call_probe(&mut engine, &chunk);
    let cells =
        unsafe { &*(chunk.jit_call_cache_ptr(cache) as *const [Cell<CallIc>; CALL_IC_WAYS]) };
    let object = Object::new(None);
    let key = crate::value::Gc::as_ptr(&object) as usize;
    let realm = Env::as_ptr(&engine.interp.global_env) as usize;
    let mut context = ProbeContext([0; size_of::<JitCtx>().div_ceil(size_of::<usize>())]);
    context.0[offset_of!(JitCtx, interp) / size_of::<usize>()] =
        &mut *engine.interp as *mut _ as usize;
    context.0[offset_of!(JitCtx, genv) / size_of::<usize>()] = realm;
    let slow = helper(1);
    let hit = helper(2);
    let mut helpers = [0usize; N_HELPERS];
    helpers[H_CALL] = slow.as_ptr() as usize;
    helpers[H_CALL_HIT] = hit.as_ptr() as usize;
    let mut operand = [PackedValue::pack(Value::Obj(object.clone()))];
    let top = unsafe { operand.as_mut_ptr().add(1) };
    let run: unsafe extern "C" fn(*mut usize, *mut PackedValue, *const usize) -> *mut PackedValue =
        unsafe { std::mem::transmute(code.as_ptr()) };
    let invoke = |context: &mut ProbeContext| {
        context.0[0] = 0;
        context.0[1] = 0;
        assert_eq!(
            unsafe { run(context.0.as_mut_ptr(), top, helpers.as_ptr()) },
            top
        );
        (context.0[0], context.0[1])
    };
    let seed = CallIc {
        callee: key,
        global_env: realm,
        ..CallIc::EMPTY
    };
    for epoch in [0, u32::MAX - 1] {
        CALL_IC_EPOCH.store(epoch, Relaxed);
        cells[0].set(CallIc { epoch, ..seed });
        assert_eq!(invoke(&mut context), (2, 0), "real primary hit at {epoch}");
        cells[0].set(CallIc::EMPTY);
        engine.interp.call_overflow.insert(
            CallIc { epoch, ..seed },
            crate::value::Gc::downgrade(&object),
        );
        // A non-direct overflow hit intentionally re-probes in H_CALL, with way=4.
        assert_eq!(
            invoke(&mut context),
            (1, CALL_IC_WAYS),
            "real overflow hit at {epoch}"
        );
    }
    // Forge an otherwise matching MAX entry: real producers refuse these, but a
    // consumer must still reject MAX BEFORE using any cached raw addresses.
    crate::bytecode::invalidate_call_caches();
    assert_eq!(CALL_IC_EPOCH.load(Relaxed), u32::MAX);
    cells[0].set(CallIc {
        epoch: u32::MAX,
        ..seed
    });
    assert_eq!(
        invoke(&mut context),
        (1, u32::MAX as usize),
        "primary must take pre-probe fallback"
    );
    cells[0].set(CallIc::EMPTY);
    let set = ((key as u64).wrapping_mul(crate::bytecode::CALL_OVERFLOW_HASH)
        >> engine.interp.call_overflow.hash_shift) as usize;
    // Build exclusively machine-readable backing with a deliberately matching MAX
    // entry. Never mutate through a pointer derived from the real shared table and
    // never create/drop Rust CallOverflowSet/Weak values from synthetic bytes.
    let count = 1usize << (64 - engine.interp.call_overflow.hash_shift);
    let mut forged = vec![0usize; count * CallOverflow::SET_STRIDE / size_of::<usize>()];
    unsafe {
        forged
            .as_mut_ptr()
            .cast::<u8>()
            .add(set * CallOverflow::SET_STRIDE)
            .cast::<CallIc>()
            .write(CallIc {
                epoch: u32::MAX,
                ..seed
            });
    }
    let original_sets = engine.interp.call_overflow.raw_sets;
    engine.interp.call_overflow.raw_sets = forged.as_ptr().cast();
    for _ in 0..4 {
        crate::bytecode::invalidate_call_caches();
        assert_eq!(CALL_IC_EPOCH.load(Relaxed), u32::MAX);
        assert_eq!(
            invoke(&mut context),
            (1, u32::MAX as usize),
            "overflow must take pre-probe fallback"
        );
        assert!(engine
            .interp
            .call_overflow
            .lookup(key, realm, u32::MAX)
            .is_none());
    }
    engine.interp.call_overflow.raw_sets = original_sets;
    // gc-heap owners carry no count: the probe leaves the callee's operand word and cell
    // intact, which a collection (and, in debug builds, its heap verifier) confirms.
    #[cfg(feature = "gc-heap")]
    {
        assert!(
            matches!(&operand[0].unpack(), Value::Obj(o) if crate::value::Gc::ptr_eq(o, &object)),
            "the probe must not consume the callee operand"
        );
        engine.interp.gc_collect();
        assert!(object.try_borrow_mut().is_ok(), "the callee cell is intact");
    }
}
