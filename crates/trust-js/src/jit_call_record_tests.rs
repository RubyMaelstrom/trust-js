//! A call site's record of its first code-keyed hit (P3 stage 2, step 15), on the real call
//! template and its `CallSecondary` stub: the code hint starts at `CALL_SITE_NO_CODE_HIT`, which
//! the hinted-way test cannot hit, so the first code-keyed hit takes the full code probe, which
//! writes the hint; identity hits and the hinted-way test's hits write nothing; and the hinted-way
//! test of the starting value reads nothing past the site.
use super::*;
use crate::bytecode::{CallIc, CallSite, Tier, CALL_IC_EPOCH, CALL_SITE_NO_CODE_HIT};
use crate::value::{ObjectKind, Value};
use std::mem::{offset_of, size_of};
use std::sync::atomic::Ordering::Relaxed;

fn executable(words: Vec<u32>) -> ExecutableBuffer {
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    ExecutableBuffer::from_bytes(&bytes).expect("the probe's code")
}

/// Whether this process runs the test `name` of this module alone (a child of the test binary,
/// which runs the body); otherwise runs it so and checks that it passed. The call-cache epoch is
/// process-wide, and another test's discard moving it would stale the ways these tests fill.
fn alone(name: &str) -> bool {
    const CHILD: &str = "TRUST_JS_TEST_CALL_RECORD_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("jit::call_record_tests::{name}"),
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // A read past the guard page ends the child with a signal, which its status names.
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "the child {}:\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

/// The closures the tests call: three of one function literal, and a function of another.
const CLOSURES: &str = "function recordCaller(f) { return f(); }\n\
                        function make(k) { return function () { return k; }; }\n\
                        function other() { return 0; }\n\
                        var a = make(1), b = make(2), c = make(3);\n\
                        recordCaller(other);";

/// `recordCaller`'s bytecode chunk, `CLOSURES` evaluated.
fn caller_chunk(engine: &mut crate::Engine) -> Rc<Chunk> {
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    assert!(matches!(
        engine.eval(CLOSURES, false),
        Ok(crate::Completion::Value(_))
    ));
    let object = global(engine, "recordCaller");
    let object = object.as_obj().unwrap();
    let b = object.borrow();
    let ObjectKind::User(user) = &b.kind else {
        panic!("a user function")
    };
    user.func
        .runtime()
        .code
        .get()
        .and_then(Option::as_ref)
        .expect("bytecode caller")
        .clone()
}

fn global(engine: &mut crate::Engine, name: &str) -> Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("bound name")
}

/// The global closure `name` and the way a fill at `epoch` would make for it.
fn closure(engine: &mut crate::Engine, name: &str, epoch: u32) -> (Value, CallIc) {
    let value = global(engine, name);
    let object = value.as_obj().expect("a function");
    let b = object.borrow();
    let ObjectKind::User(user) = &b.kind else {
        panic!("{name} is a user function")
    };
    let ic = CallIc {
        callee: crate::value::Gc::as_ptr(&object) as usize,
        env: Env::as_ptr(&user.env),
        func: Rc::as_ptr(&user.func),
        epoch,
        global_env: Env::as_ptr(&engine.interp.global_env) as usize,
        realm: user.realm,
        ..CallIc::EMPTY
    };
    (value, ic)
}

// These aligned words are a machine-code input only, never a Rust JitCtx.
#[repr(align(16))]
struct ProbeContext([usize; size_of::<JitCtx>().div_ceil(size_of::<usize>())]);

impl ProbeContext {
    fn new(engine: &mut crate::Engine) -> ProbeContext {
        let mut context = ProbeContext([0; size_of::<JitCtx>().div_ceil(size_of::<usize>())]);
        context.0[offset_of!(JitCtx, interp) / size_of::<usize>()] =
            &mut *engine.interp as *mut _ as usize;
        context.0[offset_of!(JitCtx, genv) / size_of::<usize>()] =
            Env::as_ptr(&engine.interp.global_env) as usize;
        context.0[offset_of!(JitCtx, call_ic_epoch) / size_of::<usize>()] =
            CALL_IC_EPOCH.as_ptr() as usize;
        context
    }
}

/// A helper that records its route and the incoming x15 (the way's tagged index) in the
/// context's first two words, and returns as a completed call keeping the operand top.
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

/// The real call template of `chunk`'s zero-argument call with the inline probe alone (no direct
/// sequence), the probe after way 0 in the chunk's `CallSecondary` stub: a hit calls H_CALL_HIT,
/// a miss H_CALL.
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
    emit_call_inline(
        &mut a, chunk, &layout, &ilayout, pc, 524288, false, false, unwind, finish,
    );
    a.bind(unwind);
    a.bind(finish);
    a.mov(0, 20);
    a.ldp_post(21, 30, 16);
    a.ldp_post(19, 20, 16);
    a.ret();
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
    (executable(a.finish()), cache)
}

#[test]
fn a_sites_first_code_keyed_hit_takes_the_full_probe_which_writes_the_hint() {
    if !alone("a_sites_first_code_keyed_hit_takes_the_full_probe_which_writes_the_hint") {
        return;
    }
    let mut engine = crate::Engine::new();
    let chunk = caller_chunk(&mut engine);
    let (code, cache) = call_probe(&mut engine, &chunk);
    let site = chunk.test_call_site(cache);
    let epoch = CALL_IC_EPOCH.load(Relaxed);
    let (a, way_a) = closure(&mut engine, "a", epoch);
    let (b, way_b) = closure(&mut engine, "b", epoch);
    let (c, _) = closure(&mut engine, "c", epoch);
    let (other, _) = closure(&mut engine, "other", epoch);
    for way in &site.entries {
        way.set(CallIc::EMPTY);
    }
    site.entries[0].set(way_a);
    let mut context = ProbeContext::new(&mut engine);
    let slow = helper(1);
    let hit = helper(2);
    let mut helpers = [0usize; N_HELPERS];
    helpers[H_CALL] = slow.as_ptr() as usize;
    helpers[H_CALL_HIT] = hit.as_ptr() as usize;
    let run: unsafe extern "C" fn(*mut usize, *mut Value, *const usize) -> *mut Value =
        unsafe { std::mem::transmute(code.as_ptr()) };
    let mut invoke = |callee: &Value| {
        let mut operand = [*callee];
        let top = unsafe { operand.as_mut_ptr().add(1) };
        context.0[0] = 0;
        context.0[1] = 0;
        assert_eq!(
            unsafe { run(context.0.as_mut_ptr(), top, helpers.as_ptr()) },
            top
        );
        (context.0[0], context.0[1])
    };
    let code_hit = CALL_CODE_HIT as usize;
    assert_eq!(site.code_hint.get(), CALL_SITE_NO_CODE_HIT);
    // An identity hit, way 0's in line: no record.
    assert_eq!(invoke(&a), (2, 0));
    assert!(!site.met_another_closure());
    // A callee of another function: the hinted-way test of the starting value, every way and
    // the full probe miss it, to the generic call; no record.
    assert_eq!(invoke(&other).0, 1);
    assert!(!site.met_another_closure());
    // The first code-keyed hit: the hinted-way test of the starting value fails, and the full
    // probe hits way 0 and writes the hint, the site's record.
    assert_eq!(invoke(&b), (2, code_hit));
    assert_eq!(site.code_hint.get(), 0);
    assert!(site.met_another_closure());
    // The hinted-way test hits first and writes nothing: with closures of the function in ways
    // 0 and 1 and the hint at 1, `c` hits way 1 (the full probe would find way 0).
    site.entries[1].set(way_b);
    site.code_hint.set(1);
    assert_eq!(invoke(&c), (2, code_hit | 1));
    assert_eq!(site.code_hint.get(), 1);
    // From the starting value, the same call takes the full probe: way 0's hit, written.
    site.code_hint.set(CALL_SITE_NO_CODE_HIT);
    assert_eq!(invoke(&c), (2, code_hit));
    assert_eq!(site.code_hint.get(), 0);
    // Identity hits on ways 0 and 1 write nothing.
    site.code_hint.set(CALL_SITE_NO_CODE_HIT);
    assert_eq!(invoke(&a), (2, 0));
    assert_eq!(invoke(&b), (2, 1));
    assert_eq!(site.code_hint.get(), CALL_SITE_NO_CODE_HIT);
}

#[cfg(target_os = "linux")]
#[test]
fn the_hinted_test_of_the_starting_value_reads_nothing_past_the_site() {
    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    const PROT_NONE: i32 = 0;
    const PROT_READ_WRITE: i32 = 3;
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x22;
    // A multiple of every page size Linux has on ARM64.
    const PAGE: usize = 1 << 16;
    if !alone("the_hinted_test_of_the_starting_value_reads_nothing_past_the_site") {
        return;
    }
    let mut engine = crate::Engine::new();
    caller_chunk(&mut engine);
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let epoch = CALL_IC_EPOCH.load(Relaxed);
    let (_, way_a) = closure(&mut engine, "a", epoch);
    let (b, _) = closure(&mut engine, "b", epoch);
    let (other, _) = closure(&mut engine, "other", epoch);
    // The site ends where its page does, and the next page cannot be read: a read past the site
    // faults.
    let base = unsafe {
        mmap(
            std::ptr::null_mut(),
            2 * PAGE,
            PROT_READ_WRITE,
            MAP_PRIVATE_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(base as isize, -1);
    assert_eq!(unsafe { mprotect(base.add(PAGE), PAGE, PROT_NONE) }, 0);
    let at = unsafe { base.add(PAGE - size_of::<CallSite>()) } as *mut CallSite;
    assert!(at.is_aligned());
    unsafe { at.write(CallSite::empty()) };
    let site = unsafe { &*at };
    site.entries[0].set(way_a);
    // The probe after way 0 (`emit_call_ways`), as the `CallSecondary` stub runs it: x0 = the
    // context, x1 = the site, x2 = the callee's word; it answers x15 on a hit, 0xffff on a miss.
    let mut m = asm::Asm::new();
    m.stp_pre(19, 20, -16);
    m.stp_pre(21, 30, -16);
    m.mov(19, 0);
    m.mov(12, 1);
    m.mov(9, 2);
    m.mov_imm64(16, crate::value::PACK_OBJ);
    m.logic_x(2, 10, 9, 16);
    m.ldr_imm(11, 19, CALL_IC_EPOCH_OFF);
    m.ldr_w_imm(15, 11, 0);
    m.ldr_imm(17, 19, offset_of!(JitCtx, genv) as u32);
    let hit = m.new_label();
    let miss = m.new_label();
    let out = m.new_label();
    emit_call_ways(&mut m, &layout, &ilayout, hit, miss);
    m.bind(hit);
    m.mov(0, 15);
    m.b(out);
    m.bind(miss);
    m.movz(0, 0xffff, 0);
    m.bind(out);
    m.ldp_post(21, 30, 16);
    m.ldp_post(19, 20, 16);
    m.ret();
    let code = executable(m.finish());
    let run: unsafe extern "C" fn(*mut usize, *const CallSite, u64) -> u64 =
        unsafe { std::mem::transmute(code.as_ptr()) };
    let mut context = ProbeContext::new(&mut engine);
    let mut invoke = |callee: &Value| unsafe { run(context.0.as_mut_ptr(), at, callee.bits()) };
    // A callee no way matches: the starting value's test, every way and the probes, no fault.
    assert_eq!(invoke(&other), 0xffff);
    assert_eq!(site.code_hint.get(), CALL_SITE_NO_CODE_HIT);
    // A closure of way 0's function: the full probe's hit, written.
    assert_eq!(invoke(&b), u64::from(CALL_CODE_HIT));
    assert_eq!(site.code_hint.get(), 0);
    // The hinted-way test of way 0 hits it again.
    assert_eq!(invoke(&b), u64::from(CALL_CODE_HIT));
    unsafe {
        at.drop_in_place();
        assert_eq!(munmap(base, 2 * PAGE), 0);
    }
}
