//! Native name-cache dispatch proofs. ECMA-262 e28783d5 GetIdentifierReference/GetValue:
//! relocation of cold code must not weaken identity, with, generation, import or TDZ checks.
use super::*;
use crate::bytecode::NameIc;
use crate::interpreter::{new_binding_layout_id, new_scope, Binding, Env};
use std::cell::Cell;
use std::mem::{offset_of, size_of};

#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
struct ProbeResult {
    address: usize,
    packed: usize,
}

// Only generated code reads these aligned bytes. They are NOT a Rust JitCtx value: no invalid
// enums/owners are constructed or dropped. Every pointer field the probe reads is installed.
#[repr(align(16))]
struct ProbeContext([u8; size_of::<JitCtx>()]);

impl ProbeContext {
    fn new(engine: &crate::Engine, env: &Env) -> Self {
        let mut context = Self([0; size_of::<JitCtx>()]);
        context.word(offset_of!(JitCtx, env_raw), Rc::as_ptr(env) as usize);
        context.word(
            offset_of!(JitCtx, env_parent_raw),
            jit_env_parent_raw(env) as usize,
        );
        context.word(
            offset_of!(JitCtx, global_body),
            &*engine.interp.global.borrow() as *const crate::value::Object as usize,
        );
        context.word(
            offset_of!(JitCtx, genv),
            Rc::as_ptr(&engine.interp.global_env) as usize,
        );
        context
    }

    fn word(&mut self, offset: usize, value: usize) {
        self.0[offset..offset + size_of::<usize>()].copy_from_slice(&value.to_ne_bytes());
    }
}

fn probe_words(
    layout: &crate::value::JitLayout,
    cache: &Cell<NameIc>,
    packed_ok: bool,
) -> Vec<u32> {
    let mut a = asm::Asm::new();
    let slow = a.new_label();
    a.stp_pre(19, 30, -16);
    a.mov(19, 0);
    emit_name_ic_value_ptr(&mut a, layout, cache.as_ptr() as usize, slow, packed_ok);
    a.mov(0, 14);
    a.mov(1, 7);
    a.ldp_post(19, 30, 16);
    a.ret();
    a.bind(slow);
    a.movz(0, 0, 0);
    a.movz(1, 0, 0);
    a.ldp_post(19, 30, 16);
    a.ret();
    // The probe may be the shared out-of-line stub (reached with `bl`; lr is saved above).
    emit_shared_stubs(
        &mut a,
        &StubContext {
            layout,
            ilayout: None,
            direct: None,
        },
    );
    a.finish()
}

/// The validator body emitted inline at a site-shaped continuation (the shared stub wraps
/// exactly this body), for emission-order assertions.
fn body_words(layout: &crate::value::JitLayout, cache: &Cell<NameIc>, packed_ok: bool) -> Vec<u32> {
    let mut a = asm::Asm::new();
    let slow = a.new_label();
    a.stp_pre(19, 30, -16);
    a.mov(19, 0);
    a.mov_imm64(12, cache.as_ptr() as u64);
    emit_name_ic_value_ptr_body(&mut a, layout, slow, packed_ok);
    a.mov(0, 14);
    a.mov(1, 7);
    a.ldp_post(19, 30, 16);
    a.ret();
    a.bind(slow);
    a.movz(0, 0, 0);
    a.movz(1, 0, 0);
    a.ldp_post(19, 30, 16);
    a.ret();
    a.finish()
}

fn native_probe(
    layout: &crate::value::JitLayout,
    cache: &Cell<NameIc>,
    packed_ok: bool,
) -> ExecutableBuffer {
    let bytes: Vec<_> = probe_words(layout, cache, packed_ok)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    ExecutableBuffer::from_bytes(&bytes).expect("native cache probe")
}

fn probe(code: &ExecutableBuffer, context: &ProbeContext) -> ProbeResult {
    let run: unsafe extern "C" fn(*const u8) -> ProbeResult =
        unsafe { std::mem::transmute(code.as_ptr()) };
    // The context, live cache cell and all environments remain owned across this call. The
    // emitter performs no author calls/GC; its miss returns a sentinel, not the slow helper.
    unsafe { run(context.0.as_ptr()) }
}

fn binding_scope(parent: Option<Env>) -> Env {
    let env = new_scope(parent);
    env.borrow_mut()
        .vars
        .insert("outer", Binding::data(Value::Num(17.0), true, true));
    env
}

fn binding_ic(env: &Env) -> (NameIc, ProbeResult) {
    let scope = env.borrow();
    let binding = scope.vars.get("outer").unwrap();
    (
        NameIc {
            env: Rc::as_ptr(env) as usize,
            binding: binding as *const Binding as usize as u64,
            gen: scope.vars.generation(),
            act_gen: 0,
        },
        ProbeResult {
            address: &binding.value as *const Value as usize,
            packed: 0,
        },
    )
}

#[test]
fn name_dispatch_cmn32_max_predicate_preserves_source_and_ignores_upper_bits() {
    let mut a = asm::Asm::new();
    a.cmn_imm_w(0, 1);
    a.cset_w(2, C_EQ);
    a.str_imm(0, 1, 0); // The WZR destination must leave the entire incoming X0 unchanged.
    a.mov_w(0, 2);
    a.ret();
    let words = a.finish();
    assert_eq!(words[0], 0x3100_041f, "ADDS WZR,W0,#1, unshifted immediate");
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let code = ExecutableBuffer::from_bytes(&bytes).expect("CMN32 probe");
    let run: unsafe extern "C" fn(u64, *mut u64) -> u32 =
        unsafe { std::mem::transmute(code.as_ptr()) };
    let check = |input: u64| {
        let mut preserved = 0;
        assert_eq!(
            unsafe { run(input, &mut preserved) },
            u32::from(input as u32 == u32::MAX)
        );
        assert_eq!(preserved, input);
    };
    for lane in 0..=u16::MAX {
        for low in [0, 1, 0x7fff, 0xfffe, 0xffff] {
            let input = (u64::from(lane) << 16) | low;
            check(input);
            check(input | 0x9abc_def0_0000_0000);
        }
    }
    let mut random = 0x3186_541e_d5a9_721f_u64;
    for _ in 0..4096 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        check(random);
    }
}

#[test]
fn name_dispatch_fixed_layout_uses_current_wide_activation_and_live_flags() {
    use crate::interpreter::VarMap;
    let engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(layout.scope_small_valid);
    for indexed in [false, true] {
        let identity = new_binding_layout_id();
        let shared = std::cell::OnceCell::new();
        let make = || {
            let env = new_scope(None);
            env.borrow_mut().vars = VarMap::from_fixed_bindings(
                (0..24)
                    .map(|n| {
                        (
                            Rc::<str>::from(format!("v{n}")),
                            Binding::data(Value::Num(n as f64), true, true),
                        )
                    })
                    .collect(),
                None,
            );
            if indexed {
                env.borrow_mut().vars.share_layout(&shared);
            }
            env.borrow_mut().vars.publish_layout(identity);
            env
        };
        let first = make();
        let second = make();
        let cache = Cell::new(NameIc {
            env: crate::bytecode::FIXED_NAME_IC,
            binding: 23,
            gen: identity,
            act_gen: 0,
        });
        let code = native_probe(&layout, &cache, true);
        for env in [&first, &second] {
            let expected = &env.borrow().vars.get("v23").unwrap().value as *const Value as usize;
            assert_eq!(
                probe(&code, &ProbeContext::new(&engine, env)),
                ProbeResult {
                    address: expected,
                    packed: 0
                }
            );
        }
        let context = ProbeContext::new(&engine, &second);
        second.borrow_mut().vars.get_mut("v23").unwrap().initialized = false;
        assert_eq!(probe(&code, &context).address, 0);
        second.borrow_mut().vars.get_mut("v23").unwrap().initialized = true;
        second
            .borrow_mut()
            .vars
            .get_mut("v23")
            .unwrap()
            .set_import_reference(Some((first.clone(), "v23".into())));
        assert_eq!(probe(&code, &context).address, 0);
        second
            .borrow_mut()
            .vars
            .get_mut("v23")
            .unwrap()
            .set_import_reference(None);
        cache.set(NameIc {
            binding: 24,
            ..cache.get()
        });
        assert_eq!(probe(&code, &context).address, 0);
        cache.set(NameIc {
            binding: 23,
            gen: 0,
            ..cache.get()
        });
        assert_eq!(probe(&code, &context).address, 0);
        cache.set(NameIc {
            gen: identity,
            ..cache.get()
        });
        second.borrow_mut().with_obj = Some(Value::Obj(engine.interp.global.clone()));
        assert_eq!(probe(&code, &context).address, 0);
        second.borrow_mut().with_obj = None;
        second.borrow_mut().vars.remove("v0");
        assert_eq!(probe(&code, &context).address, 0);
    }
}

#[test]
fn name_dispatch_emission_places_deep_island_after_ordinary_mode_proofs() {
    let engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let cache = Cell::new(NameIc::EMPTY);
    let words = body_words(&layout, &cache, true);
    let instruction = |emit: fn(&mut asm::Asm)| {
        let mut a = asm::Asm::new();
        emit(&mut a);
        a.finish()[0]
    };
    let position = |word| {
        words
            .iter()
            .position(|&candidate| candidate == word)
            .expect("instruction")
    };
    let direct = position(instruction(|a| a.cmp_reg_x(9, 10)));
    let global_identity = position(instruction(|a| a.cmp_reg_x(17, 10)));
    let global_generation = position(instruction(|a| a.cmp_reg_w(11, 13)));
    let deep = position(instruction(|a| {
        a.cmp_imm_x(10, crate::bytecode::lexical_cache::DEEP_NAME_IC as u32)
    }));
    assert!(
        direct < global_identity && global_identity < global_generation && global_generation < deep
    );
    assert_eq!(words[direct + 1] & 0xff00_001f, 0x5400_0000 | C_EQ);
    let displacement = ((words[direct + 1] << 8) as i32) >> 13; // signed imm19 in words
    assert!(
        (direct + 1) as isize + displacement as isize > deep as isize,
        "ordinary Binding branch must jump past the entire cold island"
    );
    let return_value = position(instruction(|a| a.mov(0, 14)));
    assert_eq!(
        words[return_value - 1],
        instruction(|a| a.movz(7, 0, 0)),
        "Binding success falls through without a branch over deep code"
    );
    assert!(position(instruction(|a| a.cmn_imm_w(13, 1))) < direct);
    assert!(words.contains(&instruction(|a| a.cmn_imm_w(15, 1))));
}

#[test]
fn name_dispatch_native_binding_preserves_identity_with_tdz_import_and_exhaustion_checks() {
    let engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let env = binding_scope(None);
    let context = ProbeContext::new(&engine, &env);
    let (ic, expected) = binding_ic(&env);
    let cache = Cell::new(ic);
    let code = native_probe(&layout, &cache, true);
    assert_eq!(probe(&code, &context), expected);
    cache.set(NameIc::EMPTY);
    assert_eq!(probe(&code, &context).address, 0);
    let other = binding_scope(None);
    cache.set(binding_ic(&other).0);
    assert_eq!(probe(&code, &context).address, 0);
    cache.set(ic);
    env.borrow_mut().with_obj = Some(Value::Obj(engine.interp.global.clone()));
    assert_eq!(probe(&code, &context).address, 0);
    env.borrow_mut().with_obj = None;
    env.borrow_mut().vars.get_mut("outer").unwrap().initialized = false;
    assert_eq!(probe(&code, &context).address, 0);
    env.borrow_mut().vars.get_mut("outer").unwrap().initialized = true;
    env.borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .set_import_reference(Some((other, "outer".into())));
    assert_eq!(probe(&code, &context).address, 0);
    env.borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .set_import_reference(None);
    for generation in [0, 1, u32::MAX - 1, u32::MAX] {
        env.borrow_mut().vars.set_generation_for_test(generation);
        cache.set(NameIc {
            gen: generation,
            ..ic
        });
        assert_eq!(
            probe(&code, &context).address,
            if generation == u32::MAX {
                0
            } else {
                expected.address
            }
        );
    }
    env.borrow_mut().vars.set_generation_for_test(7);
    cache.set(NameIc { gen: 6, ..ic });
    assert_eq!(probe(&code, &context).address, 0);
}

#[test]
fn name_dispatch_native_global_checks_identity_before_generation_without_losing_generation() {
    let mut engine = crate::Engine::new();
    // Script var is non-configurable (GlobalDeclarationInstantiation), so it
    // cannot exercise a later legal data-to-accessor descriptor transition.
    assert!(matches!(engine.eval(
        "Object.defineProperty(globalThis,'dispatchGlobal',{value:23,writable:true,configurable:true})",
        false,
    ), Ok(crate::Completion::Value(_))));
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let env = engine.interp.global_env.clone();
    let context = ProbeContext::new(&engine, &env);
    let (slot, shape, expected) = {
        let global = engine.interp.global.borrow();
        let slot = global.props.slot_of("dispatchGlobal").unwrap();
        (
            slot,
            global.props.shape(),
            global.props.property_at(slot).unwrap() as *const crate::value::Property as usize
                + layout.property_value,
        )
    };
    let ic = NameIc {
        env: Rc::as_ptr(&env) as usize | 1,
        binding: (u64::from(shape) << 32) | slot as u64,
        gen: env.borrow().vars.generation(),
        act_gen: 0,
    };
    let cache = Cell::new(ic);
    let code = native_probe(&layout, &cache, true);
    assert_eq!(
        probe(&code, &context),
        ProbeResult {
            address: expected,
            packed: 1
        }
    );
    let no_packed = native_probe(&layout, &cache, false);
    assert_eq!(probe(&no_packed, &context).address, 0);
    cache.set(NameIc {
        gen: ic.gen.wrapping_add(1),
        ..ic
    });
    assert_eq!(probe(&code, &context).address, 0);
    cache.set(NameIc {
        gen: u32::MAX,
        ..ic
    });
    assert_eq!(probe(&code, &context).address, 0);
    let other = binding_scope(None);
    cache.set(NameIc {
        env: Rc::as_ptr(&other) as usize | 1,
        ..ic
    });
    assert_eq!(probe(&code, &context).address, 0); // unrelated identity is never dereferenced
    cache.set(ic);
    env.borrow_mut()
        .vars
        .set_generation_for_test(ic.gen.wrapping_add(1));
    assert_eq!(
        probe(&code, &context).address,
        0,
        "identity match must still guard current generation"
    );
    env.borrow_mut().vars.set_generation_for_test(ic.gen);
    cache.set(NameIc {
        binding: (u64::from(shape.wrapping_add(1)) << 32) | slot as u64,
        ..ic
    });
    assert_eq!(
        probe(&code, &context).address,
        0,
        "global layout must still match"
    );
    cache.set(ic);
    assert!(matches!(
        engine.eval(
            "Object.defineProperty(globalThis,'dispatchGlobal',{get(){return 31;}})",
            false,
        ),
        Ok(crate::Completion::Value(_))
    ));
    assert_eq!(
        probe(&code, &context).address,
        0,
        "live accessor must not become a data pointer"
    );
}

#[test]
fn name_dispatch_native_depth_one_keeps_layout_and_parent_generation_distinct() {
    let engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    for fixed_layout in [false, true] {
        let parent = binding_scope(None);
        let child = new_scope(Some(parent.clone()));
        child
            .borrow_mut()
            .vars
            .insert("local", Binding::data(Value::Num(1.0), true, true));
        let layout_id = new_binding_layout_id();
        if fixed_layout {
            child.borrow_mut().vars.publish_layout(layout_id);
        }
        let context = ProbeContext::new(&engine, &child);
        let (base, expected) = binding_ic(&parent);
        let ic = NameIc {
            env: base.env | if fixed_layout { 6 } else { 2 },
            act_gen: if fixed_layout {
                layout_id
            } else {
                child.borrow().vars.generation()
            },
            ..base
        };
        let cache = Cell::new(ic);
        let code = native_probe(&layout, &cache, true);
        assert_eq!(probe(&code, &context), expected);
        parent.borrow_mut().with_obj = Some(Value::Obj(engine.interp.global.clone()));
        assert_eq!(probe(&code, &context).address, 0);
        parent.borrow_mut().with_obj = None;
        child.borrow_mut().vars.set_generation_for_test(u32::MAX);
        if fixed_layout {
            child.borrow_mut().vars.publish_layout(layout_id);
        }
        cache.set(NameIc {
            act_gen: if fixed_layout { layout_id } else { u32::MAX },
            ..ic
        });
        assert_eq!(
            probe(&code, &context).address,
            if fixed_layout { expected.address } else { 0 }
        );
        parent.borrow_mut().vars.set_generation_for_test(u32::MAX);
        cache.set(NameIc {
            gen: u32::MAX,
            ..cache.get()
        });
        assert_eq!(
            probe(&code, &context).address,
            0,
            "parent pointer generation is not a layout token"
        );
    }
}

#[test]
fn name_dispatch_native_deep_hits_ignore_shallow_generation_and_follow_live_parents() {
    let mut engine = crate::Engine::new();
    let statements = crate::parser::parse_script("function f(){return outer;}", false)
        .ok()
        .expect("source");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).unwrap();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = super::compile(&chunk, &layout, &ilayout).unwrap();
    let holder = binding_scope(None);
    let middle = new_scope(Some(holder));
    let child = new_scope(Some(middle.clone()));
    child
        .borrow_mut()
        .vars
        .insert("local", Binding::data(Value::Num(9.0), true, true));
    assert_ne!(
        child.borrow().vars.generation(),
        0,
        "DEEP cache publishes gen=0, unlike this scope"
    );
    let run = |engine: &mut crate::Engine| {
        super::run(
            &mut engine.interp,
            &chunk,
            &code,
            &child,
            Value::Undefined,
            &[],
        )
    };
    assert!(matches!(run(&mut engine), Ok(Value::Num(17.0))));
    let before = names::TEST_NATIVE_DEEP_NAMES.with(Cell::get);
    assert!(matches!(run(&mut engine), Ok(Value::Num(17.0))));
    assert_eq!(names::TEST_NATIVE_DEEP_NAMES.with(Cell::get), before + 1);
    let replacement = binding_scope(None);
    replacement
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .value = Value::Num(29.0);
    middle.borrow_mut().parent = Some(replacement);
    assert!(
        matches!(run(&mut engine), Ok(Value::Num(29.0))),
        "cache miss must follow current links"
    );
}
