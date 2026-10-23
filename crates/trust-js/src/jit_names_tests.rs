//! Actual ARM64 execution proofs for captured References and bounded lexical addresses.
//! ECMA-262 e28783d5 GetIdentifierReference/GetValue/PutValue: no lookup replay after effects.
use super::*;
use crate::interpreter::{new_binding_layout_id, new_scope, Binding, Env};
use crate::value::ValueView;
use crate::{bytecode::Tier, Completion, Engine};
use std::cell::Cell;

fn compile(engine: &mut Engine, source: &str) -> (Rc<Chunk>, JitCode) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("source parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).expect("bytecode");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = super::super::compile(&chunk, &layout, &ilayout).expect("native code");
    (chunk, code)
}

fn scope(parent: Option<Env>, key: &str, value: f64, layout: u32) -> Env {
    let scope = new_scope(parent);
    scope
        .borrow_mut()
        .vars
        .insert(key, Binding::data(Value::Num(value), true, true));
    scope.borrow_mut().vars.publish_layout(layout);
    scope
}

fn run(engine: &mut Engine, chunk: &Rc<Chunk>, code: &JitCode, env: &Env) -> f64 {
    match (super::super::run(&mut engine.interp, chunk, code, env, Value::Undefined, &[]))
        .map(|__value| (__value).view())
    {
        Ok(crate::value::ValueView::Num(n)) => n,
        _ => panic!("expected numeric native result"),
    }
}

#[test]
fn native_deep_names_use_fresh_holder_layouts_and_live_flags() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
    let layouts = [
        new_binding_layout_id(),
        new_binding_layout_id(),
        new_binding_layout_id(),
    ];
    let holder = scope(None, "outer", 11.0, layouts[0]);
    let child = scope(
        Some(scope(Some(holder), "middle", 0.0, layouts[1])),
        "local",
        0.0,
        layouts[2],
    );
    assert_eq!(run(&mut engine, &chunk, &code, &child), 11.0); // cold fill
    let before = TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get);
    assert_eq!(run(&mut engine, &chunk, &code, &child), 11.0);
    let other = scope(None, "outer", 29.0, layouts[0]);
    let fresh = scope(
        Some(scope(Some(other), "middle", 0.0, layouts[1])),
        "local",
        0.0,
        layouts[2],
    );
    assert_eq!(run(&mut engine, &chunk, &code, &fresh), 29.0);
    assert_eq!(
        TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get),
        before + 2
    );
    other
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .initialized = false;
    assert!(super::super::run(
        &mut engine.interp,
        &chunk,
        &code,
        &fresh,
        Value::Undefined,
        &[]
    )
    .is_err());
    other
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .initialized = true;
    other.borrow_mut().vars.set_generation_for_test(u32::MAX);
    assert_eq!(run(&mut engine, &chunk, &code, &fresh), 29.0);
    assert_eq!(
        TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get),
        before + 2,
        "exhausted generation must take exact-record helper"
    );
}

#[test]
fn native_deep_global_names_load_store_update_without_helper_walks() {
    let mut engine = Engine::new();
    engine.eval("var outer=0;", false).unwrap();
    let (chunk, code) = compile(&mut engine, "function f(){outer=29;return ++outer;}");
    let child = new_scope(Some(new_scope(Some(engine.interp.global_env))));
    assert_eq!(run(&mut engine, &chunk, &code, &child), 30.0);
    let before = TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get);
    assert_eq!(run(&mut engine, &chunk, &code, &child), 30.0);
    assert!(TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get) >= before + 2);
}

#[test]
fn numeric_chains_and_loop_preloads_commit_packed_name_reads() {
    let mut engine = Engine::new();
    let (chain, chain_code) = compile(&mut engine, "function chain(){return outer*2+3;}");
    let (loop_chunk, loop_code) = compile(
        &mut engine,
        "function loop(n){var sum=0;for(var k=0;k<n;k++)sum+=outer;return sum;}",
    );
    let target = scope(None, "outer", 7.0, new_binding_layout_id());
    let env = new_scope(Some(new_scope(Some(target))));
    for _ in 0..8 {
        run(&mut engine, &chain, &chain_code, &env);
        super::super::run(
            &mut engine.interp,
            &loop_chunk,
            &loop_code,
            &env,
            Value::Undefined,
            &[Value::Num(20.0)],
        )
        .unwrap_or_else(|_| panic!("loop throws"));
    }
    let before = TEST_NATIVE_NAME_COMMITS.with(|counts| [counts[3].get(), counts[4].get()]);
    assert_eq!(run(&mut engine, &chain, &chain_code, &env), 17.0);
    assert!(
        TEST_NATIVE_NAME_COMMITS.with(|counts| counts[3].get()) > before[0],
        "numeric chain must finish its own packed Number read before the loop runs"
    );
    let preload_before = TEST_NATIVE_NAME_COMMITS.with(|counts| counts[4].get());
    let sum = super::super::run(
        &mut engine.interp,
        &loop_chunk,
        &loop_code,
        &env,
        Value::Undefined,
        &[Value::Num(20.0)],
    )
    .unwrap_or_else(|_| panic!("loop throws"));
    assert_eq!(sum.as_num_opt(), Some(140.0));
    let after = TEST_NATIVE_NAME_COMMITS.with(|counts| [counts[3].get(), counts[4].get()]);
    assert!(
        after[1] > preload_before,
        "loop preamble must finish its packed Number preload"
    );
}

#[test]
fn native_binding_store_keeps_its_old_input_domain_before_any_commit() {
    use crate::bytecode::NameIc;
    use crate::gc_core::{CARD_SHIFT, CARD_TABLE_OFFSET, PAGE_SIZE};
    use std::mem::{offset_of, size_of};
    let engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let owner = new_scope(None);
    let owner_word = Env::as_ptr(&owner) as usize;
    let cache = Cell::new(NameIc {
        env: owner_word,
        ..NameIc::EMPTY
    });
    let mut context = vec![0u64; size_of::<JitCtx>().div_ceil(8)];
    context[offset_of!(JitCtx, env_raw) / 8] = owner_word as u64;
    let card = ((owner_word & !(PAGE_SIZE - 1))
        + CARD_TABLE_OFFSET
        + ((owner_word & (PAGE_SIZE - 1)) >> CARD_SHIFT)) as *mut u8;
    let mut a = asm::Asm::new();
    let slow = a.new_label();
    let done = a.new_label();
    a.stp_pre(19, 20, -16);
    a.mov(19, 0);
    a.mov(20, 1);
    a.mov(14, 2);
    a.movz(7, 0, 0); // Binding owner, not a property descriptor.
    emit_store_name_value(
        &mut a,
        &layout,
        NameStoreOwner::Cache(cache.as_ptr() as usize),
        slow,
        done,
    );
    a.bind(slow);
    a.bind(done);
    a.mov(0, 20);
    a.ldp_post(19, 20, 16);
    a.ret();
    let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
    let code = ExecutableBuffer::from_bytes(&bytes).expect("Binding input guard probe");
    let probe: unsafe extern "C" fn(*const u64, *mut u64, *mut u64) -> usize =
        unsafe { std::mem::transmute(code.as_ptr()) };
    for (incoming, accepted) in [
        (crate::value::PACK_NULL, true),
        // ARM's old emit_exec_kind/decode_wide admitted EMPTY independently of init flags.
        (crate::value::PACK_EMPTY, true),
        (crate::value::PACK_LAZY_PROTO | 0x5a5a, false),
        (0xffff_0000_0000_0000, false),
    ] {
        let mut target = [0u64; 16];
        target[0] = 11.0f64.to_bits();
        let mutable = layout.binding_mutable - layout.binding_value;
        assert!(mutable < std::mem::size_of_val(&target));
        // Synthetic bytes, not a Rust Binding: initialize only the fields this leaf reads.
        unsafe { target.as_mut_ptr().cast::<u8>().add(mutable).write(1) };
        let original_target = target;
        let mut operands = [0x1122_3344_5566_7788, incoming, 0x8877_6655_4433_2211];
        let original_operands = operands;
        let top = unsafe { operands.as_mut_ptr().add(2) };
        unsafe { card.write(0) };
        let commits = TEST_NATIVE_NAME_COMMITS.with(|counts| counts[1].get());
        let returned_top = unsafe { probe(context.as_ptr(), top, target.as_mut_ptr()) };
        assert_eq!(operands, original_operands);
        if accepted {
            assert_eq!(target[0], incoming);
            assert_eq!(target[1..], original_target[1..]);
            assert_eq!(returned_top, unsafe { top.sub(1) } as usize);
            // A dirty card is any nonzero byte; the ARM address arithmetic writes 0xff.
            assert_ne!(unsafe { card.read() }, 0);
            assert_eq!(
                TEST_NATIVE_NAME_COMMITS.with(|counts| counts[1].get()),
                commits + 1
            );
        } else {
            assert_eq!(
                target, original_target,
                "guard before target commit: {incoming:#x}"
            );
            assert_eq!(returned_top, top as usize, "guard before operand pop");
            assert_eq!(unsafe { card.read() }, 0, "guard before card commit");
            assert_eq!(
                TEST_NATIVE_NAME_COMMITS.with(|counts| counts[1].get()),
                commits
            );
        }
    }
}

#[test]
fn native_references_hot_fragment_reuses_canonical_capture_and_value_paths() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(32);
    TEST_NATIVE_REFERENCES.with(|counts| counts.iter().for_each(|n| n.set(0)));
    crate::jit::TEST_OSR_ENTRIES.with(|n| n.set(0));
    crate::bytecode::TEST_REFERENCE_HELPERS.with(|n| n.set(0));
    assert!(matches!(engine.eval(
        "var total=0;for(var index=0;index<2000;index++)total+=index;total;", false),
        Ok(Completion::Value(value)) if value=="1999000"));
    assert!(crate::jit::TEST_OSR_ENTRIES.with(std::cell::Cell::get) > 0);
    TEST_NATIVE_REFERENCES.with(|counts| {
        for count in counts {
            assert!(count.get() > 100, "actual capture/load/store native proof");
        }
    });
    assert!(
        crate::bytecode::TEST_REFERENCE_HELPERS.with(std::cell::Cell::get) < 16,
        "warm canonical global reference must not repeat helper walks"
    );
}

#[test]
fn native_references_hot_fragment_preserves_effects_owners_and_saved_base() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        engine.interp.def_method(
            &engine.interp.global,
            "collectReferenceTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        let source = r#"
            var outer='outer', result, count=0;
            function test(){return eval(`var x=1;var k=0;
                while(k++<1500){x+=(k===1200?(delete x,3):1);}
                x;`);}
            result=test();
            var values=[undefined,null,true,1,-0,NaN,'a',Symbol(),{},1n];
            var held;
            for(var j=0;j<2000;j++){held=values[j%values.length];count++;if(j%200===0)collectReferenceTest();}
            [result,count,typeof held,outer].join('|');
        "#;
        assert!(
            matches!(engine.eval(source,false),Ok(Completion::Value(value)) if value=="1503|2000|bigint|outer"),
            "{tier:?}"
        );
    }
}

#[test]
fn native_name_max_guards_accept_independent_layout_without_reusing_holder_pointer() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
    let layout_id = new_binding_layout_id();
    let holder = scope(None, "outer", 41.0, layout_id);
    holder.borrow_mut().vars.set_generation_for_test(u32::MAX);
    holder.borrow_mut().vars.publish_layout(layout_id);
    let child = scope(
        Some(scope(Some(holder), "middle", 0.0, new_binding_layout_id())),
        "local",
        0.0,
        new_binding_layout_id(),
    );
    assert_eq!(run(&mut engine, &chunk, &code, &child), 41.0); // descriptor records MAX + layout
    let before = TEST_NATIVE_DEEP_NAMES.with(Cell::get);
    assert_eq!(run(&mut engine, &chunk, &code, &child), 41.0);
    assert_eq!(
        TEST_NATIVE_DEEP_NAMES.with(Cell::get),
        before + 1,
        "MAX disables pointer proofs, not the independently guarded fixed-layout slot"
    );
}

#[test]
fn native_reference_max_generation_and_shape_guards_keep_owned_bases_unchanged() {
    use crate::eval::PreparedReference;
    let engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let mut a = asm::Asm::new();
    let slow = a.new_label();
    a.mov(8, 0);
    emit_reference_target_ptr(&mut a, &layout, slow);
    a.mov(0, 14);
    a.ret();
    a.bind(slow);
    a.movz(0, 0, 0);
    a.ret();
    let words = a.finish();
    // Both scope-generation and property-shape checks use the exact 32-bit predicate.
    assert_eq!(words.iter().filter(|&&word| word == 0x3100_055f).count(), 2);
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let code = ExecutableBuffer::from_bytes(&bytes).expect("captured target probe");
    let probe: unsafe extern "C" fn(*const PreparedReference) -> usize =
        unsafe { std::mem::transmute(code.as_ptr()) };
    let holder = scope(None, "outer", 7.0, 0);
    for generation in [0, 1, u32::MAX - 1, u32::MAX] {
        holder.borrow_mut().vars.set_generation_for_test(generation);
        let reference = PreparedReference::scope(Rc::from("outer"), holder, true);
        assert_eq!(
            unsafe { probe(&reference) },
            if generation == u32::MAX {
                0
            } else {
                reference.binding as usize
            }
        );
        assert_eq!(reference.generation, generation);
    }
    let object = crate::value::Object::new(None);
    object.borrow_mut().props.insert(
        "outer",
        crate::value::Property::data(Value::Num(13.0), true, true, true),
    );
    let mut reference = PreparedReference::object(Rc::from("outer"), object, false);
    reference.slot = object.borrow().props.slot_of("outer").unwrap();
    reference.shape = object.borrow().props.shape();
    assert_ne!(unsafe { probe(&reference) }, 0);
    for shape in [0, u32::MAX] {
        reference.shape = shape;
        assert_eq!(unsafe { probe(&reference) }, 0);
        assert_eq!(reference.shape, shape);
    }
    // Owners carry no count: the probes leave both bases intact, which a collection
    // (and, in debug builds, its heap verifier) confirms.
    let mut engine = engine;
    engine.interp.gc_collect();
    assert!(holder.try_borrow_mut().is_ok() && object.try_borrow_mut().is_ok());
    assert!(matches!(
        (object.borrow().props.get("outer").map(|p| p.value())).map(|__value| (__value).view()),
        Some(ValueView::Num(n)) if n == 13.0
    ));
}
