use super::*;
use crate::bytecode::{native_deopt::ENTRIES, Tier};
use crate::value::{Callable, Value as JsValue};
use crate::{Completion, Engine};
use std::rc::Rc;

fn child(name: &str, extra: &[(&str, &str)]) -> bool {
    const CONTROL: &str = "TRUST_JS_TEST_BORROWED_OPERANDS_CHILD";
    if std::env::var(CONTROL).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        &format!("jit::optimizing::borrowed_operands::tests::{name}"),
        "--nocapture",
        "--test-threads=1",
    ]);
    for (key, _) in std::env::vars_os() {
        if key.as_encoded_bytes().starts_with(b"TRUST_JS_")
            || key.as_encoded_bytes().starts_with(b"LUMEN_JS_")
        {
            command.env_remove(key);
        }
    }
    command
        .env(CONTROL, name)
        .env("TRUST_JS_OPT_JIT", "0")
        .env("TRUST_JS_INLINE_AT", "0")
        .env("TRUST_JS_OPT_JIT_BORROWED_OPERANDS", "1")
        .env_remove("LD_PRELOAD")
        .envs(extra.iter().copied());
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed; 0 failed;")
    );
    true
}
fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}:{message}"),
    }
}
fn function(engine: &mut Engine) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var("subject", &env)
        .ok()
        .expect("subject binding");
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary fixture")
    };
    user.func.clone()
}
fn certificate(chunk: &Chunk, deopt: Option<usize>, facts: bool) -> Option<Plan> {
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let values = facts
        .then(|| value_facts::ValuePlan::build(chunk, &cfg, &stack))
        .flatten();
    Plan::build(
        chunk,
        &cfg,
        &stack,
        None,
        None,
        deopt,
        NativeEntryKind::FreshFrame,
        values.as_ref(),
    )
}
fn install(engine: &mut Engine, deopt: Option<usize>) -> (Rc<Chunk>, Rc<JitCode>) {
    install_mode(engine, deopt, "execution")
}
fn install_mode(engine: &mut Engine, deopt: Option<usize>, mode: &str) -> (Rc<Chunk>, Rc<JitCode>) {
    let definition = function(engine);
    let chunk = bytecode::compile(&definition).unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
    let code = diagnostics::with_test_mode(mode, || {
        Rc::new(compile_with_deopt(&chunk, &values, &layout, deopt).unwrap())
    });
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(definition.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    (chunk, code)
}
fn native(engine: &mut Engine, code: &JitCode, source: &str, expected: &str) {
    code.residency.referenced.set(0);
    assert_eq!(eval(engine, source), expected);
    assert!(code.residency.referenced.get() > 0, "actual native entry");
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}
fn cross(setup: &str, source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut e = Engine::new();
        e.set_tier(tier);
        e.set_tier_threshold(0);
        eval(&mut e, setup);
        assert_eq!(eval(&mut e, source), expected, "{tier:?}");
    }
}
fn word(tag: u64, header: &Cell<usize>) -> u64 {
    let address = header as *const _ as usize as u64;
    assert_eq!(address & !PAYLOAD, 0);
    tag | address
}

#[test]
fn atomic_batch_alias_overflow_and_last_owner_failures_never_partially_mutate() {
    for tag in [PACK_OBJ, PACK_STR, PACK_SYM, PACK_BIGINT] {
        for first in [0, 1, 2, usize::MAX - 1, usize::MAX] {
            for second in [0, 1, 2, usize::MAX - 1, usize::MAX] {
                let a = Cell::new(first);
                let b = Cell::new(second);
                let input = [word(tag, &a), word(tag, &b)];
                let status = unsafe { transaction::<true>(input.as_ptr(), 2, 3) };
                let good = first != 0 && first != usize::MAX && second != 0 && second != usize::MAX;
                assert_eq!(status & SUCCESS != 0, good);
                assert_eq!(a.get(), if good { first + 1 } else { first });
                assert_eq!(b.get(), if good { second + 1 } else { second });
                a.set(first);
                b.set(second);
                let status = unsafe { transaction::<false>(input.as_ptr(), 2, 1) };
                let good = first != 0 && first != usize::MAX && second > 1;
                assert_eq!(status & SUCCESS != 0, good);
                assert_eq!(a.get(), if good { first + 1 } else { first });
                assert_eq!(b.get(), if good { second - 1 } else { second });
            }
        }
        for initial in [0, 1, 2, usize::MAX - 1, usize::MAX] {
            let owner = Cell::new(initial);
            let same = word(tag, &owner);
            let words = [same, same, same];
            let status = unsafe { transaction::<true>(words.as_ptr(), 3, 3) };
            let good = initial != 0 && initial.checked_add(2).is_some();
            assert_eq!(status & SUCCESS != 0, good);
            assert_eq!(owner.get(), if good { initial + 1 } else { initial });
        }
    }
    let a = Cell::new(2);
    let input = [word(PACK_OBJ, &a), crate::value::PACK_LAZY_PROTO | 5];
    assert_eq!(unsafe { transaction::<true>(input.as_ptr(), 2, 3) }, 0);
    assert_eq!(a.get(), 2);
    assert_eq!(unsafe { transaction::<false>(input.as_ptr(), 9, 0) }, 0);
    assert_eq!(a.get(), 2);
}

#[test]
fn counted_and_uncounted_immediates_do_not_access_heap_or_invent_transfers() {
    let words = [
        crate::value::PACK_UNDEFINED,
        crate::value::PACK_NULL,
        crate::value::PACK_EMPTY,
        PACK_BOOL,
        0,
        (-0.0f64).to_bits(),
        f64::INFINITY.to_bits(),
        PACK_CANON_NAN,
    ];
    assert_eq!(
        unsafe { transaction::<true>(words.as_ptr(), 8, 0xff) },
        SUCCESS
    );
    assert_eq!(
        unsafe { transaction::<false>(words.as_ptr(), 8, 0) },
        SUCCESS
    );
    assert_eq!(unsafe { transaction::<false>(words.as_ptr(), 0, 0) }, 0);
    let owner = Cell::new(3);
    let words = [word(PACK_STR, &owner), word(PACK_STR, &owner)];
    assert_eq!(
        unsafe { transaction::<true>(words.as_ptr(), 2, 1) },
        SUCCESS | (1 << RETAINS_SHIFT) | (1 << RELEASES_SHIFT)
    );
    assert_eq!(owner.get(), 3);
}

#[test]
fn dirty_ssa_copies_preserve_owners_types_and_native_results() {
    let name = "dirty_ssa_copies_preserve_owners_types_and_native_results";
    if child(name, &[]) {
        return;
    }
    let setup = "function subject(x){var y=x;var z=y;return z;}var o={k:7};var s=Symbol('s');var text='v'.repeat(300).slice(1);";
    let source = "[subject(o)===o,subject(s)===s,subject(text)===text,subject(7n)===7n,Object.is(subject(-0),-0),Number.isNaN(subject(NaN))].join('|')";
    cross(setup, source, "true|true|true|true|true|true");
    let mut e = Engine::new();
    eval(&mut e, setup);
    let (_, code) = install(&mut e, None);
    assert!(TEST_SELECTED.with(|sites| sites.borrow().len()) >= 2);
    TEST_TRANSACTIONS.with(|calls| calls.set(0));
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native(&mut e, &code, source, "true|true|true|true|true|true");
    assert!(TEST_TRANSACTIONS.with(Cell::get) >= 12);
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    e.interp.gc_collect();
    native(&mut e, &code, "subject(text).length", "299");
}

#[test]
fn last_destination_owner_recovers_before_original_load_exactly_once() {
    let name = "last_destination_owner_recovers_before_original_load_exactly_once";
    if child(name, &[]) {
        return;
    }
    let setup = "function subject(x){var y={p:1};y=x;return y;}";
    cross(setup, "subject({k:3}).k", "3");
    let mut e = Engine::new();
    eval(&mut e, setup);
    let (chunk, code) = install(&mut e, None);
    let selected = TEST_SELECTED.with(|sites| sites.borrow().clone());
    assert_eq!(selected.len(), 1);
    let start = selected[0].0;
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    TEST_TRANSACTIONS.with(|calls| calls.set(0));
    native(&mut e, &code, "subject({k:3}).k", "3");
    ENTRIES.with(|entries| assert_eq!(&*entries.borrow(), &[(start, stack.at[start].depth, 0)]));
    assert_eq!(TEST_TRANSACTIONS.with(Cell::get), 0);
}

fn collect(
    i: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    i.gc_collect();
    Ok(JsValue::Undefined)
}
#[test]
fn aliasing_call_arguments_become_owners_before_gc_and_throwing_callback() {
    let name = "aliasing_call_arguments_become_owners_before_gc_and_throwing_callback";
    if child(name, &[]) {
        return;
    }
    let setup = "var trace='';function finish(a,b,c){forceCollection();trace+=(a===b)+':'+c.k;throw Error('stop');}function subject(x,y){var dirty=x;try{return finish(dirty,x,y);}catch(e){return e.message+':'+trace;}finally{trace+='F';}}var value={k:9};var other={k:4};";
    let mut e = Engine::new();
    let global = e.interp.global.clone();
    e.interp.def_method(&global, "forceCollection", 0, collect);
    eval(&mut e, setup);
    let (_, code) = install(&mut e, None);
    let sites = TEST_SELECTED.with(|sites| sites.borrow().clone());
    assert!(sites.iter().any(|(a, b)| b - a >= 2));
    TEST_TRANSACTIONS.with(|calls| calls.set(0));
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native(
        &mut e,
        &code,
        "subject(value,other)+'|'+trace",
        "stop:true:4|true:4F",
    );
    assert!(TEST_TRANSACTIONS.with(Cell::get) >= 2);
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    assert!(code
        .optimizing_diagnostics
        .as_ref()
        .unwrap()
        .json()
        .contains("borrowed_call_materialization"));
}

#[test]
fn every_interior_stress_keeps_original_owning_state_and_recovery_pc() {
    let name = "every_interior_stress_keeps_original_owning_state_and_recovery_pc";
    if child(name, &[]) {
        return;
    }
    let setup = "function finish(a,b,c){return a.k+b.k+c.k;}function subject(x,y){return finish(x,y,x);}var a={k:2},b={k:3};";
    let mut probe = Engine::new();
    eval(&mut probe, setup);
    let chunk = bytecode::compile(&function(&mut probe)).unwrap();
    let plan = certificate(&chunk, None, true).unwrap();
    let site = plan
        .sites
        .iter()
        .flatten()
        .find(|site| site.destination.is_none() && site.count == 3)
        .copied()
        .unwrap();
    for target in site.start + 1..=site.end {
        let mut e = Engine::new();
        eval(&mut e, setup);
        let (chunk, code) = install(&mut e, Some(target));
        assert!(TEST_SELECTED.with(|sites| {
            sites
                .borrow()
                .iter()
                .all(|&(a, b)| !(a < target && target <= b))
        }));
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        native(&mut e, &code, "subject(a,b)", "7");
        ENTRIES
            .with(|entries| assert_eq!(&*entries.borrow(), &[(target, stack.at[target].depth, 0)]));
    }
}

#[test]
fn tdz_later_argument_recovers_before_first_load_in_fact_controls() {
    let name = "tdz_later_argument_recovers_before_first_load_in_fact_controls";
    if child(name, &[("TRUST_JS_OPT_JIT_VALUE_FACTS", "0")]) {
        assert!(child(name, &[("TRUST_JS_OPT_JIT_VALUE_FACTS", "1")]));
        return;
    }
    let setup = "function finish(a,b){return 1;}function subject(x){return finish(x,y);let y;}var held={p:1};";
    let source = "try{subject(held)}catch(e){e.name+'|'+e.stack.split('\\n').slice(1).join('|')}";
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, setup);
    let expected = eval(&mut oracle, source);
    assert!(expected.starts_with("ReferenceError|"));
    let mut e = Engine::new();
    eval(&mut e, setup);
    let (chunk, code) = install(&mut e, None);
    let selected = TEST_SELECTED.with(|sites| sites.borrow().clone());
    assert_eq!(selected.len(), 1);
    let start = selected[0].0;
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native(&mut e, &code, source, &expected);
    ENTRIES.with(|entries| assert_eq!(&*entries.borrow(), &[(start, stack.at[start].depth, 0)]));
}

#[test]
fn all_copyable_proofs_decline_batch_and_failed_scans_still_cost_work() {
    let name = "all_copyable_proofs_decline_batch_and_failed_scans_still_cost_work";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    eval(&mut e, "function subject(){var x=1;var y=2;y=x;return y;}");
    let chunk = bytecode::compile(&function(&mut e)).unwrap();
    assert!(certificate(&chunk, None, true).is_none());
    assert!(certificate(&chunk, None, false).is_some());
    assert_eq!(analysis_work(10), Some(264));
    assert!(analysis_work(usize::MAX).is_none());
    let (_, code) = install(&mut e, None);
    assert!(TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
    native(&mut e, &code, "subject()", "1");
}

#[test]
fn disabled_and_unsupported_owner_layout_keep_original_checked_ownership() {
    let name = "disabled_and_unsupported_owner_layout_keep_original_checked_ownership";
    if child(name, &[("TRUST_JS_OPT_JIT_BORROWED_OPERANDS", "0")]) {
        assert!(child(name, &[("TRUST_JS_OPT_JIT_BORROWED_OPERANDS", "1")]));
        return;
    }
    let setup = "function subject(x){var y=x;return y;}var v={k:3};";
    let mut e = Engine::new();
    eval(&mut e, setup);
    let (_, code) = install(&mut e, None);
    if !requested() {
        assert!(TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
    }
    native(&mut e, &code, "subject(v)===v", "true");
    let mut second = Engine::new();
    eval(&mut second, setup);
    let definition = function(&mut second);
    let chunk = bytecode::compile(&definition).unwrap();
    let mut values = crate::value::jit_layout(&second.interp.object_proto);
    values.rc_strong_off = usize::MAX;
    let layout = crate::interpreter::interp_layout(&mut second.interp);
    assert!(!crate::value::jit_packed_owners_supported(&values));
    TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
    let code = Rc::new(compile_with_deopt(&chunk, &values, &layout, None).unwrap());
    assert!(TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(definition.optimizing_code.set(Some(chunk)).is_ok());
    bytecode::invalidate_call_caches();
    second.set_tier(Tier::Jit);
    second.set_tier_threshold(0);
    TEST_TRANSACTIONS.with(|calls| calls.set(0));
    native(&mut second, &code, "subject(v)===v", "true");
    assert_eq!(TEST_TRANSACTIONS.with(Cell::get), 0);
}

#[test]
fn same_source_destination_and_repeated_argument_aliases_commit_once_per_owner() {
    let name = "same_source_destination_and_repeated_argument_aliases_commit_once_per_owner";
    if child(name, &[]) {
        return;
    }
    let setup = "function finish(a,b,c,d,e,f,g,h,i,j){return a===b&&a===c&&a===d&&a===e&&a===f&&a===g&&a===h&&a===i&&a===j;}function subject(x){x=x;return finish(x,x,x,x,x,x,x,x,x,x);}var v={k:1};";
    cross(setup, "subject(v)", "true");
    let mut e = Engine::new();
    eval(&mut e, setup);
    let (chunk, code) = install(&mut e, None);
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let plan = Plan::build(
        &chunk,
        &cfg,
        &stack,
        None,
        None,
        None,
        NativeEntryKind::FreshFrame,
        None,
    )
    .unwrap();
    assert!(plan.sites.iter().flatten().any(|site| site.count == 8));
    TEST_TRANSACTIONS.with(|calls| calls.set(0));
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native(&mut e, &code, "subject(v)", "true");
    assert_eq!(TEST_TRANSACTIONS.with(Cell::get), 2);
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
}

thread_local! {
    static TRACKED:std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>>=const{std::cell::RefCell::new(None)};
    static SENTINEL:std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>>=const{std::cell::RefCell::new(None)};
    static CREATIONS:Cell<usize>=const{Cell::new(0)};
    static COLLECTION_COUNTS:Cell<[usize;2]>=const{Cell::new([usize::MAX;2])};
}
fn make_tracked(
    i: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    let object = i.new_object();
    let sentinel = i.new_object();
    SENTINEL.with(|tracked| *tracked.borrow_mut() = Some(Rc::downgrade(&sentinel)));
    object.borrow_mut().props.insert(
        "sentinel",
        crate::value::Property::plain(JsValue::Obj(sentinel)),
    );
    TRACKED.with(|tracked| *tracked.borrow_mut() = Some(Rc::downgrade(&object)));
    CREATIONS.with(|count| count.set(count.get() + 1));
    Ok(JsValue::Obj(object))
}
fn collect_tracked(
    i: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    let before = TRACKED.with(|tracked| {
        tracked
            .borrow()
            .as_ref()
            .map_or(usize::MAX, |weak| weak.strong_count())
    });
    i.gc_collect();
    let after = TRACKED.with(|tracked| {
        tracked
            .borrow()
            .as_ref()
            .map_or(usize::MAX, |weak| weak.strong_count())
    });
    COLLECTION_COUNTS.with(|counts| counts.set([before, after]));
    Ok(JsValue::Undefined)
}
#[test]
fn last_owner_destruction_does_not_replay_prior_effect_or_leave_a_hidden_owner() {
    let name = "last_owner_destruction_does_not_replay_prior_effect_or_leave_a_hidden_owner";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    let global = e.interp.global.clone();
    e.interp.def_method(&global, "makeTracked", 0, make_tracked);
    eval(
        &mut e,
        "function subject(x){var y=makeTracked();y=x;return y;}var v={k:3};",
    );
    let (chunk, code) = install(&mut e, None);
    let sites = TEST_SELECTED.with(|sites| sites.borrow().clone());
    assert_eq!(sites.len(), 1);
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let start = sites[0].0;
    CREATIONS.with(|n| n.set(0));
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native(&mut e, &code, "subject(v)===v", "true");
    assert_eq!(CREATIONS.with(Cell::get), 1);
    TRACKED.with(|tracked| {
        let binding = tracked.borrow();
        let weak = binding.as_ref().unwrap();
        assert_eq!(weak.strong_count(), 0);
        assert!(weak.upgrade().is_none());
    });
    SENTINEL.with(|tracked| {
        let binding = tracked.borrow();
        let weak = binding.as_ref().unwrap();
        assert_eq!(weak.strong_count(), 0, "outer payload was actually dropped");
        assert!(weak.upgrade().is_none());
    });
    ENTRIES.with(|entries| assert_eq!(&*entries.borrow(), &[(start, stack.at[start].depth, 0)]));
}

#[test]
fn opaque_owning_handler_floor_survives_argument_promotion_and_reentrant_collection() {
    let name = "opaque_owning_handler_floor_survives_argument_promotion_and_reentrant_collection";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    let global = e.interp.global.clone();
    e.interp.def_method(&global, "makeTracked", 0, make_tracked);
    e.interp
        .def_method(&global, "forceCollection", 0, collect_tracked);
    eval(
        &mut e,
        "function finish(a,b){forceCollection();return a.k+b.k;}function subject(x,y){var keep=makeTracked();try{try{keep=null;throw 1;}catch(e){return finish(x,y);}}catch(e){return 0;}}var a={k:3},b={k:4};",
    );
    let definition = function(&mut e);
    let (chunk, _, _) =
        bytecode::compile_test_parked_handler_owner(&definition, "keep", 2).unwrap();
    let values = crate::value::jit_layout(&e.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut e.interp);
    TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
    let code = Rc::new(compile_with_deopt(&chunk, &values, &layout, None).unwrap());
    let sites = TEST_SELECTED.with(|sites| sites.borrow().clone());
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    assert!(sites.iter().any(|&(start, end)| stack.at[start].floor > 0
        && matches!(chunk.jit_ops()[end], Op::CallWithThis(..))));
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(definition.optimizing_code.set(Some(chunk)).is_ok());
    bytecode::invalidate_call_caches();
    e.set_tier(Tier::Jit);
    e.set_tier_threshold(0);
    COLLECTION_COUNTS.with(|n| n.set([usize::MAX; 2]));
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native(&mut e, &code, "subject(a,b)", "7");
    assert_eq!(COLLECTION_COUNTS.with(Cell::get), [1, 1]);
    TRACKED.with(|tracked| assert_eq!(tracked.borrow().as_ref().unwrap().strong_count(), 0));
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
}

#[test]
fn integer_and_f64_mirrors_survive_materialization_and_checked_callback_controls() {
    let name = "integer_and_f64_mirrors_survive_materialization_and_checked_callback_controls";
    if child(name, &[]) {
        assert!(child(
            name,
            &[
                ("TRUST_JS_OPT_JIT_LOCAL_EFFECTS", "0"),
                ("TRUST_JS_OPT_JIT_SPARSE_RELOADS", "0"),
                ("TRUST_JS_OPT_JIT_VALUE_FACTS", "0")
            ]
        ));
        return;
    }
    let mut e = Engine::new();
    let global = e.interp.global.clone();
    e.interp.def_method(&global, "forceCollection", 0, collect);
    eval(
        &mut e,
        "function finish(b,c){forceCollection();b.k+=c.k;}function subject(a,b,c){var x=8589934597;var y=x;var t=a*1.25;finish(b,c);return (y|0)+t;}var b={k:2},c={k:3};",
    );
    let (_, code) = install(&mut e, None);
    assert!(TEST_SELECTED.with(|sites| !sites.borrow().is_empty()));
    native(&mut e, &code, "subject(4,b,c)+'|'+b.k", "10|5");
}

#[test]
fn root_join_prepared_and_continuation_certificates_keep_owning_boundaries() {
    let name = "root_join_prepared_and_continuation_certificates_keep_owning_boundaries";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    eval(
        &mut e,
        "function finish(a,b){return a.k+b.k;}function subject(x,y,skip){try{if(skip)return 0;var z=x;return finish(z,y);}catch(e){return -1;}}var a={k:2},b={k:3};",
    );
    let mut chunk = bytecode::compile(&function(&mut e)).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let plan = certificate(&chunk, None, false).unwrap();
    let copy = *plan
        .sites
        .iter()
        .flatten()
        .find(|site| site.destination.is_some())
        .unwrap();
    assert_eq!(stack.at[copy.end].depth, 1);
    let mut altered = chunk.jit_ops().to_vec();
    let branch = altered[..copy.start]
        .iter()
        .position(|op| matches!(op, Op::JumpIfFalse(_)))
        .unwrap();
    altered[branch] = Op::JumpIfFalse(copy.end as u32);
    // JumpIfFalse's outgoing depth0 cannot enter a StoreLocal requiring depth1.
    // The complete CFG rejects this malformed edge before the loan certificate.
    assert!(Cfg::test_build_with(&altered, |pc| chunk.jit_stack_effect(pc)).is_err());
    let mut altered = chunk.jit_ops().to_vec();
    let handler = altered
        .iter()
        .position(|op| matches!(op, Op::PushHandler(_)))
        .unwrap();
    altered[handler] = Op::PushHandler(copy.end as u32);
    // An author catch entry owns the thrown word, matching this interior depth.
    let handler_cfg = Cfg::test_build_with(&altered, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(handler_cfg
        .handler_roots()
        .iter()
        .any(|root| handler_cfg.blocks()[root.target.0 as usize].start == copy.end));
    let changed = Plan::build(
        &chunk,
        &handler_cfg,
        &stack,
        None,
        None,
        None,
        NativeEntryKind::FreshFrame,
        None,
    );
    assert!(changed.as_ref().is_none_or(|plan| !plan.covers(copy.start)));
    for origin in [
        NativeEntryKind::BorrowedFrame,
        NativeEntryKind::OptimizingContinuation,
    ] {
        assert!(Plan::build(&chunk, &cfg, &stack, None, None, None, origin, None).is_none());
    }
    let mut reachable = stack.reachable.clone();
    reachable[copy.end] = false;
    let filtered = Plan::build(
        &chunk,
        &cfg,
        &stack,
        None,
        Some(&reachable),
        None,
        NativeEntryKind::FreshFrame,
        None,
    );
    assert!(filtered
        .as_ref()
        .is_none_or(|plan| !plan.covers(copy.start)));
    Rc::get_mut(&mut chunk).unwrap().prepared_entry = true;
    assert!(certificate(&chunk, None, false).is_none());
    // The original actual graph still executes both ordinary and exceptional routes.
    let (_, code) = install(&mut e, None);
    native(
        &mut e,
        &code,
        "subject(a,b,false)+'|'+subject(a,b,true)",
        "5|0",
    );
}

#[test]
fn loop_and_normal_join_execute_repeated_owning_copy_and_call_boundaries() {
    let name = "loop_and_normal_join_execute_repeated_owning_copy_and_call_boundaries";
    if child(name, &[]) {
        return;
    }
    let setup = "function finish(a,b){return a.k+b.k;}function subject(x,y,flag){var total=0;for(var i=0;i<3;i++){var z=x;total+=finish(flag?z:y,x);}return total;}var a={k:2},b={k:3};";
    cross(setup, "subject(a,b,true)+'|'+subject(a,b,false)", "12|15");
    let mut e = Engine::new();
    eval(&mut e, setup);
    let (chunk, code) = install(&mut e, None);
    let cfg = Cfg::build(&chunk).unwrap();
    let plan = certificate(&chunk, None, true).unwrap();
    for site in plan.sites.iter().flatten() {
        assert!((site.start..=site.end).all(|pc| cfg.block_at(pc) == cfg.block_at(site.start)));
        assert!((site.start + 1..=site.end).all(|pc| {
            chunk
                .jit_ops()
                .iter()
                .all(|op| crate::jit_ir::jump_target(op) != Some(pc))
        }));
    }
    TEST_TRANSACTIONS.with(|n| n.set(0));
    ENTRIES.with(|n| n.borrow_mut().clear());
    native(
        &mut e,
        &code,
        "subject(a,b,true)+'|'+subject(a,b,false)",
        "12|15",
    );
    assert!(TEST_TRANSACTIONS.with(Cell::get) >= 6);
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
}

#[test]
fn selected_copy_windows_remove_numeric_credit_without_erasing_guard_tax() {
    let name = "selected_copy_windows_remove_numeric_credit_without_erasing_guard_tax";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    eval(
        &mut e,
        "function subject(a,b){var y=b;y=a;var z=b;z=a;return 1;}",
    );
    let chunk = bytecode::compile(&function(&mut e)).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let bits = crate::feedback::ValueClass::NumberInt32.bit();
    let mut numeric = specialization::Plan::build(&chunk, &cfg, &stack, &[(0, bits)], &[]).unwrap();
    let plan = Plan::build(
        &chunk,
        &cfg,
        &stack,
        None,
        None,
        None,
        NativeEntryKind::FreshFrame,
        Some(&numeric.values),
    )
    .unwrap();
    plan.remove_numeric_credit(&mut numeric);
    assert_eq!(numeric.saved_tests, 0);
    assert!(numeric.opportunities.is_empty());
    assert_eq!(numeric.inputs.len(), 1);
    assert!(generic_admission::minimum_saved_work(
        &chunk,
        &cfg,
        Some(&numeric),
        None,
        None,
        None,
        None,
        Some(&numeric.values)
    )
    .is_none());
    let work =
        plan.lowering_work(&chunk, &stack).unwrap() + analysis_work(chunk.jit_ops().len()).unwrap();
    assert!(work > 0);
    let old = generic_admission::admission_entries_with_work(&chunk, 1, 0).unwrap();
    let priced = generic_admission::admission_entries_with_work(&chunk, 1, work).unwrap();
    assert_eq!(priced - old, work * 1024);
}

#[test]
fn explicitly_disabled_and_unset_flags_emit_identical_uninstrumented_graphs() {
    let name = "explicitly_disabled_and_unset_flags_emit_identical_uninstrumented_graphs";
    if child(name, &[("TRUST_JS_OPT_JIT_BORROWED_OPERANDS", "0")]) {
        return;
    }
    let mut e = Engine::new();
    eval(
        &mut e,
        "function finish(a,b){return a===b;}function subject(x,y){var z=x;return finish(z,y);}",
    );
    let chunk = bytecode::compile(&function(&mut e)).unwrap();
    let values = crate::value::jit_layout(&e.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut e.interp);
    let prepare = || {
        diagnostics::with_test_mode("0", || {
            prepare_entry_with_limits(&chunk, &values, &layout, None, 16384, 65536, None).unwrap()
        })
    };
    let off = prepare();
    // This process runs exactly one isolated test; no engine thread runs concurrently.
    unsafe {
        std::env::remove_var("TRUST_JS_OPT_JIT_BORROWED_OPERANDS");
    }
    assert!(!requested());
    let absent = prepare();
    assert_ne!(
        &*off.finish.residency as *const CodeResidency,
        &*absent.finish.residency as *const CodeResidency
    );
    let graph = |prepared: &PreparedCode| {
        let address = (&*prepared.finish.residency as *const CodeResidency) as u64;
        prepared
            .context
            .func
            .display()
            .to_string()
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|word| {
                        let number = word.trim_end_matches(',').replace('_', "");
                        let parsed = number
                            .strip_prefix("0x")
                            .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                            .or_else(|| number.parse::<u64>().ok());
                        if parsed == Some(address) {
                            "<residency>"
                        } else {
                            word
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(graph(&off),graph(&absent),"only independent residency addresses normalized; disabled switch adds no native signature/slot/guard/helper");
}

#[test]
fn real_inline_positions_and_resumable_roots_never_form_loans() {
    let name = "real_inline_positions_and_resumable_roots_never_form_loans";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    e.set_tier(Tier::Jit);
    e.set_tier_threshold(0);
    eval(
        &mut e,
        "function callee(x){var y=x;return y;}function subject(x){return callee(x);}var v={k:3};",
    );
    let definition = function(&mut e);
    let primary = bytecode::compile(&definition).unwrap();
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(definition.code.set(Some(primary.clone())).is_ok());
    for _ in 0..8 {
        assert_eq!(eval(&mut e, "subject(v)===v"), "true");
    }
    let env = e.interp.global_env.clone();
    let function_value = e.interp.get_var("subject", &env).ok().unwrap();
    let user = match &function_value.as_obj().unwrap().borrow().call {
        Callable::User(user) => user.clone(),
        _ => panic!("user fixture"),
    };
    let inline = bytecode::plan_inlines(&primary, &definition, user.realm, &e.interp.global_env);
    assert!(!inline.is_empty());
    let chunk = bytecode::compile_with_inlines(&definition, &inline, &primary).unwrap();
    assert!(chunk
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let plan = Plan::build(
        &chunk,
        &cfg,
        &stack,
        None,
        None,
        None,
        NativeEntryKind::FreshFrame,
        None,
    );
    let mut inline_loads = 0;
    for pc in 0..chunk.jit_ops().len() {
        if matches!(chunk.jit_ops()[pc], Op::LoadLocal(_))
            && crate::stack_trace::is_inline_position(chunk.position(pc))
        {
            inline_loads += 1;
            assert!(plan.as_ref().is_none_or(|plan| !plan.covers(pc)));
        }
    }
    assert!(inline_loads > 0);
    let mut second = Engine::new();
    eval(
        &mut second,
        "async function subject(x){var y=x;await 1;return y;}",
    );
    let chunk = bytecode::compile(&function(&mut second)).unwrap();
    assert!(chunk.jit_is_resumable());
    let cfg = Cfg::build(&chunk).unwrap();
    assert!(!cfg.resume_roots().is_empty());
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    assert!(Plan::build(
        &chunk,
        &cfg,
        &stack,
        None,
        None,
        None,
        NativeEntryKind::FreshFrame,
        None
    )
    .is_none());
}

#[test]
fn repeated_last_owner_exit_cannot_borrow_later_numeric_repayment() {
    let name = "repeated_last_owner_exit_cannot_borrow_later_numeric_repayment";
    if child(name, &[]) {
        return;
    }
    let mut e = Engine::new();
    eval(
        &mut e,
        "function subject(old,x,n){old=x;return n+n+n+n+n+n+n+n;}var x={k:1};",
    );
    let chunk = bytecode::compile(&function(&mut e)).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let bits = crate::feedback::ValueClass::NumberInt32.bit();
    let mut numeric = specialization::Plan::build(&chunk, &cfg, &stack, &[(2, bits)], &[]).unwrap();
    let plan = Plan::build(
        &chunk,
        &cfg,
        &stack,
        None,
        None,
        None,
        NativeEntryKind::FreshFrame,
        Some(&numeric.values),
    )
    .unwrap();
    plan.remove_numeric_credit(&mut numeric);
    assert!(
        generic_admission::minimum_saved_work(
            &chunk,
            &cfg,
            Some(&numeric),
            None,
            None,
            None,
            None,
            Some(&numeric.values)
        )
        .is_some(),
        "old original-op-only bound counted later work"
    );
    assert!(generic_admission::minimum_saved_work_with_borrowed(
        &chunk,
        &cfg,
        Some(&numeric),
        None,
        None,
        None,
        None,
        Some(&numeric.values),
        Some(&plan)
    )
    .is_none());
    let (actual_chunk, code) = install(&mut e, None);
    let selected = TEST_SELECTED.with(|sites| sites.borrow().clone());
    assert_eq!(selected.len(), 1);
    let start = selected[0].0;
    for _ in 0..3 {
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        native(&mut e, &code, "subject({},x,2)", "16");
        ENTRIES
            .with(|entries| assert_eq!(&*entries.borrow(), &[(start, stack.at[start].depth, 0)]));
    }
    assert_eq!(
        actual_chunk.optimizing_misses.get(),
        0,
        "ordinary ownership recovery is not specialization retirement"
    );
}

#[test]
fn lazy_owner_callees_dominate_real_loop_and_handler_entries_without_unused_ir() {
    let name = "lazy_owner_callees_dominate_real_loop_and_handler_entries_without_unused_ir";
    if child(
        name,
        &[
            ("TRUST_JS_OPT_JIT_BORROWED_OPERANDS", "0"),
            ("TRUST_JS_OPT_JIT_SHARED_OWNERS", "1"),
        ],
    ) {
        return;
    }
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ownership::TEST_CALLEE_CACHE.with(|value| value.set(self.0));
        }
    }
    let setup = "function subject(x,flag){var y=x;var i=0;try{while(i<3){y=x;i++;if(flag&&i===2)throw 1;}return y;}catch(e){return y;}}var v={k:3};";
    let mut shapes = Vec::new();
    for hoist in [false, true] {
        let _restore = Restore(ownership::TEST_CALLEE_CACHE.with(|value| value.replace(hoist)));
        ownership::TEST_CALLEE_CONSTANTS.with(|value| value.set(0));
        let mut e = Engine::new();
        eval(&mut e, setup);
        let (_, code) = install(&mut e, None);
        let record = code.optimizing_diagnostics.as_ref().unwrap();
        let constants = ownership::TEST_CALLEE_CONSTANTS.with(Cell::get);
        let calls = record.emitted_owner_leaf_calls.get() as usize;
        assert!(calls > 2, "actual shared retain and release call sites");
        if hoist {
            assert_eq!(constants, 2);
        } else {
            assert_eq!(constants, 0);
        }
        shapes.push((
            record.ir_instructions.get(),
            record.ir_blocks.get(),
            calls,
            constants,
            code.len,
        ));
        native(
            &mut e,
            &code,
            "[subject(v,false)===v,subject(v,true)===v].join('|')",
            "true|true",
        );
    }
    assert_eq!(shapes[0].1, shapes[1].1);
    assert_eq!(shapes[0].2, shapes[1].2);
    assert_eq!(
        shapes[0].0 - shapes[1].0,
        shapes[1].2 - shapes[1].3,
        "all and only repeated owner callee constants are removed"
    );
    eprintln!(
        "owner-callee-hoist old={:?} new={:?}; actual IR/code sizes, no timing claim",
        shapes[0], shapes[1]
    );
    ownership::TEST_CALLEE_CONSTANTS.with(|value| value.set(0));
    let mut e = Engine::new();
    eval(&mut e, "function subject(){return 1;}");
    let (_, code) = install(&mut e, None);
    assert_eq!(
        ownership::TEST_CALLEE_CONSTANTS.with(Cell::get),
        0,
        "lazy cache adds no callee constant to a body with no owner calls"
    );
    native(&mut e, &code, "subject()", "1");
}

#[test]
fn counted_and_ordinary_native_transactions_preserve_aliases_and_counter_scope() {
    let name = "counted_and_ordinary_native_transactions_preserve_aliases_and_counter_scope";
    if child(name, &[]) {
        return;
    }
    let setup="function finish(a,b){forceCollection();return a===b;}function subject(x){var y=x;return finish(y,x);}var v={k:3};";
    for mode in ["0", "compile", "execution", "1"] {
        let mut e = Engine::new();
        let global = e.interp.global.clone();
        e.interp.def_method(&global, "forceCollection", 0, collect);
        eval(&mut e, setup);
        let (_, code) = install_mode(&mut e, None, mode);
        assert_eq!(TEST_SELECTED.with(|sites| sites.borrow().len()), 2);
        TEST_TRANSACTIONS.with(|count| count.set(0));
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        native(&mut e, &code, "subject(v)", "true");
        assert_eq!(
            TEST_TRANSACTIONS.with(Cell::get),
            2,
            "mode={mode} actual successful copy+alias-argument transactions"
        );
        ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
        if mode == "0" {
            assert!(code.optimizing_diagnostics.is_none());
            continue;
        }
        let record = code.optimizing_diagnostics.as_ref().unwrap();
        if mode == "1" {
            assert!(record.runtime_counters);
            assert!(
                record.counters[diagnostics::RETAIN].get() >= 3,
                "counted C ABI status includes all three borrowed heap promotions"
            );
        } else {
            assert!(!record.runtime_counters);
            assert_eq!(record.counters[diagnostics::RETAIN].get(), 0);
            assert_eq!(record.counters[diagnostics::RELEASE].get(), 0);
        }
        let json = record.json();
        assert!(json.contains("borrowed-owner-window commit"));
        assert!(json.contains("not its loads or call completion"));
        if mode == "execution" {
            assert!(json.contains("borrowed_local_copy"));
            assert!(json.contains("borrowed_call_materialization"));
        }
    }
}
