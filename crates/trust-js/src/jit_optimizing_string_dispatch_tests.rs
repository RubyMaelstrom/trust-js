use super::*;
use crate::bytecode::{native_deopt::ENTRIES, Tier};
use crate::value::Callable;
use crate::{Completion, Engine};
use std::rc::Rc;

struct Recording(bool);

impl Recording {
    fn new(enabled: bool) -> Self {
        TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
        TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
        TEST_STRING_OWNERS.with(|counts| counts.set([0; 2]));
        TEST_EMPTY_GUARDS.with(|guards| guards.set(0));
        TEST_REAL_VIEWS.with(|views| views.set(0));
        Self(TEST_ENABLED.with(|value| value.replace(enabled)))
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        TEST_ENABLED.with(|value| value.set(self.0));
    }
}

fn child(name: &str) -> bool {
    child_controls(name, &[])
}

fn child_controls(name: &str, controls: &[(&str, &str)]) -> bool {
    const CONTROL: &str = "TRUST_JS_TEST_STRING_DISPATCH_CHILD";
    if std::env::var(CONTROL).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        &format!("jit::optimizing::string_dispatch::tests::{name}"),
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
        .env_remove("LD_PRELOAD");
    for (key, value) in controls {
        command.env(key, value);
    }
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

#[test]
fn original_group_analysis_and_switch_work_are_charged_without_extra_credit() {
    if child("original_group_analysis_and_switch_work_are_charged_without_extra_credit") {
        return;
    }
    let _recording = Recording::new(true);
    assert_eq!(analysis_work(10), Some(396));
    assert!(analysis_work(usize::MAX).is_none());
    let mut engine = Engine::new();
    eval(&mut engine, SWITCH);
    let chunk = bytecode::compile(&function(&mut engine, "subject")).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let literals = string_literals::Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
    let group = Plan::build(&chunk, &cfg, &stack, None, None, None, &literals).unwrap();
    assert_eq!(group.selected_groups(), 1);
    assert_eq!(group.selected_cases(), 4);
    let work = group.lowering_work(&stack).unwrap();
    assert!(work >= 5 * 20 + 64);
    // Still one original constant-owner token per comparison, never count a case body.
    assert_eq!(
        (0..chunk.jit_ops().len())
            .map(|pc| literals.credit_at(pc))
            .sum::<u32>(),
        4
    );
    let old = generic_admission::admission_entries_with_work(&chunk, 1, 0).unwrap();
    let analysis = analysis_work(chunk.jit_ops().len()).unwrap();
    let priced =
        generic_admission::admission_entries_with_work(&chunk, 1, analysis + work).unwrap();
    assert_eq!(priced - old, (analysis + work) * 1024);
    // An actual attempted plan with no groups still pays the full original scan.
    let mut no_group = Engine::new();
    eval(
        &mut no_group,
        "function subject(x){if(x==='a')return 1;return 0;}",
    );
    let chunk = bytecode::compile(&function(&mut no_group, "subject")).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let literals = string_literals::Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
    assert!(Plan::build(&chunk, &cfg, &stack, None, None, None, &literals).is_none());
    assert!(analysis_work(chunk.jit_ops().len()).unwrap() > 0);
}

#[test]
fn ordinary_hot_grouped_switch_publishes_with_full_work_and_real_native_dispatch() {
    const NAME: &str =
        "ordinary_hot_grouped_switch_publishes_with_full_work_and_real_native_dispatch";
    if child_controls(
        NAME,
        &[
            ("TRUST_JS_OPT_JIT", "hot"),
            ("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution"),
        ],
    ) {
        return;
    }
    assert!(std::env::var("TRUST_JS_OPT_JIT_HOT_AT").is_err());
    let _recording = Recording::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function subject(x){switch(x){case'a':return 1;case'b':return 2;default:return 3;}}",
    );
    let definition = function(&mut engine, "subject");
    assert_eq!(eval(&mut engine, "subject('a')"), "1");
    let baseline = definition
        .execution_code()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    let cfg = Cfg::build(&baseline).unwrap();
    let stack = stack::StackPlan::build(&baseline, &cfg).unwrap();
    let values = value_facts::ValuePlan::build(&baseline, &cfg, &stack).unwrap();
    let literals = string_literals::Plan::build(&baseline, &cfg, &stack, None, None).unwrap();
    let group = Plan::build(&baseline, &cfg, &stack, None, None, None, &literals).unwrap();
    assert_eq!(group.selected_groups(), 1);
    assert_eq!(group.selected_cases(), 2);
    let saved = generic_admission::minimum_saved_work(
        &baseline,
        &cfg,
        None,
        None,
        None,
        None,
        Some(&literals),
        Some(&values),
    )
    .unwrap();
    assert_eq!(
        saved, 1,
        "first matching case cannot earn the later comparison"
    );
    let work = string_literals::analysis_work(baseline.jit_ops().len()).unwrap()
        + analysis_work(baseline.jit_ops().len()).unwrap()
        + string_literals::checkpoint_value_work(
            baseline.jit_ops().len(),
            baseline.jit_frame().1,
            stack.capacity,
            true,
        )
        .unwrap()
        + group.lowering_work(&stack).unwrap();
    let target = generic_admission::admission_entries_with_work(&baseline, saved, work)
        .unwrap()
        .max(4096);
    let calls = target + u64::from(hot_threshold(baseline.jit_ops().len())) * 4 + 16;
    assert_eq!(
        eval(
            &mut engine,
            &format!("var sum=0;for(var k=0;k<{calls};k++)sum+=subject('a');sum")
        ),
        calls.to_string()
    );
    let candidate = definition
        .optimizing_code
        .get()
        .and_then(Option::as_ref)
        .expect("normal hot policy publication");
    assert!(baseline.optimizing_samples.get().unwrap().entries.get() >= target);
    assert!(candidate.optimizing_structural.is_empty());
    assert!(candidate.optimizing_properties.is_empty());
    let code = candidate.jit.get().flatten().unwrap();
    let json = code.optimizing_diagnostics.as_ref().unwrap().json();
    assert!(json.contains("\"selected_groups\":1"));
    assert!(json.contains("\"selected_original_comparisons\":2"));
    assert!(json.contains("\"emitted_groups\":1"));
    assert!(json.contains("\"additional_semantic_credit\":0"));
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    for (input, expected) in [
        ("'a'", "1"),
        ("'b'", "2"),
        ("'none'", "3"),
        ("new String('a')", "3"),
    ] {
        native_eval(&mut engine, &code, &format!("subject({input})"), expected);
    }
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 4);
    assert_eq!(candidate.optimizing_misses.get(), 0);
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("dispatch fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary fixture")
    };
    user.func.clone()
}

fn certificate(chunk: &Chunk, deopt: Option<usize>) -> Option<Plan> {
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let literals = string_literals::Plan::build(chunk, &cfg, &stack, None, deopt)?;
    Plan::build(chunk, &cfg, &stack, None, None, deopt, &literals)
}

fn install_chunk(
    engine: &mut Engine,
    name: &str,
    chunk: Rc<Chunk>,
    deopt: Option<usize>,
) -> Rc<JitCode> {
    let definition = function(engine, name);
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
    let code = diagnostics::with_test_mode("execution", || {
        Rc::new(compile_with_deopt(&chunk, &values, &layout, deopt).unwrap())
    });
    assert!(code.pc_offsets.is_empty());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(definition.optimizing_code.set(Some(chunk)).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    code
}

fn install(engine: &mut Engine, name: &str, deopt: Option<usize>) -> (Rc<Chunk>, Rc<JitCode>) {
    let definition = function(engine, name);
    let chunk = bytecode::compile(&definition).unwrap();
    let code = install_chunk(engine, name, chunk.clone(), deopt);
    (chunk, code)
}

fn native_eval(engine: &mut Engine, code: &JitCode, source: &str, expected: &str) {
    code.residency.referenced.set(0);
    assert_eq!(eval(engine, source), expected);
    assert!(
        code.residency.referenced.get() > 0,
        "actual optimizing-native entry"
    );
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}

fn cross_tier(setup: &str, source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, setup);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
    }
}

const SWITCH: &str = "function subject(x){var out='';switch(x){case'a':out+='A';case'b':out+='B';break;case'a':out+='duplicate';break;case'z':return'Z';default:out+='?';}return out;}";

#[test]
fn ordered_first_late_missing_duplicate_and_fallthrough_enter_one_dispatch() {
    if child("ordered_first_late_missing_duplicate_and_fallthrough_enter_one_dispatch") {
        return;
    }
    let _recording = Recording::new(true);
    let source = "subject('a')+'|'+subject('b')+'|'+subject('z')+'|'+subject('missing')";
    cross_tier(SWITCH, source, "AB|B|Z|?");
    let mut engine = Engine::new();
    eval(&mut engine, SWITCH);
    let (chunk, code) = install(&mut engine, "subject", None);
    assert_eq!(TEST_SELECTED.with(|sites| sites.borrow().len()), 1);
    assert_eq!(TEST_SELECTED.with(|sites| sites.borrow()[0].1), 4);
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(&mut engine, &code, source, "AB|B|Z|?");
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 4);
    assert_eq!(chunk.optimizing_misses.get(), 0);
    assert!(code
        .optimizing_diagnostics
        .as_ref()
        .unwrap()
        .json()
        .contains("\"kind\":\"literal_string_case_dispatch\",\"native_fast_successes\":4"));
}

#[test]
fn grouped_empty_local_recovers_original_tdz_in_both_orders_and_fact_controls() {
    let name = "grouped_empty_local_recovers_original_tdz_in_both_orders_and_fact_controls";
    if child_controls(name, &[("TRUST_JS_OPT_JIT_VALUE_FACTS", "0")]) {
        assert!(child_controls(
            name,
            &[("TRUST_JS_OPT_JIT_VALUE_FACTS", "1")]
        ));
        return;
    }
    for reversed in [false, true] {
        let _recording = Recording::new(true);
        let first = if reversed { "'a'===x" } else { "x==='a'" };
        let second = if reversed { "'b'===x" } else { "x==='b'" };
        let setup = format!("function subject(){{L:{{if({first})break L;if({second})break L;}}let x='a';return 1;}}");
        let source = r#"try{subject()}catch(e){e.name+'|'+e.stack.split('\n').slice(1).join('|')}"#;
        let mut oracle = Engine::new();
        oracle.set_tier(Tier::Interp);
        eval(&mut oracle, &setup);
        let expected = eval(&mut oracle, source);
        assert!(expected.starts_with("ReferenceError|"));
        cross_tier(&setup, source, &expected);
        let mut engine = Engine::new();
        eval(&mut engine, &setup);
        let (chunk, code) = install(&mut engine, "subject", None);
        let sites = TEST_SELECTED.with(|sites| sites.borrow().clone());
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].1, 2);
        assert_eq!(TEST_EMPTY_GUARDS.with(std::cell::Cell::get), 1);
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
        native_eval(&mut engine, &code, source, &expected);
        ENTRIES.with(|entries| {
            assert_eq!(
                &*entries.borrow(),
                &[(sites[0].0, stack.at[sites[0].0].depth, 0)]
            )
        });
        assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
        assert_eq!(chunk.optimizing_misses.get(), 0);
        assert!(code
            .optimizing_diagnostics
            .as_ref()
            .unwrap()
            .json()
            .contains("\"kind\":\"literal_string_case_dispatch\",\"native_fast_successes\":0"));
    }
}

#[test]
fn grouped_first_load_initialization_uses_original_proof_or_retains_empty_guard() {
    let name = "grouped_first_load_initialization_uses_original_proof_or_retains_empty_guard";
    if child_controls(name, &[("TRUST_JS_OPT_JIT_VALUE_FACTS", "0")]) {
        assert!(child_controls(
            name,
            &[("TRUST_JS_OPT_JIT_VALUE_FACTS", "1")]
        ));
        return;
    }
    let _recording = Recording::new(true);
    let mut engine = Engine::new();
    eval(&mut engine, SWITCH);
    let (_, code) = install(&mut engine, "subject", None);
    let facts = std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").unwrap();
    assert_eq!(
        TEST_EMPTY_GUARDS.with(std::cell::Cell::get),
        usize::from(facts == "0")
    );
    assert_eq!(TEST_SELECTED.with(|sites| sites.borrow().len()), 1);
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(&mut engine, &code, "subject('a')", "AB");
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 1);
}

#[test]
fn bounded_groups_preserve_first_duplicate_default_and_fallthrough_across_split() {
    if child("bounded_groups_preserve_first_duplicate_default_and_fallthrough_across_split") {
        return;
    }
    let _recording = Recording::new(true);
    let mut cases = String::from("case'k0':out+='A';case'k1':out+='B';break;");
    for index in 2..128 {
        cases.push_str(&format!("case'k{index}':return'{index}';"));
    }
    cases.push_str("case'k0':out+='duplicate';break;case'late':out+='L';break;default:out+='?';");
    let setup = format!("function subject(x){{var out='';switch(x){{{cases}}}return out;}}");
    let source = "[subject('k0'),subject('k100'),subject('late'),subject('missing')].join('|')";
    cross_tier(&setup, source, "AB|100|L|?");
    let mut engine = Engine::new();
    eval(&mut engine, &setup);
    let (_, code) = install(&mut engine, "subject", None);
    assert_eq!(
        TEST_SELECTED.with(|sites| sites
            .borrow()
            .iter()
            .map(|(_, count)| *count)
            .collect::<Vec<_>>()),
        [128, 2]
    );
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(&mut engine, &code, source, "AB|100|L|?");
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 6);
}

#[test]
fn nonstrings_proxies_symbols_and_boxed_strings_never_coerce() {
    if child("nonstrings_proxies_symbols_and_boxed_strings_never_coerce") {
        return;
    }
    let _recording = Recording::new(true);
    let setup = format!("{SWITCH}var traps=0;var revoked=Proxy.revocable({{}},{{}});revoked.revoke();var p=new Proxy({{}},{{get(){{traps++;throw 'trap';}}}});var values=[0,NaN,true,null,undefined,1n,Symbol('a'),new String('a'),p,revoked.proxy];");
    let source = "values.map(function(x){return subject(x)}).join('|')+'|'+traps";
    let expected = "?|?|?|?|?|?|?|?|?|?|0";
    cross_tier(&setup, source, expected);
    let mut engine = Engine::new();
    eval(&mut engine, &setup);
    let (_, code) = install(&mut engine, "subject", None);
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(&mut engine, &code, source, expected);
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 10);
}

#[test]
fn dynamic_selectors_keep_author_order_and_only_later_literal_run_is_grouped() {
    if child("dynamic_selectors_keep_author_order_and_only_later_literal_run_is_grouped") {
        return;
    }
    let _recording = Recording::new(true);
    let setup = "var trace='';function select(){trace+='S';return'dynamic';}function subject(x){switch(x){case'a':return'A';case select():return'D';case'b':return'B';case'c':return'C';default:return'?';}}";
    let source = "[subject('a'),subject('dynamic'),subject('b'),subject('c'),subject('missing')].join('|')+'|'+trace";
    let expected = "A|D|B|C|?|SSSS";
    cross_tier(setup, source, expected);
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (_, code) = install(&mut engine, "subject", None);
    assert_eq!(TEST_SELECTED.with(|sites| sites.borrow().len()), 1);
    assert_eq!(TEST_SELECTED.with(|sites| sites.borrow()[0].1), 2);
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(&mut engine, &code, source, expected);
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 3);
}

#[test]
fn unicode_surrogates_and_string_views_preserve_exact_code_units() {
    if child("unicode_surrogates_and_string_views_preserve_exact_code_units") {
        return;
    }
    let _recording = Recording::new(true);
    let setup = r#"function subject(x){switch(x){case'\ud800':return'lead';case'\udc00':return'trail';case'😀':return'pair';case'é':return'composed';case'e\u0301':return'decomposed';default:return'none';}}var owner='x'.repeat(200)+'😀'+'x'.repeat(200),view=owner.slice(200,202);"#;
    let source = r#"[subject('\ud800'),subject('\udc00'),subject('😀'),subject('é'),subject('e\u0301'),subject(view),subject('\ufffd')].join('|')"#;
    let expected = "lead|trail|pair|composed|decomposed|pair|none";
    cross_tier(setup, source, expected);
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (_, code) = install(&mut engine, "subject", None);
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(&mut engine, &code, source, expected);
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 7);
}

fn collect_only(
    i: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    i.gc_collect();
    COLLECTIONS.with(|count| count.set(count.get() + 1));
    Ok(JsValue::Undefined)
}

#[test]
fn sole_canonical_string_view_is_borrowed_and_survives_case_collection() {
    if child("sole_canonical_string_view_is_borrowed_and_survives_case_collection") {
        return;
    }
    let _recording = Recording::new(true);
    COLLECTIONS.with(|count| count.set(0));
    let literal = "x".repeat(300);
    let setup = format!("function subject(seed){{switch(seed.slice(2,-2)){{case'{literal}':forceCollection();return'A';case'other':forceCollection();return'B';default:forceCollection();return'?';}}}}var seed='aa{literal}bb';");
    let mut engine = Engine::new();
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "forceCollection", 0, collect_only);
    eval(&mut engine, &setup);
    let (_, code) = install(&mut engine, "subject", None);
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    TEST_STRING_OWNERS.with(|counts| counts.set([0; 2]));
    native_eval(
        &mut engine,
        &code,
        "subject(seed)+'|'+subject('aa'+'y'.repeat(300)+'bb')",
        "A|?",
    );
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 2);
    assert_eq!(TEST_STRING_OWNERS.with(std::cell::Cell::get), [2, 0]);
    assert_eq!(TEST_REAL_VIEWS.with(std::cell::Cell::get), 2);
    assert_eq!(COLLECTIONS.with(std::cell::Cell::get), 2);
}

#[test]
fn interior_stress_retains_original_operations_and_exact_recovery_pc() {
    if child("interior_stress_retains_original_operations_and_exact_recovery_pc") {
        return;
    }
    for offset in 1..20 {
        let _recording = Recording::new(true);
        let mut engine = Engine::new();
        eval(&mut engine, SWITCH);
        let definition = function(&mut engine, "subject");
        let chunk = bytecode::compile(&definition).unwrap();
        let plan = certificate(&chunk, None).unwrap();
        let start = (0..chunk.jit_ops().len())
            .find(|&pc| plan.site(pc).is_some())
            .unwrap();
        let target = start + offset;
        if let Some(plan) = certificate(&chunk, Some(target)) {
            assert!((0..chunk.jit_ops().len()).all(|pc| plan
                .site(pc)
                .is_none_or(|site| { !(site.start < target && target < site.end()) })));
        }
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        let code = install_chunk(&mut engine, "subject", chunk, Some(target));
        // Other, later pure groups may still be emitted. No selected group may
        // suppress the original stress operation that this invocation must reach.
        TEST_SELECTED.with(|sites| {
            assert!(sites
                .borrow()
                .iter()
                .all(|(pc, count)| { !(*pc < target && target < *pc + count * 5) }));
        });
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        // Missing input executes every comparison, but not the match-only case jumps.
        let input = if offset % 5 == 4 {
            ["a", "b", "a", "z"][offset / 5]
        } else {
            "missing"
        };
        let expected = match input {
            "a" => "AB",
            "b" => "B",
            "z" => "Z",
            _ => "?",
        };
        native_eval(&mut engine, &code, &format!("subject('{input}')"), expected);
        ENTRIES.with(|entries| {
            // The duplicate third case is unreachable for 'a', so its match-only jump
            // is intentionally not reached. All other selected stress PCs must recover.
            if offset != 14 {
                assert_eq!(&*entries.borrow(), &[(target, stack.at[target].depth, 0)]);
            } else {
                assert!(entries.borrow().is_empty());
            }
        });
    }
}

#[test]
fn external_and_handler_entry_reject_grouping_before_omission() {
    if child("external_and_handler_entry_reject_grouping_before_omission") {
        return;
    }
    let _recording = Recording::new(true);
    let mut engine = Engine::new();
    eval(&mut engine, "function subject(x,skip){try{if(skip)return'early';switch(x){case'a':return'A';case'b':return'B';case'z':return'Z';default:return'?';}}catch(e){return'error';}}");
    let chunk = bytecode::compile(&function(&mut engine, "subject")).unwrap();
    let original = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &original).unwrap();
    let literals = string_literals::Plan::build(&chunk, &original, &stack, None, None).unwrap();
    let plan = Plan::build(&chunk, &original, &stack, None, None, None, &literals).unwrap();
    let start = (0..chunk.jit_ops().len())
        .find(|&pc| plan.site(pc).is_some())
        .unwrap();
    let mut ops = chunk.jit_ops().to_vec();
    let outside = ops[..start]
        .iter()
        .position(|op| matches!(op, Op::JumpIfFalse(_)))
        .unwrap();
    ops[outside] = Op::JumpIfFalse((start + 5) as u32);
    let changed = Cfg::test_build_with(&ops, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(Plan::build(&chunk, &changed, &stack, None, None, None, &literals).is_none());
    let mut ops = chunk.jit_ops().to_vec();
    let handler = ops
        .iter()
        .position(|op| matches!(op, Op::PushHandler(_)))
        .unwrap();
    ops[handler] = Op::PushHandler((start + 5) as u32);
    assert!(
        matches!(
            Cfg::test_build_with(&ops, |pc| chunk.jit_stack_effect(pc)),
            Err(crate::jit_ir::BuildError::StackMismatch { .. })
        ),
        "malformed handler stack rejected before any omission"
    );

    // A real compiler fixture parks one owner below the handler comparisons, so
    // the independent handler root has the same depth as an interior comparison.
    // This tests the group root certificate after the full CFG accepted its stack.
    eval(&mut engine, "function handlerSubject(x){var keep={};try{try{keep=null;throw 1;}catch(e){switch(x){case'a':return'A';case'b':return'B';case'z':return'Z';default:return'?';}}}catch(e){return'error';}}");
    let (chunk, _, _) = bytecode::compile_test_parked_handler_owner(
        &function(&mut engine, "handlerSubject"),
        "keep",
        2,
    )
    .unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let literals = string_literals::Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
    let plan = Plan::build(&chunk, &cfg, &stack, None, None, None, &literals).unwrap();
    let start = (0..chunk.jit_ops().len())
        .find(|&pc| plan.site(pc).is_some())
        .unwrap();
    assert_eq!(stack.at[start + 5].depth, 1);
    let mut ops = chunk.jit_ops().to_vec();
    let handler = ops
        .iter()
        .position(|op| matches!(op, Op::PushHandler(_)))
        .unwrap();
    ops[handler] = Op::PushHandler((start + 5) as u32);
    let changed = Cfg::test_build_with(&ops, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(Plan::build(&chunk, &changed, &stack, None, None, None, &literals).is_none());
}

#[test]
fn disabled_dispatch_retains_comparison_fusion_and_native_semantics() {
    if child("disabled_dispatch_retains_comparison_fusion_and_native_semantics") {
        return;
    }
    let _recording = Recording::new(false);
    let mut engine = Engine::new();
    eval(&mut engine, SWITCH);
    let (_, code) = install(&mut engine, "subject", None);
    assert!(TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
    TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
    native_eval(
        &mut engine,
        &code,
        "subject('a')+'|'+subject('z')+'|'+subject('none')",
        "AB|Z|?",
    );
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
    assert!(string_literals::TEST_LEAF_ENTRIES.with(std::cell::Cell::get) > 0);
}

#[test]
fn formerly_cold_case_recovers_before_original_jump_and_runs_body_once() {
    if child("formerly_cold_case_recovers_before_original_jump_and_runs_body_once") {
        return;
    }
    let _recording = Recording::new(true);
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let setup = "var bodies=0;function subject(x){switch(x){case'a':bodies++;return'A';case'b':bodies+=10;return'B';case'z':bodies+=100;return'Z';default:return'?';}}";
    cross_tier(setup, "subject('a')+'|'+bodies", "A|1");
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, setup);
    assert_eq!(
        eval(&mut engine, "subject('missing');subject('missing')"),
        "?"
    );
    let definition = function(&mut engine, "subject");
    let primary = definition
        .execution_code()
        .and_then(Option::as_ref)
        .unwrap();
    let mut chunk = bytecode::compile_for_optimizer(&definition, primary, &[], &[]).unwrap();
    Rc::get_mut(&mut chunk).unwrap().optimizing_candidate = false;
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let regions =
        observed_regions::Plan::build(&chunk, &cfg, &stack, &chunk.execution_coverage).unwrap();
    let literals =
        string_literals::Plan::build(&chunk, &cfg, &stack, Some(&regions), None).unwrap();
    let plan = Plan::build(&chunk, &cfg, &stack, Some(&regions), None, None, &literals).unwrap();
    let start = (0..chunk.jit_ops().len())
        .find(|&pc| plan.site(pc).is_some())
        .unwrap();
    assert_eq!(plan.site(start).unwrap().count, 3);
    for index in 0..3 {
        assert_eq!(
            regions.emission(start + index * 5 + 4),
            observed_regions::Emission::Cut
        );
    }
    let code = install_chunk(&mut engine, "subject", chunk.clone(), None);
    assert_eq!(
        TEST_SELECTED.with(|sites| sites.borrow().as_slice().to_vec()),
        [(start, 3)]
    );
    for (input, expected, index) in [("a", "A|1", 0), ("b", "B|10", 1), ("z", "Z|100", 2)] {
        TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        native_eval(
            &mut engine,
            &code,
            &format!("bodies=0;subject('{input}')+'|'+bodies"),
            expected,
        );
        let target = start + index * 5 + 4;
        ENTRIES
            .with(|entries| assert_eq!(&*entries.borrow(), &[(target, stack.at[target].depth, 0)]));
        assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 1);
        assert_eq!(chunk.optimizing_misses.get(), 0);
    }
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 3);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        0
    );
}

thread_local! {
    static PARKED: std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>> = const { std::cell::RefCell::new(None) };
    static PARKED_GC_COUNTS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([usize::MAX; 2]) };
    static COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn make_parked(
    i: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    let object = i.new_object();
    PARKED.with(|parked| *parked.borrow_mut() = Some(Rc::downgrade(&object)));
    Ok(JsValue::Obj(object))
}

fn collect_parked(
    i: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    let count = || {
        PARKED.with(|parked| {
            parked
                .try_borrow()
                .ok()
                .and_then(|parked| parked.as_ref().map(std::rc::Weak::strong_count))
                .unwrap_or(usize::MAX)
        })
    };
    let before = count();
    i.gc_collect();
    let after = count();
    // Sample safely inside the host callback; assertions run after generated code returns.
    PARKED_GC_COUNTS.with(|counts| counts.set([before, after]));
    COLLECTIONS.with(|count| count.set(count.get() + 1));
    Ok(JsValue::Undefined)
}

#[test]
fn owning_opaque_stack_floor_survives_dispatch_collection_and_original_case_recovery() {
    if child("owning_opaque_stack_floor_survives_dispatch_collection_and_original_case_recovery") {
        return;
    }
    let source = "function subject(x){var keep=makeTracked();try{try{keep=null;throw 1;}catch(e){forceCollection();switch(x){case'a':return 1;case'b':return 2;default:return 3;}}}catch(e){return 0;}}";
    for (tier, recover) in [
        (Tier::Bytecode, false),
        (Tier::Jit, false),
        (Tier::Jit, true),
    ] {
        let _recording = Recording::new(true);
        COLLECTIONS.with(|count| count.set(0));
        let mut engine = Engine::new();
        let global = engine.interp.global.clone();
        engine
            .interp
            .def_method(&global, "makeTracked", 0, make_parked);
        engine
            .interp
            .def_method(&global, "forceCollection", 0, collect_parked);
        eval(&mut engine, source);
        let definition = function(&mut engine, "subject");
        let (chunk, _, _) =
            bytecode::compile_test_parked_handler_owner(&definition, "keep", 2).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        let plan = certificate(&chunk, None).unwrap();
        let start = (0..chunk.jit_ops().len())
            .find(|&pc| plan.site(pc).is_some())
            .unwrap();
        assert_eq!((stack.at[start].floor, stack.at[start].depth), (1, 1));
        let Op::Jump(case_target) = chunk.jit_ops()[start + 4] else {
            panic!("original case Jump")
        };
        let recovery = case_target as usize;
        assert!(recovery >= plan.site(start).unwrap().end());
        let code = if tier == Tier::Jit {
            Some(install_chunk(
                &mut engine,
                "subject",
                chunk.clone(),
                recover.then_some(recovery),
            ))
        } else {
            assert!(definition.code.set(Some(chunk.clone())).is_ok());
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            None
        };
        for (input, expected) in [("a", "1"), ("b", "2"), ("missing", "3")] {
            let call = format!("subject('{input}')");
            PARKED_GC_COUNTS.with(|counts| counts.set([usize::MAX; 2]));
            if let Some(code) = &code {
                TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
                ENTRIES.with(|entries| entries.borrow_mut().clear());
                native_eval(&mut engine, code, &call, expected);
                assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 1);
                ENTRIES.with(|entries| {
                    if recover && input == "a" {
                        assert_eq!(
                            &*entries.borrow(),
                            &[(recovery, stack.at[recovery].depth, 0)]
                        );
                    } else {
                        assert!(entries.borrow().is_empty());
                    }
                });
            } else {
                assert_eq!(eval(&mut engine, &call), expected);
            }
            assert_eq!(PARKED_GC_COUNTS.with(std::cell::Cell::get), [1, 1]);
            PARKED.with(|parked| assert_eq!(parked.borrow().as_ref().unwrap().strong_count(), 0));
        }
        assert_eq!(COLLECTIONS.with(std::cell::Cell::get), 3);
    }
}
