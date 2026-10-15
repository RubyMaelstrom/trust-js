//! Node-verified strict String semantics and actual borrowed native execution/recovery.

use super::*;
use crate::bytecode::native_deopt::ENTRIES;
use crate::{bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

struct Recording(bool);

impl Recording {
    fn new(enabled: bool) -> Self {
        TEST_LEAF_ENTRIES.with(|count| count.set(0));
        TEST_EMPTY_GUARDS.with(|count| count.set(0));
        TEST_STRING_OWNERS.with(|counts| counts.set([0; 2]));
        TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
        Self(TEST_ENABLED.with(|current| current.replace(enabled)))
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        TEST_ENABLED.with(|current| current.set(self.0));
        TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
    }
}

fn child(name: &str) -> bool {
    child_with(name, &[])
}

fn child_with(name: &str, settings: &[(&str, &str)]) -> bool {
    const CONTROL: &str = "TRUST_JS_TEST_STRING_LITERAL_CHILD";
    if std::env::var(CONTROL).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        &format!("jit::optimizing::string_literals::tests::{name}"),
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
    command.envs(settings.iter().copied());
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{name}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed; 0 failed;"),
        "one exact child test must execute"
    );
    true
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("literal fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("fixture function");
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary user function")
    };
    user.func.clone()
}

fn install(
    engine: &mut Engine,
    name: &str,
    deopt: Option<usize>,
) -> (Rc<Chunk>, Rc<JitCode>, Vec<usize>) {
    let function = function(engine, name);
    let chunk = bytecode::compile(&function).unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
    let code = diagnostics::with_test_mode("execution", || {
        Rc::new(compile_with_deopt(&chunk, &values, &layout, deopt).unwrap())
    });
    let sites = TEST_SELECTED.with(|sites| sites.borrow().clone());
    assert!(code.pc_offsets.is_empty(), "actual optimizing body");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    (chunk, code, sites)
}

fn native_eval(engine: &mut Engine, code: &JitCode, source: &str, expected: &str) {
    code.residency.referenced.set(0);
    assert_eq!(eval(engine, source), expected);
    assert!(
        code.residency.referenced.get() > 0,
        "published bytes must run"
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
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn both_orders_and_nonstring_discriminants_do_not_coerce() {
    if child("both_orders_and_nonstring_discriminants_do_not_coerce") {
        return;
    }
    let _recording = Recording::new(true);
    let setup = r#"
        var traps=0;
        function subject(x){if(x==='tag')return 1;return 0;}
        function reverse(x){if('tag'!==x)return 1;return 0;}
        var proxy=new Proxy({}, {get(){traps++;throw 'trap';}});
        var revoke=Proxy.revocable({},{});revoke.revoke();
        var values=['tag','other',new String('tag'),null,undefined,0,NaN,
                    true,1n,Symbol('tag'),function(){},proxy,revoke.proxy];
    "#;
    let source = "values.map(function(x){return subject(x)+':'+reverse(x)}).join('|')+'|'+traps";
    let expected = "1:0|0:1|0:1|0:1|0:1|0:1|0:1|0:1|0:1|0:1|0:1|0:1|0:1|0";
    cross_tier(setup, source, expected);
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (_, subject, sites) = install(&mut engine, "subject", None);
    assert_eq!(sites.len(), 1);
    let (_, reverse, sites) = install(&mut engine, "reverse", None);
    assert_eq!(sites.len(), 1);
    reverse.residency.referenced.set(0);
    TEST_LEAF_ENTRIES.with(|count| count.set(0));
    native_eval(&mut engine, &subject, source, expected);
    assert!(reverse.residency.referenced.get() > 0);
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 26);
}

#[test]
fn switch_duplicates_default_fallthrough_and_dynamic_selectors_keep_order() {
    if child("switch_duplicates_default_fallthrough_and_dynamic_selectors_keep_order") {
        return;
    }
    let setup = r#"
        var trace='';
        function disc(x){trace+='d';return x;}
        function selector(){trace+='t';return 'c';}
        function subject(x){trace='';switch(disc(x)){
          case 'a':trace+='A';
          default:trace+='D';
          case 'b':trace+='B';break;
          case 'a':trace+='Z';break;
          case selector():trace+='C';break;
        }return trace;}
    "#;
    let source = "['a','b','c','z'].map(subject).join('|')";
    let expected = "dADB|dB|dtC|dtDB";
    cross_tier(setup, source, expected);
    struct RestoreGroup(bool);
    impl Drop for RestoreGroup {
        fn drop(&mut self) {
            string_dispatch::TEST_ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    for grouped in [false, true] {
        let _recording = Recording::new(true);
        let _group =
            RestoreGroup(string_dispatch::TEST_ENABLED.with(|enabled| enabled.replace(grouped)));
        string_dispatch::TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
        string_dispatch::TEST_LEAF_ENTRIES.with(|entries| entries.set(0));
        let mut engine = Engine::new();
        eval(&mut engine, setup);
        let (_, code, sites) = install(&mut engine, "subject", None);
        let groups = string_dispatch::TEST_SELECTED.with(|sites| sites.borrow().clone());
        if grouped {
            assert!(sites.is_empty(), "all three literal cases use one group");
            assert_eq!(groups.len(), 1);
            assert_eq!(groups[0].1, 3);
        } else {
            assert_eq!(sites.len(), 3);
            assert!(groups.is_empty());
        }
        native_eval(&mut engine, &code, source, expected);
        if grouped {
            assert_eq!(
                string_dispatch::TEST_LEAF_ENTRIES.with(std::cell::Cell::get),
                4
            );
            assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
        } else {
            assert_eq!(
                string_dispatch::TEST_LEAF_ENTRIES.with(std::cell::Cell::get),
                0
            );
            assert!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get) > 0);
        }
    }
}

#[test]
fn utf16_lone_surrogates_canonical_pairs_and_real_views_compare_exactly() {
    if child("utf16_lone_surrogates_canonical_pairs_and_real_views_compare_exactly") {
        return;
    }
    let _recording = Recording::new(true);
    let literal = "q".repeat(300);
    let setup = format!(
        r#"function subject(x){{if(x==='{literal}')return 1;return 0;}}
         function high(x){{if('\ud800'===x)return 1;return 0;}}
         function pair(x){{if(x==='𝌆')return 1;return 0;}}
         function smuggle(x){{if(x==='\u{{10f800}}')return 1;return 0;}}
         var owner='{literal}!',view=owner.slice(0,300),alias=view;"#
    );
    let source = r#"[subject(view),subject(view+'x'),high('\ud800'),high('\udc00'),pair('\ud834'+'\udf06'),smuggle(String.fromCodePoint(0x10f800)),alias===view].join('|')"#;
    let expected = "1|0|1|0|1|1|true";
    cross_tier(&setup, source, expected);
    let mut engine = Engine::new();
    eval(&mut engine, &setup);
    let env = engine.interp.global_env.clone();
    let JsValue::Str(view) = engine
        .interp
        .get_var("view", &env)
        .ok()
        .expect("fixture view")
    else {
        panic!("view String")
    };
    assert!(
        view.view_owner().is_some(),
        "300-byte slice must be a real view"
    );
    let (_, code, sites) = install(&mut engine, "subject", None);
    assert_eq!(sites.len(), 1);
    let mut unicode_codes = Vec::new();
    for name in ["high", "pair", "smuggle"] {
        let (_, code, sites) = install(&mut engine, name, None);
        assert_eq!(sites.len(), 1);
        code.residency.referenced.set(0);
        unicode_codes.push(code);
    }
    native_eval(&mut engine, &code, source, expected);
    for code in unicode_codes {
        assert!(
            code.residency.referenced.get() > 0,
            "each Unicode native body executes"
        );
        assert_eq!(code.residency.active.get(), 0);
    }
    native_eval(
        &mut engine,
        &code,
        "owner+='x';subject(view)+'|'+alias.length+'|'+owner.length",
        "1|300|302",
    );
}

#[test]
fn sole_local_and_alias_owners_survive_dirty_state_and_live_operand_prefix() {
    if child("sole_local_and_alias_owners_survive_dirty_state_and_live_operand_prefix") {
        return;
    }
    let _recording = Recording::new(true);
    let setup = r#"
        function subject(seed){var x=seed.slice(1),dirty=40,owned={text:x};
          return 10+(x==='tag'?(dirty=42):(dirty=43))+dirty+owned.text.length;}
        function sole(seed){var x=seed.slice(1);if(x==='tag')return 1;return 0;}
    "#;
    cross_tier(
        setup,
        "subject('!tag')+'|'+subject('!other')+'|'+sole('!tag')",
        "97|101|1",
    );
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (_, code, sites) = install(&mut engine, "subject", None);
    assert_eq!(sites.len(), 1);
    native_eval(
        &mut engine,
        &code,
        "subject('!tag')+'|'+subject('!other')",
        "97|101",
    );
    let (_, sole, sites) = install(&mut engine, "sole", None);
    assert_eq!(sites.len(), 1);
    native_eval(&mut engine, &sole, "sole('!tag')", "1");
    let counts = TEST_STRING_OWNERS.with(std::cell::Cell::get);
    assert!(counts[0] > 0, "actual borrowed sole-local String owner");
    assert!(counts[1] > 0, "actual borrowed aliased String owner");
    engine.interp.gc_collect();
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn empty_local_recovers_before_start_in_both_orders_with_original_tdz_stack() {
    if child("empty_local_recovers_before_start_in_both_orders_with_original_tdz_stack") {
        return;
    }
    for reversed in [false, true] {
        let _recording = Recording::new(true);
        let comparison = if reversed { "'tag'===x" } else { "x==='tag'" };
        let setup = format!("function subject(){{if({comparison})return 0;let x='tag';return 1;}}");
        let source = r#"try{subject()}catch(e){e.name+'|'+e.stack.split('\n').slice(1).join('|')}"#;
        let mut oracle = Engine::new();
        oracle.set_tier(Tier::Interp);
        eval(&mut oracle, &setup);
        let expected = eval(&mut oracle, source);
        assert!(expected.starts_with("ReferenceError|"));
        let mut engine = Engine::new();
        eval(&mut engine, &setup);
        let (chunk, code, sites) = install(&mut engine, "subject", None);
        assert_eq!(sites.len(), 1);
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        native_eval(&mut engine, &code, source, &expected);
        ENTRIES.with(|entries| {
            assert_eq!(
                &*entries.borrow(),
                &[(sites[0], stack.at[sites[0]].depth, 0)]
            )
        });
        assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
        assert_eq!(chunk.optimizing_misses.get(), 0);
        let record = code.optimizing_diagnostics.as_ref().unwrap();
        assert!(unsafe { &*record.counter(diagnostics::VM_RECOVERY) }.get() > 0);
    }
}

#[test]
fn stress_deopt_at_each_interior_retains_exact_original_stack_and_side_effects() {
    if child("stress_deopt_at_each_interior_retains_exact_original_stack_and_side_effects") {
        return;
    }
    for reversed in [false, true] {
        let comparison = if reversed { "'tag'===x" } else { "x==='tag'" };
        let setup = format!("var effects=0;function subject(x){{effects++;if({comparison}){{effects+=2;return effects;}}effects+=4;return effects;}}");
        for interior in 1..=3 {
            let _recording = Recording::new(true);
            let mut engine = Engine::new();
            eval(&mut engine, &setup);
            let function = function(&mut engine, "subject");
            let baseline = bytecode::compile(&function).unwrap();
            let cfg = Cfg::build(&baseline).unwrap();
            let stack = stack::StackPlan::build(&baseline, &cfg).unwrap();
            let plan = Plan::build(&baseline, &cfg, &stack, None, None).unwrap();
            let start = (0..baseline.jit_ops().len())
                .find(|&pc| plan.site(pc).is_some())
                .unwrap();
            let (chunk, code, sites) = install(&mut engine, "subject", Some(start + interior));
            assert!(
                sites.is_empty(),
                "stress interior must retain unfused operations"
            );
            ENTRIES.with(|entries| entries.borrow_mut().clear());
            native_eval(&mut engine, &code, "subject('tag')+'|'+effects", "3|3");
            ENTRIES.with(|entries| {
                assert_eq!(
                    &*entries.borrow(),
                    &[(start + interior, stack.at[start + interior].depth, 0)]
                )
            });
            ENTRIES.with(|entries| entries.borrow_mut().clear());
            native_eval(
                &mut engine,
                &code,
                "effects=0;subject('other')+'|'+effects",
                "5|5",
            );
            ENTRIES.with(|entries| {
                assert_eq!(
                    &*entries.borrow(),
                    &[(start + interior, stack.at[start + interior].depth, 0)]
                )
            });
            assert_eq!(chunk.optimizing_misses.get(), 0);
            assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
        }
    }
}

#[test]
fn cold_selection_recovers_before_unselected_comparison_without_credit_or_replay() {
    if child("cold_selection_recovers_before_unselected_comparison_without_credit_or_replay") {
        return;
    }
    let _recording = Recording::new(true);
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let setup = "var effects=0;function subject(x,cold){effects++;if(cold){if(x==='tag'){effects+=2;return effects;}effects+=4;}return effects;}subject('tag',false);subject('tag',false);";
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, setup);
    let function = function(&mut engine, "subject");
    let baseline = function.code.get().and_then(|code| code.clone()).unwrap();
    let mut chunk = bytecode::compile_for_optimizer(&function, &baseline, &[], &[]).unwrap();
    Rc::get_mut(&mut chunk).unwrap().optimizing_candidate = false;
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let regions =
        observed_regions::Plan::build(&chunk, &cfg, &stack, &chunk.execution_coverage).unwrap();
    assert!(Plan::build(&chunk, &cfg, &stack, Some(&regions), None).is_none());
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = diagnostics::with_test_mode("execution", || {
        Rc::new(compile_with_deopt(&chunk, &values, &layout, None).unwrap())
    });
    assert!(TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    let expected_pc = cfg
        .blocks()
        .iter()
        .find(|block| regions.cuts_block(cfg.block_at(block.start).unwrap().0 as usize))
        .unwrap()
        .start;
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native_eval(
        &mut engine,
        &code,
        "effects=0;subject('tag',true)+'|'+effects",
        "3|3",
    );
    ENTRIES.with(|entries| {
        assert_eq!(
            &*entries.borrow(),
            &[(expected_pc, stack.at[expected_pc].depth, 0)]
        )
    });
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
    assert_eq!(chunk.optimizing_misses.get(), 0);
    assert!(
        unsafe {
            &*code
                .optimizing_diagnostics
                .as_ref()
                .unwrap()
                .counter(diagnostics::VM_RECOVERY)
        }
        .get()
            > 0
    );
}

#[test]
fn selected_literal_comparison_enters_original_cold_successor_once() {
    if child("selected_literal_comparison_enters_original_cold_successor_once") {
        return;
    }
    let _recording = Recording::new(true);
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let setup = "var effects=0;function subject(x){effects++;if(x==='tag'){effects+=2;return effects;}effects+=4;return effects;}subject('other');subject('other');";
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, setup);
    let function = function(&mut engine, "subject");
    let baseline = function.code.get().and_then(|code| code.clone()).unwrap();
    let mut chunk = bytecode::compile_for_optimizer(&function, &baseline, &[], &[]).unwrap();
    Rc::get_mut(&mut chunk).unwrap().optimizing_candidate = false;
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let regions =
        observed_regions::Plan::build(&chunk, &cfg, &stack, &chunk.execution_coverage).unwrap();
    let literals = Plan::build(&chunk, &cfg, &stack, Some(&regions), None).unwrap();
    let sites: Vec<_> = (0..chunk.jit_ops().len())
        .filter(|&pc| literals.site(pc).is_some())
        .collect();
    assert_eq!(sites.len(), 1);
    let target = sites[0] + 4;
    assert_eq!(regions.emission(target), observed_regions::Emission::Cut);
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = diagnostics::with_test_mode("execution", || {
        Rc::new(compile_with_deopt(&chunk, &values, &layout, None).unwrap())
    });
    assert_eq!(TEST_SELECTED.with(|sites| sites.borrow().len()), 1);
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native_eval(
        &mut engine,
        &code,
        "effects=0;subject('tag')+'|'+effects",
        "3|3",
    );
    ENTRIES.with(|entries| assert_eq!(&*entries.borrow(), &[(target, stack.at[target].depth, 0)]));
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 1);
    assert_eq!(chunk.optimizing_misses.get(), 0);
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    native_eval(
        &mut engine,
        &code,
        "effects=0;subject('other')+'|'+effects",
        "5|5",
    );
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
}

#[test]
fn original_jump_handler_abrupt_and_inline_fallback_targets_reject_fusion() {
    let _recording = Recording::new(true);
    let mut engine = Engine::new();
    eval(
        &mut engine,
        "function subject(x){if(x==='tag')return 1;return 0;}",
    );
    let definition = function(&mut engine, "subject");
    let chunk = bytecode::compile(&definition).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let literals = Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
    let start = (0..chunk.jit_ops().len())
        .find(|&pc| literals.site(pc).is_some())
        .unwrap();
    for offset in 1..=3 {
        for root in [
            Op::Jump((start + offset) as u32),
            Op::PushHandler((start + offset) as u32),
            Op::AbruptJump((start + offset) as u32, 0),
            Op::InlineGuard(0, (start + offset) as u32),
        ] {
            let mut ops = chunk.jit_ops().to_vec();
            let mut effects: Vec<_> = (0..ops.len())
                .map(|pc| chunk.jit_stack_effect(pc))
                .collect();
            // A semantic target is mandatory even from a currently unreachable source.
            // This synthetic graph gate does not claim a runtime handler/inline execution.
            ops.push(root);
            effects.push(Some((0, 0)));
            let rooted = Cfg::test_build_with(&ops, |pc| effects[pc]).unwrap();
            assert_ne!(rooted.block_at(start), rooted.block_at(start + 3));
            assert!(Plan::build(&chunk, &rooted, &stack, None, None).is_none());
        }
    }
    // Also establish a reachable real handler root in the production graph builder.
    let mut engine = Engine::new();
    eval(
        &mut engine,
        "function subject(x){try{if(x==='tag')return 1;return 0;}catch(e){return -1;}}",
    );
    let function = function(&mut engine, "subject");
    let chunk = bytecode::compile(&function).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let literals = Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
    let start = (0..chunk.jit_ops().len())
        .find(|&pc| literals.site(pc).is_some())
        .unwrap();
    let mut ops = chunk.jit_ops().to_vec();
    let push = ops
        .iter()
        .position(|op| matches!(op, Op::PushHandler(_)))
        .unwrap();
    ops[push] = Op::PushHandler((start + 1) as u32);
    let rooted = Cfg::test_build_with(&ops, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(rooted
        .handler_roots()
        .iter()
        .any(|root| rooted.blocks()[root.target.0 as usize].start == start + 1));
    assert!(Plan::build(&chunk, &rooted, &stack, None, None).is_none());
}

#[test]
fn disabled_fusion_retains_checked_semantics_and_no_literal_leaf_execution() {
    if child("disabled_fusion_retains_checked_semantics_and_no_literal_leaf_execution") {
        return;
    }
    let _recording = Recording::new(false);
    let setup = "function subject(x){if(x==='tag')return 1;return 0;}";
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (_, code, sites) = install(&mut engine, "subject", None);
    assert!(sites.is_empty());
    native_eval(&mut engine, &code, "subject('tag')+'|'+subject({})", "1|0");
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
}

thread_local! {
    static PARKED: std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>> = const { std::cell::RefCell::new(None) };
    static PARKED_COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
    PARKED.with(|parked| {
        assert_eq!(
            parked.borrow().as_ref().unwrap().strong_count(),
            1,
            "the parked canonical operand is the sole object owner"
        )
    });
    i.gc_collect();
    PARKED.with(|parked| {
        assert_eq!(
            parked.borrow().as_ref().unwrap().strong_count(),
            1,
            "opaque prefix survives reentrant collection"
        )
    });
    PARKED_COLLECTIONS.with(|count| count.set(count.get() + 1));
    Ok(JsValue::Undefined)
}

#[test]
fn parked_handler_fixture_executes_floor_positive_fusion_and_preserves_sole_owner() {
    if child("parked_handler_fixture_executes_floor_positive_fusion_and_preserves_sole_owner") {
        return;
    }
    let source = "function subject(x){var keep=makeTracked();try{try{keep=null;throw 1;}catch(e){forceCollection();if(x==='tag')return 1;return 2;}}catch(e){return 0;}}";
    // This complete Chunk parks a fixture temporary before the inner handler. The parser
    // and current baseline inliner do not naturally produce this owner prefix here.
    for tier in [Tier::Bytecode, Tier::Jit] {
        for recover_after_comparison in [false, true] {
            if tier == Tier::Bytecode && recover_after_comparison {
                continue;
            }
            let _recording = Recording::new(true);
            PARKED_COLLECTIONS.with(|count| count.set(0));
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
            let (chunk, parked, slot) =
                bytecode::compile_test_parked_handler_owner(&definition, "keep", 2).unwrap();
            assert!(matches!(chunk.jit_ops()[parked],Op::LoadLocal(s) if s==slot));
            assert!(matches!(chunk.jit_ops()[parked + 1], Op::PushHandler(_)));
            let cfg = Cfg::build(&chunk).unwrap();
            let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
            let literals = Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
            let site = (0..chunk.jit_ops().len())
                .find(|&pc| literals.site(pc).is_some())
                .unwrap();
            assert_eq!(stack.at[site].floor, 1);
            assert_eq!(stack.at[site].depth, 1);
            let code = if tier == Tier::Jit {
                let values = crate::value::jit_layout(&engine.interp.object_proto);
                let layout = crate::interpreter::interp_layout(&mut engine.interp);
                let code = diagnostics::with_test_mode("execution", || {
                    Rc::new(
                        compile_with_deopt(
                            &chunk,
                            &values,
                            &layout,
                            recover_after_comparison.then_some(site + 4),
                        )
                        .unwrap(),
                    )
                });
                assert_eq!(TEST_SELECTED.with(|sites| sites.borrow().len()), 1);
                chunk
                    .jit
                    .begin_compile()
                    .unwrap()
                    .commit(Some(code.clone()));
                assert!(definition.optimizing_code.set(Some(chunk.clone())).is_ok());
                Some(code)
            } else {
                assert!(definition.code.set(Some(chunk.clone())).is_ok());
                None
            };
            bytecode::invalidate_call_caches();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            for (input, want) in [("tag", "1"), ("other", "2")] {
                ENTRIES.with(|entries| entries.borrow_mut().clear());
                let call = format!("subject('{input}')");
                if let Some(code) = &code {
                    TEST_LEAF_ENTRIES.with(|count| count.set(0));
                    native_eval(&mut engine, code, &call, want);
                    assert!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get) > 0);
                    ENTRIES.with(|entries| {
                        if recover_after_comparison && input == "tag" {
                            assert_eq!(&*entries.borrow(), &[(site + 4, 1, 0)]);
                        } else {
                            assert!(entries.borrow().is_empty());
                        }
                    });
                } else {
                    assert_eq!(eval(&mut engine, &call), want);
                }
                assert_eq!(chunk.optimizing_misses.get(), 0);
                assert!(engine.interp.fn_frames.is_empty());
                PARKED.with(|parked| {
                    assert_eq!(
                        parked.borrow().as_ref().unwrap().strong_count(),
                        0,
                        "completed frame releases opaque owner"
                    )
                });
            }
            assert_eq!(PARKED_COLLECTIONS.with(std::cell::Cell::get), 2);
            // Scoped exact-Function fixture restoration keeps a later ordinary compilation
            // from acquiring the fixture's parked operand owner.
            let ordinary = bytecode::compile(&definition).unwrap();
            let ordinary_cfg = Cfg::build(&ordinary).unwrap();
            let ordinary_stack = stack::StackPlan::build(&ordinary, &ordinary_cfg).unwrap();
            let ordinary_literals =
                Plan::build(&ordinary, &ordinary_cfg, &ordinary_stack, None, None).unwrap();
            let ordinary_site = (0..ordinary.jit_ops().len())
                .find(|&pc| ordinary_literals.site(pc).is_some())
                .unwrap();
            assert_eq!(ordinary_stack.at[ordinary_site].floor, 0);
            assert_eq!(ordinary.jit_ops().len() + 1, chunk.jit_ops().len());
        }
    }
    let _ = PARKED.with(|parked| parked.borrow_mut().take());
}

fn facts_children(name: &str) -> Option<bool> {
    const CONTROL: &str = "TRUST_JS_TEST_LITERAL_FACTS_CHILD";
    if std::env::var("TRUST_JS_TEST_STRING_LITERAL_CHILD").as_deref() == Ok(name) {
        let enabled = std::env::var(CONTROL).expect("explicit value-facts mode");
        assert!(matches!(enabled.as_str(), "0" | "1"));
        assert_eq!(
            std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").unwrap(),
            enabled
        );
        return Some(enabled == "1");
    }
    for enabled in ["0", "1"] {
        assert!(child_with(
            name,
            &[
                (CONTROL, enabled),
                ("TRUST_JS_OPT_JIT_VALUE_FACTS", enabled)
            ],
        ));
    }
    None
}

#[test]
fn original_load_local_initialized_proof_omits_only_redundant_empty_guards() {
    let Some(facts) =
        facts_children("original_load_local_initialized_proof_omits_only_redundant_empty_guards")
    else {
        return;
    };
    let _recording = Recording::new(true);
    let setup = "function subject(x){if(x==='a')return 1;if('b'===x)return 2;if(x==='c')return 3;return 4;}";
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (chunk, code, sites) = install(&mut engine, "subject", None);
    assert_eq!(sites.len(), 3);
    assert_eq!(
        TEST_EMPTY_GUARDS.with(std::cell::Cell::get),
        if facts { 1 } else { 3 }
    );
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let values = value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
    assert!(!values.at[sites[0]].local.initialized());
    // The second comparison is reversed. Its Const input at the group head is
    // deliberately not an initialized LoadLocal proof; its actual read at+1 is.
    assert!(matches!(chunk.jit_ops()[sites[1]], Op::Const(_)));
    assert!(!values.at[sites[1]].local.initialized());
    assert!(values.at[sites[1] + 1].local.initialized());
    assert!(values.at[sites[2]].local.initialized());
    for (input, want, leaves) in [
        ("'a'", "1", 1),
        ("'b'", "2", 2),
        ("'c'", "3", 3),
        ("'none'", "4", 3),
        ("new String('a')", "4", 3),
    ] {
        TEST_LEAF_ENTRIES.with(|count| count.set(0));
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        native_eval(&mut engine, &code, &format!("subject({input})"), want);
        assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), leaves);
        ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    }
    assert_eq!(chunk.optimizing_misses.get(), 0);
}

#[test]
fn absent_or_empty_load_local_proof_keeps_reversed_before_start_recovery() {
    let Some(facts) =
        facts_children("absent_or_empty_load_local_proof_keeps_reversed_before_start_recovery")
    else {
        return;
    };
    let _recording = Recording::new(true);
    let setup = "function subject(x){if(x==='a')return 1;if('b'===y)return 2;let y='b';return 3;}";
    let call = "try{subject('no')}catch(e){e.name+'|'+e.message+'|'+e.stack}";
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, setup);
    let expected = eval(&mut oracle, call);
    assert!(expected.starts_with("ReferenceError|"));
    let mut engine = Engine::new();
    eval(&mut engine, setup);
    let (chunk, code, sites) = install(&mut engine, "subject", None);
    assert_eq!(sites.len(), 2);
    assert_eq!(
        TEST_EMPTY_GUARDS.with(std::cell::Cell::get),
        2,
        "facts={facts}: actual Empty read must never be elided"
    );
    native_eval(&mut engine, &code, "subject('a')", "1");
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    TEST_LEAF_ENTRIES.with(|count| count.set(0));
    native_eval(&mut engine, &code, call, &expected);
    assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 1);
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let original = sites[1];
    assert!(matches!(chunk.jit_ops()[original], Op::Const(_)));
    ENTRIES.with(|entries| {
        assert_eq!(
            &*entries.borrow(),
            &[(original, stack.at[original].depth, 0)]
        )
    });
    assert_eq!(chunk.optimizing_misses.get(), 0);
}

fn literal_policy(engine: &mut Engine, source: &str) -> (Rc<Chunk>, Cfg, stack::StackPlan, Plan) {
    eval(engine, source);
    let chunk = bytecode::compile(&function(engine, "subject")).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let plan = Plan::build(&chunk, &cfg, &stack, None, None).unwrap();
    (chunk, cfg, stack, plan)
}

#[test]
fn owner_credit_counts_only_constant_operations_after_both_reads() {
    if child("owner_credit_counts_only_constant_operations_after_both_reads") {
        return;
    }
    let _recording = Recording::new(true);
    for source in [
        "function subject(x){if(x==='tag')return 1;return 0;}",
        "function subject(x){if('tag'!==x)return 1;return 0;}",
        "function subject(){var x='tag';if(x==='tag')return 1;return 0;}",
        "function subject(){var x=1;if(x==='tag')return 1;return 0;}",
        "function subject(){var x=true;if(x==='tag')return 1;return 0;}",
    ] {
        let mut engine = Engine::new();
        let (chunk, cfg, stack, plan) = literal_policy(&mut engine, source);
        let pc = (0..chunk.jit_ops().len())
            .find(|&pc| plan.site(pc).is_some())
            .unwrap();
        assert_eq!(plan.selected_sites(), 1);
        assert_eq!(plan.credit_at(pc), 0);
        assert_eq!(plan.credit_at(pc + 1), 0);
        assert_eq!(plan.credit_at(pc + 2), 1);
        assert_eq!(plan.credit_at(pc + 3), 0);
        let values = value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
        assert_eq!(
            generic_admission::minimum_saved_work(
                &chunk,
                &cfg,
                None,
                None,
                None,
                None,
                Some(&plan),
                Some(&values)
            ),
            Some(1)
        );
        assert!(plan.payload_bytes() >= chunk.jit_ops().len());
    }
    let mut engine = Engine::new();
    let (chunk, cfg, stack, plan) = literal_policy(
        &mut engine,
        "function subject(){if('tag'===x)return 1;let x='tag';return 0;}",
    );
    let values = value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
    assert!(
        generic_admission::minimum_saved_work(
            &chunk,
            &cfg,
            None,
            None,
            None,
            None,
            Some(&plan),
            Some(&values)
        )
        .is_none(),
        "TDZ exits before constant owner work is removed"
    );

    let mut engine = Engine::new();
    let (chunk, cfg, stack, plan) = literal_policy(
        &mut engine,
        "function subject(x){if(x==='tag')return 1;return 0;}",
    );
    let numeric = specialization::Plan::build(
        &chunk,
        &cfg,
        &stack,
        &[(0, crate::feedback::ValueClass::NumberInt32.bit())],
        &[],
    )
    .unwrap();
    let without = generic_admission::minimum_saved_work(
        &chunk,
        &cfg,
        Some(&numeric),
        None,
        None,
        None,
        None,
        Some(&numeric.values),
    )
    .unwrap();
    let combined = generic_admission::minimum_saved_work(
        &chunk,
        &cfg,
        Some(&numeric),
        None,
        None,
        None,
        Some(&plan),
        Some(&numeric.values),
    )
    .unwrap();
    assert_eq!(
        combined,
        without + 1,
        "one constant owner-work unit, existing input guard charged once"
    );
}

#[test]
fn before_credit_author_effect_and_unreachable_sites_cannot_pay_for_literal_work() {
    if child("before_credit_author_effect_and_unreachable_sites_cannot_pay_for_literal_work") {
        return;
    }
    let _recording = Recording::new(true);
    let mut engine = Engine::new();
    let (chunk, cfg, stack, plan) = literal_policy(
        &mut engine,
        "function subject(x){observe();if(x==='tag')return 1;return 0;}",
    );
    let values = value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
    assert!(generic_admission::minimum_saved_work(
        &chunk,
        &cfg,
        None,
        None,
        None,
        None,
        Some(&plan),
        Some(&values)
    )
    .is_none());
    for pc in 0..chunk.jit_ops().len() {
        if plan.site(pc).is_some() {
            for interior in pc + 1..pc + 4 {
                assert!(Plan::build(&chunk, &cfg, &stack, None, Some(interior)).is_none());
            }
        }
    }
    let mut live = vec![true; chunk.jit_ops().len() + 1];
    let pc = (0..chunk.jit_ops().len())
        .find(|&pc| plan.site(pc).is_some())
        .unwrap();
    live[pc + 2] = false;
    assert!(
        plan.restrict_reachable(Some(&live)).is_none(),
        "a site omitted by the actual Object branch plan cannot get credit"
    );
}

#[test]
fn full_original_literal_analysis_work_and_overflow_are_explicit() {
    if child("full_original_literal_analysis_work_and_overflow_are_explicit") {
        return;
    }
    let _recording = Recording::new(true);
    assert_eq!(analysis_work(10), Some(150));
    assert_eq!(analysis_work(4096), Some(61440));
    assert_eq!(analysis_work_with_filter(10, false), Some(150));
    assert_eq!(analysis_work_with_filter(10, true), Some(180));
    assert!(analysis_work(usize::MAX).is_none());
    assert!(analysis_work_with_filter(usize::MAX, true).is_none());
    let mut engine = Engine::new();
    let (chunk, cfg, stack, _) = literal_policy(
        &mut engine,
        "function subject(x){if(x==='tag')return 1;return 0;}",
    );
    let work = analysis_work(chunk.jit_ops().len()).unwrap();
    let failed = value_facts::ValuePlan::with_budget(&chunk, &cfg, &stack, 0, usize::MAX);
    assert!(
        failed.is_none(),
        "real bounded solver returns no partial proof"
    );
    let attempted = checkpoint_value_work(
        chunk.jit_ops().len(),
        chunk.jit_frame().1,
        stack.capacity,
        true,
    )
    .unwrap();
    assert!(
        attempted > 0,
        "failed analysis still has positive checkpoint work"
    );
    assert!(value_facts::ValuePlan::with_budget(&chunk, &cfg, &stack, usize::MAX, 0).is_none());
    assert_eq!(checkpoint_value_work(10, 2, 3, true), Some(120));
    assert_eq!(
        checkpoint_value_work(usize::MAX, usize::MAX, usize::MAX, false),
        Some(0)
    );
    assert!(checkpoint_value_work(10, usize::MAX, 1, true).is_none());
    assert!(checkpoint_value_work(usize::MAX, 1, 1, true).is_none());
    assert!(generic_admission::admission_entries_with_work(&chunk, 0, work).is_none());
    assert!(generic_admission::admission_entries_with_work(&chunk, 1, u64::MAX).is_none());
    assert!(generic_admission::add_analysis_work(u64::MAX, 1, 1).is_none());
    assert!(generic_admission::add_analysis_work(1, 0, 1).is_none());
    let prior = generic_admission::admission_entries(&chunk, 1, 0).unwrap();
    let priced = generic_admission::admission_entries_with_work(&chunk, 1, work).unwrap();
    assert_eq!(priced - prior, work * 1024);
}

#[test]
fn large_owner_credit_stops_at_cuts_but_prices_the_full_original_tail() {
    if child_with(
        "large_owner_credit_stops_at_cuts_but_prices_the_full_original_tail",
        &[
            ("TRUST_JS_OPT_JIT_LARGE_REGIONS", "1"),
            ("TRUST_JS_OPT_JIT_OBSERVED_REGIONS", "1"),
        ],
    ) {
        return;
    }
    let _recording = Recording::new(true);
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let mut source = "function subject(x,flag){var sum=0;if(x==='tag')sum=1;if(flag){".to_owned();
    for _ in 0..80 {
        source.push_str("if(x==='cold')sum=sum+1;");
    }
    source.push_str("}return sum;}");
    let mut engine = Engine::new();
    eval(&mut engine, &source);
    let definition = function(&mut engine, "subject");
    let primary = bytecode::compile(&definition).unwrap();
    assert!(primary.jit_ops().len() > 256);
    assert!(definition.code.set(Some(primary.clone())).is_ok());
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    assert_eq!(eval(&mut engine, "subject('other',false)"), "0");
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let frozen = primary.execution_coverage.freeze();
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen).unwrap();
    let full = Plan::build(&primary, &cfg, &stack, None, None).unwrap();
    assert_eq!(full.selected_sites(), 81);
    let selected = Plan::build(&primary, &cfg, &stack, Some(&regions), None).unwrap();
    assert_eq!(selected.selected_sites(), 1);
    assert_eq!(
        large_regions::opportunity_work(&primary, &cfg, &regions, None, None, Some(&selected)),
        Some(1)
    );
    assert!(large_regions::opportunity_tests(&primary, &cfg, &regions, None, None).is_none());
    let frame = frame::FramePlan::build(&primary, &cfg, true).unwrap();
    let old = large_regions::work(&primary, &cfg, &stack, &frame, &regions).unwrap();
    let priced =
        large_regions::work_with_literals(&primary, &cfg, &stack, &frame, &regions).unwrap();
    assert_eq!(priced.literal_analysis, primary.jit_ops().len() as u64 * 15);
    assert_eq!(
        priced.total().unwrap() - old.total().unwrap(),
        priced.literal_analysis
    );
    assert!(
        priced.literal_analysis > regions.native_ops as u64 * 15,
        "omitted tail scan/init cost remains priced"
    );
    for pc in 0..primary.jit_ops().len() {
        if selected.credit_at(pc) != 0 {
            assert_eq!(regions.emission(pc), observed_regions::Emission::Native);
        }
    }
}

#[test]
fn ordinary_hot_literal_only_and_zero_formal_bodies_publish_without_threshold_override() {
    const NAME: &str =
        "ordinary_hot_literal_only_and_zero_formal_bodies_publish_without_threshold_override";
    if child_with(
        NAME,
        &[
            ("TRUST_JS_OPT_JIT", "hot"),
            ("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution"),
        ],
    ) {
        return;
    }
    assert_eq!(std::env::var("TRUST_JS_OPT_JIT").as_deref(), Ok("hot"));
    assert!(std::env::var("TRUST_JS_OPT_JIT_HOT_AT").is_err());
    let _recording = Recording::new(true);
    for source in [
        "function subject(x){if(x==='tag')return 1;return 0;}",
        "function subject(){var x='tag';if('tag'===x)return 1;return 0;}",
    ] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let definition = function(&mut engine, "subject");
        assert_eq!(eval(&mut engine, "subject('tag')"), "1");
        let baseline = definition
            .execution_code()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        let cfg = Cfg::build(&baseline).unwrap();
        let stack = stack::StackPlan::build(&baseline, &cfg).unwrap();
        let values = value_facts::ValuePlan::build(&baseline, &cfg, &stack).unwrap();
        let literal = Plan::build(&baseline, &cfg, &stack, None, None).unwrap();
        let credit = generic_admission::minimum_saved_work(
            &baseline,
            &cfg,
            None,
            None,
            None,
            None,
            Some(&literal),
            Some(&values),
        )
        .unwrap();
        assert_eq!(credit, 1);
        let mut work = analysis_work(baseline.jit_ops().len()).unwrap()
            + native_pcs::analysis_work(baseline.jit_ops().len()).unwrap();
        // Group construction is attempted even when this source has no contiguous
        // switch chain; the original scan must not be free merely because it fails.
        work += string_dispatch::analysis_work(baseline.jit_ops().len()).unwrap();
        work += checkpoint_value_work(
            baseline.jit_ops().len(),
            baseline.jit_frame().1,
            stack.capacity,
            true,
        )
        .unwrap();
        let target = generic_admission::admission_entries_with_work(&baseline, credit, work)
            .unwrap()
            .max(4096);
        assert!(target > u64::from(hot_threshold(baseline.jit_ops().len())) * 4);
        let calls = target + u64::from(hot_threshold(baseline.jit_ops().len())) * 4 + 16;
        assert_eq!(
            eval(
                &mut engine,
                &format!("var sum=0;for(var k=0;k<{calls};k++)sum+=subject('tag');sum")
            ),
            calls.to_string()
        );
        let candidate = definition
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .expect("ordinary policy literal publication");
        let code = candidate.jit.get().flatten().unwrap();
        let json = code.optimizing_diagnostics.as_ref().unwrap().json();
        assert!(json.contains("\"selected_native_sites\":1"));
        assert!(json.contains("\"emitted_sites\":1"));
        assert!(json.contains("\"constant_owner_operations_removed_per_site\":2"));
        assert!(json.contains("\"borrowed_tag_debit_per_site\":1"));
        assert!(json.contains("\"net_semantic_owner_work_per_site\":1"));
        assert!(json.contains("\"site_count_is_not_path_minimum\":true"));
        assert!(!json.contains("\"union_not_minimum_claim\""));
        assert!(candidate
            .optimizing_inputs
            .iter()
            .all(|(_, bits)| *bits == u32::MAX));
        assert!(candidate.optimizing_structural.is_empty());
        assert!(candidate.optimizing_properties.is_empty());
        assert!(candidate.optimizing_samples.get().is_none());
        assert!(baseline.optimizing_samples.get().unwrap().samples.get() >= 4);
        assert!(baseline.optimizing_samples.get().unwrap().entries.get() >= target);
        TEST_LEAF_ENTRIES.with(|count| count.set(0));
        native_eval(&mut engine, &code, "subject('tag')", "1");
        assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 1);
        assert_eq!(candidate.optimizing_misses.get(), 0);
    }
}

#[test]
fn disabled_literal_certificate_cannot_admit_literal_only_or_zero_formal_bodies() {
    const NAME: &str =
        "disabled_literal_certificate_cannot_admit_literal_only_or_zero_formal_bodies";
    if child_with(NAME, &[("TRUST_JS_OPT_JIT", "hot")]) {
        return;
    }
    let _recording = Recording::new(false);
    for source in [
        "function subject(x){if(x==='tag')return 1;return 0;}",
        "function subject(){var x='tag';if('tag'===x)return 1;return 0;}",
    ] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        assert_eq!(eval(&mut engine, "subject('tag')"), "1");
        let definition = function(&mut engine, "subject");
        let baseline = definition
            .execution_code()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        let calls = u64::from(hot_threshold(baseline.jit_ops().len())) * 4 + 32;
        assert_eq!(
            eval(
                &mut engine,
                &format!("var sum=0;for(var k=0;k<{calls};k++)sum+=subject('tag');sum")
            ),
            calls.to_string()
        );
        assert!(definition.optimizing_code.get().is_none());
        assert_eq!(baseline.optimizing_samples.get().unwrap().samples.get(), 4);
        assert_eq!(baseline.optimizing_remaining.get(), 0);
        assert_eq!(TEST_LEAF_ENTRIES.with(std::cell::Cell::get), 0);
        let code = baseline.jit.get().flatten().unwrap();
        assert!(code.residency.referenced.get() > 0);
        assert_eq!(code.residency.active.get(), 0);
    }
}
