//! Bounded execution observations: actual entries, committed sites and exact recovery.
use super::*;
use crate::value::{Callable, Value as JsValue};
use crate::{bytecode::Tier, Completion, Engine};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

thread_local! {
    static CAPTURE: Cell<bool> = const { Cell::new(false) };
    static GRAPH: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(super) fn capture_graph(function: &cranelift_codegen::ir::Function, residency: usize) {
    if !CAPTURE.with(Cell::get) {
        return;
    }
    let graph = function
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
                    if parsed == Some(residency as u64) {
                        "<residency>"
                    } else {
                        word
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n");
    GRAPH.with(|current| *current.borrow_mut() = graph);
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("execution fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn parts(engine: &mut Engine, name: &str) -> (Rc<crate::ast::Function>, Rc<Chunk>) {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary user function")
    };
    (
        user.func.clone(),
        user.func
            .code
            .get()
            .and_then(Option::as_ref)
            .unwrap()
            .clone(),
    )
}

fn install(
    engine: &mut Engine,
    name: &str,
    inputs: &[(u16, u32)],
    properties: &[(u32, u32)],
    deopt: Option<usize>,
) -> (Rc<Chunk>, Rc<JitCode>) {
    let (function, baseline) = parts(engine, name);
    let mut chunk =
        bytecode::compile_for_optimizer(&function, &baseline, inputs, properties).unwrap();
    if inputs.is_empty() && properties.is_empty() {
        // Force generic native coverage; hot versions still require a positive plan.
        Rc::get_mut(&mut chunk).unwrap().optimizing_candidate = false;
    }
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(compile_with_deopt(&chunk, &values, &layout, deopt).unwrap());
    assert!(code.pc_offsets.is_empty(), "actual optimizing body");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    (chunk, code)
}

fn child(test: &str) -> bool {
    const FLAG: &str = "TRUST_JS_TEST_EXECUTION_DIAGNOSTIC_CHILD";
    if std::env::var_os(FLAG).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("jit::optimizing::execution_tests::{test}"),
            "--nocapture",
        ])
        .env(FLAG, "1")
        .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution")
        .env("TRUST_JS_OPT_JIT", "0")
        .env("TRUST_JS_INLINE_AT", "0")
        .env_remove("TRUST_JS_OPT_JIT_DEOPT_AT")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

#[test]
fn actual_direct_entries_and_committed_property_sites() {
    if child("actual_direct_entries_and_committed_property_sites") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine,"var source={x:3,y:0};function leaf(o){o.y=o.x;return o.y;}function driver(o){return leaf(o);}for(var n=0;n<5;n++)driver(source);");
    let (_, code) = install(&mut engine, "leaf", &[], &[], None);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert!(record.execution_counters);
    assert!(!record.runtime_counters);
    #[cfg(target_arch = "aarch64")]
    crate::jit::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    assert_eq!(
        eval(
            &mut engine,
            "source.x=source;for(var n=0;n<200;n++)driver(source);source.y===source"
        ),
        "true"
    );
    assert_eq!(record.counters[0].get(), 200);
    assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 200);
    assert_eq!(record.counters[diagnostics::ERROR_EXIT].get(), 0);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 0);
    let sites = record.site_stats();
    assert_eq!(sites.len(), 3);
    assert!(
        sites.iter().all(|&(_, _, success)| success == 200),
        "each read/store commits exactly once: {sites:?}"
    );
    assert!(sites
        .iter()
        .any(|&(_, kind, _)| kind == "existing_property_overwrite"));
    assert!(
        record.counters[1..=diagnostics::HEAP_GUARD]
            .iter()
            .all(|count| count.get() == 0),
        "no full helper/owner instrumentation"
    );
    #[cfg(target_arch = "aarch64")]
    assert!(
        crate::jit::TEST_DIRECT_PACKED_RETURNS.with(Cell::get) > 100,
        "actual raw shared-frame calls counted"
    );
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn terminal_recovery_invokes_getter_and_coercion_once() {
    if child("terminal_recovery_invokes_getter_and_coercion_once") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "var source={x:3},reads=0,conversions=0;function subject(o){return o.x+1;}subject(source);",
    );
    let (_, baseline) = parts(&mut engine, "subject");
    let pc = baseline
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::GetPropLocal(..)))
        .unwrap();
    let (_, code) = install(
        &mut engine,
        "subject",
        &[],
        &[(pc as u32, crate::feedback::ValueClass::NumberInt32.bit())],
        None,
    );
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(eval(&mut engine, "subject(source)"), "4");
    assert_eq!(eval(&mut engine,"Object.defineProperty(source,'x',{configurable:true,get(){reads++;return {valueOf(){conversions++;return 6;}};}});[subject(source),reads,conversions].join('|')"),"7|1|1");
    assert_eq!(eval(&mut engine,"Object.defineProperty(source,'x',{get(){reads++;throw 'boom';}});try{subject(source);}catch(e){[e,reads,conversions].join('|');}"),"boom|2|1");
    assert_eq!(record.counters[0].get(), 3);
    assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 2);
    assert_eq!(record.counters[diagnostics::ERROR_EXIT].get(), 1);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        2
    );
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 2);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY_NORMAL].get(), 1);
    assert_eq!(record.site_stats(), vec![(pc, "data_property_read", 1)]);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn saturation_retirement_and_directory_do_not_retain_captured_objects() {
    if child("saturation_retirement_and_directory_do_not_retain_captured_objects") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine,"var payload={marker:3};var subject=(function(captured){return function(x){return x+x+1+captured.marker;};})(payload);subject(3);");
    let env = engine.interp.global_env.clone();
    let JsValue::Obj(payload) = engine.interp.get_var("payload", &env).ok().unwrap() else {
        panic!("payload")
    };
    let weak_payload = Rc::downgrade(&payload);
    drop(payload);
    let (chunk, code) = install(
        &mut engine,
        "subject",
        &[(0, crate::feedback::ValueClass::NumberInt32.bit())],
        &[],
        None,
    );
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    record.counters[0].set(u64::MAX - 1);
    assert_eq!(eval(&mut engine, "subject(2);subject(3)"), "10");
    assert_eq!(record.counters[0].get(), u64::MAX);
    assert!(record
        .json()
        .contains("\"saturated_counter_labels\":[\"entries\"]"));
    assert_eq!(
        eval(&mut engine, "for(var n=0;n<8;n++)subject('x');'ok'"),
        "ok"
    );
    assert_eq!(chunk.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert_eq!(record.counters[diagnostics::RETIREMENT_REQUEST].get(), 1);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        8
    );
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 8);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY_NORMAL].get(), 8);
    assert_eq!(
        eval(&mut engine, "subject=null;payload=null;'gone'"),
        "gone"
    );
    engine.interp.gc_collect();
    assert_eq!(
        weak_payload.strong_count(),
        0,
        "code counters and Weak directory retain no captured JS owner"
    );
    let weak_record = Rc::downgrade(record);
    drop(engine);
    drop(chunk);
    drop(code);
    assert!(
        weak_record.upgrade().is_none(),
        "directory retains no code/counter owner"
    );
}

#[test]
fn off_compile_graph_parity_and_bounded_execution_overhead() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function subject(x){return x+x+1;}subject(3);");
    let (_, chunk) = parts(&mut engine, "subject");
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    CAPTURE.with(|capture| capture.set(true));
    let mut results = Vec::new();
    for mode in ["0", "compile", "execution", "1"] {
        let code = diagnostics::with_test_mode(mode, || {
            compile_with_deopt(&chunk, &values, &layout, None).unwrap()
        });
        let graph = GRAPH.with(|current| current.borrow().clone());
        let size = stores::TEST_BODY_SIZE.with(Cell::get);
        if mode == "0" {
            assert!(code.optimizing_diagnostics.is_none());
        }
        if let Some(record) = &code.optimizing_diagnostics {
            assert!(record.counters.iter().all(|count| count.get() == 0));
        }
        results.push((mode, graph, size, code.len));
    }
    CAPTURE.with(|capture| capture.set(false));
    assert_eq!(results[0].1,results[1].1,"all IR except independent residency address is identical when runtime counters are disabled");
    assert_eq!(results[0].2, results[1].2);
    assert_eq!(results[0].3, results[1].3);
    assert!(
        results[2].2 .0 > results[1].2 .0,
        "execution instrumentation has an explicit IR cost"
    );
    assert!(
        results[2].2 .0 < results[3].2 .0,
        "small body uses fewer counters than full diagnostics"
    );
    if let Some(path) = std::env::var_os("TRUST_JS_TEST_EXECUTION_ARTIFACT_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        for (mode, graph, size, bytes) in results {
            std::fs::write(path.join(format!("{mode}.clif")), graph).unwrap();
            std::fs::write(path.join(format!("{mode}.json")),format!("{{\"mode\":\"{mode}\",\"ir_instructions\":{},\"ir_blocks\":{},\"native_bytes\":{bytes}}}\n",size.0,size.1)).unwrap();
        }
    }
}

#[test]
fn selected_sites_are_bounded_and_omissions_are_not_extrapolated() {
    if child("selected_sites_are_bounded_and_omissions_are_not_extrapolated") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine,"var source={a:1,b:2,c:3,d:4,e:5,f:6,g:7,h:8,i:9,j:10,k:11,l:12};function subject(o){return o.a+o.b+o.c+o.d+o.e+o.f+o.g+o.h+o.i+o.j+o.k+o.l;}subject(source);");
    let (_, code) = install(&mut engine, "subject", &[], &[], None);
    assert_eq!(eval(&mut engine, "subject(source);subject(source)"), "78");
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    let sites = record.site_stats();
    assert_eq!(sites.len(), 8);
    assert!(sites.windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert!(sites
        .iter()
        .all(|&(_, kind, success)| kind == "data_property_read" && success == 2));
    assert!(record.json().contains("\"omitted_site_count\":4"));
    assert!(record
        .json()
        .contains("\"omitted_pc_list_truncated\":false"));
    assert!(record.json().contains("no estimate for omitted sites"));
    assert_eq!(record.counters[0].get(), 2);
    assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 2);
}

#[test]
fn data_read_scope_includes_prototypes_and_marks_unicode_label_truncation() {
    if child("data_read_scope_includes_prototypes_and_marks_unicode_label_truncation") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine,"var proto={x:3};var source=Object.create(proto);function subject(o){return o.x;}subject(source);");
    let (_, baseline) = parts(&mut engine, "subject");
    let cache = baseline
        .jit_ops()
        .iter()
        .find_map(|op| match op {
            Op::GetPropLocal(_, _, cache) => Some(*cache),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        baseline.jit_cache_preferred(cache).unwrap().depth,
        1,
        "supported prototype holder, not an own receiver field"
    );
    let (_, code) = install(&mut engine, "subject", &[], &[], None);
    assert_eq!(eval(&mut engine, "subject(source);subject(source)"), "3");
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.site_stats()[0].1, "data_property_read");
    assert_eq!(record.site_stats()[0].2, 2);
    assert_eq!(eval(&mut engine,"var reads=0;Object.defineProperty(proto,'x',{get(){reads++;return 7;}});[subject(source),reads].join('|')"),"7|1");
    assert_eq!(
        record.site_stats()[0].2,
        2,
        "checked getter success is outside native data-read credit"
    );
    assert_eq!(
        record.counters[diagnostics::VM_RECOVERY].get(),
        0,
        "generic checked read rejoins native control flow"
    );
    assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 3);
    let key = "界".repeat(65);
    eval(&mut engine,&format!("var unicodeSource={{'{key}':9}};function unicodeSubject(o){{return o.{key};}}unicodeSubject(unicodeSource);"));
    let (_, unicode) = install(&mut engine, "unicodeSubject", &[], &[], None);
    assert_eq!(eval(&mut engine, "unicodeSubject(unicodeSource)"), "9");
    let json = unicode.optimizing_diagnostics.as_ref().unwrap().json();
    assert!(json.contains("\"key_truncated\":true"));
    assert!(json.contains(&"界".repeat(64)));
    assert!(!json.contains(&key));
    assert!(json.contains("\"slot_label_limit_chars\":32"));
    assert!(json.contains("\"key_label_limit_chars\":64"));
}

#[test]
fn actual_loop_continuation_retirement_is_counted_once() {
    const FLAG: &str = "TRUST_JS_TEST_EXEC_LOOP_RETIRE_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let output=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","jit::optimizing::execution_tests::actual_loop_continuation_retirement_is_counted_once","--nocapture"])
            .env(FLAG,"1").env("TRUST_JS_OPT_JIT_DIAGNOSTICS","execution")
            .env("TRUST_JS_OPT_JIT","0").env("TRUST_JS_OPT_JIT_OSR","1")
            .env("TRUST_JS_OPT_JIT_OSR_AT","4").env("TRUST_JS_INLINE_AT","0")
            .env_remove("TRUST_JS_OPT_JIT_DEOPT_AT").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(eval(&mut engine,"var prefixes=0,coercions=0;function ping(x){return x}function subject(n,seed,change){prefixes++;var sum=seed;for(var k=0;k<n;k++){if(change&&k===20)sum={valueOf(){coercions++;return 10}};sum+=1;ping(k)}return sum}[subject(100,0,true),prefixes,coercions].join(':')"),"90:1:1");
    let (_, chunk) = parts(&mut engine, "subject");
    let state = chunk.optimizing_loop.get().unwrap();
    let primary = chunk.jit.get().flatten().unwrap();
    let continuation = state.code.get().flatten().unwrap();
    assert!(continuation.pc_offsets.is_empty());
    let record = continuation.optimizing_diagnostics.as_ref().unwrap();
    assert!(record.counters[0].get() > 0, "actual continuation entry");
    let entry_before = record.counters[0].get();
    assert_eq!(eval(&mut engine,"var lengths=[];for(var r=0;r<9;r++)lengths.push(subject(100,'s',false).length);[lengths.join(','),prefixes,coercions].join(':')"),"101,101,101,101,101,101,101,101,101:10:1");
    assert_eq!(state.misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert_eq!(record.counters[diagnostics::RETIREMENT_REQUEST].get(), 1);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 8);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY_NORMAL].get(), 8);
    assert_eq!(
        record.counters[0].get(),
        entry_before + 8,
        "ninth invocation stays on primary after continuation retirement"
    );
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        8,
        "actual continuation guard misses are specialization bailouts too"
    );
    assert_eq!(chunk.optimizing_misses.get(), 0);
    assert!(Rc::ptr_eq(&primary, &chunk.jit.get().flatten().unwrap()));
    assert!(state.counters.iter().all(|(_, counter)| counter.get() == 0));
    assert_eq!(continuation.residency.active.get(), 0);
    assert!(record
        .json()
        .contains("\"entry_kind\":\"loop_continuation\""));

    // Exercise the other continuation miss path: scalar-field validation,
    // where a descriptor change must recover before any getter is observed.
    assert_eq!(eval(&mut engine,"var fieldPrefixes=0,gets=0;var fieldSource={x:1,y:4};function fieldSubject(n,object){fieldPrefixes++;var sum=0;for(var k=0;k<n;k++){sum+=object.x+object.y;object.x=k&1}return sum}fieldSubject(100,fieldSource)"),"450");
    let (_, field_chunk) = parts(&mut engine, "fieldSubject");
    let field_state = field_chunk.optimizing_loop.get().unwrap();
    let field_code = field_state.code.get().flatten().unwrap();
    let field_record = field_code.optimizing_diagnostics.as_ref().unwrap();
    let field_entries = field_record.counters[0].get();
    assert!(field_entries > 0);
    assert_eq!(field_record.counters[diagnostics::VM_RECOVERY].get(), 0);
    assert_eq!(eval(&mut engine,"Object.defineProperty(fieldSource,'y',{get(){gets++;return 4}});var fieldResults=[];for(var r=0;r<9;r++)fieldResults.push(fieldSubject(100,fieldSource));[fieldResults.join(','),gets,fieldPrefixes].join(':')"),"450,450,450,450,450,450,450,450,450:900:10");
    assert_eq!(field_state.misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert_eq!(field_record.counters[0].get(), field_entries + 8);
    assert_eq!(
        field_record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        8
    );
    assert_eq!(field_record.counters[diagnostics::VM_RECOVERY].get(), 8);
    assert_eq!(
        field_record.counters[diagnostics::VM_RECOVERY_NORMAL].get(),
        8
    );
    assert_eq!(
        field_record.counters[diagnostics::RETIREMENT_REQUEST].get(),
        1
    );
    assert!(
        field_record.site_stats().is_empty(),
        "loop scalar operations are explicitly outside site credit"
    );
    assert_eq!(field_chunk.optimizing_misses.get(), 0);
}

#[test]
fn named_method_fast_sites_preserve_unicode_key_identity() {
    if child("named_method_fast_sites_preserve_unicode_key_identity") {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let key = "界".repeat(65);
    eval(&mut engine,&format!("var proto={{{key}(){{return 11;}}}};var source=Object.create(proto);function subject(o){{return o.{key}();}}subject(source);"));
    let (_, baseline) = parts(&mut engine, "subject");
    let pc = baseline
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::GetMethod(..)))
        .unwrap();
    let (_, code) = install(&mut engine, "subject", &[], &[], None);
    assert_eq!(eval(&mut engine, "subject(source)"), "11");
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.site_stats(), vec![(pc, "data_property_read", 1)]);
    let json = record.json();
    assert!(json.contains(&format!("\"key\":\"{}\"", "界".repeat(64))));
    assert!(json.contains("\"key_truncated\":true"));
    assert!(json.contains("GetMethod("));
    assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 1);
}
