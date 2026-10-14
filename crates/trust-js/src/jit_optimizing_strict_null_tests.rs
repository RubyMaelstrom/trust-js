//! Exact-Null comparison semantics, actual native/helper paths and canonical owner recovery.
use super::*;
use crate::{bytecode::Tier, value::Callable, value::Value as JsValue, Completion, Engine};
use std::rc::{Rc, Weak};

struct Recording(bool);
impl Recording {
    fn new() -> Self {
        TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
        Self(TEST_CAPTURE.with(|enabled| enabled.replace(true)))
    }
}
impl Drop for Recording {
    fn drop(&mut self) {
        TEST_CAPTURE.with(|enabled| enabled.set(self.0));
        TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
        TEST_SITES.with(|sites| sites.borrow_mut().clear());
        WITNESS.with(|witness| witness.borrow_mut().take());
        CHECKED_OWNERS.with(|counts| counts.borrow_mut().clear());
    }
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func.clone()
}

fn install(
    engine: &mut Engine,
    name: &str,
    deopt: Option<usize>,
    invalid_owners: bool,
) -> (Rc<Chunk>, Rc<JitCode>, Vec<(usize, bool)>) {
    let function = function(engine, name);
    let chunk = bytecode::compile(&function).unwrap();
    let mut values = crate::value::jit_layout(&engine.interp.object_proto);
    if invalid_owners {
        values.rc_strong_off = usize::MAX;
        assert!(!crate::value::jit_packed_owners_supported(&values));
    }
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    TEST_SITES.with(|sites| sites.borrow_mut().clear());
    let code = Rc::new(compile_with_deopt(&chunk, &values, &layout, deopt).unwrap());
    let sites = TEST_SITES.with(|sites| sites.borrow().clone());
    assert!(code.pc_offsets.is_empty(), "actual optimizing native body");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.code.set(Some(chunk.clone())).is_ok());
    (chunk, code, sites)
}

fn engine(source: &str) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    engine
}

fn child(name: &str, variants: &[(&str, &str)]) -> bool {
    child_with_env(name, variants, &[])
}

fn child_with_env(name: &str, variants: &[(&str, &str)], extra: &[(&str, &str)]) -> bool {
    const CASE: &str = "TRUST_JS_TEST_STRICT_NULL_CASE";
    if std::env::var(CASE).as_deref() == Ok(name) {
        return false;
    }
    for &(key, value) in variants {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("jit::optimizing::strict_null::tests::{name}"),
                "--nocapture",
            ])
            .env(CASE, name)
            .env("TRUST_JS_OPT_JIT", "0")
            .env("TRUST_JS_INLINE_AT", "0")
            .env("TRUST_JS_OPT_JIT_GENERIC_OWN", "0")
            .env("TRUST_JS_OPT_JIT_STRICT_NULL", "1")
            .env("TRUST_JS_OPT_JIT_VALUE_FACTS", "1")
            .env(key, value)
            .envs(extra.iter().copied())
            .env_remove("TRUST_JS_OPT_JIT_DEOPT_AT")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    true
}

const MATRIX: &str = r#"
 var trace=[];var revoked=Proxy.revocable({},{});revoked.revoke();
 var trap={get(){trace.push('get');throw 'trap'},getPrototypeOf(){trace.push('proto');throw 'trap'}};
 var values=[null,undefined,false,true,0,-0,NaN,Infinity,-Infinity,42,3.5,'','text','\ud800','prefix:needle'.slice(7),Symbol('s'),0n,1n,123456789012345678901234567890n,{},[],function(){},(function(){}).bind(null),new Proxy({},trap),revoked.proxy,$262.IsHTMLDDA];
 function eqRight(x){return x===null}function eqLeft(x){return null===x}
 function neRight(x){return x!==null}function neLeft(x){return null!==x}
 var rows=[];for(var i=0;i<values.length;i++)rows.push([eqRight(values[i]),eqLeft(values[i]),neRight(values[i]),neLeft(values[i])].join(','));
 rows.join('|')+'#'+trace.length;
"#;

#[test]
fn exact_null_all_peer_categories_and_orders_bypass_comparison_helpers() {
    if child(
        "exact_null_all_peer_categories_and_orders_bypass_comparison_helpers",
        &[
            ("TRUST_JS_OPT_JIT_SHARED_OWNERS", "0"),
            ("TRUST_JS_OPT_JIT_SHARED_OWNERS", "1"),
        ],
    ) {
        return;
    }
    let _recording = Recording::new();
    let expected = format!(
        "true,true,false,false|{}#0",
        vec!["false,false,true,true"; 25].join("|")
    );
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut oracle = Engine::new();
        oracle.set_tier(tier);
        oracle.set_tier_threshold(0);
        assert_eq!(eval(&mut oracle, MATRIX), expected);
    }
    // Install before the first call. The exact same JS matrix then exercises the
    // optimizing bodies, while other framework/helper functions retain their tier.
    let declarations = MATRIX.split(" var rows=[]").next().unwrap();
    let mut engine = engine(declarations);
    assert_eq!(
        eval(
            &mut engine,
            "values[13].length+':'+values[13].charCodeAt(0)"
        ),
        "1:55296"
    );
    let mut codes = Vec::new();
    for name in ["eqRight", "eqLeft", "neRight", "neLeft"] {
        let (_, code, sites) = install(&mut engine, name, None, false);
        assert_eq!(sites.len(), 1);
        codes.push(code);
    }
    TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            MATRIX
                .split_once(" var rows=[]")
                .map(|(_, tail)| format!("var rows=[]{tail}"))
                .unwrap()
                .as_str()
        ),
        expected
    );
    TEST_CHECKED.with(|checked| {
        assert!(
            checked.borrow().is_empty(),
            "shared/immediate peers never enter checked comparison"
        )
    });
    for code in codes {
        assert_eq!(code.residency.referenced.get(), 1);
        assert_eq!(code.residency.active.get(), 0);
    }
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn disable_controls_and_mixed_nullish_facts_keep_checked_semantics() {
    if child(
        "disable_controls_and_mixed_nullish_facts_keep_checked_semantics",
        &[
            ("TRUST_JS_OPT_JIT_STRICT_NULL", "0"),
            ("TRUST_JS_OPT_JIT_VALUE_FACTS", "0"),
            ("TRUST_JS_OPT_JIT_STRICT_NULL", "1"),
        ],
    ) {
        return;
    }
    let _recording = Recording::new();
    let disabled = std::env::var("TRUST_JS_OPT_JIT_STRICT_NULL").as_deref() == Ok("0")
        || std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() == Ok("0");
    let mut engine=engine("var root={};function subject(x){return null===x}function mixed(flag,x){var n=flag?null:undefined;return n===x}function exact(flag,x){var n=flag?null:null;return n===x}function unknown(a,b){return a===b}");
    let (_, code, sites) = install(&mut engine, "subject", None, false);
    assert_eq!(sites.is_empty(), disabled);
    TEST_CHECKED.with(|c| c.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "[subject(root),subject(null),subject(undefined)].join('|')"
        ),
        "false|true|false"
    );
    TEST_CHECKED.with(|c| assert_eq!(c.borrow().len(), if disabled { 3 } else { 0 }));
    assert_eq!(code.residency.referenced.get(), 1);
    let (_, _, mixed) = install(&mut engine, "mixed", None, false);
    assert!(mixed.is_empty(), "Null|Undefined is not exact Null");
    let (_, _, unknown) = install(&mut engine, "unknown", None, false);
    assert!(unknown.is_empty());
    TEST_CHECKED.with(|c| c.borrow_mut().clear());
    assert_eq!(eval(&mut engine,"[mixed(true,null),mixed(false,undefined),mixed(true,undefined),mixed(false,null),unknown(root,root),unknown(null,null)].join('|')"),"true|true|false|false|true|true");
    TEST_CHECKED.with(|c| assert_eq!(c.borrow().len(), 6));
    let (_, _, exact) = install(&mut engine, "exact", None, false);
    assert_eq!(
        exact.len(),
        usize::from(!disabled),
        "a settled exact-Null phi fact is sufficient without an adjacent constant"
    );
    TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
    assert_eq!(eval(&mut engine,"[exact(true,null),exact(false,null),exact(true,root),exact(false,undefined)].join('|')"),"true|true|false|false");
    TEST_CHECKED.with(|checked| assert_eq!(checked.borrow().len(), if disabled { 4 } else { 0 }));
}

#[test]
fn canonical_number_words_and_boolean_consumers_preserve_all_mirrors() {
    if child(
        "canonical_number_words_and_boolean_consumers_preserve_all_mirrors",
        &[("TRUST_JS_OPT_JIT_SHARED_OWNERS", "0")],
    ) {
        return;
    }
    let _recording = Recording::new();
    let mut engine=engine("function number(n){return ((n|0)^3)===null}function scalar(n){return (n-n)===null}function both(){return null===null}function consumer(x){var b=null===x;return [b,(b|0),b+1,b?7:9].join(':')}function prefix(x){return (13.5 + ((x===null)?1:0))}");
    for name in ["number", "scalar", "both", "consumer", "prefix"] {
        let (_, _, sites) = install(&mut engine, name, None, false);
        assert_eq!(sites.len(), 1);
    }
    TEST_CHECKED.with(|c| c.borrow_mut().clear());
    assert_eq!(eval(&mut engine,"[number(0),number(-0),number(NaN),number(Infinity),number(2147483649),scalar(3),scalar(NaN),both(),consumer(null),consumer({}),prefix(null),prefix({})].join('|')"),"false|false|false|false|false|false|false|true|true:1:2:7|false:0:1:9|14.5|13.5");
    TEST_CHECKED.with(|c| assert!(c.borrow().is_empty()));
    install_observers(&mut engine);
    eval(&mut engine,"var aliasRoot={marker:{value:7}};watch(aliasRoot);function aliases(x){var keep=x;var b=null===x;$262.gc();return [b,keep===x,keep.marker.value].join(':')}");
    let (_, code, sites) = install(&mut engine, "aliases", None, false);
    assert_eq!(sites.len(), 1);
    let owners = WITNESS.with(|witness| witness.borrow().as_ref().unwrap().peer.strong_count());
    TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
    assert_eq!(eval(&mut engine, "aliases(aliasRoot)"), "false:true:7");
    TEST_CHECKED.with(|checked| {
        let checked = checked.borrow();
        assert_eq!(
            checked.len(),
            1,
            "only the separate unknown===unknown identity check remains checked"
        );
        assert_ne!(checked[0].0, sites[0].0);
    });
    assert_eq!(
        WITNESS.with(|witness| witness.borrow().as_ref().unwrap().peer.strong_count()),
        owners,
        "consuming the comparison's duplicate owner does not consume either local alias"
    );
    assert_eq!(code.residency.referenced.get(), 1);
    eval(&mut engine, "aliasRoot=null");
    engine.interp.gc_collect();
    WITNESS.with(|witness| {
        let witness = witness.borrow();
        assert!(witness.as_ref().unwrap().peer.upgrade().is_none());
        assert!(witness.as_ref().unwrap().sentinel.upgrade().is_none());
    });
}

struct OwnerWitness {
    peer: Weak<std::cell::RefCell<crate::value::Object>>,
    sentinel: Weak<std::cell::RefCell<crate::value::Object>>,
    packed: u64,
}

thread_local! {
    static WITNESS: std::cell::RefCell<Option<OwnerWitness>> = const { std::cell::RefCell::new(None) };
    static CHECKED_OWNERS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    static COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn watch(
    _: &mut crate::interpreter::Interp,
    _: JsValue,
    args: &[JsValue],
) -> Result<JsValue, JsValue> {
    let peer = args[0].as_obj().unwrap();
    let marker = peer.borrow().props.get("marker").unwrap().value();
    let packed = PackedValue::pack(args[0].clone()).bits();
    WITNESS.with(|witness| {
        *witness.borrow_mut() = Some(OwnerWitness {
            peer: Rc::downgrade(peer),
            sentinel: Rc::downgrade(marker.as_obj().unwrap()),
            packed,
        });
    });
    Ok(JsValue::Undefined)
}

fn cleared(
    interp: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    interp.gc_collect();
    WITNESS.with(|witness| {
        let witness = witness.borrow();
        assert_eq!(
            witness.as_ref().unwrap().peer.strong_count(),
            1,
            "the evaluated operand or getter local is the only live peer owner at callback GC"
        );
        assert_eq!(witness.as_ref().unwrap().sentinel.strong_count(), 1);
    });
    COLLECTIONS.with(|collections| collections.set(collections.get() + 1));
    Ok(JsValue::Undefined)
}

pub(super) fn observe_checked_owner() {
    WITNESS.with(|witness| {
        if let Some(witness) = witness.borrow().as_ref() {
            CHECKED_OWNERS.with(|counts| counts.borrow_mut().push(witness.peer.strong_count()));
        }
    });
}

fn install_observers(engine: &mut Engine) {
    let global = engine.interp.global.clone();
    engine.interp.def_method(&global, "watch", 1, watch);
    engine.interp.def_method(&global, "cleared", 0, cleared);
}

#[test]
fn last_peer_owner_uses_original_checked_operands_once_and_releases_children() {
    if child(
        "last_peer_owner_uses_original_checked_operands_once_and_releases_children",
        &[
            ("TRUST_JS_OPT_JIT_SHARED_OWNERS", "0"),
            ("TRUST_JS_OPT_JIT_SHARED_OWNERS", "1"),
        ],
    ) {
        return;
    }
    let _recording = Recording::new();
    for (left_null, operator, expected) in [
        (false, "===", "false"),
        (true, "===", "false"),
        (false, "!==", "true"),
        (true, "!==", "true"),
    ] {
        let mut engine = engine("");
        install_observers(&mut engine);
        let source = if left_null {
            format!("var trace=[],saved={{marker:{{}}}};watch(saved);var carrier={{get x(){{trace.push('get');var value=saved;saved=null;cleared();return value}}}};function subject(o){{return null{operator}o.x}}")
        } else {
            format!("var trace=[],carrier={{x:{{marker:{{}}}}}};watch(carrier.x);function clear(o){{trace.push('clear');o.x=null;cleared();return null}}function subject(o){{return o.x{operator}(clear(o),null)}}")
        };
        eval(&mut engine, &source);
        let (chunk, code, sites) = install(&mut engine, "subject", None, false);
        assert_eq!(sites.len(), 1);
        assert!(
            !sites[0].1,
            "the peer remains generic, rather than a field specialization"
        );
        TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
        CHECKED_OWNERS.with(|counts| counts.borrow_mut().clear());
        COLLECTIONS.with(|collections| collections.set(0));
        crate::bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
        assert_eq!(eval(&mut engine, "subject(carrier)"), expected);
        assert_eq!(COLLECTIONS.with(|collections| collections.get()), 1);
        assert_eq!(
            eval(&mut engine, "trace.join(',')"),
            if left_null { "get" } else { "clear" }
        );
        TEST_CHECKED.with(|checked| {
            let checked = checked.borrow();
            assert_eq!(checked.len(), 1);
            assert_eq!((checked[0].0, checked[0].1), (sites[0].0, 2));
            WITNESS.with(|witness| {
                let packed = witness.borrow().as_ref().unwrap().packed;
                assert_eq!(
                    checked[0].2,
                    if left_null {
                        [PACK_NULL, packed]
                    } else {
                        [packed, PACK_NULL]
                    }
                );
            });
        });
        CHECKED_OWNERS.with(|counts| assert_eq!(*counts.borrow(), [1]));
        assert_eq!(chunk.optimizing_misses.get(), 0);
        crate::bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
        assert_eq!(code.residency.referenced.get(), 1);
        assert_eq!(code.residency.active.get(), 0);
        engine.interp.gc_collect();
        WITNESS.with(|witness| {
            let witness = witness.borrow();
            assert!(witness.as_ref().unwrap().peer.upgrade().is_none());
            assert!(
                witness.as_ref().unwrap().sentinel.upgrade().is_none(),
                "the peer's actual destructor released its child"
            );
        });
        WITNESS.with(|witness| witness.borrow_mut().take());
    }
}

#[test]
fn unsupported_owner_layout_keeps_the_checked_path_and_immediate_fast_path() {
    if child(
        "unsupported_owner_layout_keeps_the_checked_path_and_immediate_fast_path",
        &[("TRUST_JS_OPT_JIT_SHARED_OWNERS", "0")],
    ) {
        return;
    }
    let _recording = Recording::new();
    let mut engine = engine("var root={},text='retained';function subject(x){return x===null}");
    let (chunk, code, sites) = install(&mut engine, "subject", None, true);
    assert_eq!(sites.len(), 1);
    TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "[subject(root),subject(text),subject(null),subject(undefined),subject(0)].join('|')"
        ),
        "false|false|true|false|false"
    );
    TEST_CHECKED.with(|checked| {
        let checked = checked.borrow();
        assert_eq!(checked.len(), 2);
        for (index, tag) in [crate::value::PACK_OBJ, crate::value::PACK_STR]
            .into_iter()
            .enumerate()
        {
            assert_eq!((checked[index].0, checked[index].1), (sites[0].0, 2));
            assert_eq!(checked[index].2[0] & 0xffff_0000_0000_0000, tag);
            assert_eq!(checked[index].2[1], PACK_NULL);
        }
    });
    assert_eq!(chunk.optimizing_misses.get(), 0);
    assert_eq!(code.residency.referenced.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
}

const RECOVERY: &str = r#"
 var trace=[],held={value:7},carrier={get x(){trace.push('get');$262.gc();return held}};
 function subject(o){try{return 13.5+((o.x===null)?1:0)}finally{trace.push('finally')}}
 function consumer(o){try{var b=null===o.x;return [b,b|0,b+1,b?7:9].join(':')}finally{trace.push('finally')}}
 function throwing(o){try{return o.x!==null}finally{trace.push('finally')}}
"#;

#[test]
fn exact_comparison_boundaries_recover_canonical_owners_and_boolean_once() {
    if child(
        "exact_comparison_boundaries_recover_canonical_owners_and_boolean_once",
        &[("TRUST_JS_OPT_JIT_SHARED_OWNERS", "0")],
    ) {
        return;
    }
    let _recording = Recording::new();
    for name in ["subject", "consumer"] {
        let mut probe = engine(RECOVERY);
        let probe_chunk = bytecode::compile(&function(&mut probe, name)).unwrap();
        let comparison = probe_chunk
            .jit_ops()
            .iter()
            .position(|op| matches!(op, Op::StrictEq))
            .unwrap();
        for deopt in [None, Some(comparison), Some(comparison + 1)] {
            for is_null in [false, true] {
                let mut engine = engine(RECOVERY);
                let witness = {
                    let env = engine.interp.global_env.clone();
                    let held = engine.interp.get_var("held", &env).ok().unwrap();
                    Rc::downgrade(held.as_obj().unwrap())
                };
                if is_null {
                    eval(&mut engine, "held=null");
                }
                let (chunk, code, sites) = install(&mut engine, name, deopt, false);
                if deopt == Some(comparison) {
                    assert!(sites.is_empty());
                } else {
                    assert_eq!(sites.len(), 1);
                }
                crate::bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
                TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
                let expected = match (name, is_null) {
                    ("subject", false) => "13.5",
                    ("subject", true) => "14.5",
                    ("consumer", false) => "false:0:1:9",
                    ("consumer", true) => "true:1:2:7",
                    _ => unreachable!(),
                };
                assert_eq!(eval(&mut engine, &format!("{name}(carrier)")), expected);
                assert_eq!(eval(&mut engine, "trace.join(',')"), "get,finally");
                TEST_CHECKED.with(|checked| assert!(checked.borrow().is_empty()));
                crate::bytecode::native_deopt::ENTRIES.with(|entries| {
                    let entries = entries.borrow();
                    if let Some(pc) = deopt {
                        assert_eq!(entries.len(), 1);
                        assert_eq!(entries[0].0, pc);
                        assert_eq!(
                            entries[0].1,
                            if name == "subject" {
                                if pc == comparison {
                                    3
                                } else {
                                    2
                                }
                            } else if pc == comparison {
                                2
                            } else {
                                1
                            }
                        );
                    } else {
                        assert!(entries.is_empty());
                    }
                });
                assert_eq!(chunk.optimizing_misses.get(), 0);
                assert_eq!(code.residency.referenced.get(), 1);
                assert_eq!(code.residency.active.get(), 0);
                assert!(engine.interp.fn_frames.is_empty());
                eval(&mut engine, "held=null;carrier=null");
                engine.interp.gc_collect();
                assert!(witness.upgrade().is_none());
            }
        }
    }
    // The property operation throws before equality; its checked observer invokes
    // the getter exactly once and preserves the caller's finalizer and stack frame.
    let mut engine = engine(RECOVERY);
    let (_, code, sites) = install(&mut engine, "throwing", None, false);
    assert_eq!(sites.len(), 1);
    assert_eq!(eval(&mut engine,"var boom={get x(){trace.push('throw');throw new Error('boom')}};try{throwing(boom)}catch(e){e.message+'#'+e.stack.includes('at throwing')+'#'+trace.join(',')}"),"boom#true#throw,finally");
    assert_eq!(code.residency.referenced.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
}

#[test]
fn null_owner_benefit_matches_selected_number_boolean_and_disabled_paths() {
    if child(
        "null_owner_benefit_matches_selected_number_boolean_and_disabled_paths",
        &[
            ("TRUST_JS_OPT_JIT_STRICT_NULL", "1"),
            ("TRUST_JS_OPT_JIT_STRICT_NULL", "0"),
            ("TRUST_JS_OPT_JIT_VALUE_FACTS", "0"),
            ("TRUST_JS_OPT_JIT_SHARED_OWNERS", "1"),
        ],
    ) {
        return;
    }
    let _recording = Recording::new();
    let use_facts = std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() != Ok("0");
    let null_enabled = enabled() && use_facts;
    for (left_null, operator) in [(false, "==="), (true, "==="), (false, "!=="), (true, "!==")] {
        let expression = if left_null {
            format!("null{operator}x")
        } else {
            format!("x{operator}null")
        };
        for (class, argument) in [
            (None, "root"),
            (Some(crate::feedback::ValueClass::NumberDouble), "3.5"),
            (Some(crate::feedback::ValueClass::Boolean), "true"),
        ] {
            let mut engine = engine(&format!(
                "var root={{}};function subject(x){{return {expression}}}"
            ));
            let function = function(&mut engine, "subject");
            let mut chunk = bytecode::compile(&function).unwrap();
            if let Some(class) = class {
                Rc::get_mut(&mut chunk).unwrap().optimizing_inputs =
                    vec![(0, class.bit())].into_boxed_slice();
            }
            let cfg = Cfg::build(&chunk).unwrap();
            let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
            let pc = chunk
                .jit_ops()
                .iter()
                .position(|op| matches!(op, Op::StrictEq | Op::StrictNotEq))
                .unwrap();
            let op = &chunk.jit_ops()[pc];
            let generic = value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
            let guarded =
                class.map(|class| value_facts::Types::guarded_class(class.bit()).unwrap());
            let values = value_facts::ValuePlan::with_inputs(
                &chunk,
                &cfg,
                &stack,
                &guarded.map(|ty| vec![(0, ty)]).unwrap_or_default(),
                &[],
            )
            .unwrap();
            let before_facts = use_facts.then_some(generic.at[pc]);
            let after_facts = use_facts.then_some(values.at[pc]);
            let before = specialization::tests(op, before_facts, stack.at[pc].known());
            let after = specialization::tests(op, after_facts, stack.at[pc].known());
            if null_enabled {
                assert_eq!(
                    before, 1,
                    "the generic peer has one dynamic owner-classification family"
                );
                assert_eq!(
                    after,
                    u32::from(class.is_none()),
                    "only a guarded copyable peer removes that existing family"
                );
                assert_eq!(
                    selected_peer(op, after_facts, 2).unwrap().0,
                    if left_null { 1 } else { 2 }
                );
            } else {
                assert!(selected_peer(op, after_facts, 2).is_none());
                assert_eq!(before, 2);
                assert_eq!(
                    after,
                    if use_facts && class == Some(crate::feedback::ValueClass::NumberDouble) {
                        1
                    } else {
                        2
                    },
                    "disabled Null lowering retains the existing Number-check model"
                );
            }
            assert!(
                selected_peer(op, None, 2).is_none(),
                "absent facts are never manufactured"
            );
            assert_eq!(specialization::tests(op, None, 2), 2);
            for known in [0, 1] {
                assert!(selected_peer(op, after_facts, known).is_none());
                assert_eq!(specialization::tests(op, before_facts, known), 0);
                assert_eq!(specialization::tests(op,after_facts,known),0,
                    "an opaque suffix emits the unchanged checked comparison, earning no inline-guard benefit");
            }
            let plan = use_facts
                .then(|| {
                    specialization::Plan::build(&chunk, &cfg, &stack, &chunk.optimizing_inputs, &[])
                })
                .flatten();
            if class.is_some() && use_facts {
                let plan = plan.as_ref().unwrap();
                let credited = plan
                    .opportunities
                    .iter()
                    .find(|op| op.pc == pc)
                    .map_or(0, |op| op.removed_tests);
                assert_eq!(credited, before.saturating_sub(after));
                assert_eq!(
                    plan.saved_tests,
                    plan.opportunities
                        .iter()
                        .map(|op| op.removed_tests)
                        .sum::<u32>()
                        - plan.inputs.len() as u32,
                    "entry guard tax stays unchanged; no new Null-comparison bonus exists"
                );
            } else {
                assert!(plan.is_none());
            }
            TEST_SITES.with(|sites| sites.borrow_mut().clear());
            ownership::TEST_OWNER_IR.with(|counts| counts.set([0; 9]));
            let layout = crate::interpreter::interp_layout(&mut engine.interp);
            let code = Rc::new(
                compile_with_deopt(
                    &chunk,
                    &crate::value::jit_layout(&engine.interp.object_proto),
                    &layout,
                    None,
                )
                .unwrap(),
            );
            let sites = TEST_SITES.with(|sites| sites.borrow().clone());
            if null_enabled {
                assert_eq!(sites, [(pc, class.is_some())]);
                let release_calls = ownership::TEST_OWNER_IR.with(|counts| counts.get()[3]);
                assert_eq!(
                    release_calls == 0,
                    class.is_some(),
                    "guarded copyable peers actually emit no comparison or formal release guards"
                );
            } else {
                assert!(sites.is_empty());
            }
            chunk
                .jit
                .begin_compile()
                .unwrap()
                .commit(Some(code.clone()));
            assert!(function.code.set(Some(chunk.clone())).is_ok());
            TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
            assert_eq!(
                eval(&mut engine, &format!("subject({argument})")),
                if operator == "===" { "false" } else { "true" }
            );
            TEST_CHECKED
                .with(|checked| assert_eq!(checked.borrow().len(), usize::from(!null_enabled)));
            assert_eq!(code.residency.referenced.get(), 1);
            assert_eq!(chunk.optimizing_misses.get(), 0);
            crate::bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
            assert_eq!(
                eval(&mut engine, "subject(null)"),
                if operator == "===" { "true" } else { "false" }
            );
            if class.is_some() && use_facts {
                assert_eq!(chunk.optimizing_misses.get(), 1);
                crate::bytecode::native_deopt::ENTRIES
                    .with(|entries| assert_eq!(entries.borrow()[0].0, 0));
            } else {
                assert_eq!(chunk.optimizing_misses.get(), 0);
                crate::bytecode::native_deopt::ENTRIES
                    .with(|entries| assert!(entries.borrow().is_empty()));
            }
        }
    }
}

#[test]
fn continuation_null_facts_remain_available_with_whole_function_value_facts_disabled() {
    if child_with_env(
        "continuation_null_facts_remain_available_with_whole_function_value_facts_disabled",
        &[
            ("TRUST_JS_OPT_JIT_VALUE_FACTS", "0"),
            ("TRUST_JS_OPT_JIT_VALUE_FACTS", "1"),
        ],
        &[
            ("TRUST_JS_OPT_JIT_OSR", "1"),
            ("TRUST_JS_OPT_JIT_OSR_AT", "8"),
        ],
    ) {
        return;
    }
    let _recording = Recording::new();
    let mut engine=engine("var root={};function subject(n,x){var sum=0;for(var k=0;k<n;k++){sum+=k;if(x===null)sum+=1;}return sum}");
    TEST_SITES.with(|sites| sites.borrow_mut().clear());
    TEST_CHECKED.with(|checked| checked.borrow_mut().clear());
    assert_eq!(eval(&mut engine, "subject(100,root)"), "4950");
    let function = function(&mut engine, "subject");
    let chunk = function.code.get().and_then(Option::as_ref).unwrap();
    let state = chunk.optimizing_loop.get().unwrap();
    assert!(
        state.entries.get() > 0,
        "the existing live-frame path actually entered its continuation"
    );
    let header = state.selected.get().unwrap();
    let code = state.code.get().flatten().unwrap();
    assert_eq!(code.entry_kind, NativeEntryKind::OptimizingContinuation);
    let sites = TEST_SITES.with(|sites| sites.borrow().clone());
    assert_eq!(
        sites.len(),
        1,
        "continuation lowering receives its own settled exact-Null fact"
    );
    assert!(
        !sites[0].1,
        "the object formal supplies no guarded scalar class"
    );
    TEST_CHECKED.with(|checked| assert!(checked.borrow().is_empty()));
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let facts = value_facts::ValuePlan::at_entry(chunk, &cfg, &stack, header, &[]).unwrap();
    assert_eq!(
        specialization::tests(
            &chunk.jit_ops()[sites[0].0],
            Some(facts.at[sites[0].0]),
            stack.at[sites[0].0].known()
        ),
        1
    );
    assert_eq!(eval(&mut engine, "subject(100,null)"), "5050");
    assert_eq!(state.misses.get(), 0);
    assert_eq!(chunk.optimizing_misses.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}
