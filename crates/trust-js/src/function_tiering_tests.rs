//! ECMA-262 FunctionDeclarationInstantiation / ForBodyEvaluation, official local snapshot
//! e28783d5fc9dc12b3de905961e2c71410b38a202. A tier transition continues the current call.
use crate::{bytecode::Tier, Completion, Engine};

fn value(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("tiering fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine, name: &str) -> std::rc::Rc<crate::ast::Function> {
    let environment = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &environment).ok().unwrap();
    let object = value.as_obj().unwrap();
    let crate::value::Callable::User(user) = &object.borrow().call else {
        panic!("user function")
    };
    user.func.clone()
}

fn reset_counts() {
    crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.set(0));
    crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.set(0));
    crate::bytecode::loop_fragment::TEST_PRIVATE_FRAGMENT_ENTRIES.with(|count| count.set(0));
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
}

fn check(source: &str, expected: &str, whole_frame: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        reset_counts();
        assert_eq!(value(&mut engine, source), expected, "{tier:?}");
        let entries = crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get());
        if matches!(tier, Tier::Interp) || !whole_frame {
            assert_eq!(
                entries, 0,
                "ineligible or interpreter frame must remain on its established path"
            );
        } else {
            assert!(
                entries > 0,
                "must enter the live whole-function continuation"
            );
            #[cfg(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(target_os = "linux", target_os = "macos", target_os = "windows")
            ))]
            if matches!(tier, Tier::Jit) {
                assert!(
                    crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) > 0,
                    "hot current invocation must enter native code"
                );
            }
        }
    }
}

#[test]
fn cold_loop_functions_wait_for_executed_hotness_or_call_threshold() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        value(&mut engine, "function cold(n){let sum=0;for(let k=0;k<n;k++)sum+=k;return sum;}function unused(){while(false){}};");
        let cold = function(&mut engine, "cold");
        let unused = function(&mut engine, "unused");
        for _ in 0..32 {
            assert_eq!(value(&mut engine, "cold(100)"), "4950");
        }
        assert!(
            cold.code.get().is_none(),
            "short loop calls must not trigger compilation"
        );
        assert!(unused.code.get().is_none());
        assert_eq!(value(&mut engine, "cold(100)"), "4950");
        assert!(
            cold.code.get().is_some_and(Option::is_some),
            "call threshold still admits hot functions"
        );
    }
}

#[test]
fn ordinary_and_base_constructor_frames_preserve_this_new_target_and_fields() {
    check("var prefixes=0;function F(n){prefixes++;for(let k=0;k<3000;k++)n++;this.value=n;this.target=new.target;return 7;}var f=new F(2);[f.value,f.target===F,prefixes].join(':');", "3002:true:1", true);
    check("var fields=0;class F{value=(fields++,7);constructor(n){for(let k=0;k<3000;k++)n++;this.value+=n;}}[new F(2).value,fields].join(':');", "3009:1", true);
    check(
        "function F(n){for(let k=0;k<3000;k++)n++;return {value:n};}new F(2).value;",
        "3002",
        true,
    );
    check("var receiver={value:7};function outer(){var f=n=>{for(let k=0;k<3000;k++)n++;return n+':'+this.value;};return f(2);}outer.call(receiver);", "3002:7", true);
}

#[test]
fn live_frame_skips_defaults_destructuring_and_prefix_effects() {
    check("var events=0;function f(n=(events++,3000),[x]=[7]){events++;let sum=x;for(let k=0;k<n;k++){sum+=k;}events++;return sum+':'+events;}f();", "4498507:3", true);
    check("var calls=0;function f(){let sum=0;let k=0;while(k<3000){sum+=k++;}calls++;return sum+':'+calls;}f();", "4498500:1", true);
    check("var tests=0;function f(){let sum=0;let k=0;do{sum+=k++;}while((tests++,k<3000));return sum+':'+tests;}f();", "4498500:3000", true);
}

#[test]
fn live_frame_preserves_tdz_shadowing_this_new_target_and_caller() {
    check("function caller(){return f.call({x:7});}function f(){let sum=0;for(let k=0;k<3000;k++)sum+=k;return [sum,this.x,new.target===undefined,f.caller===caller].join(':');}caller();", "4498500:7:true:true", true);
    check("function f(){let k=10;let sum=0;for(let k=0;k<3000;k++)sum+=k;try{later;}catch(e){return sum+':'+k+':'+e.name;}let later=1;}f();", "4498500:10:ReferenceError", true);
    check(
        "function f(){let sum=0;for(let k=0;k<3000;k++)sum+=k;return sum;throw 1;}f();",
        "4498500",
        true,
    );
}

#[test]
fn migrated_loop_routes_its_future_finalizers_and_completions_once() {
    check("var cleanup=0;function f(){let sum=0;outer:for(let k=0;k<3000;k++){try{if(k%2===0)continue outer;if(k===2701)break outer;sum+=k;}finally{cleanup++;}}return sum+':'+cleanup;}f();", "1822500:2702", true);
    check("var cleanup=0;function f(){let k=0;while(k++<3000){try{if(k===2200)return k;}finally{cleanup++;}}return 0;}f()+':'+cleanup;", "2200:2200", true);
}

#[test]
fn prefix_created_closures_keep_the_live_function_activation() {
    check("function f(n){let sum=0;const read=()=>sum;for(let k=0;k<n;k++){sum+=k;if(k===1500&&read()!==1125750)throw 1;}return read();}f(3000);", "4498500", true);
    check("function f(n){const read=()=>n;let sum=0;for(let k=0;k<3000;k++){n++;sum+=read();}return sum+':'+n;}f(3);", "4510500:3003", true);
}

#[test]
fn legacy_arguments_aliases_live_parameters_before_and_after_transfer() {
    check(
        "function f(n){for(var k=0;k<3000;k++)n++;return n+':'+f.arguments[0];}f(0);",
        "3000:3000",
        true,
    );
    check("function f(n){var args=f.arguments;for(var k=0;k<3000;k++){n++;if(k===1500)args[0]=7;}return n+':'+args[0];}f(0);", "1506:1506", true);
    check("function observe(){f.arguments[0]=7;}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500)observe();}return n+':'+f.arguments[0];}f(0);", "1506:1506", true);
    check("function read(){return f.arguments[0];}function f(n){for(var k=0;k<3000;k++)n++;return n+':'+read();}f(0);", "3000:3000", true);
    check("function f(n){function n(){}n=0;for(var k=0;k<3000;k++)n++;return n+':'+f.arguments[0];}f(7);", "3000:3000", true);
    check(
        "function f(n){for(let n=0;n<3000;n++){}return n+':'+f.arguments[0];}f(7);",
        "7:7",
        true,
    );
}

#[test]
fn mapped_eval_and_iteration_bindings_reuse_their_original_environments() {
    check("function f(n){let sum=0;for(let k=0;k<3000;k++){if(k===1500)arguments[0]=7;sum+=n;}return sum+':'+n;}f(3);", "15000:7", true);
    check("function f(){let sum=0;for(let k=0;k<3000;k++){sum+=k;if(k===1500)eval('sum+=7');}return sum;}f();", "4498507", true);
    check("function f(){let functions=[];for(let k=0;k<3000;k++)if(k===1500||k===2500)functions.push(()=>k);return functions.map(f=>f()).join(':');}f();", "1500:2500", true);
}

#[test]
fn pending_outer_author_cleanup_uses_loop_fragments() {
    check("var cleanup=0;function f(){let sum=0;try{for(let k=0;k<3000;k++)sum+=k;return sum;}finally{cleanup++;}}f()+':'+cleanup;", "4498500:1", false);
}

#[test]
fn strict_private_fragments_publish_before_outer_author_cleanup() {
    check("function f(n){'use strict';let sum=0;try{for(let k=0;k<n;k++){sum+=k;if(k===2200)throw 7;}}catch(e){return sum+':'+e;}finally{if(sum!==2421100)throw 'stale';}}f(3000);", "2421100:7", false);
    check("function f(){'use strict';let x=1;try{const x=2;for(let k=0;k<3000;k++){if(k===1500)x=3;}}catch(e){return e.name+':'+x;}}f();", "TypeError:1", false);
    check("function f(){'use strict';const x=1;let result;try{let x=2;for(let k=0;k<3000;k++){x++;}result=x;}finally{if(x!==1)throw 'outer const';}return result;}f();", "3002", false);
    check("function f(){'use strict';let sum=0;let alias=()=>sum;try{for(let k=0;k<3000;k++){sum+=k;if(k===1500&&alias()!==1125750)throw 7;}return alias();}finally{if(alias()!==4498500)throw 'capture';}}f();", "4498500", false);
    check("function f(){'use strict';return (0,eval)(\"'use strict';var x=0;var read=()=>x;try{for(let k=0;k<3000;k++){x++;if(k===1500&&read()!==1501)throw Error('stale');}}finally{};read()\");}f();", "3000", false);
    check("function f(){'use strict';try{for(let k=0;k<3000;k++){if(k===1500)later=3;}}catch(e){return e.name;}let later;return 'allowed';}f();", "ReferenceError", false);
}

#[test]
fn sloppy_private_fragments_keep_parameter_aliases_and_outer_cleanup_live() {
    check("function f(n){let args=arguments;try{for(let k=0;k<3000;k++){n++;if(k===1500)args[0]=7;}}finally{if(n!==1506)throw 'stale';}return n+':'+args[0];}f(0);", "1506:1506", false);
    check("function observe(){f.arguments[0]=7;}function f(n){try{for(let k=0;k<3000;k++){n++;if(k===1500)observe();}}finally{if(n!==1506)throw 'stale';}return n+':'+f.arguments[0];}f(0);", "1506:1506", false);
    check("function f(n){try{for(let n=0;n<3000;n++){};}finally{if(n!==7)throw 'shadow';}return n+':'+f.arguments[0];}f(7);", "7:7", false);
    check("function f(n){try{for(let k=0;k<3000;k++){n++;if(k===2200)throw 7;}}catch(e){return n+':'+e;}finally{if(n!==2201)throw 'stale';}}f(0);", "2201:7", false);
    assert!(
        crate::bytecode::loop_fragment::TEST_PRIVATE_FRAGMENT_ENTRIES.with(|count| count.get()) > 0,
        "sloppy outer-cleanup loop must actually enter its projected private fragment"
    );
}

#[test]
fn migrated_argument_maps_keep_duplicate_deleted_and_redefined_bindings() {
    // Node 24.21.0 verified these results. The ordinary arguments object is kept alive
    // across entry; mapping changes must keep updating the exact live parameter cells.
    check("function f(n,n){var args=arguments;for(var k=0;k<3000;k++){n++;if(k===1500)args[0]=5;if(k===1600)args[1]=7;}return n+':'+args[0]+':'+args[1];}f(8,9);", "1406:5:1406", true);
    check("function f(n){var args=arguments;for(var k=0;k<3000;k++){n++;if(k===1500)delete args[0];}return n+':'+args[0];}f(2);", "3002:undefined", true);
    check("function f(n){var args=arguments;for(var k=0;k<3000;k++){n++;if(k===1500)Object.defineProperty(args,'0',{get:function(){return 11;},configurable:true});}return n+':'+args[0];}f(2);", "3002:11", true);
    check("function f(arguments){for(var k=0;k<3000;k++)arguments++;return arguments+':'+f.arguments[0];}f(2);", "3002:3002", true);
}

#[test]
fn an_indirect_eval_loop_keeps_its_own_captured_environment() {
    // The active strict caller does not own the eval's loop or its captured local. Its
    // capture scan therefore cannot authorize private slots for the eval environment.
    check(
        r#"function f(){'use strict';return (0,eval)("'use strict';var x=0;var read=()=>x;try{for(let k=0;k<3000;k++){x++;if(k===1500&&read()!==1501)throw Error('stale');}}finally{};read()");}f();"#,
        "3000",
        false,
    );
}

#[test]
fn borrowed_parameters_publish_and_reload_across_implicit_observers() {
    // Legacy fn.arguments writes are an engine extension. These expectations come from
    // the interpreter; a proxy observer checks publication before replacing a Number
    // with an object, forcing the native region's post-effect category guard to miss.
    check("var proxy=new Proxy({}, {has(){var before=f.arguments[0];f.arguments[0]={tag:before};return true;}});function f(n){for(var k=0;k<3000;k++){n++;if(k===1500){var had='x' in proxy;if(!had||n.tag!==1501)throw 'publication';n=7;}}return n+':'+f.arguments[0];}f(0);", "1506:1506", true);
    check("var token={valueOf(){f.arguments[1]=7;return 2;}};function f(n,m){for(var k=0;k<3000;k++){m++;if(k===1500)n++;}return n+':'+m+':'+f.arguments[1];}f(token,0);", "3:1506:1506", true);
    check("var receiver={get value(){f.arguments[0]={tag:7};return 0;}};function f(n){for(var k=0;k<3000;k++){n++;if(k===1500){void receiver.value;if(n.tag!==7)throw 'category';n=n.tag;}}return n+':'+f.arguments[0];}f(0);", "1506:1506", true);
    check("var key={toString(){f.arguments[0]=7;return 'x';}},receiver={x:1};function f(n){for(var k=0;k<3000;k++){n++;if(k===1500)receiver[key];}return n+':'+f.arguments[0];}f(0);", "1506:1506", true);
    // The ordinary arguments path is specified by ECMA-262 and also verified in Node.
    check("function f(n){var args=arguments,proxy=new Proxy({}, {has(){if(args[0]!==1501)throw 'publication';args[0]=7;return true;}});for(var k=0;k<3000;k++){n++;if(k===1500)'x' in proxy;}return n+':'+args[0];}f(0);", "1506:1506", true);
}

#[test]
fn borrowed_parameter_maps_keep_binding_identity_and_mapping_changes() {
    check("function change(){var args=f.arguments;args[0]=5;args[1]=7;}function f(n,n){for(var k=0;k<3000;k++){n++;if(k===1500)change();}return n+':'+f.arguments[0]+':'+f.arguments[1];}f(8,9);", "1506:5:1506", true);
    check("var saved;function change(){saved=f.arguments;delete saved[0];}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500)change();}return n+':'+saved[0];}f(0);", "3000:undefined", true);
    check("var saved;function change(){saved=f.arguments;Object.defineProperty(saved,'0',{value:7,writable:false});}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500)change();}return n+':'+saved[0];}f(0);", "1506:7", true);
    check("function change(){f.arguments[0]=20;}function f(n){for(let n=0;n<3000;n++){if(n===1500)change();}return n+':'+f.arguments[0];}f(7);", "20:20", true);
}

#[test]
fn borrowed_parameters_survive_returns_throws_and_retained_projection_fallback() {
    check("var saved;function change(){saved=f.arguments;saved[0]=7;}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500)change();if(k===2200)return n;}return n;}var result=f(0),before=saved[0];saved[0]=11;result+':'+before+':'+saved[0]+':'+f.arguments;", "707:707:11:null", true);
    check("var saved;function change(){saved=f.arguments;saved[0]={tag:7};throw 'finish';}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500)change();}}var result;try{f(0);}catch(e){var tag=saved[0].tag;saved[0]={tag:11};result=e+':'+tag+':'+saved[0].tag+':'+f.arguments;}result;", "finish:7:11:null", true);
    // A retained class projects n into a temporary Environment Record. Copying that
    // unchanged record back after change() would overwrite the mapped-parameter write.
    // Whole-frame migration must decline this plan and use the retained AST environment.
    check("function change(){f.arguments[0]=7;return 0;}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500){var C=class{[n+(change(),0)](){}};}}return n+':'+f.arguments[0];}f(0);", "1506:1506", false);
    check("var source={get x(){f.arguments[0]=7;return 1;}};function f(n){var z=0;for(var k=0;k<3000;k++){n++;if(k===1500)({x:z}=source);}return n+':'+z+':'+f.arguments[0];}f(0);", "1506:1:1506", true);
}

fn install_native(engine: &mut Engine, name: &str, callback: crate::value::NativeFn) {
    let native = engine.interp.make_native(name, 0, callback);
    engine.interp.global.borrow_mut().props.insert(
        name,
        crate::value::Property::data(crate::value::Value::Obj(native), true, true, true),
    );
}

#[test]
fn recursive_borrowed_parameter_scopes_keep_both_live_activations() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Interp);
        engine.set_tier_threshold(32);
        install_native(
            &mut engine,
            "enableTier",
            match tier {
                Tier::Interp => |i, _, _| {
                    i.tier = Tier::Interp;
                    Ok(crate::value::Value::Undefined)
                },
                Tier::Bytecode => |i, _, _| {
                    i.tier = Tier::Bytecode;
                    Ok(crate::value::Value::Undefined)
                },
                Tier::Jit => |i, _, _| {
                    i.tier = Tier::Jit;
                    Ok(crate::value::Value::Undefined)
                },
            },
        );
        install_native(&mut engine, "enterInterpreter", |i, _, _| {
            i.tier = Tier::Interp;
            Ok(crate::value::Value::Undefined)
        });
        reset_counts();
        assert_eq!(value(&mut engine, "var saved;function f(n,depth){enableTier();var own=f.arguments;if(depth===0)saved=own;for(var k=0;k<3000;k++){n++;if(k===1500){if(depth===0){enterInterpreter();var nested=f(10,1);if(nested!==1506||n!==15)throw 'recursive alias';own[0]=20;}else{saved[0]=15;own[0]=7;}}}return n;}f(0,0)+':'+saved[0];"), "1519:1519", "{tier:?}");
        let entries = crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get());
        if tier == Tier::Interp {
            assert_eq!(entries, 0);
        } else {
            assert!(entries >= 2, "both live activations must migrate: {tier:?}");
            #[cfg(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(target_os = "linux", target_os = "macos", target_os = "windows")
            ))]
            if tier == Tier::Jit {
                assert!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) >= 2);
            }
        }
    }
}

#[test]
fn borrowed_parameter_object_owners_survive_collection_and_host_interruption() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        install_native(&mut engine, "collectNow", |i, _, _| {
            i.collect_garbage_for_host();
            Ok(crate::value::Value::Undefined)
        });
        reset_counts();
        assert_eq!(value(&mut engine, "function change(){var args=f.arguments,node={tag:7};node.self=node;args[0]=node;delete args[0];}function f(n){for(var k=0;k<3000;k++){n++;if(k===1500){change();collectNow();if(n.self!==n)throw 'lost root';n=n.tag;}}return n;}f(0);"), "1506", "{tier:?}");
        if tier != Tier::Interp {
            assert!(crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()) > 0);
        }

        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        install_native(&mut engine, "stopNow", |i, _, _| {
            i.runtime_interrupt.request_user_navigation();
            Ok(crate::value::Value::Undefined)
        });
        reset_counts();
        let outcome = engine.eval_interruptible("var saved,caught=0,finalized=0;function f(n){for(var k=0;k<3000;k++){n++;if(k===1500){saved=f.arguments;saved[0]={tag:7};stopNow();}}}try{f(0);}catch(e){caught++;}finally{finalized++;}", false).expect("interruption fixture parses");
        assert!(
            matches!(
                outcome,
                crate::ExecutionOutcome::Interrupted {
                    reason: crate::InterruptReason::UserNavigation
                }
            ),
            "host interruption must remain outside JavaScript completions: {tier:?}"
        );
        if tier != Tier::Interp {
            assert!(crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()) > 0);
        }
        engine.interrupt_handle().begin_user_interaction();
        assert_eq!(value(&mut engine, "var before=saved[0].tag;saved[0]={tag:11};function reuse(n){for(var k=0;k<3000;k++)n++;return n;}reuse(0);before+':'+saved[0].tag+':'+caught+':'+finalized+':'+f.arguments;"), "7:11:0:0:null", "{tier:?}");
        engine.interp.collect_garbage_for_host();
        assert_eq!(value(&mut engine, "saved[0].tag"), "11", "{tier:?}");
    }
}

fn check_fragment(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        reset_counts();
        assert_eq!(value(&mut engine, source), expected, "{tier:?}");
        assert_eq!(
            crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()),
            0
        );
        let fragments =
            crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get());
        if tier == Tier::Interp {
            assert_eq!(fragments, 0);
        } else {
            assert!(fragments > 0, "must enter the loop fragment: {tier:?}");
            #[cfg(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(target_os = "linux", target_os = "macos", target_os = "windows")
            ))]
            if tier == Tier::Jit {
                assert!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) > 0);
            }
        }
    }
}

#[test]
fn sloppy_fragments_publish_parameter_objects_before_parked_handlers() {
    check_fragment("var saved;var receiver={get value(){var before=Object.getOwnPropertyDescriptor(saved,'0').value;saved[0]={tag:before};return 0;}};function f(n){saved=arguments;try{for(var k=0;k<3000;k++){n++;if(k===1500){void receiver.value;if(n.tag!==1501)throw 'publication';n=n.tag;}}}finally{if(n!==3000||saved[0]!==3000)throw 'outer finally';}return n+':'+saved[0];}f(0);", "3000:3000");
    check_fragment("var saved;function change(){saved=f.arguments;var before=saved[0];saved[0]={tag:before};throw 9;}function f(n){try{for(var k=0;k<3000;k++){n++;if(k===1500)change();}}catch(e){return n.tag+':'+e;}finally{if(n.tag!==1501)throw 'outer finally';saved[0]={tag:12};if(n.tag!==12)throw 'retired alias';}}var returned=f(0);returned+':'+saved[0].tag;", "1501:9:12");
    // A retained evaluator must keep the real environment canonical when its temporary
    // projection could otherwise overwrite a legacy arguments-map write.
    check_fragment("function change(){f.arguments[0]=7;return 0;}function f(n){try{for(var k=0;k<3000;k++){n++;if(k===1500){var C=class{[n+(change(),0)](){}};}}}finally{if(n!==1506)throw 'projection';}return n+':'+f.arguments[0];}f(0);", "1506:1506");
}

#[test]
fn sloppy_fragments_keep_shadowed_and_captured_lexical_homes_distinct() {
    check_fragment("function change(){f.arguments[0]=7;}function f(n){var inside;try{let n=2;for(let k=0;k<3000;k++){n++;if(k===1500)change();}inside=n;}finally{if(n!==7)throw 'outer parameter';}return inside+':'+n+':'+f.arguments[0];}f(0);", "3002:7:7");
    check_fragment("function change(){f.arguments[0]=20;}function f(n){try{for(let n=0;n<3000;n++){if(n===1500)change();}}finally{if(n!==20)throw 'head shadow';}return n+':'+f.arguments[0];}f(7);", "20:20");
    check_fragment("function f(n){var args=arguments,inside;try{let n=2;const read=()=>n;for(let k=0;k<3000;k++){n++;if(k===1500){args[0]=7;if(read()!==1503)throw 'captured shadow';}}inside=read();}finally{if(n!==7)throw 'outer parameter';}return inside+':'+n+':'+args[0];}f(0);", "3002:7:7");
}

#[test]
fn sloppy_fragment_aliases_follow_nested_labels_and_finally_completions() {
    check_fragment("var outerCleanup=0;function f(n){var args=arguments,cleaned=0;try{outer:for(let k=0;k<3000;k++){n++;try{if(k===1500)args[0]=7;if(k%2===0)continue outer;if(k===2201)break outer;}finally{cleaned++;}}}finally{outerCleanup++;if(n!==708||args[0]!==708)throw 'parked finally';}return n+':'+args[0]+':'+cleaned+':'+outerCleanup;}f(0);", "708:708:2202:1");
    check_fragment("var saved;function f(n){saved=arguments;try{for(let k=0;k<3000;k++){n++;if(k===2200)return n;}}finally{saved[0]++;if(n!==2202)throw 'finally write';}}var result=f(0);result+':'+saved[0];", "2201:2202");
    check_fragment("function f(n){var args=arguments,cleaned=0;outer:for(var round=0;round<2;round++){try{for(var k=0;k<3000;k++){n++;if(k===1500)args[0]=7;if(k===2200)continue outer;}}finally{cleaned++;if(n!==707||args[0]!==707)throw 'external continue';}}return n+':'+args[0]+':'+cleaned;}f(0);", "707:707:2");
    check_fragment("function f(n){var args=arguments,cleaned=0;block:{try{for(var k=0;k<3000;k++){n++;if(k===2200)break block;}}finally{cleaned++;args[0]++;if(n!==2202)throw 'external break';}}return n+':'+args[0]+':'+cleaned;}f(0);", "2202:2202:1");
}

#[test]
fn constructor_migration_keeps_reflect_new_target_and_return_rules() {
    check("var saved;function change(){saved=F.arguments;saved[0]=7;}function F(n){for(let k=0;k<3000;k++){n++;if(k===1500)change();}this.value=n;this.target=new.target;return 17;}function Other(){}var result=Reflect.construct(F,[0],Other);[result.value,result.target===Other,Object.getPrototypeOf(result)===Other.prototype,saved[0]].join(':');", "1506:true:true:1506", true);
    check("var fields=0,receiver;var observer={get value(){receiver.marker=9;return 0;}};class Base{value=(fields++,7);constructor(n){receiver=this;for(let k=0;k<3000;k++){n++;if(k===1500)void observer.value;}this.value+=n;this.target=new.target;return 17;}}function Other(){}var result=Reflect.construct(Base,[0],Other);[result.value,result.marker,result.target===Other,Object.getPrototypeOf(result)===Other.prototype,fields].join(':');", "3007:9:true:true:1", true);
    check("function F(n){for(let k=0;k<3000;k++){n++;if(k===2200)return {value:n,target:new.target};}throw 'unreachable';}function Other(){}var result=Reflect.construct(F,[0],Other);[result.value,result.target===Other,Object.getPrototypeOf(result)===Object.prototype].join(':');", "2201:true:true", true);
    check_fragment("var saved;function F(n){saved=F.arguments;try{for(let k=0;k<3000;k++){n++;if(k===2200){this.value=n;return 17;}}}finally{this.cleaned=n===2201;saved[0]=8;this.after=n;}}var result=new F(0);[result.value,result.cleaned,result.after,saved[0]].join(':');", "2201:true:8:8");
}

#[test]
fn derived_constructor_loops_preserve_their_established_completion_path() {
    check_fragment("class Base{}class Derived extends Base{constructor(n){for(let k=0;k<3000;k++)n++;return {value:n,target:new.target};}}var result=new Derived(0);[result.value,result.target===Derived].join(':');", "3000:true");
    check_fragment("class Base{}class Derived extends Base{constructor(n){for(let k=0;k<3000;k++)n++;return 7;}}var result;try{new Derived(0);}catch(e){result=e.name;}result;", "TypeError");
    check_fragment("var bases=0,fields=0;class Base{constructor(){bases++;}}class Derived extends Base{value=(fields++,7);constructor(n){super();for(let k=0;k<3000;k++)n++;this.value+=n;}}var result=new Derived(0);[result.value,bases,fields].join(':');", "3007:1:1");
}

#[test]
fn sloppy_fragment_aliases_retire_on_collection_and_host_interruption() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        install_native(&mut engine, "collectNow", |i, _, _| {
            i.collect_garbage_for_host();
            Ok(crate::value::Value::Undefined)
        });
        reset_counts();
        assert_eq!(value(&mut engine, "function change(){var args=f.arguments,node={tag:7};node.self=node;args[0]=node;delete args[0];}function f(n){try{for(var k=0;k<3000;k++){n++;if(k===1500){change();collectNow();if(n.self!==n)throw 'lost fragment root';n=n.tag;}}}finally{if(n!==1506)throw 'parked finally';}return n;}f(0);"), "1506", "{tier:?}");
        assert_eq!(
            crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()),
            0
        );
        if tier != Tier::Interp {
            assert!(
                crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get()) > 0
            );
        }

        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        install_native(&mut engine, "stopNow", |i, _, _| {
            i.runtime_interrupt.request_user_navigation();
            Ok(crate::value::Value::Undefined)
        });
        reset_counts();
        let outcome = engine.eval_interruptible("var saved,caught=0,finalized=0;function f(n){try{for(var k=0;k<3000;k++){n++;if(k===1500){saved=f.arguments;saved[0]={tag:7};stopNow();}}}catch(e){caught++;}finally{finalized++;}}f(0);", false).expect("fragment interruption fixture parses");
        assert!(
            matches!(
                outcome,
                crate::ExecutionOutcome::Interrupted {
                    reason: crate::InterruptReason::UserNavigation
                }
            ),
            "{tier:?}"
        );
        assert_eq!(
            crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()),
            0
        );
        if tier != Tier::Interp {
            assert!(
                crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get()) > 0
            );
        }
        engine.interrupt_handle().begin_user_interaction();
        assert_eq!(value(&mut engine, "var before=saved[0].tag;saved[0]={tag:11};function reuse(n){for(var k=0;k<3000;k++)n++;return n;}reuse(0);before+':'+saved[0].tag+':'+caught+':'+finalized+':'+f.arguments;"), "7:11:0:0:null", "{tier:?}");
        engine.interp.collect_garbage_for_host();
        assert_eq!(value(&mut engine, "saved[0].tag"), "11", "{tier:?}");
    }
}

type WatchedBufferOwner = (
    std::rc::Weak<std::cell::RefCell<crate::value::Object>>,
    usize,
);
thread_local! {
    static WATCHED_BUFFER_OWNER: std::cell::RefCell<Option<WatchedBufferOwner>> = const {
        std::cell::RefCell::new(None)
    };
}

#[test]
fn migrated_private_homes_release_overwritten_objects_before_returning() {
    // Check actual owner and backing-store retirement during the continuation, rather
    // than relying on allocator RSS. Captured homes remain environment-backed, and both
    // non-formal locals and parameters replaced in the prefix must transfer ownership.
    let fixtures = [
        ("function f(){'use strict';let old=new Uint8Array(10485760);old[0]=7;watch(old);for(let k=0;k<3000;k++){if(k===150){alive();if(old[0]!==7)throw 'lost slot owner';}if(k===200)old=null;if(k===500)released();}return 1;}f();", true, false),
        ("function f(old){old=new Uint8Array(10485760);old[0]=7;watch(old);for(var k=0;k<3000;k++){if(k===150){alive();if(old[0]!==7)throw 'lost formal owner';}if(k===200)old=null;if(k===500){released();if(f.arguments[0]!==null)throw 'formal publication';}}return 1;}f(null);", true, false),
        ("function f(){'use strict';let old=new Uint8Array(10485760);old[0]=7;watch(old);try{for(let k=0;k<3000;k++){if(k===150){alive();if(old[0]!==7)throw 'lost fragment owner';}if(k===200)old=null;if(k===500)released();}}finally{if(old!==null)throw 'fragment restoration';}return 1;}f();", false, true),
        ("function f(old){old=new Uint8Array(10485760);old[0]=7;watch(old);try{for(var k=0;k<3000;k++){if(k===150){alive();if(old[0]!==7)throw 'lost formal fragment owner';}if(k===200)old=null;if(k===500){released();if(f.arguments[0]!==null)throw 'formal publication';}}}finally{if(old!==null)throw 'formal restoration';}return 1;}f(null);", false, true),
        ("function f(){'use strict';let old=new Uint8Array(10485760);old[0]=7;watch(old);let read=()=>old;for(let k=0;k<3000;k++){if(k===150){alive();if(read()[0]!==7)throw 'lost captured owner';}if(k===200)old=null;if(k===500)released();}return 1;}f();", true, false),
        ("function f(){'use strict';let old=new Uint8Array(10485760);old[0]=7;watch(old);let read=()=>old;try{for(let k=0;k<3000;k++){if(k===150){alive();if(read()[0]!==7)throw 'lost captured owner';}if(k===200)old=null;if(k===500)released();}}finally{if(read()!==null)throw 'captured restoration';}return 1;}f();", false, true),
    ];
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for (source, whole_frame, fragment) in fixtures {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(32);
            install_native(&mut engine, "watch", |i, _, args| {
                let object = args[0].as_obj().expect("watched typed array");
                let info = i.typed_arrays[&(crate::value::Gc::as_ptr(object) as usize)];
                WATCHED_BUFFER_OWNER.with(|owner| {
                    *owner.borrow_mut() = Some((crate::value::Gc::downgrade(object), info.buffer));
                });
                Ok(crate::value::Value::Undefined)
            });
            install_native(&mut engine, "alive", |i, _, _| {
                i.gc_collect();
                WATCHED_BUFFER_OWNER.with(|owner| {
                    let owner = owner.borrow();
                    let (weak, buffer) = owner.as_ref().unwrap();
                    assert!(weak.strong_count() > 0, "canonical owner survives GC");
                    assert!(i.array_buffers.contains_key(buffer));
                });
                Ok(crate::value::Value::Undefined)
            });
            install_native(&mut engine, "released", |i, _, _| {
                i.gc_collect();
                WATCHED_BUFFER_OWNER.with(|owner| {
                    let owner = owner.borrow();
                    let (weak, buffer) = owner.as_ref().unwrap();
                    assert_eq!(
                        weak.strong_count(),
                        0,
                        "parked home must release its old object"
                    );
                    assert!(
                        !i.array_buffers.contains_key(buffer),
                        "old backing store must be swept during the live continuation"
                    );
                });
                Ok(crate::value::Value::Undefined)
            });
            reset_counts();
            assert_eq!(value(&mut engine, source), "1", "{tier:?}: {source}");
            if tier != Tier::Interp {
                assert_eq!(
                    crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()) > 0,
                    whole_frame
                );
                assert_eq!(
                    crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get())
                        > 0,
                    fragment
                );
                if fragment {
                    assert!(
                        crate::bytecode::loop_fragment::TEST_PRIVATE_FRAGMENT_ENTRIES
                            .with(|count| count.get())
                            > 0
                    );
                }
            }
            WATCHED_BUFFER_OWNER.with(|owner| *owner.borrow_mut() = None);
        }
    }
}

#[test]
fn native_callees_of_tree_walker_calls_reach_their_inline_checkpoint() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(32);
    value(&mut engine, "'use strict';function inner(n){return n+1;}function hot(n){let sum=0;for(let k=0;k<n;k++)sum+=inner(k);return sum;}");
    for _ in 0..150 {
        // Each top-level script has no loop and executes on the reference path. The short
        // callee becomes ordinary native code by call count; it never migrates a loop frame.
        assert_eq!(value(&mut engine, "hot(8)"), "36");
    }
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    if crate::bytecode::inline_recompile_at() == 100 {
        let hot = function(&mut engine, "hot");
        let primary = hot.code.get().and_then(Option::as_ref).unwrap();
        assert_eq!(
            primary.jit_runs.get(),
            100,
            "each generic native entry must count once, until promotion"
        );
        let spliced = hot
            .code2
            .get()
            .and_then(Option::as_ref)
            .expect("tree-walker calls must admit hot ordinary callee inlining");
        assert!(spliced
            .jit_ops()
            .iter()
            .any(|op| matches!(op, crate::bytecode::Op::InlineGuard(..))));
        assert_eq!(
            spliced.jit_runs.get(),
            18,
            "later calls select the published chunk"
        );
    }
}

#[test]
fn borrowed_outer_frames_keep_promoting_ordinary_callees_after_ast_warmup() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        PROMOTION_OBSERVATIONS.with(|observations| observations.borrow_mut().clear());
        install_native(&mut engine, "observePromotion", |i, _, _| {
            // Record only: assertions belong outside the native helper's extern-C boundary.
            let mut observation = PromotionObservation::default();
            if let Some(frame) = i.fn_frames.last() {
                let callee = frame.callee();
                let outer = callee.borrow();
                if let crate::value::Callable::User(user) = &outer.call {
                    observation.outer_second_stage = user.func.code2.get().is_some();
                    if let Some(chunk) = user.func.code.get().and_then(Option::as_ref) {
                        observation.outer_borrowed = chunk.osr.get().is_some();
                        observation.outer_primary = chunk.jit.get().is_some();
                    }
                }
            }
            let global = i.global_env.clone();
            if let Ok(hot) = i.get_var("hot", &global) {
                if let Some(hot) = hot.as_obj() {
                    let hot = hot.borrow();
                    if let crate::value::Callable::User(user) = &hot.call {
                        observation.primary_runs = user
                            .func
                            .code
                            .get()
                            .and_then(Option::as_ref)
                            .map(|chunk| chunk.jit_runs.get());
                        if let Some(spliced) = user.func.code2.get().and_then(Option::as_ref) {
                            observation.spliced_runs = Some(spliced.jit_runs.get());
                            observation.spliced_native = spliced.jit.get().flatten().is_some();
                            observation.spliced_guard = spliced
                                .jit_ops()
                                .iter()
                                .any(|op| matches!(op, crate::bytecode::Op::InlineGuard(..)));
                        }
                    }
                }
            }
            PROMOTION_OBSERVATIONS.with(|observations| observations.borrow_mut().push(observation));
            Ok(crate::value::Value::Undefined)
        });
        reset_counts();
        assert_eq!(value(&mut engine, "'use strict';function inner(n){return n+1;}function hot(n){if(n<0)throw 7;let sum=0;for(let k=0;k<n;k++)sum+=inner(k);return sum;}function drive(){let sum=0;for(let k=0;k<200;k++)sum+=hot(8);for(let k=0;k<10;k++){sum+=hot(8);if(k===0)observePromotion();}observePromotion();try{hot(-1);}catch(e){sum+=e;}finally{sum++;}return sum;}drive();"), "7568", "{tier:?}");
        if tier != Tier::Interp {
            assert!(crate::bytecode::TEST_FUNCTION_OSR_ENTRIES.with(|count| count.get()) > 0);
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit && crate::bytecode::inline_recompile_at() == 100 {
            PROMOTION_OBSERVATIONS.with(|observations| {
                let observations = observations.borrow();
                assert_eq!(observations.len(), 2);
                for observation in observations.iter() {
                    assert!(
                        observation.outer_borrowed
                            && !observation.outer_primary
                            && !observation.outer_second_stage,
                        "active borrowed outer body must keep its original sidecar"
                    );
                    assert!(
                        observation.spliced_native && observation.spliced_guard,
                        "ordinary hot callee must run its published splice"
                    );
                    assert_eq!(observation.primary_runs, Some(100));
                }
                // The hot function's first32 calls stay on the tree-walker. Both probes
                // occur after compilation and splice publication: all201/210 calls so far
                // therefore contain169/178 actual native entries across the two versions.
                // This includes first cache fills and the epoch refill after publication.
                assert_eq!(
                    observations[0].primary_runs.unwrap() + observations[0].spliced_runs.unwrap(),
                    169
                );
                assert_eq!(
                    observations[1].primary_runs.unwrap() + observations[1].spliced_runs.unwrap(),
                    178
                );
                assert_eq!(
                    observations[1].spliced_runs.unwrap() - observations[0].spliced_runs.unwrap(),
                    9,
                    "each established checked-cache hit must count once"
                );
            });
        }
    }
}

#[derive(Default)]
struct PromotionObservation {
    outer_second_stage: bool,
    outer_borrowed: bool,
    outer_primary: bool,
    primary_runs: Option<u32>,
    spliced_runs: Option<u32>,
    spliced_native: bool,
    spliced_guard: bool,
}
thread_local! {
    static PROMOTION_OBSERVATIONS: std::cell::RefCell<Vec<PromotionObservation>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

#[test]
fn generic_native_promotion_uses_the_callees_running_realm() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm, set_global};
    use crate::value::Value;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(32);
    let i = &mut *engine.interp;
    set_global(i, "marker", Value::str("main"));
    let child = i.create_realm();
    set_global(i, "childRealm", child.clone());
    let callee = in_realm(i, &child, |i| {
        eval_string(i, "globalThis.marker='child';globalThis.inner=function(){'use strict';return [marker];};globalThis.hot=function(){'use strict';const out=inner();return out;};");
        let global = i.global_env.clone();
        i.get_var("hot", &global)
            .unwrap_or_else(|_| panic!("child hot function"))
    });
    for _ in 0..150 {
        // The parent Realm initiates this generic call. Realm activation must precede
        // both the promotion proof and the selected native body's literal allocation.
        let out = i
            .call(callee.clone(), Value::Undefined, &[])
            .unwrap_or_else(|_| panic!("child call completes"));
        set_global(i, "last", out);
        assert_eq!(
            eval_string(
                i,
                "last[0]==='child'&&Object.getPrototypeOf(last)===childRealm.Array.prototype"
            ),
            "true"
        );
    }
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    if crate::bytecode::inline_recompile_at() == 100 {
        let callee = callee.as_obj().unwrap().borrow();
        let crate::value::Callable::User(user) = &callee.call else {
            panic!("child user callee")
        };
        let spliced = user
            .func
            .code2
            .get()
            .and_then(Option::as_ref)
            .expect("generic cross-Realm native calls must admit inlining");
        assert!(spliced
            .jit_ops()
            .iter()
            .any(|op| matches!(op, crate::bytecode::Op::InlineGuard(..))));
    }
}
