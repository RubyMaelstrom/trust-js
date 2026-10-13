//! ECMA-262 VariableDeclaration/ResolveBinding, ForIn/OfHeadEvaluation and
//! LexicallyScopedDeclarations/BlockDeclarationInstantiation (Annex B enabled).
//! Local official snapshot e28783d5fc9dc12b3de905961e2c71410b38a202.

use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.set((0, 0)));
        match engine.eval(source, false).expect("scope fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}: {source}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}\n{source}"),
        }
        let (vm, native) = crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.get());
        if !matches!(tier, Tier::Interp) {
            assert!(vm + native > 0, "must execute compiled Script: {source}");
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if matches!(tier, Tier::Jit) {
            assert!(native > 0, "must enter native Script code: {source}");
        }
    }
}

#[test]
fn script_scope_catch_var_initializer_writes_nearest_slot() {
    check(
        "var x='outer', seen;try{throw 'caught';}catch(x){var x='inner';seen=x;}seen+'|'+x;",
        "inner|outer",
    );
    check(
        "var x='outer', seen;try{throw 'caught';}catch(x){var [x]=['array'];var {v:x}={v:'object'};seen=x;}seen+'|'+x;",
        "object|outer",
    );
}

#[test]
fn script_scope_catch_var_does_not_mutate_captured_outer_binding() {
    check(
        "function outer(){return x;}var x='outer',seen;try{throw 'caught';}catch(x){var x='inner';seen=x;}seen+'|'+outer();",
        "inner|outer",
    );
    check(
        "var x='outer',inner;try{throw 'caught';}catch(x){inner=()=>x;var x='inner';}inner()+'|'+x;",
        "inner|outer",
    );
}

#[test]
fn script_scope_catch_for_initializer_preserves_outer_var() {
    check(
        "var x='outer',seen=[];try{throw 'caught';}catch(x){for(var x='inner';x!=='done';x='done'){seen.push(x);}seen.push(x);}seen.push(x);seen.join('|');",
        "inner|done|outer",
    );
}

#[test]
fn script_scope_for_in_rhs_closure_keeps_uninitialized_head() {
    check(
        "let x='outer';var before=()=>x,probe,seen=[];for(let x in {i:probe=()=>typeof x}){seen.push(x);}try{probe();}catch(e){seen.push(e.name);}seen.push(before());seen.join('|');",
        "i|ReferenceError|outer",
    );
    check(
        "let x='outer';var probe;for(const x in (probe=()=>typeof x,{})){}try{probe();}catch(e){e.name+'|'+x;}",
        "ReferenceError|outer",
    );
}

#[test]
fn script_scope_for_of_rhs_head_and_iteration_closures_are_distinct() {
    check(
        "let x='outer';var probe,iterations=[],seen=[];for(let x of (probe=()=>typeof x,[1,2])){iterations.push(()=>x);}try{probe();}catch(e){seen.push(e.name);}seen.push(iterations[0](),iterations[1](),x);seen.join('|');",
        "ReferenceError|1|2|outer",
    );
    check(
        "let x='outer';var probe;for(const [x] of (probe=()=>typeof x,[[7]])){}try{probe();}catch(e){e.name+'|'+x;}",
        "ReferenceError|outer",
    );
}

#[test]
fn script_scope_labelled_duplicate_functions_instantiate_before_body() {
    check(
        "var seen=[];{seen.push(f());function f(){return 3;}seen.push(f());a:b:function f(){return 4;}seen.push(f());}seen.push(f());seen.join('|');",
        "4|4|4|4",
    );
    check(
        "var read,seen=[];{read=()=>f;seen.push(f());function f(){return 3;}seen.push(f());label:function f(){return 4;}seen.push(f());}seen.push(read()===f,f());seen.join('|');",
        "4|4|4|true|4",
    );
    check(
        "var seen=[];switch(1){case 1:seen.push(f());function f(){return 3;}seen.push(f());case 2:label:function f(){return 4;}seen.push(f());}seen.push(f());seen.join('|');",
        "4|4|4|4",
    );
}
