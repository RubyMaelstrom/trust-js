//! GetBindingValue / GetIdentifierReference / EvaluateCall conformance and
//! evidence that warmed reads actually bypass the checked name helper.
use super::tests::{check_real_call, check_warmed, check_warmed_then};
use super::*;

#[test]
fn native_names_keep_dirty_owner_publications_after_fast_reads() {
    // The first observer publishes the aliased old owner. Replacing it can
    // use a native non-last-owner drop. A cache hit must NOT tell FramePlan
    // that the replacement was published before the next call/collection.
    super::tests::check_with_gc_pressure(
        r#"
        function subject(){var kept=outside,answer=0;
          for(var k=0;k<30;k++){
            observe();kept={value:k};var marker=external;
            observe();answer+=kept.value+marker;kept=outside;
          }
          return answer+':'+outside.value}
        var outside={value:71},external=3;
        function observe(){$262.gc()}
    "#,
        "525:71",
        true,
    );
}

#[test]
fn native_names_read_warmed_global_and_lexical_values_without_helpers() {
    names::TEST_NAME_HELPERS.with(|n| n.set(0));
    check_warmed(
        r#"
        function subject(){var sum=0;for(var k=0;k<40;k++)sum+=external+lexical;return sum}
        var external=7;let lexical=3;
    "#,
        "400",
    );
    assert_eq!(names::TEST_NAME_HELPERS.with(|n| n.get()), 0);
}

#[test]
fn native_names_decode_wide_values_and_keep_heap_owners() {
    names::TEST_NAME_HELPERS.with(|n| n.set(0));
    check_warmed(
        r#"
        function subject(){return [undef===undefined,nil===null,yes,!no,
          negative===-3.25,1/zero,notNumber!==notNumber,
          text+'!',big+2n,symbol===symbol,object===object,object.value].join(':')}
        let undef=undefined,nil=null,yes=true,no=false,negative=-3.25,zero=-0,
          notNumber=NaN,text='words',big=9n,symbol=Symbol('key'),object={value:23};
    "#,
        "true:true:true:true:true:-Infinity:true:words!:11:true:true:23",
    );
    assert_eq!(names::TEST_NAME_HELPERS.with(|n| n.get()), 0);
}

#[test]
fn native_names_recheck_global_accessors_and_dynamic_lexical_shadowing() {
    check_warmed_then(
        r#"
        function subject(){var result=outside;return result+'|'+reads}
        globalThis.outside=7;var reads=0;
    "#,
        "7|0",
        "Object.defineProperty(globalThis,'outside',{get(){reads++;return 13},configurable:true});",
        "13|1",
    );
    check_real_call(
        r#"
        function subject(){var total=0;for(var k=0;k<30;k++){
          if(k===12)eval('var outside=9');total+=outside}
          return total}
        var outside=2;
    "#,
        "186",
    );
}

#[test]
fn native_names_preserve_with_receiver_unscopables_and_tdz() {
    let mut engine = crate::Engine::new();
    engine.set_tier(bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    engine.eval(r#"
        var blocked=false,outer=7,checks=0;
        var scope={outer:19,f(){return this===scope},get [Symbol.unscopables](){checks++;return {outer:blocked}}};
        var subject;with(scope){subject=function(){return [outer,f()].join(':')}}
    "#,false).unwrap();
    let code = super::call_tests::optimized(&mut engine, "subject", None);
    // Ordinary admission keeps functions created under with in the VM. Test
    // this backend's fallback using the real captured environment, like the
    // existing explicit native-body tests, without changing admission policy.
    let global = engine.interp.global_env.clone();
    let function = engine.interp.get_var("subject", &global).ok().unwrap();
    let function = function.as_obj().unwrap().borrow();
    let crate::value::Callable::User(user) = &function.call else {
        panic!("function")
    };
    let env = user.env.clone();
    let chunk = user.func.code.get().unwrap().as_ref().unwrap().clone();
    assert!(!chunk.prepared_entry);
    drop(function);
    code.residency.referenced.set(0);
    for (prepare, expected) in [("", "19:true"), ("blocked=true;", "7:true")] {
        engine.eval(prepare, false).unwrap();
        let value = crate::jit::run(
            &mut engine.interp,
            &chunk,
            &code,
            &env,
            crate::value::Value::Undefined,
            &[],
        )
        .unwrap_or_else(|_| panic!("checked with lookup threw"));
        assert_eq!(
            engine.interp.to_string(&value).ok().unwrap().as_ref(),
            expected
        );
    }
    assert!(
        matches!(engine.eval("checks",false).unwrap(),crate::Completion::Value(value) if value=="4"),
        "with resolution must evaluate Symbol.unscopables at each lookup"
    );
    assert_eq!(code.residency.referenced.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
    check_real_call(
        r#"
        function subject(){try{return later;let later=3}catch(e){return e.name}}
    "#,
        "ReferenceError",
    );
}

#[test]
fn native_names_follow_fresh_closure_instances_and_collection() {
    let source = r#"
        function factory(value){return function(){return value}}
        var subject=factory({value:7});
    "#;
    let mut engine = crate::Engine::new();
    engine.set_tier(bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    engine.eval(source, false).unwrap();
    let code = super::call_tests::optimized(&mut engine, "subject", None);
    code.residency.referenced.set(0);
    for (text, expected) in [
        ("subject().value", "7"),
        ("subject=factory({value:11});subject().value", "11"),
        ("subject().value", "11"),
    ] {
        let actual = engine.eval(text, false).unwrap();
        assert!(matches!(actual,crate::Completion::Value(value) if value==expected));
        engine.interp.gc_collect();
    }
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(code.residency.referenced.get(), 1);
}

#[test]
fn native_names_walk_deep_live_scopes_and_reuse_only_their_layouts() {
    let mut engine = crate::Engine::new();
    engine.set_tier(bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    engine
        .eval(
            r#"
        function factory(one){return function(){var two=one+1;
          return function(){return one+two}}}
        var subject=factory(3)();
    "#,
            false,
        )
        .unwrap();
    let code = super::call_tests::optimized(&mut engine, "subject", None);
    for (prepare, expected) in [("", "7"), ("subject=factory(13)();", "27")] {
        engine.eval(prepare, false).unwrap();
        // The first call fills the live cache and may change its cache mode.
        assert!(
            matches!(engine.eval("subject()",false).unwrap(),crate::Completion::Value(value) if value==expected)
        );
        names::TEST_NAME_HELPERS.with(|n| n.set(0));
        code.residency.referenced.set(0);
        engine.interp.gc_collect();
        assert!(
            matches!(engine.eval("subject()",false).unwrap(),crate::Completion::Value(value) if value==expected)
        );
        assert_eq!(names::TEST_NAME_HELPERS.with(|n| n.get()), 0);
        assert_eq!(
            code.residency.referenced.get(),
            1,
            "fresh closure uses the same optimized body"
        );
    }
}
