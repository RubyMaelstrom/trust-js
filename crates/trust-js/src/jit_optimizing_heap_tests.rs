//! Native heap operations: results/effects, actual-path coverage, storage and owner invariants.
//! Normative expectations follow ECMA-262 e28783d5 OrdinaryGet/Set, array [[DefineOwnProperty]],
//! ToPropertyKey and WeakRef liveness, not merely agreement with another engine tier.

use super::*;
use crate::bytecode::Tier;
use crate::value::{Gc, Value as JsValue};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("heap fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env.clone();
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .unwrap()
        .as_obj()
        .unwrap()
        .clone()
}

fn native_helpers(actual: &[usize], expected: &[usize], context: &str) {
    if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0") {
        assert!(
            actual.iter().sum::<usize>() > expected.iter().sum(),
            "the explicit checked-path ablation must actually exercise more helpers: {context}"
        );
    } else {
        assert_eq!(actual, expected, "{context}");
    }
}

fn run(
    source: &str,
    invocation: &str,
    expected: &str,
    prepare: Option<fn(&mut Engine)>,
) -> (Engine, [usize; 3]) {
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(
        eval(&mut oracle, invocation),
        expected,
        "independent interpreter/normative result"
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    if let Some(prepare) = prepare {
        prepare(&mut engine);
    }
    let code = call_tests::optimized(&mut engine, "subject", None);
    code.residency.referenced.set(0);
    elements::TEST_HEAP_HELPERS.with(|count| count.set([0; 3]));
    assert_eq!(eval(&mut engine, invocation), expected);
    let counts = elements::TEST_HEAP_HELPERS.with(|count| count.get());
    assert_eq!(
        code.residency.referenced.get(),
        1,
        "exact optimized subject entered"
    );
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
    (engine, counts)
}

#[test]
fn native_elements_keep_every_value_category_and_all_receiver_conventions() {
    for held in [
        "undefined",
        "null",
        "false",
        "true",
        "0",
        "-0",
        "NaN",
        "Infinity",
        "-Infinity",
        "'shared string'",
        "Symbol('shared')",
        "({x:1})",
        "123456789012345678901234567890n",
    ] {
        let source = format!(
            r#"
            function subject() {{
                var a=root,total=0;
                for(var k=0;k<40;k++) {{
                    a[0]=held;
                    var kept=(a[1]=held);
                    (0,a)[2]=held;
                    kept=((0,a)[3]=held);
                    if(Object.is(a[0],held)&&Object.is((0,a)[1],kept))total++;
                    total+=a[4]();
                }}
                return total;
            }}
            var held={held};
            var root=[held,held,held,held,function(){{return this===root?1:1000;}}];
        "#
        );
        let (_, counts) = run(&source, "subject()", "80", None);
        native_helpers(
            &counts[1..],
            &[0, 0],
            &format!("native read/write and method receiver for {held}"),
        );
    }
}

#[test]
fn native_elements_cover_inline_vector_and_classic_dense_storage() {
    for initializer in [
        "[1,2,3]",
        "[1,2,3,4,5,6,7,8,9,10,11,12]",
        "({'0':1,'1':2,'2':3})",
    ] {
        let source = format!(
            r#"
            function subject() {{
                var a=root,total=0;
                for(var k=0;k<40;k++){{a[1]=k;total+=a[1];(0,a)[2]=k;total+=(0,a)[2];}}
                return total+'|'+a[0];
            }}
            var root={initializer};
        "#
        );
        let (_, counts) = run(&source, "subject()", "1560|1", None);
        native_helpers(
            &counts[1..],
            &[0, 0],
            &format!("all dense representations: {initializer}"),
        );
    }
}

#[test]
fn native_field_stores_cover_this_local_stack_and_assignment_result() {
    for held in [
        "({x:1})",
        "'shared string'",
        "Symbol('field')",
        "12345678901234567890n",
    ] {
        let source = format!(
            r#"
            function subject() {{
                var row=root, total=0;
                for(var k=0;k<40;k++) {{
                    row.value=held;
                    (0,row).value=held;
                    var result=(row.value=held);
                    this.value=held;
                    if(result===held&&row.value===held)total++;
                }}
                return total;
            }}
            var held={held}, root={{value:held}};
        "#
        );
        let (_, counts) = run(&source, "subject.call(root)", "40", None);
        native_helpers(
            &counts,
            &[4, 0, 0],
            &format!("one fill for each of four native store sites, {held}"),
        );
    }
}

#[test]
fn native_polymorphic_stores_probe_every_cache_way() {
    let (_, counts) = run(
        r#"
        function subject() {
            var total=0;
            for(var k=0;k<120;k++){var row=rows[k%4];row.value=held;if(row.value===held)total++;}
            return total;
        }
        var held={},rows=[{value:held},{a:0,value:held},{a:0,b:0,value:held},{a:0,b:0,c:0,value:held}];
    "#,
        "subject()",
        "120",
        None,
    );
    native_helpers(
        &counts,
        &[4, 0, 0],
        "four fills, not one checked call for every shape rotation",
    );
}

#[test]
fn aliased_property_receiver_and_result_owners_are_combined_before_commit() {
    let (mut engine, counts) = run(
        r#"
        function subject() {
            var row=root,a=array,total=0;
            for(var k=0;k<100;k++) {
                row.self=row;var result=((0,row).self=row);
                a[0]=a;result=((0,a)[0]=a);
                if(result===a&&row.self===row&&(0,a)[0]===a)total++;
            }
            return total;
        }
        var root={self:null};root.self=root;
        var array=[null];array[0]=array;
    "#,
        "subject()",
        "100",
        None,
    );
    native_helpers(&counts, &[2, 0, 0], "self aliases");
    // Every iteration transfers owners, not cumulative retains. Root binding + self edge
    // + this Rust check's temporary strong owner are the only surviving references.
    assert_eq!(std::rc::Rc::strong_count(&object(&mut engine, "root")), 3);
    assert_eq!(std::rc::Rc::strong_count(&object(&mut engine, "array")), 3);
}

#[test]
fn consumed_self_cycles_with_no_remaining_owner_take_real_destruction() {
    let (_, counts) = run(
        r#"
        function subject() {
            for(var k=0;k<5;k++)make(k).self=null;
            for(var k=0;k<5;k++)makeArray(k)[0]=null;
            return root.self+'|'+array[0];
        }
        var root={self:null};root.self=root;var array=[null];array[0]=array;
        function make(k){if(k===0)return root;var x={self:null};x.self=x;return x;}
        function makeArray(k){if(k===0)return array;var x=[null];x[0]=x;return x;}
    "#,
        "subject()",
        "null|null",
        None,
    );
    if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0") {
        assert_eq!(counts, [5, 1, 5]);
    } else {
        assert_eq!(
            counts,
            [5, 0, 4],
            "joint last-owner drops must not decrement to zero inline"
        );
    }
}

#[test]
fn descriptor_changes_holes_prototypes_and_nonwritable_lengths_keep_standard_behavior() {
    run(
        r#"
        function subject() {
            var a=[1,2,3],row={x:1},trace=[];
            a[1]=4;row.x=2;
            Object.defineProperty(a,'1',{get(){trace.push('ag');return 8;},set(v){trace.push('as'+v);},configurable:true});
            Object.defineProperty(row,'x',{get(){trace.push('og');return 9;},set(v){trace.push('os'+v);},configurable:true});
            a[1]=5;row.x=6;trace.push(a[1],row.x);
            delete a[1];Object.setPrototypeOf(a,{get 1(){trace.push('pg');return 10;},set 1(v){trace.push('ps'+v);}});
            a[1]=7;trace.push(a[1]);
            Object.defineProperty(a,'0',{writable:false});a[0]=11;trace.push(a[0]);
            Object.defineProperty(a,'length',{writable:false});a[2]=12;a[3]=13;
            trace.push(a[2],a.length,a[3]);
            Object.preventExtensions(row);row.extra=14;trace.push('extra' in row);
            return trace.join('|');
        }
    "#,
        "subject()",
        "as5|os6|ag|og|8|9|ps7|pg|10|1|12|3||false",
        None,
    );
}

#[test]
fn strict_rejections_and_throwing_coercions_preserve_completed_private_writes() {
    run(
        r#"
        function subject() {
            'use strict';var x=1,a=[1],row={x:2},trace=[];
            Object.freeze(a);Object.freeze(row);
            try{x=3;a[0]=4;}catch(e){trace.push(x,e.name);}
            try{x=5;row.x=6;}catch(e){trace.push(x,e.name);}
            var key={toString(){trace.push('key');throw 'stop';}};
            try{x=7;a[key]=8;}catch(e){trace.push(x,e);}
            try{x=9;null[0]=10;}catch(e){trace.push(x,e.name);}
            return trace.join('|');
        }
    "#,
        "subject()",
        "3|TypeError|5|TypeError|key|7|stop|9|TypeError",
        None,
    );
}

#[test]
fn exotic_receivers_and_nonindex_keys_keep_exactly_once_conversion_and_traps() {
    run(
        r#"
        function subject() {
            var trace=[],target={x:0,0:1},p=new Proxy(target,{
                get(t,k,r){trace.push('g'+String(k));return Reflect.get(t,k,r);},
                set(t,k,v,r){trace.push('s'+String(k));return Reflect.set(t,k,v,r);}
            });
            p[0]=2;p.x=3;trace.push(p[0],p.x);
            var a=[1],key={toString(){trace.push('key');return '0';}};
            a[key]=4;trace.push(a[key]);
            a[-0]=5;a[1.5]=6;a[-1]=7;a[NaN]=8;a[4294967295]=9;
            var s=Symbol('key');a[s]=10;
            trace.push(a[0],a[1.5],a[-1],a[NaN],a[4294967295],a[s],a.length);
            var t=new Uint8Array(1);t[0]={valueOf(){trace.push('value');return 258;}};trace.push(t[0]);
            return trace.join('|');
        }
    "#,
        "subject()",
        "s0|sx|g0|gx|2|3|key|key|4|5|6|7|8|9|10|1|value|2",
        None,
    );
}

#[test]
fn native_numeric_stores_preserve_mirrors_signed_zero_and_real_i32_proofs() {
    fn prepare(engine: &mut Engine) {
        assert!(object(engine, "root")
            .borrow_mut()
            .props
            .prepare_packed_numeric_mirror());
    }
    for assignment in ["a[0]=value", "(0,a)[0]=value", "a['0']=value"] {
        for (value, bits) in [
            ("0.5", 0.5f64.to_bits()),
            ("-0", (-0.0f64).to_bits()),
            ("NaN", f64::NAN.to_bits()),
            ("Infinity", f64::INFINITY.to_bits()),
        ] {
            let source = format!(
                r#"
                function subject(){{var a=root;{assignment};return Object.is(a[0],value);}}
                var root=[1,2,3],value={value};
            "#
            );
            let (mut engine, counts) = run(&source, "subject()", "true", Some(prepare));
            native_helpers(&counts, &[0; 3], &format!("{assignment} with {value}"));
            let root = object(&mut engine, "root");
            let borrowed = root.borrow();
            let JsValue::Num(actual) = borrowed.props.get_index(0).unwrap().value() else {
                panic!("canonical element must remain numeric: {assignment} with {value}");
            };
            assert_eq!(
                actual.to_bits(),
                bits,
                "canonical {assignment} with {value}"
            );
            let layout = crate::value::jit_layout(&engine.interp.object_proto);
            let flag = unsafe {
                *(&borrowed.props as *const _ as *const u8).add(layout.props_mirror_flags)
            };
            if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0")
                && assignment == "a['0']=value"
            {
                // The checked String-key [[Set]] path requests a mutable Property
                // and therefore invalidates the optional mirror; numeric SetElem
                // uses mirror-coherent set_index_value instead.
                // Keep the native preservation assertion below strict when enabled.
                assert!(borrowed.props.mirror_get(0).is_none());
                assert_eq!(flag & crate::value::MIRROR_OK, 0);
                continue;
            }
            assert_eq!(
                borrowed
                    .props
                    .mirror_get(0)
                    .unwrap_or_else(|| panic!("numeric mirror retained: {assignment} with {value}"))
                    .to_bits(),
                bits
            );
            assert_ne!(flag & crate::value::MIRROR_OK, 0);
            assert_eq!(flag & crate::value::MIRROR_ALL_I32, 0);
        }
    }
}

#[test]
fn native_heterogeneous_write_invalidates_only_the_numeric_view_and_clears_retry_hints() {
    fn prepare(engine: &mut Engine) {
        assert!(object(engine, "root")
            .borrow_mut()
            .props
            .prepare_packed_numeric_mirror());
    }
    let (mut engine, counts) = run(
        r#"
        function subject(){var a=root;a[0]=held;return a[0]===held&&a[1]===2;}
        var root=[1,2,3],held={value:9};
    "#,
        "subject()",
        "true",
        Some(prepare),
    );
    native_helpers(&counts, &[0; 3], "numeric mirror invalidation");
    let root = object(&mut engine, "root");
    assert!(root.borrow().props.mirror_get(0).is_none());
    assert!(root.borrow().props.mirror_get(1).is_none());
    assert!(matches!(
        root.borrow().props.get_index(1).unwrap().value(),
        JsValue::Num(2.0)
    ));
    assert!(!root.borrow_mut().props.prepare_packed_numeric_mirror());
    // A second native numeric write must invalidate MIRROR_PACKED_FAILED so preparation
    // can retry without a structure-changing operation or mutable-property escape.
    eval(
        &mut engine,
        "function repair(){var a=root;a[0]=1;return a[0];}",
    );
    let code = call_tests::optimized(&mut engine, "repair", None);
    assert_eq!(eval(&mut engine, "repair()"), "1");
    assert_eq!(code.residency.referenced.get(), 1);
    assert!(root.borrow_mut().props.prepare_packed_numeric_mirror());
    assert_eq!(root.borrow().props.mirror_get(0), Some(1.0));
}

#[test]
fn canonical_string_indices_are_native_and_keep_aliased_key_owners() {
    let (mut engine, counts) = run(
        r#"
        function subject() {
            var a=root,total=0;
            for(var k=0;k<100;k++) {
                a[key]=key;var value=((0,a)[key]=key);
                if(a[key]===value&&(0,a)[key]===key)total++;
                a[ten]=key;if(a[ten]===key)total++;
                total+=a[method]();
            }
            return total;
        }
        var key='0',ten='10',method='11';
        var root=[key,1,2,3,4,5,6,7,8,9,key,function(){return this===root?1:1000;}];
    "#,
        "subject()",
        "300",
        None,
    );
    native_helpers(
        &counts,
        &[0; 3],
        "canonical string index aliases/method receiver",
    );
    fn count(engine: &mut Engine) -> usize {
        let env = engine.interp.global_env.clone();
        let JsValue::Str(key) = engine.interp.get_var("key", &env).ok().unwrap() else {
            panic!("string key")
        };
        key.strong_count()
    }
    let before = count(&mut engine);
    assert_eq!(eval(&mut engine, "subject()"), "300");
    engine.interp.gc_collect();
    assert_eq!(
        count(&mut engine),
        before,
        "the same key/old value/result allocation must not leak per iteration"
    );
}

#[test]
fn numeric_string_guards_reject_noncanonical_unicode_and_out_of_range_names() {
    run(
        r#"
        function subject() {
            var a=[1],result=[];
            var keys=['00','-0','+0',' 0','0 ','0.0','1e0','0x0','4294967295',
                '4294967296','9999999999','10000000000','', '\uFF10','\uD800','\0'];
            for(var k=0;k<keys.length;k++){a[keys[k]]=k+10;result.push(a[keys[k]]===k+10);}
            var target={};target['4294967294']=7;
            result.push(a[0]===1,a.length===1,target['4294967294']===7);
            return result.every(Boolean);
        }
    "#,
        "subject()",
        "true",
        None,
    );
}

#[test]
fn string_index_fallback_destroys_unique_keys_without_replaying_rhs() {
    let (_, counts) = run(
        r#"
        function subject() {
            var a=[1],trace=[],value;
            function key(){trace.push('key');return String(0.1).slice(0,1);}
            for(var k=0;k<20;k++){value=((0,a)[key()]=(trace.push('rhs'),k));trace.push(value,a[0]);}
            return trace.length+'|'+a[0];
        }
    "#,
        "subject()",
        "80|19",
        None,
    );
    assert_eq!(
        counts[2], 20,
        "each unique index String must take the checked last-owner destruction path"
    );
}

thread_local! {
    static LIVE_OWNER: std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>> =
        const { std::cell::RefCell::new(None) };
    static COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn make_owner(
    interp: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    let object = interp.new_object();
    object
        .borrow_mut()
        .props
        .insert("value", crate::value::Property::plain(JsValue::Num(7.0)));
    object.borrow_mut().props.insert(
        "self",
        crate::value::Property::plain(JsValue::Obj(object.clone())),
    );
    LIVE_OWNER.with(|owner| *owner.borrow_mut() = Some(std::rc::Rc::downgrade(&object)));
    Ok(JsValue::Obj(object))
}

fn collect(
    interp: &mut crate::interpreter::Interp,
    _: JsValue,
    _: &[JsValue],
) -> Result<JsValue, JsValue> {
    interp.gc_collect();
    LIVE_OWNER.with(|owner| {
        assert!(
            owner.borrow().as_ref().unwrap().upgrade().is_some(),
            "the last reachable owner transferred to a native field must survive collection"
        )
    });
    COLLECTIONS.with(|count| count.set(count.get() + 1));
    Ok(JsValue::Undefined)
}

#[test]
fn native_heap_transfers_remain_canonical_at_reentrant_gc_boundaries() {
    COLLECTIONS.with(|count| count.set(0));
    tests::check_with_installer(
        r#"
        function subject() {
            var row={value:makeOwner()},a=[row.value];row.value=null;
            for(var k=0;k<16;k++) {
                row.value=a[0];a[0]=null;collectOwners();
                a[0]=row.value;row.value=null;collectOwners();
            }
            return a[0].value;
        }
    "#,
        "7",
        |engine| {
            let global = engine.interp.global.clone();
            engine
                .interp
                .def_method(&global, "makeOwner", 0, make_owner);
            engine
                .interp
                .def_method(&global, "collectOwners", 0, collect);
        },
    );
    assert_eq!(
        COLLECTIONS.with(|count| count.get()),
        192,
        "three native invocations and three independent oracles"
    );
}

#[test]
fn native_heap_layout_checks_refuse_incompatible_property_and_sidecar_layouts() {
    let engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(elements::supported(&layout));
    let mut changed = layout;
    changed.packed_elems_valid = false;
    assert!(!elements::supported(&changed));
    let mut changed = layout;
    changed.property_size += 8;
    assert!(!elements::supported(&changed));
    let mut changed = layout;
    changed.property_value += 8;
    assert!(!elements::supported(&changed));
    let mut changed = layout;
    changed.property_meta += 1;
    assert!(!elements::supported(&changed));
    let mut changed = layout;
    changed.dense_inline_data = i32::MAX as usize + 1;
    assert!(!elements::supported(&changed));
}

#[test]
fn expanded_heap_ir_work_limit_declines_without_publishing_partial_machine_code() {
    let source = "function subject(a,k,v){a[k]=v;return a[k];}";
    let ast = crate::parser::parse_script(source, false).ok().unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &ast[0] else {
        panic!("function")
    };
    let chunk = bytecode::compile(function).unwrap();
    let mut engine = Engine::new();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let interp = crate::interpreter::interp_layout(&mut engine.interp);
    for (instructions, blocks) in [(0, 65_536), (262_144, 0)] {
        assert!(matches!(
            super::compile_with_limits(&chunk, &values, &interp, None, instructions, blocks),
            Err(CompileError::LoweringBudget)
        ));
    }
    assert!(
        super::compile_with_deopt(&chunk, &values, &interp, None).is_ok(),
        "a refused attempt must not poison the chunk or a subsequent complete compilation"
    );
    eval(&mut engine, source);
    assert_eq!(eval(&mut engine, "subject([1],0,7)"), "7");
}
