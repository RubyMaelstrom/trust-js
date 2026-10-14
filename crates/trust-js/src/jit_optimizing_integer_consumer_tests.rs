//! Integer equality/ordering/truthiness and exact publication at native helper bypasses.
//! ECMA-262 e28783d5 IsLooselyEqual, Number::equal and ToBoolean.

use super::integer_consumers::TEST_SCALARS;
use crate::bytecode::{self, Op, Tier, native_deopt::ENTRIES};
use crate::interpreter::Interp;
use crate::value::Value;
use crate::{Completion, Engine};

thread_local! {
    static COLLECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn force_collection(interp: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    interp.gc_collect();
    COLLECTIONS.with(|count| count.set(count.get() + 1));
    Ok(Value::Undefined)
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("scalar fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

const SOURCE: &str = r#"
    function subject(input,flag){
        var unsigned=input>>>0,chosen=flag?unsigned:(input|0);
        var held={value:unsigned};
        var eq=chosen==4294967295,ne=chosen!=-1;
        var strict=chosen===4294967295,lt=chosen<0,le=chosen<=4294967295;
        var gt=chosen>-1,ge=chosen>=0,not=!chosen;
        var branch=chosen?1:2,short=(chosen&&3)|0,alternate=(chosen||4)|0;
        return eq+':'+ne+':'+strict+':'+lt+':'+le+':'+gt+':'+ge
            +':'+not+':'+branch+':'+short+':'+alternate+':'+held.value;
    }
"#;

#[test]
fn native_integer_equality_order_and_truthiness_use_semantic_i64_values() {
    let cases = [
        (
            "-1",
            true,
            "true:true:true:false:true:true:true:false:1:3:-1:4294967295",
        ),
        (
            "-1",
            false,
            "false:false:false:true:true:false:false:false:1:3:-1:4294967295",
        ),
        (
            "2147483648",
            true,
            "false:true:false:false:true:true:true:false:1:3:-2147483648:2147483648",
        ),
        (
            "2147483648",
            false,
            "false:true:false:true:true:false:false:false:1:3:-2147483648:2147483648",
        ),
        (
            "0",
            true,
            "false:true:false:false:true:true:true:true:2:0:4:0",
        ),
        (
            "-0",
            false,
            "false:true:false:false:true:true:true:true:2:0:4:0",
        ),
        (
            "NaN",
            true,
            "false:true:false:false:true:true:true:true:2:0:4:0",
        ),
        (
            "Infinity",
            false,
            "false:true:false:false:true:true:true:true:2:0:4:0",
        ),
    ];
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, SOURCE);
        let code = if tier == Tier::Jit {
            TEST_SCALARS.with(|counts| counts.set([0; 4]));
            let code = super::call_tests::optimized(&mut engine, "subject", None);
            let counts = TEST_SCALARS.with(|counts| counts.get());
            eprintln!("integer scalar emission [loose,strict/relational,branch,not]={counts:?}");
            if std::env::var("TRUST_JS_OPT_JIT_INT_SSA").as_deref() == Ok("0")
                || std::env::var("TRUST_JS_OPT_JIT_RANGES").as_deref() == Ok("0")
            {
                assert_eq!(counts, [0; 4]);
            } else {
                assert!(
                    counts.iter().all(|&count| count > 0),
                    "all real scalar consumer families {counts:?}"
                );
            }
            code.residency.referenced.set(0);
            Some(code)
        } else {
            None
        };
        for (input, flag, expected) in cases {
            assert_eq!(
                eval(&mut engine, &format!("subject({input},{flag})")),
                expected,
                "{tier:?} {input} {flag}"
            );
        }
        if let Some(code) = code {
            assert_eq!(code.residency.referenced.get(), 1);
            assert_eq!(code.residency.active.get(), 0);
        }
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn mixed_loose_equality_coercions_nan_signed_zero_and_bigint_remain_checked() {
    let source = r#"
        var trace=[];
        function subject(input){var scalar=3&7;
            return [input==scalar,input!=scalar,input===scalar,input!==scalar,input<scalar,
                !input,input?1:0,input??'nullish'].join(':');}
        var object={[Symbol.toPrimitive](hint){trace.push(hint);return 3;}};
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        TEST_SCALARS.with(|counts| counts.set([0; 4]));
        let code =
            (tier == Tier::Jit).then(|| super::call_tests::optimized(&mut engine, "subject", None));
        if tier == Tier::Jit {
            assert_eq!(
                TEST_SCALARS.with(|counts| counts.get()),
                [0; 4],
                "one exact operand must not select Number-only scalar paths"
            );
        }
        let cases = [
            ("3", "true:false:true:false:false:false:1:3"),
            ("'3'", "true:false:false:true:false:false:1:3"),
            ("3n", "true:false:false:true:false:false:1:3"),
            ("true", "false:true:false:true:true:false:1:true"),
            ("null", "false:true:false:true:true:true:0:nullish"),
            ("undefined", "false:true:false:true:false:true:0:nullish"),
            ("NaN", "false:true:false:true:false:true:0:NaN"),
            ("-0", "false:true:false:true:true:true:0:0"),
            ("Infinity", "false:true:false:true:false:false:1:Infinity"),
        ];
        for (input, expected) in cases {
            assert_eq!(
                eval(&mut engine, &format!("subject({input})")),
                expected,
                "{tier:?} {input}"
            );
        }
        assert_eq!(
            eval(&mut engine, "subject(object)+'|'+trace.join(',')"),
            "true:false:false:true:false:false:1:3|default,default,number,string"
        );
        if let Some(code) = code {
            assert_eq!(code.residency.active.get(), 0);
        }
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn exact_int53_phi_comparisons_do_not_truncate_to_signed_i32() {
    let source = r#"
        function subject(flag){var x=flag===2?0:flag?8589934597:-8589934597;
            return [x==5,x!=5,x===5,x<0,x>4294967295,x<=9007199254740991,x>=-9007199254740991,!x,x?1:0,x&&3,x||4,x!==5,x??9].join(':');}
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let code =
            (tier == Tier::Jit).then(|| super::call_tests::optimized(&mut engine, "subject", None));
        assert_eq!(
            eval(&mut engine, "subject(true)"),
            "false:true:false:false:true:true:true:false:1:3:8589934597:true:8589934597",
            "{tier:?}"
        );
        assert_eq!(
            eval(&mut engine, "subject(false)"),
            "false:true:false:true:false:true:true:false:1:3:-8589934597:true:-8589934597",
            "{tier:?}"
        );
        assert_eq!(
            eval(&mut engine, "subject(2)"),
            "false:true:false:false:false:true:true:true:0:0:4:true:0",
            "{tier:?}"
        );
        if let Some(code) = code {
            assert_eq!(code.residency.active.get(), 0);
        }
    }
}

#[test]
fn mixed_equality_fallback_publishes_private_owners_before_observers_and_throws() {
    let source = r#"
        var calls=0;
        function coercion(){calls++;this.owner.value=9;forceCollection();if(this.fail)throw 'boom';return 7;}
        function subject(input,fail){var privateOwner={value:7};
            privateOwner.self=privateOwner;
            var observer={owner:privateOwner,fail:fail,valueOf:coercion};var mask=input&7;
            try{var equal=mask==observer;return equal+':'+privateOwner.value;}
            catch(error){return error+':'+privateOwner.value;}
            finally{calls+=10;}}
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        COLLECTIONS.with(|count| count.set(0));
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let global = engine.interp.global.clone();
        engine
            .interp
            .def_method(&global, "forceCollection", 0, force_collection);
        eval(&mut engine, source);
        let code =
            (tier == Tier::Jit).then(|| super::call_tests::optimized(&mut engine, "subject", None));
        assert_eq!(
            eval(&mut engine, "subject(-1,false)+'|'+calls"),
            "true:9|11",
            "{tier:?}"
        );
        assert_eq!(
            eval(&mut engine, "subject(-1,true)+'|'+calls"),
            "boom:9|22",
            "{tier:?}"
        );
        if let Some(code) = code {
            assert_eq!(code.residency.referenced.get(), 1);
            assert_eq!(code.residency.active.get(), 0);
        }
        assert!(engine.interp.fn_frames.is_empty());
        assert_eq!(COLLECTIONS.with(|count| count.get()), 2, "{tier:?}");
        engine.interp.gc_collect();
    }
}

#[test]
fn scalar_helper_bypasses_keep_private_object_owners_dirty_for_every_recovery_boundary() {
    let source = r#"
        function subject(input){var held={value:input>>>0};var mask=input&7;
            var eq=mask==7,ne=mask!=0,not=!mask;
            hook();return held.value+':'+eq+':'+ne+':'+not;}
        var calls=0;function hook(){calls++;}
    "#;
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("fixture parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function fixture")
    };
    let chunk = bytecode::compile(function).unwrap();
    let count = chunk.jit_ops().len();
    assert!(chunk.jit_ops().iter().any(|op| matches!(op, Op::EqEq)));
    let mut reached = 0;
    for pc in std::iter::once(None).chain((0..=count).map(Some)) {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let code = super::call_tests::optimized(&mut engine, "subject", pc);
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        engine.interp.gc_next = 0;
        assert_eq!(
            eval(&mut engine, "subject(-1)+'|'+calls"),
            "4294967295:true:true:false|1",
            "PC {pc:?}"
        );
        ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert!(entries.len() <= 1);
            if let Some(&(actual, _, _)) = entries.first() {
                assert_eq!(Some(actual), pc);
                reached += 1;
            }
        });
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
    }
    eprintln!(
        "integer scalar recovery: {} compiled boundaries, {reached} actually reached",
        count + 1
    );
    assert!(reached > 20);
}
