//! Actual native integer mirror consumers, effect boundaries and exact recovery.
//! ECMA-262 e28783d5 ToInt32/ToUint32, Number shifts and completion records.

use super::{
    integers::{Demand, Point, TEST_MIRRORS},
    ranges::NumericPlan,
};
use crate::bytecode::{self, native_deopt::ENTRIES, Op, Tier};
use crate::jit_ir::Cfg;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("integer fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn fixture_plan(
    source: &str,
) -> (
    std::rc::Rc<bytecode::Chunk>,
    super::stack::StackPlan,
    NumericPlan,
) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("fixture parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function fixture")
    };
    let chunk = bytecode::compile(function).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = super::stack::StackPlan::build(&chunk, &cfg).unwrap();
    let values = super::value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
    let demand = Demand::build(&chunk, &stack).unwrap();
    let plan = NumericPlan::with_integer_demand(&chunk, &cfg, &stack, Some(&values), Some(&demand))
        .unwrap();
    (chunk, stack, plan)
}

#[test]
fn integer_demand_is_sparse_and_preserves_unsigned_phi_semantics() {
    let (_, _, empty) = fixture_plan("function subject(a){var x=a+1;return x;}");
    let empty = empty.integers.as_ref().unwrap();
    assert!(!empty.locals.iter().any(|&active| active));
    assert!(!empty.operands.iter().any(|&active| active));

    let (chunk, stack, plan) = fixture_plan(
        "function subject(a,flag){var x=a>>>0;var y=flag?x:(a|0);side();return(y^3)&7;}",
    );
    let facts = plan.integers.as_ref().unwrap();
    let pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::BitXor))
        .unwrap();
    let location = chunk.jit_frame().1 + stack.at[pc].depth - 2;
    let range = facts.at(pc, Point::Before, location).unwrap();
    assert_eq!((range.min, range.max), (i32::MIN as i64, u32::MAX as i64));
    assert!(
        facts.locals.iter().any(|&active| active),
        "copy demand reaches private homes"
    );
    let call = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..)))
        .unwrap();
    assert!(
        (0..chunk.jit_frame().1).any(|slot| {
            facts.at(call, Point::Before, slot).is_some()
                && facts.at(call, Point::Before, slot) == facts.at(call, Point::After, slot)
        }),
        "an observer preserves audited private numeric words"
    );
}

#[test]
fn captured_and_mixed_bigint_definitions_never_gain_integer_mirrors() {
    let (chunk, stack, plan) = fixture_plan(
        "function subject(a){var x=a>>>0;function change(){x=1.5;}change();return x|0;}",
    );
    let pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::BitOr))
        .unwrap();
    let facts = plan.integers.as_ref().unwrap();
    assert!(facts
        .at(
            pc,
            Point::Before,
            chunk.jit_frame().1 + stack.at[pc].depth - 2
        )
        .is_none());
    let (chunk, stack, plan) =
        fixture_plan("function subject(a,b,flag){var x=flag?(a&b):(a>>>0);return x|0;}");
    let pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::BitOr))
        .unwrap();
    assert!(plan
        .integers
        .as_ref()
        .unwrap()
        .at(
            pc,
            Point::Before,
            chunk.jit_frame().1 + stack.at[pc].depth - 2
        )
        .is_none());
}

const NATIVE: &str = r#"
    var calls=0,coercions=0;
    function hook(){calls++;var temp={self:null};temp.self=temp;}
    function subject(items,input,flag){
        var unsigned=input>>>0;
        var chosen=flag?unsigned:(input|0);
        hook();
        var index=(chosen^3)&7;
        items[index]=chosen;
        return unsigned+':'+chosen+':'+(chosen|0)+':'+(chosen>>>0)+':'+items[index];
    }
    var items=[0,0,0,0,0,0,0,0];
    var object={valueOf(){coercions++;return -1;}};
"#;

#[test]
fn native_integer_mirrors_feed_bitwise_and_indexes_with_eager_canonical_values() {
    let cases = [
        ("-1", true, "4294967295:4294967295:-1:4294967295:4294967295"),
        ("-1", false, "4294967295:-1:-1:4294967295:-1"),
        (
            "2147483648",
            true,
            "2147483648:2147483648:-2147483648:2147483648:2147483648",
        ),
        (
            "2147483648",
            false,
            "2147483648:-2147483648:-2147483648:2147483648:-2147483648",
        ),
        ("4294967296", true, "0:0:0:0:0"),
        ("-0", false, "0:0:0:0:0"),
        ("NaN", true, "0:0:0:0:0"),
        ("Infinity", false, "0:0:0:0:0"),
        ("-Infinity", true, "0:0:0:0:0"),
        ("1.5", false, "1:1:1:1:1"),
        (
            "9007199254740991",
            true,
            "4294967295:4294967295:-1:4294967295:4294967295",
        ),
        ("9007199254740992", false, "0:0:0:0:0"),
    ];
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, NATIVE);
        let code = if tier == Tier::Jit {
            TEST_MIRRORS.with(|counts| counts.set([0; 4]));
            let code = super::call_tests::optimized(&mut engine, "subject", None);
            let counts = TEST_MIRRORS.with(|counts| counts.get());
            eprintln!("integer mirror emission [ToInt32,index,copy,definition]={counts:?}");
            if std::env::var("TRUST_JS_OPT_JIT_INT_SSA").as_deref() == Ok("0")
                || std::env::var("TRUST_JS_OPT_JIT_RANGES").as_deref() == Ok("0")
            {
                assert_eq!(counts, [0; 4]);
            } else {
                assert!(
                    counts[0] > 0 && counts[2] > 0 && counts[3] > 0,
                    "real mirror lowering: {counts:?}"
                );
                if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0") {
                    assert!(counts[1] > 0, "native index consumes an integer mirror");
                }
            }
            code.residency.referenced.set(0);
            Some(code)
        } else {
            None
        };
        for (input, flag, expected) in cases {
            assert_eq!(
                eval(&mut engine, &format!("subject(items,{input},{flag})")),
                expected,
                "{tier:?} {input} {flag}"
            );
        }
        assert_eq!(
            eval(&mut engine, "subject(items,object,true)+'|'+coercions"),
            "4294967295:4294967295:-1:4294967295:4294967295|1"
        );
        assert_eq!(
            eval(
                &mut engine,
                "try{subject(items,1n,true)}catch(error){error.name}"
            ),
            "TypeError"
        );
        if let Some(code) = code {
            assert_eq!(
                code.residency.referenced.get(),
                1,
                "actual optimizing native entry"
            );
            assert_eq!(code.residency.active.get(), 0);
        }
        assert_eq!(engine.interp.depth, 0);
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
    }
}

#[test]
fn native_mirrors_survive_polls_and_reset_at_effectful_completion_landings() {
    let source = r#"
        function subject(n){
            var total=0,items=[0,0,0,0];
            outer:for(var i=0;i<20;i++){
                var key=(i^3)&3;
                try { items[key]=i; if(i===3)continue outer; if(i===7)throw 'x'; }
                catch(error){total+=items[key]|0;}
                finally {total+=(items[key]^1)>>>0;}
            }
            return total;
        }
    "#;
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    let expected = eval(&mut oracle, "subject(0)");
    for pc in [None, Some(0)] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let code = super::call_tests::optimized(&mut engine, "subject", pc);
        engine.interp.gc_next = 0;
        engine.interp.interrupt_poll_tick = 16_383;
        assert_eq!(eval(&mut engine, "subject(0)"), expected);
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn native_integer_definitions_keep_precision_negative_zero_and_bigint_fallbacks() {
    let source = r#"
        function subject(flag){
            var safe=9007199254740991,large=4294967296*2+5;
            var changed=safe;changed++;
            var zero=0;zero=-zero;
            var product=5,before=product|0;product=0*-1;
            var signed=flag?large:-large;
            var chosen=flag?(1n&3n):large;
            var out=(large|0)+':'+(safe|0)+':'+(changed|0)+':'+(1/zero)
                +':'+before+':'+(product|0)+':'+(1/product)+':'+(signed|0);
            try{out+=':'+(chosen|0);}catch(e){out+=':'+e.name;}
            var last=4294967295,items=[];items[last]=7;
            return out+':'+items[last]+':'+items.length;
        }
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let code =
            (tier == Tier::Jit).then(|| super::call_tests::optimized(&mut engine, "subject", None));
        assert_eq!(
            eval(&mut engine, "subject(false)"),
            "5:-1:0:-Infinity:5:0:-Infinity:-5:5:7:0",
            "{tier:?}"
        );
        assert_eq!(
            eval(&mut engine, "subject(true)"),
            "5:-1:0:-Infinity:5:0:-Infinity:5:TypeError:7:0",
            "{tier:?}"
        );
        if let Some(code) = code {
            assert_eq!(code.residency.active.get(), 0);
        }
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn native_integer_mirrors_keep_captured_mapped_and_eval_homes_observable() {
    let source = r#"
        function captured(input){var value=input>>>0;function change(){value=1.5;}
            change();return value|0;}
        function mapped(input){var privateValue=input>>>0;arguments[0]=1.5;
            return(privateValue|0)+':'+(input|0);}
        function evaluated(input){var value=input>>>0;eval('value=1.5');return value|0;}
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let codes = if tier == Tier::Jit {
            ["captured", "mapped", "evaluated"]
                .map(|name| super::call_tests::optimized(&mut engine, name, None))
                .to_vec()
        } else {
            Vec::new()
        };
        assert_eq!(
            eval(&mut engine, "captured(-1)+'|'+mapped(-1)+'|'+evaluated(-1)"),
            "1|-1:1|1",
            "{tier:?}"
        );
        for code in codes {
            assert_eq!(code.residency.referenced.get(), 1);
            assert_eq!(code.residency.active.get(), 0);
        }
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn every_integer_boundary_recovers_exact_semantic_numbers_and_effects_once() {
    let source = r#"
        function subject(input){var direct=8589934597;var x=input>>>0;var y=(x^3)&7;
            hook();items[y]=x;try{return(x|0)+':'+items[y]+':'+(direct|0)+':'+direct;}finally{calls+=10;}}
        var items=[0,0,0,0,0,0,0,0],calls=0;
        function hook(){calls++;}
    "#;
    let (chunk, _, _) = fixture_plan(source);
    let count = chunk.jit_ops().len();
    let mut reached = 0;
    for pc in 0..=count {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        let code = super::call_tests::optimized(&mut engine, "subject", Some(pc));
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        assert_eq!(
            eval(&mut engine, "subject(-1)+'|'+calls"),
            "-1:4294967295:5:8589934597|11",
            "PC {pc}"
        );
        ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert!(entries.len() <= 1, "terminal recovery");
            if let Some(&(actual, _, _)) = entries.first() {
                assert_eq!(actual, pc);
                reached += 1;
            }
        });
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
    assert!(
        reached > 10,
        "exercise actual before-effect continuation boundaries"
    );
}
