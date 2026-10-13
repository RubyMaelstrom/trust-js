//! ECMA-262 e28783d5 Number::lessThan/equal and While/ForBodyEvaluation:
//! preserve NaN, condition ordering, owners and polling under the original
//! false-exit latch. Semantic coverage survives the rejected inversion experiment.
use super::*;
use crate::{bytecode::Tier, Completion, Engine};

fn executable(words: Vec<u32>) -> ExecutableBuffer {
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    ExecutableBuffer::from_bytes(&bytes).expect("latch native probe")
}

fn condition_holds(condition: u32, flags: u32) -> bool {
    let n = flags & 8 != 0;
    let z = flags & 4 != 0;
    let c = flags & 2 != 0;
    let v = flags & 1 != 0;
    match condition {
        0 => z,
        1 => !z,
        4 => n,
        5 => !n,
        8 => c && !z,
        9 => !c || z,
        10 => n == v,
        11 => n != v,
        12 => !z && n == v,
        13 => z || n != v,
        _ => panic!("unexpected loop predicate {condition}"),
    }
}

#[test]
fn loop_latch_native_complements_cover_every_nzcv_state() {
    // Arm 100076_0100 D1.8: each admitted condition and its xor-1 counterpart
    // are complements, including FP unordered NZCV=0011. MSR NZCV,X0 is EL0-safe.
    for false_condition in [0, 1, 5, 8, 10, 11, 12, 13] {
        for condition in [false_condition, false_condition ^ 1] {
            let mut a = asm::Asm::new();
            a.buf.push(0xd51b_4200); // MSR NZCV,X0
            a.cset_w(0, condition);
            a.ret();
            let code = executable(a.finish());
            let run: unsafe extern "C" fn(u64) -> u32 =
                unsafe { std::mem::transmute(code.as_ptr()) };
            for flags in 0..16 {
                assert_eq!(
                    unsafe { run(u64::from(flags) << 28) },
                    u32::from(condition_holds(condition, flags)),
                    "condition={condition} NZCV={flags:04b}"
                );
                assert_ne!(
                    condition_holds(condition, flags),
                    condition_holds(condition ^ 1, flags)
                );
            }
        }
    }
}

// Six distinct production CmpBranch operand cases. Minimal plans are emission-only:
// every target is bound, but these deliberately body-free loops are never executed.
fn comparison_plan(case: usize, negative: u32) -> (LoopPlan, u32, bool) {
    let (left, right, left_integer, right_integer, right_integral, left_integral) = match case {
        0 => (SlotRes::I(2), SlotRes::I(3), true, true, false, false),
        1 => (SlotRes::I(2), SlotRes::F(9), true, false, false, false),
        2 => (SlotRes::I(2), SlotRes::F(9), true, false, true, false),
        3 => (SlotRes::F(8), SlotRes::I(3), false, true, false, true),
        4 | 5 => (SlotRes::F(8), SlotRes::F(9), false, false, false, false),
        _ => unreachable!(),
    };
    let slot = |off, res, int_checked| SlotPlan {
        off,
        res,
        preload: true,
        stored: false,
        virgin: false,
        int_checked,
    };
    let left_kind = if left_integer {
        PushKind::I { neg: true }
    } else {
        PushKind::D { iv: left_integral }
    };
    let right_kind = if right_integer {
        PushKind::I { neg: true }
    } else {
        PushKind::D { iv: right_integral }
    };
    let (right_op, right_kind) = match case {
        1 => (
            ChainOp::ConstNum(3.0_f64.to_bits()),
            PushKind::K(3.0_f64.to_bits()),
        ),
        4 => (
            ChainOp::ConstNum(0.0_f64.to_bits()),
            PushKind::K(0.0_f64.to_bits()),
        ),
        _ => (ChainOp::Load(8), right_kind),
    };
    let mut cmp = asm::Asm::new();
    match case {
        0 => cmp.cmp_reg_x(2, 3),
        1 => cmp.cmp_imm_x(2, 3),
        2 => cmp.cmp_reg_x(2, 9),
        3 => cmp.cmp_reg_x(9, 3),
        4 => cmp.fcmp_zero(8),
        5 => cmp.fcmp(8, 9),
        _ => unreachable!(),
    }
    (
        LoopPlan {
            head: 0,
            exit_pc: 1,
            initialization_guards: vec![],
            chain: vec![
                (ChainOp::Load(0), 0),
                (right_op, 0),
                (ChainOp::CmpBranch(negative, 1), 0),
                (ChainOp::KeyNop, 0),
            ],
            cond_len: 3,
            kinds: vec![left_kind, right_kind, PushKind::None, PushKind::None],
            slots: vec![slot(0, left, left_integral), slot(8, right, right_integral)],
            receivers: vec![],
            elem_retain: vec![],
            elem_reuse: vec![],
            conv_retain: vec![],
            conv_reuse: vec![],
            setelem_i32: Default::default(),
            names: vec![],
            uses_ext: false,
        },
        cmp.finish()[0],
        case < 4,
    )
}

fn conditional_target(word: u32, pc: usize) -> usize {
    assert_eq!(word & 0xff00_0010, 0x5400_0000);
    let displacement = ((word << 8) as i32) >> 13;
    pc.checked_add_signed(displacement as isize)
        .expect("branch target")
}

fn unconditional_target(word: u32, pc: usize) -> usize {
    assert_eq!(word & 0xfc00_0000, 0x1400_0000);
    let displacement = ((word << 6) as i32) >> 6;
    pc.checked_add_signed(displacement as isize)
        .expect("unconditional branch target")
}

#[test]
fn loop_latch_emission_preserves_false_exits_in_all_six_cases_after_integer_mapping() {
    let mut engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    assert!(ilayout.valid);
    for negative in [5, 13, 8, 11, 1, 0] {
        for case in 0..6 {
            let (plan, compare, integer) = comparison_plan(case, negative);
            let mut a = asm::Asm::new();
            let labels = [a.new_label(), a.new_label()];
            let fallback =
                emit_loop_chain(&mut a, &layout, &ilayout, &plan, &labels, &mut [false; 2]);
            a.bind(fallback);
            for label in labels {
                a.bind(label);
            }
            a.ret();
            let words = a.finish();
            let compares: Vec<_> = words
                .iter()
                .enumerate()
                .filter_map(|(pc, word)| (*word == compare).then_some(pc))
                .collect();
            assert_eq!(compares.len(), 2, "case={case} predicate={negative}");
            let false_condition = if integer {
                match negative {
                    5 => 10,
                    8 => 12,
                    other => other,
                }
            } else {
                negative
            };
            let initial = compares[0] + 1;
            let bottom = compares[1] + 1;
            assert_eq!(words[initial] & 15, false_condition);
            assert!(conditional_target(words[initial], initial) > bottom + 1);
            assert_eq!(words[bottom] & 15, false_condition);
            assert!(conditional_target(words[bottom], bottom) > bottom + 1);
            let body = unconditional_target(words[bottom + 1], bottom + 1);
            assert!(body < bottom && body > initial);
            assert!(
                words[body..bottom]
                    .iter()
                    .all(|word| word & 0xfc00_0000 != 0x1400_0000),
                "body/poll/condition prefix has no extra unconditional branch"
            );
        }
    }
}

#[test]
fn loop_latch_native_floating_predicates_preserve_nan_zeros_and_infinities() {
    let values = [
        f64::NEG_INFINITY,
        -2147483649.0,
        -1.5,
        -0.0,
        0.0,
        0.5,
        2147483648.0,
        f64::INFINITY,
        f64::NAN,
    ];
    for (false_condition, relation) in [5, 13, 8, 11, 1, 0].into_iter().zip(0..6) {
        let mut a = asm::Asm::new();
        let yes = a.new_label();
        a.fcmp(0, 1);
        a.b_cond(false_condition ^ 1, yes);
        a.movz(0, 0, 0);
        a.ret();
        a.bind(yes);
        a.movz(0, 1, 0);
        a.ret();
        let code = executable(a.finish());
        let run: unsafe extern "C" fn(f64, f64) -> u32 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        for left in values {
            for right in values {
                let expected = match relation {
                    0 => left < right,
                    1 => left > right,
                    2 => left <= right,
                    3 => left >= right,
                    4 => left == right,
                    5 => left != right,
                    _ => unreachable!(),
                };
                assert_eq!(
                    unsafe { run(left, right) },
                    u32::from(expected),
                    "predicate={relation}, left={left}, right={right}"
                );
            }
        }
    }
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("latch fixture parse") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn reset_entries() {
    TEST_NUMERIC_REGION_ENTRIES.with(|count| count.set(0));
}
fn assert_entered(tier: Tier) {
    if tier == Tier::Jit {
        assert!(
            TEST_NUMERIC_REGION_ENTRIES.with(|count| count.get()) > 0,
            "actual numeric loop entry, not only code presence"
        );
    }
}

#[test]
fn loop_latch_language_relations_take_entry_and_bottom_paths_in_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        for (operator, first, bound, next) in [
            ("<", "-0", "0.5", "0.5"),
            (">", "2", "0.5", "0.5"),
            ("<=", "-0", "0", "0.5"),
            (">=", "0", "-0", "-0.5"),
            ("==", "-0", "0", "0.5"),
            ("===", "-0", "0", "0.5"),
            ("!=", "NaN", "0", "0"),
            ("!==", "NaN", "0", "0"),
        ] {
            eval(&mut engine, &format!("function latch(a,b,next){{var count=0;while(a{operator}b){{count++;a=next;}}return count;}}"));
            reset_entries();
            assert_eq!(
                eval(&mut engine, &format!("latch({first},{bound},{next})")),
                "1",
                "{tier:?} {operator}"
            );
            assert_entered(tier);
            assert_eq!(
                eval(&mut engine, &format!("latch({next},{bound},{next})")),
                "0",
                "entry {tier:?} {operator}"
            );
            if !["!=", "!=="].contains(&operator) {
                assert_eq!(
                    eval(&mut engine, "latch(NaN,0,1)+':'+latch(0,NaN,1)"),
                    "0:0"
                );
            }
        }
    }
}

#[test]
fn loop_latch_condition_mutation_and_integer_bailouts_preserve_committed_state() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            function nanEnd(limit){var count=0;for(var i=0;i<limit;i++){count++;limit=NaN;}return count+':'+i;}
            function overflow(){var total=0;for(var i=2147483646;i<2147483650;i++)total+=i;return total+':'+i;}
            function firstStore(limit){var value;for(var i=0;i<limit;i++)value=i+0.5;return String(value)+':'+i;}
            function fractional(){var count=0;for(var i=0;i<2;i+=0.5)count++;return count+':'+i;}
        "#,
        );
        reset_entries();
        assert_eq!(
            eval(
                &mut engine,
                "nanEnd(3)+','+overflow()+','+firstStore(0)+','+firstStore(2)+','+fractional()"
            ),
            "1:1,8589934590:2147483650,undefined:0,1.5:2,4:2"
        );
        assert_entered(tier);
    }
}

#[test]
fn loop_latch_coercive_bound_fallback_never_replays_updates_or_conversion() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let source = r#"
            var conversions=0,saved='',token={};
            function coercive(bound,next){var count=0,i=0;try{
                for(;i<bound;i++){count++;bound=next;}
            }finally{saved=count+':'+i;}return saved;}
            var normal={valueOf(){conversions++;return 0;}};
            var throwing={valueOf(){conversions++;throw token;}};
            var a=coercive(3,normal),first=conversions;conversions=0;
            var caught=false;try{coercive(3,throwing);}catch(e){caught=e===token;}
            a+':'+first+'|'+saved+':'+conversions+':'+caught;
        "#;
        assert_eq!(eval(&mut engine, source), "1:1:1|1:1:1:true", "{tier:?}");
    }
    // A non-Number in a preloaded local can reject the numeric preamble. This
    // test protects observable fallback ordering, not an in-loop native hit.
}

#[test]
fn loop_latch_loop_completions_and_finally_remain_tier_equivalent() {
    let source = r#"
        var log=[];function run(){var sum=0;try{for(var i=0;i<6;i++){
            try{if(i===1)continue;if(i===4)break;sum+=i;}finally{log.push(i);}
        }return sum;}finally{log.push('outer');}}
        var result=run();result+':'+log.join(',');
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), "5:0,1,2,3,4,outer", "{tier:?}");
    }
}

thread_local! {
    static LOCAL_ROOT: std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>> = const { std::cell::RefCell::new(None) };
}

fn arm_latch_gc(i: &mut Interp, _: Value, args: &[Value]) -> Result<Value, Value> {
    let object = args[0].as_obj().expect("local root");
    LOCAL_ROOT.with(|root| *root.borrow_mut() = Some(Rc::downgrade(object)));
    i.gc_next = 0;
    Ok(Value::Undefined)
}

fn verify_latch_gc(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    assert!(
        i.gc_next > 0,
        "requested allocation collection was serviced and rearmed"
    );
    LOCAL_ROOT.with(|root| {
        assert!(
            root.borrow().as_ref().unwrap().upgrade().is_some(),
            "active native frame owns local object"
        )
    });
    Ok(Value::Undefined)
}

#[test]
fn loop_latch_repeated_polls_gc_and_deadline_preserve_local_frame_owner() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "armLatchGc", 1, arm_latch_gc);
    engine
        .interp
        .def_method(&global, "verifyLatchGc", 0, verify_latch_gc);
    eval(&mut engine, "function longLoop(n){var only={answer:7};only.self=only;armLatchGc(only);var sum=0;for(var i=0;i<n;i++)sum+=i&255;verifyLatchGc();return only.answer+':'+sum;}");
    reset_entries();
    assert_eq!(eval(&mut engine, "longLoop(40000)"), "7:5093856");
    assert_entered(Tier::Jit);
    engine.interp.gc_collect();
    LOCAL_ROOT.with(|root| {
        assert!(
            root.borrow().as_ref().unwrap().upgrade().is_none(),
            "completed frame releases its self-cycle"
        )
    });
    engine.interrupt_handle().set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(20),
    ));
    let result = engine
        .eval_interruptible("longLoop(1e12)", false)
        .expect("deadline fixture");
    assert!(matches!(
        result,
        crate::ExecutionOutcome::Interrupted {
            reason: crate::InterruptReason::DeadlineExceeded
        }
    ));
    engine.interrupt_handle().set_deadline(None);
    engine.interp.gc_collect();
    LOCAL_ROOT.with(|root| {
        assert!(
            root.borrow_mut().take().unwrap().upgrade().is_none(),
            "interrupted frame releases its self-cycle"
        )
    });
}
