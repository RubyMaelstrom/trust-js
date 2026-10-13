//! ECMA-262 e28783d5: GetValue/Call execute once; Number preserves -0 and NaN,
//! while Boolean temporaries remain Boolean until an explicit numeric coercion.
//! Canonical-slot identity is independent of type proof and register residency.
use super::*;
use crate::{bytecode::Tier, value::Value, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("scalar fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn warmed(tier: Tier, source: &str) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(if tier == Tier::Jit {
        Tier::Bytecode
    } else {
        tier
    });
    engine.set_tier_threshold(0);
    engine.interp.def_method(
        &engine.interp.global,
        "collectScalarTest",
        0,
        |interp, _, _| {
            interp.gc_collect();
            Ok(Value::Undefined)
        },
    );
    evaluate(&mut engine, source);
    engine.set_tier(tier);
    engine
}

#[test]
fn scalar_state_clean_publication_is_empty_and_residency_is_lazy() {
    let interp = crate::interpreter::Interp::new();
    let layout = crate::value::jit_layout(&interp.object_proto);
    assert!(
        std::mem::size_of::<Operand>() <= 8,
        "bounded operand metadata"
    );
    for kind in [ScalarKind::Number, ScalarKind::Integer, ScalarKind::Boolean] {
        let mut a = asm::Asm::new();
        let mut stack = vec![Operand::clean(kind, 0, false)];
        materialize(&mut a, &layout, &stack, 1);
        assert!(
            a.buf.is_empty(),
            "clean canonical value needs no store or SP pair"
        );
        scalar_register(&mut a, &mut stack, 1, 0);
        assert_eq!(a.buf.len(), 2, "one word load and one bit-preserving move");
        assert_eq!(stack[0], Operand::clean(kind, 0, true));
        scalar_register(&mut a, &mut stack, 1, 0);
        materialize(&mut a, &layout, &stack, 1);
        assert_eq!(
            a.buf.len(),
            2,
            "resident clean scalar must not reload or rewrite"
        );

        duplicate_operand(&mut a, &mut stack, 1);
        assert_eq!(stack[0], Operand::clean(kind, 0, true));
        assert_eq!(
            stack[1],
            Operand::temporary(kind),
            "Dup creates a distinct dirty slot"
        );
        let checkpoint = a.buf.len();
        materialize(&mut a, &layout, &stack, 1);
        assert!(
            a.buf.len() > checkpoint,
            "new duplicate really gets published"
        );
        stack.pop();
        stack.push(Operand::temporary(ScalarKind::Number));
        assert_ne!(
            stack[1],
            Operand::clean(kind, 1, true),
            "slot reuse cannot inherit old identity"
        );
    }
}

#[test]
fn scalar_state_clean_identity_is_checked_before_publication_or_reload() {
    let interp = crate::interpreter::Interp::new();
    let layout = crate::value::jit_layout(&interp.object_proto);
    let malformed = Operand::clean(ScalarKind::Number, 1, false);
    let mut a = asm::Asm::new();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        materialize(&mut a, &layout, &[malformed], 1);
    }))
    .is_err());
    assert!(a.buf.is_empty());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scalar_register(&mut a, &mut [malformed], 1, 0);
    }))
    .is_err());
    assert!(a.buf.is_empty());
}

#[test]
fn scalar_state_native_materialization_uses_original_base_and_exact_new_extent() {
    let interp = crate::interpreter::Interp::new();
    let layout = crate::value::jit_layout(&interp.object_proto);
    for duplicate in [true, false] {
        let physical = if duplicate { 1 } else { 2 };
        let mut a = asm::Asm::new();
        a.stp_pre(19, 20, -16);
        a.mov(19, 0);
        a.add_imm(20, 0, physical as u32 * 8);
        let mut stack = vec![Operand::clean(ScalarKind::Number, 0, false)];
        if duplicate {
            duplicate_operand(&mut a, &mut stack, physical);
        } else {
            scalar_register(&mut a, &mut stack, physical, 0);
        }
        let index = stack.len() - 1;
        a.fneg(16 + index as u32, 16 + index as u32);
        stack[index] = Operand::temporary(ScalarKind::Number);
        // Grow at offset0, or overwrite offset-16 and shrink. Both addresses
        // refer to the original x20 and must preserve neighboring canary words.
        materialize(&mut a, &layout, &stack, physical);
        a.sub_reg(0, 20, 19);
        a.ldp_post(19, 20, 16);
        a.ret();
        let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
        let code = ExecutableBuffer::from_bytes(&bytes).unwrap();
        let run: unsafe extern "C" fn(*mut u64) -> usize =
            unsafe { std::mem::transmute(code.as_ptr()) };
        for value in [
            0.0f64,
            -0.0,
            3.25,
            -7.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let original = if value.is_nan() {
                crate::value::PACK_CANON_NAN
            } else {
                value.to_bits()
            };
            let expected = if value.is_nan() {
                crate::value::PACK_CANON_NAN
            } else {
                (-value).to_bits()
            };
            let mut words = [original, 0xdead_beef, 0xcafe_babe];
            assert_eq!(
                unsafe { run(words.as_mut_ptr()) },
                if duplicate { 16 } else { 8 }
            );
            assert_eq!(
                words,
                if duplicate {
                    [original, expected, 0xcafe_babe]
                } else {
                    [expected, 0xdead_beef, 0xcafe_babe]
                },
                "{value:?}"
            );
        }
    }
}

#[test]
fn scalar_state_consecutive_indexed_reads_and_surplus_calls_keep_native_prefix() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = warmed(
            tier,
            r#"
            var scalarArray=[1.25,2.25,3.25],extra={keep:1};
            var setter={value:0,set(gl,value){try{this.value=value+gl.bias;return this.value;}finally{}}};
            var gl={bias:0.5};
            function indexed(n){var sum=0;for(var i=0;i<n;i++){
                scalarArray[0]=i+0.5;sum+=scalarArray[0]+scalarArray[1]+scalarArray[2];}return sum;}
            function surplus(n){var sum=0;for(var i=0;i<n;i++)sum+=setter.set(gl,i,extra);return sum;}
            for(var warm=0;warm<40;warm++){indexed(12);surplus(12);}
        "#,
        );
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(&mut engine, "[indexed(12),surplus(12)].join('|')"),
            "138|72",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(after[4] > before[4], "actual checked numeric result");
            assert!(
                after[6] > before[6],
                "actual preserved canonical scalar prefix"
            );
        }
    }
}

#[test]
fn scalar_state_second_guard_miss_keeps_clean_prefix_and_exact_effect_order() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = warmed(
            tier,
            r#"
            var mode=0,readsA=0,readsB=0,coercions=0,token={};
            var source={get a(){readsA++;collectScalarTest();return mode?{valueOf(){
                coercions++;collectScalarTest();if(mode===2)throw token;return 4;}}:3;},
                get b(){readsB++;collectScalarTest();return 7;}};
            function combine(v,o){var sum=0;for(var i=0;i<8;i++)sum+=(v*2+1)+o.a+o.b;return sum;}
            for(var warm=0;warm<40;warm++)combine(2,source);
            readsA=0;readsB=0;coercions=0;
        "#,
        );
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var a=combine(2,source);mode=1;var b=combine(2,source);mode=2;var caught=false;
            try{combine(2,source);}catch(e){caught=e===token;}
            [a,b,caught,readsA,readsB,coercions].join('|');
        "#
            ),
            "120|128|true|17|16|9",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(
                after[4] > before[4] && after[5] > before[5],
                "actual post-effect success and miss"
            );
            assert!(
                after[6] > before[6],
                "scalar prefix is preserved across author code"
            );
        }
    }
}

#[test]
fn scalar_state_boolean_dup_store_phi_and_effects_never_use_numeric_payloads() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = warmed(
            tier,
            r#"
            var calls=0,same={};
            function touch(){try{calls++;collectScalarTest();return 0;}finally{}}
            function boolean(a,b,which){var saved,phi;
                if(which)phi=(a===b);else phi=(a!==b);
                var first=(saved=phi)|touch();
                var second=(a===b)<<touch();
                return (saved?100:0)+first+second+(~phi);}
            for(var warm=0;warm<40;warm++){boolean(same,same,true);boolean({},same,false);}
            calls=0;
        "#,
        );
        let entries = EXECUTED_REGIONS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            [boolean(same,same,true),boolean({},same,true),
             boolean(same,same,false),boolean({},same,false),calls].join('|');
        "#
            ),
            "100|-1|0|99|8",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert!(
                EXECUTED_REGIONS.with(std::cell::Cell::get) > entries,
                "actual native scalar region"
            );
        }
    }
}

#[test]
fn scalar_state_nan_zero_and_borrowed_owner_replacement_cross_effects() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = warmed(
            tier,
            r#"
            var one={get value(){collectScalarTest();return 1;}};
            function special(v,o){var result=v*1;for(var i=0;i<6;i++)result=result*o.value;return result;}
            function overwrite(o,replacement){var prior=o.child;return (o.child=replacement,collectScalarTest(),prior.value+o.child.value);}
            function choose(v,o){var x=+(v*o.value);return (x&&x)||(x??7);}
            for(var warm=0;warm<40;warm++){special(2,one);overwrite({child:{value:2}},{value:3});choose(2,one);}
        "#,
        );
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            [Object.is(special(-0,one),-0),Number.isNaN(special(NaN,one)),
             special(Infinity,one),special(-Infinity,one),
             Object.is(choose(-0,one),-0),Number.isNaN(choose(NaN,one)),
             overwrite({child:{value:2}},{value:3})].join('|');
        "#
            ),
            "true|true|Infinity|-Infinity|true|true|5",
            "{tier:?}"
        );
    }
}

#[test]
fn scalar_state_inline_success_carries_clean_prefix_and_distinct_duplicates() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = warmed(
            tier,
            r#"
            var receiver={value:3};
            function scalarInline(value){return value*2+1;}
            function caller(o,n){var sum=0,saved;for(var i=0;i<n;i++)
                sum+=(saved=(i*2+1))+scalarInline(o.value)+saved;return sum;}
            for(var warm=0;warm<40;warm++)caller(receiver,10);
        "#,
        );
        let before = GUARDED_REGIONS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var good=true;for(var r=0;r<180;r++)good=caller(receiver,10)===270&&good;
            var reads=0,coercions=0;
            Object.defineProperty(receiver,'value',{get(){reads++;collectScalarTest();
                return {valueOf(){coercions++;collectScalarTest();return 4;}};}});
            [good,caller(receiver,10),reads,coercions].join('|');
        "#
            ),
            "true|290|10|10",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = GUARDED_REGIONS.with(std::cell::Cell::get);
            assert!(
                after[1] > before[1],
                "actual inline guard success with carried operand state"
            );
        }
    }
}

#[test]
fn scalar_state_private_homes_survive_unary_and_key_conversion_observers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = warmed(
            tier,
            r#"
            var stateRecord={x:3},stateMarker={},stateThrows=false,stateGets=0;
            function scalarConversions(o,n,v){
                var changed=0;
                var key={toString(){changed++;collectScalarTest();return 'x';}};
                var sum=0;
                for(var i=0;i<n;i++){
                    sum+=i+1;
                    void v;
                    var category=typeof v,text=`${key}`;
                    var value=o[key],{x}=o;
                    if(category!=='number'||text!=='x')throw 'wrong conversion';
                    sum+=value+x+changed;
                }
                return sum;
            }
            for(var warm=0;warm<8;warm++)scalarConversions(stateRecord,4,7);
        "#,
        );
        let before = GUARDED_REGIONS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
                var good=true;
                for(var r=0;r<8;r++)good=scalarConversions(stateRecord,4,NaN)===54&&good;
                Object.defineProperty(stateRecord,'x',{get(){
                    stateGets++;collectScalarTest();if(stateThrows)throw stateMarker;return 5;
                }});
                good=scalarConversions(stateRecord,4,-0)===70&&stateGets===8&&good;
                stateThrows=true;
                try{scalarConversions(stateRecord,4,Infinity);good=false;}
                catch(e){good=e===stateMarker&&stateGets===9&&good;}
                good;
                "#,
            ),
            "true",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = GUARDED_REGIONS.with(std::cell::Cell::get);
            assert!(after[0] > before[0], "actual native region execution");
        }
    }
}
