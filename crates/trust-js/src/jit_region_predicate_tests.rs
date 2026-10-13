//! Predicate/ownership lowering, ECMA-262 e28783d5: Number::equal, IsStrictlyEqual,
//! ToBoolean, OrdinaryGet, and left-to-right equality Evaluation/GetValue.
use super::*;
use crate::{bytecode::Tier, value::Value, Completion, Engine};

#[test]
fn equality_numeric_template_executes_all_number_pairs() {
    use crate::value::{PackedValue, PACK_BOOL};

    #[repr(C)]
    struct Outcome {
        bits: u64,
        stack: *const u64,
    }

    let engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.global);
    assert!(eq_inlinable(&layout));
    let numbers = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0xfff8_0000_0000_0001),
        f64::from_bits(0x7ff0_0000_0000_0001),
    ];
    for strict in [false, true] {
        for negate in [false, true] {
            for branch in [false, true] {
                let mut a = asm::Asm::new();
                let unwind = a.new_label();
                let false_result = a.new_label();
                let exit = a.new_label();
                a.stp_pre(19, 20, -16);
                a.add_imm(20, 0, 16);
                emit_eq_inline(
                    &mut a,
                    &layout,
                    0,
                    unwind,
                    strict,
                    negate,
                    branch.then_some(false_result),
                );
                if branch {
                    a.mov_imm64(0, PACK_BOOL | 1);
                    a.b(exit);
                    a.bind(false_result);
                    a.mov_imm64(0, PACK_BOOL);
                } else {
                    a.bind(false_result);
                    emit_exec_word_load(&mut a, 0, 20, -8);
                }
                a.b(exit);
                a.bind(unwind);
                a.mov_imm64(0, u64::MAX);
                a.bind(exit);
                a.mov(1, 20);
                a.ldp_post(19, 20, 16);
                a.ret();
                let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
                let code = ExecutableBuffer::from_bytes(&bytes).expect("executable test template");
                let call: unsafe extern "C" fn(*mut u64) -> Outcome =
                    unsafe { std::mem::transmute(code.as_ptr()) };
                for left in numbers {
                    for right in numbers {
                        // Only canonical packed Numbers enter this isolated ABI:
                        // the fast template cannot call a helper or drop an owner.
                        // The executable allocation outlives every invocation.
                        let mut stack = [
                            PackedValue::scalar_bits(&Value::Num(left)).unwrap(),
                            PackedValue::scalar_bits(&Value::Num(right)).unwrap(),
                        ];
                        let result = unsafe { call(stack.as_mut_ptr()) };
                        let equal = (left == right) ^ negate;
                        assert_eq!(result.bits, PACK_BOOL | u64::from(equal));
                        assert_eq!(result.stack, unsafe {
                            stack.as_ptr().add(usize::from(!branch))
                        });
                    }
                }
            }
        }
    }
}

#[test]
fn equality_numeric_warmup_preserves_later_coercion_gc_and_throws() {
    check(
        r#"
        function se(a,b){return a[0]===b[0];}
        function sn(a,b){return a[0]!==b[0];}
        function le(a,b){return a[0]==b[0];}
        function ln(a,b){return a[0]!=b[0];}
        function seb(a,b){if(a[0]===b[0])return 11;return 22;}
        function snb(a,b){if(a[0]!==b[0])return 11;return 22;}
        function leb(a,b){if(a[0]==b[0])return 11;return 22;}
        function lnb(a,b){if(a[0]!=b[0])return 11;return 22;}
        function all(a,b){return [se(a,b),sn(a,b),le(a,b),ln(a,b),
            seb(a,b),snb(a,b),leb(a,b),lnb(a,b)].join(':');}
        for(var r=0;r<150;r++)all([r],[r]);
        var trace=[],token={},coercing={valueOf(){trace.push('C');collectPredicateTest();return 0;}};
        var a={get 0(){trace.push('L');collectPredicateTest();return NaN;}},
            b={get 0(){trace.push('R');collectPredicateTest();return coercing;}};
        var observed=all(a,b),order=trace.join('');
        var mixed=all([7],['7']),zero=all([-0],[0]),identity=all([token],[token]);
        coercing.valueOf=function(){trace.push('T');collectPredicateTest();throw token;};
        var caught=0;
        for(var f of [le,ln,leb,lnb])try{f(a,b);}catch(e){if(e===token)caught++;}
        var strict=se(a,b)===false&&sn(a,b)===true&&seb(a,b)===22&&snb(a,b)===11;
        observed+'|'+order+'|'+mixed+'|'+zero+'|'+identity+'|'+caught+'|'+strict;
        "#,
        "false:true:false:true:22:11:22:11|LRLRLRCLRCLRLRLRCLRC|false:true:true:false:22:11:11:22|true:false:true:false:11:22:11:22|true:false:true:false:11:22:11:22|4|true",
        false,
    );
}

fn check(source: &str, expected: &str, fused: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.interp.def_method(
            &engine.interp.global,
            "collectPredicateTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        let before = FUSED_PREDICATES.with(std::cell::Cell::get);
        match engine
            .eval(source, false)
            .expect("predicate fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        if fused && tier == Tier::Jit {
            assert!(
                FUSED_PREDICATES.with(std::cell::Cell::get) > before,
                "no actual fused region predicate executed"
            );
        }
    }
}

#[test]
fn region_predicates_numeric_unordered_zero_and_escaping_booleans() {
    check(
        r#"
        function compare(a,b){var s=0;if(a===b)s|=1;if(a!==b)s|=2;
            if(a<b)s|=4;if(a<=b)s|=8;if(a>b)s|=16;if(a>=b)s|=32;return s;}
        function numberResults(a,b){var x=a+0,y=b+0,s=0;
            if(x===y)s++;if(x!==y)s+=2;return [s,x===y,x!==y,(x<y)&&'less',(x>=y)||'less'].join(':');}
        var out=[];for(var r=0;r<4;r++){
            out=[compare(-0,0),compare(NaN,NaN),compare(NaN,1),compare(1,NaN),
                compare(Infinity,Infinity),compare(-Infinity,Infinity),compare(2,1),
                numberResults(1,2),numberResults(NaN,NaN),numberResults(-0,0)];
        }out.join('|');
    "#,
        "41|2|2|2|41|14|50|2:false:true:less:less|2:false:true:false:less|1:true:false:false:true",
        true,
    );
}

#[test]
fn region_predicates_generic_equality_and_coercion_order() {
    check(
        r#"
        function compare(a,b){var s=0;if(a===b)s|=1;if(a!==b)s|=2;if(a==b)s|=4;if(a!=b)s|=8;return s;}
        var same={},sym=Symbol('s'),trace=[],coercing={valueOf(){trace.push('v');return 7;}};
        var out=[];for(var r=0;r<4;r++)out=[compare(undefined,null),compare(-0,0),compare(NaN,NaN),
            compare('same',('s'+'ame').slice(0)),compare(7n,7),compare(7n,7n),compare(sym,sym),
            compare(Symbol('s'),sym),compare(same,same),compare({},{}),compare(coercing,7),
            compare($262.IsHTMLDDA,null)];
        var original=out.join('|')+'|'+trace.length;
        // Keep the tiny original shape, whose baseline already fuses every
        // comparison. The loop supplies genuinely amortized region work while
        // proving the same generic kinds, HTMLDDA and observable coercions.
        function compareLoop(a,b,n){var s=0;for(var i=0;i<n;i++){
            if(a===b)s|=1;if(a!==b)s|=2;if(a==b)s|=4;if(a!=b)s|=8;}return s;}
        var loop=[compareLoop(undefined,null,2),compareLoop(-0,0,2),compareLoop(NaN,NaN,2),
            compareLoop('same',('s'+'ame').slice(0),2),compareLoop(7n,7,2),compareLoop(7n,7n,2),compareLoop(sym,sym,2),
            compareLoop(Symbol('s'),sym,2),compareLoop(same,same,2),compareLoop({},{},2),compareLoop(coercing,7,2),
            compareLoop($262.IsHTMLDDA,null,2)];original+'|'+loop.join('|')+'|'+trace.length;
    "#,
        "6|5|10|5|6|5|5|10|5|10|6|6|8|6|5|10|5|6|5|5|10|5|10|6|6|12",
        true,
    );
}

#[test]
fn region_predicates_checked_reads_never_replay_on_miss_or_throw() {
    check(
        r#"
        var trace=[],token={};
        function compare(a,b){var left=a[0],right=b[0];
            collectPredicateTest();if(left===right)return 1;return 0;}
        var a={get 0(){trace.push('L');collectPredicateTest();return token;}},
            b={get 0(){trace.push('R');collectPredicateTest();return token;}};
        var out=[];for(var i=0;i<5;i++)out.push(compare(a,b));
        Object.defineProperty(b,'0',{get(){trace.push('T');throw token;}});
        try{compare(a,b);}catch(e){out.push(e===token);}
        out.join(',')+'|'+trace.join('');
    "#,
        "1,1,1,1,1,true|LRLRLRLRLRLT",
        true,
    );
}

#[test]
fn region_predicates_truthiness_nullish_and_last_owners() {
    check(
        r#"
        function observe(x){var a=x&&11,b=x||22,c=x??33;
            if(x)return a===11&&b===x&&c===x;
            return a===x&&b===22&&(x==null&&x!==$262.IsHTMLDDA?c===33:Object.is(c,x));}
        var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,42,'','text',Symbol(),{},0n,1n,$262.IsHTMLDDA];
        var good=true;for(var r=0;r<4;r++)for(var i=0;i<values.length;i++){
            if(Number.isNaN(values[i]))continue;good=observe(values[i])&&good;
        }
        function numeric(x){var n=x+0;return [n&&7,n||8,n??9].map(String).join(':');}
        function owners(){var a={value:41};if(a===a){var alias=(a=a);collectPredicateTest();return alias.value;}return 0;}
        good+'|'+numeric(NaN)+'|'+numeric(-0)+'|'+owners();
    "#,
        "true|NaN:8:NaN|0:8:0|41",
        true,
    );
}

#[test]
fn region_predicates_native_peeks_do_not_call_condition_helper() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert!(matches!(
        engine.eval(
            r#"
        function f(x,y){var a=x+1,b=y+1;return ((a===b)&&x)||y;}
        for(var i=0;i<150;i++)f(1,1);
    "#,
            false
        ),
        Ok(Completion::Value(_))
    ));
    let global = Value::Obj(engine.interp.global.clone());
    let function = engine.interp.get_member(&global, "f").ok().unwrap();
    let regions = EXECUTED_REGIONS.with(std::cell::Cell::get);
    let helpers = crate::bytecode::TEST_JIT_COND_HELPERS.with(std::cell::Cell::get);
    for (x, y, want) in [(1.0, 1.0, 1.0), (1.0, 2.0, 2.0), (0.0, 0.0, 0.0)] {
        let result = engine
            .interp
            .call(
                function.clone(),
                Value::Undefined,
                &[Value::Num(x), Value::Num(y)],
            )
            .ok()
            .unwrap();
        assert!(matches!(result,Value::Num(n) if n==want));
    }
    assert!(
        EXECUTED_REGIONS.with(std::cell::Cell::get) > regions,
        "actual native region proof"
    );
    assert_eq!(
        crate::bytecode::TEST_JIT_COND_HELPERS.with(std::cell::Cell::get),
        helpers,
        "known Boolean and ordinary numeric peeks must stay in native code"
    );
}

#[test]
fn region_predicates_boolean_numeric_consumers_and_join_owners() {
    check(
        r#"
        function consume(a,b,k){var x=a+0,y=b+0,p,q;
            p=(x===y);q=p;
            if(k)p=(x<y);else p=(x!==y);
            return [(x===y)|0,~(x===y),(x===y)<<3,(x===y)+2,
                (x!==y)*4,(x===y)>>>0,q===true,p|0].join(':');}
        var result;for(var r=0;r<5;r++)result=[consume(1,1,0),consume(1,2,1),consume(NaN,NaN,0)];
        result.join('|');
    "#,
        "1:-2:8:3:0:1:true:0|0:-1:0:2:4:0:false:1|0:-1:0:2:4:0:false:1",
        false,
    );
    // This consumer fixture need not contain a branch-only predicate. It must
    // nevertheless run an actual region, not merely rely on the reference tier.
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let before = EXECUTED_REGIONS.with(std::cell::Cell::get);
    assert!(
        matches!(engine.eval("function f(a,b){var x=a+0,y=b+0;return ((x===y)|0)+(~(x===y))+((x===y)<<3);}f(1,1)",false),
        Ok(Completion::Value(value)) if value=="7")
    );
    assert!(
        EXECUTED_REGIONS.with(std::cell::Cell::get) > before,
        "actual region consumer proof"
    );
}

#[test]
fn region_predicates_reused_chains_keep_committed_effects_and_tagged_local_writes() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        // A property chain is admitted from monomorphic bytecode feedback. Warm
        // those exact function chunks before requesting their first native entry.
        engine.set_tier(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        engine.set_tier_threshold(0);
        engine.interp.def_method(
            &engine.interp.global,
            "collectPredicateTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        fn run(engine: &mut Engine, source: &str) -> String {
            match engine.eval(source, false).expect("chain fixture parses") {
                Completion::Value(value) => value,
                Completion::Throw { name, message } => panic!("{name}: {message}"),
            }
        }
        run(
            &mut engine,
            r#"
            var spare={retained:1},token={},o={x:2,y:4},trace=[];
            function f(o,keep){var cached=keep;if(keep===null)return cached;
                if((cached=o.x+1)<9)return cached+o.y;return cached;}
            function writes(a,b){if((a[0]=a[0]+1)+b[0]<20)return a[0];return -1;}
            var a=[1],b=[2];for(var r=0;r<150;r++){f(o,spare);a[0]=1;writes(a,b);}
        "#,
        );
        engine.set_tier(tier);
        let before = REUSED_CHAINS.with(std::cell::Cell::get);
        assert_eq!(run(&mut engine, "f(o,spare)"), "7", "{tier:?}");
        if tier == Tier::Jit {
            assert!(
                REUSED_CHAINS.with(std::cell::Cell::get) > before,
                "the selected function must actually reuse a native chain"
            );
        }
        assert_eq!(
            run(
                &mut engine,
                r#"
            o.x={valueOf(){trace.push('C');collectPredicateTest();return 3;}};
            var first=f(o,spare);
            Object.defineProperty(o,'x',{configurable:true,get(){trace.push('G');
                return {valueOf(){trace.push('V');collectPredicateTest();return 5;}};}});
            Object.defineProperty(o,'y',{get(){trace.push('Y');return 4;}});
            var second=f(o,spare);
            Object.defineProperty(o,'x',{get(){trace.push('T');return {valueOf(){trace.push('X');throw token;}};}});
            var caught=false;try{f(o,spare);}catch(e){caught=e===token;}
            [first,second,caught,trace.join(''),spare.retained].join('|');
        "#
            ),
            "8|10|true|CGVYTX|1",
            "{tier:?}"
        );
        assert_eq!(
            run(
                &mut engine,
                r#"
            a[0]=2;var reads=0;
            Object.defineProperty(b,'0',{configurable:true,get(){reads++;collectPredicateTest();return 2;}});
            var value=writes(a,b);
            Object.defineProperty(b,'0',{get(){reads++;throw token;}});
            var caught=false;try{writes(a,b);}catch(e){caught=e===token;}
            [value,a[0],reads,caught].join('|');
        "#
            ),
            "3|4|2|true",
            "{tier:?}"
        );
    }
}
