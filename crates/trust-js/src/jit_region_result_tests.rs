//! ECMA-262 e28783d5: OrdinaryGet, Call, ToNumeric and left-to-right GetValue.
//! A checked result remains an owner until its post-effect Number guard succeeds.
use super::*;
use crate::{bytecode::Tier, value::Value, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("result fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn engine(tier: Tier, warmup: &str) -> Engine {
    let mut engine = Engine::new();
    // Feedback selects a capability, but every native hit still proves live
    // receiver/descriptor/chain state. Warm the exact chunks, never a duplicate.
    engine.set_tier(if tier == Tier::Jit {
        Tier::Bytecode
    } else {
        tier
    });
    engine.set_tier_threshold(0);
    engine.interp.def_method(
        &engine.interp.global,
        "collectResultTest",
        0,
        |interp, _, _| {
            interp.gc_collect();
            Ok(Value::Undefined)
        },
    );
    evaluate(&mut engine, warmup);
    engine.set_tier(tier);
    engine
}

#[test]
fn getter_cache_regions_keep_post_effect_guards_and_native_continuation() {
    let mut engine = engine(
        Tier::Jit,
        r#"
        var calls=0,mode=0,token={};
        var o={get value(){calls++;collectResultTest();
            if(mode===1)return {valueOf(){return 7}};
            if(mode===2)throw token;
            return 3;}};
        function getterRegion(o,n){var prefix=n+1;return (prefix+o.value)*2+1+o.value*2;}
        for(var k=0;k<40;k++)getterRegion(o,2);calls=0;
    "#,
    );
    let before = PROPERTY_PATHS.with(std::cell::Cell::get);
    assert_eq!(
        evaluate(
            &mut engine,
            r#"
        var a=getterRegion(o,2);mode=1;var b=getterRegion(o,2);mode=2;
        var caught=false;try{getterRegion(o,2)}catch(e){caught=e===token}
        [a,b,caught,calls].join('|');
    "#
        ),
        "19|35|true|5"
    );
    let after = PROPERTY_PATHS.with(std::cell::Cell::get);
    assert!(
        after[2] > before[2],
        "getters enter the checked property helper"
    );
    assert!(
        after[3] > before[3],
        "actual native region property continuation"
    );
    assert!(
        after[4] > before[4],
        "numeric getter results enter the numeric continuation"
    );
    assert!(
        after[5] > before[5],
        "non-numeric result exits after its one read"
    );
}

#[test]
fn region_results_preserve_inherited_negative_array_and_string_native_capabilities() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(
            tier,
            r#"
            var proto={value:7},child=Object.create(proto),absent=Object.create(null),array=[1,2,3];
            function inherited(o,n){var base=n+1;return base+o.value*2;}
            function missing(o,n){var base=n+1;if(o.missing===undefined)base+=2;return base+3;}
            function length(o,n){var base=n+1;return base+o.length*2;}
            function string(s,n){var base=n+1;return base+s.charCodeAt(0)*2;}
            for(var i=0;i<40;i++){inherited(child,2);missing(absent,2);length(array,2);string('A',2);}
        "#,
        );
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        // No JS join/array construction inside these measured intervals: a
        // native hit must belong to the specific tested property/method site.
        for (source, expected, event) in [
            ("inherited(child,2)", "17", 0),
            ("missing(absent,2)", "8", 1),
            ("length(array,2)", "9", 0),
            ("string('A',2)", "133", 0),
        ] {
            let site_before = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert_eq!(
                evaluate(&mut engine, source),
                expected,
                "{tier:?}: {source}"
            );
            if tier == Tier::Jit {
                let site_after = PROPERTY_PATHS.with(std::cell::Cell::get);
                assert!(
                    site_after[event] > site_before[event],
                    "actual native capability: {source}"
                );
            }
        }
        if tier == Tier::Jit {
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(
                after[3] > before[3],
                "property result continues inside an actual region"
            );
            assert!(
                after[4] > before[4],
                "checked result becomes an actual numeric temporary"
            );
        }
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            proto.value=9;Object.setPrototypeOf(child,{value:11});
            Object.setPrototypeOf(absent,{missing:1});array.push(4);
            [inherited(child,2),missing(absent,2),length(array,2)].join('|');
        "#
            ),
            "25|6|11",
            "{tier:?}"
        );
    }
}

#[test]
fn region_results_cold_feedback_keeps_baseline_native_property_continuation() {
    let mut engine = engine(Tier::Jit, "var cold={value:7};");
    let before = PROPERTY_PATHS.with(std::cell::Cell::get);
    assert_eq!(
        evaluate(
            &mut engine,
            r#"
        function coldRead(o,n){var base=n+1;return base+o.value*2;}
        var good=true;for(var r=0;r<6;r++)good=coldRead(cold,2)===17&&good;good;
    "#
        ),
        "true"
    );
    let after = PROPERTY_PATHS.with(std::cell::Cell::get);
    assert!(
        after[3] > before[3],
        "empty feedback must not guarantee an own-only region exit"
    );
    assert!(
        after[0] > before[0],
        "live filled IC still executes native baseline data probe"
    );
}

#[test]
fn region_results_warmed_own_reads_keep_borrowing_and_receiver_guard_reuse() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(
            tier,
            r#"
            var own={a:2,b:3};
            function sumOwn(o,n){var s=0;for(var i=0;i<n;i++){s+=o.a;s++;s+=o.b;}return s;}
            for(var r=0;r<40;r++)sumOwn(own,30);
        "#,
        );
        let before = BORROWED_READS.with(std::cell::Cell::get);
        assert_eq!(evaluate(&mut engine, "sumOwn(own,30)"), "180", "{tier:?}");
        if tier == Tier::Jit {
            let after = BORROWED_READS.with(std::cell::Cell::get);
            assert!(
                after.0 > before.0,
                "warmed ordinary-own data still borrows its owner"
            );
            assert!(after.1 > before.1, "same receiver proof is actually reused");
        }
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            delete own.a;Object.setPrototypeOf(own,{a:7});
            var inherited=sumOwn(own,30),gets=0;
            Object.defineProperty(own,'a',{configurable:true,get(){gets++;this.b=9;return 4;}});
            var accessor=sumOwn(own,30);
            [inherited,accessor,gets].join('|');
        "#
            ),
            "330|420|30",
            "{tier:?}"
        );
    }
}

#[test]
fn region_results_post_effect_guard_never_replays_getter_coercion_or_throw() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(
            tier,
            r#"
            var trace=[],token={},mode=0;
            var o={get x(){trace.push('X');collectResultTest();
                if(mode===0)return 2;
                if(mode===1)return {valueOf(){trace.push('C');collectResultTest();return 5;}};
                return {valueOf(){trace.push('T');throw token;}};},
                get y(){trace.push('Y');collectResultTest();return 4;}};
            function readBaselineShape(o,n){var prefix=n+1,first=o.x;return prefix+first+o.y;}
            function read(o,n){var prefix=n+1,first=o.x;return ((prefix+first+o.y)*2+1)/2-0.5;}
            for(var i=0;i<40;i++){readBaselineShape(o,2);read(o,2);}trace=[];
        "#,
        );
        // The short original shape is better served by existing numeric chains.
        // Keep its exact cold semantics separately; the longer continuation has
        // real unboxed work to amortize a region entry and must prove that path.
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var first=readBaselineShape(o,2);mode=1;var second=readBaselineShape(o,2);mode=2;
            var caught=false;try{readBaselineShape(o,2);}catch(e){caught=e===token;}
            [first,second,caught,trace.join('')].join('|');
        "#
            ),
            "9|12|true|XYXCYXT",
            "{tier:?}"
        );
        evaluate(&mut engine, "mode=0;trace=[];");
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var first=read(o,2);mode=1;var second=read(o,2);mode=2;
            var caught=false;try{read(o,2);}catch(e){caught=e===token;}
            [first,second,caught,trace.join('')].join('|');
        "#
            ),
            "9|12|true|XYXCYXT",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(
                after[4] > before[4],
                "successful checked-result numeric continuation"
            );
            assert!(
                after[5] > before[5],
                "actual type miss after the one committed read"
            );
            assert!(
                after[6] > before[6],
                "actual scalar prefix/local reuse across checked effects"
            );
        }
    }
}

#[test]
fn region_results_ssa_phi_demand_is_not_an_entry_type_proof() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(
            tier,
            r#"
            var trace=[],mode=0;
            var o={get a(){trace.push('A');return 3;},get b(){trace.push('B');return mode?7n:5;},
                get c(){trace.push('C');return 7;}};
            function chooseBaselineShape(o,which){var value;if(which)value=o.a;else value=o.b;return value*2+1;}
            function choose(o,which){var value;if(which)value=o.a;else value=o.b;return value*2+1+o.c*2;}
            for(var i=0;i<40;i++){chooseBaselineShape(o,true);chooseBaselineShape(o,false);choose(o,true);choose(o,false);}trace=[];
        "#,
        );
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var a=chooseBaselineShape(o,true),b=chooseBaselineShape(o,false);mode=1;var caught=false;
            try{chooseBaselineShape(o,false);}catch(e){caught=e instanceof TypeError;}
            [a,b,caught,trace.join('')].join('|');
        "#
            ),
            "7|11|true|ABB",
            "{tier:?}"
        );
        evaluate(&mut engine, "mode=0;trace=[];");
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var a=choose(o,true),b=choose(o,false);mode=1;var caught=false;
            try{choose(o,false);}catch(e){caught=e instanceof TypeError;}
            [a,b,caught,trace.join('')].join('|');
        "#
            ),
            "21|25|true|ACBCB",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(
                after[4] > before[4] && after[5] > before[5],
                "phi-driven live guard and exact miss"
            );
        }
    }
}

#[test]
fn region_results_calls_keep_scalar_prefix_and_last_owner_across_collection() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(
            tier,
            r#"
            var trace=[],mode=0,token={};
            function first(){try{trace.push('F');collectResultTest();return 3;}finally{}}
            function second(){try{trace.push('S');collectResultTest();
                return mode?{valueOf(){trace.push('C');collectResultTest();return 4;}}:4;}finally{}}
            function calls(n){var base=n+1;return (base+first())*second();}
            function owners(o,n){var base=n+1,child=o.child;return base+(o=null,collectResultTest(),child.value);}
            for(var i=0;i<40;i++){calls(2);owners({child:{value:6}},2);}trace=[];
        "#,
        );
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var a=calls(2);mode=1;var b=calls(2);
            [a,b,owners({child:{value:6}},2),trace.join('')].join('|');
        "#
            ),
            "24|24|9|FSFSC",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(
                after[4] > before[4] && after[5] > before[5],
                "call result guard/miss both execute"
            );
            assert!(
                after[6] > before[6],
                "live scalar survives the later author call"
            );
        }
    }
}

#[test]
fn region_results_borrowed_osr_resumes_after_exactly_one_effect() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier, "");
        engine.set_tier_threshold(32);
        let entries = crate::jit::TEST_OSR_ENTRIES.with(std::cell::Cell::get);
        let before = PROPERTY_PATHS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var reads=0,conversions=0,cleaned=0,k=0,total=0;
            var source={get value(){reads++;
                if(k===1500)return {valueOf(){conversions++;collectResultTest();return 2;}};
                return 1;}};
            try{for(;k<2000;k++)total+=((source.value*2+1)*3+4);}
            finally{cleaned++;}
            [total,reads,conversions,cleaned,k].join('|');
        "#
            ),
            "26006|2000|1|1|2000",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert!(
                crate::jit::TEST_OSR_ENTRIES.with(std::cell::Cell::get) > entries,
                "must execute the borrowed native entry, not only normal-call code"
            );
            let after = PROPERTY_PATHS.with(std::cell::Cell::get);
            assert!(
                after[4] > before[4],
                "borrowed entry executes numeric result continuation"
            );
            assert!(
                after[5] > before[5],
                "borrowed entry executes the post-get type miss"
            );
        }
    }
}

#[test]
fn region_results_profitability_rejection_publishes_no_baseline_targets() {
    let mut a = asm::Asm::new();
    let baseline = a.new_label();
    a.bind(baseline);
    a.movz(0, 17, 0);
    let checkpoint = a.checkpoint();
    let plain = a.new_label();
    let bailout = a.new_label();
    a.b(bailout);
    a.bind(bailout);
    a.b(baseline);
    let emission = Emission {
        plain,
        targets: vec![1, 2],
        saved_work: 0,
        entry_work: 1,
        baseline_bytes: 128,
        bytes: 8,
        acyclic: true,
    };
    let mut targeted = [true, false, false];
    assert!(!emission.profitable());
    if emission.profitable() {
        emission.publish(&mut targeted);
    } else {
        a.rewind(checkpoint);
    }
    assert_eq!(targeted, [true, false, false]);
    assert_eq!(a.checkpoint(), checkpoint);
    a.ret();
    assert_eq!(a.finish().len(), 2);
    let oversized = Emission {
        plain,
        targets: vec![1],
        saved_work: 100,
        entry_work: 1,
        baseline_bytes: 64,
        bytes: 64 * 1024,
        acyclic: false,
    };
    assert!(
        !oversized.profitable(),
        "hot loops also obey baseline-relative expansion limits"
    );
}
