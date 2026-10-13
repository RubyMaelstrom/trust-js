//! Frame-independent helpers operate only on canonical operand owners. Calls made
//! from abstract operations and `in` may reenter through proxies or coercion, but
//! cannot inspect a caller's private register homes.
use crate::{bytecode::Tier, value::Value, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("effect fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn prepared(tier: Tier, source: &str) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(if tier == Tier::Jit {
        Tier::Bytecode
    } else {
        tier
    });
    engine.set_tier_threshold(0);
    engine.interp.def_method(
        &engine.interp.global,
        "collectEffectTest",
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
fn in_and_abstract_sites_run_with_numeric_loop_state_live() {
    let setup = r#"
        var source=[3,4],trace=[],token={},fail=false,calls=0,collectNow=false;
        var defineHandler={defineProperty(target,key,descriptor){
            trace.push(key);
            if(key==='0'){
                source[1]=11;
                if(collectNow)collectEffectTest();
            }
            if(fail&&key==='1')throw token;
            return Reflect.defineProperty(target,key,descriptor);
        }};
        function Species(){return new Proxy([],defineHandler);}
        source.constructor={[Symbol.species]:Species};
        function makeValue(value,index){calls++;return {value:value,index:index};}
        function mapResult(){return source.map(makeValue);}

        var hasTarget={},hasCalls=0;
        var hasProxy=new Proxy(hasTarget,{has(target,key){
            hasCalls++;
            target.touched=true;
            if(collectNow)collectEffectTest();
            return key==='present';
        }});
        var hasBox={current:hasProxy};
        function hasLoop(box,key,count){
            var sum=0;
            for(var index=0;index<count;index++){
                if(key in box.current)sum+=index;
            }
            return sum;
        }
    "#;

    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = prepared(tier, setup);
        if tier == Tier::Jit {
            evaluate(
                &mut engine,
                "for(var warm=0;warm<16;warm++){mapResult();hasLoop(hasBox,'present',4);}",
            );
        }

        let regions = super::EXECUTED_REGIONS.with(std::cell::Cell::get);
        let abstract_effects = super::independent_effect_live_home_paths();
        let define_helpers =
            crate::bytecode::TEST_JIT_DEFINE_ELEMENT_HELPERS.with(std::cell::Cell::get);
        evaluate(&mut engine, "source[1]=4;trace=[];calls=0;collectNow=true;");
        assert_eq!(
            evaluate(
                &mut engine,
                "var result=mapResult();collectNow=false;\
                 result[0].value+','+result[1].value+','+result[1].index+'|'+trace.join(',')+'|'+calls",
            ),
            "3,11,1|0,1|2",
            "{tier:?}: Proxy species reentry may mutate the next source element and collect",
        );
        if tier == Tier::Jit {
            assert!(
                super::EXECUTED_REGIONS.with(std::cell::Cell::get) > regions,
                "map's native loop region executes"
            );
            assert!(
                crate::bytecode::TEST_JIT_DEFINE_ELEMENT_HELPERS.with(std::cell::Cell::get)
                    > define_helpers,
                "Proxy species takes the checked CreateDataPropertyOrThrow path inside the region"
            );
            assert!(
                super::independent_effect_live_home_paths()[1] > abstract_effects[1],
                "the native CreateDataPropertyOrThrow site keeps a numeric loop home resident"
            );
        }

        assert_eq!(
            evaluate(
                &mut engine,
                "source[1]=4;trace=[];calls=0;fail=true;var caught=false;\
                 try{mapResult()}catch(error){caught=error===token;}\
                 fail=false;caught+'|'+trace.join(',')+'|'+calls",
            ),
            "true|0,1|2",
            "{tier:?}: an abrupt Proxy definition exits after the failing element once",
        );

        let in_helpers = crate::bytecode::TEST_JIT_IN_HELPERS.with(std::cell::Cell::get);
        let in_effects = super::independent_effect_live_home_paths();
        assert_eq!(
            evaluate(
                &mut engine,
                "collectNow=true;hasCalls=0;var present=hasLoop(hasBox,'present',4);\
                 collectNow=false;present+'|'+hasCalls+'|'+hasTarget.touched",
            ),
            "6|4|true",
            "{tier:?}: Proxy has traps reenter and observe each in-operation",
        );
        if tier == Tier::Jit {
            assert!(
                crate::bytecode::TEST_JIT_IN_HELPERS.with(std::cell::Cell::get) > in_helpers,
                "the Proxy `in` fallback executes in the native loop"
            );
            assert!(
                super::independent_effect_live_home_paths()[0] > in_effects[0],
                "the native `in` site keeps the numeric accumulator/index home resident"
            );
        }
    }
}
