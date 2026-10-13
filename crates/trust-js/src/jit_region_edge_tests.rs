//! ECMA-262 e28783d5 If/Logical/Conditional/ForBody/Try evaluation.
//! Removing pure control-flow stubs must not change owners, polls, effects or values.
use super::*;
use crate::{bytecode::Tier, value::Value, Completion, Engine};

#[test]
fn cfg_edges_route_forward_internal_but_defer_every_poll_and_exit() {
    let mut a = asm::Asm::new();
    let labels = [(0, a.new_label()), (10, a.new_label()), (30, a.new_label())]
        .into_iter()
        .collect();
    let mut edges = CanonicalEdges::new(&labels);
    assert_eq!(edges.target(&mut a, 2, 10), labels[&10]);
    assert_eq!(edges.target(&mut a, 10, 30), labels[&30]);
    assert!(edges.deferred.is_empty());
    for (from, target) in [(10, 10), (30, 0), (30, 10), (2, 20), (30, 5)] {
        let label = edges.target(&mut a, from, target);
        assert_eq!(edges.deferred.last(), Some(&(label, from, target)));
        assert!(!labels.values().any(|&known| known == label));
    }
    assert_eq!(edges.deferred.len(), 5);
    assert!(
        a.buf.is_empty(),
        "routing must not clobber live condition flags"
    );
}

#[test]
fn cfg_edges_fallthrough_requires_forward_internal_physical_adjacency() {
    for (from, target, next, emitted, deferred) in [
        (2, 10, Some(10), 0, 0),
        (2, 10, Some(30), 1, 0),
        (2, 10, None, 1, 0),
        (10, 10, Some(10), 1, 1),
        (30, 10, Some(10), 1, 1),
        (2, 20, Some(20), 1, 1),
        (30, 5, Some(5), 1, 1),
    ] {
        let mut a = asm::Asm::new();
        let labels = [(10, a.new_label()), (30, a.new_label())]
            .into_iter()
            .collect();
        let mut edges = CanonicalEdges::new(&labels);
        edges.transfer(&mut a, from, target, next);
        assert_eq!(a.buf.len(), emitted, "{from}->{target}, next={next:?}");
        assert_eq!(edges.deferred.len(), deferred);
    }
}

#[test]
fn cfg_edges_native_diamond_branches_directly_and_falls_through_at_join() {
    let mut a = asm::Asm::new();
    let labels = [(4, a.new_label()), (8, a.new_label())]
        .into_iter()
        .collect();
    let mut edges = CanonicalEdges::new(&labels);
    a.cmp_imm_w(0, 0);
    let right = edges.target(&mut a, 0, 4);
    a.b_cond(C_EQ, right);
    a.movz(0, 11, 0);
    edges.transfer(&mut a, 2, 8, Some(4));
    a.bind(labels[&4]);
    a.movz(0, 22, 0);
    edges.transfer(&mut a, 4, 8, Some(8));
    a.bind(labels[&8]);
    a.ret();
    assert!(edges.deferred.is_empty());
    let words = a.finish();
    assert_eq!(words.len(), 6);
    assert_eq!(
        words[1], 0x5400_0060,
        "BEQ directly to right block at word4"
    );
    assert_eq!(words[3], 0x1400_0002, "B directly to join at word5");
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let code = ExecutableBuffer::from_bytes(&bytes).unwrap();
    let run: unsafe extern "C" fn(u32) -> u32 = unsafe { std::mem::transmute(code.as_ptr()) };
    for (value, expected) in [(0, 22), (1, 11), (u32::MAX, 11)] {
        assert_eq!(unsafe { run(value) }, expected);
    }
}

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("edge fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
        .interp
        .def_method(&engine.interp.global, "collectEdgesTest", 0, |i, _, _| {
            i.gc_collect();
            Ok(Value::Undefined)
        });
    engine
}

#[test]
fn cfg_edges_actual_branching_region_keeps_backedge_poll_and_deadline() {
    let mut engine = engine(Tier::Bytecode);
    evaluate(&mut engine, "function cfgEdges(n){var sum=0;for(var i=0;i<n;i++){if((i&3)===0)sum+=i;else if((i&3)===1)sum-=i;else sum+=2;}return sum;}cfgEdges(64);");
    let global = Value::Obj(engine.interp.global.clone());
    let function = engine.interp.get_member(&global, "cfgEdges").ok().unwrap();
    engine.set_tier(Tier::Jit);
    let before = EXECUTED_REGIONS.with(std::cell::Cell::get);
    let value = engine
        .interp
        .call(function, Value::Undefined, &[Value::Num(40000.0)])
        .ok()
        .unwrap();
    assert!(matches!(value, Value::Num(n) if n == 30000.0));
    assert!(
        EXECUTED_REGIONS.with(std::cell::Cell::get) > before,
        "direct host call must execute the tested function's optimized region"
    );
    engine.interrupt_handle().set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(20),
    ));
    let outcome = engine.eval_interruptible("cfgEdges(1e12)", false).unwrap();
    assert!(matches!(
        outcome,
        crate::ExecutionOutcome::Interrupted {
            reason: crate::InterruptReason::DeadlineExceeded
        }
    ));
    engine.interrupt_handle().set_deadline(None);
    assert_eq!(evaluate(&mut engine, "cfgEdges(4)"), "3");
}

#[test]
fn cfg_edges_short_circuit_joins_keep_exact_values_in_every_tier() {
    let source = r#"
        function route(x){return [x&&'R',x||'R',x??'N',x?'T':'F'];}
        var values=[undefined,null,false,true,0,-0,NaN,0n,1n,'','x',Symbol(),{},$262.IsHTMLDDA];
        var truths=[false,false,false,true,false,false,false,false,true,false,true,true,true,false];
        var good=true;
        for(var round=0;round<5;round++)for(var j=0;j<values.length;j++){
            var v=values[j],t=truths[j],out=route(v);
            good=Object.is(out[0],t?'R':v)&&Object.is(out[1],t?v:'R')&&
                Object.is(out[2],j<2?'N':v)&&out[3]===(t?'T':'F')&&good;
        }good;
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        assert_eq!(evaluate(&mut engine(tier), source), "true", "{tier:?}");
    }
}

#[test]
fn cfg_edges_joined_owners_getters_gc_and_finally_preserve_effect_order() {
    let source = r#"
        var trace=[],token={};
        var o={get left(){trace.push('L');return {value:2};},
               get right(){trace.push('R');return {value:4};}};
        function joined(o,which){var prior=o.left;try{
            var chosen=which?prior:o.right;
            collectEdgesTest();return chosen.value;
        }finally{trace.push('F');}}
        var a=joined(o,true),b=joined(o,false);
        Object.defineProperty(o,'right',{get(){trace.push('T');collectEdgesTest();throw token;}});
        var caught=false;try{joined(o,false);}catch(e){caught=e===token;}
        a+':'+b+':'+caught+'|'+trace.join('');
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        assert_eq!(
            evaluate(&mut engine(tier), source),
            "2:4:true|LFLRFLTF",
            "{tier:?}"
        );
    }
}

#[test]
fn cfg_edges_nested_continue_break_and_finally_preserve_completions() {
    let source = r#"
        var trace=[];
        function loops(){var sum=0;outer:for(var i=0;i<4;i++){
            for(var j=0;j<4;j++){try{
                if(j===1)continue;if(i===2)break outer;sum+=i+j;
            }finally{trace.push(i+':'+j);}}
        }return sum;}
        loops()+'|'+trace.join(',');
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        assert_eq!(
            evaluate(&mut engine(tier), source),
            "13|0:0,0:1,0:2,0:3,1:0,1:1,1:2,1:3,2:0",
            "{tier:?}"
        );
    }
}

#[test]
fn cfg_edges_inline_success_still_carries_virtual_scalar_owners() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        evaluate(
            &mut engine,
            r#"
            var receiver={value:3};
            function innerEdge(x){return x*2+1;}
            function outerEdge(o,n){var sum=0;for(var i=0;i<n;i++){
                var keep=i*2+1;sum+=(i&1)?keep+innerEdge(o.value):keep+innerEdge(o.value)*2;
            }return sum;}
            for(var r=0;r<40;r++)outerEdge(receiver,6);
        "#,
        );
        engine.set_tier(tier);
        let before = GUARDED_REGIONS.with(std::cell::Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var good=true;for(var r=0;r<180;r++)good=outerEdge(receiver,6)===99&&good;
            var reads=0,coercions=0;
            Object.defineProperty(receiver,'value',{get(){reads++;collectEdgesTest();
                return {valueOf(){coercions++;collectEdgesTest();return 4;}};}});
            [good,outerEdge(receiver,6),reads,coercions].join('|');
        "#
            ),
            "true|117|6|6",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert!(
                GUARDED_REGIONS.with(std::cell::Cell::get)[1] > before[1],
                "actual InlineGuard success must retain carried state"
            );
        }
    }
}
