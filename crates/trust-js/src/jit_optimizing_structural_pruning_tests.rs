//! Complete zero-reuse spans must not become gratuitous specialization exits.
use super::super::{object_facts, specialization, value_facts};
use super::*;
use crate::{
    bytecode::{self, Tier},
    value::Callable,
    Completion, Engine,
};
use std::rc::Rc;

struct Enabled(Option<bool>);
impl Enabled {
    fn new() -> Self {
        Self(generic_own::TEST_ENABLED.with(|v| v.replace(Some(true))))
    }
}
impl Drop for Enabled {
    fn drop(&mut self) {
        generic_own::TEST_ENABLED.with(|v| v.set(self.0));
    }
}
fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(v) => v,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}
fn warm(
    source: &str,
) -> (
    Engine,
    Rc<crate::ast::Function>,
    Rc<Chunk>,
    Vec<(u32, IcState)>,
) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, "subject(node,other)");
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("function")
    };
    let function = user.func.clone();
    drop(object);
    let chunk = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    let observed = observations(&chunk);
    for _ in 0..4 {
        observe(&chunk, &observed)
    }
    let selected = selected(&observed);
    (engine, function, chunk, selected)
}
fn analyses(chunk: &Chunk) -> (Cfg, stack::StackPlan, frame::FramePlan) {
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let frame = frame::FramePlan::build(chunk, &cfg, true).unwrap();
    (cfg, stack, frame)
}
fn flags(site: generic_own::Site) -> (bool, bool, bool, bool) {
    (
        site.removed.ordinary,
        site.removed.shape,
        site.removed.bounds,
        site.removed.data,
    )
}
const SOURCE:&str="function subject(node,other){var sum=other.x+other.y;var value=node.tag;return sum+value;}var node={tag:10},other={x:3,y:4};";

#[test]
fn zero_span_is_pruned_and_useful_first_producer_and_every_elision_are_kept() {
    let _enabled = Enabled::new();
    let (_, _, chunk, selected) = warm(SOURCE);
    let (cfg, stack, frame) = analyses(&chunk);
    let original =
        generic_own::Plan::build_hot_selection(&chunk, &cfg, &frame, &stack, &selected).unwrap();
    assert_eq!(selected.len(), 3);
    assert!(
        !original
            .read(selected[2].0 as usize)
            .unwrap()
            .reuse_span_profitable
    );
    assert_eq!(
        flags(original.read(selected[0].0 as usize).unwrap()),
        (false, false, false, false)
    );
    assert!(
        original
            .read(selected[0].0 as usize)
            .unwrap()
            .reuse_span_profitable,
        "zero-credit producer supports later read"
    );
    let plan = Plan::build(&chunk, &cfg, &stack, &selected).unwrap();
    assert_eq!(plan.selected.len(), 2);
    assert!(plan.minimum_tests > 0);
    assert!(same(plan.selected[0].1, selected[0].1));
    assert_eq!(plan.selected[0].0, selected[0].0);
    let rebuilt =
        generic_own::Plan::build_hot_selection(&chunk, &cfg, &frame, &stack, &plan.selected)
            .unwrap();
    for &(pc, _) in &plan.selected {
        assert_eq!(
            flags(original.read(pc as usize).unwrap()),
            flags(rebuilt.read(pc as usize).unwrap())
        );
    }
    let mut stale = selected.clone();
    stale[2].1.recv_shape = stale[2].1.recv_shape.wrapping_add(1);
    assert!(
        Plan::build(&chunk, &cfg, &stack, &stale).is_none(),
        "unprofitable does not excuse stale snapshot"
    );
}

#[test]
fn local_copy_checked_effect_and_cfg_boundaries_end_metadata_spans() {
    let _enabled = Enabled::new();
    for source in [
        "function subject(node,other){var a=node.tag;var b=other.x;return other.y+a+b;}var node={tag:10},other={x:3,y:4};",
        "function subject(node,other){var a=node.tag;other.x;return other.y+a;}var node={tag:10},other={x:3,y:4};",
        "function subject(node,other){if(node.tag){return other.x;}return other.y;}var node={tag:10},other={x:3,y:4};",
        "function subject(node,other){var a=node.tag;try{other.x;}finally{other.y;}return a;}var node={tag:10},other={x:3,y:4};",
        "function subject(node,other){for(var i=0;i<2;i++){node.tag;}return other.x;}var node={tag:10},other={x:3,y:4};",
        "function subject(node,other){var a=node.tag;other[0];return other.y+a;}var node={tag:10},other={0:3,x:3,y:4};",
        "function subject(node,other){var a=node.tag;observe();return other.y+a;}function observe(){}var node={tag:10},other={x:3,y:4};",
    ] {
        let(_,_,chunk,selected)=warm(source);let(cfg,stack,frame)=analyses(&chunk);
        let proof=generic_own::Plan::build_hot_selection(&chunk,&cfg,&frame,&stack,&selected).unwrap();
        for &(pc,_) in &selected {let site=proof.read(pc as usize).unwrap();assert!(!site.reuse_span_profitable,"{source} PC{pc}");assert_eq!(flags(site),(false,false,false,false));}
        assert!(Plan::build(&chunk,&cfg,&stack,&selected).is_none());
    }
}

#[test]
fn profitable_overwrite_spans_keep_writable_descriptor_guards_and_read_implications() {
    let _enabled = Enabled::new();
    for (source,write_first,expected_sites) in [
        ("function subject(node,other){other.x=other.x;return node.tag;}var node={tag:10},other={x:3,y:4};",false,3),
        ("function subject(node,other){other.x=7;var value=other.x;return node.tag+value;}var node={tag:10},other={x:3,y:4};",true,2),
    ] {
        let(_,_,chunk,selected)=warm(source);let(cfg,stack,frame)=analyses(&chunk);
        let original=generic_own::Plan::build_hot_selection(&chunk,&cfg,&frame,&stack,&selected).unwrap();
        let plan=Plan::build(&chunk,&cfg,&stack,&selected).unwrap();assert_eq!(plan.selected.len(),expected_sites);
        let last=plan.selected[1].0 as usize;
        if write_first {assert!(original.read(last).unwrap().removed.data,"writable producer proves readable data");}
        else {assert!(!original.write(last).unwrap().removed.data,"readable producer never proves writable");}
        let rebuilt=generic_own::Plan::build_hot_selection(&chunk,&cfg,&frame,&stack,&plan.selected).unwrap();
        for &(pc,_)in &plan.selected {let old=original.read(pc as usize).or_else(||original.write(pc as usize)).unwrap();let new=rebuilt.read(pc as usize).or_else(||rebuilt.write(pc as usize)).unwrap();assert_eq!(flags(old),flags(new));}
    }
}

fn native(source: &str) -> (Engine, Rc<Chunk>, Rc<crate::jit::JitCode>) {
    let (mut engine, function, baseline, selected) = warm(source);
    let (cfg, stack, _) = analyses(&baseline);
    let plan = Plan::build(&baseline, &cfg, &stack, &selected).unwrap();
    assert!(plan.selected.len() < selected.len());
    let chunk = bytecode::compile_for_optimizer_with_structural(
        &function,
        &baseline,
        &[],
        &[],
        &plan.selected,
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(super::super::compile_with_deopt(&chunk, &values, &layout, None).unwrap());
    assert!(code.pc_offsets.is_empty());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    engine.set_tier(Tier::Jit);
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|v| v.borrow_mut().clear());
    (engine, chunk, code)
}
#[test]
fn isolated_polymorphic_read_uses_complete_get_and_never_retires_the_native_body() {
    let _enabled = Enabled::new();
    let (mut engine, chunk, code) = native(SOURCE);
    super::super::property::TEST_READ_HELPERS.with(|v| v.set(0));
    for (setup, expected, reads) in [
        ("node=[];", "NaN", "0"),
        (
            "node={};Object.defineProperty(node,'tag',{get(){reads++;return 20;}});",
            "27",
            "1",
        ),
        (
            "node=new Proxy({}, {get(target,key){reads++;return 30;}});",
            "37",
            "1",
        ),
        ("node={tag:10};", "17", "0"),
    ] {
        eval(&mut engine, "var reads=0;");
        eval(&mut engine, &format!("{setup}void 0;"));
        code.residency.referenced.set(0);
        assert_eq!(eval(&mut engine, "subject(node,other)"), expected);
        assert_eq!(
            eval(&mut engine, "reads"),
            reads,
            "exact getter/proxy execution"
        );
        assert_eq!(
            code.residency.referenced.get(),
            1,
            "this invocation entered native subject"
        );
        assert_eq!(chunk.optimizing_misses.get(), 0);
        assert_eq!(code.residency.active.get(), 0);
    }
    assert!(super::super::property::TEST_READ_HELPERS.with(Cell::get) >= 3);
    assert_eq!(code.residency.referenced.get(), 1);
    bytecode::native_deopt::ENTRIES.with(|v| assert!(v.borrow().is_empty()));
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn getter_rejoin_gc_mutation_and_throw_preserve_exact_effect_and_cleanup_boundaries() {
    fn collect(
        i: &mut crate::interpreter::Interp,
        _: crate::value::Value,
        _: &[crate::value::Value],
    ) -> Result<crate::Value, crate::Value> {
        i.gc_collect();
        Ok(crate::value::Value::Undefined)
    }
    let _enabled = Enabled::new();
    let source="function subject(node,other){var initial=other.x+other.y;var value=node.tag;if(initial!==7)throw 'initial';return value+(other.x+other.y);}var node={tag:10},other={x:3,y:4},trace='';";
    let (mut engine, chunk, code) = native(source);
    let global = engine.interp.global.clone();
    engine.interp.def_method(&global, "observeGC", 0, collect);
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine,"Object.defineProperty(node,'tag',{get(){trace+='g';observeGC();Object.defineProperty(other,'x',{get(){trace+='x';return 20;},configurable:true});return 10;},configurable:true});subject(node,other)+':'+trace"),"34:gx");
    assert_eq!(
        chunk.optimizing_misses.get(),
        1,
        "later useful data guard catches mutated descriptor"
    );
    assert_eq!(
        code.residency.referenced.get(),
        1,
        "mutating getter invocation entered native"
    );
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine,"Object.defineProperty(other,'x',{value:3,writable:true,configurable:true});trace='';Object.defineProperty(node,'tag',{get(){trace+='g';observeGC();throw Error('tag');},configurable:true});try{subject(node,other);}catch(e){trace+='c';var stackOK=e.stack.indexOf('subject')>=0;}finally{trace+='f';}trace+':'+stackOK"),"gcf:true");
    assert_eq!(
        chunk.optimizing_misses.get(),
        1,
        "throwing isolated getter uses exact checked operation, not specialization bailout"
    );
    assert_eq!(
        code.residency.referenced.get(),
        1,
        "throwing getter invocation entered native"
    );
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn isolated_write_rejoins_once_through_setter_proxy_readonly_and_throw_with_live_owners() {
    fn collect(
        i: &mut crate::interpreter::Interp,
        _: crate::value::Value,
        _: &[crate::value::Value],
    ) -> Result<crate::Value, crate::Value> {
        i.gc_collect();
        Ok(crate::value::Value::Undefined)
    }
    let _enabled = Enabled::new();
    let source = "function subject(node,other){'use strict';var initial=other.x+other.y;node.tag=makeValue();var barrier=initial;if(barrier!==7)throw 'initial';return barrier+(other.x+other.y);}var node={tag:{old:true}},other={x:3,y:4},held={keep:1,child:{alive:1}},trace='',captured=null;function makeValue(){trace+='r';return held;}";
    let (mut engine, function, baseline, selected) = warm(source);
    let (cfg, stack, frame) = analyses(&baseline);
    let write = selected
        .iter()
        .find(|(pc, _)| matches!(baseline.jit_ops()[*pc as usize], Op::SetPropLocalDrop(..)))
        .unwrap()
        .0;
    let before =
        generic_own::Plan::build_hot_selection(&baseline, &cfg, &frame, &stack, &selected).unwrap();
    assert!(!before.write(write as usize).unwrap().reuse_span_profitable);
    let plan = Plan::build(&baseline, &cfg, &stack, &selected).unwrap();
    assert!(!plan.selected.iter().any(|(pc, _)| *pc == write));
    assert!(plan
        .selected
        .iter()
        .all(|(pc, _)| before.read(*pc as usize).is_some()));
    let chunk = bytecode::compile_for_optimizer_with_structural(
        &function,
        &baseline,
        &[],
        &[],
        &plan.selected,
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(super::super::compile_with_deopt(&chunk, &values, &layout, None).unwrap());
    assert!(code.pc_offsets.is_empty());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    engine.set_tier(Tier::Jit);
    let global = engine.interp.global.clone();
    engine.interp.def_method(&global, "observeGC", 0, collect);
    eval(
        &mut engine,
        "makeValue=function(){trace+='r';observeGC();return held;};void 0;",
    );
    bytecode::native_deopt::ENTRIES.with(|v| v.borrow_mut().clear());
    // Replacement consumes the field's sole old owner and moves the real RHS
    // owner, even though this now-unselected write follows the checked path.
    eval(&mut engine, "node={tag:{old:true}};trace='';void 0;");
    let old = {
        let env = engine.interp.global_env.clone();
        let node = engine.interp.get_var("node", &env).ok().unwrap();
        let node = node.as_obj().unwrap().borrow();
        let value = node.props.property_at(0).unwrap().value();
        Rc::downgrade(value.as_obj().unwrap())
    };
    assert_eq!(old.strong_count(), 1);
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine, "subject(node,other)+':'+trace"), "14:r");
    assert_eq!(code.residency.referenced.get(), 1);
    assert!(old.upgrade().is_none());
    assert_eq!(
        eval(&mut engine, "node.tag===held && node.tag.child.alive===1"),
        "true"
    );
    for (setup,call,expected) in [
        ("node={};Object.defineProperty(node,'tag',{set(value){trace+='s';captured=value;observeGC();other.x=20;},configurable:true});", "subject(node,other)+':'+trace+':'+(captured===held && captured.child.alive===1)", "31:rs:true"),
        ("node=new Proxy({tag:{old:true}},{set(target,key,value){trace+='p';observeGC();return Reflect.set(target,key,value,target);}});", "subject(node,other)+':'+trace+':'+(node.tag===held && node.tag.child.alive===1)", "14:rp:true"),
        ("node={};Object.defineProperty(node,'tag',{value:held,writable:false});", "try{subject(node,other);}catch(e){trace+='c';var typeOK=e instanceof TypeError;}finally{trace+='f';}trace+':'+typeOK+':'+(node.tag===held)", "rcf:true:true"),
        ("node={};Object.defineProperty(node,'tag',{set(value){trace+='s';captured=value;observeGC();throw Error('setter');}});", "try{subject(node,other);}catch(e){trace+='c';var stackOK=e.stack.indexOf('subject')>=0;}finally{trace+='f';}trace+':'+stackOK+':'+(captured===held && captured.child.alive===1)", "rscf:true:true"),
    ] {
        eval(&mut engine,&format!("other.x=3;trace='';captured=null;{setup}void 0;"));
        code.residency.referenced.set(0);
        assert_eq!(eval(&mut engine,call),expected);
        assert_eq!(code.residency.referenced.get(),1,"each call entered native subject");
        assert_eq!(code.residency.active.get(),0);
        assert_eq!(chunk.optimizing_misses.get(),0,"ordinary isolated Set must not retire the body");
        bytecode::native_deopt::ENTRIES.with(|v|assert!(v.borrow().is_empty()));
        assert!(engine.interp.fn_frames.is_empty());
    }
    let assigned = {
        let env = engine.interp.global_env.clone();
        let held = engine.interp.get_var("held", &env).ok().unwrap();
        Rc::downgrade(held.as_obj().unwrap())
    };
    eval(&mut engine, "node={};captured=null;held=null;void 0;");
    engine.interp.gc_collect();
    assert!(
        assigned.upgrade().is_none(),
        "assigned object and child retain no stray native owner"
    );
}

struct ObjectEnabled(Option<bool>);
impl ObjectEnabled {
    fn new() -> Self {
        Self(object_facts::TEST_ENABLED.with(|value| value.replace(Some(true))))
    }
}
impl Drop for ObjectEnabled {
    fn drop(&mut self) {
        object_facts::TEST_ENABLED.with(|value| value.set(self.0));
    }
}
const OBJECT_SOURCE: &str = "function cold(){throw Error('cold');}function finish(other){return other.d;}function subject(node,other){'use strict';var value=node.tag;if(value===null)return cold();other.a=other.x;other.b=other.a;other.c=other.b;other.d=other.c;var barrier=other.d;if(barrier!=='v')throw Error('barrier');var count=node.num;return finish(other)+(count|0);}var node={tag:{keep:1},num:2},other={x:'v',a:0,b:0,c:0,d:0};";

#[test]
fn required_object_producer_survives_pruning_and_recomposes_exact_branch_proofs() {
    let _enabled = Enabled::new();
    let _objects = ObjectEnabled::new();
    let (mut engine, function, baseline, selected) = warm(OBJECT_SOURCE);
    let (cfg, stack, frame) = analyses(&baseline);
    let at_name = |name: &str| {
        selected.iter().find(|(pc,_)| matches!(baseline.jit_ops()[*pc as usize],Op::GetPropLocal(_,key,_) if baseline.jit_name(key)==name)).copied().unwrap()
    };
    let tag = at_name("tag");
    let numeric = at_name("num");
    let objects = [tag];
    let numeric_properties = [(numeric.0, value_facts::Types::NUMBER)];
    let object = object_facts::Plan::build(
        &baseline,
        &cfg,
        &stack,
        &objects,
        &selected,
        &[],
        &numeric_properties,
    )
    .unwrap();
    let original =
        generic_own::Plan::build_hot_selection(&baseline, &cfg, &frame, &stack, &selected).unwrap();
    assert!(!original.read(tag.0 as usize).unwrap().reuse_span_profitable);
    assert!(
        !original
            .read(numeric.0 as usize)
            .unwrap()
            .reuse_span_profitable
    );
    let plan = Plan::with_objects(&baseline, &cfg, &stack, &selected, Some(&object)).unwrap();
    assert!(
        plan.selected
            .iter()
            .any(|&(pc, state)| pc == tag.0 && same(state, tag.1)),
        "the same Object-result producer state is required by its comparison/branch plan"
    );
    assert!(!plan.selected.iter().any(|&(pc,_)| pc==numeric.0),"Number properties in the same Object Plan do not require retaining isolated structural snapshots");
    assert_eq!(
        plan.per_pc[tag.0 as usize],
        Some(0),
        "retaining the required producer invents no metadata savings"
    );
    let mut stale = selected.clone();
    stale
        .iter_mut()
        .find(|(pc, _)| *pc == tag.0)
        .unwrap()
        .1
        .recv_shape = tag.1.recv_shape.wrapping_add(1);
    assert!(
        Plan::with_objects(&baseline, &cfg, &stack, &stale, Some(&object)).is_none(),
        "required Object dependency cannot excuse an invalid original scalar snapshot"
    );
    let without_object: Vec<_> = plan
        .selected
        .iter()
        .copied()
        .filter(|(pc, _)| *pc != tag.0)
        .collect();
    assert!(
        object_facts::Plan::build(
            &baseline,
            &cfg,
            &stack,
            &objects,
            &without_object,
            &[],
            &numeric_properties
        )
        .is_none(),
        "historical omission really loses the independent Object proof prerequisite"
    );
    let proof =
        generic_own::Plan::build_hot_selection(&baseline, &cfg, &frame, &stack, &plan.selected)
            .unwrap();
    for &(pc, _) in &plan.selected {
        let before = original
            .read(pc as usize)
            .or_else(|| original.write(pc as usize))
            .unwrap();
        let after = proof
            .read(pc as usize)
            .or_else(|| proof.write(pc as usize))
            .unwrap();
        assert_eq!(flags(before), flags(after));
    }
    let candidate = bytecode::compile_for_optimizer_with_objects(
        &function,
        &baseline,
        &[],
        &[(numeric.0, crate::feedback::ValueClass::NumberInt32.bit())],
        &plan.selected,
        &objects,
    )
    .unwrap();
    let (candidate_cfg, candidate_stack, _) = analyses(&candidate);
    let recomposed = object_facts::Plan::build(
        &candidate,
        &candidate_cfg,
        &candidate_stack,
        &candidate.optimizing_objects,
        &candidate.optimizing_structural,
        &[],
        &numeric_properties,
    )
    .unwrap();
    assert_eq!(object.guard_cost, recomposed.guard_cost);
    assert_eq!(object.reachable, recomposed.reachable);
    let mut branches = 0;
    for pc in 0..baseline.jit_ops().len() {
        assert_eq!(object.comparison(pc), recomposed.comparison(pc));
        assert_eq!(object.branch(pc), recomposed.branch(pc));
        branches += usize::from(object.branch(pc).is_some());
    }
    assert_eq!(
        branches, 1,
        "exact Object-versus-Null branch survives the rebuild"
    );
    let final_plan = Plan::with_objects(
        &candidate,
        &candidate_cfg,
        &candidate_stack,
        &candidate.optimizing_structural,
        Some(&recomposed),
    )
    .unwrap();
    assert_eq!(plan.minimum_tests, final_plan.minimum_tests);
    assert_eq!(plan.per_pc, final_plan.per_pc);
    assert_eq!(plan.selected.len(), final_plan.selected.len());
    let numeric = specialization::Plan::build(
        &candidate,
        &candidate_cfg,
        &candidate_stack,
        &candidate.optimizing_inputs,
        &candidate.optimizing_properties,
    )
    .unwrap();
    assert!(numeric
        .properties
        .iter()
        .any(|&(pc, ty)| pc == numeric_properties[0].0 && ty.is_number()));
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    object_facts::TEST_EMITTED.with(|counts| counts.set([0; 3]));
    let code =
        Rc::new(super::super::compile_with_deopt(&candidate, &values, &layout, None).unwrap());
    object_facts::TEST_EMITTED.with(|counts| {
        let counts = counts.get();
        assert_eq!(counts[0], 1, "actual Object result guard emitted");
        assert!(
            counts[1] > 0 && counts[2] > 0,
            "actual constant comparison and feasible branch emission"
        );
    });
    assert!(code.pc_offsets.is_empty());
    candidate
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function
        .optimizing_code
        .set(Some(candidate.clone()))
        .is_ok());
    engine.set_tier(Tier::Jit);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine, "subject(node,other)"), "v2");
    assert_eq!(code.residency.referenced.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(candidate.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    // A descriptor miss remains BEFORE the required read. The getter and cold
    // branch execute once; a folded native branch never suppresses author effects.
    eval(&mut engine,"var reads=0;Object.defineProperty(node,'tag',{get(){reads++;return null;},configurable:true});void 0;");
    code.residency.referenced.set(0);
    assert_eq!(
        eval(
            &mut engine,
            "try{subject(node,other);}catch(e){var coldOK=e.message==='cold';}reads+':'+coldOK"
        ),
        "1:true"
    );
    assert_eq!(code.residency.referenced.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(candidate.optimizing_misses.get(), 1);
    bytecode::native_deopt::ENTRIES.with(|entries| {
        assert_eq!(
            entries
                .borrow()
                .iter()
                .map(|row| (row.0, row.1))
                .collect::<Vec<_>>(),
            [(tag.0 as usize, 0)]
        )
    });
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}
