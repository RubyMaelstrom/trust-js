//! Actual native guard elision and observer regression tests, ECMA-262 e28783d5
//! OrdinaryGet, descriptor mutation, execution contexts and WeakRef liveness.

use super::*;
use crate::{bytecode::Tier, feedback::ValueClass, value::Callable, Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("heap proof fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn install(engine: &mut Engine, inputs: &[(u16, u32)], forced_budget: bool) -> Rc<JitCode> {
    let env = engine.interp.global_env.clone();
    let subject = engine
        .interp
        .get_var("subject", &env)
        .ok()
        .expect("declared subject");
    let function = {
        let object = subject.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user function")
        };
        user.func.clone()
    };
    let primary = function
        .code
        .get()
        .and_then(Option::as_ref)
        .expect("real warmed body");
    let properties: Vec<_> = primary
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| {
            let (Op::GetPropLocal(_, name, cache) | Op::GetPropThis(name, cache)) = *op else {
                return None;
            };
            (primary.jit_name(name) == "x")
                .then(|| {
                    primary
                        .jit_cache_preferred(cache)
                        .filter(|state| state.depth == 0)
                })
                .flatten()
                .map(|_| (pc as u32, ValueClass::NumberDouble.bit()))
        })
        .collect();
    assert!(properties.len() >= 2, "two warmed own-number accesses");
    let chunk = bytecode::compile_for_optimizer(&function, primary, inputs, &properties).unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    property::TEST_METADATA_ELISIONS.with(|counts| counts.set([0; 4]));
    // Mutation coverage can exceed hot admission's work estimate. Read-only
    // fixtures must still compile under the actual production candidate budget.
    let compiled = if forced_budget {
        compile_with_limits(&chunk, &values, &layout, None, 262_144, 65_536)
    } else {
        compile_with_deopt(&chunk, &values, &layout, None)
    };
    let code = Rc::new(compiled.expect("native optimizer"));
    assert!(code.pc_offsets.is_empty(), "no template fallback");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk)).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    code
}

fn fixture(
    source: &str,
    warm: &str,
    run: &str,
    expected: &str,
    inputs: &[(u16, u32)],
    forced_budget: bool,
) -> [usize; 4] {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut oracle = Engine::new();
        oracle.set_tier(tier);
        oracle.set_tier_threshold(0);
        eval(&mut oracle, source);
        eval(&mut oracle, warm);
        assert_eq!(eval(&mut oracle, run), expected, "{tier:?}");
        oracle.interp.gc_collect();
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, warm);
    let code = install(&mut engine, inputs, forced_budget);
    let removed = property::TEST_METADATA_ELISIONS.with(|counts| counts.get());
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine, run), expected);
    assert!(code.residency.referenced.get() > 0, "real optimized entry");
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
    removed
}

#[test]
fn numeric_own_metadata_guards_are_reused_across_a_diamond() {
    let removed = fixture(
        r#"function subject(o, flag) { 'use strict';
            return o.x+(flag?o.x:o.x)+o.x+o.x;
        } var receiver={x:3};"#,
        "subject(receiver,true);subject(receiver,false);",
        "[subject(receiver,true),subject(receiver,false)].join('|')",
        "12|12",
        &[(1, ValueClass::Boolean.bit())],
        false,
    );
    assert_eq!(removed, [4; 4], "four repeated sites omit metadata guards");
}

#[test]
fn generic_getter_fallback_rejoining_native_code_invalidates_metadata() {
    let removed = fixture(
        r#"function subject(o) { 'use strict';
            var first=o.x;var ignored=o.middle;return first+o.x;
        } var receiver={x:2,middle:'m'},trace=[];"#,
        "subject(receiver);",
        r#"var normal=subject(receiver);
        Object.defineProperty(receiver,'middle',{get(){trace.push('middle');
            Object.defineProperty(receiver,'x',{get(){trace.push('x');return 7;},configurable:true});
            return 'changed';},configurable:true});
        [normal,subject(receiver),trace.join(',')].join('|')"#,
        "4|9|middle,x",
        &[],
        false,
    );
    assert_eq!(
        removed, [0; 4],
        "generic access may invoke a rejoining getter"
    );
}

#[test]
fn same_key_write_retains_live_values_and_reestablishes_metadata() {
    let removed = fixture(
        r#"function subject(o, value) { 'use strict';
            var first=o.x;o.x=value;return first+o.x+o.x;
        } var receiver={x:3};"#,
        "subject(receiver,5);receiver.x=3;",
        "var normal=subject(receiver,5);receiver.x=3;[normal,subject(receiver,'s')].join('|')",
        "13|3ss",
        &[],
        true,
    );
    assert_eq!(
        removed, [1; 4],
        "write is a conservative barrier; later read proves again"
    );
}

#[test]
fn current_this_receiver_keeps_metadata_without_embedding_its_identity() {
    let removed = fixture(
        r#"function subject() { 'use strict';return this.x+this.x*2; }
        var receiver={x:3},other={x:7};"#,
        "subject.call(receiver);subject.call(other);",
        "[subject.call(receiver),subject.call(other)].join('|')",
        "9|21",
        &[],
        false,
    );
    assert_eq!(removed, [1; 4]);
}

#[test]
fn receiver_reassignment_never_reuses_physical_slot_metadata() {
    let removed = fixture(
        r#"function subject(o, other) { 'use strict';
            var first=o.x;o=other;return first+o.x+o.x;
        } var receiver={x:2},other={x:7};"#,
        "subject(receiver,other);",
        "subject(receiver,other)",
        "16",
        &[],
        false,
    );
    assert_eq!(
        removed, [0; 4],
        "mutable receiver requires future SSA-origin mapping"
    );
}

#[test]
fn backedge_poll_keeps_header_metadata_guards() {
    let removed = fixture(
        r#"function subject(o) { 'use strict';
            var total=o.x;for(var k=0;k<2;k++)total+=o.x;return total;
        } var receiver={x:2};"#,
        "subject(receiver);",
        "subject(receiver)",
        "6",
        &[],
        false,
    );
    assert_eq!(
        removed, [0; 4],
        "header's conditional observer breaks preheader proof"
    );
}

#[test]
fn iterator_close_before_finally_clears_this_metadata() {
    // ECMA-262 IteratorClose invokes return before the enclosing finally. The
    // return operand is already computed, but its descriptor proof must not
    // enter the independent finalizer root after that arbitrary callback.
    let removed = fixture(
        r#"function subject(iterable) { 'use strict';
            try { for (const item of iterable) { return this.x+this.x; } }
            finally { trace.push('finally:'+(this.x+this.x)); }
        }
        var receiver={x:3},trace=[];
        var iterable={ [Symbol.iterator]() { return {
            next(){return {value:1,done:false};},
            return(){trace.push('close');
                Object.defineProperty(receiver,'x',{configurable:true,
                    get(){trace.push('get');return 11;}});
                return {done:true};}
        };}};"#,
        "subject.call(receiver,[1]);trace=[];",
        "var result=subject.call(receiver,iterable);result+'|'+trace.join(',')",
        "6|close,get,get,finally:22",
        &[],
        true,
    );
    assert_eq!(
        removed, [2; 4],
        "each region proves its first access independently"
    );
}
