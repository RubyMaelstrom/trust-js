use super::*;
use crate::bytecode::{native_deopt::ENTRIES, Tier};
use crate::feedback::ValueClass;
use crate::interpreter::Interp;
use crate::value::{Callable, Object, Property, Value as JsValue};
use crate::{Completion, Engine};
use std::rc::{Rc, Weak};

struct Enabled(Option<bool>);
impl Enabled {
    fn new(enabled: bool) -> Self {
        Self(TEST_ENABLED.with(|value| value.replace(Some(enabled))))
    }
}
impl Drop for Enabled {
    fn drop(&mut self) {
        TEST_ENABLED.with(|value| value.set(self.0));
    }
}

thread_local! {
    static OWNED: std::cell::RefCell<Vec<Weak<std::cell::RefCell<Object>>>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

fn make_owned(i: &mut Interp, _: JsValue, _: &[JsValue]) -> Result<JsValue, JsValue> {
    let object = Object::new(Some(i.object_proto.clone()));
    object.borrow_mut().props.insert(
        "tag",
        Property::data(JsValue::str("alive"), true, true, true),
    );
    OWNED.with(|owned| owned.borrow_mut().push(Rc::downgrade(&object)));
    Ok(JsValue::Obj(object))
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("recovery fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

struct Fixture {
    engine: Engine,
    function: Rc<crate::ast::Function>,
    candidate: Rc<Chunk>,
    sites: Vec<(usize, String)>,
}

fn fixture() -> Fixture {
    OWNED.with(|owned| owned.borrow_mut().clear());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "makeOwned", 0, make_owned);
    eval(
        &mut engine,
        r#"
        var trace=[];
        function subject(o) {
            'use strict';
            try {
                var hold=makeOwned(), total=0;
                total+=o.a;
                total+=o.b;
                total+=o.c;
                total+=o.d;
                return hold.tag+':'+total;
            } finally {trace.push('F');}
        }
        var receiver={a:1,b:2,c:3,d:4};
        subject(receiver);subject(receiver);trace=[];
        "#,
    );
    let env = engine.interp.global_env.clone();
    let object = engine.interp.get_var("subject", &env).ok().unwrap();
    let function = match &object.as_obj().unwrap().borrow().call {
        Callable::User(user) => user.func.clone(),
        _ => panic!("fixture User"),
    };
    let baseline = function.code.get().and_then(Option::as_ref).unwrap();
    let sites: Vec<_> = baseline
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| {
            let name = match *op {
                Op::GetPropLocal(_, name, _) | Op::GetProp(name, _) => name,
                _ => return None,
            };
            let name = baseline.jit_name(name);
            ["a", "b", "c", "d"]
                .contains(&name)
                .then(|| (pc, name.to_owned()))
        })
        .collect();
    assert_eq!(sites.len(), 4);
    let properties: Vec<_> = sites
        .iter()
        .map(|(pc, _)| (*pc as u32, ValueClass::NumberInt32.bit()))
        .collect();
    let candidate = bytecode::compile_for_optimizer(&function, baseline, &[], &properties)
        .expect("canonical specializing snapshot");
    Fixture {
        engine,
        function,
        candidate,
        sites,
    }
}

fn compile(fixture: &mut Fixture, enabled: bool) -> (JitCode, [usize; 4], [usize; 4]) {
    let _enabled = Enabled::new(enabled);
    TEST_COUNTS.with(|counts| counts.set([0; 4]));
    let values = crate::value::jit_layout(&fixture.engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut fixture.engine.interp);
    let code = compile_with_deopt(&fixture.candidate, &values, &layout, None)
        .expect("specializing fixture native code");
    (
        code,
        TEST_COUNTS.with(std::cell::Cell::get),
        TEST_IR_SHAPE.with(std::cell::Cell::get),
    )
}

#[test]
fn identical_publication_shapes_share_real_calls_and_bound_phi_work() {
    let mut fixture = fixture();
    let (separate, separate_sites, separate_ir) = compile(&mut fixture, false);
    let (shared, shared_sites, shared_ir) = compile(&mut fixture, true);
    assert_eq!(separate_sites[0], shared_sites[0]);
    assert_eq!(separate_sites[1], separate_sites[0]);
    assert!(
        shared_sites[1] < shared_sites[0],
        "actual shared cold groups"
    );
    assert!(
        shared_sites[3] < separate_sites[3],
        "fewer actual word stores"
    );
    assert!(
        shared_ir[0] < separate_ir[0],
        "real lowered instruction savings"
    );
    assert!(shared.len < separate.len, "actual emitted code shrinks");
    assert!(shared_ir[3] <= separate_ir[3] + shared_sites[0] * 8);
    eprintln!("terminal recovery separate sites={separate_sites:?} IR={separate_ir:?} bytes={}; shared sites={shared_sites:?} IR={shared_ir:?} bytes={}", separate.len, shared.len);
}

#[test]
fn shared_dynamic_pc_preserves_dirty_owner_gc_completion_and_effect_order() {
    for throwing in [false, true] {
        for selected in 0..4 {
            let mut fixture = fixture();
            let (code, _, _) = compile(&mut fixture, true);
            let code = Rc::new(code);
            assert!(code.pc_offsets.is_empty(), "actual optimizing body");
            fixture
                .candidate
                .jit
                .begin_compile()
                .unwrap()
                .commit(Some(code.clone()));
            assert!(fixture
                .function
                .optimizing_code
                .set(Some(fixture.candidate.clone()))
                .is_ok());
            bytecode::invalidate_call_caches();
            fixture.engine.set_tier(Tier::Jit);
            let (pc, key) = &fixture.sites[selected];
            eval(
                &mut fixture.engine,
                &format!(
                    "receiver.{key}={{valueOf(){{trace.push('C');$262.gc();{} }}}};trace=[];",
                    if throwing {
                        "throw 'boom';".into()
                    } else {
                        format!("return {};", selected + 1)
                    }
                ),
            );
            ENTRIES.with(|entries| entries.borrow_mut().clear());
            let result = eval(&mut fixture.engine,
                "var result;try{result='return:'+subject(receiver);}catch(e){result='throw:'+e;}result+'|'+trace.join(',')");
            assert_eq!(
                result,
                if throwing {
                    "throw:boom|C,F"
                } else {
                    "return:alive:10|C,F"
                }
            );
            assert_eq!(fixture.candidate.optimizing_misses.get(), 1);
            ENTRIES.with(|entries| {
                let entries = entries.borrow();
                assert_eq!(entries.len(), 1, "one terminal recovery, same invocation");
                assert_eq!(entries[0].0, *pc, "exact dynamic before-read PC");
            });
            assert_ne!(code.residency.referenced.get(), 0);
            assert_eq!(code.residency.active.get(), 0);
            assert!(fixture.engine.interp.fn_frames.is_empty());
            OWNED.with(|owned| {
                assert_eq!(owned.borrow().len(), 3, "two warmups plus one invocation");
            });
            fixture.engine.interp.gc_collect();
            OWNED.with(|owned| {
                assert!(
                    owned.borrow().iter().all(|owner| owner.strong_count() == 0),
                    "native/VM completion releases every created owner exactly once"
                );
            });
        }
    }
}

thread_local! {
    static RELEASED: std::cell::RefCell<Vec<Weak<std::cell::RefCell<Object>>>> = const {
        std::cell::RefCell::new(Vec::new())
    };
    static RELEASE_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CHECK_COUNTS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn check_released(i: &mut Interp, _: JsValue, _: &[JsValue]) -> Result<JsValue, JsValue> {
    i.gc_collect();
    RELEASED.with(|roots| {
        CHECK_COUNTS.with(|counts| {
            *counts.borrow_mut() = roots.borrow().iter().map(Weak::strong_count).collect()
        })
    });
    RELEASE_CHECKS.with(|count| count.set(count.get() + 1));
    Ok(JsValue::Undefined)
}

#[test]
fn clean_stale_inline_callee_word_is_not_republished_after_reset_and_collection() {
    // ResetSlots is a real checked private-slot writer. The inline callee word is
    // dead after its exit, so sparse reloads deliberately leave its SSA copy stale.
    // Widening a recovery's dirty set to all tracked locals would resurrect that
    // destroyed callable (and corrupt its owner count) at the later boundary.
    for selected in ["a", "b"] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let global = engine.interp.global.clone();
        engine
            .interp
            .def_method(&global, "checkReleased", 0, check_released);
        eval(
            &mut engine,
            r#"
            function make(secret) {return function(drop) {
                if(drop){holder.f=null;return secret.tag.length+1;}
                return 2;
            };}
            var sentinel={tag:'x'},holder={f:make(sentinel)};
            function subject(o,drop){var total=holder.f(drop);
                if(drop)checkReleased();return total+o.a+o.b;}
            var receiver={a:3,b:5};
        "#,
        );
        let env = engine.interp.global_env.clone();
        let callable = engine.interp.get_var("subject", &env).ok().unwrap();
        let user = match &callable.as_obj().unwrap().borrow().call {
            Callable::User(user) => user.clone(),
            _ => panic!("User"),
        };
        let primary = bytecode::compile(&user.func).unwrap();
        primary.inline_attempted.set(true);
        primary.inline_retry_at.set(0);
        assert!(user.func.code.set(Some(primary.clone())).is_ok());
        for _ in 0..8 {
            eval(&mut engine, "subject(receiver,false)");
        }
        let plan =
            bytecode::plan_inlines(&primary, &user.func, user.realm, &engine.interp.global_env);
        assert!(!plan.is_empty());
        let flattened = bytecode::compile_with_inlines(&user.func, &plan, &primary).unwrap();
        // The temporary compiler plan owns the callee; executable inline targets
        // retain only Weak headers. Release that planning lease before the GC probe.
        drop(plan);
        let callee = flattened
            .jit_ops()
            .iter()
            .find_map(|op| match *op {
                Op::InlineGuard(target, _) => Some(flattened.jit_inline_target(target).pin.clone()),
                _ => None,
            })
            .expect("actual inline target");
        let callee_code = {
            let object = callee.upgrade().unwrap();
            let object = object.borrow();
            let Callable::User(callee_user) = &object.call else {
                panic!("User inline target")
            };
            callee_user
                .func
                .execution_code()
                .and_then(Option::as_ref)
                .unwrap()
                .jit
                .get()
                .flatten()
                .expect("JIT-warmed original callee")
        };
        callee_code.residency.referenced.set(0);
        let sentinel = engine.interp.get_var("sentinel", &env).ok().unwrap();
        let payload = Rc::downgrade(sentinel.as_obj().unwrap());
        drop(sentinel);
        engine.set_tier(Tier::Bytecode);
        assert!(user.func.code2.set(Some(flattened.clone())).is_ok());
        bytecode::invalidate_call_caches();
        for _ in 0..2 {
            eval(&mut engine, "subject(receiver,false)");
        }
        let pc = flattened
            .jit_ops()
            .iter()
            .position(|op| match *op {
                Op::GetPropLocal(_, name, _) | Op::GetProp(name, _) => {
                    flattened.jit_name(name) == selected
                }
                _ => false,
            })
            .unwrap();
        let cfg = Cfg::build(&flattened).unwrap();
        let frame = frame::FramePlan::build(&flattened, &cfg, true).unwrap();
        let stale = flattened
            .jit_ops()
            .iter()
            .enumerate()
            .find_map(|(reset, op)| match *op {
                Op::ResetSlots(slot, 1)
                    if frame.tracked.contains(&slot)
                        && !frame.live_after[reset].contains(&slot)
                        && !frame.dirty_at[pc].contains(&slot) =>
                {
                    Some(slot)
                }
                _ => None,
            })
            .expect("real checked reset leaves a clean tracked SSA word stale");
        let _enabled = Enabled::new(true);
        let values = crate::value::jit_layout(&engine.interp.object_proto);
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let code = Rc::new(compile_with_deopt(&flattened, &values, &layout, Some(pc)).unwrap());
        flattened
            .jit
            .begin_compile()
            .unwrap()
            .commit(Some(code.clone()));
        engine.set_tier(Tier::Jit);
        RELEASED.with(|roots| *roots.borrow_mut() = vec![callee.clone(), payload.clone()]);
        RELEASE_CHECKS.with(|count| count.set(0));
        eval(&mut engine, "sentinel=null;");
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        assert_eq!(eval(&mut engine, "subject(receiver,true)"), "10");
        assert_eq!(RELEASE_CHECKS.with(std::cell::Cell::get), 1);
        assert_eq!(
            callee_code.residency.referenced.get(),
            0,
            "real inline hit executes ResetSlots; original callee fallback is not sufficient"
        );
        CHECK_COUNTS.with(|counts| {
            assert_eq!(
                *counts.borrow(),
                vec![0, 0],
                "owners destroyed before recovery"
            )
        });
        ENTRIES.with(|entries| {
            assert_eq!(
                entries
                    .borrow()
                    .iter()
                    .map(|entry| entry.0)
                    .collect::<Vec<_>>(),
                vec![pc]
            )
        });
        assert_eq!(code.residency.referenced.get(), 1);
        assert_eq!(code.residency.active.get(), 0);
        engine.interp.gc_collect();
        assert_eq!(
            callee.strong_count(),
            0,
            "callee slot {stale} never revived"
        );
        assert_eq!(payload.strong_count(), 0);
        assert!(engine.interp.fn_frames.is_empty());
        RELEASED.with(|roots| roots.borrow_mut().clear());
    }
}
