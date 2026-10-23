//! Host callbacks may run a second Engine. The caller's allocation heap and Symbol Agent must
//! be restored before any engine work resumes, including builtin continuations and unwinding.

use crate::bytecode::Tier;
use crate::value::ValueView;
use crate::value::{Gc, NativeCaptures, Property, Value};
use crate::{Completion, Engine};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("activation fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn owns(engine: &Engine, object: &Gc) -> bool {
    engine.interp.gc_heap.core.borrow().owns_cell(
        Gc::as_ptr(object) as usize,
        engine.interp.gc_heap.object_desc,
    )
}

fn install_hop(engine: &mut Engine, inner: Rc<RefCell<Engine>>, fail: bool) {
    engine.interp.activate_gc_heap();
    let native = engine.interp.make_native_closure(
        "hop",
        0,
        Rc::new(move |_, _, _| {
            assert_eq!(eval(&mut inner.borrow_mut(), "var innerKept={};42"), "42");
            if fail {
                Err(Value::Undefined)
            } else {
                Ok(Value::Undefined)
            }
        }),
        NativeCaptures::None,
    );
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("hop", Property::plain(Value::Obj(native)));
}

#[test]
fn native_reentry_restores_the_allocation_heap_on_success_and_throw_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for fail in [false, true] {
            let mut outer = Engine::new();
            outer.set_tier(tier);
            outer.set_tier_threshold(0);
            let inner = Rc::new(RefCell::new(Engine::new()));
            install_hop(&mut outer, inner.clone(), fail);
            assert_eq!(
                eval(
                    &mut outer,
                    "function make(){try{hop();}catch(e){} return {answer:42};}\
                 var kept;for(var n=0;n<80;n++)kept=make();kept.answer"
                ),
                "42"
            );
            let env = outer.interp.global_env;
            let ValueView::Obj(kept) = (outer
                .interp
                .get_var("kept", &env)
                .unwrap_or_else(|_| panic!("kept read")))
            .view() else {
                panic!("object")
            };
            assert!(owns(&outer, &kept), "{tier:?}, native error={fail}");
            inner.borrow_mut().interp.gc_collect();
            assert_eq!(eval(&mut outer, "kept.answer"), "42");
            outer.interp.gc_collect();
        }
    }
}

#[test]
fn native_reentry_restores_the_agent_during_unwinding() {
    let mut outer = Engine::new();
    let inner = Rc::new(RefCell::new(Engine::new()));
    outer.interp.activate_gc_heap();
    let native = outer.interp.make_native_closure(
        "panicHop",
        0,
        Rc::new(move |_, _, _| {
            eval(&mut inner.borrow_mut(), "42");
            panic!("host callback unwind");
        }),
        NativeCaptures::None,
    );
    let call = native.borrow().call.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        outer.interp.dispatch_native(&call, Value::Undefined, &[])
    }));
    assert!(result.is_err());
    assert!(Rc::ptr_eq(
        &crate::value::active_gc_heap(),
        &outer.interp.gc_heap
    ));
    // No engine entry here may mask a stale allocation heap by activating the owner again.
    assert!(owns(&outer, &outer.interp.new_object()));
}

#[test]
fn clocks_restore_the_heap_before_builtin_allocation_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut outer = Engine::new();
        outer.set_tier(tier);
        outer.set_tier_threshold(0);
        let inner = Rc::new(RefCell::new(Engine::new()));
        outer.interp.wall_clock = Some(Rc::new(move || {
            eval(&mut inner.borrow_mut(), "0");
            1234.0
        }));
        assert_eq!(
            eval(&mut outer, "var date=new Date();date.getTime()"),
            "1234"
        );
        assert!(Rc::ptr_eq(
            &crate::value::active_gc_heap(),
            &outer.interp.gc_heap
        ));
        outer.interp.gc_collect();
        assert_eq!(eval(&mut outer, "date.getTime()"), "1234");
    }
}

#[test]
fn module_loaders_restore_the_heap_on_success_fallback_and_error_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut outer = Engine::new();
        outer.set_tier(tier);
        outer.set_tier_threshold(0);
        let inner = Rc::new(RefCell::new(Engine::new()));
        let sync = inner.clone();
        outer.set_module_loader(move |specifier, _| {
            eval(&mut sync.borrow_mut(), "0");
            (specifier == "found").then(|| {
                (
                    "found".to_string(),
                    "export const value={answer:42};".to_string(),
                )
            })
        });
        outer.set_async_dynamic_module_loader(move |_, _, _, _| {
            eval(&mut inner.borrow_mut(), "0");
            false // fall through to the synchronous loader after restoring the owner
        });
        eval(
            &mut outer,
            "var loaded,missing;import('found').then(ns=>loaded=ns.value);\
             import('missing').catch(e=>missing=e.name);",
        );
        assert_eq!(
            eval(&mut outer, "loaded.answer+':'+missing"),
            "42:TypeError"
        );
        outer.interp.gc_collect();
        assert_eq!(eval(&mut outer, "loaded.answer"), "42");
    }
}

#[test]
fn settings_hooks_restore_the_heap_before_suspended_work_and_after_leave() {
    fn hop(interp: &mut crate::interpreter::Interp) {
        let inner = interp.host::<Rc<RefCell<Engine>>>().unwrap().clone();
        eval(&mut inner.borrow_mut(), "0");
    }
    fn enter(interp: &mut crate::interpreter::Interp, _: u64) {
        hop(interp);
    }
    let mut outer = Engine::new();
    let inner = Rc::new(RefCell::new(Engine::new()));
    outer.interp.op_state().put(inner);
    outer.interp.host_job_context_enter = Some(enter);
    outer.interp.host_job_context_leave = Some(hop);
    outer.interp.activate_gc_heap();
    let global = outer.interp.global;
    let made = outer
        .interp
        .with_suspended_context(&global, 23, |interp| interp.new_object());
    assert!(owns(&outer, &made));
    assert!(owns(&outer, &outer.interp.new_object()));
}

#[test]
fn native_payload_gc_and_teardown_can_release_an_engine_with_live_weak_observers() {
    struct Witness(Rc<Cell<usize>>);
    impl Drop for Witness {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    for teardown in [false, true] {
        let mut outer = Engine::new();
        let mut inner = Engine::new();
        eval(&mut inner, "var target={},reference=new WeakRef(target);");
        // Foreign weak observations are internal-only but supported. During A's sweep, B's
        // metadata destructor must remove this live A subscription while A's core is borrowed.
        outer.interp.activate_gc_heap();
        let foreign_target = outer.interp.new_object();
        outer
            .interp
            .global
            .borrow_mut()
            .props
            .insert("foreignTarget", Property::plain(Value::Obj(foreign_target)));
        let foreign_weak = crate::interpreter::WeakTarget::of(&Value::Obj(foreign_target)).unwrap();
        inner.interp.activate_gc_heap();
        inner
            .interp
            .weak_metadata
            .subscribe(foreign_weak, crate::weak_metadata::Subscriber::WeakRef(0));
        let weak_heap = Rc::downgrade(&inner.interp.gc_heap);
        let dropped = Rc::new(Cell::new(0));
        let witness = Witness(dropped.clone());
        outer.interp.activate_gc_heap();
        crate::gc_test::in_frame(|| {
            let native = outer.interp.make_native_closure(
                "ownedEngine",
                0,
                Rc::new(move |_, _, _| {
                    let _keep = (&inner, &witness);
                    Ok(Value::Undefined)
                }),
                NativeCaptures::None,
            );
            outer
                .interp
                .global
                .borrow_mut()
                .props
                .insert("heldNative", Property::plain(Value::Obj(native)));
        });
        if !teardown {
            eval(&mut outer, "heldNative=null;0");
            // During this collection, the native's captured Engine is destroyed while the
            // active outer heap's core is mutably borrowed by its object finalizer.
            outer.interp.gc_collect();
            assert!(weak_heap.upgrade().is_none());
            assert_eq!(dropped.get(), 1);
            // The parent Engine remains usable after the nested Engine retired.
            assert_eq!(eval(&mut outer, "({answer:42}).answer"), "42");
        }
        drop(outer);
        assert!(weak_heap.upgrade().is_none());
        assert_eq!(dropped.get(), 1, "teardown={teardown}");
    }
}

#[test]
fn bare_native_reentry_restores_the_heap_on_committed_calls_and_interruption() {
    fn hop(interp: &mut crate::interpreter::Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
        let inner = interp.host::<Rc<RefCell<Engine>>>().unwrap().clone();
        eval(&mut inner.borrow_mut(), "0");
        Ok(Value::Undefined)
    }
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut outer = Engine::new();
        outer.set_tier(tier);
        outer.set_tier_threshold(0);
        let inner = Rc::new(RefCell::new(Engine::new()));
        outer.interp.op_state().put(inner.clone());
        outer.interp.activate_gc_heap();
        let global = outer.interp.global;
        outer.interp.def_method(&global, "bareHop", 0, hop);
        assert_eq!(
            eval(
                &mut outer,
                "function makeBare(){bareHop();return {answer:42};}\
             var kept;for(var n=0;n<80;n++)kept=makeBare();kept.answer"
            ),
            "42"
        );
        let env = outer.interp.global_env;
        let ValueView::Obj(kept) = (outer
            .interp
            .get_var("kept", &env)
            .unwrap_or_else(|_| panic!("kept read")))
        .view() else {
            panic!("object")
        };
        assert!(owns(&outer, &kept));
        let interrupt = outer.interrupt_handle();
        let interrupter = outer.interp.make_native_closure(
            "interruptHop",
            0,
            Rc::new(move |_, _, _| {
                eval(&mut inner.borrow_mut(), "0");
                interrupt.cancel();
                Ok(Value::Undefined)
            }),
            NativeCaptures::None,
        );
        let call = interrupter.borrow().call.clone();
        assert!(matches!(
            outer.interp.dispatch_native(&call, Value::Undefined, &[]),
            Err(crate::interpreter::Abrupt::Interrupt(_))
        ));
        assert!(Rc::ptr_eq(
            &crate::value::active_gc_heap(),
            &outer.interp.gc_heap
        ));
    }
}

#[test]
#[cfg(feature = "embed")]
fn scoped_text_conversions_reactivate_the_owner_after_user_conversion() {
    struct Text(Rc<RefCell<Engine>>);
    impl AsRef<str> for Text {
        fn as_ref(&self) -> &str {
            eval(&mut self.0.borrow_mut(), "0");
            "converted"
        }
    }
    impl From<Text> for String {
        fn from(text: Text) -> Self {
            text.as_ref().to_string()
        }
    }
    let mut outer = Engine::new();
    let inner = Rc::new(RefCell::new(Engine::new()));
    outer.scope(|cx| {
        let text = cx.str(Text(inner.clone()));
        assert!(cx.handle(&text).is_ok());
        let text = cx.string_from_engine_text(Text(inner.clone()));
        assert!(cx.handle(&text).is_ok());
        let error = cx.make_error("Error", Text(inner));
        assert!(cx.handle(&error).is_ok());
        let message = cx
            .member_get(&error, "message")
            .unwrap_or_else(|_| panic!("message"));
        assert!(
            cx.handle(&message).is_ok(),
            "the message payload belongs to its Error's heap"
        );
    });
}
