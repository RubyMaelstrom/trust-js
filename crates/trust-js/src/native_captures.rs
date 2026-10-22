//! Native functions' explicit JavaScript captures are internal graph edges (traced with their
//! payload, `GcTrace for NativeCallable`), which these tests check.
//!
//! ECMA-262 #sec-liveness / #sec-weakref-execution, local snapshot e28783d5:
//! preserve observable captures, but do not turn unreachable callable cycles
//! into external roots. A shared native payload owns its Value handles once,
//! even when several function objects share it. Active Rust leases are roots.

mod tests {
    use crate::interpreter::Interp;
    use crate::value::ValueView;
    use crate::value::{Gc, Object, Property, Value};

    fn object(value: &Value) -> Gc {
        let ValueView::Obj(object) = value.view() else {
            panic!("fixture must be an object");
        };
        object
    }

    fn returning_capture(
        _: &mut Interp,
        _: Value,
        _: &[Value],
        captures: &[Value],
    ) -> Result<Value, Value> {
        Ok(captures[0].clone())
    }

    #[test]
    fn native_captures_collect_cycles_and_count_repeated_handles() {
        let mut interp = Interp::new();
        interp.gc_collect();
        // Strong handles live only in this helper frame (the collector scans the stack).
        let (weak_target, weak_callback) = crate::gc_test::in_frame(|| {
            let target = Object::new(None);
            let callback = interp.new_native_fn_with_captures(
                "capture",
                0,
                returning_capture,
                vec![Value::Obj(target.clone()), Value::Obj(target.clone())],
            );
            target
                .borrow_mut()
                .props
                .insert("callback", Property::builtin(callback.clone()));
            (Gc::downgrade(&target), Gc::downgrade(&object(&callback)))
        });
        interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&weak_target),
            "capture cycle survived"
        );
        assert!(
            !crate::gc_test::alive(&weak_callback),
            "callback cycle survived"
        );
    }

    #[test]
    fn native_captures_shared_payload_is_counted_once_and_traced_from_every_owner() {
        for live_second in [false, true] {
            let mut interp = Interp::new();
            // Strong handles live only in helper frames (the collector scans the stack).
            let weak_target = crate::gc_test::in_frame(|| {
                let (live, weak_target) = crate::gc_test::in_frame(|| {
                    let target = Object::new(None);
                    let first = interp.new_native_fn_with_captures(
                        "shared",
                        0,
                        returning_capture,
                        vec![Value::Obj(target.clone())],
                    );
                    let second = Object::new(None);
                    second.borrow_mut().call = object(&first).borrow().call.clone();
                    target
                        .borrow_mut()
                        .props
                        .insert("first", Property::builtin(first.clone()));
                    target
                        .borrow_mut()
                        .props
                        .insert("second", Property::builtin(Value::Obj(second.clone())));
                    let live = if live_second {
                        Value::Obj(second.clone())
                    } else {
                        first.clone()
                    };
                    (live, Gc::downgrade(&target))
                });
                interp.gc_collect();
                let result = interp
                    .invoke(live.clone(), Value::Null, &[])
                    .unwrap_or_else(|_| panic!("shared native invocation failed"));
                assert!(Gc::ptr_eq(
                    &object(&result),
                    &weak_target.upgrade().expect("live capture")
                ));
                weak_target
            });
            interp.gc_collect();
            assert!(
                !crate::gc_test::alive(&weak_target),
                "shared payload was counted more than once or left rooted"
            );
        }
    }

    #[test]
    fn native_captures_old_owner_preserves_a_shared_young_payload() {
        let mut interp = Interp::new();
        // Strong handles live only in helper frames (the collector scans the stack).
        let weak_target = crate::gc_test::in_frame(|| {
            let old_owner = Object::new(None);
            interp.gc_collect();
            let weak_target = crate::gc_test::in_frame(|| {
                let target = Object::new(None);
                let young_owner = interp.new_native_fn_with_captures(
                    "young",
                    0,
                    returning_capture,
                    vec![Value::Obj(target.clone())],
                );
                // Internal construction can share a newly allocated payload with an
                // existing function object.
                old_owner.borrow_mut().call = object(&young_owner).borrow().call.clone();
                target
                    .borrow_mut()
                    .props
                    .insert("callback", Property::builtin(young_owner.clone()));
                Gc::downgrade(&target)
            });
            interp.gc_collect();
            assert!(
                crate::gc_test::alive(&weak_target),
                "old native owner lost young captures"
            );
            let result = interp
                .invoke(Value::Obj(old_owner.clone()), Value::Undefined, &[])
                .unwrap_or_else(|_| panic!("old owner invocation failed"));
            assert!(Gc::ptr_eq(
                &object(&result),
                &weak_target.upgrade().expect("old owner's capture")
            ));
            weak_target
        });
        interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&weak_target),
            "cross-generation cycle survived full GC"
        );
    }

    #[test]
    fn native_captures_reentrant_collection_preserves_return_and_throw_identity() {
        fn reentrant(
            interp: &mut Interp,
            _: Value,
            _: &[Value],
            captures: &[Value],
        ) -> Result<Value, Value> {
            let target = &captures[0];
            interp.member_set(target, "calls", Value::Num(1.))?;
            interp.gc_collect();
            assert_eq!(interp.member_get(target, "calls")?.as_num_opt(), Some(1.));
            if matches!((captures[1]).view(), ValueView::Bool(true)) {
                Err(target.clone())
            } else {
                Ok(target.clone())
            }
        }
        for throws in [false, true] {
            let mut interp = Interp::new();
            // Strong handles live only in helper frames (the collector scans the stack).
            let weak_target = crate::gc_test::in_frame(|| {
                let (lease, weak_target) = crate::gc_test::in_frame(|| {
                    let target = Object::new(None);
                    target
                        .borrow_mut()
                        .props
                        .insert("self", Property::builtin(Value::Obj(target.clone())));
                    let callback = interp.new_native_fn_with_captures(
                        "reentrant",
                        0,
                        reentrant,
                        vec![Value::Obj(target.clone()), Value::Bool(throws)],
                    );
                    let lease = object(&callback).borrow().call.clone();
                    (lease, Gc::downgrade(&target))
                });
                // Exercise dispatch with no function Object owner at all. Its
                // in-flight NativeCallable lease must protect borrowed captures.
                let result = match interp.dispatch_native(&lease, Value::Null, &[]) {
                    Ok(value) if !throws => value,
                    Err(crate::interpreter::Abrupt::Throw(value)) if throws => value,
                    _ => panic!("native completion changed"),
                };
                assert!(Gc::ptr_eq(
                    &object(&result),
                    &weak_target.upgrade().expect("active capture")
                ));
                weak_target
            });
            interp.gc_collect();
            assert!(
                !crate::gc_test::alive(&weak_target),
                "completed native call left a root"
            );
        }
    }
}
