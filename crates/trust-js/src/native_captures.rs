//! Native functions' explicit JavaScript captures are internal graph edges.
//!
//! ECMA-262 #sec-liveness / #sec-weakref-execution, local snapshot e28783d5:
//! preserve observable captures, but do not turn unreachable callable cycles
//! into external roots. A shared native payload owns its Value handles once,
//! even when several function objects share it. Active Rust leases are roots.

use crate::fasthash::FastMap;
use crate::host::HostGcVisitor;
use crate::value::{Callable, Gc, NativeCallable, NativeCallableBody, Value};
use std::rc::Rc;

struct CaptureGroup {
    callable: Rc<NativeCallable>,
    owners: Vec<Gc>,
}

pub(crate) struct NativeCaptureSnapshot {
    groups: FastMap<usize, CaptureGroup>,
}

impl NativeCaptureSnapshot {
    pub(crate) fn empty() -> Self {
        Self {
            groups: FastMap::default(),
        }
    }

    /// Record `owner` (whose body is `object`) if it is a native function with explicit
    /// captures. Lets a collector pass that already visits every object build the snapshot.
    pub(crate) fn observe(&mut self, owner: &Gc, object: &crate::value::Object) {
        let Callable::NativeData(callable) = &object.call else {
            return;
        };
        let NativeCallableBody::Captured { captures, .. } = &callable.body else {
            return;
        };
        if captures.is_empty() {
            return;
        }
        self.groups
            .entry(Rc::as_ptr(callable) as usize)
            .or_insert_with(|| CaptureGroup {
                callable: callable.clone(),
                owners: Vec::new(),
            })
            .owners
            .push(owner.clone());
    }

    /// Consume the snapshot before the collector classifies ordinary Rc roots:
    /// its owner handles must not inflate that classification. No author code
    /// runs and no JavaScript handles are added to the identity-only edge sink.
    pub(crate) fn trace(self, visitor: &mut dyn HostGcVisitor) {
        for group in self.groups.into_values() {
            let NativeCallableBody::Captured { captures, .. } = &group.callable.body else {
                unreachable!("snapshot only records explicit captures");
            };
            // One temporary Rc belongs to this snapshot. Any remaining owner
            // not represented by a candidate Object is an active Rust lease,
            // foreign-heap owner, or old-generation owner during a minor GC.
            // Leave its captures conservative roots rather than discounting
            // them. This requires no old-heap scan or mutable write barrier.
            let internal_only = Rc::strong_count(&group.callable) == group.owners.len() + 1;
            for value in captures {
                if internal_only {
                    visitor.internal(value);
                }
                for owner in &group.owners {
                    visitor.edge(&Value::Obj(owner.clone()), value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interp;
    use crate::value::{GcCause, Object, Property};

    fn object(value: &Value) -> &Gc {
        let Value::Obj(object) = value else {
            panic!("fixture must be an object");
        };
        object
    }

    fn collect(interp: &mut Interp, minor: bool) {
        if minor {
            interp.gc_collect_young(GcCause::Explicit);
        } else {
            interp.gc_collect();
        }
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
    fn native_captures_collect_cycles_and_count_repeated_handles_in_both_collectors() {
        for minor in [false, true] {
            let mut interp = Interp::new();
            interp.gc_collect();
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
            let weak_target = Rc::downgrade(&target);
            let weak_callback = Rc::downgrade(object(&callback));
            drop((target, callback));
            collect(&mut interp, minor);
            assert!(
                weak_target.upgrade().is_none(),
                "capture cycle survived minor={minor}"
            );
            assert!(
                weak_callback.upgrade().is_none(),
                "callback cycle survived minor={minor}"
            );
        }
    }

    #[test]
    fn native_captures_shared_payload_is_counted_once_and_traced_from_every_owner() {
        for live_second in [false, true] {
            let mut interp = Interp::new();
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
            let weak_target = Rc::downgrade(&target);
            drop((target, first, second));
            interp.gc_collect_young(GcCause::Explicit);
            interp.gc_collect();
            let result = interp
                .invoke(live.clone(), Value::Null, &[])
                .unwrap_or_else(|_| panic!("shared native invocation failed"));
            assert!(Rc::ptr_eq(
                object(&result),
                &weak_target.upgrade().expect("live capture")
            ));
            drop((live, result));
            interp.gc_collect();
            assert!(
                weak_target.upgrade().is_none(),
                "shared payload was counted more than once or left rooted"
            );
        }
    }

    #[test]
    fn native_captures_external_callable_leases_remain_roots() {
        let mut interp = Interp::new();
        let target = Object::new(None);
        let callback = interp.new_native_fn_with_captures(
            "leased",
            0,
            returning_capture,
            vec![Value::Obj(target.clone())],
        );
        let lease = object(&callback).borrow().call.clone();
        target
            .borrow_mut()
            .props
            .insert("callback", Property::builtin(callback.clone()));
        let weak_target = Rc::downgrade(&target);
        drop((target, callback));
        interp.gc_collect_young(GcCause::Explicit);
        interp.gc_collect();
        assert!(
            weak_target.upgrade().is_some(),
            "Rust lease lost its capture"
        );
        let result = interp
            .dispatch_native(&lease, Value::Undefined, &[])
            .unwrap_or_else(|_| panic!("leased native invocation failed"));
        assert!(Rc::ptr_eq(
            object(&result),
            &weak_target.upgrade().expect("leased capture")
        ));
        drop((result, lease));
        interp.gc_collect();
        assert!(
            weak_target.upgrade().is_none(),
            "dropped lease kept capture alive"
        );
    }

    #[test]
    fn native_captures_old_owner_preserves_a_shared_young_payload() {
        let mut interp = Interp::new();
        let old_owner = Object::new(None);
        interp.gc_collect();
        let target = Object::new(None);
        let young_owner = interp.new_native_fn_with_captures(
            "young",
            0,
            returning_capture,
            vec![Value::Obj(target.clone())],
        );
        // Internal construction can share a newly allocated payload with an
        // existing function object. A minor snapshot sees only young_owner.
        old_owner.borrow_mut().call = object(&young_owner).borrow().call.clone();
        target
            .borrow_mut()
            .props
            .insert("callback", Property::builtin(young_owner.clone()));
        let weak_target = Rc::downgrade(&target);
        drop((target, young_owner));
        interp.gc_collect_young(GcCause::Explicit);
        assert!(
            weak_target.upgrade().is_some(),
            "old native owner lost young captures"
        );
        let result = interp
            .invoke(Value::Obj(old_owner.clone()), Value::Undefined, &[])
            .unwrap_or_else(|_| panic!("old owner invocation failed"));
        assert!(Rc::ptr_eq(
            object(&result),
            &weak_target.upgrade().expect("old owner's capture")
        ));
        drop((old_owner, result));
        interp.gc_collect();
        assert!(
            weak_target.upgrade().is_none(),
            "cross-generation cycle survived full GC"
        );
    }

    #[test]
    fn native_captures_foreign_heap_owner_remains_a_root_until_released() {
        let mut interp = Interp::new();
        let target = Object::new(None);
        let callback = interp.new_native_fn_with_captures(
            "foreign",
            0,
            returning_capture,
            vec![Value::Obj(target.clone())],
        );
        target
            .borrow_mut()
            .props
            .insert("callback", Property::builtin(callback.clone()));
        let weak_target = Rc::downgrade(&target);
        let foreign_interp = Interp::new();
        let foreign_owner = Object::new(None);
        foreign_owner.borrow_mut().call = object(&callback).borrow().call.clone();
        drop((target, callback));
        interp.activate_gc_heap();
        // Force the uncached full-graph walk as well as the normal collector.
        interp.gc_collect_with_edge_budget(GcCause::Explicit, 0);
        assert!(
            weak_target.upgrade().is_some(),
            "foreign owner lost its capture"
        );
        let result = interp
            .invoke(Value::Obj(foreign_owner.clone()), Value::Null, &[])
            .unwrap_or_else(|_| panic!("foreign owner's invocation failed"));
        assert!(Rc::ptr_eq(
            object(&result),
            &weak_target.upgrade().expect("foreign capture")
        ));
        drop((result, foreign_owner, foreign_interp));
        interp.gc_collect_with_edge_budget(GcCause::Explicit, 0);
        assert!(
            weak_target.upgrade().is_none(),
            "released foreign owner left a root"
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
            interp.gc_collect_young(GcCause::Explicit);
            interp.gc_collect();
            assert_eq!(interp.member_get(target, "calls")?.as_num_opt(), Some(1.));
            if matches!(captures[1], Value::Bool(true)) {
                Err(target.clone())
            } else {
                Ok(target.clone())
            }
        }
        for throws in [false, true] {
            let mut interp = Interp::new();
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
            let weak_target = Rc::downgrade(&target);
            // Exercise dispatch with no function Object owner at all. Its
            // in-flight NativeCallable lease must protect borrowed captures.
            drop((target, callback));
            let result = match interp.dispatch_native(&lease, Value::Null, &[]) {
                Ok(value) if !throws => value,
                Err(crate::interpreter::Abrupt::Throw(value)) if throws => value,
                _ => panic!("native completion changed"),
            };
            assert!(Rc::ptr_eq(
                object(&result),
                &weak_target.upgrade().expect("active capture")
            ));
            drop((lease, result));
            interp.gc_collect();
            assert!(
                weak_target.upgrade().is_none(),
                "completed native call left a root"
            );
        }
    }
}
