#![allow(dead_code)]
#[path = "../../../crates/trust-js/src/agent_access.rs"]
mod agent_access;

#[cfg(test)]
mod tests {
    use super::agent_access::{Access, Owned, RealmId};
    use std::cell::Cell;
    use std::ptr::NonNull;
    use std::rc::Rc;

    struct Vm {
        id: RealmId,
        coordinator: Rc<Access<Vm>>,
        value: u64,
        child: Option<Owned<Vm>>,
        deaths: Rc<Cell<usize>>,
    }
    impl Drop for Vm {
        fn drop(&mut self) {
            self.deaths.set(self.deaths.get() + 1);
        }
    }

    fn vm(id: RealmId, coordinator: &Rc<Access<Vm>>, deaths: &Rc<Cell<usize>>) -> Vm {
        Vm {
            id,
            coordinator: coordinator.clone(),
            value: 0,
            child: None,
            deaths: deaths.clone(),
        }
    }

    // Mirrors Interp::with_access: metadata is copied before suspend, and the callback's
    // reference is derived from the published pointer. Suspension inside it refreshes that
    // callback's CURRENT borrow before accessing any other active allocation.
    fn execute<R>(current: &mut Vm, operation: impl FnOnce(&mut Vm) -> R) -> R {
        // Exactly the production structure: coordinator metadata comes from CURRENT T before
        // its last-use publication, not from an independent argument avoiding T's borrow.
        let access = current.coordinator.clone();
        let id = current.id;
        access.suspend(id, current, |pointer| {
            operation(unsafe { &mut *pointer.as_ptr() })
        })
    }

    fn switch<R>(current: &mut Vm, target: RealmId, operation: impl FnOnce(&mut Vm) -> R) -> R {
        let access = current.coordinator.clone();
        let id = current.id;
        access.suspend(id, current, |_| {
            let target = access.resolve(target).unwrap();
            execute(unsafe { &mut *target.as_ptr() }, operation)
        })
    }

    // Marking holds WHOLE-NODE references simultaneously, as the production Marking client
    // does, rather than reading individual fields and immediately forgetting the reference.
    fn mark(current: &Vm) -> u64 {
        let access = current.coordinator.clone();
        let nodes: Vec<&Vm> = access
            .shared_snapshot(current.id, current)
            .iter()
            .map(|&(_, pointer)| unsafe { &*pointer })
            .collect();
        let result = nodes.iter().map(|node| node.value).sum();
        std::hint::black_box(&nodes);
        drop(nodes);
        result
    }

    fn sweep(current: &mut Vm) {
        let access = current.coordinator.clone();
        let id = current.id;
        access.suspend(id, current, |_| {
            for (_, pointer) in access.snapshot() {
                unsafe {
                    (*pointer.as_ptr()).value += 1;
                }
            }
        });
    }

    #[test]
    fn current_borrow_refresh_across_parent_child_parent_and_same_target() {
        let deaths = Rc::new(Cell::new(0));
        let access = Rc::new(Access::default());
        let mut root = vm(RealmId::ROOT, &access, &deaths);
        let child = access.fresh_id();
        root.child = Some(access.adopt(child, RealmId::ROOT, vm(child, &access, &deaths)));
        execute(&mut root, |root| {
            root.value = 2;
            switch(root, child, |child_vm| {
                child_vm.value = 3;
                assert_eq!(mark(child_vm), 5);
                switch(child_vm, RealmId::ROOT, |root| {
                    root.value += 5;
                    switch(root, child, |child_vm| {
                        child_vm.value += 7;
                        assert_eq!(mark(child_vm), 17);
                        sweep(child_vm);
                        assert_eq!(child_vm.value, 11);
                    });
                    assert_eq!(root.value, 8);
                });
                assert_eq!(child_vm.value, 11);
            });
            assert_eq!(root.value, 8);
        });
        assert!(access.resolve(RealmId::ROOT).is_none());
        drop(root);
        assert_eq!(deaths.get(), 2);
    }

    #[test]
    fn root_move_between_entries_and_shared_phase_before_sweep() {
        let deaths = Rc::new(Cell::new(0));
        let access = Rc::new(Access::default());
        let mut root = vm(RealmId::ROOT, &access, &deaths);
        execute(&mut root, |root| {
            root.value = 4;
            assert_eq!(mark(root), 4);
        });
        let mut moved = (19, root);
        execute(&mut moved.1, |root| {
            sweep(root);
            assert_eq!(root.value, 5);
        });
        assert!(access.resolve(RealmId::ROOT).is_none());
    }

    #[test]
    fn quarantine_finishes_all_visits_before_raw_owners_drop() {
        let deaths = Rc::new(Cell::new(0));
        let access = Rc::new(Access::default());
        let mut root = vm(RealmId::ROOT, &access, &deaths);
        let child = access.fresh_id();
        root.child = Some(access.adopt(child, RealmId::ROOT, vm(child, &access, &deaths)));
        execute(&mut root, |root| {
            let id = root.id;
            access.suspend(id, root, |_| {
                let pointers = access.snapshot();
                let mut quarantine = Vec::new();
                for (id, pointer) in pointers {
                    let vm = unsafe { &mut *pointer.as_ptr() };
                    if id == RealmId::ROOT {
                        quarantine.push(vm.child.take().unwrap());
                    }
                    vm.value += 1;
                    assert_eq!(deaths.get(), 0);
                }
                drop(quarantine);
            });
            assert_eq!(deaths.get(), 1);
            assert!(access.resolve(child).is_none());
        });
    }

    #[test]
    fn unwind_restores_nested_lineage_and_other_agent_is_independent() {
        let deaths = Rc::new(Cell::new(0));
        let a = Rc::new(Access::default());
        let b = Rc::new(Access::default());
        let mut root = vm(RealmId::ROOT, &a, &deaths);
        let mut independent = vm(RealmId::ROOT, &b, &deaths);
        let child = a.fresh_id();
        root.child = Some(a.adopt(child, RealmId::ROOT, vm(child, &a, &deaths)));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(&mut root, |root| {
                switch(root, child, |child_vm| {
                    execute(&mut independent, |other| {
                        other.value = 31;
                        sweep(other);
                    });
                    assert_eq!(mark(child_vm), 0);
                    panic!("test unwind");
                })
            });
        }));
        assert!(caught.is_err());
        assert!(a.resolve(RealmId::ROOT).is_none());
        assert!(!a.active(child));
        execute(&mut root, |root| {
            sweep(root);
            assert_eq!(root.value, 1);
        });
        assert_eq!(independent.value, 32);
    }

    #[test]
    fn local_host_callback_restores_after_independent_agent_and_unwind() {
        fn callback<R>(
            current: &mut Vm,
            active: &Cell<u64>,
            operation: impl FnOnce(&mut Vm) -> R,
        ) -> R {
            super::agent_access::suspend_local(current, |pointer| {
                struct Restore<'a>(NonNull<Vm>, &'a Cell<u64>);
                impl Drop for Restore<'_> {
                    fn drop(&mut self) {
                        self.1.set(unsafe { (*self.0.as_ptr()).value });
                    }
                }
                let _restore = Restore(pointer, active);
                operation(unsafe { &mut *pointer.as_ptr() })
            })
        }
        let deaths = Rc::new(Cell::new(0));
        let a = Rc::new(Access::default());
        let b = Rc::new(Access::default());
        let mut first = vm(RealmId::ROOT, &a, &deaths);
        let mut second = vm(RealmId::ROOT, &b, &deaths);
        first.value = 11;
        second.value = 22;
        let active = Cell::new(11);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            callback(&mut first, &active, |first| {
                assert_eq!(mark(first), 11);
                execute(&mut second, |second| {
                    active.set(second.value);
                    sweep(second);
                });
                assert_eq!(active.get(), 22);
                first.value = 12;
                panic!("host callback unwind");
            });
        }));
        assert!(result.is_err());
        assert_eq!(active.get(), 12);
        assert_eq!(first.value, 12);
        assert_eq!(second.value, 23);
        assert!(a.resolve(RealmId::ROOT).is_none());
        callback(&mut first, &active, |first| {
            sweep(first);
            active.set(99);
        });
        assert_eq!(active.get(), 13);
    }

    #[test]
    fn local_callbacks_compose_with_outer_lease_down_up_and_collection() {
        fn local<R>(current: &mut Vm, operation: impl FnOnce(&mut Vm) -> R) -> R {
            super::agent_access::suspend_local(current, |pointer| {
                operation(unsafe { &mut *pointer.as_ptr() })
            })
        }
        for throw in [false, true] {
            let deaths = Rc::new(Cell::new(0));
            let access = Rc::new(Access::default());
            let mut root = vm(RealmId::ROOT, &access, &deaths);
            root.value = 1;
            let child = access.fresh_id();
            root.child = Some(access.adopt(child, RealmId::ROOT, vm(child, &access, &deaths)));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute(&mut root, |root| {
                    local(root, |root| {
                        root.value += 1;
                        switch(root, child, |child_vm| {
                            local(child_vm, |child_vm| {
                                child_vm.value = 3;
                                switch(child_vm, RealmId::ROOT, |root| {
                                    assert_eq!(mark(root), 5);
                                    sweep(root);
                                    if throw {
                                        panic!("nested local callback unwind");
                                    }
                                });
                                child_vm.value += 1;
                            });
                        });
                        root.value += 1;
                    });
                });
            }));
            assert_eq!(result.is_err(), throw);
            assert_eq!(root.value, if throw { 3 } else { 4 });
            assert!(access.resolve(RealmId::ROOT).is_none());
            assert!(!access.active(child));
            execute(&mut root, |root| {
                let expected = if throw { 7 } else { 9 };
                assert_eq!(mark(root), expected);
                sweep(root);
            });
        }
    }

    // Explicit optional negative control: Miri must reject an older-reference access between
    // publication and traversal. It is not part of the ordinary passing harness inventory.
    #[test]
    #[cfg(feature = "stale-suspension")]
    fn stale_suspension_is_a_miri_negative_control() {
        let deaths = Rc::new(Cell::new(0));
        let access = Rc::new(Access::default());
        let mut root = vm(RealmId::ROOT, &access, &deaths);
        let pointer = NonNull::from(&mut root);
        let _scope = unsafe { access.enter(RealmId::ROOT, pointer) };
        root.value = 7;
        unsafe {
            (*access.resolve(RealmId::ROOT).unwrap().as_ptr()).value += 1;
        }
        std::hint::black_box(root.value);
    }
}
