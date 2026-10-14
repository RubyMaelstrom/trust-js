use super::*;

thread_local! {
    static BEFORE_RETIREMENT_DROP: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
}

pub(super) fn before_retirement_drop() {
    let hook = BEFORE_RETIREMENT_DROP.with(|hook| hook.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

fn assert_published_registration_invariant() {
    REGISTRY.with(|registry| {
        let registry = registry.borrow();
        let Some(directory) = registry.entries.as_ref() else {
            return;
        };
        let mut cursor = 0;
        let mut seen = 0;
        for _ in 0..MAX_RESIDENT_ENTRIES {
            if let Some(index) = directory.inspect(&mut cursor) {
                if let Some(state) = directory.get(index).upgrade() {
                    let registered = matches!(
                        *state.code.borrow(),
                        CodeState::Pending | CodeState::Resident(_)
                    );
                    #[cfg(feature = "optimizing-jit")]
                    let registered =
                        registered || matches!(*state.code.borrow(), CodeState::Retiring(_));
                    assert!(registered);
                    assert_eq!(state.registration.get(), index as u32);
                }
                seen += 1;
                if seen == directory.len {
                    break;
                }
            }
        }
        assert_eq!(seen, directory.len);
    });
}

/// Each unit test gets a separate TLS registry without dropping the old one
/// inside a RefCell borrow. No live JS activation crosses this test boundary.
struct IsolatedRegistry(Option<Registry>);
impl IsolatedRegistry {
    fn new(limit: usize) -> Self {
        let replacement = Registry {
            entry_limit: Some(limit),
            ..Registry::default()
        };
        Self(Some(REGISTRY.with(|registry| {
            std::mem::replace(&mut *registry.borrow_mut(), replacement)
        })))
    }
}
impl Drop for IsolatedRegistry {
    fn drop(&mut self) {
        let old = self.0.take().unwrap();
        let removed = REGISTRY.with(|registry| std::mem::replace(&mut *registry.borrow_mut(), old));
        assert_eq!(
            removed.pending, 0,
            "all scoped permits settled before fixture exit"
        );
        drop(removed);
    }
}

#[test]
fn cold_slot_has_no_sidecar_or_registry_allocation() {
    let _registry = IsolatedRegistry::new(4);
    let slot = NativeCodeSlot::new();
    assert!(slot.get().is_none());
    assert!(slot.ready_to_compile());
    assert_eq!(slot.retained_metadata_bytes(), 0);
    assert!(slot.state.get().is_none());
    REGISTRY.with(|registry| assert!(registry.borrow().entries.is_none()));
    assert_eq!(
        std::mem::size_of::<NativeCodeSlot>(),
        std::mem::size_of::<usize>()
    );
}

#[test]
fn pending_permits_reserve_actual_cells_and_nested_attempts_cannot_steal_them() {
    let _registry = IsolatedRegistry::new(2);
    let outer = NativeCodeSlot::new();
    let first = outer.begin_compile().unwrap();
    assert!(outer.get().is_none());
    assert!(!outer.ready_to_compile());
    assert!(outer.begin_compile().is_none());
    let inner = NativeCodeSlot::new();
    let second = inner.begin_compile().unwrap();
    let refused = NativeCodeSlot::new();
    assert!(refused.begin_compile().is_none());
    reclaim(usize::MAX);
    REGISTRY.with(|registry| {
        let registry = registry.borrow();
        assert_eq!(registry.pending, 2);
        assert_eq!(registry.len(), 2);
    });
    drop(second);
    assert!(inner.ready_to_compile());
    // Rollback resets shared metadata denial immediately. This is a new slot,
    // not a bypass of the denied function's existing 256-query cooldown.
    let replacement = NativeCodeSlot::new();
    replacement.begin_compile().unwrap().commit(None);
    first.commit(None);
    assert!(matches!(outer.get(), Some(None)));
    REGISTRY.with(|registry| {
        let registry = registry.borrow();
        assert_eq!(registry.pending, 0);
        assert_eq!(registry.len(), 0);
        assert!(registry.entries.is_none());
    });
}

#[test]
fn cheap_maintenance_never_upgrades_or_allocates_for_live_entries() {
    let _registry = IsolatedRegistry::new(2);
    let first = NativeCodeSlot::new();
    let second = NativeCodeSlot::new();
    // Hooks stand at the exact boundary after publication/cleanup and before
    // any retired owners might invoke an external destructor.
    BEFORE_RETIREMENT_DROP
        .with(|hook| *hook.borrow_mut() = Some(Box::new(assert_published_registration_invariant)));
    let first_permit = first.begin_compile().unwrap();
    let second_permit = second.begin_compile().unwrap();
    let first_count = Rc::strong_count(first.state.get().unwrap());
    let second_count = Rc::strong_count(second.state.get().unwrap());
    let mut retired = Retirement::default();
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry.cursor = 0;
        registry.collect(Goal::DeadOnly, COLD_MAINTENANCE_STEPS, &mut retired);
    });
    assert_eq!(Rc::strong_count(first.state.get().unwrap()), first_count);
    assert_eq!(Rc::strong_count(second.state.get().unwrap()), second_count);
    assert_eq!(retired.states.capacity(), 0);
    assert_eq!(retired.code.capacity(), 0);
    assert_eq!(retired.weak.capacity(), 0);
    assert_eq!(retired.pages.capacity(), 0);
    assert_eq!(retired.directories.capacity(), 0);
    retired.finish();
    BEFORE_RETIREMENT_DROP
        .with(|hook| *hook.borrow_mut() = Some(Box::new(assert_published_registration_invariant)));
    first_permit.commit(None);
    BEFORE_RETIREMENT_DROP
        .with(|hook| *hook.borrow_mut() = Some(Box::new(assert_published_registration_invariant)));
    drop(second_permit);
}

#[test]
fn unwind_rolls_back_once_and_structural_unavailability_is_distinct() {
    let _registry = IsolatedRegistry::new(1);
    let slot = NativeCodeSlot::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _permit = slot.begin_compile().unwrap();
        panic!("synthetic compiler unwind");
    }));
    assert!(result.is_err());
    assert!(slot.ready_to_compile());
    REGISTRY.with(|registry| assert_eq!(registry.borrow().pending, 0));
    slot.begin_compile().unwrap().commit(None);
    assert!(matches!(slot.get(), Some(None)));
    assert!(!slot.ready_to_compile());
    assert!(slot.begin_compile().is_none());
}

#[test]
fn metadata_pressure_scans_are_bounded_across_distinct_new_slots() {
    let _registry = IsolatedRegistry::new(1);
    let held = NativeCodeSlot::new();
    let permit = held.begin_compile().unwrap();
    let before = REGISTRY.with(|registry| registry.borrow().steps);
    let denied: Vec<_> = (0..32)
        .map(|_| {
            let slot = NativeCodeSlot::new();
            assert!(slot.begin_compile().is_none());
            assert_eq!(slot.state.get().unwrap().cooldown.get(), 256);
            slot
        })
        .collect();
    let after = REGISTRY.with(|registry| registry.borrow().steps);
    assert!(after - before <= MAX_CLOCK_VISITS + 32 * COLD_MAINTENANCE_STEPS);
    assert_eq!(REGISTRY.with(|registry| registry.borrow().len()), 1);
    drop(permit);
    drop(denied);
}

#[test]
fn fixed_pages_have_an_exact_capacity_bound_and_no_shrink_regrow_overshoot() {
    assert_eq!(
        std::mem::size_of::<Option<Weak<SlotState>>>(),
        std::mem::size_of::<usize>(),
        "fingerprint requested payload rather than assuming an opaque allocator header"
    );
    assert_eq!(
        std::mem::size_of::<Page>(),
        directory::PAGE_ENTRIES * std::mem::size_of::<Option<Weak<SlotState>>>()
            + 4 * std::mem::size_of::<u64>()
    );
    let mut directory = Directory::default();
    let owner = Rc::new(SlotState::default());
    // Low-level directory storage, deliberately one shared Weak target; the
    // production Registry separately enforces one reservation per SlotState.
    for index in 0..MAX_RESIDENT_ENTRIES {
        assert_eq!(directory.insert(&owner), index);
    }
    assert_eq!(directory.allocated_pages, directory::PAGE_COUNT);
    let bound =
        std::mem::size_of::<Directory>() + directory::PAGE_COUNT * std::mem::size_of::<Page>();
    assert_eq!(directory.allocated_bytes(), bound);
    let mut retired = Retirement::default();
    for index in 30_000..MAX_RESIDENT_ENTRIES {
        directory.remove(index, &mut retired);
    }
    retired.finish();
    for index in 30_000..MAX_RESIDENT_ENTRIES {
        assert_eq!(directory.insert(&owner), index);
    }
    assert_eq!(directory.allocated_bytes(), bound);
    let mut retired = Retirement::default();
    for index in 0..MAX_RESIDENT_ENTRIES {
        directory.remove(index, &mut retired);
    }
    assert_eq!(directory.allocated_pages, 0);
    assert_eq!(
        directory.allocated_bytes(),
        std::mem::size_of::<Directory>()
    );
    retired.finish();
}

#[test]
fn permit_pins_its_state_after_the_last_slot_owner_disappears() {
    let _registry = IsolatedRegistry::new(1);
    let slot = NativeCodeSlot::new();
    let permit = slot.begin_compile().unwrap();
    let state = Rc::downgrade(slot.state.get().unwrap());
    drop(slot);
    assert!(state.upgrade().is_some());
    assert_eq!(REGISTRY.with(|registry| registry.borrow().pending), 1);
    permit.commit(None);
    assert!(state.upgrade().is_none());
    assert_eq!(REGISTRY.with(|registry| registry.borrow().pending), 0);
    assert_eq!(REGISTRY.with(|registry| registry.borrow().len()), 0);
}

#[test]
fn dead_weak_backing_is_censused_and_retires_only_outside_the_borrow() {
    let _registry = IsolatedRegistry::new(4);
    let state = Rc::new(SlotState::default());
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        let index = registry
            .entries
            .get_or_insert_with(Box::default)
            .insert(&state);
        state.registration.set(index as u32);
    });
    drop(state);
    let census = shared_metadata_json();
    assert!(census.contains("\"dead_entries\":1"));
    assert!(census.contains("\"scope\":\"current_thread_shared_cache\""));
    assert!(census.contains("\"evictions\":0"));
    assert!(census.contains(&format!(
        "\"dead_slot_value_payload_bytes\":{}",
        std::mem::size_of::<SlotState>()
    )));
    assert!(census.contains("\"included_in_managed_memory\":false"));
    let payload = std::mem::size_of::<Registry>()
        + std::mem::size_of::<Directory>()
        + std::mem::size_of::<Page>()
        + std::mem::size_of::<SlotState>();
    assert!(census.contains(&format!("\"known_requested_bytes\":{payload}")));
    let mut retired = Retirement::default();
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry.collect(Goal::DeadOnly, COLD_MAINTENANCE_STEPS, &mut retired);
        assert_eq!(registry.len(), 0);
        assert!(registry.entries.is_none());
        assert_eq!(retired.weak.len(), 1);
        assert_eq!(retired.pages.len(), 1);
        assert_eq!(retired.directories.len(), 1);
    });
    assert!(REGISTRY.with(|registry| registry.try_borrow_mut().is_ok()));
    retired.finish();
    assert!(shared_metadata_json().contains("\"directory_and_page_payload_bytes\":0"));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn code(referenced: bool) -> Rc<JitCode> {
    // Never entered. The existing native cache tests exercise actual calls,
    // constructors, generator/await resumption and reentrant native returns.
    let executable = super::super::ExecutableBuffer::from_bytes(&[0; 16]).unwrap();
    Rc::new(JitCode {
        mem: executable.as_ptr() as *mut u8,
        len: executable.len(),
        pc_offsets: Vec::new(),
        max_stack: 0,
        needs_global: false,
        resume_depths: Vec::new(),
        entry_kind: super::super::NativeEntryKind::FreshFrame,
        osr_entry_depths: Vec::new(),
        executable,
        residency: Box::new(CodeResidency {
            active: Cell::new(0),
            referenced: Cell::new(u8::from(referenced)),
        }),
        #[cfg(feature = "optimizing-jit")]
        call_stubs: Vec::new(),
        #[cfg(feature = "optimizing-jit")]
        optimizing_diagnostics: None,
        #[cfg(feature = "optimizing-jit")]
        execution_coverage: None,
    })
}

#[cfg(all(
    feature = "optimizing-jit",
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn shared_call_code_is_credited_only_when_its_last_owner_retires() {
    let mut engine = crate::Engine::new();
    let body = crate::parser::parse_script("function subject(f){return f(1);}", false)
        .ok()
        .unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &body[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let interp = crate::interpreter::interp_layout(&mut engine.interp);
    let stub = super::super::call_stub::prepare(&chunk, &values, &interp)
        .pop()
        .expect("ARM64 stub");
    let shared_bytes = stub.executable_bytes();
    let weak = Rc::downgrade(&stub);
    let mut first = code(false);
    let mut second = code(false);
    Rc::get_mut(&mut first)
        .unwrap()
        .call_stubs
        .push(stub.clone());
    Rc::get_mut(&mut second).unwrap().call_stubs.push(stub);
    let first_bytes = first.len;
    let second_bytes = second.len;
    let mut retired = Retirement::default();
    assert_eq!(retired.push_code(first), first_bytes);
    assert_eq!(retired.push_code(second), second_bytes + shared_bytes);
    assert!(
        weak.upgrade().is_some(),
        "retirement cannot unmap inside the registry borrow"
    );
    retired.finish();
    assert!(
        weak.upgrade().is_none(),
        "weak directory does not retain executable storage"
    );
}

#[cfg(all(
    feature = "optimizing-jit",
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn shared_stub_reclamation_does_not_evict_an_unneeded_body() {
    let _registry = IsolatedRegistry::new(3);
    let mut engine = crate::Engine::new();
    let body = crate::parser::parse_script("function subject(f){return f(1);}", false)
        .ok()
        .unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &body[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let interp = crate::interpreter::interp_layout(&mut engine.interp);
    let stub = super::super::call_stub::prepare(&chunk, &values, &interp)
        .pop()
        .unwrap();
    let shared_bytes = stub.executable_bytes();
    let weak = Rc::downgrade(&stub);
    let first = NativeCodeSlot::new();
    let second = NativeCodeSlot::new();
    let spare = NativeCodeSlot::new();
    let mut first_code = code(false);
    let mut second_code = code(false);
    let required = first_code.len + second_code.len + shared_bytes;
    Rc::get_mut(&mut first_code)
        .unwrap()
        .call_stubs
        .push(stub.clone());
    Rc::get_mut(&mut second_code).unwrap().call_stubs.push(stub);
    first.begin_compile().unwrap().commit(Some(first_code));
    second.begin_compile().unwrap().commit(Some(second_code));
    spare.begin_compile().unwrap().commit(Some(code(false)));
    REGISTRY.with(|registry| registry.borrow_mut().cursor = 0);
    reclaim(required);
    assert!(first.get().is_none());
    assert!(second.get().is_none());
    assert!(
        spare.get().flatten().is_some(),
        "the last shared owner already satisfies the byte request"
    );
    assert!(weak.upgrade().is_none());
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn metadata_victims_respect_active_marks_owned_leases_and_existing_cooldowns() {
    let _registry = IsolatedRegistry::new(1);
    let slot = NativeCodeSlot::new();
    slot.begin_compile().unwrap().commit(Some(code(true)));
    let mut maintenance = Retirement::default();
    let contents = slot.state.get().unwrap().code.borrow_mut();
    REGISTRY.with(|registry| {
        let mut registry = registry.borrow_mut();
        registry.cursor = 0;
        registry.collect(Goal::DeadOnly, COLD_MAINTENANCE_STEPS, &mut maintenance);
    });
    assert_eq!(
        maintenance.states.capacity(),
        0,
        "live Resident is neither borrowed nor upgraded"
    );
    drop(contents);
    maintenance.finish();
    let lease = slot.get().flatten().unwrap();
    assert!(
        NativeCodeSlot::new().begin_compile().is_none(),
        "owned lease refuses eviction"
    );
    lease.residency.active.set(1);
    drop(lease);
    // Remove only shared retry delay to independently exercise protection.
    REGISTRY.with(|registry| registry.borrow_mut().metadata_retry = 0);
    assert!(
        NativeCodeSlot::new().begin_compile().is_none(),
        "native active frame refuses eviction"
    );
    slot.get().flatten().unwrap().residency.active.set(0);
    REGISTRY.with(|registry| registry.borrow_mut().metadata_retry = 0);
    let replacement = NativeCodeSlot::new();
    let pending = replacement
        .begin_compile()
        .expect("inactive code can free METADATA capacity");
    assert!(slot.get().is_none());
    assert_eq!(slot.state.get().unwrap().cooldown.get(), 8);
    for _ in 0..8 {
        assert!(!slot.ready_to_compile());
    }
    assert!(slot.ready_to_compile());
    drop(pending);
    slot.state.get().unwrap().evictions.set(u8::MAX);
    slot.begin_compile().unwrap().commit(Some(code(false)));
    reclaim(usize::MAX);
    assert_eq!(slot.state.get().unwrap().cooldown.get(), 256);
    assert_eq!(slot.state.get().unwrap().evictions.get(), u8::MAX);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn clock_keeps_recent_code_while_evicting_an_unreferenced_peer() {
    let _registry = IsolatedRegistry::new(2);
    let recent = NativeCodeSlot::new();
    let cold = NativeCodeSlot::new();
    recent.begin_compile().unwrap().commit(Some(code(true)));
    cold.begin_compile().unwrap().commit(Some(code(true)));
    cold.get().flatten().unwrap().residency.referenced.set(0);
    REGISTRY.with(|registry| registry.borrow_mut().cursor = 0);
    reclaim(1);
    assert!(
        recent.get().flatten().is_some(),
        "referenced entry receives its second chance"
    );
    assert!(
        cold.get().is_none(),
        "unreferenced peer satisfies the byte request first"
    );
    assert_eq!(recent.state.get().unwrap().evictions.get(), 0);
    assert_eq!(cold.state.get().unwrap().cooldown.get(), 8);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn resident_reads_do_not_enter_registry_or_allocate_and_epoch_precedes_retirement() {
    let _registry = IsolatedRegistry::new(2);
    let slot = NativeCodeSlot::new();
    slot.begin_compile().unwrap().commit(Some(code(false)));
    let weak = Rc::downgrade(&slot.get().flatten().unwrap());
    let before = REGISTRY.with(|registry| {
        let registry = registry.borrow_mut();
        // A held mutable registry borrow would panic if get/ready touched it.
        for _ in 0..100 {
            assert!(slot.get().flatten().is_some());
            assert!(!slot.ready_to_compile());
        }
        registry.steps
    });
    assert_eq!(REGISTRY.with(|registry| registry.borrow().steps), before);
    let mut retired = Retirement::default();
    REGISTRY.with(|registry| {
        registry
            .borrow_mut()
            .collect(Goal::Bytes(1), MAX_CLOCK_VISITS, &mut retired)
    });
    assert!(
        weak.upgrade().is_some(),
        "mapping remains alive until borrow is released"
    );
    let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    let checked = Rc::new(Cell::new(false));
    let observed = checked.clone();
    let before_drop = weak.clone();
    BEFORE_RETIREMENT_DROP.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            assert!(REGISTRY.with(|registry| registry.try_borrow_mut().is_ok()));
            assert!(
                crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed) > epoch
            );
            assert!(
                before_drop.upgrade().is_some(),
                "epoch changes BEFORE mapping owner is dropped"
            );
            let nested = NativeCodeSlot::new();
            nested.begin_compile().unwrap().commit(None);
            observed.set(true);
        }))
    });
    retired.finish();
    assert!(checked.get());
    assert!(weak.upgrade().is_none());
    assert!(crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed) > epoch);
    assert!(REGISTRY.with(|registry| registry.try_borrow_mut().is_ok()));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn admission_denial_precedes_actual_ordinary_continuation_and_osr_compilation() {
    const CHILD: &str = "TRUST_JS_TEST_METADATA_ADMISSION_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "jit::cache::tests::admission_denial_precedes_actual_ordinary_continuation_and_osr_compilation", "--test-threads=1"])
            .env(CHILD, "1").env("TRUST_JS_PERF_METRICS", "1").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use crate::bytecode::Tier;
    use crate::value::Value;
    use crate::{Completion, Engine};
    use std::sync::atomic::Ordering::Relaxed;
    fn eval(e: &mut Engine, source: &str) -> String {
        match e.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
    fn call(e: &mut Engine, name: &str) -> Value {
        let env = e.interp.global_env.clone();
        let f = e.interp.get_var(name, &env).ok().expect("fixture function");
        e.interp
            .call(f, Value::Undefined, &[])
            .ok()
            .expect("function call")
    }
    let isolated = IsolatedRegistry::new(0);
    let mut e = Engine::new();
    e.set_tier(Tier::Interp);
    eval(
        &mut e,
        "function ordinary(){return 41;}function* generator(){yield 7;return 8;}",
    );
    e.set_tier(Tier::Jit);
    e.set_tier_threshold(0);
    let before = super::super::PERF_COMPILE_ATTEMPTS.load(Relaxed);
    assert!(matches!(call(&mut e, "ordinary"), Value::Num(41.0)));
    let generator = call(&mut e, "generator");
    let next = e
        .interp
        .get_member_key(&generator, "next")
        .ok()
        .expect("next method");
    let item = e
        .interp
        .call(next, generator, &[])
        .ok()
        .expect("continuation fallback");
    assert!(matches!(
        e.interp.get_member_key(&item, "value"),
        Ok(Value::Num(7.0))
    ));
    e.set_tier_threshold(32);
    assert_eq!(
        eval(&mut e, "var total=0;for(var n=0;n<1000;n++)total+=n;total;"),
        "499500"
    );
    assert_eq!(
        super::super::PERF_COMPILE_ATTEMPTS.load(Relaxed),
        before,
        "no compiler entry, not compile-then-discard, on any persistent admission path"
    );
    assert_eq!(REGISTRY.with(|registry| registry.borrow().len()), 0);
    drop(isolated);
    // Fresh normal code proves the platform/compiler was not globally disabled.
    e.set_tier(Tier::Interp);
    eval(&mut e, "function admitted(){return 42;}");
    e.set_tier(Tier::Jit);
    e.set_tier_threshold(0);
    assert!(matches!(call(&mut e, "admitted"), Value::Num(42.0)));
    assert!(super::super::PERF_COMPILE_ATTEMPTS.load(Relaxed) > before);
}
