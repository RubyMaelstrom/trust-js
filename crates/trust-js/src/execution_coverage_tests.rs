//! Scope and negative-feedback tests supplement native route tests: collection reports only
//! before-effect region entry, with conservative Unknown on every unrepresented execution.
use super::*;
use crate::bytecode::{self, Op, Tier};
use crate::value::Callable;
use crate::{Completion, Engine};

fn function(source: &str) -> Rc<crate::ast::Function> {
    let body = crate::parser::parse_script(source, false).ok().unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &body[0] else {
        panic!("function fixture")
    };
    function.clone()
}

fn compile(source: &str) -> (Rc<crate::ast::Function>, Rc<Chunk>) {
    let function = function(source);
    let chunk = bytecode::compile(&function).expect("coverage fixture compiles");
    (function, chunk)
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("coverage fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn compiled(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let value = engine
        .interp
        .get_var(name, &engine.interp.global_env.clone())
        .ok()
        .expect("global function exists");
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func.code.get().unwrap().as_ref().unwrap().clone()
}

fn snapshot(chunk: &Chunk) -> Rc<Snapshot> {
    chunk.execution_coverage.source().unwrap().freeze().unwrap()
}

#[test]
fn disabled_collection_allocates_no_directory_bitmap_or_snapshot() {
    let _control = TestControl::new(false);
    let bytes = BYTES.with(Cell::get);
    let versions = VERSIONS.with(Cell::get);
    let (_, chunk) = compile("function subject(a){if(a)return 7;return 11;}");
    assert!(matches!(
        chunk.execution_coverage,
        Handle::Unavailable(Reason::Disabled)
    ));
    assert!(matches!(
        chunk.execution_coverage.freeze(),
        Handle::Unavailable(Reason::Disabled)
    ));
    assert!(snapshot_json().is_none());
    assert_eq!(BYTES.with(Cell::get), bytes);
    assert_eq!(VERSIONS.with(Cell::get), versions);
}

#[test]
fn exact_version_window_zero_is_unknown_and_entry_is_before_completion() {
    let _control = TestControl::new(true);
    let (_, chunk) = compile("function subject(a){if(a)throw 7;return 11;}");
    let source = chunk.execution_coverage.source().unwrap();
    let no_mark = source.freeze().unwrap();
    assert!(no_mark
        .block_starts
        .iter()
        .all(|&pc| no_mark.state_for(&chunk, pc as usize) == State::Unknown));
    source.enter_pc(0);
    let entered = source.freeze().unwrap();
    assert_eq!(entered.state_for(&chunk, 0), State::Entered);
    assert!(entered
        .block_starts
        .iter()
        .skip(1)
        .any(|&pc| entered.state_for(&chunk, pc as usize) == State::UnseenInWindow));
    assert_eq!(entered.state(source.id() + 1, 0), State::Unknown);
    assert_eq!(
        entered.state(source.id(), chunk.jit_ops().len()),
        State::Unknown
    );
    source.invalidate(Reason::AstBody);
    let invalidated = source.freeze().unwrap();
    assert_eq!(invalidated.state_for(&chunk, 0), State::Entered);
    assert!(invalidated
        .block_starts
        .iter()
        .skip(1)
        .all(|&pc| invalidated.state_for(&chunk, pc as usize) == State::Unknown));
}

#[test]
fn immutable_snapshot_lineage_is_disjoint_from_new_primary_and_inline_versions() {
    let _control = TestControl::new(true);
    let (function, primary) = compile("function subject(a){if(a)return 7;return 11;}");
    let source = primary.execution_coverage.source().unwrap();
    source.enter_pc(0);
    let optimizer = bytecode::compile_for_optimizer(&function, &primary, &[], &[]).unwrap();
    let frozen = optimizer.execution_coverage.frozen_snapshot().unwrap();
    assert_eq!(frozen.source_version, source.id());
    assert_eq!(frozen.state_for(&optimizer, 0), State::Entered);
    let unentered = frozen.block_starts[1] as usize;
    assert_eq!(
        frozen.state_for(&optimizer, unentered),
        State::UnseenInWindow
    );
    source.enter_pc(unentered);
    optimizer
        .execution_coverage
        .invalidate(Reason::OptimizedNative);
    assert_eq!(
        frozen.state_for(&optimizer, unentered),
        State::UnseenInWindow
    );
    assert_eq!(
        source.unknown.get(),
        0,
        "separate hot clone does not mutate baseline window"
    );
    let inlined =
        bytecode::compile_with_inlines(&function, &crate::fasthash::FastMap::default(), &primary)
            .unwrap();
    assert_ne!(
        inlined.execution_coverage.source_version(),
        Some(source.id())
    );
    assert_eq!(frozen.state_for(&inlined, 0), State::Unknown);
    let (_, same_text) = compile("function subject(a){if(a)return 7;return 11;}");
    assert_eq!(same_text.jit_ops().len(), primary.jit_ops().len());
    assert_eq!(frozen.state_for(&same_text, 0), State::Unknown);
}

#[test]
fn snapshot_quota_failure_and_shared_handles_release_only_at_final_owner() {
    let _control = TestControl::new(true);
    let (_, chunk) = compile("function subject(a){if(a)return 7;return 11;}");
    let source = chunk.execution_coverage.source().unwrap();
    source.enter_pc(0);
    let before = BYTES.with(Cell::get);
    {
        let _full = Reservation::new(MAX_BYTES - before, false).unwrap();
        assert!(matches!(source.freeze(), Err(Reason::Quota)));
        assert_eq!(BYTES.with(Cell::get), MAX_BYTES);
    }
    assert_eq!(BYTES.with(Cell::get), before);
    let frozen = chunk.execution_coverage.freeze();
    let shared = frozen.freeze();
    let charged = BYTES.with(Cell::get);
    assert!(charged > before);
    assert_eq!(shared.source_version(), frozen.source_version());
    drop(frozen);
    assert_eq!(BYTES.with(Cell::get), charged);
    drop(shared);
    assert_eq!(BYTES.with(Cell::get), before);
}

#[test]
fn scalar_lease_outlives_source_chunk_and_weak_directory_header() {
    let _control = TestControl::new(true);
    with_directory(|_| Ok(())).unwrap();
    let before = BYTES.with(Cell::get);
    let (_, chunk) = compile("function subject(a){return a+1;}");
    let weak = Rc::downgrade(chunk.execution_coverage.source().unwrap());
    let lease = Lease::new(chunk.execution_coverage.source().unwrap());
    drop(chunk);
    {
        let source = weak.upgrade().unwrap();
        assert_eq!(source.chunk_handles.get(), 0);
        assert_eq!(source.native_leases.get(), 1);
        assert!(BYTES.with(Cell::get) > before);
    }
    drop(lease);
    assert!(weak.upgrade().is_none());
    drop(weak);
    with_directory(|_| Ok(())).unwrap();
    assert_eq!(
        BYTES.with(Cell::get),
        before,
        "dead scalar and directory Weak header both reclaimed"
    );
}

#[test]
fn bounded_layout_denials_and_invalid_ranges_never_invent_block_zero() {
    let _control = TestControl::new(true);
    let (_, chunk) = compile("function subject(a){if(a)return 7;return 11;}");
    let source = chunk.execution_coverage.source().unwrap();
    assert!(matches!(
        source.markers(&[(0, chunk.jit_ops().len() + 1)]),
        Err(Reason::InvalidMapping)
    ));
    source.enter_pc(chunk.jit_ops().len());
    let frozen = source.freeze().unwrap();
    assert_eq!(frozen.state_for(&chunk, 0), State::Unknown);
    assert_eq!(source.unknown.get(), Reason::InvalidMapping as u32);
    assert_eq!(source.entered.iter().map(Cell::get).sum::<u8>(), 0);
    let reserved = Reservation::new(MAX_BYTES - BYTES.with(Cell::get), false).unwrap();
    let (_, denied) = compile("function another(){return 1;}");
    assert!(matches!(
        denied.execution_coverage,
        Handle::Unavailable(Reason::Quota)
    ));
    drop(reserved);
}

#[test]
fn vm_and_native_getters_proxies_and_throwing_pads_are_entered_with_empty_ics() {
    let _control = TestControl::new(true);
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var calls=0,finalized=0,token={};
            function subject(o){try{if(o) return o.value;return 99;}
                catch(e){if(e!==token)throw e;return 17;}finally{finalized++;}}
            var getter=new Proxy({get value(){calls++;return 13;}},{});
            var proxy=new Proxy({}, {get:function(){calls++;return 19;}});
            var thrown=new Proxy({get value(){calls++;throw token;}},{});
            var revoked=Proxy.revocable({},{});revoked.revoke();
            [subject(getter),subject(proxy),subject(thrown),calls,finalized].join('|');
        "#
            ),
            "13|19|17|3|3"
        );
        let chunk = compiled(&mut engine, "subject");
        let frozen = snapshot(&chunk);
        let mut property_count = 0;
        for (pc, op) in chunk.jit_ops().iter().enumerate() {
            let cache = match op {
                Op::GetProp(_, cache) | Op::GetPropLocal(_, _, cache) => Some(*cache),
                _ => None,
            };
            if let Some(cache) = cache {
                property_count += 1;
                assert_eq!(frozen.state_for(&chunk, pc), State::Entered);
                let state = unsafe {
                    (*(chunk.jit_cache_ptr(cache) as *const Cell<bytecode::IcState>)).get()
                };
                assert_eq!(
                    state.depth,
                    bytecode::IC_EMPTY,
                    "real accessor/proxy execution is independent of empty IC"
                );
            }
        }
        assert!(property_count > 0);
        if tier == Tier::Jit {
            let code = chunk
                .jit
                .get()
                .flatten()
                .expect("actual template native body");
            assert_eq!(code.test_residency_marks(false).1, 1);
        }
        assert_eq!(engine.interp.depth, 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn fresh_native_numeric_and_branch_regions_cover_independent_vm_windows() {
    let _control = TestControl::new(true);
    const SOURCE: &str = r#"
        function subject(n,flag){var sum=0;
            for(var i=0;i<n;i++){if(flag && i===3)sum+=100;else sum+=(i*3+1)|0;}
            return sum;}
        subject(41,false);
    "#;
    let mut windows = Vec::new();
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        #[cfg(target_arch = "aarch64")]
        let regions_before = crate::jit::test_executed_register_regions();
        assert_eq!(eval(&mut engine, SOURCE), "2501");
        let chunk = compiled(&mut engine, "subject");
        let frozen = snapshot(&chunk);
        if tier == Tier::Jit {
            #[cfg(target_arch = "aarch64")]
            assert!(
                crate::jit::test_executed_register_regions() > regions_before,
                "fresh native window must actually execute a register region"
            );
            assert_eq!(
                chunk
                    .jit
                    .get()
                    .flatten()
                    .unwrap()
                    .test_residency_marks(false)
                    .1,
                1
            );
            assert_eq!(
                crate::execution_coverage::test_vm_marks(&chunk),
                0,
                "fresh native region coverage cannot be supplied by VM warm-up/fallback"
            );
        }
        windows.push((chunk, frozen));
    }
    assert_eq!(
        format!("{:?}", windows[0].0.jit_ops()),
        format!("{:?}", windows[1].0.jit_ops())
    );
    assert_eq!(windows[0].1.block_starts, windows[1].1.block_starts);
    for &pc in &windows[0].1.block_starts {
        if windows[0].1.state_for(&windows[0].0, pc as usize) == State::Entered {
            assert_eq!(
                windows[1].1.state_for(&windows[1].0, pc as usize),
                State::Entered,
                "native region must conservatively contain VM-entered block {pc}"
            );
        }
    }
}

#[test]
fn checked_version_limit_and_id_overflow_restore_exact_reservations() {
    let _control = TestControl::new(true);
    with_directory(|_| Ok(())).unwrap();
    let before = BYTES.with(Cell::get);
    let existing = VERSIONS.with(Cell::get);
    let function = function("function subject(){return 1;}");
    let chunks = (existing..MAX_VERSIONS)
        .map(|_| bytecode::compile(&function).unwrap())
        .collect::<Vec<_>>();
    assert!(chunks
        .iter()
        .all(|chunk| chunk.execution_coverage.source().is_some()));
    let denied = bytecode::compile(&function).unwrap();
    assert!(matches!(
        denied.execution_coverage,
        Handle::Unavailable(Reason::VersionLimit)
    ));
    assert_eq!(VERSIONS.with(Cell::get), MAX_VERSIONS);
    drop(denied);
    drop(chunks);
    with_directory(|_| Ok(())).unwrap();
    assert_eq!(VERSIONS.with(Cell::get), existing);
    assert_eq!(BYTES.with(Cell::get), before);
    struct RestoreId(u64);
    impl Drop for RestoreId {
        fn drop(&mut self) {
            NEXT_ID.with(|id| id.set(self.0));
        }
    }
    let _restore = RestoreId(NEXT_ID.with(|id| id.replace(u64::MAX)));
    let denied = bytecode::compile(&function).unwrap();
    assert!(matches!(
        denied.execution_coverage,
        Handle::Unavailable(Reason::IdOverflow)
    ));
    assert_eq!(NEXT_ID.with(Cell::get), u64::MAX);
    assert_eq!(BYTES.with(Cell::get), before);
    assert_eq!(VERSIONS.with(Cell::get), existing);
}

#[test]
fn natural_thread_directory_teardown_releases_every_reservation() {
    struct Probe(std::sync::Arc<std::sync::Mutex<Option<usize>>>);
    impl Drop for Probe {
        fn drop(&mut self) {
            *self.0.lock().unwrap() = Some(BYTES.with(Cell::get));
        }
    }
    thread_local! {
        static AFTER_DIRECTORY: RefCell<Option<Probe>> = const { RefCell::new(None) };
    }
    let after = std::sync::Arc::new(std::sync::Mutex::new(None));
    let report = after.clone();
    std::thread::spawn(move || {
        // TLS destructors run in reverse initialization order. Keep the byte counter alive
        // after the probe, and initialize the directory after the probe so it drops first.
        assert_eq!(BYTES.with(Cell::get), 0);
        AFTER_DIRECTORY.with(|probe| *probe.borrow_mut() = Some(Probe(report)));
        let _control = TestControl::new(true);
        let (_, chunk) = compile("function subject(a){if(a)return 7;return 11;}");
        let frozen = chunk.execution_coverage.freeze();
        let shared = frozen.freeze();
        assert!(BYTES.with(Cell::get) > 0);
        drop(chunk);
        drop(frozen);
        drop(shared);
        // Leave dead Weak headers and the directory for natural TLS cleanup, not cold purge.
        assert!(BYTES.with(Cell::get) > 0);
    })
    .join()
    .expect("natural thread cleanup must not access destroyed TLS");
    assert_eq!(*after.lock().unwrap(), Some(0));
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn actual_jit_code_lease_and_generated_cell_writes_outlive_chunk_without_js_owners() {
    let _control = TestControl::new(true);
    with_directory(|_| Ok(())).unwrap();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        eval(
            &mut engine,
            "var token={};function subject(a){if(a)return 7;return 11;} subject(token);"
        ),
        "7"
    );
    let token = engine
        .interp
        .get_var("token", &engine.interp.global_env.clone())
        .ok()
        .unwrap();
    let weak_token = Rc::downgrade(token.as_obj().unwrap());
    drop(token);
    let chunk = compiled(&mut engine, "subject");
    let code = chunk.jit.get().flatten().expect("actual baseline JitCode");
    let source = chunk.execution_coverage.source().unwrap();
    let weak_source = Rc::downgrade(source);
    let initially_cold = source
        .block_starts
        .iter()
        .copied()
        .find(|&pc| {
            source.freeze().unwrap().state_for(&chunk, pc as usize) == State::UnseenInWindow
        })
        .unwrap();
    let leaf = crate::jit::CoverageMarkerLeaf::new(
        &chunk,
        &[(initially_cold as usize, initially_cold as usize + 1)],
    );
    assert!(leaf.len() > if cfg!(target_arch = "aarch64") { 4 } else { 1 });
    assert_eq!(eval(&mut engine, "token=null;"), "null");
    engine.interp.gc_collect();
    assert!(
        weak_token.upgrade().is_none(),
        "native/scalar leases retain no JavaScript argument"
    );
    drop(engine);
    drop(chunk);
    {
        let source = weak_source.upgrade().unwrap();
        assert_eq!(source.chunk_handles.get(), 0);
        assert_eq!(
            source.native_leases.get(),
            2,
            "actual body plus marker-only JitCode"
        );
        leaf.run(); // no dangling JitCtx: only the leased scalar Cell is written.
        assert_eq!(
            source
                .freeze()
                .unwrap()
                .state(source.id(), initially_cold as usize),
            State::Entered
        );
    }
    drop(code);
    assert_eq!(weak_source.upgrade().unwrap().native_leases.get(), 1);
    drop(leaf);
    assert!(weak_source.upgrade().is_none());
    drop(weak_source);
    with_directory(|_| Ok(())).unwrap();
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn disabled_native_leaf_has_only_return_and_no_scalar_lease() {
    let _control = TestControl::new(false);
    let bytes = BYTES.with(Cell::get);
    let versions = VERSIONS.with(Cell::get);
    let (_, chunk) = compile("function subject(a){if(a)return 7;return 11;}");
    let leaf = crate::jit::CoverageMarkerLeaf::new(&chunk, &[(0, 1)]);
    assert_eq!(
        leaf.len(),
        if cfg!(target_arch = "aarch64") { 4 } else { 1 }
    );
    leaf.run();
    assert_eq!(BYTES.with(Cell::get), bytes);
    assert_eq!(VERSIONS.with(Cell::get), versions);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn actual_threaded_equality_enters_remote_condition_region_in_fresh_native_window() {
    // Earlier default acyclic/register-chain lowerings may consume this valid graph.
    // Select the existing equality-only emission ablation in an isolated child, without
    // mutating process environment or weakening the physical threaded-entry assertion.
    if std::env::var("TRUST_JS_JIT_FAST").as_deref() != Ok("2") {
        let selector = "execution_coverage::tests::actual_threaded_equality_enters_remote_condition_region_in_fresh_native_window";
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([selector, "--exact", "--nocapture", "--test-threads=1"])
            .env("TRUST_JS_JIT_FAST", "2")
            .output()
            .expect("equality-only native mechanism child");
        let stdout = String::from_utf8_lossy(&child.stdout);
        let stderr = String::from_utf8_lossy(&child.stderr);
        println!("existing diagnostic ablation TRUST_JS_JIT_FAST=2; exact child selector={selector}\n{stdout}");
        assert!(
            child.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "child must actually select and pass the mechanism test: {stderr}"
        );
        return;
    }
    let _control = TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let before = crate::jit::test_coverage_threaded_equalities();
    assert_eq!(
        eval(
            &mut engine,
            "function subject(a,b,c,d){if(a===b ? a===c : c===d)return 7;return 11;}subject(1,1,1,3);"
        ),
        "7"
    );
    let chunk = compiled(&mut engine, "subject");
    let remote = chunk
        .jit_ops()
        .windows(2)
        .filter_map(|pair| match pair {
            [Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq, Op::Jump(pc)]
                if matches!(chunk.jit_ops().get(*pc as usize), Some(Op::JumpIfFalse(_))) =>
            {
                Some(*pc as usize)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        !remote.is_empty(),
        "fixture must have the actual threaded original opcode graph"
    );
    assert!(
        crate::jit::test_coverage_threaded_equalities() > before,
        "threaded fusion must actually execute"
    );
    let frozen = snapshot(&chunk);
    for pc in remote {
        assert_eq!(frozen.state_for(&chunk, pc), State::Entered);
    }
    assert_eq!(
        chunk
            .jit
            .get()
            .flatten()
            .unwrap()
            .test_residency_marks(false),
        (0, 1)
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn actual_direct_callee_entries_preserve_exact_separate_source_coverage() {
    let _control = TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let before = crate::jit::test_direct_packed_returns();
    assert_eq!(
        eval(
            &mut engine,
            r#"
        function leaf(a){try{if(a>10)return a+1;return a+2;}catch(e){return -1;}}
        function caller(n){var sum=0;for(var j=0;j<n;j++)sum+=leaf(j);return sum;}
        caller(180);
    "#
        ),
        "16301"
    );
    let leaf = compiled(&mut engine, "leaf");
    let caller = compiled(&mut engine, "caller");
    assert_ne!(
        leaf.execution_coverage.source_version(),
        caller.execution_coverage.source_version()
    );
    assert!(
        crate::jit::test_direct_packed_returns() > before + 20,
        "actual guarded direct calls must return from the only user callee"
    );
    assert_eq!(
        leaf.jit
            .get()
            .flatten()
            .unwrap()
            .test_residency_marks(false),
        (0, 1)
    );
    let leaf_snapshot = snapshot(&leaf);
    assert_eq!(leaf_snapshot.state_for(&leaf, 0), State::Entered);
    assert_eq!(
        leaf.jit_ops()
            .iter()
            .enumerate()
            .filter(|(pc, op)| matches!(op, Op::Return)
                && leaf_snapshot.state_for(&leaf, *pc) == State::Entered)
            .count(),
        2,
        "both successful return regions execute; the catch remains unobserved"
    );
    assert_eq!(leaf_snapshot.state_for(&caller, 0), State::Unknown);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn actual_optional_region_rewind_and_compilation_never_mark_execution() {
    let _control = TestControl::new(true);
    let mut engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let before = crate::jit::test_coverage_rewound_regions();
    for text in [
        "function subject(a,b){var x=a+1;if(b)x+=2;else x+=3;return x;}",
        "function subject(n,o){var x=0;for(var j=0;j<n;j++){if(j&1)x+=o.value;else x+=j;}return x;}",
        "function subject(a,o){var x=a+1;if(x>3)o.value=x;else o.value=x+1;return o.value;}",
    ] {
        let (_, chunk) = compile(text);
        let code = crate::jit::compile(&chunk, &layout, &ilayout).unwrap();
        let frozen = snapshot(&chunk);
        assert!(frozen.block_starts.iter().all(|&pc| frozen.state_for(&chunk, pc as usize) == State::Unknown));
        assert_eq!(code.test_residency_marks(false), (0, 0));
    }
    assert!(
        crate::jit::test_coverage_rewound_regions() > before,
        "fixture must exercise actual optional-region checkpoint rollback"
    );
}

#[test]
fn ast_fallback_invalidates_all_retained_compiled_versions() {
    let _control = TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        eval(
            &mut engine,
            "function callee(a){return a+1;}function subject(a){if(a)return callee(6);return 11;} subject(true);subject(true);subject(true);"
        ),
        "7"
    );
    let chunk = compiled(&mut engine, "subject");
    assert_eq!(chunk.execution_coverage.source().unwrap().unknown.get(), 0);
    // Match the established flattened-body fixture: cold callee compilation precedes
    // call-cache publication, and subsequent real entries warm the inline planning proof.
    for _ in 0..8 {
        assert_eq!(eval(&mut engine, "subject(true);"), "7");
    }
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let user = {
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user function")
        };
        user.clone()
    };
    let plan = bytecode::plan_inlines(&chunk, &user.func, user.realm, &env);
    assert!(
        !plan.is_empty(),
        "warm actual callee must supply a real splice plan"
    );
    let inline = bytecode::compile_with_inlines(&user.func, &plan, &chunk).unwrap();
    assert!(
        inline
            .jit_ops()
            .iter()
            .any(|op| matches!(op, Op::InlineGuard(..))),
        "code2 fixture must contain a real guarded inline body"
    );
    assert!(matches!(
        bytecode::run(
            &mut engine.interp,
            &inline,
            &user.env,
            crate::value::Value::Undefined,
            &[crate::value::Value::Bool(true)]
        ),
        Ok(crate::value::Value::Num(7.0))
    ));
    let optimizer = bytecode::compile_for_optimizer(&user.func, &inline, &[], &[]).unwrap();
    let frozen = optimizer.execution_coverage.frozen_snapshot().unwrap();
    let before = frozen
        .block_starts
        .iter()
        .map(|&pc| frozen.state_for(&optimizer, pc as usize))
        .collect::<Vec<_>>();
    assert!(user.func.code2.set(Some(inline.clone())).is_ok());
    assert!(user
        .func
        .optimizing_code
        .set(Some(optimizer.clone()))
        .is_ok());
    engine.set_tier(Tier::Interp);
    assert_eq!(eval(&mut engine, "subject(false);"), "11");
    for live in [&chunk, &inline] {
        let source = live.execution_coverage.source().unwrap();
        assert_eq!(
            source.unknown.get() & Reason::AstBody as u32,
            Reason::AstBody as u32
        );
        let snapshot = source.freeze().unwrap();
        assert_eq!(snapshot.state_for(live, 0), State::Entered);
        assert!(snapshot
            .block_starts
            .iter()
            .any(|&pc| snapshot.state_for(live, pc as usize) == State::Unknown));
    }
    let after = frozen
        .block_starts
        .iter()
        .map(|&pc| frozen.state_for(&optimizer, pc as usize))
        .collect::<Vec<_>>();
    assert_eq!(
        before, after,
        "AST fallback cannot mutate frozen inline admission feedback"
    );
}

#[test]
fn borrowed_coroutine_segments_never_claim_suspended_successors_entered() {
    let _control = TestControl::new(true);
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine,"function* subject(a){yield a;var b=a+1;yield b;return b+1;} var g=subject(7);g.next().value;"),"7");
        let chunk = compiled(&mut engine, "subject");
        let before = snapshot(&chunk);
        let yield_pc = chunk
            .jit_ops()
            .iter()
            .position(|op| matches!(op, Op::Yield))
            .unwrap();
        assert_eq!(before.state_for(&chunk, yield_pc), State::Entered);
        assert_ne!(before.state_for(&chunk, yield_pc + 1), State::Entered);
        assert_eq!(eval(&mut engine, "g.next().value;"), "8");
        let after = snapshot(&chunk);
        assert_eq!(after.state_for(&chunk, yield_pc + 1), State::Entered);
        assert_eq!(eval(&mut engine, "g.next().value;"), "9");
        if tier == Tier::Jit {
            assert!(
                crate::jit::TEST_NATIVE_SLICES.with(Cell::get) > 0,
                "actual borrowed native slices"
            );
        }
    }
}
