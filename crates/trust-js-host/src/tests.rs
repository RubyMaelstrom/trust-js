use super::*;
use trust_js::embed::ValueKind;

/// The Phase-1 acceptance test: an external crate registers a native global and JS calls it.
fn host_add<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let a = ctx.coerce_number(args.first().unwrap_or(&Value::Undefined))?;
    let b = ctx.coerce_number(args.get(1).unwrap_or(&Value::Undefined))?;
    Ok(Value::Num(a + b))
}

fn eval_str(engine: &mut Engine, src: &str) -> String {
    match engine.eval(src, false).expect("parse") {
        Completion::Value(v) => v,
        Completion::Throw { name, message } => panic!("uncaught {name}: {message}"),
    }
}

#[test]
fn native_global_callable_from_js() {
    let mut engine = Engine::new();
    engine.define_global("hostAdd", 2, host_add);
    assert_eq!(eval_str(&mut engine, "hostAdd(2, 40)"), "42");
    // Coercion path (user valueOf) works, and a native fn is a real function object.
    assert_eq!(
        eval_str(&mut engine, "hostAdd({ valueOf: () => 1 }, '2.5')"),
        "3.5"
    );
    assert_eq!(eval_str(&mut engine, "typeof hostAdd"), "function");
    assert_eq!(eval_str(&mut engine, "hostAdd.length"), "2");
}

#[test]
fn extension_installs_state_globals_and_namespaces() {
    struct Counter(u32);
    fn bump<'s>(
        ctx: &mut Ctx<'s>,
        _this: Value<'s>,
        _args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let c = ctx.host_mut::<Counter>().expect("state_init ran");
        c.0 += 1;
        Ok(Value::Num(c.0 as f64))
    }
    static EXT: Extension = Extension {
        name: "counter",
        globals: ops!["bump" (0) => bump],
        namespaces: &[("counterNs", ops!["bumpToo" (0) => bump])],
        state_init: Some(|state| state.put(Counter(0))),
        js_init: None,
        js_init_snapshot: None,
    };
    let mut engine = Engine::new();
    install(&mut engine, std::slice::from_ref(&EXT));
    assert_eq!(eval_str(&mut engine, "bump(); bump()"), "2");
    assert_eq!(eval_str(&mut engine, "counterNs.bumpToo()"), "3");
}

#[test]
fn native_fn_survives_the_bytecode_tier() {
    // The embed hooks must work on every execution tier (a native call from JIT/bytecode
    // frames goes through the same Callable::Native dispatch, but verify, don't assume).
    let mut engine = Engine::new();
    engine.set_tier(trust_js::Tier::Bytecode);
    engine.set_tier_threshold(0);
    engine.define_global("hostAdd", 2, host_add);
    assert_eq!(
        eval_str(
            &mut engine,
            "function f() { let s = 0; for (let i = 0; i < 1000; i++) s = hostAdd(s, 1); return s } f(); f()"
        ),
        "1000"
    );
}

#[test]
fn call_function_reenters_the_engine() {
    let mut engine = Engine::new();
    match engine
        .eval("globalThis.cb = (x) => x * 2; 0", false)
        .unwrap()
    {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
    engine.scope(|cx| {
        let g = cx.global_this();
        let cb = cx.get_member(&g, "cb").map_err(|_| ()).expect("cb defined");
        let out = cx
            .call_function(&cb, Value::Undefined, &[Value::Num(21.0)])
            .map_err(|_| "js threw")
            .expect("call ok");
        assert!(matches!((out).kind(), ValueKind::Num(n) if n == 42.0));
    });
}

#[test]
fn microtask_hooks_drive_promises() {
    let mut engine = Engine::new();
    // Queue a reaction without draining: eval() drains at the end, so queue one from a
    // native call instead... simplest honest check: after eval, nothing pending; then a
    // manually enqueued callback via call_function leaves the queue drainable by hand.
    engine
        .eval(
            "globalThis.done = 0; Promise.resolve(1).then(v => done = v)",
            false,
        )
        .unwrap();
    // Engine::eval already ran the checkpoint: the reaction fired.
    assert_eq!(eval_str(&mut engine, "done"), "1");
    assert!(!engine.has_pending_jobs());
    // Now queue a job by calling a promise-creating function through call_function (which
    // does NOT run a microtask checkpoint) and drain it step by step.
    engine
        .eval(
            "globalThis.queue = () => { Promise.resolve(2).then(v => done = v) }",
            false,
        )
        .unwrap();
    engine.scope(|cx| {
        let g = cx.global_this();
        let queue = cx.get_member(&g, "queue").map_err(|_| ()).unwrap();
        cx.call_function(&queue, Value::Undefined, &[])
            .map_err(|_| "js threw")
            .unwrap();
    });
    assert!(engine.has_pending_jobs());
    assert!(engine.run_one_job());
    engine.run_microtasks();
    assert!(!engine.has_pending_jobs());
    assert_eq!(eval_str(&mut engine, "done"), "2");
}

#[test]
fn resource_table_roundtrip() {
    let mut table = ResourceTable::default();
    let rid = table.add(String::from("a file handle"));
    assert!(table.has(rid));
    assert_eq!(*table.get::<String>(rid).unwrap(), "a file handle");
    assert!(table.get::<u32>(rid).is_none(), "downcast is type-checked");
    assert!(table.close(rid).is_some());
    assert!(!table.has(rid));
    assert!(table.is_empty());
}

#[test]
fn threadpool_completions_arrive() {
    let (tx, rx) = std::sync::mpsc::channel();
    let pool = ThreadPool::new(4, tx);
    for id in 0..16u64 {
        pool.spawn_blocking(id, move || {
            Box::new(id * 2) as Box<dyn std::any::Any + Send>
        });
    }
    let mut seen = std::collections::HashMap::new();
    for _ in 0..16 {
        let done = rx.recv().expect("completion");
        seen.insert(
            done.task,
            *done
                .result
                .expect("task succeeds")
                .downcast::<u64>()
                .unwrap(),
        );
    }
    assert_eq!(seen.len(), 16);
    assert!((0..16).all(|id| seen[&id] == id * 2));
    drop(pool); // joins workers; must not deadlock
}

#[test]
fn threadpool_contains_panics_and_keeps_the_worker_alive() {
    let (tx, rx) = std::sync::mpsc::channel();
    let pool = ThreadPool::with_limits(1, 4, 128 << 10, tx);
    pool.spawn_blocking(1, || panic!("contained worker panic"));
    pool.spawn_blocking(2, || Box::new(42u64));

    let first = rx.recv().unwrap();
    assert_eq!(first.task, 1);
    assert!(first
        .result
        .err()
        .unwrap()
        .message
        .contains("contained worker panic"));
    let second = rx.recv().unwrap();
    assert_eq!(second.task, 2);
    assert_eq!(*second.result.unwrap().downcast::<u64>().unwrap(), 42);
}

#[test]
fn threadpool_queue_is_bounded_and_reports_overload() {
    let (completion_tx, completion_rx) = std::sync::mpsc::channel();
    let pool = ThreadPool::with_limits(1, 1, 128 << 10, completion_tx);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    pool.spawn_blocking(1, move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        Box::new(())
    });
    started_rx.recv().unwrap();
    pool.spawn_blocking(2, || Box::new(()));
    pool.spawn_blocking(3, || Box::new(()));

    let overloaded = completion_rx.recv().unwrap();
    assert_eq!(overloaded.task, 3);
    assert!(overloaded
        .result
        .err()
        .unwrap()
        .message
        .contains("queue is full"));
    release_tx.send(()).unwrap();
    assert!(completion_rx.recv().unwrap().result.is_ok());
    assert!(completion_rx.recv().unwrap().result.is_ok());
}

#[test]
fn threadpool_drop_has_a_bounded_grace_period() {
    let (completion_tx, _completion_rx) = std::sync::mpsc::channel();
    let pool = ThreadPool::with_limits(1, 1, 128 << 10, completion_tx);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    pool.spawn_blocking(1, move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        Box::new(())
    });
    started_rx.recv().unwrap();
    let start = std::time::Instant::now();
    drop(pool);
    assert!(start.elapsed() < std::time::Duration::from_millis(750));
    release_tx.send(()).unwrap();
}

#[test]
fn dedicated_executor_bounds_threads_and_contains_panics() {
    let (completion_tx, completion_rx) = std::sync::mpsc::channel();
    let executor = DedicatedExecutor::new(1, 128 << 10, completion_tx, TaskCanceller::default());
    let handle = executor.handle();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    handle.run_blocking(1, move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        Box::new(())
    });
    started_rx.recv().unwrap();
    handle.run_blocking(2, || Box::new(()));
    let limited = completion_rx.recv().unwrap();
    assert_eq!(limited.task, 2);
    assert!(limited
        .result
        .err()
        .unwrap()
        .message
        .contains("thread limit"));
    release_tx.send(()).unwrap();
    assert!(completion_rx.recv().unwrap().result.is_ok());

    handle.run_blocking(3, || panic!("dedicated panic"));
    let panicked = completion_rx.recv().unwrap();
    assert_eq!(panicked.task, 3);
    assert!(panicked
        .result
        .err()
        .unwrap()
        .message
        .contains("dedicated panic"));

    drop(executor);
    handle.run_blocking(4, || Box::new(()));
    let stopped = completion_rx.recv().unwrap();
    assert_eq!(stopped.task, 4);
    assert!(stopped
        .result
        .err()
        .unwrap()
        .message
        .contains("shutting down"));
}

#[test]
fn task_registry_cancellation_prevents_queued_work_from_starting() {
    fn decode<'s>(
        _ctx: &mut Ctx<'s>,
        _payload: Box<dyn std::any::Any + Send>,
    ) -> Result<Vec<Value<'s>>, Value<'s>> {
        Ok(vec![])
    }

    let (completion_tx, completion_rx) = std::sync::mpsc::channel();
    let pool = ThreadPool::with_limits(1, 2, 128 << 10, completion_tx);
    let mut registry = TaskRegistry::with_canceller(pool.canceller());
    let mut engine = Engine::new();
    let mut inert =
        || engine.scope(|cx| Handle::new(cx, &Value::Undefined).expect("a primitive handle"));
    let first = registry.register(inert(), None, decode);
    let second = registry.register(inert(), None, decode);
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    pool.spawn_blocking(first, move || {
        started_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        Box::new(())
    });
    started_rx.recv().unwrap();
    pool.spawn_blocking(second, || panic!("cancelled task must not run"));
    assert!(registry.cancel(second).is_some());
    release_tx.send(()).unwrap();
    assert_eq!(completion_rx.recv().unwrap().task, first);
    assert!(completion_rx
        .recv_timeout(std::time::Duration::from_millis(100))
        .is_err());
}

/// `queue(callback, arg)`: queue `callback(arg)` on the [`CallbackQueue`] and register it as an
/// in-flight task's success and failure callbacks; returns the task's id.
fn queue_both<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    fn decode<'s>(
        _: &mut Ctx<'s>,
        _: Box<dyn std::any::Any + Send>,
    ) -> Result<Vec<Value<'s>>, Value<'s>> {
        Ok(vec![])
    }
    let callback = args.first().cloned().unwrap_or(Value::Undefined);
    CallbackQueue::enqueue(ctx, &callback, &args[1..])?;
    let id = TaskRegistry::register_callbacks(ctx, &callback, Some(&callback), decode)?;
    Ok(Value::Num(id as f64))
}

/// Queued callbacks and their arguments, and in-flight tasks' callbacks, are held by handles in
/// host state: collections keep them though no script value does, and they run afterwards.
#[test]
fn queued_and_task_callbacks_survive_collections_in_host_state() {
    let mut engine = Engine::new();
    engine.op_state().put(TaskRegistry::default());
    engine.define_global("queue", 2, queue_both);
    let id = eval_str(
        &mut engine,
        "globalThis.log = [];
         queue(function (x) { log.push(x === undefined ? 'settled' : x.tag); },
               {tag: ['queued', 'arg'].join(' ')})",
    );
    for _ in 0..2 {
        eval_str(
            &mut engine,
            "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
        );
        engine.collect_garbage_at_idle();
    }
    let (callback, args) = engine
        .host_mut::<CallbackQueue>()
        .expect("queue installed")
        .queue
        .pop_front()
        .expect("a queued callback");
    engine.scope(|cx| {
        let (callback, args) = (value(cx, &callback), values(cx, &args));
        assert!(cx.call_function(&callback, Value::Undefined, &args).is_ok());
    });
    let entry = engine
        .host_mut::<TaskRegistry>()
        .expect("registry installed")
        .take(id.parse().expect("a task id"))
        .expect("the task is in flight");
    for settle in [Some(&entry.on_ok), entry.on_err.as_ref()] {
        engine.scope(|cx| {
            let settle = value(cx, settle.expect("both callbacks"));
            assert!(cx.call_function(&settle, Value::Undefined, &[]).is_ok());
        });
    }
    assert_eq!(
        eval_str(&mut engine, "log.join()"),
        "queued arg,settled,settled"
    );
}
