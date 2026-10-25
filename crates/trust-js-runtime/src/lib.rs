//! trust-js-runtime — the event loop that turns the TRust JS engine into a runtime.
//!
//! One [`Runtime`] = one engine (one realm) + the installed extensions (timers, console,
//! process) + bounded executors for finite and long-lived blocking work. The engine is `!Send`, so the thread that
//! creates the runtime owns it; everything asynchronous funnels back to that thread as
//! either a queued JS callback or an mpsc [`TaskCompletion`].
//!
//! A loop turn, in order (see [`Runtime::run_to_completion`]):
//! 1. drain **microtasks** (promise reactions),
//! 2. run queued **callbacks** (`process.nextTick`, `setImmediate`),
//! 3. fire due **timers**,
//! 4. dispatch ready **completions** from the threadpool,
//! then block on the completion channel until the next timer deadline (or indefinitely if
//! only completions remain). The loop exits when nothing is pending anywhere. A readiness
//! reactor (epoll/kqueue) would need raw syscalls and stays out unless explicitly authorized;
//! threadpool + completions is libuv's own fs strategy and covers everything we host today.

use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use trust_js_host::ValueKind;

use trust_js_host::{
    install, CallbackQueue, DedicatedExecutor, Engine, EvalError, Handle, RuntimeInterrupt,
    TaskCompletion, TaskDecoder, TaskRegistry, ThreadPool, Value,
};

mod console;
mod esm;
mod jsx;
mod process;
mod worker;

pub use console::{describe_error, render_value, ConsoleOut};
pub use trust_js_host::{Completion, Ctx};

/// Conservative defaults; deployments can tune them without rebuilding. All values are clamped
/// so an accidental environment setting cannot recreate unbounded queues/stacks/threads.
const DEFAULT_POOL_SIZE: usize = 4;
const DEFAULT_WORKER_STACK_SIZE: usize = 1 << 20;
const DEFAULT_DEDICATED_THREAD_LIMIT: usize = 256;
const DEFAULT_DEDICATED_STACK_SIZE: usize = 512 << 10;

pub struct Runtime {
    engine: Engine,
    _dedicated: DedicatedExecutor,
    pool: ThreadPool,
    completions: mpsc::Receiver<TaskCompletion>,
    /// The error-reporting shims (see `Runtime::new`): `(error) -> suppressed` for the global
    /// `onerror` convention and `(promise, reason) -> suppressed` for `onunhandledrejection`.
    /// No JavaScript reference keeps them alive once their globals are deleted: the handles do.
    fire_error: Handle,
    fire_rejection: Handle,
}

/// The callback batch the loop is running. A pass takes everything queued so far out of the
/// [`CallbackQueue`] (callbacks queued meanwhile wait for the next pass); each stays held here,
/// like the queue, until it starts, so a collection inside one callback cannot reclaim the rest.
#[derive(Default)]
struct RunningCallbacks(std::collections::VecDeque<(Handle, Vec<Handle>)>);

/// Unhandled rejections being reported, taken from the engine in one batch: each (promise,
/// reason) pair stays held here until its report starts, since `onunhandledrejection` runs
/// script (and may collect) between reports.
#[derive(Default)]
struct ReportingRejections(std::collections::VecDeque<(Handle, Handle)>);

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    /// An engine with the runtime globals installed: timers, streaming `console`, minimal
    /// `process`, `queueMicrotask`.
    pub fn new() -> Runtime {
        Self::new_with_interrupt(Default::default())
    }

    /// Construct a runtime with a control handle already shared with its owner. Dedicated workers
    /// use this to make termination effective even during realm/bootstrap startup.
    pub fn new_with_interrupt(interrupt: Arc<RuntimeInterrupt>) -> Runtime {
        let (tx, rx) = mpsc::channel();
        let pool_size = env_usize("TRUST_JS_BLOCKING_THREADS", DEFAULT_POOL_SIZE, 1, 64);
        let queue_capacity = env_usize(
            "TRUST_JS_BLOCKING_QUEUE_CAPACITY",
            pool_size * 64,
            1,
            65_536,
        );
        let worker_stack = env_usize(
            "TRUST_JS_BLOCKING_STACK_BYTES",
            DEFAULT_WORKER_STACK_SIZE,
            64 << 10,
            64 << 20,
        );
        let pool = ThreadPool::with_limits(pool_size, queue_capacity, worker_stack, tx.clone());
        let canceller = pool.canceller();
        let dedicated = DedicatedExecutor::new(
            env_usize(
                "TRUST_JS_DEDICATED_THREAD_LIMIT",
                DEFAULT_DEDICATED_THREAD_LIMIT,
                1,
                4096,
            ),
            env_usize(
                "TRUST_JS_DEDICATED_STACK_BYTES",
                DEFAULT_DEDICATED_STACK_SIZE,
                64 << 10,
                64 << 20,
            ),
            tx.clone(),
            canceller.clone(),
        );
        // Install the runtime's finite trusted bootstrap before attaching the externally
        // cancellable handle. Otherwise an early Worker.terminate() can interrupt an extension's
        // own initialization and make the installer mistake host control flow for broken glue.
        let mut engine = Engine::new();
        // Substrate first: fs's js_init runs during install and its ops need these.
        engine.op_state().put(pool.handle());
        // Dedicated-thread completions for unbounded-blocking work (child stdio) that must not
        // occupy a shared pool worker.
        engine.op_state().put(dedicated.handle());
        engine
            .op_state()
            .put(TaskRegistry::with_canceller(canceller));
        install(
            &mut engine,
            &[
                trust_js_timers::extension(),
                console::extension(),
                process::extension(),
                trust_js_fs::extension(),
                trust_js_web::extension(),
                // Last: node's glue wraps the fs global, Buffer uses TextEncoder (web), and
                // require() calls process.cwd().
                trust_js_node::extension(),
                worker::extension(),
            ],
        );
        process::install_data_props(&mut engine);
        // queueMicrotask, via the promise queue the engine already has. Close enough to spec
        // for v0 (a thrown callback error becomes an unhandled rejection, not a reported
        // exception); a native microtask hook can replace it if that gap ever matters.
        engine
            .eval(
                "globalThis.queueMicrotask = (cb) => {
                    if (typeof cb !== 'function')
                        throw new TypeError('queueMicrotask expects a function');
                    Promise.resolve().then(cb);
                };",
                false,
            )
            .expect("shim parses");
        // The HTML error-reporting globals (WinterTC Minimum Common API §5.2): `onerror` /
        // `onunhandledrejection` global event-handler properties and `reportError`. The fire
        // helpers return whether the default report is suppressed (`onerror` returning `true`;
        // `unhandledrejection`'s `event.preventDefault()`); the loop's uncaught/rejection
        // reporting consults them through handles grabbed (and then unglobaled) below. A throw
        // inside a handler never re-enters it — the original error still default-reports.
        engine
            .eval(
                r#"
                globalThis.onerror = null;
                globalThis.onunhandledrejection = null;
                globalThis.__trust_js_fire_error = function (error) {
                    const h = globalThis.onerror;
                    if (typeof h !== 'function') return false;
                    let message = '';
                    try {
                        message =
                            error instanceof Error
                                ? `Uncaught ${error.name}: ${error.message}`
                                : `Uncaught ${String(error)}`;
                    } catch {}
                    try {
                        return h.call(globalThis, message, '', 0, 0, error) === true;
                    } catch {
                        return false;
                    }
                };
                globalThis.__trust_js_fire_rejection = function (promise, reason) {
                    const h = globalThis.onunhandledrejection;
                    if (typeof h !== 'function') return false;
                    let prevented = false;
                    const event = {
                        type: 'unhandledrejection',
                        promise,
                        reason,
                        cancelable: true,
                        preventDefault() { prevented = true; },
                        get defaultPrevented() { return prevented; },
                    };
                    try {
                        h.call(globalThis, event);
                    } catch {}
                    return prevented;
                };
                {
                    const fire = globalThis.__trust_js_fire_error;
                    const report = console.__reportUncaught;
                    delete console.__reportUncaught;
                    globalThis.reportError = function reportError(e) {
                        if (arguments.length === 0)
                            throw new TypeError('reportError requires at least 1 argument');
                        if (!fire(e)) report(e);
                    };
                }
                "#,
                false,
            )
            .expect("error-reporting shim parses");
        let (fire_error, fire_rejection) = engine.scope(|cx| {
            let global = cx.global_this();
            let mut shim = |name: &str| {
                let shim = cx
                    .get_member(&global, name)
                    .unwrap_or_else(|_| panic!("error-reporting shim installed"));
                Handle::new(cx, &shim).expect("the engine's own function")
            };
            (
                shim("__trust_js_fire_error"),
                shim("__trust_js_fire_rejection"),
            )
        });
        engine
            .eval(
                "delete globalThis.__trust_js_fire_error; delete globalThis.__trust_js_fire_rejection;",
                false,
            )
            .expect("shim cleanup");
        engine.set_interrupt_handle(interrupt);
        let state = engine.op_state();
        state.put(RunningCallbacks::default());
        state.put(ReportingRejections::default());
        Runtime {
            engine,
            _dedicated: dedicated,
            pool,
            completions: rx,
            fire_error,
            fire_rejection,
        }
    }

    /// The engine, for embedder access beyond script evaluation (defining globals, etc.).
    pub fn engine(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Run `path` as a CommonJS program entry (`require.main === module`, with `__dirname`/
    /// `__filename`/`require` in scope), then loop to quiescence. `Err` is the rendered
    /// uncaught error. This is what the CLI uses for `trust-js-cli file.js`.
    pub fn run_main(&mut self, path: &str) -> Result<(), String> {
        let result = self.start_main(path);
        self.run_to_completion();
        result
    }

    /// Start `path` as the CJS main module WITHOUT pumping the macrotask loop (only the top-level
    /// microtask checkpoint runs). Worker threads use this: the caller arms its message inbox
    /// first and then drives the loop itself. `Err` is the rendered uncaught error.
    pub(crate) fn start_main(&mut self, path: &str) -> Result<(), String> {
        // A CJS script can still dynamic-`import()`: give the engine the ESM loader and resolve
        // bare relative specifiers against the entry file.
        let loader = esm::make_loader(self.builtin_modules());
        self.engine.set_module_loader_attrs(loader);
        if let Ok(abs) = std::fs::canonicalize(path) {
            self.engine.set_import_base(&abs.to_string_lossy());
        }
        self.engine.scope(|cx| {
            let global = cx.global_this();
            let run_main = cx
                .get_member(&global, "__runMain")
                .map_err(|_| "node runtime not installed".to_string())?;
            let path = cx.string(path.to_string());
            let result = cx.call_function_interruptible(&run_main, Value::Undefined, &[path]);
            if let Err(reason) = cx.run_microtasks_interruptible() {
                return Err(reason.message().to_string());
            }
            match result {
                Ok(_) => Ok(()),
                Err(EvalError::Throw(error)) => Err(describe_error(cx, &error)),
                Err(EvalError::Interrupted(reason)) => Err(reason.message().to_string()),
            }
        })
    }

    /// Run `path` as an ES module: its `import` graph resolves against disk + `node_modules`
    /// (and the `node:` builtins), then the loop runs to quiescence so top-level `await`,
    /// timers, and I/O settle. `Err` is the rendered uncaught error.
    pub fn run_module(&mut self, path: &str) -> Result<(), String> {
        let source =
            std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        // A `.jsx` entry is lowered to plain JS before the engine parses it.
        let source = if path.ends_with(".jsx") {
            jsx::transform(&source).map_err(|e| format!("JSX transform failed for {path}: {e}"))?
        } else {
            source
        };
        let key = std::fs::canonicalize(path)
            .unwrap_or_else(|_| std::path::PathBuf::from(path))
            .to_string_lossy()
            .into_owned();
        let loader = esm::make_loader(self.builtin_modules());
        let result = self.engine.eval_module_attrs(&source, &key, loader);
        self.run_to_completion();
        match result {
            Ok(Completion::Value(_)) => Ok(()),
            Ok(Completion::Throw { name, message }) => Err(if name.is_empty() {
                message
            } else {
                format!("{name}: {message}")
            }),
            Err(e) => Err(format!("SyntaxError: {} (line {})", e.message, e.line)),
        }
    }

    /// Evaluate a Worker entry `source` (a module when `is_module`, else a classic script) WITHOUT
    /// running the loop — the caller arms the message inbox first, then pumps the loop itself so
    /// the worker stays alive for messages. `base` seeds relative-import resolution. `Err` is the
    /// rendered load/parse/top-level error.
    pub fn eval_worker_entry(
        &mut self,
        source: &str,
        base: &str,
        is_module: bool,
    ) -> Result<(), String> {
        let loader = esm::make_loader(self.builtin_modules());
        self.engine.set_module_loader_attrs(loader);
        self.engine.set_import_base(base);
        let result = if is_module {
            let loader = esm::make_loader(self.builtin_modules());
            self.engine.eval_module_attrs(source, base, loader)
        } else {
            self.engine.eval(source, false)
        };
        // Drain the microtask checkpoint from top-level code, but not the macrotask loop.
        if let Err(reason) = self.engine.run_microtasks_interruptible() {
            return Err(reason.message().to_string());
        }
        match result {
            Ok(Completion::Value(_)) => Ok(()),
            Ok(Completion::Throw { name, message }) => Err(if name.is_empty() {
                message
            } else {
                format!("{name}: {message}")
            }),
            Err(e) => Err(format!("SyntaxError: {} (line {})", e.message, e.line)),
        }
    }

    /// Like [`run_to_completion`](Runtime::run_to_completion), but for a Worker: it does NOT exit
    /// when idle (the armed message inbox keeps it alive), and it returns promptly when `stop` is
    /// set — a cooperative terminate that also drops any still-pending timers. The blocking wait
    /// polls `stop` on a short interval so a `terminate()` from another thread is noticed even
    /// with no message or timer due.
    pub fn run_worker_loop(&mut self, stop: &std::sync::atomic::AtomicBool) {
        use std::sync::atomic::Ordering;
        let poll = Duration::from_millis(50);
        loop {
            if stop.load(Ordering::SeqCst) {
                return;
            }
            if self.engine.run_microtasks_interruptible().is_err() {
                return;
            }
            self.report_unhandled_rejections();
            loop {
                let mut progressed = false;
                self.start_callback_batch();
                while let Some((cb, args)) = self.next_batch_callback() {
                    progressed = true;
                    self.fire_held(&cb, &args);
                }
                for task in self.take_due_timers() {
                    progressed = true;
                    self.fire_timer(task);
                }
                while let Ok(done) = self.completions.try_recv() {
                    progressed = true;
                    self.dispatch(done);
                }
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                if !progressed {
                    break;
                }
            }
            if stop.load(Ordering::SeqCst) || self.idle() {
                return;
            }
            let wait = match self.next_timer_deadline() {
                Some(deadline) => {
                    let now = Instant::now();
                    if deadline <= now {
                        continue;
                    }
                    (deadline - now).min(poll)
                }
                None => poll,
            };
            if let Ok(done) = self.completions.recv_timeout(wait) {
                self.dispatch(done);
            }
        }
    }

    /// Pull the JS-precomputed synthetic ESM source for each `node:` builtin out of the engine
    /// (the loader can't enumerate a builtin's exports from Rust; see the node module glue).
    fn builtin_modules(&mut self) -> esm::BuiltinModules {
        self.engine.scope(Self::builtin_modules_in)
    }

    /// [`Runtime::builtin_modules`] in the scope `ctx`.
    fn builtin_modules_in(ctx: &mut Ctx<'_>) -> esm::BuiltinModules {
        let global = ctx.global_this();
        let mut map = std::collections::HashMap::new();
        let names = ctx
            .get_member(&global, "__builtinNames")
            .ok()
            .and_then(|v| ctx.coerce_string(&v).ok())
            .map(|s| s.to_string())
            .unwrap_or_default();
        let sources = ctx.get_member(&global, "__esmBuiltinSources").ok();
        if let Some(sources) = sources {
            for name in names.split(',').filter(|s| !s.is_empty()) {
                let key = format!("node:{name}");
                if let Ok(src) = ctx.get_member(&sources, &key) {
                    if let Ok(src) = ctx.coerce_string(&src) {
                        map.insert(key, src.to_string());
                    }
                }
            }
        }
        esm::BuiltinModules(map)
    }

    /// Evaluate a script, then run the event loop until quiescent — timers fired, spawned
    /// work completed, promise queue empty.
    pub fn eval(&mut self, src: &str) -> Result<Completion, trust_js_host::ParseError> {
        let result = self.engine.eval(src, false);
        self.run_to_completion();
        result
    }

    /// Spawn blocking work on the pool; when it finishes, `decode` turns its payload into
    /// arguments and `callback`, held by a handle of this runtime's engine, runs on the loop
    /// thread. This is the pattern async fs ops will use from inside native fns (via the
    /// `SpawnHandle`/`TaskRegistry` in `OpState`).
    pub fn spawn_blocking(
        &mut self,
        work: impl FnOnce() -> Box<dyn std::any::Any + Send> + Send + 'static,
        callback: Handle,
        decode: TaskDecoder,
    ) {
        let registry = self
            .engine
            .host_mut::<TaskRegistry>()
            .expect("installed in new()");
        let id = registry.register(callback, None, decode);
        self.pool.spawn_blocking(id, work);
    }

    /// Run the loop until nothing is pending: no microtasks, no queued callbacks, no live
    /// timers, no in-flight tasks.
    pub fn run_to_completion(&mut self) {
        loop {
            // Run everything already runnable. Each JS entry is followed by a microtask
            // checkpoint, matching the "after every macrotask" model.
            if self.engine.run_microtasks_interruptible().is_err() {
                return;
            }
            self.report_unhandled_rejections();
            loop {
                let mut progressed = false;
                self.start_callback_batch();
                while let Some((cb, args)) = self.next_batch_callback() {
                    progressed = true;
                    self.fire_held(&cb, &args);
                }
                for task in self.take_due_timers() {
                    progressed = true;
                    self.fire_timer(task);
                }
                while let Ok(done) = self.completions.try_recv() {
                    progressed = true;
                    self.dispatch(done);
                }
                if !progressed {
                    break;
                }
            }

            if self.idle() {
                self.engine.collect_garbage_at_idle();
                // Collection may enqueue a FinalizationRegistry cleanup job. It is a future JS
                // job, never an interruption of the task whose unreachable target was found.
                if self.engine.has_pending_jobs() {
                    continue;
                }
                return;
            }

            // Blocked: only a timer deadline or a task completion can make progress now.
            match self.next_timer_deadline() {
                Some(deadline) => {
                    let now = Instant::now();
                    if deadline > now {
                        if let Ok(done) = self.completions.recv_timeout(deadline - now) {
                            self.dispatch(done);
                        }
                    }
                }
                None => match self.completions.recv() {
                    Ok(done) => self.dispatch(done),
                    // The pool is gone (unreachable while `self.pool` lives); nothing can
                    // ever complete, so pending tasks are abandoned rather than spun on.
                    Err(_) => return,
                },
            }
        }
    }

    fn idle(&mut self) -> bool {
        if self.engine.has_pending_jobs() {
            return false;
        }
        let state = self.engine.op_state();
        let callbacks_queued = state
            .get::<CallbackQueue>()
            .is_some_and(|q| !q.queue.is_empty());
        let timers_pending = state
            .get::<trust_js_timers::Timers>()
            .is_some_and(|t| t.has_pending());
        let tasks_pending = state
            .get::<TaskRegistry>()
            .is_some_and(|r| r.has_ref_pending());
        !callbacks_queued && !timers_pending && !tasks_pending
    }

    /// Start a pass's callback batch: move everything queued so far into [`RunningCallbacks`].
    fn start_callback_batch(&mut self) {
        let state = self.engine.op_state();
        let queued = match state.get_mut::<CallbackQueue>() {
            Some(q) if !q.queue.is_empty() => std::mem::take(&mut q.queue),
            _ => return,
        };
        state
            .get_mut::<RunningCallbacks>()
            .expect("installed in new()")
            .0
            .extend(queued);
    }

    /// The batch's next callback. It leaves the batch as it starts, its handles held by the
    /// caller until the call returns.
    fn next_batch_callback(&mut self) -> Option<(Handle, Vec<Handle>)> {
        self.engine.host_mut::<RunningCallbacks>()?.0.pop_front()
    }

    fn take_due_timers(&mut self) -> Vec<trust_js_timers::TimerTask> {
        let now = Instant::now();
        match self.engine.host_mut::<trust_js_timers::Timers>() {
            Some(t) => t.take_due(now),
            None => Vec::new(),
        }
    }

    /// The next timer deadline, or an earlier Atomics.waitAsync timeout, whose promise resolves
    /// at the checkpoint after it (a pending wait alone does not keep the loop alive).
    fn next_timer_deadline(&mut self) -> Option<Instant> {
        let timer = self
            .engine
            .host_mut::<trust_js_timers::Timers>()
            .and_then(|timers| timers.next_deadline());
        match (timer, self.engine.next_atomics_wait_deadline()) {
            (Some(timer), Some(wait)) => Some(timer.min(wait)),
            (timer, wait) => timer.or(wait),
        }
    }

    /// Settle one completed task: decode the payload, then run the success callback — or the
    /// failure one (a promise's reject) when the decoder says the work failed.
    fn dispatch(&mut self, done: TaskCompletion) {
        let entry = self
            .engine
            .host_mut::<TaskRegistry>()
            .and_then(|r| r.take(done.task));
        let Some(entry) = entry else {
            return; // cancelled while in flight
        };
        let fire_error = &self.fire_error;
        // Whether a callback ran, and if so whether it was interrupted. The entry's handles hold
        // its callbacks until it is dropped, after the call.
        let fired = self.engine.scope(|cx| {
            let result = match done.result {
                Ok(payload) => (entry.decode)(cx, payload),
                Err(failure) => Err(cx.make_error("Error", failure.message)),
            };
            match result {
                Ok(args) => {
                    let on_ok = trust_js_host::value(cx, &entry.on_ok);
                    Some(call_reporting(cx, fire_error, &on_ok, &args))
                }
                Err(e) => match &entry.on_err {
                    Some(reject) => {
                        let reject = trust_js_host::value(cx, reject);
                        Some(call_reporting(
                            cx,
                            fire_error,
                            &reject,
                            std::slice::from_ref(&e),
                        ))
                    }
                    None => {
                        report_uncaught(cx, fire_error, &e);
                        None
                    }
                },
            }
        });
        if fired == Some(false) {
            self.checkpoint();
        }
        self.checkpoint();
    }

    /// A callback entry for a callback and arguments host state held, which their handles hold
    /// until the call returns: call, report an uncaught throw, then the microtask checkpoint.
    fn fire_held(&mut self, callback: &Handle, args: &[Handle]) {
        let fire_error = &self.fire_error;
        let interrupted = self.engine.scope(|cx| {
            let callback = trust_js_host::value(cx, callback);
            let args = trust_js_host::values(cx, args);
            call_reporting(cx, fire_error, &callback, &args)
        });
        if !interrupted {
            self.checkpoint();
        }
    }

    /// The microtask checkpoint after a JS entry, then the unhandled rejections' reports.
    fn checkpoint(&mut self) {
        if self.engine.run_microtasks_interruptible().is_err() {
            return;
        }
        self.report_unhandled_rejections();
    }

    /// Run one HTML timer task. Interval reinitialization and nesting-state teardown happen after
    /// the handler (including exception reporting) but before the microtask checkpoint, matching
    /// HTML §8.7's timer task algorithm and the event-loop task boundary.
    fn fire_timer(&mut self, task: trust_js_timers::TimerTask) {
        let valid = self
            .engine
            .host_mut::<trust_js_timers::Timers>()
            .is_some_and(|timers| timers.begin_task(&task));
        if !valid {
            return;
        }

        let fire_error = &self.fire_error;
        self.engine.scope(|cx| {
            let thrown = match &task.handler {
                trust_js_timers::TimerHandler::Function(callback) => {
                    let this = cx.global_this();
                    // The task's handles hold the handler and arguments until it is dropped.
                    let callback = trust_js_host::value(cx, callback);
                    let args = trust_js_host::values(cx, &task.args);
                    match cx.call_function_interruptible(&callback, this, &args) {
                        Ok(_) => None,
                        Err(EvalError::Throw(error)) => Some(error),
                        Err(EvalError::Interrupted(_)) => None,
                    }
                }
                trust_js_timers::TimerHandler::Code(code) => {
                    match cx.eval_value_interruptible(code) {
                        Ok(Ok(_)) => None,
                        Ok(Err(EvalError::Throw(error))) => Some(error),
                        Ok(Err(EvalError::Interrupted(_))) => None,
                        Err(error) => Some(cx.make_error(
                            "SyntaxError",
                            format!("{} (line {})", error.message, error.line),
                        )),
                    }
                }
            };
            if let Some(error) = thrown {
                report_uncaught(cx, fire_error, &error);
            }
        });

        if let Some(timers) = self.engine.host_mut::<trust_js_timers::Timers>() {
            timers.finish_task(&task, Instant::now());
        }
        self.checkpoint();
    }

    /// Report promises rejected without a handler (Node prints `Uncaught (in promise) …`). Called
    /// after each microtask checkpoint; a rejection handled in the same checkpoint won't appear.
    fn report_unhandled_rejections(&mut self) {
        let taken = self.engine.take_unhandled_rejections_full();
        if taken.is_empty() {
            return;
        }
        self.reporting_rejections().0.extend(taken);
        while let Some(pair) = self.reporting_rejections().0.pop_front() {
            // The pair's handles hold it until this report ends.
            let fire_rejection = &self.fire_rejection;
            let interrupted = self.engine.scope(|cx| {
                let (promise, reason) = (
                    trust_js_host::value(cx, &pair.0),
                    trust_js_host::value(cx, &pair.1),
                );
                // The global `onunhandledrejection` handler runs first; `event.preventDefault()`
                // suppresses the default report.
                let fire = trust_js_host::value(cx, fire_rejection);
                match (cx.call_function_interruptible(&fire, Value::Undefined, &[promise, reason]))
                    .map(|__value| (__value).kind())
                {
                    Ok(ValueKind::Bool(true)) => return false,
                    Err(EvalError::Interrupted(_)) => return true,
                    _ => {}
                }
                let text = console::describe_error(cx, &reason);
                console::write_err_line(cx, format!("Uncaught (in promise) {text}"));
                false
            });
            if interrupted {
                // An interruption abandons the rest of the batch.
                self.reporting_rejections().0.clear();
                return;
            }
        }
    }

    fn reporting_rejections(&mut self) -> &mut ReportingRejections {
        self.engine
            .host_mut::<ReportingRejections>()
            .expect("installed in new()")
    }
}

/// Call a loop-fired callback with no receiver, reporting an uncaught throw; whether a host
/// interruption stopped it.
fn call_reporting<'s>(
    ctx: &mut Ctx<'s>,
    fire_error: &Handle,
    callback: &Value<'s>,
    args: &[Value<'s>],
) -> bool {
    match ctx.call_function_interruptible(callback, Value::Undefined, args) {
        Ok(_) => false,
        Err(EvalError::Throw(error)) => {
            report_uncaught(ctx, fire_error, &error);
            false
        }
        Err(EvalError::Interrupted(_)) => true,
    }
}

/// An exception escaped a loop-fired callback. Node prints and keeps the loop alive (we don't
/// model `process.on('uncaughtException')`/exit semantics yet).
fn report_uncaught<'s>(ctx: &mut Ctx<'s>, fire_error: &Handle, error: &Value<'s>) {
    // HTML "report an exception": the global `onerror` handler runs first; returning `true`
    // suppresses the default report.
    let fire = trust_js_host::value(ctx, fire_error);
    match (ctx.call_function_interruptible(&fire, Value::Undefined, std::slice::from_ref(error)))
        .map(|__value| (__value).kind())
    {
        Ok(ValueKind::Bool(true)) | Err(EvalError::Interrupted(_)) => return,
        _ => {}
    }
    let text = console::describe_error(ctx, error);
    console::write_err_line(ctx, format!("Uncaught {text}"));
}

fn env_usize(name: &str, default: usize, minimum: usize, maximum: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(minimum, maximum)
}

#[cfg(test)]
mod crypto_asym_tests;
#[cfg(test)]
mod tests;
