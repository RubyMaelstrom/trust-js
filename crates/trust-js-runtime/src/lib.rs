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

use trust_js_host::{
    install, CallbackQueue, DedicatedExecutor, Engine, EvalError, RuntimeInterrupt, TaskCompletion,
    TaskDecoder, TaskRegistry, ThreadPool, Value,
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
    fire_error: Value,
    fire_rejection: Value,
}

/// The callback batch the loop is running. A pass takes everything queued so far out of the
/// [`CallbackQueue`] (callbacks queued meanwhile wait for the next pass); each stays rooted here,
/// like the queue, until it starts, so a collection inside one callback cannot reclaim the rest.
#[derive(Default)]
struct RunningCallbacks(std::collections::VecDeque<(Value, Vec<Value>)>);

impl trust_js_host::HostTrace for RunningCallbacks {
    fn trace_gc(&self, visitor: &mut dyn trust_js_host::HostGcVisitor) {
        for (callback, args) in &self.0 {
            visitor.root(callback);
            for arg in args {
                visitor.root(arg);
            }
        }
    }
}

/// Unhandled rejections being reported, taken from the engine in one batch: each (promise,
/// reason) pair stays rooted here until its report starts, since `onunhandledrejection` runs
/// script (and may collect) between reports.
#[derive(Default)]
struct ReportingRejections(std::collections::VecDeque<(Value, Value)>);

impl trust_js_host::HostTrace for ReportingRejections {
    fn trace_gc(&self, visitor: &mut dyn trust_js_host::HostGcVisitor) {
        for (promise, reason) in &self.0 {
            visitor.root(promise);
            visitor.root(reason);
        }
    }
}

/// The error-reporting shims: no JavaScript reference keeps `fire_rejection` alive once its
/// global is deleted, and the `Runtime` that holds both may live where no scan looks.
struct ReportingShims {
    fire_error: Value,
    fire_rejection: Value,
}

impl trust_js_host::HostTrace for ReportingShims {
    fn trace_gc(&self, visitor: &mut dyn trust_js_host::HostGcVisitor) {
        visitor.root(&self.fire_error);
        visitor.root(&self.fire_rejection);
    }
}

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
        engine.ctx().op_state().put(pool.handle());
        // Dedicated-thread completions for unbounded-blocking work (child stdio) that must not
        // occupy a shared pool worker.
        engine.ctx().op_state().put(dedicated.handle());
        engine
            .ctx()
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
        let global = engine.global_this();
        let fire_error = engine
            .ctx()
            .get_member(&global, "__trust_js_fire_error")
            .unwrap_or_else(|_| panic!("error-reporting shim installed"));
        let fire_rejection = engine
            .ctx()
            .get_member(&global, "__trust_js_fire_rejection")
            .unwrap_or_else(|_| panic!("error-reporting shim installed"));
        engine
            .eval(
                "delete globalThis.__trust_js_fire_error; delete globalThis.__trust_js_fire_rejection;",
                false,
            )
            .expect("shim cleanup");
        engine.set_interrupt_handle(interrupt);
        let state = engine.ctx().op_state();
        state.put(ReportingShims {
            fire_error: fire_error.clone(),
            fire_rejection: fire_rejection.clone(),
        });
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
        let global = self.engine.global_this();
        let run_main = self
            .engine
            .ctx()
            .get_member(&global, "__runMain")
            .map_err(|_| "node runtime not installed".to_string())?;
        let result = self.engine.call_function_interruptible(
            &run_main,
            Value::Undefined,
            &[Value::from_string(path.to_string())],
        );
        if let Err(reason) = self.engine.run_microtasks_interruptible() {
            return Err(reason.message().to_string());
        }
        match result {
            Ok(_) => Ok(()),
            Err(EvalError::Throw(error)) => Err(describe_error(self.engine.ctx(), &error)),
            Err(EvalError::Interrupted(reason)) => Err(reason.message().to_string()),
        }
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
                    self.fire(&cb, &args);
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
        let global = self.engine.global_this();
        let ctx = self.engine.ctx();
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
    /// arguments and `callback` runs on the loop thread. This is the pattern async fs ops
    /// will use from inside native fns (via the `SpawnHandle`/`TaskRegistry` in `OpState`).
    pub fn spawn_blocking(
        &mut self,
        work: impl FnOnce() -> Box<dyn std::any::Any + Send> + Send + 'static,
        callback: Value,
        decode: TaskDecoder,
    ) {
        let registry = self
            .engine
            .ctx()
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
                    self.fire(&cb, &args);
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
        let state = self.engine.ctx().op_state();
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
        let state = self.engine.ctx().op_state();
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

    /// The batch's next callback. It leaves the rooted batch only as it starts: the call roots
    /// its callee and arguments from then on.
    fn next_batch_callback(&mut self) -> Option<(Value, Vec<Value>)> {
        self.engine
            .ctx()
            .host_mut::<RunningCallbacks>()?
            .0
            .pop_front()
    }

    fn take_due_timers(&mut self) -> Vec<trust_js_timers::TimerTask> {
        let now = Instant::now();
        match self.engine.ctx().host_mut::<trust_js_timers::Timers>() {
            Some(t) => t.take_due(now),
            None => Vec::new(),
        }
    }

    /// The next timer deadline, or an earlier Atomics.waitAsync timeout, whose promise resolves
    /// at the checkpoint after it (a pending wait alone does not keep the loop alive).
    fn next_timer_deadline(&mut self) -> Option<Instant> {
        let timer = self
            .engine
            .ctx()
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
            .ctx()
            .host_mut::<TaskRegistry>()
            .and_then(|r| r.take(done.task));
        let Some(entry) = entry else {
            return; // cancelled while in flight
        };
        let result = match done.result {
            Ok(payload) => (entry.decode)(self.engine.ctx(), payload),
            Err(failure) => Err(self.engine.ctx().make_error("Error", failure.message)),
        };
        match result {
            Ok(args) => self.fire(&entry.on_ok, &args),
            Err(e) => match &entry.on_err {
                Some(reject) => self.fire(reject, std::slice::from_ref(&e)),
                None => self.report_uncaught(&e),
            },
        }
        if self.engine.run_microtasks_interruptible().is_err() {
            return;
        }
        self.report_unhandled_rejections();
    }

    /// One JS callback entry: call, report an uncaught throw, then the microtask checkpoint.
    fn fire(&mut self, callback: &Value, args: &[Value]) {
        match self
            .engine
            .call_function_interruptible(callback, Value::Undefined, args)
        {
            Ok(_) => {}
            Err(EvalError::Throw(error)) => self.report_uncaught(&error),
            Err(EvalError::Interrupted(_)) => return,
        }
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
            .ctx()
            .host_mut::<trust_js_timers::Timers>()
            .is_some_and(|timers| timers.begin_task(&task));
        if !valid {
            return;
        }

        let thrown = match &task.handler {
            trust_js_timers::TimerHandler::Function(callback) => {
                let this = self.engine.global_this();
                match self
                    .engine
                    .call_function_interruptible(callback, this, &task.args)
                {
                    Ok(_) => None,
                    Err(EvalError::Throw(error)) => Some(error),
                    Err(EvalError::Interrupted(_)) => None,
                }
            }
            trust_js_timers::TimerHandler::Code(code) => {
                match self.engine.eval_value_interruptible(code) {
                    Ok(Ok(_)) => None,
                    Ok(Err(EvalError::Throw(error))) => Some(error),
                    Ok(Err(EvalError::Interrupted(_))) => None,
                    Err(error) => Some(self.engine.ctx().make_error(
                        "SyntaxError",
                        format!("{} (line {})", error.message, error.line),
                    )),
                }
            }
        };
        if let Some(error) = thrown {
            self.report_uncaught(&error);
        }

        if let Some(timers) = self.engine.ctx().host_mut::<trust_js_timers::Timers>() {
            timers.finish_task(&task, Instant::now());
        }
        if self.engine.run_microtasks_interruptible().is_err() {
            return;
        }
        self.report_unhandled_rejections();
    }

    /// An exception escaped a loop-fired callback. Node prints and keeps the loop alive (we
    /// don't model `process.on('uncaughtException')`/exit semantics yet).
    fn report_uncaught(&mut self, error: &Value) {
        // HTML "report an exception": the global `onerror` handler runs first; returning `true`
        // suppresses the default report.
        let fire = self.fire_error.clone();
        match self.engine.call_function_interruptible(
            &fire,
            Value::Undefined,
            std::slice::from_ref(error),
        ) {
            Ok(Value::Bool(true)) | Err(EvalError::Interrupted(_)) => return,
            _ => {}
        }
        let text = console::describe_error(self.engine.ctx(), error);
        console::write_err_line(self.engine.ctx(), format!("Uncaught {text}"));
    }

    /// Report promises rejected without a handler (Node prints `Uncaught (in promise) …`). Called
    /// after each microtask checkpoint; a rejection handled in the same checkpoint won't appear.
    fn report_unhandled_rejections(&mut self) {
        let taken = self.engine.take_unhandled_rejections_full();
        if taken.is_empty() {
            return;
        }
        self.reporting_rejections().0.extend(taken);
        while let Some((promise, reason)) = self.reporting_rejections().0.pop_front() {
            // The global `onunhandledrejection` handler runs first; `event.preventDefault()`
            // suppresses the default report.
            let fire = self.fire_rejection.clone();
            match self.engine.call_function_interruptible(
                &fire,
                Value::Undefined,
                &[promise, reason.clone()],
            ) {
                Ok(Value::Bool(true)) => continue,
                Err(EvalError::Interrupted(_)) => {
                    // An interruption abandons the rest of the batch.
                    self.reporting_rejections().0.clear();
                    return;
                }
                _ => {}
            }
            let text = console::describe_error(self.engine.ctx(), &reason);
            console::write_err_line(self.engine.ctx(), format!("Uncaught (in promise) {text}"));
        }
    }

    fn reporting_rejections(&mut self) -> &mut ReportingRejections {
        self.engine
            .ctx()
            .host_mut::<ReportingRejections>()
            .expect("installed in new()")
    }
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
