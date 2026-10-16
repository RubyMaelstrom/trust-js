//! trust-js-host — the substrate shared by every op crate and the runtime.
//!
//! An *op crate* (timers, fs, ...) exports one [`Extension`]: a named bundle of native
//! functions plus host-state initialization. A runtime is assembled by [`install`]ing a list
//! of extensions into an [`Engine`]. Rust state never lives in the native fns themselves
//! (they are bare `fn` pointers): it lives in [`OpState`], reached through the `&mut Ctx`
//! argument every native fn receives.
//!
//! Two scheduling primitives serve every async op (this mirrors libuv's own fs strategy —
//! regular files are not pollable, so async fs is threadpool + completion, not readiness):
//! - [`ThreadPool::spawn_blocking`]: run blocking work off-thread; its result comes back to
//!   the loop thread as a [`TaskCompletion`] over `mpsc`.
//! - [`CallbackQueue`]: loop-thread-local queue of JS callbacks to fire on the next turn
//!   (JS values are `!Send`, so they never cross threads; off-thread work refers to its
//!   callback by [`TaskId`]).

use std::any::Any;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// Hard ceiling for one-shot codec output. Keep decompression bounded independently of the
/// JavaScript engine's larger ArrayBuffer limit: compressed input must not consume unbounded
/// native memory before its output can be handed to JavaScript.
pub const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;

pub(crate) fn checked_decompressed_len(
    current: usize,
    additional: usize,
    limit: usize,
    codec: &str,
) -> Result<usize, String> {
    current
        .checked_add(additional)
        .filter(|&total| total <= limit)
        .ok_or_else(|| format!("{codec}: decompressed output exceeds byte limit"))
}

thread_local! {
    static TASK_PANIC_IS_CONTAINED: Cell<bool> = const { Cell::new(false) };
}

fn install_task_panic_hook() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |information| {
            if !TASK_PANIC_IS_CONTAINED.get() {
                previous(information);
            }
        }));
    });
}

pub use trust_js::bytecode::Tier;
pub use trust_js::embed::{
    ArrayBufferBytes, Ctx, EvalError, HostGcVisitor, HostRetainedMemoryVisitor, HostTrace,
    NativeCallableRetained, NativeCaptures, NativeClosure, NativeFn, NativeRetainedMemoryVisitor,
    OpState, ResourceId, ResourceTable, RetainedBytes, RetainedExternalAllocation,
    RetainedExternalMemory, RetainedManagedAllocation, RetainedMemory, Value, WeakValue,
};
pub use trust_js::value_free;
pub use trust_js::{
    Completion, Engine, ExecutionOutcome, InterruptReason, ParseError, RuntimeInterrupt,
};

/// DEFLATE/zlib/gzip codec (std-only), shared by web CompressionStream and node:zlib.
pub mod deflate;

/// Brotli (RFC 7932) codec (std-only), for node:zlib brotli* APIs.
pub mod brotli;

/// Zstandard (RFC 8878) codec (std-only), for node:zlib zstd* and Bun.zstd* APIs.
pub mod zstd;

/// One native op: a named native function with its JS arity.
#[derive(Clone, Copy)]
pub struct OpDecl {
    pub name: &'static str,
    pub len: usize,
    pub f: NativeFn,
}

/// Declarative op-declaration table: `ops!["add" (2) => add_impl, ...]`. Uniform by design so
/// op registration stays a table, never hand-written glue (deno's `#[op2]` lesson).
#[macro_export]
macro_rules! ops {
    ($($name:literal ($len:expr) => $f:expr),* $(,)?) => {
        &[$($crate::OpDecl { name: $name, len: $len, f: $f }),*]
    };
}

/// A bundle of native ops + host-state init, exported one per op crate. Composing a runtime
/// is `install(&mut engine, &[timers::extension(), fs::extension(), ...])`.
pub struct Extension {
    pub name: &'static str,
    /// Installed as `globalThis.<name>` functions (e.g. `setTimeout`).
    pub globals: &'static [OpDecl],
    /// Installed as `globalThis.<ns>.<name>` namespace methods (e.g. a `__trust_js_fs` ops
    /// object that a JS shim wraps into the public API).
    pub namespaces: &'static [(&'static str, &'static [OpDecl])],
    /// Installs this extension's state (timer heap, fd table, ...) into [`OpState`].
    pub state_init: Option<fn(&mut OpState)>,
    /// JS glue evaluated after this extension's ops are installed — the promise-returning
    /// public API is JS wrapping raw callback ops (e.g. `fs.promises` over `__fs_async`).
    /// A parse/throw here is a bug in the extension: `install` panics with its name.
    pub js_init: Option<&'static str>,
    /// A build-time snapshot of `js_init`'s parsed AST (see `Engine::eval_snapshot`). When
    /// present, `install` decodes it instead of re-lexing/parsing `js_init` every boot — the
    /// dominant cold-start cost. A decode failure (version skew) falls back to `js_init`, so it
    /// is a pure optimization. `js_init` must still be set (the fallback source).
    pub js_init_snapshot: Option<&'static [u8]>,
}

impl Extension {
    /// An empty extension named `name`; fill in the fields that apply.
    pub const fn new(name: &'static str) -> Extension {
        Extension {
            name,
            globals: &[],
            namespaces: &[],
            state_init: None,
            js_init: None,
            js_init_snapshot: None,
        }
    }
}

/// Install extensions into an engine: state first (an op may fire during install), then ops,
/// then JS glue.
pub fn install(engine: &mut Engine, extensions: &[Extension]) {
    for ext in extensions {
        if let Some(init) = ext.state_init {
            init(engine.ctx().op_state());
        }
        for op in ext.globals {
            engine.define_global(op.name, op.len, op.f);
        }
        for (ns, ops) in ext.namespaces {
            let table: Vec<(&str, usize, NativeFn)> =
                ops.iter().map(|o| (o.name, o.len, o.f)).collect();
            engine.define_namespace(ns, &table);
        }
        if let Some(src) = ext.js_init {
            // Prefer the precompiled snapshot (skips lex+parse); on a decode failure fall back to
            // parsing the source, so the snapshot can never change behavior — only speed.
            let completion = ext
                .js_init_snapshot
                .and_then(|bytes| engine.eval_snapshot(bytes, false).ok())
                .map(Ok)
                .unwrap_or_else(|| engine.eval(src, false));
            match completion {
                Ok(Completion::Value(_)) => {}
                Ok(Completion::Throw { name, message }) => {
                    panic!("extension '{}' js_init threw {name}: {message}", ext.name)
                }
                Err(e) => panic!(
                    "extension '{}' js_init: SyntaxError: {}",
                    ext.name, e.message
                ),
            }
        }
    }
}

/// Identifies an in-flight async task. The op that spawns work registers `TaskId -> JS
/// callback/promise` in its own [`OpState`] slot; the completion carries the id back so the
/// loop thread can look the JS value up (JS values themselves are `!Send`).
pub type TaskId = u64;

/// What off-thread work sends back to the loop thread. `result` is whatever `Send` payload
/// the spawning op chose; that op downcasts it when the loop hands the completion over.
pub struct TaskCompletion {
    pub task: TaskId,
    pub result: Result<Box<dyn Any + Send>, TaskFailure>,
}

/// Scheduler-level failure that occurs before an op-specific payload can be decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskFailure {
    pub message: String,
}

impl TaskFailure {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// JS callbacks queued (on the loop thread) to run on the next loop turn — the
/// `enqueue_callback` primitive. Lives in [`OpState`]; the runtime drains it each turn.
#[derive(Default)]
pub struct CallbackQueue {
    pub queue: VecDeque<(Value, Vec<Value>)>,
}

/// Queued callbacks and their arguments run on a later turn: roots until then.
impl HostTrace for CallbackQueue {
    fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
        for (callback, args) in &self.queue {
            visitor.root(callback);
            for arg in args {
                visitor.root(arg);
            }
        }
    }
}

impl CallbackQueue {
    /// Queue `callback(args...)` for the next loop turn.
    pub fn enqueue(state: &mut OpState, callback: Value, args: Vec<Value>) {
        if !state.has::<CallbackQueue>() {
            state.put(CallbackQueue::default());
        }
        state
            .get_mut::<CallbackQueue>()
            .expect("just installed")
            .queue
            .push_back((callback, args));
    }
}

struct Task {
    id: TaskId,
    work: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
    cancelled: Arc<AtomicBool>,
}

/// Shared cancellation index used by the task registry and both blocking executors. Cancellation
/// prevents queued work from starting; resource-specific cancellation (socket shutdown, process
/// kill, etc.) remains responsible for waking work that is already inside an OS call.
#[derive(Clone, Default)]
pub struct TaskCanceller {
    tasks: Arc<Mutex<std::collections::HashMap<TaskId, Arc<AtomicBool>>>>,
}

impl TaskCanceller {
    fn register(&self, id: TaskId) -> Arc<AtomicBool> {
        let token = Arc::new(AtomicBool::new(false));
        let previous = self
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, Arc::clone(&token));
        if let Some(previous) = previous {
            previous.store(true, Ordering::Release);
        }
        token
    }

    fn finish(&self, id: TaskId) {
        self.tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
    }

    fn cancel(&self, id: TaskId) {
        if let Some(token) = self
            .tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id)
        {
            token.store(true, Ordering::Release);
        }
    }
}

/// A fixed pool of std worker threads running finite blocking work (`std::fs`, bounded network
/// setup); completions come back over the channel given at construction. Its work queue and native
/// stacks are bounded, panics become failed completions, and shutdown has a fixed grace period.
pub struct ThreadPool {
    work_tx: Option<mpsc::SyncSender<Task>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    state: Arc<PoolState>,
}

struct PoolState {
    shutting_down: AtomicBool,
    available: AtomicBool,
    completions: mpsc::Sender<TaskCompletion>,
    canceller: TaskCanceller,
}

const WORKER_POLL: Duration = Duration::from_millis(50);
const SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
const DEFAULT_WORKER_STACK: usize = 1 << 20;

impl ThreadPool {
    /// `size` worker threads sending [`TaskCompletion`]s to `completions` (the loop thread
    /// holds the receiving end).
    pub fn new(size: usize, completions: mpsc::Sender<TaskCompletion>) -> ThreadPool {
        Self::with_limits(size, size.max(1) * 64, DEFAULT_WORKER_STACK, completions)
    }

    /// Construct a pool with explicit queue and native-stack bounds. A queue limit caps retained
    /// closures and their buffers; a failed worker spawn degrades capacity instead of panicking.
    pub fn with_limits(
        size: usize,
        queue_capacity: usize,
        stack_size: usize,
        completions: mpsc::Sender<TaskCompletion>,
    ) -> ThreadPool {
        let (work_tx, work_rx) = mpsc::sync_channel::<Task>(queue_capacity.max(1));
        // std's mpsc receiver is single-consumer: share it across workers behind a mutex.
        let work_rx = Arc::new(Mutex::new(work_rx));
        let state = Arc::new(PoolState {
            shutting_down: AtomicBool::new(false),
            available: AtomicBool::new(false),
            completions,
            canceller: TaskCanceller::default(),
        });
        let mut workers = Vec::with_capacity(size.max(1));
        for index in 0..size.max(1) {
            let work_rx = Arc::clone(&work_rx);
            let state = Arc::clone(&state);
            let spawn = std::thread::Builder::new()
                .name(format!("trust-js-blocking-{index}"))
                .stack_size(stack_size.max(64 << 10))
                .spawn(move || worker_loop(&work_rx, &state));
            if let Ok(worker) = spawn {
                workers.push(worker);
            }
        }
        state
            .available
            .store(!workers.is_empty(), Ordering::Release);
        ThreadPool {
            work_tx: Some(work_tx),
            workers,
            state,
        }
    }

    /// Run `work` on a pool thread; its return value comes back to the loop as a
    /// [`TaskCompletion`] tagged with `id`.
    pub fn spawn_blocking(
        &self,
        id: TaskId,
        work: impl FnOnce() -> Box<dyn Any + Send> + Send + 'static,
    ) {
        self.handle().spawn_blocking(id, work);
    }

    /// A cloneable spawn handle. The runtime puts one in [`OpState`], which is how a native fn
    /// (holding only `&mut Ctx`) reaches the pool.
    pub fn handle(&self) -> SpawnHandle {
        SpawnHandle {
            work_tx: self.work_tx.clone().expect("pool shut down"),
            state: Arc::clone(&self.state),
        }
    }

    pub fn canceller(&self) -> TaskCanceller {
        self.state.canceller.clone()
    }
}

fn worker_loop(work_rx: &Mutex<mpsc::Receiver<Task>>, state: &PoolState) {
    loop {
        if state.shutting_down.load(Ordering::Acquire) {
            return;
        }
        let received = work_rx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv_timeout(WORKER_POLL);
        let task = match received {
            Ok(task) => task,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };
        if task.cancelled.load(Ordering::Acquire) {
            state.canceller.finish(task.id);
            continue;
        }
        let result = run_task(task.work);
        state.canceller.finish(task.id);
        let _ = state.completions.send(TaskCompletion {
            task: task.id,
            result,
        });
    }
}

fn run_task(
    work: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
) -> Result<Box<dyn Any + Send>, TaskFailure> {
    install_task_panic_hook();
    TASK_PANIC_IS_CONTAINED.set(true);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
    TASK_PANIC_IS_CONTAINED.set(false);
    result.map_err(|panic| {
        let detail = panic
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("non-string panic payload");
        TaskFailure::new(format!("blocking task panicked: {detail}"))
    })
}

/// [`ThreadPool::spawn_blocking`] as an [`OpState`]-storable handle, so op crates can spawn
/// blocking work from inside a native fn.
#[derive(Clone)]
pub struct SpawnHandle {
    work_tx: mpsc::SyncSender<Task>,
    state: Arc<PoolState>,
}

impl SpawnHandle {
    pub fn spawn_blocking(
        &self,
        id: TaskId,
        work: impl FnOnce() -> Box<dyn Any + Send> + Send + 'static,
    ) {
        let cancelled = self.state.canceller.register(id);
        let task = Task {
            id,
            work: Box::new(work),
            cancelled,
        };
        let failure = if self.state.shutting_down.load(Ordering::Acquire) {
            Some((task, TaskFailure::new("blocking executor is shutting down")))
        } else if self.state.workers_unavailable() {
            Some((task, TaskFailure::new("blocking executor has no workers")))
        } else {
            match self.work_tx.try_send(task) {
                Ok(()) => None,
                Err(mpsc::TrySendError::Full(task)) => {
                    Some((task, TaskFailure::new("blocking executor queue is full")))
                }
                Err(mpsc::TrySendError::Disconnected(task)) => {
                    Some((task, TaskFailure::new("blocking executor is unavailable")))
                }
            }
        };
        if let Some((task, failure)) = failure {
            self.state.canceller.finish(task.id);
            let _ = self.state.completions.send(TaskCompletion {
                task: task.id,
                result: Err(failure),
            });
        }
    }
}

impl PoolState {
    fn workers_unavailable(&self) -> bool {
        !self.available.load(Ordering::Acquire)
    }
}

/// Sends [`TaskCompletion`]s straight to the loop from a *dedicated* thread, bypassing the fixed
/// [`ThreadPool`]. For work that blocks for an unbounded time — a subprocess's stdout read, waiting
/// on a child to exit — where occupying a shared pool worker for the whole duration would starve
/// everything else. The runtime stores one in [`OpState`]. `run_blocking` spawns a fresh thread per
/// call; a runtime-wide limit and bounded stack reserve cap their memory cost.
#[derive(Clone)]
pub struct CompletionSender {
    state: Arc<DedicatedState>,
}

struct DedicatedState {
    tx: mpsc::Sender<TaskCompletion>,
    canceller: TaskCanceller,
    active: AtomicUsize,
    max_threads: usize,
    stack_size: usize,
    shutting_down: AtomicBool,
}

impl DedicatedState {
    /// Count one more active thread unless `max_threads` are already running. This is the
    /// compare-exchange loop of `AtomicUsize::try_update` (stable since Rust 1.95, above the
    /// workspace MSRV), which replaces the deprecated `fetch_update`.
    fn try_acquire_thread(&self) -> bool {
        let mut active = self.active.load(Ordering::Acquire);
        while active < self.max_threads {
            match self.active.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(current) => active = current,
            }
        }
        false
    }
}

/// Owner for bounded, one-thread-per-operation work that may wait indefinitely. It isolates live
/// sockets/listeners/children from the finite shared pool without permitting unbounded threads.
pub struct DedicatedExecutor {
    state: Arc<DedicatedState>,
}

impl DedicatedExecutor {
    pub fn new(
        max_threads: usize,
        stack_size: usize,
        tx: mpsc::Sender<TaskCompletion>,
        canceller: TaskCanceller,
    ) -> Self {
        Self {
            state: Arc::new(DedicatedState {
                tx,
                canceller,
                active: AtomicUsize::new(0),
                max_threads: max_threads.max(1),
                stack_size: stack_size.max(64 << 10),
                shutting_down: AtomicBool::new(false),
            }),
        }
    }

    pub fn handle(&self) -> CompletionSender {
        CompletionSender {
            state: Arc::clone(&self.state),
        }
    }
}

impl Drop for DedicatedExecutor {
    fn drop(&mut self) {
        self.state.shutting_down.store(true, Ordering::Release);
    }
}

impl CompletionSender {
    /// Run `work` on a new dedicated thread; its result comes back to the loop as a
    /// [`TaskCompletion`] tagged with `id` (settled through the [`TaskRegistry`], like pool work).
    pub fn run_blocking(
        &self,
        id: TaskId,
        work: impl FnOnce() -> Box<dyn Any + Send> + Send + 'static,
    ) {
        let cancelled = self.state.canceller.register(id);
        let acquired = self.state.try_acquire_thread();
        if self.state.shutting_down.load(Ordering::Acquire) || !acquired {
            self.state.canceller.finish(id);
            let message = if acquired {
                self.state.active.fetch_sub(1, Ordering::AcqRel);
                "dedicated executor is shutting down"
            } else {
                "dedicated executor thread limit reached"
            };
            let _ = self.state.tx.send(TaskCompletion {
                task: id,
                result: Err(TaskFailure::new(message)),
            });
            return;
        }

        let state = Arc::clone(&self.state);
        let spawn = std::thread::Builder::new()
            .name(format!("trust-js-dedicated-{id}"))
            .stack_size(state.stack_size)
            .spawn(move || {
                let result = if cancelled.load(Ordering::Acquire) {
                    Err(TaskFailure::new("blocking task was cancelled"))
                } else {
                    run_task(Box::new(work))
                };
                state.canceller.finish(id);
                state.active.fetch_sub(1, Ordering::AcqRel);
                let _ = state.tx.send(TaskCompletion { task: id, result });
            });
        if let Err(error) = spawn {
            self.state.canceller.finish(id);
            self.state.active.fetch_sub(1, Ordering::AcqRel);
            let _ = self.state.tx.send(TaskCompletion {
                task: id,
                result: Err(TaskFailure::new(format!(
                    "spawn dedicated blocking thread: {error}"
                ))),
            });
        }
    }
}

/// Turns a completed task's `Send` payload back into JS callback arguments, on the loop
/// thread. `Err` is a JS value to report as an uncaught exception (later: a rejection).
pub type TaskDecoder = fn(&mut Ctx, Box<dyn Any + Send>) -> Result<Vec<Value>, Value>;

/// In-flight async tasks: `TaskId -> (JS callback, payload decoder)`. Lives in [`OpState`];
/// the op that spawns work registers here, the loop settles from [`TaskCompletion`]s. The
/// event loop stays alive while this is non-empty.
pub struct TaskRegistry {
    next: TaskId,
    map: std::collections::HashMap<TaskId, TaskEntry>,
    canceller: TaskCanceller,
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::with_canceller(TaskCanceller::default())
    }
}

/// An in-flight task's settlement callbacks run when it completes: roots until then.
impl HostTrace for TaskRegistry {
    fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
        for entry in self.map.values() {
            visitor.root(&entry.on_ok);
            if let Some(on_err) = &entry.on_err {
                visitor.root(on_err);
            }
        }
    }
}

// Executor handles carry native work and completions, never JavaScript values.
value_free!(SpawnHandle, CompletionSender);

/// How to settle one in-flight task: success callback, optional failure callback (a promise's
/// reject — when absent, a decode error is reported as an uncaught exception), and the
/// payload decoder.
pub struct TaskEntry {
    pub on_ok: Value,
    pub on_err: Option<Value>,
    pub decode: TaskDecoder,
    /// An `unref`'d task still settles when it completes, but does not by itself keep the event
    /// loop alive (Node's `child.unref()` — e.g. esbuild's persistent service child).
    pub unref: bool,
}

impl TaskRegistry {
    pub fn with_canceller(canceller: TaskCanceller) -> Self {
        Self {
            next: 0,
            map: std::collections::HashMap::new(),
            canceller,
        }
    }

    /// Reserve an id for work about to be spawned, remembering how to settle it.
    pub fn register(&mut self, on_ok: Value, on_err: Option<Value>, decode: TaskDecoder) -> TaskId {
        let id = self.next;
        self.next += 1;
        self.map.insert(
            id,
            TaskEntry {
                on_ok,
                on_err,
                decode,
                unref: false,
            },
        );
        id
    }
    /// Claim a completed task's settlement entry (a missing id means it was cancelled).
    pub fn take(&mut self, id: TaskId) -> Option<TaskEntry> {
        self.canceller.finish(id);
        self.map.remove(&id)
    }
    /// Cancel a pending settlement and prevent its queued native work from starting.
    pub fn cancel(&mut self, id: TaskId) -> Option<TaskEntry> {
        self.canceller.cancel(id);
        self.map.remove(&id)
    }
    /// Mark a pending task as `unref`'d (see [`TaskEntry::unref`]).
    pub fn set_unref(&mut self, id: TaskId) {
        if let Some(e) = self.map.get_mut(&id) {
            e.unref = true;
        }
    }
    /// Re-`ref` a pending task so it keeps the loop alive again (Node's `handle.ref()`).
    pub fn set_ref(&mut self, id: TaskId) {
        if let Some(e) = self.map.get_mut(&id) {
            e.unref = false;
        }
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Whether any *ref*'d (loop-keeping) task is pending. Unref'd tasks are ignored — they
    /// settle if they complete but must not hold the process open.
    pub fn has_ref_pending(&self) -> bool {
        self.map.values().any(|e| !e.unref)
    }
}

impl Drop for TaskRegistry {
    fn drop(&mut self) {
        for id in self.map.keys() {
            self.canceller.cancel(*id);
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.state.shutting_down.store(true, Ordering::Release);
        self.state.available.store(false, Ordering::Release);
        self.work_tx.take();
        // A native call cannot be forcibly unwound safely. Give finite work a short grace period,
        // join workers that exit, and detach any still inside an OS call so Runtime::drop itself
        // has a hard latency bound.
        let deadline = Instant::now() + SHUTDOWN_GRACE;
        while Instant::now() < deadline && self.workers.iter().any(|worker| !worker.is_finished()) {
            std::thread::sleep(Duration::from_millis(1));
        }
        for w in self.workers.drain(..) {
            if w.is_finished() {
                let _ = w.join();
            }
        }
    }
}

#[cfg(test)]
mod tests;
