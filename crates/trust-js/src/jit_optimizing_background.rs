//! Isolated opt-in hot-codegen experiment. No engine owner crosses the worker boundary.
use super::{codegen, diagnostics, NativeFinish, PreparedCode};
use crate::ast::Function;
use crate::bytecode::Chunk;
use crate::interpreter::Env;
use crate::jit::cache::CompilePermit;
use cranelift_codegen::{isa::TargetIsa, Context};
use std::rc::{Rc, Weak};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
    mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    Arc, OnceLock,
};
use std::time::Instant;

const CAPACITY: usize = 4;
const QUEUED: u8 = 0;
const RUNNING: u8 = 1;
const READY: u8 = 2;
const LOST: u8 = 3;
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static WORKER_INITIALIZED: AtomicBool = AtomicBool::new(false);
static COUNTED_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static COUNTED_SCHEDULE_REQUESTS: AtomicU64 = AtomicU64::new(0);
static COUNTED_NATIVE_SUCCESSES: AtomicU64 = AtomicU64::new(0);
static COUNTED_HOT_PUBLISHED: AtomicU64 = AtomicU64::new(0);
static COUNTED_COMPILE_DECLINES: AtomicU64 = AtomicU64::new(0);
static COUNTED_HOT_DECLINES: AtomicU64 = AtomicU64::new(0);
static RUNNING_JOBS: AtomicUsize = AtomicUsize::new(0);
static READY_REPLIES: AtomicUsize = AtomicUsize::new(0);
static OPEN_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static SUBMITTED: AtomicU64 = AtomicU64::new(0);
static FINISHED: AtomicU64 = AtomicU64::new(0);
static PUBLISHED: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static STALE: AtomicU64 = AtomicU64::new(0);
static CANCELLED: AtomicU64 = AtomicU64::new(0);
static PRESSURE: AtomicU64 = AtomicU64::new(0);
static SCHEDULE_REQUESTS: AtomicU64 = AtomicU64::new(0);
static UNAVAILABLE: AtomicU64 = AtomicU64::new(0);
static ID_EXHAUSTED: AtomicU64 = AtomicU64::new(0);
static QUEUE_NS: AtomicU64 = AtomicU64::new(0);
static RESERVE_NS: AtomicU64 = AtomicU64::new(0);
static PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static CODEGEN_NS: AtomicU64 = AtomicU64::new(0);
static FINISH_NS: AtomicU64 = AtomicU64::new(0);

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        super::mode() == super::Mode::Hot
            && std::env::var("TRUST_JS_OPT_JIT_BACKGROUND").as_deref() == Ok("1")
    })
}
pub(crate) fn work_in_flight() -> bool {
    IN_FLIGHT.load(Ordering::Acquire) != 0
}
pub(crate) fn compile_declines(attempts: u64, successes: u64) -> u64 {
    let async_uncompleted = COUNTED_ATTEMPTS
        .load(Ordering::Acquire)
        .saturating_sub(COUNTED_NATIVE_SUCCESSES.load(Ordering::Acquire));
    attempts
        .saturating_sub(successes)
        .saturating_sub(async_uncompleted)
        .saturating_add(COUNTED_COMPILE_DECLINES.load(Ordering::Acquire))
}
pub(super) fn hot_declines(requests: u64, published: u64) -> u64 {
    let other_requests = requests.saturating_sub(COUNTED_SCHEDULE_REQUESTS.load(Ordering::Acquire));
    let other_published = published.saturating_sub(COUNTED_HOT_PUBLISHED.load(Ordering::Acquire));
    other_requests
        .saturating_sub(other_published)
        .saturating_add(COUNTED_HOT_DECLINES.load(Ordering::Acquire))
}
pub(super) fn metrics_json() -> String {
    format!("{{\"worker_initialized\":{},\"process_worker_storage_in_managed_snapshot\":false,\"schedule_requests\":{},\"submitted\":{},\"finished\":{},\"published\":{},\"failed\":{},\"stale\":{},\"cancelled\":{},\"pressure\":{},\"unavailable\":{},\"id_exhausted\":{},\"in_flight\":{},\"open_attempts\":{},\"running\":{},\"ready_replies\":{},\"reserve_ns\":{},\"queue_ns\":{},\"prepare_ns\":{},\"codegen_ns\":{},\"finish_ns\":{}}}",
        WORKER_INITIALIZED.load(Ordering::Relaxed), SCHEDULE_REQUESTS.load(Ordering::Relaxed),
        SUBMITTED.load(Ordering::Relaxed), FINISHED.load(Ordering::Relaxed), PUBLISHED.load(Ordering::Relaxed),
        FAILED.load(Ordering::Relaxed), STALE.load(Ordering::Relaxed), CANCELLED.load(Ordering::Relaxed),
        PRESSURE.load(Ordering::Relaxed), UNAVAILABLE.load(Ordering::Relaxed), ID_EXHAUSTED.load(Ordering::Relaxed),
        IN_FLIGHT.load(Ordering::Relaxed), OPEN_ATTEMPTS.load(Ordering::Relaxed),
        RUNNING_JOBS.load(Ordering::Relaxed), READY_REPLIES.load(Ordering::Relaxed), RESERVE_NS.load(Ordering::Relaxed), QUEUE_NS.load(Ordering::Relaxed),
        PREPARE_NS.load(Ordering::Relaxed), CODEGEN_NS.load(Ordering::Relaxed), FINISH_NS.load(Ordering::Relaxed))
}

// A process-wide work slot is acquired BEFORE snapshotting/lowering. It travels only with
// ordinary worker data and is released on every failure, discarded result and shutdown path.
struct WorkSlot;
impl WorkSlot {
    fn reserve() -> Option<Self> {
        let mut count = IN_FLIGHT.load(Ordering::Relaxed);
        while count < CAPACITY {
            match IN_FLIGHT.compare_exchange(count, count + 1, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => return Some(Self),
                Err(current) => count = current,
            }
        }
        None
    }
}
impl Drop for WorkSlot {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::Release);
    }
}

pub(super) struct Ticket {
    id: u64,
    slot: WorkSlot,
    preparation: Preparation,
}
/// Capacity is temporary; a dead process worker or exhausted identity space is terminal.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ScheduleError {
    Pressure,
    Unavailable,
    IdExhausted,
}
pub(super) fn rearm(chunk: &Chunk, outcome: Result<(), ScheduleError>) {
    let interval = match outcome {
        Ok(()) | Err(ScheduleError::Pressure) => super::hot_threshold(chunk.jit_ops().len()),
        Err(ScheduleError::Unavailable | ScheduleError::IdExhausted) => 0,
    };
    set_interval(chunk, interval);
}
fn stop(chunk: &Chunk) {
    set_interval(chunk, 0);
}
fn set_interval(chunk: &Chunk, interval: u32) {
    if let Some(samples) = chunk.optimizing_samples.get() {
        samples.interval.set(interval);
    }
    chunk.optimizing_remaining.set(interval);
}
struct Job {
    id: u64,
    context: Option<Context>,
    isa: Arc<dyn TargetIsa>,
    cancelled: Arc<AtomicBool>,
    status: Arc<AtomicU8>,
    reply: SyncSender<Done>,
    diagnostic_id: Option<u64>,
    counted: bool,
    queued: Instant,
    slot: Option<WorkSlot>,
    #[cfg(test)]
    gate: Option<(SyncSender<()>, Receiver<()>)>,
    #[cfg(test)]
    panic: bool,
    #[cfg(test)]
    panic_worker: bool,
    #[cfg(test)]
    finished: Option<SyncSender<()>>,
}
impl Drop for Job {
    fn drop(&mut self) {
        if self.status.load(Ordering::Acquire) != READY {
            self.status.store(LOST, Ordering::Release);
        }
        #[cfg(test)]
        if let Some(finished) = self.finished.take() {
            let _ = finished.send(());
        }
    }
}
struct Running;
impl Running {
    fn new() -> Self {
        RUNNING_JOBS.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for Running {
    fn drop(&mut self) {
        RUNNING_JOBS.fetch_sub(1, Ordering::Release);
    }
}
struct Ready;
impl Ready {
    fn new() -> Self {
        READY_REPLIES.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for Ready {
    fn drop(&mut self) {
        READY_REPLIES.fetch_sub(1, Ordering::Release);
    }
}
struct Done {
    id: u64,
    result: Result<Vec<u8>, String>,
    codegen_ns: u64,
    _slot: Option<WorkSlot>,
    _ready: Option<Ready>,
}

struct Worker {
    sender: SyncSender<Job>,
    alive: Arc<AtomicBool>,
    #[cfg(test)]
    stopped: std::sync::Mutex<Receiver<()>>,
}
struct WorkerLife {
    alive: Arc<AtomicBool>,
    #[cfg(test)]
    stopped: SyncSender<()>,
}
impl Drop for WorkerLife {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
        #[cfg(test)]
        let _ = self.stopped.send(());
    }
}
fn worker() -> Option<&'static Worker> {
    static WORKER: OnceLock<Option<Worker>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            let worker = Worker::start();
            if worker.is_some() {
                WORKER_INITIALIZED.store(true, Ordering::Release);
            }
            worker
        })
        .as_ref()
}
impl Worker {
    fn start() -> Option<Self> {
        let (tx, rx) = mpsc::sync_channel::<Job>(CAPACITY);
        let alive = Arc::new(AtomicBool::new(true));
        let worker_alive = alive.clone();
        #[cfg(test)]
        let (stopped, observed) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("trust-js-opt-codegen".into())
            .spawn(move || {
                let _lifetime = WorkerLife {
                    alive: worker_alive,
                    #[cfg(test)]
                    stopped,
                };
                while let Ok(mut job) = rx.recv() {
                    #[cfg(test)]
                    if job.panic_worker {
                        panic!("injected worker-loop panic");
                    }
                    #[cfg(test)]
                    if let Some((entered, gate)) = job.gate.take() {
                        let _ = entered.send(());
                        let _ = gate.recv();
                    }
                    if job.cancelled.load(Ordering::Acquire) {
                        continue;
                    }
                    job.status.store(RUNNING, Ordering::Release);
                    let running = Running::new();
                    QUEUE_NS.fetch_add(diagnostics::nanos(job.queued), Ordering::Relaxed);
                    let mut context = job.context.take().expect("owned worker IR");
                    let started = Instant::now();
                    // Scope is worker-local and receives only its numeric diagnostic identity.
                    let passes = job.diagnostic_id.map(diagnostics::passes::Scope::start_id);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        #[cfg(test)]
                        if job.panic {
                            panic!("injected backend panic");
                        }
                        codegen::compile(&mut context, &*job.isa).and_then(|code| {
                            if !code.buffer.relocs().is_empty() {
                                Err("unexpected external relocation".into())
                            } else {
                                Ok(code.code_buffer().to_vec())
                            }
                        })
                    }))
                    .unwrap_or_else(|_| Err("backend worker panicked".into()));
                    let codegen_ns = diagnostics::nanos(started);
                    drop(passes);
                    drop(context);
                    drop(running);
                    CODEGEN_NS.fetch_add(codegen_ns, Ordering::Relaxed);
                    FINISHED.fetch_add(1, Ordering::Relaxed);
                    if job.counted {
                        super::COMPILE_NANOS.fetch_add(codegen_ns, Ordering::Relaxed);
                        super::super::PERF_COMPILE_NANOS.fetch_add(codegen_ns, Ordering::Relaxed);
                    }
                    if !job.cancelled.load(Ordering::Acquire) {
                        let done = Done {
                            id: job.id,
                            result,
                            codegen_ns,
                            _slot: job.slot.take(),
                            _ready: Some(Ready::new()),
                        };
                        // Global work slots bound all pending replies to CAPACITY, so no living
                        // Agent can fill a CAPACITY reply queue and block another Agent's worker.
                        if job.reply.send(done).is_ok() {
                            job.status.store(READY, Ordering::Release);
                        }
                    }
                }
            })
            .ok()?;
        Some(Worker {
            sender: tx,
            alive,
            #[cfg(test)]
            stopped: std::sync::Mutex::new(observed),
        })
    }
}

struct OpenAttempt {
    counted: bool,
    compiler_started: bool,
    deferred: bool,
    native_succeeded: bool,
    published: bool,
}
impl OpenAttempt {
    fn new(counted: bool) -> Self {
        OPEN_ATTEMPTS.fetch_add(1, Ordering::AcqRel);
        Self {
            counted,
            compiler_started: false,
            deferred: false,
            native_succeeded: false,
            published: false,
        }
    }
}
impl Drop for OpenAttempt {
    fn drop(&mut self) {
        OPEN_ATTEMPTS.fetch_sub(1, Ordering::Release);
        if self.counted && self.compiler_started && !self.native_succeeded {
            COUNTED_COMPILE_DECLINES.fetch_add(1, Ordering::Relaxed);
        }
        if self.counted && !self.published && !self.deferred {
            COUNTED_HOT_DECLINES.fetch_add(1, Ordering::Relaxed);
        }
    }
}
/// Starts before reservation/snapshot/lowering, and records elapsed preparation on EVERY
/// exit. Accepted jobs transfer only OpenAttempt into the engine-owned pending version.
pub(super) struct Preparation {
    started: Instant,
    counted: bool,
    compiler_started: bool,
    open: Option<OpenAttempt>,
}
impl Preparation {
    pub(super) fn new(counted: bool) -> Self {
        SCHEDULE_REQUESTS.fetch_add(1, Ordering::Relaxed);
        if counted {
            COUNTED_SCHEDULE_REQUESTS.fetch_add(1, Ordering::Relaxed);
            super::HOT_REQUESTS.fetch_add(1, Ordering::Relaxed);
        }
        Self {
            started: Instant::now(),
            counted,
            compiler_started: false,
            open: Some(OpenAttempt::new(counted)),
        }
    }
    fn start_compile(&mut self) {
        RESERVE_NS.fetch_add(diagnostics::nanos(self.started), Ordering::Relaxed);
        self.started = Instant::now();
        self.compiler_started = true;
        self.open
            .as_mut()
            .expect("open scheduling request")
            .compiler_started = true;
        if self.counted {
            COUNTED_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            super::COMPILE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
            super::super::PERF_COMPILE_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn defer(&mut self) {
        self.open
            .as_mut()
            .expect("open scheduling request")
            .deferred = true;
    }
}
impl Drop for Preparation {
    fn drop(&mut self) {
        let elapsed = diagnostics::nanos(self.started);
        if self.compiler_started {
            PREPARE_NS.fetch_add(elapsed, Ordering::Relaxed);
            if self.counted {
                super::COMPILE_NANOS.fetch_add(elapsed, Ordering::Relaxed);
                super::super::PERF_COMPILE_NANOS.fetch_add(elapsed, Ordering::Relaxed);
            }
        } else {
            RESERVE_NS.fetch_add(elapsed, Ordering::Relaxed);
        }
    }
}
struct Pending {
    id: u64,
    function: Weak<Function>,
    baseline: Weak<Chunk>,
    genv: std::rc::Weak<std::cell::RefCell<crate::interpreter::Scope>>,
    next: Rc<Chunk>,
    finish: NativeFinish,
    permit: CompilePermit,
    cancelled: Arc<AtomicBool>,
    status: Arc<AtomicU8>,
    counted: bool,
    _open_attempt: OpenAttempt,
}
pub(crate) struct State {
    reply: SyncSender<Done>,
    done: Receiver<Done>,
    pending: Vec<Pending>,
    next_id: u64,
    #[cfg(test)]
    worker: Option<Worker>,
    #[cfg(test)]
    gate: Option<(SyncSender<()>, Receiver<()>)>,
    #[cfg(test)]
    panic: bool,
    #[cfg(test)]
    panic_worker: bool,
    #[cfg(test)]
    finished: Option<SyncSender<()>>,
}
impl Default for State {
    fn default() -> Self {
        let (reply, done) = mpsc::sync_channel(CAPACITY);
        Self {
            reply,
            done,
            pending: Vec::new(),
            next_id: 1,
            #[cfg(test)]
            worker: None,
            #[cfg(test)]
            gate: None,
            #[cfg(test)]
            panic: false,
            #[cfg(test)]
            panic_worker: false,
            #[cfg(test)]
            finished: None,
        }
    }
}
impl State {
    fn worker(&self) -> Option<&Worker> {
        #[cfg(test)]
        if let Some(worker) = &self.worker {
            return Some(worker);
        }
        worker()
    }
    pub(super) fn contains(&self, chunk: &Chunk) -> bool {
        self.pending
            .iter()
            .any(|pending| std::ptr::eq(pending.baseline.as_ptr(), chunk))
    }
    pub(super) fn available(&self) -> bool {
        self.pending.len() < CAPACITY
    }
    pub(super) fn reserve(
        &mut self,
        mut preparation: Preparation,
    ) -> Result<Ticket, ScheduleError> {
        if self
            .worker()
            .is_none_or(|worker| !worker.alive.load(Ordering::Acquire))
        {
            UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
            return Err(ScheduleError::Unavailable);
        }
        if !self.available() {
            PRESSURE.fetch_add(1, Ordering::Relaxed);
            preparation.defer();
            return Err(ScheduleError::Pressure);
        }
        let Some(next_id) = self.next_id.checked_add(1) else {
            ID_EXHAUSTED.fetch_add(1, Ordering::Relaxed);
            return Err(ScheduleError::IdExhausted);
        };
        let Some(slot) = WorkSlot::reserve() else {
            PRESSURE.fetch_add(1, Ordering::Relaxed);
            preparation.defer();
            return Err(ScheduleError::Pressure);
        };
        let id = self.next_id;
        self.next_id = next_id;
        preparation.start_compile();
        Ok(Ticket {
            id,
            slot,
            preparation,
        })
    }
    pub(super) fn submit(
        &mut self,
        ticket: Ticket,
        function: &Rc<Function>,
        baseline: &Rc<Chunk>,
        genv: &Env,
        next: Rc<Chunk>,
        prepared: PreparedCode,
        permit: CompilePermit,
    ) -> Result<(), ScheduleError> {
        let Some(worker) = self.worker() else {
            UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
            return Err(ScheduleError::Unavailable);
        };
        let sender = worker.sender.clone();
        let worker_alive = worker.alive.clone();
        let Ticket {
            id,
            slot,
            mut preparation,
        } = ticket;
        let cancelled = Arc::new(AtomicBool::new(false));
        let status = Arc::new(AtomicU8::new(QUEUED));
        let PreparedCode {
            context,
            isa,
            finish,
        } = prepared;
        let diagnostic_id = finish.diagnostic.as_ref().map(|record| record.id());
        let counted = preparation.counted;
        let job = Job {
            id,
            context: Some(context),
            isa,
            cancelled: cancelled.clone(),
            status: status.clone(),
            reply: self.reply.clone(),
            diagnostic_id,
            counted,
            queued: Instant::now(),
            slot: Some(slot),
            #[cfg(test)]
            gate: self.gate.take(),
            #[cfg(test)]
            panic: self.panic,
            #[cfg(test)]
            panic_worker: self.panic_worker,
            #[cfg(test)]
            finished: self.finished.take(),
        };
        match sender.try_send(job) {
            Ok(()) => (),
            Err(TrySendError::Full(_)) => {
                PRESSURE.fetch_add(1, Ordering::Relaxed);
                preparation.defer();
                return Err(ScheduleError::Pressure);
            }
            Err(TrySendError::Disconnected(_)) => {
                worker_alive.store(false, Ordering::Release);
                UNAVAILABLE.fetch_add(1, Ordering::Relaxed);
                return Err(ScheduleError::Unavailable);
            }
        }
        SUBMITTED.fetch_add(1, Ordering::Relaxed);
        let open_attempt = preparation.open.take().expect("one admitted attempt");
        drop(preparation);
        self.pending.push(Pending {
            id,
            function: Rc::downgrade(function),
            baseline: Rc::downgrade(baseline),
            genv: Rc::downgrade(genv),
            next,
            finish,
            permit,
            cancelled,
            status,
            counted,
            _open_attempt: open_attempt,
        });
        Ok(())
    }
    pub(super) fn poll(&mut self) {
        for _ in 0..CAPACITY {
            match self.done.try_recv() {
                Ok(done) => self.finish(done),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.cancel_all();
                    break;
                }
            }
        }
        // A worker/queue panic or disconnected reply must not retain engine pins forever.
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].status.load(Ordering::Acquire) == LOST {
                let pending = self.pending.swap_remove(index);
                pending.cancelled.store(true, Ordering::Release);
                if let Some(baseline) = pending.baseline.upgrade() {
                    stop(&baseline);
                }
                FAILED.fetch_add(1, Ordering::Relaxed);
            } else {
                index += 1;
            }
        }
    }
    fn finish(&mut self, done: Done) {
        let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.id == done.id)
        else {
            return;
        };
        let mut pending = self.pending.swap_remove(index);
        // This version's attempt has ended, including stale completion after code2 changed.
        // A zero interval explicitly labels the old baseline's sampling as stopped.
        if let Some(baseline) = pending.baseline.upgrade() {
            stop(&baseline);
        }
        if pending.cancelled.load(Ordering::Acquire) {
            return;
        }
        if let Some(record) = &pending.finish.diagnostic {
            record.codegen_ns.set(done.codegen_ns);
        }
        let (Some(function), Some(baseline), Some(genv)) = (
            pending.function.upgrade(),
            pending.baseline.upgrade(),
            pending.genv.upgrade(),
        ) else {
            STALE.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if function.optimizing_code.get().is_some()
            || function
                .execution_code_for(Rc::as_ptr(&genv) as usize)
                .and_then(Option::as_ref)
                .is_none_or(|current| !Rc::ptr_eq(current, &baseline))
        {
            STALE.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Ok(bytes) = done.result else {
            FAILED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let started = Instant::now();
        let result = pending.finish.finish(&bytes);
        let finish_ns = diagnostics::nanos(started);
        FINISH_NS.fetch_add(finish_ns, Ordering::Relaxed);
        if pending.counted {
            super::COMPILE_NANOS.fetch_add(finish_ns, Ordering::Relaxed);
            super::super::PERF_COMPILE_NANOS.fetch_add(finish_ns, Ordering::Relaxed);
        }
        let Ok(code) = result else {
            FAILED.fetch_add(1, Ordering::Relaxed);
            return;
        };
        // Native allocation, registration, owner/cache commit and publication are engine-only.
        super::super::profiler::register(&pending.next, code.mem, code.len);
        pending._open_attempt.native_succeeded = true;
        if pending.counted {
            COUNTED_NATIVE_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            let bytes = code.len as u64;
            super::COMPILE_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            super::GENERATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
            super::super::PERF_COMPILE_SUCCESSES.fetch_add(1, Ordering::Relaxed);
            super::super::PERF_GENERATED_CODE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            super::super::PERF_LARGEST_CODE_BYTES.fetch_max(bytes, Ordering::Relaxed);
        }
        if function.optimizing_code.get().is_some()
            || function
                .execution_code_for(Rc::as_ptr(&genv) as usize)
                .and_then(Option::as_ref)
                .is_none_or(|current| !Rc::ptr_eq(current, &baseline))
        {
            STALE.fetch_add(1, Ordering::Relaxed);
            return;
        }
        pending.permit.commit(Some(Rc::new(code)));
        if function.optimizing_code.set(Some(pending.next)).is_ok() {
            pending._open_attempt.published = true;
            crate::bytecode::invalidate_call_caches();
            PUBLISHED.fetch_add(1, Ordering::Relaxed);
            if pending.counted {
                COUNTED_HOT_PUBLISHED.fetch_add(1, Ordering::Relaxed);
                super::HOT_PUBLISHED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    fn cancel_all(&mut self) {
        for pending in &self.pending {
            pending.cancelled.store(true, Ordering::Release);
            if let Some(baseline) = pending.baseline.upgrade() {
                stop(&baseline);
            }
        }
        CANCELLED.fetch_add(self.pending.len() as u64, Ordering::Relaxed);
        self.pending.clear();
    }
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) {
        visitor.add_function_bytecode_bytes(
            std::mem::size_of::<Self>() + self.pending.capacity() * std::mem::size_of::<Pending>(),
        );
        for pending in &self.pending {
            visitor.chunk(&pending.next);
        }
        // Even an empty live Agent controller retains opaque standard-library channels.
        visitor.mark_background_compiler_unmeasured();
    }
}
impl Drop for State {
    fn drop(&mut self) {
        self.cancel_all();
    }
}
#[allow(dead_code)]
fn assert_send_boundary() {
    fn send<T: Send>() {}
    send::<Context>();
    send::<Job>();
    send::<Done>();
}
#[cfg(test)]
mod tests {
    use super::*;
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    use crate::{
        bytecode::{self, Tier},
        feedback::ValueClass,
        Completion, Engine,
    };

    fn fixture() -> (Engine, Rc<Function>, Rc<Chunk>) {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        engine
            .eval(
                "function subject(n){'use strict';return n+2}subject(1)",
                false,
            )
            .unwrap();
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let function = {
            let object = value.as_obj().unwrap().borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("user function")
            };
            user.func.clone()
        };
        let baseline = function
            .code
            .get()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        (engine, function, baseline)
    }
    fn submit(
        state: &mut State,
        engine: &mut Engine,
        function: &Rc<Function>,
        baseline: &Rc<Chunk>,
    ) -> Weak<Chunk> {
        submit_counted(state, engine, function, baseline, false)
    }
    fn submit_counted(
        state: &mut State,
        engine: &mut Engine,
        function: &Rc<Function>,
        baseline: &Rc<Chunk>,
        counted: bool,
    ) -> Weak<Chunk> {
        let ticket = state
            .reserve(Preparation::new(counted))
            .expect("bounded work reservation");
        // Exercise terminal/cancel/stale accounting from an admitted running sample state.
        baseline
            .optimizing_samples
            .get_or_init(|| Box::new(crate::feedback::EntrySamples::new(4096)));
        baseline
            .optimizing_samples
            .get()
            .unwrap()
            .interval
            .set(4096);
        baseline.optimizing_remaining.set(4096);
        let next = bytecode::compile_for_optimizer(
            function,
            baseline,
            &[(0, ValueClass::NumberInt32.bit())],
            &[],
        )
        .unwrap();
        let weak = Rc::downgrade(&next);
        let values = crate::value::jit_layout(&engine.interp.object_proto);
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let prepared = super::super::prepare_entry_with_limits(
            &next, &values, &layout, None, 16_384, 65_536, None,
        )
        .unwrap();
        let permit = next.jit.begin_compile().unwrap();
        assert!(state
            .submit(
                ticket,
                function,
                baseline,
                &engine.interp.global_env,
                next,
                prepared,
                permit
            )
            .is_ok());
        weak
    }
    fn gate(state: &mut State) -> (Receiver<()>, SyncSender<()>) {
        let (entered, observed) = mpsc::sync_channel(0);
        let (release, wait) = mpsc::sync_channel(0);
        state.gate = Some((entered, wait));
        (observed, release)
    }
    fn child(name: &str, envs: &[(&str, &str)]) -> bool {
        const CASE: &str = "TRUST_JS_TEST_BACKGROUND_CASE";
        if std::env::var(CASE).as_deref() == Ok(name) {
            return true;
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                &format!("jit::optimizing::background::tests::{name}"),
                "--nocapture",
            ])
            .env(CASE, name);
        for (key, value) in envs {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        false
    }
    fn hot_envs() -> [(&'static str, &'static str); 5] {
        [
            ("TRUST_JS_OPT_JIT", "hot"),
            ("TRUST_JS_OPT_JIT_BACKGROUND", "1"),
            ("TRUST_JS_OPT_JIT_HOT_AT", "32"),
            ("TRUST_JS_INLINE_AT", "0"),
            ("TRUST_JS_PERF_METRICS", "1"),
        ]
    }
    fn disconnected_worker() -> Worker {
        let (sender, receiver) = mpsc::sync_channel(CAPACITY);
        drop(receiver);
        let (_, stopped) = mpsc::sync_channel(1);
        Worker {
            sender,
            alive: Arc::new(AtomicBool::new(true)),
            stopped: std::sync::Mutex::new(stopped),
        }
    }
    fn metric(json: &str, key: &str) -> u64 {
        let marker = format!("\"{key}\":");
        let start = json.find(&marker).expect("numeric metric") + marker.len();
        let digits: String = json[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().unwrap()
    }
    fn assert_stopped(baseline: &Chunk) {
        assert_eq!(baseline.optimizing_remaining.get(), 0);
        assert_eq!(
            baseline.optimizing_samples.get().unwrap().interval.get(),
            0,
            "terminal attempts stop entry-sample progress interpretation"
        );
    }

    #[test]
    fn failed_preparations_record_attempts_and_elapsed_work() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let before_attempts = super::super::COMPILE_ATTEMPTS.load(Ordering::Acquire);
        let before_global = super::super::super::PERF_COMPILE_ATTEMPTS.load(Ordering::Acquire);
        let before_ns = super::super::COMPILE_NANOS.load(Ordering::Acquire);
        let before_prepare = PREPARE_NS.load(Ordering::Acquire);
        let before_declines = COUNTED_COMPILE_DECLINES.load(Ordering::Acquire);
        for invalid_layout in [false, true] {
            let ticket = state.reserve(Preparation::new(true)).unwrap();
            let next = bytecode::compile_for_optimizer(
                &function,
                &baseline,
                &[(0, ValueClass::NumberInt32.bit())],
                &[],
            )
            .unwrap();
            let values = crate::value::jit_layout(&engine.interp.object_proto);
            let mut layout = crate::interpreter::interp_layout(&mut engine.interp);
            layout.valid = !invalid_layout;
            let result = super::super::prepare_entry_with_limits(
                &next,
                &values,
                &layout,
                None,
                if invalid_layout { 16_384 } else { 0 },
                65_536,
                None,
            );
            assert!(
                matches!(result, Err(super::super::CompileError::Unsupported)) == invalid_layout
            );
            assert!(result.is_err());
            drop(ticket);
        }
        assert_eq!(
            super::super::COMPILE_ATTEMPTS.load(Ordering::Acquire),
            before_attempts + 2
        );
        assert_eq!(
            super::super::super::PERF_COMPILE_ATTEMPTS.load(Ordering::Acquire),
            before_global + 2
        );
        assert_eq!(
            COUNTED_COMPILE_DECLINES.load(Ordering::Acquire),
            before_declines + 2
        );
        assert!(super::super::COMPILE_NANOS.load(Ordering::Acquire) > before_ns);
        assert!(PREPARE_NS.load(Ordering::Acquire) > before_prepare);
        assert!(state.pending.is_empty());
    }

    #[test]
    fn disconnected_submission_disables_hot_candidate_without_repreparation() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child(
            "disconnected_submission_disables_hot_candidate_without_repreparation",
            &hot_envs(),
        ) {
            return;
        }
        let mut state = State::default();
        state.worker = Some(disconnected_worker());
        let mut engine = Engine::new();
        engine.interp.optimizing_background = Some(Box::new(state));
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let before_attempts = COUNTED_ATTEMPTS.load(Ordering::Acquire);
        let before_prepares = super::super::TEST_PREPARATIONS.with(|count| count.get());
        engine
            .eval(
                "function subject(n){return n+2}for(var k=0;k<256;k++)subject(k)",
                false,
            )
            .unwrap();
        let attempts = COUNTED_ATTEMPTS.load(Ordering::Acquire);
        let prepares = super::super::TEST_PREPARATIONS.with(|count| count.get());
        assert_eq!(attempts, before_attempts + 1);
        assert_eq!(prepares, before_prepares + 1);
        let state = engine.interp.optimizing_background.as_ref().unwrap();
        assert!(!state.worker.as_ref().unwrap().alive.load(Ordering::Acquire));
        assert!(state.pending.is_empty());
        engine
            .eval("for(var k=0;k<2048;k++)subject(k)", false)
            .unwrap();
        assert_eq!(COUNTED_ATTEMPTS.load(Ordering::Acquire), attempts);
        assert_eq!(
            super::super::TEST_PREPARATIONS.with(|count| count.get()),
            prepares
        );
        assert_eq!(COUNTED_COMPILE_DECLINES.load(Ordering::Acquire), 1);
        assert_eq!(COUNTED_HOT_DECLINES.load(Ordering::Acquire), 1);
        // A separate candidate sees the dead worker before clone/lowering and is disabled too.
        engine
            .eval(
                "function sibling(n){return n+3}for(var k=0;k<2048;k++)sibling(k)",
                false,
            )
            .unwrap();
        assert_eq!(COUNTED_ATTEMPTS.load(Ordering::Acquire), attempts);
        assert_eq!(COUNTED_SCHEDULE_REQUESTS.load(Ordering::Acquire), 2);
        assert_eq!(
            super::super::TEST_PREPARATIONS.with(|count| count.get()),
            prepares
        );
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
    }

    #[test]
    fn worker_loop_panic_marks_loss_and_disables_future_reservations() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        state.worker = Worker::start();
        assert!(state.worker.is_some());
        state.panic_worker = true;
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        state
            .worker
            .as_ref()
            .unwrap()
            .stopped
            .lock()
            .unwrap()
            .recv()
            .unwrap();
        assert_eq!(state.pending[0].status.load(Ordering::Acquire), LOST);
        state.poll();
        assert!(state.pending.is_empty());
        assert!(weak.upgrade().is_none());
        assert!(matches!(
            state.reserve(Preparation::new(false)),
            Err(ScheduleError::Unavailable)
        ));
        assert_eq!(baseline.optimizing_remaining.get(), 0);
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
    }

    #[test]
    fn queue_send_pressure_releases_preparation_and_all_engine_owners() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child(
            "queue_send_pressure_releases_preparation_and_all_engine_owners",
            &hot_envs(),
        ) {
            return;
        }
        let (mut engine, function, baseline) = fixture();
        // A rendezvous queue with a living but non-receiving backend deterministically returns
        // Full. Production work-slot bounds prevent saturation from being an unbounded path.
        let (sender, receiver) = mpsc::sync_channel(0);
        let (_, stopped) = mpsc::sync_channel(1);
        let mut state = State::default();
        state.worker = Some(Worker {
            sender,
            alive: Arc::new(AtomicBool::new(true)),
            stopped: std::sync::Mutex::new(stopped),
        });
        let before_attempts = super::super::COMPILE_ATTEMPTS.load(Ordering::Acquire);
        let before_declines = COUNTED_COMPILE_DECLINES.load(Ordering::Acquire);
        let before_hot = COUNTED_HOT_DECLINES.load(Ordering::Acquire);
        let before_ns = super::super::COMPILE_NANOS.load(Ordering::Acquire);
        let before_slots = IN_FLIGHT.load(Ordering::Acquire);
        let ticket = state.reserve(Preparation::new(true)).unwrap();
        let next = bytecode::compile_for_optimizer(
            &function,
            &baseline,
            &[(0, ValueClass::NumberInt32.bit())],
            &[],
        )
        .unwrap();
        let weak = Rc::downgrade(&next);
        let values = crate::value::jit_layout(&engine.interp.object_proto);
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let prepared = super::super::prepare_entry_with_limits(
            &next, &values, &layout, None, 16_384, 65_536, None,
        )
        .unwrap();
        let permit = next.jit.begin_compile().unwrap();
        let outcome = state.submit(
            ticket,
            &function,
            &baseline,
            &engine.interp.global_env,
            next,
            prepared,
            permit,
        );
        assert_eq!(outcome, Err(ScheduleError::Pressure));
        let samples = baseline
            .optimizing_samples
            .get_or_init(|| Box::new(crate::feedback::EntrySamples::new(4096)));
        rearm(&baseline, outcome);
        assert_eq!(
            samples.interval.get(),
            baseline.optimizing_remaining.get(),
            "pressure must preserve the actual entry-sample distance"
        );
        assert_ne!(
            baseline.optimizing_remaining.get(),
            0,
            "queue pressure remains retryable"
        );
        assert!(state.pending.is_empty());
        assert!(weak.upgrade().is_none());
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), before_slots);
        assert_eq!(
            super::super::COMPILE_ATTEMPTS.load(Ordering::Acquire),
            before_attempts + 1
        );
        assert_eq!(
            COUNTED_COMPILE_DECLINES.load(Ordering::Acquire),
            before_declines + 1
        );
        assert_eq!(COUNTED_HOT_DECLINES.load(Ordering::Acquire), before_hot);
        assert!(super::super::COMPILE_NANOS.load(Ordering::Acquire) > before_ns);
        drop(receiver);
    }

    #[test]
    fn background_disabled_and_forced_modes_preserve_sync_publication() {
        let _serialized = TEST_LOCK.lock().unwrap();
        const SELECTED: &str = "TRUST_JS_TEST_BACKGROUND_SYNC_MODE";
        let Ok(selected) = std::env::var(SELECTED) else {
            for mode in ["hot", "1"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "jit::optimizing::background::tests::background_disabled_and_forced_modes_preserve_sync_publication", "--nocapture"])
                    .env(SELECTED, mode).env("TRUST_JS_OPT_JIT", mode)
                    .env("TRUST_JS_OPT_JIT_BACKGROUND", if mode == "hot" { "0" } else { "1" })
                    .env("TRUST_JS_OPT_JIT_HOT_AT", "32").env("TRUST_JS_INLINE_AT", "0")
                    .output().unwrap();
                assert!(
                    output.status.success(),
                    "{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            return;
        };
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine
            .eval(
                "function subject(n){return n+2}for(var k=0;k<256;k++)subject(k)",
                false,
            )
            .unwrap();
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let function = {
            let object = value.as_obj().unwrap().borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("function")
            };
            user.func.clone()
        };
        let code = if selected == "hot" {
            function
                .optimizing_code
                .get()
                .and_then(Option::as_ref)
                .expect("synchronous hot publication")
        } else {
            function
                .code
                .get()
                .and_then(Option::as_ref)
                .expect("forced synchronous compilation")
        };
        assert!(code.jit.get().flatten().unwrap().pc_offsets.is_empty());
        assert!(engine.interp.optimizing_background.is_none());
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), 0);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
    }

    #[test]
    fn executable_budget_failure_releases_permit_and_disables_candidate() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child(
            "executable_budget_failure_releases_permit_and_disables_candidate",
            &[("TRUST_JS_JIT_CODE_BUDGET_MB", "1")],
        ) {
            return;
        }
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        let done = state.done.recv().unwrap();
        assert!(done.result.is_ok());
        let budget = super::super::super::executable_code_budget();
        let remaining = budget.remaining.load(Ordering::Acquire);
        let reservation =
            super::super::super::ExecutableCodeReservation::try_new(budget, remaining).unwrap();
        baseline.optimizing_remaining.set(64);
        state.finish(done);
        assert!(state.pending.is_empty());
        assert!(weak.upgrade().is_none());
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
        assert_eq!(baseline.optimizing_remaining.get(), 0);
        drop(reservation);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
    }

    #[test]
    fn memory_reports_pending_compiler_population_as_incomplete() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child(
            "memory_reports_pending_compiler_population_as_incomplete",
            &[("TRUST_JS_PERF_METRICS", "1")],
        ) {
            return;
        }
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let (entered, release) = gate(&mut state);
        submit(&mut state, &mut engine, &function, &baseline);
        entered.recv().unwrap();
        engine.interp.optimizing_background = Some(Box::new(state));
        engine.interp.gc_collect();
        let json = crate::memory::json(&engine.interp);
        assert!(json.contains(
            "background compiler IR, scratch, channels and pending native metadata are unmeasured"
        ));
        assert!(json.contains(
            "one or more requested-payload allocation families lack retained-memory traversal"
        ));
        assert!(json.contains("\"complete\":false"));
        let state = engine.interp.optimizing_background.as_mut().unwrap();
        state.cancel_all();
        engine.interp.gc_collect();
        assert!(crate::memory::json(&engine.interp).contains("background compiler IR, scratch, channels and pending native metadata are unmeasured"), "cancelled worker IR remains outside the Agent inventory until its work slot drops");
        release.send(()).unwrap();
        // An acknowledged barrier on a second job proves the cancelled job has dropped its slot.
        let mut barrier = State::default();
        let (arrived, unpark) = gate(&mut barrier);
        submit(&mut barrier, &mut engine, &function, &baseline);
        arrived.recv().unwrap();
        unpark.send(()).unwrap();
        let done = barrier.done.recv().unwrap();
        barrier.finish(done);
        drop(barrier);
        engine.interp.gc_collect();
        assert!(crate::memory::json(&engine.interp).contains(
            "background compiler IR, scratch, channels and pending native metadata are unmeasured"
        ), "the empty live controller still owns unmeasured channel storage");
        drop(engine.interp.optimizing_background.take());
        engine.interp.gc_collect();
        assert!(!crate::memory::json(&engine.interp).contains(
            "background compiler IR, scratch, channels and pending native metadata are unmeasured"
        ));
        assert!(WORKER_INITIALIZED.load(Ordering::Acquire));
        assert!(metrics_json().contains("\"process_worker_storage_in_managed_snapshot\":false"));
    }

    #[test]
    fn capacity_pressure_skips_preparation_and_resumes_at_hot_checkpoint() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child(
            "capacity_pressure_skips_preparation_and_resumes_at_hot_checkpoint",
            &hot_envs(),
        ) {
            return;
        }
        let mut quota = State::default();
        let tickets: Vec<_> = (0..CAPACITY)
            .map(|_| quota.reserve(Preparation::new(false)).unwrap())
            .collect();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let prepares = super::super::TEST_PREPARATIONS.with(|count| count.get());
        engine
            .eval(
                "function subject(n){return n+2}for(var k=0;k<2048;k++)subject(k)",
                false,
            )
            .unwrap();
        assert_eq!(
            COUNTED_ATTEMPTS.load(Ordering::Acquire),
            0,
            "capacity checks are scheduling requests, not compiler attempts"
        );
        assert_eq!(COUNTED_COMPILE_DECLINES.load(Ordering::Acquire), 0);
        assert_eq!(
            COUNTED_HOT_DECLINES.load(Ordering::Acquire),
            0,
            "capacity deferral is not a terminal admission rejection"
        );
        assert_eq!(
            super::super::TEST_PREPARATIONS.with(|count| count.get()),
            prepares
        );
        assert!(engine
            .interp
            .optimizing_background
            .as_ref()
            .unwrap()
            .pending
            .is_empty());
        assert!(PRESSURE.load(Ordering::Acquire) > 0);
        drop(tickets);
        engine
            .eval("for(var k=0;k<128;k++)subject(k)", false)
            .unwrap();
        let state = engine.interp.optimizing_background.as_mut().unwrap();
        if !state.pending.is_empty() {
            let done = state.done.recv().unwrap();
            state.reply.send(done).unwrap();
            engine
                .eval("for(var k=0;k<128;k++)subject(k)", false)
                .unwrap();
        }
        assert_eq!(
            super::super::TEST_PREPARATIONS.with(|count| count.get()),
            prepares + 1
        );
        assert_eq!(PUBLISHED.load(Ordering::Acquire), 1);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
    }

    #[test]
    fn idle_ready_agents_hold_quota_until_drain_or_drop() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child("idle_ready_agents_hold_quota_until_drain_or_drop", &[]) {
            return;
        }
        let mut engines = Vec::new();
        for _ in 0..CAPACITY {
            let (mut engine, function, baseline) = fixture();
            let mut state = State::default();
            submit(&mut state, &mut engine, &function, &baseline);
            let done = state.done.recv().unwrap();
            state.reply.send(done).unwrap();
            engine.interp.optimizing_background = Some(Box::new(state));
            engines.push(engine);
        }
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), CAPACITY);
        assert_eq!(READY_REPLIES.load(Ordering::Acquire), CAPACITY);
        let mut candidate = State::default();
        assert!(matches!(
            candidate.reserve(Preparation::new(false)),
            Err(ScheduleError::Pressure)
        ));
        drop(engines.pop());
        let ticket = candidate.reserve(Preparation::new(false)).unwrap();
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), CAPACITY);
        drop(ticket);
        for engine in &mut engines {
            engine.interp.optimizing_background.as_mut().unwrap().poll();
        }
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), 0);
        assert_eq!(READY_REPLIES.load(Ordering::Acquire), 0);
        assert!(candidate.reserve(Preparation::new(false)).is_ok());
    }

    #[test]
    fn hot_checkpoint_submits_then_publishes_ready_code() {
        let _serialized = TEST_LOCK.lock().unwrap();
        const CHILD: &str = "TRUST_JS_TEST_BACKGROUND_HOT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "jit::optimizing::background::tests::hot_checkpoint_submits_then_publishes_ready_code", "--nocapture"])
                .env(CHILD, "1").env("TRUST_JS_OPT_JIT", "hot")
                .env("TRUST_JS_OPT_JIT_BACKGROUND", "1")
                .env("TRUST_JS_OPT_JIT_HOT_AT", "32").env("TRUST_JS_INLINE_AT", "0")
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine
            .eval(
                "function subject(n){return n+2}for(var k=0;k<64;k++)subject(k)",
                false,
            )
            .unwrap();
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let function = {
            let object = value.as_obj().unwrap().borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("function")
            };
            user.func.clone()
        };
        if function.optimizing_code.get().is_none() {
            let state = engine
                .interp
                .optimizing_background
                .as_mut()
                .expect("actual hot checkpoint submitted work");
            assert!(state
                .pending
                .iter()
                .any(|pending| pending.function.as_ptr() == Rc::as_ptr(&function)));
            // Wait for the real worker, then put the already-ready plain reply back. A later
            // ordinary hot countdown checkpoint performs the actual publication.
            let done = state.done.recv().unwrap();
            state.reply.send(done).unwrap();
            engine
                .eval("for(var k=0;k<128;k++)subject(k)", false)
                .unwrap();
        }
        let optimized = function
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .expect("checkpoint publishes finished code");
        let code = optimized.jit.get().flatten().unwrap();
        assert!(code.pc_offsets.is_empty());
        code.residency.referenced.set(0);
        assert_eq!(optimized.optimizing_misses.get(), 0);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
        assert!(
            code.residency.referenced.get() > 0,
            "published worker bytes entered native code"
        );
        assert_eq!(code.residency.active.get(), 0);
        assert_eq!(optimized.optimizing_misses.get(), 0);
    }

    #[test]
    fn publishes_actual_native_code_after_engine_thread_finish() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        let done = state.done.recv().unwrap();
        state.finish(done);
        let next = function
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .unwrap();
        assert!(Rc::ptr_eq(next, &weak.upgrade().unwrap()));
        let code = next.jit.get().flatten().unwrap();
        assert!(code.pc_offsets.is_empty());
        engine.set_tier(Tier::Jit);
        code.residency.referenced.set(0);
        assert_eq!(next.optimizing_misses.get(), 0);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
        assert!(
            code.residency.referenced.get() > 0,
            "published worker bytes entered native code"
        );
        assert_eq!(code.residency.active.get(), 0);
        assert_eq!(next.optimizing_misses.get(), 0);
        assert!(state.pending.is_empty());
        assert_stopped(&baseline);
    }

    #[test]
    fn queued_native_misses_recover_once_retire_and_release_owners() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        engine.eval(
            "var entries=0, conversions=0; function subject(n){var out=n+2;entries++;return out} subject(1); entries=0; var input={valueOf(){conversions++;return 40}};",
            false,
        ).unwrap();
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let function = {
            let object = value.as_obj().unwrap().borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("function")
            };
            user.func.clone()
        };
        let baseline = function
            .code
            .get()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        let input = {
            let value = engine.interp.get_var("input", &env).ok().unwrap();
            Rc::downgrade(value.as_obj().unwrap())
        };
        let mut state = State::default();
        submit(&mut state, &mut engine, &function, &baseline);
        let done = state.done.recv().unwrap();
        state.finish(done);
        let optimized = function
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        let code = optimized.jit.get().flatten().unwrap();
        let weak_code = Rc::downgrade(&code);
        engine.set_tier(Tier::Jit);
        code.residency.referenced.set(0);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
        assert!(
            code.residency.referenced.get() > 0,
            "worker-generated guarded version ran"
        );
        assert_eq!(code.residency.active.get(), 0);
        assert_eq!(optimized.optimizing_misses.get(), 0);
        engine.eval("entries=0", false).unwrap();
        for miss in 1..=crate::bytecode::SPECIALIZATION_MISS_LIMIT {
            code.residency.referenced.set(0);
            assert!(
                matches!(engine.eval("subject(input)", false), Ok(Completion::Value(value)) if value == "42")
            );
            assert!(
                code.residency.referenced.get() > 0,
                "the changed input entered the queued native guard"
            );
            assert_eq!(optimized.optimizing_misses.get(), miss);
            assert_eq!(code.residency.active.get(), 0);
            // The entry guard recovers before the first increment and coercion. Neither effect
            // can be repeated or skipped by the VM continuation.
            assert!(
                matches!(engine.eval("entries+'|'+conversions", false), Ok(Completion::Value(value)) if value == format!("{miss}|{miss}"))
            );
        }
        assert!(Rc::ptr_eq(
            function
                .execution_code_for(Rc::as_ptr(&env) as usize)
                .and_then(Option::as_ref)
                .unwrap(),
            &baseline
        ));
        assert!(
            weak_code.upgrade().is_some(),
            "the explicit Rust lease pins retired bytes"
        );
        code.residency.referenced.set(0);
        assert!(
            matches!(engine.eval("subject(input)", false), Ok(Completion::Value(value)) if value == "42")
        );
        assert_eq!(
            code.residency.referenced.get(),
            0,
            "retired optimized bytes were not entered"
        );
        assert_eq!(
            optimized.optimizing_misses.get(),
            crate::bytecode::SPECIALIZATION_MISS_LIMIT
        );
        assert!(
            matches!(engine.eval("entries+'|'+conversions", false), Ok(Completion::Value(value)) if value == "9|9")
        );
        drop(code);
        assert!(optimized.jit.get().is_none());
        assert!(
            weak_code.upgrade().is_none(),
            "retired mapping reclaimed after native activity and Rust leases ended"
        );
        engine.eval("input=null", false).unwrap();
        engine.interp.gc_collect();
        assert!(
            input.upgrade().is_none(),
            "compiled versions/continuations retained no JavaScript input owner"
        );
        assert!(state.pending.is_empty());
        assert_stopped(&baseline);
    }

    #[test]
    fn a_new_selected_baseline_refuses_stale_background_publication() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let (entered, release) = gate(&mut state);
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        entered.recv().unwrap();
        // A second-stage publisher changes the selected Chunk while the worker owns only IR.
        let replacement =
            bytecode::compile_with_inlines(&function, &Default::default(), &baseline).unwrap();
        assert!(function.code2.set(Some(replacement)).is_ok());
        engine.interp.gc_collect();
        release.send(()).unwrap();
        let done = state.done.recv().unwrap();
        state.finish(done);
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
        assert!(weak.upgrade().is_none());
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
    }

    #[test]
    fn shutdown_drops_engine_pins_without_waiting_for_a_running_worker() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let (entered, release) = gate(&mut state);
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        entered.recv().unwrap();
        engine.interp.gc_collect();
        // Deterministic barrier keeps the worker blocked. State/Interp destruction must finish
        // before it is released, proving that no join or engine owner sits on that thread.
        engine.interp.optimizing_background = Some(Box::new(state));
        drop(engine);
        assert!(weak.upgrade().is_none());
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
        release.send(()).unwrap();
    }

    #[test]
    fn unknown_completed_job_ids_do_not_mutate_a_live_pending_version() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let (entered, release) = gate(&mut state);
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        entered.recv().unwrap();
        state.finish(Done {
            id: u64::MAX,
            result: Ok(vec![0]),
            codegen_ns: 0,
            _slot: None,
            _ready: None,
        });
        assert!(function.optimizing_code.get().is_none());
        assert!(weak.upgrade().is_some());
        release.send(()).unwrap();
        let done = state.done.recv().unwrap();
        state.finish(done);
        assert!(function
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .is_some());
    }
    #[test]
    fn work_reservations_bound_lowering_and_refuse_id_wrap() {
        const CHILD: &str = "TRUST_JS_TEST_BACKGROUND_PRESSURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "jit::optimizing::background::tests::work_reservations_bound_lowering_and_refuse_id_wrap"])
                .env(CHILD, "1").output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let mut state = State::default();
        let mut tickets: Vec<_> = (0..CAPACITY)
            .map(|_| state.reserve(Preparation::new(false)).unwrap())
            .collect();
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), CAPACITY);
        assert!(
            matches!(
                state.reserve(Preparation::new(false)),
                Err(ScheduleError::Pressure)
            ),
            "refuse before any CLIF allocation/lowering"
        );
        drop(tickets.pop());
        assert!(
            state.reserve(Preparation::new(false)).is_ok(),
            "failed preparation releases its work slot"
        );
        drop(tickets);
        assert_eq!(IN_FLIGHT.load(Ordering::Acquire), 0);
        state.next_id = u64::MAX;
        assert!(matches!(
            state.reserve(Preparation::new(false)),
            Err(ScheduleError::IdExhausted)
        ));
        assert_eq!(
            IN_FLIGHT.load(Ordering::Acquire),
            0,
            "no ID wrap, no work-slot leak"
        );
    }

    #[test]
    fn backend_panic_returns_failure_and_releases_pending_owners() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        state.panic = true;
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        let done = state.done.recv().unwrap();
        assert!(done.result.is_err());
        state.finish(done);
        assert!(state.pending.is_empty());
        assert!(weak.upgrade().is_none());
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
        assert!(
            matches!(engine.eval("subject(40)", false), Ok(Completion::Value(value)) if value == "42")
        );
    }

    #[test]
    fn lost_completion_releases_engine_owners_at_the_checkpoint() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let (entered, release) = gate(&mut state);
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        entered.recv().unwrap();
        state.pending[0].status.store(LOST, Ordering::Release);
        state.poll();
        assert!(state.pending.is_empty());
        assert!(weak.upgrade().is_none());
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
        release.send(()).unwrap();
    }
    #[test]
    fn disconnected_reply_is_detected_without_leaking_pending_owners() {
        let _serialized = TEST_LOCK.lock().unwrap();
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let (entered, release) = gate(&mut state);
        let (finished, observed) = mpsc::sync_channel(0);
        state.finished = Some(finished);
        let weak = submit(&mut state, &mut engine, &function, &baseline);
        entered.recv().unwrap();
        let (keep_open, replacement) = mpsc::sync_channel(CAPACITY);
        drop(std::mem::replace(&mut state.done, replacement));
        release.send(()).unwrap();
        observed.recv().unwrap();
        assert_eq!(state.pending[0].status.load(Ordering::Acquire), LOST);
        state.poll();
        assert!(state.pending.is_empty());
        assert!(weak.upgrade().is_none());
        assert!(function.optimizing_code.get().is_none());
        assert_stopped(&baseline);
        drop(keep_open);
    }

    #[test]
    fn unfinished_attempts_are_not_reported_as_completed_declines() {
        let _serialized = TEST_LOCK.lock().unwrap();
        if !child(
            "unfinished_attempts_are_not_reported_as_completed_declines",
            &[("TRUST_JS_PERF_METRICS", "1")],
        ) {
            return;
        }
        let (mut engine, function, baseline) = fixture();
        let mut state = State::default();
        let compile = || {
            compile_declines(
                super::super::COMPILE_ATTEMPTS.load(Ordering::Acquire),
                super::super::COMPILE_SUCCESSES.load(Ordering::Acquire),
            )
        };
        let hot = || {
            hot_declines(
                super::super::HOT_REQUESTS.load(Ordering::Acquire),
                super::super::HOT_PUBLISHED.load(Ordering::Acquire),
            )
        };
        let before_compile = compile();
        let before_hot = hot();
        let global_json = || crate::jit::performance_metrics_json("{}").unwrap();
        let before_global = metric(&global_json(), "jit_compile_failures");
        submit_counted(&mut state, &mut engine, &function, &baseline, true);
        assert_eq!(
            compile(),
            before_compile,
            "a pending worker is not a completed failure"
        );
        assert_eq!(
            hot(),
            before_hot,
            "an unpublished pending version is not a decline"
        );
        assert_eq!(
            metric(&global_json(), "jit_compile_failures"),
            before_global
        );
        assert_eq!(metric(&global_json(), "compile_declines"), before_compile);
        assert_eq!(metric(&global_json(), "hot_declines"), before_hot);
        let done = state.done.recv().unwrap();
        assert_eq!(
            compile(),
            before_compile,
            "a ready but unpolled reply is not a failure"
        );
        assert_eq!(hot(), before_hot);
        assert_eq!(
            metric(&global_json(), "jit_compile_failures"),
            before_global
        );
        state.finish(done);
        assert!(function
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .is_some());
        assert_eq!(compile(), before_compile);
        assert_eq!(hot(), before_hot);
        drop(state.reserve(Preparation::new(true)).unwrap());
        assert_eq!(compile(), before_compile + 1);
        assert_eq!(hot(), before_hot + 1);
        assert_eq!(
            metric(&global_json(), "jit_compile_failures"),
            before_global + 1
        );
    }
}
