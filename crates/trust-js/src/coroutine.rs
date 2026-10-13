//! Heap-owned coroutine continuations for generators, async functions, async modules, and
//! built-in async algorithms.
//!
//! Every source coroutine is lowered to [`VmCoro`](crate::bytecode::VmCoro). Suspension therefore
//! retains an explicit operand stack and execution state without reserving an OS thread, moving
//! the interpreter across threads, or coordinating through channels. A compiler regression is
//! represented by [`Coroutine::Unavailable`], which preserves suspended-start behavior and
//! surfaces a contained JavaScript error when execution is first requested.

use crate::interpreter::Interp;
use crate::value::Value;

/// Driver → generator: resume the body.
#[derive(Clone)]
pub enum Resume {
    /// `next(v)` — the `yield` expression evaluates to `v`.
    Next(Value),
    /// `return(v)` — inject a return completion at the suspended `yield`.
    Return(Value),
    /// `throw(e)` — inject a throw at the suspended `yield`.
    Throw(Value),
    /// Host realm teardown — terminate without running author cleanup code.
    Terminate,
}

/// Coroutine → driver: the body parked or finished.
pub enum Suspend {
    /// `yield v` — parked, produced `v` (a generator value).
    Yield(Value),
    /// `await v` — parked waiting for `v` to settle (async functions/generators).
    Await(Value),
    /// The body ran to completion / `return v`.
    Done(Value),
    /// The body threw `e` and it escaped.
    Throw(Value),
}

/// The resumable state owned by a generator object, async promise, or module evaluation.
pub enum Coroutine {
    Vm(Box<crate::bytecode::VmCoro>),
    /// A Source Text Module AsyncBlock backed by a VM continuation. The wrapper runs the module
    /// fulfilment/rejection cascade when the bytecode execution context completes.
    Module(Box<crate::modules::ModuleCoro>),
    /// The explicit state machine for the built-in async algorithm Array.fromAsync. Unlike a
    /// source async function it has no bytecode body, but it parks at the same Await boundaries.
    FromAsync(Box<crate::builtins::FromAsyncCoro>),
    /// Contained response to a future compiler coverage regression. This is not an execution
    /// fallback: it owns no source body, native stack, channel, or worker.
    Unavailable(UnavailableCoro),
}

impl Coroutine {
    /// The continuation is an internal slot of its generator/result promise, not
    /// an independent GC root. Trace only handles physically owned by this payload.
    pub(crate) fn trace_gc(&self, edges: &mut crate::gc_edges::DirectGcEdges<'_>) {
        match self {
            Coroutine::Vm(c) => c.trace_gc(edges),
            Coroutine::Module(c) => c.trace_gc(edges),
            Coroutine::FromAsync(c) => c.trace_gc(edges),
            Coroutine::Unavailable(c) => {
                if let Some(reason) = &c.reason {
                    edges.value(reason);
                }
            }
        }
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        match self {
            Coroutine::Vm(coroutine) => std::mem::size_of::<crate::bytecode::VmCoro>()
                .saturating_add(coroutine.scan_retained_memory(visitor)),
            Coroutine::Module(coroutine) => coroutine.scan_retained_memory(visitor),
            Coroutine::FromAsync(coroutine) => coroutine.scan_retained_memory(visitor),
            Coroutine::Unavailable(coroutine) => {
                if let Some(reason) = &coroutine.reason {
                    visitor.value(reason);
                }
                0
            }
        }
    }

    #[inline]
    pub(crate) fn resume(&mut self, i: &mut Interp, signal: Resume) -> Suspend {
        match self {
            Coroutine::Vm(c) => c.resume(i, signal),
            Coroutine::Module(c) => c.resume(i, signal),
            Coroutine::FromAsync(c) => c.resume(i, signal),
            Coroutine::Unavailable(c) => c.resume(signal),
        }
    }

    /// Whether the body has finished (further resumes are no-ops).
    #[inline]
    pub fn done(&self) -> bool {
        match self {
            Coroutine::Vm(c) => c.done,
            Coroutine::Module(c) => c.done(),
            Coroutine::FromAsync(c) => c.done(),
            Coroutine::Unavailable(c) => c.done,
        }
    }

    /// Whether the first resume has happened (distinguishes suspendedStart from a suspended yield).
    #[inline]
    pub fn started(&self) -> bool {
        match self {
            Coroutine::Vm(c) => c.started,
            Coroutine::Module(c) => c.started(),
            Coroutine::FromAsync(c) => c.started(),
            Coroutine::Unavailable(c) => c.started,
        }
    }

    pub(crate) fn terminate(&mut self, i: &mut Interp) {
        match self {
            Coroutine::Vm(_) => {}
            Coroutine::Module(c) => c.terminate(i),
            Coroutine::FromAsync(c) => c.terminate(),
            Coroutine::Unavailable(c) => {
                c.reason.take();
                c.done = true;
            }
        }
    }
}

impl Resume {
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) {
        match self {
            Resume::Next(value) | Resume::Return(value) | Resume::Throw(value) => {
                visitor.value(value)
            }
            Resume::Terminate => {}
        }
    }
}

pub const VM_UNAVAILABLE_MSG: &str = "coroutine body could not be lowered to a VM continuation";

/// Preserve GeneratorResumeAbrupt semantics if a future compiler regression reaches production:
/// return/throw before first execution still use the caller's completion, while the first ordinary
/// drive reports the internal lowering error. No source evaluator or native execution context is
/// retained here.
pub struct UnavailableCoro {
    reason: Option<Value>,
    done: bool,
    started: bool,
}

impl UnavailableCoro {
    fn resume(&mut self, signal: Resume) -> Suspend {
        if self.done {
            return Suspend::Done(Value::Undefined);
        }
        self.done = true;
        match signal {
            Resume::Next(_) => {
                self.started = true;
                Suspend::Throw(
                    self.reason
                        .take()
                        .expect("an unavailable coroutine retains its lowering error"),
                )
            }
            Resume::Return(value) => Suspend::Done(value),
            Resume::Throw(error) => Suspend::Throw(error),
            Resume::Terminate => Suspend::Done(Value::Undefined),
        }
    }
}

pub fn unavailable(reason: Value) -> Coroutine {
    Coroutine::Unavailable(UnavailableCoro {
        reason: Some(reason),
        done: false,
        started: false,
    })
}

// The normative evaluator still contains defensive tree-walker arms for YieldExpression and
// AwaitExpression. Source coroutines never enter them now; returning an injected throw keeps an
// accidental call contained instead of recreating a native-stack execution path.
pub fn in_coroutine() -> bool {
    false
}

pub fn in_async_gen() -> bool {
    false
}

pub fn coroutine_yield(i: &mut Interp, _value: Value) -> Resume {
    Resume::Throw(i.make_error("Error", VM_UNAVAILABLE_MSG))
}

pub fn coroutine_await(i: &mut Interp, _value: Value) -> Resume {
    Resume::Throw(i.make_error("Error", VM_UNAVAILABLE_MSG))
}

#[cfg(test)]
mod tests {
    use super::{Resume, Suspend, UnavailableCoro};
    use crate::Value;

    fn unavailable(reason: Value) -> UnavailableCoro {
        UnavailableCoro {
            reason: Some(reason),
            done: false,
            started: false,
        }
    }

    #[test]
    fn unavailable_coroutine_preserves_suspended_start_abrupt_resumption() {
        let mut returned = unavailable(Value::Num(1.0));
        assert!(matches!(
            returned.resume(Resume::Return(Value::Num(2.0))),
            Suspend::Done(Value::Num(2.0))
        ));
        assert!(returned.done);
        assert!(!returned.started);

        let mut thrown = unavailable(Value::Num(1.0));
        assert!(matches!(
            thrown.resume(Resume::Throw(Value::Num(3.0))),
            Suspend::Throw(Value::Num(3.0))
        ));
        assert!(thrown.done);
        assert!(!thrown.started);

        let mut driven = unavailable(Value::Num(4.0));
        assert!(matches!(
            driven.resume(Resume::Next(Value::Undefined)),
            Suspend::Throw(Value::Num(4.0))
        ));
        assert!(driven.done);
        assert!(driven.started);
    }
}
