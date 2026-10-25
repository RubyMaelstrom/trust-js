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
        // P1 S2: removed from its owner's table while it runs, the continuation is reachable
        // only from Rust frames; register it so collections in the body trace it.
        let _running = crate::gc_trace::enter_coroutine(self);
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

impl crate::gc_trace::GcTrace for UnavailableCoro {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let UnavailableCoro {
            reason,
            done: _,
            started: _,
        } = self;
        crate::gc_trace::GcTrace::trace_gc(reason, sink);
    }
}

#[cfg(test)]
mod tests {
    use super::{Resume, Suspend, UnavailableCoro};
    use crate::bytecode::Tier;
    use crate::value::ValueView;
    use crate::{Completion, Engine, Value};

    fn value(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("GC fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

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
            Suspend::Done(value) if matches!(value.view(), ValueView::Num(2.0))
        ));
        assert!(returned.done);
        assert!(!returned.started);

        let mut thrown = unavailable(Value::Num(1.0));
        assert!(matches!(
            thrown.resume(Resume::Throw(Value::Num(3.0))),
            Suspend::Throw(value) if matches!(value.view(), ValueView::Num(3.0))
        ));
        assert!(thrown.done);
        assert!(!thrown.started);

        let mut driven = unavailable(Value::Num(4.0));
        assert!(matches!(
            driven.resume(Resume::Next(Value::Undefined)),
            Suspend::Throw(value) if matches!(value.view(), ValueView::Num(4.0))
        ));
        assert!(driven.done);
        assert!(driven.started);
    }

    #[test]
    fn gc_releases_abandoned_continuations_without_running_author_cleanup() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            value(
                &mut engine,
                r#"
                var targets = [], cleaned = 0;
                function make(mode) {
                    const target = {mode}; target.self = target;
                    function* sequence() {
                        try { yield target; return 1; }
                        finally { cleaned++; }
                    }
                    const generator = sequence();
                    // Keep an actual cycle in every tier, even when compiled capture
                    // analysis omits the otherwise unused local generator binding.
                    target.generator = generator;
                    if (mode !== 0) generator.next();
                    if (mode === 2) generator.return();
                    if (mode === 3) generator.next();
                    targets.push(new WeakRef(target));
                }
                for (var mode = 0; mode < 4; mode++) make(mode);
                'ready'
            "#,
            );
            for _ in 0..3 {
                engine.interp.gc_collect();
                assert_eq!(
                    value(
                        &mut engine,
                        "targets.map(ref => ref.deref() === undefined).join(',') + ':' + cleaned"
                    ),
                    "true,true,true,true:2",
                    "{tier:?}"
                );
            }
            assert!(
                engine.interp.generators.is_empty(),
                "dead continuation metadata must be evicted"
            );
        }
    }

    #[test]
    fn gc_preserves_live_suspended_and_executing_continuations() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            value(
                &mut engine,
                r#"
                var live, weak;
                (() => {
                    const target = {n: 39}; weak = new WeakRef(target);
                    live = (function*(input) {
                        const values = [input];
                        yield values[0].n;
                        // While running, this state has left the side table. Its native
                        // stack-owned handles must NOT be discounted as internal owners.
                        $262.gc();
                        yield input.n + 1;
                        return values[0];
                    })(target);
                })();
                'ready'
            "#,
            );
            for expected in ["39", "40", "[object Object]"] {
                engine.interp.gc_collect();
                assert_eq!(value(&mut engine, "weak.deref().n"), "39");
                assert_eq!(
                    value(&mut engine, "live.next().value"),
                    expected,
                    "{tier:?}"
                );
            }
            value(&mut engine, "live = null; 'released'");
            engine.interp.gc_collect();
            assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
        }
    }

    #[test]
    fn gc_coroutine_edges_preserve_external_promise_resumption_and_weak_ephemerons() {
        let mut engine = Engine::new();
        value(
            &mut engine,
            r#"
            var resolve, done, weak, abandoned, wm = new WeakMap();
            (() => {
                const target = {n: 41}; weak = new WeakRef(target);
                wm.set(target, {n: 42});
                (async function(input) {
                    await new Promise(r => resolve = r);
                    $262.gc();
                    done = wm.get(input).n;
                })(target);
            })();
            (() => {
                const target = {}; target.self = target;
                abandoned = new WeakRef(target);
                (async function(input) {
                    // No resolver escapes, so this await cannot ever resume.
                    await new Promise(() => {});
                    globalThis.unreachable = input;
                })(target);
            })();
            'ready'
        "#,
        );
        engine.interp.gc_collect();
        assert_eq!(
            value(
                &mut engine,
                "[weak.deref().n, abandoned.deref() === undefined, done].join(':')"
            ),
            "41:true:"
        );
        value(&mut engine, "resolve(); resolve = null; 'queued'");
        assert_eq!(value(&mut engine, "done"), "42");
        engine.interp.gc_collect();
        assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
    }

    #[test]
    fn gc_async_generator_reactions_own_the_generator_across_await() {
        for return_mode in [false, true] {
            for reject in [false, true] {
                let mut engine = Engine::new();
                value(
                    &mut engine,
                    r#"
                    var settle, result = 'pending', weak;
                    function fulfilled(value) { result = 'ok:' + value.value; }
                    function rejected(error) { result = 'error:' + error; }
                    function start(returnMode, reject) {
                        const gate = new Promise((resolve, fail) => settle = reject ? fail : resolve);
                        const generator = (async function*() { yield await gate; })();
                        weak = new WeakRef(generator);
                        // The caller retains only the external resolver, not the iterator.
                        const request = returnMode ? generator.return(gate) : generator.next();
                        request.then(fulfilled, rejected);
                    }
                "#,
                );
                value(
                    &mut engine,
                    &format!("start({return_mode}, {reject}); 'ready'"),
                );
                engine.interp.gc_collect();
                assert_eq!(value(&mut engine, "weak.deref() !== undefined"), "true");
                value(&mut engine, "settle(42); settle = null; 'queued'");
                assert_eq!(
                    value(&mut engine, "result"),
                    if reject { "error:42" } else { "ok:42" },
                    "return={return_mode}, reject={reject}"
                );
                engine.interp.gc_collect();
                assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
            }
        }
    }

    #[test]
    fn gc_continuation_reference_class_delegate_and_disposal_state_survives() {
        let mut engine = Engine::new();
        value(
            &mut engine,
            r#"
            var refWeak, classWeak, iterWeak, resourceWeak, disposed = 0;
            function referenceTarget() {
                const object = {x:1}; refWeak = new WeakRef(object); return object;
            }
            function parentClass() {
                const parent = class {base(){return 40}};
                classWeak = new WeakRef(parent); return parent;
            }
            function iterator() {
                const iter = {n:0, next(){return {value:++this.n, done:this.n>2}},
                              [Symbol.iterator](){return this}};
                iterWeak = new WeakRef(iter); return iter;
            }
            function resource() {
                const object = {[Symbol.dispose](){disposed++}};
                resourceWeak = new WeakRef(object); return object;
            }
            var referenceGen = (function*(){referenceTarget().x += yield 'rhs';return 7})();
            var classGen = (function*(){return class extends (yield 'parent') {
                [yield 'name'](){return 2}
            }})();
            var delegateGen = (function*(){return yield* iterator()})();
            var resourceGen = (function*(){using item = resource();yield 1;return 9})();
            referenceGen.next(); classGen.next(); classGen.next(parentClass());
            delegateGen.next(); resourceGen.next();
            'ready'
        "#,
        );
        engine.interp.gc_collect();
        assert_eq!(
            value(
                &mut engine,
                "[refWeak,classWeak,iterWeak,resourceWeak].every(ref => ref.deref() !== undefined)"
            ),
            "true"
        );
        assert_eq!(
            value(
                &mut engine,
                r#"
            var refTarget = refWeak.deref();
            var a = referenceGen.next(5).value;
            var sum = refTarget.x;
            var C = classGen.next('extra').value;
            var c = new C();
            [a, sum, c.base()+c.extra(), delegateGen.next().value,
             resourceGen.next().value, disposed].join(':')
        "#
            ),
            "7:6:42:2:9:1"
        );
        value(
            &mut engine,
            "referenceGen=classGen=delegateGen=resourceGen=C=c=refTarget=null; 'released'",
        );
        engine.interp.gc_collect();
        assert_eq!(
            value(
                &mut engine,
                "[refWeak,classWeak,iterWeak,resourceWeak].every(ref => ref.deref() === undefined)"
            ),
            "true"
        );
        assert_eq!(value(&mut engine, "disposed"), "1");
    }

    #[test]
    fn gc_from_async_retains_its_iterator_until_external_resumption() {
        let mut engine = Engine::new();
        value(
            &mut engine,
            r#"
            var settle, result, weak;
            function done(values) { result = values.join(','); }
            (() => {
                const iter = {
                    index:0,
                    next() { return this.index++ ? {done:true} : new Promise(r => settle=r); },
                    [Symbol.asyncIterator]() { return this; }
                };
                weak = new WeakRef(iter);
                Array.fromAsync(iter).then(done);
            })();
            'ready'
        "#,
        );
        engine.interp.gc_collect();
        assert_eq!(value(&mut engine, "weak.deref() !== undefined"), "true");
        value(
            &mut engine,
            "settle({done:false,value:42}); settle = null; 'queued'",
        );
        assert_eq!(value(&mut engine, "result"), "42");
        engine.interp.gc_collect();
        assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
    }
}
