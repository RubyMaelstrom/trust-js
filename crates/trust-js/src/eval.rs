//! Statement execution, expression evaluation, and the ECMAScript abstract operations. Split out
//! of `interpreter.rs` to keep each file readable; this is the same `impl Interp`.

use crate::ast::*;
use crate::interpreter::*;
use crate::stack_trace::NO_POSITION;
use crate::value::ValueView;
use crate::value::*;
use std::fmt::Write as _;
use std::rc::Rc;

/// Complete ordinary-chain proof: key-layout allocation, live prefix and enumerable flags.
/// A live cache entry pins every nonempty layout, preventing pointer ABA. Stale LRU records
/// contain only these numbers; the LRU's generation check disambiguates reused addresses.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(crate) struct EnumerationShape {
    levels: [(usize, u8, u64); 6],
    depth: u8,
}

type EnumerationLayouts = [Option<PropertyLayout>; 6];

/// Only key strings are pinned: never a receiver, prototype, Realm, descriptor or JS value.
/// NamedEntries detaches shared data on mutation, so pins also make key identity immutable.
pub(crate) struct CachedForInKeys {
    keys: ForInKeys,
    layouts: EnumerationLayouts,
}

/// The cached key backing and predicted key layouts are cells retained by this cache.
impl crate::gc_trace::GcTrace for CachedForInKeys {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let CachedForInKeys {
            keys: ForInKeys(keys),
            layouts,
        } = self;
        sink.object(keys);
        crate::gc_trace::GcTrace::trace_gc(layouts.as_slice(), sink);
    }
}

impl CachedForInKeys {
    fn retained_layout_bytes(&self) -> usize {
        self.layouts.iter().flatten().fold(0usize, |bytes, layout| {
            // Conservatively charge duplicates and unused predicted suffixes. The census below
            // instead de-duplicates shared allocations across objects, hints and cache entries.
            // The keys' atoms belong to the heap's table.
            let storage = std::mem::size_of::<PropertyLayoutData>()
                + layout.capacity() * std::mem::size_of::<crate::atom::PropKey>();
            bytes
                .saturating_add(storage)
                .saturating_add(layout.index_bytes().unwrap_or(0))
        })
    }

    pub(crate) fn scan_retained_layouts(&self, visitor: &mut crate::memory::Visitor) {
        for layout in self.layouts.iter().flatten() {
            visitor.property_layout(layout);
        }
    }
}

/// Immutable, private backing for EnumerateObjectProperties candidate keys. The object has no
/// prototype or author values, and is never exposed to JavaScript. Only the backing is shared:
/// each AST/VM activation owns its cursor and source, so reentrancy and suspension are independent.
/// ECMA-262 §14.7.5.9 explicitly permits an implementation-defined inaccessible iterator.
#[derive(Clone)]
pub(crate) struct ForInKeys(Gc);

impl ForInKeys {
    fn new(keys: Vec<String>) -> (Self, usize) {
        // Include the RefCell<Object> (plus two words) and conservatively charge string
        // headers/index keys.
        // Unlike a normal Array, this backing cannot retain %Array.prototype% or its realm.
        let mut bytes =
            std::mem::size_of::<std::cell::RefCell<Object>>() + 2 * std::mem::size_of::<usize>();
        let mut props = Props::new();
        props.reserve_dense_exact(keys.len(), false);
        for key in keys {
            bytes = bytes.saturating_add(key.len().saturating_add(64));
            props.push_dense(Property::data(Value::Str(key.into()), false, false, false));
        }
        bytes = bytes.saturating_add(props.retained_requested_storage_bytes().0);
        let object = Object::new_with_parts(None, props, Exotic::None);
        object.borrow_mut().extensible = false;
        (Self(object), bytes)
    }

    fn key(&self, index: u32) -> Option<Value> {
        self.0.borrow().props.get_index(index).map(Property::value)
    }

    /// Only internal compiler temporaries may contain this value; it has no JavaScript owner.
    pub(crate) fn into_value(self) -> Value {
        Value::Obj(self.0)
    }
}

/// Partially evaluated ClassDefinitionEvaluation retained by a heap VM continuation while a
/// heritage or computed-name expression is suspended. ECMA-262 §15.7.14 creates both class
/// environments before heritage evaluation, then retains them through every class element.
pub(crate) struct PreparedClassEvaluation {
    pub(crate) outer_class_env: Env,
    pub(crate) class_env: Env,
    proto_parent: Option<Gc>,
    ctor_parent: Option<Value>,
    derived: bool,
    keys: Vec<Option<crate::value::PropertyKey>>,
    /// TC39 decorators proposal, "Evaluating decorators": expression values are captured in
    /// source order, interspersed with computed names, and called only after that phase finishes.
    decorator_values: Option<PreparedDecoratorValues>,
}

impl PreparedClassEvaluation {
    /// Publish only after the observable prototype Get has completed with no state loan.
    pub(crate) fn publish_heritage(&mut self, values: (Option<Gc>, Option<Value>, bool)) {
        self.proto_parent = values.0;
        self.ctor_parent = values.1;
        self.derived = values.2;
    }
    /// Publish only after ToPropertyKey has completed; no author code runs here.
    pub(crate) fn publish_key(&mut self, member: usize, key: crate::value::PropertyKey) {
        self.keys[member] = Some(key);
    }
}

struct PreparedDecoratorValues {
    class: Vec<PreparedDecoratorValue>,
    members: Vec<Vec<PreparedDecoratorValue>>,
}

struct PreparedDecoratorValue {
    callback: Value,
    this_value: Value,
}

/// Rust-side state kept across DisposeResources callbacks and await-only markers.
struct DisposeFrame {
    resources: Vec<Disposable>,
    completion: Completion,
}

/// Class records own their keys and callable transforms until publication. A companion list
/// of values is insufficient: ToPropertyKey can create a Symbol not held by its input object.
#[derive(Default)]
struct PendingClassElements {
    instance_fields: Vec<FieldInit>,
    private_members: Vec<(String, Property)>,
    static_elements: Vec<StaticClassElement>,
}

enum StaticClassElement {
    Field(FieldInit),
    Block(Rc<Function>),
}

impl PreparedClassEvaluation {
    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        let mut bytes = self
            .keys
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<crate::value::PropertyKey>>());
        for key in self.keys.iter().flatten() {
            bytes = bytes.saturating_add(key.scan_retained_memory(visitor));
        }
        if let Some(parent) = &self.ctor_parent {
            visitor.value(parent);
        }
        if let Some(decorators) = &self.decorator_values {
            bytes = bytes
                .saturating_add(
                    decorators
                        .class
                        .capacity()
                        .saturating_mul(std::mem::size_of::<PreparedDecoratorValue>()),
                )
                .saturating_add(
                    decorators
                        .members
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Vec<PreparedDecoratorValue>>()),
                );
            for decorator in &decorators.class {
                visitor.value(&decorator.callback);
                visitor.value(&decorator.this_value);
            }
            for member in &decorators.members {
                bytes = bytes.saturating_add(
                    member
                        .capacity()
                        .saturating_mul(std::mem::size_of::<PreparedDecoratorValue>()),
                );
                for decorator in member {
                    visitor.value(&decorator.callback);
                    visitor.value(&decorator.this_value);
                }
            }
        }
        // Class environments and the optional prototype object are canonical to the collector
        // snapshots; this method only credits the continuation-owned buffers and Value payloads.
        bytes
    }
}

impl Interp {
    // ----- statements -------------------------------------------------------------------------

    /// Bind a (possibly destructuring) pattern to `value` in `env`. `Var` assigns to the hoisted
    /// binding; `Lexical` creates fresh bindings (params, `let`/`const`).
    pub(crate) fn bind_pattern(
        &mut self,
        pat: &Pattern,
        value: Value,
        env: &Env,
        mode: BindMode,
    ) -> Result<(), Abrupt> {
        if crate::interpreter::ast_stack_low() {
            return crate::interpreter::with_ast_stack(|| self.bind_pattern(pat, value, env, mode));
        }
        match pat {
            Pattern::Ident(name) => {
                match mode {
                    BindMode::Lexical(is_const) => self.init_lexical(name, value, is_const, env),
                    BindMode::Var => self.assign_var(name, value, env)?,
                }
                Ok(())
            }
            Pattern::Array(elems) => {
                // Fresh lexical names bound from Array elements that execute no code: the
                // interleaving of steps and initializations is unobservable.
                if matches!(mode, BindMode::Lexical(_))
                    && elems.iter().all(|element| {
                        matches!(
                            element,
                            ArrayPatElem::Hole
                                | ArrayPatElem::Elem {
                                    pattern: Pattern::Ident(_),
                                    default: None,
                                }
                        )
                    })
                {
                    // Rooted: binding one element may run code that empties the source.
                    let mut values =
                        crate::rooting::RootedVec::from_vec(Vec::with_capacity(elems.len()));
                    if let Some(Ok(())) =
                        self.inert_array_prefix(&value, elems.len(), |value| values.push(value))
                    {
                        for (element, value) in elems.iter().zip(values) {
                            if let ArrayPatElem::Elem { pattern, .. } = element {
                                self.bind_pattern(pattern, value, env, mode)?;
                            }
                        }
                        return Ok(());
                    }
                }
                // Iterate lazily, pulling only what the pattern needs and closing the iterator when
                // we stop early (or on an abrupt completion).
                let (iter, next) = self.get_iterator(&value)?;
                let mut done = false;
                let result = (|me: &mut Self| -> Result<(), Abrupt> {
                    // A throw from `next` marks the record done (IteratorClose skipped); a throw from
                    // binding a sub-pattern leaves it not-done, so the iterator is still closed.
                    macro_rules! step {
                        () => {
                            match me.iterator_step(&iter, &next) {
                                Ok(v) => v,
                                Err(e) => {
                                    done = true;
                                    return Err(e);
                                }
                            }
                        };
                    }
                    for el in elems {
                        match el {
                            ArrayPatElem::Hole => {
                                if !done && step!().is_none() {
                                    done = true;
                                }
                            }
                            ArrayPatElem::Elem { pattern, default } => {
                                // IteratorBindingInitialization resolves a SingleNameBinding
                                // before stepping. Keep that Reference across user next/default
                                // callbacks; nested patterns instead resolve when entered.
                                let reference = me.prepare_binding_reference(pattern, env, mode)?;
                                let mut v = if done {
                                    Value::Undefined
                                } else {
                                    match step!() {
                                        Some(x) => x,
                                        None => {
                                            done = true;
                                            Value::Undefined
                                        }
                                    }
                                };
                                if matches!((v).view(), ValueView::Undefined) {
                                    if let Some(d) = default {
                                        v = me.eval(d, env)?;
                                        if let (Pattern::Ident(n), true) =
                                            (pattern, is_anonymous_fn(d))
                                        {
                                            me.set_fn_name(&v, n);
                                        }
                                    }
                                }
                                me.finish_binding_reference(reference, pattern, v, env, mode)?;
                            }
                            ArrayPatElem::Rest(pattern) => {
                                let reference = me.prepare_binding_reference(pattern, env, mode)?;
                                let rest = me.drain_iterator_rest(&iter, &next, &mut done)?;
                                let arr = me.make_array(rest);
                                me.finish_binding_reference(reference, pattern, arr, env, mode)?;
                            }
                        }
                    }
                    Ok(())
                })(self);
                // IteratorClose: propagate its abrupt on normal completion, swallow on abrupt.
                match result {
                    Ok(()) => {
                        if !done {
                            self.iterator_close_normal(&iter)?;
                        }
                        Ok(())
                    }
                    Err(e) => {
                        if matches!(e, Abrupt::Interrupt(_)) {
                            return Err(e);
                        }
                        if !done {
                            if matches!(e, Abrupt::Throw(_)) {
                                self.iterator_close(&iter);
                            } else {
                                // A non-throw completion (return/break/continue): a throwing or
                                // non-object `return` replaces it; otherwise it propagates.
                                self.iterator_close_normal(&iter)?;
                            }
                        }
                        Err(e)
                    }
                }
            }
            Pattern::Object(objpat) => {
                if matches!((value).view(), ValueView::Undefined | ValueView::Null) {
                    return Err(self.throw("TypeError", "cannot destructure null or undefined"));
                }
                let mut used: Vec<String> = Vec::new();
                for prop in &objpat.props {
                    let key = self.eval_prop_key(&prop.key, env)?;
                    used.push(key.to_string());
                    // KeyedBindingInitialization order for a var-mode identifier target:
                    // ResolveBinding *before* GetV (observable through a `with` env's has trap).
                    let reference = self.prepare_binding_reference(&prop.value, env, mode)?;
                    let mut v = self.get_member_key(&value, &key)?;
                    if matches!((v).view(), ValueView::Undefined) {
                        if let Some(d) = &prop.default {
                            v = self.eval(d, env)?;
                            if let (Pattern::Ident(n), true) = (&prop.value, is_anonymous_fn(d)) {
                                self.set_fn_name(&v, n);
                            }
                        }
                    }
                    self.finish_binding_reference(reference, &prop.value, v, env, mode)?;
                }
                if let Some(rest_name) = &objpat.rest {
                    let pattern = Pattern::Ident(rest_name.clone());
                    // RestBindingInitialization resolves before CopyDataProperties, whose
                    // ownKeys/getOwnPropertyDescriptor/get operations may change the scope.
                    let reference = self.prepare_binding_reference(&pattern, env, mode)?;
                    let obj = self.copy_data_properties(&value, &used)?;
                    self.finish_binding_reference(reference, &pattern, Value::Obj(obj), env, mode)?;
                }
                Ok(())
            }
            // A member target (`o.p`/`o[k]`): assign to it (never a declaration).
            Pattern::Member(target) => self.assign_to_target(target, value, env),
        }
    }

    fn prepare_binding_reference(
        &mut self,
        pattern: &Pattern,
        env: &Env,
        mode: BindMode,
    ) -> Result<Option<PreparedReference>, Abrupt> {
        if let (BindMode::Var, Pattern::Ident(name)) = (mode, pattern) {
            return self.prepare_name_reference(name, env).map(Some);
        }
        Ok(None)
    }

    fn finish_binding_reference(
        &mut self,
        reference: Option<PreparedReference>,
        pattern: &Pattern,
        value: Value,
        env: &Env,
        mode: BindMode,
    ) -> Result<(), Abrupt> {
        if let Some(mut reference) = reference {
            self.write_prepared_reference(&mut reference, value)
        } else {
            self.bind_pattern(pattern, value, env, mode)
        }
    }

    /// Run a statement list that has already had its bindings instantiated (hoisting + lexical
    /// declaration done by the caller). Used by module evaluation, where the module's environment is
    /// set up during the separate Instantiate phase before any body runs.
    pub(crate) fn run_stmt_list(&mut self, body: &[Stmt], env: &Env) -> Result<Value, Abrupt> {
        let has_using = body.iter().any(stmt_declares_using);
        if has_using {
            self.using_stack.push(Vec::new());
        }
        let mut last = Value::Undefined;
        let mut result: Completion = Ok(Value::Undefined);
        for stmt in body {
            match self.exec_stmt(stmt, env) {
                Ok(v) => {
                    if !matches!((v).view(), ValueView::Empty) {
                        last = v;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        if has_using {
            let frame = self.using_stack.pop().unwrap_or_default();
            result = self.dispose_frame(frame, result);
        }
        result?;
        Ok(last)
    }

    pub(crate) fn exec_block(&mut self, stmts: &[Stmt], parent: &Env) -> Completion {
        // An empty block needs no environment (a hot case in machine-generated code).
        if stmts.is_empty() {
            return Ok(Value::Empty);
        }
        // A block declaring nothing block-scoped needs no environment either: nothing would bind
        // in it, so running the statements in the parent scope is observationally identical and
        // skips a scope allocation + declaration walk per entry (hot for loop and if bodies).
        if !stmts.iter().any(block_declares_lexical) {
            let mut v = Value::Empty;
            for s in stmts {
                match self.exec_stmt(s, parent) {
                    Ok(sv) => {
                        if !matches!((sv).view(), ValueView::Empty) {
                            v = sv;
                        }
                    }
                    Err(e) => return Err(crate::interpreter::update_abrupt_empty(e, v)),
                }
            }
            return Ok(v);
        }
        let scope = new_scope(Some(*parent));
        self.declare_block_lexicals(stmts, &scope, true);
        // A block is a disposal boundary only when it actually declares a `using` resource.
        let has_using = stmts.iter().any(stmt_declares_using);
        if has_using {
            self.using_stack.push(Vec::new());
        }
        // StatementList completion: the last non-EMPTY statement value (V); an abrupt
        // break/continue escaping the block carries V per UpdateEmpty.
        let mut v = Value::Empty;
        let mut result = Ok(Value::Empty);
        for s in stmts {
            match self.exec_stmt(s, &scope) {
                Ok(sv) => {
                    if !matches!((sv).view(), ValueView::Empty) {
                        v = sv;
                    }
                    result = Ok(v);
                }
                Err(e) => {
                    result = Err(crate::interpreter::update_abrupt_empty(e, v));
                    break;
                }
            }
        }
        if has_using {
            let frame = self.using_stack.pop().unwrap_or_default();
            result = self.dispose_frame(frame, result);
        }
        result
    }

    /// Capture a `using` resource's dispose method for disposal at scope exit. `null`/`undefined`
    /// resources are ignored; a non-callable dispose method is a TypeError.
    fn add_disposable(&mut self, value: &Value, is_async: bool) -> Result<(), Abrupt> {
        let Some(resource) = self.create_disposable(value, is_async)? else {
            return Ok(());
        };
        if self.using_stack.is_empty() {
            self.using_stack.push(Vec::new());
        }
        self.using_stack.last_mut().unwrap().push(resource);
        Ok(())
    }

    /// Create one DisposableResource record. Bytecode continuations keep these records in their
    /// own scope stack rather than the interpreter-global tree-walker stack, so two suspended
    /// coroutines cannot accidentally dispose each other's resources.
    pub(crate) fn create_disposable(
        &mut self,
        value: &Value,
        is_async: bool,
    ) -> Result<Option<Disposable>, Abrupt> {
        if matches!(&(value).view(), ValueView::Undefined | ValueView::Null) {
            // An evaluated `await using x = null` still records a pending Await for the
            // block's end (an empty async disposal awaits once).
            if is_async {
                return Ok(Some(Disposable {
                    value: Value::Undefined,
                    method: Value::Undefined,
                    kind_is_async: true,
                    method_is_async: false,
                }));
            }
            return Ok(None);
        }
        let (method, method_is_async) = self.dispose_method(value, is_async)?;
        Ok(Some(Disposable {
            value: *value,
            method,
            kind_is_async: is_async,
            method_is_async,
        }))
    }

    /// GetDisposeMethod: `@@asyncDispose` (falling back to `@@dispose`) for `await using`, else
    /// `@@dispose`. Throws if the resolved method isn't callable.
    fn dispose_method(&mut self, value: &Value, is_async: bool) -> Result<(Value, bool), Abrupt> {
        let mut m = Value::Undefined;
        let mut from_async = false;
        if is_async {
            if let Some(k) = crate::builtins::well_known_key(self, "asyncDispose") {
                m = self.get_member_key(value, &k)?;
                from_async = m.is_callable();
            }
        }
        if matches!((m).view(), ValueView::Undefined | ValueView::Null) {
            if let Some(k) = crate::builtins::well_known_key(self, "dispose") {
                m = self.get_member_key(value, &k)?;
            }
        }
        if !m.is_callable() {
            return Err(self.throw("TypeError", "value is not disposable"));
        }
        Ok((m, from_async))
    }

    /// Dispose a frame's resources in reverse order. An error thrown while disposing either becomes
    /// the completion (if it was previously normal) or is folded into a `SuppressedError` chain.
    pub(crate) fn dispose_frame(
        &mut self,
        frame: Vec<Disposable>,
        result: Completion,
    ) -> Completion {
        // Script termination is host control flow, not ECMAScript abrupt completion cleanup.
        // Like `finally`, explicit-resource-management disposal must not run while HTML aborts
        // the execution-context stack.
        if matches!(result, Err(Abrupt::Interrupt(_))) {
            return result;
        }
        // ECMA-262 DisposeResources tracks an await-only marker separately so a nullish
        // `await using` still yields at least one job turn, at the precise point before a later
        // synchronous resource (or at the end when there is no such resource).
        let mut needs_await = false;
        let mut has_awaited = false;
        let pending = crate::rooting::Rooted::new(std::cell::RefCell::new(DisposeFrame {
            resources: frame,
            completion: result,
        }));
        // Keep the current record in the traced list until its call/await finishes. In
        // particular, an await-only marker can suspend before this resource is called.
        loop {
            let resource = pending.borrow().resources.last().map(|r| Disposable {
                value: r.value,
                method: r.method,
                kind_is_async: r.kind_is_async,
                method_is_async: r.method_is_async,
            });
            let Some(r) = resource else {
                break;
            };
            if !r.kind_is_async && needs_await && !has_awaited {
                self.coro_await(Value::Undefined)?;
                needs_await = false;
            }

            if !r.method.is_callable() {
                debug_assert!(r.kind_is_async);
                needs_await = true;
                pending.borrow_mut().resources.pop();
                continue;
            }

            let called = self.call(r.method, r.value, &[]);
            let disposed = if r.kind_is_async && r.method_is_async {
                // A directly selected @@asyncDispose method is awaited only when Call completed
                // normally. A synchronous throw becomes the disposal result immediately.
                match called {
                    Ok(value) => {
                        has_awaited = true;
                        self.coro_await(value)
                    }
                    Err(error) => Err(error),
                }
            } else if r.kind_is_async {
                // GetDisposeMethod's sync fallback is an unobservable async wrapper: ignore the
                // sync method's return value, translate a synchronous throw to rejection, and
                // Await the wrapper promise.
                let promise = self.new_promise();
                match called {
                    Ok(_) => self.resolve_promise(&promise, Value::Undefined),
                    Err(Abrupt::Throw(error)) => self.reject_promise(&promise, error),
                    Err(other) => return Err(other),
                }
                has_awaited = true;
                self.coro_await(promise)
            } else {
                called
            };
            match disposed {
                Ok(_) => {}
                Err(Abrupt::Throw(new_err)) => {
                    let (previous_throw, interrupted) = {
                        let state = pending.borrow();
                        match &state.completion {
                            Err(Abrupt::Throw(previous)) => (Some(*previous), None),
                            Err(Abrupt::Interrupt(reason)) => (None, Some(*reason)),
                            _ => (None, None),
                        }
                    };
                    // The old completion stays rooted until after SuppressedError construction.
                    let completion = if let Some(reason) = interrupted {
                        Err(Abrupt::Interrupt(reason))
                    } else {
                        Err(Abrupt::Throw(match previous_throw {
                            Some(previous) => self.make_suppressed(new_err, previous),
                            None => new_err,
                        }))
                    };
                    pending.borrow_mut().completion = completion;
                }
                Err(other) => return Err(other),
            }
            pending.borrow_mut().resources.pop();
        }
        if needs_await && !has_awaited {
            self.coro_await(Value::Undefined)?;
        }
        pending.into_inner().into_inner().completion
    }

    /// Run one `for-of` iteration: bind the element and execute the body. For a `using`/`await using`
    /// binding (`dispose` is `Some`), the element is a per-iteration disposal resource — its dispose
    /// method runs (awaited for `await using`) at the end of the iteration.
    fn for_of_iteration(
        &mut self,
        left: &Pattern,
        v: Value,
        mode: BindMode,
        body: &Stmt,
        iter_env: &Env,
        dispose: Option<bool>,
    ) -> Completion {
        match dispose {
            None => {
                self.bind_pattern(left, v, iter_env, mode)?;
                self.exec_stmt(body, iter_env)
            }
            Some(is_async) => {
                self.using_stack.push(Vec::new());
                let step = (|me: &mut Self| -> Completion {
                    me.add_disposable(&v, is_async)?;
                    me.bind_pattern(left, v, iter_env, mode)?;
                    me.exec_stmt(body, iter_env)
                })(self);
                let frame = self.using_stack.pop().unwrap_or_default();
                self.dispose_frame_maybe_async(frame, step, is_async)
            }
        }
    }

    /// Dispose a frame, awaiting each `@@asyncDispose` result when `is_async` (an `await using`
    /// boundary). For a sync boundary this is exactly `dispose_frame`.
    pub(crate) fn dispose_frame_maybe_async(
        &mut self,
        frame: Vec<Disposable>,
        result: Completion,
        _is_async: bool,
    ) -> Completion {
        self.dispose_frame(frame, result)
    }

    /// Build a `SuppressedError(error, suppressed)`.
    pub(crate) fn make_suppressed(&mut self, error: Value, suppressed: Value) -> Value {
        let err = self.make_error("SuppressedError", "");
        if let Some(o) = err.as_obj() {
            o.borrow_mut().proto = self.error_protos.get("SuppressedError").cloned();
            o.borrow_mut()
                .props
                .insert("error", crate::value::Property::builtin(error));
            o.borrow_mut()
                .props
                .insert("suppressed", crate::value::Property::builtin(suppressed));
        }
        err
    }

    /// Pre-declare `let`/`const` (uninitialised — TDZ) and, when `with_functions`, block-level
    /// function declarations (initialised) for the statements directly in a block.
    pub(crate) fn declare_block_lexicals(
        &mut self,
        stmts: &[Stmt],
        scope: &Env,
        with_functions: bool,
    ) {
        for s in stmts {
            // Annex B labelled function declarations (`l: function f() {}`) instantiate at block
            // entry exactly like unlabelled ones — labels are transparent here.
            let mut s = crate::interpreter::unwrap_export(s);
            while let Stmt::Labeled { body, .. } = s {
                s = body;
            }
            match s {
                Stmt::VarDecl { kind, decls }
                    if matches!(
                        kind,
                        DeclKind::Let | DeclKind::Const | DeclKind::Using | DeclKind::AwaitUsing
                    ) =>
                {
                    // ECMA-262 BlockDeclarationInstantiation creates immutable bindings for
                    // `const` and resource declarations before evaluation; initialization clears
                    // TDZ state but does not change that mutability.
                    let immutable = matches!(
                        kind,
                        DeclKind::Const | DeclKind::Using | DeclKind::AwaitUsing
                    );
                    for (pat, _) in decls {
                        let mut names = Vec::new();
                        crate::interpreter::pattern_ident_strs(pat, &mut names);
                        for name in names {
                            let mut b = scope.borrow_mut();
                            // Only a variable environment's lexical attributes are read (by
                            // EvalDeclarationInstantiation's var/lexical check).
                            let lexical = b.var_boundary();
                            b.vars.insert(
                                name,
                                Binding::data(Value::Undefined, !immutable, false).lexical(lexical),
                            );
                        }
                    }
                }
                Stmt::FuncDecl(func) if with_functions => {
                    if let Some(name) = &func.name {
                        let f = self.make_function(func.clone(), *scope);
                        scope
                            .borrow_mut()
                            .vars
                            .insert(name.clone(), Binding::data(f, true, true));
                    }
                }
                Stmt::ClassDecl(class) => {
                    if let Some(name) = &class.name {
                        // Classes are lexically scoped with a TDZ until the declaration executes.
                        let mut b = scope.borrow_mut();
                        let lexical = b.var_boundary();
                        b.vars.insert(
                            name.clone(),
                            Binding::data(Value::Undefined, true, false).lexical(lexical),
                        );
                    }
                }
                _ => {}
            }
        }
    }

    /// The right operand of a logical operator whose left operand evaluated to `left`.
    fn eval_logical_right(
        &mut self,
        op: &str,
        left: Value,
        right: &Expr,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        let take_right = match op {
            "&&" => self.to_boolean(&left),
            "||" => !self.to_boolean(&left),
            "??" => matches!((left).view(), ValueView::Undefined | ValueView::Null),
            _ => unreachable!(),
        };
        if take_right {
            self.eval(right, env)
        } else {
            Ok(left)
        }
    }

    /// A left-nested chain of binary and logical operators (`a + b + c`, `a && b || c`),
    /// evaluated left to right in a loop rather than recursing once per link: generated and
    /// obfuscated code chains hundreds of thousands of operands.
    fn eval_operator_chain(&mut self, expr: &Expr, env: &Env) -> Result<Value, Abrupt> {
        let mut links = Vec::new();
        let mut node = expr;
        loop {
            node = match node {
                Expr::Binary { op, left, right } => {
                    links.push((true, *op, &**right));
                    left
                }
                Expr::Logical { op, left, right } => {
                    links.push((false, *op, &**right));
                    left
                }
                _ => break,
            };
        }
        let mut value = self.eval(node, env)?;
        for (binary, op, right) in links.into_iter().rev() {
            value = if binary {
                let right = self.eval(right, env)?;
                self.binary(op, value, right)?
            } else {
                self.eval_logical_right(op, value, right, env)?
            };
        }
        Ok(value)
    }

    /// [`Self::exec_stmt`] continued on a fresh native stack segment (see `with_ast_stack`); kept
    /// out of line so the gateway adds no frame state to the hot statement dispatch.
    #[cold]
    #[inline(never)]
    fn exec_stmt_on_fresh_stack(&mut self, stmt: &Stmt, env: &Env) -> Completion {
        crate::interpreter::with_ast_stack(|| self.exec_stmt(stmt, env))
    }

    /// [`Self::eval`] continued on a fresh native stack segment, out of line like
    /// [`Self::exec_stmt_on_fresh_stack`].
    #[cold]
    #[inline(never)]
    fn eval_on_fresh_stack(&mut self, expr: &Expr, env: &Env) -> Result<Value, Abrupt> {
        crate::interpreter::with_ast_stack(|| self.eval(expr, env))
    }

    pub(crate) fn exec_stmt(&mut self, stmt: &Stmt, env: &Env) -> Completion {
        self.ast_tick = self.ast_tick.wrapping_add(1);
        if self.ast_tick & 7 == 0 && crate::interpreter::ast_stack_low() {
            return self.exec_stmt_on_fresh_stack(stmt, env);
        }
        match stmt {
            Stmt::Empty | Stmt::Debugger => Ok(Value::Empty),
            Stmt::FuncDecl(func) => {
                // Annex B.3.3 web-compat: evaluating a sloppy block/if-position function
                // declaration copies the block binding's function into the function-scope var
                // binding created at instantiation (see `hoist_block_funcs`).
                if let Some(name) = &func.name {
                    if self
                        .annexb_fn_sync
                        .contains_key(&(Rc::as_ptr(func) as usize))
                    {
                        self.annexb_fn_sync_eval(name, func, env);
                    }
                }
                Ok(Value::Empty)
            }
            Stmt::Expr(e) => self.eval(e, env),
            Stmt::Block(body) => self.exec_block(body, env),
            Stmt::VarDecl { kind, decls } => {
                for (pat, init) in decls {
                    match kind {
                        DeclKind::Var => {
                            // `var x;` (no init) keeps the hoisted binding untouched. For an
                            // identifier target, the binding Reference resolves BEFORE the
                            // initializer runs (a `with` base captured then is written even if
                            // the initializer deletes the property).
                            if let Some(e) = init {
                                if let Pattern::Ident(n) = pat {
                                    if matches!(e, Expr::Class(c) if c.name.is_none()) {
                                        self.pending_fn_name = Some(n.clone());
                                    }
                                    let mut lref = self.resolve_reference(
                                        &Expr::Ident(n.clone(), NO_POSITION),
                                        env,
                                    )?;
                                    let value = self.eval(e, env)?;
                                    self.pending_fn_name = None;
                                    if is_anonymous_fn(e) {
                                        self.set_fn_name(&value, n);
                                    }
                                    self.put_reference(&mut lref, value)?;
                                } else {
                                    let value = self.eval(e, env)?;
                                    self.bind_pattern(pat, value, env, BindMode::Var)?;
                                }
                            }
                        }
                        DeclKind::Let | DeclKind::Const => {
                            if let (Pattern::Ident(n), Some(e)) = (pat, init) {
                                if matches!(e, Expr::Class(c) if c.name.is_none()) {
                                    self.pending_fn_name = Some(n.clone());
                                }
                            }
                            let value = match init {
                                Some(e) => self.eval(e, env)?,
                                None => Value::Undefined,
                            };
                            self.pending_fn_name = None;
                            if let (Pattern::Ident(n), Some(e)) = (pat, init) {
                                if is_anonymous_fn(e) {
                                    self.set_fn_name(&value, n);
                                }
                            }
                            self.bind_pattern(
                                pat,
                                value,
                                env,
                                BindMode::Lexical(*kind == DeclKind::Const),
                            )?;
                        }
                        DeclKind::Using | DeclKind::AwaitUsing => {
                            let is_async = *kind == DeclKind::AwaitUsing;
                            let value = match init {
                                Some(e) => self.eval(e, env)?,
                                None => Value::Undefined,
                            };
                            if let (Pattern::Ident(n), Some(e)) = (pat, init) {
                                if is_anonymous_fn(e) {
                                    self.set_fn_name(&value, n);
                                }
                            }
                            // Capture the dispose method now (TypeError if not disposable).
                            self.add_disposable(&value, is_async)?;
                            // A `using` binding is immutable: assignment is a TypeError.
                            self.bind_pattern(pat, value, env, BindMode::Lexical(true))?;
                        }
                    }
                }
                Ok(Value::Empty)
            }
            Stmt::Return(arg) => {
                // A strict ordinary function's `return <expr>` walks the expression's tail
                // positions (conditional branches, logical right sides, the last of a comma
                // sequence): a call there is a proper tail call, handed to the trampoline in
                // Interp::call so the current frame unwinds first.
                if let Some(e) = arg {
                    if self.tco_ok && !self.using_stack.iter().any(|f| !f.is_empty()) {
                        return match self.eval_return_expr(e, env)? {
                            // A shallow tail call runs as an ordinary call; deep or unbounded
                            // tail recursion still retires this frame first, as compiled code
                            // does (see `bytecode::Op::TailDeep`).
                            TailEval::Tail(f, t, a, position)
                                if self.depth < crate::bytecode::ordinary_tail_call_depth() =>
                            {
                                Err(Abrupt::Return(self.call_at(position, f, t, &a)?))
                            }
                            TailEval::Tail(f, t, a, _) => {
                                self.pending_tail = Some(Box::new((f, t, a.into_vec())));
                                Err(Abrupt::Return(Value::Undefined))
                            }
                            TailEval::Val(v) => Err(Abrupt::Return(v)),
                        };
                    }
                }
                let v = match arg {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Undefined,
                };
                // In an async generator's own body, `return <expr>` awaits the expression (a
                // bare `return`, an implicit completion, or a nested plain function does not).
                let v = if arg.is_some()
                    && self.in_async_gen_body
                    && crate::coroutine::in_async_gen()
                {
                    self.coro_await(v)?
                } else {
                    v
                };
                Err(Abrupt::Return(v))
            }
            Stmt::Throw(e) => {
                let v = self.eval(e, env)?;
                Err(Abrupt::Throw(v))
            }
            Stmt::If { test, cons, alt } => {
                // IfStatement completion: UpdateEmpty(branch, undefined) — never EMPTY, and an
                // abrupt break/continue leaving the branch has its empty value filled too.
                let t = self.eval(test, env)?;
                let r = if self.to_boolean(&t) {
                    self.exec_stmt(cons, env)
                } else if let Some(a) = alt {
                    self.exec_stmt(a, env)
                } else {
                    Ok(Value::Empty)
                };
                match r {
                    Ok(v) => Ok(crate::interpreter::update_empty(v)),
                    Err(e) => Err(crate::interpreter::update_abrupt_empty(e, Value::Undefined)),
                }
            }
            Stmt::While { test, body } => self.exec_while(&body.site, test, body, env, &[]),
            Stmt::DoWhile { body, test } => self.exec_do_while(&body.site, body, test, env, &[]),
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.exec_for(&body.site, init, test, update, body, env, &[]),
            Stmt::ForInOf {
                decl,
                left,
                right,
                of,
                is_await,
                body,
                ..
            } => self.exec_for_in_of(
                &body.site,
                *decl,
                left,
                right,
                *of,
                *is_await,
                body,
                env,
                &[],
            ),
            Stmt::Break(label) => Err(Abrupt::Break(label.clone(), Value::Empty)),
            Stmt::Continue(label) => Err(Abrupt::Continue(label.clone(), Value::Empty)),
            Stmt::Try {
                block,
                handler,
                finalizer,
            } => self.exec_try(block, handler, finalizer, env),
            Stmt::Switch { disc, cases } => self.exec_switch(disc, cases, env),
            Stmt::Labeled { label, body } => self.exec_labeled(label, body, env),
            Stmt::With { obj, body } => {
                let value = self.eval(obj, env)?;
                if matches!((value).view(), ValueView::Undefined | ValueView::Null) {
                    return Err(
                        self.throw("TypeError", "Cannot convert undefined or null to object")
                    );
                }
                // WithStatement Evaluation performs ToObject before creating the Object
                // Environment Record. Keep the resulting wrapper alive for the whole statement:
                // it supplies both HasBinding and WithBaseObject (the receiver of an identifier
                // call such as `charAt(0)`).
                let o = crate::builtins::box_primitive_pub(self, value);
                let with_env = crate::interpreter::new_with_scope(*env, o);
                match self.exec_stmt(body, &with_env) {
                    Ok(v) => Ok(crate::interpreter::update_empty(v)),
                    Err(e) => Err(crate::interpreter::update_abrupt_empty(e, Value::Undefined)),
                }
            }
            Stmt::ClassDecl(class) => {
                let value = self.eval_class(class, env)?;
                if let Some(name) = &class.name {
                    self.init_lexical(name, value, false, env);
                }
                Ok(Value::Empty)
            }
            // Module declarations: imports are resolved at link time (runtime no-op); exports run
            // their inner declaration (the export itself is link-time metadata).
            Stmt::Import(_) | Stmt::ExportNamed { .. } | Stmt::ExportAll { .. } => Ok(Value::Empty),
            Stmt::ExportDecl(inner) => self.exec_stmt(inner, env),
            Stmt::ExportDefault(inner) => match &**inner {
                Stmt::Expr(e) => {
                    if matches!(e, Expr::Class(c) if c.name.is_none()) {
                        self.pending_fn_name = Some("default".to_string());
                    }
                    let v = self.eval(e, env)?;
                    self.pending_fn_name = None;
                    // NamedEvaluation: an anonymous function/class default export is named "default".
                    if is_anonymous_fn(e) {
                        self.set_fn_name(&v, "default");
                    }
                    self.init_lexical("*default*", v, false, env);
                    Ok(Value::Empty)
                }
                // `export default function(){}` / `class{}` with no name: the value is bound to the
                // synthetic `*default*` local (which the "default" export resolves to).
                Stmt::FuncDecl(f) if f.name.is_none() => {
                    let v = self.make_function(f.clone(), *env);
                    self.set_fn_name(&v, "default");
                    self.init_lexical("*default*", v, false, env);
                    Ok(Value::Empty)
                }
                Stmt::ClassDecl(c) if c.name.is_none() => {
                    self.pending_fn_name = Some("default".to_string());
                    let v = self.eval_class(c, env)?;
                    self.pending_fn_name = None;
                    self.set_fn_name(&v, "default");
                    self.init_lexical("*default*", v, false, env);
                    Ok(Value::Empty)
                }
                other => self.exec_stmt(other, env),
            },
        }
    }

    fn exec_labeled(&mut self, label: &str, body: &Stmt, env: &Env) -> Completion {
        // Collect a chain of stacked labels (`a: b: loop`): the labelled statement's inner statement
        // may itself be labelled, and a labelled break/continue can target *any* label in the chain.
        // All labels are threaded into the loop so it can catch break/continue naming any of them.
        let mut labels = vec![label];
        let mut inner = body;
        while let Stmt::Labeled { label, body } = inner {
            labels.push(label);
            inner = body;
        }
        let result = match inner {
            Stmt::For {
                init,
                test,
                update,
                body,
                ..
            } => self.exec_for(&body.site, init, test, update, body, env, &labels),
            Stmt::ForInOf {
                decl,
                left,
                right,
                of,
                is_await,
                body,
                ..
            } => self.exec_for_in_of(
                &body.site, *decl, left, right, *of, *is_await, body, env, &labels,
            ),
            Stmt::While { test, body } => self.exec_while(&body.site, test, body, env, &labels),
            Stmt::DoWhile { body, test } => {
                self.exec_do_while(&body.site, body, test, env, &labels)
            }
            // A non-loop labelled statement (e.g. `a: { … break a; }`): the loop helpers can't catch
            // the break, so it unwinds to the post-match below. Labels also can't be nested here
            // because the `while let` above already peeled the whole chain.
            other => self.exec_stmt(other, env),
        };
        match result {
            Err(Abrupt::Break(Some(l), bv)) if labels.contains(&l.as_str()) => Ok(bv),
            other => other,
        }
    }

    fn exec_while(
        &mut self,
        site: &LoopSite,
        test: &Expr,
        body: &Stmt,
        env: &Env,
        labels: &[&str],
    ) -> Completion {
        use crate::bytecode::loop_fragment::{enter, Admission, Seed, Source};
        let mut admission = Admission::new(self);
        self.run_loop(labels, env, |me, env, completion| {
            if admission.ready() {
                if let Some(result) = enter(
                    me,
                    site,
                    Source::While(test, body),
                    labels,
                    env,
                    completion,
                    Seed::None,
                )? {
                    return Ok(LoopStep::Transferred(result));
                }
            }
            let t = me.eval(test, env)?;
            if !me.to_boolean(&t) {
                return Ok(LoopStep::Done(Value::Empty));
            }
            let bv = me.exec_stmt(body, env)?;
            Ok(LoopStep::Continue(bv))
        })
    }

    fn exec_do_while(
        &mut self,
        site: &LoopSite,
        body: &Stmt,
        test: &Expr,
        env: &Env,
        labels: &[&str],
    ) -> Completion {
        use crate::bytecode::loop_fragment::{enter, Admission, Seed, Source};
        let mut first = true;
        let mut admission = Admission::new(self);
        self.run_loop(labels, env, |me, env, completion| {
            // DoWhileLoopEvaluation evaluates the condition exactly once after a body that
            // completes normally or continues this loop. run_loop handles both completions
            // and retains their UpdateEmpty value; enter their shared condition phase here.
            // A break/return/throw escapes without another condition. ECMA-262 §14.7.2.2.
            if !first {
                let t = me.eval(test, env)?;
                if !me.to_boolean(&t) {
                    return Ok(LoopStep::Done(Value::Empty));
                }
            }
            first = false;
            // The preceding condition has already completed true. The fragment begins with
            // the body, never with a second evaluation of that condition.
            if admission.ready() {
                if let Some(result) = enter(
                    me,
                    site,
                    Source::DoWhile(body, test),
                    labels,
                    env,
                    completion,
                    Seed::None,
                )? {
                    return Ok(LoopStep::Transferred(result));
                }
            }
            let bv = me.exec_stmt(body, env)?;
            Ok(LoopStep::Continue(bv))
        })
    }

    fn run_loop(
        &mut self,
        labels: &[&str],
        env: &Env,
        mut step: impl FnMut(&mut Interp, &Env, &Value) -> Result<LoopStep, Abrupt>,
    ) -> Completion {
        // The loop's completion value: the most recent non-EMPTY body completion (UpdateEmpty);
        // V starts at undefined per ForBodyEvaluation, so a value-less loop completes undefined.
        let mut v = Value::Undefined;
        let keep = |bv: Value, v: &mut Value| {
            if !matches!((bv).view(), ValueView::Empty) {
                *v = bv;
            }
        };
        loop {
            // Safe point: collect when due / enforce the live-object ceiling. Catches tight
            // loops (`for(;;){ x = {}; }`) that never call a function.
            self.interrupt_poll()?;
            self.gc_check()?;
            match step(self, env, &v) {
                Ok(LoopStep::Transferred(result)) => return result,
                Ok(LoopStep::Continue(bv)) => keep(bv, &mut v),
                Ok(LoopStep::Done(bv)) => {
                    keep(bv, &mut v);
                    return Ok(v);
                }
                Err(Abrupt::Break(None, bv)) => {
                    keep(bv, &mut v);
                    return Ok(v);
                }
                Err(Abrupt::Break(Some(l), bv)) if labels.contains(&l.as_str()) => {
                    keep(bv, &mut v);
                    return Ok(v);
                }
                Err(Abrupt::Continue(None, bv)) => keep(bv, &mut v),
                Err(Abrupt::Continue(Some(l), bv)) if labels.contains(&l.as_str()) => {
                    keep(bv, &mut v)
                }
                // A break/continue targeting an outer label: thread this loop's V outward.
                Err(e) => return Err(crate::interpreter::update_abrupt_empty(e, v)),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_for(
        &mut self,
        site: &LoopSite,
        init: &Option<Box<ForInit>>,
        test: &Option<Expr>,
        update: &Option<Expr>,
        body: &Stmt,
        env: &Env,
        labels: &[&str],
    ) -> Completion {
        let loop_env = new_scope(Some(*env));
        // A `for (using x = r; …)` head is a disposal boundary: its resources are disposed once,
        // when the whole loop completes (normally or abruptly).
        let dispose_async = match init.as_deref() {
            Some(ForInit::VarDecl {
                kind: DeclKind::Using,
                ..
            }) => Some(false),
            Some(ForInit::VarDecl {
                kind: DeclKind::AwaitUsing,
                ..
            }) => Some(true),
            _ => None,
        };
        if dispose_async.is_some() {
            self.using_stack.push(Vec::new());
        }
        let result = self.exec_c_for_body(site, init, test, update, body, &loop_env, labels);
        if let Some(is_async) = dispose_async {
            let frame = self.using_stack.pop().unwrap_or_default();
            return self.dispose_frame_maybe_async(frame, result, is_async);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_c_for_body(
        &mut self,
        site: &LoopSite,
        init: &Option<Box<ForInit>>,
        test: &Option<Expr>,
        update: &Option<Expr>,
        body: &Stmt,
        loop_env: &Env,
        labels: &[&str],
    ) -> Completion {
        if let Some(init) = init {
            match init.as_ref() {
                ForInit::Expr(e) => {
                    self.eval(e, loop_env)?;
                }
                ForInit::VarDecl { kind, decls } => {
                    if matches!(kind, DeclKind::Let | DeclKind::Const) {
                        for (pat, _) in decls {
                            let mut names = Vec::new();
                            pattern_idents(pat, &mut names);
                            for name in names {
                                loop_env
                                    .borrow_mut()
                                    .vars
                                    .insert(name, Binding::data(Value::Undefined, true, false));
                            }
                        }
                    }
                    let mode = match kind {
                        DeclKind::Var => BindMode::Var,
                        DeclKind::Let => BindMode::Lexical(false),
                        // const and using/await-using bindings are immutable.
                        _ => BindMode::Lexical(true),
                    };
                    let is_using = matches!(kind, DeclKind::Using | DeclKind::AwaitUsing);
                    let is_async = matches!(kind, DeclKind::AwaitUsing);
                    for (pat, e) in decls {
                        // ECMA-262 §§14.7.4.2 and 14.3.2.1: evaluating a `var`
                        // declaration without an initializer performs no assignment. In
                        // particular, `for (var parameter; ... )` preserves the parameter.
                        if matches!(kind, DeclKind::Var) && e.is_none() {
                            continue;
                        }
                        // ForLoopEvaluation evaluates VariableDeclarationList by its ordinary
                        // algorithm: ResolveBinding precedes the initializer. Retain that exact
                        // with/global base even when evaluation deletes or changes the property.
                        if let (DeclKind::Var, Pattern::Ident(name), Some(initializer)) =
                            (kind, pat, e)
                        {
                            if matches!(initializer, Expr::Class(class) if class.name.is_none()) {
                                self.pending_fn_name = Some(name.clone());
                            }
                            let mut reference = self.resolve_reference(
                                &Expr::Ident(name.clone(), NO_POSITION),
                                loop_env,
                            )?;
                            let value = self.eval(initializer, loop_env)?;
                            self.pending_fn_name = None;
                            if is_anonymous_fn(initializer) {
                                self.set_fn_name(&value, name);
                            }
                            self.put_reference(&mut reference, value)?;
                            continue;
                        }
                        let v = match e {
                            Some(e) => self.eval(e, loop_env)?,
                            None => Value::Undefined,
                        };
                        // A `using`/`await using` init captures its dispose method now.
                        if is_using {
                            self.add_disposable(&v, is_async)?;
                        }
                        self.bind_pattern(pat, v, loop_env, mode)?;
                    }
                }
            }
        }
        // CreatePerIterationEnvironment: a `let` head re-binds its names each round, copying the
        // previous round's values, so closures made in the test/body/update capture that round's
        // bindings. (`const` heads and expression inits share one environment.)
        let per_iter: Vec<crate::atom::Atom> = match init.as_deref() {
            Some(ForInit::VarDecl {
                kind: DeclKind::Let,
                decls,
            }) => {
                let mut names = Vec::new();
                for (pat, _) in decls {
                    pattern_idents(pat, &mut names);
                }
                names
                    .iter()
                    .map(|name| crate::atom::atomize(name))
                    .collect()
            }
            _ => Vec::new(),
        };
        let copy_env = |from: &Env| -> Env {
            let parent = from.borrow().parent;
            let e = new_scope(parent);
            for &n in &per_iter {
                let b = from.borrow().vars.get_atom(n).map(|b| b.to_binding());
                if let Some(b) = b {
                    e.borrow_mut().vars.insert(n, b);
                }
            }
            e
        };
        let mut cur_env = if per_iter.is_empty() {
            *loop_env
        } else {
            copy_env(loop_env)
        };
        let mut first = true;
        use crate::bytecode::loop_fragment::{enter, Admission, Seed, Source};
        let mut admission = Admission::new(self);
        self.run_loop(labels, loop_env, |me, _env, completion| {
            if !first {
                if !per_iter.is_empty() {
                    cur_env = copy_env(&cur_env);
                }
                if let Some(u) = update {
                    me.eval(u, &cur_env)?;
                }
            }
            first = false;
            // CreatePerIterationEnvironment and increment have completed. The fragment starts
            // at the test with this exact live record and takes over subsequent copies.
            if admission.ready() {
                if let Some(result) = enter(
                    me,
                    site,
                    Source::For(init.as_deref(), test.as_ref(), update.as_ref(), body),
                    labels,
                    &cur_env,
                    completion,
                    Seed::None,
                )? {
                    return Ok(LoopStep::Transferred(result));
                }
            }
            if let Some(t) = test {
                let tv = me.eval(t, &cur_env)?;
                if !me.to_boolean(&tv) {
                    return Ok(LoopStep::Done(Value::Empty));
                }
            }
            let bv = me.exec_stmt(body, &cur_env)?;
            Ok(LoopStep::Continue(bv))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_for_in_of(
        &mut self,
        site: &LoopSite,
        decl: Option<DeclKind>,
        left: &Pattern,
        right: &Expr,
        of: bool,
        is_await: bool,
        body: &Stmt,
        env: &Env,
        labels: &[&str],
    ) -> Completion {
        // A lexical head's bound names are already in scope — uninitialized (TDZ) — while the
        // RHS evaluates, so `for (const x of x)` is a ReferenceError.
        let rhs = if matches!(
            decl,
            Some(DeclKind::Let | DeclKind::Const | DeclKind::Using | DeclKind::AwaitUsing)
        ) {
            let tdz = new_scope(Some(*env));
            let mut names = Vec::new();
            pattern_idents(left, &mut names);
            for n in names {
                tdz.borrow_mut()
                    .vars
                    .insert(n, Binding::data(Value::Undefined, true, false));
            }
            self.eval(right, &tdz)?
        } else {
            self.eval(right, env)?
        };
        // No-decl form assigns to an existing binding; a declaration creates a fresh one per round.
        let mode = match decl {
            Some(DeclKind::Var) | None => BindMode::Var,
            Some(DeclKind::Let) => BindMode::Lexical(false),
            // const and using/await-using bindings are immutable.
            Some(_) => BindMode::Lexical(true),
        };
        // A `for (using x of y)` / `for await (await using x of y)` head disposes each element at
        // the end of its iteration.
        let dispose = match decl {
            Some(DeclKind::Using) => Some(false),
            Some(DeclKind::AwaitUsing) => Some(true),
            _ => None,
        };
        if of && is_await {
            // `for await (x of asyncIterable)`: drive the @@asyncIterator (or a sync iterator
            // wrapped async-from-sync), awaiting each `next()` result — and, for a sync source,
            // each value, closing the sync iterator when that await rejects.
            let akey = crate::builtins::async_iterator_key(self);
            let method = match &akey {
                Some(k) => self.get_member_key(&rhs, k)?,
                None => Value::Undefined,
            };
            // GetMethod: a non-callable, non-nullish @@asyncIterator is a TypeError.
            if !matches!((method).view(), ValueView::Undefined | ValueView::Null)
                && !method.is_callable()
            {
                return Err(self.throw("TypeError", "@@asyncIterator is not callable"));
            }
            let (iter, from_sync) = if method.is_callable() {
                let it = self.call(method, rhs, &[])?;
                if !matches!((it).view(), ValueView::Obj(_)) {
                    return Err(self.throw("TypeError", "@@asyncIterator did not return an object"));
                }
                (it, false)
            } else {
                (self.get_iterator(&rhs)?.0, true)
            };
            let next = self.get_member_key(&iter, "next")?;
            let iter_close = iter;
            // A failure in the iteration step itself marks the iterator done — only abrupt
            // completions from the loop body (or an early break/return) close it afterwards.
            let (mut exhausted, mut step_failed) = (false, false);
            let result = self.run_loop(labels, env, |me, env, _completion| {
                step_failed = true;
                let res = me.call(next, iter, &[])?;
                // Await the step result by parking the coroutine (real microtask ticks).
                let res = if from_sync { res } else { me.coro_await(res)? };
                if !matches!((res).view(), ValueView::Obj(_)) {
                    return Err(me.throw("TypeError", "iterator result is not an object"));
                }
                let done = me.get_member_key(&res, "done")?;
                let done = me.to_boolean(&done);
                let v = if from_sync {
                    let raw = me.get_member_key(&res, "value")?;
                    // AsyncFromSyncIteratorContinuation: PromiseResolve(%Promise%, value) reads
                    // the value's constructor observably; an abrupt read rejects the wrapper
                    // promise, so the throw lands a tick later at the await. A rejection closes
                    // the sync iterator (closeOnRejection) before rethrowing.
                    let px = match me.promise_resolve_checked(raw) {
                        // The wrapper promise settles via its own reaction job (one tick), and
                        // the outer Await adds another — two ticks per unwrapped value, like the
                        // spec's PerformPromiseThen into the AsyncFromSyncIterator capability.
                        Ok(p) => me.promise_then(&p, Value::Undefined, Value::Undefined),
                        // IfAbruptRejectPromise: the wrapper rejects directly (no extra hop).
                        Err(e) => {
                            let p = me.new_promise();
                            me.reject_promise(&p, e);
                            p
                        }
                    };
                    match me.coro_await(px) {
                        Ok(v) => v,
                        Err(e) => {
                            if !done {
                                me.iterator_close(&iter);
                            }
                            return Err(e);
                        }
                    }
                } else if done {
                    Value::Undefined
                } else {
                    me.get_member_key(&res, "value")?
                };
                if done {
                    exhausted = true;
                    return Ok(LoopStep::Done(Value::Empty));
                }
                step_failed = false;
                let iter_env = new_scope(Some(*env));
                let bv = me.for_of_iteration(left, v, mode, body, &iter_env, dispose)?;
                Ok(LoopStep::Continue(bv))
            });
            if !(exhausted || step_failed && result.is_err()) {
                // AsyncIteratorClose: a throw completion keeps its error (close failures are
                // swallowed); close failures surface for a normal or return completion.
                let close = self.async_for_await_close(&iter_close, from_sync);
                if !matches!(result, Err(Abrupt::Throw(_))) {
                    close?;
                }
            }
            return result;
        }
        if of {
            use crate::bytecode::loop_fragment::{enter, Admission, Seed, Source};
            // Step the iterator lazily; close it if the loop exits early (break/return/throw).
            let (iter, next) = self.get_iterator(&rhs)?;
            let iter_close = iter;
            let mut exhausted = false;
            // A failure in the iteration step itself (next throwing, a non-object result, or a
            // `value` getter throwing) marks the iterator done: it is NOT closed.
            let mut step_failed = false;
            let mut transferred = false;
            let mut admission = Admission::new(self);
            let result = self.run_loop(labels, env, |me, env, completion| {
                if admission.ready() {
                    if let Some(result) = enter(
                        me,
                        site,
                        Source::ForInOf(decl, left, body, true),
                        labels,
                        env,
                        completion,
                        Seed::ForOf(&iter, &next),
                    )? {
                        transferred = true;
                        return Ok(LoopStep::Transferred(result));
                    }
                }
                step_failed = true;
                let v = match me.iterator_step(&iter, &next)? {
                    Some(x) => x,
                    None => {
                        exhausted = true;
                        return Ok(LoopStep::Done(Value::Empty));
                    }
                };
                step_failed = false;
                let iter_env = new_scope(Some(*env));
                let bv = me.for_of_iteration(left, v, mode, body, &iter_env, dispose)?;
                Ok(LoopStep::Continue(bv))
            });
            if !(transferred || exhausted || step_failed && result.is_err()) {
                match &result {
                    // A throw completion swallows close errors; every other completion (normal,
                    // break, return) propagates them and requires an Object result.
                    Err(Abrupt::Throw(_)) => self.iterator_close(&iter_close),
                    _ => self.iterator_close_normal(&iter_close)?,
                }
            }
            return result;
        }
        let items = self.for_in_keys(&rhs)?;
        let mut idx = 0;
        use crate::bytecode::loop_fragment::{enter, Admission, Seed, Source};
        let mut admission = Admission::new(self);
        self.run_loop(labels, env, |me, env, completion| {
            if admission.ready() {
                if let Some(result) = enter(
                    me,
                    site,
                    Source::ForInOf(decl, left, body, false),
                    labels,
                    env,
                    completion,
                    Seed::ForIn(&rhs, &items, idx),
                )? {
                    return Ok(LoopStep::Transferred(result));
                }
            }
            // A property deleted while the enumeration is under way is not visited.
            let v = loop {
                let Some(v) = items.key(idx) else {
                    return Ok(LoopStep::Done(Value::Empty));
                };
                idx += 1;
                if let ValueView::Str(k) = &(v).view() {
                    // (A string-primitive RHS's index keys are always present.)
                    if matches!((rhs).view(), ValueView::Obj(_))
                        && !me.js_has_property(&rhs, &k.text())?
                    {
                        continue;
                    }
                }
                // Keep ordinary candidate strings and decode only escaped leading-NUL keys.
                break match v.view() {
                    ValueView::Str(key) if key.text().starts_with("\0\0") => {
                        crate::value::string_key_value(&*key.text())
                    }
                    _ => v,
                };
            };
            let iter_env = new_scope(Some(*env));
            me.bind_pattern(left, v, &iter_env, mode)?;
            let bv = me.exec_stmt(body, &iter_env)?;
            Ok(LoopStep::Continue(bv))
        })
    }

    /// `yield value`: park the coroutine, then resume per the driver's signal.
    fn yield_one(&mut self, value: Value) -> Completion {
        // AsyncGeneratorYield: an async generator awaits its operand before suspending, so a
        // rejected/thrown awaited value makes the `yield` itself complete abruptly.
        let value = if crate::coroutine::in_async_gen() {
            match crate::coroutine::coroutine_await(self, value) {
                crate::coroutine::Resume::Next(v) => v,
                crate::coroutine::Resume::Throw(e) => return Err(Abrupt::Throw(e)),
                crate::coroutine::Resume::Return(v) => return Err(Abrupt::Return(v)),
                crate::coroutine::Resume::Terminate => {
                    return Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled));
                }
            }
        } else {
            value
        };
        match crate::coroutine::coroutine_yield(self, value) {
            crate::coroutine::Resume::Next(v) => Ok(v),
            crate::coroutine::Resume::Return(v) => {
                // AsyncGeneratorUnwrapYieldResumption: a return() value is awaited before the
                // yield completes; its rejection becomes a (catchable) throw at the yield.
                if crate::coroutine::in_async_gen() {
                    match crate::coroutine::coroutine_await(self, v) {
                        crate::coroutine::Resume::Next(x) | crate::coroutine::Resume::Return(x) => {
                            Err(Abrupt::Return(x))
                        }
                        crate::coroutine::Resume::Throw(e) => Err(Abrupt::Throw(e)),
                        crate::coroutine::Resume::Terminate => {
                            Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled))
                        }
                    }
                } else {
                    Err(Abrupt::Return(v))
                }
            }
            crate::coroutine::Resume::Throw(e) => Err(Abrupt::Throw(e)),
            crate::coroutine::Resume::Terminate => {
                Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled))
            }
        }
    }

    /// `yield* iterable`: delegate to the inner iterator, forwarding next/return/throw (14.4.14).
    fn yield_delegate(&mut self, value: &Value) -> Completion {
        use crate::coroutine::Resume;
        let (iterator, next) = self.get_iterator(value)?;
        let mut received = Resume::Next(Value::Undefined);
        loop {
            let (result, returning) = match received {
                Resume::Next(v) => (self.call(next, iterator, &[v])?, false),
                Resume::Throw(e) => {
                    // GetMethod(iterator, "throw"): a non-callable non-nullish value throws; an
                    // absent method is a protocol violation — close the inner iterator (close
                    // errors propagate) and then throw TypeError.
                    let throw = self.get_member_key(&iterator, "throw")?;
                    if matches!((throw).view(), ValueView::Undefined | ValueView::Null) {
                        self.iterator_close_normal(&iterator)?;
                        return Err(
                            self.throw("TypeError", "the delegated iterator has no 'throw' method")
                        );
                    }
                    if !throw.is_callable() {
                        return Err(self.throw("TypeError", "iterator 'throw' is not callable"));
                    }
                    (self.call(throw, iterator, &[e])?, false)
                }
                Resume::Return(v) => {
                    let ret = self.get_member_key(&iterator, "return")?;
                    if matches!((ret).view(), ValueView::Undefined | ValueView::Null) {
                        return Err(Abrupt::Return(v));
                    }
                    if !ret.is_callable() {
                        return Err(self.throw("TypeError", "iterator 'return' is not callable"));
                    }
                    (self.call(ret, iterator, &[v])?, true)
                }
                Resume::Terminate => {
                    return Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled));
                }
            };
            if !matches!((result).view(), ValueView::Obj(_)) {
                return Err(self.throw("TypeError", "iterator result is not an object"));
            }
            let done = self.get_member_key(&result, "done")?;
            if self.to_boolean(&done) {
                let v = self.get_member_key(&result, "value")?;
                return if returning {
                    Err(Abrupt::Return(v))
                } else {
                    Ok(v)
                };
            }
            // GeneratorYield forwards the inner result object as-is — its `value` is not read
            // here, and the driver must not re-wrap it.
            self.yield_raw_result = true;
            received = crate::coroutine::coroutine_yield(self, result);
        }
    }

    /// AsyncIteratorClose for the inlined `for await` loop: fetch `return`, call it with no
    /// argument, and (per the async-from-sync wrapper) validate/unwrap its result.
    fn async_for_await_close(&mut self, iter: &Value, from_sync: bool) -> Result<(), Abrupt> {
        let ret = self.get_member_key(iter, "return")?;
        if matches!((ret).view(), ValueView::Undefined | ValueView::Null) {
            return Ok(());
        }
        if !ret.is_callable() {
            return Err(self.throw("TypeError", "iterator 'return' is not callable"));
        }
        let r = self.call(ret, *iter, &[])?;
        let r = if from_sync { r } else { self.await_value(r)? };
        if !matches!((r).view(), ValueView::Obj(_)) {
            return Err(self.throw("TypeError", "iterator result is not an object"));
        }
        if from_sync {
            let raw = self.get_member_key(&r, "value")?;
            self.await_value(raw)?;
        }
        Ok(())
    }

    /// Await `v`: park the coroutine until it settles, surfacing the resume signal. Outside a
    /// coroutine (top-level await in module code) this is the synchronous event-loop drive.
    fn coro_await(&mut self, v: Value) -> Completion {
        if !crate::coroutine::in_coroutine() {
            return self.await_value(v);
        }
        match crate::coroutine::coroutine_await(self, v) {
            crate::coroutine::Resume::Next(x) => Ok(x),
            crate::coroutine::Resume::Throw(e) => Err(Abrupt::Throw(e)),
            crate::coroutine::Resume::Return(rv) => Err(Abrupt::Return(rv)),
            crate::coroutine::Resume::Terminate => {
                Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled))
            }
        }
    }

    /// AsyncGeneratorYield for `yield*` delegation: await `value`, suspend producing it, and return
    /// the driver's resume signal to continue the delegation loop.
    fn async_gen_yield_resume(&mut self, value: Value) -> Result<crate::coroutine::Resume, Abrupt> {
        use crate::coroutine::Resume;
        // The delegated value passes through unawaited: `yield*` never unwraps a promise
        // produced by the inner iterator.
        Ok(match crate::coroutine::coroutine_yield(self, value) {
            // AsyncGeneratorUnwrapYieldResumption: await a return() value before the delegation
            // loop sees it; a rejection turns into a throw resumption.
            Resume::Return(v) => match crate::coroutine::coroutine_await(self, v) {
                Resume::Next(x) | Resume::Return(x) => Resume::Return(x),
                Resume::Throw(e) => Resume::Throw(e),
                Resume::Terminate => Resume::Terminate,
            },
            other => other,
        })
    }

    /// `yield* iterable` inside an *async* generator (14.4.14, async path): drive the inner async
    /// iterator (or a sync one wrapped async-from-sync), awaiting each step result and value, and
    /// re-yielding through AsyncGeneratorYield.
    fn yield_delegate_async(&mut self, value: &Value) -> Completion {
        use crate::coroutine::Resume;
        // GetIterator(value, async): prefer @@asyncIterator; else a sync iterator whose values are
        // awaited (CreateAsyncFromSyncIterator).
        let akey = crate::builtins::async_iterator_key(self);
        let amethod = match &akey {
            Some(k) => self.get_member_key(value, k)?,
            None => Value::Undefined,
        };
        // GetMethod: a non-callable, non-nullish @@asyncIterator is a TypeError (no sync fallback).
        if !matches!((amethod).view(), ValueView::Undefined | ValueView::Null)
            && !amethod.is_callable()
        {
            return Err(self.throw("TypeError", "@@asyncIterator is not callable"));
        }
        let (iterator, next, from_sync) = if amethod.is_callable() {
            let it = self.call(amethod, *value, &[])?;
            if !matches!((it).view(), ValueView::Obj(_)) {
                return Err(self.throw("TypeError", "@@asyncIterator did not return an object"));
            }
            let n = self.get_member_key(&it, "next")?;
            (it, n, false)
        } else {
            let (it, n) = self.get_iterator(value)?;
            (it, n, true)
        };
        // Read a settled iterator result: await the call result; validate it is an object; for a
        // sync source also await the `value` field. Returns (done, value).
        macro_rules! settle {
            ($result:expr) => {{
                let r = self.coro_await($result)?;
                if !matches!(r.view(), ValueView::Obj(_)) {
                    return Err(self.throw("TypeError", "iterator result is not an object"));
                }
                let done = self.get_member_key(&r, "done")?;
                let done = self.to_boolean(&done);
                let mut v = self.get_member_key(&r, "value")?;
                if from_sync {
                    // AsyncFromSyncIteratorContinuation with closeOnRejection: an abrupt unwrap of
                    // a live (not-done) step closes the sync iterator before rethrowing.
                    v = match self.coro_await(v) {
                        Ok(x) => x,
                        Err(e) => {
                            if !done {
                                self.iterator_close(&iterator);
                            }
                            return Err(e);
                        }
                    };
                }
                (done, v)
            }};
        }
        let mut received = Resume::Next(Value::Undefined);
        loop {
            match received {
                Resume::Next(v) => {
                    let result = self.call(next, iterator, &[v])?;
                    let (done, inner) = settle!(result);
                    if done {
                        return Ok(inner);
                    }
                    received = self.async_gen_yield_resume(inner)?;
                }
                Resume::Throw(e) => {
                    let throw = self.get_member_key(&iterator, "throw")?;
                    if !throw.is_callable() {
                        // No `throw` method: close the iterator (awaiting the close) then TypeError.
                        let ret = self.get_member_key(&iterator, "return")?;
                        if ret.is_callable() {
                            if let Ok(r) = self.call(ret, iterator, &[]) {
                                let _ = self.coro_await(r);
                            }
                        }
                        return Err(
                            self.throw("TypeError", "the delegated iterator has no 'throw' method")
                        );
                    }
                    let result = self.call(throw, iterator, &[e])?;
                    let (done, inner) = settle!(result);
                    if done {
                        return Ok(inner);
                    }
                    received = self.async_gen_yield_resume(inner)?;
                }
                Resume::Return(v) => {
                    let ret = self.get_member_key(&iterator, "return")?;
                    if !ret.is_callable() {
                        return Err(Abrupt::Return(self.coro_await(v)?));
                    }
                    let result = self.call(ret, iterator, &[v])?;
                    let (done, inner) = settle!(result);
                    if done {
                        return Err(Abrupt::Return(inner));
                    }
                    received = self.async_gen_yield_resume(inner)?;
                }
                Resume::Terminate => {
                    return Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled));
                }
            }
        }
    }

    /// GetIterator: returns (iterator, next-method).
    pub(crate) fn get_iterator(&mut self, v: &Value) -> Result<(Value, Value), Abrupt> {
        // Under the intrinsic Array protocol, reading @@iterator, calling
        // %Array.prototype.values% and reading `next` execute no ECMAScript code.
        if let Some(next) = crate::builtins::intrinsic_array_iteration_next(self, v) {
            return Ok((crate::builtins::array_values_iterator(self, *v, 0), next));
        }
        let perf_started = crate::jit::perf_stage_start();
        let result: Result<(Value, Value), Abrupt> = (|| {
            // The key formatted once when @@iterator was minted, not a new string per GetIterator.
            let Some(key) = crate::builtins::well_known_key(self, "iterator") else {
                return Err(self.throw("TypeError", "no iterator symbol"));
            };
            let itfn = self.get_member_key(v, &key)?;
            if !itfn.is_callable() {
                return Err(self.throw("TypeError", "value is not iterable"));
            }
            let iter = self.call(itfn, *v, &[])?;
            // GetIteratorFromMethod: the iterator must be an object.
            if !matches!((iter).view(), ValueView::Obj(_)) {
                return Err(self.throw("TypeError", "@@iterator returned a non-object"));
            }
            // GetIterator only *reads* `next`; it is validated as callable when actually called
            // (IteratorNext), so a missing/non-callable `next` doesn't fail at open time.
            let next = self.get_member_key(&iter, "next")?;
            Ok((iter, next))
        })();
        crate::jit::perf_iterator_get_end(perf_started, result.is_ok());
        result
    }
    /// IteratorStep: `Some(value)` or `None` when done.
    pub(crate) fn iterator_step(
        &mut self,
        iter: &Value,
        next: &Value,
    ) -> Result<Option<Value>, Abrupt> {
        let perf_started = crate::jit::perf_stage_start();
        let result: Result<Option<Value>, Abrupt> = (|| {
            if let Some(result) = self.try_intrinsic_iterator_step(iter, next) {
                return result;
            }
            let res = self.call(*next, *iter, &[])?;
            self.iterator_result_value(res)
        })();
        crate::jit::perf_iterator_step_end(perf_started, result.is_ok());
        result
    }

    /// IteratorComplete followed by IteratorValue only for a non-exhausted result.
    /// Custom iterator getters remain observable in precisely this order.
    pub(crate) fn iterator_result_value(&mut self, result: Value) -> Result<Option<Value>, Abrupt> {
        if !matches!((result).view(), ValueView::Obj(_)) {
            return Err(self.throw("TypeError", "iterator result is not an object"));
        }
        let done = self.get_member_key(&result, "done")?;
        if self.to_boolean(&done) {
            Ok(None)
        } else {
            Ok(Some(self.get_member_key(&result, "value")?))
        }
    }
    /// IteratorClose: call `return()` if present (swallowing its result/most errors).
    pub(crate) fn iterator_close(&mut self, iter: &Value) {
        if crate::builtins::array_iterator_close_is_inert(self, iter) {
            return;
        }
        let perf_started = crate::jit::perf_stage_start();
        if let Ok(ret) = self.get_member_key(iter, "return") {
            if ret.is_callable() {
                let _ = self.call(ret, *iter, &[]);
            }
        }
        crate::jit::perf_iterator_close_end(perf_started);
    }

    /// IteratorClose for a *normal* completion: errors from reading or calling `return` propagate
    /// (unlike the throw-completion `iterator_close`, which swallows them).
    pub(crate) fn iterator_close_normal(&mut self, iter: &Value) -> Result<(), Abrupt> {
        if crate::builtins::array_iterator_close_is_inert(self, iter) {
            return Ok(());
        }
        let perf_started = crate::jit::perf_stage_start();
        let result: Result<(), Abrupt> = (|| {
            let ret = self.get_member_key(iter, "return")?;
            if !matches!((ret).view(), ValueView::Undefined | ValueView::Null) {
                if !ret.is_callable() {
                    return Err(self.throw("TypeError", "iterator 'return' is not callable"));
                }
                let r = self.call(ret, *iter, &[])?;
                if !matches!((r).view(), ValueView::Obj(_)) {
                    return Err(self.throw("TypeError", "iterator 'return' must return an object"));
                }
            }
            Ok(())
        })();
        crate::jit::perf_iterator_close_end(perf_started);
        result
    }

    /// Collect every value an iterable yields. Strings and ordinary arrays use a fast path only
    /// while their intrinsic iterator methods are provably unmodified; otherwise this follows
    /// GetIterator (call `@@iterator`, then drain `.next()`) exactly.
    /// The values `v` yields. Rooted while author code runs (I1), and returned rooted: callers
    /// usually run more author code before consuming them.
    pub(crate) fn iterate(&mut self, v: &Value) -> Result<crate::rooting::RootedVec, Abrupt> {
        match &(v).view() {
            ValueView::Str(s) if crate::builtins::intrinsic_string_iterator_is_unmodified(self) => {
                let perf_started = crate::jit::perf_stage_start();
                // String.prototype[@@iterator] returns one String value per code point
                // (ECMA-262 §22.1.3.36).  ASCII code points are also one UTF-16 code unit, so
                // use the thread-local immutable single-unit values instead of allocating a new
                // LStr for every character.  Non-ASCII points retain the canonical UTF-16
                // reconstruction, including lone-surrogate handling.
                if s.ascii_hint() {
                    let result = Ok(s
                        .text()
                        .bytes()
                        .map(|unit| Value::Str(crate::jstr::unit_lstr(unit as u16)))
                        .collect());
                    crate::jit::perf_iterate_fast_end(perf_started);
                    return result;
                }
                let result = Ok(crate::jstr::CodePointIter::new(&s.text())
                    .map(|point| {
                        if point < 0x80 {
                            Value::Str(crate::jstr::unit_lstr(point as u16))
                        } else {
                            Value::lstr(crate::jstr::from_code_point(point))
                        }
                    })
                    .collect());
                crate::jit::perf_iterate_fast_end(perf_started);
                return result;
            }
            ValueView::Obj(o) if crate::builtins::array_iterator_fast_path_is_safe(self, v) => {
                let perf_started = crate::jit::perf_stage_start();
                let result: Result<crate::rooting::RootedVec, Abrupt> = (|| {
                    // %ArrayIteratorPrototype%.next performs LengthOfArrayLike followed by
                    // Get(array, ToString(index)) (ECMA-262 §23.1.5.2.1).  A dense ordinary
                    // array whose indexed entries are all own plain data properties has no
                    // observable work in that Get, so reuse the packed storage directly.  The
                    // helper deliberately returns None for holes/accessors and those arrays
                    // retain the checked property path below.
                    if let Some(values) = crate::builtins::dense_array_snapshot(self, v) {
                        return Ok(crate::rooting::RootedVec::from_vec(values));
                    }
                    let len = self.checked_array_len(o)?;
                    let mut out =
                        crate::rooting::RootedVec::from_vec(Vec::with_capacity(len.min(1024)));
                    for i in 0..len {
                        out.push(self.get_member_key(v, &i.to_string())?);
                    }
                    Ok(out)
                })();
                crate::jit::perf_iterate_fast_end(perf_started);
                return result;
            }
            _ => {}
        }
        // General iterator protocol.
        if let Some(key) = crate::builtins::well_known_key(self, "iterator") {
            let itfn = self.get_member_key(v, &key)?;
            if itfn.is_callable() {
                return self.iterate_with(v, itfn);
            }
        }
        Err(self.throw("TypeError", "value is not iterable"))
    }

    /// Drive the iterator protocol on `v` using an already-resolved `@@iterator` method `itfn`
    /// (so callers that fetched it via GetMethod don't re-read the property).
    pub(crate) fn iterate_with(
        &mut self,
        v: &Value,
        itfn: Value,
    ) -> Result<crate::rooting::RootedVec, Abrupt> {
        let perf_started = crate::jit::perf_stage_start();
        let result: Result<crate::rooting::RootedVec, Abrupt> = (|| {
            let iter = self.call(itfn, *v, &[])?;
            // GetIteratorFromMethod rejects a primitive result before creating an Iterator Record;
            // there is therefore nothing to close in this case.
            if !matches!((iter).view(), ValueView::Obj(_)) {
                return Err(self.throw("TypeError", "@@iterator returned a non-object"));
            }
            let next = self.get_member_key(&iter, "next")?;
            if !next.is_callable() {
                // GetIteratorDirect stores a non-callable next method in the record and the
                // subsequent IteratorStep call closes that record on its abrupt completion.
                let error = self.throw("TypeError", "iterator.next is not a function");
                self.iterator_close(&iter);
                return Err(error);
            }
            let mut out = crate::rooting::RootedVec::new();
            loop {
                let step = self.iterator_step(&iter, &next);
                match step {
                    Ok(None) => break,
                    Ok(Some(value)) => {
                        out.push(value);
                        if out.len() > crate::interpreter::MAX_ARRAY_OP_LEN {
                            let error =
                                self.throw("RangeError", "iterator produced too many values");
                            self.iterator_close(&iter);
                            return Err(error);
                        }
                    }
                    Err(error) => {
                        // Spread/ArgumentListEvaluation use IfAbruptCloseIterator: preserve the
                        // original abrupt completion and swallow any error from return().
                        self.iterator_close(&iter);
                        return Err(error);
                    }
                }
            }
            Ok(out)
        })();
        crate::jit::perf_iterate_protocol_end(perf_started, result.is_ok());
        result
    }

    /// Snapshot the candidate keys for EnumerateObjectProperties. Each prototype level's
    /// [[OwnPropertyKeys]] is consulted once; the loop rechecks deletion before yielding a key.
    pub(crate) fn for_in_keys(&mut self, v: &Value) -> Result<ForInKeys, Abrupt> {
        // EnumerateObjectProperties / CreateForInIterator, ECMA-262 snapshot e28783d5fc9d.
        // Reuse immutable backing only when the complete live ordinary chain proves the same
        // ordered keys, shadowing and enumerable flags. Exotics use their internal methods;
        // the loop still checks deletion immediately before yielding each candidate.
        let shape = self.enumeration_shape(v, None);
        if let Some(keys) = shape.as_ref().and_then(|shape| {
            self.enumeration_keys
                .get_mapped(shape, |entry| entry.keys.clone())
        }) {
            return Ok(keys);
        }
        // A miss pins the exact layouts before building candidates. Both walks run no author
        // code, GC callback or mutation; a complete ordinary-chain proof excludes exotics.
        // Rechecking also fails closed if a future proof extension cannot pin the same chain.
        let mut layouts: EnumerationLayouts = std::array::from_fn(|_| None);
        let pinned = shape.as_ref().is_some_and(|expected| {
            self.enumeration_shape(v, Some(&mut layouts)).as_ref() == Some(expected)
        });
        let keys = self.enum_keys(v)?;
        // A module namespace's [[GetOwnProperty]] runs during enumeration, so an uninitialized
        // export makes the loop throw before any iteration.
        if let ValueView::Obj(object) = &(v).view() {
            let ptr = crate::value::Gc::as_ptr(object) as usize;
            if self.is_namespace(ptr) {
                for key in &keys {
                    if let Some(result) = self.namespace_own_property(ptr, key) {
                        result?;
                    }
                }
            }
        }
        let (keys, bytes) = ForInKeys::new(keys);
        if let Some(shape) = shape.filter(|_| pinned) {
            let entry = CachedForInKeys {
                keys: keys.clone(),
                layouts,
            };
            let bytes = bytes.saturating_add(entry.retained_layout_bytes());
            self.enumeration_keys.insert(shape, entry, bytes);
        }
        Ok(keys)
    }

    fn enumeration_shape(
        &self,
        value: &Value,
        mut pins: Option<&mut EnumerationLayouts>,
    ) -> Option<EnumerationShape> {
        let mut current = Gc::as_ptr(&value.as_obj()?);
        let mut shape = EnumerationShape {
            levels: [(0, 0, 0); 6],
            depth: 0,
        };
        loop {
            if shape.depth as usize == shape.levels.len() {
                return None;
            }
            // The receiver transitively owns every prototype. This walk runs no author code.
            let object = unsafe { (*current).borrow() };
            if !object.ic_plain.get() || !object.exotic.is_ordinary() {
                return None;
            }
            let (layout, prefix, enumerable) = object.props.enumeration_layout()?;
            let index = shape.depth as usize;
            shape.levels[index] = (
                layout.map_or(0, |keys| keys.as_ptr() as usize),
                prefix,
                enumerable,
            );
            if let Some(pins) = pins.as_mut() {
                pins[index] = layout.cloned();
            }
            shape.depth += 1;
            match object.proto.as_ref() {
                Some(parent) => current = Gc::as_ptr(parent),
                None => return Some(shape),
            }
        }
    }

    fn enum_keys(&mut self, v: &Value) -> Result<Vec<String>, Abrupt> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        // A string primitive enumerates its index keys (ToObject makes them own properties).
        if let ValueView::Str(sv) = &(v).view() {
            for k in 0..sv.unit_len() {
                out.push(k.to_string());
            }
            return Ok(out);
        }
        let mut cur = match &(v).view() {
            ValueView::Obj(o) => Some(*o),
            _ => None,
        };
        while let Some(o) = cur {
            let ov = Value::Obj(o);
            // A proxy level enumerates via its [[OwnPropertyKeys]] filtered by [[GetOwnProperty]]'s
            // enumerable flag, then walks its [[GetPrototypeOf]].
            if self.proxies.contains_key(&(Gc::as_ptr(&o) as usize)) {
                let keys =
                    crate::builtins::proxy_enum_string_keys(self, &ov).map_err(Abrupt::Throw)?;
                for k in keys {
                    if let ValueView::Str(ks) = (k).view() {
                        let key = crate::value::encode_owned_string_key(ks.to_string());
                        if seen.insert(key.clone()) {
                            out.push(key);
                        }
                    }
                }
                let parent =
                    crate::builtins::js_get_prototype_of(self, &ov).map_err(Abrupt::Throw)?;
                cur = match (parent).view() {
                    ValueView::Obj(p) => Some(p),
                    _ => None,
                };
                continue;
            }
            if self.host_indexed_len(&o).is_some() {
                // Snapshot platform keys once; test each current descriptor.
                // Native named properties are unenumerable but still shadow.
                let keys =
                    crate::builtins::ordinary_own_keys_ordered(self, &o).map_err(Abrupt::Throw)?;
                for key in keys {
                    if Self::is_sym_key(&key) {
                        continue;
                    }
                    let enumerable = self
                        .host_platform_enumerable(&o, &key)?
                        .or_else(|| o.borrow().props.get(&key).map(|p| p.enumerable()));
                    if let Some(enumerable) = enumerable {
                        if seen.insert(key.clone()) && enumerable {
                            out.push(key);
                        }
                    }
                }
                cur = match (crate::builtins::js_get_prototype_of(self, &Value::Obj(o))
                    .map_err(Abrupt::Throw)?)
                .view()
                {
                    ValueView::Obj(parent) => Some(parent),
                    _ => None,
                };
                continue;
            }
            // for-in visits own enumerable string keys in spec order, then up the prototype chain.
            // TypedArray elements enumerate first (they live outside the property map).
            let level_keys: Vec<String> = o
                .borrow()
                .props
                .ordered_keys()
                .into_iter()
                .filter(|key| !key.is_symbol())
                .map(|key| key.text().to_string())
                .collect();
            if let Some(info) = self.typed_arrays.get(&(Gc::as_ptr(&o) as usize)).copied() {
                for idx in 0..self.ta_len(&info).unwrap_or(0) {
                    let k = idx.to_string();
                    if seen.insert(k.clone()) {
                        out.push(k);
                    }
                }
            }
            for k in level_keys {
                let descriptor = o
                    .borrow()
                    .props
                    .get(&k)
                    .map(|property| property.enumerable());
                if let Some(enumerable) = descriptor {
                    // A non-enumerable own property still *shadows* an enumerable prototype one.
                    if seen.insert(k.clone()) && enumerable {
                        out.push(k);
                    }
                }
            }
            cur = o.borrow().proto;
        }
        Ok(out)
    }

    fn exec_try(
        &mut self,
        block: &[Stmt],
        handler: &Option<(Option<Pattern>, Vec<Stmt>)>,
        finalizer: &Option<Vec<Stmt>>,
        env: &Env,
    ) -> Completion {
        // HasCallInTailPosition: the try block is never a tail position (its calls must run
        // inside the protection of the catch/finally); a catch block is one only when there is
        // no finalizer.
        let saved_tco = self.tco_ok;
        self.tco_ok = false;
        let result = self.exec_block(block, env);
        self.tco_ok = saved_tco;
        let after_catch = match result {
            Err(Abrupt::Throw(ex)) => {
                self.settle_caught(&ex);
                if let Some((param, body)) = handler {
                    crate::jit::perf_error_caught();
                    // The catch parameter lives in its own environment (flagged so a sloppy `eval`'s
                    // var-hoisting walk skips it); the body's lexicals + statements run in a child
                    // block environment.
                    let body_env = if let Some(pat) = param {
                        let catch_env = new_catch_scope(*env);
                        self.bind_pattern(pat, ex, &catch_env, BindMode::Lexical(false))?;
                        new_scope(Some(catch_env))
                    } else {
                        new_scope(Some(*env))
                    };
                    let mut v = Value::Empty;
                    let mut last = Ok(Value::Empty);
                    self.declare_block_lexicals(body, &body_env, true);
                    let catch_tco = self.tco_ok && finalizer.is_none();
                    let saved_tco = std::mem::replace(&mut self.tco_ok, catch_tco);
                    for s in body {
                        match self.exec_stmt(s, &body_env) {
                            Ok(sv) => {
                                if !matches!((sv).view(), ValueView::Empty) {
                                    v = sv;
                                }
                                last = Ok(v);
                            }
                            Err(e) => {
                                last = Err(crate::interpreter::update_abrupt_empty(e, v));
                                break;
                            }
                        }
                    }
                    self.tco_ok = saved_tco;
                    last
                } else {
                    Err(Abrupt::Throw(ex))
                }
            }
            other => other,
        };
        // HTML §8.1.4.5: aborting a running script empties the execution-context stack without
        // triggering normal language mechanisms such as `finally`.
        if matches!(after_catch, Err(Abrupt::Interrupt(_))) {
            return after_catch;
        }
        if let Some(fin) = finalizer {
            // An abrupt completion in `finally` overrides the try/catch completion; its normal
            // value is discarded (the try/catch completion stands). A pending tail call from the
            // try/catch is parked during the finalizer (which may schedule its own) and dropped
            // if the finalizer overrides the completion.
            let parked = self.pending_tail.take();
            // A finalizer IS a tail-position context (its `return f()` overrides everything
            // after it), so tco_ok is inherited here.
            match self.exec_block(fin, env) {
                Ok(_) => {
                    if parked.is_some() {
                        self.pending_tail = parked;
                    }
                }
                Err(e) => {
                    // The finalizer overrides the try/catch completion: the parked tail call
                    // is dropped, but one the finalizer itself scheduled stays pending. The
                    // TryStatement's UpdateEmpty(F, undefined) still applies.
                    return Err(crate::interpreter::update_abrupt_empty(e, Value::Undefined));
                }
            }
        }
        // TryStatement completion: UpdateEmpty(result, undefined) — an EMPTY value (normal or in
        // an escaping break/continue) becomes undefined.
        match after_catch {
            Ok(v) => Ok(crate::interpreter::update_empty(v)),
            Err(e) => Err(crate::interpreter::update_abrupt_empty(e, Value::Undefined)),
        }
    }

    fn exec_switch(&mut self, disc: &Expr, cases: &[SwitchCase], env: &Env) -> Completion {
        let d = self.eval(disc, env)?;
        let scope = new_scope(Some(*env));
        for case in cases {
            for s in &case.body {
                self.declare_block_lexicals(std::slice::from_ref(s), &scope, true);
            }
        }
        let mut matched = None;
        for (i, case) in cases.iter().enumerate() {
            if let Some(test) = &case.test {
                let t = self.eval(test, &scope)?;
                if self.strict_equals(&d, &t) {
                    matched = Some(i);
                    break;
                }
            }
        }
        let start = match matched.or_else(|| cases.iter().position(|c| c.test.is_none())) {
            Some(i) => i,
            None => return Ok(Value::Undefined),
        };
        let mut v = Value::Empty;
        for case in &cases[start..] {
            for s in &case.body {
                match self.exec_stmt(s, &scope) {
                    Ok(sv) => {
                        if !matches!((sv).view(), ValueView::Empty) {
                            v = sv;
                        }
                    }
                    Err(Abrupt::Break(None, bv)) => {
                        if !matches!((bv).view(), ValueView::Empty) {
                            v = bv;
                        }
                        return Ok(crate::interpreter::update_empty(v));
                    }
                    // Abrupt exits thread the accumulated V per UpdateEmpty.
                    Err(e) => return Err(crate::interpreter::update_abrupt_empty(e, v)),
                }
            }
        }
        // SwitchStatement completion is UpdateEmpty(V, undefined).
        Ok(crate::interpreter::update_empty(v))
    }

    // ----- variable binding -------------------------------------------------------------------

    fn init_lexical(&mut self, name: &str, value: Value, is_const: bool, env: &Env) {
        env.borrow_mut()
            .vars
            .insert(name, Binding::data(value, !is_const, true));
    }

    /// Object-environment HasBinding for a `with (obj)` scope: HasProperty, then an object-valued
    /// `obj[@@unscopables]` whose `name` property is truthy blocks the binding.
    fn with_has_binding(&mut self, obj: &Value, name: &str) -> Result<bool, Abrupt> {
        if !self.js_has_property(obj, name)? {
            return Ok(false);
        }
        if let Some(k) = crate::builtins::well_known_key(self, "unscopables") {
            let uns = self.get_member_key(obj, &k)?;
            if matches!((uns).view(), ValueView::Obj(_)) {
                let blocked = self.get_member_key(&uns, name)?;
                if self.to_boolean(&blocked) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    pub(crate) fn get_var(&mut self, name: &str, env: &Env) -> Result<Value, Abrupt> {
        self.get_var_with(name, env).map(|(v, _)| v)
    }

    /// ECMA-262 typeof: only an unresolvable Reference produces "undefined"
    /// without GetValue. Resolution/getter/TDZ exceptions propagate unchanged.
    /// Decide at the lookup's missing-name branch, not by throwing an Error and
    /// then guessing whether to swallow it (which also allocates on feature checks).
    pub(crate) fn typeof_name_vm(&mut self, name: &str, env: &Env) -> Result<Value, Abrupt> {
        let (value, _) = self.get_var_with_mode::<true>(name, env)?;
        if self.is_htmldda(&value) {
            Ok(Value::lstr("undefined"))
        } else {
            Ok(Value::lstr(value.type_of()))
        }
    }

    /// Resolve like [`Interp::get_var`], also yielding the `with` object the binding came from —
    /// the implicit call receiver for `f()` resolved through a with scope.
    pub(crate) fn get_var_with(
        &mut self,
        name: &str,
        env: &Env,
    ) -> Result<(Value, Option<Value>), Abrupt> {
        self.get_var_with_mode::<false>(name, env)
    }

    fn get_var_with_mode<const ALLOW_UNRESOLVED: bool>(
        &mut self,
        name: &str,
        env: &Env,
    ) -> Result<(Value, Option<Value>), Abrupt> {
        // Walk the scope chain by raw pointer: every scope is transitively kept alive by `env`
        // (the caller holds it) via the `parent` links, and nothing mutates the chain during a
        // lookup. (With counted handles the walk saved a bump and drop per hop, the dominant
        // cost when a free/global name missed every intermediate scope.) `with`-object scopes
        // take the owned slow path.
        type ScopeCell = std::cell::RefCell<crate::interpreter::Scope>;
        let mut cur: *const ScopeCell = Env::as_ptr(env);
        // Every binding's name is an atom of the heap: one lookup serves the whole chain, each
        // record then compares words, and a name with no atom is bound in no record (until
        // author code runs: a `with` object's checks look again).
        let mut atom = crate::atom::lookup(name);
        loop {
            let b = unsafe { (*cur).borrow() };
            if let Some(binding) = atom.and_then(|atom| b.vars.get_atom(atom)) {
                if !binding.initialized() {
                    return Err(self.throw(
                        "ReferenceError",
                        format!("cannot access '{name}' before initialization"),
                    ));
                }
                // A live module import reads through to the exporter's binding.
                if let Some((src_env, local)) = binding.import() {
                    drop(b);
                    return Ok((self.get_var(&local.text(), &src_env)?, None));
                }
                return Ok((binding.value(), None));
            }
            // `with (obj)`: resolve against the object's properties if it has the name (the proxy
            // `has` trap and @@unscopables participate here). Rare — fall to the owned path.
            if let Some(obj) = b.with_object().filter(|value| value.as_obj().is_some()) {
                drop(b);
                if self.with_has_binding(&obj, name)? {
                    // GetBindingValue re-checks HasProperty (the property may have vanished while
                    // @@unscopables ran): strict code throws, sloppy reads undefined.
                    if !self.js_has_property(&obj, name)? {
                        if self.strict {
                            return Err(
                                self.throw("ReferenceError", format!("{name} is not defined"))
                            );
                        }
                        return Ok((Value::Undefined, Some(obj)));
                    }
                    return Ok((self.get_member_key(&obj, name)?, Some(obj)));
                }
                // `with_has_binding` said no — continue up the chain. Re-borrow to read `parent`.
                // Its traps and getters may have declared the name since.
                atom = atom.or_else(|| crate::atom::lookup(name));
                let b = unsafe { (*cur).borrow() };
                match b.parent.as_ref() {
                    Some(p) => cur = Env::as_ptr(p),
                    None => break,
                }
                continue;
            }
            match b.parent.as_ref() {
                Some(p) => cur = Env::as_ptr(p),
                None => break,
            }
        }
        // Fast path: an own data property of an ordinary global (the overwhelmingly common
        // resolution for builtins and script-level bindings) — one hash lookup, no trap walk.
        if self.ordinary_get_ptr(Gc::as_ptr(&self.global) as usize) {
            let b = self.global.borrow();
            if matches!(b.exotic, crate::value::Exotic::None) {
                if let Some(p) = b
                    .props
                    .get_walking(name, &mut atom.map(crate::atom::PropKey::atom))
                {
                    if !p.accessor() {
                        let v = p.value();
                        drop(b);
                        return Ok((v, None));
                    }
                }
            }
        }
        // Fall back to a property of the global object (where builtins live). The lookup is the
        // full trap-aware [[HasProperty]]: the global's prototype may be (or contain) a proxy.
        let g = Value::Obj(self.global);
        if self.js_has_property(&g, name)? {
            return Ok((self.get_member_key(&g, name)?, None));
        }
        if ALLOW_UNRESOLVED {
            Ok((Value::Undefined, None))
        } else {
            Err(self.throw("ReferenceError", format!("{name} is not defined")))
        }
    }

    /// Walk the scope chain for an initialized binding without the `with`/global fallback or the
    /// not-defined throw. Used for internal `%...%` markers that only ever live in scopes.
    /// Whether `env` is inside an ordinary (non-arrow) function, which is what supplies `new.target`.
    /// Non-arrow functions bind `this` in their scope; arrows do not (they inherit it), so an arrow
    /// at the top level is *not* function code for `new.target` purposes. Governs `new.target`
    /// validity in a direct eval.
    fn in_function_code(&self, env: &Env) -> bool {
        let mut cur = Some(*env);
        while let Some(s) = cur {
            if Env::ptr_eq(&s, &self.global_env) {
                return false;
            }
            let (has_this, parent) = {
                let b = s.borrow();
                (b.vars.contains_key("this"), b.parent)
            };
            if has_this {
                return true;
            }
            cur = parent;
        }
        false
    }

    fn peek_binding(&self, name: &str, env: &Env) -> Option<Value> {
        let mut cur = Some(*env);
        while let Some(s) = cur {
            let b = s.borrow();
            if let Some(binding) = b.vars.get(name) {
                return Some(binding.value());
            }
            cur = b.parent;
        }
        None
    }

    /// Resolve a source-level private name (`#x`) to this class evaluation's runtime key through
    /// the scope chain (each class evaluation binds its private names in its class scope — the
    /// spec's PrivateEnvironment). An unresolved name keeps its literal spelling.
    /// [`Self::resolve_private`] without copying the bound key string.
    fn resolve_private_key(&self, name: &str, env: &Env) -> crate::lstr::LStr {
        let mut cur = Some(*env);
        while let Some(s) = cur {
            let b = s.borrow();
            if let Some(binding) = b.vars.get(name) {
                if let ValueView::Str(k) = &(binding.value()).view() {
                    return *k;
                }
            }
            cur = b.parent;
        }
        crate::lstr::LStr::from(name)
    }

    fn resolve_private(&self, name: &str, env: &Env) -> String {
        let mut cur = Some(*env);
        while let Some(s) = cur {
            let b = s.borrow();
            if let Some(binding) = b.vars.get(name) {
                if let ValueView::Str(k) = &(binding.value()).view() {
                    return k.to_string();
                }
            }
            cur = b.parent;
        }
        name.to_string()
    }

    /// Validate a dynamic import's already-evaluated options value, returning the `with.type`
    /// attribute. Non-object options/attributes and non-string attribute values are TypeErrors
    /// (which the caller turns into a rejected promise).
    fn import_attributes(&mut self, o: &Value) -> Result<Option<String>, Abrupt> {
        if matches!(&(o).view(), ValueView::Undefined) {
            return Ok(None);
        }
        if !matches!(&(o).view(), ValueView::Obj(_)) {
            return Err(self.throw("TypeError", "import options must be an object"));
        }
        let with = self.get_member_key(o, "with")?;
        if matches!((with).view(), ValueView::Undefined) {
            return Ok(None);
        }
        if !matches!((with).view(), ValueView::Obj(_)) {
            return Err(self.throw("TypeError", "import attributes must be an object"));
        }
        // EnumerableOwnProperties(with, key+value): own string keys in order (through a proxy's
        // ownKeys/getOwnPropertyDescriptor traps), re-checking each descriptor before its Get;
        // every attribute value must be a string.
        let mut ty = None;
        if let Some((t, h)) = crate::builtins::proxy_pair(self, &with) {
            let keys = crate::builtins::proxy_own_keys(self, &t, &h).map_err(Abrupt::Throw)?;
            for k in keys {
                let pk = self.to_property_key(&k)?;
                if Interp::is_sym_key(&pk) {
                    continue;
                }
                let enumerable = crate::builtins::proxy_get_own_property(self, &t, &h, &pk)
                    .map_err(Abrupt::Throw)?
                    .is_some_and(|desc| desc.enumerable());
                if !enumerable {
                    continue;
                }
                let v = self.get_member_key(&with, &pk)?;
                let ValueView::Str(sv) = (v).view() else {
                    return Err(self.throw("TypeError", "import attribute values must be strings"));
                };
                if pk == "type" {
                    ty = Some(sv.to_string());
                }
            }
            return Ok(ty);
        }
        let object = with.as_obj().unwrap();
        let keys =
            crate::builtins::ordinary_own_keys_ordered(self, &object).map_err(Abrupt::Throw)?;
        for k in keys {
            if Self::is_sym_key(&k) {
                continue;
            }
            let enumerable = self
                .host_platform_enumerable(&object, &k)?
                .unwrap_or_else(|| {
                    object
                        .borrow()
                        .props
                        .get(&k)
                        .is_some_and(|property| property.enumerable())
                });
            if !enumerable {
                continue;
            }
            let v = self.get_member_key(&with, &k)?;
            let ValueView::Str(sv) = (v).view() else {
                return Err(self.throw("TypeError", "import attribute values must be strings"));
            };
            if &*k == "type" {
                ty = Some(sv.to_string());
            }
        }
        Ok(ty)
    }

    /// Annex B.3.3 sync step: copy the block-scope binding of `name` (or a freshly-made function
    /// for a bare `if (x) function f(){}` position) into the nearest variable environment's
    /// binding — or the global object's property for global code.
    pub(crate) fn annexb_fn_sync_eval(&mut self, name: &str, func: &Rc<Function>, env: &Env) {
        // The value: the block's own binding of the name (instantiated at block entry). A bare
        // `if (x) function f(){}` position has no block scope — per B.3.4 it acts as an implicit
        // block, so a fresh function is made here. Only the *immediate* scope counts: an ancestor
        // binding (a catch parameter, an outer var) is not the block binding.
        let own = {
            let b = env.borrow();
            if b.var_boundary() {
                None
            } else {
                b.vars.get(name).map(|x| x.value())
            }
        };
        let val = own.unwrap_or_else(|| {
            // B.3.4: the bare if-position declaration acts as `{ function f(){} }` — the function
            // closes over an implicit block scope binding its own name, so `f = ...` inside the
            // body rebinds the block binding, not the promoted var.
            let block = crate::interpreter::new_scope(Some(*env));
            let f = self.make_function(func.clone(), block);
            bind(&block, name, f);
            f
        });
        let mut cur = Some(*env);
        while let Some(s) = cur {
            if s.borrow().var_boundary() {
                if let Some(mut b) = s.borrow_mut().vars.get_mut(name) {
                    b.set_value(val);
                    return;
                }
                // The binding can be missing here only if a deletable (eval-created) one was
                // deleted, or for a function's `arguments`, which FunctionDeclarationInstantiation
                // does not create (no arguments object, or one in a separate parameter scope).
                // SetMutableBinding then creates it, deletable, as B.3.3.3 re-creates the former.
                if !Env::ptr_eq(&s, &self.global_env) {
                    s.borrow_mut()
                        .vars
                        .insert(name, Binding::data(val, true, true).deletable());
                    return;
                }
                break;
            }
            let parent = s.borrow().parent;
            cur = parent;
        }
        // Global code: the var binding lives as a global-object property (re-created when a
        // deletable eval-made one was deleted).
        let mut g = self.global.borrow_mut();
        if g.props
            .get(name)
            .is_some_and(|p| p.writable() && !p.accessor())
        {
            g.props.set_value(name, val);
        } else if g.props.get(name).is_none() && g.extensible {
            g.props.insert(name, Property::data(val, true, true, true));
        }
    }

    /// PrivateGet: read a private field/method after a brand check. An object that was not
    /// constructed with this private name in scope (`#x` absent) is a TypeError, not `undefined`.
    pub(crate) fn get_private_member(&mut self, base: &Value, name: &str) -> Completion {
        let prop = match &(base).view() {
            ValueView::Obj(o) => o.borrow().internal_slots().get(name).cloned(),
            _ => None,
        };
        match prop {
            Some(p) if p.accessor() => match p.getter() {
                Some(g) => self.call(*g, *base, &[]),
                None => {
                    let shown = private_display(name);
                    Err(self.throw(
                        "TypeError",
                        format!("private accessor {shown} has no getter"),
                    ))
                }
            },
            Some(p) => Ok(p.into_value()),
            None => {
                let shown = private_display(name);
                Err(self.throw(
                    "TypeError",
                    format!("cannot read private member {shown} from an object whose class did not declare it"),
                ))
            }
        }
    }

    /// PrivateSet: write a private field (or invoke a private setter) after a brand check. A
    /// private *method* (non-writable) or a getter-only private accessor is a TypeError — never a
    /// silent sloppy-mode no-op.
    pub(crate) fn set_private_member(
        &mut self,
        base: &Value,
        name: &str,
        value: Value,
    ) -> Result<(), Abrupt> {
        let shown = private_display(name).to_string();
        let (o, prop) = match &(base).view() {
            ValueView::Obj(o) => (*o, o.borrow().internal_slots().get(name).cloned()),
            _ => {
                return Err(self.throw(
                    "TypeError",
                    format!("cannot write private member {shown} to an object whose class did not declare it"),
                ))
            }
        };
        match prop {
            Some(p) if p.accessor() => {
                let Some(setter) = p.setter().cloned() else {
                    return Err(self.throw(
                        "TypeError",
                        format!("private accessor {shown} has no setter"),
                    ));
                };
                self.call(setter, *base, &[value])?;
                Ok(())
            }
            Some(p) => {
                if !p.writable() {
                    return Err(self.throw(
                        "TypeError",
                        format!("private method {shown} is not writable"),
                    ));
                }
                if let Some(p) = o.borrow_mut().internal_slots_mut().get_mut(name) {
                    p.set_value(value);
                }
                Ok(())
            }
            None => Err(self.throw(
                "TypeError",
                format!("cannot write private member {shown} to an object whose class did not declare it"),
            )),
        }
    }

    /// The object `super.x` reads/writes through: `GetPrototypeOf([[HomeObject]])` when a home
    /// object is in scope (object-literal & class methods bound via `%homeobject%`), else the
    /// statically-resolved `%superproto%` carried by older class-method environments.
    fn super_base(&mut self, env: &Env) -> Result<Value, Abrupt> {
        if let Some(ValueView::Obj(home)) =
            (self.peek_binding("%homeobject%", env)).map(|__value| (__value).view())
        {
            return Ok(match home.borrow().proto {
                Some(p) => Value::Obj(p),
                None => Value::Null,
            });
        }
        // No home object in scope (e.g. `super.x` reached through an `eval` in a non-method): a
        // super property reference here is an early SyntaxError.
        match self.peek_binding("%superproto%", env) {
            Some(v) => Ok(v),
            None => Err(self.throw("SyntaxError", "'super' keyword unexpected here")),
        }
    }

    pub(crate) fn assign_var(&mut self, name: &str, value: Value, env: &Env) -> Result<(), Abrupt> {
        let mut cur = Some(*env);
        // As in `get_var_with_mode`: the name's atom once for the whole chain, looked up again
        // after a `with` object's checks ran author code.
        let mut atom = crate::atom::lookup(name);
        while let Some(s) = cur {
            let (with_obj, parent) = {
                let mut b = s.borrow_mut();
                if let Some(mut binding) = atom.and_then(|atom| b.vars.get_atom_mut(atom)) {
                    // Var initializers and assignment-pattern/loop-head writes perform
                    // PutValue, not InitializeBinding. Hoisted var bindings are already
                    // initialized; a lexical binding found here must retain its TDZ.
                    if !binding.initialized() {
                        return Err(self.throw(
                            "ReferenceError",
                            format!("cannot access '{name}' before initialization"),
                        ));
                    }
                    if !binding.mutable() {
                        // A const always throws; a non-strict immutable binding (a named function
                        // expression's own name) is a silent no-op in sloppy code.
                        if binding.strict_immutable() || self.strict {
                            return Err(
                                self.throw("TypeError", format!("assignment to constant '{name}'"))
                            );
                        }
                        return Ok(());
                    }
                    binding.initialize(value);
                    return Ok(());
                }
                (b.with_object(), b.parent)
            };
            if let Some(obj) = with_obj.as_ref().filter(|value| value.as_obj().is_some()) {
                if self.with_has_binding(obj, name)? {
                    // SetMutableBinding re-checks HasProperty; strict code throws if the
                    // property vanished while @@unscopables ran.
                    if !self.js_has_property(obj, name)? && self.strict {
                        return Err(self.throw("ReferenceError", format!("{name} is not defined")));
                    }
                    return self.set_member_key(obj, name, value);
                }
                atom = atom.or_else(|| crate::atom::lookup(name));
            }
            cur = parent;
        }
        // A declared global `var`/`function` lives as a property of the global object (the check
        // is the trap-aware [[HasProperty]] — the global's proto chain may contain a proxy).
        if self.js_has_property(&Value::Obj(self.global), name)? {
            return self.set_member_key(&Value::Obj(self.global), name, value);
        }
        // Truly undeclared: strict → ReferenceError; sloppy → create a global property.
        if self.strict {
            return Err(self.throw("ReferenceError", format!("{name} is not defined")));
        }
        let g = Value::Obj(self.global);
        self.set_member_key(&g, name, value)
    }

    /// CreateListFromArrayLike: read `obj.length` then elements `0..length` into a Vec (used by the
    /// Proxy `ownKeys` trap, which returns an array-like, not an iterable).
    /// CreateListFromArrayLike, rooted while getters run and returned rooted (I1).
    pub(crate) fn create_list_from_arraylike(
        &mut self,
        obj: &Value,
    ) -> Result<crate::rooting::RootedVec, Abrupt> {
        let len_v = self.get_member_key(obj, "length")?;
        let len = self.to_number(&len_v)?;
        let len = if len.is_nan() || len < 0.0 {
            0
        } else {
            len as usize
        };
        // A trap's array-like `length` is author data: bound the list before reading it.
        if len > crate::interpreter::MAX_ARRAY_OP_LEN {
            return Err(self.throw("RangeError", "array length exceeds engine limit"));
        }
        let mut out = crate::rooting::RootedVec::from_vec(Vec::with_capacity(len.min(1024)));
        for k in 0..len {
            out.push(self.get_member_key(obj, &k.to_string())?);
        }
        Ok(out)
    }

    /// The `in` operator (ECMA-262 §13.10.1, RelationalExpression : RelationalExpression `in`
    /// ShiftExpression, steps 5–6): a *TypeError* unless `object` is an Object, then
    /// HasProperty(object, ? ToPropertyKey(key)). An array-index Number that names an element of
    /// an ordinary object's own element storage is answered without building its string key.
    /// Reflect.has shares it once its target is an Object (§28.1.9).
    pub(crate) fn in_operator(&mut self, key: &Value, object: &Value) -> Result<bool, Abrupt> {
        let ValueView::Obj(target) = &(object).view() else {
            return Err(self.throw("TypeError", "'in' requires an object on the right"));
        };
        // ToPropertyKey of a String or a Symbol is itself, and of an integral Number its decimal
        // text: none runs code, so such a key's text is tested along an ordinary chain, or given
        // to the general algorithm, without building a key string. `key` keeps a Symbol alive.
        match key.view() {
            ValueView::Num(n) => {
                if n >= 0.0
                    && n < u32::MAX as f64
                    && n.fract() == 0.0
                    && Self::plain_own_element_present(target, n as u32)
                {
                    return Ok(true);
                }
                if let Some(text) = crate::value::KeyText::integer(n) {
                    if let Some(present) = Self::ordinary_chain_has_property(target, text.as_str())
                    {
                        return Ok(present);
                    }
                    return self.js_has_property(object, text.as_str());
                }
            }
            ValueView::Str(name) => {
                let text = crate::value::encode_lstr_key(&name);
                if let Some(present) = Self::ordinary_chain_has_property(target, &text) {
                    return Ok(present);
                }
                return self.js_has_property(object, &text);
            }
            ValueView::Sym(symbol) => {
                let text = crate::value::KeyText::symbol(symbol.id);
                if let Some(present) = Self::ordinary_chain_has_property(target, text.as_str()) {
                    return Ok(present);
                }
                return self.js_has_property(object, text.as_str());
            }
            _ => {}
        }
        let key = self.to_property_key(key)?;
        self.js_has_property(object, &key)
    }

    /// HasProperty(O, P) for a String key along a prototype chain of ordinary objects and
    /// Arrays (ECMA-262 §10.1.7.1 OrdinaryHasProperty: [[GetOwnProperty]], then the
    /// prototype's [[HasProperty]]), which runs no author code: exactly `js_has_property`'s
    /// own-property test at every hop, without copying the key. `None` as soon as an object on
    /// the chain is in an interpreter side table (`ic_plain` clear: proxies, typed arrays,
    /// namespaces, Web IDL indexed objects) or is any other exotic object.
    pub(crate) fn ordinary_chain_has_property(start: &Gc, key: &str) -> Option<bool> {
        // The key's atom, once a map on the chain has looked it up (`Props::get_walking`).
        let mut atom = None;
        let mut current = *start;
        loop {
            let next = {
                let object = current.borrow();
                if !object.ic_plain.get()
                    || !(matches!(
                        object.exotic,
                        crate::value::Exotic::None | crate::value::Exotic::Array
                    ) || object.exotic.is_collection())
                {
                    return None;
                }
                if object.props.get_walking(key, &mut atom).is_some() {
                    return Some(true);
                }
                object.proto
            };
            match next {
                Some(prototype) => current = prototype,
                None => return Some(false),
            }
        }
    }

    /// OrdinaryGet (ECMA-262 §10.1.8.1) of a String key along a prototype chain of ordinary
    /// objects and Arrays, as far as it runs no author code: `Some(value)` of the first data
    /// property found, `Some(undefined)` when no object on the chain has the key, `None` for an
    /// accessor or when an object on the chain has other internal methods (see
    /// [`Interp::ordinary_chain_has_property`]); the caller then takes the general [[Get]].
    pub(crate) fn ordinary_chain_get_data(start: &Gc, key: &str) -> Option<Value> {
        let mut current = *start;
        loop {
            let next = {
                let object = current.borrow();
                if !object.ic_plain.get()
                    || !(matches!(
                        object.exotic,
                        crate::value::Exotic::None | crate::value::Exotic::Array
                    ) || object.exotic.is_collection())
                {
                    return None;
                }
                if let Some(property) = object.props.get(key) {
                    return (!property.accessor()).then(|| property.value());
                }
                object.proto
            };
            match next {
                Some(prototype) => current = prototype,
                None => return Some(Value::Undefined),
            }
        }
    }

    /// `object[index] = value` for an existing own writable data property of an ordinary
    /// object or Array whose key is `index`'s decimal text: OrdinarySetWithOwnDescriptor then
    /// only replaces [[Value]] (for an Array the index is below `length`). Gives the value
    /// back for every other case. Reached after `fast_set_elem` declined, so the property is
    /// not a dense element (a sparse index, typically).
    pub(crate) fn ordinary_set_existing_index(
        object: &Gc,
        index: f64,
        value: Value,
    ) -> Result<(), Value> {
        let Some(text) = crate::value::KeyText::integer(index) else {
            return Err(value);
        };
        let mut object = object.borrow_mut();
        if !object.ic_plain.get()
            || !(matches!(
                object.exotic,
                crate::value::Exotic::None | crate::value::Exotic::Array
            ) || object.exotic.is_collection())
        {
            return Err(value);
        }
        match object.props.get(text.as_str()) {
            Some(property) if !property.accessor() && property.writable() => {
                object.props.set_value(text.as_str(), value);
                Ok(())
            }
            _ => Err(value),
        }
    }

    /// `[[HasProperty]]`: the trap-aware `in` check. Handles a proxy anywhere on the chain (its `has`
    /// trap, or forwarding to the target's own `[[HasProperty]]`), TypedArray index slots, then the
    /// ordinary own-property + prototype walk.
    pub(crate) fn js_has_property(&mut self, obj: &Value, key: &str) -> Result<bool, Abrupt> {
        let o = match &(obj).view() {
            ValueView::Obj(o) => *o,
            _ => return Ok(false),
        };
        // A deferred namespace evaluates on [[HasProperty]] with a string key.
        self.defer_trigger(&o, Some(key))?;
        let ptr = Gc::as_ptr(&o) as usize;
        if let Some((target, handler)) = self.proxy_at(ptr) {
            if matches!((handler).view(), ValueView::Null) {
                return Err(self.throw("TypeError", "cannot perform 'has' on a revoked proxy"));
            }
            let Some(trap) =
                self.proxy_get_trap(&handler, crate::interpreter::PROXY_TRAP_HAS, "has")?
            else {
                return self.proxy_forward(|me| me.js_has_property(&target, key));
            };
            // The trap receives the original property key — a symbol stays a symbol.
            let key_val = self.key_value(key);
            let res = self.proxy_call_trap(
                crate::interpreter::PROXY_TRAP_HAS,
                &trap,
                handler,
                [target, key_val],
            )?;
            let present = self.to_boolean(&res);
            if !present {
                // ECMA-262 §10.5.7 step 9: target.[[GetOwnProperty]](P), then IsExtensible(target)
                // only for a configurable target property; both may run a Proxy target's traps.
                if let ValueView::Obj(t) = &(target).view() {
                    let target_desc = crate::builtins::object_get_own_property(self, t, key)
                        .map_err(Abrupt::Throw)?;
                    if let Some(target_desc) = target_desc {
                        if !target_desc.configurable() {
                            return Err(self.throw(
                                "TypeError",
                                "proxy 'has' trap hid a non-configurable property",
                            ));
                        }
                        if !crate::builtins::js_is_extensible(self, &target)
                            .map_err(Abrupt::Throw)?
                        {
                            return Err(self.throw(
                                "TypeError",
                                "proxy 'has' trap hid a property of a non-extensible target",
                            ));
                        }
                    }
                }
            }
            return Ok(present);
        }
        if let Some(info) = self.typed_arrays.get(&ptr).copied() {
            match self.ta_index_kind(&info, key) {
                TaIndex::Element(_) => return Ok(true),
                TaIndex::Exotic => return Ok(false),
                TaIndex::Ordinary => {}
            }
        }
        // HasProperty calls the Web IDL legacy object's [[GetOwnProperty]], so observing a
        // supported index also runs the platform getter and propagates an abrupt completion.
        if self.host_indexed_own_value(&o, key)?.is_some() {
            return Ok(true);
        }
        // A String wrapper's `length` and in-range indices are own exotic properties.
        if let crate::value::Exotic::StrWrap(s) = &o.borrow().exotic {
            if key == "length" {
                return Ok(true);
            }
            if let Some(idx) = crate::value::canonical_index(key) {
                if (idx as usize) < s.unit_len() {
                    return Ok(true);
                }
            }
        }
        if o.borrow().props.contains(key) {
            return Ok(true);
        }
        let proto = o.borrow().proto;
        match proto {
            Some(p) => self.js_has_property(&Value::Obj(p), key),
            None => Ok(false),
        }
    }

    // ----- expressions ------------------------------------------------------------------------

    /// ECMA-262 GetThisEnvironment / ResolveThisBinding (§§9.4.3–4). Object Environment
    /// Records (including `with`) never supply the keyword's this binding; ordinary name
    /// resolution must not consult their author-controlled property named "this".
    pub(crate) fn resolve_this_binding(&mut self, env: &Env) -> Result<Value, Abrupt> {
        let mut current = Some(*env);
        while let Some(scope) = current {
            let parent = {
                let record = scope.borrow();
                if let Some(binding) = record.vars.get("this") {
                    if binding.initialized() && !binding.is_import() {
                        return Ok(binding.value());
                    }
                    drop(record);
                    // Use the selected record, not the original chain, for its TDZ/import
                    // error path. A nearer with object cannot hide an uninitialized binding.
                    return self.get_var("this", &scope);
                }
                record.parent
            };
            current = parent;
        }
        Ok(Value::Undefined)
    }

    pub(crate) fn eval(&mut self, expr: &Expr, env: &Env) -> Result<Value, Abrupt> {
        // Querying the native stack on every evaluation would cost the tree-walker several
        // percent; every eighth (each level of nesting is at least one evaluation) suffices
        // for the gateway's red zone.
        self.ast_tick = self.ast_tick.wrapping_add(1);
        if self.ast_tick & 7 == 0 && crate::interpreter::ast_stack_low() {
            return self.eval_on_fresh_stack(expr, env);
        }
        match expr {
            Expr::Num(n) => Ok(Value::Num(*n)),
            Expr::BigInt(n) => Ok(Value::BigInt(n.value())),
            Expr::Str(s) => Ok(Value::Str(s.clone().into())),
            Expr::ToStr(inner) => {
                let v = self.eval(inner, env)?;
                Ok(Value::Str(self.to_string(&v)?))
            }
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Null => Ok(Value::Null),
            Expr::Undefined => Ok(Value::Undefined),
            Expr::Ident(name, pos) => self
                .get_var_with(name, env)
                .map(|(value, _)| value)
                .map_err(|error| self.placed_error(error, *pos)),
            Expr::This => self.resolve_this_binding(env),
            Expr::Regex { body, flags } => self.make_regexp(body, flags),
            Expr::Array(elems) => self.eval_array(elems, env),
            Expr::Object(props) => self.eval_object(props, env),
            Expr::Func(func) => Ok(self.make_function(func.clone(), *env)),
            Expr::Class(class) => self.eval_class(class, env),
            Expr::Yield { delegate, arg } => {
                let value = match arg {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Undefined,
                };
                if !crate::coroutine::in_coroutine() {
                    return Err(self.throw("SyntaxError", "yield outside a generator"));
                }
                if *delegate {
                    if crate::coroutine::in_async_gen() {
                        self.yield_delegate_async(&value)
                    } else {
                        self.yield_delegate(&value)
                    }
                } else {
                    self.yield_one(value)
                }
            }
            Expr::Await(e) => {
                let v = self.eval(e, env)?;
                if crate::coroutine::in_coroutine() {
                    match crate::coroutine::coroutine_await(self, v) {
                        crate::coroutine::Resume::Next(settled) => Ok(settled),
                        crate::coroutine::Resume::Throw(err) => Err(Abrupt::Throw(err)),
                        crate::coroutine::Resume::Return(rv) => Err(Abrupt::Return(rv)),
                        crate::coroutine::Resume::Terminate => {
                            Err(Abrupt::Interrupt(crate::InterruptReason::Cancelled))
                        }
                    }
                } else {
                    self.await_value(v)
                }
            }
            Expr::Super => Err(self.throw("SyntaxError", "'super' keyword unexpected here")),
            Expr::Seq(items) => {
                let mut last = Value::Undefined;
                for e in items {
                    last = self.eval(e, env)?;
                }
                Ok(last)
            }
            Expr::Cond { test, cons, alt } => {
                let t = self.eval(test, env)?;
                if self.to_boolean(&t) {
                    self.eval(cons, env)
                } else {
                    self.eval(alt, env)
                }
            }
            Expr::Binary { left, .. } | Expr::Logical { left, .. }
                if matches!(**left, Expr::Binary { .. } | Expr::Logical { .. }) =>
            {
                self.eval_operator_chain(expr, env)
            }
            Expr::Logical { op, left, right } => {
                let l = self.eval(left, env)?;
                self.eval_logical_right(op, l, right, env)
            }
            Expr::Unary { op, arg } => self.eval_unary(op, arg, env),
            Expr::Update { op, prefix, arg } => self.eval_update(op, *prefix, arg, env),
            Expr::Binary { op, left, right } => {
                let l = self.eval(left, env)?;
                let r = self.eval(right, env)?;
                self.binary(op, l, r)
            }
            Expr::Paren(inner) => self.eval(inner, env),
            Expr::Assign {
                op,
                target,
                value,
                pos,
            } => self.eval_assign(op, target, value, *pos, env),
            Expr::ImportMeta => Ok(self.import_meta_vm(env)),
            Expr::NewTarget => Ok(self.new_target_vm(env)),
            Expr::ImportCall {
                spec,
                phase,
                options,
            } => {
                let specifier = self.eval(spec, env)?;
                // The options argument evaluates synchronously (abrupt completions propagate,
                // before any promise exists); its validation happens inside the promise.
                let opts_val = match options {
                    Some(o) => Some(self.eval(o, env)?),
                    None => None,
                };
                self.import_call_vm(specifier, opts_val, *phase, env)
            }
            Expr::PrivateIn { name, obj } => {
                let o = self.eval(obj, env)?;
                self.private_in_vm(name, o, env, None)
            }
            Expr::OptionalChain(inner) => {
                let saved = self.short_circuit;
                self.short_circuit = false;
                let v = self.eval(inner, env);
                let short = self.short_circuit;
                self.short_circuit = saved;
                if short {
                    Ok(Value::Undefined)
                } else {
                    v
                }
            }
            Expr::Member { obj, .. } | Expr::Index { obj, .. }
                if matches!(&**obj, Expr::Member { obj, .. } | Expr::Index { obj, .. }
                    if !matches!(**obj, Expr::Super)) =>
            {
                self.eval_member_chain(expr, env)
            }
            Expr::Member {
                obj,
                prop,
                optional,
                pos,
            } => self.eval_member(obj, prop, *optional, *pos, env),
            Expr::Index {
                obj,
                index,
                optional,
                pos,
            } => self.eval_index(obj, index, *optional, *pos, env),
            Expr::Call { .. }
                if expr
                    .call_chain_link()
                    .and_then(Expr::call_chain_link)
                    .is_some() =>
            {
                self.eval_call_chain(expr, env)
            }
            Expr::Call {
                callee,
                args,
                optional,
                pos,
            } => self.eval_call(callee, args, *optional, *pos, env),
            Expr::TaggedTemplate {
                tag,
                site,
                quasis,
                subs,
                pos,
            } => self.eval_tagged_template(tag, *site, quasis, subs, *pos, env),
            Expr::New { callee, args, pos } => self.eval_new(callee, args, *pos, env),
        }
    }

    // `eval` evaluates each node that has a source position through one of these functions,
    // which give that position to errors thrown while the node is evaluated (see
    // [`Interp::placed_error`]). They map errors by value (`map_err`, or the error branch of a
    // test the evaluation already makes) instead of inspecting a finished completion by
    // reference, which kept every completion in a temporary and copied it on this recursive
    // path, stalling store-to-load forwarding. An explicit call's own errors need no placing:
    // the frame was given the call's position when it called.

    /// `new callee(args)` whose `new` is at source `position`.
    fn eval_new(
        &mut self,
        callee: &Expr,
        args: &[ArrayElem],
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        let c = self
            .eval(callee, env)
            .map_err(|error| self.placed_error(error, position))?;
        let argv = self
            .eval_args(args, env)
            .map_err(|error| self.placed_error(error, position))?;
        // EvaluateNew step 7: IsConstructor follows ArgumentListEvaluation.
        if !self.value_is_constructor(&c) {
            let error = self.throw(
                "TypeError",
                crate::callee_name::not_callable_message(callee, true),
            );
            return Err(self.placed_error(error, position));
        }
        self.construct_at(position, c, &argv)
    }

    /// A property read `obj.prop` (or `obj.#prop`) whose name is at source `position`.
    fn eval_member(
        &mut self,
        obj: &Expr,
        prop: &str,
        optional: bool,
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        if matches!(obj, Expr::Super) {
            return self
                .read_super_member(prop, env)
                .map_err(|error| self.placed_error(error, position));
        }
        let base = self
            .eval(obj, env)
            .map_err(|error| self.placed_error(error, position))?;
        if self.short_circuit {
            return Ok(Value::Undefined); // an earlier `?.` link short-circuited
        }
        if optional && matches!((base).view(), ValueView::Undefined | ValueView::Null) {
            self.short_circuit = true;
            return Ok(Value::Undefined);
        }
        if prop.starts_with('#') {
            let k = self.resolve_private(prop, env);
            return self
                .get_private_member(&base, &k)
                .map_err(|error| self.placed_error(error, position));
        }
        self.get_member_key(&base, prop)
            .map_err(|error| self.placed_error(error, position))
    }

    /// A chain of property reads (`a.b.c`, `a[0]?.b`), evaluated from its base outwards in a
    /// loop rather than recursing once per link: generated code chains hundreds of thousands.
    /// Each link performs the steps of [`Self::eval_member`] / [`Self::eval_index`] on the value
    /// below it, and an error is placed at the link that raised it and then at each enclosing
    /// link, as the recursion would.
    fn eval_member_chain(&mut self, expr: &Expr, env: &Env) -> Result<Value, Abrupt> {
        let mut links = Vec::new();
        let mut node = expr;
        loop {
            match node {
                Expr::Member { obj, .. } | Expr::Index { obj, .. }
                    if !matches!(**obj, Expr::Super) =>
                {
                    links.push(node);
                    node = obj;
                }
                _ => break,
            }
        }
        let mut value = match self.eval(node, env) {
            Ok(value) => value,
            Err(error) => return Err(self.placed_chain_error(error, &links)),
        };
        for index in (0..links.len()).rev() {
            value = match self.member_link(links[index], value, env) {
                Ok(value) => value,
                Err(error) => return Err(self.placed_chain_error(error, &links[..=index])),
            };
        }
        Ok(value)
    }

    /// One link of [`Self::eval_member_chain`]: the read of `link` from the already evaluated
    /// `base`.
    fn member_link(&mut self, link: &Expr, base: Value, env: &Env) -> Result<Value, Abrupt> {
        if self.short_circuit {
            return Ok(Value::Undefined); // an earlier `?.` link short-circuited
        }
        let optional = matches!(
            link,
            Expr::Member { optional: true, .. } | Expr::Index { optional: true, .. }
        );
        if optional && matches!((base).view(), ValueView::Undefined | ValueView::Null) {
            self.short_circuit = true;
            return Ok(Value::Undefined);
        }
        match link {
            Expr::Member { prop, .. } if prop.starts_with('#') => {
                let k = self.resolve_private(prop, env);
                self.get_private_member(&base, &k)
            }
            Expr::Member { prop, .. } => self.get_member_key(&base, prop),
            Expr::Index { index, .. } => {
                let idx = self.eval(index, env)?;
                // GetValue: ToObject(base) throws before ToPropertyKey coerces the key.
                if matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    trace_nullish_property("eval-index", &idx);
                    return Err(
                        self.throw("TypeError", "cannot read property of null or undefined")
                    );
                }
                if let (ValueView::Obj(o), ValueView::Num(n)) = (&(base).view(), &(idx).view()) {
                    if let Some(v) = self.fast_get_elem(o, *n) {
                        return Ok(v);
                    }
                }
                let key = self.to_property_key(&idx)?;
                self.get_member_key(&base, &key)
            }
            _ => unreachable!("member chain links are property reads"),
        }
    }

    /// A chain of calls (`f()()()`, `f().m().n()`; see [`Expr::call_chain_link`]), evaluated
    /// from its innermost call outwards in a loop rather than recursing once per link. Each link
    /// performs the steps of [`Self::eval_call`] and [`Self::eval_callee_reference`] on the value
    /// of the call below it.
    fn eval_call_chain(&mut self, expr: &Expr, env: &Env) -> Result<Value, Abrupt> {
        let mut links = Vec::new();
        let mut node = expr;
        while let Some(below) = node.call_chain_link() {
            links.push(node);
            node = below;
        }
        let mut value = match self.eval(node, env) {
            Ok(value) => value,
            Err(error) => return Err(self.placed_chain_error(error, &links)),
        };
        for index in (0..links.len()).rev() {
            value = match self.call_link(links[index], value, env) {
                Ok(value) => value,
                Err(error) => return Err(self.placed_chain_error(error, &links[..=index])),
            };
        }
        Ok(value)
    }

    /// One link of [`Self::eval_call_chain`]: the call `link` whose callee is, or is a method
    /// of, the value `below`.
    fn call_link(&mut self, link: &Expr, below: Value, env: &Env) -> Result<Value, Abrupt> {
        let Expr::Call {
            callee, args, pos, ..
        } = link
        else {
            unreachable!("call chain links are calls");
        };
        if self.short_circuit {
            return Ok(Value::Undefined);
        }
        let (func, this) = match &**callee {
            Expr::Member { prop, .. } if prop.starts_with('#') => {
                let k = self.resolve_private(prop, env);
                (self.get_private_member(&below, &k)?, below)
            }
            Expr::Member { prop, .. } => (self.get_member_key(&below, prop)?, below),
            Expr::Index { index, .. } => {
                let idx = self.eval(index, env)?;
                let key = self.ref_prop_key(&below, &mut RefKey::Raw(idx))?;
                (self.get_member_key(&below, &key)?, below)
            }
            _ => (below, Value::Undefined),
        };
        let argv = self.eval_args(args, env)?;
        if !func.is_callable() {
            return Err(self.throw(
                "TypeError",
                crate::callee_name::not_callable_message(callee, false),
            ));
        }
        self.call_at(*pos, func, this, &argv)
    }

    /// `error`, raised inside the innermost of `links` (outermost first), placed at each of
    /// them from the inside out.
    #[cold]
    #[inline(never)]
    fn placed_chain_error(&self, error: Abrupt, links: &[&Expr]) -> Abrupt {
        for link in links.iter().rev() {
            if let Some(position) = link.position() {
                self.place_thrown(&error, position);
            }
        }
        error
    }

    /// `super.prop`.
    fn read_super_member(&mut self, prop: &str, env: &Env) -> Result<Value, Abrupt> {
        // GetThisBinding first (TDZ ReferenceError), then Get(base, key, actualThis):
        // a getter on the super prototype sees the current `this`.
        let this = self.get_var("this", env)?;
        let home = self.super_base(env)?;
        if matches!((home).view(), ValueView::Undefined | ValueView::Null) {
            return Err(self.throw(
                "TypeError",
                format!("cannot read property '{prop}' of a null super base"),
            ));
        }
        crate::builtins::reflect_ordinary_get(self, &home, prop, &this).map_err(Abrupt::Throw)
    }

    /// A computed property read `obj[index]` whose `[` is at source `position`.
    fn eval_index(
        &mut self,
        obj: &Expr,
        index: &Expr,
        optional: bool,
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        if matches!(obj, Expr::Super) {
            return self
                .read_super_index(index, env)
                .map_err(|error| self.placed_error(error, position));
        }
        let base = self
            .eval(obj, env)
            .map_err(|error| self.placed_error(error, position))?;
        if self.short_circuit {
            return Ok(Value::Undefined);
        }
        if optional && matches!((base).view(), ValueView::Undefined | ValueView::Null) {
            self.short_circuit = true;
            return Ok(Value::Undefined);
        }
        let idx = self
            .eval(index, env)
            .map_err(|error| self.placed_error(error, position))?;
        // GetValue: ToObject(base) throws before ToPropertyKey coerces the key.
        if matches!((base).view(), ValueView::Undefined | ValueView::Null) {
            trace_nullish_property("eval-index", &idx);
            let error = self.throw("TypeError", "cannot read property of null or undefined");
            return Err(self.placed_error(error, position));
        }
        if let (ValueView::Obj(o), ValueView::Num(n)) = (&(base).view(), &(idx).view()) {
            if let Some(v) = self.fast_get_elem(o, *n) {
                return Ok(v);
            }
        }
        let key = self
            .to_property_key(&idx)
            .map_err(|error| self.placed_error(error, position))?;
        self.get_member_key(&base, &key)
            .map_err(|error| self.placed_error(error, position))
    }

    /// `super[index]`.
    fn read_super_index(&mut self, index: &Expr, env: &Env) -> Result<Value, Abrupt> {
        // GetThisBinding, the key expression, GetSuperBase, then ToPropertyKey.
        let this = self.get_var("this", env)?;
        let idx = self.eval(index, env)?;
        let home = self.super_base(env)?;
        let key = self.to_property_key(&idx)?;
        if matches!((home).view(), ValueView::Undefined | ValueView::Null) {
            return Err(self.throw(
                "TypeError",
                format!(
                    "cannot read property '{}' of a null super base",
                    crate::value::decode_string_key(&key)
                ),
            ));
        }
        crate::builtins::reflect_ordinary_get(self, &home, &key, &this).map_err(Abrupt::Throw)
    }

    /// `error`, thrown while the node at source `position` was evaluated, once given that
    /// position (see [`Interp::place_thrown`]).
    #[cold]
    #[inline(never)]
    fn placed_error(&self, error: Abrupt, position: u32) -> Abrupt {
        self.place_thrown(&error, position);
        error
    }

    /// Call `func` as the explicit call at source `position`, which the callee's frame records
    /// as its caller's location (see `crate::stack_trace`).
    pub(crate) fn call_at(
        &mut self,
        position: u32,
        func: Value,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        self.call_site = position;
        let result = self.call(func, this, args);
        self.call_site = crate::stack_trace::NO_POSITION;
        result
    }

    /// [`Interp::call_at`] for `new`.
    pub(crate) fn construct_at(
        &mut self,
        position: u32,
        callee: Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        self.call_site = position;
        let result = self.construct(callee, args);
        self.call_site = crate::stack_trace::NO_POSITION;
        result
    }

    /// Meta-property reads used by both the tree-walker and heap bytecode continuations. The
    /// importing module is carried lexically so an async continuation keeps its original
    /// referrer after suspension.
    pub(crate) fn import_meta_vm(&self, env: &Env) -> Value {
        self.peek_binding("%importmeta%", env)
            .or(self.import_meta)
            .unwrap_or(Value::Undefined)
    }

    pub(crate) fn new_target_vm(&self, env: &Env) -> Value {
        self.peek_binding("%newtarget%", env)
            .unwrap_or_else(|| self.constructor_new_target())
    }

    /// EvaluateImportCall after the specifier and optional options expressions have already run.
    /// Those expressions can suspend in a VM coroutine, but every operation from
    /// NewPromiseCapability onward is shared with the ordinary evaluator. In particular,
    /// ToString/options validation failures reject the returned promise rather than throwing
    /// synchronously.
    pub(crate) fn import_call_vm(
        &mut self,
        specifier: Value,
        options: Option<Value>,
        phase: ImportPhase,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        let specifier = match self.to_string(&specifier) {
            Ok(specifier) => specifier,
            Err(error) => {
                let promise = self.new_promise();
                let reason = crate::interpreter::abrupt_value(error);
                self.reject_promise(&promise, reason);
                return Ok(promise);
            }
        };
        match phase {
            // ContinueDynamicImport at the source phase resolves with [[ModuleSource]] or
            // rejects when that field is empty. The Test262 host exposes one concrete source
            // module; ordinary Source Text Module Records have no source object.
            ImportPhase::Source => {
                let promise = self.new_promise();
                if &*specifier.text() == "<module source>" {
                    let source = self.module_source_of(&specifier.text());
                    self.resolve_promise(&promise, source);
                } else {
                    let reason = crate::interpreter::abrupt_value(
                        self.throw("SyntaxError", "source phase import is not available"),
                    );
                    self.reject_promise(&promise, reason);
                }
                Ok(promise)
            }
            // `import.defer(x)` defers evaluation of the module; for specifier handling it
            // behaves like a plain dynamic import.
            ImportPhase::Evaluation | ImportPhase::Defer => {
                let mut attr_type = None;
                if let Some(options) = &options {
                    match self.import_attributes(options) {
                        Ok(value) => attr_type = value,
                        Err(error) => {
                            let promise = self.new_promise();
                            let reason = crate::interpreter::abrupt_value(error);
                            self.reject_promise(&promise, reason);
                            return Ok(promise);
                        }
                    }
                }
                let referrer = self.peek_binding("%importmeta%", env);
                Ok(self.dynamic_import(
                    &specifier.text(),
                    attr_type.as_deref(),
                    matches!(phase, ImportPhase::Defer),
                    referrer,
                ))
            }
        }
    }

    /// ResolvePrivateIdentifier from `env`, through the operation site's cache when it has one.
    #[inline]
    fn private_key_vm(
        &self,
        name: &str,
        env: &Env,
        site: Option<&crate::bytecode::PrivateSite>,
    ) -> crate::lstr::LStr {
        match site {
            Some(site) => site.key(env, || self.resolve_private_key(name, env)),
            None => self.resolve_private_key(name, env),
        }
    }

    pub(crate) fn private_in_vm(
        &mut self,
        name: &str,
        value: Value,
        env: &Env,
        site: Option<&crate::bytecode::PrivateSite>,
    ) -> Result<Value, Abrupt> {
        let key = self.private_key_vm(name, env, site);
        match (value).view() {
            // Private fields, methods and accessors are own internal state, never inherited.
            ValueView::Obj(object) => {
                let object = object.borrow();
                Ok(Value::Bool(match site {
                    Some(site) => site.find(object.internal_slots(), &key.text()).is_some(),
                    None => object.internal_slots().contains(&key.text()),
                }))
            }
            _ => Err(self.throw("TypeError", "the right-hand side of 'in' must be an object")),
        }
    }

    pub(crate) fn private_get_vm(
        &mut self,
        name: &str,
        value: &Value,
        env: &Env,
        site: Option<&crate::bytecode::PrivateSite>,
    ) -> Result<Value, Abrupt> {
        let key = self.private_key_vm(name, env, site);
        // PrivateGet on a private field or method is the element's value; an accessor (and a
        // missing element's TypeError) takes the general path, after this borrow ends.
        if let (Some(site), ValueView::Obj(object)) = (site, &(value).view()) {
            let object = object.borrow();
            if let Some(property) = site
                .find(object.internal_slots(), &key.text())
                .and_then(|slot| object.internal_slots().property_at(slot))
            {
                if !property.accessor() {
                    return Ok(property.value());
                }
            }
        }
        self.get_private_member(value, &key.text())
    }

    pub(crate) fn private_set_vm(
        &mut self,
        name: &str,
        base: &Value,
        value: Value,
        env: &Env,
        site: Option<&crate::bytecode::PrivateSite>,
    ) -> Result<(), Abrupt> {
        let key = self.private_key_vm(name, env, site);
        // PrivateSet on a private field writes it in place; a method (not writable), an accessor
        // and a missing element take the general path with its TypeErrors and setter call.
        if let (Some(site), ValueView::Obj(object)) = (site, &(base).view()) {
            let mut object = object.borrow_mut();
            if let Some(slot) = site.find(object.internal_slots(), &key.text()) {
                if let Some((_, property)) = object.internal_slots_mut().entry_at_mut(slot) {
                    if !property.accessor() && property.writable() {
                        property.set_value(value);
                        return Ok(());
                    }
                }
            }
        }
        self.set_private_member(base, &key.text(), value)
    }

    pub(crate) fn super_base_vm(&mut self, env: &Env) -> Result<Value, Abrupt> {
        self.super_base(env)
    }

    pub(crate) fn super_get_vm(
        &mut self,
        base: &Value,
        receiver: Value,
        key: Value,
    ) -> Result<Value, Abrupt> {
        if matches!(&(base).view(), ValueView::Null | ValueView::Undefined) {
            return Err(self.throw("TypeError", "cannot read property of null super base"));
        }
        let key = self.to_property_key(&key)?;
        self.get_member_recv(base, &key, receiver)
    }

    pub(crate) fn super_set_vm(
        &mut self,
        base: &Value,
        receiver: Value,
        key: Value,
        value: Value,
    ) -> Result<(), Abrupt> {
        if matches!(&(base).view(), ValueView::Null | ValueView::Undefined) {
            return Err(self.throw("TypeError", "cannot set property on null super base"));
        }
        let key = self.to_property_key(&key)?;
        let strict = self.strict;
        let success = self.set_member_recv(base, &key, value, receiver)?;
        self.finish_assignment_set(&key, strict, success)
    }

    fn eval_array(&mut self, elems: &[ArrayElem], env: &Env) -> Result<Value, Abrupt> {
        // Holes leave the index absent (a real elision), not `undefined`.
        // Elements are created as own data properties (CreateDataProperty), so accessors on
        // Array.prototype are never consulted.
        let arr = self.make_array(Vec::new());
        for e in elems {
            match e {
                ArrayElem::Item(e) => {
                    let v = self.eval(e, env)?;
                    self.append_array_literal(&arr, Some(v))?;
                }
                ArrayElem::Hole => self.append_array_literal(&arr, None)?,
                ArrayElem::Spread(e) => {
                    let v = self.eval(e, env)?;
                    let prefix = self.array_length(&arr.as_obj().expect("array literal"));
                    self.expand_spread(&v, prefix, |i, value| {
                        i.append_array_literal(&arr, Some(value))
                    })?;
                }
            }
        }
        Ok(arr)
    }

    /// The NamedEvaluation name for a property key: a symbol key names the function "[desc]"
    /// (or "" without a description); a string key is used as-is.
    pub(crate) fn fn_name_for_key(&self, key: &str) -> String {
        if Interp::is_sym_key(key) {
            match (self.sym_from_key(key)).map(|__value| (__value).view()) {
                Some(ValueView::Sym(d)) => match &d.description {
                    Some(desc) => format!("[{desc}]"),
                    None => String::new(),
                },
                _ => String::new(),
            }
        } else {
            crate::value::decode_string_key(key).to_string()
        }
    }

    /// NamedEvaluation: give an anonymous function/class the binding/property name it's assigned to,
    /// unless it already has a non-empty name.
    pub(crate) fn set_fn_name(&mut self, v: &Value, name: &str) {
        if !v.is_callable() {
            return;
        }
        if let ValueView::Obj(o) = &(v).view() {
            let empty = o
                .borrow()
                .props
                .get("name")
                .map(|p| matches!((p.value()).view(), ValueView::Str(s) if s.is_empty()))
                .unwrap_or(true);
            if empty {
                o.borrow_mut().props.insert(
                    "name".to_string(),
                    Property::data(Value::lstr(name.to_string()), false, false, true),
                );
            }
        }
    }

    fn eval_object(&mut self, props: &[PropDef], env: &Env) -> Result<Value, Abrupt> {
        let obj = self.new_object();
        // Methods/getters/setters carry a [[HomeObject]] (the literal itself) so `super.x` resolves
        // against the object's *current* prototype, evaluated dynamically at access time. Only
        // methods that can observe it close over that record (see `Function::observes_home_object`);
        // the rest close over `env` directly and so do not form a cycle with the literal.
        let mut home_env: Option<Env> = None;
        let mut method_env = |func: &Rc<Function>| -> Env {
            if !func.observes_home_object() {
                return *env;
            }
            *home_env.get_or_insert_with(|| {
                let home = new_scope(Some(*env));
                bind(&home, "%homeobject%", Value::Obj(obj));
                home
            })
        };
        for prop in props {
            match prop {
                // A Cover reaching evaluation means the parse accepted it as part of a pattern
                // that was then used as an expression — unreachable (the parser rejects it), but
                // evaluate like a plain key/value for robustness.
                PropDef::KeyValue { key, value } | PropDef::Cover { key, value } => {
                    let k = self.eval_prop_key(key, env)?;
                    let v = self.eval(value, env)?;
                    if is_anonymous_fn(value) {
                        let name = self.fn_name_for_key(&k);
                        self.set_fn_name(&v, &name);
                    }
                    obj.borrow_mut()
                        .props
                        .insert(k.as_str(), Property::plain(v));
                }
                PropDef::Method { key, func } => {
                    let k = self.eval_prop_key(key, env)?;
                    let f = self.make_function(func.clone(), method_env(func));
                    let name = self.fn_name_for_key(&k);
                    self.set_fn_name(&f, &name);
                    obj.borrow_mut()
                        .props
                        .insert(k.as_str(), Property::plain(f));
                }
                PropDef::Getter { key, func } => {
                    let k = self.eval_prop_key(key, env)?;
                    let f = self.make_function(func.clone(), method_env(func));
                    let name = self.fn_name_for_key(&k);
                    self.set_fn_name(&f, &format!("get {name}"));
                    self.define_accessor(&obj, &k, Some(f), None);
                }
                PropDef::Setter { key, func } => {
                    let k = self.eval_prop_key(key, env)?;
                    let f = self.make_function(func.clone(), method_env(func));
                    let name = self.fn_name_for_key(&k);
                    self.set_fn_name(&f, &format!("set {name}"));
                    self.define_accessor(&obj, &k, None, Some(f));
                }
                PropDef::Spread(e) => {
                    // CopyDataProperties (no excluded names) — proxy traps, symbols and string
                    // indices included.
                    let v = self.eval(e, env)?;
                    self.copy_data_properties_into(&obj, &v, &[])?;
                }
                // `__proto__: value` sets the prototype when value is an Object or Null; any other
                // value type is ignored (no property is created).
                PropDef::Proto(e) => {
                    let v = self.eval(e, env)?;
                    match (v).view() {
                        ValueView::Obj(p) => obj.borrow_mut().proto = Some(p),
                        ValueView::Null => obj.borrow_mut().proto = None,
                        _ => {}
                    }
                }
            }
        }
        Ok(Value::Obj(obj))
    }

    pub(crate) fn define_accessor(
        &self,
        obj: &Gc,
        key: &str,
        get: Option<Value>,
        set: Option<Value>,
    ) {
        let mut b = obj.borrow_mut();
        if b.props.merge_accessor(key, get, set) {
            return;
        }
        b.props
            .insert(key, Property::accessor_prop(get, set, true, true));
    }

    fn eval_prop_key(
        &mut self,
        key: &PropKey,
        env: &Env,
    ) -> Result<crate::value::PropertyKey, Abrupt> {
        match key {
            PropKey::Ident(s) => Ok(crate::value::PropertyKey::string(s.clone())),
            PropKey::Str(s) => Ok(crate::value::PropertyKey::string(s.to_string())),
            PropKey::Num(n) => Ok(crate::value::PropertyKey::string(self.num_to_str(*n))),
            PropKey::Computed(e) => {
                let v = self.eval(e, env)?;
                self.to_property_key(&v)
            }
        }
    }

    /// ArgumentListEvaluation. The list is rooted while later arguments evaluate (I1), and the
    /// caller keeps it rooted through the call.
    fn eval_args(
        &mut self,
        args: &[ArrayElem],
        env: &Env,
    ) -> Result<crate::rooting::RootedVec, Abrupt> {
        let mut out = crate::rooting::RootedVec::new();
        for a in args {
            match a {
                ArrayElem::Item(e) => {
                    let value = self.eval(e, env)?;
                    self.append_argument(&mut out, value)?;
                }
                ArrayElem::Spread(e) => {
                    let v = self.eval(e, env)?;
                    self.append_spread_arguments(&v, &mut out)?;
                }
                ArrayElem::Hole => self.append_argument(&mut out, Value::Undefined)?,
            }
        }
        Ok(out)
    }

    /// Freeze an object in place (non-extensible, all own props non-writable/non-configurable).
    pub(crate) fn freeze_object(&self, v: &Value) {
        if let ValueView::Obj(o) = &(v).view() {
            let mut b = o.borrow_mut();
            b.extensible = false;
            b.props.freeze_descriptors();
        }
    }

    fn eval_tagged_template(
        &mut self,
        tag: &Expr,
        site_id: u64,
        quasis: &[(Option<String>, String)],
        subs: &[Expr],
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        let (func, this) = self
            .eval_callee_reference(tag, env)
            .map_err(|error| self.placed_error(error, position))?;
        // TaggedTemplate delegates to EvaluateCall: GetValue precedes ArgumentListEvaluation,
        // but IsCallable follows every substitution (§§13.3.11.1, 13.3.6.2).
        let strings = self
            .template_object(site_id, quasis)
            .map_err(|error| self.placed_error(error, position))?;
        let mut argv = crate::rooting::RootedVec::from_vec(vec![strings]);
        for s in subs {
            let value = self
                .eval(s, env)
                .map_err(|error| self.placed_error(error, position))?;
            argv.push(value);
        }
        if !func.is_callable() {
            let error = self.throw(
                "TypeError",
                crate::callee_name::not_callable_message(tag, false),
            );
            return Err(self.placed_error(error, position));
        }
        self.call_at(position, func, this, &argv)
    }

    /// GetTemplateObject: one frozen strings array (with a frozen `.raw`) per template *site* —
    /// re-evaluating the same site passes the identical object.
    pub(crate) fn template_object(
        &mut self,
        site_id: u64,
        quasis: &[(Option<String>, String)],
    ) -> Result<Value, Abrupt> {
        let site = (Gc::as_ptr(&self.global) as usize, site_id);
        let strings = match self.template_cache.get(&site) {
            Some(v) => *v,
            None => {
                let cooked: Vec<Value> = quasis
                    .iter()
                    .map(|(c, _)| {
                        c.as_ref()
                            .map(|s| Value::lstr(s.clone()))
                            .unwrap_or(Value::Undefined)
                    })
                    .collect();
                let raw: Vec<Value> = quasis.iter().map(|(_, r)| Value::lstr(r.clone())).collect();
                let strings = self.make_array(cooked);
                let raw_arr = self.make_array(raw);
                self.freeze_object(&raw_arr);
                if let ValueView::Obj(so) = &(strings).view() {
                    // `raw` is a non-enumerable data property (then frozen with the rest).
                    so.borrow_mut().props.insert(
                        "raw",
                        crate::value::Property::data(raw_arr, false, false, false),
                    );
                }
                self.freeze_object(&strings);
                if let ValueView::Obj(o) = &(strings).view() {
                    self.gc_pin(o);
                }
                self.template_cache.insert(site, strings);
                strings
            }
        };
        Ok(strings)
    }

    /// Evaluate a `return` operand, treating the expression's tail positions as proper tail
    /// calls. Anything else evaluates normally to a value.
    fn eval_return_expr(&mut self, e: &Expr, env: &Env) -> Result<TailEval, Abrupt> {
        match e {
            Expr::Call {
                callee,
                args,
                optional: false,
                pos,
            } if matches!(&**callee, Expr::Ident(name, _) if name == "eval") => {
                // Resolve once: even testing whether this is direct eval can invoke a
                // with-object getter. A shadowing ordinary function still gets a tail call.
                let result = self.eval_named_eval_call(args, *pos, env, true);
                if let Err(error) = &result {
                    self.place_thrown(error, *pos);
                }
                result
            }
            Expr::Call { pos, .. } | Expr::TaggedTemplate { pos, .. } => {
                let tail = if matches!(e, Expr::Call { .. }) {
                    self.eval_tail_call(e, env)
                } else {
                    self.eval_tail_tagged(e, env)
                };
                match tail {
                    Ok(Some((f, t, a))) => Ok(TailEval::Tail(f, t, a, *pos)),
                    Ok(None) => Ok(TailEval::Val(self.eval(e, env)?)),
                    Err(error) => {
                        self.place_thrown(&error, *pos);
                        Err(error)
                    }
                }
            }
            Expr::Cond { test, cons, alt } => {
                let tv = self.eval(test, env)?;
                if self.to_boolean(&tv) {
                    self.eval_return_expr(cons, env)
                } else {
                    self.eval_return_expr(alt, env)
                }
            }
            Expr::Seq(v) if !v.is_empty() => {
                for x in &v[..v.len() - 1] {
                    self.eval(x, env)?;
                }
                self.eval_return_expr(&v[v.len() - 1], env)
            }
            Expr::Logical { op, left, right } => {
                let lv = self.eval(left, env)?;
                let take_right = match *op {
                    "&&" => self.to_boolean(&lv),
                    "||" => !self.to_boolean(&lv),
                    "??" => matches!((lv).view(), ValueView::Undefined | ValueView::Null),
                    _ => false,
                };
                if take_right {
                    self.eval_return_expr(right, env)
                } else {
                    Ok(TailEval::Val(lv))
                }
            }
            _ => Ok(TailEval::Val(self.eval(e, env)?)),
        }
    }

    /// If `e` is a plain call usable as a proper tail call, evaluate its callee, `this` and
    /// arguments and return them for the trampoline; `None` falls back to a normal evaluation.
    fn eval_tail_call(
        &mut self,
        e: &Expr,
        env: &Env,
    ) -> Result<Option<(Value, Value, crate::rooting::RootedVec)>, Abrupt> {
        let Expr::Call {
            callee,
            args,
            optional: false,
            ..
        } = e
        else {
            return Ok(None);
        };
        if matches!(&**callee, Expr::Super)
            || matches!(&**callee, Expr::Ident(name, _) if name == "eval")
        {
            // Constructor continuation and direct eval retain their dedicated evaluation.
            return Ok(None);
        }
        let (func, this) = self.eval_callee_reference(callee, env)?;
        let argv = self.eval_args(args, env)?;
        if !func.is_callable() {
            return Err(self.throw(
                "TypeError",
                crate::callee_name::not_callable_message(callee, false),
            ));
        }
        Ok(Some((func, this, argv)))
    }

    /// A tagged template in tail position: evaluate the tag and assemble the argument list for
    /// the trampoline.
    fn eval_tail_tagged(
        &mut self,
        e: &Expr,
        env: &Env,
    ) -> Result<Option<(Value, Value, crate::rooting::RootedVec)>, Abrupt> {
        let Expr::TaggedTemplate {
            tag,
            site,
            quasis,
            subs,
            ..
        } = e
        else {
            return Ok(None);
        };
        let (func, this) = self.eval_callee_reference(tag, env)?;
        let strings = self.template_object(*site, quasis)?;
        let mut argv = crate::rooting::RootedVec::from_vec(vec![strings]);
        for s in subs {
            argv.push(self.eval(s, env)?);
        }
        if !func.is_callable() {
            return Err(self.throw(
                "TypeError",
                crate::callee_name::not_callable_message(tag, false),
            ));
        }
        Ok(Some((func, this, argv)))
    }

    fn super_call_context_available(&self, env: &Env) -> bool {
        // HasDirectSuper/MakeSuperPropertyReference inherit through arrows, but not through an
        // intervening ordinary function. Model GetThisEnvironment: the first Environment Record
        // with a `this` binding is the only one whose derived-constructor marker can authorize
        // SuperCall. A blind lexical lookup would incorrectly reach an outer constructor from a
        // nested ordinary function's direct eval.
        let mut cursor = Some(*env);
        while let Some(scope) = cursor {
            let binding = scope.borrow();
            if binding.vars.contains_key("this") {
                return matches!(
                    binding.vars.get("%supercallok%"),
                    Some(marker) if marker.initialized() && matches!(&(marker.value()).view(), ValueView::Bool(true))
                );
            }
            cursor = binding.parent;
        }
        false
    }

    /// SuperCall steps 1–3 (ECMA-262 §13.3.7.1): retain GetNewTarget and the live
    /// GetSuperConstructor result before ArgumentListEvaluation begins. Keeping these as Values
    /// on the VM operand stack preserves that ordering across an argument's `yield`/`await`.
    pub(crate) fn prepare_super_call(&mut self, env: &Env) -> Result<(Value, Value), Abrupt> {
        if !self.super_call_context_available(env) {
            return Err(self.throw("SyntaxError", "'super' keyword unexpected here"));
        }
        let new_target = self.new_target_vm(env);
        let active_function = self.get_var("%thisctor%", env)?;
        let super_constructor =
            crate::builtins::js_get_prototype_of(self, &active_function).map_err(Abrupt::Throw)?;
        Ok((new_target, super_constructor))
    }

    /// Complete SuperCall after ArgumentListEvaluation (ECMA-262 §13.3.7.1 steps 5–13):
    /// Construct with the retained new.target, bind the derived constructor's lexical `this`,
    /// then initialize that constructor's instance elements. The environment walk deliberately
    /// skips any intervening arrow activations and remains valid after the constructor returned.
    pub(crate) fn finish_super_call(
        &mut self,
        new_target: Value,
        super_constructor: Value,
        args: &[Value],
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        // IsConstructor follows ArgumentListEvaluation, even when the retained value is null or
        // otherwise obviously invalid.
        if !self.value_is_constructor(&super_constructor) {
            return Err(self.throw("TypeError", "super constructor is not a constructor"));
        }
        self.call_site = position;
        let result = self.construct_nt(super_constructor, args, new_target);
        self.call_site = crate::stack_trace::NO_POSITION;
        let result = result?;

        // GetThisEnvironment and BindThisValue occur after Construct. A second super() therefore
        // still constructs its parent before the already-initialized binding raises ReferenceError.
        let mut cursor = Some(*env);
        let mut this_env = None;
        while let Some(scope) = cursor {
            if scope.borrow().vars.contains_key("this") {
                this_env = Some(scope);
                break;
            }
            cursor = scope.borrow().parent;
        }
        let Some(this_env) = this_env else {
            return Err(self.throw("SyntaxError", "'super' keyword unexpected here"));
        };
        if this_env
            .borrow()
            .vars
            .get("this")
            .is_some_and(|binding| binding.initialized())
        {
            return Err(self.throw("ReferenceError", "super() may only be called once"));
        }
        if let Some(mut binding) = this_env.borrow_mut().vars.get_mut("this") {
            binding.initialize(result);
        }

        let active_function = self.get_var("%thisctor%", env)?;
        self.init_instance_fields(&active_function, &result)?;
        Ok(result)
    }

    fn eval_named_eval_call(
        &mut self,
        args: &[ArrayElem],
        position: u32,
        env: &Env,
        tail: bool,
    ) -> Result<TailEval, Abrupt> {
        // ECMA-262 §13.3.6: an identifier resolved through a with Environment Record
        // is still a non-property Reference. Its WithBaseObject is the receiver for
        // an ordinary call, not a condition for recognizing this Realm's %eval%.
        let (func, receiver) = self.get_var_with("eval", env)?;
        let direct = matches!(
            (&(func).view(), &self.eval_fn),
            (ValueView::Obj(function), Some(intrinsic)) if Gc::ptr_eq(function, intrinsic)
        );
        let argv = self.eval_args(args, env)?;
        if direct {
            self.call_site = position;
            let result = self.direct_eval(argv.first(), env);
            self.call_site = crate::stack_trace::NO_POSITION;
            return result.map(TailEval::Val);
        }
        if !func.is_callable() {
            return Err(self.throw("TypeError", "eval is not a function"));
        }
        let receiver = receiver.unwrap_or(Value::Undefined);
        if tail {
            Ok(TailEval::Tail(func, receiver, argv, position))
        } else {
            self.call_at(position, func, receiver, &argv)
                .map(TailEval::Val)
        }
    }

    fn eval_call(
        &mut self,
        callee: &Expr,
        args: &[ArrayElem],
        optional: bool,
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        if !optional && matches!(callee, Expr::Ident(name, _) if name == "eval") {
            return match self.eval_named_eval_call(args, position, env, false) {
                Ok(TailEval::Val(value)) => Ok(value),
                Ok(TailEval::Tail(..)) => unreachable!("non-tail eval call"),
                Err(error) => Err(self.placed_error(error, position)),
            };
        }
        // SuperCall deliberately stages GetNewTarget/GetSuperConstructor before arguments; the
        // bytecode VM uses the same two helpers with suspension between those phases.
        if matches!(callee, Expr::Super) {
            return self
                .eval_super_call(args, position, env)
                .map_err(|error| self.placed_error(error, position));
        }
        let (func, this) = self
            .eval_callee_reference(callee, env)
            .map_err(|error| self.placed_error(error, position))?;
        if self.short_circuit {
            return Ok(Value::Undefined);
        }
        // `f?.()` short-circuits the whole chain when the callee is nullish.
        if optional && matches!((func).view(), ValueView::Undefined | ValueView::Null) {
            self.short_circuit = true;
            return Ok(Value::Undefined);
        }
        let argv = self
            .eval_args(args, env)
            .map_err(|error| self.placed_error(error, position))?;
        if !func.is_callable() {
            let error = self.throw(
                "TypeError",
                crate::callee_name::not_callable_message(callee, false),
            );
            return Err(self.placed_error(error, position));
        }
        self.call_at(position, func, this, &argv)
    }

    /// `super(args)` at source `position`.
    fn eval_super_call(
        &mut self,
        args: &[ArrayElem],
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        let (new_target, super_constructor) = self.prepare_super_call(env)?;
        let argv = self.eval_args(args, env)?;
        self.finish_super_call(new_target, super_constructor, &argv, position, env)
    }

    /// Evaluate a callee once while retaining GetThisValue/WithBaseObject. Parentheses retain
    /// References; an OptionalChain boundary only contains its own short-circuit, not the
    /// enclosing call's argument evaluation. Shared by ordinary and tail calls and templates.
    fn eval_callee_reference(
        &mut self,
        callee: &Expr,
        env: &Env,
    ) -> Result<(Value, Value), Abrupt> {
        let reference = match callee {
            Expr::Paren(inner) => return self.eval_callee_reference(inner, env),
            Expr::OptionalChain(inner) => {
                let saved = std::mem::replace(&mut self.short_circuit, false);
                let result = self.eval_callee_reference(inner, env);
                let short = std::mem::replace(&mut self.short_circuit, saved);
                return result.map(|reference| {
                    if short {
                        (Value::Undefined, Value::Undefined)
                    } else {
                        reference
                    }
                });
            }
            Expr::Ident(name, _) => {
                let (f, recv) = self.get_var_with(name, env)?;
                (f, recv.unwrap_or(Value::Undefined))
            }
            Expr::Member {
                obj,
                prop,
                optional,
                ..
            } => {
                if matches!(**obj, Expr::Super) {
                    let receiver = self.get_var("this", env)?;
                    let proto = self.super_base(env)?;
                    let value = self.get_reference(&mut Reference::Super {
                        proto,
                        receiver,
                        key: RefKey::Static(prop.clone()),
                    })?;
                    return Ok((value, receiver));
                }
                let base = self.eval(obj, env)?;
                if self.short_circuit {
                    return Ok((Value::Undefined, Value::Undefined));
                }
                if *optional && matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    self.short_circuit = true;
                    return Ok((Value::Undefined, Value::Undefined));
                }
                let f = if prop.starts_with('#') {
                    let k = self.resolve_private(prop, env);
                    self.get_private_member(&base, &k)?
                } else {
                    self.get_member_key(&base, prop)?
                };
                (f, base)
            }
            Expr::Index {
                obj,
                index,
                optional,
                ..
            } => {
                if matches!(**obj, Expr::Super) {
                    let receiver = self.get_var("this", env)?;
                    let index = self.eval(index, env)?;
                    let proto = self.super_base(env)?;
                    let value = self.get_reference(&mut Reference::Super {
                        proto,
                        receiver,
                        key: RefKey::Raw(index),
                    })?;
                    return Ok((value, receiver));
                }
                let base = self.eval(obj, env)?;
                if self.short_circuit {
                    return Ok((Value::Undefined, Value::Undefined));
                }
                if *optional && matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    self.short_circuit = true;
                    return Ok((Value::Undefined, Value::Undefined));
                }
                let idx = self.eval(index, env)?;
                // GetValue checks ToObject(base) before coercing a computed method key,
                // just as it does for a non-call property read. Optional access already
                // short-circuited above; the key expression still evaluates for ordinary null.
                let key = self.ref_prop_key(&base, &mut RefKey::Raw(idx))?;
                let f = self.get_member_key(&base, &key)?;
                (f, base)
            }
            _ => {
                let f = self.eval(callee, env)?;
                if self.short_circuit {
                    return Ok((Value::Undefined, Value::Undefined));
                }
                (f, Value::Undefined)
            }
        };
        Ok(reference)
    }

    /// Legacy AST compatibility only. Real async/generator/module bodies use heap VM
    /// continuations even under Tier::Interp. A promise cannot be synchronously awaited inside
    /// an active Agent job: report the existing unavailable-VM failure rather than draining
    /// nested jobs or turning a pending completion into undefined.
    pub(crate) fn await_value(&mut self, v: Value) -> Result<Value, Abrupt> {
        let ptr = match &(v).view() {
            ValueView::Obj(o) if self.promises.contains_key(&(Gc::as_ptr(o) as usize)) => {
                Gc::as_ptr(o) as usize
            }
            ValueView::Obj(_) => {
                // Await of a plain object goes through the promise resolution procedure: a
                // thenable is adopted (its `then` called once), anything else settles as-is.
                let then = self.get_member_key(&v, "then")?;
                if then.is_callable() {
                    let p = self.new_promise();
                    let (res, rej) = self.make_resolver_pair(&p);
                    if let Err(Abrupt::Throw(e)) = self.call(then, v, &[res, rej]) {
                        self.reject_promise(&p, e);
                    }
                    return self.await_value(p);
                }
                return Ok(v);
            }
            _ => return Ok(v),
        };
        // Await → PromiseResolve(%Promise%, v): the `constructor` read on a native promise is
        // observable, and its abrupt completion is the await's.
        self.get_member_key(&v, "constructor")?;
        let _ = ptr;
        Err(self.throw("Error", crate::coroutine::VM_UNAVAILABLE_MSG))
    }

    // ----- promises ---------------------------------------------------------------------------

    pub(crate) fn promise_rejection(&self, promise: &Value) -> Option<Value> {
        let object = promise.as_obj()?;
        let state = self.promises.get(&(Gc::as_ptr(&object) as usize))?;
        (state.status == 2).then_some(state.value)
    }

    pub(crate) fn new_promise(&mut self) -> Value {
        let obj = Object::new(self.extra_protos.get("Promise").cloned());
        let p = Gc::as_ptr(&obj) as usize;
        self.gc_pin(&obj);
        self.promises
            .insert(&self.gc_heap, p, PromiseState::default());
        Value::Obj(obj)
    }

    /// One resolving function of a pair; both share `flag` — their [[AlreadyResolved]] record.
    pub(crate) fn make_resolver_with(
        &mut self,
        promise: &Value,
        fulfilling: bool,
        flag: &crate::value::Gc,
    ) -> Value {
        let target = self.make_native(
            if fulfilling { "resolve" } else { "reject" },
            1,
            if fulfilling {
                promise_resolve_native
            } else {
                promise_reject_native
            },
        );
        let bound = Object::new(Some(self.function_proto));
        bound.borrow_mut().call = Callable::bound(target, *promise, vec![Value::Obj(*flag)]);
        // A promise resolving function has `length` 1 and an empty `name`.
        bound.borrow_mut().props.insert(
            "length",
            crate::value::Property::data(Value::Num(1.0), false, false, true),
        );
        bound.borrow_mut().props.insert(
            "name",
            crate::value::Property::data(Value::lstr(""), false, false, true),
        );
        Value::Obj(bound)
    }

    /// The standard resolver pair (shared [[AlreadyResolved]]).
    pub(crate) fn make_resolver_pair(&mut self, promise: &Value) -> (Value, Value) {
        let flag = Object::new(None);
        (
            self.make_resolver_with(promise, true, &flag),
            self.make_resolver_with(promise, false, &flag),
        )
    }

    /// The resolver pair of an embedder's platform resolution: its [[AlreadyResolved]] record
    /// carries the private marker that [`Self::is_host_resolver`] recognizes.
    fn make_host_resolver_pair(&mut self, promise: &Value) -> (Value, Value) {
        let flag = Object::new(Some(self.host_resolution_marker));
        (
            self.make_resolver_with(promise, true, &flag),
            self.make_resolver_with(promise, false, &flag),
        )
    }

    /// Whether `callback` is a resolving function made by [`Self::make_host_resolver_pair`]:
    /// a bound resolver whose [[AlreadyResolved]] record has the private marker prototype.
    pub(crate) fn is_host_resolver(&self, callback: &Value) -> bool {
        let ValueView::Obj(function) = &(callback).view() else {
            return false;
        };
        let function = function.borrow();
        let Callable::Bound(bound) = &function.call else {
            return false;
        };
        matches!((bound.args.first()).map(|__value| (__value).view()), Some(ValueView::Obj(flag))
        if flag.borrow().proto.as_ref().is_some_and(|proto| {
            Gc::ptr_eq(proto, &self.host_resolution_marker)
        }))
    }

    pub(crate) fn resolve_promise(&mut self, promise: &Value, value: Value) {
        self.resolve_promise_from(promise, value, false);
    }

    /// Promise Resolve Functions steps 7-16 for `promise`. `host_steps` marks an embedder's
    /// platform resolution (see [`Interp::promise_resolved_with`]): adopting a thenable's state
    /// then belongs to no Realm or settings object.
    pub(crate) fn resolve_promise_from(&mut self, promise: &Value, value: Value, host_steps: bool) {
        // Follow a subclass graft's forwarding link (see run_constructor_on's native arm).
        let mut promise = *promise;
        if let ValueView::Obj(o) = &(promise).view() {
            if let Some(f) = self.promise_forward.get(&(Gc::as_ptr(o) as usize)) {
                promise = *f;
            }
        }
        let promise = &promise;
        let ptr = match &(promise).view() {
            ValueView::Obj(o) => Gc::as_ptr(o) as usize,
            _ => return,
        };
        if self.promises.get(&ptr).map(|s| s.status).unwrap_or(1) != 0 {
            return;
        }
        // Resolving a promise with ITSELF is a TypeError rejection (chaining cycle).
        if let (ValueView::Obj(p), ValueView::Obj(v)) = (&(promise).view(), &(value).view()) {
            if Gc::ptr_eq(p, v) {
                let e = crate::interpreter::abrupt_value(
                    self.throw("TypeError", "Chaining cycle detected for promise"),
                );
                self.reject_promise(promise, e);
                return;
            }
        }
        // Adopt a thenable's eventual state; a throwing `then` getter rejects. The `then` CALL
        // itself happens in a microtask (PromiseResolveThenableJob).
        if matches!((value).view(), ValueView::Obj(_)) {
            match self.get_member_key(&value, "then") {
                Ok(then) if then.is_callable() => {
                    // HostEnqueuePromiseJob(job, GetFunctionRealm(then)); embedder platform
                    // steps pass a null Realm instead (null job settings).
                    let (realm, host_context) = if host_steps {
                        (None, 0)
                    } else {
                        let settings = self.reaction_settings(self.host_job_context);
                        self.promise_job_target(&then, settings)
                    };
                    let (res, rej) = if host_steps {
                        self.make_host_resolver_pair(promise)
                    } else {
                        self.make_resolver_pair(promise)
                    };
                    let mut then = self.make_job_callback(then);
                    // The job's own callback is a built-in that calls `then`, so HostCallJob-
                    // Callback's script execution context for `then` sits below both.
                    let active_script = then.active_script.take();
                    let runner = crate::builtins::make_thenable_job(self, then, value, res, rej);
                    self.enqueue_promise_job(crate::interpreter::Job {
                        handler: crate::interpreter::JobCallback {
                            active_script,
                            ..crate::interpreter::JobCallback::plain(runner)
                        },
                        result: Value::Undefined,
                        value: Value::Undefined,
                        fulfilled: true,
                        realm,
                        host_context,
                    });
                    return;
                }
                Ok(_) => {}
                Err(Abrupt::Throw(e)) => {
                    self.reject_promise(promise, e);
                    return;
                }
                Err(_) => {}
            }
        }
        self.settle(promise, value, true);
    }

    pub(crate) fn reject_promise(&mut self, promise: &Value, reason: Value) {
        let mut promise = *promise;
        if let ValueView::Obj(o) = &(promise).view() {
            if let Some(f) = self.promise_forward.get(&(Gc::as_ptr(o) as usize)) {
                promise = *f;
            }
        }
        self.settle(&promise, reason, false);
    }

    fn settle(&mut self, promise: &Value, value: Value, fulfilled: bool) {
        let ptr = match &(promise).view() {
            ValueView::Obj(o) => Gc::as_ptr(o) as usize,
            _ => return,
        };
        let (reactions, handled) = match self.promises.get_mut(&self.gc_heap, &ptr) {
            Some(s) if s.status == 0 => {
                s.status = if fulfilled { 1 } else { 2 };
                s.value = value;
                (std::mem::take(&mut s.reactions), s.handled)
            }
            _ => return,
        };
        // RejectPromise: HostPromiseRejectionTracker(promise, "reject") unless
        // [[PromiseIsHandled]]. A later handler clears it (see `perform_promise_then`).
        // HTML's implementation appends it to the about-to-be-notified list of the running
        // script's settings object's global; the active host settings context is the one of the
        // realm whose code is running (a cross-realm call switches it with the realm).
        if !fulfilled && !handled {
            let order = self.next_unhandled_rejection;
            self.next_unhandled_rejection += 1;
            self.unhandled_rejections
                .insert(ptr, (order, *promise, value, self.host_job_context));
        }
        for (on_f, on_r, result, settings) in reactions {
            let handler = if fulfilled { on_f } else { on_r };
            let (realm, host_context) = self.reaction_job_target(&handler, settings);
            self.enqueue_promise_job(Job {
                handler,
                result,
                value,
                fulfilled,
                realm,
                host_context,
            });
        }
    }

    /// The `then` operation: register reactions, returning a new dependent promise.
    pub(crate) fn promise_then(&mut self, promise: &Value, on_f: Value, on_r: Value) -> Value {
        let result = self.new_promise();
        self.promise_then_into(promise, on_f, on_r, result);
        result
    }

    /// PerformPromiseThen with a caller-supplied result promise (so `Promise.prototype.then` can
    /// route through SpeciesConstructor + NewPromiseCapability).
    pub(crate) fn promise_then_into(
        &mut self,
        promise: &Value,
        on_f: Value,
        on_r: Value,
        result: Value,
    ) {
        if !matches!(&(promise).view(), ValueView::Obj(_)) {
            return;
        }
        // HostMakeJobCallback runs at registration, not settlement or job execution.
        let on_f = self.make_job_callback(on_f);
        let on_r = self.make_job_callback(on_r);
        self.perform_promise_then(promise, on_f, on_r, result);
    }

    /// ECMA-262 Await steps 3-7: PerformPromiseThen(promise, onFulfilled, onRejected) with the
    /// two unobservable built-in resumption steps and no result capability. `async_promise` is the
    /// suspended async function's result promise (its coroutine key).
    pub(crate) fn perform_await_then(&mut self, promise: &Value, async_promise: &Value) {
        if !matches!(&(promise).view(), ValueView::Obj(_)) {
            return;
        }
        let on_f = self.make_await_job_callback(*async_promise, true);
        let on_r = self.make_await_job_callback(*async_promise, false);
        self.perform_promise_then(promise, on_f, on_r, Value::Undefined);
    }

    /// PerformPromiseThen (ECMA-262 `#sec-performpromisethen`) for already-made JobCallbacks.
    /// `result` is the capability's promise, or `undefined` when there is no capability.
    pub(crate) fn perform_promise_then(
        &mut self,
        promise: &Value,
        on_f: crate::interpreter::JobCallback,
        on_r: crate::interpreter::JobCallback,
        result: Value,
    ) {
        let ptr = match &(promise).view() {
            ValueView::Obj(o) => Gc::as_ptr(o) as usize,
            _ => return,
        };
        let (status, handled, outstanding) =
            self.promises
                .get_mut(&self.gc_heap, &ptr)
                .map_or((0, true, None), |state| {
                    let flags = (state.status, state.handled, state.outstanding);
                    // Step 12: set promise.[[PromiseIsHandled]] to true.
                    state.handled = true;
                    state.outstanding = None;
                    flags
                });
        // Embedder reaction steps belong to no settings object (see `InternalJob::HostReaction`).
        let settings = self.reaction_settings(
            if on_f.internal == crate::interpreter::InternalJob::HostReaction {
                0
            } else {
                self.host_job_context
            },
        );
        // Step 11: a rejected, unhandled promise gets HostPromiseRejectionTracker(promise,
        // "handle"). HTML: one still on the about-to-be-notified list just leaves it; one in the
        // outstanding rejected promises weak set leaves that and is reported for
        // `rejectionhandled`.
        if status == 2 && !handled && self.unhandled_rejections.remove(&ptr).is_none() {
            if let Some(context) = outstanding {
                let reason = self.promises[&ptr].value;
                self.handled_rejections.push((*promise, reason, context));
            }
        }
        match status {
            0 => {
                if let Some(s) = self.promises.get_mut(&self.gc_heap, &ptr) {
                    s.reactions.push((on_f, on_r, result, settings));
                }
            }
            1 => {
                let v = self.promises[&ptr].value;
                let (realm, host_context) = self.reaction_job_target(&on_f, settings);
                self.enqueue_promise_job(Job {
                    handler: on_f,
                    result,
                    value: v,
                    fulfilled: true,
                    realm,
                    host_context,
                });
            }
            _ => {
                let v = self.promises[&ptr].value;
                let (realm, host_context) = self.reaction_job_target(&on_r, settings);
                self.enqueue_promise_job(Job {
                    handler: on_r,
                    result,
                    value: v,
                    fulfilled: false,
                    realm,
                    host_context,
                });
            }
        }
    }

    /// HostEnqueuePromiseJob's Realm/settings for a reaction handler. An internal step stands for
    /// a built-in function whose Realm was fixed when it was created.
    fn reaction_job_target(
        &mut self,
        handler: &crate::interpreter::JobCallback,
        settings: crate::interpreter::PromiseReactionSettings,
    ) -> (Option<usize>, u64) {
        if let Some((_, realm, context, _)) = handler.internal.routed_import() {
            return (Some(realm), context);
        }
        if handler.internal == crate::interpreter::InternalJob::HostReaction {
            // HTML HostEnqueuePromiseJob with a null Realm: null job settings.
            return (None, 0);
        }
        let Some(realm) = handler.internal.realm() else {
            return self.promise_job_target(&handler.callback, settings);
        };
        let host_context = self.promise_job_settings(realm, settings);
        (Some(realm), host_context)
    }

    /// The settings of a reaction registered now with the host settings token `context`.
    pub(crate) fn reaction_settings(
        &self,
        context: u64,
    ) -> crate::interpreter::PromiseReactionSettings {
        crate::interpreter::PromiseReactionSettings {
            realm: Gc::as_ptr(&self.global) as usize,
            context,
        }
    }

    /// HTML #hostenqueuepromisejob's job settings: those of the handler Realm `realm`. The token
    /// captured at registration tells apart settings objects that share its registration Realm,
    /// and is never reinterpreted in another Realm, the one settling the promise included.
    fn promise_job_settings(
        &self,
        realm: usize,
        registration: crate::interpreter::PromiseReactionSettings,
    ) -> u64 {
        if realm == registration.realm {
            registration.context
        } else if realm == Gc::as_ptr(&self.global) as usize {
            self.host_job_context
        } else {
            self.realm_host_context(realm, registration.context)
        }
    }

    /// Return the Realm/settings pair passed conceptually to HostEnqueuePromiseJob.
    ///
    /// ECMA-262 NewPromiseReactionJob and NewPromiseResolveThenableJob derive the Realm from the
    /// callable that will run. HTML then obtains the job settings from that Realm. A real
    /// cross-Realm callable therefore uses the target Realm's settings; the captured context is
    /// retained only when an embedder multiplexes multiple settings objects through one Realm.
    pub(crate) fn promise_job_target(
        &mut self,
        handler: &Value,
        settings: crate::interpreter::PromiseReactionSettings,
    ) -> (Option<usize>, u64) {
        if !handler.is_callable() {
            return (None, settings.context);
        }
        let current_realm = Gc::as_ptr(&self.global) as usize;
        let realm = match &(handler).view() {
            ValueView::Obj(handler) => {
                match self.get_function_realm_global(handler) {
                    Ok(Some(realm)) => realm,
                    // NewPromiseReactionJob uses the current Realm when GetFunctionRealm completes
                    // abruptly (for example, a revoked callable Proxy).
                    Ok(None) | Err(_) => current_realm,
                }
            }
            _ => current_realm,
        };
        let host_context = self.promise_job_settings(realm, settings);
        (Some(realm), host_context)
    }

    /// Resolve this agent's `Atomics.waitAsync` promises that can settle: with "ok" those that
    /// other agents notified (ECMA-262 EnqueueResolveInAgentJob), then, in deadline order, with
    /// "timed-out" those whose timeout passed while they were still listed
    /// (EnqueueAtomicsWaitAsyncTimeoutJob). A waiter a notify took just before its timeout is
    /// that notify's, and resolves "ok" when its notification arrives. Returns whether any
    /// promise resolved; their reactions are queued jobs.
    pub(crate) fn settle_async_waits(&mut self) -> bool {
        let heap = self.gc_heap.clone();
        let inbox = heap.agent.work.borrow().inbox.clone();
        let notified = inbox.map(|inbox| inbox.take()).unwrap_or_default();
        let mut settled = false;
        for token in notified {
            settled |= self.resolve_async_wait(token, "ok");
        }
        let now = std::time::Instant::now();
        let mut expired: Vec<_> = heap
            .agent
            .work
            .borrow()
            .waits
            .iter()
            .filter(|wait| {
                wait.payload
                    .deadline
                    .is_some_and(|deadline| deadline <= now)
            })
            .map(|wait| {
                (
                    wait.payload.deadline,
                    wait.payload.token,
                    wait.payload.id,
                    wait.payload.index,
                )
            })
            .collect();
        expired.sort_unstable();
        for (_, token, id, index) in expired {
            if crate::interpreter::futex_remove_async(id, index, token) {
                settled |= self.resolve_async_wait(token, "timed-out");
            }
        }
        settled
    }

    /// Token identities are Agent-wide. Remove the accepted host wait before dispatch, but
    /// keep its promise and complete ancestor route rooted until settlement has published jobs.
    pub(crate) fn resolve_async_wait(&mut self, token: u64, result: &str) -> bool {
        let wait = {
            let mut work = self.gc_heap.agent.work.borrow_mut();
            let Some(at) = work
                .waits
                .iter()
                .position(|wait| wait.payload.token == token)
            else {
                return false;
            };
            work.waits.remove(at)
        };
        let wait = crate::rooting::Rooted::new(wait);
        self.with_route(wait.route.target, |target| {
            target.resolve_promise(&wait.payload.promise, Value::lstr(result));
        });
        true
    }

    /// Earliest timeout anywhere in this Agent; future waits do not themselves count as ready
    /// checkpoint work or keep an ordinary host event loop alive.
    pub(crate) fn next_async_wait_deadline(&self) -> Option<std::time::Instant> {
        self.gc_heap
            .agent
            .work
            .borrow()
            .waits
            .iter()
            .filter_map(|wait| wait.payload.deadline)
            .min()
    }

    pub(crate) fn async_waits_ready(&self) -> bool {
        let work = self.gc_heap.agent.work.borrow();
        !work.waits.is_empty()
            && (work
                .inbox
                .as_ref()
                .is_some_and(|inbox| inbox.has_notifications())
                || work.waits.iter().any(|wait| {
                    wait.payload
                        .deadline
                        .is_some_and(|deadline| deadline <= std::time::Instant::now())
                }))
    }

    /// Test262's host explicitly waits for future Agent work. Ordinary checkpoints only run
    /// work that is ready now. A timer is one routed host task followed by a full checkpoint.
    pub(crate) fn run_agent_event_loop(&mut self) -> Result<(), crate::InterruptReason> {
        if self.realm_id != crate::agent_access::RealmId::ROOT
            || self.gc_heap.agent.execution_depth.get() != 0
            || self.gc_heap.agent.checkpoint.get()
        {
            self.gc_heap.agent.checkpoint_requested.set(true);
            return Ok(());
        }
        const IDLE_LIMIT: std::time::Duration = std::time::Duration::from_secs(4);
        self.drain_microtasks_interruptible()?;
        let mut idle_since = std::time::Instant::now();
        loop {
            let pending = {
                let work = self.gc_heap.agent.work.borrow();
                !work.waits.is_empty() || !work.timers.is_empty()
            };
            if !pending {
                break;
            }
            if let Err(Abrupt::Interrupt(reason)) = self.interrupt_poll_force() {
                self.cancel_agent_checkpoint();
                return Err(reason);
            }
            let mut resolved_any = self.settle_async_waits();
            loop {
                let timer = {
                    let mut work = self.gc_heap.agent.work.borrow_mut();
                    let now = std::time::Instant::now();
                    let at = work
                        .timers
                        .iter()
                        .enumerate()
                        .filter(|(_, timer)| timer.payload.deadline <= now)
                        .min_by_key(|(_, timer)| (timer.payload.deadline, timer.payload.order))
                        .map(|(index, _)| index);
                    at.map(|at| work.timers.remove(at))
                };
                let Some(timer) = timer else {
                    break;
                };
                let timer = crate::rooting::Rooted::new(timer);
                let result = self.with_route(timer.route.target, |target| {
                    target.with_author_entry(|target| {
                        target.call(timer.payload.callback, Value::Undefined, &[])
                    })
                });
                if let Err(Abrupt::Interrupt(reason)) = result {
                    self.cancel_agent_checkpoint();
                    return Err(reason);
                }
                self.drain_microtasks_interruptible()?;
                resolved_any = true;
            }
            if resolved_any {
                self.drain_microtasks_interruptible()?;
                idle_since = std::time::Instant::now();
                continue;
            }
            let give_up = idle_since + IDLE_LIMIT;
            if std::time::Instant::now() >= give_up {
                break;
            }
            let next_timer = self
                .gc_heap
                .agent
                .work
                .borrow()
                .timers
                .iter()
                .map(|timer| timer.payload.deadline)
                .min();
            let wake = [self.next_async_wait_deadline(), next_timer]
                .into_iter()
                .flatten()
                .fold(give_up, std::cmp::min);
            self.agent_async_inbox()
                .wait_until(wake, &self.runtime_interrupt);
        }
        Ok(())
    }

    pub(crate) fn drain_microtasks(&mut self) {
        let _ = self.drain_microtasks_interruptible();
    }

    /// HTML "perform a microtask checkpoint" (local snapshot e5071a20, `source`:123479): run
    /// jobs *while the queue is not empty*. There is deliberately no job budget — discarding
    /// queued PromiseReactionJobs would silently drop author continuations. Runaway loops are
    /// the host's concern and are stopped through [`crate::RuntimeInterrupt`], polled before
    /// every job below.
    pub(crate) fn drain_microtasks_interruptible(&mut self) -> Result<(), crate::InterruptReason> {
        self.agent_checkpoint(None).map(|_| ())
    }

    /// HostEnqueueFinalizationRegistryCleanupJob / CleanupFinalizationRegistry (ECMA-262
    /// §9.9.4.1 and §9.12). The entire held-value loop is one host cleanup task. Each target
    /// cell is removed before invocation; queued Promise jobs run after this whole task returns.
    pub(crate) fn run_finalization_cleanup_job(
        &mut self,
        registry: Value,
    ) -> Result<(), crate::InterruptReason> {
        let Some(ptr) = registry.as_obj().map(|object| Gc::as_ptr(&object) as usize) else {
            return Ok(());
        };
        if let Some(state) = self.finalization_registries.get_mut(&self.gc_heap, &ptr) {
            state.cleanup_scheduled = false;
        }
        self.weak_metadata.ready_registries.remove(&ptr);
        loop {
            let next = self
                .finalization_registries
                .get_mut(&self.gc_heap, &ptr)
                .and_then(|state| {
                    let cell = state.cells.pop_ready()?;
                    Some((state.cleanup_callback.clone(), cell.held_value))
                });
            let Some((callback, held_value)) = next else {
                self.weak_metadata.ready_registries.remove(&ptr);
                return Ok(());
            };
            if self
                .finalization_registries
                .get(&ptr)
                .is_some_and(|state| state.cells.has_ready())
            {
                // A callback may throw/interrupt or collect reentrantly. Preserve rescheduling
                // without making the next GC scan every registry and every live target.
                self.weak_metadata.ready_registries.insert(ptr);
            } else {
                self.weak_metadata.ready_registries.remove(&ptr);
            }
            let current = crate::rooting::Rooted::new((callback, held_value));
            match self.call_job_callback(
                &current.0,
                Value::Undefined,
                std::slice::from_ref(&current.1),
            ) {
                Ok(_) => {}
                // CleanupFinalizationRegistry uses `?`: the first callback throw ends this job.
                // The bare engine has no host error reporter; browser/Node embedders can add one
                // without changing the cell-removal semantics here.
                Err(Abrupt::Throw(_))
                | Err(Abrupt::Return(_))
                | Err(Abrupt::Break(_, _))
                | Err(Abrupt::Continue(_, _)) => return Ok(()),
                Err(Abrupt::Interrupt(reason)) => {
                    return Err(reason);
                }
            }
        }
    }

    /// Run one promise-reaction job (settle `job.result` from the handler, or pass the
    /// settlement through when there is no handler).
    pub(crate) fn run_job_interruptible(
        &mut self,
        job: &crate::interpreter::Job,
    ) -> Result<(), crate::InterruptReason> {
        // The current job remains a precise root throughout Realm/settings preparation and
        // host hooks. Its immutable payload is only shared while author callbacks execute.
        let current_realm = Gc::as_ptr(&self.global) as usize;
        if let Some(target_realm) = job.realm.filter(|realm| *realm != current_realm) {
            if let Some(target) = self
                .realms
                .get(&target_realm)
                .map(RealmState::snapshot_clone)
            {
                return self.with_realm_snapshot(target_realm, target, |interp| {
                    interp.run_job_in_active_realm(job)
                });
            }
        }
        self.run_job_in_active_realm(job)
    }

    /// Run an [`InternalJob`] step with the same entry-Realm and incumbent script-caller context
    /// that `call_job_callback` establishes around an ordinary handler.
    fn run_internal_job(&mut self, handler: &crate::interpreter::JobCallback, value: Value) {
        use crate::interpreter::InternalJob;
        let signal = match handler.internal {
            InternalJob::None
            | InternalJob::HostReaction
            | InternalJob::RoutedImportFulfilled { .. }
            | InternalJob::RoutedImportRejected { .. } => return,
            InternalJob::AwaitFulfilled { .. } => crate::coroutine::Resume::Next(value),
            InternalJob::AwaitRejected { .. } => crate::coroutine::Resume::Throw(value),
        };
        let ValueView::Obj(async_promise) = &(handler.callback).view() else {
            return;
        };
        let key = Gc::as_ptr(async_promise) as usize;
        let entry = if self.capture_job_script_caller {
            handler.internal.realm()
        } else {
            self.host_entry_realm
        };
        let async_promise = handler.callback;
        self.with_entry_context(self.script_entry, entry, |interp| {
            interp.with_job_script_context(handler.active_script.as_ref(), |ctx| {
                if let Some(global) = &handler.script_caller {
                    ctx.with_callback_script_caller(&Value::Obj(*global), |ctx| {
                        ctx.drive_async(key, async_promise, signal)
                    });
                } else {
                    ctx.drive_async(key, async_promise, signal);
                }
            });
        });
    }

    fn run_job_in_active_realm(
        &mut self,
        job: &crate::interpreter::Job,
    ) -> Result<(), crate::InterruptReason> {
        if job.realm.is_some() && job.host_context != self.host_job_context {
            return self.with_settings_context(job.host_context, |interp| interp.run_job_body(job));
        }
        self.run_job_body(job)
    }

    fn run_job_body(
        &mut self,
        job: &crate::interpreter::Job,
    ) -> Result<(), crate::InterruptReason> {
        if let Some((_, _, _, fulfilled)) = job.handler.internal.routed_import() {
            return self.finish_routed_import(job.handler.callback, job.value, fulfilled);
        }
        if job.handler.internal.resumes_await() {
            // Await's reaction built-ins return undefined and have no result capability; the
            // resumed function reports its own completion through its result promise.
            self.run_internal_job(&job.handler, job.value);
        } else if job.handler.callback.is_callable() {
            match self.call_job_callback(
                &job.handler,
                Value::Undefined,
                std::slice::from_ref(&job.value),
            ) {
                // A platform step's result is a platform resolution too.
                Ok(r) => self.resolve_promise_from(
                    &job.result,
                    r,
                    job.handler.internal == crate::interpreter::InternalJob::HostReaction,
                ),
                Err(Abrupt::Throw(e)) => self.reject_promise(&job.result, e),
                Err(Abrupt::Interrupt(reason)) => {
                    return Err(reason);
                }
                Err(_) => {}
            }
        } else if job.fulfilled {
            self.resolve_promise(&job.result, job.value);
        } else {
            self.reject_promise(&job.result, job.value);
        }
        Ok(())
    }

    /// Compile a regular expression and build a RegExp object (its metadata stored as own props,
    /// the compiled program in the `regexps` side table). A bad pattern throws a SyntaxError.
    pub(crate) fn make_regexp(&mut self, source: &str, flags: &str) -> Result<Value, Abrupt> {
        let proto = self.regexp_alloc_proto()?;
        self.make_regexp_with_proto(source, flags, proto)
    }

    pub(crate) fn make_regexp_compiled(
        &mut self,
        re: Rc<crate::regex::Regex>,
    ) -> Result<Value, Abrupt> {
        let proto = self.regexp_alloc_proto()?;
        Ok(self.make_regexp_object(re, proto))
    }

    /// A RegExp of the compiled program `re` whose prototype is `proto`, as RegExpAlloc makes it
    /// once GetPrototypeFromConstructor found `proto`.
    pub(crate) fn make_regexp_compiled_with_proto(
        &mut self,
        re: Rc<crate::regex::Regex>,
        proto: Option<crate::value::Gc>,
    ) -> Value {
        self.make_regexp_object(re, proto)
    }

    /// GetPrototypeFromConstructor for RegExpAlloc: `new` with a subclass/cross-realm newTarget
    /// overrides the instance prototype (falling back to newTarget's realm's %RegExp.prototype%).
    pub(crate) fn regexp_alloc_proto(&mut self) -> Result<Option<crate::value::Gc>, Abrupt> {
        Ok(if self.constructing {
            let nt = self.constructor_new_target();
            match nt.view() {
                ValueView::Obj(_) => match (self.get_member_key(&nt, "prototype")?).view() {
                    ValueView::Obj(p) => Some(p),
                    _ => crate::builtins::regexp_realm_proto(self, &nt)
                        .map_err(Abrupt::Throw)?
                        .or_else(|| self.extra_protos.get("RegExp").cloned()),
                },
                _ => self.extra_protos.get("RegExp").cloned(),
            }
        } else {
            self.extra_protos.get("RegExp").cloned()
        })
    }

    pub(crate) fn make_regexp_with_proto(
        &mut self,
        source: &str,
        flags: &str,
        proto: Option<crate::value::Gc>,
    ) -> Result<Value, Abrupt> {
        let re = self.compiled_regexp(source, flags)?;
        Ok(self.make_regexp_object(re, proto))
    }

    fn make_regexp_object(
        &mut self,
        re: Rc<crate::regex::Regex>,
        proto: Option<crate::value::Gc>,
    ) -> Value {
        let obj = Object::new(proto);
        let ptr = Gc::as_ptr(&obj) as usize;
        // source/flags/global/... are accessor getters on RegExp.prototype (computed from the
        // matcher); only `lastIndex` is an own writable data property.
        obj.borrow_mut().props.insert(
            "lastIndex",
            Property::data(Value::Num(0.0), true, false, false),
        );
        // Every RegExp starts at the same shape, which the RegExp protector's receivers keep.
        let shape = obj.borrow().props.shape();
        self.regexp_protector
            .borrow_mut()
            .note_instance_shape(shape);
        self.gc_pin(&obj);
        self.regexps.insert(ptr, re);
        Value::Obj(obj)
    }

    /// Return the immutable matcher shared by every RegExp with this source/flag pair. Observable
    /// state lives on the freshly allocated JS object, not in `Regex`, so sharing the program does
    /// not share `lastIndex` or object identity.
    pub(crate) fn compiled_regexp(
        &mut self,
        source: &str,
        flags: &str,
    ) -> Result<Rc<crate::regex::Regex>, Abrupt> {
        if let Some(re) = self.regexp_programs.get(source, flags) {
            return Ok(re);
        }
        let re = Rc::new(
            crate::regex::Regex::new(source, flags).map_err(|e| self.throw("SyntaxError", e))?,
        );
        let bytes = re
            .heap_bytes()
            .saturating_add(source.len())
            .saturating_add(flags.len())
            .saturating_add(std::mem::size_of::<(
                Rc<str>,
                Rc<str>,
                Rc<crate::regex::Regex>,
            )>());
        self.regexp_programs
            .insert(source, flags, re.clone(), bytes);
        Ok(re)
    }

    /// Direct eval: a non-string argument is returned unchanged; a string is parsed and executed.
    pub(crate) fn direct_eval(&mut self, arg: Option<&Value>, env: &Env) -> Result<Value, Abrupt> {
        let Some(value) = arg else {
            return Ok(Value::Undefined);
        };
        let code = match value.view() {
            ValueView::Str(s) => s,
            _ => return Ok(*value),
        };
        self.perform_eval(&code.text(), env, true)
    }

    /// PerformEval: parse `code`, set up its variable + lexical environments, run
    /// EvalDeclarationInstantiation, then execute the body.
    ///
    /// A *direct* eval (`direct` = true) inherits the caller's strictness and runs in `caller_env`:
    /// sloppy code hoists its `var`/function declarations into the caller's nearest variable
    /// environment while its lexical declarations stay private to a fresh scope. An *indirect* eval
    /// always runs in the global scope and is only strict via its own `"use strict"` directive.
    pub(crate) fn perform_eval(
        &mut self,
        code: &str,
        caller_env: &Env,
        direct: bool,
    ) -> Result<Value, Abrupt> {
        self.with_script_entry(|this| this.perform_eval_body(code, caller_env, direct))
    }

    fn perform_eval_body(
        &mut self,
        code: &str,
        caller_env: &Env,
        direct: bool,
    ) -> Result<Value, Abrupt> {
        let base_strict = direct && self.strict;
        // `new.target` is valid at the top level of a direct eval whose caller is in function code.
        let allow_new_target = direct && self.in_function_code(caller_env);
        // A direct eval may contain a SuperProperty; the runtime `super_base` lookup enforces that a
        // home object is actually in scope (throwing SyntaxError otherwise), so parse permissively.
        // A direct eval's code may reference any private name visible at the call site: collect
        // the `#name` bindings on the caller's scope chain to seed the parser's validation.
        let mut private_names: Vec<String> = Vec::new();
        if direct {
            let mut cur = Some(*caller_env);
            while let Some(s) = cur {
                let b = s.borrow();
                for k in b.vars.keys() {
                    let k = k.text();
                    if k.starts_with('#') {
                        private_names.push(k.into_owned());
                    }
                }
                cur = b.parent;
            }
        }
        // A direct eval may contain a SuperProperty only when its caller is method code (a home
        // object is in scope) — otherwise it's an early SyntaxError, before anything evaluates.
        let allow_super = direct
            && (self.peek_binding("%homeobject%", caller_env).is_some()
                || self.peek_binding("%superproto%", caller_env).is_some());
        let (body, source) = crate::parser::parse_script_eval(
            code,
            base_strict,
            allow_new_target,
            allow_super,
            &private_names,
        )
        .map_err(|e| self.throw("SyntaxError", e.message))?;
        // PerformEval: eval code runs with the caller's ScriptOrModule, which its functions also
        // capture (OrdinaryFunctionCreate takes GetActiveScriptOrModule()).
        if let Some(referrer) = self.active_script_referrer() {
            source.set_referrer(referrer);
        }
        // A direct `eval` inherits the caller's lexical super-call context: a `super(...)` in the
        // eval is an early SyntaxError unless GetThisEnvironment reaches a derived constructor
        // (possibly through arrows). Check before any eval-body side effect.
        if direct && !self.super_call_context_available(caller_env) && stmts_have_super_call(&body)
        {
            return Err(self.throw("SyntaxError", "'super' keyword unexpected here"));
        }
        // Likewise a `super.x` reference requires the eval to sit (lexically) in method or class
        // code — an ordinary nested function inside a method shields it, an arrow is transparent.
        if direct && stmts_have_super_prop(&body) {
            let mut ok = false;
            let mut cur = Some(*caller_env);
            while let Some(sc) = cur {
                let b = sc.borrow();
                if b.vars.contains_key("%superpropok%") || b.vars.contains_key("%homeobject%") {
                    ok = true;
                    break;
                }
                // An ordinary function boundary (binds `arguments`) without the marker shields
                // the capability.
                if b.vars.contains_key("arguments") {
                    break;
                }
                cur = b.parent;
            }
            if !ok {
                return Err(self.throw("SyntaxError", "'super' keyword unexpected here"));
            }
        }
        // A direct eval from class-field-initializer code may not reference `arguments` (also an
        // early error, checked before the body runs). The context is *lexical*: an arrow created
        // in an initializer keeps it, while an ordinary function (whose scope binds `arguments`)
        // shields it — so walk the caller's scope chain.
        let in_field_init = direct && {
            let mut found = false;
            let mut cur = Some(*caller_env);
            while let Some(sc) = cur {
                let b = sc.borrow();
                if b.vars.contains_key("arguments") {
                    break;
                }
                if b.vars.contains_key("%fieldinit%") {
                    found = true;
                    break;
                }
                cur = b.parent;
            }
            found
        };
        if in_field_init && stmts_have_arguments_ref(&body) {
            return Err(self.throw(
                "SyntaxError",
                "'arguments' is not allowed in class field initializer code",
            ));
        }
        let directive_strict = matches!(
            body.first(),
            Some(Stmt::Expr(Expr::Str(s))) if &**s == "use strict"
        );
        let strict = base_strict || directive_strict;

        // PerformEval steps: choose the variable and lexical environments.
        let (var_env, lex_env) = if !direct {
            if strict {
                // Strict indirect eval: its own environments — top-level declarations don't leak to
                // the global object.
                let e = new_var_scope(Some(self.global_env));
                (e, e)
            } else {
                // Sloppy indirect eval runs in the global scope.
                (self.global_env, new_scope(Some(self.global_env)))
            }
        } else if strict {
            // Strict direct eval: its own variable + lexical environment — nothing leaks out.
            let e = new_var_scope(Some(*caller_env));
            (e, e)
        } else {
            // Sloppy direct eval: `var`/function declarations hoist into the caller's nearest
            // variable environment; lexical declarations stay in a fresh, private lexical scope.
            (nearest_var_env(caller_env), new_scope(Some(*caller_env)))
        };

        self.in_code(Some(source), |this| {
            this.eval_declaration_instantiation(&body, &var_env, &lex_env, strict)?;
            let saved = this.strict;
            this.strict = strict;
            let result = this.run_eval_body(&body, &lex_env);
            this.strict = saved;
            result
        })
    }

    /// EvalDeclarationInstantiation: validate the eval body's `var`/function declarations against the
    /// surrounding environments (throwing `SyntaxError`/`TypeError` on a conflict), then create them.
    fn eval_declaration_instantiation(
        &mut self,
        body: &[Stmt],
        var_env: &Env,
        lex_env: &Env,
        strict: bool,
    ) -> Result<(), Abrupt> {
        // VarDeclaredNames (every `var` binding name plus hoisted function names, top-level and Annex
        // B.3.3 block-scoped) together with their instantiation values — gathered by hoisting into a
        // throwaway scope, which mirrors the interpreter's own var-scoping and function-precedence
        // rules exactly. Its parent is the eval's lexical environment so any function closes over it.
        // The hoist runs in the eval's strict mode so Annex B block functions apply only when sloppy.
        let probe = new_scope(Some(*lex_env));
        let saved_strict = self.strict;
        self.strict = strict;
        let hoist_ops = collect_hoist_ops(body, strict, &[]);
        self.apply_hoist_ops(&hoist_ops, &probe, false);
        self.strict = saved_strict;
        let var_bindings = ordered_global_hoist_bindings(&hoist_ops);
        let is_global = Env::ptr_eq(var_env, &self.global_env);

        if !strict {
            // A sloppy eval must not hoist a `var` over a same-named global lexical declaration...
            if is_global {
                for &(name, _) in &var_bindings {
                    if name != "this" && self.global_env.borrow().vars.contains_key(name) {
                        return Err(self.throw(
                            "SyntaxError",
                            format!("Identifier '{name}' has already been declared"),
                        ));
                    }
                }
            }
            // ...nor over a like-named lexical binding in any scope between the eval and its variable
            // environment (block `let`/`const`, a parameter scope's `arguments`/parameters, etc.).
            let mut cur = Some(*lex_env);
            while let Some(s) = cur {
                if Env::ptr_eq(&s, var_env) {
                    break;
                }
                let (skip, parent) = {
                    let b = s.borrow();
                    // A `with` object environment holds no lexical declarations, and a `catch`
                    // parameter environment is exempt from the var/lexical conflict check.
                    (b.has_with_object() || b.catch_param(), b.parent)
                };
                if !skip {
                    for &(name, _) in &var_bindings {
                        if s.borrow().vars.contains_key(name) {
                            return Err(self.throw(
                                "SyntaxError",
                                format!("Identifier '{name}' has already been declared"),
                            ));
                        }
                    }
                }
                cur = parent;
            }
            // The variable environment itself may hold body-level lexicals (our function body
            // scope carries both); a var may not hoist over one of those either.
            for &(name, _) in &var_bindings {
                if var_env
                    .borrow()
                    .vars
                    .get(name)
                    .is_some_and(|binding| binding.lexical())
                {
                    return Err(self.throw(
                        "SyntaxError",
                        format!("Identifier '{name}' has already been declared"),
                    ));
                }
            }
        }

        // A global variable environment can refuse a declaration (non-extensible global, or a
        // non-configurable same-named property) with a TypeError.
        if is_global {
            for &(name, is_func) in &var_bindings {
                let ok = if is_func {
                    self.can_declare_global_function(name)
                } else {
                    self.can_declare_global_var(name)
                };
                if !ok {
                    return Err(self.throw(
                        "TypeError",
                        format!("cannot declare global binding '{name}'"),
                    ));
                }
            }
        }

        // Instantiate each declared name from the value the probe hoist computed (a function object —
        // top-level or Annex B.3.3 block-scoped, later hoists winning — or `undefined` for a plain
        // `var`). A pre-existing `var` binding keeps its value; a function binding always overwrites.
        for &(name, is_func) in &var_bindings {
            let value = probe
                .borrow()
                .vars
                .get(name)
                .map(|b| b.value())
                .unwrap_or(Value::Undefined);
            if is_global {
                if is_func {
                    self.create_global_function_binding(name, value);
                } else {
                    self.create_global_var_binding(name);
                }
            } else if is_func {
                var_env.borrow_mut().vars.insert(
                    name.to_owned(),
                    Binding::data(value, true, true).deletable(),
                );
            } else if !var_env.borrow().vars.contains_key(name) {
                var_env.borrow_mut().vars.insert(
                    name.to_owned(),
                    Binding::data(Value::Undefined, true, true).deletable(),
                );
            }
        }
        // The probe scope sits in every hoisted function's closure chain; empty it so their
        // variable references resolve through to the real environments (the probe's stale
        // `undefined` bindings must not shadow the caller's vars).
        probe.borrow_mut().vars.clear();
        // Lexical declarations (`let`/`const`/`class`) stay in the eval's private lexical scope.
        self.declare_block_lexicals(body, lex_env, false);
        Ok(())
    }

    /// CanDeclareGlobalVar: a global `var` is definable if the property already exists or the global
    /// object is extensible.
    fn can_declare_global_var(&self, name: &str) -> bool {
        if self.global.borrow().props.contains(name) {
            return true;
        }
        self.global.borrow().extensible
    }

    /// CanDeclareGlobalFunction: definable if an existing property is configurable (or a writable,
    /// enumerable data property), or — when absent — the global object is extensible.
    fn can_declare_global_function(&self, name: &str) -> bool {
        let g = self.global.borrow();
        match g.props.get(name) {
            None => g.extensible,
            Some(p) => p.configurable() || (!p.accessor() && p.writable() && p.enumerable()),
        }
    }

    /// CreateGlobalFunctionBinding(name, value, deletable=true): define (or overwrite) a configurable
    /// global function property.
    fn create_global_function_binding(&mut self, name: &str, value: Value) {
        let existing = self
            .global
            .borrow()
            .props
            .get(name)
            .map(|p| (p.configurable(), !p.accessor()));
        match existing {
            Some((false, true)) => {
                // A pre-existing non-configurable data property keeps its attributes; only its value
                // is replaced.
                self.global.borrow_mut().props.set_value(name, value);
            }
            _ => {
                self.global
                    .borrow_mut()
                    .props
                    .insert(name, Property::data(value, true, true, true));
            }
        }
    }

    /// CreateGlobalVarBinding(name, deletable=true): create a configurable global var property if the
    /// global object doesn't already have one.
    fn create_global_var_binding(&mut self, name: &str) {
        if self.global.borrow().props.contains(name) {
            return;
        }
        if self.global.borrow().extensible {
            self.global
                .borrow_mut()
                .props
                .insert(name, Property::data(Value::Undefined, true, true, true));
        }
    }

    /// Execute an eval body's statements in its lexical environment (declarations already
    /// instantiated), returning the completion value of the last value-producing statement.
    fn run_eval_body(&mut self, body: &[Stmt], lex_env: &Env) -> Result<Value, Abrupt> {
        let has_using = body.iter().any(stmt_declares_using);
        if has_using {
            self.using_stack.push(Vec::new());
        }
        let mut last = Value::Undefined;
        let mut result: Completion = Ok(Value::Undefined);
        for stmt in body {
            match self.exec_stmt(stmt, lex_env) {
                Ok(v) => {
                    if !matches!((v).view(), ValueView::Empty) {
                        last = v;
                    }
                }
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        if has_using {
            let frame = self.using_stack.pop().unwrap_or_default();
            result = self.dispose_frame(frame, result);
        }
        result?;
        Ok(last)
    }

    // ----- classes ----------------------------------------------------------------------------

    fn eval_class(&mut self, class: &Rc<Class>, env: &Env) -> Result<Value, Abrupt> {
        // The decorators proposal evaluates class decorator expressions before
        // ClassDefinitionEvaluation (and therefore before the class name's TDZ and heritage).
        // Evaluated decorators stay rooted while later expressions run author code (I1).
        let class_decorators = crate::rooting::Rooted::new(std::cell::RefCell::new(
            Vec::with_capacity(class.decorators.len()),
        ));
        for decorator in &class.decorators {
            let value = self.eval_decorator_expression(decorator, env)?;
            class_decorators.borrow_mut().push(value);
        }
        // ClassDefinitionEvaluation is strict code throughout — the heritage expression and
        // computed member keys included, whatever the surrounding mode.
        let saved_strict = self.strict;
        self.strict = true;
        let r = self.eval_class_strict(class, env, class_decorators);
        self.strict = saved_strict;
        r
    }

    fn eval_class_strict(
        &mut self,
        class: &Rc<Class>,
        env: &Env,
        class_decorators: crate::rooting::Rooted<std::cell::RefCell<Vec<PreparedDecoratorValue>>>,
    ) -> Result<Value, Abrupt> {
        // Rooted through heritage, decorator and computed-key evaluation and the decorator
        // application that finishes the class (I1).
        let prepared = crate::rooting::Rooted::new(std::cell::RefCell::new(
            self.begin_class_evaluation(class, env),
        ));
        for decorator in class_decorators.borrow().iter() {
            self.prepare_class_decorator(
                &mut prepared.borrow_mut(),
                None,
                decorator.callback,
                decorator.this_value,
            );
        }
        drop(class_decorators);
        let outer = prepared.borrow().outer_class_env;
        let parent = match &class.superclass {
            Some(expression) => Some(self.eval(expression, &outer)?),
            None => None,
        };
        let heritage = self.class_heritage(parent)?;
        prepared.borrow_mut().publish_heritage(heritage);
        for (index, member) in class.members.iter().enumerate() {
            let env = prepared.borrow().class_env;
            for decorator in &member.decorators {
                let value = self.eval_decorator_expression(decorator, &env)?;
                self.prepare_class_decorator(
                    &mut prepared.borrow_mut(),
                    Some(index),
                    value.callback,
                    value.this_value,
                );
            }
            if member.kind == MemberKind::Constructor {
                continue;
            }
            if let PropKey::Computed(expression) = &member.key {
                let value = self.eval(expression, &env)?;
                let key = self.class_computed_key(class, index, value)?;
                prepared.borrow_mut().publish_key(index, key);
            }
        }
        let prepared = crate::rooting::Rooted::new(prepared.into_inner().into_inner());
        self.finish_class_evaluation(class, prepared)
    }

    /// Create the lexical and private environments that precede heritage evaluation.
    pub(crate) fn begin_class_evaluation(
        &mut self,
        class: &Rc<Class>,
        env: &Env,
    ) -> PreparedClassEvaluation {
        // The class scope opens before the heritage evaluates: a named class's own name is in
        // scope there — uninitialized (TDZ) until the constructor exists, and immutable.
        let outer_class_env = new_scope(Some(*env));
        if let Some(n) = &class.name {
            outer_class_env
                .borrow_mut()
                .vars
                .insert(n.clone(), Binding::data(Value::Undefined, false, false));
        }
        // The spec's PrivateEnvironment is also allocated before heritage but becomes active only
        // for class elements. TRust JS represents it as a child lexical record carrying unforgeable
        // runtime private-name keys.
        let class_env = new_scope(Some(outer_class_env));
        for member in &class.members {
            let source = match &member.key {
                PropKey::Ident(name) if name.starts_with('#') => name.clone(),
                _ => continue,
            };
            if !class_env.borrow().vars.contains_key(source.as_str()) {
                self.accessor_seq += 1;
                let runtime = format!("{}\u{1}{}", source, self.accessor_seq);
                bind(&class_env, &source, Value::lstr(runtime.as_str()));
            }
        }
        PreparedClassEvaluation {
            outer_class_env,
            class_env,
            proto_parent: Some(self.object_proto),
            ctor_parent: None,
            derived: false,
            keys: (0..class.members.len()).map(|_| None).collect(),
            decorator_values: (!class.decorators.is_empty()
                || class
                    .members
                    .iter()
                    .any(|member| !member.decorators.is_empty()))
            .then(|| PreparedDecoratorValues {
                class: Vec::with_capacity(class.decorators.len()),
                members: class
                    .members
                    .iter()
                    .map(|member| Vec::with_capacity(member.decorators.len()))
                    .collect(),
            }),
        }
    }

    /// Retain one already-evaluated decorator expression in source order. The proposal separates
    /// this observable expression-evaluation phase from reverse-order decorator application.
    pub(crate) fn prepare_class_decorator(
        &mut self,
        prepared: &mut PreparedClassEvaluation,
        member: Option<usize>,
        callback: Value,
        this_value: Value,
    ) {
        let value = PreparedDecoratorValue {
            callback,
            this_value,
        };
        let values = prepared
            .decorator_values
            .as_mut()
            .expect("decorated classes allocate decorator continuation state");
        match member {
            Some(index) => values.members[index].push(value),
            None => values.class.push(value),
        }
    }

    /// Evaluate one proposal Decorator expression and preserve the Reference receiver separately
    /// from GetValue. This is observable when `@holder.decorator` is later invoked: the March 2023
    /// normative update calls it with `holder`, whereas a bare/call expression uses `undefined`.
    fn eval_decorator_expression(
        &mut self,
        expression: &Expr,
        env: &Env,
    ) -> Result<PreparedDecoratorValue, Abrupt> {
        match expression {
            Expr::Paren(inner) => self.eval_decorator_expression(inner, env),
            Expr::Member {
                obj,
                prop,
                optional: false,
                ..
            } if !matches!(**obj, Expr::Super) => {
                let this_value = self.eval(obj, env)?;
                let callback = if prop.starts_with('#') {
                    let key = self.resolve_private(prop, env);
                    self.get_private_member(&this_value, &key)?
                } else {
                    self.get_member_key(&this_value, prop)?
                };
                Ok(PreparedDecoratorValue {
                    callback,
                    this_value,
                })
            }
            _ => Ok(PreparedDecoratorValue {
                callback: self.eval(expression, env)?,
                this_value: Value::Undefined,
            }),
        }
    }

    /// Complete the heritage portion through Get(superclass, "prototype") before any computed
    /// member name is evaluated.
    pub(crate) fn class_heritage(
        &mut self,
        parent: Option<Value>,
    ) -> Result<(Option<Gc>, Option<Value>, bool), Abrupt> {
        // (IsConstructor runs BEFORE any `prototype` read; a callable non-constructor like
        // %IsHTMLDDA% is a TypeError without observable gets.)
        let (proto_parent, ctor_parent): (Option<Gc>, Option<Value>) =
            match parent.map(|value| value.view()) {
                None => (Some(self.object_proto), None),
                Some(ValueView::Null) => (None, None),
                Some(ValueView::Obj(pc)) if self.value_is_constructor(parent.as_ref().unwrap()) => {
                    let pp = self.get_member_key(parent.as_ref().unwrap(), "prototype")?;
                    let pp = match (pp).view() {
                        ValueView::Obj(o) => Some(o),
                        ValueView::Null => None,
                        _ => {
                            return Err(self.throw(
                                "TypeError",
                                "Class extends value does not have a valid prototype property",
                            ));
                        }
                    };
                    (pp, Some(Value::Obj(pc)))
                }
                _ => {
                    return Err(self.throw(
                        "TypeError",
                        "Class extends value is not a constructor or null",
                    ));
                }
            };
        Ok((proto_parent, ctor_parent, parent.is_some()))
    }

    /// Record one already-evaluated computed ClassElementName after its single ToPropertyKey.
    pub(crate) fn class_computed_key(
        &mut self,
        class: &Class,
        member: usize,
        value: Value,
    ) -> Result<crate::value::PropertyKey, Abrupt> {
        let key = self.to_property_key(&value)?;
        if class.members[member].is_static && key == "prototype" {
            return Err(self.throw(
                "TypeError",
                "classes may not have a static property named 'prototype'",
            ));
        }
        Ok(key)
    }

    /// Finish the atomic portion of ClassDefinitionEvaluation from retained environments,
    /// validated heritage, and precomputed keys. No expression which can suspend remains here.
    pub(crate) fn finish_class_evaluation(
        &mut self,
        class: &Rc<Class>,
        prepared: crate::rooting::Rooted<PreparedClassEvaluation>,
    ) -> Result<Value, Abrupt> {
        let outer_class_env = prepared.outer_class_env;
        let class_env = prepared.class_env;
        let proto_parent = prepared.proto_parent;
        let ctor_parent = prepared.ctor_parent;
        let derived = prepared.derived;

        let proto = Object::new(proto_parent);

        // The constructor: explicit member, or a synthesized default. Either way its rendered
        // source (Function.prototype.toString) is the whole class's source text.
        let ctor_func = class
            .members
            .iter()
            .find(|m| m.kind == MemberKind::Constructor)
            .and_then(|m| m.func.clone())
            .unwrap_or_else(|| Rc::new(default_constructor(derived, class)));
        let ctor_func = if class.source.is_some() {
            let mut f = (*ctor_func).clone();
            f.source = class.source.clone();
            Rc::new(f)
        } else {
            ctor_func
        };

        // Environments that carry the `super` and already-minted private-name bindings into
        // methods/fields.
        let inst_env = new_scope(Some(class_env));
        // Instance members' [[HomeObject]] is the prototype: `super.x` resolves against its
        // *live* [[GetPrototypeOf]].
        bind(&inst_env, "%homeobject%", Value::Obj(proto));
        bind(&inst_env, "%superproto%", opt_obj(&proto_parent));
        // `extends null` binds Null (a super() call is a TypeError, not a SyntaxError).
        bind(
            &inst_env,
            "%superclass%",
            ctor_parent.unwrap_or(if derived {
                Value::Null
            } else {
                Value::Undefined
            }),
        );
        let static_env = new_scope(Some(class_env));
        // Static elements' [[HomeObject]] is the constructor: their super base is the parent
        // constructor, or %Function.prototype% for a base class.
        bind(
            &static_env,
            "%superproto%",
            ctor_parent.unwrap_or_else(|| Value::Obj(self.function_proto)),
        );

        // Build the constructor object on `proto`.
        let ctor_val = self.make_function(ctor_func, inst_env);
        let ctor_obj = ctor_val.as_obj().unwrap();
        // Static members' [[HomeObject]] is the constructor itself (bound after it exists; the
        // scope cell is shared with the already-captured static member environments).
        bind(&static_env, "%homeobject%", ctor_val);
        {
            let mut b = ctor_obj.borrow_mut();
            b.props.insert(
                "prototype",
                Property::data(Value::Obj(proto), false, false, false),
            );
            b.proto = match ctor_parent.map(|__value| (__value).view()) {
                Some(ValueView::Obj(p)) => Some(p),
                _ => Some(self.function_proto),
            };
            if let Some(n) = &class.name {
                b.props.insert(
                    "name",
                    Property::data(Value::lstr(n.clone()), false, false, true),
                );
            }
        }
        proto
            .borrow_mut()
            .props
            .insert("constructor", Property::builtin(ctor_val));
        bind(&inst_env, "%thisctor%", ctor_val);
        // NamedEvaluation: an anonymous class takes its target name *before* any static
        // initializer runs.
        if class.name.is_none() {
            if let Some(n) = self.pending_fn_name.take() {
                self.set_fn_name(&ctor_val, &n);
            }
        }
        // A named class binds its own name in the class scope, so methods, static blocks and
        // field initializers can reference the class itself. The binding stays in its TDZ until
        // every member's computed key has evaluated (spec: InitializeBinding runs after
        // ClassElementEvaluation, below), so `[Name]` as a computed key is a ReferenceError.

        // Methods, accessors and fields retain their actual records, including computed
        // Symbol keys, while later decorators and static elements run author code (I1).
        let pending =
            crate::rooting::Rooted::new(std::cell::RefCell::new(PendingClassElements::default()));
        // Instance private methods/accessors: stamped onto each instance when `this` is created
        // (PrivateMethodOrAccessorAdd), not placed on the prototype — so brand checks, double
        // initialization, and return-override semantics fall out of own-property checks.
        // Static elements (field initializers and static blocks) defer until every member's
        // computed key has been evaluated, then run in declaration order.
        let mut instance_inits = crate::rooting::RootedVec::new();
        let mut static_inits = crate::rooting::RootedVec::new();
        for (member_index, m) in class.members.iter().enumerate() {
            if m.kind == MemberKind::Constructor {
                continue;
            }
            // Computed keys evaluate with the class's PrivateEnvironment active: an arrow made
            // inside one may capture `A.#x` (class_env holds the runtime private-name bindings).
            let key = match prepared.keys[member_index].clone() {
                Some(key) => key,
                None => self.eval_prop_key(&m.key, &class_env)?,
            };
            // A *computed* static member key evaluating to "prototype" is a runtime TypeError
            // (the syntactic form is already an early error).
            if m.is_static && key == "prototype" && matches!(m.key, PropKey::Computed(_)) {
                return Err(self.throw(
                    "TypeError",
                    "classes may not have a static property named 'prototype'",
                ));
            }
            // Only a syntactic PrivateIdentifier is private; a computed key that merely
            // evaluates to a "#..." string is an ordinary property name.
            let is_private = matches!(&m.key, PropKey::Ident(n) if n.starts_with('#'));
            let key = if is_private {
                crate::value::PropertyKey::private(self.resolve_private(&key, &class_env))
            } else {
                key
            };
            let menv = if m.is_static { &static_env } else { &inst_env };
            let target = if m.is_static { ctor_obj } else { proto };
            match m.kind {
                MemberKind::Accessor => {
                    // An auto-accessor: a private backing field plus a brand-checked getter/setter.
                    // The backing key is globally unique so a subclass accessor never collides with
                    // a superclass one on the same instance.
                    self.accessor_seq += 1;
                    let backing: Rc<str> =
                        Rc::from(format!("#\u{0}acc{}", self.accessor_seq).as_str());
                    let display = self.fn_name_for_key(if key.is_private() {
                        private_display(&key)
                    } else {
                        &key
                    });
                    let mut getter = self.make_accessor_fn(&display, &backing, true);
                    let mut setter = self.make_accessor_fn(&display, &backing, false);
                    let mut transforms = Vec::new();
                    if !m.decorators.is_empty() {
                        let decorators = &prepared
                            .decorator_values
                            .as_ref()
                            .expect("decorated class retains evaluated expressions")
                            .members[member_index];
                        let sink = if m.is_static {
                            &mut static_inits
                        } else {
                            &mut instance_inits
                        };
                        let (g, s, t) = self.decorate_accessor(
                            decorators,
                            &key,
                            m.is_static,
                            getter,
                            setter,
                            sink,
                        )?;
                        getter = g;
                        setter = s;
                        transforms = t;
                    }
                    if is_private && !m.is_static {
                        pending.borrow_mut().private_members.push((
                            key.to_string(),
                            Property::accessor_prop(Some(getter), Some(setter), false, false),
                        ));
                    } else {
                        self.define_class_accessor(&target, &key, Some(getter), Some(setter));
                    }
                    if m.is_static {
                        pending
                            .borrow_mut()
                            .static_elements
                            .push(StaticClassElement::Field(FieldInit {
                                key: crate::value::PropertyKey::private(backing.to_string()),
                                init: m.value.clone(),
                                transforms,
                            }));
                    } else {
                        pending.borrow_mut().instance_fields.push(FieldInit {
                            key: crate::value::PropertyKey::private(backing.to_string()),
                            init: m.value.clone(),
                            transforms,
                        });
                    }
                }
                MemberKind::Method => {
                    let mut f = self.make_function(m.func.clone().unwrap(), *menv);
                    if let ValueView::Obj(fo) = &(f).view() {
                        let name = self.fn_name_for_key(if key.is_private() {
                            private_display(&key)
                        } else {
                            &key
                        });
                        fo.borrow_mut().props.insert(
                            "name",
                            Property::data(Value::lstr(name), false, false, true),
                        );
                    }
                    if !m.decorators.is_empty() {
                        let decorators = &prepared
                            .decorator_values
                            .as_ref()
                            .expect("decorated class retains evaluated expressions")
                            .members[member_index];
                        let sink = if m.is_static {
                            &mut static_inits
                        } else {
                            &mut instance_inits
                        };
                        f = self.decorate_callable(
                            decorators,
                            f,
                            "method",
                            &key,
                            m.is_static,
                            is_private,
                            sink,
                        )?;
                    }
                    let prop = if is_private {
                        // A private method is not writable: PrivateSet on it must TypeError.
                        Property::data(f, false, false, false)
                    } else {
                        Property::builtin(f)
                    };
                    if is_private && !m.is_static {
                        pending
                            .borrow_mut()
                            .private_members
                            .push((key.into_string(), prop));
                    } else if is_private {
                        target
                            .borrow_mut()
                            .internal_slots_mut()
                            .insert(key.as_str(), prop);
                    } else {
                        target.borrow_mut().props.insert(key.as_str(), prop);
                    }
                }
                MemberKind::Get | MemberKind::Set => {
                    let mut f = self.make_function(m.func.clone().unwrap(), *menv);
                    let is_get = m.kind == MemberKind::Get;
                    if let ValueView::Obj(fo) = &(f).view() {
                        let prefix = if is_get { "get " } else { "set " };
                        let name = self.fn_name_for_key(if key.is_private() {
                            private_display(&key)
                        } else {
                            &key
                        });
                        fo.borrow_mut().props.insert(
                            "name",
                            Property::data(
                                Value::lstr(format!("{prefix}{name}")),
                                false,
                                false,
                                true,
                            ),
                        );
                    }
                    if !m.decorators.is_empty() {
                        let decorators = &prepared
                            .decorator_values
                            .as_ref()
                            .expect("decorated class retains evaluated expressions")
                            .members[member_index];
                        let sink = if m.is_static {
                            &mut static_inits
                        } else {
                            &mut instance_inits
                        };
                        let kind = if is_get { "getter" } else { "setter" };
                        f = self.decorate_callable(
                            decorators,
                            f,
                            kind,
                            &key,
                            m.is_static,
                            is_private,
                            sink,
                        )?;
                    }
                    let (get, set) = if is_get {
                        (Some(f), None)
                    } else {
                        (None, Some(f))
                    };
                    if is_private && !m.is_static {
                        let mut records = pending.borrow_mut();
                        if let Some((_, p)) = records
                            .private_members
                            .iter_mut()
                            .find(|(k, _)| k.as_str() == key.as_str())
                        {
                            if get.is_some() {
                                p.set_getter(get);
                            }
                            if set.is_some() {
                                p.set_setter(set);
                            }
                        } else {
                            records.private_members.push((
                                key.into_string(),
                                Property::accessor_prop(get, set, false, false),
                            ));
                        }
                    } else {
                        self.define_class_accessor(&target, &key, get, set);
                    }
                }
                MemberKind::Field => {
                    let transforms = if m.decorators.is_empty() {
                        Vec::new()
                    } else {
                        let decorators = &prepared
                            .decorator_values
                            .as_ref()
                            .expect("decorated class retains evaluated expressions")
                            .members[member_index];
                        let sink = if m.is_static {
                            &mut static_inits
                        } else {
                            &mut instance_inits
                        };
                        self.decorate_field(decorators, &key, m.is_static, is_private, sink)?
                    };
                    if m.is_static {
                        pending
                            .borrow_mut()
                            .static_elements
                            .push(StaticClassElement::Field(FieldInit {
                                key,
                                init: m.value.clone(),
                                transforms,
                            }));
                    } else {
                        pending.borrow_mut().instance_fields.push(FieldInit {
                            key,
                            init: m.value.clone(),
                            transforms,
                        });
                    }
                }
                MemberKind::StaticBlock => {
                    if let Some(func) = &m.func {
                        pending
                            .borrow_mut()
                            .static_elements
                            .push(StaticClassElement::Block(func.clone()));
                    }
                }
                MemberKind::Constructor => {}
            }
        }

        // Every member key has evaluated: the class name binding leaves its TDZ before static
        // field initializers and static blocks run (InitializeBinding).
        if let Some(n) = &class.name {
            if let Some(mut b) = outer_class_env.borrow_mut().vars.get_mut(n) {
                b.initialize(ctor_val);
            }
        }

        // ECMA-262 ClassDefinitionEvaluation installs [[Fields]] and [[PrivateMethods]]
        // before running static elements. A static initializer can construct this class,
        // including a derived class, and must see its complete instance initialization.
        // Keep this metadata even if a static element throws: the constructor may escape.
        // https://tc39.es/ecma262/#sec-runtime-semantics-classdefinitionevaluation
        self.gc_pin(&ctor_obj);
        let (fields, private_members) = {
            let mut records = pending.borrow_mut();
            (
                std::mem::take(&mut records.instance_fields).into(),
                std::mem::take(&mut records.private_members),
            )
        };
        self.class_info.insert(
            &self.gc_heap,
            Gc::as_ptr(&ctor_obj) as usize,
            ClassInfo {
                fields,
                field_env: inst_env,
                instance_initializers: instance_inits.into_vec(),
                private_members,
            },
        );
        ctor_obj.borrow_mut().class_kind = if derived {
            crate::value::CLASS_DERIVED
        } else {
            crate::value::CLASS_BASE
        };

        // Borrow each record without detaching the backing list: both the current key and
        // the remaining static records stay traced through initializer/decorator callbacks.
        let static_elements = pending.borrow();
        for element in static_elements.static_elements.iter() {
            let scope = new_scope(Some(static_env));
            bind(&scope, "this", ctor_val);
            if let StaticClassElement::Block(func) = element {
                // A static block instantiates its declarations like a function body,
                // including a `using` disposal frame.
                self.hoist(&func.body, &scope, &[]);
                self.declare_block_lexicals(&func.body, &scope, false);
                let has_using = func.body.iter().any(stmt_declares_using);
                if has_using {
                    self.using_stack.push(Vec::new());
                }
                let mut result: Completion = Ok(Value::Undefined);
                for stmt in &func.body {
                    if let Err(e) = self.exec_stmt(stmt, &scope) {
                        result = Err(e);
                        break;
                    }
                }
                if has_using {
                    let frame = self.using_stack.pop().unwrap_or_default();
                    result = self.dispose_frame(frame, result);
                }
                result?;
                continue;
            }
            let StaticClassElement::Field(FieldInit {
                key,
                init,
                transforms,
            }) = element
            else {
                unreachable!("static blocks continue above");
            };
            bind(&scope, "%fieldinit%", Value::Bool(true));
            // A static field initializer is field-initializer code: no super() and no
            // `arguments` (even through a direct eval); an anonymous function value takes
            // the field's name.
            let saved_super = self.super_call_ok;
            let saved_field = self.in_field_init_code;
            self.super_call_ok = false;
            self.in_field_init_code = true;
            let v = match init {
                Some(e) => {
                    let v = self.eval(e, &scope);
                    if let (Ok(v), true) = (&v, is_anonymous_fn(e)) {
                        let name = self.fn_name_for_key(if key.is_private() {
                            private_display(key)
                        } else {
                            key
                        });
                        self.set_fn_name(v, &name);
                    }
                    v
                }
                None => Ok(Value::Undefined),
            };
            self.super_call_ok = saved_super;
            self.in_field_init_code = saved_field;
            let mut v = v?;
            for tr in transforms {
                v = self.call(*tr, ctor_val, &[v])?;
            }
            // The supported nonextensible-applies-to-private proposal checks IsExtensible
            // before adding a new Private Name. Existing private state remains writable
            // after freeze; this check does not make it an ordinary property.
            if key.is_private()
                && !crate::builtins::js_is_extensible(self, &ctor_val).map_err(Abrupt::Throw)?
            {
                return Err(self.throw(
                    "TypeError",
                    "cannot add a private field to a non-extensible object",
                ));
            }
            if key.is_private() && ctor_obj.borrow().internal_slots().contains(key.as_str()) {
                return Err(self.throw(
                    "TypeError",
                    "cannot initialize the same private field twice",
                ));
            }
            if key.is_private() {
                ctor_obj
                    .borrow_mut()
                    .internal_slots_mut()
                    .insert(key.as_str(), Property::plain(v));
            } else {
                ctor_obj
                    .borrow_mut()
                    .props
                    .insert(key.as_str(), Property::plain(v));
            }
        }

        // Class decorators apply after the body is built; a callable return replaces the class.
        let mut class_value = ctor_val;
        if !class.decorators.is_empty() {
            let name = class.name.clone().unwrap_or_default();
            let decorators = &prepared
                .decorator_values
                .as_ref()
                .expect("decorated class retains evaluated expressions")
                .class;
            class_value = self.decorate_callable(
                decorators,
                class_value,
                "class",
                &name,
                false,
                false,
                &mut static_inits,
            )?;
        }
        // Static element initializers and class initializers run with `this` = the class.
        for init in &static_inits {
            self.call(*init, class_value, &[])?;
        }
        Ok(class_value)
    }

    /// Build an auto-accessor's getter (`is_get`) or setter as a function object backed by the
    /// private `backing` field, with a spec-shaped `name` (`get x`/`set x`) and `length`.
    fn make_accessor_fn(&self, name: &str, backing: &Rc<str>, is_get: bool) -> Value {
        let o = Object::new(Some(self.function_proto));
        {
            let mut b = o.borrow_mut();
            b.call = if is_get {
                Callable::accessor_key(Callable::AccessorGet, backing.clone())
            } else {
                Callable::accessor_key(Callable::AccessorSet, backing.clone())
            };
            let prefix = if is_get { "get " } else { "set " };
            b.props.insert(
                "name",
                Property::data(Value::lstr(format!("{prefix}{name}")), false, false, true),
            );
            b.props.insert(
                "length",
                Property::data(
                    Value::Num(if is_get { 0.0 } else { 1.0 }),
                    false,
                    false,
                    true,
                ),
            );
        }
        Value::Obj(o)
    }

    fn define_class_accessor(
        &self,
        target: &Gc,
        key: &crate::value::PropertyKey,
        get: Option<Value>,
        set: Option<Value>,
    ) {
        let mut b = target.borrow_mut();
        if key.is_private() {
            if let Some(p) = b.internal_slots_mut().get_mut(key) {
                if p.accessor() {
                    if get.is_some() {
                        p.set_getter(get);
                    }
                    if set.is_some() {
                        p.set_setter(set);
                    }
                    return;
                }
            }
        } else if b.props.merge_accessor(key.as_str(), get, set) {
            return;
        }
        let property = Property::accessor_prop(get, set, false, true);
        if key.is_private() {
            b.internal_slots_mut().insert(key.as_str(), property);
        } else {
            b.props.insert(key.as_str(), property);
        }
    }

    /// A function object wrapping an internal [`Callable`] (used for accessor get/set and decorator
    /// `access` helpers), with a spec-shaped `name`/`length`.
    fn make_callable(&self, call: Callable, name: &str, len: f64) -> Value {
        let o = Object::new(Some(self.function_proto));
        {
            let mut b = o.borrow_mut();
            b.call = call;
            b.props.insert(
                "name",
                Property::data(Value::lstr(name.to_string()), false, false, true),
            );
            b.props.insert(
                "length",
                Property::data(Value::Num(len), false, false, true),
            );
        }
        Value::Obj(o)
    }

    /// Build a decorator context object: `{ kind, name, static, private, access, addInitializer,
    /// metadata }`. `access.get/set` read/write the element's property on a receiver.
    fn make_decorator_context(
        &mut self,
        kind: &str,
        key: &str,
        is_static: bool,
        is_private: bool,
    ) -> Value {
        let ctx = self.new_object();
        let cv = Value::Obj(ctx);
        let key_rc: Rc<str> = Rc::from(key);
        let name = if !is_private && Self::is_sym_key(key) {
            self.sym_from_key(key).unwrap_or(Value::Undefined)
        } else if is_private {
            // A private member's context name is its source spelling, not the runtime key.
            Value::lstr(private_display(key).to_string())
        } else {
            crate::value::string_key_value(key)
        };
        let access = self.new_object();
        if matches!(kind, "method" | "getter" | "field" | "accessor") {
            let call = if is_private {
                Callable::accessor_key(Callable::PrivatePropGet, key_rc.clone())
            } else {
                Callable::accessor_key(Callable::PropGet, key_rc.clone())
            };
            let g = self.make_callable(call, "get", 1.0);
            access.borrow_mut().props.insert("get", Property::plain(g));
        }
        if matches!(kind, "setter" | "field" | "accessor") {
            let call = if is_private {
                Callable::accessor_key(Callable::PrivatePropSet, key_rc.clone())
            } else {
                Callable::accessor_key(Callable::PropSet, key_rc.clone())
            };
            let s = self.make_callable(call, "set", 2.0);
            access.borrow_mut().props.insert("set", Property::plain(s));
        }
        let add_init = self.make_native("addInitializer", 1, dec_add_initializer);
        {
            let mut b = ctx.borrow_mut();
            b.props.insert("kind", Property::plain(Value::lstr(kind)));
            b.props.insert("name", Property::plain(name));
            b.props
                .insert("static", Property::plain(Value::Bool(is_static)));
            b.props
                .insert("private", Property::plain(Value::Bool(is_private)));
            b.props
                .insert("access", Property::plain(Value::Obj(access)));
            b.props
                .insert("addInitializer", Property::plain(Value::Obj(add_init)));
            b.props
                .insert("metadata", Property::plain(Value::Undefined));
        }
        cv
    }

    /// Apply a list of decorators (innermost/last first) to a callable element (method/getter/setter
    /// or whole class), folding each non-undefined callable return in as the replacement.
    fn decorate_callable(
        &mut self,
        decorators: &[PreparedDecoratorValue],
        mut value: Value,
        kind: &str,
        key: &str,
        is_static: bool,
        is_private: bool,
        inits: &mut crate::rooting::RootedVec,
    ) -> Result<Value, Abrupt> {
        for decorator in decorators.iter().rev() {
            if !decorator.callback.is_callable() {
                return Err(self.throw("TypeError", "decorator is not callable"));
            }
            let ctx = self.make_decorator_context(kind, key, is_static, is_private);
            let r = self.call(decorator.callback, decorator.this_value, &[value, ctx])?;
            inits.append(&mut std::mem::take(&mut self.decorator_initializers));
            match r {
                v if matches!(v.view(), ValueView::Undefined) => {}
                v if v.is_callable() => value = v,
                _ => {
                    return Err(
                        self.throw("TypeError", "decorator must return a function or undefined")
                    );
                }
            }
        }
        Ok(value)
    }

    /// Apply field decorators, returning the initializer transforms they contribute.
    fn decorate_field(
        &mut self,
        decorators: &[PreparedDecoratorValue],
        key: &str,
        is_static: bool,
        is_private: bool,
        inits: &mut crate::rooting::RootedVec,
    ) -> Result<Vec<Value>, Abrupt> {
        // Rooted across the later decorators' calls (I1).
        let mut transforms = crate::rooting::RootedVec::new();
        for decorator in decorators.iter().rev() {
            if !decorator.callback.is_callable() {
                return Err(self.throw("TypeError", "decorator is not callable"));
            }
            let ctx = self.make_decorator_context("field", key, is_static, is_private);
            let r = self.call(
                decorator.callback,
                decorator.this_value,
                &[Value::Undefined, ctx],
            )?;
            inits.append(&mut std::mem::take(&mut self.decorator_initializers));
            match r {
                v if matches!(v.view(), ValueView::Undefined) => {}
                v if v.is_callable() => transforms.push(v),
                _ => {
                    return Err(self.throw(
                        "TypeError",
                        "field decorator must return a function or undefined",
                    ));
                }
            }
        }
        Ok(transforms.into_vec())
    }

    /// Apply accessor decorators, threading the `{get, set}` pair and collecting `init` transforms.
    #[allow(clippy::type_complexity)]
    fn decorate_accessor(
        &mut self,
        decorators: &[PreparedDecoratorValue],
        key: &crate::value::PropertyKey,
        is_static: bool,
        mut get: Value,
        mut set: Value,
        inits: &mut crate::rooting::RootedVec,
    ) -> Result<(Value, Value, Vec<Value>), Abrupt> {
        let is_private = key.is_private();
        // Rooted across the later decorators' calls (I1).
        let mut transforms = crate::rooting::RootedVec::new();
        for decorator in decorators.iter().rev() {
            if !decorator.callback.is_callable() {
                return Err(self.throw("TypeError", "decorator is not callable"));
            }
            let ctx = self.make_decorator_context("accessor", key, is_static, is_private);
            let pair = self.new_object();
            pair.borrow_mut().props.insert("get", Property::plain(get));
            pair.borrow_mut().props.insert("set", Property::plain(set));
            let r = self.call(
                decorator.callback,
                decorator.this_value,
                &[Value::Obj(pair), ctx],
            )?;
            inits.append(&mut std::mem::take(&mut self.decorator_initializers));
            match (r).view() {
                ValueView::Undefined => {}
                ValueView::Obj(_) => {
                    let ng = self.get_member_key(&r, "get")?;
                    if ng.is_callable() {
                        get = ng;
                    }
                    let ns = self.get_member_key(&r, "set")?;
                    if ns.is_callable() {
                        set = ns;
                    }
                    let init = self.get_member_key(&r, "init")?;
                    if init.is_callable() {
                        transforms.push(init);
                    }
                }
                _ => {
                    return Err(self.throw(
                        "TypeError",
                        "accessor decorator must return an object or undefined",
                    ));
                }
            }
        }
        Ok((get, set, transforms.into_vec()))
    }

    /// Run a constructor's body against an already-allocated `this`, used by both `construct` and
    /// `super(...)`. Handles base-class field init, derived classes (their `super()` does the work),
    /// plain function constructors, and native parents (e.g. `extends Error`).
    pub(crate) fn run_constructor_on(
        &mut self,
        ctor: &Value,
        this: &Value,
        args: &[Value],
    ) -> Result<Value, Abrupt> {
        let obj = match &(ctor).view() {
            ValueView::Obj(o) => *o,
            _ => return Err(self.throw("TypeError", "super target is not a constructor")),
        };
        let ptr = Gc::as_ptr(&obj) as usize;
        // A proxy parent constructs through its [[Construct]] trap machinery; the returned object
        // is what super() binds as `this` (the object-override path in the super-call code).
        if self.proxies.contains_key(&ptr) {
            let nt = self.constructor_new_target();
            return self.construct_nt(*ctor, args, nt);
        }
        let class_kind = obj.borrow().class_kind;
        let is_class = class_kind != 0;
        let derived = class_kind == crate::value::CLASS_DERIVED;
        let call = obj.borrow().call.clone();
        match call {
            // A bound-function parent constructs through its target with the bound arguments
            // prepended (BoundFunction [[Construct]]).
            Callable::Bound(bound) => {
                let mut all = bound.args.clone();
                all.extend_from_slice(args);
                self.run_constructor_on(&Value::Obj(bound.target), this, &all)
            }
            Callable::User(user) => {
                // A base class initializes its fields before its body runs; a derived class does so
                // inside its own `super()`.
                if is_class && !derived {
                    self.init_instance_fields(ctor, this)?;
                }
                // A `super(...)` call is legal only directly within a *derived* constructor body.
                let saved_super = self.super_call_ok;
                self.super_call_ok = is_class && derived;
                let r = self.call_user(&user.func, user.env, *this, args, true, &obj);
                self.super_call_ok = saved_super;
                r
            }
            call if call.is_native() => {
                // Native parent (e.g. Error/Map): a super() call is a construct, so set the flag for
                // constructors that require `new`. Run it, then graft its own props onto `this`.
                let saved = self.constructing;
                self.constructing = true;
                let made = self.dispatch_native(&call, *this, args);
                self.constructing = saved;
                let made = made?;
                if let (ValueView::Obj(src), ValueView::Obj(dst)) = (&(made).view(), &(this).view())
                {
                    if !Gc::ptr_eq(src, dst) {
                        for k in src.borrow().props.keys() {
                            let p = src.borrow().props.get_key(k).cloned().unwrap();
                            dst.borrow_mut().props.insert(k, p);
                        }
                        for (key, slot) in src.borrow().internal_slots().iter() {
                            dst.borrow_mut()
                                .internal_slots_mut()
                                .insert(key.clone(), slot.clone());
                        }
                        // A Function (or GeneratorFunction/AsyncFunction) subclass instance is
                        // itself callable: carry the built-in's [[Call]] behavior onto `this`.
                        let src_call = src.borrow().call.clone();
                        if !matches!(src_call, Callable::None)
                            && matches!(dst.borrow().call, Callable::None)
                        {
                            let is_ctor = src.borrow().is_constructor;
                            let mut db = dst.borrow_mut();
                            db.call = src_call;
                            db.is_constructor = is_ctor;
                        }
                        // Move the native object's internal slots (Map/Set/TypedArray/buffer/etc.)
                        // onto `this`, so a subclass instance carries the built-in's state.
                        let (sp, dp) = (Gc::as_ptr(src) as usize, Gc::as_ptr(dst) as usize);
                        self.gc_pin(dst);
                        let mut moved_collection = self.move_collection(src, dst);
                        // A subclass instance inherits the built-in's exotic behavior (e.g. an Array
                        // subclass is itself an Array exotic). The native object is discarded, so
                        // its exotic state moves rather than copies.
                        let src_exotic = std::mem::replace(
                            &mut src.borrow_mut().exotic,
                            crate::value::Exotic::None,
                        );
                        if !matches!(src_exotic, crate::value::Exotic::None) {
                            if matches!(src_exotic, crate::value::Exotic::Array) {
                                dst.borrow().props.mark_array();
                            }
                            dst.borrow_mut().exotic = src_exotic;
                        }
                        // WeakMap/WeakSet's [[WeakMapData]]/[[WeakSetData]] and its acceleration
                        // index are one internal slot implementation and must move together to the
                        // subclass instance created by super() (ECMA-262 §§24.3.1, 24.4.1).
                        moved_collection |= self.move_weak_slots(sp, dp);
                        if moved_collection {
                            // The temporary object returned by the native constructor no longer
                            // owns side-table state; the actual subclass instance is pinned above.
                            self.gc_pins.remove(&sp);
                        }
                        if let Some(v) = self.typed_arrays.remove(&sp) {
                            dst.borrow().ic_plain.set(false);
                            self.typed_arrays.insert(dp, v);
                            dst.borrow_mut().native_typed_array =
                                src.borrow_mut().native_typed_array.take();
                        }
                        // The TypedArray's `buffer` slot lives in a parallel side table keyed by the
                        // view's pointer, so it must move to `this` alongside its TaInfo.
                        if let Some(v) = self.ta_buffer.remove(&sp) {
                            self.ta_buffer.insert(&self.gc_heap, dp, v);
                        }
                        if let Some(v) = self.data_views.remove(&sp) {
                            self.data_views.insert(dp, v);
                        }
                        // An ArrayBuffer's bytes or a SharedArrayBuffer's Shared Data Block.
                        if let Some(v) = self.array_buffers.remove(&sp) {
                            self.array_buffers.insert(dp, v);
                        }
                        if let Some(v) = self.native_buffers.remove(&sp) {
                            self.native_buffers.insert(dp, v);
                        }
                        if let Some(v) = self.array_buffer_versions.remove(&sp) {
                            self.array_buffer_versions.insert(dp, v);
                        }
                        if let Some(v) = self.array_buffer_dirty_ranges.remove(&sp) {
                            self.array_buffer_dirty_ranges.insert(dp, v);
                        }
                        if let Some(v) = self.regexps.remove(&sp) {
                            self.regexps.insert(dp, v);
                        }
                        if let Some(v) = self.promises.remove(&sp) {
                            self.promises.insert(&self.gc_heap, dp, v);
                            // The native ctor's resolvers are bound to `src`; forward them.
                            self.promise_forward.insert(&self.gc_heap, sp, *this);
                        }
                        if let Some(v) = self.temporal.remove(&sp) {
                            self.temporal.insert(dp, v);
                        }
                    }
                }
                Ok(Value::Undefined)
            }
            _ => Err(self.throw("TypeError", "super target is not a constructor")),
        }
    }

    /// CopyDataProperties(rest, ToObject(value), excludedNames): own keys in [[OwnPropertyKeys]]
    /// order (through a proxy's ownKeys/getOwnPropertyDescriptor traps, symbols included), copying
    /// each enumerable non-excluded property.
    pub(crate) fn copy_data_properties(
        &mut self,
        value: &Value,
        excluded: &[String],
    ) -> Result<Gc, Abrupt> {
        let rest = self.new_object();
        self.copy_data_properties_into(&rest, value, excluded)?;
        Ok(rest)
    }

    pub(crate) fn copy_data_properties_into(
        &mut self,
        rest: &Gc,
        value: &Value,
        excluded: &[String],
    ) -> Result<(), Abrupt> {
        let is_excluded = |k: &str| excluded.iter().any(|x| x == k);
        // ECMA-262 CopyDataProperties: a closed ordinary data record has no observable
        // OwnPropertyKeys/GetOwnProperty/Get work. Keep its shared layout and copy the owning
        // packed values directly instead of rebuilding keys, lookups and shape transitions.
        if excluded.is_empty() {
            if let ValueView::Obj(source) = &(value).view() {
                if !Gc::ptr_eq(source, rest) {
                    let source = source.borrow();
                    if source.ic_plain.get() && matches!(source.exotic, Exotic::None) {
                        let mut target = rest.borrow_mut();
                        if target.ic_plain.get()
                            && matches!(target.exotic, Exotic::None)
                            && target.extensible
                            && target.props.try_copy_named_data_from(&source.props)
                        {
                            return Ok(());
                        }
                    }
                }
            }
        }
        if let Some((t, h)) = crate::builtins::proxy_pair(self, value) {
            let keys = crate::builtins::proxy_own_keys(self, &t, &h).map_err(Abrupt::Throw)?;
            for k in keys {
                let pk = self.to_property_key(&k)?;
                if is_excluded(&pk) {
                    continue;
                }
                let enumerable = crate::builtins::proxy_get_own_property(self, &t, &h, &pk)
                    .map_err(Abrupt::Throw)?
                    .is_some_and(|desc| desc.enumerable());
                if enumerable {
                    let v = self.get_member_key(value, &pk)?;
                    rest.borrow_mut()
                        .props
                        .insert(pk.as_str(), crate::value::Property::plain(v));
                }
            }
            return Ok(());
        }
        match &(value).view() {
            ValueView::Obj(src) => {
                // CopyDataProperties snapshots [[OwnPropertyKeys]] before any descriptor getter
                // runs. Indexed platform getters may mutate expandos, but cannot add newly-created
                // keys to this operation's snapshot.
                if self.host_indexed_len(src).is_some() {
                    let keys = crate::builtins::ordinary_own_keys_ordered(self, src)
                        .map_err(Abrupt::Throw)?;
                    for key in keys {
                        if is_excluded(&key) {
                            continue;
                        }
                        let enumerable =
                            self.host_platform_enumerable(src, &key)?
                                .unwrap_or_else(|| {
                                    src.borrow()
                                        .props
                                        .get(&key)
                                        .is_some_and(|property| property.enumerable())
                                });
                        if enumerable {
                            let value = self.get_member_key(value, &key)?;
                            rest.borrow_mut()
                                .props
                                .insert(key, crate::value::Property::plain(value));
                        }
                    }
                    return Ok(());
                }
                let keys = src.borrow().props.ordered_keys();
                for k in keys {
                    if is_excluded(&k.text()) {
                        continue;
                    }
                    let enumerable = src
                        .borrow()
                        .props
                        .get_key(k)
                        .map(|p| p.enumerable())
                        .unwrap_or(false);
                    if enumerable {
                        let v = self.get_member_key(value, &k.text())?;
                        rest.borrow_mut()
                            .props
                            .insert(k, crate::value::Property::plain(v));
                    }
                }
            }
            // ToObject(string): copy the enumerable index properties, one a code unit
            // (StringGetOwnProperty), a surrogate pair's halves included.
            ValueView::Str(s) => {
                for (idx, unit) in s.units().iter().enumerate() {
                    let k = idx.to_string();
                    if is_excluded(&k) {
                        continue;
                    }
                    set_data(rest, &k, Value::Str(crate::jstr::unit_lstr(unit)));
                }
            }
            // Other primitives wrap to an object with no own enumerable properties.
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn init_instance_fields(
        &mut self,
        ctor: &Value,
        this: &Value,
    ) -> Result<(), Abrupt> {
        let obj = match &(ctor).view() {
            ValueView::Obj(o) => *o,
            _ => return Ok(()),
        };
        let ptr = Gc::as_ptr(&obj) as usize;
        let (fields, field_env, initializers, priv_members) = match self.class_info.get(&ptr) {
            // InitializeInstanceElements has nothing to add for a class without elements.
            Some(i) if !i.has_instance_elements() => return Ok(()),
            Some(i) => (
                i.fields.clone(),
                i.field_env,
                i.instance_initializers.clone(),
                i.private_members.clone(),
            ),
            None => return Ok(()),
        };
        // PrivateMethodOrAccessorAdd: stamp the class's private methods/accessors on the instance
        // before any field initializer runs; a second initialization (return-override tricks
        // constructing over the same object twice) is a TypeError.
        if let ValueView::Obj(o) = &(this).view() {
            // InitializeInstanceElements checks extensibility once for the complete
            // private-method/accessor group (Stage 3 nonextensible-applies-to-private).
            if !priv_members.is_empty()
                && !crate::builtins::js_is_extensible(self, this).map_err(Abrupt::Throw)?
            {
                return Err(self.throw(
                    "TypeError",
                    "cannot add private members to a non-extensible object",
                ));
            }
            if let Some((k, _)) = priv_members.first() {
                if o.borrow().internal_slots().contains(k.as_str()) {
                    return Err(self.throw(
                        "TypeError",
                        "cannot initialize private methods of a class twice on the same object",
                    ));
                }
            }
            for (k, p) in &priv_members {
                o.borrow_mut()
                    .internal_slots_mut()
                    .insert(k.as_str(), p.clone());
            }
        }
        // A field initializer is not a constructor: a `super(...)` reached from here (e.g. through a
        // direct `eval`) is illegal, so clear the flag for the duration of the initializers.
        let saved_super = self.super_call_ok;
        let saved_field = self.in_field_init_code;
        self.super_call_ok = false;
        self.in_field_init_code = true;
        let result = (|me: &mut Self| -> Result<(), Abrupt> {
            for field in fields.iter() {
                let key = &field.key;
                let mut v = match &field.init {
                    // An initializer that reads no binding (not even `this`), declares nothing
                    // and creates no closure cannot observe which environment it runs in.
                    Some(e) if env_free_initializer(e) => me.eval(e, &field_env)?,
                    Some(e) => {
                        // Each initializer evaluates in its own environment (it is a method
                        // body in ClassFieldDefinitionEvaluation); one without an initializer
                        // evaluates nothing, so it needs none.
                        let scope = new_scope(Some(field_env));
                        bind_shared_name(&scope, field_init_name(0), *this);
                        // Field-initializer code: a direct eval from here (or from an arrow
                        // created here) may not reference `arguments`.
                        bind_shared_name(&scope, field_init_name(1), Value::Bool(true));
                        let v = me.eval(e, &scope)?;
                        if is_anonymous_fn(e) {
                            let name = me.fn_name_for_key(if key.is_private() {
                                private_display(key)
                            } else {
                                key
                            });
                            me.set_fn_name(&v, &name);
                        }
                        v
                    }
                    None => Value::Undefined,
                };
                // Decorator-supplied field initializers transform the value in turn.
                for t in &field.transforms {
                    v = me.call(*t, *this, &[v])?;
                }
                if key.is_private() {
                    // PrivateFieldAdd: stamped directly on the object (bypassing proxy traps);
                    // the supported proposal first performs IsExtensible, whose Proxy trap
                    // is observable, then checks for a duplicate Private Name.
                    let ValueView::Obj(o) = &(this).view() else {
                        return Err(me.throw("TypeError", "cannot add a private field"));
                    };
                    if !crate::builtins::js_is_extensible(me, this).map_err(Abrupt::Throw)? {
                        return Err(me.throw(
                            "TypeError",
                            "cannot add a private field to a non-extensible object",
                        ));
                    }
                    if o.borrow().internal_slots().contains(key.as_str()) {
                        return Err(me.throw(
                            "TypeError",
                            "cannot initialize the same private field twice",
                        ));
                    }
                    o.borrow_mut().internal_slots_mut().insert(
                        key.as_str(),
                        crate::value::Property::data(v, true, false, false),
                    );
                } else {
                    // DefineField: CreateDataPropertyOrThrow (an own data property, even over a
                    // setter — and a [[DefineOwnProperty]] on a deferred namespace receiver).
                    crate::builtins::cdp_or_throw(me, this, key, v).map_err(Abrupt::Throw)?;
                }
            }
            // Decorator addInitializer callbacks run after the fields, with `this` = the instance.
            for init in &initializers {
                me.call(*init, *this, &[])?;
            }
            Ok(())
        })(self);
        self.super_call_ok = saved_super;
        self.in_field_init_code = saved_field;
        result
    }

    /// Unary operator on an already-evaluated value (bytecode VM fallback path); mirrors
    /// [`eval_unary`] exactly for `- + ~ typeof` (never `delete`/`!`/`void` — the VM handles
    /// those without coercion hooks).
    pub(crate) fn eval_unary_vm(&mut self, op: &str, v: Value) -> Result<Value, Abrupt> {
        if op == "typeof" {
            if self.is_htmldda(&v) {
                return Ok(Value::lstr("undefined"));
            }
            return Ok(Value::lstr(v.type_of()));
        }
        let v = if matches!(op, "-" | "~") && matches!((v).view(), ValueView::Obj(_)) {
            self.to_primitive(&v, Hint::Number)?
        } else {
            v
        };
        if let ValueView::BigInt(n) = (v).view() {
            return match op {
                "-" => Ok(Value::BigInt(n.neg())),
                "~" => Ok(Value::BigInt(n.not())),
                "+" => Err(self.throw("TypeError", "Cannot convert a BigInt value to a number")),
                _ => unreachable!("unary_vm {op}"),
            };
        }
        match op {
            "-" => Ok(Value::Num(-self.to_number(&v)?)),
            "+" => Ok(Value::Num(self.to_number(&v)?)),
            "~" => Ok(Value::Num(!(self.to_int32(&v)?) as f64)),
            _ => unreachable!("unary_vm {op}"),
        }
    }

    /// Assign to a free (non-slot) name from the bytecode VM: exactly the tree-walker's
    /// Reference::Var resolution + PutValue.
    pub(crate) fn assign_free_name(
        &mut self,
        name: &str,
        v: Value,
        env: &Env,
    ) -> Result<(), Abrupt> {
        let target = Expr::Ident(name.to_string(), NO_POSITION);
        let mut r = self.resolve_reference(&target, env)?;
        self.put_reference(&mut r, v)
    }

    /// A plain-data object literal for the bytecode VM (`{ a: 1, b: 2 }` — no accessors,
    /// methods, spreads, computed keys, or `__proto__`; the compiler bails on those).
    pub(crate) fn make_plain_object_vm(
        &mut self,
        keys: &[std::rc::Rc<str>],
        values: Vec<Value>,
    ) -> Value {
        let obj = crate::value::Object::new(Some(self.object_proto));
        {
            let mut b = obj.borrow_mut();
            for (k, v) in keys.iter().zip(values) {
                b.props.insert(k.clone(), crate::value::Property::plain(v));
            }
        }
        Value::Obj(obj)
    }

    /// [`Interp::make_plain_object_templated`] reading the values straight off the JIT operand
    /// stack (moved; the caller forgets them) — no intermediate `Vec` per instantiation.
    ///
    /// # Safety
    /// `base..base+count` must be initialized `Value`s the caller relinquishes.
    pub(crate) unsafe fn make_plain_object_templated_from(
        &mut self,
        tmpl: &std::cell::OnceCell<Box<crate::value::Props>>,
        keys: &[std::rc::Rc<str>],
        base: *const Value,
        count: usize,
    ) -> Value {
        let map = tmpl.get_or_init(|| {
            let mut p = crate::value::Props::new();
            for k in keys {
                p.insert(k.clone(), crate::value::Property::plain(Value::Undefined));
            }
            p.mark_layout_shared();
            // A late fill of the chunk's template, a registered code root.
            crate::value::remember_code_edges(&p);
            Box::new(p)
        });
        let props = map.instantiate_plain((0..count).map(|slot| unsafe { base.add(slot).read() }));
        let obj = crate::value::Object::new_with_parts(
            Some(self.object_proto),
            props,
            crate::value::Exotic::None,
        );
        Value::Obj(obj)
    }

    /// [`Interp::make_plain_object_vm`] through a per-site pre-shaped map (distinct keys only —
    /// the compiler guarantees it): the first execution builds the final `Props` once via the
    /// insert path with placeholder values; later instances share its key layout and move values
    /// into a fresh contiguous field vector — no per-key ownership, hashing, or shape transitions.
    /// Values arrive in key order, one slot per key.
    pub(crate) fn make_plain_object_templated(
        &mut self,
        tmpl: &std::cell::OnceCell<Box<crate::value::Props>>,
        keys: &[std::rc::Rc<str>],
        values: Vec<Value>,
    ) -> Value {
        let map = tmpl.get_or_init(|| {
            let mut p = crate::value::Props::new();
            for k in keys {
                p.insert(k.clone(), crate::value::Property::plain(Value::Undefined));
            }
            p.mark_layout_shared();
            // A late fill of the chunk's template, a registered code root.
            crate::value::remember_code_edges(&p);
            Box::new(p)
        });
        let obj = crate::value::Object::new_with_parts(
            Some(self.object_proto),
            map.instantiate_plain(values.into_iter()),
            crate::value::Exotic::None,
        );
        Value::Obj(obj)
    }

    fn eval_unary(&mut self, op: &str, arg: &Expr, env: &Env) -> Result<Value, Abrupt> {
        if op == "typeof" {
            // typeof on an unresolved identifier yields "undefined" rather than throwing.
            if let Expr::Ident(name, _) = arg {
                return self.typeof_name_vm(name, env);
            }
            let v = self.eval(arg, env)?;
            if self.is_htmldda(&v) {
                return Ok(Value::lstr("undefined"));
            }
            return Ok(Value::lstr(v.type_of()));
        }
        if op == "delete" {
            return self.eval_delete(arg, env);
        }
        let v = self.eval(arg, env)?;
        // `-`/`~` apply ToNumeric: a BigInt (or BigInt wrapper object) stays a BigInt.
        let v = if matches!(op, "-" | "~") && matches!((v).view(), ValueView::Obj(_)) {
            self.to_primitive(&v, Hint::Number)?
        } else {
            v
        };
        if let ValueView::BigInt(n) = (v).view() {
            return match op {
                "!" => Ok(Value::Bool(n.is_zero())),
                "-" => Ok(Value::BigInt(n.neg())),
                "~" => Ok(Value::BigInt(n.not())),
                "void" => Ok(Value::Undefined),
                "+" => Err(self.throw("TypeError", "Cannot convert a BigInt value to a number")),
                _ => unreachable!("unary {op}"),
            };
        }
        match op {
            "!" => Ok(Value::Bool(!self.to_boolean(&v))),
            "-" => Ok(Value::Num(-self.to_number(&v)?)),
            "+" => Ok(Value::Num(self.to_number(&v)?)),
            "~" => Ok(Value::Num(!(self.to_int32(&v)?) as f64)),
            "void" => Ok(Value::Undefined),
            _ => unreachable!("unary {op}"),
        }
    }

    /// Whether `key` names a non-configurable own property of a string primitive ("length" or an
    /// in-range canonical integer index).
    fn string_own_key(&self, s: &crate::lstr::LStr, key: &str) -> bool {
        if key == "length" {
            return true;
        }
        crate::value::canonical_index(key).is_some_and(|i| (i as usize) < s.unit_len())
    }

    fn eval_delete(&mut self, arg: &Expr, env: &Env) -> Result<Value, Abrupt> {
        match arg {
            // `delete a?.b` deletes when the base is non-nullish; a short-circuited chain is
            // simply true.
            Expr::OptionalChain(inner) => {
                let saved = self.short_circuit;
                self.short_circuit = false;
                let r = self.eval_delete(inner, env);
                let short = std::mem::replace(&mut self.short_circuit, saved);
                if short && r.is_ok() {
                    return Ok(Value::Bool(true));
                }
                r
            }
            Expr::Member {
                obj,
                prop,
                optional,
                ..
            } if *optional => {
                let base = self.eval(obj, env)?;
                if self.short_circuit {
                    return Ok(Value::Bool(true));
                }
                if matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    self.short_circuit = true;
                    return Ok(Value::Bool(true));
                }
                self.delete_prop_on(base, prop)
            }
            Expr::Index {
                obj,
                index,
                optional,
                ..
            } if *optional => {
                let base = self.eval(obj, env)?;
                if self.short_circuit {
                    return Ok(Value::Bool(true));
                }
                if matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    self.short_circuit = true;
                    return Ok(Value::Bool(true));
                }
                let idx = self.eval(index, env)?;
                let key = self.to_property_key(&idx)?;
                self.delete_prop_on(base, &key)
            }
            Expr::Member { obj, prop, .. } => {
                // `delete super.x` is a runtime ReferenceError (the reference is a super
                // reference) — after GetThisBinding (TDZ ReferenceError first).
                if matches!(**obj, Expr::Super) {
                    self.get_var("this", env)?;
                    return Err(self.throw("ReferenceError", "cannot delete a super property"));
                }
                let base = self.eval(obj, env)?;
                if self.short_circuit {
                    return Ok(Value::Bool(true));
                }
                if matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    return Err(self.throw(
                        "TypeError",
                        format!("cannot delete property '{prop}' of null or undefined"),
                    ));
                }
                self.delete_prop_on(base, prop)
            }
            Expr::Index { obj, index, .. } => {
                if matches!(**obj, Expr::Super) {
                    // GetThisBinding first (TDZ ReferenceError), then the key expression
                    // evaluates — without ToPropertyKey — before the ReferenceError.
                    self.get_var("this", env)?;
                    self.eval(index, env)?;
                    return Err(self.throw("ReferenceError", "cannot delete a super property"));
                }
                let base = self.eval(obj, env)?;
                // A short-circuited optional base skips the key expression entirely.
                if self.short_circuit {
                    return Ok(Value::Bool(true));
                }
                let idx = self.eval(index, env)?;
                let key = self.to_property_key(&idx)?;
                if matches!((base).view(), ValueView::Undefined | ValueView::Null) {
                    return Err(self.throw(
                        "TypeError",
                        format!(
                            "cannot delete property '{}' of null or undefined",
                            crate::value::decode_string_key(&key)
                        ),
                    ));
                }
                self.delete_prop_on(base, &key)
            }
            // A parenthesized reference still deletes.
            Expr::Paren(inner) => self.eval_delete(inner, env),
            Expr::Ident(name, _) => self.delete_ident(name, env),
            // Any other operand evaluates (for its side effects) and deletes to `true`.
            other => {
                self.eval(other, env)?;
                Ok(Value::Bool(true))
            }
        }
    }

    /// [[Delete]] on an evaluated base (shared by the plain and optional `delete` paths).
    fn delete_prop_on(&mut self, base: Value, prop: &str) -> Result<Value, Abrupt> {
        let strict = self.strict;
        self.delete_prop_with(base, prop, strict)
    }

    /// [[Delete]] with the caller's strictness: the VM's delete ops carry the compiled
    /// function's flag (a direct JIT→JIT call sequence does not maintain `self.strict`).
    pub(crate) fn delete_prop_with(
        &mut self,
        base: Value,
        prop: &str,
        strict: bool,
    ) -> Result<Value, Abrupt> {
        {
            {
                if let ValueView::Obj(o) = &(base).view() {
                    self.defer_trigger(o, Some(prop))?;
                    let ptr = Gc::as_ptr(o) as usize;
                    if let Some((target, handler)) = self.proxy_at(ptr) {
                        let ok = self.proxy_delete(target, handler, prop)?;
                        if !ok && strict {
                            return Err(self.throw(
                                "TypeError",
                                format!(
                                    "cannot delete property '{}'",
                                    crate::value::decode_string_key(prop)
                                ),
                            ));
                        }
                        return Ok(Value::Bool(ok));
                    }
                    // A TypedArray integer index can't be deleted ([[Delete]] → false; strict
                    // throws); a canonical-numeric non-index reports success.
                    if let Some(info) = self.typed_arrays.get(&ptr).copied() {
                        match self.ta_index_kind(&info, prop) {
                            TaIndex::Element(_) => {
                                if strict {
                                    return Err(
                                        self.throw("TypeError", "cannot delete a TypedArray index")
                                    );
                                }
                                return Ok(Value::Bool(false));
                            }
                            TaIndex::Exotic => return Ok(Value::Bool(true)),
                            TaIndex::Ordinary => {}
                        }
                    }
                    // Web IDL legacy platform object [[Delete]]: supported indices are
                    // undeletable; an unsupported array-index key succeeds immediately.
                    if self.host_indexed_array_key(o, prop) {
                        let ok = crate::value::canonical_index(prop)
                            .is_some_and(|index| index >= self.host_indexed_len(o).unwrap_or(0));
                        if !ok && strict {
                            return Err(self.throw(
                                "TypeError",
                                format!(
                                    "cannot delete indexed property '{}'",
                                    crate::value::decode_string_key(prop)
                                ),
                            ));
                        }
                        return Ok(Value::Bool(ok));
                    }
                    if self.host_named_visible(o, prop)? {
                        if strict {
                            return Err(self.throw("TypeError", "cannot delete named property"));
                        }
                        return Ok(Value::Bool(false));
                    }
                    let configurable = o
                        .borrow()
                        .props
                        .get(prop)
                        .map(|p| p.configurable())
                        .unwrap_or(true);
                    if configurable {
                        Self::unmap_argument(o, prop);
                        o.borrow_mut().props.remove(prop);
                        return Ok(Value::Bool(true));
                    }
                    // [[Delete]] returned false: strict-mode `delete` throws.
                    if strict {
                        return Err(self.throw(
                            "TypeError",
                            format!(
                                "cannot delete non-configurable property '{}'",
                                crate::value::decode_string_key(prop)
                            ),
                        ));
                    }
                    return Ok(Value::Bool(false));
                }
                // A string primitive's `length` and in-range indices are non-configurable own
                // properties of the string exotic wrapper: [[Delete]] is false (TypeError strict).
                if let ValueView::Str(sv) = &(base).view() {
                    if self.string_own_key(sv, prop) {
                        if strict {
                            return Err(self.throw(
                                "TypeError",
                                format!(
                                    "cannot delete non-configurable property '{}'",
                                    crate::value::decode_string_key(prop)
                                ),
                            ));
                        }
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            }
        }
    }

    /// `delete <identifier>`: an environment binding is removable only if it is `deletable`
    /// (a `var`/function created by a sloppy `eval`); a global-object property follows its own
    /// configurability. An unresolvable reference deletes to `true`.
    pub(crate) fn delete_ident(&mut self, name: &str, env: &Env) -> Result<Value, Abrupt> {
        {
            {
                let mut cur = Some(*env);
                while let Some(s) = cur {
                    let (has, deletable, with_obj, parent) = {
                        let b = s.borrow();
                        let binding = b.vars.get(name);
                        (
                            binding.is_some(),
                            binding.is_some_and(|x| x.deletable()),
                            b.with_object(),
                            b.parent,
                        )
                    };
                    if has {
                        if deletable {
                            s.borrow_mut().vars.remove(name);
                            return Ok(Value::Bool(true));
                        }
                        return Ok(Value::Bool(false));
                    }
                    if let Some((wobj, o)) = with_obj
                        .as_ref()
                        .and_then(|wobj| wobj.as_obj().map(|o| (wobj, o)))
                    {
                        // @@unscopables-blocked names skip this scope entirely; a present
                        // binding deletes via the object's [[Delete]] (own property only —
                        // an inherited one is untouched and reports true per OrdinaryDelete).
                        let (wobj, o) = (*wobj, o);
                        if self.with_has_binding(&wobj, name)? {
                            if self.host_named_visible(&o, name)? {
                                return Ok(Value::Bool(false));
                            }
                            let configurable = o
                                .borrow()
                                .props
                                .get(name)
                                .map(|p| p.configurable())
                                .unwrap_or(true);
                            if configurable {
                                o.borrow_mut().props.remove(name);
                                return Ok(Value::Bool(true));
                            }
                            return Ok(Value::Bool(false));
                        }
                    }
                    cur = parent;
                }
                // Fall back to a global-object property.
                let g = self.global;
                let existing = g.borrow().props.get(name).map(|p| p.configurable());
                match existing {
                    Some(true) => {
                        g.borrow_mut().props.remove(name);
                        Ok(Value::Bool(true))
                    }
                    Some(false) => Ok(Value::Bool(false)),
                    None => Ok(Value::Bool(true)),
                }
            }
        }
    }

    /// Proxy `[[Delete]]`: call the deleteProperty trap, or forward to the target.
    pub(crate) fn proxy_delete(
        &mut self,
        target: Value,
        handler: Value,
        key: &str,
    ) -> Result<bool, Abrupt> {
        if matches!((handler).view(), ValueView::Null) {
            return Err(self.throw("TypeError", "cannot delete on a revoked proxy"));
        }
        let trap = self.proxy_get_trap(
            &handler,
            crate::interpreter::PROXY_TRAP_DELETE,
            "deleteProperty",
        )?;
        let Some(trap) = trap else {
            // ECMA-262 §10.5.10 step 7: target.[[Delete]](P), with the target's own internal
            // method (a Proxy, TypedArray, mapped arguments or host object has its own).
            let deleted = self.delete_prop_with(target, key, false)?;
            return Ok(matches!((deleted).view(), ValueView::Bool(true)));
        };
        let kv = self.key_value(key);
        let res = self.proxy_call_trap(
            crate::interpreter::PROXY_TRAP_DELETE,
            &trap,
            handler,
            [target, kv],
        )?;
        if !self.to_boolean(&res) {
            return Ok(false);
        }
        // ECMA-262 §10.5.10 steps 10-15: a non-configurable property, or any property of a
        // non-extensible target, can't be reported as deleted. Both queries use the target's
        // own internal methods.
        if let ValueView::Obj(t) = &(target).view() {
            let target_desc =
                crate::builtins::object_get_own_property(self, t, key).map_err(Abrupt::Throw)?;
            if let Some(target_desc) = target_desc {
                if !target_desc.configurable() {
                    return Err(self.throw(
                        "TypeError",
                        "proxy 'deleteProperty' removed a non-configurable property",
                    ));
                }
                if !crate::builtins::js_is_extensible(self, &target).map_err(Abrupt::Throw)? {
                    return Err(self.throw(
                        "TypeError",
                        "proxy 'deleteProperty' removed a non-extensible target's property",
                    ));
                }
            }
        }
        Ok(true)
    }

    /// The scope holding an ordinary, initialized, mutable, non-import binding for `name` —
    /// walking plain scopes only. `None` (any `with` frame, TDZ, const, import, or not found)
    /// sends the caller down the full Reference path.
    fn plain_binding_scope(&self, name: &str, env: &Env) -> Option<Env> {
        let mut cur = Some(*env);
        while let Some(s) = cur {
            let (found, parent) = {
                let b = s.borrow();
                if b.has_with_object() {
                    return None;
                }
                match b.vars.get(name) {
                    Some(bd) => {
                        if !bd.initialized() || bd.is_import() || !bd.mutable() {
                            return None;
                        }
                        (true, None)
                    }
                    None => (false, b.parent),
                }
            };
            if found {
                return Some(s);
            }
            cur = parent;
        }
        None
    }

    fn eval_update(
        &mut self,
        op: &str,
        prefix: bool,
        arg: &Expr,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        // Numeric update of an ordinary binding: read and write it in place (one scope walk, no
        // Reference allocation) — the ubiquitous `i++` of every loop.
        if let Expr::Ident(name, _) = arg {
            if let Some(scope) = self.plain_binding_scope(name, env) {
                let old = scope
                    .borrow()
                    .vars
                    .get(name.as_str())
                    .map(|bd| bd.value())
                    .unwrap_or(Value::Undefined);
                if let ValueView::Num(n) = (old).view() {
                    let new = if op == "++" { n + 1.0 } else { n - 1.0 };
                    if let Some(mut bd) = scope.borrow_mut().vars.get_mut(name) {
                        bd.set_value(Value::Num(new));
                        return Ok(Value::Num(if prefix { new } else { n }));
                    }
                }
                // Non-number (BigInt, coercible object) or vanished binding: generic path.
            }
        }
        // Resolve the reference once, then GetValue/PutValue through it (spec Reference semantics).
        let mut lref = self.resolve_reference(arg, env)?;
        let old = self.get_reference(&mut lref)?;
        let old_numeric = self.to_numeric(old)?;
        match (old_numeric).view() {
            ValueView::BigInt(old) => {
                let one = crate::bigint::JsBigInt::from_u64(1);
                let new = if op == "++" {
                    old.add(&one)
                } else {
                    old.sub(&one)
                };
                self.put_reference(&mut lref, Value::BigInt(new))?;
                Ok(Value::BigInt(if prefix { new } else { old }))
            }
            ValueView::Num(old) => {
                let new = if op == "++" { old + 1.0 } else { old - 1.0 };
                self.put_reference(&mut lref, Value::Num(new))?;
                Ok(Value::Num(if prefix { new } else { old }))
            }
            _ => unreachable!("ToNumeric returns only Number or BigInt"),
        }
    }

    /// An assignment whose operator is at source `position`.
    fn eval_assign(
        &mut self,
        op: &str,
        target: &Expr,
        value: &Expr,
        position: u32,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        self.assign(op, target, value, env)
            .map_err(|error| self.placed_error(error, position))
    }

    fn assign(
        &mut self,
        op: &str,
        target: &Expr,
        value: &Expr,
        env: &Env,
    ) -> Result<Value, Abrupt> {
        // Compound assignment to an ordinary binding: GetValue, RHS, op, PutValue — all against
        // the one scope found up front. Logical forms short-circuit and are excluded. If the RHS
        // deleted the binding (direct eval tricks), PutValue falls back to the Reference path.
        if op != "=" && !matches!(op, "&&=" | "||=" | "??=") {
            if let Expr::Ident(name, _) = target {
                if let Some(scope) = self.plain_binding_scope(name, env) {
                    let strict = self.strict;
                    let old = scope
                        .borrow()
                        .vars
                        .get(name.as_str())
                        .map(|bd| bd.value())
                        .unwrap_or(Value::Undefined);
                    let rhs = self.eval(value, env)?;
                    let result = self.binary(&op[..op.len() - 1], old, rhs)?;
                    let stored = {
                        let mut record = scope.borrow_mut();
                        match record.vars.get_mut(name) {
                            Some(mut bd) if bd.mutable() && bd.initialized() && !bd.is_import() => {
                                bd.set_value(result);
                                true
                            }
                            _ => false,
                        }
                    };
                    if !stored {
                        let mut reference =
                            PreparedReference::scope(Rc::from(name.as_str()), scope, strict);
                        self.write_prepared_reference(&mut reference, result)?;
                    }
                    return Ok(result);
                }
            }
        }
        if op == "=" {
            // A destructuring target evaluates the RHS first, then iterates the pattern.
            if matches!(target, Expr::Array(_) | Expr::Object(_)) {
                let v = self.eval(value, env)?;
                self.assign_to_target(target, v, env)?;
                return Ok(v);
            }
            // A simple target evaluates its Reference (base + computed key expression) BEFORE the
            // RHS; ToPropertyKey and the base's RequireObjectCoercible are deferred to PutValue.
            let mut lref = self.resolve_reference(target, env)?;
            if let Expr::Ident(n, _) = target {
                if matches!(value, Expr::Class(c) if c.name.is_none()) {
                    self.pending_fn_name = Some(n.clone());
                }
            }
            let v = self.eval(value, env)?;
            self.pending_fn_name = None;
            // `f = function(){}` names the anonymous function after the target identifier.
            if let Expr::Ident(n, _) = target {
                if is_anonymous_fn(value) {
                    self.set_fn_name(&v, n);
                }
            }
            self.put_reference(&mut lref, v)?;
            return Ok(v);
        }
        // The LHS reference is resolved once and reused for GetValue + PutValue, so a `with`-object
        // getter that mutates the binding (or a member base with side effects) is evaluated once.
        let mut lref = self.resolve_reference(target, env)?;
        // Logical assignment (&&=, ||=, ??=) short-circuits.
        if matches!(op, "&&=" | "||=" | "??=") {
            let cur = self.get_reference(&mut lref)?;
            let do_assign = match op {
                "&&=" => self.to_boolean(&cur),
                "||=" => !self.to_boolean(&cur),
                "??=" => matches!((cur).view(), ValueView::Undefined | ValueView::Null),
                _ => unreachable!(),
            };
            if !do_assign {
                return Ok(cur);
            }
            let v = self.eval(value, env)?;
            // `x ||= function(){}` names the anonymous function after an identifier target.
            if let Expr::Ident(n, _) = target {
                if is_anonymous_fn(value) {
                    self.set_fn_name(&v, n);
                }
            }
            self.put_reference(&mut lref, v)?;
            return Ok(v);
        }
        // Compound arithmetic/bitwise: a op= b  ≡  a = a <op> b.
        let cur = self.get_reference(&mut lref)?;
        let rhs = self.eval(value, env)?;
        let bin_op = &op[..op.len() - 1];
        let result = self.binary(bin_op, cur, rhs)?;
        self.put_reference(&mut lref, result)?;
        Ok(result)
    }

    pub(crate) fn assign_to_target(
        &mut self,
        target: &Expr,
        value: Value,
        env: &Env,
    ) -> Result<(), Abrupt> {
        if crate::interpreter::ast_stack_low() {
            return crate::interpreter::with_ast_stack(|| {
                self.assign_to_target(target, value, env)
            });
        }
        match target {
            Expr::Ident(name, _) => self.assign_var(name, value, env),
            Expr::Member { obj, prop, .. } => {
                // `super.x = v` resolves the super base for invariant checks but writes through the
                // `this` receiver (CreateDataProperty on the actual object), per [[Set]] semantics.
                if matches!(**obj, Expr::Super) {
                    self.super_base(env)?;
                    let this = self.get_var("this", env)?;
                    return self.set_member_key(&this, prop, value);
                }
                let base = self.eval(obj, env)?;
                if prop.starts_with('#') {
                    let k = self.resolve_private(prop, env);
                    return self.set_private_member(&base, &k, value);
                }
                self.set_member_key(&base, prop, value)
            }
            Expr::Index { obj, index, .. } => {
                if matches!(**obj, Expr::Super) {
                    self.super_base(env)?;
                    let idx = self.eval(index, env)?;
                    let key = self.to_property_key(&idx)?;
                    let this = self.get_var("this", env)?;
                    return self.set_member_key(&this, &key, value);
                }
                let base = self.eval(obj, env)?;
                let idx = self.eval(index, env)?;
                let key = self.to_property_key(&idx)?;
                self.set_member_key(&base, &key, value)
            }
            // Destructuring assignment: an array/object literal reinterpreted as a target.
            Expr::Array(elems) => {
                let (iter, next) = self.get_iterator(&value)?;
                let iter_close = iter;
                let mut done = false;
                let result = (|me: &mut Self| -> Result<(), Abrupt> {
                    // A throw from `next` (IteratorStep) marks the record done, so IteratorClose
                    // is skipped; a throw from a target assignment leaves it not-done (still closes).
                    macro_rules! step {
                        () => {
                            match me.iterator_step(&iter, &next) {
                                Ok(v) => v,
                                Err(e) => {
                                    done = true;
                                    return Err(e);
                                }
                            }
                        };
                    }
                    // One iterator element (undefined past the end).
                    macro_rules! next_value {
                        () => {
                            if done {
                                Value::Undefined
                            } else {
                                match step!() {
                                    Some(x) => x,
                                    None => {
                                        done = true;
                                        Value::Undefined
                                    }
                                }
                            }
                        };
                    }
                    for el in elems {
                        match el {
                            ArrayElem::Hole => {
                                if !done && step!().is_none() {
                                    done = true;
                                }
                            }
                            ArrayElem::Spread(t) => {
                                // A non-literal rest target's Reference is evaluated BEFORE the
                                // iterator is drained (AssignmentRestElement step 1).
                                if matches!(t, Expr::Array(_) | Expr::Object(_)) {
                                    let rest = me.drain_iterator_rest(&iter, &next, &mut done)?;
                                    let arr = me.make_array(rest);
                                    me.assign_to_target(t, arr, env)?;
                                } else {
                                    let mut lref = me.resolve_reference(t, env)?;
                                    let rest = me.drain_iterator_rest(&iter, &next, &mut done)?;
                                    let arr = me.make_array(rest);
                                    me.put_reference(&mut lref, arr)?;
                                }
                            }
                            ArrayElem::Item(t) => {
                                // Split an optional `= default`; a non-literal target's Reference is
                                // evaluated BEFORE the iterator step (AssignmentElement step 1).
                                let (core, default) = match t {
                                    Expr::Assign {
                                        op: "=",
                                        target,
                                        value,
                                        ..
                                    } => (&**target, Some(&**value)),
                                    _ => (t, None),
                                };
                                if matches!(core, Expr::Array(_) | Expr::Object(_)) {
                                    let mut v = next_value!();
                                    if matches!((v).view(), ValueView::Undefined) {
                                        if let Some(d) = default {
                                            v = me.eval(d, env)?;
                                        }
                                    }
                                    me.assign_to_target(core, v, env)?;
                                } else {
                                    let mut lref = me.resolve_reference(core, env)?;
                                    let mut v = next_value!();
                                    if matches!((v).view(), ValueView::Undefined) {
                                        if let Some(d) = default {
                                            v = me.eval(d, env)?;
                                            if let (Expr::Ident(n, _), true) =
                                                (core, is_anonymous_fn(d))
                                            {
                                                me.set_fn_name(&v, n);
                                            }
                                        }
                                    }
                                    me.put_reference(&mut lref, v)?;
                                }
                            }
                        }
                    }
                    Ok(())
                })(self);
                // IteratorClose: on normal completion propagate its abrupt (a throwing/non-object
                // `return`); on abrupt completion close but keep the original error.
                match result {
                    Ok(()) => {
                        if !done {
                            self.iterator_close_normal(&iter_close)?;
                        }
                        Ok(())
                    }
                    Err(e) => {
                        if matches!(e, Abrupt::Interrupt(_)) {
                            return Err(e);
                        }
                        if !done {
                            if matches!(e, Abrupt::Throw(_)) {
                                self.iterator_close(&iter_close);
                            } else {
                                // A non-throw completion (return/break/continue): a throwing or
                                // non-object `return` replaces it; otherwise it propagates.
                                self.iterator_close_normal(&iter_close)?;
                            }
                        }
                        Err(e)
                    }
                }
            }
            Expr::Object(props) => {
                if matches!((value).view(), ValueView::Undefined | ValueView::Null) {
                    return Err(self.throw("TypeError", "cannot destructure null or undefined"));
                }
                let mut taken: Vec<String> = Vec::new();
                for prop in props {
                    match prop {
                        PropDef::KeyValue { .. } | PropDef::Cover { .. } | PropDef::Proto(_) => {
                            // KeyedDestructuringAssignmentEvaluation: evaluate the property name,
                            // then the target Reference, THEN GetV(value, name) — for a non-literal
                            // target the reference is evaluated before the source is read. As a
                            // pattern, `__proto__: t` is a normal keyed target (not a proto-setter).
                            let (k, t) = match prop {
                                PropDef::KeyValue { key, value }
                                | PropDef::Cover { key, value } => {
                                    (self.propkey_to_string(key, env)?, value)
                                }
                                PropDef::Proto(value) => ("__proto__".to_string(), value),
                                _ => unreachable!(),
                            };
                            taken.push(k.clone());
                            let (core, default) = match t {
                                Expr::Assign {
                                    op: "=",
                                    target,
                                    value,
                                    ..
                                } => (&**target, Some(&**value)),
                                _ => (t, None),
                            };
                            if matches!(core, Expr::Array(_) | Expr::Object(_)) {
                                let mut v = self.get_member_key(&value, &k)?;
                                if matches!((v).view(), ValueView::Undefined) {
                                    if let Some(d) = default {
                                        v = self.eval(d, env)?;
                                    }
                                }
                                self.assign_to_target(core, v, env)?;
                            } else {
                                let mut lref = self.resolve_reference(core, env)?;
                                let mut v = self.get_member_key(&value, &k)?;
                                if matches!((v).view(), ValueView::Undefined) {
                                    if let Some(d) = default {
                                        v = self.eval(d, env)?;
                                        if let (Expr::Ident(n, _), true) =
                                            (core, is_anonymous_fn(d))
                                        {
                                            self.set_fn_name(&v, n);
                                        }
                                    }
                                }
                                self.put_reference(&mut lref, v)?;
                            }
                        }
                        PropDef::Spread(t) => {
                            // CopyDataProperties(rest, ToObject(value), excludedNames = taken).
                            let rest = self.copy_data_properties(&value, &taken)?;
                            self.assign_to_target(t, Value::Obj(rest), env)?;
                        }
                        _ => return Err(self.throw("SyntaxError", "invalid destructuring target")),
                    }
                }
                Ok(())
            }
            // Annex B web compat: assigning to a CallExpression evaluates the call, then throws.
            Expr::Call { .. } => {
                self.eval(target, env)?;
                Err(self.throw("ReferenceError", "invalid assignment target"))
            }
            _ => Err(self.throw("ReferenceError", "invalid assignment target")),
        }
    }

    fn propkey_to_string(&mut self, key: &PropKey, env: &Env) -> Result<String, Abrupt> {
        Ok(match key {
            PropKey::Ident(s) => crate::value::encode_owned_string_key(s.clone()),
            PropKey::Str(s) => crate::value::encode_owned_string_key(s.to_string()),
            PropKey::Num(n) => self.num_to_str(*n),
            PropKey::Computed(e) => {
                let kv = self.eval(e, env)?;
                self.to_property_key(&kv)?.into_string()
            }
        })
    }

    // ----- operators --------------------------------------------------------------------------

    pub(crate) fn binary(&mut self, op: &str, l: Value, r: Value) -> Result<Value, Abrupt> {
        // Hot path: number ⊕ number has no observable coercion steps (no ToPrimitive hooks, no
        // BigInt mixing) — compute directly. Rust f64 comparisons share JS NaN semantics (all
        // false), and +0 == -0 holds, so the comparison arms are exact.
        if let (ValueView::Num(a), ValueView::Num(b)) = (&(l).view(), &(r).view()) {
            let (a, b) = (*a, *b);
            match op {
                "+" => return Ok(Value::Num(a + b)),
                "-" => return Ok(Value::Num(a - b)),
                "*" => return Ok(Value::Num(a * b)),
                "/" => return Ok(Value::Num(a / b)),
                "%" => return Ok(Value::Num(js_mod(a, b))),
                "**" => {
                    return Ok(Value::Num(
                        if b.is_nan() || (a.abs() == 1.0 && b.is_infinite()) {
                            f64::NAN
                        } else {
                            a.powf(b)
                        },
                    ));
                }
                "&" => return Ok(Value::Num((to_int32(a) & to_int32(b)) as f64)),
                "|" => return Ok(Value::Num((to_int32(a) | to_int32(b)) as f64)),
                "^" => return Ok(Value::Num((to_int32(a) ^ to_int32(b)) as f64)),
                "<<" => {
                    return Ok(Value::Num(
                        to_int32(a).wrapping_shl(to_int32(b) as u32 & 31) as f64,
                    ));
                }
                ">>" => {
                    return Ok(Value::Num(
                        (to_int32(a) >> (to_int32(b) as u32 & 31)) as f64,
                    ));
                }
                ">>>" => {
                    return Ok(Value::Num(
                        ((to_int32(a) as u32) >> (to_int32(b) as u32 & 31)) as f64,
                    ));
                }
                "<" => return Ok(Value::Bool(a < b)),
                ">" => return Ok(Value::Bool(a > b)),
                "<=" => return Ok(Value::Bool(a <= b)),
                ">=" => return Ok(Value::Bool(a >= b)),
                "==" | "===" => return Ok(Value::Bool(a == b)),
                "!=" | "!==" => return Ok(Value::Bool(a != b)),
                _ => {}
            }
        }
        // Primitive strings cannot invoke conversion hooks. Append to the left operand's buffer
        // at its frontier where it can, just like native string addition.
        if op == "+"
            && matches!(
                (&(l).view(), &(r).view()),
                (ValueView::Str(_), ValueView::Str(_))
            )
        {
            let (ValueView::Str(left), ValueView::Str(right)) = ((l).view(), (r).view()) else {
                unreachable!()
            };
            if left.unit_len().saturating_add(right.unit_len()) > MAX_STR_LEN {
                return Err(self.throw("RangeError", "Invalid string length"));
            }
            return Ok(Value::Str(left.concat_owned(&right)));
        }
        // Arithmetic, bitwise, and `+` convert both operands to primitives first (left then right),
        // then dispatch on BigInt / string (for `+`) / number. ToPrimitive runs before the BigInt
        // mixing check so a wrapped BigInt object coerces correctly.
        if matches!(
            op,
            "+" | "-" | "*" | "/" | "%" | "**" | "&" | "|" | "^" | "<<" | ">>" | ">>>"
        ) {
            // `+` primes both operands with ToPrimitive first (string concatenation dispatch);
            // every other operator applies ToNumeric to the left operand *completely* before
            // touching the right one.
            if op != "+" {
                let lp = self.to_primitive(&l, Hint::Number)?;
                let ln = if matches!((lp).view(), ValueView::BigInt(_)) {
                    lp
                } else {
                    Value::Num(self.to_number(&lp)?)
                };
                let rp = self.to_primitive(&r, Hint::Number)?;
                let rn = if matches!((rp).view(), ValueView::BigInt(_)) {
                    rp
                } else {
                    Value::Num(self.to_number(&rp)?)
                };
                if matches!((ln).view(), ValueView::BigInt(_))
                    || matches!((rn).view(), ValueView::BigInt(_))
                {
                    if let (ValueView::BigInt(x), ValueView::BigInt(y)) =
                        (&(ln).view(), &(rn).view())
                    {
                        return self.bigint_binop(op, x, y);
                    }
                    return Err(self.throw(
                        "TypeError",
                        "Cannot mix BigInt and other types, use explicit conversions",
                    ));
                }
                let (a, b) = match (&(ln).view(), &(rn).view()) {
                    (ValueView::Num(a), ValueView::Num(b)) => (*a, *b),
                    _ => unreachable!(),
                };
                return match op {
                    "-" => Ok(Value::Num(a - b)),
                    "*" => Ok(Value::Num(a * b)),
                    "/" => Ok(Value::Num(a / b)),
                    "%" => Ok(Value::Num(js_mod(a, b))),
                    // Number::exponentiate: a NaN exponent is NaN even for base 1, and |base| 1
                    // with an infinite exponent is NaN (powf disagrees on both).
                    "**" => Ok(Value::Num(
                        if b.is_nan() || (a.abs() == 1.0 && b.is_infinite()) {
                            f64::NAN
                        } else {
                            a.powf(b)
                        },
                    )),
                    "&" => Ok(Value::Num(
                        (self.to_int32(&ln)? & self.to_int32(&rn)?) as f64,
                    )),
                    "|" => Ok(Value::Num(
                        (self.to_int32(&ln)? | self.to_int32(&rn)?) as f64,
                    )),
                    "^" => Ok(Value::Num(
                        (self.to_int32(&ln)? ^ self.to_int32(&rn)?) as f64,
                    )),
                    "<<" => Ok(Value::Num(
                        (self.to_int32(&ln)?.wrapping_shl(self.to_uint32(&rn)? & 31)) as f64,
                    )),
                    ">>" => Ok(Value::Num(
                        (self.to_int32(&ln)? >> (self.to_uint32(&rn)? & 31)) as f64,
                    )),
                    ">>>" => Ok(Value::Num(
                        (self.to_uint32(&ln)? >> (self.to_uint32(&rn)? & 31)) as f64,
                    )),
                    _ => unreachable!(),
                };
            }
            let hint = Hint::Default;
            let lp = self.to_primitive(&l, hint)?;
            let rp = self.to_primitive(&r, hint)?;
            if op == "+"
                && (matches!((lp).view(), ValueView::Str(_))
                    || matches!((rp).view(), ValueView::Str(_)))
            {
                let ls = self.to_string(&lp)?;
                let rs = self.to_string(&rp)?;
                if ls.unit_len() + rs.unit_len() > MAX_STR_LEN {
                    return Err(self.throw("RangeError", "Invalid string length"));
                }
                return Ok(Value::Str(ls.concat_owned(&rs)));
            }
            if matches!((lp).view(), ValueView::BigInt(_))
                || matches!((rp).view(), ValueView::BigInt(_))
            {
                if let (ValueView::BigInt(x), ValueView::BigInt(y)) = (&(lp).view(), &(rp).view()) {
                    return self.bigint_binop(op, x, y);
                }
                return Err(self.throw(
                    "TypeError",
                    "Cannot mix BigInt and other types, use explicit conversions",
                ));
            }
            return match op {
                "+" => Ok(Value::Num(self.to_number(&lp)? + self.to_number(&rp)?)),
                "-" => Ok(Value::Num(self.to_number(&lp)? - self.to_number(&rp)?)),
                "*" => Ok(Value::Num(self.to_number(&lp)? * self.to_number(&rp)?)),
                "/" => Ok(Value::Num(self.to_number(&lp)? / self.to_number(&rp)?)),
                "%" => {
                    let a = self.to_number(&lp)?;
                    let b = self.to_number(&rp)?;
                    Ok(Value::Num(js_mod(a, b)))
                }
                "**" => Ok(Value::Num(self.to_number(&lp)?.powf(self.to_number(&rp)?))),
                "&" => Ok(Value::Num(
                    (self.to_int32(&lp)? & self.to_int32(&rp)?) as f64,
                )),
                "|" => Ok(Value::Num(
                    (self.to_int32(&lp)? | self.to_int32(&rp)?) as f64,
                )),
                "^" => Ok(Value::Num(
                    (self.to_int32(&lp)? ^ self.to_int32(&rp)?) as f64,
                )),
                "<<" => {
                    let a = self.to_int32(&lp)?;
                    let b = (self.to_uint32(&rp)?) & 31;
                    Ok(Value::Num((a.wrapping_shl(b)) as f64))
                }
                ">>" => {
                    let a = self.to_int32(&lp)?;
                    let b = (self.to_uint32(&rp)?) & 31;
                    Ok(Value::Num((a >> b) as f64))
                }
                _ => unreachable!(),
            };
        }
        match op {
            "==" => Ok(Value::Bool(self.loose_equals(&l, &r)?)),
            "!=" => Ok(Value::Bool(!self.loose_equals(&l, &r)?)),
            "===" => Ok(Value::Bool(self.strict_equals(&l, &r))),
            "!==" => Ok(Value::Bool(!self.strict_equals(&l, &r))),
            "<" | ">" | "<=" | ">=" => self.compare(op, l, r),
            "&" => Ok(Value::Num((self.to_int32(&l)? & self.to_int32(&r)?) as f64)),
            "|" => Ok(Value::Num((self.to_int32(&l)? | self.to_int32(&r)?) as f64)),
            "^" => Ok(Value::Num((self.to_int32(&l)? ^ self.to_int32(&r)?) as f64)),
            "<<" => {
                let a = self.to_int32(&l)?;
                let b = (self.to_uint32(&r)?) & 31;
                Ok(Value::Num((a.wrapping_shl(b)) as f64))
            }
            ">>" => {
                let a = self.to_int32(&l)?;
                let b = (self.to_uint32(&r)?) & 31;
                Ok(Value::Num((a >> b) as f64))
            }
            ">>>" => unreachable!("handled by the numeric block"),
            "instanceof" => self.instanceof(&l, &r),
            "in" => Ok(Value::Bool(self.in_operator(&l, &r)?)),
            _ => unreachable!("binary {op}"),
        }
    }

    fn bigint_binop(
        &self,
        op: &str,
        x: &crate::bigint::JsBigInt,
        y: &crate::bigint::JsBigInt,
    ) -> Result<Value, Abrupt> {
        use crate::bigint::JsBigInt;
        let mut poll = crate::bigint::Poll::new(&self.runtime_interrupt);
        let stop = |stop| self.bigint_stop(stop);
        let v = match op {
            "+" => x.add_bounded(y).map_err(stop)?,
            "-" => x.sub_bounded(y).map_err(stop)?,
            "*" => x.mul_bounded(y, &mut poll).map_err(stop)?,
            "/" | "%" if y.is_zero() => {
                return Err(self.throw("RangeError", "Division by zero"));
            }
            "/" => x.div_bounded(y, &mut poll).map_err(stop)?,
            "%" => x.rem_bounded(y, &mut poll).map_err(stop)?,
            "**" if y.is_negative() => {
                return Err(self.throw("RangeError", "Exponent must be non-negative"));
            }
            "**" => x.pow_bounded(y, &mut poll).map_err(stop)?,
            "&" => x.bitand(y),
            "|" => x.bitor(y),
            "^" => x.bitxor(y),
            // A negative shift count shifts the other way; the rightward shift is arithmetic
            // (flooring toward negative infinity).
            "<<" | ">>" => {
                let leftward = (op == "<<") != y.is_negative();
                let count = y.to_i128().map(|v| v.unsigned_abs()).unwrap_or(u128::MAX);
                if leftward {
                    x.shl_bounded(count).map_err(stop)?
                } else if count >= (1 << 30) {
                    // Every bit shifted out: floor(x / 2^count) is 0 or -1 by sign.
                    if x.is_negative() {
                        JsBigInt::from_i128(-1)
                    } else {
                        JsBigInt::zero()
                    }
                } else {
                    x.shr(count as u64)
                }
            }
            _ => return Err(self.throw("TypeError", "unsupported BigInt operator")),
        };
        Ok(Value::BigInt(v))
    }

    /// The completion for a BigInt operation that stopped: V8's RangeError for a result over
    /// the size limit, or the host's interrupt.
    pub(crate) fn bigint_stop(&self, stop: crate::bigint::Stop) -> Abrupt {
        match stop {
            crate::bigint::Stop::TooLarge => {
                self.throw("RangeError", "Maximum BigInt size exceeded")
            }
            crate::bigint::Stop::Interrupted(reason) => Abrupt::Interrupt(reason),
        }
    }

    /// The digits of a BigInt in `radix` (Number::toString for BigInts, ECMA-262 §6.1.6.2.23).
    /// A result longer than the engine's string limit throws before the quadratic conversion
    /// runs, and a long conversion polls the host interrupt.
    pub(crate) fn bigint_to_string(
        &self,
        n: &crate::bigint::JsBigInt,
        radix: u32,
    ) -> Result<String, Abrupt> {
        // |n| >= 2^(bits - 1), so it has at least floor((bits - 1) * log_radix 2) + 1 digits.
        let bits = n.bit_len().saturating_sub(1) as f64;
        let digits = (bits * std::f64::consts::LN_2 / (radix as f64).ln()).floor();
        if digits >= MAX_STR_LEN as f64 {
            return Err(self.throw("RangeError", "Invalid string length"));
        }
        let mut poll = crate::bigint::Poll::new(&self.runtime_interrupt);
        n.to_string_radix_bounded(radix, &mut poll)
            .map_err(|stop| self.bigint_stop(stop))
    }

    /// ToBigInt: BigInt stays; Boolean → 0n/1n; String parses (SyntaxError if malformed); Number /
    /// undefined / null / Symbol throw TypeError.
    pub(crate) fn to_bigint(&mut self, v: &Value) -> Result<crate::bigint::JsBigInt, Abrupt> {
        let p = self.to_primitive(v, Hint::Number)?;
        match (p).view() {
            ValueView::BigInt(n) => Ok(n),
            ValueView::Bool(b) => Ok(crate::bigint::JsBigInt::from_u64(b as u64)),
            ValueView::Str(s) => {
                let mut poll = crate::bigint::Poll::new(&self.runtime_interrupt);
                string_to_bigint_bounded(&s.text(), &mut poll)
                    .map_err(|stop| self.bigint_stop(stop))?
                    .ok_or_else(|| self.throw("SyntaxError", "Cannot convert string to a BigInt"))
            }
            ValueView::Num(_) => {
                Err(self.throw("TypeError", "Cannot convert a Number to a BigInt"))
            }
            _ => Err(self.throw("TypeError", "Cannot convert value to a BigInt")),
        }
    }

    fn compare(&mut self, op: &str, l: Value, r: Value) -> Result<Value, Abrupt> {
        let lp = self.to_primitive(&l, Hint::Number)?;
        let rp = self.to_primitive(&r, Hint::Number)?;
        if let (ValueView::Str(a), ValueView::Str(b)) = (&(lp).view(), &(rp).view()) {
            // String relational comparison is per UTF-16 code unit (which differs from `str`
            // byte order once supplementary-plane characters are involved).
            let ord = a.cmp_units(b);
            let res = match op {
                "<" => ord.is_lt(),
                ">" => ord.is_gt(),
                "<=" => ord.is_le(),
                ">=" => ord.is_ge(),
                _ => unreachable!(),
            };
            return Ok(Value::Bool(res));
        }
        // BigInt operands compare exactly (never through f64, which would lose precision).
        let ord: Option<std::cmp::Ordering> = match (&(lp).view(), &(rp).view()) {
            (ValueView::BigInt(a), ValueView::BigInt(b)) => Some(a.cmp(b)),
            (ValueView::BigInt(a), ValueView::Str(t)) => {
                // StringToBigInt: an unparsable string makes the comparison undefined (false).
                string_to_bigint(&t.text()).map(|b| a.cmp(&b))
            }
            (ValueView::Str(t), ValueView::BigInt(b)) => {
                string_to_bigint(&t.text()).map(|a| a.cmp(b))
            }
            (ValueView::BigInt(a), _) => {
                let n = self.to_number(&rp)?;
                a.cmp_f64(n)
            }
            (_, ValueView::BigInt(b)) => {
                let n = self.to_number(&lp)?;
                b.cmp_f64(n).map(std::cmp::Ordering::reverse)
            }
            _ => {
                let a = self.to_number(&lp)?;
                let b = self.to_number(&rp)?;
                if a.is_nan() || b.is_nan() {
                    None
                } else {
                    a.partial_cmp(&b)
                }
            }
        };
        let res = match ord {
            None => false,
            Some(ord) => match op {
                "<" => ord.is_lt(),
                ">" => ord.is_gt(),
                "<=" => ord.is_le(),
                ">=" => ord.is_ge(),
                _ => unreachable!(),
            },
        };
        Ok(Value::Bool(res))
    }

    fn instanceof(&mut self, l: &Value, r: &Value) -> Result<Value, Abrupt> {
        if !matches!(&(r).view(), ValueView::Obj(_)) {
            return Err(self.throw(
                "TypeError",
                "right-hand side of instanceof is not an object",
            ));
        }
        // Defer to a `@@hasInstance` method if the RHS has one.
        if let Some(key) = crate::builtins::well_known_key(self, "hasInstance") {
            // Ordinary functions inherit the immutable default handler directly from the
            // canonical Function.prototype. Prove that shape without performing a generic
            // prototype-chain property read: an own override or altered function prototype
            // falls through, as do proxies and other side-table exotics.
            let inherits_default = match &(r).view() {
                ValueView::Obj(o) if self.ordinary_get_ptr(Gc::as_ptr(o) as usize) => {
                    let b = o.borrow();
                    b.proto
                        .as_ref()
                        .is_some_and(|p| Gc::ptr_eq(p, &self.function_proto))
                        && !b.props.contains(&key)
                }
                _ => false,
            };
            if inherits_default {
                return Ok(Value::Bool(self.ordinary_has_instance(r, l)?));
            }
            let handler = self.get_member_key(r, &key)?;
            if handler.is_callable() {
                // The overwhelmingly common handler is the immutable intrinsic inherited from
                // Function.prototype. Calling that native would only turn around and invoke this
                // same abstract operation, while paying call-frame and GC polling overhead.
                let is_default = matches!(
                    &(handler).view(),
                    ValueView::Obj(o)
                        if matches!(
                            o.borrow().call,
                            Callable::Native(f)
                                if f as usize
                                    == crate::builtins::default_has_instance as NativeFn as usize
                        )
                );
                if is_default {
                    return Ok(Value::Bool(self.ordinary_has_instance(r, l)?));
                }
                let res = self.call(handler, *r, std::slice::from_ref(l))?;
                return Ok(Value::Bool(self.to_boolean(&res)));
            }
            // GetMethod: a present but non-callable @@hasInstance is a TypeError.
            if !matches!((handler).view(), ValueView::Undefined | ValueView::Null) {
                return Err(self.throw("TypeError", "@@hasInstance is not callable"));
            }
        }
        if !r.is_callable() {
            return Err(self.throw("TypeError", "right-hand side of instanceof is not callable"));
        }
        Ok(Value::Bool(self.ordinary_has_instance(r, l)?))
    }

    /// Execute `instanceof` and seed the JIT's narrow ordinary-constructor cache. Filling is
    /// observationally inert: the real abstract operation still runs on every miss, and the
    /// generated hit revalidates every fact captured here before touching the operand stack.
    pub(crate) fn instanceof_ic(
        &mut self,
        l: &Value,
        r: &Value,
        cache: &std::cell::Cell<crate::bytecode::IcState>,
    ) -> Result<Value, Abrupt> {
        let key = crate::builtins::well_known_key(self, "hasInstance");
        // Checked counterpart of the machine-code hit. This also makes helper fallbacks cheap
        // when the generated path cannot consume last-reference operands: the cached RHS shape
        // proves no own @@hasInstance appeared, while the inherited intrinsic is read live.
        let cached = cache.get();
        if self.inline_ic_safe.get() && cached.depth == 0 {
            if let (Some(key), ValueView::Obj(lhs), ValueView::Obj(rhs)) =
                (key.as_deref(), &(l).view(), &(r).view())
            {
                let canonical = self
                    .function_proto
                    .borrow()
                    .props
                    .get(key)
                    .filter(|property| !property.accessor())
                    .map(Property::value)
                    .is_some_and(|value| {
                        matches!(
                            (value).view(),
                            ValueView::Obj(ref function)
                                if matches!(
                                    function.borrow().call,
                                    Callable::Native(native)
                                        if native as usize
                                            == crate::builtins::default_has_instance
                                                as *const () as usize
                                )
                        )
                    });
                let target = if canonical {
                    let b = rhs.borrow();
                    (b.is_constructor
                        && matches!(b.exotic, Exotic::None)
                        && b.ic_plain.get()
                        && b.props.shape() == cached.recv_shape
                        && b.proto
                            .as_ref()
                            .is_some_and(|p| Gc::ptr_eq(p, &self.function_proto)))
                    .then(|| {
                        b.props
                            .entry_at(cached.slot as usize)
                            .and_then(|(_, property)| {
                                (!property.accessor()).then(|| property.value())
                            })
                    })
                    .flatten()
                } else {
                    None
                };
                if let Some(ValueView::Obj(target)) = (target).map(|__value| (__value).view()) {
                    let mut cur = Some(*lhs);
                    while let Some(object) = cur {
                        let next = object.borrow().proto;
                        match next {
                            Some(proto) if Gc::ptr_eq(&proto, &target) => {
                                return Ok(Value::Bool(true));
                            }
                            Some(proto) => cur = Some(proto),
                            None => return Ok(Value::Bool(false)),
                        }
                    }
                }
            }
        }
        let fill = match (key.as_deref(), &(r).view()) {
            (Some(key), ValueView::Obj(o)) if self.ordinary_get_ptr(Gc::as_ptr(o) as usize) => {
                let b = o.borrow();
                if b.is_constructor
                    && matches!(b.exotic, Exotic::None)
                    && b.ic_plain.get()
                    && crate::value::is_cacheable_shape(b.props.shape())
                    && b.proto
                        .as_ref()
                        .is_some_and(|p| Gc::ptr_eq(p, &self.function_proto))
                    && !b.props.contains(key)
                {
                    b.props.prototype_slot().and_then(|slot| {
                        let prop = b.props.get("prototype")?;
                        (!prop.accessor() && matches!((prop.value()).view(), ValueView::Obj(_)))
                            .then_some((b.props.shape(), slot))
                    })
                } else {
                    None
                }
            }
            _ => None,
        };
        let out = self.instanceof(l, r)?;
        if let Some((shape, slot)) = fill {
            cache.set(crate::bytecode::IcState {
                recv_shape: shape,
                holder_shape: shape,
                slot,
                depth: 0,
                mid_ok: 0,
                mid_shape: 0,
                mid2_shape: 0,
                mid3_shape: 0,
                mid4_shape: 0,
            });
        }
        Ok(out)
    }

    /// OrdinaryHasInstance(C, O): unwrap bound functions to their target, then test whether `C`'s
    /// `prototype` appears on `O`'s prototype chain.
    pub(crate) fn ordinary_has_instance(&mut self, c: &Value, o: &Value) -> Result<bool, Abrupt> {
        let co = match &(c).view() {
            ValueView::Obj(x) if !matches!(x.borrow().call, Callable::None) => *x,
            _ => return Ok(false),
        };
        if let Callable::Bound(bound) = &co.borrow().call {
            let target = Value::Obj(bound.target);
            return self.ordinary_has_instance(&target, o);
        }
        let o_obj = match &(o).view() {
            ValueView::Obj(x) => *x,
            _ => return Ok(false),
        };
        let proto = match (self.get_member_key(c, "prototype")?).view() {
            ValueView::Obj(p) => p,
            _ => return Err(self.throw("TypeError", "prototype property is not an object")),
        };
        // With no proxies in the realm, [[GetPrototypeOf]] is a direct field read for every hop.
        // Keep the walk in `Gc`s so the common `instanceof` path avoids a proxy-table probe and
        // two `Value` constructions per prototype level. The moment a realm contains any proxy,
        // retain the fully observable path below (a proxy may appear anywhere in the chain).
        if self.proxies.is_empty() {
            let mut cur = o_obj;
            loop {
                let next = cur.borrow().proto;
                match next {
                    Some(x) => {
                        if Gc::ptr_eq(&x, &proto) {
                            return Ok(true);
                        }
                        cur = x;
                    }
                    None => return Ok(false),
                }
            }
        }
        // Walk O's prototype chain via [[GetPrototypeOf]] (so a proxy's trap participates).
        let mut cur = Value::Obj(o_obj);
        loop {
            let next = crate::builtins::js_get_prototype_of(self, &cur).map_err(Abrupt::Throw)?;
            match (next).view() {
                ValueView::Obj(x) => {
                    if Gc::ptr_eq(&x, &proto) {
                        return Ok(true);
                    }
                    cur = Value::Obj(x);
                }
                _ => return Ok(false),
            }
        }
    }

    // ----- abstract operations ----------------------------------------------------------------

    pub fn to_boolean(&self, v: &Value) -> bool {
        match &(v).view() {
            ValueView::Undefined | ValueView::Empty | ValueView::Null => false,
            ValueView::Bool(b) => *b,
            ValueView::Num(n) => *n != 0.0 && !n.is_nan(),
            ValueView::BigInt(n) => !n.is_zero(),
            ValueView::Str(s) => !s.is_empty(),
            // The [[IsHTMLDDA]] object is the one falsy Object.
            ValueView::Obj(_) => !self.is_htmldda(v),
            ValueView::Sym(_) => true,
        }
    }

    pub(crate) fn to_number(&mut self, v: &Value) -> Result<f64, Abrupt> {
        Ok(match &(v).view() {
            ValueView::Undefined | ValueView::Empty => f64::NAN,
            ValueView::Null => 0.0,
            ValueView::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            ValueView::Num(n) => *n,
            ValueView::BigInt(_) => {
                return Err(self.throw("TypeError", "Cannot convert a BigInt value to a number"));
            }
            ValueView::Str(s) => parse_number(&s.text()),
            ValueView::Sym(_) => {
                return Err(self.throw("TypeError", "Cannot convert a Symbol value to a number"));
            }
            ValueView::Obj(_) => {
                let p = self.to_primitive(v, Hint::Number)?;
                self.to_number(&p)?
            }
        })
    }

    /// ECMA-262 §7.1.3 ToNumeric: apply ToPrimitive with a number hint, preserve a BigInt
    /// primitive, and otherwise apply ToNumber. Keeping this operation shared is important for
    /// update expressions: an object can observably coerce to BigInt even though the original
    /// reference value was not itself a BigInt.
    pub(crate) fn to_numeric(&mut self, value: Value) -> Result<Value, Abrupt> {
        match value.view() {
            ValueView::Num(_) | ValueView::BigInt(_) => Ok(value),
            _ => {
                let primitive = self.to_primitive(&value, Hint::Number)?;
                if matches!((primitive).view(), ValueView::BigInt(_)) {
                    Ok(primitive)
                } else {
                    Ok(Value::Num(self.to_number(&primitive)?))
                }
            }
        }
    }

    pub(crate) fn to_int32(&mut self, v: &Value) -> Result<i32, Abrupt> {
        let n = self.to_number(v)?;
        Ok(to_int32(n))
    }
    pub(crate) fn to_uint32(&mut self, v: &Value) -> Result<u32, Abrupt> {
        let n = self.to_number(v)?;
        Ok(to_int32(n) as u32)
    }

    pub(crate) fn to_string(&mut self, v: &Value) -> Result<crate::lstr::LStr, Abrupt> {
        match &(v).view() {
            ValueView::Undefined | ValueView::Empty => Ok(crate::lstr::LStr::from("undefined")),
            ValueView::Null => Ok(crate::lstr::LStr::from("null")),
            ValueView::Bool(b) => Ok(crate::lstr::LStr::from(if *b { "true" } else { "false" })),
            ValueView::Num(n) => Ok(crate::lstr::LStr::from(self.num_to_str(*n).as_str())),
            ValueView::BigInt(n) => Ok(crate::lstr::LStr::from(
                self.bigint_to_string(n, 10)?.as_str(),
            )),
            ValueView::Str(s) => Ok(*s),
            ValueView::Sym(_) => {
                Err(self.throw("TypeError", "Cannot convert a Symbol value to a string"))
            }
            ValueView::Obj(_) => {
                let perf_started = crate::jit::perf_stage_start();
                let result = (|| {
                    let p = self.to_primitive(v, Hint::String)?;
                    {
                        let __value_match_original = p;
                        match (__value_match_original).view() {
                            ValueView::Obj(_) => {
                                Err(self.throw("TypeError", "cannot convert object to string"))
                            }
                            _ => {
                                let other = __value_match_original;
                                self.to_string(&other)
                            }
                        }
                    }
                })();
                crate::jit::perf_to_string_object_end(perf_started, result.is_ok());
                result
            }
        }
    }

    pub(crate) fn to_property_key(
        &mut self,
        v: &Value,
    ) -> Result<crate::value::PropertyKey, Abrupt> {
        // A symbol key maps to its internal NUL-prefixed key; everything else is its string form.
        if let ValueView::Sym(s) = &(v).view() {
            return Ok(crate::value::PropertyKey::symbol(*s));
        }
        // ToPropertyKey: ToPrimitive(hint String) first — a Symbol result stays a symbol key (rather
        // than being stringified, which would throw), so `obj[wrapperWhoseToStringReturnsASymbol]`
        // is a symbol-keyed access.
        let prim = self.to_primitive(v, Hint::String)?;
        if let ValueView::Sym(s) = &(prim).view() {
            return Ok(crate::value::PropertyKey::symbol(*s));
        }
        Ok(crate::value::PropertyKey::string(
            self.to_string(&prim)?.to_string(),
        ))
    }

    pub(crate) fn to_primitive(&mut self, v: &Value, hint: Hint) -> Result<Value, Abrupt> {
        let obj = match &(v).view() {
            ValueView::Obj(o) => *o,
            _ => return Ok(*v),
        };
        let perf_started = crate::jit::perf_stage_start();
        let result: Result<Value, Abrupt> = (|| {
            // A `@@toPrimitive` method takes precedence over valueOf/toString.
            if let Some(key) = crate::builtins::well_known_key(self, "toPrimitive") {
                let f = self.get_member_key(&Value::Obj(obj), &key)?;
                // GetMethod: a present-but-non-callable @@toPrimitive (not undefined/null) is a TypeError.
                if !matches!((f).view(), ValueView::Undefined | ValueView::Null) && !f.is_callable()
                {
                    return Err(self.throw("TypeError", "@@toPrimitive is not callable"));
                }
                if f.is_callable() {
                    let hint_str = match hint {
                        Hint::String => "string",
                        Hint::Number => "number",
                        Hint::Default => "default",
                    };
                    let r = self.call(f, *v, &[Value::lstr(hint_str)])?;
                    if matches!((r).view(), ValueView::Obj(_)) {
                        return Err(
                            self.throw("TypeError", "Cannot convert object to a primitive value")
                        );
                    }
                    return Ok(r);
                }
            }
            let order: [&str; 2] = match hint {
                Hint::String => ["toString", "valueOf"],
                _ => ["valueOf", "toString"],
            };
            for method in order {
                let f = self.get_member_key(&Value::Obj(obj), method)?;
                if f.is_callable() {
                    let r = self.call(f, *v, &[])?;
                    if !matches!((r).view(), ValueView::Obj(_)) {
                        return Ok(r);
                    }
                }
            }
            Err(self.throw("TypeError", "cannot convert object to primitive value"))
        })();
        crate::jit::perf_to_primitive_end(perf_started, result.is_ok());
        result
    }

    /// ECMAScript `Number::toString` (base 10): the shortest round-tripping digit string, formatted
    /// fixed or exponential per the spec's exponent thresholds (≥1e21 or <1e-6 → exponential).
    pub(crate) fn num_to_str(&self, n: f64) -> String {
        // Integer fast path: array indices and loop counters dominate ToString(Number) traffic,
        // and integers never need the shortest-float machinery. (-0.0 correctly prints "0".)
        if n.trunc() == n && n.abs() <= 9_007_199_254_740_992.0 {
            return (n as i64).to_string();
        }
        if n.is_nan() {
            return "NaN".to_string();
        }
        if n == 0.0 {
            return "0".to_string();
        }
        if n.is_infinite() {
            return if n > 0.0 {
                "Infinity".to_string()
            } else {
                "-Infinity".to_string()
            };
        }
        // The spec's `n`: value = digits × 10^(n-k).
        let (digits, np) = shortest_digits(n.abs());
        let digits = digits.as_str();
        let k = digits.len() as i32;
        let mut out = String::with_capacity(digits.len() + 8);
        if n < 0.0 {
            out.push('-');
        }
        if k <= np && np <= 21 {
            out.push_str(digits);
            out.extend(std::iter::repeat_n('0', (np - k) as usize));
        } else if 0 < np && np <= 21 {
            let (int, fraction) = digits.split_at(np as usize);
            out.push_str(int);
            out.push('.');
            out.push_str(fraction);
        } else if -6 < np && np <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-np) as usize));
            out.push_str(digits);
        } else {
            let (first, rest) = digits.split_at(1);
            out.push_str(first);
            if !rest.is_empty() {
                out.push('.');
                out.push_str(rest);
            }
            let exponent = np - 1;
            let _ = write!(
                out,
                "e{}{}",
                if exponent >= 0 { '+' } else { '-' },
                exponent.unsigned_abs()
            );
        }
        out
    }

    pub(crate) fn strict_equals(&self, a: &Value, b: &Value) -> bool {
        match (&(a).view(), &(b).view()) {
            (ValueView::Undefined, ValueView::Undefined) => true,
            (ValueView::Null, ValueView::Null) => true,
            (ValueView::Bool(x), ValueView::Bool(y)) => x == y,
            (ValueView::Num(x), ValueView::Num(y)) => x == y,
            (ValueView::BigInt(x), ValueView::BigInt(y)) => x == y,
            (ValueView::Str(x), ValueView::Str(y)) => x == y,
            (ValueView::Sym(x), ValueView::Sym(y)) => x.id == y.id,
            (ValueView::Obj(x), ValueView::Obj(y)) => Gc::ptr_eq(x, y),
            _ => false,
        }
    }

    /// IsConstructor: whether `v` has a [[Construct]] internal method.
    pub(crate) fn value_is_constructor(&self, v: &Value) -> bool {
        let ValueView::Obj(o) = &(v).view() else {
            return false;
        };
        if let Some((target, _)) = self.proxies.get(&(Gc::as_ptr(o) as usize)) {
            // A proxy is a constructor exactly when its target is.
            return self.value_is_constructor(&target.clone());
        }
        let b = o.borrow();
        match &b.call {
            Callable::User(user) => {
                !(user.func.is_arrow
                    || user.func.is_method
                    || user.func.is_generator
                    || user.func.is_async)
            }
            call if call.is_native() => {
                b.is_constructor
                    || b.props
                        .get("prototype")
                        .map(|p| !p.accessor())
                        .unwrap_or(false)
            }
            // ECMA-262 BoundFunctionCreate step 5: [[Construct]] is installed at creation exactly
            // when IsConstructor(targetFunc) held then; `bind` records that answer. Reading the
            // record (instead of re-walking the target chain) keeps long bind chains O(1) here.
            Callable::Bound(_) => b.is_constructor,
            _ => false,
        }
    }

    pub(crate) fn loose_equals(&mut self, a: &Value, b: &Value) -> Result<bool, Abrupt> {
        // [[IsHTMLDDA]] compares loosely equal to undefined/null (and to itself, as an object).
        if matches!(&(a).view(), ValueView::Undefined | ValueView::Null) && self.is_htmldda(b)
            || matches!(&(b).view(), ValueView::Undefined | ValueView::Null) && self.is_htmldda(a)
        {
            return Ok(true);
        }
        Ok(match (&(a).view(), &(b).view()) {
            (ValueView::Undefined | ValueView::Null, ValueView::Undefined | ValueView::Null) => {
                true
            }
            (ValueView::BigInt(x), ValueView::BigInt(y)) => x == y,
            (ValueView::BigInt(x), ValueView::Num(y))
            | (ValueView::Num(y), ValueView::BigInt(x)) => x.eq_f64(*y),
            (ValueView::BigInt(x), ValueView::Str(s))
            | (ValueView::Str(s), ValueView::BigInt(x)) => {
                // StringToBigInt: an unparsable string compares unequal.
                string_to_bigint(&s.text())
                    .map(|n| n == *x)
                    .unwrap_or(false)
            }
            (ValueView::BigInt(_), ValueView::Obj(_)) => {
                let bp = self.to_primitive(b, Hint::Default)?;
                self.loose_equals(a, &bp)?
            }
            (ValueView::Obj(_), ValueView::BigInt(_)) => {
                let ap = self.to_primitive(a, Hint::Default)?;
                self.loose_equals(&ap, b)?
            }
            (ValueView::Num(_), ValueView::Num(_))
            | (ValueView::Str(_), ValueView::Str(_))
            | (ValueView::Bool(_), ValueView::Bool(_))
            | (ValueView::Sym(_), ValueView::Sym(_))
            | (ValueView::Obj(_), ValueView::Obj(_)) => self.strict_equals(a, b),
            (ValueView::Num(_), ValueView::Str(_)) => {
                let bn = self.to_number(b)?;
                self.strict_equals(a, &Value::Num(bn))
            }
            (ValueView::Str(_), ValueView::Num(_)) => {
                let an = self.to_number(a)?;
                self.strict_equals(&Value::Num(an), b)
            }
            (ValueView::Bool(_), _) => {
                let an = self.to_number(a)?;
                self.loose_equals(&Value::Num(an), b)?
            }
            (_, ValueView::Bool(_)) => {
                let bn = self.to_number(b)?;
                self.loose_equals(a, &Value::Num(bn))?
            }
            (ValueView::Obj(_), ValueView::Num(_) | ValueView::Str(_) | ValueView::Sym(_)) => {
                let ap = self.to_primitive(a, Hint::Default)?;
                self.loose_equals(&ap, b)?
            }
            (ValueView::Num(_) | ValueView::Str(_) | ValueView::Sym(_), ValueView::Obj(_)) => {
                let bp = self.to_primitive(b, Hint::Default)?;
                self.loose_equals(a, &bp)?
            }
            _ => false,
        })
    }
}

#[derive(Clone, Copy)]
pub enum Hint {
    Default,
    Number,
    String,
}

enum LoopStep {
    /// A borrowed fragment already completed LoopEvaluation, including all local cleanup.
    Transferred(Completion),
    /// Keep looping; the value is this iteration's body completion value (for the loop's own
    /// completion value, per UpdateEmpty in ForBodyEvaluation).
    Continue(Value),
    /// Stop looping; the value is the final iteration's body completion value (or undefined).
    Done(Value),
}

/// A field initializer whose evaluation reads no binding (not even `this` or `undefined`'s
/// resolution, which the parser already folds), declares nothing and creates no closure:
/// primitive literals, and array/object literals built only from them.
fn env_free_initializer(e: &Expr) -> bool {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| env_free_initializer(e));
    }
    match e {
        Expr::Num(_)
        | Expr::BigInt(_)
        | Expr::Str(_)
        | Expr::Bool(_)
        | Expr::Null
        | Expr::Undefined => true,
        Expr::Paren(inner) => env_free_initializer(inner),
        Expr::Array(items) => items.iter().all(|item| match item {
            ArrayElem::Item(e) => env_free_initializer(e),
            ArrayElem::Hole => true,
            ArrayElem::Spread(_) => false,
        }),
        Expr::Object(props) => props.iter().all(|prop| {
            matches!(
                prop,
                PropDef::KeyValue {
                    key: PropKey::Ident(_) | PropKey::Str(_) | PropKey::Num(_),
                    value,
                } if env_free_initializer(value)
            )
        }),
        _ => false,
    }
}

/// [`bind`] with an already shared name, which the binding map adopts without copying.
fn bind_shared_name(env: &Env, name: Rc<str>, value: Value) {
    env.borrow_mut()
        .vars
        .insert(name, Binding::data(value, true, true));
}

/// The hidden binding names of a class field initializer's environment (`0` = `this`,
/// `1` = `%fieldinit%`), shared instead of allocated for every field of every instance. The
/// thread-local has no destructor: its single reference per name is never released, so a
/// pooled coroutine thread's exit cannot touch counts of names still in use elsewhere.
fn field_init_name(which: usize) -> Rc<str> {
    thread_local! {
        static NAMES: [std::mem::ManuallyDrop<Rc<str>>; 2] = [
            std::mem::ManuallyDrop::new(Rc::from("this")),
            std::mem::ManuallyDrop::new(Rc::from("%fieldinit%")),
        ];
    }
    NAMES.with(|names| Rc::clone(&names[which]))
}

/// Insert an initialized, mutable binding into `env` (used for the hidden `%super*%`/`this` slots).
pub(crate) fn bind(env: &Env, name: &str, value: Value) {
    env.borrow_mut()
        .vars
        .insert(name, Binding::data(value, true, true));
}

fn opt_obj(o: &Option<Gc>) -> Value {
    match o {
        Some(g) => Value::Obj(*g),
        None => Value::Undefined,
    }
}

/// The synthesized default class constructor. Derived: `constructor(...args) { super(...args); }`;
/// base: `constructor() {}`. Its stack frames report the class's position.
fn default_constructor(derived: bool, class: &Class) -> Function {
    let body = if derived {
        vec![Stmt::Expr(Expr::Call {
            callee: Box::new(Expr::Super),
            args: vec![ArrayElem::Spread(Expr::Ident(
                "args".to_string(),
                NO_POSITION,
            ))],
            optional: false,
            pos: NO_POSITION,
        })]
    } else {
        Vec::new()
    };
    let params = if derived {
        vec![Param {
            pattern: Pattern::Ident("args".to_string()),
            default: None,
            rest: true,
        }]
    } else {
        Vec::new()
    };
    Function {
        scan: std::cell::Cell::new(0),
        hoist: std::cell::OnceCell::new(),
        runtime: std::cell::OnceCell::new(),

        env_layouts: std::cell::OnceCell::new(),
        name: None,
        params,
        body: body.into(),
        is_arrow: false,
        is_strict: true,
        expr_body: false,
        is_generator: false,
        is_async: false,
        is_method: false,
        is_fn_expr: false,
        default_ctor: derived,
        self_hosted: false,
        run_once: false,
        source: None,
        script: class.script.clone(),
        start: class.start,
    }
}

/// Whether any statement contains a `super(...)` call within its own super-context. The walk is
/// transparent through arrow functions and ordinary control flow but stops at a (non-arrow) function
/// or class boundary, each of which establishes a fresh super-context.
/// Whether a statement directly in a block is a `using` / `await using` declaration (so the block
/// is a disposal boundary). Does not recurse — disposal scopes are per lexical block.
pub(crate) fn stmt_declares_using(s: &Stmt) -> bool {
    matches!(
        crate::interpreter::unwrap_export(s),
        Stmt::VarDecl {
            kind: DeclKind::Using | DeclKind::AwaitUsing,
            ..
        }
    )
}

/// `Contains` over eval code: search for a node matching `pred`, descending into arrow functions
/// (their parameter defaults and bodies) but not into ordinary functions or class bodies.
fn stmts_contain(stmts: &[Stmt], pred: fn(&Expr) -> bool) -> bool {
    stmts.iter().any(|s| stmt_contains(s, pred))
}

fn stmt_contains(s: &Stmt, pred: fn(&Expr) -> bool) -> bool {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| stmt_contains(s, pred));
    }
    let e = |x: &Expr| expr_contains(x, pred);
    match s {
        Stmt::Expr(x) | Stmt::Throw(x) => e(x),
        Stmt::Return(x) => x.as_ref().is_some_and(e),
        Stmt::VarDecl { decls, .. } => decls.iter().any(|(_, init)| init.as_ref().is_some_and(e)),
        Stmt::If { test, cons, alt } => {
            e(test)
                || stmt_contains(cons, pred)
                || alt.as_deref().is_some_and(|s| stmt_contains(s, pred))
        }
        Stmt::Block(b) => stmts_contain(b, pred),
        Stmt::While { test, body, .. } | Stmt::DoWhile { body, test, .. } => {
            e(test) || stmt_contains(body, pred)
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => {
            init.as_deref().is_some_and(|i| match i {
                ForInit::Expr(x) => e(x),
                ForInit::VarDecl { decls, .. } => {
                    decls.iter().any(|(_, x)| x.as_ref().is_some_and(e))
                }
            }) || test.as_ref().is_some_and(e)
                || update.as_ref().is_some_and(e)
                || stmt_contains(body, pred)
        }
        Stmt::ForInOf { right, body, .. } => e(right) || stmt_contains(body, pred),
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => {
            stmts_contain(block, pred)
                || handler
                    .as_ref()
                    .is_some_and(|(_, b)| stmts_contain(b, pred))
                || finalizer.as_ref().is_some_and(|b| stmts_contain(b, pred))
        }
        Stmt::Switch { disc, cases } => {
            e(disc)
                || cases
                    .iter()
                    .any(|c| c.test.as_ref().is_some_and(e) || stmts_contain(&c.body, pred))
        }
        Stmt::Labeled { body, .. } | Stmt::With { body, .. } => stmt_contains(body, pred),
        // A (non-arrow) function or class declaration opens its own context.
        _ => false,
    }
}

pub(crate) fn expr_contains(x: &Expr, pred: fn(&Expr) -> bool) -> bool {
    expr_contains_impl(x, pred, true)
}

/// Suspension belongs to the executing expression, never to a newly created function's body
/// (including arrows). Keep this separate from lexical Contains queries such as arguments/super.
pub(crate) fn expr_has_own_suspension(x: &Expr) -> bool {
    expr_contains_impl(
        x,
        |expr| matches!(expr, Expr::Yield { .. } | Expr::Await(_)),
        false,
    )
}

fn expr_contains_impl(x: &Expr, pred: fn(&Expr) -> bool, descend_arrows: bool) -> bool {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| expr_contains_impl(x, pred, descend_arrows));
    }
    if pred(x) {
        return true;
    }
    let e = |x: &Expr| expr_contains_impl(x, pred, descend_arrows);
    let array = |elements: &[ArrayElem]| {
        elements.iter().any(|element| match element {
            ArrayElem::Item(value) | ArrayElem::Spread(value) => e(value),
            ArrayElem::Hole => false,
        })
    };
    match x {
        Expr::Call { callee, args, .. } => e(callee) || array(args),
        Expr::New { callee, args, .. } => e(callee) || array(args),
        Expr::Unary { arg, .. }
        | Expr::Update { arg, .. }
        | Expr::Await(arg)
        | Expr::ToStr(arg)
        | Expr::Paren(arg) => e(arg),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => e(left) || e(right),
        Expr::Assign { target, value, .. } => e(target) || e(value),
        Expr::Cond { test, cons, alt } => e(test) || e(cons) || e(alt),
        Expr::Member { obj, .. } | Expr::OptionalChain(obj) => e(obj),
        Expr::Index { obj, index, .. } => e(obj) || e(index),
        Expr::Seq(v) => v.iter().any(e),
        Expr::Array(elems) => array(elems),
        Expr::Yield { arg, .. } => arg.as_deref().is_some_and(e),
        Expr::ImportCall { spec, options, .. } => e(spec) || options.as_deref().is_some_and(e),
        Expr::PrivateIn { obj, .. } => e(obj),
        Expr::TaggedTemplate { tag, subs, .. } => e(tag) || subs.iter().any(e),
        Expr::Object(props) => props.iter().any(|p| match p {
            PropDef::KeyValue { key, value } | PropDef::Cover { key, value } => {
                matches!(key, PropKey::Computed(key) if e(key)) || e(value)
            }
            PropDef::Spread(x) => e(x),
            PropDef::Method { key, .. }
            | PropDef::Getter { key, .. }
            | PropDef::Setter { key, .. } => matches!(key, PropKey::Computed(key) if e(key)),
            PropDef::Proto(value) => e(value),
        }),
        Expr::Class(class) => {
            class.decorators.iter().any(e)
                || class.superclass.as_deref().is_some_and(e)
                || class.members.iter().any(|member| {
                    member.decorators.iter().any(e)
                        || matches!(&member.key, PropKey::Computed(key) if e(key))
                })
        }
        // `Contains` descends into arrow functions; an ordinary function/class does not.
        Expr::Func(f) if descend_arrows && f.is_arrow => {
            f.params.iter().any(|p| p.default.as_ref().is_some_and(e))
                || stmts_contain(&f.body, pred)
        }
        _ => false,
    }
}

fn stmts_have_super_call(stmts: &[Stmt]) -> bool {
    stmts_contain(
        stmts,
        |e| matches!(e, Expr::Call { callee, .. } if matches!(**callee, Expr::Super)),
    )
}

/// Contains a `super.x` / `super[x]` reference (arrow-descending, like `Contains`).
fn stmts_have_super_prop(stmts: &[Stmt]) -> bool {
    stmts_contain(stmts, |e| {
        matches!(e, Expr::Member { obj, .. } | Expr::Index { obj, .. }
            if matches!(&**obj, Expr::Super))
    })
}

/// ContainsArguments: an `arguments` identifier reference (arrow-descending, like `Contains`).
fn stmts_have_arguments_ref(stmts: &[Stmt]) -> bool {
    stmts_contain(
        stmts,
        |e| matches!(e, Expr::Ident(n, _) if n == "arguments"),
    )
}

/// A class field initializer (or computed field name) may not contain `arguments` or a `super(...)`
/// call — both are early errors. The walk is transparent through arrow functions (which inherit the
/// field's `arguments`/super context) but stops at ordinary functions and classes. Returns the error
/// message on the first violation.
pub(crate) fn field_init_error(e: &Expr) -> Option<&'static str> {
    fi_expr(e, true)
}

/// A `static { … }` block may not contain `arguments` or a `super(...)` call (the scan stops at
/// nested ordinary functions and classes).
pub(crate) fn static_block_error(func: &crate::ast::Function) -> Option<&'static str> {
    fi_stmts(&func.body, true)
}

/// A non-constructor method body may not contain a `super(...)` call (only a derived constructor
/// can). Descends into arrow functions (which inherit the method's super context).
/// A non-constructor method may not contain a `super(...)` call in its parameter list *or* its body.
/// Script/global code may not contain a `super(...)` call anywhere outside a class.
pub(crate) fn top_level_super_call_error(body: &[Stmt]) -> Option<&'static str> {
    fi_stmts(body, false)
}

pub(crate) fn method_super_call_error_full(func: &crate::ast::Function) -> Option<&'static str> {
    for p in &func.params {
        if let Some(d) = &p.default {
            if let Some(msg) = fi_expr(d, false) {
                return Some(msg);
            }
        }
    }
    fi_stmts(&func.body, false)
}

/// `args` also flags `arguments` (a field-initializer rule); methods omit it since they may use it.
fn fi_expr(e: &Expr, args: bool) -> Option<&'static str> {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| fi_expr(e, args));
    }
    match e {
        Expr::Ident(n, _) if args && n == "arguments" => {
            Some("'arguments' is not allowed in a class field initializer")
        }
        Expr::Call {
            callee, args: a, ..
        } => {
            if matches!(**callee, Expr::Super) {
                return Some("a super call is not allowed here");
            }
            fi_expr(callee, args).or_else(|| fi_arr(a, args))
        }
        Expr::New {
            callee, args: a, ..
        } => fi_expr(callee, args).or_else(|| fi_arr(a, args)),
        Expr::Unary { arg, .. } | Expr::Update { arg, .. } | Expr::Await(arg) => fi_expr(arg, args),
        Expr::Binary { left, right, .. } | Expr::Logical { left, right, .. } => {
            fi_expr(left, args).or_else(|| fi_expr(right, args))
        }
        Expr::Assign { target, value, .. } => {
            fi_expr(target, args).or_else(|| fi_expr(value, args))
        }
        Expr::Cond { test, cons, alt } => fi_expr(test, args)
            .or_else(|| fi_expr(cons, args))
            .or_else(|| fi_expr(alt, args)),
        Expr::Member { obj, .. } | Expr::OptionalChain(obj) => fi_expr(obj, args),
        Expr::Index { obj, index, .. } => fi_expr(obj, args).or_else(|| fi_expr(index, args)),
        Expr::Seq(v) => v.iter().find_map(|e| fi_expr(e, args)),
        Expr::Array(elems) => fi_arr(elems, args),
        Expr::Yield { arg, .. } => arg.as_deref().and_then(|e| fi_expr(e, args)),
        Expr::ImportCall { spec, .. } => fi_expr(spec, args),
        Expr::PrivateIn { obj, .. } => fi_expr(obj, args),
        Expr::TaggedTemplate { tag, subs, .. } => {
            fi_expr(tag, args).or_else(|| subs.iter().find_map(|e| fi_expr(e, args)))
        }
        Expr::Object(props) => props.iter().find_map(|p| match p {
            PropDef::KeyValue { key, value } | PropDef::Cover { key, value } => {
                fi_key(key, args).or_else(|| fi_expr(value, args))
            }
            PropDef::Spread(e) | PropDef::Proto(e) => fi_expr(e, args),
            // Methods/getters/setters open their own context (only their computed key inherits).
            PropDef::Method { key, .. }
            | PropDef::Getter { key, .. }
            | PropDef::Setter { key, .. } => fi_key(key, args),
        }),
        // Arrow functions inherit the surrounding context: descend into params and body.
        Expr::Func(f) if f.is_arrow => {
            fi_params(&f.params, args).or_else(|| fi_stmts(&f.body, args))
        }
        // Ordinary functions / classes establish their own context.
        _ => None,
    }
}

fn fi_key(key: &PropKey, args: bool) -> Option<&'static str> {
    match key {
        PropKey::Computed(e) => fi_expr(e, args),
        _ => None,
    }
}

fn fi_arr(elems: &[ArrayElem], args: bool) -> Option<&'static str> {
    elems.iter().find_map(|el| match el {
        ArrayElem::Item(e) | ArrayElem::Spread(e) => fi_expr(e, args),
        ArrayElem::Hole => None,
    })
}

fn fi_params(params: &[Param], args: bool) -> Option<&'static str> {
    params
        .iter()
        .find_map(|p| p.default.as_ref().and_then(|e| fi_expr(e, args)))
}

fn fi_stmts(stmts: &[Stmt], args: bool) -> Option<&'static str> {
    stmts.iter().find_map(|s| fi_stmt(s, args))
}

fn fi_stmt(s: &Stmt, args: bool) -> Option<&'static str> {
    if crate::interpreter::ast_stack_low() {
        return crate::interpreter::with_ast_stack(|| fi_stmt(s, args));
    }
    match s {
        Stmt::Expr(e) | Stmt::Throw(e) => fi_expr(e, args),
        Stmt::Return(e) => e.as_ref().and_then(|e| fi_expr(e, args)),
        Stmt::VarDecl { decls, .. } => decls
            .iter()
            .find_map(|(_, i)| i.as_ref().and_then(|e| fi_expr(e, args))),
        Stmt::If { test, cons, alt } => fi_expr(test, args)
            .or_else(|| fi_stmt(cons, args))
            .or_else(|| alt.as_deref().and_then(|s| fi_stmt(s, args))),
        Stmt::Block(b) => fi_stmts(b, args),
        Stmt::While { test, body, .. } | Stmt::DoWhile { body, test, .. } => {
            fi_expr(test, args).or_else(|| fi_stmt(body, args))
        }
        Stmt::For {
            init,
            test,
            update,
            body,
            ..
        } => init
            .as_deref()
            .and_then(|i| match i {
                ForInit::Expr(e) => fi_expr(e, args),
                ForInit::VarDecl { decls, .. } => decls
                    .iter()
                    .find_map(|(_, x)| x.as_ref().and_then(|e| fi_expr(e, args))),
            })
            .or_else(|| test.as_ref().and_then(|e| fi_expr(e, args)))
            .or_else(|| update.as_ref().and_then(|e| fi_expr(e, args)))
            .or_else(|| fi_stmt(body, args)),
        Stmt::ForInOf { right, body, .. } => fi_expr(right, args).or_else(|| fi_stmt(body, args)),
        Stmt::Try {
            block,
            handler,
            finalizer,
        } => fi_stmts(block, args)
            .or_else(|| handler.as_ref().and_then(|(_, b)| fi_stmts(b, args)))
            .or_else(|| finalizer.as_ref().and_then(|b| fi_stmts(b, args))),
        Stmt::Switch { disc, cases } => fi_expr(disc, args).or_else(|| {
            cases.iter().find_map(|c| {
                c.test
                    .as_ref()
                    .and_then(|e| fi_expr(e, args))
                    .or_else(|| fi_stmts(&c.body, args))
            })
        }),
        Stmt::Labeled { body, .. } | Stmt::With { body, .. } => fi_stmt(body, args),
        _ => None,
    }
}

/// A decorator context's `addInitializer(fn)`: records `fn` to run when the element is installed
/// (collected per-decorator in `Interp.decorator_initializers`).
fn dec_add_initializer(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let f = args.first().cloned().unwrap_or(Value::Undefined);
    if !f.is_callable() {
        return Err(i.make_error("TypeError", "addInitializer expects a callable"));
    }
    i.decorator_initializers.push(f);
    Ok(Value::Undefined)
}

/// The resolver pair's shared [[AlreadyResolved]] cell (`args[0]`): true (and mark) on first use.
fn promise_mark_already(flag: &Value) -> bool {
    if let ValueView::Obj(o) = &(flag).view() {
        let used = o.borrow().internal_slots().contains("#\u{0}called");
        if used {
            return false;
        }
        o.borrow_mut().internal_slots_mut().insert(
            "#\u{0}called",
            crate::value::Property::data(Value::Bool(true), true, false, true),
        );
        return true;
    }
    true
}

fn promise_resolve_native(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    if promise_mark_already(&arg_at(args, 0)) {
        i.resolve_promise(&this, arg_at(args, 1));
    }
    Ok(Value::Undefined)
}

fn promise_reject_native(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    if promise_mark_already(&arg_at(args, 0)) {
        i.reject_promise(&this, arg_at(args, 1));
    }
    Ok(Value::Undefined)
}

fn arg_at(args: &[Value], k: usize) -> Value {
    args.get(k).cloned().unwrap_or(Value::Undefined)
}

/// Whether an expression is an anonymous function/arrow/class (eligible for NamedEvaluation).
pub(crate) fn is_anonymous_fn(e: &Expr) -> bool {
    match e {
        Expr::Func(f) => f.name.is_none(),
        Expr::Class(c) => c.name.is_none(),
        _ => false,
    }
}

/// Whether `s` (labels and export wrappers unwrapped) is a block-scoped declaration — the set
/// [`Interp::declare_block_lexicals`] instantiates at block entry.
fn block_declares_lexical(s: &Stmt) -> bool {
    let mut s = crate::interpreter::unwrap_export(s);
    while let Stmt::Labeled { body, .. } = s {
        s = body;
    }
    matches!(
        s,
        Stmt::VarDecl {
            kind: DeclKind::Let | DeclKind::Const | DeclKind::Using | DeclKind::AwaitUsing,
            ..
        } | Stmt::FuncDecl(_)
            | Stmt::ClassDecl(_)
    )
}

pub(crate) fn js_mod(a: f64, b: f64) -> f64 {
    if b == 0.0 || a.is_nan() || b.is_nan() || a.is_infinite() {
        return f64::NAN;
    }
    if b.is_infinite() {
        return a;
    }
    if a == 0.0 {
        return a;
    }
    a % b
}

/// The digits of Number::toString's shortest form of the finite, positive `x` and its `n`:
/// `x` is 0.d₁…dₖ × 10ⁿ (ECMA-262 §6.1.6.1.20 step 5). Among the shortest digit strings that
/// read back as `x`, NOTE 2 recommends the one closest to `x`, an even one on a tie, which V8
/// and SpiderMonkey print. Core's shortest formatting returns some shortest string: below 16
/// digits it is the only one (the decimal spacing exceeds the rounding interval), but at 16
/// or 17 digits a second can lie in the interval, so the correctly rounded string of that
/// length (ties to even) wins whenever it differs and also reads back as `x`.
pub(crate) fn shortest_digits(x: f64) -> (Digits, i32) {
    let (digits, n) = Digits::scientific(format_args!("{x:e}"));
    if digits.len < 16 {
        return (digits, n);
    }
    let (rounded, rounded_n) = Digits::scientific(format_args!("{x:.*e}", digits.len - 1));
    if (rounded.as_str(), rounded_n) == (digits.as_str(), n) || rounded.len > digits.len {
        return (digits, n);
    }
    let mut text = FloatText::default();
    let _ = write!(
        text,
        "{}e{}",
        rounded.as_str(),
        rounded_n - rounded.len as i32
    );
    if text.as_str().parse::<f64>() == Ok(x) {
        (rounded, rounded_n)
    } else {
        (digits, n)
    }
}

/// At most 17 significant decimal digits of a double, without trailing zeros.
pub(crate) struct Digits {
    bytes: [u8; 17],
    len: usize,
}

impl Digits {
    /// The digits and `n` (one more than the exponent) of core's scientific form of a double.
    fn scientific(args: std::fmt::Arguments<'_>) -> (Digits, i32) {
        let mut text = FloatText::default();
        let _ = text.write_fmt(args);
        let (mantissa, exponent) = text.as_str().split_once('e').unwrap_or(("0", "0"));
        let mut digits = Digits {
            bytes: [b'0'; 17],
            len: 0,
        };
        for digit in mantissa.bytes().filter(u8::is_ascii_digit).take(17) {
            digits.bytes[digits.len] = digit;
            digits.len += 1;
        }
        while digits.len > 1 && digits.bytes[digits.len - 1] == b'0' {
            digits.len -= 1;
        }
        (digits, exponent.parse::<i32>().unwrap_or(0) + 1)
    }

    pub(crate) fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("0")
    }
}

/// A stack buffer for one double's scientific text (at most 17 digits, a point and `e-324`).
#[derive(Default)]
struct FloatText {
    bytes: [u8; 32],
    len: usize,
}

impl FloatText {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }
}

impl std::fmt::Write for FloatText {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let end = self.len + s.len();
        self.bytes
            .get_mut(self.len..end)
            .ok_or(std::fmt::Error)?
            .copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

pub(crate) fn to_int32(n: f64) -> i32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    let n = n.trunc();
    let m = n.rem_euclid(4294967296.0);
    if m >= 2147483648.0 {
        (m - 4294967296.0) as i32
    } else {
        m as i32
    }
}

/// StringToNumber (ECMA-262 #sec-stringtonumber): the StringNumericLiteral grammar with its
/// StrWhiteSpace trimmed (empty is 0), `Infinity`, decimal literals and the `0x`/`0o`/`0b`
/// NonDecimalIntegerLiterals; anything else is NaN.
fn parse_number(s: &str) -> f64 {
    // StrUnsignedDecimalLiteral of at most 15 DecimalDigits (an index key, typically) is an
    // integer below 2^53, so its mathematical value is exactly representable.
    let digits = s.as_bytes();
    if !digits.is_empty() && digits.len() <= 15 && digits.iter().all(u8::is_ascii_digit) {
        return digits
            .iter()
            .fold(0u64, |value, &digit| value * 10 + u64::from(digit - b'0'))
            as f64;
    }
    // StrWhiteSpaceChar is WhiteSpace or LineTerminator: U+FEFF but not U+0085 (NEL), unlike
    // Rust's char::is_whitespace.
    let t = s.trim_matches(crate::regex::js_whitespace);
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => return f64::INFINITY,
        "-Infinity" => return f64::NEG_INFINITY,
        _ => {}
    }
    // Radix literals take no sign and may exceed i64, so fold digit-by-digit into an f64.
    let radix = |digits: &str, r: u32| -> f64 {
        if digits.is_empty() {
            return f64::NAN;
        }
        let mut acc = 0f64;
        for c in digits.chars() {
            match c.to_digit(r) {
                Some(d) => acc = acc * r as f64 + d as f64,
                None => return f64::NAN,
            }
        }
        acc
    };
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return radix(hex, 16);
    }
    if let Some(oct) = t.strip_prefix("0o").or_else(|| t.strip_prefix("0O")) {
        return radix(oct, 8);
    }
    if let Some(bin) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        return radix(bin, 2);
    }
    // Rust's f64 parser accepts "inf"/"nan" spellings the StrDecimalLiteral grammar doesn't,
    // so validate the shape first.
    if is_decimal_literal(t) {
        t.parse::<f64>().unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// StrDecimalLiteral (sans Infinity): `[+-]? (Digits ('.' Digits?)? | '.' Digits) ([eE][+-]?Digits)?`.
fn is_decimal_literal(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let int_digits = i - start;
    let mut frac_digits = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let fs = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        frac_digits = i - fs;
    }
    if int_digits == 0 && frac_digits == 0 {
        return false;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let es = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == es {
            return false;
        }
    }
    i == b.len()
}

#[path = "eval_prepared_reference.rs"]
mod prepared_reference;
pub(crate) use prepared_reference::{PreparedReference, PreparedReferenceSlot};

/// A member reference's key is coerced once, at GetValue/PutValue, after its base check.
enum RefKey {
    Static(String),
    Private(String),
    Coerced(crate::value::PropertyKey),
    Raw(Value),
}

/// The base and strictness of an identifier reference are captured before evaluating its RHS.
enum Reference {
    Var(PreparedReference),
    Prop(Value, RefKey),
    Super {
        proto: Value,
        receiver: Value,
        key: RefKey,
    },
}

impl Interp {
    /// Evaluate `target` to a `Reference` exactly once (its base object/binding location and,
    /// for member/index targets, its property key).
    fn resolve_reference(&mut self, target: &Expr, env: &Env) -> Result<Reference, Abrupt> {
        match target {
            Expr::Ident(name, _) => self.prepare_name_reference(name, env).map(Reference::Var),
            Expr::Member { obj, prop, .. } => {
                if matches!(**obj, Expr::Super) {
                    let proto = self.super_base(env)?;
                    let receiver = self.get_var("this", env)?;
                    return Ok(Reference::Super {
                        proto,
                        receiver,
                        key: RefKey::Static(prop.clone()),
                    });
                }
                let base = self.eval(obj, env)?;
                let key = if prop.starts_with('#') {
                    RefKey::Private(self.resolve_private(prop, env))
                } else {
                    RefKey::Static(prop.clone())
                };
                Ok(Reference::Prop(base, key))
            }
            Expr::Index { obj, index, .. } => {
                if matches!(**obj, Expr::Super) {
                    let proto = self.super_base(env)?;
                    let receiver = self.get_var("this", env)?;
                    // ToPropertyKey is deferred to GetValue/PutValue (after both sides evaluate).
                    let idx = self.eval(index, env)?;
                    return Ok(Reference::Super {
                        proto,
                        receiver,
                        key: RefKey::Raw(idx),
                    });
                }
                // The index expression is evaluated now, but ToPropertyKey is deferred to GetValue.
                let base = self.eval(obj, env)?;
                let idx = self.eval(index, env)?;
                Ok(Reference::Prop(base, RefKey::Raw(idx)))
            }
            // Annex B web compat: a CallExpression target evaluates the call, then throws.
            Expr::Call { .. } => {
                self.eval(target, env)?;
                Err(self.throw("ReferenceError", "invalid assignment target"))
            }
            // Other targets never reach compound/logical assignment (a SyntaxError at parse).
            _ => Err(self.throw("ReferenceError", "invalid assignment target")),
        }
    }

    /// Coerce a reference key via `ToPropertyKey`, caching the result so it runs at most once.
    fn coerce_ref_key(&mut self, key: &mut RefKey) -> Result<crate::value::PropertyKey, Abrupt> {
        match key {
            RefKey::Static(s) => Ok(crate::value::PropertyKey::string(s.clone())),
            RefKey::Private(s) => Ok(crate::value::PropertyKey::private(s.clone())),
            RefKey::Coerced(key) => Ok(key.clone()),
            RefKey::Raw(v) => {
                let v = *v;
                let k = self.to_property_key(&v)?;
                *key = RefKey::Coerced(k.clone());
                Ok(k)
            }
        }
    }

    /// Coerce a member reference's property key, deferring `ToPropertyKey` until after the base's
    /// RequireObjectCoercible check and caching the result so it runs at most once.
    fn ref_prop_key(
        &mut self,
        base: &Value,
        key: &mut RefKey,
    ) -> Result<crate::value::PropertyKey, Abrupt> {
        if matches!(key, RefKey::Raw(_))
            && matches!(&(base).view(), ValueView::Null | ValueView::Undefined)
        {
            return Err(self.throw("TypeError", "cannot access property of null or undefined"));
        }
        self.coerce_ref_key(key)
    }

    /// GetValue on a resolved reference.
    fn get_reference(&mut self, r: &mut Reference) -> Result<Value, Abrupt> {
        match r {
            Reference::Var(reference) => self.read_prepared_reference(reference),
            Reference::Prop(base, key) => {
                if let RefKey::Raw(raw) = &*key {
                    if let (ValueView::Obj(o), ValueView::Num(n)) = (base.view(), raw.view()) {
                        if let Some(v) = self.fast_get_elem(&o, n) {
                            return Ok(v);
                        }
                    }
                }
                let k = self.ref_prop_key(&base.clone(), key)?;
                if k.is_private() {
                    return self.get_private_member(&base.clone(), &k);
                }
                self.get_member_key(base, &k)
            }
            Reference::Super {
                proto,
                receiver,
                key,
            } => {
                // PutValue/GetValue step: ToObject(V.[[Base]]) — a null/undefined super base
                // (a null home-object prototype) throws a TypeError.
                if matches!((proto).view(), ValueView::Null | ValueView::Undefined) {
                    return Err(self.throw("TypeError", "cannot read property of null super base"));
                }
                let k = self.coerce_ref_key(key)?;
                let proto = *proto;
                let receiver = *receiver;
                self.get_member_recv(&proto, &k, receiver)
            }
        }
    }

    /// PutValue on a resolved reference.
    fn put_reference(&mut self, r: &mut Reference, value: Value) -> Result<(), Abrupt> {
        match r {
            Reference::Var(reference) => self.write_prepared_reference(reference, value),
            Reference::Prop(base, key) => {
                let mut value = value;
                if let RefKey::Raw(raw) = &*key {
                    if let (ValueView::Obj(o), ValueView::Num(n)) = (base.view(), raw.view()) {
                        match self.fast_set_elem(&o, n, value) {
                            Ok(()) => return Ok(()),
                            Err(v) => value = v,
                        }
                    }
                }
                let k = self.ref_prop_key(&base.clone(), key)?;
                if k.is_private() {
                    return self.set_private_member(&base.clone(), &k, value);
                }
                self.set_member_key(base, &k, value)
            }
            Reference::Super {
                proto,
                receiver,
                key,
            } => {
                if matches!((proto).view(), ValueView::Null | ValueView::Undefined) {
                    return Err(self.throw("TypeError", "cannot set property on null super base"));
                }
                let k = self.coerce_ref_key(key)?;
                // OrdinarySet from the super base: accessors on the base's chain win, and the
                // final write lands on the receiver.
                let proto = *proto;
                let receiver = *receiver;
                let strict = self.strict;
                let success = self.set_member_recv(&proto, &k, value, receiver)?;
                self.finish_assignment_set(&k, strict, success)
            }
        }
    }
}

/// The source spelling of a private name: strips the per-class-evaluation `\u{1}<serial>` suffix
/// from a runtime key (see `Interp::resolve_private`).
pub(crate) fn private_display(key: &str) -> &str {
    key.split('\u{1}').next().unwrap_or(key)
}

/// StringToBigInt: trimmed decimal / 0x / 0o / 0b text (empty is 0); None when unparsable.
/// StringToBigInt for comparisons with a string operand (unpolled).
fn string_to_bigint(s: &str) -> Option<crate::bigint::JsBigInt> {
    string_to_bigint_bounded(s, &mut crate::bigint::Poll::never())
        .ok()
        .flatten()
}

/// StringToBigInt (ECMA-262 §7.1.14): `Ok(None)` when the string is not a valid integer.
fn string_to_bigint_bounded(
    s: &str,
    poll: &mut crate::bigint::Poll,
) -> Result<Option<crate::bigint::JsBigInt>, crate::bigint::Stop> {
    use crate::bigint::JsBigInt;
    let t = s.trim_matches(crate::regex::js_whitespace);
    if t.is_empty() {
        return Ok(Some(JsBigInt::zero()));
    }
    for (prefixes, radix) in [(["0x", "0X"], 16), (["0o", "0O"], 8), (["0b", "0B"], 2)] {
        if let Some(digits) = prefixes.iter().find_map(|prefix| t.strip_prefix(prefix)) {
            return JsBigInt::parse_radix_bounded(digits, radix, poll);
        }
    }
    let (neg, digits) = match t.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Ok(None);
    }
    Ok(JsBigInt::parse_radix_bounded(digits, 10, poll)?.map(|v| if neg { v.neg() } else { v }))
}

/// The outcome of evaluating a `return` operand: a pending proper tail call, or a plain value.
enum TailEval {
    /// A call in tail position: callee, `this`, arguments and the call's source position.
    Tail(Value, Value, crate::rooting::RootedVec, u32),
    Val(Value),
}

impl crate::gc_trace::GcTrace for DisposeFrame {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self {
            resources,
            completion,
        } = self;
        resources.trace_gc(sink);
        match completion {
            Ok(value) => value.trace_gc(sink),
            Err(abrupt) => abrupt.trace_gc(sink),
        }
    }
}

impl crate::gc_trace::GcTrace for PendingClassElements {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self {
            instance_fields,
            private_members,
            static_elements,
        } = self;
        instance_fields.trace_gc(sink);
        private_members.trace_gc(sink);
        static_elements.trace_gc(sink);
    }
}

impl crate::gc_trace::GcTrace for StaticClassElement {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        match self {
            Self::Field(field) => field.trace_gc(sink),
            // Parsed syntax is Rust-owned; runtime function caches have their own registry.
            Self::Block(_) => {}
        }
    }
}

impl crate::gc_trace::GcTrace for PreparedDecoratorValue {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let PreparedDecoratorValue {
            callback,
            this_value,
        } = self;
        sink.value(callback);
        sink.value(this_value);
    }
}

/// Every cell the suspended evaluation holds: objects and environments, and also the Symbols of
/// computed keys and decorator receivers of any type (`@a.b` captures `a`, which may be a
/// string).
impl crate::gc_trace::GcTrace for PreparedClassEvaluation {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Self {
            outer_class_env,
            class_env,
            proto_parent,
            ctor_parent,
            derived: _,
            keys,
            decorator_values,
        } = self;
        sink.env(outer_class_env);
        sink.env(class_env);
        if let Some(parent) = proto_parent {
            sink.object(parent);
        }
        if let Some(parent) = ctor_parent {
            sink.value(parent);
        }
        for key in keys.iter().flatten() {
            key.trace_gc(sink);
        }
        if let Some(PreparedDecoratorValues { class, members }) = decorator_values {
            for decorator in class.iter().chain(members.iter().flatten()) {
                decorator.trace_gc(sink);
            }
        }
    }
}
