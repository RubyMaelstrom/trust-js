//! Side-effect-free OrdinaryGet resolution, followed by the actual Call boundary.
//!
//! ECMA-262 e28783d5, https://tc39.es/ecma262/#sec-ordinaryget: a cached layout
//! never caches a getter's identity or result. Read its live descriptor, root the
//! selected getter, release every object/cache borrow, then Call with the original
//! receiver. Reentrant mutation or collection cannot invalidate that owned snapshot.

use super::{Abrupt, Interp};
use crate::feedback::{CurrentPropertyTrace, PropertyOutcome};
use crate::value::{Property, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ReadKind {
    Data,
    Accessor,
    MissingGetter,
}

enum BindingGetterProbe {
    Value(Value),
    /// The structural proof still holds, but the live binding remains in its TDZ or is an import.
    UninitializedOrImport,
    /// A scope identity or structural generation changed; re-resolve the cached identifier.
    Stale,
}

/// One owned value plus lookup metadata fits in the same two words as `Value`.
/// The kind's unused discriminants also keep `Option<CachedGet>` at two words.
/// In particular, a successful data read must not acquire an out-of-line getter
/// dispatch call or a larger indirect-return buffer merely because getters cache.
pub(super) struct CachedGet {
    value: Value,
    holder_shape: u32,
    depth: u8,
    array_key_check: bool,
    kind: ReadKind,
}

impl CachedGet {
    #[inline(always)]
    pub(super) fn undefined() -> Self {
        Self::from_data(Value::Undefined)
    }

    #[inline(always)]
    fn from_data(value: Value) -> Self {
        Self {
            value,
            holder_shape: 0,
            depth: 0,
            array_key_check: false,
            kind: ReadKind::Data,
        }
    }

    /// Speculative constructor forwarding must not execute author code before
    /// committing its argument ownership. A resolved getter is still a miss there.
    #[inline(always)]
    pub(super) fn data(self) -> Option<Value> {
        if self.kind == ReadKind::Data {
            Some(self.value)
        } else {
            None
        }
    }

    /// Preserve the compact owner on a data hit. An accessor still needs its
    /// original receiver rooted across Call, so return the complete read on a miss.
    #[inline(always)]
    pub(super) fn into_data(self) -> Result<Value, Self> {
        if self.kind == ReadKind::Data {
            Ok(self.value)
        } else {
            Err(self)
        }
    }
}

impl Interp {
    #[inline(always)]
    pub(super) fn snapshot_cached_get(
        &self,
        property: &Property,
        name: &str,
        holder_shape: u32,
        depth: u8,
        array_key_check: bool,
    ) -> Option<CachedGet> {
        if !property.accessor() {
            return Some(CachedGet::from_data(property.value()));
        }
        self.snapshot_cached_accessor(property, name, holder_shape, depth, array_key_check)
    }

    #[inline(never)]
    fn snapshot_cached_accessor(
        &self,
        property: &Property,
        name: &str,
        holder_shape: u32,
        depth: u8,
        array_key_check: bool,
    ) -> Option<CachedGet> {
        // TRust JS's legacy function reflection is handled by get_from_chain, not
        // by Call(%ThrowTypeError%). Retain that extension and its strict-caller
        // censorship; ordinary user-defined getters with these names can cache.
        let getter = property.getter().cloned();
        if matches!(name, "caller" | "arguments") && self.is_throw_type_error(&getter) {
            return None;
        }
        let kind = if getter.is_some() {
            ReadKind::Accessor
        } else {
            ReadKind::MissingGetter
        };
        Some(CachedGet {
            value: (getter.unwrap_or(Value::Undefined)),
            holder_shape,
            depth,
            array_key_check,
            kind,
        })
    }

    #[inline(always)]
    pub(super) fn finish_cached_get(
        &mut self,
        read: CachedGet,
        receiver: &Value,
        trace: Option<&mut CurrentPropertyTrace>,
    ) -> Result<Value, Abrupt> {
        if read.kind == ReadKind::Data {
            Ok(read.value)
        } else {
            self.finish_cached_accessor(read, receiver, trace)
        }
    }

    #[inline(never)]
    fn finish_cached_accessor(
        &mut self,
        read: CachedGet,
        receiver: &Value,
        trace: Option<&mut CurrentPropertyTrace>,
    ) -> Result<Value, Abrupt> {
        if let Some(trace) = trace {
            trace.record(
                PropertyOutcome::Accessor,
                Some(read.holder_shape),
                read.depth,
                None,
            );
            trace.array_key_check = read.array_key_check;
        }
        if read.kind == ReadKind::MissingGetter {
            Ok(Value::Undefined)
        } else {
            let getter = read.value;
            if let Some(value) = self.try_binding_getter(&getter) {
                return Ok(value);
            }
            if let Some(result) = self.try_record_getter(&getter, receiver) {
                return result;
            }
            self.call_callback(getter, *receiver, &[])
        }
    }

    /// Validated accessor path for the JIT template's accessor landing: the template has
    /// already shape-validated the receiver's chain and the holder slot live, and no
    /// JavaScript ran since, so the descriptor at (`holder`, `slot`) is this name's
    /// OrdinaryGet resolution. Re-read the live getter there (identity is never cached),
    /// release the borrow, then run the same binding/record proofs and Call boundary as
    /// `finish_cached_accessor`. `None` declines to the ordinary resolution.
    ///
    /// # Safety
    /// `holder` must be the `Gc::as_ptr` of an object on `receiver`'s live prototype
    /// chain (or the receiver itself), published by the template during this synchronous
    /// helper call. The caller roots `receiver`, which transitively keeps the holder
    /// alive for the read, exactly as in `ic_shape_probe`'s raw chain walk.
    pub(crate) unsafe fn accessor_validated_get(
        &mut self,
        holder: usize,
        slot: u32,
        name: &str,
        receiver: &Value,
    ) -> Option<Result<Value, Abrupt>> {
        let holder_ref = unsafe { &*(holder as *const std::cell::RefCell<crate::value::Object>) };
        let getter = {
            let b = holder_ref.try_borrow().ok()?;
            let p = b.props.property_at(slot as usize)?;
            if !p.accessor() {
                return None;
            }
            p.getter().cloned()
        };
        #[cfg(test)]
        TEST_VALIDATED_ACCESSORS.with(|count| count.set(count.get() + 1));
        // Retain the legacy ThrowTypeError censorship from snapshot_cached_accessor.
        if matches!(name, "caller" | "arguments") && self.is_throw_type_error(&getter) {
            return None;
        }
        let Some(getter) = getter else {
            return Some(Ok(Value::Undefined));
        };
        if let Some(value) = self.try_binding_getter(&getter) {
            return Some(Ok(value));
        }
        if let Some(result) = self.try_record_getter(&getter, receiver) {
            return Some(result);
        }
        Some(self.call_callback(getter, *receiver, &[]))
    }

    /// A getter whose entire body is `return <identifier>` — the export accessors module
    /// bundlers emit (`() => binding`) — observes nothing but that binding. When the name
    /// resolves, before any object Environment Record, to an initialized ordinary declarative
    /// binding, its value is exactly the result of Call(getter, receiver): no parameters,
    /// declarations, `this` or `arguments` are involved (ECMA-262 §10.2.1.3
    /// OrdinaryCallEvaluateBody, §9.1.2.1 GetIdentifierReference, §9.1.1.1.6 GetBindingValue).
    /// TDZ bindings, imports, `with` scopes and global lookups keep the real Call.
    pub(crate) fn try_binding_getter(&self, getter: &Value) -> Option<Value> {
        use crate::ast::{Expr, Stmt};
        use crate::value::Callable;
        let getter_object = getter.as_obj()?;
        let getter_object = getter_object.borrow();
        let Callable::User(user) = &getter_object.call else {
            return None;
        };

        // The first successful proof retains the identifier and a bounded structural proof for
        // this closure's own environment. The AST is shared by closures and Realms, so dynamic
        // binding state belongs on UserCallable and is guarded by the live environment chain.
        if let Some(cache) = user.take_binding_getter_cache() {
            match self.probe_binding_getter_cache(&user.env, &cache) {
                BindingGetterProbe::Value(value) => {
                    user.put_binding_getter_cache(Some(cache));
                    #[cfg(test)]
                    record_binding_getter(1);
                    return Some(value);
                }
                BindingGetterProbe::UninitializedOrImport => {
                    // The structural proof still holds. Read TDZ/import state live on every
                    // access, and retain the cache so a later ordinary binding write can hit.
                    user.put_binding_getter_cache(Some(cache));
                    return None;
                }
                BindingGetterProbe::Stale => {
                    if let Some((next, value)) =
                        self.resolve_binding_getter(&user.env, cache.name.as_ref())
                    {
                        user.put_binding_getter_cache(Some(next));
                        #[cfg(test)]
                        record_binding_getter(2);
                        return Some(value);
                    }
                    user.put_binding_getter_cache(None);
                    return None;
                }
            }
        }

        let f = &user.func;
        if !f.params.is_empty() || f.is_generator || f.is_async || f.body.len() != 1 {
            return None;
        }
        let Stmt::Return(Some(Expr::Ident(name, _))) = &f.body[0] else {
            return None;
        };
        if name == "arguments" {
            return None;
        }
        let (cache, value) = self.resolve_binding_getter(&user.env, name)?;
        user.put_binding_getter_cache(Some(cache));
        #[cfg(test)]
        record_binding_getter(0);
        Some(value)
    }

    /// A cache is a bounded proof of the full resolution path. Parent links are followed live,
    /// every scope identity is compared, and every VarMap generation is checked before the raw
    /// Binding pointer is read. The closure's [[Environment]] owns the first scope and each live
    /// parent link owns the next; the pointer cannot outlive or move within the guarded map.
    fn probe_binding_getter_cache(
        &self,
        start: &crate::interpreter::Env,
        cache: &crate::interpreter::BindingGetterCache,
    ) -> BindingGetterProbe {
        use crate::interpreter::MAX_BINDING_GETTER_SCOPES;

        if cache.scopes.is_empty() || cache.scopes.len() > MAX_BINDING_GETTER_SCOPES {
            return BindingGetterProbe::Stale;
        }
        let mut scope_ptr = crate::interpreter::Env::as_ptr(start);
        for (index, guard) in cache.scopes.iter().enumerate() {
            if scope_ptr != guard.pin.as_ptr() {
                return BindingGetterProbe::Stale;
            }
            // SAFETY: the called UserCallable owns `start`, and each parent link owns the next
            // scope. No JS or GC operation runs in this guarded walk, so those links stay live.
            let scope = unsafe { &*scope_ptr }.borrow();
            // The previous optimization deliberately excludes global and with environments.
            // Rechecking the live parent and with state also covers scope-chain edits.
            if scope.parent.is_none()
                || scope.has_with_object()
                || scope.under_with
                || !scope.vars.matches_generation(guard.generation)
            {
                return BindingGetterProbe::Stale;
            }
            if index + 1 == cache.scopes.len() {
                // SAFETY: resolution installed this pointer from this scope's VarMap. The live
                // identity and unchanged non-saturated generation above prove no structural
                // mutation has moved or removed it. In-place binding changes are inspected live.
                let binding = unsafe { &*cache.binding };
                return if binding.initialized && !binding.imported && binding.import_ref.is_none() {
                    BindingGetterProbe::Value(binding.value())
                } else {
                    BindingGetterProbe::UninitializedOrImport
                };
            }
            let Some(parent) = scope.parent.as_ref() else {
                return BindingGetterProbe::Stale;
            };
            scope_ptr = crate::interpreter::Env::as_ptr(parent);
        }
        BindingGetterProbe::Stale
    }

    /// Cold resolution performs ordinary lexical lookup once, then stores only scope identities,
    /// generations, the immutable name and the selected Binding address. Dynamic binding values
    /// and TDZ/import flags are never cached. Deep or observable environments retain Call.
    fn resolve_binding_getter(
        &self,
        start: &crate::interpreter::Env,
        name: &str,
    ) -> Option<(crate::interpreter::BindingGetterCache, Value)> {
        use crate::interpreter::{BindingGetterCache, BindingGetterScopeGuard};
        use std::rc::Rc;

        // Failed proofs are common for module imports and globals. Resolve with raw
        // pointers first so those getters keep the allocation-free Call fallback. The
        // captured environment owns the first scope and every live parent link owns the
        // next; no JavaScript or GC boundary occurs during this bounded walk.
        let mut scope_ptr = crate::interpreter::Env::as_ptr(start);
        let mut scope_count = None;
        for depth in 1..=crate::interpreter::MAX_BINDING_GETTER_SCOPES {
            // SAFETY: `start` and each traversed scope's parent keep this chain alive for
            // the duration of the walk; the function invokes no user code or collector.
            let scope = unsafe { &*scope_ptr }.borrow();
            // A global Environment Record's object part and with environments may run code.
            // Saturated generations cannot validate a stable Binding address.
            if scope.parent.is_none()
                || scope.has_with_object()
                || scope.under_with
                || scope.vars.generation() == u32::MAX
            {
                return None;
            }
            if let Some(binding) = scope.vars.get(name) {
                if !binding.initialized || binding.imported || binding.import_ref.is_some() {
                    return None;
                }
                scope_count = Some(depth);
                break;
            }
            scope_ptr = crate::interpreter::Env::as_ptr(scope.parent.as_ref()?);
        }
        let scope_count = scope_count?;

        // Only a successful initialized ordinary binding justifies owning cache metadata.
        // Rewalk the proven path to take weak guards and the live Binding/value snapshot;
        // there is still no callout or possible mutation between the two traversals.
        let mut scopes = Vec::with_capacity(scope_count);
        let mut env = *start;
        let mut binding_ptr = std::ptr::null();
        let mut value = None;
        for index in 0..scope_count {
            let scope = env.borrow();
            if scope.parent.is_none()
                || scope.has_with_object()
                || scope.under_with
                || scope.vars.generation() == u32::MAX
            {
                return None;
            }
            scopes.push(BindingGetterScopeGuard {
                pin: crate::interpreter::Env::downgrade(&env),
                generation: scope.vars.generation(),
            });
            if index + 1 == scope_count {
                let binding = scope.vars.get(name)?;
                if !binding.initialized || binding.imported || binding.import_ref.is_some() {
                    return None;
                }
                binding_ptr = binding as *const crate::interpreter::Binding;
                value = Some(binding.value());
                break;
            }
            let parent = *scope.parent.as_ref()?;
            drop(scope);
            env = parent;
        }
        Some((
            BindingGetterCache {
                name: Rc::<str>::from(name),
                scopes,
                binding: binding_ptr,
            },
            value?,
        ))
    }

    /// Resolve the live getter first, then prove its complete body and every
    /// receiver field. No observer can see an omitted logical frame on a hit;
    /// getters, proxies, conversions, throws and all other bodies keep Call.
    fn try_record_getter(
        &mut self,
        getter: &Value,
        receiver: &Value,
    ) -> Option<Result<Value, Abrupt>> {
        use crate::interpreter::{
            execution_stack_exhausted, with_execution_stack, GC_CALL_POLL_MASK,
        };
        use crate::value::Callable;
        if self.tier != crate::bytecode::Tier::Jit || self.pending_tail.is_some() {
            return None;
        }
        let getter = getter.as_obj()?;
        let receiver = receiver.as_obj()?;
        let key = crate::value::Gc::as_ptr(&getter) as usize;
        if !self.ordinary_get_ptr(key) {
            return None;
        }
        let chunk = {
            let getter = getter.borrow();
            let Callable::User(user) = &getter.call else {
                return None;
            };
            let f = &user.func;
            if user.realm != crate::value::Gc::as_ptr(&self.global) as usize
                || !f.params.is_empty()
                || f.is_arrow
                || f.is_async
                || f.is_generator
            {
                return None;
            }
            let runtime = f.runtime();
            let chunk = runtime.code.get()?.as_ref()?;
            // Most getters have another body. Reject the cached negative proof
            // before acquiring a chunk owner or inspecting call/receiver state.
            if !chunk.is_record_getter() {
                return None;
            }
            chunk.clone()
        };
        if !self.ordinary_get_ptr(crate::value::Gc::as_ptr(&receiver) as usize)
            || getter.borrow().class_kind != 0
            || self.gc_tick.wrapping_add(1) & GC_CALL_POLL_MASK == 0
            || crate::value::heap_live_objects(&self.gc_heap) > self.gc_next
            || execution_stack_exhausted(self.depth + 1)
        {
            return None;
        }
        let values = {
            let receiver = receiver.borrow();
            if !receiver.ic_plain.get() || !receiver.exotic.is_ordinary() {
                return None;
            }
            chunk.record_getter_values(&receiver.props)?
        };
        self.depth += 1;
        self.gc_tick = self.gc_tick.wrapping_add(1);
        let result = with_execution_stack(self.depth, || {
            self.interrupt_poll_force()?;
            #[cfg(test)]
            TEST_RECORD_GETTERS.with(|count| count.set(count.get() + 1));
            Ok(chunk.make_record_getter_result(self, values))
        });
        self.depth -= 1;
        Some(result)
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_RECORD_GETTERS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    pub(super) static TEST_VALIDATED_ACCESSORS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    pub(super) static TEST_BINDING_GETTERS: std::cell::Cell<[usize; 3]> = const {
        std::cell::Cell::new([0; 3])
    };
}

#[cfg(test)]
fn record_binding_getter(path: usize) {
    TEST_BINDING_GETTERS.with(|counts| {
        let mut next = counts.get();
        next[path] += 1;
        counts.set(next);
    });
}
