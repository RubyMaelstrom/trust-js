//! Captured identifier References (ECMA-262 GetIdentifierReference, GetValue and PutValue;
//! local official snapshot e28783d5fc9d). Resolution precedes the RHS / ToNumeric. Neither a
//! changed key set nor a failed address hint permits resolving the name in an outer record.

use super::*;

pub(crate) const REFERENCE_SCOPE: u32 = 0;
pub(crate) const REFERENCE_OBJECT: u32 = 1;
pub(crate) const REFERENCE_UNRESOLVABLE: u32 = 2;

/// Explicit layout for native capture/load/store. Owners live in the canonical activation;
/// `binding` is only a same-record, generation-checked hint, never an independent GC root.
/// Interned bytecode names are shared without allocating a String at each resolution.
#[repr(C)]
pub(crate) struct PreparedReference {
    pub(crate) kind: u32,
    pub(crate) strict: u32,
    pub(crate) name: Rc<str>,
    pub(crate) scope: Option<Env>,
    pub(crate) object: Option<Gc>,
    pub(crate) generation: u32,
    pub(crate) shape: u32,
    pub(crate) binding: *mut Binding,
    pub(crate) slot: usize,
}

/// Canonical VM/native Reference owner. Explicit presence avoids depending on an Option niche
/// in generated code. A slot is never moved or resized while lent to a native activation.
#[repr(C)]
pub(crate) struct PreparedReferenceSlot {
    pub(crate) present: usize,
    pub(crate) reference: std::mem::MaybeUninit<PreparedReference>,
}

impl Default for PreparedReferenceSlot {
    fn default() -> Self {
        Self {
            present: 0,
            reference: std::mem::MaybeUninit::uninit(),
        }
    }
}

impl PreparedReferenceSlot {
    pub(crate) fn as_ref(&self) -> Option<&PreparedReference> {
        (self.present != 0).then(|| unsafe { self.reference.assume_init_ref() })
    }
    pub(crate) fn as_mut(&mut self) -> Option<&mut PreparedReference> {
        (self.present != 0).then(|| unsafe { self.reference.assume_init_mut() })
    }
    pub(crate) fn set(&mut self, reference: PreparedReference) {
        if std::mem::replace(&mut self.present, 0) != 0 {
            unsafe {
                self.reference.assume_init_drop();
            }
        }
        self.reference.write(reference);
        self.present = 1;
    }
}

impl Drop for PreparedReferenceSlot {
    fn drop(&mut self) {
        if self.present != 0 {
            unsafe {
                self.reference.assume_init_drop();
            }
        }
    }
}

impl<'a> IntoIterator for &'a PreparedReferenceSlot {
    type Item = &'a PreparedReference;
    type IntoIter = std::option::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_ref().into_iter()
    }
}

#[cfg(test)]
#[path = "prepared_reference_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "prepared_reference_own_tests.rs"]
mod own_tests;

#[cfg(test)]
thread_local! {
    // Global capture, refreshed hint, direct read, direct write, checked fallback.
    static TEST_REFERENCE_OWN_PATHS: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) };
}

#[cfg(test)]
fn record_reference_own_path(index: usize) {
    TEST_REFERENCE_OWN_PATHS.with(|counts| {
        let mut next = counts.get();
        next[index] += 1;
        counts.set(next);
    });
}

impl PreparedReference {
    /// The caller holds a matching live structural proof for this exact scope. This is not
    /// a new resolution: subsequent reads/writes validate the holder again after user code.
    pub(crate) fn scope_with_hint(
        name: Rc<str>,
        scope: Env,
        strict: bool,
        generation: u32,
        binding: *mut Binding,
    ) -> Self {
        Self {
            kind: REFERENCE_SCOPE,
            strict: u32::from(strict),
            name,
            scope: Some(scope),
            object: None,
            generation,
            shape: 0,
            binding,
            slot: usize::MAX,
        }
    }

    pub(crate) fn scope(name: Rc<str>, scope: Env, strict: bool) -> Self {
        let (generation, binding) = {
            let record = scope.borrow();
            (
                record.vars.generation(),
                record.vars.get(&name).map_or(std::ptr::null_mut(), |b| {
                    b as *const Binding as *mut Binding
                }),
            )
        };
        Self::scope_with_hint(name, scope, strict, generation, binding)
    }

    pub(crate) fn object(name: Rc<str>, object: Gc, strict: bool) -> Self {
        Self {
            kind: REFERENCE_OBJECT,
            strict: u32::from(strict),
            name,
            scope: None,
            object: Some(object),
            generation: u32::MAX,
            shape: 0,
            binding: std::ptr::null_mut(),
            slot: usize::MAX,
        }
    }

    pub(crate) fn unresolvable(name: Rc<str>, strict: bool) -> Self {
        Self {
            kind: REFERENCE_UNRESOLVABLE,
            strict: u32::from(strict),
            name,
            scope: None,
            object: None,
            generation: u32::MAX,
            shape: 0,
            binding: std::ptr::null_mut(),
            slot: usize::MAX,
        }
    }

    pub(crate) fn trace_gc(&self, edges: &mut crate::gc_edges::DirectGcEdges<'_>) {
        if let Some(scope) = &self.scope {
            edges.scope(scope);
        }
        if let Some(object) = &self.object {
            edges.object(object);
        }
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        visitor.rc_str(&self.name);
        if let Some(object) = &self.object {
            visitor.value(&Value::Obj(object.clone()));
        }
        // Scope storage is enumerated once by the existing scope graph census.
        0
    }
}

impl Interp {
    /// A hint for THIS saved Object Environment Record, not another name resolution.
    /// ECMA-262 OrdinaryHasProperty/Get/SetWithOwnDescriptor: an ordinary current own
    /// data property proves HasProperty=true without consulting the prototype. Descriptor
    /// flags and values remain live; capture never rejects a non-writable binding early.
    ///
    /// ordinary_get_ptr alone is insufficient for writes: mapped arguments have a separate
    /// parameter alias hook. Exclude that owner even for internal numeric-name callers.
    /// Array/string/other Exotic variants and all side-table exotics remain checked paths.
    fn refresh_reference_own_data_hint(&self, reference: &mut PreparedReference) -> bool {
        debug_assert_eq!(reference.kind, REFERENCE_OBJECT);
        let object = reference.object.as_ref().expect("object Reference owner");
        let pointer = Rc::as_ptr(object) as usize;
        let hint = if self.ordinary_get_ptr(pointer)
            && (self.mapped_arguments.is_empty() || !self.mapped_arguments.contains_key(&pointer))
        {
            let body = object.borrow();
            let shape = body.props.shape();
            if matches!(body.exotic, Exotic::None) && crate::value::is_cacheable_shape(shape) {
                let slot = if shape == reference.shape {
                    Some(reference.slot)
                } else {
                    body.props.slot_of(&reference.name)
                };
                slot.filter(|slot| body.props.property_at(*slot).is_some_and(|p| !p.accessor()))
                    .map(|slot| (shape, slot))
            } else {
                None
            }
        } else {
            None
        };
        if let Some((shape, slot)) = hint {
            #[cfg(test)]
            if reference.shape != shape || reference.slot != slot {
                record_reference_own_path(1);
            }
            reference.shape = shape;
            reference.slot = slot;
            true
        } else {
            // Never leave a stale raw hint behind after a changed descriptor or owner kind.
            reference.shape = 0;
            reference.slot = usize::MAX;
            false
        }
    }

    pub(crate) fn prepare_name_reference(
        &mut self,
        name: &str,
        env: &Env,
    ) -> Result<PreparedReference, Abrupt> {
        self.prepare_shared_name_reference(Rc::from(name), env)
    }

    pub(crate) fn prepare_shared_name_reference(
        &mut self,
        name: Rc<str>,
        env: &Env,
    ) -> Result<PreparedReference, Abrupt> {
        let strict = self.strict;
        let mut current = Some(env.clone());
        while let Some(scope) = current {
            let (has_binding, with_object, parent) = {
                let record = scope.borrow();
                (
                    record.vars.contains_key(&name),
                    record.with_obj.clone(),
                    record.parent.clone(),
                )
            };
            if has_binding {
                return Ok(PreparedReference::scope(name, scope, strict));
            }
            if let Some(Value::Obj(object)) = with_object {
                if self.with_has_binding(&Value::Obj(object.clone()), &name)? {
                    // HasBinding has already run HasProperty and @@unscopables in order.
                    // The latter may mutate the descriptor: refresh only after it completes.
                    let mut reference = PreparedReference::object(name, object, strict);
                    self.refresh_reference_own_data_hint(&mut reference);
                    return Ok(reference);
                }
            }
            current = parent;
        }
        // Capture the binding object, not a later ambient Realm's global object.
        let global = self.global.clone();
        let mut reference = PreparedReference::object(name, global, strict);
        if self.refresh_reference_own_data_hint(&mut reference) {
            #[cfg(test)]
            record_reference_own_path(0);
            Ok(reference)
        } else if self.js_has_property(
            &Value::Obj(reference.object.as_ref().unwrap().clone()),
            &reference.name,
        )? {
            // An exotic HasProperty may execute author code. Any later optimization must
            // establish a new current own-data proof, never reuse pre-effect descriptors.
            self.refresh_reference_own_data_hint(&mut reference);
            Ok(reference)
        } else {
            Ok(PreparedReference::unresolvable(reference.name, strict))
        }
    }

    pub(crate) fn read_prepared_reference(
        &mut self,
        reference: &mut PreparedReference,
    ) -> Result<Value, Abrupt> {
        match reference.kind {
            REFERENCE_SCOPE => {
                let name = &reference.name;
                let scope = reference
                    .scope
                    .as_ref()
                    .expect("declarative Reference owner");
                let (initialized, value, import) = {
                    let record = scope.borrow();
                    let binding = if record.vars.matches_generation(reference.generation)
                        && !reference.binding.is_null()
                    {
                        // The captured record is strongly owned; only its non-recycled structural
                        // generation proves this address. Exhaustion always takes the map lookup.
                        Some(unsafe { &*reference.binding })
                    } else {
                        record.vars.get(name)
                    };
                    let Some(binding) = binding else {
                        return Err(self.throw("ReferenceError", format!("{name} is not defined")));
                    };
                    reference.generation = record.vars.generation();
                    reference.binding = binding as *const Binding as *mut Binding;
                    (
                        binding.initialized,
                        binding.value.clone(),
                        binding
                            .import_ref
                            .as_deref()
                            .map(|(exporter, local)| (exporter.clone(), local.clone())),
                    )
                };
                if !initialized {
                    return Err(self.throw(
                        "ReferenceError",
                        format!("cannot access '{name}' before initialization"),
                    ));
                }
                if let Some((exporter, local)) = import {
                    return self.get_var(&local, &exporter);
                }
                Ok(value)
            }
            REFERENCE_OBJECT => {
                if self.refresh_reference_own_data_hint(reference) {
                    // No author code or structural mutation lies between proof and load.
                    let body = reference.object.as_ref().unwrap().borrow();
                    let value = body.props.property_at(reference.slot).unwrap().value();
                    #[cfg(test)]
                    record_reference_own_path(2);
                    return Ok(value);
                }
                #[cfg(test)]
                record_reference_own_path(4);
                let name = &reference.name;
                let object = Value::Obj(
                    reference
                        .object
                        .as_ref()
                        .expect("object Reference owner")
                        .clone(),
                );
                // ObjectEnvironmentRecord.GetBindingValue always does HasProperty, including in
                // sloppy code. Do not repeat HasBinding / @@unscopables after capture.
                if !self.js_has_property(&object, name)? {
                    return if reference.strict != 0 {
                        Err(self.throw("ReferenceError", format!("{name} is not defined")))
                    } else {
                        Ok(Value::Undefined)
                    };
                }
                self.get_member(&object, name)
            }
            REFERENCE_UNRESOLVABLE => {
                let name = &reference.name;
                Err(self.throw("ReferenceError", format!("{name} is not defined")))
            }
            _ => unreachable!("invalid prepared Reference kind"),
        }
    }

    pub(crate) fn write_prepared_reference(
        &mut self,
        reference: &mut PreparedReference,
        value: Value,
    ) -> Result<(), Abrupt> {
        match reference.kind {
            REFERENCE_SCOPE => {
                let name = &reference.name;
                let scope = reference
                    .scope
                    .as_ref()
                    .expect("declarative Reference owner");
                let mut record = scope.borrow_mut();
                let binding = if record.vars.matches_generation(reference.generation)
                    && !reference.binding.is_null()
                {
                    Some(unsafe { &mut *reference.binding })
                } else {
                    record.vars.get_mut(name)
                };
                if let Some(binding) = binding {
                    if binding.import_ref.is_some() {
                        return Err(self.throw(
                            "TypeError",
                            format!("assignment to import binding '{name}'"),
                        ));
                    }
                    if !binding.initialized {
                        return Err(self.throw(
                            "ReferenceError",
                            format!("cannot access '{name}' before initialization"),
                        ));
                    }
                    if !binding.mutable {
                        return if binding.strict_immutable || reference.strict != 0 {
                            Err(self.throw("TypeError", format!("assignment to constant '{name}'")))
                        } else {
                            Ok(())
                        };
                    }
                    binding.value = value;
                    reference.binding = binding;
                    reference.generation = record.vars.generation();
                    return Ok(());
                }
                if reference.strict != 0 {
                    return Err(self.throw("ReferenceError", format!("{name} is not defined")));
                }
                // Declarative SetMutableBinding's missing-binding branch: recreate here, never
                // search an outer record. This occurs when sloppy direct eval deletes its var
                // binding while its assignment RHS is executing.
                let mut binding = Binding::data(value, true, true);
                binding.deletable = true;
                record.vars.insert(name.clone(), binding);
                reference.generation = record.vars.generation();
                reference.binding = record.vars.get_mut(name).expect("new binding");
                Ok(())
            }
            REFERENCE_OBJECT => {
                if self.refresh_reference_own_data_hint(reference) {
                    // Revalidate AFTER the RHS / coercions. Writable is a live descriptor
                    // check, not a capture assumption; rejected writes retain their errors.
                    let mut body = reference.object.as_ref().unwrap().borrow_mut();
                    let (_, property) = body.props.entry_at_mut(reference.slot).unwrap();
                    if property.writable() {
                        property.set_value(value);
                        #[cfg(test)]
                        record_reference_own_path(3);
                        return Ok(());
                    }
                }
                #[cfg(test)]
                record_reference_own_path(4);
                let name = &reference.name;
                let object = Value::Obj(
                    reference
                        .object
                        .as_ref()
                        .expect("object Reference owner")
                        .clone(),
                );
                if !self.js_has_property(&object, name)? && reference.strict != 0 {
                    return Err(self.throw("ReferenceError", format!("{name} is not defined")));
                }
                self.set_reference_object(&object, name, value, reference.strict != 0)
            }
            REFERENCE_UNRESOLVABLE => {
                let name = &reference.name;
                if reference.strict != 0 {
                    return Err(self.throw("ReferenceError", format!("{name} is not defined")));
                }
                self.set_reference_object(&Value::Obj(self.global.clone()), name, value, false)
            }
            _ => unreachable!("invalid prepared Reference kind"),
        }
    }

    fn set_reference_object(
        &mut self,
        object: &Value,
        name: &str,
        value: Value,
        strict: bool,
    ) -> Result<(), Abrupt> {
        let saved = std::mem::replace(&mut self.strict, strict);
        let result = self.set_member(object, name, value);
        self.strict = saved;
        result
    }
}
