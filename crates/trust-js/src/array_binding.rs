//! Allocation-free closed Array prefixes for fresh lexical destructuring.
//! ECMA-262 e28783d5: GetIteratorFromMethod, IteratorBindingInitialization, IteratorClose.
use crate::interpreter::{Abrupt, Interp};
use crate::value::{Callable, Exotic, Value};
use std::rc::Rc;

impl Interp {
    /// A cloned slot owner is supplied here: two owners can still mean that replacing the
    /// slot will destroy its object when this temporary drops. Keep that observer on the
    /// ordinary protocol, before the next indexed read.
    pub(crate) fn array_binding_old_slot_is_inert(value: &Value) -> bool {
        match value {
            Value::Obj(object) => {
                if Rc::strong_count(object) > 2 {
                    return true;
                }
                let object = object.borrow();
                let Exotic::ArrayIterator(state) = &object.exotic else {
                    return false;
                };
                let shared_edges = match (&object.proto, &state.target) {
                    (Some(proto), Value::Obj(target)) if Rc::ptr_eq(proto, target) => 2,
                    _ => 1,
                };
                // A preceding due-GC miss may leave the actual intrinsic iterator in this
                // slot. Its empty shell is inert if dropping target/prototype cannot be final.
                matches!(object.call, Callable::None)
                    && object.native_typed_array.is_none()
                    && object.props.values().next().is_none()
                    && object
                        .proto
                        .as_ref()
                        .is_none_or(|proto| Rc::strong_count(proto) > shared_edges)
                    && match &state.target {
                        Value::Obj(target) => Rc::strong_count(target) > shared_edges,
                        _ => true,
                    }
            }
            _ => true,
        }
    }

    /// Open a closed, non-exhausting prefix of `count` elements: under the intrinsic Array
    /// protocol GetIterator, each step reading an own data element and the final IteratorClose
    /// execute no ECMAScript code and need no iterator object (see
    /// [`crate::builtins::dense_array_binding_next`]). No step calls a function, so no collector
    /// or weak-metadata boundary runs between this proof and the last fresh binding.
    pub(crate) fn try_array_binding(
        &mut self,
        source: &Value,
        count: usize,
        old_source: &Value,
        old_next: &Value,
    ) -> Result<Option<Value>, Abrupt> {
        if self.pending_tail.is_some()
            || !Self::array_binding_old_slot_is_inert(old_source)
            || !Self::array_binding_old_slot_is_inert(old_next)
        {
            return Ok(None);
        }
        // A custom iterator may also be its own captured next function. Retiring both
        // slots and both temporary clones together must leave an independent owner.
        if matches!((old_source, old_next), (Value::Obj(source), Value::Obj(next))
            if Rc::ptr_eq(source, next) && Rc::strong_count(source) <= 4)
        {
            return Ok(None);
        }
        if let (Value::Obj(source), Value::Obj(next)) = (old_source, old_next) {
            let source = source.borrow();
            if let Exotic::ArrayIterator(state) = &source.exotic {
                let target_edge = usize::from(
                    matches!(&state.target, Value::Obj(target) if Rc::ptr_eq(target, next)),
                );
                let prototype_edge = usize::from(
                    source
                        .proto
                        .as_ref()
                        .is_some_and(|proto| Rc::ptr_eq(proto, next)),
                );
                let retired_edges = target_edge + prototype_edge;
                if retired_edges != 0 && Rc::strong_count(next) <= retired_edges + 2 {
                    return Ok(None);
                }
            }
        }
        let Some(next) = crate::builtins::dense_array_binding_next(self, source, count) else {
            return Ok(None);
        };
        #[cfg(test)]
        TEST_ARRAY_BINDING_OPENS.with(|count| count.set(count.get() + 1));
        Ok(Some(next))
    }

    /// One step of an opened prefix: Get(source, index) of an element the proof found to be
    /// an own data property.
    pub(crate) fn array_binding_step(
        &mut self,
        source: &Value,
        index: usize,
    ) -> Result<Value, Abrupt> {
        // No intervening initialization can execute JS. Keep the checked read available
        // rather than introduce raw layouts.
        self.fast_get_elem(source.as_obj().expect("binding Array"), index as f64)
            .map(Ok)
            .unwrap_or_else(|| self.get_member_key(source, &index.to_string()))
    }

    /// The iterator walk of a flat array pattern (`Op::DestructureArr`): `count` element values
    /// of an ordinary Array under the intrinsic iteration protocol, without creating the Array
    /// Iterator. GetIterator and every step that reads an own data element execute no
    /// ECMAScript code, and nothing can observe the iterator before IteratorClose, which the
    /// protocol proof makes a no-op. A hole or accessor element takes the specified
    /// Get(array, index); its code may install a `return` method, so the close then runs on the
    /// iterator the walk stands for. Each step re-reads the length, as the iterator does.
    /// `None` means the proof failed before any effect.
    pub(crate) fn destructure_array_prefix(
        &mut self,
        source: &Value,
        count: usize,
        mut push: impl FnMut(Value),
    ) -> Option<Result<(), Abrupt>> {
        let mut index = match self.inert_array_prefix(source, count, &mut push)? {
            Ok(()) => return Some(Ok(())),
            Err(index) => index,
        };
        let array = source.as_obj()?;
        let mut done = false;
        let remaining = count.saturating_sub(index);
        for _ in 0..remaining {
            if !done {
                if index < self.array_length(array) {
                    let value = match self.fast_get_elem(array, index as f64) {
                        Some(value) => value,
                        None => match self.get_member_key(source, &index.to_string()) {
                            Ok(value) => value,
                            Err(error) => return Some(Err(error)),
                        },
                    };
                    index += 1;
                    push(value);
                    continue;
                }
                done = true;
            }
            push(Value::Undefined);
        }
        if !done {
            let iterator = crate::builtins::array_values_iterator(self, source.clone(), index);
            return Some(self.iterator_close_normal(&iterator));
        }
        Some(Ok(()))
    }

    /// The part of [`Self::destructure_array_prefix`] that executes no ECMAScript code: the
    /// leading own data elements, read under one borrow, then `undefined` once the iterator is
    /// exhausted. `Err(index)` stops before the element at `index`, whose Get may run code.
    pub(crate) fn inert_array_prefix(
        &self,
        source: &Value,
        count: usize,
        mut push: impl FnMut(Value),
    ) -> Option<Result<(), usize>> {
        if !crate::builtins::array_iteration_is_intrinsic(self, source) {
            return None;
        }
        let body = source.as_obj()?.borrow();
        let length = crate::builtins::ordinary_array_length(&body.props)?;
        let mut index = 0;
        while index < count.min(length) {
            let Some(value) = crate::builtins::own_data_element(&body.props, index) else {
                return Some(Err(index));
            };
            push(value);
            index += 1;
        }
        (index..count).for_each(|_| push(Value::Undefined));
        Some(Ok(()))
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_ARRAY_BINDING_OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(crate) static TEST_JIT_ARRAY_BINDING_OPENS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
