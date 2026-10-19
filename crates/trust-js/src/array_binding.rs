//! Allocation-free closed Array prefixes for fresh lexical destructuring.
//! ECMA-262 e28783d5: GetIteratorFromMethod, IteratorBindingInitialization, IteratorClose.
use crate::interpreter::{Abrupt, Interp};
use crate::value::Value;

impl Interp {
    /// Open a closed, non-exhausting prefix of `count` elements: under the intrinsic Array
    /// protocol GetIterator, each step reading an own data element and the final IteratorClose
    /// execute no ECMAScript code and need no iterator object (see
    /// [`crate::builtins::dense_array_binding_next`]). No step calls a function, so no collector
    /// or weak-metadata boundary runs between this proof and the last fresh binding.
    pub(crate) fn try_array_binding(
        &mut self,
        source: &Value,
        count: usize,
    ) -> Result<Option<Value>, Abrupt> {
        if self.pending_tail.is_some() {
            return Ok(None);
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
