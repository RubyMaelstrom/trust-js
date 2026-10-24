#![allow(dead_code)]
// Only the Value/trace interface is inert. The production rooting implementation is imported
// verbatim; no storage owner, registry, mutation operation or iterator is copied here.
mod value {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Value(pub u64);
}
mod gc_trace {
    use crate::value::Value;
    pub trait GcSink {
        fn word(&mut self, word: u64);
    }
    pub trait GcTrace {
        fn trace_gc(&self, sink: &mut dyn GcSink);
    }
    impl GcTrace for Value {
        fn trace_gc(&self, sink: &mut dyn GcSink) {
            sink.word(self.0);
        }
    }
    impl<T: GcTrace> GcTrace for [T] {
        fn trace_gc(&self, sink: &mut dyn GcSink) {
            for value in self {
                value.trace_gc(sink);
            }
        }
    }
    impl<T: GcTrace> GcTrace for Vec<T> {
        fn trace_gc(&self, sink: &mut dyn GcSink) {
            self.as_slice().trace_gc(sink);
        }
    }
    impl<T: GcTrace> GcTrace for std::cell::RefCell<T> {
        fn trace_gc(&self, sink: &mut dyn GcSink) {
            self.try_borrow()
                .expect("I9: mutable root loan at collection")
                .trace_gc(sink);
        }
    }
}
#[path = "../../../crates/trust-js/src/rooting.rs"]
mod rooting;

#[cfg(test)]
mod tests {
    use super::{
        gc_trace::GcSink,
        rooting::{Rooted, RootedVec},
        value::Value,
    };
    use std::cell::RefCell;
    fn roots() -> Vec<u64> {
        struct Sink(Vec<u64>);
        impl GcSink for Sink {
            fn word(&mut self, word: u64) {
                self.0.push(word);
            }
        }
        let mut sink = Sink(Vec::new());
        super::rooting::trace_rooted(&mut sink);
        sink.0.sort_unstable();
        sink.0
    }
    fn has(expected: &[u64]) {
        let words = roots();
        for word in expected {
            assert!(words.contains(word), "missing {word}: {words:?}");
        }
    }

    #[test]
    fn immutable_generic_move_drop_and_registry_free_slot_reuse() {
        let first = Rooted::new(Value(1));
        let second = Rooted::new(Value(2));
        let moved = (31, first);
        assert_eq!(roots(), vec![1, 2]);
        drop(moved);
        let replacement = Rooted::new(Value(3));
        assert_eq!(roots(), vec![2, 3]);
        drop(second);
        drop(replacement);
        assert!(roots().is_empty());
    }

    #[test]
    fn mutable_generic_refcell_short_loans_and_shared_trace() {
        let root = Rooted::new(RefCell::new(vec![Value(5)]));
        root.borrow_mut().push(Value(6));
        let shared = root.borrow();
        assert_eq!(roots(), vec![5, 6]);
        assert_eq!(shared.len(), 2);
        drop(shared);
        root.borrow_mut()[0] = Value(7);
        assert_eq!(roots(), vec![6, 7]);
        let extracted = root.into_inner().into_inner();
        assert_eq!(extracted, vec![Value(7), Value(6)]);
        assert!(roots().is_empty());
    }

    #[test]
    fn mutable_generic_loan_is_checked_before_trace_not_aliased() {
        let root = Rooted::new(RefCell::new(vec![Value(8)]));
        let mut loan = root.borrow_mut();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(roots));
        assert!(result.is_err());
        loan.push(Value(9));
        drop(loan);
        assert_eq!(roots(), vec![8, 9]);
    }

    #[test]
    fn argument_evaluation_and_callback_iterator_never_loan_payload() {
        let mut values = RootedVec::new();
        values.push(Value(10));
        values.push({
            has(&[10]);
            Value(11)
        });
        struct Reentrant {
            next: u64,
        }
        impl Iterator for Reentrant {
            type Item = Value;
            fn next(&mut self) -> Option<Value> {
                has(&[10, 11]);
                self.next += 1;
                (self.next <= 15).then_some(Value(self.next))
            }
        }
        impl Drop for Reentrant {
            fn drop(&mut self) {
                has(&[10, 11, 12, 13, 14, 15]);
            }
        }
        values.extend(Reentrant { next: 11 });
        assert_eq!(roots(), vec![10, 11, 12, 13, 14, 15]);
        assert_eq!(values.pop(), Some(Value(15)));
        assert_eq!(roots(), vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn reallocation_clone_by_value_iteration_drain_and_into_vec() {
        let mut values = RootedVec::new();
        for n in 0..256 {
            values.push(Value(n));
        }
        values.reserve(1024);
        assert_eq!(roots().len(), 256);
        let cloned = values.clone();
        assert_eq!(roots().len(), 512);
        let mut iter = cloned.into_iter();
        assert_eq!(iter.next(), Some(Value(0)));
        has(&[0, 127, 255]);
        drop(iter);
        assert_eq!(values.pop(), Some(Value(255)));
        assert_eq!(roots().len(), 255);
        let plain = values.into_vec();
        assert_eq!(plain.len(), 255);
        assert!(roots().is_empty());
    }

    #[test]
    fn unwind_and_iterator_drop_can_trace_the_registered_owner() {
        let mut values = RootedVec::from_vec(vec![Value(20)]);
        struct Panics;
        impl Iterator for Panics {
            type Item = Value;
            fn next(&mut self) -> Option<Value> {
                has(&[20]);
                panic!("iterator panic");
            }
        }
        impl Drop for Panics {
            fn drop(&mut self) {
                has(&[20]);
            }
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| values.extend(Panics)));
        assert!(result.is_err());
        assert_eq!(roots(), vec![20]);
        drop(values);
        assert!(roots().is_empty());
    }

    #[test]
    fn short_indexed_updates_and_comparator_reentry() {
        let mut values = RootedVec::from_vec(vec![Value(33), Value(31), Value(32)]);
        let left = values[0];
        let right = values[1];
        let comparison = |a: &Value, b: &Value| {
            assert_eq!(roots(), vec![31, 32, 33]);
            a.0.cmp(&b.0)
        };
        assert!(comparison(&left, &right).is_gt());
        values.set(0, right);
        values.set(1, left);
        assert_eq!(roots(), vec![31, 32, 33]);
        let mut scalar = Rooted::new(Value(40));
        scalar.replace({
            has(&[40]);
            Value(41)
        });
        has(&[41]);
    }

    #[test]
    fn borrowed_slice_and_raw_word_window_survive_root_tracing() {
        let plain = vec![Value(50), Value(51)];
        let slice = super::rooting::root_slice(&plain);
        assert_eq!(roots(), vec![50, 51]);
        drop(slice);
        let mut words = RootedVec::from_vec(vec![Value(52), Value(53)]);
        let pointer = words.as_mut_ptr();
        assert_eq!(roots(), vec![52, 53]);
        unsafe {
            pointer.add(1).write(Value(54));
        }
        assert_eq!(roots(), vec![52, 54]);
        assert_eq!(words[1], Value(54));
    }
}
