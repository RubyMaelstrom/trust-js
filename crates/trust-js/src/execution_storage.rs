//! Owned execution storage. JavaScript abstract operations and host APIs still consume `Value`;
//! persistent local/operand owners use `PackedValue` in both the VM and native backends. Only
//! actual argument lists cross the public wide-Value API; whole activations are never widened.

use crate::value::{PackedValue, Value};

pub(crate) trait StoredValue: Sized + Clone {
    fn from_value(value: Value) -> Self;
    fn into_value(self) -> Value;
    fn clone_value(&self) -> Value;
    fn is_empty(&self) -> bool;
    fn with_values<R>(values: &[Self], f: impl FnOnce(&[Value]) -> R) -> R;
}

impl StoredValue for PackedValue {
    #[inline]
    fn from_value(value: Value) -> Self {
        Self::pack(value)
    }
    #[inline]
    fn into_value(self) -> Value {
        PackedValue::into_value(self)
    }
    #[inline]
    fn clone_value(&self) -> Value {
        self.unpack()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        PackedValue::is_empty(self)
    }
    #[inline]
    fn with_values<R>(values: &[Self], f: impl FnOnce(&[Value]) -> R) -> R {
        // Decode only the actual argument list, never a whole activation. Common arities stay
        // on the Rust stack; each temporary owns its handle across arbitrary JS/GC reentry.
        match values {
            [] => return f(&[]),
            [a] => return f(&[a.unpack()]),
            [a, b] => return f(&[a.unpack(), b.unpack()]),
            [a, b, c] => return f(&[a.unpack(), b.unpack(), c.unpack()]),
            [a, b, c, d] => return f(&[a.unpack(), b.unpack(), c.unpack(), d.unpack()]),
            _ => {}
        }
        let arguments: Vec<Value> = values.iter().map(PackedValue::unpack).collect();
        f(&arguments)
    }
}

/// Move `argc` owned packed words starting at `args` into wide values for the duration of `f`
/// (the caller forgets the words). Common arities stay on the Rust stack.
///
/// # Safety
/// `args..args + argc` must be initialized, owned words that the caller never reads or drops
/// again.
pub(crate) unsafe fn with_moved_values<R>(
    args: *mut PackedValue,
    argc: usize,
    f: impl FnOnce(&[Value]) -> R,
) -> R {
    let take = |k: usize| unsafe { args.add(k).read() }.into_value();
    match argc {
        0 => f(&[]),
        1 => f(&[take(0)]),
        2 => f(&[take(0), take(1)]),
        3 => f(&[take(0), take(1), take(2)]),
        4 => f(&[take(0), take(1), take(2), take(3)]),
        _ => {
            let values: Vec<Value> = (0..argc).map(take).collect();
            f(&values)
        }
    }
}

pub(crate) trait SlotAccess {
    fn read_value(&self, index: usize) -> Value;
    fn write_value(&mut self, index: usize, value: Value);
}

impl<S: StoredValue> SlotAccess for [S] {
    #[inline]
    fn read_value(&self, index: usize) -> Value {
        self[index].clone_value()
    }
    #[inline]
    fn write_value(&mut self, index: usize, value: Value) {
        self[index] = S::from_value(value);
    }
}

pub(crate) struct ValueStack<S: StoredValue>(Vec<S>);

impl<S: StoredValue> Default for ValueStack<S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<S: StoredValue> ValueStack<S> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }
    #[inline]
    pub(crate) fn push(&mut self, value: Value) {
        self.0.push(S::from_value(value));
    }
    #[inline]
    pub(crate) fn push_stored(&mut self, value: S) {
        self.0.push(value);
    }
    #[inline]
    pub(crate) fn pop_stored(&mut self) -> Option<S> {
        self.0.pop()
    }
    #[inline]
    pub(crate) fn last_stored(&self) -> Option<&S> {
        self.0.last()
    }
    #[inline]
    pub(crate) fn pop(&mut self) -> Option<Value> {
        self.0.pop().map(S::into_value)
    }
    #[inline]
    pub(crate) fn last(&self) -> Option<Value> {
        self.0.last().map(S::clone_value)
    }
    #[inline]
    pub(crate) fn read_value(&self, index: usize) -> Value {
        self.0[index].clone_value()
    }
    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<Value> {
        self.0.get(index).map(S::clone_value)
    }
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.0.capacity()
    }
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
    #[inline]
    pub(crate) fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }
    #[inline]
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }
    #[cfg(test)]
    #[inline]
    pub(crate) fn extend(&mut self, values: impl IntoIterator<Item = Value>) {
        self.0.extend(values.into_iter().map(S::from_value));
    }
    pub(crate) fn split_off(&mut self, at: usize) -> Vec<Value> {
        self.0.drain(at..).map(S::into_value).collect()
    }
    pub(crate) fn with_tail<R>(&self, at: usize, f: impl FnOnce(&[Value]) -> R) -> R {
        S::with_values(&self.0[at..], f)
    }
    pub(crate) fn raw_slice(&self) -> &[S] {
        &self.0
    }
}

// These raw transfers only connect native code to the same representation of the VM adapter.
// There is no cast from packed owners to Value. Capacity/initialization ownership is the caller's.
impl<S: StoredValue> ValueStack<S> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut S {
        self.0.as_mut_ptr()
    }
    pub(crate) fn as_ptr(&self) -> *const S {
        self.0.as_ptr()
    }
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        unsafe { self.0.set_len(len) }
    }
}

pub(crate) type CompactFrame = (Vec<PackedValue>, ValueStack<PackedValue>);

/// Borrow an actual call's owners while seeding an activation. Only captured parameters need
/// decoding; an arguments object can still request the complete, owning public Value list.
/// The caller keeps every input initialized and rooted until instantiation finishes.
#[derive(Clone, Copy)]
pub(crate) enum CallArgs<'a> {
    Values(&'a [Value]),
    Packed(&'a [PackedValue]),
}

impl CallArgs<'_> {
    pub(crate) fn read(&self, index: usize) -> Value {
        match self {
            Self::Values(values) => values.get(index).cloned(),
            Self::Packed(values) => values.get(index).map(PackedValue::unpack),
        }
        .unwrap_or(Value::Undefined)
    }

    pub(crate) fn with_values<R>(&self, f: impl FnOnce(&[Value]) -> R) -> R {
        match self {
            Self::Values(values) => f(values),
            Self::Packed(values) => PackedValue::with_values(values, f),
        }
    }

    /// Owned copies of the arguments from `first` on (a rest parameter's list).
    pub(crate) fn rest_values(&self, first: usize) -> Vec<Value> {
        match self {
            Self::Values(values) => values.get(first..).unwrap_or(&[]).to_vec(),
            Self::Packed(values) => values
                .get(first..)
                .unwrap_or(&[])
                .iter()
                .map(PackedValue::unpack)
                .collect(),
        }
    }
}

/// A bounded argument-boundary decode, never an activation conversion. Most JS/native calls
/// fit four arguments and need no heap allocation; all temporary handles remain rooted until
/// the called operation returns or throws.
pub(crate) enum DecodedArgs {
    Inline { values: [Value; 4], len: usize },
    Heap(Vec<Value>),
}

impl DecodedArgs {
    pub(crate) fn new(arguments: &[PackedValue]) -> Self {
        if arguments.len() <= 4 {
            let mut values = std::array::from_fn(|_| Value::Undefined);
            for (value, argument) in values.iter_mut().zip(arguments) {
                *value = argument.unpack();
            }
            Self::Inline {
                values,
                len: arguments.len(),
            }
        } else {
            Self::Heap(arguments.iter().map(PackedValue::unpack).collect())
        }
    }
}

impl std::ops::Deref for DecodedArgs {
    type Target = [Value];
    fn deref(&self) -> &[Value] {
        match self {
            Self::Inline { values, len } => &values[..*len],
            Self::Heap(values) => values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bytecode::Tier, Completion, Engine};
    use std::cell::RefCell;
    use std::rc::Rc;

    thread_local! {
        static BOUNDARY_KEY_WEAK: RefCell<Option<std::rc::Weak<RefCell<crate::value::Object>>>> = const {
            RefCell::new(None)
        };
        static BOUNDARY_RECEIVER_WEAK: RefCell<Option<std::rc::Weak<RefCell<crate::value::Object>>>> = const {
            RefCell::new(None)
        };
    }

    fn boundary_gc_then_call(
        interp: &mut crate::interpreter::Interp,
        this: Value,
        args: &[Value],
    ) -> Result<Value, Value> {
        let Value::Obj(receiver) = &this else {
            panic!("native receiver owner")
        };
        assert_eq!(
            Rc::strong_count(receiver),
            2,
            "packed receiver plus native value"
        );
        let Value::Obj(key) = &args[1] else {
            panic!("native argument owner")
        };
        assert_eq!(
            Rc::strong_count(key),
            1,
            "argument moved without a second owner"
        );
        interp.gc_collect();
        assert_eq!(Rc::strong_count(key), 1, "argument remains a GC root");
        match interp.call(args[0].clone(), Value::Undefined, &[args[1].clone()]) {
            Ok(value) => Ok(value),
            Err(crate::interpreter::Abrupt::Throw(error)) => Err(error),
            Err(_) => panic!("callback produced a non-throw completion"),
        }
    }

    fn boundary_tail_probe(
        _interp: &mut crate::interpreter::Interp,
        _this: Value,
        _args: &[Value],
    ) -> Result<Value, Value> {
        let key_count = BOUNDARY_KEY_WEAK.with(|weak| {
            weak.borrow()
                .as_ref()
                .expect("key weak handle")
                .strong_count()
        });
        let receiver_count = BOUNDARY_RECEIVER_WEAK.with(|weak| {
            weak.borrow()
                .as_ref()
                .expect("receiver weak handle")
                .strong_count()
        });
        assert_eq!(
            key_count, 1,
            "moved argument stays rooted through tail drain"
        );
        assert_eq!(
            receiver_count, 1,
            "original receiver owner stays through tail drain"
        );
        Ok(Value::Undefined)
    }

    fn boundary_enqueue_tail(
        interp: &mut crate::interpreter::Interp,
        _this: Value,
        args: &[Value],
    ) -> Result<Value, Value> {
        let Value::Obj(key) = &args[0] else {
            panic!("native argument owner")
        };
        assert_eq!(
            Rc::strong_count(key),
            1,
            "argument moved without a second owner"
        );
        let tail = Value::Obj(interp.make_native("boundaryTailProbe", 0, boundary_tail_probe));
        interp.pending_tail = Some(Box::new((tail, Value::Undefined, Vec::new())));
        Ok(Value::Undefined)
    }

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("storage fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
        }
    }

    #[test]
    fn compact_storage_preserves_scalar_bits_and_heap_ownership() {
        assert_eq!(std::mem::size_of::<PackedValue>(), 8);
        let interp = crate::interpreter::Interp::new();
        let object = interp.new_object();
        let baseline = Rc::strong_count(&object);
        let mut stack = ValueStack::<PackedValue>::default();
        stack.push(Value::Obj(object.clone()));
        assert_eq!(Rc::strong_count(&object), baseline + 1);
        stack.with_tail(0, |args| {
            assert!(matches!(&args[0], Value::Obj(actual) if Rc::ptr_eq(actual, &object)));
            assert_eq!(Rc::strong_count(&object), baseline + 2);
        });
        assert_eq!(Rc::strong_count(&object), baseline + 1);
        stack.push(Value::Num(-0.0));
        assert!(
            matches!(stack.pop(), Some(Value::Num(value)) if value.to_bits() == (-0.0f64).to_bits())
        );
        for number in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfffb_0000_0000_1234),
        ] {
            stack.push(Value::Num(number));
            assert!(matches!(stack.pop(), Some(Value::Num(value)) if
                (number.is_nan() && value.is_nan()) || value.to_bits() == number.to_bits()));
        }
        let moved = stack.split_off(0);
        assert!(stack.is_empty());
        assert_eq!(Rc::strong_count(&object), baseline + 1);
        drop(moved);
        assert_eq!(Rc::strong_count(&object), baseline);

        let mut slots = [PackedValue::pack(Value::Obj(object.clone()))];
        slots.write_value(0, Value::Empty);
        assert_eq!(Rc::strong_count(&object), baseline);
        assert!(matches!(slots.read_value(0), Value::Empty));
        slots.write_value(0, Value::Undefined);
        assert!(matches!(slots.read_value(0), Value::Undefined));
    }

    #[test]
    fn compact_storage_decodes_only_argument_owners_with_exact_lifetime() {
        let object = crate::value::Object::new(None);
        let baseline = Rc::strong_count(&object);
        for count in 0..=8 {
            let packed: Vec<_> = (0..count)
                .map(|_| PackedValue::pack(Value::Obj(object.clone())))
                .collect();
            let decoded = DecodedArgs::new(&packed);
            assert_eq!(decoded.len(), count);
            assert_eq!(Rc::strong_count(&object), baseline + count * 2);
            assert!(matches!(&decoded, DecodedArgs::Inline { .. }) == (count <= 4));
            assert!(decoded
                .iter()
                .all(|value| matches!(value, Value::Obj(actual) if Rc::ptr_eq(actual, &object))));
            drop(decoded);
            assert_eq!(Rc::strong_count(&object), baseline + count);
            drop(packed);
            assert_eq!(Rc::strong_count(&object), baseline);
        }
    }

    #[test]
    fn compact_native_arguments_move_without_refcount_churn_and_stay_live_for_the_call() {
        let object = crate::value::Object::new(None);
        let weak = Rc::downgrade(&object);
        let baseline = weak.strong_count();
        for count in [0, 1, 4, 5, 9] {
            let mut packed: Vec<_> = (0..count)
                .map(|_| PackedValue::pack(Value::Obj(object.clone())))
                .collect();
            assert_eq!(weak.strong_count(), baseline + count);
            unsafe {
                with_moved_values(packed.as_mut_ptr(), count, |arguments| {
                    assert_eq!(arguments.len(), count);
                    assert_eq!(weak.strong_count(), baseline + count);
                    assert!(arguments.iter().all(
                        |value| matches!(value, Value::Obj(actual) if Rc::ptr_eq(actual, &object))
                    ));
                });
                // The packed words were moved out; the helper's unsafe contract transfers
                // responsibility for shortening the source slice back to this caller.
                packed.set_len(0);
            }
            assert_eq!(weak.strong_count(), baseline);
        }
    }

    #[test]
    fn moved_native_arguments_survive_collection_callbacks_and_abrupt_results_in_all_tiers() {
        let source = r#"
            function exercise(count) {
                var result;
                for (var run = 0; run < count; run++) {
                    var key = {}, weak = new WeakMap(), map = new Map([[key, 1]]), trace = [];
                    weak.set(key, 1);
                    var callbackError;
                    try {
                        map.forEach(function (value, entryKey, receiver) {
                            if (receiver !== map) throw 'callback receiver';
                            trace.push(value);
                            weak.set(entryKey, value + 1);
                            if (value === 1) receiver.set({}, 2);
                            if (value === 2) throw 'callback-error';
                        });
                    } catch (error) { callbackError = error; }
                    result = [weak.get(key), weak.has(key), callbackError, trace.join(',')].join(':');
                }
                return result;
            }
            exercise(120)
        "#;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                eval(&mut engine, source),
                "2:true:callback-error:1,2",
                "tier {tier:?}"
            );
        }
    }

    #[test]
    fn committed_native_arguments_remain_rooted_across_collection_callback_throw_and_tail() {
        let mut interp = crate::interpreter::Interp::new();
        let realm = Value::Obj(interp.global.clone());
        interp
            .eval_in_realm(
                &realm,
                "globalThis.boundaryCallback = function (key) { throw 'callback-error'; };",
            )
            .ok()
            .expect("callback fixture evaluates");
        let callback = interp
            .global
            .borrow()
            .props
            .get("boundaryCallback")
            .expect("callback global")
            .value();
        let receiver = interp.new_object();
        let receiver_weak = Rc::downgrade(&receiver);
        let this_slot = std::mem::ManuallyDrop::new(PackedValue::pack(Value::Obj(receiver)));
        let key = interp.new_object();
        let key_weak = Rc::downgrade(&key);
        let mut args = vec![
            PackedValue::pack(callback),
            PackedValue::pack(Value::Obj(key)),
        ];
        assert_eq!(key_weak.strong_count(), 1);
        let result = unsafe {
            interp.call_native_committed(
                boundary_gc_then_call,
                &*this_slot,
                args.as_mut_ptr(),
                args.len(),
            )
        };
        unsafe { args.set_len(0) };
        assert!(matches!(
            result,
            Err(crate::interpreter::Abrupt::Throw(Value::Str(ref value)))
                if &**value == "callback-error"
        ));
        assert_eq!(
            key_weak.strong_count(),
            0,
            "throw path releases moved argument"
        );
        assert_eq!(
            receiver_weak.strong_count(),
            0,
            "throw path releases receiver"
        );

        let key = interp.new_object();
        let key_weak = Rc::downgrade(&key);
        BOUNDARY_KEY_WEAK.with(|weak| *weak.borrow_mut() = Some(key_weak.clone()));
        let receiver = interp.new_object();
        let receiver_weak = Rc::downgrade(&receiver);
        BOUNDARY_RECEIVER_WEAK.with(|weak| *weak.borrow_mut() = Some(receiver_weak.clone()));
        let this_slot = std::mem::ManuallyDrop::new(PackedValue::pack(Value::Obj(receiver)));
        let mut args = vec![PackedValue::pack(Value::Obj(key))];
        let result = unsafe {
            interp.call_native_committed(
                boundary_enqueue_tail,
                &*this_slot,
                args.as_mut_ptr(),
                args.len(),
            )
        };
        unsafe { args.set_len(0) };
        assert!(matches!(result, Ok(Value::Undefined)));
        assert_eq!(
            key_weak.strong_count(),
            0,
            "tail path releases moved argument"
        );
        assert_eq!(
            receiver_weak.strong_count(),
            0,
            "tail path releases receiver after draining"
        );
        BOUNDARY_KEY_WEAK.with(|weak| *weak.borrow_mut() = None);
        BOUNDARY_RECEIVER_WEAK.with(|weak| *weak.borrow_mut() = None);
    }

    #[test]
    fn compact_storage_owner_moves_and_duplicates_do_not_cross_value_boundary() {
        let object = crate::value::Object::new(None);
        let weak = Rc::downgrade(&object);
        let mut slots = [
            PackedValue::pack(Value::Obj(object)),
            PackedValue::pack(Value::Empty),
        ];
        let mut stack = ValueStack::<PackedValue>::default();
        assert!(slots[1].is_empty());
        stack.push_stored(slots[0].clone());
        assert_eq!(weak.strong_count(), 2);
        stack.push_stored(stack.last_stored().unwrap().clone());
        assert_eq!(weak.strong_count(), 3);
        slots[1] = stack.pop_stored().unwrap();
        assert_eq!(
            weak.strong_count(),
            3,
            "slot move does not retain or release"
        );
        drop(stack.pop_stored());
        assert_eq!(weak.strong_count(), 2);
        slots[0] = PackedValue::pack(Value::Undefined);
        assert_eq!(weak.strong_count(), 1);
        drop(slots);
        assert_eq!(weak.strong_count(), 0);
    }

    #[test]
    fn compact_storage_raw_literals_move_packed_owners_without_refcount_churn() {
        let mut interp = crate::interpreter::Interp::new();
        let object = interp.new_object();
        let baseline = Rc::strong_count(&object);
        // Exercise inline-packed, heap-packed and large dense array factories.
        for count in [0, 1, 10, 11, 32, 33, 70] {
            let mut operands: Vec<_> = (0..count)
                .map(|_| PackedValue::pack(Value::Obj(object.clone())))
                .collect();
            let array = unsafe { interp.make_array_from_raw(operands.as_mut_ptr(), count) };
            unsafe { operands.set_len(0) };
            assert_eq!(Rc::strong_count(&object), baseline + count);
            let Value::Obj(array) = array else {
                panic!("array factory returned a primitive")
            };
            assert!(
                matches!(array.borrow().props.get("length").unwrap().value(),
                Value::Num(length) if length == count as f64)
            );
            for index in 0..count {
                assert!(
                    matches!(array.borrow().props.get_index(index as u32).unwrap().value(),
                    Value::Obj(actual) if Rc::ptr_eq(&actual, &object))
                );
            }
            drop(array);
            assert_eq!(Rc::strong_count(&object), baseline);
        }
        let template = std::cell::OnceCell::new();
        let keys = [Rc::from("held"), Rc::from("negativeZero")];
        for _ in 0..3 {
            let mut operands = std::mem::ManuallyDrop::new([
                PackedValue::pack(Value::Obj(object.clone())),
                PackedValue::pack(Value::Num(-0.0)),
            ]);
            let result = unsafe {
                interp.make_plain_object_templated_from(&template, &keys, operands.as_mut_ptr(), 2)
            };
            assert_eq!(Rc::strong_count(&object), baseline + 1);
            let Value::Obj(result) = result else {
                panic!("object factory returned a primitive")
            };
            assert!(
                matches!(result.borrow().props.get("negativeZero").unwrap().value(),
                Value::Num(number) if number.to_bits() == (-0.0f64).to_bits())
            );
            drop(result);
            assert_eq!(Rc::strong_count(&object), baseline);
        }
    }

    #[test]
    fn compact_storage_calls_destructuring_and_abrupt_completion_agree() {
        let source = r#"
            var key=Symbol('key'), object={n:3}, trace=[];
            function inspect(a,b,c,d,e,f) {
                if(a!==undefined || b!==null || c!==key || d!==object || e!==12345678901234567890n ||
                   !Object.is(f,-0)) throw 'arguments';
                var held=object, big=e, text='\uD800x';
                try {
                    var {x=held}={};
                    [big, text]=[big+2n, text+'y'];
                    if(x!==held)throw 'identity';
                    throw {held:held,big:big,text:text};
                } catch(error) {
                    return [error.held===object,String(error.big),error.text.length,error.text.charCodeAt(0)];
                } finally { trace.push('finally'); }
            }
            function throughFrame(a,b,c,d,e,f) { return inspect(a,b,c,d,e,f); }
            var result;
            for(var k=0;k<120;k++)result=throughFrame(undefined,null,key,object,12345678901234567890n,-0);
            result.join(':')+':'+trace.length;
        "#;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                eval(&mut engine, source),
                "true:12345678901234567892:3:55296:120",
                "{tier:?}"
            );
            if tier == Tier::Bytecode {
                assert!(!engine.interp.vm_pool.is_empty());
                assert!(engine
                    .interp
                    .vm_pool
                    .iter()
                    .all(|(slots, stack)| slots.is_empty() && stack.is_empty()));
            }
            engine.interp.gc_collect();
        }
    }

    #[test]
    fn compact_native_method_receivers_and_returns_preserve_every_value_kind() {
        let source = r#"
            function identity() {'use strict'; return this;}
            Boolean.prototype.identity=identity;
            Number.prototype.identity=identity;
            String.prototype.identity=identity;
            Symbol.prototype.identity=identity;
            BigInt.prototype.identity=identity;
            var symbol=Symbol('receiver'), object={identity:identity};
            var values=[false,true,0,-0,NaN,Infinity,-Infinity,'\uD800x',symbol,12345678901234567890n,object];
            function invoke(value) {return value.identity();}
            var good=true;
            for(var round=0;round<220;round++)for(var k=0;k<values.length;k++)
                good=good && Object.is(invoke(values[k]),values[k]);
            good;
        "#;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(eval(&mut engine, source), "true", "{tier:?}");
        }
    }

    #[test]
    fn compact_native_shared_calls_restore_live_lexical_parent_caches() {
        let source = r#"
            function make(value) {
                return function outer() {
                    return function inner() {return value;};
                }();
            }
            function invoke(callback) {return callback();}
            var left=make(17),right=make(29),good=true;
            for(var k=0;k<800;k++)
                good=good && invoke(left)===17 && invoke(right)===29;
            for(var k=0;k<80;k++)good=good && invoke(make(k))===k;
            good;
        "#;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(eval(&mut engine, source), "true", "{tier:?}");
        }
    }

    #[test]
    fn compact_storage_suspended_generator_keeps_operand_owners_and_finalizers() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert_eq!(
                eval(
                    &mut engine,
                    r#"
                var trace=[];
                function combine(object, value){return object.n+value}
                function* sequence(){
                    var object={n:23}, symbol=Symbol('saved'), big=9007199254740993n;
                    try {
                        var sum=combine(object, yield 'ready');
                        yield [sum, symbol.description, String(big), object.n].join(':');
                        return object;
                    } finally {trace.push(object.n);}
                }
                var iterator=sequence(); iterator.next().value;
            "#
                ),
                "ready"
            );
            engine.interp.gc_collect();
            assert_eq!(
                eval(&mut engine, "iterator.next(7).value"),
                "30:saved:9007199254740993:23"
            );
            engine.interp.gc_collect();
            assert_eq!(
                eval(
                    &mut engine,
                    r#"
                var sentinel={}; var caught=false;
                try{iterator.throw(sentinel)}catch(error){caught=error===sentinel}
                [caught,trace.join(','),iterator.next().done].join(':');
            "#
                ),
                "true:23:true"
            );
        }
    }

    #[test]
    fn compact_storage_await_keeps_live_slots_and_pending_completion() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                var release, answer, trace=[];
                async function pending(){
                    var held={n:29}, symbol=Symbol('awaited'), big=9007199254740993n;
                    try {
                        var value=held.n+await new Promise(resolve=>{release=resolve});
                        throw {held:held,value:value,symbol:symbol,big:big};
                    } catch(error) {
                        return [error.held===held,error.value,error.symbol.description,String(error.big)].join(':');
                    } finally {trace.push(held.n);}
                }
                pending().then(value=>{answer=value});
            "#,
            );
            engine.interp.gc_collect();
            eval(&mut engine, "release(11)");
            engine.interp.drain_microtasks();
            assert_eq!(
                eval(&mut engine, "answer+':'+trace.join(',')"),
                "true:40:awaited:9007199254740993:29"
            );
            engine.interp.gc_collect();
        }
    }
}
