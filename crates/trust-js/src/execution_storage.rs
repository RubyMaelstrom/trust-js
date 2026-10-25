//! Canonical execution values and scoped read-only argument loans. Mutable local slots use
//! Cell so parameter aliases can replace values without an overlapping exclusive slice loan.
//! Only immutable evaluated argument windows may be loaned as Value slices; Cell-backed
//! slots and live property storage must first be copied to such a window.

use crate::value::Value;
use std::cell::Cell;

/// Loan canonical arguments for the complete operation, including any post-call polls and
/// tail calls performed by `f`. The caller retires its window only after this returns.
///
/// # Safety
/// `args` must be aligned and non-null even when `argc` is zero. `args..args + argc` must be
/// initialized execution values in an immutable evaluated argument window whose allocation is
/// neither modified, pooled nor reused until `f` returns. The caller roots the window through
/// every poll performed by `f`. Production empty windows come from Rust slices or the
/// initialized native stack and retain that canonical empty-slice address.
#[inline]
pub(crate) unsafe fn with_borrowed_values<R>(
    args: *const Value,
    argc: usize,
    f: impl FnOnce(&[Value]) -> R,
) -> R {
    f(unsafe { std::slice::from_raw_parts(args, argc) })
}

pub(crate) trait SlotAccess {
    fn read_value(&self, index: usize) -> Value;
    fn write_value(&self, index: usize, value: Value);
}

impl SlotAccess for [Cell<Value>] {
    #[inline]
    fn read_value(&self, index: usize) -> Value {
        self[index].get()
    }
    #[inline]
    fn write_value(&self, index: usize, value: Value) {
        self[index].set(value);
    }
}

/// An operand stack. Its whole capacity is initialized memory — new capacity is
/// zero-filled where the buffer is created or grown, and nothing else changes it — so native
/// code that borrows the buffer past its length (`jit::run`, `jit::run_borrowed_frame`) can
/// have the collector scan values it has not written yet without reading uninitialized memory.
#[derive(Default)]
pub(crate) struct ValueStack(Vec<Value>);

impl ValueStack {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let mut stack = Self(Vec::with_capacity(capacity));
        stack.zero_spare_capacity();
        stack
    }
    #[inline]
    pub(crate) fn push(&mut self, value: Value) {
        let len = self.0.len();
        if len == self.0.capacity() {
            self.grow(1);
        }
        // SAFETY: the capacity exceeds the length; the slot past it becomes the top.
        unsafe {
            self.0.as_mut_ptr().add(len).write(value);
            self.0.set_len(len + 1);
        }
    }
    /// Grow by at least `additional` and zero-fill the new spare capacity.
    #[cold]
    #[inline(never)]
    fn grow(&mut self, additional: usize) {
        self.0.reserve(additional);
        self.zero_spare_capacity();
    }
    fn zero_spare_capacity(&mut self) {
        let spare = self.0.spare_capacity_mut();
        // SAFETY: the spare capacity is this Vec's memory past its length; zero bytes are a
        // valid fill for it (a canonical zero word is the Number +0).
        unsafe { std::ptr::write_bytes(spare.as_mut_ptr(), 0, spare.len()) };
    }
    #[inline]
    pub(crate) fn pop(&mut self) -> Option<Value> {
        self.0.pop()
    }
    #[inline]
    pub(crate) fn last(&self) -> Option<Value> {
        self.0.last().copied()
    }
    #[inline]
    pub(crate) fn read_value(&self, index: usize) -> Value {
        self.0[index]
    }
    #[inline]
    pub(crate) fn get(&self, index: usize) -> Option<Value> {
        self.0.get(index).copied()
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
        if self.0.capacity() - self.0.len() < additional {
            self.grow(additional);
        }
    }
    #[cfg(test)]
    #[inline]
    pub(crate) fn extend(&mut self, values: impl IntoIterator<Item = Value>) {
        for value in values {
            self.push(value);
        }
    }
    pub(crate) fn split_off(&mut self, at: usize) -> Vec<Value> {
        self.0.drain(at..).collect()
    }
    pub(crate) fn with_tail<R>(&self, at: usize, f: impl FnOnce(&[Value]) -> R) -> R {
        f(&self.0[at..])
    }
    pub(crate) fn raw_slice(&self) -> &[Value] {
        &self.0
    }

    // Native code borrows the same values the VM owns; Cell-backed mutable slots remain
    // separate. Capacity and initialization of raw native storage are the caller's contract.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut Value {
        self.0.as_mut_ptr()
    }
    pub(crate) fn as_ptr(&self) -> *const Value {
        self.0.as_ptr()
    }
    pub(crate) unsafe fn set_len(&mut self, len: usize) {
        unsafe { self.0.set_len(len) }
    }
}

pub(crate) type ExecutionFrame = (Vec<Cell<Value>>, ValueStack);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::ValueView;
    use crate::{bytecode::Tier, Completion, Engine};
    use std::rc::Rc;

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("storage fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
        }
    }

    /// An operand stack and shared slots keep scalar bits (-0, infinities, NaN) and object
    /// identity through pushes, pops, an immutable argument loan and a split.
    #[test]
    fn execution_storage_preserves_scalar_bits_and_object_identity() {
        assert_eq!(std::mem::size_of::<Value>(), 8);
        let interp = crate::interpreter::Interp::new();
        let object = interp.new_object();
        let mut stack = ValueStack::default();
        stack.push(Value::Obj(object));
        let word = crate::value::PACK_OBJ | crate::value::Gc::as_ptr(&object) as u64;
        assert_eq!(stack.last().unwrap().bits(), word);
        stack.with_tail(0, |args| {
            assert!(
                matches!(&(args[0]).view(), ValueView::Obj(actual) if crate::value::Gc::ptr_eq(actual, &object))
            );
        });
        assert_eq!(stack.last().unwrap().bits(), word);
        stack.push(Value::Num(-0.0));
        assert!(
            matches!((stack.pop()).map(|__value| (__value).view()), Some(ValueView::Num(value)) if value.to_bits() == (-0.0f64).to_bits())
        );
        for number in [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfffb_0000_0000_1234),
        ] {
            stack.push(Value::Num(number));
            assert!(
                matches!((stack.pop()).map(|__value| (__value).view()), Some(ValueView::Num(value)) if
                (number.is_nan() && value.is_nan()) || value.to_bits() == number.to_bits())
            );
        }
        let moved = stack.split_off(0);
        assert!(stack.is_empty());
        assert!(matches!(
            moved.first().map(Value::view),
            Some(ValueView::Obj(actual)) if moved.len() == 1 && crate::value::Gc::ptr_eq(&actual, &object)
        ));
        drop(moved);

        let slots = [Cell::new(Value::Obj(object))];
        slots.write_value(0, Value::Empty);
        assert!(matches!((slots.read_value(0)).view(), ValueView::Empty));
        slots.write_value(0, Value::Undefined);
        assert!(matches!((slots.read_value(0)).view(), ValueView::Undefined));
    }

    /// Even an empty loan retains the canonical window address; no substitute slice or
    /// decoded argument storage is introduced.
    #[test]
    fn empty_argument_loans_keep_the_interior_canonical_window_address() {
        let words = [Value::Num(1.0); 4];
        let empty = &words[2..2];
        assert_ne!(empty.as_ptr(), [].as_ptr());
        // SAFETY: the zero-length interior slice is aligned/non-null and its array stays live.
        unsafe {
            with_borrowed_values(empty.as_ptr(), 0, |values| {
                assert!(values.is_empty());
                assert_eq!(values.as_ptr(), empty.as_ptr());
            })
        };
    }

    #[test]
    fn execution_storage_loans_the_original_argument_window_without_copying() {
        let object = crate::value::Object::new(None);
        let word = Value::Obj(object).bits();
        for count in 0..=8 {
            let values = vec![Value::Obj(object); count];
            // SAFETY: values is an immutable initialized window for this complete loan. This
            // inspection has no allocation or poll, and the original object stays live.
            unsafe {
                with_borrowed_values(values.as_ptr(), count, |loan| {
                    assert_eq!(loan.len(), count);
                    assert_eq!(loan.as_ptr(), values.as_ptr());
                    assert!(loan.iter().all(|value| {
                        matches!(value.view(), ValueView::Obj(actual)
                            if crate::value::Gc::ptr_eq(&actual, &object))
                    }));
                    assert!(loan.iter().all(|value| value.bits() == word));
                });
            }
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

    /// Slot-to-stack copies (LoadLocal, Dup) and stack-to-slot moves (StoreLocal) of an object's
    /// word: every copy and move has the source value's bits.
    /// Copies are bit copies and drops and overwrites release nothing, so the object
    /// outlives every word until a collection finds it dead.
    #[test]
    fn slot_stack_copies_and_replacements_preserve_the_cell_identity() {
        let object = crate::value::Object::new(None);
        let weak = crate::value::Gc::downgrade(&object);
        let slots = [Cell::new(Value::Obj(object)), Cell::new(Value::Empty)];
        let word = slots[0].get().bits();
        let mut stack = ValueStack::default();
        assert!(slots[1].get().is_empty());
        stack.push(slots[0].get());
        stack.push(stack.last().unwrap());
        assert!(stack.raw_slice().iter().all(|copy| copy.bits() == word));
        slots[1].set(stack.pop().unwrap());
        assert_eq!(slots[1].get().bits(), word);
        drop(stack.pop());
        slots[0].set(Value::Undefined);
        drop(slots);
        assert!(crate::gc_test::alive(&weak), "a word released its object");
    }

    /// A shared alias of a slot (a mapped parameter's) observes a replacement written through
    /// the slice, and a replacement through the alias empties the slot. The alias's
    /// word is the replacement's bits, and replacements release nothing: both objects outlive
    /// their words until a collection.
    #[test]
    fn shared_slot_aliases_observe_canonical_value_replacement() {
        let first = crate::value::Object::new(None);
        let first_weak = crate::value::Gc::downgrade(&first);
        let second = crate::value::Object::new(None);
        let second_weak = crate::value::Gc::downgrade(&second);
        let second_word = crate::value::PACK_OBJ | crate::value::Gc::as_ptr(&second) as u64;
        let slots = [Cell::new(Value::Obj(first))];
        let shared = &slots[..];
        let alias = &shared[0];
        shared.write_value(0, Value::Obj(second));
        assert_eq!(alias.get().bits(), second_word);
        let seen = alias.get();
        assert!(crate::value::Gc::ptr_eq(
            &seen.as_obj().unwrap(),
            &second_weak.upgrade().unwrap()
        ));
        alias.set(Value::Empty);
        assert!(shared[0].get().is_empty());
        drop(seen);
        assert!(
            crate::gc_test::alive(&first_weak) && crate::gc_test::alive(&second_weak),
            "a replacement released its object"
        );
    }

    /// The raw literal factories (inline, heap-packed and large arrays; templated objects) move
    /// their operand words into elements and fields: lengths, identities and -0 hold.
    #[test]
    fn raw_literal_construction_copies_canonical_values() {
        let mut interp = crate::interpreter::Interp::new();
        let object = interp.new_object();
        // Exercise inline-packed, heap-packed and large dense array factories.
        for count in [0, 1, 10, 11, 32, 33, 70] {
            let mut operands: Vec<_> = (0..count).map(|_| Value::Obj(object)).collect();
            let array = unsafe { interp.make_array_from_raw(operands.as_mut_ptr(), count) };
            unsafe { operands.set_len(0) };
            let ValueView::Obj(array) = (array).view() else {
                panic!("array factory returned a primitive")
            };
            assert!(
                matches!((array.borrow().props.get("length").unwrap().value()).view(),
                ValueView::Num(length) if length == count as f64)
            );
            for index in 0..count {
                assert!(
                    matches!((array.borrow().props.get_index(index as u32).unwrap().value()).view(),
                    ValueView::Obj(actual) if crate::value::Gc::ptr_eq(&actual, &object))
                );
            }
            drop(array);
        }
        let template = std::cell::OnceCell::new();
        let keys = [Rc::from("held"), Rc::from("negativeZero")];
        for _ in 0..3 {
            let mut operands = std::mem::ManuallyDrop::new([Value::Obj(object), Value::Num(-0.0)]);
            let result = unsafe {
                interp.make_plain_object_templated_from(&template, &keys, operands.as_mut_ptr(), 2)
            };
            let ValueView::Obj(result) = (result).view() else {
                panic!("object factory returned a primitive")
            };
            assert!(
                matches!((result.borrow().props.get("negativeZero").unwrap().value()).view(),
                ValueView::Num(number) if number.to_bits() == (-0.0f64).to_bits())
            );
            assert!(
                matches!((result.borrow().props.get("held").unwrap().value()).view(),
                ValueView::Obj(actual) if crate::value::Gc::ptr_eq(&actual, &object)),
                "the field holds the moved operand's object"
            );
            drop(result);
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
