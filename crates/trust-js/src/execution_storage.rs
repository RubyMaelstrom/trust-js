//! Owned execution storage. JavaScript abstract operations and host APIs still consume `Value`;
//! persistent local/operand owners use `PackedValue` in both the VM and native backends. Local
//! cells are shared during execution so scoped parameter aliases can replace their owning
//! words through Cell, without an overlapping exclusive slice borrow. Only actual argument
//! lists cross the public wide-Value API; whole activations are never widened.

use crate::value::{PackedValue, Value};

pub(crate) trait StoredValue: Sized + Clone {
    fn from_value(value: Value) -> Self;
    fn replace_stored(&self, value: Self);
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
    fn replace_stored(&self, value: Self) {
        PackedValue::replace_stored(self, value);
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
    fn write_value(&self, index: usize, value: Value);
}

impl<S: StoredValue> SlotAccess for [S] {
    #[inline]
    fn read_value(&self, index: usize) -> Value {
        self[index].clone_value()
    }
    #[inline]
    fn write_value(&self, index: usize, value: Value) {
        self[index].replace_stored(S::from_value(value));
    }
}

/// An operand stack. gc-heap: its whole capacity is initialized memory — new capacity is
/// zero-filled where the buffer is created or grown, and nothing else changes it — so native
/// code that borrows the buffer past its length (`jit::run`, `jit::run_borrowed_frame`) can
/// have the collector scan words it has not written yet without reading uninitialized memory.
pub(crate) struct ValueStack<S: StoredValue>(Vec<S>);

impl<S: StoredValue> Default for ValueStack<S> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<S: StoredValue> ValueStack<S> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        #[cfg_attr(not(feature = "gc-heap"), allow(unused_mut))]
        let mut stack = Self(Vec::with_capacity(capacity));
        #[cfg(feature = "gc-heap")]
        stack.zero_spare_capacity();
        stack
    }
    #[inline]
    pub(crate) fn push(&mut self, value: Value) {
        self.push_stored(S::from_value(value));
    }
    #[inline]
    pub(crate) fn push_stored(&mut self, value: S) {
        #[cfg(feature = "gc-heap")]
        {
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
        #[cfg(not(feature = "gc-heap"))]
        self.0.push(value);
    }
    /// gc-heap: grow by at least `additional` and zero-fill the new spare capacity.
    #[cfg(feature = "gc-heap")]
    #[cold]
    #[inline(never)]
    fn grow(&mut self, additional: usize) {
        self.0.reserve(additional);
        self.zero_spare_capacity();
    }
    #[cfg(feature = "gc-heap")]
    fn zero_spare_capacity(&mut self) {
        let spare = self.0.spare_capacity_mut();
        // SAFETY: the spare capacity is this Vec's memory past its length; zero bytes are a
        // valid fill for it (a packed zero word is the Number +0).
        unsafe { std::ptr::write_bytes(spare.as_mut_ptr(), 0, spare.len()) };
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
        #[cfg(feature = "gc-heap")]
        if self.0.capacity() - self.0.len() < additional {
            self.grow(additional);
        }
        #[cfg(not(feature = "gc-heap"))]
        self.0.reserve(additional);
    }
    #[cfg(test)]
    #[inline]
    pub(crate) fn extend(&mut self, values: impl IntoIterator<Item = Value>) {
        for value in values {
            self.push(value);
        }
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
    use std::rc::Rc;

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("storage fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
        }
    }

    /// An operand stack and shared slots keep scalar bits exactly (-0, infinities, NaN) and an
    /// object's identity through pushes, pops, an argument decode and a split. gc-heap: the
    /// stack's word is the object tag over the cell's address and stays so through the decode,
    /// and nothing is retained or released.
    #[test]
    fn compact_storage_preserves_scalar_bits_and_heap_ownership() {
        assert_eq!(std::mem::size_of::<PackedValue>(), 8);
        let interp = crate::interpreter::Interp::new();
        let object = interp.new_object();
        let mut stack = ValueStack::<PackedValue>::default();
        stack.push(Value::Obj(object.clone()));
        #[cfg(feature = "gc-heap")]
        let word = crate::value::PACK_OBJ | crate::value::Gc::as_ptr(&object) as u64;
        #[cfg(feature = "gc-heap")]
        assert_eq!(stack.last_stored().unwrap().bits(), word);
        stack.with_tail(0, |args| {
            assert!(
                matches!(&args[0], Value::Obj(actual) if crate::value::Gc::ptr_eq(actual, &object))
            );
        });
        #[cfg(feature = "gc-heap")]
        assert_eq!(stack.last_stored().unwrap().bits(), word);
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
        assert!(matches!(
            moved.as_slice(),
            [Value::Obj(actual)] if crate::value::Gc::ptr_eq(actual, &object)
        ));
        drop(moved);

        let slots = [PackedValue::pack(Value::Obj(object.clone()))];
        slots.write_value(0, Value::Empty);
        assert!(matches!(slots.read_value(0), Value::Empty));
        slots.write_value(0, Value::Undefined);
        assert!(matches!(slots.read_value(0), Value::Undefined));
    }

    /// Decoding an argument list makes one Value per argument, on the Rust stack up to four,
    /// each the argument's object, and leaves the words as they were. gc-heap: the words keep
    /// their bits through the decode, and neither the decode nor the words have anything to
    /// release.
    #[test]
    fn compact_storage_decodes_only_argument_owners_with_exact_lifetime() {
        let object = crate::value::Object::new(None);
        let word = PackedValue::pack(Value::Obj(object.clone())).bits();
        for count in 0..=8 {
            let packed: Vec<_> = (0..count)
                .map(|_| PackedValue::pack(Value::Obj(object.clone())))
                .collect();
            let decoded = DecodedArgs::new(&packed);
            assert_eq!(decoded.len(), count);
            assert!(matches!(&decoded, DecodedArgs::Inline { .. }) == (count <= 4));
            assert!(decoded
                .iter()
                .all(|value| matches!(value, Value::Obj(actual) if crate::value::Gc::ptr_eq(actual, &object))));
            drop(decoded);
            assert!(packed.iter().all(|packed| packed.bits() == word));
            drop(packed);
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
    /// word, with no wide Value between them: every copy and move has the source word's bits.
    /// gc-heap: copies are bit copies and drops and overwrites release nothing, so the object
    /// outlives every word until a collection finds it dead.
    #[test]
    fn compact_storage_owner_moves_and_duplicates_do_not_cross_value_boundary() {
        let object = crate::value::Object::new(None);
        #[cfg(feature = "gc-heap")]
        let weak = crate::value::Gc::downgrade(&object);
        let slots = [
            PackedValue::pack(Value::Obj(object)),
            PackedValue::pack(Value::Empty),
        ];
        let word = slots[0].bits();
        let mut stack = ValueStack::<PackedValue>::default();
        assert!(slots[1].is_empty());
        stack.push_stored(slots[0].clone());
        stack.push_stored(stack.last_stored().unwrap().clone());
        assert!(stack.raw_slice().iter().all(|copy| copy.bits() == word));
        slots[1].replace_stored(stack.pop_stored().unwrap());
        assert_eq!(slots[1].bits(), word);
        drop(stack.pop_stored());
        slots[0].replace_stored(PackedValue::pack(Value::Undefined));
        drop(slots);
        #[cfg(feature = "gc-heap")]
        assert!(crate::gc_test::alive(&weak), "a word released its object");
    }

    /// A shared alias of a slot (a mapped parameter's) observes a replacement written through
    /// the slice, and a replacement through the alias empties the slot. gc-heap: the alias's
    /// word is the replacement's bits, and replacements release nothing: both objects outlive
    /// their words until a collection.
    #[test]
    fn compact_storage_shared_alias_observes_replacement_and_exact_owner_release() {
        let first = crate::value::Object::new(None);
        #[cfg(feature = "gc-heap")]
        let first_weak = crate::value::Gc::downgrade(&first);
        let second = crate::value::Object::new(None);
        let second_weak = crate::value::Gc::downgrade(&second);
        #[cfg(feature = "gc-heap")]
        let second_word = crate::value::PACK_OBJ | crate::value::Gc::as_ptr(&second) as u64;
        let slots = [PackedValue::pack(Value::Obj(first))];
        let shared = &slots[..];
        let alias = &shared[0];
        shared.write_value(0, Value::Obj(second));
        #[cfg(feature = "gc-heap")]
        assert_eq!(alias.bits(), second_word);
        let seen = alias.unpack();
        assert!(crate::value::Gc::ptr_eq(
            seen.as_obj().unwrap(),
            &second_weak.upgrade().unwrap()
        ));
        alias.replace_value(Value::Empty);
        assert!(shared[0].is_empty());
        drop(seen);
        #[cfg(feature = "gc-heap")]
        assert!(
            crate::gc_test::alive(&first_weak) && crate::gc_test::alive(&second_weak),
            "a replacement released its object"
        );
    }

    /// The raw literal factories (inline, heap-packed and large arrays; templated objects) move
    /// their operand words into elements and fields: lengths, identities and -0 hold.
    #[test]
    fn compact_storage_raw_literals_move_packed_owners_without_refcount_churn() {
        let mut interp = crate::interpreter::Interp::new();
        let object = interp.new_object();
        // Exercise inline-packed, heap-packed and large dense array factories.
        for count in [0, 1, 10, 11, 32, 33, 70] {
            let mut operands: Vec<_> = (0..count)
                .map(|_| PackedValue::pack(Value::Obj(object.clone())))
                .collect();
            let array = unsafe { interp.make_array_from_raw(operands.as_mut_ptr(), count) };
            unsafe { operands.set_len(0) };
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
                    Value::Obj(actual) if crate::value::Gc::ptr_eq(&actual, &object))
                );
            }
            drop(array);
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
            let Value::Obj(result) = result else {
                panic!("object factory returned a primitive")
            };
            assert!(
                matches!(result.borrow().props.get("negativeZero").unwrap().value(),
                Value::Num(number) if number.to_bits() == (-0.0f64).to_bits())
            );
            #[cfg(feature = "gc-heap")]
            assert!(
                matches!(result.borrow().props.get("held").unwrap().value(),
                Value::Obj(actual) if crate::value::Gc::ptr_eq(&actual, &object)),
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
