//! S6 step 11b: the embedder's scoped values (`trust_js::embed`).
//!
//! The embed API hands out values that cannot be stored: a [`Value<'s>`] carries the lifetime
//! `'s` of the scope that produced it, a generative brand that is never `'static` and is
//! invariant, so a value can neither outlive its scope nor meet another scope's value. A scope
//! is an [`Engine::scope`](crate::Engine::scope) closure or one native call, and its context
//! [`Ctx<'s>`] is a `repr(transparent)` façade over the interpreter that carries every method
//! taking or returning a value. Rust code that keeps a value past its scope keeps it in a
//! [`Handle`](crate::embed::Handle) (S6 step 11a), made from a value through a context and
//! read back into one.
//!
//! Scoped and internal values are transparent eight-byte execution words; each handle is
//! transparent over its cell pointer. Bare native functions retain their original declared
//! scoped signature and are called at that type, with explicit Result mapping. Ordinary
//! immutable argument windows are borrowed directly; EMPTY is normalized at public boundaries.

use crate::value::ValueView;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::handles::{Handle, HandleError};
use crate::interpreter::{abrupt_value, Abrupt, ArrayBufferBytes, Interp};
use crate::stack_trace::SourceOrigin;
use crate::value::Value as Internal;
use crate::{InterruptReason, ParseError};

/// The brand of scope `'s`: invariant in `'s`, so values of two scopes never unify, and neither
/// `Send` nor `Sync`, so no scoped value leaves its thread.
type Brand<'s> = PhantomData<*mut &'s ()>;

/// A thread-bound JavaScript execution word branded by its scope. Its transparent word
/// representation matches the engine's Value; decoded ValueKind views are never storage.
/// Stack values stay rooted during the scope; heap collections held across JS calls use a
/// ValueList, and values kept beyond the scope use a Handle.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Value<'s> {
    word: Internal,
    scope: Brand<'s>,
}

impl Default for Value<'_> {
    fn default() -> Self {
        Self::Undefined
    }
}

/// An ephemeral decoded match view with the same invariant, thread-bound scope brand.
#[derive(Clone, Copy)]
#[repr(u64)]
pub enum ValueKind<'s> {
    Undefined = 0,
    Null = 2,
    Bool(bool) = 3,
    Num(f64) = 4,
    BigInt(BigInt<'s>) = 5,
    Str(JsString<'s>) = 6,
    Sym(Symbol<'s>) = 7,
    Obj(Object<'s>) = 8,
}

/// An object of the scope `'s`: the address of its cell, with no access to the cell itself.
/// Use it through the scope's [`Ctx`] (or [`Object::get`], [`Object::set`], [`Object::call`]);
/// two `Object`s are equal when they are the same object.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Object<'s> {
    cell: NonNull<u8>,
    scope: Brand<'s>,
}

/// A string of the scope `'s`; read its text with [`JsString::as_text`] or
/// [`JsString::as_engine_text`].
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct JsString<'s> {
    cell: NonNull<u8>,
    scope: Brand<'s>,
}

/// A Symbol of the scope `'s`. Two `Symbol`s are equal when they are the same Symbol.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Symbol<'s> {
    cell: NonNull<u8>,
    scope: Brand<'s>,
}

/// A BigInt of the scope `'s`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BigInt<'s> {
    cell: NonNull<u8>,
    scope: Brand<'s>,
}

// The layouts the conversions below rely on: the scoped value is the internal value without its
// EMPTY marker, each wrapper the internal cell handle it stands for.
const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<Value<'static>>() == size_of::<Internal>());
    assert!(align_of::<Value<'static>>() == align_of::<Internal>());
    assert!(size_of::<Value<'static>>() == 8);
    assert!(size_of::<Object<'static>>() == size_of::<crate::value::Gc>());
    assert!(size_of::<JsString<'static>>() == size_of::<crate::lstr::LStr>());
    assert!(size_of::<Symbol<'static>>() == size_of::<crate::value::SymbolRef>());
    assert!(size_of::<BigInt<'static>>() == size_of::<crate::bigint::JsBigInt>());
    // Regression evidence only: typed native calls never transmute Result signatures.
    assert!(
        size_of::<Result<Value<'static>, Value<'static>>>()
            == size_of::<Result<Internal, Internal>>()
    );
    assert!(
        align_of::<Result<Value<'static>, Value<'static>>>()
            == align_of::<Result<Internal, Internal>>()
    );
};

#[allow(non_snake_case, non_upper_case_globals)]
impl<'s> Value<'s> {
    pub const Undefined: Self = Self {
        word: Internal::Undefined,
        scope: PhantomData,
    };
    pub const Null: Self = Self {
        word: Internal::Null,
        scope: PhantomData,
    };

    #[inline]
    pub const fn Bool(value: bool) -> Self {
        Self {
            word: Internal::Bool(value),
            scope: PhantomData,
        }
    }

    /// Construct a Number, canonicalizing every IEEE NaN while preserving signed zeros.
    #[inline]
    pub const fn Num(value: f64) -> Self {
        Self {
            word: Internal::Num(value),
            scope: PhantomData,
        }
    }

    #[inline]
    pub fn BigInt(value: BigInt<'s>) -> Self {
        Self::from_internal(Internal::BigInt(*value.bigint()))
    }
    #[inline]
    pub fn Str(value: JsString<'s>) -> Self {
        Self::from_internal(Internal::Str(*value.lstr()))
    }
    #[inline]
    pub fn Sym(value: Symbol<'s>) -> Self {
        Self::from_internal(Internal::Sym(*value.symbol()))
    }
    #[inline]
    pub fn Obj(value: Object<'s>) -> Self {
        Self::from_internal(Internal::Obj(value.gc()))
    }

    /// Inspect a value through a decoded scoped view. Even a raw internal EMPTY word reads
    /// as undefined; no invalid Rust enum representation is ever manufactured.
    #[inline(always)]
    pub fn kind(&self) -> ValueKind<'s> {
        match self.word.view() {
            ValueView::Undefined | ValueView::Empty => ValueKind::Undefined,
            ValueView::Null => ValueKind::Null,
            ValueView::Bool(value) => ValueKind::Bool(value),
            ValueView::Num(value) => ValueKind::Num(value),
            ValueView::BigInt(value) => ValueKind::BigInt(unsafe {
                // Each handle is transparent over the same cell pointer; the output retains
                // this receiver's invariant brand and the original scope owns the cell.
                std::mem::transmute::<crate::bigint::JsBigInt, BigInt<'s>>(value)
            }),
            ValueView::Str(value) => ValueKind::Str(JsString::from_lstr(value)),
            ValueView::Sym(value) => ValueKind::Sym(unsafe {
                std::mem::transmute::<crate::value::SymbolRef, Symbol<'s>>(value)
            }),
            ValueView::Obj(value) => ValueKind::Obj(Object::from_gc(value)),
        }
    }

    #[inline]
    pub(crate) fn from_internal(value: Internal) -> Value<'s> {
        let word = if value.bits() == crate::value::PACK_EMPTY {
            Internal::Undefined
        } else {
            value
        };
        Self {
            word,
            scope: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn into_internal(self) -> Internal {
        *self.as_internal()
    }

    /// Ordinary values lend their original word. The internal completion marker is exposed
    /// as an immutable undefined constant, including through coercion/equality entry points.
    #[inline]
    pub(crate) fn as_internal(&self) -> &Internal {
        if self.word.bits() == crate::value::PACK_EMPTY {
            &Internal::Undefined
        } else {
            &self.word
        }
    }

    /// Borrow already normalized immutable words (ValueList maintains this invariant).
    #[inline]
    pub(crate) fn slice_from_internal<'a>(values: &'a [Internal]) -> &'a [Value<'s>] {
        debug_assert!(values
            .iter()
            .all(|value| value.bits() != crate::value::PACK_EMPTY));
        // SAFETY: both wrappers are transparent execution words. The original immutable loan
        // and its roots remain live for the whole returned borrow; this never casts Cell slots.
        unsafe { &*(std::ptr::from_ref(values) as *const [Value<'s>]) }
    }

    /// A native-boundary loan. Ordinary calls retain the canonical window's address; an
    /// unexpected EMPTY uses a rare normalized copy without changing the caller's storage.
    pub(crate) fn normalized_slice_from_internal<'a>(
        values: &'a [Internal],
    ) -> Cow<'a, [Value<'s>]> {
        if values
            .iter()
            .any(|value| value.bits() == crate::value::PACK_EMPTY)
        {
            Cow::Owned(values.iter().copied().map(Self::from_internal).collect())
        } else {
            Cow::Borrowed(Self::slice_from_internal(values))
        }
    }

    /// Forwarded public slices normalize raw EMPTY values to real undefined elements, never
    /// holes. Ordinary inputs remain the original immutable word window.
    pub(crate) fn slice_as_internal<'a>(values: &'a [Value<'s>]) -> Cow<'a, [Internal]> {
        if values
            .iter()
            .any(|value| value.word.bits() == crate::value::PACK_EMPTY)
        {
            Cow::Owned(values.iter().copied().map(Self::into_internal).collect())
        } else {
            // SAFETY: the same transparent word representation and immutable loan. The scope
            // still roots every reference, and the callee establishes its normal slice roots.
            Cow::Borrowed(unsafe { &*(std::ptr::from_ref(values) as *const [Internal]) })
        }
    }

    /// Whether this is undefined.
    pub fn is_undefined(&self) -> bool {
        matches!(&(self).kind(), ValueKind::Undefined)
    }

    /// Whether this is undefined or null.
    pub fn is_nullish(&self) -> bool {
        matches!(&(self).kind(), ValueKind::Undefined | ValueKind::Null)
    }

    /// The object, if this is one.
    pub fn as_obj(&self) -> Option<Object<'s>> {
        match &(self).kind() {
            ValueKind::Obj(object) => Some(*object),
            _ => None,
        }
    }

    /// The number, if this is a Number (an embedder convenience for reading op arguments).
    pub fn as_num_opt(&self) -> Option<f64> {
        match &(self).kind() {
            ValueKind::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// A string value as Rust text: the conversion to a Web IDL `USVString`, in which every lone
    /// surrogate becomes U+FFFD. `None` for any other value; see [`Ctx::coerce_string`] for
    /// ToString.
    pub fn as_text(&self) -> Option<Cow<'_, str>> {
        self.as_internal().as_text()
    }

    /// A string value's engine text, unchanged (see [`JsString::as_engine_text`]). `None` for
    /// any other value; see [`Ctx::coerce_engine_text`] for ToString.
    pub fn as_engine_text(&self) -> Option<&str> {
        self.as_internal().as_engine_text()
    }

    /// Read a BigInt as an `i64` (wrapping past ±2^63), for the embedder's 64-bit bridge. `None`
    /// when the value isn't a BigInt.
    pub fn bigint_as_i64(&self) -> Option<i64> {
        match &(self).kind() {
            ValueKind::BigInt(bigint) => Some(bigint.to_i64_wrapping()),
            _ => None,
        }
    }

    /// Whether this is a callable object.
    pub fn is_callable(&self) -> bool {
        self.as_internal().is_callable()
    }

    /// The `typeof` result determined by the value alone. The Annex B.3.6 `[[IsHTMLDDA]]` object
    /// (`document.all`) is branded in its Agent, so this reports "function" for it; use
    /// [`Ctx::type_of`] for the operator's exact result.
    pub fn type_of(&self) -> &'static str {
        self.as_internal().type_of()
    }
}

impl std::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &(self).kind() {
            ValueKind::Undefined => f.write_str("Undefined"),
            ValueKind::Null => f.write_str("Null"),
            ValueKind::Bool(value) => f.debug_tuple("Bool").field(value).finish(),
            ValueKind::Num(value) => f.debug_tuple("Num").field(value).finish(),
            ValueKind::BigInt(value) => f.debug_tuple("BigInt").field(value).finish(),
            ValueKind::Str(value) => f.debug_tuple("Str").field(value).finish(),
            ValueKind::Sym(value) => f.debug_tuple("Sym").field(value).finish(),
            ValueKind::Obj(value) => f.debug_tuple("Obj").field(value).finish(),
        }
    }
}

impl<'s> From<bool> for Value<'s> {
    fn from(value: bool) -> Value<'s> {
        Value::Bool(value)
    }
}

impl<'s> From<f64> for Value<'s> {
    fn from(value: f64) -> Value<'s> {
        Value::Num(value)
    }
}

impl<'s> From<Object<'s>> for Value<'s> {
    fn from(object: Object<'s>) -> Value<'s> {
        Value::Obj(object)
    }
}

impl<'s> From<JsString<'s>> for Value<'s> {
    fn from(string: JsString<'s>) -> Value<'s> {
        Value::Str(string)
    }
}

impl<'s> Object<'s> {
    /// The engine handle of this object.
    #[inline]
    pub(crate) fn gc(self) -> crate::value::Gc {
        // SAFETY: the address of a live object cell (a scoped value is rooted by its frame).
        unsafe { crate::value::Gc::from_raw(self.cell.as_ptr().cast()) }
    }

    /// The scoped form of an engine handle.
    #[inline]
    pub(crate) fn from_gc(object: crate::value::Gc) -> Object<'s> {
        Object {
            cell: NonNull::new(crate::value::Gc::as_ptr(&object).cast_mut().cast())
                .expect("an object handle is never null"),
            scope: PhantomData,
        }
    }

    /// The object's identity while it lives: the address of its cell.
    pub fn addr(&self) -> usize {
        self.cell.as_ptr() as usize
    }

    /// Whether the object is callable.
    pub fn is_callable(&self) -> bool {
        Value::Obj(*self).is_callable()
    }

    /// `object[key]`: `[[Get]]` with a String key (see [`Ctx::member_get`]).
    pub fn get(self, cx: &mut Ctx<'s>, key: &str) -> Result<Value<'s>, Value<'s>> {
        cx.member_get(&Value::Obj(self), key)
    }

    /// `object[key] = value`: `[[Set]]` with a String key (see [`Ctx::member_set`]).
    pub fn set(self, cx: &mut Ctx<'s>, key: &str, value: Value<'s>) -> Result<(), Value<'s>> {
        cx.member_set(&Value::Obj(self), key, value)
    }

    /// Call the object as a function (see [`Ctx::invoke`]).
    pub fn call(
        self,
        cx: &mut Ctx<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        cx.invoke(Value::Obj(self), this, args)
    }
}

impl PartialEq for Object<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

impl Eq for Object<'_> {}

impl std::fmt::Debug for Object<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Object({:#x})", self.addr())
    }
}

impl<'s> JsString<'s> {
    /// The engine string.
    #[inline]
    fn lstr(&self) -> &crate::lstr::LStr {
        // SAFETY: `LStr` is `repr(transparent)` over its cell's address, as this is.
        unsafe { &*std::ptr::from_ref(self).cast::<crate::lstr::LStr>() }
    }

    /// The scoped form of an engine string.
    #[inline]
    pub(crate) fn from_lstr(string: crate::lstr::LStr) -> JsString<'s> {
        // SAFETY: as for `lstr`; an engine string releases nothing when dropped.
        unsafe { std::mem::transmute::<crate::lstr::LStr, JsString<'s>>(string) }
    }

    /// The string's engine text, unchanged: UTF-8 in which a lone surrogate is a scalar in
    /// U+10F800..=U+10FFFF (and a character in that range is a pair of them). Every code unit
    /// survives a round trip through [`Ctx::string_from_engine_text`]. Convert it with
    /// [`engine_to_text`](crate::embed::engine_to_text) before treating it as Rust text.
    pub fn as_engine_text(&self) -> &str {
        self.lstr().as_str()
    }

    /// The string as Rust text: the Web IDL `USVString` conversion, in which every lone
    /// surrogate becomes U+FFFD.
    pub fn as_text(&self) -> Cow<'_, str> {
        crate::jstr::to_text(self.as_engine_text())
    }

    /// The string's UTF-16 code units, lone surrogates included.
    pub fn to_utf16(&self) -> Vec<u16> {
        crate::jstr::units(self.as_engine_text())
    }
}

impl std::fmt::Debug for JsString<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.as_text(), f)
    }
}

impl Symbol<'_> {
    /// The engine Symbol.
    #[inline]
    fn symbol(&self) -> &crate::value::SymbolRef {
        // SAFETY: a Symbol handle is `repr(transparent)` over its cell's address, as this is.
        unsafe { &*std::ptr::from_ref(self).cast::<crate::value::SymbolRef>() }
    }

    /// The Symbol's description, if it has one.
    pub fn description(&self) -> Option<Rc<str>> {
        self.symbol()
            .description
            .as_deref()
            .map(|description| Rc::from(crate::jstr::to_text(description)))
    }
}

impl PartialEq for Symbol<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cell == other.cell
    }
}

impl Eq for Symbol<'_> {}

impl std::fmt::Debug for Symbol<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Symbol({:?})", self.description())
    }
}

impl BigInt<'_> {
    /// The engine BigInt.
    #[inline]
    fn bigint(&self) -> &crate::bigint::JsBigInt {
        // SAFETY: `JsBigInt` is `repr(transparent)` over its cell's address, as this is.
        unsafe { &*std::ptr::from_ref(self).cast::<crate::bigint::JsBigInt>() }
    }

    /// The BigInt as an `i64`, wrapping past ±2^63.
    pub fn to_i64_wrapping(&self) -> i64 {
        self.bigint().to_i128_wrapping() as i64
    }

    /// The BigInt as an `i128`, wrapping past ±2^127.
    pub fn to_i128_wrapping(&self) -> i128 {
        self.bigint().to_i128_wrapping()
    }
}

impl std::fmt::Debug for BigInt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BigInt({})", self.to_i128_wrapping())
    }
}

/// A list of values of the scope `'s` that stay rooted while the list lives: the form of a
/// builder that holds several values across a call that can run JavaScript, and so collect (a
/// `[[Set]]` can call a setter). A `Vec<Value<'s>>` held across such a call is not rooted: its
/// values can be collected under it. A list reads as a slice of its values.
pub struct ValueList<'s> {
    values: crate::rooting::RootedVec,
    scope: Brand<'s>,
}

impl<'s> ValueList<'s> {
    /// An empty list.
    pub fn new() -> ValueList<'s> {
        ValueList {
            values: crate::rooting::RootedVec::new(),
            scope: PhantomData,
        }
    }

    /// An empty list with room for `capacity` values.
    pub fn with_capacity(capacity: usize) -> ValueList<'s> {
        ValueList {
            values: crate::rooting::RootedVec::from_vec(Vec::with_capacity(capacity)),
            scope: PhantomData,
        }
    }

    /// Hold `value` at the end of the list.
    pub fn push(&mut self, value: Value<'s>) {
        self.values.push(value.into_internal());
    }

    /// Take the last value.
    pub fn pop(&mut self) -> Option<Value<'s>> {
        self.values.pop().map(Value::from_internal)
    }

    /// Hold `value` at `index`, shifting the values after it.
    pub fn insert(&mut self, index: usize, value: Value<'s>) {
        self.values.insert(index, value.into_internal());
    }

    /// Take the value at `index`, shifting the values after it.
    pub fn remove(&mut self, index: usize) -> Value<'s> {
        Value::from_internal(self.values.remove(index))
    }

    /// Keep the first `len` values.
    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }

    /// Release every value.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// The values, as a slice.
    pub fn as_slice(&self) -> &[Value<'s>] {
        Value::slice_from_internal(&self.values)
    }

    /// A new Array of the values, in order (making it runs no JavaScript).
    pub fn into_array(self, cx: &Ctx<'s>) -> Value<'s> {
        cx.make_array_of(self.as_slice())
    }
}

impl Default for ValueList<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'s> std::ops::Deref for ValueList<'s> {
    type Target = [Value<'s>];
    fn deref(&self) -> &[Value<'s>] {
        self.as_slice()
    }
}

impl<'s> Extend<Value<'s>> for ValueList<'s> {
    fn extend<I: IntoIterator<Item = Value<'s>>>(&mut self, values: I) {
        self.values
            .extend(values.into_iter().map(Value::into_internal));
    }
}

impl<'s> FromIterator<Value<'s>> for ValueList<'s> {
    fn from_iter<I: IntoIterator<Item = Value<'s>>>(values: I) -> Self {
        let mut list = ValueList::new();
        list.extend(values);
        list
    }
}

impl std::fmt::Debug for ValueList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

/// A native (Rust-implemented) function. It receives the context of its call's scope, its
/// receiver and its arguments, and returns its result or (as `Err`) the value it throws. It is a
/// bare `fn` and cannot capture: Rust state lives in [`OpState`](crate::embed::OpState), reached
/// through the context, and JavaScript values a function keeps are engine captures
/// ([`Ctx::new_native_fn_with_captures`]). Declare one generic over the scope:
///
/// ```
/// use trust_js::embed::{Ctx, Value};
/// fn first<'s>(
///     _cx: &mut Ctx<'s>,
///     _this: Value<'s>,
///     args: &[Value<'s>],
/// ) -> Result<Value<'s>, Value<'s>> {
///     Ok(args.first().copied().unwrap_or_default())
/// }
/// let mut engine = trust_js::Engine::new();
/// engine.define_global("first", 1, first);
/// let n = engine.scope(|cx| cx.eval_value("first(7, 8)").unwrap().ok()?.as_num_opt());
/// assert_eq!(n, Some(7.0));
/// ```
pub type NativeFn =
    for<'s> fn(&mut Ctx<'s>, Value<'s>, &[Value<'s>]) -> Result<Value<'s>, Value<'s>>;

/// A native function with engine-owned JavaScript captures, the last argument: the values
/// given when it was made, which live as long as the function does (or while it runs).
pub type NativeCaptureFn =
    for<'s> fn(&mut Ctx<'s>, Value<'s>, &[Value<'s>], &[Value<'s>]) -> Result<Value<'s>, Value<'s>>;

/// The length of a live indexed collection ([`Ctx::install_live_readonly_indexed_properties`]):
/// a native read of the host's state, which runs no JavaScript.
pub type IndexedLength = for<'s> fn(&Ctx<'s>, &Value<'s>) -> u32;

/// The supported property names of a live named collection
/// ([`Ctx::install_live_readonly_indexed_properties`]): a native read, as [`IndexedLength`].
pub type IndexedNames = for<'s> fn(&Ctx<'s>, &Value<'s>) -> Vec<String>;

/// Invoke a bare scoped callback at its actual declared Result signature. The caller roots
/// the original immutable windows through the complete native boundary (including polls/tails).
#[inline]
pub(crate) fn invoke_native(
    function: NativeFn,
    interp: &mut Interp,
    this: Internal,
    args: &[Internal],
) -> Result<Internal, Internal> {
    // SAFETY: one fresh call scope; the higher-ranked fn cannot export its brand. Results are
    // converted before leaving this call. No aggregate-return function pointer is recast.
    let cx = unsafe { Ctx::new(interp) };
    function(
        cx,
        Value::from_internal(this),
        &Value::normalized_slice_from_internal(args),
    )
    .map(Value::into_internal)
    .map_err(Value::into_internal)
}

#[inline]
pub(crate) fn invoke_captured_native(
    function: NativeCaptureFn,
    interp: &mut Interp,
    this: Internal,
    args: &[Internal],
    captures: &[Internal],
) -> Result<Internal, Internal> {
    // SAFETY: as in invoke_native, with the immutable engine-owned captures rooted by the call.
    let cx = unsafe { Ctx::new(interp) };
    function(
        cx,
        Value::from_internal(this),
        &Value::normalized_slice_from_internal(args),
        &Value::normalized_slice_from_internal(captures),
    )
    .map(Value::into_internal)
    .map_err(Value::into_internal)
}

/// The engine form of a native closure: a closure of the engine's signature calling it.
fn native_closure<F>(f: F) -> Rc<crate::value::NativeClosure>
where
    F: for<'t> Fn(&mut Ctx<'t>, Value<'t>, &[Value<'t>]) -> Result<Value<'t>, Value<'t>> + 'static,
{
    Rc::new(
        move |interp: &mut Interp, this: Internal, args: &[Internal]| {
            // SAFETY: the call's own scope; `f` is generic over it, and its results are converted
            // back before the scope ends.
            let cx = unsafe { Ctx::new(interp) };
            f(
                cx,
                Value::from_internal(this),
                &Value::normalized_slice_from_internal(args),
            )
            .map(Value::into_internal)
            .map_err(Value::into_internal)
        },
    )
}

/// The engine form of a native closure with captures (as [`native_closure`]).
fn native_capture_closure<F>(f: F) -> Rc<crate::value::CaptureClosure>
where
    F: for<'t> Fn(
            &mut Ctx<'t>,
            Value<'t>,
            &[Value<'t>],
            &[Value<'t>],
        ) -> Result<Value<'t>, Value<'t>>
        + 'static,
{
    Rc::new(
        move |interp: &mut Interp, this: Internal, args: &[Internal], captures: &[Internal]| {
            // SAFETY: as in `native_closure`.
            let cx = unsafe { Ctx::new(interp) };
            f(
                cx,
                Value::from_internal(this),
                &Value::normalized_slice_from_internal(args),
                &Value::normalized_slice_from_internal(captures),
            )
            .map(Value::into_internal)
            .map_err(Value::into_internal)
        },
    )
}

/// Why an interrupt-aware entry point did not complete: a thrown value, or a host interruption,
/// which is not a JavaScript exception (author `catch` and `finally` blocks do not observe it).
pub enum EvalError<'s> {
    Throw(Value<'s>),
    Interrupted(InterruptReason),
}

impl std::fmt::Debug for EvalError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Throw(value) => f.debug_tuple("Throw").field(value).finish(),
            EvalError::Interrupted(reason) => f.debug_tuple("Interrupted").field(reason).finish(),
        }
    }
}

impl<'s> EvalError<'s> {
    fn from_internal(failure: crate::EvalFailure) -> EvalError<'s> {
        match failure {
            crate::EvalFailure::Throw(value) => EvalError::Throw(Value::from_internal(value)),
            crate::EvalFailure::Interrupted(reason) => EvalError::Interrupted(reason),
        }
    }
}

/// An engine completion of an interrupt-aware entry point in scoped form.
fn eval_result<'s>(
    result: Result<Internal, crate::EvalFailure>,
) -> Result<Value<'s>, EvalError<'s>> {
    result
        .map(Value::from_internal)
        .map_err(EvalError::from_internal)
}

/// An engine result whose `Err` is a thrown value, in scoped form.
#[inline]
fn thrown<'s>(result: Result<Internal, Internal>) -> Result<Value<'s>, Value<'s>> {
    result
        .map(Value::from_internal)
        .map_err(Value::from_internal)
}

/// An abrupt completion lowered to the value it throws (a host interruption to undefined; see
/// [`Ctx::invoke`]).
#[inline]
fn lowered<'s>(abrupt: Abrupt) -> Value<'s> {
    Value::from_internal(abrupt_value(abrupt))
}

/// The context of the scope `'s`: the interpreter seen through the embedder's API. A native
/// function receives one for its call; [`Engine::scope`](crate::Engine::scope) gives one to its
/// closure. Every method that takes or returns a [`Value`] is here, so values live only as long
/// as the scope that produced them.
///
/// Methods whose JavaScript can throw return the thrown value as `Err`. A host interruption
/// ([`RuntimeInterrupt`](crate::RuntimeInterrupt)) is not a JavaScript exception and cannot be a
/// thrown value: these methods lower it to `Err(undefined)`, which inside a native function is
/// enough, because the engine resumes the interruption when the function returns. Host code
/// calling outside a native function uses the `_interruptible` entry points (or checks
/// [`Ctx::pending_interrupt`]).
#[repr(transparent)]
pub struct Ctx<'s> {
    interp: Interp,
    scope: Brand<'s>,
}

impl<'s> Ctx<'s> {
    /// The context of a scope over `interp`.
    ///
    /// # Safety
    /// `'s` must be the lifetime of one scope: a caller generic over it (a native call, an
    /// [`Engine::scope`](crate::Engine::scope) closure), no value of which outlives the scope or
    /// meets another scope's values.
    #[inline]
    pub(crate) unsafe fn new(interp: &mut Interp) -> &mut Ctx<'s> {
        // SAFETY: `Ctx` is `repr(transparent)` over `Interp`.
        unsafe { &mut *std::ptr::from_mut(interp).cast::<Ctx<'s>>() }
    }

    /// The interpreter, with its heap the active one: a call that allocates or runs JavaScript
    /// allocates in the scope's heap whatever another Engine left active.
    #[inline]
    fn it(&mut self) -> &mut Interp {
        self.interp.activate_gc_heap();
        &mut self.interp
    }

    /// [`Ctx::it`] for a shared borrow.
    #[inline]
    fn it_ref(&self) -> &Interp {
        self.interp.activate_gc_heap();
        &self.interp
    }

    /// The interpreter, read-only.
    #[inline]
    pub(crate) fn interp(&self) -> &Interp {
        &self.interp
    }

    /// The interpreter (tests).
    #[cfg(test)]
    pub(crate) fn interp_mut(&mut self) -> &mut Interp {
        &mut self.interp
    }

    // ----- making values ----------------------------------------------------------------------

    /// The JS string whose code points are the characters of the Rust text `s`.
    ///
    /// The engine stores a lone surrogate as a scalar in U+10F800..=U+10FFFF, so a character of
    /// `s` in that range is stored as its surrogate pair: `cx.str("\u{10FFFD}")` has length 2. A
    /// Rust string cannot hold a lone surrogate; build one from code units with
    /// [`Ctx::string_from_utf16`].
    pub fn str(&self, s: impl AsRef<str>) -> Value<'s> {
        let s = s.as_ref();
        self.it_ref();
        Value::from_internal(Internal::str(s))
    }

    /// [`Ctx::str`] for an owned `String`, reusing its buffer when no character needs the
    /// surrogate-pair form.
    pub fn string(&self, s: String) -> Value<'s> {
        self.it_ref();
        Value::from_internal(Internal::from_string(s))
    }

    /// The JS string with exactly these UTF-16 code units, lone surrogates included.
    pub fn string_from_utf16(&self, units: &[u16]) -> Value<'s> {
        self.it_ref();
        Value::from_internal(Internal::from_utf16(units))
    }

    /// The JS string whose engine text is `s`, with no conversion: the lossless path for a host
    /// that keeps JS strings as Rust strings, such as a DOM storing Web IDL `DOMString`s. `s`
    /// must come from [`JsString::as_engine_text`], [`Ctx::coerce_engine_text`], or Rust text
    /// converted by [`text_to_engine`](crate::embed::text_to_engine); Rust text passed here
    /// directly loses characters in U+10F800..=U+10FFFF to lone surrogates.
    pub fn string_from_engine_text(&self, s: impl Into<String>) -> Value<'s> {
        let s = s.into();
        self.it_ref();
        Value::from_internal(Internal::from_engine_text(s))
    }

    /// The JS string with the characters of Rust text; the same as [`Ctx::string`].
    pub fn string_from_utf8(&self, value: String) -> Value<'s> {
        self.string(value)
    }

    /// A BigInt from an `i64` (for the embedder's 64-bit integer bridge, e.g. wasm i64).
    pub fn bigint_from_i64(&self, value: i64) -> Value<'s> {
        self.it_ref();
        Value::from_internal(Internal::bigint_from_i64(value))
    }

    /// A BigInt from a `u64` (for the embedder's 64-bit bridge, e.g. an unsigned FFI return).
    pub fn bigint_from_u64(&self, value: u64) -> Value<'s> {
        self.it_ref();
        Value::from_internal(Internal::bigint_from_u64(value))
    }

    /// A BigInt from an `i128` (an FFI `int64_t` widened to preserve its sign).
    pub fn bigint_from_i128(&self, value: i128) -> Value<'s> {
        self.it_ref();
        Value::from_internal(Internal::bigint_from_i128(value))
    }

    /// Mint a fresh Symbol in this Agent.
    pub fn new_symbol(&mut self, description: Option<Rc<str>>) -> Value<'s> {
        Value::from_internal(self.it().new_symbol(description))
    }

    /// A fresh ordinary object with the current Realm's %Object.prototype%.
    pub fn new_object(&self) -> Object<'s> {
        Object::from_gc(self.it_ref().new_object())
    }

    /// A fresh object with an explicit `[[Prototype]]` (`null`, or anything that is not an
    /// object, gives none). For host addons creating instances with a class's prototype.
    pub fn new_object_with_proto(&self, proto: &Value<'s>) -> Value<'s> {
        Value::from_internal(self.it_ref().new_object_with_proto(proto.as_internal()))
    }

    /// A new Array of `items`, in order (making it runs no JavaScript). `items` is a plain
    /// `Vec`: build it with no call that can run JavaScript in between, or use a [`ValueList`].
    pub fn make_array(&self, items: Vec<Value<'s>>) -> Value<'s> {
        let items = items.into_iter().map(Value::into_internal).collect();
        Value::from_internal(self.it_ref().make_array(items))
    }

    /// [`Ctx::make_array`] from a slice.
    pub fn make_array_of(&self, items: &[Value<'s>]) -> Value<'s> {
        Value::from_internal(
            self.it_ref()
                .make_array(Value::slice_as_internal(items).to_vec()),
        )
    }

    /// A new Error of the current Realm: `kind` names its constructor (`"TypeError"`, ...).
    pub fn make_error(&self, kind: &str, message: impl Into<String>) -> Value<'s> {
        let message = message.into();
        Value::from_internal(self.it_ref().make_error(kind, message))
    }

    /// [`Ctx::make_error`], to return as a native function's `Err`.
    pub fn throw(&self, kind: &str, message: impl Into<String>) -> Value<'s> {
        self.make_error(kind, message)
    }

    /// `Err` of a new TypeError.
    pub fn type_err<T>(&self, message: impl Into<String>) -> Result<T, Value<'s>> {
        Err(self.make_error("TypeError", message))
    }

    /// The global object (`globalThis`) of the current Realm.
    pub fn global_this(&self) -> Value<'s> {
        Value::from_internal(self.interp.global_this())
    }

    /// A fresh `ArrayBuffer` initialized with `bytes`, with the realm's intrinsic
    /// %ArrayBuffer.prototype% (no author code runs).
    pub fn make_array_buffer(&mut self, bytes: &[u8]) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().make_array_buffer(bytes))
    }

    /// A fresh fixed-length `ArrayBuffer` with an embedder detach key: author JavaScript cannot
    /// transfer or detach it, while [`Ctx::detach_array_buffer`] can. This models the
    /// `"WebAssembly.Memory"` key required for `Memory.buffer`.
    pub fn make_host_keyed_array_buffer(&mut self, bytes: &[u8]) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().make_host_keyed_array_buffer(bytes))
    }

    /// Account for `bytes` of new host-allocated data that an `ArrayBuffer` can expose, such as
    /// a WebAssembly memory's initial or grown pages. Like an engine buffer allocation, this
    /// counts toward the Agent's buffer pressure and may collect.
    pub fn note_external_buffer_growth(&mut self, bytes: usize) {
        self.it().note_external_buffer_growth(bytes);
    }

    /// A fixed-length host-keyed `ArrayBuffer` over an existing Data Block, which writes through
    /// either side update (the WebAssembly JS API §4.1).
    pub fn make_host_keyed_array_buffer_from_storage(
        &mut self,
        storage: ArrayBufferBytes,
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().make_host_keyed_array_buffer_from_storage(storage))
    }

    /// A fresh `Uint8Array` holding `bytes`, with the current Realm's intrinsic prototype (no
    /// author code runs).
    pub fn make_uint8array(&mut self, bytes: &[u8]) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().make_uint8array(bytes))
    }

    /// Create a pending intrinsic Promise and its resolving functions for a host operation
    /// (ECMA-262 CreateResolvingFunctions): only the first call of either settles it.
    pub fn new_promise_with_resolvers(&mut self) -> (Value<'s>, Value<'s>, Value<'s>) {
        let (promise, resolve, reject) = self.it().new_promise_with_resolvers();
        (
            Value::from_internal(promise),
            Value::from_internal(resolve),
            Value::from_internal(reject),
        )
    }

    /// Web IDL "a promise resolved with" `value`: a new promise of the current Realm, resolved
    /// with `value` (adopting a thenable's state as platform work).
    pub fn promise_resolved_with(&mut self, value: Value<'s>) -> Value<'s> {
        Value::from_internal(
            self.it()
                .with_author_entry(|interp| interp.promise_resolved_with(value.into_internal())),
        )
    }

    /// Web IDL "react to a promise" with embedder steps: PerformPromiseThen(`promise`,
    /// `on_fulfilled`, `on_rejected`) into a new promise of the current Realm, which is returned.
    /// The steps are platform-internal: their jobs get null job settings. A missing step
    /// (undefined) passes the settlement through. TypeError when `promise` is not a promise.
    pub fn react_to_promise(
        &mut self,
        promise: &Value<'s>,
        on_fulfilled: Value<'s>,
        on_rejected: Value<'s>,
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().with_author_entry(|interp| {
            interp.react_to_promise(
                promise.as_internal(),
                on_fulfilled.into_internal(),
                on_rejected.into_internal(),
            )
        }))
    }

    // ----- native functions -------------------------------------------------------------------

    /// A native function object of the current Realm.
    pub fn make_native(&self, name: &str, len: usize, f: NativeFn) -> Object<'s> {
        Object::from_gc(self.it_ref().make_scoped_native(name, len, f))
    }

    /// Define a native method on `target` (non-enumerable, as built-ins are).
    pub fn def_method(&self, target: &Object<'s>, name: &str, len: usize, f: NativeFn) {
        self.it_ref().def_scoped_method(&target.gc(), name, len, f);
    }

    /// Define a native function on the active Realm's global object.
    pub fn define_embed_global(&mut self, name: &str, len: usize, function: NativeFn) {
        self.it().define_scoped_global(name, len, function);
    }

    /// A function backed by a native closure that carries Rust state (N-API callbacks carrying
    /// a C function pointer and its `void*`), unlike the bare-`fn` [`NativeFn`]. The closure is
    /// `for<'t>` over its call's scope, as a native function is, and holds no JavaScript value
    /// (it is `'static`): the values a function keeps are engine captures
    /// ([`Ctx::new_native_closure_with_captures`]). `retained` describes the closure's
    /// allocations for the managed-memory diagnostics.
    pub fn new_native_fn(
        &self,
        name: &str,
        len: usize,
        f: impl for<'t> Fn(&mut Ctx<'t>, Value<'t>, &[Value<'t>]) -> Result<Value<'t>, Value<'t>>
            + 'static,
        retained: crate::value::NativeCaptures,
    ) -> Value<'s> {
        Value::from_internal(
            self.it_ref()
                .new_native_fn(name, len, native_closure(f), retained),
        )
    }

    /// [`Ctx::new_native_fn`] as an object.
    pub fn make_native_closure(
        &self,
        name: &str,
        len: usize,
        f: impl for<'t> Fn(&mut Ctx<'t>, Value<'t>, &[Value<'t>]) -> Result<Value<'t>, Value<'t>>
            + 'static,
        retained: crate::value::NativeCaptures,
    ) -> Object<'s> {
        Object::from_gc(
            self.it_ref()
                .make_native_closure(name, len, native_closure(f), retained),
        )
    }

    /// A native function whose JavaScript captures are owned and traced by the engine,
    /// including cycles through the function itself; it receives them as its last argument.
    /// The function lives as long as any JavaScript reference to it, and keeps its captures
    /// alive while it lives or runs. Captures are immutable as a list; captured objects keep
    /// their ordinary mutable behavior.
    pub fn new_native_fn_with_captures(
        &self,
        name: &str,
        len: usize,
        function: NativeCaptureFn,
        captures: &[Value<'s>],
    ) -> Value<'s> {
        Value::from_internal(self.it_ref().new_scoped_native_with_captures(
            name,
            len,
            function,
            Value::slice_as_internal(captures).to_vec(),
        ))
    }

    /// [`Ctx::new_native_fn_with_captures`] for a closure that also carries Rust state, which
    /// `retained` reports for the managed-memory diagnostics, as [`Ctx::new_native_fn`].
    pub fn new_native_closure_with_captures(
        &self,
        name: &str,
        len: usize,
        f: impl for<'t> Fn(
                &mut Ctx<'t>,
                Value<'t>,
                &[Value<'t>],
                &[Value<'t>],
            ) -> Result<Value<'t>, Value<'t>>
            + 'static,
        captures: &[Value<'s>],
        retained: crate::value::NativeCaptures,
    ) -> Value<'s> {
        Value::Obj(Object::from_gc(self.it_ref().make_native_capture_closure(
            name,
            len,
            native_capture_closure(f),
            Value::slice_as_internal(captures).to_vec(),
            retained,
        )))
    }

    /// Mark a native function as a constructor and wire up its `prototype`/`constructor` link,
    /// so `new ctor()` is legal and instances inherit from `proto`.
    pub fn set_constructor_prototype(&self, ctor: &Value<'s>, proto: &Value<'s>) {
        self.it_ref()
            .set_constructor_prototype(ctor.as_internal(), proto.as_internal());
    }

    /// Define an accessor (getter/setter) property on `target`.
    pub fn define_accessor_value(
        &self,
        target: &Value<'s>,
        name: &str,
        get: Option<Value<'s>>,
        set: Option<Value<'s>>,
        enumerable: bool,
    ) {
        self.it_ref().define_accessor_value(
            target.as_internal(),
            name,
            get.map(Value::into_internal),
            set.map(Value::into_internal),
            enumerable,
        );
    }

    /// Install a contiguous set of lazy, read-only indexed properties on a fresh ordinary object
    /// (Web IDL legacy platform objects' indexed properties): `getter` is called as
    /// `getter.call(target, index)` only when a supported property is observed.
    pub fn install_readonly_indexed_properties(
        &mut self,
        target: &Value<'s>,
        length: u32,
        getter: Value<'s>,
    ) -> Result<(), Value<'s>> {
        self.it()
            .install_readonly_indexed_properties(
                target.as_internal(),
                length,
                getter.into_internal(),
            )
            .map_err(Value::from_internal)
    }

    /// Install a live Web IDL collection: membership comes from native reads of `state`
    /// (`length`, and the names of `named`), values from the getters. The reads must run no
    /// JavaScript, allocate no JS object, collect or change observable state.
    pub fn install_live_readonly_indexed_properties(
        &mut self,
        target: &Value<'s>,
        state: Value<'s>,
        length: IndexedLength,
        getter: Value<'s>,
        named: Option<(IndexedNames, Value<'s>)>,
    ) -> Result<(), Value<'s>> {
        self.it()
            .install_live_readonly_indexed_properties(
                target.as_internal(),
                state.into_internal(),
                indexed_length(length),
                getter.into_internal(),
                named.map(|(names, getter)| (indexed_names(names), getter.into_internal())),
            )
            .map_err(Value::from_internal)
    }

    /// Create a minimal callable `[[IsHTMLDDA]]` exotic (Annex B.3.6), as test262's
    /// `$262.IsHTMLDDA` specifies.
    pub fn make_html_dda(&mut self) -> Value<'s> {
        Value::from_internal(self.it().make_html_dda())
    }

    /// The web platform's `HTMLAllCollection`: the `[[IsHTMLDDA]]` object a Document's `all`
    /// returns, with `prototype`, live indexed and named properties read through `state` (as
    /// [`Ctx::install_live_readonly_indexed_properties`]) and `call` as its legacy caller.
    pub fn make_html_all_collection(
        &mut self,
        prototype: &Value<'s>,
        state: Value<'s>,
        length: IndexedLength,
        getter: Value<'s>,
        named: (IndexedNames, Value<'s>),
        call: Value<'s>,
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().make_html_all_collection(
            prototype.as_internal(),
            state.into_internal(),
            indexed_length(length),
            getter.into_internal(),
            (indexed_names(named.0), named.1.into_internal()),
            call.into_internal(),
        ))
    }

    // ----- host state -------------------------------------------------------------------------

    /// Embedder host state (typed per-subsystem slots and the resource table): how a native
    /// function reaches Rust state.
    pub fn op_state(&mut self) -> &mut crate::host::OpState {
        self.interp.op_state()
    }

    /// Shorthand for `op_state().get_mut::<T>()`.
    pub fn host_mut<T: std::any::Any>(&mut self) -> Option<&mut T> {
        self.interp.host_mut()
    }

    /// Read-only host state access for non-reentrant platform membership queries.
    pub fn host<T: std::any::Any>(&self) -> Option<&T> {
        self.interp.host()
    }

    /// Shorthand for `op_state().resources`.
    pub fn resource_table(&mut self) -> &mut crate::host::ResourceTable {
        self.interp.resource_table()
    }

    /// Hold `value` in a [`Handle`]: shorthand for [`Handle::new`].
    pub fn handle(&self, value: &Value<'s>) -> Result<Handle, HandleError> {
        Handle::new(self, value)
    }

    // ----- properties, calls and conversions ---------------------------------------------------

    /// `[[Get]]` of a String key; `Err` is the thrown value.
    pub fn member_get(&mut self, base: &Value<'s>, key: &str) -> Result<Value<'s>, Value<'s>> {
        thrown(
            self.it()
                .with_author_entry(|interp| interp.member_get(base.as_internal(), key)),
        )
    }

    /// `[[Set]]` of a String key; `Err` is the thrown value.
    pub fn member_set(
        &mut self,
        base: &Value<'s>,
        key: &str,
        value: Value<'s>,
    ) -> Result<(), Value<'s>> {
        self.it()
            .with_author_entry(|interp| {
                interp.member_set(base.as_internal(), key, value.into_internal())
            })
            .map_err(Value::from_internal)
    }

    /// [`Ctx::member_get`].
    pub fn get_member(&mut self, base: &Value<'s>, key: &str) -> Result<Value<'s>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.get_member(base.as_internal(), key))
            .map(Value::from_internal)
            .map_err(lowered)
    }

    /// [`Ctx::member_set`].
    pub fn set_member(
        &mut self,
        base: &Value<'s>,
        key: &str,
        value: Value<'s>,
    ) -> Result<(), Value<'s>> {
        self.it()
            .with_author_entry(|interp| {
                interp.set_member(base.as_internal(), key, value.into_internal())
            })
            .map_err(lowered)
    }

    /// Call `callee` with `this` and `args`; `Err` is the thrown value.
    pub fn invoke(
        &mut self,
        callee: Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().with_author_entry(|interp| {
            interp.invoke(
                callee.into_internal(),
                this.into_internal(),
                &Value::slice_as_internal(args),
            )
        }))
    }

    /// [`Ctx::invoke`].
    pub fn call(
        &mut self,
        callee: Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| {
                interp.call(
                    callee.into_internal(),
                    this.into_internal(),
                    &Value::slice_as_internal(args),
                )
            })
            .map(Value::from_internal)
            .map_err(lowered)
    }

    /// [`Ctx::invoke`] keeping a host interruption distinct from a JavaScript throw.
    pub fn invoke_interruptible(
        &mut self,
        callee: Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, EvalError<'s>> {
        eval_result(self.it().with_author_entry(|interp| {
            interp.invoke_interruptible(
                callee.into_internal(),
                this.into_internal(),
                &Value::slice_as_internal(args),
            )
        }))
    }

    /// Enter an author callback through a host boundary: GetFunctionRealm of `callee` (bound
    /// functions and proxies included) is the entry Realm until the call completes.
    pub fn invoke_callback_entry(
        &mut self,
        callee: Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().with_author_entry(|interp| {
            interp.invoke_callback_entry(
                callee.into_internal(),
                this.into_internal(),
                &Value::slice_as_internal(args),
            )
        }))
    }

    /// `new callee(...args)`; `Err` is the thrown value.
    pub fn construct_value(
        &mut self,
        callee: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().with_author_entry(|interp| {
            interp.construct_value(callee.into_internal(), &Value::slice_as_internal(args))
        }))
    }

    /// [`Ctx::construct_value`].
    pub fn construct(
        &mut self,
        callee: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        self.construct_value(callee, args)
    }

    /// Run a host callback with a previously captured incumbent global as its fallback script
    /// caller (HTML prepare/clean up after running a callback). This does not switch the
    /// callable's Realm or settings.
    pub fn with_callback_script_caller<R>(
        &mut self,
        global: &Value<'s>,
        operation: impl FnOnce(&mut Ctx<'s>) -> R,
    ) -> R {
        self.it()
            .with_callback_script_caller(global.as_internal(), |interp| {
                // SAFETY: the same scope, re-entered.
                operation(unsafe { Ctx::new(interp) })
            })
    }

    /// The host interruption currently in force, if any: the reason a lowered `Err(undefined)`
    /// was an interruption rather than a throw.
    pub fn pending_interrupt(&self) -> Option<InterruptReason> {
        self.interp.pending_interrupt()
    }

    /// ToBoolean.
    pub fn to_boolean(&self, v: &Value<'s>) -> bool {
        self.interp.to_boolean(v.as_internal())
    }

    /// ToNumber; `Err` is the thrown value (conversions can run user `valueOf`/`toString`).
    pub fn to_number(&mut self, v: &Value<'s>) -> Result<f64, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.to_number(v.as_internal()))
            .map_err(lowered)
    }

    /// ToString, as a string of this scope.
    pub fn to_string(&mut self, v: &Value<'s>) -> Result<JsString<'s>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.to_string(v.as_internal()))
            .map(JsString::from_lstr)
            .map_err(lowered)
    }

    /// [`Ctx::to_number`].
    pub fn coerce_number(&mut self, v: &Value<'s>) -> Result<f64, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.coerce_number(v.as_internal()))
            .map_err(Value::from_internal)
    }

    /// ToBigInt followed by wrapping conversion to an `i64` (the WebAssembly JS interface's
    /// `i64`: Numbers are rejected).
    pub fn coerce_bigint_i64(&mut self, v: &Value<'s>) -> Result<i64, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.coerce_bigint_i64(v.as_internal()))
            .map_err(Value::from_internal)
    }

    /// ToString as Rust text: the Web IDL `USVString` conversion, in which every lone surrogate
    /// becomes U+FFFD. Use [`Ctx::coerce_utf16`] where lone surrogates must survive.
    pub fn coerce_string(&mut self, v: &Value<'s>) -> Result<Rc<str>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.coerce_string(v.as_internal()))
            .map_err(Value::from_internal)
    }

    /// Web IDL `USVString`: [`Ctx::coerce_string`] as an owned `String`.
    pub fn coerce_usv_string(&mut self, v: &Value<'s>) -> Result<String, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.coerce_usv_string(v.as_internal()))
            .map_err(Value::from_internal)
    }

    /// ToString as UTF-16 code units, lone surrogates included (a Web IDL `DOMString`).
    pub fn coerce_utf16(&mut self, v: &Value<'s>) -> Result<Vec<u16>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.coerce_utf16(v.as_internal()))
            .map_err(Value::from_internal)
    }

    /// ToString as engine text, lone surrogates included: [`Ctx::string_from_engine_text`]
    /// rebuilds exactly the same JS string.
    pub fn coerce_engine_text(&mut self, v: &Value<'s>) -> Result<Rc<str>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.coerce_engine_text(v.as_internal()))
            .map_err(Value::from_internal)
    }

    /// The `typeof` operator's result, including the Annex B.3.6 `[[IsHTMLDDA]]` exception.
    pub fn type_of(&self, v: &Value<'s>) -> &'static str {
        self.interp.type_of(v.as_internal())
    }

    /// The `===` predicate.
    pub fn values_strict_equal(&self, a: &Value<'s>, b: &Value<'s>) -> bool {
        self.interp
            .values_strict_equal(a.as_internal(), b.as_internal())
    }

    /// An object's identity while it lives (its cell's address), or `None` for anything else.
    /// Key host state by it only together with a [`WeakHandle`](crate::embed::WeakHandle) to
    /// the object, which tells a dead object's address, reusable by a new object, from a live
    /// one.
    pub fn object_addr(&self, v: &Value<'s>) -> Option<usize> {
        self.interp.object_addr(v.as_internal())
    }

    /// Whether `v` is a Proxy (Node's `util.types.isProxy()`).
    pub fn is_proxy_value(&self, v: &Value<'s>) -> bool {
        self.interp.is_proxy_value(v.as_internal())
    }

    /// Whether `v` is an Error object (on itself or anywhere up its prototype chain).
    pub fn is_error_value(&self, v: &Value<'s>) -> bool {
        self.interp.is_error_value(v.as_internal())
    }

    /// A bounded, side-effect-free diagnostic for an actual Error exotic: its own string data
    /// message and the engine-captured stack. Runs no `[[Get]]`.
    pub fn error_diagnostic(&self, value: &Value<'s>) -> Option<String> {
        self.interp.error_diagnostic(value.as_internal())
    }

    /// The `[[Prototype]]` of `ToObject(v)` without running author code (see
    /// [`Ctx::get_prototype_of`] for `Object.getPrototypeOf`).
    pub fn prototype_of(&self, v: &Value<'s>) -> Value<'s> {
        Value::from_internal(self.interp.prototype_of(v.as_internal()))
    }

    /// `Object.getPrototypeOf(v)`, a Proxy's trap included; `Err` is the thrown value.
    pub fn get_prototype_of(&mut self, v: &Value<'s>) -> Result<Value<'s>, Value<'s>> {
        thrown(
            self.it()
                .with_author_entry(|interp| interp.get_prototype_of(v.as_internal())),
        )
    }

    /// Whether the current native call is a construct (`new`).
    pub fn is_constructing(&self) -> bool {
        self.interp.is_constructing()
    }

    /// The active native constructor's NewTarget, or undefined for an ordinary call.
    pub fn constructor_new_target(&self) -> Value<'s> {
        Value::from_internal(self.interp.constructor_new_target())
    }

    /// GetFunctionRealm's global, for host bindings implementing constructor prototype fallback.
    pub fn callable_realm_global(&mut self, function: &Value<'s>) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().callable_realm_global(function.as_internal()))
    }

    /// Copy a bounded ordinary Array of own numeric data properties, or `None` (proxies, holes,
    /// accessors, non-numbers): no author code runs.
    pub fn copy_numeric_array(&self, value: &Value<'s>, limit: usize) -> Option<Vec<f64>> {
        self.interp.copy_numeric_array(value.as_internal(), limit)
    }

    /// Intrinsic WeakMap.prototype.get for host-owned private-slot tables.
    pub fn weak_map_get(
        &mut self,
        map: &Value<'s>,
        key: &Value<'s>,
    ) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().weak_map_get(map.as_internal(), key.as_internal()))
    }

    /// Intrinsic Map lookup for private host state.
    pub fn map_get(&mut self, map: &Value<'s>, key: &Value<'s>) -> Result<Value<'s>, Value<'s>> {
        thrown(self.it().map_get(map.as_internal(), key.as_internal()))
    }

    /// Intrinsic Map assignment for private host state.
    pub fn map_set(
        &mut self,
        map: &Value<'s>,
        key: Value<'s>,
        value: Value<'s>,
    ) -> Result<(), Value<'s>> {
        self.it()
            .map_set(
                map.as_internal(),
                key.into_internal(),
                value.into_internal(),
            )
            .map_err(Value::from_internal)
    }

    // ----- binary data ------------------------------------------------------------------------

    /// A TypedArray's `(N-API element-type code, byte length, data pointer)`, or `None`. The
    /// pointer aliases the live backing store until the buffer is resized or detached.
    pub fn typed_array_raw(&mut self, v: &Value<'s>) -> Option<(u8, usize, *mut u8)> {
        self.interp.typed_array_raw(v.as_internal())
    }

    /// The bytes a TypedArray view covers (`None` for anything else or a detached buffer).
    pub fn typed_array_bytes(&self, v: &Value<'s>) -> Option<Vec<u8>> {
        self.interp.typed_array_bytes(v.as_internal())
    }

    /// Run `f` over the bytes a TypedArray view covers, in place; `f` must not call back into
    /// the engine.
    pub fn with_typed_array_bytes<R>(
        &self,
        v: &Value<'s>,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Option<R> {
        self.interp.with_typed_array_bytes(v.as_internal(), f)
    }

    /// Snapshot a matching TypedArray over fixed backing storage for Web IDL unions; user
    /// properties are never read.
    pub fn fixed_typed_array_bytes(
        &self,
        value: &Value<'s>,
        name: &str,
        allow_shared: bool,
    ) -> Option<Vec<u8>> {
        self.interp
            .fixed_typed_array_bytes(value.as_internal(), name, allow_shared)
    }

    /// Copy the bytes of a Web IDL `BufferSource` (`ArrayBuffer`, `DataView` or TypedArray);
    /// `allow_shared` only for an `[AllowShared]` argument.
    pub fn buffer_source_bytes(&self, v: &Value<'s>, allow_shared: bool) -> Option<Vec<u8>> {
        self.interp
            .buffer_source_bytes(v.as_internal(), allow_shared)
    }

    /// Replace an ordinary attached `ArrayBuffer`'s bytes (same length) without changing its
    /// identity.
    pub fn array_buffer_set_bytes(&mut self, v: &Value<'s>, bytes: &[u8]) -> bool {
        self.interp.array_buffer_set_bytes(v.as_internal(), bytes)
    }

    /// Replace a range of an ordinary attached `ArrayBuffer`.
    pub fn array_buffer_set_range(&mut self, v: &Value<'s>, start: usize, bytes: &[u8]) -> bool {
        self.interp
            .array_buffer_set_range(v.as_internal(), start, bytes)
    }

    /// Copy a range of an ordinary attached `ArrayBuffer` into `destination`.
    pub fn array_buffer_copy_range(
        &self,
        v: &Value<'s>,
        start: usize,
        len: usize,
        destination: &mut [u8],
    ) -> bool {
        self.interp
            .array_buffer_copy_range(v.as_internal(), start, len, destination)
    }

    /// The byte ranges JavaScript wrote since the previous call (coalesced).
    pub fn take_array_buffer_dirty_ranges(&mut self, v: &Value<'s>) -> Option<Vec<Range<usize>>> {
        self.interp.take_array_buffer_dirty_ranges(v.as_internal())
    }

    /// The mutation generation of an attached ordinary `ArrayBuffer`.
    pub fn array_buffer_version(&self, v: &Value<'s>) -> Option<u64> {
        self.interp.array_buffer_version(v.as_internal())
    }

    /// Detach an ordinary `ArrayBuffer`.
    pub fn detach_array_buffer(&mut self, v: &Value<'s>) -> bool {
        self.interp.detach_array_buffer(v.as_internal())
    }

    /// Overwrite a TypedArray's leading bytes (they must fit its current byte length).
    pub fn typed_array_set_bytes(&mut self, v: &Value<'s>, bytes: &[u8]) -> bool {
        self.interp.typed_array_set_bytes(v.as_internal(), bytes)
    }

    // ----- realms, scripts and host settings ----------------------------------------------------

    /// Create a fresh ECMAScript Realm in this Agent and return its global object.
    pub fn create_embed_realm(&mut self) -> Value<'s> {
        Value::from_internal(self.it().create_embed_realm())
    }

    /// Run `operation` with `realm_global`'s Realm active, then restore the caller's Realm, even
    /// when the operation fails. `Err` is a TypeError when `realm_global` is not a Realm's.
    pub fn with_embed_realm<R>(
        &mut self,
        realm_global: &Value<'s>,
        operation: impl FnOnce(&mut Ctx<'s>) -> R,
    ) -> Result<R, Value<'s>> {
        self.it()
            .with_embed_realm(realm_global.as_internal(), |interp| {
                // SAFETY: the same scope, re-entered.
                operation(unsafe { Ctx::new(interp) })
            })
            .map_err(Value::from_internal)
    }

    /// Make `realm_global`'s Realm the active one until the returned entry is left
    /// ([`Ctx::leave_realm`] or [`Engine::leave_realm`](crate::Engine::leave_realm)), for an
    /// operation spanning the embedder's own state. Entries nest: leave the innermost first.
    pub fn enter_realm(
        &mut self,
        realm_global: &Value<'s>,
    ) -> Result<crate::RealmEntry, Value<'s>> {
        let interp = self.it();
        interp
            .enter_embed_realm_entry(realm_global.as_internal())
            .map_err(Value::from_internal)
    }

    /// Restore the Realm that was active when `entry` was entered.
    pub fn leave_realm(&mut self, entry: crate::RealmEntry) {
        self.it().leave_embed_realm_entry(entry);
    }

    /// Run the Rust text `src` as a Script in `realm_global`'s Realm (ParseScript then
    /// ScriptEvaluation); a parse failure is that Realm's SyntaxError.
    pub fn eval_in_realm(
        &mut self,
        realm_global: &Value<'s>,
        src: &str,
    ) -> Result<Value<'s>, Value<'s>> {
        self.it()
            .with_author_entry(|interp| interp.eval_in_realm(realm_global.as_internal(), src))
            .map(Value::from_internal)
            .map_err(lowered)
    }

    /// Set the opaque host settings token captured by subsequently-created promise reactions.
    pub fn set_host_job_context(&mut self, context: u64) {
        self.it().set_host_job_context(context);
    }

    /// The opaque host settings token of the active host operation.
    pub fn host_job_context(&self) -> u64 {
        self.interp.host_job_context()
    }

    /// Retire one embedder settings object after its owning Document is destroyed.
    pub fn release_host_job_context(&mut self, context: u64) -> bool {
        self.it().release_host_job_context(context)
    }

    /// Global of the topmost executing Script, Module, eval or ECMAScript function.
    pub fn script_caller_global(&self) -> Value<'s> {
        Value::from_internal(self.interp.script_caller_global())
    }

    /// Global of the embedder's most recent script or callback entry.
    pub fn script_entry_global(&self) -> Value<'s> {
        Value::from_internal(self.interp.script_entry_global())
    }

    /// Parse and evaluate one ECMAScript Script Record in the active Realm while a host operation
    /// is already running (HTML's "run a classic script"); the embedder owns the following
    /// microtask checkpoint.
    pub fn eval_classic_script_interruptible(
        &mut self,
        src: &str,
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.eval_classic_script_interruptible_with_origin(src, None)
    }

    /// [`Ctx::eval_classic_script_interruptible`] for a script whose `Error.prototype.stack`
    /// frames report `origin` (`<anonymous>` when None).
    pub fn eval_classic_script_interruptible_with_origin(
        &mut self,
        src: &str,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.it().eval_classic_script(src, origin).map(eval_result)
    }

    /// Decode and evaluate a precompiled Script Record in the active Realm (the snapshot
    /// counterpart of [`Ctx::eval_classic_script_interruptible`]).
    pub fn eval_classic_snapshot_interruptible(
        &mut self,
        bytes: &[u8],
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.it().eval_classic_snapshot(bytes).map(eval_result)
    }

    /// [`Ctx::eval_classic_snapshot_interruptible`] for static host bootstrap code many Realms of
    /// this Agent install: the first call decodes the snapshot and later ones reuse its program.
    pub fn eval_shared_classic_snapshot_interruptible(
        &mut self,
        bytes: &'static [u8],
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.it()
            .eval_shared_classic_snapshot(bytes)
            .map(eval_result)
    }

    /// Run `src` as a script and return its completion value (`Err` the thrown value). No
    /// microtask checkpoint runs: the caller owns the event loop. An interruption is lowered to
    /// a thrown Error of its legacy name.
    pub fn eval_value(&mut self, src: &str) -> Result<Result<Value<'s>, Value<'s>>, ParseError> {
        Ok(match self.eval_value_interruptible(src)? {
            Ok(value) => Ok(value),
            Err(EvalError::Throw(value)) => Err(value),
            Err(EvalError::Interrupted(reason)) => Err(self.make_error(
                crate::Engine::legacy_interrupt_name(reason),
                reason.message(),
            )),
        })
    }

    /// [`Ctx::eval_value`] with host interruption kept distinct from a catchable throw.
    pub fn eval_value_interruptible(
        &mut self,
        src: &str,
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.eval_value_interruptible_with_origin(src, None)
    }

    /// [`Ctx::eval_value_interruptible`] for a script whose `Error.prototype.stack` frames report
    /// `origin` (`<anonymous>` when None).
    pub fn eval_value_interruptible_with_origin(
        &mut self,
        src: &str,
        origin: Option<&SourceOrigin<'_>>,
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        let interp = self.it();
        crate::scrubbed(|| interp.eval_value_entry(src, origin)).map(eval_result)
    }

    /// The snapshot counterpart of [`Ctx::eval_value_interruptible`].
    pub fn eval_snapshot_value_interruptible(
        &mut self,
        bytes: &[u8],
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.it().eval_snapshot_value_entry(bytes).map(eval_result)
    }

    /// [`Ctx::eval_snapshot_value_interruptible`] through the Agent's shared program table (see
    /// [`Ctx::eval_shared_classic_snapshot_interruptible`]).
    pub fn eval_shared_snapshot_value_interruptible(
        &mut self,
        bytes: &'static [u8],
    ) -> Result<Result<Value<'s>, EvalError<'s>>, ParseError> {
        self.it()
            .eval_shared_snapshot_value_entry(bytes)
            .map(eval_result)
    }

    /// The promise of a loaded module's evaluation, top-level await included.
    pub fn module_evaluation_promise(&self, key: &str) -> Option<Value<'s>> {
        self.interp
            .module_evaluation_promise(key)
            .map(Value::from_internal)
    }

    /// Call a JS function value as the event loop does to fire a callback; `Err` is the thrown
    /// value (an interruption lowered to an Error of its legacy name). Works on every tier.
    pub fn call_function(
        &mut self,
        func: &Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        match self.call_function_interruptible(func, this, args) {
            Ok(value) => Ok(value),
            Err(EvalError::Throw(value)) => Err(value),
            Err(EvalError::Interrupted(reason)) => Err(self.make_error(
                crate::Engine::legacy_interrupt_name(reason),
                reason.message(),
            )),
        }
    }

    /// [`Ctx::call_function`] with a host interruption kept out of JavaScript exception flow.
    pub fn call_function_interruptible(
        &mut self,
        func: &Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, EvalError<'s>> {
        self.call_callback_interruptible(func, this, args, &Value::Undefined)
    }

    /// Call with a captured host callback source context: self-hosted platform frames below the
    /// callback cannot replace its incumbent source (see [`Ctx::with_callback_script_caller`]).
    pub fn call_callback_interruptible(
        &mut self,
        func: &Value<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
        script_caller: &Value<'s>,
    ) -> Result<Value<'s>, EvalError<'s>> {
        eval_result(self.it().call_callback_entry(
            func.as_internal(),
            this.into_internal(),
            &Value::slice_as_internal(args),
            script_caller.as_internal(),
        ))
    }

    // ----- promises and rejections --------------------------------------------------------------

    /// A promise's `[[PromiseIsHandled]]` (`None` when `promise` is not a promise).
    pub fn promise_is_handled(&mut self, promise: &Value<'s>) -> Option<bool> {
        self.interp.promise_is_handled(promise.as_internal())
    }

    /// Web IDL "mark as handled". False when `promise` is not a promise.
    pub fn mark_promise_handled(&mut self, promise: &Value<'s>) -> bool {
        self.interp.mark_promise_handled(promise.as_internal())
    }

    /// Add a rejected, still unhandled promise to the outstanding rejected promises weak set of
    /// the active host settings context's global (HTML "notify about rejected promises").
    pub fn add_outstanding_rejection(&mut self, promise: &Value<'s>) -> bool {
        self.interp.add_outstanding_rejection(promise.as_internal())
    }

    /// The HostPromiseRejectionTracker "handle" operations since the last call on promises of an
    /// outstanding rejected promises weak set, as `(promise, reason)` handles in order.
    pub fn take_handled_rejections(&mut self) -> Vec<(Handle, Handle)> {
        self.take_handled_rejections_in_contexts()
            .into_iter()
            .map(|(_, promise, reason)| (promise, reason))
            .collect()
    }

    /// [`Ctx::take_handled_rejections`] as `(context, promise, reason)`, with the host settings
    /// context whose global's weak set held each promise.
    pub fn take_handled_rejections_in_contexts(&mut self) -> Vec<(u64, Handle, Handle)> {
        let taken = self.interp.take_handled_rejections_in_contexts();
        self.rejection_handles(taken)
    }

    /// Drain the promises rejected without a handler, as `(context, promise, reason)` handles in
    /// rejection order (see [`Engine::take_unhandled_rejections_in_contexts`](
    /// crate::Engine::take_unhandled_rejections_in_contexts)).
    pub fn take_unhandled_rejections_in_contexts(&mut self) -> Vec<(u64, Handle, Handle)> {
        let taken = self.interp.take_unhandled_rejection_values();
        self.rejection_handles(taken)
    }

    /// Hold taken `(context, promise, reason)` triples in handles.
    fn rejection_handles(
        &self,
        taken: Vec<(u64, Internal, Internal)>,
    ) -> Vec<(u64, Handle, Handle)> {
        // Making a handle runs no collection: the taken values need no other root meanwhile.
        taken
            .into_iter()
            .map(|(context, promise, reason)| {
                let hold = |value: Internal| {
                    Handle::new(self, &Value::from_internal(value))
                        .expect("the engine's own values, outside a collection")
                };
                (context, hold(promise), hold(reason))
            })
            .collect()
    }

    // ----- jobs and collections -----------------------------------------------------------------

    /// Execute all currently queued promise reaction jobs synchronously.
    pub fn drain_microtasks_for_host(&mut self) {
        self.it().drain_microtasks_for_host();
    }

    /// Drain the microtask queue to quiescence (a checkpoint), as
    /// [`Engine::run_microtasks`](crate::Engine::run_microtasks).
    pub fn run_microtasks(&mut self) {
        self.it().run_microtasks_entry();
    }

    /// [`Ctx::run_microtasks`] keeping a host interruption as control flow, as
    /// [`Engine::run_microtasks_interruptible`](crate::Engine::run_microtasks_interruptible).
    pub fn run_microtasks_interruptible(&mut self) -> Result<(), InterruptReason> {
        let interp = self.it();
        crate::scrubbed(|| interp.run_microtasks_interruptible_entry())
    }

    /// Whether jobs are queued, as [`Engine::has_pending_jobs`](crate::Engine::has_pending_jobs).
    pub fn has_pending_jobs(&self) -> bool {
        self.interp.has_pending_jobs()
    }

    /// Run a full (major) collection and return the number of reclaimed heap objects.
    #[inline(always)]
    pub fn collect_garbage_for_host(&mut self) -> i64 {
        self.interp.collect_garbage_for_host()
    }

    /// The number of heap objects the engine tracks.
    pub fn live_object_count(&self) -> i64 {
        self.interp.live_object_count()
    }

    /// Defer collections (nestable) until [`Ctx::resume_gc`].
    pub fn suspend_gc(&mut self) {
        self.interp.suspend_gc();
    }

    /// End one [`Ctx::suspend_gc`].
    pub fn resume_gc(&mut self) {
        self.interp.resume_gc();
    }
}

/// The engine form of an [`IndexedLength`].
fn indexed_length(f: IndexedLength) -> fn(&Interp, &Internal) -> u32 {
    // SAFETY: references to `Ctx` and `Interp`, and to values of either form, are thin
    // references, ABI-compatible as fn pointer arguments.
    unsafe { std::mem::transmute::<IndexedLength, fn(&Interp, &Internal) -> u32>(f) }
}

/// The engine form of an [`IndexedNames`].
fn indexed_names(f: IndexedNames) -> fn(&Interp, &Internal) -> Vec<String> {
    // SAFETY: as for `indexed_length`.
    unsafe { std::mem::transmute::<IndexedNames, fn(&Interp, &Internal) -> Vec<String>>(f) }
}

/// The hooks entering and leaving an embedder's settings context around a promise job
/// ([`Engine::set_host_job_context_hooks`](crate::Engine::set_host_job_context_hooks)).
pub type HostJobContextEnter = for<'s> fn(&mut Ctx<'s>, u64);

/// See [`HostJobContextEnter`].
pub type HostJobContextLeave = for<'s> fn(&mut Ctx<'s>);

/// The engine forms of the settings-context hooks.
pub(crate) fn host_job_context_hooks(
    enter: HostJobContextEnter,
    leave: HostJobContextLeave,
) -> (fn(&mut Interp, u64), fn(&mut Interp)) {
    // SAFETY: as for `indexed_length`: thin references and a `u64`.
    unsafe {
        (
            std::mem::transmute::<HostJobContextEnter, fn(&mut Interp, u64)>(enter),
            std::mem::transmute::<HostJobContextLeave, fn(&mut Interp)>(leave),
        )
    }
}
