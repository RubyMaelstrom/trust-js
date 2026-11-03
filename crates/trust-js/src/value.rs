//! Runtime values and the object model. Objects are `RefCell<Object>` cells of their Agent's
//! mark-sweep heap ([`Gc`] is a cell's address). Properties are stored in insertion order in a
//! small map.

use crate::ast::Function;
use crate::atom::PropKey;
use crate::interpreter::{Env, Interp};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
#[cfg(feature = "architecture-diagnostics")]
#[path = "allocation_diagnostics.rs"]
mod allocation_diagnostics;
#[path = "elem_census.rs"]
pub(crate) mod elem_census;
/// W2c step 2b; its users come with step 2c (the plan's "W2c step 2: words").
#[path = "word_block.rs"]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod word_block;

#[path = "property_shapes.rs"]
mod property_shapes;
#[cfg(test)]
use property_shapes::SHAPE_LAYOUT_CACHE_LIMIT;
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) use property_shapes::SHAPE_LAYOUT_PAGE_BITS;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )
))]
pub(crate) use property_shapes::SHAPE_LAYOUT_PAGE_COUNT;
pub(crate) use property_shapes::{is_cacheable_shape, SHAPE_LAYOUT_PAGE_SIZE, SHAPE_UNCACHEABLE};
use property_shapes::{LayoutEntry, LayoutPage, ShapeLayouts};

#[cfg(all(test, feature = "embed"))]
#[path = "property_shape_tests.rs"]
mod property_shape_tests;

pub(crate) use crate::gc_handle::{Gc, WeakGc};

/// A native (Rust-implemented) function. It can only throw (via `Err`), never break/return/continue,
/// so a plain `Result<Value, Value>` (Err = the thrown value) is the whole contract. An embedder's
/// callback retains its own scoped signature and maps the Result at a typed boundary.
pub(crate) type NativeFn = fn(&mut Interp, Value, &[Value]) -> Result<Value, Value>;

/// One identity-bearing allocation retained below a data-carrying native callable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RetainedManagedAllocation {
    pub(crate) identity_domain: &'static str,
    pub(crate) identity: usize,
    pub(crate) requested_bytes: usize,
}

impl RetainedManagedAllocation {
    /// Describe one live allocation. The `(identity_domain, identity)` pair must be unique among
    /// simultaneously live allocations and stable for the duration of a diagnostic visit.
    pub fn new(identity_domain: &'static str, identity: usize, requested_bytes: usize) -> Self {
        Self {
            identity_domain,
            identity,
            requested_bytes,
        }
    }

    /// Describe the requested payload of an `Rc` allocation without relying on Rust's private
    /// refcount-header layout.
    pub fn rc<T: ?Sized>(
        identity_domain: &'static str,
        value: &Rc<T>,
        requested_bytes: usize,
    ) -> Self {
        Self::new(
            identity_domain,
            Rc::as_ptr(value) as *const () as usize,
            requested_bytes,
        )
    }
}

/// Canonical sinks available to a native callable's retained-memory reporter.
pub trait NativeRetainedMemoryVisitor {
    fn allocation(&mut self, allocation: RetainedManagedAllocation);
}

/// A native function that carries captured state, unlike the bare-`fn` [`NativeFn`]. Its state
/// holds no JavaScript value (S6 step 11b): those are engine captures ([`CaptureClosure`]).
pub(crate) type NativeClosure = dyn Fn(&mut Interp, Value, &[Value]) -> Result<Value, Value>;

/// A native operation with engine-owned JavaScript captures. Unlike opaque Rust
/// closures, these captures participate in the collector's ordinary reachability
/// graph. The final slice contains the immutable registration-time captures.
pub(crate) type NativeCaptureFn =
    fn(&mut Interp, Value, &[Value], &[Value]) -> Result<Value, Value>;

/// A [`NativeClosure`] with engine-owned JavaScript captures, the final slice: a native
/// function that carries Rust state and JavaScript values both (S6 step 11b).
pub(crate) type CaptureClosure =
    dyn Fn(&mut Interp, Value, &[Value], &[Value]) -> Result<Value, Value>;

/// Retained-memory companion of a native closure's Rust state, for the managed-memory
/// diagnostics: it enumerates every allocation below its own inline payload. It reports no
/// JavaScript value (S6 step 11b): a native closure keeps its values as engine captures, which
/// the collector traces with the function.
pub trait NativeCallableRetained {
    fn scan_retained_memory(&self, visitor: &mut dyn NativeRetainedMemoryVisitor);
}

/// The retained-memory companion of a native closure's Rust state: what its allocations are for
/// the managed-memory diagnostics.
pub enum NativeCaptures {
    /// The closure's state is not described.
    None,
    /// Its state's managed allocations are reported by this companion, which may share the
    /// closure's state allocation.
    Reported(Rc<dyn NativeCallableRetained>),
}

impl NativeCaptures {
    pub(crate) fn into_reporter(self) -> Option<Rc<dyn NativeCallableRetained>> {
        match self {
            NativeCaptures::None => None,
            NativeCaptures::Reported(reporter) => Some(reporter),
        }
    }
}

/// The engine's thread-bound execution word. Match views are decoded only while inspecting
/// a value; execution storage, bindings and immutable argument windows all use this type.
/// Every Number constructor canonicalizes NaN immediately.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct Value(u64, std::marker::PhantomData<*const ()>);

/// An ephemeral decoded match view. This enum is never used as execution storage.
#[derive(Clone, Copy)]
#[repr(u64)]
pub(crate) enum ValueView {
    Undefined = 0,
    Empty = 1,
    Null = 2,
    Bool(bool) = 3,
    Num(f64) = 4,
    BigInt(crate::bigint::JsBigInt) = 5,
    Str(crate::lstr::LStr) = 6,
    Sym(SymbolRef) = 7,
    Obj(Gc) = 8,
}

impl Default for Value {
    fn default() -> Self {
        Self::Undefined
    }
}

const _: () = {
    assert!(std::mem::size_of::<Value>() == 8);
    assert!(std::mem::align_of::<Value>() == std::mem::align_of::<u64>());
    assert!(!std::mem::needs_drop::<Value>());
};

#[allow(non_snake_case, non_upper_case_globals)]
impl Value {
    pub(crate) const Undefined: Self = Self(PACK_UNDEFINED, std::marker::PhantomData);
    /// The internal completion marker. Public value boundaries normalize it to undefined.
    pub(crate) const Empty: Self = Self(PACK_EMPTY, std::marker::PhantomData);
    pub(crate) const Null: Self = Self(PACK_NULL, std::marker::PhantomData);

    #[inline]
    pub(crate) const fn Bool(value: bool) -> Self {
        Self(PACK_BOOL | value as u64, std::marker::PhantomData)
    }

    /// ECMA-262 Number Type: all IEEE NaNs denote one language NaN. This leaves backing
    /// ArrayBuffer bytes and Wasm's internal floating payloads untouched.
    #[inline]
    pub(crate) const fn Num(value: f64) -> Self {
        let bits = if value.is_nan() {
            PACK_CANON_NAN
        } else {
            value.to_bits()
        };
        Self(bits, std::marker::PhantomData)
    }

    #[inline]
    pub(crate) fn BigInt(value: crate::bigint::JsBigInt) -> Self {
        Self(
            PACK_BIGINT | unsafe { Self::into_word(value) },
            std::marker::PhantomData,
        )
    }

    /// UTF-8 engine text, with a lone UTF-16 surrogate represented by the scalar
    /// U+10F800 + (unit - 0xD800). Use the text/UTF-16 conversion methods at host boundaries;
    /// this constructor receives an already encoded immutable string-cell handle.
    #[inline]
    pub(crate) fn Str(value: crate::lstr::LStr) -> Self {
        Self(
            PACK_STR | unsafe { Self::into_word(value) },
            std::marker::PhantomData,
        )
    }

    #[inline]
    pub(crate) fn Sym(value: SymbolRef) -> Self {
        Self(
            PACK_SYM | unsafe { Self::into_word(value) },
            std::marker::PhantomData,
        )
    }

    #[inline]
    pub(crate) fn Obj(value: Gc) -> Self {
        Self(
            PACK_OBJ | unsafe { Self::into_word(value) },
            std::marker::PhantomData,
        )
    }

    /// Reconstruct an execution value from its bits.
    ///
    /// # Safety
    /// The word must be a valid execution encoding. A reference payload must name the start
    /// of a live cell of the encoded kind in the owning Agent. Bit validation cannot establish
    /// cell liveness, type, ownership, or provenance, and never admits a property-only word.
    #[inline]
    pub(crate) unsafe fn from_bits(bits: u64) -> Self {
        debug_assert_ne!(
            bits & !PACK_PAYLOAD,
            PACK_LAZY_PROTO,
            "a deferred prototype's word copied without materializing"
        );
        debug_assert!(
            Self::valid_bits(bits),
            "invalid execution word {bits:#018x}"
        );
        Self(bits, std::marker::PhantomData)
    }

    #[inline]
    pub(crate) const fn bits(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub(crate) fn view(&self) -> ValueView {
        match self.tag() {
            PACK_UNDEFINED => ValueView::Undefined,
            PACK_EMPTY => ValueView::Empty,
            PACK_NULL => ValueView::Null,
            PACK_BOOL => ValueView::Bool(self.0 & 1 != 0),
            PACK_BIGINT => ValueView::BigInt(unsafe { self.read_word() }),
            PACK_STR => ValueView::Str(unsafe { self.read_word() }),
            PACK_SYM => ValueView::Sym(unsafe { self.read_word() }),
            PACK_OBJ => ValueView::Obj(unsafe { self.read_word() }),
            PACK_LAZY_PROTO => unreachable!("a deferred prototype reached execution storage"),
            _ => ValueView::Num(f64::from_bits(self.0)),
        }
    }

    /// The untagged cell address held as a strong GC edge. Scalar words are never roots.
    #[inline]
    pub(crate) fn cell_word(&self) -> Option<u64> {
        // The tags of BigInt, string and Symbol cells are consecutive, the object tag apart.
        const _: () =
            assert!(PACK_STR == PACK_BIGINT + (1 << 48) && PACK_SYM == PACK_STR + (1 << 48));
        let tag = self.0 >> 48;
        (tag.wrapping_sub(PACK_BIGINT >> 48) < 3 || tag == PACK_OBJ >> 48)
            .then_some(self.0 & PACK_PAYLOAD)
    }
}

/// Emit a compact diagnostic for a computed property read whose base is nullish.
///
/// This is intentionally opt-in: the normal error remains the spec-compatible generic
/// TypeError, while `TRUST_JS_NULLISH_TRACE=1` makes failures in minified third-party code
/// actionable without formatting or traversing arbitrary heap objects.
pub(crate) fn trace_nullish_property(site: &str, key: &Value) {
    if std::env::var_os("TRUST_JS_NULLISH_TRACE").is_none() {
        return;
    }
    let key = match &(key).view() {
        ValueView::Str(s) => format!("string:{:?}", s.text()),
        ValueView::Num(n) => format!("number:{n:?}"),
        ValueView::Bool(v) => format!("boolean:{v}"),
        ValueView::Null => "null".to_owned(),
        ValueView::Undefined => "undefined".to_owned(),
        ValueView::Empty => "empty".to_owned(),
        ValueView::BigInt(_) => "bigint".to_owned(),
        ValueView::Sym(_) => "symbol".to_owned(),
        ValueView::Obj(_) => "object".to_owned(),
    };
    eprintln!("TRust JS: nullish computed-property read site={site} key={key}");
}

/// Property-only word storage. A deferred prototype can be held and promoted only here.
#[repr(transparent)]
struct PropWord(Cell<u64>, std::marker::PhantomData<*const ()>);

const PACK_PAYLOAD: u64 = 0x0000_ffff_ffff_ffff;
pub(crate) const PACK_UNDEFINED: u64 = 0x7ff9_0000_0000_0000;
pub(crate) const PACK_EMPTY: u64 = 0x7ffa_0000_0000_0000;
pub(crate) const PACK_NULL: u64 = 0x7ffb_0000_0000_0000;
pub(crate) const PACK_BOOL: u64 = 0x7ffc_0000_0000_0000;
pub(crate) const PACK_BIGINT: u64 = 0x7ffd_0000_0000_0000;
pub(crate) const PACK_STR: u64 = 0x7ffe_0000_0000_0000;
pub(crate) const PACK_SYM: u64 = 0x7fff_0000_0000_0000;
pub(crate) const PACK_OBJ: u64 = 0xfff9_0000_0000_0000;
pub(crate) const PACK_LAZY_PROTO: u64 = 0xfffa_0000_0000_0000;
pub(crate) const PACK_CANON_NAN: u64 = 0x7ff8_0000_0000_0000;

#[path = "lazy_function_prototype.rs"]
mod lazy_function_prototype;
#[cfg(test)]
pub(crate) use lazy_function_prototype::materialized as prototypes_materialized;
use lazy_function_prototype::LazyFunctionPrototype;

/// The requested bytes a deferred prototype's payload adds to property storage (tests).
#[cfg(test)]
pub(crate) const DEFERRED_PROTOTYPE_BYTES: usize = std::mem::size_of::<LazyFunctionPrototype>();

/// The handle a PACK_LAZY_PROTO word holds: a function's deferred `prototype`. P1 S5: a payload
/// cell that traces the function and the prototype's parent (or, once
/// materialized, the prototype), so the word owns nothing and dropping or overwriting it
/// releases nothing; the cell dies at the first collection that finds no word naming it.
type LazyProto = crate::gc_box::GcBox<LazyFunctionPrototype>;

impl Value {
    pub(crate) fn scalar_bits(value: &Value) -> Option<u64> {
        value.cell_word().is_none().then_some(value.bits())
    }

    #[inline]
    fn tag(&self) -> u64 {
        self.0 & !PACK_PAYLOAD
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.0 == PACK_EMPTY
    }

    #[inline]
    pub(crate) fn is_undefined(&self) -> bool {
        self.0 == PACK_UNDEFINED
    }

    /// Inspect whether this execution value is a primitive.
    #[inline]
    pub(crate) fn is_primitive(&self) -> bool {
        !matches!(self.tag(), PACK_OBJ | PACK_LAZY_PROTO)
    }

    /// Inspect an object without exposing its payload to a reentrant observer. Property
    /// lookup uses the callback to take a separate value snapshot before running JS.
    #[inline]
    pub(crate) fn with_object<R>(&self, f: impl FnOnce(&Gc) -> R) -> Option<R> {
        if self.tag() != PACK_OBJ {
            return None;
        }
        let object = unsafe { self.read_word::<Gc>() };
        Some(f(&object))
    }

    /// Borrow string text for an observer-free primitive operation. As with `with_object`,
    /// mutable frame slots must not stay borrowed across JS execution.
    #[inline]
    pub(crate) fn with_string<R>(&self, f: impl FnOnce(&crate::lstr::LStr) -> R) -> Option<R> {
        if self.tag() != PACK_STR {
            return None;
        }
        let string = unsafe { self.read_word::<crate::lstr::LStr>() };
        Some(f(&string))
    }

    /// Inspect a Number without decoding a match view or inspecting a reference payload.
    #[inline]
    pub(crate) fn number(&self) -> Option<f64> {
        match self.tag() {
            PACK_UNDEFINED | PACK_EMPTY | PACK_NULL | PACK_BOOL | PACK_BIGINT | PACK_STR
            | PACK_SYM | PACK_OBJ | PACK_LAZY_PROTO => None,
            _ => Some(f64::from_bits(self.0)),
        }
    }

    unsafe fn into_word<T: Copy>(value: T) -> u64 {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<usize>());
        let mut word = 0usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                &value as *const T as *const u8,
                &mut word as *mut usize as *mut u8,
                std::mem::size_of::<T>(),
            );
        }
        let word = word as u64;
        assert_eq!(
            word & !PACK_PAYLOAD,
            0,
            "pointer does not fit NaN-box payload"
        );
        word
    }

    /// Encoding validation only: reference liveness, exact cell type and ownership remain
    /// the raw constructor's caller contract. No heap borrow is taken during validation.
    const fn valid_bits(bits: u64) -> bool {
        let payload = bits & PACK_PAYLOAD;
        match bits & !PACK_PAYLOAD {
            PACK_UNDEFINED | PACK_EMPTY | PACK_NULL => payload == 0,
            PACK_BOOL => payload <= 1,
            PACK_BIGINT | PACK_STR | PACK_SYM | PACK_OBJ => payload != 0 && payload & 7 == 0,
            PACK_LAZY_PROTO => false,
            _ => !f64::from_bits(bits).is_nan() || bits == PACK_CANON_NAN,
        }
    }

    unsafe fn read_word<T: Copy>(&self) -> T {
        unsafe { Self::decode_payload(self.0) }
    }

    unsafe fn decode_payload<T: Copy>(bits: u64) -> T {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<usize>());
        let word = (bits & PACK_PAYLOAD) as usize;
        let mut value = std::mem::MaybeUninit::<T>::uninit();
        unsafe {
            std::ptr::copy_nonoverlapping(
                &word as *const usize as *const u8,
                value.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            );
            value.assume_init()
        }
    }
}

impl PropWord {
    #[inline]
    fn from_value(value: Value) -> Self {
        debug_assert_ne!(value.tag(), PACK_LAZY_PROTO);
        Self(Cell::new(value.0), std::marker::PhantomData)
    }

    #[inline]
    fn tag(&self) -> u64 {
        self.0.get() & !PACK_PAYLOAD
    }

    #[cfg(test)]
    fn bits(&self) -> u64 {
        self.0.get()
    }

    #[inline]
    fn materialize(&self) -> Value {
        if self.tag() == PACK_LAZY_PROTO {
            // SAFETY: only defer_function_prototype publishes this property-only tag.
            let lazy = unsafe { Value::decode_payload::<LazyProto>(self.0.get()) };
            let prototype = lazy.materialize();
            // Representation-only promotion invokes no JS. Record the owning function before
            // publishing its new object edge, with no safepoint between barrier and store.
            lazy.mark_owner_card();
            self.0
                .set(PACK_OBJ | unsafe { Value::into_word(prototype) });
            Value::Obj(prototype)
        } else {
            // SAFETY: ordinary property words hold valid execution values; the property-only
            // deferred tag was handled above, before constructing an execution value.
            unsafe { Value::from_bits(self.0.get()) }
        }
    }

    #[inline]
    fn with_object<R>(&self, f: impl FnOnce(&Gc) -> R) -> Option<R> {
        if self.tag() != PACK_OBJ {
            return None;
        }
        // SAFETY: the property holds an ordinary Object word, whose live cell stays in the
        // observer-free inspection's owning property snapshot.
        unsafe { Value::from_bits(self.0.get()) }.with_object(f)
    }

    #[inline]
    fn number(&self) -> Option<f64> {
        if self.tag() == PACK_LAZY_PROTO {
            return None;
        }
        // SAFETY: a non-deferred property contains an execution word.
        unsafe { Value::from_bits(self.0.get()) }.number()
    }

    #[inline]
    fn trace_gc_edges<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if self.tag() == PACK_LAZY_PROTO {
            sink.word(self.0.get() & PACK_PAYLOAD);
        } else {
            // SAFETY: the property-only tag was handled separately above.
            sink.value(&unsafe { Value::from_bits(self.0.get()) });
        }
    }
}

// A word owns nothing, so it has no drop glue; dropping or overwriting one, and every
// teardown loop over execution words, compiles to nothing.
const _: () = assert!(!std::mem::needs_drop::<Value>());

#[cfg(test)]
mod packed_value_tests {
    use super::*;
    #[cfg(feature = "embed")]
    use crate::scope::ValueKind;
    use crate::value::ValueView;

    #[test]
    fn copy_values_views_and_slot_cells_have_pinned_layouts_and_thread_affinity() {
        fn copy<T: Copy>() {}
        copy::<Value>();
        copy::<ValueView>();
        copy::<crate::lstr::LStr>();
        copy::<crate::bigint::JsBigInt>();
        assert_eq!(std::mem::size_of::<Value>(), 8);
        assert_eq!(std::mem::size_of::<ValueView>(), 16);
        assert_eq!(std::mem::size_of::<Cell<Value>>(), 8);
        assert_eq!(
            std::mem::align_of::<Cell<Value>>(),
            std::mem::align_of::<Value>()
        );
        assert_eq!(std::mem::size_of::<PropWord>(), 8);
        assert_eq!(std::mem::size_of::<Property>(), 16);
        // If any type becomes Send/Sync, the placeholder below becomes ambiguous and this
        // fixture stops compiling. Plain packed bits must retain the Agent's thread affinity.
        macro_rules! thread_bound {
            ($ty:ty) => {{
                trait AmbiguousIfSend<A> {
                    fn check() {}
                }
                struct IsSend;
                impl<T: ?Sized> AmbiguousIfSend<()> for T {}
                impl<T: ?Sized + Send> AmbiguousIfSend<IsSend> for T {}
                let _ = <$ty as AmbiguousIfSend<_>>::check;
                trait AmbiguousIfSync<A> {
                    fn check() {}
                }
                struct IsSync;
                impl<T: ?Sized> AmbiguousIfSync<()> for T {}
                impl<T: ?Sized + Sync> AmbiguousIfSync<IsSync> for T {}
                let _ = <$ty as AmbiguousIfSync<_>>::check;
            }};
        }
        thread_bound!(Value);
        thread_bound!(ValueView);
        thread_bound!(PropWord);
        thread_bound!(Property);
        thread_bound!(crate::lstr::LStr);
        thread_bound!(crate::bigint::JsBigInt);
        thread_bound!(Gc);
        thread_bound!(Env);
        #[cfg(feature = "embed")]
        {
            copy::<crate::scope::Value<'static>>();
            copy::<ValueKind<'static>>();
            thread_bound!(crate::scope::Value<'static>);
            thread_bound!(crate::scope::ValueKind<'static>);
        }
    }

    #[test]
    fn views_preserve_every_discriminant_and_payload() {
        assert!(matches!(Value::Undefined.view(), ValueView::Undefined));
        assert!(matches!(Value::Empty.view(), ValueView::Empty));
        assert!(matches!(Value::Null.view(), ValueView::Null));
        for boolean in [false, true] {
            assert!(
                matches!(Value::Bool(boolean).view(), ValueView::Bool(actual) if actual == boolean)
            );
        }
        for number in [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::from_bits(0xfff8_0123_4567_89ab),
        ] {
            assert!(
                matches!(Value::Num(number).view(), ValueView::Num(actual) if actual.to_bits() == if number.is_nan() { PACK_CANON_NAN } else { number.to_bits() })
            );
        }
        let mut interp = crate::interpreter::Interp::new();
        let integer = crate::bigint::JsBigInt::parse_dec("123456789012345678901234567890").unwrap();
        assert!(
            matches!(Value::BigInt(integer).view(), ValueView::BigInt(actual) if actual.cell_word() == integer.cell_word())
        );
        let string = crate::lstr::LStr::from("view payload");
        assert!(
            matches!(Value::Str(string).view(), ValueView::Str(actual) if crate::lstr::LStr::ptr_eq(&actual, &string))
        );
        let symbol = interp.new_symbol(Some("view symbol".into()));
        assert!(
            matches!(symbol.view(), ValueView::Sym(actual) if symbol.cell_word() == Some(symbol_identity(&actual) as u64))
        );
        let object = Object::new(None);
        assert!(
            matches!(Value::Obj(object).view(), ValueView::Obj(actual) if Gc::ptr_eq(&actual, &object))
        );
    }

    #[test]
    fn number_constructors_canonicalize_every_reserved_nan_prefix() {
        for high in (0x7ff0u64..=0x7fff).chain(0xfff0..=0xffff) {
            for payload in [1, 0x8000_0000_0000, PACK_PAYLOAD] {
                let number = f64::from_bits(high << 48 | payload);
                assert!(number.is_nan());
                let value = Value::Num(number);
                assert_eq!(value.bits(), PACK_CANON_NAN);
                assert!(matches!(value.view(), ValueView::Num(n) if n.is_nan()));
            }
        }
        for number in [
            0.0,
            -0.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(Value::Num(number).bits(), number.to_bits());
        }
    }

    /// Migrated from the dormant 32-bit tagged-handle scaffold. Production references use
    /// 48-bit cell addresses: validate null/alignment, never impose that old 32-bit limit.
    #[test]
    fn raw_execution_encoding_rejects_bad_scalars_nan_and_reference_payloads() {
        for valid in [
            PACK_UNDEFINED,
            PACK_EMPTY,
            PACK_NULL,
            PACK_BOOL,
            PACK_BOOL | 1,
            PACK_CANON_NAN,
            0.0f64.to_bits(),
            (-0.0f64).to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
        ] {
            assert!(Value::valid_bits(valid));
            assert_eq!(unsafe { Value::from_bits(valid) }.bits(), valid);
        }
        for invalid in [
            PACK_UNDEFINED | 1,
            PACK_EMPTY | 1,
            PACK_NULL | 1,
            PACK_BOOL | 2,
            PACK_OBJ,
            PACK_STR,
            PACK_BIGINT,
            PACK_SYM,
            PACK_OBJ | 3,
            PACK_STR | 1,
            PACK_BIGINT | 7,
            PACK_SYM | 5,
            PACK_CANON_NAN | 1,
            0xfff8_0000_0000_0000,
            PACK_LAZY_PROTO | 8,
            0xfffb_0000_0000_0001,
        ] {
            assert!(!Value::valid_bits(invalid), "{invalid:#018x}");
            #[cfg(debug_assertions)]
            assert!(std::panic::catch_unwind(|| {
                // The checked constructor must reject before constructing any cell handle.
                let _ = unsafe { Value::from_bits(invalid) };
            })
            .is_err());
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn cell_payload_fit_remains_a_release_assertion_and_accepts_48_bits() {
        let wide = 0x0000_1234_5678_9000usize;
        assert_eq!(unsafe { Value::into_word(wide) }, wide as u64);
        assert!(std::panic::catch_unwind(|| unsafe { Value::into_word(1usize << 48) }).is_err());
    }

    #[test]
    fn borrowed_engine_text_survives_copied_frontier_append_growth_and_precise_gc() {
        use crate::gc_core::CollectionKind;
        use crate::rooting::Rooted;
        use crate::value::property_layout_cell_tests::{collect, isolated, Words};

        let (heap, _release, _active) = isolated();
        let prefix = "borrowed engine text: ".repeat(64);
        let initial = crate::lstr::LStr::from(prefix.as_str()).concat_grown("");
        let original = Rooted::new(Value::Str(initial));
        let original_word = initial.cell_word() as usize;
        let original_weak = crate::gc_handle::WeakRef::to(original_word);
        let borrowed = original.as_engine_text().unwrap();
        let original_address = borrowed.as_ptr();
        let original_len = borrowed.len();

        let mut copied = initial;
        assert!(copied.append_in_place("frontier"));
        assert!(crate::lstr::LStr::ptr_eq(
            copied.view_owner().unwrap(),
            &initial
        ));
        assert_eq!(copied.text().as_ptr(), original_address);
        let mut current = Rooted::new(Value::Str(copied));
        let mut expected = format!("{prefix}frontier");
        // No conservative stack scan or additional explicit roots: only these Rooted
        // receivers retain the dynamic strings, including the original borrowed bytes.
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            collect(&heap, &Words(Vec::new()), kind);
            assert!(heap
                .core
                .borrow()
                .owns_cell(original_word, heap.string_desc));
            assert_eq!(borrowed, prefix);
            assert_eq!(borrowed.len(), original_len);
            assert_eq!(borrowed.as_ptr(), original_address);
            assert_eq!(current.as_engine_text().as_deref(), Some(expected.as_str()));
        }

        let suffix = "replacement growth: ".repeat(prefix.len());
        assert!(!copied.append_in_place(&suffix));
        copied.append_or_replace(&crate::lstr::LStr::from(suffix.as_str()));
        assert_ne!(copied.text().as_ptr(), original_address);
        current.replace(Value::Str(copied));
        expected.push_str(&suffix);
        let replacement_word = copied.cell_word() as usize;
        let replacement_weak = crate::gc_handle::WeakRef::to(replacement_word);
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            collect(&heap, &Words(Vec::new()), kind);
            let core = heap.core.borrow();
            assert!(core.owns_cell(original_word, heap.string_desc));
            assert!(core.owns_cell(replacement_word, heap.string_desc));
            assert_eq!(borrowed, prefix);
            assert_eq!(borrowed.len(), original_len);
            assert_eq!(borrowed.as_ptr(), original_address);
            assert_eq!(current.as_engine_text().as_deref(), Some(expected.as_str()));
        }

        drop(current);
        drop(original);
        collect(&heap, &Words(Vec::new()), CollectionKind::Major);
        assert!(original_weak.live_ptr().is_none());
        assert!(replacement_weak.live_ptr().is_none());
    }

    /// Execution values and packed operands are the same complete canonical word. Decoded
    /// views remain separate enums and are never used to infer execution payload offsets.
    #[test]
    fn execution_value_layout_is_one_canonical_word() {
        assert_eq!(std::mem::size_of::<Value>(), 8);
        assert_eq!(std::mem::align_of::<Value>(), 8);
        assert_eq!(std::mem::size_of::<Option<Value>>(), 16);
        let word = |value: &Value| unsafe { std::ptr::from_ref(value).cast::<u64>().read() };
        assert_eq!(word(&Value::Undefined), PACK_UNDEFINED);
        assert_eq!(word(&Value::Empty), PACK_EMPTY);
        assert_eq!(word(&Value::Null), PACK_NULL);
        for boolean in [false, true] {
            assert_eq!(word(&Value::Bool(boolean)), PACK_BOOL | u64::from(boolean));
        }
        assert_eq!(word(&Value::Num(1.5)), 1.5f64.to_bits());
        let object = Object::new(None);
        let layout = jit_layout(&object);
        let value = Value::Obj(object);
        assert_eq!(word(&value) & !PACK_PAYLOAD, PACK_OBJ);
        assert_eq!(
            (word(&value) & PACK_PAYLOAD) as usize + layout.obj_from_cell,
            object.object_ptr() as usize,
        );
    }

    /// Number and emptiness inspection reads a word's bits and nothing else: every word keeps
    /// its bits, a Number's (or NaN's) value comes back exactly, and an object's word still
    /// names its object. A word owns nothing, so its bits are all there is to keep.
    #[test]
    fn numeric_inspection_preserves_number_bits_without_materializing_properties() {
        for number in [
            0.0,
            -0.0,
            1.25,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(PACK_OBJ | 123),
            f64::from_bits(PACK_SYM | 456),
        ] {
            let packed = Value::Num(number);
            let bits = packed.bits();
            let actual = packed.number().expect("Number tag");
            assert!(number.is_nan() && actual.is_nan() || number.to_bits() == actual.to_bits());
            assert!(!packed.is_empty());
            assert_eq!(packed.bits(), bits);
        }
        let object = Object::new(None);
        for value in [
            Value::Undefined,
            Value::Empty,
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::Str("number".into()),
            Value::Obj(object),
        ] {
            let empty = matches!((value).view(), ValueView::Empty);
            let is_object = matches!((value).view(), ValueView::Obj(_));
            let packed = value;
            let bits = packed.bits();
            for _ in 0..20 {
                assert!(packed.number().is_none());
                assert_eq!(packed.is_empty(), empty);
            }
            assert_eq!(packed.bits(), bits);
            if is_object {
                assert!(
                    matches!(packed.clone().view(), ValueView::Obj(actual) if Gc::ptr_eq(&actual, &object))
                );
            }
        }
    }

    #[test]
    fn property_scalar_inspection_never_materializes_a_deferred_prototype() {
        let mut engine = crate::Engine::new();
        engine.eval("function unobserved(){}", false).unwrap();
        let env = engine.interp.global_env;
        let function = engine.interp.get_var("unobserved", &env).ok().unwrap();
        let object_handle = function.as_obj().unwrap();
        let object = object_handle.borrow();
        let prototype = object.props.get_named("prototype").unwrap();
        assert_eq!(prototype.packed.tag(), PACK_LAZY_PROTO);
        let bits = prototype.packed.bits();
        for _ in 0..20 {
            assert!(!prototype.is_empty());
            assert!(prototype.number_value().is_none());
        }
        assert_eq!(prototype.packed.bits(), bits);
        assert!(matches!((prototype.value()).view(), ValueView::Obj(_)));
        assert_eq!(prototype.packed.tag(), PACK_OBJ);
        let mut property = Property::plain(Value::Num(3.0));
        assert_eq!(property.number_value(), Some(3.0));
        property.set_accessor(true);
        assert!(
            property.number_value().is_none(),
            "accessor payload is never a data Number"
        );
    }

    #[test]
    fn execution_value_round_trips_scalar_encodings() {
        let scalar = |value: Value| {
            // SAFETY: scalar constructors provide valid encodings without reference payloads.
            unsafe { Value::from_bits(value.bits()) }
        };
        assert!(matches!(
            scalar(Value::Undefined).view(),
            ValueView::Undefined
        ));
        assert!(matches!(scalar(Value::Empty).view(), ValueView::Empty));
        assert!(matches!(scalar(Value::Null).view(), ValueView::Null));
        for boolean in [false, true] {
            assert!(matches!(scalar(Value::Bool(boolean)).view(),
                ValueView::Bool(actual) if actual == boolean));
        }
        for number in [0.0f64, -0.0, 42.5, f64::INFINITY, f64::NAN] {
            let ValueView::Num(actual) = scalar(Value::Num(number)).view() else {
                panic!("Number changed kind")
            };
            assert!(number.is_nan() && actual.is_nan() || number.to_bits() == actual.to_bits());
        }
    }

    /// The object encoding contains its exact cell address; a raw execution value reconstructed
    /// from those bits observes the same object and has no destructor.
    #[test]
    fn execution_value_reference_bits_preserve_the_cell_identity() {
        let obj = Object::new(None);
        let packed = Value::Obj(obj);
        assert_eq!(packed.bits(), PACK_OBJ | Gc::as_ptr(&obj) as u64);
        assert!(!std::mem::needs_drop::<Value>());
        // SAFETY: these exact bits came from the live, same-Agent object above.
        let out = unsafe { Value::from_bits(packed.bits()) };
        assert!(matches!(&(out).view(), ValueView::Obj(actual) if Gc::ptr_eq(actual, &obj)));
        drop(out);
    }

    /// Copies preserve every scalar bit pattern and cell identity, with no destructor or
    /// count-based lifetime. The collecting heap determines cell liveness.
    #[test]
    fn copied_execution_values_preserve_all_kinds_and_cell_identities() {
        for value in [
            Value::Undefined,
            Value::Empty,
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::Num(0.0),
            Value::Num(-0.0),
            Value::Num(f64::INFINITY),
            Value::Num(f64::NEG_INFINITY),
            Value::Num(f64::from_bits(0xfffa_0000_0000_1234)),
        ] {
            let packed = value;
            let cloned = packed;
            assert_eq!(packed.bits(), cloned.bits());
            assert_eq!(cloned.is_empty(), cloned.bits() == PACK_EMPTY);
        }
        let mut interp = crate::interpreter::Interp::new();
        let object = Object::new(None);
        let string = crate::lstr::LStr::from("copied value identity");
        let ValueView::Sym(symbol) = (interp.new_symbol(None)).view() else {
            unreachable!()
        };
        let original = [Value::Obj(object), Value::Str(string), Value::Sym(symbol)];
        assert_eq!(
            [original[0].bits(), original[2].bits()],
            [
                PACK_OBJ | Gc::as_ptr(&object) as u64,
                PACK_SYM | symbol_identity(&symbol) as u64
            ]
        );
        let copies = original;
        for (a, b) in original.iter().zip(&copies) {
            assert_eq!(a.bits(), b.bits());
        }
        drop(copies);
        assert!(matches!(
            original[0].clone().view(),
            ValueView::Obj(actual) if Gc::ptr_eq(&actual, &object)
        ));
        assert!(matches!(
            original[1].clone().view(),
            ValueView::Str(actual) if crate::lstr::LStr::ptr_eq(&actual, &string)
                && &*actual.text() == "copied value identity"
        ));
        assert!(matches!(
            original[2].clone().view(),
            ValueView::Sym(actual) if symbol_ptr_eq(&actual, &symbol)
        ));
        let integer = crate::bigint::JsBigInt::parse_dec("123456789012345678901234567890").unwrap();
        let packed = Value::BigInt(integer);
        let copied = packed;
        assert_eq!(packed.bits(), copied.bits());
        drop(packed);
        assert!(matches!(copied.view(), ValueView::BigInt(value) if value == integer));
    }

    #[test]
    fn packed_numeric_constants_canonicalize_every_reserved_tag_collision() {
        // ECMA-262 Number identifies all NaNs as one language value. NaN payloads received
        // through buffers must never be mistaken for owning tagged pointers in execution slots.
        for sign in [0, 1u64 << 63] {
            for high in 0x7ff0u64..=0x7fff {
                let bits = sign | (high << 48) | 0x1234;
                let number = f64::from_bits(bits);
                assert!(number.is_nan());
                let value = Value::Num(number);
                assert_eq!(Value::scalar_bits(&value), Some(PACK_CANON_NAN));
                let packed = value;
                assert_eq!(packed.bits(), PACK_CANON_NAN);
                assert!(matches!(packed.view(), ValueView::Num(value) if value.is_nan()));
            }
        }
        for number in [
            0.0f64,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
        ] {
            assert_eq!(
                Value::scalar_bits(&Value::Num(number)),
                Some(number.to_bits())
            );
        }
        assert_eq!(Value::scalar_bits(&Value::lstr("not copyable")), None);
    }

    /// Shared slot cells support copied reads/replacement and native raw inner-word stores
    /// without fabricating an exclusive slice. The word layout and neighboring slots stay exact.
    #[test]
    fn cell_frame_copies_and_replacements_preserve_bits_and_neighbors() {
        let obj = Object::new(None);
        let frame: [Cell<Value>; 5] = std::array::from_fn(|_| Cell::new(Value::Undefined));
        frame[0].set(Value::Num(1.5));
        frame[1].set(Value::Obj(obj));
        frame[2].set(Value::Bool(true));
        frame[3].set(Value::Null);
        frame[4].set(Value::Num(-0.0));
        let moved = frame[1].replace(Value::Undefined);
        assert_eq!(frame[0].get().bits(), 1.5f64.to_bits());
        assert!(matches!(moved.clone().view(), ValueView::Obj(o) if Gc::ptr_eq(&o, &obj)));
        // SAFETY: this raw loan addresses the inner word of a live Cell. No exclusive Rust
        // slice/reference exists, and this test performs no collection or reentrant call.
        unsafe { frame[3].as_ptr().write(moved) };
        assert!(matches!(frame[3].get().clone().view(), ValueView::Obj(o) if Gc::ptr_eq(&o, &obj)));
        assert!(matches!(
            frame[1].get().clone().view(),
            ValueView::Undefined
        ));
        assert!(matches!(
            frame[2].get().clone().view(),
            ValueView::Bool(true)
        ));
        assert!(
            matches!(frame[4].get().clone().view(), ValueView::Num(n) if n.to_bits() == (-0.0f64).to_bits())
        );
    }

    #[test]
    fn shared_layout_cache_is_bounded_and_does_not_own_the_agent() {
        let heap = new_gc_heap();
        let symbols = new_symbol_agent();
        let weak_heap = Rc::downgrade(&heap);
        let active = enter_agent(&heap, &symbols);
        for n in 0..SHAPE_LAYOUT_CACHE_LIMIT + 32 {
            let mut props = Props::new();
            props.insert(format!("field{n}"), Property::plain(Value::Undefined));
            assert!(props.contains(&format!("field{n}")));
        }
        let shapes = heap.shapes.borrow();
        // Parallel tests may allocate other globally unique IDs between ours. Hints may
        // collide/evict, but both the slot count and allocated pages are always bounded.
        assert!(shapes.layouts.len() <= SHAPE_LAYOUT_CACHE_LIMIT);
        assert_eq!(shapes.layouts.iter().count(), shapes.layouts.len());
        assert!(
            shapes.layouts.allocated_bytes()
                <= SHAPE_LAYOUT_PAGE_COUNT * std::mem::size_of::<LayoutPage>()
        );
        assert!(shapes
            .layouts
            .iter()
            .all(|keys| keys.len() <= SHARED_LAYOUT_MAX_FIELDS));
        drop(shapes);
        let mut a = Props::new();
        let mut b = Props::new();
        for n in 0..40 {
            a.insert(format!("late{n}"), Property::plain(Value::Num(n as f64)));
            b.insert(
                format!("late{n}"),
                Property::plain(Value::Num((n + 1) as f64)),
            );
        }
        assert_eq!(
            a.shape(),
            b.shape(),
            "uncached layouts still retain correct shape identities"
        );
        a.remove("late1");
        assert!(b.contains("late1"));
        assert_eq!(a.keys().len(), 39);
        drop(active);
        drop(heap);
        assert!(
            weak_heap.upgrade().is_none(),
            "layout cache retained its Agent"
        );
    }
}

/// A unique Symbol. Identity is the `id` (every `Symbol()` call gets a fresh one); `description` is
/// the optional label. Well-known symbols (`Symbol.iterator`, …) are just pre-allocated instances.
pub struct SymbolData {
    pub id: u64,
    pub description: Option<Rc<str>>,
    pub(crate) weak_observers: RefCell<Option<Box<crate::weak_metadata::DeathObservers>>>,
    /// The Agent holds this Symbol for its whole life (a well-known or registered Symbol), so
    /// a property key it names needs no owner of its own (`DenseStorage::retain_symbol_key`).
    pub(crate) agent_held: Cell<bool>,
}

impl Drop for SymbolData {
    fn drop(&mut self) {
        // A Symbol can die while SymbolAgent itself is borrowed/destroyed. Its observers are
        // independent of that registry and of whichever same-Agent heap is currently active.
        if let Some(observers) = self.weak_observers.get_mut().take() {
            observers.notify();
        }
    }
}

/// A Symbol value's handle: a cell in the Agent heap that was active when the Symbol was minted
/// (P1 S2). Identity is the handle's address ([`symbol_ptr_eq`]), equivalently the `id`.
pub type SymbolRef = crate::gc_box::GcBox<SymbolData>;

/// A weak Symbol handle: it upgrades until a collection (or the heap's teardown) finds the
/// Symbol dead and clears the shared weak cell.
#[derive(Clone, Default)]
pub(crate) struct WeakSymbol(crate::gc_handle::WeakRef);

impl WeakSymbol {
    #[inline]
    pub(crate) fn upgrade(&self) -> Option<SymbolRef> {
        // SAFETY: the alive flag is cleared before the target's cell is freed.
        self.0
            .live_ptr()
            .map(|ptr| unsafe { crate::gc_box::GcBox::from_raw(ptr as *const SymbolData) })
    }

    /// 1 while the Symbol is alive, 0 after it died.
    #[inline]
    pub(crate) fn strong_count(&self) -> usize {
        usize::from(self.0.live_ptr().is_some())
    }
}

/// Mint the handle of a new Symbol (in the active Agent's heap).
pub(crate) fn new_symbol_ref(data: SymbolData) -> SymbolRef {
    crate::gc_box::new(data)
}

#[inline]
pub(crate) fn symbol_ptr_eq(a: &SymbolRef, b: &SymbolRef) -> bool {
    crate::gc_box::GcBox::ptr_eq(a, b)
}

/// The Symbol's allocation address, for identity-keyed diagnostics.
#[inline]
pub(crate) fn symbol_identity(symbol: &SymbolRef) -> usize {
    let identity = crate::gc_box::GcBox::as_ptr(symbol);
    identity as usize
}

#[inline]
pub(crate) fn symbol_downgrade(symbol: &SymbolRef) -> WeakSymbol {
    WeakSymbol(crate::gc_handle::WeakRef::to(symbol_identity(symbol)))
}

impl crate::gc_box::Payload for SymbolData {
    fn desc(heap: &GcState) -> crate::gc_core::DescId {
        heap.symbol_desc
    }
}

/// The descriptor of Symbol cells: no edges; the finalizer delivers death notifications.
fn symbol_cell_layout() -> crate::gc_core::LayoutDesc {
    crate::gc_core::LayoutDesc {
        name: "symbol",
        trace: crate::gc_box::trace_none,
        mark_trace: None,
        finalize: Some(crate::gc_box::finalize::<SymbolData>),
        carded: false,
    }
}

/// An ECMAScript Property Key produced by `ToPropertyKey`. TRust JS's object maps use a compact
/// encoded string for Symbol keys, but the key itself must keep the Symbol identity alive until
/// an object property map can take ownership of it. This mirrors the specification's actual
/// String-or-Symbol result instead of temporarily reducing a Symbol to an unowned integer.
#[derive(Clone)]
pub(crate) struct PropertyKey {
    text: String,
    symbol: Option<SymbolRef>,
    private: bool,
}

impl PropertyKey {
    pub(crate) fn string(text: String) -> Self {
        let text = encode_owned_string_key(text);
        Self {
            text,
            symbol: None,
            private: false,
        }
    }

    /// A runtime Private Name, supplied only by class evaluation, never by ToPropertyKey.
    pub(crate) fn private(text: String) -> Self {
        Self {
            text,
            symbol: None,
            private: true,
        }
    }

    pub(crate) fn is_private(&self) -> bool {
        self.private
    }

    /// A key already encoded in an ordinary property map or compiled property-name table.
    pub(crate) fn stored(text: String) -> Self {
        let symbol = encoded_symbol_id(&text).and_then(active_symbol);
        Self {
            text,
            symbol,
            private: false,
        }
    }

    pub(crate) fn symbol(symbol: SymbolRef) -> Self {
        Self {
            text: Interp::sym_key(&symbol),
            symbol: Some(symbol),
            private: false,
        }
    }

    pub(crate) fn into_string(self) -> String {
        self.text
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.text
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        if let Some(symbol) = &self.symbol {
            visitor.symbol(symbol);
        }
        self.text.capacity()
    }

    pub(crate) fn into_value(self) -> Value {
        match self.symbol {
            Some(symbol) => Value::Sym(symbol),
            None => string_key_value(self.text),
        }
    }
}

impl std::ops::Deref for PropertyKey {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl std::fmt::Display for PropertyKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.text.fmt(f)
    }
}

impl PartialEq<str> for PropertyKey {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for PropertyKey {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

/// Ordinary property maps encode leading-NUL String keys with one extra NUL. A single
/// NUL followed by a decimal identity is reserved for Symbol keys. This is injective for
/// every ECMAScript String, including strings containing arbitrary numbers of NULs.
#[inline]
pub(crate) fn encode_string_key(text: &str) -> std::borrow::Cow<'_, str> {
    if text.starts_with('\0') {
        let mut encoded = String::with_capacity(text.len() + 1);
        encoded.push('\0');
        encoded.push_str(text);
        std::borrow::Cow::Owned(encoded)
    } else {
        std::borrow::Cow::Borrowed(text)
    }
}

#[inline]
pub(crate) fn encode_owned_string_key(mut text: String) -> String {
    if text.starts_with('\0') {
        text.insert(0, '\0');
    }
    text
}

/// [`encode_string_key`] of an engine string's text.
#[inline]
pub(crate) fn encode_lstr_key(s: &crate::lstr::LStr) -> std::borrow::Cow<'_, str> {
    match s.text() {
        std::borrow::Cow::Borrowed(text) => encode_string_key(text),
        std::borrow::Cow::Owned(text) => std::borrow::Cow::Owned(encode_owned_string_key(text)),
    }
}

#[inline]
pub(crate) fn decode_string_key(key: &str) -> &str {
    if key.starts_with("\0\0") {
        &key[1..]
    } else {
        key
    }
}

pub(crate) fn string_key_value(key: impl AsRef<str>) -> Value {
    Value::lstr(decode_string_key(key.as_ref()))
}

/// Symbol state owned by one ECMAScript Agent. ECMA-262 §9.9.2 and §20.4 put the global symbol
/// registry on the surrounding Agent, shared by all of its realms but not by unrelated Agents.
/// `symbols` is only an identity lookup for encoded property keys and therefore holds ordinary
/// Symbols weakly; global-registry and well-known Symbols have their normative strong owners.
pub(crate) struct SymbolAgentState {
    /// Process-local diagnostic identity. Shared by every realm implementation in this Agent.
    pub(crate) agent_id: u64,
    pub(crate) next_id: u64,
    pub(crate) symbols: crate::fasthash::FastMap<u64, WeakSymbol>,
    pub(crate) global_by_key: crate::fasthash::FastMap<Rc<str>, SymbolRef>,
    pub(crate) global_key_by_id: crate::fasthash::FastMap<u64, Rc<str>>,
    pub(crate) well_known: crate::fasthash::FastMap<&'static str, SymbolRef>,
    /// Post-collection diagnostics keyed by the independently collected heap implementation.
    /// ShadowRealms share this Agent state but currently own distinct collector registries.
    pub(crate) memory_snapshots: crate::fasthash::FastMap<u64, crate::memory::Snapshot>,
}

pub(crate) type SymbolAgent = Rc<RefCell<SymbolAgentState>>;

pub(crate) fn new_symbol_agent() -> SymbolAgent {
    static NEXT_AGENT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    Rc::new(RefCell::new(SymbolAgentState {
        agent_id: NEXT_AGENT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        next_id: 0,
        symbols: Default::default(),
        global_by_key: Default::default(),
        global_key_by_id: Default::default(),
        well_known: Default::default(),
        memory_snapshots: Default::default(),
    }))
}

/// Byte offsets the JIT's inline property-cache templates read directly out of the object graph.
/// Every field is *measured at runtime* against the real types of the build (never hardcoded);
/// the layouts that aren't guaranteed — where a `Vec`'s or an `InlineVec`'s words sit, a
/// handle's header size, the `Option<Gc>` niche — are located by probing and checked against
/// live values, and reported in `valid`. If `valid` is false the JIT emits no inline caches and
/// everything routes through the checked helper, so a future libstd layout change degrades
/// performance, never correctness.
///
/// Object offsets are relative to the *stored* handle word — the value in a `Value::Obj` payload,
/// a packed object word's payload and an `Option<Gc>` (proto) field: the cell's address, where
/// the `RefCell<Object>` starts. It is the handle's [`Gc::as_ptr`] identity itself (checked
/// live), so generated code compares it with recorded identities as it is. The `Object` is
/// `obj_from_cell` bytes on, past the `RefCell`'s borrow flag. The inline templates only ever
/// have the stored word, so measuring from it is what makes them correct.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct JitLayout {
    /// Stored handle word → the `Object`, past the `RefCell` wrapper, whose borrow flag is the
    /// first word of an object cell.
    pub obj_from_cell: usize,
    /// The build's representation facts beyond these offsets ([`JitRepr`]): where a key layout's
    /// named fields and binding-map entries hold their elements, a slot
    /// memo's width and the Collection tag, probed once per process.
    pub(crate) repr: &'static JitRepr,
    pub obj_proto: usize,
    pub obj_props: usize,
    /// The flag byte within `Object` (in its map, W2d step 1): [`FLAG_IC_PLAIN`],
    /// [`FLAG_EXTENSIBLE`], [`FLAG_ELEM_MODE`], [`FLAG_PROTO`], [`FLAG_HAS_FAR`],
    /// [`FLAG_CONSTRUCTOR`] and the class kind's two bits.
    pub obj_flags: usize,
    pub props_shape: usize,
    /// The contiguous instance-field storage within `Props` (an `InlineVec` holding the first
    /// fields in the cell). Generated code reaches its elements, length and capacity only as
    /// `JitRepr::props_fields` describes.
    pub props_entries: usize,
    /// Nullable stored layout cell address; live keys are its fields-length prefix.
    pub props_layout: usize,
    /// Stored layout cell address -> its key count (a `u32`) and the address of its keys, one
    /// [`crate::atom::PropKey`] word each, in a storage block while there are any.
    pub layout_keys_len: usize,
    pub layout_keys_ptr: usize,
    pub shape_layout_entry_size: usize,
    pub shape_layout_entry_id: usize,
    pub shape_layout_entry_keys: usize,
    /// The map's own key index within `Props`: a nullable pointer, non-null while the map keeps
    /// an index of its own (`Props::build_index`).
    pub props_elems: usize,
    /// The element words' pointer within `Props` (W2c, [`word_block::Words`]): the address of
    /// the first word, never null (a map without words points past the static empty block's
    /// header, whose used length is zero). A word is a plain element's value, a hole the
    /// [`PACK_EMPTY`] word.
    pub props_words: usize,
    /// The word header's fields, as bytes before the first word: the used length, the
    /// capacity, the front room and the kind bits ([`word_block::KIND_NO_HOLES`] and the
    /// rest), each a u32.
    pub words_used: usize,
    pub words_capacity: usize,
    pub words_shifted: usize,
    pub words_flags: usize,
    /// `size_of::<Property>()` — the keyless instance-field stride.
    pub entry_size: usize,
    /// Packed value within a keyless instance field.
    pub entry_value: usize,
    /// Descriptor flags byte within an entry (used to test `PROP_ACCESSOR`).
    pub entry_accessor: usize,
    /// Descriptor flags byte within an entry (used to test `PROP_WRITABLE`).
    pub entry_writable: usize,
    /// Standalone `Property` layout used by keyless packed elements.
    pub property_size: usize,
    pub property_value: usize,
    pub property_meta: usize,
    /// The element words' pointer and header matched a live probe.
    pub words_valid: bool,
    /// `ObjectKind::Array`'s discriminant byte (the element templates also accept arrays). The
    /// kinds an ordinary-object guard admits are one tag range ([`KIND_ORDINARY_LAST`]).
    pub exotic_array_tag: u8,
    /// `ObjectKind::StrWrap`'s discriminant byte (String.prototype IS a StrWrap — the GetMethod
    /// template accepts it as a named-read holder; index/length reads never take that path).
    pub exotic_strwrap_tag: u8,
    /// `ObjectKind::NumWrap`'s discriminant byte (Number.prototype and Number wrappers: named
    /// entries are shape-pinned like StrWrap's, and no name is exotic-provided).
    pub exotic_numwrap_tag: u8,
    /// `ObjectKind::BoolWrap`'s discriminant byte (Boolean.prototype and Boolean wrappers).
    pub exotic_boolwrap_tag: u8,
    /// Fail-closed probe for ordinary numeric typed-array backing storage.
    pub native_typed_array: Option<crate::native_typed_array::Layout>,
    /// Scope offsets are relative to [`Env::as_ptr`] (the `RefCell<Scope>`), what generated code
    /// keeps for an environment and what a stored environment handle word (a parent field, a
    /// closure's `env`) holds (checked live). `Env::as_ptr(env)` → the scope's `VarMap`
    /// generation counter (through the `RefCell`).
    pub scope_gen: usize,
    /// The record's names word (64 bits): a shared descriptor's address is its layout identity
    /// (`VarMap::layout_id`), which caches compare whole.
    pub scope_layout: usize,
    /// Parent `Option<Env>` within `Env::as_ptr(env)`: a stored environment handle word, or
    /// null (the handle's niche; checked live).
    pub scope_parent: usize,
    pub scope_parent_valid: bool,
    /// A record's binding slots, relative to `Env::as_ptr(env)`: one word each, held as
    /// `JitRepr::scope_small` describes (in place after the record, or in a storage block).
    pub scope_small_vec: usize,
    pub scope_binding_stride: usize,
    pub scope_binding_offset: usize,
    pub scope_small_valid: bool,
    /// Packed object-environment word; a complete EMPTY word means absent, checked live.
    pub scope_with: usize,
    pub scope_with_none: u64,
    pub scope_with_valid: bool,
    /// The execution word within a `BindingSlot` (the LoadName template's eight-byte source): the
    /// slot itself, `PACK_EMPTY` while the binding is in its TDZ. A binding's attributes live with
    /// its record's names: caches check them when they fill.
    pub binding_value: usize,
    /// The length word within an `Rc<str>` fat pointer (0 or 8 — layout is unstable).
    pub str_len_word: usize,
    /// The pointer word within an `Rc<str>` fat pointer (the other one).
    pub str_ptr_word: usize,
    /// Stored `Rc<str>` pointer word → the first byte of the string data (the RcBox header).
    pub str_data_off: usize,
    /// Whether the four fields above probed successfully (key-checked array-holder entries can
    /// inline their key compare only when they did).
    pub key_probe_ok: bool,
    /// `kind` within `Object`: [`ObjectKind`]'s discriminant byte at +0, its payload word at +8.
    pub obj_kind: usize,
    /// `ObjectKind::User`'s payload word (a [`UserFn`]: the closure cell's address) → the
    /// [`UserCallable`]'s `func`, `env` (a stored environment handle word) and `realm` fields.
    pub user_func: usize,
    pub user_env: usize,
    pub user_realm: usize,
    /// Stored `Rc<Function>` pointer → `Rc::as_ptr` (the AST identity a `CallIc` records; the
    /// AST is reference-counted).
    pub func_data_off: usize,
    /// `Env::as_ptr(env)` → `Scope::under_with` (through the `RefCell`).
    pub scope_under_with: usize,
    /// The six fields above matched a live closure's representation, so the JIT's code-keyed
    /// call probe may read a callee's [`UserCallable`] directly.
    pub call_probe_valid: bool,
    pub valid: bool,
}

/// How generated code finds a growable sequence's elements, length and capacity, at offsets
/// from the sequence field: a `Vec`, or an `InlineVec` (`crate::inline_vec`) such as `Props`'
/// named values and a binding map's entries. Measured and checked against live sequences by
/// [`jit_repr`]; the ARM64 storage emitters (`jit_storage.rs`) read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JitVecLayout {
    /// A `Vec`: the 8-byte data pointer, length and capacity words. The elements are always
    /// behind the data pointer.
    Vec { ptr: usize, len: usize, cap: usize },
    /// An `InlineVec<T, N>`: 4-byte (`u32`) length and capacity words. Exactly while the
    /// capacity word is 0 the elements are in place, from `inline`, with room for
    /// `inline_capacity` (N) of them; otherwise the 8-byte word at `heap` points at a heap buffer
    /// holding the capacity word's count (the pointer takes the place of the in-place elements,
    /// so the two may overlap).
    Inline {
        len: usize,
        cap: usize,
        inline: usize,
        heap: usize,
        inline_capacity: usize,
    },
}

/// The build's representation facts that generated code depends on beyond [`JitLayout`]'s
/// offsets. They are facts of the build's types, so [`jit_repr`] probes them on live values once
/// per process, and every `JitLayout` refers to them. Kept out of line, they leave the size of a
/// `JitLayout` unchanged: each `Interp` caches one inline, and generated code reads the
/// interpreter's fields at probed offsets.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct JitRepr {
    /// How the instance-field storage at `JitLayout::props_entries` holds its `Property` values.
    pub(crate) props_fields: JitVecLayout,
    /// Where a small (or indexed) binding map keeps its ordered entries, and how: an `InlineVec`
    /// keeping the first binding in the environment cell. The source of `JitLayout`'s
    /// `scope_small_*` fields; `None` when the probe disagreed.
    pub(crate) scope_small: Option<crate::interpreter::JitBindingStorage>,
    /// The named-field probe held: a `JitLayout` is valid only if it did.
    pub(crate) valid: bool,
}

/// The build's [`JitRepr`], probed on live values on first use.
pub(crate) fn jit_repr() -> &'static JitRepr {
    static REPR: std::sync::OnceLock<JitRepr> = std::sync::OnceLock::new();
    REPR.get_or_init(probe_jit_repr)
}

fn probe_jit_repr() -> JitRepr {
    // Named fields: an InlineVec keeping the first INLINE_NAMED_SLOTS in the cell, probed in
    // place and spilled.
    let props_fields = Fields::probe_jit_layout(|| Property::plain(Value::Num(0.0)));

    JitRepr {
        props_fields: props_fields.unwrap_or(JitVecLayout::Vec {
            ptr: 0,
            len: 0,
            cap: 0,
        }),
        scope_small: Some(crate::interpreter::VarMap::jit_slots_layout()),
        valid: props_fields.is_some(),
    }
}

/// Measure [`JitLayout`] against the live types of the build, probing the non-guaranteed
/// layouts and checking them on live values.
pub(crate) fn jit_layout(sample: &Gc) -> JitLayout {
    use std::mem::offset_of;
    // Rc<str> fat-pointer probe (word order and RcBox data offset are not std-guaranteed): a
    // known 8-byte string tells us which word holds the length; the data pointer is the other,
    // and `as_ptr` minus the stored word gives the RcBox header size. Fails closed.
    let (str_len_word, str_ptr_word, str_data_off, key_probe_ok) = {
        let probe: Rc<str> = "probe_8B".into();
        let words: [usize; 2] = unsafe { std::mem::transmute_copy::<Rc<str>, [usize; 2]>(&probe) };
        let data = probe.as_ptr() as usize;
        if words[0] == 8 && words[1] != 8 && data > words[1] && data - words[1] < 256 {
            (0usize, 8usize, data - words[1], true)
        } else if words[1] == 8 && words[0] != 8 && data > words[0] && data - words[0] < 256 {
            (8usize, 0usize, data - words[0], true)
        } else {
            (0, 0, 0, false)
        }
    };
    // The stored handle word — a `Value::Obj` payload, a packed object word's payload, an
    // `Option<Gc>` (proto) field — is the cell address, `Gc::as_ptr` itself (the cell starts
    // with the RefCell<Object>): generated code compares it with recorded `Gc::as_ptr`
    // identities as it is, and `object_identity_ok` checks that. Every object offset is
    // measured from it, through the RefCell wrapper.
    let stored = unsafe { *(sample as *const Gc as *const usize) };
    let as_ptr = Gc::as_ptr(sample) as usize; // the RefCell<Object>
    let obj_addr = &*sample.borrow() as *const Object as usize;
    let object_identity_ok = stored == as_ptr;
    let obj_from_cell = obj_addr.wrapping_sub(stored);
    // An Option<Gc> (a proto field) holds exactly the stored word, and None is null (the
    // handle's non-null niche).
    let some_proto: Option<Gc> = Some(*sample);
    let none_proto: Option<Gc> = None;
    let niche_ok = std::mem::size_of::<Option<Gc>>() == std::mem::size_of::<usize>()
        && unsafe { *(&some_proto as *const Option<Gc> as *const usize) } == stored
        && unsafe { *(&none_proto as *const Option<Gc> as *const usize) } == 0
        && as_ptr <= obj_addr;
    let repr = jit_repr();

    // Generated creation code follows the stored cell address to PropertyLayoutData and then to
    // its key vector. Measure this path instead of relying on the Rust struct layout, and check
    // it on a live layout.
    let key_layout = Some(new_property_layout([PropKey::index(7), PropKey::index(9)]));
    let keys: &LayoutKeys = &key_layout.as_ref().unwrap().keys;
    let layout_word = unsafe { *(&key_layout as *const Option<PropertyLayout> as *const usize) };
    let keys_off = (keys as *const LayoutKeys as usize).wrapping_sub(layout_word);
    let (layout_keys_len, layout_keys_ptr, keys_ok) =
        match LayoutKeys::probe_jit_layout(|| PropKey::index(7)) {
            Some(JitVecLayout::Inline { len, heap, .. }) => {
                let (len, ptr) = (keys_off + len, keys_off + heap);
                // SAFETY: measured offsets of fields within the live layout cell.
                let (live_len, live_ptr) = unsafe {
                    (
                        ((layout_word + len) as *const u32).read(),
                        ((layout_word + ptr) as *const usize).read(),
                    )
                };
                let ok = live_len == 2 && live_ptr == <[PropKey]>::as_ptr(keys) as usize;
                (len, ptr, ok)
            }
            _ => (0, 0, false),
        };
    let empty_layout: Option<PropertyLayout> = None;
    let layout_ok = keys_off < 256
        && std::mem::size_of::<PropKey>() == 8
        && std::mem::size_of::<Option<PropertyLayout>>() == std::mem::size_of::<usize>()
        && unsafe { *(&empty_layout as *const Option<PropertyLayout> as *const usize) } == 0;
    let page = Some(Box::new(
        std::array::from_fn::<_, SHAPE_LAYOUT_PAGE_SIZE, _>(|_| LayoutEntry::default()),
    ));
    let page_word = unsafe { *(&page as *const Option<Box<LayoutPage>> as *const usize) };
    let no_page: Option<Box<LayoutPage>> = None;
    let shape_layout_entry_size = std::mem::size_of::<LayoutEntry>();
    let shape_layout_entry_id = offset_of!(LayoutEntry, shape);
    let shape_layout_entry_keys = offset_of!(LayoutEntry, keys);
    let shape_directory_ok = std::mem::size_of::<Option<Box<LayoutPage>>>()
        == std::mem::size_of::<usize>()
        && page_word == page.as_deref().unwrap().as_ptr() as usize
        && unsafe { *(&no_page as *const Option<Box<LayoutPage>> as *const usize) } == 0
        && shape_layout_entry_size == 16
        && shape_layout_entry_id == 0
        && shape_layout_entry_keys == 8;
    // DenseStorage's first word is deliberately a nullable pointer to the map's own key index
    // (the creation template reads it as "this map keeps a key index"). Verify the niche rather
    // than relying on it silently if a future compiler changes the representation.
    let thin_some = DenseStorage(
        Some(crate::gc_storage::StorageBox::new(KeyIndex::of(&[]))),
        None,
        word_block::Words::new(),
    );
    let thin_word = unsafe { *(&thin_some as *const DenseStorage as *const usize) };
    let thin_expected = thin_some.0.as_deref().unwrap() as *const KeyIndex as usize;
    let thin_none = DenseStorage(None, None, word_block::Words::new());
    let thin_none_word = unsafe { *(&thin_none as *const DenseStorage as *const usize) };
    let thin_vec_ok = thin_word == thin_expected && thin_none_word == 0;
    // The element words: the pointer at `props_words` is the first word's address, with the
    // header just before it.
    let header_bytes = std::mem::size_of::<word_block::WordHeader>();
    let words_at = |field: usize| header_bytes - field;
    let props_words = offset_of!(Props, elems) + offset_of!(DenseStorage, 2);
    let (words_used, words_capacity, words_shifted, words_flags) = (
        words_at(offset_of!(word_block::WordHeader, used)),
        words_at(offset_of!(word_block::WordHeader, capacity)),
        words_at(offset_of!(word_block::WordHeader, shifted)),
        words_at(offset_of!(word_block::WordHeader, flags)),
    );
    let words_probe = Props::packed_array([Value::Num(1.0), Value::Null].into_iter());
    let words_valid = {
        let words = words_probe.elems.words().expect("a probe with elements");
        let first = words.as_slice().as_ptr();
        // SAFETY: reads of the probe's own words pointer and of its block's header fields.
        unsafe {
            let base = (&words_probe as *const Props).cast::<u8>();
            let header = |back: usize| first.cast::<u8>().sub(back).cast::<u32>().read();
            std::mem::size_of::<word_block::Words>() == 8
                && base.add(props_words).cast::<*const u64>().read() == first
                && header(words_used) as usize == words.len()
                && header(words_capacity) as usize == words.capacity()
                && header(words_shifted) as usize == words.front_room()
                && header(words_flags) == words.kind()
        }
    };

    // The kinds' discriminants are fixed (`ObjectKind` is repr(u8)): the tag byte at offset 0,
    // whatever the payload. Check the ones generated code compares on live values.
    let kind_tag = |kind: &ObjectKind| kind.tag();
    let exotic_array_tag = kind_tag(&ObjectKind::Array);
    let exotic_strwrap_tag = kind_tag(&ObjectKind::str_wrap("".into()));
    let exotic_numwrap_tag = kind_tag(&ObjectKind::NumWrap(0.0));
    let exotic_boolwrap_tag = kind_tag(&ObjectKind::BoolWrap(false));
    let kinds_ok = exotic_array_tag == KIND_ARRAY_TAG
        && exotic_strwrap_tag == 19
        && exotic_numwrap_tag == 18
        && exotic_boolwrap_tag == 17
        && kind_tag(&ObjectKind::NumWrap(-1.25)) == exotic_numwrap_tag
        && kind_tag(&ObjectKind::BoolWrap(true)) == exotic_boolwrap_tag
        && kind_tag(&ObjectKind::Plain) == 0
        && {
            use crate::ordered_collection::{CollectionKind, OrderedCollection};
            let map = ObjectKind::Collection(crate::gc_storage::PayloadBox::new(
                OrderedCollection::new(CollectionKind::Map),
            ));
            kind_tag(&map) == KIND_ORDINARY_LAST
        };

    // Scope offsets for the inline LoadName template, relative to `Env::as_ptr` (the
    // RefCell<Scope>) → Scope.vars → VarMap generation. The RefCell value offset is probed on a
    // live scope; a stored environment handle word is the cell address, `Env::as_ptr` itself,
    // which generated code uses as it is (`scope_identity_ok` checks that).
    let probe_env = crate::interpreter::new_scope(None);
    let scope_addr = {
        let b = probe_env.borrow();
        &*b as *const crate::interpreter::Scope as usize
    };
    let scope_refcell = scope_addr - Env::as_ptr(&probe_env) as usize;
    let scope_stored = unsafe { *(&probe_env as *const Env as *const usize) };
    let scope_identity_ok = scope_stored == Env::as_ptr(&probe_env) as usize;
    let parent_some = Some(probe_env);
    let parent_none: Option<Env> = None;
    let scope_parent = scope_refcell + offset_of!(crate::interpreter::Scope, parent);
    // A live child's parent field holds the parent's stored word, at the measured offset.
    let probe_child = crate::interpreter::new_scope(Some(probe_env));
    let child_parent =
        unsafe { *((Env::as_ptr(&probe_child) as usize + scope_parent) as *const usize) };
    let scope_parent_valid = std::mem::size_of::<Option<Env>>() == std::mem::size_of::<usize>()
        && scope_identity_ok
        && unsafe { *(&parent_some as *const Option<Env> as *const usize) } == scope_stored
        && unsafe { *(&parent_none as *const Option<Env> as *const usize) } == 0
        && child_parent == scope_stored;
    let scope_gen = scope_refcell
        + offset_of!(crate::interpreter::Scope, vars)
        + crate::interpreter::VarMap::generation_offset();
    // The names word: a shared descriptor's address is the record's layout identity.
    let scope_layout = scope_refcell
        + offset_of!(crate::interpreter::Scope, vars)
        + crate::interpreter::VarMap::names_offset();
    let vars_offset = scope_refcell + offset_of!(crate::interpreter::Scope, vars);
    // A binding map's values in slot order, probed per build.
    let scope_small_valid = repr.scope_small.is_some();
    let small = repr
        .scope_small
        .unwrap_or(crate::interpreter::JitBindingStorage {
            entries: 0,
            storage: JitVecLayout::Vec {
                ptr: 0,
                len: 0,
                cap: 0,
            },
            stride: 0,
            binding: 0,
        });
    let scope_small_vec = vars_offset + small.entries;
    let scope_with = scope_refcell + crate::interpreter::Scope::with_object_offset();
    let probe_with = crate::interpreter::new_with_scope(probe_env, Value::Null);
    let scope_with_none = PACK_EMPTY;
    let scope_with_valid = std::mem::size_of::<Value>() == 8
        && std::mem::align_of::<Value>() == 8
        && scope_with.is_multiple_of(8)
        && scope_with + 8 <= std::mem::size_of::<RefCell<crate::interpreter::Scope>>()
        && unsafe { *((Env::as_ptr(&probe_child) as usize + scope_with) as *const u64) }
            == PACK_EMPTY
        && unsafe { *((Env::as_ptr(&probe_with) as usize + scope_with) as *const u64) }
            == PACK_NULL;
    let binding_value = 0;
    const {
        assert!(std::mem::size_of::<crate::interpreter::BindingSlot>() == 8);
    }

    // Code-keyed call probe: `Callable` is repr(u8) (discriminant byte, then one payload word)
    // and `UserCallable` is repr(C). Verify both on live values: a native callable, and a
    // closure of a parsed function, whose payload word (its `UserFn`: the closure cell's
    // address) must lead through the measured offsets to the closure's func, env and realm,
    // and whose func word plus `func_data_off` must be the AST identity a CallIc records.
    fn call_layout_probe(_: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
        Ok(Value::Undefined)
    }
    let native: NativeFn = call_layout_probe;
    let native_callable = ObjectKind::Native(native);
    // The kind's discriminant is one initialized byte; the rest of its first word is padding.
    // Read only that byte and its initialized pointer payload, never the padding as a usize.
    let callable_words = |call: &ObjectKind| {
        let bytes = std::ptr::from_ref(call).cast::<u8>();
        unsafe {
            (
                bytes.read(),
                bytes
                    .add(std::mem::size_of::<usize>())
                    .cast::<usize>()
                    .read(),
            )
        }
    };
    let native_words = callable_words(&native_callable);
    let callable_ok = std::mem::size_of::<ObjectKind>() == 16
        && native_words.0 == KIND_CALLABLE_FIRST
        && native_words.1 == native as usize
        && std::mem::align_of::<UserCallable>() == 8
        && std::mem::align_of::<Function>() <= 8;
    #[cfg(feature = "embed")]
    let callable_ok = {
        fn scoped_probe<'s>(
            _: &mut crate::scope::Ctx<'s>,
            _: crate::scope::Value<'s>,
            _: &[crate::scope::Value<'s>],
        ) -> Result<crate::scope::Value<'s>, crate::scope::Value<'s>> {
            Ok(crate::scope::Value::Undefined)
        }
        let function: crate::scope::NativeFn = scoped_probe;
        let words = callable_words(&ObjectKind::ScopedNative(function));
        callable_ok && words.0 == KIND_EXOTIC_NONE_LAST && words.1 == function as usize
    };
    let parsed = crate::parser::parse_script("function probe(){}", false).ok();
    let probe_func = match parsed.as_deref() {
        Some([crate::ast::Stmt::FuncDecl(func)]) => Some(func.clone()),
        _ => None,
    };
    let (user_func, user_env, user_realm, func_data_off, closure_ok) = match probe_func {
        Some(func) => {
            let realm = obj_addr; // any identity the probe can recognize
            let closure = ObjectKind::user(
                func.clone(),
                probe_env,
                realm,
                None,
                crate::agent_access::RealmId::ROOT,
            );
            let words = callable_words(&closure);
            let ObjectKind::User(user) = &closure else {
                unreachable!("ObjectKind::user makes a user callable")
            };
            let payload = words.1;
            let user_data_off = (user_fn_ptr(user) as usize).wrapping_sub(payload);
            let func_word = unsafe { std::mem::transmute_copy::<Rc<Function>, usize>(&func) };
            let func_data_off = (Rc::as_ptr(&func) as usize).wrapping_sub(func_word);
            let user_func = user_data_off + offset_of!(UserCallable, func);
            let user_env = user_data_off + offset_of!(UserCallable, env);
            let user_realm = user_data_off + offset_of!(UserCallable, realm);
            let read = |off: usize| unsafe { *((payload + off) as *const usize) };
            let ok = words.0 == CALLABLE_USER_TAG
                && user_data_off < 256
                && func_data_off < 256
                && read(user_func) == func_word
                && read(user_func).wrapping_add(func_data_off) == Rc::as_ptr(&user.func) as usize
                && read(user_env) == Env::as_ptr(&user.env) as usize
                && read(user_realm) == realm;
            (user_func, user_env, user_realm, func_data_off, ok)
        }
        None => (0, 0, 0, 0, false),
    };
    let scope_under_with = scope_refcell
        + offset_of!(crate::interpreter::Scope, vars)
        + crate::interpreter::VarMap::under_with_offset();
    let call_probe_valid = callable_ok && closure_ok;

    let valid = repr.valid
        && kinds_ok
        && object_identity_ok
        && scope_identity_ok
        && niche_ok
        && thin_vec_ok
        && keys_ok
        && layout_ok
        && shape_directory_ok;
    JitLayout {
        obj_from_cell,
        repr,
        obj_proto: offset_of!(Object, proto),
        // The view is the kind's payload word, read once the tag says TypedArray.
        native_typed_array: crate::native_typed_array::probe_layout(
            offset_of!(Object, kind) + std::mem::size_of::<usize>(),
        ),
        obj_props: offset_of!(Object, props),
        obj_flags: offset_of!(Object, props) + offset_of!(Props, flags),
        props_shape: offset_of!(Props, shape),
        props_entries: offset_of!(Props, entries) + offset_of!(NamedEntries, fields),
        props_layout: offset_of!(Props, entries) + offset_of!(NamedEntries, layout),
        layout_keys_len,
        layout_keys_ptr,
        shape_layout_entry_size,
        shape_layout_entry_id,
        shape_layout_entry_keys,
        props_elems: offset_of!(Props, elems),
        props_words,
        words_used,
        words_capacity,
        words_shifted,
        words_flags,
        entry_size: std::mem::size_of::<Property>(),
        str_len_word,
        str_ptr_word,
        str_data_off,
        key_probe_ok,
        entry_value: offset_of!(Property, packed),
        entry_accessor: offset_of!(Property, meta),
        entry_writable: offset_of!(Property, meta),
        property_size: std::mem::size_of::<Property>(),
        property_value: offset_of!(Property, packed),
        property_meta: offset_of!(Property, meta),
        words_valid,
        exotic_array_tag,
        exotic_strwrap_tag,
        exotic_numwrap_tag,
        exotic_boolwrap_tag,
        scope_gen,
        scope_layout,
        scope_parent,
        scope_parent_valid,
        scope_small_vec,
        scope_binding_stride: small.stride,
        scope_binding_offset: small.binding,
        scope_small_valid,
        scope_with,
        scope_with_none,
        scope_with_valid,
        binding_value,
        obj_kind: offset_of!(Object, kind),
        user_func,
        user_env,
        user_realm,
        func_data_off,
        scope_under_with,
        call_probe_valid,
        valid,
    }
}

#[cfg(all(test, target_pointer_width = "64"))]
#[path = "jit_layout_tests.rs"]
mod jit_layout_tests;

impl Value {
    /// The JS string whose code points are the characters of the Rust text `s`.
    ///
    /// The engine stores a lone surrogate as a scalar in U+10F800..=U+10FFFF (see
    /// `Value::Str`), so a character of `s` in that range is stored as its surrogate pair: a
    /// native function returning `Value::str("\u{10FFFD}")` gives a string of length 2. A Rust
    /// string cannot hold a lone surrogate; build one from code units with
    /// [`Value::from_utf16`].
    pub fn str(s: impl AsRef<str>) -> Value {
        Value::Str(crate::jstr::from_text(s.as_ref()).as_ref().into())
    }
    /// [`Value::str`] for an owned `String`, reusing its buffer when no character needs the
    /// surrogate-pair form.
    pub fn from_string(s: String) -> Value {
        match crate::jstr::from_text(&s) {
            std::borrow::Cow::Borrowed(_) => Value::Str(s.into()),
            std::borrow::Cow::Owned(converted) => Value::Str(converted.into()),
        }
    }
    /// The JS string with exactly these UTF-16 code units, lone surrogates included.
    pub fn from_utf16(units: &[u16]) -> Value {
        Value::Str(crate::jstr::from_units(units).into())
    }
    /// A string already in the engine representation (see `Value::Str`): engine strings and
    /// their slices, property keys, and ASCII literals.
    pub(crate) fn lstr(s: impl Into<crate::lstr::LStr>) -> Value {
        Value::Str(s.into())
    }
    /// The JS string whose engine text is `s` (see `Value::Str`), with no conversion: the
    /// lossless path for a host that keeps JS strings as Rust strings, such as a DOM storing
    /// Web IDL `DOMString`s. `s` must come from [`Value::as_engine_text`],
    /// `Ctx::coerce_engine_text`, or Rust text converted by `embed::text_to_engine`; Rust text
    /// passed here directly loses characters in U+10F800..=U+10FFFF to lone surrogates.
    pub fn from_engine_text(s: impl Into<String>) -> Value {
        Value::Str(s.into().into())
    }
    /// A string's engine text: borrowed from a one-byte string's immutable cell for this
    /// original receiver's lifetime, converted from a two-byte one.
    pub(crate) fn as_engine_text(&self) -> Option<std::borrow::Cow<'_, str>> {
        let ValueView::Str(string) = self.view() else {
            return None;
        };
        Some(match string.as_ascii() {
            Some(text) => {
                let text = text as *const str;
                // SAFETY: the view copied the original receiver's cell handle. The bytes belong
                // to that cell, not the temporary handle, and already exposed bytes never change.
                // The original receiver remains the root/lifetime owner of the returned borrow.
                std::borrow::Cow::Borrowed(unsafe { &*text })
            }
            None => std::borrow::Cow::Owned(string.text().into_owned()),
        })
    }

    /// A string as USVString text, replacing lone surrogates with U+FFFD.
    pub(crate) fn as_text(&self) -> Option<std::borrow::Cow<'_, str>> {
        self.as_engine_text().map(crate::jstr::to_text_cow)
    }
    /// A BigInt from an `i64` (for the embedder's 64-bit integer bridge, e.g. wasm i64).
    pub fn bigint_from_i64(v: i64) -> Value {
        Value::BigInt(crate::bigint::JsBigInt::from(v))
    }
    /// A BigInt from a `u64` (for the embedder's 64-bit bridge, e.g. an unsigned FFI return).
    pub fn bigint_from_u64(v: u64) -> Value {
        Value::BigInt(crate::bigint::JsBigInt::from_u64(v))
    }
    /// A BigInt from an `i128` (an FFI `int64_t` widened to preserve its sign).
    pub fn bigint_from_i128(v: i128) -> Value {
        Value::BigInt(crate::bigint::JsBigInt::from_i128(v))
    }
    pub(crate) fn as_obj(&self) -> Option<Gc> {
        match self.view() {
            ValueView::Obj(object) => Some(object),
            _ => None,
        }
    }
    /// The number, if this is a `Number` (an embedder convenience for reading op arguments).
    pub fn as_num_opt(&self) -> Option<f64> {
        match &(self).view() {
            ValueView::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn is_callable(&self) -> bool {
        matches!(&(self).view(), ValueView::Obj(o) if o.borrow().kind.is_callable())
    }
    /// The `typeof` result determined by the value alone. The Annex B.3.6 [[IsHTMLDDA]] object
    /// (`document.all`) is branded in its Agent, so this reports "function" for it; use
    /// `Ctx::type_of` for the operator's exact result.
    pub fn type_of(&self) -> &'static str {
        match &(self).view() {
            ValueView::Undefined | ValueView::Empty => "undefined",
            ValueView::Null => "object",
            ValueView::Bool(_) => "boolean",
            ValueView::Num(_) => "number",
            ValueView::BigInt(_) => "bigint",
            ValueView::Str(_) => "string",
            ValueView::Sym(_) => "symbol",
            ValueView::Obj(o) => {
                if !o.borrow().kind.is_callable() {
                    "object"
                } else {
                    "function"
                }
            }
        }
    }
}

/// The literal result category of a typeof comparison. Scalar discriminants
/// match the existing native Value-kind decoder; objects need live call/HTMLDDA
/// checks. Unknown literals never match any typeof result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TypeofTest {
    Undefined = 0,
    Boolean = 3,
    Number = 4,
    BigInt = 5,
    String = 6,
    Symbol = 7,
    Object = 8,
    Function = 9,
    Never = 255,
}

impl TypeofTest {
    pub(crate) fn from_literal(name: &str) -> Self {
        match name {
            "undefined" => Self::Undefined,
            "boolean" => Self::Boolean,
            "number" => Self::Number,
            "bigint" => Self::BigInt,
            "string" => Self::String,
            "symbol" => Self::Symbol,
            "object" => Self::Object,
            "function" => Self::Function,
            _ => Self::Never,
        }
    }
}

/// A closure's [`UserCallable`], shared by handle. P1 S4: a payload cell that traces
/// its environment, so creating a closure takes one 32-byte cell, and a call copies the handle
/// without touching a count.
pub type UserFn = crate::gc_box::GcBox<UserCallable>;

/// The address of a closure's [`UserCallable`]: its identity.
pub(crate) fn user_fn_ptr(user: &UserFn) -> *const UserCallable {
    crate::gc_box::GcBox::as_ptr(user)
}

/// What kind of object a cell holds, beyond its properties (W2d step 1): an ordinary object
/// (`Plain`), a typed array's view, one of the callables, a Map's or Set's data, or one of the
/// exotic built-in kinds. One field replaces the object's former `call`, `exotic` and typed-array
/// view fields: no object holds two of them (W2d's census).
///
/// `repr(u8)` with fixed discriminants gives the enum a defined layout: the discriminant byte at
/// offset 0 and the single pointer-sized payload at offset 8 (16 bytes). The tags are ordered so
/// that each of the old tests is one unsigned compare, which generated code relies on (see
/// [`KIND_CALLABLE_FIRST`], [`KIND_EXOTIC_NONE_LAST`] and [`KIND_ORDINARY_LAST`]):
/// - the former "`exotic` is None" (plain objects, typed arrays and functions): tags up to
///   [`KIND_EXOTIC_NONE_LAST`];
/// - the former `ObjectKind::is_ordinary` (those, and a Map's or Set's data): tags up to
///   [`KIND_ORDINARY_LAST`];
/// - the former "`call` is not None": [`KIND_CALLABLE_FIRST`] ..= [`KIND_EXOTIC_NONE_LAST`].
///
/// The JIT's code-keyed call probe reads [`ObjectKind::User`]'s tag and its [`UserFn`] word (the
/// closure cell's address) straight from a callee object (see `JitLayout::obj_kind` and
/// `JitLayout::user_func`).
#[repr(u8)]
#[allow(dead_code)]
pub enum ObjectKind {
    /// An ordinary object with nothing beyond its properties.
    Plain = 0,
    /// A typed array's view (its buffer, offset and element kind), which generated element
    /// accesses read. The object's behavior otherwise lives in the interpreter's side table, and
    /// its clear `ic_plain` keeps the property caches off it.
    TypedArray(crate::gc_storage::PayloadBox<crate::native_typed_array::NativeTypedArray>) = 1,
    Native(NativeFn) = 2,
    /// A native function carrying captured state (see [`NativeClosure`]).
    NativeData(crate::gc_storage::SharedPayload<Rc<NativeCallable>>) = 3,
    /// An interpreted function: its AST plus the lexical environment it closed over.
    User(UserFn) = CALLABLE_USER_TAG,
    /// The result of `Function.prototype.bind`.
    Bound(crate::gc_storage::SharedPayload<BoundCallable>) = 5,
    /// A ShadowRealm wrapped function: `target` is a callable inside the realm of the ShadowRealm
    /// object `realm`, which the wrapper keeps alive. Calls marshal primitive args in and the
    /// primitive result out.
    WrappedShadow(crate::gc_storage::SharedPayload<Rc<WrappedShadowCallable>>) = 6,
    /// The inverse: a function living *inside* a ShadowRealm whose `target` is a callable of the
    /// host realm. It runs only in its own realm's interpreter, which knows its host (see
    /// `Interp::shadow_home`).
    WrappedCross(crate::gc_storage::SharedPayload<WrappedCrossCallable>) = 7,
    /// An auto-accessor's synthesized getter: reads the private backing field (brand-checked) off
    /// the receiver.
    AccessorGet(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) = 8,
    /// An auto-accessor's synthesized setter: writes the private backing field (brand-checked).
    AccessorSet(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) = 9,
    /// A decorator `context.access.get`: returns `args[0][name]`.
    PropGet(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) = 10,
    /// A decorator `context.access.set`: performs `args[0][name] = args[1]`.
    PropSet(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) = 11,
    /// A decorator's private access functions use PrivateGet/PrivateSet, not [[Get]]/[[Set]].
    PrivatePropGet(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) = 12,
    PrivatePropSet(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) = 13,
    /// A host callback called through its actual scoped Rust signature. Its result is mapped
    /// explicitly at the public boundary, never reinterpreted as an engine-native Result.
    #[cfg(feature = "embed")]
    ScopedNative(crate::scope::NativeFn) = 14,
    /// P1 S4: a Map's or Set's [[MapData]]/[[SetData]], in its own cell (see
    /// `Interp::collection`). Its tag directly follows the callables', so that
    /// [`ObjectKind::is_ordinary`] tests one tag range.
    Collection(crate::gc_storage::PayloadBox<crate::ordered_collection::OrderedCollection>) = 15,
    Array = 16,
    /// The [[BooleanData]], [[NumberData]], [[StringData]], [[SymbolData]] and [[BigIntData]] of
    /// primitive wrapper objects.
    BoolWrap(bool) = 17,
    NumWrap(f64) = 18,
    StrWrap(crate::gc_storage::PayloadBox<crate::lstr::LStr>) = 19,
    SymWrap(SymbolRef) = 20,
    BigIntWrap(crate::gc_storage::PayloadBox<crate::bigint::JsBigInt>) = 21,
    /// An error object. Carries the stack frames captured at construction (see
    /// `crate::stack_trace`), None when `Error.stackTraceLimit` was not a Number then; the
    /// `Error.prototype.stack` getter formats them after the live `name: message` head.
    /// name/message live as ordinary properties, and the tag lets `Error.prototype.toString` /
    /// the test262 runner recognise an error cheaply.
    Error(Option<crate::gc_storage::PayloadBox<crate::stack_trace::StackTrace>>) = 22,
    /// An `arguments` exotic object; a mapped one's ParameterMap, while it maps an index, in a
    /// storage block it reports (`interpreter::ParameterMap`).
    Arguments(Option<crate::gc_storage::StorageBox<crate::interpreter::ParameterMap>>) = 23,
    ArrayIterator(crate::gc_storage::PayloadBox<ArrayIteratorState>) = 24,
    StringIterator(crate::gc_storage::PayloadBox<StringIteratorState>) = 25,
    /// ECMA-262 CreateMapIterator's and CreateSetIterator's private state: the collection and a
    /// live cursor over it, which its collection's deletions and clears keep in place.
    CollectionIterator(
        crate::gc_storage::PayloadBox<crate::ordered_collection::CollectionIterator>,
    ) = 26,
}

/// The discriminant of [`ObjectKind::User`] (see the enum's layout note).
pub(crate) const CALLABLE_USER_TAG: u8 = 4;
/// The first callable's tag: callables are [`KIND_CALLABLE_FIRST`] ..= [`KIND_EXOTIC_NONE_LAST`].
pub(crate) const KIND_CALLABLE_FIRST: u8 = 2;
/// The last tag of the kinds that carry no exotic internal state (plain objects, typed arrays'
/// views and the callables): the former `ObjectKind::ExoticNone`.
pub(crate) const KIND_EXOTIC_NONE_LAST: u8 = 14;
/// The last tag [`ObjectKind::is_ordinary`] accepts: those, and a Map's or Set's data.
pub(crate) const KIND_ORDINARY_LAST: u8 = 15;
/// [`ObjectKind::TypedArray`]'s tag.
pub(crate) const KIND_TYPED_ARRAY_TAG: u8 = 1;
/// [`ObjectKind::Array`]'s tag.
pub(crate) const KIND_ARRAY_TAG: u8 = 16;

/// Cold payloads boxed out of [`Callable`], so every non-callable ordinary object does not pay
/// for the largest function variants inline.
#[derive(Clone)]
pub struct BoundCallable {
    pub(crate) target: Gc,
    pub(crate) this: Value,
    pub(crate) args: Vec<Value>,
    /// The arguments' buffer, charged to the heap's external ledger ([`ExternalCharge`]) and
    /// released when the payload cell drops it (the struct itself is in the cell).
    #[allow(dead_code)]
    charge: ExternalCharge,
}

pub struct NativeCallable {
    pub(crate) body: NativeCallableBody,
    pub(crate) retained: Option<Rc<dyn NativeCallableRetained>>,
    /// Registration identity used only by opt-in diagnostics. This is deliberately separate
    /// from the observable `name` property: author code may rewrite that property at any time,
    /// while a profile must continue to attribute calls to the operation the embedder registered.
    pub(crate) identity: Rc<str>,
}

pub(crate) enum NativeCallableBody {
    Opaque(Rc<NativeClosure>),
    Captured {
        function: NativeCaptureFn,
        captures: Box<[Value]>,
    },
    #[cfg(feature = "embed")]
    ScopedCaptured {
        function: crate::scope::NativeCaptureFn,
        captures: Box<[Value]>,
    },
    /// A closure with engine-owned captures (S6 step 11b).
    CapturedClosure {
        function: Rc<CaptureClosure>,
        captures: Box<[Value]>,
    },
}

/// `repr(C)`: the JIT's code-keyed call probe reads `func`, `env` and `realm` at fixed offsets
/// from the `UserCallable` data, which the [`UserFn`] word leads to (see `JitLayout::user_func`).
#[repr(C)]
pub struct UserCallable {
    pub(crate) func: ClosureFunction,
    pub(crate) env: Env,
    /// Heap identity of the Realm whose global environment created this
    /// function. Cross-Realm calls must temporarily activate that Realm so
    /// intrinsics and host settings follow the function's [[Realm]].
    pub(crate) realm: usize,
    /// A lazily allocated proof for a getter body that returns one captured binding. Kept after
    /// the JIT-visible prefix so ordinary call probes retain their established field offsets;
    /// most functions pay only this nullable pointer and allocate no cache. A payload cell the
    /// closure reports, so the closure itself has nothing to drop; installing one in a closure
    /// that may be old sets the closure's card (`UserCallable::put_binding_getter_cache`).
    pub(crate) cache: Cell<Option<crate::gc_storage::PayloadBox<UserCallableCache>>>,
}

/// P1 S6: closures die without finalizers: the closure cell holds no counted reference.
const _: () = assert!(!std::mem::needs_drop::<UserCallable>());

/// A closure's function: an uncounted copy of the `Rc` that its heap's closure registry holds
/// once for all the function's closures there (`GcState::closure_functions`), so a dead closure
/// has nothing to release. The registry keeps the function until a collection finds no closure
/// of it alive in the heap (`sweep_closure_functions`): the next minor collection for a
/// function whose closures all died young, a major one once one had survived, which is when the
/// closures' finalizers used to release it. The same word as the `Rc`, which the JIT reads
/// (`JitLayout::user_func`).
#[repr(transparent)]
pub(crate) struct ClosureFunction(std::mem::ManuallyDrop<Rc<Function>>);

impl ClosureFunction {
    /// A closure's copy of `func`, which the active heap's registry must hold
    /// ([`hold_closure_function`]) for as long as the closure can be reached.
    fn new(func: &Rc<Function>) -> Self {
        // SAFETY: a bitwise copy of a live `Rc`, never dropped (`ManuallyDrop`) and never moved
        // out: the registry's count stands for it.
        Self(std::mem::ManuallyDrop::new(unsafe { std::ptr::read(func) }))
    }
}

impl std::ops::Deref for ClosureFunction {
    type Target = Rc<Function>;
    #[inline]
    fn deref(&self) -> &Rc<Function> {
        &self.0
    }
}

/// The existing nullable closure sidecar also pins foreign runtime entries. Primary closures
/// need no pin: their AST owns its stable primary entry. Runtime leases survive getter resets.
pub(crate) struct UserCallableCache {
    pub(crate) runtime: Option<Rc<crate::function_runtime::FunctionRuntime>>,
    pub(crate) binding_getter: Option<crate::interpreter::BindingGetterCache>,
    /// This closure's own runtime ScriptOrModule (`None` included) when `own_origin`. Otherwise
    /// the closure shares its function's primary record (`Function::shared_origin`).
    pub(crate) origin: Option<crate::agent_control::RuntimeOrigin>,
    pub(crate) own_origin: bool,
}

impl UserCallable {
    pub(crate) fn take_binding_getter_cache(
        &self,
    ) -> Option<crate::interpreter::BindingGetterCache> {
        let mut sidecar = self.cache.take()?;
        let proof = sidecar.binding_getter.take();
        self.cache.set(Some(sidecar));
        proof
    }

    pub(crate) fn put_binding_getter_cache(
        &self,
        proof: Option<crate::interpreter::BindingGetterCache>,
    ) {
        let mut sidecar = self.cache.take();
        if let Some(sidecar) = &mut sidecar {
            sidecar.binding_getter = proof;
        } else if proof.is_some() {
            let made = crate::gc_storage::PayloadBox::new(UserCallableCache {
                runtime: None,
                binding_getter: proof,
                origin: None,
                own_origin: false,
            });
            // A new edge from a closure that may be old (see `UserCallable::cache`).
            // SAFETY: closure cells are carded, and the payload is the cell's start
            // (`gc_box::new`).
            unsafe { crate::gc_core::card_mark((self as *const Self).cast()) };
            sidecar = Some(made);
        }
        if sidecar.as_ref().is_some_and(|sidecar| {
            sidecar.runtime.is_none() && sidecar.binding_getter.is_none() && !sidecar.own_origin
        }) {
            sidecar = None;
        }
        self.cache.set(sidecar);
    }

    /// Immutable source-record ownership is installed before the closure is published. Even
    /// context-zero functions must retain that actual record when later invoked in another
    /// interpreter; the current interpreter/parse cache cannot reconstruct its identity.
    pub(crate) fn runtime_origin(
        &self,
        _owner: crate::agent_access::RealmId,
        _agent: u64,
    ) -> Option<crate::agent_control::RuntimeOrigin> {
        if self.func.self_hosted {
            return None;
        }
        let sidecar = self.cache.take();
        let own = sidecar
            .as_ref()
            .filter(|sidecar| sidecar.own_origin)
            .map(|sidecar| sidecar.origin.clone());
        self.cache.set(sidecar);
        own.unwrap_or_else(|| self.func.shared_origin().cloned())
    }

    fn cloned_runtime_pin(&self) -> Option<crate::gc_storage::PayloadBox<UserCallableCache>> {
        let sidecar = self.cache.take();
        let metadata = sidecar.as_ref().map(|sidecar| {
            (
                sidecar.runtime.clone(),
                sidecar.origin.clone(),
                sidecar.own_origin,
            )
        });
        self.cache.set(sidecar);
        metadata.and_then(|(runtime, origin, own_origin)| {
            (runtime.is_some() || own_origin).then(|| {
                crate::gc_storage::PayloadBox::new(UserCallableCache {
                    runtime,
                    binding_getter: None,
                    origin,
                    own_origin,
                })
            })
        })
    }
}

impl Clone for UserCallable {
    fn clone(&self) -> Self {
        hold_closure_function(&self.func);
        Self {
            func: ClosureFunction::new(&self.func),
            env: self.env,
            realm: self.realm,
            // A cloned callable starts without a resolution proof. Its new owner can refill a
            // cache against the same live environment when it is first used.
            cache: Cell::new(self.cloned_runtime_pin()),
        }
    }
}

#[derive(Clone)]
pub struct WrappedShadowCallable {
    /// The ShadowRealm object whose realm `target` lives in (a traced edge).
    pub(crate) realm: Gc,
    pub(crate) target: Box<Value>,
}

#[derive(Clone)]
pub struct WrappedCrossCallable {
    pub(crate) target: Value,
}

impl ObjectKind {
    /// The discriminant byte (see the enum's layout note).
    #[inline]
    pub(crate) fn tag(&self) -> u8 {
        // SAFETY: `repr(u8)` puts the discriminant in the first byte.
        unsafe { *(self as *const ObjectKind as *const u8) }
    }

    /// Whether the object has a [[Call]] (any callable kind): the former "`call` is not None".
    #[inline]
    pub(crate) fn is_callable(&self) -> bool {
        (KIND_CALLABLE_FIRST..=KIND_EXOTIC_NONE_LAST).contains(&self.tag())
    }

    /// No exotic internal state: a plain object, a typed array's view (whose behavior is the
    /// side table's) or a callable. The former "`exotic` is None", which functions and typed
    /// arrays passed.
    #[inline]
    pub(crate) fn exotic_is_none(&self) -> bool {
        self.tag() <= KIND_EXOTIC_NONE_LAST
    }

    /// No exotic internal methods: [`ObjectKind::exotic_is_none`], or a Map's or Set's
    /// [[MapData]]/[[SetData]], which an otherwise ordinary object carries in its cell. Property
    /// caches and fast paths that admit plain objects ask this, so Maps and Sets keep them.
    #[inline]
    pub(crate) fn is_ordinary(&self) -> bool {
        self.tag() <= KIND_ORDINARY_LAST
    }

    /// A Map's or Set's in-cell data.
    #[cfg(test)]
    #[inline]
    pub(crate) fn is_collection(&self) -> bool {
        matches!(self, ObjectKind::Collection(_))
    }

    /// A typed array's view, which generated element accesses read.
    #[inline]
    pub(crate) fn typed_array(
        &self,
    ) -> Option<&crate::gc_storage::PayloadBox<crate::native_typed_array::NativeTypedArray>> {
        match self {
            ObjectKind::TypedArray(view) => Some(view),
            _ => None,
        }
    }

    /// A copy of the object's [[Call]] (the former `call` field's clone): a callable kind's
    /// copy, which shares its payload cell, or `Plain` for an object that is not callable.
    pub(crate) fn clone_callable(&self) -> ObjectKind {
        match self {
            ObjectKind::Native(f) => ObjectKind::Native(*f),
            ObjectKind::NativeData(d) => ObjectKind::NativeData(*d),
            ObjectKind::User(u) => ObjectKind::User(*u),
            ObjectKind::Bound(b) => ObjectKind::Bound(*b),
            ObjectKind::WrappedShadow(w) => ObjectKind::WrappedShadow(*w),
            ObjectKind::WrappedCross(w) => ObjectKind::WrappedCross(*w),
            ObjectKind::AccessorGet(k) => ObjectKind::AccessorGet(*k),
            ObjectKind::AccessorSet(k) => ObjectKind::AccessorSet(*k),
            ObjectKind::PropGet(k) => ObjectKind::PropGet(*k),
            ObjectKind::PropSet(k) => ObjectKind::PropSet(*k),
            ObjectKind::PrivatePropGet(k) => ObjectKind::PrivatePropGet(*k),
            ObjectKind::PrivatePropSet(k) => ObjectKind::PrivatePropSet(*k),
            #[cfg(feature = "embed")]
            ObjectKind::ScopedNative(f) => ObjectKind::ScopedNative(*f),
            _ => ObjectKind::Plain,
        }
    }

    /// Native capability, independent of the Rust callback signature. Intrinsic identity
    /// recognizers must continue to match only `Native`, whose type is the engine signature.
    pub(crate) fn is_native(&self) -> bool {
        match self {
            Self::Native(_) | Self::NativeData(_) => true,
            #[cfg(feature = "embed")]
            Self::ScopedNative(_) => true,
            _ => false,
        }
    }

    pub(crate) fn user(
        func: Rc<Function>,
        env: Env,
        realm: usize,
        origin: Option<crate::agent_control::RuntimeOrigin>,
        _owner: crate::agent_access::RealmId,
    ) -> ObjectKind {
        let lease = func.runtime();
        // The active heap's: `runtime` finds the entry of the active heap.
        let heap_id = lease.heap.id;
        let (runtime, own_origin) = match lease {
            // Most closures of one function in one heap come from the same record: they share
            // the primary entry's copy, and only a differing (or absent) record is boxed.
            crate::function_runtime::RuntimeLease::Primary(primary) => {
                let shared = origin.as_ref().is_some_and(|origin| {
                    Rc::ptr_eq(primary.origin.get_or_init(|| origin.clone()), origin)
                });
                (None, !shared && !func.self_hosted)
            }
            crate::function_runtime::RuntimeLease::Foreign(runtime) => (Some(runtime), true),
        };
        let metadata = (runtime.is_some() || own_origin).then(|| {
            crate::gc_storage::PayloadBox::new(UserCallableCache {
                runtime,
                binding_getter: None,
                origin: if own_origin { origin } else { None },
                own_origin,
            })
        });
        if func.runtime_cache().closure_heap.get() != heap_id {
            hold_closure_function_in(&func, heap_id);
        }
        let user = UserCallable {
            func: ClosureFunction::new(&func),
            env,
            realm,
            cache: Cell::new(metadata),
        };
        ObjectKind::User(crate::gc_box::new(user))
    }

    pub(crate) fn wrapped_shadow(realm: Gc, target: Value) -> ObjectKind {
        ObjectKind::WrappedShadow(crate::gc_storage::SharedPayload::new(Rc::new(
            WrappedShadowCallable {
                realm,
                target: Box::new(target),
            },
        )))
    }

    /// A native function with captured state, its data in a payload cell.
    pub(crate) fn native_data(data: Rc<NativeCallable>) -> ObjectKind {
        ObjectKind::NativeData(crate::gc_storage::SharedPayload::new(data))
    }

    /// An accessor or decorator access function (`variant`) for the field `key`, its key in a
    /// payload cell.
    pub(crate) fn accessor_key(
        variant: fn(crate::gc_storage::SharedPayload<Rc<Rc<str>>>) -> ObjectKind,
        key: Rc<str>,
    ) -> ObjectKind {
        variant(crate::gc_storage::SharedPayload::new(Rc::new(key)))
    }

    pub(crate) fn bound(target: Gc, this: Value, args: Vec<Value>) -> ObjectKind {
        let mut charge = ExternalCharge::default();
        charge.settle(args.capacity() * std::mem::size_of::<Value>());
        ObjectKind::Bound(crate::gc_storage::SharedPayload::new(BoundCallable {
            target,
            this,
            args,
            charge,
        }))
    }

    pub(crate) fn wrapped_cross(target: Value) -> ObjectKind {
        ObjectKind::WrappedCross(crate::gc_storage::SharedPayload::new(
            WrappedCrossCallable { target },
        ))
    }
}

/// ECMA-262 CreateArrayIterator's private slots. They are not own properties:
/// Reflect.ownKeys, assignment and Object.freeze cannot observe or alter them.
#[derive(Clone)]
pub struct ArrayIteratorState {
    pub(crate) target: Value,
    pub(crate) index: usize,
    pub(crate) kind: u8,
}

/// String iterator closure state. `index` is a code unit index; each transition consumes exactly
/// one ECMAScript code point.
#[derive(Clone)]
pub struct StringIteratorState {
    pub(crate) string: Option<crate::lstr::LStr>,
    pub(crate) index: usize,
}

impl ObjectKind {
    pub(crate) fn str_wrap(value: crate::lstr::LStr) -> ObjectKind {
        ObjectKind::StrWrap(crate::gc_storage::PayloadBox::new(value))
    }

    pub(crate) fn bigint_wrap(value: crate::bigint::JsBigInt) -> ObjectKind {
        ObjectKind::BigIntWrap(crate::gc_storage::PayloadBox::new(value))
    }

    pub(crate) fn error(stack: Option<Box<crate::stack_trace::StackTrace>>) -> ObjectKind {
        ObjectKind::Error(stack.map(|stack| crate::gc_storage::PayloadBox::new(*stack)))
    }
}

const _: () = {
    assert!(std::mem::size_of::<ObjectKind>() == 16);
    assert!(std::mem::align_of::<ObjectKind>() == 8);
};

pub(crate) struct Object {
    pub(crate) proto: Option<Gc>,
    pub(crate) props: Props,
    /// What the object is beyond its properties: plain, a typed array's view, a callable, a
    /// collection or an exotic built-in kind ([`ObjectKind`]). Its flags ([[Extensible]],
    /// `ic_plain`, IsConstructor and the class kind) are bits of its map's flag byte
    /// ([`FLAG_IC_PLAIN`] and the rest), read and written through the methods below.
    pub(crate) kind: ObjectKind,
}

impl Object {
    /// [[Extensible]] ([`FLAG_EXTENSIBLE`]).
    #[inline]
    pub(crate) fn extensible(&self) -> bool {
        self.props.flag(FLAG_EXTENSIBLE)
    }
    #[inline]
    pub(crate) fn set_extensible(&self, extensible: bool) {
        self.props.set_flag(FLAG_EXTENSIBLE, extensible);
    }
    /// No side-table behavior ([`FLAG_IC_PLAIN`]).
    #[inline]
    pub(crate) fn ic_plain(&self) -> bool {
        self.props.flag(FLAG_IC_PLAIN)
    }
    #[inline]
    pub(crate) fn set_ic_plain(&self, plain: bool) {
        self.props.set_flag(FLAG_IC_PLAIN, plain);
    }
    /// IsConstructor ([`FLAG_CONSTRUCTOR`]).
    #[inline]
    pub(crate) fn is_constructor(&self) -> bool {
        self.props.flag(FLAG_CONSTRUCTOR)
    }
    #[inline]
    pub(crate) fn set_constructor(&self, constructor: bool) {
        self.props.set_flag(FLAG_CONSTRUCTOR, constructor);
    }
    /// A class constructor's kind ([`CLASS_BASE`] or [`CLASS_DERIVED`]; 0 for every other
    /// object, see [`FLAG_CLASS_SHIFT`]).
    #[inline]
    pub(crate) fn class_kind(&self) -> u8 {
        (self.props.flags.get() & FLAG_CLASS_MASK) >> FLAG_CLASS_SHIFT
    }
    #[inline]
    pub(crate) fn set_class_kind(&self, kind: u8) {
        debug_assert!(kind <= CLASS_DERIVED);
        let flags = self.props.flags.get() & !FLAG_CLASS_MASK;
        self.props.flags.set(flags | (kind << FLAG_CLASS_SHIFT));
    }
}

/// [`Object::class_kind`] of a base class's constructor.
pub(crate) const CLASS_BASE: u8 = 1;
/// [`Object::class_kind`] of a derived class's constructor (`class C extends B`).
pub(crate) const CLASS_DERIVED: u8 = 2;

// An object cell owns nothing to release: its Rust-heap payloads (a Map's tables, a bound
// function's arguments, an error's stack trace, a map's hash index...) sit in payload cells of
// their own, whose finalizers drop them (`gc_storage::PayloadBox`), and its properties and
// elements in storage blocks. So an object dies without a finalizer, its death a bit in its
// page; a field that needed dropping would fail here.
const _: () = assert!(!std::mem::needs_drop::<RefCell<Object>>());

impl Object {
    /// Engine state shares the existing lazy storage allocation but remains separate from
    /// ECMAScript properties. Author strings, reflection, proxies and integrity levels cannot
    /// address this map (ECMA-262 6.1.7.2). Ordinary objects retain their compact body layout.
    pub(crate) fn internal_slots(&self) -> &InternalSlots {
        self.props.elems.internal_slots()
    }

    pub(crate) fn internal_slots_mut(&mut self) -> &mut InternalSlots {
        &mut self.props.elems.extras_mut().internal_slots
    }

    fn take_internal_slots(&mut self) -> InternalSlots {
        self.props
            .elems
            .1
            .as_deref_mut()
            .map_or_else(InternalSlots::default, |extras| {
                std::mem::take(&mut extras.internal_slots)
            })
    }

    /// Replace only the receiver's ordinary property payload. Engine state belongs to this
    /// receiver, even when an Array.fill fast path materializes fresh dense storage around it.
    pub(crate) fn replace_properties(&mut self, mut properties: Props) {
        // The object's flag bits stay with it: its extensibility, `ic_plain`, constructor and
        // class bits, and `proto_flag`, a property of the OBJECT's history (it served as a live
        // prototype), not of either property map: a whole-map swap must not clear it, or later
        // structural changes would stop bumping the creation-IC epoch.
        let proto = (self.props.flags.get() | properties.flags.get()) & FLAG_PROTO;
        properties.flags.set(
            properties.flags.get() & !OBJECT_FLAGS | self.props.flags.get() & OBJECT_FLAGS | proto,
        );
        let slots = self.take_internal_slots();
        if let Some(extras) = properties.elems.1.as_deref_mut() {
            extras.internal_slots = slots;
        } else if slots.entries.is_some() {
            properties.elems.extras_mut().internal_slots = slots;
        }
        // The words' remembered record described their old owner's marks.
        properties.elems.2.force_full();
        self.props = properties;
    }

    /// Install a map built for this object (a function template's copy) in one move. The
    /// object's flag bits stay its own (see [`OBJECT_FLAGS`]); the map's own bits come with it.
    #[inline]
    pub(crate) fn adopt_props(&mut self, props: Props) {
        let object_bits = self.props.flags.get() & OBJECT_FLAGS;
        self.props = props;
        let map_bits = self.props.flags.get() & !OBJECT_FLAGS;
        self.props.flags.set(map_bits | object_bits);
    }

    // Objects exist only behind their collected handle; `Gc` is that handle.
    #[allow(clippy::new_ret_no_self)]
    #[cfg_attr(feature = "architecture-diagnostics", track_caller)]
    pub(crate) fn new(proto: Option<Gc>) -> Gc {
        Self::new_with_capacity(proto, 0)
    }

    /// Allocate an ordinary object's named-property vector at its known final size. Constructor
    /// chunks derive a conservative straight-line field count, replacing the usual 1 → 2 → 4
    /// growth sequence with one exact allocation. The hint lives on shared code, not instances.
    #[cfg_attr(feature = "architecture-diagnostics", track_caller)]
    pub(crate) fn new_with_capacity(proto: Option<Gc>, property_capacity: usize) -> Gc {
        Self::new_with_parts(
            proto,
            Props::with_capacity(property_capacity),
            ObjectKind::Plain,
        )
    }

    /// Allocate an object around an already-finalized property map. Literal fast paths can build
    /// the map from moved stack values before allocation, avoiding an empty map plus RefCell
    /// replacement on every object.
    #[cfg_attr(feature = "architecture-diagnostics", track_caller)]
    pub(crate) fn new_with_parts(proto: Option<Gc>, props: Props, kind: ObjectKind) -> Gc {
        // Moving a property payload to a new receiver cannot inherit the old owner's proof.
        props.elems.2.force_full();
        // Nor its flags: a new object is extensible and ordinary, with the map's own bits.
        props
            .flags
            .set(props.flags.get() & !OBJECT_FLAGS | OBJECT_FLAGS_INITIAL);
        #[cfg(feature = "architecture-diagnostics")]
        let caller = std::panic::Location::caller();
        with_active_gc_heap(|heap| {
            debug_assert!(
                proto.as_ref().is_none_or(|prototype| {
                    heap.core
                        .borrow()
                        .owns_cell(Gc::as_ptr(prototype) as usize, heap.object_desc)
                }),
                "object prototype belongs to another Agent heap (I6)"
            );
            heap.allocated.set(heap.allocated.get().wrapping_add(1));
            let obj = Gc::new_in(heap, RefCell::new(Object { proto, props, kind }));
            #[cfg(feature = "architecture-diagnostics")]
            {
                let object = obj.borrow();
                heap.diagnostics.borrow_mut().born(
                    Gc::as_ptr(&obj) as usize,
                    caller,
                    &object.kind,
                    object.props.entries.capacity(),
                );
            }
            obj
        })
    }
}

/// A lazy internal-slot map inside the property map's Rust-heap extras. Built-in instances
/// keep their state without adding property keys, descriptors or shape transitions; ordinary
/// objects allocate neither a separate map nor an extra pointer in their object body.
///
/// Its storage is charged to the heap's external ledger ([`ExternalCharge`]) as it grows, and
/// released when it drops.
#[derive(Default)]
pub(crate) struct InternalSlots {
    entries: Option<Box<Vec<(Rc<str>, Property)>>>,
    charge: ExternalCharge,
}

// Host-selected essential internal methods, never ECMAScript properties
// (`Ctx::set_host_object_invariants`). HTML's WindowProxy uses both rules; ECMA-262
// #sec-immutable-prototype-exotic-objects permits host-defined combinations.
pub(crate) const HOST_IMMUTABLE_PROTOTYPE: &str = "#\0hostImmutablePrototype";
pub(crate) const HOST_ALWAYS_EXTENSIBLE: &str = "#\0hostAlwaysExtensible";

impl Clone for InternalSlots {
    fn clone(&self) -> Self {
        let mut copy = Self {
            entries: self.entries.clone(),
            charge: ExternalCharge::new(),
        };
        copy.charge.settle(copy.retained_storage_bytes());
        copy
    }
}

impl InternalSlots {
    pub(crate) fn get(&self, key: &str) -> Option<&Property> {
        self.entries
            .as_ref()?
            .iter()
            .find(|(name, _)| &**name == key)
            .map(|(_, value)| value)
    }

    pub(crate) fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut Property> {
        self.entries
            .as_mut()?
            .iter_mut()
            .find(|(name, _)| &**name == key)
            .map(|(_, value)| value)
    }

    pub(crate) fn entry_at(&self, slot: usize) -> Option<&(Rc<str>, Property)> {
        self.entries.as_ref()?.get(slot)
    }

    pub(crate) fn entry_at_mut(&mut self, slot: usize) -> Option<&mut (Rc<str>, Property)> {
        self.entries.as_mut()?.get_mut(slot)
    }

    pub(crate) fn slot_of(&self, key: &str) -> Option<usize> {
        self.entries
            .as_ref()?
            .iter()
            .position(|(name, _)| &**name == key)
    }

    pub(crate) fn property_at(&self, slot: usize) -> Option<&Property> {
        self.entry_at(slot).map(|(_, property)| property)
    }

    pub(crate) fn insert(&mut self, key: impl IntoSlotKey, value: Property) {
        let key = key.into_slot_key();
        let before = self.retained_storage_bytes();
        let entries = self.entries.get_or_insert_with(|| Box::new(Vec::new()));
        if let Some((_, previous)) = entries.iter_mut().find(|(name, _)| **name == *key) {
            *previous = value;
        } else {
            entries.push((key, value));
        }
        let after = self.retained_storage_bytes();
        if after != before {
            self.charge.settle(after);
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&Rc<str>, &Property)> {
        self.entries
            .iter()
            .flat_map(|entries| entries.iter())
            .map(|(key, value)| (key, value))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &Property> {
        self.iter().map(|(_, value)| value)
    }

    pub(crate) fn retained_storage_bytes(&self) -> usize {
        self.entries.as_ref().map_or(0, |entries| {
            std::mem::size_of_val(&**entries)
                + entries.capacity() * std::mem::size_of::<(Rc<str>, Property)>()
        })
    }
}

pub(crate) struct GcState {
    pub(crate) agent: crate::agent_control::AgentControl,
    heap_id: u64,
    #[cfg(feature = "architecture-diagnostics")]
    diagnostics: RefCell<allocation_diagnostics::State>,
    weak_observers: RefCell<crate::fasthash::FastMap<usize, crate::weak_metadata::DeathObservers>>,
    shapes: RefCell<ShapeTable>,
    function_runtimes: RefCell<
        crate::fasthash::FastMap<u64, std::rc::Weak<crate::function_runtime::FunctionRuntime>>,
    >,
    /// The functions of this heap's closures, each held once for all of them, by address (see
    /// [`ClosureFunction`]).
    closure_functions: RefCell<crate::fasthash::FastMap<usize, Rc<Function>>>,
    /// The registry's functions taken since the last collection, none of whose closures has
    /// survived one: a minor collection, which traces every live young closure, lets go of
    /// those it found no closure of.
    young_closure_functions: RefCell<Vec<usize>>,
    /// The heap's markings (`gc_core::Control::markings`) when a collection last judged the
    /// registry: the young functions' closures are young only at the marking after it.
    closure_functions_judged: Cell<u64>,
    array_length_shape: Cell<u32>,
    array_length_layout: RefCell<Option<PropertyLayout>>,
    live: LiveCount,
    allocated: Cell<u64>,
    /// P1 S2: the cells holding this Agent's objects. Allocation never collects (see
    /// `new_object_heap`); collections run at interpreter safepoints, which supply the roots.
    pub(crate) core: RefCell<crate::gc_core::Heap>,
    /// The core's control block, through which the containers of its cells' Rust-heap payloads
    /// charge and release their bytes without borrowing the core, which a sweep's finalizers
    /// run under (see [`ExternalCharge`]).
    ledger: Rc<crate::gc_core::Control>,
    pub(crate) object_desc: crate::gc_core::DescId,
    /// Storage blocks (`gc_storage`).
    pub(crate) storage_desc: crate::gc_core::DescId,
    /// Payload cells, by kind (`gc_storage::PayloadBox`).
    payload_descs: PayloadDescs,
    /// Accessor-pair cells (each an [`Accessors`]).
    pub(crate) accessor_desc: crate::gc_core::DescId,
    pub(crate) scope_desc: crate::gc_core::DescId,
    /// BigInt payload cells.
    pub(crate) bigint_desc: crate::gc_core::DescId,
    /// Symbol cells.
    pub(crate) symbol_desc: crate::gc_core::DescId,
    /// String cells (flat buffers and views).
    pub(crate) string_desc: crate::gc_core::DescId,
    /// Closure cells (each a [`UserCallable`]).
    pub(crate) user_fn_desc: crate::gc_core::DescId,
    /// Deferred-prototype cells (each a function's [`LazyFunctionPrototype`]).
    pub(crate) lazy_proto_desc: crate::gc_core::DescId,
    /// Ordered property-key payload cells.
    pub(crate) property_layout_desc: crate::gc_core::DescId,
    /// This heap's interned one-unit ASCII and small array-index strings (`jstr::unit_lstr`,
    /// `jstr::index_lstr`), roots for the heap's lifetime.
    pub(crate) interned_strings: RefCell<Vec<Option<crate::lstr::LStr>>>,
    /// This heap's atoms, roots for the heap's lifetime (see [`crate::atom`]).
    pub(crate) atoms: RefCell<crate::atom::AtomTable>,
    /// The scope descriptors this heap's environments share, kept for the heap's lifetime so that
    /// an environment names its descriptor without owning it ([`keep_scope_desc`]).
    scope_descs: RefCell<Vec<Rc<crate::interpreter::ScopeDesc>>>,
    /// The descriptors of the tree-walked functions' shared activation layouts, which their
    /// ASTs cache as text ([`scope_desc_for`]), by layout, with the layout kept.
    layout_descs: RefCell<
        crate::fasthash::FastMap<
            usize,
            (
                Rc<crate::interpreter::BindingLayout>,
                Rc<crate::interpreter::ScopeDesc>,
            ),
        >,
    >,
    /// Actual AST template and compiled chunk owners, weakly registered in their heap. Their
    /// constants and layouts are roots for the Rust owner's lifetime, including late fields.
    pub(crate) code_roots: RefCell<Vec<std::rc::Weak<dyn crate::gc_trace::GcTrace>>>,
    /// The code roots registered since the last collection. A minor collection traces these
    /// and `young_code_words` instead of every code root: an older root holds old cells, apart
    /// from those stored into it since, which its late fills report ([`remember_code_edges`]).
    young_code_roots: RefCell<Vec<std::rc::Weak<dyn crate::gc_trace::GcTrace>>>,
    /// The edges stored into older code roots since the last collection (see
    /// `young_code_roots`), as cell words.
    young_code_words: RefCell<Vec<u64>>,
    /// S6 step 11b: the Realms an embedder entered and has not left (`Ctx::enter_realm`), each
    /// with the state of the Realm it replaced, which is a root of every collection until the
    /// entry is left. Innermost last.
    #[cfg(feature = "embed")]
    pub(crate) entered_realms: RefCell<Vec<EnteredRealm>>,
    /// S6 step 10d: what the tables of this heap's interpreters wrote since the last
    /// collection's marking consumed it, which a minor collection reports instead of the tables
    /// (see `gc_collect::ClientRecords`).
    pub(crate) records: RefCell<crate::interpreter::gc_collect::ClientRecords>,
    /// S6 step 11a: this heap's rooted handles (`handles::HandleTable`), shared by the handles
    /// themselves, which keep the table, not the heap, alive.
    pub(crate) handles: Rc<crate::handles::HandleTable>,
    /// What the client of this heap's last collection reported (test builds).
    #[cfg(test)]
    pub(crate) client_report: RefCell<crate::interpreter::gc_collect::ClientReport>,
}

pub(crate) type GcHeap = Rc<GcState>;

/// A Realm an embedder entered (`GcState::entered_realms`).
#[cfg(feature = "embed")]
pub(crate) struct EnteredRealm {
    /// The interpreter that entered it, by address.
    pub(crate) interp: crate::agent_access::RealmId,
    /// The entered Realm's key (its global object's address).
    pub(crate) key: usize,
    /// The state of the Realm it replaced, restored when the entry is left.
    pub(crate) saved: crate::interpreter::RealmState,
}

/// The live-object count, which the cell heap keeps (attached by [`new_gc_heap`]): allocation
/// increments it, and each sweep subtracts the objects it finds dead.
pub(crate) struct LiveCount {
    count: Cell<i64>,
}

impl LiveCount {
    fn new(count: i64) -> Self {
        Self {
            count: Cell::new(count),
        }
    }

    fn get(&self) -> i64 {
        self.count.get()
    }

    #[cfg(test)]
    fn as_ptr(&self) -> *mut i64 {
        self.count.as_ptr()
    }
}

/// Keep `word`'s heap bit (`interrupt::POLL_HEAP`) for `heap` from now on: set exactly while a
/// collection is due, the live-object count above its ceiling included (see
/// `gc_core::Control::attach_poll_word`).
pub(crate) fn attach_poll_word(heap: &GcHeap, word: &std::sync::Arc<crate::interrupt::PollWord>) {
    heap.ledger.attach_poll_word(word);
}

/// Stop keeping `word`'s heap bit (its interpreter ends). Needs no borrow of the core: a
/// ShadowRealm's interpreter may end inside its parent's collection.
pub(crate) fn detach_poll_word(heap: &GcHeap, word: &std::sync::Arc<crate::interrupt::PollWord>) {
    heap.ledger.detach_poll_word(word);
}

/// Debug builds: the poll words' heap bit agrees with the core's `poll_due` (checked at
/// `Interp::gc_check`).
pub(crate) fn debug_assert_poll_bit(heap: &GcHeap) {
    if cfg!(debug_assertions) {
        if let Ok(core) = heap.core.try_borrow() {
            assert!(
                core.poll_bit_agrees(),
                "the poll words' heap bit disagrees with poll_due ({})",
                core.poll_due()
            );
        }
    }
}

/// The descriptors of the payload cells objects keep their Rust-heap values in
/// (`gc_storage::PayloadBox`), one per kind.
pub(crate) struct PayloadDescs {
    collection: crate::gc_core::DescId,
    string_wrapper: crate::gc_core::DescId,
    bigint_wrapper: crate::gc_core::DescId,
    stack_trace: crate::gc_core::DescId,
    array_iterator: crate::gc_core::DescId,
    string_iterator: crate::gc_core::DescId,
    collection_iterator: crate::gc_core::DescId,
    native: crate::gc_core::DescId,
    bound: crate::gc_core::DescId,
    wrapped_shadow: crate::gc_core::DescId,
    wrapped_cross: crate::gc_core::DescId,
    accessor_key: crate::gc_core::DescId,
    typed_array: crate::gc_core::DescId,
    extras: crate::gc_core::DescId,
    closure_cache: crate::gc_core::DescId,
    name_index: crate::gc_core::DescId,
}

macro_rules! payload_kinds {
    ($($ty:ty => $field:ident),* $(,)?) => {
        $(
            impl crate::gc_storage::PayloadKind for $ty {
                fn desc(heap: &GcState) -> crate::gc_core::DescId {
                    heap.payload_descs.$field
                }
            }
        )*
    };
}

payload_kinds! {
    crate::ordered_collection::OrderedCollection => collection,
    crate::lstr::LStr => string_wrapper,
    crate::bigint::JsBigInt => bigint_wrapper,
    crate::stack_trace::StackTrace => stack_trace,
    ArrayIteratorState => array_iterator,
    StringIteratorState => string_iterator,
    crate::ordered_collection::CollectionIterator => collection_iterator,
    Rc<NativeCallable> => native,
    BoundCallable => bound,
    Rc<WrappedShadowCallable> => wrapped_shadow,
    WrappedCrossCallable => wrapped_cross,
    Rc<Rc<str>> => accessor_key,
    crate::native_typed_array::NativeTypedArray => typed_array,
    DenseExtras => extras,
    UserCallableCache => closure_cache,
    crate::interpreter::NameIndex => name_index,
}

impl PayloadDescs {
    fn register(core: &mut crate::gc_core::Heap) -> Self {
        use crate::gc_storage::payload_layout;
        let mut leaf = |desc: crate::gc_core::LayoutDesc| {
            let id = core.register_desc(desc);
            core.set_leaf(id);
            id
        };
        Self {
            collection: leaf(
                payload_layout::<crate::ordered_collection::OrderedCollection>("map or set data"),
            ),
            string_wrapper: leaf(payload_layout::<crate::lstr::LStr>("string wrapper data")),
            bigint_wrapper: leaf(payload_layout::<crate::bigint::JsBigInt>(
                "bigint wrapper data",
            )),
            stack_trace: leaf(payload_layout::<crate::stack_trace::StackTrace>(
                "stack trace",
            )),
            array_iterator: leaf(payload_layout::<ArrayIteratorState>("array iterator state")),
            string_iterator: leaf(payload_layout::<StringIteratorState>(
                "string iterator state",
            )),
            collection_iterator: leaf(payload_layout::<
                crate::ordered_collection::CollectionIterator,
            >("map or set iterator state")),
            native: leaf(payload_layout::<Rc<NativeCallable>>("native function data")),
            bound: leaf(payload_layout::<BoundCallable>("bound function data")),
            wrapped_shadow: leaf(payload_layout::<Rc<WrappedShadowCallable>>(
                "wrapped function data",
            )),
            wrapped_cross: leaf(payload_layout::<WrappedCrossCallable>(
                "wrapped function data",
            )),
            accessor_key: leaf(payload_layout::<Rc<Rc<str>>>("accessor function key")),
            typed_array: leaf(
                payload_layout::<crate::native_typed_array::NativeTypedArray>("typed array view"),
            ),
            extras: leaf(payload_layout::<DenseExtras>("property map extras")),
            closure_cache: leaf(payload_layout::<UserCallableCache>("closure cache")),
            name_index: leaf(payload_layout::<crate::interpreter::NameIndex>(
                "environment name index",
            )),
        }
    }
}

/// The descriptors of a [`GcState`]'s cell kinds.
struct CellDescs {
    payloads: PayloadDescs,
    storage: crate::gc_core::DescId,
    accessor: crate::gc_core::DescId,
    object: crate::gc_core::DescId,
    scope: crate::gc_core::DescId,
    bigint: crate::gc_core::DescId,
    symbol: crate::gc_core::DescId,
    string: crate::gc_core::DescId,
    user_fn: crate::gc_core::DescId,
    lazy_proto: crate::gc_core::DescId,
    property_layout: crate::gc_core::DescId,
}

/// The cell heap behind a [`GcState`] and the descriptors of its cell kinds.
fn new_cell_heap() -> (crate::gc_core::Heap, CellDescs) {
    let mut core = crate::gc_core::Heap::new();
    core.set_collect_on_alloc(false);
    crate::interpreter::gc_collect::configure_heap(&mut core);
    // Storage blocks have no edges of their own: their owners trace what they hold.
    let storage = core.register_desc(crate::gc_storage::layout());
    core.set_leaf(storage);
    // Nor have payload cells; their finalizers drop the values objects keep in them.
    let payloads = PayloadDescs::register(&mut core);
    let accessor = core.register_desc(crate::gc_core::LayoutDesc {
        name: "accessor pair",
        trace: trace_accessor_cell,
        mark_trace: None,
        finalize: None,
        // A pair is immutable: a property that changes one gets a new pair.
        carded: false,
    });
    // S6 step 10b: objects, environments and deferred prototypes gain references after their
    // initialization, through writes the barrier records in their pages' card tables
    // (`gc_core::card_mark`); the other kinds' edges are set when their cells are made.
    let object = core.register_desc(crate::gc_core::LayoutDesc {
        name: "object",
        trace: trace_object_cell,
        mark_trace: Some(mark_object_cell),
        finalize: layout_census::enabled().then_some(census_dead_object as _),
        carded: true,
    });
    // An environment record's slots follow it in its cell, after the whole RefCell.
    crate::interpreter::check_record_layout();
    let scope = core.register_desc(crate::gc_core::LayoutDesc {
        name: "environment",
        trace: trace_scope_cell,
        mark_trace: Some(mark_scope_cell),
        finalize: layout_census::enabled().then_some(census_dead_scope as _),
        carded: true,
    });
    // BigInts and Symbols hold no edges either: marking one queues no trace.
    let bigint = core.register_desc(crate::bigint::cell_layout());
    core.set_leaf(bigint);
    let symbol = core.register_desc(symbol_cell_layout());
    core.set_leaf(symbol);
    let string = core.register_desc(crate::lstr::cell_layout());
    let user_fn = core.register_desc(crate::gc_core::LayoutDesc {
        name: "closure",
        trace: trace_user_fn_cell,
        mark_trace: Some(mark_user_fn_cell),
        // A closure owns nothing to drop (`UserCallable`).
        finalize: None,
        // A closure's environment is set when the closure is made; a later cache sets the card.
        carded: true,
    });
    let lazy_proto = core.register_desc(crate::gc_core::LayoutDesc {
        name: "lazy prototype",
        trace: lazy_function_prototype::trace_cell,
        mark_trace: None,
        finalize: None,
        // Materialization writes the cell's state (`LazyFunctionPrototype::materialize`).
        carded: true,
    });
    // A property layout's edges are its storage blocks (its keys' atoms live as long as the
    // heap). It owns nothing else, so it needs no finalizer; a private layout gains new blocks
    // as its keys change, which its card records (`PropertyLayout::get_mut_private`).
    let property_layout = core.register_desc(crate::gc_core::LayoutDesc {
        name: "property layout",
        trace: trace_property_layout_cell,
        mark_trace: None,
        finalize: None,
        carded: true,
    });
    (
        core,
        CellDescs {
            payloads,
            storage,
            accessor,
            object,
            scope,
            bigint,
            symbol,
            string,
            user_fn,
            lazy_proto,
            property_layout,
        },
    )
}

impl crate::gc_box::Payload for UserCallable {
    fn desc(heap: &GcState) -> crate::gc_core::DescId {
        heap.user_fn_desc
    }
}

impl crate::gc_box::Payload for LazyFunctionPrototype {
    fn desc(heap: &GcState) -> crate::gc_core::DescId {
        heap.lazy_proto_desc
    }
}

/// The marker's trace of a closure ([`LayoutDesc::mark_trace`]): its edges, and the record
/// that the major collection marking found a closure of its function alive, for the closure
/// registry ([`mark_closure_function`]). Every other trace (the collector's checks among them)
/// uses [`trace_user_fn_cell`], which records nothing.
///
/// [`LayoutDesc::mark_trace`]: crate::gc_core::LayoutDesc::mark_trace
fn mark_user_fn_cell(
    cell: *const u8,
    _purpose: crate::gc_core::TracePurpose,
    sink: &mut crate::gc_core::MarkSink<'_>,
) {
    // SAFETY: as in `trace_user_fn_cell`.
    let user = unsafe { &*cell.cast::<UserCallable>() };
    mark_closure_function(&user.func);
    trace_user_fn(user, sink);
}

/// A closure's environment and immutable runtime source-record edges.
fn trace_user_fn_cell(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
    // SAFETY: closure cells are initialized by `gc_box::new` before any collection.
    let user = unsafe { &*cell.cast::<UserCallable>() };
    trace_user_fn(user, &mut crate::gc_trace::VisitSink(visit));
}

/// [`trace_user_fn_cell`]'s edges, for any sink.
#[inline]
fn trace_user_fn<S: crate::gc_trace::TraceSink + ?Sized>(user: &UserCallable, sink: &mut S) {
    sink.env(&user.env);
    let metadata = user.cache.take();
    if let Some(metadata) = &metadata {
        crate::gc_storage::PayloadBox::trace(metadata, sink);
    }
    let origin = match metadata.as_ref().filter(|metadata| metadata.own_origin) {
        Some(metadata) => metadata.origin.as_ref(),
        None if !user.func.self_hosted => user.func.shared_origin(),
        None => None,
    };
    if let Some(origin) = origin {
        origin.trace_edges(sink);
    }
    user.cache.set(metadata);
}

/// `desc`, which the active heap keeps from now on for as long as it lives, as its environments
/// name it: they share it without owning it ([`crate::interpreter::DescRef`]). A compiled scope
/// keeps its descriptor once, when it is made.
pub(crate) fn keep_scope_desc(
    desc: &Rc<crate::interpreter::ScopeDesc>,
) -> crate::interpreter::DescRef {
    with_agent_heap(|heap| heap.scope_descs.borrow_mut().push(desc.clone()));
    // SAFETY: the active heap keeps the descriptor from now on.
    unsafe { crate::interpreter::DescRef::kept(desc) }
}

/// The active heap's descriptor of a tree-walked function's activation layout, made from the
/// layout's names on first use and kept, with the layout, for as long as the heap lives.
pub(crate) fn scope_desc_for(
    layout: &Rc<crate::interpreter::BindingLayout>,
) -> crate::interpreter::DescRef {
    with_agent_heap(|heap| {
        let key = Rc::as_ptr(layout) as usize;
        if let Some((_, desc)) = heap.layout_descs.borrow().get(&key) {
            // SAFETY: the heap keeps the descriptor.
            return unsafe { crate::interpreter::DescRef::kept(desc) };
        }
        let desc = crate::interpreter::ScopeDesc::of_layout(layout);
        // SAFETY: the heap keeps the descriptor from now on.
        let kept = unsafe { crate::interpreter::DescRef::kept(&desc) };
        heap.layout_descs
            .borrow_mut()
            .insert(key, (layout.clone(), desc));
        kept
    })
}

/// `f` on the active Agent heap's atom table (see [`crate::atom`]).
#[inline]
pub(crate) fn with_atoms<R>(f: impl FnOnce(&mut crate::atom::AtomTable) -> R) -> R {
    with_agent_heap(|heap| f(&mut heap.atoms.borrow_mut()))
}

/// A fresh string cell of `size` bytes in the active Agent's heap (see `LStr`).
#[inline]
pub(crate) fn alloc_string_cell(size: usize) -> *mut u8 {
    with_agent_heap(|heap| heap.core.borrow_mut().alloc(heap.string_desc, size))
}

/// The active heap's interned string at `slot`, made by `make` on first use. Engine
/// strings are immutable values; interning per heap keeps one heap's cells out of another's.
pub(crate) fn interned_string(
    slot: usize,
    make: impl FnOnce() -> crate::lstr::LStr,
) -> crate::lstr::LStr {
    with_active_gc_heap(|heap| {
        if let Some(Some(string)) = heap.interned_strings.borrow().get(slot) {
            return *string;
        }
        let string = make();
        let mut interned = heap.interned_strings.borrow_mut();
        if interned.len() <= slot {
            interned.resize(slot + 1, None);
        }
        interned[slot] = Some(string);
        string
    })
}

/// Report this heap's permanent caches and actual live code/template owners, all of them, as a
/// major collection traces them (debug checks and tests; collections call
/// [`trace_heap_roots_for`]).
#[cfg(any(test, debug_assertions))]
pub(crate) fn trace_heap_roots(heap: &GcState, sink: &mut dyn crate::gc_trace::GcSink) {
    trace_fixed_heap_roots(heap, sink);
    trace_code_roots(heap, sink);
}

/// The heap's roots other than its code roots.
#[cfg(any(test, debug_assertions))]
fn trace_fixed_heap_roots(heap: &GcState, sink: &mut dyn crate::gc_trace::GcSink) {
    trace_fixed_heap_roots_but_shapes(heap, sink);
    use crate::gc_trace::GcTrace;
    heap.shapes.borrow().layouts.trace_gc(sink);
    for atom in heap.atoms.borrow().atoms() {
        sink.word(atom.cell_word());
    }
}

/// The heap's roots other than its code roots and its shape directory's layouts.
fn trace_fixed_heap_roots_but_shapes(heap: &GcState, sink: &mut dyn crate::gc_trace::GcSink) {
    heap.agent.trace_gc(sink);
    for string in heap.interned_strings.borrow().iter().flatten() {
        sink.word(string.cell_word());
    }
    use crate::gc_trace::GcTrace;
    heap.array_length_layout.borrow().trace_gc(sink);
}

/// Every live code root, dropping the registrations of dead ones.
fn trace_code_roots(heap: &GcState, sink: &mut dyn crate::gc_trace::GcSink) {
    heap.code_roots
        .borrow_mut()
        .retain(|owner| match owner.upgrade() {
            Some(owner) => {
                owner.trace_gc(sink);
                true
            }
            None => false,
        });
}

/// The heap's roots as a collection of `kind` traces them: a major collection every code root,
/// a minor one only those registered since the last collection and the edges stored into
/// older ones since (see `GcState::young_code_roots`). Either way the young lists start over.
pub(crate) fn trace_heap_roots_for(
    heap: &GcState,
    sink: &mut dyn crate::gc_trace::GcSink,
    kind: crate::gc_core::CollectionKind,
) {
    trace_fixed_heap_roots_but_shapes(heap, sink);
    // A minor collection traces the layouts the directory took since the last collection, and
    // the atoms made since.
    heap.shapes.borrow().layouts.trace_roots(kind, sink);
    heap.atoms.borrow_mut().trace_roots(kind, sink);
    let young_roots = std::mem::take(&mut *heap.young_code_roots.borrow_mut());
    let young_words = std::mem::take(&mut *heap.young_code_words.borrow_mut());
    match kind {
        crate::gc_core::CollectionKind::Major => trace_code_roots(heap, sink),
        crate::gc_core::CollectionKind::Minor => {
            for owner in young_roots.iter().filter_map(std::rc::Weak::upgrade) {
                owner.trace_gc(sink);
            }
            for word in young_words {
                sink.word(word);
            }
        }
    }
}

/// Heap verification, after a minor collection's marking: every code root's edges are marked, so
/// the minor collection, which traced only the young code roots and the reported stores, missed
/// no store of a young cell into an older root (see [`remember_code_edges`]).
#[cfg(debug_assertions)]
pub(crate) fn check_code_roots_marked(heap: &GcState, core: &crate::gc_core::Heap) {
    use crate::gc_trace::GcTrace;
    struct Check<'a>(&'a crate::gc_core::Heap);
    impl crate::gc_trace::GcSink for Check<'_> {
        fn word(&mut self, word: u64) {
            assert!(
                self.0.cell_age(word) != Some(crate::gc_core::CellAge::Young),
                "a code root holds the young {} cell {word:#x} unmarked by a minor collection: \
                 a late store into the root was not reported (value::remember_code_edges)",
                self.0.kind_name(word).unwrap_or("?")
            );
        }
    }
    for owner in heap
        .code_roots
        .borrow()
        .iter()
        .filter_map(std::rc::Weak::upgrade)
    {
        owner.trace_gc(&mut Check(core));
    }
}

/// Heap verification, after a minor collection's marking: every layout the shape directory holds is
/// marked, so the minor collection, which traced only the entries written since the last one
/// (`ShapeLayouts::trace_roots`), missed none.
#[cfg(debug_assertions)]
pub(crate) fn check_shape_layouts_marked(heap: &GcState, core: &crate::gc_core::Heap) {
    struct Check<'a>(&'a crate::gc_core::Heap);
    impl crate::gc_trace::GcSink for Check<'_> {
        fn word(&mut self, word: u64) {
            assert!(
                self.0.cell_age(word) != Some(crate::gc_core::CellAge::Young),
                "the shape directory holds the young layout {word:#x} after a minor \
                 collection's marking"
            );
        }
    }
    crate::gc_trace::GcTrace::trace_gc(&heap.shapes.borrow().layouts, &mut Check(core));
}

/// Verify the COW invariant by actual owning-slot identity, independently of repeated collector
/// visits. Only words resolved as this heap's layout cells may have their shared bit read.
#[cfg(debug_assertions)]
pub(crate) fn check_layout_owners(
    heap: &GcState,
    core: &crate::gc_core::Heap,
    trace_roots: impl FnOnce(&mut dyn crate::gc_trace::GcSink),
) {
    use crate::gc_trace::{GcSink, GcTrace};
    struct Owners<'a> {
        core: &'a crate::gc_core::Heap,
        desc: crate::gc_core::DescId,
        slots: crate::fasthash::FastMap<usize, usize>,
    }
    impl GcSink for Owners<'_> {
        fn word(&mut self, _: u64) {}
        fn layout(&mut self, layout: &PropertyLayout) {
            let cell = layout.as_ptr() as usize;
            if !self.core.owns_cell(cell, self.desc) {
                return;
            }
            if layout.is_shared() {
                return;
            }
            let slot = std::ptr::from_ref(layout) as usize;
            if let Some(&previous) = self.slots.get(&cell) {
                assert_eq!(
                    previous, slot,
                    "private property layout {cell:#x} has two owning slots"
                );
            } else {
                self.slots.insert(cell, slot);
            }
        }
    }
    let mut owners = Owners {
        core,
        desc: heap.property_layout_desc,
        slots: Default::default(),
    };
    for cell in core.live_cells(heap.object_desc) {
        // SAFETY: allocated non-zombie cells of this descriptor hold initialized Objects; all
        // engine borrows have ended at the safepoint (I9), as for the ordinary descriptor walk.
        let object = unsafe { (&*(cell as *const RefCell<Object>)).try_borrow_unguarded() }
            .expect("object mutably borrowed during layout verification (I9)");
        object.trace_gc(&mut owners);
    }
    for cell in core.live_cells(heap.lazy_proto_desc) {
        // SAFETY: same initialized-cell and safepoint contract.
        unsafe { &*(cell as *const LazyFunctionPrototype) }.trace_gc(&mut owners);
    }
    trace_heap_roots(heap, &mut owners);
    trace_roots(&mut owners);
}

/// The active Agent heap's external ledger, if a heap is active and this thread's storage is
/// not being torn down: charges come from containers' finalizers too, which a heap's teardown
/// at thread exit runs after the thread's other storage may be gone. Without one, a charge is
/// not recorded (no heap would collect for it).
fn active_ledger() -> Option<Rc<crate::gc_core::Control>> {
    ACTIVE_GC_HEAP
        .try_with(|active| {
            let active = active.try_borrow().ok()?;
            Some(active.as_ref()?.ledger.clone())
        })
        .ok()
        .flatten()
}

/// The bytes one Rust-heap payload container has charged to a heap's external ledger: bytes
/// that a container belonging to one of its cells (or about to) acquired outside the heap, which
/// only a collection of the owner frees and which therefore count toward the collection
/// triggers (see `gc_core::ExternalBytes`). The record keeps the ledger of the heap active when
/// it first charged, so that whatever heap is active when it changes or drops (a payload cell's
/// finalizer runs in its heap's sweep or teardown), the bytes go back to the ledger they came
/// from.
/// [`ExternalCharge::settle`] brings the charge to the container's current size; dropping the
/// record releases it. A clone charges what the original holds again, for a cloned container,
/// until that one settles its own size.
#[derive(Default)]
pub(crate) struct ExternalCharge {
    bytes: usize,
    ledger: Option<Rc<crate::gc_core::Control>>,
}

impl ExternalCharge {
    /// Nothing charged.
    pub(crate) const fn new() -> Self {
        Self {
            bytes: 0,
            ledger: None,
        }
    }

    /// Charge or release the difference between `bytes` and what is charged now.
    #[inline]
    pub(crate) fn settle(&mut self, bytes: usize) {
        if bytes == self.bytes {
            return;
        }
        if self.ledger.is_none() {
            self.ledger = active_ledger();
        }
        let Some(ledger) = &self.ledger else {
            return;
        };
        if bytes > self.bytes {
            ledger.charge_external(bytes - self.bytes);
        } else {
            ledger.release_external(self.bytes - bytes);
        }
        self.bytes = bytes;
    }
}

impl Clone for ExternalCharge {
    fn clone(&self) -> Self {
        if let Some(ledger) = &self.ledger {
            ledger.charge_external(self.bytes);
        }
        Self {
            bytes: self.bytes,
            ledger: self.ledger.clone(),
        }
    }
}

impl Drop for ExternalCharge {
    fn drop(&mut self) {
        if let Some(ledger) = &self.ledger {
            ledger.release_external(self.bytes);
        }
    }
}

/// A storage block of `bytes` bytes in the active Agent's heap (see `gc_storage`), or `None`
/// when the memory of a large one cannot be had.
#[inline]
pub(crate) fn alloc_storage_block(bytes: usize) -> Option<*mut u8> {
    with_agent_heap(|heap| {
        heap.core
            .borrow_mut()
            .try_alloc_uninit(heap.storage_desc, bytes)
    })
}

/// A fresh payload cell of `size` bytes in the active Agent's heap (see `GcBox::new`).
#[inline]
pub(crate) fn alloc_payload_cell(
    desc: fn(&GcState) -> crate::gc_core::DescId,
    size: usize,
) -> *mut u8 {
    with_agent_heap(|heap| heap.core.borrow_mut().alloc(desc(heap), size))
}

/// Strong edges an object holds itself. Edges kept for it in the interpreter's owner-keyed
/// side tables are reported by the interpreter once the object is marked.
fn trace_object_cell(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
    use crate::gc_trace::GcTrace;
    // SAFETY: collections run at safepoints, where no object is mutably borrowed (I9);
    // `try_borrow_unguarded` panics if one is.
    let object = unsafe { (*(cell as *const RefCell<Object>)).try_borrow_unguarded() }
        .expect("object mutably borrowed during GC (I9)");
    object.trace_gc(&mut crate::gc_trace::VisitSink(visit));
}

/// Only Heap::drain invokes this hook. Ordinary descriptor tracing/verifiers do not consume.
fn mark_object_cell(
    cell: *const u8,
    purpose: crate::gc_core::TracePurpose,
    sink: &mut crate::gc_core::MarkSink<'_>,
) {
    // SAFETY: the actual marker runs at the same unborrowed safepoint as trace_object_cell.
    let object = unsafe { (*(cell as *const RefCell<Object>)).try_borrow_unguarded() }
        .expect("object mutably borrowed during GC (I9)");
    mark_object(object, purpose, sink);
}

/// [`mark_object_cell`]'s work on its object, for any sink.
#[inline]
fn mark_object<S: crate::gc_trace::TraceSink + ?Sized>(
    object: &Object,
    purpose: crate::gc_core::TracePurpose,
    sink: &mut S,
) {
    object.trace_gc_fields(sink, Some(purpose));
    // Commit only after ALL element and non-element edges were reported. A panic leaves the
    // words' obligations intact; writes between phases are recorded afterwards.
    object.props.elems.2.finish_mark_trace();
}

/// Strong edges of an environment record.
fn trace_scope_cell(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
    use crate::gc_trace::GcTrace;
    // SAFETY: as for objects (I9).
    let scope =
        unsafe { (*(cell as *const RefCell<crate::interpreter::Scope>)).try_borrow_unguarded() }
            .expect("environment mutably borrowed during GC (I9)");
    scope.trace_gc(&mut crate::gc_trace::VisitSink(visit));
}

/// The marker's trace of an environment record ([`LayoutDesc::mark_trace`]): the same edges as
/// [`trace_scope_cell`]'s.
///
/// [`LayoutDesc::mark_trace`]: crate::gc_core::LayoutDesc::mark_trace
fn mark_scope_cell(
    cell: *const u8,
    _purpose: crate::gc_core::TracePurpose,
    sink: &mut crate::gc_core::MarkSink<'_>,
) {
    // SAFETY: as in `trace_scope_cell`.
    let scope =
        unsafe { (*(cell as *const RefCell<crate::interpreter::Scope>)).try_borrow_unguarded() }
            .expect("environment mutably borrowed during GC (I9)");
    scope.trace_fields(sink);
}

/// An environment record owns nothing outside the heap: its entries spill into storage
/// blocks, a large map lives in a payload cell, and its names are atoms. A dying record is
/// only freed, and its cell has no finalizer but the census's ([`census_dead_scope`]).
const _: () = assert!(!std::mem::needs_drop::<RefCell<crate::interpreter::Scope>>());

/// Record a dying environment in the layout census.
fn census_dead_scope(cell: *mut u8) {
    // SAFETY: a dying cell is initialized and borrowed by no one.
    if let Ok(scope) =
        unsafe { (*(cell as *const RefCell<crate::interpreter::Scope>)).try_borrow_unguarded() }
    {
        layout_census::record_env(layout_census::DEAD, scope);
    }
}

/// Where generated code takes an environment record's cell of `class` in line in the active
/// heap (W2e; W3 item 3's step 7), unless `TRUST_JS_JIT_INLINE_ALLOC=0` turns it off.
pub(crate) fn scope_inline_alloc_site(class: u8) -> Option<crate::gc_core::InlineAllocSite> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("TRUST_JS_JIT_INLINE_ALLOC").as_deref() != Ok("0")) {
        return None;
    }
    with_agent_heap(|heap| heap.core.borrow().inline_alloc_site(heap.scope_desc, class))
}

/// A fresh cell for one environment record in the active heap, of the small size class `class`
/// (the record's header and its in-place slots, see [`crate::interpreter::VarMap`]), not zeroed:
/// its maker writes the header and the counted slots before the next safepoint (`Env::new`,
/// `interpreter::new_record`), and a trace reads no other word.
#[inline]
pub(crate) fn alloc_scope_cell(class: u8) -> *mut RefCell<crate::interpreter::Scope> {
    debug_assert!(
        crate::gc_core::small_class_size(class)
            >= std::mem::size_of::<RefCell<crate::interpreter::Scope>>()
    );
    with_agent_heap(|heap| {
        heap.core
            .borrow_mut()
            .alloc_small_uninit(heap.scope_desc, class)
            .cast()
    })
}

impl crate::gc_trace::GcTrace for Object {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        self.trace_gc_fields(sink, None);
    }
}

impl Object {
    /// A single exhaustive edge inventory for ordinary tracing and both actual marker modes.
    #[inline]
    fn trace_gc_fields<S: crate::gc_trace::TraceSink + ?Sized>(
        &self,
        sink: &mut S,
        purpose: Option<crate::gc_core::TracePurpose>,
    ) {
        use crate::gc_trace::GcTrace;
        let Object { proto, props, kind } = self;
        if let Some(proto) = proto {
            sink.object(proto);
        }
        // The storage blocks the payloads live in, in every trace: they live as long as an owner
        // reports them (`gc_storage`).
        props.trace_storage(sink);
        // The element words' edges, in every trace (the block itself is in trace_storage); a
        // marker's remembered trace reports only those written since its last mark.
        props.elems.2.trace_values(purpose, sink);
        // Named properties and every non-element edge are ALWAYS complete.
        for property in props
            .entries
            .fields
            .iter()
            .chain(self.internal_slots().values())
        {
            property.trace_edges(sink);
        }
        if let Some(layout) = &props.entries.layout {
            sink.layout(layout);
        }
        props.trace_symbol_keys(sink);
        // Each payload cell (`gc_storage::PayloadBox`, `gc_storage::SharedPayload`), with what
        // its value holds.
        use crate::gc_storage::{PayloadBox, SharedPayload};
        match kind {
            // The view's payload cell; its buffer is byte storage.
            ObjectKind::TypedArray(view) => PayloadBox::trace(view, sink),
            ObjectKind::ArrayIterator(state) => {
                PayloadBox::trace(state, sink);
                sink.value(&state.target);
            }
            ObjectKind::BigIntWrap(bigint) => {
                PayloadBox::trace(bigint, sink);
                sink.word(bigint.cell_word());
            }
            ObjectKind::SymWrap(symbol) => sink.word(symbol_identity(symbol) as u64),
            ObjectKind::StrWrap(string) => {
                PayloadBox::trace(string, sink);
                sink.word(string.cell_word());
            }
            ObjectKind::StringIterator(state) => {
                PayloadBox::trace(state, sink);
                state.string.trace_gc(sink.as_dyn());
            }
            ObjectKind::CollectionIterator(state) => {
                PayloadBox::trace(state, sink);
                if let Some(target) = &state.target {
                    sink.object(target);
                }
            }
            ObjectKind::Error(stack) => stack.trace_gc(sink.as_dyn()), // the cell and the frame names
            ObjectKind::Collection(data) => data.trace_gc(sink.as_dyn()), // the cell and the entries
            ObjectKind::Arguments(Some(map)) => {
                crate::gc_storage::StorageBox::trace(map, sink);
                map.trace(sink);
            }
            // The closure cell, which traces its environment.
            ObjectKind::User(user) => sink.word(crate::gc_box::GcBox::as_ptr(user) as u64),
            ObjectKind::Bound(bound) => {
                SharedPayload::trace(bound, sink);
                sink.object(&bound.target);
                sink.value(&bound.this);
                for argument in &bound.args {
                    sink.value(argument);
                }
            }
            ObjectKind::WrappedShadow(shadow) => {
                SharedPayload::trace(shadow, sink);
                sink.object(&shadow.realm);
                sink.value(&shadow.target);
            }
            ObjectKind::WrappedCross(cross) => {
                SharedPayload::trace(cross, sink);
                sink.value(&cross.target);
            }
            ObjectKind::NativeData(native) => native.trace_gc(sink.as_dyn()), // the cell and the captures
            ObjectKind::AccessorGet(key)
            | ObjectKind::AccessorSet(key)
            | ObjectKind::PropGet(key)
            | ObjectKind::PropSet(key)
            | ObjectKind::PrivatePropGet(key)
            | ObjectKind::PrivatePropSet(key) => SharedPayload::trace(key, sink),
            ObjectKind::Plain
            | ObjectKind::Native(_)
            | ObjectKind::Array
            | ObjectKind::BoolWrap(_)
            | ObjectKind::NumWrap(_)
            | ObjectKind::Arguments(None) => {}
            #[cfg(feature = "embed")]
            ObjectKind::ScopedNative(_) => {}
        }
    }
}

impl crate::gc_trace::GcTrace for Props {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        self.trace_storage(sink);
        crate::gc_trace::GcTrace::trace_gc(&self.entries.layout, sink);
        self.elems.2.trace_values(None, sink);
        for property in self.entries.fields.iter() {
            crate::gc_trace::GcTrace::trace_gc(property, sink);
        }
        self.trace_symbol_keys(sink);
    }
}

impl Props {
    /// Report the storage blocks this map's payloads live in (`gc_storage`): the named
    /// properties' spill, the element words and the key index. Every trace of the map's
    /// owner reports them, since nothing else keeps them alive.
    #[inline]
    pub(crate) fn trace_storage<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(spill) = self.entries.fields.spill_ptr() {
            crate::gc_storage::trace_block(spill, sink);
        }
        self.elems.trace_storage(sink);
    }

    /// The Symbols of Symbol-valued keys, which the encoded keys only name by id.
    #[inline]
    pub(crate) fn trace_symbol_keys<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(symbols) = self.elems.symbols() {
            for symbol in symbols.values() {
                sink.word(symbol_identity(symbol) as u64);
            }
        }
    }
}

impl crate::gc_trace::GcTrace for PropertyKey {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let PropertyKey {
            text: _,
            symbol,
            private: _,
        } = self;
        if let Some(symbol) = symbol {
            sink.word(symbol_identity(symbol) as u64);
        }
    }
}

impl crate::gc_trace::GcTrace for Property {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        self.trace_edges(sink);
    }
}

impl Property {
    /// [`GcTrace`](crate::gc_trace::GcTrace)'s edges, for any sink: the value, and an
    /// accessor's pair out of line.
    #[inline(always)]
    fn trace_edges<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        self.packed.trace_gc_edges(sink);
        if self.accessor_cell_word().is_some() {
            self.trace_accessor_edges(sink);
        }
    }

    /// An accessor property's pair: its cell, and its functions, which its trace reports again.
    #[inline(never)]
    fn trace_accessor_edges<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        if let Some(cell) = self.accessor_cell_word() {
            sink.word(cell);
        }
        if let Some(getter) = self.getter() {
            sink.value(getter);
        }
        if let Some(setter) = self.setter() {
            sink.value(setter);
        }
    }
}

impl crate::gc_trace::GcTrace for NativeCallable {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        match &self.body {
            NativeCallableBody::Captured { captures, .. }
            | NativeCallableBody::CapturedClosure { captures, .. } => {
                for capture in captures.iter() {
                    sink.value(capture);
                }
            }
            #[cfg(feature = "embed")]
            NativeCallableBody::ScopedCaptured { captures, .. } => {
                for capture in captures.iter() {
                    sink.value(capture);
                }
            }
            // An opaque closure holds no JavaScript values (Decision 2; S6 step 11b): a
            // closure keeps its values as engine captures.
            NativeCallableBody::Opaque(_) => {}
        }
    }
}

impl Value {
    /// Whether this word is undefined or null. Reads no payload.
    pub(crate) fn is_nullish(&self) -> bool {
        matches!(self.tag(), PACK_UNDEFINED | PACK_NULL)
    }

    /// Whether this word is an object's (a deferred prototype's is not). Reads no payload.
    pub(crate) fn is_object(&self) -> bool {
        self.tag() == PACK_OBJ
    }
}

/// The object finalizer of the layout census (`TRUST_JS_GC_CENSUS`): it records each death.
/// Objects have nothing to drop, so without the census they have no finalizer.
fn census_dead_object(cell: *mut u8) {
    // SAFETY: a dying object is initialized and no borrow may cross a safepoint (I9).
    let object = unsafe { &*cell.cast::<RefCell<Object>>() };
    if let Ok(object) = unsafe { object.try_borrow_unguarded() } {
        layout_census::record(layout_census::DEAD, object);
    }
}

/// The S3 layout census (`TRUST_JS_GC_CENSUS=1`; diagnostic only): how many named
/// properties objects hold, sampled over the live heap after every collection and over objects
/// as they die, by family. It sizes inline property slots (Decision 3). S4 adds environment
/// records: bindings per record and the binding storage beside the cell.
pub(crate) mod layout_census {
    use super::{GcHeap, Object, ObjectKind};
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub(crate) const LIVE: usize = 0;
    pub(crate) const DEAD: usize = 1;
    const FAMILIES: [&str; 4] = ["function", "array", "exotic", "plain"];
    /// Inclusive upper bounds of the named-property buckets; a last bucket takes the rest.
    const BOUNDS: [usize; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 32];
    const BUCKETS: usize = BOUNDS.len() + 1;

    struct Population {
        named: [[AtomicU64; BUCKETS]; FAMILIES.len()],
        named_total: [AtomicU64; FAMILIES.len()],
        named_capacity: [AtomicU64; FAMILIES.len()],
        sidecars: [AtomicU64; FAMILIES.len()],
    }

    static POPULATIONS: [Population; 2] = [const {
        Population {
            named: [const { [const { AtomicU64::new(0) }; BUCKETS] }; FAMILIES.len()],
            named_total: [const { AtomicU64::new(0) }; FAMILIES.len()],
            named_capacity: [const { AtomicU64::new(0) }; FAMILIES.len()],
            sidecars: [const { AtomicU64::new(0) }; FAMILIES.len()],
        }
    }; 2];
    static SAMPLES: AtomicU64 = AtomicU64::new(0);

    /// Variable environments (function bodies, global, eval) and the rest (blocks, catch, with).
    const ENV_KINDS: [&str; 2] = ["var", "block"];

    struct EnvPopulation {
        bindings: [[AtomicU64; BUCKETS]; ENV_KINDS.len()],
        bindings_total: [AtomicU64; ENV_KINDS.len()],
        /// Bytes of binding storage allocated beside the cell.
        storage_bytes: [AtomicU64; ENV_KINDS.len()],
    }

    static ENV_POPULATIONS: [EnvPopulation; 2] = [const {
        EnvPopulation {
            bindings: [const { [const { AtomicU64::new(0) }; BUCKETS] }; ENV_KINDS.len()],
            bindings_total: [const { AtomicU64::new(0) }; ENV_KINDS.len()],
            storage_bytes: [const { AtomicU64::new(0) }; ENV_KINDS.len()],
        }
    }; 2];

    pub(crate) fn record_env(population: usize, scope: &crate::interpreter::Scope) {
        let kind = usize::from(!scope.var_boundary());
        let bindings = scope.vars.iter().count();
        let bucket = BOUNDS.partition_point(|&bound| bound < bindings);
        let counts = &ENV_POPULATIONS[population];
        counts.bindings[kind][bucket].fetch_add(1, Relaxed);
        counts.bindings_total[kind].fetch_add(bindings as u64, Relaxed);
        counts.storage_bytes[kind].fetch_add(
            scope.vars.retained_requested_storage_bytes().0 as u64,
            Relaxed,
        );
    }

    pub(crate) fn enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var_os("TRUST_JS_GC_CENSUS").is_some())
    }

    pub(crate) fn record(population: usize, object: &Object) {
        let family = if object.kind.is_callable() {
            0
        } else if matches!(object.kind, ObjectKind::Array) {
            1
        } else if !object.kind.exotic_is_none() || !object.ic_plain() {
            2
        } else {
            3
        };
        let named = object.props.entries.len();
        let bucket = BOUNDS.partition_point(|&bound| bound < named);
        let counts = &POPULATIONS[population];
        counts.named[family][bucket].fetch_add(1, Relaxed);
        counts.named_total[family].fetch_add(named as u64, Relaxed);
        counts.named_capacity[family]
            .fetch_add(object.props.entries.fields.capacity() as u64, Relaxed);
        // The map's first word: its own key index since W2c step 3 removed the sidecar around
        // it (the column keeps its name).
        if object.props.elems.0.is_some() {
            counts.sidecars[family].fetch_add(1, Relaxed);
        }
    }

    /// Record every live object and environment of `heap`; call right after a collection's
    /// sweep.
    pub(crate) fn sample_live(heap: &GcHeap) {
        let cells = heap.core.borrow().live_cells(heap.object_desc);
        for cell in cells {
            // SAFETY: a live object cell is initialized; safepoints hold no mutable borrow.
            if let Ok(object) =
                unsafe { (*(cell as *const RefCell<Object>)).try_borrow_unguarded() }
            {
                record(LIVE, object);
            }
        }
        let cells = heap.core.borrow().live_cells(heap.scope_desc);
        for cell in cells {
            // SAFETY: as above, for environment cells.
            if let Ok(scope) = unsafe {
                (*(cell as *const RefCell<crate::interpreter::Scope>)).try_borrow_unguarded()
            } {
                record_env(LIVE, scope);
            }
        }
        SAMPLES.fetch_add(1, Relaxed);
    }

    /// The census as a JSON object.
    pub(crate) fn json() -> String {
        let bounds = BOUNDS.map(|b| b.to_string()).join(",");
        let population = |counts: &Population| {
            FAMILIES
                .iter()
                .enumerate()
                .map(|(family, name)| {
                    let histogram = counts.named[family]
                        .iter()
                        .map(|count| count.load(Relaxed).to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    format!(
                        "\"{name}\":{{\"named_histogram\":[{histogram}],\"named_total\":{},\"named_capacity\":{},\"sidecars\":{}}}",
                        counts.named_total[family].load(Relaxed),
                        counts.named_capacity[family].load(Relaxed),
                        counts.sidecars[family].load(Relaxed),
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let environments = |counts: &EnvPopulation| {
            ENV_KINDS
                .iter()
                .enumerate()
                .map(|(kind, name)| {
                    let histogram = counts.bindings[kind]
                        .iter()
                        .map(|count| count.load(Relaxed).to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    format!(
                        "\"{name}\":{{\"bindings_histogram\":[{histogram}],\"bindings_total\":{},\"storage_bytes\":{}}}",
                        counts.bindings_total[kind].load(Relaxed),
                        counts.storage_bytes[kind].load(Relaxed),
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "{{\"live_samples\":{},\"named_bucket_upper_bounds\":[{bounds},null],\"live\":{{{}}},\"dead\":{{{}}},\"environments\":{{\"live\":{{{}}},\"dead\":{{{}}}}}}}",
            SAMPLES.load(Relaxed),
            population(&POPULATIONS[LIVE]),
            population(&POPULATIONS[DEAD]),
            environments(&ENV_POPULATIONS[LIVE]),
            environments(&ENV_POPULATIONS[DEAD]),
        )
    }
}

// S3: the header and its inline named slots fit in the 192 B size class (184 B after S4 narrowed
// the slots' header, 176 B after S6 dropped two unused fields); growing either past it moves
// every object into the 256 B class.
const _: () = assert!(std::mem::size_of::<RefCell<Object>>() <= 192);

/// A fresh cell for one object in `heap` (see `Gc::new_in`).
pub(crate) fn alloc_object_cell(heap: &GcState) -> *mut RefCell<Object> {
    heap.core
        .borrow_mut()
        .alloc(heap.object_desc, std::mem::size_of::<RefCell<Object>>())
        .cast()
}

/// Ends an Agent's object heap: declared as the interpreter's LAST field, so it runs after every
/// other field (anything that might still read an object) has dropped. Makes the heap's rooted
/// handles inert, finalizes every remaining object, clears weak handles into the heap and frees
/// its memory.
pub(crate) struct HeapRelease(Option<GcHeap>);

impl HeapRelease {
    /// The heap's owner: the interpreter that collects it and releases it when dropped.
    pub(crate) fn new(heap: &GcHeap) -> Self {
        Self(Some(heap.clone()))
    }

    /// A ShadowRealm interpreter's: the heap belongs to the root of its ShadowRealm tree.
    pub(crate) fn shared() -> Self {
        Self(None)
    }

    pub(crate) fn owns_heap(&self) -> bool {
        self.0.is_some()
    }
}

impl Drop for HeapRelease {
    fn drop(&mut self) {
        let Some(heap) = &self.0 else {
            return;
        };
        // S6 step 11a: before the cells go, every handle still held becomes inert.
        heap.handles.release();
        // S6 step 11b: and every Realm entered and not left is forgotten.
        #[cfg(feature = "embed")]
        heap.entered_realms.borrow_mut().clear();
        {
            let core = heap.core.borrow();
            let _ =
                crate::gc_handle::clear_dead_weak_cells(|ptr| core.resolve(ptr as u64).is_some());
        }
        // Notify foreign observing heaps before these addresses can be reused. Notifications
        // enqueue identities only; cleanup callbacks still wait for a host job checkpoint.
        for (_, observers) in heap.weak_observers.borrow_mut().drain() {
            observers.notify();
        }
        #[cfg(feature = "architecture-diagnostics")]
        {
            let cells = heap.core.borrow().live_cells(heap.object_desc);
            let mut diagnostics = heap.diagnostics.borrow_mut();
            for cell in cells {
                diagnostics.died(cell);
            }
        }
        heap.core.borrow_mut().release_all();
        // Do not keep even a Weak allocation for a retired Engine until the next Agent switch.
        KNOWN_GC_HEAPS.with(|known| {
            known
                .borrow_mut()
                .retain(|entry| entry.strong_count() != 0 && entry.as_ptr() != Rc::as_ptr(heap));
        });
    }
}

/// Evict object death metadata while every dead cell is still intact. Address keys disappear
/// before freeing, including when weak pins reserve finalized cells as zombies.
pub(crate) fn sweep_object_metadata(heap: &GcHeap, dead: &crate::gc_sweep::DeadObjects<'_>) {
    #[cfg(feature = "architecture-diagnostics")]
    {
        let mut diagnostics = heap.diagnostics.borrow_mut();
        for cell in dead.iter() {
            diagnostics.died(cell);
        }
    }
    let mut watched = heap.weak_observers.borrow_mut();
    if watched.is_empty() || dead.is_empty() {
        return;
    }
    if dead.len() < watched.capacity() {
        for cell in dead.iter() {
            if let Some(observers) = watched.remove(&cell) {
                observers.notify();
            }
        }
    } else {
        let cells: Vec<_> = watched
            .keys()
            .copied()
            .filter(|&cell| dead.contains(cell))
            .collect();
        for cell in cells {
            watched.remove(&cell).unwrap().notify();
        }
    }
    crate::weak_metadata::shrink_map(&mut watched);
}

/// Sparse destruction watches, owned by the target's heap. A foreign observing interpreter may
/// have another heap active; locate the actual owner from heap metadata without an object field.
pub(crate) fn observe_weak_target(
    target: &crate::interpreter::WeakTarget,
    queue: &Rc<crate::weak_metadata::DeathQueue>,
) {
    use crate::weak_metadata::DeathObservers;
    match target.upgrade().map(|value| value.view()) {
        Some(ValueView::Obj(object)) => {
            let cell = Gc::as_ptr(&object) as usize;
            object_heap(cell)
                .weak_observers
                .borrow_mut()
                .entry(cell)
                .or_insert_with(|| DeathObservers::new(target.clone()))
                .subscribe(queue);
        }
        Some(ValueView::Sym(symbol)) => {
            symbol
                .weak_observers
                .borrow_mut()
                .get_or_insert_with(|| Box::new(DeathObservers::new(target.clone())))
                .subscribe(queue);
        }
        None => {
            queue.enqueue(target.clone());
        }
        _ => unreachable!("weak target can only be an object or symbol"),
    }
}

pub(crate) fn unobserve_weak_target(
    target: &crate::interpreter::WeakTarget,
    queue: &Rc<crate::weak_metadata::DeathQueue>,
) {
    match (target.upgrade()).map(|__value| (__value).view()) {
        Some(ValueView::Obj(object)) => {
            let cell = Gc::as_ptr(&object) as usize;
            let heap = object_heap(cell);
            let mut watched = heap.weak_observers.borrow_mut();
            if watched
                .get_mut(&cell)
                .is_some_and(|observers| observers.unsubscribe(queue))
            {
                watched.remove(&cell);
                crate::weak_metadata::shrink_map(&mut watched);
            }
        }
        Some(ValueView::Sym(symbol)) => {
            let mut watched = symbol.weak_observers.borrow_mut();
            if watched
                .as_mut()
                .is_some_and(|observers| observers.unsubscribe(queue))
            {
                *watched = None;
            }
        }
        // Destruction already moved these subscriptions into their independent queues.
        _ => {}
    }
}

pub(crate) fn scan_gc_heap_retained_memory(
    heap: &GcHeap,
    visitor: &mut crate::memory::Visitor,
) -> (usize, bool) {
    let mut bytes = std::mem::size_of::<GcState>();
    {
        let observed = heap.weak_observers.borrow();
        bytes = bytes.saturating_add(
            observed.capacity()
                * std::mem::size_of::<(usize, crate::weak_metadata::DeathObservers)>(),
        );
        bytes = bytes.saturating_add(
            observed
                .values()
                .map(crate::weak_metadata::DeathObservers::allocated_bytes)
                .sum::<usize>(),
        );
    }
    bytes = bytes.saturating_add(
        heap.function_runtimes.borrow().capacity()
            * std::mem::size_of::<(u64, std::rc::Weak<crate::function_runtime::FunctionRuntime>)>(),
    );
    let shapes = heap.shapes.borrow();
    bytes = bytes.saturating_add(
        shapes
            .transitions
            .len()
            .saturating_mul(std::mem::size_of::<((u32, PropKey, PropClass), u32)>()),
    );
    bytes = bytes.saturating_add(shapes.layouts.allocated_bytes());
    bytes = bytes.saturating_add(
        shapes
            .recent
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<(u32, PropKey, PropClass, u32)>>()),
    );
    for layout in shapes.layouts.iter() {
        visitor.property_layout(layout);
    }
    if let Some(layout) = &*heap.array_length_layout.borrow() {
        visitor.property_layout(layout);
    }
    bytes = bytes.saturating_add(
        heap.code_roots.borrow().capacity()
            * std::mem::size_of::<std::rc::Weak<dyn crate::gc_trace::GcTrace>>(),
    );
    (bytes, shapes.transitions.is_empty())
}

// Process-wide collector diagnostics for the benchmark shell. Collection is already a global
// safepoint for an Agent, so relaxed aggregate counters are sufficient. The environment check and
// clock read occur only at a collection entry, never on the allocation or property-access paths.
const GC_PAUSE_BUCKET_UPPER_NANOS: [u64; 15] = [
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    1_000_000_000,
    2_500_000_000,
];
static GC_PERF_METRICS_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static GC_COLLECTIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GC_PAUSE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GC_MAX_PAUSE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GC_PAUSE_BUCKET_COUNTS: [std::sync::atomic::AtomicU64; 16] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 16];
static GC_CAUSE_COUNTS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// `GC_PAUSE_NANOS` by cause, so that a measurement can leave out a kind of collection (the
/// shell's exit collection is `Explicit`).
static GC_CAUSE_PAUSE_NANOS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Collections by kind (`gc_core::CollectionKind`: minor, major), and their pause time.
static GC_KIND_COUNTS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static GC_KIND_PAUSE_NANOS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];
static GC_OBJECTS_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GC_OBJECTS_RECLAIMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GC_PEAK_OBJECTS_BEFORE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GC_LAST_OBJECTS_AFTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Internal collection trigger for diagnostics. Collection scheduling is not observable
/// ECMAScript behavior; retaining the reason lets tuning distinguish allocation pressure from a
/// host task boundary or an explicit embedder request without changing that scheduling contract.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GcCause {
    AllocationThreshold = 0,
    TaskBoundary = 1,
    Explicit = 2,
}

#[inline]
pub(crate) fn gc_performance_metrics_start() -> Option<std::time::Instant> {
    let enabled = *GC_PERF_METRICS_ENABLED
        .get_or_init(|| std::env::var_os("TRUST_JS_PERF_METRICS").is_some());
    enabled.then(std::time::Instant::now)
}

#[inline]
fn saturating_u64(value: usize) -> u64 {
    value.try_into().unwrap_or(u64::MAX)
}

pub(crate) fn gc_performance_metrics_finish(
    started: std::time::Instant,
    objects_before: usize,
    objects_after: i64,
    cause: GcCause,
    kind: crate::gc_core::CollectionKind,
) {
    use std::sync::atomic::Ordering::Relaxed;

    let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let objects_before = saturating_u64(objects_before);
    let objects_after = u64::try_from(objects_after).unwrap_or(0);
    let bucket = GC_PAUSE_BUCKET_UPPER_NANOS.partition_point(|upper| elapsed > *upper);

    GC_COLLECTIONS.fetch_add(1, Relaxed);
    GC_CAUSE_COUNTS[cause as usize].fetch_add(1, Relaxed);
    GC_KIND_COUNTS[kind as usize].fetch_add(1, Relaxed);
    GC_PAUSE_NANOS.fetch_add(elapsed, Relaxed);
    GC_CAUSE_PAUSE_NANOS[cause as usize].fetch_add(elapsed, Relaxed);
    GC_KIND_PAUSE_NANOS[kind as usize].fetch_add(elapsed, Relaxed);
    GC_MAX_PAUSE_NANOS.fetch_max(elapsed, Relaxed);
    GC_PAUSE_BUCKET_COUNTS[bucket].fetch_add(1, Relaxed);
    GC_OBJECTS_SEEN.fetch_add(objects_before, Relaxed);
    GC_OBJECTS_RECLAIMED.fetch_add(objects_before.saturating_sub(objects_after), Relaxed);
    GC_PEAK_OBJECTS_BEFORE.fetch_max(objects_before, Relaxed);
    GC_LAST_OBJECTS_AFTER.store(objects_after, Relaxed);
}

/// The S3 layout census as a JSON field, when it was requested.
fn layout_census_field() -> String {
    if layout_census::enabled() {
        return format!(",\"gc_layout_census\":{}", layout_census::json());
    }
    String::new()
}

/// JSON fields appended to the unstable process-level performance record. Object and scope counts
/// are exact graph-node populations at collector boundaries; they intentionally are not described
/// as bytes because strings, buffers, side tables, and shared allocations need separate accounting.
pub(crate) fn gc_performance_metrics_json_fields() -> String {
    use std::sync::atomic::Ordering::Relaxed;

    let counts = GC_PAUSE_BUCKET_COUNTS
        .iter()
        .map(|count| count.load(Relaxed).to_string())
        .collect::<Vec<_>>()
        .join(",");
    let upper_bounds = GC_PAUSE_BUCKET_UPPER_NANOS
        .iter()
        .map(u64::to_string)
        .chain(std::iter::once("null".to_owned()))
        .collect::<Vec<_>>()
        .join(",");
    let pause_nanos = GC_PAUSE_NANOS.load(Relaxed);
    let max_pause_nanos = GC_MAX_PAUSE_NANOS.load(Relaxed);
    let cause_pause_seconds = |cause: GcCause| {
        GC_CAUSE_PAUSE_NANOS[cause as usize].load(Relaxed) as f64 / 1_000_000_000.0
    };
    use crate::gc_core::CollectionKind::{Major, Minor};
    let kind_pause_seconds = |kind: crate::gc_core::CollectionKind| {
        GC_KIND_PAUSE_NANOS[kind as usize].load(Relaxed) as f64 / 1_000_000_000.0
    };
    format!(
        "\"gc_collections\":{},\"gc_causes\":{{\"allocation_threshold\":{},\"task_boundary\":{},\"explicit\":{}}},\"gc_kinds\":{{\"minor\":{},\"major\":{}}},\"gc_pause_seconds\":{:.9},\"gc_pause_seconds_by_cause\":{{\"allocation_threshold\":{:.9},\"task_boundary\":{:.9},\"explicit\":{:.9}}},\"gc_pause_seconds_by_kind\":{{\"minor\":{:.9},\"major\":{:.9}}},\"gc_max_pause_seconds\":{:.9},\"gc_pause_histogram\":{{\"unit\":\"nanoseconds\",\"upper_bounds\":[{upper_bounds}],\"counts\":[{counts}]}},\"gc_objects_seen\":{},\"gc_objects_reclaimed\":{},\"gc_peak_objects_before\":{},\"gc_last_objects_after\":{}{}",
        GC_COLLECTIONS.load(Relaxed),
        GC_CAUSE_COUNTS[GcCause::AllocationThreshold as usize].load(Relaxed),
        GC_CAUSE_COUNTS[GcCause::TaskBoundary as usize].load(Relaxed),
        GC_CAUSE_COUNTS[GcCause::Explicit as usize].load(Relaxed),
        GC_KIND_COUNTS[Minor as usize].load(Relaxed),
        GC_KIND_COUNTS[Major as usize].load(Relaxed),
        pause_nanos as f64 / 1_000_000_000.0,
        cause_pause_seconds(GcCause::AllocationThreshold),
        cause_pause_seconds(GcCause::TaskBoundary),
        cause_pause_seconds(GcCause::Explicit),
        kind_pause_seconds(Minor),
        kind_pause_seconds(Major),
        max_pause_nanos as f64 / 1_000_000_000.0,
        GC_OBJECTS_SEEN.load(Relaxed),
        GC_OBJECTS_RECLAIMED.load(Relaxed),
        GC_PEAK_OBJECTS_BEFORE.load(Relaxed),
        GC_LAST_OBJECTS_AFTER.load(Relaxed),
        layout_census_field(),
    )
}

thread_local! {
    /// Heap ownership lookup for the sparse object weak-observer paths. Weak references never
    /// retain an Engine; creation and actual Agent switches prune retired heaps. A coroutine
    /// worker learns the shared heap when it activates it, without enlarging Object or WeakGc.
    static KNOWN_GC_HEAPS: RefCell<Vec<std::rc::Weak<GcState>>> = const { RefCell::new(Vec::new()) };
    /// The ECMAScript Agent surrounding the code currently running on this native thread. This is
    /// the allocation heap: `Interp` restores it whenever execution moves between the driver and a coroutine
    /// worker. See ECMA-262, Agents and GeneratorResume/RunSuspendedContext.
    static ACTIVE_GC_HEAP: ActiveHeapCell = const { ActiveHeapCell(RefCell::new(None)) };
    /// Symbol identity/registry state for the surrounding Agent. It is separate from `GcHeap`
    /// because TRust JS currently isolates a ShadowRealm's object heap while the specification still
    /// requires the ShadowRealm to share its Agent's Symbols.
    static ACTIVE_SYMBOL_AGENT: RefCell<Option<SymbolAgent>> = const { RefCell::new(None) };
    /// Fast identity of the surrounding Agent implementation currently installed in the two
    /// owning slots above. Ordinary calls stay within one Agent (ECMA-262 §9.6), so their entry
    /// check must not repeatedly borrow two TLS RefCells and compare/clone two `Rc`s. A different
    /// Realm implementation can share Symbols while owning a distinct heap, hence both pointers
    /// participate in the tag.
    static ACTIVE_AGENT_TAG: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    #[cfg(test)]
    static ACTIVE_AGENT_SLOW_SWITCHES: Cell<u64> = const { Cell::new(0) };
}

pub(crate) fn new_gc_heap() -> GcHeap {
    static NEXT_HEAP_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let (core, descs) = new_cell_heap();
    let ledger = core.control();
    let heap_id = NEXT_HEAP_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let heap = Rc::new(GcState {
        agent: crate::agent_control::AgentControl::new(heap_id),
        heap_id,
        #[cfg(feature = "architecture-diagnostics")]
        diagnostics: RefCell::new(allocation_diagnostics::State::default()),
        weak_observers: RefCell::new(Default::default()),
        shapes: RefCell::new(ShapeTable::new()),
        function_runtimes: RefCell::default(),
        closure_functions: RefCell::default(),
        young_closure_functions: RefCell::default(),
        closure_functions_judged: Cell::new(0),
        array_length_shape: Cell::new(SHAPE_EMPTY),
        array_length_layout: RefCell::default(),
        live: LiveCount::new(0),
        allocated: Cell::new(0),
        core: RefCell::new(core),
        ledger,
        object_desc: descs.object,
        storage_desc: descs.storage,
        payload_descs: descs.payloads,
        accessor_desc: descs.accessor,
        scope_desc: descs.scope,
        bigint_desc: descs.bigint,
        symbol_desc: descs.symbol,
        string_desc: descs.string,
        user_fn_desc: descs.user_fn,
        lazy_proto_desc: descs.lazy_proto,
        property_layout_desc: descs.property_layout,
        interned_strings: RefCell::default(),
        atoms: RefCell::default(),
        scope_descs: RefCell::default(),
        layout_descs: RefCell::default(),
        code_roots: RefCell::default(),
        young_code_roots: RefCell::default(),
        young_code_words: RefCell::default(),
        #[cfg(feature = "embed")]
        entered_realms: RefCell::default(),
        records: RefCell::default(),
        handles: crate::handles::HandleTable::new(heap_id),
        #[cfg(test)]
        client_report: RefCell::default(),
    });
    {
        let mut core = heap.core.borrow_mut();
        // SAFETY: the count lives in this heap's `GcState`, which the `Rc` keeps at one address
        // for as long as the core, one of its fields, lives; nothing but the core writes it.
        unsafe {
            core.attach_live_count(heap.object_desc, std::ptr::NonNull::from(&heap.live.count))
        };
        core.set_live_limit(heap.agent.next.get());
    }
    remember_gc_heap(&heap);
    heap
}

fn remember_gc_heap(heap: &GcHeap) {
    KNOWN_GC_HEAPS.with(|known| {
        let mut known = known.borrow_mut();
        known.retain(|heap| heap.strong_count() != 0);
        if !known.iter().any(|entry| entry.as_ptr() == Rc::as_ptr(heap)) {
            known.push(Rc::downgrade(heap));
        }
    });
}

pub(crate) fn object_heap(cell: usize) -> GcHeap {
    let owns = |heap: &GcHeap| match heap.core.try_borrow() {
        Ok(core) => core.owns_cell(cell, heap.object_desc),
        // A collected native payload can drop a foreign observing Engine while this core is
        // borrowed by sweep. Its existing observer entry already proves ownership: insertion
        // checked the descriptor, and eviction removes the address before freeing the cell.
        // This path only removes a subscription; sweep callbacks cannot run JS or create one.
        Err(_) => heap.weak_observers.borrow().contains_key(&cell),
    };
    let active = active_gc_heap();
    if owns(&active) {
        return active;
    }
    // A native payload's destructor can retire a nested Engine while an unrelated active heap
    // is sweeping under a mutable core borrow. Skip that unrelated candidate; never borrow the
    // heap list while releasing candidate Rcs (their destructors can retire another heap).
    let candidates = KNOWN_GC_HEAPS.with(|known| {
        let mut known = known.borrow_mut();
        known.retain(|heap| heap.strong_count() != 0);
        known
            .iter()
            .filter_map(std::rc::Weak::upgrade)
            .collect::<Vec<_>>()
    });
    candidates
        .into_iter()
        .find(owns)
        .expect("a live weak target belongs to a known Agent heap")
}

pub(crate) fn heap_id(heap: &GcHeap) -> u64 {
    heap.heap_id
}

impl GcState {
    /// How many markings of the heap have begun (`gc_core::Control::markings`).
    #[inline]
    pub(crate) fn markings(&self) -> u64 {
        self.ledger.markings()
    }

    /// Whether `word` names a cell of this heap allocated since the last collection.
    pub(crate) fn is_young(&self, word: u64) -> bool {
        self.core.borrow().cell_age(word) == Some(crate::gc_core::CellAge::Young)
    }
}

/// The active Agent heap's identity (never reused).
#[inline]
pub(crate) fn active_heap_id() -> u64 {
    with_agent_heap(|heap| heap.heap_id)
}

/// Heap identity and an inert lifetime token. Parsed code may outlive its first Engine;
/// this token can be checked without reading any cell or retaining the heap.
#[derive(Clone, Debug)]
pub(crate) struct HeapLifetime {
    pub(crate) id: u64,
    handles: std::rc::Weak<crate::handles::HandleTable>,
}

impl HeapLifetime {
    pub(crate) fn is_alive(&self) -> bool {
        self.handles
            .upgrade()
            .is_some_and(|handles| handles.is_alive())
    }
}

pub(crate) fn active_heap_lifetime() -> HeapLifetime {
    with_active_gc_heap(|heap| HeapLifetime {
        id: heap.heap_id,
        handles: Rc::downgrade(&heap.handles),
    })
}

/// Recover an evicted foreign entry while a live closure pins it. Never-reused cache identities
/// avoid pointer ABA; the index is weak and pruned on cold insertion and after major GC.
pub(crate) fn foreign_function_runtime(cache: u64) -> Rc<crate::function_runtime::FunctionRuntime> {
    with_active_gc_heap(|heap| {
        let mut entries = heap.function_runtimes.borrow_mut();
        if let Some(entry) = entries.get(&cache).and_then(std::rc::Weak::upgrade) {
            return entry;
        }
        if entries.len().is_multiple_of(64) {
            entries.retain(|_, entry| entry.strong_count() != 0);
        }
        let entry = Rc::new(crate::function_runtime::FunctionRuntime::new());
        entries.insert(cache, Rc::downgrade(&entry));
        entry
    })
}

/// Make the active heap's closure registry hold `func` for a closure being made of it (see
/// [`ClosureFunction`]).
pub(crate) fn hold_closure_function(func: &Rc<Function>) {
    let heap_id = active_heap_id();
    if func.runtime_cache().closure_heap.get() != heap_id {
        hold_closure_function_in(func, heap_id);
    }
}

/// [`hold_closure_function`] for the active heap, `heap_id`, which does not hold it yet as far
/// as the function's cache knows: the registry may still hold it if another heap made a
/// closure of the function since.
#[cold]
fn hold_closure_function_in(func: &Rc<Function>, heap_id: u64) {
    with_active_gc_heap(|heap| {
        debug_assert_eq!(heap.heap_id, heap_id);
        let key = Rc::as_ptr(func) as usize;
        if let std::collections::hash_map::Entry::Vacant(entry) =
            heap.closure_functions.borrow_mut().entry(key)
        {
            entry.insert(func.clone());
            // A closure made while a collection finishes is allocated marked, old once it ends:
            // only a major collection judges its function.
            if !heap.ledger.collecting() {
                heap.young_closure_functions.borrow_mut().push(key);
            }
        }
    });
    func.runtime_cache().closure_heap.set(heap_id);
}

thread_local! {
    /// While a collection marks on this thread, its stamp ([`closure_mark_stamp`]); otherwise
    /// 0.
    static CLOSURE_MARK_STAMP: Cell<u64> = const { Cell::new(0) };
    static LAST_CLOSURE_MARK_STAMP: Cell<u64> = const { Cell::new(0) };
}

/// Mark `heap` for a collection of `kind` ([`crate::gc_core::Heap::start_marking`]) and let
/// its closure registry go of the functions the marking found no live closure of (see
/// [`ClosureFunction`]). Every marking of an Agent heap goes through here: one that did not
/// would make closures old that the registry takes for young.
pub(crate) fn mark_agent_heap(
    heap: &GcState,
    native_roots: bool,
    client: &dyn crate::gc_core::MarkClient,
    kind: crate::gc_core::CollectionKind,
) -> bool {
    let stamp = closure_mark_stamp();
    let marked = heap
        .core
        .borrow_mut()
        .start_marking(native_roots, client, kind);
    sweep_closure_functions(heap, stamp, marked, kind);
    marked
}

/// Start a collection's marking of closures: the marker's trace of a closure records the
/// returned stamp in its function's cache, unique among this thread's collections (a function
/// is used only on its thread). [`sweep_closure_functions`] ends it.
fn closure_mark_stamp() -> u64 {
    let stamp = LAST_CLOSURE_MARK_STAMP.with(|last| {
        last.set(last.get() + 1);
        last.get()
    });
    CLOSURE_MARK_STAMP.with(|current| current.set(stamp));
    stamp
}

/// End the marking that [`closure_mark_stamp`] started. Once it has completed (`marked`), let
/// go of each function of the registry that it found no live closure of, of those it can
/// judge: every function after a major collection, the young ones after a minor collection
/// (see `GcState::young_closure_functions`). No closure of such a function is left in the heap
/// but dead ones, which hold nothing (see [`ClosureFunction`]). Runs as the marking ends,
/// before any closure can be made again.
fn sweep_closure_functions(
    heap: &GcState,
    stamp: u64,
    marked: bool,
    kind: crate::gc_core::CollectionKind,
) {
    CLOSURE_MARK_STAMP.with(|current| current.set(0));
    if !marked {
        return;
    }
    // The young functions' closures were made since the registry was last judged: all young
    // at this marking if it is the next one. A marking that bypassed `mark_agent_heap` made
    // them old, and only a major collection can judge them now.
    let markings = heap.ledger.markings();
    let young_judged = heap.closure_functions_judged.replace(markings) + 1 == markings;
    debug_assert!(
        young_judged,
        "an Agent heap marked without `mark_agent_heap`"
    );
    let mut released: Vec<Rc<Function>> = Vec::new();
    let mut release = |func: Rc<Function>| {
        let cache = func.runtime_cache();
        if cache.closure_heap.get() == heap.heap_id {
            cache.closure_heap.set(0);
        }
        released.push(func);
    };
    let traced = |func: &Rc<Function>| func.runtime_cache().closure_mark.get() == stamp;
    {
        let mut registry = heap.closure_functions.borrow_mut();
        let mut young = heap.young_closure_functions.borrow_mut();
        match kind {
            crate::gc_core::CollectionKind::Minor => {
                for key in young.drain(..) {
                    if young_judged && registry.get(&key).is_some_and(|func| !traced(func)) {
                        release(registry.remove(&key).expect("the young function is held"));
                    }
                }
            }
            crate::gc_core::CollectionKind::Major => {
                young.clear();
                registry.retain(|_, func| {
                    if traced(func) {
                        return true;
                    }
                    release(func.clone());
                    false
                });
            }
        }
    }
    // A function's drop can reach code caches; release them outside the registry's borrow.
    drop(released);
}

/// How many functions `heap`'s closure registry holds (see [`ClosureFunction`]).
#[cfg(test)]
pub(crate) fn closure_function_count(heap: &GcState) -> usize {
    heap.closure_functions.borrow().len()
}

/// Record, for the registry sweep, that the collection marking traced a closure of `func`.
#[inline]
fn mark_closure_function(func: &Function) {
    let stamp = CLOSURE_MARK_STAMP.with(Cell::get);
    if stamp != 0 {
        if let Some(cache) = func.runtime.get() {
            cache.closure_mark.set(stamp);
        }
    }
}

/// Major-only maintenance runs after payload finalizers released closure/runtime leases.
/// Minors leave this index alone; cold misses still prune between major collections.
pub(crate) fn prune_function_runtimes_after_major(heap: &GcState) {
    let mut entries = heap.function_runtimes.borrow_mut();
    entries.retain(|_, entry| entry.strong_count() != 0);
    // Preserve a modest floor and 2x live headroom; shrink only beyond 4x underfill so a
    // fluctuating set of foreign functions does not repeatedly resize the index.
    let capacity = entries.capacity();
    if capacity > 128 && entries.len().saturating_mul(4) < capacity {
        let target = entries.len().saturating_mul(2).max(64);
        entries.shrink_to(target);
    }
}

#[cfg(test)]
pub(crate) fn function_runtime_index_stats(heap: &GcState) -> (usize, usize, usize) {
    let entries = heap.function_runtimes.borrow();
    (
        entries.len(),
        entries.capacity(),
        entries
            .values()
            .filter(|entry| entry.strong_count() == 0)
            .count(),
    )
}

/// Register the actual owner, so fields installed after publication are traced too. The heap
/// keeps only a Weak: discarded code stops rooting cells and no code/heap cycle is made.
pub(crate) fn register_code_roots<T: crate::gc_trace::GcTrace + 'static>(owner: &Rc<T>) {
    let traced: Rc<dyn crate::gc_trace::GcTrace> = owner.clone();
    with_active_gc_heap(|heap| {
        heap.code_roots.borrow_mut().push(Rc::downgrade(&traced));
        heap.young_code_roots
            .borrow_mut()
            .push(Rc::downgrade(&traced));
    });
}

/// Record the edges of `value`, which was just stored into a registered code root (a late
/// template or layout fill): the next minor collection traces them, since it traces only the
/// code roots registered since the last collection (see `GcState::young_code_roots`). Every
/// store of a new cell into an older code root must be reported here; heap verification checks
/// it after every minor collection's marking ([`check_code_roots_marked`]).
pub(crate) fn remember_code_edges(value: &dyn crate::gc_trace::GcTrace) {
    struct Words<'a>(&'a mut Vec<u64>);
    impl crate::gc_trace::GcSink for Words<'_> {
        fn word(&mut self, word: u64) {
            self.0.push(word);
        }
    }
    with_active_gc_heap(|heap| value.trace_gc(&mut Words(&mut heap.young_code_words.borrow_mut())));
}

#[inline]
pub(crate) fn activate_agent(heap: &GcHeap, symbols: &SymbolAgent) {
    let next = (Rc::as_ptr(heap) as usize, Rc::as_ptr(symbols) as usize);
    ACTIVE_AGENT_TAG.with(|tag| {
        if tag.get() == next {
            return;
        }
        remember_gc_heap(heap);
        // The tag names a heap only while it lives (see `with_agent_heap`): it moves first, and
        // the replaced heap, which the slot may hold last, drops once both name the new one.
        tag.set(next);
        let replaced = (
            ACTIVE_GC_HEAP.with(|active| active.borrow_mut().replace(heap.clone())),
            ACTIVE_SYMBOL_AGENT.with(|active| active.borrow_mut().replace(symbols.clone())),
        );
        drop(replaced);
        #[cfg(test)]
        ACTIVE_AGENT_SLOW_SWITCHES.with(|switches| {
            switches.set(switches.get().wrapping_add(1));
        });
    });
}

/// Restores the surrounding Agent after a synchronous cross-Agent/heap transition.
///
/// Ordinary ECMAScript calls do not need this: they push an execution context on the current
/// Agent's stack. ShadowRealm's separately allocated interpreter heap is the exceptional nested
/// transition in TRust JS, and restoring it by scope keeps early returns and throws correct.
#[must_use]
pub(crate) struct AgentActivationGuard {
    previous: Option<(Option<GcHeap>, Option<SymbolAgent>, (usize, usize))>,
}

impl Drop for AgentActivationGuard {
    fn drop(&mut self) {
        let Some((heap, symbols, tag)) = self.previous.take() else {
            return;
        };
        // As in `activate_agent`: the tag first, the replaced heap dropped last.
        ACTIVE_AGENT_TAG.with(|active| active.set(tag));
        let replaced = (
            ACTIVE_GC_HEAP.with(|active| std::mem::replace(&mut *active.borrow_mut(), heap)),
            ACTIVE_SYMBOL_AGENT
                .with(|active| std::mem::replace(&mut *active.borrow_mut(), symbols)),
        );
        drop(replaced);
        #[cfg(test)]
        ACTIVE_AGENT_SLOW_SWITCHES.with(|switches| {
            switches.set(switches.get().wrapping_add(1));
        });
    }
}

#[inline]
pub(crate) fn enter_agent(heap: &GcHeap, symbols: &SymbolAgent) -> AgentActivationGuard {
    let next = (Rc::as_ptr(heap) as usize, Rc::as_ptr(symbols) as usize);
    if ACTIVE_AGENT_TAG.with(Cell::get) == next {
        return AgentActivationGuard { previous: None };
    }

    let previous_heap = ACTIVE_GC_HEAP.with(|active| active.borrow().clone());
    let previous_symbols = ACTIVE_SYMBOL_AGENT.with(|active| active.borrow().clone());
    let previous_tag = ACTIVE_AGENT_TAG.with(Cell::get);
    activate_agent(heap, symbols);
    AgentActivationGuard {
        previous: Some((previous_heap, previous_symbols, previous_tag)),
    }
}

fn active_symbol(id: u64) -> Option<SymbolRef> {
    ACTIVE_SYMBOL_AGENT.with(|active| {
        active
            .borrow()
            .as_ref()?
            .borrow()
            .symbols
            .get(&id)?
            .upgrade()
    })
}

pub(crate) fn deactivate_agent_if(heap: &GcHeap, symbols: &SymbolAgent) {
    let current = (Rc::as_ptr(heap) as usize, Rc::as_ptr(symbols) as usize);
    ACTIVE_AGENT_TAG.with(|tag| {
        if tag.get() == current {
            // As in `activate_agent`: the tag first, the heap dropped last.
            tag.set((0, 0));
            let taken = (
                ACTIVE_GC_HEAP.with(|active| active.borrow_mut().take()),
                ACTIVE_SYMBOL_AGENT.with(|active| active.borrow_mut().take()),
            );
            drop(taken);
        }
    });
}

#[cfg(test)]
pub(crate) fn active_agent_slow_switches() -> u64 {
    ACTIVE_AGENT_SLOW_SWITCHES.with(Cell::get)
}

thread_local! {
    /// Where allocation goes outside any Agent: values made before an Engine exists,
    /// or after its heap was deactivated. Never collected and released only when the thread
    /// exits, so such values stay valid wherever they end up. A heap made per activation gap
    /// instead freed them as soon as an Engine activated its own.
    static DEFAULT_GC_HEAP: GcHeap = new_gc_heap();
}

/// The heap allocation uses while no Agent is active.
fn unowned_gc_heap() -> GcHeap {
    DEFAULT_GC_HEAP.with(Rc::clone)
}

pub(crate) fn active_gc_heap() -> GcHeap {
    ACTIVE_GC_HEAP.with(|active| {
        let mut active = active.borrow_mut();
        active.get_or_insert_with(unowned_gc_heap).clone()
    })
}

/// The active Agent's heap slot (`ACTIVE_GC_HEAP`). Its teardown with the thread's storage
/// also clears the Agent tag, which names the heap the slot holds and which
/// [`with_agent_heap`] reads in place of borrowing the slot: a destructor that runs later and
/// allocates then finds no Agent, rather than the heap the slot just released.
struct ActiveHeapCell(RefCell<Option<GcHeap>>);

impl std::ops::Deref for ActiveHeapCell {
    type Target = RefCell<Option<GcHeap>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for ActiveHeapCell {
    fn drop(&mut self) {
        let _ = ACTIVE_AGENT_TAG.try_with(|tag| tag.set((0, 0)));
    }
}

/// [`with_active_gc_heap`] for the hot paths (allocation, the heap's identity): the heap the
/// active Agent's tag names, without the active-heap cell's borrows; outside an Agent, the
/// same as `with_active_gc_heap`. `f` must not change the active Agent.
#[inline(always)]
fn with_agent_heap<R>(f: impl FnOnce(&GcState) -> R) -> R {
    let state = ACTIVE_AGENT_TAG.with(|tag| tag.get().0) as *const GcState;
    if !state.is_null() {
        // SAFETY: while the tag names a heap, the active-heap cell holds an `Rc` of it:
        // `activate_agent` sets both, and the activation guard and `deactivate_agent_if` change
        // both together; `f` changes neither.
        return f(unsafe { &*state });
    }
    with_active_gc_heap(f)
}

fn with_active_gc_heap<R>(f: impl FnOnce(&GcState) -> R) -> R {
    ACTIVE_GC_HEAP.with(|active| {
        if active.borrow().is_none() {
            *active.borrow_mut() = Some(unowned_gc_heap());
        }
        let active = active.borrow();
        f(active
            .as_ref()
            .expect("active GC heap was just initialized"))
    })
}

/// Number of live heap objects in the surrounding Agent.
#[cfg(test)]
pub fn live_objects() -> i64 {
    heap_live_objects(&active_gc_heap())
}

pub(crate) fn heap_live_objects(heap: &GcHeap) -> i64 {
    heap.live.get()
}

pub(crate) fn heap_allocated_objects(heap: &GcHeap) -> u64 {
    heap.allocated.get()
}

/// Number of memoized `(parent, key)` shape transitions in the active heap's shape
/// table: the shape-growth census Pilot C gates on (each entry is one shared layout
/// edge; fresh unique shapes minted by removals are deliberately not counted).
pub(crate) fn active_shape_transition_count() -> u64 {
    with_active_gc_heap(|heap| {
        heap.shapes
            .try_borrow()
            .map_or(0, |shapes| shapes.transitions.len() as u64)
    })
}

/// Stable address of this Agent's live-object counter (tests).
#[cfg(test)]
pub(crate) fn live_objects_ptr(heap: &GcHeap) -> *const i64 {
    heap.live.as_ptr()
}

/// This heap's fixed-size shape-directory array, borrowed by each native activation. Its pages
/// may be replaced as shapes are cached, but the directory itself stays inside the stable heap.
pub(crate) fn shape_layouts_ptr(heap: &GcHeap) -> *const u8 {
    // Derive the raw address from RefCell's UnsafeCell, without a shared Ref whose provenance
    // a later mutable shape-table borrow could invalidate. GcState stays at its Rc allocation;
    // `pages` is an in-place fixed array, never a resizing vector. Native probes use it only
    // outside shape-table borrows, with no helper or author code between their read and commit.
    let shapes = heap.shapes.as_ptr();
    // SAFETY: the RefCell is initialized and remains in this heap for the activation's life.
    // addr_of! only computes its field's address; it neither reads nor creates a reference.
    unsafe { std::ptr::addr_of!((*shapes).layouts.pages).cast() }
}

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod jit_heap_tests {
    use super::*;

    /// An allocation counts in the active Agent's heap, and a collection counts its own heap's
    /// deaths in that heap's counter, whichever Agent is active (cells are freed at
    /// collections).
    #[test]
    fn jit_live_object_counters_follow_their_heaps_across_agent_switches() {
        // ECMA-262, Agents (https://tc39.es/ecma262/#sec-agents): independent Agents may
        // share a native thread. A heap's counter must keep tracking that heap.
        use crate::gc_test::{alive, in_frame};
        let mut first = crate::Engine::new();
        let mut second = crate::Engine::new();
        let first_counter = live_objects_ptr(&first.interp.gc_heap);
        let second_counter = live_objects_ptr(&second.interp.gc_heap);
        assert_ne!(first_counter, second_counter);
        // SAFETY: both heap owners remain alive throughout these counter reads.
        let counts = || unsafe { (*first_counter, *second_counter) };
        {
            let _first_active = first.interp.enter_agent();
            first.interp.gc_collect();
        }
        {
            let _second_active = second.interp.enter_agent();
            second.interp.gc_collect();
        }
        let (first_base, second_base) = counts();

        let _first_active = first.interp.enter_agent();
        let first_object = in_frame(|| Gc::downgrade(&Object::new(None)));
        {
            let _second_active = second.interp.enter_agent();
            let second_object = in_frame(|| Gc::downgrade(&Object::new(None)));
            assert_eq!(counts(), (first_base + 1, second_base + 1));
            first.interp.gc_collect();
            assert!(!alive(&first_object));
            assert_eq!(counts(), (first_base, second_base + 1));
            second.interp.gc_collect();
            assert!(!alive(&second_object));
            assert_eq!(counts(), (first_base, second_base));
        }

        let first_object = in_frame(|| Gc::downgrade(&Object::new(None)));
        assert_eq!(counts(), (first_base + 1, second_base));
        first.interp.gc_collect();
        assert!(!alive(&first_object));
        assert_eq!(counts(), (first_base, second_base));
    }
}

/// The initialized object cells not yet swept (zombies excluded).
pub(crate) fn heap_gc_snapshot(heap: &GcHeap) -> Vec<Gc> {
    let cells = heap.core.borrow().live_cells(heap.object_desc);
    // SAFETY: allocation initializes every object before a safepoint can run; a snapshot never
    // includes a finalized cell, even if weak pins keep its address reserved as a zombie.
    cells
        .into_iter()
        .map(|cell| unsafe { Gc::from_raw(cell as *const RefCell<Object>) })
        .collect()
}

/// The end of a collection: the architecture-diagnostics reports, if built.
pub(crate) fn gc_finish_generation(heap: &GcHeap) {
    #[cfg(feature = "architecture-diagnostics")]
    {
        heap.diagnostics.borrow_mut().finish_generation();
        allocation_diagnostics::report(heap);
        crate::bytecode::call_cache_diagnostics::report();
    }
    #[cfg(not(feature = "architecture-diagnostics"))]
    let _ = heap;
}

/// The environment cells of `heap` not yet swept.
pub(crate) fn gc_scope_snapshot(heap: &GcHeap) -> Vec<Env> {
    let cells = heap.core.borrow().live_cells(heap.scope_desc);
    // SAFETY: an allocated cell of the environment descriptor holds an initialized record
    // (`Env::new` writes it before anything can run).
    cells
        .into_iter()
        .map(|cell| unsafe { Env::from_raw(cell as *const RefCell<crate::interpreter::Scope>) })
        .collect()
}

/// The element type of a TypedArray.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TaKind {
    I8,
    U8,
    U8Clamped,
    I16,
    U16,
    I32,
    U32,
    F16,
    F32,
    F64,
    I64,
    U64,
}

impl TaKind {
    pub(crate) fn elsize(self) -> usize {
        match self {
            TaKind::I8 | TaKind::U8 | TaKind::U8Clamped => 1,
            TaKind::I16 | TaKind::U16 | TaKind::F16 => 2,
            TaKind::I32 | TaKind::U32 | TaKind::F32 => 4,
            TaKind::F64 | TaKind::I64 | TaKind::U64 => 8,
        }
    }
    /// Whether elements are BigInt (BigInt64Array / BigUint64Array) rather than Number.
    pub(crate) fn is_bigint(self) -> bool {
        matches!(self, TaKind::I64 | TaKind::U64)
    }
    /// Constructor / prototype name, e.g. "Int8Array".
    pub(crate) fn name(self) -> &'static str {
        match self {
            TaKind::I8 => "Int8Array",
            TaKind::U8 => "Uint8Array",
            TaKind::U8Clamped => "Uint8ClampedArray",
            TaKind::I16 => "Int16Array",
            TaKind::U16 => "Uint16Array",
            TaKind::I32 => "Int32Array",
            TaKind::U32 => "Uint32Array",
            TaKind::F16 => "Float16Array",
            TaKind::F32 => "Float32Array",
            TaKind::F64 => "Float64Array",
            TaKind::I64 => "BigInt64Array",
            TaKind::U64 => "BigUint64Array",
        }
    }
    /// The key of the constructor's intrinsic in a Realm's `extra_protos`, e.g. "%Int8Array%":
    /// TypedArraySpeciesCreate's default constructor, which page script cannot replace.
    pub(crate) fn intrinsic_key(self) -> &'static str {
        match self {
            TaKind::I8 => "%Int8Array%",
            TaKind::U8 => "%Uint8Array%",
            TaKind::U8Clamped => "%Uint8ClampedArray%",
            TaKind::I16 => "%Int16Array%",
            TaKind::U16 => "%Uint16Array%",
            TaKind::I32 => "%Int32Array%",
            TaKind::U32 => "%Uint32Array%",
            TaKind::F16 => "%Float16Array%",
            TaKind::F32 => "%Float32Array%",
            TaKind::F64 => "%Float64Array%",
            TaKind::I64 => "%BigInt64Array%",
            TaKind::U64 => "%BigUint64Array%",
        }
    }
    /// Read a BigInt element (little-endian) from `b` (8 bytes) as an i128.
    pub(crate) fn read_bigint(self, b: &[u8]) -> i128 {
        let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        match self {
            TaKind::U64 => u64::from_le_bytes(arr) as i128,
            _ => i64::from_le_bytes(arr) as i128,
        }
    }
    /// Convert a BigInt (i128) to this element's 8 little-endian bytes, wrapping mod 2^64.
    pub(crate) fn write_bigint(self, n: i128) -> Vec<u8> {
        (n as u64).to_le_bytes().to_vec()
    }
    /// Read one element (little-endian) from `b` (which must be `elsize()` bytes) as a Number.
    pub(crate) fn read(self, b: &[u8]) -> f64 {
        match self {
            TaKind::I8 => b[0] as i8 as f64,
            TaKind::U8 | TaKind::U8Clamped => b[0] as f64,
            TaKind::I16 => i16::from_le_bytes([b[0], b[1]]) as f64,
            TaKind::U16 => u16::from_le_bytes([b[0], b[1]]) as f64,
            TaKind::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            TaKind::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            TaKind::F16 => f16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64,
            TaKind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64,
            TaKind::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            TaKind::I64 | TaKind::U64 => self.read_bigint(b) as f64,
        }
    }
    /// Convert a Number to this element type's little-endian bytes (JS integer-conversion rules).
    pub(crate) fn write(self, n: f64) -> Vec<u8> {
        let mut bytes = vec![0; self.elsize()];
        self.write_into(n, &mut bytes);
        bytes
    }

    /// NumericToRawBytes into existing storage. A numeric TypedArray store must not allocate a
    /// temporary byte vector for every element. Integer conversions wrap modulo 2^N, including
    /// finite Numbers outside Rust's integer range (ECMA-262 NumericToRawBytes and ToInt32/ToUint32).
    pub(crate) fn write_into(self, n: f64, bytes: &mut [u8]) {
        debug_assert_eq!(bytes.len(), self.elsize());
        let int = || {
            if (-2147483648.0..2147483648.0).contains(&n) {
                n as i32
            } else {
                crate::eval::to_int32(n)
            }
        };
        match self {
            TaKind::I8 | TaKind::U8 => bytes[0] = int() as u8,
            TaKind::U8Clamped => {
                // ToUint8Clamp: round-half-to-even (0.5 → 0, 1.5 → 2, 2.5 → 2), clamped to [0,255].
                let c = if n.is_nan() || n <= 0.0 {
                    0.0
                } else if n >= 255.0 {
                    255.0
                } else {
                    let f = n.floor();
                    if f + 0.5 < n {
                        f + 1.0
                    } else if n < f + 0.5 {
                        f
                    } else if (f as i64) % 2 == 1 {
                        f + 1.0
                    } else {
                        f
                    }
                };
                bytes[0] = c as u8;
            }
            TaKind::I16 | TaKind::U16 => bytes.copy_from_slice(&(int() as u16).to_le_bytes()),
            TaKind::I32 | TaKind::U32 => bytes.copy_from_slice(&int().to_le_bytes()),
            TaKind::F16 => bytes.copy_from_slice(&f64_to_f16(n).to_le_bytes()),
            TaKind::F32 => bytes.copy_from_slice(&(n as f32).to_le_bytes()),
            TaKind::F64 => bytes.copy_from_slice(&n.to_le_bytes()),
            TaKind::I64 | TaKind::U64 => {
                let n = if n.is_finite() { n.trunc() as i64 } else { 0 };
                bytes.copy_from_slice(&n.to_le_bytes());
            }
        }
    }
}

/// A TypedArray view's internal state (the engine's `[[ViewedArrayBuffer]]`/`[[ByteOffset]]`/
/// `[[ArrayLength]]`/`[[TypedArrayName]]`). Stored in an `Interp` side table keyed by object ptr.
#[derive(Clone, Copy)]
pub struct TaInfo {
    /// Pointer of the backing ArrayBuffer or SharedArrayBuffer object (key into
    /// `Interp::array_buffers`).
    pub buffer: usize,
    pub offset: usize,
    pub len: usize,
    pub kind: TaKind,
    /// Length-tracking view (created on a resizable buffer with no explicit length): its length is
    /// recomputed from the buffer's current size rather than fixed at `len`.
    pub track: bool,
}

impl TaInfo {
    /// ECMA-262 IsTypedArrayOutOfBounds/TypedArrayLength (snapshot e28783d5fc9d).
    /// Use the length of the currently borrowed Data Block, never a cached buffer pointer or
    /// size: fixed views become wholly out of bounds when shrunk, tracking views round down.
    #[inline]
    pub(crate) fn length_for_buffer(&self, byte_length: usize) -> Option<usize> {
        if self.track {
            Some(byte_length.checked_sub(self.offset)? / self.kind.elsize())
        } else if self
            .offset
            .checked_add(self.len.checked_mul(self.kind.elsize())?)?
            <= byte_length
        {
            Some(self.len)
        } else {
            None
        }
    }
}

/// How a property key relates to a TypedArray's integer-indexed exotic behavior.
pub enum TaIndex {
    /// A valid in-range element index.
    Element(usize),
    /// A canonical numeric key that isn't a valid index (inert: get→undefined, set/define→no-op,
    /// has→false, delete→true; never stored, never reaches the prototype).
    Exotic,
    /// An ordinary string/symbol key (handled by the normal property machinery).
    Ordinary,
}

/// An element as a property map holds it (W2c step 2): a word of its element block, which is a
/// plain data element (writable, enumerable, configurable) of that value, or the entry that
/// stores it (an index key in the named entries: a non-plain element, a far index, a frozen
/// object's). It reads as a [`Property`] does, and [`Element::to_property`] makes one when a caller
/// needs to own or store it.
///
/// A word's integrity level is its variant's (W2c step 7a), not a field: with one 8-byte field
/// per variant an element is a register pair, which an `Option<Element>` result stays in, where
/// a second field made it an aggregate returned through memory by every lookup, named or not.
#[derive(Clone, Copy)]
pub(crate) enum Element<'a> {
    /// A word of a block with no integrity level: an enumerable, writable, configurable element.
    Word(Value),
    /// A word of a sealed block ([`word_block::KIND_SEALED`]): enumerable, writable,
    /// non-configurable.
    SealedWord(Value),
    /// A word of a frozen block ([`word_block::KIND_FROZEN`]): enumerable, non-writable,
    /// non-configurable.
    FrozenWord(Value),
    Entry(&'a Property),
}

impl<'a> Element<'a> {
    #[inline]
    pub(crate) fn entry(property: &'a Property) -> Self {
        Element::Entry(property)
    }
    /// The element of a word in a block with the integrity bits `integrity`
    /// ([`word_block::Words::integrity`]).
    #[inline]
    pub(crate) fn word(value: Value, integrity: u32) -> Self {
        // Every level sets SEALED (`Words::sealed`): one bit tells a word without one.
        if integrity & word_block::KIND_SEALED == 0 {
            Element::Word(value)
        } else {
            Self::word_with_integrity(value, integrity)
        }
    }
    /// A sealed or frozen word's element, out of line: a lookup of any other word pays one
    /// test of a bit, not the selection of a level.
    #[cold]
    #[inline(never)]
    fn word_with_integrity(value: Value, integrity: u32) -> Self {
        if integrity & word_block::KIND_FROZEN == 0 {
            Element::SealedWord(value)
        } else {
            Element::FrozenWord(value)
        }
    }
    /// The property this element is.
    #[inline]
    pub(crate) fn to_property(self) -> Property {
        match self {
            Element::Word(value) => Property::plain(value),
            Element::SealedWord(value) => Property::data(value, true, true, false),
            Element::FrozenWord(value) => Property::data(value, false, true, false),
            Element::Entry(property) => property.clone(),
        }
    }
    /// The stored entry, for tests of the entries' own representation.
    #[cfg(test)]
    pub(crate) fn as_entry(self) -> &'a Property {
        match self {
            Element::Entry(property) => property,
            _ => panic!("a word element has no entry"),
        }
    }
    #[inline]
    pub(crate) fn accessor(self) -> bool {
        matches!(self, Element::Entry(property) if property.accessor())
    }
    #[inline]
    pub(crate) fn writable(self) -> bool {
        match self {
            Element::Word(_) | Element::SealedWord(_) => true,
            Element::FrozenWord(_) => false,
            Element::Entry(property) => property.writable(),
        }
    }
    #[inline]
    pub(crate) fn enumerable(self) -> bool {
        match self {
            Element::Entry(property) => property.enumerable(),
            _ => true,
        }
    }
    #[inline]
    pub(crate) fn configurable(self) -> bool {
        match self {
            Element::Word(_) => true,
            Element::SealedWord(_) | Element::FrozenWord(_) => false,
            Element::Entry(property) => property.configurable(),
        }
    }
    #[inline]
    pub(crate) fn value(self) -> Value {
        match self {
            Element::Word(value) | Element::SealedWord(value) | Element::FrozenWord(value) => value,
            Element::Entry(property) => property.value(),
        }
    }
    #[inline]
    pub(crate) fn number_value(self) -> Option<f64> {
        match self {
            Element::Word(value) | Element::SealedWord(value) | Element::FrozenWord(value) => {
                value.number()
            }
            Element::Entry(property) => property.number_value(),
        }
    }
    #[inline]
    pub(crate) fn holds_object(self, expected: *const RefCell<Object>) -> bool {
        match self {
            Element::Word(value) | Element::SealedWord(value) | Element::FrozenWord(value) => value
                .as_obj()
                .is_some_and(|object| std::ptr::eq(Gc::as_ptr(&object), expected)),
            Element::Entry(property) => property.holds_object(expected),
        }
    }
    #[inline]
    pub(crate) fn getter(self) -> Option<&'a Value> {
        match self {
            Element::Entry(property) => property.getter(),
            _ => None,
        }
    }
    #[inline]
    pub(crate) fn setter(self) -> Option<&'a Value> {
        match self {
            Element::Entry(property) => property.setter(),
            _ => None,
        }
    }
}

/// A property descriptor. A data property uses `value`/`writable`; an accessor uses the
/// getter/setter pair. The low bits of `meta` hold the four descriptor flags; its aligned upper
/// bits point to an accessor pair only for accessor properties. Thus ordinary properties are 16
/// bytes and allocate no metadata, while still keeping the flags directly readable by the JIT.
///
/// P1 S6: a property owns nothing outside the heap. Its accessor pair is an immutable cell
/// ([`Accessors`]) that copies of the property share and a change replaces, so property storage
/// needs no destructor and can live in storage blocks (`gc_storage`).
pub struct Property {
    packed: PropWord,
    meta: usize,
}

/// The getter/setter pair of an accessor property: an immutable payload cell (16-byte aligned,
/// as every cell is, which leaves `meta`'s flag bits free). Its edges are its two functions.
/// `repr(C)`: generated code reads the getter's and the setter's words ([`ACCESSOR_GET_WORD_OFF`],
/// [`ACCESSOR_SET_WORD_OFF`]).
#[repr(C, align(16))]
#[derive(Clone, Default)]
pub(crate) struct Accessors {
    get: AccessorFn,
    set: AccessorFn,
}

/// One function of an accessor pair, its [[Get]] or [[Set]], or none (`present` 0, for
/// undefined). An absent function's word is Undefined's: no function's identity matches it, so
/// generated code compares the word with a cached callee's without reading `present`.
#[repr(C)]
#[derive(Clone, Copy)]
struct AccessorFn {
    present: u64,
    word: Value,
}

impl AccessorFn {
    fn new(function: Option<Value>) -> AccessorFn {
        AccessorFn {
            present: function.is_some().into(),
            word: function.unwrap_or(Value::Undefined),
        }
    }

    fn get(&self) -> Option<&Value> {
        (self.present != 0).then_some(&self.word)
    }
}

impl Default for AccessorFn {
    fn default() -> Self {
        AccessorFn::new(None)
    }
}

/// The byte offset of an accessor pair's getter word (its [[Get]] function's, or Undefined's)
/// from the pair cell, which `meta`'s pointer bits address.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) const ACCESSOR_GET_WORD_OFF: usize =
    std::mem::offset_of!(Accessors, get) + std::mem::offset_of!(AccessorFn, word);
/// The byte offset of an accessor pair's setter word (its [[Set]] function's, or Undefined's).
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) const ACCESSOR_SET_WORD_OFF: usize =
    std::mem::offset_of!(Accessors, set) + std::mem::offset_of!(AccessorFn, word);

impl crate::gc_box::Payload for Accessors {
    fn desc(heap: &GcState) -> crate::gc_core::DescId {
        heap.accessor_desc
    }
}

impl crate::gc_trace::GcTrace for Accessors {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        let Accessors { get, set } = self;
        if let Some(get) = get.get() {
            sink.value(get);
        }
        if let Some(set) = set.get() {
            sink.value(set);
        }
    }
}

/// The edges of an accessor-pair cell: its getter and setter.
fn trace_accessor_cell(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
    use crate::gc_trace::GcTrace;
    // SAFETY: accessor cells are initialized by `gc_box::new` before any collection and never
    // change afterwards.
    unsafe { &*(cell as *const Accessors) }.trace_gc(&mut crate::gc_trace::VisitSink(visit));
}

/// A new accessor-pair cell in the active heap, as `meta`'s pointer bits.
fn accessor_cell(get: Option<Value>, set: Option<Value>) -> usize {
    let pair = crate::gc_box::new(Accessors {
        get: AccessorFn::new(get),
        set: AccessorFn::new(set),
    });
    let ptr = crate::gc_box::GcBox::as_ptr(&pair) as usize;
    debug_assert_eq!(ptr & PROP_FLAG_MASK, 0);
    ptr
}

/// Descriptor class as encoded in shape transitions (Pilot C): the only descriptor facts
/// a shape may eventually prove. Until every `&mut Property` escape demotes on class
/// change, a shape's classes are recorded but are NOT yet a consumer proof — consumers
/// still re-check descriptor bits live.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PropClass {
    DataWritable,
    DataNonWritable,
    Accessor,
}

pub(crate) const PROP_ACCESSOR: usize = 1;
pub(crate) const PROP_WRITABLE: usize = 2;
pub(crate) const PROP_ENUMERABLE: usize = 4;
pub(crate) const PROP_CONFIGURABLE: usize = 8;
const PROP_FLAG_MASK: usize = 15;

impl Clone for Property {
    fn clone(&self) -> Self {
        Property {
            // A snapshot can be stored under another key or on another object (a descriptor
            // define, a property copy), where a deferred prototype must not travel unmaterialized.
            packed: PropWord::from_value(self.packed.materialize()),
            // The accessor pair is immutable: the copy shares its cell.
            meta: self.meta,
        }
    }
}

impl Property {
    /// The requested bytes this property owns: its accessor pair and a deferred prototype's
    /// payload. A lazy word's cell is its own, since copying the word materializes it, so
    /// counting the payload per word counts each allocation once.
    fn retained_requested_storage_bytes(&self) -> usize {
        self.accessors()
            .map_or(0, |_| std::mem::size_of::<Accessors>())
            + if self.packed.tag() == PACK_LAZY_PROTO {
                std::mem::size_of::<LazyFunctionPrototype>()
            } else {
                0
            }
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        self.visit_retained_value(visitor);
        if let Some(getter) = self.getter() {
            visitor.value(getter);
        }
        if let Some(setter) = self.setter() {
            visitor.value(setter);
        }
        self.retained_requested_storage_bytes()
    }

    pub(crate) fn data(
        value: Value,
        writable: bool,
        enumerable: bool,
        configurable: bool,
    ) -> Property {
        let meta = ((writable as usize) * PROP_WRITABLE)
            | ((enumerable as usize) * PROP_ENUMERABLE)
            | ((configurable as usize) * PROP_CONFIGURABLE);
        Property {
            packed: PropWord::from_value(value),
            meta,
        }
    }
    /// An accessor property (`accessor: true`, value `Undefined`, not writable).
    pub(crate) fn accessor_prop(
        get: Option<Value>,
        set: Option<Value>,
        enumerable: bool,
        configurable: bool,
    ) -> Property {
        let flags = PROP_ACCESSOR
            | ((enumerable as usize) * PROP_ENUMERABLE)
            | ((configurable as usize) * PROP_CONFIGURABLE);
        Property {
            packed: PropWord::from_value(Value::Undefined),
            meta: accessor_cell(get, set) | flags,
        }
    }
    #[inline]
    fn accessors(&self) -> Option<&Accessors> {
        let ptr = self.meta & !PROP_FLAG_MASK;
        // SAFETY: a property that names a pair keeps its cell alive (its holder traces it).
        (ptr != 0).then(|| unsafe { &*(ptr as *const Accessors) })
    }
    /// The accessor pair's cell, as its holder reports it.
    #[inline]
    fn accessor_cell_word(&self) -> Option<u64> {
        let ptr = self.meta & !PROP_FLAG_MASK;
        (ptr != 0).then_some(ptr as u64)
    }
    #[inline]
    pub(crate) fn accessor(&self) -> bool {
        self.meta & PROP_ACCESSOR != 0
    }
    #[inline]
    pub(crate) fn writable(&self) -> bool {
        self.meta & PROP_WRITABLE != 0
    }
    /// The descriptor class this property contributes to its object's shape. Accessor
    /// takes precedence (an accessor descriptor's writable bit is meaningless).
    #[inline]
    pub(crate) fn class(&self) -> PropClass {
        if self.accessor() {
            PropClass::Accessor
        } else if self.writable() {
            PropClass::DataWritable
        } else {
            PropClass::DataNonWritable
        }
    }
    #[inline]
    pub(crate) fn enumerable(&self) -> bool {
        self.meta & PROP_ENUMERABLE != 0
    }
    #[inline]
    pub(crate) fn configurable(&self) -> bool {
        self.meta & PROP_CONFIGURABLE != 0
    }
    fn set_flag(&mut self, flag: usize, value: bool) {
        if value {
            self.meta |= flag;
        } else {
            self.meta &= !flag;
        }
    }
    pub(crate) fn set_accessor(&mut self, value: bool) {
        if !value {
            self.clear_accessors();
        }
        self.set_flag(PROP_ACCESSOR, value);
    }
    pub(crate) fn set_writable(&mut self, value: bool) {
        self.set_flag(PROP_WRITABLE, value);
    }
    pub(crate) fn set_enumerable(&mut self, value: bool) {
        self.set_flag(PROP_ENUMERABLE, value);
    }
    pub(crate) fn set_configurable(&mut self, value: bool) {
        self.set_flag(PROP_CONFIGURABLE, value);
    }
    pub(crate) fn into_value(mut self) -> Value {
        self.take_value()
    }
    #[inline]
    pub(crate) fn value(&self) -> Value {
        self.packed.materialize()
    }
    /// Whether this data property holds exactly the object at `expected`, compared without
    /// creating an owner. A deferred prototype has no address yet and never matches.
    #[inline]
    pub(crate) fn holds_object(&self, expected: *const RefCell<Object>) -> bool {
        !self.accessor()
            && self
                .packed
                .with_object(|object| std::ptr::eq(Gc::as_ptr(object), expected))
                .unwrap_or(false)
    }
    /// OrdinaryGet's own-data Number case, with no getter invocation or value conversion.
    /// Callers must still prove ordinary internal methods and fall back for every other case.
    #[inline]
    pub(crate) fn number_value(&self) -> Option<f64> {
        if self.accessor() {
            None
        } else {
            self.packed.number()
        }
    }
    pub(crate) fn visit_retained_value(&self, visitor: &mut crate::memory::Visitor) {
        if self.packed.tag() == PACK_LAZY_PROTO {
            let lazy = std::mem::ManuallyDrop::new(unsafe {
                Value::decode_payload::<LazyProto>(self.packed.0.get())
            });
            lazy.visit_retained_memory(visitor);
        } else {
            visitor.value(&self.value());
        }
    }
    /// Make this property's value `owner`'s deferred MakeConstructor prototype: a lazy word
    /// naming a new [`LazyProto`], a cell in the active heap (which is `owner`'s).
    pub(crate) fn defer_function_prototype(&mut self, owner: &Gc, parent: Gc, template: &Props) {
        let lazy = LazyFunctionPrototype::new(owner, parent, template);
        let lazy: LazyProto = crate::gc_box::new(lazy);
        self.packed = PropWord(
            Cell::new(PACK_LAZY_PROTO | unsafe { Value::into_word(lazy) }),
            std::marker::PhantomData,
        );
    }
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.packed.tag() == PACK_EMPTY
    }
    /// A present element as Array methods create it: a writable, enumerable and configurable
    /// data property (no accessor, no hole).
    #[inline]
    pub(crate) fn is_plain_element(&self) -> bool {
        self.meta == PROP_WRITABLE | PROP_ENUMERABLE | PROP_CONFIGURABLE && !self.is_empty()
    }
    /// A plain element's attributes, whether the slot holds an element or a hole (packed
    /// storage writes a hole as `Property::plain(Value::Empty)`).
    #[inline]
    pub(crate) fn has_plain_element_attributes(&self) -> bool {
        self.meta == PROP_WRITABLE | PROP_ENUMERABLE | PROP_CONFIGURABLE
    }
    #[inline]
    pub(crate) fn set_value(&mut self, value: Value) {
        self.packed = PropWord::from_value(value);
    }
    #[inline]
    pub(crate) fn replace_value(&mut self, value: Value) -> Value {
        let old = std::mem::replace(&mut self.packed, PropWord::from_value(value));
        old.materialize()
    }
    #[inline]
    pub(crate) fn take_value(&mut self) -> Value {
        self.replace_value(Value::Undefined)
    }
    #[inline]
    pub(crate) fn getter(&self) -> Option<&Value> {
        self.accessors().and_then(|a| a.get.get())
    }
    #[inline]
    pub(crate) fn setter(&self) -> Option<&Value> {
        self.accessors().and_then(|a| a.set.get())
    }
    /// Replace the getter. The pair is immutable, so this makes a new one (with the setter), or
    /// none when there is no pair and no getter.
    pub(crate) fn set_getter(&mut self, g: Option<Value>) {
        if g.is_some() || self.accessors().is_some() {
            let set = self.setter().copied();
            let flags = self.meta & PROP_FLAG_MASK;
            self.meta = accessor_cell(g, set) | flags;
        }
    }
    /// Replace the setter, as [`Property::set_getter`] does the getter.
    pub(crate) fn set_setter(&mut self, s: Option<Value>) {
        if s.is_some() || self.accessors().is_some() {
            let get = self.getter().copied();
            let flags = self.meta & PROP_FLAG_MASK;
            self.meta = accessor_cell(get, s) | flags;
        }
    }
    /// Forget the accessor pair (used when a define converts an accessor back to a data
    /// property); the collector frees its cell.
    pub(crate) fn clear_accessors(&mut self) {
        self.meta &= PROP_FLAG_MASK;
    }
    /// A default plain data property: writable, enumerable, configurable.
    pub(crate) fn plain(value: Value) -> Property {
        Property::data(value, true, true, true)
    }
    /// A non-enumerable method/builtin property: writable + configurable, not enumerable.
    pub(crate) fn builtin(value: Value) -> Property {
        Property::data(value, true, false, true)
    }
}

// No Drop: properties own nothing outside the heap, so the slots need no destructor and the
// blocks that hold them are plain data.
const _: () = assert!(!std::mem::needs_drop::<Property>());

/// One slot of a [`KeyIndex`]: a key and its entry's position, or empty.
#[derive(Clone, Copy, Default)]
struct KeySlot {
    key: Option<PropKey>,
    position: u32,
}

/// A property map's own index of its named keys, when its layout has none past
/// [`INDEX_THRESHOLD`] keys: open addressing (linear probing) over a power of two of
/// [`KeySlot`]s, at most half full, in storage blocks the map reports
/// ([`DenseStorage::trace_storage`]). Plain data only: dropping the map never reads the blocks,
/// which may have gone with its heap (a template can outlive its Agent).
#[derive(Clone)]
struct KeyIndex {
    /// Never dropped: the block is the collector's.
    slots: std::mem::ManuallyDrop<
        crate::inline_vec::InlineVec<KeySlot, 0, crate::gc_storage::GcSpill>,
    >,
    len: u32,
}

impl KeyIndex {
    /// An index of `keys`, each at its position.
    fn of(keys: &[PropKey]) -> Self {
        let mut index = KeyIndex::with_room(keys.len());
        for (position, &key) in keys.iter().enumerate() {
            index.insert(key, position);
        }
        index
    }

    fn with_room(keys: usize) -> Self {
        let capacity = (keys.max(INDEX_THRESHOLD) * 2).next_power_of_two();
        let mut slots = crate::inline_vec::InlineVec::with_capacity(capacity);
        slots.extend(std::iter::repeat_n(KeySlot::default(), capacity));
        KeyIndex {
            slots: std::mem::ManuallyDrop::new(slots),
            len: 0,
        }
    }

    /// The slot where `key`'s probe starts.
    #[inline]
    fn home(&self, key: PropKey) -> usize {
        let bits = self.slots.len().trailing_zeros();
        (key.word().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
    }

    /// The slot holding `key`, or the empty one where its probe ends.
    #[inline]
    fn probe(&self, key: PropKey) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        let mut slot = self.home(key);
        loop {
            match self.slots[slot].key {
                None => return Err(slot),
                Some(stored) if stored == key => return Ok(slot),
                Some(_) => slot = (slot + 1) & mask,
            }
        }
    }

    #[inline]
    fn get(&self, key: PropKey) -> Option<usize> {
        let slot = self.probe(key).ok()?;
        Some(self.slots[slot].position as usize)
    }

    /// Map `key` to `position`, adding it if absent.
    fn insert(&mut self, key: PropKey, position: usize) {
        if (self.len as usize + 1) * 2 > self.slots.len() {
            let mut grown = KeyIndex::with_room(self.len as usize + 1);
            for entry in self.slots.iter().filter(|entry| entry.key.is_some()) {
                grown.insert(entry.key.expect("filtered"), entry.position as usize);
            }
            *self = grown;
        }
        let slot = match self.probe(key) {
            Ok(slot) => slot,
            Err(slot) => {
                self.len += 1;
                slot
            }
        };
        self.slots[slot] = KeySlot {
            key: Some(key),
            position: position as u32,
        };
    }

    /// Unmap `key`: the later slots of its run move back over the gap (deletion under linear
    /// probing), so every probe still ends at the right slot.
    fn remove(&mut self, key: PropKey) {
        let Ok(mut gap) = self.probe(key) else {
            return;
        };
        let mask = self.slots.len() - 1;
        let mut next = (gap + 1) & mask;
        while let Some(stored) = self.slots[next].key {
            // The entry can fill the gap if its probe passes the gap on the way to it.
            let home = self.home(stored);
            if (gap.wrapping_sub(home) & mask) < (next.wrapping_sub(home) & mask) {
                self.slots[gap] = self.slots[next];
                gap = next;
            }
            next = (next + 1) & mask;
        }
        self.slots[gap] = KeySlot::default();
        self.len -= 1;
    }

    /// Every position past `removed` moves down one, after the entry there left the map.
    fn close_gap(&mut self, removed: usize) {
        for entry in self.slots.iter_mut() {
            if entry.key.is_some() && entry.position as usize > removed {
                entry.position -= 1;
            }
        }
    }

    /// The index's slots: a power of two, at least twice its keys.
    #[cfg(feature = "architecture-diagnostics")]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// The bytes of the index's block.
    fn bytes(&self) -> usize {
        self.slots.heap_capacity() * std::mem::size_of::<KeySlot>()
    }
}

/// What a property map keeps in the Rust heap beside its blocks, each part only once needed:
/// the Symbols of its Symbol keys and internal slots. The Symbol map's storage is charged to the
/// heap's external ledger ([`DenseExtras::settle_charge`]), the internal slots' by themselves.
/// An object whose map has extras needs its finalizer.
#[derive(Default)]
struct DenseExtras {
    charge: ExternalCharge,
    /// Strong ownership for Symbol-valued property keys. String keys stay inline in `entries`;
    /// this cold map exists only for objects that actually acquire a Symbol key.
    symbols: Option<Box<crate::fasthash::FastMap<u64, SymbolRef>>>,
    internal_slots: InternalSlots,
}

impl Clone for DenseExtras {
    fn clone(&self) -> Self {
        let mut copy = Self {
            charge: ExternalCharge::new(),
            symbols: self.symbols.clone(),
            // Ordinary property templates and array copies never duplicate private brands or
            // built-in internal state. Receiver transplantation copies slots explicitly.
            internal_slots: InternalSlots::default(),
        };
        copy.settle_charge();
        copy
    }
}

impl DenseExtras {
    /// Bring the external charge to the Symbol-key map's current capacity (buckets with a
    /// control byte each, and the box), after it changed.
    fn settle_charge(&mut self) {
        fn map_bytes<K, V>(map: &Option<Box<crate::fasthash::FastMap<K, V>>>) -> usize {
            map.as_ref().map_or(0, |map| {
                std::mem::size_of_val(&**map) + map.capacity() * (std::mem::size_of::<(K, V)>() + 1)
            })
        }
        let bytes = map_bytes(&self.symbols);
        self.charge.settle(bytes);
    }
}

struct EmptyInternalSlots(InternalSlots);
// Never mutated: writers make the map's own extras first (`DenseStorage::extras_mut`).
unsafe impl Sync for EmptyInternalSlots {}

static EMPTY_INTERNAL_SLOTS: EmptyInternalSlots = EmptyInternalSlots(InternalSlots {
    entries: None,
    charge: ExternalCharge::new(),
});

/// A property map's own key index (its first word, the nullable pointer the JIT's creation
/// template reads), its Rust-heap extras, and its element words (W2c; the static empty block
/// until it has elements).
#[derive(Default)]
#[repr(C)]
struct DenseStorage(
    Option<crate::gc_storage::StorageBox<KeyIndex>>,
    Option<crate::gc_storage::PayloadBox<DenseExtras>>,
    word_block::Words,
);

impl Clone for DenseStorage {
    /// Most maps a template or literal clones keep none of the three: that copy is a few loads,
    /// in line.
    #[inline]
    fn clone(&self) -> Self {
        if self.0.is_none() && self.1.is_none() && self.2.is_empty() {
            return DenseStorage::default();
        }
        self.clone_parts()
    }
}

impl DenseStorage {
    #[inline(never)]
    fn clone_parts(&self) -> Self {
        DenseStorage(self.0.clone(), self.1.clone(), self.2.clone())
    }
}

/// Whether no used word is a hole: the kind bit, else a scan (a cleared bit is not proof of a
/// hole).
#[inline]
fn words_have_no_holes(words: &word_block::Words) -> bool {
    words.kind() & word_block::KIND_NO_HOLES != 0 || !words.as_slice().contains(&PACK_EMPTY)
}

impl DenseStorage {
    #[inline]
    fn extras_mut(&mut self) -> &mut DenseExtras {
        self.1.get_or_insert_with(Default::default)
    }

    /// The map's own index, when it keeps one (`Props::build_index`).
    #[inline]
    fn index(&self) -> Option<&KeyIndex> {
        self.0.as_deref()
    }

    /// The Symbols of the map's Symbol-valued keys, if it has any.
    #[inline]
    fn symbols(&self) -> Option<&crate::fasthash::FastMap<u64, SymbolRef>> {
        self.1.as_deref()?.symbols.as_deref()
    }

    /// The map's internal slots (empty while it has none).
    #[inline]
    fn internal_slots(&self) -> &InternalSlots {
        self.1
            .as_deref()
            .map_or(&EMPTY_INTERNAL_SLOTS.0, |extras| &extras.internal_slots)
    }

    /// Report the map's blocks: the extras, the element words and the key index with its
    /// slots (see [`Props::trace_storage`]).
    #[inline]
    fn trace_storage<S: crate::gc_trace::GcSink + ?Sized>(&self, sink: &mut S) {
        use crate::gc_storage::{trace_block, StorageBox};
        if let Some(extras) = &self.1 {
            crate::gc_storage::PayloadBox::trace(extras, sink);
        }
        if let Some(block) = self.2.block_start() {
            trace_block(block, sink);
        }
        if let Some(index) = &self.0 {
            StorageBox::trace(index, sink);
            if let Some(block) = index.slots.spill_ptr() {
                trace_block(block, sink);
            }
        }
    }

    fn retain_symbol_key(&mut self, key: PropKey) {
        let Some(id) = key.symbol_id() else {
            return;
        };
        let Some(symbol) = active_symbol(id).filter(|symbol| !symbol.agent_held.get()) else {
            return;
        };
        let extras = self.extras_mut();
        extras
            .symbols
            .get_or_insert_with(Default::default)
            .insert(id, symbol);
        extras.settle_charge();
    }
    fn release_symbol_key(&mut self, key: PropKey) {
        let Some(id) = key.symbol_id() else {
            return;
        };
        if let Some(extras) = self.1.as_deref_mut() {
            if let Some(symbols) = extras.symbols.as_deref_mut() {
                symbols.remove(&id);
            }
        }
    }
    fn index_mut(&mut self) -> Option<&mut KeyIndex> {
        self.0.as_deref_mut()
    }
    /// The element words, while the map keeps its elements as words.
    #[inline]
    fn words(&self) -> Option<&word_block::Words> {
        (!self.2.is_static_empty()).then_some(&self.2)
    }
    #[inline]
    fn words_mut(&mut self) -> Option<&mut word_block::Words> {
        (!self.2.is_static_empty()).then_some(&mut self.2)
    }
    /// The words, the static empty block included, for a map that starts keeping them.
    #[inline]
    fn words_start(&mut self) -> &mut word_block::Words {
        &mut self.2
    }
    #[inline]
    fn has_words(&self) -> bool {
        !self.2.is_static_empty()
    }
    /// Stop keeping element words.
    fn clear_words(&mut self) {
        self.2 = word_block::Words::new();
    }
    fn set_index(&mut self, index: Option<KeyIndex>) {
        self.0 = index.map(crate::gc_storage::StorageBox::new);
    }
    #[cfg(test)]
    fn clear(&mut self) {
        self.0 = None;
        self.2 = word_block::Words::new();
    }
}

/// A layout's keys: a vector in a storage block (`gc_storage::GcSpill`) that the layout's cell
/// traces. The keys' atoms live as long as their heap, so the cell reports only the block.
type LayoutKeys = crate::inline_vec::InlineVec<PropKey, 0, crate::gc_storage::GcSpill>;

/// The slots of a layout's index: twice the most keys a layout with an index has.
const LAYOUT_INDEX_SLOTS: usize = 2 * SHARED_LAYOUT_MAX_FIELDS;

/// A layout's index, in a storage block: open addressing over [`LAYOUT_INDEX_SLOTS`] one-byte
/// slots, each 0 or one more than the position of a key whose probe passes through it.
#[derive(Clone)]
struct LayoutIndex([u8; LAYOUT_INDEX_SLOTS]);

impl LayoutIndex {
    /// The slot where `key`'s probe starts.
    #[inline]
    fn home(key: PropKey) -> usize {
        (key.word().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - LAYOUT_INDEX_SLOTS.ilog2()))
            as usize
    }

    fn of(keys: &[PropKey]) -> Self {
        debug_assert!(keys.len() <= SHARED_LAYOUT_MAX_FIELDS);
        let mut slots = [0u8; LAYOUT_INDEX_SLOTS];
        for (position, &key) in keys.iter().enumerate() {
            let mut slot = Self::home(key);
            while slots[slot] != 0 {
                slot = (slot + 1) % LAYOUT_INDEX_SLOTS;
            }
            slots[slot] = position as u8 + 1;
        }
        LayoutIndex(slots)
    }

    /// The position of `key` in `keys`, the keys the index was made of.
    #[inline]
    fn find(&self, keys: &[PropKey], key: PropKey) -> Option<usize> {
        let mut slot = Self::home(key);
        loop {
            // At most half the slots are taken, so a probe ends at an empty one.
            let position = (self.0[slot] as usize).checked_sub(1)?;
            if keys[position] == key {
                return Some(position);
            }
            slot = (slot + 1) % LAYOUT_INDEX_SLOTS;
        }
    }
}

/// Shared ordered keys, independent of instance values and descriptor attributes. A constructor
/// may reserve a longer layout than its live prefix; only `NamedEntries::fields.len()` keys are
/// observable. The immutable name index is built once with the layout and shared by every object
/// with that ordered key sequence. Layouts contain no JS values, prototypes, or SymbolData
/// ownership, and nothing outside the heap: the cell needs no finalizer.
#[repr(C)]
pub(crate) struct PropertyLayoutData {
    /// Keep the key vector first: the native creation paths probe this field's live offset and
    /// use it to compare a constructor's predicted next key. Never dropped: keys own nothing and
    /// the block is the collector's.
    keys: std::mem::ManuallyDrop<LayoutKeys>,
    /// Read-only after publication. A structural mutation first detaches shared data, updates
    /// `keys`, then rebuilds this table on that private copy.
    index: Option<crate::gc_storage::StorageBox<LayoutIndex>>,
    shared: Cell<bool>,
}

const _: () = assert!(!std::mem::needs_drop::<PropertyLayoutData>());

/// A non-Copy, count-free handle. A cloned handle permanently marks its data shared;
/// structural mutation of shared data copies it into a new private cell.
#[repr(transparent)]
pub(crate) struct PropertyLayout(crate::gc_box::GcBox<PropertyLayoutData>);

impl PropertyLayout {
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const PropertyLayoutData {
        crate::gc_box::GcBox::as_ptr(&self.0)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.as_ptr() == b.as_ptr()
    }

    #[inline]
    pub(crate) fn cell_word(&self) -> u64 {
        self.as_ptr() as u64
    }

    #[inline]
    pub(crate) fn mark_shared(&self) {
        self.shared.set(true);
    }

    #[inline]
    pub(crate) fn is_shared(&self) -> bool {
        self.shared.get()
    }

    fn make_mut(&mut self) -> &mut PropertyLayoutData {
        if self.is_shared() {
            *self = PropertyLayout(crate::gc_box::new((**self).clone()));
        }
        self.get_mut_private()
            .expect("detached property layout is private")
    }

    fn get_mut_private(&mut self) -> Option<&mut PropertyLayoutData> {
        if self.is_shared() {
            return None;
        }
        // SAFETY: non-Copy handles mark shared on Clone. A private cell has one owning slot
        // (checked at safepoints), exclusively borrowed here. Its new storage blocks are edges
        // an old cell gains, which its card records (the kind is carded).
        unsafe { crate::gc_core::card_mark(self.as_ptr().cast()) };
        Some(unsafe { &mut *self.as_ptr().cast_mut() })
    }
}

impl Clone for PropertyLayout {
    #[inline]
    fn clone(&self) -> Self {
        self.mark_shared();
        Self(self.0)
    }
}

impl std::ops::Deref for PropertyLayout {
    type Target = PropertyLayoutData;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for PropertyLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PropertyLayout")
            .field(&self.as_ptr())
            .finish()
    }
}

impl crate::gc_trace::GcTrace for PropertyLayout {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        sink.layout(self);
    }
}

impl crate::gc_box::Payload for PropertyLayoutData {
    fn desc(heap: &GcState) -> crate::gc_core::DescId {
        heap.property_layout_desc
    }
}

/// The edges of a layout's cell: its storage blocks.
fn trace_property_layout_cell(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
    // SAFETY: layout cells are initialized by `gc_box::new` before any collection, and change
    // only through their single owner, never during one.
    let layout = unsafe { &*(cell as *const PropertyLayoutData) };
    layout.trace_blocks(&mut crate::gc_trace::VisitSink(visit));
}

impl Clone for PropertyLayoutData {
    fn clone(&self) -> Self {
        Self {
            keys: std::mem::ManuallyDrop::new((*self.keys).clone()),
            index: self.index.clone(),
            shared: Cell::new(false),
        }
    }
}

impl PropertyLayoutData {
    pub(crate) fn new(keys: impl IntoIterator<Item = PropKey>) -> Self {
        Self {
            keys: std::mem::ManuallyDrop::new(keys.into_iter().collect()),
            index: None,
            shared: Cell::new(false),
        }
    }

    fn with_shared_index(keys: impl IntoIterator<Item = PropKey>) -> Self {
        let mut layout = Self::new(keys);
        if layout.keys.len() > INDEX_THRESHOLD && layout.keys.len() <= SHARED_LAYOUT_MAX_FIELDS {
            layout.index = Some(layout.build_index());
        }
        layout
    }

    /// Report the storage blocks: the keys' and the index's.
    fn trace_blocks(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        if let Some(block) = self.keys.spill_ptr() {
            crate::gc_storage::trace_block(block, sink);
        }
        if let Some(index) = &self.index {
            crate::gc_storage::StorageBox::trace(index, sink);
        }
    }

    fn rebuild_index(&mut self) {
        if self.index.is_none() {
            return;
        }
        if self.keys.len() <= INDEX_THRESHOLD || self.keys.len() > SHARED_LAYOUT_MAX_FIELDS {
            self.index = None;
        } else {
            self.index = Some(self.build_index());
        }
    }

    fn build_index(&self) -> crate::gc_storage::StorageBox<LayoutIndex> {
        crate::gc_storage::StorageBox::new(LayoutIndex::of(&self.keys))
    }

    #[inline]
    fn indexed_slot(&self, key: PropKey) -> Option<usize> {
        self.index.as_ref()?.find(&self.keys, key)
    }

    pub(crate) fn has_index(&self) -> bool {
        self.index.is_some()
    }

    /// The bytes of the index's block, if the layout has one.
    pub(crate) fn index_bytes(&self) -> Option<usize> {
        self.index
            .as_ref()
            .map(|_| std::mem::size_of::<LayoutIndex>())
    }

    pub(crate) fn capacity(&self) -> usize {
        self.keys.capacity()
    }

    #[inline]
    fn as_slice(&self) -> &[PropKey] {
        &self.keys
    }

    fn keys_mut(&mut self) -> &mut LayoutKeys {
        &mut self.keys
    }
}

impl std::ops::Deref for PropertyLayoutData {
    type Target = [PropKey];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

pub(crate) fn new_property_layout<K: IntoPropKey>(
    keys: impl IntoIterator<Item = K>,
) -> PropertyLayout {
    PropertyLayout(crate::gc_box::new(PropertyLayoutData::new(
        keys.into_iter().map(IntoPropKey::into_prop_key),
    )))
}

/// Build an immutable layout index for a bounded shape/constructor key sequence. Longer maps
/// retain the existing per-object index and its incremental insertion behavior.
pub(crate) fn new_shared_indexed_property_layout<K: IntoPropKey>(
    keys: impl IntoIterator<Item = K>,
) -> PropertyLayout {
    let layout = PropertyLayout(crate::gc_box::new(PropertyLayoutData::with_shared_index(
        keys.into_iter().map(IntoPropKey::into_prop_key),
    )));
    layout.mark_shared();
    layout
}

/// Named properties kept in the object cell (S3, Decision 3): the measured count that holds
/// nearly every short-lived plain object in one allocation.
pub(crate) const INLINE_NAMED_SLOTS: usize = 4;

/// Named-property values in key-layout order: in the cell up to [`INLINE_NAMED_SLOTS`], beyond
/// that in one heap buffer (see `crate::inline_vec`), a layout the JIT reads.
type Fields =
    crate::inline_vec::InlineVec<Property, INLINE_NAMED_SLOTS, crate::gc_storage::GcSpill>;

/// Bytes a map with room for `capacity` named values owns outside its object: none while they
/// fit the inline slots.
#[cfg(test)]
pub(crate) fn detached_field_bytes(capacity: usize) -> usize {
    if capacity <= INLINE_NAMED_SLOTS {
        return 0;
    }
    capacity * std::mem::size_of::<Property>()
}

#[derive(Clone, Default)]
struct NamedEntries {
    /// Never dropped: its elements and its spill own nothing to release (`Property` holds no
    /// drop glue, the spill is a storage block), so a map, and an object, carry no drop glue.
    fields: std::mem::ManuallyDrop<Fields>,
    layout: Option<PropertyLayout>,
}

impl NamedEntries {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            fields: std::mem::ManuallyDrop::new(Fields::with_capacity(capacity)),
            layout: None,
        }
    }
    /// The capacity of a separate heap buffer for the values: none while they sit in the cell.
    fn heap_capacity(&self) -> usize {
        if !self.fields.spilled() {
            return 0;
        }
        self.fields.capacity()
    }
    fn len(&self) -> usize {
        self.fields.len()
    }
    fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
    fn capacity(&self) -> usize {
        self.fields.capacity()
    }
    fn reserve_exact(&mut self, additional: usize) {
        self.fields.reserve_exact(additional);
    }
    #[inline]
    fn keys(&self) -> &[PropKey] {
        self.layout
            .as_deref()
            .map_or(&[], |keys| &keys[..self.len()])
    }
    #[inline]
    fn iter(&self) -> impl Iterator<Item = (PropKey, &Property)> {
        self.keys().iter().copied().zip(self.fields.iter())
    }
    #[inline]
    fn get(&self, slot: usize) -> Option<(PropKey, &Property)> {
        Some((*self.keys().get(slot)?, self.fields.get(slot)?))
    }
    fn push(&mut self, (key, property): (PropKey, Property)) {
        let len = self.len();
        // A template/constructor's predicted next key is already owned once by the layout.
        // A mismatch detaches before mutation; speculation never creates an observable key.
        if !self
            .layout
            .as_ref()
            .is_some_and(|keys| keys.get(len) == Some(&key))
        {
            if len == 0 && key == crate::atom::names::LENGTH.key() {
                // Array builders outside the native literal helper share this key-only prefix
                // too; element slots remain in their existing independent dense representation.
                self.layout = Some(array_length_layout());
            } else {
                let layout = self
                    .layout
                    .get_or_insert_with(|| new_property_layout::<PropKey>([]))
                    .make_mut();
                layout.keys_mut().truncate(len);
                layout.keys_mut().push(key);
                layout.rebuild_index();
            }
        }
        self.fields.push(property);
    }
    fn predicts(&self, key: PropKey) -> bool {
        self.layout
            .as_ref()
            .is_some_and(|keys| keys.get(self.len()) == Some(&key))
    }
    fn remove(&mut self, slot: usize) {
        let len = self.len();
        let layout = self.layout.as_mut().expect("live entry layout").make_mut();
        layout.keys_mut().truncate(len);
        layout.keys_mut().remove(slot);
        self.fields.remove(slot);
        layout.rebuild_index();
    }
    fn retain(&mut self, mut keep: impl FnMut((PropKey, &Property)) -> bool) {
        let len = self.len();
        if len == 0 {
            return;
        }
        let layout = self.layout.as_mut().expect("live entry layout").make_mut();
        let keys = layout.keys_mut();
        keys.truncate(len);
        let mut slot = 0;
        let mut retained = 0;
        self.fields.retain(|property| {
            let yes = keep((keys[slot], property));
            if yes {
                keys.swap(retained, slot);
                retained += 1;
            }
            slot += 1;
            yes
        });
        keys.truncate(retained);
        layout.rebuild_index();
    }
    #[cfg(test)]
    fn clear(&mut self) {
        self.fields.clear();
        self.layout = None;
    }
}

/// The flag byte of an object and its property map ([`Props::flags`], W2d step 1). The object's
/// bits live in its map's byte, which a whole-map replacement keeps ([`Object::replace_properties`]);
/// the order puts the five that the array-append guard tests in the low bits, so one mask and
/// compare tests them all.
///
/// `false` for objects whose behavior lives in an interpreter side table — proxies, typed
/// arrays, module namespaces — which the kind can't reveal. The JIT's inline property/element
/// caches test this bit on the receiver and take the checked helper when it is clear, so ONE
/// proxy existing somewhere does not disable the caches for every plain object in the program
/// (the old global `inline_ic_safe` latch).
pub(crate) const FLAG_IC_PLAIN: u8 = 1 << 0;
/// [[Extensible]].
pub(crate) const FLAG_EXTENSIBLE: u8 = 1 << 1;
/// The map belongs to an `ObjectKind::Array` object: canonical-index key inserts skip the
/// shape transition. Array shapes encode the *named*-key sequence only (matching `push_dense`,
/// which never transitioned) — the get-IC only ever uses an array's shape to prove a named key's
/// ABSENCE or with a per-hit key re-check, never for bare slot trust, so elements must not churn
/// it: a stable shape is what lets `arr.push(..)`/`arr.length` sites cache at all.
pub(crate) const FLAG_ELEM_MODE: u8 = 1 << 2;
/// This object serves (or once served) as some object's prototype: structural changes to it bump
/// the global [`proto_epoch`], invalidating every property-*creation* inline cache (their
/// fill-time chain walks proved "no hop shadows this name" — see [`crate::bytecode::IC_CREATE`]).
/// Set by the creation-IC fill walk itself, one-way; a property of the object's history, so a
/// whole-map replacement keeps it.
pub(crate) const FLAG_PROTO: u8 = 1 << 3;
/// Some canonical-index key lives in the named entries (SpiderMonkey's `Indexed`): a non-plain
/// element, one past the sparse frontier, or a frozen object's (see `note_inserted`). While it is
/// clear, an index that is not a word is absent with no lookup; while it is set, the words'
/// append/pop fast paths stand down. Truncation rebuilds it; deleting a named index can clear it
/// after a bounded scan of the remaining entries.
pub(crate) const FLAG_HAS_FAR: u8 = 1 << 4;
/// IsConstructor, recorded when the object gets its [[Call]]: set for user functions other than
/// arrows, methods, generators and async functions, for natives marked as constructors, and for
/// bound functions and callable proxies whose target was one. A native with an own data
/// `prototype` is a constructor too (`Interp::value_is_constructor`).
pub(crate) const FLAG_CONSTRUCTOR: u8 = 1 << 5;
/// A class constructor's kind, two bits ([`CLASS_BASE`] or [`CLASS_DERIVED`]; 0 for every other
/// object), set as the class is defined: calls refuse a class constructor, and construction
/// treats a derived class's specially, by these bits rather than a probe of the class side table
/// (`Interp::class_info`), which keeps the class's elements.
pub(crate) const FLAG_CLASS_SHIFT: u32 = 6;
const FLAG_CLASS_MASK: u8 = 0b11 << FLAG_CLASS_SHIFT;
/// The object's bits, which stay with the object when its map is replaced.
const OBJECT_FLAGS: u8 =
    FLAG_IC_PLAIN | FLAG_EXTENSIBLE | FLAG_PROTO | FLAG_CONSTRUCTOR | FLAG_CLASS_MASK;
/// A new object's: extensible and ordinary (`ic_plain`).
const OBJECT_FLAGS_INITIAL: u8 = FLAG_IC_PLAIN | FLAG_EXTENSIBLE;

/// Insertion-ordered string-keyed property map. A `Vec` of entries preserves order (good enough for
/// `for-in`/`Object.keys`); a side `HashMap` keeps lookup O(1).
#[derive(Clone)]
pub struct Props {
    entries: NamedEntries,
    /// The object's and the map's flag bits ([`FLAG_IC_PLAIN`] and the rest), one byte that
    /// generated code tests a bit of (W2d step 1).
    flags: Cell<u8>,
    /// Object shape (hidden class): the id encoding this map's ordered (key, descriptor
    /// class) sequence (see [`ShapeTable`]). A shared cacheable id proves the same keys in
    /// the same order, so an inline cache that recorded (shape, slot) from one object can
    /// trust that slot on any other object of the same shape — without a key compare.
    /// Bumped to a child on new-key insert, to a fresh unique on a structural removal or
    /// a class-changing descriptor replacement. Only consulted for non-exotic
    /// objects (arrays keep the key-compare path — same shape can mean different element counts).
    shape: u32,
    /// The map's own key index (a nullable pointer), the Rust-heap extras, and the
    /// element words (W2c): every map's plain data elements, as V8, SpiderMonkey and JSC give
    /// every object element storage. Ordinary small named-property objects allocate none of it.
    elems: DenseStorage,
    /// The `entries` slot of the `"prototype"` key, or `NO_SLOT` — same memo discipline as
    /// `len_slot`. Every `new` reads the constructor's `.prototype`; function objects are
    /// ordinary maps, so this skips the scan on the construct hot path.
    proto_slot: std::cell::Cell<SlotMemo>,
    /// The `entries` slot of the `"length"` key, or `NO_SLOT`. Array `length` can't live in the
    /// inline caches (element entries occupy slots without transitioning the shape, so a shape
    /// match doesn't pin the slot) — this memo makes the every-time re-derive a direct slot read
    /// instead of a hashed key lookup. Maintained by `insert`; any slot-shifting removal resets
    /// it (`remove` re-memoizes on the next lookup via `length_slot`).
    len_slot: std::cell::Cell<SlotMemo>,
}

/// Decode a Symbol-valued property key. Escaped String keys begin with two NULs and
/// therefore never enter Symbol ownership, even when their author text spells a Symbol id.
fn encoded_symbol_id(key: &str) -> Option<u64> {
    let digits = key.strip_prefix('\0')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Exact-i32 (and not -0.0): the value survives an i32 round trip bit-identically.
#[inline]
pub(crate) fn f64_exact_i32(f: f64) -> bool {
    (f as i32 as f64).to_bits() == f.to_bits()
}

/// A memoized `entries` slot (of `length` or `prototype`), or none. Narrow, so that the object
/// header fits its size class (S3); a slot past its range is not memoized.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct SlotMemo(SlotMemoWord);
type SlotMemoWord = u16;

impl SlotMemo {
    const NONE: Self = Self(SlotMemoWord::MAX);

    #[inline]
    fn of(slot: usize) -> Self {
        SlotMemoWord::try_from(slot).map_or(Self::NONE, Self)
    }

    #[inline]
    fn slot(self) -> Option<usize> {
        (self.0 != SlotMemoWord::MAX).then_some(self.0 as usize)
    }
}

/// V8's limits for an element joining the words (`Props::words_admit`, W2c step 6): the gap past
/// the capacity (`JSObject::kMaxGap`), the grown capacity under which no density test applies
/// (`kMaxUncheckedFastElementsLength`), and the largest span the named entries may move to
/// (`NumberDictionary::kRequiresSlowElementsLimit`).
const WORDS_MAX_GAP: usize = 1024;
const WORDS_MAX_UNCHECKED: usize = 5000;
const WORDS_MAX_SPAN: usize = (1 << 29) - 1;

/// The words V8's NumberDictionary takes for `n` elements: ComputeCapacity(n) (1.5 n rounded up
/// to a power of two, at least 4) entries of 3 words.
fn dictionary_words(n: usize) -> usize {
    (n + n / 2).next_power_of_two().max(4) * 3
}

/// The empty-object shape: every `Props` starts here and all empty objects share it, so adding
/// the same first key to two of them lands on the same child shape.
const SHAPE_EMPTY: u32 = 0;

/// The property-creation epoch (see [`Props::proto_flag`]): bumped whenever a marked prototype
/// mutates structurally, any `[[SetPrototypeOf]]` succeeds, or a `defineProperty` rewrites
/// attributes — every event that could shadow a creation IC's "the chain has no setter /
/// non-writable / own copy of this name" proof. Process-global and atomic, NOT thread-local:
/// generator/async bodies run JS on pooled worker threads sharing the same `Interp` (one thread
/// at a time via channel handoff, which also orders these accesses), so a bump from a worker
/// must be visible to caches validated on the main thread. Starts at 1; saturates at `u32::MAX`,
/// which no cache hit accepts — after ~4e9 invalidations the creation ICs simply turn off
/// instead of ABA-cycling.
static PROTO_EPOCH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// The current creation-IC epoch. `u32::MAX` = permanently invalidated (see [`PROTO_EPOCH`]).
#[inline]
pub(crate) fn proto_epoch() -> u32 {
    PROTO_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
}

/// Stable address used by the ARM64 creation-IC template for the same relaxed epoch check.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[inline]
pub(crate) fn proto_epoch_ptr() -> *const u32 {
    PROTO_EPOCH.as_ptr()
}

/// Invalidate every property-creation inline cache (see [`PROTO_EPOCH`]).
pub(crate) fn bump_proto_epoch() {
    let _ = crate::atomic_compat::try_update_u32(
        &PROTO_EPOCH,
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
        |v| Some(v.saturating_add(1)),
    );
}

/// The object-shape (hidden-class) transition tree. A shape id encodes an *ordered sequence
/// of (property key, descriptor class) pairs*. A shared cacheable id proves the same keys
/// in the same order; the recorded classes are not yet a consumer proof — the inline cache
/// still re-checks accessor/writable at the slot until every `&mut Property` escape demotes
/// on class change (Pilot C steps 5-6).
/// `transitions[(parent, key, class)] = child` is memoized within an Agent, so its structurally-identical
/// objects converge on one id. Fresh IDs are process-wide and never reused: foreign objects and
/// foreign-context mutations cannot collide with another Agent's proofs. A structural *removal* can't be a tree
/// transition (it doesn't extend the key sequence), so it mints a fresh unique id that no cache
/// ever holds — forcing a re-derive.
struct ShapeTable {
    transitions: crate::fasthash::FastMap<(u32, PropKey, PropClass), u32>,
    /// Bounded tagged lookup: process-wide IDs are never dense within a particular heap.
    layouts: ShapeLayouts,
    /// Direct-mapped memo of recent `transitions` hits keyed by parent shape and key (allocated
    /// on first use): most transitions skip hashing the triple. `transitions` is append-only,
    /// so an entry never disagrees with it.
    recent: Vec<Option<(u32, PropKey, PropClass, u32)>>,
}

const RECENT_TRANSITIONS: usize = 512;

#[inline]
fn recent_transition_slot(parent: u32, key: PropKey) -> usize {
    let identity = key.word() as usize;
    let mixed = (identity >> 4) ^ (identity >> 13) ^ (parent as usize).wrapping_mul(0x9E37_79B9);
    mixed % RECENT_TRANSITIONS
}

// Eager prefix sharing is bounded independently of the existing shape identity table. Larger or
// highly irregular maps use private copy-on-write keys; they do not build a quadratic collection
// of full layouts. Layouts contain key strings only, never JS objects, prototype chains or values.
const SHARED_LAYOUT_MAX_FIELDS: usize = 16;

impl ShapeTable {
    fn new() -> ShapeTable {
        ShapeTable {
            transitions: Default::default(),
            layouts: ShapeLayouts::default(),
            recent: Vec::new(),
        }
    }

    fn fresh(&mut self) -> u32 {
        property_shapes::fresh_shape()
    }
}

/// The child shape reached by adding `key` with descriptor class `class` to shape
/// `parent` (memoized so it is shared). Two maps with the same keys but different
/// descriptor classes now get different shapes; the layout still records keys only.
fn shape_transition(
    parent: u32,
    key: PropKey,
    class: PropClass,
    prefix: Option<&[(PropKey, PropClass)]>,
) -> (u32, Option<PropertyLayout>) {
    with_active_gc_heap(|heap| {
        let mut shapes = heap.shapes.borrow_mut();
        // Unknown parents do not describe one key sequence. In particular, never memoize
        // (uncacheable, key): two exhausted maps can have entirely different prefixes.
        if !is_cacheable_shape(parent) {
            return (SHAPE_UNCACHEABLE, None);
        }
        let slot = recent_transition_slot(parent, key);
        let recent = shapes.recent.get(slot).and_then(Option::as_ref).and_then(
            |(cached_parent, cached_key, cached_class, child)| {
                (*cached_parent == parent && *cached_key == key && *cached_class == class)
                    .then_some(*child)
            },
        );
        let id = if let Some(id) = recent {
            id
        } else {
            let pair = (parent, key, class);
            let id = if let Some(&id) = shapes.transitions.get(&pair) {
                id
            } else {
                let id = shapes.fresh();
                if is_cacheable_shape(id) {
                    shapes.transitions.insert(pair, id);
                }
                id
            };
            if is_cacheable_shape(id) {
                if shapes.recent.is_empty() {
                    shapes.recent.resize(RECENT_TRANSITIONS, None);
                }
                shapes.recent[slot] = Some((parent, key, class, id));
            }
            id
        };
        if !is_cacheable_shape(id) {
            return (id, None);
        }
        let Some(prefix) = prefix else {
            return (id, None);
        };
        if let Some(layout) = shapes.layouts.get(id) {
            return (id, Some(layout.clone()));
        }
        if prefix.len() >= SHARED_LAYOUT_MAX_FIELDS {
            return (id, None);
        }
        let names = prefix
            .iter()
            .map(|&(name, _)| name)
            .chain(std::iter::once(key));
        let layout = new_shared_indexed_property_layout(names);
        shapes.layouts.insert(id, layout.clone());
        // Cold transition learning: a previously seen prefix can predict the rest of this
        // ordered insertion chain. Existing instances retain their pinned layout; subsequent
        // instances can reserve once and append into its live prefix without another lookup.
        // Extend only matching predictions, never oscillate between incompatible branches.
        // No shape identity or observable property changes: field count still defines presence.
        let mut prefix_shape = SHAPE_EMPTY;
        for &(name, class) in prefix {
            let Some(&prefix_id) = shapes.transitions.get(&(prefix_shape, name, class)) else {
                break;
            };
            if let Some(prediction) = shapes.layouts.get_mut(prefix_id) {
                if prediction.len() < layout.len() && layout.starts_with(prediction.as_slice()) {
                    *prediction = layout.clone();
                }
            }
            prefix_shape = prefix_id;
        }
        (id, Some(layout))
    })
}

fn shape_layout(id: u32) -> Option<PropertyLayout> {
    with_active_gc_heap(|heap| heap.shapes.borrow().layouts.get(id).cloned())
}

/// A fresh unique shape id (a structural removal / deopt — no cache should still match).
fn shape_fresh() -> u32 {
    with_active_gc_heap(|heap| heap.shapes.borrow_mut().fresh())
}

/// Entry count up to which a `Props` runs without a hash index (linear-scan lookups, no hash
/// allocation or rehash on insert). Most objects — instance fields, cons cells, literals — stay
/// under it for their whole life.
pub(crate) const INDEX_THRESHOLD: usize = 8;

/// Shape reached by adding the intrinsic `"length"` key to an empty map. Array literals create
/// this same one-property named map constantly. The memo belongs to the Agent alongside its
/// transition table; globally unique IDs also remain valid on pooled coroutine workers.
fn array_length_layout() -> PropertyLayout {
    with_active_gc_heap(|heap| {
        let mut cached = heap.array_length_layout.borrow_mut();
        cached
            .get_or_insert_with(|| {
                let layout = new_property_layout([crate::atom::names::LENGTH.key()]);
                layout.mark_shared();
                layout
            })
            .clone()
    })
}

fn array_length_shape() -> u32 {
    with_active_gc_heap(|heap| {
        let cached = heap.array_length_shape.get();
        if cached != SHAPE_EMPTY {
            return cached;
        }
        let mut shapes = heap.shapes.borrow_mut();
        // Array `length` is created writable; a later non-writable redefinition demotes
        // through the ordinary insert path like any other class change.
        let key = (
            SHAPE_EMPTY,
            crate::atom::names::LENGTH.key(),
            PropClass::DataWritable,
        );
        let shape = if let Some(&shape) = shapes.transitions.get(&key) {
            shape
        } else {
            let shape = shapes.fresh();
            if is_cacheable_shape(shape) {
                shapes.transitions.insert(key, shape);
            }
            shape
        };
        heap.array_length_shape.set(shape);
        shape
    })
}

/// A property key accepted by [`Props::insert`]: a key, or text made a key of the active heap
/// ([`PropKey::from_text`]).
pub(crate) trait IntoPropKey {
    fn into_prop_key(self) -> PropKey;
}

impl IntoPropKey for PropKey {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        self
    }
}

impl IntoPropKey for crate::atom::Atom {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        PropKey::atom(self)
    }
}

impl IntoPropKey for Rc<str> {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        PropKey::from_text(&self)
    }
}

impl IntoPropKey for &Rc<str> {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        PropKey::from_text(self)
    }
}

impl IntoPropKey for String {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        PropKey::from_text(&self)
    }
}

impl IntoPropKey for &String {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        PropKey::from_text(self)
    }
}

impl IntoPropKey for &str {
    #[inline]
    fn into_prop_key(self) -> PropKey {
        PropKey::from_text(self)
    }
}

/// An internal slot's name accepted by [`InternalSlots::insert`]. Owned and shared strings are
/// adopted as they are; a borrowed `&str` goes through [`intern_key`] instead of allocating a
/// fresh copy for every new slot. Slot names are not atoms: a class evaluation mints its private
/// names, which die with it.
pub(crate) trait IntoSlotKey {
    fn into_slot_key(self) -> Rc<str>;
}

impl IntoSlotKey for Rc<str> {
    #[inline]
    fn into_slot_key(self) -> Rc<str> {
        self
    }
}

impl IntoSlotKey for &Rc<str> {
    #[inline]
    fn into_slot_key(self) -> Rc<str> {
        self.clone()
    }
}

impl IntoSlotKey for String {
    #[inline]
    fn into_slot_key(self) -> Rc<str> {
        Rc::from(self)
    }
}

impl IntoSlotKey for &String {
    #[inline]
    fn into_slot_key(self) -> Rc<str> {
        intern_key(self)
    }
}

impl IntoSlotKey for &str {
    #[inline]
    fn into_slot_key(self) -> Rc<str> {
        intern_key(self)
    }
}

const KEY_CACHE_SLOTS: usize = 512;
/// Longer keys are rare and are not worth comparing on every lookup.
const KEY_CACHE_MAX_LEN: usize = 48;

thread_local! {
    /// Recently created slot-name strings, indexed by content hash: built-in code names the same
    /// few slots over and over. Each slot owns one strong reference, released when the slot is
    /// replaced. The table has no destructor, so a pooled coroutine thread's exit never touches
    /// the counts of names still in use by objects elsewhere; at most these few hundred short
    /// strings stay allocated.
    static KEY_CACHE: [Cell<std::mem::ManuallyDrop<Option<Rc<str>>>>; KEY_CACHE_SLOTS] =
        const { [const { Cell::new(std::mem::ManuallyDrop::new(None)) }; KEY_CACHE_SLOTS] };
}

/// An `Rc<str>` with `key`'s text, shared with a recent identical name when possible. Text that
/// starts with a digit is not cached: its cardinality would only evict reusable names.
pub(crate) fn intern_key(key: &str) -> Rc<str> {
    let bytes = key.as_bytes();
    if bytes.is_empty() || bytes.len() > KEY_CACHE_MAX_LEN || bytes[0].is_ascii_digit() {
        return Rc::from(key);
    }
    let mut hasher = crate::fasthash::FxHasher::default();
    std::hash::Hasher::write(&mut hasher, bytes);
    let slot = (std::hash::Hasher::finish(&hasher) as usize) % KEY_CACHE_SLOTS;
    let Ok(interned) = KEY_CACHE.try_with(|slots| {
        let cell = &slots[slot];
        let cached =
            std::mem::ManuallyDrop::into_inner(cell.replace(std::mem::ManuallyDrop::new(None)));
        let interned = match cached {
            Some(cached) if &*cached == key => cached,
            evicted => {
                drop(evicted);
                Rc::from(key)
            }
        };
        cell.set(std::mem::ManuallyDrop::new(Some(interned.clone())));
        interned
    }) else {
        return Rc::from(key);
    };
    interned
}

#[cfg(test)]
mod key_interning_tests {
    use super::*;

    #[test]
    fn key_text_matches_number_to_string_and_symbol_keys() {
        for (n, text) in [
            (0.0, "0"),
            (-0.0, "0"),
            (7.0, "7"),
            (-7.0, "-7"),
            (4294967295.0, "4294967295"),
            (9007199254740991.0, "9007199254740991"),
            (-9007199254740991.0, "-9007199254740991"),
        ] {
            assert_eq!(
                KeyText::integer(n)
                    .map(|t| t.as_str().to_owned())
                    .as_deref(),
                Some(text)
            );
        }
        for n in [
            0.5,
            -1.5,
            f64::NAN,
            f64::INFINITY,
            -f64::INFINITY,
            9007199254740992.0,
            1e21,
        ] {
            assert!(KeyText::integer(n).is_none(), "{n}");
        }
        for id in [0, 7, 1234, u64::MAX] {
            assert_eq!(KeyText::symbol(id).as_str(), format!("\u{0}{id}"));
        }
    }

    #[test]
    fn borrowed_slot_names_share_text() {
        let (first, second) = (String::from("alpha"), String::from("alpha"));
        assert!(Rc::ptr_eq(&intern_key(&first), &intern_key(&second)));
        assert_eq!(&*intern_key("alphb"), "alphb");
        let long = "x".repeat(KEY_CACHE_MAX_LEN + 1);
        for text in ["", "0", "12", "7up", long.as_str()] {
            assert_eq!(&*intern_key(text), text);
        }
        for text in (0..5000).map(|n| format!("k{n}")) {
            assert_eq!(&*intern_key(&text), text.as_str());
        }
    }

    #[test]
    fn equal_key_text_is_one_key_and_reaches_one_shape() {
        let heap = new_gc_heap();
        let symbols = new_symbol_agent();
        let _active = enter_agent(&heap, &symbols);
        let (first, second) = (String::from("alpha"), String::from("alpha"));
        assert!(first.as_str().into_prop_key() == (&second).into_prop_key());
        assert!(PropKey::lookup_text("alpha") == Some(PropKey::from_text("alpha")));
        assert!(PropKey::lookup_text("never inserted").is_none());
        assert_eq!(PropKey::from_text("12").as_index(), Some(12));
        assert!(PropKey::from_text("012").as_atom().is_some());
        // Equal key text reaches one shape whatever string carries it; a different order does
        // not. The recent-transition memo always agrees with the full table, also after other
        // transitions have replaced its entries.
        let texts: Vec<String> = (0..2000).map(|n| format!("k{n}")).collect();
        let build = |keys: &[&str], shared: bool| {
            let mut props = Props::new();
            for key in keys {
                if shared {
                    props.insert(*key, Property::plain(Value::Undefined));
                } else {
                    props.insert(Rc::<str>::from(*key), Property::plain(Value::Undefined));
                }
            }
            props.shape()
        };
        let shape = build(&["x", "y", "z"], false);
        assert_eq!(build(&["x", "y", "z"], true), shape);
        assert_ne!(build(&["y", "x", "z"], true), shape);
        for text in &texts {
            build(&["x", text.as_str()], true);
        }
        assert_eq!(build(&["x", "y", "z"], false), shape);
        assert_eq!(build(&["x", "y", "z"], true), shape);
    }
}

/// A property key's text formatted on the stack, for computed member accesses whose key needs
/// no owned string: an integral Number's ToString, or a Symbol's internal encoded key.
pub(crate) struct KeyText {
    bytes: [u8; 24],
    start: u8,
}

impl KeyText {
    /// The decimal text of an integral Number: what ToString (and so ToPropertyKey) produces
    /// for it. Only exact integers below 2^53 in magnitude qualify (their text has no exponent
    /// or fraction); `-0` is `"0"`.
    pub(crate) fn integer(n: f64) -> Option<KeyText> {
        const LIMIT: f64 = 9_007_199_254_740_992.0; // 2^53
        if n.abs() >= LIMIT || n.trunc() != n {
            return None;
        }
        let mut text = KeyText::digits(n.abs() as u64);
        if n < 0.0 && text.as_str() != "0" {
            text.start -= 1;
            text.bytes[text.start as usize] = b'-';
        }
        Some(text)
    }

    /// The encoded key of the Symbol with this id (see `Interp::sym_key`).
    pub(crate) fn symbol(id: u64) -> KeyText {
        let mut text = KeyText::digits(id);
        text.start -= 1;
        text.bytes[text.start as usize] = 0;
        text
    }

    fn digits(mut value: u64) -> KeyText {
        let mut bytes = [0u8; 24];
        let mut start = bytes.len();
        loop {
            start -= 1;
            bytes[start] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        KeyText {
            bytes,
            start: start as u8,
        }
    }

    pub(crate) fn as_str(&self) -> &str {
        // SAFETY: only ASCII digits, '-' and NUL are written from `start` on.
        unsafe { std::str::from_utf8_unchecked(&self.bytes[self.start as usize..]) }
    }
}

/// The property key for array index `n`.
#[inline]
pub(crate) fn index_key(n: usize) -> PropKey {
    PropKey::index(u32::try_from(n).expect("an array index"))
}

/// The `"length"` / `"name"` / `"prototype"` / `"constructor"` keys every function object
/// carries, static names.
pub(crate) fn fn_key(i: usize) -> PropKey {
    use crate::atom::names;
    [
        names::LENGTH,
        names::NAME,
        names::PROTOTYPE,
        names::CONSTRUCTOR,
    ][i]
        .key()
}

impl std::fmt::Debug for Props {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Props")
            .field("entries", &self.entries.len())
            .field("shape", &self.shape)
            .finish()
    }
}

impl Default for Props {
    fn default() -> Self {
        Self::new()
    }
}

impl Props {
    pub(crate) fn new() -> Props {
        Self::with_capacity(0)
    }

    pub(crate) fn with_capacity(capacity: usize) -> Props {
        Self::with_entries(NamedEntries::with_capacity(capacity))
    }

    pub(crate) fn with_layout(capacity: usize, layout: Option<PropertyLayout>) -> Props {
        Self::with_entries(NamedEntries {
            fields: std::mem::ManuallyDrop::new(Fields::with_capacity(capacity)),
            layout,
        })
    }

    /// Cached layouts are immutable before their first instance is created.
    pub(crate) fn mark_layout_shared(&self) {
        if let Some(layout) = &self.entries.layout {
            layout.mark_shared();
        }
    }

    pub(crate) fn shared_layout(&self) -> Option<&PropertyLayout> {
        self.entries.layout.as_ref()
    }

    /// No indexed/Symbol/private storage or filtered/accessor descriptors: insertion order is
    /// [[OwnPropertyKeys]] order and every Get observes an own data value without author code.
    pub(crate) fn named_data_keys(&self) -> Option<&[PropKey]> {
        if self.elems.has_words()
            || self.elem_mode()
            || self.has_far()
            || self.entries.iter().any(|(key, property)| {
                !property.enumerable() || property.accessor() || key.is_symbol()
            })
        {
            return None;
        }
        Some(self.entries.keys())
    }

    /// CopyDataProperties for an ordinary named data record. Validate every descriptor before
    /// writing: an enumerable getter would make later descriptor reads observable. The caller
    /// proves ordinary internal methods and distinct source/target Objects.
    pub(crate) fn try_copy_named_data_from(&mut self, source: &Props) -> bool {
        if source.named_data_keys().is_none() {
            return false;
        }
        if source.entries.is_empty() {
            return true;
        }
        if self.entries.is_empty()
            && self.elems.0.is_none()
            && !self.elem_mode()
            && source
                .entries
                .fields
                .iter()
                .all(|property| property.class() == PropClass::DataWritable)
        {
            // Shapes encode (key, class): sharing the source shape is sound only when
            // every copied field's class matches its plain replacement (all data
            // writable). Otherwise the per-key insert path below rebuilds the correct
            // all-writable chain. In particular a frozen source must not produce frozen
            // copied properties — nor may its non-writable shape be reused for them.
            self.note_structural();
            self.entries.layout = source.entries.layout.clone();
            self.entries.fields.reserve_exact(source.entries.len());
            self.entries.fields.extend(
                source
                    .entries
                    .fields
                    .iter()
                    .map(|property| Property::plain(property.value())),
            );
            self.shape = source.shape;
            self.len_slot.set(source.len_slot.get());
            self.proto_slot.set(source.proto_slot.get());
            self.ensure_fallback_index();
        } else {
            for (key, property) in source.entries.iter() {
                self.insert(key, Property::plain(property.value()));
            }
        }
        true
    }

    /// Supply keys to a fresh, still-empty receiver after its initializer plan is validated.
    /// The existing field capacity remains usable; no predicted property becomes observable.
    pub(crate) fn predict_empty_layout(&mut self, layout: &PropertyLayout) {
        assert_eq!(self.shape, SHAPE_EMPTY);
        assert!(self.entries.is_empty());
        self.entries.fields.reserve_exact(layout.len());
        self.entries.layout = Some(layout.clone());
    }

    /// Requested bytes in allocations owned directly by this property map. The `Props` body is
    /// already part of its containing `Object`; this counts only backing allocations and boxed
    /// cold storage. `exact` is false when a standard-library hash table owns opaque bucket
    /// storage whose requested allocation size Rust does not expose.
    pub(crate) fn retained_requested_storage_bytes(&self) -> (usize, bool) {
        let mut bytes = self
            .entries
            .heap_capacity()
            .saturating_mul(std::mem::size_of::<Property>());
        let mut exact = true;
        for property in self.entries.fields.iter() {
            bytes = bytes.saturating_add(property.retained_requested_storage_bytes());
        }
        if let Some(words) = self.elems.words() {
            // The block: front room, header, words and remembered record.
            bytes = bytes.saturating_add(words.block_bytes());
        }
        if let Some(index) = self.elems.index() {
            bytes = bytes
                .saturating_add(std::mem::size_of::<KeyIndex>())
                .saturating_add(index.bytes());
        }
        if self.elems.symbols().is_some() {
            bytes = bytes
                .saturating_add(std::mem::size_of::<crate::fasthash::FastMap<u64, SymbolRef>>());
            exact = false;
        }
        (bytes, exact)
    }

    fn with_entries(entries: NamedEntries) -> Props {
        debug_assert!(entries.is_empty());
        Props {
            entries,
            shape: SHAPE_EMPTY,
            elems: DenseStorage::default(),
            flags: Cell::new(OBJECT_FLAGS_INITIAL),
            proto_slot: std::cell::Cell::new(SlotMemo::NONE),
            len_slot: std::cell::Cell::new(SlotMemo::NONE),
        }
    }

    /// Whether `bit` of the flag byte is set.
    #[inline]
    fn flag(&self, bit: u8) -> bool {
        self.flags.get() & bit != 0
    }
    #[inline]
    fn set_flag(&self, bit: u8, on: bool) {
        let flags = self.flags.get();
        self.flags.set(if on { flags | bit } else { flags & !bit });
    }
    #[inline]
    fn proto_flag(&self) -> bool {
        self.flag(FLAG_PROTO)
    }
    #[inline]
    fn set_proto_flag(&self, on: bool) {
        self.set_flag(FLAG_PROTO, on);
    }
    #[inline]
    fn has_far(&self) -> bool {
        self.flag(FLAG_HAS_FAR)
    }
    #[inline]
    fn set_has_far(&self, on: bool) {
        self.set_flag(FLAG_HAS_FAR, on);
    }
    #[inline]
    fn elem_mode(&self) -> bool {
        self.flag(FLAG_ELEM_MODE)
    }
    #[inline]
    fn set_elem_mode(&self, on: bool) {
        self.set_flag(FLAG_ELEM_MODE, on);
    }

    /// Construct a dense array by copying canonical words from a JIT argument window.
    ///
    /// # Safety
    /// `items..items+len` contains immutable initialized execution words valid through this call.
    pub(crate) unsafe fn packed_array_from_raw(items: *const Value, len: usize) -> Props {
        // SAFETY: the caller's `len` initialized words; a Value is its word (repr transparent).
        let words = unsafe { std::slice::from_raw_parts(items.cast::<u64>(), len) };
        Self::packed_array_words(word_block::Words::from_words(words), len)
    }

    /// [`Props::packed_array`] of a list in hand: its words are copied in one pass.
    pub(crate) fn packed_array_from_slice(items: &[Value]) -> Props {
        // SAFETY: a Value is its word (repr transparent over u64).
        let words =
            unsafe { std::slice::from_raw_parts(items.as_ptr().cast::<u64>(), items.len()) };
        Self::packed_array_words(word_block::Words::from_words(words), items.len())
    }

    fn packed_array_words(words: word_block::Words, len: usize) -> Props {
        let shape = array_length_shape();
        Self::packed_array_named_words(words, len, shape, array_length_layout(), std::iter::empty())
    }

    /// CreateArrayFromList / ArrayCreate (ECMA-262 e28783d5). Host/runtime lists
    /// use the same canonical contiguous storage as literal operands. Values
    /// move into plain own properties; no getters, conversions or species run.
    pub(crate) fn packed_array(items: impl ExactSizeIterator<Item = Value>) -> Props {
        Self::packed_array_values(items)
    }

    fn packed_array_values(items: impl ExactSizeIterator<Item = Value>) -> Props {
        let shape = array_length_shape();
        Self::packed_array_named(items, shape, array_length_layout(), std::iter::empty())
    }

    /// The map of an array of `items`, packed, whose named properties are ArrayCreate's `length`
    /// and then plain data properties valued `named`: `layout` lists their keys, and `shape` is
    /// the map's shape.
    #[inline(always)]
    fn packed_array_named(
        items: impl ExactSizeIterator<Item = Value>,
        shape: u32,
        layout: PropertyLayout,
        named: impl ExactSizeIterator<Item = Value>,
    ) -> Props {
        let len = items.len();
        Self::packed_array_named_words(
            word_block::Words::from_values(items),
            len,
            shape,
            layout,
            named,
        )
    }

    /// [`Props::packed_array_named`] with the element words built: `len` of them.
    #[inline(always)]
    fn packed_array_named_words(
        words: word_block::Words,
        len: usize,
        shape: u32,
        layout: PropertyLayout,
        named: impl ExactSizeIterator<Item = Value>,
    ) -> Props {
        debug_assert_eq!(words.len(), len);
        let length = Property::data(Value::Num(len as f64), true, false, false);
        Props {
            entries: NamedEntries {
                fields: std::mem::ManuallyDrop::new(Fields::from_iter(
                    std::iter::once(length).chain(named.map(Property::plain)),
                )),
                layout: Some(layout),
            },
            shape,
            // An empty array keeps the static empty block until its first element.
            elems: DenseStorage(None, None, words),
            flags: Cell::new(OBJECT_FLAGS_INITIAL | FLAG_ELEM_MODE),
            proto_slot: Cell::new(SlotMemo::NONE),
            len_slot: Cell::new(SlotMemo::of(0)),
        }
    }

    /// Instantiate a compiler-proved plain-data object template with its final values.
    ///
    /// Cloning the whole template would clone every placeholder [`Value`] and then drop it
    /// again as the caller overwrote each slot. Object-heavy parsers do this millions of times.
    /// The key/shape and lookup sidecars are the reusable part; plain property descriptors are
    /// cheaper and safer to construct directly around the moved values.
    pub(crate) fn instantiate_plain<I>(&self, values: I) -> Props
    where
        I: ExactSizeIterator<Item = Value>,
    {
        assert_eq!(values.len(), self.entries.len(), "object-template arity");
        // The copied shape records the template's classes; the instantiated fields are
        // all plain (data writable), so the template must be too.
        debug_assert!(
            self.entries
                .fields
                .iter()
                .all(|property| property.class() == PropClass::DataWritable),
            "plain object template with a non-writable or accessor field"
        );
        let entries = NamedEntries {
            fields: std::mem::ManuallyDrop::new(values.map(Property::plain).collect()),
            layout: self.entries.layout.clone(),
        };
        Props {
            entries,
            // The template's map bits; a new object's own.
            flags: Cell::new(
                self.flags.get() & (FLAG_HAS_FAR | FLAG_ELEM_MODE) | OBJECT_FLAGS_INITIAL,
            ),
            shape: self.shape,
            elems: self.elems.clone(),
            proto_slot: std::cell::Cell::new(self.proto_slot.get()),
            len_slot: std::cell::Cell::new(self.len_slot.get()),
        }
    }

    /// The map of an array of `items` that then gained the named properties of `self`, an array
    /// template: an empty array's map (ArrayCreate's `length`) into which plain data properties
    /// (writable, enumerable, configurable) were inserted. `named` gives their values, in the
    /// template's key order. An array's shape records its named keys only, so the template's
    /// shape is the one that inserting the same keys gives an array of any elements; its key
    /// layout is shared by every instance, where inserting the keys copies the shared `length`
    /// layout into a new private one per array. An instance that later gains, loses or redefines
    /// a named property diverges by the ordinary copy-on-write and shape-transition paths, as
    /// any map sharing a layout does.
    pub(crate) fn packed_array_from_template(
        &self,
        items: impl ExactSizeIterator<Item = Value>,
        named: impl ExactSizeIterator<Item = Value>,
    ) -> Props {
        assert_eq!(named.len() + 1, self.entries.len(), "array-template arity");
        debug_assert!(
            self.elem_mode()
                && !self.elems.has_words()
                && self
                    .entries
                    .keys()
                    .first()
                    .is_some_and(|&key| key == crate::atom::names::LENGTH.key())
                && self.entries.iter().skip(1).all(|(key, property)| {
                    key.as_index().is_none()
                        && property.class() == PropClass::DataWritable
                        && property.enumerable()
                        && property.configurable()
                }),
            "array template: an empty array's `length` and plain named properties"
        );
        let layout = self
            .entries
            .layout
            .clone()
            .expect("an array template lists its keys");
        Self::packed_array_named(items, self.shape, layout, named)
    }

    /// Grow tiny property maps exactly: `Vec`'s default first allocation has room for four
    /// 16-byte fields, while one- and two-property objects dominate real heaps. Past two entries
    /// resume geometric growth so larger maps retain amortized insertion.
    #[inline]
    fn reserve_entry(&mut self) {
        if self.entries.len() == self.entries.capacity() {
            let predicted = self.entries.layout.as_ref().map_or(0, |keys| keys.len());
            let additional = if predicted > self.entries.len() {
                predicted - self.entries.len()
            } else if self.entries.len() < 2 {
                1
            } else {
                self.entries.len()
            };
            self.entries.reserve_exact(additional);
        }
    }

    /// Reserve the element words of a fresh dense array (or arguments object) of known size; the
    /// named entries need only `length`.
    pub(crate) fn reserve_dense_exact(&mut self, len: usize) {
        if len != 0 {
            self.entries.reserve_exact(1); // own `length`
            self.elems.words_start().reserve_exact(len);
        }
    }

    /// Represent a very small holey array with keyless packed property slots. `Value::Empty`
    /// remains an absent property to every reflective operation, but a later indexed write can
    /// activate the already-allocated slot without allocating an index string or growing the
    /// entry/dense vectors. Keep this deliberately tiny: an untouched `new Array(n)` must not
    /// turn a length word into an unbounded allocation, and eight slots cap the speculative
    /// footprint at 128 bytes while becoming smaller than the classic representation once filled.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn reserve_small_holes(&mut self, len: usize) {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var_os("TRUST_JS_JIT_NO_PACKED_HOLES").is_none())
            || len == 0
            || len > 8
            || self.elems.has_words()
        {
            return;
        }
        let words = self.elems.words_start();
        words.reserve_exact(len);
        words.resize_with_holes(len);
    }

    /// The element words, for tests outside this module (the ARM64 JIT's).
    #[cfg(test)]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn words(&self) -> Option<&word_block::Words> {
        self.elems.words()
    }

    /// Make the element words' kind bits exact by a scan (`Words::reprove`) unless one since
    /// the last write already did, for a loop region's receiver proof. No JavaScript, GC,
    /// allocation or edge change.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn reprove_words(&mut self) {
        if let Some(words) = self.elems.words_mut().filter(|words| !words.scanned()) {
            words.reprove();
        }
    }

    /// Append one plain element as a word, with no decimal key or index-to-field map. Array
    /// exotic validation (length, extensibility and prototype setters) stays at the caller.
    /// ECMA-262 Array [[DefineOwnProperty]] / OrdinarySet, e28783d5.
    #[inline]
    fn push_packed_property(&mut self, property: Property) {
        debug_assert!(property.is_plain_element(), "a word is a plain element");
        self.elems.words_start().push(property.into_value());
    }

    /// Whether the map's first element word may start at `index`, on any map (every object has
    /// element storage, as in V8, SpiderMonkey and JSC): no words yet, no index key in the
    /// named entries (its element would be stored twice), and admitted by the sparse limit.
    fn can_start_packed_elements(&self, index: usize) -> bool {
        !self.has_far() && !self.elems.has_words() && self.words_admit(index)
    }

    /// Whether element `index`, which the words do not hold, may join them, holes before it
    /// (W2c step 6; V8's ShouldConvertToSlowElements, js-objects-inl.h): an index below their
    /// capacity always; one [`WORDS_MAX_GAP`] or more past it never; otherwise while V8's grown
    /// capacity, (index + 1) * 1.5 + 16, is at most [`WORDS_MAX_UNCHECKED`] words, or the named
    /// entries would not save a factor of 3 over it. V8 checks an old object past 500 words; a
    /// TRust JS object carries no age (its pages are young or old, not its cells), so every
    /// object has V8's young limit.
    fn words_admit(&self, index: usize) -> bool {
        let words = self.elems.words();
        let capacity = words.map_or(0, word_block::Words::capacity);
        if index < capacity {
            return true;
        }
        if index - capacity >= WORDS_MAX_GAP {
            return false;
        }
        let needed = index + 1;
        let grown = needed + needed / 2 + 16;
        grown <= WORDS_MAX_UNCHECKED
            || 3 * dictionary_words(words.map_or(0, word_block::Words::present)) > grown
    }

    /// After an index key joins an array's named entries (W2c step 6): when their count
    /// reaches a power of two (SpiderMonkey's measure, NativeObject.cpp, which keeps the count
    /// amortized), move every index entry into the words if each is a plain element and the
    /// named form saves less than half the space (V8's ShouldConvertToFastElements). Arrays
    /// only: V8 never converts an arguments object, and an ordinary object's index keys are
    /// part of its shape.
    fn maybe_densify(&mut self) {
        if !self.elem_mode() || !self.has_far() || !self.entries.len().is_power_of_two() {
            return;
        }
        let mut count = 0usize;
        let mut span = 0usize;
        for (key, property) in self.entries.iter() {
            if let Some(n) = key.as_index() {
                if !property.is_plain_element() {
                    return;
                }
                count += 1;
                span = span.max(n as usize + 1);
            }
        }
        let words = self.elems.words();
        let span = span.max(words.map_or(0, word_block::Words::len));
        let present = words.map_or(0, word_block::Words::present);
        if count == 0 || span > WORDS_MAX_SPAN || 2 * dictionary_words(present + count) < span {
            return;
        }
        self.note_structural();
        let moved: Vec<(usize, Value)> = self
            .entries
            .iter()
            .filter_map(|(key, property)| key.as_index().map(|n| (n as usize, property.value())))
            .collect();
        let words = self.elems.words_start();
        words.resize_with_holes(span);
        for (n, value) in moved {
            words.set(n, value);
        }
        self.entries.retain(|(key, _)| key.as_index().is_none());
        self.len_slot.set(SlotMemo::NONE);
        self.proto_slot.set(SlotMemo::NONE);
        self.elems.set_index(None);
        if self.entries.len() > INDEX_THRESHOLD {
            self.build_index();
        }
        self.set_has_far(false);
        let _ = self.length_property();
    }

    /// Create a bounded absent prefix on the first indexed write, never in response
    /// to the array's logical length. new Array(2**32-1) remains a small object.
    fn start_packed_elements(&mut self, index: usize, property: Property) {
        debug_assert!(self.can_start_packed_elements(index));
        if index != 0 {
            let words = self.elems.words_start();
            words.reserve_exact(index + 1);
            words.resize_with_holes(index);
        }
        self.push_packed_property(property);
    }

    /// Mark this map as an array's (see `elem_mode`). One-way, set when the owning object
    /// becomes `ObjectKind::Array`.
    #[inline]
    pub(crate) fn mark_array(&self) {
        self.set_elem_mode(true);
    }

    /// The `"length"` property, resolved through the `len_slot` memo (one compare, no hashing).
    /// `None` when there is no own `length`.
    pub(crate) fn length_property(&self) -> Option<&Property> {
        let length = crate::atom::names::LENGTH.key();
        if let Some(s) = self.len_slot.get().slot() {
            debug_assert!(matches!(self.entries.get(s), Some((k, _)) if k == length));
            debug_assert!(!self.elem_mode() || s == 0, "an Array's length is slot 0");
            return self.entries.fields.get(s);
        }
        let slot = self.find(length)?;
        // Generated code reads an Array's `length` at slot 0 (W2c step 7b, V8's descriptor 0):
        // ArrayCreate defines it first, and as a non-configurable property it is never removed.
        debug_assert!(
            !self.elem_mode() || slot == 0,
            "an Array's length is slot 0"
        );
        self.len_slot.set(SlotMemo::of(slot));
        Some(&self.entries.fields[slot])
    }

    /// Replace the value of an Array's own writable `length` data property, found through the
    /// `len_slot` memo. The caller has validated writability and the new length.
    pub(crate) fn set_array_length_value(&mut self, length: f64) {
        let key = crate::atom::names::LENGTH.key();
        let slot = if let Some(slot) = self.len_slot.get().slot() {
            slot
        } else {
            let slot = self.find(key).expect("array has own length");
            self.len_slot.set(SlotMemo::of(slot));
            slot
        };
        debug_assert!(matches!(self.entries.get(slot), Some((k, _)) if k == key));
        self.entries.fields[slot].set_value(Value::Num(length));
    }

    /// Mark this object as a live prototype (see `proto_flag`).
    #[inline]
    pub(crate) fn mark_proto(&self) {
        self.set_proto_flag(true);
    }

    /// Bump the creation-IC epoch if this object is a marked prototype (called by every
    /// structural mutation).
    #[inline]
    fn note_structural(&self) {
        if self.proto_flag() {
            bump_proto_epoch();
        }
    }

    /// Whether indexed elements live in the keyless packed representation (no index keys among
    /// the named entries).
    pub(crate) fn has_packed_elements(&self) -> bool {
        self.elems.has_words()
    }

    /// This map's shape id — the inline cache's structural validation token (see the `shape` field).
    #[inline]
    pub(crate) fn shape(&self) -> u32 {
        self.shape
    }

    /// Exercise the exhausted identity path without resetting the process-wide allocator or
    /// changing identities owned by other tests/Agents. This only withdraws this map's proof.
    #[cfg(test)]
    pub(crate) fn force_uncacheable_shape_for_test(&mut self) {
        self.shape = SHAPE_UNCACHEABLE;
    }

    /// Borrow the immutable key layout, live prefix and attributes for enumeration. Numeric
    /// shape IDs are globally unique now, but layout identity remains valid even after shape
    /// identity exhaustion and avoids retaining any source object or prototype chain.
    /// Empty prefixes need no layout pin; predictions do not create observable properties.
    pub(crate) fn enumeration_layout(&self) -> Option<(Option<&PropertyLayout>, u8, u64)> {
        if self.elem_mode() || self.elems.has_words() || self.entries.len() > 64 {
            return None;
        }
        let layout = if self.entries.is_empty() {
            None
        } else {
            Some(
                self.entries
                    .layout
                    .as_ref()
                    .filter(|keys| keys.len() >= self.entries.len())?,
            )
        };
        let mut enumerable = 0;
        for (index, property) in self.entries.fields.iter().enumerate() {
            enumerable |= u64::from(property.enumerable()) << index;
        }
        Some((layout, self.entries.len() as u8, enumerable))
    }
    /// The named-field count and capacity, as the JIT's storage emitters read them.
    #[cfg(test)]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn named_len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn named_capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Final named-property count of a small ordinary instance. The construct JIT records this
    /// after a successful call so forwarding constructors whose own bytecode has no direct
    /// `this.x` stores can reserve the right capacity on later allocations.
    pub(crate) fn observed_instance_capacity(&self) -> usize {
        if self.elems.0.is_none() && self.entries.len() <= 16 {
            self.entries.len()
        } else {
            0
        }
    }

    /// The own property for canonical index `n`: its element word, or the named entry that holds
    /// it (only with `has_far`); `None` when the map has no such own property.
    #[inline]
    pub(crate) fn get_index(&self, n: u32) -> Option<Element<'_>> {
        if let Some(words) = self.elems.words() {
            if let Some(value) = words.get(n as usize) {
                return Some(Element::word(value, words.integrity()));
            }
        }
        if !self.has_far() {
            return None;
        }
        self.find(PropKey::index(n))
            .map(|slot| Element::entry(&self.entries.fields[slot]))
    }

    /// Own element `n`'s value when it is a data property (a word at any integrity level, or a
    /// named data entry); `None` when it is absent or an accessor. A read needs no attributes,
    /// so a word answers without an [`Element`], whose level a caller would have to tell apart.
    #[inline]
    pub(crate) fn data_element(&self, n: u32) -> Option<Value> {
        if let Some(value) = self.elems.words().and_then(|words| words.get(n as usize)) {
            return Some(value);
        }
        if !self.has_far() {
            return None;
        }
        self.find(PropKey::index(n))
            .map(|slot| &self.entries.fields[slot])
            .filter(|property| !property.accessor())
            .map(Property::value)
    }

    /// Overwrite dense element `n`'s value. `Err` hands the value
    /// back: no such element, or it isn't a plain writable data property — the caller runs the
    /// generic path.
    #[inline]
    pub(crate) fn set_index_value(&mut self, n: u32, v: Value) -> Result<(), Value> {
        if let Some(words) = self
            .elems
            .words_mut()
            .filter(|words| words.integrity() & word_block::KIND_FROZEN == 0)
        {
            // An existing word only: a hole is an absent element (the prototype chain's, or a
            // named entry's after a non-plain define), which the caller's slow path handles.
            if words.get(n as usize).is_none() {
                return Err(v);
            }
            words.set(n as usize, v);
            return Ok(());
        }
        Err(v)
    }

    /// Record a fresh named entry at `slot`: an index key there sets `has_far`, so lookups of
    /// that index consult the named entries (find, get_key).
    fn note_inserted(&mut self, slot: usize) {
        if self.entries.keys()[slot].as_index().is_some() {
            self.set_has_far(true);
        }
    }

    /// Whether an array map's insertion should build its own key index: past
    /// [`INDEX_THRESHOLD`] named keys, index keys among them (an ordinary map's index is its
    /// layout's or the fallback, `ensure_fallback_index`).
    fn should_build_index(&self) -> bool {
        self.elem_mode() && self.entries.len() + 1 > INDEX_THRESHOLD
    }

    /// Dense tail append: insert element `n` as a word when `n` is exactly the words' end and no
    /// index key lives in the named entries (`has_far`) — which together prove the key is
    /// absent, so the existence lookup of [`Props::insert`] is skipped. The shape is untouched
    /// (words are not in it). `Err` (nothing changed) when the gates don't hold; the caller runs
    /// the generic path. Array validation (length, extensibility) stays at the caller.
    pub(crate) fn try_append_element(&mut self, n: u32, prop: Property) -> Result<(), Property> {
        if n == 0 && prop.is_plain_element() && self.can_start_packed_elements(0) {
            self.note_structural();
            self.start_packed_elements(0, prop);
            return Ok(());
        }
        let Some(words) = self.elems.words() else {
            return Err(prop);
        };
        // A sealed or frozen block's object is non-extensible (W2c step 7a), which every caller
        // refuses before an append.
        debug_assert!(!words.sealed(), "an element appended to a sealed block");
        let len = words.len();
        if self.has_far() || n as usize != len || !prop.is_plain_element() {
            return Err(prop);
        }
        self.note_structural();
        self.push_packed_property(prop);
        Ok(())
    }

    /// Insert an absent canonical index directly into the element words, including a bounded
    /// run of holes. The caller has already proved ordinary Array prototype semantics. This is
    /// the numeric-key counterpart of `insert`: it avoids parsing/comparing a decimal key we
    /// already know.
    pub(crate) fn try_define_dense_element(
        &mut self,
        n: u32,
        prop: Property,
    ) -> Result<(), Property> {
        if prop.is_plain_element() && self.can_start_packed_elements(n as usize) {
            self.note_structural();
            self.start_packed_elements(n as usize, prop);
            return Ok(());
        }
        if self.has_far() || !prop.is_plain_element() {
            return Err(prop);
        }
        let Some(words) = self.elems.words() else {
            return Err(prop);
        };
        // A sealed or frozen block's object is non-extensible (W2c step 7a), which every caller
        // refuses before it defines an element.
        debug_assert!(!words.sealed(), "an element defined in a sealed block");
        let index = n as usize;
        let len = words.len();
        if words.get(index).is_some() || !self.words_admit(index) {
            return Err(prop);
        }
        self.note_structural();
        let words = self.elems.words_mut().unwrap();
        if index < len {
            words.set(index, prop.into_value());
        } else {
            words.resize_with_holes(index);
            words.push(prop.into_value());
        }
        Ok(())
    }

    pub(crate) fn append_element(&mut self, n: u32, prop: Property) -> bool {
        self.try_append_element(n, prop).is_ok()
    }

    /// Dense tail pop: remove element `n` (the array's last) when it is the last word and no
    /// index key lives in the named entries. An O(1) pop: no entry shift, no re-index, no shape
    /// change (words are not in the shape). `Some(value)` = removed; `None` = gates failed,
    /// nothing changed, caller goes generic.
    pub(crate) fn pop_last_element(&mut self, n: u32) -> Option<Value> {
        if self.has_far() || !self.elem_mode() {
            return None;
        }
        let words = self.elems.words().filter(|words| !words.sealed())?;
        if n as usize + 1 != words.len() || words.as_slice().last() == Some(&PACK_EMPTY) {
            return None;
        }
        self.note_structural();
        let word = self.elems.words_mut().unwrap().pop()?;
        // SAFETY: a used word is a Value's bits.
        Some(unsafe { Value::from_bits(word) })
    }

    /// The packed element slots of an Array that keeps every index in element storage (no index
    /// far past the others lives in the named map), for a search that reads them in place:
    /// index `n` is slot `n`, a hole is an empty slot ([`Property::is_empty`]), and no index at
    /// or past the slice's end is an own property. `None` for any other representation.
    /// The used element words of an Array whose every element is a word (no far or non-plain
    /// index in the named entries), for a search that compares them in place; a hole is
    /// [`PACK_EMPTY`].
    pub(crate) fn element_words(&self) -> Option<&[u64]> {
        if !self.elem_mode() || self.has_far() {
            return None;
        }
        self.elems.words().map(word_block::Words::as_slice)
    }

    /// The packed elements of an Array whose `length` is the writable data value `len` and
    /// whose every index below it holds a plain element ([`Property::is_plain_element`]), for
    /// an edit that leaves them plain and contiguous. The caller proves ordinary Array internal
    /// methods and that the Array.prototype algorithm being batched only gets, sets and deletes
    /// such elements (plus appends, for which it proves extensibility and prototypes without
    /// indexed properties): its effect is then exactly the edited list. The caller records the
    /// new length with [`Self::set_array_length_value`]. `None` changes nothing.
    pub(crate) fn plain_elements_mut(&mut self, len: usize) -> Option<&mut word_block::Words> {
        if len == 0 || len > u32::MAX as usize || !self.elem_mode() || self.has_far() {
            return None;
        }
        let length = self.length_property()?;
        if length.accessor() || !length.writable() || length.number_value() != Some(len as f64) {
            return None;
        }
        let words = self.elems.words().filter(|words| !words.sealed())?;
        if words.len() != len || !words_have_no_holes(words) {
            return None;
        }
        self.note_structural();
        self.elems.words_mut()
    }

    /// ECMA-262 Array.prototype.shift / OrdinarySetWithOwnDescriptor (e28783d5,
    /// #sec-array.prototype.shift): batch only an entirely unobservable own-data
    /// operation. The caller proves ordinary Array internal methods. A miss changes
    /// no property, so the generic algorithm retains its partial effects
    /// and abrupt completions. `length` is included in the proof and update.
    ///
    /// `holes_read_absent`: the caller also proved that no prototype of the Array has an
    /// indexed property and that the Array is extensible. A hole then reads as undefined,
    /// reports absent to HasProperty and takes CreateDataProperty with no inherited setter, so
    /// packed elements with holes shift in place too: each element and each hole moves down
    /// one index, which is what the spec's HasProperty, Set and DeletePropertyOrThrow sequence
    /// leaves, and a leading hole returns undefined.
    pub(crate) fn shift_dense_array(
        &mut self,
        len: usize,
        holes_read_absent: bool,
    ) -> Option<Value> {
        if len == 0 || len > u32::MAX as usize || !self.elem_mode() || self.has_far() {
            return None;
        }
        let length = self.length_property()?;
        if !length.writable() || length.number_value() != Some(len as f64) {
            return None;
        }
        // Every word is a plain element or a hole, so each moves down one index with its
        // attributes; holes need the caller's proof that they read as absent.
        let words = self.elems.words().filter(|words| !words.sealed())?;
        if words.len() != len || (!holes_read_absent && !words_have_no_holes(words)) {
            return None;
        }
        self.note_structural();
        let first = self.elems.words_mut().unwrap().shift().unwrap();
        self.set_array_length_value((len - 1) as f64);
        // SAFETY: a used word is a Value's bits.
        let first = unsafe { Value::from_bits(first) };
        Some(if first.is_empty() {
            Value::Undefined
        } else {
            first
        })
    }

    /// ECMA-262 #sec-array.prototype.unshift (e28783d5): a plain contiguous own
    /// element list may prepend owners after the caller proves extensibility and
    /// absence of indexed prototype properties. Shifted destinations retain plain
    /// attributes; cached proof avoids rescanning the list on each prepend.
    pub(crate) fn unshift_dense_array(&mut self, len: usize, items: &[Value]) -> bool {
        let Some(next) = len
            .checked_add(items.len())
            .filter(|&n| n <= u32::MAX as usize)
        else {
            return false;
        };
        if !self.elem_mode() || self.has_far() {
            return false;
        }
        let Some(length) = self.length_property() else {
            return false;
        };
        if !length.writable() || length.number_value() != Some(len as f64) {
            return false;
        }
        // A sealed or frozen block takes no element (W2c step 7a); a map without words has
        // no integrity bits.
        if (len != 0 || self.elems.has_words())
            && self.elems.words().is_none_or(|words| {
                words.len() != len || words.sealed() || !words_have_no_holes(words)
            })
        {
            // A word array whose every element is present: the destinations keep plain
            // attributes, as the spec's Set and CreateDataProperty leave them.
            return false;
        }
        self.note_structural();
        self.elems.words_start().unshift(items);
        self.set_array_length_value(next as f64);
        true
    }

    /// The entry slot for `key`. Small maps (≤ [`INDEX_THRESHOLD`] entries — most objects) have
    /// no hash index at all: lookup is a short linear scan and inserts never hash or rehash.
    /// The index is built once when a map grows past the threshold and is authoritative from
    /// then on (an emptied-but-once-large map keeps using it).
    #[inline(always)]
    fn find(&self, key: PropKey) -> Option<usize> {
        // An index key lives in the named entries only with `has_far` (SpiderMonkey's
        // `Indexed`): every other element is a word or absent, which the callers answer. With
        // it set, the index key is found as any key is (the map's index, else a short scan).
        let array_index = key.as_index();
        if array_index.is_some() && !self.has_far() {
            return None;
        }
        // `length` and `prototype` are the hottest keys in array-heavy / allocation-heavy code
        // (every push/pop/length read; every `new`); their slots are memoized — answer without
        // hashing or scanning.
        let (length, prototype) = (
            crate::atom::names::LENGTH.key(),
            crate::atom::names::PROTOTYPE.key(),
        );
        if key == length {
            if let Some(s) = self.len_slot.get().slot() {
                debug_assert!(matches!(self.entries.get(s), Some((k, _)) if k == length));
                return Some(s);
            }
        } else if key == prototype {
            if let Some(s) = self.proto_slot.get().slot() {
                debug_assert!(matches!(self.entries.get(s), Some((k, _)) if k == prototype));
                return Some(s);
            }
        }
        // Array shapes describe named keys only. The intrinsic length-only
        // shape therefore proves absence of EVERY other non-index key, even
        // when millions of index keys live in the named entries.
        // Computed method reads (`a[method](...)`) have no static-property IC:
        // scanning those elements here turned a linear push/XOR loop quadratic.
        // This proves only OrdinaryGetOwnProperty absence (ECMA-262 §10.1.5);
        // callers still walk the real prototype chain, preserving getters,
        // overrides and Proxy traps. Named inserts change the shape, and far
        // canonical indices must retain their ordinary map lookup.
        if self.entries.len() > INDEX_THRESHOLD
            && self.elem_mode()
            && array_index.is_none()
            && key != length
            && is_cacheable_shape(self.shape)
            && self.shape == array_length_shape()
        {
            return None;
        }
        // An ordinary map's immutable key layout owns a shared index once its live key sequence
        // crosses INDEX_THRESHOLD. The table may include predicted tail keys; a slot beyond the
        // field vector is not an own property. This miss is conclusive because layout keys are
        // always the exact live prefix followed only by predictions.
        if !self.elem_mode() && self.entries.len() > INDEX_THRESHOLD {
            if let Some(layout) = self
                .entries
                .layout
                .as_ref()
                .filter(|layout| layout.has_index())
            {
                let slot = layout
                    .indexed_slot(key)
                    .filter(|slot| *slot < self.entries.len());
                if let Some(slot) = slot {
                    debug_assert!(
                        matches!(self.entries.get(slot), Some((stored, _)) if stored == key)
                    );
                    if key == length {
                        self.len_slot.set(SlotMemo::of(slot));
                    } else if key == prototype {
                        self.proto_slot.set(SlotMemo::of(slot));
                    }
                }
                return slot;
            }
        }
        let found = match self.elems.index() {
            None => self.entries.keys().iter().position(|&k| k == key),
            Some(index) => index.get(key),
        };
        if let Some(s) = found {
            if key == length {
                self.len_slot.set(SlotMemo::of(s));
            } else if key == prototype {
                self.proto_slot.set(SlotMemo::of(s));
            }
        }
        found
    }
    /// [`Props::find`] of a key's text. A small map without an index compares its keys' text
    /// and makes no atom lookup; a larger one takes the text's atom once, and text with no atom
    /// in the active heap names no property (every named key was made one when inserted).
    #[inline(always)]
    fn find_text(&self, text: &str) -> Option<usize> {
        self.find_text_with(text, &mut None)
    }
    /// [`Props::find_text`] with the text's atom, if `atom` has it; a lookup of the atom fills
    /// it, so a walk over several maps (a prototype chain) looks the atom up once. An atom, once
    /// found, lasts as long as its heap.
    #[inline(always)]
    fn find_text_with(&self, text: &str, atom: &mut Option<PropKey>) -> Option<usize> {
        if let Some(n) = canonical_index(text) {
            return self.find(PropKey::index(n));
        }
        if self.entries.len() > INDEX_THRESHOLD || self.elems.index().is_some() {
            // The array-length shape proves every other named key absent (see `find`), with
            // no atom to look up.
            if self.entries.len() > INDEX_THRESHOLD
                && self.elem_mode()
                && text != "length"
                && is_cacheable_shape(self.shape)
                && self.shape == array_length_shape()
            {
                return None;
            }
            let found = match *atom {
                Some(key) => self.find(key),
                None => PropKey::lookup_text(text).and_then(|key| self.find(*atom.insert(key))),
            };
            #[cfg(debug_assertions)]
            if found.is_none() {
                self.check_keys_of_active_heap();
            }
            return found;
        }
        // As in `find`: the memoized `length` and `prototype` slots first.
        let memo = if text == "length" {
            Some(&self.len_slot)
        } else if text == "prototype" {
            Some(&self.proto_slot)
        } else {
            None
        };
        if let Some(slot) = memo.and_then(|memo| memo.get().slot()) {
            debug_assert!(matches!(self.entries.get(slot), Some((k, _)) if k.is(text)));
            return Some(slot);
        }
        let found = self
            .entries
            .keys()
            .iter()
            .position(|key| key.as_atom().is_some_and(|atom| atom.is(text)));
        if let (Some(memo), Some(slot)) = (memo, found) {
            memo.set(SlotMemo::of(slot));
        }
        found
    }
    /// Heap verification for a lookup of text that missed: the map's keys are the active heap's
    /// (I6), or text whose atom the map's heap has would miss here.
    #[cfg(debug_assertions)]
    fn check_keys_of_active_heap(&self) {
        for atom in self
            .entries
            .keys()
            .iter()
            .filter_map(|key| key.as_atom())
            .filter(|atom| !atom.is_static())
            .take(4)
        {
            assert!(
                crate::atom::lookup(&atom.text()) == Some(atom),
                "a property map of another heap read under the active one (I6): {atom:?}"
            );
        }
    }
    /// Build the hash index for every current entry (crossing the small-map threshold).
    fn build_index(&mut self) {
        let index = KeyIndex::of(self.entries.keys());
        self.elems.set_index(Some(index));
    }

    /// Recover the generic lookup table if a non-array map reached the indexed size without a
    /// usable layout index (for example shape-identity exhaustion). Common shaped objects keep
    /// the one index on their shared layout instead of one duplicate per receiver.
    fn ensure_fallback_index(&mut self) {
        if self.elem_mode() {
            return;
        }
        if self
            .entries
            .layout
            .as_ref()
            .is_some_and(|layout| layout.has_index())
        {
            self.elems.set_index(None);
        } else if self.entries.len() > INDEX_THRESHOLD && self.elems.index().is_none() {
            self.build_index();
        }
    }
    /// The property whose key's text is `key` (see [`Props::find_text`]).
    #[inline]
    pub(crate) fn get(&self, key: &str) -> Option<Element<'_>> {
        if let Some(n) = canonical_index(key) {
            return self.get_key(PropKey::index(n));
        }
        self.find_text(key)
            .map(|i| Element::entry(&self.entries.fields[i]))
    }
    /// A named key's property: never an element, so always a stored entry.
    pub(crate) fn get_named(&self, key: &str) -> Option<&Property> {
        debug_assert!(canonical_index(key).is_none(), "get_named of an index key");
        self.find_text(key).map(|i| &self.entries.fields[i])
    }
    pub(crate) fn get_key(&self, key: PropKey) -> Option<Element<'_>> {
        if let Some(n) = key.as_index() {
            if let Some(p) = self.get_index(n) {
                return Some(p);
            }
            // An index that is not a word lives in the named entries only with `has_far`:
            // without it a word miss proves absence; scanning every string-key entry is both
            // redundant and especially costly for a hole read from a large array.
            if !self.has_far() {
                return None;
            }
        }
        self.find(key)
            .map(|i| Element::entry(&self.entries.fields[i]))
    }
    /// Semantic lookup plus the named-entry slot when the result lives in `entries`, in a walk
    /// over several maps (see [`Props::get_walking`]). Dense indexed results have no named field
    /// slot. Used by opt-in feedback collection so it can retain location metadata without
    /// performing a second lookup.
    pub(crate) fn get_with_slot_walking(
        &self,
        key: &str,
        atom: &mut Option<PropKey>,
    ) -> Option<(Element<'_>, Option<usize>)> {
        if let Some(n) = canonical_index(key) {
            if let Some(property) = self.get_index(n) {
                return Some((property, None));
            }
            if !self.has_far() {
                return None;
            }
        }
        self.find_text_with(key, atom)
            .map(|slot| (Element::entry(&self.entries.fields[slot]), Some(slot)))
    }
    /// [`Props::get`] in a walk that reads `key` from several maps (a prototype chain): `atom`
    /// carries the text's atom from the map that looked it up to the next ones.
    #[inline]
    pub(crate) fn get_walking(&self, key: &str, atom: &mut Option<PropKey>) -> Option<Element<'_>> {
        if let Some(n) = canonical_index(key) {
            return self.get_key(PropKey::index(n));
        }
        self.find_text_with(key, atom)
            .map(|i| Element::entry(&self.entries.fields[i]))
    }
    /// The memoized own `prototype` slot, for guarded constructor fast paths.
    #[inline]
    pub(crate) fn prototype_slot(&self) -> Option<u32> {
        self.find(crate::atom::names::PROTOTYPE.key())
            .map(|slot| slot as u32)
    }
    /// Test-only mutable escape (descriptor fixtures simulate descriptor edits
    /// through it). Production descriptor changes go through `insert`; value-only writes
    /// go through `set_value`/`set_value_at`. Acquisition still demotes, matching the
    /// contract those fixtures must not accidentally rely on.
    #[cfg(test)]
    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut Property> {
        let text = key;
        let index = canonical_index(text);
        if let Some(n) = index {
            if let Some(value) = self.elems.words().and_then(|words| words.get(n as usize)) {
                // The escape may change the element's attributes, which a word does not keep:
                // the element leaves for the named entries first, as a non-plain one does (a
                // non-configurable define moves it), then takes back its plain attributes. A
                // sealed or frozen element leaves with its implied ones (W2c step 7a).
                let integrity = self.word_integrity();
                if integrity == 0 {
                    self.insert(PropKey::index(n), Property::data(value, true, true, false));
                    let slot = self.find(PropKey::index(n)).expect("the moved element");
                    self.entries.fields[slot].set_configurable(true);
                } else {
                    self.note_structural();
                    self.elems.words_mut().unwrap().move_out(n as usize);
                    self.insert(
                        PropKey::index(n),
                        Property::data(
                            value,
                            integrity & word_block::KIND_FROZEN == 0,
                            true,
                            false,
                        ),
                    );
                }
            }
        }
        match self.find_text(text) {
            Some(i) => {
                // Handing out `&mut Property` can change the descriptor class behind the
                // shape's back, so acquiring one for a shape-tracked entry demotes the
                // shape (array element entries are exempt, as in `remove`). Value-only
                // writers use `set_value`/`set_value_at` and pay nothing.
                if !(self.elem_mode() && index.is_some()) {
                    self.demote_shape();
                }
                Some(&mut self.entries.fields[i])
            }
            None => None,
        }
    }
    /// Mint a fresh unique shape: no inline cache recorded against the old id can hit
    /// again, so any proof it carried (slot, and eventually class) is abandoned.
    pub(crate) fn demote_shape(&mut self) {
        self.shape = shape_fresh();
    }
    /// Value-only write to an existing property (the caller has established it is a
    /// writable data property): the descriptor — and therefore the shape class — is
    /// untouched, so no demotion. Routes to packed element storage exactly like
    /// `get_mut`. Returns false when the key is absent.
    pub(crate) fn set_value(&mut self, key: &str, v: Value) -> bool {
        let text = key;
        if let Some(n) = canonical_index(text) {
            if let Some(words) = self
                .elems
                .words_mut()
                .filter(|words| words.get(n as usize).is_some())
            {
                words.set(n as usize, v);
                return true;
            }
        }
        match self.find_text(text) {
            Some(i) => {
                self.entries.fields[i].set_value(v);
                true
            }
            None => false,
        }
    }
    /// Value-only write at an already-resolved entries slot (the IC-store form of
    /// `set_value`). The caller has shape/key-validated the slot and established a
    /// writable data descriptor there.
    pub(crate) fn set_value_at(&mut self, slot: usize, v: Value) -> bool {
        match self.entries.fields.get_mut(slot) {
            Some(p) => {
                p.set_value(v);
                true
            }
            None => false,
        }
    }
    /// Move an existing property's value out (leaving `Empty`), descriptor untouched:
    /// the value-only counterpart used by the string-concatenation store path, which
    /// immediately writes a replacement value back through `set_value`.
    pub(crate) fn take_value(&mut self, key: &str) -> Option<Value> {
        let text = key;
        if let Some(n) = canonical_index(text) {
            // A word's value is copied out and its word kept: the caller writes the
            // replacement back through `set_value`.
            if let Some(value) = self.elems.words().and_then(|words| words.get(n as usize)) {
                return Some(value);
            }
        }
        let i = self.find_text(text)?;
        Some(self.entries.fields[i].take_value())
    }
    /// Merge `get`/`set` into an EXISTING accessor descriptor at `key` (the
    /// `__defineGetter__`/`__defineSetter__` and accessor-merge paths): the class stays
    /// Accessor, so no demotion. Returns false when there is no entry or it is a data
    /// descriptor — the caller then takes its ordinary insert path, which demotes.
    pub(crate) fn merge_accessor(
        &mut self,
        key: &str,
        get: Option<Value>,
        set: Option<Value>,
    ) -> bool {
        let Some(i) = self.find_text(key) else {
            return false;
        };
        let p = &mut self.entries.fields[i];
        if !p.accessor() {
            return false;
        }
        if get.is_some() {
            p.set_getter(get);
        }
        if set.is_some() {
            p.set_setter(set);
        }
        true
    }
    /// Install the deferred function-`prototype` thunk at a resolved slot: a value-only
    /// replacement (the `prototype` descriptor itself never changes), so no demotion.
    pub(crate) fn defer_function_prototype_at(
        &mut self,
        slot: usize,
        owner: &Gc,
        parent: Gc,
        template: &Props,
    ) -> bool {
        match self.entries.fields.get_mut(slot) {
            Some(p) => {
                p.defer_function_prototype(owner, parent, template);
                true
            }
            None => false,
        }
    }
    /// `Object.freeze`'s descriptor pass: every own data property becomes non-writable
    /// and every own property non-configurable. Demotes once when a shape-tracked
    /// entry actually flips class (writable -> non-writable).
    pub(crate) fn freeze_descriptors(&mut self) {
        if let Some(words) = self.elems.words() {
            // A frozen element is not plain: each word leaves for the named entries (until
            // step 7's SEALED and FROZEN kinds), frozen with them below.
            let elements: Vec<(u32, Value)> = (0..words.len())
                .filter_map(|n| words.get(n).map(|value| (n as u32, value)))
                .collect();
            self.elems.clear_words();
            for (n, value) in elements {
                self.insert(PropKey::index(n), Property::data(value, false, true, false));
            }
        }
        let mut class_changed = false;
        for p in self.entries.fields.iter_mut() {
            if !p.accessor() && p.writable() {
                class_changed = true;
            }
            p.set_writable(false);
            p.set_configurable(false);
        }
        if class_changed {
            self.demote_shape();
        }
    }
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
    pub(crate) fn contains_key(&self, key: PropKey) -> bool {
        self.get_key(key).is_some()
    }
    /// The `entries` slot for `key`, or `None`. Backs the bytecode property inline cache: a hit
    /// records the slot so the next access can skip the lookup (see `Interp::try_ic_get`).
    #[inline]
    pub(crate) fn slot_of(&self, key: &str) -> Option<usize> {
        self.find_text(key)
    }
    /// [`Props::slot_of`] of a key.
    #[inline]
    pub(crate) fn slot_of_key(&self, key: PropKey) -> Option<usize> {
        self.find(key)
    }
    /// The (key, property) at `slot`, or `None` if out of range. The caller re-checks the key —
    /// slots shift on `remove`, so a cached slot is only trusted after the key matches.
    #[inline]
    pub(crate) fn entry_at(&self, slot: usize) -> Option<(PropKey, &Property)> {
        self.entries.get(slot)
    }
    /// A live field at an already-resolved slot. Shape/memo-validated callers do not need to
    /// follow the key layout; array-holder ICs still use `entry_at` to re-check their key.
    #[inline]
    pub(crate) fn property_at(&self, slot: usize) -> Option<&Property> {
        self.entries.fields.get(slot)
    }
    /// Drop every property.
    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.note_structural();
        self.entries.clear();
        self.elems.clear();
        self.len_slot.set(SlotMemo::NONE);
        self.proto_slot.set(SlotMemo::NONE);
        self.set_has_far(false);
        self.shape = shape_fresh();
    }
    /// Append the next element word while *building a fresh array in order* (its elements so
    /// far are exactly the words 0..len, with no index key in the named entries): skips the
    /// canonical-index parse and the existence lookup. A word is a plain element.
    pub(crate) fn push_dense(&mut self, prop: Property) {
        debug_assert!(!self.has_far(), "push_dense beside a named index key");
        self.push_packed_property(prop);
    }

    /// Insert a key *known to be absent* (the caller shape-validated the map), landing on a
    /// *known* child shape: skips the existence scan. A predicted constructor layout also skips
    /// the transition table; otherwise a bounded small-layout lookup supplies shared keys.
    /// `new_shape` must be the transition recorded when this pair was first inserted normally.
    pub(crate) fn append_new(&mut self, key: PropKey, prop: Property, new_shape: u32) {
        self.elems.retain_symbol_key(key);
        self.note_structural();
        let slot = self.entries.len();
        if self.elems.index().is_some() {
            if let Some(index) = self.elems.index_mut() {
                index.insert(key, slot);
            }
        } else if self.should_build_index() {
            self.build_index();
            self.elems.index_mut().unwrap().insert(key, slot);
        }
        // If the constructor did not predict this key, reuse the small ordinary transition's
        // key layout. The creation IC still saves the semantic absence/prototype-chain walk.
        if !self.elem_mode()
            && !self.entries.predicts(key)
            && self.entries.len() < SHARED_LAYOUT_MAX_FIELDS
        {
            if let Some(layout) = shape_layout(new_shape) {
                self.entries.layout = Some(layout);
            }
        }
        self.shape = new_shape;
        self.reserve_entry();
        self.entries.push((key, prop));
        self.note_inserted(slot);
        self.ensure_fallback_index();
    }

    /// Build the named-property prefix of a brand-new ordinary object after creation ICs proved
    /// every key absent/non-indexed and supplied the complete shape chain. Up to the small-map
    /// threshold no dense/index sidecar or special-slot memo can be required, so the whole batch
    /// is just entry appends followed by its already-known final shape.
    pub(crate) fn append_proven_plain(&mut self, key: PropKey, prop: Property) {
        self.elems.retain_symbol_key(key);
        debug_assert!(self.entries.len() < INDEX_THRESHOLD);
        debug_assert!(key.as_index().is_none());
        debug_assert!(self.elems.0.is_none());
        self.reserve_entry();
        if self.entries.predicts(key) {
            self.entries.fields.push(prop);
        } else {
            self.entries.push((key, prop));
        }
    }

    /// The immutable initializer plan proved this exact ordered layout before the batch began.
    /// No key ownership or runtime key comparison is needed while moving its field values.
    pub(crate) fn append_initialized_field(&mut self, key: PropKey, prop: Property) {
        debug_assert!(self.entries.predicts(key));
        debug_assert!(key.as_index().is_none(), "an index key is an element word");
        self.elems.retain_symbol_key(key);
        let slot = self.entries.len();
        if let Some(index) = self.elems.index_mut() {
            index.insert(key, slot);
        }
        self.entries.fields.push(prop);
        self.ensure_fallback_index();
    }

    pub(crate) fn finish_proven_plain_shape(&mut self, shape: u32) {
        debug_assert!(!self.entries.is_empty());
        debug_assert!(self.entries.len() <= INDEX_THRESHOLD);
        self.shape = shape;
    }

    pub(crate) fn insert(&mut self, key: impl IntoPropKey, prop: Property) {
        let key = key.into_prop_key();
        self.elems.retain_symbol_key(key);
        if let Some(n) = key.as_index() {
            if prop.is_plain_element() && self.can_start_packed_elements(n as usize) {
                self.note_structural();
                self.start_packed_elements(n as usize, prop);
                return;
            }
        }
        if let (Some(n), Some(len)) = (
            key.as_index(),
            self.elems.words().map(word_block::Words::len),
        ) {
            let n = n as usize;
            // An index a named entry already holds keeps it there (replaced below), so no
            // element is ever stored twice.
            let named = self.has_far() && self.find(key).is_some();
            let integrity = self.word_integrity();
            let sealed_word = integrity & word_block::KIND_SEALED != 0
                && !named
                && self
                    .elems
                    .words()
                    .is_some_and(|words| words.get(n).is_some());
            if sealed_word {
                // An element of a sealed or frozen block (W2c step 7a). A define that keeps its
                // implied attributes (enumerable data, non-configurable, writable unless frozen)
                // changes at most its value (a frozen one's not at all: the caller found the
                // values the same); any other moves it to a named entry with the attributes the
                // define gives in full (V8 13.6 normalized a sealed array's elements with default
                // attributes, so it stopped being sealed).
                debug_assert!(!prop.is_empty(), "a sealed element is deleted");
                let frozen = integrity & word_block::KIND_FROZEN != 0;
                if !prop.accessor()
                    && prop.enumerable()
                    && !prop.configurable()
                    && prop.writable() != frozen
                {
                    if !frozen {
                        self.note_structural();
                        self.elems.words_mut().unwrap().set(n, prop.into_value());
                    }
                    return;
                }
                self.note_structural();
                self.elems.words_mut().unwrap().move_out(n);
            } else if prop.is_plain_element() && !named {
                if n < len {
                    self.note_structural();
                    self.elems.words_mut().unwrap().set(n, prop.into_value());
                    return;
                }
                if !self.has_far() && self.words_admit(n) {
                    self.note_structural();
                    let words = self.elems.words_mut().unwrap();
                    words.resize_with_holes(n);
                    words.push(prop.into_value());
                    return;
                }
            } else if n < len {
                // A non-plain element leaves the words for the named entries, its word a hole
                // (W2c's design 3, SpiderMonkey's indexed properties); an absent one is a hole.
                self.elems.words_mut().unwrap().move_out(n);
                if prop.is_empty() && !named {
                    self.note_structural();
                    return;
                }
            }
            self.set_has_far(true);
        }
        if let Some(i) = self.find(key) {
            // A whole-descriptor replacement that changes descriptor class cannot stay
            // on a shape that records the old class: demote to a fresh unique shape
            // (the removal precedent). Array element entries are exempt exactly as in
            // `remove` — their shape tracks named keys only. Class-preserving
            // replacements (including enumerable/configurable-only changes) keep it.
            if self.entries.fields[i].class() != prop.class()
                && !(self.elem_mode() && key.as_index().is_some())
            {
                self.shape = shape_fresh();
            }
            self.entries.fields[i] = prop;
        } else {
            self.note_structural();
            let slot = self.entries.len();
            if self.elems.index().is_some() {
                if let Some(index) = self.elems.index_mut() {
                    index.insert(key, slot);
                }
            } else if self.should_build_index() {
                self.build_index();
                self.elems.index_mut().unwrap().insert(key, slot);
            }
            if !(self.elem_mode() && key.as_index().is_some()) {
                let prefix = (!self.elem_mode()
                    && self.entries.len() < SHARED_LAYOUT_MAX_FIELDS
                    && !self.entries.predicts(key))
                .then(|| {
                    self.entries
                        .iter()
                        .map(|(name, property)| (name, property.class()))
                        .collect::<Vec<_>>()
                });
                let (shape, layout) =
                    shape_transition(self.shape, key, prop.class(), prefix.as_deref());
                self.shape = shape;
                if let Some(layout) = layout {
                    self.entries.layout = Some(layout);
                }
            }
            if key == crate::atom::names::LENGTH.key() {
                self.len_slot.set(SlotMemo::of(slot));
            } else if key == crate::atom::names::PROTOTYPE.key() {
                self.proto_slot.set(SlotMemo::of(slot));
            }
            self.reserve_entry();
            self.entries.push((key, prop));
            self.note_inserted(slot);
            self.ensure_fallback_index();
            if key.as_index().is_some() {
                self.maybe_densify();
            }
        }
    }
    /// Remove every canonical-index key `>= from` in one pass — array truncation
    /// (`arr.length = n`). Entries compact and the lookup/dense maps rebuild once: O(n) total,
    /// where the per-key [`Props::remove`] loop it replaces was O(n) *per key*.
    pub(crate) fn remove_indices_from(&mut self, from: usize) {
        let keep = |k: PropKey| match k.as_index() {
            Some(n) => (n as usize) < from,
            None => true,
        };
        let words_remove = self.elems.words().is_some_and(|words| {
            words.len() > from && words.as_slice()[from..].iter().any(|&w| w != PACK_EMPTY)
        });
        let entries_remove = !self.entries.iter().all(|(k, _)| keep(k));
        if !words_remove && !entries_remove {
            return;
        }
        self.note_structural();
        if let Some(words) = self.elems.words_mut() {
            words.truncate(from);
        }
        if !entries_remove {
            // Only element words went: every named entry keeps its slot, so the slot memos,
            // the key index and the shape (named keys only) stand, and an `a.length = 0`
            // refill appends in native code from its first word (W2c step 4). A set `has_far`
            // may have outlived its last named index, as the full path below rebuilds it.
            if self.has_far() {
                self.set_has_far(false);
                for slot in 0..self.entries.len() {
                    self.note_inserted(slot);
                }
            }
            return;
        }
        self.entries.retain(|(k, _)| keep(k));
        self.len_slot.set(SlotMemo::NONE);
        self.proto_slot.set(SlotMemo::NONE);
        self.elems.set_index(None);
        if self.entries.len() > INDEX_THRESHOLD
            && (self.elem_mode()
                || !self
                    .entries
                    .layout
                    .as_ref()
                    .is_some_and(|layout| layout.has_index()))
        {
            self.build_index();
        }
        // Forget the old conservative proof and let the surviving named keys reassert it.
        self.set_has_far(false);
        for slot in 0..self.entries.len() {
            self.note_inserted(slot);
        }
        // A removal shifts slots: it can't be a tree transition, so deopt to a fresh unique id.
        self.shape = shape_fresh();
    }

    pub(crate) fn remove(&mut self, key: &str) -> bool {
        let key = match canonical_index(key) {
            Some(n) => PropKey::index(n),
            None => match self.find_text(key) {
                Some(slot) => self.entries.keys()[slot],
                None => return false,
            },
        };
        self.remove_key(key)
    }

    pub(crate) fn remove_key(&mut self, key: PropKey) -> bool {
        if let (Some(n), Some(words)) = (key.as_index(), self.elems.words()) {
            if words.get(n as usize).is_some() {
                debug_assert!(words.integrity() == 0, "a sealed element is deleted");
                self.note_structural();
                self.elems
                    .words_mut()
                    .unwrap()
                    .set(n as usize, Value::Empty);
                return true;
            }
        }
        let Some(i) = self.find(key) else {
            return false;
        };
        // An index key found in the named entries is one `has_far` records (see `find`).
        let removed_far = key.as_index().is_some();
        self.note_structural();
        self.entries.remove(i);
        // Slots shifted — deopt to a fresh shape id (see remove_indices_from). Array maps skip
        // this for ELEMENT keys: their shape tracks named keys only, and array entry slots are
        // only ever trusted through key-checked ICs (IC_ARR_KEYCHK), which re-verify on hit.
        if !(self.elem_mode() && key.as_index().is_some()) {
            self.shape = shape_fresh();
        }
        self.len_slot.set(SlotMemo::NONE);
        self.proto_slot.set(SlotMemo::NONE);
        if let Some(index) = self.elems.index_mut() {
            index.remove(key);
            // Every later entry moved down one slot. The index covers every entry, so
            // adjust positions in place instead of rehashing and reinserting each later key:
            // removing the early keys of a large map (a Window bootstrap deleting its host
            // bindings from a global with a thousand properties) stays linear and cheap.
            index.close_gap(i);
        }
        if !self.elem_mode()
            && self
                .entries
                .layout
                .as_ref()
                .is_some_and(|layout| layout.has_index())
        {
            self.elems.set_index(None);
        }
        self.elems.release_symbol_key(key);
        if removed_far {
            self.recover_dense_coverage();
        }
        true
    }

    /// After a named index key's removal, whether any remains (`has_far`): a single sparse
    /// write must not permanently send a surviving array or object to its named lookups. Cap
    /// this extra work: large maps keep their conservative proof until a truncation rebuilds
    /// it, rather than rescanning on every deletion.
    fn recover_dense_coverage(&mut self) {
        const MAX_RECOVERY_ENTRIES: usize = 256;
        if self.entries.len() > MAX_RECOVERY_ENTRIES {
            return;
        }
        let remains = self
            .entries
            .keys()
            .iter()
            .any(|key| key.as_index().is_some());
        self.set_has_far(remains);
    }
    /// Ordinary property keys in insertion order. Engine state and private elements live in a
    /// separate slot map and never participate in ordinary key, descriptor or lookup operations.
    pub(crate) fn keys(&self) -> Vec<PropKey> {
        self.word_indices()
            .map(index_key)
            .chain(self.entries.keys().iter().copied())
            .collect()
    }
    /// The indices of the element words that are not holes.
    fn word_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.elems
            .words()
            .into_iter()
            .flat_map(|words| words.as_slice().iter().enumerate())
            .filter(|&(_, &word)| word != PACK_EMPTY)
            .map(|(n, _)| n)
    }
    /// Keys in spec [[OwnPropertyKeys]] order: array-index keys ascending, then other string keys
    /// in insertion order, then symbol keys in insertion order.
    pub(crate) fn ordered_keys(&self) -> Vec<PropKey> {
        let mut ints: Vec<u32> = Vec::new();
        let mut strs: Vec<PropKey> = Vec::new();
        let mut syms: Vec<PropKey> = Vec::new();
        ints.extend(self.word_indices().map(|n| n as u32));
        for &k in self.entries.keys() {
            if let Some(n) = k.as_index() {
                ints.push(n);
            } else if k.is_symbol() {
                syms.push(k);
            } else {
                strs.push(k);
            }
        }
        ints.sort_unstable();
        ints.into_iter()
            .map(PropKey::index)
            .chain(strs)
            .chain(syms)
            .collect()
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (PropKey, &Property)> {
        self.entries.iter()
    }

    /// Every live property value, including keyless packed elements (for GC tracing).
    pub(crate) fn values(&self) -> impl Iterator<Item = Element<'_>> {
        let integrity = self.word_integrity();
        self.word_values()
            .map(move |value| Element::word(value, integrity))
            .chain(self.entries.fields.iter().map(Element::entry))
    }

    /// The element words' values, holes left out: the elements no named entry holds.
    /// Managed-memory traversal visits named entries together with their keys, then uses this
    /// iterator so those entries are not scanned twice.
    pub(crate) fn word_values(&self) -> impl Iterator<Item = Value> + '_ {
        self.elems
            .words()
            .into_iter()
            .flat_map(|words| words.as_slice().iter())
            .filter(|&&word| word != PACK_EMPTY)
            // SAFETY: a used word is a Value's bits.
            .map(|&word| unsafe { Value::from_bits(word) })
    }

    /// All own array-index descriptors, independent of physical storage. Guards
    /// for OrdinarySet / indexed Get must not mistake a packed descriptor for
    /// an absent property. Named iteration intentionally excludes packed elements.
    pub(crate) fn indexed_properties(&self) -> impl Iterator<Item = Element<'_>> {
        let integrity = self.word_integrity();
        self.word_values()
            .map(move |value| Element::word(value, integrity))
            .chain(
                self.entries
                    .iter()
                    .filter_map(|(key, property)| key.as_index().map(|_| Element::entry(property))),
            )
    }

    /// The highest non-configurable index at or past `from`: always a named entry's, since a
    /// word is a configurable element.
    pub(crate) fn highest_nonconfig_index_from(&self, from: usize) -> Option<usize> {
        // A sealed or frozen block's present words are non-configurable elements too.
        let word = self
            .elems
            .words()
            .filter(|words| words.sealed())
            .and_then(|words| {
                words
                    .as_slice()
                    .iter()
                    .rposition(|&word| word != PACK_EMPTY)
                    .filter(|&n| n >= from)
            });
        self.entries
            .iter()
            .filter_map(|(k, p)| {
                (!p.configurable())
                    .then(|| k.as_index().map(|n| n as usize))
                    .flatten()
                    .filter(|&n| n >= from)
            })
            .chain(word)
            .max()
    }

    pub(crate) fn integrity_ok(&self, frozen: bool) -> bool {
        let valid = |p: &Property| !p.configurable() && (!frozen || p.accessor() || !p.writable());
        // Present words are elements of the block's integrity level (W2c step 7a).
        let need = if frozen {
            word_block::KIND_FROZEN
        } else {
            word_block::KIND_SEALED
        };
        (self.word_integrity() & need != 0 || self.word_indices().next().is_none())
            && self.entries.iter().all(|(_, p)| valid(p))
    }

    /// The element words' integrity bits, 0 for a map without words.
    #[inline]
    pub(crate) fn word_integrity(&self) -> u32 {
        self.elems.words().map_or(0, word_block::Words::integrity)
    }

    /// SetIntegrityLevel's step for the element words of an object made non-extensible: every
    /// word becomes a sealed (or frozen) element at once, as SpiderMonkey's elements flags and
    /// V8's elements kinds make them, where a define per element moved each to a named entry.
    /// The named entries take their defines as before.
    pub(crate) fn seal_elements(&mut self, freeze: bool) {
        if self.elems.words().is_some() {
            self.note_structural();
        }
        if let Some(words) = self.elems.words_mut() {
            words.set_integrity(if freeze {
                word_block::KIND_INTEGRITY
            } else {
                word_block::KIND_SEALED
            });
        }
    }
}

#[cfg(test)]
#[path = "value_word_kind_tests.rs"]
mod word_kind_tests;

#[cfg(test)]
#[path = "value_array_shift_tests.rs"]
mod array_shift_tests;
#[cfg(test)]
#[path = "value_host_array_tests.rs"]
mod host_array_tests;

#[cfg(test)]
#[path = "value_dense_elements_tests.rs"]
mod dense_elements_tests;

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[path = "value_remembered_elements_tests.rs"]
mod remembered_elements_tests;

/// A canonical array-index property key (`"0"`, `"42"` — decimal, no leading zeros, fits u32).
#[inline(always)]
pub(crate) fn canonical_index(k: &str) -> Option<u32> {
    let bytes = k.as_bytes();
    let &first = bytes.first()?;
    // Named properties dominate. Reject them after one byte instead of setting up the full
    // iterator/parser path; this function sits in every generic property lookup.
    if !first.is_ascii_digit() {
        return None;
    }
    if first == b'0' {
        return (bytes.len() == 1).then_some(0);
    }
    if !bytes[1..].iter().all(u8::is_ascii_digit) {
        return None;
    }
    k.parse::<u32>().ok().filter(|&n| n != u32::MAX)
}

/// Convenience: define a plain own data property by key/value.
pub(crate) fn set_data(obj: &Gc, key: &str, value: Value) {
    obj.borrow_mut().props.insert(key, Property::plain(value));
}

/// Convenience: define a non-enumerable builtin property by key/value.
pub(crate) fn set_builtin(obj: &Gc, key: &str, value: Value) {
    obj.borrow_mut().props.insert(key, Property::builtin(value));
}

/// Set engine-only state without exposing an ECMAScript property.
pub(crate) fn set_internal_slot(obj: &Gc, key: &str, value: Value) {
    obj.borrow_mut()
        .internal_slots_mut()
        .insert(key, Property::plain(value));
}

/// IEEE-754 half-precision (binary16) to single-precision conversion.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize into a single-precision normal number.
            let mut e: i32 = -1;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let m = m & 0x3ff;
            sign | (((127 - 15 - e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp as u32) + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// IEEE-754 double-precision to half-precision (binary16), round-to-nearest-even, rounding **once**.
/// Going through `f32` first would double-round — e.g. `2^-25 + ε` collapses to an exact tie at
/// `f32` and then rounds to zero instead of up to the smallest subnormal.
pub fn f64_to_f16(value: f64) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 48) & 0x8000) as u16;
    let exp = ((x >> 52) & 0x7ff) as i32;
    let mant = x & 0x000f_ffff_ffff_ffff; // 52-bit fraction
    if exp == 0x7ff {
        return if mant != 0 {
            sign | 0x7e00 // NaN
        } else {
            sign | 0x7c00 // infinity
        };
    }
    if exp == 0 && mant == 0 {
        return sign; // signed zero
    }
    let half_exp = exp - 1023 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00; // overflow → infinity
    }
    if half_exp <= 0 {
        // Subnormal half (or underflow to zero). Drop the low bits of the full significand,
        // rounding to nearest even. `exp == 0` doubles are far below f16 range → they fall out as 0.
        let m = if exp == 0 { mant } else { mant | (1u64 << 52) };
        let shift = 43 - half_exp; // 52-bit fraction → 10-bit fraction, minus the exponent deficit
        if shift >= 64 {
            return sign;
        }
        let mut h = (m >> shift) as u16;
        let round_bit = (m >> (shift - 1)) & 1;
        let sticky = (m & ((1u64 << (shift - 1)) - 1)) != 0;
        if round_bit != 0 && (sticky || (h & 1) != 0) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = (((half_exp as u32) << 10) | ((mant >> 42) as u32)) as u16;
    let round_bit = (mant >> 41) & 1;
    let sticky = (mant & ((1u64 << 41) - 1)) != 0;
    if round_bit != 0 && (sticky || (h & 1) != 0) {
        h = h.wrapping_add(1); // carry into exponent is intentional
    }
    sign | h
}

#[cfg(test)]
mod object_accounting_tests {
    use super::*;
    use crate::gc_core::{CollectionKind, EdgeSink, MarkClient};
    use crate::interpreter::WeakTarget;
    use crate::weak_metadata::{Subscriber, WeakMetadata};

    struct Roots<'a>(Vec<u64>, &'a GcHeap);
    impl MarkClient for Roots<'_> {
        fn roots(&self, sink: &mut dyn FnMut(u64)) {
            for &word in &self.0 {
                sink(word);
            }
            trace_heap_roots(
                self.1,
                &mut crate::gc_trace::VisitSink(&mut |cell| sink(cell as u64)),
            );
        }
        fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}
        fn trace_side_slots(&self, _: usize, _: &mut dyn FnMut(u64)) {}
        fn ephemerons(&self, _: &mut EdgeSink<'_>) {}
    }

    /// Drive the real object descriptor with precise roots, so death assertions do not depend
    /// on conservative retention in the test's frame. Preserve the production phase ordering.
    fn collect(heap: &GcHeap, roots: &[Gc], kind: CollectionKind) {
        let client = Roots(roots.iter().map(|o| Gc::as_ptr(o) as u64).collect(), heap);
        assert!(mark_agent_heap(heap, false, &client, kind));
        let dead = crate::gc_sweep::DeadObjects::new(heap);
        sweep_object_metadata(heap, &dead);
        let pins = {
            let core = heap.core.borrow();
            crate::gc_handle::clear_dead_weak_cells(|cell| core.is_dead_target(cell))
        };
        heap.core.borrow_mut().finish_collection(&pins);
    }

    #[test]
    fn object_count_and_snapshots_follow_minor_major_and_teardown() {
        let heap = new_gc_heap();
        let symbols = new_symbol_agent();
        let _active = enter_agent(&heap, &symbols);
        let release = HeapRelease::new(&heap);
        let count = live_objects_ptr(&heap);
        let objects: Vec<_> = (0..1100).map(|_| Object::new(None)).collect();
        let pinned = Gc::downgrade(&objects[1]);
        assert_eq!(heap_live_objects(&heap), 1100);
        collect(&heap, &[objects[0], objects[2]], CollectionKind::Major);
        assert!(pinned.upgrade().is_none());
        assert_eq!(heap_live_objects(&heap), 2);
        let mut snapshot: Vec<_> = heap_gc_snapshot(&heap).iter().map(Gc::as_ptr).collect();
        snapshot.sort_unstable();
        let mut expected = [Gc::as_ptr(&objects[0]), Gc::as_ptr(&objects[2])];
        expected.sort_unstable();
        assert_eq!(snapshot, expected);
        let young = Object::new(None);
        for _ in 0..100 {
            Object::new(None);
        }
        collect(&heap, &[objects[0], young], CollectionKind::Minor);
        assert_eq!(
            heap_live_objects(&heap),
            3,
            "old garbage stays counted until a major"
        );
        assert_eq!(heap_gc_snapshot(&heap).len(), 3);
        collect(&heap, &[objects[0], young], CollectionKind::Major);
        assert_eq!(heap_live_objects(&heap), 2);
        assert_eq!(count, live_objects_ptr(&heap));
        // SAFETY: the heap owns this stable counter throughout the test.
        assert_eq!(unsafe { *count }, 2);
        drop(release);
        assert_eq!(heap_live_objects(&heap), 0);
        assert!(heap_gc_snapshot(&heap).is_empty());
        assert!(pinned.upgrade().is_none());
        heap.core.borrow_mut().release_all();
        assert_eq!(heap_live_objects(&heap), 0);
    }

    #[test]
    fn object_allocation_uses_active_heap_without_borrowing_its_prototype() {
        let active = crate::Engine::new();
        let prototype = Object::new(None);
        let before = heap_live_objects(&active.interp.gc_heap);
        let borrowed = prototype.borrow_mut();
        let object = Object::new(Some(prototype));
        assert!(active.interp.gc_heap.core.borrow().owns_cell(
            Gc::as_ptr(&object) as usize,
            active.interp.gc_heap.object_desc,
        ));
        assert_eq!(heap_live_objects(&active.interp.gc_heap), before + 1);
        assert!(Gc::ptr_eq(&object.borrow().proto.unwrap(), &prototype));
        drop(borrowed);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "object prototype belongs to another Agent heap (I6)")]
    fn object_allocation_rejects_foreign_prototypes() {
        let _owner = crate::Engine::new();
        let prototype = Object::new(None);
        let _active = crate::Engine::new();
        Object::new(Some(prototype));
    }

    #[test]
    fn object_observers_follow_foreign_heaps_removal_and_address_reuse() {
        let heap = new_gc_heap();
        let symbols = new_symbol_agent();
        let _active = enter_agent(&heap, &symbols);
        let release = HeapRelease::new(&heap);
        let target = Object::new(None);
        let address = Gc::as_ptr(&target) as usize;
        let weak = WeakTarget::of(&Value::Obj(target)).unwrap();
        let foreign = crate::Engine::new();
        let mut first = WeakMetadata::default();
        let mut second = WeakMetadata::default();
        first.subscribe(weak.clone(), Subscriber::WeakRef(1));
        second.subscribe(weak.clone(), Subscriber::WeakRef(2));
        assert_eq!(heap.weak_observers.borrow().len(), 1);
        assert!(foreign.interp.gc_heap.weak_observers.borrow().is_empty());
        first.unsubscribe(weak.key(), Subscriber::WeakRef(1));
        assert_eq!(heap.weak_observers.borrow().len(), 1);
        second.unsubscribe(weak.key(), Subscriber::WeakRef(2));
        assert!(heap.weak_observers.borrow().is_empty());
        second.subscribe(weak.clone(), Subscriber::WeakRef(2));
        collect(&heap, &[], CollectionKind::Major);
        assert!(weak.upgrade().is_none());
        assert!(heap.weak_observers.borrow().is_empty());
        assert_eq!(second.take_deaths().len(), 1);
        assert_eq!(
            second.take_subscribers(weak.key()),
            vec![Subscriber::WeakRef(2)]
        );
        // The old identity is now unobserved, so its pinned cell can be freed and reused.
        drop(weak);
        collect(&heap, &[], CollectionKind::Major);
        activate_agent(&heap, &symbols);
        let reused = (0..1024)
            .map(|_| Object::new(None))
            .find(|o| Gc::as_ptr(o) as usize == address)
            .expect("the released page reuses the original cell");
        let weak = WeakTarget::of(&Value::Obj(reused)).unwrap();
        first.subscribe(weak.clone(), Subscriber::WeakRef(3));
        assert!(
            first.take_deaths().is_empty(),
            "new identity must not inherit an old notification"
        );
        assert!(weak.upgrade().is_some());
        drop(release);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            first.take_deaths().len(),
            1,
            "teardown notifies the new observer once"
        );
        assert!(heap.weak_observers.borrow().is_empty());
    }

    #[test]
    fn object_payloads_drop_with_their_payload_cells() {
        struct Witness(Rc<Cell<usize>>);
        impl Drop for Witness {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let heap = new_gc_heap();
        let symbols = new_symbol_agent();
        let _active = enter_agent(&heap, &symbols);
        let release = HeapRelease::new(&heap);
        let named = Object::new(None);
        named
            .borrow_mut()
            .props
            .insert("field", Property::plain(Value::Num(1.0)));
        let layout = named
            .borrow()
            .props
            .entries
            .layout
            .as_ref()
            .unwrap()
            .clone();
        let identity = layout.as_ptr();
        // Internal slots live in the map's extras, a payload cell with a finalizer.
        let sidecar = Object::new(None);
        sidecar
            .borrow_mut()
            .internal_slots_mut()
            .insert("state", Property::plain(Value::Num(2.0)));
        // A native function's Rust state lives in a payload cell, dropped when the cell dies.
        let dropped = Rc::new(Cell::new(0));
        let witness = Witness(dropped.clone());
        let native = Object::new(None);
        native.borrow_mut().kind = ObjectKind::native_data(Rc::new(NativeCallable {
            body: NativeCallableBody::Opaque(Rc::new(move |_, _, _| {
                let _keep = &witness;
                Ok(Value::Undefined)
            })),
            retained: None,
            identity: Rc::from("finalization witness"),
        }));
        collect(&heap, &[], CollectionKind::Major);
        assert_eq!(layout.as_ptr(), identity);
        assert_eq!(&*layout[0].text(), "field");
        assert_eq!(dropped.get(), 1, "native Rust state drops exactly once");
        drop(release);
        assert_eq!(dropped.get(), 1);
    }
}

#[cfg(test)]
#[path = "property_layout_cell_tests.rs"]
pub(crate) mod property_layout_cell_tests;
