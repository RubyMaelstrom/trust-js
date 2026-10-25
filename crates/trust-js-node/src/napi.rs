//! N-API (Node-API) host implementation — enough of the stable C ABI to load and run Node
//! native addons (`.node` files), with no third-party crate.
//!
//! # How it fits together
//!
//! A `.node` addon is a shared library. [`crate::dylib::DynLib`] `dlopen`s it and resolves its
//! `napi_register_module_v1` entry point; [`op_load_addon`] calls that entry with a fresh
//! [`Env`] and an empty exports object, and returns whatever the addon populated.
//!
//! The addon, in turn, calls back into the `napi_*` C functions defined below. Those symbols are
//! exported from the host executable (trust-js-cli is linked with `-export_dynamic` / `-rdynamic`),
//! so the dynamic linker resolves the addon's undefined `napi_*` references against them — the
//! same mechanism the real `node` binary uses. [`keep_napi_exports`] takes each symbol's address
//! so the linker cannot drop them from the static archive.
//!
//! # The handle model
//!
//! - [`napi_env`] is a pointer to an [`Env`], which carries a raw pointer back to the engine
//!   ([`Ctx`]) plus a handle-scope arena of boxed [`Value`]s.
//! - [`napi_value`] is a pointer to one of those boxed `Value`s. Boxing gives each value a stable
//!   address even as the arena grows; the arena keeps it alive for the duration of the call,
//!   holding each object, string, symbol and BigInt in a rooted handle.
//! - References ([`napi_ref`]) keep their values in counted handles, strong while the count is
//!   above 0 and weak at 0 (Node-API's semantics), wraps name their objects through weak
//!   handles, and deferreds hold their promise's resolving functions in handles. A wrapped
//!   object's finalizer runs once the object has died, at the next call into an addon.
//! - [`napi_callback_info`] is a pointer to a [`CbInfo`] holding a native call's `this`, args, and
//!   `data`.
//!
//! # Safety
//!
//! This module executes third-party machine code and speaks a raw C ABI, so it is unavoidably
//! `unsafe`. The engine is single-threaded and `!Send`; every N-API call happens synchronously on
//! the loop thread while the engine is live, so recovering `&mut Ctx` from the stored pointer is
//! sound as long as addons never stash an `env` for later or hand it to another thread (neither is
//! permitted by the N-API contract).

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};
use std::rc::Rc;
use trust_js_host::ValueKind;

use trust_js_host::{
    CountedHandle, Ctx, Handle, HostGc, HostGcVisitor, HostTrace, NativeCallableRetained,
    NativeRetainedMemoryVisitor, Value, WeakHandle,
};

use crate::dylib::DynLib;

// ---- ABI handle types ------------------------------------------------------------------------

/// `napi_env` — passed to every N-API call; opaque to the addon.
pub type napi_env = *mut Env;
/// `napi_value` — an opaque handle to a JS value (internally `*const Value` into the env arena).
pub type napi_value = *mut c_void;
/// `napi_callback_info` — opaque handle to an in-flight native call's arguments.
pub type napi_callback_info = *mut CbInfo;
/// `napi_status` — `0` (`napi_ok`) on success.
pub type napi_status = c_int;
/// `napi_handle_scope` — opaque; our arena makes scopes no-ops, so this is a non-null token.
pub type napi_handle_scope = *mut c_void;
/// The addon's C callback: `(env, info) -> return value`.
pub type napi_callback = Option<unsafe extern "C" fn(napi_env, napi_callback_info) -> napi_value>;

// napi_status values (subset of the C enum).
const NAPI_OK: napi_status = 0;
const NAPI_INVALID_ARG: napi_status = 1;
const NAPI_OBJECT_EXPECTED: napi_status = 2;
const NAPI_NUMBER_EXPECTED: napi_status = 6;
const NAPI_BOOLEAN_EXPECTED: napi_status = 7;
const NAPI_PENDING_EXCEPTION: napi_status = 10;

// napi_valuetype values (the C enum returned by napi_typeof).
const NAPI_UNDEFINED: c_int = 0;
const NAPI_NULL: c_int = 1;
const NAPI_BOOLEAN: c_int = 2;
const NAPI_NUMBER: c_int = 3;
const NAPI_STRING: c_int = 4;
const NAPI_SYMBOL: c_int = 5;
const NAPI_OBJECT: c_int = 6;
const NAPI_FUNCTION: c_int = 7;
const NAPI_BIGINT: c_int = 9;

/// `NAPI_AUTO_LENGTH` — a length argument meaning "the string is NUL-terminated".
const NAPI_AUTO_LENGTH: usize = usize::MAX;

// ---- env / handle machinery ------------------------------------------------------------------

/// The concrete backing of an [`napi_env`]: a pointer to the engine's context plus a per-call
/// arena of handles that keeps every [`napi_value`] alive for the duration of the call (in
/// [`NapiState::handles`]).
pub struct Env {
    /// The context of the native call (or engine scope) the env serves, its scope's lifetime
    /// erased: the env lives within that call, and each N-API function reads the values it
    /// uses into a scope of its own ([`Env::interp`]), never storing one.
    interp: *mut Ctx<'static>,
    /// Where this call's handles start in [`NapiState::handles`]; dropping the env releases
    /// them (calls nest, so envs drop in reverse order of creation).
    scope_start: usize,
    /// An exception raised by the addon (`napi_throw*`), surfaced to JS once the call returns.
    pending: Option<Handle>,
}

impl Env {
    fn new(interp: *mut Ctx<'_>) -> Env {
        let mut env = Env {
            interp: interp.cast(),
            scope_start: 0,
            pending: None,
        };
        env.scope_start = unsafe { env.napi_state() }.handles.len();
        env
    }

    /// Hold a value in the handle arena and return a stable `napi_value` handle to it: the
    /// address of a boxed handle, which keeps the value until the call's arena is released.
    fn handle(&mut self, v: Value<'_>) -> napi_value {
        // A value an addon handles is its engine's, and no N-API function runs inside a
        // collection (only then is a handle refused), so the handle is always made.
        let handle = Box::new(
            Handle::new(unsafe { self.interp() }, &v).expect("a value of the addon's engine"),
        );
        let ptr = (&*handle as *const Handle) as *mut c_void;
        unsafe { self.napi_state() }.handles.push(handle);
        ptr
    }

    /// The value behind a `napi_value`, read into the calling function's scope. A null handle
    /// reads as `undefined`.
    unsafe fn value<'s>(&self, v: napi_value) -> Value<'s> {
        if v.is_null() {
            Value::Undefined
        } else {
            (*(v as *const Handle))
                .get(self.interp())
                .expect("a handle of the addon's engine")
        }
    }

    /// Raise `error` as the call's pending exception.
    fn set_pending(&mut self, error: Value<'_>) {
        let held =
            Handle::new(unsafe { self.interp() }, &error).expect("a value of the addon's engine");
        self.pending = Some(held);
    }

    /// Take the call's pending exception, read into the scope `cx`.
    fn take_pending<'s>(&mut self, cx: &Ctx<'s>) -> Option<Value<'s>> {
        self.pending
            .take()
            .map(|error| error.get(cx).expect("a handle of the addon's engine"))
    }

    /// Run the finalizers of wrapped objects that have died since the last call into an addon
    /// (see [`NapiState`]'s sweep): after their collection, outside it, before addon code runs
    /// again. They run in an env of their own, so that an exception one raises is not taken for
    /// the call's (it is dropped).
    fn run_pending_finalizers(&mut self) {
        let pending = std::mem::take(&mut unsafe { self.napi_state() }.pending_finalizers);
        if pending.is_empty() {
            return;
        }
        let mut env = Env::new(self.interp);
        for finalizer in pending {
            unsafe { (finalizer.callback)(&mut env, finalizer.data, finalizer.hint) };
        }
    }

    /// Recover the context, in a scope of the caller's choosing. Unsafe: relies on the call
    /// happening on the engine's loop thread, within the native call the env serves; the values
    /// read through it do not outlive the N-API function that reads them.
    #[allow(clippy::mut_from_ref)]
    unsafe fn interp<'s, 'a>(&self) -> &'a mut Ctx<'s> {
        &mut *self.interp.cast::<Ctx<'s>>()
    }
}

impl Drop for Env {
    /// Release this call's handles; an enclosing call's stay.
    fn drop(&mut self) {
        unsafe { self.napi_state() }
            .handles
            .truncate(self.scope_start);
    }
}

/// A native call's receiver and arguments, behind an [`napi_callback_info`], as `napi_value`s
/// of the call's arena.
pub struct CbInfo {
    this: napi_value,
    args: Vec<napi_value>,
    data: *mut c_void,
}

/// Read a possibly-NUL-terminated (`NAPI_AUTO_LENGTH`) or length-counted UTF-8 C string.
unsafe fn read_utf8(s: *const c_char, len: usize) -> String {
    if s.is_null() {
        return String::new();
    }
    let bytes: &[u8] = if len == NAPI_AUTO_LENGTH {
        std::ffi::CStr::from_ptr(s).to_bytes()
    } else {
        std::slice::from_raw_parts(s.cast::<u8>(), len)
    };
    String::from_utf8_lossy(bytes).into_owned()
}

/// ECMAScript ToInt32 — the coercion `napi_get_value_int32` applies to a non-integral number.
fn to_int32(n: f64) -> i32 {
    if !n.is_finite() {
        return 0;
    }
    let m = n.trunc().rem_euclid(4294967296.0); // mod 2^32, always in [0, 2^32)
    m as u32 as i32
}

// ---- the JS-function trampoline --------------------------------------------------------------

/// Wrap an addon C callback (`cb` + its `data`) as a TRust JS function value. When JS calls it, a
/// fresh [`Env`] and [`CbInfo`] are built and the C callback is invoked.
fn make_callback_fn<'s>(
    interp: &mut Ctx<'s>,
    name: &str,
    cb: napi_callback,
    data: *mut c_void,
) -> Value<'s> {
    let state = Rc::new(NapiCallbackClosure { callback: cb, data });
    let callable_state = state.clone();
    interp.new_native_fn(
        name,
        0,
        move |interp, this, args| callable_state.call(interp, this, args),
        trust_js_host::NativeCaptures::Reported(state),
    )
}

struct NapiCallbackClosure {
    callback: napi_callback,
    data: *mut c_void,
}

impl NapiCallbackClosure {
    fn call<'s>(
        &self,
        interp: &mut Ctx<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        invoke_napi_callback(interp, self.callback, self.data, this, args)
    }
}

impl NativeCallableRetained for NapiCallbackClosure {
    fn scan_retained_memory(&self, _visitor: &mut dyn NativeRetainedMemoryVisitor) {
        // Both captures are non-owning C pointers stored in this callable's inline payload.
    }
}

// Node-API “Working with JavaScript functions” and `napi_define_class` require the callback's
// opaque `data` pointer and the already-created instance/prototype relationship to survive until
// invocation. The constructor keeps its prototype as an engine capture (S6 step 11b): the C
// callback and its data are the closure's Rust state.
struct NapiConstructorClosure {
    callback: napi_callback,
    data: *mut c_void,
}

impl NapiConstructorClosure {
    /// A call of the constructor whose prototype is `prototype` (its capture).
    fn call<'s>(
        &self,
        interp: &mut Ctx<'s>,
        this: Value<'s>,
        args: &[Value<'s>],
        prototype: &Value<'s>,
    ) -> Result<Value<'s>, Value<'s>> {
        let instance = match this.kind() {
            ValueKind::Undefined => interp.new_object_with_proto(prototype),
            _ => this,
        };
        invoke_napi_callback(interp, self.callback, self.data, instance, args)?;
        Ok(instance)
    }
}

impl NativeCallableRetained for NapiConstructorClosure {
    fn scan_retained_memory(&self, _visitor: &mut dyn NativeRetainedMemoryVisitor) {
        // Both captures are non-owning C pointers stored in this callable's inline payload;
        // the prototype is an engine capture of the function.
    }
}

/// `napi_create_promise`'s executor: the Promise constructor calls it with the resolving
/// functions, which it keeps in handles (Rust state) for the deferred.
struct NapiPromiseExecutor {
    slot: Rc<std::cell::RefCell<Option<(Handle, Handle)>>>,
}

impl NapiPromiseExecutor {
    fn call<'s>(
        &self,
        interp: &mut Ctx<'s>,
        _this: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let resolve = args.first().copied().unwrap_or(Value::Undefined);
        let reject = args.get(1).copied().unwrap_or(Value::Undefined);
        let resolve = trust_js_host::handle(interp, &resolve)?;
        let reject = trust_js_host::handle(interp, &reject)?;
        *self.slot.borrow_mut() = Some((resolve, reject));
        Ok(Value::Undefined)
    }
}

impl NativeCallableRetained for NapiPromiseExecutor {
    fn scan_retained_memory(&self, visitor: &mut dyn NativeRetainedMemoryVisitor) {
        visitor.allocation(trust_js_host::RetainedManagedAllocation::rc(
            "trust-js-node.napi-promise-executor-slot",
            &self.slot,
            std::mem::size_of::<std::cell::RefCell<Option<(Handle, Handle)>>>(),
        ));
    }
}

/// Drive one call into an addon C callback, translating its return / thrown exception back into
/// the engine's `Result<Value, Value>`.
fn invoke_napi_callback<'s>(
    ip: &mut Ctx<'s>,
    cb: napi_callback,
    data: *mut c_void,
    this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    let cb = match cb {
        Some(f) => f,
        None => return Ok(Value::Undefined),
    };
    let mut env = Env::new(ip as *mut Ctx);
    env.run_pending_finalizers();
    let this = env.handle(this);
    let args = args.iter().map(|arg| env.handle(*arg)).collect();
    let mut info = CbInfo { this, args, data };
    let ret = unsafe { cb(&mut env as *mut Env, &mut info as *mut CbInfo) };
    if let Some(err) = env.take_pending(ip) {
        return Err(err);
    }
    Ok(unsafe { env.value(ret) })
}

// ---- the loader op ---------------------------------------------------------------------------

/// Loaded native libraries, kept alive for the process lifetime: an addon's function pointers
/// dangle the moment its library is `dlclose`d, and (like Node) addons are never unloaded.
#[derive(Default)]
pub struct AddonRegistry {
    libs: Vec<DynLib>,
}

/// Persistent N-API state that outlives individual calls: references (values kept alive past a
/// handle scope), object wraps (native pointers attached to JS objects), promise deferreds, and
/// registered async work. Lives in `OpState`, reached from an [`Env`] via the engine pointer.
#[derive(Default)]
struct NapiState {
    /// The handles behind the `napi_value`s of the calls in progress, innermost call last, each
    /// holding its value. The `Box` is load-bearing: a napi_value is a raw pointer into this
    /// arena, so each handle needs a stable heap address that survives the Vec growing.
    #[allow(clippy::vec_box)]
    handles: Vec<Box<Handle>>,
    /// References, each a counted handle: strong while its count is above 0, weak at 0.
    refs: Vec<Option<CountedHandle>>,
    ref_free: Vec<usize>,
    /// object address -> wrapped native pointer (`napi_wrap`). Each entry names its object
    /// through a weak handle, which is cleared as the object dies (before its address can be
    /// reused), so a new object at a dead one's address never unwraps to the old pointer. The
    /// collection that clears an entry's handle moves its finalizer to `pending_finalizers`.
    wraps: std::collections::HashMap<usize, Wrap>,
    /// Finalizers of wrapped objects that have died, run at the next call into an addon.
    pending_finalizers: Vec<PendingFinalizer>,
    deferreds: Vec<Option<Deferred>>,
    async_works: Vec<Option<AsyncWork>>,
}

/// `napi_finalize`: `(env, finalize_data, finalize_hint)`.
pub type napi_finalize = Option<unsafe extern "C" fn(napi_env, *mut c_void, *mut c_void)>;

struct Wrap {
    object: WeakHandle,
    native: *mut c_void,
    /// The addon's finalizer and its hint, called with `native` once the object has died.
    finalize: Option<(
        unsafe extern "C" fn(napi_env, *mut c_void, *mut c_void),
        *mut c_void,
    )>,
}

/// A dead wrapped object's finalizer call.
struct PendingFinalizer {
    callback: unsafe extern "C" fn(napi_env, *mut c_void, *mut c_void),
    data: *mut c_void,
    hint: *mut c_void,
}

/// Everything N-API keeps is held by handles: there is no graph to report.
impl HostTrace for NapiState {
    fn trace_gc(&self, _: &mut dyn HostGcVisitor) {}
}

/// Between a collection's phases, after the weak handles of its dead objects have been
/// cleared: a wrap whose object died leaves the table, and its finalizer waits for the next
/// call into an addon, outside the collection.
impl HostGc for NapiState {
    fn sweep_gc_with_native(&mut self, _: &dyn Fn(trust_js_host::NativeGcId) -> bool) {
        let pending = &mut self.pending_finalizers;
        self.wraps.retain(|_, wrap| {
            if !wrap.object.is_cleared() {
                return true;
            }
            if let Some((callback, hint)) = wrap.finalize {
                pending.push(PendingFinalizer {
                    callback,
                    data: wrap.native,
                    hint,
                });
            }
            false
        });
    }
}

struct Deferred {
    resolve: Handle,
    reject: Handle,
}

struct AsyncWork {
    execute: napi_async_execute_callback,
    complete: napi_async_complete_callback,
    data: *mut c_void,
}

/// `napi_ref` — an opaque handle to a reference; internally `index + 1` (non-null) into
/// `NapiState::refs`.
pub type napi_ref = *mut c_void;
/// `napi_deferred` — the resolve/reject side of a promise; `index + 1` into `NapiState::deferreds`.
pub type napi_deferred = *mut c_void;
/// `napi_async_work` — a registered unit of async work; `index + 1` into `NapiState::async_works`.
pub type napi_async_work = *mut c_void;
/// `napi_threadsafe_function` — opaque; stubbed (see the threadsafe-function functions).
pub type napi_threadsafe_function = *mut c_void;

/// The off-thread half of an async work item: `(env, data)`. Must not touch JS.
pub type napi_async_execute_callback = Option<unsafe extern "C" fn(napi_env, *mut c_void)>;
/// The on-loop-thread half of an async work item: `(env, status, data)`.
pub type napi_async_complete_callback =
    Option<unsafe extern "C" fn(napi_env, napi_status, *mut c_void)>;

/// A property descriptor for `napi_define_class` / `napi_define_properties` (C ABI layout).
#[repr(C)]
pub struct napi_property_descriptor {
    utf8name: *const c_char,
    name: napi_value,
    method: napi_callback,
    getter: napi_callback,
    setter: napi_callback,
    value: napi_value,
    attributes: c_int,
    data: *mut c_void,
}

/// `napi_static` attribute bit — the property lives on the constructor, not the prototype.
const NAPI_STATIC: c_int = 1 << 10;
/// `napi_enumerable` attribute bit.
const NAPI_ENUMERABLE: c_int = 1 << 1;

impl Env {
    /// The process-wide N-API state, lazily installed in `OpState` (and registered for the
    /// collector's sweeps, which notice dead wrapped objects).
    #[allow(clippy::mut_from_ref)]
    unsafe fn napi_state(&self) -> &mut NapiState {
        let os = (*self.interp).op_state();
        if os.get::<NapiState>().is_none() {
            os.put(NapiState::default());
            os.register_gc::<NapiState>();
        }
        os.get_mut::<NapiState>().unwrap()
    }

    /// The native pointer wrapped onto `object`, an object at `addr`. The entry's weak handle
    /// still names a live object, which is the one at `addr`: it was cleared before any dead
    /// object's address could be reused.
    unsafe fn unwrap<'s>(&self, object: &Value<'s>, addr: usize) -> Option<*mut c_void> {
        let (weak, native) = self
            .napi_state()
            .wraps
            .get(&addr)
            .map(|wrap| (wrap.object.clone(), wrap.native))?;
        let wrapped = weak.upgrade(self.interp()).ok().flatten()?;
        (self.interp().object_addr(&wrapped) == self.interp().object_addr(object)).then_some(native)
    }
}

/// Encode a `NapiState` slot index as a non-null opaque handle.
fn encode_slot(idx: usize) -> *mut c_void {
    (idx + 1) as *mut c_void
}

/// Decode an opaque handle back to a slot index (`None` for a null handle).
fn decode_slot(handle: *mut c_void) -> Option<usize> {
    if handle.is_null() {
        None
    } else {
        Some(handle as usize - 1)
    }
}

/// `__node.loadNativeAddon(path)` — dlopen a `.node` addon, run its N-API registration, and
/// return the exports object.
pub fn op_load_addon<'s>(
    ctx: &mut Ctx<'s>,
    _this: Value<'s>,
    args: &[Value<'s>],
) -> Result<Value<'s>, Value<'s>> {
    keep_napi_exports();

    let path = ctx
        .coerce_string(args.first().unwrap_or(&Value::Undefined))?
        .to_string();

    let lib = DynLib::open(&path)
        .map_err(|e| ctx.make_error("Error", format!("cannot load native addon '{path}': {e}")))?;
    let sym = lib.symbol("napi_register_module_v1").ok_or_else(|| {
        ctx.make_error(
            "Error",
            format!("'{path}' is not an N-API addon (no napi_register_module_v1)"),
        )
    })?;

    type Register = unsafe extern "C" fn(napi_env, napi_value) -> napi_value;
    let register: Register = unsafe { std::mem::transmute::<*mut c_void, Register>(sym) };

    // Build the exports object and the call env, then hand control to the addon.
    let exports = Value::Obj(ctx.new_object());
    let mut env = Env::new(ctx as *mut Ctx);
    env.run_pending_finalizers();
    let exports_handle = env.handle(exports);
    let ret = unsafe { register(&mut env as *mut Env, exports_handle) };
    let pending = env.take_pending(ctx);
    let result = if ret.is_null() {
        exports
    } else {
        unsafe { env.value(ret) }
    };

    // Retain the library so its code stays mapped for the process lifetime.
    if ctx.host_mut::<AddonRegistry>().is_none() {
        ctx.op_state().put(AddonRegistry::default());
    }
    ctx.host_mut::<AddonRegistry>().unwrap().libs.push(lib);

    match pending {
        Some(err) => Err(err),
        None => Ok(result),
    }
}

// ---- napi_* C ABI surface --------------------------------------------------------------------
//
// Every function here is `#[no_mangle] extern "C"` so a dlopened addon resolves it. They are the
// host side of the N-API contract; see the module docs for the handle model. Unsupported N-API
// functions are simply absent, so an addon that needs one fails loudly at `dlopen` (RTLD_NOW)
// naming the missing symbol, rather than misbehaving at runtime.

// -- value creation --

#[no_mangle]
pub unsafe extern "C" fn napi_get_undefined(env: napi_env, result: *mut napi_value) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Undefined);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_null(env: napi_env, result: *mut napi_value) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Null);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_global(env: napi_env, result: *mut napi_value) -> napi_status {
    let env = &mut *env;
    let g = env.interp().global_this();
    *result = env.handle(g);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_boolean(
    env: napi_env,
    value: bool,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Bool(value));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_double(
    env: napi_env,
    value: f64,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Num(value));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_int32(
    env: napi_env,
    value: i32,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Num(value as f64));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_uint32(
    env: napi_env,
    value: u32,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Num(value as f64));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_int64(
    env: napi_env,
    value: i64,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    *result = env.handle(Value::Num(value as f64));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_string_utf8(
    env: napi_env,
    s: *const c_char,
    len: usize,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let string = read_utf8(s, len);
    let string = env.interp().string(string);
    *result = env.handle(string);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_object(env: napi_env, result: *mut napi_value) -> napi_status {
    let env = &mut *env;
    let o = Value::Obj(env.interp().new_object());
    *result = env.handle(o);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_array(env: napi_env, result: *mut napi_value) -> napi_status {
    let env = &mut *env;
    let a = env.interp().make_array(Vec::new());
    *result = env.handle(a);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_array_with_length(
    env: napi_env,
    length: usize,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let a = env.interp().make_array(vec![Value::Undefined; length]);
    *result = env.handle(a);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_function(
    env: napi_env,
    utf8name: *const c_char,
    length: usize,
    cb: napi_callback,
    data: *mut c_void,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let name = read_utf8(utf8name, length);
    let f = make_callback_fn(env.interp(), &name, cb, data);
    *result = env.handle(f);
    NAPI_OK
}

// -- value inspection / extraction --

#[no_mangle]
pub unsafe extern "C" fn napi_typeof(
    env: napi_env,
    value: napi_value,
    result: *mut c_int,
) -> napi_status {
    let env = &*env;
    let v = env.value(value);
    let t = if matches!((v).kind(), ValueKind::Null) {
        NAPI_NULL
    } else {
        match v.type_of() {
            "undefined" => NAPI_UNDEFINED,
            "boolean" => NAPI_BOOLEAN,
            "number" => NAPI_NUMBER,
            "string" => NAPI_STRING,
            "symbol" => NAPI_SYMBOL,
            "bigint" => NAPI_BIGINT,
            "function" => NAPI_FUNCTION,
            _ => NAPI_OBJECT,
        }
    };
    *result = t;
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_value_double(
    env: napi_env,
    value: napi_value,
    result: *mut f64,
) -> napi_status {
    let env = &*env;
    match env.value(value).as_num_opt() {
        Some(n) => {
            *result = n;
            NAPI_OK
        }
        None => NAPI_NUMBER_EXPECTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_value_int32(
    env: napi_env,
    value: napi_value,
    result: *mut i32,
) -> napi_status {
    let env = &*env;
    match env.value(value).as_num_opt() {
        Some(n) => {
            *result = to_int32(n);
            NAPI_OK
        }
        None => NAPI_NUMBER_EXPECTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_value_uint32(
    env: napi_env,
    value: napi_value,
    result: *mut u32,
) -> napi_status {
    let env = &*env;
    match env.value(value).as_num_opt() {
        Some(n) => {
            *result = to_int32(n) as u32;
            NAPI_OK
        }
        None => NAPI_NUMBER_EXPECTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_value_bool(
    env: napi_env,
    value: napi_value,
    result: *mut bool,
) -> napi_status {
    let env = &*env;
    match (env.value(value)).kind() {
        ValueKind::Bool(b) => {
            *result = b;
            NAPI_OK
        }
        _ => NAPI_BOOLEAN_EXPECTED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_value_string_utf8(
    env: napi_env,
    value: napi_value,
    buf: *mut c_char,
    bufsize: usize,
    result: *mut usize,
) -> napi_status {
    let env = &mut *env;
    let v = env.value(value);
    let s = match env.interp().coerce_string(&v) {
        Ok(s) => s,
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };
    let bytes = s.as_bytes();
    if buf.is_null() {
        // Measurement mode: report the byte length (excluding the NUL).
        if !result.is_null() {
            *result = bytes.len();
        }
        return NAPI_OK;
    }
    if bufsize == 0 {
        if !result.is_null() {
            *result = 0;
        }
        return NAPI_OK;
    }
    let n = core::cmp::min(bytes.len(), bufsize - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast::<u8>(), n);
    *buf.add(n) = 0;
    if !result.is_null() {
        *result = n;
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_coerce_to_string(
    env: napi_env,
    value: napi_value,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let v = env.value(value);
    match env.interp().coerce_string(&v) {
        Ok(s) => {
            let string = env.interp().str(s);
            *result = env.handle(string);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

// -- properties --

#[no_mangle]
pub unsafe extern "C" fn napi_set_named_property(
    env: napi_env,
    object: napi_value,
    utf8name: *const c_char,
    value: napi_value,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    let name = read_utf8(utf8name, NAPI_AUTO_LENGTH);
    match env.interp().member_set(&obj, &name, env.value(value)) {
        Ok(()) => NAPI_OK,
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_named_property(
    env: napi_env,
    object: napi_value,
    utf8name: *const c_char,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    let name = read_utf8(utf8name, NAPI_AUTO_LENGTH);
    match env.interp().member_get(&obj, &name) {
        Ok(v) => {
            *result = env.handle(v);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_has_named_property(
    env: napi_env,
    object: napi_value,
    utf8name: *const c_char,
    result: *mut bool,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    let name = read_utf8(utf8name, NAPI_AUTO_LENGTH);
    match env.interp().member_get(&obj, &name) {
        Ok(v) => {
            *result = !matches!((v).kind(), ValueKind::Undefined);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_set_property(
    env: napi_env,
    object: napi_value,
    key: napi_value,
    value: napi_value,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    let k = match env.interp().coerce_string(&env.value(key)) {
        Ok(k) => k.to_string(),
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };
    match env.interp().member_set(&obj, &k, env.value(value)) {
        Ok(()) => NAPI_OK,
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_property(
    env: napi_env,
    object: napi_value,
    key: napi_value,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    let k = match env.interp().coerce_string(&env.value(key)) {
        Ok(k) => k.to_string(),
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };
    match env.interp().member_get(&obj, &k) {
        Ok(v) => {
            *result = env.handle(v);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_set_element(
    env: napi_env,
    object: napi_value,
    index: u32,
    value: napi_value,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    match env
        .interp()
        .member_set(&obj, &index.to_string(), env.value(value))
    {
        Ok(()) => NAPI_OK,
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_element(
    env: napi_env,
    object: napi_value,
    index: u32,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(object);
    match env.interp().member_get(&obj, &index.to_string()) {
        Ok(v) => {
            *result = env.handle(v);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

// -- callbacks / calls --

#[no_mangle]
pub unsafe extern "C" fn napi_get_cb_info(
    env: napi_env,
    cbinfo: napi_callback_info,
    argc: *mut usize,
    argv: *mut napi_value,
    this_arg: *mut napi_value,
    data: *mut *mut c_void,
) -> napi_status {
    let env = &mut *env;
    let info = &*cbinfo;
    if !argc.is_null() {
        let cap = *argc;
        if !argv.is_null() {
            for i in 0..cap {
                *argv.add(i) = match info.args.get(i) {
                    Some(arg) => *arg,
                    None => env.handle(Value::Undefined),
                };
            }
        }
        // Report the true argument count (may exceed the caller's buffer capacity).
        *argc = info.args.len();
    }
    if !this_arg.is_null() {
        *this_arg = info.this;
    }
    if !data.is_null() {
        *data = info.data;
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_call_function(
    env: napi_env,
    recv: napi_value,
    func: napi_value,
    argc: usize,
    argv: *const napi_value,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let recv = env.value(recv);
    let func = env.value(func);
    let mut args = Vec::with_capacity(argc);
    for i in 0..argc {
        args.push(env.value(*argv.add(i)));
    }
    match env.interp().invoke(func, recv, &args) {
        Ok(v) => {
            if !result.is_null() {
                *result = env.handle(v);
            }
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

// -- errors / exceptions --

#[no_mangle]
pub unsafe extern "C" fn napi_throw(env: napi_env, error: napi_value) -> napi_status {
    let env = &mut *env;
    env.set_pending(env.value(error));
    NAPI_OK
}

/// Shared body for the `napi_throw_*_error` family.
unsafe fn throw_named(env: napi_env, kind: &str, msg: *const c_char) -> napi_status {
    let env = &mut *env;
    let message = read_utf8(msg, NAPI_AUTO_LENGTH);
    let err = env.interp().make_error(kind, message);
    env.set_pending(err);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_throw_error(
    env: napi_env,
    _code: *const c_char,
    msg: *const c_char,
) -> napi_status {
    throw_named(env, "Error", msg)
}

#[no_mangle]
pub unsafe extern "C" fn napi_throw_type_error(
    env: napi_env,
    _code: *const c_char,
    msg: *const c_char,
) -> napi_status {
    throw_named(env, "TypeError", msg)
}

#[no_mangle]
pub unsafe extern "C" fn napi_throw_range_error(
    env: napi_env,
    _code: *const c_char,
    msg: *const c_char,
) -> napi_status {
    throw_named(env, "RangeError", msg)
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_error(
    env: napi_env,
    _code: napi_value,
    msg: napi_value,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let message = env
        .interp()
        .coerce_string(&env.value(msg))
        .map(|s| s.to_string())
        .unwrap_or_default();
    let err = env.interp().make_error("Error", message);
    *result = env.handle(err);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_is_exception_pending(
    env: napi_env,
    result: *mut bool,
) -> napi_status {
    let env = &mut *env;
    *result = env.pending.is_some();
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_and_clear_last_exception(
    env: napi_env,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let cx = env.interp();
    let v = env.take_pending(cx).unwrap_or(Value::Undefined);
    *result = env.handle(v);
    NAPI_OK
}

// -- handle scopes (no-ops: our per-call arena keeps every handle alive) --

#[no_mangle]
pub unsafe extern "C" fn napi_open_handle_scope(
    env: napi_env,
    result: *mut napi_handle_scope,
) -> napi_status {
    let _ = env;
    if !result.is_null() {
        // A non-null token; addons only compare it, they don't dereference it.
        *result = std::ptr::dangling_mut::<c_void>();
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_close_handle_scope(
    env: napi_env,
    scope: napi_handle_scope,
) -> napi_status {
    let _ = (env, scope);
    NAPI_OK
}

// -- misc --

#[no_mangle]
pub unsafe extern "C" fn napi_get_version(env: napi_env, result: *mut u32) -> napi_status {
    let _ = env;
    // Node-API version this host targets.
    *result = 8;
    NAPI_OK
}

// -- value predicates --

#[no_mangle]
pub unsafe extern "C" fn napi_strict_equals(
    env: napi_env,
    lhs: napi_value,
    rhs: napi_value,
    result: *mut bool,
) -> napi_status {
    let env = &mut *env;
    *result = env
        .interp()
        .values_strict_equal(&env.value(lhs), &env.value(rhs));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_is_error(
    env: napi_env,
    value: napi_value,
    result: *mut bool,
) -> napi_status {
    let env = &mut *env;
    *result = env.interp().is_error_value(&env.value(value));
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_prototype(
    env: napi_env,
    object: napi_value,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let value = env.value(object);
    // Node's CHECK_TO_OBJECT: ToObject, so null and undefined are not objects.
    if matches!((value).kind(), ValueKind::Undefined | ValueKind::Null) {
        return NAPI_OBJECT_EXPECTED;
    }
    // [[GetPrototypeOf]] (ECMA-262 #sec-object.getprototypeof), which runs a Proxy's
    // getPrototypeOf trap and its invariant checks; a throw becomes the pending exception.
    match env.interp().get_prototype_of(&value) {
        Ok(proto) => {
            *result = env.handle(proto);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

// -- references (values kept alive beyond the handle scope) --

#[no_mangle]
pub unsafe extern "C" fn napi_create_reference(
    env: napi_env,
    value: napi_value,
    initial_refcount: u32,
    result: *mut napi_ref,
) -> napi_status {
    let env = &mut *env;
    let v = env.value(value);
    let Ok(entry) = CountedHandle::new(env.interp(), &v, initial_refcount) else {
        return NAPI_INVALID_ARG;
    };
    let state = env.napi_state();
    let idx = match state.ref_free.pop() {
        Some(i) => {
            state.refs[i] = Some(entry);
            i
        }
        None => {
            state.refs.push(Some(entry));
            state.refs.len() - 1
        }
    };
    *result = encode_slot(idx);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_delete_reference(env: napi_env, r: napi_ref) -> napi_status {
    let env = &mut *env;
    if let Some(idx) = decode_slot(r) {
        let state = env.napi_state();
        if idx < state.refs.len() && state.refs[idx].take().is_some() {
            state.ref_free.push(idx);
        }
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_reference_value(
    env: napi_env,
    r: napi_ref,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    // Shared borrows only: the reference is read through the engine that holds it.
    let ctx: &Ctx = env.interp();
    let v = decode_slot(r)
        .and_then(|idx| ctx.host::<NapiState>()?.refs.get(idx))
        .and_then(|slot| slot.as_ref())
        .map(|e| e.get(ctx).ok().flatten());
    *result = match v {
        Some(Some(v)) => env.handle(v),
        // The referent died at a count of 0: Node-API returns NULL.
        Some(None) => std::ptr::null_mut(),
        None => env.handle(Value::Undefined),
    };
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_reference_ref(
    env: napi_env,
    r: napi_ref,
    result: *mut u32,
) -> napi_status {
    let env = &mut *env;
    let mut count = 0;
    if let Some(idx) = decode_slot(r) {
        if let Some(Some(e)) = env.napi_state().refs.get_mut(idx) {
            count = e.increment().unwrap_or_else(|_| e.count());
        }
    }
    if !result.is_null() {
        *result = count;
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_reference_unref(
    env: napi_env,
    r: napi_ref,
    result: *mut u32,
) -> napi_status {
    let env = &mut *env;
    let mut count = 0;
    if let Some(idx) = decode_slot(r) {
        if let Some(Some(e)) = env.napi_state().refs.get_mut(idx) {
            count = e.decrement().unwrap_or_else(|_| e.count());
        }
    }
    if !result.is_null() {
        *result = count;
    }
    NAPI_OK
}

// -- object wrapping (attach a native pointer to a JS object) --

#[no_mangle]
pub unsafe extern "C" fn napi_wrap(
    env: napi_env,
    js_object: napi_value,
    native_object: *mut c_void,
    finalize_cb: napi_finalize,
    finalize_hint: *mut c_void,
    result: *mut napi_ref,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(js_object);
    let Some(addr) = env.interp().object_addr(&obj) else {
        return NAPI_INVALID_ARG;
    };
    let Ok(object) = WeakHandle::new(env.interp(), &obj) else {
        return NAPI_INVALID_ARG;
    };
    env.napi_state().wraps.insert(
        addr,
        Wrap {
            object,
            native: native_object,
            finalize: finalize_cb.map(|callback| (callback, finalize_hint)),
        },
    );
    // Optionally hand back a reference to the wrapped object: weak, at a count of 0.
    if !result.is_null() {
        let mut r: napi_ref = std::ptr::null_mut();
        napi_create_reference(env, js_object, 0, &mut r);
        *result = r;
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_unwrap(
    env: napi_env,
    js_object: napi_value,
    result: *mut *mut c_void,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(js_object);
    let Some(addr) = env.interp().object_addr(&obj) else {
        return NAPI_INVALID_ARG;
    };
    *result = env.unwrap(&obj, addr).unwrap_or(std::ptr::null_mut());
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_remove_wrap(
    env: napi_env,
    js_object: napi_value,
    result: *mut *mut c_void,
) -> napi_status {
    let env = &mut *env;
    let obj = env.value(js_object);
    let Some(addr) = env.interp().object_addr(&obj) else {
        return NAPI_INVALID_ARG;
    };
    // Node-API: the wrap goes without its finalizer being called.
    let data = env.unwrap(&obj, addr).unwrap_or(std::ptr::null_mut());
    env.napi_state().wraps.remove(&addr);
    if !result.is_null() {
        *result = data;
    }
    NAPI_OK
}

// -- classes --

#[no_mangle]
pub unsafe extern "C" fn napi_define_class(
    env: napi_env,
    utf8name: *const c_char,
    length: usize,
    constructor: napi_callback,
    data: *mut c_void,
    property_count: usize,
    properties: *const napi_property_descriptor,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let name = read_utf8(utf8name, length);
    let interp = env.interp();

    // The prototype instances inherit from, captured by the constructor trampoline so `new`
    // produces an object with the right shape before the C constructor wraps native state onto it.
    let proto = Value::Obj(interp.new_object());
    let state = Rc::new(NapiConstructorClosure {
        callback: constructor,
        data,
    });
    let callable_state = state.clone();
    let ctor = interp.new_native_closure_with_captures(
        &name,
        0,
        move |interp, this, args, captures| callable_state.call(interp, this, args, &captures[0]),
        &[proto],
        trust_js_host::NativeCaptures::Reported(state),
    );
    interp.set_constructor_prototype(&ctor, &proto);

    // Attach each declared method / accessor / value to the prototype (or the constructor, for
    // `napi_static` members).
    for i in 0..property_count {
        let d = &*properties.add(i);
        let target = if d.attributes & NAPI_STATIC != 0 {
            ctor
        } else {
            proto
        };
        let pname = if !d.utf8name.is_null() {
            read_utf8(d.utf8name, NAPI_AUTO_LENGTH)
        } else {
            env.interp()
                .coerce_string(&env.value(d.name))
                .map(|s| s.to_string())
                .unwrap_or_default()
        };
        let enumerable = d.attributes & NAPI_ENUMERABLE != 0;

        if d.method.is_some() {
            let f = make_callback_fn(env.interp(), &pname, d.method, d.data);
            let _ = env.interp().member_set(&target, &pname, f);
        } else if d.getter.is_some() || d.setter.is_some() {
            let g = d
                .getter
                .map(|_| make_callback_fn(env.interp(), &pname, d.getter, d.data));
            let s = d
                .setter
                .map(|_| make_callback_fn(env.interp(), &pname, d.setter, d.data));
            env.interp()
                .define_accessor_value(&target, &pname, g, s, enumerable);
        } else if !d.value.is_null() {
            let _ = env.interp().member_set(&target, &pname, env.value(d.value));
        }
    }

    *result = env.handle(ctor);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_new_instance(
    env: napi_env,
    constructor: napi_value,
    argc: usize,
    argv: *const napi_value,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let ctor = env.value(constructor);
    let mut args = Vec::with_capacity(argc);
    for i in 0..argc {
        args.push(env.value(*argv.add(i)));
    }
    match env.interp().construct_value(ctor, &args) {
        Ok(v) => {
            *result = env.handle(v);
            NAPI_OK
        }
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

// -- promises --

#[no_mangle]
pub unsafe extern "C" fn napi_create_promise(
    env: napi_env,
    deferred: *mut napi_deferred,
    promise: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let interp = env.interp();
    let global = interp.global_this();
    let promise_ctor = match interp.member_get(&global, "Promise") {
        Ok(c) => c,
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };

    // Capture the executor's (resolve, reject) in handles — the Promise constructor calls the
    // executor synchronously, so the slot is filled by the time `construct_value` returns.
    let state = Rc::new(NapiPromiseExecutor {
        slot: Rc::new(std::cell::RefCell::new(None)),
    });
    let slot = state.slot.clone();
    let callable_state = state.clone();
    let executor = interp.new_native_fn(
        "",
        2,
        move |interp, this, args| callable_state.call(interp, this, args),
        trust_js_host::NativeCaptures::Reported(state),
    );

    let p = match interp.construct_value(promise_ctor, &[executor]) {
        Ok(p) => p,
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };
    let Some((resolve, reject)) = slot.borrow_mut().take() else {
        return NAPI_INVALID_ARG;
    };

    let state = env.napi_state();
    let idx = {
        state.deferreds.push(Some(Deferred { resolve, reject }));
        state.deferreds.len() - 1
    };
    *deferred = encode_slot(idx);
    *promise = env.handle(p);
    NAPI_OK
}

/// Shared body for resolve/reject: invoke the stored settler with `value` and consume the deferred.
unsafe fn settle_deferred(
    env: napi_env,
    deferred: napi_deferred,
    value: napi_value,
    reject: bool,
) -> napi_status {
    let env = &mut *env;
    let settler = decode_slot(deferred)
        .and_then(|idx| {
            env.napi_state()
                .deferreds
                .get_mut(idx)
                .and_then(|d| d.take())
        })
        .map(|d| if reject { d.reject } else { d.resolve });
    let Some(settler) = settler else {
        return NAPI_INVALID_ARG;
    };
    let settler = trust_js_host::value(env.interp(), &settler);
    match env
        .interp()
        .invoke(settler, Value::Undefined, &[env.value(value)])
    {
        Ok(_) => NAPI_OK,
        Err(e) => {
            env.set_pending(e);
            NAPI_PENDING_EXCEPTION
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn napi_resolve_deferred(
    env: napi_env,
    deferred: napi_deferred,
    resolution: napi_value,
) -> napi_status {
    settle_deferred(env, deferred, resolution, false)
}

#[no_mangle]
pub unsafe extern "C" fn napi_reject_deferred(
    env: napi_env,
    deferred: napi_deferred,
    rejection: napi_value,
) -> napi_status {
    settle_deferred(env, deferred, rejection, true)
}

// -- buffers / typed arrays --

#[no_mangle]
pub unsafe extern "C" fn napi_create_buffer_copy(
    env: napi_env,
    length: usize,
    data: *const c_void,
    result_data: *mut *mut c_void,
    result: *mut napi_value,
) -> napi_status {
    let env = &mut *env;
    let bytes: &[u8] = if data.is_null() || length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(data as *const u8, length)
    };
    let buf = match env.interp().make_uint8array(bytes) {
        Ok(b) => b,
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };
    if !result_data.is_null() {
        *result_data = env
            .interp()
            .typed_array_raw(&buf)
            .map(|(_, _, p)| p as *mut c_void)
            .unwrap_or(std::ptr::null_mut());
    }
    *result = env.handle(buf);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_create_external_buffer(
    env: napi_env,
    length: usize,
    data: *mut c_void,
    _finalize_cb: *mut c_void,
    _finalize_hint: *mut c_void,
    result: *mut napi_value,
) -> napi_status {
    // No zero-copy over foreign memory: copy the bytes into an engine-owned buffer. The finalize
    // callback is not run (the source is left to the addon / process exit).
    let env = &mut *env;
    let bytes: &[u8] = if data.is_null() || length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(data as *const u8, length)
    };
    let buf = match env.interp().make_uint8array(bytes) {
        Ok(b) => b,
        Err(e) => {
            env.set_pending(e);
            return NAPI_PENDING_EXCEPTION;
        }
    };
    *result = env.handle(buf);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_get_typedarray_info(
    env: napi_env,
    typedarray: napi_value,
    ta_type: *mut c_int,
    length: *mut usize,
    data: *mut *mut c_void,
    arraybuffer: *mut napi_value,
    byte_offset: *mut usize,
) -> napi_status {
    let env = &mut *env;
    let v = env.value(typedarray);
    let Some((code, byte_len, ptr)) = env.interp().typed_array_raw(&v) else {
        return NAPI_INVALID_ARG;
    };
    let elem = match code {
        0..=2 => 1usize,
        3 | 4 => 2,
        5..=7 => 4,
        _ => 8,
    };
    if !ta_type.is_null() {
        *ta_type = code as c_int;
    }
    if !length.is_null() {
        *length = byte_len / elem;
    }
    if !data.is_null() {
        *data = ptr as *mut c_void;
    }
    if !byte_offset.is_null() {
        // `data` already points at the view's first element, so the offset relative to it is 0.
        *byte_offset = 0;
    }
    if !arraybuffer.is_null() {
        *arraybuffer = env.handle(Value::Undefined);
    }
    NAPI_OK
}

// -- async work (run synchronously on the loop thread: correct, just not off-thread) --

#[no_mangle]
pub unsafe extern "C" fn napi_create_async_work(
    env: napi_env,
    _async_resource: napi_value,
    _async_resource_name: napi_value,
    execute: napi_async_execute_callback,
    complete: napi_async_complete_callback,
    data: *mut c_void,
    result: *mut napi_async_work,
) -> napi_status {
    let env = &mut *env;
    let state = env.napi_state();
    state.async_works.push(Some(AsyncWork {
        execute,
        complete,
        data,
    }));
    *result = encode_slot(state.async_works.len() - 1);
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_queue_async_work(
    env: napi_env,
    work: napi_async_work,
) -> napi_status {
    let env = &mut *env;
    let Some(w) = decode_slot(work).and_then(|idx| {
        env.napi_state()
            .async_works
            .get_mut(idx)
            .and_then(|w| w.take())
    }) else {
        return NAPI_INVALID_ARG;
    };
    // Run the two halves inline. `execute` must not touch JS (it normally runs off-thread);
    // `complete` runs on the loop thread with JS access, which is where we already are.
    let mut work_env = Env::new(env.interp() as *mut Ctx);
    if let Some(execute) = w.execute {
        execute(&mut work_env as *mut Env, w.data);
    }
    if let Some(complete) = w.complete {
        complete(&mut work_env as *mut Env, NAPI_OK, w.data);
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_delete_async_work(
    env: napi_env,
    work: napi_async_work,
) -> napi_status {
    let env = &mut *env;
    if let Some(idx) = decode_slot(work) {
        if let Some(slot) = env.napi_state().async_works.get_mut(idx) {
            *slot = None;
        }
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_cancel_async_work(
    _env: napi_env,
    _work: napi_async_work,
) -> napi_status {
    // Work runs synchronously on queue, so by the time anything could cancel it, it is done.
    NAPI_OK
}

// -- threadsafe functions (stubbed: symbols must resolve, but the loop is single-threaded) --

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn napi_create_threadsafe_function(
    _env: napi_env,
    _func: napi_value,
    _async_resource: napi_value,
    _async_resource_name: napi_value,
    _max_queue_size: usize,
    _initial_thread_count: usize,
    _thread_finalize_data: *mut c_void,
    _thread_finalize_cb: *mut c_void,
    _context: *mut c_void,
    _call_js_cb: *mut c_void,
    result: *mut napi_threadsafe_function,
) -> napi_status {
    if !result.is_null() {
        *result = 1 as napi_threadsafe_function;
    }
    NAPI_OK
}

#[no_mangle]
pub unsafe extern "C" fn napi_unref_threadsafe_function(
    _env: napi_env,
    _func: napi_threadsafe_function,
) -> napi_status {
    NAPI_OK
}

/// Take the address of every exported `napi_*` symbol so the linker cannot drop them from the
/// static archive (they are only ever referenced by an addon at runtime, via the dynamic linker,
/// which the Rust link step can't see). Called once from [`op_load_addon`].
pub fn keep_napi_exports() {
    let anchors: [*const (); 64] = [
        napi_get_undefined as *const (),
        napi_get_null as *const (),
        napi_get_global as *const (),
        napi_get_boolean as *const (),
        napi_create_double as *const (),
        napi_create_int32 as *const (),
        napi_create_uint32 as *const (),
        napi_create_int64 as *const (),
        napi_create_string_utf8 as *const (),
        napi_create_object as *const (),
        napi_create_array as *const (),
        napi_create_array_with_length as *const (),
        napi_create_function as *const (),
        napi_typeof as *const (),
        napi_get_value_double as *const (),
        napi_get_value_int32 as *const (),
        napi_get_value_uint32 as *const (),
        napi_get_value_bool as *const (),
        napi_get_value_string_utf8 as *const (),
        napi_coerce_to_string as *const (),
        napi_set_named_property as *const (),
        napi_get_named_property as *const (),
        napi_has_named_property as *const (),
        napi_set_property as *const (),
        napi_get_property as *const (),
        napi_set_element as *const (),
        napi_get_element as *const (),
        napi_get_cb_info as *const (),
        napi_call_function as *const (),
        napi_throw as *const (),
        napi_throw_error as *const (),
        napi_throw_type_error as *const (),
        napi_throw_range_error as *const (),
        napi_create_error as *const (),
        napi_is_exception_pending as *const (),
        napi_get_and_clear_last_exception as *const (),
        napi_open_handle_scope as *const (),
        napi_close_handle_scope as *const (),
        napi_get_version as *const (),
        napi_strict_equals as *const (),
        napi_is_error as *const (),
        napi_get_prototype as *const (),
        napi_create_reference as *const (),
        napi_delete_reference as *const (),
        napi_get_reference_value as *const (),
        napi_reference_ref as *const (),
        napi_reference_unref as *const (),
        napi_wrap as *const (),
        napi_unwrap as *const (),
        napi_remove_wrap as *const (),
        napi_define_class as *const (),
        napi_new_instance as *const (),
        napi_create_promise as *const (),
        napi_resolve_deferred as *const (),
        napi_reject_deferred as *const (),
        napi_create_buffer_copy as *const (),
        napi_create_external_buffer as *const (),
        napi_get_typedarray_info as *const (),
        napi_create_async_work as *const (),
        napi_queue_async_work as *const (),
        napi_delete_async_work as *const (),
        napi_cancel_async_work as *const (),
        napi_create_threadsafe_function as *const (),
        napi_unref_threadsafe_function as *const (),
    ];
    std::hint::black_box(anchors);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use trust_js_host::ValueKind;

    use trust_js_host::RetainedManagedAllocation;

    #[derive(Default)]
    struct RetainedVisit {
        allocations: Vec<RetainedManagedAllocation>,
    }

    impl NativeRetainedMemoryVisitor for RetainedVisit {
        fn allocation(&mut self, allocation: RetainedManagedAllocation) {
            self.allocations.push(allocation);
        }
    }

    /// The engine's context, as N-API's `Env` keeps it: a raw pointer, valid while the engine
    /// lives (its interpreter is boxed), its scope's lifetime erased.
    fn ctx_ptr(engine: &mut trust_js_host::Engine) -> *mut Ctx<'static> {
        engine.scope(|cx| (cx as *mut Ctx<'_>).cast())
    }

    #[test]
    fn to_int32_matches_ecmascript() {
        assert_eq!(to_int32(0.0), 0);
        assert_eq!(to_int32(42.9), 42);
        assert_eq!(to_int32(-1.0), -1);
        assert_eq!(to_int32(4294967296.0), 0); // 2^32 wraps to 0
        assert_eq!(to_int32(4294967297.0), 1); // 2^32 + 1
        assert_eq!(to_int32(2147483648.0), i32::MIN); // 2^31 wraps to negative
        assert_eq!(to_int32(f64::INFINITY), 0);
        assert_eq!(to_int32(f64::NAN), 0);
    }

    #[test]
    fn read_utf8_counted_and_nul_terminated() {
        let s = CString::new("hello").unwrap();
        // NUL-terminated (NAPI_AUTO_LENGTH).
        assert_eq!(unsafe { read_utf8(s.as_ptr(), NAPI_AUTO_LENGTH) }, "hello");
        // Explicit length truncates.
        assert_eq!(unsafe { read_utf8(s.as_ptr(), 3) }, "hel");
        // A null pointer reads as empty.
        assert_eq!(unsafe { read_utf8(std::ptr::null(), NAPI_AUTO_LENGTH) }, "");
    }

    /// An object held only by an N-API handle: made in a frame of its own, so no stack word but
    /// the handle arena's names it.
    #[inline(never)]
    fn marked_object(env: &mut Env, mark: f64) -> napi_value {
        let mut object: napi_value = std::ptr::null_mut();
        let mut value: napi_value = std::ptr::null_mut();
        unsafe {
            assert_eq!(napi_create_object(env, &mut object), NAPI_OK);
            assert_eq!(napi_create_double(env, mark, &mut value), NAPI_OK);
            assert_eq!(
                napi_set_named_property(env, object, c"mark".as_ptr(), value),
                NAPI_OK
            );
        }
        object
    }

    /// The values behind a call's napi_values stay alive while the call runs, whatever
    /// collections its callbacks into JavaScript cause.
    #[test]
    fn handles_keep_their_values_across_a_collection() {
        let mut engine = trust_js_host::Engine::new();
        let mut env = Env::new(ctx_ptr(&mut engine));
        let objects: Vec<napi_value> = (0..64).map(|i| marked_object(&mut env, i as f64)).collect();
        let ctx = unsafe { env.interp() };
        ctx.collect_garbage_for_host();
        for _ in 0..2000 {
            ctx.new_object();
        }
        ctx.collect_garbage_for_host();
        for (i, object) in objects.into_iter().enumerate() {
            let mut mark: napi_value = std::ptr::null_mut();
            let mut number = 0.0;
            unsafe {
                assert_eq!(
                    napi_get_named_property(&mut env, object, c"mark".as_ptr(), &mut mark),
                    NAPI_OK
                );
                assert_eq!(napi_get_value_double(&mut env, mark, &mut number), NAPI_OK);
            }
            assert_eq!(number, i as f64);
        }
    }

    /// Wrap 100 objects held only by the handles of a call that then returns, in a frame of its
    /// own; their addresses.
    #[inline(never)]
    fn wrap_objects_of_a_finished_call(engine: &mut trust_js_host::Engine) -> Vec<usize> {
        let mut call = Env::new(ctx_ptr(engine));
        (0..100)
            .map(|i| {
                let object = marked_object(&mut call, i as f64);
                unsafe {
                    assert_eq!(
                        napi_wrap(
                            &mut call,
                            object,
                            (i + 1) as *mut c_void,
                            None,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                        ),
                        NAPI_OK
                    );
                    let mut native: *mut c_void = std::ptr::null_mut();
                    assert_eq!(napi_unwrap(&mut call, object, &mut native), NAPI_OK);
                    assert_eq!(native, (i + 1) as *mut c_void);
                    call.interp().object_addr(&call.value(object)).unwrap()
                }
            })
            .collect()
    }

    /// A wrap ends with its object: the collection that finds the object dead clears the
    /// wrap's weak handle, and its sweep removes the entry, so an address reused by a new object
    /// never yields the old pointer.
    #[test]
    fn wraps_end_with_their_objects() {
        let mut engine = trust_js_host::Engine::new();
        let addresses = wrap_objects_of_a_finished_call(&mut engine);
        let mut outer = Env::new(ctx_ptr(&mut engine));
        let ctx = unsafe { outer.interp() };
        ctx.collect_garbage_for_host();
        let state = unsafe { outer.napi_state() };
        for address in &addresses {
            assert!(
                !state.wraps.contains_key(address),
                "a dead object's wrap is gone"
            );
        }
        // New objects wrapped at reused addresses have entries of their own.
        for i in 0..200 {
            let object = marked_object(&mut outer, i as f64);
            unsafe {
                napi_wrap(
                    &mut outer,
                    object,
                    std::ptr::null_mut(),
                    None,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
            }
        }
        let state = unsafe { outer.napi_state() };
        assert!(addresses.iter().all(|address| {
            state
                .wraps
                .get(address)
                .is_none_or(|wrap| !wrap.object.is_cleared())
        }));
    }

    /// A reference made in a call that has returned, to a new object marked `mark`, with
    /// `count`; in a frame of its own.
    #[inline(never)]
    fn reference_of_a_finished_call(
        engine: &mut trust_js_host::Engine,
        mark: f64,
        count: u32,
    ) -> napi_ref {
        let mut call = Env::new(ctx_ptr(engine));
        let object = marked_object(&mut call, mark);
        let mut reference: napi_ref = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                napi_create_reference(&mut call, object, count, &mut reference),
                NAPI_OK
            );
        }
        reference
    }

    /// The mark of the object `reference` names, or `None` once it has died; read in a call of
    /// its own.
    #[inline(never)]
    fn referenced_mark(engine: &mut trust_js_host::Engine, reference: napi_ref) -> Option<f64> {
        let mut call = Env::new(ctx_ptr(engine));
        let mut object: napi_value = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                napi_get_reference_value(&mut call, reference, &mut object),
                NAPI_OK
            );
            if object.is_null() {
                return None;
            }
            let mut mark: napi_value = std::ptr::null_mut();
            let mut number = 0.0;
            assert_eq!(
                napi_get_named_property(&mut call, object, c"mark".as_ptr(), &mut mark),
                NAPI_OK
            );
            assert_eq!(napi_get_value_double(&mut call, mark, &mut number), NAPI_OK);
            Some(number)
        }
    }

    /// Add `delta` (+1 or -1) to a reference's count; the new count.
    fn adjust_reference(
        engine: &mut trust_js_host::Engine,
        reference: napi_ref,
        delta: i32,
    ) -> u32 {
        let mut call = Env::new(ctx_ptr(engine));
        let mut count = u32::MAX;
        let status = unsafe {
            if delta > 0 {
                napi_reference_ref(&mut call, reference, &mut count)
            } else {
                napi_reference_unref(&mut call, reference, &mut count)
            }
        };
        assert_eq!(status, NAPI_OK);
        count
    }

    /// References follow Node-API's counts: a reference keeps its object while its count is
    /// above 0, and at 0 lets it die (reading NULL afterwards); a count taken to 0 and back
    /// before a collection keeps it.
    #[test]
    fn references_are_strong_above_a_count_of_zero_and_weak_at_zero() {
        let mut engine = trust_js_host::Engine::new();
        let strong = reference_of_a_finished_call(&mut engine, 1.0, 1);
        let weak = reference_of_a_finished_call(&mut engine, 2.0, 0);
        for _ in 0..2 {
            engine.collect_garbage_at_idle();
        }
        assert_eq!(referenced_mark(&mut engine, strong), Some(1.0));
        assert_eq!(referenced_mark(&mut engine, weak), None, "count 0: it died");
        assert_eq!(adjust_reference(&mut engine, strong, -1), 0);
        assert_eq!(adjust_reference(&mut engine, strong, 1), 1);
        assert_eq!(adjust_reference(&mut engine, strong, 1), 2);
        engine.collect_garbage_at_idle();
        assert_eq!(referenced_mark(&mut engine, strong), Some(1.0));
        assert_eq!(adjust_reference(&mut engine, strong, -1), 1);
        assert_eq!(adjust_reference(&mut engine, strong, -1), 0);
        assert_eq!(
            referenced_mark(&mut engine, strong),
            Some(1.0),
            "not collected yet"
        );
        engine.collect_garbage_at_idle();
        assert_eq!(
            referenced_mark(&mut engine, strong),
            None,
            "count 0: it died"
        );
        assert_eq!(adjust_reference(&mut engine, strong, 1), 1);
        assert_eq!(referenced_mark(&mut engine, strong), None, "and stays dead");
        let mut call = Env::new(ctx_ptr(&mut engine));
        for reference in [strong, weak] {
            assert_eq!(
                unsafe { napi_delete_reference(&mut call, reference) },
                NAPI_OK
            );
        }
    }

    thread_local! {
        /// The (data, hint) pairs `record_finalize` was called with.
        static FINALIZED: std::cell::RefCell<Vec<(usize, usize)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    unsafe extern "C" fn record_finalize(_env: napi_env, data: *mut c_void, hint: *mut c_void) {
        FINALIZED.with(|calls| calls.borrow_mut().push((data as usize, hint as usize)));
    }

    unsafe extern "C" fn empty_callback(_env: napi_env, _info: napi_callback_info) -> napi_value {
        std::ptr::null_mut()
    }

    /// Wrap a new object, made in a call that has returned, with native pointer `n` and the
    /// recording finalizer with hint `n + 100`, removing the wrap again when `remove`; the
    /// wrap's reference.
    #[inline(never)]
    fn wrap_with_finalizer(engine: &mut trust_js_host::Engine, n: usize, remove: bool) -> napi_ref {
        let mut call = Env::new(ctx_ptr(engine));
        let object = marked_object(&mut call, n as f64);
        let mut reference: napi_ref = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                napi_wrap(
                    &mut call,
                    object,
                    n as *mut c_void,
                    Some(record_finalize),
                    (n + 100) as *mut c_void,
                    &mut reference,
                ),
                NAPI_OK
            );
            if remove {
                let mut native: *mut c_void = std::ptr::null_mut();
                assert_eq!(napi_remove_wrap(&mut call, object, &mut native), NAPI_OK);
                assert_eq!(native, n as *mut c_void);
            }
        }
        reference
    }

    /// A wrapped object can die (the wrap and its reference, at a count of 0, are weak), and
    /// its finalizer then runs with its data and hint: after the collection, at the next call
    /// into an addon. A wrap removed first, or whose object a counted reference keeps, is not
    /// finalized.
    #[test]
    fn a_wrapped_object_that_dies_is_finalized_at_the_next_call_into_an_addon() {
        let mut engine = trust_js_host::Engine::new();
        let dying: Vec<napi_ref> = (1..=3)
            .map(|n| wrap_with_finalizer(&mut engine, n, false))
            .collect();
        let _removed = wrap_with_finalizer(&mut engine, 4, true);
        let kept = wrap_with_finalizer(&mut engine, 5, false);
        assert_eq!(adjust_reference(&mut engine, kept, 1), 1);
        engine.collect_garbage_at_idle();
        assert!(
            FINALIZED.with(|calls| calls.borrow().is_empty()),
            "no finalizer runs inside the collection"
        );
        let call = engine.scope(|cx| {
            invoke_napi_callback(
                cx,
                Some(empty_callback),
                std::ptr::null_mut(),
                Value::Undefined,
                &[],
            )
            .is_ok()
        });
        assert!(call);
        let mut finalized = FINALIZED.with(|calls| calls.take());
        finalized.sort_unstable();
        assert_eq!(finalized, [(1, 101), (2, 102), (3, 103)]);
        for reference in dying {
            assert_eq!(referenced_mark(&mut engine, reference), None);
        }
        assert_eq!(referenced_mark(&mut engine, kept), Some(5.0));
    }

    /// A deferred made in a call that has returned, its promise stored as `globalThis.pending`.
    #[inline(never)]
    fn deferred_of_a_finished_call(engine: &mut trust_js_host::Engine) -> napi_deferred {
        let mut call = Env::new(ctx_ptr(engine));
        let mut deferred: napi_deferred = std::ptr::null_mut();
        let mut promise: napi_value = std::ptr::null_mut();
        let mut global: napi_value = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                napi_create_promise(&mut call, &mut deferred, &mut promise),
                NAPI_OK
            );
            assert_eq!(napi_get_global(&mut call, &mut global), NAPI_OK);
            assert_eq!(
                napi_set_named_property(&mut call, global, c"pending".as_ptr(), promise),
                NAPI_OK
            );
        }
        deferred
    }

    /// A deferred holds its promise's resolving functions, which nothing else does, across
    /// collections, and settles the promise afterwards.
    #[test]
    fn deferreds_hold_their_resolving_functions_across_collections() {
        let mut engine = trust_js_host::Engine::new();
        let deferred = deferred_of_a_finished_call(&mut engine);
        assert!(engine
            .eval(
                "globalThis.settled = 'no'; pending.then(v => { settled = v; }); 0",
                false
            )
            .is_ok());
        for _ in 0..2 {
            assert!(engine
                .eval(
                    "var churn = []; for (let i = 0; i < 3000; i++) churn.push({i}); churn = null; 0",
                    false,
                )
                .is_ok());
            engine.collect_garbage_at_idle();
        }
        let mut call = Env::new(ctx_ptr(&mut engine));
        let mut text: napi_value = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                napi_create_string_utf8(
                    &mut call,
                    c"resolved".as_ptr(),
                    NAPI_AUTO_LENGTH,
                    &mut text
                ),
                NAPI_OK
            );
            assert_eq!(napi_resolve_deferred(&mut call, deferred, text), NAPI_OK);
        }
        drop(call);
        engine.run_microtasks();
        let settled = engine.scope(|cx| {
            let global = cx.global_this();
            let settled = cx
                .member_get(&global, "settled")
                .unwrap_or_else(|_| panic!("settled"));
            settled.as_text().map(|text| text.into_owned())
        });
        assert_eq!(settled.as_deref(), Some("resolved"));
    }

    #[test]
    fn get_prototype_runs_the_proxy_trap() {
        let mut engine = trust_js_host::Engine::new();
        engine.scope(get_prototype_checks);
    }

    fn get_prototype_checks(cx: &mut Ctx<'_>) {
        let Ok(Ok(values)) = cx.eval_value(
            "globalThis.target = {}; globalThis.reported = {};
             [new Proxy(target, {getPrototypeOf: () => reported}),
              new Proxy(target, {getPrototypeOf() { throw new RangeError('trap'); }}),
              Object.create(reported), null]",
        ) else {
            panic!("fixture evaluates")
        };
        let mut env = Env::new(cx as *mut Ctx);
        let interp = unsafe { env.interp() };
        let mut item = |index: &str| {
            interp
                .member_get(&values, index)
                .unwrap_or_else(|_| panic!("fixture element"))
        };
        let (trapped, throwing, ordinary, null) = (item("0"), item("1"), item("2"), item("3"));
        let reported = interp
            .member_get(&interp.global_this(), "reported")
            .unwrap_or_else(|_| panic!("reported"));
        let mut result: napi_value = std::ptr::null_mut();
        for subject in [trapped, ordinary] {
            let handle = env.handle(subject);
            assert_eq!(
                unsafe { napi_get_prototype(&mut env, handle, &mut result) },
                NAPI_OK
            );
            let ValueKind::Obj(prototype) = (unsafe { env.value(result) }).kind() else {
                panic!("an object prototype")
            };
            let ValueKind::Obj(expected) = &(reported).kind() else {
                panic!("an object")
            };
            assert!(prototype == *expected);
        }
        let handle = env.handle(throwing);
        assert_eq!(
            unsafe { napi_get_prototype(&mut env, handle, &mut result) },
            NAPI_PENDING_EXCEPTION
        );
        assert!(env.take_pending(interp).is_some());
        let handle = env.handle(null);
        assert_eq!(
            unsafe { napi_get_prototype(&mut env, handle, &mut result) },
            NAPI_OBJECT_EXPECTED
        );
    }

    /// The callback closures' Rust state reports its allocations only (S6 step 11b): the
    /// constructor's prototype is an engine capture, passed to its call, and the promise
    /// executor keeps the resolving functions in handles.
    #[test]
    fn retained_callback_wrappers_preserve_callback_and_capture_semantics() {
        let mut engine = trust_js_host::Engine::new();
        engine.scope(retained_callback_checks);
    }

    fn retained_callback_checks(interp: &mut Ctx<'_>) {
        let callback = NapiCallbackClosure {
            callback: None,
            data: std::ptr::null_mut(),
        };
        let Ok(callback_result) = callback.call(interp, Value::Undefined, &[]) else {
            panic!("empty N-API callback should not throw")
        };
        assert!(matches!((callback_result).kind(), ValueKind::Undefined));
        let mut visit = RetainedVisit::default();
        callback.scan_retained_memory(&mut visit);
        assert!(visit.allocations.is_empty());

        let prototype = Value::Obj(interp.new_object());
        let constructor = NapiConstructorClosure {
            callback: None,
            data: std::ptr::null_mut(),
        };
        let Ok(constructor_result) = constructor.call(interp, Value::Undefined, &[], &prototype)
        else {
            panic!("empty N-API constructor should not throw")
        };
        let made = interp.prototype_of(&constructor_result);
        assert!(interp.values_strict_equal(&made, &prototype));
        constructor.scan_retained_memory(&mut visit);
        assert!(visit.allocations.is_empty());

        let slot = Rc::new(std::cell::RefCell::new(None));
        let executor = NapiPromiseExecutor { slot: slot.clone() };
        assert!(executor
            .call(
                interp,
                Value::Undefined,
                &[Value::Num(3.0), Value::Num(4.0)],
            )
            .is_ok());
        {
            let slot = slot.borrow();
            let (resolve, reject) = slot.as_ref().expect("the executor ran");
            assert_eq!(resolve.get(interp).unwrap().as_num_opt(), Some(3.0));
            assert_eq!(reject.get(interp).unwrap().as_num_opt(), Some(4.0));
        }
        executor.scan_retained_memory(&mut visit);
        assert_eq!(visit.allocations.len(), 1);
    }

    #[test]
    fn napi_double_ingress_canonicalizes_every_nan_prefix_and_preserves_scalars() {
        let mut engine = trust_js_host::Engine::new();
        let mut env = Env::new(ctx_ptr(&mut engine));
        let numbers = (0x7ff0u64..=0x7fff)
            .chain(0xfff0..=0xffff)
            .flat_map(|high| {
                [
                    f64::from_bits(high << 48 | 1),
                    f64::from_bits(high << 48 | 0x8000_1234_5678),
                ]
            })
            .chain([
                0.0,
                -0.0,
                f64::from_bits(1),
                -f64::from_bits(1),
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.5,
            ]);
        for number in numbers {
            let mut handle = std::ptr::null_mut();
            assert_eq!(
                unsafe { napi_create_double(&mut env, number, &mut handle) },
                NAPI_OK
            );
            let value = unsafe { env.value(handle) };
            assert_eq!(std::mem::size_of_val(&value), 8);
            let word = unsafe { std::ptr::from_ref(&value).cast::<u64>().read() };
            let expected = if number.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                number.to_bits()
            };
            assert_eq!(word, expected);
            let mut read = 0.0;
            assert_eq!(
                unsafe { napi_get_value_double(&mut env, handle, &mut read) },
                NAPI_OK
            );
            assert_eq!(read.to_bits(), expected);
        }
    }
}
