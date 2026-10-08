//! The built-in objects and global functions. This is the realm: a freshly-constructed [`Interp`]
//! calls [`install`] to populate `globalThis`, the standard constructors/prototypes, `Math`, and
//! the global functions. test262 exercises this realm as the primary conformance oracle.

use crate::interpreter::{Abrupt, Interp, MAX_ARRAY_OP_LEN, MAX_BUFFER_BYTES, MAX_STR_LEN};
use crate::lstr::LStr;
use crate::value::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::rc::Rc;

// Per-object modules split out of this file (behavior-preserving). Shared helpers remain here and
// are reachable from each submodule via `use super::*`.
mod atomics;
mod collections;
pub(crate) use collections::{is_operand_only_native, map_get, map_set, weak_map_get};
mod dataview;
mod date;
pub(crate) use date::system_time_zone;
#[cfg(test)]
pub(crate) use date::TEST_TIME_ZONE;
mod disposable;
mod errors;
mod function_proto;
mod globals;
mod host;
mod json;
mod math;
mod primitives;
mod promise;
mod proxy;
mod reflect;
mod regexp;
mod shadowrealm;
mod typedarray;
pub(crate) use typedarray::{array_buffer_with_storage, typed_array_from_bytes};
mod weakrefs;

#[cfg(test)]
#[path = "../array_length_tests.rs"]
mod array_length_tests;

#[cfg(test)]
#[path = "../array_shift_tests.rs"]
mod array_shift_tests;

pub(crate) use function_proto::nf_function_call;
pub(crate) use globals::nf_is_nan;
pub(crate) use math::{is_numeric_only_native, nf_math_sqrt};
pub(crate) use primitives::nf_number_is_nan;
pub(crate) use reflect::nf_reflect_apply;

/// `args[i]` or `undefined`.
fn arg(args: &[Value], i: usize) -> Value {
    args.get(i).cloned().unwrap_or(Value::Undefined)
}

/// ToNumber of an argument position, with the Number identity fast path.
///
/// ECMA-262 §7.1.4.1 makes `ToNumber` the identity for a Number and it has no other observable
/// effect, so a primitive `Number` argument can skip the generic coercion dispatch entirely. Every
/// other input — including an absent argument, which is `undefined` and coerces to NaN — keeps the
/// complete coercive path with its abrupt completions and `valueOf`/`toString` ordering.
#[inline]
fn arg_to_number(i: &mut Interp, args: &[Value], idx: usize) -> Result<f64, Value> {
    match args.get(idx) {
        Some(Value::Num(n)) => Ok(*n),
        _ => ab(i.to_number(&arg(args, idx))),
    }
}

/// Map an `Abrupt` (which, from inside a native function, can only be a `Throw`) to its value so it
/// fits the native `Result<_, Value>` contract.
fn ab<T>(r: Result<T, Abrupt>) -> Result<T, Value> {
    r.map_err(|a| match a {
        Abrupt::Throw(v) => v,
        _ => Value::Undefined,
    })
}

/// The default `Function.prototype[@@hasInstance]` intrinsic. Kept as a named native so the
/// `instanceof` fast path can recognize the unmodified intrinsic by function identity and call
/// `OrdinaryHasInstance` directly, without building a second native-call frame.
pub(crate) fn default_has_instance(
    i: &mut Interp,
    this: Value,
    a: &[Value],
) -> Result<Value, Value> {
    Ok(Value::Bool(ab(i.ordinary_has_instance(&this, &arg(a, 0)))?))
}

/// `Function.prototype.apply` is named so the JIT call cache can recognize the unmodified
/// intrinsic and bypass this native frame for the common dense-arguments → compiled-function
/// case. The JIT helper falls back to this exact implementation whenever its guards do not hold.
pub(crate) fn nf_function_apply(
    i: &mut Interp,
    this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let this_arg = arg(args, 0);
    let list = match arg(args, 1) {
        Value::Undefined | Value::Null => i.take_native_arg_buf(0),
        Value::Obj(o) => {
            let len = ab(i.checked_array_len(&o))?;
            let direct_dense = i.ordinary_get_ptr(Rc::as_ptr(&o) as usize)
                && !i.mapped_arguments.contains_key(&(Rc::as_ptr(&o) as usize));
            let mut v = i.take_native_arg_buf(len);
            for k in 0..len {
                // Arrays and unmapped arguments objects overwhelmingly contain plain own dense
                // entries. Read those by slot: the generic path allocates a decimal key and walks
                // the prototype chain for every argument. Holes, accessors, proxies, typed
                // arrays, and mapped arguments retain full [[Get]] semantics.
                let own = direct_dense.then(|| {
                    let b = o.borrow();
                    b.props
                        .get_index(k as u32)
                        .filter(|p| !p.accessor())
                        .map(|p| p.value())
                });
                match own.flatten() {
                    Some(value) => v.push(value),
                    None => v.push(ab(i.get_member(&Value::Obj(o.clone()), &k.to_string()))?),
                }
            }
            v
        }
        _ => return Err(i.make_error("TypeError", "apply: argument list must be array-like")),
    };
    // The recycled buffer is returned even when the call throws: `call` never retains the slice
    // (every callee copies arguments into its own activation), so recycling is unobservable.
    let call_result = i.call(this, this_arg, &list);
    i.recycle_native_arg_buf(list);
    ab(call_result)
}

fn this_obj(this: &Value) -> Option<Gc> {
    this.as_obj().cloned()
}

/// The newTarget realm's %RegExp.prototype% (GetFunctionRealm fallback for RegExp construction).
pub(crate) fn regexp_realm_proto(i: &mut Interp, nt: &Value) -> Result<Option<Gc>, Value> {
    ctor_realm_proto(i, nt, "RegExp")
}

/// Set(O, P, V, true): perform [[Set]] and throw a TypeError if it returns false, matching the
/// spec's `Set(..., Throw=true)` used by Array mutators regardless of the surrounding strict mode.
fn set_throw(i: &mut Interp, base: &Value, key: &str, value: Value) -> Result<(), Value> {
    // Overwriting an existing own writable data property (the shape of every hot builtin write,
    // e.g. a RegExp's `lastIndex` after each exec) needs none of the [[Set]] machinery.
    if let Value::Obj(o) = base {
        if matches!(o.borrow().exotic, crate::value::Exotic::None)
            && i.ordinary_get_ptr(Rc::as_ptr(o) as usize)
        {
            let mut b = o.borrow_mut();
            if let Some(p) = b.props.get_mut(key) {
                if !p.accessor() && p.writable() {
                    p.set_value(value);
                    return Ok(());
                }
            }
        }
    }
    let ok = ab(i.set_member_recv(base, key, value, base.clone()))?;
    if !ok {
        return Err(i.make_error(
            "TypeError",
            format!("Cannot assign to read only property '{key}'"),
        ));
    }
    Ok(())
}

/// A `'static` copy of an intrinsic name built at Realm creation (`%Intl.Collator%`,
/// `%AsyncFunction.prototype%`, …), allocated once per process: every Realm of every Agent
/// shares the copy, so creating Realms (one per browsing context) allocates no new names.
pub(crate) fn static_name(name: &str) -> &'static str {
    static NAMES: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut names = NAMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&existing) = names.iter().find(|&&existing| existing == name) {
        return existing;
    }
    let interned: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.push(interned);
    interned
}

/// Define an internal-slot-style data property: writable, non-enumerable, non-configurable.
fn set_internal(obj: &Gc, key: &str, v: Value) {
    obj.borrow_mut()
        .props
        .insert(key, Property::data(v, true, false, false));
}

/// ToDateTimeOptions default for toLocaleDateString/toLocaleTimeString: when `options` is undefined,
/// supply the date-only or time-only component defaults; otherwise pass the options through.
fn date_style_default(i: &mut Interp, options: &Value, date: bool) -> Result<Value, Value> {
    // ToDateTimeOptions(options, required, defaults) for toLocaleDateString(date)/toLocaleTimeString(time):
    // add the dimension's numeric defaults unless a component of that dimension (or a dateStyle/
    // timeStyle) is already present. The user's other-dimension components are kept (via the proto chain).
    let user_obj = match options {
        Value::Undefined => None,
        Value::Obj(o) => Some(o.clone()),
        _ => return Err(i.make_error("TypeError", "options must be an object")),
    };
    let dim: &[&str] = if date {
        &["weekday", "year", "month", "day"]
    } else {
        &[
            "dayPeriod",
            "hour",
            "minute",
            "second",
            "fractionalSecondDigits",
        ]
    };
    let mut need = true;
    if user_obj.is_some() {
        for k in dim.iter().chain(["dateStyle", "timeStyle"].iter()) {
            if !matches!(ab(i.get_member(options, k))?, Value::Undefined) {
                need = false;
                break;
            }
        }
    }
    let o = i.new_object();
    if let Some(uo) = &user_obj {
        o.borrow_mut().proto = Some(uo.clone());
    }
    if need {
        let defs: &[&str] = if date {
            &["year", "month", "day"]
        } else {
            &["hour", "minute", "second"]
        };
        for k in defs {
            set_data(&o, k, Value::lstr("numeric"));
        }
    }
    Ok(Value::Obj(o))
}

/// ToDateTimeOptions(options, "any", "all") for toLocaleString: add both date and time numeric
/// defaults unless the caller already requested a date/time component or a dateStyle/timeStyle.
fn date_all_default(i: &mut Interp, options: &Value) -> Result<Value, Value> {
    let user_obj = match options {
        Value::Undefined => None,
        Value::Obj(o) => Some(o.clone()),
        _ => return Err(i.make_error("TypeError", "options must be an object")),
    };
    let mut need = true;
    if user_obj.is_some() {
        for k in [
            "weekday",
            "year",
            "month",
            "day",
            "dayPeriod",
            "hour",
            "minute",
            "second",
            "fractionalSecondDigits",
            "dateStyle",
            "timeStyle",
        ] {
            if !matches!(ab(i.get_member(options, k))?, Value::Undefined) {
                need = false;
                break;
            }
        }
    }
    let o = i.new_object();
    if let Some(uo) = &user_obj {
        o.borrow_mut().proto = Some(uo.clone());
    }
    if need {
        for k in ["year", "month", "day", "hour", "minute", "second"] {
            set_data(&o, k, Value::lstr("numeric"));
        }
    }
    Ok(Value::Obj(o))
}

/// [`intl_delegate`] without ECMA-402: the `toLocale*`/`localeCompare` surface must still exist
/// (it's ECMA-262), so degrade the way engines built without i18n do — the locale-independent
/// equivalents. Services with no reasonable 262-only fallback report the missing feature.
#[cfg(not(feature = "intl"))]
pub(crate) fn intl_delegate(
    i: &mut Interp,
    service: &str,
    _locales: Value,
    _options: Value,
    method: &str,
    call_args: &[Value],
) -> Result<Value, Value> {
    match (service, method) {
        // String.prototype.localeCompare → code-unit order (the spec's only hard requirement
        // is a consistent total order over equal/before/after).
        ("Collator", "compare") => {
            let a = ab(i.to_string(&arg(call_args, 0)))?;
            let b = ab(i.to_string(&arg(call_args, 1)))?;
            Ok(Value::Num(match crate::jstr::cmp_units(&a, &b) {
                std::cmp::Ordering::Less => -1.0,
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Greater => 1.0,
            }))
        }
        // Number/BigInt.prototype.toLocaleString → ToString.
        ("NumberFormat", "format") => Ok(Value::Str(ab(i.to_string(&arg(call_args, 0)))?)),
        // Date.prototype.toLocale{,Date,Time}String → the plain toString rendering.
        ("DateTimeFormat", "format") => {
            let t = match arg(call_args, 0) {
                Value::Num(t) => t,
                other => ab(i.to_number(&other))?,
            };
            Ok(Value::lstr(date::date_to_string(t)))
        }
        _ => Err(i.make_error(
            "TypeError",
            format!("Intl.{service} is unavailable: lumen was built without the `intl` feature"),
        )),
    }
}

/// Construct `Intl.<service>(locales, options)` and invoke `method(args…)` on it. Used to route the
/// `toLocale*`/`localeCompare` methods through the Intl services now that they exist.
#[cfg(feature = "intl")]
pub(crate) fn intl_delegate(
    i: &mut Interp,
    service: &str,
    locales: Value,
    options: Value,
    method: &str,
    call_args: &[Value],
) -> Result<Value, Value> {
    // Use the cached intrinsic (immune to tainted Intl globals), falling back to the global.
    let ctor = match i.extra_protos.get(format!("%Intl.{service}%").as_str()) {
        Some(c) => Value::Obj(c.clone()),
        None => {
            let intl = ab(i.get_member(&Value::Obj(i.global.clone()), "Intl"))?;
            ab(i.get_member(&intl, service))?
        }
    };
    let inst = ab(i.construct(ctor, &[locales, options]))?;
    let f = ab(i.get_member(&inst, method))?;
    ab(i.call(f, inst, call_args))
}

/// ECMA-262 [[OwnPropertyKeys]]() for any object, as String/Symbol key values in the spec's
/// order: a Proxy runs §10.5.11; a TypedArray lists its valid integer indices first
/// (§10.4.5.7); every other object uses OrdinaryOwnPropertyKeys order (array-index keys
/// ascending, then strings, then symbols, each in insertion order).
pub(crate) fn object_own_property_keys(i: &mut Interp, o: &Gc) -> Result<Vec<Value>, Value> {
    ab(i.defer_trigger(o, None))?;
    if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
        return proxy_own_keys(i, &target, &handler);
    }
    let mut out: Vec<Value> = if let Some(info) = ta_info(i, o) {
        (0..i.ta_len(&info).unwrap_or(0))
            .map(|k| Value::lstr(k.to_string()))
            .collect()
    } else {
        Vec::new()
    };
    for k in ordinary_own_keys_ordered(i, o)? {
        if Interp::is_sym_key(&k) {
            if let Some(s) = i.sym_from_key(&k) {
                out.push(s);
            }
        } else {
            out.push(Value::lstr(k));
        }
    }
    Ok(out)
}

/// Proxy `[[OwnPropertyKeys]]`: the trap result (must be a list of strings/symbols) or the target's
/// own keys.
pub(crate) fn proxy_own_keys(
    i: &mut Interp,
    target: &Value,
    handler: &Value,
) -> Result<Vec<Value>, Value> {
    // ECMA-262 §10.5.11, steps 1-22 in order.
    if matches!(handler, Value::Null) {
        return Err(i.make_error("TypeError", "proxy is revoked"));
    }
    let Value::Obj(target_object) = target else {
        return Ok(Vec::new());
    };
    let trap = ab(i.get_member(handler, "ownKeys"))?;
    if matches!(trap, Value::Undefined | Value::Null) {
        return object_own_property_keys(i, target_object);
    }
    if !trap.is_callable() {
        return Err(i.make_error("TypeError", "proxy 'ownKeys' trap is not callable"));
    }
    let res = ab(i.call(trap, handler.clone(), std::slice::from_ref(target)))?;
    if !matches!(res, Value::Obj(_)) {
        return Err(i.make_error("TypeError", "ownKeys trap must return an array-like object"));
    }
    let keys = ab(i.create_list_from_arraylike(&res))?;
    let mut key_strs: Vec<crate::value::PropertyKey> = Vec::with_capacity(keys.len());
    for k in &keys {
        if !matches!(k, Value::Str(_) | Value::Sym(_)) {
            return Err(i.make_error(
                "TypeError",
                "ownKeys trap result must contain only strings and symbols",
            ));
        }
        key_strs.push(ab(i.to_property_key(k))?);
    }
    let mut unchecked: std::collections::HashSet<&str> =
        key_strs.iter().map(|s| s.as_str()).collect();
    if unchecked.len() != key_strs.len() {
        return Err(i.make_error("TypeError", "ownKeys trap result has duplicate keys"));
    }
    let extensible_target = js_is_extensible(i, target)?;
    let target_keys = object_own_property_keys(i, target_object)?;
    let mut configurable_keys = Vec::new();
    let mut nonconfigurable_keys = Vec::new();
    for key in target_keys {
        let key = ab(i.to_property_key(&key))?;
        match object_get_own_property(i, target_object, &key)? {
            Some(desc) if !desc.configurable() => nonconfigurable_keys.push(key),
            _ => configurable_keys.push(key),
        }
    }
    if extensible_target && nonconfigurable_keys.is_empty() {
        return Ok(keys);
    }
    for key in &nonconfigurable_keys {
        if !unchecked.remove(key.as_str()) {
            return Err(i.make_error(
                "TypeError",
                "ownKeys trap omitted a non-configurable target key",
            ));
        }
    }
    if extensible_target {
        return Ok(keys);
    }
    for key in &configurable_keys {
        if !unchecked.remove(key.as_str()) {
            return Err(i.make_error(
                "TypeError",
                "ownKeys trap omitted a key of a non-extensible target",
            ));
        }
    }
    if !unchecked.is_empty() {
        return Err(i.make_error(
            "TypeError",
            "ownKeys trap added a key to a non-extensible target",
        ));
    }
    Ok(keys)
}

/// Proxy `[[GetPrototypeOf]]`: call the trap or forward to the target.
/// `[[IsExtensible]]`, proxy-aware (recurses into a proxy target, enforcing the trap invariant).
pub(crate) fn js_is_extensible(i: &mut Interp, obj: &Value) -> Result<bool, Value> {
    if let Some((target, handler)) = proxy_pair(i, obj) {
        if matches!(handler, Value::Null) {
            return Err(i.make_error("TypeError", "proxy is revoked"));
        }
        let trap = ab(i.get_member(&handler, "isExtensible"))?;
        if matches!(trap, Value::Undefined | Value::Null) {
            return js_is_extensible(i, &target);
        }
        if !trap.is_callable() {
            return Err(i.make_error("TypeError", "proxy 'isExtensible' trap is not callable"));
        }
        let res = ab(i.call(trap, handler, std::slice::from_ref(&target)))?;
        let result = i.to_boolean(&res);
        if result != js_is_extensible(i, &target)? {
            return Err(i.make_error("TypeError", "proxy 'isExtensible' must match the target"));
        }
        return Ok(result);
    }
    Ok(matches!(obj, Value::Obj(o) if o.borrow().extensible))
}

/// `[[SetPrototypeOf]]`, proxy-aware. Returns whether the operation succeeded (Reflect-style).
fn js_set_prototype_of(i: &mut Interp, obj: &Value, proto: &Value) -> Result<bool, Value> {
    let o = match obj {
        Value::Obj(o) => o.clone(),
        _ => return Ok(true),
    };
    if let Some((target, handler)) = proxy_pair(i, obj) {
        if matches!(handler, Value::Null) {
            return Err(i.make_error("TypeError", "proxy is revoked"));
        }
        let trap = ab(i.get_member(&handler, "setPrototypeOf"))?;
        if matches!(trap, Value::Undefined | Value::Null) {
            return js_set_prototype_of(i, &target, proto);
        }
        if !trap.is_callable() {
            return Err(i.make_error("TypeError", "proxy 'setPrototypeOf' trap is not callable"));
        }
        let res = ab(i.call(trap, handler, &[target.clone(), proto.clone()]))?;
        if !i.to_boolean(&res) {
            return Ok(false);
        }
        // Invariant: a non-extensible target's prototype can't be changed.
        if !js_is_extensible(i, &target)? {
            let cur = js_get_prototype_of(i, &target)?;
            if !i.strict_equals(proto, &cur) {
                return Err(i.make_error(
                    "TypeError",
                    "proxy 'setPrototypeOf' changed a non-extensible target's prototype",
                ));
            }
        }
        return Ok(true);
    }
    // Ordinary [[SetPrototypeOf]].
    let cur = o
        .borrow()
        .proto
        .clone()
        .map(Value::Obj)
        .unwrap_or(Value::Null);
    if i.strict_equals(proto, &cur) {
        return Ok(true);
    }
    // %Object.prototype% is an immutable-prototype exotic object: any actual change fails.
    if Rc::ptr_eq(&o, &i.object_proto) {
        return Ok(false);
    }
    if !o.borrow().extensible {
        return Ok(false);
    }
    // Cycle check: refuse if `o` already lies on the candidate prototype's chain (walking stops at
    // any proxy, whose [[GetPrototypeOf]] may be non-deterministic).
    let mut p = proto.clone();
    while let Value::Obj(po) = &p {
        if Rc::ptr_eq(po, &o) {
            return Ok(false);
        }
        if proxy_pair(i, &p).is_some() {
            break;
        }
        let next = po.borrow().proto.clone();
        p = match next {
            Some(n) => Value::Obj(n),
            None => break,
        };
    }
    // Any successful prototype swap invalidates every property-creation inline cache: a chain
    // that was proven clean at fill time may now route through different objects.
    crate::value::bump_proto_epoch();
    o.borrow_mut().proto = match proto {
        Value::Obj(p) => Some(p.clone()),
        _ => None,
    };
    Ok(true)
}

fn ta_byteoffset_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let (info, _) = ta_receiver(i, &this)?;
    Ok(Value::Num(if i.ta_len(&info).is_none() {
        0.0
    } else {
        info.offset as f64
    }))
}

/// A proxy's own enumerable string keys (for Object.keys/values/entries): the ownKeys trap result
/// filtered by each key's [[GetOwnProperty]] enumerable flag.
pub(crate) fn proxy_enum_string_keys(i: &mut Interp, proxy: &Value) -> Result<Vec<Value>, Value> {
    let (target, handler) = proxy_pair(i, proxy).unwrap();
    let keys = proxy_own_keys(i, &target, &handler)?;
    let mut out = Vec::new();
    for k in keys {
        if let Value::Str(ks) = &k {
            if proxy_key_enumerable(i, &target, &handler, ks)? {
                out.push(k);
            }
        }
    }
    Ok(out)
}

/// `[[PreventExtensions]]`, proxy-aware. Returns whether it succeeded.
fn js_prevent_extensions(i: &mut Interp, obj: &Value) -> Result<bool, Value> {
    if let Some((target, handler)) = proxy_pair(i, obj) {
        if matches!(handler, Value::Null) {
            return Err(i.make_error("TypeError", "proxy is revoked"));
        }
        let trap = ab(i.get_member(&handler, "preventExtensions"))?;
        if matches!(trap, Value::Undefined | Value::Null) {
            return js_prevent_extensions(i, &target);
        }
        if !trap.is_callable() {
            return Err(i.make_error(
                "TypeError",
                "proxy 'preventExtensions' trap is not callable",
            ));
        }
        let res = ab(i.call(trap, handler, std::slice::from_ref(&target)))?;
        let ok = i.to_boolean(&res);
        // Invariant: a true result requires the target to actually be non-extensible.
        if ok && js_is_extensible(i, &target)? {
            return Err(i.make_error(
                "TypeError",
                "proxy 'preventExtensions' reported success but the target is extensible",
            ));
        }
        return Ok(ok);
    }
    if let Value::Obj(o) = obj {
        // Web IDL §3.9.5: legacy platform objects must remain extensible.
        if i.host_indexed_len(o).is_some() {
            return Ok(false);
        }
        // TypedArray [[PreventExtensions]] refuses unless IsTypedArrayFixedLength: length-tracking
        // views, and any view over a resizable (non-shared) buffer, can change length.
        if let Some(info) = ta_info(i, o) {
            let resizable = Rc::as_ptr(o) as usize;
            let resizable = i
                .ta_buffer
                .get(&resizable)
                .and_then(|b| b.as_obj())
                .is_some_and(|b| {
                    matches!(
                        b.borrow().props.get("#\u{0}abResizable").map(|p| p.value()),
                        Some(Value::Bool(true))
                    )
                });
            let shared = i.shared_buffers.contains_key(&info.buffer);
            if info.track || (resizable && !shared) {
                return Ok(false);
            }
        }
        o.borrow_mut().extensible = false;
    }
    Ok(true)
}

fn proxy_get_prototype(i: &mut Interp, target: &Value, handler: &Value) -> Result<Value, Value> {
    let trap = ab(i.get_member(handler, "getPrototypeOf"))?;
    if matches!(trap, Value::Undefined | Value::Null) {
        // Forward to the target's [[GetPrototypeOf]] (recursing for a proxy target).
        return js_get_prototype_of(i, target);
    }
    if !trap.is_callable() {
        return Err(i.make_error("TypeError", "proxy 'getPrototypeOf' trap is not callable"));
    }
    if trap.is_callable() {
        let res = ab(i.call(trap, handler.clone(), std::slice::from_ref(target)))?;
        if !matches!(res, Value::Obj(_) | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "getPrototypeOf trap must return an object or null",
            ));
        }
        // Step 9: IsExtensible(target) runs unconditionally (its trap is observable and may
        // throw); a non-extensible target pins the reported prototype to the real one (via the
        // target's own [[GetPrototypeOf]], which may itself be a trap).
        let extensible = js_is_extensible(i, target)?;
        if !extensible {
            let actual = js_get_prototype_of(i, target)?;
            if !i.strict_equals(&res, &actual) {
                return Err(i.make_error(
                    "TypeError",
                    "getPrototypeOf trap result differs from a non-extensible target's prototype",
                ));
            }
        }
        Ok(res)
    } else if let Value::Obj(t) = target {
        Ok(t.borrow()
            .proto
            .clone()
            .map(Value::Obj)
            .unwrap_or(Value::Null))
    } else {
        Ok(Value::Null)
    }
}

fn ta_length_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let (info, _) = ta_receiver(i, &this)?;
    Ok(Value::Num(i.ta_len(&info).unwrap_or(0) as f64))
}

fn ta_buffer_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let _ = ta_receiver(i, &this)?;
    let ptr = map_ptr(&this).unwrap();
    Ok(i.ta_buffer.get(&ptr).cloned().unwrap_or(Value::Undefined))
}

/// If `v` is a Proxy, its (target, handler) pair.
pub(crate) fn proxy_pair(i: &Interp, v: &Value) -> Option<(Value, Value)> {
    if let Value::Obj(o) = v {
        i.proxies.get(&(Rc::as_ptr(o) as usize)).cloned()
    } else {
        None
    }
}

fn ta_bytelength_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let (info, _) = ta_receiver(i, &this)?;
    Ok(Value::Num(
        (i.ta_len(&info).unwrap_or(0) * info.kind.elsize()) as f64,
    ))
}

const TA_META_KEYS: [&str; 5] = [
    "length",
    "byteLength",
    "byteOffset",
    "buffer",
    "BYTES_PER_ELEMENT",
];

/// `[[GetPrototypeOf]]`, proxy-aware.
pub(crate) fn js_get_prototype_of(i: &mut Interp, obj: &Value) -> Result<Value, Value> {
    if let Some((target, handler)) = proxy_pair(i, obj) {
        if matches!(handler, Value::Null) {
            return Err(i.make_error("TypeError", "proxy is revoked"));
        }
        return proxy_get_prototype(i, &target, &handler);
    }
    Ok(match obj {
        Value::Obj(o) => o
            .borrow()
            .proto
            .clone()
            .map(Value::Obj)
            .unwrap_or(Value::Null),
        _ => Value::Null,
    })
}

/// HasOwnProperty for an already-coerced object and property key. Share the actual
/// [[GetOwnProperty]] dispatch between the Object builtins and native intrinsic: proxies,
/// integer-indexed objects, live namespaces, and host indices cannot use a raw map lookup.
pub(crate) fn has_own_property_trapped(
    i: &mut Interp,
    v: &Value,
    key: &str,
) -> Result<bool, Value> {
    if Interp::is_private_key(key) {
        return Ok(false);
    }
    let Value::Obj(o) = v else { return Ok(false) };
    {
        let object = o.borrow();
        if object.ic_plain.get() {
            return Ok(object.props.contains(key));
        }
    }
    ab(i.defer_trigger(o, Some(key)))?;
    if let Some(info) = ta_info(i, o) {
        match i.ta_index_kind(&info, key) {
            TaIndex::Element(_) => return Ok(true),
            TaIndex::Exotic => return Ok(false),
            TaIndex::Ordinary => {}
        }
    }
    if let Some((t, h)) = proxy_pair(i, v) {
        return Ok(!matches!(
            proxy_gopd_value(i, &t, &h, key)?,
            Value::Undefined
        ));
    }
    if ab(i.host_indexed_own_value(o, key))?.is_some() {
        return Ok(true);
    }
    let ptr = Rc::as_ptr(o) as usize;
    if i.is_namespace(ptr) {
        if let Some(result) = i.namespace_own_property(ptr, key) {
            ab(result)?;
            return Ok(true);
        }
    }
    Ok(o.borrow().props.contains(key))
}

/// ECMA-262 §6.2.6 / §10.1.5 [[GetOwnProperty]](P) for any object: the complete descriptor of
/// its own property `key`, or None. Every exotic kind applies its own internal method: Proxy
/// objects run §10.5.5 (trap and validation), typed arrays and namespaces their exotic rules,
/// mapped arguments report the live parameter value, and Web IDL indexed objects their
/// platform getter. Ordinary objects (set `ic_plain`, no exotic tag) read their own storage.
pub(crate) fn object_get_own_property(
    i: &mut Interp,
    o: &Gc,
    key: &str,
) -> Result<Option<Property>, Value> {
    {
        let body = o.borrow();
        if body.ic_plain.get()
            && matches!(body.exotic, Exotic::None | Exotic::Array)
            && !Interp::is_private_key(key)
        {
            return Ok(body.props.get(key).cloned());
        }
    }
    ab(i.defer_trigger(o, Some(key)))?;
    if Interp::is_private_key(key) {
        return Ok(None); // a private-name slot is not a property
    }
    let ptr = Rc::as_ptr(o) as usize;
    // A mapped arguments index reports the live parameter value (§10.4.4.1).
    if let Some(value) = i.mapped_arg_value(ptr, key) {
        let mut property = o.borrow().props.get(key).cloned();
        if let Some(property) = &mut property {
            property.set_value(value);
        }
        return Ok(property);
    }
    // A TypedArray canonical numeric index is an own data property reading from the buffer; a
    // canonical-but-invalid index has no own property (§10.4.5.1). Non-canonical keys such as
    // "1.0" or "+1" are ordinary.
    if let Some(info) = ta_info(i, o) {
        if i.canonical_numeric_index(key).is_some() {
            return Ok(match i.ta_index_kind(&info, key) {
                TaIndex::Element(index) => {
                    let value = i.ta_read(&info, index);
                    Some(Property::data(value, true, true, true))
                }
                _ => None,
            });
        }
    }
    if let Some(value) = ab(i.host_indexed_own_value(o, key))? {
        return Ok(Some(Property::data(
            value,
            false,
            crate::value::canonical_index(key).is_some(),
            true,
        )));
    }
    if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
        return proxy_get_own_property(i, &target, &handler, key);
    }
    if i.is_namespace(ptr) {
        if let Some(result) = i.namespace_own_property(ptr, key) {
            return Ok(Some(ab(result)?));
        }
    }
    Ok(o.borrow().props.get(key).cloned())
}

/// [[GetOwnProperty]] of a Proxy's target: the target is always an object.
fn target_own_property(
    i: &mut Interp,
    target: &Value,
    key: &str,
) -> Result<Option<Property>, Value> {
    match target {
        Value::Obj(target) => object_get_own_property(i, target, key),
        _ => Ok(None),
    }
}

/// ECMA-262 §10.5.5 Proxy [[GetOwnProperty]](P), steps 1-17 in order.
fn proxy_get_own_property(
    i: &mut Interp,
    target: &Value,
    handler: &Value,
    key: &str,
) -> Result<Option<Property>, Value> {
    if matches!(handler, Value::Null) {
        return Err(i.make_error("TypeError", "proxy is revoked"));
    }
    let trap = ab(i.get_member(handler, "getOwnPropertyDescriptor"))?;
    if matches!(trap, Value::Undefined | Value::Null) {
        return target_own_property(i, target, key);
    }
    if !trap.is_callable() {
        return Err(i.make_error(
            "TypeError",
            "proxy 'getOwnPropertyDescriptor' trap is not callable",
        ));
    }
    let key_val = i
        .sym_from_key(key)
        .unwrap_or_else(|| Value::lstr(key.to_string()));
    let result = ab(i.call(trap, handler.clone(), &[target.clone(), key_val]))?;
    if !matches!(result, Value::Obj(_) | Value::Undefined) {
        return Err(i.make_error(
            "TypeError",
            "getOwnPropertyDescriptor trap must return an object or undefined",
        ));
    }
    let target_desc = target_own_property(i, target, key)?;
    if matches!(result, Value::Undefined) {
        let Some(target_desc) = target_desc else {
            return Ok(None);
        };
        if !target_desc.configurable() {
            return Err(i.make_error(
                "TypeError",
                "gOPD trap reported undefined for a non-configurable property",
            ));
        }
        if !js_is_extensible(i, target)? {
            return Err(i.make_error(
                "TypeError",
                "gOPD trap reported undefined for a non-extensible target's property",
            ));
        }
        return Ok(None);
    }
    let extensible_target = js_is_extensible(i, target)?;
    let result_desc = completed_partial(ab(build_partial(i, &result))?);
    if !is_compatible_property_descriptor(extensible_target, &result_desc, target_desc.as_ref()) {
        return Err(i.make_error(
            "TypeError",
            format!("proxy reported an incompatible descriptor for '{key}'"),
        ));
    }
    if result_desc.configurable == Some(false) {
        match &target_desc {
            None => {
                return Err(i.make_error(
                    "TypeError",
                    "gOPD trap reported a non-configurable descriptor the target lacks",
                ))
            }
            Some(target_desc) if target_desc.configurable() => {
                return Err(i.make_error(
                    "TypeError",
                    "gOPD trap reported a configurable target property as non-configurable",
                ))
            }
            Some(target_desc) => {
                if result_desc.writable == Some(false)
                    && !target_desc.accessor()
                    && target_desc.writable()
                {
                    return Err(i.make_error(
                        "TypeError",
                        "gOPD trap reported non-writable for a writable target property",
                    ));
                }
            }
        }
    }
    Ok(Some(complete_descriptor(result_desc)))
}

/// Proxy [[GetOwnProperty]] as a descriptor object (FromPropertyDescriptor).
pub(crate) fn proxy_gopd_value(
    i: &mut Interp,
    target: &Value,
    handler: &Value,
    key: &str,
) -> Result<Value, Value> {
    Ok(match proxy_get_own_property(i, target, handler, key)? {
        Some(property) => descriptor_from_prop(i, property),
        None => Value::Undefined,
    })
}

/// CompletePropertyDescriptor (§6.2.6.6) of a partial descriptor, keeping the field-presence
/// representation so it can feed IsCompatiblePropertyDescriptor.
fn completed_partial(mut desc: PartialDesc) -> PartialDesc {
    if desc.is_accessor() {
        desc.get.get_or_insert(Value::Undefined);
        desc.set.get_or_insert(Value::Undefined);
    } else {
        desc.value.get_or_insert(Value::Undefined);
        desc.writable.get_or_insert(false);
    }
    desc.enumerable.get_or_insert(false);
    desc.configurable.get_or_insert(false);
    desc
}

/// ECMA-262 §10.1.6.2 IsCompatiblePropertyDescriptor(Extensible, Desc, Current), i.e.
/// ValidateAndApplyPropertyDescriptor(undefined, "", Extensible, Desc, Current).
fn is_compatible_property_descriptor(
    extensible: bool,
    desc: &PartialDesc,
    current: Option<&Property>,
) -> bool {
    let Some(current) = current else {
        return extensible;
    };
    let has_fields = desc.is_accessor()
        || desc.is_data()
        || desc.enumerable.is_some()
        || desc.configurable.is_some();
    if !has_fields {
        return true;
    }
    if !current.configurable() {
        if desc.configurable == Some(true) {
            return false;
        }
        if desc
            .enumerable
            .is_some_and(|enumerable| enumerable != current.enumerable())
        {
            return false;
        }
        let generic = !desc.is_accessor() && !desc.is_data();
        if !generic && desc.is_accessor() != current.accessor() {
            return false;
        }
        if current.accessor() {
            let same = |field: &Option<Value>, attribute: Option<&Value>| match field {
                None => true,
                Some(value) => same_value(value, attribute.unwrap_or(&Value::Undefined)),
            };
            if !same(&desc.get, current.getter()) || !same(&desc.set, current.setter()) {
                return false;
            }
        } else if !current.writable() {
            if desc.writable == Some(true) {
                return false;
            }
            if let Some(value) = &desc.value {
                return same_value(value, &current.value());
            }
        }
    }
    true
}

fn proxy_key_enumerable(
    i: &mut Interp,
    target: &Value,
    handler: &Value,
    key: &str,
) -> Result<bool, Value> {
    let trap = ab(i.get_member(handler, "getOwnPropertyDescriptor"))?;
    if trap.is_callable() {
        let kv = i
            .sym_from_key(key)
            .unwrap_or_else(|| Value::lstr(key.to_string()));
        let res = ab(i.call(trap, handler.clone(), &[target.clone(), kv]))?;
        if matches!(res, Value::Undefined) {
            return Ok(false);
        }
        let enum_v = ab(i.get_member(&res, "enumerable"))?;
        Ok(i.to_boolean(&enum_v))
    } else if let Some((t2, h2)) = proxy_pair(i, target) {
        // Absent gOPD trap over a proxy target: forward to the target's [[GetOwnProperty]].
        proxy_key_enumerable(i, &t2, &h2, key)
    } else if let Value::Obj(t) = target {
        if ab(i.host_indexed_own_value(t, key))?.is_some() {
            return Ok(crate::value::canonical_index(key).is_some());
        }
        Ok(t.borrow()
            .props
            .get(key)
            .map(|p| p.enumerable())
            .unwrap_or(false))
    } else {
        Ok(false)
    }
}

fn proxy_define_property(
    i: &mut Interp,
    target: &Value,
    handler: &Value,
    key: &str,
    desc: &Value,
) -> Result<bool, Abrupt> {
    let trap = i.get_member(handler, "defineProperty")?;
    if matches!(trap, Value::Undefined | Value::Null) {
        if let Some((t2, h2)) = proxy_pair(i, target) {
            return proxy_define_property(i, &t2, &h2, key, desc);
        }
        return if let Value::Obj(t) = target {
            define_own_property(i, t, key, desc)
        } else {
            Ok(false)
        };
    }
    if !trap.is_callable() {
        return Err(i.throw("TypeError", "proxy 'defineProperty' trap is not callable"));
    }
    let key_val = i
        .sym_from_key(key)
        .unwrap_or_else(|| Value::lstr(key.to_string()));
    // The trap receives FromPropertyDescriptor(desc): a fresh object holding only the present
    // fields, in the spec's order (value, writable, get, set, enumerable, configurable).
    let pd = build_partial(i, desc)?;
    let norm = i.new_object();
    if let Some(v) = &pd.value {
        set_data(&norm, "value", v.clone());
    }
    if let Some(w) = pd.writable {
        set_data(&norm, "writable", Value::Bool(w));
    }
    if let Some(g) = &pd.get {
        set_data(&norm, "get", g.clone());
    }
    if let Some(st) = &pd.set {
        set_data(&norm, "set", st.clone());
    }
    if let Some(e) = pd.enumerable {
        set_data(&norm, "enumerable", Value::Bool(e));
    }
    if let Some(c) = pd.configurable {
        set_data(&norm, "configurable", Value::Bool(c));
    }
    let res = i.call(
        trap,
        handler.clone(),
        &[target.clone(), key_val, Value::Obj(norm)],
    )?;
    if !i.to_boolean(&res) {
        return Ok(false);
    }
    // ECMA-262 §10.5.6 steps 11-16: validate against target.[[GetOwnProperty]](P) and
    // IsExtensible(target), both of which may run a Proxy target's traps.
    let target_desc = match target {
        Value::Obj(t) => object_get_own_property(i, t, key).map_err(Abrupt::Throw)?,
        _ => None,
    };
    let extensible_target = js_is_extensible(i, target).map_err(Abrupt::Throw)?;
    let setting_config_false = matches!(pd.configurable, Some(false));
    match target_desc {
        None => {
            if !extensible_target {
                return Err(i.throw(
                    "TypeError",
                    "proxy 'defineProperty' added a property to a non-extensible target",
                ));
            }
            if setting_config_false {
                return Err(i.throw(
                    "TypeError",
                    "proxy 'defineProperty' defined a non-configurable property the target lacks",
                ));
            }
        }
        Some(p) => {
            if !is_compatible_property_descriptor(extensible_target, &pd, Some(&p)) {
                return Err(i.throw(
                    "TypeError",
                    format!(
                        "proxy 'defineProperty' reported an incompatible descriptor for '{key}'"
                    ),
                ));
            }
            if setting_config_false && p.configurable() {
                return Err(i.throw(
                    "TypeError",
                    "proxy 'defineProperty' made a configurable target property non-configurable",
                ));
            }
            if !p.accessor()
                && !p.configurable()
                && p.writable()
                && matches!(pd.writable, Some(false))
            {
                return Err(i.throw(
                    "TypeError",
                    "proxy 'defineProperty' reported a non-configurable writable property as non-writable",
                ));
            }
        }
    }
    Ok(true)
}

/// `(info, detached)` for a TypedArray receiver, or a TypeError (brand check for the meta getters).
fn ta_receiver(i: &mut Interp, this: &Value) -> Result<(crate::value::TaInfo, bool), Value> {
    let ptr = map_ptr(this).filter(|p| i.typed_arrays.contains_key(p));
    match ptr {
        Some(p) => {
            let info = i.typed_arrays[&p];
            Ok((info, !i.array_buffers.contains_key(&info.buffer)))
        }
        None => Err(i.make_error(
            "TypeError",
            "TypedArray.prototype accessor called on a non-TypedArray",
        )),
    }
}

/// RequireObjectCoercible for an `Array.prototype` method receiver (null/undefined → TypeError).
fn arr_require_coercible(i: &mut Interp, this: &Value) -> Result<(), Value> {
    if matches!(this, Value::Undefined | Value::Null) {
        return Err(i.make_error(
            "TypeError",
            "Array.prototype method called on null or undefined",
        ));
    }
    Ok(())
}

/// TypedArray info for `o`, if it is one.
fn ta_info(i: &Interp, o: &Gc) -> Option<crate::value::TaInfo> {
    i.typed_arrays.get(&(Rc::as_ptr(o) as usize)).copied()
}

/// ToObject for an `Object.*` argument: an object is returned as-is, a primitive is boxed to its
/// wrapper object, and null/undefined throw a TypeError.
fn to_object_arg(i: &mut Interp, v: Value, method: &str) -> Result<Gc, Value> {
    match v {
        Value::Obj(o) => Ok(o),
        Value::Undefined | Value::Null => {
            Err(i.make_error("TypeError", format!("{method} called on null or undefined")))
        }
        other => match box_primitive(i, other) {
            Value::Obj(o) => Ok(o),
            _ => Err(i.make_error("TypeError", "ToObject failed")),
        },
    }
}

/// ToObject for a built-in method's `this` value; the TypeError names `method`.
pub(crate) fn to_object_for_method(i: &mut Interp, v: Value, method: &str) -> Result<Gc, Value> {
    to_object_arg(i, v, method)
}

/// A Promise combinator (`Promise.all`/`race`/…) requires `this` to be a constructor.
/// ToObject for a generic `Array.prototype` method receiver: primitives are boxed (so the method
/// reads the inherited array-like properties), null/undefined throw.
fn arr_to_object(i: &mut Interp, this: &Value) -> Result<Gc, Value> {
    to_object_arg(i, this.clone(), "Array.prototype method")
}

/// Set(to, key, value) with Throw=true, for Object.assign: a non-writable data property or a
/// setter-less accessor (own or inherited along the chain) makes the assignment throw a TypeError.
fn assign_set(i: &mut Interp, to: &Value, key: &str, value: Value) -> Result<(), Value> {
    // Set(to, key, value, true): a false [[Set]] throws even in sloppy mode (read-only
    // properties, and property creation on a non-extensible/sealed target).
    set_throw(i, to, key, value)
}

/// IsArray, seeing through proxies (a Proxy whose target is an Array is itself an Array).
fn json_is_array(i: &mut Interp, v: &Value) -> Result<bool, Value> {
    if let Some((target, handler)) = proxy_pair(i, v) {
        if matches!(handler, Value::Null) {
            return Err(i.make_error("TypeError", "Cannot perform IsArray on a revoked Proxy"));
        }
        return json_is_array(i, &target);
    }
    Ok(matches!(v, Value::Obj(o) if matches!(o.borrow().exotic, Exotic::Array)))
}

/// EnumerableOwnProperties for Object.values/entries: for each own string key (in order), do
/// [[GetOwnProperty]] then — if enumerable — [[Get]] the value, interleaved per key (matters for the
/// observable trap order on proxies). `entries` controls value vs `[key, value]` output.
fn enumerable_own_value_list(i: &mut Interp, o: &Value, entries: bool) -> Result<Value, Value> {
    // ECMA-262 EnumerableOwnProperties: a fully validated ordinary named data
    // record has no observable key/descriptor/Get effects. Its insertion order
    // is the required key order. A failed proof runs the ordered algorithm below.
    if let Some(object) = o
        .as_obj()
        .filter(|object| i.ordinary_get_ptr(Rc::as_ptr(object) as usize))
    {
        let object = object.borrow();
        if object.ic_plain.get()
            && matches!(object.exotic, Exotic::None)
            && matches!(object.call, Callable::None)
            && object.props.named_data_keys().is_some()
        {
            #[cfg(test)]
            TEST_RECORD_ENUMERATIONS.with(|count| count.set(count.get() + 1));
            let mut out = Vec::with_capacity(object.props.iter().size_hint().0);
            for (key, property) in object.props.iter() {
                out.push(if entries {
                    let key = i.property_key_string(key);
                    i.make_entry_pair(key, property.clone_value_packed())
                } else {
                    property.value()
                });
            }
            return Ok(i.make_array(out));
        }
    }
    let mut out = Vec::new();
    if let Some((t, h)) = proxy_pair(i, o) {
        for k in proxy_own_keys(i, &t, &h)? {
            if let Value::Str(ks) = &k {
                if proxy_key_enumerable(i, &t, &h, ks)? {
                    let v = ab(i.get_member(o, ks))?;
                    out.push(if entries {
                        i.make_array(vec![Value::Str(ks.clone()), v])
                    } else {
                        v
                    });
                }
            }
        }
    } else if let Some((object, length)) = o.as_obj().and_then(|object| {
        i.host_indexed_len(object)
            .map(|length| (object.clone(), length))
    }) {
        debug_assert_eq!(i.host_indexed_len(&object), Some(length));
        let keys = ordinary_own_keys_ordered(i, &object)?;
        for key in keys
            .into_iter()
            .filter(|key| !Interp::is_sym_key(key) && !Interp::is_private_key(key))
        {
            // EnumerableOwnProperties snapshots [[OwnPropertyKeys]], then observes each current
            // descriptor and performs a separate Get. An earlier indexed getter can therefore
            // delete or change a later expando, but cannot add it to the snapshot.
            let enumerable = if ab(i.host_indexed_own_value(&object, &key))?.is_some() {
                crate::value::canonical_index(&key).is_some()
            } else {
                object
                    .borrow()
                    .props
                    .get(&key)
                    .is_some_and(|property| property.enumerable())
            };
            if !enumerable {
                continue;
            }
            let value = ab(i.get_member(o, &key))?;
            out.push(if entries {
                i.make_array(vec![Value::lstr(key), value])
            } else {
                value
            });
        }
    } else if let Some(info) = o.as_obj().and_then(|obj| ta_info(i, obj)) {
        // A TypedArray's enumerable own properties are its integer indices (plus any expandos).
        let n = i.ta_len(&info).unwrap_or(0);
        for idx in 0..n {
            let v = i.ta_read(&info, idx);
            out.push(if entries {
                i.make_array(vec![Value::lstr(idx.to_string()), v])
            } else {
                v
            });
        }
        let expandos: Vec<Rc<str>> = o
            .as_obj()
            .map(|obj| {
                obj.borrow()
                    .props
                    .ordered_keys()
                    .into_iter()
                    .filter(|k| {
                        !Interp::is_sym_key(k)
                            && k.parse::<usize>().is_err()
                            && !TA_META_KEYS.contains(&&**k)
                    })
                    .collect()
            })
            .unwrap_or_default();
        for k in expandos {
            let enumerable = o
                .as_obj()
                .and_then(|obj| obj.borrow().props.get(&k).map(|p| p.enumerable()));
            if enumerable == Some(true) {
                let v = ab(i.get_member(o, &k))?;
                out.push(if entries {
                    i.make_array(vec![Value::Str(k.clone().into()), v])
                } else {
                    v
                });
            }
        }
    } else {
        // EnumerableOwnProperties: snapshot the key list, but re-read each property's descriptor
        // just before its Get — a getter may delete or hide keys not yet visited.
        let keys: Vec<Rc<str>> = o
            .as_obj()
            .map(|obj| {
                obj.borrow()
                    .props
                    .ordered_keys()
                    .into_iter()
                    .filter(|k| !Interp::is_sym_key(k))
                    .collect()
            })
            .unwrap_or_default();
        for k in keys {
            let live = o
                .as_obj()
                .and_then(|obj| obj.borrow().props.get(&k).map(|p| p.enumerable()));
            if live != Some(true) {
                continue;
            }
            let v = ab(i.get_member(o, &k))?;
            out.push(if entries {
                i.make_array(vec![Value::Str(k.into()), v])
            } else {
                v
            });
        }
    }
    Ok(i.make_array(out))
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_RECORD_ENUMERATIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

fn global_fn(it: &Interp, name: &str, len: usize, f: NativeFn) {
    let func = it.make_native(name, len, f);
    set_builtin(&it.global, name, Value::Obj(func));
}

// ---------------------------------------------------------------------------------------------
// Function.prototype
// ---------------------------------------------------------------------------------------------

/// OrdinarySet(target, key, value, receiver): walks target's prototype chain to find the controlling
/// descriptor, then applies the result on `receiver` (which may differ from `target`). Returns the
/// success boolean. Proxies on the chain fall back to the receiver-less [[Set]].
fn reflect_ordinary_set(
    i: &mut Interp,
    target: &Value,
    key: &str,
    value: Value,
    receiver: &Value,
) -> Result<bool, Value> {
    // Integer-indexed exotic [[Set]]: a canonical numeric index on a TypedArray is handled by
    // TypedArraySetElement (which coerces the value unconditionally, writing only when the index is
    // in range) and never falls back to creating an ordinary shadowing property.
    // A module namespace exotic object's [[Set]] always returns false.
    if let Value::Obj(o) = target {
        if i.is_namespace(Rc::as_ptr(o) as usize) {
            return Ok(false);
        }
    }
    let mut current = target.clone();
    loop {
        if proxy_pair(i, &current).is_some() {
            // A proxy's [[Set]] returns its own success boolean (via the trap or a forwarded [[Set]]).
            return ab(i.set_member_recv(&current, key, value, receiver.clone()));
        }
        // Integer-indexed exotic [[Set]] — applies to the target or any TypedArray reached along
        // the prototype chain. With the TypedArray itself as receiver, TypedArraySetElement
        // coerces unconditionally and writes only in-range; with a foreign receiver, a valid
        // index acts as a writable data property (created on the receiver) and a canonical
        // non-index is an inert success.
        if let Some(info) = map_ptr(&current).and_then(|p| i.typed_arrays.get(&p).copied()) {
            if i.canonical_numeric_index(key).is_some() {
                let same = matches!((&current, receiver), (Value::Obj(a), Value::Obj(b)) if Rc::ptr_eq(a, b));
                if same {
                    if info.kind.is_bigint() {
                        let n = ab(i.to_bigint(&value))?;
                        if let TaIndex::Element(idx) = i.ta_index_kind(&info, key) {
                            i.ta_write_bigint(&info, idx, n.to_i128_wrapping());
                        }
                    } else {
                        let n = ab(i.to_number(&value))?;
                        if let TaIndex::Element(idx) = i.ta_index_kind(&info, key) {
                            i.ta_write(&info, idx, n);
                        }
                    }
                    return Ok(true);
                }
                match i.ta_index_kind(&info, key) {
                    TaIndex::Exotic => return Ok(true),
                    TaIndex::Element(_) => {
                        return reflect_define_on_receiver(i, receiver, key, value);
                    }
                    TaIndex::Ordinary => {}
                }
            }
        }
        let obj = match &current {
            Value::Obj(o) => o.clone(),
            _ => return Ok(false),
        };
        if i.host_indexed_array_key(&obj, key) && ab(i.host_indexed_own_value(&obj, key))?.is_some()
        {
            return Ok(false);
        }
        let own = obj.borrow().props.get(key).cloned();
        match own {
            Some(p) if p.accessor() => {
                return match p.setter().cloned() {
                    Some(setter) if setter.is_callable() => {
                        ab(i.call_callback(setter, receiver.clone(), &[value]))?;
                        Ok(true)
                    }
                    _ => Ok(false),
                };
            }
            Some(p) => {
                if !p.writable() {
                    return Ok(false);
                }
                return reflect_define_on_receiver(i, receiver, key, value);
            }
            None => {
                let proto = obj.borrow().proto.clone();
                match proto {
                    Some(pp) => current = Value::Obj(pp),
                    None => return reflect_define_on_receiver(i, receiver, key, value),
                }
            }
        }
    }
}

/// Apply a writable data assignment to `receiver`: update an existing writable data property, or
/// CreateDataProperty when absent (respecting extensibility); accessor/non-writable existing → false.
pub(crate) fn reflect_define_on_receiver(
    i: &mut Interp,
    receiver: &Value,
    key: &str,
    value: Value,
) -> Result<bool, Value> {
    if proxy_pair(i, receiver).is_some() {
        let (target, handler) = proxy_pair(i, receiver).unwrap();
        // OrdinarySetWithOwnDescriptor: the receiver's [[GetOwnProperty]] runs first (its trap is
        // observable); an existing accessor or non-writable property rejects, an existing data
        // property is redefined with just {value}, and an absent one gets CreateDataProperty.
        let existing = proxy_gopd_value(i, &target, &handler, key)?;
        let desc = i.new_object();
        if let Value::Obj(d) = &existing {
            let (is_accessor, writable) = {
                let db = d.borrow();
                (
                    db.props.contains("get") || db.props.contains("set"),
                    matches!(
                        db.props.get("writable").map(|p| p.value()),
                        Some(Value::Bool(true))
                    ),
                )
            };
            if is_accessor || !writable {
                return Ok(false);
            }
            set_data(&desc, "value", value);
        } else {
            set_data(&desc, "value", value);
            set_data(&desc, "writable", Value::Bool(true));
            set_data(&desc, "enumerable", Value::Bool(true));
            set_data(&desc, "configurable", Value::Bool(true));
        }
        return ab(proxy_define_property(
            i,
            &target,
            &handler,
            key,
            &Value::Obj(desc),
        ));
    }
    let ro = match receiver {
        Value::Obj(o) => o.clone(),
        _ => return Ok(false),
    };
    // A TypedArray receiver's integer index is written through its exotic [[DefineOwnProperty]]
    // (CreateDataProperty semantics), not stored as an ordinary property.
    if let Some(info) = ta_info(i, &ro) {
        if i.canonical_numeric_index(key).is_some() {
            return match i.ta_index_kind(&info, key) {
                TaIndex::Element(idx) => {
                    ab(i.ta_store(&info, idx, &value))?;
                    Ok(true)
                }
                _ => Ok(false),
            };
        }
    }
    if i.host_indexed_array_key(&ro, key) || i.host_named_reject_define(&ro, key) {
        return Ok(false);
    }
    let existing = ro.borrow().props.get(key).cloned();
    // OrdinarySetWithOwnDescriptor calls the receiver's [[DefineOwnProperty]], not [[Set]].
    // Re-running [[Set]] here would invoke its own/inherited setter instead of defining a
    // data property, and would repeat the prototype search on the wrong object.
    if matches!(ro.borrow().exotic, Exotic::Array) {
        let mut descriptor = PartialDesc {
            value: Some(value),
            ..PartialDesc::default()
        };
        match existing {
            Some(property) if property.accessor() || !property.writable() => return Ok(false),
            Some(_) => {}
            None => {
                descriptor.writable = Some(true);
                descriptor.enumerable = Some(true);
                descriptor.configurable = Some(true);
            }
        }
        return ab(define_own_property_ordinary(i, &ro, key, &descriptor));
    }
    match existing {
        Some(ep) if ep.accessor() || !ep.writable() => Ok(false),
        Some(_) => {
            ro.borrow_mut().props.get_mut(key).unwrap().set_value(value);
            Ok(true)
        }
        None => {
            if !ro.borrow().extensible {
                return Ok(false);
            }
            ro.borrow_mut()
                .props
                .insert(key, Property::data(value, true, true, true));
            Ok(true)
        }
    }
}

/// OrdinaryGet(target, key, receiver): walk target's chain; a data property returns its value, an
/// accessor invokes its getter with `receiver` as `this`. Proxies on the chain fall back to the
/// receiver-less [[Get]].
pub(crate) fn reflect_ordinary_get(
    i: &mut Interp,
    target: &Value,
    key: &str,
    receiver: &Value,
) -> Result<Value, Value> {
    let mut current = target.clone();
    loop {
        if proxy_pair(i, &current).is_some() {
            // A proxy's [[Get]](P, Receiver) threads the explicit receiver.
            return ab(i.get_member_recv(&current, key, receiver.clone()));
        }
        let obj = match &current {
            Value::Obj(o) => o.clone(),
            _ => return Ok(Value::Undefined),
        };
        // A deferred namespace on the chain triggers its module's evaluation before the read.
        ab(i.defer_trigger(&obj, Some(key)))?;
        // A module namespace's [[Get]] reads the export's live value (throwing for a TDZ binding).
        let ptr = Rc::as_ptr(&obj) as usize;
        if i.is_namespace(ptr) {
            if let Some(res) = i.namespace_own_property(ptr, key) {
                return Ok(ab(res)?.into_value());
            }
        }
        // A TypedArray's integer-indexed elements are own properties; a canonical-numeric
        // non-index short-circuits to undefined without consulting the prototype chain.
        if let Some(info) = i.typed_arrays.get(&ptr).copied() {
            match i.ta_index_kind(&info, key) {
                TaIndex::Element(idx) => return Ok(i.ta_read(&info, idx)),
                TaIndex::Exotic => return Ok(Value::Undefined),
                TaIndex::Ordinary => {}
            }
        }
        if let Some(value) = ab(i.host_indexed_own_value(&obj, key))? {
            return Ok(value);
        }
        let own = obj.borrow().props.get(key).cloned();
        match own {
            Some(p) if p.accessor() => {
                return match p.getter().cloned() {
                    Some(g) if g.is_callable() => ab(i.call_callback(g, receiver.clone(), &[])),
                    _ => Ok(Value::Undefined),
                };
            }
            Some(p) => return Ok(p.into_value()),
            None => {
                let proto = obj.borrow().proto.clone();
                match proto {
                    Some(pp) => current = Value::Obj(pp),
                    None => return Ok(Value::Undefined),
                }
            }
        }
    }
}

/// DeletePropertyOrThrow(O, P): [[Delete]] and throw a TypeError if it returns false. An absent
/// property deletes successfully (returns true), matching the spec used by Array mutators.
fn delete_or_throw(i: &mut Interp, holder: &Value, key: &str) -> Result<(), Value> {
    if let Some((target, handler)) = proxy_pair(i, holder) {
        let ok = ab(i.proxy_delete(target, handler, key))?;
        if !ok {
            return Err(i.make_error("TypeError", format!("Cannot delete property '{key}'")));
        }
        return Ok(());
    }
    if let Value::Obj(o) = holder {
        if i.host_indexed_array_key(o, key) {
            let supported = crate::value::canonical_index(key)
                .is_some_and(|index| index < i.host_indexed_len(o).unwrap_or(0));
            if supported {
                return Err(i.make_error("TypeError", format!("Cannot delete property '{key}'")));
            }
            return Ok(());
        }
        if ab(i.host_named_visible(o, key))? {
            return Err(i.make_error("TypeError", format!("Cannot delete named property '{key}'")));
        }
        let present = o.borrow().props.contains(key);
        if !present {
            return Ok(());
        }
        let configurable = o
            .borrow()
            .props
            .get(key)
            .map(|p| p.configurable())
            .unwrap_or(true);
        if !configurable {
            return Err(i.make_error("TypeError", format!("Cannot delete property '{key}'")));
        }
        o.borrow_mut().props.remove(key);
    }
    Ok(())
}

fn set_internal_obj(target: &Value, key: &str, v: Value) {
    if let Value::Obj(o) = target {
        o.borrow_mut()
            .props
            .insert(key, Property::data(v, true, false, false));
    }
}

fn make_aggregate_error(i: &mut Interp, errors: Value) -> Result<Value, Value> {
    let ctor = ab(i.get_member(&Value::Obj(i.global.clone()), "AggregateError"))?;
    ab(i.construct(ctor, &[errors]))
}

/// `Promise.resolve(x)` as a value helper (returns existing promises unchanged).
fn promise_resolve_value(i: &mut Interp, v: Value) -> Value {
    if let Value::Obj(o) = &v {
        if i.promises.contains_key(&(Rc::as_ptr(o) as usize)) {
            return v;
        }
    }
    let p = i.new_promise();
    i.resolve_promise(&p, v);
    p
}

/// Sentinel call slot for a function-targeted proxy, so `is_callable()` is true; the actual
/// dispatch happens in `call_inner`/`construct_inner` via the proxies table (this never runs).
fn proxy_uncallable(i: &mut Interp, _t: Value, _a: &[Value]) -> Result<Value, Value> {
    Err(i.make_error("TypeError", "proxy call dispatch error"))
}

/// OrdinaryCreateFromConstructor: a new object whose [[Prototype]] is `new.target.prototype` when
/// that's an object (subclassing / Reflect.construct), else the named intrinsic prototype. Safe to
/// call from any native constructor: outside a `new`, `new_target` is Undefined so the default wins.
/// CanBeHeldWeakly: an object, or a non-registered (collectable) symbol.
fn can_be_held_weakly(i: &Interp, v: &Value) -> bool {
    match v {
        Value::Obj(_) => true,
        Value::Sym(s) => !i.symbol_is_registered(s),
        _ => false,
    }
}

/// Decode (sec-decodeuri-decode): percent-escapes become bytes, validated as strict UTF-8 per
/// sequence (a bad hex digit, truncated escape, stray continuation byte, overlong form, encoded
/// surrogate, or > U+10FFFF is a URIError -> None). An escape whose decoded octet is an ASCII
/// character in `preserve` (decodeURI's reservedSet) keeps its original `%XX` text instead.
fn uri_decode(s: &str, preserve: &str) -> Option<String> {
    fn hex_at(bytes: &[u8], i: usize) -> Option<u8> {
        if i + 2 >= bytes.len() || bytes[i] != b'%' {
            return None;
        }
        let h = (bytes[i + 1] as char).to_digit(16)?;
        let l = (bytes[i + 2] as char).to_digit(16)?;
        Some((h * 16 + l) as u8)
    }
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            let c = s[i..].chars().next()?;
            out.push(c);
            i += c.len_utf8();
            continue;
        }
        let start = i;
        let b0 = hex_at(bytes, i)?;
        i += 3;
        if b0 < 0x80 {
            let c = b0 as char;
            if preserve.contains(c) {
                out.push_str(&s[start..i]);
            } else {
                out.push(c);
            }
            continue;
        }
        let cont = match b0 {
            0xC2..=0xDF => 1,
            0xE0..=0xEF => 2,
            0xF0..=0xF4 => 3,
            _ => return None,
        };
        let mut buf = [b0, 0, 0, 0];
        for k in 0..cont {
            let bc = hex_at(bytes, i)?;
            if bc & 0xC0 != 0x80 {
                return None;
            }
            buf[k + 1] = bc;
            i += 3;
        }
        let decoded = std::str::from_utf8(&buf[..cont + 1]).ok()?;
        for c in decoded.chars() {
            let cp = c as u32;
            if cp >= crate::jstr::SMUGGLE_BASE {
                // A real code point in the lone-surrogate smuggle range must take the
                // engine's smuggled-pair representation (see jstr).
                out.push_str(&crate::jstr::from_code_points(&[cp]));
            } else {
                out.push(c);
            }
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------------------------
// ArrayBuffer + TypedArrays. Backing bytes live in `Interp::array_buffers`; each view's state in
// `Interp::typed_arrays`. Integer-index get/set is wired in `get_member`/`set_member`; the named
// metadata (length/byteLength/byteOffset/buffer/BYTES_PER_ELEMENT) is stored as real own props.
// ---------------------------------------------------------------------------------------------

/// SetIntegrityLevel(obj, frozen?): PreventExtensions then make every own property non-configurable
/// (and, when freezing, non-writable for data properties), all through trap-aware operations.
fn set_integrity_level(i: &mut Interp, obj: &Value, freeze: bool) -> Result<bool, Value> {
    if !matches!(obj, Value::Obj(_)) {
        return Ok(true);
    }
    if !js_prevent_extensions(i, obj)? {
        return Ok(false);
    }
    // TypedArray elements can never be made non-configurable (or non-writable), so sealing or
    // freezing any view that currently has elements fails at the first DefinePropertyOrThrow.
    if let Value::Obj(o) = obj {
        if let Some(info) = ta_info(i, o) {
            if i.ta_len(&info).unwrap_or(0) > 0 {
                return Ok(false);
            }
        }
    }
    let proxy = proxy_pair(i, obj);
    let keys: Vec<crate::value::PropertyKey> = if let Some((t, h)) = &proxy {
        let mut ks = Vec::new();
        for k in proxy_own_keys(i, t, h)? {
            ks.push(ab(i.to_property_key(&k))?);
        }
        ks
    } else {
        obj.as_obj()
            .unwrap()
            .borrow()
            .props
            .ordered_keys()
            .iter()
            .filter(|k| !Interp::is_private_key(k))
            .map(|k| crate::value::PropertyKey::string(k.to_string()))
            .collect()
    };
    for key in keys {
        let is_accessor = if let Some((t, h)) = &proxy {
            let d = proxy_gopd_value(i, t, h, &key)?;
            matches!(&d, Value::Obj(o) if o.borrow().props.contains("get") || o.borrow().props.contains("set"))
        } else {
            obj.as_obj()
                .unwrap()
                .borrow()
                .props
                .get(&key)
                .map(|p| p.accessor())
                .unwrap_or(false)
        };
        let desc = i.new_object();
        set_data(&desc, "configurable", Value::Bool(false));
        if freeze && !is_accessor {
            set_data(&desc, "writable", Value::Bool(false));
        }
        let ok = if let Some((t, h)) = &proxy {
            ab(proxy_define_property(i, t, h, &key, &Value::Obj(desc)))?
        } else {
            ab(define_own_property(
                i,
                obj.as_obj().unwrap(),
                &key,
                &Value::Obj(desc),
            ))?
        };
        if !ok {
            return Err(i.make_error("TypeError", "could not set the object's integrity level"));
        }
    }
    Ok(true)
}

pub(crate) fn new_from_ctor(i: &mut Interp, default_proto: &str) -> Result<Gc, Value> {
    let proto = match &i.new_target {
        nt @ Value::Obj(_) => match ab(i.get_member(&nt.clone(), "prototype"))? {
            Value::Obj(p) => Some(p),
            // The constructor's `prototype` isn't an object — use its realm's intrinsic
            // (GetFunctionRealm, which throws for a proxy revoked mid-construct — the
            // `prototype` getter may have revoked it).
            _ => ctor_realm_proto(i, &i.new_target.clone(), default_proto)?
                .or_else(|| i.extra_protos.get(default_proto).cloned()),
        },
        _ => i.extra_protos.get(default_proto).cloned(),
    };
    Ok(Object::new(proto))
}

/// Encode (sec-encodeuri-encode): per character; a lone (smuggled) surrogate is a URIError
/// (`None`).
fn uri_encode(s: &str, keep: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(mut c) = chars.next() {
        if crate::jstr::smuggled(c).is_some() {
            // A smuggled pair encodes a real smuggle-range character; a lone one is malformed.
            c = chars.peek().and_then(|&n| crate::jstr::paired_char(c, n))?;
            chars.next();
        }
        if c.is_ascii()
            && (c.is_ascii_alphanumeric() || "-_.!~*'()".contains(c) || keep.contains(c))
        {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for b in c.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    Some(out)
}

/// ObjectDefineProperties(O, Properties): ToObject(Properties), then for each *enumerable* own key
/// (string or symbol, proxy-aware) collect its descriptor object, then DefinePropertyOrThrow each on
/// O (which may itself be a proxy).
fn object_define_properties(
    i: &mut Interp,
    o: &Value,
    props_arg: Value,
    who: &str,
) -> Result<(), Value> {
    let props = Value::Obj(to_object_arg(i, props_arg, who)?);
    let keys: Vec<crate::value::PropertyKey> = if let Some((t, h)) = proxy_pair(i, &props) {
        let mut ks = Vec::new();
        for k in proxy_own_keys(i, &t, &h)? {
            ks.push(ab(i.to_property_key(&k))?);
        }
        ks
    } else {
        ordinary_own_keys_ordered(i, props.as_obj().unwrap())?
            .into_iter()
            .filter(|k| !Interp::is_private_key(k))
            .map(crate::value::PropertyKey::string)
            .collect()
    };
    // Collect descriptor objects for enumerable own keys first (so all Gets precede any Defines).
    let mut descs: Vec<(crate::value::PropertyKey, Value)> = Vec::new();
    for key in keys {
        let enumerable = if let Some((t, h)) = proxy_pair(i, &props) {
            let d = proxy_gopd_value(i, &t, &h, &key)?;
            match &d {
                Value::Obj(_) => {
                    let e = ab(i.get_member(&d, "enumerable"))?;
                    i.to_boolean(&e)
                }
                _ => false,
            }
        } else {
            let object = props.as_obj().unwrap();
            if ab(i.host_indexed_own_value(object, &key))?.is_some() {
                crate::value::canonical_index(&key).is_some()
            } else {
                object
                    .borrow()
                    .props
                    .get(&key)
                    .map(|p| p.enumerable())
                    .unwrap_or(false)
            }
        };
        if enumerable {
            let desc_obj = ab(i.get_member(&props, &key))?;
            descs.push((key, desc_obj));
        }
    }
    for (key, desc_obj) in descs {
        let ok = if let Some((t, h)) = proxy_pair(i, o) {
            ab(proxy_define_property(i, &t, &h, &key, &desc_obj))?
        } else {
            ab(define_own_property(i, o.as_obj().unwrap(), &key, &desc_obj))?
        };
        if !ok {
            return Err(i.make_error("TypeError", "cannot define property"));
        }
    }
    Ok(())
}

/// CreateListFromArrayLike: the object must be array-like (TypeError otherwise); reads `length`
/// (ToLength) then the indexed elements in order. Used by Reflect.apply/construct.
fn create_list_from_array_like(i: &mut Interp, v: &Value) -> Result<Vec<Value>, Value> {
    let o = match v {
        Value::Obj(o) => o.clone(),
        _ => {
            return Err(i.make_error(
                "TypeError",
                "CreateListFromArrayLike called on a non-object",
            ));
        }
    };
    let len = ab(i.to_length(&o))?;
    let mut out = Vec::with_capacity(len.min(1024));
    for k in 0..len {
        out.push(array_get_index(i, &o, v, k)?);
    }
    Ok(out)
}

/// GetFunctionRealm-style lookup: when `new.target`'s `prototype` isn't an object, the instance's
/// [[Prototype]] is `new.target`'s realm's intrinsic. The realm is identified by which registered
/// realm's `Function.prototype` lies on `new.target`'s own prototype chain.
/// GetFunctionRealm-backed intrinsic lookup for OrdinaryCreateFromConstructor's fallback: a
/// revoked proxy on the newTarget chain is a TypeError (the spec's GetFunctionRealm throws).
fn ctor_realm_proto(i: &mut Interp, nt: &Value, default_proto: &str) -> Result<Option<Gc>, Value> {
    let Some(o) = nt.as_obj() else {
        return Ok(None);
    };
    let mut ntobj: Gc = o.clone();
    // GetFunctionRealm unwraps bound functions and (unrevoked) proxies to their targets.
    for _ in 0..64 {
        if let Some((target, handler)) = i.proxies.get(&(Rc::as_ptr(&ntobj) as usize)) {
            if matches!(handler, Value::Null) {
                return Err(i.make_error("TypeError", "cannot get the realm of a revoked proxy"));
            }
            match target {
                Value::Obj(t) => {
                    ntobj = t.clone();
                    continue;
                }
                _ => return Ok(None),
            }
        }
        let bound = match &ntobj.borrow().call {
            Callable::Bound(bound) => Some(bound.target.clone()),
            _ => None,
        };
        match bound {
            Some(t) => ntobj = t,
            None => break,
        }
    }
    let Some(g) = i.callee_realm_global(&ntobj) else {
        return Ok(None);
    };
    let Some(rs) = i.realms.get(&g) else {
        return Ok(None);
    };
    // Core intrinsics live in named fields; errors in error_protos; the rest in extra_protos.
    Ok(match default_proto {
        "Object" => Some(rs.object_proto.clone()),
        "Array" => Some(rs.array_proto.clone()),
        "Function" => Some(rs.function_proto.clone()),
        "String" => Some(rs.string_proto.clone()),
        "Number" => Some(rs.number_proto.clone()),
        "Boolean" => Some(rs.boolean_proto.clone()),
        "Symbol" => Some(rs.symbol_proto.clone()),
        other => rs
            .error_protos
            .get(other)
            .cloned()
            .or_else(|| rs.extra_protos.get(other).cloned()),
    })
}

/// AdvanceStringIndex over UNIT offsets: one code unit, or a whole surrogate pair in unicode mode.
fn advance_string_index(index: usize, s: &str, unicode: bool) -> usize {
    if !unicode {
        return index + 1;
    }
    let units = crate::jstr::units(s);
    if index + 1 < units.len()
        && (0xD800..0xDC00).contains(&(units[index] as u32))
        && (0xDC00..0xE000).contains(&(units[index + 1] as u32))
    {
        index + 2
    } else {
        index + 1
    }
}

/// RegExpExec(R, S): call `R.exec` if it is callable (validating the result is an Object or null),
/// otherwise fall back to the built-in `RegExp.prototype.exec` (which requires a real RegExp).
fn regexp_exec_abstract(i: &mut Interp, r: &Value, s: crate::lstr::LStr) -> Result<Value, Value> {
    let exec = ab(i.get_member(r, "exec"))?;
    if exec.is_callable() {
        let res = ab(i.call(exec, r.clone(), &[Value::Str(s)]))?;
        if !matches!(res, Value::Obj(_) | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "RegExp exec method returned something other than an Object or null",
            ));
        }
        return Ok(res);
    }
    if map_ptr(r).map(|p| i.regexps.contains_key(&p)) != Some(true) {
        return Err(i.make_error("TypeError", "exec called on a non-RegExp object"));
    }
    regexp_exec(i, r.clone(), &[Value::Str(s)])
}

/// Coerce `v` to a RegExp object (returning it unchanged if already one).
fn regexp_match_error_value(i: &Interp, error: crate::regex::MatchError) -> Value {
    match error {
        crate::regex::MatchError::ResourceExhausted => i.make_error(
            "RangeError",
            "regular expression execution exceeded engine resource limits",
        ),
        // NativeFn's Value-shaped error channel cannot carry host control flow. Every native
        // dispatch polls again immediately after returning and replaces this placeholder with the
        // original non-catchable interruption completion.
        crate::regex::MatchError::Interrupted(reason) => i.make_error("Error", reason.message()),
    }
}

fn regexp_match_result<T>(
    i: &Interp,
    result: crate::regex::MatchResult<T>,
) -> Result<Option<T>, Value> {
    result.map_err(|error| regexp_match_error_value(i, error))
}

fn regexp_match_abrupt<T>(
    i: &Interp,
    result: crate::regex::MatchResult<T>,
) -> Result<Option<T>, Abrupt> {
    result.map_err(|error| match error {
        crate::regex::MatchError::ResourceExhausted => Abrupt::Throw(i.make_error(
            "RangeError",
            "regular expression execution exceeded engine resource limits",
        )),
        crate::regex::MatchError::Interrupted(reason) => Abrupt::Interrupt(reason),
    })
}

#[inline]
pub(crate) fn regexp_interrupt_tick(i: &mut Interp) -> Result<(), crate::regex::MatchError> {
    match i.interrupt_poll() {
        Ok(()) => Ok(()),
        Err(Abrupt::Interrupt(reason)) => Err(crate::regex::MatchError::Interrupted(reason)),
        Err(_) => unreachable!("interrupt polling only produces an Interrupt completion"),
    }
}

#[inline]
fn regexp_exec_text_shared(
    i: &mut Interp,
    re: &crate::regex::Regex,
    text: &crate::regex::ReText,
    start: usize,
) -> crate::regex::MatchResult<crate::regex::Captures> {
    re.exec_text_shared_entry_polled(text, start, &i.runtime_interrupt)
}

#[inline]
fn regexp_exec_text_discard_shared(
    i: &mut Interp,
    re: &crate::regex::Regex,
    text: &crate::regex::ReText,
    start: usize,
) -> crate::regex::MatchResult<crate::regex::Captures> {
    re.exec_text_discard_shared_entry_polled(text, start, &i.runtime_interrupt)
}

#[inline]
pub(crate) fn regexp_find_text_shared(
    i: &mut Interp,
    re: &crate::regex::Regex,
    text: &crate::regex::ReText,
    start: usize,
) -> crate::regex::MatchResult<(usize, usize)> {
    re.find_text_shared_entry_polled(text, start, &i.runtime_interrupt)
}

/// All non-overlapping matches of `re` in `text`, each as capture spans (element indices).
fn regex_find_all(
    i: &mut Interp,
    re: &crate::regex::Regex,
    text: &crate::regex::ReText,
) -> Result<Vec<Vec<Option<(usize, usize)>>>, Value> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos <= text.len() {
        let interrupt = regexp_interrupt_tick(i).map(|()| Some(()));
        regexp_match_result(i, interrupt)?;
        let matched = regexp_exec_text_shared(i, re, text, pos);
        match regexp_match_result(i, matched)? {
            None => break,
            Some(caps) => {
                let (a, b) = caps[0].unwrap();
                pos = if b > a { b } else { b + 1 };
                out.push(caps.as_ref().to_vec());
                if out.len() > MAX_ARRAY_OP_LEN {
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// ToLength of a value (clamped to a non-negative integer).
fn to_length_val(i: &mut Interp, v: &Value) -> Result<usize, Value> {
    let n = ab(i.to_number(v))?;
    Ok(if n.is_nan() || n < 0.0 {
        0usize
    } else {
        n.min(9_007_199_254_740_991.0) as usize
    })
}

/// `RegExp.prototype.exec`: returns the match array (with `index`/`input`) or `null`, advancing
/// `lastIndex` for global/sticky regexes.
pub(crate) fn regexp_exec(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    let ptr = map_ptr(&this).ok_or_else(|| i.make_error("TypeError", "exec on non-RegExp"))?;
    if !i.regexps.contains_key(&ptr) {
        return Err(i.make_error("TypeError", "exec on non-RegExp"));
    }
    let input = ab(i.to_string(&arg(args, 0)))?;
    // lastIndex is read (and length-coerced) unconditionally; it only *steers* the match when
    // the regexp is global or sticky. Its valueOf can recompile the regexp (`compile()`), so the
    // matcher slot is only sampled afterwards. Reading an own non-accessor `lastIndex` (its
    // invariable shape — it is non-configurable) skips the [[Get]] machinery.
    let li = {
        let own = match &this {
            Value::Obj(o) => {
                let b = o.borrow();
                b.props
                    .get("lastIndex")
                    .filter(|p| !p.accessor())
                    .map(|p| p.value())
            }
            _ => None,
        };
        match own {
            Some(v) => v,
            None => ab(i.get_member(&this, "lastIndex"))?,
        }
    };
    let read_units = to_length_val(i, &li)?;
    let re = i
        .regexps
        .get(&ptr)
        .cloned()
        .ok_or_else(|| i.make_error("TypeError", "exec on non-RegExp"))?;
    // Elements are code units (non-unicode) or code points (u/v); JS-visible indices are units.
    let text = i.re_text(re.unicode, &input);
    let use_last = re.global || re.sticky;
    let last_units = if use_last { read_units } else { 0 };
    if last_units > text.unit_index(text.len()) {
        if use_last {
            set_throw(i, &this, "lastIndex", Value::Num(0.0))?;
        }
        return Ok(Value::Null);
    }
    let last = text.elem_at_unit(last_units);
    let matched = regexp_exec_text_discard_shared(i, &re, &text, last);
    match regexp_match_result(i, matched)? {
        None => {
            if use_last {
                set_throw(i, &this, "lastIndex", Value::Num(0.0))?;
            }
            Ok(Value::Null)
        }
        Some(caps) => {
            let (start, end) = caps[0].unwrap();
            if use_last {
                set_throw(
                    i,
                    &this,
                    "lastIndex",
                    Value::Num(text.unit_index(end) as f64),
                )?;
            }
            update_regexp_legacy_statics(i, &re, &caps, &text, &input);
            let mut items = vec![Value::lstr(text.slice(start, end))];
            for g in 1..=re.ngroups {
                items.push(match caps[g] {
                    Some((a, b)) => Value::lstr(text.slice(a, b)),
                    None => Value::Undefined,
                });
            }
            let result = i.make_array(items);
            // `groups`: a null-prototype object of named captures, or undefined if there are none.
            let groups = if re.names.is_empty() {
                Value::Undefined
            } else {
                let g = i.new_object();
                g.borrow_mut().proto = None;
                // One property per name, in first-occurrence order; for duplicate names the value
                // comes from whichever same-named group actually matched (else undefined).
                let mut seen: Vec<&str> = Vec::new();
                for (name, _) in &re.names {
                    if seen.contains(&name.as_str()) {
                        continue;
                    }
                    seen.push(name.as_str());
                    let v = re
                        .names
                        .iter()
                        .filter(|(n, _)| n == name)
                        .find_map(|(_, idx)| caps.get(*idx).copied().flatten())
                        .map(|(a, b)| Value::lstr(text.slice(a, b)))
                        .unwrap_or(Value::Undefined);
                    set_data(&g, name, v);
                }
                Value::Obj(g)
            };
            // `indices` (the `d` flag): [start, end] pairs per capture (undefined if unmatched),
            // plus a null-prototype `groups` of the same for named captures.
            let indices = if re.flags.contains('d') {
                let pair = |i: &mut Interp, span: Option<(usize, usize)>| match span {
                    Some((a, b)) => i.make_array(vec![
                        Value::Num(text.unit_index(a) as f64),
                        Value::Num(text.unit_index(b) as f64),
                    ]),
                    None => Value::Undefined,
                };
                let mut idx_items = Vec::with_capacity(re.ngroups + 1);
                for g in 0..=re.ngroups {
                    let span = caps.get(g).copied().flatten();
                    idx_items.push(pair(i, span));
                }
                let arr = i.make_array(idx_items);
                let igroups = if re.names.is_empty() {
                    Value::Undefined
                } else {
                    let g = i.new_object();
                    g.borrow_mut().proto = None;
                    let mut seen: Vec<String> = Vec::new();
                    for (name, _) in &re.names {
                        if seen.iter().any(|s| s == name) {
                            continue;
                        }
                        seen.push(name.clone());
                        let span = re
                            .names
                            .iter()
                            .filter(|(n, _)| n == name)
                            .find_map(|(_, idx)| caps.get(*idx).copied().flatten());
                        let v = pair(i, span);
                        set_data(&g, name, v);
                    }
                    Value::Obj(g)
                };
                if let Value::Obj(o) = &arr {
                    set_data(o, "groups", igroups);
                }
                Some(arr)
            } else {
                None
            };
            if let Value::Obj(o) = &result {
                set_data(o, "index", Value::Num(text.unit_index(start) as f64));
                set_data(o, "input", Value::Str(input));
                set_data(o, "groups", groups);
                if let Some(ind) = indices {
                    set_data(o, "indices", ind);
                }
            }
            Ok(result)
        }
    }
}

/// Allocation-free `RegExp.prototype.exec` for a JIT call whose result is immediately discarded.
/// Returns `None` unless every potentially observable coercion/property operation is proven to be
/// the ordinary built-in shape, in which case all matcher side effects (lastIndex and the legacy
/// RegExp statics) are preserved.
pub(crate) fn regexp_exec_discard_fast(
    i: &mut Interp,
    this: &Value,
    input: &crate::lstr::LStr,
) -> Option<Result<bool, Value>> {
    let Value::Obj(obj) = this else {
        return None;
    };
    let ptr = Rc::as_ptr(obj) as usize;
    let re = i.regexps.get(&ptr)?.clone();
    {
        let b = obj.borrow();
        if !matches!(b.exotic, Exotic::None) {
            return None;
        }
        let p = b.props.get("lastIndex")?;
        if p.accessor() || !p.writable() {
            return None;
        }
        match p.value() {
            Value::Num(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => {}
            _ => return None,
        }
    }
    Some(regexp_exec_discard_direct(i, obj, &re, input))
}

/// Committed half of [`regexp_exec_discard_fast`] for a caller that has already proven the
/// direct ordinary RegExp object and pinned its matcher. No JavaScript runs during this routine,
/// so a structural loop helper can validate those guards once and reuse them for every subject.
pub(crate) fn regexp_exec_discard_direct(
    i: &mut Interp,
    obj: &Gc,
    re: &Rc<crate::regex::Regex>,
    input: &crate::lstr::LStr,
) -> Result<bool, Value> {
    let use_last = re.global || re.sticky;
    let read_units = if use_last {
        let b = obj.borrow();
        match b
            .props
            .get("lastIndex")
            .expect("direct regexp lastIndex")
            .value()
        {
            Value::Num(n) => n as usize,
            _ => unreachable!("direct regexp lastIndex guard"),
        }
    } else {
        0
    };
    let text = i.re_text(re.unicode, input);
    let last_units = if use_last { read_units } else { 0 };
    let set_last = |n: usize| {
        let mut b = obj.borrow_mut();
        let p = b.props.get_mut("lastIndex").expect("guarded lastIndex");
        p.set_value(Value::Num(n as f64));
    };
    if last_units > text.unit_index(text.len()) {
        if use_last {
            set_last(0);
        }
        return Ok(false);
    }
    let last = text.elem_at_unit(last_units);
    let matched = regexp_find_text_shared(i, re, &text, last);
    match regexp_match_result(i, matched)? {
        None => {
            if use_last {
                set_last(0);
            }
            Ok(false)
        }
        Some(whole @ (_, end)) => {
            if use_last {
                set_last(text.unit_index(end));
            }
            update_regexp_legacy_statics_lazy(i, re, whole, last, &text, input);
            Ok(true)
        }
    }
}

/// Whether a fresh RegExp literal's inherited `exec` lookup is the untouched intrinsic. A fused
/// JIT expression checks this at the original GetMethod point; any customization executes the
/// ordinary allocation/property/call sequence.
pub(crate) fn regexp_literal_exec_is_canonical(i: &Interp) -> bool {
    i.extra_protos
        .get("RegExp")
        .and_then(|proto| {
            let p = proto.borrow();
            p.props
                .get("exec")
                .filter(|property| !property.accessor())
                .map(Property::value)
        })
        .is_some_and(|value| {
            matches!(
                value,
                Value::Obj(ref function)
                    if matches!(
                        function.borrow().call,
                        Callable::Native(native)
                            if native as usize == regexp_exec as *const () as usize
                    )
            )
        })
}

/// Execute the matcher side of a fresh RegExp literal whose `exec` result and wrapper identity
/// are both dead. The immutable program is still shared by source/flags, every subject is
/// actually matched, and successful captures update the realm's legacy RegExp statics.
pub(crate) fn regexp_literal_exec_discard(
    i: &mut Interp,
    re: &Rc<crate::regex::Regex>,
    input: &crate::lstr::LStr,
) -> Result<(), Abrupt> {
    let text = i.re_text(re.unicode, input);
    let matched = regexp_find_text_shared(i, re, &text, 0);
    if let Some(whole) = regexp_match_abrupt(i, matched)? {
        update_regexp_legacy_statics_lazy(i, re, whole, 0, &text, input);
    }
    Ok(())
}

/// Live builtin-identity guards for a fused dead-result
/// `string.replace(/literal/, "constant")`. Raw data-property reads are intentionally
/// side-effect-free; an accessor or override declines to the ordinary observable path.
pub(crate) fn regexp_literal_replace_is_canonical(i: &Interp) -> bool {
    let string_replace = i
        .string_proto
        .borrow()
        .props
        .get("replace")
        .filter(|property| !property.accessor())
        .map(Property::value)
        .is_some_and(|value| {
            matches!(
                value,
                Value::Obj(ref function)
                    if matches!(
                        function.borrow().call,
                        Callable::Native(native)
                            if native as usize == nf_string_replace as *const () as usize
                    )
            )
        });
    if !string_replace || !regexp::literal_match_dependencies_canonical(i) {
        return false;
    }
    let Some(key) = well_known_key(i, "replace") else {
        return false;
    };
    i.extra_protos
        .get("RegExp")
        .and_then(|proto| {
            let p = proto.borrow();
            p.props
                .get(&key)
                .filter(|property| !property.accessor())
                .map(Property::value)
        })
        .is_some_and(|value| {
            matches!(
                value,
                Value::Obj(ref function)
                    if matches!(
                        function.borrow().call,
                        Callable::Native(native)
                            if native as usize
                                == regexp::re_sym_replace as *const () as usize
                    )
            )
        })
}

/// Execute all matching side effects of a fresh literal used by a dead-result string replace.
/// Replacement construction itself is unobservable once its already-string value and the
/// builtin identities are proven; matching, global iteration, empty-match advancement, and
/// legacy RegExp statics remain live.
pub(crate) fn regexp_literal_replace_discard(
    i: &mut Interp,
    re: &Rc<crate::regex::Regex>,
    input: &crate::lstr::LStr,
) -> Result<(), Abrupt> {
    let text = i.re_text(re.unicode, input);
    let mut position = 0usize;
    loop {
        i.interrupt_poll()?;
        let search_start = position;
        let matched = regexp_find_text_shared(i, re, &text, position);
        let Some((start, end)) = regexp_match_abrupt(i, matched)? else {
            break;
        };
        update_regexp_legacy_statics_lazy(i, re, (start, end), search_start, &text, input);
        if !re.global {
            break;
        }
        position = if end > start {
            end
        } else {
            end.saturating_add(1)
        };
        if position > text.len() {
            break;
        }
    }
    Ok(())
}

/// Live builtin-identity guards for a dead-result `string.match(/literal/)` fusion.
pub(crate) fn regexp_literal_match_is_canonical(i: &Interp) -> bool {
    let string_match = i
        .string_proto
        .borrow()
        .props
        .get("match")
        .filter(|property| !property.accessor())
        .map(Property::value)
        .is_some_and(|value| {
            matches!(
                value,
                Value::Obj(ref function)
                    if matches!(
                        function.borrow().call,
                        Callable::Native(native)
                            if native as usize == nf_string_match as *const () as usize
                    )
            )
        });
    if !string_match || !regexp::literal_match_dependencies_canonical(i) {
        return false;
    }
    let Some(key) = well_known_key(i, "match") else {
        return false;
    };
    i.extra_protos
        .get("RegExp")
        .and_then(|proto| {
            let p = proto.borrow();
            p.props
                .get(&key)
                .filter(|property| !property.accessor())
                .map(Property::value)
        })
        .is_some_and(|value| {
            matches!(
                value,
                Value::Obj(ref function)
                    if matches!(
                        function.borrow().call,
                        Callable::Native(native)
                            if native as usize
                                == regexp::re_sym_match as *const () as usize
                    )
            )
        })
}

/// Run the real matcher for a fresh RegExp literal used by a dead-result `String#match`.
/// The result array and wrapper are unobservable, but global iteration and legacy statics are
/// preserved.
pub(crate) fn regexp_literal_match_discard(
    i: &mut Interp,
    re: &Rc<crate::regex::Regex>,
    input: &crate::lstr::LStr,
) -> Result<(), Abrupt> {
    let text = i.re_text(re.unicode, input);
    let mut position = 0usize;
    loop {
        i.interrupt_poll()?;
        let search_start = position;
        let matched = regexp_find_text_shared(i, re, &text, position);
        let Some((start, end)) = regexp_match_abrupt(i, matched)? else {
            break;
        };
        update_regexp_legacy_statics_lazy(i, re, (start, end), search_start, &text, input);
        if !re.global {
            break;
        }
        position = if end > start {
            end
        } else {
            end.saturating_add(1)
        };
        if position > text.len() {
            break;
        }
    }
    Ok(())
}

/// Allocation-free dead-result `String.prototype.replace` for the direct ordinary RegExp case.
/// The raw property checks are side-effect-free; any customization declines to the exact builtin.
pub(crate) fn string_replace_discard_fast(
    i: &mut Interp,
    input: &Value,
    search: &Value,
    replacement: &Value,
) -> Option<Result<Value, Value>> {
    let (Value::Str(input), Value::Obj(obj), Value::Str(_)) = (input, search, replacement) else {
        return None;
    };
    let ptr = Rc::as_ptr(obj) as usize;
    let re = i.regexps.get(&ptr)?.clone();
    let proto = i.extra_protos.get("RegExp")?.clone();
    let replace_key = well_known_key(i, "replace")?;
    {
        let b = obj.borrow();
        if !matches!(b.exotic, Exotic::None)
            || b.props.contains(&replace_key)
            || b.proto.as_ref().is_none_or(|p| !Rc::ptr_eq(p, &proto))
            || [
                "exec",
                "flags",
                "hasIndices",
                "global",
                "ignoreCase",
                "multiline",
                "dotAll",
                "unicode",
                "unicodeSets",
                "sticky",
            ]
            .iter()
            .any(|name| b.props.contains(name))
        {
            return None;
        }
        let last = b.props.get("lastIndex")?;
        if last.accessor() || !last.writable() {
            return None;
        }
        if re.sticky
            && !re.global
            && !matches!(
                last.value(),
                Value::Num(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0
            )
        {
            return None;
        }
    }
    if !regexp::literal_match_dependencies_canonical(i) {
        return None;
    }
    let canonical = {
        let p = proto.borrow();
        let Value::Obj(method) = p.props.get(&replace_key)?.value() else {
            return None;
        };
        let canonical = matches!(
            method.borrow().call,
            Callable::Native(nf)
                if nf as usize == regexp::re_sym_replace as *const () as usize
        );
        canonical
    };
    if !canonical {
        return None;
    }
    Some(regexp::re_sym_replace_discard_direct(i, obj, &re, input))
}

/// Allocation-free dead-result `String.prototype.split` for a direct ordinary RegExp separator.
/// The child routine validates every protocol/species/flag dependency before touching state.
pub(crate) fn string_split_discard_fast(
    i: &mut Interp,
    input: &Value,
    separator: &Value,
    limit: &Value,
) -> Option<Result<Value, Value>> {
    regexp::re_sym_split_discard_fast(i, input, separator, limit)
}

/// Allocation-free dead-result `String.prototype.match` for a direct ordinary RegExp argument.
/// Every observable lookup (IsRegExp, `@@match`, flags, exec) is proven to resolve to the realm
/// intrinsics; the matcher still runs and updates `lastIndex`/legacy statics, while the result
/// array or `null` is never materialized. Returns `None` before touching any state on a guard
/// miss, so the generic builtin runs unmodified.
pub(crate) fn string_match_discard_fast(
    i: &mut Interp,
    input: &Value,
    search: &Value,
) -> Option<Result<Value, Value>> {
    let (Value::Str(input), Value::Obj(obj)) = (input, search) else {
        return None;
    };
    let ptr = Rc::as_ptr(obj) as usize;
    let re = i.regexps.get(&ptr)?.clone();
    let proto = i.extra_protos.get("RegExp")?.clone();
    let match_key = well_known_key(i, "match")?;
    {
        let b = obj.borrow();
        if !matches!(b.exotic, Exotic::None)
            || b.props.contains(&match_key)
            || b.proto.as_ref().is_none_or(|p| !Rc::ptr_eq(p, &proto))
            || [
                "exec",
                "flags",
                "hasIndices",
                "global",
                "ignoreCase",
                "multiline",
                "dotAll",
                "unicode",
                "unicodeSets",
                "sticky",
            ]
            .iter()
            .any(|name| b.props.contains(name))
        {
            return None;
        }
        let last = b.props.get("lastIndex")?;
        if last.accessor() || !last.writable() {
            return None;
        }
        match last.value() {
            Value::Num(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => {}
            _ => return None,
        }
    }
    if !regexp::literal_match_dependencies_canonical(i) {
        return None;
    }
    let canonical = {
        let p = proto.borrow();
        let Value::Obj(method) = p.props.get(&match_key)?.value() else {
            return None;
        };
        let canonical = matches!(
            method.borrow().call,
            Callable::Native(nf)
                if nf as usize == regexp::re_sym_match as *const () as usize
        );
        canonical
    };
    if !canonical {
        return None;
    }
    Some(regexp::re_sym_match_discard_direct(i, obj, &re, input))
}

/// Dead-result call routing for the bytecode VM. A `CallWithThis` immediately followed by
/// `Op::Pop` proves the call's result is unobservable; the JIT fuses exactly this pattern into
/// the allocation-free discard intrinsics, and the bytecode VM now shares the same guarded
/// routines for the three canonical natives (RegExp exec, String replace, String split). Every
/// receiver/prototype/flag guard lives inside the discard routines, which return `None` before
/// touching any state; the caller then runs the generic call unmodified, so a miss is
/// observably identical. Errors from a committed discard path propagate as the thrown value.
pub(crate) fn vm_discard_call(
    i: &mut Interp,
    this: &Value,
    method: &Value,
    args: &[Value],
) -> Result<Option<Value>, Value> {
    let Value::Obj(obj) = method else {
        return Ok(None);
    };
    let Callable::Native(nf) = &obj.borrow().call else {
        return Ok(None);
    };
    let native = (*nf) as usize;
    if native == regexp_exec as *const () as usize {
        // RegExp.prototype.exec(subject): receiver is the regexp, the subject is arg 0.
        let Some((first, _)) = args.split_first() else {
            return Ok(None);
        };
        let Value::Str(input) = first else {
            return Ok(None);
        };
        return match regexp_exec_discard_fast(i, this, input) {
            Some(Ok(_)) => Ok(Some(Value::Undefined)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        };
    }
    if native == nf_string_replace as *const () as usize {
        // String.prototype.replace(search, replacement): receiver is the string.
        if args.len() < 2 {
            return Ok(None);
        }
        return match string_replace_discard_fast(i, this, &args[0], &args[1]) {
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        };
    }
    if native == nf_string_split as *const () as usize {
        // String.prototype.split(separator): receiver is the string; only an undefined limit
        // (the common statement form) takes the discard path, matching the JIT intrinsic.
        if args.is_empty() {
            return Ok(None);
        }
        return match string_split_discard_fast(i, this, &args[0], &Value::Undefined) {
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        };
    }
    if native == nf_string_match as *const () as usize {
        // String.prototype.match(regexp): receiver is the string.
        if args.is_empty() {
            return Ok(None);
        }
        return match string_match_discard_fast(i, this, &args[0]) {
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        };
    }
    Ok(None)
}

/// Refresh the %RegExp% constructor's legacy static state after a successful RegExpBuiltinExec.
/// Record a successful match for the legacy `RegExp.$1`-style statics — deferred: the 14 strings
/// (including full left/right-context copies of the subject) only materialize when one of the
/// accessors actually reads them (`flush_regexp_legacy`). A pending match for a *different*
/// realm's constructor flushes first so its statics aren't lost.
fn update_regexp_legacy_statics(
    i: &mut Interp,
    re: &crate::regex::Regex,
    caps: &[Option<(usize, usize)>],
    text: &Rc<crate::regex::ReText>,
    input: &crate::lstr::LStr,
) {
    let ctor = match i.extra_protos.get("%RegExpCtor%") {
        Some(c) => c.clone(),
        None => return,
    };
    if let Some(prev) = &i.regexp_last {
        if prev.ctor.as_ptr() != Rc::as_ptr(&ctor) {
            flush_regexp_legacy(i);
        }
    }
    if let Some(previous) = i
        .regexp_last
        .as_mut()
        .filter(|previous| previous.ctor.as_ptr() == Rc::as_ptr(&ctor))
    {
        previous.input = input.clone();
        previous.text = text.clone();
        previous.caps.clear();
        previous.caps.extend_from_slice(caps);
        previous.ngroups = re.ngroups;
        previous.lazy_captures = None;
        return;
    }
    i.regexp_last = Some(crate::interpreter::RegexpLastMatch {
        ctor: Rc::downgrade(&ctor),
        input: input.clone(),
        text: text.clone(),
        caps: caps.to_vec(),
        ngroups: re.ngroups,
        lazy_captures: None,
    });
}

fn update_regexp_legacy_statics_lazy(
    i: &mut Interp,
    re: &Rc<crate::regex::Regex>,
    whole: (usize, usize),
    search_start: usize,
    text: &Rc<crate::regex::ReText>,
    input: &crate::lstr::LStr,
) {
    let ctor = match i.extra_protos.get("%RegExpCtor%") {
        Some(c) => c.clone(),
        None => return,
    };
    if let Some(prev) = &i.regexp_last {
        if prev.ctor.as_ptr() != Rc::as_ptr(&ctor) {
            flush_regexp_legacy(i);
        }
    }
    if let Some(previous) = i
        .regexp_last
        .as_mut()
        .filter(|previous| previous.ctor.as_ptr() == Rc::as_ptr(&ctor))
    {
        previous.input = input.clone();
        previous.text = text.clone();
        previous.caps.clear();
        previous.caps.push(Some(whole));
        previous.ngroups = re.ngroups;
        previous.lazy_captures = Some((re.clone(), search_start));
        return;
    }
    i.regexp_last = Some(crate::interpreter::RegexpLastMatch {
        ctor: Rc::downgrade(&ctor),
        input: input.clone(),
        text: text.clone(),
        caps: vec![Some(whole)],
        ngroups: re.ngroups,
        lazy_captures: Some((re.clone(), search_start)),
    });
}

/// Materialize the pending legacy-statics match (if any) into its constructor's hidden props.
/// Called by the `RegExp.$1`/`lastMatch`/… accessors before they read.
pub(super) fn flush_regexp_legacy(i: &mut Interp) {
    let Some(mut m) = i.regexp_last.take() else {
        return;
    };
    // The constructor's Realm was collected: nothing can observe its statics.
    let Some(ctor) = m.ctor.upgrade() else {
        return;
    };
    if let Some((re, start)) = m.lazy_captures.take() {
        let matched = regexp_exec_text_shared(i, &re, &m.text, start);
        if let Ok(Some(captures)) = matched {
            m.caps.clear();
            m.caps.extend_from_slice(&captures);
        }
    }
    let (caps, text, ctor) = (&m.caps, &m.text, &ctor);
    let put = |k: &'static str, v: String| {
        ctor.borrow_mut()
            .props
            .insert(k, Property::data(Value::lstr(v), true, false, false));
    };
    let (start, end) = caps[0].unwrap();
    put("#\u{0}legacy_input", m.input.to_string());
    put("#\u{0}legacy_lastMatch", text.slice(start, end));
    put("#\u{0}legacy_leftContext", text.slice(0, start));
    put("#\u{0}legacy_rightContext", text.slice(end, text.len()));
    let cap_str = |k: usize| {
        caps.get(k)
            .copied()
            .flatten()
            .map(|(a, b)| text.slice(a, b))
            .unwrap_or_default()
    };
    put(
        "#\u{0}legacy_lastParen",
        if m.ngroups >= 1 {
            cap_str(m.ngroups)
        } else {
            String::new()
        },
    );
    const DOLLARS: [&str; 9] = [
        "#\u{0}legacy_$1",
        "#\u{0}legacy_$2",
        "#\u{0}legacy_$3",
        "#\u{0}legacy_$4",
        "#\u{0}legacy_$5",
        "#\u{0}legacy_$6",
        "#\u{0}legacy_$7",
        "#\u{0}legacy_$8",
        "#\u{0}legacy_$9",
    ];
    for (k, slot) in DOLLARS.iter().enumerate() {
        put(slot, cap_str(k + 1));
    }
}

/// ToIndex: a non-negative integer in [0, 2^53-1]; anything else is a RangeError.
fn to_index(i: &mut Interp, v: &Value) -> Result<usize, Value> {
    let n = ab(i.to_number(v))?;
    let n = if n.is_nan() { 0.0 } else { n.trunc() };
    if !(0.0..=9007199254740991.0).contains(&n) {
        return Err(i.make_error("RangeError", "index out of range"));
    }
    Ok(n as usize)
}

pub fn install(it: &mut Interp) {
    // Primitive globals.
    let g = it.global.clone();
    // The global value properties are { writable:false, enumerable:false, configurable:false }.
    for (name, val) in [
        ("undefined", Value::Undefined),
        ("NaN", Value::Num(f64::NAN)),
        ("Infinity", Value::Num(f64::INFINITY)),
    ] {
        g.borrow_mut()
            .props
            .insert(name, Property::data(val, false, false, false));
    }
    set_builtin(&g, "globalThis", Value::Obj(g.clone()));

    function_proto::install_function_proto(it);
    install_object(it);
    // Symbol before Array/String so `Symbol.iterator` exists when they define `@@iterator`.
    primitives::install_symbol(it);
    // After Symbol so the intrinsics' @@toStringTag resolves.
    function_proto::install_generator_function_ctors(it);
    // Function.prototype[@@hasInstance] (default OrdinaryHasInstance) — installed after Symbol so the
    // well-known key exists; non-writable/non-configurable.
    if let Some(key) = well_known_key(it, "hasInstance") {
        let f = it.make_native("[Symbol.hasInstance]", 1, default_has_instance);
        it.function_proto
            .borrow_mut()
            .props
            .insert(key, Property::data(Value::Obj(f), false, false, false));
    }
    install_iterator(it);
    install_array(it);
    install_string(it);
    primitives::install_number(it);
    primitives::install_boolean(it);
    primitives::install_bigint(it);
    math::install_math(it);
    errors::install_errors(it);
    reflect::install_reflect(it);
    proxy::install_proxy(it);
    promise::install_promise(it);
    json::install_json(it);
    collections::install_collections(it);
    date::install_date(it);
    typedarray::install_typed_arrays(it);
    dataview::install_dataview(it);
    typedarray::install_shared_array_buffer(it);
    regexp::install_regexp(it);
    globals::install_globals(it);
    globals::install_console(it);
    host::install_host(it);
    atomics::install_atomics(it);
    weakrefs::install_weak_refs(it);
    disposable::install_disposable_stack(it);
    shadowrealm::install_shadow_realm(it);
    crate::temporal::install(it);
    #[cfg(feature = "intl")]
    crate::intl::install(it);
}

/// Whether a value is a constructor: a callable with an own `prototype` (built-in ctors / user
/// non-arrow functions, which also set is_constructor) — matching the `new` constructability rule.
fn is_constructor_value(v: &Value) -> bool {
    matches!(v, Value::Obj(o) if {
        let b = o.borrow();
        match &b.call {
            Callable::None => false,
            // A user function's constructability is determined by its shape: arrows, methods,
            // generators and async functions are never constructors (a generator still has a
            // `prototype` property, so the fallback below would misclassify it).
            Callable::User(user) => {
                !user.func.is_arrow
                    && !user.func.is_method
                    && !user.func.is_generator
                    && !user.func.is_async
            }
            _ => b.is_constructor || b.props.contains("prototype"),
        }
    })
}

/// Install the `get [Symbol.species]` accessor (returns the receiver `this`) on a constructor.
fn install_species(it: &Interp, ctor: &Gc) {
    if let Some(key) = well_known_key(it, "species") {
        let getter = it.make_native("get [Symbol.species]", 0, nf_species_getter);
        ctor.borrow_mut().props.insert(
            key,
            Property::accessor_prop(Some(Value::Obj(getter)), None, false, true),
        );
    }
}

fn nf_species_getter(_i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    Ok(this)
}

/// The internal property key for a well-known `Symbol.<name>`.
pub(crate) fn well_known_key(it: &Interp, name: &str) -> Option<Rc<str>> {
    // `install_symbol` mints these in this fixed order. Protocol dispatch is hot enough that a
    // name scan here (notably every `instanceof`) is visible in native profiles.
    let index = match name {
        "iterator" => 0,
        "asyncIterator" => 1,
        "hasInstance" => 2,
        "isConcatSpreadable" => 3,
        "match" => 4,
        "matchAll" => 5,
        "replace" => 6,
        "search" => 7,
        "species" => 8,
        "split" => 9,
        "toPrimitive" => 10,
        "toStringTag" => 11,
        "unscopables" => 12,
        "dispose" => 13,
        "asyncDispose" => 14,
        "metadata" => 15,
        _ => return None,
    };
    it.wk_syms.get(index).and_then(|(n, _, key)| {
        debug_assert_eq!(*n, name);
        (*n == name).then(|| key.clone())
    })
}

fn map_ptr(this: &Value) -> Option<usize> {
    this.as_obj().map(|o| Rc::as_ptr(o) as usize)
}

/// ArrayCreate(length) (ECMA-262 §10.4.2.2) with the current Realm's %Array.prototype%: a
/// *RangeError* past 2^32 − 1, else an Array of that `length` with no elements.
fn array_create(i: &mut Interp, length: u64) -> Result<Value, Value> {
    if length > u64::from(u32::MAX) {
        return Err(i.make_error("RangeError", "invalid array length"));
    }
    let array = i.make_array(Vec::new());
    if length != 0 {
        if let Value::Obj(object) = &array {
            object
                .borrow_mut()
                .props
                .set_array_length_value(length as f64);
        }
    }
    Ok(array)
}

/// ArraySpeciesCreate(originalArray, length) for a Number length (a LengthOfArrayLike result:
/// a non-negative integer no greater than 2^53 − 1, exact in an f64).
pub(crate) fn array_species_create_length(
    i: &mut Interp,
    original: &Value,
    length: f64,
) -> Result<Value, Value> {
    debug_assert!(length >= 0.0 && length.fract() == 0.0 && length <= 9007199254740991.0);
    array_species_create(i, original, length as u64)
}

/// Whether ArraySpeciesCreate(`original`, length) is unobservably ArrayCreate(length): an
/// ordinary Array (IsArray, not a Proxy) without an own "constructor", inheriting from the
/// current Realm's %Array.prototype% a "constructor" data property that holds this Realm's
/// %Array%, whose @@species is still the intrinsic getter (which returns its receiver). Every
/// Get in steps 3–5 then reads a data property or calls that getter, nothing author-visible
/// runs, and step 8 constructs with %Array% itself: Construct(%Array%, « 𝔽(length) ») is
/// ArrayCreate(length) with %Array.prototype% (ECMA-262 §23.1.1.1; its *RangeError* is
/// ArrayCreate's).
fn array_species_is_intrinsic(i: &Interp, original: &Gc) -> bool {
    {
        let body = original.borrow();
        if !body.ic_plain.get()
            || !matches!(body.exotic, Exotic::Array)
            || !body
                .proto
                .as_ref()
                .is_some_and(|p| Rc::ptr_eq(p, &i.array_proto))
            || body.props.contains("constructor")
        {
            return false;
        }
    }
    let (Some(array_ctor), Some(slot)) = (i.array_ctor.as_ref(), intrinsic_species_slot(i)) else {
        return false;
    };
    let prototype = i.array_proto.borrow();
    prototype
        .props
        .property_at(slot)
        .is_some_and(|constructor| {
            !constructor.accessor()
                && matches!(constructor.value(), Value::Obj(c) if Rc::ptr_eq(&c, array_ctor))
        })
}

/// The `entries` slot of %Array.prototype%'s own "constructor" data property while %Array%'s
/// @@species is the intrinsic getter, under the epoch-validated [`Interp::species_protector`].
fn intrinsic_species_slot(i: &Interp) -> Option<usize> {
    let epoch = crate::value::proto_epoch();
    if epoch == u32::MAX {
        return None; // permanently invalidated: never trust a proof
    }
    let (proven, slot) = i.species_protector.get();
    if proven != epoch {
        let slot = verify_intrinsic_species(i).map_or(u32::MAX, |slot| slot as u32);
        i.species_protector.set((epoch, slot));
        return (slot != u32::MAX).then_some(slot as usize);
    }
    (slot != u32::MAX).then_some(slot as usize)
}

/// Establish [`Interp::species_protector`]'s proof from scratch, marking both objects as
/// prototypes so later redefinitions or structural changes invalidate it.
fn verify_intrinsic_species(i: &Interp) -> Option<usize> {
    let array_ctor = i.array_ctor.as_ref()?;
    let species = well_known_key(i, "species")?;
    let intrinsic_getter = array_ctor.borrow().props.get(&species).is_some_and(|p| {
        p.accessor()
            && matches!(p.getter(), Some(Value::Obj(getter)) if matches!(
                getter.borrow().call,
                crate::value::Callable::Native(f)
                    if f as *const () == nf_species_getter as *const ()
            ))
    });
    if !intrinsic_getter {
        return None;
    }
    let prototype = i.array_proto.borrow();
    let slot = prototype.props.slot_of("constructor")?;
    if prototype.props.property_at(slot)?.accessor() {
        return None;
    }
    prototype.props.mark_proto();
    array_ctor.borrow().props.mark_proto();
    Some(slot)
}

/// ArraySpeciesCreate(originalArray, length) (ECMA-262 §10.4.2.3).
fn array_species_create(i: &mut Interp, original: &Value, len: u64) -> Result<Value, Value> {
    if let Value::Obj(object) = original {
        if array_species_is_intrinsic(i, object) {
            return array_create(i, len);
        }
    }
    // 1–2. IsArray pierces proxies (a proxy over an array is an array).
    if !json_is_array(i, original)? {
        return array_create(i, len);
    }
    // 3. Get(originalArray, "constructor").
    let mut c = ab(i.get_member(original, "constructor"))?;
    // 4. Another Realm's intrinsic %Array% counts as the default; its @@species is never read.
    //    SameValue(C, ctorRealm.[[Intrinsics]].[[%Array%]]) holds exactly when C is the %Array%
    //    of the Realm GetFunctionRealm(C) names.
    if is_constructor_value(&c) {
        if let Value::Obj(co) = &c {
            let is_array_ctor =
                |ctor: &Option<Gc>| ctor.as_ref().is_some_and(|ac| Rc::ptr_eq(ac, co));
            let foreign_array = !is_array_ctor(&i.array_ctor)
                && i.realms
                    .values()
                    .any(|realm| is_array_ctor(&realm.array_ctor));
            if foreign_array {
                return array_create(i, len);
            }
        }
    }
    // 5. Get(C, @@species); null becomes undefined.
    if matches!(&c, Value::Obj(_)) {
        if let Some(key) = well_known_key(i, "species") {
            c = ab(i.get_member(&c, &key))?;
        }
        if matches!(c, Value::Null) {
            c = Value::Undefined;
        }
    }
    // 6. Undefined selects ArrayCreate.
    if matches!(c, Value::Undefined) {
        return array_create(i, len);
    }
    // 7. IsConstructor.
    if !is_constructor_value(&c) {
        return Err(i.make_error("TypeError", "Array @@species is not a constructor"));
    }
    // 8. Construct(C, « 𝔽(length) »); the current Realm's %Array% constructs ArrayCreate(length).
    if let (Value::Obj(s), Some(ac)) = (&c, i.array_ctor.as_ref()) {
        if Rc::ptr_eq(s, ac) {
            return array_create(i, len);
        }
    }
    ab(i.construct(c, &[Value::Num(len as f64)]))
}

/// Own enumerable string keys in spec [[OwnPropertyKeys]] order (array-index ascending first).
fn ordered_enum_keys(o: &Gc) -> Vec<Rc<str>> {
    let b = o.borrow();
    b.props
        .ordered_keys()
        .into_iter()
        .filter(|k| {
            !Interp::is_sym_key(k) && b.props.get(k).map(|p| p.enumerable()).unwrap_or(false)
        })
        .collect()
}

/// The property key for `@@toStringTag` (`Symbol.toStringTag`), if Symbol is installed.
pub(crate) fn to_string_tag_key(i: &Interp) -> Option<String> {
    let sym = i.global.borrow().props.get("Symbol").map(|p| p.value())?;
    if let Value::Obj(o) = sym {
        if let Some(p) = o.borrow().props.get("toStringTag") {
            if let Value::Sym(d) = p.value() {
                return Some(Interp::sym_key(&d));
            }
        }
    }
    None
}

/// Install a non-enumerable, configurable `@@toStringTag` data property.
pub(crate) fn set_to_string_tag(i: &Interp, obj: &Gc, tag: &str) {
    if let Some(key) = to_string_tag_key(i) {
        obj.borrow_mut().props.insert(
            key,
            Property::data(Value::lstr(tag.to_string()), false, false, true),
        );
    }
}

/// The Object.prototype.toString builtin tag for `this` (the `[object <tag>]` body without an
/// overriding `@@toStringTag`).
fn builtin_tag(i: &Interp, this: &Value) -> &'static str {
    match this {
        Value::Num(_) => "Number",
        Value::Str(_) => "String",
        Value::Bool(_) => "Boolean",
        Value::Obj(o) => {
            let b = o.borrow();
            if matches!(b.exotic, Exotic::Array) {
                "Array"
            } else if matches!(b.exotic, Exotic::Arguments) {
                "Arguments"
            } else if !matches!(b.call, Callable::None) {
                "Function"
            } else if matches!(b.exotic, Exotic::Error(_)) {
                "Error"
            } else if matches!(b.exotic, Exotic::BoolWrap(_)) {
                "Boolean"
            } else if matches!(b.exotic, Exotic::NumWrap(_)) {
                "Number"
            } else if matches!(b.exotic, Exotic::StrWrap(_)) {
                "String"
            } else if b.props.contains("#\u{0}date_ms") {
                "Date"
            } else if i.regexps.contains_key(&(Rc::as_ptr(o) as usize)) {
                "RegExp"
            } else {
                "Object"
            }
        }
        _ => "Object",
    }
}

/// Brand-check a Map/Set receiver: it must be an object carrying a collection data slot (every
/// Map/Set/WeakMap/WeakSet gets one at construction), else TypeError.
fn coll_ptr(i: &Interp, this: &Value) -> Result<usize, Value> {
    coll_ptr_kind(i, this, None)
}

/// Resolve a Map/Set backing pointer with a brand check. `want = Some("Map"|"Set")` requires that
/// exact kind; `None` accepts either strong collection but still rejects WeakMap/WeakSet and plain
/// objects (which lack a strong `[[MapData]]`/`[[SetData]]` slot).
fn coll_ptr_kind(i: &Interp, this: &Value, want: Option<&str>) -> Result<usize, Value> {
    let err = || i.make_error("TypeError", "method called on an incompatible receiver");
    let o = this.as_obj().ok_or_else(err)?;
    let ptr = Rc::as_ptr(o) as usize;
    let data = i.map_data.get(&ptr).ok_or_else(err)?;
    if want.is_some_and(|want| data.kind().name() != want) {
        return Err(err());
    }
    Ok(ptr)
}

/// Build a live Map/Set iterator. `kind`: 0 = values, 1 = keys, 2 = [key,value].
/// Like [`collection_iter`] but brand-checks the exact collection kind ("Set" / "Map").
fn collection_iter_kind(
    i: &mut Interp,
    this: &Value,
    kind: u8,
    want: &str,
) -> Result<Value, Value> {
    coll_ptr_kind(i, this, Some(want))?;
    collection_iter(i, this, kind)
}

/// forEach shared by Map/Set, brand-checking the exact kind.
fn collection_for_each(
    i: &mut Interp,
    this: Value,
    a: &[Value],
    want: Option<&str>,
) -> Result<Value, Value> {
    let ptr = coll_ptr_kind(i, &this, want)?;
    let cb = arg(a, 0);
    if !cb.is_callable() {
        return Err(i.make_error("TypeError", "forEach callback is not callable"));
    }
    let cb = crate::callback::Callback::new(cb);
    let cb_this = arg(a, 1);
    // Cursors survive clear/compaction, including recursively nested forEach calls. Never hold a
    // collection borrow across JavaScript: callbacks may mutate this collection arbitrarily.
    let cursor = i.map_data.get_mut(&ptr).unwrap().cursor();
    while let Some((k, v)) = i.map_data[&ptr].next(&cursor) {
        ab(cb.call(i, cb_this.clone(), [v, k, this.clone()]))?;
    }
    Ok(Value::Undefined)
}

fn collection_iter(i: &mut Interp, this: &Value, kind: u8) -> Result<Value, Value> {
    let ptr = coll_ptr(i, this)?;
    let data = i.map_data.get_mut(&ptr).unwrap();
    let brand = data.kind();
    let cursor = data.cursor();
    let key = if brand == crate::ordered_collection::CollectionKind::Set {
        "%SetIteratorPrototype%"
    } else {
        "%MapIteratorPrototype%"
    };
    let proto = i
        .extra_protos
        .get(key)
        .cloned()
        .or_else(|| i.extra_protos.get("%IteratorPrototype%").cloned());
    let obj = Object::new(proto);
    i.gc_pin(&obj);
    i.collection_iterators.insert(
        Rc::as_ptr(&obj) as usize,
        crate::ordered_collection::CollectionIterator {
            target: Some(this.as_obj().unwrap().clone()),
            cursor: Some(cursor),
            kind,
            brand,
        },
    );
    Ok(Value::Obj(obj))
}

/// ECMA-262 GeneratorValidate checks the exact native iterator brand, even after exhaustion.
fn map_set_iter_step(
    i: &mut Interp,
    this: Value,
    brand: crate::ordered_collection::CollectionKind,
) -> Result<Option<Value>, Value> {
    let ptr = map_ptr(&this).ok_or_else(|| i.make_error("TypeError", "not a Map/Set Iterator"))?;
    let state = i
        .collection_iterators
        .get(&ptr)
        .filter(|state| state.brand == brand)
        .ok_or_else(|| i.make_error("TypeError", "incompatible collection iterator"))?;
    let kind = state.kind;
    let entry = match (&state.target, &state.cursor) {
        (Some(target), Some(cursor)) => i
            .map_data
            .get(&(Rc::as_ptr(target) as usize))
            .and_then(|data| data.next(cursor)),
        _ => None,
    };
    if let Some((key, value)) = entry {
        let value = match kind {
            1 => key,
            2 => i.make_array(vec![key, value]),
            _ => value,
        };
        Ok(Some(value))
    } else {
        let state = i.collection_iterators.get_mut(&ptr).unwrap();
        state.target = None;
        state.cursor = None;
        Ok(None)
    }
}

fn map_iter_step(i: &mut Interp, this: Value) -> Result<Option<Value>, Value> {
    map_set_iter_step(i, this, crate::ordered_collection::CollectionKind::Map)
}

fn set_iter_step(i: &mut Interp, this: Value) -> Result<Option<Value>, Value> {
    map_set_iter_step(i, this, crate::ordered_collection::CollectionKind::Set)
}

fn map_iter_next(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let step = map_iter_step(i, this)?;
    Ok(materialize_iterator_step(i, step))
}

fn set_iter_next(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let step = set_iter_step(i, this)?;
    Ok(materialize_iterator_step(i, step))
}

/// IteratorStepValue immediately consumes the intrinsic's fresh result object.
/// This entry returns the same state transition/value without materializing that
/// unobservable wrapper. Public next() calls still allocate distinct result objects.
/// ECMA-262 IteratorNext / IteratorStepValue / CreateIteratorResultObject, e28783d5.
pub(crate) type IntrinsicIteratorStep = fn(&mut Interp, Value) -> Result<Option<Value>, Value>;

pub(crate) fn intrinsic_iterator_step(native: NativeFn) -> Option<IntrinsicIteratorStep> {
    if std::ptr::fn_addr_eq(native, array_iter_next as NativeFn) {
        Some(array_iter_step)
    } else if std::ptr::fn_addr_eq(native, string_iter_next as NativeFn) {
        Some(string_iter_step)
    } else if std::ptr::fn_addr_eq(native, map_iter_next as NativeFn) {
        Some(map_iter_step)
    } else if std::ptr::fn_addr_eq(native, set_iter_next as NativeFn) {
        Some(set_iter_step)
    } else {
        None
    }
}

fn materialize_iterator_step(i: &mut Interp, step: Option<Value>) -> Value {
    match step {
        Some(value) => iter_result(i, value, false),
        None => iter_result(i, Value::Undefined, true),
    }
}

/// A bound handler `(target, this=Undefined, [bound...])` used to thread per-element state into a
/// `Promise.all` reaction without closures.
fn make_bound(i: &Interp, target: NativeFn, bound_args: Vec<Value>) -> Value {
    make_bound_len(i, target, bound_args, 1.0)
}

/// Like [`make_bound`] but with an explicit observable `length`. These internal closures are
/// anonymous built-in functions, so they carry own `name` (the empty string) and `length` data
/// properties (both non-writable, non-enumerable, configurable), which test262's verifyProperty checks.
pub(crate) fn make_bound_len(
    i: &Interp,
    target: NativeFn,
    bound_args: Vec<Value>,
    length: f64,
) -> Value {
    let t = i.make_native("", 1, target);
    let obj = Object::new(Some(i.function_proto.clone()));
    {
        let mut b = obj.borrow_mut();
        b.call = Callable::bound(t, Value::Undefined, bound_args);
        b.props.insert(
            "length",
            Property::data(Value::Num(length), false, false, true),
        );
        b.props
            .insert("name", Property::data(Value::lstr(""), false, false, true));
    }
    Value::Obj(obj)
}

/// The executor passed to `new C(executor)` inside NewPromiseCapability: capture the resolve/reject
/// functions (and reject a second invocation). `args = [cap, resolve, reject]`.
fn capability_executor(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let cap = arg(args, 0);
    let already = !matches!(ab(i.get_member(&cap, "#\u{0}resolve"))?, Value::Undefined)
        || !matches!(ab(i.get_member(&cap, "#\u{0}reject"))?, Value::Undefined);
    if already {
        return Err(i.make_error("TypeError", "promise capability executor already invoked"));
    }
    set_internal_obj(&cap, "#\u{0}resolve", arg(args, 1));
    set_internal_obj(&cap, "#\u{0}reject", arg(args, 2));
    Ok(Value::Undefined)
}

/// `Promise.prototype.finally` helpers: the value/reason pass-through thunks and the then/catch
/// reactions that run `onFinally` then forward the original settlement.
fn pf_return(_i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    Ok(arg(a, 0))
}
fn pf_throw(_i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    Err(arg(a, 0))
}
fn pf_then_finally(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    // bound args [onFinally, C]; called with (value)
    let (on_finally, c, value) = (arg(a, 0), arg(a, 1), arg(a, 2));
    let result = ab(i.call(on_finally, Value::Undefined, &[]))?;
    let resolve = ab(i.get_member(&c, "resolve"))?;
    let p = ab(i.call(resolve, c.clone(), &[result]))?;
    let thunk = make_bound_len(i, pf_return, vec![value], 0.0);
    let then = ab(i.get_member(&p, "then"))?;
    ab(i.call(then, p, &[thunk]))
}
fn pf_catch_finally(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let (on_finally, c, reason) = (arg(a, 0), arg(a, 1), arg(a, 2));
    let result = ab(i.call(on_finally, Value::Undefined, &[]))?;
    let resolve = ab(i.get_member(&c, "resolve"))?;
    let p = ab(i.call(resolve, c.clone(), &[result]))?;
    let thrower = make_bound_len(i, pf_throw, vec![reason], 0.0);
    let then = ab(i.get_member(&p, "then"))?;
    ab(i.call(then, p, &[thrower]))
}

/// SpeciesConstructor(O, defaultCtor): O.constructor[@@species], or the default.
fn species_constructor(i: &mut Interp, obj: &Value, default_ctor: &Value) -> Result<Value, Value> {
    let c = ab(i.get_member(obj, "constructor"))?;
    if matches!(c, Value::Undefined) {
        return Ok(default_ctor.clone());
    }
    if !matches!(c, Value::Obj(_)) {
        return Err(i.make_error("TypeError", "constructor is not an object"));
    }
    let species = match well_known_key(i, "species") {
        Some(k) => ab(i.get_member(&c, &k))?,
        None => Value::Undefined,
    };
    if matches!(species, Value::Undefined | Value::Null) {
        return Ok(default_ctor.clone());
    }
    if !is_constructor_value(&species) {
        return Err(i.make_error("TypeError", "@@species is not a constructor"));
    }
    Ok(species)
}

/// GetPromiseResolve(C): read `C.resolve` once, requiring it to be callable.
fn get_promise_resolve(i: &mut Interp, ctor: &Value) -> Result<Value, Value> {
    let r = ab(i.get_member(ctor, "resolve"))?;
    if !r.is_callable() {
        return Err(i.make_error(
            "TypeError",
            "Promise combinator: this.resolve is not callable",
        ));
    }
    Ok(r)
}

/// NewPromiseCapability(C): `new C(executor)`, capturing the resolve/reject functions the executor is
/// called with (and validating they're callable). Returns the result promise built by `C`.
fn new_promise_capability(i: &mut Interp, ctor: &Value) -> Result<Value, Value> {
    new_promise_capability_full(i, ctor).map(|(p, _, _)| p)
}

/// NewPromiseCapability returning `(promise, resolve, reject)`. The resolve/reject are the
/// constructor's own capability functions, so a combinator that resolves through them works with a
/// subclass / foreign Promise constructor (not just the native machinery).
/// PromiseResolveThenableJob: a microtask handler that invokes `then.call(thenable, res, rej)`,
/// rejecting through `rej` if the call throws.
pub(crate) fn make_thenable_job(
    i: &mut Interp,
    then: crate::interpreter::JobCallback,
    thenable: Value,
    res: Value,
    rej: Value,
) -> Value {
    let mut args = vec![then.callback, thenable, res, rej];
    if let Some(global) = then.script_caller {
        args.push(Value::Obj(global));
    }
    make_bound(i, thenable_job_run, args)
}

fn thenable_job_run(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    let then = crate::interpreter::JobCallback {
        callback: arg(args, 0),
        script_caller: match args.get(4) {
            Some(Value::Obj(global)) => Some(global.clone()),
            _ => None,
        },
        internal: crate::interpreter::InternalJob::None,
    };
    let thenable = arg(args, 1);
    let res = arg(args, 2);
    let rej = arg(args, 3);
    if let Err(e) = ab(i.call_job_callback(&then, thenable, &[res, rej.clone()])) {
        let _ = i.call(rej, Value::Undefined, &[e]);
    }
    Ok(Value::Undefined)
}

/// The combinators' final `Call(capability.[[Resolve]])`: an abrupt completion rejects the
/// capability instead of being swallowed (IfAbruptRejectPromise).
fn capability_resolve_or_reject(i: &mut Interp, resolve_fn: Value, reject_fn: Value, v: Value) {
    if let Err(e) = i.call(resolve_fn, Value::Undefined, &[v]) {
        let reason = crate::interpreter::abrupt_value(e);
        let _ = i.call(reject_fn, Value::Undefined, &[reason]);
    }
}

fn new_promise_capability_full(
    i: &mut Interp,
    ctor: &Value,
) -> Result<(Value, Value, Value), Value> {
    if !ctor.is_callable() {
        return Err(i.make_error(
            "TypeError",
            "NewPromiseCapability: receiver is not a constructor",
        ));
    }
    let cap = i.new_object();
    // GetCapabilitiesExecutor is an anonymous built-in function of length 2.
    let executor = make_bound_len(i, capability_executor, vec![Value::Obj(cap.clone())], 2.0);
    let promise = ab(i.construct(ctor.clone(), &[executor]))?;
    let resolve = ab(i.get_member(&Value::Obj(cap.clone()), "#\u{0}resolve"))?;
    let reject = ab(i.get_member(&Value::Obj(cap.clone()), "#\u{0}reject"))?;
    if !resolve.is_callable() || !reject.is_callable() {
        return Err(i.make_error("TypeError", "promise capability functions are not callable"));
    }
    Ok((promise, resolve, reject))
}

/// `Promise.all` per-element fulfill reaction. `args = [resultPromise, index, value]`.
fn promise_all_element(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let state = arg(args, 0);
    let idx = ab(i.to_number(&arg(args, 1)))? as usize;
    let already = arg(args, 2);
    let resolve_fn = arg(args, 3);
    let value = arg(args, 4);
    // [[AlreadyCalled]]: a second settlement of the same element is a no-op.
    if let Value::Obj(o) = &already {
        if matches!(
            o.borrow().props.get("#\u{0}called").map(|p| p.value()),
            Some(Value::Bool(true))
        ) {
            return Ok(Value::Undefined);
        }
        set_internal(o, "#\u{0}called", Value::Bool(true));
    }
    let results = ab(i.get_member(&state, "#\u{0}results"))?;
    // CreateDataProperty: a direct own data property, so Array.prototype index setters aren't invoked.
    if let Value::Obj(o) = &results {
        crate::value::set_data(o, &idx.to_string(), value);
    }
    let rem_v = ab(i.get_member(&state, "#\u{0}remaining"))?;
    let rem = ab(i.to_number(&rem_v))? - 1.0;
    ab(i.set_member(&state, "#\u{0}remaining", Value::Num(rem)))?;
    if rem == 0.0 {
        ab(i.call(resolve_fn, Value::Undefined, &[results]))?;
    }
    Ok(Value::Undefined)
}

/// Promise.allKeyed / Promise.allSettledKeyed (the await-dictionary proposal): iterate the
/// input object's enumerable own string keys ([[OwnPropertyKeys]] + [[GetOwnProperty]]), resolve
/// each value through C.resolve, and settle with a key-ordered plain result object.
fn promise_keyed_combinator(
    i: &mut Interp,
    t: Value,
    input: Value,
    settled: bool,
) -> Result<Value, Value> {
    let (result, resolve_fn, reject_fn) = new_promise_capability_full(i, &t)?;
    macro_rules! reject_with {
        ($e:expr) => {{
            let e = $e;
            let _ = i.call(reject_fn.clone(), Value::Undefined, &[e]);
            return Ok(result);
        }};
    }
    let promise_resolve = match get_promise_resolve(i, &t) {
        Ok(r) => r,
        Err(e) => reject_with!(e),
    };
    if !matches!(input, Value::Obj(_)) {
        reject_with!(i.make_error("TypeError", "argument must be an object"));
    }
    // allKeys = ? input.[[OwnPropertyKeys]](); keep enumerable string keys (GetOwnProperty each).
    let keys: Vec<String> = {
        let attempt = (|i: &mut Interp| -> Result<Vec<String>, Value> {
            let mut out = Vec::new();
            if let Some((target, handler)) = proxy_pair(i, &input) {
                for k in proxy_own_keys(i, &target, &handler)? {
                    let Value::Str(ks) = &k else { continue };
                    let desc = proxy_gopd_value(i, &target, &handler, ks)?;
                    if matches!(desc, Value::Obj(_)) {
                        let en = ab(i.get_member(&desc, "enumerable"))?;
                        if i.to_boolean(&en) {
                            out.push(ks.to_string());
                        }
                    }
                }
            } else if let Value::Obj(o) = &input {
                // Enumerable own keys — strings AND symbols, in [[OwnPropertyKeys]] order.
                let mut strs: Vec<String> = Vec::new();
                let mut syms: Vec<String> = Vec::new();
                for k in ordinary_own_keys_ordered(i, o)? {
                    let indexed = ab(i.host_platform_enumerable(o, &k))?.unwrap_or(false);
                    let enumerable = indexed
                        || o.borrow()
                            .props
                            .get(&k)
                            .is_some_and(|property| property.enumerable());
                    if !enumerable {
                        continue;
                    }
                    if Interp::is_sym_key(&k) {
                        syms.push(k);
                    } else {
                        strs.push(k);
                    }
                }
                out.extend(strs);
                out.extend(syms);
            }
            Ok(out)
        })(i);
        match attempt {
            Ok(k) => k,
            Err(e) => reject_with!(e),
        }
    };
    let state = i.new_object();
    let values = i.make_array(vec![Value::Undefined; keys.len()]);
    let keys_arr = i.make_array(keys.iter().map(|k| Value::lstr(k.clone())).collect());
    set_internal(&state, "#\u{0}values", values);
    set_internal(&state, "#\u{0}keys", keys_arr);
    set_internal(&state, "#\u{0}remaining", Value::Num(1.0));
    for (idx, key) in keys.iter().enumerate() {
        let value = match ab(i.get_member(&input, key)) {
            Ok(v) => v,
            Err(e) => reject_with!(e),
        };
        let rem_v = ab(i.get_member(&Value::Obj(state.clone()), "#\u{0}remaining"))?;
        let rem = ab(i.to_number(&rem_v))?;
        set_internal(&state, "#\u{0}remaining", Value::Num(rem + 1.0));
        let p = match i.call(promise_resolve.clone(), t.clone(), &[value]) {
            Ok(p) => p,
            Err(e) => reject_with!(crate::interpreter::abrupt_value(e)),
        };
        let already = i.new_object();
        set_internal(&already, "#\u{0}called", Value::Bool(false));
        let mk = |i: &mut Interp, f: NativeFn| {
            make_bound(
                i,
                f,
                vec![
                    Value::Obj(state.clone()),
                    Value::Num(idx as f64),
                    Value::Obj(already.clone()),
                    resolve_fn.clone(),
                ],
            )
        };
        let (on_f, on_r) = if settled {
            (mk(i, promise_keyed_settle_f), mk(i, promise_keyed_settle_r))
        } else {
            (mk(i, promise_keyed_element), reject_fn.clone())
        };
        let then = match i.get_member(&p, "then") {
            Ok(t) => t,
            Err(e) => reject_with!(crate::interpreter::abrupt_value(e)),
        };
        if let Err(e) = i.call(then, p, &[on_f, on_r]) {
            reject_with!(crate::interpreter::abrupt_value(e));
        }
    }
    let rem_v = ab(i.get_member(&Value::Obj(state.clone()), "#\u{0}remaining"))?;
    let rem = ab(i.to_number(&rem_v))?;
    set_internal(&state, "#\u{0}remaining", Value::Num(rem - 1.0));
    if rem - 1.0 == 0.0 {
        let out = promise_keyed_result(i, &Value::Obj(state.clone()))?;
        capability_resolve_or_reject(i, resolve_fn, reject_fn, out);
    }
    Ok(result)
}

/// Assemble the keyed combinator's result: a plain object with the recorded values in key order.
fn promise_keyed_result(i: &mut Interp, state: &Value) -> Result<Value, Value> {
    let keys = ab(i.get_member(state, "#\u{0}keys"))?;
    let values = ab(i.get_member(state, "#\u{0}values"))?;
    let len = match ab(i.get_member(&keys, "length"))? {
        Value::Num(n) => n as usize,
        _ => 0,
    };
    let out = i.new_object();
    out.borrow_mut().proto = None; // the keyed result is a null-prototype object
    for k in 0..len {
        let kv = ab(i.get_member(&keys, &k.to_string()))?;
        let key = ab(i.to_string(&kv))?;
        let v = ab(i.get_member(&values, &k.to_string()))?;
        set_data(&out, &key, v);
    }
    Ok(Value::Obj(out))
}

/// Shared element bookkeeping for the keyed combinators. `payload` is what lands in the values
/// slot; the last settlement assembles and resolves the keyed result.
fn promise_keyed_record(i: &mut Interp, args: &[Value], payload: Value) -> Result<Value, Value> {
    let state = arg(args, 0);
    let idx = ab(i.to_number(&arg(args, 1)))? as usize;
    let already = arg(args, 2);
    let resolve_fn = arg(args, 3);
    if let Value::Obj(o) = &already {
        if matches!(
            o.borrow().props.get("#\u{0}called").map(|p| p.value()),
            Some(Value::Bool(true))
        ) {
            return Ok(Value::Undefined);
        }
        set_internal(o, "#\u{0}called", Value::Bool(true));
    }
    let values = ab(i.get_member(&state, "#\u{0}values"))?;
    if let Value::Obj(o) = &values {
        crate::value::set_data(o, &idx.to_string(), payload);
    }
    let rem_v = ab(i.get_member(&state, "#\u{0}remaining"))?;
    let rem = ab(i.to_number(&rem_v))? - 1.0;
    ab(i.set_member(&state, "#\u{0}remaining", Value::Num(rem)))?;
    if rem == 0.0 {
        let out = promise_keyed_result(i, &state)?;
        ab(i.call(resolve_fn, Value::Undefined, &[out]))?;
    }
    Ok(Value::Undefined)
}

fn promise_keyed_element(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    let payload = arg(args, 4);
    promise_keyed_record(i, args, payload)
}

fn promise_keyed_settle(i: &mut Interp, args: &[Value], fulfilled: bool) -> Result<Value, Value> {
    let value = arg(args, 4);
    let status = i.new_object();
    set_data(
        &status,
        "status",
        Value::lstr(if fulfilled { "fulfilled" } else { "rejected" }),
    );
    set_data(&status, if fulfilled { "value" } else { "reason" }, value);
    promise_keyed_record(i, args, Value::Obj(status))
}

fn promise_keyed_settle_f(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    promise_keyed_settle(i, a, true)
}
fn promise_keyed_settle_r(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    promise_keyed_settle(i, a, false)
}

fn promise_settled(i: &mut Interp, args: &[Value], fulfilled: bool) -> Result<Value, Value> {
    let state = arg(args, 0);
    let idx = ab(i.to_number(&arg(args, 1)))? as usize;
    let already = arg(args, 2);
    let resolve_fn = arg(args, 3);
    let value = arg(args, 4);
    // The fulfill and reject functions for one index share this [[AlreadyCalled]] record.
    if let Value::Obj(o) = &already {
        if matches!(
            o.borrow().props.get("#\u{0}called").map(|p| p.value()),
            Some(Value::Bool(true))
        ) {
            return Ok(Value::Undefined);
        }
        set_internal(o, "#\u{0}called", Value::Bool(true));
    }
    let status = i.new_object();
    set_data(
        &status,
        "status",
        Value::lstr(if fulfilled { "fulfilled" } else { "rejected" }),
    );
    set_data(&status, if fulfilled { "value" } else { "reason" }, value);
    let results = ab(i.get_member(&state, "#\u{0}results"))?;
    if let Value::Obj(o) = &results {
        crate::value::set_data(o, &idx.to_string(), Value::Obj(status));
    }
    let rem_v = ab(i.get_member(&state, "#\u{0}remaining"))?;
    let rem = ab(i.to_number(&rem_v))? - 1.0;
    ab(i.set_member(&state, "#\u{0}remaining", Value::Num(rem)))?;
    if rem == 0.0 {
        ab(i.call(resolve_fn, Value::Undefined, &[results]))?;
    }
    Ok(Value::Undefined)
}
fn promise_settled_fulfill(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    promise_settled(i, a, true)
}
fn promise_settled_reject(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    promise_settled(i, a, false)
}
fn promise_any_reject(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    let state = arg(a, 0);
    let idx = ab(i.to_number(&arg(a, 1)))? as usize;
    let already = arg(a, 2);
    let reject_fn = arg(a, 3);
    let reason = arg(a, 4);
    if let Value::Obj(o) = &already {
        if matches!(
            o.borrow().props.get("#\u{0}called").map(|p| p.value()),
            Some(Value::Bool(true))
        ) {
            return Ok(Value::Undefined);
        }
        set_internal(o, "#\u{0}called", Value::Bool(true));
    }
    let errors = ab(i.get_member(&state, "#\u{0}errors"))?;
    ab(i.set_member(&errors, &idx.to_string(), reason))?;
    let rem_v = ab(i.get_member(&state, "#\u{0}remaining"))?;
    let rem = ab(i.to_number(&rem_v))? - 1.0;
    ab(i.set_member(&state, "#\u{0}remaining", Value::Num(rem)))?;
    if rem == 0.0 {
        let agg = make_aggregate_error(i, errors)?;
        ab(i.call(reject_fn, Value::Undefined, &[agg]))?;
    }
    Ok(Value::Undefined)
}
/// Object.assign uses Set, unlike spread's CreateDataProperty. A fresh ordinary target can
/// reuse the data-record copy only after proving that no prototype supplies any copied key.
/// This live walk keeps inherited setters (including __proto__) and non-writable properties
/// on the ordered checked path. No getter, trap or conversion runs while a borrow is held.
fn try_assign_named_data(target: &Value, source: &Value) -> bool {
    let (Value::Obj(target), Value::Obj(source)) = (target, source) else {
        return false;
    };
    if Rc::ptr_eq(target, source) {
        return false;
    }
    let source = source.borrow();
    if !source.ic_plain.get() || !matches!(source.exotic, Exotic::None) {
        return false;
    }
    let Some(keys) = source.props.named_data_keys() else {
        return false;
    };
    if keys.is_empty() {
        return true;
    }
    let mut target = target.borrow_mut();
    if !target.ic_plain.get()
        || !matches!(target.exotic, Exotic::None)
        || !target.extensible
        || !target
            .props
            .named_data_keys()
            .is_some_and(|keys| keys.is_empty())
    {
        return false;
    }
    let mut prototype = target.proto.clone();
    while let Some(parent) = prototype {
        let parent = parent.borrow();
        if !parent.ic_plain.get()
            || !matches!(parent.exotic, Exotic::None)
            || keys.iter().any(|key| parent.props.contains(key))
        {
            return false;
        }
        prototype = parent.proto.clone();
    }
    target.props.try_copy_named_data_from(&source.props)
}

fn install_object(it: &mut Interp) {
    let op = it.object_proto.clone();
    it.def_method(&op, "hasOwnProperty", 1, |i, this, args| {
        // ToPropertyKey of a String or an integral Number cannot run author code or fail, so
        // those keys skip building an owned key string; ToObject still follows it.
        let key = arg(args, 0);
        if let Value::Str(key) = &key {
            let o = to_object_arg(i, this, "Object.prototype.hasOwnProperty")?;
            return has_own_property_trapped(i, &Value::Obj(o), key.as_str()).map(Value::Bool);
        }
        if let Some(text) = key.as_num_opt().and_then(crate::value::KeyText::integer) {
            let o = to_object_arg(i, this, "Object.prototype.hasOwnProperty")?;
            return has_own_property_trapped(i, &Value::Obj(o), text.as_str()).map(Value::Bool);
        }
        let key = ab(i.to_property_key(&key))?;
        let o = to_object_arg(i, this, "Object.prototype.hasOwnProperty")?;
        has_own_property_trapped(i, &Value::Obj(o), &key).map(Value::Bool)
    });
    // Annex B __defineGetter__/__defineSetter__/__lookupGetter__/__lookupSetter__.
    fn define_accessor(
        i: &mut Interp,
        this: &Value,
        args: &[Value],
        is_get: bool,
    ) -> Result<Value, Value> {
        let name = if is_get {
            "__defineGetter__"
        } else {
            "__defineSetter__"
        };
        let o = to_object_arg(i, this.clone(), name)?;
        let f = arg(args, 1);
        if !f.is_callable() {
            return Err(i.make_error("TypeError", "accessor must be a function"));
        }
        // DefinePropertyOrThrow with { get/set: f, enumerable: true, configurable: true }.
        let desc = i.new_object();
        set_data(&desc, if is_get { "get" } else { "set" }, f);
        set_data(&desc, "enumerable", Value::Bool(true));
        set_data(&desc, "configurable", Value::Bool(true));
        let key = ab(i.to_property_key(&arg(args, 0)))?;
        let ov = Value::Obj(o.clone());
        let ok = if let Some((t, h)) = proxy_pair(i, &ov) {
            ab(proxy_define_property(i, &t, &h, &key, &Value::Obj(desc)))?
        } else {
            ab(define_own_property(i, &o, &key, &Value::Obj(desc)))?
        };
        if !ok {
            return Err(i.make_error("TypeError", format!("cannot define accessor '{key}'")));
        }
        Ok(Value::Undefined)
    }
    fn lookup_accessor(
        i: &mut Interp,
        this: &Value,
        args: &[Value],
        is_get: bool,
    ) -> Result<Value, Value> {
        let name = if is_get {
            "__lookupGetter__"
        } else {
            "__lookupSetter__"
        };
        let o = to_object_arg(i, this.clone(), name)?;
        let key = ab(i.to_property_key(&arg(args, 0)))?;
        let mut cur = Value::Obj(o);
        loop {
            // [[GetOwnProperty]] goes through a proxy's trap, propagating abrupt completions.
            if let Some((t, h)) = proxy_pair(i, &cur) {
                let desc = proxy_gopd_value(i, &t, &h, &key)?;
                if let Value::Obj(d) = &desc {
                    let is_accessor =
                        d.borrow().props.contains("get") || d.borrow().props.contains("set");
                    if !is_accessor {
                        return Ok(Value::Undefined);
                    }
                    return ab(i.get_member(&desc, if is_get { "get" } else { "set" }));
                }
            } else if let Value::Obj(co) = &cur {
                if let Some(p) = co.borrow().props.get(&key) {
                    if p.accessor() {
                        let f = if is_get {
                            p.getter().cloned()
                        } else {
                            p.setter().cloned()
                        };
                        return Ok(f.unwrap_or(Value::Undefined));
                    }
                    return Ok(Value::Undefined);
                }
            }
            cur = js_get_prototype_of(i, &cur)?;
            if !matches!(cur, Value::Obj(_)) {
                return Ok(Value::Undefined);
            }
        }
    }
    it.def_method(&op, "__defineGetter__", 2, |i, this, a| {
        define_accessor(i, &this, a, true)
    });
    it.def_method(&op, "__defineSetter__", 2, |i, this, a| {
        define_accessor(i, &this, a, false)
    });
    it.def_method(&op, "__lookupGetter__", 1, |i, this, a| {
        lookup_accessor(i, &this, a, true)
    });
    it.def_method(&op, "__lookupSetter__", 1, |i, this, a| {
        lookup_accessor(i, &this, a, false)
    });
    it.def_method(&op, "isPrototypeOf", 1, |i, this, args| {
        let v = arg(args, 0);
        if !matches!(v, Value::Obj(_)) {
            return Ok(Value::Bool(false));
        }
        let me = to_object_arg(i, this, "Object.prototype.isPrototypeOf")?;
        let mut cur = js_get_prototype_of(i, &v)?;
        loop {
            match &cur {
                Value::Obj(o) if Rc::ptr_eq(o, &me) => return Ok(Value::Bool(true)),
                Value::Obj(_) => {}
                _ => return Ok(Value::Bool(false)),
            }
            cur = js_get_prototype_of(i, &cur)?;
        }
    });
    it.def_method(&op, "propertyIsEnumerable", 1, |i, this, args| {
        let key = ab(i.to_property_key(&arg(args, 0)))?;
        let o = to_object_arg(i, this, "Object.prototype.propertyIsEnumerable")?;
        let ptr = Rc::as_ptr(&o) as usize;
        if i.is_namespace(ptr) {
            if let Some(res) = i.namespace_own_property(ptr, &key) {
                ab(res)?;
                return Ok(Value::Bool(true));
            }
        }
        // A proxy reads [[GetOwnProperty]]'s enumerable flag through its trap.
        if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
            let desc = proxy_gopd_value(i, &target, &handler, &key)?;
            let e = match desc {
                Value::Undefined => false,
                d => {
                    let ev = ab(i.get_member(&d, "enumerable"))?;
                    i.to_boolean(&ev)
                }
            };
            return Ok(Value::Bool(e));
        }
        if ab(i.host_indexed_own_value(&o, &key))?.is_some() {
            return Ok(Value::Bool(crate::value::canonical_index(&key).is_some()));
        }
        let e = o
            .borrow()
            .props
            .get(&key)
            .map(|p| p.enumerable())
            .unwrap_or(false);
        Ok(Value::Bool(e))
    });
    it.def_method(&op, "toString", 0, |i, this, _args| {
        if matches!(this, Value::Undefined) {
            return Ok(Value::lstr("[object Undefined]"));
        }
        if matches!(this, Value::Null) {
            return Ok(Value::lstr("[object Null]"));
        }
        // IsArray pierces proxies (a revoked proxy throws).
        let builtin = if json_is_array(i, &this)? {
            "Array"
        } else {
            builtin_tag(i, &this)
        };
        // Object.prototype.toString's final step is the concatenation of three strings
        // (ECMA-262 §20.1.3.6). Keep a String @@toStringTag borrowed through the one result
        // allocation instead of copying it into a temporary Rust `String` first.
        match to_string_tag_key(i) {
            Some(key) => match ab(i.get_member(&this, &key))? {
                Value::Str(s) => Ok(Value::Str(crate::lstr::LStr::concat3(
                    "[object ",
                    s.as_str(),
                    "]",
                ))),
                _ => Ok(Value::Str(crate::lstr::LStr::concat3(
                    "[object ", builtin, "]",
                ))),
            },
            None => Ok(Value::Str(crate::lstr::LStr::concat3(
                "[object ", builtin, "]",
            ))),
        }
    });
    it.def_method(&op, "valueOf", 0, |i, this, _args| {
        to_object_arg(i, this, "Object.prototype.valueOf").map(Value::Obj)
    });
    // Object.prototype.toLocaleString(): the default just invokes `this.toString()`.
    it.def_method(&op, "toLocaleString", 0, |i, this, _args| {
        let to_string = ab(i.get_member(&this, "toString"))?;
        if !to_string.is_callable() {
            return Err(i.make_error("TypeError", "toString is not callable"));
        }
        ab(i.call(to_string, this, &[]))
    });
    // Object.prototype.__proto__ — an accessor (Annex B) over [[GetPrototypeOf]]/[[SetPrototypeOf]].
    {
        let getter = it.make_native("get __proto__", 0, |i, this, _| {
            // RequireObjectCoercible, then ToObject before reading the prototype.
            let o = to_object_arg(i, this, "get __proto__")?;
            js_get_prototype_of(i, &Value::Obj(o))
        });
        let setter = it.make_native("set __proto__", 1, |i, this, args| {
            if matches!(this, Value::Undefined | Value::Null) {
                return Err(i.make_error("TypeError", "__proto__ set on null or undefined"));
            }
            let v = arg(args, 0);
            // Only an Object receiver with an Object/Null value actually changes the prototype;
            // everything else is a silent no-op.
            if matches!(&this, Value::Obj(_))
                && matches!(v, Value::Obj(_) | Value::Null)
                && !js_set_prototype_of(i, &this, &v)?
            {
                return Err(i.make_error("TypeError", "cyclic __proto__ value"));
            }
            Ok(Value::Undefined)
        });
        op.borrow_mut().props.insert(
            "__proto__",
            Property::accessor_prop(
                Some(Value::Obj(getter)),
                Some(Value::Obj(setter)),
                false,
                true,
            ),
        );
    }

    let ctor = it.make_native("Object", 1, |i, _this, args| {
        // `new Object()` with a newTarget other than %Object% itself (a subclass or a
        // Reflect.construct target): OrdinaryCreateFromConstructor(newTarget, %Object.prototype%).
        if i.constructing {
            if let Value::Obj(nt) = i.new_target.clone() {
                let is_self = matches!(
                    i.extra_protos.get("%ObjectCtor%"),
                    Some(c) if Rc::ptr_eq(c, &nt)
                );
                if !is_self {
                    let proto = match ab(i.get_member(&Value::Obj(nt.clone()), "prototype"))? {
                        Value::Obj(p) => Some(p),
                        _ => ctor_realm_proto(i, &Value::Obj(nt), "Object")?
                            .or_else(|| Some(i.object_proto.clone())),
                    };
                    return Ok(Value::Obj(Object::new(proto)));
                }
            }
        }
        Ok(match arg(args, 0) {
            Value::Undefined | Value::Null => Value::Obj(i.new_object()),
            Value::Obj(o) => Value::Obj(o),
            // ToObject of a primitive yields its wrapper object.
            other => box_primitive(i, other),
        })
    });
    Rc::make_mut(&mut it.extra_protos).insert("%ObjectCtor%", ctor.clone());
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(op.clone()), false, false, false),
    );
    op.borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));

    it.def_method(&ctor, "hasOwn", 2, nf_object_has_own);
    it.def_method(&ctor, "groupBy", 2, |i, _this, args| {
        let cb = arg(args, 1);
        if !cb.is_callable() {
            return Err(i.make_error("TypeError", "Object.groupBy callback is not callable"));
        }
        let elems = ab(i.iterate(&arg(args, 0)))?;
        let mut groups: Vec<(crate::value::PropertyKey, Vec<Value>)> = Vec::new();
        for (idx, el) in elems.into_iter().enumerate() {
            let key_v = ab(i.call(
                cb.clone(),
                Value::Undefined,
                &[el.clone(), Value::Num(idx as f64)],
            ))?;
            let key = ab(i.to_property_key(&key_v))?;
            match groups.iter_mut().find(|(k, _)| k.as_str() == key.as_str()) {
                Some(g) => g.1.push(el),
                None => groups.push((key, vec![el])),
            }
        }
        let result = i.new_object();
        result.borrow_mut().proto = None; // groupBy returns a null-prototype object
        for (k, v) in groups {
            let arr = i.make_array(v);
            result
                .borrow_mut()
                .props
                .insert(k.as_str(), Property::plain(arr));
        }
        Ok(Value::Obj(result))
    });
    it.def_method(&ctor, "keys", 1, |i, _this, args| {
        let o = to_object_arg(i, arg(args, 0), "Object.keys")?;
        ab(i.defer_trigger(&o, None))?;
        if proxy_pair(i, &Value::Obj(o.clone())).is_some() {
            let keys = proxy_enum_string_keys(i, &Value::Obj(o.clone()))?;
            return Ok(i.make_array(keys));
        }
        if let Some(length) = i.host_indexed_len(&o) {
            let own_keys = ordinary_own_keys_ordered(i, &o)?;
            let mut keys = Vec::with_capacity(own_keys.len().max(length as usize));
            // EnumerableOwnProperties asks [[GetOwnProperty]] for each key. For a Web IDL
            // indexed property that operation invokes the platform getter even though
            // Object.keys ultimately keeps only the key.
            for key in own_keys
                .into_iter()
                .filter(|key| !Interp::is_sym_key(key) && !Interp::is_private_key(key))
            {
                let enumerable = if ab(i.host_indexed_own_value(&o, &key))?.is_some() {
                    crate::value::canonical_index(&key).is_some()
                } else {
                    o.borrow()
                        .props
                        .get(&key)
                        .is_some_and(|property| property.enumerable())
                };
                if enumerable {
                    keys.push(Value::lstr(key));
                }
            }
            return Ok(i.make_array(keys));
        }
        // A TypedArray's enumerable own keys are its integer indices plus string expandos.
        if let Some(info) = ta_info(i, &o) {
            let n = i.ta_len(&info).unwrap_or(0);
            let mut keys: Vec<Value> = (0..n).map(|k| Value::lstr(k.to_string())).collect();
            for k in ordered_enum_keys(&o) {
                if k.parse::<usize>().is_err() && !TA_META_KEYS.contains(&&*k) {
                    keys.push(Value::Str(k.into()));
                }
            }
            return Ok(i.make_array(keys));
        }
        let names = ordered_enum_keys(&o);
        // A module namespace's Object.keys reads each binding's [[GetOwnProperty]], throwing for an
        // uninitialized export.
        let ptr = Rc::as_ptr(&o) as usize;
        if i.is_namespace(ptr) {
            for k in &names {
                if let Some(res) = i.namespace_own_property(ptr, k) {
                    ab(res)?;
                }
            }
        }
        let keys: Vec<Value> = names.into_iter().map(|k| Value::Str(k.into())).collect();
        Ok(i.make_array(keys))
    });
    it.def_method(&ctor, "getOwnPropertyNames", 1, |i, _this, args| {
        let o = to_object_arg(i, arg(args, 0), "Object.getOwnPropertyNames")?;
        ab(i.defer_trigger(&o, None))?;
        if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
            let keys = proxy_own_keys(i, &target, &handler)?;
            let strs: Vec<Value> = keys
                .into_iter()
                .filter(|k| matches!(k, Value::Str(_)))
                .collect();
            return Ok(i.make_array(strs));
        }
        // A TypedArray's own keys are its integer indices, then any string expandos (the
        // length/buffer/... metadata are inherited, not own).
        if let Some(info) = ta_info(i, &o) {
            let n = i.ta_len(&info).unwrap_or(0);
            let mut keys: Vec<Value> = (0..n).map(|k| Value::lstr(k.to_string())).collect();
            for k in o.borrow().props.keys() {
                if !Interp::is_sym_key(&k)
                    && k.parse::<usize>().is_err()
                    && !TA_META_KEYS.contains(&&*k)
                {
                    keys.push(Value::Str(k.into()));
                }
            }
            return Ok(i.make_array(keys));
        }
        // Spec order: array-index keys ascending, then other string keys in insertion order.
        let keys: Vec<Value> = ordinary_own_keys_ordered(i, &o)?
            .into_iter()
            .filter(|key| !Interp::is_sym_key(key) && !Interp::is_private_key(key))
            .map(Value::lstr)
            .collect();
        Ok(i.make_array(keys))
    });
    it.def_method(&ctor, "getOwnPropertySymbols", 1, |i, _this, args| {
        // ToObject coerces primitives (and throws for null/undefined).
        let o = to_object_arg(i, arg(args, 0), "Object.getOwnPropertySymbols")?;
        ab(i.defer_trigger(&o, None))?;
        let ov = Value::Obj(o.clone());
        // A proxy's symbol keys come from its [[OwnPropertyKeys]] trap.
        if let Some((target, handler)) = proxy_pair(i, &ov) {
            let syms: Vec<Value> = proxy_own_keys(i, &target, &handler)?
                .into_iter()
                .filter(|k| matches!(k, Value::Sym(_)))
                .collect();
            return Ok(i.make_array(syms));
        }
        let syms: Vec<Value> = ordinary_own_keys_ordered(i, &o)?
            .into_iter()
            .filter(|k| Interp::is_sym_key(k))
            .filter_map(|k| i.sym_from_key(&k))
            .collect();
        Ok(i.make_array(syms))
    });
    it.def_method(&ctor, "getPrototypeOf", 1, |i, _this, args| {
        match arg(args, 0) {
            Value::Obj(o) => {
                if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
                    return proxy_get_prototype(i, &target, &handler);
                }
                Ok(o.borrow()
                    .proto
                    .clone()
                    .map(Value::Obj)
                    .unwrap_or(Value::Null))
            }
            // ToObject coerces a primitive (Object.getPrototypeOf('') → String.prototype).
            Value::Undefined | Value::Empty | Value::Null => {
                Err(i.make_error("TypeError", "called on null or undefined"))
            }
            Value::Str(_) => Ok(Value::Obj(i.string_proto.clone())),
            Value::Num(_) => Ok(Value::Obj(i.number_proto.clone())),
            Value::Bool(_) => Ok(Value::Obj(i.boolean_proto.clone())),
            Value::Sym(_) => Ok(Value::Obj(i.symbol_proto.clone())),
            Value::BigInt(_) => Ok(i
                .extra_protos
                .get("BigInt")
                .cloned()
                .map(Value::Obj)
                .unwrap_or(Value::Null)),
        }
    });
    it.def_method(&ctor, "setPrototypeOf", 2, |i, _this, args| {
        let proto = arg(args, 1);
        if !matches!(proto, Value::Obj(_) | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "Object prototype may only be an Object or null",
            ));
        }
        if matches!(arg(args, 0), Value::Undefined | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "Object.setPrototypeOf called on null or undefined",
            ));
        }
        let obj = arg(args, 0);
        // Object.setPrototypeOf throws if [[SetPrototypeOf]] returns false.
        if !js_set_prototype_of(i, &obj, &proto)? {
            return Err(i.make_error("TypeError", "could not set prototype"));
        }
        Ok(obj)
    });
    it.def_method(&ctor, "create", 2, |i, _this, args| {
        let proto = match arg(args, 0) {
            Value::Obj(o) => Some(o),
            Value::Null => None,
            _ => {
                return Err(i.make_error("TypeError", "Object.create proto must be object or null"));
            }
        };
        let obj = Object::new(proto);
        // Object.create(O, Properties): when Properties is not undefined, ObjectDefineProperties.
        let props_arg = arg(args, 1);
        if !matches!(props_arg, Value::Undefined) {
            object_define_properties(i, &Value::Obj(obj.clone()), props_arg, "Object.create")?;
        }
        Ok(Value::Obj(obj))
    });
    it.def_method(&ctor, "defineProperty", 3, |i, _this, args| {
        let o = match arg(args, 0) {
            Value::Obj(o) => o,
            _ => {
                return Err(i.make_error("TypeError", "Object.defineProperty called on non-object"));
            }
        };
        let key = ab(i.to_property_key(&arg(args, 1)))?;
        if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
            if !ab(proxy_define_property(
                i,
                &target,
                &handler,
                &key,
                &arg(args, 2),
            ))? {
                return Err(
                    i.make_error("TypeError", "proxy defineProperty returned a falsish value")
                );
            }
            return Ok(Value::Obj(o));
        }
        if !ab(define_own_property(i, &o, &key, &arg(args, 2)))? {
            return Err(i.make_error("TypeError", "Cannot redefine property"));
        }
        Ok(Value::Obj(o))
    });
    it.def_method(&ctor, "getOwnPropertyDescriptor", 2, |i, _this, args| {
        // ECMA-262 §20.1.2.8: ToObject coerces a primitive target (throws for null/undefined),
        // then ToPropertyKey, then obj.[[GetOwnProperty]](key) and FromPropertyDescriptor.
        let o = to_object_arg(i, arg(args, 0), "Object.getOwnPropertyDescriptor")?;
        let key = ab(i.to_property_key(&arg(args, 1)))?;
        Ok(match object_get_own_property(i, &o, &key)? {
            Some(property) => descriptor_from_prop(i, property),
            None => Value::Undefined,
        })
    });
    it.def_method(&ctor, "getOwnPropertyDescriptors", 1, |i, _this, args| {
        // ECMA-262 §20.1.2.9: [[OwnPropertyKeys]], then [[GetOwnProperty]] per key in order.
        let o = to_object_arg(i, arg(args, 0), "Object.getOwnPropertyDescriptors")?;
        let keys = object_own_property_keys(i, &o)?;
        let result = i.new_object();
        for k in keys {
            let key = ab(i.to_property_key(&k))?;
            if let Some(property) = object_get_own_property(i, &o, &key)? {
                let descriptor = descriptor_from_prop(i, property);
                result
                    .borrow_mut()
                    .props
                    .insert(key.as_str(), Property::plain(descriptor));
            }
        }
        Ok(Value::Obj(result))
    });
    it.def_method(&ctor, "freeze", 1, |i, _this, args| {
        let o = arg(args, 0);
        if matches!(o, Value::Obj(_)) && !set_integrity_level(i, &o, true)? {
            return Err(i.make_error("TypeError", "Object.freeze could not freeze the object"));
        }
        Ok(o)
    });
    it.def_method(&ctor, "preventExtensions", 1, |i, _this, args| {
        let obj = arg(args, 0);
        if matches!(obj, Value::Obj(_)) && !js_prevent_extensions(i, &obj)? {
            return Err(i.make_error("TypeError", "could not prevent extensions"));
        }
        Ok(obj)
    });
    it.def_method(&ctor, "isExtensible", 1, |i, _this, args| {
        let obj = arg(args, 0);
        // Object.isExtensible returns false for a non-object (ES2015+).
        if !matches!(obj, Value::Obj(_)) {
            return Ok(Value::Bool(false));
        }
        Ok(Value::Bool(js_is_extensible(i, &obj)?))
    });
    it.def_method(&ctor, "assign", 2, |i, _this, args| {
        // ToObject(target) — throws a TypeError for null/undefined.
        let to = to_object_arg(i, arg(args, 0), "Object.assign")?;
        let to_val = Value::Obj(to);
        for src in args.iter().skip(1) {
            if matches!(src, Value::Undefined | Value::Null) {
                continue;
            }
            let from = Value::Obj(to_object_arg(i, src.clone(), "Object.assign")?);
            if try_assign_named_data(&to_val, &from) {
                continue;
            }
            if let Some((t, h)) = proxy_pair(i, &from) {
                // Proxy source: enumerate all own keys (string + symbol) via the traps, copying each
                // enumerable one.
                for key in proxy_own_keys(i, &t, &h)? {
                    let pk = ab(i.to_property_key(&key))?;
                    if proxy_key_enumerable(i, &t, &h, &pk)? {
                        let v = ab(i.get_member(&from, &pk))?;
                        assign_set(i, &to_val, &pk, v)?;
                    }
                }
                continue;
            }
            let o = from.as_obj().unwrap();
            let keys = ordinary_own_keys_ordered(i, o)?;
            for k in keys {
                if Interp::is_private_key(&k) {
                    continue;
                }
                let enumerable = if ab(i.host_indexed_own_value(o, &k))?.is_some() {
                    crate::value::canonical_index(&k).is_some()
                } else {
                    o.borrow()
                        .props
                        .get(&k)
                        .is_some_and(|property| property.enumerable())
                };
                if enumerable {
                    let v = ab(i.get_member(&from, &k))?;
                    assign_set(i, &to_val, &k, v)?;
                }
            }
        }
        Ok(to_val)
    });
    it.def_method(&ctor, "is", 2, |_i, _this, args| {
        Ok(Value::Bool(same_value(&arg(args, 0), &arg(args, 1))))
    });
    it.def_method(&ctor, "values", 1, |i, _this, args| {
        let o = to_object_arg(i, arg(args, 0), "Object.values")?;
        ab(i.defer_trigger(&o, None))?;
        enumerable_own_value_list(i, &Value::Obj(o), false)
    });
    it.def_method(&ctor, "entries", 1, |i, _this, args| {
        let o = to_object_arg(i, arg(args, 0), "Object.entries")?;
        ab(i.defer_trigger(&o, None))?;
        enumerable_own_value_list(i, &Value::Obj(o), true)
    });
    it.def_method(&ctor, "fromEntries", 1, |i, _this, args| {
        // RequireObjectCoercible(iterable).
        let iterable = arg(args, 0);
        if matches!(iterable, Value::Undefined | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "Object.fromEntries called on null or undefined",
            ));
        }
        let obj = i.new_object();
        // Lazily step the iterator; an abrupt completion while processing an entry closes it.
        let (iter, next) = ab(i.get_iterator(&iterable))?;
        loop {
            let entry = match ab(i.iterator_step(&iter, &next))? {
                Some(v) => v,
                None => break,
            };
            let processed = (|i: &mut Interp| -> Result<(), Value> {
                // Each entry must be an Object; keys/values are added via CreateDataPropertyOnObject,
                // which defines a plain data property and never invokes inherited setters.
                if !matches!(entry, Value::Obj(_)) {
                    return Err(
                        i.make_error("TypeError", "Object.fromEntries entry is not an object")
                    );
                }
                let k = ab(i.get_member(&entry, "0"))?;
                let v = ab(i.get_member(&entry, "1"))?;
                let key = ab(i.to_property_key(&k))?;
                obj.borrow_mut()
                    .props
                    .insert(key.as_str(), Property::data(v, true, true, true));
                Ok(())
            })(i);
            if let Err(e) = processed {
                i.iterator_close(&iter);
                return Err(e);
            }
        }
        Ok(Value::Obj(obj))
    });
    it.def_method(&ctor, "defineProperties", 2, |i, _this, args| {
        let o = arg(args, 0);
        if !matches!(o, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Object.defineProperties on non-object"));
        }
        object_define_properties(i, &o, arg(args, 1), "Object.defineProperties")?;
        Ok(o)
    });
    it.def_method(&ctor, "seal", 1, |i, _this, args| {
        let o = arg(args, 0);
        if matches!(o, Value::Obj(_)) && !set_integrity_level(i, &o, false)? {
            return Err(i.make_error("TypeError", "Object.seal could not seal the object"));
        }
        Ok(o)
    });
    it.def_method(&ctor, "isSealed", 1, |i, _this, args| {
        Ok(Value::Bool(test_integrity_level(i, &arg(args, 0), false)?))
    });
    it.def_method(&ctor, "isFrozen", 1, |i, _this, args| {
        Ok(Value::Bool(test_integrity_level(i, &arg(args, 0), true)?))
    });

    set_builtin(&it.global, "Object", Value::Obj(ctor));
}

/// Named entry so the bytecode call cache can recognize and specialize `Object.hasOwn`.
pub(crate) fn nf_object_has_own(
    i: &mut Interp,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    // Object.hasOwn orders ToObject before ToPropertyKey (unlike hasOwnProperty).
    // https://tc39.es/ecma262/multipage/fundamental-objects.html#sec-object.hasown
    let o = to_object_arg(i, arg(args, 0), "Object.hasOwn")?;
    let key = ab(i.to_property_key(&arg(args, 1)))?;
    has_own_property_trapped(i, &Value::Obj(o), &key).map(Value::Bool)
}

/// TestIntegrityLevel: extensibility plus per-key configurability (and, for frozen, data-property
/// writability), through the proxy [[IsExtensible]]/[[OwnPropertyKeys]]/[[GetOwnProperty]] traps.
fn test_integrity_level(i: &mut Interp, v: &Value, frozen: bool) -> Result<bool, Value> {
    if !matches!(v, Value::Obj(_)) {
        return Ok(true);
    }
    if js_is_extensible(i, v)? {
        return Ok(false);
    }
    // A TypedArray's in-bounds elements are always configurable and writable, so a view with any
    // elements is never sealed or frozen.
    if let Value::Obj(o) = v {
        if let Some(info) = ta_info(i, o) {
            if i.ta_len(&info).unwrap_or(0) > 0 {
                return Ok(false);
            }
        }
    }
    if let Some((t, h)) = proxy_pair(i, v) {
        for k in proxy_own_keys(i, &t, &h)? {
            let key = ab(i.to_property_key(&k))?;
            let desc = proxy_gopd_value(i, &t, &h, &key)?;
            let d = match &desc {
                Value::Obj(d) => d.clone(),
                _ => continue,
            };
            let c = ab(i.get_member(&desc, "configurable"))?;
            if i.to_boolean(&c) {
                return Ok(false);
            }
            if frozen && d.borrow().props.contains("value") {
                let w = ab(i.get_member(&desc, "writable"))?;
                if i.to_boolean(&w) {
                    return Ok(false);
                }
            }
        }
        return Ok(true);
    }
    let o = v.as_obj().unwrap();
    // Private fields are invisible to integrity levels (they stay mutable on a frozen object).
    let ok = o.borrow().props.integrity_ok(frozen);
    Ok(ok)
}

/// Build a property descriptor from a JS descriptor object.
/// Build a descriptor object (`{value, writable, enumerable, configurable}` or `{get, set, ...}`).
fn descriptor_from_prop(i: &mut Interp, mut p: Property) -> Value {
    // FromPropertyDescriptor (ECMA-262): an ordinary object of the current Realm with the
    // fields created by CreateDataPropertyOrThrow in this order. The two possible key
    // sequences are fixed, so the finished map is built once per Agent and instantiated.
    let accessor = p.accessor();
    let values = if accessor {
        [
            p.getter().cloned().unwrap_or(Value::Undefined),
            p.setter().cloned().unwrap_or(Value::Undefined),
            Value::Bool(p.enumerable()),
            Value::Bool(p.configurable()),
        ]
    } else {
        [
            p.take_value(),
            Value::Bool(p.writable()),
            Value::Bool(p.enumerable()),
            Value::Bool(p.configurable()),
        ]
    };
    let keys: [&str; 4] = if accessor {
        ["get", "set", "enumerable", "configurable"]
    } else {
        ["value", "writable", "enumerable", "configurable"]
    };
    let map = i.descriptor_maps[usize::from(accessor)].get_or_init(|| {
        let mut map = Props::new();
        for key in keys {
            map.insert(key, Property::plain(Value::Undefined));
        }
        map
    });
    let props = map.instantiate_plain(values.into_iter());
    Value::Obj(Object::new_with_parts(
        Some(i.object_proto.clone()),
        props,
        Exotic::None,
    ))
}

/// A property descriptor with only the explicitly-present fields populated.
#[derive(Default, Clone)]
struct PartialDesc {
    value: Option<Value>,
    get: Option<Value>,
    set: Option<Value>,
    writable: Option<bool>,
    enumerable: Option<bool>,
    configurable: Option<bool>,
}
/// CompletePropertyDescriptor: fill in default attributes for a partial descriptor.
fn complete_descriptor(pd: PartialDesc) -> Property {
    if pd.is_accessor() {
        Property::accessor_prop(
            Some(pd.get.unwrap_or(Value::Undefined)),
            Some(pd.set.unwrap_or(Value::Undefined)),
            pd.enumerable.unwrap_or(false),
            pd.configurable.unwrap_or(false),
        )
    } else {
        Property::data(
            pd.value.unwrap_or(Value::Undefined),
            pd.writable.unwrap_or(false),
            pd.enumerable.unwrap_or(false),
            pd.configurable.unwrap_or(false),
        )
    }
}
impl PartialDesc {
    fn is_accessor(&self) -> bool {
        self.get.is_some() || self.set.is_some()
    }
    fn is_data(&self) -> bool {
        self.value.is_some() || self.writable.is_some()
    }
}

/// ToPropertyDescriptor's six HasProperty/Get pairs, answered without running code when
/// every object on `o`'s prototype chain is ordinary (no Proxy, typed array, namespace,
/// Web IDL indexed or exotic behavior) and every field found is a data property: then each
/// HasProperty is the first own lookup along the chain and each Get returns that slot's value,
/// exactly what the generic reads observe. Returns the fields in the generic path's order,
/// or `None` to take that path (an accessor, an exotic object, or a borrowed one).
fn plain_descriptor_fields(o: &Gc) -> Option<[Option<Value>; 6]> {
    const FIELDS: [&str; 6] = [
        "enumerable",
        "configurable",
        "value",
        "writable",
        "get",
        "set",
    ];
    let mut fields: [Option<Value>; 6] = Default::default();
    let mut current = Some(o.clone());
    while let Some(object) = current {
        let body = object.try_borrow().ok()?;
        if !body.ic_plain.get() || !matches!(body.exotic, Exotic::None) {
            return None;
        }
        for (field, name) in fields.iter_mut().zip(FIELDS) {
            if field.is_none() {
                if let Some(property) = body.props.get(name) {
                    if property.accessor() {
                        return None;
                    }
                    *field = Some(property.value());
                }
            }
        }
        if fields.iter().all(Option::is_some) {
            break;
        }
        current = body.proto.clone();
    }
    Some(fields)
}

/// Read + validate a descriptor object into a PartialDesc (ToPropertyDescriptor).
fn build_partial(i: &mut Interp, desc: &Value) -> Result<PartialDesc, Abrupt> {
    let o = match desc {
        Value::Obj(o) => o.clone(),
        _ => return Err(i.throw("TypeError", "Property description must be an object")),
    };
    if let Some([enumerable, configurable, value, writable, get, set]) = plain_descriptor_fields(&o)
    {
        let enumerable = enumerable.map(|v| i.to_boolean(&v));
        let configurable = configurable.map(|v| i.to_boolean(&v));
        let writable = writable.map(|v| i.to_boolean(&v));
        return validated_partial(i, value, get, set, writable, enumerable, configurable);
    }
    // ToPropertyDescriptor reads each field with HasProperty/Get (both trap-aware — the
    // descriptor may itself be a proxy), in the spec's field order: enumerable, configurable,
    // value, writable, get, set.
    let base = Value::Obj(o.clone());
    let bool_field = |i: &mut Interp, k: &str| -> Result<Option<bool>, Abrupt> {
        if i.js_has_property(&base, k)? {
            let v = i.get_member(&base, k)?;
            Ok(Some(i.to_boolean(&v)))
        } else {
            Ok(None)
        }
    };
    let enumerable = bool_field(i, "enumerable")?;
    let configurable = bool_field(i, "configurable")?;
    let value = if i.js_has_property(&base, "value")? {
        Some(i.get_member(&base, "value")?)
    } else {
        None
    };
    let writable = bool_field(i, "writable")?;
    let get = if i.js_has_property(&base, "get")? {
        Some(i.get_member(&base, "get")?)
    } else {
        None
    };
    let set = if i.js_has_property(&base, "set")? {
        Some(i.get_member(&base, "set")?)
    } else {
        None
    };
    validated_partial(i, value, get, set, writable, enumerable, configurable)
}

/// ToPropertyDescriptor's validation steps after the field reads.
fn validated_partial(
    i: &mut Interp,
    value: Option<Value>,
    get: Option<Value>,
    set: Option<Value>,
    writable: Option<bool>,
    enumerable: Option<bool>,
    configurable: Option<bool>,
) -> Result<PartialDesc, Abrupt> {
    if (get.is_some() || set.is_some()) && (value.is_some() || writable.is_some()) {
        return Err(i.throw(
            "TypeError",
            "Invalid property descriptor. Cannot both specify accessors and a value or writable attribute",
        ));
    }
    if let Some(g) = &get {
        if !matches!(g, Value::Undefined) && !g.is_callable() {
            return Err(i.throw("TypeError", "Getter must be a function"));
        }
    }
    if let Some(s) = &set {
        if !matches!(s, Value::Undefined) && !s.is_callable() {
            return Err(i.throw("TypeError", "Setter must be a function"));
        }
    }
    Ok(PartialDesc {
        value,
        get,
        set,
        writable,
        enumerable,
        configurable,
    })
}

/// `Number.prototype.toPrecision(p)`: `p` significant digits, fixed or exponential per the exponent.
fn to_precision(n: f64, p: usize) -> String {
    if n == 0.0 {
        return if p == 1 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(p - 1))
        };
    }
    let neg = n < 0.0;
    // `p` significant digits via scientific notation (`d.ddde±E`, the mantissa has exactly `p` digits).
    let sci = format!("{:.*e}", p - 1, n.abs());
    let (mantissa, exp_str) = sci.split_once('e').unwrap();
    let e: i32 = exp_str.parse().unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let body = if e < -6 || e >= p as i32 {
        let sign = if e >= 0 { "+" } else { "-" };
        if p == 1 {
            format!("{}e{}{}", digits, sign, e.abs())
        } else {
            format!("{}.{}e{}{}", &digits[..1], &digits[1..], sign, e.abs())
        }
    } else if e >= 0 {
        let ip = (e + 1) as usize;
        if ip >= p {
            digits
        } else {
            format!("{}.{}", &digits[..ip], &digits[ip..])
        }
    } else {
        format!("0.{}{}", "0".repeat((-e - 1) as usize), digits)
    };
    if neg {
        format!("-{body}")
    } else {
        body
    }
}

fn opt_norm(v: Option<Value>) -> Option<Value> {
    v.filter(|x| !matches!(x, Value::Undefined))
}

/// OrdinaryDefineOwnProperty with the non-configurable / non-extensible invariant checks.
/// Deferred-namespace trigger for builtin MOP entry points (`None` key = key-less operations
/// like [[OwnPropertyKeys]], which always trigger).
fn defer_trigger_v(i: &mut Interp, v: &Value, key: Option<&str>) -> Result<(), Value> {
    if let Value::Obj(o) = v {
        ab(i.defer_trigger(o, key))?;
    }
    Ok(())
}

fn define_own_property(i: &mut Interp, o: &Gc, key: &str, desc: &Value) -> Result<bool, Abrupt> {
    i.defer_trigger(o, Some(key))?;
    let d = build_partial(i, desc)?;
    define_own_property_record(i, o, key, &d)
}

/// The data descriptor { [[Value]]: v, all attributes true } of CreateDataProperty.
fn data_record(v: Value) -> PartialDesc {
    PartialDesc {
        value: Some(v),
        writable: Some(true),
        enumerable: Some(true),
        configurable: Some(true),
        ..PartialDesc::default()
    }
}

/// [[DefineOwnProperty]] of a non-proxy object with a Property Descriptor record. Callers
/// that construct the record themselves (CreateDataProperty) need no descriptor object: only
/// a proxy trap observes one (ECMA-262 FromPropertyDescriptor in [[DefineOwnProperty]]).
fn define_own_property_record(
    i: &mut Interp,
    o: &Gc,
    key: &str,
    d: &PartialDesc,
) -> Result<bool, Abrupt> {
    // ArgumentsExoticObject [[DefineOwnProperty]]: sync the live parameter value into the
    // ordinary property first; after a successful ordinary define, a plain value write goes
    // through the map, and only an accessor or writable:false severs the alias.
    let args_ptr = Rc::as_ptr(o) as usize;
    let mapped = i.mapped_arg_name(args_ptr, key).is_some();
    if mapped {
        if let Some(cur) = i.mapped_arg_value(args_ptr, key) {
            if let Some(p) = o.borrow_mut().props.get_mut(key) {
                p.set_value(cur);
            }
        }
        let allowed = define_own_property_ordinary(i, o, key, d)?;
        if !allowed {
            return Ok(false);
        }
        if d.is_accessor() {
            i.unmap_argument(args_ptr, key);
        } else {
            if let Some(v) = d.value.clone() {
                i.mapped_arg_write(args_ptr, key, v);
            }
            if d.writable == Some(false) {
                i.unmap_argument(args_ptr, key);
            }
        }
        return Ok(true);
    }
    define_own_property_ordinary(i, o, key, d)
}

fn define_own_property_ordinary(
    i: &mut Interp,
    o: &Gc,
    key: &str,
    d: &PartialDesc,
) -> Result<bool, Abrupt> {
    let d = d.clone();
    // Module namespace [[DefineOwnProperty]]: a String key is only redefinable to a descriptor that
    // matches the export's fixed shape (writable, enumerable, non-configurable, same value); adding
    // a new String key fails. Symbol keys fall through to the ordinary algorithm.
    let ptr = Rc::as_ptr(o) as usize;
    if i.is_namespace(ptr) && !Interp::is_sym_key(key) {
        return match i.namespace_own_property(ptr, key) {
            Some(res) => {
                let cur = res?;
                let ok = d.configurable != Some(true)
                    && d.enumerable != Some(false)
                    && !d.is_accessor()
                    && d.writable != Some(false)
                    && match &d.value {
                        Some(v) => same_value(v, &cur.value()),
                        None => true,
                    };
                Ok(ok)
            }
            None => Ok(false),
        };
    }
    // TypedArray integer-index [[DefineOwnProperty]]: a valid in-range index accepts only a
    // configurable+enumerable+writable data descriptor and writes the value; a canonical-numeric
    // non-index can't be defined; both never touch the property map.
    if let Some(info) = ta_info(i, o) {
        match i.ta_index_kind(&info, key) {
            crate::value::TaIndex::Element(idx) => {
                if d.is_accessor()
                    || d.configurable == Some(false)
                    || d.enumerable == Some(false)
                    || d.writable == Some(false)
                {
                    return Ok(false);
                }
                if let Some(v) = d.value.clone() {
                    i.ta_store(&info, idx, &v)?;
                }
                return Ok(true);
            }
            crate::value::TaIndex::Exotic => return Ok(false),
            crate::value::TaIndex::Ordinary => {}
        }
    }
    // Web IDL legacy platform object [[DefineOwnProperty]] rejects every array-index key when
    // the interface has an indexed getter but no indexed setter, supported or not.
    if i.host_indexed_array_key(o, key) || i.host_named_reject_define(o, key) {
        return Ok(false);
    }
    let is_array = matches!(o.borrow().exotic, crate::value::Exotic::Array);
    // Array exotic [[DefineOwnProperty]]: `length` and array indices have special rules.
    if is_array && key == "length" {
        crate::value::bump_proto_epoch();
        return array_set_length(i, o, &d);
    }
    let array_index = if is_array { canonical_index(key) } else { None };
    if let Some(idx) = array_index {
        // Adding an index at or past a non-writable `length` is rejected.
        let len = i.array_length(o);
        if idx as usize >= len
            && !o
                .borrow()
                .props
                .get("length")
                .map(|p| p.writable())
                .unwrap_or(true)
        {
            return Ok(false);
        }
    }
    let existing = o.borrow().props.get(key).cloned();

    let mut cur = match existing {
        None => {
            if !o.borrow().extensible {
                return Ok(false);
            }
            let prop = if d.is_accessor() {
                Property::accessor_prop(
                    opt_norm(d.get),
                    opt_norm(d.set),
                    d.enumerable.unwrap_or(false),
                    d.configurable.unwrap_or(false),
                )
            } else {
                Property::data(
                    d.value.unwrap_or(Value::Undefined),
                    d.writable.unwrap_or(false),
                    d.enumerable.unwrap_or(false),
                    d.configurable.unwrap_or(false),
                )
            };
            o.borrow_mut().props.insert(key, prop);
            grow_array_length(i, o, array_index);
            return Ok(true);
        }
        Some(p) => p,
    };
    // Redefining an existing property: a non-configurable property restricts what may change.
    if !cur.configurable() {
        if d.configurable == Some(true) {
            return Ok(false);
        }
        if let Some(e) = d.enumerable {
            if e != cur.enumerable() {
                return Ok(false);
            }
        }
        if d.is_accessor() != cur.accessor() && (d.is_accessor() || d.is_data()) {
            return Ok(false);
        }
        if cur.accessor() {
            if let Some(g) = &d.get {
                if !same_value(g, cur.getter().unwrap_or(&Value::Undefined)) {
                    return Ok(false);
                }
            }
            if let Some(s) = &d.set {
                if !same_value(s, cur.setter().unwrap_or(&Value::Undefined)) {
                    return Ok(false);
                }
            }
        } else if !cur.writable() {
            if d.writable == Some(true) {
                return Ok(false);
            }
            if let Some(v) = &d.value {
                if !same_value(v, &cur.value()) {
                    return Ok(false);
                }
            }
        }
    }
    // Redefining can rewrite attributes (data → accessor, writable → false) without a
    // structural change: invalidate the property-creation inline caches and protectors, whose
    // proofs assumed no such shadow could appear. A new key needs no bump here: Props::insert
    // already notes structural changes to marked prototypes.
    crate::value::bump_proto_epoch();
    // Apply the present fields.
    if d.is_accessor() {
        cur.set_accessor(true);
        cur.set_value(Value::Undefined);
        cur.set_writable(false);
        if let Some(g) = d.get {
            cur.set_getter(opt_norm(Some(g)));
        }
        if let Some(s) = d.set {
            cur.set_setter(opt_norm(Some(s)));
        }
    } else if d.is_data() || !cur.accessor() {
        if cur.accessor() {
            cur.clear_accessors();
            cur.set_accessor(false);
            cur.set_value(Value::Undefined);
            cur.set_writable(false);
        }
        if let Some(v) = d.value {
            cur.set_value(v);
        }
        if let Some(w) = d.writable {
            cur.set_writable(w);
        }
    }
    if let Some(e) = d.enumerable {
        cur.set_enumerable(e);
    }
    if let Some(c) = d.configurable {
        cur.set_configurable(c);
    }
    o.borrow_mut().props.insert(key, cur);
    grow_array_length(i, o, array_index);
    Ok(true)
}

/// After defining an array index, grow `length` to `index + 1` if the index reached past the end.
fn grow_array_length(i: &mut Interp, o: &Gc, array_index: Option<u32>) {
    if let Some(idx) = array_index {
        if idx as usize >= i.array_length(o) {
            let writable = o
                .borrow()
                .props
                .get("length")
                .map(|p| p.writable())
                .unwrap_or(true);
            o.borrow_mut().props.insert(
                "length",
                Property::data(Value::Num((idx as f64) + 1.0), writable, false, false),
            );
        }
    }
}

/// The value-only [[DefineOwnProperty]] used after OrdinarySet has checked the receiver's
/// writable own descriptor. Share ArraySetLength with Object/Reflect.defineProperty: coercion
/// can change the array's length and attributes, so neither caller can retain an earlier proof.
pub(crate) fn array_set_length_value(i: &mut Interp, o: &Gc, value: Value) -> Result<bool, Abrupt> {
    array_set_length(
        i,
        o,
        &PartialDesc {
            value: Some(value),
            ..PartialDesc::default()
        },
    )
}

/// ECMA-262 ArraySetLength (#sec-arraysetlength), local snapshot e28783d5fc9d.
/// Read the current descriptor only after both observable conversions. Only canonical array
/// indices participate in truncation; other numeric-looking names are ordinary properties.
fn array_set_length(i: &mut Interp, o: &Gc, d: &PartialDesc) -> Result<bool, Abrupt> {
    let new_len = match &d.value {
        None => {
            // `length` is a non-configurable, non-enumerable data property.
            if d.is_accessor()
                || matches!(d.configurable, Some(true))
                || matches!(d.enumerable, Some(true))
            {
                return Ok(false);
            }
            let len_writable = o
                .borrow()
                .props
                .get("length")
                .map(|p| p.writable())
                .unwrap_or(true);
            // No value: only a writable change.
            if let Some(w) = d.writable {
                if !len_writable && w {
                    return Ok(false); // can't make a non-writable length writable again
                }
                let cur = i.array_length(o) as f64;
                o.borrow_mut()
                    .props
                    .insert("length", Property::data(Value::Num(cur), w, false, false));
            }
            return Ok(true);
        }
        Some(Value::Num(number)) => {
            // ToNumber(Number) is the identity and cannot run author code. The two-conversion
            // SameValueZero test succeeds exactly for integral Numbers in [0, 2^32 - 1],
            // including -0 (normalized to +0). Saturating conversion plus an exact round trip
            // proves this range/integrality; unlike the object case below, no modulo result
            // outside that range could compare equal to the unchanged original Number.
            let new_len = *number as u32;
            if *number != new_len as f64 {
                return Err(i.throw("RangeError", "Invalid array length"));
            }
            new_len as usize
        }
        Some(v) => {
            // ArraySetLength coerces first — ToUint32(value) then ToNumber(value), both
            // observable — and a mismatch (fraction, negative, ≥ 2^32) is a RangeError before
            // any attribute validation.
            // ToUint32 is modulo 2^32 over the full Number range, not a saturating integer
            // cast. An object can return a huge Number here and a valid length on its second
            // conversion, so comparing only the original input range is not equivalent.
            let new_len = i.to_uint32(v)?;
            let number_len = i.to_number(v)?;
            let same = if number_len == 0.0 {
                new_len == 0
            } else {
                (new_len as f64) == number_len
            };
            if !same {
                return Err(i.throw("RangeError", "Invalid array length"));
            }
            new_len as usize
        }
    };
    // Now the ordinary-define validation against the current `length`.
    if d.is_accessor() || matches!(d.configurable, Some(true)) || matches!(d.enumerable, Some(true))
    {
        return Ok(false);
    }
    let len_writable = o
        .borrow()
        .props
        .get("length")
        .map(|p| p.writable())
        .unwrap_or(true);
    let old_len = i.array_length(o);
    if !len_writable && (new_len != old_len || matches!(d.writable, Some(true))) {
        return Ok(false);
    }
    if !len_writable {
        // OrdinaryDefineOwnProperty accepts the same normalized value on a non-writable
        // property. This includes a length frozen by one of the conversions above.
        return Ok(true);
    }
    let writable = d.writable.unwrap_or(len_writable);
    if new_len == old_len && writable == len_writable {
        // Validation and every observable conversion have already completed. Length is an
        // ordinary non-configurable, non-enumerable data property: reapplying this exact
        // value/attributes changes nothing (ValidateAndApplyPropertyDescriptor). In particular,
        // do not return here when the descriptor freezes a currently writable length.
        return Ok(true);
    }
    if new_len < old_len {
        // Publish the tentative length first, with writable still true. A requested freeze
        // is deferred until the deletion phase (including a possible failed deletion).
        let mut array = o.borrow_mut();
        array.props.insert(
            "length",
            Property::data(Value::Num(new_len as f64), true, false, false),
        );
        // Array own [[Delete]] cannot invoke an accessor or proxy trap. Finding the highest
        // non-configurable index and removing all indices above it is therefore equivalent
        // to descending deletion, without sorting/key allocation or O(n)-per-key compaction.
        let blocker = array.props.highest_nonconfig_index_from(new_len);
        let final_len = blocker.map_or(new_len, |index| index + 1);
        array.props.remove_indices_from(final_len);
        if blocker.is_some() || !writable {
            array.props.insert(
                "length",
                Property::data(Value::Num(final_len as f64), writable, false, false),
            );
        }
        return Ok(blocker.is_none());
    }
    o.borrow_mut().props.insert(
        "length",
        Property::data(Value::Num(new_len as f64), writable, false, false),
    );
    Ok(true)
}

pub(crate) fn same_value_pub(a: &Value, b: &Value) -> bool {
    same_value(a, b)
}
/// ToObject on a primitive (for sloppy-mode `this` coercion). Objects pass through.
pub(crate) fn box_primitive_pub(i: &mut Interp, v: Value) -> Value {
    box_primitive(i, v)
}
fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => {
            if x.is_nan() && y.is_nan() {
                return true;
            }
            if *x == 0.0 && *y == 0.0 {
                return x.is_sign_negative() == y.is_sign_negative();
            }
            x == y
        }
        (Value::Undefined, Value::Undefined) => true,
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::BigInt(x), Value::BigInt(y)) => x == y,
        (Value::Sym(x), Value::Sym(y)) => Rc::ptr_eq(x, y),
        (Value::Obj(x), Value::Obj(y)) => Rc::ptr_eq(x, y),
        _ => false,
    }
}

/// Whether `value` is one exact built-in native function. Function-pointer identity is used only
/// as a guard for semantics-preserving fast paths; wrappers, bound functions, and user functions
/// always take the observable algorithm.
fn is_native_function(value: &Value, expected: NativeFn) -> bool {
    matches!(
        value,
        Value::Obj(object)
            if matches!(object.borrow().call, Callable::Native(actual)
                if std::ptr::fn_addr_eq(actual, expected))
    )
}

fn is_intrinsic_array_constructor(i: &Interp, value: &Value) -> bool {
    matches!((i.array_ctor.as_ref(), value), (Some(expected), Value::Obj(actual)) if Rc::ptr_eq(expected, actual))
}

/// Whether the intrinsic Array constructor still creates instances with this Realm's ordinary
/// `%Array.prototype%`. `Array.of` reads the constructor's `prototype` through GetPrototypeFrom
/// Constructor, so changing that writable property (or replacing it with an accessor) must disable
/// allocation shortcuts even when the constructor function itself is still the intrinsic.
fn intrinsic_array_constructor_is_default(i: &Interp, value: &Value) -> bool {
    let Value::Obj(actual) = value else {
        return false;
    };
    if !matches!(i.array_ctor.as_ref(), Some(expected) if Rc::ptr_eq(expected, actual)) {
        return false;
    }
    actual
        .borrow()
        .props
        .get("prototype")
        .filter(|property| !property.accessor())
        .is_some_and(|property| {
            matches!(property.value(), Value::Obj(proto) if Rc::ptr_eq(&proto, &i.array_proto))
        })
}

/// Array iteration also performs `Get(iterator, "next")`; retaining the original `values`
/// function is insufficient if `%ArrayIteratorPrototype%.next` was replaced.
pub(crate) fn intrinsic_array_iterator_is_unmodified(i: &Interp) -> bool {
    i.extra_protos
        .get("%ArrayIteratorPrototype%")
        .and_then(|prototype| prototype.borrow().props.get("next").map(|p| p.value()))
        .is_some_and(|next| is_native_function(&next, array_iter_next))
}

/// Cardinality of an already-open Array iterator only when its captured next method and every
/// indexed Get/state transition are inert. Used solely to reject impossible bounded rest drains;
/// custom next methods and length-changing element getters must run normally.
pub(crate) fn inert_array_iterator_remaining(
    i: &Interp,
    iterator: &Value,
    next: &Value,
) -> Option<usize> {
    if !is_native_function(next, array_iter_next) {
        return None;
    }
    let object = iterator.as_obj()?;
    if !i.ordinary_get_ptr(Rc::as_ptr(object) as usize) {
        return None;
    }
    let object = object.borrow();
    let Exotic::ArrayIterator(state) = &object.exotic else {
        return None;
    };
    let target = state.target.as_obj()?;
    if !matches!(target.borrow().exotic, Exotic::Array)
        || !i.ordinary_get_ptr(Rc::as_ptr(target) as usize)
        || !i.array_append_unshadowed(target)
        || target
            .borrow()
            .props
            .indexed_properties()
            .any(|property| property.accessor())
    {
        return None;
    }
    Some(i.array_length(target).saturating_sub(state.index))
}

/// Whether an array can use the allocation-free iterator snapshot in `Interp::iterate`.
///
/// The language-level spread and argument-list algorithms begin with `GetIterator`, so this
/// proof must include the observable `@@iterator` lookup as well as the iterator's `next` method.
/// Requiring the ordinary realm prototype and no own symbol override keeps the check entirely
/// side-effect free; customized arrays take the protocol path.
pub(crate) fn array_iterator_fast_path_is_safe(i: &Interp, value: &Value) -> bool {
    let Value::Obj(object) = value else {
        return false;
    };
    if !matches!(object.borrow().exotic, Exotic::Array)
        || !i.ordinary_get_ptr(Rc::as_ptr(object) as usize)
        || !matches!(
            object.borrow().proto.as_ref(),
            Some(proto) if Rc::ptr_eq(proto, &i.array_proto)
        )
    {
        return false;
    }
    let Some(key) = i
        .iterator_sym
        .as_ref()
        .map(|symbol| Interp::sym_key(symbol.as_ref()))
    else {
        return false;
    };
    if object.borrow().props.get(&key).is_some() {
        return false;
    }
    let method = i
        .array_proto
        .borrow()
        .props
        .get(&key)
        .filter(|property| !property.accessor())
        .map(|property| property.value());
    method.is_some_and(|method| {
        is_native_function(&method, nf_array_values) && intrinsic_array_iterator_is_unmodified(i)
    })
}

/// Closed, non-exhausting Array binding prefix. No lookup here may invoke author code.
/// The caller preserves the captured next owner and native call boundaries, and only uses
/// this proof between fresh lexical initializations (no defaults or old binding destructors).
pub(crate) fn dense_array_binding_methods(
    i: &Interp,
    value: &Value,
    count: usize,
) -> Option<(Value, Value)> {
    let object = value.as_obj()?;
    if count > 8
        || !array_iterator_fast_path_is_safe(i, value)
        || !i.ordinary_get_ptr(Rc::as_ptr(&i.array_proto) as usize)
        || i.array_length(object) < count
    {
        return None;
    }
    {
        let object = object.borrow();
        for index in 0..count {
            if object.props.get_index(index as u32)?.accessor() {
                return None;
            }
        }
    }
    let key = Interp::sym_key(i.iterator_sym.as_ref()?);
    let method = i.array_proto.borrow().props.get(&key)?.value();
    let prototype = i.extra_protos.get("%ArrayIteratorPrototype%")?;
    if !i.ordinary_get_ptr(Rc::as_ptr(prototype) as usize) {
        return None;
    }
    let next = prototype
        .borrow()
        .props
        .get("next")
        .filter(|property| !property.accessor())?
        .value();
    for function in [&method, &next] {
        let function = function.as_obj()?;
        if !i.ordinary_get_ptr(Rc::as_ptr(function) as usize)
            || !i.native_call_in_current_realm(function)
        {
            return None;
        }
    }
    // IteratorClose reads return at the end. With no observers in the admitted prefix,
    // absence (or a nullish plain value) remains valid; a getter or proxy must run normally.
    let mut current = Some(prototype.clone());
    for _ in 0..64 {
        let Some(object) = current else {
            return Some((method, next));
        };
        if !i.ordinary_get_ptr(Rc::as_ptr(&object) as usize) {
            return None;
        }
        let object = object.borrow();
        if !matches!(object.exotic, Exotic::None) {
            return None;
        }
        if let Some(property) = object.props.get("return") {
            return (!property.accessor()
                && matches!(property.value(), Value::Undefined | Value::Null))
            .then_some((method, next));
        }
        current = object.proto.clone();
    }
    None
}

/// Snapshot an Array whose current iteration cannot execute ECMAScript code: every index below
/// `length` is an own plain data property. Callers separately prove that the selected iterator is
/// the intrinsic Array iterator. Holes and accessors deliberately miss so prototype lookup and
/// side effects retain the ordinary iterator path.
pub(crate) fn dense_array_snapshot(i: &Interp, value: &Value) -> Option<Vec<Value>> {
    let Value::Obj(object) = value else {
        return None;
    };
    if !matches!(object.borrow().exotic, Exotic::Array)
        || !i.ordinary_get_ptr(Rc::as_ptr(object) as usize)
    {
        return None;
    }
    let length = i.array_length(object);
    let object = object.borrow();
    let mut values = Vec::with_capacity(length);
    for index in 0..length {
        let property = object.props.get_index(index as u32)?;
        if property.accessor() {
            return None;
        }
        values.push(property.value());
    }
    Some(values)
}

fn nf_array_values(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    arr_require_coercible(i, &this)?;
    Ok(make_array_iterator(i, this, 0))
}

// ---------------------------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------------------------

fn install_array(it: &mut Interp) {
    let ap = it.array_proto.clone();
    ap.borrow().props.mark_array();
    ap.borrow_mut().exotic = Exotic::Array;
    ap.borrow_mut().props.insert(
        "length",
        Property::data(Value::Num(0.0), true, false, false),
    );
    // %Array.prototype% [@@unscopables]: a null-prototype object naming the `with`-transparent
    // methods; { writable: false, enumerable: false, configurable: true }.
    if let Some(key) = well_known_key(it, "unscopables") {
        let un = Object::new(None);
        for name in [
            "at",
            "copyWithin",
            "entries",
            "fill",
            "find",
            "findIndex",
            "findLast",
            "findLastIndex",
            "flat",
            "flatMap",
            "includes",
            "keys",
            "toReversed",
            "toSorted",
            "toSpliced",
            "values",
        ] {
            un.borrow_mut()
                .props
                .insert(name, Property::data(Value::Bool(true), true, true, true));
        }
        ap.borrow_mut()
            .props
            .insert(key, Property::data(Value::Obj(un), false, false, true));
    }

    it.def_method(&ap, "push", 1, nf_array_push);
    it.def_method(&ap, "pop", 0, nf_array_pop);
    install_array_rest(it, ap.clone());
    // forEach, map, filter, some, every, find, findIndex, findLast, findLastIndex, reduce and
    // reduceRight are self-hosted.
    crate::self_hosted::install_array_methods(it, &ap);
}

pub(crate) fn nf_array_push(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    let o = arr_to_object(i, &this)?;
    // Dense fast path: a plain array whose `length` is a writable own data property and
    // whose tail is exactly the dense frontier appends in place — no key strings, no
    // existence scans, no observable coercions (a whole-number own `length` needs none).
    if matches!(o.borrow().exotic, Exotic::Array)
        && i.ordinary_get_ptr(Rc::as_ptr(&o) as usize)
        && i.array_append_unshadowed(&o)
    {
        let mut b = o.borrow_mut();
        // Set(O, len, E, true) creates an own element: a non-extensible array rejects it
        // (§10.1.6.3 ValidateAndApplyPropertyDescriptor) and the general path throws.
        let len = match b.props.length_property() {
            Some(p) if b.extensible && !p.accessor() && p.writable() => match p.value() {
                Value::Num(n) if n.trunc() == n && (0.0..=u32::MAX as f64).contains(&n) => {
                    Some(n as u32)
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(mut len) = len {
            if (len as u64 + args.len() as u64) <= u32::MAX as u64 {
                let mut ok = true;
                for a in args {
                    if b.props.append_element(len, Property::plain(a.clone())) {
                        len += 1;
                    } else {
                        ok = false;
                        break;
                    }
                }
                // Sync length up to what actually landed (partial success still moved it).
                if !matches!(b.props.length_property().map(|p| p.value()),
                             Some(Value::Num(n)) if n == len as f64)
                {
                    let s = b.props.slot_of("length").unwrap();
                    b.props
                        .entry_at_mut(s)
                        .unwrap()
                        .1
                        .set_value(Value::Num(len as f64));
                }
                if ok {
                    return Ok(Value::Num(len as f64));
                }
            }
        }
    }
    let ov = Value::Obj(o.clone());
    // `push` only writes at the tail, so a huge array-like length is fine (use ToLength, not the
    // engine's materialization cap).
    let mut len = ab(i.to_length(&o))? as u64;
    // The resulting length may not exceed 2^53-1.
    if len + args.len() as u64 > 9007199254740991 {
        return Err(i.make_error("TypeError", "push would exceed the maximum array length"));
    }
    for a in args {
        set_throw(i, &ov, &len.to_string(), a.clone())?;
        len += 1;
    }
    // Generic objects don't auto-track length the way arrays do, so set it explicitly.
    set_throw(i, &ov, "length", Value::Num(len as f64))?;
    Ok(Value::Num(len as f64))
}

pub(crate) fn nf_array_pop(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let o = arr_to_object(i, &this)?;
    // Dense fast path (mirror of push's): take the last element straight off the entries
    // tail — no key strings, no hash lookups, and crucially NO shape reset, so the array's
    // inline caches survive a pop (stack-discipline arrays live on push/pop).
    if matches!(o.borrow().exotic, Exotic::Array) && i.ordinary_get_ptr(Rc::as_ptr(&o) as usize) {
        let mut b = o.borrow_mut();
        let len = match b.props.length_property() {
            Some(p) if !p.accessor() && p.writable() => match p.value() {
                Value::Num(n) if n.trunc() == n && (1.0..=u32::MAX as f64).contains(&n) => {
                    Some(n as u32)
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(len) = len {
            if let Some(v) = b.props.pop_last_element(len - 1) {
                let s = b.props.slot_of("length").unwrap();
                b.props
                    .entry_at_mut(s)
                    .unwrap()
                    .1
                    .set_value(Value::Num((len - 1) as f64));
                return Ok(v);
            }
        }
    }
    let ov = Value::Obj(o.clone());
    // `pop` only touches the last index, so a huge array-like length is fine (use ToLength).
    let len = ab(i.to_length(&o))?;
    if len == 0 {
        set_throw(i, &ov, "length", Value::Num(0.0))?;
        return Ok(Value::Undefined);
    }
    let last = ab(i.get_member(&ov, &(len - 1).to_string()))?;
    delete_or_throw(i, &ov, &(len - 1).to_string())?;
    set_throw(i, &ov, "length", Value::Num((len - 1) as f64))?;
    Ok(last)
}

/// JIT-only ownership-transfer form of the one-argument dense `Array#push` case.
///
/// `Err(value)` means no mutation occurred and hands the argument back to the caller for the
/// exact generic builtin path. `Ok(length)` means the argument's ownership moved into the array.
pub(crate) fn jit_array_push_one(i: &mut Interp, o: &Gc, value: Value) -> Result<Value, Value> {
    array_push_dense_packed(i, o, PackedValue::pack(value))
        .map(|length| Value::Num(length as f64))
        .map_err(PackedValue::into_value)
}

/// `Array.prototype.push(value)` (ECMA-262 §23.1.3.23) on an extensible ordinary Array whose
/// `length` is a writable whole-number data property and whose prototypes provide no indexed
/// properties: Set(O, len, value, true) is then exactly CreateDataProperty at the dense tail,
/// and Set(O, "length", len + 1, true) the matching length update. Returns the new length,
/// or hands the value back (nothing changed) for the general algorithm.
pub(crate) fn array_push_dense_packed(
    i: &Interp,
    o: &Gc,
    value: PackedValue,
) -> Result<u32, PackedValue> {
    {
        let b = o.borrow();
        if !matches!(b.exotic, Exotic::Array) || !b.ic_plain.get() || !b.extensible {
            return Err(value);
        }
    }
    if !i.array_append_unshadowed(o) {
        return Err(value);
    }
    let mut b = o.borrow_mut();
    let len = match b.props.length_property() {
        Some(p) if !p.accessor() && p.writable() => match p.number_value() {
            Some(n) if n.trunc() == n && (0.0..u32::MAX as f64).contains(&n) => n as u32,
            _ => return Err(value),
        },
        _ => return Err(value),
    };
    match b
        .props
        .try_append_element(len, Property::plain_packed(value))
    {
        Ok(()) => {
            let next = len + 1;
            b.props.set_array_length_value(next as f64);
            Ok(next)
        }
        Err(prop) => Err(PackedValue::pack(prop.into_value())),
    }
}

/// JIT-only dense `Array#pop` form. `None` is a guard miss with no mutation.
pub(crate) fn jit_array_pop(i: &Interp, o: &Gc) -> Option<Value> {
    if !matches!(o.borrow().exotic, Exotic::Array) || !i.ordinary_get_ptr(Rc::as_ptr(o) as usize) {
        return None;
    }
    let mut b = o.borrow_mut();
    let len = match b.props.length_property() {
        Some(p) if !p.accessor() && p.writable() => match p.value() {
            Value::Num(n) if n.trunc() == n && (0.0..=u32::MAX as f64).contains(&n) => n as u32,
            _ => return None,
        },
        _ => return None,
    };
    if len == 0 {
        return Some(Value::Undefined);
    }
    let value = b.props.pop_last_element(len - 1)?;
    let slot = b.props.slot_of("length").unwrap();
    b.props
        .entry_at_mut(slot)
        .unwrap()
        .1
        .set_value(Value::Num((len - 1) as f64));
    Some(value)
}

fn install_array_rest(it: &mut Interp, ap: Gc) {
    it.def_method(&ap, "shift", 0, |i, this, _args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))?;
        if len == 0 {
            set_throw(i, &ov, "length", Value::Num(0.0))?;
            return Ok(Value::Undefined);
        }
        // ECMA-262 #sec-array.prototype.shift: own data reads/writes cannot
        // observe the prototype, even on non-extensible Arrays. The storage
        // proof rejects holes/accessors/failing descriptors before any mutation.
        if matches!(o.borrow().exotic, Exotic::Array) && i.ordinary_get_ptr(Rc::as_ptr(&o) as usize)
        {
            if let Some(first) = o.borrow_mut().props.shift_dense_array(len) {
                return Ok(first);
            }
        }
        let first = ab(i.get_member(&ov, "0"))?;
        for k in 1..len {
            let to = (k - 1).to_string();
            if let Some(v) = array_get_present_index(i, &o, &ov, k)? {
                set_throw(i, &ov, &to, v)?;
            } else {
                delete_or_throw(i, &ov, &to)?;
            }
        }
        delete_or_throw(i, &ov, &(len - 1).to_string())?;
        set_throw(i, &ov, "length", Value::Num((len - 1) as f64))?;
        Ok(first)
    });
    it.def_method(&ap, "unshift", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.to_length(&o))? as u64;
        let n = args.len() as u64;
        if n > 0 {
            if len + n > 9007199254740991 {
                return Err(i.make_error("TypeError", "unshift result is too long"));
            }
            for k in (0..len).rev() {
                let to = (k + n).to_string();
                if let Some(v) = array_get_present_index(i, &o, &ov, k as usize)? {
                    set_throw(i, &ov, &to, v)?;
                } else {
                    delete_or_throw(i, &ov, &to)?;
                }
            }
            for (idx, a) in args.iter().enumerate() {
                set_throw(i, &ov, &idx.to_string(), a.clone())?;
            }
        }
        set_throw(i, &ov, "length", Value::Num((len + n) as f64))?;
        Ok(Value::Num((len + n) as f64))
    });
    it.def_method(&ap, "slice", 2, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        // The total length may be near 2^53 for an array-like; only the copied span (end-start) is
        // bounded by the engine's materialization cap.
        let len = ab(i.to_length(&o))? as i64;
        let start = norm_index(ab(i.to_number(&arg(args, 0)))?, len);
        let end = match arg(args, 1) {
            Value::Undefined => len,
            v => norm_index(ab(i.to_number(&v))?, len),
        };
        let count = (end - start).max(0) as usize;
        let result = array_species_create(i, &this, count as u64)?;
        let ov = Value::Obj(o.clone());
        let mut k = start;
        let mut to = 0usize;
        while k < end {
            // Preserve holes: only copy indices the source actually has (HasProperty).
            if let Some(v) = array_get_present_index(i, &o, &ov, k as usize)? {
                cdp_index_or_throw(i, &result, to as u64, v)?;
            }
            k += 1;
            to += 1;
        }
        set_length_throw(i, &result, to as f64)?;
        Ok(result)
    });
    it.def_method(&ap, "indexOf", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let len = ab(i.to_length(&o))?;
        if len == 0 {
            // The length check precedes the fromIndex coercion.
            return Ok(Value::Num(-1.0));
        }
        let target = arg(args, 0);
        let from = match arg(args, 1) {
            Value::Undefined => 0usize,
            v => {
                let n = ab(i.to_number(&v))?;
                if n == f64::INFINITY {
                    return Ok(Value::Num(-1.0));
                }
                let n = if n.is_nan() { 0.0 } else { n.trunc() };
                if n >= 0.0 {
                    n as usize
                } else {
                    (len as f64 + n).max(0.0) as usize
                }
            }
        };
        let ov = Value::Obj(o.clone());
        for k in from..len {
            let Some(v) = array_get_present_index(i, &o, &ov, k)? else {
                continue; // indexOf skips holes
            };
            if i.strict_equals(&v, &target) {
                return Ok(Value::Num(k as f64));
            }
        }
        Ok(Value::Num(-1.0))
    });
    it.def_method(&ap, "includes", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let len = ab(i.to_length(&o))? as i64;
        if len == 0 {
            // The length check precedes the fromIndex coercion.
            return Ok(Value::Bool(false));
        }
        let target = arg(args, 0);
        let mut k = match arg(args, 1) {
            Value::Undefined => 0i64,
            v => {
                let n = ab(i.to_number(&v))?;
                if n == f64::INFINITY {
                    return Ok(Value::Bool(false));
                }
                if n == f64::NEG_INFINITY || n.is_nan() {
                    0
                } else if n >= 0.0 {
                    n.trunc() as i64
                } else {
                    (len + n.trunc() as i64).max(0)
                }
            }
        };
        let ov = Value::Obj(o.clone());
        while k < len {
            let v = array_get_index(i, &o, &ov, k as usize)?;
            if same_value_zero(&v, &target) {
                return Ok(Value::Bool(true));
            }
            k += 1;
        }
        Ok(Value::Bool(false))
    });
    it.def_method(&ap, "join", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))?;
        let sep = match arg(args, 0) {
            Value::Undefined => crate::lstr::LStr::from(","),
            v => ab(i.to_string(&v))?,
        };
        // ECMA-262 §23.1.3.18 performs separator conversion before the left-to-right element
        // Gets/ToStrings.  Append directly to one result buffer in that same order: `to_string`
        // already returns an LStr, so this avoids one temporary Rust String per element while
        // retaining the canonical UTF-16 fixup pass for the final sequence.
        let mut out = String::new();
        for k in 0..len {
            if k > 0 {
                out.push_str(&sep);
            }
            let v = array_get_index(i, &o, &ov, k)?;
            if !matches!(v, Value::Undefined | Value::Null) {
                out.push_str(&ab(i.to_string(&v))?);
            }
        }
        Ok(Value::lstr(crate::jstr::canonicalize(&out).unwrap_or(out)))
    });
    it.def_method(&ap, "concat", 1, |i, this, args| {
        // ToObject(this): a primitive receiver is boxed (and spread/appended as its wrapper).
        let recv = Value::Obj(arr_to_object(i, &this)?);
        let result = array_species_create(i, &recv, 0)?;
        let mut n = 0u64;
        // ECMA-262 §23.1.3.2 prepends the receiver to `items`, then processes each item in order.
        // Borrow that sequence directly so ordinary concat calls do not allocate a temporary
        // argument vector or clone every argument before the observable spreadability checks.
        for v in std::iter::once(&recv).chain(args.iter()) {
            // IsConcatSpreadable: @@isConcatSpreadable if defined, else IsArray.
            let spreadable = if let Value::Obj(_) = v {
                let key = well_known_key(i, "isConcatSpreadable");
                let flag = match &key {
                    Some(k) => ab(i.get_member(v, k))?,
                    None => Value::Undefined,
                };
                match flag {
                    Value::Undefined => json_is_array(i, v)?,
                    other => i.to_boolean(&other),
                }
            } else {
                false
            };
            if spreadable {
                let len = ab(i.to_length(&match v {
                    Value::Obj(o) => o.clone(),
                    _ => unreachable!(),
                }))? as u64;
                if n + len > 9007199254740991 {
                    return Err(i.make_error("TypeError", "concat result is too long"));
                }
                for k in 0..len {
                    let Value::Obj(object) = v else {
                        unreachable!("spreadable values are objects")
                    };
                    if let Some(elem) = array_get_present_index(i, object, v, k as usize)? {
                        cdp_index_or_throw(i, &result, n, elem)?;
                    }
                    n += 1; // increment for holes too, preserving their position
                }
            } else {
                if n >= 9007199254740991 {
                    return Err(i.make_error("TypeError", "concat result is too long"));
                }
                cdp_index_or_throw(i, &result, n, v.clone())?;
                n += 1;
            }
        }
        set_length_throw(i, &result, n as f64)?;
        Ok(result)
    });
    it.def_method(&ap, "reverse", 0, |i, this, _args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.to_length(&o))?;
        for k in 0..len / 2 {
            let lower = k.to_string();
            let upper = (len - 1 - k).to_string();
            // HasProperty/Get the two ends, then swap — preserving holes (a hole moves as a
            // DeletePropertyOrThrow).
            let lower_val = array_get_present_index(i, &o, &ov, k)?;
            let upper_val = array_get_present_index(i, &o, &ov, len - 1 - k)?;
            match (lower_val, upper_val) {
                (Some(lv), Some(uv)) => {
                    set_throw(i, &ov, &lower, uv)?;
                    set_throw(i, &ov, &upper, lv)?;
                }
                (None, Some(uv)) => {
                    set_throw(i, &ov, &lower, uv)?;
                    delete_or_throw(i, &ov, &upper)?;
                }
                (Some(lv), None) => {
                    delete_or_throw(i, &ov, &lower)?;
                    set_throw(i, &ov, &upper, lv)?;
                }
                (None, None) => {}
            }
        }
        Ok(ov)
    });
    it.def_method(&ap, "toString", 0, |i, this, _args| {
        let ov = Value::Obj(arr_to_object(i, &this)?);
        let join = ab(i.get_member(&ov, "join"))?;
        if join.is_callable() {
            ab(i.call(join, ov, &[]))
        } else {
            // Fall back to the %Object.prototype.toString% INTRINSIC semantics (the live
            // Object.prototype.toString may have been deleted or replaced). IsArray pierces
            // proxies (and throws on a revoked one).
            let builtin = if json_is_array(i, &ov)? {
                "Array"
            } else {
                builtin_tag(i, &ov)
            };
            match to_string_tag_key(i) {
                Some(key) => match ab(i.get_member(&ov, &key))? {
                    Value::Str(s) => Ok(Value::Str(crate::lstr::LStr::concat3(
                        "[object ",
                        s.as_str(),
                        "]",
                    ))),
                    _ => Ok(Value::Str(crate::lstr::LStr::concat3(
                        "[object ", builtin, "]",
                    ))),
                },
                None => Ok(Value::Str(crate::lstr::LStr::concat3(
                    "[object ", builtin, "]",
                ))),
            }
        }
    });
    it.def_method(&ap, "toLocaleString", 0, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))?;
        let mut out = String::new();
        for k in 0..len {
            if k > 0 {
                out.push(',');
            }
            let el = array_get_index(i, &o, &ov, k)?;
            if !matches!(el, Value::Undefined | Value::Null) {
                // ToString(? Invoke(element, "toLocaleString", « locales, options »)).
                let tls = ab(i.get_member(&el, "toLocaleString"))?;
                if !tls.is_callable() {
                    return Err(i.make_error("TypeError", "toLocaleString is not a function"));
                }
                let s = ab(i.call(tls, el, &[arg(args, 0), arg(args, 1)]))?;
                out.push_str(&ab(i.to_string(&s))?);
            }
        }
        Ok(Value::lstr(out))
    });
    it.def_method(&ap, "at", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let len = ab(i.checked_array_len(&o))? as i64;
        let mut idx = ab(i.to_number(&arg(args, 0)))? as i64;
        if idx < 0 {
            idx += len;
        }
        if idx < 0 || idx >= len {
            return Ok(Value::Undefined);
        }
        let ov = Value::Obj(o.clone());
        array_get_index(i, &o, &ov, idx as usize)
    });
    it.def_method(&ap, "fill", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let len = ab(i.to_length(&o))? as i64;
        let v = arg(args, 0);
        let start = norm_index(ab(i.to_number(&arg(args, 1)))?, len);
        let end = match arg(args, 2) {
            Value::Undefined => len,
            x => norm_index(ab(i.to_number(&x))?, len),
        };
        // A real Array's filled span is bounded by the engine cap (it materializes one property
        // per index); a generic array-like iterates lazily — its accessors typically throw or
        // the per-op caps stop runaway growth.
        if matches!(o.borrow().exotic, Exotic::Array)
            && (end - start).max(0) as usize > MAX_ARRAY_OP_LEN
        {
            return Err(i.make_error("RangeError", "array length exceeds engine limit"));
        }
        // A fresh holey Array with its standard sole `length` property has no indexed writes
        // that can invoke ECMAScript code once the prototype-index protector is valid. This is a
        // frequent initialization idiom (`new Array(n).fill(v)`); materialize its dense property
        // map in one allocation instead of doing n decimal conversions and OrdinarySet walks.
        let can_materialize = start == 0
            && end == len
            && len > 0
            && i.ordinary_get_ptr(Rc::as_ptr(&o) as usize)
            && i.array_append_unshadowed(&o)
            && {
                let object = o.borrow();
                object.extensible
                    && object.props.iter().count() == 1
                    && object.props.indexed_properties().next().is_none()
                    && object.props.get("length").is_some_and(|length| {
                        !length.accessor()
                            && length.writable()
                            && !length.enumerable()
                            && !length.configurable()
                            && matches!(length.value(), Value::Num(n) if n == len as f64)
                    })
            };
        if can_materialize {
            let Value::Obj(materialized) = i.make_array(vec![v; len as usize]) else {
                unreachable!("make_array returns an Array object")
            };
            let properties = std::mem::take(&mut materialized.borrow_mut().props);
            o.borrow_mut().props = properties;
            return Ok(Value::Obj(o));
        }
        let ov = Value::Obj(o.clone());
        for k in start..end {
            set_throw(i, &ov, &k.to_string(), v.clone())?;
        }
        Ok(ov)
    });
    it.def_method(&ap, "lastIndexOf", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let len = ab(i.to_length(&o))? as i64;
        if len == 0 {
            return Ok(Value::Num(-1.0));
        }
        let target = arg(args, 0);
        // fromIndex (default len-1): the highest index to search from, going backward.
        let mut k = if args.len() > 1 {
            let n = ab(i.to_number(&arg(args, 1)))?;
            if n == f64::NEG_INFINITY {
                return Ok(Value::Num(-1.0));
            }
            let n = if n.is_nan() { 0 } else { n.trunc() as i64 };
            if n >= 0 {
                n.min(len - 1)
            } else {
                len + n
            }
        } else {
            len - 1
        };
        let ov = Value::Obj(o.clone());
        while k >= 0 {
            let Some(v) = array_get_present_index(i, &o, &ov, k as usize)? else {
                k -= 1;
                continue;
            };
            if i.strict_equals(&v, &target) {
                return Ok(Value::Num(k as f64));
            }
            k -= 1;
        }
        Ok(Value::Num(-1.0))
    });
    it.def_method(&ap, "flat", 0, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let source_len = ab(i.to_length(&o))?;
        // ToIntegerOrInfinity(depth); undefined defaults to 1.
        let depth = match arg(args, 0) {
            Value::Undefined => 1i64,
            v => {
                let n = ab(i.to_number(&v))?;
                if n.is_nan() || n <= 0.0 {
                    0
                } else if n == f64::INFINITY {
                    i64::MAX
                } else {
                    n as i64
                }
            }
        };
        let a = array_species_create(i, &this, 0)?;
        let ov = Value::Obj(o.clone());
        flatten_into(i, &a, &ov, source_len, 0, depth, None, &Value::Undefined)?;
        Ok(a)
    });
    it.def_method(&ap, "flatMap", 1, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let len = ab(i.to_length(&o))?;
        let cb = arg(args, 0);
        if !cb.is_callable() {
            return Err(i.make_error(
                "TypeError",
                "Array.prototype.flatMap mapper is not callable",
            ));
        }
        let cb_this = arg(args, 1);
        let a = array_species_create(i, &this, 0)?;
        let ov = Value::Obj(o.clone());
        let cb = crate::callback::Callback::new(cb);
        flatten_into(i, &a, &ov, len, 0, 1, Some(&cb), &cb_this)?;
        Ok(a)
    });
    it.def_method(&ap, "splice", 2, array_splice);
    it.def_method(&ap, "sort", 1, |i, this, args| {
        let cmp = arg(args, 0);
        if !matches!(cmp, Value::Undefined) && !cmp.is_callable() {
            return Err(i.make_error(
                "TypeError",
                "the comparator must be a function or undefined",
            ));
        }
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))?;
        // SortIndexedProperties: read only the present indices (holes are skipped, not read).
        let mut items = Vec::new();
        for k in 0..len {
            let key = k.to_string();
            if ab(i.js_has_property(&ov, &key))? {
                items.push(array_get_index_after_has(i, &o, &ov, k, &key)?);
            }
        }
        let item_count = items.len();
        merge_sort(i, &mut items, &cmp)?;
        // Set(O, k, v, true): a failed write (non-writable element) always throws.
        for (k, v) in items.into_iter().enumerate() {
            let ok = ab(i.set_member_recv(&ov, &k.to_string(), v, ov.clone()))?;
            if !ok {
                return Err(i.make_error(
                    "TypeError",
                    format!("cannot assign to read-only element {k}"),
                ));
            }
        }
        // Vacated trailing indices (originally holes, or beyond the present count) are deleted
        // in ascending order through [[Delete]] (a proxy's deleteProperty trap observes each).
        for k in item_count..len {
            delete_or_throw(i, &ov, &k.to_string())?;
        }
        Ok(ov)
    });
    // ----- change-array-by-copy (return a new Array, leave the receiver untouched) -----
    it.def_method(&ap, "toReversed", 0, |i, this, _| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))?;
        // Change-by-copy methods use ArrayCreate and Get. A dense ordinary Array has no
        // observable work in those indexed Gets, so snapshot its own data once and reverse the
        // values in place (ECMA-262 §23.1.3.33).
        if len <= MAX_ARRAY_OP_LEN {
            if let Some(mut items) = dense_array_snapshot(i, &ov) {
                items.reverse();
                return Ok(i.make_array(items));
            }
        }
        // Elements are read from the end down (from = len - k - 1) into ascending targets.
        let mut items = Vec::with_capacity(len);
        for k in 0..len {
            items.push(array_get_index(i, &o, &ov, len - k - 1)?);
        }
        Ok(i.make_array(items))
    });
    it.def_method(&ap, "toSorted", 1, |i, this, args| {
        // The comparator is validated before the receiver is read.
        let cmp = arg(args, 0);
        if !matches!(cmp, Value::Undefined) && !cmp.is_callable() {
            return Err(i.make_error("TypeError", "comparator is not callable"));
        }
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))?;
        if len <= MAX_ARRAY_OP_LEN {
            if let Some(mut items) = dense_array_snapshot(i, &ov) {
                merge_sort(i, &mut items, &cmp)?;
                return Ok(i.make_array(items));
            }
        }
        let mut items = Vec::with_capacity(len);
        for k in 0..len {
            items.push(array_get_index(i, &o, &ov, k)?);
        }
        merge_sort(i, &mut items, &cmp)?;
        Ok(i.make_array(items))
    });
    it.def_method(&ap, "with", 2, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.checked_array_len(&o))? as i64;
        // ToIntegerOrInfinity truncates first (so -0.5 → 0), then negatives count from the end.
        let rel = ab(i.to_number(&arg(args, 0)))?;
        let rel = if rel.is_nan() { 0.0 } else { rel.trunc() };
        let idx = if rel < 0.0 {
            len + rel as i64
        } else {
            rel as i64
        };
        if idx < 0 || idx >= len {
            return Err(i.make_error("RangeError", "invalid index"));
        }
        if len as usize <= MAX_ARRAY_OP_LEN {
            if let Some(mut items) = dense_array_snapshot(i, &ov) {
                items[idx as usize] = arg(args, 1);
                return Ok(i.make_array(items));
            }
        }
        // The replaced index is never read.
        let mut items = Vec::with_capacity(len as usize);
        for k in 0..len {
            if k == idx {
                items.push(arg(args, 1));
            } else {
                items.push(array_get_index(i, &o, &ov, k as usize)?);
            }
        }
        Ok(i.make_array(items))
    });
    it.def_method(&ap, "toSpliced", 2, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.to_length(&o))? as u64;
        let start = norm_index(ab(i.to_number(&arg(args, 0)))?, len as i64) as u64;
        // No start → skip 0; start but no deleteCount → delete through the end.
        let del = if args.is_empty() {
            0
        } else if args.len() < 2 {
            len - start
        } else {
            let d = ab(i.to_number(&arg(args, 1)))?;
            let d = if d.is_nan() { 0.0 } else { d.trunc().max(0.0) };
            (d.min((len - start) as f64)) as u64
        };
        // The rest parameter is already an immutable argument slice. Borrow it directly instead
        // of cloning every insertion into a temporary Vec before the result is built; the
        // specification processes these values only after the source indices are determined.
        let inserts = args.get(2..).unwrap_or(&[]);
        let new_len = len - del + inserts.len() as u64;
        if new_len > 9007199254740991 {
            return Err(i.make_error("TypeError", "toSpliced result is too long"));
        }
        if new_len as usize > MAX_ARRAY_OP_LEN {
            return Err(i.make_error("RangeError", "array length exceeds engine limit"));
        }
        if len as usize <= MAX_ARRAY_OP_LEN {
            if let Some(source) = dense_array_snapshot(i, &ov) {
                let mut items = Vec::with_capacity(new_len as usize);
                items.extend_from_slice(&source[..start as usize]);
                items.extend(inserts.iter().cloned());
                items.extend_from_slice(&source[(start + del) as usize..len as usize]);
                return Ok(i.make_array(items));
            }
        }
        // The discarded span is never read.
        let mut items = Vec::with_capacity(new_len as usize);
        for k in 0..start {
            items.push(array_get_index(i, &o, &ov, k as usize)?);
        }
        items.extend(inserts.iter().cloned());
        for k in (start + del)..len {
            items.push(array_get_index(i, &o, &ov, k as usize)?);
        }
        Ok(i.make_array(items))
    });
    it.def_method(&ap, "copyWithin", 2, |i, this, args| {
        let o = arr_to_object(i, &this)?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.to_length(&o))? as i64;
        let target = norm_index(ab(i.to_number(&arg(args, 0)))?, len);
        let start = norm_index(ab(i.to_number(&arg(args, 1)))?, len);
        let end = match arg(args, 2) {
            Value::Undefined => len,
            v => norm_index(ab(i.to_number(&v))?, len),
        };
        // Copy element by element (backwards when the ranges overlap that way), with live
        // HasProperty/Get/Set and DeletePropertyOrThrow for source holes.
        let mut count = (end - start).min(len - target).max(0);
        let (mut from, mut to, dir) = if start < target && target < start + count {
            (start + count - 1, target + count - 1, -1i64)
        } else {
            (start, target, 1i64)
        };
        while count > 0 {
            let tk = to.to_string();
            if let Some(v) = array_get_present_index(i, &o, &ov, from as usize)? {
                set_throw(i, &ov, &tk, v)?;
            } else {
                delete_or_throw(i, &ov, &tk)?;
            }
            from += dir;
            to += dir;
            count -= 1;
        }
        Ok(ov)
    });
    it.def_method(&ap, "values", 0, nf_array_values);
    // %Array.prototype.values%: arguments objects install the intrinsic itself as @@iterator
    // (ECMA-262 CreateUnmappedArgumentsObject / CreateMappedArgumentsObject), not whatever the
    // property holds later.
    let values_intrinsic = ap.borrow().props.get("values").map(|p| p.value());
    if let Some(Value::Obj(values)) = values_intrinsic {
        Rc::make_mut(&mut it.extra_protos).insert("%Array.prototype.values%", values);
    }
    it.def_method(&ap, "keys", 0, |i, this, _| {
        arr_require_coercible(i, &this)?;
        Ok(make_array_iterator(i, this, 1))
    });
    it.def_method(&ap, "entries", 0, |i, this, _| {
        arr_require_coercible(i, &this)?;
        Ok(make_array_iterator(i, this, 2))
    });
    // `arr[Symbol.iterator]` is `Array.prototype.values`.
    if let Some(sym) = it.iterator_sym.clone() {
        let values_fn = ap.borrow().props.get("values").map(|p| p.value()).unwrap();
        ap.borrow_mut()
            .props
            .insert(Interp::sym_key(&sym), Property::builtin(values_fn));
    }

    let ctor = it.make_native("Array", 1, nf_array_ctor);
    it.array_ctor = Some(ctor.clone());
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(ap.clone()), false, false, false),
    );
    ap.borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
    it.def_method(&ctor, "isArray", 1, |i, _this, args| {
        // IsArray unwraps Proxies: a proxy of an array is an array (a revoked proxy throws).
        let mut v = arg(args, 0);
        loop {
            match &v {
                Value::Obj(o) if matches!(o.borrow().exotic, Exotic::Array) => {
                    return Ok(Value::Bool(true));
                }
                Value::Obj(_) => match proxy_pair(i, &v) {
                    Some((_, Value::Null)) => {
                        return Err(i.make_error("TypeError", "proxy is revoked"));
                    }
                    Some((target, _)) => v = target,
                    None => return Ok(Value::Bool(false)),
                },
                _ => return Ok(Value::Bool(false)),
            }
        }
    });
    it.def_method(&ctor, "of", 0, |i, this, args| {
        // If `this` is a constructor, build the result via `new this(len)`; else a plain Array.
        let len = args.len();
        // For the untouched intrinsic constructor, ArrayCreate plus the ordered
        // CreateDataPropertyOrThrow loop has the same result as the engine's dense allocation.
        // Keep the generic path for subclasses, replaced `Array.prototype`, and every other
        // constructor because GetPrototypeFromConstructor and user code remain observable
        // (ECMA-262 §23.1.2.4).
        if intrinsic_array_constructor_is_default(i, &this) {
            return Ok(i.make_array(args.to_vec()));
        }
        let arr = if is_constructor_value(&this) {
            ab(i.construct(this.clone(), &[Value::Num(len as f64)]))?
        } else {
            i.make_array(Vec::new())
        };
        for (k, v) in args.iter().enumerate() {
            // CreateDataPropertyOrThrow(A, k, v).
            if let Value::Obj(o) = &arr {
                let desc = i.new_object();
                set_data(&desc, "value", v.clone());
                set_data(&desc, "writable", Value::Bool(true));
                set_data(&desc, "enumerable", Value::Bool(true));
                set_data(&desc, "configurable", Value::Bool(true));
                if !ab(define_own_property(i, o, &k.to_string(), &Value::Obj(desc)))? {
                    return Err(i.make_error("TypeError", "Array.of: cannot define property"));
                }
            }
        }
        ab(i.set_member(&arr, "length", Value::Num(len as f64)))?;
        Ok(arr)
    });
    it.def_method(&ctor, "from", 1, |i, this, args| {
        let source = arg(args, 0);
        let mapfn = arg(args, 1);
        let this_arg = arg(args, 2);
        if !matches!(mapfn, Value::Undefined) && !mapfn.is_callable() {
            return Err(i.make_error("TypeError", "Array.from: mapFn is not callable"));
        }
        let mapfn = mapfn
            .is_callable()
            .then(|| crate::callback::Callback::new(mapfn));
        // GetMethod(items, @@iterator): a throwing getter propagates; non-callable non-nullish
        // is a TypeError.
        let iter_method = match well_known_key(i, "iterator") {
            Some(k) if !matches!(source, Value::Undefined | Value::Null) => {
                ab(i.get_member(&source, &k))?
            }
            _ => Value::Undefined,
        };
        if !matches!(iter_method, Value::Undefined | Value::Null) && !iter_method.is_callable() {
            return Err(i.make_error("TypeError", "@@iterator is not callable"));
        }
        let use_ctor = i.value_is_constructor(&this);
        if iter_method.is_callable() {
            // With the intrinsic Array constructor and iterator, a fully dense own-data source
            // has no per-element observable calls. Build the identical result in one pass rather
            // than allocating an iterator result and a property descriptor for every element.
            if mapfn.is_none()
                && is_intrinsic_array_constructor(i, &this)
                && is_native_function(&iter_method, nf_array_values)
                && intrinsic_array_iterator_is_unmodified(i)
            {
                if let Some(items) = dense_array_snapshot(i, &source) {
                    return Ok(i.make_array(items));
                }
            }
            // Iterator path: the target is constructed BEFORE iteration (constructor errors
            // propagate as-is), then elements are defined one at a time; a map/define failure
            // closes the iterator.
            let arr = if use_ctor {
                ab(i.construct(this, &[]))?
            } else {
                i.make_array(Vec::new())
            };
            let iter = ab(i.call(iter_method, source.clone(), &[]))?;
            let next = crate::callback::Callback::new(ab(i.get_member(&iter, "next"))?);
            let mut k = 0u64;
            loop {
                let Some(raw) = ab(next.iterator_step(i, &iter))? else {
                    break;
                };
                let step = (|i: &mut Interp| -> Result<(), Value> {
                    let v = if let Some(mapfn) = &mapfn {
                        ab(mapfn.call(i, this_arg.clone(), [raw, Value::Num(k as f64)]))?
                    } else {
                        raw
                    };
                    cdp_index_or_throw(i, &arr, k, v)
                })(i);
                if let Err(e) = step {
                    i.iterator_close(&iter);
                    return Err(e);
                }
                k += 1;
            }
            set_length_throw(i, &arr, k as f64)?;
            return Ok(arr);
        }
        // Array-like path: ToObject, read length, construct with «len», then per-index Get
        // (lazily, so mutations during mapping are observed).
        let o = to_object_arg(i, source.clone(), "Array.from")?;
        let ov = Value::Obj(o.clone());
        let len = ab(i.to_length(&o))?;
        let arr = if use_ctor {
            ab(i.construct(this, &[Value::Num(len as f64)]))?
        } else {
            if len as u64 > 4294967295 {
                return Err(i.make_error("RangeError", "invalid array length"));
            }
            i.make_array(Vec::new())
        };
        for k in 0..len {
            let raw = array_get_index(i, &o, &ov, k)?;
            let v = if let Some(mapfn) = &mapfn {
                ab(mapfn.call(i, this_arg.clone(), [raw, Value::Num(k as f64)]))?
            } else {
                raw
            };
            cdp_index_or_throw(i, &arr, k as u64, v)?;
        }
        set_length_throw(i, &arr, len as f64)?;
        Ok(arr)
    });
    it.def_method(&ctor, "fromAsync", 1, array_from_async);
    install_species(it, &ctor);
    set_builtin(&it.global, "Array", Value::Obj(ctor));
}

pub(crate) fn nf_array_ctor(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let a = if args.len() == 1 && matches!(args[0], Value::Num(_)) {
        // `new Array(len)` sets length without materializing elements; the length setter
        // validates that it is a valid uint32 (else RangeError: Invalid array length).
        let a = i.make_array(Vec::new());
        ab(i.set_member(&a, "length", args[0].clone()))?;
        #[cfg(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        ))]
        if let (Value::Obj(o), Value::Num(n)) = (&a, &args[0]) {
            // Tiny fixed-length work arrays are commonly filled immediately. Keyless Empty
            // slots preserve hole semantics while avoiding four separate key/entry
            // allocations for cases such as `new Array(4)`.
            o.borrow_mut().props.reserve_small_holes(*n as usize);
        }
        a
    } else {
        i.make_array(args.to_vec())
    };
    // GetPrototypeFromConstructor: a subclass / cross-realm new.target redirects the
    // instance prototype (falling back to new.target's realm's %Array.prototype%).
    if i.constructing {
        let nt = i.new_target.clone();
        if let Value::Obj(_) = &nt {
            let proto = match ab(i.get_member(&nt, "prototype"))? {
                Value::Obj(p) => Some(p),
                _ => ctor_realm_proto(i, &nt, "Array")?,
            };
            if let (Value::Obj(o), Some(p)) = (&a, proto) {
                if !Rc::ptr_eq(&p, &i.array_proto) {
                    o.borrow_mut().proto = Some(p);
                }
            }
        }
    }
    Ok(a)
}

/// Read an indexed array-like element using the exact generic `Get` fallback.
///
/// The dense probe is valid only for an own data property on an ordinary Array/plain object;
/// misses deliberately stringify the index and dispatch through `[[Get]]`, preserving holes,
/// accessors, prototypes, proxies, host indexed objects, and other exotics. This is the same
/// observable operation required by ECMA-262 §23.1.3 (Array methods), with the allocation-free
/// path only replacing an unobservable own-data lookup.
fn array_get_index(
    i: &mut Interp,
    object: &Gc,
    receiver: &Value,
    index: usize,
) -> Result<Value, Value> {
    if let Some(value) = i.fast_get_elem(object, index as f64) {
        return Ok(value);
    }
    let key = index.to_string();
    ab(i.get_member(receiver, &key))
}

/// `Get` after a caller has already performed the method's required `HasProperty` operation.
/// Keeping the key supplied by the caller preserves the mandated order while avoiding a second
/// decimal-index allocation on dense own-data arrays.
#[inline]
fn array_get_index_after_has(
    i: &mut Interp,
    object: &Gc,
    receiver: &Value,
    index: usize,
    key: &str,
) -> Result<Value, Value> {
    if let Some(value) = i.fast_get_elem(object, index as f64) {
        return Ok(value);
    }
    ab(i.get_member(receiver, key))
}

/// Read an index for the HasProperty-based Array callbacks. An own dense data element has no
/// observable `HasProperty`/`Get` work, so the existing checked element probe can answer both in
/// one operation. Holes, accessors, inherited values, proxies, and other exotics retain the exact
/// two-step generic algorithm from ECMA-262 §23.1.3.
fn array_get_present_index(
    i: &mut Interp,
    object: &Gc,
    receiver: &Value,
    index: usize,
) -> Result<Option<Value>, Value> {
    if let Some(value) = i.fast_get_elem(object, index as f64) {
        // A successful numeric TypedArray Get can return undefined for a detached or
        // out-of-bounds index. That is not a present element. Array callback/search methods
        // require HasProperty first (ECMA-262 §23.1.3; TypedArray.[[HasProperty]] and
        // IsValidIntegerIndex, snapshot e28783d5fc9d). Valid numeric elements cannot contain
        // undefined; ordinary own properties containing undefined must still be visited.
        if matches!(value, Value::Undefined)
            && i.typed_arrays.contains_key(&(Rc::as_ptr(object) as usize))
        {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    let key = index.to_string();
    if !ab(i.js_has_property(receiver, &key))? {
        return Ok(None);
    }
    Ok(Some(array_get_index_after_has(
        i, object, receiver, index, &key,
    )?))
}

/// FlattenIntoArray: write `source`'s (mapped, one-level-per-depth flattened) elements onto
/// `target` starting at `start`, via CreateDataPropertyOrThrow. Returns the next target index.
#[allow(clippy::too_many_arguments)]
fn flatten_into(
    i: &mut Interp,
    target: &Value,
    source: &Value,
    source_len: usize,
    start: usize,
    depth: i64,
    mapper: Option<&crate::callback::Callback>,
    mapper_this: &Value,
) -> Result<usize, Value> {
    // FlattenIntoArray uses HasProperty/Get for every source index. If no mapper is present and
    // both source and target are distinct ordinary Arrays, a dense own-data snapshot proves those
    // operations cannot execute JavaScript; recurse through the same algorithm for nested arrays.
    // A non-ordinary target, proxy, hole, accessor, or alias deliberately stays on the generic path
    // because target writes could otherwise affect later source Gets (ECMA-262 §23.1.3.13.1).
    if mapper.is_none()
        && source_len <= MAX_ARRAY_OP_LEN
        && flatten_dense_target_safe(i, target, source)
    {
        if let Some(values) = dense_array_snapshot(i, source) {
            let mut target_index = start;
            for element in values {
                if depth > 0 && json_is_array(i, &element)? {
                    let len_val = ab(i.get_member(&element, "length"))?;
                    let el_len = to_length_val(i, &len_val)?;
                    target_index = flatten_into(
                        i,
                        target,
                        &element,
                        el_len,
                        target_index,
                        depth - 1,
                        None,
                        &Value::Undefined,
                    )?;
                } else {
                    if target_index as u64 >= 9_007_199_254_740_991 {
                        return Err(
                            i.make_error("TypeError", "flattened array length exceeds 2^53 - 1")
                        );
                    }
                    cdp_index_or_throw(i, target, target_index as u64, element)?;
                    target_index += 1;
                }
            }
            return Ok(target_index);
        }
    }
    let mut target_index = start;
    for k in 0..source_len {
        let key = k.to_string();
        if !ab(i.js_has_property(source, &key))? {
            continue; // FlattenIntoArray skips holes
        }
        let mut element = match source {
            Value::Obj(object) => array_get_index_after_has(i, object, source, k, &key)?,
            _ => ab(i.get_member(source, &key))?,
        };
        if let Some(m) = mapper {
            element = ab(m.call(
                i,
                mapper_this.clone(),
                [element, Value::Num(k as f64), source.clone()],
            ))?;
        }
        if depth > 0 && json_is_array(i, &element)? {
            let len_val = ab(i.get_member(&element, "length"))?;
            let el_len = to_length_val(i, &len_val)?;
            target_index = flatten_into(
                i,
                target,
                &element,
                el_len,
                target_index,
                depth - 1,
                None,
                &Value::Undefined,
            )?;
        } else {
            // Compared as u64: usize is 32-bit on wasm32, where 2^53 - 1 overflows the type.
            if target_index as u64 >= 9_007_199_254_740_991 {
                return Err(i.make_error("TypeError", "flattened array length exceeds 2^53 - 1"));
            }
            cdp_index_or_throw(i, target, target_index as u64, element)?;
            target_index += 1;
        }
    }
    Ok(target_index)
}

fn flatten_dense_target_safe(i: &Interp, target: &Value, source: &Value) -> bool {
    let (Value::Obj(target), Value::Obj(source)) = (target, source) else {
        return false;
    };
    !Rc::ptr_eq(target, source)
        && matches!(target.borrow().exotic, Exotic::Array)
        && i.ordinary_get_ptr(Rc::as_ptr(target) as usize)
}

fn array_splice(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    let o = arr_to_object(i, &this)?;
    let ov = Value::Obj(o.clone());
    // The total length may be near 2^53 for an array-like; only the elements actually moved are
    // bounded by the engine's materialization cap.
    let len = ab(i.to_length(&o))? as i64;
    let start = norm_index(ab(i.to_number(&arg(args, 0)))?, len);
    let delete_count = if args.is_empty() {
        0
    } else if args.len() < 2 {
        len - start
    } else {
        (ab(i.to_number(&arg(args, 1)))? as i64).clamp(0, len - start)
    };
    let items: Vec<Value> = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        Vec::new()
    };
    // A result length past 2^53-1 is a TypeError (before any mutation or species construction).
    if (len - delete_count + items.len() as i64) as u64 > 9007199254740991 {
        return Err(i.make_error("TypeError", "splice result is too long"));
    }
    // The removed array (ArraySpeciesCreate) preserves holes via HasProperty.
    let removed = array_species_create(i, &this, delete_count.max(0) as u64)?;
    for k in 0..delete_count {
        let from = (start + k).to_string();
        if ab(i.js_has_property(&ov, &from))? {
            let v = array_get_index_after_has(i, &o, &ov, (start + k) as usize, &from)?;
            cdp_index_or_throw(i, &removed, k as u64, v)?;
        }
    }
    ab(i.set_member(&removed, "length", Value::Num(delete_count as f64)))?;
    let item_count = items.len() as i64;
    // Shift the trailing elements (preserving holes) to open or close the gap.
    if item_count < delete_count {
        for k in start..(len - delete_count) {
            let from = (k + delete_count).to_string();
            let to = (k + item_count).to_string();
            if ab(i.js_has_property(&ov, &from))? {
                let v = array_get_index_after_has(i, &o, &ov, (k + delete_count) as usize, &from)?;
                set_throw(i, &ov, &to, v)?;
            } else {
                delete_or_throw(i, &ov, &to)?;
            }
        }
        for k in ((len - delete_count + item_count)..len).rev() {
            delete_or_throw(i, &ov, &k.to_string())?;
        }
    } else if item_count > delete_count {
        for k in ((start + 1)..=(len - delete_count)).rev() {
            let from = (k + delete_count - 1).to_string();
            let to = (k + item_count - 1).to_string();
            if ab(i.js_has_property(&ov, &from))? {
                let v =
                    array_get_index_after_has(i, &o, &ov, (k + delete_count - 1) as usize, &from)?;
                set_throw(i, &ov, &to, v)?;
            } else {
                delete_or_throw(i, &ov, &to)?;
            }
        }
    }
    for (off, v) in items.iter().enumerate() {
        set_throw(i, &ov, &(start + off as i64).to_string(), v.clone())?;
    }
    set_throw(
        i,
        &ov,
        "length",
        Value::Num((len - delete_count + item_count) as f64),
    )?;
    Ok(removed)
}

fn merge_sort(i: &mut Interp, items: &mut [Value], cmp: &Value) -> Result<(), Value> {
    let n = items.len();
    if n <= 1 {
        return Ok(());
    }

    // CompareArrayElements with an absent comparator calls ToString for each pair
    // (ECMA-262 §23.1.3.30.2). Primitive values have no user code in ToString, so precompute
    // their stable sort keys once. Objects, Symbols, and internal values retain the complete
    // comparison path because coercion can throw or execute observable code.
    if !cmp.is_callable() {
        if let Some(keys) = default_sort_keys(i, items)? {
            return merge_sort_with_keys(items, keys);
        }
    }
    let cmp = cmp
        .is_callable()
        .then(|| crate::callback::Callback::new(cmp.clone()));

    // SortIndexedProperties permits any stable implementation-defined comparison sequence
    // (ECMA-262 §23.1.3.30.1). Keep one source and one destination buffer for all merge passes;
    // the previous recursive implementation allocated two fresh vectors at every tree level.
    // Values move between the buffers, so the only per-sort allocation is O(n), while an abrupt
    // comparator completion still leaves the original array untouched because `items` is copied
    // into `source` before sorting begins.
    let mut source = items.to_vec();
    let mut destination: Vec<Value> = (0..n).map(|_| Value::Undefined).collect();
    let mut width = 1usize;
    loop {
        let mut start = 0usize;
        while start < n {
            let mid = start.saturating_add(width).min(n);
            let end = mid.saturating_add(width).min(n);
            let (mut left, mut right, mut out) = (start, mid, start);
            while left < mid && right < end {
                let take_left = compare_values(i, cmp.as_ref(), &source[left], &source[right])?
                    != Ordering::Greater;
                let selected = if take_left {
                    let value = std::mem::replace(&mut source[left], Value::Undefined);
                    left += 1;
                    value
                } else {
                    let value = std::mem::replace(&mut source[right], Value::Undefined);
                    right += 1;
                    value
                };
                destination[out] = selected;
                out += 1;
            }
            while left < mid {
                destination[out] = std::mem::replace(&mut source[left], Value::Undefined);
                left += 1;
                out += 1;
            }
            while right < end {
                destination[out] = std::mem::replace(&mut source[right], Value::Undefined);
                right += 1;
                out += 1;
            }
            start = end;
        }
        std::mem::swap(&mut source, &mut destination);
        if width >= n.div_ceil(2) {
            break;
        }
        width = width.saturating_mul(2);
    }
    for (slot, value) in items.iter_mut().zip(source) {
        *slot = value;
    }
    Ok(())
}

fn default_sort_keys(i: &mut Interp, items: &[Value]) -> Result<Option<Vec<Option<LStr>>>, Value> {
    let mut keys = Vec::with_capacity(items.len());
    for value in items {
        match value {
            Value::Undefined => keys.push(None),
            Value::Bool(_) | Value::Null | Value::Num(_) | Value::Str(_) | Value::BigInt(_) => {
                keys.push(Some(ab(i.to_string(value))?));
            }
            // Objects may invoke user-defined coercion; Symbols throw from ToString. Preserve
            // the ordinary comparator so its exact completion behavior remains observable.
            _ => return Ok(None),
        }
    }
    Ok(Some(keys))
}

fn merge_sort_with_keys(items: &mut [Value], keys: Vec<Option<LStr>>) -> Result<(), Value> {
    let n = items.len();
    debug_assert_eq!(n, keys.len());
    let mut source: Vec<(Value, Option<LStr>)> = items.iter().cloned().zip(keys).collect();
    let mut destination: Vec<(Value, Option<LStr>)> =
        (0..n).map(|_| (Value::Undefined, None)).collect();
    let mut width = 1usize;
    loop {
        let mut start = 0usize;
        while start < n {
            let mid = start.saturating_add(width).min(n);
            let end = mid.saturating_add(width).min(n);
            let (mut left, mut right, mut out) = (start, mid, start);
            while left < mid && right < end {
                let take_left =
                    compare_sort_keys(&source[left].1, &source[right].1) != Ordering::Greater;
                let selected = if take_left {
                    let value = std::mem::replace(&mut source[left], (Value::Undefined, None));
                    left += 1;
                    value
                } else {
                    let value = std::mem::replace(&mut source[right], (Value::Undefined, None));
                    right += 1;
                    value
                };
                destination[out] = selected;
                out += 1;
            }
            while left < mid {
                destination[out] = std::mem::replace(&mut source[left], (Value::Undefined, None));
                left += 1;
                out += 1;
            }
            while right < end {
                destination[out] = std::mem::replace(&mut source[right], (Value::Undefined, None));
                right += 1;
                out += 1;
            }
            start = end;
        }
        std::mem::swap(&mut source, &mut destination);
        if width >= n.div_ceil(2) {
            break;
        }
        width = width.saturating_mul(2);
    }
    for (slot, (value, _)) in items.iter_mut().zip(source) {
        *slot = value;
    }
    Ok(())
}

#[inline]
fn compare_sort_keys(a: &Option<LStr>, b: &Option<LStr>) -> Ordering {
    // `None` is the precomputed marker for undefined, which CompareArrayElements always moves to
    // the end. Equal keys use <= at the merge site, preserving stable ordering.
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.as_ref().cmp(b.as_ref()),
    }
}

fn compare_values(
    i: &mut Interp,
    cmp: Option<&crate::callback::Callback>,
    a: &Value,
    b: &Value,
) -> Result<Ordering, Value> {
    // `undefined` always sorts to the end.
    match (matches!(a, Value::Undefined), matches!(b, Value::Undefined)) {
        (true, true) => return Ok(Ordering::Equal),
        (true, false) => return Ok(Ordering::Greater),
        (false, true) => return Ok(Ordering::Less),
        _ => {}
    }
    if let Some(cmp) = cmp {
        let r = ab(cmp.call(i, Value::Undefined, [a.clone(), b.clone()]))?;
        let n = ab(i.to_number(&r))?;
        Ok(if n < 0.0 {
            Ordering::Less
        } else if n > 0.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        })
    } else {
        let sa = ab(i.to_string(a))?;
        let sb = ab(i.to_string(b))?;
        Ok(sa.as_ref().cmp(sb.as_ref()))
    }
}

/// A native that returns its `this` — the `@@iterator` of an iterator object is itself.
fn return_this(_i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    Ok(this)
}

fn install_iterator(it: &mut Interp) {
    // %IteratorPrototype%: the common prototype of all built-in iterators; `[@@iterator]()` is the
    // identity function so an iterator is itself iterable.
    let proto = Object::new(Some(it.object_proto.clone()));
    Rc::make_mut(&mut it.extra_protos).insert("%IteratorPrototype%", proto.clone());
    if let Some(sym) = it.iterator_sym.clone() {
        let f = it.make_native("[Symbol.iterator]", 0, return_this);
        proto
            .borrow_mut()
            .props
            .insert(Interp::sym_key(&sym), Property::builtin(Value::Obj(f)));
    }
    // Iterator-helper methods on %IteratorPrototype%.
    it.def_method(&proto, "map", 1, |i, t, a| {
        make_iter_helper(i, t, "map", arg(a, 0))
    });
    it.def_method(&proto, "filter", 1, |i, t, a| {
        make_iter_helper(i, t, "filter", arg(a, 0))
    });
    it.def_method(&proto, "take", 1, |i, t, a| {
        make_iter_helper(i, t, "take", arg(a, 0))
    });
    it.def_method(&proto, "drop", 1, |i, t, a| {
        make_iter_helper(i, t, "drop", arg(a, 0))
    });
    it.def_method(&proto, "chunks", 1, |i, t, a| {
        make_chunking_helper(i, t, arg(a, 0), None)
    });
    it.def_method(&proto, "windows", 1, |i, t, a| {
        make_chunking_helper(i, t, arg(a, 0), Some(arg(a, 1)))
    });
    it.def_method(&proto, "includes", 1, iterator_includes);
    it.def_method(&proto, "join", 1, iterator_join);
    it.def_method(&proto, "flatMap", 1, |i, t, a| {
        make_iter_helper(i, t, "flatMap", arg(a, 0))
    });
    it.def_method(&proto, "toArray", 0, |i, this, _| {
        require_iterator_object(i, &this)?;
        let next = ab(i.get_member(&this, "next"))?;
        let mut out = Vec::new();
        while let Some(v) = step_iter_with(i, &this, &next)? {
            out.push(v);
        }
        Ok(i.make_array(out))
    });
    it.def_method(&proto, "forEach", 1, |i, this, a| {
        require_iterator_object(i, &this)?;
        let f = arg(a, 0);
        if !f.is_callable() {
            // Closes the underlying iterator without reading `next`; the TypeError wins.
            i.iterator_close(&this);
            return Err(i.make_error(
                "TypeError",
                "Iterator.prototype.forEach argument is not callable",
            ));
        }
        let next = ab(i.get_member(&this, "next"))?;
        let mut k = 0.0;
        let f = crate::callback::Callback::new(f);
        while let Some(v) = step_iter_with(i, &this, &next)? {
            if let Err(e) = f.call(i, Value::Undefined, [v, Value::Num(k)]) {
                i.iterator_close(&this);
                return Err(crate::interpreter::abrupt_value(e));
            }
            k += 1.0;
        }
        Ok(Value::Undefined)
    });
    it.def_method(&proto, "reduce", 1, |i, this, a| {
        require_iterator_object(i, &this)?;
        let f = arg(a, 0);
        if !f.is_callable() {
            // Closes the underlying iterator without reading `next`; the TypeError wins.
            i.iterator_close(&this);
            return Err(i.make_error("TypeError", "reducer is not callable"));
        }
        let next = ab(i.get_member(&this, "next"))?;
        let mut acc;
        let mut k = 0.0;
        if a.len() >= 2 {
            acc = arg(a, 1);
        } else {
            acc = match step_iter_with(i, &this, &next)? {
                Some(v) => v,
                None => {
                    return Err(i.make_error(
                        "TypeError",
                        "Reduce of empty iterator with no initial value",
                    ));
                }
            };
            k = 1.0;
        }
        let f = crate::callback::Callback::new(f);
        while let Some(v) = step_iter_with(i, &this, &next)? {
            acc = match f.call(i, Value::Undefined, [acc, v, Value::Num(k)]) {
                Ok(r) => r,
                Err(e) => {
                    i.iterator_close(&this);
                    return Err(crate::interpreter::abrupt_value(e));
                }
            };
            k += 1.0;
        }
        Ok(acc)
    });
    it.def_method(&proto, "some", 1, |i, t, a| iter_some_every(i, t, a, true));
    it.def_method(&proto, "every", 1, |i, t, a| {
        iter_some_every(i, t, a, false)
    });
    it.def_method(&proto, "find", 1, |i, this, a| {
        require_iterator_object(i, &this)?;
        let f = arg(a, 0);
        if !f.is_callable() {
            // Closes the underlying iterator without reading `next`; the TypeError wins.
            i.iterator_close(&this);
            return Err(i.make_error("TypeError", "predicate is not callable"));
        }
        let next = ab(i.get_member(&this, "next"))?;
        let mut k = 0.0;
        let f = crate::callback::Callback::new(f);
        while let Some(v) = step_iter_with(i, &this, &next)? {
            let r = match f.call(i, Value::Undefined, [v.clone(), Value::Num(k)]) {
                Ok(r) => r,
                Err(e) => {
                    i.iterator_close(&this);
                    return Err(crate::interpreter::abrupt_value(e));
                }
            };
            if i.to_boolean(&r) {
                ab(i.iterator_close_normal(&this))?;
                return Ok(v);
            }
            k += 1.0;
        }
        Ok(Value::Undefined)
    });

    // Iterator.prototype[@@dispose]: close the iterator (calls its return method).
    if let Some(key) = well_known_key(it, "dispose") {
        let disp = it.make_native("[Symbol.dispose]", 0, |i, this, _a| {
            i.iterator_close(&this);
            Ok(Value::Undefined)
        });
        proto
            .borrow_mut()
            .props
            .insert(key, Property::builtin(Value::Obj(disp)));
    }

    // %WrapForValidIteratorPrototype%: the prototype of Iterator.from's wrappers.
    {
        let wrap_proto = Object::new(Some(proto.clone()));
        fn wrap_slot(this: &Value, key: &str) -> Option<Value> {
            // [[Iterated]] internal-slot access: read own properties directly (no observable get).
            let o = this.as_obj()?;
            let v = o.borrow().props.get(key).map(|p| p.value());
            v.filter(|v| !matches!(v, Value::Undefined))
        }
        it.def_method(&wrap_proto, "next", 0, |i, this, _a| {
            let Some(iter) = wrap_slot(&this, "#\u{0}wrap_iter") else {
                return Err(i.make_error("TypeError", "next called on an incompatible receiver"));
            };
            let nx = wrap_slot(&this, "#\u{0}wrap_next").unwrap_or(Value::Undefined);
            ab(i.call(nx, iter, &[]))
        });
        it.def_method(&wrap_proto, "return", 0, |i, this, _a| {
            let Some(iter) = wrap_slot(&this, "#\u{0}wrap_iter") else {
                return Err(i.make_error("TypeError", "return called on an incompatible receiver"));
            };
            // GetMethod(iterator, "return"); its result is returned as-is.
            let rm = ab(i.get_member(&iter, "return"))?;
            if matches!(rm, Value::Undefined | Value::Null) {
                return Ok(iter_result(i, Value::Undefined, true));
            }
            if !rm.is_callable() {
                return Err(i.make_error("TypeError", "iterator return is not callable"));
            }
            ab(i.call(rm, iter, &[]))
        });
        Rc::make_mut(&mut it.extra_protos).insert("%WrapForValidIteratorPrototype%", wrap_proto);
    }

    // %IteratorHelperPrototype%: the shared prototype of every map/filter/take/drop/flatMap
    // helper (so cross-realm helpers interoperate — the methods live here, not per-object).
    {
        let helper_proto = Object::new(Some(proto.clone()));
        it.def_method(&helper_proto, "next", 0, iter_helper_next);
        it.def_method(&helper_proto, "return", 0, iter_helper_return);
        set_to_string_tag(it, &helper_proto, "Iterator Helper");
        Rc::make_mut(&mut it.extra_protos).insert("%IteratorHelperPrototype%", helper_proto);
    }

    let ctor = it.make_native("Iterator", 0, |i, t, _a| {
        // Abstract: NewTarget must be present and must not be %Iterator% itself.
        let nt = i.new_target.clone();
        let is_self = match (&nt, i.extra_protos.get("%IteratorCtorMarker%")) {
            (Value::Obj(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        };
        if !i.constructing || matches!(nt, Value::Undefined) || is_self {
            return Err(i.make_error(
                "TypeError",
                "Abstract class Iterator not directly constructable",
            ));
        }
        // `super()` from a subclass arrives with the instance already created.
        if matches!(t, Value::Obj(_)) {
            return Ok(t);
        }
        // OrdinaryCreateFromConstructor: a newTarget whose `prototype` is not an object falls
        // back to the newTarget realm's %Iterator.prototype%.
        let proto = match ab(i.get_member(&nt, "prototype"))? {
            Value::Obj(p) => Some(p),
            _ => ctor_realm_proto(i, &nt, "%IteratorPrototype%")?
                .or_else(|| i.extra_protos.get("%IteratorPrototype%").cloned()),
        };
        Ok(Value::Obj(Object::new(proto)))
    });
    ctor.borrow_mut().is_constructor = true;
    Rc::make_mut(&mut it.extra_protos).insert("%IteratorCtorMarker%", ctor.clone());
    it.def_method(&ctor, "from", 1, |i, _t, a| {
        let v = arg(a, 0);
        // GetIteratorFlattenable (strings allowed): a string/iterable via @@iterator, or an iterator
        // used directly. If the result already inherits %Iterator.prototype%, return it; else wrap it.
        let iter = get_iterator_flattenable(i, &v, true)?;
        // GetIteratorDirect's `next` get happens before the OrdinaryHasInstance walk, whose
        // [[GetPrototypeOf]] a proxy observes.
        let next = ab(i.get_member(&iter, "next"))?;
        let iter_proto = i.extra_protos.get("%IteratorPrototype%").cloned();
        let mut inherits = false;
        let mut cur = iter.clone();
        loop {
            let p = js_get_prototype_of(i, &cur)?;
            match p {
                Value::Obj(pp) => {
                    if iter_proto.as_ref().is_some_and(|ip| Rc::ptr_eq(&pp, ip)) {
                        inherits = true;
                        break;
                    }
                    cur = Value::Obj(pp);
                }
                _ => break,
            }
        }
        if inherits {
            return Ok(iter);
        }
        // Wrap: a %WrapForValidIteratorPrototype% object forwarding next/return to `iter`.
        let obj = Object::new(
            i.extra_protos
                .get("%WrapForValidIteratorPrototype%")
                .cloned(),
        );
        set_builtin(&obj, "#\u{0}wrap_iter", iter);
        set_builtin(&obj, "#\u{0}wrap_next", next);
        Ok(Value::Obj(obj))
    });
    it.def_method(&ctor, "zip", 1, |i, _t, a| iterator_zip(i, a, false));
    it.def_method(&ctor, "zipKeyed", 1, |i, _t, a| iterator_zip(i, a, true));
    it.def_method(&ctor, "concat", 0, |i, _t, a| {
        // Each argument must be an object; capture its @@iterator method now, in order.
        let iter_key = i
            .iterator_sym
            .clone()
            .map(|s| Interp::sym_key(&s))
            .unwrap_or_default();
        let mut items = Vec::new();
        let mut methods = Vec::new();
        for v in a {
            if !matches!(v, Value::Obj(_)) {
                return Err(i.make_error("TypeError", "Iterator.concat argument is not an object"));
            }
            let m = ab(i.get_member(v, &iter_key))?;
            if !m.is_callable() {
                return Err(i.make_error("TypeError", "Iterator.concat argument is not iterable"));
            }
            items.push(v.clone());
            methods.push(m);
        }
        let obj = Object::new(i.extra_protos.get("%IteratorPrototype%").cloned());
        let items_arr = i.make_array(items);
        let methods_arr = i.make_array(methods);
        set_builtin(&obj, "#\u{0}cc_items", items_arr);
        set_builtin(&obj, "#\u{0}cc_methods", methods_arr);
        set_builtin(&obj, "#\u{0}cc_idx", Value::Num(0.0));
        set_builtin(&obj, "#\u{0}cc_cur", Value::Undefined);
        set_builtin(&obj, "#\u{0}cc_curnext", Value::Undefined);
        set_builtin(&obj, "#\u{0}cc_done", Value::Bool(false));
        i.def_method(&obj, "next", 0, concat_next);
        // return() closes the currently-open inner iterator (once), then reports done.
        i.def_method(&obj, "return", 0, |i, this, _a| {
            let running = ab(i.get_member(&this, "#\u{0}cc_running"))?;
            if i.to_boolean(&running) {
                return Err(
                    i.make_error("TypeError", "Iterator.concat iterator is already running")
                );
            }
            let done = ab(i.get_member(&this, "#\u{0}cc_done"))?;
            if !i.to_boolean(&done) {
                let o = this.as_obj().unwrap().clone();
                let cur = ab(i.get_member(&this, "#\u{0}cc_cur"))?;
                if matches!(cur, Value::Obj(_)) {
                    let started = ab(i.get_member(&this, "#\u{0}cc_gstarted"))?;
                    let started = i.to_boolean(&started);
                    if started {
                        // Suspended at a yield: the close runs in the executing state.
                        set_internal(&o, "#\u{0}cc_running", Value::Bool(true));
                    } else {
                        set_internal(&o, "#\u{0}cc_done", Value::Bool(true));
                    }
                    let res = i.iterator_close_normal(&cur);
                    if started {
                        set_internal(&o, "#\u{0}cc_running", Value::Bool(false));
                        set_internal(&o, "#\u{0}cc_done", Value::Bool(true));
                    }
                    ab(res)?;
                } else {
                    set_internal(&o, "#\u{0}cc_done", Value::Bool(true));
                }
            }
            Ok(iter_result(i, Value::Undefined, true))
        });
        if let Some(sym) = i.iterator_sym.clone() {
            let itf = i.make_native("[Symbol.iterator]", 0, return_this);
            obj.borrow_mut()
                .props
                .insert(Interp::sym_key(&sym), Property::builtin(Value::Obj(itf)));
        }
        Ok(Value::Obj(obj))
    });
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto.clone()), false, false, false),
    );
    // Iterator.prototype.constructor and [@@toStringTag] are accessor pairs: the getter yields
    // the value; the setter throws for the prototype itself as receiver but defines an own data
    // property on any other receiver (SetterThatIgnoresPrototypeProperties).
    {
        Rc::make_mut(&mut it.extra_protos).insert("%IteratorProtoMarker%", proto.clone());
        let getter_ctor = it.make_native("get constructor", 0, |i, _t, _a| {
            Ok(i.global
                .borrow()
                .props
                .get("Iterator")
                .map(|p| p.value())
                .unwrap_or(Value::Undefined))
        });
        let getter_tag = it.make_native("get [Symbol.toStringTag]", 0, |_i, _t, _a| {
            Ok(Value::lstr("Iterator"))
        });
        let set_ctor = it.make_native("set constructor", 1, |i, this, a| {
            iterator_proto_weird_set(i, this, arg(a, 0), "constructor")
        });
        let set_tag = it.make_native("set [Symbol.toStringTag]", 1, |i, this, a| {
            let key = to_string_tag_key(i).unwrap_or_default();
            iterator_proto_weird_set(i, this, arg(a, 0), &key)
        });
        let acc = |get: Gc, set: Gc| {
            Property::accessor_prop(Some(Value::Obj(get)), Some(Value::Obj(set)), false, true)
        };
        proto
            .borrow_mut()
            .props
            .insert("constructor", acc(getter_ctor, set_ctor));
        if let Some(tag) = to_string_tag_key(it) {
            proto
                .borrow_mut()
                .props
                .insert(tag, acc(getter_tag, set_tag));
        }
    }
    set_builtin(&it.global, "Iterator", Value::Obj(ctor));

    // %ArrayIteratorPrototype%: the intermediate prototype of Array iterators (its own [[Prototype]]
    // is %IteratorPrototype%), so getPrototypeOf(getPrototypeOf(arrIter)) lands on %IteratorPrototype%.
    let arr_iter_proto = Object::new(it.extra_protos.get("%IteratorPrototype%").cloned());
    set_to_string_tag(it, &arr_iter_proto, "Array Iterator");
    it.def_method(&arr_iter_proto, "next", 0, array_iter_next);
    Rc::make_mut(&mut it.extra_protos).insert("%ArrayIteratorPrototype%", arr_iter_proto);

    // %StringIteratorPrototype%: the prototype of `String.prototype[@@iterator]()` iterators, which
    // walk the string lazily by code point.
    let str_iter_proto = Object::new(it.extra_protos.get("%IteratorPrototype%").cloned());
    set_to_string_tag(it, &str_iter_proto, "String Iterator");
    it.def_method(&str_iter_proto, "next", 0, string_iter_next);
    Rc::make_mut(&mut it.extra_protos).insert("%StringIteratorPrototype%", str_iter_proto);

    // %RegExpStringIteratorPrototype%: the prototype of the iterator returned by
    // `RegExp.prototype[@@matchAll]` and `String.prototype.matchAll`.
    let rsi_proto = Object::new(it.extra_protos.get("%IteratorPrototype%").cloned());
    set_to_string_tag(it, &rsi_proto, "RegExp String Iterator");
    it.def_method(&rsi_proto, "next", 0, regexp_string_iterator_next);
    Rc::make_mut(&mut it.extra_protos).insert("%RegExpStringIteratorPrototype%", rsi_proto);

    // %AsyncIteratorPrototype%: [@@asyncIterator]() returns this, plus [@@asyncDispose] (which calls
    // the iterator's return()). The prototype of every built-in async iterator.
    let async_iter_proto = Object::new(Some(it.object_proto.clone()));
    if let Some(k) = well_known_key(it, "asyncIterator") {
        let f = it.make_native("[Symbol.asyncIterator]", 0, return_this);
        async_iter_proto
            .borrow_mut()
            .props
            .insert(k, Property::builtin(Value::Obj(f)));
    }
    if let Some(k) = well_known_key(it, "asyncDispose") {
        let f = it.make_native("[Symbol.asyncDispose]", 0, async_dispose_via_return);
        async_iter_proto
            .borrow_mut()
            .props
            .insert(k, Property::builtin(Value::Obj(f)));
    }
    Rc::make_mut(&mut it.extra_protos).insert("%AsyncIteratorPrototype%", async_iter_proto.clone());

    // %GeneratorPrototype% (proto %IteratorPrototype%) and %AsyncGeneratorPrototype% (proto
    // %AsyncIteratorPrototype%): the [[Prototype]] of a generator function's `.prototype`, carrying
    // next/return/throw (which brand-check the receiver via the generators table).
    let gen_proto = Object::new(it.extra_protos.get("%IteratorPrototype%").cloned());
    set_to_string_tag(it, &gen_proto, "Generator");
    it.def_method(&gen_proto, "next", 1, generator_next);
    it.def_method(&gen_proto, "return", 1, generator_return);
    it.def_method(&gen_proto, "throw", 1, generator_throw);
    // %Generator% (= %GeneratorFunction.prototype%) .prototype === %GeneratorPrototype%, and the
    // latter's .constructor points back — both { writable: false, enumerable: false, configurable: true }.
    link_generator_proto(it, "%GeneratorFunction.prototype%", &gen_proto);
    Rc::make_mut(&mut it.extra_protos).insert("%GeneratorPrototype%", gen_proto);
    let async_gen_proto = Object::new(Some(async_iter_proto));
    set_to_string_tag(it, &async_gen_proto, "AsyncGenerator");
    it.def_method(&async_gen_proto, "next", 1, async_generator_next);
    it.def_method(&async_gen_proto, "return", 1, async_generator_return);
    it.def_method(&async_gen_proto, "throw", 1, async_generator_throw);
    link_generator_proto(it, "%AsyncGeneratorFunction.prototype%", &async_gen_proto);
    Rc::make_mut(&mut it.extra_protos).insert("%AsyncGeneratorPrototype%", async_gen_proto);
}

/// Wire the `.prototype` ↔ `.constructor` pair between a `%(Async)GeneratorFunction.prototype%`
/// intrinsic (`%Generator%`/`%AsyncGenerator%`) and its instance prototype.
fn link_generator_proto(it: &mut Interp, gen_ctor_key: &str, inst_proto: &Gc) {
    let Some(gen_ctor) = it.extra_protos.get(gen_ctor_key).cloned() else {
        return;
    };
    gen_ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(inst_proto.clone()), false, false, true),
    );
    inst_proto.borrow_mut().props.insert(
        "constructor",
        Property::data(Value::Obj(gen_ctor), false, false, true),
    );
}

/// %AsyncIteratorPrototype%[@@asyncDispose]: return a promise that runs the iterator's `return()`
/// (if any) and resolves to undefined.
fn async_dispose_via_return(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    // %AsyncIteratorPrototype%[@@asyncDispose]: every failure — a throwing `return` getter, a
    // throwing call, or a rejected result — surfaces as a rejection of the returned promise, and a
    // fulfilled result is discarded (the promise fulfills with undefined).
    let promise = i.new_promise();
    let ret = match i.get_member(&this, "return") {
        Ok(v) => v,
        Err(e) => {
            let reason = crate::interpreter::abrupt_value(e);
            i.reject_promise(&promise, reason);
            return Ok(promise);
        }
    };
    if !ret.is_callable() {
        i.resolve_promise(&promise, Value::Undefined);
        return Ok(promise);
    }
    // Spec step 6.a: Call(return, O, « ») — no arguments (observable via arguments.length).
    match i.call(ret, this, &[]) {
        Ok(v) => {
            let inner = promise_resolve_value(i, v);
            let on_f = make_bound_len(i, dispose_settle_fulfil, vec![promise.clone()], 1.0);
            let on_r = make_bound_len(i, dispose_settle_reject, vec![promise.clone()], 1.0);
            i.promise_then(&inner, on_f, on_r);
        }
        Err(e) => {
            let reason = crate::interpreter::abrupt_value(e);
            i.reject_promise(&promise, reason);
        }
    }
    Ok(promise)
}

/// Reaction for [@@asyncDispose]: the awaited `return()` result fulfilled — fulfil the dispose
/// promise (`args[0]`) with undefined, dropping the value.
fn dispose_settle_fulfil(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    i.resolve_promise(&arg(args, 0), Value::Undefined);
    Ok(Value::Undefined)
}

/// Reaction for [@@asyncDispose]: the awaited `return()` result rejected — reject the dispose
/// promise (`args[0]`) with the same reason (`args[1]`).
fn dispose_settle_reject(i: &mut Interp, _t: Value, args: &[Value]) -> Result<Value, Value> {
    i.reject_promise(&arg(args, 0), arg(args, 1));
    Ok(Value::Undefined)
}

/// CreateRegExpStringIterator: a lazy iterator over a regex's matches in a string. Its state lives in
/// internal properties; `next` drives it via RegExpExec.
fn create_regexp_string_iterator(
    i: &Interp,
    matcher: Value,
    s: Rc<str>,
    global: bool,
    unicode: bool,
) -> Value {
    let obj = Object::new(
        i.extra_protos
            .get("%RegExpStringIteratorPrototype%")
            .cloned(),
    );
    set_internal(&obj, "#\u{0}rsi_regexp", matcher);
    set_internal(&obj, "#\u{0}rsi_string", Value::Str(s.into()));
    set_internal(&obj, "#\u{0}rsi_global", Value::Bool(global));
    set_internal(&obj, "#\u{0}rsi_unicode", Value::Bool(unicode));
    set_internal(&obj, "#\u{0}rsi_done", Value::Bool(false));
    Value::Obj(obj)
}

fn regexp_string_iterator_next(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = match &this {
        Value::Obj(o) if o.borrow().props.contains("#\u{0}rsi_regexp") => o.clone(),
        _ => return Err(i.make_error("TypeError", "next called on a non-RegExp-String-Iterator")),
    };
    let done = o
        .borrow()
        .props
        .get("#\u{0}rsi_done")
        .map(|p| matches!(p.value(), Value::Bool(true)))
        .unwrap_or(true);
    if done {
        return Ok(i.iter_result_obj(Value::Undefined, true));
    }
    let r = o.borrow().props.get("#\u{0}rsi_regexp").unwrap().value();
    let s = match o.borrow().props.get("#\u{0}rsi_string").map(|p| p.value()) {
        Some(Value::Str(s)) => s,
        _ => crate::lstr::LStr::from(""),
    };
    let global = matches!(
        o.borrow().props.get("#\u{0}rsi_global").map(|p| p.value()),
        Some(Value::Bool(true))
    );
    let unicode = matches!(
        o.borrow().props.get("#\u{0}rsi_unicode").map(|p| p.value()),
        Some(Value::Bool(true))
    );
    let m = regexp_exec_abstract(i, &r, s.clone())?;
    if matches!(m, Value::Null) {
        set_internal(&o, "#\u{0}rsi_done", Value::Bool(true));
        return Ok(i.iter_result_obj(Value::Undefined, true));
    }
    if global {
        let m0_v = ab(i.get_member(&m, "0"))?;
        let m0 = ab(i.to_string(&m0_v))?;
        if m0.is_empty() {
            let li_v = ab(i.get_member(&r, "lastIndex"))?;
            let li = to_length_val(i, &li_v)?;
            let next = advance_string_index(li, &s, unicode);
            ab(i.set_member(&r, "lastIndex", Value::Num(next as f64)))?;
        }
    } else {
        set_internal(&o, "#\u{0}rsi_done", Value::Bool(true));
    }
    Ok(i.iter_result_obj(m, false))
}

/// `Array.fromAsync` is specified as an async function (ECMA-262 §23.1.2.2), but it is a built-in
/// rather than source text the bytecode compiler can lower. Keep its execution context as this
/// explicit heap state machine and let the ordinary async driver park it at each normative Await.
pub(crate) struct FromAsyncCoro {
    stage: FromAsyncStage,
    started: bool,
    done: bool,
}

struct FromAsyncIter {
    array: Value,
    iterator: Value,
    next: Value,
    mapper: Value,
    this_arg: Value,
    index: u64,
    from_sync: bool,
}

struct FromAsyncArrayLike {
    array: Value,
    source: Value,
    mapper: Value,
    this_arg: Value,
    index: usize,
    length: usize,
}

enum FromAsyncStage {
    Start {
        ctor: Value,
        source: Value,
        mapper: Value,
        this_arg: Value,
    },
    IteratorNext(FromAsyncIter),
    IteratorResult(FromAsyncIter),
    IteratorMapped(FromAsyncIter),
    ArrayLikeNext(FromAsyncArrayLike),
    ArrayLikeValue(FromAsyncArrayLike),
    ArrayLikeMapped(FromAsyncArrayLike),
    Closing {
        error: Value,
        iterator: Value,
    },
    Done,
}

impl FromAsyncCoro {
    fn visit_values(&self, mut visit: impl FnMut(&Value)) {
        let values: &[&Value] = match &self.stage {
            FromAsyncStage::Start {
                ctor,
                source,
                mapper,
                this_arg,
            } => &[ctor, source, mapper, this_arg],
            FromAsyncStage::IteratorNext(state)
            | FromAsyncStage::IteratorResult(state)
            | FromAsyncStage::IteratorMapped(state) => {
                let FromAsyncIter {
                    array,
                    iterator,
                    next,
                    mapper,
                    this_arg,
                    index: _,
                    from_sync: _,
                } = state;
                &[array, iterator, next, mapper, this_arg]
            }
            FromAsyncStage::ArrayLikeNext(state)
            | FromAsyncStage::ArrayLikeValue(state)
            | FromAsyncStage::ArrayLikeMapped(state) => {
                let FromAsyncArrayLike {
                    array,
                    source,
                    mapper,
                    this_arg,
                    index: _,
                    length: _,
                } = state;
                &[array, source, mapper, this_arg]
            }
            FromAsyncStage::Closing { error, iterator } => &[error, iterator],
            FromAsyncStage::Done => &[],
        };
        for value in values {
            visit(value);
        }
    }

    pub(crate) fn trace_gc(&self, edges: &mut crate::gc_edges::DirectGcEdges<'_>) {
        self.visit_values(|value| edges.value(value));
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut crate::memory::Visitor) -> usize {
        self.visit_values(|value| visitor.value(value));
        std::mem::size_of::<FromAsyncCoro>()
    }

    fn new(ctor: Value, source: Value, mapper: Value, this_arg: Value) -> Self {
        Self {
            stage: FromAsyncStage::Start {
                ctor,
                source,
                mapper,
                this_arg,
            },
            started: false,
            done: false,
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.done
    }

    pub(crate) fn started(&self) -> bool {
        self.started
    }

    pub(crate) fn terminate(&mut self) {
        self.stage = FromAsyncStage::Done;
        self.done = true;
    }

    fn finish(&mut self, result: crate::coroutine::Suspend) -> crate::coroutine::Suspend {
        self.stage = FromAsyncStage::Done;
        self.done = true;
        result
    }

    fn close_after_error(
        &mut self,
        i: &mut Interp,
        state: FromAsyncIter,
        error: Value,
    ) -> crate::coroutine::Suspend {
        use crate::coroutine::Suspend;
        let iterator = state.iterator;
        if state.from_sync {
            // The iterator record is the spec's async-from-sync wrapper. Its `return()` always
            // returns a promise, and AsyncIteratorClose awaits that promise even though the
            // original throw completion ultimately wins.
            let awaited = from_async_sync_close(i, &iterator);
            self.stage = FromAsyncStage::Closing { error, iterator };
            return Suspend::Await(awaited);
        }

        // AsyncIteratorClose(iteratorRecord, ThrowCompletion(error)), ECMA-262 §7.4.15. An abrupt
        // return getter/call is discarded in favour of the original throw; only a normal call
        // result introduces the required Await.
        let ret = match i.get_member(&iterator, "return") {
            Ok(value) => value,
            Err(_) => return self.finish(Suspend::Throw(error)),
        };
        if matches!(ret, Value::Undefined | Value::Null) {
            return self.finish(Suspend::Throw(error));
        }
        if !ret.is_callable() {
            return self.finish(Suspend::Throw(error));
        }
        let awaited = match i.call(ret, iterator.clone(), &[]) {
            Ok(value) => value,
            Err(_) => return self.finish(Suspend::Throw(error)),
        };
        self.stage = FromAsyncStage::Closing { error, iterator };
        Suspend::Await(awaited)
    }

    pub(crate) fn resume(
        &mut self,
        i: &mut Interp,
        signal: crate::coroutine::Resume,
    ) -> crate::coroutine::Suspend {
        use crate::coroutine::{Resume, Suspend};
        if self.done {
            return Suspend::Done(Value::Undefined);
        }
        if matches!(signal, Resume::Terminate) {
            self.terminate();
            return Suspend::Done(Value::Undefined);
        }
        self.started = true;
        let mut input = Some(signal);

        loop {
            let stage = std::mem::replace(&mut self.stage, FromAsyncStage::Done);
            match stage {
                FromAsyncStage::Start {
                    ctor,
                    source,
                    mapper,
                    this_arg,
                } => {
                    // The initial drive is AsyncFunctionStart rather than an Await resumption.
                    let _ = input.take();
                    match from_async_start(i, ctor, source, mapper, this_arg) {
                        Ok(stage) => self.stage = stage,
                        Err(error) => return self.finish(Suspend::Throw(error)),
                    }
                }
                FromAsyncStage::IteratorNext(state) => {
                    if state.index >= 9_007_199_254_740_991 {
                        let error =
                            i.make_error("TypeError", "Array.fromAsync result is too large");
                        return self.close_after_error(i, state, error);
                    }
                    let awaited = if state.from_sync {
                        from_async_sync_next(i, &state.iterator, &state.next)
                    } else {
                        match i.call(state.next.clone(), state.iterator.clone(), &[]) {
                            Ok(value) => value,
                            Err(error) => {
                                return self.finish(Suspend::Throw(
                                    crate::interpreter::abrupt_value(error),
                                ));
                            }
                        }
                    };
                    self.stage = FromAsyncStage::IteratorResult(state);
                    return Suspend::Await(awaited);
                }
                FromAsyncStage::IteratorResult(mut state) => {
                    let result = match input.take().expect("an Await stage resumes with a signal") {
                        Resume::Next(value) | Resume::Return(value) => value,
                        Resume::Throw(error) => return self.finish(Suspend::Throw(error)),
                        Resume::Terminate => unreachable!(),
                    };
                    if !matches!(result, Value::Obj(_)) {
                        let error = i.make_error("TypeError", "iterator result is not an object");
                        return self.finish(Suspend::Throw(error));
                    }
                    let done = match i.get_member(&result, "done") {
                        Ok(value) => i.to_boolean(&value),
                        Err(error) => {
                            return self
                                .finish(Suspend::Throw(crate::interpreter::abrupt_value(error)));
                        }
                    };
                    if done {
                        if let Err(error) = set_length_throw(i, &state.array, state.index as f64) {
                            return self.finish(Suspend::Throw(error));
                        }
                        return self.finish(Suspend::Done(state.array));
                    }
                    let value = match i.get_member(&result, "value") {
                        Ok(value) => value,
                        Err(error) => {
                            return self
                                .finish(Suspend::Throw(crate::interpreter::abrupt_value(error)));
                        }
                    };
                    if state.mapper.is_callable() {
                        let mapped = match i.call(
                            state.mapper.clone(),
                            state.this_arg.clone(),
                            &[value, Value::Num(state.index as f64)],
                        ) {
                            Ok(value) => value,
                            Err(error) => {
                                let error = crate::interpreter::abrupt_value(error);
                                return self.close_after_error(i, state, error);
                            }
                        };
                        self.stage = FromAsyncStage::IteratorMapped(state);
                        return Suspend::Await(mapped);
                    }
                    if let Err(error) = cdp_index_or_throw(i, &state.array, state.index, value) {
                        return self.close_after_error(i, state, error);
                    }
                    state.index += 1;
                    self.stage = FromAsyncStage::IteratorNext(state);
                }
                FromAsyncStage::IteratorMapped(mut state) => {
                    let value = match input.take().expect("an Await stage resumes with a signal") {
                        Resume::Next(value) | Resume::Return(value) => value,
                        Resume::Throw(error) => return self.close_after_error(i, state, error),
                        Resume::Terminate => unreachable!(),
                    };
                    if let Err(error) = cdp_index_or_throw(i, &state.array, state.index, value) {
                        return self.close_after_error(i, state, error);
                    }
                    state.index += 1;
                    self.stage = FromAsyncStage::IteratorNext(state);
                }
                FromAsyncStage::ArrayLikeNext(state) => {
                    if state.index >= state.length {
                        if let Err(error) = set_length_throw(i, &state.array, state.length as f64) {
                            return self.finish(Suspend::Throw(error));
                        }
                        return self.finish(Suspend::Done(state.array));
                    }
                    let value = match i.get_member(&state.source, &state.index.to_string()) {
                        Ok(value) => value,
                        Err(error) => {
                            return self
                                .finish(Suspend::Throw(crate::interpreter::abrupt_value(error)));
                        }
                    };
                    self.stage = FromAsyncStage::ArrayLikeValue(state);
                    return Suspend::Await(value);
                }
                FromAsyncStage::ArrayLikeValue(mut state) => {
                    let value = match input.take().expect("an Await stage resumes with a signal") {
                        Resume::Next(value) | Resume::Return(value) => value,
                        Resume::Throw(error) => return self.finish(Suspend::Throw(error)),
                        Resume::Terminate => unreachable!(),
                    };
                    if state.mapper.is_callable() {
                        let mapped = match i.call(
                            state.mapper.clone(),
                            state.this_arg.clone(),
                            &[value, Value::Num(state.index as f64)],
                        ) {
                            Ok(value) => value,
                            Err(error) => {
                                return self.finish(Suspend::Throw(
                                    crate::interpreter::abrupt_value(error),
                                ));
                            }
                        };
                        self.stage = FromAsyncStage::ArrayLikeMapped(state);
                        return Suspend::Await(mapped);
                    }
                    if let Err(error) =
                        cdp_index_or_throw(i, &state.array, state.index as u64, value)
                    {
                        return self.finish(Suspend::Throw(error));
                    }
                    state.index += 1;
                    self.stage = FromAsyncStage::ArrayLikeNext(state);
                }
                FromAsyncStage::ArrayLikeMapped(mut state) => {
                    let value = match input.take().expect("an Await stage resumes with a signal") {
                        Resume::Next(value) | Resume::Return(value) => value,
                        Resume::Throw(error) => return self.finish(Suspend::Throw(error)),
                        Resume::Terminate => unreachable!(),
                    };
                    if let Err(error) =
                        cdp_index_or_throw(i, &state.array, state.index as u64, value)
                    {
                        return self.finish(Suspend::Throw(error));
                    }
                    state.index += 1;
                    self.stage = FromAsyncStage::ArrayLikeNext(state);
                }
                FromAsyncStage::Closing { error, iterator } => {
                    // AsyncIteratorClose with a throw completion awaits a normal return-call
                    // result, then preserves the original throw regardless of settlement/shape.
                    let _ = input.take();
                    drop(iterator);
                    return self.finish(Suspend::Throw(error));
                }
                FromAsyncStage::Done => return self.finish(Suspend::Done(Value::Undefined)),
            }
        }
    }
}

/// GetMethod with the well-known iterator key.
fn from_async_get_method(i: &mut Interp, source: &Value, name: &str) -> Result<Value, Value> {
    let method = match well_known_key(i, name) {
        Some(key) => ab(i.get_member(source, &key))?,
        None => Value::Undefined,
    };
    if matches!(method, Value::Undefined | Value::Null) {
        Ok(Value::Undefined)
    } else if method.is_callable() {
        Ok(method)
    } else {
        Err(i.make_error("TypeError", "iterator method is not callable"))
    }
}

/// Run Array.fromAsync through iterator acquisition/array-like preparation, stopping immediately
/// before its first Await. GetIteratorFromMethod precedes result construction per §23.1.2.2.
fn from_async_start(
    i: &mut Interp,
    ctor: Value,
    source: Value,
    mapper: Value,
    this_arg: Value,
) -> Result<FromAsyncStage, Value> {
    if !matches!(mapper, Value::Undefined) && !mapper.is_callable() {
        return Err(i.make_error("TypeError", "Array.fromAsync: mapFn is not callable"));
    }
    let async_method = from_async_get_method(i, &source, "asyncIterator")?;
    let sync_method = if async_method.is_callable() {
        Value::Undefined
    } else {
        from_async_get_method(i, &source, "iterator")?
    };
    if async_method.is_callable() || sync_method.is_callable() {
        let from_sync = !async_method.is_callable();
        let method = if from_sync { sync_method } else { async_method };
        let iterator = ab(i.call(method, source, &[]))?;
        if !matches!(iterator, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "iterator method returned a non-object"));
        }
        let next = ab(i.get_member(&iterator, "next"))?;
        let array = if i.value_is_constructor(&ctor) {
            ab(i.construct(ctor, &[]))?
        } else {
            i.make_array(Vec::new())
        };
        return Ok(FromAsyncStage::IteratorNext(FromAsyncIter {
            array,
            iterator,
            next,
            mapper,
            this_arg,
            index: 0,
            from_sync,
        }));
    }

    let object = to_object_arg(i, source, "Array.fromAsync")?;
    let length = ab(i.to_length(&object))?;
    let array = if i.value_is_constructor(&ctor) {
        ab(i.construct(ctor, &[Value::Num(length as f64)]))?
    } else {
        if length as u64 > u32::MAX as u64 {
            return Err(i.make_error("RangeError", "invalid array length"));
        }
        i.make_array(Vec::new())
    };
    Ok(FromAsyncStage::ArrayLikeNext(FromAsyncArrayLike {
        array,
        source: Value::Obj(object),
        mapper,
        this_arg,
        index: 0,
        length,
    }))
}

/// AsyncFromSyncIteratorContinuation's unwrap closure: package the awaited value with the saved
/// `done` flag into a fresh IteratorResult object.
fn from_async_sync_unwrap(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let done = matches!(arg(args, 0), Value::Bool(true));
    Ok(i.iter_result_obj(arg(args, 1), done))
}

/// AsyncFromSyncIteratorContinuation's closeOnRejection closure. IteratorClose receives a throw
/// completion, so close failures are discarded and the yielded-value rejection remains primary.
fn from_async_sync_reject(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    i.iterator_close(&arg(args, 0));
    Err(arg(args, 1))
}

/// Emulate %AsyncFromSyncIteratorPrototype%.next plus AsyncFromSyncIteratorContinuation, returning
/// its intrinsic promise for the outer Array.fromAsync Await.
fn from_async_sync_next(i: &mut Interp, iterator: &Value, next: &Value) -> Value {
    let promise = i.new_promise();
    let result = match i.call(next.clone(), iterator.clone(), &[]) {
        Ok(value) => value,
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    if !matches!(result, Value::Obj(_)) {
        let error = i.make_error("TypeError", "iterator result is not an object");
        i.reject_promise(&promise, error);
        return promise;
    }
    let done = match i.get_member(&result, "done") {
        Ok(value) => i.to_boolean(&value),
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    let value = match i.get_member(&result, "value") {
        Ok(value) => value,
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    let value_wrapper = match i.promise_resolve_checked(value) {
        Ok(promise) => promise,
        Err(error) => {
            if !done {
                i.iterator_close(iterator);
            }
            i.reject_promise(&promise, error);
            return promise;
        }
    };
    let on_f = make_bound_len(i, from_async_sync_unwrap, vec![Value::Bool(done)], 1.0);
    let on_r = if done {
        Value::Undefined
    } else {
        make_bound_len(i, from_async_sync_reject, vec![iterator.clone()], 1.0)
    };
    i.promise_then_into(&value_wrapper, on_f, on_r, promise.clone());
    promise
}

/// %AsyncFromSyncIteratorPrototype%.return() for AsyncIteratorClose. This wrapper uses
/// closeOnRejection=false, exactly as ECMA-262 §27.1.5.2.2 requires.
fn from_async_sync_close(i: &mut Interp, iterator: &Value) -> Value {
    let promise = i.new_promise();
    let ret = match i.get_member(iterator, "return") {
        Ok(value) => value,
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    if matches!(ret, Value::Undefined | Value::Null) {
        let result = i.iter_result_obj(Value::Undefined, true);
        i.resolve_promise(&promise, result);
        return promise;
    }
    if !ret.is_callable() {
        let error = i.make_error("TypeError", "iterator return method is not callable");
        i.reject_promise(&promise, error);
        return promise;
    }
    let result = match i.call(ret, iterator.clone(), &[]) {
        Ok(value) => value,
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    if !matches!(result, Value::Obj(_)) {
        let error = i.make_error("TypeError", "iterator return result is not an object");
        i.reject_promise(&promise, error);
        return promise;
    }
    let done = match i.get_member(&result, "done") {
        Ok(value) => i.to_boolean(&value),
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    let value = match i.get_member(&result, "value") {
        Ok(value) => value,
        Err(error) => {
            i.reject_promise(&promise, crate::interpreter::abrupt_value(error));
            return promise;
        }
    };
    let value_wrapper = match i.promise_resolve_checked(value) {
        Ok(promise) => promise,
        Err(error) => {
            i.reject_promise(&promise, error);
            return promise;
        }
    };
    let on_f = make_bound_len(i, from_async_sync_unwrap, vec![Value::Bool(done)], 1.0);
    i.promise_then_into(&value_wrapper, on_f, Value::Undefined, promise.clone());
    promise
}

/// Start the explicit Array.fromAsync state machine and return its result promise.
fn array_from_async(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    let promise = i.new_promise();
    let coroutine = crate::coroutine::Coroutine::FromAsync(Box::new(FromAsyncCoro::new(
        this,
        arg(a, 0),
        arg(a, 1),
        arg(a, 2),
    )));
    if let Value::Obj(object) = &promise {
        let key = Rc::as_ptr(object) as usize;
        i.generators.insert(key, coroutine);
        i.drive_async(
            key,
            promise.clone(),
            crate::coroutine::Resume::Next(Value::Undefined),
        );
    }
    Ok(promise)
}

/// Numeric CreateDataPropertyOrThrow for Array-producing algorithms. Recheck the live receiver
/// after every callback/await: a species result may escape, become non-extensible, freeze its
/// length, or gain a descriptor before the next write. An absent dense index needs no decimal
/// key or temporary descriptor object; prototypes do not participate in [[DefineOwnProperty]].
/// [`cdp_index_or_throw`] for a value already in packed form: the ordinary fresh-Array case
/// moves the word into element storage without widening it.
fn cdp_index_packed_or_throw(
    i: &mut Interp,
    target: &Value,
    index: u64,
    value: PackedValue,
) -> Result<(), Value> {
    let value = match try_define_plain_array_index(target, index, Property::plain_packed(value)) {
        Ok(()) => return Ok(()),
        Err(property) => property.into_value(),
    };
    cdp_or_throw(i, target, &index.to_string(), value)
}

/// CreateDataProperty(A, index, value) for an extensible ordinary Array whose `length` is a
/// writable data property and whose element `index` is absent: Array [[DefineOwnProperty]]
/// (ECMA-262 §10.4.2.1) defines the element and, at or past `length`, raises `length`.
/// `Err` hands the property back for the general, trap-aware algorithm.
fn try_define_plain_array_index(
    target: &Value,
    index: u64,
    property: Property,
) -> Result<(), Property> {
    if index >= u32::MAX as u64 {
        return Err(property);
    }
    let Value::Obj(object) = target else {
        return Err(property);
    };
    let mut object = object.borrow_mut();
    if !object.ic_plain.get() || !matches!(object.exotic, Exotic::Array) || !object.extensible {
        return Err(property);
    }
    let Some(length) = object.props.length_property().and_then(|length| {
        if !length.accessor() && length.writable() {
            if let Value::Num(length) = length.value() {
                return Some(length as u32);
            }
        }
        None
    }) else {
        return Err(property);
    };
    object
        .props
        .try_define_dense_element(index as u32, property)?;
    if index >= u64::from(length) {
        object.props.set_array_length_value((index + 1) as f64);
    }
    Ok(())
}

fn cdp_index_or_throw(
    i: &mut Interp,
    target: &Value,
    index: u64,
    value: Value,
) -> Result<(), Value> {
    let value = match try_define_plain_array_index(target, index, Property::plain(value)) {
        Ok(()) => return Ok(()),
        Err(property) => property.into_value(),
    };
    cdp_or_throw(i, target, &index.to_string(), value)
}

/// CreateDataPropertyOrThrow(O, P, V) (ECMA-262 §7.3.7) for a property key value, as the
/// self-hosted built-ins' intrinsic (`Op::DefineElem`). A Number that is an array index
/// (ToString of it is the canonical index string, -0 included) takes the packed element path
/// without materializing the string; any other key is converted by ToPropertyKey.
pub(crate) fn create_data_property_or_throw_key(
    i: &mut Interp,
    target: &Value,
    key: &Value,
    value: PackedValue,
) -> Result<(), Value> {
    if let Value::Num(n) = *key {
        if n >= 0.0 && n < u32::MAX as f64 && n.fract() == 0.0 {
            return cdp_index_packed_or_throw(i, target, n as u64, value);
        }
    }
    let key = ab(i.to_property_key(key))?;
    cdp_or_throw(i, target, key.as_str(), value.into_value())
}

/// CreateDataPropertyOrThrow (trap-aware for proxy targets).
pub(crate) fn cdp_or_throw(
    i: &mut Interp,
    target: &Value,
    key: &str,
    v: Value,
) -> Result<(), Value> {
    defer_trigger_v(i, target, Some(key))?;
    let Value::Obj(o) = target else {
        return Err(i.make_error("TypeError", "cannot define a property on a non-object"));
    };
    // Fast path: a fresh element on a plain, extensible Array with writable length skips the
    // descriptor-object round trip (this is the hot loop of from/map/filter/slice/concat).
    if !i.proxies.contains_key(&(Rc::as_ptr(o) as usize)) {
        let fast = {
            let b = o.borrow();
            matches!(b.exotic, Exotic::Array)
                && b.extensible
                && !b.props.contains(key)
                && b.props.get("length").map(|p| p.writable()).unwrap_or(true)
        };
        if fast {
            if let Ok(idx) = key.parse::<u64>() {
                if idx < 4294967295 {
                    let len = i.array_length(o);
                    o.borrow_mut()
                        .props
                        .insert(key, Property::data(v, true, true, true));
                    if idx as usize >= len {
                        o.borrow_mut().props.insert(
                            "length",
                            Property::data(Value::Num((idx + 1) as f64), true, false, false),
                        );
                    }
                    return Ok(());
                }
            }
        }
    }
    let r = if let Some((t, h)) = proxy_pair(i, target) {
        let desc = i.new_object();
        set_data(&desc, "value", v);
        set_data(&desc, "writable", Value::Bool(true));
        set_data(&desc, "enumerable", Value::Bool(true));
        set_data(&desc, "configurable", Value::Bool(true));
        proxy_define_property(i, &t, &h, key, &Value::Obj(desc))
    } else {
        define_own_property_record(i, o, key, &data_record(v))
    };
    match r {
        Ok(true) => Ok(()),
        Ok(false) => Err(i.make_error("TypeError", format!("cannot define element '{key}'"))),
        Err(a) => Err(crate::interpreter::abrupt_value(a)),
    }
}

/// Set(A, "length", n, true): setter errors propagate, a refused write is a TypeError (via the
/// strict-mode write semantics).
fn set_length_throw(i: &mut Interp, arr: &Value, n: f64) -> Result<(), Value> {
    let saved = i.strict;
    i.strict = true;
    let r = i.set_member(arr, "length", Value::Num(n));
    i.strict = saved;
    r.map_err(crate::interpreter::abrupt_value)
}

fn iter_some_every(i: &mut Interp, this: Value, a: &[Value], want: bool) -> Result<Value, Value> {
    require_iterator_object(i, &this)?;
    let f = arg(a, 0);
    if !f.is_callable() {
        // Closes the underlying iterator without reading `next`; the TypeError wins.
        i.iterator_close(&this);
        return Err(i.make_error("TypeError", "predicate is not callable"));
    }
    let next = ab(i.get_member(&this, "next"))?;
    let mut k = 0.0;
    let f = crate::callback::Callback::new(f);
    while let Some(v) = step_iter_with(i, &this, &next)? {
        let r = match f.call(i, Value::Undefined, [v, Value::Num(k)]) {
            Ok(r) => r,
            Err(e) => {
                i.iterator_close(&this);
                return Err(crate::interpreter::abrupt_value(e));
            }
        };
        if i.to_boolean(&r) == want {
            ab(i.iterator_close_normal(&this))?;
            return Ok(Value::Bool(want));
        }
        k += 1.0;
    }
    Ok(Value::Bool(!want))
}

/// Iterator.prototype.includes, from the TC39 stage-3 Iterator Includes proposal.
///
/// `skippedElements` deliberately is not coerced: the proposal accepts only an integral Number
/// (or either infinity), and validates it before GetIteratorDirect observes `next`.
fn iterator_includes(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_iterator_object(i, &this)?;
    let search = arg(a, 0);
    let to_skip = match arg(a, 1) {
        Value::Undefined => 0.0,
        Value::Num(n) if n.is_infinite() || (!n.is_nan() && n.fract() == 0.0) => n,
        _ => {
            i.iterator_close(&this);
            return Err(i.make_error("TypeError", "skippedElements must be an integral number"));
        }
    };
    if to_skip < 0.0 || (to_skip.is_finite() && to_skip > 9_007_199_254_740_991.0) {
        i.iterator_close(&this);
        return Err(i.make_error(
            "RangeError",
            "skippedElements is outside the permitted range",
        ));
    }
    let next = ab(i.get_member(&this, "next"))?;
    let mut skipped = 0.0;
    while let Some(value) = step_iter_with(i, &this, &next)? {
        if skipped < to_skip {
            skipped += 1.0;
        } else if same_value_zero(&value, &search) {
            ab(i.iterator_close_normal(&this))?;
            return Ok(Value::Bool(true));
        }
    }
    Ok(Value::Bool(false))
}

/// Iterator.prototype.join, from the TC39 stage-3 Iterator Join proposal.
/// Separator conversion precedes GetIteratorDirect; conversion of an element closes the iterator
/// on an abrupt completion, whereas iterator-protocol errors do not perform an additional close.
fn iterator_join(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    require_iterator_object(i, &this)?;
    let separator = match arg(a, 0) {
        Value::Undefined => ",".to_string(),
        value => match i.to_string(&value) {
            Ok(s) => s.to_string(),
            Err(e) => {
                i.iterator_close(&this);
                return Err(crate::interpreter::abrupt_value(e));
            }
        },
    };
    let next = ab(i.get_member(&this, "next"))?;
    let mut result = String::new();
    let mut first = true;
    while let Some(value) = step_iter_with(i, &this, &next)? {
        if first {
            first = false;
        } else {
            result.push_str(&separator);
        }
        if !matches!(value, Value::Undefined | Value::Null) {
            match i.to_string(&value) {
                Ok(s) => result.push_str(&s),
                Err(e) => {
                    i.iterator_close(&this);
                    return Err(crate::interpreter::abrupt_value(e));
                }
            }
        }
    }
    Ok(Value::lstr(
        crate::jstr::canonicalize(&result).unwrap_or(result),
    ))
}

/// Step an iterator object (`this`) once: `Some(value)` or `None` when done.
/// GetIteratorDirect's receiver check: an Iterator.prototype helper requires an Object `this`.
fn require_iterator_object(i: &Interp, this: &Value) -> Result<(), Value> {
    if matches!(this, Value::Obj(_)) {
        Ok(())
    } else {
        Err(i.make_error("TypeError", "Iterator method called on a non-object"))
    }
}

/// Advance an iterator using an already-captured `next` method (iterator helpers read `next` exactly
/// once at creation per GetIteratorDirect, then reuse it).
fn step_iter_with(i: &mut Interp, src: &Value, next: &Value) -> Result<Option<Value>, Value> {
    if !next.is_callable() {
        return Err(i.make_error("TypeError", "iterator.next is not a function"));
    }
    ab(i.iterator_step(src, next))
}

/// Build a lazy iterator-helper (map/filter/take/drop/flatMap) wrapping `source`.
fn make_iter_helper(i: &mut Interp, source: Value, kind: &str, f: Value) -> Result<Value, Value> {
    // GetIteratorDirect requires an Object receiver (checked before the callable/limit checks).
    if !matches!(source, Value::Obj(_)) {
        return Err(i.make_error("TypeError", "Iterator helper called on a non-object"));
    }
    if matches!(kind, "map" | "filter" | "flatMap") && !f.is_callable() {
        // A non-callable mapper/predicate still closes the underlying iterator.
        i.iterator_close(&source);
        return Err(i.make_error("TypeError", "Iterator helper argument is not callable"));
    }
    // ECMA-262 §27.1.3.3.2 / §27.1.3.3.11: validate the limit before GetIteratorDirect
    // reads `next`. Argument conversion and every validation failure close the provisional
    // Iterator Record, while +Infinity remains valid.
    let limit = if matches!(kind, "take" | "drop") {
        let raw = match i.to_number(&f) {
            Ok(n) => n,
            Err(e) => {
                i.iterator_close(&source);
                return Err(crate::interpreter::abrupt_value(e));
            }
        };
        if raw.is_nan() || (raw.is_finite() && raw > 9_007_199_254_740_991.0) || raw.trunc() < 0.0 {
            i.iterator_close(&source);
            return Err(i.make_error("RangeError", "limit must be a non-negative number"));
        }
        Some(raw.trunc())
    } else {
        None
    };
    let proto = i
        .extra_protos
        .get("%IteratorHelperPrototype%")
        .or_else(|| i.extra_protos.get("%IteratorPrototype%"))
        .cloned();
    let obj = Object::new(proto);
    // GetIteratorDirect: read the source's `next` method exactly once, now.
    let next = ab(i.get_member(&source, "next"))?;
    set_builtin(&obj, "#\u{0}ih_next", next);
    set_builtin(&obj, "#\u{0}ih_src", source);
    set_builtin(&obj, "#\u{0}ih_kind", Value::lstr(kind));
    set_builtin(&obj, "#\u{0}ih_fn", f.clone());
    if let Some(n) = limit {
        set_builtin(&obj, "#\u{0}ih_n", Value::Num(n));
        set_builtin(&obj, "#\u{0}ih_started", Value::Bool(false));
    }
    set_builtin(&obj, "#\u{0}ih_count", Value::Num(0.0));
    set_builtin(&obj, "#\u{0}ih_source_done", Value::Bool(false));
    set_builtin(&obj, "#\u{0}ih_done", Value::Bool(false));
    Ok(Value::Obj(obj))
}

/// Build the lazy helper specified by the TC39 stage-3 Iterator Chunking proposal.
/// Validation is intentionally non-coercing and precedes GetIteratorDirect's observable `next`
/// lookup. `undersized` is present only for `windows`.
fn make_chunking_helper(
    i: &mut Interp,
    source: Value,
    size: Value,
    undersized: Option<Value>,
) -> Result<Value, Value> {
    require_iterator_object(i, &source)?;
    let is_windows = undersized.is_some();
    let size = match size {
        Value::Num(n) if n.is_finite() && n.fract() == 0.0 => n,
        _ => {
            i.iterator_close(&source);
            return Err(i.make_error("TypeError", "chunk size must be an integral number"));
        }
    };
    if !(1.0..=4_294_967_295.0).contains(&size) {
        i.iterator_close(&source);
        return Err(i.make_error("RangeError", "chunk size is outside the permitted range"));
    }
    let mode = match undersized {
        None | Some(Value::Undefined) => "only-full",
        Some(Value::Str(s)) if s.to_string() == "only-full" => "only-full",
        Some(Value::Str(s)) if s.to_string() == "allow-partial" => "allow-partial",
        Some(_) => {
            i.iterator_close(&source);
            return Err(i.make_error("TypeError", "invalid undersized mode"));
        }
    };
    let next = ab(i.get_member(&source, "next"))?;
    let proto = i
        .extra_protos
        .get("%IteratorHelperPrototype%")
        .or_else(|| i.extra_protos.get("%IteratorPrototype%"))
        .cloned();
    let obj = Object::new(proto);
    set_builtin(&obj, "#\u{0}ih_next", next);
    set_builtin(&obj, "#\u{0}ih_src", source);
    set_builtin(
        &obj,
        "#\u{0}ih_kind",
        Value::lstr(if is_windows { "windows" } else { "chunks" }),
    );
    set_builtin(&obj, "#\u{0}ih_fn", Value::Undefined);
    set_builtin(&obj, "#\u{0}ih_size", Value::Num(size));
    set_builtin(&obj, "#\u{0}ih_undersized", Value::lstr(mode));
    set_builtin(&obj, "#\u{0}ih_buf", i.make_array(Vec::new()));
    set_builtin(&obj, "#\u{0}ih_source_done", Value::Bool(false));
    set_builtin(&obj, "#\u{0}ih_count", Value::Num(0.0));
    set_builtin(&obj, "#\u{0}ih_done", Value::Bool(false));
    Ok(Value::Obj(obj))
}

/// %IteratorHelperPrototype%.return: closes the underlying iterator once.
fn iter_helper_return(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    if !matches!(&this, Value::Obj(o) if o.borrow().props.contains("#\u{0}ih_kind")) {
        return Err(i.make_error("TypeError", "return called on an incompatible receiver"));
    }
    let running = ab(i.get_member(&this, "#\u{0}ih_running"))?;
    if i.to_boolean(&running) {
        return Err(i.make_error("TypeError", "iterator helper is already running"));
    }
    let done = ab(i.get_member(&this, "#\u{0}ih_done"))?;
    if !i.to_boolean(&done) {
        let o = this.as_obj().unwrap().clone();
        let started = ab(i.get_member(&this, "#\u{0}ih_gstarted"))?;
        let started = i.to_boolean(&started);
        if started {
            // Suspended at a yield: the close runs in the executing state.
            set_internal(&o, "#\u{0}ih_running", Value::Bool(true));
        } else {
            // Suspended-start: the generator completes before the close.
            set_internal(&o, "#\u{0}ih_done", Value::Bool(true));
        }
        let res = (|i: &mut Interp| -> Result<(), Value> {
            // A helper suspended inside a flatMap inner iterator closes it first.
            let inner = ab(i.get_member(&this, "#\u{0}ih_inner"))?;
            if matches!(inner, Value::Obj(_)) {
                ab(i.iterator_close_normal(&inner))?;
            }
            let source_done = ab(i.get_member(&this, "#\u{0}ih_source_done"))?;
            if !i.to_boolean(&source_done) {
                let src = ab(i.get_member(&this, "#\u{0}ih_src"))?;
                // A normal return() propagates an error from the source's return method.
                ab(i.iterator_close_normal(&src))?;
            }
            Ok(())
        })(i);
        if started {
            set_internal(&o, "#\u{0}ih_running", Value::Bool(false));
            set_internal(&o, "#\u{0}ih_done", Value::Bool(true));
        }
        res?;
    }
    Ok(iter_result(i, Value::Undefined, true))
}

/// SetterThatIgnoresPrototypeProperties: assigning through Iterator.prototype's accessor throws
/// when the receiver IS the prototype, else creates an own data property on the receiver.
fn iterator_proto_weird_set(
    i: &mut Interp,
    this: Value,
    v: Value,
    key: &str,
) -> Result<Value, Value> {
    if let (Some(h), Value::Obj(t)) = (i.extra_protos.get("%IteratorProtoMarker%"), &this) {
        if Rc::ptr_eq(h, t) {
            return Err(i.make_error(
                "TypeError",
                "cannot assign to a property of Iterator.prototype",
            ));
        }
    }
    match &this {
        Value::Obj(o) => {
            o.borrow_mut()
                .props
                .insert(key, Property::data(v, true, true, true));
            Ok(Value::Undefined)
        }
        _ => Err(i.make_error("TypeError", "cannot create property on a primitive")),
    }
}

fn iter_result(i: &mut Interp, value: Value, done: bool) -> Value {
    // CreateIterResultObject (ECMA-262): an ordinary object of the current Realm whose own data
    // properties `value` then `done` are created by CreateDataPropertyOrThrow. The key sequence
    // is fixed, so the finished map is built once per Agent and instantiated.
    let map = i.iter_result_map.get_or_init(|| {
        let mut map = Props::new();
        map.insert("value", Property::plain(Value::Undefined));
        map.insert("done", Property::plain(Value::Undefined));
        map
    });
    let props = map.instantiate_plain([value, Value::Bool(done)].into_iter());
    Value::Obj(Object::new_with_parts(
        Some(i.object_proto.clone()),
        props,
        Exotic::None,
    ))
}

/// `Iterator.zip` / `Iterator.zipKeyed`: validate options, open every input iterator eagerly
/// (closing already-open ones, in reverse order, if anything goes wrong), then step them in
/// lockstep per the mode (shortest/longest/strict) with `padding` filling exhausted inputs.
fn iterator_zip(i: &mut Interp, a: &[Value], keyed: bool) -> Result<Value, Value> {
    let input = arg(a, 0);
    if !matches!(input, Value::Obj(_)) {
        return Err(i.make_error("TypeError", "Iterator.zip input is not an object"));
    }
    // GetOptionsObject + mode: undefined -> "shortest"; anything else must be one of the three
    // mode strings exactly (no coercion) or it's a TypeError.
    let options = arg(a, 1);
    let mode = match &options {
        Value::Undefined => "shortest".to_string(),
        Value::Obj(_) => match ab(i.get_member(&options, "mode"))? {
            Value::Undefined => "shortest".to_string(),
            Value::Str(m) if matches!(&*m.to_string(), "shortest" | "longest" | "strict") => {
                m.to_string()
            }
            _ => return Err(i.make_error("TypeError", "invalid Iterator.zip mode")),
        },
        _ => return Err(i.make_error("TypeError", "Iterator.zip options is not an object")),
    };
    // The padding option is read (and type-checked) before the inputs are opened.
    let padding_value = if mode == "longest" && matches!(&options, Value::Obj(_)) {
        let p = ab(i.get_member(&options, "padding"))?;
        if !matches!(p, Value::Undefined | Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Iterator.zip padding is not an object"));
        }
        p
    } else {
        Value::Undefined
    };
    // Open the inputs in order, GetIteratorFlattenable each as it is read.
    let mut keys: Vec<crate::value::PropertyKey> = Vec::new();
    let mut iters: Vec<Value> = Vec::new();
    let mut nexts: Vec<Value> = Vec::new();
    let open_one = |i: &mut Interp,
                    v: &Value,
                    iters: &mut Vec<Value>,
                    nexts: &mut Vec<Value>|
     -> Result<(), Value> {
        let iter = get_iterator_flattenable(i, v, false)?;
        let next = ab(i.get_member(&iter, "next"))?;
        iters.push(iter);
        nexts.push(next);
        Ok(())
    };
    // IteratorCloseAll during setup: close the open inputs in reverse order, swallowing close
    // errors (the pending error wins), optionally ending with the outer iterables iterator.
    let close_setup = |i: &mut Interp, iters: &[Value], also: Option<&Value>| {
        for it in iters.iter().rev() {
            i.iterator_close(it);
        }
        if let Some(it) = also {
            i.iterator_close(it);
        }
    };
    if keyed {
        // zipKeyed: each own enumerable non-undefined-valued key of `iterables` is an input.
        let all_keys: Vec<crate::value::PropertyKey> = if let Some((t, h)) = proxy_pair(i, &input) {
            let mut ks = Vec::new();
            for k in proxy_own_keys(i, &t, &h)? {
                ks.push(ab(i.to_property_key(&k))?);
            }
            ks
        } else {
            let o = input.as_obj().unwrap();
            ordinary_own_keys_ordered(i, o)?
                .into_iter()
                .map(crate::value::PropertyKey::string)
                .collect()
        };
        for k in all_keys {
            // [[GetOwnProperty]] fresh per key: skip missing or non-enumerable properties.
            let enumerable = if let Some((t, h)) = proxy_pair(i, &input) {
                match proxy_gopd_value(i, &t, &h, &k) {
                    Ok(Value::Undefined) => false,
                    Ok(d) => {
                        let ev = ab(i.get_member(&d, "enumerable"))?;
                        i.to_boolean(&ev)
                    }
                    Err(e) => {
                        close_setup(i, &iters, None);
                        return Err(e);
                    }
                }
            } else {
                let object = input.as_obj().unwrap();
                if ab(i.host_indexed_own_value(object, &k))?.is_some() {
                    crate::value::canonical_index(&k).is_some()
                } else {
                    object
                        .borrow()
                        .props
                        .get(&k)
                        .map(|property| property.enumerable())
                        .unwrap_or(false)
                }
            };
            if !enumerable {
                continue;
            }
            let v = match ab(i.get_member(&input, &k)) {
                Ok(v) => v,
                Err(e) => {
                    close_setup(i, &iters, None);
                    return Err(e);
                }
            };
            if matches!(v, Value::Undefined) {
                continue;
            }
            if let Err(e) = open_one(i, &v, &mut iters, &mut nexts) {
                close_setup(i, &iters, None);
                return Err(e);
            }
            keys.push(k);
        }
    } else {
        // zip: step the iterable-of-iterables, opening each input as it is produced. An error
        // opening one closes the already-open inputs (reverse) and then the iterables iterator.
        let (input_iter, input_next) = ab(i.get_iterator(&input))?;
        loop {
            match step_iter_with(i, &input_iter, &input_next) {
                Ok(None) => break,
                Ok(Some(v)) => {
                    if let Err(e) = open_one(i, &v, &mut iters, &mut nexts) {
                        close_setup(i, &iters, Some(&input_iter));
                        return Err(e);
                    }
                }
                Err(e) => {
                    // The failed step already finished the iterables iterator; only the
                    // open inputs are closed.
                    close_setup(i, &iters, None);
                    return Err(e);
                }
            }
        }
    }
    let n_iters = iters.len();
    // `longest` padding, aligned with the inputs: zip iterates the padding iterable at most
    // iterCount times (it may be infinite); zipKeyed reads padding[key] per input key.
    let mut padding = vec![Value::Undefined; n_iters];
    if mode == "longest" && !matches!(padding_value, Value::Undefined) {
        if keyed {
            for (j, k) in keys.iter().enumerate() {
                match ab(i.get_member(&padding_value, k)) {
                    Ok(v) => padding[j] = v,
                    Err(e) => {
                        close_setup(i, &iters, None);
                        return Err(e);
                    }
                }
            }
        } else {
            let (pit, pnext) = match ab(i.get_iterator(&padding_value)) {
                Ok(p) => p,
                Err(e) => {
                    close_setup(i, &iters, None);
                    return Err(e);
                }
            };
            let mut using = true;
            for slot in padding.iter_mut().take(n_iters) {
                if !using {
                    break;
                }
                match step_iter_with(i, &pit, &pnext) {
                    Ok(Some(v)) => *slot = v,
                    Ok(None) => using = false,
                    Err(e) => {
                        close_setup(i, &iters, None);
                        return Err(e);
                    }
                }
            }
            if using {
                // The padding iterator wasn't exhausted: close it (normal completion).
                if let Err(e) = i.iterator_close_normal(&pit) {
                    let e = crate::interpreter::abrupt_value(e);
                    close_setup(i, &iters, None);
                    return Err(e);
                }
            }
        }
    }

    let obj = Object::new(
        i.extra_protos
            .get("%IteratorHelperPrototype%")
            .or_else(|| i.extra_protos.get("%IteratorPrototype%"))
            .cloned(),
    );
    set_builtin(&obj, "#\u{0}zip_iters", i.make_array(iters));
    set_builtin(&obj, "#\u{0}zip_nexts", i.make_array(nexts));
    set_builtin(
        &obj,
        "#\u{0}zip_state",
        i.make_array(vec![Value::Bool(false); n_iters]),
    );
    set_builtin(&obj, "#\u{0}zip_mode", Value::lstr(mode.clone()));
    set_builtin(&obj, "#\u{0}zip_pad", i.make_array(padding));
    set_builtin(&obj, "#\u{0}zip_finished", Value::Bool(false));
    if keyed {
        let karr = i.make_array(keys.into_iter().map(|key| key.into_value()).collect());
        set_builtin(&obj, "#\u{0}zip_keys", karr);
    }
    i.def_method(&obj, "next", 0, zip_next);
    // return(): close every still-open input in reverse order; the close runs in the executing
    // state, and the first close error wins (later ones are swallowed).
    i.def_method(&obj, "return", 0, |i, this, _a| {
        let finished = ab(i.get_member(&this, "#\u{0}zip_finished"))?;
        if i.to_boolean(&finished) {
            return Ok(iter_result(i, Value::Undefined, true));
        }
        let running = ab(i.get_member(&this, "#\u{0}zip_running"))?;
        if i.to_boolean(&running) {
            return Err(i.make_error("TypeError", "Iterator.zip iterator is already running"));
        }
        let o = this.as_obj().unwrap().clone();
        let started = ab(i.get_member(&this, "#\u{0}zip_started"))?;
        let res = if i.to_boolean(&started) {
            // Suspended at a yield: the close runs in the executing state and the
            // generator completes afterwards.
            set_internal(&o, "#\u{0}zip_running", Value::Bool(true));
            let res = zip_close_open(i, &this, None);
            set_internal(&o, "#\u{0}zip_running", Value::Bool(false));
            set_internal(&o, "#\u{0}zip_finished", Value::Bool(true));
            res
        } else {
            // Suspended-start: the generator completes first, then the inputs close
            // (reentrant next/return during the close see the completed state).
            set_internal(&o, "#\u{0}zip_finished", Value::Bool(true));
            zip_close_open(i, &this, None)
        };
        res?;
        Ok(iter_result(i, Value::Undefined, true))
    });
    if let Some(sym) = i.iterator_sym.clone() {
        let itf = i.make_native("[Symbol.iterator]", 0, return_this);
        obj.borrow_mut()
            .props
            .insert(Interp::sym_key(&sym), Property::builtin(Value::Obj(itf)));
    }
    Ok(Value::Obj(obj))
}

/// OrdinaryOwnPropertyKeys as internal key strings: array indices ascending, then other string
/// keys in insertion order, then symbol keys in insertion order.
pub(crate) fn ordinary_own_keys_ordered(i: &mut Interp, o: &Gc) -> Result<Vec<String>, Value> {
    let mut indices: Vec<u32> = i
        .host_indexed_len(o)
        .map(|length| (0..length).collect())
        .unwrap_or_default();
    let names = ab(i.host_named_keys(o))?;
    let all = o.borrow().props.ordered_keys();
    let mut strings: Vec<String> = Vec::new();
    let mut symbols: Vec<String> = Vec::new();
    for k in all {
        if Interp::is_sym_key(&k) {
            symbols.push(k.to_string());
        } else if let Ok(n) = k.parse::<u32>() {
            if n != u32::MAX && n.to_string() == *k {
                indices.push(n);
            } else {
                strings.push(k.to_string());
            }
        } else {
            strings.push(k.to_string());
        }
    }
    indices.sort_unstable();
    indices.dedup();
    let mut out: Vec<String> = indices.into_iter().map(|n| n.to_string()).collect();
    out.extend(names);
    out.extend(strings);
    out.extend(symbols);
    Ok(out)
}

/// IteratorCloseAll over the zip iterator's still-open inputs, in reverse order. With a pending
/// error (`err`) all close errors are swallowed and the pending error is returned; on a normal
/// completion the first close error wins (later closes still run but are swallowed).
fn zip_close_open(i: &mut Interp, this: &Value, err: Option<Value>) -> Result<(), Value> {
    let iters = ab(i.get_member(this, "#\u{0}zip_iters"))?;
    let state = ab(i.get_member(this, "#\u{0}zip_state"))?;
    let n = match &iters {
        Value::Obj(o) => i.array_length(o),
        _ => 0,
    };
    let mut pending = err;
    for j in (0..n).rev() {
        let closed = ab(i.get_member(&state, &j.to_string()))?;
        if i.to_boolean(&closed) {
            continue;
        }
        ab(i.set_member(&state, &j.to_string(), Value::Bool(true)))?;
        let it = ab(i.get_member(&iters, &j.to_string()))?;
        if pending.is_some() {
            i.iterator_close(&it);
        } else if let Err(e) = i.iterator_close_normal(&it) {
            pending = Some(crate::interpreter::abrupt_value(e));
        }
    }
    match pending {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn zip_next(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    // GeneratorValidate: re-entering the running zip iterator throws.
    let running = ab(i.get_member(&this, "#\u{0}zip_running"))?;
    if i.to_boolean(&running) {
        return Err(i.make_error("TypeError", "Iterator.zip iterator is already running"));
    }
    let finished = ab(i.get_member(&this, "#\u{0}zip_finished"))?;
    if i.to_boolean(&finished) {
        return Ok(iter_result(i, Value::Undefined, true));
    }
    let Some(o) = this.as_obj().cloned() else {
        return Err(i.make_error("TypeError", "next called on an incompatible receiver"));
    };
    set_internal(&o, "#\u{0}zip_started", Value::Bool(true));
    set_internal(&o, "#\u{0}zip_running", Value::Bool(true));
    let res = zip_step(i, this);
    set_internal(&o, "#\u{0}zip_running", Value::Bool(false));
    let done = match &res {
        Err(_) => true,
        Ok(r) => matches!(r, Value::Obj(ro) if matches!(
            ro.borrow().props.get("done").map(|p| p.value()),
            Some(Value::Bool(true))
        )),
    };
    if done {
        set_internal(&o, "#\u{0}zip_finished", Value::Bool(true));
    }
    res
}

/// One lockstep round of the zip closure.
fn zip_step(i: &mut Interp, this: Value) -> Result<Value, Value> {
    let iters = ab(i.get_member(&this, "#\u{0}zip_iters"))?;
    let nexts = ab(i.get_member(&this, "#\u{0}zip_nexts"))?;
    let state = ab(i.get_member(&this, "#\u{0}zip_state"))?;
    let pad = ab(i.get_member(&this, "#\u{0}zip_pad"))?;
    let mode_v = ab(i.get_member(&this, "#\u{0}zip_mode"))?;
    let mode = ab(i.to_string(&mode_v))?.to_string();
    let n = match &iters {
        Value::Obj(o) => i.array_length(o),
        _ => 0,
    };
    if n == 0 {
        return Ok(iter_result(i, Value::Undefined, true));
    }
    let mark = |i: &mut Interp, state: &Value, j: usize| -> Result<(), Value> {
        ab(i.set_member(state, &j.to_string(), Value::Bool(true)))
    };
    let mut values = vec![Value::Undefined; n];
    for j in 0..n {
        let closed = ab(i.get_member(&state, &j.to_string()))?;
        if i.to_boolean(&closed) {
            // An exhausted input in `longest` mode contributes its padding value.
            values[j] = ab(i.get_member(&pad, &j.to_string()))?;
            continue;
        }
        let it = ab(i.get_member(&iters, &j.to_string()))?;
        let nx = ab(i.get_member(&nexts, &j.to_string()))?;
        match step_iter_with(i, &it, &nx) {
            Ok(Some(v)) => values[j] = v,
            Ok(None) => {
                mark(i, &state, j)?;
                match mode.as_str() {
                    "shortest" => {
                        zip_close_open(i, &this, None)?;
                        return Ok(iter_result(i, Value::Undefined, true));
                    }
                    "strict" => {
                        if j != 0 {
                            let e = i.make_error(
                                "TypeError",
                                "Iterator.zip strict: input iterators have different lengths",
                            );
                            return Err(zip_close_open(i, &this, Some(e)).unwrap_err());
                        }
                        // The first input finished: every other input must be done too
                        // (checked with IteratorStep — `done` is read, values are not).
                        for k in 1..n {
                            let kit = ab(i.get_member(&iters, &k.to_string()))?;
                            let knx = ab(i.get_member(&nexts, &k.to_string()))?;
                            let res = (|i: &mut Interp| -> Result<bool, Value> {
                                if !knx.is_callable() {
                                    return Err(i.make_error(
                                        "TypeError",
                                        "iterator.next is not a function",
                                    ));
                                }
                                let r = ab(i.call(knx.clone(), kit.clone(), &[]))?;
                                if !matches!(r, Value::Obj(_)) {
                                    return Err(i.make_error(
                                        "TypeError",
                                        "iterator result is not an object",
                                    ));
                                }
                                let d = ab(i.get_member(&r, "done"))?;
                                Ok(i.to_boolean(&d))
                            })(i);
                            match res {
                                Err(e) => {
                                    mark(i, &state, k)?;
                                    return Err(zip_close_open(i, &this, Some(e)).unwrap_err());
                                }
                                Ok(true) => mark(i, &state, k)?,
                                Ok(false) => {
                                    let e = i.make_error(
                                        "TypeError",
                                        "Iterator.zip strict: input iterators have different lengths",
                                    );
                                    return Err(zip_close_open(i, &this, Some(e)).unwrap_err());
                                }
                            }
                        }
                        return Ok(iter_result(i, Value::Undefined, true));
                    }
                    _ => {
                        // longest: exhausted inputs are padded; finish when all are done.
                        values[j] = ab(i.get_member(&pad, &j.to_string()))?;
                        let mut any_open = false;
                        for k in 0..n {
                            let c = ab(i.get_member(&state, &k.to_string()))?;
                            if !i.to_boolean(&c) {
                                any_open = true;
                                break;
                            }
                        }
                        if !any_open {
                            return Ok(iter_result(i, Value::Undefined, true));
                        }
                    }
                }
            }
            Err(e) => {
                mark(i, &state, j)?;
                return Err(zip_close_open(i, &this, Some(e)).unwrap_err());
            }
        }
    }
    // finishResults: zip yields an Array; zipKeyed a null-prototype object keyed like the input.
    let keys = ab(i.get_member(&this, "#\u{0}zip_keys"))?;
    let result = if matches!(keys, Value::Obj(_)) {
        let o = Object::new(None);
        for (j, v) in values.into_iter().enumerate() {
            let k = ab(i.get_member(&keys, &j.to_string()))?;
            let k = ab(i.to_property_key(&k))?;
            o.borrow_mut()
                .props
                .insert(k.as_str(), Property::data(v, true, true, true));
        }
        Value::Obj(o)
    } else {
        i.make_array(values)
    };
    Ok(iter_result(i, result, false))
}

/// `Iterator.concat`'s iterator: opens each captured iterable in order and yields its values.
fn concat_next(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    // GeneratorValidate: re-entering the running concat iterator throws.
    let running = ab(i.get_member(&this, "#\u{0}cc_running"))?;
    if i.to_boolean(&running) {
        return Err(i.make_error("TypeError", "Iterator.concat iterator is already running"));
    }
    let done = ab(i.get_member(&this, "#\u{0}cc_done"))?;
    if i.to_boolean(&done) {
        return Ok(iter_result(i, Value::Undefined, true));
    }
    let Some(o) = this.as_obj().cloned() else {
        return Err(i.make_error("TypeError", "next called on an incompatible receiver"));
    };
    set_internal(&o, "#\u{0}cc_gstarted", Value::Bool(true));
    set_internal(&o, "#\u{0}cc_running", Value::Bool(true));
    let res = concat_step(i, this);
    set_internal(&o, "#\u{0}cc_running", Value::Bool(false));
    let finished = match &res {
        Err(_) => true,
        Ok(r) => matches!(r, Value::Obj(ro) if matches!(
            ro.borrow().props.get("done").map(|p| p.value()),
            Some(Value::Bool(true))
        )),
    };
    if finished {
        set_internal(&o, "#\u{0}cc_done", Value::Bool(true));
    }
    res
}

fn concat_step(i: &mut Interp, this: Value) -> Result<Value, Value> {
    loop {
        let cur = ab(i.get_member(&this, "#\u{0}cc_cur"))?;
        if matches!(cur, Value::Undefined) {
            // Open the next item's iterator (or finish).
            let idx = ab(i.get_member(&this, "#\u{0}cc_idx"))?;
            let idx = ab(i.to_number(&idx))? as usize;
            let methods = ab(i.get_member(&this, "#\u{0}cc_methods"))?;
            let mlen = match &methods {
                Value::Obj(o) => i.array_length(o),
                _ => 0,
            };
            if idx >= mlen {
                return Ok(iter_result(i, Value::Undefined, true));
            }
            let items = ab(i.get_member(&this, "#\u{0}cc_items"))?;
            let item = ab(i.get_member(&items, &idx.to_string()))?;
            let method = ab(i.get_member(&methods, &idx.to_string()))?;
            let iter = ab(i.call(method, item, &[]))?;
            if !matches!(iter, Value::Obj(_)) {
                return Err(i.make_error("TypeError", "@@iterator did not return an object"));
            }
            let next = ab(i.get_member(&iter, "next"))?;
            set_internal(this.as_obj().unwrap(), "#\u{0}cc_cur", iter);
            set_internal(this.as_obj().unwrap(), "#\u{0}cc_curnext", next);
            set_internal(
                this.as_obj().unwrap(),
                "#\u{0}cc_idx",
                Value::Num((idx + 1) as f64),
            );
        }
        let cur = ab(i.get_member(&this, "#\u{0}cc_cur"))?;
        let next = ab(i.get_member(&this, "#\u{0}cc_curnext"))?;
        match step_iter_with(i, &cur, &next)? {
            Some(v) => return Ok(iter_result(i, v, false)),
            None => set_internal(this.as_obj().unwrap(), "#\u{0}cc_cur", Value::Undefined),
        }
    }
}

/// GetIteratorFlattenable returning the iterator object: a string (when allowed) or an object's
/// @@iterator, or the object itself when it has no @@iterator (it's already an iterator).
fn get_iterator_flattenable(
    i: &mut Interp,
    v: &Value,
    allow_strings: bool,
) -> Result<Value, Value> {
    if !(matches!(v, Value::Obj(_)) || (allow_strings && matches!(v, Value::Str(_)))) {
        return Err(i.make_error("TypeError", "value is not iterable"));
    }
    let iter_key = i
        .iterator_sym
        .clone()
        .map(|s| Interp::sym_key(&s))
        .unwrap_or_default();
    let itf = ab(i.get_member(v, &iter_key))?;
    let iter = if matches!(itf, Value::Undefined | Value::Null) {
        v.clone()
    } else if itf.is_callable() {
        ab(i.call(itf, v.clone(), &[]))?
    } else {
        return Err(i.make_error("TypeError", "@@iterator is not callable"));
    };
    if !matches!(iter, Value::Obj(_)) {
        return Err(i.make_error("TypeError", "iterator is not an object"));
    }
    Ok(iter)
}

fn iter_helper_next(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    // Brand check: only real iterator-helper objects carry the [[UnderlyingIterator]] slots (a
    // generator also inherits from %IteratorPrototype% but must be rejected here).
    if !matches!(&this, Value::Obj(o) if o.borrow().props.contains("#\u{0}ih_kind") || o.borrow().props.contains("#\u{0}zip_iters"))
    {
        return Err(i.make_error("TypeError", "next called on an incompatible receiver"));
    }
    // GeneratorValidate: re-entering a running helper throws.
    let running = ab(i.get_member(&this, "#\u{0}ih_running"))?;
    if i.to_boolean(&running) {
        return Err(i.make_error("TypeError", "iterator helper is already running"));
    }
    // A helper that has already finished stays done (and never re-touches the source).
    let done = ab(i.get_member(&this, "#\u{0}ih_done"))?;
    if i.to_boolean(&done) {
        return Ok(iter_result(i, Value::Undefined, true));
    }
    let Some(o) = this.as_obj().cloned() else {
        return Err(i.make_error("TypeError", "next called on an incompatible receiver"));
    };
    set_internal(&o, "#\u{0}ih_gstarted", Value::Bool(true));
    set_internal(&o, "#\u{0}ih_running", Value::Bool(true));
    let res = iter_helper_step(i, this);
    set_internal(&o, "#\u{0}ih_running", Value::Bool(false));
    // Any completion — a done result or a throw — moves the helper to the completed state.
    let finished = match &res {
        Err(_) => true,
        Ok(r) => matches!(r, Value::Obj(ro) if matches!(
            ro.borrow().props.get("done").map(|p| p.value()),
            Some(Value::Bool(true))
        )),
    };
    if finished {
        set_internal(&o, "#\u{0}ih_done", Value::Bool(true));
    }
    res
}

fn iter_helper_step(i: &mut Interp, this: Value) -> Result<Value, Value> {
    let src = ab(i.get_member(&this, "#\u{0}ih_src"))?;
    let inext = ab(i.get_member(&this, "#\u{0}ih_next"))?;
    let kind_v = ab(i.get_member(&this, "#\u{0}ih_kind"))?;
    let kind = ab(i.to_string(&kind_v))?;
    let f = ab(i.get_member(&this, "#\u{0}ih_fn"))?;
    let count_v = ab(i.get_member(&this, "#\u{0}ih_count"))?;
    let count = ab(i.to_number(&count_v))?;
    match &*kind {
        "map" => match step_iter_with(i, &src, &inext)? {
            None => Ok(iter_result(i, Value::Undefined, true)),
            Some(v) => {
                let mv = match i.call(f, Value::Undefined, &[v, Value::Num(count)]) {
                    Ok(r) => r,
                    Err(e) => {
                        i.iterator_close(&src);
                        return Err(crate::interpreter::abrupt_value(e));
                    }
                };
                set_internal(
                    this.as_obj().unwrap(),
                    "#\u{0}ih_count",
                    Value::Num(count + 1.0),
                );
                Ok(iter_result(i, mv, false))
            }
        },
        "filter" => {
            let mut k = count;
            loop {
                match step_iter_with(i, &src, &inext)? {
                    None => return Ok(iter_result(i, Value::Undefined, true)),
                    Some(v) => {
                        let r = match i.call(
                            f.clone(),
                            Value::Undefined,
                            &[v.clone(), Value::Num(k)],
                        ) {
                            Ok(r) => r,
                            Err(e) => {
                                i.iterator_close(&src);
                                return Err(crate::interpreter::abrupt_value(e));
                            }
                        };
                        k += 1.0;
                        if i.to_boolean(&r) {
                            set_internal(this.as_obj().unwrap(), "#\u{0}ih_count", Value::Num(k));
                            return Ok(iter_result(i, v, false));
                        }
                    }
                }
            }
        }
        "take" => {
            let nv = ab(i.get_member(&this, "#\u{0}ih_n"))?;
            let n = ab(i.to_number(&nv))?;
            if count >= n {
                set_internal(this.as_obj().unwrap(), "#\u{0}ih_done", Value::Bool(true));
                ab(i.iterator_close_normal(&src))?;
                return Ok(iter_result(i, Value::Undefined, true));
            }
            set_internal(
                this.as_obj().unwrap(),
                "#\u{0}ih_count",
                Value::Num(count + 1.0),
            );
            match step_iter_with(i, &src, &inext)? {
                None => Ok(iter_result(i, Value::Undefined, true)),
                Some(v) => Ok(iter_result(i, v, false)),
            }
        }
        "drop" => {
            let started_v = ab(i.get_member(&this, "#\u{0}ih_started"))?;
            let started = i.to_boolean(&started_v);
            if !started {
                let nv = ab(i.get_member(&this, "#\u{0}ih_n"))?;
                let n = ab(i.to_number(&nv))?;
                let mut skipped = 0.0;
                while skipped < n {
                    if step_iter_with(i, &src, &inext)?.is_none() {
                        // Exhausted while skipping: the helper completes here — no further
                        // step of the underlying iterator.
                        let o = this.as_obj().unwrap();
                        set_internal(o, "#\u{0}ih_started", Value::Bool(true));
                        set_internal(o, "#\u{0}ih_done", Value::Bool(true));
                        return Ok(iter_result(i, Value::Undefined, true));
                    }
                    skipped += 1.0;
                }
                set_internal(
                    this.as_obj().unwrap(),
                    "#\u{0}ih_started",
                    Value::Bool(true),
                );
            }
            match step_iter_with(i, &src, &inext)? {
                None => Ok(iter_result(i, Value::Undefined, true)),
                Some(v) => Ok(iter_result(i, v, false)),
            }
        }
        "chunks" => {
            let source_done = ab(i.get_member(&this, "#\u{0}ih_source_done"))?;
            if i.to_boolean(&source_done) {
                return Ok(iter_result(i, Value::Undefined, true));
            }
            let size = ab(i.get_member(&this, "#\u{0}ih_size"))?;
            let size = ab(i.to_number(&size))? as usize;
            let mut buffer = Vec::with_capacity(size.min(1024));
            loop {
                match step_iter_with(i, &src, &inext)? {
                    Some(value) => {
                        buffer.push(value);
                        if buffer.len() == size {
                            return Ok(iter_result(i, i.make_array(buffer), false));
                        }
                    }
                    None => {
                        set_internal(
                            this.as_obj().unwrap(),
                            "#\u{0}ih_source_done",
                            Value::Bool(true),
                        );
                        return if buffer.is_empty() {
                            Ok(iter_result(i, Value::Undefined, true))
                        } else {
                            Ok(iter_result(i, i.make_array(buffer), false))
                        };
                    }
                }
            }
        }
        "windows" => {
            let source_done = ab(i.get_member(&this, "#\u{0}ih_source_done"))?;
            if i.to_boolean(&source_done) {
                return Ok(iter_result(i, Value::Undefined, true));
            }
            let size = ab(i.get_member(&this, "#\u{0}ih_size"))?;
            let size = ab(i.to_number(&size))? as usize;
            let stored = ab(i.get_member(&this, "#\u{0}ih_buf"))?;
            let mut buffer = match stored {
                Value::Obj(o) => {
                    let len = i.array_length(&o);
                    (0..len)
                        .map(|index| {
                            o.borrow()
                                .props
                                .get(&index.to_string())
                                .map(|p| p.value())
                                .unwrap_or(Value::Undefined)
                        })
                        .collect::<VecDeque<_>>()
                }
                _ => VecDeque::new(),
            };
            loop {
                match step_iter_with(i, &src, &inext)? {
                    Some(value) => {
                        if buffer.len() == size {
                            buffer.pop_front();
                        }
                        buffer.push_back(value);
                        if buffer.len() == size {
                            let window = buffer.iter().cloned().collect::<Vec<_>>();
                            set_internal(
                                this.as_obj().unwrap(),
                                "#\u{0}ih_buf",
                                i.make_array(window.clone()),
                            );
                            return Ok(iter_result(i, i.make_array(window), false));
                        }
                    }
                    None => {
                        set_internal(
                            this.as_obj().unwrap(),
                            "#\u{0}ih_source_done",
                            Value::Bool(true),
                        );
                        let mode = ab(i.get_member(&this, "#\u{0}ih_undersized"))?;
                        let allow_partial =
                            matches!(mode, Value::Str(s) if s.to_string() == "allow-partial");
                        return if allow_partial && !buffer.is_empty() && buffer.len() < size {
                            Ok(iter_result(
                                i,
                                i.make_array(buffer.into_iter().collect()),
                                false,
                            ))
                        } else {
                            Ok(iter_result(i, Value::Undefined, true))
                        };
                    }
                }
            }
        }
        "flatMap" => {
            let mut c = count;
            loop {
                // Drain the active inner iterator first (lazily — it may be infinite).
                let inner = ab(i.get_member(&this, "#\u{0}ih_inner"))?;
                if matches!(inner, Value::Obj(_)) {
                    let inext = ab(i.get_member(&this, "#\u{0}ih_inner_next"))?;
                    match step_iter_with(i, &inner, &inext) {
                        Ok(Some(v)) => return Ok(iter_result(i, v, false)),
                        Ok(None) => {
                            set_internal(
                                this.as_obj().unwrap(),
                                "#\u{0}ih_inner",
                                Value::Undefined,
                            );
                            continue;
                        }
                        Err(e) => {
                            // An abrupt inner step closes the outer iterator with the same error.
                            i.iterator_close(&src);
                            return Err(e);
                        }
                    }
                }
                // Refill: map the next outer value to an iterable and open it.
                match step_iter_with(i, &src, &inext)? {
                    None => return Ok(iter_result(i, Value::Undefined, true)),
                    Some(v) => {
                        let mapped = match i.call(f.clone(), Value::Undefined, &[v, Value::Num(c)])
                        {
                            Ok(m) => m,
                            Err(e) => {
                                i.iterator_close(&src);
                                return Err(crate::interpreter::abrupt_value(e));
                            }
                        };
                        c += 1.0;
                        set_internal(this.as_obj().unwrap(), "#\u{0}ih_count", Value::Num(c));
                        // GetIteratorFlattenable (reject primitives), fetching `next` once.
                        let opened = (|i: &mut Interp| -> Result<(Value, Value), Value> {
                            let it = get_iterator_flattenable(i, &mapped, false)?;
                            let n = ab(i.get_member(&it, "next"))?;
                            Ok((it, n))
                        })(i);
                        match opened {
                            Ok((it, n)) => {
                                set_internal(this.as_obj().unwrap(), "#\u{0}ih_inner", it);
                                set_internal(this.as_obj().unwrap(), "#\u{0}ih_inner_next", n);
                            }
                            Err(e) => {
                                i.iterator_close(&src);
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        _ => Ok(iter_result(i, Value::Undefined, true)),
    }
}

/// Build an Array Iterator over `target`. `kind`: 0 = values, 1 = keys, 2 = [key, value] entries.
/// ECMA-262 e28783d5, CreateArrayIterator: private slots, independent of own properties.
fn make_array_iterator(i: &mut Interp, target: Value, kind: u8) -> Value {
    let proto = i
        .extra_protos
        .get("%ArrayIteratorPrototype%")
        .cloned()
        .or_else(|| i.extra_protos.get("%IteratorPrototype%").cloned());
    let obj = Object::new_with_parts(
        proto,
        crate::value::Props::new(),
        Exotic::ArrayIterator(Box::new(crate::value::ArrayIteratorState {
            target,
            index: 0,
            kind,
        })),
    );
    Value::Obj(obj)
}

/// `next()` for a String Iterator: advance one code point through the private closure state.
fn string_iter_next(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let step = string_iter_step(i, this)?;
    Ok(materialize_iterator_step(i, step))
}

fn string_iter_step(i: &mut Interp, this: Value) -> Result<Option<Value>, Value> {
    let Some(object) = this.as_obj() else {
        return Err(i.make_error(
            "TypeError",
            "String Iterator next called on an incompatible receiver",
        ));
    };
    let mut object = object.borrow_mut();
    let Exotic::StringIterator(state) = &mut object.exotic else {
        drop(object);
        return Err(i.make_error(
            "TypeError",
            "String Iterator next called on an incompatible receiver",
        ));
    };
    let Some(s) = &state.string else {
        return Ok(None);
    };
    let idx = state.index;
    // An ASCII String has one UTF-16 code unit per byte.  Keep the iterator's observable
    // code-point semantics while avoiding UTF-8 char decoding and a fresh allocation for each
    // result; the shared single-unit strings are immutable and therefore valid String values.
    if s.ascii_hint() {
        let Some(&unit) = s.as_str().as_bytes().get(idx) else {
            state.string = None;
            return Ok(None);
        };
        state.index = idx + 1;
        return Ok(Some(Value::Str(crate::jstr::unit_lstr(unit as u16))));
    }
    let mut rest = s[idx.min(s.len())..].chars();
    let Some(ch) = rest.next() else {
        // Exhausted: clear the string so the iterator stays done.
        state.string = None;
        return Ok(None);
    };
    // A smuggled surrogate pair is one code point: yield both scalars together.
    if let Some(next) = rest.next() {
        if crate::jstr::paired_char(ch, next).is_some() {
            state.index = idx + ch.len_utf8() + next.len_utf8();
            let mut both = String::new();
            both.push(ch);
            both.push(next);
            return Ok(Some(Value::lstr(both)));
        }
    }
    state.index = idx + ch.len_utf8();
    Ok(Some(Value::lstr(ch.to_string())))
}

/// `next()` for a generator object built by `make_generator`: walk the buffered values, then throw
/// any stored error, then return `{ value: <return>, done: true }`.
/// Drive a generator's coroutine with a resume signal, returning its `{value, done}` (or propagating
/// a throw). The generators map doubles as the brand check.
fn drive_generator(
    i: &mut Interp,
    this: &Value,
    signal: crate::coroutine::Resume,
) -> Result<Value, Value> {
    use crate::coroutine::{Resume, Suspend};
    let key = match this {
        Value::Obj(o) => Rc::as_ptr(o) as usize,
        _ => return Err(i.make_error("TypeError", "Generator method called on a non-generator")),
    };
    let mut coro = match i.generators.remove(&key) {
        Some(c) => c,
        None => return Err(i.make_error("TypeError", "Generator method called on a non-generator")),
    };
    if coro.done() {
        i.generators.insert(key, coro);
        return match signal {
            Resume::Throw(e) => Err(e),
            Resume::Return(v) => Ok(iter_result(i, v, true)),
            Resume::Next(_) => Ok(iter_result(i, Value::Undefined, true)),
            Resume::Terminate => Ok(iter_result(i, Value::Undefined, true)),
        };
    }
    let suspend = coro.resume(i, signal);
    i.generators.insert(key, coro);
    // A `yield*` forwards the inner iterator's result object unwrapped.
    let raw = std::mem::take(&mut i.yield_raw_result);
    match suspend {
        Suspend::Yield(v) if raw && matches!(v, Value::Obj(_)) => Ok(v),
        Suspend::Yield(v) => Ok(iter_result(i, v, false)),
        Suspend::Done(v) => Ok(iter_result(i, v, true)),
        Suspend::Throw(e) => Err(e),
        Suspend::Await(_) => Err(i.make_error("TypeError", "await in a non-async generator")),
    }
}

pub(crate) fn generator_next(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    drive_generator(i, &this, crate::coroutine::Resume::Next(arg(args, 0)))
}

/// `gen.return(v)`: inject a return completion at the suspended `yield` (runs any `finally`).
pub(crate) fn generator_return(
    i: &mut Interp,
    this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    drive_generator(i, &this, crate::coroutine::Resume::Return(arg(args, 0)))
}

/// `gen.throw(e)`: inject a throw at the suspended `yield`.
pub(crate) fn generator_throw(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    drive_generator(i, &this, crate::coroutine::Resume::Throw(arg(args, 0)))
}
fn async_gen_drive(
    i: &mut Interp,
    this: &Value,
    signal: crate::coroutine::Resume,
) -> Result<Value, Value> {
    let r = i.new_promise();
    match this {
        // Brand check: the receiver must be an actual async generator instance; a mismatch
        // rejects the returned promise rather than throwing.
        Value::Obj(o) if i.async_gens.contains(&(Rc::as_ptr(o) as usize)) => {
            i.drive_async_gen(Rc::as_ptr(o) as usize, r.clone(), signal)
        }
        _ => {
            let e = i.make_error(
                "TypeError",
                "AsyncGenerator method called on an incompatible receiver",
            );
            i.reject_promise(&r, e);
        }
    }
    Ok(r)
}
pub(crate) fn async_generator_next(
    i: &mut Interp,
    this: Value,
    a: &[Value],
) -> Result<Value, Value> {
    async_gen_drive(i, &this, crate::coroutine::Resume::Next(arg(a, 0)))
}
pub(crate) fn async_generator_return(
    i: &mut Interp,
    this: Value,
    a: &[Value],
) -> Result<Value, Value> {
    async_gen_drive(i, &this, crate::coroutine::Resume::Return(arg(a, 0)))
}
pub(crate) fn async_generator_throw(
    i: &mut Interp,
    this: Value,
    a: &[Value],
) -> Result<Value, Value> {
    async_gen_drive(i, &this, crate::coroutine::Resume::Throw(arg(a, 0)))
}
/// Microtask reactions that re-drive an async generator when an awaited value settles. `args` is
/// `[generator, resultPromise, settledValue]`.
pub(crate) fn async_gen_react_fulfil(
    i: &mut Interp,
    _t: Value,
    a: &[Value],
) -> Result<Value, Value> {
    let key = match arg(a, 0) {
        Value::Obj(object) => Rc::as_ptr(&object) as usize,
        _ => return Ok(Value::Undefined),
    };
    i.drive_async_gen_inner(key, arg(a, 1), crate::coroutine::Resume::Next(arg(a, 2)));
    Ok(Value::Undefined)
}
pub(crate) fn async_gen_react_reject(
    i: &mut Interp,
    _t: Value,
    a: &[Value],
) -> Result<Value, Value> {
    let key = match arg(a, 0) {
        Value::Obj(object) => Rc::as_ptr(&object) as usize,
        _ => return Ok(Value::Undefined),
    };
    i.drive_async_gen_inner(key, arg(a, 1), crate::coroutine::Resume::Throw(arg(a, 2)));
    Ok(Value::Undefined)
}
/// AsyncGeneratorAwaitReturn settled: complete the generator and settle the request promise.
/// `args` is `[generator, resultPromise, settledValue]`.
pub(crate) fn async_gen_return_fulfil(
    i: &mut Interp,
    _t: Value,
    a: &[Value],
) -> Result<Value, Value> {
    let key = match arg(a, 0) {
        Value::Obj(object) => Rc::as_ptr(&object) as usize,
        _ => return Ok(Value::Undefined),
    };
    let (r, x) = (arg(a, 1), arg(a, 2));
    if let Some(mut coro) = i.generators.remove(&key) {
        if !coro.done() {
            let _ = coro.resume(i, crate::coroutine::Resume::Return(x.clone()));
        }
        i.generators.insert(key, coro);
    }
    let res = i.iter_result_obj(x, true);
    i.resolve_promise(&r, res);
    i.finish_async_gen_step(key);
    Ok(Value::Undefined)
}
pub(crate) fn async_gen_return_reject(
    i: &mut Interp,
    _t: Value,
    a: &[Value],
) -> Result<Value, Value> {
    let key = match arg(a, 0) {
        Value::Obj(object) => Rc::as_ptr(&object) as usize,
        _ => return Ok(Value::Undefined),
    };
    let (r, e) = (arg(a, 1), arg(a, 2));
    if let Some(mut coro) = i.generators.remove(&key) {
        if !coro.done() {
            let _ = coro.resume(i, crate::coroutine::Resume::Throw(e.clone()));
        }
        i.generators.insert(key, coro);
    }
    i.reject_promise(&r, e);
    i.finish_async_gen_step(key);
    Ok(Value::Undefined)
}
pub(crate) fn async_iterator_key(i: &Interp) -> Option<Rc<str>> {
    well_known_key(i, "asyncIterator")
}

fn array_iter_next(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let step = array_iter_step(i, this)?;
    Ok(materialize_iterator_step(i, step))
}

fn array_iter_step(i: &mut Interp, this: Value) -> Result<Option<Value>, Value> {
    let Some(object) = this.as_obj() else {
        return Err(i.make_error(
            "TypeError",
            "Array Iterator next called on an incompatible receiver",
        ));
    };
    // Release this borrow before LengthOfArrayLike or indexed Get can reenter
    // JavaScript. The index snapshot and update order follow the native slots.
    let (target, idx, kind) = {
        let object = object.borrow();
        let Exotic::ArrayIterator(state) = &object.exotic else {
            return Err(i.make_error(
                "TypeError",
                "Array Iterator next called on an incompatible receiver",
            ));
        };
        (state.target.clone(), state.index, state.kind)
    };
    // An exhausted iterator clears its target so it stays done even if the source later grows.
    if matches!(target, Value::Undefined) {
        return Ok(None);
    }
    // A TypedArray target re-derives its length each step; an out-of-bounds (detached/shrunk-past)
    // view throws TypeError.
    let typed_info = map_ptr(&target).and_then(|p| i.typed_arrays.get(&p).copied());
    let len = if let Some(info) = typed_info {
        match i.ta_len(&info) {
            Some(l) => l,
            None => return Err(i.make_error("TypeError", "TypedArray is out of bounds")),
        }
    } else {
        match &target {
            // LengthOfArrayLike is a [[Get]] of "length". An ordinary Array's length is an
            // array-exotic data slot, never an accessor, so reading the maintained slot is the
            // same operation without allocating the property key or coercing a Number. Proxies,
            // host objects, and ordinary array-likes retain the observable generic path.
            Value::Obj(object)
                if matches!(object.borrow().exotic, Exotic::Array)
                    && i.ordinary_get_ptr(Rc::as_ptr(object) as usize) =>
            {
                i.array_length(object)
            }
            // LengthOfArrayLike through [[Get]], so a proxy target's traps are honored.
            Value::Obj(_) => {
                let lv = ab(i.get_member(&target, "length"))?;
                let n = ab(i.to_number(&lv))?;
                if n.is_nan() || n <= 0.0 {
                    0
                } else {
                    n.min(9007199254740991.0) as usize
                }
            }
            Value::Str(s) => crate::jstr::unit_len(s),
            _ => 0,
        }
    };
    if idx >= len {
        let mut object = object.borrow_mut();
        let Exotic::ArrayIterator(state) = &mut object.exotic else {
            unreachable!()
        };
        state.target = Value::Undefined;
        return Ok(None);
    }
    {
        let mut object = object.borrow_mut();
        let Exotic::ArrayIterator(state) = &mut object.exotic else {
            unreachable!()
        };
        state.index = idx + 1;
    }
    // %ArrayIteratorPrototype%.next requests indexed Get only for values/entries.
    // A keys iterator must not invoke a getter at the corresponding array index.
    if kind == 1 {
        return Ok(Some(Value::Num(idx as f64)));
    }
    // ArrayIteratorPrototype.next performs Get(array, ToString(index)). The ordinary dense
    // element helper is an exact result for own data entries and falls back for holes, accessors,
    // prototype properties, and exotics; this avoids the per-step index-string allocation in the
    // common packed-array case while preserving the specified Get semantics on every miss.
    let elem = if let Some(info) = typed_info {
        // ValidateTypedArrayBounds above succeeded for this step. TypedArray [[Get]] of a
        // canonical in-range integer index is the backing-store read, so avoid reconstructing a
        // property key and dispatching through the generic indexed-property machinery.
        i.ta_read(&info, idx)
    } else {
        ab(match &target {
            Value::Obj(object) => i
                .fast_get_elem(object, idx as f64)
                .map(Ok)
                .unwrap_or_else(|| i.get_member(&target, &idx.to_string())),
            _ => i.get_member(&target, &idx.to_string()),
        })?
    };
    let value = match kind {
        2 => i.make_array(vec![Value::Num(idx as f64), elem]),
        _ => elem,
    };
    Ok(Some(value))
}

fn norm_index(n: f64, len: i64) -> i64 {
    // ToIntegerOrInfinity: NaN maps to 0 (undefined args are handled by callers).
    if n.is_nan() {
        return 0;
    }
    let i = if n.is_infinite() {
        if n > 0.0 {
            len
        } else {
            -len - 1
        }
    } else {
        n as i64
    };
    if i < 0 {
        (len + i).max(0)
    } else {
        i.min(len)
    }
}

fn same_value_zero(a: &Value, b: &Value) -> bool {
    if let (Value::Num(x), Value::Num(y)) = (a, b) {
        if x.is_nan() && y.is_nan() {
            return true;
        }
    }
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x == y,
        _ => same_value(a, b),
    }
}

// ---------------------------------------------------------------------------------------------
// String / Number / Boolean / Math / errors / globals
// ---------------------------------------------------------------------------------------------

/// Canonicalize the locale argument to toLocale{Lower,Upper}Case and return its language subtag
/// (lowercased), or `None` when no locale was supplied. Invalid locales throw RangeError.
#[cfg(feature = "intl")]
fn locale_case_lang(i: &mut Interp, locales: &Value) -> Result<Option<String>, Value> {
    let list = crate::intl::canonicalize_locale_list(i, locales)?;
    Ok(list
        .first()
        .and_then(|l| l.split('-').next())
        .map(|s| s.to_ascii_lowercase()))
}

/// Without ECMA-402 the locale argument is ignored: default (root) case mapping.
#[cfg(not(feature = "intl"))]
fn locale_case_lang(_i: &mut Interp, _locales: &Value) -> Result<Option<String>, Value> {
    Ok(None)
}

/// Language-sensitive lowercasing. Turkish/Azeri map the dotted/dotless I pair specially; all other
/// languages use the default Unicode mapping.
/// The canonical combining class used by the SpecialCasing conditions (0 = base).
fn ccc(cp: u32) -> u8 {
    crate::unicode_norm_impl::ccc(cp)
}

/// From `chars[k+1..]`, the index of a COMBINING DOT ABOVE reachable across marks of combining
/// class other than 0/230 (the SpecialCasing After_I / After_Soft_Dotted scan), if any.
fn absorbable_dot_above(chars: &[char], k: usize) -> Option<usize> {
    let mut j = k + 1;
    while let Some(&m) = chars.get(j) {
        if m == '\u{0307}' {
            return Some(j);
        }
        match ccc(m as u32) {
            0 | 230 => return None,
            _ => j += 1,
        }
    }
    None
}

/// SpecialCasing More_Above: `chars[k]` is followed by a class-230 mark with no intervening
/// class 0/230 character.
fn more_above(chars: &[char], k: usize) -> bool {
    let mut j = k + 1;
    while let Some(&m) = chars.get(j) {
        match ccc(m as u32) {
            230 => return true,
            0 => return false,
            _ => j += 1,
        }
    }
    false
}

/// Unicode Soft_Dotted (the subset with lowercase dots that interact with case mapping).
fn is_soft_dotted(cp: u32) -> bool {
    matches!(
        cp,
        0x69 | 0x6A
            | 0x12F
            | 0x249
            | 0x268
            | 0x29D
            | 0x2B2
            | 0x3F3
            | 0x456
            | 0x458
            | 0x1D62
            | 0x1D96
            | 0x1DA4
            | 0x1DA8
            | 0x1E2D
            | 0x1ECB
            | 0x2071
            | 0x2148..=0x2149
            | 0x2C7C
            | 0x1D422..=0x1D423
            | 0x1D456..=0x1D457
            | 0x1D48A..=0x1D48B
            | 0x1D4BE..=0x1D4BF
            | 0x1D4F2..=0x1D4F3
            | 0x1D526..=0x1D527
            | 0x1D55A..=0x1D55B
            | 0x1D58E..=0x1D58F
            | 0x1D5C2..=0x1D5C3
            | 0x1D5F6..=0x1D5F7
            | 0x1D62A..=0x1D62B
            | 0x1D65E..=0x1D65F
            | 0x1D692..=0x1D693
            | 0x1DF1A
            | 0x1E04C..=0x1E04D
            | 0x1E068
    )
}

fn locale_lower(s: &str, lang: Option<&str>) -> String {
    match lang {
        Some("tr") | Some("az") => {
            let chars: Vec<char> = s.chars().collect();
            let mut out = String::with_capacity(s.len());
            let mut k = 0;
            while k < chars.len() {
                match chars[k] {
                    '\u{0130}' => out.push('i'), // İ → i
                    'I' => {
                        // After_I: a following combining dot above (reachable across non-0/230
                        // marks) is absorbed and I lowercases dotted; otherwise → dotless ı.
                        if let Some(dj) = absorbable_dot_above(&chars, k) {
                            out.push('i');
                            for &m in &chars[k + 1..dj] {
                                out.push(m);
                            }
                            k = dj;
                        } else {
                            out.push('\u{0131}');
                        }
                    }
                    c => out.extend(c.to_lowercase()),
                }
                k += 1;
            }
            out
        }
        Some("lt") => {
            // Lithuanian retains the dot above a lowercased I/J/Į when further above-marks
            // follow, and bakes it into the dotted-accent capitals.
            let chars: Vec<char> = s.chars().collect();
            let mut out = String::with_capacity(s.len());
            for (k, &c) in chars.iter().enumerate() {
                match c {
                    'I' if more_above(&chars, k) => out.push_str("i\u{0307}"),
                    'J' if more_above(&chars, k) => out.push_str("j\u{0307}"),
                    '\u{012E}' if more_above(&chars, k) => out.push_str("\u{012F}\u{0307}"),
                    '\u{00CC}' => out.push_str("i\u{0307}\u{0300}"),
                    '\u{00CD}' => out.push_str("i\u{0307}\u{0301}"),
                    '\u{0128}' => out.push_str("i\u{0307}\u{0303}"),
                    c => out.extend(c.to_lowercase()),
                }
            }
            out
        }
        _ => s.to_lowercase(),
    }
}

/// Language-sensitive uppercasing (Turkish/Azeri dotted-I, Lithuanian dot-above removal).
fn locale_upper(s: &str, lang: Option<&str>) -> String {
    match lang {
        Some("tr") | Some("az") => {
            let mut out = String::with_capacity(s.len());
            for c in s.chars() {
                match c {
                    'i' => out.push('\u{0130}'), // i → İ
                    '\u{0131}' => out.push('I'), // ı → I
                    c => out.extend(c.to_uppercase()),
                }
            }
            out
        }
        Some("lt") => {
            // A combining dot above after a soft-dotted base disappears when uppercasing.
            let chars: Vec<char> = s.chars().collect();
            let mut out = String::with_capacity(s.len());
            let mut k = 0;
            while k < chars.len() {
                let c = chars[k];
                out.extend(c.to_uppercase());
                if is_soft_dotted(c as u32) {
                    if let Some(dj) = absorbable_dot_above(&chars, k) {
                        for &m in &chars[k + 1..dj] {
                            out.push(m);
                        }
                        k = dj;
                    }
                }
                k += 1;
            }
            out
        }
        _ => s.to_uppercase(),
    }
}

/// The ToUint16 code unit of a Number (ECMA-262 §7.1.9), as `String.fromCharCode` applies it.
pub(crate) fn uint16_of_number(num: f64) -> u16 {
    if num.is_finite() {
        num.trunc().rem_euclid(65536.0) as u16
    } else {
        0
    }
}

/// `String.fromCharCode` (named so the JIT call cache can prove and tag its exact identity).
pub(crate) fn nf_from_char_code(
    i: &mut crate::interpreter::Interp,
    _this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    // One Number argument is the decoder idiom (`String.fromCharCode(c)` per character):
    // ToUint16 it and share the interned one-unit ASCII strings instead of allocating.
    if let [Value::Num(num)] = args {
        return Ok(Value::Str(crate::jstr::unit_lstr(uint16_of_number(*num))));
    }
    let mut units: Vec<u16> = Vec::with_capacity(args.len());
    for a in args {
        // Each argument is ToUint16'd (so -1 -> 0xFFFF, 0x10000 -> 0), not truncated.
        // ToNumber is the identity for Numbers (ECMA-262 §7.1.4); other inputs keep
        // the full coercive path with its throws and valueOf ordering.
        let num = match a {
            Value::Num(num) => *num,
            v => ab(i.to_number(v))?,
        };
        units.push(uint16_of_number(num));
    }
    // UTF-16 decode: a surrogate pair combines into one code point; a lone half is smuggled
    // (see `jstr`), so the resulting string round-trips its exact unit sequence.
    Ok(Value::lstr(crate::jstr::from_units(&units)))
}

/// `String.prototype.charCodeAt` (named: the JIT's call-IC fill compares the fn pointer to
/// tag intrinsic entries — see `bytecode::INTRINSIC_CHAR_CODE_AT`).
pub(crate) fn nf_char_code_at(
    i: &mut crate::interpreter::Interp,
    this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    let n = ab(i.to_number(&arg(args, 0)))?;
    let idx = if n.is_nan() { 0.0 } else { n.trunc() };
    if idx < 0.0 || !idx.is_finite() {
        return Ok(Value::Num(f64::NAN));
    }
    Ok(match i.unit_at(&s, idx as usize) {
        Some(u) => Value::Num(u as f64),
        None => Value::Num(f64::NAN),
    })
}

/// `String.prototype.charAt` (named so the JIT call IC can prove and tag its exact identity).
pub(crate) fn nf_char_at(
    i: &mut crate::interpreter::Interp,
    this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    let n = ab(i.to_number(&arg(args, 0)))?;
    let idx = if n.is_nan() { 0.0 } else { n.trunc() };
    if idx < 0.0 || !idx.is_finite() {
        return Ok(Value::lstr(""));
    }
    Ok(match i.unit_at(&s, idx as usize) {
        Some(u) => Value::Str(crate::jstr::unit_lstr(u)),
        None => Value::lstr(""),
    })
}

/// Named builtin identity for the guarded numeric-index native intrinsic.
pub(crate) fn nf_code_point_at(
    i: &mut Interp,
    this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    let n = arg_to_number(i, args, 0)?;
    let n = if n.is_nan() { 0.0 } else { n.trunc() };
    Ok(if n < 0.0 || !n.is_finite() {
        Value::Undefined
    } else {
        i.code_point_at(&s, n as usize)
            .map_or(Value::Undefined, |point| Value::Num(point as f64))
    })
}

fn this_string(i: &mut Interp, this: &Value) -> Result<crate::lstr::LStr, Value> {
    match this {
        Value::Str(s) => Ok(s.clone()),
        Value::Obj(o) => match &o.borrow().exotic {
            Exotic::StrWrap(s) => Ok((**s).clone()),
            _ => ab(i.to_string(this)),
        },
        // String.prototype methods RequireObjectCoercible(this): null/undefined → TypeError.
        Value::Undefined | Value::Null => Err(i.make_error(
            "TypeError",
            "String.prototype method called on null or undefined",
        )),
        _ => ab(i.to_string(this)),
    }
}

/// The same WhiteSpace + LineTerminator set as RegExp's `\s`: unlike Rust's Unicode
/// whitespace predicate, it includes U+FEFF but excludes U+0085 (NEL).
fn is_js_ws(c: char) -> bool {
    crate::regex::js_whitespace(c)
}

/// IsRegExp(arg): true if it has a truthy `@@match`, or (fallback) is a compiled RegExp object.
fn arg_is_regexp(i: &mut Interp, v: &Value) -> Result<bool, Value> {
    if let Value::Obj(o) = v {
        if let Some(k) = well_known_key(i, "match") {
            let m = ab(i.get_member(v, &k))?;
            if !matches!(m, Value::Undefined | Value::Null) {
                return Ok(i.to_boolean(&m));
            }
        }
        return Ok(i.regexps.contains_key(&(Rc::as_ptr(o) as usize)));
    }
    Ok(false)
}

/// ToIntegerOrInfinity of an optional position argument, clamped to `[0, len]`.
fn str_clamp_pos(i: &mut Interp, v: Option<&Value>, len: i64) -> Result<usize, Value> {
    // `Value::Num` is the ToNumber identity (§7.1.4.1); everything else keeps the coercive path.
    let n = match v {
        Some(Value::Num(n)) => *n,
        Some(v) if !matches!(v, Value::Undefined) => ab(i.to_number(v))?,
        _ => 0.0,
    };
    let n = if n.is_nan() { 0 } else { n.trunc() as i64 };
    Ok(n.clamp(0, len) as usize)
}

/// CreateHTML (Annex B): wrap the coerced `this` string in `<tag attr="value">…</tag>`.
fn create_html(
    i: &mut Interp,
    this: Value,
    tag: &str,
    attr: &str,
    value: &Value,
) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    let mut p = format!("<{tag}");
    if !attr.is_empty() {
        let v = ab(i.to_string(value))?;
        p.push_str(&format!(" {attr}=\"{}\"", v.replace('"', "&quot;")));
    }
    Ok(Value::lstr(format!("{p}>{s}</{tag}>")))
}

pub(crate) fn nf_string_replace(
    i: &mut Interp,
    this: Value,
    args: &[Value],
) -> Result<Value, Value> {
    if matches!(this, Value::Undefined | Value::Null) {
        return Err(i.make_error(
            "TypeError",
            "String.prototype.replace called on null or undefined",
        ));
    }
    // An object search value with a @@replace method handles the operation.
    let search = arg(args, 0);
    if matches!(search, Value::Obj(_)) {
        if let Some(key) = well_known_key(i, "replace") {
            let replacer = ab(i.get_member(&search, &key))?;
            if !matches!(replacer, Value::Undefined | Value::Null) {
                if !replacer.is_callable() {
                    return Err(i.make_error("TypeError", "@@replace is not callable"));
                }
                return ab(i.call(replacer, search.clone(), &[this.clone(), arg(args, 1)]));
            }
        }
    }
    // ECMA-262 §22.1.3.19 returns `string` unchanged when StringIndexOf finds no match.
    // Keep the already-coerced LStr in that branch: converting it to a Rust String and then
    // allocating a second engine string is pure overhead, while replacement coercion above still
    // preserves the specified observable ordering.
    let s = this_string(i, &this)?;
    let pat = ab(i.to_string(&arg(args, 0)))?;
    let repl = prep_repl(i, &arg(args, 1))?;
    // `ascii_hint` is representation-local and avoids an LRU lookup on every hot literal call;
    // a cleared hint conservatively falls through to the exact UTF-16 path.
    let source_ascii = s.ascii_hint();
    let pattern_ascii = pat.ascii_hint();
    if source_ascii && pattern_ascii {
        let Some(pos) = s.as_str().find(pat.as_ref()) else {
            return Ok(Value::Str(s));
        };
        let matched = &s[pos..pos + pat.len()];
        let rep = string_replacement(i, &repl, matched, &s, pos)?;
        return Ok(Value::lstr(format!(
            "{}{}{}",
            &s[..pos],
            rep,
            &s[pos + pat.len()..]
        )));
    }
    // A pure-ASCII receiver cannot contain a non-ASCII search string. This check avoids
    // materializing the receiver's UTF-16 cache for the guaranteed no-match case.
    if source_ascii {
        return Ok(Value::Str(s));
    }
    // StringIndexOf is defined over UTF-16 code units (ECMA-262 §6.1.4.1). The engine stores
    // UTF-8, so a non-ASCII search uses the cached unit view; this also permits matching one half
    // of an astral code point, which has no UTF-8 byte boundary.
    let source_units = i.units_full(&s);
    let pattern_units = i.units_full(&pat);
    let Some(pos) = source_units
        .windows(pattern_units.len())
        .position(|window| window == pattern_units.as_ref())
    else {
        return Ok(Value::Str(s));
    };
    let end = pos + pattern_units.len();
    let before = crate::jstr::from_units(&source_units[..pos]);
    let matched = crate::jstr::from_units(&source_units[pos..end]);
    let after = crate::jstr::from_units(&source_units[end..]);
    let rep = string_replacement_parts(i, &repl, &matched, &s, pos, &before, &after)?;
    let joined = crate::jstr::concat(&crate::jstr::concat(&before, &rep), &after);
    Ok(Value::lstr(
        crate::jstr::canonicalize(&joined).unwrap_or(joined),
    ))
}

pub(crate) fn nf_string_match(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    // RequireObjectCoercible(this), then dispatch to an Object regexp's @@match.
    if matches!(this, Value::Undefined | Value::Null) {
        return Err(i.make_error(
            "TypeError",
            "String.prototype.match called on null or undefined",
        ));
    }
    let regexp = arg(a, 0);
    if matches!(regexp, Value::Obj(_)) {
        if let Some(key) = well_known_key(i, "match") {
            let matcher = ab(i.get_member(&regexp, &key))?;
            if !matches!(matcher, Value::Undefined | Value::Null) {
                if !matcher.is_callable() {
                    return Err(i.make_error("TypeError", "@@match is not callable"));
                }
                return ab(i.call(matcher, regexp.clone(), std::slice::from_ref(&this)));
            }
        }
    }
    // Otherwise build a RegExp from the argument and invoke its @@match.
    let s = this_string(i, &this)?;
    let pattern = if matches!(regexp, Value::Undefined) {
        String::new()
    } else {
        ab(i.to_string(&regexp))?.to_string()
    };
    let rx = ab(i.make_regexp(&pattern, ""))?;
    let key = well_known_key(i, "match").unwrap();
    let matcher = ab(i.get_member(&rx, &key))?;
    ab(i.call(matcher, rx, &[Value::Str(s)]))
}

pub(crate) fn nf_string_split(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    // RequireObjectCoercible(this), then dispatch to an Object separator's @@split.
    if matches!(this, Value::Undefined | Value::Null) {
        return Err(i.make_error(
            "TypeError",
            "String.prototype.split called on null or undefined",
        ));
    }
    let separator = arg(args, 0);
    if matches!(separator, Value::Obj(_)) {
        if let Some(key) = well_known_key(i, "split") {
            let splitter = ab(i.get_member(&separator, &key))?;
            if !matches!(splitter, Value::Undefined | Value::Null) {
                if !splitter.is_callable() {
                    return Err(i.make_error("TypeError", "@@split is not callable"));
                }
                return ab(i.call(splitter, separator.clone(), &[this.clone(), arg(args, 1)]));
            }
        }
    }
    let s = this_string(i, &this)?;
    if s.len() > MAX_ARRAY_OP_LEN {
        return Err(i.make_error("RangeError", "string too large to split in this engine"));
    }
    // `limit` (ToUint32) caps the number of pieces; 0 → empty result — but ToString of the
    // separator is observable BEFORE the zero-limit shortcut.
    let limit = match arg(args, 1) {
        Value::Undefined => u32::MAX as usize,
        v => {
            let n = ab(i.to_number(&v))?;
            (if n.is_finite() { n as i64 as u32 } else { 0 }) as usize
        }
    };
    let is_live_regexp =
        matches!(&arg(args, 0), Value::Obj(o) if i.regexps.contains_key(&(Rc::as_ptr(o) as usize)));
    let sep_str: Option<Rc<str>> = match &arg(args, 0) {
        Value::Undefined => None,
        _ if is_live_regexp => None,
        v => Some(ab(i.to_string(v))?.into()),
    };
    if limit == 0 {
        return Ok(i.make_array(Vec::new()));
    }
    // Regex separator: split on each match (group captures are inserted between pieces).
    if let Value::Obj(o) = &arg(args, 0) {
        if i.regexps.contains_key(&(Rc::as_ptr(o) as usize)) {
            let re = i.regexps[&(Rc::as_ptr(o) as usize)].clone();
            let text = i.re_text(re.unicode, &s);
            let mut parts = Vec::new();
            let mut last = 0;
            'outer: for caps in regex_find_all(i, &re, &text)? {
                let (a, b) = caps[0].unwrap();
                // Skip a zero-width match at the very start or end of the string.
                if a == b && (b == 0 || a >= text.len()) {
                    continue;
                }
                parts.push(Value::lstr(text.slice(last, a)));
                if parts.len() >= limit {
                    break;
                }
                for g in 1..=re.ngroups {
                    parts.push(match caps[g] {
                        Some((x, y)) => Value::lstr(text.slice(x, y)),
                        None => Value::Undefined,
                    });
                    if parts.len() >= limit {
                        break 'outer;
                    }
                }
                last = b;
            }
            if parts.len() < limit {
                parts.push(Value::lstr(text.slice(last, text.len())));
            }
            parts.truncate(limit);
            return Ok(i.make_array(parts));
        }
    }
    match sep_str {
        None => Ok(i.make_array(vec![Value::Str(s)])),
        Some(sep) => {
            // Splitting on "" yields one piece per UTF-16 code unit (a surrogate pair
            // becomes its two lone halves).
            let mut parts: Vec<Value> = if sep.is_empty() {
                crate::jstr::units(&s)
                    .into_iter()
                    .map(|u| Value::Str(crate::jstr::unit_lstr(u)))
                    .collect()
            } else if s.ascii_hint() && sep.is_ascii() {
                // For ASCII operands, byte and UTF-16-unit boundaries coincide. Keep each
                // substring as an LStr directly instead of allocating a Rust String before the
                // engine copies it; `.take(limit)` preserves Split's post-processing truncation
                // (unlike `splitn`, whose final remainder would be incorrect here).
                s.split(sep.as_ref()).take(limit).map(Value::lstr).collect()
            } else {
                // String separators are matched by UTF-16 code units (ECMA-262 §22.1.3.23),
                // not Rust scalar values. This matters when a separator is one half of an astral
                // character: splitting `"😀"` on its high surrogate must leave the low surrogate
                // in the second result. Convert once, search the unit sequence, and rebuild each
                // piece independently so a separator boundary cannot accidentally recombine the
                // pair across pieces.
                let units = crate::jstr::units(&s);
                let sep_units = crate::jstr::units(sep.as_ref());
                let mut parts = Vec::new();
                let mut start = 0usize;
                while start <= units.len() && parts.len() < limit {
                    let found = units[start..]
                        .windows(sep_units.len())
                        .position(|window| window == sep_units.as_slice())
                        .map(|offset| start + offset);
                    let Some(position) = found else {
                        parts.push(Value::lstr(crate::jstr::from_units(&units[start..])));
                        break;
                    };
                    parts.push(Value::lstr(crate::jstr::from_units(
                        &units[start..position],
                    )));
                    start = position + sep_units.len();
                }
                parts
            };
            parts.truncate(limit);
            Ok(i.make_array(parts))
        }
    }
}

/// String.prototype[@@iterator]. Kept as a named native so iterator fast paths can prove that
/// the realm still exposes the exact intrinsic method after user code mutates prototypes.
fn nf_string_iterator(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    let obj = Object::new_with_parts(
        i.extra_protos.get("%StringIteratorPrototype%").cloned(),
        crate::value::Props::new(),
        Exotic::StringIterator(Box::new(crate::value::StringIteratorState {
            string: Some(s),
            index: 0,
        })),
    );
    Ok(Value::Obj(obj))
}

/// The primitive-string iterator fast path is valid only while both observable intrinsic
/// functions are still installed. A replacement getter/function on String.prototype must be
/// allowed to run through GetIterator.
pub(crate) fn intrinsic_string_iterator_is_unmodified(i: &Interp) -> bool {
    let Some(key) = i
        .iterator_sym
        .as_ref()
        .map(|symbol| Interp::sym_key(symbol.as_ref()))
    else {
        return false;
    };
    let method = i
        .string_proto
        .borrow()
        .props
        .get(&key)
        .filter(|property| !property.accessor())
        .map(|property| property.value());
    let next = i
        .extra_protos
        .get("%StringIteratorPrototype%")
        .and_then(|prototype| prototype.borrow().props.get("next").map(|p| p.value()));
    method.is_some_and(|method| {
        is_native_function(&method, nf_string_iterator)
            && next.is_some_and(|next| is_native_function(&next, string_iter_next))
    })
}

fn install_string(it: &mut Interp) {
    let sp = it.string_proto.clone();
    // String.prototype[@@iterator]: a lazy String Iterator over the receiver, by code point.
    if let Some(sym) = it.iterator_sym.clone() {
        let f = it.make_native("[Symbol.iterator]", 0, nf_string_iterator);
        sp.borrow_mut()
            .props
            .insert(Interp::sym_key(&sym), Property::builtin(Value::Obj(f)));
    }
    it.def_method(&sp, "toString", 0, |i, this, _| string_this_value(i, &this));
    it.def_method(&sp, "valueOf", 0, |i, this, _| string_this_value(i, &this));
    it.def_method(&sp, "charAt", 1, nf_char_at);
    it.def_method(&sp, "charCodeAt", 1, nf_char_code_at);
    it.def_method(&sp, "indexOf", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        let needle = ab(i.to_string(&arg(args, 0)))?;
        if let crate::interpreter::StrUnits::Ascii = i.units_of(&s) {
            // Byte index == unit index; a non-ASCII needle simply can't occur.
            let pos = str_clamp_pos(i, args.get(1), s.len() as i64)?.min(s.len());
            let r = s[pos..].find(&*needle).map(|k| (pos + k) as f64);
            return Ok(Value::Num(r.unwrap_or(-1.0)));
        }
        let chars = i.units_full(&s);
        let nchars = i.units_full(&needle);
        let len = chars.len() as i64;
        let pos = str_clamp_pos(i, args.get(1), len)?;
        let nlen = nchars.len();
        let result = (pos..=chars.len())
            .find(|&start| start + nlen <= chars.len() && chars[start..start + nlen] == nchars[..]);
        Ok(Value::Num(result.map(|r| r as f64).unwrap_or(-1.0)))
    });
    it.def_method(&sp, "lastIndexOf", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        let needle = ab(i.to_string(&arg(args, 0)))?;
        // StringLastIndexOf operates on UTF-16 code-unit indices (ECMA-262 §6.1.4.2 and
        // §22.1.3.11). For ASCII operands byte offsets are identical, so avoid materializing
        // two unit vectors and search the bounded byte prefix directly.
        if matches!(i.units_of(&s), crate::interpreter::StrUnits::Ascii)
            && matches!(i.units_of(&needle), crate::interpreter::StrUnits::Ascii)
        {
            // ECMA-262 §22.1.3.11 performs ToNumber(position) before the early
            // `maxStart < 0` return, so retain that observable coercion even when
            // the search string is longer than the receiver.
            let number_position = match arg(args, 1) {
                Value::Undefined => None,
                v => Some(ab(i.to_number(&v))?),
            };
            let Some(max_start) = s.len().checked_sub(needle.len()) else {
                return Ok(Value::Num(-1.0));
            };
            let start = match number_position {
                None => max_start,
                Some(p) if p.is_nan() => max_start,
                Some(p) => p.trunc().clamp(0.0, max_start as f64) as usize,
            };
            // `start <= max_start` proves this slice endpoint is in bounds.
            let end = start + needle.len();
            let result = s[..end].rfind(needle.as_str());
            return Ok(Value::Num(result.map(|n| n as f64).unwrap_or(-1.0)));
        }
        // Optional `position`: search for the last occurrence starting at or before it.
        let chars = i.units_full(&s);
        let nchars = i.units_full(&needle);
        let limit = match arg(args, 1) {
            Value::Undefined => chars.len(),
            v => {
                let p = ab(i.to_number(&v))?;
                if p.is_nan() {
                    chars.len()
                } else {
                    (p.max(0.0) as usize).min(chars.len())
                }
            }
        };
        let mut found = -1.0;
        let mut start = 0;
        while start + nchars.len() <= chars.len() && start <= limit {
            if chars[start..start + nchars.len()] == nchars[..] {
                found = start as f64;
            }
            start += 1;
        }
        Ok(Value::Num(found))
    });
    it.def_method(&sp, "toLocaleLowerCase", 0, |i, this, args| {
        let s = this_string(i, &this)?;
        let lang = locale_case_lang(i, &arg(args, 0))?;
        Ok(Value::lstr(locale_lower(&s, lang.as_deref())))
    });
    it.def_method(&sp, "toLocaleUpperCase", 0, |i, this, args| {
        let s = this_string(i, &this)?;
        let lang = locale_case_lang(i, &arg(args, 0))?;
        Ok(Value::lstr(locale_upper(&s, lang.as_deref())))
    });
    it.def_method(&sp, "normalize", 0, |i, this, args| {
        let s = this_string(i, &this)?;
        let form = match arg(args, 0) {
            Value::Undefined => "NFC".to_string(),
            v => ab(i.to_string(&v))?.to_string(),
        };
        if !matches!(form.as_str(), "NFC" | "NFD" | "NFKC" | "NFKD") {
            return Err(i.make_error(
                "RangeError",
                "The normalization form should be one of NFC, NFD, NFKC, NFKD",
            ));
        }
        let cps = crate::jstr::code_points(&s);
        let out = crate::unicode_norm_impl::normalize(&cps, &form);
        Ok(Value::lstr(crate::jstr::from_code_points(&out)))
    });
    it.def_method(&sp, "includes", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        if arg_is_regexp(i, &arg(args, 0))? {
            return Err(i.make_error("TypeError", "argument must not be a regular expression"));
        }
        let needle = ab(i.to_string(&arg(args, 0)))?;
        if matches!(i.units_of(&s), crate::interpreter::StrUnits::Ascii)
            && matches!(i.units_of(&needle), crate::interpreter::StrUnits::Ascii)
        {
            let pos = str_clamp_pos(i, args.get(1), s.len() as i64)?;
            return Ok(Value::Bool(s[pos..].contains(needle.as_str())));
        }
        let chars = i.units_full(&s);
        let len = chars.len() as i64;
        let pos = str_clamp_pos(i, args.get(1), len)?;
        let nchars = i.units_full(&needle);
        let found = (pos..=chars.len())
            .any(|k| k + nchars.len() <= chars.len() && chars[k..k + nchars.len()] == nchars[..]);
        Ok(Value::Bool(found))
    });
    it.def_method(&sp, "startsWith", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        if arg_is_regexp(i, &arg(args, 0))? {
            return Err(i.make_error("TypeError", "argument must not be a regular expression"));
        }
        let needle = ab(i.to_string(&arg(args, 0)))?;
        if matches!(i.units_of(&s), crate::interpreter::StrUnits::Ascii)
            && matches!(i.units_of(&needle), crate::interpreter::StrUnits::Ascii)
        {
            let pos = str_clamp_pos(i, args.get(1), s.len() as i64)?;
            return Ok(Value::Bool(s[pos..].starts_with(needle.as_str())));
        }
        let chars = i.units_full(&s);
        let len = chars.len() as i64;
        let pos = str_clamp_pos(i, args.get(1), len)?;
        let nchars = i.units_full(&needle);
        Ok(Value::Bool(
            pos + nchars.len() <= chars.len() && chars[pos..pos + nchars.len()] == nchars[..],
        ))
    });
    it.def_method(&sp, "endsWith", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        if arg_is_regexp(i, &arg(args, 0))? {
            return Err(i.make_error("TypeError", "argument must not be a regular expression"));
        }
        let needle = ab(i.to_string(&arg(args, 0)))?;
        if matches!(i.units_of(&s), crate::interpreter::StrUnits::Ascii)
            && matches!(i.units_of(&needle), crate::interpreter::StrUnits::Ascii)
        {
            let end = match args.get(1) {
                Some(v) if !matches!(v, Value::Undefined) => {
                    str_clamp_pos(i, Some(v), s.len() as i64)?
                }
                _ => s.len(),
            };
            return Ok(Value::Bool(
                end >= needle.len() && s[..end].ends_with(needle.as_str()),
            ));
        }
        let chars = i.units_full(&s);
        let len = chars.len() as i64;
        // endsWith's optional argument is the END position (default = length).
        let end = match args.get(1) {
            Some(v) if !matches!(v, Value::Undefined) => str_clamp_pos(i, Some(v), len)?,
            _ => len as usize,
        };
        let nchars = i.units_full(&needle);
        Ok(Value::Bool(
            end >= nchars.len() && chars[end - nchars.len()..end] == nchars[..],
        ))
    });
    it.def_method(&sp, "slice", 2, nf_string_slice);
    it.def_method(&sp, "substring", 2, |i, this, args| {
        let s = this_string(i, &this)?;
        if matches!(i.units_of(&s), crate::interpreter::StrUnits::Ascii) {
            // String.prototype.substring clamps and swaps UTF-16 indices (ECMA-262 §22.1.3.25).
            // ASCII byte offsets are the same code-unit indices, so keep the common path flat.
            let len = s.len() as i64;
            let mut a = (arg_to_number(i, args, 0)? as i64).clamp(0, len);
            let mut b = match args.get(1) {
                None | Some(Value::Undefined) => len,
                Some(Value::Num(n)) => (*n as i64).clamp(0, len),
                Some(v) => (ab(i.to_number(v))? as i64).clamp(0, len),
            };
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            return Ok(Value::Str(s.slice_bytes(a as usize, b as usize)));
        }
        let chars = i.units_full(&s);
        let len = chars.len() as i64;
        let mut a = (arg_to_number(i, args, 0)? as i64).clamp(0, len);
        let mut b = match args.get(1) {
            None | Some(Value::Undefined) => len,
            Some(Value::Num(n)) => (*n as i64).clamp(0, len),
            Some(v) => (ab(i.to_number(v))? as i64).clamp(0, len),
        };
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        Ok(Value::Str(
            i.slice_units(&s, &chars, a as usize, b as usize),
        ))
    });
    // Annex B B.2.3.1 String.prototype.substr(start, length).
    it.def_method(&sp, "substr", 2, |i, this, args| {
        let s = this_string(i, &this)?;
        if matches!(i.units_of(&s), crate::interpreter::StrUnits::Ascii) {
            let size = s.len() as i64;
            let n = arg_to_number(i, args, 0)?.trunc();
            let mut start = if n.is_nan() {
                0
            } else if n < 0.0 {
                (size + n as i64).max(0)
            } else {
                (n as i64).min(size)
            };
            // The whole-length default stays on `undefined`; a primitive Number needs no coercion
            // and everything else retains the full coercive path in its original position.
            let length = match args.get(1) {
                None | Some(Value::Undefined) => None,
                Some(Value::Num(l)) => Some(*l),
                Some(v) => Some(ab(i.to_number(v))?),
            };
            let len = match length {
                None => size,
                Some(l) if l.is_nan() => 0,
                Some(l) => (l as i64).max(0),
            };
            let count = len.min(size - start).max(0);
            if count <= 0 {
                return Ok(Value::lstr(""));
            }
            if start < 0 {
                start = 0;
            }
            return Ok(Value::Str(
                s.slice_bytes(start as usize, (start + count) as usize),
            ));
        }
        let chars = i.units_full(&s);
        let size = chars.len() as i64;
        // ToIntegerOrInfinity truncates first, so -0.5 is +0, not a from-the-end index.
        let n = arg_to_number(i, args, 0)?.trunc();
        let mut start = if n.is_nan() {
            0
        } else if n < 0.0 {
            (size + n as i64).max(0)
        } else {
            (n as i64).min(size)
        };
        let length = match args.get(1) {
            None | Some(Value::Undefined) => None,
            Some(Value::Num(l)) => Some(*l),
            Some(v) => Some(ab(i.to_number(v))?),
        };
        let len = match length {
            None => size,
            Some(l) if l.is_nan() => 0,
            Some(l) => (l as i64).max(0),
        };
        let count = len.min(size - start).max(0);
        if count <= 0 {
            return Ok(Value::lstr(String::new()));
        }
        if start < 0 {
            start = 0;
        }
        Ok(Value::Str(i.slice_units(
            &s,
            &chars,
            start as usize,
            (start + count) as usize,
        )))
    });
    // Annex B B.2.3 HTML-wrapper methods (CreateHTML): each wraps the string in a tag.
    it.def_method(&sp, "anchor", 1, |i, t, a| {
        create_html(i, t, "a", "name", &arg(a, 0))
    });
    it.def_method(&sp, "link", 1, |i, t, a| {
        create_html(i, t, "a", "href", &arg(a, 0))
    });
    it.def_method(&sp, "fontcolor", 1, |i, t, a| {
        create_html(i, t, "font", "color", &arg(a, 0))
    });
    it.def_method(&sp, "fontsize", 1, |i, t, a| {
        create_html(i, t, "font", "size", &arg(a, 0))
    });
    it.def_method(&sp, "big", 0, |i, t, _| {
        create_html(i, t, "big", "", &Value::Undefined)
    });
    it.def_method(&sp, "blink", 0, |i, t, _| {
        create_html(i, t, "blink", "", &Value::Undefined)
    });
    it.def_method(&sp, "bold", 0, |i, t, _| {
        create_html(i, t, "b", "", &Value::Undefined)
    });
    it.def_method(&sp, "fixed", 0, |i, t, _| {
        create_html(i, t, "tt", "", &Value::Undefined)
    });
    it.def_method(&sp, "italics", 0, |i, t, _| {
        create_html(i, t, "i", "", &Value::Undefined)
    });
    it.def_method(&sp, "small", 0, |i, t, _| {
        create_html(i, t, "small", "", &Value::Undefined)
    });
    it.def_method(&sp, "strike", 0, |i, t, _| {
        create_html(i, t, "strike", "", &Value::Undefined)
    });
    it.def_method(&sp, "sub", 0, |i, t, _| {
        create_html(i, t, "sub", "", &Value::Undefined)
    });
    it.def_method(&sp, "sup", 0, |i, t, _| {
        create_html(i, t, "sup", "", &Value::Undefined)
    });
    it.def_method(&sp, "toUpperCase", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        // ECMA-262 §22.1.3.30 applies Unicode Default Case Conversion. ASCII has no
        // one-to-many or context-sensitive mappings, so return unchanged strings by reference
        // and map changed strings directly into one LStr allocation.
        if s.ascii_hint() {
            if !s.bytes().any(|byte| byte.is_ascii_lowercase()) {
                return Ok(Value::Str(s));
            }
            return Ok(Value::Str(crate::lstr::LStr::map_ascii_case(&s, true)));
        }
        Ok(Value::lstr(s.to_uppercase()))
    });
    it.def_method(&sp, "toLowerCase", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        // ECMA-262 §22.1.3.28 uses the same Unicode mapping with lowercase output. The ASCII
        // fast path is exact and avoids both a Unicode iterator and a temporary Rust String.
        if s.ascii_hint() {
            if !s.bytes().any(|byte| byte.is_ascii_uppercase()) {
                return Ok(Value::Str(s));
            }
            return Ok(Value::Str(crate::lstr::LStr::map_ascii_case(&s, false)));
        }
        Ok(Value::lstr(s.to_lowercase()))
    });
    // lumen strings are valid UTF-8, so they're always well-formed.
    it.def_method(&sp, "isWellFormed", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        Ok(Value::Bool(!crate::jstr::has_lone_surrogate(&s)))
    });
    it.def_method(&sp, "toWellFormed", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        if !crate::jstr::has_lone_surrogate(&s) {
            return Ok(Value::Str(s));
        }
        let fixed: String = s
            .chars()
            .map(|c| {
                if crate::jstr::smuggled(c).is_some() {
                    '\u{FFFD}'
                } else {
                    c
                }
            })
            .collect();
        Ok(Value::lstr(fixed))
    });
    it.def_method(&sp, "trim", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        // TrimString removes the ECMA-262 WhiteSpace + LineTerminator set
        // (ECMA-262 §22.1.3.32.1). The ASCII subset includes VT (U+000B), which Rust's
        // trim_ascii excludes. Consult cached ASCII metadata rather than scanning the body,
        // and preserve the allocation when no code units are removed.
        if s.ascii_hint() {
            let trimmed = s.trim_matches(|c| matches!(c, '\u{9}'..='\u{d}' | ' '));
            return Ok(if trimmed.len() == s.len() {
                Value::Str(s)
            } else {
                Value::lstr(trimmed)
            });
        }
        let trimmed = s.trim_matches(is_js_ws);
        Ok(if trimmed.len() == s.len() {
            Value::Str(s)
        } else {
            Value::Str(crate::lstr::LStr::from(trimmed))
        })
    });
    it.def_method(&sp, "localeCompare", 1, |i, this, args| {
        // RequireObjectCoercible + ToString this, then delegate to Intl.Collator.
        let a = Value::Str(this_string(i, &this)?);
        let b = Value::Str(ab(i.to_string(&arg(args, 0)))?);
        intl_delegate(
            i,
            "Collator",
            arg(args, 1),
            arg(args, 2),
            "compare",
            &[a, b],
        )
    });
    it.def_method(&sp, "toLocaleString", 0, |i, this, _| {
        Ok(Value::Str(this_string(i, &this)?))
    });
    it.def_method(&sp, "concat", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        // ECMA-262 §22.1.3.5 returns the receiver string unchanged when there are no
        // arguments. String primitives have no observable identity, so avoid copying it.
        if args.is_empty() {
            return Ok(Value::Str(s));
        }
        // The one-argument form is common in generated code. Coerce the argument exactly once,
        // then allocate the result directly when no surrogate-boundary canonicalization is needed.
        if args.len() == 1 {
            let next = ab(i.to_string(&args[0]))?;
            s.len()
                .checked_add(next.len())
                .filter(|&len| len <= MAX_STR_LEN)
                .ok_or_else(|| i.make_error("RangeError", "Invalid string length"))?;
            return Ok(Value::Str(s.concat_owned(&next)));
        }
        // Preserve the spec's left-to-right ToString order and the existing per-step length
        // error timing.  If a surrogate boundary needs canonicalization, fall back immediately
        // to the old stepwise path; otherwise each intermediate byte length is just the sum and
        // all pieces can be copied once into one LStr allocation.
        let mut parts = Vec::with_capacity(args.len() + 1);
        parts.push(s);
        let mut total = parts[0].len();
        for (index, argument) in args.iter().enumerate() {
            let next = ab(i.to_string(argument))?;
            if parts
                .last()
                .is_some_and(|previous| crate::jstr::needs_join_fixup(previous, &next))
            {
                let mut out = parts[0].to_string();
                for part in parts.iter().skip(1) {
                    out.push_str(part);
                }
                out = crate::jstr::concat(&out, &next);
                if out.len() > MAX_STR_LEN {
                    return Err(i.make_error("RangeError", "Invalid string length"));
                }
                for remaining in args.iter().skip(index + 1) {
                    out = crate::jstr::concat(&out, &ab(i.to_string(remaining))?);
                    if out.len() > MAX_STR_LEN {
                        return Err(i.make_error("RangeError", "Invalid string length"));
                    }
                }
                return Ok(Value::lstr(out));
            }
            total = total
                .checked_add(next.len())
                .filter(|&len| len <= MAX_STR_LEN)
                .ok_or_else(|| i.make_error("RangeError", "Invalid string length"))?;
            parts.push(next);
        }
        Ok(Value::Str(crate::lstr::LStr::concat_many(&parts, total)))
    });
    it.def_method(&sp, "repeat", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        let n = arg_to_number(i, args, 0)?;
        if n < 0.0 || n.is_infinite() {
            return Err(i.make_error("RangeError", "invalid count value"));
        }
        let count = n as usize;
        if s.len().saturating_mul(count) > MAX_STR_LEN {
            return Err(i.make_error("RangeError", "Invalid string length"));
        }
        // The observable coercion and implementation length guard above follow ECMA-262
        // §22.1.3.18.  Canonical inputs whose repeated boundaries cannot join surrogate units can
        // be copied directly into the engine string, avoiding the temporary Rust `String` plus
        // conversion copy.  Inputs needing UTF-16 canonicalization retain the path below.
        if s.ascii_hint() {
            return Ok(Value::Str(s.repeat_ascii(count)));
        }
        if !crate::jstr::needs_repeat_fixup(&s, count) {
            return Ok(Value::Str(s.repeat_direct(count)));
        }
        let out = s.repeat(count);
        Ok(Value::lstr(crate::jstr::canonicalize(&out).unwrap_or(out)))
    });
    it.def_method(&sp, "split", 2, nf_string_split);
    it.def_method(&sp, "at", 1, |i, this, args| {
        let s = this_string(i, &this)?;
        // String.prototype.at performs ToAbsoluteIndex over UTF-16 code units
        // (ECMA-262 §22.1.3.1). Keep the representation classification from
        // `units_of` for both the length and selected unit so short non-ASCII
        // strings are not materialized twice on this hot indexed-read path.
        let units = i.units_of(&s);
        let len = match &units {
            crate::interpreter::StrUnits::Ascii => s.len(),
            crate::interpreter::StrUnits::Units(units) => units.len(),
        } as i64;
        let mut idx = arg_to_number(i, args, 0)? as i64;
        if idx < 0 {
            idx += len;
        }
        Ok(if idx < 0 || idx >= len {
            Value::Undefined
        } else {
            match &units {
                crate::interpreter::StrUnits::Ascii => s
                    .as_bytes()
                    .get(idx as usize)
                    .map(|&unit| Value::Str(crate::jstr::unit_lstr(unit as u16)))
                    .unwrap_or(Value::Undefined),
                crate::interpreter::StrUnits::Units(units) => units
                    .get(idx as usize)
                    .copied()
                    .map(crate::jstr::unit_lstr)
                    .map(Value::Str)
                    .unwrap_or(Value::Undefined),
            }
        })
    });
    it.def_method(&sp, "codePointAt", 1, nf_code_point_at);
    it.def_method(&sp, "trimStart", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        if s.ascii_hint() {
            let trimmed = s.trim_start_matches(|c| matches!(c, '\u{9}'..='\u{d}' | ' '));
            return Ok(if trimmed.len() == s.len() {
                Value::Str(s)
            } else {
                Value::lstr(trimmed)
            });
        }
        let trimmed = s.trim_start_matches(is_js_ws);
        Ok(if trimmed.len() == s.len() {
            Value::Str(s)
        } else {
            Value::Str(crate::lstr::LStr::from(trimmed))
        })
    });
    it.def_method(&sp, "trimEnd", 0, |i, this, _| {
        let s = this_string(i, &this)?;
        if s.ascii_hint() {
            let trimmed = s.trim_end_matches(|c| matches!(c, '\u{9}'..='\u{d}' | ' '));
            return Ok(if trimmed.len() == s.len() {
                Value::Str(s)
            } else {
                Value::lstr(trimmed)
            });
        }
        let trimmed = s.trim_end_matches(is_js_ws);
        Ok(if trimmed.len() == s.len() {
            Value::Str(s)
        } else {
            Value::Str(crate::lstr::LStr::from(trimmed))
        })
    });
    // Annex B aliases: trimLeft/trimRight ARE trimStart/trimEnd (same function objects).
    for (alias, target) in [("trimLeft", "trimStart"), ("trimRight", "trimEnd")] {
        let p = sp.borrow().props.get(target).cloned();
        if let Some(p) = p {
            sp.borrow_mut().props.insert(alias, p);
        }
    }
    it.def_method(&sp, "padStart", 1, |i, this, args| {
        string_pad(i, this, args, true)
    });
    it.def_method(&sp, "padEnd", 1, |i, this, args| {
        string_pad(i, this, args, false)
    });
    it.def_method(&sp, "match", 1, nf_string_match);
    it.def_method(&sp, "search", 1, |i, this, a| {
        // RequireObjectCoercible(this), then dispatch to an Object regexp's @@search.
        if matches!(this, Value::Undefined | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "String.prototype.search called on null or undefined",
            ));
        }
        let regexp = arg(a, 0);
        if matches!(regexp, Value::Obj(_)) {
            if let Some(key) = well_known_key(i, "search") {
                let searcher = ab(i.get_member(&regexp, &key))?;
                if !matches!(searcher, Value::Undefined | Value::Null) {
                    if !searcher.is_callable() {
                        return Err(i.make_error("TypeError", "@@search is not callable"));
                    }
                    return ab(i.call(searcher, regexp.clone(), std::slice::from_ref(&this)));
                }
            }
        }
        // Otherwise build a RegExp from the argument and invoke its @@search.
        let s = this_string(i, &this)?;
        let pattern = if matches!(regexp, Value::Undefined) {
            String::new()
        } else {
            ab(i.to_string(&regexp))?.to_string()
        };
        let rx = ab(i.make_regexp(&pattern, ""))?;
        let key = well_known_key(i, "search").unwrap();
        let searcher = ab(i.get_member(&rx, &key))?;
        ab(i.call(searcher, rx, &[Value::Str(s)]))
    });
    it.def_method(&sp, "matchAll", 1, |i, this, a| {
        // RequireObjectCoercible(this).
        if matches!(this, Value::Undefined | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "String.prototype.matchAll called on null or undefined",
            ));
        }
        let regexp = arg(a, 0);
        if !matches!(regexp, Value::Undefined | Value::Null) {
            // A RegExp argument must be global; then dispatch to its @@matchAll if present.
            if arg_is_regexp(i, &regexp)? {
                let flags = ab(i.get_member(&regexp, "flags"))?;
                if matches!(flags, Value::Undefined | Value::Null) {
                    return Err(i.make_error("TypeError", "regexp.flags is null or undefined"));
                }
                let fs = ab(i.to_string(&flags))?;
                if !fs.contains('g') {
                    return Err(i.make_error(
                        "TypeError",
                        "String.prototype.matchAll called with a non-global RegExp",
                    ));
                }
            }
            // Only an Object's @@matchAll is consulted; a primitive goes to the string path.
            if matches!(regexp, Value::Obj(_)) {
                if let Some(key) = well_known_key(i, "matchAll") {
                    let matcher = ab(i.get_member(&regexp, &key))?;
                    if !matches!(matcher, Value::Undefined | Value::Null) {
                        if !matcher.is_callable() {
                            return Err(i.make_error("TypeError", "@@matchAll is not callable"));
                        }
                        return ab(i.call(matcher, regexp.clone(), std::slice::from_ref(&this)));
                    }
                }
            }
        }
        // Otherwise build a global RegExp from the argument and invoke its @@matchAll.
        let s = this_string(i, &this)?;
        let pattern = if matches!(regexp, Value::Undefined) {
            String::new()
        } else {
            ab(i.to_string(&regexp))?.to_string()
        };
        let rx = ab(i.make_regexp(&pattern, "g"))?;
        let key = well_known_key(i, "matchAll").unwrap();
        let matcher = ab(i.get_member(&rx, &key))?;
        ab(i.call(matcher, rx, &[Value::Str(s)]))
    });
    it.def_method(&sp, "replace", 2, nf_string_replace);
    it.def_method(&sp, "replaceAll", 2, |i, this, args| {
        // RequireObjectCoercible(this) — but keep the raw receiver O for @@replace, which runs
        // BEFORE ToString(this).
        if matches!(this, Value::Undefined | Value::Null) {
            return Err(i.make_error(
                "TypeError",
                "String.prototype.replaceAll called on null or undefined",
            ));
        }
        let search = arg(args, 0);
        // Only an *Object* searchValue is inspected for regexp-ness / @@replace; a primitive's
        // Symbol.replace is never accessed.
        if matches!(search, Value::Obj(_)) {
            // A RegExp search value must be global: read `flags`, require it coercible, and check
            // for "g" — a non-global (or missing-flags) regexp is a TypeError.
            if arg_is_regexp(i, &search)? {
                let flags = ab(i.get_member(&search, "flags"))?;
                if matches!(flags, Value::Undefined | Value::Null) {
                    return Err(i.make_error("TypeError", "regexp flags is null or undefined"));
                }
                let flags_str = ab(i.to_string(&flags))?;
                if !flags_str.contains('g') {
                    return Err(i.make_error(
                        "TypeError",
                        "replaceAll must be called with a global RegExp",
                    ));
                }
            }
            // GetMethod(search, @@replace): if present, delegate with the raw receiver O.
            if let Some(key) = well_known_key(i, "replace") {
                let replacer = ab(i.get_member(&search, &key))?;
                if !matches!(replacer, Value::Undefined | Value::Null) {
                    if !replacer.is_callable() {
                        return Err(i.make_error("TypeError", "@@replace is not callable"));
                    }
                    return ab(i.call(replacer, search.clone(), &[this.clone(), arg(args, 1)]));
                }
            }
        }
        // Keep the engine string through matching so the no-match result can reuse it directly;
        // replacement coercion above still occurs before this search (ECMA-262 §22.1.3.20).
        let s = this_string(i, &this)?;
        let pat = ab(i.to_string(&arg(args, 0)))?;
        // ToString(replaceValue) happens exactly once, before any matching.
        let repl = prep_repl(i, &arg(args, 1))?;
        if pat.is_empty() {
            // An empty search matches at every UTF-16 code-unit position (ECMA-262
            // §22.1.3.20), not merely between Rust Unicode scalar values. Keep the existing byte
            // path for ASCII; astral and lone-surrogate inputs use the cached unit view so a pair
            // is split into its two specified code-unit results.
            if s.ascii_hint() {
                let mut out = String::new();
                let mut byte = 0usize;
                for ch in s.chars() {
                    out.push_str(&string_replacement(i, &repl, "", &s, byte)?);
                    out.push(ch);
                    byte += ch.len_utf8();
                }
                out.push_str(&string_replacement(i, &repl, "", &s, byte)?);
                return Ok(Value::lstr(out));
            }
            let units = i.units_full(&s);
            let needs_context = replacement_needs_context(&repl);
            let mut out = String::new();
            for pos in 0..=units.len() {
                let before = if needs_context {
                    crate::jstr::from_units(&units[..pos])
                } else {
                    String::new()
                };
                let after = if needs_context {
                    crate::jstr::from_units(&units[pos..])
                } else {
                    String::new()
                };
                out.push_str(&string_replacement_parts(
                    i, &repl, "", &s, pos, &before, &after,
                )?);
                if let Some(&unit) = units.get(pos) {
                    out.push_str(&crate::jstr::unit_str(unit));
                }
            }
            return Ok(Value::lstr(crate::jstr::canonicalize(&out).unwrap_or(out)));
        }
        let source_ascii = s.ascii_hint();
        let pattern_ascii = pat.ascii_hint();
        if source_ascii && pattern_ascii {
            let Some(first) = s.as_str().find(pat.as_ref()) else {
                return Ok(Value::Str(s));
            };
            if let Repl::Text(template) = &repl {
                if !template.as_bytes().contains(&b'$') {
                    return Ok(Value::lstr(replace_all_ascii_literal(
                        &s, &pat, template, first,
                    )));
                }
            }
            let mut out = String::with_capacity(s.len());
            out.push_str(&s[..first]);
            out.push_str(&string_replacement(i, &repl, pat.as_ref(), &s, first)?);
            let mut rest = &s[first + pat.len()..];
            let mut base = first + pat.len();
            while let Some(pos) = rest.find(pat.as_ref()) {
                out.push_str(&rest[..pos]);
                let rep = string_replacement(i, &repl, pat.as_ref(), &s, base + pos)?;
                out.push_str(&rep);
                rest = &rest[pos + pat.len()..];
                base += pos + pat.len();
            }
            out.push_str(rest);
            return Ok(Value::lstr(out));
        }
        // A pure-ASCII receiver cannot contain a non-ASCII search string. Avoid allocating a
        // UTF-16 view when the result is necessarily the original string.
        if source_ascii {
            return Ok(Value::Str(s));
        }
        let source_units = i.units_full(&s);
        let pattern_units = i.units_full(&pat);
        let Some(first) = source_units
            .windows(pattern_units.len())
            .position(|window| window == pattern_units.as_ref())
        else {
            return Ok(Value::Str(s));
        };
        let pattern_len = pattern_units.len();
        let needs_context = replacement_needs_context(&repl);
        let mut out = String::with_capacity(s.len());
        let mut cursor = 0usize;
        let mut pos = first;
        loop {
            out.push_str(&crate::jstr::from_units(&source_units[cursor..pos]));
            let matched = crate::jstr::from_units(&source_units[pos..pos + pattern_len]);
            let before = if needs_context {
                crate::jstr::from_units(&source_units[..pos])
            } else {
                String::new()
            };
            let after = if needs_context {
                crate::jstr::from_units(&source_units[pos + pattern_len..])
            } else {
                String::new()
            };
            out.push_str(&string_replacement_parts(
                i, &repl, &matched, &s, pos, &before, &after,
            )?);
            cursor = pos + pattern_len;
            let Some(next) = source_units[cursor..]
                .windows(pattern_len)
                .position(|window| window == pattern_units.as_ref())
            else {
                break;
            };
            pos = cursor + next;
        }
        out.push_str(&crate::jstr::from_units(&source_units[cursor..]));
        Ok(Value::lstr(crate::jstr::canonicalize(&out).unwrap_or(out)))
    });

    let ctor = it.make_native("String", 1, |i, _this, args| {
        let s = match args.first() {
            None => Value::lstr(""),
            // `String(sym)` stringifies a symbol to its descriptive string; `new String(sym)`
            // instead throws (via ToString below).
            Some(Value::Sym(s)) if !i.constructing => Value::lstr(format!(
                "Symbol({})",
                s.description.as_deref().unwrap_or("")
            )),
            Some(v) => Value::Str(ab(i.to_string(v))?),
        };
        maybe_box(i, s)
    });
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(sp.clone()), false, false, false),
    );
    sp.borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
    it.def_method(&ctor, "fromCharCode", 1, nf_from_char_code);
    it.def_method(&ctor, "raw", 1, |i, _this, args| {
        let template = arg(args, 0);
        let raw = ab(i.get_member(&template, "raw"))?;
        let raw_obj =
            this_obj(&raw).ok_or_else(|| i.make_error("TypeError", "raw is not an object"))?;
        let len = ab(i.to_length(&raw_obj))?;
        let mut out = String::new();
        for k in 0..len {
            let seg = ab(i.get_member(&raw, &k.to_string()))?;
            out.push_str(&ab(i.to_string(&seg))?);
            if k + 1 < len {
                if let Some(sub) = args.get(k + 1) {
                    out.push_str(&ab(i.to_string(sub))?);
                }
            }
        }
        Ok(Value::lstr(out))
    });
    it.def_method(&ctor, "fromCodePoint", 1, |i, _this, args| {
        let mut s = String::new();
        for (index, a) in args.iter().enumerate() {
            // ToNumber identity for Numbers; validation below is identical either way.
            let n = match a {
                Value::Num(n) => *n,
                v => ab(i.to_number(v))?,
            };
            // Each argument must be an integer code point in [0, 0x10FFFF].
            if !n.is_finite() || n.fract() != 0.0 || n < 0.0 || n > 0x10FFFF as f64 {
                // Diagnose the already-converted number without invoking author coercion
                // hooks a second time or changing the required RangeError completion.
                static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                static TRACED: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                if *TRACE.get_or_init(|| {
                    std::env::var_os("LUMEN_TRACE_CODE_POINT_ERRORS").is_some()
                }) && TRACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32
                {
                    eprintln!(
                        "lumen-code-point: argument_index={index} converted={n:?} argument_count={}",
                        args.len()
                    );
                }
                return Err(i.make_error("RangeError", "Invalid code point"));
            }
            let cp = n as u32;
            if (0xD800..0xE000).contains(&cp) {
                // A lone surrogate is a valid argument: smuggle it (see `jstr`).
                s.push(crate::jstr::smuggle(cp as u16));
            } else if cp >= crate::jstr::SMUGGLE_BASE {
                // A smuggle-range character is canonically its smuggled pair.
                let hi = 0xD800 + ((cp - 0x10000) >> 10);
                let lo = 0xDC00 + ((cp - 0x10000) & 0x3FF);
                s.push(crate::jstr::smuggle(hi as u16));
                s.push(crate::jstr::smuggle(lo as u16));
            } else {
                s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
            }
        }
        Ok(Value::lstr(s))
    });
    set_builtin(&it.global, "String", Value::Obj(ctor));
}

/// Named entry so the bytecode call cache can recognize and specialize ASCII string slicing.
pub(crate) fn nf_string_slice(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    // Index coercion is ToNumber (ECMA-262 §22.1.3.24); Numbers are already coerced (§7.1.4
    // identity), so bypass the generic dispatch for the common numeric-index case. All other inputs keep the full path with its abrupt
    // completions and valueOf/toString ordering. `undefined` end keeps its fast `len` default.
    if let crate::interpreter::StrUnits::Ascii = i.units_of(&s) {
        let len = s.len() as i64;
        let start_num = match arg(args, 0) {
            Value::Num(n) => n,
            v => ab(i.to_number(&v))?,
        };
        let start = norm_index(start_num, len);
        let end = match arg(args, 1) {
            Value::Undefined => len,
            Value::Num(n) => norm_index(n, len),
            v => norm_index(ab(i.to_number(&v))?, len),
        };
        return Ok(if start < end {
            Value::Str(s.slice_bytes(start as usize, end as usize))
        } else {
            Value::lstr("")
        });
    }
    let chars = i.units_full(&s);
    let len = chars.len() as i64;
    let start_num = match arg(args, 0) {
        Value::Num(n) => n,
        v => ab(i.to_number(&v))?,
    };
    let start = norm_index(start_num, len);
    let end = match arg(args, 1) {
        Value::Undefined => len,
        Value::Num(n) => norm_index(n, len),
        v => norm_index(ab(i.to_number(&v))?, len),
    };
    let out = if start < end {
        i.slice_units(&s, &chars, start as usize, end as usize)
    } else {
        crate::lstr::LStr::from("")
    };
    Ok(Value::Str(out))
}

/// Box a Number/String/Boolean primitive into a wrapper object (right prototype + exotic). Other
/// values pass through unchanged (Symbol/BigInt wrappers are not modeled yet).
fn box_primitive(i: &mut Interp, v: Value) -> Value {
    let (proto, exotic) = match &v {
        Value::Num(n) => (i.number_proto.clone(), Exotic::NumWrap(*n)),
        Value::Bool(b) => (i.boolean_proto.clone(), Exotic::BoolWrap(*b)),
        Value::Str(s) => (i.string_proto.clone(), Exotic::str_wrap(s.clone())),
        Value::Sym(s) => (i.symbol_proto.clone(), Exotic::SymWrap(s.clone())),
        Value::BigInt(n) => match i.extra_protos.get("BigInt").cloned() {
            Some(p) => (p, Exotic::bigint_wrap(n.clone())),
            None => return v,
        },
        _ => return v,
    };
    let obj = Object::new(Some(proto));
    // A String exotic object exposes each character as an own, enumerable, non-writable,
    // non-configurable index property, plus a non-enumerable `length`.
    if let Exotic::StrWrap(s) = &exotic {
        let units = crate::jstr::units(s);
        let mut b = obj.borrow_mut();
        for (idx, u) in units.iter().enumerate() {
            b.props.insert(
                idx.to_string().as_str(),
                Property::data(Value::Str(crate::jstr::unit_lstr(*u)), false, true, false),
            );
        }
        b.props.insert(
            "length",
            Property::data(Value::Num(units.len() as f64), false, false, false),
        );
    }
    obj.borrow_mut().exotic = exotic;
    Value::Obj(obj)
}

/// Box only when a wrapper constructor is invoked via `new` (`new Number(x)` boxes, `Number(x)` does
/// not). The instance prototype honors newTarget (OrdinaryCreateFromConstructor), falling back to
/// newTarget's *realm's* intrinsic when its `prototype` isn't an object.
fn maybe_box(i: &mut Interp, v: Value) -> Result<Value, Value> {
    if !i.constructing {
        return Ok(v);
    }
    let boxed = box_primitive(i, v);
    let nt = i.new_target.clone();
    if let (Value::Obj(o), Value::Obj(_)) = (&boxed, &nt) {
        let key = match o.borrow().exotic {
            Exotic::NumWrap(_) => "Number",
            Exotic::BoolWrap(_) => "Boolean",
            Exotic::StrWrap(_) => "String",
            _ => "",
        };
        if !key.is_empty() {
            match i.get_member(&nt, "prototype") {
                Ok(Value::Obj(p)) => o.borrow_mut().proto = Some(p),
                Ok(_) => {
                    if let Some(p) = ctor_realm_proto(i, &nt, key)? {
                        o.borrow_mut().proto = Some(p);
                    }
                }
                Err(e) => return Err(crate::interpreter::abrupt_value(e)),
            }
        }
    }
    Ok(boxed)
}

fn string_pad(i: &mut Interp, this: Value, args: &[Value], at_start: bool) -> Result<Value, Value> {
    let s = this_string(i, &this)?;
    let target = ab(i.to_number(&arg(args, 0)))? as usize;
    let cur = crate::jstr::unit_len(&s);
    if cur >= target {
        // ECMA-262 §22.1.3.17.1 returns the already-coerced string before it reads or coerces
        // fillString. Keep that string allocation instead of copying it into a Rust String.
        return Ok(Value::Str(s));
    }
    let pad = match arg(args, 1) {
        Value::Undefined => " ".to_string(),
        v => ab(i.to_string(&v))?.to_string(),
    };
    if pad.is_empty() {
        return Ok(Value::Str(s));
    }
    let s = s.as_str();
    let need = target - cur;
    if s.is_ascii() && pad.is_ascii() {
        // StringPad repeats and truncates by UTF-16 code units (ECMA-262
        // §22.1.3.17.2); ASCII bytes are those same units, so avoid building
        // a temporary fill string. Both operands are ASCII, so their UTF-16 concatenation needs
        // no canonicalization pass; write the fill and source into one final LStr allocation.
        return Ok(Value::Str(crate::lstr::LStr::pad_ascii(
            s, &pad, need, at_start,
        )));
    }
    let pad_units = crate::jstr::units(&pad);
    let fill_units: Vec<u16> = pad_units.iter().copied().cycle().take(need).collect();
    let fill = crate::jstr::from_units(&fill_units);
    Ok(Value::lstr(if at_start {
        format!("{fill}{s}")
    } else {
        format!("{s}{fill}")
    }))
}

/// thisStringValue: a String primitive or wrapper; anything else is a TypeError (the
/// String.prototype toString/valueOf methods are not generic).
fn string_this_value(i: &mut Interp, this: &Value) -> Result<Value, Value> {
    match this {
        Value::Str(s) => Ok(Value::Str(s.clone())),
        Value::Obj(o) => match &o.borrow().exotic {
            Exotic::StrWrap(s) => Ok(Value::Str((**s).clone())),
            _ => Err(i.make_error("TypeError", "not a String object")),
        },
        _ => Err(i.make_error("TypeError", "not a String object")),
    }
}

/// A prepared replacement: the non-callable form is ToString'd exactly once, up front.
enum Repl {
    Fn(Value),
    Text(Rc<str>),
}

fn prep_repl(i: &mut Interp, v: &Value) -> Result<Repl, Value> {
    if v.is_callable() {
        Ok(Repl::Fn(v.clone()))
    } else {
        Ok(Repl::Text(ab(i.to_string(v))?.into()))
    }
}

/// Whether a string replacement needs the potentially expensive preceding/following context.
/// `$&`, `$$`, and a callable replacement do not inspect those contexts, so UTF-16 empty-search
/// loops can avoid rebuilding a prefix and suffix for every insertion.
fn replacement_needs_context(repl: &Repl) -> bool {
    match repl {
        Repl::Fn(_) => false,
        Repl::Text(template) => template
            .as_bytes()
            .windows(2)
            .any(|pair| pair == b"$`" || pair == b"$'"),
    }
}

/// Fast literal replacement for an ASCII `replaceAll`. With no `$` marker, GetSubstitution is
/// exactly the replacement text, so avoid reparsing the template and constructing a temporary
/// `String` for every match (ECMA-262 §22.1.3.20.15.3–4).
fn replace_all_ascii_literal(
    source: &crate::lstr::LStr,
    pattern: &crate::lstr::LStr,
    replacement: &str,
    first: usize,
) -> String {
    let mut out = String::with_capacity(source.len());
    out.push_str(&source[..first]);
    out.push_str(replacement);
    let mut rest = &source[first + pattern.len()..];
    while let Some(pos) = rest.find(pattern.as_ref()) {
        out.push_str(&rest[..pos]);
        out.push_str(replacement);
        rest = &rest[pos + pattern.len()..];
    }
    out.push_str(rest);
    out
}

#[inline(always)]
fn string_replacement(
    i: &mut Interp,
    repl: &Repl,
    matched: &str,
    whole: &str,
    pos: usize,
) -> Result<String, Value> {
    let before = &whole[..pos.min(whole.len())];
    let after = whole.get(pos + matched.len()..).unwrap_or("");
    // Text replacements never consume the callback offset; avoid rescanning a long ASCII prefix
    // for every match. Callable replacements still receive the required UTF-16 position.
    let unit_pos = if matches!(repl, Repl::Fn(_)) {
        crate::jstr::unit_len(before)
    } else {
        0
    };
    string_replacement_parts(i, repl, matched, whole, unit_pos, before, after)
}

/// Replacement construction with explicitly separated UTF-16 context. Literal string search can
/// match one half of an astral code point, which has no UTF-8 byte boundary; callers on that path
/// rebuild `matched`/`before`/`after` from code units before entering this helper (ECMA-262
/// §22.1.3.19–20).
#[inline(always)]
fn string_replacement_parts(
    i: &mut Interp,
    repl: &Repl,
    matched: &str,
    whole: &str,
    unit_pos: usize,
    before: &str,
    after: &str,
) -> Result<String, Value> {
    if let Repl::Fn(f) = repl {
        // The callback's position argument counts UTF-16 code units, not bytes.
        let r = ab(i.call(
            f.clone(),
            Value::Undefined,
            &[
                Value::lstr(matched.to_string()),
                Value::Num(unit_pos as f64),
                Value::lstr(whole.to_string()),
            ],
        ))?;
        return Ok(ab(i.to_string(&r))?.to_string());
    }
    {
        let template = match repl {
            Repl::Text(t) => t.clone(),
            Repl::Fn(_) => unreachable!(),
        };
        // GetSubstitution for a string match: $$ → $, $& → match, $` → preceding, $' → following.
        // (No captures, so $n and $<name> stay literal.) Growth past the engine's string
        // ceiling dies as a RangeError, not an OOM.
        // Scan the template directly. The previous Vec<char> materialization made every
        // replacement pay an allocation even though GetSubstitution only needs one-character
        // lookahead; iterator peeking preserves the exact `$` handling without that temporary.
        let mut out = String::with_capacity(template.len());
        let mut chars = template.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '$' {
                let Some(&next) = chars.peek() else {
                    out.push('$');
                    continue;
                };
                match next {
                    '$' => {
                        out.push('$');
                        chars.next();
                        continue;
                    }
                    '&' => {
                        out.push_str(matched);
                        chars.next();
                        continue;
                    }
                    '`' => {
                        out.push_str(before);
                        chars.next();
                        continue;
                    }
                    '\'' => {
                        out.push_str(after);
                        chars.next();
                        continue;
                    }
                    _ => {}
                }
            }
            out.push(ch);
        }
        Ok(out)
    }
}

fn to_uint32(n: f64) -> u32 {
    if !n.is_finite() || n == 0.0 {
        return 0;
    }
    n.trunc().rem_euclid(4294967296.0) as u32
}

fn this_number(i: &mut Interp, this: &Value) -> Result<f64, Value> {
    // thisNumberValue: only a Number primitive or Number wrapper is acceptable.
    match this {
        Value::Num(n) => Ok(*n),
        Value::Obj(o) => match o.borrow().exotic {
            Exotic::NumWrap(n) => Ok(n),
            _ => Err(i.make_error("TypeError", "Number method called on incompatible receiver")),
        },
        _ => Err(i.make_error("TypeError", "Number method called on incompatible receiver")),
    }
}

/// Number.prototype.toExponential's digit selection: the exact decimal expansion of the binary
/// value, rounded to `f` fraction digits with ties away from zero (the spec's "larger n").
/// `shortest` (fractionDigits undefined) picks the minimal digits that round-trip.
fn to_exponential(x: f64, f: usize, shortest: bool) -> String {
    // The sign follows x < 0 exclusively; `+ 0.0` then clears a negative zero so it formats as "0".
    let (sign, x) = if x < 0.0 { ("-", -x) } else { ("", x + 0.0) };
    let fmt_exp = |exp: i32| format!("e{}{}", if exp < 0 { '-' } else { '+' }, exp.abs());
    if shortest {
        let s = format!("{x:e}");
        let (m, e) = s.split_once('e').unwrap();
        let exp: i32 = e.parse().unwrap();
        return format!("{sign}{m}{}", fmt_exp(exp));
    }
    if x == 0.0 {
        let m = if f == 0 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(f))
        };
        return format!("{sign}{m}e+0");
    }
    // 780 fraction digits is past the longest exact decimal expansion of any f64 mantissa, so the
    // digits (including everything after the rounding point) are exact.
    let s = format!("{x:.780e}");
    let (m, e) = s.split_once('e').unwrap();
    let mut exp: i32 = e.parse().unwrap();
    let mut digits: Vec<u8> = m
        .bytes()
        .filter(|b| b.is_ascii_digit())
        .map(|b| b - b'0')
        .collect();
    if digits.len() > f + 1 && digits[f + 1] >= 5 {
        let mut k = f + 1;
        loop {
            if k == 0 {
                digits.insert(0, 1);
                exp += 1;
                break;
            }
            k -= 1;
            if digits[k] == 9 {
                digits[k] = 0;
            } else {
                digits[k] += 1;
                break;
            }
        }
    }
    digits.truncate(f + 1);
    while digits.len() < f + 1 {
        digits.push(0);
    }
    let ds: String = digits.iter().map(|d| (d + b'0') as char).collect();
    let m = if f == 0 {
        ds
    } else {
        format!("{}.{}", &ds[..1], &ds[1..])
    };
    format!("{sign}{m}{}", fmt_exp(exp))
}
