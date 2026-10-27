//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

pub(super) fn install_reflect(it: &mut Interp) {
    let r = it.new_object();
    it.def_method(&r, "get", 2, |i, _t, a| {
        let target = arg(a, 0);
        let ValueView::Obj(object) = target.view() else {
            return Err(i.make_error("TypeError", "Reflect.get called on non-object"));
        };
        // Reflect.get(target, key, receiver) — the receiver is the third argument.
        let receiver = if a.len() > 2 { arg(a, 2) } else { target };
        let key = arg(a, 1);
        match key.view() {
            // ToPropertyKey of a String is itself: an ordinary chain answers through the property
            // ICs, by the key's atom, as `object[key]` would.
            ValueView::Str(name) => {
                let text = crate::value::encode_lstr_key(&name);
                if let Some(result) = i.get_named_cached(&object, &text, &receiver) {
                    return ab(result);
                }
                reflect_ordinary_get(i, &target, &text, &receiver)
            }
            _ => {
                let text = property_key_text(i, &key)?;
                reflect_ordinary_get(i, &target, &text, &receiver)
            }
        }
    });
    it.def_method(&r, "set", 3, |i, _t, a| {
        let target = arg(a, 0);
        let ValueView::Obj(object) = target.view() else {
            return Err(i.make_error("TypeError", "Reflect.set called on non-object"));
        };
        let key = arg(a, 1);
        let value = arg(a, 2);
        let receiver = if a.len() > 3 { arg(a, 3) } else { target };
        // ToPropertyKey of a String is itself.
        if let ValueView::Str(name) = key.view() {
            let text = crate::value::encode_lstr_key(&name);
            return reflect_set(i, &object, &text, value, &receiver);
        }
        let text = property_key_text(i, &key)?;
        reflect_set(i, &object, &text, value, &receiver)
    });
    it.def_method(&r, "has", 2, |i, _t, a| {
        let target = arg(a, 0);
        if !matches!((target).view(), ValueView::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.has called on non-object"));
        }
        // HasProperty(target, ? ToPropertyKey(key)) is the `in` operator's algorithm once its
        // right operand is an Object, trap-aware for a Proxy on the chain.
        Ok(Value::Bool(ab(i.in_operator(&arg(a, 1), &target))?))
    });
    it.def_method(&r, "getOwnPropertyDescriptor", 2, |i, _t, a| {
        // ECMA-262 §28.1.7: TypeError for a non-object target, then ToPropertyKey exactly once,
        // then target.[[GetOwnProperty]](key) and FromPropertyDescriptor.
        let ValueView::Obj(o) = (arg(a, 0)).view() else {
            return Err(i.make_error(
                "TypeError",
                "Reflect.getOwnPropertyDescriptor on non-object",
            ));
        };
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        Ok(match object_get_own_property(i, &o, &key)? {
            Some(property) => descriptor_from_prop(i, property),
            None => Value::Undefined,
        })
    });
    it.def_method(&r, "deleteProperty", 2, |i, _t, a| {
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        if let ValueView::Obj(o) = (arg(a, 0)).view() {
            ab(i.defer_trigger(&o, Some(&key)))?;
            if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o)) {
                return Ok(Value::Bool(ab(i.proxy_delete(target, handler, &key))?));
            }
            if matches!(o.borrow().exotic, Exotic::Arguments) {
                // Arguments [[Delete]] severs a parameter mapping only after a
                // successful ordinary deletion. Share the delete operator's path;
                // a proxy trap that merely reports success never reaches this branch.
                return ab(i.delete_prop_with(Value::Obj(o), &key, false));
            }
            // A TypedArray integer index can't be deleted; a canonical-numeric non-index reports true.
            if let Some(info) = ta_info(i, &o) {
                match i.ta_index_kind(&info, &key) {
                    crate::value::TaIndex::Element(_) => return Ok(Value::Bool(false)),
                    crate::value::TaIndex::Exotic => return Ok(Value::Bool(true)),
                    crate::value::TaIndex::Ordinary => {}
                }
            }
            if i.host_indexed_array_key(&o, &key) {
                let supported = crate::value::canonical_index(&key)
                    .is_some_and(|index| index < i.host_indexed_len(&o).unwrap_or(0));
                return Ok(Value::Bool(!supported));
            }
            if ab(i.host_named_visible(&o, &key))? {
                return Ok(Value::Bool(false));
            }
            let configurable = o
                .borrow()
                .props
                .get(&key)
                .map(|p| p.configurable())
                .unwrap_or(true);
            if configurable {
                o.borrow_mut().props.remove(&key);
                return Ok(Value::Bool(true));
            }
            return Ok(Value::Bool(false));
        }
        Err(i.make_error("TypeError", "Reflect.deleteProperty called on non-object"))
    });
    it.def_method(&r, "ownKeys", 1, |i, _t, a| {
        let o = match (arg(a, 0)).view() {
            ValueView::Obj(o) => o,
            _ => return Err(i.make_error("TypeError", "Reflect.ownKeys called on non-object")),
        };
        let keys = object_own_property_keys(i, &o)?;
        Ok(i.make_array(keys.into_vec()))
    });
    it.def_method(&r, "getPrototypeOf", 1, |i, _t, a| {
        match (arg(a, 0)).view() {
            ValueView::Obj(o) => {
                if proxy_pair(i, &Value::Obj(o)).is_some() {
                    return js_get_prototype_of(i, &Value::Obj(o));
                }
                Ok(o.borrow().proto.map(Value::Obj).unwrap_or(Value::Null))
            }
            _ => Err(i.make_error("TypeError", "Reflect.getPrototypeOf called on non-object")),
        }
    });
    it.def_method(&r, "setPrototypeOf", 2, |i, _t, a| {
        let obj = arg(a, 0);
        if !matches!((obj).view(), ValueView::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.setPrototypeOf called on non-object"));
        }
        let proto = arg(a, 1);
        if !matches!((proto).view(), ValueView::Obj(_) | ValueView::Null) {
            return Err(i.make_error("TypeError", "prototype must be an object or null"));
        }
        Ok(Value::Bool(js_set_prototype_of(i, &obj, &proto)?))
    });
    it.def_method(&r, "defineProperty", 3, |i, _t, a| {
        let o = match (arg(a, 0)).view() {
            ValueView::Obj(o) => o,
            _ => return Err(i.make_error("TypeError", "Reflect.defineProperty on non-object")),
        };
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o)) {
            return Ok(Value::Bool(ab(proxy_define_property(
                i,
                &target,
                &handler,
                &key,
                &arg(a, 2),
            ))?));
        }
        let ok = ab(define_own_property(i, &o, &key, &arg(a, 2)))?;
        Ok(Value::Bool(ok))
    });
    it.def_method(&r, "apply", 3, nf_reflect_apply);
    it.def_method(&r, "construct", 2, |i, _t, a| {
        let target = arg(a, 0);
        if !is_constructor_value(&target) {
            return Err(i.make_error("TypeError", "Reflect.construct target is not a constructor"));
        }
        // The optional newTarget (3rd arg) must also be a constructor.
        if a.len() >= 3 && !is_constructor_value(&arg(a, 2)) {
            return Err(i.make_error(
                "TypeError",
                "Reflect.construct newTarget is not a constructor",
            ));
        }
        let args = create_list_from_array_like(i, &arg(a, 1))?;
        let new_target = if a.len() >= 3 { arg(a, 2) } else { target };
        ab(i.construct_nt(target, &args, new_target))
    });
    it.def_method(&r, "isExtensible", 1, |i, _t, a| {
        let obj = arg(a, 0);
        if !matches!((obj).view(), ValueView::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.isExtensible called on non-object"));
        }
        Ok(Value::Bool(js_is_extensible(i, &obj)?))
    });
    it.def_method(&r, "preventExtensions", 1, |i, _t, a| {
        let obj = arg(a, 0);
        if !matches!((obj).view(), ValueView::Obj(_)) {
            return Err(i.make_error(
                "TypeError",
                "Reflect.preventExtensions called on non-object",
            ));
        }
        Ok(Value::Bool(js_prevent_extensions(i, &obj)?))
    });
    set_to_string_tag(it, &r, "Reflect");
    set_builtin(&it.global, "Reflect", Value::Obj(r));
}

/// ToPropertyKey(key) (ECMA-262 §7.1.19) as engine key text, for a key other than a String
/// (whose text is its own): a Symbol's encoded key and an integral Number's decimal text are
/// written without allocating, as neither conversion runs code; any other key is converted.
/// The key value keeps a Symbol alive while its text is used.
enum PropertyKeyText {
    Written(crate::value::KeyText),
    Converted(crate::value::PropertyKey),
}

impl std::ops::Deref for PropertyKeyText {
    type Target = str;

    fn deref(&self) -> &str {
        match self {
            PropertyKeyText::Written(text) => text.as_str(),
            PropertyKeyText::Converted(key) => key,
        }
    }
}

fn property_key_text(i: &mut Interp, key: &Value) -> Result<PropertyKeyText, Value> {
    let written = match key.view() {
        ValueView::Sym(symbol) => Some(crate::value::KeyText::symbol(symbol.id)),
        ValueView::Num(n) => crate::value::KeyText::integer(n),
        _ => None,
    };
    match written {
        Some(text) => Ok(PropertyKeyText::Written(text)),
        None => Ok(PropertyKeyText::Converted(ab(i.to_property_key(key))?)),
    }
}

/// Reflect.set's step 4 (ECMA-262 §28.1.13): `object`.[[Set]](key, value, receiver) for the key
/// text `key`, as its success.
fn reflect_set(
    i: &mut Interp,
    object: &Gc,
    key: &str,
    value: Value,
    receiver: &Value,
) -> Result<Value, Value> {
    let receiver_is_target = matches!(receiver.view(), ValueView::Obj(r) if Gc::ptr_eq(&r, object));
    if receiver_is_target && set_own_data_property(object, key, value) {
        return Ok(Value::Bool(true));
    }
    let target = Value::Obj(*object);
    // A proxy's [[Set]] returns the ToBoolean of its trap result; surface that as Reflect.set's
    // own boolean rather than always reporting success.
    if let Some((ptarget, handler)) = proxy_pair(i, &target) {
        if matches!((handler).view(), ValueView::Null) {
            return Err(i.make_error("TypeError", "proxy is revoked"));
        }
        if let Some(trap) =
            ab(i.proxy_get_trap(&handler, crate::interpreter::PROXY_TRAP_SET, "set"))?
        {
            let key_value = i
                .sym_from_key(key)
                .unwrap_or_else(|| crate::value::string_key_value(key));
            let res = ab(i.proxy_call_trap(
                crate::interpreter::PROXY_TRAP_SET,
                &trap,
                handler,
                [ptarget, key_value, value, *receiver],
            ))?;
            let ok = i.to_boolean(&res);
            if ok {
                ab(i.proxy_set_invariant(&ptarget, key, &value))?;
            }
            return Ok(Value::Bool(ok));
        }
        // Missing/undefined trap: forward to the target's [[Set]] with the original receiver,
        // returning its actual success boolean. [[Set]] reports failure as `false` (never a strict
        // PutValue throw), so evaluate it in a non-strict context.
        let saved = i.strict;
        i.strict = false;
        let r = i.set_member_recv(&ptarget, key, value, *receiver);
        i.strict = saved;
        return Ok(Value::Bool(ab(r)?));
    }
    Ok(Value::Bool(reflect_ordinary_set(
        i, &target, key, value, receiver,
    )?))
}

/// OrdinarySet(object, P, V, object) (ECMA-262 §10.1.9.2 OrdinarySetWithOwnDescriptor) where
/// P, whose text is `key`, is an own writable data property of an ordinary object: the
/// receiver's own descriptor is that property, so its [[DefineOwnProperty]] replaces only the
/// [[Value]], with no code to run. `false` declines, having changed nothing: an exotic or
/// side-table object, or an absent, inherited, read-only or accessor property.
fn set_own_data_property(object: &Gc, key: &str, value: Value) -> bool {
    // Text with no atom names no property.
    let Some(key) = crate::atom::PropKey::lookup_text(key) else {
        return false;
    };
    let slot = {
        let body = object.borrow();
        if !body.ic_plain.get() || !body.exotic.is_ordinary() {
            return false;
        }
        let Some(slot) = body.props.slot_of_key(key) else {
            return false;
        };
        if body
            .props
            .property_at(slot)
            .is_none_or(|property| property.accessor() || !property.writable())
        {
            return false;
        }
        slot
    };
    object.borrow_mut().props.set_value_at(slot, value);
    true
}

/// `Reflect.apply` (named so the JIT call cache can prove and tag its exact identity).
pub(crate) fn nf_reflect_apply(i: &mut Interp, _this: Value, a: &[Value]) -> Result<Value, Value> {
    // IsCallable(target) is checked before the argument list is read.
    if !arg(a, 0).is_callable() {
        return Err(i.make_error("TypeError", "Reflect.apply target is not callable"));
    }
    let args = create_list_from_array_like(i, &arg(a, 2))?;
    ab(i.call(arg(a, 0), arg(a, 1), &args))
}
