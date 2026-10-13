//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;

pub(super) fn install_reflect(it: &mut Interp) {
    let r = it.new_object();
    it.def_method(&r, "get", 2, |i, _t, a| {
        let target = arg(a, 0);
        if !matches!(target, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.get called on non-object"));
        }
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        // Reflect.get(target, key, receiver) — the receiver is the third argument.
        let receiver = if a.len() > 2 {
            arg(a, 2)
        } else {
            target.clone()
        };
        reflect_ordinary_get(i, &target, &key, &receiver)
    });
    it.def_method(&r, "set", 3, |i, _t, a| {
        let target = arg(a, 0);
        if !matches!(target, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.set called on non-object"));
        }
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        let value = arg(a, 2);
        // A proxy's [[Set]] returns the ToBoolean of its trap result; surface that as Reflect.set's
        // own boolean rather than always reporting success.
        if let Some((ptarget, handler)) = proxy_pair(i, &target) {
            if matches!(handler, Value::Null) {
                return Err(i.make_error("TypeError", "proxy is revoked"));
            }
            let trap = ab(i.get_member(&handler, "set"))?;
            if trap.is_callable() {
                let receiver = if a.len() > 3 {
                    arg(a, 3)
                } else {
                    target.clone()
                };
                let res = ab(i.call(
                    trap,
                    handler,
                    &[
                        ptarget.clone(),
                        i.sym_from_key(&key)
                            .unwrap_or_else(|| Value::lstr(key.to_string())),
                        value.clone(),
                        receiver,
                    ],
                ))?;
                let ok = i.to_boolean(&res);
                if ok {
                    ab(i.proxy_set_invariant(&ptarget, &key, &value))?;
                }
                return Ok(Value::Bool(ok));
            }
            // Missing/undefined trap: forward to the target's [[Set]] with the original receiver,
            // returning its actual success boolean. [[Set]] reports failure as `false` (never a strict
            // PutValue throw), so evaluate it in a non-strict context.
            let receiver = if a.len() > 3 {
                arg(a, 3)
            } else {
                target.clone()
            };
            let saved = i.strict;
            i.strict = false;
            let r = i.set_member_recv(&ptarget, &key, value, receiver);
            i.strict = saved;
            return Ok(Value::Bool(ab(r)?));
        }
        let receiver = if a.len() > 3 {
            arg(a, 3)
        } else {
            target.clone()
        };
        Ok(Value::Bool(reflect_ordinary_set(
            i, &target, &key, value, &receiver,
        )?))
    });
    it.def_method(&r, "has", 2, |i, _t, a| {
        let target = arg(a, 0);
        if !matches!(target, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.has called on non-object"));
        }
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        // Trap-aware [[HasProperty]] so a proxy's `has` trap (and its throws) are honored.
        Ok(Value::Bool(ab(i.js_has_property(&target, &key))?))
    });
    it.def_method(&r, "getOwnPropertyDescriptor", 2, |i, _t, a| {
        // ECMA-262 §28.1.7: TypeError for a non-object target, then ToPropertyKey exactly once,
        // then target.[[GetOwnProperty]](key) and FromPropertyDescriptor.
        let Value::Obj(o) = arg(a, 0) else {
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
        if let Value::Obj(o) = arg(a, 0) {
            ab(i.defer_trigger(&o, Some(&key)))?;
            if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
                return Ok(Value::Bool(ab(i.proxy_delete(target, handler, &key))?));
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
        let o = match arg(a, 0) {
            Value::Obj(o) => o,
            _ => return Err(i.make_error("TypeError", "Reflect.ownKeys called on non-object")),
        };
        let keys = object_own_property_keys(i, &o)?;
        Ok(i.make_array(keys))
    });
    it.def_method(&r, "getPrototypeOf", 1, |i, _t, a| match arg(a, 0) {
        Value::Obj(o) => {
            if proxy_pair(i, &Value::Obj(o.clone())).is_some() {
                return js_get_prototype_of(i, &Value::Obj(o.clone()));
            }
            Ok(o.borrow()
                .proto
                .clone()
                .map(Value::Obj)
                .unwrap_or(Value::Null))
        }
        _ => Err(i.make_error("TypeError", "Reflect.getPrototypeOf called on non-object")),
    });
    it.def_method(&r, "setPrototypeOf", 2, |i, _t, a| {
        let obj = arg(a, 0);
        if !matches!(obj, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.setPrototypeOf called on non-object"));
        }
        let proto = arg(a, 1);
        if !matches!(proto, Value::Obj(_) | Value::Null) {
            return Err(i.make_error("TypeError", "prototype must be an object or null"));
        }
        Ok(Value::Bool(js_set_prototype_of(i, &obj, &proto)?))
    });
    it.def_method(&r, "defineProperty", 3, |i, _t, a| {
        let o = match arg(a, 0) {
            Value::Obj(o) => o,
            _ => return Err(i.make_error("TypeError", "Reflect.defineProperty on non-object")),
        };
        let key = ab(i.to_property_key(&arg(a, 1)))?;
        if let Some((target, handler)) = proxy_pair(i, &Value::Obj(o.clone())) {
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
        let new_target = if a.len() >= 3 {
            arg(a, 2)
        } else {
            target.clone()
        };
        ab(i.construct_nt(target, &args, new_target))
    });
    it.def_method(&r, "isExtensible", 1, |i, _t, a| {
        let obj = arg(a, 0);
        if !matches!(obj, Value::Obj(_)) {
            return Err(i.make_error("TypeError", "Reflect.isExtensible called on non-object"));
        }
        Ok(Value::Bool(js_is_extensible(i, &obj)?))
    });
    it.def_method(&r, "preventExtensions", 1, |i, _t, a| {
        let obj = arg(a, 0);
        if !matches!(obj, Value::Obj(_)) {
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

/// `Reflect.apply` (named so the JIT call cache can prove and tag its exact identity).
pub(crate) fn nf_reflect_apply(i: &mut Interp, _this: Value, a: &[Value]) -> Result<Value, Value> {
    // IsCallable(target) is checked before the argument list is read.
    if !arg(a, 0).is_callable() {
        return Err(i.make_error("TypeError", "Reflect.apply target is not callable"));
    }
    let args = create_list_from_array_like(i, &arg(a, 2))?;
    ab(i.call(arg(a, 0), arg(a, 1), &args))
}
