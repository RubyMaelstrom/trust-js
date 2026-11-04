//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

fn make_proxy(i: &mut Interp, target: Value, handler: Value) -> Result<Value, Value> {
    if !matches!((target).view(), ValueView::Obj(_))
        || !matches!((handler).view(), ValueView::Obj(_))
    {
        return Err(i.make_error(
            "TypeError",
            "Cannot create proxy with a non-object as target or handler",
        ));
    }
    let proto = match &(target).view() {
        ValueView::Obj(o) => o.borrow().proto,
        _ => None,
    };
    let obj = Object::new(proto);
    {
        let mut proxy = obj.borrow_mut();
        // A proxy has a [[Construct]] exactly when its target is a constructor (ProxyCreate
        // step 5), and a [[Call]] when the target is callable (its kind's).
        proxy.set_constructor(is_constructor_value(&target));
        // Its behavior is its kind's: the property caches stay off it.
        proxy.set_ic_plain(false);
        proxy.kind = ObjectKind::new_proxy(target, handler);
    }
    Ok(Value::Obj(obj))
}

fn revoke_proxy(_i: &mut Interp, _this: Value, a: &[Value]) -> Result<Value, Value> {
    if let ValueView::Obj(o) = (arg(a, 0)).view() {
        // Null the handler but keep the kind, so the object stays a (revoked) Proxy and every
        // operation throws a TypeError rather than silently acting like a plain object. The
        // mutable borrow records the write in the proxy's card.
        if let ObjectKind::Proxy(proxy) = &mut o.borrow_mut().kind {
            proxy.handler = Value::Null;
        }
    }
    Ok(Value::Undefined)
}

pub(super) fn install_proxy(it: &mut Interp) {
    let ctor = it.make_native("Proxy", 2, |i, _t, a| {
        if !i.constructing {
            return Err(i.make_error("TypeError", "Constructor Proxy requires 'new'"));
        }
        make_proxy(i, arg(a, 0), arg(a, 1))
    });
    ctor.borrow_mut().set_constructor(true); // Proxy is a constructor but has no `.prototype`
    it.def_method(&ctor, "revocable", 2, |i, _t, a| {
        let proxy = make_proxy(i, arg(a, 0), arg(a, 1))?;
        let revoke = make_bound(i, revoke_proxy, vec![proxy]);
        // The revocation function has own `length` (0) then `name` ("") like a built-in function.
        if let ValueView::Obj(ro) = &(revoke).view() {
            ro.borrow_mut().props.insert(
                "length",
                Property::data(Value::Num(0.0), false, false, true),
            );
            ro.borrow_mut()
                .props
                .insert("name", Property::data(Value::lstr(""), false, false, true));
        }
        let result = i.new_object();
        set_data(&result, "proxy", proxy);
        set_data(&result, "revoke", revoke);
        Ok(Value::Obj(result))
    });
    set_builtin(&it.global, "Proxy", Value::Obj(ctor));
}
