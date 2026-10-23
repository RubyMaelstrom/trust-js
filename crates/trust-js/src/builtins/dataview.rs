//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

thread_local! {
    /// Exact native identities; the call cache validates their Realm and live callee object.
    static DATAVIEW_INTRINSICS: std::cell::RefCell<crate::fasthash::FastMap<usize, u8>> =
        std::cell::RefCell::new(Default::default());
}

pub(crate) fn dataview_intrinsic(f: NativeFn) -> u8 {
    DATAVIEW_INTRINSICS.with(|map| {
        map.borrow()
            .get(&(f as *const () as usize))
            .copied()
            .unwrap_or(0)
    })
}

fn dv_method(it: &mut Interp, proto: &Gc, name: &str, len: usize, f: NativeFn, setter: bool) {
    DATAVIEW_INTRINSICS.with(|map| {
        map.borrow_mut().insert(
            f as *const () as usize,
            if setter {
                crate::bytecode::INTRINSIC_DATAVIEW_SET
            } else {
                crate::bytecode::INTRINSIC_DATAVIEW_GET
            },
        );
    });
    it.def_method(proto, name, len, f);
}

fn dv_info(i: &mut Interp, this: &Value) -> Result<(usize, usize, usize, bool), Value> {
    let ptr =
        map_ptr(this).ok_or_else(|| i.make_error("TypeError", "receiver is not a DataView"))?;
    i.data_views
        .get(&ptr)
        .copied()
        .ok_or_else(|| i.make_error("TypeError", "receiver is not a DataView"))
}

fn dv_len_for_buffer(blen: usize, off: usize, len: usize, track: bool) -> Option<usize> {
    let available = blen.checked_sub(off)?;
    if track {
        Some(available)
    } else {
        (len <= available).then_some(len)
    }
}

/// GetViewByteLength, including detachment and a fixed view becoming out of bounds.
fn dv_view_len(i: &Interp, buf: usize, off: usize, len: usize, track: bool) -> Option<usize> {
    dv_len_for_buffer(i.array_buffer_byte_length(buf)?, off, len, track)
}

enum DvAccessError {
    View,
    Offset,
}

impl DvAccessError {
    fn value(self, i: &mut Interp) -> Value {
        match self {
            Self::View => i.make_error(
                "TypeError",
                "DataView's buffer is detached or out of bounds",
            ),
            Self::Offset => {
                i.make_error("RangeError", "Offset is outside the bounds of the DataView")
            }
        }
    }
}

/// Make the buffer witness and access its bytes under one borrow/lock. Arguments have already
/// been coerced, so no author code or collection can run while this witness is in use.
fn dv_range(
    blen: usize,
    off: usize,
    len: usize,
    track: bool,
    byte_off: usize,
    size: usize,
) -> Result<std::ops::Range<usize>, DvAccessError> {
    let vlen = dv_len_for_buffer(blen, off, len, track).ok_or(DvAccessError::View)?;
    if byte_off > vlen || size > vlen - byte_off {
        return Err(DvAccessError::Offset);
    }
    let start = off + byte_off;
    Ok(start..start + size)
}

fn dv_bytes(
    i: &Interp,
    (buf, off, len, track): (usize, usize, usize, bool),
    byte_off: usize,
    size: usize,
) -> Result<[u8; 8], DvAccessError> {
    i.buffer_data(buf)
        .ok_or(DvAccessError::View)?
        .with_bytes(|source| {
            let range = dv_range(source.len(), off, len, track, byte_off, size)?;
            let mut bytes = [0; 8];
            bytes[..size].copy_from_slice(&source[range]);
            Ok(bytes)
        })
        .ok_or(DvAccessError::View)?
}

fn dv_put(
    i: &mut Interp,
    (buf, off, len, track): (usize, usize, usize, bool),
    byte_off: usize,
    bytes: &[u8],
) -> Result<(), DvAccessError> {
    let data = i.buffer_data(buf).ok_or(DvAccessError::View)?;
    let local = matches!(data, BufferData::Local(_));
    let start = data
        .with_bytes_mut(|storage| {
            let range = dv_range(storage.len(), off, len, track, byte_off, bytes.len())?;
            let start = range.start;
            storage[range].copy_from_slice(bytes);
            Ok(start)
        })
        .ok_or(DvAccessError::View)??;
    if local {
        i.mark_array_buffer_dirty_range(buf, start, bytes.len());
    }
    Ok(())
}

/// ToIndex's common Number/undefined inputs avoid cloning and general coercion dispatch.
fn dv_index(i: &mut Interp, args: &[Value]) -> Result<usize, Value> {
    let value = arg(args, 0);
    match value.view() {
        ValueView::Undefined => Ok(0),
        ValueView::Num(n) => {
            let n = if n.is_nan() { 0.0 } else { n.trunc() };
            if !(0.0..=9007199254740991.0).contains(&n) {
                return Err(i.make_error("RangeError", "index out of range"));
            }
            Ok(n as usize)
        }
        _ => to_index(i, &value),
    }
}
fn dv_buffer_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    dv_info(i, &this)?;
    ab(i.get_internal_slot(&this, "#\u{0}dv_buffer"))
}
fn dv_bytelength_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let (buf, off, len, track) = dv_info(i, &this)?;
    match dv_view_len(i, buf, off, len, track) {
        Some(l) => Ok(Value::Num(l as f64)),
        None => Err(i.make_error(
            "TypeError",
            "DataView's buffer is detached or out of bounds",
        )),
    }
}
fn dv_byteoffset_get(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let (buf, off, len, track) = dv_info(i, &this)?;
    if dv_view_len(i, buf, off, len, track).is_none() {
        return Err(i.make_error(
            "TypeError",
            "DataView's buffer is detached or out of bounds",
        ));
    }
    Ok(Value::Num(off as f64))
}

#[inline]
fn dv_get(i: &mut Interp, this: &Value, args: &[Value], kind: TaKind) -> Result<Value, Value> {
    let ptr = map_ptr(this).ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    let (buf, off, len, track) = *i
        .data_views
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    let byte_off = dv_index(i, args)?;
    let little = i.to_boolean(args.get(1).unwrap_or(&Value::Undefined));
    let es = kind.elsize();
    let mut b =
        dv_bytes(i, (buf, off, len, track), byte_off, es).map_err(|error| error.value(i))?;
    // `kind.read` is little-endian; DataView defaults to big-endian, so reverse unless little.
    if !little {
        b[..es].reverse();
    }
    Ok(Value::Num(kind.read(&b[..es])))
}

#[inline]
fn dv_set(i: &mut Interp, this: &Value, args: &[Value], kind: TaKind) -> Result<Value, Value> {
    let ptr = map_ptr(this).ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    let (buf, off, len, track) = *i
        .data_views
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    // IsImmutableBuffer is checked before ToIndex(byteOffset)/ToNumber(value) read any arguments.
    if i.immutable_buffers.contains(&buf) {
        return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
    }
    let byte_off = dv_index(i, args)?;
    let value = arg_to_number(i, args, 1)?;
    let little = i.to_boolean(args.get(2).unwrap_or(&Value::Undefined));
    // Coercing the index/value can detach or resize the buffer. The byte access below makes a
    // new witness only after both conversions and retains it through the write.
    let es = kind.elsize();
    let mut bytes = [0; 8];
    let bytes = &mut bytes[..es];
    kind.write_into(value, bytes);
    if !little {
        bytes.reverse();
    }
    dv_put(i, (buf, off, len, track), byte_off, bytes).map_err(|error| error.value(i))?;
    Ok(Value::Undefined)
}

fn dv_get_big(i: &mut Interp, this: &Value, args: &[Value], signed: bool) -> Result<Value, Value> {
    let ptr = map_ptr(this).ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    let (buf, off, len, track) = *i
        .data_views
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    let byte_off = dv_index(i, args)?;
    let little = i.to_boolean(args.get(1).unwrap_or(&Value::Undefined));
    let mut b = dv_bytes(i, (buf, off, len, track), byte_off, 8).map_err(|error| error.value(i))?;
    if !little {
        b.reverse();
    }
    let raw = u64::from_le_bytes(b);
    Ok(Value::BigInt(if signed {
        crate::bigint::JsBigInt::from_i128(raw as i64 as i128)
    } else {
        crate::bigint::JsBigInt::from_u64(raw)
    }))
}

fn dv_set_big(i: &mut Interp, this: &Value, args: &[Value]) -> Result<Value, Value> {
    let ptr = map_ptr(this).ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    let (buf, off, len, track) = *i
        .data_views
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "not a DataView"))?;
    if i.immutable_buffers.contains(&buf) {
        return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
    }
    let byte_off = dv_index(i, args)?;
    let value = ab(i.to_bigint(&arg(args, 1)))?;
    let little = i.to_boolean(args.get(2).unwrap_or(&Value::Undefined));
    let mut bytes = (value.to_i128_wrapping() as u64).to_le_bytes();
    if !little {
        bytes.reverse();
    }
    dv_put(i, (buf, off, len, track), byte_off, &bytes).map_err(|error| error.value(i))?;
    Ok(Value::Undefined)
}

pub(super) fn install_dataview(it: &mut Interp) {
    let proto = Object::new(Some(it.object_proto));
    Rc::make_mut(&mut it.extra_protos).insert("DataView", proto);
    // buffer / byteLength / byteOffset are brand-checked accessor getters; byteLength/byteOffset
    // additionally throw if the backing buffer has been detached.
    for (name, getter) in [
        (
            "buffer",
            dv_buffer_get as fn(&mut Interp, Value, &[Value]) -> Result<Value, Value>,
        ),
        ("byteLength", dv_bytelength_get),
        ("byteOffset", dv_byteoffset_get),
    ] {
        let g = it.make_native(&format!("get {name}"), 0, getter);
        proto.borrow_mut().props.insert(
            name,
            Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
        );
    }
    // DataView.prototype[@@toStringTag] = "DataView" (non-writable, non-enumerable, configurable).
    set_to_string_tag(it, &proto, "DataView");
    macro_rules! dvm {
        ($getname:expr, $setname:expr, $kind:expr) => {
            dv_method(
                it,
                &proto,
                $getname,
                1,
                |i, this, a| dv_get(i, &this, a, $kind),
                false,
            );
            dv_method(
                it,
                &proto,
                $setname,
                2,
                |i, this, a| dv_set(i, &this, a, $kind),
                true,
            );
        };
    }
    dvm!("getInt8", "setInt8", TaKind::I8);
    dvm!("getUint8", "setUint8", TaKind::U8);
    dvm!("getInt16", "setInt16", TaKind::I16);
    dvm!("getUint16", "setUint16", TaKind::U16);
    dvm!("getInt32", "setInt32", TaKind::I32);
    dvm!("getUint32", "setUint32", TaKind::U32);
    dvm!("getFloat16", "setFloat16", TaKind::F16);
    dvm!("getFloat32", "setFloat32", TaKind::F32);
    dvm!("getFloat64", "setFloat64", TaKind::F64);
    dv_method(
        it,
        &proto,
        "getBigInt64",
        1,
        |i, this, a| dv_get_big(i, &this, a, true),
        false,
    );
    dv_method(
        it,
        &proto,
        "getBigUint64",
        1,
        |i, this, a| dv_get_big(i, &this, a, false),
        false,
    );
    dv_method(
        it,
        &proto,
        "setBigInt64",
        2,
        |i, this, a| dv_set_big(i, &this, a),
        true,
    );
    dv_method(
        it,
        &proto,
        "setBigUint64",
        2,
        |i, this, a| dv_set_big(i, &this, a),
        true,
    );

    let ctor = it.make_native("DataView", 1, |i, _t, a| {
        if !i.constructing {
            return Err(i.make_error("TypeError", "DataView constructor requires 'new'"));
        }
        // An ArrayBuffer object is identified by its [[ArrayBufferData]] slot (the internal
        // `__abMaxByteLength` marker), which survives detachment — a detached buffer is still an
        // ArrayBuffer, so ToNumber(byteOffset) must run before the detached check throws.
        let (bv, bp) = match (arg(a, 0)).view() {
            ValueView::Obj(o)
                if o.borrow()
                    .internal_slots()
                    .contains("#\u{0}abMaxByteLength") =>
            {
                (Value::Obj(o), Gc::as_ptr(&o) as usize)
            }
            _ => return Err(i.make_error("TypeError", "DataView requires an ArrayBuffer")),
        };
        // ToIndex(byteOffset) may run user code that detaches/resizes the buffer.
        let offset = {
            let __value_match_original = arg(a, 1);
            match (__value_match_original).view() {
                ValueView::Undefined => 0,
                _ => {
                    let v = __value_match_original;
                    to_index(i, &v)?
                }
            }
        };
        let has_len = !matches!((arg(a, 2)).view(), ValueView::Undefined);
        let len_arg = if has_len {
            Some(to_index(i, &arg(a, 2))?)
        } else {
            None
        };
        // Re-read the (possibly mutated) buffer state after all coercions.
        let Some(buflen) = i.array_buffer_byte_length(bp) else {
            return Err(i.make_error("TypeError", "ArrayBuffer is detached"));
        };
        if offset > buflen {
            return Err(i.make_error("RangeError", "DataView byteOffset is out of bounds"));
        }
        // IsFixedLengthArrayBuffer: a resizable ArrayBuffer or growable SharedArrayBuffer has an
        // [[ArrayBufferMaxByteLength]] slot. Read the slot, not the author-visible accessors.
        let resizable = matches!(
            (bv.as_obj().and_then(|o| o
                .borrow()
                .internal_slots()
                .get("#\u{0}abResizable")
                .map(|p| p.value())))
            .map(|__value| (__value).view()),
            Some(ValueView::Bool(true))
        );
        if let Some(l) = len_arg {
            if l > buflen - offset {
                return Err(i.make_error("RangeError", "DataView byteLength is out of bounds"));
            }
        }
        // OrdinaryCreateFromConstructor does Get(newTarget, "prototype"), which can run a custom
        // proto getter that detaches or resizes the buffer — re-validate everything afterwards.
        let obj = new_from_ctor(i, "DataView")?;
        let Some(buflen) = i.array_buffer_byte_length(bp) else {
            return Err(i.make_error("TypeError", "ArrayBuffer is detached"));
        };
        if offset > buflen {
            return Err(i.make_error("RangeError", "DataView byteOffset is out of bounds"));
        }
        let len = match len_arg {
            Some(l) => {
                if l > buflen - offset {
                    return Err(i.make_error("RangeError", "DataView byteLength is out of bounds"));
                }
                l
            }
            None => buflen - offset,
        };
        // A length-tracking DataView (no explicit byteLength) over a resizable buffer follows the
        // buffer's current length; its stored `len` is only the initial snapshot.
        let track = !has_len && resizable;
        let p = Gc::as_ptr(&obj) as usize;
        i.gc_pin(&obj);
        i.data_views.insert(p, (bp, offset, len, track));
        // buffer/byteOffset/byteLength are accessor getters on the prototype, not own properties;
        // only the buffer object itself is kept (hidden) for the `buffer` getter.
        set_internal(&obj, "#\u{0}dv_buffer", arg(a, 0));
        Ok(Value::Obj(obj))
    });
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor)));
    set_builtin(&it.global, "DataView", Value::Obj(ctor));
}

#[cfg(test)]
#[path = "../dataview_access_tests.rs"]
mod tests;
