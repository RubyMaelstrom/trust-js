//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;

pub(super) fn install_shared_array_buffer(it: &mut Interp) {
    // Modeled as a plain ArrayBuffer (no real sharing) — enough for tests that just need the type.
    let proto = Object::new(Some(it.object_proto.clone()));
    Rc::make_mut(&mut it.extra_protos).insert("SharedArrayBuffer", proto.clone());
    // byteLength/maxByteLength/growable accessor getters on the prototype.
    // SharedArrayBuffer.prototype accessors require a shared buffer: reject a plain ArrayBuffer
    // `this` with a TypeError (the shared side table is the discriminator).
    fn require_shared_buffer(i: &Interp, this: &Value) -> Result<(), Value> {
        let shared = matches!(this, Value::Obj(o) if i.shared_buffers.contains_key(&(Rc::as_ptr(o) as usize)));
        if !shared {
            return Err(i.make_error("TypeError", "requires a SharedArrayBuffer"));
        }
        Ok(())
    }
    let sab_getter = |it: &mut Interp, proto: &Gc, name: &str, f: NativeFn| {
        let g = it.make_native(&format!("get {name}"), 0, f);
        proto.borrow_mut().props.insert(
            name,
            Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
        );
    };
    sab_getter(it, &proto, "byteLength", |i, this, _| {
        require_shared_buffer(i, &this)?;
        let p = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .map(|o| Rc::as_ptr(o) as usize)
            .ok_or_else(|| i.make_error("TypeError", "not a SharedArrayBuffer"))?;
        Ok(Value::Num(i.array_buffer_byte_length(p).unwrap_or(0) as f64))
    });
    sab_getter(it, &proto, "maxByteLength", |i, this, _| {
        require_shared_buffer(i, &this)?;
        this.as_obj()
            .and_then(|o| {
                o.borrow()
                    .props
                    .get("#\u{0}abMaxByteLength")
                    .map(|p| p.value())
            })
            .ok_or_else(|| i.make_error("TypeError", "not a SharedArrayBuffer"))
    });
    sab_getter(it, &proto, "growable", |i, this, _| {
        require_shared_buffer(i, &this)?;
        this.as_obj()
            .and_then(|o| o.borrow().props.get("#\u{0}abResizable").map(|p| p.value()))
            .ok_or_else(|| i.make_error("TypeError", "not a SharedArrayBuffer"))
    });
    // grow(newLength): only allowed for a growable buffer and only to a larger size.
    it.def_method(&proto, "grow", 1, |i, this, a| {
        // ECMA-262 #sec-sharedarraybuffer.prototype.grow: RequireInternalSlot(O,
        // [[ArrayBufferMaxByteLength]]), IsSharedArrayBuffer(O), then ToIndex(newLength).
        let max = resizable_max_byte_length(&this)
            .ok_or_else(|| i.make_error("TypeError", "SharedArrayBuffer is not growable"))?;
        let ptr = this.as_obj().map(|o| Rc::as_ptr(o) as usize).unwrap_or(0);
        if !i.shared_buffers.contains_key(&ptr) {
            return Err(i.make_error("TypeError", "grow requires a SharedArrayBuffer"));
        }
        let new_len = to_index(i, &arg(a, 0))?;
        // The length is the Shared Data Block's, which every agent sharing the buffer sees.
        // Holding its lock across the check and the resize orders racing grows, as the spec's
        // compare-exchange loop on [[ArrayBufferByteLengthData]] does.
        let Some(block) = crate::interpreter::shared_mem_get(i.shared_buffers[&ptr]) else {
            return Err(i.make_error("TypeError", "grow requires a SharedArrayBuffer"));
        };
        let mut bytes = block.lock().unwrap();
        let cur = bytes.len();
        if new_len == cur {
            return Ok(Value::Undefined);
        }
        if new_len < cur || new_len > max || bytes.try_reserve_exact(new_len - cur).is_err() {
            return Err(i.make_error("RangeError", "SharedArrayBuffer grow out of range"));
        }
        bytes.resize(new_len, 0);
        Ok(Value::Undefined)
    });
    if let Some(key) = well_known_key(it, "toStringTag") {
        proto.borrow_mut().props.insert(
            key,
            Property::data(Value::lstr("SharedArrayBuffer"), false, false, true),
        );
    }
    it.def_method(&proto, "slice", 2, |i, this, a| {
        // RequireInternalSlot + IsSharedArrayBuffer(O).
        let o = this
            .as_obj()
            .filter(|o| i.shared_buffers.contains_key(&(Rc::as_ptr(o) as usize)))
            .ok_or_else(|| {
                i.make_error(
                    "TypeError",
                    "SharedArrayBuffer.prototype.slice requires a SharedArrayBuffer",
                )
            })?;
        let ptr = Rc::as_ptr(o) as usize;
        let len = i.array_buffer_byte_length(ptr).unwrap_or(0) as i64;
        let begin = norm_index(ab(i.to_number(&arg(a, 0)))?, len);
        let end = match arg(a, 1) {
            Value::Undefined => len,
            v => norm_index(ab(i.to_number(&v))?, len),
        };
        let new_len = (end - begin).max(0) as usize;
        // SpeciesConstructor(O, %SharedArrayBuffer%) makes the result, validated as a distinct,
        // large-enough SharedArrayBuffer.
        let sab_ctor = i
            .global
            .borrow()
            .props
            .get("SharedArrayBuffer")
            .map(|p| p.value())
            .unwrap_or(Value::Undefined);
        let ctor = species_constructor(i, &this, &sab_ctor)?;
        let new_buf = ab(i.construct(ctor, &[Value::Num(new_len as f64)]))?;
        let nptr = match &new_buf {
            Value::Obj(no) if i.shared_buffers.contains_key(&(Rc::as_ptr(no) as usize)) => {
                Rc::as_ptr(no) as usize
            }
            _ => {
                return Err(i.make_error(
                    "TypeError",
                    "slice species did not create a SharedArrayBuffer",
                ));
            }
        };
        if nptr == ptr {
            return Err(i.make_error(
                "TypeError",
                "slice species returned the same SharedArrayBuffer",
            ));
        }
        // The new buffer must be at least `new_len` bytes.
        if i.array_buffer_byte_length(nptr).unwrap_or(0) < new_len {
            return Err(i.make_error("TypeError", "slice species buffer is too small"));
        }
        // Both buffers' bytes live in their process-global Shared Data Blocks.
        let block = |i: &Interp, p: usize| crate::interpreter::shared_mem_get(i.shared_buffers[&p]);
        if let (true, Some(src), Some(dst)) = (begin < end, block(i, ptr), block(i, nptr)) {
            let slice = src
                .lock()
                .unwrap()
                .get(begin as usize..end as usize)
                .map(<[u8]>::to_vec);
            if let Some(slice) = slice {
                let mut buf = dst.lock().unwrap();
                let n = slice.len().min(buf.len());
                buf[..n].copy_from_slice(&slice[..n]);
            }
        }
        Ok(new_buf)
    });
    let ctor = it.make_native("SharedArrayBuffer", 1, |i, _t, a| {
        if !i.constructing {
            return Err(i.make_error("TypeError", "SharedArrayBuffer constructor requires 'new'"));
        }
        // ToIndex(length), then the options bag's maxByteLength — both *before* object creation
        // (a poisoned newTarget.prototype getter fires next), with allocation limits checked last.
        let nraw = ab(i.to_number(&arg(a, 0)))?;
        let n = if nraw.is_nan() { 0.0 } else { nraw.trunc() };
        if !(0.0..=9007199254740991.0).contains(&n) {
            return Err(i.make_error("RangeError", "Invalid SharedArrayBuffer length"));
        }
        let mut max: Option<f64> = None;
        if let Value::Obj(_) = arg(a, 1) {
            let mbl = ab(i.get_member(&arg(a, 1), "maxByteLength"))?;
            if !matches!(mbl, Value::Undefined) {
                let m = ab(i.to_number(&mbl))?;
                let m = if m.is_nan() { 0.0 } else { m.trunc() };
                if !(0.0..=9007199254740991.0).contains(&m) || m < n {
                    return Err(i.make_error("RangeError", "Invalid maxByteLength"));
                }
                max = Some(m);
            }
        }
        // OrdinaryCreateFromConstructor: the prototype comes from newTarget (abrupt propagates).
        let obj = new_from_ctor(i, "SharedArrayBuffer")?;
        // Data allocation happens after the object exists — an oversized request fails here.
        let len = n as usize;
        if n as u128 > MAX_BUFFER_BYTES as u128
            || max.is_some_and(|m| m as u128 > MAX_BUFFER_BYTES as u128)
        {
            return Err(i.make_error("RangeError", "SharedArrayBuffer allocation too large"));
        }
        let bp = Rc::as_ptr(&obj) as usize;
        i.gc_pin(&obj);
        // The bytes live in the Shared Data Block allocated below; this entry holds none.
        i.array_buffers.insert(bp, Default::default());
        set_internal(&obj, "#\u{0}abMaxByteLength", Value::Num(max.unwrap_or(n)));
        set_internal(&obj, "#\u{0}abResizable", Value::Bool(max.is_some()));
        let id = crate::interpreter::alloc_shared_mem(len);
        i.shared_buffers.insert(bp, id);
        set_internal(&obj, "#\u{0}sab_id", Value::Num(id as f64));
        Ok(Value::Obj(obj))
    });
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto.clone()), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
    install_species(it, &ctor); // SharedArrayBuffer[@@species] returns `this`
    set_builtin(&it.global, "SharedArrayBuffer", Value::Obj(ctor));
}

/// ArrayBuffer.prototype.transfer / transferToFixedLength: move the bytes into a fresh fixed-length
/// buffer and detach the source (a TypeError if it's already detached).
fn ab_transfer(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    ab_transfer_impl(i, this, a, false)
}
fn ab_transfer_fixed(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
    ab_transfer_impl(i, this, a, true)
}
/// ArrayBuffer.prototype.transfer / transferToFixedLength: copy into a new buffer of `newLength` and
/// detach the source. `transfer` preserves the source's resizability (its maxByteLength);
/// `transferToFixedLength` always produces a fixed-length buffer.
fn ab_transfer_impl(i: &mut Interp, this: Value, a: &[Value], fixed: bool) -> Result<Value, Value> {
    let o = this
        .as_obj()
        .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
        .cloned()
        .ok_or_else(|| i.make_error("TypeError", "not an ArrayBuffer"))?;
    let ptr = Rc::as_ptr(&o) as usize;
    if i.shared_buffers.contains_key(&ptr) {
        return Err(i.make_error("TypeError", "transfer requires a non-shared ArrayBuffer"));
    }
    // The newLength argument is read (observably) before detachability is verified.
    let new_len = match arg(a, 0) {
        Value::Undefined => None,
        v => {
            let n = ab(i.to_number(&v))?;
            let n = if n.is_nan() { 0.0 } else { n.trunc() };
            if n < 0.0 || !n.is_finite() || n as usize > MAX_BUFFER_BYTES {
                return Err(i.make_error("RangeError", "invalid transfer length"));
            }
            Some(n as usize)
        }
    };
    if i.immutable_buffers.contains(&ptr) {
        return Err(i.make_error("TypeError", "an immutable ArrayBuffer is not detachable"));
    }
    if i.host_keyed_buffers.contains(&ptr) {
        return Err(i.make_error("TypeError", "ArrayBuffer detach key does not match"));
    }
    if !i.array_buffers.contains_key(&ptr) {
        return Err(i.make_error("TypeError", "ArrayBuffer is detached"));
    }
    let new_len = new_len.unwrap_or_else(|| i.array_buffers[&ptr].borrow().len());
    // ArrayBufferCopyAndDetach: transfer() preserves the source's resizability (its
    // [[ArrayBufferMaxByteLength]] slot); transferToFixedLength() is always fixed.
    let src_max = resizable_max_byte_length(&this);
    // Transfer ownership of the Vec rather than cloning it. This is the engine primitive behind
    // HTML structured transfer; same-length transfers must remain O(1) in backing-store bytes.
    i.invalidate_native_buffer(ptr);
    let bytes = i
        .array_buffers
        .remove(&ptr)
        .expect("the detached check above established a backing store");
    bytes.borrow_mut().resize(new_len, 0);
    let (bv, bp) = make_array_buffer(i, new_len);
    i.array_buffers.insert(bp, bytes);
    if let (false, Some(src_max)) = (fixed, src_max) {
        if let Value::Obj(nb) = &bv {
            set_internal(nb, "#\u{0}abMaxByteLength", Value::Num(src_max as f64));
            set_internal(nb, "#\u{0}abResizable", Value::Bool(true));
        }
    }
    // Removing the source backing above detached it; byteLength/detached derive from this table.
    Ok(bv)
}

/// The [[ArrayBufferMaxByteLength]] internal slot of a resizable ArrayBuffer or growable
/// SharedArrayBuffer; `None` for a fixed-length buffer (which lacks the slot) or another value.
/// Read from the hidden internal properties, never through the author-visible `resizable` /
/// `growable` / `maxByteLength` accessors, which a page can shadow or redefine.
fn resizable_max_byte_length(this: &Value) -> Option<usize> {
    let o = this.as_obj()?.borrow();
    if !matches!(
        o.props.get("#\u{0}abResizable").map(|p| p.value()),
        Some(Value::Bool(true))
    ) {
        return None;
    }
    match o.props.get("#\u{0}abMaxByteLength").map(|p| p.value()) {
        Some(Value::Num(max)) => Some(max as usize),
        _ => None,
    }
}

/// ECMA-262 #sec-toindex for the resize/grow length argument.
fn to_index(i: &mut Interp, value: &Value) -> Result<usize, Value> {
    let n = ab(i.to_number(value))?;
    let n = if n.is_nan() { 0.0 } else { n.trunc() };
    if !(0.0..=9_007_199_254_740_991.0).contains(&n) {
        return Err(i.make_error("RangeError", "length out of range"));
    }
    Ok(n as usize)
}

fn make_array_buffer(i: &mut Interp, byte_len: usize) -> (Value, usize) {
    make_array_buffer_from_bytes(i, vec![0u8; byte_len])
}

fn make_array_buffer_from_bytes(i: &mut Interp, bytes: Vec<u8>) -> (Value, usize) {
    array_buffer_with_storage(i, Rc::new(RefCell::new(bytes)))
}

/// A fresh fixed-length ArrayBuffer whose [[ArrayBufferData]] is `storage`, with this realm's
/// intrinsic %ArrayBuffer.prototype%. It runs no author code: unlike `new globalThis.ArrayBuffer`,
/// a replaced global binding or species constructor cannot observe or substitute the buffer.
pub(crate) fn array_buffer_with_storage(
    i: &mut Interp,
    storage: crate::interpreter::ArrayBufferBytes,
) -> (Value, usize) {
    let byte_len = storage.borrow().len();
    let obj = Object::new(i.extra_protos.get("ArrayBuffer").cloned());
    let p = Rc::as_ptr(&obj) as usize;
    i.gc_pin(&obj);
    i.array_buffers.insert(p, storage);
    // byteLength/detached derive from the side table; only max/resizable need stored slots, hidden
    // behind the `__ab*` prefix and surfaced through prototype accessor getters.
    set_internal(&obj, "#\u{0}abMaxByteLength", Value::Num(byte_len as f64));
    set_internal(&obj, "#\u{0}abResizable", Value::Bool(false));
    (Value::Obj(obj), p)
}

pub(super) fn install_array_buffer(it: &mut Interp) {
    let proto = Object::new(Some(it.object_proto.clone()));
    Rc::make_mut(&mut it.extra_protos).insert("ArrayBuffer", proto.clone());
    set_to_string_tag(it, &proto, "ArrayBuffer");
    // byteLength/maxByteLength/resizable/detached are accessor getters on the prototype.
    // ArrayBuffer.prototype accessors/methods require a non-shared buffer: reject a SharedArrayBuffer
    // `this` with a TypeError (both buffer kinds carry `__abMaxByteLength`, so brand alone isn't enough).
    fn reject_shared_buffer(i: &Interp, this: &Value) -> Result<(), Value> {
        if let Value::Obj(o) = this {
            if i.shared_buffers.contains_key(&(Rc::as_ptr(o) as usize)) {
                return Err(i.make_error("TypeError", "requires a non-shared ArrayBuffer"));
            }
        }
        Ok(())
    }
    let ab_getter = |it: &mut Interp, proto: &Gc, name: &str, f: NativeFn| {
        let g = it.make_native(&format!("get {name}"), 0, f);
        proto.borrow_mut().props.insert(
            name,
            Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
        );
    };
    ab_getter(it, &proto, "byteLength", |i, this, _| {
        reject_shared_buffer(i, &this)?;
        let p = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .map(|o| Rc::as_ptr(o) as usize)
            .ok_or_else(|| i.make_error("TypeError", "not an ArrayBuffer"))?;
        // Detached (absent from the side table) → 0.
        Ok(Value::Num(
            i.array_buffers
                .get(&p)
                .map(|b| b.borrow().len())
                .unwrap_or(0) as f64,
        ))
    });
    ab_getter(it, &proto, "maxByteLength", |i, this, _| {
        reject_shared_buffer(i, &this)?;
        match this.as_obj().and_then(|o| {
            o.borrow()
                .props
                .get("#\u{0}abMaxByteLength")
                .map(|pr| pr.value())
        }) {
            Some(v) => {
                // A detached buffer reports 0.
                let detached = this
                    .as_obj()
                    .map(|o| !i.array_buffers.contains_key(&(Rc::as_ptr(o) as usize)))
                    .unwrap_or(true);
                Ok(if detached { Value::Num(0.0) } else { v })
            }
            None => Err(i.make_error("TypeError", "not an ArrayBuffer")),
        }
    });
    ab_getter(it, &proto, "resizable", |i, this, _| {
        reject_shared_buffer(i, &this)?;
        match this.as_obj().and_then(|o| {
            o.borrow()
                .props
                .get("#\u{0}abResizable")
                .map(|pr| pr.value())
        }) {
            Some(v) => Ok(v),
            None => Err(i.make_error("TypeError", "not an ArrayBuffer")),
        }
    });
    ab_getter(it, &proto, "detached", |i, this, _| {
        reject_shared_buffer(i, &this)?;
        let o = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .ok_or_else(|| i.make_error("TypeError", "not an ArrayBuffer"))?;
        Ok(Value::Bool(
            !i.array_buffers.contains_key(&(Rc::as_ptr(o) as usize)),
        ))
    });
    ab_getter(it, &proto, "immutable", |i, this, _| {
        reject_shared_buffer(i, &this)?;
        let o = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .ok_or_else(|| i.make_error("TypeError", "not an ArrayBuffer"))?;
        Ok(Value::Bool(
            i.immutable_buffers.contains(&(Rc::as_ptr(o) as usize)),
        ))
    });
    it.def_method(&proto, "slice", 2, |i, this, a| {
        // RequireInternalSlot([[ArrayBufferData]]), reject a shared buffer, then a detached one.
        reject_shared_buffer(i, &this)?;
        let o = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .ok_or_else(|| {
                i.make_error(
                    "TypeError",
                    "ArrayBuffer.prototype.slice requires an ArrayBuffer",
                )
            })?;
        let ptr = Rc::as_ptr(o) as usize;
        if !i.array_buffers.contains_key(&ptr) {
            return Err(i.make_error("TypeError", "Cannot slice a detached ArrayBuffer"));
        }
        let len = i.array_buffers[&ptr].borrow().len() as i64;
        let begin = norm_index(ab(i.to_number(&arg(a, 0)))?, len);
        let end = match arg(a, 1) {
            Value::Undefined => len,
            v => norm_index(ab(i.to_number(&v))?, len),
        };
        let new_len = (end - begin).max(0) as usize;
        // SpeciesConstructor(O, %ArrayBuffer%) creates the result buffer, then it is validated.
        let ab_ctor = i
            .global
            .borrow()
            .props
            .get("ArrayBuffer")
            .map(|p| p.value())
            .unwrap_or(Value::Undefined);
        let ctor = species_constructor(i, &this, &ab_ctor)?;
        let new_buf = ab(i.construct(ctor, &[Value::Num(new_len as f64)]))?;
        let nptr = match &new_buf {
            Value::Obj(no) if no.borrow().props.contains("#\u{0}abMaxByteLength") => {
                Rc::as_ptr(no) as usize
            }
            _ => {
                return Err(i.make_error("TypeError", "slice species did not create an ArrayBuffer"))
            }
        };
        if i.shared_buffers.contains_key(&nptr) {
            return Err(i.make_error("TypeError", "slice species returned a SharedArrayBuffer"));
        }
        if !i.array_buffers.contains_key(&nptr) {
            return Err(i.make_error("TypeError", "slice species returned a detached ArrayBuffer"));
        }
        if nptr == ptr {
            return Err(i.make_error("TypeError", "slice species returned the same ArrayBuffer"));
        }
        if i.immutable_buffers.contains(&nptr) {
            return Err(i.make_error("TypeError", "slice species returned an immutable buffer"));
        }
        if i.array_buffers[&nptr].borrow().len() < new_len {
            return Err(i.make_error("TypeError", "slice species buffer is too small"));
        }
        // The species constructor may have detached or shrunk the source.
        if !i.array_buffers.contains_key(&ptr) {
            return Err(i.make_error("TypeError", "source ArrayBuffer was detached during slice"));
        }
        // CopyDataBlockBytes(toBuf, 0, fromBuf, first, min(newLength, currentLength - first)):
        // copy just that range. Two buffer objects can share one host Data Block (the embedder
        // API identifies storage), so a shared block copies within itself instead of borrowing
        // it twice.
        let (Some(src), Some(dst)) = (
            i.array_buffers.get(&ptr).cloned(),
            i.array_buffers.get(&nptr).cloned(),
        ) else {
            return Ok(new_buf);
        };
        if Rc::ptr_eq(&src, &dst) {
            let mut bytes = src.borrow_mut();
            let len = bytes.len() as i64;
            let (begin, end) = (begin.min(len), end.min(len));
            if begin < end {
                bytes.copy_within(begin as usize..end as usize, 0);
            }
        } else {
            let src = src.borrow();
            let (begin, end) = (begin.min(src.len() as i64), end.min(src.len() as i64));
            if begin < end {
                let range = &src[begin as usize..end as usize];
                dst.borrow_mut()[..range.len()].copy_from_slice(range);
            }
        }
        Ok(new_buf)
    });
    it.def_method(&proto, "resize", 1, |i, this, a| {
        // ECMA-262 #sec-arraybuffer.prototype.resize: RequireInternalSlot(O,
        // [[ArrayBufferMaxByteLength]]) — a fixed-length buffer lacks it — before any coercion.
        let o = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .cloned()
            .ok_or_else(|| i.make_error("TypeError", "not an ArrayBuffer"))?;
        let ptr = Rc::as_ptr(&o) as usize;
        let max = resizable_max_byte_length(&this)
            .ok_or_else(|| i.make_error("TypeError", "ArrayBuffer is not resizable"))?;
        if i.shared_buffers.contains_key(&ptr) {
            return Err(i.make_error("TypeError", "resize requires a non-shared ArrayBuffer"));
        }
        let new_len = to_index(i, &arg(a, 0))?;
        // ToIndex may have detached the buffer: a TypeError, before the max-length RangeError.
        if !i.array_buffers.contains_key(&ptr) {
            return Err(i.make_error("TypeError", "ArrayBuffer is detached"));
        }
        if new_len > max {
            return Err(i.make_error("RangeError", "ArrayBuffer resize out of range"));
        }
        if let Some(buf) = i.array_buffers.get(&ptr) {
            buf.borrow_mut().resize(new_len, 0);
            i.mark_array_buffer_dirty(ptr);
        }
        // byteLength derives from the backing store length, which the resize above updated.
        Ok(Value::Undefined)
    });
    it.def_method(&proto, "transfer", 0, ab_transfer);
    it.def_method(&proto, "transferToFixedLength", 0, ab_transfer_fixed);
    // Immutable-ArrayBuffer proposal: move the bytes into a fresh immutable (always fixed-length)
    // buffer, detaching the source; or copy a range into an immutable buffer (source intact).
    it.def_method(&proto, "transferToImmutable", 0, |i, this, a| {
        let bv = ab_transfer_fixed(i, this, a)?;
        if let Value::Obj(o) = &bv {
            i.immutable_buffers.insert(Rc::as_ptr(o) as usize);
        }
        Ok(bv)
    });
    it.def_method(&proto, "sliceToImmutable", 2, |i, this, a| {
        reject_shared_buffer(i, &this)?;
        let ptr = this
            .as_obj()
            .filter(|o| o.borrow().props.contains("#\u{0}abMaxByteLength"))
            .map(|o| Rc::as_ptr(o) as usize)
            .ok_or_else(|| i.make_error("TypeError", "not an ArrayBuffer"))?;
        if !i.array_buffers.contains_key(&ptr) {
            return Err(i.make_error("TypeError", "ArrayBuffer is detached"));
        }
        let len = i.array_buffers[&ptr].borrow().len() as i64;
        let begin = norm_index(ab(i.to_number(&arg(a, 0)))?, len);
        let end = match arg(a, 1) {
            Value::Undefined => len,
            v => norm_index(ab(i.to_number(&v))?, len),
        };
        // Coercing start/end may have detached or shrunk the source buffer: detachment is a
        // TypeError, and a resolved range past the current length is a RangeError.
        let storage = i
            .array_buffers
            .get(&ptr)
            .cloned()
            .ok_or_else(|| i.make_error("TypeError", "ArrayBuffer is detached"))?;
        let bytes = storage.borrow();
        let cur = bytes.len() as i64;
        if begin < end && end > cur {
            return Err(i.make_error(
                "RangeError",
                "source ArrayBuffer shrank below the resolved range",
            ));
        }
        let slice = if begin < end {
            bytes[begin as usize..end as usize].to_vec()
        } else {
            Vec::new()
        };
        let (bv, bp) = make_array_buffer(i, slice.len());
        if let Some(buf) = i.array_buffers.get(&bp) {
            buf.borrow_mut().copy_from_slice(&slice);
        }
        i.immutable_buffers.insert(bp);
        Ok(bv)
    });
    let ctor = it.make_native("ArrayBuffer", 1, |i, _t, a| {
        if !i.constructing {
            return Err(i.make_error("TypeError", "ArrayBuffer constructor requires 'new'"));
        }
        // ToIndex(length): NaN → 0, truncate; negative or non-finite is a RangeError.
        let n = ab(i.to_number(&arg(a, 0)))?;
        let n = if n.is_nan() { 0.0 } else { n.trunc() };
        if n < 0.0 || !n.is_finite() {
            return Err(i.make_error("RangeError", "Invalid ArrayBuffer length"));
        }
        // GetArrayBufferMaxByteLengthOption: coerced before the object is created.
        let max: Option<f64> = if let Value::Obj(_) = arg(a, 1) {
            let mbl = ab(i.get_member(&arg(a, 1), "maxByteLength"))?;
            if matches!(mbl, Value::Undefined) {
                None
            } else {
                let m = ab(i.to_number(&mbl))?;
                let m = if m.is_nan() { 0.0 } else { m.trunc() };
                if m < 0.0 || !m.is_finite() {
                    return Err(i.make_error("RangeError", "Invalid maxByteLength"));
                }
                Some(m)
            }
        } else {
            None
        };
        if let Some(m) = max {
            if m < n {
                return Err(i.make_error("RangeError", "maxByteLength is below the length"));
            }
        }
        // OrdinaryCreateFromConstructor: new.target's prototype is read (observably) before the
        // data block is allocated, so a poisoned getter beats an allocation RangeError.
        let obj = new_from_ctor(i, "ArrayBuffer")?;
        if n as usize > MAX_BUFFER_BYTES || max.is_some_and(|m| m as usize > MAX_BUFFER_BYTES) {
            return Err(i.make_error("RangeError", "ArrayBuffer allocation too large"));
        }
        let len = n as usize;
        let p = Rc::as_ptr(&obj) as usize;
        i.gc_pin(&obj);
        i.array_buffers
            .insert(p, Rc::new(RefCell::new(vec![0u8; len])));
        set_internal(&obj, "#\u{0}abMaxByteLength", Value::Num(max.unwrap_or(n)));
        set_internal(&obj, "#\u{0}abResizable", Value::Bool(max.is_some()));
        Ok(Value::Obj(obj))
    });
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto.clone()), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
    it.def_method(&ctor, "isView", 1, |i, _t, a| {
        // A view is a TypedArray or a DataView (identified by its `__dv_buffer` internal slot).
        let is_view = match arg(a, 0) {
            Value::Obj(o) => {
                i.typed_arrays.contains_key(&(Rc::as_ptr(&o) as usize))
                    || o.borrow().props.contains("#\u{0}dv_buffer")
            }
            _ => false,
        };
        Ok(Value::Bool(is_view))
    });
    install_species(it, &ctor); // ArrayBuffer[@@species] returns `this`
    set_builtin(&it.global, "ArrayBuffer", Value::Obj(ctor));
}

/// A %TypedArray%.prototype method: brand-check the receiver, then delegate to the like-named
/// Array.prototype method.
fn ta_delegate(i: &mut Interp, this: &Value, method: &str, args: &[Value]) -> Result<Value, Value> {
    let info = map_ptr(this).and_then(|p| i.typed_arrays.get(&p).copied());
    let info = match info {
        Some(info) => info,
        None => return Err(i.make_error("TypeError", "method called on a non-TypedArray receiver")),
    };
    // ValidateTypedArray: a detached (or out-of-bounds, after a resizable buffer shrank) backing
    // buffer makes the operation throw.
    if i.ta_len(&info).is_none() {
        return Err(i.make_error(
            "TypeError",
            "Cannot perform operation on a detached or out-of-bounds TypedArray",
        ));
    }
    // %TypedArray%.prototype.with has TypedArray-specific semantics (value coerced to the element
    // type *before* the index bounds check) that the generic Array delegation can't reproduce.
    if method == "with" {
        let len = i.ta_len(&info).unwrap_or(0);
        let rel = ab(i.to_number(&arg(args, 0)))?;
        let rel = if rel.is_nan() { 0.0 } else { rel.trunc() };
        let actual = if rel >= 0.0 { rel } else { len as f64 + rel };
        // Coerce the value to the element type (this can run user code).
        let coerced = if info.kind.is_bigint() {
            Value::BigInt(ab(i.to_bigint(&arg(args, 1)))?)
        } else {
            Value::Num(ab(i.to_number(&arg(args, 1)))?)
        };
        // IsValidIntegerIndex is re-checked against the *current* length: coercing the value may
        // have grown or shrunk a resizable backing buffer.
        let cur_len = i.ta_len(&info).unwrap_or(0);
        if actual < 0.0 || actual >= cur_len as f64 {
            return Err(i.make_error("RangeError", "invalid TypedArray index"));
        }
        let actual = actual as usize;
        // `with` uses TypedArrayCreateSameType — it ignores `@@species`.
        let (new_ta, new_info) = ta_create_same(i, info.kind, len)?;
        if !i.shared_buffers.contains_key(&info.buffer) && i.ta_len(&info) == Some(len) {
            if let Some(mut bytes) = i.ta_read_bytes(&info, 0, len) {
                // TypedArrayCreateSameType gives the result a distinct ordinary buffer. Once the
                // required coercions and index check have completed, copy the source encoding and
                // patch the replacement element without boxing every other element (ECMA-262
                // §23.2.3.36).
                let replacement = match &coerced {
                    Value::BigInt(n) if info.kind.is_bigint() => {
                        info.kind.write_bigint(n.to_i128_wrapping())
                    }
                    Value::Num(n) if !info.kind.is_bigint() => info.kind.write(*n),
                    _ => Vec::new(),
                };
                if replacement.len() == info.kind.elsize() {
                    let offset = actual * replacement.len();
                    bytes[offset..offset + replacement.len()].copy_from_slice(&replacement);
                    i.ta_write_bytes(&new_info, 0, &bytes);
                    return Ok(new_ta);
                }
            }
        }
        for k in 0..len {
            let val = if k == actual {
                coerced.clone()
            } else {
                i.ta_read(&info, k)
            };
            ab(i.ta_store(&new_info, k, &val))?;
        }
        return Ok(new_ta);
    }
    // Most iteration / search / mutation methods have TypedArray-specific semantics the generic
    // Array delegation can't reproduce: the length is captured once up front, out-of-bounds reads
    // (after a mid-operation detach or resizable-buffer shrink) yield `undefined` instead of
    // stopping early, and the mutators re-validate the buffer after argument coercion.
    if let Some(r) = ta_native(i, this, &info, method, args) {
        return r;
    }
    let f = it_array_method(i, method);
    let result = ab(i.call(f, this.clone(), args))?;
    // Methods that produce a new collection return a TypedArray built via TypedArraySpeciesCreate
    // (the receiver's `constructor[@@species]`, falling back to the kind's intrinsic constructor).
    if matches!(
        method,
        "map" | "filter" | "slice" | "toReversed" | "toSorted" | "with"
    ) {
        let len = ab(i.get_member(&result, "length"))?;
        let len = ab(i.to_number(&len))? as usize;
        let new_ta = ta_species_create(i, this, info.kind, &[Value::Num(len as f64)], true)?;
        // SpeciesConstructor access can run user code that detaches the source — re-validate.
        if !i.array_buffers.contains_key(&info.buffer) {
            return Err(i.make_error(
                "TypeError",
                "Cannot perform operation on a detached ArrayBuffer",
            ));
        }
        for k in 0..len {
            let v = ab(i.get_member(&result, &k.to_string()))?;
            ab(i.set_member(&new_ta, &k.to_string(), v))?;
        }
        return Ok(new_ta);
    }
    Ok(result)
}

/// ECMA-262 #sec-typedarraycreatesametype for toSorted/toReversed/with: construct through the
/// Table 71 *intrinsic* constructor of the exemplar's type (never a global binding or
/// `@@species`). Constructing the intrinsic with a length runs no author code and allocates
/// with its non-writable `prototype`, so allocating directly is observably identical.
fn ta_create_same(i: &mut Interp, kind: TaKind, len: usize) -> Result<(Value, TaInfo), Value> {
    if len > MAX_BUFFER_BYTES / kind.elsize() {
        return Err(i.make_error(
            "RangeError",
            "TypedArray backing buffer exceeds allocation limit",
        ));
    }
    Ok(typed_array_from_bytes(
        i,
        kind,
        vec![0; len * kind.elsize()],
    ))
}

/// AllocateTypedArray (ECMA-262 #sec-allocatetypedarray) with the current Realm's intrinsic
/// %<kind>.prototype% over a fresh fixed-length ArrayBuffer holding `bytes` (a whole number of
/// elements). It runs no author code, so a replaced global constructor cannot observe or
/// substitute the result.
pub(crate) fn typed_array_from_bytes(
    i: &mut Interp,
    kind: TaKind,
    bytes: Vec<u8>,
) -> (Value, TaInfo) {
    let len = bytes.len() / kind.elsize();
    let (buf_val, buf_ptr) = make_array_buffer_from_bytes(i, bytes);
    typed_array_over(i, kind, buf_val, buf_ptr, len)
}

/// A `bytes` module's default export: a Uint8Array over a fresh immutable ArrayBuffer holding
/// `bytes` (Import Bytes proposal), from the current Realm's intrinsics without author code.
pub(crate) fn immutable_uint8_array(i: &mut Interp, bytes: Vec<u8>) -> Value {
    let len = bytes.len();
    let (buf_val, buf_ptr) = make_array_buffer_from_bytes(i, bytes);
    // Immutable before the view registers, so native code never gets a writable view of it.
    i.immutable_buffers.insert(buf_ptr);
    typed_array_over(i, TaKind::U8, buf_val, buf_ptr, len).0
}

/// A `kind` TypedArray viewing all `len` elements of the fresh ArrayBuffer `buf_val`.
fn typed_array_over(
    i: &mut Interp,
    kind: TaKind,
    buf_val: Value,
    buf_ptr: usize,
    len: usize,
) -> (Value, TaInfo) {
    let obj = Object::new(i.extra_protos.get(kind.name()).cloned());
    let p = Rc::as_ptr(&obj) as usize;
    i.gc_pin(&obj);
    obj.borrow().ic_plain.set(false);
    let info = TaInfo {
        buffer: buf_ptr,
        offset: 0,
        len,
        kind,
        track: false,
    };
    i.typed_arrays.insert(p, info);
    i.ta_buffer.insert(p, buf_val);
    i.register_native_typed_array(&obj, info);
    (Value::Obj(obj), info)
}

/// CompareTypedArrayElements default (no comparefn): a total order over numbers where NaN sorts last
/// and -0 precedes +0.
fn ta_default_cmp_num(a: f64, b: f64) -> i32 {
    if a.is_nan() {
        return if b.is_nan() { 0 } else { 1 };
    }
    if b.is_nan() {
        return -1;
    }
    if a < b {
        -1
    } else if a > b {
        1
    } else {
        // Equal magnitude: distinguish -0 (< +0).
        match (a.is_sign_negative(), b.is_sign_negative()) {
            (true, false) => -1,
            (false, true) => 1,
            _ => 0,
        }
    }
}

/// SortCompare for a TypedArray element pair: a user comparefn (result ToNumber, NaN treated as 0)
/// or the default numeric order.
fn ta_sort_compare(
    i: &mut Interp,
    a: &Value,
    b: &Value,
    cmp: Option<&crate::callback::Callback>,
    is_bigint: bool,
) -> Result<i32, Value> {
    if let Some(cmp) = cmp {
        let r = ab(cmp.call(i, Value::Undefined, [a.clone(), b.clone()]))?;
        let n = ab(i.to_number(&r))?;
        return Ok(if n.is_nan() {
            0
        } else if n < 0.0 {
            -1
        } else if n > 0.0 {
            1
        } else {
            0
        });
    }
    if is_bigint {
        let ord = match (a, b) {
            (Value::BigInt(x), Value::BigInt(y)) => x.cmp(y),
            _ => std::cmp::Ordering::Equal,
        };
        Ok(ord as i32)
    } else {
        let x = if let Value::Num(x) = a { *x } else { f64::NAN };
        let y = if let Value::Num(y) = b { *y } else { f64::NAN };
        Ok(ta_default_cmp_num(x, y))
    }
}

/// A stable merge sort that propagates a throwing comparefn.
fn ta_merge_sort(
    i: &mut Interp,
    vals: &mut [Value],
    cmp: Option<&crate::callback::Callback>,
    is_bigint: bool,
) -> Result<(), Value> {
    let n = vals.len();
    if n < 2 {
        return Ok(());
    }
    let mid = n / 2;
    let mut left = vals[..mid].to_vec();
    let mut right = vals[mid..].to_vec();
    ta_merge_sort(i, &mut left, cmp, is_bigint)?;
    ta_merge_sort(i, &mut right, cmp, is_bigint)?;
    let (mut li, mut ri, mut k) = (0usize, 0usize, 0usize);
    while li < left.len() && ri < right.len() {
        if ta_sort_compare(i, &left[li], &right[ri], cmp, is_bigint)? <= 0 {
            vals[k] = left[li].clone();
            li += 1;
        } else {
            vals[k] = right[ri].clone();
            ri += 1;
        }
        k += 1;
    }
    while li < left.len() {
        vals[k] = left[li].clone();
        li += 1;
        k += 1;
    }
    while ri < right.len() {
        vals[k] = right[ri].clone();
        ri += 1;
        k += 1;
    }
    Ok(())
}

/// ToIntegerOrInfinity then clamp to a relative index in `[0, len]` (negatives count from the end).
fn rel_index(n: f64, len: usize) -> usize {
    if n.is_nan() {
        return 0;
    }
    let n = if n.is_infinite() {
        if n > 0.0 {
            len as f64
        } else {
            0.0
        }
    } else {
        n.trunc()
    };
    if n < 0.0 {
        (len as f64 + n).max(0.0) as usize
    } else {
        n.min(len as f64) as usize
    }
}

/// TypedArray-specific implementations of the iteration/search/mutation prototype methods. Returns
/// `None` for methods handled by the generic Array delegation. `len` semantics: captured once here,
/// so out-of-bounds element reads (after a detach/shrink during a callback) surface as `undefined`.
fn ta_native(
    i: &mut Interp,
    this: &Value,
    info: &TaInfo,
    method: &str,
    args: &[Value],
) -> Option<Result<Value, Value>> {
    let info = *info;
    let len = i.ta_len(&info).unwrap_or(0);
    let cb = arg(args, 0);
    let this_arg = arg(args, 1);
    // A predicate/callback method requires its first argument to be callable.
    let require_cb = |i: &mut Interp| {
        if cb.is_callable() {
            // Keep one guarded entry across the algorithm, without changing any live element
            // read or species/Call order (ECMA-262 %TypedArray%.prototype callback methods).
            Ok(crate::callback::Callback::new(cb.clone()))
        } else {
            Err(i.make_error("TypeError", "callback is not a function"))
        }
    };
    match method {
        "at" => Some((|| {
            // Uses [[ArrayLength]], never a `length` Get (an own accessor must not run).
            let rel = ab(i.to_number(&arg(args, 0)))?;
            let rel = if rel.is_nan() { 0.0 } else { rel.trunc() };
            let k = if rel >= 0.0 { rel } else { len as f64 + rel };
            if k < 0.0 || k >= len as f64 {
                return Ok(Value::Undefined);
            }
            Ok(i.ta_read(&info, k as usize))
        })()),
        "forEach" => Some((|| {
            let cb = require_cb(i)?;
            for k in 0..len {
                let v = i.ta_read(&info, k);
                ab(cb.call(i, this_arg.clone(), [v, Value::Num(k as f64), this.clone()]))?;
            }
            Ok(Value::Undefined)
        })()),
        "every" => Some((|| {
            let cb = require_cb(i)?;
            for k in 0..len {
                let v = i.ta_read(&info, k);
                let r = ab(cb.call(i, this_arg.clone(), [v, Value::Num(k as f64), this.clone()]))?;
                if !i.to_boolean(&r) {
                    return Ok(Value::Bool(false));
                }
            }
            Ok(Value::Bool(true))
        })()),
        "some" => Some((|| {
            let cb = require_cb(i)?;
            for k in 0..len {
                let v = i.ta_read(&info, k);
                let r = ab(cb.call(i, this_arg.clone(), [v, Value::Num(k as f64), this.clone()]))?;
                if i.to_boolean(&r) {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        })()),
        "find" | "findIndex" | "findLast" | "findLastIndex" => Some((|| {
            let cb = require_cb(i)?;
            let want_value = method == "find" || method == "findLast";
            let reverse = method == "findLast" || method == "findLastIndex";
            // The spec captures [[ArrayLength]] once, then visits indices in direction order.
            // Keep that order without allocating a temporary index vector for every call.
            let mut remaining = len;
            while remaining > 0 {
                let k = if reverse {
                    remaining - 1
                } else {
                    len - remaining
                };
                remaining -= 1;
                let v = i.ta_read(&info, k);
                let r = ab(cb.call(
                    i,
                    this_arg.clone(),
                    [v.clone(), Value::Num(k as f64), this.clone()],
                ))?;
                if i.to_boolean(&r) {
                    return Ok(if want_value { v } else { Value::Num(k as f64) });
                }
            }
            Ok(if want_value {
                Value::Undefined
            } else {
                Value::Num(-1.0)
            })
        })()),
        "map" => Some((|| {
            let cb = require_cb(i)?;
            let new_ta = ta_species_create(i, this, info.kind, &[Value::Num(len as f64)], true)?;
            let new_info = map_ptr(&new_ta)
                .and_then(|p| i.typed_arrays.get(&p).copied())
                .ok_or_else(|| {
                    i.make_error("TypeError", "map: species did not return a TypedArray")
                })?;
            for k in 0..len {
                let v = i.ta_read(&info, k);
                let mapped =
                    ab(cb.call(i, this_arg.clone(), [v, Value::Num(k as f64), this.clone()]))?;
                ab(i.ta_store(&new_info, k, &mapped))?;
            }
            Ok(new_ta)
        })()),
        "filter" => Some((|| {
            let cb = require_cb(i)?;
            let mut kept: Vec<Value> = Vec::new();
            for k in 0..len {
                let v = i.ta_read(&info, k);
                let r = ab(cb.call(
                    i,
                    this_arg.clone(),
                    [v.clone(), Value::Num(k as f64), this.clone()],
                ))?;
                if i.to_boolean(&r) {
                    kept.push(v);
                }
            }
            let new_ta =
                ta_species_create(i, this, info.kind, &[Value::Num(kept.len() as f64)], true)?;
            let new_info = map_ptr(&new_ta)
                .and_then(|p| i.typed_arrays.get(&p).copied())
                .ok_or_else(|| {
                    i.make_error("TypeError", "filter: species did not return a TypedArray")
                })?;
            for (n, v) in kept.iter().enumerate() {
                ab(i.ta_store(&new_info, n, v))?;
            }
            Ok(new_ta)
        })()),
        "reduce" | "reduceRight" => Some((|| {
            let cb = require_cb(i)?;
            let right = method == "reduceRight";
            let mut acc: Value;
            let mut remaining = len;
            if args.len() >= 2 {
                acc = arg(args, 1);
            } else {
                if len == 0 {
                    return Err(
                        i.make_error("TypeError", "Reduce of empty array with no initial value")
                    );
                }
                let first = if right { len - 1 } else { 0 };
                acc = i.ta_read(&info, first);
                remaining -= 1;
            }
            while remaining > 0 {
                let k = if right {
                    remaining - 1
                } else {
                    len - remaining
                };
                remaining -= 1;
                let v = i.ta_read(&info, k);
                acc = ab(cb.call(
                    i,
                    Value::Undefined,
                    [acc, v, Value::Num(k as f64), this.clone()],
                ))?;
            }
            Ok(acc)
        })()),
        "indexOf" | "lastIndexOf" => Some((|| {
            let search = arg(args, 0);
            if len == 0 {
                return Ok(Value::Num(-1.0));
            }
            let last = method == "lastIndexOf";
            // fromIndex: ToIntegerOrInfinity when *present* (even if explicitly `undefined`, which
            // coerces to 0); the default when absent is 0 (indexOf) / len-1 (lastIndexOf).
            let from = if args.len() >= 2 {
                let n = ab(i.to_number(&arg(args, 1)))?;
                if n.is_nan() {
                    0.0
                } else {
                    n.trunc()
                }
            } else if last {
                (len - 1) as f64
            } else {
                0.0
            };
            // Argument coercion may have detached/resized the buffer; re-derive the live length
            // (indexOf/lastIndexOf only inspect indices that still HasProperty, i.e. in-bounds).
            let curlen = match i.ta_len(&info) {
                Some(l) => l,
                None => return Ok(Value::Num(-1.0)),
            };
            let (mut k, end, step) = if last {
                let k = if from >= 0.0 {
                    from.min((len - 1) as f64) as i64
                } else {
                    (len as f64 + from).max(-1.0) as i64
                };
                (k, -1, -1)
            } else {
                let k = if from >= 0.0 {
                    // ToClampedIndex (ECMA-262 e28783d5, spec.html:5841/42775):
                    // fromIndex >= length, including +Infinity, performs no search. Without
                    // this clamp the equality-terminated loop can step away from its end.
                    from.min(len as f64) as i64
                } else {
                    (len as f64 + from).max(0.0) as i64
                };
                (k, len as i64, 1)
            };
            while k != end {
                if k < 0 || k as usize >= curlen {
                    k += step;
                    continue;
                }
                let v = i.ta_read(&info, k as usize);
                if i.strict_equals(&v, &search) {
                    return Ok(Value::Num(k as f64));
                }
                k += step;
            }
            Ok(Value::Num(-1.0))
        })()),
        "includes" => Some((|| {
            let search = arg(args, 0);
            // A zero-length array is false before ToIntegerOrInfinity(fromIndex) runs any user code.
            if len == 0 {
                return Ok(Value::Bool(false));
            }
            let from = match args.get(1) {
                Some(v) if !matches!(v, Value::Undefined) => {
                    let n = ab(i.to_number(v))?;
                    if n.is_nan() {
                        0.0
                    } else {
                        n.trunc()
                    }
                }
                _ => 0.0,
            };
            // includes iterates the *captured* length and reads out-of-bounds indices as `undefined`
            // (so `includes(undefined)` is true after a mid-coercion detach/shrink).
            let lo = if from < 0.0 {
                (len as f64 + from).max(0.0) as usize
            } else {
                (from as usize).min(len)
            };
            for k in lo..len {
                let v = i.ta_read(&info, k);
                if same_value_zero(&v, &search) {
                    return Ok(Value::Bool(true));
                }
            }
            Ok(Value::Bool(false))
        })()),
        "fill" => Some((|| {
            if i.immutable_buffers.contains(&info.buffer) {
                return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
            }
            // The fill value is coerced to the element type exactly once.
            let value = if info.kind.is_bigint() {
                Value::BigInt(ab(i.to_bigint(&arg(args, 0)))?)
            } else {
                Value::Num(ab(i.to_number(&arg(args, 0)))?)
            };
            let start = rel_index(ab(i.to_number(&arg(args, 1)))?, len);
            let end = match args.get(2) {
                Some(v) if !matches!(v, Value::Undefined) => rel_index(ab(i.to_number(v))?, len),
                _ => len,
            };
            // Re-validate after coercions: a detach/shrink that put the view out of bounds throws.
            let curlen = i
                .ta_len(&info)
                .ok_or_else(|| i.make_error("TypeError", "TypedArray is out of bounds"))?;
            let start = start.min(curlen);
            let end = end.min(curlen);
            if !i.shared_buffers.contains_key(&info.buffer) {
                // After the required coercions and bounds recheck, ordinary-buffer fill has no
                // user-observable operation between element writes. Encode the value once and
                // write the repeated bytes in one range, avoiding a boxed conversion and dirty
                // tracking update for every element (ECMA-262 §23.2.3.9).
                let element = match &value {
                    Value::BigInt(n) if info.kind.is_bigint() => {
                        info.kind.write_bigint(n.to_i128_wrapping())
                    }
                    Value::Num(n) if !info.kind.is_bigint() => info.kind.write(*n),
                    _ => Vec::new(),
                };
                if !element.is_empty() || start == end {
                    let count = end.saturating_sub(start);
                    let total = count.saturating_mul(element.len());
                    // Zeroed storage is free (calloc) and is already the encoding of a zero
                    // fill; any other element is stored once per slot by an indexed copy loop
                    // rather than appended one slot at a time.
                    let mut bytes = vec![0u8; total];
                    if element.iter().any(|&byte| byte != 0) {
                        // Constant-width stores: a runtime-width copy per slot is a memcpy call.
                        match element.len() {
                            1 => bytes.fill(element[0]),
                            2 => {
                                let one = <[u8; 2]>::try_from(&element[..]).unwrap();
                                bytes.as_chunks_mut::<2>().0.fill(one);
                            }
                            4 => {
                                let one = <[u8; 4]>::try_from(&element[..]).unwrap();
                                bytes.as_chunks_mut::<4>().0.fill(one);
                            }
                            8 => {
                                let one = <[u8; 8]>::try_from(&element[..]).unwrap();
                                bytes.as_chunks_mut::<8>().0.fill(one);
                            }
                            width => {
                                for slot in bytes.chunks_exact_mut(width) {
                                    slot.copy_from_slice(&element);
                                }
                            }
                        }
                    }
                    i.ta_write_bytes(&info, start, &bytes);
                    return Ok(this.clone());
                }
            }
            for k in start..end {
                ab(i.ta_store(&info, k, &value))?;
            }
            Ok(this.clone())
        })()),
        "copyWithin" => Some((|| {
            if i.immutable_buffers.contains(&info.buffer) {
                return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
            }
            let to = rel_index(ab(i.to_number(&arg(args, 0)))?, len);
            let from = rel_index(ab(i.to_number(&arg(args, 1)))?, len);
            let final_ = match args.get(2) {
                Some(v) if !matches!(v, Value::Undefined) => rel_index(ab(i.to_number(v))?, len),
                _ => len,
            };
            let count = (final_ as i64 - from as i64).min(len as i64 - to as i64);
            // Re-validate after coercions.
            let curlen = i
                .ta_len(&info)
                .ok_or_else(|| i.make_error("TypeError", "TypedArray is out of bounds"))?;
            let to = to.min(curlen);
            let from = from.min(curlen);
            let count = count
                .min(curlen as i64 - to as i64)
                .min(curlen as i64 - from as i64);
            if count > 0 {
                // TypedArray copyWithin has no user-observable conversion between reads and
                // writes. Snapshot raw bytes before the write so overlapping ranges retain the
                // specified source-before-target behavior and NaN payloads, while avoiding a
                // boxed Value and numeric conversion per element (ECMA-262 §23.2.3.6).
                if let Some(bytes) = i.ta_read_bytes(&info, from, count as usize) {
                    i.ta_write_bytes(&info, to, &bytes);
                    return Ok(this.clone());
                }
                let snap: Vec<Value> = (0..count as usize)
                    .map(|j| i.ta_read(&info, from + j))
                    .collect();
                for (j, v) in snap.iter().enumerate() {
                    ab(i.ta_store(&info, to + j, v))?;
                }
            }
            Ok(this.clone())
        })()),
        "sort" | "toSorted" => Some((|| {
            let cmp = arg(args, 0);
            if !matches!(cmp, Value::Undefined) && !cmp.is_callable() {
                return Err(i.make_error("TypeError", "comparefn must be a function"));
            }
            let in_place = method == "sort";
            if in_place && i.immutable_buffers.contains(&info.buffer) {
                return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
            }
            // Default (no comparator) numeric sort runs natively — the Value-boxed merge sort
            // is far too slow for the quarter-million-element perf tests.
            if matches!(cmp, Value::Undefined) && !info.kind.is_bigint() {
                // Ordinary buffers cannot be resized or observed between the captured read and
                // writes. Keep the source as bytes until sorting, then encode the sorted values
                // into one output range; shared-buffer races retain the conservative path below
                // (ECMA-262 §23.2.3.29).
                if !i.shared_buffers.contains_key(&info.buffer) && i.ta_len(&info) == Some(len) {
                    if let Some(raw) = i.ta_read_bytes(&info, 0, len) {
                        let es = info.kind.elsize();
                        let mut nums: Vec<f64> = raw
                            .chunks_exact(es)
                            .map(|bytes| info.kind.read(bytes))
                            .collect();
                        nums.sort_unstable_by(|a, b| {
                            match (a.is_nan(), b.is_nan()) {
                                (true, true) => std::cmp::Ordering::Equal,
                                (true, false) => std::cmp::Ordering::Greater,
                                (false, true) => std::cmp::Ordering::Less,
                                _ => {
                                    if a == b {
                                        // -0 sorts before +0.
                                        (1.0f64 / a)
                                            .partial_cmp(&(1.0f64 / b))
                                            .unwrap_or(std::cmp::Ordering::Equal)
                                    } else {
                                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                }
                            }
                        });
                        let mut sorted = Vec::with_capacity(raw.len());
                        for n in nums {
                            sorted.extend_from_slice(&info.kind.write(n));
                        }
                        if in_place {
                            i.ta_write_bytes(&info, 0, &sorted);
                            return Ok(this.clone());
                        }
                        let (new_ta, new_info) = ta_create_same(i, info.kind, len)?;
                        i.ta_write_bytes(&new_info, 0, &sorted);
                        return Ok(new_ta);
                    }
                }
                let mut nums: Vec<f64> = (0..len)
                    .map(|k| match i.ta_read(&info, k) {
                        Value::Num(n) => n,
                        _ => f64::NAN,
                    })
                    .collect();
                // TypedArray sort order: numeric ascending, -0 before +0, NaNs last.
                nums.sort_unstable_by(|a, b| {
                    match (a.is_nan(), b.is_nan()) {
                        (true, true) => std::cmp::Ordering::Equal,
                        (true, false) => std::cmp::Ordering::Greater,
                        (false, true) => std::cmp::Ordering::Less,
                        _ => {
                            if a == b {
                                // -0 sorts before +0.
                                (1.0f64 / a)
                                    .partial_cmp(&(1.0f64 / b))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            } else {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            }
                        }
                    }
                });
                if in_place {
                    for (k, n) in nums.iter().enumerate() {
                        ab(i.ta_store(&info, k, &Value::Num(*n)))?;
                    }
                    return Ok(this.clone());
                }
                let (new_ta, new_info) = ta_create_same(i, info.kind, len)?;
                for (k, n) in nums.iter().enumerate() {
                    ab(i.ta_store(&new_info, k, &Value::Num(*n)))?;
                }
                return Ok(new_ta);
            }
            let mut vals: Vec<Value> = (0..len).map(|k| i.ta_read(&info, k)).collect();
            let cmp = cmp
                .is_callable()
                .then(|| crate::callback::Callback::new(cmp));
            ta_merge_sort(i, &mut vals, cmp.as_ref(), info.kind.is_bigint())?;
            if in_place {
                for (k, v) in vals.iter().enumerate() {
                    ab(i.ta_store(&info, k, v))?;
                }
                Ok(this.clone())
            } else {
                let (new_ta, new_info) = ta_create_same(i, info.kind, len)?;
                for (k, v) in vals.iter().enumerate() {
                    ab(i.ta_store(&new_info, k, v))?;
                }
                Ok(new_ta)
            }
        })()),
        "toReversed" => Some((|| {
            let (new_ta, new_info) = ta_create_same(i, info.kind, len)?;
            if i.ta_len(&info) == Some(len) && i.ta_len(&new_info) == Some(len) {
                // TypedArrayCreateSameType gives the result its own buffer. Snapshot the source
                // bytes once and reverse whole elements, preserving NaN payloads and avoiding a
                // boxed read/conversion/write for every element (ECMA-262 §23.2.3.32).
                let es = info.kind.elsize();
                if let Some(source) = i.ta_read_bytes(&info, 0, len) {
                    let mut reversed = Vec::with_capacity(source.len());
                    for k in (0..len).rev() {
                        reversed.extend_from_slice(&source[k * es..(k + 1) * es]);
                    }
                    i.ta_write_bytes(&new_info, 0, &reversed);
                    return Ok(new_ta);
                }
            }
            for k in 0..len {
                let v = i.ta_read(&info, len - 1 - k);
                ab(i.ta_store(&new_info, k, &v))?;
            }
            Ok(new_ta)
        })()),
        "reverse" => Some((|| {
            if i.immutable_buffers.contains(&info.buffer) {
                return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
            }
            if !i.shared_buffers.contains_key(&info.buffer) && i.ta_len(&info) == Some(len) {
                if let Some(source) = i.ta_read_bytes(&info, 0, len) {
                    // Ordinary TypedArray element swaps cannot invoke JavaScript. Snapshot and
                    // rewrite the complete fixed-length byte range, preserving NaN payloads while
                    // avoiding two boxed conversions per element (ECMA-262 §23.2.3.25).
                    let es = info.kind.elsize();
                    let mut reversed = Vec::with_capacity(source.len());
                    for k in (0..len).rev() {
                        reversed.extend_from_slice(&source[k * es..(k + 1) * es]);
                    }
                    i.ta_write_bytes(&info, 0, &reversed);
                    return Ok(this.clone());
                }
            }
            for k in 0..len / 2 {
                let a = i.ta_read(&info, k);
                let b = i.ta_read(&info, len - 1 - k);
                ab(i.ta_store(&info, k, &b))?;
                ab(i.ta_store(&info, len - 1 - k, &a))?;
            }
            Ok(this.clone())
        })()),
        "slice" => Some((|| {
            let start = rel_index(ab(i.to_number(&arg(args, 0)))?, len);
            let end = match args.get(1) {
                Some(v) if !matches!(v, Value::Undefined) => rel_index(ab(i.to_number(v))?, len),
                _ => len,
            };
            let count = end.saturating_sub(start);
            let new_ta = ta_species_create(i, this, info.kind, &[Value::Num(count as f64)], true)?;
            let new_info = map_ptr(&new_ta)
                .and_then(|p| i.typed_arrays.get(&p).copied())
                .ok_or_else(|| i.make_error("TypeError", "slice: species did not return a view"))?;
            if count > 0 {
                // The species constructor ran user code; re-validate the source view.
                let curlen = i.ta_len(&info).ok_or_else(|| {
                    i.make_error("TypeError", "TypedArray source is out of bounds")
                })?;
                if new_info.kind == info.kind {
                    // Same element type: copy bitwise (NaN payloads must survive). The ordinary
                    // species result has a distinct buffer, so snapshot the whole range once;
                    // a custom species may alias the source, in which case the spec's sequential
                    // forward Set order must remain observable and we retain the element loop.
                    let es = info.kind.elsize();
                    let target_len = i.ta_len(&new_info).unwrap_or(0);
                    let n = count.min(curlen.saturating_sub(start)).min(target_len);
                    let source_start = info.offset + start * es;
                    let source_end = source_start + n * es;
                    let target_start = new_info.offset;
                    let target_end = target_start + n * es;
                    let aliases = info.buffer == new_info.buffer
                        && source_start < target_end
                        && target_start < source_end;
                    if !aliases {
                        if let Some(bytes) = i.ta_read_bytes(&info, start, n) {
                            i.ta_write_bytes(&new_info, 0, &bytes);
                        }
                    } else {
                        for j in 0..n {
                            if let Some(bytes) = i.ta_read_bytes(&info, start + j, 1) {
                                debug_assert_eq!(bytes.len(), es);
                                i.ta_write_bytes(&new_info, j, &bytes);
                            }
                        }
                    }
                } else {
                    for j in 0..count {
                        let src_idx = start + j;
                        // Indices past the (possibly shrunk) source stay zero-initialised.
                        if src_idx < curlen {
                            let v = i.ta_read(&info, src_idx);
                            ab(i.ta_store(&new_info, j, &v))?;
                        }
                    }
                }
            }
            Ok(new_ta)
        })()),
        "join" => Some((|| {
            let sep = match args.first() {
                Some(v) if !matches!(v, Value::Undefined) => ab(i.to_string(v))?,
                _ => crate::lstr::LStr::from(","),
            };
            // %TypedArray%.prototype.join delegates its separator/element ordering to
            // ECMA-262 §23.2.3.18 and §23.1.3.18. Keep the separator as an LStr so each element
            // conversion can append directly without an intermediate Rust String.
            let mut out = String::new();
            for k in 0..len {
                if k > 0 {
                    out.push_str(&sep);
                }
                let v = i.ta_read(&info, k);
                if !matches!(v, Value::Undefined | Value::Null) {
                    out.push_str(&ab(i.to_string(&v))?);
                }
            }
            Ok(Value::lstr(out))
        })()),
        _ => None,
    }
}

/// TypedArraySpeciesCreate(O, args): construct a new TypedArray using `O.constructor[@@species]`,
/// falling back to the receiver kind's intrinsic constructor. Validates the result is a TypedArray.
fn ta_species_create(
    i: &mut Interp,
    this: &Value,
    kind: TaKind,
    args: &[Value],
    write: bool,
) -> Result<Value, Value> {
    let default_ctor = ab(i.get_member(&Value::Obj(i.global.clone()), kind.name()))?;
    let ctor = ab(i.get_member(this, "constructor"))?;
    let chosen = if matches!(ctor, Value::Undefined) {
        default_ctor
    } else if matches!(ctor, Value::Obj(_)) {
        let species_key = well_known_key(i, "species").unwrap_or_default();
        let species = ab(i.get_member(&ctor, &species_key))?;
        if matches!(species, Value::Undefined | Value::Null) {
            default_ctor
        } else {
            species
        }
    } else {
        return Err(i.make_error(
            "TypeError",
            "TypedArray constructor property is not an object",
        ));
    };
    if !chosen.is_callable() {
        return Err(i.make_error("TypeError", "TypedArray species is not a constructor"));
    }
    let result = ab(i.construct(chosen, args))?;
    ta_validate_created(i, result, args, write)
}

/// TypedArrayCreate validation of a just-constructed result: it must be a TypedArray, in bounds
/// (not detached/shrunk), and — for a single-Number argument list — at least that long. When
/// `write` is set (the result will be written into), an immutable backing buffer is also rejected.
fn ta_validate_created(
    i: &mut Interp,
    result: Value,
    args: &[Value],
    write: bool,
) -> Result<Value, Value> {
    let new_info = match map_ptr(&result).and_then(|p| i.typed_arrays.get(&p).copied()) {
        Some(info) => info,
        None => {
            return Err(i.make_error(
                "TypeError",
                "TypedArray constructor did not return a TypedArray",
            ));
        }
    };
    let actual_len = match i.ta_len(&new_info) {
        Some(l) => l,
        None => {
            return Err(i.make_error(
                "TypeError",
                "TypedArray destination is detached or out of bounds",
            ));
        }
    };
    if write && i.immutable_buffers.contains(&new_info.buffer) {
        return Err(i.make_error(
            "TypeError",
            "TypedArray destination is backed by an immutable buffer",
        ));
    }
    if args.len() == 1 {
        if let Value::Num(n) = args[0] {
            if (actual_len as f64) < n {
                return Err(i.make_error("TypeError", "TypedArray destination is too small"));
            }
        }
    }
    Ok(result)
}
fn it_array_method(i: &Interp, method: &str) -> Value {
    i.array_proto
        .borrow()
        .props
        .get(method)
        .map(|p| p.value())
        .unwrap_or(Value::Undefined)
}
macro_rules! ta_methods {
    ($($name:literal => $fname:ident),* $(,)?) => {
        $(fn $fname(i: &mut Interp, this: Value, a: &[Value]) -> Result<Value, Value> {
            ta_delegate(i, &this, $name, a)
        })*
        const TA_METHODS: &[(&str, NativeFn)] = &[$(($name, $fname)),*];
    };
}
ta_methods! {
    "forEach" => tad_foreach, "map" => tad_map, "filter" => tad_filter, "reduce" => tad_reduce,
    "reduceRight" => tad_reduceright, "some" => tad_some, "every" => tad_every, "find" => tad_find,
    "findIndex" => tad_findindex, "findLast" => tad_findlast, "findLastIndex" => tad_findlastindex,
    "indexOf" => tad_indexof, "lastIndexOf" => tad_lastindexof, "includes" => tad_includes,
    "join" => tad_join, "fill" => tad_fill, "reverse" => tad_reverse, "sort" => tad_sort,
    "slice" => tad_slice, "at" => tad_at, "keys" => tad_keys, "values" => tad_values,
    "entries" => tad_entries, "copyWithin" => tad_copywithin,
    "toReversed" => tad_toreversed, "toSorted" => tad_tosorted, "with" => tad_with,
}

fn ta_construct(i: &mut Interp, args: &[Value], kind: TaKind) -> Result<Value, Value> {
    if !i.constructing {
        return Err(i.make_error("TypeError", "TypedArray constructor requires 'new'"));
    }
    // ECMA-262 §23.2.5.1: primitive ToIndex precedes AllocateTypedArray; an
    // object's coercions follow it. GetPrototypeFromConstructor is observed ONCE.
    let numeric_length = match args.first() {
        Some(v) if !matches!(v, Value::Obj(_)) => Some(to_index(i, v)?),
        _ => None,
    };
    let proto = match &i.new_target.clone() {
        nt @ Value::Obj(_) => match ab(i.get_member(nt, "prototype"))? {
            Value::Obj(p) => Some(p),
            _ => ctor_realm_proto(i, nt, kind.name())?
                .or_else(|| i.extra_protos.get(kind.name()).cloned()),
        },
        _ => i.extra_protos.get(kind.name()).cloned(),
    };
    let es = kind.elsize();
    // Typed arrays allocate byte buffers, not Vec<Value> array-operation lists.
    // Share ArrayBuffer's byte ceiling; division bounds multiplication safely.
    let checked_bytes = |i: &Interp, len: usize| -> Result<usize, Value> {
        if len > MAX_BUFFER_BYTES / es {
            Err(i.make_error(
                "RangeError",
                "TypedArray backing buffer exceeds allocation limit",
            ))
        } else {
            Ok(len * es)
        }
    };
    let (buf_val, buf_ptr, offset, len, track) = match args.first() {
        None => {
            let (bv, bp) = make_array_buffer(i, 0);
            (bv, bp, 0, 0, false)
        }
        Some(source) if map_ptr(source).is_some_and(|p| i.typed_arrays.contains_key(&p)) => {
            // InitializeTypedArrayFromTypedArray bypasses @@iterator, .length,
            // .constructor and @@species. Same-type copies preserve all raw bits
            // (including NaN payloads), and do not allocate an iterator result or
            // Value per pixel. The resulting buffer is independent and fixed-length.
            let source_info = i.typed_arrays[&map_ptr(source).unwrap()];
            let len = i
                .ta_len(&source_info)
                .ok_or_else(|| i.make_error("TypeError", "TypedArray source is out of bounds"))?;
            let bytes = checked_bytes(i, len)?;
            if kind == source_info.kind {
                let source_bytes = i.ta_read_bytes(&source_info, 0, len).ok_or_else(|| {
                    i.make_error("TypeError", "TypedArray source is out of bounds")
                })?;
                let (bv, bp) = make_array_buffer_from_bytes(i, source_bytes);
                (bv, bp, 0, len, false)
            } else {
                if kind.is_bigint() != source_info.kind.is_bigint() {
                    return Err(i.make_error(
                        "TypeError",
                        "TypedArray content types (BigInt vs Number) differ",
                    ));
                }
                let (bv, bp) = make_array_buffer(i, bytes);
                let target_info = TaInfo {
                    buffer: bp,
                    offset: 0,
                    len,
                    kind,
                    track: false,
                };
                for index in 0..len {
                    let value = i.ta_read(&source_info, index);
                    ab(i.ta_store(&target_info, index, &value))?;
                }
                (bv, bp, 0, len, false)
            }
        }
        // An ArrayBuffer (or SharedArrayBuffer) backing store: identified by the live side table, or
        // by the [[ArrayBufferData]] marker for a detached buffer (still an ArrayBuffer, so it can't
        // fall through to the array-like path — using a detached buffer is a TypeError).
        Some(Value::Obj(o))
            if i.array_buffers.contains_key(&(Rc::as_ptr(o) as usize))
                || o.borrow().props.contains("#\u{0}abMaxByteLength") =>
        {
            let bp = Rc::as_ptr(o) as usize;
            let bv = Value::Obj(o.clone());
            // byteOffset is a ToIndex value and must be a multiple of the element size — both
            // observed BEFORE the detached-buffer check (spec steps 6-7 precede step 9).
            let offset = match args.get(1) {
                Some(v) if !matches!(v, Value::Undefined) => to_index(i, v)?,
                _ => 0,
            };
            if offset % es != 0 {
                return Err(i.make_error(
                    "RangeError",
                    "byteOffset is not aligned to the element size",
                ));
            }
            let explicit = matches!(args.get(2), Some(v) if !matches!(v, Value::Undefined));
            let len_arg = match args.get(2) {
                Some(v) if !matches!(v, Value::Undefined) => Some(to_index(i, v)?),
                _ => None,
            };
            // Coercing byteOffset/length above may have detached the buffer.
            let Some(buflen) = i.array_buffer_byte_length(bp) else {
                return Err(i.make_error("TypeError", "ArrayBuffer is detached"));
            };
            // A resizable ArrayBuffer or a growable SharedArrayBuffer makes an auto-length view
            // length-tracking.
            let resizable = matches!(
                o.borrow().props.get("#\u{0}abResizable").map(|p| p.value()),
                Some(Value::Bool(true))
            );
            let len = match len_arg {
                Some(l) => {
                    if offset > buflen || l > (buflen - offset) / es {
                        return Err(i.make_error("RangeError", "invalid typed array length"));
                    }
                    l
                }
                None => {
                    // The "not a multiple of element size" rule only applies to a fixed-length
                    // buffer; over a resizable buffer the view is length-tracking (auto length).
                    if !resizable && !buflen.is_multiple_of(es) {
                        return Err(i.make_error(
                            "RangeError",
                            "buffer length is not a multiple of the element size",
                        ));
                    }
                    if offset > buflen {
                        return Err(i.make_error("RangeError", "byteOffset is out of bounds"));
                    }
                    buflen.saturating_sub(offset) / es
                }
            };
            (bv, bp, offset, len, !explicit && resizable)
        }
        // A non-object first argument is a length (ToIndex): NaN→0, negative/too-large→RangeError,
        // a Symbol/BigInt → TypeError via ToNumber.
        Some(v) if !matches!(v, Value::Obj(_)) => {
            let len = numeric_length.expect("primitive ToIndex already performed");
            let bytes = checked_bytes(i, len)?;
            let (bv, bp) = make_array_buffer(i, bytes);
            (bv, bp, 0, len, false)
        }
        Some(other) => {
            // TypedArray / array-like / iterable object: copy element values into a fresh buffer.
            // GetMethod(@@iterator): a non-callable non-nullish value is a TypeError, and a
            // callable one is used (so a patched Array.prototype[@@iterator] is honored).
            let iter_key = well_known_key(i, "iterator");
            let iter_method = match &iter_key {
                Some(k) => ab(i.get_member(other, k))?,
                None => Value::Undefined,
            };
            if !matches!(iter_method, Value::Undefined | Value::Null) && !iter_method.is_callable()
            {
                return Err(i.make_error("TypeError", "@@iterator is not callable"));
            }
            if iter_method.is_callable() {
                // InitializeTypedArrayFromList is observably identical to intrinsic Array
                // iteration when every indexed value is an own data property. Avoid allocating
                // one iterator-result object and one decimal property key per element in that
                // common case; patched iterators, holes, and accessors retain the full algorithm.
                let items = if is_native_function(&iter_method, nf_array_values)
                    && intrinsic_array_iterator_is_unmodified(i)
                {
                    match dense_array_snapshot(i, other) {
                        Some(items) => items,
                        None => ab(i.iterate_with(other, iter_method))?,
                    }
                } else {
                    ab(i.iterate_with(other, iter_method))?
                };
                let len = items.len();
                let bytes = checked_bytes(i, len)?;
                let (bv, bp) = make_array_buffer(i, bytes);
                let info = TaInfo {
                    buffer: bp,
                    offset: 0,
                    len,
                    kind,
                    track: false,
                };
                for (idx, item) in items.iter().enumerate() {
                    ab(i.ta_store(&info, idx, item))?;
                }
                (bv, bp, 0, len, false)
            } else {
                let n = ab(i.to_length(other.as_obj().unwrap()))?;
                let bytes = checked_bytes(i, n)?;
                let (bv, bp) = make_array_buffer(i, bytes);
                let info = TaInfo {
                    buffer: bp,
                    offset: 0,
                    len: n,
                    kind,
                    track: false,
                };
                // Array-like Get/convert/store operations interleave per element;
                // unlike iterable input, later getters must see earlier conversions.
                for k in 0..n {
                    let own = other
                        .as_obj()
                        .and_then(|object| i.fast_get_elem(object, k as f64));
                    let value = match own {
                        Some(value) => value,
                        None => ab(i.get_member(other, &k.to_string()))?,
                    };
                    ab(i.ta_store(&info, k, &value))?;
                }
                (bv, bp, 0, n, false)
            }
        }
    };
    // The instance prototype comes from new.target.prototype when it's an object (subclassing /
    // Reflect.construct), else the intrinsic %TypedArray.prototype% for this element type in
    // new.target's realm (GetPrototypeFromConstructor).
    let obj = Object::new(proto);
    let p = Rc::as_ptr(&obj) as usize;
    i.gc_pin(&obj);
    obj.borrow().ic_plain.set(false);
    i.typed_arrays.insert(
        p,
        TaInfo {
            buffer: buf_ptr,
            offset,
            len,
            kind,
            track,
        },
    );
    // length / byteLength / byteOffset / buffer / BYTES_PER_ELEMENT are inherited accessors+constants,
    // not own properties: length/byteLength/byteOffset are computed in get_member, the buffer object
    // is kept in a side table, and BYTES_PER_ELEMENT lives on the prototype.
    let _ = es;
    i.ta_buffer.insert(p, buf_val);
    i.register_native_typed_array(&obj, i.typed_arrays[&p]);
    Ok(Value::Obj(obj))
}

fn ta_set(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    let ptr = map_ptr(&this).ok_or_else(|| i.make_error("TypeError", "set on non-TypedArray"))?;
    let info = *i
        .typed_arrays
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "set on non-TypedArray"))?;
    // An immutable target buffer can't be written — verified before any argument is coerced.
    if i.immutable_buffers.contains(&info.buffer) {
        return Err(i.make_error("TypeError", "Cannot write to an immutable ArrayBuffer"));
    }
    // targetOffset = ToIntegerOrInfinity(offset); a negative offset is a RangeError.
    let offset_n = match arg(args, 1) {
        Value::Undefined => 0.0,
        v => {
            let n = ab(i.to_number(&v))?;
            if n.is_nan() {
                0.0
            } else {
                n.trunc()
            }
        }
    };
    if offset_n < 0.0 {
        return Err(i.make_error("RangeError", "TypedArray.prototype.set offset is negative"));
    }
    // Re-validate the target after offset coercion: an out-of-bounds (detached or shrunk-resizable)
    // target is a TypeError, and an immutable target buffer can't be written.
    let target_len = i
        .ta_len(&info)
        .ok_or_else(|| i.make_error("TypeError", "TypedArray target is out of bounds"))?;
    let source = arg(args, 0);
    let src_info = source
        .as_obj()
        .and_then(|o| i.typed_arrays.get(&(Rc::as_ptr(o) as usize)).copied());
    if let Some(src_info) = src_info {
        // SetTypedArrayFromTypedArray: the source view must be in bounds, the target long enough,
        // and the content types must match (mixing BigInt and Number is a TypeError).
        let src_len = i
            .ta_len(&src_info)
            .ok_or_else(|| i.make_error("TypeError", "TypedArray source is out of bounds"))?;
        if offset_n + src_len as f64 > target_len as f64 {
            return Err(i.make_error("RangeError", "source is too large for the target at offset"));
        }
        if src_info.kind.is_bigint() != info.kind.is_bigint() {
            return Err(i.make_error(
                "TypeError",
                "TypedArray content types (BigInt vs Number) differ",
            ));
        }
        let offset = offset_n as usize;
        if src_info.kind == info.kind {
            // SetTypedArrayFromTypedArray copies bytes when element types match (ECMA-262
            // §23.2.3.26.2), including overlapping views and NaN payloads. Snapshot the exact
            // byte range once, then write it, avoiding one boxed `Value` and conversion per
            // element while retaining the required overlap behavior.
            if let Some(bytes) = i.ta_read_bytes(&src_info, 0, src_len) {
                i.ta_write_bytes(&info, offset, &bytes);
                return Ok(Value::Undefined);
            }
        } else if !i.shared_buffers.contains_key(&src_info.buffer)
            && !i.shared_buffers.contains_key(&info.buffer)
        {
            // Different numeric element types still have no user-observable work between source
            // reads and target writes. Snapshot the source encodings, convert each element once,
            // and publish one target range; aliases therefore retain the specified source-before-
            // target ordering without allocating a Value for every element (ECMA-262 §23.2.3.26.2).
            if let Some(source) = i.ta_read_bytes(&src_info, 0, src_len) {
                let source_es = src_info.kind.elsize();
                let target_es = info.kind.elsize();
                let mut converted = Vec::with_capacity(src_len * target_es);
                for bytes in source.chunks_exact(source_es) {
                    if info.kind.is_bigint() {
                        converted.extend_from_slice(
                            &info.kind.write_bigint(src_info.kind.read_bigint(bytes)),
                        );
                    } else {
                        converted.extend_from_slice(&info.kind.write(src_info.kind.read(bytes)));
                    }
                }
                i.ta_write_bytes(&info, offset, &converted);
                return Ok(Value::Undefined);
            }
        }
        // Snapshot the source first so an overlapping same-buffer copy reads pre-write values.
        let vals: Vec<Value> = (0..src_len).map(|k| i.ta_read(&src_info, k)).collect();
        for (k, v) in vals.iter().enumerate() {
            ab(i.ta_store(&info, offset + k, v))?;
        }
        return Ok(Value::Undefined);
    }
    // SetTypedArrayFromArrayLike: ToObject(source), read its length, bounds-check, then copy element
    // by element (coercing each value; a throwing getter leaves earlier elements written).
    let src = to_object_arg(i, source, "TypedArray.prototype.set")?;
    let src_len = ab(i.to_length(&src))?;
    if offset_n + src_len as f64 > target_len as f64 {
        return Err(i.make_error("RangeError", "source is too large for the target at offset"));
    }
    let offset = offset_n as usize;
    // ECMA-262 SetTypedArrayFromArrayLike (snapshot e28783d5fc9d): preserve length and
    // bounds/error ordering above; only an effect-free numeric source may batch the copy.
    if i.ta_set_numeric_array(&info, offset, &src, src_len) {
        return Ok(Value::Undefined);
    }
    let src = Value::Obj(src);
    for k in 0..src_len {
        let item = ab(i.get_member(&src, &k.to_string()))?;
        ab(i.ta_store(&info, offset + k, &item))?;
    }
    Ok(Value::Undefined)
}

fn ta_subarray(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    let ptr =
        map_ptr(&this).ok_or_else(|| i.make_error("TypeError", "subarray on non-TypedArray"))?;
    let info = *i
        .typed_arrays
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "subarray on non-TypedArray"))?;
    let es = info.kind.elsize();
    // srcLength is the length at entry (0 if the view is already out of bounds); the begin/end
    // coercions below clamp against it even if they resize the buffer.
    let len = i.ta_len(&info).unwrap_or(0);
    let begin = rel_index(ab(i.to_number(&arg(args, 0)))?, len);
    let new_offset = info.offset + begin * es;
    let buf_val = ab(i.get_member(&this, "buffer"))?;
    // A length-tracking source subarrayed with no explicit end stays length-tracking: pass only
    // «buffer, byteOffset» so the result auto-sizes to the buffer (TypedArraySpeciesCreate honors
    // a subclass `@@species`).
    let end_absent = matches!(args.get(1), None | Some(Value::Undefined));
    if info.track && end_absent {
        return ta_species_create(
            i,
            &this,
            info.kind,
            &[buf_val, Value::Num(new_offset as f64)],
            false,
        );
    }
    let end = match arg(args, 1) {
        Value::Undefined => len,
        v => rel_index(ab(i.to_number(&v))?, len),
    };
    let new_len = end.saturating_sub(begin);
    ta_species_create(
        i,
        &this,
        info.kind,
        &[
            buf_val,
            Value::Num(new_offset as f64),
            Value::Num(new_len as f64),
        ],
        false,
    )
}

macro_rules! ta_ctor {
    ($name:ident, $kind:expr) => {
        fn $name(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
            ta_construct(i, a, $kind)
        }
    };
}
ta_ctor!(ta_ctor_i8, TaKind::I8);
ta_ctor!(ta_ctor_u8, TaKind::U8);
ta_ctor!(ta_ctor_u8c, TaKind::U8Clamped);
ta_ctor!(ta_ctor_i16, TaKind::I16);
ta_ctor!(ta_ctor_u16, TaKind::U16);
ta_ctor!(ta_ctor_i32, TaKind::I32);
ta_ctor!(ta_ctor_u32, TaKind::U32);
ta_ctor!(ta_ctor_f16, TaKind::F16);
ta_ctor!(ta_ctor_f32, TaKind::F32);
ta_ctor!(ta_ctor_f64, TaKind::F64);
ta_ctor!(ta_ctor_i64, TaKind::I64);
ta_ctor!(ta_ctor_u64, TaKind::U64);

pub(super) fn install_typed_arrays(it: &mut Interp) {
    install_array_buffer(it);

    // Shared %TypedArray% prototype. Each method brand-checks the receiver is a TypedArray, then
    // delegates to the generic Array method (which works through get_member/array_length/set_member).
    let ta_proto = Object::new(Some(it.object_proto.clone()));
    for (name, f) in TA_METHODS {
        // Each method's `length` matches its required-parameter count.
        let len = match *name {
            "copyWithin" | "slice" | "subarray" | "with" => 2,
            "keys" | "values" | "entries" | "toReversed" | "reverse" => 0,
            _ => 1,
        };
        it.def_method(&ta_proto, name, len, *f);
    }
    // %TypedArray%.prototype.toString is the very same function object as %Array.prototype.toString%.
    if let Some(p) = it.array_proto.borrow().props.get("toString").cloned() {
        ta_proto.borrow_mut().props.insert("toString", p);
    }
    // %TypedArray%.prototype[@@iterator] is the same function object as its own `values`.
    if let Some(sym) = it.iterator_sym.clone() {
        let k = Interp::sym_key(&sym);
        let values_prop = ta_proto.borrow().props.get("values").cloned();
        if let Some(p) = values_prop {
            ta_proto.borrow_mut().props.insert(k, p);
        }
    }
    it.def_method(&ta_proto, "set", 1, ta_set);
    it.def_method(&ta_proto, "subarray", 2, ta_subarray);
    it.def_method(&ta_proto, "toLocaleString", 0, |i, this, a| {
        // ValidateTypedArray then Invoke `toLocaleString` on each element, joining with ",". The
        // length is captured once; an out-of-bounds/detached receiver throws, but an element that
        // reads back `undefined` after a mid-iteration shrink is skipped.
        let info = map_ptr(&this).and_then(|p| i.typed_arrays.get(&p).copied());
        let info = info.ok_or_else(|| i.make_error("TypeError", "not a TypedArray"))?;
        let len = i
            .ta_len(&info)
            .ok_or_else(|| i.make_error("TypeError", "TypedArray is out of bounds"))?;
        let mut out = String::new();
        for k in 0..len {
            if k > 0 {
                out.push(',');
            }
            let v = i.ta_read(&info, k);
            if matches!(v, Value::Undefined | Value::Null) {
                continue;
            }
            let tls = ab(i.get_member(&v, "toLocaleString"))?;
            if !tls.is_callable() {
                return Err(i.make_error("TypeError", "toLocaleString is not a function"));
            }
            let s = ab(i.call(tls, v, &[arg(a, 0), arg(a, 1)]))?;
            out.push_str(&ab(i.to_string(&s))?);
        }
        Ok(Value::lstr(out))
    });
    // length / byteLength / byteOffset / buffer are accessor getters on %TypedArray.prototype% that
    // brand-check the receiver (calling one on a non-TypedArray is a TypeError).
    for &(accessor, getter) in ta_accessor_getters() {
        let name = accessor.name();
        let g = it.make_native(&format!("get {name}"), 0, getter);
        ta_proto.borrow_mut().props.insert(
            name,
            Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
        );
    }
    // %TypedArray.prototype%[@@toStringTag]: a getter returning the kind name (e.g. "Int8Array")
    // for a TypedArray receiver, else undefined (no throw, no setter).
    if let Some(key) = well_known_key(it, "toStringTag") {
        let g = it.make_native("get [Symbol.toStringTag]", 0, |i, this, _| {
            Ok(match map_ptr(&this).and_then(|p| i.typed_arrays.get(&p)) {
                Some(info) => Value::lstr(info.kind.name()),
                None => Value::Undefined,
            })
        });
        ta_proto.borrow_mut().props.insert(
            key,
            Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
        );
    }

    // The abstract %TypedArray% intrinsic: each concrete TypedArray constructor inherits from it,
    // and its `.prototype` is the shared `ta_proto`. Tests reach it via Object.getPrototypeOf(Int8Array).
    let ta_ctor = it.make_native("TypedArray", 0, |i, t, _a| {
        // Abstract: a direct `new %TypedArray%()` throws; subclass `super()` (this is set) is allowed.
        if matches!(t, Value::Undefined) {
            return Err(i.make_error(
                "TypeError",
                "Abstract class TypedArray not directly constructable",
            ));
        }
        Ok(t)
    });
    ta_ctor.borrow_mut().is_constructor = true;
    ta_ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(ta_proto.clone()), false, false, false),
    );
    ta_proto.borrow_mut().props.insert(
        "constructor",
        Property::builtin(Value::Obj(ta_ctor.clone())),
    );
    it.def_method(&ta_ctor, "of", 0, ta_of);
    it.def_method(&ta_ctor, "from", 1, ta_from);
    install_species(it, &ta_ctor);

    let kinds: [(TaKind, NativeFn); 12] = [
        (TaKind::I8, ta_ctor_i8),
        (TaKind::U8, ta_ctor_u8),
        (TaKind::U8Clamped, ta_ctor_u8c),
        (TaKind::I16, ta_ctor_i16),
        (TaKind::U16, ta_ctor_u16),
        (TaKind::I32, ta_ctor_i32),
        (TaKind::U32, ta_ctor_u32),
        (TaKind::F16, ta_ctor_f16),
        (TaKind::F32, ta_ctor_f32),
        (TaKind::F64, ta_ctor_f64),
        (TaKind::I64, ta_ctor_i64),
        (TaKind::U64, ta_ctor_u64),
    ];
    for (kind, ctor_fn) in kinds {
        let proto = Object::new(Some(ta_proto.clone()));
        Rc::make_mut(&mut it.extra_protos).insert(kind.name(), proto.clone());
        // BYTES_PER_ELEMENT is a non-writable, non-enumerable, non-configurable constant.
        proto.borrow_mut().props.insert(
            "BYTES_PER_ELEMENT",
            Property::data(Value::Num(kind.elsize() as f64), false, false, false),
        );
        let ctor = it.make_native(kind.name(), 3, ctor_fn);
        ctor.borrow_mut().proto = Some(ta_ctor.clone()); // [[Prototype]] is %TypedArray%
        ctor.borrow_mut().props.insert(
            "prototype",
            Property::data(Value::Obj(proto.clone()), false, false, false),
        );
        proto
            .borrow_mut()
            .props
            .insert("constructor", Property::builtin(Value::Obj(ctor.clone())));
        ctor.borrow_mut().props.insert(
            "BYTES_PER_ELEMENT",
            Property::data(Value::Num(kind.elsize() as f64), false, false, false),
        );
        // from/of are inherited from %TypedArray% (they construct through `this`), not own here.
        set_builtin(&it.global, kind.name(), Value::Obj(ctor));
    }
    install_uint8_base64(it);
}

// --- Uint8Array base64 / hex (the Stage-3 proposal) -------------------------------------------

const B64_STD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_URL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

fn b64_encode(bytes: &[u8], url: bool, pad: bool) -> String {
    let alpha = if url { B64_URL } else { B64_STD };
    // RFC 4648 emits four characters per complete three-byte group.  Reserve the exact
    // output size up front so a large Uint8Array does not repeatedly grow the result.
    let complete = bytes.len() / 3;
    let remainder = bytes.len() % 3;
    let tail = match (remainder, pad) {
        (0, _) => 0,
        (1, true) => 4,
        (1, false) => 2,
        (2, true) => 4,
        (2, false) => 3,
        _ => unreachable!(),
    };
    let mut out = String::with_capacity(complete.saturating_mul(4).saturating_add(tail));
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(alpha[(n >> 18 & 63) as usize] as char);
        out.push(alpha[(n >> 12 & 63) as usize] as char);
        match chunk.len() {
            1 => {
                if pad {
                    out.push_str("==");
                }
            }
            2 => {
                out.push(alpha[(n >> 6 & 63) as usize] as char);
                if pad {
                    out.push('=');
                }
            }
            _ => {
                out.push(alpha[(n >> 6 & 63) as usize] as char);
                out.push(alpha[(n & 63) as usize] as char);
            }
        }
    }
    out
}

#[inline]
fn b64_value(c: char, url: bool) -> Option<u8> {
    match c {
        'A'..='Z' => Some(c as u8 - b'A'),
        'a'..='z' => Some(c as u8 - b'a' + 26),
        '0'..='9' => Some(c as u8 - b'0' + 52),
        '+' if !url => Some(62),
        '-' if url => Some(62),
        '/' if !url => Some(63),
        '_' if url => Some(63),
        _ => None,
    }
}

#[inline]
fn hex_value(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// FromBase64: decode at most `max_len` bytes honoring `handling` (loose / strict /
/// stop-before-partial). Returns `(read, bytes)` where `read` is the code units consumed through
/// the last fully decoded chunk; `Err(())` is a syntax error.
fn b64_decode_spec(s: &str, url: bool, handling: &str, max_len: usize) -> (usize, Vec<u8>, bool) {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let is_ws = |c: char| matches!(c, '\t' | '\n' | '\x0c' | '\r' | ' ');
    let decode_chunk = |chunk: &[u8], throw_extra: bool, bytes: &mut Vec<u8>| -> Result<(), ()> {
        let mut n = 0u32;
        for (idx, &v) in chunk.iter().enumerate() {
            n |= (v as u32) << (18 - 6 * idx);
        }
        match chunk.len() {
            2 => {
                if throw_extra && n & 0xFFFF != 0 {
                    return Err(());
                }
                bytes.push((n >> 16) as u8);
            }
            3 => {
                if throw_extra && n & 0xFF != 0 {
                    return Err(());
                }
                bytes.extend([(n >> 16) as u8, (n >> 8) as u8]);
            }
            _ => bytes.extend([(n >> 16) as u8, (n >> 8) as u8, n as u8]),
        }
        Ok(())
    };
    if max_len == 0 {
        return (0, Vec::new(), false);
    }
    // Four input characters produce at most three bytes.  Whitespace and padding can only
    // reduce this, so this bound avoids growth reallocations without over-reserving by max_len.
    let capacity = (len / 4).saturating_mul(3).saturating_add(3).min(max_len);
    let mut bytes = Vec::with_capacity(capacity);
    let (mut read, mut index) = (0usize, 0usize);
    let mut chunk: Vec<u8> = Vec::with_capacity(4);
    loop {
        while index < len && is_ws(chars[index]) {
            index += 1;
        }
        if index == len {
            if !chunk.is_empty() {
                match handling {
                    "stop-before-partial" => return (read, bytes, false),
                    "loose" => {
                        if chunk.len() == 1 {
                            return (read, bytes, true);
                        }
                        if decode_chunk(&chunk, false, &mut bytes).is_err() {
                            return (read, bytes, true);
                        }
                    }
                    _ => return (read, bytes, true), // strict: a partial chunk must be padded
                }
            }
            return (len, bytes, false);
        }
        let c = chars[index];
        index += 1;
        if c == '=' {
            if chunk.len() < 2 {
                return (read, bytes, true);
            }
            while index < len && is_ws(chars[index]) {
                index += 1;
            }
            if chunk.len() == 2 {
                if index == len {
                    if handling == "stop-before-partial" {
                        return (read, bytes, false);
                    }
                    return (read, bytes, true);
                }
                if chars[index] != '=' {
                    return (read, bytes, true);
                }
                index += 1;
                while index < len && is_ws(chars[index]) {
                    index += 1;
                }
            }
            if index < len {
                return (read, bytes, true); // trailing characters after padding
            }
            if decode_chunk(&chunk, handling == "strict", &mut bytes).is_err() {
                return (read, bytes, true);
            }
            return (len, bytes, false);
        }
        let Some(v) = b64_value(c, url) else {
            return (read, bytes, true);
        };
        let remaining = max_len - bytes.len();
        if (remaining == 1 && chunk.len() == 2) || (remaining == 2 && chunk.len() == 3) {
            return (read, bytes, false); // the next chunk wouldn't fit: stop before it
        }
        chunk.push(v);
        if chunk.len() == 4 {
            if decode_chunk(&chunk, false, &mut bytes).is_err() {
                return (read, bytes, true);
            }
            chunk.clear();
            read = index;
            if bytes.len() == max_len {
                return (read, bytes, false);
            }
        }
    }
}
/// FromHex: decode at most `max_len` bytes; `read` is the number of code units consumed.
fn hex_decode_spec(s: &str, max_len: usize) -> (usize, Vec<u8>, bool) {
    // FromHex accepts only ASCII hexits.  Keep the UTF-16-aware character path below for
    // unusual engine strings, but avoid materializing a Vec<char> for the overwhelmingly common
    // ASCII case (ECMA-262 §23.3.3.8).
    if s.is_ascii() {
        let input = s.as_bytes();
        let mut out = Vec::with_capacity((input.len() / 2).min(max_len));
        if !input.len().is_multiple_of(2) {
            return (0, out, true);
        }
        let mut index = 0;
        while index < input.len() && out.len() < max_len {
            let (Some(hi), Some(lo)) = (hex_value(input[index]), hex_value(input[index + 1]))
            else {
                return (index, out, true);
            };
            out.push((hi << 4) | lo);
            index += 2;
        }
        return (index, out, false);
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::with_capacity((chars.len() / 2).min(max_len));
    if !chars.len().is_multiple_of(2) {
        return (0, out, true);
    }
    let mut index = 0;
    while index < chars.len() && out.len() < max_len {
        let (Some(hi), Some(lo)) = (chars[index].to_digit(16), chars[index + 1].to_digit(16))
        else {
            return (index, out, true);
        };
        out.push((hi * 16 + lo) as u8);
        index += 2;
    }
    (index, out, false)
}

/// ValidateUint8Array, then GetUint8ArrayBytes: the bytes the view covers now (a length-tracking
/// view follows its buffer; a SharedArrayBuffer is read from its shared Data Block). A detached
/// buffer or a view out of bounds of a shrunk resizable buffer is a TypeError
/// (ValidateTypedArrayBounds).
fn u8_bytes(i: &mut Interp, this: &Value) -> Result<Vec<u8>, Value> {
    let info = u8_validate(i, this)?;
    i.ta_len(&info)
        .and_then(|len| i.ta_read_bytes(&info, 0, len))
        .ok_or_else(|| i.make_error("TypeError", "Uint8Array is detached or out of bounds"))
}
/// The new Uint8Array of `Uint8Array.fromBase64`/`fromHex`: AllocateTypedArray("Uint8Array",
/// %Uint8Array%, %Uint8Array.prototype%, length) and its bytes (proposal-arraybuffer-base64).
fn make_u8array(i: &mut Interp, bytes: Vec<u8>) -> Result<Value, Value> {
    Ok(typed_array_from_bytes(i, TaKind::U8, bytes).0)
}
fn b64_option_url(i: &mut Interp, opts: &Value) -> Result<bool, Value> {
    if let Value::Obj(_) = opts {
        let a = ab(i.get_member(opts, "alphabet"))?;
        match a {
            Value::Undefined => Ok(false),
            Value::Str(s) if &*s == "base64" => Ok(false),
            Value::Str(s) if &*s == "base64url" => Ok(true),
            _ => Err(i.make_error("TypeError", "alphabet must be 'base64' or 'base64url'")),
        }
    } else if matches!(opts, Value::Undefined) {
        Ok(false)
    } else {
        Err(i.make_error("TypeError", "options must be an object"))
    }
}

/// Read `{alphabet, lastChunkHandling}` for the base64 decode methods.
fn b64_decode_options(i: &mut Interp, opts: &Value) -> Result<(bool, Rc<str>), Value> {
    let url = b64_option_url(i, opts)?;
    let handling = if let Value::Obj(_) = opts {
        match ab(i.get_member(opts, "lastChunkHandling"))? {
            Value::Undefined => crate::lstr::LStr::from("loose"),
            Value::Str(s) if matches!(&*s, "loose" | "strict" | "stop-before-partial") => s,
            _ => {
                return Err(i.make_error(
                    "TypeError",
                    "lastChunkHandling must be 'loose', 'strict', or 'stop-before-partial'",
                ));
            }
        }
    } else {
        crate::lstr::LStr::from("loose")
    };
    Ok((url, (&handling).into()))
}

pub(super) fn install_uint8_base64(it: &mut Interp) {
    let proto = it.extra_protos.get("Uint8Array").cloned().unwrap();
    let ctor = match it
        .global
        .borrow()
        .props
        .get("Uint8Array")
        .map(|p| p.value())
    {
        Some(Value::Obj(c)) => c,
        _ => return,
    };
    it.def_method(&proto, "toHex", 0, |i, this, _| {
        let bytes = u8_bytes(i, &this)?;
        let mut s = String::with_capacity(bytes.len() * 2);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for b in bytes {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0x0f) as usize] as char);
        }
        Ok(Value::lstr(s))
    });
    it.def_method(&proto, "toBase64", 0, |i, this, a| {
        // Receiver validation, then options, then the (detachment-sensitive) byte read.
        u8_validate(i, &this)?;
        let url = b64_option_url(i, &arg(a, 0))?;
        let omit_padding = if let Value::Obj(_) = arg(a, 0) {
            let op = ab(i.get_member(&arg(a, 0), "omitPadding"))?;
            i.to_boolean(&op)
        } else {
            false
        };
        let bytes = u8_bytes(i, &this)?;
        Ok(Value::lstr(b64_encode(&bytes, url, !omit_padding)))
    });
    it.def_method(&ctor, "fromHex", 1, |i, _t, a| {
        let s = match arg(a, 0) {
            Value::Str(s) => s,
            _ => return Err(i.make_error("TypeError", "fromHex requires a string")),
        };
        let (_, bytes, err) = hex_decode_spec(&s, usize::MAX);
        if err {
            return Err(i.make_error("SyntaxError", "invalid hex string"));
        }
        make_u8array(i, bytes)
    });
    it.def_method(&ctor, "fromBase64", 1, |i, _t, a| {
        let s = match arg(a, 0) {
            Value::Str(s) => s,
            _ => return Err(i.make_error("TypeError", "fromBase64 requires a string")),
        };
        let (url, handling) = b64_decode_options(i, &arg(a, 1))?;
        let (_, bytes, err) = b64_decode_spec(&s, url, &handling, usize::MAX);
        if err {
            return Err(i.make_error("SyntaxError", "invalid base64 string"));
        }
        make_u8array(i, bytes)
    });
    it.def_method(&proto, "setFromHex", 1, |i, this, a| {
        let info = u8_validate_writable(i, &this)?;
        let s = match arg(a, 0) {
            Value::Str(s) => s,
            _ => return Err(i.make_error("TypeError", "setFromHex requires a string")),
        };
        let max = i
            .ta_len(&info)
            .ok_or_else(|| i.make_error("TypeError", "detached buffer"))?;
        // Valid chunks decoded before an error are still written, then the SyntaxError surfaces.
        let (read, bytes, err) = hex_decode_spec(&s, max);
        let result = u8_set_bytes(i, &info, read, &bytes)?;
        if err {
            return Err(i.make_error("SyntaxError", "invalid hex string"));
        }
        Ok(result)
    });
    it.def_method(&proto, "setFromBase64", 1, |i, this, a| {
        let info = u8_validate_writable(i, &this)?;
        let s = match arg(a, 0) {
            Value::Str(s) => s,
            _ => return Err(i.make_error("TypeError", "setFromBase64 requires a string")),
        };
        let (url, handling) = b64_decode_options(i, &arg(a, 1))?;
        let max = i
            .ta_len(&info)
            .ok_or_else(|| i.make_error("TypeError", "detached buffer"))?;
        // Valid chunks decoded before an error are still written, then the SyntaxError surfaces.
        let (read, bytes, err) = b64_decode_spec(&s, url, &handling, max);
        let result = u8_set_bytes(i, &info, read, &bytes)?;
        if err {
            return Err(i.make_error("SyntaxError", "invalid base64 string"));
        }
        Ok(result)
    });
}

/// ValidateUint8Array for the mutating methods: also rejects an immutable backing buffer
/// (before any argument is read).
fn u8_validate_writable(i: &mut Interp, this: &Value) -> Result<TaInfo, Value> {
    let info = u8_validate(i, this)?;
    if i.immutable_buffers.contains(&info.buffer) {
        return Err(i.make_error(
            "TypeError",
            "cannot write into a view over an immutable ArrayBuffer",
        ));
    }
    Ok(info)
}

/// ValidateUint8Array: the receiver must be a Uint8Array (detachment is checked separately).
fn u8_validate(i: &mut Interp, this: &Value) -> Result<TaInfo, Value> {
    let ptr = map_ptr(this).ok_or_else(|| i.make_error("TypeError", "not a Uint8Array"))?;
    let info = *i
        .typed_arrays
        .get(&ptr)
        .ok_or_else(|| i.make_error("TypeError", "not a Uint8Array"))?;
    if !matches!(info.kind, TaKind::U8) {
        return Err(i.make_error("TypeError", "requires a Uint8Array"));
    }
    Ok(info)
}

/// SetUint8ArrayBytes + result object: write decoded `bytes` into the view, report `{read, written}`.
fn u8_set_bytes(i: &mut Interp, info: &TaInfo, read: usize, bytes: &[u8]) -> Result<Value, Value> {
    if !bytes.is_empty() && i.immutable_buffers.contains(&info.buffer) {
        return Err(i.make_error(
            "TypeError",
            "cannot write into a view over an immutable ArrayBuffer",
        ));
    }
    // The caller decoded at most the view's current length, and decoding runs no author code.
    i.ta_write_bytes(info, 0, bytes);
    let result = i.new_object();
    set_data(&result, "read", Value::Num(read as f64));
    set_data(&result, "written", Value::Num(bytes.len() as f64));
    Ok(Value::Obj(result))
}

fn ta_of(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    if !is_constructor_value(&this) {
        return Err(i.make_error("TypeError", "TypedArray.of requires a constructor receiver"));
    }
    let len_args = [Value::Num(args.len() as f64)];
    let ta = ab(i.construct(this, &len_args))?;
    let ta = ta_validate_created(i, ta, &len_args, true)?;
    for (k, v) in args.iter().enumerate() {
        ab(i.set_member(&ta, &k.to_string(), v.clone()))?;
    }
    Ok(ta)
}
fn ta_from(i: &mut Interp, this: Value, args: &[Value]) -> Result<Value, Value> {
    if !is_constructor_value(&this) {
        return Err(i.make_error(
            "TypeError",
            "TypedArray.from requires a constructor receiver",
        ));
    }
    let mapfn = arg(args, 1);
    if !matches!(mapfn, Value::Undefined) && !mapfn.is_callable() {
        return Err(i.make_error("TypeError", "mapfn is not callable"));
    }
    let mapfn = mapfn
        .is_callable()
        .then(|| crate::callback::Callback::new(mapfn));
    let this_arg = arg(args, 2);
    let source = arg(args, 0);
    // GetMethod(source, @@iterator): reading @@iterator off undefined/null throws, and a throwing
    // getter propagates. A callable result means the source is iterated; otherwise it is array-like.
    let iter_key = well_known_key(i, "iterator");
    let using_iter = match &iter_key {
        Some(k) => ab(i.get_member(&source, k))?,
        None => Value::Undefined,
    };
    if !matches!(using_iter, Value::Undefined | Value::Null) && !using_iter.is_callable() {
        return Err(i.make_error("TypeError", "@@iterator is not callable"));
    }
    if using_iter.is_callable() {
        // Iterable source: collect all values (reusing the already-fetched @@iterator), then
        // construct the target and fill it.
        let items = ab(i.iterate_with(&source, using_iter))?;
        let len_args = [Value::Num(items.len() as f64)];
        let ta = ab(i.construct(this, &len_args))?;
        let ta = ta_validate_created(i, ta, &len_args, true)?;
        for (k, v) in items.into_iter().enumerate() {
            let val = if let Some(mapfn) = &mapfn {
                ab(mapfn.call(i, this_arg.clone(), [v, Value::Num(k as f64)]))?
            } else {
                v
            };
            ab(i.set_member(&ta, &k.to_string(), val))?;
        }
        return Ok(ta);
    }
    // Array-like source: ToObject, read its length, construct the target *before* visiting any
    // element, then read and set each element in turn.
    let src = Value::Obj(to_object_arg(i, source, "TypedArray.from")?);
    let len = ab(i.to_length(src.as_obj().unwrap()))?;
    let len_args = [Value::Num(len as f64)];
    let ta = ab(i.construct(this, &len_args))?;
    let ta = ta_validate_created(i, ta, &len_args, true)?;
    for k in 0..len {
        let v = ab(i.get_member(&src, &k.to_string()))?;
        let val = if let Some(mapfn) = &mapfn {
            ab(mapfn.call(i, this_arg.clone(), [v, Value::Num(k as f64)]))?
        } else {
            v
        };
        ab(i.set_member(&ta, &k.to_string(), val))?;
    }
    Ok(ta)
}

// ---------------------------------------------------------------------------------------------
// Date  (treated entirely as UTC — getTimezoneOffset is 0 — which is enough for most test262 Date
// tests, which use explicit timestamps. Calendar math uses the days-from-civil algorithm.)
// ---------------------------------------------------------------------------------------------
