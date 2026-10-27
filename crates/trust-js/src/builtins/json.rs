//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::value::ValueView;

pub(super) fn install_json(it: &mut Interp) {
    let j = it.new_object();
    it.def_method(&j, "stringify", 3, |i, _t, args| {
        let value = arg(args, 0);
        let replacer = arg(args, 1);
        // The replacer is either a function, or an array PropertyList of keys (strings/numbers).
        let opts = if replacer.is_callable() {
            JsonOpts {
                func: Some(replacer),
                keys: None,
            }
        } else if json_is_array(i, &replacer)? {
            let len = match &(replacer).view() {
                ValueView::Obj(o) if proxy_pair(i, &replacer).is_none() => i.array_length(o),
                ValueView::Obj(o) => ab(i.to_length(o))?,
                _ => 0,
            };
            let mut list: Vec<crate::atom::PropKey> = Vec::new();
            // JSON.stringify's PropertyList keeps the first occurrence of each String while
            // preserving input order (ECMA-262 §25.5.4.1). Hash the already-coerced keys so a
            // large replacer array does not repeatedly rescan the prefix of `list`.
            let mut list_seen = crate::fasthash::FastSet::default();
            for k in 0..len {
                let item = ab(i.get_member_key(&replacer, &k.to_string()))?;
                // String/Number primitives and their wrappers contribute a key via ToString.
                let key = match &(item).view() {
                    ValueView::Str(s) => Some(s.to_string()),
                    ValueView::Num(n) => Some(i.num_to_str(*n)),
                    ValueView::Obj(o)
                        if matches!(o.borrow().exotic, Exotic::StrWrap(_) | Exotic::NumWrap(_)) =>
                    {
                        Some(ab(i.to_string(&item))?.to_string())
                    }
                    _ => None,
                };
                if let Some(key) = key {
                    if list_seen.insert(key.clone()) {
                        list.push(crate::atom::PropKey::from_text(
                            &crate::value::encode_owned_string_key(key),
                        ));
                    }
                }
            }
            JsonOpts {
                func: None,
                keys: Some(list),
            }
        } else {
            JsonOpts {
                func: None,
                keys: None,
            }
        };
        // The `space` argument: a Number/String wrapper is unwrapped first; then a Number becomes
        // that many spaces (clamped 0..10), a String its first 10 code units, else no indentation.
        let mut space = arg(args, 2);
        if let ValueView::Obj(o) = &(space).view() {
            // Classify first: the conversions may run JavaScript, which needs `o` unborrowed.
            let (number, string) = {
                let exotic = &o.borrow().exotic;
                (
                    matches!(exotic, Exotic::NumWrap(_)),
                    matches!(exotic, Exotic::StrWrap(_)),
                )
            };
            if number {
                space = Value::Num(ab(i.to_number(&space))?);
            } else if string {
                space = Value::Str(ab(i.to_string(&space))?);
            }
        }
        let gap = match (space).view() {
            ValueView::Num(n) => {
                let n = if n.is_nan() { 0.0 } else { n.trunc() };
                " ".repeat(n.clamp(0.0, 10.0) as usize)
            }
            ValueView::Str(s) if s.ascii_hint() => {
                let text = s.text();
                text[..text.len().min(10)].to_owned()
            }
            // JSON.stringify takes the first ten UTF-16 code units, including a lone
            // surrogate when that boundary splits an astral character. Decode only
            // this bounded prefix instead of materializing the full indentation string.
            ValueView::Str(s) => {
                crate::jstr::from_units(&s.units().iter().take(10).collect::<Vec<_>>())
            }
            _ => String::new(),
        };
        // SerializeJSONProperty starts from a wrapper holder `{ "": value }`; only a replacer
        // function can observe it (as its receiver).
        let holder = if opts.func.is_some() {
            let wrapper = i.new_object();
            set_data(&wrapper, "", value);
            Value::Obj(wrapper)
        } else {
            Value::Undefined
        };
        let mut serializer = Serializer {
            opts: &opts,
            gap: &gap,
            indent: String::new(),
            stack: Vec::new(),
            deep_stack: crate::fasthash::FastSet::default(),
            out: JsonOut::with_capacity(64),
        };
        if json_serialize_property(i, &mut serializer, &holder, JsonKey::Name(""), value)? {
            Ok(Value::Str(serializer.out.finish()))
        } else {
            Ok(Value::Undefined)
        }
    });
    it.def_method(&j, "parse", 2, |i, _t, args| {
        let text = ab(i.to_string(&arg(args, 0)))?;
        // JSON grammar is ASCII-delimited: parse a one-byte string's bytes and a two-byte
        // string's code units as stored.
        let input = match text.units() {
            crate::lstr::Units::Ascii(bytes) => JsonInput::Bytes(bytes),
            crate::lstr::Units::Utf16(units) => JsonInput::Units(units),
        };
        let mut pos = 0;
        let reviver = arg(args, 1);
        // A callable reviver walks the result via InternalizeJSONProperty from a `{ "": v }` root,
        // recording primitive source spans for its `context` argument.
        if reviver.is_callable() {
            let (v, record) = json_parse_recorded(i, input, &mut pos)?;
            // A reviver can replace a later property before it is visited. Its originally
            // parsed value still participates in SameValue/source matching, even when only
            // this Rust-owned record retains it across the reviver's safepoints (I1).
            let record = crate::rooting::Rooted::new(record);
            json_skip_ws(input, &mut pos);
            if pos != input.len() {
                return Err(i.make_error("SyntaxError", "Unexpected non-whitespace after JSON"));
            }
            let root = i.new_object();
            set_data(&root, "", v);
            return internalize_json_property(i, &Value::Obj(root), "", &reviver, Some(&record));
        }
        let v = json_parse_value(i, input, &mut pos)?;
        json_skip_ws(input, &mut pos);
        if pos != input.len() {
            return Err(i.make_error("SyntaxError", "Unexpected non-whitespace after JSON"));
        }
        Ok(v)
    });
    it.def_method(&j, "rawJSON", 1, |i, _t, args| {
        let text = ab(i.to_string(&arg(args, 0)))?.to_string();
        let chars = (!text.is_ascii()).then(|| text.chars().collect::<Vec<_>>());
        let input = match chars.as_deref() {
            Some(chars) => JsonInput::Chars(chars),
            None => JsonInput::Bytes(text.as_bytes()),
        };
        if input.is_empty() {
            return Err(i.make_error("SyntaxError", "JSON.rawJSON: empty string"));
        }
        let is_ws = |c: char| matches!(c, '\t' | '\n' | '\r' | ' ');
        if is_ws(input.get(0).unwrap()) || is_ws(input.get(input.len() - 1).unwrap()) {
            return Err(i.make_error("SyntaxError", "JSON.rawJSON: leading/trailing whitespace"));
        }
        if matches!(input.get(0), Some('{') | Some('[')) {
            return Err(i.make_error("SyntaxError", "JSON.rawJSON value must be a primitive"));
        }
        // Validate it is exactly one JSON value.
        let mut pos = 0;
        json_parse_value(i, input, &mut pos)?;
        json_skip_ws(input, &mut pos);
        if pos != input.len() {
            return Err(i.make_error("SyntaxError", "JSON.rawJSON: invalid JSON text"));
        }
        let o = i.new_object();
        o.borrow_mut().proto = None;
        set_data(&o, "rawJSON", Value::lstr(text.clone()));
        set_internal(&o, "\u{0}raw_json", Value::lstr(text));
        i.freeze_object(&Value::Obj(o));
        Ok(Value::Obj(o))
    });
    it.def_method(&j, "isRawJSON", 1, |_i, _t, args| {
        Ok(Value::Bool(
            matches!((arg(args, 0)).view(), ValueView::Obj(o) if o.borrow().internal_slots().contains("\u{0}raw_json")),
        ))
    });
    set_to_string_tag(it, &j, "JSON");
    set_builtin(&it.global, "JSON", Value::Obj(j));
}

type JsonOut = crate::lstr::Builder;

/// One ASCII unit of QuoteJSONString: the quote, backslash and control characters escaped.
#[inline]
fn push_json_ascii(out: &mut JsonOut, b: u8) {
    match b {
        b'"' => out.push_str("\\\""),
        b'\\' => out.push_str("\\\\"),
        b'\n' => out.push_str("\\n"),
        b'\r' => out.push_str("\\r"),
        b'\t' => out.push_str("\\t"),
        0x08 => out.push_str("\\b"),
        0x0c => out.push_str("\\f"),
        b if b < 0x20 => push_json_unicode_escape(out, u16::from(b)),
        b => out.push_ascii(b),
    }
}

/// Append QuoteJSONString (ECMA-262 §25.5.2.3) of a string's code units to `out`: each unit as
/// itself, the quote, backslash and control characters escaped, and a surrogate outside a pair
/// as its \u escape (well-formed JSON.stringify).
fn json_quote_units(out: &mut JsonOut, units: crate::lstr::Units<'_>) {
    out.push_ascii(b'"');
    match units {
        crate::lstr::Units::Ascii(bytes) => {
            for &b in bytes {
                push_json_ascii(out, b);
            }
        }
        crate::lstr::Units::Utf16(units) => {
            let mut index = 0;
            while index < units.len() {
                let unit = units[index];
                index += 1;
                match unit {
                    0..=0x7f => push_json_ascii(out, unit as u8),
                    0xD800..=0xDBFF
                        if units
                            .get(index)
                            .is_some_and(|next| (0xDC00..0xE000).contains(next)) =>
                    {
                        out.push_unit(unit);
                        out.push_unit(units[index]);
                        index += 1;
                    }
                    0xD800..=0xDFFF => push_json_unicode_escape(out, unit),
                    _ => out.push_unit(unit),
                }
            }
        }
    }
    out.push_ascii(b'"');
}

/// [`json_quote_units`] of engine text (a property key).
fn json_quote_text(out: &mut JsonOut, s: &str) {
    if s.is_ascii() {
        json_quote_units(out, crate::lstr::Units::Ascii(s.as_bytes()));
    } else {
        json_quote_units(out, crate::lstr::Units::Utf16(&crate::jstr::units(s)));
    }
}

/// QuoteJSONString(s) of engine text, appended to a Rust string (tests).
#[cfg(test)]
fn json_quote_into(out: &mut String, s: &str) {
    let mut quoted = JsonOut::with_capacity(16);
    json_quote_text(&mut quoted, s);
    out.push_str(&quoted.finish().text());
}

#[inline]
fn push_json_unicode_escape(out: &mut JsonOut, unit: u16) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push_str("\\u");
    for shift in [12, 8, 4, 0] {
        out.push_ascii(HEX[((unit >> shift) & 0x0f) as usize]);
    }
}

/// JSON.stringify options: an optional function replacer and/or an array PropertyList of keys.
struct JsonOpts {
    func: Option<Value>,
    keys: Option<Vec<crate::atom::PropKey>>,
}

/// [[Delete]] for the JSON reviver: trap-aware, discarding the boolean status (per `Perform ?`).
fn json_delete_prop(i: &mut Interp, holder: &Value, key: &str) -> Result<(), Value> {
    if let Some((target, handler)) = proxy_pair(i, holder) {
        ab(i.proxy_delete(target, handler, key))?;
        return Ok(());
    }
    if let ValueView::Obj(o) = &(holder).view() {
        let configurable = o
            .borrow()
            .props
            .get(key)
            .map(|p| p.configurable())
            .unwrap_or(true);
        if configurable {
            o.borrow_mut().props.remove(key);
        }
    }
    Ok(())
}

/// CreateDataProperty for the JSON reviver: a { value, writable, enumerable, configurable: true }
/// data property, trap-aware for proxy holders.
fn json_create_data_prop(i: &mut Interp, holder: &Value, key: &str, v: Value) -> Result<(), Value> {
    // CreateDataProperty defines a fully-permissive data descriptor via [[DefineOwnProperty]], which
    // validates against any existing (e.g. non-configurable) property; a false result is not an error.
    let desc = i.new_object();
    set_data(&desc, "value", v);
    set_data(&desc, "writable", Value::Bool(true));
    set_data(&desc, "enumerable", Value::Bool(true));
    set_data(&desc, "configurable", Value::Bool(true));
    if let Some((target, handler)) = proxy_pair(i, holder) {
        ab(proxy_define_property(
            i,
            &target,
            &handler,
            key,
            &Value::Obj(desc),
        ))?;
        return Ok(());
    }
    if let ValueView::Obj(o) = &(holder).view() {
        ab(define_own_property(i, o, key, &Value::Obj(desc)))?;
    }
    Ok(())
}

/// InternalizeJSONProperty: recursively walk the parsed value, applying the reviver bottom-up. The
/// optional record carries primitive source text for the reviver's `context.source`.
fn internalize_json_property(
    i: &mut Interp,
    holder: &Value,
    name: &str,
    reviver: &Value,
    record: Option<&JsonRecord>,
) -> Result<Value, Value> {
    let val = ab(i.get_member_key(holder, name))?;
    if matches!((val).view(), ValueView::Obj(_)) {
        if json_is_array(i, &val)? {
            let len = match val.as_obj() {
                Some(o) => ab(i.to_length(&o))?,
                None => 0,
            };
            for idx in 0..len {
                let k = idx.to_string();
                let child = match record {
                    Some(JsonRecord::Arr(elems)) => elems.get(idx),
                    _ => None,
                };
                let new_el = nested(i, |i| {
                    internalize_json_property(i, &val, &k, reviver, child)
                })?;
                if matches!((new_el).view(), ValueView::Undefined) {
                    json_delete_prop(i, &val, &k)?;
                } else {
                    json_create_data_prop(i, &val, &k, new_el)?;
                }
            }
        } else {
            let keys: Vec<String> = if proxy_pair(i, &val).is_some() {
                proxy_enum_string_keys(i, &val)?
                    .iter()
                    .filter_map(|k| match &(k).view() {
                        ValueView::Str(s) => {
                            Some(crate::value::encode_owned_string_key(s.to_string()))
                        }
                        _ => None,
                    })
                    .collect()
            } else if let Some(info) = val.as_obj().and_then(|o| ta_info(i, &o)) {
                // A TypedArray holder (a reviver may graft one in) walks its integer indices.
                (0..i.ta_len(&info).unwrap_or(0))
                    .map(|k| k.to_string())
                    .collect()
            } else {
                val.as_obj()
                    .map(|o| {
                        ordered_enum_keys(&o)
                            .iter()
                            .map(|k| k.text().to_string())
                            .collect()
                    })
                    .unwrap_or_default()
            };
            for k in keys {
                let child = match record {
                    Some(JsonRecord::Obj(entries)) => entries
                        .iter()
                        .find(|(key, _)| key == crate::value::decode_string_key(&k))
                        .map(|(_, r)| r),
                    _ => None,
                };
                let new_el = nested(i, |i| {
                    internalize_json_property(i, &val, &k, reviver, child)
                })?;
                if matches!((new_el).view(), ValueView::Undefined) {
                    json_delete_prop(i, &val, &k)?;
                } else {
                    json_create_data_prop(i, &val, &k, new_el)?;
                }
            }
        }
    }
    // The `context` argument: a primitive leaf whose value is still the originally-parsed one (i.e.
    // not forward-modified by the reviver) exposes its exact source text.
    let context = i.new_object();
    if let Some(JsonRecord::Prim(src, parsed)) = record {
        if !matches!((val).view(), ValueView::Obj(_)) && same_value(&val, parsed) {
            set_data(&context, "source", Value::lstr(src.clone()));
        }
    }
    ab(i.call(
        *reviver,
        *holder,
        &[
            crate::value::string_key_value(name),
            val,
            Value::Obj(context),
        ],
    ))
}

/// One nested level of JSON.parse, its reviver walk or JSON.stringify, which recurse once per
/// level of the text or value: on a fresh native stack segment when this one runs low, so deep
/// nesting either completes (node parses a million levels) or throws the RangeError a deep
/// JavaScript recursion would, instead of overflowing the stack.
fn nested<T>(
    i: &mut Interp,
    level: impl FnOnce(&mut Interp) -> Result<T, Value>,
) -> Result<T, Value> {
    match crate::interpreter::with_data_stack(|| level(&mut *i)) {
        Some(result) => result,
        None => Err(i.make_error("RangeError", "Maximum call stack size exceeded")),
    }
}

/// JSON.stringify's state (ECMA-262 §25.5.2.1): the options, the indentation, the objects being
/// serialized (for the cycle check) and the one buffer the whole result is written to.
struct Serializer<'a> {
    opts: &'a JsonOpts,
    gap: &'a str,
    indent: String,
    /// The objects being serialized, outermost first. The cycle check scans the first
    /// [`JSON_STACK_SCAN`] of them and looks the rest up in `deep_stack`.
    stack: Vec<usize>,
    deep_stack: crate::fasthash::FastSet<usize>,
    out: JsonOut,
}

/// Nesting levels whose cycle check is a linear scan: shallow values (almost all of them) skip
/// hashing, while deep nesting stays linear overall.
const JSON_STACK_SCAN: usize = 32;

/// A property key for SerializeJSONProperty, made into a String only when toJSON or a replacer
/// function observes it.
#[derive(Clone, Copy)]
enum JsonKey<'k> {
    Name(&'k str),
    Index(usize),
}

impl JsonKey<'_> {
    fn value(self) -> Value {
        match self {
            JsonKey::Name(name) => crate::value::string_key_value(name),
            JsonKey::Index(index) => Value::lstr(index.to_string()),
        }
    }
}

/// Whether Get on `object` for a string key has no side effects to observe when it finds an own
/// data property: an ordinary object (no Proxy, exotic host or namespace methods; `ic_plain`) that
/// is not a mapped arguments object.
fn json_plain_get(i: &Interp, object: &Gc) -> bool {
    object.borrow().ic_plain.get()
        && (i.mapped_arguments.is_empty()
            || !i
                .mapped_arguments
                .contains_key(&(Gc::as_ptr(object) as usize)))
}

/// Get(holder, key) for SerializeJSONProperty: an own data property of a plain object is read
/// directly; anything else (accessors, absent keys, exotic objects) runs the full [[Get]].
fn json_get(
    i: &mut Interp,
    holder: &Value,
    object: &Gc,
    key: crate::atom::PropKey,
) -> Result<Value, Value> {
    if json_plain_get(i, object) {
        if let Some(property) = object.borrow().props.get_key(key) {
            if !property.accessor() {
                return Ok(property.value());
            }
        }
    }
    ab(i.get_member_key(holder, &key.text()))
}

/// [`json_get`] for an array element.
fn json_get_index(
    i: &mut Interp,
    holder: &Value,
    object: &Gc,
    index: usize,
) -> Result<Value, Value> {
    if let (true, Ok(slot)) = (json_plain_get(i, object), u32::try_from(index)) {
        if let Some(property) = object.borrow().props.get_index(slot) {
            if !property.accessor() {
                return Ok(property.value());
            }
        }
    }
    ab(i.get_member_key(holder, &index.to_string()))
}

/// GetV(value, "toJSON") for SerializeJSONProperty step 2. Most values have no toJSON anywhere
/// on an ordinary prototype chain; walk such a chain directly, and leave accessors and exotic
/// objects to the full [[Get]] (whose result is the same, including getter calls).
fn json_to_json_method(i: &mut Interp, value: &Value) -> Result<Value, Value> {
    if let ValueView::Obj(start) = &(value).view() {
        let mut current = *start;
        loop {
            let next = {
                let object = current.borrow();
                if !object.ic_plain.get() {
                    break;
                }
                match object.props.get("toJSON") {
                    Some(property) if !property.accessor() => return Ok(property.value()),
                    Some(_) => break,
                    None => object.proto,
                }
            };
            match next {
                Some(proto) => current = proto,
                None => return Ok(Value::Undefined),
            }
        }
    }
    ab(i.get_member_key(value, "toJSON"))
}

/// SerializeJSONProperty (ECMA-262 §25.5.2.2) for a `value` already read from `holder[key]`,
/// appended to the serializer's buffer. Returns false, appending nothing, when the property
/// serializes to undefined.
fn json_serialize_property(
    i: &mut Interp,
    s: &mut Serializer,
    holder: &Value,
    key: JsonKey,
    mut value: Value,
) -> Result<bool, Value> {
    if matches!((value).view(), ValueView::Obj(_) | ValueView::BigInt(_)) {
        let to_json = json_to_json_method(i, &value)?;
        if to_json.is_callable() {
            value = ab(i.call(to_json, value, &[key.value()]))?;
        }
    }
    if let Some(func) = &s.opts.func {
        value = ab(i.call(*func, *holder, &[key.value(), value]))?;
    }
    if let ValueView::Obj(o) = &(value).view() {
        // A JSON.rawJSON object (always created with a null prototype) serializes as its stored
        // raw text, verbatim.
        let raw = {
            let object = o.borrow();
            match object.proto {
                None => object
                    .internal_slots()
                    .get("\u{0}raw_json")
                    .map(|property| property.value()),
                Some(_) => None,
            }
        };
        if let Some(ValueView::Str(raw)) = (raw).map(|__value| (__value).view()) {
            s.out.push_units(raw.units());
            return Ok(true);
        }
        // A primitive-wrapper object re-coerces through ToNumber/ToString (so an overridden
        // valueOf/toString is observed); booleans read the wrapped datum directly.
        let wrapped = match o.borrow().exotic {
            Exotic::NumWrap(_) => Some(0),
            Exotic::StrWrap(_) => Some(1),
            Exotic::BoolWrap(b) => Some(2 + b as u8),
            Exotic::BigIntWrap(_) => {
                return Err(i.make_error("TypeError", "Do not know how to serialize a BigInt"));
            }
            _ => None,
        };
        match wrapped {
            Some(0) => value = Value::Num(ab(i.to_number(&value))?),
            Some(1) => value = Value::Str(ab(i.to_string(&value))?),
            Some(b) => value = Value::Bool(b == 3),
            None => {}
        }
    }
    match &(value).view() {
        ValueView::Undefined | ValueView::Empty | ValueView::Sym(_) => return Ok(false),
        ValueView::Null => s.out.push_str("null"),
        ValueView::Bool(b) => s.out.push_str(if *b { "true" } else { "false" }),
        ValueView::Num(n) if !n.is_finite() => s.out.push_str("null"),
        ValueView::Num(n) if n.fract() == 0.0 && n.abs() < 1e15 => {
            // Integral and below 10^21, so Number::toString prints its digits (-0 prints "0").
            use std::fmt::Write;
            let _ = write!(s.out, "{}", *n as i64);
        }
        ValueView::Num(n) => s.out.push_str(&i.num_to_str(*n)),
        // JSON.stringify of a BigInt throws (matches the spec).
        ValueView::BigInt(_) => {
            return Err(i.make_error("TypeError", "Do not know how to serialize a BigInt"));
        }
        ValueView::Str(text) => {
            // Escaping only lengthens the text.
            check_string_length(i, s.out.len().saturating_add(text.unit_len()))?;
            json_quote_units(&mut s.out, text.units());
        }
        ValueView::Obj(o) => {
            if !matches!(o.borrow().call, Callable::None) {
                return Ok(false); // functions are omitted
            }
            let o = *o;
            if json_is_array(i, &value)? {
                nested(i, |i| json_serialize_array(i, s, &value, &o))?;
            } else {
                nested(i, |i| json_serialize_object(i, s, &value, &o))?;
            }
        }
    }
    check_string_length(i, s.out.len())?;
    Ok(true)
}

/// The start of SerializeJSONObject/Array: the cycle check and the deeper indentation.
fn json_enter(i: &mut Interp, s: &mut Serializer, object: &Gc) -> Result<usize, Value> {
    let ptr = Gc::as_ptr(object) as usize;
    let shallow = &s.stack[..s.stack.len().min(JSON_STACK_SCAN)];
    if shallow.contains(&ptr) || s.deep_stack.contains(&ptr) {
        return Err(i.make_error("TypeError", "Converting circular structure to JSON"));
    }
    if s.stack.len() >= JSON_STACK_SCAN {
        s.deep_stack.insert(ptr);
    }
    s.stack.push(ptr);
    let stepback = s.indent.len();
    s.indent.push_str(s.gap);
    Ok(stepback)
}

/// The end of SerializeJSONObject/Array: the closing bracket, on its own line when the gap
/// indents a non-empty body.
fn json_leave(s: &mut Serializer, object: &Gc, stepback: usize, any: bool, close: char) {
    s.indent.truncate(stepback);
    if any && !s.gap.is_empty() {
        s.out.push('\n');
        s.out.push_str(&s.indent);
    }
    s.out.push(close);
    let ptr = s.stack.pop();
    debug_assert_eq!(ptr, Some(Gc::as_ptr(object) as usize));
    if s.stack.len() >= JSON_STACK_SCAN {
        s.deep_stack.remove(&(Gc::as_ptr(object) as usize));
    }
}

/// The separator before a member or element: a comma after the first, and with a gap a newline
/// and the current indentation.
fn json_separator(s: &mut Serializer, first: bool) {
    if !first {
        s.out.push(',');
    }
    if !s.gap.is_empty() {
        s.out.push('\n');
        s.out.push_str(&s.indent);
    }
}

/// SerializeJSONObject (ECMA-262 §25.5.2.5).
fn json_serialize_object(
    i: &mut Interp,
    s: &mut Serializer,
    value: &Value,
    object: &Gc,
) -> Result<(), Value> {
    let stepback = json_enter(i, s, object)?;
    // K: the replacer's PropertyList, or EnumerableOwnProperties(value, key).
    let keys: Vec<crate::atom::PropKey> = match &s.opts.keys {
        Some(list) => list.clone(),
        None if proxy_pair(i, value).is_some() => proxy_enum_string_keys(i, value)?
            .iter()
            .filter_map(|key| match &(key).view() {
                ValueView::Str(key) => Some(crate::atom::PropKey::from_text(
                    &crate::value::encode_lstr_key(key),
                )),
                _ => None,
            })
            .collect(),
        None if object.borrow().ic_plain.get() => {
            // An ordinary object's [[OwnPropertyKeys]] is its own key order; enumerability is
            // its stored attribute. A record of enumerable named data properties needs no
            // filtering.
            let object = object.borrow();
            if let Some(keys) = object.props.named_data_keys() {
                keys.to_vec()
            } else {
                let mut keys = object.props.ordered_keys();
                keys.retain(|&key| {
                    !key.is_symbol()
                        && object
                            .props
                            .get_key(key)
                            .is_some_and(|property| property.enumerable())
                });
                keys
            }
        }
        None => {
            let mut keys = Vec::new();
            let own_keys = ordinary_own_keys_ordered(i, object)?;
            for key in own_keys.into_iter().filter(|key| !Interp::is_sym_key(key)) {
                let enumerable = if ab(i.host_indexed_own_value(object, &key))?.is_some() {
                    crate::value::canonical_index(&key).is_some()
                } else {
                    object
                        .borrow()
                        .props
                        .get(&key)
                        .is_some_and(|property| property.enumerable())
                };
                if enumerable {
                    keys.push(crate::atom::PropKey::from_text(&key));
                }
            }
            keys
        }
    };
    s.out.push('{');
    let mut any = false;
    for &key in &keys {
        let member_start = s.out.len();
        json_separator(s, !any);
        let text = key.text();
        json_quote_text(&mut s.out, crate::value::decode_string_key(&text));
        s.out.push(':');
        if !s.gap.is_empty() {
            s.out.push(' ');
        }
        let member = json_get(i, value, object, key)?;
        if json_serialize_property(i, s, value, JsonKey::Name(&text), member)? {
            any = true;
        } else {
            s.out.truncate(member_start);
        }
    }
    json_leave(s, object, stepback, any, '}');
    Ok(())
}

/// SerializeJSONArray (ECMA-262 §25.5.2.6).
fn json_serialize_array(
    i: &mut Interp,
    s: &mut Serializer,
    value: &Value,
    object: &Gc,
) -> Result<(), Value> {
    let stepback = json_enter(i, s, object)?;
    let len = if proxy_pair(i, value).is_some() {
        ab(i.to_length(object))?
    } else {
        i.array_length(object)
    };
    s.out.push('[');
    for index in 0..len {
        json_separator(s, index == 0);
        let element = json_get_index(i, value, object, index)?;
        if !json_serialize_property(i, s, value, JsonKey::Index(index), element)? {
            s.out.push_str("null");
        }
        // Every element adds at least "null", so a huge sparse length stops here.
        check_string_length(i, s.out.len())?;
    }
    json_leave(s, object, stepback, len > 0, ']');
    Ok(())
}

#[derive(Clone, Copy)]
enum JsonInput<'a> {
    Bytes(&'a [u8]),
    Chars(&'a [char]),
    /// A two-byte engine string's code units, as stored.
    Units(&'a [u16]),
}

impl JsonInput<'_> {
    #[inline]
    fn len(self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Chars(chars) => chars.len(),
            Self::Units(units) => units.len(),
        }
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// The character at `index`: a unit of two-byte input is its own character, a surrogate
    /// its smuggled scalar (JSON's grammar outside strings is ASCII).
    #[inline]
    fn get(self, index: usize) -> Option<char> {
        match self {
            Self::Bytes(bytes) => bytes.get(index).copied().map(char::from),
            Self::Chars(chars) => chars.get(index).copied(),
            Self::Units(units) => units.get(index).map(|&unit| {
                char::from_u32(u32::from(unit)).unwrap_or_else(|| crate::jstr::smuggle(unit))
            }),
        }
    }

    fn to_string(self, start: usize, end: usize) -> String {
        match self {
            Self::Bytes(bytes) => String::from_utf8(bytes[start..end].to_vec())
                .expect("ASCII JSON input is valid UTF-8"),
            Self::Chars(chars) => chars[start..end].iter().collect(),
            Self::Units(units) => crate::jstr::from_units(&units[start..end]),
        }
    }

    fn parse_number(self, start: usize, end: usize) -> Result<f64, ()> {
        match self {
            Self::Bytes(bytes) => std::str::from_utf8(&bytes[start..end])
                .map_err(|_| ())?
                .parse::<f64>()
                .map_err(|_| ()),
            Self::Chars(chars) => chars[start..end]
                .iter()
                .collect::<String>()
                .parse::<f64>()
                .map_err(|_| ()),
            Self::Units(units) => crate::jstr::from_units(&units[start..end])
                .parse::<f64>()
                .map_err(|_| ()),
        }
    }
}

/// A parsed JSON string: engine text from byte or character input, code units from two-byte
/// input, which becomes a string value without conversion.
enum JsonStr {
    Text(String),
    Units(Vec<u16>),
}

impl JsonStr {
    fn into_value(self) -> Value {
        match self {
            JsonStr::Text(text) => Value::lstr(text),
            JsonStr::Units(units) => Value::Str(crate::lstr::LStr::from_utf16(&units)),
        }
    }

    /// As engine text, for a property key.
    fn into_key(self) -> String {
        match self {
            JsonStr::Text(text) => text,
            JsonStr::Units(units) => crate::jstr::from_units(&units),
        }
    }
}

/// ParseJSON (ECMA-262 #sec-ParseJSON) of `text` in the engine's string representation: the value
/// of a JSON module (ParseJSONModule), which must not depend on the author-replaceable
/// `JSON.parse`. Invalid JSON is a SyntaxError.
pub(crate) fn parse_json(i: &mut Interp, text: &str) -> Result<Value, Value> {
    let chars = (!text.is_ascii()).then(|| text.chars().collect::<Vec<_>>());
    let input = match chars.as_deref() {
        Some(chars) => JsonInput::Chars(chars),
        None => JsonInput::Bytes(text.as_bytes()),
    };
    let mut pos = 0;
    let value = json_parse_value(i, input, &mut pos)?;
    json_skip_ws(input, &mut pos);
    if pos != input.len() {
        return Err(i.make_error("SyntaxError", "Unexpected non-whitespace after JSON"));
    }
    Ok(value)
}

fn json_skip_ws(input: JsonInput<'_>, pos: &mut usize) {
    while *pos < input.len() && matches!(input.get(*pos), Some(' ' | '\t' | '\n' | '\r')) {
        *pos += 1;
    }
}

fn json_parse_value(i: &mut Interp, input: JsonInput<'_>, pos: &mut usize) -> Result<Value, Value> {
    json_skip_ws(input, pos);
    let c = input
        .get(*pos)
        .ok_or_else(|| i.make_error("SyntaxError", "Unexpected end of JSON input"))?;
    match c {
        '{' => {
            *pos += 1;
            let obj = i.new_object();
            json_skip_ws(input, pos);
            if input.get(*pos) == Some('}') {
                *pos += 1;
                return Ok(Value::Obj(obj));
            }
            loop {
                json_skip_ws(input, pos);
                if input.get(*pos) != Some('"') {
                    return Err(i.make_error("SyntaxError", "Expected string key in JSON object"));
                }
                let key = json_parse_string(i, input, pos)?.into_key();
                json_skip_ws(input, pos);
                if input.get(*pos) != Some(':') {
                    return Err(i.make_error("SyntaxError", "Expected ':' in JSON object"));
                }
                *pos += 1;
                let v = nested(i, |i| json_parse_value(i, input, pos))?;
                set_data(&obj, &crate::value::encode_string_key(&key), v);
                json_skip_ws(input, pos);
                match input.get(*pos) {
                    Some(',') => {
                        *pos += 1;
                    }
                    Some('}') => {
                        *pos += 1;
                        break;
                    }
                    _ => {
                        return Err(
                            i.make_error("SyntaxError", "Expected ',' or '}' in JSON object")
                        );
                    }
                }
            }
            Ok(Value::Obj(obj))
        }
        '[' => {
            *pos += 1;
            let mut items = Vec::new();
            json_skip_ws(input, pos);
            if input.get(*pos) == Some(']') {
                *pos += 1;
                return Ok(i.make_array(items));
            }
            loop {
                items.push(nested(i, |i| json_parse_value(i, input, pos))?);
                json_skip_ws(input, pos);
                match input.get(*pos) {
                    Some(',') => {
                        *pos += 1;
                    }
                    Some(']') => {
                        *pos += 1;
                        break;
                    }
                    _ => {
                        return Err(
                            i.make_error("SyntaxError", "Expected ',' or ']' in JSON array")
                        );
                    }
                }
            }
            Ok(i.make_array(items))
        }
        '"' => Ok(json_parse_string(i, input, pos)?.into_value()),
        't' => json_parse_lit(i, input, pos, "true", Value::Bool(true)),
        'f' => json_parse_lit(i, input, pos, "false", Value::Bool(false)),
        'n' => json_parse_lit(i, input, pos, "null", Value::Null),
        '-' | '0'..='9' => {
            let start = *pos;
            // Strict JSON number grammar: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)? — no leading
            // zeros, a mandatory integer part, and at least one digit after `.` / exponent.
            let err = || i.make_error("SyntaxError", "Invalid number in JSON");
            let at = |p: usize| input.get(p);
            if at(*pos) == Some('-') {
                *pos += 1;
            }
            match at(*pos) {
                Some('0') => *pos += 1,
                Some('1'..='9') => {
                    while matches!(at(*pos), Some('0'..='9')) {
                        *pos += 1;
                    }
                }
                _ => return Err(err()),
            }
            if at(*pos) == Some('.') {
                *pos += 1;
                if !matches!(at(*pos), Some('0'..='9')) {
                    return Err(err());
                }
                while matches!(at(*pos), Some('0'..='9')) {
                    *pos += 1;
                }
            }
            if matches!(at(*pos), Some('e' | 'E')) {
                *pos += 1;
                if matches!(at(*pos), Some('+' | '-')) {
                    *pos += 1;
                }
                if !matches!(at(*pos), Some('0'..='9')) {
                    return Err(err());
                }
                while matches!(at(*pos), Some('0'..='9')) {
                    *pos += 1;
                }
            }
            input
                .parse_number(start, *pos)
                .map(Value::Num)
                .map_err(|_| err())
        }
        _ => Err(i.make_error("SyntaxError", "Unexpected token in JSON")),
    }
}

/// A parallel parse tree recording the source text of every primitive leaf, so the JSON.parse
/// reviver can receive a `context` argument with a `source` property (ES2025 source-text access).
enum JsonRecord {
    Prim(String, Value),
    Arr(Vec<JsonRecord>),
    Obj(Vec<(String, JsonRecord)>),
}

impl crate::gc_trace::GcTrace for JsonRecord {
    fn trace_gc(&self, sink: &mut dyn crate::gc_trace::GcSink) {
        // Like Drop below, tracing must not use one native frame per nesting level.
        let mut pending = vec![self];
        while let Some(record) = pending.pop() {
            match record {
                JsonRecord::Prim(_source, parsed) => sink.value(parsed),
                JsonRecord::Arr(children) => pending.extend(children),
                JsonRecord::Obj(entries) => {
                    pending.extend(entries.iter().map(|(_key, child)| child));
                }
            }
        }
    }
}

impl JsonRecord {
    fn take_children(&mut self, out: &mut Vec<JsonRecord>) {
        match self {
            JsonRecord::Prim(..) => {}
            JsonRecord::Arr(items) => out.append(items),
            JsonRecord::Obj(entries) => out.extend(entries.drain(..).map(|(_, record)| record)),
        }
    }
}

impl Drop for JsonRecord {
    /// Records nest as deeply as the text (a million levels parse), so release them from an
    /// explicit worklist rather than one native frame per level.
    fn drop(&mut self) {
        let mut pending = Vec::new();
        self.take_children(&mut pending);
        while let Some(mut record) = pending.pop() {
            record.take_children(&mut pending);
        }
    }
}

/// Mirror of `json_parse_value` that also returns a `JsonRecord`. Containers are framed inline;
/// primitive leaves delegate to `json_parse_value` and capture their exact source span.
fn json_parse_recorded(
    i: &mut Interp,
    input: JsonInput<'_>,
    pos: &mut usize,
) -> Result<(Value, JsonRecord), Value> {
    json_skip_ws(input, pos);
    let c = input
        .get(*pos)
        .ok_or_else(|| i.make_error("SyntaxError", "Unexpected end of JSON input"))?;
    match c {
        '{' => {
            *pos += 1;
            let obj = i.new_object();
            let mut rec: Vec<(String, JsonRecord)> = Vec::new();
            json_skip_ws(input, pos);
            if input.get(*pos) == Some('}') {
                *pos += 1;
                return Ok((Value::Obj(obj), JsonRecord::Obj(rec)));
            }
            loop {
                json_skip_ws(input, pos);
                if input.get(*pos) != Some('"') {
                    return Err(i.make_error("SyntaxError", "Expected string key in JSON object"));
                }
                let key = json_parse_string(i, input, pos)?.into_key();
                json_skip_ws(input, pos);
                if input.get(*pos) != Some(':') {
                    return Err(i.make_error("SyntaxError", "Expected ':' in JSON object"));
                }
                *pos += 1;
                let (v, vr) = nested(i, |i| json_parse_recorded(i, input, pos))?;
                set_data(&obj, &crate::value::encode_string_key(&key), v);
                rec.retain(|(k, _)| k != &key);
                rec.push((key, vr));
                json_skip_ws(input, pos);
                match input.get(*pos) {
                    Some(',') => *pos += 1,
                    Some('}') => {
                        *pos += 1;
                        break;
                    }
                    _ => {
                        return Err(
                            i.make_error("SyntaxError", "Expected ',' or '}' in JSON object")
                        );
                    }
                }
            }
            Ok((Value::Obj(obj), JsonRecord::Obj(rec)))
        }
        '[' => {
            *pos += 1;
            let mut items = Vec::new();
            let mut rec = Vec::new();
            json_skip_ws(input, pos);
            if input.get(*pos) == Some(']') {
                *pos += 1;
                return Ok((i.make_array(items), JsonRecord::Arr(rec)));
            }
            loop {
                let (v, vr) = nested(i, |i| json_parse_recorded(i, input, pos))?;
                items.push(v);
                rec.push(vr);
                json_skip_ws(input, pos);
                match input.get(*pos) {
                    Some(',') => *pos += 1,
                    Some(']') => {
                        *pos += 1;
                        break;
                    }
                    _ => {
                        return Err(
                            i.make_error("SyntaxError", "Expected ',' or ']' in JSON array")
                        );
                    }
                }
            }
            Ok((i.make_array(items), JsonRecord::Arr(rec)))
        }
        _ => {
            let start = *pos;
            let v = json_parse_value(i, input, pos)?;
            let src = input.to_string(start, *pos);
            Ok((v, JsonRecord::Prim(src, v)))
        }
    }
}

fn json_parse_lit(
    i: &mut Interp,
    input: JsonInput<'_>,
    pos: &mut usize,
    lit: &str,
    val: Value,
) -> Result<Value, Value> {
    for expect in lit.chars() {
        if input.get(*pos) != Some(expect) {
            return Err(i.make_error("SyntaxError", "Invalid literal in JSON"));
        }
        *pos += 1;
    }
    Ok(val)
}

fn json_parse_string(
    i: &mut Interp,
    input: JsonInput<'_>,
    pos: &mut usize,
) -> Result<JsonStr, Value> {
    match input {
        JsonInput::Units(units) => json_parse_units(i, units, pos).map(JsonStr::Units),
        _ => json_parse_text(i, input, pos).map(JsonStr::Text),
    }
}

/// A JSON string of two-byte input, as code units: its plain runs copied, its escapes decoded
/// (a `\u` escape is the unit it names, a surrogate half included).
fn json_parse_units(i: &mut Interp, units: &[u16], pos: &mut usize) -> Result<Vec<u16>, Value> {
    *pos += 1; // opening quote
    let mut out = Vec::new();
    loop {
        let start = *pos;
        while let Some(&unit) = units.get(*pos) {
            if unit == u16::from(b'"') || unit == u16::from(b'\\') || unit < 0x20 {
                break;
            }
            *pos += 1;
        }
        out.extend_from_slice(&units[start..*pos]);
        let unit = *units
            .get(*pos)
            .ok_or_else(|| i.make_error("SyntaxError", "Unterminated JSON string"))?;
        *pos += 1;
        if unit == u16::from(b'"') {
            return Ok(out);
        }
        if unit < 0x20 {
            return Err(i.make_error("SyntaxError", "Unescaped control character in JSON string"));
        }
        let escape = units
            .get(*pos)
            .and_then(|&unit| u8::try_from(unit).ok())
            .ok_or_else(|| i.make_error("SyntaxError", "Bad escape in JSON"))?;
        *pos += 1;
        let decoded = match escape {
            b'"' => u16::from(b'"'),
            b'\\' => u16::from(b'\\'),
            b'/' => u16::from(b'/'),
            b'n' => u16::from(b'\n'),
            b't' => u16::from(b'\t'),
            b'r' => u16::from(b'\r'),
            b'b' => 0x08,
            b'f' => 0x0c,
            b'u' => {
                let digits = units.get(*pos..*pos + 4).unwrap_or_default();
                let mut value = 0u16;
                for &digit in digits {
                    let digit = char::from_u32(u32::from(digit))
                        .and_then(|digit| digit.to_digit(16))
                        .ok_or_else(|| i.make_error("SyntaxError", "Bad \\u escape in JSON"))?;
                    value = value * 16 + digit as u16;
                }
                if digits.len() != 4 {
                    return Err(i.make_error("SyntaxError", "Bad \\u escape in JSON"));
                }
                *pos += 4;
                value
            }
            _ => return Err(i.make_error("SyntaxError", "Bad escape in JSON")),
        };
        out.push(decoded);
    }
}

fn json_parse_text(i: &mut Interp, input: JsonInput<'_>, pos: &mut usize) -> Result<String, Value> {
    *pos += 1; // opening quote
    let mut s = String::new();
    // Most JSON strings contain no escapes. Scan that prefix in bulk and copy it once; the
    // character-by-character loop below remains the exact path for escapes, controls, and
    // surrogate decoding.
    let plain_start = *pos;
    match input {
        JsonInput::Bytes(bytes) => {
            while *pos < bytes.len() {
                match bytes[*pos] {
                    b'"' => {
                        let plain = String::from_utf8(bytes[plain_start..*pos].to_vec())
                            .expect("ASCII JSON input is valid UTF-8");
                        *pos += 1;
                        return Ok(plain);
                    }
                    b'\\' | 0..=0x1f => break,
                    _ => *pos += 1,
                }
            }
            if *pos > plain_start {
                s.push_str(
                    std::str::from_utf8(&bytes[plain_start..*pos])
                        .expect("ASCII JSON input is valid UTF-8"),
                );
            }
        }
        JsonInput::Chars(chars) => {
            while *pos < chars.len() {
                match chars[*pos] {
                    '"' => {
                        let plain: String = chars[plain_start..*pos].iter().collect();
                        *pos += 1;
                        return Ok(plain);
                    }
                    '\\' => break,
                    c if (c as u32) < 0x20 => break,
                    _ => *pos += 1,
                }
            }
            if *pos > plain_start {
                s.extend(chars[plain_start..*pos].iter().copied());
            }
        }
        JsonInput::Units(_) => unreachable!("two-byte input parses its strings as units"),
    }
    loop {
        let c = input
            .get(*pos)
            .ok_or_else(|| i.make_error("SyntaxError", "Unterminated JSON string"))?;
        *pos += 1;
        match c {
            '"' => return Ok(s),
            '\\' => {
                let e = input
                    .get(*pos)
                    .ok_or_else(|| i.make_error("SyntaxError", "Bad escape in JSON"))?;
                *pos += 1;
                match e {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    '/' => s.push('/'),
                    'n' => s.push('\n'),
                    't' => s.push('\t'),
                    'r' => s.push('\r'),
                    'b' => s.push('\u{0008}'),
                    'f' => s.push('\u{000C}'),
                    'u' => {
                        let hex = input.to_string(*pos, (*pos + 4).min(input.len()));
                        *pos += 4;
                        let n = u32::from_str_radix(&hex, 16)
                            .map_err(|_| i.make_error("SyntaxError", "Bad \\u escape in JSON"))?;
                        if (0xD800..0xDC00).contains(&n)
                            && input.get(*pos) == Some('\\')
                            && input.get(*pos + 1) == Some('u')
                        {
                            // A high surrogate followed by \uDCxx forms a pair.
                            let hex2 = input.to_string(
                                (*pos + 2).min(input.len()),
                                (*pos + 6).min(input.len()),
                            );
                            if let Ok(n2) = u32::from_str_radix(&hex2, 16) {
                                if (0xDC00..0xE000).contains(&n2) {
                                    *pos += 6;
                                    let c = 0x10000 + ((n - 0xD800) << 10) + (n2 - 0xDC00);
                                    crate::jstr::push_code_point(&mut s, c);
                                    continue;
                                }
                            }
                        }
                        if (0xD800..0xE000).contains(&n) {
                            s.push(crate::jstr::smuggle(n as u16));
                        } else {
                            s.push(char::from_u32(n).unwrap_or('\u{FFFD}'));
                        }
                    }
                    _ => return Err(i.make_error("SyntaxError", "Bad escape in JSON")),
                }
            }
            // Unescaped control characters (U+0000–U+001F) are not allowed in JSON strings.
            c if (c as u32) < 0x20 => {
                return Err(
                    i.make_error("SyntaxError", "Unescaped control character in JSON string")
                );
            }
            c => s.push(c),
        }
    }
}

#[cfg(all(
    test,
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
mod tests {
    use super::*;
    use crate::bytecode::Tier;
    use crate::gc_core::{CellAge, CollectionKind, MarkSource};
    use crate::gc_handle::WeakRef;
    use crate::rooting::Rooted;
    use crate::{Completion, Engine};
    use std::cell::RefCell;

    // ECMA-262 e28783d5, spec.html:47996-48045, InternalizeJSONProperty;
    // spec.html:6065-6067, SameValueNonNumber. Equal String code units keep the
    // original source token even when the replacement has a different GC cell.

    struct Leaf {
        key: String,
        token: String,
        replacement: String,
    }

    fn leaves(keys: [&str; 3]) -> Vec<Leaf> {
        let suffix = "-json-source-record-root-".repeat(4);
        [
            (format!(r#""\u0061{suffix}""#), format!("a{suffix}")),
            (format!(r#""é{suffix}""#), format!("é{suffix}")),
            (
                format!(r#""\ud800{suffix}""#),
                format!("{}{suffix}", crate::jstr::smuggle(0xd800)),
            ),
        ]
        .into_iter()
        .zip(keys)
        .map(|((token, replacement), key)| Leaf {
            key: key.to_owned(),
            token,
            replacement,
        })
        .collect()
    }

    fn literal(text: &str) -> String {
        let mut result = String::new();
        json_quote_into(&mut result, text);
        result
    }

    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine
            .eval(source, false)
            .expect("JSON rooting fixture parses")
        {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn assert_live_strings(interp: &Interp, watched: &[WeakRef], explicit_root: bool) {
        let heap = &interp.gc_heap;
        let core = heap.core.borrow();
        for weak in watched {
            // Check the clearing weak flag and descriptor before any possible payload read.
            let address = weak
                .live_ptr()
                .expect("the original parsed string stays live");
            assert!(core.owns_cell(address, heap.string_desc));
            if explicit_root {
                let path = core.retention_path(address);
                assert!(
                    matches!(path.first(), Some((_, _, MarkSource::Root))),
                    "the source record must explicitly root its parsed string: {path:?}"
                );
            }
        }
    }

    #[derive(Default)]
    struct Watched {
        strings: Vec<WeakRef>,
        calls: usize,
    }

    fn check_replaced_siblings(text: &str, first_key: &str, leaves: Vec<Leaf>, abrupt: bool) {
        let expected = leaves
            .iter()
            .map(|leaf| format!("{}:{}", literal(&leaf.key), literal(&leaf.token)))
            .collect::<Vec<_>>()
            .join(",");
        let order = leaves
            .iter()
            .map(|leaf| leaf.key.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let source = format!(
            r#"(function() {{
                var expected = {{{expected}}}, seen = [];
                var caught = false;
                try {{
                    JSON.parse({text}, function(key, value, context) {{
                        if (key === {first_key}) {{
                            __replaceJSONStrings.call(this);
                            if ({abrupt}) throw 73;
                        }}
                        if (Object.hasOwn(expected, key)) {{
                            if (!Object.hasOwn(context, 'source') || context.source !== expected[key])
                                throw new Error('source for ' + key);
                            seen.push(key);
                        }}
                        return value;
                    }});
                }} catch (error) {{
                    if (!{abrupt} || error !== 73) throw error;
                    caught = true;
                }}
                if ({abrupt} ? !caught : seen.join(',') !== {order}) throw new Error('visits');
                return 'ok';
            }})()"#,
            text = literal(text),
            first_key = literal(first_key),
            order = literal(&order),
        );
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine.interp.activate_gc_heap();
            engine
                .interp
                .gc_heap
                .core
                .borrow_mut()
                .set_retention_tracing(true);
            let watched = Rc::new(RefCell::new(Watched::default()));
            let callback_state = watched.clone();
            let replacements = leaves
                .iter()
                .map(|leaf| (leaf.key.clone(), leaf.replacement.clone()))
                .collect::<Vec<_>>();
            let native = engine.interp.new_native_fn(
                "__replaceJSONStrings",
                0,
                Rc::new(move |interp, holder, _| {
                    let object = holder.as_obj().expect("parsed array or object holder");
                    let mut originals = Vec::new();
                    for (key, text) in &replacements {
                        let old = ab(interp.get_member_key(&holder, key))?;
                        let ValueView::Str(string) = old.view() else {
                            panic!("a later parsed property is a string");
                        };
                        let weak = WeakRef::to(string.cell_word() as usize);
                        let replacement = Value::lstr(text.as_str());
                        assert!(same_value(&old, &replacement));
                        assert_ne!(old.cell_word(), replacement.cell_word());
                        set_data(&object, key, replacement);
                        originals.push(weak);
                    }
                    assert_eq!(originals.len(), 3);
                    for kind in [CollectionKind::Minor, CollectionKind::Major] {
                        let before = interp.gc_heap.core.borrow().stats().collections;
                        // AST evaluation needs its legitimate native temporaries. Roots are
                        // visited before that scan, so require Root rather than accepting a
                        // stale parser-frame copy as evidence of source-record ownership.
                        interp.gc_heap_collect_explicit(kind);
                        assert_eq!(interp.gc_heap.core.borrow().stats().collections, before + 1);
                        assert_live_strings(interp, &originals, kind == CollectionKind::Major);
                    }
                    let mut state = callback_state.borrow_mut();
                    state.strings = originals;
                    state.calls += 1;
                    Ok(Value::Undefined)
                }),
                // Only Rust strings, counters and weak observers are captured.
                crate::value::NativeCaptures::None,
            );
            set_data(&engine.interp.global, "__replaceJSONStrings", native);
            assert_eq!(eval(&mut engine, &source), "ok", "{tier:?}");
            assert_eq!(watched.borrow().calls, 1, "{tier:?}");
            // Outside JS, exclude all stale native words and prove both normal and abrupt
            // completion unregister the record instead of keeping the originals forever.
            engine
                .interp
                .gc_collect_precise_for_test(CollectionKind::Major);
            assert!(watched
                .borrow()
                .strings
                .iter()
                .all(|weak| weak.live_ptr().is_none()));
        }
    }

    #[test]
    fn json_reviver_equal_sibling_strings_are_explicit_roots_in_all_tiers() {
        let object = leaves(["a", "b", "c"]);
        let text = format!(
            r#"{{"kick":0,"a":{},"b":{},"c":{}}}"#,
            object[0].token, object[1].token, object[2].token
        );
        check_replaced_siblings(&text, "kick", object, false);
        let array = leaves(["1", "2", "3"]);
        let text = format!(
            r#"[0,{},{},{}]"#,
            array[0].token, array[1].token, array[2].token
        );
        check_replaced_siblings(&text, "0", array, false);
    }

    #[test]
    fn json_reviver_duplicate_keys_keep_the_last_source_after_gc() {
        let entries = leaves(["dup", "b", "c"]);
        let text = format!(
            r#"{{"kick":0,"dup":"discarded first token","b":{},"dup":{},"c":{}}}"#,
            entries[1].token, entries[0].token, entries[2].token
        );
        check_replaced_siblings(&text, "kick", entries, false);
    }

    #[test]
    fn json_reviver_throw_retires_source_record_roots() {
        let entries = leaves(["a", "b", "c"]);
        let text = format!(
            r#"{{"kick":0,"a":{},"b":{},"c":{}}}"#,
            entries[0].token, entries[1].token, entries[2].token
        );
        check_replaced_siblings(&text, "kick", entries, true);
    }

    #[test]
    fn json_reviver_samevalue_and_scalar_source_controls() {
        let source = r#"(function() {
            function check(ok) { if (!ok) throw new Error('source control'); }
            var visits = 0;
            JSON.parse('[0,"same","change","delete",-0,0,1.50]', function(k,v,c) {
                if (k === '0') {
                    this[1] = 'sam' + 'e'; this[2] = 'different'; delete this[3];
                    this[4] = 0; this[5] = -0; this[6] = 1.5;
                    __forceJSONCollection();
                }
                if (k === '1') check(c.source === '"same"');
                if (k === '2' || k === '3' || k === '4' || k === '5')
                    check(!Object.hasOwn(c, 'source'));
                if (k === '6') check(c.source === '1.50');
                if (k !== '') visits++;
                return v;
            });
            check(visits === 7);
            for (var text of ['  "scalar"  ', ' 1.50 ', ' true ', ' null ', ' -0 ']) {
                var count = 0;
                JSON.parse(text, function(k,v,c) {
                    check(k === '' && c.source === text.trim());
                    if (text.trim() === '-0') check(Object.is(v, -0));
                    __forceJSONCollection();
                    count++;
                    return v;
                });
                check(count === 1);
            }
            return 'ok';
        })()"#;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            let native = engine.interp.new_native_fn(
                "__forceJSONCollection",
                0,
                Rc::new(|interp, _, _| {
                    interp.gc_heap_collect_explicit(CollectionKind::Major);
                    Ok(Value::Undefined)
                }),
                crate::value::NativeCaptures::None,
            );
            set_data(&engine.interp.global, "__forceJSONCollection", native);
            assert_eq!(eval(&mut engine, source), "ok", "{tier:?}");
        }
    }

    fn watch_record_strings(record: &JsonRecord) -> Vec<WeakRef> {
        // Independent of GcTrace: an omitted trace arm must not omit its own witnesses.
        let mut watched = Vec::new();
        let mut pending = vec![record];
        while let Some(record) = pending.pop() {
            match record {
                JsonRecord::Prim(_source, parsed) => {
                    if let ValueView::Str(string) = parsed.view() {
                        watched.push(WeakRef::to(string.cell_word() as usize));
                    }
                }
                JsonRecord::Arr(children) => pending.extend(children),
                JsonRecord::Obj(entries) => {
                    pending.extend(entries.iter().map(|(_key, child)| child));
                }
            }
        }
        watched
    }

    fn check_precise_record(engine: &mut Engine, record: JsonRecord, leaves: usize) {
        let watched = watch_record_strings(&record);
        assert_eq!(watched.len(), leaves);
        for weak in &watched {
            assert_eq!(
                engine
                    .interp
                    .gc_heap
                    .core
                    .borrow()
                    .cell_age(weak.ptr() as u64),
                Some(CellAge::Young)
            );
        }
        let record = Rooted::new(record);
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            engine.interp.gc_collect_precise_for_test(kind);
            assert_live_strings(&engine.interp, &watched, true);
        }
        drop(record);
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Major);
        assert!(watched.iter().all(|weak| weak.live_ptr().is_none()));
    }

    #[test]
    fn json_record_scalar_array_and_object_are_precise_roots_until_drop() {
        let token = literal(&"fresh noninterned parsed string ".repeat(8));
        for text in [
            token.clone(),
            format!("[{token}]"),
            format!(r#"{{"nested":[{token}]}}"#),
        ] {
            let mut engine = Engine::new();
            engine.interp.activate_gc_heap();
            engine
                .interp
                .gc_heap
                .core
                .borrow_mut()
                .set_retention_tracing(true);
            let mut position = 0;
            let (_, record) = json_parse_recorded(
                &mut engine.interp,
                JsonInput::Bytes(text.as_bytes()),
                &mut position,
            )
            .unwrap_or_else(|_| panic!("recorded fixture parses"));
            assert_eq!(position, text.len());
            check_precise_record(&mut engine, record, 1);
        }
    }

    #[test]
    fn json_record_deep_trace_and_drop_are_iterative() {
        std::thread::Builder::new()
            .name("deep-json-record".to_owned())
            .stack_size(1 << 20)
            .spawn(|| {
                let mut engine = Engine::new();
                engine.interp.activate_gc_heap();
                engine
                    .interp
                    .gc_heap
                    .core
                    .borrow_mut()
                    .set_retention_tracing(true);
                let parsed = Value::lstr("deep source record string ".repeat(8));
                let mut record = JsonRecord::Prim("a Rust source token".to_owned(), parsed);
                for depth in 0..50_000 {
                    record = if depth & 1 == 0 {
                        JsonRecord::Arr(vec![record])
                    } else {
                        JsonRecord::Obj(vec![(String::new(), record)])
                    };
                }
                check_precise_record(&mut engine, record, 1);
            })
            .expect("spawn deep JSON record test")
            .join()
            .expect("deep JSON record trace/drop panicked");
    }
}
