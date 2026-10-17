//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;

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
            let len = match &replacer {
                Value::Obj(o) if proxy_pair(i, &replacer).is_none() => i.array_length(o),
                Value::Obj(o) => ab(i.to_length(o))?,
                _ => 0,
            };
            let mut list: Vec<Rc<str>> = Vec::new();
            // JSON.stringify's PropertyList keeps the first occurrence of each String while
            // preserving input order (ECMA-262 §25.5.4.1). Hash the already-coerced keys so a
            // large replacer array does not repeatedly rescan the prefix of `list`.
            let mut list_seen = crate::fasthash::FastSet::default();
            for k in 0..len {
                let item = ab(i.get_member_key(&replacer, &k.to_string()))?;
                // String/Number primitives and their wrappers contribute a key via ToString.
                let key = match &item {
                    Value::Str(s) => Some(s.to_string()),
                    Value::Num(n) => Some(i.num_to_str(*n)),
                    Value::Obj(o)
                        if matches!(o.borrow().exotic, Exotic::StrWrap(_) | Exotic::NumWrap(_)) =>
                    {
                        Some(ab(i.to_string(&item))?.to_string())
                    }
                    _ => None,
                };
                if let Some(key) = key {
                    if list_seen.insert(key.clone()) {
                        list.push(Rc::from(crate::value::encode_owned_string_key(key)));
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
        if let Value::Obj(o) = &space {
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
        let gap = match space {
            Value::Num(n) => {
                let n = if n.is_nan() { 0.0 } else { n.trunc() };
                " ".repeat(n.clamp(0.0, 10.0) as usize)
            }
            Value::Str(s) if s.ascii_hint() => s[..s.len().min(10)].to_owned(),
            // JSON.stringify takes the first ten UTF-16 code units, including a lone
            // surrogate when that boundary splits an astral character. Decode only
            // this bounded prefix instead of materializing the full indentation string.
            Value::Str(s) => {
                crate::jstr::from_units(&crate::jstr::unit_iter(&s).take(10).collect::<Vec<_>>())
            }
            _ => String::new(),
        };
        // SerializeJSONProperty starts from a wrapper holder `{ "": value }`; only a replacer
        // function can observe it (as its receiver).
        let holder = if opts.func.is_some() {
            let wrapper = i.new_object();
            set_data(&wrapper, "", value.clone());
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
            out: String::with_capacity(64),
        };
        if json_serialize_property(i, &mut serializer, &holder, JsonKey::Name(""), value)? {
            // Repeated indentation can join a trailing high surrogate to the next
            // gap's leading low surrogate. Keep the serializer's byte-based rollback
            // intact and canonicalize those pairs only in the completed output.
            if crate::jstr::needs_join_fixup(&gap, &gap) {
                if let Some(out) = crate::jstr::canonicalize(&serializer.out) {
                    return Ok(Value::lstr(out));
                }
            }
            Ok(Value::lstr(serializer.out))
        } else {
            Ok(Value::Undefined)
        }
    });
    it.def_method(&j, "parse", 2, |i, _t, args| {
        let text = ab(i.to_string(&arg(args, 0)))?;
        // JSON grammar is ASCII-delimited. Keep ASCII source in its original byte slice rather
        // than widening every byte to a four-byte `char`; escaped `\u` sequences are still
        // decoded by `json_parse_string` into the engine's UTF-16 representation.
        let chars = (!text.as_str().is_ascii()).then(|| text.chars().collect::<Vec<_>>());
        let input = match chars.as_deref() {
            Some(chars) => JsonInput::Chars(chars),
            None => JsonInput::Bytes(text.as_str().as_bytes()),
        };
        let mut pos = 0;
        let reviver = arg(args, 1);
        // A callable reviver walks the result via InternalizeJSONProperty from a `{ "": v }` root,
        // recording primitive source spans for its `context` argument.
        if reviver.is_callable() {
            let (v, record) = json_parse_recorded(i, input, &mut pos)?;
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
        i.freeze_object(&Value::Obj(o.clone()));
        Ok(Value::Obj(o))
    });
    it.def_method(&j, "isRawJSON", 1, |_i, _t, args| {
        Ok(Value::Bool(
            matches!(arg(args, 0), Value::Obj(o) if o.borrow().internal_slots().contains("\u{0}raw_json")),
        ))
    });
    set_to_string_tag(it, &j, "JSON");
    set_builtin(&it.global, "JSON", Value::Obj(j));
}

/// Append QuoteJSONString(s) (ECMA-262 §25.5.2.3) to `out`.
fn json_quote_into(out: &mut String, s: &str) {
    // QuoteJSONString (ECMA-262 §25.5.4.3) operates on UTF-16 code points. ASCII has no
    // surrogate or multi-unit cases, so keep this common path byte-based and avoid the Unicode
    // iterator/peek state entirely.
    out.reserve(s.len() + 2);
    if s.is_ascii() {
        out.push('"');
        for &b in s.as_bytes() {
            match b {
                b'"' => out.push_str("\\\""),
                b'\\' => out.push_str("\\\\"),
                b'\n' => out.push_str("\\n"),
                b'\r' => out.push_str("\\r"),
                b'\t' => out.push_str("\\t"),
                0x08 => out.push_str("\\b"),
                0x0c => out.push_str("\\f"),
                b if b < 0x20 => push_json_unicode_escape(out, b as u16),
                b => out.push(b as char),
            }
        }
        out.push('"');
        return;
    }

    out.push('"');
    let mut chars = s.chars().peekable();
    while let Some(mut c) = chars.next() {
        // A smuggled pair round-trips as its real character. If that character itself falls in
        // the smuggle range (a real plane-16 PUA code point), keep the smuggled representation so
        // it isn't re-read as a lone surrogate below.
        if let Some(real) = chars.peek().and_then(|&n| crate::jstr::paired_char(c, n)) {
            if crate::jstr::smuggled(real).is_some() {
                out.push(c);
                out.push(chars.next().unwrap());
                continue;
            }
            chars.next();
            c = real;
        }
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => push_json_unicode_escape(out, c as u16),
            c => match crate::jstr::smuggled(c) {
                // Well-formed JSON.stringify: a lone surrogate is written as its \u escape.
                Some(u) => push_json_unicode_escape(out, u),
                None => out.push(c),
            },
        }
    }
    out.push('"');
}

#[inline]
fn push_json_unicode_escape(out: &mut String, unit: u16) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push_str("\\u");
    out.push(HEX[((unit >> 12) & 0x0f) as usize] as char);
    out.push(HEX[((unit >> 8) & 0x0f) as usize] as char);
    out.push(HEX[((unit >> 4) & 0x0f) as usize] as char);
    out.push(HEX[(unit & 0x0f) as usize] as char);
}

/// JSON.stringify options: an optional function replacer and/or an array PropertyList of keys.
struct JsonOpts {
    func: Option<Value>,
    keys: Option<Vec<Rc<str>>>,
}

/// [[Delete]] for the JSON reviver: trap-aware, discarding the boolean status (per `Perform ?`).
fn json_delete_prop(i: &mut Interp, holder: &Value, key: &str) -> Result<(), Value> {
    if let Some((target, handler)) = proxy_pair(i, holder) {
        ab(i.proxy_delete(target, handler, key))?;
        return Ok(());
    }
    if let Value::Obj(o) = holder {
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
    if let Value::Obj(o) = holder {
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
    if matches!(val, Value::Obj(_)) {
        if json_is_array(i, &val)? {
            let len = match val.as_obj() {
                Some(o) => ab(i.to_length(o))?,
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
                if matches!(new_el, Value::Undefined) {
                    json_delete_prop(i, &val, &k)?;
                } else {
                    json_create_data_prop(i, &val, &k, new_el)?;
                }
            }
        } else {
            let keys: Vec<String> = if proxy_pair(i, &val).is_some() {
                proxy_enum_string_keys(i, &val)?
                    .iter()
                    .filter_map(|k| match k {
                        Value::Str(s) => Some(crate::value::encode_owned_string_key(s.to_string())),
                        _ => None,
                    })
                    .collect()
            } else if let Some(info) = val.as_obj().and_then(|o| ta_info(i, o)) {
                // A TypedArray holder (a reviver may graft one in) walks its integer indices.
                (0..i.ta_len(&info).unwrap_or(0))
                    .map(|k| k.to_string())
                    .collect()
            } else {
                val.as_obj()
                    .map(|o| ordered_enum_keys(o).iter().map(|k| k.to_string()).collect())
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
                if matches!(new_el, Value::Undefined) {
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
        if !matches!(val, Value::Obj(_)) && same_value(&val, parsed) {
            set_data(&context, "source", Value::lstr(src.clone()));
        }
    }
    ab(i.call(
        reviver.clone(),
        holder.clone(),
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
    out: String,
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
fn json_get(i: &mut Interp, holder: &Value, object: &Gc, key: &str) -> Result<Value, Value> {
    if json_plain_get(i, object) {
        if let Some(property) = object.borrow().props.get(key) {
            if !property.accessor() {
                return Ok(property.value());
            }
        }
    }
    ab(i.get_member_key(holder, key))
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
    if let Value::Obj(start) = value {
        let mut current = start.clone();
        loop {
            let next = {
                let object = current.borrow();
                if !object.ic_plain.get() {
                    break;
                }
                match object.props.get("toJSON") {
                    Some(property) if !property.accessor() => return Ok(property.value()),
                    Some(_) => break,
                    None => object.proto.clone(),
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
    if matches!(value, Value::Obj(_) | Value::BigInt(_)) {
        let to_json = json_to_json_method(i, &value)?;
        if to_json.is_callable() {
            value = ab(i.call(to_json, value.clone(), &[key.value()]))?;
        }
    }
    if let Some(func) = &s.opts.func {
        value = ab(i.call(func.clone(), holder.clone(), &[key.value(), value]))?;
    }
    if let Value::Obj(o) = &value {
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
        if let Some(Value::Str(raw)) = raw {
            s.out.push_str(&raw);
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
    match &value {
        Value::Undefined | Value::Empty | Value::Sym(_) => return Ok(false),
        Value::Null => s.out.push_str("null"),
        Value::Bool(b) => s.out.push_str(if *b { "true" } else { "false" }),
        Value::Num(n) if !n.is_finite() => s.out.push_str("null"),
        Value::Num(n) if n.fract() == 0.0 && n.abs() < 1e15 => {
            // Integral and below 10^21, so Number::toString prints its digits (-0 prints "0").
            use std::fmt::Write;
            let _ = write!(s.out, "{}", *n as i64);
        }
        Value::Num(n) => s.out.push_str(&i.num_to_str(*n)),
        // JSON.stringify of a BigInt throws (matches the spec).
        Value::BigInt(_) => {
            return Err(i.make_error("TypeError", "Do not know how to serialize a BigInt"));
        }
        Value::Str(text) => {
            // Escaping only lengthens the text.
            check_string_length(i, s.out.len().saturating_add(text.len()))?;
            json_quote_into(&mut s.out, text);
        }
        Value::Obj(o) => {
            if !matches!(o.borrow().call, Callable::None) {
                return Ok(false); // functions are omitted
            }
            let o = o.clone();
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
    let keys: Vec<Rc<str>> = match &s.opts.keys {
        Some(list) => list.clone(),
        None if proxy_pair(i, value).is_some() => proxy_enum_string_keys(i, value)?
            .iter()
            .filter_map(|key| match key {
                Value::Str(key) => Some(Rc::from(crate::value::encode_string_key(key).as_ref())),
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
                keys.retain(|key| {
                    !Interp::is_sym_key(key)
                        && object
                            .props
                            .get(key)
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
                    keys.push(Rc::from(key.as_str()));
                }
            }
            keys
        }
    };
    s.out.push('{');
    let mut any = false;
    for key in &keys {
        let member_start = s.out.len();
        json_separator(s, !any);
        json_quote_into(&mut s.out, crate::value::decode_string_key(key));
        s.out.push(':');
        if !s.gap.is_empty() {
            s.out.push(' ');
        }
        let member = json_get(i, value, object, key)?;
        if json_serialize_property(i, s, value, JsonKey::Name(key), member)? {
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
}

impl JsonInput<'_> {
    #[inline]
    fn len(self) -> usize {
        match self {
            Self::Bytes(bytes) => bytes.len(),
            Self::Chars(chars) => chars.len(),
        }
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn get(self, index: usize) -> Option<char> {
        match self {
            Self::Bytes(bytes) => bytes.get(index).copied().map(char::from),
            Self::Chars(chars) => chars.get(index).copied(),
        }
    }

    fn to_string(self, start: usize, end: usize) -> String {
        match self {
            Self::Bytes(bytes) => String::from_utf8(bytes[start..end].to_vec())
                .expect("ASCII JSON input is valid UTF-8"),
            Self::Chars(chars) => chars[start..end].iter().collect(),
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
                let key = json_parse_string(i, input, pos)?;
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
        '"' => Ok(Value::lstr(json_parse_string(i, input, pos)?)),
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
                let key = json_parse_string(i, input, pos)?;
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
            Ok((v.clone(), JsonRecord::Prim(src, v)))
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
) -> Result<String, Value> {
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
