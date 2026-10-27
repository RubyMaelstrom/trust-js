//! A from-scratch ECMA-402 (`Intl`) without an ICU dependency. The locale data (aliases, number
//! symbols, date patterns, time zone names, …) is CLDR's, embedded as generated tables for the
//! advertised locales. The structural machinery — BCP-47 tag parsing/canonicalization, option
//! coercion, locale negotiation — covers every locale.

use crate::interpreter::Interp;
use crate::value::ValueView;
use crate::value::{set_builtin, Gc, Object, Property, Value};
use std::rc::Rc;

mod collator;
mod data;
mod datetimeformat;
mod displaynames;
mod dtf_interval;
mod dtf_pattern;
mod durationformat;
mod listformat;
mod locale;
mod numberformat;
mod pluralrules;
mod relativetimeformat;
mod segmenter;
mod service;
mod tags;
mod tz_names;

pub(crate) use collator::locale_compare;
pub use tags::{canonicalize_language_tag, is_structurally_valid_tag};

/// The English (CLDR `en`) specific non-location long name of a time zone at an instant, such
/// as "Central European Summer Time", when CLDR has one: the implementation-defined name
/// `Date.prototype.toString` appends (ECMA-262 #sec-timezoneestring).
pub(crate) fn time_zone_long_name(zone: &str, epoch_sec: i64) -> Option<String> {
    tz_names::english_long_name(zone, epoch_sec)
}

fn arg(args: &[Value], i: usize) -> Value {
    args.get(i).cloned().unwrap_or(Value::Undefined)
}

/// Install `Intl` and its services on the global object.
pub fn install(it: &mut Interp) {
    let intl = Object::new(Some(it.object_proto));

    // Intl[@@toStringTag] = "Intl" { writable:false, enumerable:false, configurable:true }.
    if let Some(key) = crate::builtins::to_string_tag_key(it) {
        intl.borrow_mut()
            .props
            .insert(key, Property::data(Value::lstr("Intl"), false, false, true));
    }

    // The per-realm %Intl%.[[FallbackSymbol]] backing the legacy constructed-instance behavior
    // (kept in a holder object so it snapshots with the realm's extra_protos).
    let fsym = it.new_symbol(Some("IntlLegacyConstructedSymbol".into()));
    let holder = Object::new(None);
    holder
        .borrow_mut()
        .props
        .insert("sym", Property::data(fsym, false, false, false));
    Rc::make_mut(&mut it.extra_protos).insert("%IntlFallbackSymbol%", holder);

    // Intl.getCanonicalLocales(locales) — returns a fresh Array of canonicalized tags.
    let f = it.make_native("getCanonicalLocales", 1, |i, _t, a| {
        let list = canonicalize_locale_list(i, &arg(a, 0))?;
        Ok(i.make_array(list.into_iter().map(Value::lstr).collect()))
    });
    set_builtin(&intl, "getCanonicalLocales", Value::Obj(f));

    // Intl.supportedValuesOf(key).
    let f = it.make_native("supportedValuesOf", 1, |i, _t, a| {
        supported_values_of(i, &arg(a, 0))
    });
    set_builtin(&intl, "supportedValuesOf", Value::Obj(f));

    locale::install(it, &intl);
    listformat::install(it, &intl);
    pluralrules::install(it, &intl);
    relativetimeformat::install(it, &intl);
    numberformat::install(it, &intl);
    collator::install(it, &intl);
    displaynames::install(it, &intl);
    segmenter::install(it, &intl);
    durationformat::install(it, &intl);
    datetimeformat::install(it, &intl);

    // Cache the service constructors as intrinsics: toLocaleString and friends delegate to
    // %Intl.NumberFormat% etc. directly, so tainting the Intl globals can't break them.
    for svc in [
        "NumberFormat",
        "DateTimeFormat",
        "Collator",
        "ListFormat",
        "PluralRules",
        "RelativeTimeFormat",
        "DisplayNames",
        "Segmenter",
        "DurationFormat",
        "Locale",
    ] {
        let ctor = intl.borrow().props.get(svc).map(|p| p.value());
        if let Some(ValueView::Obj(c)) = (ctor).map(|__value| (__value).view()) {
            let key = crate::builtins::static_name(&format!("%Intl.{svc}%"));
            Rc::make_mut(&mut it.extra_protos).insert(key, c);
        }
    }
    it.global
        .borrow_mut()
        .props
        .insert("Intl", Property::builtin(Value::Obj(intl)));
}

/// CanonicalizeLocaleList(locales): `undefined` → empty; a String is treated as a single-element
/// list; otherwise an array-like whose elements must be Strings or Locale objects. Returns the
/// deduplicated list of canonical tags (throwing RangeError on a structurally invalid tag).
pub(crate) fn canonicalize_locale_list(
    i: &mut Interp,
    locales: &Value,
) -> Result<Vec<String>, Value> {
    let mut seen: Vec<String> = Vec::new();
    if matches!(&(locales).view(), ValueView::Undefined) {
        return Ok(seen);
    }
    // A String or Locale value is a single-element list; otherwise iterate the array-like, reading
    // and canonicalizing each element in turn (an element's ToString may mutate later indices).
    match &(locales).view() {
        ValueView::Str(_) => process_locale_item(i, locales, &mut seen)?,
        ValueView::Obj(o) if o.borrow().internal_slots().contains("#\u{0}locale_tag") => {
            process_locale_item(i, locales, &mut seen)?
        }
        ValueView::Null => return Err(i.make_error("TypeError", "Cannot convert null to object")),
        _ => {
            // CanonicalizeLocaleList does ToObject(locales) first, so a primitive (number, boolean,
            // symbol) is boxed and its inherited `length`/indexed properties are read.
            let obj = crate::builtins::box_primitive_pub(i, *locales);
            let lenv = ab(i.get_member_key(&obj, "length"))?;
            let len = to_length(i, &lenv)?;
            for k in 0..len {
                let key = k.to_string();
                if ab(i.js_has_property(&obj, &key))? {
                    let item = ab(i.get_member_key(&obj, &key))?;
                    process_locale_item(i, &item, &mut seen)?;
                }
            }
        }
    }
    Ok(seen)
}

/// Canonicalize one locale-list element (a String, Locale, or Object) and append it (deduplicated).
fn process_locale_item(i: &mut Interp, item: &Value, seen: &mut Vec<String>) -> Result<(), Value> {
    let tag = match &(item).view() {
        ValueView::Obj(o) if o.borrow().internal_slots().contains("#\u{0}locale_tag") => {
            match (o
                .borrow()
                .internal_slots()
                .get("#\u{0}locale_tag")
                .map(|p| p.value()))
            .map(|__value| (__value).view())
            {
                Some(ValueView::Str(s)) => s.to_string(),
                _ => String::new(),
            }
        }
        ValueView::Str(_) | ValueView::Obj(_) => {
            let s = ab(i.to_string(item))?;
            let text = s.text();
            if !is_structurally_valid_tag(&text) {
                return Err(i.make_error(
                    "RangeError",
                    format!("Incorrect locale information provided: {s}"),
                ));
            }
            match canonicalize_language_tag(&text) {
                Some(c) => c,
                None => {
                    return Err(i.make_error(
                        "RangeError",
                        format!("Incorrect locale information provided: {s}"),
                    ));
                }
            }
        }
        _ => return Err(i.make_error("TypeError", "locale must be a string or object")),
    };
    if !seen.contains(&tag) {
        seen.push(tag);
    }
    Ok(())
}

fn supported_values_of(i: &mut Interp, key: &Value) -> Result<Value, Value> {
    let k = ab(i.to_string(key))?;
    let text = k.text();
    // The `timeZone` list is every canonical IANA zone; the others are static. SupportedValuesOf must
    // return a sorted array with no duplicates.
    if &*text == "timeZone" {
        // Etc/GMT and Etc/UTC canonicalize to "UTC" and must not appear themselves.
        let mut zs: Vec<&str> = crate::tz::canonical_zone_names();
        zs.retain(|z| *z != "Etc/GMT" && *z != "Etc/UTC");
        if !zs.contains(&"UTC") {
            zs.push("UTC");
        }
        zs.sort_unstable();
        zs.dedup();
        return Ok(i.make_array(zs.iter().map(|s| Value::lstr(*s)).collect()));
    }
    if &*text == "numberingSystem" {
        // Every numeric system with a simple digit mapping, plus latn.
        let mut ns: Vec<&str> = crate::numbering::NUMBERING
            .iter()
            .map(|(n, _)| *n)
            .collect();
        ns.push("latn");
        ns.sort_unstable();
        return Ok(i.make_array(ns.iter().map(|s| Value::lstr(*s)).collect()));
    }
    if &*text == "currency" {
        // ECMA-402 AvailableCurrencies contains every canonical ISO 4217 code for which both
        // NumberFormat and DisplayNames provide functionality. The generated CLDR table is sorted.
        let currencies = crate::cldr_numbers::currency_codes()
            .map(Value::lstr)
            .collect();
        return Ok(i.make_array(currencies));
    }
    let vals: &[&str] = match &*text {
        // "islamic" and "islamic-rgsa" resolve to islamic-civil, so they are not fixed points
        // of DateTimeFormat and are excluded (the era/monthCode proposal's required set).
        "calendar" => &[
            "buddhist",
            "chinese",
            "coptic",
            "dangi",
            "ethioaa",
            "ethiopic",
            "gregory",
            "hebrew",
            "indian",
            "islamic-civil",
            "islamic-tbla",
            "islamic-umalqura",
            "iso8601",
            "japanese",
            "persian",
            "roc",
        ],
        // Exactly the collations some locale's Collator resolves.
        "collation" => collator::available_collations(),
        "numberingSystem" => &[
            "adlm", "ahom", "arab", "arabext", "bali", "beng", "deva", "fullwide", "gujr", "guru",
            "hanidec", "khmr", "knda", "laoo", "latn", "mlym", "mymr", "orya", "tamldec", "telu",
            "thai", "tibt",
        ],
        "unit" => &crate::units::SANCTIONED_UNITS,
        _ => return Err(i.make_error("RangeError", format!("invalid key: {k}"))),
    };
    let mut v: Vec<&str> = vals.to_vec();
    v.sort_unstable();
    v.dedup();
    Ok(i.make_array(v.iter().map(|s| Value::lstr(*s)).collect()))
}

// ----- shared helpers used across the intl services --------------------------------------------

/// Map an interpreter `Abrupt` to the native `Result<_, Value>` contract.
pub(crate) fn ab<T>(r: Result<T, crate::interpreter::Abrupt>) -> Result<T, Value> {
    r.map_err(|a| match a {
        crate::interpreter::Abrupt::Throw(v) => v,
        _ => Value::Undefined,
    })
}

/// Build a constructor/prototype pair with the given `@@toStringTag`, install it on `ns`.
pub(crate) fn make_service(
    it: &mut Interp,
    ns: &Gc,
    name: &'static str,
    len: usize,
    ctor_fn: crate::value::NativeFn,
) -> (Gc, Gc) {
    let proto = Object::new(Some(it.object_proto));
    let ctor = it.make_native(name, len, ctor_fn);
    ctor.borrow_mut().is_constructor = true;
    ctor.borrow_mut().props.insert(
        "prototype",
        Property::data(Value::Obj(proto), false, false, false),
    );
    proto
        .borrow_mut()
        .props
        .insert("constructor", Property::builtin(Value::Obj(ctor)));
    if let Some(key) = crate::builtins::to_string_tag_key(it) {
        proto.borrow_mut().props.insert(
            key,
            Property::data(Value::lstr(format!("Intl.{name}")), false, false, true),
        );
    }
    Rc::make_mut(&mut it.extra_protos)
        .insert(crate::builtins::static_name(&format!("Intl.{name}")), proto);
    set_builtin(ns, name, Value::Obj(ctor));
    (ctor, proto)
}

/// UnwrapNumberFormat / UnwrapDateTimeFormat: a receiver without the brand may carry a legacy
/// chained instance under the realm's %Intl%.[[FallbackSymbol]].
pub(crate) fn brand_slot_legacy(
    i: &mut Interp,
    this: &Value,
    marker: &str,
    proto_key: &str,
) -> Result<Gc, Value> {
    if let Some(o) = this.as_obj() {
        if o.borrow().internal_slots().contains(marker) {
            return Ok(o);
        }
        let symv = i
            .extra_protos
            .get("%IntlFallbackSymbol%")
            .and_then(|h| h.borrow().props.get("sym").map(|p| p.value()));
        if let Some(ValueView::Sym(sd)) = (symv).map(|__value| (__value).view()) {
            if inherits_service_proto(i, &o, proto_key) {
                let key = crate::interpreter::Interp::sym_key(&sd);
                let v = ab(i.get_member_key(this, &key))?;
                if let Some(inner) = v.as_obj() {
                    if inner.borrow().internal_slots().contains(marker) {
                        return Ok(inner);
                    }
                }
            }
        }
    }
    Err(i.make_error("TypeError", "method called on an incompatible receiver"))
}

/// Whether `o` inherits from the service prototype registered under `proto_key`.
fn inherits_service_proto(i: &Interp, o: &Gc, proto_key: &str) -> bool {
    let proto = match i.extra_protos.get(proto_key) {
        Some(p) => *p,
        None => return false,
    };
    let mut cur = o.borrow().proto;
    for _ in 0..64 {
        match cur {
            Some(p) => {
                if Gc::ptr_eq(&p, &proto) {
                    return true;
                }
                cur = p.borrow().proto;
            }
            None => break,
        }
    }
    false
}

/// The legacy Intl-constructor chain behavior (%Intl%.[[FallbackSymbol]]): when a service
/// constructor is *called* on an object inheriting from its prototype, the fresh instance is
/// defined on that object under the realm's fallback symbol and the object itself is returned.
pub(crate) fn legacy_chain(
    i: &mut Interp,
    this: &Value,
    proto_key: &str,
    instance: &Gc,
) -> Option<Value> {
    let this_o = this.as_obj()?;
    if !inherits_service_proto(i, &this_o, proto_key) {
        return None;
    }
    let symv = i
        .extra_protos
        .get("%IntlFallbackSymbol%")?
        .borrow()
        .props
        .get("sym")
        .map(|p| p.value());
    match (symv).map(|__value| (__value).view()) {
        Some(ValueView::Sym(sd)) => {
            this_o.borrow_mut().props.insert(
                crate::interpreter::Interp::sym_key(&sd),
                Property::data(Value::Obj(*instance), false, false, false),
            );
            Some(*this)
        }
        _ => None,
    }
}

// (get_string_option removed; per-service option readers live in each module)

/// GetOption(options, property, "boolean", …).
#[allow(dead_code)]
pub(crate) fn get_boolean_option(
    i: &mut Interp,
    options: &Value,
    property: &str,
    fallback: bool,
) -> Result<bool, Value> {
    let v = ab(i.get_member_key(options, property))?;
    if matches!((v).view(), ValueView::Undefined) {
        return Ok(fallback);
    }
    Ok(i.to_boolean(&v))
}

/// CoerceOptionsToObject: `undefined` → a fresh ordinary object; an object is used as-is; `null`
/// throws; other primitives yield an empty object (their reads fall through to option defaults).
pub(crate) fn coerce_options(i: &mut Interp, options: &Value) -> Result<Value, Value> {
    match &(options).view() {
        // CoerceOptionsToObject(undefined) = OrdinaryObjectCreate(null): a null-prototype object, so
        // option reads never observe Object.prototype pollution.
        ValueView::Undefined => Ok(Value::Obj(Object::new(None))),
        ValueView::Obj(_) => Ok(*options),
        ValueView::Null => Err(i.make_error("TypeError", "options cannot be null")),
        // CoerceOptionsToObject: a primitive is ToObject-boxed (so inherited properties are read).
        _ => Ok(crate::builtins::box_primitive_pub(i, *options)),
    }
}

/// GetOptionsObject: `undefined` → a fresh null-prototype object; an object is used as-is; any other
/// value is a TypeError. Used by the newer services (ListFormat, PluralRules, …).
pub(crate) fn get_options_object(i: &mut Interp, options: &Value) -> Result<Value, Value> {
    match &(options).view() {
        ValueView::Undefined => Ok(Value::Obj(Object::new(None))),
        ValueView::Obj(_) => Ok(*options),
        _ => Err(i.make_error("TypeError", "options must be an object or undefined")),
    }
}

/// ToLength on an already-evaluated value.
pub(crate) fn to_length(i: &mut Interp, v: &Value) -> Result<usize, Value> {
    let n = ab(i.to_number(v))?;
    if n.is_nan() || n <= 0.0 {
        return Ok(0);
    }
    Ok(n.min(9007199254740991.0) as usize)
}

/// Install a getter-only accessor on `proto`.
pub(crate) fn def_getter(it: &mut Interp, proto: &Gc, name: &str, f: crate::value::NativeFn) {
    let g = it.make_native(&format!("get {name}"), 0, f);
    proto.borrow_mut().props.insert(
        name,
        Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
    );
}

#[allow(dead_code)]
pub(crate) fn tag_of_str(v: &Value) -> Option<Rc<str>> {
    match &(v).view() {
        ValueView::Str(s) => Some((*s).into()),
        _ => None,
    }
}
