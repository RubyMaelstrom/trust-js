//! `Intl.RelativeTimeFormat` with CLDR 48 phrases and plural patterns for every locale Lumen
//! advertises.

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
};
use super::{ab, arg, canonicalize_locale_list, coerce_options, make_service};
use crate::interpreter::Interp;
use crate::value::{set_builtin, set_data, Gc, Value};

pub fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "RelativeTimeFormat", 0, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "format", 2, |i, this, args| {
        format(i, &this, &arg(args, 0), &arg(args, 1), false)
    });
    it.def_method(&proto, "formatToParts", 2, |i, this, args| {
        format(i, &this, &arg(args, 0), &arg(args, 1), true)
    });
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
}

fn construct(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    if !i.constructing {
        return Err(i.make_error("TypeError", "Intl.RelativeTimeFormat requires 'new'"));
    }
    let requested = canonicalize_locale_list(i, &arg(args, 0))?;
    let options = coerce_options(i, &arg(args, 1))?;
    read_locale_matcher(i, &options)?;
    // ECMA-402 option access order: numberingSystem, style, numeric.
    let numbering = get_option(i, &options, "numberingSystem", &[], None)?;
    if let Some(numbering) = &numbering {
        if !numbering.split('-').all(|part| {
            (3..=8).contains(&part.len()) && part.bytes().all(|b| b.is_ascii_alphanumeric())
        }) {
            return Err(i.make_error(
                "RangeError",
                format!("invalid numberingSystem: {numbering}"),
            ));
        }
    }
    let style = get_option(
        i,
        &options,
        "style",
        &["long", "short", "narrow"],
        Some("long"),
    )?
    .unwrap();
    let numeric = get_option(i, &options, "numeric", &["always", "auto"], Some("always"))?.unwrap();
    let (locale, numbering) = super::service::resolve_locale_nu(&requested, numbering.as_deref());

    let object = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.RelativeTimeFormat")? {
        object.borrow_mut().proto = Some(proto);
    }
    set_builtin(&object, "#\u{0}rtf", Value::Bool(true));
    set_builtin(&object, "#\u{0}rtf_locale", Value::lstr(locale));
    set_builtin(&object, "#\u{0}rtf_numeric", Value::lstr(numeric));
    set_builtin(&object, "#\u{0}rtf_style", Value::lstr(style));
    set_builtin(&object, "#\u{0}rtf_nu", Value::lstr(numbering));
    Ok(Value::Obj(object))
}

fn singular(unit: &str) -> Option<&'static str> {
    match unit {
        "year" | "years" => Some("year"),
        "quarter" | "quarters" => Some("quarter"),
        "month" | "months" => Some("month"),
        "week" | "weeks" => Some("week"),
        "day" | "days" => Some("day"),
        "hour" | "hours" => Some("hour"),
        "minute" | "minutes" => Some("minute"),
        "second" | "seconds" => Some("second"),
        _ => None,
    }
}

fn string_slot(object: &Gc, key: &str, fallback: &str) -> String {
    match object
        .borrow()
        .props
        .get(key)
        .map(|property| property.value())
    {
        Some(Value::Str(value)) => value.to_string(),
        _ => fallback.to_string(),
    }
}

fn literal_result(i: &mut Interp, value: &str, to_parts: bool) -> Value {
    if !to_parts {
        return Value::lstr(value.to_string());
    }
    let part = i.new_object();
    set_data(&part, "type", Value::lstr("literal"));
    set_data(&part, "value", Value::lstr(value.to_string()));
    i.make_array(vec![Value::Obj(part)])
}

fn format(
    i: &mut Interp,
    this: &Value,
    value: &Value,
    unit: &Value,
    to_parts: bool,
) -> Result<Value, Value> {
    let object = brand_slot(i, this, "#\u{0}rtf")?;
    let number = ab(i.to_number(value))?;
    if !number.is_finite() {
        return Err(i.make_error("RangeError", "value must be finite"));
    }
    let unit_string = ab(i.to_string(unit))?.to_string();
    let unit = singular(&unit_string)
        .ok_or_else(|| i.make_error("RangeError", format!("invalid unit: {unit_string}")))?;
    let locale = string_slot(&object, "#\u{0}rtf_locale", "en-US");
    let language = super::service::data_locale(&locale);
    let style = string_slot(&object, "#\u{0}rtf_style", "long");
    let numeric = string_slot(&object, "#\u{0}rtf_numeric", "always");

    // PartitionRelativeTimePattern uses ToString(value) as the automatic phrase key. The CLDR keys
    // are small integers, so a fractional value must not truncate into an automatic phrase.
    if numeric == "auto"
        && number.fract() == 0.0
        && number >= i8::MIN as f64
        && number <= i8::MAX as f64
    {
        if let Some(phrase) = crate::cldr_relative_time::auto(language, unit, &style, number as i8)
        {
            return Ok(literal_result(i, phrase, to_parts));
        }
    }

    let past = number.is_sign_negative(); // includes -0, as required by ECMA-402.
    let magnitude = number.abs();
    let category = crate::intl::data::plural_cardinal(language, magnitude, 0);
    let tense = if past { "past" } else { "future" };
    let pattern = crate::cldr_relative_time::pattern(language, unit, &style, tense, category)
        .or_else(|| crate::cldr_relative_time::pattern("en", unit, &style, tense, category))
        .expect("generated English relative-time fallback");
    let Some(marker) = pattern.find("{0}") else {
        // Some CLDR categories intentionally spell out the quantity (for example Arabic one/two).
        return Ok(literal_result(i, pattern, to_parts));
    };

    // ECMA-402's internal NumberFormat hides the sign and supplies both the displayed string and
    // the exact part records spliced by MakePartsList.
    let numbering = string_slot(&object, "#\u{0}rtf_nu", "latn");
    let nf_options = i.new_object();
    set_data(&nf_options, "numberingSystem", Value::lstr(numbering));
    set_data(&nf_options, "signDisplay", Value::lstr("never"));
    let number_format = new_service(i, "NumberFormat", &locale, nf_options)?;
    let format = ab(i.get_member(&number_format, "format"))?;
    let formatted = match ab(i.call(format, number_format.clone(), &[Value::Num(number)]))? {
        Value::Str(value) => value.to_string(),
        _ => magnitude.to_string(),
    };
    if !to_parts {
        return Ok(Value::lstr(format!(
            "{}{}{}",
            &pattern[..marker],
            formatted,
            &pattern[marker + 3..]
        )));
    }

    let mut output = Vec::new();
    let push_literal = |i: &mut Interp, output: &mut Vec<Value>, text: &str| {
        if text.is_empty() {
            return;
        }
        let part = i.new_object();
        set_data(&part, "type", Value::lstr("literal"));
        set_data(&part, "value", Value::lstr(text.to_string()));
        output.push(Value::Obj(part));
    };
    push_literal(i, &mut output, &pattern[..marker]);
    let format_to_parts = ab(i.get_member(&number_format, "formatToParts"))?;
    let number_parts = ab(i.call(format_to_parts, number_format, &[Value::Num(number)]))?;
    let length = match ab(i.get_member(&number_parts, "length"))? {
        Value::Num(value) => value as usize,
        _ => 0,
    };
    for index in 0..length {
        let source = ab(i.get_member(&number_parts, &index.to_string()))?;
        let part = i.new_object();
        set_data(&part, "type", ab(i.get_member(&source, "type"))?);
        set_data(&part, "value", ab(i.get_member(&source, "value"))?);
        set_data(&part, "unit", Value::lstr(unit));
        output.push(Value::Obj(part));
    }
    push_literal(i, &mut output, &pattern[marker + 3..]);
    Ok(i.make_array(output))
}

fn new_service(i: &mut Interp, service: &str, locale: &str, options: Gc) -> Result<Value, Value> {
    let constructor = crate::builtins::intl_service(i, service)?;
    ab(i.construct(
        constructor,
        &[Value::lstr(locale.to_string()), Value::Obj(options)],
    ))
}

fn resolved_options(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let object = brand_slot(i, &this, "#\u{0}rtf")?;
    let get = |key: &str| {
        object
            .borrow()
            .props
            .get(key)
            .map(|property| property.value())
            .unwrap_or(Value::Undefined)
    };
    let result = i.new_object();
    set_data(&result, "locale", get("#\u{0}rtf_locale"));
    set_data(&result, "style", get("#\u{0}rtf_style"));
    set_data(&result, "numeric", get("#\u{0}rtf_numeric"));
    set_data(&result, "numberingSystem", get("#\u{0}rtf_nu"));
    Ok(Value::Obj(result))
}
