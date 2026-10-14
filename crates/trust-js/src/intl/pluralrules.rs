//! `Intl.PluralRules` following ECMA-402 ResolvePlural with CLDR 48 cardinal, ordinal, and range
//! rules for every locale TRust JS advertises.

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
    resolve_locale,
};
use super::{
    ab, arg, canonicalize_locale_list, get_options_object as coerce_options, make_service,
};
use crate::interpreter::Interp;
use crate::value::{set_data, set_internal_slot, Gc, Value};

pub fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "PluralRules", 0, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "select", 1, |i, this, args| {
        select(i, &this, &arg(args, 0))
    });
    it.def_method(&proto, "selectRange", 2, |i, this, args| {
        select_range(i, &this, &arg(args, 0), &arg(args, 1))
    });
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
}

fn construct(i: &mut Interp, _this: Value, args: &[Value]) -> Result<Value, Value> {
    if !i.constructing {
        return Err(i.make_error("TypeError", "Intl.PluralRules requires 'new'"));
    }
    let requested = canonicalize_locale_list(i, &arg(args, 0))?;
    let options = coerce_options(i, &arg(args, 1))?;
    read_locale_matcher(i, &options)?;
    let kind = get_option(
        i,
        &options,
        "type",
        &["cardinal", "ordinal"],
        Some("cardinal"),
    )?
    .unwrap();
    let notation = get_option(
        i,
        &options,
        "notation",
        &["standard", "scientific", "engineering", "compact"],
        Some("standard"),
    )?
    .unwrap();
    // ECMA-402 reads compactDisplay even when notation is not compact; it is only omitted from
    // resolvedOptions in that case.
    let compact_display = get_option(
        i,
        &options,
        "compactDisplay",
        &["short", "long"],
        Some("short"),
    )?
    .unwrap();

    let raw_digits = super::numberformat::read_raw_digits(i, &options)?;
    let rounding_increment = {
        let value = ab(i.get_member_key(&options, "roundingIncrement"))?;
        if matches!(value, Value::Undefined) {
            1
        } else {
            let number = ab(i.to_number(&value))?;
            const ALLOWED: &[u32] = &[
                1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
            ];
            if !number.is_finite() || number.fract() != 0.0 || !ALLOWED.contains(&(number as u32)) {
                return Err(i.make_error("RangeError", "invalid roundingIncrement"));
            }
            number as u32
        }
    };
    let rounding_mode = get_option(
        i,
        &options,
        "roundingMode",
        &[
            "ceil",
            "floor",
            "expand",
            "trunc",
            "halfCeil",
            "halfFloor",
            "halfExpand",
            "halfTrunc",
            "halfEven",
        ],
        Some("halfExpand"),
    )?
    .unwrap();
    let rounding_priority = get_option(
        i,
        &options,
        "roundingPriority",
        &["auto", "morePrecision", "lessPrecision"],
        Some("auto"),
    )?
    .unwrap();
    let digits = super::numberformat::interpret_digits(
        i,
        &raw_digits,
        "decimal",
        &notation,
        &rounding_priority,
        0,
        rounding_increment,
    )?;
    if rounding_increment != 1 {
        if digits.rounding_type != "fraction" {
            return Err(i.make_error(
                "TypeError",
                "roundingIncrement requires fraction-digits rounding",
            ));
        }
        if digits.min_frac != digits.max_frac {
            return Err(i.make_error(
                "RangeError",
                "maximumFractionDigits must equal minimumFractionDigits with roundingIncrement",
            ));
        }
    }
    let trailing_zero_display = get_option(
        i,
        &options,
        "trailingZeroDisplay",
        &["auto", "stripIfInteger"],
        Some("auto"),
    )?
    .unwrap();
    let resolved = resolve_locale(i, &requested, &[]);

    let object = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.PluralRules")? {
        object.borrow_mut().proto = Some(proto);
    }
    set_internal_slot(&object, "#\u{0}pr", Value::Bool(true));
    set_internal_slot(&object, "#\u{0}pr_locale", Value::lstr(resolved.locale));
    set_internal_slot(&object, "#\u{0}pr_type", Value::lstr(kind));
    set_internal_slot(&object, "#\u{0}pr_notation", Value::lstr(notation));
    set_internal_slot(
        &object,
        "#\u{0}pr_compactdisplay",
        Value::lstr(compact_display),
    );
    set_internal_slot(
        &object,
        "#\u{0}pr_minint",
        Value::Num(digits.min_int as f64),
    );
    set_internal_slot(
        &object,
        "#\u{0}pr_minfrac",
        Value::Num(digits.min_frac as f64),
    );
    set_internal_slot(
        &object,
        "#\u{0}pr_maxfrac",
        Value::Num(digits.max_frac as f64),
    );
    if let Some(value) = digits.min_sig {
        set_internal_slot(&object, "#\u{0}pr_minsig", Value::Num(value as f64));
    }
    if let Some(value) = digits.max_sig {
        set_internal_slot(&object, "#\u{0}pr_maxsig", Value::Num(value as f64));
    }
    set_internal_slot(
        &object,
        "#\u{0}pr_roundingincrement",
        Value::Num(rounding_increment as f64),
    );
    set_internal_slot(&object, "#\u{0}pr_roundingmode", Value::lstr(rounding_mode));
    set_internal_slot(
        &object,
        "#\u{0}pr_roundingpriority",
        Value::lstr(rounding_priority),
    );
    set_internal_slot(
        &object,
        "#\u{0}pr_roundingtype",
        Value::lstr(digits.rounding_type),
    );
    set_internal_slot(
        &object,
        "#\u{0}pr_trailingzero",
        Value::lstr(trailing_zero_display),
    );
    Ok(Value::Obj(object))
}

fn string_slot(object: &Gc, key: &str) -> String {
    match object
        .borrow()
        .internal_slots()
        .get(key)
        .map(|property| property.value())
    {
        Some(Value::Str(value)) => value.to_string(),
        _ => String::new(),
    }
}

fn number_slot(object: &Gc, key: &str) -> Option<u32> {
    match object
        .borrow()
        .internal_slots()
        .get(key)
        .map(|property| property.value())
    {
        Some(Value::Num(value)) => Some(value as u32),
        _ => None,
    }
}

fn compact_exponent(decimal: &str) -> u32 {
    let decimal = decimal.strip_prefix(['+', '-']).unwrap_or(decimal);
    let integer = decimal
        .split_once('.')
        .map_or(decimal, |(integer, _)| integer);
    let significant = integer.trim_start_matches('0');
    if significant.len() < 4 {
        0
    } else {
        ((significant.len() - 1) / 3 * 3) as u32
    }
}

/// ECMA-402 ResolvePlural: round first through FormatNumericToString, then apply the locale's
/// cardinal/ordinal rule to that decimal string. This ordering makes visible trailing zeros and
/// digit options participate in CLDR's v/f/t operands.
fn resolve_plural(
    i: &mut Interp,
    object: &Gc,
    value: &Value,
) -> Result<(&'static str, String), Value> {
    let decimal = match super::numberformat::to_intl_mv(i, value)? {
        super::numberformat::IntlMv::Finite(decimal) => decimal,
        super::numberformat::IntlMv::NotANumber => return Ok(("other", "NaN".to_string())),
        super::numberformat::IntlMv::PositiveInfinity => {
            return Ok(("other", "Infinity".to_string()))
        }
        super::numberformat::IntlMv::NegativeInfinity => {
            return Ok(("other", "-Infinity".to_string()))
        }
    };

    let min_int = number_slot(object, "#\u{0}pr_minint").unwrap_or(1);
    let min_frac = number_slot(object, "#\u{0}pr_minfrac").unwrap_or(0);
    let max_frac = number_slot(object, "#\u{0}pr_maxfrac").unwrap_or(3);
    let min_sig = number_slot(object, "#\u{0}pr_minsig");
    let max_sig = number_slot(object, "#\u{0}pr_maxsig");
    let increment = number_slot(object, "#\u{0}pr_roundingincrement").unwrap_or(1);
    let mode = string_slot(object, "#\u{0}pr_roundingmode");
    let rounding_type = string_slot(object, "#\u{0}pr_roundingtype");
    let notation = string_slot(object, "#\u{0}pr_notation");
    // FormatNumericToString is notation-independent; the already-resolved compact digit
    // defaults still control rounding.
    let decimal = super::numberformat::exact_magnitude_options(
        &decimal,
        min_int,
        min_frac,
        max_frac,
        min_sig,
        max_sig,
        increment,
        &mode,
        &rounding_type,
    );
    let exponent = if notation == "compact" {
        compact_exponent(&decimal)
    } else {
        0
    };
    let locale = string_slot(object, "#\u{0}pr_locale");
    let language = super::service::data_locale(&locale);
    let category = if string_slot(object, "#\u{0}pr_type") == "ordinal" {
        crate::cldr_plurals::select_ordinal(language, &decimal, exponent)
    } else {
        crate::cldr_plurals::select_cardinal(language, &decimal, exponent)
    };
    Ok((category, decimal))
}

fn select(i: &mut Interp, this: &Value, value: &Value) -> Result<Value, Value> {
    let object = brand_slot(i, this, "#\u{0}pr")?;
    let (category, _) = resolve_plural(i, &object, value)?;
    Ok(Value::lstr(category))
}

fn select_range(i: &mut Interp, this: &Value, start: &Value, end: &Value) -> Result<Value, Value> {
    let object = brand_slot(i, this, "#\u{0}pr")?;
    if matches!(start, Value::Undefined) || matches!(end, Value::Undefined) {
        return Err(i.make_error("TypeError", "selectRange requires two values"));
    }
    let (start_category, start_decimal) = resolve_plural(i, &object, start)?;
    let (end_category, end_decimal) = resolve_plural(i, &object, end)?;
    if start_decimal == "NaN" || end_decimal == "NaN" {
        return Err(i.make_error("RangeError", "selectRange arguments must not be NaN"));
    }
    // ResolvePluralRange returns the first category immediately when both formatted strings are
    // equal; only distinct formatted endpoints enter the CLDR plural-range table.
    if start_decimal == end_decimal {
        return Ok(Value::lstr(start_category));
    }
    let locale = string_slot(&object, "#\u{0}pr_locale");
    let language = super::service::data_locale(&locale);
    Ok(Value::lstr(crate::cldr_plurals::select_range(
        language,
        start_category,
        end_category,
    )))
}

fn resolved_options(i: &mut Interp, this: Value, _args: &[Value]) -> Result<Value, Value> {
    let object = brand_slot(i, &this, "#\u{0}pr")?;
    let result = i.new_object();
    let get = |key: &str| {
        object
            .borrow()
            .internal_slots()
            .get(key)
            .map(|property| property.value())
            .unwrap_or(Value::Undefined)
    };
    set_data(&result, "locale", get("#\u{0}pr_locale"));
    set_data(&result, "type", get("#\u{0}pr_type"));
    set_data(&result, "notation", get("#\u{0}pr_notation"));
    if matches!(get("#\u{0}pr_notation"), Value::Str(value) if &*value == "compact") {
        set_data(&result, "compactDisplay", get("#\u{0}pr_compactdisplay"));
    }
    set_data(&result, "minimumIntegerDigits", get("#\u{0}pr_minint"));
    set_data(&result, "minimumFractionDigits", get("#\u{0}pr_minfrac"));
    set_data(&result, "maximumFractionDigits", get("#\u{0}pr_maxfrac"));
    if !matches!(get("#\u{0}pr_minsig"), Value::Undefined) {
        set_data(&result, "minimumSignificantDigits", get("#\u{0}pr_minsig"));
        set_data(&result, "maximumSignificantDigits", get("#\u{0}pr_maxsig"));
    }
    let locale = string_slot(&object, "#\u{0}pr_locale");
    let language = super::service::data_locale(&locale);
    let kind = string_slot(&object, "#\u{0}pr_type");
    let categories = crate::cldr_plurals::categories(language, &kind)
        .iter()
        .map(|category| Value::lstr(*category))
        .collect();
    set_data(&result, "pluralCategories", i.make_array(categories));
    set_data(
        &result,
        "roundingIncrement",
        get("#\u{0}pr_roundingincrement"),
    );
    set_data(&result, "roundingMode", get("#\u{0}pr_roundingmode"));
    set_data(
        &result,
        "roundingPriority",
        get("#\u{0}pr_roundingpriority"),
    );
    set_data(&result, "trailingZeroDisplay", get("#\u{0}pr_trailingzero"));
    Ok(Value::Obj(result))
}
