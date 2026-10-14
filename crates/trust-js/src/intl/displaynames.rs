//! `Intl.DisplayNames` backed by CLDR 48 names for every locale TRust JS advertises.

use super::service::{
    brand_slot, get_option, install_supported_locales, read_locale_matcher, resolve_locale,
};
use super::{
    ab, arg, canonicalize_locale_list, get_options_object as coerce_options, make_service,
};
use crate::interpreter::Interp;
use crate::intl::tags;
use crate::value::{set_data, set_internal_slot, Gc, Value};

pub fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "DisplayNames", 2, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "of", 1, |i, this, a| of(i, &this, &arg(a, 0)));
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
}

fn construct(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    if !i.constructing {
        return Err(i.make_error("TypeError", "Intl.DisplayNames requires 'new'"));
    }
    // OrdinaryCreateFromConstructor first: a poisoned newTarget.prototype getter fires before
    // any locale/options validation.
    let obj = crate::builtins::new_from_ctor(i, "Intl.DisplayNames")?;
    let requested = canonicalize_locale_list(i, &arg(a, 0))?;
    // options is required and must be an object.
    let opt_arg = arg(a, 1);
    if matches!(opt_arg, Value::Undefined) {
        return Err(i.make_error("TypeError", "options is required for Intl.DisplayNames"));
    }
    let options = coerce_options(i, &opt_arg)?;
    read_locale_matcher(i, &options)?;
    let style = get_option(
        i,
        &options,
        "style",
        &["narrow", "short", "long"],
        Some("long"),
    )?
    .unwrap();
    let kind = get_option(
        i,
        &options,
        "type",
        &[
            "language",
            "region",
            "script",
            "currency",
            "calendar",
            "dateTimeField",
        ],
        None,
    )?;
    let kind = kind.ok_or_else(|| i.make_error("TypeError", "type option is required"))?;
    let fallback = get_option(i, &options, "fallback", &["code", "none"], Some("code"))?.unwrap();
    let language_display = get_option(
        i,
        &options,
        "languageDisplay",
        &["dialect", "standard"],
        Some("dialect"),
    )?
    .unwrap();
    let resolved = resolve_locale(i, &requested, &[]);

    set_internal_slot(&obj, "#\u{0}dn", Value::Bool(true));
    set_internal_slot(&obj, "#\u{0}dn_locale", Value::lstr(resolved.locale));
    set_internal_slot(&obj, "#\u{0}dn_style", Value::lstr(style));
    let is_language = kind == "language";
    set_internal_slot(&obj, "#\u{0}dn_type", Value::lstr(kind));
    set_internal_slot(&obj, "#\u{0}dn_fallback", Value::lstr(fallback));
    if is_language {
        set_internal_slot(&obj, "#\u{0}dn_langdisplay", Value::lstr(language_display));
    }
    Ok(Value::Obj(obj))
}

fn string_slot(object: &Gc, key: &str, fallback: &str) -> String {
    match object
        .borrow()
        .internal_slots()
        .get(key)
        .map(|property| property.value())
    {
        Some(Value::Str(value)) => value.to_string(),
        _ => fallback.to_string(),
    }
}

fn of(i: &mut Interp, this: &Value, code: &Value) -> Result<Value, Value> {
    let o = brand_slot(i, this, "#\u{0}dn")?;
    let kind = string_slot(&o, "#\u{0}dn_type", "");
    let fallback = string_slot(&o, "#\u{0}dn_fallback", "code");
    let s = ab(i.to_string(code))?.to_string();
    // Validate the code per type.
    let canonical = match kind.as_str() {
        "language" => {
            // DisplayNames `language` requires a `unicode_language_id` (language[-script][-region]
            // [-variants]) — a tag carrying extensions or singleton subtags is a RangeError.
            let is_language_id = tags::parse(&s)
                .map(|t| {
                    t.unicode.is_none()
                        && t.transform.is_none()
                        && t.other_ext.is_empty()
                        && t.private.is_empty()
                })
                .unwrap_or(false);
            if !is_language_id {
                return Err(i.make_error("RangeError", format!("invalid language code: {s}")));
            }
            tags::canonicalize_language_tag(&s).unwrap_or(s.clone())
        }
        "region" => {
            if !(s.len() == 2 && s.bytes().all(|b| b.is_ascii_alphabetic())
                || s.len() == 3 && s.bytes().all(|b| b.is_ascii_digit()))
            {
                return Err(i.make_error("RangeError", format!("invalid region code: {s}")));
            }
            s.to_uppercase()
        }
        "script" => {
            if s.len() != 4 || !s.bytes().all(|b| b.is_ascii_alphabetic()) {
                return Err(i.make_error("RangeError", format!("invalid script code: {s}")));
            }
            let mut c = s.to_lowercase();
            c[..1].make_ascii_uppercase();
            c
        }
        "currency" => {
            if s.len() != 3 || !s.bytes().all(|b| b.is_ascii_alphabetic()) {
                return Err(i.make_error("RangeError", format!("invalid currency code: {s}")));
            }
            s.to_uppercase()
        }
        "calendar" => {
            let ok = !s.is_empty()
                && s.split('-').all(|p| {
                    p.len() >= 3 && p.len() <= 8 && p.bytes().all(|b| b.is_ascii_alphanumeric())
                });
            if !ok {
                return Err(i.make_error("RangeError", format!("invalid calendar code: {s}")));
            }
            s.to_ascii_lowercase()
        }
        "dateTimeField" => {
            const FIELDS: &[&str] = &[
                "era",
                "year",
                "quarter",
                "month",
                "weekOfYear",
                "weekday",
                "day",
                "dayPeriod",
                "hour",
                "minute",
                "second",
                "timeZoneName",
            ];
            if !FIELDS.contains(&s.as_str()) {
                return Err(i.make_error("RangeError", format!("invalid dateTimeField: {s}")));
            }
            s.clone()
        }
        _ => s.clone(),
    };
    let locale = string_slot(&o, "#\u{0}dn_locale", "en-US");
    let language = super::service::data_locale(&locale);
    let style = string_slot(&o, "#\u{0}dn_style", "long");
    let name = if kind == "language" {
        let language_display = string_slot(&o, "#\u{0}dn_langdisplay", "dialect");
        display_language(language, &style, &language_display, &canonical)
    } else {
        crate::cldr_display_names::name(language, &kind, &style, &canonical).map(str::to_string)
    };
    match name {
        Some(name) => Ok(Value::lstr(name)),
        None => {
            if fallback == "code" {
                Ok(Value::lstr(canonical))
            } else {
                Ok(Value::Undefined)
            }
        }
    }
}

fn display_language(
    locale: &str,
    style: &str,
    language_display: &str,
    code: &str,
) -> Option<String> {
    let tag = tags::parse(code)?;

    // UTS #35 Locale Display Name Algorithm: dialect mode first takes the longest available
    // compound language match. Direct full matches cover the common path without any assembly.
    if language_display == "dialect" {
        if let Some(name) = crate::cldr_display_names::name(locale, "language", style, code) {
            return Some(name.to_string());
        }
    }

    let mut used_script = false;
    let mut used_region = false;
    let mut used_variant: Option<usize> = None;
    let mut base = None;

    if language_display == "dialect" {
        let mut consider =
            |candidate: String, script: bool, region: bool, variant: Option<usize>| {
                if base.is_none() {
                    if let Some(name) =
                        crate::cldr_display_names::name(locale, "language", style, &candidate)
                    {
                        base = Some(name.to_string());
                        used_script = script;
                        used_region = region;
                        used_variant = variant;
                    }
                }
            };

        // CLDR compound language records primarily specialize language+script+region,
        // language+region, language+script, or language+variant. Check in descending number of
        // consumed fields, without an exponential subset search for adversarial variant lists.
        if !tag.script.is_empty() && !tag.region.is_empty() {
            consider(
                format!("{}-{}-{}", tag.language, tag.script, tag.region),
                true,
                true,
                None,
            );
        }
        if !tag.region.is_empty() {
            consider(
                format!("{}-{}", tag.language, tag.region),
                false,
                true,
                None,
            );
        }
        if !tag.script.is_empty() {
            consider(
                format!("{}-{}", tag.language, tag.script),
                true,
                false,
                None,
            );
        }
        for (index, variant) in tag.variants.iter().enumerate() {
            consider(
                format!("{}-{}", tag.language, variant),
                false,
                false,
                Some(index),
            );
        }
    }

    let base = base.or_else(|| {
        crate::cldr_display_names::name(locale, "language", style, &tag.language)
            .map(str::to_string)
    })?;
    let mut qualifiers = Vec::new();
    if !tag.script.is_empty() && !used_script {
        qualifiers.push(
            crate::cldr_display_names::name(locale, "script", style, &tag.script)
                .unwrap_or(&tag.script)
                .to_string(),
        );
    }
    if !tag.region.is_empty() && !used_region {
        qualifiers.push(
            crate::cldr_display_names::name(locale, "region", style, &tag.region)
                .unwrap_or(&tag.region)
                .to_string(),
        );
    }
    for (index, variant) in tag.variants.iter().enumerate() {
        if used_variant != Some(index) {
            qualifiers.push(
                crate::cldr_display_names::name(locale, "variant", style, variant)
                    .unwrap_or(variant)
                    .to_string(),
            );
        }
    }
    if qualifiers.is_empty() {
        return Some(base);
    }

    let (pattern, separator) = crate::cldr_display_names::locale_patterns(locale);
    let mut joined = qualifiers.remove(0);
    for qualifier in qualifiers {
        joined = apply_pattern(separator, &joined, &qualifier);
    }
    Some(apply_pattern(pattern, &base, &joined))
}

fn apply_pattern(pattern: &str, first: &str, second: &str) -> String {
    pattern.replace("{0}", first).replace("{1}", second)
}

fn resolved_options(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = brand_slot(i, &this, "#\u{0}dn")?;
    let get = |k: &str| {
        o.borrow()
            .internal_slots()
            .get(k)
            .map(|p| p.value())
            .unwrap_or(Value::Undefined)
    };
    let res = i.new_object();
    set_data(&res, "locale", get("#\u{0}dn_locale"));
    set_data(&res, "style", get("#\u{0}dn_style"));
    set_data(&res, "type", get("#\u{0}dn_type"));
    set_data(&res, "fallback", get("#\u{0}dn_fallback"));
    let language_display = get("#\u{0}dn_langdisplay");
    if !matches!(language_display, Value::Undefined) {
        set_data(&res, "languageDisplay", language_display);
    }
    Ok(Value::Obj(res))
}
