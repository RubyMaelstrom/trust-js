//! ECMA-402 `Intl.DateTimeFormat`, backed by generated CLDR locale patterns and calendar names.

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
    resolve_locale,
};
use super::{ab, arg, canonicalize_locale_list, coerce_options, make_service};
use crate::interpreter::Interp;
use crate::value::{set_data, set_internal_slot, Gc, Value};

/// Days since the Unix epoch for a proleptic-Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// The epoch-milliseconds a Temporal value formats at (ISO calendar, UTC).
fn temporal_to_ms(t: &crate::temporal::Temporal) -> f64 {
    use crate::temporal::Temporal as T;
    let date_ms = |d: &crate::temporal::IsoDate| {
        days_from_civil(d.year, d.month as i64, d.day as i64) as f64 * 86_400_000.0
    };
    let time_ms = |t: &crate::temporal::IsoTime| {
        (t.hour as f64) * 3_600_000.0
            + (t.minute as f64) * 60_000.0
            + (t.second as f64) * 1000.0
            + (t.ms as f64)
    };
    match t {
        T::Date(d) | T::YearMonth(d) | T::MonthDay(d) => date_ms(d),
        T::DateTime(d, tm) => date_ms(d) + time_ms(tm),
        T::Time(tm) => time_ms(tm),
        T::Instant(ns) => (*ns / 1_000_000) as f64,
        // A ZonedDateTime formats at its *local* wall-clock time (epoch + offset), since our
        // DateTimeFormat renders components in UTC.
        T::Zoned {
            epoch_ns,
            offset_ns,
            ..
        } => ((*epoch_ns + *offset_ns as i128) / 1_000_000) as f64,
        T::Duration(_) => 0.0,
    }
}

pub(crate) fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "DateTimeFormat", 0, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "formatToParts", 1, |i, this, a| {
        let o = brand_slot(i, &this, "#\u{0}dtf")?;
        let (ms, kind) = dtf_ms_kind(i, &o, &arg(a, 0))?;
        let nu = dtf_nu(&o);
        let parts = build_parts(&o, ms, kind);
        let arr: Vec<Value> = parts
            .into_iter()
            .map(|(t, v)| {
                let ob = i.new_object();
                set_data(&ob, "type", Value::lstr(t));
                // Localize digits to the numbering system (name/literal parts have no ASCII digits).
                set_data(
                    &ob,
                    "value",
                    Value::lstr(crate::intl::numberformat::xlate_digits(&v, &nu)),
                );
                Value::Obj(ob)
            })
            .collect();
        Ok(i.make_array(arr))
    });
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
    it.def_method(&proto, "formatRange", 2, |i, this, a| {
        let o = brand_slot(i, &this, "#\u{0}dtf")?;
        let (s, e, kind) = range_dates(i, &o, &arg(a, 0), &arg(a, 1))?;
        let out: String = build_range_parts(&o, s, e, kind)
            .into_iter()
            .map(|(_, value, _)| value)
            .collect();
        Ok(Value::lstr(crate::intl::numberformat::xlate_digits(
            &out,
            &dtf_nu(&o),
        )))
    });
    it.def_method(&proto, "formatRangeToParts", 2, |i, this, a| {
        let o = brand_slot(i, &this, "#\u{0}dtf")?;
        let (s, e, kind) = range_dates(i, &o, &arg(a, 0), &arg(a, 1))?;
        let nu = dtf_nu(&o);
        let mut arr: Vec<Value> = Vec::new();
        let emit = |i: &mut Interp, arr: &mut Vec<Value>, ty: &str, val: &str, src: &str| {
            let ob = i.new_object();
            set_data(&ob, "type", Value::lstr(ty));
            set_data(
                &ob,
                "value",
                Value::lstr(crate::intl::numberformat::xlate_digits(val, &nu)),
            );
            set_data(&ob, "source", Value::lstr(src));
            arr.push(Value::Obj(ob));
        };
        for (kind, value, source) in build_range_parts(&o, s, e, kind) {
            emit(i, &mut arr, kind, &value, source);
        }
        Ok(i.make_array(arr))
    });
    install_format_getter(it, &proto);
}

/// Resolve both range endpoints (numbers or Temporal objects) to epoch-ms + kind. The two endpoints
/// must be the same kind; NaN or start > end throws.
fn range_dates(i: &mut Interp, o: &Gc, a: &Value, b: &Value) -> Result<(f64, f64, u8), Value> {
    if matches!(a, Value::Undefined) || matches!(b, Value::Undefined) {
        return Err(i.make_error("TypeError", "formatRange requires two dates"));
    }
    // ToDateTimeFormattable each endpoint IN ORDER: a Temporal value keeps its type, a non-Temporal
    // is ToNumber-coerced now (its valueOf side-effects must run before the kind-mismatch check).
    let ta = range_type_tag(i, a);
    let na = if ta == 0 {
        Some(ab(i.to_number(a))?)
    } else {
        None
    };
    let tb = range_type_tag(i, b);
    let nb = if tb == 0 {
        Some(ab(i.to_number(b))?)
    } else {
        None
    };
    // Two Temporal endpoints must share a calendar (RangeError otherwise).
    let cal_of = |i: &Interp, v: &Value| -> Option<String> {
        let ptr = Gc::as_ptr(v.as_obj()?) as usize;
        if i.temporal.contains_key(&ptr) {
            Some(
                i.temporal_cal
                    .get(&ptr)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "iso8601".to_string()),
            )
        } else {
            None
        }
    };
    if let (Some(ca), Some(cb)) = (cal_of(i, a), cal_of(i, b)) {
        if ca != cb {
            return Err(i.make_error(
                "RangeError",
                "formatRange endpoints have different calendars",
            ));
        }
    }
    // The two endpoints must be the same type (checked AFTER coercion, before TimeClip).
    if ta != tb {
        return Err(i.make_error("TypeError", "formatRange endpoints must be the same type"));
    }
    // TimeClip a coerced number; a Temporal reads its own fields (no re-coercion).
    let resolve = |i: &mut Interp, v: &Value, n: Option<f64>| -> Result<(f64, u8), Value> {
        match n {
            Some(x) => {
                if !x.is_finite() || x.abs() > 8.64e15 {
                    return Err(i.make_error("RangeError", "Invalid time value"));
                }
                Ok((x.trunc() + 0.0, 0))
            }
            None => dtf_ms_kind(i, o, v),
        }
    };
    let (s, _) = resolve(i, a, na)?;
    let (e, ks) = resolve(i, b, nb)?;
    Ok((s, e, ks))
}

/// Map a temporal era CODE to its CLDR era index for name lookup (single-era calendars are "0";
/// two-era calendars split by sign; japanese eras have no simple index so fall back to the code).
fn era_cldr_index(cal: &str, code: &str) -> &'static str {
    match code {
        "ce" | "ad" => "1",
        "bce" | "bc" => "0",
        "ah" => "0",
        "bh" => "1",
        "roc" | "minguo" => "1",
        "broc" => "0",
        "am" if cal == "coptic" || cal == "ethiopic" => "1",
        _ => "0",
    }
}

/// A type discriminant for formatRange: 0 = non-Temporal (number/Date), else the Temporal variant.
fn range_type_tag(i: &Interp, v: &Value) -> u8 {
    use crate::temporal::Temporal as T;
    if let Some(o) = v.as_obj() {
        if let Some(t) = i.temporal.get(&(Gc::as_ptr(o) as usize)) {
            return match t {
                T::Date(_) => 1,
                T::Time(_) => 2,
                T::DateTime(..) => 3,
                T::YearMonth(_) => 4,
                T::MonthDay(_) => 5,
                T::Instant(_) => 6,
                T::Zoned { .. } => 7,
                T::Duration(_) => 8,
            };
        }
    }
    0
}

fn install_format_getter(it: &mut Interp, proto: &Gc) {
    let g = it.make_native("get format", 0, |i, this, _| {
        let o = crate::intl::brand_slot_legacy(i, &this, "#\u{0}dtf", "Intl.DateTimeFormat")?;
        if let Some(f) = o
            .borrow()
            .internal_slots()
            .get("#\u{0}dtf_bound")
            .map(|p| p.value())
        {
            return Ok(f);
        }
        let f = i.make_native("", 1, |i, that, a| {
            Ok(Value::lstr(do_format(i, &that, &arg(a, 0))?))
        });
        let bound = crate::intl::numberformat::bind_this(i, Value::Obj(f), Value::Obj(o.clone()));
        set_internal_slot(&o, "#\u{0}dtf_bound", bound.clone());
        Ok(bound)
    });
    proto.borrow_mut().props.insert(
        "format",
        crate::value::Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
    );
}

/// A Unicode key type identifier: one or more "-"-joined 3..8 alnum subtags.
fn valid_type_id(s: &str) -> bool {
    !s.is_empty()
        && s.split('-')
            .all(|p| p.len() >= 3 && p.len() <= 8 && p.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// Return the run width of the first requested field in a UTS #35 date pattern. Quoted ASCII
/// letters are literals, and doubled apostrophes do not change quote state.
fn dtf_pattern_width(pattern: &str, fields: &str) -> Option<usize> {
    let mut chars = pattern.chars().peekable();
    let mut quoted = false;
    while let Some(character) = chars.next() {
        if character == '\'' {
            if chars.peek() == Some(&'\'') {
                chars.next();
            } else {
                quoted = !quoted;
            }
            continue;
        }
        if quoted || !fields.contains(character) {
            continue;
        }
        let mut width = 1;
        while chars.peek() == Some(&character) {
            chars.next();
            width += 1;
        }
        return Some(width);
    }
    None
}

/// Populate the internal component values used to compute a style formatter's fields. ECMA-402
/// styles are locale patterns, not a fixed English component preset, so widths must come from the
/// selected CLDR pattern (for example Japanese full dates use numeric `M`, not named `MMMM`).
fn install_style_components(obj: &Gc, pattern: &str) {
    let put_width = |slot: &str, fields: &str, value: fn(usize) -> &'static str| {
        if let Some(width) = dtf_pattern_width(pattern, fields) {
            set_internal_slot(obj, slot, Value::lstr(value(width)));
        }
    };
    put_width("#\u{0}dtfx_weekday", "Eec", |width| match width {
        4 => "long",
        5 => "narrow",
        _ => "short",
    });
    put_width("#\u{0}dtfx_era", "G", |width| match width {
        4 => "long",
        5 => "narrow",
        _ => "short",
    });
    put_width("#\u{0}dtfx_year", "yYuUr", |width| {
        if width == 2 {
            "2-digit"
        } else {
            "numeric"
        }
    });
    put_width("#\u{0}dtfx_month", "ML", |width| match width {
        2 => "2-digit",
        3 => "short",
        4 => "long",
        5 => "narrow",
        _ => "numeric",
    });
    put_width("#\u{0}dtfx_day", "d", |width| {
        if width == 2 {
            "2-digit"
        } else {
            "numeric"
        }
    });
    put_width("#\u{0}dtfx_hour", "hHKk", |width| {
        if width == 2 {
            "2-digit"
        } else {
            "numeric"
        }
    });
    put_width("#\u{0}dtfx_minute", "m", |width| {
        if width == 2 {
            "2-digit"
        } else {
            "numeric"
        }
    });
    put_width("#\u{0}dtfx_second", "s", |width| {
        if width == 2 {
            "2-digit"
        } else {
            "numeric"
        }
    });
    if let Some(width) = dtf_pattern_width(pattern, "zZvVOXx") {
        set_internal_slot(
            obj,
            "#\u{0}dtfx_tzname",
            Value::lstr(if width >= 4 { "long" } else { "short" }),
        );
    }
}

/// Update the month slot when best-fit selection changes between a named and numeric CLDR pattern.
/// This avoids attaching a numeric locale suffix twice and makes `resolvedOptions()` describe the
/// selected format (Japanese `month: "short"`, for example, resolves to numeric here).
fn install_resolved_components(obj: &Gc, pattern: &str) {
    if obj.borrow().internal_slots().contains("#\u{0}dtf_month") {
        if let Some(width) = dtf_pattern_width(pattern, "ML") {
            // The requested width stays in `dtf_month`: the format pattern is matched from it.
            set_internal_slot(
                obj,
                "#\u{0}dtfr_month",
                Value::lstr(match width {
                    2 => "2-digit",
                    3 => "short",
                    4 => "long",
                    5 => "narrow",
                    _ => "numeric",
                }),
            );
        }
    }
}

fn construct(i: &mut Interp, t: Value, a: &[Value]) -> Result<Value, Value> {
    // Legacy service: callable without `new` (returns a fresh instance either way).
    let requested = canonicalize_locale_list(i, &arg(a, 0))?;
    let options = coerce_options(i, &arg(a, 1))?;
    read_locale_matcher(i, &options)?;

    // Spec option read order: calendar, numberingSystem, hour12, hourCycle, timeZone, then the
    // component options, then formatMatcher, dateStyle, timeStyle.
    let calendar = get_option(i, &options, "calendar", &[], None)?.map(|c| {
        // A deprecated calendar id canonicalizes (e.g. ethiopic-amete-alem -> ethioaa).
        let lc = c.to_lowercase();
        crate::intl::tags::canonical_ca(&lc).unwrap_or(lc)
    });
    if let Some(c) = &calendar {
        if !valid_type_id(c) {
            return Err(i.make_error("RangeError", format!("invalid calendar: {c}")));
        }
    }
    // "islamic"/"islamic-rgsa" are not directly available; fall back to a concrete Islamic calendar.
    // Deprecated islamic ids fall back to islamic-civil; any calendar not in AvailableCalendars
    // (a not-yet-supported one like "bangla") falls back to gregory.
    const AVAILABLE_CALENDARS: [&str; 17] = [
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
        "islamicc",
    ];
    // Resolve a calendar VALUE: islamic/islamic-rgsa → islamic-civil, an available id is kept, any
    // other (unsupported) value is None so it is ignored (the -u-ca extension, then gregory, apply).
    let resolve_cal = |c: &str| -> Option<String> {
        if c == "islamic" || c == "islamic-rgsa" {
            Some("islamic-civil".to_string())
        } else if AVAILABLE_CALENDARS.contains(&c) {
            Some(c.to_string())
        } else {
            None
        }
    };
    let calendar = calendar.and_then(|c| resolve_cal(&c));
    let numbering = get_option(i, &options, "numberingSystem", &[], None)?;
    if let Some(ns) = &numbering {
        if !valid_type_id(ns) {
            return Err(i.make_error("RangeError", format!("invalid numberingSystem: {ns}")));
        }
    }
    let hour12 = {
        let v = ab(i.get_member_key(&options, "hour12"))?;
        if matches!(v, Value::Undefined) {
            None
        } else {
            Some(i.to_boolean(&v))
        }
    };
    let hour_cycle = get_option(
        i,
        &options,
        "hourCycle",
        &["h11", "h12", "h23", "h24"],
        None,
    )?;
    let time_zone = {
        let v = ab(i.get_member_key(&options, "timeZone"))?;
        if matches!(v, Value::Undefined) {
            // ECMA-402 #sec-createdatetimeformat: SystemTimeZoneIdentifier().
            crate::builtins::system_time_zone().to_string()
        } else {
            let raw = ab(i.to_string(&v))?.to_string();
            match canonicalize_time_zone(&raw) {
                Some(tz) => tz,
                None => return Err(i.make_error("RangeError", format!("invalid time zone: {raw}"))),
            }
        }
    };

    let weekday = get_option(i, &options, "weekday", &["narrow", "short", "long"], None)?;
    let era = get_option(i, &options, "era", &["narrow", "short", "long"], None)?;
    let year = get_option(i, &options, "year", &["2-digit", "numeric"], None)?;
    let month = get_option(
        i,
        &options,
        "month",
        &["2-digit", "numeric", "narrow", "short", "long"],
        None,
    )?;
    let day = get_option(i, &options, "day", &["2-digit", "numeric"], None)?;
    let day_period = get_option(i, &options, "dayPeriod", &["narrow", "short", "long"], None)?;
    let hour = get_option(i, &options, "hour", &["2-digit", "numeric"], None)?;
    let minute = get_option(i, &options, "minute", &["2-digit", "numeric"], None)?;
    let second = get_option(i, &options, "second", &["2-digit", "numeric"], None)?;
    let frac_sec = {
        let v = ab(i.get_member_key(&options, "fractionalSecondDigits"))?;
        if matches!(v, Value::Undefined) {
            None
        } else {
            // GetNumberOption(1, 3): reject NaN/out-of-range, otherwise floor into [1,3].
            let n = ab(i.to_number(&v))?;
            if n.is_nan() || !(1.0..=3.0).contains(&n) {
                return Err(i.make_error("RangeError", "fractionalSecondDigits out of range"));
            }
            Some(n.floor() as u32)
        }
    };
    let tz_name = get_option(
        i,
        &options,
        "timeZoneName",
        &[
            "short",
            "long",
            "shortOffset",
            "longOffset",
            "shortGeneric",
            "longGeneric",
        ],
        None,
    )?;
    let _format_matcher = get_option(
        i,
        &options,
        "formatMatcher",
        &["basic", "best fit"],
        Some("best fit"),
    )?;
    let date_style = get_option(
        i,
        &options,
        "dateStyle",
        &["full", "long", "medium", "short"],
        None,
    )?;
    let time_style = get_option(
        i,
        &options,
        "timeStyle",
        &["full", "long", "medium", "short"],
        None,
    )?;

    // A dateStyle/timeStyle may not be combined with explicit component options.
    let has_components = weekday.is_some()
        || era.is_some()
        || year.is_some()
        || month.is_some()
        || day.is_some()
        || day_period.is_some()
        || hour.is_some()
        || minute.is_some()
        || second.is_some()
        || frac_sec.is_some()
        || tz_name.is_some();
    if (date_style.is_some() || time_style.is_some()) && has_components {
        return Err(i.make_error(
            "TypeError",
            "dateStyle/timeStyle cannot be combined with explicit date-time components",
        ));
    }

    // era and timeZoneName do not count toward ToDateTimeOptions' needDefaults check (only the date
    // and time components do), so a formatter with only `era` still gets the default date components.
    let has_explicit = weekday.is_some()
        || year.is_some()
        || month.is_some()
        || day.is_some()
        || hour.is_some()
        || minute.is_some()
        || second.is_some()
        || day_period.is_some()
        || frac_sec.is_some();
    let resolved = resolve_locale(i, &requested, &["ca", "nu", "hc"]);
    // The `-u-hc-` locale-extension hour cycle (the `hourCycle` option, read later, overrides it).
    let hc_ext = resolved
        .keywords
        .iter()
        .find(|(k, _)| k == "hc")
        .map(|(_, v)| v.clone());
    // ResolveLocale for the `nu` key gives both the numbering system and the resolved locale string
    // (reflecting a surviving `-u-nu-` extension).
    let (resolved_locale, nu_final) =
        super::service::resolve_locale_nu(&requested, numbering.as_deref());

    let obj = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.DateTimeFormat")? {
        obj.borrow_mut().proto = Some(proto);
    }
    set_internal_slot(&obj, "#\u{0}dtf", Value::Bool(true));
    let locale_lang = resolved_locale.split('-').next().unwrap_or("").to_string();
    // ResolveLocale for the `ca` key: the locale's -u-ca- extension value (if supported) is used and
    // reflected in the resolved locale, unless a supported calendar option overrides it with a
    // different value (then the reflection is dropped).
    let ca_ext = resolved
        .keywords
        .iter()
        .find(|(k, _)| k == "ca")
        .and_then(|(_, v)| {
            let lc = v.to_lowercase();
            resolve_cal(&crate::intl::tags::canonical_ca(&lc).unwrap_or(lc))
        });
    let mut ca_addition = ca_ext.clone();
    if let Some(opt) = &calendar {
        if ca_ext.as_ref() != Some(opt) {
            ca_addition = None;
        }
    }
    // Without a calendar option or -u-ca- extension, the locale's preferred calendar applies
    // (ResolveLocale's [[LocaleData]] default: th → buddhist, ar-SA → islamic-umalqura).
    let eff_cal = calendar
        .clone()
        .or(ca_ext)
        .unwrap_or_else(|| super::locale::default_calendar(&resolved_locale));
    // ResolveLocale for the `hc` key: reflect a valid -u-hc- unless the hourCycle/hour12 options
    // override it with a different value.
    let hc_valid = hc_ext
        .clone()
        .filter(|v| matches!(v.as_str(), "h11" | "h12" | "h23" | "h24"));
    let hc_addition = match (&hc_valid, &hour_cycle, hour12) {
        (Some(e), Some(opt), _) if opt != e => None,
        (Some(_), _, Some(_)) => None,
        (Some(e), _, _) => Some(e.clone()),
        _ => None,
    };
    // Rebuild the -u- extension from the surviving ca/hc/nu additions (keys sorted alphabetically).
    let base = resolved_locale
        .split("-u-")
        .next()
        .unwrap_or(&resolved_locale)
        .to_string();
    let nu_add = resolved_locale
        .strip_prefix(&format!("{base}-u-nu-"))
        .map(|s| s.to_string());
    let mut ext = String::new();
    if let Some(ca) = &ca_addition {
        ext.push_str(&format!("-ca-{ca}"));
    }
    if let Some(hc) = &hc_addition {
        ext.push_str(&format!("-hc-{hc}"));
    }
    if let Some(nu) = &nu_add {
        ext.push_str(&format!("-nu-{nu}"));
    }
    let resolved_locale = if ext.is_empty() {
        base
    } else {
        format!("{base}-u{ext}")
    };
    set_internal_slot(&obj, "#\u{0}dtf_locale", Value::lstr(resolved_locale));
    set_internal_slot(&obj, "#\u{0}dtf_ca", Value::lstr(eff_cal));
    set_internal_slot(&obj, "#\u{0}dtf_nu", Value::lstr(nu_final));
    set_internal_slot(&obj, "#\u{0}dtf_tz", Value::lstr(time_zone));
    let put = |obj: &Gc, k: &str, v: &Option<String>| {
        if let Some(v) = v {
            set_internal_slot(obj, k, Value::lstr(v.clone()));
        }
    };
    put(&obj, "#\u{0}dtf_weekday", &weekday);
    put(&obj, "#\u{0}dtf_era", &era);
    put(&obj, "#\u{0}dtf_year", &year);
    put(&obj, "#\u{0}dtf_month", &month);
    put(&obj, "#\u{0}dtf_day", &day);
    put(&obj, "#\u{0}dtf_dayperiod", &day_period);
    put(&obj, "#\u{0}dtf_hour", &hour);
    put(&obj, "#\u{0}dtf_minute", &minute);
    put(&obj, "#\u{0}dtf_second", &second);
    if let Some(f) = frac_sec {
        set_internal_slot(&obj, "#\u{0}dtf_fracsec", Value::Num(f as f64));
    }
    put(&obj, "#\u{0}dtf_tzname", &tz_name);
    put(&obj, "#\u{0}dtf_datestyle", &date_style);
    put(&obj, "#\u{0}dtf_timestyle", &time_style);
    // ECMA-402 DateTime Style Formats selects locale-specific CLDR patterns. Derive the working
    // component set from those actual patterns rather than applying an English-shaped preset.
    let cldr_locale = dtf_cldr_locale(&obj);
    if let Some(ds) = &date_style {
        install_style_components(
            &obj,
            crate::cldr_datetime_patterns::date_style(cldr_locale, &dtf_calendar(&obj), ds).0,
        );
    }
    if let Some(ts) = &time_style {
        install_style_components(
            &obj,
            crate::cldr_datetime_patterns::time_style(cldr_locale, ts),
        );
    }
    // The hour cycle is always resolved (a Temporal PlainTime/Instant/PlainDateTime toLocaleString
    // shows a defaulted hour that must honor hour12/hourCycle), but only *reported* in
    // resolvedOptions when the formatter's own pattern shows an hour (explicit hour or a timeStyle).
    let shows_hour = hour.is_some() || time_style.is_some();
    {
        // hour12 overrides hourCycle: true → the locale's 12-hour cycle (h11 for ja, else h12);
        // false → h23. Absent both, fall back to the requested hourCycle or h23.
        let lang = locale_lang.as_str();
        let locale_hour_cycle = crate::cldr_datetime_patterns::default_hour_cycle(cldr_locale);
        let hc = if let Some(h12) = hour12 {
            if h12 {
                if lang == "ja" { "h11" } else { "h12" }.to_string()
            } else {
                "h23".to_string()
            }
        } else {
            // The `hourCycle` option wins, then the locale's `-u-hc-` extension, then the locale
            // default (en is 12-hour; most others are h23).
            hour_cycle
                .clone()
                .or_else(|| hc_ext.clone())
                .unwrap_or_else(|| locale_hour_cycle.to_string())
        };
        let h12 = matches!(hc.as_str(), "h11" | "h12");
        set_internal_slot(&obj, "#\u{0}dtf_hourcycle", Value::lstr(hc));
        set_internal_slot(&obj, "#\u{0}dtf_hour12", Value::Bool(h12));
        set_internal_slot(&obj, "#\u{0}dtf_hourshown", Value::Bool(shows_hour));
    }
    // Default components when nothing was requested: year/month/day numeric. Flagged so a Temporal
    // receiver's compatibility check ignores them (only *explicit* options can conflict).
    if !has_explicit && date_style.is_none() && time_style.is_none() {
        set_internal_slot(&obj, "#\u{0}dtf_year", Value::lstr("numeric"));
        set_internal_slot(&obj, "#\u{0}dtf_month", Value::lstr("numeric"));
        set_internal_slot(&obj, "#\u{0}dtf_day", Value::lstr("numeric"));
        set_internal_slot(&obj, "#\u{0}dtf_defaults", Value::Bool(true));
    }
    if date_style.is_none() && time_style.is_none() {
        let mut requested = Vec::new();
        for (slot, kind) in [
            ("#\u{0}dtf_weekday", "weekday"),
            ("#\u{0}dtf_era", "era"),
            ("#\u{0}dtf_year", "year"),
            ("#\u{0}dtf_month", "month"),
            ("#\u{0}dtf_day", "day"),
            ("#\u{0}dtf_dayperiod", "dayPeriod"),
            ("#\u{0}dtf_hour", "hour"),
            ("#\u{0}dtf_minute", "minute"),
            ("#\u{0}dtf_second", "second"),
            ("#\u{0}dtf_tzname", "timeZoneName"),
        ] {
            if obj.borrow().internal_slots().contains(slot) {
                requested.push((kind, String::new()));
            }
        }
        let _ = requested;
        let (pattern, _) = formatter_pattern(&obj, 0);
        install_resolved_components(&obj, &pattern);
    }
    // Legacy "ChainDateTimeFormat": called without `new` on an object inheriting from
    // %DateTimeFormat.prototype%, the fresh instance is stored ON that object under the realm's
    // [[FallbackSymbol]] and the object itself is returned.
    if !i.constructing {
        if let Some(chained) = crate::intl::legacy_chain(i, &t, "Intl.DateTimeFormat", &obj) {
            return Ok(chained);
        }
    }
    Ok(Value::Obj(obj))
}

/// Break epoch-ms into UTC (Y, M, D, h, m, s, weekday).
fn ymd(ms: f64) -> (i64, u32, u32, u32, u32, u32, u32) {
    let ms_i = ms.floor() as i64;
    let mut days = ms_i.div_euclid(86_400_000);
    let mut rem = ms_i.rem_euclid(86_400_000);
    let h = (rem / 3_600_000) as u32;
    rem %= 3_600_000;
    let mi = (rem / 60_000) as u32;
    rem %= 60_000;
    let sec = (rem / 1000) as u32;
    let weekday = ((days % 7 + 4) % 7 + 7) as u32 % 7; // 0=Sun; 1970-01-01 was Thursday(4)
                                                       // civil from days
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let year = if m <= 2 { y + 1 } else { y };
    let _ = &mut days;
    (year, m, d, h, mi, sec, weekday)
}

const WD_LONG: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const WD_SHORT: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Resolve the format argument to epoch-milliseconds. A Temporal receiver uses its ISO fields and
/// must be compatible with the formatter's requested components (else TypeError/RangeError).
/// Resolve to (epoch-ms, Temporal-kind); kind gates the components a Temporal receiver may show
/// (1=Date, 2=Time, 3=DateTime, 4=YearMonth, 5=MonthDay, 0=any/number/instant/zoned).
fn dtf_ms_kind(i: &mut Interp, o: &Gc, date: &Value) -> Result<(f64, u8), Value> {
    use crate::temporal::Temporal as T;
    if matches!(date, Value::Undefined) {
        return Ok((0.0, 0));
    }
    if let Some(dobj) = date.as_obj() {
        let ptr = Gc::as_ptr(dobj) as usize;
        if let Some(t) = i.temporal.get(&ptr).cloned() {
            // format()/formatRange() do not accept Temporal.ZonedDateTime directly (the caller can't
            // know which time zone to use); Temporal.ZonedDateTime.prototype.toLocaleString handles it.
            if matches!(t, T::Zoned { .. }) {
                return Err(i.make_error(
                    "TypeError",
                    "Intl.DateTimeFormat does not support Temporal.ZonedDateTime; use zonedDateTime.toLocaleString()",
                ));
            }
            temporal_compat_check(i, o, &t)?;
            // Calendar mismatch: a non-ISO Temporal calendar must equal the formatter's calendar.
            let tcal = i
                .temporal_cal
                .get(&ptr)
                .map(|c| c.to_string())
                .unwrap_or_else(|| "iso8601".to_string());
            let dcal = match o
                .borrow()
                .internal_slots()
                .get("#\u{0}dtf_ca")
                .map(|p| p.value())
            {
                Some(Value::Str(s)) => s.to_string(),
                _ => "iso8601".to_string(),
            };
            let kind = match t {
                T::Date(_) => 1,
                T::Time(_) => 2,
                T::DateTime(..) => 3,
                T::YearMonth(_) => 4,
                T::MonthDay(_) => 5,
                T::Instant(_) => 6, // absolute (tz-shifted like kind 0) but defaults to date+time
                _ => 0,
            };
            // YearMonth/MonthDay require an EXACT calendar match (no ISO exception, since a bare
            // month-day/year-month is calendar-specific); other types accept an ISO instance.
            let exact = matches!(kind, 4 | 5);
            if tcal != dcal && (exact || tcal != "iso8601") {
                return Err(
                    i.make_error("RangeError", format!("calendar mismatch: {tcal} vs {dcal}"))
                );
            }
            return Ok((temporal_to_ms(&t), kind));
        }
    }
    let n = ab(i.to_number(date))?;
    // TimeClip: the representable range is ±8.64e15 ms; anything outside (or non-finite) is invalid.
    if !n.is_finite() || n.abs() > 8.64e15 {
        return Err(i.make_error("RangeError", "Invalid time value"));
    }
    // TimeClip truncates toward zero (so -0.9 -> +0, not floor's -1).
    Ok((n.trunc() + 0.0, 0))
}

/// A Temporal receiver must overlap the formatter's requested fields, else TypeError. The requested
/// set includes both explicit component options and any dateStyle/timeStyle expansion, so e.g. a
/// lone `timeStyle` (time fields only) has no overlap with a `PlainDate` and throws, while a
/// `dateStyle`+`timeStyle` formatter overlaps every receiver.
fn temporal_compat_check(
    i: &mut Interp,
    o: &Gc,
    t: &crate::temporal::Temporal,
) -> Result<(), Value> {
    use crate::temporal::Temporal as T;
    // Field set of the receiver (era is auxiliary to year and never stands alone, so it is omitted).
    let recv: &[&str] = match t {
        T::Date(_) => &["weekday", "year", "month", "day"],
        T::YearMonth(_) => &["year", "month"],
        T::MonthDay(_) => &["month", "day"],
        T::Time(_) => &["hour", "minute", "second", "fracsec", "dayperiod"],
        _ => return Ok(()), // DateTime / Instant / Zoned overlap everything
    };
    // A fully-defaulted formatter adapts its fields to the receiver, so it always overlaps.
    if o.borrow().internal_slots().contains("#\u{0}dtf_defaults") {
        return Ok(());
    }
    // A field is requested if set explicitly (`__dtf_`) or via a dateStyle/timeStyle (`__dtfx_`).
    let present = |field: &str| {
        let b = o.borrow();
        b.internal_slots().contains(&format!("#\u{0}dtf_{field}"))
            || b.internal_slots().contains(&format!("#\u{0}dtfx_{field}"))
    };
    const ALL: &[&str] = &[
        "weekday",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "fracsec",
        "dayperiod",
    ];
    let requested: Vec<&str> = ALL.iter().copied().filter(|f| present(f)).collect();
    if !requested.is_empty() && !requested.iter().any(|f| recv.contains(f)) {
        return Err(i.make_error(
            "TypeError",
            "no overlap between the formatter and the Temporal value",
        ));
    }
    Ok(())
}

fn do_format(i: &mut Interp, this: &Value, date: &Value) -> Result<String, Value> {
    let o = brand_slot(i, this, "#\u{0}dtf")?;
    let (ms, kind) = dtf_ms_kind(i, &o, date)?;
    Ok(do_format_ms(&o, ms, kind))
}

/// The formatter's numbering system id (`latn` unless set).
fn dtf_nu(o: &Gc) -> String {
    match o
        .borrow()
        .internal_slots()
        .get("#\u{0}dtf_nu")
        .map(|p| p.value())
    {
        Some(Value::Str(s)) => s.to_string(),
        _ => "latn".to_string(),
    }
}

/// The shipped CLDR locale whose data serves the formatter.
fn dtf_cldr_locale(o: &Gc) -> &'static str {
    super::service::data_locale(&dtf_slot(o, "#\u{0}dtf_locale").unwrap_or_default())
}

/// Format an already-resolved epoch-ms + Temporal kind to the joined string.
fn do_format_ms(o: &Gc, ms: f64, kind: u8) -> String {
    let s: String = build_parts(o, ms, kind)
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    crate::intl::numberformat::xlate_digits(&s, &dtf_nu(o))
}

fn dtf_slot(o: &Gc, slot: &str) -> Option<String> {
    match o
        .borrow()
        .internal_slots()
        .get(slot)
        .map(|property| property.value())
    {
        Some(Value::Str(value)) => Some(value.to_string()),
        _ => None,
    }
}

/// The formatter's calendar as a locale-data key (an unset calendar is Gregorian). The ISO 8601
/// calendar has CLDR patterns of its own, with the Gregorian names and eras.
fn dtf_calendar(o: &Gc) -> String {
    match dtf_slot(o, "#\u{0}dtf_ca").as_deref() {
        None | Some("" | "gregory") => "gregory".to_string(),
        Some(calendar) => calendar.to_string(),
    }
}

/// The pattern letter of the formatter's resolved hour cycle.
fn hour_letter(o: &Gc) -> char {
    match dtf_slot(o, "#\u{0}dtf_hourcycle").as_deref() {
        Some("h11") => 'K',
        Some("h12") => 'h',
        Some("h24") => 'k',
        _ => 'H',
    }
}

/// Whether a Temporal receiver of `kind` can show a component (the Temporal additions to
/// ECMA-402's DateTimeFormat: each Plain type formats only its own fields, and only the absolute
/// kinds have a time zone).
fn kind_allows(kind: u8, field: &str) -> bool {
    if field == "tzname" && matches!(kind, 1..=5) {
        return false;
    }
    match kind {
        1 => !matches!(
            field,
            "hour" | "minute" | "second" | "dayperiod" | "fracsec"
        ),
        2 => matches!(
            field,
            "hour" | "minute" | "second" | "dayperiod" | "fracsec"
        ),
        4 => matches!(field, "year" | "month" | "era"),
        5 => matches!(field, "month" | "day"),
        _ => true,
    }
}

/// The skeleton field name of a UTS #35 pattern letter.
fn letter_field(letter: char) -> Option<&'static str> {
    Some(match letter {
        'G' => "era",
        'y' | 'Y' | 'u' | 'U' | 'r' => "year",
        'M' | 'L' => "month",
        'd' => "day",
        'E' | 'c' | 'e' => "weekday",
        'a' | 'b' | 'B' => "dayperiod",
        'h' | 'H' | 'K' | 'k' => "hour",
        'm' => "minute",
        's' => "second",
        'S' => "fracsec",
        'z' | 'Z' | 'O' | 'v' | 'V' | 'X' | 'x' => "tzname",
        _ => return None,
    })
}

/// The UTS #35 skeleton of the formatter's requested components for a receiver of `kind`. A
/// formatter with defaulted components shows the receiver's natural fields.
fn requested_skeleton(o: &Gc, kind: u8) -> String {
    let slot = |field: &str| dtf_slot(o, &format!("#\u{0}dtf_{field}"));
    let width = |value: &str, letter: char, text: bool| -> String {
        let count = match value {
            "2-digit" => 2,
            "narrow" => 5,
            "short" if text => 3,
            "long" => 4,
            _ => 1,
        };
        std::iter::repeat_n(letter, count).collect()
    };
    let hour = hour_letter(o);
    let mut skeleton = String::new();
    let defaulted = o.borrow().internal_slots().contains("#\u{0}dtf_defaults");
    if defaulted {
        if let Some(era) = slot("era").filter(|_| kind_allows(kind, "era")) {
            skeleton.push_str(&width(&era, 'G', true));
        }
        match kind {
            4 => skeleton.push_str("yM"),
            5 => skeleton.push_str("Md"),
            2 => {}
            _ => skeleton.push_str("yMd"),
        }
        if matches!(kind, 2 | 3 | 6) {
            skeleton.push(hour);
            skeleton.push_str("mmss");
        }
    } else {
        for (field, letter, text) in [
            ("era", 'G', true),
            ("year", 'y', false),
            ("month", 'M', true),
            ("weekday", 'E', true),
            ("day", 'd', false),
            ("dayperiod", 'B', true),
            ("hour", hour, false),
            ("minute", 'm', false),
            ("second", 's', false),
        ] {
            if !kind_allows(kind, field) {
                continue;
            }
            if let Some(value) = slot(field) {
                skeleton.push_str(&width(&value, letter, text));
            }
        }
        if kind_allows(kind, "fracsec") {
            if let Some(Value::Num(digits)) = o
                .borrow()
                .internal_slots()
                .get("#\u{0}dtf_fracsec")
                .map(|p| p.value())
            {
                skeleton.extend(std::iter::repeat_n('S', digits as usize));
            }
        }
    }
    if kind_allows(kind, "tzname") {
        if let Some(name) = slot("tzname") {
            skeleton.push_str(match name.as_str() {
                "long" => "zzzz",
                "shortOffset" => "O",
                "longOffset" => "OOOO",
                "shortGeneric" => "v",
                "longGeneric" => "vvvv",
                _ => "z",
            });
        }
    }
    skeleton
}

/// The pattern and UTS #35 `numbers` override that format a receiver of `kind`: the
/// dateStyle/timeStyle patterns (DateTimeStyleFormat), restricted to the receiver's fields, or
/// the best match for the requested components (BestFitFormatMatcher). Cached per kind.
fn formatter_pattern(o: &Gc, kind: u8) -> (String, String) {
    let pattern_slot = format!("#\u{0}dtf_pattern{kind}");
    let numbers_slot = format!("#\u{0}dtf_numbers{kind}");
    if let (Some(pattern), Some(numbers)) = (dtf_slot(o, &pattern_slot), dtf_slot(o, &numbers_slot))
    {
        return (pattern, numbers);
    }
    let (pattern, numbers) = select_pattern(o, kind);
    set_internal_slot(o, &pattern_slot, Value::lstr(pattern.clone()));
    set_internal_slot(o, &numbers_slot, Value::lstr(numbers.clone()));
    (pattern, numbers)
}

fn select_pattern(o: &Gc, kind: u8) -> (String, String) {
    let locale = dtf_cldr_locale(o);
    let calendar = dtf_calendar(o);
    let decimal = crate::cldr_numbers::symbols(locale, &dtf_nu(o)).decimal;
    let hour = hour_letter(o);
    let best = |skeleton: &str| -> String {
        let pattern = super::dtf_pattern::best_pattern(locale, &calendar, skeleton, true, decimal);
        if matches!(hour, 'K' | 'k') {
            super::dtf_pattern::with_hour_letter(&pattern, hour)
        } else {
            pattern
        }
    };
    let date_style = dtf_slot(o, "#\u{0}dtf_datestyle");
    let time_style = dtf_slot(o, "#\u{0}dtf_timestyle");
    if date_style.is_none() && time_style.is_none() {
        return (best(&requested_skeleton(o, kind)), String::new());
    }
    // DateTimeStyleFormat: the locale's date pattern for the calendar, the time pattern in the
    // formatter's hour cycle (the 12- or 24-hour variant of the style), and the connector.
    let date = date_style.as_deref().map(|style| {
        let (pattern, numbers) =
            crate::cldr_datetime_patterns::date_style(locale, &calendar, style);
        (pattern.to_string(), numbers.to_string())
    });
    let time = time_style.as_deref().map(|style| {
        let pattern = crate::cldr_datetime_patterns::time_style(locale, style);
        let twelve = |letter: char| matches!(letter, 'h' | 'K');
        let pattern_hour = super::dtf_pattern::tokenize(pattern)
            .into_iter()
            .find_map(|token| match token {
                super::dtf_pattern::Token::Field(letter @ ('h' | 'H' | 'K' | 'k'), _) => {
                    Some(letter)
                }
                _ => None,
            });
        match pattern_hour {
            Some(letter) if twelve(letter) != twelve(hour) => {
                // The other clock: rebuild the style's skeleton with the cycle's hour letter.
                let skeleton = super::dtf_pattern::with_hour_letter(
                    &super::dtf_pattern::pattern_skeleton(pattern),
                    if twelve(hour) { 'h' } else { 'H' },
                );
                best(&skeleton)
            }
            Some(_) => super::dtf_pattern::with_hour_letter(pattern, hour),
            None => pattern.to_string(),
        }
    });
    let allowed = |pattern: &str| {
        super::dtf_pattern::tokenize(pattern)
            .into_iter()
            .all(|token| match token {
                super::dtf_pattern::Token::Field(letter, _) => {
                    letter_field(letter).is_none_or(|field| kind_allows(kind, field))
                }
                super::dtf_pattern::Token::Literal(_) => true,
            })
    };
    let has_date_fields = !matches!(kind, 2);
    let has_time_fields = !matches!(kind, 1 | 4 | 5);
    let date = date.filter(|_| has_date_fields);
    let time = time.filter(|_| has_time_fields);
    let (pattern, numbers) = match (date, time) {
        (Some((date, numbers)), Some(time)) => {
            let connector = crate::cldr_datetime_patterns::connector(
                locale,
                &calendar,
                date_style.as_deref().unwrap(),
            );
            (
                connector.replace("{1}", &date).replace("{0}", &time),
                numbers,
            )
        }
        (Some(date), None) => date,
        (None, Some(time)) => (time, String::new()),
        // The receiver shares no fields with the styles: its own natural fields.
        (None, None) => (best(&requested_skeleton(o, kind)), String::new()),
    };
    if allowed(&pattern) {
        return (pattern, numbers);
    }
    // A Temporal receiver shows only its own fields of the style pattern.
    let skeleton: String = super::dtf_pattern::tokenize(&pattern)
        .into_iter()
        .filter_map(|token| match token {
            super::dtf_pattern::Token::Field(letter, len)
                if letter_field(letter).is_some_and(|field| kind_allows(kind, field)) =>
            {
                Some(std::iter::repeat_n(letter, len).collect::<String>())
            }
            _ => None,
        })
        .collect();
    (best(&skeleton), String::new())
}

fn push_dtf_literal(parts: &mut Vec<(&'static str, String)>, value: &str) {
    if value.is_empty() {
        return;
    }
    // DateTimeFormat locale data is implementation-defined. Use the web-platform's ordinary-space
    // rendering for CLDR's day-period NNBSP so `format()` remains the concatenation of
    // `formatToParts()` and matches Test262's fixed numbering-system expectations.
    let value = value.replace('\u{202f}', " ");
    if let Some(("literal", previous)) = parts.last_mut() {
        previous.push_str(&value);
    } else {
        parts.push(("literal", value));
    }
}

/// ECMA-402 ToLocalTime: a receiver's fields in the formatter's calendar and time zone.
struct LocalFields {
    calendar: String,
    /// The calendar's (proleptic) year: the related Gregorian year for Chinese and Dangi.
    year: i64,
    era: Option<&'static str>,
    era_year: Option<i64>,
    /// The ISO year (UTS #35 `r`).
    related_year: i64,
    /// The 1-based month number naming the month (Hebrew uses CLDR's month index).
    month: u32,
    /// The month as a number: the ordinal month in the year for Hebrew (as ICU and Temporal
    /// number it, so a common year's Adar is 6), otherwise `month`.
    month_number: u32,
    leap_month: bool,
    /// Hebrew Adar II: month 7 of a leap year, which CLDR names separately.
    leap_year_month: bool,
    day: u32,
    /// 0 = Sunday.
    weekday: u32,
    hour: u32,
    minute: u32,
    second: u32,
    millisecond: u32,
    /// The instant, for time zone names.
    epoch_sec: i64,
}

fn local_fields(o: &Gc, ms: f64, kind: u8) -> LocalFields {
    let epoch_sec = (ms / 1000.0).floor() as i64;
    // An absolute instant (number/Date kind 0, Temporal.Instant kind 6) is shifted into the
    // formatter's time zone; Temporal wall-clock values (kinds 1-5) already carry their local time.
    let local_ms = if kind == 0 || kind == 6 {
        match dtf_slot(o, "#\u{0}dtf_tz") {
            Some(tz) => {
                let off_ms = crate::tz::offset_at(&tz, epoch_sec)
                    .map(|s| s as f64 * 1000.0)
                    .unwrap_or_else(|| tz_offset_ms(&tz));
                ms + off_ms
            }
            None => ms,
        }
    } else {
        ms
    };
    let (iso_year, iso_month, iso_day, hour, minute, second, weekday) = ymd(local_ms);
    let millisecond = local_ms.rem_euclid(1000.0) as u32;
    let iso_date = crate::temporal::IsoDate {
        year: iso_year,
        month: iso_month as u8,
        day: iso_day as u8,
    };
    let calendar = dtf_calendar(o);
    let mut fields = LocalFields {
        calendar: calendar.clone(),
        year: iso_year,
        era: None,
        era_year: None,
        related_year: iso_year,
        month: iso_month,
        month_number: iso_month,
        leap_month: false,
        leap_year_month: false,
        day: iso_day,
        weekday,
        hour,
        minute,
        second,
        millisecond,
        epoch_sec,
    };
    if calendar == "gregory" || calendar == "iso8601" {
        let (era, era_year) = crate::temporal::cal_era("gregory", iso_date);
        fields.era = era;
        fields.era_year = era_year;
        return fields;
    }
    let calendar_fields = crate::temporal::cal_fields(&calendar, iso_date);
    fields.year = crate::temporal::cal_year_num(&calendar, iso_date);
    fields.month = calendar_fields.1 as u32;
    fields.month_number = fields.month;
    fields.day = calendar_fields.2 as u32;
    let (era, era_year) = crate::temporal::cal_era(&calendar, iso_date);
    fields.era = era;
    fields.era_year = era_year;
    match calendar.as_str() {
        "chinese" | "dangi" => {
            // The month number (a leap month repeats its predecessor's number).
            let (number, leap) = crate::temporal::lunisolar_month_num(&calendar, iso_date);
            fields.month = number as u32;
            fields.month_number = number as u32;
            fields.leap_month = leap;
            fields.related_year = fields.year;
        }
        "hebrew" => {
            let month = crate::temporal::hebrew_cldr_month(iso_date) as u32;
            fields.leap_year_month = month == 7 && calendar_fields.7;
            fields.month = month;
        }
        _ => {}
    }
    fields
}

/// The CLDR era key of a calendar's era code, and the calendar whose era names apply.
fn era_key(calendar: &str, code: &str) -> (&'static str, &'static str) {
    match (calendar, code) {
        ("japanese", "reiwa") => ("japanese", "236"),
        ("japanese", "heisei") => ("japanese", "235"),
        ("japanese", "showa") => ("japanese", "234"),
        ("japanese", "taisho") => ("japanese", "233"),
        ("japanese", "meiji") => ("japanese", "232"),
        // Before Meiji, Temporal's Japanese calendar uses the Gregorian eras.
        ("japanese", code) => ("gregory", era_cldr_index("gregory", code)),
        // CLDR's ISO 8601 calendar aliases the Gregorian eras.
        ("iso8601", code) => ("gregory", era_cldr_index("gregory", code)),
        _ => (static_calendar(calendar), era_cldr_index(calendar, code)),
    }
}

fn static_calendar(calendar: &str) -> &'static str {
    match calendar {
        "buddhist" => "buddhist",
        "chinese" => "chinese",
        "coptic" => "coptic",
        "dangi" => "dangi",
        "ethioaa" => "ethioaa",
        "ethiopic" => "ethiopic",
        "hebrew" => "hebrew",
        "indian" => "indian",
        "islamic-civil" => "islamic-civil",
        "islamic-tbla" => "islamic-tbla",
        "islamic-umalqura" => "islamic-umalqura",
        "japanese" => "japanese",
        "persian" => "persian",
        "roc" => "roc",
        _ => "gregory",
    }
}

/// The digits of a numeric numbering system a CLDR pattern's `numbers` override names (CLDR
/// numberingSystems.xml `type="numeric"`): its fields keep their padded width, in those digits.
fn numbering_digits(system: &str) -> Option<[char; 10]> {
    match system {
        "hanidec" => Some(['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九']),
        _ => None,
    }
}

/// Spell `value` in an algorithmic numbering system a CLDR pattern's `numbers` override names
/// (CLDR numberingSystems.xml `type="algorithmic"` and the RBNF rules it cites). An algorithmic
/// system spells the whole value, so a two-letter field is neither padded nor truncated.
fn numbering(system: &str, value: i64) -> Option<String> {
    const HAN: [char; 10] = ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    match system {
        // zh/SpelloutRules/spellout-numbering-days: 初一…初十, 十一…二十, 廿一…廿九, 三十.
        "hanidays" => {
            let spell = |n: i64| -> String {
                match n {
                    0..=9 => HAN[n as usize].to_string(),
                    10 => "十".to_string(),
                    11..=19 => format!("十{}", HAN[(n - 10) as usize]),
                    20 => "二十".to_string(),
                    _ => format!(
                        "{}十{}",
                        HAN[(n / 10) as usize],
                        if n % 10 == 0 {
                            String::new()
                        } else {
                            HAN[(n % 10) as usize].to_string()
                        }
                    ),
                }
            };
            Some(match value {
                1..=10 => format!("初{}", spell(value)),
                11..=20 => spell(value),
                21..=29 => format!("廿{}", HAN[(value - 20) as usize]),
                30 => "三十".to_string(),
                _ => value.to_string(),
            })
        }
        // ja/SpelloutRules/spellout-numbering-year-latn: year 1 is 元.
        "jpanyear" => Some(if value == 1 {
            "元".to_string()
        } else {
            value.to_string()
        }),
        _ => None,
    }
}

/// The numbering-system override a pattern's `numbers` string applies to a field letter:
/// `hanidec` for every numeric field, or `d=hanidays;y=jpanyear`-style per-letter overrides.
/// The `numbers` attribute belongs to a date style pattern (UTS #35 Part 4, dateFormats), so a
/// whole-pattern override reaches only the date fields when that pattern is combined with a time
/// pattern: `ja-u-ca-chinese`'s hanidec day numbers do not apply to the hour.
fn numbering_override(numbers: &str, letter: char) -> Option<&str> {
    /// The date field letters (UTS #35 Part 4 Date Field Symbol Table: era through week day,
    /// excluding the names that never take numerals).
    const DATE_FIELDS: &str = "yYuUrQqMLwWdDFgEec";
    if numbers.is_empty() {
        return None;
    }
    if !numbers.contains('=') {
        return DATE_FIELDS.contains(letter).then_some(numbers);
    }
    numbers.split(';').find_map(|entry| {
        let (fields, system) = entry.split_once('=')?;
        fields.contains(letter).then_some(system)
    })
}

/// The flexible day period (UTS #35 Part 4 Day Period Rules) of a time for `B`. As in ICU, noon
/// applies at exactly 12:00 (counting the minutes and seconds the pattern shows) and midnight is
/// not used.
fn flexible_day_period(locale: &str, hour: u32, exact_hour: bool) -> Option<&'static str> {
    let rules = crate::cldr_datetime_patterns::day_period_rules(locale);
    if hour == 12 && exact_hour && rules.iter().any(|rule| rule.0 == "noon") {
        return Some("noon");
    }
    let hour = hour as u8;
    rules.iter().find_map(|&(period, from, before)| {
        let applies = if from < before {
            (from..before).contains(&hour)
        } else if from > before {
            hour >= from || hour < before
        } else {
            false
        };
        applies.then_some(period)
    })
}

/// FormatDateTimePattern (ECMA-402 #sec-formatdatetimepattern) for a UTS #35 pattern: each
/// field's value in the width its pattern letter gives, with the pattern's literals.
fn render_pattern(
    o: &Gc,
    pattern: &str,
    numbers: &str,
    fields: &LocalFields,
) -> Vec<(&'static str, String)> {
    use super::dtf_pattern::Token;
    let locale = dtf_cldr_locale(o);
    let language = locale.split('-').next().unwrap_or("en");
    let calendar = fields.calendar.as_str();
    let names_calendar = static_calendar(calendar);
    // CLDR gives every ja Japanese-calendar date format that writes 年 the `y=jpanyear` override,
    // so an era's first year reads 元年; a flexible pattern writing 年 has no `numbers` attribute
    // of its own and takes the same year numbering (as ICU's SimpleDateFormat::applyPattern does).
    let numbers = if numbers.is_empty()
        && language == "ja"
        && calendar == "japanese"
        && pattern.contains('年')
    {
        "y=jpanyear"
    } else {
        numbers
    };
    let tokens = super::dtf_pattern::tokenize(pattern);
    let shows = |letters: &str| {
        tokens
            .iter()
            .any(|token| matches!(token, Token::Field(letter, _) if letters.contains(*letter)))
    };
    let shows_minute = shows("m");
    let shows_second = shows("s");
    let name_width = |len: usize| match len {
        4 => "long",
        5 => "narrow",
        _ => "short",
    };
    let pad = |value: i64, len: usize| -> String {
        let text = value.unsigned_abs().to_string();
        let padded = if text.len() < len {
            format!("{}{text}", "0".repeat(len - text.len()))
        } else {
            text
        };
        if value < 0 {
            format!("-{padded}")
        } else {
            padded
        }
    };
    // A numeric field in `len` digits; UTS #35 `yy` is the year truncated to its last two.
    let numeric = |letter: char, value: i64, len: usize| -> String {
        let two_digit_year = letter == 'y' && len == 2;
        let digits = |value: i64| {
            if two_digit_year {
                pad(value.rem_euclid(100), 2)
            } else {
                pad(value, len)
            }
        };
        match numbering_override(numbers, letter) {
            Some(system) => {
                if let Some(table) = numbering_digits(system) {
                    digits(value)
                        .chars()
                        .map(|c| c.to_digit(10).map_or(c, |d| table[d as usize]))
                        .collect()
                } else {
                    numbering(system, value).unwrap_or_else(|| digits(value))
                }
            }
            None => digits(value),
        }
    };
    let mut parts = Vec::new();
    for token in &tokens {
        let (letter, len) = match token {
            Token::Literal(text) => {
                push_dtf_literal(&mut parts, text);
                continue;
            }
            Token::Field(letter, len) => (*letter, *len),
        };
        let part: Option<(&'static str, String)> = match letter {
            'G' => fields.era.map(|code| {
                let (era_calendar, key) = era_key(calendar, code);
                let name = crate::cldr_dates::era_name(locale, era_calendar, name_width(len), key)
                    .map(str::to_string)
                    .unwrap_or_else(|| code.to_uppercase());
                ("era", name)
            }),
            'y' => {
                let value = if matches!(calendar, "chinese" | "dangi") {
                    (fields.related_year - 4).rem_euclid(60) + 1
                } else if let Some(era_year) = fields.era_year {
                    era_year
                } else if fields.year <= 0 {
                    1 - fields.year
                } else {
                    fields.year
                };
                Some(("year", numeric('y', value, len)))
            }
            'Y' | 'u' => Some(("year", numeric(letter, fields.year, len))),
            'r' => Some(("relatedYear", numeric('r', fields.related_year, len))),
            'U' => {
                let cyclic = ((fields.related_year - 4).rem_euclid(60) + 1) as u8;
                let name = crate::cldr_datetime_patterns::cyclic_year(
                    locale,
                    names_calendar,
                    name_width(len),
                    cyclic,
                )
                .map(str::to_string)
                .unwrap_or_else(|| cyclic.to_string());
                Some(("yearName", name))
            }
            'M' | 'L' => {
                let standalone = letter == 'L';
                let text = if len >= 3 {
                    let width = name_width(len);
                    let name = |month: u32| {
                        if standalone {
                            crate::cldr_dates::standalone_month_name(
                                locale,
                                names_calendar,
                                width,
                                month as u8,
                            )
                        } else {
                            crate::cldr_dates::month_name(
                                locale,
                                names_calendar,
                                width,
                                month as u8,
                            )
                        }
                    };
                    let named = if fields.leap_year_month {
                        name(100 + fields.month).or_else(|| name(fields.month))
                    } else {
                        name(fields.month)
                    };
                    named.map(|text| {
                        if fields.leap_month {
                            let key = format!(
                                "{}-{width}",
                                if standalone { "stand-alone" } else { "format" }
                            );
                            crate::cldr_datetime_patterns::leap_month_pattern(
                                locale,
                                names_calendar,
                                &key,
                            )
                            .map_or_else(|| text.to_string(), |leap| leap.replace("{0}", text))
                        } else {
                            text.to_string()
                        }
                    })
                } else {
                    None
                };
                let text = text.unwrap_or_else(|| {
                    let number = numeric(letter, fields.month_number as i64, len.min(2));
                    if fields.leap_month {
                        crate::cldr_datetime_patterns::leap_month_pattern(
                            locale,
                            names_calendar,
                            "numeric",
                        )
                        .map_or_else(
                            || format!("{number}bis"),
                            |leap| leap.replace("{0}", &number),
                        )
                    } else {
                        number
                    }
                });
                Some(("month", text))
            }
            'd' => Some(("day", numeric('d', fields.day as i64, len))),
            'E' | 'c' | 'e' => {
                const DAYS: [&str; 7] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
                let key = DAYS[fields.weekday as usize];
                if len <= 2 && letter != 'E' {
                    // A numeric local day of the week, counted from the region's first day.
                    let region = locale_region(o);
                    let (first_day, _) = crate::cldr_locale_info::week_info(&region);
                    let iso = if fields.weekday == 0 {
                        7
                    } else {
                        fields.weekday
                    };
                    let local = (iso as i64 - first_day as i64).rem_euclid(7) + 1;
                    Some(("weekday", numeric(letter, local, len)))
                } else {
                    let width = if len == 6 {
                        "short"
                    } else {
                        name_width(len.max(3))
                    };
                    let name = (letter == 'c')
                        .then(|| {
                            crate::cldr_datetime_patterns::standalone_weekday_override(
                                locale, width, key,
                            )
                        })
                        .flatten()
                        .or_else(|| crate::cldr_datetime_patterns::weekday(locale, width, key))
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            if width == "long" {
                                WD_LONG[fields.weekday as usize].to_string()
                            } else {
                                WD_SHORT[fields.weekday as usize].to_string()
                            }
                        });
                    Some(("weekday", name))
                }
            }
            'a' | 'b' | 'B' => {
                let width = name_width(len);
                let am_pm = if fields.hour < 12 { "am" } else { "pm" };
                let period = if letter == 'B' {
                    let exact = (!shows_minute || fields.minute == 0)
                        && (!shows_second || fields.second == 0);
                    flexible_day_period(locale, fields.hour, exact).unwrap_or(am_pm)
                } else if letter == 'b' && fields.hour == 12 && fields.minute == 0 {
                    "noon"
                } else {
                    am_pm
                };
                let name = crate::cldr_datetime_patterns::day_period(locale, width, period)
                    .or_else(|| crate::cldr_datetime_patterns::day_period(locale, width, am_pm))
                    .unwrap_or(if fields.hour < 12 { "AM" } else { "PM" });
                Some(("dayPeriod", name.to_string()))
            }
            'h' | 'H' | 'K' | 'k' => {
                let hour = fields.hour;
                let value = match letter {
                    'h' => {
                        if hour.is_multiple_of(12) {
                            12
                        } else {
                            hour % 12
                        }
                    }
                    'K' => hour % 12,
                    'k' => {
                        if hour == 0 {
                            24
                        } else {
                            hour
                        }
                    }
                    _ => hour,
                };
                Some(("hour", numeric(letter, value as i64, len)))
            }
            'm' => Some(("minute", numeric('m', fields.minute as i64, len))),
            's' => Some(("second", numeric('s', fields.second as i64, len))),
            'S' => {
                let mut digits = format!("{:03}", fields.millisecond);
                while digits.len() < len {
                    digits.push('0');
                }
                digits.truncate(len);
                Some(("fractionalSecond", digits))
            }
            'z' | 'Z' | 'O' | 'v' | 'V' | 'X' | 'x' => {
                let style = match (letter, len) {
                    ('z', 1..=3) => "short",
                    ('z', _) => "long",
                    ('O', 1) => "shortOffset",
                    ('O', _) | ('Z', 4) => "longOffset",
                    ('v', 1) | ('V', 1) => "shortGeneric",
                    ('v', _) | ('V', _) => "longGeneric",
                    _ => "longOffset",
                };
                let tz = dtf_slot(o, "#\u{0}dtf_tz")
                    .unwrap_or_else(|| crate::builtins::system_time_zone().to_string());
                let name = super::tz_names::display_name(
                    locale,
                    &locale_region(o),
                    &tz,
                    style,
                    fields.epoch_sec,
                );
                Some(("timeZoneName", name))
            }
            _ => None,
        };
        if let Some(part) = part {
            parts.push(part);
        }
    }
    parts
}

/// The region subtag of the formatter's locale, else its likely region.
/// The formatter locale's region: its region subtag, else the likely region of its language
/// and script (UTS #35 Part 1 Likely Subtags), else the world (`001`). It selects the week data
/// and the preferred zones of metazones.
fn locale_region(o: &Gc) -> String {
    let locale = dtf_slot(o, "#\u{0}dtf_locale").unwrap_or_default();
    match super::tags::parse(&locale) {
        Some(tag) if !tag.region.is_empty() => tag.region,
        Some(tag) => super::locale::add_likely(&tag.language, &tag.script, "")
            .map(|(_, _, region)| region)
            .filter(|region| !region.is_empty())
            .unwrap_or_else(|| "001".to_string()),
        None => "001".to_string(),
    }
}

/// ECMA-402 PartitionDateTimeRangePattern: the parts of a range with their sources. The range
/// patterns come from CLDR intervalFormats ([`super::dtf_interval`]); the greatest field in which
/// the two local times differ (era through fractional seconds, as the format shows them) selects
/// one. Equal fields format a single date whose parts are all shared.
fn build_range_parts(
    o: &Gc,
    start_ms: f64,
    end_ms: f64,
    kind: u8,
) -> Vec<(&'static str, String, &'static str)> {
    use super::dtf_interval::{self, Field, RangePattern};
    let (pattern, numbers) = formatter_pattern(o, kind);
    let start = local_fields(o, start_ms, kind);
    let end = local_fields(o, end_ms, kind);
    let single = |fields: &LocalFields| {
        render_pattern(o, &pattern, &numbers, fields)
            .into_iter()
            .map(|(kind, value)| (kind, value, "shared"))
            .collect::<Vec<_>>()
    };
    // [[FractionalSecondDigits]] compares milliseconds at the displayed precision.
    let fraction_digits = super::dtf_pattern::tokenize(&pattern)
        .into_iter()
        .find_map(|token| match token {
            super::dtf_pattern::Token::Field('S', len) => Some(len.min(3) as u32),
            _ => None,
        });
    let key = |fields: &LocalFields| -> [i64; 9] {
        let fraction = match fraction_digits {
            Some(digits) => i64::from(fields.millisecond / 10u32.pow(3 - digits)),
            None => i64::from(fields.millisecond),
        };
        [
            0,
            fields.year,
            i64::from(fields.month_number) * 2 + i64::from(fields.leap_month),
            i64::from(fields.day),
            i64::from(fields.hour >= 12),
            i64::from(fields.hour),
            i64::from(fields.minute),
            i64::from(fields.second),
            fraction,
        ]
    };
    let mut start_key = key(&start);
    let end_key = key(&end);
    if start.era != end.era {
        start_key[0] = 1;
    }
    let Some(field) = dtf_interval::greatest_difference(&start_key, &end_key) else {
        return single(&start);
    };
    let locale = dtf_cldr_locale(o);
    let calendar = dtf_calendar(o);
    let decimal = crate::cldr_numbers::symbols(locale, &dtf_nu(o)).decimal;
    let patterns = dtf_interval::range_patterns(locale, &calendar, &pattern, decimal);
    let numbers_for = |candidate: &str| {
        if candidate == pattern {
            numbers.as_str()
        } else {
            ""
        }
    };
    let same_day = matches!(
        field,
        Field::AmPm | Field::Hour | Field::Minute | Field::Second | Field::Millisecond
    );
    // UTS #35 intervalFormatFallback: both dates in a full pattern, the later first when the
    // fallback is "{1} – {0}". A difference within one day of a date-time pattern shows the
    // date once beside the time range, combined by the dateTimeFormat.
    let fallback = |full: &str| -> Vec<(&'static str, String, &'static str)> {
        let range = |time_pattern: &str| {
            let mut parts = Vec::new();
            let mut rest = patterns.fallback.as_str();
            while let Some(open) = rest.find('{') {
                push_shared_literal(&mut parts, &rest[..open]);
                let (fields, source) = if rest[open..].starts_with("{0}") {
                    (&start, "startRange")
                } else {
                    (&end, "endRange")
                };
                parts.extend(
                    render_pattern(o, time_pattern, numbers_for(time_pattern), fields)
                        .into_iter()
                        .map(|(kind, value)| (kind, value, source)),
                );
                rest = &rest[open + 3..];
            }
            push_shared_literal(&mut parts, rest);
            parts
        };
        match patterns.date_and_time().filter(|_| same_day) {
            Some((date_pattern, time_pattern)) => {
                let mut parts = Vec::new();
                let mut rest = patterns.date_time.as_str();
                while let Some(open) = rest.find('{') {
                    push_shared_literal(&mut parts, &rest[..open]);
                    if rest[open..].starts_with("{0}") {
                        parts.extend(range(time_pattern));
                    } else {
                        parts.extend(
                            render_pattern(o, date_pattern, numbers_for(date_pattern), &start)
                                .into_iter()
                                .map(|(kind, value)| (kind, value, "shared")),
                        );
                    }
                    rest = &rest[open + 3..];
                }
                push_shared_literal(&mut parts, rest);
                parts
            }
            None => range(full),
        }
    };
    match patterns.get(field) {
        None if dtf_interval::field_unit_ignored(&patterns.full, field) => single(&start),
        None => fallback(&patterns.full),
        Some(RangePattern::Fallback(full)) => fallback(full),
        Some(RangePattern::Split {
            first,
            second,
            later_first,
        }) => {
            let (first_fields, first_source, second_fields, second_source) = if *later_first {
                (&end, "endRange", &start, "startRange")
            } else {
                (&start, "startRange", &end, "endRange")
            };
            let first_parts = render_pattern(o, first, "", first_fields);
            let second_parts = render_pattern(o, second, "", second_fields);
            split_range_parts(first_parts, first_source, second_parts, second_source)
        }
    }
}

/// Append a fallback or dateTimeFormat literal as a shared part.
fn push_shared_literal(parts: &mut Vec<(&'static str, String, &'static str)>, text: &str) {
    if !text.is_empty() {
        parts.push(("literal", text.to_string(), "shared"));
    }
}

/// The parts of an interval pattern's two halves. The literal at the split is the shared range
/// separator; a field shown once (with equal values) belongs to both dates, as do the literals
/// beside such fields.
fn split_range_parts(
    first: Vec<(&'static str, String)>,
    first_source: &'static str,
    second: Vec<(&'static str, String)>,
    second_source: &'static str,
) -> Vec<(&'static str, String, &'static str)> {
    let boundary = first.len();
    let mut combined: Vec<_> = first
        .into_iter()
        .map(|(kind, value)| (kind, value, first_source))
        .chain(
            second
                .into_iter()
                .map(|(kind, value)| (kind, value, second_source)),
        )
        .collect();
    if boundary > 0 && combined[boundary - 1].0 == "literal" {
        combined[boundary - 1].2 = "shared";
    } else if combined
        .get(boundary)
        .is_some_and(|part| part.0 == "literal")
    {
        combined[boundary].2 = "shared";
    }
    for index in 0..combined.len() {
        let kind = combined[index].0;
        if kind != "literal" && combined.iter().filter(|part| part.0 == kind).count() == 1 {
            combined[index].2 = "shared";
        }
    }
    for index in 0..combined.len() {
        if combined[index].0 != "literal" || combined[index].2 == "shared" {
            continue;
        }
        let previous_shared = combined[..index]
            .iter()
            .rev()
            .find(|part| part.0 != "literal")
            .is_some_and(|part| part.2 == "shared");
        let next_shared = combined[index + 1..]
            .iter()
            .find(|part| part.0 != "literal")
            .is_some_and(|part| part.2 == "shared");
        if previous_shared || next_shared {
            combined[index].2 = "shared";
        }
    }
    combined
}

/// FormatDateTime's parts (ECMA-402 #sec-partitiondatetimepattern) for epoch milliseconds and a
/// receiver kind (see [`dtf_ms_kind`]).
fn build_parts(o: &Gc, ms: f64, kind: u8) -> Vec<(&'static str, String)> {
    let (pattern, numbers) = formatter_pattern(o, kind);
    render_pattern(o, &pattern, &numbers, &local_fields(o, ms, kind))
}

/// Validate and canonicalize a `timeZone` option. Offset forms (`±HH`, `±HH:MM`, `±HHMM`) are
/// checked strictly (ASCII sign, hour 00-23, minute 00-59) and normalized to `±HH:MM`; a name is
/// looked up case-insensitively in the IANA registry (ECMA-402 GetAvailableNamedTimeZoneIdentifier)
/// and keeps its registry spelling, or is rejected.
fn canonicalize_time_zone(tz: &str) -> Option<String> {
    let first = tz.bytes().next()?;
    if first == b'+' || first == b'-' {
        return canon_utc_offset(tz);
    }
    // A named IANA zone keeps the given identifier (case-normalized), matching Temporal — the
    // resolved timeZone preserves an alias like Asia/Calcutta rather than canonicalizing it.
    crate::tz::registry_name(tz).map(|s| s.to_string())
}

/// The signed millisecond offset of a canonical `±HH:MM[:SS]` UTC-offset zone (0 for anything else).
fn tz_offset_ms(tz: &str) -> f64 {
    let b = tz.as_bytes();
    if b.first().is_none_or(|&c| c != b'+' && c != b'-') {
        return 0.0;
    }
    let sign = if b[0] == b'-' { -1.0 } else { 1.0 };
    let mut parts = tz[1..].split(':');
    let h: f64 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
    let m: f64 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
    let s: f64 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0.0);
    sign * (h * 3_600_000.0 + m * 60_000.0 + s * 1000.0)
}

/// Parse a strict minute-precision UTC offset string, returning the `±HH:MM` canonical form.
fn canon_utc_offset(s: &str) -> Option<String> {
    let sign = s.as_bytes()[0];
    let rest = &s[1..];
    if rest.len() < 2 || !rest.as_bytes()[..2].iter().all(u8::is_ascii_digit) {
        return None;
    }
    let h: u32 = rest[..2].parse().ok()?;
    if h > 23 {
        return None;
    }
    let after = &rest[2..];
    let m: u32 = if after.is_empty() {
        0
    } else {
        let digits = after.strip_prefix(':').unwrap_or(after);
        if digits.len() != 2 || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        digits.parse().ok()?
    };
    if m > 59 {
        return None;
    }
    // A zero offset canonicalizes to "+00:00" regardless of the written sign.
    let sign_c = if sign == b'-' && (h != 0 || m != 0) {
        '-'
    } else {
        '+'
    };
    Some(format!("{sign_c}{h:02}:{m:02}"))
}

fn resolved_options(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = crate::intl::brand_slot_legacy(i, &this, "#\u{0}dtf", "Intl.DateTimeFormat")?;
    let res = i.new_object();
    let put = |i: &mut Interp, res: &Gc, k: &str, slot: &str| {
        if let Some(v) = o.borrow().internal_slots().get(slot).map(|p| p.value()) {
            set_data(res, k, v);
        }
        let _ = i;
    };
    put(i, &res, "locale", "#\u{0}dtf_locale");
    put(i, &res, "calendar", "#\u{0}dtf_ca");
    put(i, &res, "numberingSystem", "#\u{0}dtf_nu");
    put(i, &res, "timeZone", "#\u{0}dtf_tz");
    // hourCycle/hour12 are resolved internally for every formatter but only surface in
    // resolvedOptions when the resolved pattern actually shows an hour.
    if matches!(
        o.borrow()
            .internal_slots()
            .get("#\u{0}dtf_hourshown")
            .map(|p| p.value()),
        Some(Value::Bool(true))
    ) {
        put(i, &res, "hourCycle", "#\u{0}dtf_hourcycle");
        put(i, &res, "hour12", "#\u{0}dtf_hour12");
    }
    put(i, &res, "weekday", "#\u{0}dtf_weekday");
    put(i, &res, "era", "#\u{0}dtf_era");
    put(i, &res, "year", "#\u{0}dtf_year");
    if o.borrow().internal_slots().contains("#\u{0}dtfr_month") {
        put(i, &res, "month", "#\u{0}dtfr_month");
    } else {
        put(i, &res, "month", "#\u{0}dtf_month");
    }
    put(i, &res, "day", "#\u{0}dtf_day");
    put(i, &res, "dayPeriod", "#\u{0}dtf_dayperiod");
    put(i, &res, "hour", "#\u{0}dtf_hour");
    put(i, &res, "minute", "#\u{0}dtf_minute");
    put(i, &res, "second", "#\u{0}dtf_second");
    put(i, &res, "fractionalSecondDigits", "#\u{0}dtf_fracsec");
    put(i, &res, "timeZoneName", "#\u{0}dtf_tzname");
    put(i, &res, "dateStyle", "#\u{0}dtf_datestyle");
    put(i, &res, "timeStyle", "#\u{0}dtf_timestyle");
    Ok(Value::Obj(res))
}
