//! `Intl.NumberFormat` with ECMA-402 rounding and generated CLDR 48 locale data.

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
};
use super::{ab, arg, canonicalize_locale_list, coerce_options, make_service};
use crate::interpreter::Interp;
use crate::value::ValueView;
use crate::value::{set_data, set_internal_slot, Gc, Value};

pub(crate) fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "NumberFormat", 0, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "format", 1, |i, this, a| {
        // `format` is a bound-ish getter in the spec; here it's a plain method (sufficient for most
        // tests) — but many tests read `.format` then call it, so return a stable per-instance fn.
        format_number(i, &this, &arg(a, 0))
    });
    it.def_method(&proto, "formatToParts", 1, |i, this, a| {
        format_to_parts(i, &this, &arg(a, 0))
    });
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
    it.def_method(&proto, "formatRange", 2, |i, this, a| {
        format_range(i, &this, &arg(a, 0), &arg(a, 1))
    });
    it.def_method(&proto, "formatRangeToParts", 2, |i, this, a| {
        format_range_to_parts(i, &this, &arg(a, 0), &arg(a, 1))
    });
    // A `format` accessor that returns a bound function is what the spec mandates; provide it.
    install_format_getter(it, &proto);
}

/// The two range endpoints as intl mathematical values, rejecting NaN/undefined per
/// FormatNumericRange step 1 (`start`/`end` must not be undefined; NaN throws RangeError).
fn range_endpoints(
    i: &mut Interp,
    this: &Value,
    x: &Value,
    y: &Value,
) -> Result<(Gc, IntlMv, IntlMv), Value> {
    let o = instance(i, this)?;
    if matches!(&(x).view(), ValueView::Undefined) || matches!(&(y).view(), ValueView::Undefined) {
        return Err(i.make_error("TypeError", "formatRange requires two arguments"));
    }
    let a = to_intl_mv(i, x)?;
    let b = to_intl_mv(i, y)?;
    if a.is_nan() || b.is_nan() {
        return Err(i.make_error("RangeError", "formatRange arguments must not be NaN"));
    }
    Ok((o, a, b))
}

fn format_range(i: &mut Interp, this: &Value, x: &Value, y: &Value) -> Result<Value, Value> {
    let (o, a, b) = range_endpoints(i, this, x, y)?;
    let sa = assemble_number_exact(i, &o, &a).text;
    let sb = assemble_number_exact(i, &o, &b).text;
    let nu = get_str(&o, "#\u{0}nf_nu");
    // Endpoints that FORMAT identically collapse to a single approximate value.
    if sa == sb {
        return Ok(Value::lstr(format!(
            "{}{}",
            cldr_number_symbols(&o).approximately,
            xlate_digits(&sa, &nu)
        )));
    }
    let (start, sep, end) = range_join(&o, &sa, &sb);
    Ok(Value::lstr(format!(
        "{}{}{}",
        xlate_digits(&start, &nu),
        sep,
        xlate_digits(&end, &nu)
    )))
}

/// The locale's range join: separator plus the ICU affix-collapsing rules (a prefix-sign start
/// keeps only its own affixes; a suffix-currency locale drops the start's suffix).
fn range_join(o: &Gc, sa: &str, sb: &str) -> (String, &'static str, String) {
    let lang = get_str(o, "#\u{0}nf_locale");
    let lang = lang.split('-').next().unwrap_or("en").to_string();
    let currency = get_str(o, "#\u{0}nf_style") == "currency";
    if lang == "pt" {
        // Suffix-currency: the start keeps only its digits ("3 - 5 €").
        let start = if currency {
            sa.trim_end_matches(|c: char| !c.is_ascii_digit())
                .to_string()
        } else {
            sa.to_string()
        };
        // A shared sign is shown on the start only.
        let end = if sa.starts_with('+') || sa.starts_with('-') {
            sb.trim_start_matches(['+', '-']).to_string()
        } else {
            sb.to_string()
        };
        return (start, " - ", end);
    }
    if currency {
        if sa.starts_with('+') || sa.starts_with('-') {
            // A signed start absorbs the shared prefix; the end shows bare digits.
            let end = sb
                .trim_start_matches(|c: char| !c.is_ascii_digit())
                .to_string();
            return (sa.to_string(), "\u{2013}", end);
        }
        return (sa.to_string(), " \u{2013} ", sb.to_string());
    }
    (sa.to_string(), "\u{2013}", sb.to_string())
}

fn format_range_to_parts(
    i: &mut Interp,
    this: &Value,
    x: &Value,
    y: &Value,
) -> Result<Value, Value> {
    let (o, a, b) = range_endpoints(i, this, x, y)?;
    let stype = suffix_type_of(&o);
    let symbols = cldr_number_symbols(&o);
    let nu = get_str(&o, "#\u{0}nf_nu");
    let mut out: Vec<Value> = Vec::new();
    let push_parts = |i: &mut Interp, whole: &str, source: &str, out: &mut Vec<Value>| {
        for (t, mut v) in decompose_parts(whole, &stype, &symbols) {
            let ob = i.new_object();
            set_data(&ob, "type", Value::lstr(t));
            if matches!(t, "integer" | "fraction" | "exponentInteger") {
                v = xlate_digits(&v, &nu);
            }
            set_data(&ob, "value", Value::lstr(v));
            set_data(&ob, "source", Value::lstr(source));
            out.push(Value::Obj(ob));
        }
    };
    let sa = assemble_number_exact(i, &o, &a).text;
    let sb = assemble_number_exact(i, &o, &b).text;
    if sa == sb {
        let approx = i.new_object();
        set_data(&approx, "type", Value::lstr("approximatelySign"));
        set_data(
            &approx,
            "value",
            Value::lstr(cldr_number_symbols(&o).approximately),
        );
        set_data(&approx, "source", Value::lstr("shared"));
        out.push(Value::Obj(approx));
        push_parts(i, &sa, "shared", &mut out);
        return Ok(i.make_array(out));
    }
    let (start, sep, end) = range_join(&o, &sa, &sb);
    push_parts(i, &start, "startRange", &mut out);
    let lit = i.new_object();
    set_data(&lit, "type", Value::lstr("literal"));
    set_data(&lit, "value", Value::lstr(sep));
    set_data(&lit, "source", Value::lstr("shared"));
    out.push(Value::Obj(lit));
    push_parts(i, &end, "endRange", &mut out);
    Ok(i.make_array(out))
}

fn install_format_getter(it: &mut Interp, proto: &Gc) {
    let g = it.make_native("get format", 0, |i, this, _| {
        let o = instance_unwrap(i, &this)?;
        // Cache a bound function on the instance so repeated reads return the same object.
        if let Some(f) = o
            .borrow()
            .internal_slots()
            .get("#\u{0}nf_boundformat")
            .map(|p| p.value())
        {
            return Ok(f);
        }
        let f = i.make_native("", 1, |i, that, a| format_number(i, &that, &arg(a, 0)));
        // Bind `this` = the (possibly unwrapped) NumberFormat instance.
        let bound = crate::intl::numberformat::bind_this(i, Value::Obj(f), Value::Obj(o));
        set_internal_slot(&o, "#\u{0}nf_boundformat", bound);
        Ok(bound)
    });
    proto.borrow_mut().props.insert(
        "format",
        crate::value::Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
    );
}

/// Bind `this_arg` onto `target` via Function.prototype.bind, then normalise the result to match the
/// spec's `format`/`compare` bound functions: name `""`, length 1, and not a constructor.
pub(crate) fn bind_this(i: &mut Interp, target: Value, this_arg: Value) -> Value {
    if let Ok(bindfn) = i.get_member_key(&target, "bind") {
        if let Ok(bound) = i.call(bindfn, target, &[this_arg]) {
            if let ValueView::Obj(o) = &(bound).view() {
                o.borrow_mut().is_constructor = false;
                o.borrow_mut().props.insert(
                    "name",
                    crate::value::Property::data(Value::lstr(""), false, false, true),
                );
            }
            return bound;
        }
    }
    target
}

pub(super) struct DigitOpts {
    pub(super) min_int: u32,
    pub(super) min_frac: u32,
    pub(super) max_frac: u32,
    pub(super) min_sig: Option<u32>,
    pub(super) max_sig: Option<u32>,
    /// "significant" | "fraction" | "morePrecision" | "lessPrecision".
    pub(super) rounding_type: &'static str,
}

/// The raw significant/fraction-digit options, read in spec order before the rounding options.
pub(super) struct RawDigits {
    min_int: u32,
    mnfd: Option<u32>,
    mxfd: Option<u32>,
    mnsd: Option<u32>,
    mxsd: Option<u32>,
}

fn construct(i: &mut Interp, t: Value, a: &[Value]) -> Result<Value, Value> {
    // Legacy service: callable without `new` (returns a fresh instance either way).
    let requested = canonicalize_locale_list(i, &arg(a, 0))?;
    let options = coerce_options(i, &arg(a, 1))?;
    read_locale_matcher(i, &options)?;
    // numberingSystem is read right after localeMatcher, and must be a valid type identifier.
    let numbering = get_option(i, &options, "numberingSystem", &[], None)?;
    if let Some(ns) = &numbering {
        if !ns
            .split('-')
            .all(|p| p.len() >= 3 && p.len() <= 8 && p.bytes().all(|b| b.is_ascii_alphanumeric()))
        {
            return Err(i.make_error("RangeError", format!("invalid numberingSystem: {ns}")));
        }
    }
    let (resolved_locale, numbering) =
        super::service::resolve_locale_nu(&requested, numbering.as_deref());

    let style = get_option(
        i,
        &options,
        "style",
        &["decimal", "percent", "currency", "unit"],
        Some("decimal"),
    )?
    .unwrap();

    // currency
    let currency = get_option(i, &options, "currency", &[], None)?;
    if style == "currency" && currency.is_none() {
        return Err(i.make_error("TypeError", "currency is required for currency style"));
    }
    if let Some(c) = &currency {
        if !is_well_formed_currency(c) {
            return Err(i.make_error("RangeError", format!("invalid currency: {c}")));
        }
    }
    let currency_display = get_option(
        i,
        &options,
        "currencyDisplay",
        &["code", "symbol", "narrowSymbol", "name"],
        Some("symbol"),
    )?
    .unwrap();
    let currency_sign = get_option(
        i,
        &options,
        "currencySign",
        &["standard", "accounting"],
        Some("standard"),
    )?
    .unwrap();

    // unit
    let unit = get_option(i, &options, "unit", &[], None)?;
    if style == "unit" && unit.is_none() {
        return Err(i.make_error("TypeError", "unit is required for unit style"));
    }
    if let Some(u) = &unit {
        if !is_well_formed_unit(u) {
            return Err(i.make_error("RangeError", format!("invalid unit: {u}")));
        }
    }
    let unit_display = get_option(
        i,
        &options,
        "unitDisplay",
        &["short", "narrow", "long"],
        Some("short"),
    )?
    .unwrap();

    // notation is read before the digit options.
    let notation = get_option(
        i,
        &options,
        "notation",
        &["standard", "scientific", "engineering", "compact"],
        Some("standard"),
    )?
    .unwrap();

    // digit options (minInt, min/maxFrac, min/maxSig), then the rounding options.
    let cur_digits = currency
        .as_deref()
        .map(|c| currency_fraction_digits(&c.to_uppercase()))
        .unwrap_or(2);
    let raw_digits = read_raw_digits(i, &options)?;

    let rounding_increment = {
        let v = ab(i.get_member_key(&options, "roundingIncrement"))?;
        if matches!((v).view(), ValueView::Undefined) {
            1u32
        } else {
            let n = ab(i.to_number(&v))?;
            let allowed = [
                1u32, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000,
            ];
            if n.fract() != 0.0 || !allowed.contains(&(n as u32)) {
                return Err(i.make_error("RangeError", "invalid roundingIncrement"));
            }
            n as u32
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
    let digits = interpret_digits(
        i,
        &raw_digits,
        &style,
        &notation,
        &rounding_priority,
        cur_digits,
        rounding_increment,
    )?;
    // A roundingIncrement other than 1 requires fraction-digit rounding (TypeError otherwise), and
    // then maximumFractionDigits must equal minimumFractionDigits (RangeError otherwise).
    if rounding_increment != 1 {
        if digits.rounding_type != "fraction" {
            return Err(i.make_error(
                "TypeError",
                "roundingIncrement is only supported with fractionDigits rounding",
            ));
        }
        if digits.min_frac != digits.max_frac {
            return Err(i.make_error(
                "RangeError",
                "maximumFractionDigits must equal minimumFractionDigits with roundingIncrement",
            ));
        }
    }
    let trailing_zero = get_option(
        i,
        &options,
        "trailingZeroDisplay",
        &["auto", "stripIfInteger"],
        Some("auto"),
    )?
    .unwrap();

    let compact_display = get_option(
        i,
        &options,
        "compactDisplay",
        &["short", "long"],
        Some("short"),
    )?
    .unwrap();
    let use_grouping = read_use_grouping(i, &options, &notation)?;
    let sign_display = get_option(
        i,
        &options,
        "signDisplay",
        &["auto", "never", "always", "exceptZero", "negative"],
        Some("auto"),
    )?
    .unwrap();

    let obj = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.NumberFormat")? {
        obj.borrow_mut().proto = Some(proto);
    }
    set_internal_slot(&obj, "#\u{0}nf", Value::Bool(true));
    set_internal_slot(&obj, "#\u{0}nf_locale", Value::lstr(resolved_locale));
    set_internal_slot(&obj, "#\u{0}nf_nu", Value::lstr(numbering));
    set_internal_slot(
        &obj,
        "#\u{0}nf_roundingincrement",
        Value::Num(rounding_increment as f64),
    );
    set_internal_slot(
        &obj,
        "#\u{0}nf_roundingpriority",
        Value::lstr(rounding_priority),
    );
    set_internal_slot(&obj, "#\u{0}nf_trailingzero", Value::lstr(trailing_zero));
    set_internal_slot(&obj, "#\u{0}nf_style", Value::lstr(style));
    if let Some(c) = currency.filter(|_| get_str(&obj, "#\u{0}nf_style") == "currency") {
        set_internal_slot(&obj, "#\u{0}nf_currency", Value::lstr(c.to_uppercase()));
        set_internal_slot(
            &obj,
            "#\u{0}nf_currencydisplay",
            Value::lstr(currency_display),
        );
        set_internal_slot(&obj, "#\u{0}nf_currencysign", Value::lstr(currency_sign));
    }
    if let Some(u) = unit {
        set_internal_slot(&obj, "#\u{0}nf_unit", Value::lstr(u));
        set_internal_slot(&obj, "#\u{0}nf_unitdisplay", Value::lstr(unit_display));
    }
    set_internal_slot(&obj, "#\u{0}nf_minint", Value::Num(digits.min_int as f64));
    set_internal_slot(&obj, "#\u{0}nf_minfrac", Value::Num(digits.min_frac as f64));
    set_internal_slot(&obj, "#\u{0}nf_maxfrac", Value::Num(digits.max_frac as f64));
    if let Some(v) = digits.min_sig {
        set_internal_slot(&obj, "#\u{0}nf_minsig", Value::Num(v as f64));
    }
    if let Some(v) = digits.max_sig {
        set_internal_slot(&obj, "#\u{0}nf_maxsig", Value::Num(v as f64));
    }
    set_internal_slot(&obj, "#\u{0}nf_notation", Value::lstr(notation));
    set_internal_slot(
        &obj,
        "#\u{0}nf_compactdisplay",
        Value::lstr(compact_display),
    );
    set_internal_slot(&obj, "#\u{0}nf_grouping", use_grouping);
    set_internal_slot(&obj, "#\u{0}nf_signdisplay", Value::lstr(sign_display));
    set_internal_slot(&obj, "#\u{0}nf_roundingmode", Value::lstr(rounding_mode));
    set_internal_slot(
        &obj,
        "#\u{0}nf_roundingtype",
        Value::lstr(digits.rounding_type),
    );
    // Legacy "ChainNumberFormat" (see datetimeformat.rs).
    if !i.constructing {
        if let Some(chained) = crate::intl::legacy_chain(i, &t, "Intl.NumberFormat", &obj) {
            return Ok(chained);
        }
    }
    Ok(Value::Obj(obj))
}

/// The CLDR default fraction-digit count for a currency (ISO 4217 minor units).
fn currency_fraction_digits(code: &str) -> u32 {
    crate::cldr_numbers::currency_digits(code)
}

/// SetNumberFormatDigitOptions, phase 1: read the raw digit options (spec order: minInt, minFrac,
/// maxFrac, minSig, maxSig). The rounding options are read by the caller afterwards; interpretation
/// happens in `interpret_digits` once the rounding priority is known.
pub(super) fn read_raw_digits(i: &mut Interp, options: &Value) -> Result<RawDigits, Value> {
    let min_int = read_range(i, options, "minimumIntegerDigits", 1, 21, 1)?;
    let mnfd = read_range_opt(i, options, "minimumFractionDigits", 0, 100)?;
    let mxfd = read_range_opt(i, options, "maximumFractionDigits", 0, 100)?;
    let mnsd = read_range_opt(i, options, "minimumSignificantDigits", 1, 21)?;
    let mxsd = read_range_opt(i, options, "maximumSignificantDigits", 1, 21)?;
    Ok(RawDigits {
        min_int,
        mnfd,
        mxfd,
        mnsd,
        mxsd,
    })
}

/// SetNumberFormatDigitOptions, phase 2: derive the effective digit bounds and rounding type from the
/// raw options, the style/notation defaults, and the rounding priority.
pub(super) fn interpret_digits(
    i: &mut Interp,
    raw: &RawDigits,
    style: &str,
    notation: &str,
    priority: &str,
    cur_digits: u32,
    rounding_increment: u32,
) -> Result<DigitOpts, Value> {
    // The currency minor-unit defaults only apply in standard notation.
    let (mnfd_default, mut mxfd_default) = if style == "currency" && notation == "standard" {
        (cur_digits, cur_digits)
    } else if style == "percent" {
        (0, 0)
    } else {
        (0, 3)
    };
    if rounding_increment != 1 {
        mxfd_default = mnfd_default;
    }
    let has_sd = raw.mnsd.is_some() || raw.mxsd.is_some();
    let has_fd = raw.mnfd.is_some() || raw.mxfd.is_some();
    let mut need_sd = true;
    let mut need_fd = true;
    if priority == "auto" {
        need_sd = has_sd;
        if need_sd || (!has_fd && notation == "compact") {
            need_fd = false;
        }
    }

    let (mut min_sig, mut max_sig) = (None, None);
    if need_sd {
        if has_sd {
            let mn = raw.mnsd.unwrap_or(1);
            let mx = raw.mxsd.unwrap_or(21);
            if mn > mx {
                return Err(i.make_error(
                    "RangeError",
                    "minimumSignificantDigits > maximumSignificantDigits",
                ));
            }
            min_sig = Some(mn);
            max_sig = Some(mx);
        } else {
            min_sig = Some(1);
            max_sig = Some(21);
        }
    }

    let (mut min_frac, mut max_frac) = (0u32, 0u32);
    if need_fd {
        let (mn, mx) = if has_fd {
            match (raw.mnfd, raw.mxfd) {
                (None, Some(mx)) => (mnfd_default.min(mx), mx),
                (Some(mn), None) => (mn, mxfd_default.max(mn)),
                (Some(mn), Some(mx)) => {
                    if mn > mx {
                        return Err(i.make_error(
                            "RangeError",
                            "minimumFractionDigits > maximumFractionDigits",
                        ));
                    }
                    (mn, mx)
                }
                (None, None) => (mnfd_default, mxfd_default),
            }
        } else {
            (mnfd_default, mxfd_default)
        };
        min_frac = mn;
        max_frac = mx;
    }

    let rounding_type = if !need_sd && !need_fd {
        // Neither range requested (compact, auto): the default rounding is morePrecision over 0
        // fraction and (1..2) significant digits.
        min_frac = 0;
        max_frac = 0;
        min_sig = Some(1);
        max_sig = Some(2);
        "morePrecision"
    } else if priority == "morePrecision" {
        "morePrecision"
    } else if priority == "lessPrecision" {
        "lessPrecision"
    } else if has_sd {
        "significant"
    } else {
        "fraction"
    };

    Ok(DigitOpts {
        min_int: raw.min_int,
        min_frac,
        max_frac,
        min_sig,
        max_sig,
        rounding_type,
    })
}

fn read_range(
    i: &mut Interp,
    options: &Value,
    prop: &str,
    lo: u32,
    hi: u32,
    fallback: u32,
) -> Result<u32, Value> {
    Ok(read_range_opt(i, options, prop, lo, hi)?.unwrap_or(fallback))
}
fn read_range_opt(
    i: &mut Interp,
    options: &Value,
    prop: &str,
    lo: u32,
    hi: u32,
) -> Result<Option<u32>, Value> {
    let v = ab(i.get_member_key(options, prop))?;
    if matches!((v).view(), ValueView::Undefined) {
        return Ok(None);
    }
    let n = ab(i.to_number(&v))?;
    if n.is_nan() {
        return Err(i.make_error("RangeError", format!("{prop} is NaN")));
    }
    let f = n.floor();
    if f < lo as f64 || f > hi as f64 {
        return Err(i.make_error("RangeError", format!("{prop} out of range")));
    }
    Ok(Some(f as u32))
}

fn read_use_grouping(i: &mut Interp, options: &Value, notation: &str) -> Result<Value, Value> {
    // GetStringOrBooleanOption: undefined → fallback; `true` → "always"; a falsy value → false;
    // otherwise ToString and validate against ["min2","auto","always"] (so the string "true" is a
    // RangeError, not "always"). The fallback is "auto" (or "min2" for compact notation).
    let v = ab(i.get_member_key(options, "useGrouping"))?;
    let default = Value::lstr(if notation == "compact" {
        "min2"
    } else {
        "auto"
    });
    if matches!((v).view(), ValueView::Undefined) {
        return Ok(default);
    }
    if matches!((v).view(), ValueView::Bool(true)) {
        return Ok(Value::lstr("always"));
    }
    if !i.to_boolean(&v) {
        return Ok(Value::Bool(false));
    }
    let s = ab(i.to_string(&v))?.to_string();
    // The strings "true"/"false" fall back to the default (they are not valid values, but
    // GetBooleanOrStringNumberFormatOption returns the fallback for them instead of throwing).
    if s == "true" || s == "false" {
        return Ok(default);
    }
    if !["always", "auto", "min2"].contains(&s.as_str()) {
        return Err(i.make_error("RangeError", format!("invalid useGrouping: {s}")));
    }
    Ok(Value::lstr(s))
}

fn is_well_formed_currency(c: &str) -> bool {
    c.len() == 3 && c.bytes().all(|b| b.is_ascii_alphabetic())
}
fn is_well_formed_unit(u: &str) -> bool {
    // A sanctioned single unit, or "X-per-Y" with both X and Y sanctioned (IsWellFormedUnitIdentifier).
    let sanctioned = |s: &str| crate::units::SANCTIONED_UNITS.contains(&s);
    match u.split_once("-per-") {
        Some((a, b)) => sanctioned(a) && sanctioned(b),
        None => sanctioned(u),
    }
}

// ---- formatting ------------------------------------------------------------------------------

fn instance(i: &mut Interp, this: &Value) -> Result<Gc, Value> {
    brand_slot(i, this, "#\u{0}nf")
}

/// UnwrapNumberFormat: like `instance`, but follows a legacy chained receiver.
fn instance_unwrap(i: &mut Interp, this: &Value) -> Result<Gc, Value> {
    crate::intl::brand_slot_legacy(i, this, "#\u{0}nf", "Intl.NumberFormat")
}

fn get_str(o: &Gc, k: &str) -> String {
    match (o.borrow().internal_slots().get(k).map(|p| p.value())).map(|__value| (__value).view()) {
        Some(ValueView::Str(s)) => s.to_string(),
        _ => String::new(),
    }
}

fn cldr_number_locale(o: &Gc) -> &'static str {
    super::service::data_locale(&get_str(o, "#\u{0}nf_locale"))
}

/// The formatter's number symbols: a currency format separates with the locale's
/// currencyDecimal and currencyGroup (UTS #35 Part 3 Number Symbols), which default to the
/// ordinary decimal and group symbols.
fn cldr_number_symbols(o: &Gc) -> crate::cldr_numbers::Symbols {
    let mut symbols =
        crate::cldr_numbers::symbols(cldr_number_locale(o), &get_str(o, "#\u{0}nf_nu"));
    if get_str(o, "#\u{0}nf_style") == "currency" {
        symbols.decimal = symbols.currency_decimal;
        symbols.group = symbols.currency_group;
        // A currency's own separators (UTS #35 Part 3 Currencies) replace the locale's.
        if let Some(format) = crate::cldr_numbers::currency_format(
            cldr_number_locale(o),
            &get_str(o, "#\u{0}nf_currency"),
        ) {
            if !format.decimal.is_empty() {
                symbols.decimal = format.decimal;
            }
            if !format.group.is_empty() {
                symbols.group = format.group;
            }
        }
    }
    symbols
}

fn get_num(o: &Gc, k: &str) -> Option<u32> {
    match (o.borrow().internal_slots().get(k).map(|p| p.value())).map(|__value| (__value).view()) {
        Some(ValueView::Num(n)) => Some(n as u32),
        _ => None,
    }
}

/// `FormatNumericToString`'s decimal rounding and minimum-integer padding for a Number, shared
/// with `Intl.PluralRules`: the Number's exact value is its Number::toString digits.
#[allow(clippy::too_many_arguments)]
pub(super) fn format_magnitude_options(
    x: f64,
    min_int: u32,
    min_frac: u32,
    max_frac: u32,
    min_sig: Option<u32>,
    max_sig: Option<u32>,
    increment: u32,
    mode: &str,
    rtype: &str,
) -> String {
    exact_magnitude_options(
        &ExactDec::from_f64(x),
        min_int,
        min_frac,
        max_frac,
        min_sig,
        max_sig,
        increment,
        mode,
        rtype,
    )
}

/// The shortest round-trip decimal digits of `|x|` as (integer_digits, fraction_digits). Rounding is
/// performed on this decimal expansion (not the binary f64) so `1.015` rounds like the source `1.015`.
fn decimal_digits(x: f64) -> (String, String) {
    let s = format!("{}", x.abs());
    match s.split_once('.') {
        Some((a, b)) => (a.to_string(), b.to_string()),
        None => (s, String::new()),
    }
}

/// Round the decimal `int_str.frac_str` to `keep` fraction digits (may be negative, rounding integer
/// places), snapping to a multiple of `inc` at that scale, per the ECMA-402 `mode`. Returns the exact
/// decimal string (unsigned, `keep` fraction digits when `keep > 0`, no min-frac padding).
///
/// ToRawPrecision/ToRawFixed choose between the two candidates around the value with
/// ApplyUnsignedRoundingMode (#sec-applyunsignedroundingmode). The arithmetic is exact on the
/// digit strings: every permitted rounding increment divides 10^4, so the coefficient's residue
/// modulo the increment (and the parity of its quotient) comes from its last five digits.
fn round_decimal(
    int_str: &str,
    frac_str: &str,
    keep: i32,
    inc: u32,
    mode: &str,
    negative: bool,
) -> String {
    let inc = inc.max(1);
    let digits: Vec<u8> = int_str
        .bytes()
        .chain(frac_str.bytes())
        .map(|b| b - b'0')
        .collect();
    let cut = int_str.len() as i64 + i64::from(keep);
    // The retained coefficient (in units of 10^-keep) and the discarded tail, as digits.
    let (mut coefficient, discarded): (Vec<u8>, &[u8]) = if cut <= 0 {
        (vec![0], &digits[..])
    } else {
        let cut = cut as usize;
        let mut retained = digits[..cut.min(digits.len())].to_vec();
        retained.resize(cut, 0);
        (retained, &digits[cut.min(digits.len())..])
    };
    // When everything is discarded below the first retained place, leading zeros of the tail
    // that lie between the retained place and the digits make the tail smaller still.
    let tail_scale_zeros = if cut < 0 { (-cut) as usize } else { 0 };
    let low_digits = |coefficient: &[u8], count: usize| -> u32 {
        coefficient[coefficient.len().saturating_sub(count)..]
            .iter()
            .fold(0u32, |acc, &digit| acc * 10 + u32::from(digit))
    };
    let remainder = low_digits(&coefficient, 4) % inc;
    let quotient_odd = (low_digits(&coefficient, 5) % (2 * inc)) / inc == 1;
    let tail_nonzero = discarded.iter().any(|&digit| digit != 0);
    // Compare the discarded position (remainder + tail, in units) with half an increment:
    // 2·remainder + 2·tail against inc, with 0 ≤ tail < 1.
    let twice = 2 * remainder;
    let versus_half = if twice > inc {
        std::cmp::Ordering::Greater
    } else if twice + 2 <= inc {
        std::cmp::Ordering::Less
    } else if twice == inc {
        if tail_nonzero {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    } else {
        // inc = 2·remainder + 1: compare the tail with one half.
        if tail_scale_zeros > 0 || discarded.is_empty() {
            std::cmp::Ordering::Less
        } else {
            match discarded[0].cmp(&5) {
                std::cmp::Ordering::Equal if discarded[1..].iter().any(|&d| d != 0) => {
                    std::cmp::Ordering::Greater
                }
                other => other,
            }
        }
    };
    let inexact = remainder != 0 || tail_nonzero;
    let up = round_up_decision(mode, negative, inexact, versus_half, quotient_odd);
    // coefficient - remainder (+ inc when rounding up), in decimal digits.
    sub_small(&mut coefficient, remainder);
    if up {
        add_small(&mut coefficient, inc);
    }
    let text: String = coefficient.iter().map(|d| (d + b'0') as char).collect();
    let trimmed = text.trim_start_matches('0');
    place_decimal(if trimmed.is_empty() { "0" } else { trimmed }, keep)
}

/// `digits -= value` for a decimal digit vector no smaller than `value`.
fn sub_small(digits: &mut [u8], mut value: u32) {
    for digit in digits.iter_mut().rev() {
        if value == 0 {
            break;
        }
        let take = (value % 10) as u8;
        value /= 10;
        if *digit >= take {
            *digit -= take;
        } else {
            *digit += 10 - take;
            value += 1;
        }
    }
}

/// `digits += value`, growing the vector on a final carry.
fn add_small(digits: &mut Vec<u8>, mut value: u32) {
    for digit in digits.iter_mut().rev() {
        if value == 0 {
            return;
        }
        let sum = u32::from(*digit) + value % 10;
        value /= 10;
        *digit = (sum % 10) as u8;
        value += sum / 10;
    }
    while value > 0 {
        digits.insert(0, (value % 10) as u8);
        value /= 10;
    }
}

/// Reconstruct a decimal string from an integer coefficient at scale `10^-keep`.
fn place_decimal(coeff: &str, keep: i32) -> String {
    if keep <= 0 && coeff == "0" {
        return "0".to_string();
    }
    if keep <= 0 {
        return format!("{coeff}{}", "0".repeat((-keep) as usize));
    }
    let keep = keep as usize;
    let padded = if coeff.len() <= keep {
        format!("{:0>width$}", coeff, width = keep + 1)
    } else {
        coeff.to_string()
    };
    let split = padded.len() - keep;
    format!("{}.{}", &padded[..split], &padded[split..])
}

/// Whether to round the coefficient up by one increment: ApplyUnsignedRoundingMode for the
/// ECMA-402 `mode` (GetUnsignedRoundingMode maps it by sign), given whether the value was
/// inexact, how its discarded position compares with half an increment, and whether the
/// rounded-down quotient is odd (for halfEven).
fn round_up_decision(
    mode: &str,
    negative: bool,
    inexact: bool,
    versus_half: std::cmp::Ordering,
    quotient_odd: bool,
) -> bool {
    use std::cmp::Ordering::{Equal, Greater};
    if !inexact {
        return false;
    }
    match mode {
        "trunc" => false,
        "expand" => true,
        "ceil" => !negative,
        "floor" => negative,
        "halfTrunc" => versus_half == Greater,
        "halfCeil" => versus_half == Greater || (versus_half == Equal && !negative),
        "halfFloor" => versus_half == Greater || (versus_half == Equal && negative),
        "halfEven" => versus_half == Greater || (versus_half == Equal && quotient_odd),
        // halfExpand, the default.
        _ => versus_half != std::cmp::Ordering::Less,
    }
}

fn group_integer(
    int_part: &str,
    grouping: &Value,
    sep: &str,
    sizes: (usize, usize),
    minimum_grouping: usize,
) -> String {
    let enabled = !matches!(&(grouping).view(), ValueView::Bool(false));
    let min2 = matches!(&(grouping).view(), ValueView::Str(s) if &**s == "min2")
        || (minimum_grouping >= 2
            && matches!(&(grouping).view(), ValueView::Str(s) if &**s == "auto"));
    let (primary, secondary) = sizes;
    if !enabled || int_part.len() <= primary || (min2 && int_part.len() <= primary + 1) {
        return int_part.to_string();
    }
    // Insert separators right-to-left: the first (rightmost) group is `primary` digits, the rest are
    // `secondary` digits (Indian-style 3;2 when they differ).
    let digits: Vec<char> = int_part.chars().collect();
    let n = digits.len();
    let mut out = String::new();
    for (idx, c) in digits.iter().enumerate() {
        let from_right = n - idx;
        let boundary =
            idx > 0 && from_right >= primary && (from_right - primary).is_multiple_of(secondary);
        if boundary {
            out.push_str(sep);
        }
        out.push(*c);
    }
    out
}

/// An exact decimal value (sign + integer/fraction digit strings) that must not round through
/// f64: a BigInt, or a plain decimal string input (ToIntlMathematicalValue keeps exact digits).
#[derive(Clone)]
pub(crate) struct ExactDec {
    pub int: String,
    pub frac: String,
    pub negative: bool,
}

impl ExactDec {
    /// Parse a simple decimal literal (`[+-]?digits[.digits]`, surrounding whitespace allowed);
    /// anything fancier (exponents, hex, Infinity) falls back to the f64 path.
    fn parse(s: &str) -> Option<ExactDec> {
        let t = s.trim();
        let negative = t.starts_with('-');
        let t = t.strip_prefix(['-', '+']).unwrap_or(t);
        let (int, frac) = match t.split_once('.') {
            Some((a, b)) => (a, b),
            None => (t, ""),
        };
        if int.is_empty() && frac.is_empty() {
            return None;
        }
        if !int.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut i = int.trim_start_matches('0').to_string();
        if i.is_empty() {
            i.push('0');
        }
        Some(ExactDec {
            int: i,
            frac: frac.to_string(),
            negative,
        })
    }
}

impl ExactDec {
    /// The decimal `0.digits × 10^point`, canonicalized: no leading integer zeros, no trailing
    /// fraction zeros.
    fn from_digits(digits: &str, point: i64, negative: bool) -> ExactDec {
        let len = digits.len() as i64;
        let (int, frac) = if point <= 0 {
            (
                String::new(),
                format!("{}{digits}", "0".repeat((-point) as usize)),
            )
        } else if point >= len {
            (
                format!("{digits}{}", "0".repeat((point - len) as usize)),
                String::new(),
            )
        } else {
            let (int, frac) = digits.split_at(point as usize);
            (int.to_string(), frac.to_string())
        };
        let int = int.trim_start_matches('0');
        ExactDec {
            int: if int.is_empty() { "0" } else { int }.to_string(),
            frac: frac.trim_end_matches('0').to_string(),
            negative,
        }
    }

    /// The exact value of a finite Number via Number::toString(x, 10), as ToIntlMathematicalValue
    /// prescribes (so 0.1 is exactly one tenth, not its binary expansion).
    pub(crate) fn from_f64(x: f64) -> ExactDec {
        let (int, frac) = decimal_digits(x);
        ExactDec::from_digits(
            &format!("{int}{frac}"),
            int.len() as i64,
            x.is_sign_negative(),
        )
    }
}

/// An ECMA-402 Intl mathematical value: a mathematical value (a [`ExactDec`]; a negative zero
/// is a zero with `negative` set), positive or negative infinity, or not-a-number.
#[derive(Clone)]
pub(crate) enum IntlMv {
    Finite(ExactDec),
    PositiveInfinity,
    NegativeInfinity,
    NotANumber,
}

impl IntlMv {
    pub(crate) fn is_nan(&self) -> bool {
        matches!(self, IntlMv::NotANumber)
    }

    /// The Number nearest the value: used only for NaN, infinity and sign tests.
    fn approximate(&self) -> f64 {
        match self {
            IntlMv::Finite(decimal) => exact_f64(decimal),
            IntlMv::PositiveInfinity => f64::INFINITY,
            IntlMv::NegativeInfinity => f64::NEG_INFINITY,
            IntlMv::NotANumber => f64::NAN,
        }
    }

    fn exact(&self) -> Option<ExactDec> {
        match self {
            IntlMv::Finite(decimal) => Some(decimal.clone()),
            _ => None,
        }
    }
}

/// ECMA-402 #sec-tointlmathematicalvalue (snapshot b1c961988b9a).
pub(crate) fn to_intl_mv(i: &mut Interp, value: &Value) -> Result<IntlMv, Value> {
    let primitive = ab(i.to_primitive(value, crate::eval::Hint::Number))?;
    match primitive.view() {
        // A BigInt is its exact value, without RoundMVResult.
        ValueView::BigInt(b) => {
            let text = b.to_string_radix(10);
            let digits = text.trim_start_matches('-');
            Ok(IntlMv::Finite(ExactDec::from_digits(
                digits,
                digits.len() as i64,
                b.is_negative(),
            )))
        }
        ValueView::Str(s) => Ok(string_intl_mv(&s)),
        _ => {
            let x = ab(i.to_number(&primitive))?;
            if x == 0.0 && x.is_sign_negative() {
                return Ok(IntlMv::Finite(ExactDec::from_digits("0", 1, true)));
            }
            Ok(string_intl_mv(&i.num_to_str(x)))
        }
    }
}

/// ParseText(str, StringNumericLiteral) and its StringIntlMV, with ToIntlMathematicalValue's
/// RoundMVResult steps: a value whose magnitude rounds to an infinite Number is an infinity and
/// one that rounds to zero is a (signed) zero; any other value keeps every digit.
fn string_intl_mv(text: &str) -> IntlMv {
    let t = text.trim_matches(crate::regex::js_whitespace);
    if t.is_empty() {
        return IntlMv::Finite(ExactDec::from_digits("0", 1, false));
    }
    match t {
        "Infinity" | "+Infinity" => return IntlMv::PositiveInfinity,
        "-Infinity" => return IntlMv::NegativeInfinity,
        _ => {}
    }
    // NonDecimalIntegerLiteral: unsigned, exact.
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = t.strip_prefix(prefix) {
            return match crate::bigint::JsBigInt::parse_radix(digits, radix) {
                Some(value) if !digits.is_empty() && !digits.contains(['+', '-']) => {
                    let decimal = value.to_string_radix(10);
                    let decimal = ExactDec::from_digits(&decimal, decimal.len() as i64, false);
                    if exact_f64(&decimal).is_infinite() {
                        IntlMv::PositiveInfinity
                    } else {
                        IntlMv::Finite(decimal)
                    }
                }
                _ => IntlMv::NotANumber,
            };
        }
    }
    // StrDecimalLiteral: [+-] (Digits [. Digits?] | . Digits) [(e|E) [+-] Digits].
    let bytes = t.as_bytes();
    let mut at = 0;
    let negative = bytes[0] == b'-';
    if matches!(bytes[0], b'+' | b'-') {
        at = 1;
    }
    let int_start = at;
    while at < bytes.len() && bytes[at].is_ascii_digit() {
        at += 1;
    }
    let int_digits = &t[int_start..at];
    let mut frac_digits = "";
    if at < bytes.len() && bytes[at] == b'.' {
        at += 1;
        let frac_start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        frac_digits = &t[frac_start..at];
    }
    if int_digits.is_empty() && frac_digits.is_empty() {
        return IntlMv::NotANumber;
    }
    let mut exponent: i64 = 0;
    if at < bytes.len() && matches!(bytes[at], b'e' | b'E') {
        at += 1;
        let exponent_negative = at < bytes.len() && bytes[at] == b'-';
        if at < bytes.len() && matches!(bytes[at], b'+' | b'-') {
            at += 1;
        }
        let exponent_start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            // Saturate: any exponent this large already rounds to an infinity or zero below.
            exponent = exponent
                .saturating_mul(10)
                .saturating_add(i64::from(bytes[at] - b'0'))
                .min(1 << 40);
            at += 1;
        }
        if at == exponent_start {
            return IntlMv::NotANumber;
        }
        if exponent_negative {
            exponent = -exponent;
        }
    }
    if at != bytes.len() {
        return IntlMv::NotANumber;
    }
    let digits = format!("{int_digits}{frac_digits}");
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return IntlMv::Finite(ExactDec::from_digits("0", 1, negative));
    }
    // The decimal point position relative to the first significant digit.
    let leading_zeros = (digits.len() - significant.len()) as i64;
    let point = int_digits.len() as i64 - leading_zeros + exponent;
    let significant = significant.trim_end_matches('0');
    // RoundMVResult(abs(intlMV)): Rust's parse rounds a decimal string correctly (half-even).
    let rounded: f64 = if point > 400 {
        f64::INFINITY
    } else if point < -400 {
        0.0
    } else {
        format!("0.{significant}e{point}")
            .parse()
            .unwrap_or(f64::NAN)
    };
    if rounded.is_infinite() {
        return if negative {
            IntlMv::NegativeInfinity
        } else {
            IntlMv::PositiveInfinity
        };
    }
    if rounded == 0.0 {
        return IntlMv::Finite(ExactDec::from_digits("0", 1, negative));
    }
    IntlMv::Finite(ExactDec::from_digits(significant, point, negative))
}

fn exact_msd(ed: &ExactDec) -> i32 {
    let integer = ed.int.trim_start_matches('0');
    if !integer.is_empty() {
        integer.len().saturating_sub(1).min(i32::MAX as usize) as i32
    } else if let Some(first) = ed.frac.bytes().position(|byte| byte != b'0') {
        -(first.min(i32::MAX as usize - 1) as i32) - 1
    } else {
        0
    }
}

fn exact_zero(ed: &ExactDec) -> bool {
    ed.int.bytes().all(|byte| byte == b'0') && ed.frac.bytes().all(|byte| byte == b'0')
}

fn exact_f64(ed: &ExactDec) -> f64 {
    let sign = if ed.negative { "-" } else { "" };
    let text = if ed.frac.is_empty() {
        format!("{sign}{}", ed.int)
    } else {
        format!("{sign}{}.{}", ed.int, ed.frac)
    };
    text.parse().unwrap_or(if ed.negative {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    })
}

/// Multiply an exact decimal by 10^places by moving its radix point without touching its digits.
fn shift_exact(ed: &ExactDec, places: i32) -> ExactDec {
    let mut digits = format!("{}{}", ed.int, ed.frac);
    let point = ed.int.len() as i64 + places as i64;
    let (int, frac) = if point <= 0 {
        (
            "0".to_string(),
            format!("{}{}", "0".repeat((-point) as usize), digits),
        )
    } else if point as usize >= digits.len() {
        digits.push_str(&"0".repeat(point as usize - digits.len()));
        (digits, String::new())
    } else {
        let frac = digits.split_off(point as usize);
        (digits, frac)
    };
    let int = int.trim_start_matches('0');
    ExactDec {
        int: if int.is_empty() { "0" } else { int }.to_string(),
        frac,
        negative: ed.negative,
    }
}

/// Format an exact magnitude after any percent/notation scaling has moved its radix point.
fn exact_magnitude(ed: &ExactDec, o: &Gc) -> String {
    exact_magnitude_options(
        ed,
        get_num(o, "#\u{0}nf_minint").unwrap_or(1),
        get_num(o, "#\u{0}nf_minfrac").unwrap_or(0),
        get_num(o, "#\u{0}nf_maxfrac").unwrap_or(3),
        get_num(o, "#\u{0}nf_minsig"),
        get_num(o, "#\u{0}nf_maxsig"),
        get_num(o, "#\u{0}nf_roundingincrement").unwrap_or(1),
        &get_str(o, "#\u{0}nf_roundingmode"),
        &get_str(o, "#\u{0}nf_roundingtype"),
    )
}

/// ECMA-402 #sec-formatnumerictostring over an exact decimal: ToRawPrecision / ToRawFixed (or
/// both, for morePrecision/lessPrecision) and minimum-integer padding. Returns the unsigned
/// rounded magnitude; `ed.negative` selects the rounding mode's unsigned direction.
#[allow(clippy::too_many_arguments)]
pub(super) fn exact_magnitude_options(
    ed: &ExactDec,
    min_int: u32,
    min_frac: u32,
    max_frac: u32,
    min_sig: Option<u32>,
    max_sig: Option<u32>,
    increment: u32,
    mode: &str,
    rtype: &str,
) -> String {
    if matches!(rtype, "morePrecision" | "lessPrecision") {
        let significant = exact_magnitude_options(
            ed,
            min_int,
            0,
            0,
            Some(min_sig.unwrap_or(1)),
            Some(max_sig.unwrap_or(21)),
            1,
            mode,
            "significant",
        );
        let fraction = exact_magnitude_options(
            ed, min_int, min_frac, max_frac, None, None, increment, mode, "fraction",
        );
        let significant_magnitude = exact_msd(ed) - max_sig.unwrap_or(21) as i32 + 1;
        let fraction_magnitude = -(max_frac as i32);
        let fixed_is_more_precise = fraction_magnitude < significant_magnitude;
        return if (rtype == "morePrecision" && fixed_is_more_precise)
            || (rtype == "lessPrecision" && !fixed_is_more_precise)
        {
            fraction
        } else {
            significant
        };
    }

    let significant = rtype == "significant" || min_sig.is_some() || max_sig.is_some();
    let mut rounded = if significant && exact_zero(ed) {
        let minimum = min_sig.unwrap_or(1).max(1);
        if minimum == 1 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat((minimum - 1) as usize))
        }
    } else if significant {
        let maximum = max_sig.unwrap_or(21).max(1);
        let keep = maximum as i32 - 1 - exact_msd(ed);
        round_decimal(&ed.int, &ed.frac, keep, 1, mode, ed.negative)
    } else {
        round_decimal(
            &ed.int,
            &ed.frac,
            max_frac as i32,
            increment.max(1),
            mode,
            ed.negative,
        )
    };

    let minimum = if significant {
        min_sig.unwrap_or(1).max(1)
    } else {
        min_frac
    } as usize;
    if significant && !exact_zero(ed) {
        let significant_digits = |text: &str| {
            text.chars()
                .filter(char::is_ascii_digit)
                .skip_while(|character| *character == '0')
                .count()
                .max(1)
        };
        while rounded.ends_with('0')
            && rounded.contains('.')
            && significant_digits(&rounded) > minimum
        {
            rounded.pop();
        }
    } else {
        while rounded.ends_with('0')
            && rounded.contains('.')
            && rounded.split_once('.').unwrap().1.len() > minimum
        {
            rounded.pop();
        }
    }
    if rounded.ends_with('.') {
        rounded.pop();
    }

    let (mut integer, mut fraction) = rounded
        .split_once('.')
        .map(|(integer, fraction)| (integer.to_string(), fraction.to_string()))
        .unwrap_or_else(|| (rounded, String::new()));
    if significant {
        let shown = if exact_zero(ed) {
            1 + fraction.len()
        } else {
            integer
                .chars()
                .chain(fraction.chars())
                .skip_while(|character| *character == '0')
                .count()
                .max(1)
        };
        if shown < minimum {
            fraction.push_str(&"0".repeat(minimum - shown));
        }
    } else if fraction.len() < minimum {
        fraction.push_str(&"0".repeat(minimum - fraction.len()));
    }
    if integer.len() < min_int as usize {
        integer.insert_str(0, &"0".repeat(min_int as usize - integer.len()));
    }
    if fraction.is_empty() {
        integer
    } else {
        format!("{integer}.{fraction}")
    }
}

struct FormattedNumber {
    text: String,
    // Retain the selected pattern so formatToParts uses the same rounded
    // quantity as format, including placeholder-free unit forms.
    unit_pattern: Option<String>,
}

/// ECMA-402 #sec-partitionnumberpattern over an Intl mathematical value. Every finite value is
/// formatted from its exact decimal digits; only NaN and the infinities take the symbol paths.
fn assemble_number_exact(i: &mut Interp, o: &Gc, mv: &IntlMv) -> FormattedNumber {
    let style = get_str(o, "#\u{0}nf_style");
    let cldr_locale = cldr_number_locale(o);
    let number_system = get_str(o, "#\u{0}nf_nu");
    let symbols = cldr_number_symbols(o);
    let mut value = mv.approximate();
    let mut exact = mv.exact();
    if style == "percent" {
        value *= 100.0;
        exact = exact.as_ref().map(|decimal| shift_exact(decimal, 2));
    }
    // Negative zero counts as negative for sign display (so -0 formats as "-0").
    let negative = exact
        .as_ref()
        .map(|decimal| decimal.negative)
        .unwrap_or_else(|| value.is_sign_negative() && !value.is_nan());
    // scientific / engineering notation: mantissa in [1,10) or [1,1000), plus an exponent.
    let notation = get_str(o, "#\u{0}nf_notation");
    let mut exponent: Option<i32> = None;
    if notation == "scientific" || notation == "engineering" {
        if let Some(decimal) = exact.as_ref() {
            // ComputeExponent: zero (including negative zero) has exponent 0, and the scientific
            // sub-pattern still applies ("0E0").
            let mut e = if exact_zero(decimal) {
                0
            } else {
                exact_msd(decimal)
            };
            if notation == "engineering" {
                e -= e.rem_euclid(3);
            }
            exact = Some(shift_exact(decimal, -e));
            value = exact.as_ref().map(exact_f64).unwrap_or(value);
            exponent = Some(e);
        }
    }
    // ComputeExponentForMagnitude selects the CLDR compact pattern for the input magnitude; the
    // pattern's zero count determines the scaling exponent.
    let compact_input = value;
    let compact_exact_input = exact.clone();
    let mut compact: Option<(u8, crate::cldr_numbers::Compact)> = None;
    if notation == "compact" && (value.is_finite() || exact.is_some()) {
        let magnitude = exact
            .as_ref()
            .map(|decimal| exact_msd(decimal).max(0).min(u8::MAX as i32) as u8)
            .unwrap_or_else(|| {
                if value.abs() >= 1.0 {
                    value.abs().log10().floor().clamp(0.0, u8::MAX as f64) as u8
                } else {
                    0
                }
            });
        let display = get_str(o, "#\u{0}nf_compactdisplay");
        if let Some(pattern) = crate::cldr_numbers::compact(
            cldr_locale,
            &number_system,
            &display,
            magnitude,
            "other",
            false,
        ) {
            if pattern.exponent != 0 {
                value /= 10f64.powi(pattern.exponent);
                exact = exact
                    .as_ref()
                    .map(|decimal| shift_exact(decimal, -pattern.exponent));
                if let Some(decimal) = &exact {
                    value = exact_f64(decimal);
                }
            }
            compact = Some((magnitude, pattern));
        }
    }
    let (mut int_part, mut frac_part) = if value.is_nan() && exact.is_none() {
        (symbols.nan.to_string(), None)
    } else if value.is_infinite() && exact.is_none() {
        (symbols.infinity.to_string(), None)
    } else {
        // FormatNumericToString on the exact (scaled) value; compact notation's default
        // "morePrecision" of 2 significant / 0 fraction digits is in the stored digit options.
        let decimal = exact
            .as_ref()
            .expect("a finite Intl mathematical value is exact");
        let mag = exact_magnitude(decimal, o);
        match mag.split_once('.') {
            Some((a, b)) => (a.to_string(), Some(b.to_string())),
            None => (mag.clone(), None),
        }
    };
    // ComputeExponent repeats after rounding when a compact result crosses into the next
    // magnitude (for example 999,500 → 1M rather than 1000K).
    if let Some((magnitude, seed)) = compact {
        // Did rounding reach 10^(magnitude - exponent + 1)? Compare integer digit counts exactly.
        let threshold_power = magnitude as i32 - seed.exponent + 1;
        let integer_digits = int_part.trim_start_matches('0').len() as i32;
        let crossed = threshold_power >= 0 && integer_digits > threshold_power;
        if crossed {
            let next_magnitude = magnitude.saturating_add(1);
            if let Some(next) = crate::cldr_numbers::compact(
                cldr_locale,
                &number_system,
                &get_str(o, "#\u{0}nf_compactdisplay"),
                next_magnitude,
                "other",
                false,
            ) {
                exact = compact_exact_input
                    .as_ref()
                    .map(|decimal| shift_exact(decimal, -next.exponent));
                value = exact
                    .as_ref()
                    .map(exact_f64)
                    .unwrap_or_else(|| compact_input / 10f64.powi(next.exponent));
                let remade = exact_magnitude(
                    exact
                        .as_ref()
                        .expect("a finite Intl mathematical value is exact"),
                    o,
                );
                match remade.split_once('.') {
                    Some((integer, fraction)) => {
                        int_part = integer.to_string();
                        frac_part = Some(fraction.to_string());
                    }
                    None => {
                        int_part = remade;
                        frac_part = None;
                    }
                }
                compact = Some((next_magnitude, next));
            }
        }
    }
    // A value that rounds to zero (e.g. -0.0001 with the default 3 fraction digits) is "zero" for
    // the purpose of the `exceptZero`/`negative` sign rules, even though its sign bit is negative.
    let rounded_zero = value.is_finite()
        && int_part.chars().all(|c| c == '0')
        && frac_part.as_deref().unwrap_or("").chars().all(|c| c == '0');
    let grouping = o
        .borrow()
        .internal_slots()
        .get("#\u{0}nf_grouping")
        .map(|p| p.value())
        .unwrap_or(Value::lstr("auto"));
    // Grouping is suppressed in scientific/engineering notation.
    let grouped = if exponent.is_some() || (!value.is_finite() && exact.is_none()) {
        int_part.clone()
    } else {
        group_integer(
            &int_part,
            &grouping,
            symbols.group,
            (symbols.primary_group, symbols.secondary_group),
            symbols.minimum_grouping,
        )
    };
    let mut num = match frac_part {
        Some(ref f) => format!("{grouped}{}{f}", symbols.decimal),
        None => grouped,
    };
    if let Some(e) = exponent {
        let exponent_digits = if e < 0 {
            format!("{}{}", symbols.minus, e.unsigned_abs())
        } else {
            e.to_string()
        };
        num = format!("{num}{}{exponent_digits}", symbols.exponential);
        let scientific = crate::cldr_numbers::pattern(cldr_locale, &number_system, "scientific");
        num = format!(
            "{}{num}{}",
            scientific.positive_prefix, scientific.positive_suffix
        );
    }
    if let Some((magnitude, seed)) = compact {
        let rounded = match &frac_part {
            Some(fraction) => format!("{int_part}.{fraction}"),
            None => int_part.clone(),
        };
        let compact_exponent = seed.exponent.max(0) as u32;
        let category = crate::cldr_plurals::select_cardinal(o_lang(o), &rounded, compact_exponent);
        let exact_one = int_part.trim_start_matches('0') == "1"
            && frac_part
                .as_deref()
                .unwrap_or("")
                .bytes()
                .all(|digit| digit == b'0');
        let selected = crate::cldr_numbers::compact(
            cldr_locale,
            &number_system,
            &get_str(o, "#\u{0}nf_compactdisplay"),
            magnitude,
            category,
            exact_one,
        )
        .unwrap_or(seed);
        num = if selected.has_number {
            format!("{}{num}{}", selected.prefix, selected.suffix)
        } else {
            format!("{}{}", selected.prefix, selected.suffix)
        };
    }

    // Sign display. `auto`/`always` key off the sign bit (so -0 and values rounding to zero still
    // show "-0"); `exceptZero`/`negative` suppress the sign when the displayed value is zero or NaN.
    let sign_display = get_str(o, "#\u{0}nf_signdisplay");
    let zeroish = rounded_zero || value.is_nan();
    let sign = match sign_display.as_str() {
        "never" => 0,
        "always" => {
            if negative {
                -1
            } else {
                1
            }
        }
        "exceptZero" => {
            if zeroish {
                0
            } else if negative {
                -1
            } else {
                1
            }
        }
        "negative" => {
            if negative && !zeroish {
                -1
            } else {
                0
            }
        }
        _ => {
            if negative {
                -1
            } else {
                0
            }
        }
    };

    let rounded = match &frac_part {
        Some(fraction) => format!("{int_part}.{fraction}"),
        None => int_part.clone(),
    };
    let compact_exponent = compact
        .map(|(_, pattern)| pattern.exponent.max(0) as u32)
        .unwrap_or(0);
    let quantity_exponent = exponent.unwrap_or(compact_exponent as i32);
    let mut unit_pattern = None;

    // GetNumberFormatPattern selects the localized sign/style affixes around the notation
    // subpattern. Currency names use CLDR's plural currency unit pattern instead of a symbol slot.
    match style.as_str() {
        "percent" => {
            num = apply_number_pattern(
                crate::cldr_numbers::pattern(cldr_locale, &number_system, "percent"),
                sign,
                &num,
                &symbols,
                "",
            );
        }
        "currency" => {
            let code = get_str(o, "#\u{0}nf_currency");
            let display = get_str(o, "#\u{0}nf_currencydisplay");
            let category =
                quantity_plural_category(o_lang(o), &rounded, quantity_exponent, compact_exponent);
            let currency = crate::cldr_numbers::currency(cldr_locale, &code, category);
            if display == "name" {
                let signed = apply_number_pattern(
                    crate::cldr_numbers::pattern(cldr_locale, &number_system, "decimal"),
                    sign,
                    &num,
                    &symbols,
                    "",
                );
                let name = currency.map(|value| value.name).unwrap_or(&code);
                num = crate::cldr_numbers::currency_unit(cldr_locale, category)
                    .replace("{0}", &signed)
                    .replace("{1}", name);
            } else {
                let text = match display.as_str() {
                    "code" => code.as_str(),
                    "narrowSymbol" => currency.map(|value| value.narrow).unwrap_or(&code),
                    _ => currency.map(|value| value.symbol).unwrap_or(&code),
                };
                let accounting = get_str(o, "#\u{0}nf_currencysign") == "accounting";
                let base_kind = if accounting { "accounting" } else { "currency" };
                let base_pattern =
                    crate::cldr_numbers::pattern(cldr_locale, &number_system, base_kind);
                // UTS #35's alphaNextToNumber variant applies only when the edge of the
                // substituted currency string adjacent to the number is alphabetic. A symbol
                // such as `US$` contains letters but ends in `$`, so a prefix pattern must not
                // insert the alphabetic spacing used by a currency code such as `USD`.
                let alpha = if base_pattern.positive_prefix.contains("{currency}") {
                    text.chars().next_back().is_some_and(char::is_alphabetic)
                } else if base_pattern.positive_suffix.contains("{currency}") {
                    text.chars().next().is_some_and(char::is_alphabetic)
                } else {
                    false
                };
                let kind = match (accounting, alpha) {
                    (false, false) => "currency",
                    (false, true) => "currencyAlpha",
                    (true, false) => "accounting",
                    (true, true) => "accountingAlpha",
                };
                // A currency's own pattern (UTS #35 Part 3 Currencies) replaces the locale's
                // standard currency format.
                let own_pattern = (!accounting)
                    .then(|| crate::cldr_numbers::currency_format(cldr_locale, &code))
                    .flatten()
                    .and_then(|format| format.pattern);
                num = apply_number_pattern(
                    own_pattern.unwrap_or_else(|| {
                        crate::cldr_numbers::pattern(cldr_locale, &number_system, kind)
                    }),
                    sign,
                    &num,
                    &symbols,
                    text,
                );
            }
        }
        "unit" => {
            let signed = apply_number_pattern(
                crate::cldr_numbers::pattern(cldr_locale, &number_system, "decimal"),
                sign,
                &num,
                &symbols,
                "",
            );
            let category =
                quantity_plural_category(o_lang(o), &rounded, quantity_exponent, compact_exponent);
            unit_pattern = unit_pattern_for_category(o, category);
            // Every sanctioned unit has CLDR patterns, so the bare number is never reached.
            num = match unit_pattern.as_deref() {
                Some(p) => p.replace("{0}", &signed),
                None => signed,
            };
        }
        _ => {
            num = apply_number_pattern(
                crate::cldr_numbers::pattern(cldr_locale, &number_system, "decimal"),
                sign,
                &num,
                &symbols,
                "",
            );
        }
    }
    let _ = i;
    FormattedNumber {
        text: num,
        unit_pattern,
    }
}

/// ECMA-402 #sec-partitionnumberpattern (snapshot b1c961988b9a) uses the
/// rounded result for locale-dependent affixes. UTS #35 Part 3 #Operands
/// (snapshot 1c6bc010ee9a) counts digits of the represented quantity, so
/// 1E4 has i=10000 and 1.20050c3 has i=1200, v=2, f=50. Move the radix
/// point exactly, retaining visible zeros and BigInt/decimal-string digits.
/// This is distinct from choosing the compact power word itself, which
/// follows UTS #35 #Compact_Number_Formats using the scaled mantissa.
fn quantity_plural_category(
    lang: &str,
    rounded: &str,
    exponent: i32,
    compact_exponent: u32,
) -> &'static str {
    let Some(decimal) = ExactDec::parse(rounded) else {
        return "other"; // NaN and infinities do not have plural operands.
    };
    let quantity = shift_exact(&decimal, exponent);
    let digits = if quantity.frac.is_empty() {
        quantity.int
    } else {
        format!("{}.{}", quantity.int, quantity.frac)
    };
    crate::cldr_plurals::select_cardinal(lang, &digits, compact_exponent)
}

fn apply_number_pattern(
    pattern: crate::cldr_numbers::Pattern,
    sign: i8,
    number: &str,
    symbols: &crate::cldr_numbers::Symbols,
    currency: &str,
) -> String {
    let (mut prefix, suffix) = if sign < 0 {
        (pattern.negative_prefix.to_string(), pattern.negative_suffix)
    } else {
        (pattern.positive_prefix.to_string(), pattern.positive_suffix)
    };
    if sign > 0 {
        // CLDR derives the plus pattern from the positive pattern. Keep leading bidi controls ahead
        // of the localized plus sign so Arabic and other RTL affixes remain well-ordered.
        let byte = prefix
            .char_indices()
            .find(|(_, character)| !matches!(character, '\u{061c}' | '\u{200e}' | '\u{200f}'))
            .map(|(index, _)| index)
            .unwrap_or(prefix.len());
        prefix.insert_str(byte, "{plusSign}");
    }
    format!(
        "{}{number}{}",
        render_number_affix(&prefix, symbols, currency),
        render_number_affix(suffix, symbols, currency)
    )
}

fn render_number_affix(
    affix: &str,
    symbols: &crate::cldr_numbers::Symbols,
    currency: &str,
) -> String {
    affix
        .replace("{currency}", currency)
        .replace("{percentSign}", symbols.percent)
        .replace("{minusSign}", symbols.minus)
        .replace("{plusSign}", symbols.plus)
}

/// Replace the ASCII digits of a formatted number with the glyphs of numbering system `nu` (a no-op
/// for `latn` or an unknown system).
pub(crate) fn xlate_digits(s: &str, nu: &str) -> String {
    if nu == "latn" {
        return s.to_string();
    }
    match crate::numbering::NUMBERING.iter().find(|(id, _)| *id == nu) {
        Some((_, glyphs)) => s
            .chars()
            .map(|c| {
                if c.is_ascii_digit() {
                    glyphs[c as usize - '0' as usize]
                } else {
                    c
                }
            })
            .collect(),
        None => s.to_string(),
    }
}

fn format_number(i: &mut Interp, this: &Value, x: &Value) -> Result<Value, Value> {
    let o = instance(i, this)?;
    let n = to_intl_mv(i, x)?;
    let s = assemble_number_exact(i, &o, &n).text;
    Ok(Value::lstr(xlate_digits(&s, &get_str(&o, "#\u{0}nf_nu"))))
}

/// The trailing-affix classification for this formatter's parts (compact suffix vs plain).
fn suffix_type_of(o: &Gc) -> String {
    if get_str(o, "#\u{0}nf_style") == "unit" {
        "unit".to_string()
    } else if get_str(o, "#\u{0}nf_style") == "currency" {
        "currency".to_string()
    } else if get_str(o, "#\u{0}nf_notation") == "compact" {
        "compact".to_string()
    } else {
        "literal".to_string()
    }
}

fn format_to_parts(i: &mut Interp, this: &Value, x: &Value) -> Result<Value, Value> {
    let o = instance(i, this)?;
    let n = to_intl_mv(i, x)?;
    let FormattedNumber {
        text: whole,
        unit_pattern,
    } = assemble_number_exact(i, &o, &n);
    let nu = get_str(&o, "#\u{0}nf_nu");
    let symbols = cldr_number_symbols(&o);
    // Unit style: rebuild from the CLDR pattern so a unit prefix/suffix (e.g. ko "시속 {0}킬로미터")
    // is tagged as unit/literal around the number's own parts.
    let parts = if get_str(&o, "#\u{0}nf_style") == "unit" {
        if let Some(pat) = unit_pattern {
            if !pat.contains("{0}") {
                vec![("unit", whole)]
            } else {
                let (pre, post) = pat.split_once("{0}").unwrap();
                let num_only = whole
                    .strip_prefix(pre)
                    .and_then(|s| s.strip_suffix(post))
                    .unwrap_or(&whole);
                let mut p = unit_affix_parts(pre, true);
                p.extend(decompose_parts(num_only, "literal", &symbols));
                p.extend(unit_affix_parts(post, false));
                p
            }
        } else {
            decompose_parts(&whole, &suffix_type_of(&o), &symbols)
        }
    } else {
        decompose_parts(&whole, &suffix_type_of(&o), &symbols)
    };
    let arr: Vec<Value> = parts
        .into_iter()
        .map(|(t, v)| {
            let ob = i.new_object();
            set_data(&ob, "type", Value::lstr(t));
            // Localize the digits of numeric parts to the numbering system.
            let v = if matches!(t, "integer" | "fraction" | "exponentInteger") {
                xlate_digits(&v, &nu)
            } else {
                v
            };
            set_data(&ob, "value", Value::lstr(v));
            Value::Obj(ob)
        })
        .collect();
    Ok(i.make_array(arr))
}

/// Break an assembled number string into typed parts (minusSign/plusSign, currency/percentSign/
/// literal affixes, grouped integer, decimal, fraction, and the exponent group). `suffix_type`
/// classifies the trailing affix ("compact" for compact notation, else percent/literal by content).
/// The formatter locale's primary language subtag.
/// The shipped locale whose plural rules apply (rules are per language, with locale-specific
/// sets such as pt-PT along the parent chain).
fn o_lang(o: &Gc) -> &'static str {
    cldr_number_locale(o)
}

/// The CLDR unit-display pattern ("{0} km/h") for the already-selected plural
/// category, along the formatter locale's parent chain.
fn unit_pattern_for_category(o: &Gc, category: &str) -> Option<String> {
    let unit = get_str(o, "#\u{0}nf_unit");
    let disp = get_str(o, "#\u{0}nf_unitdisplay");
    let style = if disp.is_empty() {
        "short"
    } else {
        disp.as_str()
    };
    let locale = cldr_number_locale(o);
    // UTS #35 Part 2: a missing narrow pattern inherits short, a missing short inherits long;
    // a plural category without a pattern uses `other`. Every shipped locale has CLDR unit
    // data along its parent chain; English is the default locale's data for any other.
    let styles: &[&str] = match style {
        "narrow" => &["narrow", "short", "long"],
        "short" => &["short", "long"],
        _ => &["long"],
    };
    [locale, "en"].into_iter().find_map(|loc| {
        styles
            .iter()
            .find_map(|style| {
                crate::cldr_units::unit_pattern(loc, &unit, style, category)
                    .or_else(|| crate::cldr_units::unit_pattern(loc, &unit, style, "other"))
            })
            .map(str::to_string)
            .or_else(|| compound_unit_pattern(loc, &unit, style, category))
    })
}

/// UTS #35 Part 2 Compound Units, `pattern(unitId, …)`, for an ECMA-402 "X-per-Y" unit without a
/// precomposed CLDR pattern: the numerator's pattern for the plural category (the per structure
/// keeps the compound's category for its first value) with its placeholder extracted, then the
/// denominator's precomposed `perUnitPattern` when it has one, else the `per` compound pattern
/// with the denominator's singular (`one`) unit name; finally the placeholder is reattached.
fn compound_unit_pattern(loc: &str, unit: &str, style: &str, category: &str) -> Option<String> {
    let (numerator, denominator) = unit.split_once("-per-")?;
    // Lateral inheritance: a missing narrow pattern uses short, a missing short uses long.
    let styles: &[&str] = match style {
        "narrow" => &["narrow", "short", "long"],
        "short" => &["short", "long"],
        _ => &["long"],
    };
    let simple = |name: &str, count: &str| {
        styles.iter().find_map(|style| {
            crate::cldr_units::unit_pattern(loc, name, style, count)
                .or_else(|| crate::cldr_units::unit_pattern(loc, name, style, "other"))
        })
    };
    let per = |name: &str| {
        styles
            .iter()
            .find_map(|style| crate::cldr_units::per_pattern(loc, name, style))
    };
    let (core, placeholder, at_start) = extract_placeholder(simple(numerator, category)?)?;
    let combined = match per(denominator) {
        Some(per_unit) => per_unit.replacen("{0}", &core, 1),
        None => {
            let (denominator_core, _, _) = extract_placeholder(simple(denominator, "one")?)?;
            per("per")?
                .replacen("{0}", &core, 1)
                .replacen("{1}", &denominator_core, 1)
        }
    };
    Some(if at_start {
        format!("{placeholder}{combined}")
    } else {
        format!("{combined}{placeholder}")
    })
}

/// UTS #35 Extract(corePattern, coreUnit, placeholder, placeholderPosition): split a unit pattern
/// into its unit text and its `{0}` placeholder with the adjoining whitespace. A placeholder in
/// the middle of the pattern cannot be composed.
fn extract_placeholder(pattern: &str) -> Option<(String, String, bool)> {
    let at = pattern.find("{0}")?;
    let (before, after) = (&pattern[..at], &pattern[at + 3..]);
    if before.trim().is_empty() {
        let unit = after.trim_start();
        let placeholder = format!("{before}{{0}}{}", &after[..after.len() - unit.len()]);
        Some((unit.to_string(), placeholder, true))
    } else if after.trim().is_empty() {
        let unit = before.trim_end();
        let placeholder = format!("{}{{0}}{after}", &before[unit.len()..]);
        Some((unit.to_string(), placeholder, false))
    } else {
        None
    }
}

/// Tokenize a unit pattern's pre/post affix into (unit, literal) parts: the unit name is the
/// non-space run, adjoining spaces are literals (a `pre` affix ends with the separator, a `post`
/// affix begins with it).
fn unit_affix_parts(text: &str, is_pre: bool) -> Vec<(&'static str, String)> {
    if text.is_empty() {
        return Vec::new();
    }
    let sp = |c: char| c == ' ' || c == '\u{a0}';
    let mut v = Vec::new();
    if is_pre {
        let unit = text.trim_end_matches(sp);
        if !unit.is_empty() {
            v.push(("unit", unit.to_string()));
        }
        let tail = &text[unit.len()..];
        if !tail.is_empty() {
            v.push(("literal", tail.to_string()));
        }
    } else {
        let unit = text.trim_start_matches(sp);
        let lead = &text[..text.len() - unit.len()];
        if !lead.is_empty() {
            v.push(("literal", lead.to_string()));
        }
        if !unit.is_empty() {
            v.push(("unit", unit.to_string()));
        }
    }
    v
}

fn decompose_parts(
    s: &str,
    suffix_type: &str,
    symbols: &crate::cldr_numbers::Symbols,
) -> Vec<(&'static str, String)> {
    let mut parts: Vec<(&'static str, String)> = Vec::new();
    let numeric_start = s
        .char_indices()
        .find(|(index, character)| {
            character.is_ascii_digit()
                || s[*index..].starts_with(symbols.infinity)
                || s[*index..].starts_with(symbols.nan)
        })
        .map(|(index, _)| index);
    let Some(mut idx) = numeric_start else {
        // CLDR has placeholder-free compact/unit forms such as French `mille` and Arabic `متر`.
        push_affix_parts(&mut parts, s, suffix_type, symbols);
        return parts;
    };
    push_affix_parts(&mut parts, &s[..idx], suffix_type, symbols);

    // Non-finite body: emit a single infinity/nan part, then fall through to any trailing affix.
    if s[idx..].starts_with(symbols.infinity) {
        push_part(&mut parts, "infinity", symbols.infinity);
        idx += symbols.infinity.len();
    } else if s[idx..].starts_with(symbols.nan) {
        push_part(&mut parts, "nan", symbols.nan);
        idx += symbols.nan.len();
    }

    // Integer digits and grouping separators. Generated CLDR symbols are strings rather than
    // assumed one-byte punctuation, so token matching stays correct for every numbering system.
    let mut integer = String::new();
    while idx < s.len() {
        let rest = &s[idx..];
        if let Some(character) = rest.chars().next().filter(char::is_ascii_digit) {
            integer.push(character);
            idx += character.len_utf8();
        } else if rest.starts_with(symbols.group) {
            if !integer.is_empty() {
                push_part(&mut parts, "integer", &integer);
                integer.clear();
            }
            push_part(&mut parts, "group", symbols.group);
            idx += symbols.group.len();
        } else {
            break;
        }
    }
    if !integer.is_empty() {
        push_part(&mut parts, "integer", &integer);
    }
    // Decimal + fraction.
    if idx < s.len() && s[idx..].starts_with(symbols.decimal) {
        push_part(&mut parts, "decimal", symbols.decimal);
        idx += symbols.decimal.len();
        let start = idx;
        while idx < s.len() {
            let Some(character) = s[idx..].chars().next() else {
                break;
            };
            if !character.is_ascii_digit() {
                break;
            }
            idx += character.len_utf8();
        }
        push_part(&mut parts, "fraction", &s[start..idx]);
    }

    // Scientific/engineering exponent, followed by any style affix (currency/unit/percent).
    if idx < s.len() && s[idx..].starts_with(symbols.exponential) {
        push_part(&mut parts, "exponentSeparator", symbols.exponential);
        idx += symbols.exponential.len();
        if s[idx..].starts_with(symbols.minus) {
            push_symbol_parts(&mut parts, symbols.minus, "exponentMinusSign");
            idx += symbols.minus.len();
        } else if s[idx..].starts_with(symbols.plus) {
            push_symbol_parts(&mut parts, symbols.plus, "exponentPlusSign");
            idx += symbols.plus.len();
        }
        let start = idx;
        while idx < s.len() {
            let Some(character) = s[idx..].chars().next() else {
                break;
            };
            if !character.is_ascii_digit() {
                break;
            }
            idx += character.len_utf8();
        }
        push_part(&mut parts, "exponentInteger", &s[start..idx]);
    }
    push_affix_parts(&mut parts, &s[idx..], suffix_type, symbols);
    parts
}

fn is_affix_literal(character: char) -> bool {
    character.is_whitespace()
        || matches!(character, '(' | ')' | '\u{061c}' | '\u{200e}' | '\u{200f}')
}

fn push_part(parts: &mut Vec<(&'static str, String)>, kind: &'static str, value: &str) {
    if value.is_empty() {
        return;
    }
    if let Some((last_kind, last_value)) = parts.last_mut() {
        if *last_kind == kind && kind == "literal" {
            last_value.push_str(value);
            return;
        }
    }
    parts.push((kind, value.to_string()));
}

/// Split directional controls out of a localized sign/percent symbol. ECMA-402 emits the visible
/// field as the semantic part while the controls remain literal parts preserving bidi ordering.
fn push_symbol_parts(parts: &mut Vec<(&'static str, String)>, symbol: &str, kind: &'static str) {
    let mut run = String::new();
    let mut run_literal = None;
    for character in symbol.chars() {
        let literal = matches!(character, '\u{061c}' | '\u{200e}' | '\u{200f}');
        if run_literal.is_some_and(|previous| previous != literal) {
            push_part(
                parts,
                if run_literal.unwrap() {
                    "literal"
                } else {
                    kind
                },
                &run,
            );
            run.clear();
        }
        run_literal = Some(literal);
        run.push(character);
    }
    if !run.is_empty() {
        push_part(
            parts,
            if run_literal.unwrap() {
                "literal"
            } else {
                kind
            },
            &run,
        );
    }
}

/// Tokenize a prefix/suffix from a CLDR pattern. Spaces, parentheses and bidi controls are
/// literals; the remaining field is currency/unit/compact data. Localized signs and percent signs
/// are recognized even when their CLDR symbol contains directional controls.
fn push_affix_parts(
    parts: &mut Vec<(&'static str, String)>,
    mut text: &str,
    semantic: &str,
    symbols: &crate::cldr_numbers::Symbols,
) {
    while !text.is_empty() {
        if text.starts_with(symbols.minus) {
            push_symbol_parts(parts, symbols.minus, "minusSign");
            text = &text[symbols.minus.len()..];
            continue;
        }
        if text.starts_with(symbols.plus) {
            push_symbol_parts(parts, symbols.plus, "plusSign");
            text = &text[symbols.plus.len()..];
            continue;
        }
        if text.starts_with(symbols.percent) {
            push_symbol_parts(parts, symbols.percent, "percentSign");
            text = &text[symbols.percent.len()..];
            continue;
        }
        let first = text.chars().next().unwrap();
        if is_affix_literal(first) {
            let end = text
                .char_indices()
                .find(|(_, character)| !is_affix_literal(*character))
                .map(|(index, _)| index)
                .unwrap_or(text.len());
            push_part(parts, "literal", &text[..end]);
            text = &text[end..];
            continue;
        }
        let end = text
            .char_indices()
            .skip(1)
            .find(|(index, character)| {
                is_affix_literal(*character)
                    || text[*index..].starts_with(symbols.minus)
                    || text[*index..].starts_with(symbols.plus)
                    || text[*index..].starts_with(symbols.percent)
            })
            .map(|(index, _)| index)
            .unwrap_or(text.len());
        let kind = match semantic {
            "currency" => "currency",
            "unit" => "unit",
            "compact" => "compact",
            _ => "literal",
        };
        push_part(parts, kind, &text[..end]);
        text = &text[end..];
    }
}

fn resolved_options(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = instance_unwrap(i, &this)?;
    let res = i.new_object();
    let put = |i: &mut Interp, res: &Gc, k: &str, slot: &str| {
        if let Some(v) = o.borrow().internal_slots().get(slot).map(|p| p.value()) {
            set_data(res, k, v);
        }
        let _ = i;
    };
    put(i, &res, "locale", "#\u{0}nf_locale");
    put(i, &res, "numberingSystem", "#\u{0}nf_nu");
    put(i, &res, "style", "#\u{0}nf_style");
    put(i, &res, "currency", "#\u{0}nf_currency");
    put(i, &res, "currencyDisplay", "#\u{0}nf_currencydisplay");
    put(i, &res, "currencySign", "#\u{0}nf_currencysign");
    put(i, &res, "unit", "#\u{0}nf_unit");
    put(i, &res, "unitDisplay", "#\u{0}nf_unitdisplay");
    put(i, &res, "minimumIntegerDigits", "#\u{0}nf_minint");
    {
        // Spec key order: fraction digits, then significant digits. Under a *Precision rounding
        // type (e.g. the compact-notation default) BOTH pairs are present.
        let rtype = get_str(&o, "#\u{0}nf_roundingtype");
        let has_sig = o.borrow().internal_slots().contains("#\u{0}nf_minsig");
        if !has_sig || rtype.contains("Precision") {
            put(i, &res, "minimumFractionDigits", "#\u{0}nf_minfrac");
            put(i, &res, "maximumFractionDigits", "#\u{0}nf_maxfrac");
        }
        if has_sig {
            put(i, &res, "minimumSignificantDigits", "#\u{0}nf_minsig");
            put(i, &res, "maximumSignificantDigits", "#\u{0}nf_maxsig");
        }
    }
    // useGrouping/notation/compactDisplay/signDisplay precede the rounding options in key order.
    set_data(
        &res,
        "useGrouping",
        o.borrow()
            .internal_slots()
            .get("#\u{0}nf_grouping")
            .map(|p| p.value())
            .unwrap_or(Value::lstr("auto")),
    );
    put(i, &res, "notation", "#\u{0}nf_notation");
    // compactDisplay only appears when notation is compact.
    if matches!((o.borrow().internal_slots().get("#\u{0}nf_notation").map(|p| p.value())).map(|__value| (__value).view()), Some(ValueView::Str(s)) if &*s == "compact")
    {
        put(i, &res, "compactDisplay", "#\u{0}nf_compactdisplay");
    }
    put(i, &res, "signDisplay", "#\u{0}nf_signdisplay");
    put(i, &res, "roundingIncrement", "#\u{0}nf_roundingincrement");
    put(i, &res, "roundingMode", "#\u{0}nf_roundingmode");
    put(i, &res, "roundingPriority", "#\u{0}nf_roundingpriority");
    put(i, &res, "trailingZeroDisplay", "#\u{0}nf_trailingzero");
    Ok(Value::Obj(res))
}

#[cfg(test)]
mod tests {
    use crate::{bytecode::Tier, Completion, Engine};

    fn check_all_tiers(source: &str) {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            match engine.eval(source, false).expect("valid test script") {
                Completion::Value(value) => assert_eq!(value, "ok"),
                Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
            }
        }
    }

    #[test]
    fn numberformat_plural_quantities_preserve_notation_exponents() {
        // ECMA-402 PartitionNumberPattern and UTS #35 Part 3, Operands:
        // the quantity 1E4 has i=10000, not i=1. Currency/unit names use
        // this quantity; the separate compact power word uses its mantissa.
        check_all_tiers(
            r#"
            function check(actual, expected) {
                if (actual !== expected) throw Error(JSON.stringify({actual, expected}));
            }
            for (const [notation, value, expected] of [
                ['scientific', 10000, '1E4 bits'],
                ['scientific', -10000, '-1E4 bits'],
                ['scientific', .1, '1E-1 bits'],
                ['scientific', 1, '1E0 bit'],
                ['engineering', 1000, '1E3 bits'],
                ['engineering', .1, '100E-3 bits'],
                ['compact', 1000, '1K bits'],
                ['compact', 1000000, '1M bits']
            ]) {
                const format = new Intl.NumberFormat('en', {
                    notation, style: 'unit', unit: 'bit', unitDisplay: 'long'
                });
                check(format.format(value), expected);
                check(format.formatToParts(value).map(p => p.value).join(''), expected);
                check(format.formatToParts(value).filter(p => p.type === 'unit').length, 1);
            }
            for (const notation of ['scientific', 'engineering', 'compact']) {
                const format = new Intl.NumberFormat('en', {
                    notation, style: 'currency', currency: 'USD', currencyDisplay: 'name',
                    minimumFractionDigits: 0, maximumFractionDigits: 0
                });
                const prefix = notation === 'compact' ? '1K' : '1E3';
                check(format.format(1000), prefix + ' US dollars');
            }
            check(new Intl.NumberFormat('ru', {notation: 'compact', compactDisplay: 'long'}).format(2000), '2 тысячи');
            'ok';
        "#,
        );
    }

    #[test]
    fn numberformat_unit_plurals_share_rounded_digits_with_parts() {
        // ECMA-402 #sec-partitionnumberpattern; UTS #35 #Unit_Elements
        // and #Plural_Operand_Meanings: visible zeros and exact integer
        // digits participate in plural selection after rounding.
        check_all_tiers(
            r#"
            function check(actual, expected) {
                if (actual !== expected) throw Error(JSON.stringify({actual, expected}));
            }
            for (const [value, options, expected] of [
                [1.2, {maximumFractionDigits: 0}, '1 bit'],
                [.99, {maximumFractionDigits: 0}, '1 bit'],
                [1.99, {maximumFractionDigits: 0}, '2 bits'],
                [1, {minimumFractionDigits: 1}, '1.0 bits'],
                [-1, {}, '-1 bit'],
                [NaN, {}, 'NaN bits'],
                [Infinity, {}, '∞ bits']
            ]) {
                const format = new Intl.NumberFormat('en', {
                    style: 'unit', unit: 'bit', unitDisplay: 'long', ...options
                });
                check(format.format(value), expected);
                const parts = format.formatToParts(value);
                check(parts.map(p => p.value).join(''), expected);
                check(parts.filter(p => p.type === 'unit').length, 1);
            }
            const ru = new Intl.NumberFormat('ru', {style: 'unit', unit: 'meter', unitDisplay: 'long', useGrouping: false});
            const large = 100000000000000000000000000000000000000000000000000000000000000021n;
            check(ru.format(large), large.toString() + ' метр');
            check(ru.formatToParts(large).map(p => p.value).join(''), ru.format(large));
            const ar = new Intl.NumberFormat('ar', {style: 'unit', unit: 'meter', unitDisplay: 'long', maximumFractionDigits: 0});
            check(ar.format(.99), 'متر');
            check(JSON.stringify(ar.formatToParts(.99)), '[{"type":"unit","value":"متر"}]');
            const sl = new Intl.NumberFormat('sl', {style: 'unit', unit: 'meter', unitDisplay: 'long', minimumFractionDigits: 2});
            check(sl.format(1), '1,00 metri');
            check(sl.formatToParts(1).map(p => p.value).join(''), '1,00 metri');
            'ok';
        "#,
        );
    }
}
