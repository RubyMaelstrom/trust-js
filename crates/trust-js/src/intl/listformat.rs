//! `Intl.ListFormat`.

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
    resolve_locale,
};
use super::{
    ab, arg, canonicalize_locale_list, def_getter, get_options_object as coerce_options,
    make_service,
};
use crate::interpreter::Interp;
use crate::value::ValueView;
use crate::value::{set_data, set_internal_slot, Value};

pub(crate) fn install(it: &mut Interp, ns: &crate::value::Gc) {
    let (ctor, proto) = make_service(it, ns, "ListFormat", 0, construct);
    install_supported_locales(it, &ctor);

    it.def_method(&proto, "format", 1, |i, this, a| {
        format(i, &this, &arg(a, 0), false)
    });
    it.def_method(&proto, "formatToParts", 1, |i, this, a| {
        format(i, &this, &arg(a, 0), true)
    });
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
    let _ = def_getter;
}

fn construct(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    if !i.constructing {
        return Err(i.make_error("TypeError", "Intl.ListFormat requires 'new'"));
    }
    let requested = canonicalize_locale_list(i, &arg(a, 0))?;
    let options = coerce_options(i, &arg(a, 1))?;
    read_locale_matcher(i, &options)?;
    let kind = get_option(
        i,
        &options,
        "type",
        &["conjunction", "disjunction", "unit"],
        Some("conjunction"),
    )?
    .unwrap();
    let style = get_option(
        i,
        &options,
        "style",
        &["long", "short", "narrow"],
        Some("long"),
    )?
    .unwrap();
    let resolved = resolve_locale(i, &requested, &[]);

    let obj = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.ListFormat")? {
        obj.borrow_mut().proto = Some(proto);
    }
    set_internal_slot(&obj, "#\u{0}lf", Value::Bool(true));
    set_internal_slot(&obj, "#\u{0}lf_locale", Value::lstr(resolved.locale));
    set_internal_slot(&obj, "#\u{0}lf_type", Value::lstr(kind));
    set_internal_slot(&obj, "#\u{0}lf_style", Value::lstr(style));
    Ok(Value::Obj(obj))
}

/// StringListFromIterable: iterate the argument, requiring every element to be a String.
fn string_list(i: &mut Interp, list: &Value) -> Result<Vec<String>, Value> {
    if matches!(&(list).view(), ValueView::Undefined) {
        return Ok(Vec::new());
    }
    // Iterate LAZILY so a non-string element throws immediately (stopping the iteration), rather than
    // eagerly draining the iterable first.
    let (iter, next) = ab(i.get_iterator(list))?;
    let mut out = Vec::new();
    // The formatted list contains every element, so their total bounds its length.
    let mut total = 0usize;
    loop {
        match ab(i.iterator_step(&iter, &next))?.map(|value| value.view()) {
            None => break,
            Some(ValueView::Str(s)) => {
                let text = s.text();
                total = total.saturating_add(text.len());
                if let Err(err) = crate::builtins::check_string_length(i, total) {
                    i.iterator_close(&iter);
                    return Err(err);
                }
                out.push(text.into_owned())
            }
            Some(_) => {
                let err =
                    i.make_error("TypeError", "Intl.ListFormat list elements must be strings");
                i.iterator_close(&iter);
                return Err(err);
            }
        }
    }
    Ok(out)
}

type Segment = (bool, String);

/// UTS #35's context-sensitive Spanish list alternations. ECMA-402 explicitly permits template
/// selection based on the substituted values; keeping it here avoids changing or duplicating the
/// generated CLDR data.
fn contextual_pattern<'a>(lang: &str, pattern: &'a str, second: &str) -> std::borrow::Cow<'a, str> {
    if lang != "es" {
        return std::borrow::Cow::Borrowed(pattern);
    }
    let lower = second.to_lowercase();
    let use_e = pattern.contains(" y ")
        && (lower.starts_with('i')
            || (lower.starts_with("hi") && !lower.starts_with("hie") && !lower.starts_with("hia")));
    if use_e {
        return std::borrow::Cow::Owned(pattern.replacen(" y ", " e ", 1));
    }
    let numeric_eleven = lower.starts_with("11")
        && lower
            .chars()
            .nth(2)
            .is_none_or(|character| !character.is_ascii_digit());
    let use_u = pattern.contains(" o ")
        && (lower.starts_with('o')
            || lower.starts_with("ho")
            || lower.starts_with('8')
            || numeric_eleven);
    if use_u {
        return std::borrow::Cow::Owned(pattern.replacen(" o ", " u ", 1));
    }
    std::borrow::Cow::Borrowed(pattern)
}

/// DeconstructPattern for the two list placeables. Unlike the old separator extraction this
/// preserves the prefix permitted on pair/start patterns and suffix permitted on pair/end patterns.
fn deconstruct_pattern(pattern: &str, first: Segment, second: Vec<Segment>) -> Vec<Segment> {
    fn push_literal(output: &mut Vec<Segment>, text: &str) {
        if !text.is_empty() {
            output.push((false, text.to_string()));
        }
    }

    let first_marker = pattern.find("{0}").expect("generated CLDR pattern has {0}");
    let second_marker = pattern.find("{1}").expect("generated CLDR pattern has {1}");
    debug_assert!(first_marker < second_marker);
    let mut output = Vec::with_capacity(second.len() + 4);
    push_literal(&mut output, &pattern[..first_marker]);
    output.push(first);
    push_literal(&mut output, &pattern[first_marker + 3..second_marker]);
    output.extend(second);
    push_literal(&mut output, &pattern[second_marker + 3..]);
    output
}

/// ECMA-402 CreatePartsFromList / UTS #35 List Patterns, evaluated back-to-front so nested
/// substitutions retain their element/literal records without constructing intermediate strings.
fn assemble_segments(parts: &[String], patterns: [&'static str; 4], lang: &str) -> Vec<Segment> {
    match parts {
        [] => Vec::new(),
        [only] => vec![(true, only.clone())],
        [first, second] => {
            let pattern = contextual_pattern(lang, patterns[0], second);
            deconstruct_pattern(
                &pattern,
                (true, first.clone()),
                vec![(true, second.clone())],
            )
        }
        _ => {
            let mut output = vec![(true, parts.last().unwrap().clone())];
            for index in (0..parts.len() - 1).rev() {
                let pattern = if index == 0 {
                    patterns[1]
                } else if index == parts.len() - 2 {
                    patterns[3]
                } else {
                    patterns[2]
                };
                let pattern = if index == parts.len() - 2 {
                    contextual_pattern(lang, pattern, parts.last().unwrap())
                } else {
                    std::borrow::Cow::Borrowed(pattern)
                };
                output = deconstruct_pattern(&pattern, (true, parts[index].clone()), output);
            }
            output
        }
    }
}

fn format(i: &mut Interp, this: &Value, list: &Value, to_parts: bool) -> Result<Value, Value> {
    let o = brand_slot(i, this, "#\u{0}lf")?;
    let get = |k: &str| match (o.borrow().internal_slots().get(k).map(|p| p.value()))
        .map(|__value| (__value).view())
    {
        Some(ValueView::Str(s)) => s.to_string(),
        _ => String::new(),
    };
    let (locale, kind, style) = (
        get("#\u{0}lf_locale"),
        get("#\u{0}lf_type"),
        get("#\u{0}lf_style"),
    );
    let lang = locale.split('-').next().unwrap_or("en");
    let parts = string_list(i, list)?;
    let patterns = crate::cldr_lists::patterns(super::service::data_locale(&locale), &kind, &style);
    let segments = assemble_segments(&parts, patterns, lang);
    if !to_parts {
        return Ok(Value::lstr(
            segments.iter().map(|(_, s)| s.as_str()).collect::<String>(),
        ));
    }
    let arr: Vec<Value> = segments
        .into_iter()
        .map(|(is_elem, v)| {
            let ob = i.new_object();
            let t = if is_elem { "element" } else { "literal" };
            set_data(&ob, "type", Value::lstr(t.to_string()));
            set_data(&ob, "value", Value::lstr(v));
            Value::Obj(ob)
        })
        .collect();
    Ok(i.make_array(arr))
}

fn resolved_options(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = brand_slot(i, &this, "#\u{0}lf")?;
    let get = |k: &str| {
        o.borrow()
            .internal_slots()
            .get(k)
            .map(|p| p.value())
            .unwrap_or(Value::Undefined)
    };
    let res = i.new_object();
    set_data(&res, "locale", get("#\u{0}lf_locale"));
    set_data(&res, "type", get("#\u{0}lf_type"));
    set_data(&res, "style", get("#\u{0}lf_style"));
    Ok(Value::Obj(res))
}
