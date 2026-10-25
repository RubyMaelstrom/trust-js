//! Shared machinery for the `Intl.*` formatting services: locale resolution, the `localeMatcher`
//! option, `supportedLocalesOf`, and small option readers.

use super::{ab, arg, canonicalize_locale_list};
use crate::interpreter::Interp;
use crate::intl::tags;
use crate::value::ValueView;
use crate::value::{Gc, Property, Value};

/// ECMA-402 DefaultLocale is implementation-defined. TRust JS deliberately favors US English when a
/// caller supplies no supported locale, matching TRust's primary compatibility target.
pub const DEFAULT_LOCALE: &str = "en-US";

/// LookupMatchingLocaleByPrefix (ECMA-402 #sec-lookupmatchinglocalebyprefix) of one requested
/// locale against TRust JS's [[AvailableLocales]] ([`crate::cldr_locales::AVAILABLE`], shared by
/// every service): its longest available prefix, ignoring Unicode locale extension sequences.
/// LookupMatchingLocaleByBestFit uses the same matching: the available locales include each
/// default-content and script-less form (`en-US`, `zh-TW`), so a prefix is the best fit.
pub fn best_available_locale(requested: &str) -> Option<&'static str> {
    let without_extension = strip_unicode_extension(requested);
    let mut prefix = without_extension.as_str();
    while !prefix.is_empty() {
        if let Ok(index) = crate::cldr_locales::AVAILABLE.binary_search_by(|row| row.0.cmp(prefix))
        {
            return Some(crate::cldr_locales::AVAILABLE[index].0);
        }
        let mut position = prefix.rfind('-').unwrap_or(0);
        // A singleton subtag is removed together with the subtag after it.
        while position >= 2 && prefix.as_bytes()[position - 2] == b'-' {
            position -= 2;
        }
        prefix = &prefix[..position];
    }
    None
}

/// A language tag without its Unicode locale extension sequences (`-u-…`).
fn strip_unicode_extension(tag: &str) -> String {
    let subtags: Vec<&str> = tag.split('-').collect();
    let mut kept = Vec::with_capacity(subtags.len());
    let mut index = 0;
    while index < subtags.len() {
        let subtag = subtags[index];
        if subtag.eq_ignore_ascii_case("x") {
            kept.extend_from_slice(&subtags[index..]);
            break;
        }
        if subtag.eq_ignore_ascii_case("u") && index > 0 {
            index += 1;
            while index < subtags.len() && subtags[index].len() != 1 {
                index += 1;
            }
            continue;
        }
        kept.push(subtag);
        index += 1;
    }
    kept.join("-")
}

/// The shipped CLDR locale whose data serves a resolved locale (an available tag, possibly
/// with a Unicode extension); English for anything else.
pub fn data_locale(resolved: &str) -> &'static str {
    best_available_locale(resolved)
        .and_then(crate::cldr_locales::data_locale)
        .unwrap_or("en")
}

/// The result of ResolveLocale: the chosen locale's base name plus the Unicode `-u-` keywords it
/// carried (only those relevant to the service are kept by the caller).
pub struct ResolvedLocale {
    pub locale: String,
    #[allow(dead_code)]
    pub keywords: Vec<(String, String)>,
}

/// ResolveLocale's matching (ECMA-402 #sec-resolvelocale): the first requested locale with an
/// available match supplies the available locale and its requested Unicode extension keywords
/// for `relevant_keys`; with none, [`DEFAULT_LOCALE`].
pub fn resolve_locale(
    _i: &mut Interp,
    requested: &[String],
    relevant_keys: &[&str],
) -> ResolvedLocale {
    for tag in requested {
        if let Some(parsed) = tags::parse(tag) {
            if let Some(base) = best_available_locale(tag) {
                let base = base.to_string();
                let mut keywords = Vec::new();
                if let Some((_a, kws)) = &parsed.unicode {
                    for (k, types) in kws {
                        if relevant_keys.contains(&k.as_str()) {
                            keywords.push((k.clone(), types.join("-")));
                        }
                    }
                }
                return ResolvedLocale {
                    locale: base,
                    keywords,
                };
            }
        }
    }
    ResolvedLocale {
        locale: DEFAULT_LOCALE.to_string(),
        keywords: Vec::new(),
    }
}

/// Whether `n` is a numbering system this engine can render with.
pub fn is_supported_nu(n: &str) -> bool {
    n == "latn" || crate::numbering::NUMBERING.iter().any(|(id, _)| *id == n)
}

/// ResolveLocale specialised to the single `nu` key: returns the resolved locale string (the base
/// name plus a `-u-nu-<value>` addition when it survives) and the chosen numbering system. The
/// `-u-nu-` addition is kept only when the value came from the requested locale's extension; an
/// options value that differs from it drops the addition (per ResolveLocale steps for `nu`).
pub fn resolve_locale_nu(requested: &[String], option: Option<&str>) -> (String, String) {
    let mut base = DEFAULT_LOCALE.to_string();
    let mut ext_value: Option<String> = None;
    for tag in requested {
        if let Some(parsed) = tags::parse(tag) {
            if let Some(available) = best_available_locale(tag) {
                base = available.to_string();
                if let Some((_a, kws)) = &parsed.unicode {
                    for (k, types) in kws {
                        if k == "nu" {
                            ext_value = Some(types.join("-"));
                        }
                    }
                }
                break;
            }
        }
    }
    let mut value = default_numbering_system(&base).to_string();
    let mut addition: Option<String> = None;
    if let Some(ev) = &ext_value {
        if is_supported_nu(ev) {
            value = ev.clone();
            addition = Some(ev.clone());
        }
    }
    if let Some(opt) = option {
        if is_supported_nu(opt) && opt != value {
            value = opt.to_string();
            addition = None;
        }
    }
    let locale = match addition {
        Some(a) => format!("{base}-u-nu-{a}"),
        None => base,
    };
    (locale, value)
}

/// The default numbering system of an available locale: its CLDR `defaultNumberingSystem`.
pub fn default_numbering_system(base: &str) -> &'static str {
    crate::cldr_numbers::default_numbering_system(data_locale(base))
}

/// GetOption(options, "localeMatcher", string, «"lookup","best fit"», "best fit"). Validated then
/// discarded (both matchers behave identically here).
pub fn read_locale_matcher(i: &mut Interp, options: &Value) -> Result<(), Value> {
    let v = ab(i.get_member_key(options, "localeMatcher"))?;
    if matches!((v).view(), ValueView::Undefined) {
        return Ok(());
    }
    let s = ab(i.to_string(&v))?.to_string();
    if s != "lookup" && s != "best fit" {
        return Err(i.make_error("RangeError", format!("invalid localeMatcher: {s}")));
    }
    Ok(())
}

/// GetOption string with an explicit enum + fallback.
pub fn get_option(
    i: &mut Interp,
    options: &Value,
    prop: &str,
    values: &[&str],
    fallback: Option<&str>,
) -> Result<Option<String>, Value> {
    let v = ab(i.get_member_key(options, prop))?;
    if matches!((v).view(), ValueView::Undefined) {
        return Ok(fallback.map(|s| s.to_string()));
    }
    let s = ab(i.to_string(&v))?.to_string();
    if !values.is_empty() && !values.contains(&s.as_str()) {
        return Err(i.make_error("RangeError", format!("invalid {prop}: {s}")));
    }
    Ok(Some(s))
}

/// Install `supportedLocalesOf` on a service constructor.
pub(crate) fn install_supported_locales(it: &mut Interp, ctor: &Gc) {
    let f = it.make_native("supportedLocalesOf", 1, |i, _t, a| {
        let requested = canonicalize_locale_list(i, &arg(a, 0))?;
        // Validate the options bag's localeMatcher (per spec) then filter by supported language.
        let options = arg(a, 1);
        if !matches!((options).view(), ValueView::Undefined) {
            read_locale_matcher(i, &options)?;
        }
        // FilterLocales: the requested locales with an available match.
        let out: Vec<Value> = requested
            .into_iter()
            .filter(|t| best_available_locale(t).is_some())
            .map(Value::lstr)
            .collect();
        Ok(i.make_array(out))
    });
    ctor.borrow_mut()
        .props
        .insert("supportedLocalesOf", Property::builtin(Value::Obj(f)));
}

/// Brand-check helper: fetch the internal slot object of a service instance, or throw TypeError.
pub(crate) fn brand_slot(i: &mut Interp, this: &Value, marker: &str) -> Result<Gc, Value> {
    match this.as_obj() {
        Some(o) if o.borrow().internal_slots().contains(marker) => Ok(o),
        _ => Err(i.make_error("TypeError", "method called on an incompatible receiver")),
    }
}

/// OrdinaryCreateFromConstructor prototype: `new.target.prototype` when it is an object, else the
/// intrinsic prototype for this service.
/// OrdinaryCreateFromConstructor's proto lookup: `Get(newTarget, "prototype")` (propagating a
/// poisoned getter's error), falling back to the intrinsic prototype when it isn't an object.
pub(crate) fn instance_proto(i: &mut Interp, intrinsic: &str) -> Result<Option<Gc>, Value> {
    if let ValueView::Obj(nt) = &(i.constructor_new_target()).view() {
        let nt = *nt;
        match (i.get_member_key(&Value::Obj(nt), "prototype")).map(|__value| (__value).view()) {
            Ok(ValueView::Obj(p)) => return Ok(Some(p)),
            // GetPrototypeFromConstructor: when newTarget.prototype is not an object, use the
            // *newTarget's realm* intrinsic (found by matching its prototype chain's Function.prototype
            // against each realm), not necessarily the current realm's.
            Ok(_) => {
                // GetFunctionRealm throws for a proxy revoked by its own prototype getter.
                ab(i.get_function_realm_global(&nt))?;
                if let Some(protos) = realm_protos_of(i, &nt) {
                    return Ok(protos.get(intrinsic).cloned());
                }
            }
            Err(a) => return Err(ab::<()>(Err(a)).unwrap_err()),
        }
    }
    Ok(i.extra_protos.get(intrinsic).cloned())
}

/// The `extra_protos` map of the realm that owns `func`, located by walking `func`'s prototype chain
/// and matching a realm's `Function.prototype`. Returns None (→ current realm) if no match.
fn realm_protos_of<'a>(
    i: &'a Interp,
    func: &Gc,
) -> Option<&'a crate::fasthash::FastMap<&'static str, Gc>> {
    let mut cur = func.borrow().proto;
    while let Some(p) = cur {
        if crate::value::Gc::ptr_eq(&p, &i.function_proto) {
            return Some(&i.extra_protos);
        }
        for rs in i.realms.values() {
            if crate::value::Gc::ptr_eq(&p, &rs.function_proto) {
                return Some(&rs.extra_protos);
            }
        }
        cur = p.borrow().proto;
    }
    None
}
