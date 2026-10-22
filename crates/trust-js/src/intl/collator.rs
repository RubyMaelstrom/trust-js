//! `Intl.Collator`: Unicode 17 UCA root weights with CLDR 48 locale tailorings.

use super::service::{
    brand_slot, get_option, install_supported_locales, instance_proto, read_locale_matcher,
    resolve_locale,
};
use super::{ab, arg, canonicalize_locale_list, coerce_options, make_service};
use crate::interpreter::Interp;
use crate::value::{set_data, set_internal_slot, Gc, Value};
use std::rc::Rc;

const COLLATIONS: [&str; 12] = [
    "compat", "dict", "emoji", "eor", "phonebk", "phonetic", "pinyin", "searchjl", "stroke",
    "trad", "unihan", "zhuyin",
];

/// AvailableCollations: exactly the collation identifiers for which at least one locale's
/// Collator resolves the requested functionality. The table is already code-unit sorted.
pub(super) fn available_collations() -> &'static [&'static str] {
    &COLLATIONS
}

pub(crate) fn install(it: &mut Interp, ns: &Gc) {
    let (ctor, proto) = make_service(it, ns, "Collator", 0, construct);
    install_supported_locales(it, &ctor);
    it.def_method(&proto, "resolvedOptions", 0, resolved_options);
    install_compare_getter(it, &proto);
}

fn install_compare_getter(it: &mut Interp, proto: &Gc) {
    let g = it.make_native("get compare", 0, |i, this, _| {
        let o = brand_slot(i, &this, "#\u{0}co")?;
        if let Some(f) = o
            .borrow()
            .internal_slots()
            .get("#\u{0}co_bound")
            .map(|p| p.value())
        {
            return Ok(f);
        }
        let f = i.make_native("", 2, |i, that, a| {
            compare(i, &that, &arg(a, 0), &arg(a, 1))
        });
        let bound = crate::intl::numberformat::bind_this(i, Value::Obj(f), this.clone());
        set_internal_slot(&o, "#\u{0}co_bound", bound.clone());
        Ok(bound)
    });
    proto.borrow_mut().props.insert(
        "compare",
        crate::value::Property::accessor_prop(Some(Value::Obj(g)), None, false, true),
    );
}

/// A Collator's internal slots as InitializeCollator (ECMA-402 #sec-initializecollator) resolves
/// them.
struct CollatorSlots {
    locale: String,
    usage: String,
    sensitivity: String,
    ignore_punct: bool,
    numeric: bool,
    collation: String,
    case_first: String,
}

impl CollatorSlots {
    /// The comparison parameters these slots select.
    fn collate_opts(&self) -> CollateOpts {
        CollateOpts {
            locale: self.locale.clone(),
            collation: self.collation.clone(),
            sensitivity: self.sensitivity.clone(),
            numeric: self.numeric,
            ignore_punct: self.ignore_punct,
            upper_first: self.case_first == "upper",
            // German ä/ö/ü expand to ae/oe/ue under the phonebook collation and in search usage.
            expand_umlaut: self.locale.split('-').next() == Some("de")
                && (self.collation == "phonebk" || self.usage == "search"),
        }
    }
}

fn construct(i: &mut Interp, _t: Value, a: &[Value]) -> Result<Value, Value> {
    // Legacy service: callable without `new` (returns a fresh instance either way).
    let slots = initialize(i, &arg(a, 0), &arg(a, 1))?;
    let obj = i.new_object();
    if let Some(proto) = instance_proto(i, "Intl.Collator")? {
        obj.borrow_mut().proto = Some(proto);
    }
    set_internal_slot(&obj, "#\u{0}co", Value::Bool(true));
    set_internal_slot(&obj, "#\u{0}co_locale", Value::lstr(slots.locale));
    set_internal_slot(&obj, "#\u{0}co_usage", Value::lstr(slots.usage));
    set_internal_slot(&obj, "#\u{0}co_sensitivity", Value::lstr(slots.sensitivity));
    set_internal_slot(
        &obj,
        "#\u{0}co_ignorepunct",
        Value::Bool(slots.ignore_punct),
    );
    set_internal_slot(&obj, "#\u{0}co_numeric", Value::Bool(slots.numeric));
    set_internal_slot(&obj, "#\u{0}co_collation", Value::lstr(slots.collation));
    set_internal_slot(&obj, "#\u{0}co_casefirst", Value::lstr(slots.case_first));
    // Never reused, so `compare` can recognize the Collator whose parameters it last built.
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    set_internal_slot(&obj, "#\u{0}co_id", Value::Num(id as f64));
    Ok(Value::Obj(obj))
}

/// InitializeCollator's observable steps, in order: CanonicalizeLocaleList, then each option
/// read and conversion.
fn initialize(i: &mut Interp, locales: &Value, options: &Value) -> Result<CollatorSlots, Value> {
    let requested = canonicalize_locale_list(i, locales)?;
    let options = coerce_options(i, options)?;
    let usage = get_option(i, &options, "usage", &["sort", "search"], Some("sort"))?.unwrap();
    read_locale_matcher(i, &options)?;
    let collation_opt = get_option(i, &options, "collation", &[], None)?;
    let numeric = {
        let v = ab(i.get_member_key(&options, "numeric"))?;
        if matches!(v, Value::Undefined) {
            None
        } else {
            Some(i.to_boolean(&v))
        }
    };
    let case_first = get_option(i, &options, "caseFirst", &["upper", "lower", "false"], None)?;
    let sensitivity = get_option(
        i,
        &options,
        "sensitivity",
        &["base", "accent", "case", "variant"],
        Some("variant"),
    )?
    .unwrap();
    let ignore_punct_opt = {
        let v = ab(i.get_member_key(&options, "ignorePunctuation"))?;
        if matches!(v, Value::Undefined) {
            None
        } else {
            Some(i.to_boolean(&v))
        }
    };
    let resolved = resolve_locale(i, &requested, &["co", "kn", "kf"]);
    // ignorePunctuation defaults per locale: the dictionary-ordered locales (Thai) default to true.
    let ignore_punct =
        ignore_punct_opt.unwrap_or_else(|| resolved.locale.split('-').next() == Some("th"));
    let kw = |k: &str| {
        resolved
            .keywords
            .iter()
            .find(|(kk, _)| kk == k)
            .map(|(_, v)| v.clone())
    };
    let numeric_opt = numeric;
    let case_first_opt = case_first.clone();
    // The option wins over the locale's -u- keyword; a bare -u-kn (empty value) means numeric=true.
    let numeric = numeric_opt.unwrap_or_else(|| match kw("kn") {
        Some(v) => v != "false",
        None => false,
    });
    let case_first = case_first_opt
        .clone()
        .or_else(|| kw("kf"))
        .unwrap_or_else(|| "false".to_string());
    // The `-u-co-` value must be a known collation type (never the reserved standard/search); an
    // unknown one falls back to "default".
    // A collation is *supported* per locale (phonebk for German; eor broadly). The `collation`
    // option overrides the -u-co extension when supported; an unsupported value is ignored.
    let res_lang = resolved
        .locale
        .split('-')
        .next()
        .unwrap_or("en")
        .to_string();
    let supported = |c: &str| supported_collation(&res_lang, c);
    let opt_co = collation_opt
        .filter(|c| COLLATIONS.contains(&c.as_str()) && supported(c))
        .filter(|_| usage == "sort");
    let ext_co = kw("co").filter(|c| COLLATIONS.contains(&c.as_str()) && supported(c));
    let collation = opt_co
        .clone()
        .or_else(|| ext_co.clone())
        .unwrap_or_else(|| "default".to_string());

    // ResolveLocale: reflect the surviving `-u-` keywords in the resolved locale string. A keyword
    // survives when its value came from the locale extension and no differing option overrode it
    // (keys are emitted in alphabetical order: co, kf, kn).
    let mut additions: Vec<(&str, String)> = Vec::new();
    if let Some(co) = ext_co {
        if opt_co.as_ref().is_none_or(|o| *o == co) {
            additions.push(("co", co));
        }
    }
    if let Some(kf) = kw("kf") {
        if ["upper", "lower", "false"].contains(&kf.as_str())
            && case_first_opt.as_deref().is_none_or(|o| o == kf)
        {
            additions.push(("kf", kf));
        }
    }
    if let Some(kn) = kw("kn") {
        let kn_bool = kn != "false";
        if numeric_opt.is_none_or(|o| o == kn_bool) {
            // Canonical form: the `true` value is elided (`-u-kn`), `false` is spelled out.
            additions.push((
                "kn",
                if kn_bool {
                    String::new()
                } else {
                    "false".to_string()
                },
            ));
        }
    }
    let locale = if additions.is_empty() {
        resolved.locale.clone()
    } else {
        let ext: String = additions
            .iter()
            .map(|(k, v)| {
                if v.is_empty() {
                    format!("-{k}")
                } else {
                    format!("-{k}-{v}")
                }
            })
            .collect();
        format!("{}-u{}", resolved.locale, ext)
    };
    Ok(CollatorSlots {
        locale,
        usage,
        sensitivity,
        ignore_punct,
        numeric,
        collation,
        case_first,
    })
}

thread_local! {
    /// The comparison parameters of the Collator a `compare` function last used, by the id
    /// `construct` gave it: a Collator's slots never change after construction.
    static LAST_COMPARE: std::cell::RefCell<Option<(f64, Rc<CollateOpts>)>> =
        const { std::cell::RefCell::new(None) };
    /// Collators that `localeCompare` constructs from undefined options and undefined (empty
    /// key) or String locales. InitializeCollator then executes no ECMAScript code and its
    /// result depends on nothing but the locale string.
    static LOCALE_COMPARE: std::cell::RefCell<crate::fasthash::FastMap<String, Rc<CollateOpts>>> =
        std::cell::RefCell::new(Default::default());
}

/// The comparison parameters stored in Collator `o`'s internal slots.
fn stored_collate_opts(o: &Gc) -> Rc<CollateOpts> {
    let body = o.borrow();
    let id = body
        .internal_slots()
        .get("#\u{0}co_id")
        .and_then(|p| p.number_value());
    if let Some(opts) = LAST_COMPARE.with_borrow(|last| {
        last.as_ref()
            .filter(|(last, _)| Some(*last) == id)
            .map(|(_, opts)| opts.clone())
    }) {
        return opts;
    }
    let get = |k: &str| match body.internal_slots().get(k).map(|p| p.value()) {
        Some(Value::Str(s)) => s.to_string(),
        _ => String::new(),
    };
    let getb = |k: &str| {
        matches!(
            body.internal_slots().get(k).map(|p| p.value()),
            Some(Value::Bool(true))
        )
    };
    let opts = Rc::new(
        CollatorSlots {
            locale: get("#\u{0}co_locale"),
            usage: get("#\u{0}co_usage"),
            sensitivity: get("#\u{0}co_sensitivity"),
            ignore_punct: getb("#\u{0}co_ignorepunct"),
            numeric: getb("#\u{0}co_numeric"),
            collation: get("#\u{0}co_collation"),
            case_first: get("#\u{0}co_casefirst"),
        }
        .collate_opts(),
    );
    if let Some(id) = id {
        LAST_COMPARE.set(Some((id, opts.clone())));
    }
    opts
}

fn compare(i: &mut Interp, this: &Value, a: &Value, b: &Value) -> Result<Value, Value> {
    let o = brand_slot(i, this, "#\u{0}co")?;
    let sa = ab(i.to_string(a))?;
    let sb = ab(i.to_string(b))?;
    Ok(compare_strings(&sa, &sb, &stored_collate_opts(&o)))
}

/// CompareStrings (ECMA-402 #sec-collator-comparestrings) as a Number.
fn compare_strings(a: &str, b: &str, opts: &CollateOpts) -> Value {
    Value::Num(match collate(a, b, opts) {
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.0,
    })
}

/// String.prototype.localeCompare (ECMA-402 #sec-string.prototype.localecompare) after the
/// ToString conversions: CompareStrings with a Collator constructed from `locales` and
/// `options`. Construct(%Intl.Collator%) observes nothing but InitializeCollator's reads, so
/// only those run; with undefined options and undefined or String locales there are none, and
/// the resolved parameters come from a cache.
pub(crate) fn locale_compare(
    i: &mut Interp,
    a: &str,
    b: &str,
    locales: &Value,
    options: &Value,
) -> Result<Value, Value> {
    let key = match (locales, options) {
        (Value::Undefined, Value::Undefined) => Some(""),
        (Value::Str(locale), Value::Undefined) if !locale.is_empty() => Some(locale.as_str()),
        _ => None,
    };
    let Some(key) = key else {
        let opts = initialize(i, locales, options)?.collate_opts();
        return Ok(compare_strings(a, b, &opts));
    };
    if let Some(opts) = LOCALE_COMPARE.with_borrow(|cache| cache.get(key).cloned()) {
        return Ok(compare_strings(a, b, &opts));
    }
    let opts = Rc::new(initialize(i, locales, options)?.collate_opts());
    LOCALE_COMPARE.with_borrow_mut(|cache| {
        // Locale strings come from scripts; keep the cache small.
        if cache.len() >= 16 {
            cache.clear();
        }
        cache.insert(key.to_owned(), opts.clone());
    });
    Ok(compare_strings(a, b, &opts))
}

struct CollateOpts {
    locale: String,
    collation: String,
    sensitivity: String,
    numeric: bool,
    ignore_punct: bool,
    upper_first: bool,
    expand_umlaut: bool,
}

struct CollationElements {
    elements: Vec<crate::unicode_collation::Element>,
    case: Vec<bool>,
}

fn normalized_input(s: &str, opts: &CollateOpts) -> Vec<u32> {
    // ASCII text is already in NFD and has no umlaut to expand.
    if s.is_ascii() {
        return s.bytes().map(u32::from).collect();
    }
    let cps = crate::jstr::code_points(s);
    let nfd = crate::unicode_norm_impl::decompose(&cps, false);
    if !opts.expand_umlaut {
        return nfd;
    }
    let mut output = Vec::with_capacity(nfd.len());
    let mut index = 0;
    while index < nfd.len() {
        let code_point = nfd[index];
        output.push(code_point);
        if matches!(code_point, 0x41 | 0x4f | 0x55 | 0x61 | 0x6f | 0x75)
            && nfd.get(index + 1) == Some(&0x308)
        {
            // CLDR de-phonebook: AE/ae << Ä/ä, OE/oe << Ö/ö, UE/ue << Ü/ü. Keep the
            // diaeresis as a secondary element after the primary expansion.
            output.push(if matches!(code_point, 0x41 | 0x4f | 0x55) {
                'E' as u32
            } else {
                'e' as u32
            });
            output.push(0x308);
            index += 2;
            continue;
        }
        index += 1;
    }
    output
}

fn append_numeric_run(
    input: &[u32],
    at: usize,
    output: &mut Vec<crate::unicode_collation::Element>,
) -> Option<usize> {
    if !(0x30..=0x39).contains(input.get(at)?) {
        return None;
    }
    let mut end = at;
    while input.get(end).is_some_and(|cp| (0x30..=0x39).contains(cp)) {
        end += 1;
    }
    let significant = input[at..end]
        .iter()
        .position(|cp| *cp != 0x30)
        .map(|offset| &input[at + offset..end])
        .unwrap_or(&input[end - 1..end]);
    let mut push = |primary: u64| {
        output.push(crate::unicode_collation::Element {
            primary,
            secondary: 0x20,
            tertiary: 2,
            variable: false,
        });
    };
    // A marker at the DUCET digit position, followed by length and digit pseudo-weights, makes
    // arbitrary-size digit runs compare by mathematical value without integer conversion.
    push(0x21e6_0000_0000);
    push(0x1_0000_0000_0000 + significant.len() as u64);
    for digit in significant {
        push(0x2_0000_0000_0000 + u64::from(digit - 0x30));
    }
    Some(end - at)
}

fn tailoring_fold(code_point: u32) -> u32 {
    char::from_u32(code_point)
        .and_then(|character| character.to_lowercase().next())
        .map_or(code_point, |character| character as u32)
}

fn append_tailored_if(
    input: &[u32],
    at: usize,
    source: &[u32],
    anchor: u32,
    primary_rank: u32,
    secondary_rank: u16,
    output: &mut Vec<crate::unicode_collation::Element>,
) -> Option<usize> {
    let candidate = input.get(at..at + source.len())?;
    if !candidate
        .iter()
        .zip(source)
        .all(|(actual, expected)| tailoring_fold(*actual) == *expected)
    {
        return None;
    }
    let uppercase = candidate
        .iter()
        .filter_map(|code_point| char::from_u32(*code_point))
        .find(|character| character.is_alphabetic())
        .is_some_and(char::is_uppercase);
    let anchor = crate::unicode_collation::first_primary(anchor)
        .expect("CLDR tailoring reset anchors are present in DUCET");
    output.push(crate::unicode_collation::Element {
        primary: anchor + u64::from(primary_rank),
        secondary: 0x20 + secondary_rank,
        tertiary: if uppercase { 8 } else { 2 },
        variable: false,
    });
    Some(source.len())
}

/// Apply the compact CLDR 48 rules whose relations differ from the root collation for TRust JS's
/// advertised locales. Sources are NFD because both UTS #10 and the LDML tailoring syntax operate
/// on canonically decomposed input. Longest contractions are tested before singleton relations.
fn append_cldr_tailoring(
    input: &[u32],
    at: usize,
    opts: &CollateOpts,
    output: &mut Vec<crate::unicode_collation::Element>,
) -> Option<usize> {
    let language = opts.locale.split('-').next().unwrap_or("en");
    let collation = opts.collation.as_str();

    // CLDR's large `<*...` Japanese and Chinese primary chains are generated into a compact
    // code-point/rank map. The reset is `[last regular]`, immediately before implicit Han.
    let han_order = match (language, collation) {
        ("ja", "default") => Some(crate::cldr_collation::HanOrder::Japanese),
        ("zh", "pinyin") => Some(crate::cldr_collation::HanOrder::Pinyin),
        ("zh", "stroke") => Some(crate::cldr_collation::HanOrder::Stroke),
        ("zh", "zhuyin") => Some(crate::cldr_collation::HanOrder::Zhuyin),
        ("zh", "default") if opts.locale.split('-').any(|subtag| subtag == "Hant") => {
            Some(crate::cldr_collation::HanOrder::Stroke)
        }
        ("zh", "default") => Some(crate::cldr_collation::HanOrder::Pinyin),
        _ => None,
    };
    if let Some(rank) =
        han_order.and_then(|order| crate::cldr_collation::primary_rank(order, *input.get(at)?))
    {
        output.push(crate::unicode_collation::Element {
            primary: 0xfa00_0000_0000 + u64::from(rank),
            secondary: 0x20,
            tertiary: 2,
            variable: false,
        });
        return Some(1);
    }

    let mut rule = |source: &[u32], anchor, primary, secondary| {
        append_tailored_if(input, at, source, anchor, primary, secondary, output)
    };

    // es.xml: &N<n-tilde; traditional additionally has C<ch and L<ll.
    if language == "es" {
        if collation == "trad" {
            if let Some(length) = rule(&[0x63, 0x68], 'c' as u32, 1, 0) {
                return Some(length);
            }
            if let Some(length) = rule(&[0x6c, 0x6c], 'l' as u32, 1, 0) {
                return Some(length);
            }
        }
        if let Some(length) = rule(&[0x6e, 0x303], 'n' as u32, 1, 0) {
            return Some(length);
        }
    }

    // ln.xml phonetic digraphs. Equal-primary case variants are handled by the case level.
    if language == "ln" && collation == "phonetic" {
        const DIGRAPHS: &[(&[u32], u32, u32)] = &[
            (&[0x6e, 0x67, 0x62], 0x6e, 3),
            (&[0x67, 0x62], 0x67, 1),
            (&[0x6b, 0x70], 0x6b, 1),
            (&[0x6d, 0x62], 0x6d, 1),
            (&[0x6d, 0x66], 0x6d, 2),
            (&[0x6d, 0x70], 0x6d, 3),
            (&[0x6d, 0x76], 0x6d, 4),
            (&[0x6e, 0x64], 0x6e, 1),
            (&[0x6e, 0x67], 0x6e, 2),
            (&[0x6e, 0x6b], 0x6e, 4),
            (&[0x6e, 0x73], 0x6e, 5),
            (&[0x6e, 0x74], 0x6e, 6),
            (&[0x6e, 0x79], 0x6e, 7),
            (&[0x6e, 0x7a], 0x6e, 8),
            (&[0x73, 0x68], 0x73, 1),
            (&[0x74, 0x73], 0x74, 1),
        ];
        for &(source, anchor, primary) in DIGRAPHS {
            if let Some(length) = rule(source, anchor, primary, 0) {
                return Some(length);
            }
        }
    }

    let rules: &[(&[u32], u32, u32, u16)] = match language {
        // pl.xml: each accented letter is a distinct primary after its base.
        "pl" => &[
            (&[0x61, 0x328], 0x61, 1, 0),
            (&[0x63, 0x301], 0x63, 1, 0),
            (&[0x65, 0x328], 0x65, 1, 0),
            (&[0x142], 0x6c, 1, 0),
            (&[0x6e, 0x301], 0x6e, 1, 0),
            (&[0x6f, 0x301], 0x6f, 1, 0),
            (&[0x73, 0x301], 0x73, 1, 0),
            (&[0x7a, 0x301], 0x7a, 1, 0),
            (&[0x7a, 0x307], 0x7a, 2, 0),
        ],
        // sl.xml: C<caron C<acute, D<stroke, S<caron, Z<caron.
        "sl" => &[
            (&[0x63, 0x30c], 0x63, 1, 0),
            (&[0x63, 0x301], 0x63, 2, 0),
            (&[0x111], 0x64, 1, 0),
            (&[0x73, 0x30c], 0x73, 1, 0),
            (&[0x7a, 0x30c], 0x7a, 1, 0),
        ],
        // sv.xml: the three primary groups after Z, plus its secondary-equivalent letters.
        "sv" => &[
            (&[0x61, 0x30a], 0x7a, 1, 0),
            (&[0x61, 0x308], 0x7a, 2, 0),
            (&[0xe6], 0x7a, 2, 1),
            (&[0x65, 0x328], 0x7a, 2, 2),
            (&[0x6f, 0x308], 0x7a, 3, 0),
            (&[0xf8], 0x7a, 3, 1),
            (&[0x6f, 0x30b], 0x7a, 3, 2),
            (&[0x153], 0x7a, 3, 3),
            (&[0x6f, 0x302], 0x7a, 3, 4),
            (&[0x111], 0x64, 0, 1),
            (&[0xf0], 0x64, 0, 2),
            (&[0xfe], 0x74, 0, 1),
            (&[0x75, 0x308], 0x79, 0, 1),
            (&[0x75, 0x30b], 0x79, 0, 2),
        ],
        // ln.xml standard relations; the phonetic rules above extend these.
        "ln" => &[(&[0x25b], 0x65, 1, 0), (&[0x254], 0x6f, 0, 1)],
        // hi.xml: OM < ANUSVARA << CANDRABINDU < VISARGA.
        "hi" => &[
            (&[0x902], 0x950, 1, 0),
            (&[0x901], 0x950, 1, 1),
            (&[0x903], 0x950, 2, 0),
        ],
        // si.xml: AU < ANUSVARAYA < VISARGAYA and JNYA < TAALUJA NAASIKYAYA.
        "si" => &[
            (&[0xd82], 0xd96, 1, 0),
            (&[0xd83], 0xd96, 2, 0),
            (&[0xda4], 0xda5, 1, 0),
        ],
        _ => &[],
    };
    for &(source, anchor, primary, secondary) in rules {
        if let Some(length) = rule(source, anchor, primary, secondary) {
            return Some(length);
        }
    }
    None
}

/// Whether [`append_cldr_tailoring`] has any rule for the locale's language.
fn has_tailoring(opts: &CollateOpts) -> bool {
    matches!(
        opts.locale.split('-').next(),
        Some("es" | "ln" | "pl" | "sl" | "sv" | "hi" | "si" | "ja" | "zh")
    )
}

fn elements(s: &str, opts: &CollateOpts) -> CollationElements {
    let input = normalized_input(s, opts);
    let tailored = has_tailoring(opts);
    let case = input
        .iter()
        .filter_map(|code_point| char::from_u32(*code_point))
        .filter(|character| character.is_uppercase() || character.is_lowercase())
        .map(char::is_uppercase)
        .collect();
    let mut elements = Vec::with_capacity(input.len());
    let mut at = 0;
    while at < input.len() {
        if opts.numeric {
            if let Some(length) = append_numeric_run(&input, at, &mut elements) {
                at += length;
                continue;
            }
        }
        if tailored {
            if let Some(length) = append_cldr_tailoring(&input, at, opts, &mut elements) {
                at += length;
                continue;
            }
        }
        #[cfg(test)]
        if opts.locale == "und" {
            at += crate::unicode_collation::append_ducet_mapping(&input, at, &mut elements);
            continue;
        }
        if input.get(at + 1).is_none_or(|next| *next < 0x80) {
            if let Some(mapped) = crate::unicode_collation::ascii_mapping(input[at]) {
                elements.extend_from_slice(mapped);
                at += 1;
                continue;
            }
        }
        at += crate::unicode_collation::append_mapping(&input, at, &mut elements);
    }
    CollationElements { elements, case }
}

/// UTS #10 comparison: NFD, longest-match DUCET collation-element production, then successive
/// non-zero primary, secondary, and tertiary weights according to ECMA-402 sensitivity.
fn collate(a: &str, b: &str, opts: &CollateOpts) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let ea = elements(a, opts);
    let eb = elements(b, opts);
    let active =
        |element: &&crate::unicode_collation::Element| !(opts.ignore_punct && element.variable);
    let prim = ea
        .elements
        .iter()
        .filter(&active)
        .map(|element| element.primary)
        .filter(|weight| *weight != 0)
        .cmp(
            eb.elements
                .iter()
                .filter(&active)
                .map(|element| element.primary)
                .filter(|weight| *weight != 0),
        );
    if prim != Ordering::Equal {
        return prim;
    }
    let sens = opts.sensitivity.as_str();
    if sens == "accent" || sens == "variant" {
        let sec = ea
            .elements
            .iter()
            .filter(&active)
            .map(|element| element.secondary)
            .filter(|weight| *weight != 0)
            .cmp(
                eb.elements
                    .iter()
                    .filter(&active)
                    .map(|element| element.secondary)
                    .filter(|weight| *weight != 0),
            );
        if sec != Ordering::Equal {
            return sec;
        }
    }
    if sens == "case" || (sens == "variant" && opts.upper_first) {
        let case_weight = |upper: &bool| *upper != opts.upper_first;
        let case = ea
            .case
            .iter()
            .map(case_weight)
            .cmp(eb.case.iter().map(case_weight));
        if case != Ordering::Equal {
            return case;
        }
    }
    if sens == "variant" {
        let ter = ea
            .elements
            .iter()
            .filter(&active)
            .map(|element| element.tertiary)
            .filter(|weight| *weight != 0)
            .cmp(
                eb.elements
                    .iter()
                    .filter(&active)
                    .map(|element| element.tertiary)
                    .filter(|weight| *weight != 0),
            );
        if ter != Ordering::Equal {
            return ter;
        }
    }
    Ordering::Equal
}

fn resolved_options(i: &mut Interp, this: Value, _a: &[Value]) -> Result<Value, Value> {
    let o = brand_slot(i, &this, "#\u{0}co")?;
    let get = |k: &str| {
        o.borrow()
            .internal_slots()
            .get(k)
            .map(|p| p.value())
            .unwrap_or(Value::Undefined)
    };
    let res = i.new_object();
    set_data(&res, "locale", get("#\u{0}co_locale"));
    set_data(&res, "usage", get("#\u{0}co_usage"));
    set_data(&res, "sensitivity", get("#\u{0}co_sensitivity"));
    set_data(&res, "ignorePunctuation", get("#\u{0}co_ignorepunct"));
    set_data(&res, "collation", get("#\u{0}co_collation"));
    set_data(&res, "numeric", get("#\u{0}co_numeric"));
    set_data(&res, "caseFirst", get("#\u{0}co_casefirst"));
    Ok(Value::Obj(res))
}

/// Whether a locale's Collator supports a collation type (mirrors the CLDR availability that
/// Intl.supportedValuesOf("collation") reflects). "eor"/"emoji" are available everywhere.
fn supported_collation(lang: &str, c: &str) -> bool {
    match c {
        "eor" | "emoji" => true,
        "phonebk" => lang == "de",
        "compat" => lang == "ar",
        "trad" => lang == "es",
        "dict" => lang == "si",
        "phonetic" => lang == "ln",
        "searchjl" => lang == "ko",
        "pinyin" | "stroke" | "zhuyin" | "unihan" => lang == "zh",
        _ => false,
    }
}

/// The non-default entries in this locale's `%Intl.Collator%.[[SortLocaleData]].[[co]]`, sorted as
/// required by ECMA-402 CollationsOfLocale.
pub(super) fn supported_collations(lang: &str) -> Vec<&'static str> {
    let mut collations = COLLATIONS
        .iter()
        .copied()
        .filter(|collation| supported_collation(lang, collation))
        .collect::<Vec<_>>();
    collations.sort_unstable();
    collations
}

#[cfg(test)]
mod tests {
    use super::*;

    const UCA_NON_IGNORABLE: &str =
        include_str!("../../tests/unicode-17.0.0/CollationTest_NON_IGNORABLE_SHORT.txt");
    const CLDR_NON_IGNORABLE: &str =
        include_str!("../../tests/cldr-48/CollationTest_CLDR_NON_IGNORABLE_SHORT.txt");

    fn from_code_points(line: &str) -> String {
        let code_points = line
            .split_whitespace()
            .map(|value| u32::from_str_radix(value, 16).unwrap())
            .collect::<Vec<_>>();
        crate::jstr::from_code_points(&code_points)
    }

    #[test]
    fn unicode_17_uca_non_ignorable_conformance() {
        let options = CollateOpts {
            locale: "und".to_string(),
            collation: "default".to_string(),
            sensitivity: "variant".to_string(),
            numeric: false,
            ignore_punct: false,
            upper_first: false,
            expand_umlaut: false,
        };
        let mut previous: Option<(usize, String)> = None;
        let mut checked = 0;
        for (index, raw) in UCA_NON_IGNORABLE.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let current = from_code_points(line);
            if let Some((previous_index, previous_value)) = &previous {
                assert!(
                    collate(previous_value, &current, &options) != std::cmp::Ordering::Greater,
                    "UCA order regression between corpus lines {} and {}",
                    previous_index + 1,
                    index + 1
                );
            }
            previous = Some((index, current));
            checked += 1;
        }
        assert!(checked > 10_000, "official short UCA corpus was truncated");
    }

    #[test]
    fn cldr_48_root_non_ignorable_conformance() {
        let options = CollateOpts {
            locale: "en".to_string(),
            collation: "default".to_string(),
            sensitivity: "variant".to_string(),
            numeric: false,
            ignore_punct: false,
            upper_first: false,
            expand_umlaut: false,
        };
        let mut previous: Option<(usize, String)> = None;
        let mut checked = 0;
        for (index, raw) in CLDR_NON_IGNORABLE.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let current = from_code_points(line);
            if let Some((previous_index, previous_value)) = &previous {
                assert!(
                    collate(previous_value, &current, &options) != std::cmp::Ordering::Greater,
                    "CLDR root order regression between corpus lines {} and {}",
                    previous_index + 1,
                    index + 1
                );
            }
            previous = Some((index, current));
            checked += 1;
        }
        assert!(checked > 10_000, "official short CLDR corpus was truncated");
    }
    #[test]
    fn ascii_fast_path_keeps_contractions_and_tailorings() {
        let options = |locale: &str, collation: &str| CollateOpts {
            locale: locale.to_string(),
            collation: collation.to_string(),
            sensitivity: "variant".to_string(),
            numeric: false,
            ignore_punct: false,
            upper_first: false,
            expand_umlaut: false,
        };
        let root = options("en", "default");
        // `l` before U+00B7 starts a contraction: the following non-ASCII point disables the
        // ASCII table for that position.
        let contraction = elements("l\u{b7}", &root).elements;
        let mut expected = Vec::new();
        crate::unicode_collation::append_mapping(&[0x6c, 0xb7], 0, &mut expected);
        assert_eq!(contraction, expected);
        for code_point in 0..0x80u32 {
            let text = char::from_u32(code_point).unwrap().to_string() + "a";
            let mut slow = Vec::new();
            let mut at = 0;
            let input = [code_point, 'a' as u32];
            while at < input.len() {
                at += crate::unicode_collation::append_mapping(&input, at, &mut slow);
            }
            assert_eq!(elements(&text, &root).elements, slow, "{code_point:#x}");
        }
        // Spanish traditional ch/ll and Lingala phonetic digraphs are ASCII tailorings.
        let trad = options("es", "trad");
        assert_eq!(collate("cz", "ch", &trad), std::cmp::Ordering::Less);
        assert_eq!(collate("cz", "ch", &root), std::cmp::Ordering::Greater);
        let ln = options("ln", "phonetic");
        assert_eq!(collate("gz", "gb", &ln), std::cmp::Ordering::Less);
    }
}

#[cfg(test)]
mod locale_compare_tests {
    use crate::{bytecode::Tier, Completion, Engine};

    #[test]
    fn locale_compare_constructs_its_collator_unobservably_and_caches_by_locale() {
        // Expected values from Node v24.
        let source = r#"(function(){
    var log=[],r=[];
    function show(f){try{return f()}catch(e){return e.name}}
    var words=['a-b','a0b','a_b','ab','Ab','aB','äb','ae','b','B','résumé','resume','Résumé','10','9','2','z','Z','ß','ss','','a b','áb'];
    r.push(words.slice().sort(function(x,y){return x.localeCompare(y)}).join('|'));
    r.push(words.slice().sort(function(x,y){return x.localeCompare(y,'de')}).join('|'));
    r.push(words.slice().sort(function(x,y){return x.localeCompare(y,'sv')}).join('|'));
    r.push(words.slice().sort(function(x,y){return x.localeCompare(y,undefined,{numeric:true,sensitivity:'base'})}).join('|'));
    r.push(words.slice().sort(new Intl.Collator('de',{usage:'search'}).compare).join('|'));
    var c=new Intl.Collator('en',{caseFirst:'upper'});r.push(words.slice().sort(c.compare).join('|'),c.compare('a','A'),new Intl.Collator().compare('a','A'));
    r.push('a'.localeCompare('b'),'b'.localeCompare('a'),'a'.localeCompare('a'),'a'.localeCompare('A'),'a'.localeCompare('á',undefined,{sensitivity:'base'}));
    r.push(show(()=>'a'.localeCompare('b','x-invalid-tag-')),show(()=>'a'.localeCompare('b',undefined,null)));
    var opts={get sensitivity(){log.push('sensitivity');return 'base'},get numeric(){log.push('numeric');return false},get usage(){log.push('usage');return 'sort'}};
    r.push('a'.localeCompare('A',undefined,opts),log.splice(0).join());
    r.push('a'.localeCompare('b',['de','en']),'a'.localeCompare('b',{length:1,0:'de'}));
    var proto=Intl.Collator.prototype,desc=Object.getOwnPropertyDescriptor(proto,'compare');
    Object.defineProperty(proto,'compare',{configurable:true,get(){log.push('compare getter');return function(){return 42}}});
    r.push('a'.localeCompare('b'),'a'.localeCompare('b','fr'),log.splice(0).join());Object.defineProperty(proto,'compare',desc);
    var IC=Intl.Collator;Intl.Collator=function(){log.push('global ctor');return {compare:function(){return 7}}};
    r.push('b'.localeCompare('a'),log.splice(0).join());Intl.Collator=IC;
    r.push(show(()=>String.prototype.localeCompare.call(null,'a')),String.prototype.localeCompare.call(5,'5'),'x'.localeCompare({toString(){log.push('ts');return 'y'}}),log.splice(0).join());
    var c1=new Intl.Collator('sv'),c2=new Intl.Collator('en');r.push(c1.compare('ä','z'),c2.compare('ä','z'),c1.compare('ä','z'),c2.compare('ä','z'));
    return r.join('#');
})()"#;
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            match engine
                .eval(source, false)
                .expect("collation fixture parses")
            {
                Completion::Value(actual) => assert_eq!(
                    actual,
                    "|10|2|9|a b|a_b|a-b|a0b|ab|aB|Ab|áb|äb|ae|b|B|resume|résumé|Résumé|ss|ß|z|Z\
                     #|10|2|9|a b|a_b|a-b|a0b|ab|aB|Ab|áb|äb|ae|b|B|resume|résumé|Résumé|ss|ß|z|\
                     Z#|10|2|9|a b|a_b|a-b|a0b|ab|aB|Ab|áb|ae|b|B|resume|résumé|Résumé|ss|ß|z|Z|\
                     äb#|2|9|10|a b|a_b|a-b|a0b|ab|Ab|aB|äb|áb|ae|b|B|résumé|resume|Résumé|ß|ss|\
                     z|Z#|10|2|9|a b|a_b|a-b|a0b|ab|aB|Ab|áb|ae|äb|b|B|resume|résumé|Résumé|ss|ß\
                     |z|Z#|10|2|9|a b|a_b|a-b|a0b|Ab|aB|ab|áb|äb|ae|B|b|resume|Résumé|résumé|ss|\
                     ß|Z|z#1#-1#-1#1#0#-1#0#RangeError#TypeError#0#usage,numeric,sensitivity#-1#-\
                     1#-1#-1##1##TypeError#0#-1#ts#1#-1#1#-1",
                    "{tier:?}"
                ),
                Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
            }
        }
    }
}
