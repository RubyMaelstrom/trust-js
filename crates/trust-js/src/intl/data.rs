//! Shared helpers used by the Intl formatting services.

/// CLDR cardinal category for an unlocalized numeric value. Formatting services with their own
/// digit options should select from their already-rounded decimal; this helper is for their default
/// fallback paths.
pub fn plural_cardinal(lang: &str, value: f64, compact_exponent: u32) -> &'static str {
    let decimal = super::numberformat::format_magnitude_options(
        value,
        1,
        0,
        3,
        None,
        None,
        1,
        "halfExpand",
        "fraction",
    );
    crate::cldr_plurals::select_cardinal(lang, &decimal, compact_exponent)
}
