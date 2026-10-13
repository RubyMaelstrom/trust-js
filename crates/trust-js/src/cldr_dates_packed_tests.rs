use super::*;
use std::collections::HashMap;

fn check_ref(id: TextId, seen: &mut HashMap<&'static str, TextId>) {
    assert!(
        (id as usize) < CLDR_REFS.len(),
        "text id exceeds directory: {id}"
    );
    let (offset, len) = CLDR_REFS[id as usize];
    let start = offset as usize;
    let end = start
        .checked_add(len as usize)
        .expect("CLDR string offset overflow");
    assert!(end <= CLDR_STRINGS.len(), "string ref exceeds blob: {id}");
    assert!(
        CLDR_STRINGS.is_char_boundary(start),
        "string start splits UTF-8: {id}"
    );
    assert!(
        CLDR_STRINGS.is_char_boundary(end),
        "string end splits UTF-8: {id}"
    );

    let bytes = CLDR_STRINGS
        .as_bytes()
        .get(start..end)
        .expect("checked CLDR string range");
    let decoded = std::str::from_utf8(bytes).expect("generated slice is valid UTF-8");
    assert_eq!(decoded, text(id));
    if let Some(previous) = seen.insert(decoded, id) {
        assert_eq!(
            previous, id,
            "duplicate text was not interned once: {decoded:?}"
        );
    }
}

#[test]
fn all_generated_references_are_in_bounds_utf8_and_deduplicated() {
    let mut seen = HashMap::new();
    for locale in GENERATED_LOCALES {
        for row in month_rows(locale) {
            check_ref(row.0, &mut seen);
            check_ref(row.1, &mut seen);
            check_ref(row.3, &mut seen);
        }
        for row in era_rows(locale) {
            check_ref(row.0, &mut seen);
            check_ref(row.1, &mut seen);
            check_ref(row.2, &mut seen);
            check_ref(row.3, &mut seen);
        }
    }
    assert_eq!(seen.len(), CLDR_REFS.len(), "unused string directory entry");
}

#[test]
fn lookup_api_and_unknown_locale_fallback_remain_unchanged() {
    assert_eq!(month_name("en", "gregory", "long", 1), Some("January"));
    assert_eq!(month_name("en", "gregory", "narrow", 1), Some("J"));
    assert_eq!(
        era_name("en", "gregory", "long", "0"),
        Some("Before Christ")
    );
    assert_eq!(month_name("en", "missing", "long", 1), None);
    assert_eq!(
        month_name("unsupported-locale", "gregory", "long", 1),
        Some("January")
    );
    assert_eq!(
        era_name("unsupported-locale", "gregory", "long", "0"),
        Some("Before Christ")
    );
}
