use super::*;
use std::collections::HashMap;

#[test]
fn packed_table_strings_are_bounded_valid_utf8_and_deduplicated() {
    assert!(CLDR_REFS.len() <= (u16::MAX as usize) + 1);
    assert!(CLDR_STRINGS.len() <= u32::MAX as usize);

    let mut seen = HashMap::new();
    for (index, &(offset, len)) in CLDR_REFS.iter().enumerate() {
        let id = index as TextId;
        let start = offset as usize;
        let end = start
            .checked_add(len as usize)
            .expect("generated CLDR string range overflow");
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
        let decoded = std::str::from_utf8(bytes).expect("generated text is UTF-8");
        assert_eq!(decoded, text(id));
        if let Some(previous) = seen.insert(decoded, id) {
            assert_eq!(
                previous, id,
                "duplicate text was not interned once: {decoded:?}"
            );
        }
    }
    assert_eq!(seen.len(), CLDR_REFS.len());
}

#[test]
fn every_table_preserves_code_order_and_valid_pool_ids() {
    for rows in GENERATED_TABLES {
        for row in *rows {
            assert!((row.0 as usize) < CLDR_REFS.len());
            assert!((row.1 as usize) < CLDR_REFS.len());
        }
        for pair in rows.windows(2) {
            assert!(
                text(pair[0].0) < text(pair[1].0),
                "table keys are not sorted by code text"
            );
        }
    }
}

#[test]
fn lookup_fallback_and_locale_patterns_remain_unchanged() {
    assert_eq!(name("de", "language", "long", "fr"), Some("Französisch"));
    assert_eq!(name("fr", "region", "short", "US"), Some("É.-U."));
    assert_eq!(name("fr", "region", "narrow", "US"), Some("É.-U."));
    assert_eq!(name("en", "currency", "long", "USD"), Some("US Dollar"));
    // A regional locale stores only its differences and inherits the rest along its parent
    // chain (en-GB -> en-001 -> en); an unshipped locale falls back to English.
    assert_eq!(name("en-GB", "language", "long", "fr"), Some("French"));
    assert_eq!(name("unknown", "region", "long", "US"), Some("United States"));
    assert_eq!(locale_patterns("en"), ("{0} ({1})", "{0}, {1}"));
    assert_eq!(locale_patterns("unknown"), ("{0} ({1})", "{0}, {1}"));
}
