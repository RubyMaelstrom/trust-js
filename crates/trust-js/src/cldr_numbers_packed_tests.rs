use super::*;
use std::collections::{HashMap, HashSet};

#[test]
fn cldr_text_range_proof_rejects_overflow_bounds_and_utf8_splits() {
    let bytes = "aéz".as_bytes();

    assert!(cldr_text_range_is_valid(bytes, 1, 2));
    assert!(cldr_text_range_is_valid(bytes, 0, bytes.len()));
    assert!(!cldr_text_range_is_valid(bytes, usize::MAX, 1));
    assert!(!cldr_text_range_is_valid(bytes, 1, 1));
    assert!(!cldr_text_range_is_valid(bytes, 2, 1));
    assert!(!cldr_text_range_is_valid(bytes, 3, 2));
}

#[test]
fn every_cldr_ref_satisfies_the_unchecked_slice_proof() {
    assert!(cldr_refs_are_valid(CLDR_STRINGS.as_bytes(), &CLDR_REFS));
    for (index, &(offset, len)) in CLDR_REFS.iter().enumerate() {
        assert!(
            cldr_text_range_is_valid(CLDR_STRINGS.as_bytes(), offset as usize, len as usize),
            "invalid range at text id {index}: ({offset}, {len})"
        );
    }
}

#[test]
fn cldr_ref_table_proof_rejects_an_invalid_entry() {
    let bytes = "é".as_bytes();
    assert!(!cldr_refs_are_valid(bytes, &[(0, 1)]));
    assert!(!cldr_refs_are_valid(bytes, &[(u32::MAX, 1)]));
}

fn check_id(id: TextId, seen: &mut HashSet<TextId>, values: &mut HashMap<&'static str, TextId>) {
    assert!(
        (id as usize) < CLDR_REFS.len(),
        "text id exceeds directory: {id}"
    );
    let (offset, len) = CLDR_REFS[id as usize];
    let start = offset as usize;
    let end = start
        .checked_add(len as usize)
        .expect("CLDR offset overflow");
    assert!(end <= CLDR_STRINGS.len(), "string ref exceeds blob: {id}");
    assert!(
        CLDR_STRINGS.is_char_boundary(start),
        "string start splits UTF-8: {id}"
    );
    assert!(
        CLDR_STRINGS.is_char_boundary(end),
        "string end splits UTF-8: {id}"
    );
    let value = text(id);
    assert_eq!(value.as_bytes(), &CLDR_STRINGS.as_bytes()[start..end]);
    if let Some(previous) = values.insert(value, id) {
        assert_eq!(previous, id, "duplicate string has multiple ids: {value:?}");
    }
    seen.insert(id);
}

#[test]
fn every_generated_table_reference_is_valid_and_interned_once() {
    let mut ids = HashSet::new();
    let mut values = HashMap::new();
    let mut check = |id| check_id(id, &mut ids, &mut values);

    for row in SYMBOLS {
        check(row.0);
        check(row.1);
        for id in [
            row.2.decimal,
            row.2.group,
            row.2.percent,
            row.2.plus,
            row.2.minus,
            row.2.approximately,
            row.2.exponential,
            row.2.infinity,
            row.2.nan,
            row.2.currency_decimal,
            row.2.currency_group,
        ] {
            check(id);
        }
    }
    for row in PATTERNS {
        for id in [
            row.0,
            row.1,
            row.2,
            row.3.positive_prefix,
            row.3.positive_suffix,
            row.3.negative_prefix,
            row.3.negative_suffix,
        ] {
            check(id);
        }
    }
    for row in COMPACT {
        for id in [row.0, row.1, row.2, row.4, row.5.prefix, row.5.suffix] {
            check(id);
        }
    }
    for row in COMPACT_MAX {
        check(row.0);
        check(row.1);
        check(row.2);
    }
    for locale in GENERATED_LOCALES {
        for row in currency_rows(locale) {
            check(row.0);
            check(row.1.name);
            check(row.1.symbol);
            check(row.1.narrow);
        }
        for row in currency_name_rows(locale) {
            check(row.0);
            check(row.1);
            check(row.2);
        }
    }
    for row in CURRENCY_UNITS {
        check(row.0);
        check(row.1);
        check(row.2);
    }
    for row in DEFAULT_NUMBERING {
        check(row.0);
        check(row.1);
    }
    for row in CURRENCY_FORMATS {
        check(row.0);
        check(row.1);
        check(row.2.decimal);
        check(row.2.group);
        if let Some(pattern) = row.2.pattern {
            for id in [
                pattern.positive_prefix,
                pattern.positive_suffix,
                pattern.negative_prefix,
                pattern.negative_suffix,
            ] {
                check(id);
            }
        }
    }

    assert_eq!(ids.len(), CLDR_REFS.len(), "unused string directory entry");
    assert_eq!(
        values.len(),
        CLDR_REFS.len(),
        "duplicate text was interned twice"
    );
}

#[test]
fn packed_lookups_preserve_common_values_and_fallbacks() {
    let en_symbols = symbols("en", "latn");
    assert_eq!(en_symbols.decimal, ".");
    assert_eq!(en_symbols.group, ",");
    assert_eq!(pattern("en", "latn", "decimal").positive_prefix, "");
    assert_eq!(
        compact("en", "latn", "short", 3, "other", false)
            .unwrap()
            .suffix,
        "K"
    );
    let usd = currency("en", "USD", "other").unwrap();
    assert_eq!(usd.name, "US dollars");
    assert_eq!(usd.symbol, "$");
    assert_eq!(currency_unit("en", "one"), "{0} {1}");
    assert_eq!(symbols("unsupported-locale", "latn").decimal, ".");
}
