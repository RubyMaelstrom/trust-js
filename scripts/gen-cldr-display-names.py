#!/usr/bin/env python3
"""Generate Intl.DisplayNames data for every locale TRust JS advertises from CLDR 48.

Sources are the pinned unicode-org/cldr-json 48.0.0 localenames, numbers, dates,
and BCP-47 packages.  Long/base names and sparse short/narrow overrides are kept
separate so the generated binary does not repeat the overwhelmingly identical
style fallbacks.

Every CLDR locale of a supported language is generated (scripts/cldr_locales.py): a regional or
script variant keeps only the names that differ from its parent's, and each lookup walks the
locale's parent chain (`crate::cldr_locales::chain`). `CLDR_DIR` names a local copy of the
pinned cldr-json tree.
"""

from __future__ import annotations

import json
import hashlib
import os
import pathlib
import sys
import urllib.request
from urllib.error import HTTPError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cldr_locales  # noqa: E402


TAG = "48.0.0"
SOURCE_COMMIT = "4d06be52b51bb2f75688d0abe55c52a66afed790"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/trust-js/src/cldr_display_names.rs"
BLOB_OUTPUT = ROOT / "crates/trust-js/src/cldr_display_names.data"
MANIFEST_OUTPUT = ROOT / "crates/trust-js/src/cldr_display_names.inputs.json"
REPOSITORY = "https://github.com/unicode-org/cldr-json"
BASE = f"https://raw.githubusercontent.com/unicode-org/cldr-json/{SOURCE_COMMIT}/cldr-json"
U16_MAX = (1 << 16) - 1
U32_MAX = (1 << 32) - 1
INPUT_HASHES: dict[str, tuple[int, str | None]] = {}
LOCALES = list(cldr_locales.locales())
DATE_FIELDS = {
    "era": "era",
    "year": "year",
    "quarter": "quarter",
    "month": "month",
    "weekOfYear": "week",
    "weekday": "weekday",
    "day": "day",
    "dayPeriod": "dayperiod",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "timeZoneName": "zone",
}


class StringPool:
    def __init__(self) -> None:
        self.data = bytearray()
        self.refs: dict[str, tuple[int, int, int]] = {}

    def intern(self, value: str) -> int:
        existing = self.refs.get(value)
        if existing is not None:
            return existing[0]
        string_id = len(self.refs)
        if string_id > U16_MAX:
            raise ValueError("CLDR DisplayNames string count exceeds the u16 id format")
        encoded = value.encode("utf-8")
        offset = len(self.data)
        end = offset + len(encoded)
        if end > U32_MAX:
            raise ValueError("CLDR DisplayNames string blob exceeds the u32 offset format")
        self.data.extend(encoded)
        self.refs[value] = (string_id, offset, len(encoded))
        return string_id


def fetch_json(url: str) -> dict:
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        path = pathlib.Path(cldr_dir) / url.removeprefix(f"{BASE}/")
        if not path.exists():
            INPUT_HASHES[url] = (404, None)
            raise HTTPError(url, 404, "not in the local cldr-json tree", None, None)
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "TRust JS CLDR DisplayNames generator"}
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read()
        except HTTPError as error:
            INPUT_HASHES[url] = (error.code, None)
            raise
    digest = hashlib.sha256(payload).hexdigest()
    previous = INPUT_HASHES.setdefault(url, (200, digest))
    if previous != (200, digest):
        raise ValueError(f"same CLDR input URL returned differing content: {url}")
    return json.loads(payload)


def download(package: str, locale: str, file: str) -> dict:
    url = f"{BASE}/{package}/main/{locale}/{file}.json"
    return fetch_json(url)["main"][locale]


def rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def split_alternates(
    source: dict[str, str], *, standalone: bool = False, lowercase: bool = False
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    base: dict[str, str] = {}
    alternate: dict[tuple[str, str], str] = {}
    standalones: dict[str, str] = {}
    for raw_code, name in source.items():
        if "-alt-" in raw_code:
            code, alt = raw_code.rsplit("-alt-", 1)
            code = code.lower() if lowercase else code
            if alt == "stand-alone":
                standalones[code] = name
            elif alt in ("short", "narrow"):
                alternate[(alt, code)] = name
            continue
        code = raw_code.lower() if lowercase else raw_code
        base[code] = name
    if standalone:
        base.update(standalones)
    return base, alternate


def main() -> None:
    base_rows: list[tuple[str, str, str, str]] = []
    alt_rows: list[tuple[str, str, str, str, str]] = []
    patterns: list[tuple[str, str, str]] = []

    # CLDR uses old LDML calendar keys in display-name data. Translate them to
    # their canonical Unicode BCP-47 spellings and retain the well-formed
    # deprecated alias that ECMA-402 accepts as a distinct input code.
    calendar_json = fetch_json(f"{BASE}/cldr-bcp47/bcp47/calendar.json")[
        "keyword"
    ]["u"]["ca"]
    calendar_codes: dict[str, list[str]] = {}
    for code, record in calendar_json.items():
        if code.startswith("_"):
            continue
        source = record.get("_alias", code)
        calendar_codes.setdefault(source, []).append(code)
    calendar_codes.setdefault("islamic-civil", []).append("islamicc")

    for locale in LOCALES:
        localenames: dict[str, dict[str, str]] = {}
        for file, kind, standalone, lowercase in (
            ("languages", "language", False, False),
            ("territories", "region", False, False),
            ("scripts", "script", True, False),
            ("variants", "variant", False, True),
        ):
            try:
                data = download("cldr-localenames-full", locale, file)
            except HTTPError as error:
                # A few locales intentionally inherit an entire sparse category from root.
                # Root's values are codes, so omitting them has the same observable fallback.
                if error.code == 404:
                    continue
                raise
            values = data["localeDisplayNames"][file]
            names, alternates = split_alternates(
                values, standalone=standalone, lowercase=lowercase
            )
            localenames[kind] = names
            base_rows.extend((locale, kind, code, name) for code, name in names.items())
            alt_rows.extend(
                (locale, kind, style, code, name)
                for (style, code), name in alternates.items()
            )

        locale_data = download(
            "cldr-localenames-full", locale, "localeDisplayNames"
        )["localeDisplayNames"]
        locale_pattern = locale_data["localeDisplayPattern"]
        patterns.append(
            (
                locale,
                locale_pattern["localePattern"],
                locale_pattern["localeSeparator"],
            )
        )
        calendars = locale_data.get("types", {}).get("calendar", {})
        for raw_code, name in calendars.items():
            if raw_code == "core" or "-alt-" in raw_code:
                continue
            for code in calendar_codes.get(raw_code, [raw_code]):
                base_rows.append((locale, "calendar", code, name))

        currencies = download("cldr-numbers-full", locale, "currencies")[
            "numbers"
        ]["currencies"]
        for code, record in currencies.items():
            name = record.get("displayName")
            if name is not None:
                base_rows.append((locale, "currency", code, name))

        fields = download("cldr-dates-full", locale, "dateFields")["dates"][
            "fields"
        ]
        for code, cldr_code in DATE_FIELDS.items():
            for style, suffix in (("long", ""), ("short", "-short"), ("narrow", "-narrow")):
                name = fields.get(cldr_code + suffix, {}).get("displayName")
                if name is None:
                    continue
                if style == "long":
                    base_rows.append((locale, "dateTimeField", code, name))
                else:
                    alt_rows.append(
                        (locale, "dateTimeField", style, code, name)
                    )

    # A source row can intentionally be reached by more than one alias. Last
    # writer wins deterministically.
    base_rows = sorted({row[:3]: row for row in base_rows}.values())
    alt_rows = sorted({row[:4]: row for row in alt_rows}.values())

    # Each variant keeps only the names that differ from its CLDR parent's.
    def delta_rows(rows: list[tuple], key_length: int) -> list[tuple]:
        tables: dict[str, dict] = {locale: {} for locale in LOCALES}
        for row in rows:
            tables[row[0]][row[1 : 1 + key_length]] = row[1 + key_length :]
        kept = cldr_locales.delta(tables)
        return sorted(
            (locale, *key, *value) for locale, table in kept.items() for key, value in table.items()
        )

    base_rows = delta_rows(base_rows, 2)
    alt_rows = delta_rows(alt_rows, 3)
    patterns = delta_rows(patterns, 0)

    def ident(prefix: str, *parts: str) -> str:
        text = "_".join((prefix, *parts)).upper()
        return "".join(character if character.isalnum() else "_" for character in text)

    grouped_base: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for locale, kind, code, value in base_rows:
        grouped_base.setdefault((locale, kind), []).append((code, value))
    grouped_alt: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
    for locale, kind, style, code, value in alt_rows:
        grouped_alt.setdefault((locale, kind, style), []).append((code, value))

    strings = StringPool()
    for _, _, code, value in base_rows:
        strings.intern(code)
        strings.intern(value)
    for _, _, _, code, value in alt_rows:
        strings.intern(code)
        strings.intern(value)
    for _, pattern, separator in patterns:
        strings.intern(pattern)
        strings.intern(separator)
    default_pattern_id = strings.intern("{0} ({1})")
    default_separator_id = strings.intern("{0}, {1}")

    def string_id(value: str) -> int:
        return strings.refs[value][0]

    output = [
        "//! GENERATED by scripts/gen-cldr-display-names.py from CLDR 48. Do not edit.",
        "//! Locale names use partitioned base records, sparse overrides and a deduplicated UTF-8 pool.",
        "",
        'const CLDR_STRINGS: &str = include_str!("cldr_display_names.data");',
        "type TextId = u16;",
        "type Name = (TextId, TextId);",
        "",
        "#[rustfmt::skip]",
        f"static CLDR_REFS: [(u32, u32); {len(strings.refs)}] = [",
    ]
    output.extend(
        f"    ({offset}, {length}),"
        for _, offset, length in strings.refs.values()
    )
    output.extend([
        "];",
        "",
        "#[inline]",
        "fn text(id: TextId) -> &'static str {",
        "    let (offset, len) = CLDR_REFS[id as usize];",
        "    let start = offset as usize;",
        "    let end = start + len as usize;",
        "    &CLDR_STRINGS[start..end]",
        "}",
        "",
    ])
    for key, rows in grouped_base.items():
        output.extend([
            "#[rustfmt::skip]",
            f"static {ident('B', *key)}: [Name; {len(rows)}] = [",
        ])
        output.extend(
            f"    ({string_id(code)}, {string_id(value)})," for code, value in rows
        )
        output.extend(["];", ""])
    for key, rows in grouped_alt.items():
        output.extend([
            "#[rustfmt::skip]",
            f"static {ident('A', *key)}: [Name; {len(rows)}] = [",
        ])
        output.extend(
            f"    ({string_id(code)}, {string_id(value)})," for code, value in rows
        )
        output.extend(["];", ""])

    output.extend(
        [
            "fn lookup(rows: &[Name], code: &str) -> Option<&'static str> {",
            "    rows.binary_search_by(|row| text(row.0).cmp(code))",
            "        .ok()",
            "        .map(|index| text(rows[index].1))",
            "}",
            "",
            "/// A display name for a shipped locale: the requested style along the locale's parent",
            "/// chain, then (CLDR style fallback narrow → short → long) the wider ones.",
            "pub(crate) fn name(locale: &str, kind: &str, style: &str, code: &str) -> Option<&'static str> {",
            "    let chain = crate::cldr_locales::chain(locale);",
            "    for &candidate in match style {",
            '        "narrow" => &["narrow", "short"][..],',
            '        "short" => &["short"][..],',
            "        _ => &[][..],",
            "    } {",
            "        if let Some(value) = chain.iter().find_map(|lang| lookup(alternate_rows(lang, kind, candidate), code)) {",
            "            return Some(value);",
            "        }",
            "    }",
            "    chain.iter().find_map(|lang| lookup(base_rows(lang, kind), code))",
            "}",
            "",
            "fn alternate_rows(lang: &str, kind: &str, style: &str) -> &'static [Name] {",
            "    match (lang, kind, style) {",
        ]
    )
    output.extend(
        f"            ({rust_string(locale)}, {rust_string(kind)}, {rust_string(style)}) => &{ident('A', locale, kind, style)},"
        for locale, kind, style in grouped_alt
    )
    output.extend(
        [
            "            _ => &[],",
            "    }",
            "}",
            "",
            "fn base_rows(lang: &str, kind: &str) -> &'static [Name] {",
            "    match (lang, kind) {",
        ]
    )
    output.extend(
        f"        ({rust_string(locale)}, {rust_string(kind)}) => &{ident('B', locale, kind)},"
        for locale, kind in grouped_base
    )
    output.extend(
        [
            "        _ => &[],",
            "    }",
            "}",
            "",
            "/// The localeDisplayPattern (pattern, separator) of a shipped locale.",
            "pub(crate) fn locale_patterns(locale: &str) -> (&'static str, &'static str) {",
            "    crate::cldr_locales::chain(locale)",
            "        .iter()",
            "        .find_map(|lang| own_locale_patterns(lang))",
            f"        .unwrap_or((text({default_pattern_id}), text({default_separator_id})))",
            "}",
            "",
            "fn own_locale_patterns(lang: &str) -> Option<(&'static str, &'static str)> {",
            "    Some(match lang {",
        ]
    )
    output.extend(
        f"        {rust_string(locale)} => (text({string_id(pattern)}), text({string_id(separator)})),"
        for locale, pattern, separator in patterns
    )
    output.extend(
        [
            "        _ => return None,",
            "    })",
            "}",
            "",
            "#[cfg(test)]",
            "const GENERATED_TABLES: &[&[Name]] = &[",
        ]
    )
    output.extend(f"    &{ident('B', *key)}," for key in grouped_base)
    output.extend(f"    &{ident('A', *key)}," for key in grouped_alt)
    output.extend([
        "];",
        "",
        "#[cfg(test)]",
        '#[path = "cldr_display_names_packed_tests.rs"]',
        "mod packed_tests;",
        "",
    ])
    rust_source = "\n".join(output).encode("utf-8")
    blob = bytes(strings.data)
    OUTPUT.write_bytes(rust_source)
    BLOB_OUTPUT.write_bytes(blob)
    generator_hash = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "cldr_json_repository": REPOSITORY,
        "cldr_json_tag": TAG,
        "cldr_json_commit": SOURCE_COMMIT,
        "input_url_base": BASE,
        "distinct_input_count": len(INPUT_HASHES),
        "missing_input_count": sum(1 for status, _ in INPUT_HASHES.values() if status != 200),
        "inputs": [
            {
                "url": url,
                "status": INPUT_HASHES[url][0],
                **(
                    {"sha256": INPUT_HASHES[url][1]}
                    if INPUT_HASHES[url][1] is not None
                    else {}
                ),
            }
            for url in sorted(INPUT_HASHES)
        ],
        "generator": "scripts/gen-cldr-display-names.py",
        "generator_sha256": generator_hash,
        "original_generated_snapshot": {
            "commit": "4be60d028b95e50733f9449d9948345effc5f6c8",
            "git_blob_oid": "82392c1c1ddcc27cd7478076a73c9818b249a97c",
            "sha256": "5e070cb4625de55eb1f96cb4ed52169ea7f196f9150ac46770003718a9ee052b",
        },
        "packed_source_sha256": hashlib.sha256(rust_source).hexdigest(),
        "string_blob_path": "crates/trust-js/src/cldr_display_names.data",
        "string_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "string_blob_bytes": len(blob),
        "unique_string_count": len(strings.refs),
        "base_record_count": len(base_rows),
        "style_override_count": len(alt_rows),
        "text_reference_count": len(base_rows) * 2 + len(alt_rows) * 2 + len(patterns) * 2 + 2,
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {OUTPUT.relative_to(ROOT)} "
        f"({len(base_rows)} base names, {len(alt_rows)} style overrides)"
    )


if __name__ == "__main__":
    main()
