#!/usr/bin/env python3
"""Generate localized calendar names from pinned CLDR JSON inputs.

The generated Rust tables refer into one deduplicated UTF-8 blob. The generator
also writes a provenance manifest containing the hash of each distinct input.

Every CLDR locale of a supported language is generated (scripts/cldr_locales.py): a regional or
script variant keeps only the names that differ from its parent's, and each lookup walks the
locale's parent chain (`crate::cldr_locales::chain`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cldr_locales  # noqa: E402


TAG = "48.0.0"
SOURCE_COMMIT = "4d06be52b51bb2f75688d0abe55c52a66afed790"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/trust-js/src/cldr_dates.rs"
BLOB_OUTPUT = ROOT / "crates/trust-js/src/cldr_dates.data"
MANIFEST_OUTPUT = ROOT / "crates/trust-js/src/cldr_dates.inputs.json"
RAW = f"https://raw.githubusercontent.com/unicode-org/cldr-json/{SOURCE_COMMIT}/cldr-json"
REPOSITORY = "https://github.com/unicode-org/cldr-json"
LOCALES = list(cldr_locales.locales())
# (TRust JS calendar id, CLDR package, file id, key inside dates.calendars)
CALENDARS = [
    ("gregory", "cldr-dates-full", "gregorian", "gregorian"),
    ("buddhist", "cldr-cal-buddhist-full", "buddhist", "buddhist"),
    ("chinese", "cldr-cal-chinese-full", "chinese", "chinese"),
    ("coptic", "cldr-cal-coptic-full", "coptic", "coptic"),
    ("dangi", "cldr-cal-dangi-full", "dangi", "dangi"),
    ("ethioaa", "cldr-cal-ethiopic-full", "ethiopic-amete-alem", "ethiopic-amete-alem"),
    ("ethiopic", "cldr-cal-ethiopic-full", "ethiopic", "ethiopic"),
    ("hebrew", "cldr-cal-hebrew-full", "hebrew", "hebrew"),
    ("indian", "cldr-cal-indian-full", "indian", "indian"),
    ("islamic", "cldr-cal-islamic-full", "islamic", "islamic"),
    ("islamic-civil", "cldr-cal-islamic-full", "islamic-civil", "islamic-civil"),
    ("islamic-tbla", "cldr-cal-islamic-full", "islamic-tbla", "islamic-tbla"),
    ("islamic-umalqura", "cldr-cal-islamic-full", "islamic-umalqura", "islamic-umalqura"),
    ("japanese", "cldr-cal-japanese-full", "japanese", "japanese"),
    ("persian", "cldr-cal-persian-full", "persian", "persian"),
    ("roc", "cldr-cal-roc-full", "roc", "roc"),
]
U16_MAX = (1 << 16) - 1
U32_MAX = (1 << 32) - 1


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
            raise ValueError("CLDR string count exceeds the u16 id format")
        encoded = value.encode("utf-8")
        offset = len(self.data)
        end = offset + len(encoded)
        if end > U32_MAX:
            raise ValueError("CLDR string blob exceeds the u32 offset format")
        self.data.extend(encoded)
        result = (string_id, offset, len(encoded))
        self.refs[value] = result
        return string_id


def ident(prefix: str, locale: str) -> str:
    return prefix + "_" + "".join(
        character.upper() if character.isalnum() else "_" for character in locale
    )


def download(
    task: tuple[str, tuple[str, str, str, str]],
) -> tuple[str, str, dict, str, str]:
    locale, (calendar, package, file_id, data_key) = task
    relative = f"{package}/main/{locale}/ca-{file_id}.json"
    url = f"{RAW}/{relative}"
    # CLDR_DIR names a local copy of the pinned cldr-json tree (`<package>/main/<locale>/...`);
    # the manifest still records the pinned URL and each input's hash.
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        payload = (pathlib.Path(cldr_dir) / relative).read_bytes()
    else:
        request = urllib.request.Request(
            url, headers={"User-Agent": "TRust JS CLDR calendar-name generator"}
        )
        with urllib.request.urlopen(request) as response:
            payload = response.read()
    root = json.loads(payload)["main"][locale]["dates"]["calendars"]
    return locale, calendar, root[data_key], url, hashlib.sha256(payload).hexdigest()


def rust_ref(value: str, strings: StringPool) -> str:
    return str(strings.intern(value))


def main() -> None:
    months: dict[str, dict] = {locale: {} for locale in LOCALES}
    eras: dict[str, dict] = {locale: {} for locale in LOCALES}
    tasks = [(locale, calendar) for locale in LOCALES for calendar in CALENDARS]
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(download, tasks))
    input_hashes: dict[str, str] = {}
    for locale, calendar, data, url, digest in results:
        old_digest = input_hashes.setdefault(url, digest)
        if old_digest != digest:
            raise ValueError(f"same input URL returned differing content: {url}")
        month_data = data.get("months", {}).get("format", {})
        standalone_data = data.get("months", {}).get("stand-alone", {})
        for width, cldr_width in (("long", "wide"), ("short", "abbreviated"), ("narrow", "narrow")):
            for month, value in month_data.get(cldr_width, {}).items():
                if not isinstance(value, str):
                    continue
                if month.isdigit():
                    number = int(month)
                elif month.endswith("-yeartype-leap"):
                    # The Hebrew leap-year name of month 7 (Adar II) is month 107.
                    number = 100 + int(month.split("-")[0])
                else:
                    continue
                months[locale][(calendar, width, number)] = value
                # UTS #35 `L` patterns use the stand-alone names (nominative "июль" beside the
                # format context's genitive "июля"), which default to the format names.
                standalone = standalone_data.get(cldr_width, {}).get(month)
                if not isinstance(standalone, str):
                    standalone = value
                months[locale][(calendar, f"standalone-{width}", number)] = standalone
        era_data = data.get("eras", {})
        for width, cldr_width in (("long", "eraNames"), ("short", "eraAbbr"), ("narrow", "eraNarrow")):
            for era, value in era_data.get(cldr_width, {}).items():
                if isinstance(value, str) and "-alt-" not in era:
                    eras[locale][(calendar, width, era)] = value

    # Each variant keeps only the names that differ from its CLDR parent's.
    month_delta = cldr_locales.delta(months)
    era_delta = cldr_locales.delta(eras)
    month_tables = {
        locale: sorted((*key, value) for key, value in month_delta[locale].items())
        for locale in LOCALES
    }
    era_tables = {
        locale: sorted((*key, value) for key, value in era_delta[locale].items())
        for locale in LOCALES
    }
    strings = StringPool()
    for locale in LOCALES:
        for calendar, width, month, value in month_tables[locale]:
            strings.intern(calendar)
            strings.intern(width)
            strings.intern(value)
        for calendar, width, era, value in era_tables[locale]:
            strings.intern(calendar)
            strings.intern(width)
            strings.intern(era)
            strings.intern(value)
    output = [
        "//! GENERATED by scripts/gen-cldr-dates.py from CLDR 48. Do not edit.",
        "",
        "const CLDR_STRINGS: &str = include_str!(\"cldr_dates.data\");",
        "",
        "type TextId = u16;",
        "type MonthRow = (TextId, TextId, u8, TextId);",
        "type EraRow = (TextId, TextId, TextId, TextId);",
        "",
        "#[rustfmt::skip]",
        f"static CLDR_REFS: [(u32, u32); {len(strings.refs)}] = [",
    ]
    for _, offset, length in strings.refs.values():
        output.append(f"    ({offset}, {length}),")
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

    total_month_rows = 0
    total_era_rows = 0
    for locale in LOCALES:
        month_rows = month_tables[locale]
        era_rows = era_tables[locale]
        total_month_rows += len(month_rows)
        total_era_rows += len(era_rows)
        output.extend([
            "#[rustfmt::skip]",
            f"static {ident('MO', locale)}: [MonthRow; {len(month_rows)}] = [",
        ])
        for calendar, width, month, value in month_rows:
            output.append(
                f"    ({rust_ref(calendar, strings)}, {rust_ref(width, strings)}, {month}, {rust_ref(value, strings)}),"
            )
        output.extend([
            "];",
            "#[rustfmt::skip]",
            f"static {ident('ER', locale)}: [EraRow; {len(era_rows)}] = [",
        ])
        for calendar, width, era, value in era_rows:
            output.append(
                f"    ({rust_ref(calendar, strings)}, {rust_ref(width, strings)}, {rust_ref(era, strings)}, {rust_ref(value, strings)}),"
            )
        output.extend(["];", ""])

    output.extend([
        "fn month_rows(locale: &str) -> &'static [MonthRow] {",
        "    match locale {",
    ])
    for locale in LOCALES:
        output.append(f'        "{locale}" => &{ident("MO", locale)},')
    output.extend(["        _ => &[],", "    }", "}", ""])
    output.extend([
        "fn era_rows(locale: &str) -> &'static [EraRow] {",
        "    match locale {",
    ])
    for locale in LOCALES:
        output.append(f'        "{locale}" => &{ident("ER", locale)},')
    output.extend(["        _ => &[],", "    }", "}", ""])
    output.extend([
        "/// The month name (1-based) for a locale/calendar/width along the locale's parent chain.",
        "pub fn month_name(locale: &str, calendar: &str, width: &str, month: u8) -> Option<&'static str> {",
        "    crate::cldr_locales::chain(locale).iter().find_map(|locale| {",
        "        let rows = month_rows(locale);",
        "        rows.binary_search_by(|row| (text(row.0), text(row.1), row.2).cmp(&(calendar, width, month)))",
        "            .ok().map(|index| text(rows[index].3))",
        "    })",
        "}",
        "",
        "/// The stand-alone month name (UTS #35 `L`) for a locale/calendar/width; CLDR's stand-alone",
        "/// names default to the format-context names.",
        "pub fn standalone_month_name(locale: &str, calendar: &str, width: &str, month: u8) -> Option<&'static str> {",
        "    let standalone = match width {",
        "        \"long\" => \"standalone-long\",",
        "        \"short\" => \"standalone-short\",",
        "        _ => \"standalone-narrow\",",
        "    };",
        "    month_name(locale, calendar, standalone, month).or_else(|| month_name(locale, calendar, width, month))",
        "}",
        "",
        "/// The era name for a locale/calendar/width/era-key along the locale's parent chain.",
        "pub fn era_name(locale: &str, calendar: &str, width: &str, era: &str) -> Option<&'static str> {",
        "    crate::cldr_locales::chain(locale).iter().find_map(|locale| {",
        "        let rows = era_rows(locale);",
        "        rows.binary_search_by(|row| (text(row.0), text(row.1), text(row.2)).cmp(&(calendar, width, era)))",
        "            .ok().map(|index| text(rows[index].3))",
        "    })",
        "}",
        "",
        "#[cfg(test)]",
        "const GENERATED_LOCALES: &[&str] = &[",
    ])
    for locale in LOCALES:
        output.append(f"    \"{locale}\",")
    output.extend([
        "];",
        "",
        "#[cfg(test)]",
        '#[path = "cldr_dates_packed_tests.rs"]',
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
        "input_url_base": RAW,
        "distinct_input_count": len(input_hashes),
        "inputs": [
            {"url": url, "sha256": input_hashes[url]}
            for url in sorted(input_hashes)
        ],
        "generator": "scripts/gen-cldr-dates.py",
        "generator_sha256": generator_hash,
        "original_generated_snapshot": {
            "commit": "1084161475e2e35b50a73eb06fd6885e66de9239",
            "git_blob_oid": "4bb2e529bc2d12adbe120cf86feaa6648d84c371",
            "sha256": "7a0198abceba5fe084e7747fffcbb9525e6158b0bf2ec00a87e6e019c218e53b",
        },
        "packed_source_sha256": hashlib.sha256(rust_source).hexdigest(),
        "string_blob_path": "crates/trust-js/src/cldr_dates.data",
        "string_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "string_blob_bytes": len(blob),
        "unique_string_count": len(strings.refs),
        "month_record_count": total_month_rows,
        "era_record_count": total_era_rows,
        "text_reference_count": total_month_rows * 3 + total_era_rows * 4,
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
