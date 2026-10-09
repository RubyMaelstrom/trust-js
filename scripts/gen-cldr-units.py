#!/usr/bin/env python3
"""Generate crates/lumen/src/cldr_units.rs from CLDR unit display patterns.

Source: unicode-org/cldr-json, package `cldr-units-full` (CLDR 48.0.0).
For each locale we read `<loc>/units.json` and emit every available plural-category unit pattern
for the ECMA-402 sanctioned single units and CLDR's precomposed compounds of them, across the
long/short/narrow styles, plus the `per` compound pattern and each single unit's `perUnitPattern`
that compose the remaining "X-per-Y" units.

The CLDR key for a sanctioned unit is category-prefixed (e.g. `duration-second`,
`length-centimeter`). The sanctioned id is the key with its leading category segment
removed, so we resolve each unit by matching `key.split('-', 1)[1] == unit` — matching
by bare suffix instead is what corrupted the previous table (`second` picking up
`acceleration-meter-per-square-second`, `centimeter` picking up `area-square-centimeter`).

Every CLDR locale of a supported language is generated (scripts/cldr_locales.py): a regional or
script variant keeps only the patterns that differ from its parent's, and each lookup walks the
locale's parent chain.

Usage:
    python3 scripts/gen-cldr-units.py                    # fetch from CDN, write the .rs file
    CLDR_DIR=/path/to/cldr-json python3 scripts/...      # read a local cldr-json tree
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cldr_locales  # noqa: E402

# Locales carried by the engine.
LOCALES = list(cldr_locales.locales())

STYLES = ["long", "short", "narrow"]
COUNTS = ["zero", "one", "two", "few", "many", "other"]

# ECMA-402 sanctioned single units, in table order (simple units alphabetical).
SIMPLE_UNITS = [
    "acre", "bit", "byte", "celsius", "centimeter", "day", "degree", "fahrenheit",
    "fluid-ounce", "foot", "gallon", "gigabit", "gigabyte", "gram", "hectare", "hour",
    "inch", "kilobit", "kilobyte", "kilogram", "kilometer", "liter", "megabit",
    "megabyte", "meter", "microsecond", "mile", "mile-scandinavian", "milliliter",
    "millimeter", "millisecond", "minute", "month", "nanosecond", "ounce", "percent",
    "petabyte", "pound", "second", "stone", "terabit", "terabyte", "week", "yard",
    "year",
]
# The "X-per-Y" units of two sanctioned single units that CLDR precomposes. Other such
# compounds are built at run time from the `per` compound pattern and per-unit patterns
# (UTS #35 Part 2, Compound Units).
UNITS = SIMPLE_UNITS + [
    "kilometer-per-hour", "meter-per-second", "liter-per-kilometer", "mile-per-gallon",
    "mile-per-hour",
]


def load_locale(loc):
    return cldr_locales.read_json(f"cldr-units-full/main/{loc}/units.json")["main"][loc]["units"]


def resolve_keys(long_section):
    """Map each sanctioned unit id to its category-prefixed CLDR key."""
    candidates = [k for k in long_section if "-" in k and not k[0].isdigit()]
    mapping = {}
    for unit in UNITS:
        matches = [k for k in candidates if k.split("-", 1)[1] == unit]
        if len(matches) != 1:
            raise SystemExit(f"unit {unit!r} resolved to {matches!r}, expected exactly one")
        mapping[unit] = matches[0]
    return mapping


def esc(s):
    rendered = s.replace("\\", "\\\\").replace('"', '\\"')
    for character in ("\u200b", "\u2060", "\ufeff"):
        rendered = rendered.replace(character, f"\\u{{{ord(character):x}}}")
    return rendered


def ident(locale, prefix="UNIT_"):
    return prefix + "".join(c.upper() if c.isalnum() else "_" for c in locale)


def main():
    by_locale = {}
    per_by_locale = {}
    for loc in LOCALES:
        rows = []
        per_rows = []
        units = load_locale(loc)
        keys = resolve_keys(units["long"])
        for style in STYLES:
            section = units[style]
            for unit in UNITS:
                entry = section.get(keys[unit])
                if not entry:
                    continue
                for count in COUNTS:
                    pat = entry.get(f"unitPattern-count-{count}")
                    if pat is None:
                        continue
                    rows.append((unit, style, count, pat))
            # The "per" compound pattern ("{0}/{1}") under the unit name "per", and each simple
            # unit's precomposed denominator pattern ("{0}/h").
            compound = section.get("per", {}).get("compoundUnitPattern")
            if compound is not None:
                per_rows.append(("per", style, compound))
            for unit in SIMPLE_UNITS:
                pattern = section.get(keys[unit], {}).get("perUnitPattern")
                if pattern is not None:
                    per_rows.append((unit, style, pattern))
        by_locale[loc] = {row[:3]: row[3] for row in rows}
        per_by_locale[loc] = {row[:2]: row[2] for row in per_rows}
    # Each variant keeps only the patterns that differ from its CLDR parent's.
    by_locale = {
        loc: sorted((*key, value) for key, value in table.items())
        for loc, table in cldr_locales.delta(by_locale).items()
    }
    per_by_locale = {
        loc: sorted((*key, value) for key, value in table.items())
        for loc, table in cldr_locales.delta(per_by_locale).items()
    }

    out = [
        "//! CLDR 48 unit display patterns, generated by scripts/gen-cldr-units.py. DO NOT EDIT.",
        "//! `{0}` is the number placeholder. A regional or script variant holds only the patterns",
        "//! that differ from its parent's; lookups walk the locale's parent chain.",
        "",
    ]
    for loc, rows in by_locale.items():
        if not rows:
            continue
        out.extend(["#[rustfmt::skip]", f"static {ident(loc)}: &[(&str, &str, &str, &str)] = &["])
        for unit, style, count, pat in rows:
            out.append(f'    ("{unit}", "{style}", "{count}", "{esc(pat)}"),')
        out.extend(["];", ""])
    for loc, rows in per_by_locale.items():
        if not rows:
            continue
        out.extend(["#[rustfmt::skip]", f"static {ident(loc, 'PER_')}: &[(&str, &str, &str)] = &["])
        for unit, style, pat in rows:
            out.append(f'    ("{unit}", "{style}", "{esc(pat)}"),')
        out.extend(["];", ""])
    out += [
        "fn lookup(table: &'static [(&str, &str, &str, &str)], unit: &str, style: &str, plural: &str) -> Option<&'static str> {",
        "    table",
        "        .binary_search_by(|(u, s, p, _)| (*u, *s, *p).cmp(&(unit, style, plural)))",
        "        .ok()",
        "        .map(|index| table[index].3)",
        "}",
        "",
        "/// The unit display pattern for a shipped locale, unit, style and plural category, along",
        "/// the locale's parent chain.",
        "pub fn unit_pattern(loc: &str, unit: &str, style: &str, plural: &str) -> Option<&'static str> {",
        "    crate::cldr_locales::chain(loc).iter().find_map(|locale| {",
        "        let table: &'static [(&str, &str, &str, &str)] = match *locale {",
    ]
    for loc in LOCALES:
        if by_locale[loc]:
            out.append(f'            "{loc}" => {ident(loc)},')
    out.extend([
        "            _ => return None,",
        "        };",
        "        lookup(table, unit, style, plural)",
        "    })",
        "}",
        "",
        "/// A shipped locale's `per` compound pattern (unit `\"per\"`, e.g. `{0}/{1}`) or a simple",
        "/// unit's `perUnitPattern` (e.g. `{0}/h`) for a style, along the locale's parent chain.",
        "pub fn per_pattern(loc: &str, unit: &str, style: &str) -> Option<&'static str> {",
        "    crate::cldr_locales::chain(loc).iter().find_map(|locale| {",
        "        let table: &'static [(&str, &str, &str)] = match *locale {",
    ])
    for loc in LOCALES:
        if per_by_locale[loc]:
            out.append(f'            "{loc}" => {ident(loc, "PER_")},')
    out.extend([
        "            _ => return None,",
        "        };",
        "        table",
        "            .binary_search_by(|(u, s, _)| (*u, *s).cmp(&(unit, style)))",
        "            .ok()",
        "            .map(|index| table[index].2)",
        "    })",
        "}",
        "",
    ])

    dest = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "crates", "lumen", "src", "cldr_units.rs",
    )
    with open(dest, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {sum(map(len, by_locale.values()))} rows to {dest}")


if __name__ == "__main__":
    main()
