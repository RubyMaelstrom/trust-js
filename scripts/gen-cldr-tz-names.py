#!/usr/bin/env python3
"""Generate the CLDR time zone display names used by Intl.DateTimeFormat's `timeZoneName`.

The source is CLDR 48 (cldr-json): every advertised DateTimeFormat locale's timeZoneNames
(UTS #35 Part 4 Time Zone Names: the hour, GMT, region and fallback formats, zone-specific
names and exemplar cities, and metazone names), the supplemental metazone data (UTS #35
Part 4 Metazones: each zone's `usesMetazone` intervals and the preferred zone of a metazone
in each territory), the primary zones, and the BCP 47 time zone aliases that map IANA
identifiers to CLDR's stable zone identifiers. For the regional locales, the territory names
that differ from their language's (which `cldr_display_names` already holds) are emitted too.

Usage:
    python3 scripts/gen-cldr-tz-names.py                # fetch from the CLDR CDN
    CLDR_DIR=/path/to/cldr-json python3 scripts/...     # read a local cldr-json tree
"""

from __future__ import annotations

import calendar
import datetime
import json
import os
import pathlib
import urllib.request


TAG = "48.0.0"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/lumen/src/cldr_tz_names.rs"
CLDR_JSON = f"https://raw.githubusercontent.com/unicode-org/cldr-json/{TAG}/cldr-json"
# The DateTimeFormat locales (scripts/gen-cldr-datetime-patterns.py).
LOCALES = [
    "en", "en-IN", "de", "fr", "es", "it", "pt", "pt-PT", "nl", "ja",
    "zh", "zh-Hant", "ko", "ru", "ar", "sr", "th", "gv", "sl", "pl",
    "si", "ln", "sv", "hi",
]
NAME_KEYS = [("long", "generic"), ("long", "standard"), ("long", "daylight"),
             ("short", "generic"), ("short", "standard"), ("short", "daylight")]
# UTS #35 Part 1 Inheritance: the no-inheritance marker stands for "no value".
NO_INHERITANCE = "∅∅∅"


def load(relative: str) -> dict:
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        with open(os.path.join(cldr_dir, relative), encoding="utf-8") as file:
            return json.load(file)
    request = urllib.request.Request(
        f"{CLDR_JSON}/{relative}", headers={"User-Agent": "Lumen CLDR time zone name generator"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def ident(prefix: str, locale: str) -> str:
    return prefix + "_" + "".join(c.upper() if c.isalnum() else "_" for c in locale)


def epoch(text: str) -> int:
    """A `usesMetazone` boundary ("1991-10-27 07:00", UTC) in epoch seconds."""
    moment = datetime.datetime.strptime(text, "%Y-%m-%d %H:%M")
    return calendar.timegm(moment.timetuple())


def offset_seconds(text: str) -> int:
    """A `usesMetazone` stdOffset/dstOffset ("+01", "-05:30") in seconds."""
    sign = -1 if text[0] == "-" else 1
    hours, _, minutes = text[1:].partition(":")
    return sign * (int(hours) * 3600 + int(minutes or 0) * 60)


def walk(node: dict, path: list[str], is_leaf):
    for key, value in node.items():
        if key.startswith("_"):
            continue
        if is_leaf(value):
            yield "/".join(path + [key]), value
        elif isinstance(value, dict):
            yield from walk(value, path + [key], is_leaf)


def zone_leaf(value) -> bool:
    return isinstance(value, dict) and any(
        key in value for key in ("long", "short", "exemplarCity", "_type")
    )


def names_of(entry: dict) -> tuple[str, ...]:
    values = []
    for width, kind in NAME_KEYS:
        value = entry.get(width, {}).get(kind, "")
        values.append("" if value == NO_INHERITANCE else value)
    return tuple(values)


def main() -> None:
    supplemental = load("cldr-core/supplemental/metaZones.json")["supplemental"]["metaZones"]
    uses = []
    for zone, intervals in walk(
        supplemental["metazoneInfo"]["timezone"], [], lambda value: isinstance(value, list)
    ):
        rows = []
        for item in intervals:
            use = item["usesMetazone"]
            offsets = None
            if "_stdOffset" in use:
                offsets = (offset_seconds(use["_stdOffset"]), offset_seconds(use["_dstOffset"]))
            rows.append((
                epoch(use["_from"]) if "_from" in use else None,
                epoch(use["_to"]) if "_to" in use else None,
                use["_mzone"],
                offsets,
            ))
        rows.sort(key=lambda row: row[0] if row[0] is not None else -(1 << 62))
        uses.append((zone, rows))
    uses.sort()
    references = sorted(
        (entry["mapZone"]["_other"], entry["mapZone"]["_territory"], entry["mapZone"]["_type"])
        for entry in supplemental["metazones"]
    )
    primary = sorted(
        load("cldr-core/supplemental/primaryZones.json")["supplemental"]["primaryZones"].items()
    )
    # UTS #35 Part 1 Time Zone Identifiers: the first alias of a bcp47/timezone.xml type is
    # CLDR's canonical (long) identifier; the rest are its aliases. A deprecated type's
    # identifiers resolve through its preferred type.
    types = load("cldr-bcp47/bcp47/timezone.json")["keyword"]["u"]["tz"]
    aliases: dict[str, str] = {}
    for key, value in types.items():
        if not isinstance(value, dict) or "_alias" not in value:
            continue
        target = value
        while target.get("_deprecated") and target.get("_preferred") in types:
            target = types[target["_preferred"]]
        if "_alias" not in target:
            continue
        canonical = target["_alias"].split()[0]
        for alias in value["_alias"].split():
            if alias != canonical:
                aliases[alias] = canonical

    pool = StringPool()
    lines = [
        "//! CLDR 48 time zone display names — GENERATED by scripts/gen-cldr-tz-names.py. DO NOT EDIT.",
        "//!",
        f"//! UTS #35 Part 4 Time Zone Names and Supplemental Time Zone Data, from cldr-json {TAG}:",
        "//! each locale's formats, zone and metazone names and exemplar cities; the metazone each",
        "//! zone uses over time; the preferred zone of each metazone per territory; the primary",
        "//! zones; and the BCP 47 aliases mapping IANA identifiers to CLDR's canonical zone",
        "//! identifiers. Strings live once in `cldr_tz_names.data`; tables hold their ids.",
        "",
        "const STRINGS: &str = include_str!(\"cldr_tz_names.data\");",
        "",
        "/// A string id; 0 is the empty string (no name).",
        "pub type Id = u16;",
        "",
        "/// A `usesMetazone` interval: `[from, to)` in epoch seconds (`i64::MIN`/`i64::MAX` when",
        "/// open), the metazone, and CLDR's explicit (standard, daylight) offsets in seconds, which",
        "/// decide daylight time where the TZ database's `tt_isdst` does not (Europe/Dublin).",
        "pub struct MetazoneUse {",
        "    pub from: i64,",
        "    pub to: i64,",
        "    pub metazone: Id,",
        "    pub offsets: Option<(i32, i32)>,",
        "}",
        "",
        "/// A locale's time zone formats (UTS #35 Part 4 timeZoneNames Elements Used for Fallback):",
        "/// hourFormat, gmtFormat, regionFormat, its standard and daylight forms, fallbackFormat.",
        "pub type Formats = [Id; 6];",
        "",
        "/// Long generic, standard and daylight names, then the short ones; 0 when CLDR has none.",
        "pub type Names = [Id; 6];",
        "",
        "/// One locale's names: formats, metazone names, zone-specific names and exemplar cities",
        "/// (each sorted by identifier), and territory names that differ from its language's.",
        "pub struct LocaleNames {",
        "    pub formats: Formats,",
        "    pub metazones: &'static [(Id, Names)],",
        "    pub zones: &'static [(Id, Names)],",
        "    pub cities: &'static [(Id, Id)],",
        "    pub regions: &'static [(Id, Id)],",
        "}",
        "",
        "/// The text of a string id.",
        "pub fn text(id: Id) -> &'static str {",
        "    let id = id as usize;",
        "    &STRINGS[OFFSETS[id] as usize..OFFSETS[id + 1] as usize]",
        "}",
        "",
    ]
    body = ["/// Each CLDR zone's metazone intervals, sorted by zone.",
            "pub static METAZONE_USES: &[(Id, &[MetazoneUse])] = &["]
    for zone, rows in uses:
        rendered = []
        for start, end, metazone, offsets in rows:
            rendered.append(
                "MetazoneUse { from: "
                + ("i64::MIN" if start is None else str(start))
                + ", to: "
                + ("i64::MAX" if end is None else str(end))
                + f", metazone: {pool.id(metazone)}, offsets: "
                + ("None" if offsets is None else f"Some(({offsets[0]}, {offsets[1]}))")
                + " }"
            )
        body.append(f"    ({pool.id(zone)}, &[{', '.join(rendered)}]),")
    body += [
        "];",
        "",
        "/// (metazone, territory, preferred zone), sorted; territory `001` names the golden zone.",
        "pub static REFERENCE_ZONES: &[(Id, Id, Id)] = &[",
    ]
    body += [f"    ({pool.id(m)}, {pool.id(t)}, {pool.id(z)})," for m, t, z in references]
    body += [
        "];",
        "",
        "/// (territory, primary zone), UTS #35 Part 4 Primary Zones, sorted.",
        "pub static PRIMARY_ZONES: &[(Id, Id)] = &[",
    ]
    body += [f"    ({pool.id(r)}, {pool.id(z)})," for r, z in primary]
    body += [
        "];",
        "",
        "/// (alias, CLDR canonical zone identifier), sorted by alias.",
        "pub static CLDR_IDS: &[(Id, Id)] = &[",
    ]
    body += [f"    ({pool.id(a)}, {pool.id(c)})," for a, c in sorted(aliases.items())]
    body.append("];")

    tables: dict[tuple, str] = {}
    table_lines: list[str] = []

    def table(prefix: str, locale: str, kind: str, rows: tuple) -> str:
        if not rows:
            return "&[]"
        if rows in tables:
            return tables[rows]
        name = ident(prefix, locale)
        if kind == "names":
            table_lines.append(f"static {name}: &[(Id, Names)] = &[")
            for key, values in rows:
                ids = ", ".join(str(pool.id(v)) for v in values)
                table_lines.append(f"    ({pool.id(key)}, [{ids}]),")
        else:
            table_lines.append(f"static {name}: &[(Id, Id)] = &[")
            for key, value in rows:
                table_lines.append(f"    ({pool.id(key)}, {pool.id(value)}),")
        table_lines.append("];")
        tables[rows] = name
        return name

    entries = []
    for locale in LOCALES:
        data = load(f"cldr-dates-full/main/{locale}/timeZoneNames.json")["main"][locale]
        names = data["dates"]["timeZoneNames"]
        metazones = tuple(sorted(
            (metazone, names_of(entry))
            for metazone, entry in names.get("metazone", {}).items()
            if any(names_of(entry))
        ))
        zone_rows = list(walk(names.get("zone", {}), [], zone_leaf))
        zones = tuple(sorted(
            (zone, names_of(entry)) for zone, entry in zone_rows if any(names_of(entry))
        ))
        cities = tuple(sorted(
            (zone, entry["exemplarCity"])
            for zone, entry in zone_rows
            if entry.get("exemplarCity") not in (None, NO_INHERITANCE)
        ))
        regions: tuple = ()
        language = locale.split("-")[0]
        if language != locale:
            own = load(f"cldr-localenames-full/main/{locale}/territories.json")
            base = load(f"cldr-localenames-full/main/{language}/territories.json")
            own = own["main"][locale]["localeDisplayNames"]["territories"]
            base = base["main"][language]["localeDisplayNames"]["territories"]
            regions = tuple(sorted(
                (code, name) for code, name in own.items()
                if "-alt-" not in code and base.get(code) != name
            ))
        formats = tuple(pool.id(names[key]) for key in (
            "hourFormat", "gmtFormat", "regionFormat", "regionFormat-type-standard",
            "regionFormat-type-daylight", "fallbackFormat",
        ))
        entries.append((
            locale,
            formats,
            table("MZ", locale, "names", metazones),
            table("ZN", locale, "names", zones),
            table("EC", locale, "pairs", cities),
            table("RG", locale, "pairs", regions),
        ))

    body.append("")
    body += table_lines
    body += [
        "",
        "/// The advertised DateTimeFormat locales' names, sorted by locale.",
        "pub static LOCALES: &[(&str, LocaleNames)] = &[",
    ]
    for locale, formats, metazones, zones, cities, regions in sorted(entries):
        body += [
            f"    ({rust_string(locale)}, LocaleNames {{",
            f"        formats: [{', '.join(map(str, formats))}],",
            f"        metazones: {metazones},",
            f"        zones: {zones},",
            f"        cities: {cities},",
            f"        regions: {regions},",
            "    }),",
        ]
    body += ["];", ""]
    offsets, blob = pool.layout()
    body += [
        "/// The byte offset of each string id's text in `STRINGS`, then the end of the last.",
        f"static OFFSETS: [u32; {len(offsets)}] = [",
    ]
    body += [
        "    " + ", ".join(map(str, offsets[k : k + 16])) + ","
        for k in range(0, len(offsets), 16)
    ]
    body += ["];", ""]
    OUTPUT.write_text("\n".join(lines + body), encoding="utf-8")
    OUTPUT.with_suffix(".data").write_text(blob, encoding="utf-8")
    print(f"{len(uses)} zones with metazones, {len(references)} reference zones, "
          f"{len(aliases)} aliases, {len(entries)} locales, {len(tables)} tables, "
          f"{len(offsets) - 1} strings")


class StringPool:
    """Interned strings; id 0 is the empty string. Ids are assigned in first-use order."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {"": 0}

    def id(self, value: str) -> int:
        if value not in self.ids:
            self.ids[value] = len(self.ids)
            if len(self.ids) > 1 << 16:
                raise SystemExit("more than 65536 distinct strings")
        return self.ids[value]

    def layout(self) -> tuple[list[int], str]:
        """Each id's byte offset in the concatenated text, then its total length."""
        offsets, offset = [], 0
        for value in self.ids:
            offsets.append(offset)
            offset += len(value.encode("utf-8"))
        return offsets + [offset], "".join(self.ids)


if __name__ == "__main__":
    main()
