#!/usr/bin/env python3
"""Generate DateTimeFormat patterns and field names for every advertised Lumen locale.

The source is CLDR 48 (cldr-json, which resolves UTS #35 inheritance, including each
non-Gregorian calendar's aliases to the generic or Chinese calendar). cldr-json does not publish
the ISO 8601 calendar, so its data comes from the CLDR 48 XML (common/main), resolved here by UTS
#35 Part 1 inheritance: the locale's parent chain to root, inheritance markers, draft status and
aliases (the ISO 8601 calendar's names, appendItems and missing interval formats are the
Gregorian calendar's). For every locale and
calendar the generator emits the date style patterns, the date-time connectors, the
availableFormats skeletons, the appendItems patterns and the intervalFormats; tables shared by
several calendars or locales are emitted once. Time patterns, weekday and day-period names come
from the Gregorian calendar, which every other calendar aliases for them. It also emits the
Chinese and Dangi cyclic year names and their leap-month patterns.

Usage:
    python3 scripts/gen-cldr-datetime-patterns.py            # fetch from the CLDR CDN
    CLDR_DIR=/path/to/cldr-json python3 scripts/...           # read a local cldr-json tree
    CLDR_XML_DIR=/path/to/cldr python3 scripts/...            # read a local CLDR 48 checkout
                                                              # (common/main/*.xml)
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import urllib.request
import sys
import xml.etree.ElementTree as ElementTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cldr_locales  # noqa: E402


TAG = "48.0.0"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/lumen/src/cldr_datetime_patterns.rs"
CLDR_JSON = f"https://raw.githubusercontent.com/unicode-org/cldr-json/{TAG}/cldr-json"
CLDR_XML = "https://raw.githubusercontent.com/unicode-org/cldr/release-48"
LOCALES = list(cldr_locales.locales())
# (Lumen calendar id, cldr-json package, calendar key)
CALENDARS = [
    ("gregory", "cldr-dates-full", "gregorian"),
    ("iso8601", None, "iso8601"),
    ("buddhist", "cldr-cal-buddhist-full", "buddhist"),
    ("chinese", "cldr-cal-chinese-full", "chinese"),
    ("coptic", "cldr-cal-coptic-full", "coptic"),
    ("dangi", "cldr-cal-dangi-full", "dangi"),
    ("ethioaa", "cldr-cal-ethiopic-full", "ethiopic-amete-alem"),
    ("ethiopic", "cldr-cal-ethiopic-full", "ethiopic"),
    ("hebrew", "cldr-cal-hebrew-full", "hebrew"),
    ("indian", "cldr-cal-indian-full", "indian"),
    ("islamic-civil", "cldr-cal-islamic-full", "islamic-civil"),
    ("islamic-tbla", "cldr-cal-islamic-full", "islamic-tbla"),
    ("islamic-umalqura", "cldr-cal-islamic-full", "islamic-umalqura"),
    ("japanese", "cldr-cal-japanese-full", "japanese"),
    ("persian", "cldr-cal-persian-full", "persian"),
    ("roc", "cldr-cal-roc-full", "roc"),
]
CYCLIC_CALENDARS = ["chinese", "dangi"]
STYLE_NAMES = ["full", "long", "medium", "short"]
WEEKDAY_KEYS = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
WIDTHS = (("long", "wide"), ("short", "abbreviated"), ("narrow", "narrow"))


def load(package: str, locale: str, calendar: str) -> dict:
    """One locale's resolved CLDR data for a calendar, from `CLDR_DIR` (a local cldr-json tree
    laid out as `<package>/main/<locale>/ca-<calendar>.json`) or the CLDR CDN."""
    relative = f"{package}/main/{locale}/ca-{calendar}.json"
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        with open(os.path.join(cldr_dir, relative), encoding="utf-8") as file:
            data = json.load(file)
    else:
        request = urllib.request.Request(
            f"{CLDR_JSON}/{relative}",
            headers={"User-Agent": "Lumen CLDR DateTimeFormat generator"},
        )
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
    return data["main"][locale]["dates"]["calendars"][calendar]


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Lumen CLDR DateTimeFormat generator"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


@functools.cache
def parent_locales() -> dict[str, str]:
    relative = "cldr-core/supplemental/parentLocales.json"
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir and os.path.exists(os.path.join(cldr_dir, relative)):
        with open(os.path.join(cldr_dir, relative), encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = json.loads(fetch_text(f"{CLDR_JSON}/{relative}"))
    return data["supplemental"]["parentLocales"]["parentLocale"]


def locale_chain(locale: str) -> list[str]:
    """UTS #35 Part 1 Parent Locales: explicit parents, else truncation, ending at root."""
    chain = [locale]
    while chain[-1] != "root":
        current = chain[-1]
        parent = parent_locales().get(current)
        if parent is None:
            parent = current.rsplit("-", 1)[0] if "-" in current else "root"
        chain.append("root" if parent == "und" else parent)
    return chain


# Path segments: an element name with its distinguishing attributes.
DISTINGUISHING = ("type", "id", "request", "count", "alt")
INHERITANCE_MARKER = "\u2191\u2191\u2191"
RANGES = "intervalFormatItem*"


@functools.cache
def xml_calendars(locale: str) -> tuple[dict, dict]:
    """A locale file's calendar data as ({path: value}, {path: alias target path}); paths start
    with the calendar segment. Unconfirmed and provisional data is excluded, as in cldr-json."""
    name = locale.replace("-", "_")
    xml_dir = os.environ.get("CLDR_XML_DIR")
    if xml_dir:
        with open(os.path.join(xml_dir, "common", "main", f"{name}.xml"), encoding="utf-8") as file:
            text = file.read()
    else:
        text = fetch_text(f"{CLDR_XML}/common/main/{name}.xml")
    values: dict[tuple, object] = {}
    aliases: dict[tuple, tuple] = {}
    calendars = ElementTree.fromstring(text).find("dates/calendars")
    if calendars is None:
        return values, aliases

    def segment(element) -> str:
        keys = [f"{key}={element.get(key)}" for key in DISTINGUISHING if element.get(key)]
        return element.tag + ("[" + ",".join(keys) + "]" if keys else "")

    def walk(element, path: tuple) -> None:
        if element.get("draft") in ("unconfirmed", "provisional"):
            return
        children = list(element)
        alias = element.find("alias")
        if alias is not None:
            target = list(path)
            for step in alias.get("path").split("/"):
                if step == "..":
                    target.pop()
                else:
                    # calendar[@type='gregorian'] -> calendar[type=gregorian]
                    step = step.replace("[@", "[").replace("='", "=").replace("']", "]")
                    target.append(step)
            if path[-1] == "intervalFormatRanges":
                # The intervalFormatItems a calendar lacks are those of the alias target's
                # intervalFormats (intervalFormatRanges groups them; it is no element of its own).
                aliases[path[:-1] + (RANGES,)] = tuple(target[:-1])
            else:
                aliases[path] = tuple(target)
            return
        if not children:
            if element.text is not None and element.text.strip():
                numbers = element.get("numbers")
                values[path] = (element.text, numbers) if numbers else element.text
            return
        for child in children:
            walk(child, path + (segment(child),))

    for calendar in calendars:
        walk(calendar, (segment(calendar),))
    return values, aliases


def xml_resolve(locale: str, path: tuple, depth: int = 0):
    """UTS #35 Part 1 Lookup: the first value along the parent chain (an inheritance marker
    defers to the parent); with none, the nearest aliased ancestor path, looked up afresh."""
    chain = locale_chain(locale)
    for name in chain:
        value = xml_calendars(name)[0].get(path)
        if value is not None and value != INHERITANCE_MARKER:
            return value
    if depth > 8:
        return None
    for length in range(len(path) - 1, 0, -1):
        for name in chain:
            aliases = xml_calendars(name)[1]
            target = aliases.get(path[:length])
            if target is not None:
                return xml_resolve(locale, target + path[length:], depth + 1)
            if path[length].startswith("intervalFormatItem"):
                target = aliases.get(path[:length] + (RANGES,))
                if target is not None:
                    return xml_resolve(locale, target + path[length:], depth + 1)
    return None


def xml_children(locale: str, path: tuple, depth: int = 0) -> set[tuple]:
    """The resolved child paths below `path`: those of the chain and of any alias target."""
    found = set()
    for name in locale_chain(locale):
        values, aliases = xml_calendars(name)
        for key in list(values) + list(aliases):
            if len(key) > len(path) and key[: len(path)] == path:
                found.add(key[: len(path) + 1])
        if depth >= 8:
            continue
        for key, target in aliases.items():
            if key == path:
                for child in xml_children(locale, target, depth + 1):
                    found.add(path + child[len(target):])
            elif key == path + (RANGES,):
                for child in xml_children(locale, target, depth + 1):
                    if child[-1].startswith("intervalFormatItem"):
                        found.add(path + child[len(target):])
    found.discard(path + (RANGES,))
    return found


def attribute(segment: str, key: str) -> str | None:
    if "[" not in segment:
        return None
    for pair in segment[segment.index("[") + 1 : -1].split(","):
        name, _, value = pair.partition("=")
        if name == key:
            return value
    return None


def xml_calendar(locale: str, calendar: str) -> dict:
    """One locale's resolved CLDR XML data for a calendar, in cldr-json's shape."""
    base = (f"calendar[type={calendar}]",)

    def value(*segments):
        return xml_resolve(locale, base + segments)

    def as_json(found):
        if isinstance(found, tuple):
            return {"_value": found[0], "_numbers": found[1]}
        return found

    data: dict = {"dateFormats": {}, "timeFormats": {}, "dateTimeFormats": {}}
    at_time: dict = {}
    for style in STYLE_NAMES:
        data["dateFormats"][style] = as_json(value(
            "dateFormats", f"dateFormatLength[type={style}]", "dateFormat", "pattern"))
        data["timeFormats"][style] = as_json(value(
            "timeFormats", f"timeFormatLength[type={style}]", "timeFormat", "pattern"))
        data["dateTimeFormats"][style] = value(
            "dateTimeFormats", f"dateTimeFormatLength[type={style}]", "dateTimeFormat", "pattern")
        at_time[style] = value(
            "dateTimeFormats", f"dateTimeFormatLength[type={style}]",
            "dateTimeFormat[type=atTime]", "pattern") or data["dateTimeFormats"][style]
    data["dateTimeFormats-atTime"] = {"standard": at_time}
    available = {}
    for child in xml_children(locale, base + ("dateTimeFormats", "availableFormats")):
        key = attribute(child[-1], "id")
        if key and not attribute(child[-1], "count") and not attribute(child[-1], "alt"):
            found = xml_resolve(locale, child)
            if found is not None:
                available[key] = as_json(found)
    data["dateTimeFormats"]["availableFormats"] = available
    append = {}
    for child in xml_children(locale, base + ("dateTimeFormats", "appendItems")):
        key = attribute(child[-1], "request")
        found = xml_resolve(locale, child)
        if key and found is not None:
            append[key] = found
    data["dateTimeFormats"]["appendItems"] = append
    intervals = {"intervalFormatFallback": value(
        "dateTimeFormats", "intervalFormats", "intervalFormatFallback")}
    for item in xml_children(locale, base + ("dateTimeFormats", "intervalFormats")):
        skeleton = attribute(item[-1], "id")
        if not item[-1].startswith("intervalFormatItem") or not skeleton:
            continue
        for difference in xml_children(locale, item):
            key = attribute(difference[-1], "id")
            found = xml_resolve(locale, difference)
            if key and found is not None and not attribute(difference[-1], "alt"):
                intervals.setdefault(skeleton, {})[key] = found
    data["dateTimeFormats"]["intervalFormats"] = intervals
    return data


def rust_string(value: str) -> str:
    rendered = json.dumps(value, ensure_ascii=False)
    for character in ("​", "⁠", "﻿"):
        rendered = rendered.replace(character, f"\\u{{{ord(character):x}}}")
    return rendered


def ident(prefix: str, locale: str) -> str:
    return prefix + "_" + "".join(
        character.upper() if character.isalnum() else "_" for character in locale
    )


def pattern_and_numbers(value) -> tuple[str, str]:
    """A CLDR pattern and its UTS #35 `numbers` override (`hanidec`, `d=hanidays`, …)."""
    if isinstance(value, dict):
        return value["_value"], value.get("_numbers", "")
    return value, ""


def ordinary_patterns(values: dict) -> list[tuple[str, str, str]]:
    rows = []
    for key, value in values.items():
        if "-alt-" in key or "-count-" in key:
            continue
        if not isinstance(value, (str, dict)):
            continue
        pattern, numbers = pattern_and_numbers(value)
        rows.append((key, pattern, numbers))
    return sorted(rows)


def load_day_period_rules() -> dict[str, list[tuple[str, int, int]]]:
    """CLDR supplemental dayPeriodRuleSet (UTS #35 Part 4 Day Period Rules) for each locale's
    language: (period, from hour, before hour); `at` periods (noon, midnight) are (period, h, h)."""
    relative = "cldr-core/supplemental/dayPeriods.json"
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        with open(os.path.join(cldr_dir, relative), encoding="utf-8") as file:
            data = json.load(file)
    else:
        with urllib.request.urlopen(f"{CLDR_JSON}/{relative}") as response:
            data = json.load(response)
    sets = data["supplemental"]["dayPeriodRuleSet"]
    rules = {}
    for language in sorted(key for key in sets if key.split("-")[0] in cldr_locales.LANGUAGES):
        rows = []
        for period, rule in sets.get(language, {}).items():
            hour = lambda text: int(text.split(":")[0])
            if "_at" in rule:
                rows.append((period, hour(rule["_at"]), hour(rule["_at"])))
            else:
                rows.append((period, hour(rule["_from"]), hour(rule["_before"])))
        rules[language] = sorted(rows)
    return rules


def default_hour_cycle(pattern: str) -> str:
    quoted = False
    for character in pattern:
        if character == "'":
            quoted = not quoted
        elif not quoted and character in "hHKk":
            return {"h": "h12", "K": "h11", "H": "h23", "k": "h24"}[character]
    return "h23"


class Tables:
    """Emit each distinct table once, naming it by first use."""

    def __init__(self, prefix: str, row_type: str):
        self.prefix = prefix
        self.row_type = row_type
        self.names: dict[tuple, str] = {}
        self.order: list[tuple[str, tuple]] = []

    def name(self, rows: tuple) -> str:
        if rows not in self.names:
            name = f"{self.prefix}_{len(self.names)}"
            self.names[rows] = name
            self.order.append((name, rows))
        return self.names[rows]

    def emit(self, output: list[str]) -> None:
        for name, rows in self.order:
            output.extend(["#[rustfmt::skip]", f"static {name}: &[{self.row_type}] = &["])
            for row in rows:
                if isinstance(row, tuple):
                    output.append("    (" + ", ".join(map(rust_string, row)) + "),")
                else:
                    output.append(f"    {rust_string(row)},")
            output.extend(["];", ""])


def main() -> None:
    styles: dict[str, tuple] = {}
    weekdays: dict[str, dict] = {}
    standalone_weekdays: dict[str, dict] = {}
    day_periods: dict[str, dict] = {}
    records: dict[str, dict] = {}
    available_rows: dict[str, dict] = {}
    interval_rows: dict[str, dict] = {}
    append_rows: dict[str, dict] = {}
    cyclic_rows: dict[str, dict] = {}
    leap_rows: dict[str, dict] = {}

    for locale in LOCALES:
        gregorian = load("cldr-dates-full", locale, "gregorian")
        time = [gregorian["timeFormats"][style] for style in STYLE_NAMES]
        date_time = gregorian["dateTimeFormats"]
        times = tuple(pattern_and_numbers(value)[0] for value in time)
        styles[locale] = {(): (
            times,
            default_hour_cycle(times[3]),
            pattern_and_numbers(date_time["medium"])[0],
        )}

        format_days = gregorian["days"]["format"]
        standalone_days = gregorian["days"]["stand-alone"]
        weekdays[locale] = {}
        standalone_weekdays[locale] = {}
        for width, key in WIDTHS:
            for day in WEEKDAY_KEYS:
                weekdays[locale][(width, day)] = format_days[key][day]
                # The effective stand-alone name: CLDR's, or the format name it defaults to.
                standalone_weekdays[locale][(width, day)] = (
                    standalone_days.get(key, {}).get(day) or format_days[key][day]
                )

        day_periods[locale] = {}
        periods = gregorian["dayPeriods"]["format"]
        for width, key in WIDTHS:
            for name, value in periods[key].items():
                if isinstance(value, str) and "-alt-" not in name:
                    day_periods[locale][(width, name)] = value

        records[locale] = {}
        available_rows[locale] = {}
        interval_rows[locale] = {}
        append_rows[locale] = {}
        cyclic_rows[locale] = {}
        leap_rows[locale] = {}
        for calendar, package, key in CALENDARS:
            if calendar == "gregory":
                data = gregorian
            elif package is None:
                data = xml_calendar(locale, key)
            else:
                data = load(package, locale, key)
            dates = tuple(pattern_and_numbers(data["dateFormats"][style]) for style in STYLE_NAMES)
            calendar_date_time = data["dateTimeFormats"]
            calendar_at_time = data.get("dateTimeFormats-atTime", {}).get(
                "standard", calendar_date_time
            )
            connectors = tuple(
                pattern_and_numbers(calendar_at_time[style])[0] for style in STYLE_NAMES
            )
            interval_data = calendar_date_time["intervalFormats"]
            records[locale][(calendar,)] = (
                dates,
                connectors,
                pattern_and_numbers(interval_data["intervalFormatFallback"])[0],
            )
            for skeleton, pattern, numbers in ordinary_patterns(calendar_date_time["availableFormats"]):
                available_rows[locale][(calendar, skeleton)] = (pattern, numbers)
            for skeleton, by_difference in interval_data.items():
                if skeleton == "intervalFormatFallback" or not isinstance(by_difference, dict):
                    continue
                for difference, value in by_difference.items():
                    if isinstance(value, (str, dict)):
                        interval_rows[locale][(calendar, skeleton, difference)] = (
                            pattern_and_numbers(value)[0]
                        )
            for field, pattern in calendar_date_time.get("appendItems", {}).items():
                append_rows[locale][(calendar, field)] = pattern

            if calendar in CYCLIC_CALENDARS:
                years = data.get("cyclicNameSets", {}).get("years", {}).get("format", {})
                for width, cldr_width in WIDTHS:
                    names = years.get(cldr_width)
                    if names:
                        cyclic_rows[locale][(calendar, width)] = tuple(
                            names[str(index)] for index in range(1, 61)
                        )
                month_patterns = data.get("monthPatterns", {})
                for context in ("format", "stand-alone"):
                    for width, cldr_width in WIDTHS:
                        leap = month_patterns.get(context, {}).get(cldr_width, {}).get("leap")
                        if leap:
                            leap_rows[locale][(calendar, f"{context}-{width}")] = leap
                numeric = month_patterns.get("numeric", {}).get("all", {}).get("leap")
                if numeric:
                    leap_rows[locale][(calendar, "numeric")] = numeric

    # Each variant keeps only what differs from its CLDR parent (scripts/cldr_locales.py).
    styles = cldr_locales.delta(styles)
    weekdays = cldr_locales.delta(weekdays)
    standalone_weekdays = cldr_locales.delta(standalone_weekdays)
    day_periods = cldr_locales.delta(day_periods)
    records = cldr_locales.delta(records)
    available_rows = cldr_locales.delta(available_rows)
    interval_rows = cldr_locales.delta(interval_rows)
    append_rows = cldr_locales.delta(append_rows)
    cyclic_rows = cldr_locales.delta(cyclic_rows)
    leap_rows = cldr_locales.delta(leap_rows)
    day_period_rules = load_day_period_rules()

    def flat(tables: dict[str, dict]) -> list[tuple]:
        return sorted(
            (locale, *key, *(value if isinstance(value, tuple) else (value,)))
            for locale, table in tables.items()
            for key, value in table.items()
        )

    cyclic_names = Tables("CY", "&str")
    pool: dict[str, int] = {}

    def sid(value: str) -> str:
        """A string's id in the pool (CLDR_STRINGS / OFFSETS)."""
        if value not in pool:
            pool[value] = len(pool)
            if len(pool) > 1 << 16:
                raise SystemExit("more than 65536 distinct strings")
        return str(pool[value])

    output = [
        "//! GENERATED by scripts/gen-cldr-datetime-patterns.py from CLDR 48. Do not edit.",
        "//!",
        "//! Every shipped CLDR locale ([`crate::cldr_locales`]); a regional or script variant holds",
        "//! only what differs from its parent, and each lookup walks the locale's parent chain.",
        "",
        "use crate::cldr_locales::chain;",
        "",
        "const STRINGS: &str = include_str!(\"cldr_datetime_patterns.data\");",
        "",
        "/// A string id: its text is [`text`].",
        "type Id = u16;",
        "",
        "fn text(id: Id) -> &'static str {",
        "    let id = id as usize;",
        "    &STRINGS[OFFSETS[id] as usize..OFFSETS[id + 1] as usize]",
        "}",
        "",
        "#[derive(Clone, Copy)]",
        "struct StyleIds {",
        "    times: [Id; 4],",
        "    hour_cycle: Id,",
        "    glue: Id,",
        "}",
        "",
        "/// One locale's patterns for one calendar (UTS #35 Part 4): the date styles with their",
        "/// `numbers` overrides, the at-time date-time connectors and the interval fallback.",
        "pub(crate) struct CalendarPatterns {",
        "    pub dates: [(&'static str, &'static str); 4],",
        "    pub connectors: [&'static str; 4],",
        "    pub fallback: &'static str,",
        "}",
        "",
        "#[derive(Clone, Copy)]",
        "struct CalendarIds {",
        "    dates: [(Id, Id); 4],",
        "    connectors: [Id; 4],",
        "    fallback: Id,",
        "}",
        "",
    ]
    style_rows = []
    for locale, table in sorted(styles.items()):
        for _, (times, hour_cycle, glue) in table.items():
            style_rows.append((locale, times, hour_cycle, glue))
    record_rows = []
    for locale, table in sorted(records.items()):
        for (calendar,), record in sorted(table.items()):
            record_rows.append((locale, calendar, record))
    cyclic_flat = []
    for locale, table in sorted(cyclic_rows.items()):
        for (calendar, width), names in sorted(table.items()):
            cyclic_flat.append((locale, calendar, width, cyclic_names.name(names)))
    cyclic_names.emit(output)

    output.extend(["#[rustfmt::skip]", "static STYLES: &[(Id, StyleIds)] = &["])
    for locale, times, hour_cycle, glue in style_rows:
        output.append(
            f"    ({sid(locale)}, StyleIds {{ times: [{', '.join(map(sid, times))}], "
            f"hour_cycle: {sid(hour_cycle)}, glue: {sid(glue)} }}),"
        )
    output.extend(["];", "", "#[rustfmt::skip]", "static CALENDARS: &[(Id, Id, CalendarIds)] = &["])
    for locale, calendar, (dates, connectors, fallback) in record_rows:
        output.append(
            f"    ({sid(locale)}, {sid(calendar)}, CalendarIds {{ dates: ["
            + ", ".join(f"({sid(p)}, {sid(n)})" for p, n in dates)
            + "], connectors: ["
            + ", ".join(map(sid, connectors))
            + f"], fallback: {sid(fallback)} }}),"
        )
    output.append("];")
    tables = [
        ("WEEKDAYS", 4, flat(weekdays)),
        ("STANDALONE_WEEKDAYS", 4, flat(standalone_weekdays)),
        ("DAY_PERIODS", 4, flat(day_periods)),
        ("AVAILABLE", 5, flat(available_rows)),
        ("INTERVALS", 5, flat(interval_rows)),
        ("APPEND", 4, flat(append_rows)),
        ("LEAP", 4, flat(leap_rows)),
    ]
    for name, arity, rows in tables:
        output.extend(["", "#[rustfmt::skip]", f"static {name}: &[[Id; {arity}]] = &["])
        for row in rows:
            output.append("    [" + ", ".join(map(sid, row)) + "],")
        output.append("];")
    output.extend(["", "#[rustfmt::skip]", "static CYCLIC: &[(Id, Id, Id, &[&str])] = &["])
    for locale, calendar, width, names in cyclic_flat:
        output.append(f"    ({sid(locale)}, {sid(calendar)}, {sid(width)}, {names}),")
    output.append("];")

    output.extend(
        [
            "",
            "/// The first value along a locale's parent chain.",
            "fn inherited<T>(locale: &str, mut find: impl FnMut(&str) -> Option<T>) -> Option<T> {",
            "    chain(locale).iter().find_map(|locale| find(locale))",
            "}",
            "",
            "fn styles(locale: &str) -> StyleIds {",
            "    let find = |locale: &str| {",
            "        STYLES.binary_search_by(|row| text(row.0).cmp(locale)).ok().map(|index| STYLES[index].1)",
            "    };",
            "    inherited(locale, find).or_else(|| find(\"en\")).expect(\"generated English styles\")",
            "}",
            "",
            "fn style_index(style: &str) -> usize {",
            '    match style { "full" => 0, "long" => 1, "medium" => 2, _ => 3 }',
            "}",
            "",
            "/// A locale's (Gregorian) time style pattern; every calendar uses the Gregorian times.",
            "pub(crate) fn time_style(locale: &str, style: &str) -> &'static str {",
            "    text(styles(locale).times[style_index(style)])",
            "}",
            "",
            "pub(crate) fn default_hour_cycle(locale: &str) -> &'static str { text(styles(locale).hour_cycle) }",
            "",
            "/// The Gregorian medium dateTimeFormat (the standard, not at-time, form): the default",
            "/// combination of a date with a time, as interval formats use it.",
            "pub(crate) fn date_time_glue(locale: &str) -> &'static str { text(styles(locale).glue) }",
            "",
            "/// The patterns of `calendar` (a BCP 47 calendar id; others use the Gregorian data) for a",
            "/// shipped locale.",
            "pub(crate) fn calendar_patterns(locale: &str, calendar: &str) -> CalendarPatterns {",
            "    let find = |locale: &str, calendar: &str| {",
            "        CALENDARS",
            "            .binary_search_by(|row| text(row.0).cmp(locale).then_with(|| text(row.1).cmp(calendar)))",
            "            .ok()",
            "            .map(|index| CALENDARS[index].2)",
            "    };",
            "    let ids = inherited(locale, |locale| find(locale, calendar))",
            "        .or_else(|| inherited(locale, |locale| find(locale, \"gregory\")))",
            "        .or_else(|| find(\"en\", \"gregory\"))",
            "        .expect(\"generated English Gregorian patterns\");",
            "    CalendarPatterns {",
            "        dates: ids.dates.map(|(pattern, numbers)| (text(pattern), text(numbers))),",
            "        connectors: ids.connectors.map(text),",
            "        fallback: text(ids.fallback),",
            "    }",
            "}",
            "",
            "pub(crate) fn date_style(locale: &str, calendar: &str, style: &str) -> (&'static str, &'static str) {",
            "    calendar_patterns(locale, calendar).dates[style_index(style)]",
            "}",
            "",
            "pub(crate) fn connector(locale: &str, calendar: &str, style: &str) -> &'static str {",
            "    calendar_patterns(locale, calendar).connectors[style_index(style)]",
            "}",
            "",
            "/// The rows of `table` (locale and calendar first, then `key_len` key ids) along a",
            "/// locale's chain for a calendar, the nearest locale's winning per key.",
            "fn merged<const N: usize>(table: &'static [[Id; N]], locale: &str, calendar: &str, key_len: usize) -> Vec<[Id; N]> {",
            "    let mut result: Vec<[Id; N]> = Vec::new();",
            "    for locale in chain(locale) {",
            "        let start = table.partition_point(|row| (text(row[0]), text(row[1])) < (*locale, calendar));",
            "        for row in table[start..].iter().take_while(|row| text(row[0]) == *locale && text(row[1]) == calendar) {",
            "            if !result.iter().any(|seen| seen[2..2 + key_len] == row[2..2 + key_len]) {",
            "                result.push(*row);",
            "            }",
            "        }",
            "    }",
            "    result",
            "}",
            "",
            "/// The availableFormats (skeleton, pattern, numbers) of a calendar, sorted by skeleton.",
            "pub(crate) fn available_formats(locale: &str, calendar: &str) -> Vec<(&'static str, &'static str, &'static str)> {",
            "    let mut rows = merged(AVAILABLE, locale, calendar, 1);",
            "    if rows.is_empty() {",
            "        rows = merged(AVAILABLE, locale, \"gregory\", 1);",
            "    }",
            "    let mut rows: Vec<_> = rows.into_iter().map(|row| (text(row[2]), text(row[3]), text(row[4]))).collect();",
            "    rows.sort_unstable();",
            "    rows",
            "}",
            "",
            "/// The intervalFormatItem skeletons of a calendar.",
            "pub(crate) fn interval_skeletons(locale: &str, calendar: &str) -> Vec<&'static str> {",
            "    let mut skeletons: Vec<&str> = merged(INTERVALS, locale, calendar, 2).into_iter().map(|row| text(row[2])).collect();",
            "    skeletons.sort_unstable();",
            "    skeletons.dedup();",
            "    skeletons",
            "}",
            "",
            "/// An intervalFormatItem pattern of a calendar for a skeleton and greatest difference.",
            "pub(crate) fn interval(",
            "    locale: &str,",
            "    calendar: &str,",
            "    skeleton: &str,",
            "    difference: &str,",
            ") -> Option<&'static str> {",
            "    inherited(locale, |locale| {",
            "        INTERVALS",
            "            .binary_search_by(|row| {",
            "                (text(row[0]), text(row[1]), text(row[2]), text(row[3])).cmp(&(locale, calendar, skeleton, difference))",
            "            })",
            "            .ok()",
            "            .map(|index| text(INTERVALS[index][4]))",
            "    })",
            "}",
            "",
            "fn keyed(rows: &'static [[Id; 4]], locale: &str, a: &str, b: &str) -> Option<&'static str> {",
            "    inherited(locale, |locale| {",
            "        rows.binary_search_by(|row| (text(row[0]), text(row[1]), text(row[2])).cmp(&(locale, a, b)))",
            "            .ok()",
            "            .map(|index| text(rows[index][3]))",
            "    })",
            "}",
            "",
            "/// A format-context weekday name (`E`) for a width and CLDR day key.",
            "pub(crate) fn weekday(locale: &str, width: &str, key: &str) -> Option<&'static str> {",
            "    keyed(WEEKDAYS, locale, width, key)",
            "}",
            "",
            "/// A stand-alone weekday name (`c`); CLDR's stand-alone names default to the format names.",
            "pub(crate) fn standalone_weekday_override(locale: &str, width: &str, key: &str) -> Option<&'static str> {",
            "    keyed(STANDALONE_WEEKDAYS, locale, width, key)",
            "}",
            "",
            "/// A format-context day-period name for a width and CLDR period key.",
            "pub(crate) fn day_period(locale: &str, width: &str, key: &str) -> Option<&'static str> {",
            "    keyed(DAY_PERIODS, locale, width, key)",
            "}",
            "",
            "/// A calendar's appendItems pattern for a UTS #35 field name (`Era`, `Day-Of-Week`, …).",
            "pub(crate) fn append_item(locale: &str, calendar: &str, field: &str) -> Option<&'static str> {",
            "    keyed(APPEND, locale, calendar, field)",
            "}",
            "",
            "/// The cyclic (sexagenary) year name, 1 ≤ `year` ≤ 60, of the Chinese or Dangi calendar.",
            "pub(crate) fn cyclic_year(locale: &str, calendar: &str, width: &str, year: u8) -> Option<&'static str> {",
            "    let names = inherited(locale, |locale| {",
            "        CYCLIC",
            "            .binary_search_by(|row| (text(row.0), text(row.1), text(row.2)).cmp(&(locale, calendar, width)))",
            "            .ok()",
            "            .map(|index| CYCLIC[index].3)",
            "    })?;",
            "    names.get(usize::from(year).checked_sub(1)?).copied()",
            "}",
            "",
            "/// The leap-month pattern (`{0}` is the month) of the Chinese or Dangi calendar for a",
            "/// context-width key (`format-long`, `stand-alone-short`, …) or `numeric`.",
            "pub(crate) fn leap_month_pattern(locale: &str, calendar: &str, key: &str) -> Option<&'static str> {",
            "    keyed(LEAP, locale, calendar, key)",
            "}",
            "",
            "/// A locale's CLDR day-period rules as (period, from hour, before hour); an `at` rule",
            "/// (`noon`, `midnight`) has equal hours. Ranges may wrap past midnight. Rules belong to a",
            "/// language, with a few locale-specific sets (es-CO, hi-Latn) found by truncation.",
            "pub(crate) fn day_period_rules(locale: &str) -> &'static [(&'static str, u8, u8)] {",
            "    crate::cldr_locales::truncations(locale).find_map(rules_of).unwrap_or(&[])",
            "}",
            "",
            "fn rules_of(key: &str) -> Option<&'static [(&'static str, u8, u8)]> {",
            "    Some(match key {",
        ]
    )
    for key, rows in day_period_rules.items():
        rendered = ", ".join(f"({rust_string(p)}, {a}, {b})" for p, a, b in rows)
        output.append(f'        "{key}" => &[{rendered}],')
    output.extend(["        _ => return None,", "    })", "}", ""])
    offsets, blob = [], ""
    for value in pool:
        offsets.append(len(blob.encode("utf-8")))
        blob += value
    offsets.append(len(blob.encode("utf-8")))
    output += [
        "/// The byte offset of each string id's text in `STRINGS`, then the end of the last.",
        f"static OFFSETS: [u32; {len(offsets)}] = [",
    ]
    output += ["    " + ", ".join(map(str, offsets[k : k + 16])) + "," for k in range(0, len(offsets), 16)]
    output += ["];", ""]
    OUTPUT.with_suffix(".data").write_text(blob, encoding="utf-8")
    OUTPUT.write_text("\n".join(output), encoding="utf-8")
    print(
        f"wrote {OUTPUT} ({len(record_rows)} calendar records, {len(available_rows)} locales, "
        f"{sum(map(len, available_rows.values()))} availableFormats rows, "
        f"{sum(map(len, interval_rows.values()))} interval rows)"
    )


if __name__ == "__main__":
    main()
