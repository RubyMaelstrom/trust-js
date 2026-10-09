"""The CLDR locales Lumen's Intl services ship, and their inheritance (shared by the generators).

Every CLDR 48 locale of a supported language is shipped: the language locales and all of their
regional and script variants (UTS #35 Part 1 Locale Inheritance). A variant stores only the data
that differs from what its parent resolves, and a lookup walks the variant's parent chain
(`en-GB` -> `en-001` -> `en`), so a generator emits deltas with `delta()` and the runtime walks
`crate::cldr_locales::chain`. Locales whose CLDR parent is root (`sr-Latn`, `zh-Hant`, ...) store
all of their data.

`CLDR_DIR` names a local copy of the pinned cldr-json tree (`<package>/...`, as in the release's
`cldr-48.0.0-json-full.zip`); without it the files are fetched from the pinned tag.
"""

from __future__ import annotations

import functools
import json
import os
import urllib.request

TAG = "48.0.0"
CLDR_JSON = f"https://raw.githubusercontent.com/unicode-org/cldr-json/{TAG}/cldr-json"
# The languages Lumen's Intl services support.
LANGUAGES = [
    "ar", "de", "en", "es", "fr", "gv", "hi", "it", "ja", "ko", "ln", "nl", "pl", "pt", "ru",
    "si", "sl", "sr", "sv", "th", "zh",
]


def read_json(relative: str) -> dict:
    """A cldr-json file by its path below the cldr-json root."""
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        with open(os.path.join(cldr_dir, relative), encoding="utf-8") as file:
            return json.load(file)
    request = urllib.request.Request(
        f"{CLDR_JSON}/{relative}", headers={"User-Agent": "Lumen CLDR generator"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


@functools.cache
def locales() -> tuple[str, ...]:
    """Every CLDR (full coverage) locale of a supported language, sorted."""
    available = read_json("cldr-core/availableLocales.json")["availableLocales"]["full"]
    return tuple(sorted(l for l in available if l.split("-")[0] in LANGUAGES))


@functools.cache
def _parents() -> dict[str, str]:
    data = read_json("cldr-core/supplemental/parentLocales.json")
    return data["supplemental"]["parentLocales"]["parentLocale"]


@functools.cache
def _locale_rules() -> dict[str, str]:
    data = read_json("cldr-core/supplemental/parentLocales.json")
    return data["supplemental"]["parentLocales"].get("_localeRules", {}).get("parentLocale", {})


def parent(locale: str) -> str | None:
    """The CLDR parent of a shipped locale for the main data (UTS #35 Part 1 Parent Locales: the
    explicit parent; root for a language_script locale whose script is not the language's likely
    one, per the `nonlikelyScript` locale rule; else truncation), or None when that is root."""
    explicit = _parents().get(locale)
    if explicit is not None:
        return None if explicit in ("root", "und") else explicit
    if "-" not in locale:
        return None
    parts = locale.split("-")
    if (
        len(parts) == 2
        and len(parts[1]) == 4
        and _locale_rules().get("nonlikelyScript") == "root"
        and maximize(parts[0])[1] != parts[1]
    ):
        return None
    return locale.rsplit("-", 1)[0]


def truncations(locale: str) -> list[str]:
    """The locale and its truncations (UTS #35 Part 1: the inheritance of components without
    parent-locale overrides, such as plural rules and day-period rules)."""
    result = [locale]
    while "-" in result[-1]:
        result.append(result[-1].rsplit("-", 1)[0])
    return result


def chain(locale: str) -> list[str]:
    """The locale and its ancestors up to (and excluding) root."""
    result = [locale]
    while (next_parent := parent(result[-1])) is not None:
        result.append(next_parent)
    return result


def delta(tables: dict[str, dict]) -> dict[str, dict]:
    """For each locale, the entries of its full (resolved) table that its parent's resolved table
    does not give; a locale whose parent is root keeps everything. cldr-json resolves each
    locale, so a variant's table covers every key of its parent's."""
    result = {}
    for locale in locales():
        own = tables[locale]
        base_locale = parent(locale)
        if base_locale is None:
            result[locale] = dict(own)
            continue
        base = tables[base_locale]
        result[locale] = {key: value for key, value in own.items() if base.get(key) != value}
    return result


@functools.cache
def _likely() -> dict[str, str]:
    return read_json("cldr-core/supplemental/likelySubtags.json")["supplemental"]["likelySubtags"]


def maximize(language: str, script: str = "", region: str = "") -> tuple[str, str, str]:
    """UTS #35 Part 1 Add Likely Subtags for a language with an optional script and region."""
    likely = _likely()
    for key in (
        f"{language}-{script}-{region}" if script and region else None,
        f"{language}-{region}" if region else None,
        f"{language}-{script}" if script else None,
        language,
    ):
        if key and key in likely:
            found = likely[key].split("-")
            return (language, script or found[1], region or found[2])
    return (language, script, region)


@functools.cache
def available() -> dict[str, str]:
    """ECMA-402 [[AvailableLocales]] (each a canonical tag) mapped to the shipped locale holding
    its data: the shipped locales; CLDR's default-content locales (en-US holds en's data); and
    for each language-script-region tag, the language-region tag ECMA-402 requires (#sec-internal-
    slots), whose data is that of its likely script (zh-HK is zh-Hant-HK, zh-SG zh-Hans-SG)."""
    shipped = set(locales())
    result = {locale: locale for locale in shipped}

    def holder(tag: str) -> str | None:
        while tag:
            if tag in shipped:
                return tag
            tag = tag.rsplit("-", 1)[0] if "-" in tag else ""
        return None

    default_content = read_json("cldr-core/defaultContent.json")["defaultContent"]
    for tag in default_content:
        if tag.split("-")[0] in LANGUAGES and holder(tag) is not None:
            result.setdefault(tag, holder(tag))
    for tag in sorted(result):
        parts = tag.split("-")
        if len(parts) == 3 and len(parts[1]) == 4:
            language, _, region = parts
            short = f"{language}-{region}"
            if short in result:
                continue
            _, likely_script, _ = maximize(language, "", region)
            for candidate in (f"{language}-{likely_script}-{region}", tag):
                if candidate in result:
                    result[short] = result[candidate]
                    break
    return result
