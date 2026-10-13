#!/usr/bin/env python3
"""Generate complete plural rules for TRust JS's supported locales from CLDR 48.

Authoritative inputs:
  https://github.com/unicode-org/cldr-json/tree/48.0.0/cldr-json/cldr-core/supplemental
    plurals.json, ordinals.json, and pluralRanges.json

The CLDR condition grammar is compiled to straight-line Rust. Runtime selection therefore performs
no parsing, hashing, locale allocation, or rule allocation. Decimal operands remain digit slices so
modulo tests stay exact even for BigInt and 100-digit fraction inputs.

Rule sets are keyed by language with a few locale-specific sets (pt-PT); a shipped locale uses the
first set among its truncations (plural rules have no parent-locale overrides).
`CLDR_DIR` names a local copy of the pinned cldr-json tree.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cldr_locales  # noqa: E402


CLDR_VERSION = "48"
TAG = "48.0.0"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/trust-js/src/cldr_plurals.rs"
BASE = (
    f"https://raw.githubusercontent.com/unicode-org/cldr-json/{TAG}/"
    "cldr-json/cldr-core/supplemental"
)
SUPPORTED = cldr_locales.LANGUAGES
CATEGORIES = ["zero", "one", "two", "few", "many", "other"]
RELATION = re.compile(r"^([niftvwce])(?:\s*%\s*(\d+))?\s*(!?=)\s*([0-9.,]+)$")


def fetch(name: str) -> dict:
    cldr_dir = os.environ.get("CLDR_DIR")
    if cldr_dir:
        with open(os.path.join(cldr_dir, "cldr-core/supplemental", name), encoding="utf-8") as file:
            data = json.load(file)
    else:
        request = urllib.request.Request(
            f"{BASE}/{name}", headers={"User-Agent": "TRust JS CLDR plural generator"}
        )
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
    version = data["supplemental"]["version"]["_cldrVersion"]
    if version != CLDR_VERSION:
        raise RuntimeError(f"{name}: expected CLDR {CLDR_VERSION}, got {version}")
    return data["supplemental"]


def compile_relation(text: str) -> str:
    match = RELATION.fullmatch(text.strip())
    if not match:
        raise ValueError(f"unsupported CLDR plural relation: {text!r}")
    operand, modulus, operator, values = match.groups()
    ranges = []
    for item in values.split(","):
        if ".." in item:
            first, last = item.split("..", 1)
        else:
            first = last = item
        ranges.append(f"({int(first)}, {int(last)})")
    modulus = modulus or "0"
    relation = f"relation(p, Operand::{operand.upper()}, {modulus}, &[{', '.join(ranges)}])"
    return f"!{relation}" if operator == "!=" else relation


def compile_condition(rule: str) -> str:
    condition = rule.split("@", 1)[0].strip()
    if not condition:
        return "true"
    alternatives = []
    for disjunction in condition.split(" or "):
        conjunction = [compile_relation(item) for item in disjunction.split(" and ")]
        alternatives.append(" && ".join(conjunction))
    if len(alternatives) == 1:
        return alternatives[0]
    return " || ".join(f"({item})" for item in alternatives)


def rule_body(rules: dict[str, str] | None) -> list[str]:
    if not rules:
        return ["        \"other\""]
    output = []
    for category in CATEGORIES:
        if category == "other":
            continue
        rule = rules.get(f"pluralRule-count-{category}")
        if rule is not None:
            output.append(f"        if {compile_condition(rule)} {{ return \"{category}\"; }}")
    output.append("        \"other\"")
    return output


def rule_keys(rules: dict) -> list[str]:
    """The rule sets of supported languages: languages and locale-specific sets (pt-PT)."""
    return sorted(key for key in rules if key.split("-")[0] in SUPPORTED)


def emit_selector(output: list[str], name: str, rules: dict[str, dict[str, str]]) -> None:
    keys = rule_keys(rules)
    output.extend(
        [
            f"pub(crate) fn {name}(locale: &str, decimal: &str, compact_exponent: u32) -> &'static str {{",
            "    let p = Operands::new(decimal, compact_exponent);",
            f"    match rule_set(locale, &[{', '.join(json.dumps(key) for key in keys)}]) {{",
        ]
    )
    for lang in keys:
        output.append(f'        "{lang}" => {{')
        output.extend(rule_body(rules.get(lang)))
        output.append("        }")
    output.extend(["        _ => \"other\",", "    }", "}", ""])


def categories(rules: dict[str, str] | None) -> list[str]:
    if not rules:
        return ["other"]
    return [category for category in CATEGORIES if f"pluralRule-count-{category}" in rules]


def expand_compact_sample(sample: str) -> tuple[str, int]:
    """Turn CLDR's `1.20050c3` sample into the operand decimal `1200.50` and c=3."""
    match = re.fullmatch(r"([+-]?)(\d+)(?:\.(\d+))?[ce](\d+)", sample)
    if not match:
        return sample.lstrip("+"), 0
    sign, integer, fraction, exponent = match.groups()
    fraction = fraction or ""
    exponent = int(exponent)
    digits = integer + fraction
    point = len(integer) + exponent
    if point >= len(digits):
        decimal = digits + "0" * (point - len(digits))
    else:
        decimal = f"{digits[:point]}.{digits[point:]}"
    if sign == "-":
        decimal = f"-{decimal}"
    return decimal, exponent


def official_samples(rules: dict[str, dict[str, str]]) -> list[tuple[str, str, int, str]]:
    samples = []
    for lang in rule_keys(rules):
        for key, rule in (rules.get(lang) or {}).items():
            category = key.removeprefix("pluralRule-count-")
            for section in re.findall(r"@(?:integer|decimal)\s+([^@]+)", rule):
                for item in section.split(","):
                    item = item.strip()
                    if not item or "…" in item or "..." in item:
                        continue
                    endpoints = item.split("~", 1)
                    for endpoint in endpoints:
                        decimal, exponent = expand_compact_sample(endpoint.strip())
                        samples.append((lang, decimal, exponent, category))
    return list(dict.fromkeys(samples))


def main() -> None:
    plurals_data = fetch("plurals.json")
    ordinals_data = fetch("ordinals.json")
    ranges_data = fetch("pluralRanges.json")
    cardinals = plurals_data["plurals-type-cardinal"]
    ordinals = ordinals_data["plurals-type-ordinal"]
    plural_ranges = ranges_data["plurals"]

    output = [
        "//! GENERATED by scripts/gen-cldr-plurals.py from CLDR 48. Do not edit.",
        "//! Cardinal/ordinal conditions and plural-range resolution for TRust JS's supported locales.",
        "",
        "#[allow(dead_code)]",
        "#[derive(Clone, Copy)]",
        "enum Operand { N, I, F, T, V, W, C, E }",
        "",
        "#[derive(Clone, Copy)]",
        "struct Operands<'a> {",
        "    integer: &'a str,",
        "    fraction: &'a str,",
        "    fraction_trimmed: &'a str,",
        "    compact_exponent: u32,",
        "}",
        "",
        "impl<'a> Operands<'a> {",
        "    fn new(decimal: &'a str, compact_exponent: u32) -> Self {",
        "        let decimal = decimal.strip_prefix(['+', '-']).unwrap_or(decimal);",
        "        let (integer, fraction) = decimal.split_once('.').unwrap_or((decimal, \"\"));",
        "        Self {",
        "            integer: if integer.is_empty() { \"0\" } else { integer },",
        "            fraction,",
        "            fraction_trimmed: fraction.trim_end_matches('0'),",
        "            compact_exponent,",
        "        }",
        "    }",
        "}",
        "",
        "fn digits_mod(digits: &str, modulus: u32) -> u32 {",
        "    digits.bytes().fold(0, |value, digit|",
        "        (value * 10 + u32::from(digit - b'0')) % modulus)",
        "}",
        "",
        "fn digits_in_ranges(digits: &str, modulus: u32, ranges: &[(u32, u32)]) -> bool {",
        "    let value = if modulus == 0 {",
        "        digits.parse::<u32>().ok()",
        "    } else {",
        "        Some(digits_mod(digits, modulus))",
        "    };",
        "    value.is_some_and(|value| ranges.iter().any(|&(first, last)| (first..=last).contains(&value)))",
        "}",
        "",
        "fn relation(p: Operands<'_>, operand: Operand, modulus: u32, ranges: &[(u32, u32)]) -> bool {",
        "    match operand {",
        "        Operand::N => p.fraction_trimmed.is_empty()",
        "            && digits_in_ranges(p.integer, modulus, ranges),",
        "        Operand::I => digits_in_ranges(p.integer, modulus, ranges),",
        "        Operand::F => digits_in_ranges(if p.fraction.is_empty() { \"0\" } else { p.fraction }, modulus, ranges),",
        "        Operand::T => digits_in_ranges(if p.fraction_trimmed.is_empty() { \"0\" } else { p.fraction_trimmed }, modulus, ranges),",
        "        Operand::V | Operand::W => {",
        "            let mut value = if matches!(operand, Operand::V) { p.fraction.len() } else { p.fraction_trimmed.len() } as u32;",
        "            if modulus != 0 { value %= modulus; }",
        "            ranges.iter().any(|&(first, last)| (first..=last).contains(&value))",
        "        }",
        "        Operand::C | Operand::E => {",
        "            let value = if modulus == 0 { p.compact_exponent } else { p.compact_exponent % modulus };",
        "            ranges.iter().any(|&(first, last)| (first..=last).contains(&value))",
        "        }",
        "    }",
        "}",
        "",
    ]
    output.extend(
        [
            "/// The rule set of a shipped locale: the first of `keys` among the locale's truncations",
            "/// (plural rules have no parent-locale overrides, so pt-AO uses pt's).",
            "fn rule_set(locale: &str, keys: &[&'static str]) -> &'static str {",
            "    crate::cldr_locales::truncations(locale)",
            "        .find_map(|candidate| keys.iter().copied().find(|key| *key == candidate))",
            "        .unwrap_or(\"\")",
            "}",
            "",
        ]
    )
    emit_selector(output, "select_cardinal", cardinals)
    emit_selector(output, "select_ordinal", ordinals)

    output.extend(
        [
            "pub(crate) fn categories(locale: &str, kind: &str) -> &'static [&'static str] {",
            f"    let cardinal = rule_set(locale, &[{', '.join(json.dumps(key) for key in rule_keys(cardinals))}]);",
            f"    let ordinal = rule_set(locale, &[{', '.join(json.dumps(key) for key in rule_keys(ordinals))}]);",
            "    match (kind, if kind == \"ordinal\" { ordinal } else { cardinal }) {",
        ]
    )
    for kind, rules in [("cardinal", cardinals), ("ordinal", ordinals)]:
        groups: dict[tuple[str, ...], list[str]] = {}
        for lang in rule_keys(rules):
            groups.setdefault(tuple(categories(rules.get(lang))), []).append(lang)
        for values, languages in groups.items():
            pattern = " | ".join(f'\"{lang}\"' for lang in languages)
            array = ", ".join(f'\"{value}\"' for value in values)
            output.append(f'        ("{kind}", {pattern}) => &[{array}],')
    output.extend(["        _ => &[\"other\"],", "    }", "}", ""])

    range_keys = rule_keys(plural_ranges)
    output.extend(
        [
            "pub(crate) fn select_range(locale: &str, start: &str, end: &str) -> &'static str {",
            f"    match (rule_set(locale, &[{', '.join(json.dumps(key) for key in range_keys)}]), start, end) {{",
        ]
    )
    for lang in range_keys:
        for key, result in plural_ranges.get(lang, {}).items():
            match = re.fullmatch(r"pluralRange-start-(\w+)-end-(\w+)", key)
            if not match:
                raise ValueError(f"invalid plural range key: {key}")
            start, end = match.groups()
            output.append(f'        ("{lang}", "{start}", "{end}") => "{result}",')
    output.extend(["        _ => \"other\",", "    }", "}", ""])

    for name, rules in [("CARDINAL_SAMPLES", cardinals), ("ORDINAL_SAMPLES", ordinals)]:
        samples = official_samples(rules)
        output.append("#[cfg(test)]")
        output.append(
            f"const {name}: &[(&str, &str, u32, &str)] = &["
        )
        for lang, decimal, exponent, category in samples:
            output.append(
                f'    ("{lang}", "{decimal}", {exponent}, "{category}"),'
            )
        output.extend(["];", ""])

    output.extend(
        [
            "#[cfg(test)]",
            "mod tests {",
            "    #[test]",
            "    fn cldr_48_official_plural_samples() {",
            "        for &(lang, decimal, exponent, expected) in super::CARDINAL_SAMPLES {",
            "            assert_eq!(super::select_cardinal(lang, decimal, exponent), expected, \"cardinal {lang} {decimal}c{exponent}\");",
            "        }",
            "        for &(lang, decimal, exponent, expected) in super::ORDINAL_SAMPLES {",
            "            assert_eq!(super::select_ordinal(lang, decimal, exponent), expected, \"ordinal {lang} {decimal}c{exponent}\");",
            "        }",
            "    }",
            "}",
            "",
        ]
    )

    OUTPUT.write_text("\n".join(output), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(output)} lines)")


if __name__ == "__main__":
    main()
