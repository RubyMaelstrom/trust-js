#!/usr/bin/env python3
"""Generate Unicode 17 normalization tables and the official UAX #15 test corpus."""

from __future__ import annotations

from functools import lru_cache
import pathlib
import urllib.request


VERSION = "17.0.0"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/trust-js/src/unicode_norm.rs"
TEST_OUTPUT = ROOT / "crates/trust-js/tests/unicode-17.0.0/NormalizationTest.txt"
UCD = f"https://www.unicode.org/Public/{VERSION}/ucd"


def download(name: str) -> str:
    request = urllib.request.Request(
        f"{UCD}/{name}", headers={"User-Agent": "TRust JS Unicode normalization generator"}
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def code_point_range(value: str) -> range:
    pieces = value.strip().split("..")
    start = int(pieces[0], 16)
    end = int(pieces[-1], 16)
    return range(start, end + 1)


def main() -> None:
    unicode_data = download("UnicodeData.txt")
    if "COMBINING DOUBLE CARON" not in unicode_data:
        raise RuntimeError(f"expected Unicode {VERSION} data")
    canonical: dict[int, tuple[int, ...]] = {}
    compatibility: dict[int, tuple[int, ...]] = {}
    combining_classes: dict[int, int] = {}
    for line in unicode_data.splitlines():
        fields = line.split(";")
        code_point = int(fields[0], 16)
        combining_class = int(fields[3])
        if combining_class:
            combining_classes[code_point] = combining_class
        decomposition = fields[5].split()
        if not decomposition:
            continue
        if decomposition[0].startswith("<"):
            compatibility[code_point] = tuple(int(value, 16) for value in decomposition[1:])
        else:
            canonical[code_point] = tuple(int(value, 16) for value in decomposition)

    exclusions = set()
    for raw in download("DerivedNormalizationProps.txt").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(";")]
        if len(fields) >= 2 and fields[1] == "Full_Composition_Exclusion":
            exclusions.update(code_point_range(fields[0]))

    @lru_cache(maxsize=None)
    def nfkd(code_point: int) -> tuple[int, ...]:
        mapping = compatibility.get(code_point) or canonical.get(code_point)
        if mapping is None:
            return (code_point,)
        return tuple(part for mapped in mapping for part in nfkd(mapped))

    compositions = []
    for composite, mapping in canonical.items():
        if len(mapping) == 2 and composite not in exclusions:
            compositions.append((mapping[0], mapping[1], composite))
    compositions.sort()

    output = [
        f"//! Unicode normalization data generated from the official UCD {VERSION} files.",
        "//! Canonical/compatibility decomposition, nonzero combining classes, and composition",
        "//! pairs implement UAX #15; Hangul is handled algorithmically by unicode_norm_impl.",
        "",
        "#[rustfmt::skip]",
        "pub static CANON_DECOMP: &[(u32, u32, u32)] = &[",
    ]
    for code_point, mapping in sorted(canonical.items()):
        if len(mapping) > 2:
            raise RuntimeError(f"canonical mapping longer than two at U+{code_point:04X}")
        second = mapping[1] if len(mapping) == 2 else 0
        output.append(f"    (0x{code_point:X}, 0x{mapping[0]:X}, 0x{second:X}),")
    output.extend(["];", "", "#[rustfmt::skip]", "pub static COMPAT_DECOMP: &[(u32, &[u32])] = &["])
    for code_point in sorted(compatibility):
        values = ", ".join(f"0x{part:X}" for part in nfkd(code_point))
        output.append(f"    (0x{code_point:X}, &[{values}]),")
    output.extend(["];", "", "#[rustfmt::skip]", "pub static CCC: &[(u32, u8)] = &["])
    for code_point, combining_class in sorted(combining_classes.items()):
        output.append(f"    (0x{code_point:X}, {combining_class}),")
    output.extend(["];", "", "#[rustfmt::skip]", "pub static COMPOSE: &[(u32, u32, u32)] = &["])
    for first, second, composite in compositions:
        output.append(f"    (0x{first:X}, 0x{second:X}, 0x{composite:X}),")
    output.extend(["];", ""])
    OUTPUT.write_text("\n".join(output), encoding="utf-8")

    normalization_test = download("NormalizationTest.txt")
    if f"# NormalizationTest-{VERSION}.txt" not in normalization_test[:200]:
        raise RuntimeError(f"expected Unicode {VERSION} normalization corpus")
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.write_text(normalization_test, encoding="utf-8")
    print(
        f"wrote {OUTPUT} ({len(canonical)} canonical, {len(compatibility)} compatibility, "
        f"{len(combining_classes)} CCC, {len(compositions)} composition rows)"
    )


if __name__ == "__main__":
    main()
