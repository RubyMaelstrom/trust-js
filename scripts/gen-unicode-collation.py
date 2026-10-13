#!/usr/bin/env python3
"""Generate TRust JS's compact Unicode 17 DUCET collation-element table.

The input is the normative allkeys.txt published with UTS #10. The binary layout keeps the large
weight table out of Rust code generation while retaining binary-searchable singleton mappings,
longest-match contraction data, expansions, and the DUCET variable-weight marker.
"""

from __future__ import annotations

import pathlib
import re
import struct
import io
import urllib.request
import zipfile


VERSION = "17.0.0"
# Unicode publishes the current UCA data through this stable endpoint (there is no versioned
# public directory for 17.0.0); the embedded header below is checked before generation.
URL = "https://www.unicode.org/Public/UCA/latest/allkeys.txt"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/trust-js/src/unicode_collation.bin"
TEST_OUTPUT = ROOT / "crates/trust-js/tests/unicode-17.0.0/CollationTest_NON_IGNORABLE_SHORT.txt"
ROW = re.compile(r"^([0-9A-F ]+?)\s*;\s*(.*?)\s*(?:#.*)?$")
ELEMENT = re.compile(r"\[([.*])([0-9A-F]{4})\.([0-9A-F]{4})\.([0-9A-F]{4})\]")


def download() -> str:
    request = urllib.request.Request(URL, headers={"User-Agent": "TRust JS UCA table generator"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def download_conformance_test() -> bytes:
    url = "https://www.unicode.org/Public/UCA/latest/CollationTest.zip"
    request = urllib.request.Request(url, headers={"User-Agent": "TRust JS UCA table generator"})
    with urllib.request.urlopen(request) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    return archive.read("CollationTest/CollationTest_NON_IGNORABLE_SHORT.txt")


def main() -> None:
    source = download()
    if not source.startswith(f"# allkeys-{VERSION}.txt\n"):
        raise RuntimeError(f"expected UCA {VERSION} allkeys header")
    singles: list[tuple[int, list[int]]] = []
    contractions: list[tuple[tuple[int, ...], list[int]]] = []
    for line in source.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        source = tuple(int(value, 16) for value in match.group(1).split())
        elements = []
        for variable, primary, secondary, tertiary in ELEMENT.findall(match.group(2)):
            packed = (
                (variable == "*") << 48
                | int(primary, 16) << 32
                | int(secondary, 16) << 16
                | int(tertiary, 16)
            )
            elements.append(packed)
        if not elements:
            raise RuntimeError(f"mapping has no collation elements: {line}")
        if len(source) == 1:
            singles.append((source[0], elements))
        else:
            contractions.append((source, elements))

    singles.sort(key=lambda row: row[0])
    contractions.sort(key=lambda row: row[0])
    if len({row[0] for row in singles}) != len(singles):
        raise RuntimeError("duplicate singleton DUCET mapping")
    if len({row[0] for row in contractions}) != len(contractions):
        raise RuntimeError("duplicate contraction DUCET mapping")

    weights: list[int] = []
    single_rows = []
    for code_point, elements in singles:
        offset = len(weights)
        weights.extend(elements)
        single_rows.append((code_point, offset, len(elements)))
    sequence_values: list[int] = []
    contraction_rows = []
    for source, elements in contractions:
        sequence_offset = len(sequence_values)
        sequence_values.extend(source)
        weight_offset = len(weights)
        weights.extend(elements)
        contraction_rows.append(
            (source[0], sequence_offset, len(source), weight_offset, len(elements))
        )

    output = bytearray(b"LUCA17\0\1")
    output.extend(
        struct.pack(
            "<IIII",
            len(single_rows),
            len(contraction_rows),
            len(sequence_values),
            len(weights),
        )
    )
    for code_point, offset, length in single_rows:
        output.extend(struct.pack("<IIHH", code_point, offset, length, 0))
    for first, sequence_offset, sequence_length, weight_offset, weight_length in contraction_rows:
        output.extend(
            struct.pack(
                "<IIHHIHH",
                first,
                sequence_offset,
                sequence_length,
                0,
                weight_offset,
                weight_length,
                0,
            )
        )
    output.extend(struct.pack(f"<{len(sequence_values)}I", *sequence_values))
    output.extend(struct.pack(f"<{len(weights)}Q", *weights))
    OUTPUT.write_bytes(output)
    conformance = download_conformance_test()
    if b"# UCA Version: 17.0.0\n" not in conformance[:500]:
        raise RuntimeError(f"expected UCA {VERSION} conformance corpus")
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.write_bytes(conformance)
    print(
        f"wrote {OUTPUT} ({len(singles)} singletons, {len(contractions)} contractions, "
        f"{len(weights)} collation elements, {len(output)} bytes)"
    )


if __name__ == "__main__":
    main()
