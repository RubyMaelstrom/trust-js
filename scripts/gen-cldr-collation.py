#!/usr/bin/env python3
"""Generate compact CLDR 48 Han primary-order tailorings for Intl.Collator.

The large Japanese and Chinese rules are LDML starred primary relations. Keeping the resulting
code-point/rank maps in a binary avoids over 190,000 Rust source rows while preserving the exact
order published by CLDR. Smaller alphabetic rules remain auditable beside the comparison code.
"""

from __future__ import annotations

import io
import pathlib
import re
import struct
import urllib.request
import xml.etree.ElementTree as etree
import zipfile


VERSION = "48"
URL = f"https://unicode.org/Public/cldr/{VERSION}/core.zip"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "crates/trust-js/src/cldr_collation.bin"
ROOT_OUTPUT = ROOT / "crates/trust-js/src/cldr_root_collation.bin"
TEST_OUTPUT = ROOT / "crates/trust-js/tests/cldr-48/CollationTest_CLDR_NON_IGNORABLE_SHORT.txt"
TABLES = (("ja", "standard"), ("zh", "pinyin"), ("zh", "stroke"), ("zh", "zhuyin"))
ROW = re.compile(r"^([0-9A-F ]+?)\s*;\s*(.*?)\s*(?:#.*)?$")
ELEMENT = re.compile(r"\[([.*])([0-9A-F]{4})\.([0-9A-F]{4})\.([0-9A-F]{4})\]")


def download() -> zipfile.ZipFile:
    request = urllib.request.Request(URL, headers={"User-Agent": "TRust JS CLDR generator"})
    with urllib.request.urlopen(request) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def starred_primary_ranks(archive: zipfile.ZipFile, locale: str, kind: str) -> list[tuple[int, int]]:
    root = etree.fromstring(archive.read(f"common/collation/{locale}.xml"))
    collations = root.find("collations")
    assert collations is not None
    collation = next(
        node
        for node in collations.findall("collation")
        if node.get("type") == kind and node.get("alt") is None
    )
    rules = collation.findtext("cr") or ""
    rules = "\n".join(line.split("#", 1)[0] for line in rules.splitlines())
    marker = "&[last regular]"
    if marker not in rules:
        raise RuntimeError(f"{locale}-{kind} has no last-regular Han tailoring")
    rules = rules[rules.index(marker) :]

    # UTS #35 starred relation syntax: `<*abc` is equivalent to `<a<b<c`. Later
    # relations replace earlier ones for the one duplicate present in zhuyin data.
    ordered = [character for token in re.findall(r"<\*([^\s&<]+)", rules) for character in token]
    ranks = {ord(character): rank for rank, character in enumerate(ordered, 1)}
    if len(ranks) < 6_000:
        raise RuntimeError(f"unexpectedly short {locale}-{kind} tailoring")
    return sorted(ranks.items())


def write_root_table(source: str) -> tuple[int, int, int]:
    singles: list[tuple[int, list[int]]] = []
    contractions: list[tuple[tuple[int, ...], list[int]]] = []
    for line in source.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        code_points = tuple(int(value, 16) for value in match.group(1).split())
        elements = []
        for variable, primary, secondary, tertiary in ELEMENT.findall(match.group(2)):
            elements.append(
                (variable == "*") << 48
                | int(primary, 16) << 32
                | int(secondary, 16) << 16
                | int(tertiary, 16)
            )
        if not elements:
            raise RuntimeError(f"CLDR root mapping has no elements: {line}")
        (singles if len(code_points) == 1 else contractions).append((code_points, elements))

    single_rows = []
    contraction_rows = []
    sequences: list[int] = []
    weights: list[int] = []
    for source_points, elements in sorted(singles):
        offset = len(weights)
        weights.extend(elements)
        single_rows.append((source_points[0], offset, len(elements)))
    for source_points, elements in sorted(contractions):
        sequence_offset = len(sequences)
        sequences.extend(source_points)
        weight_offset = len(weights)
        weights.extend(elements)
        contraction_rows.append(
            (source_points[0], sequence_offset, len(source_points), weight_offset, len(elements))
        )

    output = bytearray(b"LCLR17\0\1")
    output.extend(struct.pack("<4I", len(single_rows), len(contraction_rows), len(sequences), len(weights)))
    for code_point, offset, length in single_rows:
        output.extend(struct.pack("<IIHH", code_point, offset, length, 0))
    for first, sequence_offset, sequence_length, weight_offset, weight_length in contraction_rows:
        output.extend(struct.pack("<IIHHIHH", first, sequence_offset, sequence_length, 0, weight_offset, weight_length, 0))
    output.extend(struct.pack(f"<{len(sequences)}I", *sequences))
    output.extend(struct.pack(f"<{len(weights)}Q", *weights))
    ROOT_OUTPUT.write_bytes(output)
    return len(single_rows), len(contraction_rows), len(output)


def main() -> None:
    archive = download()
    if "common/uca/allkeys_CLDR.txt" not in archive.namelist():
        raise RuntimeError(f"download is not a CLDR {VERSION} core archive")
    root_source = archive.read("common/uca/allkeys_CLDR.txt")
    if b"# UCA Version: 17.0.0" not in root_source[:500]:
        raise RuntimeError("CLDR 48 archive does not contain Unicode 17 collation data")
    root_counts = write_root_table(root_source.decode("utf-8"))

    tables = [starred_primary_ranks(archive, *table) for table in TABLES]
    output = bytearray(b"LCLD48\0\1")
    output.extend(struct.pack("<4I", *(len(table) for table in tables)))
    for table in tables:
        for code_point, rank in table:
            output.extend(struct.pack("<II", code_point, rank))
    OUTPUT.write_bytes(output)
    conformance = archive.read("common/uca/CollationTest_CLDR_NON_IGNORABLE_SHORT.txt")
    if b"# UCA Version: 17.0.0" not in conformance[:500]:
        raise RuntimeError("CLDR root conformance corpus is not based on Unicode 17")
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.write_bytes(conformance)
    print(
        f"wrote {OUTPUT} ({', '.join(str(len(table)) for table in tables)} rows, "
        f"{len(output)} bytes); {ROOT_OUTPUT} ({root_counts[0]} singletons, "
        f"{root_counts[1]} contractions, {root_counts[2]} bytes)"
    )


if __name__ == "__main__":
    main()
