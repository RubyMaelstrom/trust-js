#!/usr/bin/env python3
"""Regenerate the offset and daylight tables in tzdata.rs and the footer rules in tzrules.rs.

usage: scripts/gen-tzdata.py ZONEINFO_DIR [crates/trust-js/src/tzdata.rs] [crates/trust-js/src/tzrules.rs]

ZONEINFO_DIR is an IANA tz release compiled with explicit transitions through
2037, without the out-of-scope `backzone` data. The current tables are tzdata
2026b, compiled from its source directory with
`zic -b fat -d DIR africa antarctica asia australasia europe northamerica
southamerica etcetera backward factory`. The ZONES and LINKS lists are kept;
scripts/gen-tz-primaries.py owns them.

For every zone, the TZif version 2+ data block supplies `initial` (time type 0)
and the transitions at which the UTC offset changes, and the TZif footer
(RFC 9636 section 3.3) supplies the TZ string governing every instant after the
last transition. Each local time type's `tt_isdst` (RFC 9636 section 3.2) is kept
as well: a bit set per table marks the offset transitions that enter daylight
saving time, and DST_CHANGES lists the few transitions that change only
`tt_isdst` (such as British Standard Time, 1968-1971). Zones that share a table
must have identical transitions. The script refuses to write anything unless
each footer is consistent with its zone's last transition, as RFC 9636 section
3.3 requires.
"""
import re
import struct
import sys

ZONE_RE = re.compile(
    r'Zone \{ name: "([^"]+)", initial: (-?\d+), transitions: (\w+)(?:, dst: (?:\w+|&\[\]))? \}'
)
TABLE_RE = re.compile(r"static (TZ_\w+): &\[\(i64, i32\)\] = &\[(.*?)\];")
DST_TABLE_RE = re.compile(r"static TZD_\w+: &\[u64\] = &\[.*?\];\n")
DST_CHANGES_RE = re.compile(r"\n/// Transitions that change only.*?\n\];\n", re.S)
ZONES_MARKER = "\n/// Canonical zones, sorted by name."


def read_tzif(path):
    """(time type 0 offset, [(transition, offset, isdst)] at offset changes, [(transition,
    isdst)] at changes of `tt_isdst` alone, footer TZ string)."""
    data = open(path, "rb").read()
    if data[:4] != b"TZif" or data[4:5] < b"2":
        raise ValueError(f"{path}: not a version 2+ TZif file")

    def counts(at):
        return struct.unpack(">6l", data[at + 20 : at + 44])

    def block_len(at, time_size):
        isutcnt, isstdcnt, leapcnt, timecnt, typecnt, charcnt = counts(at)
        return (timecnt * time_size + timecnt + typecnt * 6 + charcnt
                + leapcnt * (time_size + 4) + isstdcnt + isutcnt)

    v2 = 44 + block_len(0, 4)
    _, _, _, timecnt, typecnt, _ = counts(v2)
    at = v2 + 44
    times = struct.unpack(f">{timecnt}q", data[at : at + 8 * timecnt])
    at += 8 * timecnt
    indices = data[at : at + timecnt]
    at += timecnt
    # ttinfo (RFC 9636 section 3.2): tt_utoff, tt_isdst, tt_desigidx.
    types = [struct.unpack(">lBB", data[at + 6 * k : at + 6 * k + 6])[:2] for k in range(typecnt)]
    if types[0][1]:
        raise ValueError(f"{path}: time type 0 is daylight saving time")
    # Keep the first transition even when it retains type 0's offset (e.g. LMT to a mean
    # time with the same offset), then only the transitions that change the offset.
    transitions, dst_only, previous, previous_dst = [], [], None, False
    for time, index in zip(times, indices):
        offset, isdst = types[index][0], bool(types[index][1])
        if offset != previous:
            transitions.append((time, offset, isdst))
        elif isdst != previous_dst:
            dst_only.append((time, isdst))
        previous, previous_dst = offset, isdst
    tail = data[v2 + 44 + block_len(v2, 8) :]
    if not tail.startswith(b"\n"):
        raise ValueError(f"{path}: missing TZif footer")
    footer = tail[1 : tail.index(b"\n", 1)].decode("ascii")
    return types[0][0], transitions, dst_only, footer


# A reference evaluator for the footer (POSIX.1-2017 Base Definitions 8.3 with
# the RFC 9636 3.3.2 extension), used only to check consistency.

def parse_name(s, i):
    if s[i] == "<":
        end = s.index(">", i)
        return end + 1
    while i < len(s) and s[i].isalpha():
        i += 1
    return i


def parse_hms(s, i):
    sign = 1
    if s[i] in "+-":
        sign, i = (-1 if s[i] == "-" else 1), i + 1
    parts = []
    while True:
        m = re.match(r"\d+", s[i:])
        parts.append(int(m.group(0)))
        i += m.end()
        if i < len(s) and s[i] == ":" and len(parts) < 3:
            i += 1
        else:
            break
    parts += [0] * (3 - len(parts))
    return sign * (parts[0] * 3600 + parts[1] * 60 + parts[2]), i


def parse_date(s, i):
    m = re.match(r"M(\d+)\.(\d+)\.(\d+)|J(\d+)|(\d+)", s[i:])
    if m.group(1):
        rule = ("M", int(m.group(1)), int(m.group(2)), int(m.group(3)))
    elif m.group(4):
        rule = ("J", int(m.group(4)))
    else:
        rule = ("N", int(m.group(5)))
    i += m.end()
    time = 7200
    if i < len(s) and s[i] == "/":
        time, i = parse_hms(s, i + 1)
    return (rule, time), i


def parse_tz(s):
    i = parse_name(s, 0)
    std, i = parse_hms(s, i)
    if i == len(s):
        return -std, None
    i = parse_name(s, i)
    dst = std - 3600
    if s[i] != ",":
        dst, i = parse_hms(s, i)
    start, i = parse_date(s, i + 1)
    end, i = parse_date(s, i + 1)
    assert i == len(s), s
    return -std, (-dst, start, end)


def days_from_civil(y, m, d):
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * ((m + 9) % 12) + 2) // 5 + d - 1
    return era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468


def rule_day(rule, y):
    if rule[0] == "J":
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        return days_from_civil(y, 1, 1) + rule[1] - 1 + (leap and rule[1] >= 60)
    if rule[0] == "N":
        return days_from_civil(y, 1, 1) + rule[1]
    _, m, w, d = rule
    first = days_from_civil(y, m, 1)
    following = days_from_civil(y + m // 12, m % 12 + 1, 1)
    day = first + (d - (first + 4) % 7) % 7 + (w - 1) * 7
    while day >= following:
        day -= 7
    return day


def offset_at(footer, t):
    std, dst = parse_tz(footer)
    if dst is None:
        return std
    dst_offset, (start, start_time), (end, end_time) = dst
    year = 1970 + t // 31556952
    events = []
    for y in range(year - 1, year + 2):
        events.append((rule_day(start, y) * 86400 + start_time - std, 1, dst_offset))
        events.append((rule_day(end, y) * 86400 + end_time - dst_offset, 0, std))
    events.sort()
    current = std
    for when, _, offset in events:
        if when <= t:
            current = offset
    return current


def dst_words(transitions):
    """The table's daylight bit set: bit k (of word k / 64) marks transition k as entering
    daylight saving time."""
    words = [0] * ((len(transitions) + 63) // 64)
    for k, (_, _, isdst) in enumerate(transitions):
        if isdst:
            words[k // 64] |= 1 << (k % 64)
    return words if any(words) else []


def main():
    zoneinfo = sys.argv[1]
    tzdata_path = sys.argv[2] if len(sys.argv) > 2 else "crates/trust-js/src/tzdata.rs"
    rules_path = sys.argv[3] if len(sys.argv) > 3 else "crates/trust-js/src/tzrules.rs"
    text = open(tzdata_path).read()
    zones = ZONE_RE.findall(text)
    by_table, initials, footers, dst_changes, problems = {}, {}, {}, {}, []
    for name, _, table in zones:
        initial, transitions, dst_only, footer = read_tzif(f"{zoneinfo}/{name}")
        if transitions:
            last_time, last_offset, _ = transitions[-1]
        else:
            last_time, last_offset = 0, initial
        if footer and offset_at(footer, last_time) != last_offset:
            problems.append(f"{name}: footer {footer!r} disagrees with its last transition")
        if not footer:
            problems.append(f"{name}: empty TZif footer")
        if by_table.setdefault(table, transitions) != transitions:
            problems.append(f"{name}: transitions differ from the other users of {table}")
        initials[name] = initial
        footers[name] = footer
        if dst_only:
            dst_changes[name] = dst_only
    if problems:
        sys.exit("refusing to write:\n  " + "\n  ".join(problems))

    def table_line(match):
        rows = ", ".join(f"({t},{o})" for t, o, _ in by_table[match.group(1)])
        return f"static {match.group(1)}: &[(i64, i32)] = &[{rows}];"

    def dst_name(table):
        return "TZD_" + table[len("TZ_"):] if dst_words(by_table[table]) else "&[]"

    text = TABLE_RE.sub(table_line, text)
    text = DST_TABLE_RE.sub("", text)
    text = DST_CHANGES_RE.sub("", text)
    dst_tables = "".join(
        f"static TZD_{table[len('TZ_'):]}: &[u64] = &[{', '.join(f'{w:#x}' for w in words)}];\n"
        for table in sorted(by_table)
        if (words := dst_words(by_table[table]))
    )
    changes = "".join(
        f'    ("{name}", &[{", ".join(f"({t}, {str(d).lower()})" for t, d in rows)}]),\n'
        for name, rows in sorted(dst_changes.items())
    )
    block = (
        "\n" + dst_tables
        + "\n/// Transitions that change only `tt_isdst`, not the UTC offset, by zone (sorted).\n"
        + "pub static DST_CHANGES: &[(&str, &[(i64, bool)])] = &[\n" + changes + "];\n"
    )
    if ZONES_MARKER not in text:
        sys.exit(f"{tzdata_path}: missing the ZONES list")
    text = text.replace(ZONES_MARKER, block + ZONES_MARKER, 1)
    text = ZONE_RE.sub(
        lambda m: (
            f'Zone {{ name: "{m.group(1)}", initial: {initials[m.group(1)]}, '
            f"transitions: {m.group(3)}, dst: {dst_name(m.group(3))} }}"
        ),
        text,
    )
    open(tzdata_path, "w").write(text)
    with open(rules_path, "w") as out:
        out.write(
            "//! IANA TZif footer TZ strings, by zone -- GENERATED by scripts/gen-tzdata.py. DO NOT EDIT.\n"
            "//!\n"
            "//! RFC 9636 section 3.3: each string is \"a rule for computing local time changes after the\n"
            "//! last transition time\" of its zone's table in [`crate::tzdata`], in the POSIX TZ format\n"
            "//! (POSIX.1-2017 Base Definitions section 8.3) with the section 3.3.2 extension.\n\n"
            "pub static RULES: &[(&str, &str)] = &[\n"
        )
        for name in sorted(footers):
            out.write(f'    ("{name}", "{footers[name]}"),\n')
        out.write("];\n")
    print(f"{len(zones)} zones, {len(by_table)} tables, {len(footers)} rules, "
          f"{len(dst_changes)} zones with tt_isdst-only changes")


if __name__ == "__main__":
    main()
