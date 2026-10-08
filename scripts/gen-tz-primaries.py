#!/usr/bin/env python3
"""Recompute the primary/non-primary time zone identifiers and region zones in tzdata.rs.

usage: scripts/gen-tz-primaries.py TZDATA_SOURCE_DIR [crates/lumen/src/tzdata.rs]

TZDATA_SOURCE_DIR is an IANA tz source tree (tzdata.zi, zone.tab, backward). A
tzdata release tarball omits tzdata.zi; only its Zone (`Z name`) and Link
(`L target name`) lines are read, so they can be listed from the source files.
REGION_ZONES lists each zone.tab region's zones for Intl.Locale's getTimeZones.
The offset tables in tzdata.rs are kept; only its ZONES and LINKS lists are
rewritten, following ECMA-402 #sec-use-of-iana-time-zone-database:

* every IANA Zone, and every Link listed in zone.tab's TZ column, is primary;
* "UTC" is primary, and Etc/UTC, Etc/GMT and GMT resolve to it;
* any other Link resolves to a primary in the same ISO 3166-1 country: its
  `#=` annotation in `backward` when present, a same-country override below,
  else its IANA target.
"""
import re
import sys

SAME_COUNTRY = {
    # Links whose IANA target lies in another country and that carry no `#=`
    # annotation: keep them with the zone.tab entry of their own country.
    "America/Coral_Harbour": "America/Atikokan",
    "Antarctica/South_Pole": "Antarctica/McMurdo",
    "Atlantic/Jan_Mayen": "Arctic/Longyearbyen",
    "Pacific/Yap": "Pacific/Chuuk",
    "Africa/Timbuktu": "Africa/Bamako",
}
UTC_ALIASES = {"Etc/UTC", "Etc/GMT", "GMT"}


def main():
    src = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "crates/lumen/src/tzdata.rs"
    zones, links, annotated = set(), {}, {}
    for line in open(f"{src}/tzdata.zi"):
        p = line.split()
        if p and p[0] == "Z":
            zones.add(p[1])
        elif p and p[0] == "L":
            links[p[2]] = p[1]
    for line in open(f"{src}/backward"):
        m = re.match(r"Link\s+\S+\s+(\S+)\s+#=\s*(\S+)", line)
        if m:
            annotated[m.group(1)] = m.group(2)
    tab_rows = [
        line.split("\t")
        for line in open(f"{src}/zone.tab")
        if line.strip() and not line.startswith("#")
    ]
    tab = {row[2].strip() for row in tab_rows}
    # ECMA-402 TimeZonesOfLocale: the zones in common use in each ISO 3166-1 region.
    region_zones = {}
    for row in tab_rows:
        region_zones.setdefault(row[0].strip(), set()).add(row[2].strip())

    text = open(path).read()
    zstart = text.index("pub static ZONES")
    zend = text.index("];", zstart) + 2
    lstart = text.index("pub static LINKS")
    lend = text.index("];", lstart) + 2
    old_zones = {
        name: (initial, table)
        for name, initial, table in re.findall(
            r'Zone \{ name: "([^"]+)", initial: (-?\d+), transitions: (\w+) \}',
            text[zstart:zend],
        )
    }
    old_links = dict(re.findall(r'\("([^"]+)", "([^"]+)"\)', text[lstart:lend]))

    tables = dict(re.findall(r"static (TZ_\w+): &\[\(i64, i32\)\] = &\[(.*?)\];", text))

    def data(name):
        while name in old_links:
            name = old_links[name]
        return old_zones[name]

    def offsets(name):
        initial, table = data(name)
        return initial, tables[table]

    known = set(old_zones) | set(old_links)
    primary = ((zones | tab | {"UTC"}) - UTC_ALIASES) & known

    def target(name):
        if name in UTC_ALIASES or links.get(name) in UTC_ALIASES:
            return "UTC"
        for candidate in (annotated.get(name), SAME_COUNTRY.get(name)):
            if candidate in primary:
                return candidate
        t = links.get(name)
        while t in links:
            t = links[t]
        if t in primary:
            return t
        t = old_links.get(name)
        while t in old_links and t not in primary:
            t = old_links[t]
        return t

    new_links = {}
    for name in sorted(known - primary):
        t = target(name)
        if t not in primary or offsets(t) != offsets(name):
            sys.exit(f"{name}: no primary with the same offsets ({t})")
        new_links[name] = t

    zone_lines = "".join(
        f'    Zone {{ name: "{n}", initial: {data(n)[0]}, transitions: {data(n)[1]} }},\n'
        for n in sorted(primary)
    )
    link_lines = "".join(f'    ("{a}", "{b}"),\n' for a, b in sorted(new_links.items()))
    zhead = text[zstart : text.index("\n", zstart) + 1]
    lhead = text[lstart : text.index("\n", lstart) + 1]
    out = (
        text[:zstart]
        + zhead
        + zone_lines
        + "];"
        + text[zend:lstart]
        + lhead
        + link_lines
        + "];"
        + text[lend:]
    )
    region_lines = "".join(
        f'    ("{region}", &[{", ".join(f"{chr(34)}{z}{chr(34)}" for z in sorted(zones))}]),\n'
        for region, zones in sorted(region_zones.items())
    )
    region_table = (
        "/// (ISO 3166-1 region, the primary zones its zone.tab rows list), sorted by region and then\n"
        "/// zone, for ECMA-402 TimeZonesOfLocale.\n"
        "#[cfg(feature = \"intl\")]\n"
        "pub static REGION_ZONES: &[(&str, &[&str])] = &[\n" + region_lines + "];\n"
    )
    if "pub static REGION_ZONES" in out:
        rstart = out.index("/// (ISO 3166-1 region")
        rend = out.index("];", out.index("pub static REGION_ZONES")) + 3
        out = out[:rstart] + region_table + out[rend:]
    else:
        out = out.rstrip("\n") + "\n\n" + region_table
    # Drop offset tables no primary identifier uses any more.
    used = {data(n)[1] for n in primary}
    out = re.sub(
        r"static (TZ_\w+): &\[\(i64, i32\)\] = &\[.*?\];\n",
        lambda m: m.group(0) if m.group(1) in used else "",
        out,
    )
    open(path, "w").write(out)
    print(f"{len(primary)} primary identifiers, {len(new_links)} links")


if __name__ == "__main__":
    main()
