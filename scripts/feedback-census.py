#!/usr/bin/env python3
"""Join two feedback-census files into coverage by operation kind (W4; DIAGNOSTICS.md).

usage: python3 -I scripts/feedback-census.py BYTECODE_FILE JIT_FILE [--name NAME]

BYTECODE_FILE is TRUST_JS_FEEDBACK_CENSUS's file from a --tier=bytecode run of a program: every
compiled-code operation passes the VM's loop there, so its dispatch counts are each site's
executions. JIT_FILE is the file from the production --tier=jit run of the same program: each
site's production feedback at exit. Both come from an engine built with the
architecture-diagnostics feature. Chunks join on their census key (not inlined).

For each kind it prints the executions, their share of all operations, the share whose site has
usable feedback in production today, and coverage (feedback-schema.txt, section 4): the share of
executions whose site the speculative tier could compile without an exit for missing feedback.
Where the engine writes the always-on records (the header's `records=`), it also prints precision:
the share whose record, read with its operation's in-line set (rule U, feedback-schema.txt
section 5), names exactly the coarse classes the operation saw, the bytecode run's record (the VM
records every execution) being what it saw.
Branches, loops, `!` and names are counted but left out of the totals (no exemplar records
branch directions or ToBoolean; names mostly resolve to static addresses).
"""

import sys
from collections import defaultdict

KINDS = {1: "named load", 2: "named store", 3: "element load", 4: "element store", 5: "call",
         6: "construct", 7: "arithmetic", 8: "branch", 9: "loop", 10: "allocation",
         11: "comparison", 12: "name", 13: "typeof/in/instanceof", 14: "!"}
ORDER = [1, 2, 3, 4, 5, 6, 7, 11, 10, 13, 12, 14, 8, 9]
APART = {8, 9, 12, 14}
CACHED_CALLS = ("Call", "CallWithThis", "CallSpread", "CallSpreadThis")
# Allocations that need no per-site state (V8's closures and empty literals need none either).
STATELESS_ALLOCATIONS = ("MakeClosure", "NewObject", "NewArray")


def load(path):
    """(header words, {(key, inlined): chunk}, {(key, inlined, pc): op})"""
    header, chunks, ops = [], {}, {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                header = line[1:].split()
                continue
            p = line.split("\t")
            if p[0] == "C":
                chunks[(p[1], p[2])] = dict(chunks=int(p[3]), native=int(p[4]), label=p[5])
            elif p[0] == "O":
                ops[(p[1], p[2], int(p[3]))] = dict(kind=int(p[4]), count=int(p[5]),
                                                    state=int(p[6], 16), records=int(p[7]),
                                                    record=int(p[8], 16), op=p[9])
    return header, chunks, ops


def name_of(op):
    return op.split("(")[0]


def usable_today(kind, op, state):
    """Production feedback the census's state word shows (bytecode_census.rs)."""
    name = name_of(op)
    if kind in (1, 2):
        return (state & 7) > 0
    if kind == 5:
        return name in CACHED_CALLS and bool((state & 7) or state & 8)
    if kind == 6:
        return name == "New" and bool(state & 1)
    if kind == 10:
        return name in STATELESS_ALLOCATIONS or bool(state & 1)
    if kind == 12:
        return bool(state & 1)
    if kind == 13:
        return name == "InstanceOf" and bool(state & 1)
    return False


# ValueClass bits (feedback-schema.txt section 2) and the value record's roles (section 5).
INT32, DOUBLE = 1 << 3, 1 << 4
NUMBER = INT32 | DOUBLE
UNARY_UPDATES = ("Neg", "Plus", "BitNot", "UpdateLocal", "UpdateCap", "UpdateName",
                 "UpdateNameCached", "UpdateConst", "UpdatePrivate", "SuperUpdate", "UpdateProp",
                 "UpdateElem")
RELATIONAL = ("Lt", "Gt", "Le", "Ge")
EQUALITY = ("EqEq", "NotEq", "StrictEq", "StrictNotEq")


def in_line_set(op):
    """(operand 0, operand 1, result): the classes the operation's in-line set holds, which
    generated code takes without a record (feedback-schema.txt section 5); None where the
    operation is not a value record's."""
    name = name_of(op)
    if name == "Mod":
        return (INT32, INT32, INT32)
    if name in RELATIONAL:
        return (NUMBER, NUMBER, 0)
    if name in UNARY_UPDATES:
        return (NUMBER, 0, NUMBER)
    if name in ("Add", "Sub", "Mul", "Div", "UShr", "BitAnd", "BitOr", "BitXor", "Shl", "Shr",
                "GenBin"):
        return (NUMBER, NUMBER, NUMBER)
    return None


def roles(word):
    return (word & 0x1FF, (word >> 9) & 0x1FF, (word >> 18) & 0x1FF)


def coarse(bits):
    """int32 and double merged into Number."""
    return (bits & ~NUMBER) | (INT32 if bits & NUMBER else 0)


def value_precise(full, recorded, inline):
    """Rule U: the record plus the in-line set names exactly the coarse classes seen."""
    for f, g, i in zip(roles(full), roles(recorded), inline):
        if f and coarse(g | i) != coarse(f):
            return False
    return True


def precise_site(kind, op, state, full, recorded, records):
    """Precision's sense (feedback-schema.txt section 4): a cache that holds the site's targets or
    shapes, or a record that, with its in-line set, names exactly the coarse classes seen. typeof
    and `in` record nothing, nor does equality (W4's decision 3)."""
    name = name_of(op)
    if kind in (1, 2, 5, 6, 10) or (kind == 13 and name == "InstanceOf"):
        return usable_today(kind, op, state)
    inline = in_line_set(op)
    if "values" in records and inline is not None:
        return value_precise(full, recorded, inline)
    return False


def usable(kind, op, state, records):
    """Coverage's sense of usable: today's caches, plus what the records that the engine writes
    (the census header's `records=` list) make usable."""
    if usable_today(kind, op, state):
        return True
    name = name_of(op)
    if kind == 13 and name in ("Typeof", "TypeofIs", "TypeofName", "In"):
        return True  # lowered generically by the templates; no exit for missing feedback
    if "values" in records and (in_line_set(op) is not None or name in EQUALITY):
        # The record with the in-line set; equality is lowered by tag, with no record (W4's
        # decision 3), as the templates lower it.
        return True
    return False


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else args[0]
    vm_header, _, vm = load(args[0])
    jit_header, jit_chunks, jit = load(args[1])
    records = set()
    for word in jit_header:
        if word.startswith("records="):
            records = set(filter(None, word[len("records="):].split(",")))
    total, today, covered, joined, precise = (defaultdict(int) for _ in range(5))
    for (key, inlined, pc), o in vm.items():
        count, kind = o["count"], o["kind"]
        if not count or kind == 0:
            continue
        if kind == 13 and name_of(o["op"]) == "Not":
            kind = 14  # ToBoolean: no exemplar records it (A4), so it stays apart
        total[kind] += count
        j = jit.get((key, "0", pc))
        if (key, "0") in jit_chunks:
            joined[kind] += count
        state = j["state"] if j else 0
        if usable_today(kind, o["op"], state):
            today[kind] += count
        if usable(kind, o["op"], state, records):
            covered[kind] += count
        if precise_site(kind, o["op"], state, o["record"], j["record"] if j else 0, records):
            precise[kind] += count
    everything = sum(total.values())
    counted = sum(v for k, v in total.items() if k not in APART)
    print(f"== {name} (records: {', '.join(sorted(records)) or 'none'})")
    print("kind\texecutions\tshare %\tusable today %\tcoverage %\tjoined %\tprecise %")
    for kind in ORDER:
        n = total[kind]
        if not n:
            continue
        p = f"{100 * precise[kind] / n:.1f}"
        print(f"{KINDS[kind]}\t{n}\t{100 * n / everything:.2f}\t{100 * today[kind] / n:.1f}"
              f"\t{100 * covered[kind] / n:.1f}\t{100 * joined[kind] / n:.1f}\t{p}"
              + ("\t(apart)" if kind in APART else ""))
    if counted:
        t = sum(v for k, v in today.items() if k not in APART)
        c = sum(v for k, v in covered.items() if k not in APART)
        q = sum(v for k, v in precise.items() if k not in APART)
        print(f"all but branches, loops and names\t{counted}\t{100 * counted / everything:.2f}"
              f"\t{100 * t / counted:.1f}\t{100 * c / counted:.1f}\t\t{100 * q / counted:.1f}")


main()
