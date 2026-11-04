#!/usr/bin/env python3
"""Join two feedback-census files into coverage by operation kind (W4; DIAGNOSTICS.md).

usage: python3 -I scripts/feedback-census.py BYTECODE_FILE JIT_FILE [--name NAME] [--records LIST]

BYTECODE_FILE is TRUST_JS_FEEDBACK_CENSUS's file from a --tier=bytecode run of a program: every
compiled-code operation passes the VM's loop there, so its dispatch counts are each site's
executions. JIT_FILE is the file from the production --tier=jit run of the same program: each
site's production feedback at exit. Both come from an engine built with the
architecture-diagnostics feature. Chunks join on their census key (not inlined). --records
(e.g. `values`, or `none`) reads the files as if the engine wrote only those records.

For each kind it prints the executions, their share of all operations, the share whose site has
usable feedback in production today, and coverage (feedback-schema.txt, section 4): the share of
executions whose site the speculative tier could compile without an exit for missing feedback.
Where the engine writes the always-on records (the header's `records=`), it also prints precision:
the share whose record names exactly the coarse classes the operation saw, the bytecode run's
record (the VM records every execution) being what it saw. A value operation's record is read with
its in-line set (rule U, feedback-schema.txt section 5.1), an element access's alone when it is not
empty (rule R, section 5.2).
Branches, loops, `!` and names are counted but left out of the totals (no exemplar records
branch directions or ToBoolean; names mostly resolve to static addresses). Equality (==, !=, ===,
!==) has no record by design (the plan's W4 decision 3: typed from its operands, as JSC does): it
is in the comparison row and the first total, and a second total leaves it out.
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
    """Production feedback the census's state word shows (bytecode_census.rs) from the baseline's
    caches (no record's)."""
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
# The element record's sets (section 5.2): receivers in bits 0-6, keys in bits 8-12, an Array's
# facts in bits 16-20; the in-line set's receivers (an Array, a typed array, another ordinary
# object) and key (an array index).
ELEMENT_OPS = ("GetElem", "GetElemLocal", "GetMethodElem", "SetElem", "SetElemDrop",
               "SetElemLocal", "SetElemLocalDrop")
E_RECEIVERS, E_KEYS = 0x7F, 0x1F << 8
E_IN_LINE_RECEIVERS, E_KEY_INDEX = 0b111, 1 << 8
# A named site's generic mark in its state word (step 5).
GENERIC = 1 << 7


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


def element_precise(full, recorded):
    """Rule R: the record's receivers and keys alone when it holds any, else the in-line set's,
    are exactly those seen (an Array's facts are hints, not compared)."""
    if recorded & (E_RECEIVERS | E_KEYS):
        receivers, keys = recorded & E_RECEIVERS, recorded & E_KEYS
    else:
        receivers, keys = E_IN_LINE_RECEIVERS, E_KEY_INDEX
    return receivers == full & E_RECEIVERS and keys == full & E_KEYS


def precise_site(kind, op, state, full, recorded, records):
    """Precision's sense (feedback-schema.txt section 4): a cache that holds the site's targets or
    shapes, or a record that, read by its rule, names exactly the coarse classes seen. typeof
    and `in` record nothing, nor does equality (W4's decision 3)."""
    name = name_of(op)
    if kind in (1, 2) and "generic" in records and state & GENERIC:
        return True  # megamorphic or uncacheable: exactly what P3 lowers generically
    if kind in (1, 2, 5, 6, 10) or (kind == 13 and name == "InstanceOf"):
        return usable_today(kind, op, state)
    inline = in_line_set(op)
    if "values" in records and inline is not None:
        return value_precise(full, recorded, inline)
    if "elements" in records and name in ELEMENT_OPS:
        return element_precise(full, recorded)
    return False


def usable(kind, op, state, records):
    """Coverage's sense of usable: today's caches, plus what the records that the engine writes
    (the census header's `records=` list) make usable."""
    if usable_today(kind, op, state):
        return True
    name = name_of(op)
    if kind in (1, 2) and "generic" in records and state & GENERIC:
        return True  # the generic mark (step 5): a refused fill or a fifth receiver shape
    if kind == 13 and name in ("Typeof", "TypeofIs", "TypeofName", "In"):
        return True  # lowered generically by the templates; no exit for missing feedback
    if "values" in records and (in_line_set(op) is not None or name in EQUALITY):
        # The record with the in-line set; equality is lowered by tag, with no record (W4's
        # decision 3), as the templates lower it.
        return True
    if "elements" in records and name in ELEMENT_OPS:
        return True  # the record with the in-line set's two-way dispatch, as the templates'
    return False


def main():
    argv = sys.argv[1:]
    options = {}
    for flag in ("--name", "--records"):
        if flag in argv:
            k = argv.index(flag)
            options[flag] = argv[k + 1]
            del argv[k:k + 2]
    args = argv
    name = options.get("--name", args[0])
    vm_header, _, vm = load(args[0])
    jit_header, jit_chunks, jit = load(args[1])
    records = set()
    for word in jit_header:
        if word.startswith("records="):
            records = set(filter(None, word[len("records="):].split(",")))
    if "--records" in options:
        records = set(filter(None, options["--records"].split(","))) - {"none"}
    total, today, covered, joined, precise = (defaultdict(int) for _ in range(5))
    equality = defaultdict(int)  # executions, usable, precise of equality (in comparison)
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
            equality["precise"] += count if name_of(o["op"]) in EQUALITY else 0
        if name_of(o["op"]) in EQUALITY:
            equality["total"] += count
            equality["covered"] += count if usable(kind, o["op"], state, records) else 0
            equality["today"] += count if usable_today(kind, o["op"], state) else 0
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
        n = counted - equality["total"]
        if n and equality["total"]:
            t -= equality["today"]
            c -= equality["covered"]
            q -= equality["precise"]
            print(f"  and without equality\t{n}\t{100 * n / everything:.2f}\t{100 * t / n:.1f}"
                  f"\t{100 * c / n:.1f}\t\t{100 * q / n:.1f}")


main()
