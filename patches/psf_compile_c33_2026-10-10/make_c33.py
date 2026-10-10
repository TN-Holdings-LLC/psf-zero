"""Builds candidate 2026-10-10.c33 from candidate 2026-10-10.c32 by exact substitutions."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)


sub("VERSION: 2026-10-10.c32 -- candidate (from candidate 2026-10-10.c31): items 58, 59a and 60",
    "VERSION: 2026-10-10.c33 -- candidate (from candidate 2026-10-10.c32): items 58, 59a, 60 and 61")
sub('VERSION = "2026-10-10.c32"  # candidate: c31 (release 2026-10-10.2 + items 58, 59a) + item 60 (the default call places by the device\'s errors when it has a Target)',
    'VERSION = "2026-10-10.c33"  # candidate: c32 (release 2026-10-10.2 + items 58, 59a, 60) + item 61 (with a Target, one compile on the map without the failed elements)')
sub("""    Target, and calls that pass `placement_refine` explicitly, are unchanged.
\"\"\"""", """    Target, and calls that pass `placement_refine` explicitly, are unchanged.
61. **SPEED: with a Target, one compile on the map without the failed elements (candidate 2026-10-10.c33).** Item 31
    compiled on the full map first and, when the result used a failed element, compiled again on the pruned map: on
    FakeTorino 63 of BP-FINAL's 106 tests were compiled twice (ESP-C31). Now, when the Target reports failed elements,
    the circuit is compiled once, on `prune_coupling_map(...)`, and checked as item 58 checks a recompile. When no
    placement exists there, "raise" raises FailedElementsError; "keep" compiles on the full map, with item 43's
    warning. `PRUNE_FIRST = False` restores c32's order. The output can differ from c32's where c32's first compile
    used no failed element (the router then saw the whole map).
\"\"\"""")
sub('''               "unavoidable": 0, "still_failed": 0, "raised": 0}''',
    '''               "unavoidable": 0, "still_failed": 0, "raised": 0, "pruned_first": 0}
PRUNE_FIRST = True  # item 61: with failed elements in the Target, compile once on the pruned map''')
sub("""        out = compile_for_hardware(**args)
        edges, qubits = _failed_elements(target, prune_max_error)
        PRUNE_STATS["checked"] += 1
        if _uses_failed(out, edges, qubits):
            PRUNE_STATS["recompiled"] += 1
            args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
            out = _recompile_pruned(args, out, edges, qubits, on_failed_elements)  # item 58""",
    """        edges, qubits = _failed_elements(target, prune_max_error)
        PRUNE_STATS["checked"] += 1
        if PRUNE_FIRST and (edges or qubits):  # item 61: one compile, on the map without the failed elements
            PRUNE_STATS["pruned_first"] += 1
            args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
            out = _recompile_pruned(args, None, edges, qubits, on_failed_elements)
            if out is None:  # "keep", and no placement on the pruned map: the full map, as item 43 did
                args["coupling_map"] = coupling_map
                out = compile_for_hardware(**args)
        else:
            out = compile_for_hardware(**args)
            if _uses_failed(out, edges, qubits):
                PRUNE_STATS["recompiled"] += 1
                args["coupling_map"] = prune_coupling_map(coupling_map, target, prune_max_error)
                out = _recompile_pruned(args, out, edges, qubits, on_failed_elements)  # item 58""")
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
