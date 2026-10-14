"""make_c53_91.py -- builds the workplace's proposal of item 91 on prototype 2026-10-14.c53 by exact substitutions:
a Pauli evolution whose interaction graph embeds in the device is compiled by formula (first VF2 embedding, no score;
edges coloured into at most three matchings, one layer per colour). It reuses items 87-88 (placement rule, levels,
rzz / rx with the Pauli frame) and goes through item 87's comparison with c53's own output. Every other input is
compiled exactly as by c53. The code is appended at the end of the file; the wrapper gets two lines.
    python make_c53_91.py <c53 psf_compile.py> <c53_91_code.py> <output psf_compile.py>"""
import sys

src, code91, dst = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
add = open(code91, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    k = s.count(old)
    if k != count:
        raise SystemExit(f"expected {count} occurrence(s), found {k}: {old[:80]!r}")
    s = s.replace(old, new)


# 1. version (home: set to its own strings when merging)
sub("VERSION: 2026-10-14.c53 -- prototype (from c52 + the workplace's items 87-88; c43 = release 2026-10-11.4): items "
    "75 revised, 77-80, 82-90",
    "VERSION: 2026-10-14.c53+91 -- workplace proposal on prototype c53 (c43 = release 2026-10-11.4): items 75 "
    "revised, 77-80, 82-91")
sub('VERSION = "2026-10-14.c53"  # prototype:',
    'VERSION = "2026-10-14.c53+91"  # workplace proposal, item 91 (embeddable graphs by edge colouring) on prototype:')
# 2. changelog, after item 90
sub('''    excludes. `MAP_FREE_DEFAULTS = False` and `PLACE_SCORE = "esp"` restore c51.
"""''', '''    excludes. `MAP_FREE_DEFAULTS = False` and `PLACE_SCORE = "esp"` restore c51.
91. **DURATION: an interaction graph that embeds in the device, built by edge colouring (workplace proposal Q1, on
    c53).** Item 87 generalised from a path to any graph heavy-hex contains. With item 87's conditions (device given,
    cz, none of the caller's coupling_map / initial_layout / final_resynthesis / compare_level3 / compare_floor) and an
    input whose terms (items 72-73's reader, merged where exact) act on one or two qubits but are not one chain: the
    interaction graph (max degree 3, bipartite) is placed by the first embedding rustworkx's VF2 finds in the cut
    device (nodes in index order, no score), first on the couplers of the device's most common gate length, then on
    all of them. Its edges are coloured with max-degree colours (Koenig; deterministic alternating-path colouring,
    at most 3 on heavy-hex), each colour a matching. Every term keeps item 87's level (1 + the largest level of an
    earlier term that shares a qubit and does not commute with it: exact); within a level the two-qubit terms go by
    colour, one layer per colour, then the single-qubit terms. Emitted as item 87 (Qiskit level 3 on the fixed layout,
    CZ) or, with rzz and rx in the Target, item 88 (rzz, rx, Pauli frame undone inside the circuit), and compared with
    c53's own output by item 87's rule. Heavy-hex never has two degree-3 qubits side by side and has no cycle shorter
    than 12, so a graph with adjacent degree-3 vertices (most random trees), a cycle of length 4-11, or a vertex of
    degree 4 (a square lattice, C52-VAL's heis2d) does not embed: then nothing changes. `GRAPH_BRICKWORK = False`
    turns it off. Counts in `GRAPH_STATS`.
"""''')
# 3. the wrapper: item 91 after item 87
sub('''        out87 = _chain_brickwork(qc, target, a["prune_max_error"], a["seed_transpiler"])
''', '''        out87 = _chain_brickwork(qc, target, a["prune_max_error"], a["seed_transpiler"])
        if out87 is None and GRAPH_BRICKWORK:  # item 91
            out87 = _graph_brickwork(qc, target, a["prune_max_error"], a["seed_transpiler"])
''')
s = s.rstrip("\n") + "\n" + add
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
