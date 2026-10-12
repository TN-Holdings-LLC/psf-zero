"""Builds prototype 2026-10-11.c44 from prototype 2026-10-11.c43 (= release 2026-10-11.4 but its version lines) by exact
substitutions:
  - item 75, revised: item 75's search runs in c43's index order, in an order by the qubits' own errors (best first)
    and the circuit's two-qubit gate counts, and in rustworkx's own VF2 order, each with the trial limit; the lowest
    score of all is kept (C43-VAL: with VF2's index order alone, 2500 trials missed the
    placements Qiskit's pass found on the largest random circuit);
  - item 77: the layout search has no time budget by default (`layout_search_time_budget_s=None`): its work is bounded
    by call limits only, so its result does not depend on the machine's speed.
    python make_c44.py <c43 psf_compile.py> <output psf_compile.py>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

OLD_SEARCH = '''            dg, cg = rx.PyDiGraph(), rx.PyDiGraph()
            dg.add_nodes_from(range(N))
            dg.add_edges_from_no_data(sorted(dev_edges))
            cg.add_nodes_from(conn)
            cedges = {(pos[a], pos[b]) for _, (a, b) in two}
            if not self.strict:
                cedges |= {(b, a) for a, b in cedges}
            cg.add_edges_from_no_data(sorted(cedges))
            for mapping in rx.vf2_mapping(dg, cg, subgraph=True, induced=False, id_order=True,
                                          call_limit=self.call_limit):
                trials += 1
                mc = {conn[cn]: dn for dn, cn in mapping.items()}
                sc = score(mc)
                if sc < best:
                    mi, total = place_isolated(set(mc.values()), sc)
                    if total < best - 1e-12:
                        best, best_map = total, {**mc, **mi}
                if trials >= self.max_trials:
                    break'''

NEW_SEARCH = '''            # item 75 (revised in c44): c43's index order, then the device's qubits best first and the circuit's
            # busiest first, then VF2's own order; each up to the trial limit
            own = {}
            for name in [n for n in ("sx", "measure") if n in t.operation_names]:
                for qargs, props in t[name].items():
                    if qargs is None or props is None or props.error is None:
                        continue
                    e = props.error
                    v = -math.log1p(-e) if e < 1.0 else 1e3
                    own[qargs[0]] = own.get(qargs[0], 0.0) + v
            nbr = {p: [] for p in range(N)}
            for a, b in dev_edges:
                v = w(names2[0], (a, b))
                nbr[a].append(v if v < math.inf else 1e3)
                nbr[b].append(v if v < math.inf else 1e3)
            quality = sorted(range(N), key=lambda p: (own.get(p, 0.0) + (sum(nbr[p]) / len(nbr[p]) if nbr[p] else 1e3), p))
            load = {}
            for (_, (a, b)), c in two.items():
                load[a] = load.get(a, 0) + c
                load[b] = load.get(b, 0) + c
            busy = sorted(conn, key=lambda q: (-load.get(q, 0), q))
            cedges0 = {(a, b) for _, (a, b) in two}
            if not self.strict:
                cedges0 |= {(b, a) for a, b in cedges0}
            for phys_order, circ_order, id_order in ((list(range(N)), conn, True), (quality, busy, True),
                                                     (list(range(N)), conn, False)):
                at_d = {p: i for i, p in enumerate(phys_order)}
                at_c = {q: i for i, q in enumerate(circ_order)}
                dg, cg = rx.PyDiGraph(), rx.PyDiGraph()
                dg.add_nodes_from(phys_order)
                dg.add_edges_from_no_data(sorted((at_d[a], at_d[b]) for a, b in dev_edges))
                cg.add_nodes_from(circ_order)
                cg.add_edges_from_no_data(sorted((at_c[a], at_c[b]) for a, b in cedges0))
                n_here = 0
                for mapping in rx.vf2_mapping(dg, cg, subgraph=True, induced=False, id_order=id_order,
                                              call_limit=self.call_limit):
                    trials += 1
                    n_here += 1
                    mc = {circ_order[cn]: phys_order[dn] for dn, cn in mapping.items()}
                    sc = score(mc)
                    if sc < best:
                        mi, total = place_isolated(set(mc.values()), sc)
                        if total < best - 1e-12:
                            best, best_map = total, {**mc, **mi}
                    if n_here >= self.max_trials:
                        break'''

for old, new in [
    ("VERSION: 2026-10-11.c43 -- prototype (from c42; release 2026-10-11.3 = c41): items 69-76",
     "VERSION: 2026-10-11.c44 -- prototype (from c43 = release 2026-10-11.4): items 75 revised, 77"),
    ('VERSION = "2026-10-11.c43"  # prototype: release 2026-10-11.3 + items 74-76 (near-failed elements the output '
     'used; a deterministic re-placement; permutations left at the end)',
     'VERSION = "2026-10-11.c44"  # prototype: release 2026-10-11.4 + item 75 revised (search order) + item 77 (no '
     'time budget in the layout search)'),
    (OLD_SEARCH, NEW_SEARCH),
    ("    layout_search_time_budget_s: float = 2.0,", "    layout_search_time_budget_s: float | None = None,"),
    ('''            time_budget_s=layout_search_time_budget_s,''',
     '''            time_budget_s=1e9 if layout_search_time_budget_s is None else layout_search_time_budget_s,  # 77'''),
    ('''    item 70 applies after merging.''', '''    item 70 applies after merging.'''),
]:
    assert s.count(old) == 1, old[:90]
    s = s.replace(old, new)

CHANGELOG_ANCHOR = '''    gate, it is carried into the measurements and the output's final layout. Counts in `STEINER_STATS` ("permuted").
"""'''
assert s.count(CHANGELOG_ANCHOR) == 1
s = s.replace(CHANGELOG_ANCHOR, '''    gate, it is carried into the measurements and the output's final layout. Counts in `STEINER_STATS` ("permuted").
    **Item 75 revised in prototype c44:** the search runs three times, each up to the trial limit: in c43's index order;
    with the device's qubits ordered by their own error (sx and readout plus the mean of their couplers', best first)
    and the circuit's by two-qubit gate count (busiest first); and in rustworkx's own VF2 order (`id_order=False`). The
    lowest score of all is applied if below the current placement's. Each order finds placements the others miss. C43-VAL: with the index order alone, 2500
    trials missed the placements Qiskit's pass found on a 48-qubit random circuit (0.48-0.90 of the release's ESP).
77. **REPRODUCIBILITY: no time budget in the layout search (prototype c44).** `layout_search_time_budget_s` now defaults
    to None: the layout search stops by its call limits only (`layout_search_call_limit` per ordering,
    `layout_search_fallback_call_limit` for VF2's heuristic), so its result no longer depends on the machine's speed. A
    number restores the time budget. Measured before the change: of 154 compiles, one (dtc_100 on FakeTorino) reached
    the 2 s budget, without a layout either way (5.4 s without the budget). The packing search for paired and 3-qubit
    path interactions (psf_smart_layout) keeps its own short time limit; it did not bind in those 154 compiles.
"""''')
open(dst, "w", encoding="utf-8", newline="\n").write(s)
print("written", dst)
