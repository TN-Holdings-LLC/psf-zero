"""verify_feasibility_fix.py -- Addendum 89's end-to-end test.

Unlike every harness since Addendum 83, this one calls
`smart_vf2_layout()` **directly, with no bypass** -- the whole point is
to confirm the `_has_feasible_matching` fix lets the real, public entry
point work on chain-shaped circuits, which it could not before.

Three checks, matching Addendum 89's pre-registered predictions:

  P1 -- the 26 `dominant_size_sweep` configurations (chain-shaped, the
        ones the old guard rejected outright) now reach the search and
        find layouts, matching Addendum 88's bypassed numbers.
  P2 -- `dense_pairs` (the family the original guard was designed for)
        behaves identically to before at both 6x7 and 8x8: a fix that
        quietly changed the original design case would be a regression.
  P3 -- a genuinely infeasible input is STILL rejected by the guard
        before searching. A "fix" that always returned True would pass
        P1 and P2 while destroying the guard's purpose, so this is
        tested explicitly.

Every returned layout is also validated structurally (each interaction
pair must land on a physically adjacent qubit pair) -- a fast result
that is not a valid layout would be a regression, not a fix.

## Requirements

`psf_smart_layout_patched.py` (carrying both the Addendum 88
natural-first ordering and the Addendum 89 guard fix) in the same
directory, plus `qiskit`, `rustworkx`, `networkx`.

Usage:
    python verify_feasibility_fix.py
"""
from __future__ import annotations

import csv
import time

from qiskit.transpiler import CouplingMap

from psf_smart_layout_patched import smart_vf2_layout


def _edges_dominant_size_sweep(n, n_bare=17, dominant_size=30):
    bare_qubits = n_bare * 2
    remaining = n - bare_qubits
    filler_size = remaining - dominant_size
    if filler_size < 0 or (0 < filler_size < 2) or dominant_size < 2:
        raise ValueError(f"invalid dominant_size={dominant_size}")
    edges = []
    q = 0
    for _ in range(n_bare):
        edges.append((q, q + 1))
        q += 2
    for i in range(dominant_size - 1):
        edges.append((q + i, q + i + 1))
    q += dominant_size
    if filler_size > 0:
        for i in range(filler_size - 1):
            edges.append((q + i, q + i + 1))
    return edges


def _edges_dense_pairs(n):
    return [(i, i + 1) for i in range(0, n - 1, 2)]


D_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]


def validate_layout(layout_map, interaction_pairs, coupling_map):
    """Every interaction pair must map to a physically adjacent pair."""
    if layout_map is None:
        return None
    phys = set()
    for a, b in coupling_map.get_edges():
        phys.add((a, b))
        phys.add((b, a))
    for a, b in interaction_pairs:
        if a not in layout_map or b not in layout_map:
            return False
        if (layout_map[a], layout_map[b]) not in phys:
            return False
    return True


def main():
    rows = []

    # ---------------- P1 ----------------
    print("=" * 96)
    print("Addendum 89 / P1 -- chain-shaped circuits, via the REAL "
          "smart_vf2_layout (no bypass)")
    print("=" * 96)
    cm8 = CouplingMap.from_grid(8, 8)
    print(f"{'D':>4} {'feasible':>9} {'found':>7} {'time_s':>10} "
          f"{'tried':>6} {'order':>10} {'valid':>6}")
    for D in D_VALUES:
        edges = _edges_dominant_size_sweep(64, dominant_size=D)
        t0 = time.perf_counter()
        layout, info = smart_vf2_layout(cm8, edges, 64)
        el = time.perf_counter() - t0
        valid = validate_layout(layout, edges, cm8)
        print(f"{D:>4} {str(info.get('feasible')):>9} "
              f"{str(layout is not None):>7} {el:>10.4f} "
              f"{info.get('orderings_tried'):>6} "
              f"{str(info.get('order_name')):>10} {str(valid):>6}")
        rows.append(dict(check="P1", case=f"dominant_size_sweep_D{D}",
                         feasible=info.get("feasible"),
                         found=layout is not None, elapsed_s=el,
                         orderings_tried=info.get("orderings_tried"),
                         order=info.get("order_name"), layout_valid=valid))

    # ---------------- P2 ----------------
    print()
    print("=" * 96)
    print("Addendum 89 / P2 -- dense_pairs (the family the original guard "
          "was designed for): unchanged?")
    print("=" * 96)
    print(f"{'grid':>6} {'feasible':>9} {'found':>7} {'time_s':>10} "
          f"{'tried':>6} {'order':>20} {'valid':>6}")
    for rows_n, cols_n in [(6, 7), (8, 8)]:
        n = rows_n * cols_n
        cm = CouplingMap.from_grid(rows_n, cols_n)
        edges = _edges_dense_pairs(n)
        t0 = time.perf_counter()
        layout, info = smart_vf2_layout(cm, edges, n)
        el = time.perf_counter() - t0
        valid = validate_layout(layout, edges, cm)
        print(f"{f'{rows_n}x{cols_n}':>6} {str(info.get('feasible')):>9} "
              f"{str(layout is not None):>7} {el:>10.4f} "
              f"{info.get('orderings_tried'):>6} "
              f"{str(info.get('order_name')):>20} {str(valid):>6}")
        rows.append(dict(check="P2", case=f"dense_pairs_{rows_n}x{cols_n}",
                         feasible=info.get("feasible"),
                         found=layout is not None, elapsed_s=el,
                         orderings_tried=info.get("orderings_tried"),
                         order=info.get("order_name"), layout_valid=valid))

    # ---------------- P3 ----------------
    print()
    print("=" * 96)
    print("Addendum 89 / P3 -- is a genuinely impossible input STILL "
          "rejected before searching?")
    print("=" * 96)
    cm4 = CouplingMap.from_grid(4, 4)   # 16 qubits, max matching 8
    impossible = _edges_dense_pairs(64)  # 32 disjoint pairs -- cannot fit
    t0 = time.perf_counter()
    layout, info = smart_vf2_layout(cm4, impossible, 64)
    el = time.perf_counter() - t0
    rejected = (info.get("feasible") is False and layout is None)
    print(f"  4x4 grid (max matching 8) asked for 32 disjoint pairs:")
    print(f"    feasible={info.get('feasible')}, found={layout is not None}, "
          f"orderings_tried={info.get('orderings_tried')}, time={el:.4f}s")
    print(f"    -> guard still rejects before searching: {rejected} "
          f"(True is correct)")
    rows.append(dict(check="P3", case="impossible_4x4_32pairs",
                     feasible=info.get("feasible"),
                     found=layout is not None, elapsed_s=el,
                     orderings_tried=info.get("orderings_tried"),
                     order=info.get("order_name"), layout_valid=None))

    # ---------------- summary ----------------
    p1 = [r for r in rows if r["check"] == "P1"]
    p2 = [r for r in rows if r["check"] == "P2"]
    n_p1_found = sum(1 for r in p1 if r["found"])
    n_p1_valid = sum(1 for r in p1 if r["layout_valid"] is True)
    n_p1_first = sum(1 for r in p1 if r["orderings_tried"] == 1
                     and r["order"] == "natural")
    print()
    print("=" * 96)
    print(f"P1 -- found {n_p1_found}/{len(p1)}, structurally valid "
          f"{n_p1_valid}/{len(p1)}, solved on first (natural) ordering "
          f"{n_p1_first}/{len(p1)}")
    print(f"P2 -- dense_pairs found {sum(1 for r in p2 if r['found'])}/{len(p2)}, "
          f"valid {sum(1 for r in p2 if r['layout_valid'] is True)}/{len(p2)}")
    print(f"P3 -- impossible input rejected before searching: {rejected}")
    print("=" * 96)

    out = "feasibility_fix_verification_2026-09-19.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
