#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A smoke test for psf_smart_layout.smart_vf2_layout() (not used for the actual verdict).

Under exactly the condition already confirmed in addendum-9/10 -- "on a grid
at spare=0, VF2Layout returns NO_SOLUTION_FOUND both on real hardware and in
the sandbox" -- this checks:

  1. That Qiskit's default pipeline (transpile + callback) does fail at
     VF2Layout (re-confirming the premise).
  2. That psf_smart_layout.smart_vf2_layout() can find a layout under the
     same condition.
  3. That the layout found is actually valid (that each logical pair lands
     on a physical edge) -- checked directly.

The goal is only to confirm the code runs correctly in the sandbox (a 2-core
Linux VM); absolute timing is not measured.
"""
from __future__ import annotations

from vf2_probe_common import BASIS_GATES, build_dense_pair_blocks_circuit
from verify_vf2_sparse_topology import topologies
from psf_smart_layout import smart_vf2_layout


def qiskit_default_vf2_outcome(qc, cmap, level=2):
    from qiskit import transpile
    holder = {}

    def cb(**kwargs):
        if type(kwargs["pass_"]).__name__ == "VF2Layout":
            holder["stop"] = str(kwargs["property_set"].get("VF2Layout_stop_reason"))

    transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
              optimization_level=level, seed_transpiler=0, callback=cb)
    return holder.get("stop", "(VF2Layout did not run)")


def validate_layout(cmap, pairs, layout_map):
    """layout_map: {logical: physical}. Checks that each logical pair lands on a physical edge."""
    edges = set()
    for a, b in cmap.get_edges():
        edges.add((a, b))
        edges.add((b, a))
    problems = []
    for lq_a, lq_b in pairs:
        pa, pb = layout_map[lq_a], layout_map[lq_b]
        if (pa, pb) not in edges:
            problems.append((lq_a, lq_b, pa, pb))
    # Also check for duplicate physical-qubit assignments.
    phys_used = list(layout_map.values())
    dup = len(phys_used) != len(set(phys_used))
    return problems, dup


def main():
    print("=" * 78)
    print("Smoke test: psf_smart_layout.smart_vf2_layout()")
    print("=" * 78)

    rows, cols = 8, 8
    for name, cmap, deg, _ in topologies(rows, cols):
        n = cmap.size()  # spare=0 (tight) -- the condition already confirmed in
                          # addendum-8/10 to show the cliff (except diluted_p0.75)
        qc = build_dense_pair_blocks_circuit(n, seed=0)
        pairs = [(i, i + 1) for i in range(0, n - 1, 2)]

        print(f"\n--- topology={name}  avg_deg={deg:.2f}  phys={n}  spare=0 ---")

        base = qiskit_default_vf2_outcome(qc, cmap, level=2)
        print(f"Qiskit default pipeline (L2) VF2Layout outcome: {base}")
        base_ok = base.endswith(".SOLUTION_FOUND")
        print(f"  -> premise check: "
              f"{'succeeded (differs from expectation)' if base_ok else 'failed (as expected from addendum-9/10)'}")

        layout_map, info = smart_vf2_layout(cmap, pairs, n,
                                            per_attempt_call_limit=50_000,
                                            time_budget_s=5.0)
        print(f"smart_vf2_layout result: found={layout_map is not None}")
        print(f"  info: feasible={info['feasible']} orderings_tried={info['orderings_tried']} "
              f"order_name={info['order_name']} elapsed_s={info['elapsed_s']:.4f}")
        for a in info["attempts"]:
            print(f"    - {a['order']:<24} found={a['found']}  time_s={a['time_s']:.4f}")

        if layout_map is not None:
            problems, dup = validate_layout(cmap, pairs, layout_map)
            print(f"  layout validity check: invalid pairs={len(problems)}  "
                  f"duplicate physical qubits={dup}")
            if problems:
                print(f"    example failures: {problems[:5]}")
            ok = (not problems) and (not dup)
            print(f"  -> {'valid layout' if ok else 'returned an invalid layout (bug)'}")
        else:
            print("  -> nothing found (either re-confirming that this grid is "
                  "genuinely hard for VF2, or this prototype's ordering set is "
                  "insufficient)")

    print("\n" + "=" * 78)
    print("Additional check: feasibility pre-check (can it return False without calling VF2?)")
    print("=" * 78)
    # A physical graph that is a single 3-node triangle, with a required
    # number of logical pairs (2 pairs = 4 qubits needed) for which no
    # perfect matching can exist -- an obviously infeasible case.
    from qiskit.transpiler import CouplingMap
    tiny_cmap = CouplingMap([(0, 1), (1, 2), (2, 0)])
    infeasible_pairs = [(0, 1), (2, 3)]  # needs 4 logical qubits but only 3 physical exist
    layout_map, info = smart_vf2_layout(tiny_cmap, infeasible_pairs, 3, time_budget_s=1.0)
    print(f"infeasible test: found={layout_map is not None} feasible={info['feasible']} "
          f"orderings_tried={info['orderings_tried']}")
    ok = (layout_map is None and info["feasible"] is False and info["orderings_tried"] == 0)
    print(f"  -> {'returned False immediately without calling VF2, as expected' if ok else 'unexpected behaviour (needs review)'}")

    print("\n" + "=" * 78)
    print("Conclusion")
    print("=" * 78)
    print("If smart_vf2_layout can find a valid layout under a condition where")
    print("Qiskit's default pipeline fails, this prototype's basic design (several")
    print("cheap BFS orderings + a small call_limit each) can be said to work.")


if __name__ == "__main__":
    main()
