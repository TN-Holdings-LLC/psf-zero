"""xor_prereg_stage1b_sweep.py -- Stage-1b pre-registered sweep (workplace, number TBD).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration, Stage 1b" (2026-09-25).
Reuses the hash-verified Stage-1 script (xor_prereg_stage1_sweep.py, same
folder) for circuits, backend selection and measurement helpers.

No timing is measured. CPU Aer only. Nothing is sent to IBM.

Run (from ~/pennylane_gpu_mock_test):
    python -u xor_prereg_stage1b_sweep.py 2>&1 | tee ~/xor_prereg_stage1b_run.txt
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

import xor_prereg_stage1_sweep as S1

SEED = 0
MAIN_SIM_SEED = 0
PRED_SIM_SEED = 1000
PRED_SHOTS = 20000
ARMS = ("A", "B", "M1", "M4")
EXCLUDED_FROM_SCORING = ("FakeKyoto",)  # all 144 ECR errors are 1.0 (Stage-1 diagnostics)
STAGE1_CSV = os.path.expanduser("~/xor_prereg_stage1_2026-09-25.csv")
OUT = os.path.expanduser("~/xor_prereg_stage1b_2026-09-25.csv")


def with_z0_measure(qc):
    """Logical circuit with logical qubit 0 measured into one classical bit,
    BEFORE transpile, so the layout passes see the measurement."""
    out = qc.copy()
    creg = ClassicalRegister(1, "z0")
    out.add_register(creg)
    out.measure(0, creg[0])
    return out


def measured_qubit(routed):
    qs = [routed.find_bit(i.qubits[0]).index for i in routed.data if i.operation.name == "measure"]
    if len(qs) != 1:
        raise SystemExit("expected exactly one measurement, found " + str(len(qs)))
    return qs[0]


def z_from_sim(sim, qc, shots, seed):
    counts = sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    if n0 + n1 != shots:
        raise SystemExit("unexpected count keys " + str(sorted(counts)))
    return (n0 - n1) / shots


def build_arm(arm, circuits, backend):
    """Returns a list of (measured circuit, measured physical qubit), one per input."""
    out = []
    if arm in ("A", "B"):
        for qc in circuits:
            routed = S1.route(arm, qc, backend, SEED)
            q0 = S1.physical_of_logical0(routed)
            out.append((S1.add_z0_measure(routed, q0), q0, routed))
        return out
    measured = [with_z0_measure(qc) for qc in circuits]
    if arm == "M1":
        for mqc in measured:
            routed = transpile(mqc, backend=backend, optimization_level=3, seed_transpiler=SEED)
            out.append((routed, measured_qubit(routed), routed))
        return out
    # M4: one layout for all four inputs, taken from input 00's measure-aware transpile
    base = transpile(measured[0], backend=backend, optimization_level=3, seed_transpiler=SEED)
    layout = base.layout.initial_index_layout(filter_ancillas=True)
    for mqc in measured:
        routed = transpile(mqc, backend=backend, initial_layout=layout,
                           optimization_level=3, seed_transpiler=SEED)
        out.append((routed, measured_qubit(routed), routed))
    return out


def verdict(confirmed, refuted):
    return "REFUTED" if refuted else ("CONFIRMED" if confirmed else "AMBIGUOUS")


def main():
    circuits = S1.build_logical_circuits()
    S1.validate_circuits(circuits)
    labels = S1.R.XOR_LABELS
    inputs = ["".join(str(x) for x in b) for b in S1.R.XOR_INPUTS]
    backends, excluded = S1.select_backends()
    print("Included backends (" + str(len(backends)) + "):", ", ".join(n for n, _, _ in backends))

    rows = []
    for name, backend, twoq in backends:
        sim = AerSimulator.from_backend(backend)
        for arm in ARMS:
            for i, (mqc, q0, routed) in enumerate(build_arm(arm, circuits, backend)):
                ok, reason = S1.is_isa_compliant(routed, backend)
                if not ok:
                    raise SystemExit("ISA check failed on " + name + " arm " + arm + ": " + reason)
                rows.append(dict(
                    backend=name, twoq=twoq, arm=arm, seed=SEED, input=inputs[i], label=labels[i],
                    q0_physical=q0, readout_error=backend.target["measure"][(q0,)].error,
                    z_noisy=z_from_sim(sim, mqc, S1.SHOTS, MAIN_SIM_SEED),
                    z_pred=z_from_sim(sim, mqc, PRED_SHOTS, PRED_SIM_SEED),
                ))
        print("  done", name)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote", len(rows), "rows to", OUT)

    # R0: arms A and B must reproduce Stage 1 (seed 0) exactly
    ref = {(r["backend"], r["arm"], r["input"]): float(r["z_noisy"])
           for r in csv.DictReader(open(STAGE1_CSV)) if r["seed"] == "0"}
    mism = [(r["backend"], r["arm"], r["input"]) for r in rows if r["arm"] in ("A", "B")
            and ref.get((r["backend"], r["arm"], r["input"])) != r["z_noisy"]]
    print("R0 reproduce Stage 1 arms A/B exactly: mismatches =", len(mism), mism[:5])
    if mism:
        print("R0 FAILED: predictions are NOT scored.")
        return

    M, Mp, ro_max = {}, {}, {}
    g = defaultdict(list)
    for r in rows:
        g[(r["backend"], r["arm"])].append(r)
    for k, rs in g.items():
        M[k] = float(np.mean([r["label"] * r["z_noisy"] for r in rs]))
        Mp[k] = float(np.mean([r["label"] * r["z_pred"] for r in rs]))
        ro_max[k] = max(r["readout_error"] for r in rs)
    names = [n for n, _, _ in backends]
    scored = [n for n in names if n not in EXCLUDED_FROM_SCORING]
    sel = {b: max(ARMS, key=lambda a: Mp[(b, a)]) for b in names}
    best = {b: max(M[(b, a)] for a in ARMS) for b in names}

    print("\n" + "=" * 78)
    print("SCORING (thresholds exactly as pre-registered; scored backends:", len(scored), ")")
    print("=" * 78)
    ro = [ro_max[(b, a)] for b in scored for a in ("M1", "M4")]
    print("Q1 readout of measured qubit (M1, M4): max =", format(max(ro), ".4f"),
          "->", verdict(all(x <= 0.05 for x in ro), any(x > 0.10 for x in ro)))
    d2 = {b: M[(b, "M1")] - max(M[(b, "A")], M[(b, "B")]) for b in scored}
    print("Q2 M1 vs best of A/B: min diff =", format(min(d2.values()), ".4f"),
          "->", verdict(all(v >= -0.02 for v in d2.values()), any(v < -0.05 for v in d2.values())))
    k, br = M.get(("FakeKingston", "M1")), M.get(("FakeBrussels", "M1"))
    if k is None or br is None:
        print("Q3 not evaluable (FakeKingston or FakeBrussels missing)")
    else:
        print("Q3 Kingston M1 =", format(k, ".4f"), " Brussels M1 =", format(br, ".4f"),
              "->", verdict(k >= 0.90 and br >= 0.90, k < 0.80 or br < 0.80))
    d4 = {b: abs(M[(b, "M4")] - M[(b, "M1")]) for b in scored}
    print("Q4 |M4 - M1|: max =", format(max(d4.values()), ".4f"),
          "->", verdict(all(v <= 0.02 for v in d4.values()), any(v > 0.05 for v in d4.values())))
    flips = [r for r in rows if r["backend"] in scored and r["label"] * r["z_noisy"] <= 0
             and (r["arm"] in ("M1", "M4") or r["arm"] == sel[r["backend"]])]
    print("Q5 correctness in M1, M4 and the selected arm: flips =", len(flips),
          "->", verdict(len(flips) == 0, len(flips) > 0))
    d6 = {b: M[(b, sel[b])] - best[b] for b in scored}
    print("Q6 selected arm vs best arm: min diff =", format(min(d6.values()), ".4f"),
          "->", verdict(all(v >= -0.02 for v in d6.values()), any(v < -0.05 for v in d6.values())))

    print("\nPer backend: M for A B M1 M4 | selected arm | max readout M1, M4")
    for b in names:
        tag = " (not scored)" if b in EXCLUDED_FROM_SCORING else ""
        print("  {:<16} {:.3f} {:.3f} {:.3f} {:.3f} | {:<2} | {:.4f} {:.4f}{}".format(
            b, M[(b, "A")], M[(b, "B")], M[(b, "M1")], M[(b, "M4")], sel[b],
            ro_max[(b, "M1")], ro_max[(b, "M4")], tag))


if __name__ == "__main__":
    main()
