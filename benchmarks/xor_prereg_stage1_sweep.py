"""xor_prereg_stage1_sweep.py -- Stage-1 pre-registered sweep (workplace, number TBD).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration, Stage 1" (2026-09-25).
It imports the repository's own modules from /root/psf-zero/benchmarks and
is meant to live OUTSIDE the repository (~/pennylane_gpu_mock_test/).

No timing is measured. The GPU is not used (all simulation is CPU Aer).
Nothing is sent to IBM: every backend is a local fake backend.

Run:
    python -u xor_prereg_stage1_sweep.py 2>&1 | tee ~/xor_prereg_stage1_run.txt
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import sys
from collections import defaultdict

REPO_BENCH = "/root/psf-zero/benchmarks"
sys.path.insert(0, REPO_BENCH)

import numpy as np
import pennylane as qml
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
from qiskit import ClassicalRegister, transpile
from qiskit.providers import BackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider as fp

import rehearse_xor_fake127 as R
from psf_pennylane_gpu_ibm_prototype import is_isa_compliant, route_for_backend
from psf_pennylane_gpu_prototype import collect_and_consolidate

SHOTS = 4000
SEEDS = [0, 1, 2, 3, 4]
MAIN_SIM_SEED = 0
P4_SIM_SEEDS = list(range(50))
P4_BACKEND = "FakeBrisbane"
MIN_QUBITS = 100
TWO_Q_NAMES = ("ecr", "cx", "cz")
SKIP_OPS = ("barrier", "measure", "delay", "reset")

# Background values from benchmarks/rehearse_result_2.txt (not Stage-1 data).
EXPECTED_EXACT = 0.99776                              # |<Z0>|, PL and QK columns
REHEARSAL_NOISY = [-0.9040, 0.9185, 0.9085, -0.9130]  # FakeBrisbane, inputs 00,01,10,11


# ---------------------------------------------------------------- circuits

def build_logical_circuits():
    """The four XOR circuits exactly as rehearse_xor_fake127.py builds them
    after retraining seed 0. Measurements (if any) are removed."""
    res = R.retrain_seed0()
    params = res[0] if isinstance(res, tuple) else res
    circuits = []
    for bits in R.XOR_INPUTS:
        qc = R.build_qiskit(params, bits)
        if isinstance(qc, tuple):
            qc = qc[0]
        circuits.append(qc.remove_final_measurements(inplace=False))
    return circuits


def exact_z0(qc, qubit=0):
    op = SparsePauliOp.from_sparse_list([("Z", [qubit], 1.0)], num_qubits=qc.num_qubits)
    return float(np.real(Statevector(qc).expectation_value(op)))


def validate_circuits(circuits):
    """Harness check: the rebuilt circuits must reproduce the rehearsal's own
    exact values (+-0.99776 with the right signs). Stops loudly otherwise."""
    exacts = []
    for qc, bits, label in zip(circuits, R.XOR_INPUTS, R.XOR_LABELS):
        z = exact_z0(qc)
        if abs(z - label * EXPECTED_EXACT) > 6e-5:
            per_qubit = [round(exact_z0(qc, q), 5) for q in range(qc.num_qubits)]
            raise SystemExit(
                "HARNESS CHECK FAILED: input " + str(bits) + " exact <Z0> = " + str(z)
                + ", expected " + str(label * EXPECTED_EXACT)
                + ". <Z> per qubit: " + str(per_qubit)
                + ". Stopping before any Stage-1 data is produced."
            )
        exacts.append(z)
    return exacts


def count_blocks(qc):
    qcb = collect_and_consolidate(qc)
    return sum(1 for inst in qcb.data if inst.operation.name == "unitary" and len(inst.qubits) == 2)


# ---------------------------------------------------------------- backends

def select_backends():
    """Fixed rule: every Fake* class with >= MIN_QUBITS qubits whose Target has
    error values for 'measure' and a native 2-qubit gate. Exclusions are
    printed with a reason, never dropped silently."""
    chosen, excluded = [], []
    for name in sorted(dir(fp)):
        if not name.startswith("Fake"):
            continue
        cls = getattr(fp, name)
        if not isinstance(cls, type):
            excluded.append((name, "not a class"))
            continue
        try:
            backend = cls()
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            excluded.append((name, "cannot instantiate: " + type(exc).__name__))
            continue
        if not isinstance(backend, BackendV2):
            excluded.append((name, "not a BackendV2"))
            continue
        if backend.num_qubits < MIN_QUBITS:
            excluded.append((name, str(backend.num_qubits) + " qubits"))
            continue
        ops = set(backend.target.operation_names)
        twoq = [g for g in TWO_Q_NAMES if g in ops]
        if "measure" not in ops or not twoq:
            excluded.append((name, "no measure or no ecr/cx/cz in Target"))
            continue
        has_ro = any(p is not None and p.error is not None for p in backend.target["measure"].values())
        has_2q = any(p is not None and p.error is not None for p in backend.target[twoq[0]].values())
        if not (has_ro and has_2q):
            excluded.append((name, "Target lacks error values"))
            continue
        chosen.append((name, backend, twoq[0]))
    return chosen, excluded


# ---------------------------------------------------------------- routing + measurement

def route(arm, qc, backend, seed):
    if arm == "A":
        return route_for_backend(qc, backend, seed_transpiler=seed)
    return transpile(qc, backend=backend, optimization_level=3, seed_transpiler=seed)


def physical_of_logical0(routed):
    if routed.layout is None:
        return 0
    return routed.layout.final_index_layout()[0]


def add_z0_measure(routed, q0):
    qc = routed.copy()
    creg = ClassicalRegister(1, "z0")
    qc.add_register(creg)
    qc.measure(q0, creg[0])
    return qc


def noisy_z0(sim, qc, seed):
    counts = sim.run(qc, shots=SHOTS, seed_simulator=seed).result().get_counts()
    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    if n0 + n1 != SHOTS:
        raise SystemExit("unexpected count keys " + str(sorted(counts)) + " -- stopping")
    return (n0 - n1) / SHOTS


def se(z):
    return math.sqrt(max(1.0 - z * z, 0.0) / SHOTS)


# ---------------------------------------------------------------- scoring

def verdict(confirmed, refuted):
    if refuted:
        return "REFUTED"
    if confirmed:
        return "CONFIRMED"
    return "AMBIGUOUS"


def score(rows, p4):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["backend"], r["arm"], r["seed"])].append(r)
    M = {k: float(np.mean([r["label"] * r["z_noisy"] for r in rs])) for k, rs in groups.items()}
    backends = sorted({k[0] for k in groups})

    print("\n" + "=" * 78)
    print("SCORING (thresholds exactly as pre-registered)")
    print("=" * 78)

    flips = [r for r in rows if r["label"] * r["z_noisy"] <= 0]
    print("P1 correctness: sign flips =", len(flips), "->", verdict(len(flips) == 0, len(flips) > 0))

    print("P2 withdrawn before locking (see the pre-registration, section 3) -- not scored")

    spreads = {}
    for b in backends:
        for arm in ("A", "B"):
            vals = [M[(b, arm, s)] for s in SEEDS if (b, arm, s) in M]
            spreads[(b, arm)] = max(vals) - min(vals)
    print("P3 seed spread: max spread =", format(max(spreads.values()), ".4f"),
          "->", verdict(all(v <= 0.03 for v in spreads.values()), any(v > 0.06 for v in spreads.values())))

    sq = []
    for i, zs in enumerate(p4):
        sd = float(np.std(zs, ddof=1))
        theory = se(float(np.mean(zs)))
        sq.append((sd / theory) ** 2)
        print("   P4 input", R.XOR_INPUTS[i], " sd =", format(sd, ".5f"),
              " theory =", format(theory, ".5f"), " ratio =", format(sd / theory, ".3f"))
    pooled = math.sqrt(float(np.mean(sq)))
    print("P4 shot noise: pooled ratio =", format(pooled, ".3f"),
          "->", verdict(0.85 <= pooled <= 1.15, pooled < 0.7 or pooled > 1.3))

    nonzero = [r for r in rows if r["blocks"] != 0]
    print("P5 blocks: nonzero cells =", len(nonzero), "->", verdict(len(nonzero) == 0, len(nonzero) > 0))

    d6 = {(b, s): M[(b, "B", s)] - M[(b, "A", s)] for b in backends for s in SEEDS
          if (b, "B", s) in M and (b, "A", s) in M}
    frac_better = float(np.mean([d6[(b, 0)] > 0.01 for b in backends if (b, 0) in d6]))
    print("P6 free layout: min (B-A) =", format(min(d6.values()), ".4f"),
          " share of backends with B-A > 0.01 at seed 0 =", format(frac_better, ".3f"),
          "->", verdict(all(v >= -0.01 for v in d6.values()) and frac_better >= 0.5,
                        any(v < -0.03 for v in d6.values())))

    print("\nPer backend: mean margin M by seed (arm A | arm B)")
    for b in backends:
        a = " ".join(format(M[(b, "A", s)], ".3f") for s in SEEDS)
        bb = " ".join(format(M[(b, "B", s)], ".3f") for s in SEEDS)
        print("  {:<16} {} | {}".format(b, a, bb))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/xor_prereg_stage1_2026-09-25.csv"))
    args = ap.parse_args()

    env = dict(
        platform=platform.platform(), processor=platform.processor() or "unknown",
        python=platform.python_version(), qiskit=qiskit.__version__,
        qiskit_aer=qiskit_aer.__version__, qiskit_ibm_runtime=qiskit_ibm_runtime.__version__,
        pennylane=qml.__version__,
    )
    print("Environment:", env)

    circuits = build_logical_circuits()
    exacts = validate_circuits(circuits)
    blocks = [count_blocks(qc) for qc in circuits]
    print("Circuit check passed: exact <Z0> =", [round(z, 5) for z in exacts], " blocks =", blocks)

    backends, excluded = select_backends()
    print("\nIncluded backends (" + str(len(backends)) + "):",
          ", ".join(n + "(" + str(b.num_qubits) + "q," + g + ")" for n, b, g in backends))
    print("Excluded (" + str(len(excluded)) + "):", "; ".join(n + ": " + why for n, why in excluded))
    if P4_BACKEND not in [n for n, _, _ in backends]:
        raise SystemExit(P4_BACKEND + " is not among the included backends -- stopping")

    rows = []
    p4 = None
    for name, backend, twoq in backends:
        sim = AerSimulator.from_backend(backend)
        for arm in ("A", "B"):
            for s in SEEDS:
                for i, (qc, bits, label) in enumerate(zip(circuits, R.XOR_INPUTS, R.XOR_LABELS)):
                    routed = route(arm, qc, backend, s)
                    ok, reason = is_isa_compliant(routed, backend)
                    if not ok:
                        raise SystemExit("ISA check failed on " + name + " arm " + arm + ": " + reason)
                    q0 = physical_of_logical0(routed)
                    meas = add_z0_measure(routed, q0)
                    z_noisy = noisy_z0(sim, meas, MAIN_SIM_SEED)
                    rows.append(dict(
                        backend=name, num_qubits=backend.num_qubits, twoq=twoq, arm=arm, seed=s,
                        input="".join(str(x) for x in bits), label=label, z_exact=exacts[i],
                        z_noisy=z_noisy, q0_physical=q0,
                        readout_error=backend.target["measure"][(q0,)].error,
                        routed_2q=sum(1 for x in routed.data if len(x.qubits) == 2
                                      and x.operation.name not in SKIP_OPS),
                        blocks=blocks[i], shots=SHOTS, sim_seed=MAIN_SIM_SEED, **env,
                    ))
                    if name == P4_BACKEND and arm == "A" and s == 0:
                        if p4 is None:
                            p4 = [[] for _ in circuits]
                        p4[i] = [noisy_z0(sim, meas, ss) for ss in P4_SIM_SEEDS]
        m_a = np.mean([r["label"] * r["z_noisy"] for r in rows
                       if r["backend"] == name and r["arm"] == "A" and r["seed"] == 0])
        m_b = np.mean([r["label"] * r["z_noisy"] for r in rows
                       if r["backend"] == name and r["arm"] == "B" and r["seed"] == 0])
        print("  done", name, " seed-0 M_A =", format(float(m_a), ".4f"), " M_B =", format(float(m_b), ".4f"))

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nWrote", len(rows), "rows to", args.out)

    # C0: harness vs the existing rehearsal (FakeBrisbane, arm A, seed 0)
    c0 = [r for r in rows if r["backend"] == P4_BACKEND and r["arm"] == "A" and r["seed"] == 0]
    c0_ok = True
    for r, ref in zip(c0, REHEARSAL_NOISY):
        tol = 3.0 * math.sqrt(2.0) * se(ref)
        good = abs(r["z_noisy"] - ref) <= tol
        c0_ok = c0_ok and good
        print("C0 input", r["input"], " harness =", format(r["z_noisy"], ".4f"), " rehearsal =", ref,
              " tol =", format(tol, ".4f"), "OK" if good else "MISMATCH")
    if not c0_ok:
        print("\nC0 FAILED: the harness does not reproduce the rehearsal. Predictions are NOT scored.")
        return
    score(rows, p4)


if __name__ == "__main__":
    main()
