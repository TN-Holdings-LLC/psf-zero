"""diag_fallback_accuracy.py -- Addendum 215: why did 4 of 3,000 training compiles
at block_gate_floor 8 lose accuracy (Addendum 214, L4)?

Two hypotheses recorded in Addendum 214 and one about the Rust core:
  H-A  Blocks the Rust core cannot decompose fall back to Qiskit's CX synthesis,
       which snaps inputs close to a special Weyl case onto it (infidelity up to
       ~1e-9); the guard (tolerance 1e-8) accepts that; the loss error comes from
       those blocks.
  H-B  Rebuilding such a block exactly -- Qiskit's Weyl decomposition without
       specialization, and PSF-Zero's closed-form core -- removes the error.
  H-C  The core's failure (SU2ExtractionSingular) comes from its SO(4)->SU(2)
       extraction: every component of one quaternion is multiplied by the
       scalar part of the other, so a local factor with zero trace (an exact
       Pauli-type rotation) makes the extraction singular. Exploratory: the
       core's own factors are not visible from Python, so this is examined
       through Qiskit's exact local factors, which are related but not equal.

Part R  Rebuild laps 1-3000 and the four failing laps (1030, 3600, 4580, 6720)
        of Addendum 214's part E exactly (same seeds), compile each at
        block_gate_floor 8 with psf_compile.py 2026-09-27.3, and record every
        block: which qubit pair, whether the core failed and why, and for each
        fallback block the infidelity of the circuit Qiskit returned, Qiskit's
        default Weyl specialization, and the infidelity of an exact rebuild.
Part X  Recompile the four failing laps with every fallback replaced by the
        exact rebuild, and recompute their loss error.

Predictions (fixed before running):
  R1  The four laps reproduce their recorded loss errors (1.30e-7, 3.82e-9,
      3.28e-9, 1.66e-9) to within 1%.
  R2  Every fallback block whose returned circuit has infidelity > 1e-14 is one
      where Qiskit's default Weyl decomposition specializes (is not "General").
  R3  The exact rebuild has infidelity <= 1e-14 for every fallback block.
  R4  With the exact rebuild (Part X), all four laps have loss error <= 1e-14.
  R5  Every fallback block lies on a second-layer pair (1,2), (3,4), (5,6),
      (7,8) or (9,10).
Exploratory: the distribution of fallback-block infidelities over laps 1-3000;
the smallest |tr|/2 among Qiskit's four exact local factors for fallback and
non-fallback blocks (H-C).

Usage (repository root; loop_endurance.py in benchmarks/):
    python -u benchmarks/diag_fallback_accuracy.py 2>&1 | tee diag_fallback_accuracy.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import platform
import statistics as st
import sys
import time
import warnings
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import TwoQubitWeylDecomposition
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc

FLOOR = 8
N_LAPS = 3000
FAIL_LAPS = {3600: 1.296e-07, 4580: 3.822e-09, 1030: 3.277e-09, 6720: 1.658e-09}
SECOND_LAYER = {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)}
OUT_CSV = "diag_fallback_accuracy_2026-09-27.csv"
LINE = CouplingMap.from_line(le.E_QUBITS)

CAPTURE = {"blocked": None, "fallbacks": []}
ORIG_FALLBACK = pc.SU4GeodesicPSFSynthesizer._fallback
ORIG_GUARDED = pc._guarded_cx_synthesis
ORIG_PM = pc.PassManager


def fallback_wrapper(self, U_target, msg, expected):
    circ = ORIG_FALLBACK(self, U_target, msg, expected)
    CAPTURE["fallbacks"].append((np.array(U_target), msg, circ))
    return circ


class CapturingPM(ORIG_PM):
    def run(self, *a, **k):
        out = super().run(*a, **k)
        CAPTURE["blocked"] = out
        return out


def weyl_exact(u):
    """Qiskit's Weyl decomposition without specialization."""
    try:
        return TwoQubitWeylDecomposition(u, fidelity=None)
    except TypeError:
        from qiskit._accelerate.two_qubit_decompose import Specialization
        return TwoQubitWeylDecomposition(u, _specialization=Specialization.General)


def exact_rebuild(u):
    d = weyl_exact(u)
    qc = QuantumCircuit(2, global_phase=d.global_phase)
    qc.append(UnitaryGate(d.K2r), [0])
    qc.append(UnitaryGate(d.K2l), [1])
    pc._append_cx_core_closed_form(qc, d.a, d.b, d.c, force=True)
    qc.append(UnitaryGate(d.K1r), [0])
    qc.append(UnitaryGate(d.K1l), [1])
    return qc


def infid(u, circ):
    return pc._aligned_distance(u, circ)[0]


def spec_name(u):
    d = TwoQubitWeylDecomposition(u)
    return str(getattr(d, "specialization", "?")).split(".")[-1], float(getattr(d, "calculated_fidelity", float("nan")))


def min_trace(u):
    d = weyl_exact(u)
    return min(abs(np.trace(k)) / 2 for k in (d.K1l, d.K1r, d.K2l, d.K2r))


def compile_lap(qc):
    CAPTURE["blocked"], CAPTURE["fallbacks"] = None, []
    pc._CX_CORE_CACHE.clear()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        out = pc.compile_for_hardware(qc, coupling_map=LINE, basis_gates=["cx", "rz", "sx", "x"],
                                      block_gate_floor=FLOOR, entangling_basis="cx",
                                      initial_layout=list(range(le.E_QUBITS)), on_unsupported="keep",
                                      seed_transpiler=0)
    return out


def block_pairs(blocked):
    """(pair, 4x4 matrix) of every consolidated block, in circuit order."""
    out = []
    for inst in blocked.data:
        if len(inst.qubits) == 2 and inst.operation.name == "unitary":
            pair = tuple(sorted(blocked.find_bit(q).index for q in inst.qubits))
            out.append((pair, inst.operation.to_matrix()))
    return out


def main():
    print(f"platform {platform.platform()} | python {platform.python_version()} | qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-27.3":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.3. Stopping.")
        return
    t0 = time.time()
    # the circuits of Addendum 214 part E, exactly
    rng_e = np.random.default_rng(7)
    target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    rng_t = np.random.default_rng(202)
    last = max(max(FAIL_LAPS), N_LAPS)
    thetas = {lap: target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS) for lap in range(1, last + 1)}

    def loss(qc):
        return 1 - abs(target.inner(Statevector(qc))) ** 2

    pc.SU4GeodesicPSFSynthesizer._fallback = fallback_wrapper
    pc.PassManager = CapturingPM
    rows, fb_rows = [], []
    laps = sorted(set(range(1, N_LAPS + 1)) | set(FAIL_LAPS))
    recorded = {}
    try:
        for lap in laps:
            qc = le.e_circuit(thetas[lap])
            out = compile_lap(qc)
            err = abs(loss(out) - loss(qc))
            blocks = block_pairs(CAPTURE["blocked"])
            fbs = CAPTURE["fallbacks"]
            recorded[lap] = err
            for u, msg, circ in fbs:
                pair = next((p for p, m in blocks if np.allclose(m, u, atol=1e-12)), None)
                sp, sf = spec_name(u)
                fb_rows.append(dict(lap=lap, pair=str(pair), msg=msg.split(":")[-1].strip(), infid=infid(u, circ),
                                    spec=sp, spec_fidelity=sf, rebuild_infid=infid(u, exact_rebuild(u)),
                                    min_trace=min_trace(u), cx=sum(1 for i in circ.data if i.operation.name == "cx")))
            rows.append(dict(lap=lap, loss_error=err, blocks=len(blocks), fallbacks=len(fbs)))
            if lap % 500 == 0:
                print(f"  lap {lap}: {len(fb_rows)} fallback blocks so far", flush=True)
        # non-fallback blocks' min trace, sample of 200 compiles (H-C, exploratory)
        nonfb_trace = []
        for lap in range(1, 201):
            qc = le.e_circuit(thetas[lap])
            compile_lap(qc)
            fb_set = [f[0] for f in CAPTURE["fallbacks"]]
            for pair, m in block_pairs(CAPTURE["blocked"]):
                if not any(np.allclose(m, u, atol=1e-12) for u in fb_set):
                    nonfb_trace.append(min_trace(m))
    finally:
        pc.SU4GeodesicPSFSynthesizer._fallback = ORIG_FALLBACK
        pc.PassManager = ORIG_PM

    # Part X: the failing laps with every Qiskit fallback replaced by the exact rebuild
    def exact_guarded(u):
        return exact_rebuild(u), True

    pc._guarded_cx_synthesis = exact_guarded
    try:
        x_err = {lap: abs(loss(compile_lap(le.e_circuit(thetas[lap]))) - loss(le.e_circuit(thetas[lap])))
                 for lap in FAIL_LAPS}
    finally:
        pc._guarded_cx_synthesis = ORIG_GUARDED

    # ---------------- report
    print("\n=== failing laps of Addendum 214 ===")
    for lap, want in FAIL_LAPS.items():
        fbl = [r for r in fb_rows if r["lap"] == lap]
        print(f"  lap {lap}: loss error {recorded[lap]:.3e} (recorded {want:.3e}); exact rebuild {x_err[lap]:.1e}; "
              f"fallback blocks: " + "; ".join(f"{r['pair']} infid {r['infid']:.1e} spec {r['spec']} "
                                              f"rebuild {r['rebuild_infid']:.1e}" for r in fbl))
    main_rows = [r for r in fb_rows if r["lap"] <= N_LAPS]
    n_comp = N_LAPS
    print(f"\n=== laps 1-{N_LAPS} ===")
    print(f"  fallback blocks {len(main_rows)} in {n_comp} compiles; by pair {dict(Counter(r['pair'] for r in main_rows))}")
    print(f"  core error messages {dict(Counter(r['msg'] for r in main_rows))}")
    bins = [(0, 1e-14), (1e-14, 1e-12), (1e-12, 1e-10), (1e-10, 1e-8), (1e-8, 1.0)]
    for lo, hi in bins:
        sel = [r for r in main_rows if lo < r["infid"] <= hi or (lo == 0 and r["infid"] <= hi)]
        print(f"  returned-circuit infidelity in ({lo:.0e}, {hi:.0e}]: {len(sel)}; specializations "
              f"{dict(Counter(r['spec'] for r in sel))}")
    bad_loss = [r for r in rows if r["loss_error"] > 1e-10]
    print(f"  compiles with loss error > 1e-10: {len(bad_loss)} of {len(rows)}")
    fbt = [r["min_trace"] for r in main_rows]
    if fbt and nonfb_trace:
        print(f"  (H-C) smallest |tr|/2 of Qiskit's exact local factors: fallback blocks median {st.median(fbt):.2e}, "
              f"max {max(fbt):.2e}; non-fallback blocks (200 compiles) median {st.median(nonfb_trace):.2e}, "
              f"min {min(nonfb_trace):.2e}")

    print("\n=== predictions ===")

    def v(ok):
        return "CONFIRMED" if ok else "NOT CONFIRMED"

    r1 = all(abs(recorded[l] - w) <= 0.01 * w for l, w in FAIL_LAPS.items())
    print(f"R1 the four laps reproduce their recorded loss errors within 1% -> {v(r1)}")
    inexact = [r for r in fb_rows if r["infid"] > 1e-14]
    specs = Counter(r["spec"] for r in inexact)
    if "?" in specs:
        print(f"R2 not evaluable: this Qiskit does not expose the Weyl specialization ({dict(specs)})")
    else:
        r2 = len(inexact) > 0 and all(r["spec"] != "General" for r in inexact)
        print(f"R2 every inexact fallback ({len(inexact)}) has a specialized default Weyl decomposition "
              f"({dict(specs)}) -> {v(r2)}")
    worst_rb = max((r["rebuild_infid"] for r in fb_rows), default=0.0)
    print(f"R3 exact rebuild infidelity <= 1e-14 for every fallback block (worst {worst_rb:.1e}) -> {v(worst_rb <= 1e-14)}")
    print(f"R4 with the exact rebuild, all four failing laps <= 1e-14 ({', '.join(f'{x:.1e}' for x in x_err.values())}) "
          f"-> {v(all(x <= 1e-14 for x in x_err.values()))}")
    pairs = {r["pair"] for r in fb_rows}
    print(f"R5 every fallback block on a second-layer pair ({sorted(pairs)}) -> "
          f"{v(all(p in {str(q) for q in SECOND_LAYER} for p in pairs))}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lap", "pair", "msg", "infid", "spec", "spec_fidelity", "rebuild_infid",
                                          "min_trace", "cx"])
        w.writeheader()
        w.writerows(fb_rows)
    with open(OUT_CSV.replace(".csv", "_laps.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lap", "loss_error", "blocks", "fallbacks"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(fb_rows)} fallback blocks) and the per-lap file; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
