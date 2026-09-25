"""psf_vs_qiskit_depth_sweep.py -- Addendum 177.

As circuits get deeper, does PSF-Zero's block synthesis beat Qiskit's
default compilation (optimization level 3) under realistic (fake-backend)
noise?

Arms:
  Q3  Qiskit default: transpile(optimization_level=3) on the whole circuit.
  P   PSF-Zero: each block synthesized by SU4GeodesicPSFSynthesizer
      (entangling_basis="cx", on_unsupported="raise"), then transpile at
      optimization_level=1 pinned to Q3's own layout for that circuit.
  Z   Control: same as P with TwoQubitBasisDecomposer(CXGate(),
      euler_basis="ZSX").

Level 1 for P and Z: level 3 would re-synthesize every 2-qubit block and
erase the synthesizer's output (Addendum 156, Section 1a).

Usage (WSL or RunPod, repository root on sys.path, CPU only):
    python -u psf_vs_qiskit_depth_sweep.py 2>&1 | tee depth_sweep_result.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import math
import platform
import statistics as st

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeFez, FakeKingston, FakeMarrakesh

from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer
from psf_ibm_real_submit import make_sampler_submit_fn

N = 4
LAYERS = (1, 2, 4, 8, 16)
CIRCUITS = 10
SHOTS = 4000
SIM_SEED = 42
BACKENDS = (FakeFez, FakeMarrakesh, FakeKingston, FakeBrisbane)
OUT_CSV = "psf_vs_qiskit_depth_sweep_2026-09-25.csv"

_ZSX = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")


def build_logical(layers: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(1000 * layers + seed)
    qc = QuantumCircuit(N)
    for layer in range(layers):
        pairs = [(0, 1), (2, 3)] if layer % 2 == 0 else [(1, 2)]
        for a, b in pairs:
            qc.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(0, 2**31))).data), [a, b])
    return qc


def synthesize_blocks(qc: QuantumCircuit, synth) -> QuantumCircuit:
    """Replace every UnitaryGate block by the synthesizer's gates (not
    collapsed back to a matrix), and check equivalence to the input."""
    out = QuantumCircuit(N)
    for inst in qc.data:
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        with contextlib.redirect_stdout(io.StringIO()):
            sub = synth(inst.operation.to_matrix())
        out.compose(sub, qubits=qubits, inplace=True)
    if not Operator(out).equiv(Operator(qc)):
        raise RuntimeError("synthesized circuit is not equivalent to the logical circuit")
    return out


def measured(qc: QuantumCircuit) -> QuantumCircuit:
    m = qc.copy()
    m.measure_all()  # classical bit i = logical qubit i, before compilation
    return m


def tvd(counts: dict[str, int], exact: dict[str, float]) -> float:
    total = sum(counts.values())
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / total) for k in keys)


def twoq(qc: QuantumCircuit) -> int:
    return sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name not in ("barrier",))


def main():
    print(f"platform {platform.platform()} | python {platform.python_version()} | qiskit {qiskit.__version__}")
    ideal_submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SIM_SEED)
    rows = []
    for backend_cls in BACKENDS:
        backend = backend_cls()
        noisy_submit = make_sampler_submit_fn(backend, seed_simulator=SIM_SEED)
        for L in LAYERS:
            for seed in range(CIRCUITS):
                qc = build_logical(L, seed)
                exact = Statevector(qc).probabilities_dict()

                t_q3 = transpile(measured(qc), backend, optimization_level=3, seed_transpiler=0)
                layout = list(t_q3.layout.initial_index_layout(filter_ancillas=True))

                psf = SU4GeodesicPSFSynthesizer(
                    GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True
                )
                compiled = {"Q3": t_q3}
                for arm, synth in (("P", psf.synthesize), ("Z", _ZSX)):
                    qs = synthesize_blocks(qc, synth)
                    compiled[arm] = transpile(measured(qs), backend, optimization_level=1,
                                              initial_layout=layout, seed_transpiler=0)

                for arm, tc in compiled.items():
                    cn = noisy_submit(tc, SHOTS)
                    ci = ideal_submit(tc, SHOTS)
                    used = list(tc.layout.initial_index_layout(filter_ancillas=True))
                    rows.append(dict(
                        backend=backend.name, layers=L, blocks=len(qc.data), circuit=seed, arm=arm,
                        twoq=twoq(tc), depth=tc.depth(), size=tc.size(), layout=str(used),
                        tvd_noisy=tvd(cn, exact), tvd_ideal=tvd(ci, exact),
                        psf_fallbacks=psf.fallback_count if arm == "P" else "",
                    ))
            sel = [r for r in rows if r["backend"] == backend.name and r["layers"] == L]
            by = {a: [r["tvd_noisy"] for r in sel if r["arm"] == a] for a in ("Q3", "P", "Z")}
            diffs = [q - p for q, p in zip(by["Q3"], by["P"])]
            d = st.mean(diffs)
            se = st.stdev(diffs) / math.sqrt(len(diffs))
            print(f"{backend.name:16s} L={L:2d}  TVD Q3={st.mean(by['Q3']):.4f} P={st.mean(by['P']):.4f} "
                  f"Z={st.mean(by['Z']):.4f}  d(Q3-P)={d:+.4f} SE={se:.4f}  "
                  f"2q Q3/P/Z={st.mean(r['twoq'] for r in sel if r['arm']=='Q3'):.1f}/"
                  f"{st.mean(r['twoq'] for r in sel if r['arm']=='P'):.1f}/"
                  f"{st.mean(r['twoq'] for r in sel if r['arm']=='Z'):.1f}", flush=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
