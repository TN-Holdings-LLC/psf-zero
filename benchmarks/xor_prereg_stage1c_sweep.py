"""xor_prereg_stage1c_sweep.py -- Stage-1c pre-registered comparison (workplace, number TBD).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration, Stage 1c" (2026-09-25).

Arms (same 5 tapes as Addendum 156's compare_with_without_psf.py):
  A  Qiskit TwoQubitBasisDecomposer(CXGate()) default   (Addendum 156 arm A)
  Z  Qiskit TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
  P  PSF-Zero SU4GeodesicPSFSynthesizer, Rust core, on_unsupported="raise"
  T  no pre-synthesis: Qiskit transpile(optimization_level=3) of the measured
     circuit ("leave everything to Qiskit")
plus R0 (exact replication of Addendum 156 on FakeManilaV2) and the XOR null
control. Reuses the repository's compare_with_without_psf.py (make_tape,
build_synthesized_circuit with its real-GPU block check, tvd) and the
hash-verified Stage-1 script. No timing is measured.

Run (from ~/pennylane_gpu_mock_test):
    python -u xor_prereg_stage1c_sweep.py 2>&1 | tee ~/xor_prereg_stage1c_run.txt
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

import xor_prereg_stage1_sweep as S1  # also puts /root/psf-zero/benchmarks on sys.path
import compare_with_without_psf as C
from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer
from psf_ibm_real_submit import make_sampler_submit_fn
from psf_pennylane_gpu_ibm_prototype import is_isa_compliant, route_for_backend
from psf_pennylane_gpu_prototype import collect_and_consolidate, tape_to_qiskit

SHOTS = 100_000
SIM_SEED = 0
SEED = 0
MIN_2Q = 6  # two generic SU(4) blocks x 3 CX
EXCLUDED = ("FakeKyoto",)
REF_CSV = "/root/psf-zero/data/compare_with_without_psf_2026-09-24.csv"
OUT = os.path.expanduser("~/xor_prereg_stage1c_2026-09-25.csv")
ARMS = ("A", "Z", "P", "T")
SKIP = ("barrier", "measure", "delay", "reset")
_ZSX = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")


def build_all():
    psf = SU4GeodesicPSFSynthesizer(
        GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True)
    synth = {"A": C.reference_cpu_synthesize, "Z": lambda m: _ZSX(m), "P": psf.synthesize}
    circs, fallbacks = {}, {}
    for seed in C.TAPE_SEEDS:
        tape = C.make_tape(seed)
        circs[(seed, "T")] = tape_to_qiskit(tape)[0]
        for arm, fn in synth.items():
            before = psf.fallback_count
            stats = {"gpu_diffs": [], "synth_s": [], "block_twoq": []}
            circs[(seed, arm)] = C.build_synthesized_circuit(tape, fn, stats)
            if arm == "P":
                fallbacks[seed] = psf.fallback_count - before
    return circs, fallbacks


def replicate_a156(circs, fallbacks):
    """R0: Addendum 156's own path on FakeManilaV2, compared with its CSV."""
    backend = FakeManilaV2()
    noisy = make_sampler_submit_fn(backend, seed_simulator=C.SIM_SEED)
    ideal = make_sampler_submit_fn(AerSimulator(), seed_simulator=C.SIM_SEED)
    ref = {(int(r["tape_seed"]), r["arm"]): r for r in csv.DictReader(open(REF_CSV))}
    bad = []
    for seed in C.TAPE_SEEDS:
        for arm, refarm in (("A", "A_without_psf"), ("P", "B_with_psf")):
            routed = route_for_backend(circs[(seed, arm)], backend)
            m = routed.copy()
            m.measure_all()
            got = dict(routed_twoq=sum(1 for i in routed.data if len(i.qubits) == 2),
                       depth=routed.depth(), size=routed.size(),
                       tvd_noisy=C.tvd(noisy(m, C.SHOTS), routed),
                       tvd_ideal=C.tvd(ideal(m, C.SHOTS), routed))
            r = ref[(seed, refarm)]
            for k in ("routed_twoq", "depth", "size"):
                if int(r[k]) != got[k]:
                    bad.append((seed, arm, k, r[k], got[k]))
            for k in ("tvd_noisy", "tvd_ideal"):
                if abs(float(r[k]) - got[k]) > 1e-9:
                    bad.append((seed, arm, k, r[k], got[k]))
            if arm == "P" and str(fallbacks[seed]) != r["psf_fallbacks"]:
                bad.append((seed, arm, "psf_fallbacks", r["psf_fallbacks"], fallbacks[seed]))
            print("R0 tape", seed, arm, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in got.items()})
    return bad


def measure_logical(routed, n):
    final = routed.layout.final_index_layout()
    m = routed.copy()
    creg = ClassicalRegister(n, "m")
    m.add_register(creg)
    for i in range(n):
        m.measure(final[i], creg[i])
    return m


def with_measure(qc):
    m = qc.copy()
    creg = ClassicalRegister(qc.num_qubits, "m")
    m.add_register(creg)
    for i in range(qc.num_qubits):
        m.measure(i, creg[i])
    return m


def tvd4(counts, exact):
    tot = sum(counts.values())
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / tot) for k in keys)


def cell_metrics(unmeasured):
    return dict(
        routed_2q=sum(1 for i in unmeasured.data if len(i.qubits) == 2 and i.operation.name not in SKIP),
        n_1q=sum(1 for i in unmeasured.data if len(i.qubits) == 1 and i.operation.name not in SKIP),
        size=unmeasured.size(), depth=unmeasured.depth())


def sweep(circs, fallbacks):
    backends, _ = S1.select_backends()
    backends = [("FakeManilaV2", FakeManilaV2(), "cx")] + backends
    rows = []
    for name, backend, twoq in backends:
        sim = AerSimulator.from_backend(backend)
        for seed in C.TAPE_SEEDS:
            n = circs[(seed, "T")].num_qubits
            exact = Statevector(circs[(seed, "T")]).probabilities_dict()
            base = transpile(with_measure(circs[(seed, "Z")]), backend=backend,
                             optimization_level=3, seed_transpiler=SEED)
            layout = base.layout.initial_index_layout(filter_ancillas=True)
            for arm in ARMS:
                if arm == "T":
                    measured = transpile(with_measure(circs[(seed, "T")]), backend=backend,
                                         optimization_level=3, seed_transpiler=SEED)
                    unmeasured = measured.remove_final_measurements(inplace=False)
                    check = measured
                else:
                    unmeasured = transpile(circs[(seed, arm)], backend=backend, initial_layout=layout,
                                           optimization_level=1, seed_transpiler=SEED)
                    measured = measure_logical(unmeasured, n)
                    check = unmeasured
                ok, reason = is_isa_compliant(check, backend)
                if not ok:
                    raise SystemExit("ISA check failed on " + name + " arm " + arm + ": " + reason)
                counts = sim.run(measured, shots=SHOTS, seed_simulator=SIM_SEED).result().get_counts()
                rows.append(dict(backend=name, twoq=twoq, tape_seed=seed, arm=arm,
                                 **cell_metrics(unmeasured),
                                 tvd_noisy=tvd4(counts, exact),
                                 psf_fallbacks=fallbacks[seed] if arm == "P" else ""))
        print("  done", name)
    return rows


def xor_null_control():
    out = []
    for qc in S1.build_logical_circuits():
        blocked = collect_and_consolidate(qc, block_gate_floor=C.BLOCK_GATE_FLOOR)
        n_blocks = sum(1 for i in blocked.data if i.operation.name == "unitary" and len(i.qubits) == 2)
        a = [(i.operation, tuple(qc.find_bit(q).index for q in i.qubits)) for i in qc.data]
        b = [(i.operation, tuple(blocked.find_bit(q).index for q in i.qubits)) for i in blocked.data]
        out.append((n_blocks, a == b))
    return out


def verdict(confirmed, refuted):
    return "REFUTED" if refuted else ("CONFIRMED" if confirmed else "AMBIGUOUS")


def main():
    circs, fallbacks = build_all()
    print("Synthesis done (all blocks passed the real-GPU check and the equivalence check).")
    bad = replicate_a156(circs, fallbacks)
    print("R0 replicate Addendum 156 exactly: mismatches =", len(bad), bad[:6])
    rows = sweep(circs, fallbacks)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote", len(rows), "rows to", OUT)
    null = xor_null_control()
    print("XOR null control (blocks, circuit unchanged):", null)
    if bad:
        print("R0 FAILED: predictions are NOT scored.")
        return

    cell = {(r["backend"], r["tape_seed"], r["arm"]): r for r in rows}
    keys = sorted({(r["backend"], r["tape_seed"]) for r in rows if r["backend"] not in EXCLUDED})
    print("\n" + "=" * 78)
    print("SCORING (thresholds exactly as pre-registered; scored cells per arm:", len(keys), ")")
    print("=" * 78)
    fb = [fallbacks[s] for s in C.TAPE_SEEDS]
    print("C1 PSF fallbacks:", fb, "->", verdict(all(x == 0 for x in fb), any(x != 0 for x in fb)))
    c2 = [cell[k + (a,)]["routed_2q"] for k in keys for a in ("A", "Z", "P")]
    print("C2 routed 2q in A/Z/P: values", sorted(set(c2)), "->",
          verdict(all(x == MIN_2Q for x in c2), any(x != MIN_2Q for x in c2)))
    c3 = [cell[k + ("T",)]["routed_2q"] for k in keys]
    print("C3 routed 2q in T (Qiskit opt3): values", sorted(set(c3)), "->",
          verdict(all(x == MIN_2Q for x in c3), any(x > MIN_2Q for x in c3)))
    c4 = [(cell[k + ("P",)]["size"] == cell[k + ("Z",)]["size"]
           and cell[k + ("P",)]["depth"] == cell[k + ("Z",)]["depth"],
           abs(cell[k + ("P",)]["tvd_noisy"] - cell[k + ("Z",)]["tvd_noisy"])) for k in keys]
    print("C4 P vs Z: size/depth equal in", sum(x[0] for x in c4), "of", len(c4),
          " max |dTVD| =", format(max(x[1] for x in c4), ".5f"), "->",
          verdict(all(x[0] and x[1] <= 0.002 for x in c4), any((not x[0]) or x[1] > 0.002 for x in c4)))
    c5 = [cell[k + ("P",)]["size"] < cell[k + ("A",)]["size"]
          and cell[k + ("P",)]["depth"] < cell[k + ("A",)]["depth"] for k in keys]
    print("C5 P smaller than A (size and depth): in", sum(c5), "of", len(c5), "->",
          verdict(all(c5), not all(c5)))
    d = [cell[k + ("A",)]["tvd_noisy"] - cell[k + ("P",)]["tvd_noisy"] for k in keys]
    print("C6 mean (TVD_A - TVD_P) =", format(float(np.mean(d)), ".5f"),
          " P better in", sum(x > 0 for x in d), "of", len(d), "->",
          verdict(float(np.mean(d)) <= 0.01, float(np.mean(d)) > 0.02))
    print("C7 XOR null control ->", verdict(all(nb == 0 and same for nb, same in null),
                                            not all(nb == 0 and same for nb, same in null)))

    print("\nPer backend (mean over 5 tapes): size A Z P T | depth A Z P T | TVD A Z P T")
    for b in sorted({k[0] for k in keys}) + [x for x in EXCLUDED if any(r["backend"] == x for r in rows)]:
        ks = [k for k in cell if k[0] == b]
        mean = lambda arm, f: float(np.mean([cell[k][f] for k in ks if k[2] == arm]))
        print("  {:<16} {} | {} | {}{}".format(
            b, " ".join(format(mean(a, "size"), "5.1f") for a in ARMS),
            " ".join(format(mean(a, "depth"), "5.1f") for a in ARMS),
            " ".join(format(mean(a, "tvd_noisy"), ".4f") for a in ARMS),
            "  (not scored)" if b in EXCLUDED else ""))


if __name__ == "__main__":
    main()
