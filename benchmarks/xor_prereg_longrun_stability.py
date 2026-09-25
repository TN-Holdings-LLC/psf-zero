"""xor_prereg_longrun_stability.py -- long-run stability audit (workplace, number TBD).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: long-run stability" (2026-09-25).

Two workloads per iteration, 100 iterations per process:
  W1  the 2026-09-28 path: saved XOR params -> 4 circuits with the measurement
      inside -> measure-aware shared layout (M4) on FakeBrisbane -> noisy
      simulation (a different simulator seed every iteration).
  W2  a circuit where PSF-Zero acts: Addendum 156 tape 0 -> PSF-Zero synthesis
      (Rust core, real lightning.gpu block check) -> transpile to FakeBrisbane
      -> noisy simulation.
Run twice, once without and once with an explicit gc.collect() after every
iteration, then score both together:

    cd ~/pennylane_gpu_mock_test
    python -u xor_prereg_longrun_stability.py --gc none 2>&1 | tee ~/longrun_none.txt
    python -u xor_prereg_longrun_stability.py --gc each 2>&1 | tee ~/longrun_each.txt
    python -u xor_prereg_longrun_stability.py --score   2>&1 | tee ~/longrun_score.txt

Timing is measured ON THE RUNPOD POD ONLY and must not be mixed with numbers
from other machines. Nothing is sent to IBM.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import os
import platform
import time

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

import xor_prereg_stage1_sweep as S1  # also puts /root/psf-zero/benchmarks on sys.path
import compare_with_without_psf as C
from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer

N_ITER = 100
WARMUP = 5
SHOTS = 4000
W2_SHOTS = 20000
SEED = 0
PARAMS_FILE = os.path.expanduser("~/xor_params_seed0.npy")
OUT = {"none": os.path.expanduser("~/longrun_none_2026-09-25.csv"),
       "each": os.path.expanduser("~/longrun_each_2026-09-25.csv")}


def rss_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def fingerprint(qc):
    """Exact fingerprint: gate names, qubit indices and full-precision params."""
    items = []
    for inst in qc.data:
        ps = []
        for p in inst.operation.params:
            try:
                ps.append(float(p).hex())
            except (TypeError, ValueError):
                ps.append(np.asarray(p, dtype=complex).tobytes().hex())
        items.append((inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits), tuple(ps)))
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


def with_z0_measure(qc):
    out = qc.copy()
    creg = ClassicalRegister(1, "z0")
    out.add_register(creg)
    out.measure(0, creg[0])
    return out


def with_measure_all(qc):
    out = qc.copy()
    creg = ClassicalRegister(qc.num_qubits, "m")
    out.add_register(creg)
    for i in range(qc.num_qubits):
        out.measure(i, creg[i])
    return out


def measured_qubit(routed):
    qs = [routed.find_bit(i.qubits[0]).index for i in routed.data if i.operation.name == "measure"]
    return qs[0] if len(qs) == 1 else -1


def twoq(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name not in ("barrier", "measure"))


def run(gc_mode, n_iter):
    backend = FakeBrisbane()
    sim = AerSimulator.from_backend(backend)
    labels = S1.R.XOR_LABELS
    inputs = ["".join(str(x) for x in b) for b in S1.R.XOR_INPUTS]

    if not os.path.exists(PARAMS_FILE):
        res = S1.R.retrain_seed0()
        params = res[0] if isinstance(res, tuple) else res
        np.save(PARAMS_FILE, np.asarray(params))
        print("Saved trained params to", PARAMS_FILE)
    check = [S1.R.build_qiskit(np.load(PARAMS_FILE), b) for b in S1.R.XOR_INPUTS]
    check = [c[0] if isinstance(c, tuple) else c for c in check]
    S1.validate_circuits([c.remove_final_measurements(inplace=False) for c in check])
    print("Loaded params reproduce the rehearsal's exact values.")

    psf = SU4GeodesicPSFSynthesizer(
        GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True)

    offset = 0 if gc_mode == "none" else 100000  # independent random streams per process
    rows = []
    for it in range(1, n_iter + 1):
        # ---- W1: the 2026-09-28 path
        t0 = time.perf_counter()
        params = np.load(PARAMS_FILE)
        circs = []
        for b in S1.R.XOR_INPUTS:
            c = S1.R.build_qiskit(params, b)
            c = c[0] if isinstance(c, tuple) else c
            circs.append(with_z0_measure(c.remove_final_measurements(inplace=False)))
        t1 = time.perf_counter()
        base = transpile(circs[0], backend=backend, optimization_level=3, seed_transpiler=SEED)
        layout = base.layout.initial_index_layout(filter_ancillas=True)
        routed = [transpile(c, backend=backend, initial_layout=layout, optimization_level=3,
                            seed_transpiler=SEED) for c in circs]
        t2 = time.perf_counter()
        zs = []
        for i, r in enumerate(routed):
            counts = sim.run(r, shots=SHOTS, seed_simulator=offset + 10 * it + i).result().get_counts()
            zs.append((counts.get("0", 0) - counts.get("1", 0)) / SHOTS)
        t3 = time.perf_counter()
        w1 = dict(w1_load_build_ms=(t1 - t0) * 1e3, w1_compile_ms=(t2 - t1) * 1e3,
                  w1_simulate_ms=(t3 - t2) * 1e3, w1_total_ms=(t3 - t0) * 1e3)
        for i, (r, z) in enumerate(zip(routed, zs)):
            w1["w1_z_" + inputs[i]] = z
            w1["w1_correct_" + inputs[i]] = int(labels[i] * z > 0)
            w1["w1_q0_" + inputs[i]] = measured_qubit(r)
            w1["w1_2q_" + inputs[i]] = twoq(r)
            w1["w1_fp_" + inputs[i]] = fingerprint(r)

        # ---- W2: PSF-Zero acting
        before = psf.fallback_count
        stats = {"gpu_diffs": [], "synth_s": [], "block_twoq": []}
        u0 = time.perf_counter()
        qc_out = C.build_synthesized_circuit(C.make_tape(0), psf.synthesize, stats)
        u1 = time.perf_counter()
        r2 = transpile(with_measure_all(qc_out), backend=backend, optimization_level=3,
                       seed_transpiler=SEED)
        u2 = time.perf_counter()
        counts = sim.run(r2, shots=W2_SHOTS, seed_simulator=offset + 10 * it).result().get_counts()
        u3 = time.perf_counter()
        exact = Statevector(qc_out).probabilities_dict()
        tot = sum(counts.values())
        tvd = 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / tot) for k in set(exact) | set(counts))
        w2 = dict(w2_synth_verify_ms=(u1 - u0) * 1e3, w2_compile_ms=(u2 - u1) * 1e3,
                  w2_simulate_ms=(u3 - u2) * 1e3, w2_total_ms=(u3 - u0) * 1e3,
                  w2_fallbacks=psf.fallback_count - before, w2_gpu_worst=max(stats["gpu_diffs"]),
                  w2_synth_fp=fingerprint(qc_out), w2_routed_fp=fingerprint(r2),
                  w2_2q=twoq(r2), w2_tvd=tvd)

        if gc_mode == "each":
            gc.collect()
        rows.append(dict(gc_mode=gc_mode, iteration=it, timestamp=time.time(), rss_mb=rss_mb(),
                         **w1, **w2, platform=platform.platform()))
        if it % 10 == 0 or it <= 3:
            print("  it", it, "rss_mb", round(rows[-1]["rss_mb"], 1),
                  "w1_compile_ms", round(w1["w1_compile_ms"], 1), "w2_synth_ms", round(w2["w2_synth_verify_ms"], 1))

    with open(OUT[gc_mode], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote", len(rows), "rows to", OUT[gc_mode])


def verdict(confirmed, refuted):
    return "REFUTED" if refuted else ("CONFIRMED" if confirmed else "AMBIGUOUS")


def score():
    data = {}
    for mode, path in OUT.items():
        rows = list(csv.DictReader(open(path)))
        if len(rows) != N_ITER:
            raise SystemExit(path + " has " + str(len(rows)) + " rows, expected " + str(N_ITER) + " -- not scoring")
        data[mode] = rows
    inputs = ["".join(str(x) for x in b) for b in S1.R.XOR_INPUTS]
    f = lambda rows, k: [float(r[k]) for r in rows]
    print("=" * 78)
    print("SCORING (thresholds exactly as pre-registered)")
    print("=" * 78)

    wrong = sum(int(r["w1_correct_" + i]) == 0 for m in data for r in data[m] for i in inputs)
    print("L1 W1 correctness: wrong =", wrong, "of", 2 * N_ITER * 4, "->", verdict(wrong == 0, wrong > 0))

    det1 = all(len({r[k + i] for m in data for r in data[m]}) == 1
               for i in inputs for k in ("w1_fp_", "w1_q0_", "w1_2q_"))
    print("L2 W1 determinism (fingerprint, q0, 2q identical in all 200 iterations):",
          "->", verdict(det1, not det1))

    fps = {r["w2_synth_fp"] for m in data for r in data[m]}
    rfps = {r["w2_routed_fp"] for m in data for r in data[m]}
    fb = sum(int(r["w2_fallbacks"]) for m in data for r in data[m])
    print("L3 W2 PSF-Zero determinism: distinct synth fingerprints =", len(fps),
          " distinct routed fingerprints =", len(rfps), " fallbacks =", fb, "->",
          verdict(len(fps) == 1 and len(rfps) == 1 and fb == 0, len(fps) > 1 or len(rfps) > 1 or fb > 0))

    for m in data:
        sq = []
        for i in inputs:
            zs = np.array(f(data[m], "w1_z_" + i))
            sq.append((zs.std(ddof=1) / math.sqrt(max(1 - zs.mean() ** 2, 0) / SHOTS)) ** 2)
        R = math.sqrt(float(np.mean(sq)))
        print("L4 shot noise (" + m + "): pooled ratio =", format(R, ".3f"), "->",
              verdict(0.85 <= R <= 1.15, R < 0.7 or R > 1.3))

    cvs = {}
    for m in data:
        post = data[m][WARMUP:]
        for k in ("w1_compile_ms", "w2_synth_verify_ms"):
            v = np.array(f(post, k))
            cvs[(m, k)] = v.std(ddof=1) / v.mean()
            cold = float(data[m][0][k]) / float(np.median(v))
            print("   ", m, k, "median", format(float(np.median(v)), ".1f"), "ms  CV", format(cvs[(m, k)], ".3f"),
                  " first-iteration / median =", format(cold, ".2f"))
    print("L5 timing CV (iterations 6-100) ->",
          verdict(all(v < 0.10 for v in cvs.values()), any(v > 0.25 for v in cvs.values())))

    growth = {}
    for m in data:
        rss = f(data[m], "rss_mb")
        growth[m] = rss[N_ITER - 1] / rss[9]
        slope = float(np.polyfit(np.arange(10, N_ITER + 1), rss[9:], 1)[0])
        print("   ", m, "RSS it10", format(rss[9], ".1f"), "MB  it100", format(rss[N_ITER - 1], ".1f"),
              "MB  ratio", format(growth[m], ".3f"), " slope", format(slope, ".3f"), "MB/iteration")
    print("L6a memory growth it10 -> it100 ->",
          verdict(all(g <= 1.05 for g in growth.values()), any(g > 1.20 for g in growth.values())))
    cross = float(data["each"][N_ITER - 1]["rss_mb"]) / float(data["none"][N_ITER - 1]["rss_mb"])
    print("L6b RSS at it100, gc each / gc none =", format(cross, ".3f"), "->",
          verdict(abs(cross - 1) <= 0.05, abs(cross - 1) > 0.15))

    tv = f(data["none"], "w2_tvd") + f(data["each"], "w2_tvd")
    print("\nW2 TVD over 200 iterations: mean", format(float(np.mean(tv)), ".4f"), " sd", format(float(np.std(tv, ddof=1)), ".4f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", choices=["none", "each"])
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=N_ITER, help="dry runs only; scoring requires 100")
    a = ap.parse_args()
    if a.score:
        score()
    elif a.gc:
        run(a.gc, a.n)
    else:
        raise SystemExit("use --gc none, --gc each, or --score")


if __name__ == "__main__":
    main()
