"""bench_lightning_gpu.py -- Addendum 137.

Does lightning.gpu (RTX 4070, WSL2) actually beat CPU simulation, and at
what qubit count? Same redundant-block ansatz shape as Addendum 121/122
(train_heisenberg_torch.py), swept over qubit count, no noise model --
pure statevector simulation speed. Self-contained: does not import from
train_heisenberg_torch.py, so it has no dependency on which files happen to
be present in this environment.

Usage (run inside psf_zero_wsl_env_312):
    python bench_lightning_gpu.py
"""
from __future__ import annotations

import csv
import statistics
import time

import numpy as np
import pennylane as qml

LAYERS = 2
N_VALUES = (4, 8, 16, 20, 24)
# NOTE: an earlier draft went up to 32, but a 32-qubit statevector alone needs
# about 69 GB (2**32 * 16 bytes) -- far beyond the RTX 4070's 12 GB VRAM. Found
# by computing this before running anything, not by a crash. 24 qubits needs
# about 0.27 GB for the statevector itself, comfortably within budget even with
# lightning.gpu's own internal working buffers during gate application.
REPEATS = 20
SEED = 0


def build_ansatz(n, layers=LAYERS):
    """Redundant-block ansatz: ry, rz per qubit per layer, then cx on
    disjoint pairs -- the same shape as Addendum 121/122's own construction,
    reimplemented here standalone."""
    rng = np.random.default_rng(SEED)
    params = rng.uniform(-np.pi, np.pi, size=2 * n * layers)

    def circuit():
        k = 0
        for _ in range(layers):
            for q in range(n):
                qml.RY(params[k], wires=q); k += 1
                qml.RZ(params[k], wires=q); k += 1
            for a in range(0, n - 1, 2):
                qml.CNOT(wires=[a, a + 1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def bench(device_name, n):
    dev = qml.device(device_name, wires=n)
    qnode = qml.QNode(build_ansatz(n), dev)
    qnode()  # untimed warm-up (device/CUDA context setup)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        qnode()
        times.append(time.perf_counter() - t0)
    return times


def main():
    devices = ["default.qubit", "lightning.qubit", "lightning.gpu"]
    rows = []
    print(f"Ansatz: redundant blocks, {LAYERS} layers, {REPEATS} repeats after 1 warm-up call\n")
    for n in N_VALUES:
        print(f"n={n}:")
        results = {}
        for name in devices:
            try:
                times = bench(name, n)
                results[name] = times
                print(f"  {name:16s} median {statistics.median(times)*1000:9.3f} ms  "
                      f"(min {min(times)*1000:.3f}, max {max(times)*1000:.3f})")
            except Exception as exc:
                results[name] = None
                print(f"  {name:16s} FAILED: {type(exc).__name__}: {exc}")
            rows.append(dict(n=n, device=name,
                             median_ms=statistics.median(results[name]) * 1000 if results[name] else None,
                             min_ms=min(results[name]) * 1000 if results[name] else None,
                             max_ms=max(results[name]) * 1000 if results[name] else None))
        if results.get("lightning.qubit") and results.get("lightning.gpu"):
            ratio = statistics.median(results["lightning.gpu"]) / statistics.median(results["lightning.qubit"])
            print(f"  -> lightning.gpu / lightning.qubit = {ratio:.2f}x "
                  f"({'GPU faster' if ratio < 1 else 'CPU faster'})")
        print()

    out = "lightning_gpu_bench_2026-09-22.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
