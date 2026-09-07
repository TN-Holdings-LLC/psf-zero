# test_prototype_v4_correctness_and_speed.py
"""
Sanity check for psf_compile_prototype_v4.py, meant to run on the real
machine with the real psf_zero_core Rust extension (not the stub used
elsewhere in this project's own sandbox).

Two things are checked:
  1. Correctness: verify=True and verify=False must produce circuits that
     are Operator-equivalent to the original input (and to each other) --
     verify=False turns off the per-block self-check, not the math.
  2. Speed: verify=False should be meaningfully faster than verify=True on
     the same circuit, consistent with (though not necessarily identical in
     magnitude to) the stub-core measurement of 8.11x on synthesize() alone
     (benchmarks/profile_synthesize_fast_vs_verified.py) and 6.8x on the
     full compile() pipeline in this project's own sandbox re-check.

Run this before trusting the projected section 4 numbers in the README, or
before considering verify=False for any real use.
"""
import time
import numpy as np
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

import psf_compile_prototype_v4 as pcp


def build_chain_circuit(num_pairs, gates_per_pair, seed):
    """Same generator shape as phase1.py/phase2.py's dense same-pair chains."""
    r = np.random.default_rng(seed)
    qc = QuantumCircuit(num_pairs * 2)
    for p in range(num_pairs):
        for _ in range(gates_per_pair):
            U = unitary_group.rvs(4, random_state=r)
            qc.append(UnitaryGate(U), [2 * p, 2 * p + 1])
    return qc


def main():
    print("=== 1. Correctness (small circuit, Operator-equivalence check) ===")
    qc_small = build_chain_circuit(num_pairs=5, gates_per_pair=20, seed=1)
    outs = {}
    for verify in (True, False):
        outs[verify] = pcp.compile(qc_small, verify=verify)
        eq = Operator(qc_small).equiv(Operator(outs[verify]))
        print(f"verify={verify}: equivalent to original input = {eq}")
    eq_cross = Operator(outs[True]).equiv(Operator(outs[False]))
    print(f"verify=True output equivalent to verify=False output = {eq_cross}")
    if not (Operator(qc_small).equiv(Operator(outs[True]))
            and Operator(qc_small).equiv(Operator(outs[False]))):
        print("\n*** FAIL: verify=False changed correctness. Do not use it. ***")
        return

    print("\n=== 2. Speed (larger circuit, matching section 4's 156-qubit/78-block scale) ===")
    qc_big = build_chain_circuit(num_pairs=78, gates_per_pair=20, seed=2)
    times = {}
    for verify in (True, False):
        t0 = time.perf_counter()
        pcp.compile(qc_big, verify=verify)
        times[verify] = time.perf_counter() - t0
        print(f"verify={verify}: {times[verify]:.4f}s")
    print(f"\nSpeedup from verify=False: {times[True] / times[False]:.2f}x")
    print("(compare against this project's stub-core sandbox result of ~6.8x on")
    print(" the full compile() pipeline, and ~8.11x on synthesize() alone)")


if __name__ == "__main__":
    main()
