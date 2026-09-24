"""compare_with_without_psf.py -- Addendum 156.

Runs the PennyLane -> GPU-verified synthesis -> routing -> SamplerV2
connection twice per tape: Arm A with Qiskit's TwoQubitBasisDecomposer as
the synthesizer (what Addenda 152-155 used), Arm B with PSF-Zero's own
block synthesizer (psf_compile.SU4GeodesicPSFSynthesizer, Rust core,
entangling_basis="cx", on_unsupported="raise" so any fallback to Qiskit's
decomposer raises rather than passing silently).

Uses a Qiskit-level path, NOT psf_pennylane_gpu_transform: that transform
collapses each synthesized block back into a 4x4 matrix, after which
routing re-synthesizes it with Qiskit's own decomposer -- so both arms would
reach the device as the same circuit (Addendum 156, Section 1a). Here the
synthesized gates are composed directly into the circuit that is routed.

Local testing mode only (FakeManilaV2 / AerSimulator); no credentials.

Usage (WSL, psf_zero_wsl_env_fresh):
    python compare_with_without_psf.py 2>&1 | tee compare_result.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics
import time

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer
from qiskit.quantum_info import Operator

from psf_ibm_real_submit import make_sampler_submit_fn
from psf_pennylane_gpu_ibm_prototype import is_isa_compliant, route_for_backend
from psf_pennylane_gpu_prototype import (
    ConnectionContractError,
    collect_and_consolidate,
    reference_cpu_synthesize,
    tape_to_qiskit,
)
from psf_pennylane_gpu_real import verify_block_gpu_and_cpu

BLOCK_GATE_FLOOR = 12
RUNS_PER_BLOCK = BLOCK_GATE_FLOOR + 3
SHOTS = 4000
SIM_SEED = 42
TAPE_SEEDS = (0, 1, 2, 3, 4)


def make_tape(seed):
    ops = []
    for k, wires in enumerate(([0, 1], [2, 3])):
        rng = np.random.default_rng(1000 * seed + k)
        ops += [
            qml.QubitUnitary(random_unitary(4, seed=int(rng.integers(0, 2**31))).data, wires=wires)
            for _ in range(RUNS_PER_BLOCK)
        ]
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def build_synthesized_circuit(tape, synth_fn, stats):
    """Qiskit-level path (Addendum 156, Section 1a): the synthesized GATES are
    composed directly into the output circuit, not collapsed back into a
    4x4 matrix, so the arm's synthesizer actually determines what reaches
    routing and the device."""
    qc, _ = tape_to_qiskit(tape)
    qc_blocked = collect_and_consolidate(qc, block_gate_floor=BLOCK_GATE_FLOOR)
    qc_out = qc_blocked.copy_empty_like()
    for inst in qc_blocked.data:
        op = inst.operation
        if len(inst.qubits) == 2 and op.name == "unitary":
            m = op.to_matrix()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                circ = synth_fn(m)
            stats["synth_s"].append(time.perf_counter() - t0)
            stats["block_twoq"].append(sum(1 for i in circ.data if len(i.qubits) == 2))
            check = verify_block_gpu_and_cpu(m, circ)
            stats["gpu_diffs"].append(check["gpu_expval_diff"])
            if not check["gpu_pass"]:
                raise ConnectionContractError(f"block failed real-GPU check: {check}")
            qubits = [qc_blocked.find_bit(q).index for q in inst.qubits]
            qc_out.compose(circ, qubits=qubits, inplace=True)
        else:
            qc_out.append(op, inst.qubits, inst.clbits)
    if not Operator(qc_out).equiv(Operator(qc)):
        raise ConnectionContractError("synthesized circuit is not equivalent to the input circuit")
    return qc_out


def tvd(counts, qc_routed):
    exact = Statevector(qc_routed.remove_final_measurements(inplace=False)).probabilities_dict()
    total = sum(counts.values())
    keys = set(exact) | set(counts)
    return 0.5 * sum(abs(exact.get(k, 0.0) - counts.get(k, 0) / total) for k in keys)


def main():
    backend = FakeManilaV2()
    noisy_submit = make_sampler_submit_fn(backend, seed_simulator=SIM_SEED)
    ideal_submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SIM_SEED)
    rows = []
    print(f"{'tape':>4} {'arm':15s} {'blk2q':>5} {'2q':>3} {'depth':>5} {'size':>4} {'gpu_diff':>9} "
          f"{'TVD_noisy':>9} {'TVD_ideal':>9} {'synth_ms':>8} {'fallbacks':>9}")
    for seed in TAPE_SEEDS:
        for arm in ("A_without_psf", "B_with_psf"):
            stats = {"gpu_diffs": [], "synth_s": [], "block_twoq": []}
            psf = None
            if arm == "A_without_psf":
                synth_fn = reference_cpu_synthesize
            else:
                psf = SU4GeodesicPSFSynthesizer(
                    GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True
                )
                synth_fn = psf.synthesize
            qc_out = build_synthesized_circuit(make_tape(seed), synth_fn, stats)
            qc_routed = route_for_backend(qc_out, backend)
            ok, reason = is_isa_compliant(qc_routed, backend)
            if not ok:
                raise ConnectionContractError(reason)
            measured = qc_routed.copy()
            measured.measure_all()
            counts_n = noisy_submit(measured, SHOTS)
            counts_i = ideal_submit(measured, SHOTS)
            twoq = sum(1 for inst in qc_routed.data if len(inst.qubits) == 2)
            fb = psf.fallback_count if psf is not None else None
            row = dict(
                tape_seed=seed, arm=arm, block_twoq_total=sum(stats["block_twoq"]),
                routed_twoq=twoq, depth=qc_routed.depth(), size=qc_routed.size(),
                gpu_diff_worst=max(stats["gpu_diffs"]), tvd_noisy=tvd(counts_n, qc_routed),
                tvd_ideal=tvd(counts_i, qc_routed),
                synth_ms_median=statistics.median(stats["synth_s"]) * 1000,
                n_blocks=len(stats["synth_s"]), psf_fallbacks=fb,
            )
            rows.append(row)
            print(f"{seed:>4} {arm:15s} {row['block_twoq_total']:>5} {twoq:>3} {row['depth']:>5} "
                  f"{row['size']:>4} {row['gpu_diff_worst']:>9.2e} {row['tvd_noisy']:>9.4f} "
                  f"{row['tvd_ideal']:>9.4f} {row['synth_ms_median']:>8.3f} {str(fb):>9}", flush=True)

    with open("compare_with_without_psf_2026-09-24.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote compare_with_without_psf_2026-09-24.csv")


if __name__ == "__main__":
    main()
