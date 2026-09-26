"""compare_zsx_arm.py -- Addendum 158.

Adds a third arm to Addendum 156/157's with/without-PSF-Zero comparison:
Qiskit's TwoQubitBasisDecomposer configured with euler_basis="ZSX" -- the
same configuration PSF-Zero's own CX path uses -- to test whether Addendum
157's depth/size reduction is PSF-Zero-specific or just that configuration.

Reuses compare_with_without_psf.py's own building blocks unchanged, so the
path, tapes, seeds and measurements are identical to Addendum 157.

Usage (WSL, psf_zero_wsl_env_fresh, same folder as compare_with_without_psf.py):
    python compare_zsx_arm.py 2>&1 | tee compare_zsx_result.txt
"""
from __future__ import annotations

import csv
import statistics

from qiskit.circuit.library import CXGate
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

from compare_with_without_psf import (
    SHOTS,
    SIM_SEED,
    TAPE_SEEDS,
    build_synthesized_circuit,
    make_tape,
    tvd,
)
from psf_compile import GeodesicPSFHyper, SU4GeodesicPSFSynthesizer
from psf_ibm_real_submit import make_sampler_submit_fn
from psf_pennylane_gpu_ibm_prototype import is_isa_compliant, route_for_backend
from psf_pennylane_gpu_prototype import ConnectionContractError, reference_cpu_synthesize

_ZSX = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")


def main():
    backend = FakeManilaV2()
    noisy_submit = make_sampler_submit_fn(backend, seed_simulator=SIM_SEED)
    ideal_submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SIM_SEED)
    rows = []
    print(f"{'tape':>4} {'arm':20s} {'blk2q':>5} {'2q':>3} {'depth':>5} {'size':>4} {'gpu_diff':>9} "
          f"{'TVD_noisy':>9} {'TVD_ideal':>9} {'synth_ms':>8} {'fallbacks':>9}")
    for seed in TAPE_SEEDS:
        for arm in ("A_qiskit_default", "B_psf_zero", "C_qiskit_zsx"):
            stats = {"gpu_diffs": [], "synth_s": [], "block_twoq": []}
            psf = None
            if arm == "A_qiskit_default":
                synth_fn = reference_cpu_synthesize
            elif arm == "B_psf_zero":
                psf = SU4GeodesicPSFSynthesizer(
                    GeodesicPSFHyper(entangling_basis="cx", on_unsupported="raise"), verify=True
                )
                synth_fn = psf.synthesize
            else:
                synth_fn = _ZSX
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
            print(f"{seed:>4} {arm:20s} {row['block_twoq_total']:>5} {twoq:>3} {row['depth']:>5} "
                  f"{row['size']:>4} {row['gpu_diff_worst']:>9.2e} {row['tvd_noisy']:>9.4f} "
                  f"{row['tvd_ideal']:>9.4f} {row['synth_ms_median']:>8.3f} {str(fb):>9}", flush=True)

    with open("compare_zsx_arm_2026-09-24.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote compare_zsx_arm_2026-09-24.csv")


if __name__ == "__main__":
    main()
