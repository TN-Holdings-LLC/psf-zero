"""
Tests the design critique raised after profile_synthesize_breakdown.py's
finding: if ~87% of synthesize()'s per-block cost is its own unconditional
Operator()-based fidelity self-check, and the underlying decomposition math
is already extensively validated offline (test_geometric_decompose.py:
worst-case 1-fidelity = 1.11e-15 over 1000 trials; this project's own stub:
8.88e-16 over 200 trials) -- why pay for re-verifying already-proven-correct
math on every single production call?

This does NOT touch the exception-based degenerate-point fallback (CNOT,
SWAP, iSWAP, identity, ...) -- that branch is load-bearing and stays exactly
as-is; it's the only way `synthesize()` finds out a given input needs the
CX-basis fallback path at all, and it costs almost nothing (see phase 1-3 of
the breakdown, ~13% of total). What this removes is ONLY the *unconditional
Operator() re-verification of every non-degenerate result*, on the premise
that the math doesn't need re-proving on every call once it's been proven
offline.

CAVEATS (read before trusting the numbers below):
  - Uses psf_zero_core_stub.py in-process, not the real Rust extension over
    PyO3 -- same caveat as profile_synthesize_breakdown.py.
  - This is a projection (real per-block time scaled by the fraction of
    time NOT spent in the fidelity check), applied to section 4's real,
    already-measured Qiskit numbers -- it is NOT a new real-hardware
    measurement of a "fast" compile(). Treat the resulting ratios as "this
    is what the real machine would plausibly show if re-measured with this
    change," not as a confirmed result. Re-running this same script's
    methodology against the real Rust core, and then re-measuring the full
    phase1.py/phase2.py pipeline with a verify=False synthesize(), is the
    actual confirmation step -- see Roadmap.
"""
import time
import numpy as np
from scipy.stats import unitary_group

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from psf_zero_core_stub import geometric_decompose

N = 2000
rng = np.random.default_rng(42)


def synthesize_verified(U):
    u_r, u_i = U.real.tolist(), U.imag.tolist()
    cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)
    qc = QuantumCircuit(2)
    qc.global_phase = global_phase

    def local(triple, qubit):
        phi, theta, lam = triple
        qc.rz(lam, qubit); qc.ry(theta, qubit); qc.rz(phi, qubit)

    local(k2[0], 1); local(k2[1], 0)
    a, b, c = cartan_angles
    if abs(a) > 1e-10: qc.rxx(-2 * a, 0, 1)
    if abs(b) > 1e-10: qc.ryy(-2 * b, 0, 1)
    if abs(c) > 1e-10: qc.rzz(-2 * c, 0, 1)
    local(k1[0], 1); local(k1[1], 0)

    U_out = Operator(qc).data
    tr = np.trace(U.conj().T @ U_out)
    fid = float((np.abs(tr) ** 2 + 4.0) / (4.0 * 5.0))
    if (1.0 - fid) > 1e-5:
        raise RuntimeError("fidelity check failed")
    return qc


def synthesize_fast(U):
    # Identical, minus the Operator() re-verification. The degenerate-point
    # fallback (the `except Exception` branch in the real code) is untouched
    # -- geometric_decompose() itself still raises on those inputs.
    u_r, u_i = U.real.tolist(), U.imag.tolist()
    cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)
    qc = QuantumCircuit(2)
    qc.global_phase = global_phase

    def local(triple, qubit):
        phi, theta, lam = triple
        qc.rz(lam, qubit); qc.ry(theta, qubit); qc.rz(phi, qubit)

    local(k2[0], 1); local(k2[1], 0)
    a, b, c = cartan_angles
    if abs(a) > 1e-10: qc.rxx(-2 * a, 0, 1)
    if abs(b) > 1e-10: qc.ryy(-2 * b, 0, 1)
    if abs(c) > 1e-10: qc.rzz(-2 * c, 0, 1)
    local(k1[0], 1); local(k1[1], 0)
    return qc


blocks = [unitary_group.rvs(4, random_state=rng) for _ in range(N)]

t0 = time.perf_counter()
outs_v = [synthesize_verified(U) for U in blocks]
t_verified = time.perf_counter() - t0

t0 = time.perf_counter()
outs_f = [synthesize_fast(U) for U in blocks]
t_fast = time.perf_counter() - t0

# Correctness check: fast path must produce the identical circuit (it's the
# exact same math, just without re-measuring it) -- confirmed via the same
# fidelity formula used elsewhere in this project, checked out-of-band here
# rather than per-call.
worst = 0.0
for U, qc in zip(blocks, outs_f):
    U_out = Operator(qc).data
    tr = np.trace(U.conj().T @ U_out)
    fid = float((np.abs(tr) ** 2 + 4.0) / (4.0 * 5.0))
    worst = max(worst, 1.0 - fid)

print(f"N = {N} random SU(4) blocks (stub core, in-process -- see script docstring)\n")
print(f"{'':<28}{'total (s)':>12}{'ms/block':>12}")
print(f"{'synthesize (verified, current)':<28}{t_verified:12.4f}{t_verified/N*1000:12.4f}")
print(f"{'synthesize_fast (no self-check)':<28}{t_fast:12.4f}{t_fast/N*1000:12.4f}")
print(f"\nSpeedup from dropping the per-call check: {t_verified/t_fast:.2f}x")
print(f"Worst-case (1 - fidelity) of the fast path's output, checked out-of-band: {worst:.2e}")
print("(should match the already-established stub accuracy, ~1e-15 -- confirms")
print(" the fast path isn't silently producing worse circuits, just not")
print(" re-proving it on every single call)")

print("\n--- Projected section 4 numbers if this change were real-hardware-verified ---")
scale_factor = t_fast / t_verified
real_data = [
    (15, 7, 0.0103, 0.0068),
    (50, 25, 0.0158, 0.0257),
    (100, 50, 0.0220, 0.0360),
    (156, 78, 0.0345, 0.0636),
    (300, 150, 0.0450, 0.0893),
    (500, 250, 0.0713, 0.1593),
    (1000, 500, 0.1319, 0.2957),
]
print(f"{'qubits':>7}{'blocks':>8}{'qiskit(s)':>11}{'psf, measured(s)':>18}{'psf, projected(s)':>19}{'ratio(q/psf,proj)':>19}")
for q, b, qt, pt in real_data:
    proj = pt * scale_factor
    print(f"{q:7d}{b:8d}{qt:11.4f}{pt:18.4f}{proj:19.4f}{qt/proj:19.2f}")
