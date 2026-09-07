"""
Profiles SU4GeodesicPSFSynthesizer.synthesize()'s four sub-phases separately,
to test the hypothesis (README "What we haven't verified yet") that the
per-block cost gap found in section 4 comes from Python-level overhead
around the decomposition call, not the decomposition itself.

CAVEAT: this uses psf_zero_core_stub.py (an in-process Python function), not
the real psf_zero_core Rust extension reached over PyO3. That means the
"decompose" phase timed here is an UNDER-estimate of the real FFI round trip
(no cross-language marshalling, no GIL release/reacquire) -- if anything,
this makes the real core's share of the total look smaller than it likely
is, and the other three phases' share look larger. So this cannot give the
final answer, only a directional one: does most of the time go to the
decomposition call itself, or to the Python/Qiskit-object machinery around
it? Confirming the real split needs this same breakdown run against the
real Rust core, on the real machine.
"""
import time
import numpy as np
from scipy.stats import unitary_group

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from psf_zero_core_stub import geometric_decompose

N = 2000
rng = np.random.default_rng(42)

t_extract = 0.0
t_decompose = 0.0
t_build = 0.0
t_verify = 0.0

for _ in range(N):
    U = unitary_group.rvs(4, random_state=rng)

    t0 = time.perf_counter()
    u_r = U.real.tolist()
    u_i = U.imag.tolist()
    t1 = time.perf_counter()
    t_extract += t1 - t0

    cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)
    t2 = time.perf_counter()
    t_decompose += t2 - t1

    qc = QuantumCircuit(2)
    qc.global_phase = global_phase

    def local(triple, qubit):
        phi, theta, lam = triple
        qc.rz(lam, qubit)
        qc.ry(theta, qubit)
        qc.rz(phi, qubit)

    local(k2[0], 1)
    local(k2[1], 0)
    a, b, c = cartan_angles
    if abs(a) > 1e-10: qc.rxx(-2 * a, 0, 1)
    if abs(b) > 1e-10: qc.ryy(-2 * b, 0, 1)
    if abs(c) > 1e-10: qc.rzz(-2 * c, 0, 1)
    local(k1[0], 1)
    local(k1[1], 0)
    t3 = time.perf_counter()
    t_build += t3 - t2

    # the "no silent fallback" self-check every block pays in production
    U_out = Operator(qc).data
    tr = np.trace(U.conj().T @ U_out)
    fid = float((np.abs(tr) ** 2 + 4.0) / (4.0 * 5.0))
    t4 = time.perf_counter()
    t_verify += t4 - t3

total = t_extract + t_decompose + t_build + t_verify

print(f"N = {N} random SU(4) blocks (psf_zero_core_stub.py, in-process -- see caveat above)\n")
print(f"{'phase':<30}{'total (s)':>12}{'per-block (ms)':>16}{'share':>10}")
for name, t in [
    ("1. matrix -> list (u_r/u_i)", t_extract),
    ("2. geometric_decompose()", t_decompose),
    ("3. circuit construction", t_build),
    ("4. Operator() fidelity check", t_verify),
]:
    print(f"{name:<30}{t:12.4f}{t/N*1000:16.4f}{t/total*100:9.1f}%")
print(f"{'TOTAL':<30}{total:12.4f}{total/N*1000:16.4f}{100.0:9.1f}%")
