import os, sys, contextlib, io, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
P = load(REPO + "/patches/psf_compile_c53_2026-10-14/psf_compile.py", "psf_compile"); P.WARN_WITHOUT_TARGET = False
A = load(REPO + "/patches/psf_ai_compile_a17_2026-10-14/psf_ai_compile.py", "a17")
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q4_rounds.py")).read().split("for n in range(4, 17)")[0])
old_rr = A.ring_rounds
def new_rr(n):
    o, r, left = schedule(n, True)
    assert not left
    return o, [([e for e in d], s) for d, s in r]
from qiskit_ibm_runtime.fake_provider import FakeFez
T = FakeFez().target
def clf(n, L, seed):
    from qiskit import QuantumCircuit
    rng = random.Random(seed); qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n): qc.ry(rng.uniform(-1, 1) * math.pi / 2, q)
        for q in range(n): qc.ry(rng.uniform(-math.pi, math.pi), q); qc.rz(rng.uniform(-math.pi, math.pi), q)
        for q in range(n): qc.cz(q, (q + 1) % n)
    for q in range(n): qc.ry(rng.uniform(-math.pi, math.pi), q)
    qc.measure_all(); return qc
basis = ["cz", "rz", "sx", "x"]
for n in (5, 6, 7, 8):
    for L in (2, 4, 8):
        qc = clf(n, L, 100 * n + L)
        res = []
        for nm, rr in (("N2", old_rr), ("Q4", new_rr)):
            A.ring_rounds = rr
            ring = A._ring_candidate(qc, T, basis, 0)
            res.append(f"{nm} ring {dur_2q(ring, T)[0]:6.2f} ({dur_2q(ring, T)[1]:3d}) err {W2.tvd(W2.probs(ring), W2.probs(qc)):.0e}")
        A.RING_FORMULA = False
        with contextlib.redirect_stdout(io.StringIO()):
            c, info = A.compile_for_model_circuit(qc, T.build_coupling_map(), basis, target=T, return_info=True)
        A.RING_FORMULA = True
        print(f"n={n} L={L} | " + " | ".join(res) + f" | a17 without ring {dur_2q(c, T)[0]:6.2f} ({info['chosen']})", flush=True)
