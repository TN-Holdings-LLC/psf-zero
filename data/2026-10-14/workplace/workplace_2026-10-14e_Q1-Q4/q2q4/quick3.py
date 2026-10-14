import os, sys, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_common import *  # noqa
from qiskit_ibm_runtime.fake_provider import FakeFez
NEW = compiler(os.environ.get("PSF_NEW92", "psf_compile_c53_91_92.py"), "c53_9192")
T = FakeFez().target; F = fractional(T)
for n in (4, 6, 8, 11):
    for s in (2, 4, 8):
        qc = ising(n, ring(n), s, signs=True)
        r = []
        for tn, tg in (("T", T), ("F", F)):
            o92 = NEW._ring_brickwork(qc, tg, 0.5, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                o53 = NEW._compile_for_hardware_c51(qc, target=tg, seed_transpiler=0)
            r.append(f"{tn}: 92 {dur_2q(o92, tg)[0]:5.2f} ({dur_2q(o92, tg)[1]:3d}) c53 {dur_2q(o53, tg)[0]:5.2f} ({dur_2q(o53, tg)[1]:3d})")
        print(f"ring ising n={n:2d} s={s}", " | ".join(r), flush=True)
qc = heis_xyz(8, ring(8), 2)
for tn, tg in (("T", T), ("F", F)):
    o92 = NEW._ring_brickwork(qc, tg, 0.5, 0)
    with contextlib.redirect_stdout(io.StringIO()):
        o53 = NEW._compile_for_hardware_c51(qc, target=tg, seed_transpiler=0)
    print("ring xyz(by type) n=8 s=2", tn, "92", dur_2q(o92, tg)[:2], "c53", dur_2q(o53, tg)[:2], "err", err_dist(o92, qc, 8))
