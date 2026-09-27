"""probe_lap2138.py -- Addendum 218: localize the loss error of lap 2,138 (Addendum 214
part E) at block_gate_floor 8: loss after consolidation, after PSF-Zero's compile(),
after compile_for_hardware, and every block's phase-aligned Frobenius distance.
Run from the repository root:  python benchmarks/probe_lap2138.py
"""
import contextlib
import io
import sys
import warnings

sys.path.insert(0, "benchmarks")
import numpy as np
import loop_endurance as le
import psf_compile as pc
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

print(pc.VERSION)
LINE = CouplingMap.from_line(12)
rng_e = np.random.default_rng(7)
tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
tt[8::15] = 0.0
target = Statevector(le.e_circuit(tt))
rng_t = np.random.default_rng(202)
for _ in range(2137):
    rng_t.normal(0.0, 0.5, le.E_NPARAMS)
qc = le.e_circuit(tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS))


def loss(c):
    return 1 - abs(target.inner(Statevector(c))) ** 2


cap = {}
orig_many = pc.SU4GeodesicPSFSynthesizer.synthesize_many


def many(self, us):
    out = orig_many(self, us)
    cap["us"], cap["out"] = us, out
    return out


pc.SU4GeodesicPSFSynthesizer.synthesize_many = many
orig_pm = pc.PassManager


class PM(orig_pm):
    def run(self, *a, **k):
        o = super().run(*a, **k)
        cap["blocked"] = o
        return o


pc.PassManager = PM
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter("ignore")
    pc._CX_CORE_CACHE.clear()
    comp = pc.compile(qc, block_gate_floor=8, entangling_basis="cx", on_unsupported="keep")
    pc._CX_CORE_CACHE.clear()
    full = pc.compile_for_hardware(qc, coupling_map=LINE, basis_gates=["cx", "rz", "sx", "x"], block_gate_floor=8,
                                   entangling_basis="cx", initial_layout=list(range(12)), on_unsupported="keep",
                                   seed_transpiler=0)
print(f"loss error: blocked (consolidation only) {abs(loss(cap['blocked']) - loss(qc)):.2e} | after psf compile() "
      f"{abs(loss(comp) - loss(qc)):.2e} | after compile_for_hardware {abs(loss(full) - loss(qc)):.2e}")
pairs = [tuple(sorted(cap["blocked"].find_bit(q).index for q in i.qubits)) for i in cap["blocked"].data
         if len(i.qubits) == 2 and i.operation.name == "unitary"]
for p, u, (c, fb) in zip(pairs, cap["us"], cap["out"]):
    print(f"  block {p} fallback={fb}: frob {pc._aligned_errors(u, c)[2]:.2e}")
