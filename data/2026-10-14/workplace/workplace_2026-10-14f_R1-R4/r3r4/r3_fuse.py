"""R3: does fusing each edge's consecutive XX, YY, ZZ terms into one two-qubit gate shorten c54's output? (heis2d_n16_r1,
heis_n12_r2 from gens; the 16-qubit heavy-hex tree Heisenberg of C54-VAL is not rebuilt here)"""
import sys, os, contextlib, io, tempfile, importlib.util
sys.argv = ["family_map.py", tempfile.mkdtemp() + "/fm"]
spec = importlib.util.spec_from_file_location("fm", "benchmarks/family_map.py"); FM = importlib.util.module_from_spec(spec)
sys.modules["fm"] = FM; spec.loader.exec_module(FM)
P = FM.H.load_module("patches/psf_compile_c54_2026-10-14/psf_compile.py", "c54r3"); P.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from qiskit_ibm_runtime.fake_provider import FakeFez
b = FakeFez(); TG = {"T": b.target, "F": FM.fractional(b.target)}
def fused(qc):
    prefix, meas, triples, phase, suffix, perm = FM.terms_of(qc)
    n = qc.num_qubits
    out = QuantumCircuit(n, qc.num_clbits); i = 0
    for op, q in prefix: out.append(op, [q])
    T = [(lb, a * t) for lb, a, t in triples]
    while i < len(T):
        lb, ang = T[i]
        sup = tuple(sorted(n - 1 - j for j, ch in enumerate(lb) if ch != "I"))
        group = [(lb, ang)]
        while i + len(group) < len(T):
            lb2, a2 = T[i + len(group)]
            sup2 = tuple(sorted(n - 1 - j for j, ch in enumerate(lb2) if ch != "I"))
            if sup2 != sup: break
            group.append((lb2, a2))
        if len(sup) == 2:
            H = sum(SparsePauliOp("".join(l[n - 1 - q] for q in reversed(sup))).to_matrix() * a for l, a in group)
            out.append(UnitaryGate(expm(-1j * H)), list(sup))
        else:
            for l, a in group:
                from qiskit.circuit.library import PauliEvolutionGate
                out.append(PauliEvolutionGate(SparsePauliOp("".join(l[n - 1 - q] for q in reversed(sup))), time=a), list(sup))
        i += len(group)
    for op, w in suffix: out.append(op, [w])
    for q, c in meas: out.measure(perm[q] if perm else q, c)
    return out
for name in ("heis_n12_r2", "heis2d_n16_r1"):
    qc = FM.build(name); fz = fused(qc)
    print(name, "fused exact vs input:", FM.exact(qc, fz))
    for tk, t in TG.items():
        with contextlib.redirect_stdout(io.StringIO()):
            a = P.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            f = P.compile_for_hardware(fz, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)
        print(f"  {tk} c54 as written {FM.dur(a, t):6.2f} ({FM.n2(a)}) | c54 on fused {FM.dur(f, t):6.2f} ({FM.n2(f)}) exact {FM.exact(qc, f)}", flush=True)
