"""R1: how sensitive is c54 to the written order of a symmetric rzz's two qubits? (hand-layered circuits as written
by family_map (pair as read, larger index first) and with every pair sorted)"""
import sys, os, contextlib, io, tempfile, importlib.util, json
sys.argv = ["family_map.py", tempfile.mkdtemp() + "/fm"]
spec = importlib.util.spec_from_file_location("fm", "benchmarks/family_map.py"); FM = importlib.util.module_from_spec(spec)
sys.modules["fm"] = FM; spec.loader.exec_module(FM)
C54 = FM.H.load_module("patches/psf_compile_c54_2026-10-14/psf_compile.py", "c54s"); C54.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeFez
b = FakeFez(); TG = {"T": b.target, "F": FM.fractional(b.target)}
def flip(c):
    o = QuantumCircuit(*c.qregs, *c.cregs, global_phase=c.global_phase)
    for i in c.data:
        qs = list(i.qubits)
        if i.operation.name == "rzz": qs = qs[::-1]
        o.append(i.operation, qs, i.clbits)
    return o
for name in ("qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3", "ba_n14"):
    qc = FM.build(name); hl = FM.hand_layered(qc)
    for tk, t in TG.items():
        r = []
        for lbl, c in (("as written", qc), ("HL fm order", hl), ("HL flipped", flip(hl)), ("input, rzz flipped", flip(qc) if any(i.operation.name=='rzz' for i in qc.data) else None)):
            if c is None: r.append(f"{lbl} -"); continue
            with contextlib.redirect_stdout(io.StringIO()):
                o = C54.compile_for_hardware(c, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            r.append(f"{lbl} {FM.dur(o, t):6.2f}")
        print(f"{name:16s} {tk} | " + " | ".join(r), flush=True)
