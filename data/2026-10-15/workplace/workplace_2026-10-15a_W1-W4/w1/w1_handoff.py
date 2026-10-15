"""W1 check (workplace, 2026-10-15): a18's hand-off (item 33) outside the scope. For inputs that item 96 puts out of
scope on a device, is a18's output identical to Qiskit level 3's (seed 0, the device's Target; on a Target with rzz,
folded by c57's `_fold` as c57 does)? Only identity is measured. Offline.
    python w1_handoff.py"""
import contextlib
import io
import os
import sys
import tempfile
import importlib.util
import warnings

warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.argv = ["c57_val.py", tempfile.mkdtemp() + "/x"]
spec = importlib.util.spec_from_file_location("c57v", os.path.join(REPO, "benchmarks", "c57_val.py"))
V = importlib.util.module_from_spec(spec)
sys.modules["c57v"] = V
spec.loader.exec_module(V)
PC = V.load(os.path.join(REPO, "psf_compile.py"), "psf_compile")  # the release, under the name a18 imports
PC.WARN_WITHOUT_TARGET = False
AI = V.load(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "a18_w1")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import QFTGate  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402


def extra():
    """Small circuits of the kind a language model writes (no ring classifier)."""
    out = {}
    q = QuantumCircuit(6)
    q.h(range(6))
    q.append(QFTGate(6), range(6))
    q.measure_all()
    out["qft6"] = q
    q = QuantumCircuit(7)
    for a in range(7):
        for b in range(a + 1, 7):
            q.cx(a, b)
            q.rz(0.1 * (a + b), b)
    q.measure_all()
    out["allpairs_cx7"] = q
    q = QuantumCircuit(5)
    q.h(0)
    for b in range(1, 5):
        q.cz(0, b)
    for b in range(1, 5):
        q.cz(b, (b % 4) + 1)
    q.measure_all()
    out["wheel5"] = q
    return out


import time  # noqa: E402
AI.EXACT_CHECK = os.environ.get("W1_EXACT", "0") == "1"  # a18's item 30 check of the hand-off: see the timing probe
print("a18 EXACT_CHECK", AI.EXACT_CHECK, flush=True)
cases = extra()
for t in ("qaoa_n12_p2", "ba_n14", "qaoa3reg_n20_r3", "heis2d_n16_r1", "chem_n12_r1"):
    t0 = time.perf_counter()
    cases[t] = V.build(t)
    print(f"built {t} in {time.perf_counter() - t0:.1f} s", flush=True)
T, F = V.device("FakeFez")
for name, qc in cases.items():
    for tk, tg in (("T", T), ("F", F)):
        if V.in_scope(qc, tg):
            print(f"{name:16s} {tk} in scope: not measured")
            continue
        basis = [g for g in tg.operation_names if g in ("cz", "rz", "sx", "x", "rzz", "rx")]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out, info = AI.compile_for_model_circuit(qc, tg.build_coupling_map(), basis, target=tg, return_info=True)
        ref = generate_preset_pass_manager(3, target=tg, seed_transpiler=0).run(qc)
        refs = {"QK3": V.sig(ref)}
        if "rzz" in tg.operation_names:
            refs["QK3+fold"] = V.sig(PC._fold(ref, tg)) if not PC._rzz_in_range(ref) else refs["QK3"]
        same = [k for k, s in refs.items() if s == V.sig(out)]
        print(f"{name:16s} {tk} path {info.get('path')}/{info.get('chosen')}: identical to {same or 'NONE'}; "
              f"exact {info.get('exact')}; {time.perf_counter() - t0:.1f} s", flush=True)
print("A18_STATS", AI.A18_STATS)
