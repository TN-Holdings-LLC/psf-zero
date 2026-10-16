"""C63A-VAL, home part, A-D: c63a's default call against c63's on every C63-VAL input.

    python data/2026-10-16/c63a_val_home/a_d.py <out.json>   (from the PSF-Zero root)

Inputs as C63-VAL generates them (c63_val.py: the build input, then per cell `randomize` with
random.Random(f"C63|{device}|{test}|{T or F}"): part R on FakeFez, 1,000 same-sign and 100 free draws; part D on
FakeMarrakesh, FakeKingston, FakeTorino and the Marrakesh snapshot, 50 same-sign draws): 49,590 inputs. Per input,
`compile_for_hardware(qc, target=)` of both; compared bit for bit (every operation's name, qubits, clbits and angles as a
float's hex, the global phase, the initial and final layout); a refusal by its exception type. Also the refusals of
C63-VAL (no Target, every test; a three-qubit gate on every device and target).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import runpy
import shutil
import sys
import tempfile
import warnings

warnings.simplefilter("ignore")
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
W34 = os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15c_W1-W3", "w34")
_tmp = tempfile.mkdtemp() + "/imp"
sys.argv = ["c55_val", _tmp]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="a_d_import")
shutil.rmtree(_tmp, ignore_errors=True)
sys.path[:0] = [os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "benchmarks"]
import c60_tests as TT  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


with contextlib.redirect_stdout(io.StringIO()):
    M3 = load("psf_compile.py", "pc_release")
    MA = load(os.path.join("patches", "psf_compile_c66_2026-10-16", "psf_compile.py"), "pc66")
M3.WARN_WITHOUT_TARGET = MA.WARN_WITHOUT_TARGET = False
FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")


def build(name):
    return g["build"](name) if name in FORMULA else TT.build(name)


def randomize(qc, rng, same=False):  # c63_val.py's, unchanged
    out = qc.copy_empty_like()
    for ins in qc.data:
        op = ins.operation
        if op.name == "PauliEvolution":
            H = op.operator
            if same:
                new = SparsePauliOp(H.paulis, [float(c.real) * rng.uniform(0.5, 1.5) for c in H.coeffs])
            else:
                new = SparsePauliOp(H.paulis, [rng.uniform(-1.5, 1.5) for _ in range(len(H))])
            op = PauliEvolutionGate(new, time=op.time, synthesis=op.synthesis)
        elif op.params and op.name not in ("barrier", "measure", "delay") and \
                all(isinstance(p, (int, float)) for p in op.params):
            op = op.__class__(*[(float(p) * rng.uniform(0.5, 1.5)) if same else rng.uniform(-math.pi, math.pi)
                                for p in op.params])
        out.append(op, ins.qubits, ins.clbits)
    return out


def fp(c):
    if isinstance(c, tuple):
        return repr(c)
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    ops = tuple((ci.operation.name, tuple(q[x] for x in ci.qubits), tuple(k[x] for x in ci.clbits),
                 tuple(str(p) if hasattr(p, "parameters") else float(p).hex() for p in ci.operation.params))
                for ci in c.data)
    lay = getattr(c, "_layout", None)
    il = None if lay is None else tuple(lay.initial_index_layout(filter_ancillas=False))
    fl = None if lay is None or lay.final_layout is None else tuple(lay.final_index_layout())
    gp = c.global_phase
    return hashlib.sha256(repr((ops, str(gp) if hasattr(gp, "parameters") else float(gp).hex(), il, fl,
                                c.num_qubits, c.num_clbits)).encode()).hexdigest()


def run(m, qc, **kw):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return m.compile_for_hardware(qc, **kw)
    except Exception as e:  # noqa: BLE001 - a refusal compares by its type
        return ("error", type(e).__name__)


res = dict(inputs=0, different=[], refusals=0, refusals_different=[], cells={})
for part, devices, sets, free in (("R", ("FakeFez",), 1000, 100),
                                  ("D", ("FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh"), 50, 0)):
    for d in devices:
        t, ft = g["device"](d)
        for name in FORMULA + NATIVE:
            qc = build(name)
            for lab, tg in (("T", t), ("F", ft)):
                rng = random.Random(f"C63|{d}|{name}|{lab}")
                n = bad = 0
                for k in range(-1, sets + free):
                    qr = qc if k < 0 else randomize(qc, rng, same=k < sets)
                    n += 1
                    if fp(run(M3, qr, target=tg)) != fp(run(MA, qr, target=tg)):
                        bad += 1
                        if len(res["different"]) < 50:
                            res["different"].append([part, d, name, lab, k])
                res["inputs"] += n
                res["cells"][f"{part}|{d}|{name}|{lab}"] = [n, bad]
                print(part, d, name, lab, n, bad, flush=True)
for name in FORMULA + NATIVE:  # refusal: no Target
    qc = build(name)
    res["refusals"] += 1
    if fp(run(M3, qc)) != fp(run(MA, qc)):
        res["refusals_different"].append(["no_target", name])
ccx = QuantumCircuit(3)
ccx.ccx(0, 1, 2)
for d in ("FakeFez", "FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh"):
    t, ft = g["device"](d)
    for lab, tg in (("T", t), ("F", ft)):
        res["refusals"] += 1
        if fp(run(M3, ccx, target=tg)) != fp(run(MA, ccx, target=tg)):
            res["refusals_different"].append(["ccx", d, lab])
res["verdict"] = "CONFIRMED" if not res["different"] and not res["refusals_different"] else "REFUTED"
json.dump(res, open(OUT, "w"), indent=1)
print("C5", res["verdict"], "inputs", res["inputs"], "different", len(res["different"]), "refusals", res["refusals"],
      "refusals different", len(res["refusals_different"]))
