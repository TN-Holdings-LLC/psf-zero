"""Prototype c67's check on CZ devices: c67 against release 2026-10-16.2, bit for bit, on C63-VAL's inputs (item
104 changes only ECR devices; the removal of option A's unreachable code changes nothing).

    python patches/psf_compile_c67_2026-10-17/check_c67_cz.py <out.json>   (from the PSF-Zero root)

As patches/psf_compile_c66_2026-10-16/check_release.py: per cell, `compile_template` of both on the build input; every
input (49,590) bound by both and compared bit for bit, with the rebuilt and switched counts; the default call compared
on the build input and the first 10 draws of every cell (2,090 inputs).
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
    M3 = load("psf_compile.py", "pc_release")  # the installed release, 2026-10-16.2
    MA = load(os.path.join("patches", "psf_compile_c67_2026-10-17", "psf_compile.py"), "pc67")
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


def tpl(m, qc, tg):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return m.compile_template(qc, tg)
    except Exception as e:  # noqa: BLE001 - a refusal compares by its type
        return ("error", type(e).__name__)


def bind(t, qc):
    if isinstance(t, tuple):
        return t
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return t.bind(qc)
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__)


def stats(m):
    s = m.TEMPLATE100_STATS
    return (s["rebuilt"], s["switched"])


res = dict(inputs=0, different=[], default_inputs=0, default_different=[], counters_different=[], cells={})
for part, devices, sets, free in (("R", ("FakeFez",), 1000, 100),
                                  ("D", ("FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh"), 50, 0)):
    for d in devices:
        t, ft = g["device"](d)
        for name in FORMULA + NATIVE:
            qc = build(name)
            for lab, tg in (("T", t), ("F", ft)):
                rng = random.Random(f"C63|{d}|{name}|{lab}")
                ta, tb = tpl(M3, qc, tg), tpl(MA, qc, tg)
                s3, sa = stats(M3), stats(MA)
                n = bad = 0
                for k in range(-1, sets + free):
                    qr = qc if k < 0 else randomize(qc, rng, same=k < sets)
                    n += 1
                    if fp(bind(ta, qr)) != fp(bind(tb, qr)):
                        bad += 1
                        if len(res["different"]) < 50:
                            res["different"].append([part, d, name, lab, k])
                    if k < 10:
                        res["default_inputs"] += 1
                        if fp(run(M3, qr, target=tg)) != fp(run(MA, qr, target=tg)):
                            res["default_different"].append([part, d, name, lab, k])
                d3 = tuple(x - y for x, y in zip(stats(M3), s3))
                da = tuple(x - y for x, y in zip(stats(MA), sa))
                if d3 != da:
                    res["counters_different"].append([part, d, name, lab, d3, da])
                res["inputs"] += n
                res["cells"][f"{part}|{d}|{name}|{lab}"] = [n, bad, list(d3)]
                print(part, d, name, lab, n, bad, d3, flush=True)
ok = not res["different"] and not res["default_different"] and not res["counters_different"]
res["verdict"] = "SAME" if ok else "DIFFERENT"
res["release_version"], res["c67_version"] = M3.VERSION, MA.VERSION
json.dump(res, open(OUT, "w"), indent=1)
print("2026-10-16.2 vs c67:", res["verdict"], "template inputs", res["inputs"], "different", len(res["different"]),
      "default inputs", res["default_inputs"], "different", len(res["default_different"]),
      "counters different", len(res["counters_different"]))
