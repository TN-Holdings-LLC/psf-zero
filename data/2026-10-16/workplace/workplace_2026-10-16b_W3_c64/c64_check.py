"""C64 checks D1 and D5 (C64-PREREG.md): c64 against c63a on every input of C63-VAL. Offline; no real device.

D1: for each input (part R: FakeFez; part D: FakeMarrakesh, FakeKingston, FakeTorino, the Marrakesh snapshot; T and
F; the build input, then the same-sign and free draws with C63-VAL's seeds and order), `compile_for_hardware(qc,
target=tg)` and `compile_template(qc, tg).bind(qc)` of both versions: circuits equal under ==, initial and final index
layouts equal, or the same exception type. Refusals: no Target (each test), a ccx circuit (each device, T and F).
D5: on each cell's build input, compile_for_hardware timed alternately (c63a, c64), 3 times each.

    python c64_check.py <out dir> [R|D|X ...]   (from the repository root, bundle q; refuses an existing out dir)
    C64_ONLY=<tests> and C64_DEVICES=<part D devices> select a segment (run_d1.sh runs every segment)."""
import contextlib
import io
import json
import math
import os
import random
import runpy
import shutil
import statistics
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
PARTS = sys.argv[2:] or ["X", "R", "D"]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
_tmp = tempfile.mkdtemp() + "/c55imp"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c64_check_import")
shutil.rmtree(_tmp, ignore_errors=True)
os.makedirs(OUT)
W = os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30")
sys.path[:0] = [W, "benchmarks"]
import c60_tests as TT  # noqa: E402
from c29_identity import value_sig  # noqa: E402

A_FILE = os.environ.get("C63A_FILE", os.path.join(HERE, "..", "w34", "w3", "c63a", "psf_compile.py"))
B_FILE = os.environ.get("C64_FILE", os.path.join(HERE, "c64", "psf_compile.py"))
MA = g["load"](A_FILE, "psf_compile_c63a")
MB = g["load"](B_FILE, "psf_compile_c64")
MA.WARN_WITHOUT_TARGET = MB.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
SETS, FREE, DSETS = (int(os.environ.get(k, v)) for k, v in (("C64_SETS", 1000), ("C64_FREE", 100), ("C64_DSETS", 50)))
ONLY = [x for x in os.environ.get("C64_ONLY", "").split(",") if x]
DEVS = [x for x in os.environ.get("C64_DEVICES", "").split(",") if x]
D_DEVICES = ("FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh")


def build(name):
    return g["build"](name) if name in FORMULA else TT.build(name)


def randomize(qc, rng, same=False):
    """The same circuit with new random angles. same=False (free): rotation parameters uniform in [-pi, pi], Pauli
    coefficients uniform in [-1.5, 1.5]. same=True (aimed at the same signature): every rotation parameter and Pauli
    coefficient multiplied by a factor uniform in [0.5, 1.5] (signs kept). Labels, times and synthesis kept."""
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


def lay(c):
    L = getattr(c, "layout", None)
    if L is None:
        return None
    try:
        ini = tuple(L.initial_index_layout(filter_ancillas=False))
    except Exception as e:  # noqa: BLE001
        ini = "error " + type(e).__name__
    try:
        fin = tuple(L.final_index_layout(filter_ancillas=False))
    except Exception as e:  # noqa: BLE001
        fin = "error " + type(e).__name__
    return ini, fin


def run(f):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return f(), None
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__


def both(qc, tg):
    """(result per call: equal?, value_sig equal?, detail) for compile_for_hardware and compile_template().bind()."""
    res = {}
    for call in ("cfh", "tpl"):
        outs = []
        for M in (MA, MB):
            if call == "cfh":
                outs.append(run(lambda: M.compile_for_hardware(qc, target=tg)))
            else:
                outs.append(run(lambda: M.compile_template(qc, tg).bind(qc)))
        (ca, ea), (cb, eb) = outs
        if ea or eb:
            res[call] = dict(eq=ea == eb, sig=ea == eb, a=ea, b=eb)
        else:
            res[call] = dict(eq=bool(ca == cb) and lay(ca) == lay(cb), sig=value_sig(ca) == value_sig(cb),
                             circ=bool(ca == cb), layout=lay(ca) == lay(cb))
    return res


def time_cell(qc, tg):
    ta, tb = [], []
    for _ in range(3):
        for M, acc in ((MA, ta), (MB, tb)):
            s = time.perf_counter()
            run(lambda: M.compile_for_hardware(qc, target=tg))
            acc.append(time.perf_counter() - s)
    return statistics.median(ta), statistics.median(tb)


json.dump(dict(c63a_sha256=g["sha"](A_FILE), c64_sha256=g["sha"](B_FILE), script_sha256=g["sha"](os.path.abspath(__file__)),
               versions=[MA.VERSION[:16], MB.VERSION[:16]], sets=SETS, free=FREE, dsets=DSETS),
          open(os.path.join(OUT, "c64_check_head.json"), "w"), indent=1)
fh = open(os.path.join(OUT, "c64_check_rows.jsonl"), "a")
start = time.perf_counter()
if "X" in PARTS:  # refusals
    t, ft = g["device"]("FakeFez")
    for name in FORMULA + NATIVE:
        qc = build(name)
        r = {}
        for call in ("cfh", "tpl"):
            o = [run(lambda: M.compile_for_hardware(qc)) if call == "cfh" else run(lambda: M.compile_template(qc, None))
                 for M in (MA, MB)]
            r[call] = dict(a=o[0][1], b=o[1][1], eq=o[0][1] == o[1][1] and o[0][1] is not None)
        fh.write(json.dumps(dict(part="X", kind="no_target", test=name, res=r)) + "\n")
    for d in ("FakeFez",) + D_DEVICES:
        t, ft = g["device"](d)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.ccx(0, 1, 2)
        qc.measure_all()
        for lab, tg in (("T", t), ("F", ft)):
            res = both(qc, tg)
            fh.write(json.dumps(dict(part="X", kind="ccx", device=d, target=lab, res=res)) + "\n")
    fh.flush()
    print("X done", flush=True)
for part in [p for p in PARTS if p in ("R", "D")]:
    devices = ("FakeFez",) if part == "R" else tuple(x for x in D_DEVICES if not DEVS or x in DEVS)
    for d in devices:
        t, ft = g["device"](d)
        for name in [x for x in FORMULA + NATIVE if not ONLY or x in ONLY]:
            qc = build(name)
            for lab, tg in (("T", t), ("F", ft)):
                rng = random.Random(f"C63|{d}|{name}|{lab}")
                ta, tb = time_cell(qc, tg)
                n, bad, sig_bad, details = 0, 0, 0, []
                count = SETS + FREE if part == "R" else DSETS
                for k in range(-1, count):
                    q = qc if k < 0 else randomize(qc, rng, same=(k < SETS))
                    res = both(q, tg)
                    n += 1
                    ok = all(v["eq"] for v in res.values())
                    bad += not ok
                    sig_bad += not all(v["sig"] for v in res.values())
                    if not ok and len(details) < 5:
                        details.append(dict(k=k, res=res))
                fh.write(json.dumps(dict(part=part, device=d, test=name, target=lab, inputs=n, unequal=bad,
                                         sig_unequal=sig_bad, details=details, t_c63a=ta, t_c64=tb)) + "\n")
                fh.flush()
                print(f"{part} {d[-12:]:12s} {name:16s} {lab} inputs {n} unequal {bad} sig {sig_bad} "
                      f"time {tb / ta:.3f} ({time.perf_counter() - start:.0f} s)", flush=True)
print("C64 CHECK DONE")
