"""C63-VAL (workplace, 2026-10-15): prototype c63's compile template (item 100 with planned binding, built on c62)
against c62's default call. Offline; no real device.

Tests: C58-VAL's 12 in-scope tests (items 87/91; ghz_n12 is item 99's) and W1's 7 CZ / CX ansatz tests (item 99).
Part R (FakeFez, T and F): per test the template is built once; then 1000 inputs of the same structure aimed at the
same signature (every angle and Pauli coefficient times a factor uniform in [0.5, 1.5]) and 100 free inputs (angles
uniform in [-pi, pi], coefficients uniform in [-1.5, 1.5]) are bound (`bind_values(values_of(qc))`) and compared with
c62's own compilation of the same input. Recorded per input: identical by value (value_sig), exactness (<= 20 active
qubits), native, failed elements, rzz range, duration and two-qubit count of both; per test: build time, c62 compile
time, bind time (bind_values only), values_of time, the bind's own assign_parameters time, Qiskit's alternative where
it applies (level 3 on a parametrised copy, then assign_parameters), rebuilds.
Part D (the other 4 devices, T and F): 50 inputs aimed at the same signature per test, correctness only.

    python c63_val.py <out dir> [R|D ...]   (from the repository root, bundle q; refuses an existing out dir)"""
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
PARTS = sys.argv[2:] or ["R", "D"]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
_tmp = tempfile.mkdtemp() + "/c55imp"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c63_val_import")
shutil.rmtree(_tmp, ignore_errors=True)
os.makedirs(OUT)
W = os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30")
sys.path[:0] = [W, "benchmarks"]
import c60_tests as TT  # noqa: E402
from c29_identity import value_sig  # noqa: E402

C63_FILE = os.environ.get("C63_FILE", os.path.join(HERE, "c63", "psf_compile.py"))
C62_FILE = os.environ.get("C62_FILE", os.path.join(HERE, "c62", "psf_compile.py"))
M63 = g["load"](C63_FILE, "psf_compile_c63")
M62 = g["load"](C62_FILE, "psf_compile_c62")
M63.WARN_WITHOUT_TARGET = M62.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402

FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
SETS = int(os.environ.get("C63_SETS", "1000"))
FREE = int(os.environ.get("C63_FREE", "100"))
DSETS = int(os.environ.get("C63_DSETS", "50"))


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


def c62(qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return M62.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


_SIM = []


def _probs(circ):
    """The exact outcome distribution over the circuit's classical bits as an array (index bit k = clbit k), by Aer's
    double-precision state vector on the active qubits; None if not every measurement is final."""
    import numpy as np
    from qiskit_aer import AerSimulator
    if not _SIM:
        _SIM.append(AerSimulator(method="statevector", precision="double"))
    meas, body, done = {}, [], set()
    for ins in circ.data:
        nm = ins.operation.name
        qs = [circ.find_bit(q).index for q in ins.qubits]
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = qs[0]
            done.add(qs[0])
            continue
        if nm in ("barrier", "delay"):
            continue
        if ins.clbits or any(q in done for q in qs):
            return None
        body.append((ins.operation, qs))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active))
    for op, qs in body:
        small.append(op, [pos[q] for q in qs])
    small.save_statevector()
    sv = np.asarray(_SIM[0].run(small).result().get_statevector())
    na = len(active)
    p = (np.abs(sv) ** 2).reshape([2] * na)  # axis j holds qubit na-1-j
    ks = sorted(meas)
    axes = [na - 1 - pos[meas[k]] for k in ks]
    p = p.sum(axis=tuple(a for a in range(na) if a not in axes))
    rem = sorted(axes)  # the remaining axes, in increasing order
    p = np.transpose(p, [rem.index(a) for a in axes])  # axis i now holds clbit ks[i]
    out = np.zeros(2 ** circ.num_clbits)
    flat = p.reshape(-1)  # C order: axis 0 most significant
    m = len(ks)
    idx = np.arange(flat.size)
    key = np.zeros(flat.size, dtype=np.int64)
    for i in range(m):
        key |= ((idx >> (m - 1 - i)) & 1) << ks[i]
    np.add.at(out, key, flat)
    return out


def fast_exact(qc, c):
    """Largest outcome-probability difference between the input (its own synthesis, as item 93 reads it) and c."""
    from qiskit import transpile
    a = _probs(transpile(qc, basis_gates=M63._X93_BASIS, optimization_level=0))
    b = _probs(c)
    if a is None or b is None or len(a) != len(b):
        return None
    return float(abs(a - b).max())


EX20_EVERY = int(os.environ.get("C63_EX20_EVERY", "10"))  # exactness at 17-20 active qubits: every 10th input (cost)
ONLY = [x for x in os.environ.get("C63_ONLY", "").split(",") if x]


def check(qc, c, tg, ref, do_exact=True):
    fe, fq = M63._failed_elements(tg, 0.5)
    m = g["metrics"](c, tg)
    mr = g["metrics"](ref, tg)
    ex = fast_exact(qc, c) if m["active"] <= 20 and (do_exact or m["active"] <= 16) else None
    return dict(same=value_sig(c) == value_sig(ref), exact=ex, isa=m["isa"], rzz_ok=m["rzz_in_range"],
                failed=bool(M63._uses_failed(c, fe, fq)), dur=m["dur_us"], twoq=m["twoq"], dur62=mr["dur_us"],
                twoq62=mr["twoq"])


def qiskit_assign(qc, tg, rng, n=200):
    """Qiskit's alternative for a gate-level input: its parameters replaced by Parameters, transpiled once by level 3,
    then assign_parameters per bind (median seconds); None for an input with a Pauli evolution."""
    if any(i.operation.name == "PauliEvolution" for i in qc.data):
        return None
    pc = qc.copy_empty_like()
    ps = []
    for ins in qc.data:
        op = ins.operation
        if op.params and op.name not in ("barrier", "measure"):
            new = [Parameter(f"p{len(ps) + i}") for i in range(len(op.params))]
            ps += new
            op = op.__class__(*new)
        pc.append(op, ins.qubits, ins.clbits)
    out = generate_preset_pass_manager(3, target=tg, seed_transpiler=0).run(pc)
    order = sorted(out.parameters, key=lambda p: int(p.name[1:]))
    ts = []
    for _ in range(n):
        vals = {p: rng.uniform(-math.pi, math.pi) for p in order}
        s = time.perf_counter()
        out.assign_parameters(vals)
        ts.append(time.perf_counter() - s)
    return statistics.median(ts)


def part_r(fh, devices=("FakeFez",), sets=None, part="R"):
    sets = SETS if sets is None else sets
    for d in devices:
        t, ft = g["device"](d)
        for name in [x for x in FORMULA + NATIVE if not ONLY or x in ONLY]:
            qc = build(name)
            for lab, tg in (("T", t), ("F", ft)):
                rng = random.Random(f"C63|{d}|{name}|{lab}")
                t62s, tbs = [], []
                for _ in range(3):  # medians of three: one compile of c62, one template build
                    s = time.perf_counter()
                    ref0 = c62(qc, tg)
                    t62s.append(time.perf_counter() - s)
                    s = time.perf_counter()
                    tpl = M63.compile_template(qc, tg)
                    tbs.append(time.perf_counter() - s)
                t62, tb = statistics.median(t62s), statistics.median(tbs)
                before = dict(M63.TEMPLATE100_STATS)
                head = dict(part=part, device=d, test=name, target=lab, kind=tpl.kind, build_s=tb, c62_s=t62,
                            n_values=len(tpl.values_of(qc)), slots=len(tpl._slots0))
                rows, tbind, tval, svals = [], [], [], []
                c0 = tpl.bind(qc)
                rows.append(dict(check(qc, c0, tg, ref0), kind="build", rebuilt=False))
                for k in range(sets + (FREE if part == "R" else 0)):
                    same = k < sets
                    qr = randomize(qc, rng, same=same)
                    s = time.perf_counter()
                    vals = tpl.values_of(qr)
                    tval.append(time.perf_counter() - s)
                    if same:
                        svals.append(vals)
                    rb, sw = M63.TEMPLATE100_STATS["rebuilt"], M63.TEMPLATE100_STATS["switched"]
                    s = time.perf_counter()
                    c = tpl.bind_values(vals)
                    dt = time.perf_counter() - s
                    reb = M63.TEMPLATE100_STATS["rebuilt"] > rb or M63.TEMPLATE100_STATS["switched"] > sw
                    if same and not reb:
                        tbind.append(dt)
                    rows.append(dict(check(qr, c, tg, c62(qr, tg), do_exact=(k % EX20_EVERY == 0)),
                                     kind="same" if same else "free", rebuilt=reb,
                                     bind_ms=dt * 1000))
                tight = []  # the same-sign values bound back to back (no compile or simulation in between)
                for vals in svals:
                    rb, sw = M63.TEMPLATE100_STATS["rebuilt"], M63.TEMPLATE100_STATS["switched"]
                    s = time.perf_counter()
                    tpl.bind_values(vals)
                    dt = time.perf_counter() - s
                    if M63.TEMPLATE100_STATS["rebuilt"] == rb and M63.TEMPLATE100_STATS["switched"] == sw:
                        tight.append(dt)
                head["bind_tight_median_ms"] = statistics.median(tight) * 1000 if tight else None
                head["bind_tight_n"] = len(tight)
                after = M63.TEMPLATE100_STATS
                head.update(rebuilt=after["rebuilt"] - before["rebuilt"], switched=after["switched"] - before["switched"],
                            mismatch=after["structure_mismatch"] - before["structure_mismatch"],
                            bind_median_ms=statistics.median(tbind) * 1000 if tbind else None,
                            bind_max_ms=max(tbind) * 1000 if tbind else None,
                            values_of_median_ms=statistics.median(tval) * 1000 if tval else None)
                if part == "R":
                    arng = random.Random(f"C63A|{name}|{lab}")
                    ats = []
                    for _ in range(200):  # the bind's own assign_parameters, on fresh slot values
                        sl = [arng.uniform(-math.pi, math.pi) for _ in tpl._slots0]
                        s = time.perf_counter()
                        tpl.circuit.assign_parameters(sl)
                        ats.append(time.perf_counter() - s)
                    head["own_assign_median_ms"] = statistics.median(ats) * 1000
                    head["qiskit_assign_median_ms"] = (lambda v: None if v is None else v * 1000)(
                        qiskit_assign(qc, tg, rng))
                head["rows"] = rows
                fh.write(json.dumps(head) + "\n")
                fh.flush()
                n_same = sum(r["same"] for r in rows)
                n_ex = sum(1 for r in rows if r["exact"] is not None and r["exact"] <= 1e-6)
                print(f"{part} {d[-12:]:12s} {name:16s} {lab} {tpl.kind:6s} same {n_same}/{len(rows)} exact "
                      f"{n_ex}/{len(rows)} dur<= {sum(r['dur'] <= r['dur62'] + 1e-9 for r in rows)} "
                      f"bind {head['bind_median_ms'] or 0:.3f} ms tight {head['bind_tight_median_ms'] or 0:.3f} ms build {tb*1000:.1f} c62 {t62*1000:.1f} "
                      f"rebuilt {head['rebuilt']} switched {head['switched']}", flush=True)


head = dict(only=ONLY, ex20_every=EX20_EVERY, free=FREE, c63_sha256=g["sha"](C63_FILE), c62_sha256=g["sha"](C62_FILE),
            script_sha256=g["sha"](os.path.abspath(__file__)), sets=SETS, dsets=DSETS, version=M63.VERSION[:30])
json.dump(head, open(os.path.join(OUT, "c63_val_head.json"), "w"), indent=1)
with open(os.path.join(OUT, f"c63_val_rows_{'_'.join(PARTS)}.jsonl"), "a") as fh:
    for p in PARTS:
        if p == "R":
            part_r(fh)
        elif p == "D":
            part_r(fh, devices=("FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh"), sets=DSETS,
                   part="D")
print("C63-VAL DONE")
