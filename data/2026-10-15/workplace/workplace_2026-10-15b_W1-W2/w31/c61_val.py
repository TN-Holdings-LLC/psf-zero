"""C61-VAL (workplace, 2026-10-15): prototype c61's compile template (item 100) against c60. Offline; no real device.

Tests: C58-VAL's 12 in-scope tests (items 87/91; ghz_n12 is item 99's) and W1's 7 CZ / CX ansatz tests (item 99).
Part R (FakeFez, T and F): per test the template is built once; then 1000 random inputs of the same structure (every
gate angle uniform in [-pi, pi]; every Pauli coefficient uniform in [-1.5, 1.5], same labels and times) are bound
(`bind_values(values_of(qc))`) and compared with c60's own compilation of the same input. Recorded per input:
identical by value (value_sig), exactness (<= 20 active qubits), native, failed elements, rzz range, duration and
two-qubit count of both; per test: build time, c60 compile time, bind time (bind_values only; and values_of), Qiskit's
alternative where it applies (level 3 on a parametrised copy, then assign_parameters), rebuilds.
Part D (the other 4 devices, T and F): 50 random inputs per test, correctness only.
Part K: 100,000 binds of one chain (tfim_n12), T and F, angles on a random walk; every 1000th bind checked against
the exact distribution of the reader's terms; RSS every 1000 binds.

    python c61_val.py <out dir> [R|D|K ...]   (from the repository root; refuses an existing out dir)"""
import contextlib
import gc
import io
import json
import math
import os
import random
import resource
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
PARTS = sys.argv[2:] or ["R", "D", "K"]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
_tmp = tempfile.mkdtemp() + "/c55imp"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c61_val_import")
shutil.rmtree(_tmp, ignore_errors=True)
os.makedirs(OUT)
sys.path[:0] = [os.path.join(HERE, "..", "w30"), os.environ.get("C60_TESTS_DIR", ""), "benchmarks"]
import c60_tests as TT  # noqa: E402
from c29_identity import value_sig  # noqa: E402

C61_FILE = os.environ.get("C61_FILE", os.path.join(HERE, "c61", "psf_compile.py"))
C60_FILE = os.environ.get("C60_FILE", os.path.join(HERE, "..", "w30", "c60", "psf_compile.py"))
M61 = g["load"](C61_FILE, "psf_compile_c61")
M60 = g["load"](C60_FILE, "psf_compile_c60")
M61.WARN_WITHOUT_TARGET = M60.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402

FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
SETS = int(os.environ.get("C61_SETS", "1000"))
FREE = int(os.environ.get("C61_FREE", "100"))
DSETS = int(os.environ.get("C61_DSETS", "50"))
LAPS = int(os.environ.get("C61_LAPS", "100000"))


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


def c60(qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return M60.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


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
    a = _probs(transpile(qc, basis_gates=M61._X93_BASIS, optimization_level=0))
    b = _probs(c)
    if a is None or b is None or len(a) != len(b):
        return None
    return float(abs(a - b).max())


EX20_EVERY = int(os.environ.get("C61_EX20_EVERY", "10"))  # exactness at 17-20 active qubits: every 10th input (cost)
ONLY = [x for x in os.environ.get("C61_ONLY", "").split(",") if x]


def check(qc, c, tg, ref, do_exact=True):
    fe, fq = M61._failed_elements(tg, 0.5)
    m = g["metrics"](c, tg)
    mr = g["metrics"](ref, tg)
    ex = fast_exact(qc, c) if m["active"] <= 20 and (do_exact or m["active"] <= 16) else None
    return dict(same=value_sig(c) == value_sig(ref), exact=ex, isa=m["isa"], rzz_ok=m["rzz_in_range"],
                failed=bool(M61._uses_failed(c, fe, fq)), dur=m["dur_us"], twoq=m["twoq"], dur60=mr["dur_us"],
                twoq60=mr["twoq"])


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
                rng = random.Random(f"C61|{d}|{name}|{lab}")
                t60s, tbs = [], []
                for _ in range(3):  # medians of three: one compile of c60, one template build
                    s = time.perf_counter()
                    ref0 = c60(qc, tg)
                    t60s.append(time.perf_counter() - s)
                    s = time.perf_counter()
                    tpl = M61.compile_template(qc, tg)
                    tbs.append(time.perf_counter() - s)
                t60, tb = statistics.median(t60s), statistics.median(tbs)
                before = dict(M61.TEMPLATE100_STATS)
                head = dict(part=part, device=d, test=name, target=lab, kind=tpl.kind, build_s=tb, c60_s=t60,
                            n_values=len(tpl.values_of(qc)), slots=len(tpl._slots0))
                rows, tbind, tval = [], [], []
                c0 = tpl.bind(qc)
                rows.append(dict(check(qc, c0, tg, ref0), kind="build", rebuilt=False))
                for k in range(sets + (FREE if part == "R" else 0)):
                    same = k < sets
                    qr = randomize(qc, rng, same=same)
                    s = time.perf_counter()
                    vals = tpl.values_of(qr)
                    tval.append(time.perf_counter() - s)
                    rb, sw = M61.TEMPLATE100_STATS["rebuilt"], M61.TEMPLATE100_STATS["switched"]
                    s = time.perf_counter()
                    c = tpl.bind_values(vals)
                    dt = time.perf_counter() - s
                    reb = M61.TEMPLATE100_STATS["rebuilt"] > rb or M61.TEMPLATE100_STATS["switched"] > sw
                    if same and not reb:
                        tbind.append(dt)
                    rows.append(dict(check(qr, c, tg, c60(qr, tg), do_exact=(k % EX20_EVERY == 0)),
                                     kind="same" if same else "free", rebuilt=reb,
                                     bind_ms=dt * 1000))
                after = M61.TEMPLATE100_STATS
                head.update(rebuilt=after["rebuilt"] - before["rebuilt"], switched=after["switched"] - before["switched"],
                            mismatch=after["structure_mismatch"] - before["structure_mismatch"],
                            bind_median_ms=statistics.median(tbind) * 1000 if tbind else None,
                            bind_max_ms=max(tbind) * 1000 if tbind else None,
                            values_of_median_ms=statistics.median(tval) * 1000 if tval else None)
                if part == "R":
                    head["qiskit_assign_median_ms"] = (lambda v: None if v is None else v * 1000)(
                        qiskit_assign(qc, tg, rng))
                head["rows"] = rows
                fh.write(json.dumps(head) + "\n")
                fh.flush()
                n_same = sum(r["same"] for r in rows)
                n_ex = sum(1 for r in rows if r["exact"] is not None and r["exact"] <= 1e-6)
                print(f"{part} {d[-12:]:12s} {name:16s} {lab} {tpl.kind:6s} same {n_same}/{len(rows)} exact "
                      f"{n_ex}/{len(rows)} dur<= {sum(r['dur'] <= r['dur60'] + 1e-9 for r in rows)} "
                      f"bind {head['bind_median_ms'] or 0:.3f} ms build {tb*1000:.1f} c60 {t60*1000:.1f} "
                      f"rebuilt {head['rebuilt']} switched {head['switched']}", flush=True)


def part_k(fh):
    t, ft = g["device"]("FakeFez")
    qc = build("tfim_n12")
    for lab, tg in (("T", t), ("F", ft)):
        tpl = M61.compile_template(qc, tg)
        gen = tpl._gen
        vals = tpl.values_of(qc)
        rng = random.Random("C61K|" + lab)
        times, checks, rss = [], [], []
        gc.collect()
        before = dict(M61.TEMPLATE100_STATS)
        start = time.perf_counter()
        for lap in range(1, LAPS + 1):
            vals = [v + rng.gauss(0, 0.05) for v in vals]
            s = time.perf_counter()
            out = tpl.bind_values(vals)
            times.append(time.perf_counter() - s)
            if lap % 1000 == 0:
                ref = QuantumCircuit(*qc.qregs, *qc.cregs)
                for (op, q), (a, b) in zip(gen.prefix, [gen.pre_sl[j] or (0, 0) for j in range(len(gen.prefix))]):
                    ref.append(op.__class__(*vals[a:b]) if b > a else op, [q])
                for k, (lb, _) in enumerate(gen.terms):
                    ref.append(PauliEvolutionGate(SparsePauliOp(lb), time=vals[gen.term0 + k]), range(gen.n))
                for q, cb in gen.meas:
                    ref.measure(gen.perm[q], cb)
                checks.append((lap, g["exactness"](ref, out)))
                rss.append((lap, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            if lap % 10000 == 0:
                print(f"K {lab} lap {lap}: median {statistics.median(times[-10000:])*1000:.3f} ms; last check "
                      f"{checks[-1][1]:.1e}; RSS {rss[-1][1]:.0f} MB; {time.perf_counter() - start:.0f} s", flush=True)
        ms = sorted(x * 1000 for x in times)
        after = M61.TEMPLATE100_STATS
        fh.write(json.dumps(dict(part="K", test="tfim_n12", target=lab, laps=LAPS, median_ms=ms[len(ms) // 2],
                                 p99_ms=ms[int(len(ms) * 0.99)], max_ms=ms[-1],
                                 worst_check=max(c for _, c in checks), checks=checks, rss=rss,
                                 rebuilt=after["rebuilt"] - before["rebuilt"],
                                 switched=after["switched"] - before["switched"])) + "\n")
        fh.flush()


head = dict(only=ONLY, ex20_every=EX20_EVERY, free=FREE, c61_sha256=g["sha"](C61_FILE), c60_sha256=g["sha"](C60_FILE), script_sha256=g["sha"](os.path.abspath(__file__)),
            sets=SETS, dsets=DSETS, laps=LAPS, version=M61.VERSION[:30])
json.dump(head, open(os.path.join(OUT, "c61_val_head.json"), "w"), indent=1)
with open(os.path.join(OUT, f"c61_val_rows_{'_'.join(PARTS)}.jsonl"), "a") as fh:
    for p in PARTS:
        if p == "R":
            part_r(fh)
        elif p == "D":
            part_r(fh, devices=("FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh"), sets=DSETS,
                   part="D")
        else:
            part_k(fh)
print("C61-VAL DONE")
