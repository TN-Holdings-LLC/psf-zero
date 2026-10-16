"""C66-VAL copy of W3's w3_walk.py: variant A's file from $W3_A_FILE; HERE is W3's folder from the PSF-Zero root.

W3 (workplace, 2026-10-15; measurement only, home decides): item 88's frame in a training loop.
C61-VAL part K's conditions: tfim_n12, FakeFez with fractional gates (F), 100,000 binds on a random walk (every value
plus N(0, 0.05) per lap, the same seed "C61K|F"). One variant per process (peak RSS is per process):
  A  c63a: the frame fixed in the template (sign changes by local X and pinned runs; TEMPLATE_CACHE 256)
  B  c63 with TEMPLATE_CACHE = 32 (the current output, a smaller cache)
  C  c63 as validated (TEMPLATE_CACHE 256), for reference
Recorded: shape changes (rebuilt + switched), bind time per lap (median, p99, max), RSS every 1000 laps (peak), and
every 1000th lap: exactness against the reader's terms (Aer state vector), duration and two-qubit count of the output
and of c62's default call on the same input.
    python w3_walk.py <A|B|C> <out.json>   (from the repository root, bundle q)"""
import contextlib
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
HERE = os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15c_W1-W3", "w34", "w3")  # C66-VAL: from the root
VAR, OUTF = sys.argv[1], sys.argv[2]
LAPS = int(os.environ.get("W3_LAPS", "100000"))
if os.path.exists(OUTF):
    raise SystemExit(f"STOP: {OUTF} exists")
_tmp = tempfile.mkdtemp() + "/c55imp"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="w3_import")
shutil.rmtree(_tmp, ignore_errors=True)
FILES = {"A": os.environ.get("W3_A_FILE", os.path.join(HERE, "c63a", "psf_compile.py")), "B": os.path.join(HERE, "..", "c63", "psf_compile.py"),
         "C": os.path.join(HERE, "..", "c63", "psf_compile.py")}
M = g["load"](FILES[VAR], "psf_compile_w3")
M62 = g["load"](os.path.join(HERE, "..", "c62", "psf_compile.py"), "psf_compile_c62")
M.WARN_WITHOUT_TARGET = M62.WARN_WITHOUT_TARGET = False
if VAR == "B":
    M.TEMPLATE_CACHE = 32
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
M63 = M


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




def reader_circuit(qc, gen, vals):
    """The input as the reader sees it, with the walk's values: prefix gates, one Pauli evolution per term, the
    measurements (C61-VAL part K's reference)."""
    ref = QuantumCircuit(*qc.qregs, *qc.cregs)
    for (op, q), (a, b) in zip(gen.prefix, [gen.pre_sl[j] or (0, 0) for j in range(len(gen.prefix))]):
        ref.append(op.__class__(*vals[a:b]) if b > a else op, [q])
    for k, (lb, _) in enumerate(gen.terms):
        ref.append(PauliEvolutionGate(SparsePauliOp(lb), time=vals[gen.term0 + k]), range(gen.n))
    for q, cb in gen.meas:
        ref.measure(gen.perm[q], cb)
    return ref


t, ft = g["device"]("FakeFez")
qc = g["build"]("tfim_n12")
tpl = M.compile_template(qc, ft)
gen = tpl._gen
vals = tpl.values_of(qc)
rng = random.Random("C61K|F")
times, checks, rss = [], [], []
before = dict(M.TEMPLATE100_STATS)
changes = 0
start = time.perf_counter()
for lap in range(1, LAPS + 1):
    vals = [v + rng.gauss(0, 0.05) for v in vals]
    rb, sw = M.TEMPLATE100_STATS["rebuilt"], M.TEMPLATE100_STATS["switched"]
    s = time.perf_counter()
    out = tpl.bind_values(vals)
    times.append(time.perf_counter() - s)
    changes += M.TEMPLATE100_STATS["rebuilt"] > rb or M.TEMPLATE100_STATS["switched"] > sw
    if lap % 1000 == 0:
        ref = reader_circuit(qc, gen, vals)
        m = g["metrics"](out, ft)
        with contextlib.redirect_stdout(io.StringIO()):
            c62 = M62.compile_for_hardware(ref, target=ft, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0)
        m62 = g["metrics"](c62, ft)
        fe, fq = M._failed_elements(ft, 0.5)
        checks.append(dict(lap=lap, exact=fast_exact(ref, out), isa=m["isa"], rzz_ok=m["rzz_in_range"],
                           failed=bool(M._uses_failed(out, fe, fq)), dur=m["dur_us"], twoq=m["twoq"],
                           dur62=m62["dur_us"], twoq62=m62["twoq"]))
        rss.append((lap, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    if lap % 10000 == 0:
        print(f"W3 {VAR} lap {lap}: median {statistics.median(times[-10000:]) * 1000:.3f} ms; changes {changes}; "
              f"last check {checks[-1]['exact']:.1e}; RSS {rss[-1][1]:.0f} MB; {time.perf_counter() - start:.0f} s",
              flush=True)
ms = sorted(x * 1000 for x in times)
after = M.TEMPLATE100_STATS
json.dump(dict(variant=VAR, version=M.VERSION[:40], file_sha256=g["sha"](FILES[VAR]),
               script_sha256=g["sha"](os.path.abspath(__file__)), laps=LAPS, cache=M.TEMPLATE_CACHE,
               shape_changes=changes, rebuilt=after["rebuilt"] - before["rebuilt"],
               switched=after["switched"] - before["switched"], median_ms=ms[len(ms) // 2],
               p99_ms=ms[int(len(ms) * 0.99)], max_ms=ms[-1], peak_rss_mb=rss[-1][1] if rss else None,
               rss=rss, checks=checks, wall_s=time.perf_counter() - start), open(OUTF, "w"), indent=1)
print("W3 DONE", VAR)
