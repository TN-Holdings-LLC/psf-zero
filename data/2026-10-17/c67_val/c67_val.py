"""C67-VAL: prototype c67's default call on ECR devices (IBM Eagle fakes), against its pre-registration
(C67-VAL_preregistration.md next to this file).

    python data/2026-10-17/c67_val/c67_val.py <out.jsonl> <device> [<device> ...]   (from the PSF-Zero root)

Per device and C63-VAL test: the build input (as c63_val.py builds it), then 25 same-sign and 25 free draws by
c63_val.py's `randomize` with random.Random(f"C67|{device}|{test}") (the first 25 draws same-sign). Per input: c67's
`compile_for_hardware(qc, target=)` twice (compared bit for bit), the output's checks (native; no operation on a
failed element, item 31's test at 0.5; exact up to 20 active qubits), its two-qubit count and duration (as soon as
possible with the device's durations, measurements included), its compile time; Qiskit level 3 (seed 0) on the same
input, the same numbers. One JSON line per input; never overwrites.
"""
import contextlib
import hashlib
import io
import json
import math
import os
import random
import runpy
import shutil
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
OUT, DEVICES = sys.argv[1], sys.argv[2:]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
_tmp = tempfile.mkdtemp() + "/imp"
sys.argv = ["c55_val", _tmp]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c67_import")
shutil.rmtree(_tmp, ignore_errors=True)
sys.path[:0] = [os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "benchmarks"]
import c60_tests as TT  # noqa: E402
import test_release_2026_10_16_2 as T  # noqa: E402  (its exact-distribution and native checks)
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

C67 = os.path.join("patches", "psf_compile_c67_2026-10-17", "psf_compile.py")
with contextlib.redirect_stdout(io.StringIO()):
    M = g["load"](C67, "psf_compile_c67")
M.WARN_WITHOUT_TARGET = False
FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
SAME, FREE = 25, 25


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


def fp(c):  # default_call.py's: every operation, angles as a float's hex, the global phase, the layouts
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    ops = tuple((ci.operation.name, tuple(q[x] for x in ci.qubits), tuple(k[x] for x in ci.clbits),
                 tuple(str(p) if hasattr(p, "parameters") else float(p).hex() for p in ci.operation.params))
                for ci in c.data)
    lay = getattr(c, "_layout", None)
    il = None if lay is None else tuple(lay.initial_index_layout(filter_ancillas=False))
    fl = None if lay is None or lay.final_layout is None else tuple(lay.final_index_layout())
    return hashlib.sha256(repr((ops, float(c.global_phase).hex(), il, fl, c.num_qubits, c.num_clbits)).encode()) \
        .hexdigest()


def numbers(c, t):
    clock, n2 = {}, 0
    pos = {q: i for i, q in enumerate(c.qubits)}
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        p = t[nm].get(qs) if nm in t.operation_names else None
        d = (p.duration or 0.0) if p is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    return dict(q2=n2, dur_us=1e6 * max(clock.values()) if clock else 0.0, active=len(clock))


def compile_psf(qc, t):
    s = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = M.compile_for_hardware(qc, target=t)
        return out, time.perf_counter() - s, None
    except Exception as e:  # noqa: BLE001 - recorded: a refusal or an error
        return None, time.perf_counter() - s, f"{type(e).__name__}: {str(e)[:160]}"


with open(OUT, "w", encoding="utf-8") as fh:
    for d in DEVICES:
        t = getattr(fake_provider, d)().target
        fe, fq = M._failed_elements(t, 0.5)
        pm = generate_preset_pass_manager(3, target=t, seed_transpiler=0)
        for name in FORMULA + NATIVE:
            qc0 = build(name)
            rng = random.Random(f"C67|{d}|{name}")
            for k in range(-1, SAME + FREE):
                qc = qc0 if k < 0 else randomize(qc0, rng, same=k < SAME)
                row = dict(device=d, test=name, k=k, draw="build" if k < 0 else ("same" if k < SAME else "free"))
                out, ts, err = compile_psf(qc, t)
                row["psf_s"] = ts
                if err:
                    row["refused"] = err
                else:
                    out2, _, err2 = compile_psf(qc, t)
                    row["deterministic"] = err2 is None and fp(out2) == fp(out)
                    row.update(numbers(out, t))
                    row["native"] = T._native(out, t)
                    row["failed"] = bool(M._uses_failed(out, fe, fq))
                    if row["active"] <= 20:
                        ref = qc.decompose() if any(i.operation.name == "PauliEvolution" for i in qc.data) else qc
                        row["tvd"] = T._tvd(T._dist(ref), T._dist(out))
                    else:
                        row["tvd"] = None
                s = time.perf_counter()
                qk = pm.run(qc)
                row["qk3_s"] = time.perf_counter() - s
                row["qk3"] = numbers(qk, t)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(d, name, "done", flush=True)
print("C67-VAL DONE", M.VERSION, json.dumps(M.ECR104_STATS))
