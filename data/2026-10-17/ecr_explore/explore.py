"""Exploratory (not pre-registered): prototype c67's item 104 on IBM Eagle fake devices (ECR), against Qiskit level 3.

    python data/2026-10-17/ecr_explore/explore.py <out.json>   (from the PSF-Zero root)

C63-VAL's 19 tests (its build inputs, as c63_val.py builds them) on FakeSherbrooke, FakeBrisbane, FakeKyiv and
FakeOsaka. Per test and device: c67's default call (or its refusal), checked native and exact (outcome distribution,
up to 20 active qubits), with its two-qubit count, duration (scheduled as soon as possible with the device's durations,
measurements included) and compile time; Qiskit level 3 (seed 0) on the same input, the same numbers.
"""
import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
_tmp = tempfile.mkdtemp() + "/imp"
sys.argv = ["c55_val", _tmp]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="ecr_import")
shutil.rmtree(_tmp, ignore_errors=True)
sys.path[:0] = [os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "benchmarks"]
import c60_tests as TT  # noqa: E402
import test_release_2026_10_16_2 as T  # noqa: E402  (its exact-distribution and native checks)
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    M = g["load"](os.path.join("patches", "psf_compile_c67_2026-10-17", "psf_compile.py"), "psf_compile_c67")
M.WARN_WITHOUT_TARGET = False
FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
DEVICES = ("FakeSherbrooke", "FakeBrisbane", "FakeKyiv", "FakeOsaka")


def build(name):
    return g["build"](name) if name in FORMULA else TT.build(name)


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
    return dict(q2=n2, dur_us=1e6 * max(clock.values()) if clock else 0.0, active=len(clock), native=T._native(c, t))


def exact(qc, c, active):
    if active > 20:
        return None
    ref = qc.decompose() if any(i.operation.name == "PauliEvolution" for i in qc.data) else qc
    return T._tvd(T._dist(ref), T._dist(c))


rows = []
for d in DEVICES:
    t = getattr(fake_provider, d)().target
    pm = generate_preset_pass_manager(3, target=t, seed_transpiler=0)
    for name in FORMULA + NATIVE:
        qc = build(name)
        row = dict(device=d, test=name)
        s = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = M.compile_for_hardware(qc, target=t)
            row["psf_s"] = time.perf_counter() - s
            row["psf"] = numbers(out, t)
            row["psf"]["tvd"] = exact(qc, out, row["psf"]["active"])
        except Exception as e:  # noqa: BLE001 - recorded
            row["psf_s"] = time.perf_counter() - s
            row["psf"] = dict(refused=f"{type(e).__name__}: {str(e)[:120]}")
        s = time.perf_counter()
        qk = pm.run(qc)
        row["qk3_s"] = time.perf_counter() - s
        row["qk3"] = numbers(qk, t)
        rows.append(row)
        p = row["psf"]
        print(d, name, "PSF", p.get("refused") or f"q2 {p['q2']} dur {p['dur_us']:.2f} native {p['native']} tvd {p['tvd']}",
              "| QK3", f"q2 {row['qk3']['q2']} dur {row['qk3']['dur_us']:.2f}", flush=True)
json.dump(dict(version=M.VERSION, rows=rows, stats=M.ECR104_STATS), open(OUT, "w"), indent=1)
print("done", M.ECR104_STATS)
