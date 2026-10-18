"""C68-VAL: prototype c68 (item 105: chains and rings by psfz r4) against its pre-registration
(C68-VAL_preregistration.md next to this file).

    PSFZ_BIN=<psfz> python data/2026-10-18/c68_val/c68_val.py <out.jsonl> <device> [<device> ...]   (PSF-Zero root)

Per device and test (C63-VAL's 19 and PSFZ-VAL's 6 new, 25): the build input, then 5 same-sign and 5 free draws by
C63-VAL's `randomize` with random.Random(f"C68-VAL|{device}|{test}"). Per input: c68's default call twice (compared
bit for bit) and whether it used psfz (PSFZ105_STATS); psfz's program run directly on the same input (as PSFZ-VAL2);
release 2026-10-17.1's default call; c68's default call with no psfz program ($PSFZ_BIN pointing nowhere); checks
(native, failed elements, exactness on outputs psfz made: up to 16 active qubits every input, 17-20 the build input
and the first two draws), durations (as soon as possible with the device's durations, measurements included), compile
times. One JSON line per input; never overwrites.
"""
import contextlib
import hashlib
import io
import json
import os
import random
import runpy
import subprocess
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
OUT, DEVICES = sys.argv[1], sys.argv[2:]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
BIN = os.environ.get("PSFZ_BIN")
if not BIN or not os.path.isfile(BIN):
    raise SystemExit("STOP: set PSFZ_BIN to the built psfz program")
_tmp = tempfile.mkdtemp()
_argv = sys.argv
sys.argv = ["psfz_val", os.path.join(_tmp, "none.jsonl")]
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(os.path.join("data", "2026-10-17", "psfz_val", "psfz_val.py"), run_name="c68_import")
sys.argv = _argv
REL, T, g = P["REL"], P["T"], P["g"]
from qiskit import qasm3, transpile  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

C68 = os.path.join("patches", "psf_compile_c68_2026-10-18", "psf_compile.py")
with contextlib.redirect_stdout(io.StringIO()):
    M = g["load"](C68, "psf_compile_c68")
M.WARN_WITHOUT_TARGET = False
FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
NEW = P["NEW"]
SAME, FREE = 5, 5
SMOKE = os.environ.get("C68_SMOKE")  # a wiring check before the run: other draws, three tests; not scored
TESTS = ("tfim_n12", "hhI_n16", "cx_chain_n20") if SMOKE else FORMULA + NATIVE + NEW


def build(name):
    return P["build_new"](name) if name in NEW else (g["build"](name) if name in FORMULA else P["TT"].build(name))


def fp(c):  # C67-VAL's: every operation, angles as a float's hex, the global phase, the layouts
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


def call(f, qc, t):
    s = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = f(qc, target=t)
        return out, time.perf_counter() - s, None
    except Exception as e:  # noqa: BLE001 - recorded: a refusal or an error
        return None, time.perf_counter() - s, f"{type(e).__name__}: {str(e)[:200]}"


work = tempfile.mkdtemp()
with open(OUT, "w", encoding="utf-8") as fh:
    for d in DEVICES:
        t = getattr(fake_provider, d)().target
        fe, fq = REL._failed_elements(t, 0.5)
        dj = os.path.join(work, "d.json")
        json.dump(P["device_json"](t, d), open(dj, "w"))
        for name in TESTS:
            qc0 = build(name)
            rng = random.Random(f"{'C68-SMOKE' if SMOKE else 'C68-VAL'}|{d}|{name}")
            for k in range(0 if SMOKE else -1, 2 if SMOKE else SAME + FREE):
                qc = qc0 if k < 0 else P["randomize"](qc0, rng, same=k < SAME)
                row = dict(device=d, test=name, k=k)
                # c68, twice
                u0 = M.PSFZ105_STATS["used"]
                out, row["c68_s"], err = call(M.compile_for_hardware, qc, t)
                row["c68_used_psfz"] = M.PSFZ105_STATS["used"] > u0
                out2, _, err2 = call(M.compile_for_hardware, qc, t)
                row["c68_refused"] = err
                row["deterministic"] = (err == err2) if err else (err2 is None and fp(out2) == fp(out))
                # psfz directly
                src, rep_f = os.path.join(work, "in.qasm"), os.path.join(work, "rep.json")
                if os.path.exists(rep_f):
                    os.remove(rep_f)
                open(src, "w").write(qasm3.dumps(transpile(qc, basis_gates=P["BASIS"], optimization_level=0)))
                subprocess.run([BIN, "compile", dj, src, os.path.join(work, "o.qasm"), rep_f], capture_output=True)
                rep = json.load(open(rep_f)) if os.path.exists(rep_f) else dict(ok=False, reason="no report")
                row["psfz_ok"] = bool(rep.get("ok"))
                if rep.get("ok") and out is not None:
                    own = P["from_report"](rep, t.num_qubits)
                    row["same_as_psfz"] = P["fp"](own) == P["fp"](out) and \
                        list(out.layout.initial_index_layout()[:qc.num_qubits]) == [int(x) for x in rep["layout"]]
                # the release
                rel, row["rel_s"], rerr = call(REL.compile_for_hardware, qc, t)
                row["rel_refused"] = rerr
                if out is None or row["c68_used_psfz"]:
                    row["same_as_release"] = None
                else:
                    row["same_as_release"] = rel is not None and fp(rel) == fp(out)
                if out is None:
                    row["same_refusal"] = rerr is not None and rerr == err
                # c68 with no psfz program
                keep = os.environ["PSFZ_BIN"]
                os.environ["PSFZ_BIN"] = os.path.join(work, "no_such_program")
                o3, _, err3 = call(M.compile_for_hardware, qc, t)
                os.environ["PSFZ_BIN"] = keep
                row["no_program_same"] = (err3 is not None and err3 == rerr) if rel is None else \
                    (o3 is not None and fp(o3) == fp(rel))
                if out is not None:
                    row["c68"] = P["numbers"](out, t)
                    row["c68"]["failed"] = bool(REL._uses_failed(out, fe, fq))
                    act = row["c68"]["active"]
                    if row["c68_used_psfz"] and (act <= 16 or (act <= 20 and k < 2)):
                        ref = qc.decompose() if any(i.operation.name == "PauliEvolution" for i in qc.data) else qc
                        row["c68"]["tvd"] = T._tvd(T._dist(ref), T._dist(out))
                    else:
                        row["c68"]["tvd"] = None
                if rel is not None:
                    row["rel"] = P["numbers"](rel, t)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(d, name, "done", flush=True)
print("C68-VAL DONE", M.VERSION, REL.VERSION, json.dumps(M.PSFZ105_STATS))
