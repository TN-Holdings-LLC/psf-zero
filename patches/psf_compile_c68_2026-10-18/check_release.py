"""Release 2026-10-18.1's check: the installed psf_compile.py against prototype c68 (as validated in C68-VAL), bit for
bit, on C68-VAL's 4,400 inputs (16 devices, 25 tests, the build input and 10 draws each, the same draws), with the
psfz program ($PSFZ_BIN) and without it.

    PSFZ_BIN=<psfz> python patches/psf_compile_c68_2026-10-18/check_release.py <out.txt>   (from the PSF-Zero root)
"""
import contextlib
import io
import os
import random
import runpy
import sys
import tempfile
import warnings

warnings.simplefilter("ignore")
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
BIN = os.environ.get("PSFZ_BIN")
if not BIN or not os.path.isfile(BIN):
    raise SystemExit("STOP: set PSFZ_BIN to the built psfz program")
_tmp = tempfile.mkdtemp()
sys.argv = ["psfz_val", os.path.join(_tmp, "none.jsonl")]
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(os.path.join("data", "2026-10-17", "psfz_val", "psfz_val.py"), run_name="chk_import")
g = P["g"]
with contextlib.redirect_stdout(io.StringIO()):
    NEWREL = g["load"]("psf_compile.py", "psf_compile_new")
    C68 = g["load"](os.path.join("patches", "psf_compile_c68_2026-10-18", "psf_compile.py"), "psf_compile_c68")
from qiskit_ibm_runtime import fake_provider  # noqa: E402

FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
NEW = P["NEW"]
DEVICES = ("FakeAachen", "FakeBoston", "FakeKingston", "FakePittsburgh", "FakeFez", "FakeMarrakesh", "FakeTorino",
           "FakeCusco", "FakeBrussels", "FakeStrasbourg", "FakeKyiv", "FakeOsaka", "FakeKawasaki", "FakeQuebec",
           "FakeSherbrooke", "FakeBrisbane")


def build(name):
    return P["build_new"](name) if name in NEW else (g["build"](name) if name in FORMULA else P["TT"].build(name))


def fp(c):
    import hashlib
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    ops = tuple((ci.operation.name, tuple(q[x] for x in ci.qubits), tuple(k[x] for x in ci.clbits),
                 tuple(str(p) if hasattr(p, "parameters") else float(p).hex() for p in ci.operation.params))
                for ci in c.data)
    lay = getattr(c, "_layout", None)
    il = None if lay is None else tuple(lay.initial_index_layout(filter_ancillas=False))
    return hashlib.sha256(repr((ops, float(c.global_phase).hex(), il, c.num_qubits, c.num_clbits)).encode()).hexdigest()


def res(mod, qc, t):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fp(mod.compile_for_hardware(qc, target=t))
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


n = same = same_np = used = 0
for d in DEVICES:
    t = getattr(fake_provider, d)().target
    for name in FORMULA + NATIVE + NEW:
        qc0 = build(name)
        rng = random.Random(f"C68-VAL|{d}|{name}")
        for k in range(-1, 10):
            qc = qc0 if k < 0 else P["randomize"](qc0, rng, same=k < 5)
            u = NEWREL.PSFZ105_STATS["used"]
            a, b = res(NEWREL, qc, t), res(C68, qc, t)
            used += NEWREL.PSFZ105_STATS["used"] > u
            os.environ["PSFZ_BIN"] = os.path.join(_tmp, "no_program")
            a2, b2 = res(NEWREL, qc, t), res(C68, qc, t)
            os.environ["PSFZ_BIN"] = BIN
            n += 1
            same += a == b
            same_np += a2 == b2
    print(d, n, same, same_np, flush=True)
line = (f"release {NEWREL.VERSION} against prototype {C68.VERSION}: {n} inputs; the same output (or refusal) with "
        f"the psfz program {same} of {n} (psfz used on {used}); without it {same_np} of {n}")
print(line)
open(OUT, "w", encoding="utf-8").write(line + "\n")
