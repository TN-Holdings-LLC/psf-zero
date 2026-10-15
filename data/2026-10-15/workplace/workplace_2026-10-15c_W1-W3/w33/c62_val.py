"""C62-VAL (workplace, 2026-10-15): prototype c62 (item 101: items 87/91's CZ path emitted by PSF-Zero) against c60.
Offline; no real device.

Tests: C57-VAL's 28 tests (its 12 in-scope tests among them; the others are refused by both) and C60-VAL's 21 item-99
tests, on C57-VAL's 5 devices, each without (T) and with (F) fractional gates: 490 cells. Arms (seed 0, C57-VAL's
call): C62 (called twice) and C60. Recorded: output signature (value_sig), refusal, exactness (Aer state vector on the
active qubits, <= 20), native, rzz range, failed elements, duration (ASAP, measurements left out), two-qubit count,
compile time.

    python c62_val.py <out dir>   (from the repository root, bundle q; refuses an existing out dir)"""
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
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
_tmp = tempfile.mkdtemp() + "/c55imp"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c62_val_import")
shutil.rmtree(_tmp, ignore_errors=True)
os.makedirs(OUT)
W = os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30")
sys.path[:0] = [W, "benchmarks"]
import c60_tests as TT  # noqa: E402
from c29_identity import value_sig  # noqa: E402

C62_FILE = os.environ.get("C62_FILE", os.path.join(HERE, "c62", "psf_compile.py"))
C60_FILE = os.path.join(W, "c60", "psf_compile.py")
M62 = g["load"](C62_FILE, "psf_compile_c62")
M60 = g["load"](C60_FILE, "psf_compile_c60")
M62.WARN_WITHOUT_TARGET = M60.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit, transpile  # noqa: E402

ISING = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "tfim_n40_r2", "hhI_n16", "hhI_n20",
         "ring12_tfim")
HEIS = ("heis_n12_r2", "hhtree_heis_n16")
_SIM = []


def _probs(circ):
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
    p = (np.abs(sv) ** 2).reshape([2] * na)
    ks = sorted(meas)
    axes = [na - 1 - pos[meas[k]] for k in ks]
    p = p.sum(axis=tuple(a for a in range(na) if a not in axes))
    rem = sorted(axes)
    p = np.transpose(p, [rem.index(a) for a in axes])
    out = np.zeros(2 ** circ.num_clbits)
    flat = p.reshape(-1)
    m = len(ks)
    idx = np.arange(flat.size)
    key = np.zeros(flat.size, dtype=np.int64)
    for i in range(m):
        key |= ((idx >> (m - 1 - i)) & 1) << ks[i]
    np.add.at(out, key, flat)
    return out


def exact(qc, c):
    a = _probs(transpile(qc, basis_gates=M62._X93_BASIS, optimization_level=0))
    b = _probs(c)
    if a is None or b is None or len(a) != len(b):
        return None
    return float(abs(a - b).max())


def call(mod, qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def timed(f):
    s = time.perf_counter()
    try:
        return f(), None, time.perf_counter() - s
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}", time.perf_counter() - s


def describe(c, qc, tg):
    fe, fq = M62._failed_elements(tg, 0.5)
    m = g["metrics"](c, tg)
    return dict(sig=value_sig(c), exact=exact(qc, c) if m["active"] <= 20 else None,
                uses_failed=bool(M62._uses_failed(c, fe, fq)), **m)


json.dump(dict(c62_sha256=g["sha"](C62_FILE), c60_sha256=g["sha"](C60_FILE),
               script_sha256=g["sha"](os.path.abspath(__file__)), version=M62.VERSION[:30]),
          open(os.path.join(OUT, "c62_val_head.json"), "w"), indent=1)
tests = [("C57VAL", n, lambda n=n: g["build"](n)) for n in g["TESTS"]] + \
        [("C60VAL", n, lambda n=n: TT.build(n)) for n in TT.TESTS]
with open(os.path.join(OUT, "c62_val_rows.jsonl"), "w") as fh:
    for d in g["DEVICES"]:
        t, ft = g["device"](d)
        for src, name, b in tests:
            qc = b()
            for lab, tg in (("T", t), ("F", ft)):
                row = dict(device=d, source=src, test=name, target=lab)
                for arm, mod in (("C62", M62), ("C60", M60)):
                    before = dict(M62.CZ101_STATS) if arm == "C62" else None
                    c, err, secs = timed(lambda: call(mod, qc, tg))
                    r = dict(error=err, secs=round(secs, 4))
                    if c is not None:
                        r.update(describe(c, qc, tg))
                        if arm == "C62":
                            c2, _, _ = timed(lambda: call(mod, qc, tg))
                            r["sig_repeat"] = None if c2 is None else value_sig(c2)
                            r["own_emit"] = M62.CZ101_STATS["emitted"] > before["emitted"]
                    row[arm] = r
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                a, z = row["C62"], row["C60"]
                st = ("both refuse" if a["error"] and z["error"] else "identical" if a.get("sig") == z.get("sig")
                      else f"{a.get('dur_us', 0) / max(z.get('dur_us', 1e-9), 1e-9):.3f} x")
                print(f"{d[-12:]:12s} {name:16s} {lab} {st} exact {a.get('exact')}", flush=True)
print("C62-VAL DONE")
