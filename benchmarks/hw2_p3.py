"""hw2_p3.py -- HW2-PLUS's P3 (a14's estimate with exact reduced states, Addendum 466), computed after the run.

hw2_predictors.py computes P3 with est_lc.states_aer (Aer density-matrix saves). On HW2's largest circuit (ba_n20 PSF,
23 active qubits) Aer 0.17.2 ends with a segmentation fault inside libstdc++, every time, also single-threaded. The
reduced states are exact either way, so this script computes them with a plain numpy statevector instead
(states_np) and uses a14's own estimate unchanged (est_lc.estimate). It first checks on the circuits where Aer works
that both give the same reduced states (largest absolute difference reported), then scores P3 with
hw2_predictors.score().

    python benchmarks/hw2_p3.py <out dir> [hw2 dir] [--resume]
--resume keeps the rows already in <out dir>/hw2_p3.jsonl and computes only the missing circuits (the computing
container restarted during the first run)."""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, "."]
A14_DIR = os.path.join("patches", "psf_ai_compile_a14_2026-10-13")


def states_np(circ):
    """The same output as est_lc.states_aer: {instruction index: reduced density matrix of its qubits after the
    instruction, qs[0] most significant}, from an exact numpy statevector of the used qubits."""
    from qiskit.quantum_info import Operator
    import est_lc as est
    ops = est._ops(circ)
    used = sorted({q for op, qs in ops if op.name not in ("barrier",) for q in qs})
    ax = {q: i for i, q in enumerate(used)}  # tensor axis of each used qubit
    m = len(used)
    psi = np.zeros([2] * m, dtype=complex)
    psi[(0,) * m] = 1.0
    ev = set(est._events(ops))
    out = {}
    for i, (op, qs) in enumerate(ops):
        if op.name in est.SKIP:
            continue
        u = Operator(op).data
        k = len(qs)
        # Qiskit: the first listed qubit is the least significant index of the matrix
        ut = u.reshape([2] * (2 * k))  # (out_{k-1} .. out_0, in_{k-1} .. in_0)
        axes = [ax[q] for q in reversed(qs)]  # matches in_{k-1} .. in_0
        psi = np.tensordot(ut, psi, axes=(list(range(k, 2 * k)), axes))
        psi = np.moveaxis(psi, list(range(k)), axes)
        if i in ev:
            a = np.moveaxis(psi, [ax[q] for q in qs], list(range(k))).reshape(2 ** k, -1)
            out[i] = a @ a.conj().T
    return out


def _compare(qpy_path, index):
    """Run in a subprocess: largest |Aer - numpy| over every reduced state of one circuit."""
    from qiskit import qpy
    sys.path.insert(0, A14_DIR)
    import est_lc as est
    c = qpy.load(open(qpy_path, "rb"))[index]
    a, b = est.states_aer(c), states_np(c)
    print(json.dumps(dict(n=len(a), maxdiff=max(float(np.max(np.abs(a[k] - b[k]))) for k in a))))


def main(out, w, resume=False):
    from qiskit import qpy
    sys.path.insert(0, A14_DIR)
    import est_lc as est
    import hw2_predictors as HP
    import snapshot_target as ST
    import core_fix_c2_eval as CF
    CF.load_module(os.path.join("benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    CF.load_module("psf_compile.py", "psf_compile")
    a14 = CF.load_module(os.path.join(A14_DIR, "psf_ai_compile.py"), "psf_ai_compile_a14")
    os.makedirs(out, exist_ok=resume)
    man = json.load(open(os.path.join(w, "hw2_manifest.json")))
    sc = json.load(open(os.path.join(w, "hw2_score.json")))
    qpath = os.path.join(w, "hw2_circuits.qpy")
    circs = qpy.load(open(qpath, "rb"))
    tgt = ST.target_from_snapshot(os.path.join(w, f"ns_snapshot_{man['device']}.json"))
    todo = [(k, man["entries"][p["entry"]], c) for k, (p, c) in enumerate(zip(man["pubs"], circs))
            if p["repeat"] == 1 and man["entries"][p["entry"]]["submitted"]
            and (man["entries"][p["entry"]]["part"] == "S" or man["entries"][p["entry"]]["test"] in HP.READING)]
    todo.sort(key=lambda x: x[1]["active"])
    rows, checks = [], []
    jl = os.path.join(out, "hw2_p3.jsonl")
    if resume and os.path.exists(jl):
        rows = [json.loads(x) for x in open(jl)]
        checks = [float(r["engine"].split("max diff ")[1].rstrip(")")) for r in rows if "max diff" in r["engine"]]
    have = {(r["part"], r["test"], r["arm"]) for r in rows}
    f = open(jl, "a" if resume else "w")
    for k, e, c in todo:
        if (e["part"], e["test"], e["arm"]) in have:
            continue
        t0 = time.time()
        # both engines on every circuit Aer survives (a crash only ends the subprocess)
        if e["active"] >= 23:  # Aer crashed here in the first run, and its retry took the computing container down
            pr = None
            engine = "numpy (Aer not run: it crashes on this circuit)"
        else:
            pr = subprocess.run([sys.executable, os.path.abspath(__file__), "--compare", qpath, str(k)],
                                capture_output=True, text=True)
        if pr is None:
            pass
        elif pr.returncode == 0:
            chk = json.loads(pr.stdout.strip().splitlines()[-1])
            checks.append(chk["maxdiff"])
            engine = "numpy (Aer agrees, max diff %.1e)" % chk["maxdiff"]
        else:
            engine = f"numpy (Aer ended with code {pr.returncode})"
        g, idle = est.estimate(c, tgt, states_np(c), a14)[:2]
        r = dict(part=e["part"], test=e["test"], arm=e["arm"], active=e["active"],
                 f_real=sc["F"][f"{e['part']}|{e['test']}|{e['arm']}"]["f"], P3=(math.exp(-(g + idle)), None),
                 engine=engine, secs=round(time.time() - t0, 1))
        rows.append(r)
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(json.dumps(r), flush=True)
    f.close()
    print(f"largest Aer / numpy difference over {len(checks)} circuits: {max(checks):.2e}", flush=True)
    HP.score(out, rows, ("P3",))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        _compare(sys.argv[2], int(sys.argv[3]))
    else:
        args = [a for a in sys.argv[1:] if a != "--resume"]
        main(args[0], args[1] if len(args) > 1 else os.path.join("data", "2026-10-13", "hw2"), "--resume" in sys.argv)
