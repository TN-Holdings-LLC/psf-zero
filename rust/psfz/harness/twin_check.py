"""Check psfz's MPS twin (prototype, 2026-10-19, noiseless): exactness against a state vector, then size and speed.

    python rust/psfz/harness/twin_check.py OUT_DIR        (from the repository root; psfz built with cargo)

TFIM chains (rzz with random angles, rx and u fields), compiled by psfz r4 on FakeFez and simulated by `psfz twin`:
1. n = 8, 12, 16, 20, steps 2-4: <Z> of every measured bit against Qiskit's state vector of the input circuit
   (largest difference), with the formula bound and the bond dimension reached.
2. n = 50 and 80 (FakeFez's longest working chain for psfz is 80-89 qubits), steps 2-6: time, the formula bound and the bond dimension reached (no state vector exists).
Writes OUT_DIR/twin_check.json.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.getcwd()
PSFZ = os.environ.get("PSFZ_BIN", os.path.join(REPO, "rust", "psfz", "target", "release", "psfz"))


def load_pc():
    sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
    spec = importlib.util.spec_from_file_location("psf_compile", os.path.join(REPO, "psf_compile.py"))
    pc = importlib.util.module_from_spec(spec)
    sys.modules["psf_compile"] = pc
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(pc)
    return pc


def tfim(n, steps, rng):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for _ in range(steps):
        th = rng.uniform(-1.5, 1.5)
        for q in range(0, n - 1, 2):
            qc.rzz(th, q, q + 1)
        for q in range(1, n - 1, 2):
            qc.rzz(th, q, q + 1)
        for q in range(n):
            qc.rx(rng.uniform(-1.5, 1.5), q)
    for q in range(n):
        qc.u(*rng.uniform(-np.pi, np.pi, 3), q)
    qc.measure_all()
    return qc


def z_exact(qc):
    from qiskit.quantum_info import Statevector
    p = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
    idx = np.arange(len(p))
    return np.array([np.sum(p * (1 - 2 * ((idx >> k) & 1))) for k in range(qc.num_qubits)])


def main(out_dir):
    from qiskit import qasm3
    from qiskit_ibm_runtime import fake_provider as fp
    pc = load_pc()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20261019)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        dfile = os.path.join(tmp, "FakeFez.json")
        open(dfile, "w").write(pc._psfz105_device(fp.FakeFez().target))
        plan = [(n, s, True) for n in (8, 12, 16, 20) for s in (2, 3, 4)] + \
               [(n, s, False) for n in (50, 80) for s in (2, 4, 6)]
        for n, steps, exact in plan:
            qc = tfim(n, steps, rng)
            f = os.path.join(tmp, "in.qasm")
            open(f, "w").write(qasm3.dumps(qc))
            rep = os.path.join(tmp, "rep.json")
            t0 = time.perf_counter()
            r = subprocess.run([PSFZ, "twin", dfile, f, rep, "65536"], capture_output=True, text=True)
            wall = time.perf_counter() - t0
            j = json.load(open(rep)) if os.path.exists(rep) else {}
            row = dict(n=n, steps=steps, rc=r.returncode, ok=j.get("ok"), reason=j.get("reason"), wall_s=wall,
                       simulate_s=j.get("simulate_s"), compile_s=j.get("compile_s"), max_chi=j.get("max_chi"),
                       bound_max_log2=max(j.get("bound_log2") or [0]), dropped=j.get("dropped_weight"))
            if exact and j.get("ok"):
                row["max_z_diff"] = float(np.max(np.abs(np.array(j["z_clbits"]) - z_exact(qc))))
            rows.append(row)
            print(f"n {n:3d} steps {steps}: ok {row['ok']} chi {row['max_chi']} (bound 2^{row['bound_max_log2']}) "
                  f"simulate {row['simulate_s'] or 0:.3f} s dropped {row['dropped'] or 0:.1e}"
                  + (f" |dZ| {row['max_z_diff']:.1e}" if 'max_z_diff' in row else "")
                  + (f" refused: {row['reason']}" if not row['ok'] else ""), flush=True)
            if os.path.exists(rep):
                os.remove(rep)
    json.dump(rows, open(os.path.join(out_dir, "twin_check.json"), "w"), indent=1)
    ok = all(r.get("max_z_diff", 0) < 1e-9 for r in rows if r["ok"])
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
