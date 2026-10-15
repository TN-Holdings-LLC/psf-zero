"""W4 / item 97 (workplace, 2026-10-15): does writing symmetric two-qubit gates in a fixed orientation remove c57's
sensitivity to the written orientation, on in-scope circuits only? Offline; no real device.

Tests: in-scope Hamiltonians written gate by gate (C57-VAL's in-scope tests are single all-qubit PauliEvolutionGates
or cx, which carry no orientation): TFIM chains (rzz, n = 12, 16, 20), a 12-ring (rzz), heavy-hex patches (rzz,
n = 16, 20; C57-VAL's edges), a heavy-hex tree Heisenberg model (rxx, ryy, rzz; n = 16), a CZ-chain ansatz (n = 12,
16) and a chain of two-qubit ZZ PauliEvolutionGates (n = 12). Every pair is written smaller index first ("as
written"); "reversed" flips every symmetric gate; "random" flips each with probability 1/2 (fixed seed per test).
Devices: C57-VAL's five, each without (T) and with (F) fractional gates. Arms (seed 0, C57-VAL's call):
  C57      prototype c57 on each of the three orientations
  C57+97   c57 + item 97 on "reversed" everywhere, and on all three on FakeFez (the rule makes them one circuit)
Only cells that item 96 puts in scope are measured. Measured: duration (ASAP on the arm's Target, measurements left
out), two-qubit count, compile time, exactness (<= 20 active qubits).
    python w4_orient.py <out dir> [part k of m]"""
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import importlib.util
import warnings

warnings.simplefilter("ignore")
REPO = os.getcwd()
OUT = sys.argv[1]
PART = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
os.makedirs(OUT, exist_ok=True)
sys.argv = ["c57_val.py", tempfile.mkdtemp() + "/x"]
spec = importlib.util.spec_from_file_location("c57v", os.path.join(REPO, "benchmarks", "c57_val.py"))
V = importlib.util.module_from_spec(spec)
sys.modules["c57v"] = V
spec.loader.exec_module(V)
M97 = V.load(os.environ.get("C57_97", os.path.join(os.path.dirname(os.path.abspath(__file__)), "c57_97", "psf_compile.py")), "psf_c57_97")
M97.WARN_WITHOUT_TARGET = False
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit.library import PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

TESTS = ("chain_rzz_n12", "chain_rzz_n16", "chain_rzz_n20", "ring_rzz_n12", "hh_rzz_n16", "hh_rzz_n20",
         "hhtree_heis_n16", "chain_cz_n12", "chain_cz_n16", "chain_pe_n12")
ORIENT = ("as_written", "reversed", "random")


def edges_of(name, n):
    if name.startswith(("chain", )):
        return [(q, q + 1) for q in range(n - 1)]
    if name.startswith("ring"):
        return [(q, q + 1) for q in range(n - 1)] + [(0, n - 1)]
    if name.startswith("hhtree"):
        return V._hh_graph(n, "tree", 5416)
    return V._hh_graph(n, "patch", 5400 + n)


def build(name):
    n = int(name.split("_n")[1])
    E = edges_of(name, n)
    rng = random.Random("W4|" + name)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.3 + 0.07 * q, q)
    if name.startswith("chain_cz"):
        for _ in range(4):
            for q in range(n):
                qc.ry(rng.uniform(-3, 3), q)
                qc.rz(rng.uniform(-3, 3), q)
            for a, b in E:
                qc.cz(a, b)
    elif name.startswith("hhtree_heis"):
        J = {e: [rng.uniform(-1, 1) for _ in range(3)] for e in E}
        for _ in range(2):
            for a, b in E:
                qc.rxx(2 * 0.3 * J[(a, b)][0], a, b)
                qc.ryy(2 * 0.3 * J[(a, b)][1], a, b)
                qc.rzz(2 * 0.3 * J[(a, b)][2], a, b)
            for q in range(n):
                qc.rz(0.3 * rng.uniform(-1, 1), q)
    else:
        J = {e: rng.uniform(-1.5, 1.5) for e in E}
        for _ in range(4):
            for a, b in E:
                if name.startswith("chain_pe"):
                    qc.append(PauliEvolutionGate(SparsePauliOp(["ZZ"], [J[(a, b)]]), time=0.25), [a, b])
                else:
                    qc.rzz(2 * 0.25 * J[(a, b)], a, b)
            for q in range(n):
                qc.rx(2 * 0.25 * 0.7, q)
    qc.measure_all()
    return qc


def orient(qc, how, name):
    if how == "as_written":
        return qc
    rng = random.Random("W4-orient|" + name)
    out = qc.copy_empty_like()
    for ins in qc.data:
        flip = len(ins.qubits) == 2 and M97._sym97(ins.operation) and (how == "reversed" or rng.random() < 0.5)
        out.append(ins.replace(qubits=ins.qubits[::-1]) if flip else ins)
    return out


def compile_with(mod, qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def main():
    rows_path = os.path.join(OUT, f"w4_rows_{PART[0]}.jsonl")
    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done.add((r["device"], r["tk"], r["test"], r["arm"], r["orient"]))
    cells = [(d, t) for d in V.DEVICES for t in TESTS]
    cells = [c for i, c in enumerate(cells) if i % PART[1] == PART[0]]
    for d, test in cells:
        T, F = V.device(d)
        base = build(test)
        for tk, tg in (("T", T), ("F", F)):
            if not V.in_scope(base, tg):
                with open(rows_path, "a") as f:
                    f.write(json.dumps(dict(device=d, tk=tk, test=test, arm="-", orient="-", in_scope=False)) + "\n")
                continue
            jobs = [("C57", o) for o in ORIENT] + [("C57+97", "reversed")]
            if d == "FakeFez":
                jobs += [("C57+97", "as_written"), ("C57+97", "random")]
            for arm, o in jobs:
                if (d, tk, test, arm, o) in done:
                    continue
                qc = orient(base, o, test)
                mod = V.M57 if arm == "C57" else M97
                t0 = time.perf_counter()
                out = compile_with(mod, qc, tg)
                secs = time.perf_counter() - t0
                m = V.metrics(out, tg)
                ex = V.exactness(base, out) if m["active"] <= 20 else None
                row = dict(device=d, tk=tk, test=test, arm=arm, orient=o, in_scope=True, secs=round(secs, 2),
                           sig=V.sig(out), exact=ex, uses_failed=bool(V.uses_failed(out, tg)), **m)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{d:18s} {tk} {test:16s} {arm:6s} {o:10s} {m['dur_us']:7.3f} us 2q {m['twoq']:4d} "
                      f"{secs:6.1f} s exact {ex}", flush=True)


if __name__ == "__main__":
    main()
