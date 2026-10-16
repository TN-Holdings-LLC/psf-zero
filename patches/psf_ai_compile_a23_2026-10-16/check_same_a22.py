"""Checks that a23 returns the same circuit as a22, bit for bit (item 45).

    TOQB=<TOQB root> python patches/psf_ai_compile_a23_2026-10-16/check_same_a22.py

Run from the PSF-Zero root. psf_compile is c63a (the workplace's 2026-10-15c delivery). Compared per call: every
operation's name, qubits, clbits and angles (a float's hex), the global phase, the initial and the final layout; a
refusal compares by its exception type. Cases: TOQB's ring model circuits on FakeFez, FakeTorino, FakeMarrakesh and
FakeKingston, n = 4-20, L = 1, 2, 4, 8 (272), each through `compile_for_model_circuit` and `model_template(...).bind`,
plus refusals without a Target, plus 40 handed-on inputs (GHZ 5 and 12, two random circuits, a 30-qubit ring).
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import fake_provider

warnings.simplefilter("ignore")
C63A = "data/2026-10-15/workplace/workplace_2026-10-15c_W1-W3/w34/w3/c63a/psf_compile.py"
A22 = "patches/psf_ai_compile_a22_2026-10-15/psf_ai_compile.py"
A23 = "patches/psf_ai_compile_a23_2026-10-16/psf_ai_compile.py"
DEVICES = ("FakeFez", "FakeTorino", "FakeMarrakesh", "FakeKingston")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def layout_of(c, final=False):
    lay = getattr(c, "_layout", None)
    if lay is None:
        return None
    if final:
        return None if lay.final_layout is None else tuple(lay.final_index_layout())
    return tuple(lay.initial_index_layout(filter_ancillas=False))


def bits(c):
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    ops = tuple((ci.operation.name, tuple(q[x] for x in ci.qubits), tuple(k[x] for x in ci.clbits),
                 tuple(str(p) if hasattr(p, "parameters") else float(p).hex() for p in ci.operation.params))
                for ci in c.data)
    gp = c.global_phase
    return (ops, str(gp) if hasattr(gp, "parameters") else float(gp).hex(), layout_of(c), layout_of(c, True),
            c.num_qubits, c.num_clbits)


def run(mod, kind, qc, t):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if kind == "recompile":
                basis = [g for g in ("cz", "rz", "sx", "x") if g in t.operation_names]
                return mod.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)
            if kind == "template":
                tp = mod.model_template(qc, t)
                return None if tp is None else tp.bind(qc)
            return mod.compile_for_model_circuit(qc, None, ["cz", "rz", "sx", "x"], target=None)
    except Exception as e:  # noqa: BLE001 - a refusal compares by its type
        return ("error", type(e).__name__)


def same(o2, o3):
    if isinstance(o2, tuple) or isinstance(o3, tuple) or o2 is None or o3 is None:
        return o2 == o3
    return bits(o2) == bits(o3)


def ghz(n):
    c = QuantumCircuit(n, n)
    c.h(0)
    for j in range(n - 1):
        c.cx(j, j + 1)
    c.measure(range(n), range(n))
    return c


def main():
    sys.path.insert(0, os.environ["TOQB"])
    from toqb.task_loop import angle_list, model_circuit, n_params
    with contextlib.redirect_stdout(io.StringIO()):
        pc = load(C63A, "psf_compile")
        pc.WARN_WITHOUT_TARGET = False
        a22, a23 = load(A22, "ai22"), load(A23, "ai23")
    rng = np.random.default_rng(7)
    counts = {"ring": [0, 0], "handoff": [0, 0]}
    diffs = []
    for dev in DEVICES:
        t = getattr(fake_provider, dev)().target
        for n in range(4, 21):
            for L in (1, 2, 4, 8):
                qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L,
                                              np.pi / 2), n, L)
                for kind in ("recompile", "template") + (("notarget",) if n == 4 and L == 1 else ()):
                    ok = same(run(a22, kind, qc, t), run(a23, kind, qc, t))
                    counts["ring"][not ok] += 1
                    if not ok:
                        diffs.append((dev, n, L, kind))
        hand = [ghz(5), ghz(12), random_circuit(5, 6, max_operands=2, measure=True, seed=1),
                random_circuit(8, 4, max_operands=2, measure=True, seed=2),
                model_circuit(angle_list(np.full(n_params(30, 2), 0.3), np.zeros(30), 30, 2, np.pi / 2), 30, 2)]
        for j, qc in enumerate(hand):
            for kind in ("recompile", "template"):
                ok = same(run(a22, kind, qc, t), run(a23, kind, qc, t))
                counts["handoff"][not ok] += 1
                if not ok:
                    diffs.append((dev, "handoff", j, kind))
    for k, (s, d) in counts.items():
        print(f"{k}: identical {s}, different {d}")
    print("differences:", diffs)
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
