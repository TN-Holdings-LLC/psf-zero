"""diag_c29.py -- diagnostic (2026-10-10; nothing predicted): test_whole_compile_same_where_c26_repeats_itself failed
on a layout. Repeats its compiles in one process in the order c26, c26, c29, c26, c29 and reports, for each output
against the first: whole equality, equality of instructions, of the logical initial/final layouts (ancillas
filtered, as C29-ID compares), and of the ancilla assignment; and what each compile decided (counter changes).
Run from the psf-zero repository root with the c29 files in place:  python <this file> > OUT"""
import contextlib
import json
import os
import sys

REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
import core_fix_c2_eval as H  # noqa: E402
import c25_identity2 as C2  # noqa: E402
from qiskit.circuit.random import random_circuit  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeTorino  # noqa: E402

lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
mods = {"c26": H.load_module(os.path.join(REPO, "patches", "psf_compile_c26_2026-10-09", "psf_compile.py"), "pc26"),
        "c29": H.load_module(os.path.join(REPO, "patches", "psf_compile_c29_2026-10-10", "psf_compile.py"), "pc29")}
backend = FakeTorino()
basis = [g for g in backend.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
kw = dict(coupling_map=backend.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
          seed_transpiler=0, target=backend.target, **C2.RECOMMENDED)
STATS = ("COMPARE_STATS", "EXACT_STATS", "RESYNTH_STATS", "SKIP_STATS", "FEASIBILITY_STATS")


def snap(m):
    return {s: dict(getattr(m, s)) for s in STATS if hasattr(m, s)}


def delta(a, b):
    return {s: {k: b[s][k] - a[s].get(k, 0) for k in b[s] if b[s][k] - a[s].get(k, 0)} for s in b
            if any(b[s][k] - a[s].get(k, 0) for k in b[s])}


def parts(c):
    lo = c.layout
    ins = [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [repr(float(p)) for p in i.operation.params])
           for i in c.data]
    logical = (list(lo.initial_index_layout(filter_ancillas=True)), list(lo.final_index_layout(filter_ancillas=True)))
    where = {q: (r.name, i) for r in lo.initial_layout.get_registers() for i, q in enumerate(r)}
    anc = sorted((p,) + where.get(q, ("?", -1)) for p, q in lo.initial_layout.get_physical_bits().items())
    return ins, logical, anc


for k in range(5):
    qc = random_circuit(3 + k, 4 + k, max_operands=2, measure=False, seed=56_100 + k)
    for n in (12, 10 ** 9):
        for m in mods.values():
            m.EXACT_MAX_OPS = n
        outs = []
        for name in ("c26", "c26", "c29", "c26", "c29"):
            m = mods[name]
            lay.time = C2._VirtualTime()
            s0 = snap(m)
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                o = m.compile_for_hardware(qc, **kw)
            outs.append((name, o, delta(s0, snap(m))))
        ref = parts(outs[0][1])
        print(f"k={k} n={n} qubits={qc.num_qubits}")
        for j, (name, o, d) in enumerate(outs):
            p = parts(o)
            print(f"  {j} {name}: whole {o == outs[0][1] and o.layout == outs[0][1].layout}, instructions "
                  f"{p[0] == ref[0]}, logical layouts {p[1] == ref[1]}, ancillas {p[2] == ref[2]}; init "
                  f"{p[1][0]} | {json.dumps(d)}")
        sys.stdout.flush()
