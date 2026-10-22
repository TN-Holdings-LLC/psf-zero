"""Checks a24 against a23 (items 46-48), from the PSF-Zero root:

    TOQB=<TOQB root> python patches/psf_ai_compile_a24_2026-10-22/check_a24.py

psf_compile is the release (2026-10-18.1, ./psf_compile.py) for a23 and a24 alike; rust/psfbind built (ring r2).
PSF_RUST_TEMPLATE=check is set, so every Rust template a24 builds is also compared with its Python one (item 46).
Compared per call as a23's check_same_a22.py does (every operation's name, qubits, clbits and angles as a float's hex,
the global phase, the initial and final layout; a refusal by its exception type).

A. Without PSF_RING_ZIP: a23's 272 ring cases (FakeFez, FakeTorino, FakeMarrakesh, FakeKingston; n = 4-20; L = 1,
   2, 4, 8; `compile_for_model_circuit` and `model_template(...).bind`), refusals without a Target, 40 handed-on
   inputs. Expected: identical, except where a23 placed the ring by VF2's search (item 47): a24 refuses when no snake
   piece holds the ring, or uses the snake piece on every working coupler where a23 searched first.
B. Failed couplers (item 47): each device with seeded random couplers reported failed (error 0.7), 120 cases; the
   same expectation.
C. PSF_RING_ZIP=1 (item 48): on the same 272 cases, the template against TOQB's `ring_zip.ring_template_z` (a22 with
   psf_compile c61, the version of the 2026-10-22 comparison): the same template (gate list, angles within 1e-12,
   runs, layout) and the same chosen schedule, except where TOQB's a22 placed the ring by VF2 and a24 on the second
   snake (item 47: the same schedule chosen, another line); and the default call builds without error. The schedule
   chosen is counted per ring size.
"""
import contextlib
import importlib.util
import io
import os
import random
import sys
import warnings
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import InstructionProperties
from qiskit_ibm_runtime import fake_provider

warnings.simplefilter("ignore")
os.environ["PSF_RUST_TEMPLATE"] = "check"
RELEASE = "psf_compile.py"
C61 = "patches/psf_compile_c61_2026-10-15/psf_compile.py"
A22 = "patches/psf_ai_compile_a22_2026-10-15/psf_ai_compile.py"
A23 = "patches/psf_ai_compile_a23_2026-10-16/psf_ai_compile.py"
A24 = "patches/psf_ai_compile_a24_2026-10-22/psf_ai_compile.py"
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
        if type(e).__name__ == "RuntimeError" and "item 46" in str(e):
            raise  # a missing library or a Rust / Python difference stops the check
        return ("error", type(e).__name__)


def same(o_a, o_b):
    if isinstance(o_a, tuple) or isinstance(o_b, tuple) or o_a is None or o_b is None:
        return o_a == o_b
    return bits(o_a) == bits(o_b)


def ghz(n):
    c = QuantumCircuit(n, n)
    c.h(0)
    for j in range(n - 1):
        c.cx(j, j + 1)
    c.measure(range(n), range(n))
    return c


def placement(pc, n, t):
    """How a23 placed the ring and what item 47 allows: "a23 snake" (a23's own line is a snake piece: a24 must be
    identical), "item 47 refusal" (no snake piece holds it: a23 searched, a24 refuses), "item 47 second snake"
    (a23 took VF2's path on the most common couplers, a24 the snake piece on every working coupler)."""
    lat = pc._chain_lattice(t, 0.5)
    if lat is None:
        return "a23 snake"
    K, kadj, fadj, snake = lat
    hf = pc._chain_place(n, K, fadj, snake)[1]
    if hf == "snake":
        return "a23 snake"
    hk = pc._chain_place(n, K, kadj, snake)[1]
    if hf is None and hk == "snake":
        return "a23 snake"
    return "item 47 second snake" if hk == "snake" else "item 47 refusal"


def classify(pc, n, t, o3, o4):
    if same(o3, o4):
        return "identical"
    how = placement(pc, n, t)
    if how == "item 47 refusal" and o4 == ("error", "OutOfScopeError"):
        return how
    if how == "item 47 second snake" and not isinstance(o4, tuple) and o4 is not None:
        return how
    return "different"


def cut(t, p, seed):
    """A copy of `t` with each coupler reported failed (error 0.7, both directions) with probability p (seeded)."""
    import pickle
    t = pickle.loads(pickle.dumps(t))
    r = random.Random(seed)
    g2 = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
    pairs = sorted({tuple(sorted(q)) for q in t[g2] if q is not None})
    for a, b in pairs:
        if r.random() < p:
            for qa in ((a, b), (b, a)):
                if qa in t[g2]:
                    old = t[g2][qa]
                    t.update_instruction_properties(g2, qa, InstructionProperties(duration=old.duration, error=0.7))
    return t


def main():
    sys.path.insert(0, os.environ["TOQB"])
    from toqb import ring_zip as RZ
    from toqb.task_loop import angle_list, model_circuit, n_params
    with contextlib.redirect_stdout(io.StringIO()):
        c61 = load(C61, "psf_compile")
        c61.WARN_WITHOUT_TARGET = False
        a22 = load(A22, "ai22")
        rel = load(RELEASE, "psf_compile")
        rel.WARN_WITHOUT_TARGET = False
        a23, a24 = load(A23, "ai23"), load(A24, "ai24")
    print("a24", a24.AI_COMPILE_VERSION.split()[0], "| psf_compile", rel.VERSION,
          "| psfbind", a24._psfbind().version)
    rng = np.random.default_rng(7)
    cases = []
    for dev in DEVICES:
        t = getattr(fake_provider, dev)().target
        for n in range(4, 21):
            for L in (1, 2, 4, 8):
                qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L,
                                              np.pi / 2), n, L)
                cases.append((dev, t, n, L, qc))

    # A
    os.environ.pop("PSF_RING_ZIP", None)
    cnt, diffs = Counter(), []
    for dev, t, n, L, qc in cases:
        for kind in ("recompile", "template") + (("notarget",) if n == 4 and L == 1 else ()):
            c = classify(rel, n, t, run(a23, kind, qc, t), run(a24, kind, qc, t))
            cnt[c] += 1
            if c == "different":
                diffs.append((dev, n, L, kind))
    for dev in DEVICES:
        t = getattr(fake_provider, dev)().target
        hand = [ghz(5), ghz(12), random_circuit(5, 6, max_operands=2, measure=True, seed=1),
                random_circuit(8, 4, max_operands=2, measure=True, seed=2),
                model_circuit(angle_list(np.full(n_params(30, 2), 0.3), np.zeros(30), 30, 2, np.pi / 2), 30, 2)]
        for j, qc in enumerate(hand):
            for kind in ("recompile", "template"):
                c = classify(rel, qc.num_qubits, t, run(a23, kind, qc, t), run(a24, kind, qc, t))
                cnt["handed on: " + c] += 1
                if c == "different":
                    diffs.append((dev, "handoff", j, kind))
    print("A (no switch):", dict(cnt), "| differences:", diffs)
    ok_a = not diffs

    # B
    cntb, diffs_b = Counter(), []
    k = 0
    for dev in DEVICES + ("FakeFez",):
        base = getattr(fake_provider, dev)().target
        for j in range(24):
            p = (0.03, 0.06, 0.1)[j % 3]
            t = cut(base, p, 1000 * k + j)
            n, L = (6, 8, 10, 14)[j % 4], (2, 4)[(j // 4) % 2]
            qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L,
                                          np.pi / 2), n, L)
            for kind in ("recompile", "template"):
                o3, o4 = run(a23, kind, qc, t), run(a24, kind, qc, t)
                c = classify(rel, n, t, o3, o4)
                cntb[c] += 1
                if c == "different":
                    diffs_b.append((dev, k, j, n, L, kind, str(o3)[:40], str(o4)[:40]))
        k += 1
    print("B (failed couplers, 120 cases x 2 calls):", dict(cntb), "| other differences:", diffs_b)
    ok_b = not diffs_b

    # C
    os.environ["PSF_RING_ZIP"] = "1"
    chosen, cntc, diffs_c = Counter(), Counter(), []
    for dev, t, n, L, qc in cases:
        try:
            tz = a24.model_template(qc, t)
        except a24.OutOfScopeError:
            cntc["item 47 refusal"] += 1
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            if a22._ring_embeds(qc, t, 0.5) or a22._ring_read_path(qc) is None:
                ref, lab = a22.model_template(qc, t), "compiler"
            else:
                ref, lab = RZ.ring_template_z(a22, c61, qc, t)
        if tz is None or ref is None:
            if (tz is None) != (ref is None):
                diffs_c.append((dev, n, L, "one is None"))
            continue
        if lab == "compiler":
            ok = same(tz.bind(qc), ref.bind(qc))
            cntc["embeds: identical" if ok else "embeds: different"] += 1
            if not ok:
                diffs_c.append((dev, n, L, "embedded"))
            continue
        ref.schedule = lab
        d = a24.compare_templates(tz, ref)
        chosen[(n, getattr(tz, "schedule", None))] += 1
        if not d:
            cntc["identical"] += 1
        elif placement(rel, n, t) == "item 47 second snake" and tz.schedule == lab:
            cntc["item 47 second snake (same schedule)"] += 1  # TOQB's a22 searched with VF2 first
        else:
            cntc["different"] += 1
            diffs_c.append((dev, n, L, d[:3]))
        basis = [g for g in ("cz", "rz", "sx", "x") if g in t.operation_names]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                a24.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)
            cntc["default call built"] += 1
        except Exception as e:  # noqa: BLE001
            diffs_c.append((dev, n, L, "default call", type(e).__name__))
    os.environ.pop("PSF_RING_ZIP", None)
    print("C (PSF_RING_ZIP=1) against TOQB's ring_template_z:", dict(cntc), "| differences:", diffs_c)
    print("C schedule chosen per ring size:", sorted(chosen.items()))
    print("A24_STATS", a24.A24_STATS)
    ok = ok_a and ok_b and not diffs_c
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
