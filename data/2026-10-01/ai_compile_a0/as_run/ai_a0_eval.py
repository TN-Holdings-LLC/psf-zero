"""ai_a0_eval.py -- pre-registered held-out evaluation of psf_ai_compile 2026-10-01.a0 (workplace).

Arms (all with psf_compile 2026-10-01.c2, psf_smart_layout 2026-10-01.c2, the same Rust core):
  C2  compile_for_hardware(entangling_basis="cx", layout_search=True, seed_transpiler=0)
  A0  psf_ai_compile.compile_for_model_circuit (defaults)
  L3  Qiskit transpile(optimization_level=3, seed_transpiler=0), reference
Inputs (none used while developing a0):
  R  random dense circuits, 3-5 qubits, Python random seeds 3001-3060, on FakeAuckland, FakeKingston, FakeTorino
  T  12 new textbook circuits, gate form and PennyLane-style unitary form, on the same three devices
  M  measured copies of the first 30 Auckland random circuits, run on AerSimulator (A0 only)
  F  fall-through: 8 random_circuit inputs of 10-20 qubits (above SMALL_MAX_QUBITS) on FakeKingston
Helpers (component-wise equivalence, module loading, hashing) come from core_fix_c2_eval.py (locked 2026-10-01).

  python ai_a0_eval.py run --compile <psf_compile.py> --layout <psf_smart_layout.py> --ai <psf_ai_compile.py>
                           [--out ai_a0_raw.json] [--dry]
  python ai_a0_eval.py score [--out ai_a0_raw.json]
Times: wall clock in this process, one compile each (median over circuits is what is scored).
"""
import argparse
import contextlib
import io
import json
import math
import os
import random
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fix_c2_eval as H  # noqa: E402

DEVICES = ("FakeAuckland", "FakeKingston", "FakeTorino")
DRY_OFFSET = 700000


def new_textbook():
    from qiskit import QuantumCircuit
    cs = []
    qc = QuantumCircuit(4, name="IQFT4")  # inverse QFT with leading swaps
    qc.swap(0, 3); qc.swap(1, 2)
    for j in reversed(range(4)):
        for k in reversed(range(j + 1, 4)):
            qc.cp(-math.pi / 2 ** (k - j), k, j)
        qc.h(j)
    cs.append(qc)
    qc = QuantumCircuit(6, name="W6")  # W state, 6 qubits, cry cascade
    qc.x(0)
    for k in range(5):
        qc.cry(2 * math.acos(math.sqrt(1.0 / (6 - k))), k, k + 1)
        qc.cx(k + 1, k)
    cs.append(qc)
    qc = QuantumCircuit(6, name="GHZline6")
    qc.h(0)
    for k in range(5):
        qc.cx(k, k + 1)
    cs.append(qc)
    qc = QuantumCircuit(5, name="BV5")  # Bernstein-Vazirani, secret 1011, ancilla 4
    qc.x(4); qc.h(range(5))
    for k, bit in enumerate("1011"):
        if bit == "1":
            qc.cx(k, 4)
    qc.h(range(4))
    cs.append(qc)
    for n in (4, 5):  # QAOA ring p=1, rzz via cx-rz-cx
        qc = QuantumCircuit(n, name=f"QAOAring{n}")
        qc.h(range(n))
        for k in range(n):
            a, b = k, (k + 1) % n
            qc.cx(a, b); qc.rz(0.8, b); qc.cx(a, b)
        for k in range(n):
            qc.rx(0.6, k)
        cs.append(qc)
    qc = QuantumCircuit(4, name="HEA4")  # hardware-efficient ansatz, 2 layers, cz ring
    for layer in range(2):
        for k in range(4):
            qc.ry(0.3 + 0.2 * k + 0.1 * layer, k)
        for k in range(4):
            qc.cz(k, (k + 1) % 4)
    cs.append(qc)
    qc = QuantumCircuit(4, name="Heisenberg4")  # 2 Trotter steps, open chain
    for _ in range(2):
        for a in range(3):
            qc.rxx(0.2, a, a + 1); qc.ryy(0.2, a, a + 1); qc.rzz(0.2, a, a + 1)
    cs.append(qc)
    qc = QuantumCircuit(3, name="Grover3")  # one iteration, marked 111
    qc.h(range(3)); qc.ccz(0, 1, 2)
    qc.h(range(3)); qc.x(range(3)); qc.ccz(0, 1, 2); qc.x(range(3)); qc.h(range(3))
    cs.append(qc)
    qc = QuantumCircuit(4, name="QPE3")  # phase estimation of a phase gate, 3 counting qubits
    qc.x(3); qc.h(range(3))
    for k in range(3):
        qc.cp(2 * math.pi * 0.375 * 2 ** k, k, 3)
    qc.swap(0, 2)
    for j in range(3):
        for k in range(j):
            qc.cp(-math.pi / 2 ** (j - k), k, j)
        qc.h(j)
    cs.append(qc)
    qc = QuantumCircuit(5, name="Rotate5")  # cyclic shift by SWAPs after phases
    for k in range(5):
        qc.h(k); qc.rz(0.1 * (k + 1), k)
    for k in range(4):
        qc.swap(k, k + 1)
    qc.cp(0.4, 0, 4)
    cs.append(qc)
    qc = QuantumCircuit(4, name="Bell2Swap")  # two Bell pairs exchanged
    qc.h(0); qc.cx(0, 1); qc.h(2); qc.cx(2, 3); qc.swap(1, 2); qc.cz(0, 1); qc.cz(2, 3)
    cs.append(qc)
    out = []
    for c in cs:
        out.append((c.name + "/gates", c))
        out.append((c.name + "/unitary", H.to_unitary_form(c)))
    return out


def run(args):
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    comp = H.load_module(args.compile, "psf_compile")       # a0 imports psf_compile by name
    lay = H.load_module(args.layout, "psf_smart_layout")
    ai = H.load_module(args.ai, "psf_ai_compile")
    import psf_zero_core
    off = DRY_OFFSET if args.dry else 0
    meta = dict(compile_version=comp.VERSION, layout_version=lay.LAYOUT_VERSION, ai_version=ai.AI_COMPILE_VERSION,
                core_version=getattr(psf_zero_core, "CORE_VERSION", None), dry=bool(args.dry),
                qiskit=__import__("qiskit").__version__, cpu_count=os.cpu_count(),
                sha={"compile": H.norm_sha(args.compile), "layout": H.norm_sha(args.layout),
                     "ai": H.norm_sha(args.ai), "script": H.norm_sha(os.path.abspath(__file__)),
                     "helpers": H.norm_sha(H.__file__)})
    print("META", json.dumps(meta), flush=True)
    raw = {"meta": meta, "R": [], "T": [], "M": [], "F": []}

    def c2(qc, cm, nat):
        return H.cfh(comp, qc, cm, nat)

    def a0(qc, cm, nat):
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            o = ai.compile_for_model_circuit(qc, cm, nat)
        return o, time.perf_counter() - t

    def l3(qc, cm, nat):
        return H.two_q(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0))

    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    sim = AerSimulator()
    tb = new_textbook()
    for bname in DEVICES:
        cm, nat = H.backend(bname)
        seeds = range(off + 3001, off + (3005 if args.dry else 3061))
        for seed in seeds:
            rng = random.Random(seed)
            n = rng.choice([3, 4, 5])
            qc = H.rand_dense(n, rng.randint(6, 20), rng)
            oc = c2(qc, cm, nat)
            oa, dt = a0(qc, cm, nat)
            row = dict(device=bname, seed=seed, n=n, c2=H.two_q(oc), a0=H.two_q(oa), L3=l3(qc, cm, nat),
                       a0_s=round(dt, 4), a0_fid=H.component_fidelity(qc, oa), c2_fid=H.component_fidelity(qc, oc))
            raw["R"].append(row)
            print("R", json.dumps(row), flush=True)
            if bname == "FakeAuckland" and seed < off + 3031:
                ideal = Statevector(qc).probabilities_dict()
                qm = qc.copy()
                qm.measure_all()
                om, _ = a0(qm, cm, nat)
                counts = sim.run(om, shots=20000, seed_simulator=seed).result().get_counts()
                tvd = 0.5 * sum(abs(counts.get(k, 0) / 20000 - ideal.get(k, 0)) for k in set(counts) | set(ideal))
                raw["M"].append(dict(seed=seed, tvd=round(tvd, 5)))
        for name, qc in (tb[:4] if args.dry else tb):
            oc = c2(qc, cm, nat)
            oa, dt = a0(qc, cm, nat)
            row = dict(device=bname, name=name, n=qc.num_qubits, c2=H.two_q(oc), a0=H.two_q(oa), L3=l3(qc, cm, nat),
                       a0_s=round(dt, 4), a0_fid=H.component_fidelity(qc, oa), c2_fid=H.component_fidelity(qc, oc))
            raw["T"].append(row)
            print("T", json.dumps(row), flush=True)
    cm, nat = H.backend("FakeKingston")
    for k, (nq, d) in enumerate([(10, 10), (12, 8), (14, 8), (16, 6), (18, 6), (20, 5), (11, 12), (13, 10)][:2 if args.dry else 8]):
        qc = random_circuit(nq, d, max_operands=2, seed=off + 4001 + k)
        oc = c2(qc, cm, nat)
        oa, dt = a0(qc, cm, nat)
        row = dict(n=nq, depth=d, c2_digest=H.digest(oc), a0_digest=H.digest(oa), c2=H.two_q(oc), a0=H.two_q(oa))
        raw["F"].append(row)
        print("F", json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1)
    print("wrote", args.out)


def score(args):
    raw = json.load(open(args.out))
    m = raw["meta"]
    c0 = (m["compile_version"] == "2026-10-01.c2" and m["layout_version"].startswith("2026-10-01.c2")
          and m["ai_version"] == "2026-10-01.a0")
    print("C0 versions:", "OK" if c0 else "MISMATCH", m["compile_version"], m["layout_version"], m["ai_version"],
          "core", m["core_version"])
    res = {}
    V = H.verdict
    rows = raw["R"] + raw["T"]
    bad = [r for r in rows if r["a0_fid"] is not None and r["a0_fid"] <= 1 - H.FID_TOL]
    unc = [r for r in rows if r["a0_fid"] is None]
    res["A1"] = V(not bad and len(unc) <= 0.05 * len(rows), bool(bad))
    print(f"A1 A0 outputs not equivalent: {len(bad)}, not checkable: {len(unc)} of {len(rows)} -> {res['A1']}")
    above_c2 = [r for r in rows if r["a0"] > r["c2"]]
    res["A2"] = V(not above_c2, bool(above_c2))
    print(f"A2 A0 above C2: {len(above_c2)} of {len(rows)} -> {res['A2']}")
    ok3, bad3, ok4, bad4 = True, False, True, False
    for d in DEVICES:
        R = [r for r in raw["R"] if r["device"] == d]
        sa, sc, sl = sum(r["a0"] for r in R), sum(r["c2"] for r in R), sum(r["L3"] for r in R)
        frac = sum(r["a0"] > r["L3"] for r in R) / len(R)
        print(f"   {d}: sum2q C2 {sc} A0 {sa} L3 {sl} | A0/L3 {sa / sl:.3f} A0>L3 {frac:.1%} | A0/C2 {sa / sc:.3f}"
              f" | A0 median {statistics.median(r['a0_s'] for r in R) * 1000:.0f} ms")
        ok3 &= sa <= 1.02 * sl and frac <= 0.10
        bad3 |= sa > 1.06 * sl or frac > 0.20
        ok4 &= sa <= 0.97 * sc
        bad4 |= sa > 0.99 * sc
    res["A3"] = V(ok3, bad3)
    res["A4"] = V(ok4, bad4)
    print(f"A3 A0 at L3 level (sum <= 1.02 x L3 and <= 10 % above, every device) -> {res['A3']}")
    print(f"A4 A0 improves on C2 (sum <= 0.97 x C2, every device) -> {res['A4']}")
    T = raw["T"]
    share = sum(r["a0"] <= r["L3"] for r in T) / len(T)
    res["A5"] = V(share >= 0.90, share < 0.75)
    print(f"A5 textbook compiles with A0 <= L3: {share:.1%} of {len(T)} -> {res['A5']}")
    for r in T:
        print(f"   {r['device']:13s} {r['name']:22s} C2 {r['c2']:3d} A0 {r['a0']:3d} L3 {r['L3']:3d}  {r['a0_s'] * 1000:6.0f} ms")
    meds = {d: statistics.median(r["a0_s"] for r in rows if r["device"] == d) for d in DEVICES}
    mx = max(r["a0_s"] for r in rows)
    res["A6"] = V(all(v <= 0.25 for v in meds.values()) and mx <= 2.0, any(v > 0.5 for v in meds.values()))
    print(f"A6 A0 time: medians {', '.join(f'{d} {v * 1000:.0f} ms' for d, v in meds.items())}; max {mx:.2f} s -> {res['A6']}")
    M = raw["M"]
    mb = [r for r in M if r["tvd"] > 0.03]
    res["A7"] = V(not mb, bool(mb))
    print(f"A7 measured circuits with TVD > 0.03: {len(mb)} of {len(M)} (max {max(r['tvd'] for r in M):.4f}) -> {res['A7']}")
    F = raw["F"]
    fb = [r for r in F if r["a0_digest"] != r["c2_digest"]]
    res["A8"] = V(not fb, bool(fb))
    print(f"A8 circuits above 8 qubits where A0 output != C2 output: {len(fb)} of {len(F)} -> {res['A8']}")
    print("SUMMARY", json.dumps(res))
    print("DECISION:", "ALL CONFIRMED" if c0 and all(v == "CONFIRMED" for v in res.values()) else "NOT ALL CONFIRMED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--compile")
    ap.add_argument("--layout")
    ap.add_argument("--ai")
    ap.add_argument("--out", default="ai_a0_raw.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
