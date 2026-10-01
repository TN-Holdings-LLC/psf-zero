"""ai_a1_eval.py -- pre-registered held-out evaluation of psf_ai_compile 2026-10-01.a1 (workplace).

Arms (psf_compile 2026-10-01.c2, psf_smart_layout 2026-10-01.c2, the same Rust core):
  C2  compile_for_hardware(entangling_basis="cx", layout_search=True, seed_transpiler=0)
  A0  psf_ai_compile 2026-10-01.a0, defaults
  A1  psf_ai_compile 2026-10-01.a1, defaults
  L3  Qiskit transpile(optimization_level=3, seed_transpiler=0), reference
Inputs (none used while developing a0 or a1):
  R  random dense circuits, 3-5 qubits, Python random seeds 5001-5060, on FakeAuckland, FakeKingston, FakeTorino,
     FakeFez (FakeFez never used before in this project's PSF-Zero tests)
  T  12 textbook circuits of a third, new set (several with 3-qubit gates or ring-shaped interaction),
     gate form and PennyLane-style unitary form, on the same four devices
  M  measured copies of the first 30 Auckland random circuits, A1, AerSimulator 20,000 shots
  F  8 random_circuit inputs of 10-20 qubits on FakeKingston (seeds 6001-6008), above the 8-qubit limit
Helpers come from core_fix_c2_eval.py (locked 2026-10-01).

  python ai_a1_eval.py run --compile <psf_compile.py> --layout <psf_smart_layout.py> --a0 <a0.py> --a1 <a1.py>
                           [--out ai_a1_raw.json] [--dry]
  python ai_a1_eval.py score [--out ai_a1_raw.json]
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

DEVICES = ("FakeAuckland", "FakeKingston", "FakeTorino", "FakeFez")
DRY_OFFSET = 800000


def unitary_form(qc):
    """PennyLane-style: every multi-qubit gate becomes a `unitary` (matrix via Operator, so gates without
    to_matrix such as MCXGate work too)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) >= 2:
            out.append(UnitaryGate(Operator(inst.operation).data), qs)
        else:
            out.append(inst.operation, qs)
    return out


def dry_textbook():
    """Harness check only (these two were in the first dry run and are not scored)."""
    from qiskit import QuantumCircuit
    cs = []
    qc = QuantumCircuit(4, name="ToffoliAdder")  # 1-bit full adder: a, b, cin -> sum on 2, carry on 3
    qc.x(0); qc.h(1)
    qc.ccx(0, 1, 3); qc.cx(0, 1); qc.ccx(1, 2, 3); qc.cx(1, 2); qc.cx(0, 1)
    cs.append(qc)
    qc = QuantumCircuit(5, name="SwapTest5")  # ancilla 0 compares |psi>(1,2) and |phi>(3,4)
    qc.ry(0.7, 1); qc.cx(1, 2); qc.ry(1.1, 3); qc.h(4)
    qc.h(0)
    qc.cswap(0, 1, 3); qc.cswap(0, 2, 4)
    qc.h(0)
    cs.append(qc)
    out = []
    for c in cs:
        out.append((c.name + "/gates", c))
        out.append((c.name + "/unitary", unitary_form(c)))
    return out


def textbook3():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_clifford
    cs = []
    qc = QuantumCircuit(4, name="CCZchain4")  # two overlapping ccz between Hadamard layers
    qc.h(range(4)); qc.ccz(0, 1, 2); qc.h(range(4)); qc.ccz(1, 2, 3); qc.h(range(4))
    cs.append(qc)
    qc = QuantumCircuit(3, name="CuccaroMajUma3")  # MAJ then UMA block of the ripple-carry adder
    qc.x(1); qc.h(2)
    qc.cx(2, 1); qc.cx(2, 0); qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 2); qc.cx(2, 0); qc.cx(0, 1)
    cs.append(qc)
    qc = QuantumCircuit(4, name="Grover4")  # one iteration, marked 1111, mcz via h-mcx-h
    qc.h(range(4))
    qc.h(3); qc.mcx([0, 1, 2], 3); qc.h(3)
    qc.h(range(4)); qc.x(range(4))
    qc.h(3); qc.mcx([0, 1, 2], 3); qc.h(3)
    qc.x(range(4)); qc.h(range(4))
    cs.append(qc)
    qc = QuantumCircuit(6, name="QAOAring6")
    qc.h(range(6))
    for k in range(6):
        qc.rzz(0.7, k, (k + 1) % 6)
    for k in range(6):
        qc.rx(0.5, k)
    cs.append(qc)
    qc = QuantumCircuit(5, name="HEA5ring")
    for layer in range(2):
        for k in range(5):
            qc.ry(0.2 + 0.15 * k + 0.05 * layer, k); qc.rz(0.1 * k, k)
        for k in range(5):
            qc.cx(k, (k + 1) % 5)
    cs.append(qc)
    qc = QuantumCircuit(5, name="IsingRing5")  # 2 Trotter steps, rzz ring + rx field
    for _ in range(2):
        for k in range(5):
            qc.cx(k, (k + 1) % 5); qc.rz(0.4, (k + 1) % 5); qc.cx(k, (k + 1) % 5)
        for k in range(5):
            qc.rx(0.3, k)
    cs.append(qc)
    qc = QuantumCircuit(5, name="DJ4")  # Deutsch-Jozsa, balanced f(x) = x0 xor x2 xor x3, ancilla 4
    qc.x(4); qc.h(range(5))
    for k in (0, 2, 3):
        qc.cx(k, 4)
    qc.h(range(4))
    cs.append(qc)
    qc = QuantumCircuit(4, name="DraperAdd2")  # QFT adder: |a> (0,1) added into |b> (2,3)
    qc.x(0); qc.x(3)
    qc.h(3); qc.cp(math.pi / 2, 2, 3); qc.h(2)
    qc.cp(math.pi, 0, 2); qc.cp(math.pi / 2, 1, 2); qc.cp(math.pi, 1, 3)
    qc.h(2); qc.cp(-math.pi / 2, 2, 3); qc.h(3)
    cs.append(qc)
    qc = QuantumCircuit(6, name="GHZstar6")
    qc.h(0)
    for k in range(1, 6):
        qc.cx(0, k)
    cs.append(qc)
    qc = QuantumCircuit(4, name="W4tree")  # W4 by a binary tree of controlled rotations
    qc.ry(2 * math.acos(math.sqrt(0.5)), 0)
    qc.cry(2 * math.acos(math.sqrt(0.5)), 0, 1); qc.cx(1, 0)
    qc.x(0); qc.cry(2 * math.acos(math.sqrt(0.5)), 0, 2); qc.x(0); qc.cx(2, 0)
    qc.x(1); qc.x(2); qc.ccx(1, 2, 3); qc.x(1); qc.x(2)
    cs.append(qc)
    qc = QuantumCircuit(3, name="TeleportUnitary")  # coherent teleportation (corrections as cz/cx)
    qc.ry(0.9, 0); qc.rz(0.4, 0)
    qc.h(1); qc.cx(1, 2); qc.cx(0, 1); qc.h(0)
    qc.cx(1, 2); qc.cz(0, 2)
    cs.append(qc)
    qc = random_clifford(4, seed=5101).to_circuit()
    qc.name = "Clifford4"
    cs.append(qc)
    out = []
    for c in cs:
        out.append((c.name + "/gates", c))
        out.append((c.name + "/unitary", unitary_form(c)))
    return out


def run(args):
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    comp = H.load_module(args.compile, "psf_compile")
    lay = H.load_module(args.layout, "psf_smart_layout")
    a0 = H.load_module(args.a0, "psf_ai_compile_a0")
    a1 = H.load_module(args.a1, "psf_ai_compile_a1")
    import psf_zero_core
    off = DRY_OFFSET if args.dry else 0
    meta = dict(compile_version=comp.VERSION, layout_version=lay.LAYOUT_VERSION, a0_version=a0.AI_COMPILE_VERSION,
                a1_version=a1.AI_COMPILE_VERSION, core_version=getattr(psf_zero_core, "CORE_VERSION", None),
                dry=bool(args.dry), qiskit=__import__("qiskit").__version__, cpu_count=os.cpu_count(),
                sha={"compile": H.norm_sha(args.compile), "layout": H.norm_sha(args.layout),
                     "a0": H.norm_sha(args.a0), "a1": H.norm_sha(args.a1),
                     "script": H.norm_sha(os.path.abspath(__file__)), "helpers": H.norm_sha(H.__file__)})
    print("META", json.dumps(meta), flush=True)
    raw = {"meta": meta, "R": [], "T": [], "M": [], "F": []}

    def front(mod, qc, cm, nat):
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            o = mod.compile_for_model_circuit(qc, cm, nat)
        return o, time.perf_counter() - t

    def one(qc, cm, nat, extra):
        oc = H.cfh(comp, qc, cm, nat)
        o0, t0 = front(a0, qc, cm, nat)
        o1, t1 = front(a1, qc, cm, nat)
        l3 = H.two_q(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0))
        row = dict(extra, n=qc.num_qubits, c2=H.two_q(oc), a0=H.two_q(o0), a1=H.two_q(o1), L3=l3,
                   a0_s=round(t0, 4), a1_s=round(t1, 4), a1_fid=H.component_fidelity(qc, o1))
        return row

    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    sim = AerSimulator()
    tb = dry_textbook() if args.dry else textbook3()
    for bname in DEVICES:
        cm, nat = H.backend(bname)
        for seed in range(off + 5001, off + (5004 if args.dry else 5061)):
            rng = random.Random(seed)
            n = rng.choice([3, 4, 5])
            qc = H.rand_dense(n, rng.randint(6, 20), rng)
            row = one(qc, cm, nat, dict(device=bname, seed=seed))
            raw["R"].append(row)
            print("R", json.dumps(row), flush=True)
            if bname == "FakeAuckland" and seed < off + 5031:
                ideal = Statevector(qc).probabilities_dict()
                qm = qc.copy()
                qm.measure_all()
                om, _ = front(a1, qm, cm, nat)
                counts = sim.run(om, shots=20000, seed_simulator=seed).result().get_counts()
                tvd = 0.5 * sum(abs(counts.get(k, 0) / 20000 - ideal.get(k, 0)) for k in set(counts) | set(ideal))
                raw["M"].append(dict(seed=seed, tvd=round(tvd, 5)))
        for name, qc in tb:
            row = one(qc, cm, nat, dict(device=bname, name=name))
            raw["T"].append(row)
            print("T", json.dumps(row), flush=True)
    cm, nat = H.backend("FakeKingston")
    for k, (nq, d) in enumerate([(10, 10), (12, 8), (14, 8), (16, 6), (18, 6), (20, 5), (11, 12), (13, 10)][:2 if args.dry else 8]):
        qc = random_circuit(nq, d, max_operands=2, seed=off + 6001 + k)
        oc = H.cfh(comp, qc, cm, nat)
        o1, _ = front(a1, qc, cm, nat)
        row = dict(n=nq, depth=d, c2_digest=H.digest(oc), a1_digest=H.digest(o1))
        raw["F"].append(row)
        print("F", json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1)
    print("wrote", args.out)


def score(args):
    raw = json.load(open(args.out))
    m = raw["meta"]
    c0 = (m["compile_version"] == "2026-10-01.c2" and m["layout_version"].startswith("2026-10-01.c2")
          and m["a0_version"] == "2026-10-01.a0" and m["a1_version"] == "2026-10-01.a1")
    print("C0 versions:", "OK" if c0 else "MISMATCH", m["compile_version"], m["layout_version"], m["a0_version"],
          m["a1_version"], "core", m["core_version"])
    V = H.verdict
    res = {}
    rows = raw["R"] + raw["T"]
    bad = [r for r in rows if r["a1_fid"] is not None and r["a1_fid"] <= 1 - H.FID_TOL]
    unc = [r for r in rows if r["a1_fid"] is None]
    res["B1"] = V(not bad and len(unc) <= 0.05 * len(rows), bool(bad))
    print(f"B1 A1 not equivalent: {len(bad)}, not checkable: {len(unc)} of {len(rows)} -> {res['B1']}")
    ok2, bad2, ok3, bad3 = True, False, True, False
    for d in DEVICES:
        R = [r for r in raw["R"] if r["device"] == d]
        D = [r for r in rows if r["device"] == d]
        s = {k: sum(r[k] for r in R) for k in ("c2", "a0", "a1", "L3")}
        f10 = sum(r["a1"] > r["a0"] for r in D) / len(D)
        fl3 = sum(r["a1"] > r["L3"] for r in R) / len(R)
        print(f"   {d}: R sums C2 {s['c2']} A0 {s['a0']} A1 {s['a1']} L3 {s['L3']} | A1/L3 {s['a1'] / s['L3']:.3f}"
              f" A1>L3 {fl3:.1%} | A1>A0 (R+T) {f10:.1%} | A1 median {statistics.median(r['a1_s'] for r in D) * 1000:.0f} ms")
        sa1, sa0 = sum(r["a1"] for r in D), sum(r["a0"] for r in D)
        ok2 &= sa1 < sa0 and f10 <= 0.02
        bad2 |= sa1 >= sa0 or f10 > 0.05
        ok3 &= s["a1"] <= 1.00 * s["L3"] and fl3 <= 0.05
        bad3 |= s["a1"] > 1.04 * s["L3"] or fl3 > 0.15
    res["B2"] = V(ok2, bad2)
    res["B3"] = V(ok3, bad3)
    print(f"B2 A1 improves on A0 (R+T sum lower and <= 2 % of circuits worse, every device) -> {res['B2']}")
    print(f"B3 A1 at or below L3 on R (sum <= 1.00 x L3 and <= 5 % above, every device) -> {res['B3']}")
    T = raw["T"]
    share = sum(r["a1"] <= r["L3"] for r in T) / len(T)
    res["B4"] = V(share >= 0.90, share < 0.75)
    print(f"B4 textbook compiles with A1 <= L3: {share:.1%} of {len(T)} -> {res['B4']}")
    for r in T:
        print(f"   {r['device']:13s} {r['name']:24s} C2 {r['c2']:3d} A0 {r['a0']:3d} A1 {r['a1']:3d} L3 {r['L3']:3d}  {r['a1_s'] * 1000:6.0f} ms")
    meds = {d: statistics.median(r["a1_s"] for r in rows if r["device"] == d) for d in DEVICES}
    mx = max(r["a1_s"] for r in rows)
    res["B5"] = V(all(v <= 0.40 for v in meds.values()) and mx <= 2.0, any(v > 0.80 for v in meds.values()))
    print(f"B5 A1 time: medians {', '.join(f'{d} {v * 1000:.0f} ms' for d, v in meds.items())}; max {mx:.2f} s -> {res['B5']}")
    M = raw["M"]
    mb = [r for r in M if r["tvd"] > 0.03]
    res["B6"] = V(not mb, bool(mb))
    print(f"B6 measured circuits with TVD > 0.03: {len(mb)} of {len(M)} (max {max(r['tvd'] for r in M):.4f}) -> {res['B6']}")
    F = raw["F"]
    fb = [r for r in F if r["a1_digest"] != r["c2_digest"]]
    res["B7"] = V(not fb, bool(fb))
    print(f"B7 circuits above 8 qubits where A1 output != C2 output: {len(fb)} of {len(F)} -> {res['B7']}")
    print("SUMMARY", json.dumps(res))
    print("DECISION:", "ALL CONFIRMED" if c0 and all(v == "CONFIRMED" for v in res.values()) else "NOT ALL CONFIRMED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--compile")
    ap.add_argument("--layout")
    ap.add_argument("--a0")
    ap.add_argument("--a1")
    ap.add_argument("--out", default="ai_a1_raw.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
