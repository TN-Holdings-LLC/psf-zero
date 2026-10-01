"""Exploratory: a0 vs a1 vs L3 on development inputs (the a0 held-out set, now seen)."""
import sys, os, io, contextlib, warnings, random, time, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); warnings.simplefilter("ignore")
import core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a0 = H.load_module(sys.argv[3], "ai_a0"); a1 = H.load_module(sys.argv[4], "ai_a1")
import ai_a0_eval as A
from qiskit import transpile
which = sys.argv[5] if len(sys.argv) > 5 else "all"
def run(mod, qc, cm, nat):
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        o = mod.compile_for_model_circuit(qc, cm, nat)
    return o, time.perf_counter() - t
tb = A.new_textbook()
for d in A.DEVICES:
    cm, nat = H.backend(d)
    items = []
    if which in ("all", "R"):
        for seed in range(3001, 3061):
            rng = random.Random(seed); n = rng.choice([3, 4, 5]); items.append(("R", H.rand_dense(n, rng.randint(6, 20), rng)))
    if which in ("all", "T"):
        items += [("T:" + nm, qc) for nm, qc in tb]
    S = {}
    for kind, qc in items:
        k = kind[0]
        o0, t0 = run(a0, qc, cm, nat); o1, t1 = run(a1, qc, cm, nat)
        l3 = H.two_q(transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0))
        f1 = H.component_fidelity(qc, o1)
        s = S.setdefault(k, dict(a0=0, a1=0, L3=0, a1_gt_L3=0, a1_gt_a0=0, bad=0, n=0, t0=[], t1=[], textra=[]))
        s["a0"] += H.two_q(o0); s["a1"] += H.two_q(o1); s["L3"] += l3; s["n"] += 1
        s["a1_gt_L3"] += H.two_q(o1) > l3; s["a1_gt_a0"] += H.two_q(o1) > H.two_q(o0)
        s["bad"] += (f1 is None) or f1 <= 1 - 1e-9; s["t0"].append(t0); s["t1"].append(t1)
        if k == "T" and H.two_q(o1) != l3:
            s["textra"].append(f"{kind[2:]} a0 {H.two_q(o0)} a1 {H.two_q(o1)} L3 {l3}")
    for k, s in S.items():
        print(f"{d} {k}: n {s['n']} sum a0 {s['a0']} a1 {s['a1']} L3 {s['L3']} | a1>L3 {s['a1_gt_L3']} a1>a0 {s['a1_gt_a0']} bad/unchecked {s['bad']} | median ms a0 {statistics.median(s['t0'])*1000:.0f} a1 {statistics.median(s['t1'])*1000:.0f} max a1 {max(s['t1']):.2f}s", flush=True)
        for line in s["textra"]: print("    ", line)
