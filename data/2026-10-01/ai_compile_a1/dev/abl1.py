import sys, os, io, contextlib, warnings, random, time, statistics
sys.path.insert(0, '.'); warnings.simplefilter("ignore")
import core_fix_c2_eval as H
comp = H.load_module(sys.argv[1], "psf_compile"); lay = H.load_module(sys.argv[2], "psf_smart_layout")
a1 = H.load_module("psf_ai_compile.py", "ai_a1")
import ai_a0_eval as A
cm, nat = H.backend(sys.argv[3])
items = []
for seed in range(3001, 3061):
    rng = random.Random(seed); n = rng.choice([3,4,5]); items.append(H.rand_dense(n, rng.randint(6,20), rng))
items += [qc for _, qc in A.new_textbook()]
cfgs = {"a1 full": {}, "no L3 layout": dict(L3=False), "no expanded": dict(EX=False), "no split": dict(SPLIT=False), "seeds 0-1 only": dict(seeds=(0,1))}
orig_split = a1._split_multi_qubit
for name, c in cfgs.items():
    a1.L3_LAYOUT_CANDIDATE = c.get("L3", True); a1.EXPANDED_START = c.get("EX", True)
    a1._split_multi_qubit = orig_split if c.get("SPLIT", True) else (lambda q, max_rounds=6: q)
    kw = {"seeds": c["seeds"]} if "seeds" in c else {}
    tot, ts = 0, []
    for qc in items:
        t = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            o = a1.compile_for_model_circuit(qc, cm, nat, **kw)
        ts.append(time.perf_counter() - t); tot += H.two_q(o)
    print(f"{sys.argv[3]} {name:16s} sum2q {tot} median ms {statistics.median(ts)*1000:.0f}", flush=True)
