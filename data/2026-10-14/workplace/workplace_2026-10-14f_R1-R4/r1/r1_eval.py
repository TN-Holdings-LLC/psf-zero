"""R1 evaluation: c54 + item 94 against c54, QK3 (seed 0, as written), the user's seed 0 (hand-layered, QK3 seed 0)
and the user's best (family_map: seeds 0-4, as written or hand-layered); FakeFez T and F; the family map's tests."""
import sys, os, json, time, contextlib, io, tempfile, importlib.util
sys.argv = ["family_map.py", tempfile.mkdtemp() + "/fm"]
spec = importlib.util.spec_from_file_location("fm", "benchmarks/family_map.py"); FM = importlib.util.module_from_spec(spec)
sys.modules["fm"] = FM; spec.loader.exec_module(FM)
NEW = FM.H.load_module(os.environ.get("PSF_NEW", "patches/psf_compile_c54_94_2026-10-14/psf_compile.py"), "c54_94"); NEW.WARN_WITHOUT_TARGET = False
C54 = FM.H.load_module("patches/psf_compile_c54_2026-10-14/psf_compile.py", "c54_ref"); C54.WARN_WITHOUT_TARGET = False
from qiskit_ibm_runtime.fake_provider import FakeFez
b = FakeFez(); TG = {"T": b.target, "F": FM.fractional(b.target)}
fmrows = {(r["test"], r["target"]): r for r in map(json.loads, open("data/2026-10-14/family_map/family_map_rows.jsonl"))}
out = open("r1_rows.jsonl", "w")
for fam, names in FM.FAMILIES:
    if fam.startswith("classifier"):
        continue
    for name in names:
        qc = FM.build(name); hl = FM.hand_layered(qc)
        for tk, t in TG.items():
            k0 = dict(NEW.ZZ94_STATS)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                new = NEW.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            ns = time.time() - t0
            with contextlib.redirect_stdout(io.StringIO()):
                ref = C54.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            u0 = FM.dur(FM.qk3(hl, t, 0), t) if hl is not None else None
            fr = fmrows[(name, tk)]
            r = dict(test=name, family=fam, target=tk, new_us=FM.dur(new, t), new_2q=FM.n2(new), c54_us=FM.dur(ref, t),
                     c54_2q=FM.n2(ref), user_seed0_us=u0, user_best_us=fr["user_us"], user_how=fr["user_how"],
                     qk3_us=fr["qk3_us"], exact=FM.exact(qc, new), secs=round(ns, 1),
                     item94={k: NEW.ZZ94_STATS[k] - k0[k] for k in NEW.ZZ94_STATS if NEW.ZZ94_STATS[k] != k0[k]})
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"{tk} {name:16s} new {r['new_us']:6.2f} ({r['new_2q']}) | c54 {r['c54_us']:6.2f} | user s0 {u0 if u0 is None else round(u0,2)} | user best {r['user_best_us']:6.2f} | QK3 {r['qk3_us']:6.2f} | exact {r['exact']} | {r['item94']} {ns:.1f}s", flush=True)
