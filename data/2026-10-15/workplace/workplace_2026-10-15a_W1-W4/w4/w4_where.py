"""W4: where does c57's in-scope output depend on orientation? Stats per orientation for the sensitive cells."""
import os, sys, copy, json
sys.argv = [sys.argv[0], "probe_out"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4_orient.py")).read().split("def main():")[0])
M = V.M57
names = ["SELECT_STATS", "L3C_STATS", "CHAIN_STATS", "GRAPH_STATS", "STEINER_STATS", "COMPARE_STATS", "FRAC_STATS", "DET_STATS"]
def snap(): return {n: copy.deepcopy(getattr(M, n)) for n in names}
def diff(a, b):
    out = {}
    for n in names:
        for k, v in b[n].items():
            if isinstance(v, (int, float)) and v != a[n].get(k, 0):
                out[f"{n}.{k}"] = v - a[n].get(k, 0)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)) and vv != a[n].get(k, {}).get(kk, 0):
                        out[f"{n}.{k}.{kk}"] = vv - a[n].get(k, {}).get(kk, 0)
    return out
for dev, tk, test in (("FakeMarrakesh", "F", "chain_cz_n12"), ("FakeFez", "F", "chain_cz_n16"), ("FakeFez", "T", "hhtree_heis_n16")):
    T, F = V.device(dev); tg = F if tk == "F" else T
    b = build(test)
    for o in ORIENT:
        s0 = snap(); out = compile_with(M, orient(b, o, test), tg); s1 = snap()
        print(dev, tk, test, o, round(V.metrics(out, tg)["dur_us"], 3), diff(s0, s1), flush=True)
