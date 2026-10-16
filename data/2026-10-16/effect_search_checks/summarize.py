import json, statistics as st, sys
D = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/"
arms = {a: json.load(open(D + f)) ["rows"] for a, f in (("rel", "rel.json"), ("before", "before.json"), ("now", sys.argv[1] if len(sys.argv) > 1 else "now.json"), ("qk3", "qk3.json")) if __import__("os").path.exists(D + f)}
def key(r): return (r["src"], r["dev"], r["case"], r.get("L"), r["tg"])
idx = {a: {key(r): r for r in rs} for a, rs in arms.items()}
keys = list(idx["now"])
for src in ("c63val", "toqb"):
    ks = [k for k in keys if k[0] == src]
    print(f"== {src}: {len(ks)} cases")
    for a in idx:
        rs = [idx[a][k] for k in ks if k in idx[a]]
        ok = [r for r in rs if "compile_s" in r]
        T = sum(r["compile_s"] for r in ok)
        tv = [r["tvd"] for r in ok if r.get("tvd") is not None]
        line = (f"  {a:6} compile total {T:8.2f} s, median {st.median([r['compile_s'] for r in ok])*1e3:7.1f} ms, max {max(r['compile_s'] for r in ok):6.2f} s; "
                f"2q mean {st.mean(r['twoq'] for r in ok):6.1f}; duration mean {st.mean(r['dur_us'] for r in ok):5.2f} us; isa {sum(r['isa'] for r in ok)}/{len(ok)}; "
                f"rzz ok {sum(r['rzz_ok'] for r in ok)}; exact (tvd<1e-6) {sum(t < 1e-6 for t in tv)}/{len(tv)} checked; errors {len(rs)-len(ok)}")
        tb = [r for r in rs if "template_s" in r]
        if tb:
            line += f"\n         template build total {sum(r['template_s'] for r in tb):.2f} s ({len(tb)}), bind median {st.median(r['bind_s'] for r in tb)*1e3:.2f} ms"
        print(line)
    # paired
    for a in ("rel", "before", "qk3"):
        if a not in idx: continue
        pr = [(idx[a][k], idx["now"][k]) for k in ks if k in idx[a] and "compile_s" in idx[a][k] and "compile_s" in idx["now"][k]]
        rat = sorted(n["compile_s"] / o["compile_s"] for o, n in pr)
        d2 = [n["twoq"] - o["twoq"] for o, n in pr]; dd = [n["dur_us"] - o["dur_us"] for o, n in pr]
        print(f"  now vs {a}: time ratio median {st.median(rat):.3f} (10% {rat[len(rat)//10]:.3f}, 90% {rat[9*len(rat)//10]:.3f}); "
              f"2q: now fewer {sum(x<0 for x in d2)}, same {sum(x==0 for x in d2)}, more {sum(x>0 for x in d2)}; "
              f"duration: now shorter {sum(x<-1e-9 for x in dd)}, same {sum(abs(x)<=1e-9 for x in dd)}, longer {sum(x>1e-9 for x in dd)}; "
              f"mean duration {st.mean(o['dur_us'] for o,n in pr):.2f} -> {st.mean(n['dur_us'] for o,n in pr):.2f} us")
