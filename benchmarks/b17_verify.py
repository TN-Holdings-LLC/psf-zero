"""b17_verify.py -- independent re-count of the B17 run (Addendum 298), written after the run. Reads the raw jsonl
files only; checks provenance and counts, re-derives P0 and H1-H5, and adds the failure maps reported without
prediction.   python b17_verify.py <run dir>"""
import glob, json, os, sys
from collections import defaultdict
EXPECT = dict(git_head="f0095e6", script="04c0a80121170041768569737c7331cb7acbb8672a04729a977333f02e3a736a",
              psf="2026-10-01.1")
WANT = dict(W1=1050, W2=500, W3=500, W4=450)
COMP = ("QK1", "QK2", "QK3", "QK3CZ", "QK3U", "PSF", "PSFNG")
out = sys.argv[1]
rows, flags = defaultdict(list), []
for p in sorted(glob.glob(os.path.join(out, "b17_W*_n*.jsonl"))):
    lines = [json.loads(x) for x in open(p)]
    m = lines[0]["meta"]
    for k in ("git_head", "psf"):
        if m[k] != EXPECT[k]:
            flags.append(f"{p}: {k}")
    if m["sha"]["script"] != EXPECT["script"] or m["smoke"]:
        flags.append(f"{p}: script/smoke")
    if len(lines) - 1 != WANT[m["workload"]]:
        flags.append(f"{p}: {len(lines) - 1} circuits")
    for r in lines[1:]:
        r["n"] = m["n"]
        rows[m["workload"]].append(r)
bad = lambda v: "error" in v or v["inf"] > 1e-6
errors = sum("error" in v for w in rows for r in rows[w] for v in r["res"].values())
ctrl = sum(bad(r["res"][c]) for w in rows for r in rows[w] for c in ("QK3CZ", "QK3U"))
p0 = len(rows) == 4 and not flags and errors == 0 and ctrl == 0
print(f"files ok: {not flags}; compile errors {errors}; control failures {ctrl}; P0 {'PASS' if p0 else 'FAIL'}")
f = {(w, c): (sum(bad(r["res"][c]) for r in rows[w]), len(rows[w])) for w in rows for c in COMP}
for w in ("W1", "W2", "W3", "W4"):
    print(w, "  ".join(f"{c} {f[(w, c)][0]}/{f[(w, c)][1]}" for c in COMP))
for w in ("W2",):
    for n in (4, 6):
        print(f"  {w} n={n}: " + ", ".join(f"{c} {sum(bad(r['res'][c]) for r in rows[w] if r['n'] == n)}" for c in ("QK1", "QK3", "PSFNG")))
V = lambda ok, b: "REFUTED" if b else ("CONFIRMED" if ok else "AMBIGUOUS")
r_ = lambda w, c: f[(w, c)][0] / f[(w, c)][1]
print("H1", V(r_("W2", "QK2") >= 0.1 and r_("W2", "QK3") >= 0.1, f[("W2", "QK2")][0] == 0 and f[("W2", "QK3")][0] == 0),
      "H2", V(all(f[("W3", c)][0] == 0 for c in ("QK1", "QK2", "QK3")), any(f[("W3", c)][0] > 0 for c in ("QK1", "QK2", "QK3"))),
      "H3", V(all(f[(w, "PSF")][0] == 0 for w in rows), any(f[(w, "PSF")][0] > 0 for w in rows)),
      "H4", V(r_("W2", "PSFNG") >= 0.01, f[("W2", "PSFNG")][0] == 0),
      "H5", V(f[("W1", "QK3")][0] >= 1, f[("W1", "QK3")][0] == 0))
print("PSFNG W1 failures by (dt, r):")
cell = defaultdict(lambda: [0, 0])
for r in rows["W1"]:
    k = (r["params"]["dt"], r["params"]["r"])
    cell[k][0] += bad(r["res"]["PSFNG"]); cell[k][1] += 1
for dt in (1e-3, 1e-2, 0.1):
    print(f"  dt={dt:g}: " + "  ".join(f"r={rr:g} {cell[(dt, rr)][0]}/{cell[(dt, rr)][1]}" for rr in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)))
print("PSFNG W1 failures by n:", {n: sum(bad(r["res"]["PSFNG"]) for r in rows["W1"] if r["n"] == n) for n in (4, 6)})
g = {w: sum(r["res"]["PSF"].get("guard_rejected", 0) for r in rows[w]) for w in rows}
gc = {w: sum(r["res"]["PSF"].get("guard_rejected", 0) > 0 for r in rows[w]) for w in rows}
print("PSF guard rejections by workload:", g, "circuits with >=1 rejection:", gc)
both = sum(bad(r["res"]["PSFNG"]) and r["res"]["PSF"].get("guard_rejected", 0) > 0 for w in rows for r in rows[w])
ng = sum(bad(r["res"]["PSFNG"]) for w in rows for r in rows[w])
print(f"PSFNG failures where the guarded PSF rejected at least once: {both} of {ng}")
print("flags:", flags or "none")
