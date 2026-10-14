"""w2_table.py -- the CZ / duration table of W2 from w2_rows.jsonl (no new computation)."""
import json
import sys

rows = [json.loads(x) for x in open(sys.argv[1])]
R = {(r["device"], r["n"], r["steps"], r["arm"]): r for r in rows}
devs = sorted({r["device"] for r in rows}, key=lambda d: d != "FakeFez")
ok = all(r["correct"] and r["isa"] for r in rows)
print(f"rows {len(rows)}; every output correct and ISA: {ok}; largest error "
      f"{max(r['err'] for r in rows):.1e}")
for d in devs:
    print(f"\n### {d}\n")
    print("| n | steps | PSF: CZ, us | PSF-C: CZ, us | QK3: CZ, us | QK3-BW: CZ, us | PSF / QK3 time | PSF / QK3-BW time |")
    print("|---|---|---|---|---|---|---|---|")
    for n in sorted({r["n"] for r in rows}):
        for s in sorted({r["steps"] for r in rows}):
            c = [R.get((d, n, s, a)) for a in ("PSF", "PSF-C", "QK3", "QK3-BW")]
            if not all(c):
                continue
            cell = " | ".join(f"{x['cz']}, {x['dur_us']:.2f}" for x in c)
            print(f"| {n} | {s} | {cell} | {c[0]['dur_us'] / c[2]['dur_us']:.3f} | "
                  f"{c[0]['dur_us'] / c[3]['dur_us']:.3f} |")
psf = {(d, s): sorted({R[(d, n, s, 'PSF')]['dur_us'] for n in sorted({r['n'] for r in rows}) if (d, n, s, 'PSF') in R})
       for d in devs for s in sorted({r["steps"] for r in rows})}
print("\nPSF durations by (device, steps) over all n:", {f"{k[0]} s={k[1]}": v for k, v in psf.items()})
same_cz = sum(1 for (d, n, s, a), r in R.items() if a == "PSF" and (d, n, s, "QK3-BW") in R
              and r["cz"] == R[(d, n, s, "QK3-BW")]["cz"])
tot = sum(1 for k in R if k[3] == "PSF")
print(f"PSF CZ == QK3-BW CZ in {same_cz} of {tot} cells")
ratios = [R[(d, n, s, "PSF")]["dur_us"] / R[(d, n, s, "QK3-BW")]["dur_us"] for (d, n, s, a) in R if a == "PSF"
          and (d, n, s, "QK3-BW") in R]
print(f"PSF / QK3-BW duration: min {min(ratios):.3f}, max {max(ratios):.3f}")
item80 = sum(1 for r in rows if r["arm"] == "PSF" and r.get("item80") == "level3")
same = sum(1 for (d, n, s, a), r in R.items() if a == "PSF" and r["cz"] == R[(d, n, s, "PSF-C")]["cz"]
           and r["dur_us"] == R[(d, n, s, "PSF-C")]["dur_us"])
print(f"item 80 reported 'level 3 kept' in {item80} of {tot}; PSF output equal to PSF-C's (CZ and duration) in {same}")
