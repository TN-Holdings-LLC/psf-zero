"""C66-VAL: C4 (shape changes), C5 (default call), C6 (walk) and the reported durations; C1-C3 are c63a_score.py's.

    python data/2026-10-16/c66_val/score_c66.py <out dir>     (writes <out dir>/score_c66.txt)

Written after the pre-registration was fixed and while the run was going (its log lines were visible), from the
criteria as fixed; it does not change them."""
import json
import os
import statistics
import sys

out_dir = sys.argv[1]
cells = []
for root, dirs, files in os.walk(out_dir):
    for f in sorted(files):
        if f.startswith("c63_val_rows_"):
            cells += [json.loads(line) for line in open(os.path.join(root, f))]
L = []
bad4 = [(c["part"], c["device"], c["test"], c["target"], c["rebuilt"], c["switched"]) for c in cells
        if c["rebuilt"] or c["switched"]]
L.append(f"C4: cells {len(cells)} (part R {sum(c['part'] == 'R' for c in cells)}, part D "
         f"{sum(c['part'] == 'D' for c in cells)}); cells with a shape change {bad4} -> "
         f"{'CONFIRMED' if not bad4 and len(cells) == 190 else 'REFUTED' if bad4 else 'NOT COMPLETE'}")
dc = json.load(open(os.path.join(out_dir, "default_call.json")))
L.append(f"C5: inputs {dc['inputs']}, different {len(dc['different'])}; refusals {dc['refusals']}, different "
         f"{len(dc['refusals_different'])} -> {dc['verdict']}")
w = json.load(open(os.path.join(out_dir, "walk.json")))
ex = max(c["exact"] for c in w["checks"])
L.append(f"C6: laps {w['laps']}, shape changes {w['shape_changes']}; largest difference at the checks {ex:.1e} -> "
         f"{'CONFIRMED' if w['shape_changes'] == 0 and ex <= 1e-6 and w['laps'] == 100000 else 'REFUTED'}")
L += ["", "Reported: F cells (chain/graph), duration over c62's per input (median, largest, share above c62); bind "
      "interleaved / back to back (ms)"]
allmax = 0.0
for c in sorted(cells, key=lambda c: (c["part"], c["device"], c["test"])):
    if c["kind"] == "native" or c["target"] != "F":
        continue
    rat = [r["dur"] / r["dur62"] for r in c["rows"]]
    allmax = max(allmax, max(rat))
    L.append(f"  {c['part']} {c['device']:18s} {c['test']:16s} median {statistics.median(rat):.4f} largest "
             f"{max(rat):.4f} above {sum(x > 1 + 1e-9 for x in rat)}/{len(rat)}  bind {c['bind_median_ms'] or 0:.3f} / "
             f"{c['bind_tight_median_ms'] or 0:.3f}")
L.append(f"  largest ratio over every F cell: {allmax:.4f}")
L.append(f"Walk: bind median {w['median_ms']:.3f} ms, p99 {w['p99_ms']:.3f} ms; peak memory {w['peak_rss_mb']:.0f} MB; "
         f"duration over c62's median {statistics.median(c['dur'] / c['dur62'] for c in w['checks']):.4f}, largest "
         f"{max(c['dur'] / c['dur62'] for c in w['checks']):.4f}")
text = "\n".join(L) + "\n"
print(text)
open(os.path.join(out_dir, "score_c66.txt"), "w").write(text)
