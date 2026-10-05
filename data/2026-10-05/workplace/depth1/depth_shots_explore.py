"""Exploratory (not pre-registered, written after the scored run): how much accuracy does the margin buy when
shots are few? Re-samples the recorded noisy z (with each point's readout error) at fewer shots, 200 repetitions,
same random numbers in every arm. Reads scored/deploy_*.json only."""
import glob, json, sys, zlib
import numpy as np
out = sys.argv[1]
D = {}
for p in glob.glob(f"{out}/deploy_*.json"):
    d = json.load(open(p)); m = d["meta"]; D[(m["dataset"], m["n"], m["device"], m["arm"])] = d["rows"]
def shot_acc(rows, shots, reps=200):
    acc = []
    for r in rows:
        p0 = (1 + r["z_noisy"]) / 2
        p0m = p0 * (1 - r["e01"]) + (1 - p0) * r["e10"]
        rng = np.random.default_rng(zlib.crc32(f"{r['L']}|{r['i']}|{shots}".encode()))
        k = rng.binomial(shots, p0m, size=reps)
        z = 2 * k / shots - 1
        acc.append(np.mean(np.sign(z) == r["y"]) + 0.5 * np.mean(z == 0) * 0)
    return float(np.mean(acc))
print("pooled shot accuracy (datasets, n, L>=8) by device, arm and shots")
for dev in ("FakeAuckland", "FakeTorino"):
    for n in (4, 6):
        line = [f"{dev} n={n}"]
        for shots in (32, 100, 300, 1000, 4000):
            vals = {}
            for arm in ("RPSF", "C12", "L3T"):
                rows = [r for ds in ("BC", "D38") for r in D[(ds, n, dev, arm)] if r["L"] >= 8]
                vals[arm] = shot_acc(rows, shots)
            line.append(f"{shots}: " + " ".join(f"{a} {v:.4f}" for a, v in vals.items()))
        print(" | ".join(line))
