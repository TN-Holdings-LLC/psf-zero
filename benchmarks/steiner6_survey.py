"""Survey of BP-FINAL's HamLib FakeTorino tests: size, term weights, whether all terms commute, diagonal.
    cd <psf-zero repository>; python benchmarks/steiner6_survey.py <benchpress> <out json>"""
import json, os, sys, warnings
warnings.simplefilter("ignore")
REPO = os.getcwd()  # the psf-zero repository root
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
import numpy as np
import steiner3 as S3, bp_mock as B
BP = sys.argv[1]  # a Benchpress clone
print("reading", S3.calibrate())
rows = []
for t in S3.tests(BP):
    qc, _ = B.build(BP, t[2], t[3])
    evo = [i for i in qc.data if i.operation.name == "PauliEvolution"]
    if not evo:
        print(t[1][-40:], "no evo"); continue
    op = evo[0].operation.operator
    if isinstance(op, list):
        print(t[1][-40:], "list"); continue
    pl = op.paulis
    x, z = pl.x, pl.z
    w = (x | z).sum(axis=1)
    diag = not x.any()
    # anticommutation of all pairs (sampled if large)
    T = len(pl)
    idx = np.arange(T) if T <= 1500 else np.random.default_rng(0).choice(T, 1500, replace=False)
    xs, zs = x[idx].astype(np.int8), z[idx].astype(np.int8)
    anti = ((xs @ zs.T + zs @ xs.T) % 2)
    frac = anti.sum() / (len(idx) * (len(idx) - 1))
    r = dict(test=t[1].split("[", 1)[1][:-1], n=qc.num_qubits, terms=T, wmax=int(w.max()), wmean=round(float(w.mean()), 2),
             diag=bool(diag), anti_frac=round(float(frac), 4), meas=sum(1 for i in qc.data if i.operation.name == "measure"))
    rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(rows, open(sys.argv[2], "w"))
