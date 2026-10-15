"""template_100k.py -- 100,000 binds of the ring template (a20, item 35), the training-loop analogue of
long_loop_100k's part E (Addendum 213): one 6-qubit, 8-layer ring classifier (W3's model) on FakeFez with IBM's
fractional gates; its angles on a random walk; every bind's time recorded; every 1,000th bind checked against the
input's exact outcome distribution; RSS every 1,000 binds. Exploratory (Addendum 494).
    python benchmarks/template_100k.py <out dir> [laps]   (refuses an existing out dir)"""
import gc
import json
import os
import resource
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
OUT = sys.argv[1]
LAPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-14", "workplace", "workplace_2026-10-14e_Q1-Q4", "q3",
                                                "eval")]
from p1_common import load, fractional, np, W2  # noqa: E402

P = load(os.path.join("patches", "psf_compile_c59_2026-10-15", "psf_compile.py"), "psf_compile")
P.WARN_WITHOUT_TARGET = False
A = load(os.path.join("patches", "psf_ai_compile_a20_2026-10-15", "psf_ai_compile.py"), "a20")
D = load(os.path.join("data", "2026-10-05", "workplace", "depth1", "depth_eval.py"), "depth_eval")
from qiskit_ibm_runtime import fake_provider  # noqa: E402

os.makedirs(OUT)
n, L = 6, 8
rng = np.random.default_rng(20261015)
tg = fractional(fake_provider.FakeFez().target)
qc = D.circuit(rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L)), n, L)
qc.measure_all()
t0 = time.perf_counter()
tpl = A.ring_template(qc, tg)
build = time.perf_counter() - t0
vals = np.array(tpl.values_of(qc))
times, checks, rss = [], [], []
gc.collect()
gc.freeze()
start = time.perf_counter()
for lap in range(1, LAPS + 1):
    vals = vals + rng.normal(0, 0.05, len(vals))  # a training step's update
    s = time.perf_counter()
    out = tpl.bind_values(vals.tolist())
    times.append(time.perf_counter() - s)
    if lap % 1000 == 0:
        ref = qc.copy_empty_like()
        k2v = {k: v for (k, _), v in zip(tpl.var_kinds, vals)}
        for k, ins in enumerate(qc.data):
            ref.append(ins.replace(operation=ins.operation.__class__(k2v[k])) if k in k2v else ins)
        checks.append((lap, W2.tvd(W2.probs(out), W2.probs(ref))))
        rss.append((lap, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        gc.collect()
    if lap % 10000 == 0:
        print(f"lap {lap}: median {statistics.median(times[-10000:])*1000:.3f} ms; last check {checks[-1][1]:.1e}; "
              f"RSS {rss[-1][1]:.0f} MB; {time.perf_counter() - start:.0f} s", flush=True)
ms = sorted(t * 1000 for t in times)
res = dict(laps=LAPS, n=n, L=L, device="FakeFez + rzz/rx", build_ms=build * 1000, params=len(tpl.params),
           median_ms=ms[len(ms) // 2], p99_ms=ms[int(0.99 * len(ms))], max_ms=ms[-1],
           first10_median_ms=statistics.median(times[:LAPS // 10]) * 1000,
           last10_median_ms=statistics.median(times[-LAPS // 10:]) * 1000,
           worst_check=max(c for _, c in checks), checks=len(checks), rss_first_mb=rss[0][1], rss_last_mb=rss[-1][1],
           total_s=time.perf_counter() - start)
json.dump(dict(result=res, checks=checks, rss=rss), open(os.path.join(OUT, "template_100k.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
