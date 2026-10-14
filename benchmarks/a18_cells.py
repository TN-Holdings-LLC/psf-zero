"""W3's 64 classifier cells (FakeFez, FakeTorino; n = 4, 6; L = 1-16; W3's angles and circuits, as the workplace's
p2_eval.py) through one arm, on psf_compile c55 with item 94 off (dropped, Addendum 488). Offline, no device.
    python benchmarks/a18_cells.py <a17|a18|QK3> <T|F> <out .jsonl>   (refuses an existing file)
T: the device's own Target; F: with rzz and rx added (test_c51's construction). Rows: duration (ASAP on the Target,
measurements left out), two-qubit count, rzz count, total variation distance from the input's exact distribution,
the chosen candidate, compile seconds."""
import contextlib
import io
import json
import os
import sys
import time

arm, tk, out = sys.argv[1:4]
if os.path.exists(out):
    raise SystemExit(f"STOP: {out} exists (a run never overwrites another)")
EV = os.path.join("data", "2026-10-14", "workplace", "workplace_2026-10-14e_Q1-Q4", "q3", "eval")
sys.path.insert(0, EV)
from p1_common import REPO, load, fractional, dur_2q, rzz_ok, W2, np  # noqa: E402

P = load(os.path.join(REPO, "patches", "psf_compile_c55_2026-10-14", "psf_compile.py"), "psf_compile")
P.WARN_WITHOUT_TARGET = False
P.ZZ_RELAYER = False
FE = {"a17": os.path.join(REPO, "patches", "psf_ai_compile_a17_2026-10-14", "psf_ai_compile.py"),
      "a18": os.path.join(REPO, "patches", "psf_ai_compile_a18_2026-10-14", "psf_ai_compile.py")}
A = None if arm == "QK3" else load(FE[arm], "front_end")
D = load(os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1", "depth_eval.py"), "depth_eval")
from qiskit import transpile  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

rng = np.random.default_rng(20261014)  # W3's angles, drawn in W3's order
angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6)
          for L in range(1, 17)}
fh = open(out, "w")
for dev in ("FakeFez", "FakeTorino"):
    t0 = getattr(fake_provider, dev)().target
    tg = fractional(t0) if tk == "F" else t0
    basis = [g for g in tg.operation_names if g in ("cz", "rz", "sx", "x")]
    for n in (4, 6):
        for L in range(1, 17):
            x, th = angles[(n, L)]
            qc = D.circuit(x, th, n, L)
            qc.measure_all()
            s0 = time.perf_counter()
            chosen = None
            with contextlib.redirect_stdout(io.StringIO()):
                if A is None:
                    c = transpile(qc, target=tg, optimization_level=3, seed_transpiler=0)
                else:
                    c, info = A.compile_for_model_circuit(qc, tg.build_coupling_map(), basis, target=tg,
                                                          return_info=True)
                    chosen = info.get("chosen")
            secs = time.perf_counter() - s0
            d, two, nr = dur_2q(c, tg)
            r = dict(device=dev, n=n, L=L, arm=arm, target=tk, dur_us=round(d, 4), two_q=two, rzz=nr,
                     in_range=rzz_ok(c), tvd=W2.tvd(W2.probs(c), W2.probs(qc)), chosen=chosen, secs=round(secs, 2))
            fh.write(json.dumps(r) + "\n")
            fh.flush()
