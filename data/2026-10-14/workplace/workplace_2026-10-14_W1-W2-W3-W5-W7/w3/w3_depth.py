"""w3_depth.py -- W3 (2026-10-14): DEPTH / DEPTH-R's data re-uploading classifier (Addenda 345, 399; the model of
data/2026-10-05/workplace/depth1/depth_eval.py, built by its own `circuit()`), n = 4 and 6, L = 1..16 layers,
compiled by c48 and by Qiskit level 3. Measured: correctness, CZ count, circuit duration. No ESP, no search.

    python w3_depth.py --out <dir> [--devices FakeFez,FakeTorino]

Angles: one fixed data point and one fixed parameter vector per (n, L), drawn once from numpy's default_rng(20261014)
(not trained, not varied; the compiled structure does not depend on the values except through exact cancellations,
which generic angles avoid). All qubits measured. Arms as W2: PSF (c48 as shipped, item 80's choice recorded), PSF-C
(level3_candidate=False, near_failed_error=None), QK3 (generate_preset_pass_manager(3, b, seed_transpiler=0)).
Correctness: total variation distance of the exact outcome distributions (Aer statevector)."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
import warnings

import numpy as np

REPO = os.path.abspath(os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "w2"))
import w2_tfim as W  # noqa: E402

DEPTH_FILE = os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1", "depth_eval.py")
LAYERS = range(1, 17)
NS = (4, 6)


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="FakeFez,FakeTorino")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import fake_provider
    spec = importlib.util.spec_from_file_location("depth_eval", DEPTH_FILE)
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    W.load(W.LAYOUT_FILE, "psf_smart_layout")
    c48 = W.load(W.C48_FILE, "w3_c48")
    c48.WARN_WITHOUT_TARGET = False
    print(W.stamp(), "SCRIPT", W.sha(os.path.abspath(__file__)), "| depth_eval", W.sha(DEPTH_FILE), "| c48",
          W.sha(W.C48_FILE))
    rng = np.random.default_rng(20261014)
    angles = {}
    for n in NS:
        for L in LAYERS:
            angles[(n, L)] = (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L)))
    fh = open(os.path.join(a.out, "w3_rows.jsonl"), "w")
    for dev in a.devices.split(","):
        b = getattr(fake_provider, dev)()
        tgt = b.target
        pm3 = generate_preset_pass_manager(3, b, seed_transpiler=0)
        for n in NS:
            for L in LAYERS:
                x, th = angles[(n, L)]
                qc = D.circuit(x, th, n, L)
                qc.measure_all()
                ref = W.probs(qc)
                for arm in ("PSF", "PSF-C", "QK3"):
                    t0 = time.perf_counter()
                    chosen = None
                    if arm == "QK3":
                        out = pm3.run(qc)
                    else:
                        kw = {} if arm == "PSF" else dict(level3_candidate=False, near_failed_error=None)
                        before = dict(c48.L3C_STATS)
                        with contextlib.redirect_stdout(io.StringIO()):
                            out = c48.compile_for_hardware(qc, backend=b, seed_transpiler=0, **kw)
                        if arm == "PSF":
                            chosen = "level3" if c48.L3C_STATS["kept_level3"] > before.get("kept_level3", 0) else "psf"
                    secs = time.perf_counter() - t0
                    dur, cz, two = W.duration_and_cz(out, tgt)
                    err = W.tvd(W.probs(out), ref)
                    r = dict(device=dev, n=n, L=L, arm=arm, cz=cz, dur_us=round(dur, 4), err=err,
                             correct=bool(err <= W.TOL), isa=W.isa_ok(out, tgt), compile_s=round(secs, 2),
                             item80=chosen, logical_cz=n * L)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"{dev} n={n} L={L:2d} {arm:5s} CZ {cz:4d} dur {dur:6.2f} us err {err:.1e} {secs:.1f}s"
                          f"{' item80=' + chosen if chosen else ''}", flush=True)
    fh.close()
    print(W.stamp(), "done")


if __name__ == "__main__":
    main()
