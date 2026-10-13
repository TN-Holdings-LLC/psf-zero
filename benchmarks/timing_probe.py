"""TIMING-PROBE: on which inputs can 2026-10-11.4's output depend on the machine's speed? Wraps psf_smart_layout's
smart_vf2_layout and packing_layout and records, per compile, whether a time limit shaped the search: an attempt that
failed (after which call limits are set from the measured rate), a stop because time ran out, or a packing search that
timed out. One device, every test of a list, in this process.
    cd <psf-zero repository>; python <this file> --bp BP --tests TESTS.json --device FakeTorino --out OUT.jsonl"""
import argparse, contextlib, io, json, os, sys, time, warnings
warnings.simplefilter("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--bp"); ap.add_argument("--tests"); ap.add_argument("--device"); ap.add_argument("--out"); ap.add_argument("--script")
a = ap.parse_args()
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
os.environ["C42_TESTS"] = os.environ["C43_TESTS"] = a.tests
import importlib
V = importlib.import_module(a.script)
from qiskit_ibm_runtime import fake_provider
sl = V.load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
r4 = V.load(os.path.join(REPO, "psf_compile.py"), "r4")
log = []
orig_vf2, orig_pack = sl.smart_vf2_layout, sl.packing_layout


def vf2(*x, **k):
    lm, info = orig_vf2(*x, **k)
    budget = k.get("time_budget_s", 2.0)
    att = info.get("attempts", [])
    nominal = {1: k.get("per_attempt_call_limit", 50_000), 2: k.get("fallback_call_limit", 300_000)}
    cut = any(t["call_limit"] < nominal[t["phase"]] for t in att)
    ran_out = (info.get("elapsed_s") or 0) >= budget - sl.MIN_ATTEMPT_S - 0.05
    sens = cut or ran_out
    log.append(dict(kind="vf2", found=lm is not None, phase=info.get("phase"), attempts=len(att),
                    failed_attempts=sum(1 for t in att if not t["found"]), elapsed=round(info.get("elapsed_s") or 0, 3), cut=cut,
                    ran_out=ran_out,
                    timing_sensitive=bool(sens)))
    return lm, info


def pack(*x, **k):
    t0 = time.perf_counter()
    r = orig_pack(*x, **k)
    el = time.perf_counter() - t0
    budget = k.get("time_budget_s", sl.PACKING_TIME_BUDGET_S)
    log.append(dict(kind="packing", found=r is not None, elapsed=round(el, 3), timing_sensitive=r is None and el >= 0.9 * budget))
    return r


sl.smart_vf2_layout, sl.packing_layout = vf2, pack
b = getattr(fake_provider, a.device)()
tests = json.load(open(a.tests))
with open(a.out, "w") as fh:
    for t in sorted(tests):
        try:
            qc = V.build(a.bp, t)
        except Exception as e:  # noqa: BLE001
            continue
        if qc.num_qubits > b.target.num_qubits:
            continue
        log.clear()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r4.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = type(e).__name__
        fh.write(json.dumps(dict(test=t, ok=ok, t=round(time.perf_counter() - t0, 2), calls=list(log),
                                 timing_sensitive=any(c["timing_sensitive"] for c in log))) + "\n")
        fh.flush()
