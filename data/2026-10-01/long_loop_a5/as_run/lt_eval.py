"""lt_eval.py -- pre-registered long-loop test of psf_ai_compile 2026-10-01.a5 (30,000 laps), workplace 2026-10-01.

What can drift over tens of thousands of calls is state kept between calls, not the arithmetic (each compile starts
from scratch). a4 kept per-gate errors in caches keyed by id(target); a5 keys them by value. This test:
  * cycles a pool of 50 new random circuits (Python random seeds 10101-10150, 3-5 qubits) on FakeAuckland,
  * replaces the Target object every EPOCH laps, alternating calibration A (as shipped) and calibration B
    (cx errors x1.6 on every cx touching qubits 0-13, T2 x0.5 on qubits 14-26), freeing the old object first,
  * records per lap: output digest, wall time, the cost a5 reports for its choice, and an independent reference
    cost of the returned circuit (stateaware.py: the same estimate, written separately, no cache),
  * samples RSS every 250 laps and checks exactness (noiseless, component-wise) every 500 laps.
Workers: two processes, laps 0-14999 and 15000-29999 (calibration by global lap index). Control: a4 for 3,000 laps.

  python lt_eval.py worker --wid 0 --start 0 --laps 15000 --ai <a5.py> --compile <c2> --layout <c2> --out w0.json
  python lt_eval.py worker --wid 9 --start 0 --laps 3000 --ai <a4.py> ... --out control.json   (control)
  python lt_eval.py score --files w0.json w1.json --control control.json
"""
import argparse
import contextlib
import gc
import io
import json
import os
import random
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fix_c2_eval as H  # noqa: E402
import stateaware  # noqa: E402

POOL_SEEDS = range(10101, 10151)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def make_target(cal):
    from qiskit_ibm_runtime import fake_provider
    t = fake_provider.FakeAuckland().target
    if cal == "B":
        for qs, pr in t["cx"].items():
            if pr is not None and pr.error is not None and (qs[0] <= 13 or qs[1] <= 13):
                pr.error *= 1.6
        for q in range(14, 27):
            p = t.qubit_properties[q]
            if p is not None and p.t2 is not None:
                p.t2 *= 0.5
    return t


def worker(args):
    comp = H.load_module(args.compile, "psf_compile")
    lay = H.load_module(args.layout, "psf_smart_layout")
    ai = H.load_module(args.ai, "psf_ai_compile_lt")
    pool = []
    for s in POOL_SEEDS:
        rng = random.Random(s + args.pool_offset)
        n = rng.choice([3, 4, 5])
        pool.append(H.rand_dense(n, rng.randint(6, 20), rng))
    nat = ["cx", "rz", "sx", "x"]
    meta = dict(wid=args.wid, start=args.start, laps=args.laps, epoch=args.epoch, pool_offset=args.pool_offset,
                ai=ai.AI_COMPILE_VERSION,
                c2=comp.VERSION, layout=lay.LAYOUT_VERSION,
                sha={"ai": H.norm_sha(args.ai), "script": H.norm_sha(os.path.abspath(__file__)),
                     "stateaware": H.norm_sha(stateaware.__file__), "helpers": H.norm_sha(H.__file__)})
    print("META", json.dumps(meta), flush=True)
    ref = {}           # (cal, idx) -> digest
    mism = []          # determinism mismatches
    cost_bad = []      # |reported - reference| > tol
    times, rss, exact = [], [], []
    tgt, cal_now = None, None
    for lap in range(args.start, args.start + args.laps):
        cal = "A" if (lap // args.epoch) % 2 == 0 else "B"
        if cal != cal_now or lap == args.start:
            tgt = None
            gc.collect()
            tgt = make_target(cal)
            cm = tgt.build_coupling_map()
            cal_now = cal
        idx = lap % len(pool)
        qc = pool[idx]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out, info = ai.compile_for_model_circuit(qc, cm, nat, target=tgt, return_info=True)
        times.append(time.perf_counter() - t0)
        reported = [t for t in info["tried"] if t[0] == "placement"][-1][2]
        reference = stateaware.state_aware_cost(out, tgt)
        if abs(reported - reference) > 2e-6:
            cost_bad.append(dict(lap=lap, cal=cal, idx=idx, reported=reported, reference=reference))
        dg = H.digest(out)
        key = (cal, idx)
        if key in ref and ref[key] != dg:
            mism.append(dict(lap=lap, cal=cal, idx=idx))
        ref.setdefault(key, dg)
        if (lap - args.start) % 250 == 0:
            rss.append((lap - args.start, rss_mb()))
        if (lap - args.start) % 500 == 0:
            f = H.component_fidelity(qc, out)
            exact.append(dict(lap=lap, fid=f))
        if (lap - args.start) % 1000 == 999:
            print(f"lap {lap + 1} median(last 1000) {statistics.median(times[-1000:]) * 1000:.0f} ms "
                  f"rss {rss[-1][1]:.0f} MB mism {len(mism)} cost_bad {len(cost_bad)}", flush=True)
    res = dict(meta=meta, times=times, rss=rss, exact=exact, mismatches=mism, cost_bad=cost_bad,
               ref={f"{k[0]}:{k[1]}": v for k, v in ref.items()})
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("wrote", args.out, flush=True)


def score(args):
    V = H.verdict
    W = [json.load(open(p)) for p in args.files]
    C = json.load(open(args.control))
    res = {}
    c0 = all(w["meta"]["ai"] == "2026-10-01.a5" for w in W) and C["meta"]["ai"] == "2026-10-01.a4"
    total = sum(len(w["times"]) for w in W)
    print("G0 versions:", "OK" if c0 else "MISMATCH", [w["meta"]["ai"] for w in W], C["meta"]["ai"], "| laps", total)
    within = sum(len(w["mismatches"]) for w in W)
    cross = 0
    common = set(W[0]["ref"]) & set(W[1]["ref"]) if len(W) > 1 else set()
    for k in common:
        cross += W[0]["ref"][k] != W[1]["ref"][k]
    res["G1"] = V(within == 0 and cross == 0, within > 0 or cross > 0)
    print(f"G1 determinism: within-worker mismatches {within}, cross-worker mismatches {cross} of {len(common)} keys -> {res['G1']}")
    bad = sum(len(w["cost_bad"]) for w in W)
    res["G2"] = V(bad == 0, bad > 0)
    print(f"G2 reported cost = independent reference on every lap: {bad} laps off of {total} -> {res['G2']}")
    cb = len(C["cost_bad"])
    res["G3"] = V(cb >= 1, cb == 0)
    print(f"G3 control (a4) shows the defect: {cb} laps off of {len(C['times'])} -> {res['G3']}")
    ok4, bad4, ok5, bad5 = True, False, True, False
    for w in W:
        t = w["times"]
        early, late = statistics.median(t[100:1100]), statistics.median(t[-1000:])
        r = dict(w["rss"])
        r_start = min((v for k, v in w["rss"] if k >= 2000), default=float("nan"))
        r_start = [v for k, v in w["rss"] if k == 2000][0] if 2000 in r else r_start
        growth = w["rss"][-1][1] - r_start
        print(f"   worker {w['meta']['wid']}: median laps 101-1100 {early * 1000:.1f} ms, last 1000 {late * 1000:.1f} ms "
              f"(ratio {late / early:.3f}); RSS at lap 2000 {r_start:.0f} MB, end {w['rss'][-1][1]:.0f} MB (growth {growth:+.0f} MB); "
              f"max lap {max(t):.2f} s")
        ok4 &= late / early <= 1.20
        bad4 |= late / early > 1.50
        ok5 &= growth <= 50
        bad5 |= growth > 200
    res["G4"] = V(ok4, bad4)
    res["G5"] = V(ok5, bad5)
    print(f"G4 no time drift (last-1000 / laps 101-1100 median <= 1.20 per worker) -> {res['G4']}")
    print(f"G5 no memory growth (RSS growth after lap 2000 <= 50 MB per worker) -> {res['G5']}")
    ex = [e for w in W for e in w["exact"]]
    exb = [e for e in ex if e["fid"] is None or e["fid"] < 1 - 1e-9]
    res["G6"] = V(not exb, bool(exb))
    print(f"G6 sampled outputs exact: {len(ex) - len(exb)} of {len(ex)} -> {res['G6']}")
    a4_first = {k: v for k, v in C["ref"].items() if k.startswith("A:")}
    a5_first = {k: v for k, v in W[0]["ref"].items() if k.startswith("A:")}
    keys = set(a4_first) & set(a5_first)
    diff = sum(a4_first[k] != a5_first[k] for k in keys)
    res["G7"] = V(diff == 0 and len(keys) == 50, diff > 0)
    print(f"G7 a5 = a4 on calibration A (first sight of each circuit): {len(keys) - diff} of {len(keys)} identical -> {res['G7']}")
    print("SUMMARY", json.dumps(res))
    print("DECISION:", "ALL CONFIRMED" if c0 and all(v == "CONFIRMED" for v in res.values()) else "NOT ALL CONFIRMED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["worker", "score"])
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--laps", type=int, default=15000)
    ap.add_argument("--epoch", type=int, default=1000)
    ap.add_argument("--pool-offset", type=int, default=0, help="dry runs use 900000 (never the scored pool)")
    ap.add_argument("--ai")
    ap.add_argument("--compile")
    ap.add_argument("--layout")
    ap.add_argument("--out")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--control")
    a = ap.parse_args()
    worker(a) if a.mode == "worker" else score(a)


if __name__ == "__main__":
    main()
