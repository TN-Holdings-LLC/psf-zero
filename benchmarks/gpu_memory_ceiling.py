"""gpu_memory_ceiling.py -- pre-registered: does lightning.gpu slow down silently
beyond device memory on a native-Linux RTX 4090 (24 GB), as it did on WSL2 with
an RTX 4070 (12 GB, home Addendum 143: about 15x at 29 qubits, no error)?

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: the GPU memory ceiling
of lightning.gpu on native Linux -- error or silent slowdown?" (2026-09-28).

Each size runs in its own child process (timeout 900 s): H on every qubit, then
two layers of RY on every qubit plus a CNOT chain, expectation of Z0; three
calls (the first includes allocation). A background thread polls nvidia-smi for
the device memory in use every 0.2 s.

    python -u gpu_memory_ceiling.py run   2>&1 | tee ~/vram_run.txt
    python -u gpu_memory_ceiling.py score 2>&1 | tee ~/vram_score.txt
Plumbing dry run only: --device lightning.qubit --sizes 12,13,14
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import subprocess
import sys
import threading
import time

HOME = os.path.expanduser("~")
SIZES = [26, 27, 28, 29, 30, 31]
TIMEOUT_S = 900
CALLS = 3
OUT_JSON = os.path.join(HOME, "vram_2026-09-28.json")


def gpu_mem_mib():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def n_gates(n):
    return n + 2 * (n + (n - 1))


def child(device, n):
    import numpy as np
    import pennylane as qml
    peak = {"mib": gpu_mem_mib()}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            m = gpu_mem_mib()
            if m is not None and (peak["mib"] is None or m > peak["mib"]):
                peak["mib"] = m
            time.sleep(0.2)
    th = threading.Thread(target=poll, daemon=True); th.start()
    base = gpu_mem_mib()
    rng = np.random.default_rng(n)
    theta = rng.uniform(-np.pi, np.pi, (2, n))
    out = {"n": n, "device": device, "gates": n_gates(n), "base_mib": base}
    try:
        t0 = time.perf_counter()
        dev = qml.device(device, wires=n)
        out["device_s"] = time.perf_counter() - t0

        @qml.qnode(dev, diff_method=None)
        def f():
            for q in range(n):
                qml.Hadamard(wires=q)
            for layer in range(2):
                for q in range(n):
                    qml.RY(theta[layer, q], wires=q)
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
            return qml.expval(qml.PauliZ(0))
        times, vals = [], []
        for _ in range(CALLS):
            t0 = time.perf_counter(); vals.append(float(f())); times.append(time.perf_counter() - t0)
        out.update({"status": "ok", "call_s": times, "value": vals[-1], "values_agree": max(vals) - min(vals) <= 1e-12})
    except Exception as e:
        out.update({"status": "error", "error": type(e).__name__ + ": " + str(e)[:300]})
    stop.set(); th.join(timeout=1)
    out["peak_mib"] = peak["mib"]
    print("RESULT " + json.dumps(out), flush=True)


def run(device, sizes):
    import pennylane as qml
    env = {"platform": platform.platform(), "python": platform.python_version(), "pennylane": qml.__version__,
           "device": device, "sizes": sizes, "timeout_s": TIMEOUT_S}
    try:
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        env["gpu"] = None
    from importlib import metadata
    for pkg in ("pennylane_lightning", "pennylane_lightning_gpu"):
        try:
            env[pkg] = metadata.version(pkg)
        except Exception:
            env[pkg] = None
    print("ENV", json.dumps(env))
    # C0: the GPU value at n = 12 against lightning.qubit (CPU) for the same circuit
    c0 = {}
    for dv in (device, "lightning.qubit"):
        r = subprocess.run([sys.executable, "-u", __file__, "child", "--device", dv, "--n", "12"],
                           capture_output=True, text=True, timeout=300)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        c0[dv] = json.loads(line[-1][7:]) if line else {"status": "no output", "stderr": r.stderr[-300:]}
    print("C0", json.dumps(c0))
    res = {"env": env, "c0": c0, "sizes": []}
    for n in sizes:
        t0 = time.perf_counter()
        try:
            r = subprocess.run([sys.executable, "-u", __file__, "child", "--device", device, "--n", str(n)],
                               capture_output=True, text=True, timeout=TIMEOUT_S)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
            rec = json.loads(line[-1][7:]) if line else {"n": n, "status": "crash", "returncode": r.returncode,
                                                        "stderr": r.stderr[-500:]}
            rec["returncode"] = r.returncode
        except subprocess.TimeoutExpired:
            rec = {"n": n, "status": "timeout"}
        rec["wall_s"] = time.perf_counter() - t0
        res["sizes"].append(rec)
        print("SIZE", json.dumps(rec), flush=True)
        with open(OUT_JSON, "w") as fh:
            json.dump(res, fh, indent=1)
    print("wrote", OUT_JSON)


def score():
    R = json.load(open(OUT_JSON)); env = R["env"]; S = R["sizes"]

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("ENV", json.dumps(env))
    g, c = R["c0"].get(env["device"], {}), R["c0"].get("lightning.qubit", {})
    ok0 = (env["sizes"] == SIZES and env["device"] == "lightning.gpu" and g.get("status") == "ok"
           and c.get("status") == "ok" and abs(g["value"] - c["value"]) <= 1e-10)
    print(f"C0: sizes {env['sizes']}, device {env['device']}, n=12 GPU vs CPU value "
          f"{g.get('value')} vs {c.get('value')}")
    if not ok0:
        print("C0 FAILED: nothing is scored.")
        return
    per_gate = {}
    for r in S:
        if r.get("status") == "ok":
            per_gate[r["n"]] = st.median(r["call_s"][1:]) / r["gates"]
    ok_ns = sorted(per_gate)
    ratios = {n: per_gate[n] / per_gate[n - 1] for n in ok_ns if n - 1 in per_gate}
    print("   per-gate time (median of calls 2-3):", {n: f"{per_gate[n]*1e3:.3f} ms" for n in ok_ns})
    print("   ratio to n-1:", {n: round(x, 2) for n, x in ratios.items()})
    mx = max(ratios.values()) if ratios else None
    print(f"W1 no silent jump among completed sizes: max ratio {mx} (confirmed <= 3.0, refuted >= 5.0) ->",
          "CONFIRMED" if mx is not None and mx <= 3.0 else ("REFUTED" if mx is not None and mx >= 5.0 else "AMBIGUOUS"))
    first_bad = next((r for r in S if r.get("status") != "ok"), None)
    if first_bad is None:
        print("W2 the first size that does not fit ends with an explicit error: every size completed ->",
              "REFUTED" if (mx or 0) >= 5.0 else "AMBIGUOUS")
    else:
        explicit = first_bad["status"] in ("error", "crash")
        print(f"W2 the first size that does not fit (n={first_bad['n']}) ends with an explicit error, not a timeout:"
              f" status {first_bad['status']} ->", v(explicit, not explicit))
        print("   message:", (first_bad.get("error") or first_bad.get("stderr") or "")[:300].replace("\n", " "))
    largest = max(ok_ns) if ok_ns else None
    print(f"W3 largest completed size is 29 or 30: {largest} ->", v(largest in (29, 30), largest not in (29, 30)))
    print("\nReported without prediction (RunPod pod):")
    for r in S:
        print("  ", json.dumps({k: r.get(k) for k in ("n", "status", "call_s", "base_mib", "peak_mib", "wall_s", "returncode")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score", "child"])
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    if a.mode == "child":
        child(a.device, a.n)
    elif a.mode == "run":
        run(a.device, [int(s) for s in a.sizes.split(",")])
    else:
        score()
