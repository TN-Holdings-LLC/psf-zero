"""pl_heavyhex_100k.py -- pre-registered (workplace design, 2026-09-29): the PennyLane
compounding loop of pl_heavyhex_gpu.py run for 100,000 laps on a RunPod RTX 4090, and a
500-lap short version for home. No IBM account, no network access to IBM, no QPU.

Reuses pl_heavyhex_gpu.py (same folder; locked earlier today) for the device, the circuit
family T, the conversions and the checks. Device FakeAuckland (27 qubits, heavy-hex).

Parts (one process each; on the pod the three run in parallel):
  A  candidate stack (core 2026-09-29.1 + psf_smart_layout 2026-09-29.c1), spare 0 (27 wires)
  B  candidate stack, spare 4 (23 wires)
  C  release stack (core 2026-09-28.1 + psf_smart_layout 2026-09-26.m1), spare 4
Each lap: tape -> Qiskit -> compile_for_hardware (timed; the synthesis cache is cleared
first, as in the release gate v4/v5) -> back to logical qubits -> tape (the next lap's
input). Every lap: compile time, two-qubit count, core fallbacks (from psf_compile's
warning), block distance to lap 0 (qml.matrix). At checkpoint laps: a whole-circuit check
of that lap's compiled output on the GPU against lap 0 (as in pl_heavyhex_gpu.py), and
the process RSS. A lap whose output cannot be mapped back (SWAPs) reuses its input.

    PYTHONPATH=<core> python -u pl_heavyhex_100k.py run --part A --layout-dir <dir> --repo <repo>
    python -u pl_heavyhex_100k.py score [--short]
Short version (home): run --part A --laps 500 --checkpoints 1,10,100,500 --tag short
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import re
import statistics as st
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

PARTS = {"A": ("PN", 0), "B": ("PN", 4), "C": ("P", 4)}
LAPS = 100_000
CHECKPOINTS = (1, 10, 100, 1_000, 5_000) + tuple(range(10_000, 100_001, 5_000))
FELL = re.compile(r"(\d+) block\(s\) fell back to CX-basis synthesis \((\d+) degenerate/numeric, (\d+) unexpected\)")


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def run(args):
    import pl_heavyhex_gpu as g
    arm, spare = PARTS[args.part]
    if args.layout_dir:
        sys.path.insert(0, args.layout_dir)
    for p in (args.repo, os.path.join(args.repo, "benchmarks")):
        if p not in sys.path:
            sys.path.insert(1 if args.layout_dir else 0, p)
    import pennylane as qml
    import qiskit
    import psf_compile as pc
    import psf_smart_layout as sl
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    core_v, layout_v = getattr(pc, "CORE_VERSION", None), sl.LAYOUT_VERSION
    print(f"part {args.part} arm {arm} spare {spare} laps {args.laps} | CORE_VERSION {core_v} | LAYOUT {layout_v} "
          f"{g.norm_sha(sl.__file__)[:12]} | psf_compile {pc.VERSION} {g.norm_sha(pc.__file__)[:12]}")
    print(f"python {sys.version.split()[0]} | qiskit {qiskit.__version__} | pennylane {qml.__version__} | "
          f"SCRIPT {os.path.basename(__file__)} {g.norm_sha(os.path.abspath(__file__))} | "
          f"pl_heavyhex_gpu.py {g.norm_sha(g.__file__)}")
    target, label = g.load_target(args)
    nq, m, factor = g.graph_facts(target)
    print(f"Stage 0: {label}, {nq} qubits, max matching {m}, factor {'found' if factor else 'NOT FOUND'}")
    if not factor:
        print("G0 FAILED")
        return
    cmap = target.build_coupling_map()
    native = [x for x in target.operation_names if x in g.NATIVE]
    dev = qml.device(args.device, wires=nq)
    blocks = g.layout_blocks(spare, nq, m)
    n = sum(len(b) for b in blocks)
    swapfree = g.swapfree_twoq(blocks)
    tape = g.initial_tape(blocks, seed=1000 * spare)
    u0 = g.block_matrices(tape, blocks)
    ident = {q: q for q in range(n)}
    ref = g.run_expvals(dev, tape.operations, g.observables(blocks, ident))
    ops_k = list(tape.operations)
    ops_k.insert(1, qml.RX(1e-6, wires=blocks[0][0]))
    c1 = float(np.max(np.abs(g.run_expvals(dev, ops_k, g.observables(blocks, ident)) - ref)))
    print(f"device {dev.name}; {n} wires; C1 RX(1e-6) control changes the values by {c1:.2e}", flush=True)
    checkpoints = set(int(x) for x in args.checkpoints.split(",")) if args.checkpoints else set(CHECKPOINTS)
    out = f"pl_100k_{args.part}_{args.tag}.csv"
    fields = ["part", "arm", "spare", "lap", "compile_s", "lap_s", "twoq", "swapfree_twoq", "mapped", "fallbacks",
              "block_distance", "whole_circuit", "gpu_s", "rss_mb", "wall_s"]
    f = open(out, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    t_start = time.time()
    for lap in range(1, args.laps + 1):
        t_lap = time.perf_counter()
        qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
        pc._CX_CORE_CACHE.clear()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            with contextlib.redirect_stdout(io.StringIO()):
                t0 = time.perf_counter()
                routed = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native, entangling_basis="cx",
                                                 layout_search=True, on_unsupported="raise", seed_transpiler=0)
                compile_s = time.perf_counter() - t0
        fb = 0
        for wm in ws:
            mm = FELL.search(str(wm.message))
            if mm:
                fb = int(mm.group(1))
        twoq = sum(1 for i in routed.data if len(i.qubits) == 2 and i.operation.name != "barrier")
        row = dict(part=args.part, arm=arm, spare=spare, lap=lap, compile_s=f"{compile_s:.6f}", twoq=twoq,
                   swapfree_twoq=swapfree, fallbacks=fb)
        if lap in checkpoints:
            final = list(routed.layout.final_index_layout(filter_ancillas=True))
            t0 = time.perf_counter()
            vals = g.run_expvals(dev, g.wrap_to_tape(routed).operations, g.observables(blocks, dict(enumerate(final))))
            row.update(whole_circuit=f"{float(np.max(np.abs(vals - ref))):.6e}", gpu_s=f"{time.perf_counter() - t0:.3f}",
                       rss_mb=f"{rss_mb():.1f}", wall_s=f"{time.time() - t_start:.0f}")
        elif lap % 1000 == 0:
            row.update(rss_mb=f"{rss_mb():.1f}", wall_s=f"{time.time() - t_start:.0f}")
        try:
            logical = g.back_to_logical(routed, n, blocks)
            new_tape = g.wrap_to_tape(logical)
            mats = g.block_matrices(new_tape, blocks)
            row["block_distance"] = f"{max(g.aligned(u0[k], mats[k]) for k in u0):.6e}"
            row["mapped"] = 1
            tape = new_tape
        except g.SwapLap:
            row["mapped"] = 0
        row["lap_s"] = f"{time.perf_counter() - t_lap:.6f}"
        w.writerow(row)
        if lap in checkpoints or lap % 10_000 == 0:
            f.flush()
            print(f"{args.part} lap {lap:6d} compile {compile_s * 1000:7.2f} ms 2q={twoq} mapped={row['mapped']} "
                  f"fallbacks={fb} block={row.get('block_distance', '-')} whole={row.get('whole_circuit', '-')} "
                  f"rss={row.get('rss_mb', '-')} wall={time.time() - t_start:.0f}s", flush=True)
    f.close()
    print(f"part {args.part} done: {args.laps} laps in {time.time() - t_start:.0f} s; C1 {c1:.3e}; device {dev.name}; "
          f"versions {core_v} {layout_v}\nDONE")


def load(tag, part):
    with open(f"pl_100k_{part}_{tag}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["lap"], r["twoq"], r["swapfree_twoq"] = int(r["lap"]), int(r["twoq"]), int(r["swapfree_twoq"])
        r["mapped"], r["fallbacks"], r["compile_s"] = int(r["mapped"]), int(r["fallbacks"]), float(r["compile_s"])
        for k in ("block_distance", "whole_circuit", "rss_mb", "gpu_s"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
    return rows


def header(tag, part):
    with open(f"pl_100k_{part}_{tag}.txt", encoding="utf-8") as f:
        text = f.read()
    return text


def v(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def slope_1000(rows):
    d = [(r["lap"], r["block_distance"]) for r in rows if r["block_distance"] is not None and r["lap"] <= 1000]
    x, y = np.array([a for a, _ in d], float), np.array([b for _, b in d])
    k, c = np.polyfit(x, y, 1)
    return k, c


def score(args):
    print("=" * 78)
    print("SCORING (thresholds exactly as pre-registered)")
    print("=" * 78)
    if args.short:
        return score_short(args)
    R = {p: load(args.tag, p) for p in PARTS}
    logs = {p: header(args.tag, p) for p in PARTS}
    full = all(len(R[p]) == LAPS for p in PARTS)
    ok_v = ("CORE_VERSION 2026-09-29.1 | LAYOUT 2026-09-29.c1" in logs["A"] and
            "CORE_VERSION 2026-09-29.1 | LAYOUT 2026-09-29.c1" in logs["B"] and
            "CORE_VERSION 2026-09-28.1 | LAYOUT 2026-09-26.m1" in logs["C"])
    dev = all("device lightning.gpu" in logs[p] for p in PARTS)
    c1 = [float(re.search(r"changes the values by ([0-9.e+-]+)", logs[p]).group(1)) for p in PARTS]
    print(f"C0 all parts 100,000 laps: {full}; versions as expected: {ok_v}; device lightning.gpu: {dev}; "
          f"C1 control smallest {min(c1):.2e} (>= 1e-9) -> {'passed' if full and ok_v and dev and min(c1) >= 1e-9 else 'FAILED'}")
    for p in PARTS:
        r = R[p]
        t = [x["compile_s"] for x in r]
        tenth = len(t) // 10
        bd = [x for x in r if x["block_distance"] is not None]
        wc = [x["whole_circuit"] for x in r if x["whole_circuit"] is not None]
        rs = [(x["lap"], x["rss_mb"]) for x in r if x["rss_mb"] is not None]
        print(f"   {p} ({PARTS[p][0]}, spare {PARTS[p][1]}): compile median {st.median(t) * 1000:.2f} ms, p99 "
              f"{np.percentile(t, 99) * 1000:.2f} ms, max {max(t) * 1000:.0f} ms; within 1 s {sum(x <= 1.0 for x in t)}; "
              f"mapped {sum(x['mapped'] for x in r)}; 2q {sorted({x['twoq'] for x in r})} (swap-free "
              f"{r[0]['swapfree_twoq']}); fallbacks {sum(x['fallbacks'] for x in r)} in "
              f"{sum(1 for x in r if x['fallbacks'])} laps; block distance lap 1 / 1,000 / last "
              f"{bd[0]['block_distance']:.2e} / {next((x['block_distance'] for x in bd if x['lap'] >= 1000), float('nan')):.2e} / "
              f"{bd[-1]['block_distance']:.2e}; whole-circuit max {max(wc):.2e}; RSS {rs[0][1]:.0f} -> {rs[-1][1]:.0f} MB; "
              f"first/last 10% median {st.median(t[:tenth]) * 1000:.2f} / {st.median(t[-tenth:]) * 1000:.2f} ms")
    A = R["A"]
    n1 = sum(x["compile_s"] <= 1.0 for x in A)
    print(f"H1 A within 1 s in {n1}/100000 (confirmed >= 99990, refuted < 99000) -> {v(n1 >= 99990, n1 < 99000)}")
    mp = sum(x["mapped"] and x["twoq"] == x["swapfree_twoq"] for x in A)
    print(f"H2 A swap-free and mapped back in {mp}/100000 (confirmed 100000, refuted < 99900) -> "
          f"{v(mp == 100000, mp < 99900)}")
    fab = sum(x["fallbacks"] for x in R["A"] + R["B"])
    print(f"H3 candidate core fallbacks in A + B: {fab} (confirmed 0, refuted >= 3) -> {v(fab == 0, fab >= 3)}")
    k, c = slope_1000(A)
    last = [x for x in A if x["block_distance"] is not None][-1]
    pred = c + k * last["lap"]
    ratio = last["block_distance"] / pred if pred > 0 else float("inf")
    print(f"H4 A drift does not accelerate: laps 1-1,000 slope {k:.3e}/lap, extrapolated to lap {last['lap']} "
          f"{pred:.3e}, measured {last['block_distance']:.3e}, ratio {ratio:.2f} (confirmed <= 2, refuted > 10) -> "
          f"{v(ratio <= 2, ratio > 10)}")
    wmax = max(x["whole_circuit"] for p in ("A", "B") for x in R[p] if x["whole_circuit"] is not None)
    print(f"H5 whole-circuit GPU check at every checkpoint of A and B: max {wmax:.2e} (confirmed <= 1e-8, "
          f"refuted > 1e-6) -> {v(wmax <= 1e-8, wmax > 1e-6)}")
    grow = []
    for p in PARTS:
        rs = [(x["lap"], x["rss_mb"]) for x in R[p] if x["rss_mb"] is not None]
        base = next((r for lap, r in rs if lap >= 1000), rs[0][1])
        grow.append(rs[-1][1] - base)
    print(f"H6 RSS growth from lap 1,000 to the end, per part: {[round(x) for x in grow]} MB (confirmed all <= 100, "
          f"refuted any > 500) -> {v(max(grow) <= 100, max(grow) > 500)}")
    bB = [x for x in R["B"] if x["block_distance"] is not None][-1]["block_distance"]
    bC = [x for x in R["C"] if x["block_distance"] is not None][-1]["block_distance"]
    rr = bB / bC
    print(f"H7 spare 4, candidate / release block distance at the end: {bB:.3e} / {bC:.3e} = {rr:.2f} "
          f"(confirmed 0.5-2, refuted < 0.2 or > 5) -> {v(0.5 <= rr <= 2, rr < 0.2 or rr > 5)}")
    t = [x["compile_s"] for x in A]
    tenth = len(t) // 10
    q = st.median(t[-tenth:]) / st.median(t[:tenth])
    print(f"H8 A timing stable: last-10% / first-10% median {q:.2f} (confirmed <= 1.5, refuted > 3) -> "
          f"{v(q <= 1.5, q > 3)}")
    fc = sum(x["fallbacks"] for x in R["C"])
    print(f"Reported without prediction: release core fallbacks in C {fc} "
          f"({sum(1 for x in R['C'] if x['fallbacks'])} laps); mapped in C {sum(x['mapped'] for x in R['C'])}")


def score_short(args):
    r = load(args.tag, "A")
    log = header(args.tag, "A")
    n = len(r)
    ok0 = (n == 500 and "CORE_VERSION 2026-09-29.1 | LAYOUT 2026-09-29.c1" in log and "device lightning.gpu" in log)
    c1 = float(re.search(r"changes the values by ([0-9.e+-]+)", log).group(1))
    wall = float(re.search(r"500 laps in (\d+) s", log).group(1))
    print(f"S0 500 laps, candidate versions, lightning.gpu: {ok0}; C1 {c1:.2e} (>= 1e-9) -> "
          f"{'passed' if ok0 and c1 >= 1e-9 else 'FAILED'}")
    t = [x["compile_s"] for x in r]
    n1 = sum(x <= 1.0 for x in t)
    print(f"S1 within 1 s in {n1}/500 (confirmed 500, refuted <= 450) -> {v(n1 == 500, n1 <= 450)}")
    mp = sum(x["mapped"] and x["twoq"] == x["swapfree_twoq"] for x in r)
    print(f"S2 swap-free and mapped back in {mp}/500 (confirmed 500, refuted <= 450) -> {v(mp == 500, mp <= 450)}")
    fb = sum(x["fallbacks"] for x in r)
    print(f"S3 core fallbacks {fb} (confirmed 0, refuted >= 3) -> {v(fb == 0, fb >= 3)}")
    b = [x for x in r if x["block_distance"] is not None][-1]["block_distance"]
    wc = max(x["whole_circuit"] for x in r if x["whole_circuit"] is not None)
    print(f"S4 lap-500 block distance {b:.2e} (confirmed <= 1e-11, refuted > 1e-9) and whole-circuit max {wc:.2e} "
          f"(confirmed <= 1e-10, refuted > 1e-8) -> {v(b <= 1e-11 and wc <= 1e-10, b > 1e-9 or wc > 1e-8)}")
    print(f"S5 whole run {wall:.0f} s (confirmed <= 120, refuted > 600) -> {v(wall <= 120, wall > 600)}")
    if args.pod_csv:
        with open(args.pod_csv, encoding="utf-8") as f:
            pod = {int(x["lap"]): x for x in csv.DictReader(f) if int(x["lap"]) <= 500}
        pb = float(pod[max(pod)]["block_distance"])  # lap 500 in the real comparison
        rr = b / pb
        print(f"S6 lap-500 block distance, this machine / pod: {b:.3e} / {pb:.3e} = {rr:.2f} (confirmed 0.5-2, "
              f"refuted < 0.1 or > 10) -> {v(0.5 <= rr <= 2, rr < 0.1 or rr > 10)}")
    print(f"Reported without prediction: compile median {st.median(t) * 1000:.2f} ms, max {max(t) * 1000:.0f} ms; "
          f"GPU checks {[x['gpu_s'] for x in r if x['gpu_s'] is not None]} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--part", choices=tuple(PARTS))
    ap.add_argument("--layout-dir", default=None)
    ap.add_argument("--repo", default=os.path.expanduser("~/psf-zero"))
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--fake", default="FakeAuckland")
    ap.add_argument("--generic-heavy-hex", type=int, default=0)
    ap.add_argument("--laps", type=int, default=LAPS)
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--tag", default="2026-09-29")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--pod-csv", default=None, help="short score only: the pod's part A CSV, for S6")
    args = ap.parse_args()
    if args.mode == "run" and not args.part:
        ap.error("run needs --part")
    run(args) if args.mode == "run" else score(args)


if __name__ == "__main__":
    main()
