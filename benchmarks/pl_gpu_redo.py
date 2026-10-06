"""pl_gpu_redo.py -- PL-GPU-REDO (home, 2026-10-06): the PennyLane loop of Addenda 256-257 (pl_heavyhex_gpu.py,
2026-09-29, RunPod RTX 4090) repeated with the current release on the owner's own GPU, with the release's
recommended call and the AI front end as new arms. Every lap's compiled circuit is checked as a WHOLE circuit on
lightning.gpu, so outputs with SWAPs are checked too.

Unchanged from Addendum 256 (the helpers are imported from the locked pl_heavyhex_gpu.py): FakeAuckland (27 qubits,
heavy-hex); circuit family T at spares 0 and 4; initial tape seed 1000 * spare; 30 laps; 1 s per compile; the C0
check (a 3-block sub-circuit on the GPU device against lightning.qubit) and the C1 control (an extra RX(1e-6) must be
seen); the whole-circuit check (<Z>, <X> of every logical qubit and <ZZ> of every block edge, read at each logical
qubit's final physical position, against the lap-0 tape); mapping back and the block-by-block check when swap-free.

Arms:
  R    release, the call of Addendum 256's arms P and PN (no target)
  RR   release, the recommended call as in the README (target, placement_refine, select, compare_level3,
       compare_floor, hybrid)
  A12  the adopted AI front end, compile_for_model_circuit(qc, coupling_map, basis, target=target)
  Q3   transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
A lap whose compile raises is recorded as "error: ..." and the next lap reuses the same tape.

Run from the repository root (it puts the root and benchmarks/ on the path):
    python benchmarks/pl_gpu_redo.py run --arm R --out DIR [--laps N --smoke]
    python benchmarks/pl_gpu_redo.py score --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
import warnings

import numpy as np

REPO = os.getcwd()
for _p in (os.path.join(REPO, "benchmarks"), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARMS = ("R", "RR", "A12", "Q3")
SPARES = (0, 4)
LAPS = 30
DEADLINE = 1.0
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")  # as in pl_heavyhex_gpu.py
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
EXPECTED = dict(release="2026-10-06.2", core="2026-09-29.1", layout="2026-10-01.1", a12="2026-10-06.a12")


def nsha(path):
    """Normalized SHA-256: CRLF -> LF, each line right-stripped, trailing blank lines dropped, no final newline."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return f"git error: {exc!r}"


def digest(circ):
    h = hashlib.sha256()
    for i in circ.data:
        h.update(repr((i.operation.name, [circ.find_bit(q).index for q in i.qubits],
                       [repr(p) for p in i.operation.params])).encode())
    lay = circ.layout
    if lay is not None:
        h.update(repr((list(lay.initial_index_layout(filter_ancillas=True)),
                       list(lay.final_index_layout(filter_ancillas=True)))).encode())
    return h.hexdigest()[:16]


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return None


def run(args):
    import pennylane as qml
    import qiskit
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider
    import pl_heavyhex_gpu as G
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    import psf_compile as pc
    import psf_smart_layout as sl
    import psf_ai_compile as ai
    import test_release_2026_10_05_1 as t17057

    here = os.path.abspath(__file__)
    meta = dict(
        arm=args.arm, smoke=bool(args.smoke), laps=args.laps, spares=list(SPARES), fake=args.fake,
        device=args.device,
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        git_head=git("rev-parse", "--short", "HEAD"), git_dirty=git("status", "--short", "--untracked-files=no"),
        versions=dict(release=pc.VERSION, core=pc.CORE_VERSION, layout=sl.LAYOUT_VERSION, a12=ai.AI_COMPILE_VERSION,
                      qiskit=qiskit.__version__, pennylane=qml.__version__, numpy=np.__version__,
                      lightning=pkg_version("pennylane_lightning"),
                      lightning_gpu=pkg_version("pennylane_lightning_gpu"), python=platform.python_version()),
        sha=dict(script=nsha(here), gpu_helpers=nsha(G.__file__), release=nsha(pc.__file__), layout=nsha(sl.__file__),
                 a12=nsha(ai.__file__)),
        platform=platform.platform(), cpus=os.cpu_count(),
        qiskit_17057_present=bool(t17057.qiskit_17057_present()))
    try:
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        meta["gpu"] = f"nvidia-smi error: {exc!r}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = getattr(fake_provider, args.fake)().target
    nq, m, factor = G.graph_facts(target)
    if not factor:
        raise SystemExit("G0 FAILED: no {P2,P3}-factor; nothing is run.")
    fail_e, fail_q = pc._failed_elements(target, 0.5)
    meta["failed_elements"] = dict(edges=sorted(list(e) for e in fail_e), qubits=sorted(fail_q))
    print(json.dumps(meta), flush=True)
    cmap = target.build_coupling_map()
    native = [g for g in target.operation_names if g in NATIVE]
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in target.operation_names]
    dev = qml.device(args.device, wires=nq)

    def compile_arm(qc):
        if args.arm == "Q3":
            return transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.arm == "R":
                return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native, entangling_basis="cx",
                                               layout_search=True, on_unsupported="raise", seed_transpiler=0)
            if args.arm == "RR":
                return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=basis, entangling_basis="cx",
                                               layout_search=True, target=target, seed_transpiler=0, **RECOMMENDED)
            return ai.compile_for_model_circuit(qc, cmap, basis, target=target)

    rows, checks = [], {}
    for spare in SPARES:
        blocks = G.layout_blocks(spare, nq, m)
        n = sum(len(b) for b in blocks)
        tape = G.initial_tape(blocks, seed=1000 * spare)
        u0 = G.block_matrices(tape, blocks)
        ident = {q: q for q in range(n)}
        ref = G.run_expvals(dev, tape.operations, G.observables(blocks, ident))
        sub_blocks = blocks[:3]
        nsub = sum(len(b) for b in sub_blocks)
        sub = [op for op in tape.operations if max(int(w) for w in op.wires) < nsub]
        d_c0 = float(np.max(np.abs(
            G.run_expvals(qml.device(args.device, wires=nsub), sub, G.observables(sub_blocks, ident))
            - G.run_expvals(qml.device("lightning.qubit", wires=nsub), sub, G.observables(sub_blocks, ident)))))
        ops_k = list(tape.operations)
        ops_k.insert(1, qml.RX(1e-6, wires=blocks[0][0]))
        d_c1 = float(np.max(np.abs(G.run_expvals(dev, ops_k, G.observables(blocks, ident)) - ref)))
        checks[str(spare)] = dict(c0_gpu_vs_cpu=d_c0, c1_control=d_c1, n=n, values=len(ref))
        print(f"spare={spare} n={n} reference {len(ref)} values | C0 GPU vs CPU {d_c0:.2e} | C1 control {d_c1:.2e}",
              flush=True)
        prev_layout = None
        for lap in range(1, args.laps + 1):
            t_lap = time.perf_counter()
            qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
            t0 = time.perf_counter()
            try:
                routed, err = compile_arm(qc), None
            except Exception as e:  # noqa: BLE001 - recorded per lap
                routed, err = None, f"error: {type(e).__name__}: {e}"[:120]
            compile_s = time.perf_counter() - t0
            twoq = dig = used = whole = bdist = layout = None
            gpu_s = 0.0
            if routed is None:
                status = err
            else:
                twoq = sum(1 for i in routed.data if len(i.qubits) == 2 and i.operation.name != "barrier")
                dig = digest(routed)
                used = bool(pc._uses_failed(routed, fail_e, fail_q))
                final = list(routed.layout.final_index_layout(filter_ancillas=True))
                t0 = time.perf_counter()
                vals = G.run_expvals(dev, G.wrap_to_tape(routed).operations, G.observables(blocks, dict(enumerate(final))))
                gpu_s = time.perf_counter() - t0
                whole = float(np.max(np.abs(vals - ref)))
                try:
                    logical = G.back_to_logical(routed, n, blocks)
                    new_tape = G.wrap_to_tape(logical)
                    mats = G.block_matrices(new_tape, blocks)
                    bdist = max(G.aligned(u0[k], mats[k]) for k in u0)
                    status, tape, layout = "OK", new_tape, tuple(final)
                except G.SwapLap as e:
                    status = f"swap: {e}"[:80]
            row = dict(spare=spare, logical_qubits=n, lap=lap, status=status, compile_s=compile_s, gpu_check_s=gpu_s,
                       lap_s=time.perf_counter() - t_lap, twoq=twoq, swapfree_twoq=G.swapfree_twoq(blocks),
                       whole_circuit_max_diff=whole, block_distance=bdist, digest=dig, uses_failed=used,
                       layout_changed=(prev_layout is not None and layout is not None and layout != prev_layout))
            rows.append(row)
            if layout is not None:
                prev_layout = layout
            w_txt = "-" if whole is None else f"{whole:.3e}"
            b_txt = "-" if bdist is None else f"{bdist:.3e}"
            print(f"{args.arm:3s} spare={spare} lap {lap:3d} compile {compile_s:7.3f}s gpu {gpu_s:6.2f}s 2q={twoq} "
                  f"(swap-free {row['swapfree_twoq']}) whole={w_txt} block={b_txt} failed-element={used} {status}",
                  flush=True)
    meta["checks"] = checks
    meta["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"pl_gpu_redo_{args.arm}.json")
    with open(path, "w", encoding="utf-8") as fo:
        json.dump(dict(meta=meta, rows=rows), fo)
    print(f"Wrote {path} ({len(rows)} rows)\nDONE", flush=True)


def score(args):
    D = {}
    for a in ARMS:
        with open(os.path.join(args.out, f"pl_gpu_redo_{a}.json"), encoding="utf-8") as fi:
            D[a] = json.load(fi)
    n = LAPS
    heads = {D[a]["meta"]["git_head"] for a in ARMS}
    vers_ok = all(D[a]["meta"]["versions"][k] == v for a in ARMS for k, v in EXPECTED.items())
    full = all(sum(1 for r in D[a]["rows"] if r["spare"] == s) == n for a in ARMS for s in SPARES)
    clean = all(not D[a]["meta"]["smoke"] and not D[a]["meta"]["git_dirty"] for a in ARMS)
    gpu = all(D[a]["meta"]["device"] == "lightning.gpu" for a in ARMS)
    c0 = max(c["c0_gpu_vs_cpu"] for a in ARMS for c in D[a]["meta"]["checks"].values())
    c1 = min(c["c1_control"] for a in ARMS for c in D[a]["meta"]["checks"].values())
    p0 = full and vers_ok and clean and gpu and len(heads) == 1 and c0 <= 1e-12 and c1 >= 1e-9

    def cell(a, s):
        return [r for r in D[a]["rows"] if r["spare"] == s]

    def err(r):
        return r["status"].startswith("error")

    def within(c):
        return sum(r["compile_s"] <= DEADLINE and not err(r) for r in c)

    def back(c):
        return sum(r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"] for r in c)

    def whole(*arms):
        return [r["whole_circuit_max_diff"] for a in arms for r in D[a]["rows"] if r["whole_circuit_max_diff"] is not None]

    def V(good, bad):
        return "REFUTED" if bad else ("CONFIRMED" if good else "AMBIGUOUS")

    L = ["# PL-GPU-REDO score (thresholds as pre-registered in Addendum 375)", "",
         f"- P0 (validity): {'PASS' if p0 else 'FAIL'} -- {len(ARMS)} arms x 2 spares x {n} laps {full}, versions "
         f"{vers_ok}, not smoke and clean tree {clean}, lightning.gpu {gpu}, one git_head {sorted(heads)}, "
         f"C0 GPU vs CPU max {c0:.2e} (<= 1e-12), C1 control min {c1:.2e} (>= 1e-9)", ""]
    L.append("| spare | arm | compile median s | max s | within 1 s | errors | swap-free and mapped back | 2q | "
             "uses a failed element | whole-circuit max | block distance worst | GPU check median s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in SPARES:
        for a in ARMS:
            c = cell(a, s)
            w = [r["whole_circuit_max_diff"] for r in c if r["whole_circuit_max_diff"] is not None]
            bd = [r["block_distance"] for r in c if r["block_distance"] is not None]
            L.append(f"| {s} | {a} | {st.median(r['compile_s'] for r in c):.3f} | {max(r['compile_s'] for r in c):.3f} | "
                     f"{within(c)}/{len(c)} | {sum(map(err, c))} | {back(c)}/{len(c)} | "
                     f"{sorted({r['twoq'] for r in c if r['twoq'] is not None})} | "
                     f"{sum(bool(r['uses_failed']) for r in c)} | {(f'{max(w):.2e}' if w else '-')} | "
                     f"{(f'{max(bd):.2e}' if bd else '-')} | {st.median(r['gpu_check_s'] for r in c):.2f} |")
    L.append("")
    if not p0:
        L.append("P0 failed: nothing below is scored.")
    else:
        w = within(cell("R", 0))
        L.append(f"- G1 (R, spare 0, within 1 s in {w}/{n}; confirmed {n}/{n}, refuted <= {n // 2}): "
                 f"**{V(w == n, w <= n // 2)}**")
        b = back(cell("R", 0)) + back(cell("R", 4))
        L.append(f"- G2 (R, swap-free and mapped back in {b}/{2 * n}; confirmed {2 * n}/{2 * n}, refuted <= {n}): "
                 f"**{V(b == 2 * n, b <= n)}**")
        wr = whole("R")
        mr = max(wr) if wr else float("nan")
        L.append(f"- G3 (R correct as a whole circuit on every lap, max {mr:.2e}; confirmed <= 1e-12 with every "
                 f"lap checked, refuted > 1e-10): **{V(len(wr) == 2 * n and mr <= 1e-12, bool(wr) and mr > 1e-10)}**")
        e = sum(map(err, [r for a in ARMS for r in D[a]["rows"]]))
        L.append(f"- G4 (no arm raises: {e} of {len(ARMS) * 2 * n} laps; confirmed 0, refuted >= 1): **{V(e == 0, e >= 1)}**")
        wt = whole("RR", "A12", "Q3")
        mt = max(wt) if wt else float("nan")
        L.append(f"- G5 (RR, A12 and Q3 correct as whole circuits on every lap, SWAP outputs included, max "
                 f"{mt:.2e}; confirmed <= 1e-10 with every lap checked, refuted > 1e-6): "
                 f"**{V(len(wt) == 3 * 2 * n and mt <= 1e-10, bool(wt) and mt > 1e-6)}**")
        w = within(cell("Q3", 0))
        L.append(f"- G6 (Q3, spare 0, within 1 s in {w}/{n}; confirmed 0, refuted >= {n // 2}): "
                 f"**{V(w == 0, w >= n // 2)}**")
        w1, w2 = within(cell("RR", 0)), within(cell("A12", 0))
        L.append(f"- G7 (RR and A12, spare 0, within 1 s in {w1}/{n} and {w2}/{n}; confirmed both <= 2, refuted "
                 f"either >= {n // 2}): **{V(max(w1, w2) <= 2, max(w1, w2) >= n // 2)}**")
        w1, w2, w3 = within(cell("RR", 4)), within(cell("A12", 4)), within(cell("Q3", 4))
        L.append(f"- G8 (spare 4: RR, A12 and Q3 within 1 s in {w1}/{n}, {w2}/{n} and {w3}/{n}; confirmed all "
                 f">= {n - 2}, refuted any <= {n // 2}): **{V(min(w1, w2, w3) >= n - 2, min(w1, w2, w3) <= n // 2)}**")
        r0 = cell("R", 0)
        s1 = sum(x["digest"] == y["digest"] for x, y in zip(r0, cell("RR", 0)) if y["digest"])
        s2 = sum(x["digest"] == y["digest"] for x, y in zip(r0, cell("A12", 0)) if y["digest"])
        L.append(f"- G9 (spare 0: RR's and A12's outputs are R's own circuit on {s1}/{n} and {s2}/{n} laps; confirmed "
                 f"both >= {n - 2}, refuted either <= {n // 2}): **{V(min(s1, s2) >= n - 2, min(s1, s2) <= n // 2)}**")
        L.append("")
        L.append("Reported without prediction: which outputs use a failed element; whether RR's and A12's outputs are R's "
                 "at spare 4 (digests); GPU-check times; against Addendum 257 (RunPod RTX 4090).")
        for s in SPARES:
            rr = cell("R", s)
            for a in ("RR", "A12"):
                same = sum(x["digest"] == y["digest"] for x, y in zip(rr, cell(a, s)) if y["digest"])
                L.append(f"- spare {s}: {a} output = R's on {same}/{n} laps")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8") as fo:
        fo.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", default="FakeAuckland")
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--laps", type=int, default=LAPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.mode == "run" and not args.arm:
        ap.error("run needs --arm")
    run(args) if args.mode == "run" else score(args)


if __name__ == "__main__":
    main()
