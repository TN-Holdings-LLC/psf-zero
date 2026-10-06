"""pl_redo.py -- PL-REDO (workplace, 2026-10-06): the timed PennyLane compounding loop of Addenda 254-255
(pl_heavyhex_chain.py, 2026-09-29) repeated with the current release, on the owner's own Linux machine (WSL2), with
new arms: the release's recommended call and the AI front end, each also with candidate psf_compile 2026-10-06.c16
(changelog item 43), which the smoke run of this script made necessary: on the spare-0 circuit the release's
recommended call raised a TranspilerError from item 31's recompile.

Same device, circuits, seeds, laps and deadline as Addendum 254: FakeKingston (156 qubits, heavy-hex), circuit family
T at spares 0 and 16, initial tape seed 1000 * spare, 20 laps, 1 s per compile. Each lap: PennyLane tape -> Qiskit
circuit -> compile (timed) -> map back to logical qubits -> PennyLane tape, the next lap's input. Each lap's meaning
is checked block by block against PennyLane's own matrices of the lap-0 tape. The helpers (initial tape, block
matrices, mapping back) are imported unchanged from pl_heavyhex_chain.py.

Arms:
  Q3   transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
  R    the release, the call of Addendum 254's arms P and PN (no target):
       compile_for_hardware(qc, coupling_map, basis_gates=<native>, entangling_basis="cx", layout_search=True,
                            on_unsupported="raise", seed_transpiler=0)
  RR   the release's recommended call as in the README (with the target):
       compile_for_hardware(qc, coupling_map, basis_gates=<cx/cz, rz, sx, x>, entangling_basis="cx",
                            layout_search=True, target=target, seed_transpiler=0, placement_refine=True,
                            final_resynthesis="select", compare_level3=True, compare_floor=True,
                            candidate_score="hybrid")
  A12  the adopted AI front end: psf_ai_compile.compile_for_model_circuit(qc, coupling_map, <same basis>,
                                                                          target=target)
  RRC  RR with candidate c16 (patches/psf_compile_c16_2026-10-06/psf_compile.py loaded as psf_compile)
  A12C A12 with candidate c16 underneath
A lap whose compile raises is recorded as "error: ..." and the next lap reuses the same input tape.

Run from the repository root (it puts the root and benchmarks/ on the path):
    python benchmarks/pl_redo.py run --arm R --out DIR [--laps N --smoke]
    python benchmarks/pl_redo.py score --out DIR
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
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

ARMS = ("R", "RR", "A12", "RRC", "A12C", "Q3")
CAND = os.path.join("patches", "psf_compile_c16_2026-10-06", "psf_compile.py")
SPARES = (0, 16)
LAPS = 20
DEADLINE = 1.0
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")  # as in pl_heavyhex_chain.py
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
EXPECTED = dict(release="2026-10-06.1", core="2026-09-29.1", layout="2026-10-01.1", a12="2026-10-06.a12")
EXPECTED_C16 = dict(EXPECTED, release="2026-10-06.c16")


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


def run(args):
    if args.arm in ("RRC", "A12C"):  # the candidate must be psf_compile before anything imports it
        # next to this script's own folder, so a kit run before the lock and the committed copy each load theirs
        cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CAND)
        spec = importlib.util.spec_from_file_location("psf_compile", cand)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["psf_compile"] = mod
        spec.loader.exec_module(mod)
    import pennylane as qml
    import qiskit
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider
    import full_heavyhex_cliff as fh
    import pl_heavyhex_chain as plc
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    import psf_compile as pc
    import psf_smart_layout as sl
    import psf_ai_compile as ai
    import test_release_2026_10_05_1 as t17057

    here = os.path.abspath(__file__)
    meta = dict(
        arm=args.arm, smoke=bool(args.smoke), laps=args.laps, spares=list(SPARES), fake=args.fake,
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        git_head=git("rev-parse", "--short", "HEAD"), git_dirty=git("status", "--short", "--untracked-files=no"),
        versions=dict(release=pc.VERSION, core=pc.CORE_VERSION, layout=sl.LAYOUT_VERSION, a12=ai.AI_COMPILE_VERSION,
                      qiskit=qiskit.__version__, pennylane=qml.__version__, numpy=np.__version__,
                      python=platform.python_version()),
        sha=dict(script=nsha(here), chain=nsha(plc.__file__), release=nsha(pc.__file__), layout=nsha(sl.__file__),
                 a12=nsha(ai.__file__)),
        script_path=here, platform=platform.platform(), cpus=os.cpu_count(),
        qiskit_17057_present=bool(t17057.qiskit_17057_present()))
    print(json.dumps(meta), flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = getattr(fake_provider, args.fake)().target
    f = fh.graph_facts(target)
    nq, m = f["nq"], f["matching"]
    if not f["factor"]:
        raise SystemExit("G0 FAILED: no {P2,P3}-factor; nothing is run.")
    fail_e, fail_q = pc._failed_elements(target, 0.5)  # item 31's failed elements at the default threshold
    meta["failed_elements"] = dict(edges=sorted(list(e) for e in fail_e), qubits=sorted(fail_q))
    print("failed elements", json.dumps(meta["failed_elements"]), flush=True)
    cmap = target.build_coupling_map()
    native = [g for g in target.operation_names if g in NATIVE]
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in target.operation_names]

    def compile_arm(qc):
        if args.arm == "Q3":
            return transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.arm == "R":
                return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native, entangling_basis="cx",
                                               layout_search=True, on_unsupported="raise", seed_transpiler=0)
            if args.arm in ("RR", "RRC"):
                return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=basis, entangling_basis="cx",
                                               layout_search=True, target=target, seed_transpiler=0, **RECOMMENDED)
            return ai.compile_for_model_circuit(qc, cmap, basis, target=target)

    rows = []
    for spare in SPARES:
        blocks = fh.layout_blocks("T", spare, nq, m)
        n = sum(len(b) for b in blocks)
        swapfree = fh.expected_twoq(blocks)
        tape = plc.initial_tape(blocks, n, seed=1000 * spare)
        u0 = plc.block_matrices(tape, blocks)
        prev_layout = None
        for lap in range(1, args.laps + 1):
            t_lap = time.perf_counter()
            qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
            t0 = time.perf_counter()
            try:
                routed = compile_arm(qc)
            except Exception as e:  # noqa: BLE001 - recorded per lap
                routed, err = None, f"error: {type(e).__name__}: {e}"[:120]
            compile_s = time.perf_counter() - t0
            twoq = dig = used = None
            if routed is None:
                status, dist, layout = err, None, None
            else:
                twoq = sum(1 for i in routed.data if len(i.qubits) == 2 and i.operation.name != "barrier")
                dig = digest(routed)
                used = bool(pc._uses_failed(routed, fail_e, fail_q))
                try:
                    logical, layout = plc.back_to_logical(routed, n, blocks)
                    new_tape = plc.to_tape(logical)
                    mats = plc.block_matrices(new_tape, blocks)
                    dist = max(plc.aligned(u0[k], mats[k]) for k in u0)
                    status, tape = "OK", new_tape
                except plc.SwapLap as e:
                    status, dist, layout = f"swap: {e}"[:80], None, None
            lap_s = time.perf_counter() - t_lap
            row = dict(spare=spare, logical_qubits=n, lap=lap, status=status, compile_s=compile_s, lap_s=lap_s,
                       twoq=twoq, swapfree_twoq=swapfree, max_block_distance=dist, digest=dig, uses_failed=used,
                       layout_changed=(prev_layout is not None and layout is not None and layout != prev_layout))
            rows.append(row)
            if layout is not None:
                prev_layout = layout
            d_txt = "-" if dist is None else f"{dist:.3e}"
            print(f"{args.arm:3s} spare={spare:2d} lap {lap:3d} compile {compile_s:8.3f}s lap {lap_s:7.2f}s "
                  f"2q={twoq} (swap-free {swapfree}) max_block_dist={d_txt} layout_changed={row['layout_changed']} "
                  f"{status}", flush=True)
    meta["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"pl_redo_{args.arm}.json")
    with open(path, "w", encoding="utf-8") as fo:
        json.dump(dict(meta=meta, rows=rows), fo)
    print(f"Wrote {path} ({len(rows)} rows)\nDONE", flush=True)


def score(args):
    D = {}
    for a in ARMS:
        with open(os.path.join(args.out, f"pl_redo_{a}.json"), encoding="utf-8") as fi:
            D[a] = json.load(fi)
    n = LAPS
    heads = {D[a]["meta"]["git_head"] for a in ARMS}
    vers_ok = all(D[a]["meta"]["versions"][k] == v for a in ARMS
                  for k, v in (EXPECTED_C16 if a in ("RRC", "A12C") else EXPECTED).items())
    full = all(sum(1 for r in D[a]["rows"] if r["spare"] == s) == n for a in ARMS for s in SPARES)
    clean = all(not D[a]["meta"]["smoke"] and not D[a]["meta"]["git_dirty"] for a in ARMS)
    p17 = all(D[a]["meta"]["qiskit_17057_present"] for a in ARMS)
    p0 = full and vers_ok and clean and p17 and len(heads) == 1

    def cell(a, s):
        return [r for r in D[a]["rows"] if r["spare"] == s]

    def within(c):
        return sum(r["compile_s"] <= DEADLINE and not r["status"].startswith("error") for r in c)

    def back(c):
        return sum(r["status"] == "OK" and r["twoq"] == r["swapfree_twoq"] for r in c)

    def errs(c):
        return sum(r["status"].startswith("error") for r in c)

    def dists(*arms):
        return [r["max_block_distance"] for a in arms for r in D[a]["rows"] if r["max_block_distance"] is not None]

    def V(good, bad):
        return "REFUTED" if bad else ("CONFIRMED" if good else "AMBIGUOUS")

    L = ["# PL-REDO score (thresholds as pre-registered in Addendum 372)", "",
         f"- P0 (validity): {'PASS' if p0 else 'FAIL'} -- {len(ARMS)} arms x 2 spares x {n} laps {full}, versions "
         f"{vers_ok}, not smoke and clean tree {clean}, #17057 present {p17}, git_head {sorted(heads)}", ""]
    L.append("| spare | arm | compile median s | max s | within 1 s | errors | swap-free and mapped back | 2q | "
             "uses a failed element | block distance lap 1 / worst |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in SPARES:
        for a in ARMS:
            c = cell(a, s)
            ds = [r["max_block_distance"] for r in c if r["max_block_distance"] is not None]
            L.append(f"| {s} | {a} | {st.median(r['compile_s'] for r in c):.3f} | {max(r['compile_s'] for r in c):.3f} | "
                     f"{within(c)}/{len(c)} | {errs(c)} | {back(c)}/{len(c)} | "
                     f"{sorted({r['twoq'] for r in c if r['twoq'] is not None})} | "
                     f"{sum(bool(r['uses_failed']) for r in c)} | "
                     + (f"{ds[0]:.2e} / {max(ds):.2e} |" if ds else "- |"))
    L.append("")
    if not p0:
        L.append("P0 failed: nothing below is scored.")
    else:
        w = within(cell("R", 0))
        L.append(f"- D1 (R, spare 0, within 1 s in {w}/{n}; confirmed {n}/{n}, refuted <= {n // 2}): "
                 f"**{V(w == n, w <= n // 2)}**")
        b = back(cell("R", 0)) + back(cell("R", 16))
        L.append(f"- D2 (R, swap-free and mapped back in {b}/{2 * n} laps; confirmed {2 * n}/{2 * n}, refuted <= {n}): "
                 f"**{V(b == 2 * n, b <= n)}**")
        dr = dists("R")
        mx = max(dr) if dr else None
        L.append(f"- D3 (R, worst block distance {mx}; confirmed <= 1e-12 with every lap checked, refuted > 1e-10): "
                 f"**{V(len(dr) == 2 * n and mx <= 1e-12, bool(dr) and mx > 1e-10)}**")
        w = within(cell("Q3", 0))
        L.append(f"- D4 (Q3, spare 0, within 1 s in {w}/{n}; confirmed 0, refuted >= {n // 2}): "
                 f"**{V(w == 0, w >= n // 2)}**")
        e1, e2 = errs(cell("RR", 0)), errs(cell("A12", 0))
        L.append(f"- D5 (release: RR and A12 raise at spare 0 in {e1}/{n} and {e2}/{n} laps; confirmed both {n}/{n}, "
                 f"refuted either <= {n // 2}): **{V(e1 == n and e2 == n, min(e1, e2) <= n // 2)}**")
        e = sum(errs(cell(a, s)) for a in ("RRC", "A12C") for s in SPARES)
        L.append(f"- D6 (c16: RRC and A12C raise in {e}/{4 * n} laps; confirmed 0, refuted >= 1): **{V(e == 0, e >= 1)}**")
        w1, w2 = within(cell("RRC", 0)), within(cell("A12C", 0))
        L.append(f"- D7 (RRC and A12C, spare 0, within 1 s in {w1}/{n} and {w2}/{n}; confirmed both <= 2, refuted "
                 f"either >= {n // 2}): **{V(max(w1, w2) <= 2, max(w1, w2) >= n // 2)}**")
        b1, b2 = back(cell("RRC", 0)), back(cell("A12C", 0))
        L.append(f"- D8 (RRC and A12C, spare 0, swap-free and mapped back in {b1}/{n} and {b2}/{n}; confirmed both "
                 f"{n}/{n}, refuted either <= {n // 2}): **{V(b1 == n and b2 == n, min(b1, b2) <= n // 2)}**")
        dd = dists("RRC", "A12C")
        mx = max(dd) if dd else None
        L.append(f"- D9 (RRC and A12C, worst block distance over checked laps {mx}; confirmed <= 1e-12, refuted "
                 f"> 1e-10): **{V(bool(dd) and mx <= 1e-12, bool(dd) and mx > 1e-10)}**")
        pairs = []  # laps with the same input: up to the release's first error in that spare
        for rel, cand in (("RR", "RRC"), ("A12", "A12C")):
            for s in SPARES:
                for x, y in zip(cell(rel, s), cell(cand, s)):
                    if x["status"].startswith("error"):
                        break
                    pairs.append((x, y))
        same = sum(x["digest"] == y["digest"] for x, y in pairs)
        L.append(f"- D10 (c16 = release wherever the release returns: identical outputs in {same}/{len(pairs)} laps with the same input; "
                 f"confirmed all and at least {n}, refuted any differs): "
                 f"**{V(same == len(pairs) and len(pairs) >= n, same < len(pairs))}**")
        w1, w2 = within(cell("RRC", 16)), within(cell("A12C", 16))
        L.append(f"- D11 (RRC and A12C, spare 16, within 1 s in {w1}/{n} and {w2}/{n}; confirmed both >= {n - 2}, "
                 f"refuted either <= {n // 2}): **{V(min(w1, w2) >= n - 2, min(w1, w2) <= n // 2)}**")
        L.append("")
        L.append("Reported without prediction: which outputs use a failed element (table); spare 16 times of RRC and A12C; median compile time against R; Q3's "
                 "worst block distance; R against Addendum 255's arm PN (0.071 s, 276 two-qubit gates at spare 0).")
        for s in SPARES:
            mr = st.median(r["compile_s"] for r in cell("R", s))
            L.append(f"- spare {s}: RRC/R {st.median(r['compile_s'] for r in cell('RRC', s)) / mr:.1f}x, "
                     f"A12C/R {st.median(r['compile_s'] for r in cell('A12C', s)) / mr:.1f}x, "
                     f"Q3/R {st.median(r['compile_s'] for r in cell('Q3', s)) / mr:.1f}x")
        dq = dists("Q3")
        L.append(f"- Q3 worst block distance {max(dq) if dq else None} over {len(dq)} checked laps")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "score.md"), "w", encoding="utf-8") as fo:
        fo.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", default="FakeKingston")
    ap.add_argument("--laps", type=int, default=LAPS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.mode == "run" and not args.arm:
        ap.error("run needs --arm")
    run(args) if args.mode == "run" else score(args)


if __name__ == "__main__":
    main()
