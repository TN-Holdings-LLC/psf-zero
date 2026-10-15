"""C58-VAL: prototype c58 (item 97: formula only inside the scope) and AI front end a19 (item 34: formula only for
ring classifiers), for reliability and speed. Offline; no real device.

Part A: C57-VAL's 28 tests on its 5 devices, without (T) and with (F) fractional gates, through c58's default call.
Compared with C57-VAL's rows (release 2026-10-14.1 = c57: outputs, durations, compile seconds), not recompiled.
Part B: W3's 64 classifier cells (benchmarks/a18_cells.py's), T and F, through a19 on c58. Compared with a18's rows
(data/2026-10-14/a18_cells/).

    python benchmarks/c58_val.py <out dir>   (refuses an existing out dir)"""
import contextlib
import io
import json
import math
import os
import runpy
import shutil
import sys
import time
import warnings

warnings.simplefilter("ignore")
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
_tmp = OUT + "_c55val_import"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c58_val_import")  # helpers only
shutil.rmtree(_tmp, ignore_errors=True)
os.makedirs(OUT)
C58_FILE = os.path.join("patches", "psf_compile_c58_2026-10-15", "psf_compile.py")
A19_FILE = os.path.join("patches", "psf_ai_compile_a19_2026-10-15", "psf_ai_compile.py")
C57_ROWS = os.path.join("data", "2026-10-14", "c57_val", "run", "c57_val_rows.jsonl")
A18_ROWS = os.path.join("data", "2026-10-14", "a18_cells")
M = g["load"](C58_FILE, "psf_compile")  # a19 imports psf_compile
M.WARN_WITHOUT_TARGET = False
REPEAT = ("qaoa_n20_p2", "tfim_n16", "ba_n20", "tfimP_n12", "hhI_n20")


def uses_failed(c, t):
    fe, fq = M._failed_elements(t, 0.5)
    return M._uses_failed(c, fe, fq)


def part_a(fh):
    rows = []
    for d in g["DEVICES"]:
        t, ft = g["device"](d)
        for name in g["TESTS"]:
            qc = g["build"](name)
            for lab, tg in (("T", t), ("F", ft)):
                before = dict(M.FORMULA97_STATS)
                s = time.perf_counter()
                err = None
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        c = M.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True,
                                                   seed_transpiler=0)
                except Exception as e:  # noqa: BLE001
                    c, err = None, f"{type(e).__name__}: {str(e)[:200]}"
                secs = time.perf_counter() - s
                path = next((k for k in ("chain", "graph", "not_built") if M.FORMULA97_STATS[k] > before[k]), "c57")
                r = dict(part="A", device=d, test=name, target=lab, secs=round(secs, 3), error=err, path=path)
                if c is not None:
                    r.update(sig=g["sig"](c), uses_failed=uses_failed(c, tg), exact=g["exactness"](qc, c),
                             **g["metrics"](c, tg))
                    if name in REPEAT:
                        with contextlib.redirect_stdout(io.StringIO()):
                            r["sig_repeat"] = g["sig"](M.compile_for_hardware(qc, target=tg, entangling_basis="cx",
                                                                               layout_search=True, seed_transpiler=0))
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"A {d[-10:]:10s} {name:16s} {lab} {path:9s} {secs*1000:9.1f} ms dur "
                      f"{r.get('dur_us', float('nan')):7.2f} exact {r.get('exact')}", flush=True)
    return rows


def part_b(fh):
    A = g["load"](A19_FILE, "a19")
    D = g["load"](os.path.join("data", "2026-10-05", "workplace", "depth1", "depth_eval.py"), "depth_eval")
    sys.path.insert(0, os.path.join("data", "2026-10-14", "workplace", "workplace_2026-10-14e_Q1-Q4", "q3", "eval"))
    from p1_common import W2, dur_2q, fractional, np, rzz_ok
    from qiskit_ibm_runtime import fake_provider
    rng = np.random.default_rng(20261014)
    angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6)
              for L in range(1, 17)}
    rows = []
    for tk in ("T", "F"):
        for dev in ("FakeFez", "FakeTorino"):
            t0 = getattr(fake_provider, dev)().target
            tg = fractional(t0) if tk == "F" else t0
            basis = [x for x in tg.operation_names if x in ("cz", "rz", "sx", "x")]
            for n in (4, 6):
                for L in range(1, 17):
                    x, th = angles[(n, L)]
                    qc = D.circuit(x, th, n, L)
                    qc.measure_all()
                    s = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()):
                        c, info = A.compile_for_model_circuit(qc, tg.build_coupling_map(), basis, target=tg,
                                                              return_info=True)
                    secs = time.perf_counter() - s
                    d, two, nr = dur_2q(c, tg)
                    r = dict(part="B", device=dev, n=n, L=L, target=tk, dur_us=round(d, 4), two_q=two, rzz=nr,
                             in_range=rzz_ok(c), tvd=W2.tvd(W2.probs(c), W2.probs(qc)), chosen=info.get("chosen"),
                             path=info.get("path"), secs=round(secs, 3), isa=M._isa_ok(c, tg),
                             uses_failed=uses_failed(c, tg))
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
    return rows


if __name__ == "__main__":
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c58_sha256=g["sha"](C58_FILE), a19_sha256=g["sha"](A19_FILE),
                c57_rows_sha256=g["sha"](C57_ROWS), script_sha256=g["sha"](os.path.abspath(__file__)))
    json.dump(head, open(os.path.join(OUT, "c58_val_head.json"), "w"), indent=1)
    fh = open(os.path.join(OUT, "c58_val_rows.jsonl"), "w")
    ra = part_a(fh)
    rb = part_b(fh)
    fh.close()
    old = {(r["device"], r["test"], r["arm"]): r for r in map(json.loads, open(C57_ROWS))}
    a18 = {}
    for tk in ("T", "F"):
        for l in open(os.path.join(A18_ROWS, f"a18_{tk}.jsonl")):
            r = json.loads(l)
            a18[(tk, r["device"], r["n"], r["L"])] = r
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")  # noqa: E731
    L = []
    bad1 = [(r["device"], r["test"], r["target"], r.get("exact", r.get("error"))) for r in ra
            if r.get("error") or (r.get("exact") or 0) > 1e-6 or not r.get("isa") or not r.get("rzz_in_range")
            or r.get("uses_failed")]
    bad1 += [(r["device"], r["n"], r["L"], r["target"], r["tvd"]) for r in rb
             if r["tvd"] > 1e-6 or not r["in_range"] or not r["isa"] or r["uses_failed"]]
    n1 = sum(1 for r in ra if r.get("exact") is not None) + len(rb)
    L.append(f"P1 every c58 and a19 output exact (<= 1e-6 where checkable), native, rzz in range, on no failed element: "
             f"{'CONFIRMED' if not bad1 else 'REFUTED'} ({n1} checked exact; failures {bad1})")
    arm = lambda r: "C57" if r["target"] == "T" else "C57F"  # noqa: E731
    outs = [r for r in ra if r["path"] == "c57"]
    ins = [r for r in ra if r["path"] != "c57"]
    bad2 = [(r["device"], r["test"], r["target"]) for r in outs if r.get("sig") != old[(r["device"], r["test"], arm(r))]["sig"]]
    L.append(f"P2 out of scope: c58 identical to the release (C57-VAL's rows): {'CONFIRMED' if not bad2 else 'REFUTED'} "
             f"({len(outs) - len(bad2)}/{len(outs)}; differing {bad2})")
    fast = [r for r in ins if r["secs"] <= 0.1 * old[(r["device"], r["test"], arm(r))]["secs"]]
    L.append(f"P3 in scope: compile time <= 0.1 x the release's on >= 90% of cells: "
             f"{'CONFIRMED' if len(fast) >= 0.9 * len(ins) else 'REFUTED'} ({len(fast)}/{len(ins)})")
    insF = [r for r in ins if r["target"] == "F"]
    insT = [r for r in ins if r["target"] == "T"]
    okF = [r for r in insF if r["dur_us"] <= 1.01 * old[(r["device"], r["test"], "C57F")]["dur_us"]]
    badT = [(r["device"], r["test"], round(r["dur_us"] / old[(r["device"], r["test"], "C57")]["dur_us"], 3)) for r in insT
            if r["dur_us"] > 1.5 * old[(r["device"], r["test"], "C57")]["dur_us"]]
    L.append(f"P4 in scope: duration F <= 1.01 x the release's on >= 90% ({len(okF)}/{len(insF)}) and T <= 1.5 x on every "
             f"cell ({'none above' if not badT else badT}): {'CONFIRMED' if len(okF) >= 0.9 * len(insF) and not badT else 'REFUTED'}")
    eq = {tk: sum(1 for r in rb if r["target"] == tk and r["dur_us"] <= 1.0001 * a18[(tk, r["device"], r["n"], r["L"])]["dur_us"])
          for tk in ("T", "F")}
    fastb = [r for r in rb if r["secs"] <= 0.2 * a18[(r["target"], r["device"], r["n"], r["L"])]["secs"]]
    ok5 = eq["T"] >= 62 and eq["F"] >= 62 and len(fastb) >= 0.9 * len(rb)
    L.append(f"P5 a19: duration equal to a18's on >= 62 of 64 per Target (T {eq['T']}, F {eq['F']}) and compile time "
             f"<= 0.2 x a18's on >= 90% ({len(fastb)}/{len(rb)}): {'CONFIRMED' if ok5 else 'REFUTED'}")
    rep = [r for r in ra if "sig_repeat" in r]
    L.append(f"P6 same output on a second call: {'CONFIRMED' if rep and all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
             f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    for lab, rs in (("in scope", ins), ("out of scope", outs)):
        sp = [old[(r["device"], r["test"], arm(r))]["secs"] / max(r["secs"], 1e-4) for r in rs]
        du = [r["dur_us"] / old[(r["device"], r["test"], arm(r))]["dur_us"] for r in rs if r.get("dur_us")]
        L.append(f"reported, {lab} ({len(rs)} cells): release secs / c58 secs geo {gm(sp):.1f}; c58 secs median "
                 f"{sorted(r['secs'] for r in rs)[len(rs)//2]*1000:.0f} ms (max {max(r['secs'] for r in rs)*1000:.0f}); "
                 f"duration c58 / release geo {gm(du):.3f}")
    spb = [a18[(r["target"], r["device"], r["n"], r["L"])]["secs"] / max(r["secs"], 1e-4) for r in rb]
    L.append(f"reported, classifiers: a18 secs / a19 secs geo {gm(spb):.1f}; a19 secs median "
             f"{sorted(r['secs'] for r in rb)[len(rb)//2]*1000:.0f} ms (max {max(r['secs'] for r in rb)*1000:.0f}); "
             f"largest tvd {max(r['tvd'] for r in rb):.1e}; chosen {dict((k, sum(r['chosen'] == k for r in rb)) for k in set(r['chosen'] for r in rb))}")
    L.append(f"reported: FORMULA97_STATS {M.FORMULA97_STATS}; paths {dict((k, sum(r['path'] == k for r in ra)) for k in ('chain', 'graph', 'not_built', 'c57'))}")
    L.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, "c58_val_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
