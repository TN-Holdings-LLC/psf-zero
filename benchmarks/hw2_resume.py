"""hw2_resume.py -- resumes an interrupted run of ns_hw2.py (NS2), hw2_predictors.py (P1-P4) or hw2_predictors_c.py
(P1c, P2c) after the computing container restarted (2026-10-13). It computes only the circuits whose rows are not yet in
the run's jsonl, with the fixed modules' own functions and arguments, appends them, and then scores with the module's
own score(). Every circuit's simulation has its own fixed seed, so a resumed run gives the rows an uninterrupted run
gives (checked: a row recomputed here equals the stored one).

    python benchmarks/hw2_resume.py ns2   <out dir> [hw2 dir]
    python benchmarks/hw2_resume.py pred  <out dir> [hw2 dir] --only P1,P2,P4
    python benchmarks/hw2_resume.py predc <out dir> [hw2 dir] --with <pred jsonl>
    add --check to recompute the first stored row and compare it before resuming."""
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, "."]


def _hw2():
    spec = importlib.util.spec_from_file_location("hw2", os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    return H


def main():
    argv = sys.argv[1:]
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    only, with_path = None, None
    if "--only" in argv:
        i = argv.index("--only")
        only = tuple(argv[i + 1].split(","))
        del argv[i:i + 2]
    if "--with" in argv:
        i = argv.index("--with")
        with_path = argv[i + 1]
        del argv[i:i + 2]
    which, out = argv[0], argv[1]
    w = argv[2] if len(argv) > 2 else os.path.join("data", "2026-10-13", "hw2")
    from qiskit import qpy
    H = _hw2()
    man = json.load(open(os.path.join(w, "hw2_manifest.json")))
    sc = json.load(open(os.path.join(w, "hw2_score.json")))
    circs = qpy.load(open(os.path.join(w, "hw2_circuits.qpy"), "rb"))
    snap = os.path.join(w, f"ns_snapshot_{man['device']}.json")
    if which == "ns2":
        import ns_hw2 as M
        import ns_sim as N
        cal = N.calibration_from_snapshot(snap)
        jl = os.path.join(out, "ns_hw2.jsonl")
        keep = lambda e: e["part"] == "S" or e["test"] in M.READING + ("ghz_n12", "tfimP_n12", "tfimP_n16")  # noqa: E731
    elif which == "pred":
        import hw2_predictors as M
        N = M.N
        cal = N.calibration_from_snapshot(snap)
        jl = os.path.join(out, "hw2_predictors.jsonl")
        keep = lambda e: e["part"] == "S" or e["test"] in M.READING  # noqa: E731
        only = only or ("P1", "P2", "P3", "P4")
    else:
        import hw2_predictors_c as M
        HP = M.HP
        NT = M.NT
        cal = NT.N2.calibration_from_snapshot(snap)
        jl = os.path.join(out, "hw2_predictors_c.jsonl")
        keep = lambda e: e["part"] == "S" or e["test"] in HP.READING  # noqa: E731
        only = only or ("P1c", "P2c")
    os.makedirs(out, exist_ok=True)
    done = [json.loads(x) for x in open(jl)] if os.path.exists(jl) else []
    have = {(r["part"], r["test"], r["arm"]) for r in done}
    todo = [(man["entries"][p["entry"]], c) for p, c in zip(man["pubs"], circs)
            if p["repeat"] == 1 and man["entries"][p["entry"]]["submitted"] and keep(man["entries"][p["entry"]])]
    todo.sort(key=lambda x: x[0]["active"])
    dists = {}

    def dist(t):
        if t not in dists:
            qc, _ = H.build_test(t)
            dists[t] = (H.measured_distribution(H._unrolled(qc))[0], qc.num_qubits)
        return dists[t]

    def row(e, c):
        t = e["test"]
        d, n = dist(t)
        t0 = time.time()
        if which == "ns2":
            fs, se, dur, nact, ntraj = N.simulate_xeb(c, d, n, cal)
            return dict(part=e["part"], test=t, arm=e["arm"], f_sim=fs, se_sim=se, dur_us=dur * 1e6, active=nact,
                        esp=e["esp"], f_real=sc["F"][f"{e['part']}|{t}|{e['arm']}"]["f"], traj=ntraj,
                        secs=round(time.time() - t0, 1))
        r = dict(part=e["part"], test=t, arm=e["arm"], active=e["active"],
                 f_real=sc["F"][f"{e['part']}|{t}|{e['arm']}"]["f"])
        if which == "pred":
            if "P1" in only:
                r["P1"] = N.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False)[:2]
            if "P2" in only:
                r["P2"] = N.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False, idle_t2_factor=0.5)[:2]
            if "P4" in only:
                import snapshot_target as ST
                r["P4"] = (math.exp(-M.exposure(c, ST.target_from_snapshot(snap))), None)
            if "P3" in only:
                raise SystemExit("P3 is computed separately (see hw2_p3.py)")
        else:
            win, total = NT.tail_windows(c, cal, "asap")
            r["tail_s"] = sum(win.values())
            if "P1c" in only:
                r["P1c"] = NT.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False)[:2]
            if "P2c" in only:
                r["P2c"] = NT.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False, idle_t2_factor=0.5)[:2]
        r["secs"] = round(time.time() - t0, 1)
        return r

    if check and done:
        first = done[0]
        e, c = next((e, c) for e, c in todo if (e["part"], e["test"], e["arm"]) == (first["part"], first["test"], first["arm"]))
        r = row(e, c)
        keys = [k for k in r if k not in ("secs",)]
        same = all(json.dumps(r[k]) == json.dumps(first[k]) for k in keys if k in first)
        print("CHECK", first["test"], first["arm"], "identical" if same else f"DIFFERENT {r} vs {first}", flush=True)
        if not same:
            raise SystemExit("a recomputed row differs: not resuming")
    f = open(jl, "a")
    for e, c in todo:
        if (e["part"], e["test"], e["arm"]) in have:
            continue
        r = row(e, c)
        done.append(r)
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(json.dumps(r), flush=True)
    f.close()
    print(f"RESUMED: {len(done)} rows in {jl}", flush=True)
    if which == "ns2":
        M.score(out, done)
    elif which == "pred":
        M.score(out, done, only)
    else:
        sys.argv = ["hw2_predictors_c.py"]
        rows = done
        keys = list(only)
        if with_path:
            old = {(x["part"], x["test"], x["arm"]): x for x in map(json.loads, open(with_path))}
            for r in rows:
                for P in ("P1", "P2"):
                    o = old.get((r["part"], r["test"], r["arm"]))
                    if o and P in o:
                        r[P] = o[P]
            keys += [P for P in ("P1", "P2") if all(P in r for r in rows)]
        HP.score(out, rows, tuple(keys))
        res = json.load(open(os.path.join(out, "hw2_predictors_score.json")))
        lines = []
        for name, a, b in (("Q4", "P1c", "P1"), ("Q2c", "P2c", "P1c")):  # as hw2_predictors_c.py
            if a in res and b in res:
                ra, rb = float(res[a]["rho"]), float(res[b]["rho"])
                v = "NOT TESTABLE" if (math.isnan(ra) or math.isnan(rb)) else ("CONFIRMED" if ra >= rb else "REFUTED")
                lines.append(f"- {name} ({a} ranks at least as well as {b}): rho {a} {ra:+.3f} vs {b} {rb:+.3f} -> {v}")
        with open(os.path.join(out, "hw2_predictors_score.md"), "a") as fh:
            fh.write("\n" + "\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
