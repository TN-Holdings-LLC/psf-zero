"""hw2_predictors_c.py -- HW2-PLUS's corrected simulation predictors P1c and P2c (Addendum 468), fixed before HW2 ran
and computed afterwards, exactly as hw2_predictors.py (Addendum 466, kept unchanged) computes P1 and P2, with one
difference: each measured qubit's waiting from its last gate to the measurement is charged (ns_tail.py).

    python benchmarks/hw2_predictors_c.py <out dir> [hw2 dir] [--only P1c,P2c] [--with <hw2_predictors.jsonl>]

  P1c  ns_tail: as soon as possible, no preparation error on measured qubits, reported T2 while waiting
  P2c  ns_tail: as P1c, with T2 / 2 while waiting (also in the window before the measurement)

Selection of circuits, ideal distributions, the stopping rule and the scoring are hw2_predictors.py's own (imported).
With --with, the rows of hw2_predictors.jsonl (P1-P4) are read as well, to report the comparisons Q4 (P1c against P1)
and Q2c (P2c against P1c)."""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hw2_predictors as HP  # noqa: E402
import ns_tail as NT  # noqa: E402


def main(out, w, only, with_path):
    from qiskit import qpy
    sys.path[:0] = ["benchmarks", "."]
    H = HP._load(os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2")
    os.makedirs(out, exist_ok=False)
    man = json.load(open(os.path.join(w, "hw2_manifest.json")))
    sc = json.load(open(os.path.join(w, "hw2_score.json")))
    circs = qpy.load(open(os.path.join(w, "hw2_circuits.qpy"), "rb"))
    cal = NT.N2.calibration_from_snapshot(os.path.join(w, f"ns_snapshot_{man['device']}.json"))
    todo = [(man["entries"][p["entry"]], c) for p, c in zip(man["pubs"], circs)
            if p["repeat"] == 1 and man["entries"][p["entry"]]["submitted"]
            and (man["entries"][p["entry"]]["part"] == "S" or man["entries"][p["entry"]]["test"] in HP.READING)]
    todo.sort(key=lambda x: x[0]["active"])
    dists, rows = {}, []
    f = open(os.path.join(out, "hw2_predictors_c.jsonl"), "w")
    for e, c in todo:
        t = e["test"]
        if t not in dists:
            qc, _ = H.build_test(t)
            dists[t] = (H.measured_distribution(H._unrolled(qc))[0], qc.num_qubits)
        d, n = dists[t]
        r = dict(part=e["part"], test=t, arm=e["arm"], active=e["active"],
                 f_real=sc["F"][f"{e['part']}|{t}|{e['arm']}"]["f"])
        t0 = time.time()
        win, total = NT.tail_windows(c, cal, "asap")
        r["tail_s"] = sum(win.values())
        if "P1c" in only:
            r["P1c"] = NT.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False)[:2]
        if "P2c" in only:
            r["P2c"] = NT.simulate_xeb(c, d, n, cal, method="asap", init_on_measured=False, idle_t2_factor=0.5)[:2]
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        f.write(json.dumps(r) + "\n")
        f.flush()
        print(json.dumps(r), flush=True)
    f.close()
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

    def cmp(a, b, name):
        if a in res and b in res:
            ra, rb = res[a]["rho"], res[b]["rho"]
            ra = float(ra) if ra is not None else float("nan")
            rb = float(rb) if rb is not None else float("nan")
            v = "NOT TESTABLE" if (math.isnan(ra) or math.isnan(rb)) else ("CONFIRMED" if ra >= rb else "REFUTED")
            lines.append(f"- {name} ({a} ranks at least as well as {b}): rho {a} {ra:+.3f} vs {b} {rb:+.3f} -> {v}")
    cmp("P1c", "P1", "Q4")
    cmp("P2c", "P1c", "Q2c")
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(out, "hw2_predictors_score.md"), "a") as fh:
        fh.write("\n" + txt)
    print(txt)


if __name__ == "__main__":
    argv = sys.argv[1:]
    only, with_path = ("P1c", "P2c"), None
    if "--only" in argv:
        i = argv.index("--only")
        only = tuple(argv[i + 1].split(","))
        del argv[i:i + 2]
    if "--with" in argv:
        i = argv.index("--with")
        with_path = argv[i + 1]
        del argv[i:i + 2]
    main(argv[0], argv[1] if len(argv) > 1 else os.path.join("data", "2026-10-13", "hw2"), only, with_path)
