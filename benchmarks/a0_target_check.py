"""a0_target_check.py -- Phase A0 (pre-registered 2026-10-01): how often is a reported gate error below the
decoherence floor implied by the same snapshot's T1, T2 and gate duration, across every fake-provider device?

The floor of a gate is the average gate infidelity of zero-temperature thermal relaxation on each of its qubits
for the gate's duration, with T2 truncated to 2*T1. This is exactly the relaxation part of qiskit-aer's
NoiseModel.from_backend; when the reported error is below it, Aer applies the floor instead, i.e. the simulator
already uses max(reported, floor). A gate is "below floor" when 0 < reported error < floor.

  python a0_target_check.py run --out <dir> [--smoke]     needs qiskit, qiskit-aer, qiskit-ibm-runtime
  python a0_target_check.py score --out <dir>              numpy only; recomputes the floor from the raw data

--smoke runs FakeAuckland only (already examined in Addendum 286) and prints only the harness checks and row
counts, no below-floor statistics.
"""
import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

TWO_Q = ("cx", "ecr", "cz")
ONE_Q = ("sx", "x")
ERR_MAX = 0.5          # reported errors >= this are treated as disabled gates and excluded
NAMED = ("FakeAuckland", "FakeTorino", "FakeKingston")


# ---------------------------------------------------------------- the floor, analytically (no qiskit)
def floor_analytic(t1s, t2s, dur):
    """1 - average gate fidelity of the tensor product of zero-temperature T1/T2 channels.

    Single qubit: the Pauli transfer matrix has diagonal (1, e^-t/T2, e^-t/T2, e^-t/T1), so the process fidelity
    is (1 + 2 e^-t/T2 + e^-t/T1) / 4. Process fidelities multiply under tensor products, and
    F_avg = (d F_pro + 1) / (d + 1) with d = 2^n. T2 is truncated to 2*T1 as qiskit-aer does.
    """
    if not dur:
        return 0.0
    f_pro = 1.0
    for t1, t2 in zip(t1s, t2s):
        t1 = math.inf if t1 is None else t1
        t2 = 2 * t1 if t2 is None else min(t2, 2 * t1)
        f_pro *= (1 + 2 * math.exp(-dur / t2) + math.exp(-dur / t1)) / 4
    d = 2 ** len(t1s)
    return 1 - (d * f_pro + 1) / (d + 1)


# ---------------------------------------------------------------- run (qiskit side)
def devices(smoke):
    from qiskit.providers import BackendV2
    from qiskit_ibm_runtime import fake_provider as fp
    if smoke:
        return [("FakeAuckland", fp.FakeAuckland)]
    out = []
    for n in sorted(dir(fp)):
        cls = getattr(fp, n)
        if (n.startswith("Fake") and isinstance(cls, type) and issubclass(cls, BackendV2)
                and "Fractional" not in n and "Generic" not in n and n not in ("FakeBackend", "FakeBackendV2")):
            out.append((n, cls))
    return out


def snapshot_date(b):
    for get in (lambda: b.properties().last_update_date, lambda: b._props_dict.get("last_update_date")):
        try:
            v = get()
            if v:
                return str(v)
        except Exception:
            pass
    return None


def run(args):
    import qiskit
    import qiskit_aer
    import qiskit_ibm_runtime
    from qiskit.circuit import Gate
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    meta = dict(qiskit=qiskit.__version__, aer=qiskit_aer.__version__, runtime=qiskit_ibm_runtime.__version__,
                python=sys.version.split()[0], smoke=args.smoke, started=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                       time.gmtime()))
    print("META", json.dumps(meta), flush=True)
    devs, rows, skipped, seen = [], [], [], set()
    t0 = time.time()
    for cname, cls in devices(args.smoke):
        try:
            b = cls()
            tgt = b.target
        except Exception as ex:  # pragma: no cover - depends on the installed snapshot set
            skipped.append((cname, f"construct: {type(ex).__name__}: {ex}"))
            continue
        if b.name in seen:
            skipped.append((cname, f"duplicate of {b.name}"))
            continue
        seen.add(b.name)
        qp = tgt.qubit_properties or [None] * tgt.num_qubits
        t1 = [getattr(p, "t1", None) if p is not None else None for p in qp]
        t2 = [getattr(p, "t2", None) if p is not None else None for p in qp]
        try:
            nm = NoiseModel.from_backend(b)
            local = nm._local_quantum_errors
        except Exception as ex:
            nm, local = None, None
            skipped.append((cname, f"noise model: {type(ex).__name__}: {ex}"))
        nrow = 0
        for op in tgt.operation_names:
            if op not in TWO_Q + ONE_Q:
                continue
            obj = tgt.operation_from_name(op)
            if not isinstance(obj, Gate):
                continue
            for qargs, ip in (tgt[op] or {}).items():
                if ip is None or qargs is None:
                    continue
                qa = tuple(int(q) for q in qargs)
                err, dur = ip.error, ip.duration
                t1s, t2s = [t1[q] for q in qa], [t2[q] for q in qa]
                fa = floor_analytic(t1s, t2s, dur)
                fq = None
                if dur:
                    e = None
                    for a, c in zip(t1s, t2s):
                        # the same truncation as qiskit_aer.noise.device.models._truncate_t2_value
                        if a is not None:
                            c = 2 * a if c is None else min(c, 2 * a)
                        a = math.inf if a is None else a
                        c = math.inf if c is None else c
                        s = thermal_relaxation_error(a, c, dur, 0)
                        e = s if e is None else e.expand(s)
                    fq = float(1 - average_gate_fidelity(e))
                applied = None
                if local is not None:
                    qe = local.get(op, {}).get(qa)
                    applied = 0.0 if qe is None else float(1 - average_gate_fidelity(qe))
                rows.append(dict(dev=b.name, op=op, q=list(qa), err=err, dur=dur, t1=t1s, t2=t2s, floor=fa,
                                 floor_aer=fq, applied=applied))
                nrow += 1
        ops = set(tgt.operation_names)
        devs.append(dict(cls=cname, dev=b.name, n=tgt.num_qubits, snapshot=snapshot_date(b),
                         two_q=sorted(ops & set(TWO_Q)), rows=nrow,
                         t1_missing=sum(v is None for v in t1), t2_missing=sum(v is None for v in t2),
                         t2_clamped=sum(1 for a, c in zip(t1, t2) if a is not None and c is not None and c > 2 * a),
                         median_t2=float(np.median([v for v in t2 if v is not None])) if any(
                             v is not None for v in t2) else None))
        print(f"{b.name:28s} {tgt.num_qubits:4d} qubits {nrow:5d} rows  ({time.time() - t0:.0f} s)", flush=True)
    meta["seconds"] = time.time() - t0
    res = dict(meta=meta, devices=devs, skipped=skipped, rows=rows)
    name = "a0_raw_smoke.json.gz" if args.smoke else "a0_raw.json.gz"
    with gzip.open(os.path.join(args.out, name), "wt", encoding="utf-8") as f:
        json.dump(res, f)
    p0 = harness(res)
    print("HARNESS", json.dumps(p0), flush=True)
    print(f"wrote {name}: {len(devs)} devices, {len(rows)} rows, {len(skipped)} skipped", flush=True)


# ---------------------------------------------------------------- scoring (numpy only)
def usable(r):
    return (r["err"] is not None and 0 < r["err"] < ERR_MAX and r["dur"] and None not in r["t1"]
            and None not in r["t2"])


def harness(res):
    rows = res["rows"]
    rec = [abs(floor_analytic(r["t1"], r["t2"], r["dur"]) - r["floor"]) for r in rows]
    aer = [abs(r["floor"] - r["floor_aer"]) for r in rows if r["floor_aer"] is not None]
    app, app_missing = [], 0
    for r in rows:
        if r["applied"] is None or r["err"] is None or not (0 < r["err"] < ERR_MAX):
            app_missing += r["applied"] is None
            continue
        app.append(abs(r["applied"] - max(r["err"], r["floor"])))
    return dict(rows=len(rows), recompute_max=max(rec, default=0.0), floor_vs_aer_max=max(aer, default=0.0),
                floor_vs_aer_n=len(aer), applied_vs_max_max=max(app, default=0.0), applied_n=len(app),
                applied_missing=app_missing)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    path = os.path.join(args.out, "a0_raw.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        res = json.load(f)
    rows, devs = res["rows"], {d["dev"]: d for d in res["devices"]}
    L = [f"# A0 score: reported gate error below the T1/T2 floor", "",
         "versions " + json.dumps({k: res["meta"][k] for k in ("qiskit", "aer", "runtime", "python")}), ""]
    hz = harness(res)
    named_ok = all(any(d["cls"] == n for d in res["devices"]) for n in NAMED)
    p0 = (hz["recompute_max"] <= 1e-12 and hz["floor_vs_aer_max"] <= 1e-9 and hz["applied_vs_max_max"] <= 1e-9
          and hz["applied_missing"] == 0 and len(devs) >= 30 and named_ok)
    L.append(f"P0: {'PASS' if p0 else 'FAIL'} {json.dumps(hz)}; devices {len(devs)}; named devices present "
             f"{named_ok}; skipped {len(res['skipped'])}")
    for s in res["skipped"]:
        L.append(f"  skipped {s[0]}: {s[1]}")

    # per device
    per = {}
    for r in rows:
        per.setdefault(r["dev"], []).append(r)
    stat = {}
    L += ["", "## Per device", "",
          "| device | qubits | snapshot | 2q gate | 2q usable | 2q below | frac | max floor/err 2q | sx usable | sx below "
          "| median T2 (us) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    cls_of = {}
    for name in sorted(devs, key=lambda n: (devs[n]["two_q"], n)):
        d = devs[name]
        rs = per.get(name, [])
        two = [r for r in rs if r["op"] in TWO_Q and usable(r)]
        sx = [r for r in rs if r["op"] == "sx" and usable(r)]
        below2 = [r for r in two if r["err"] < r["floor"]]
        belowsx = [r for r in sx if r["err"] < r["floor"]]
        c = d["two_q"][0] if len(d["two_q"]) == 1 else ("mixed" if d["two_q"] else "none")
        cls_of[name] = c
        frac = len(below2) / len(two) if two else None
        mx = max((r["floor"] / r["err"] for r in two), default=None)
        stat[name] = dict(cls=c, n2=len(two), b2=len(below2), frac=frac, nsx=len(sx), bsx=len(belowsx),
                          max2=mx, maxsx=max((r["floor"] / r["err"] for r in sx), default=None), below_rows=below2)
        mt2 = d["median_t2"] * 1e6 if d["median_t2"] else float("nan")
        L.append(f"| {name} | {d['n']} | {(d['snapshot'] or '-')[:10]} | {c} | {len(two)} | {len(below2)} | "
                 f"{'-' if frac is None else f'{frac:.3f}'} | {'-' if mx is None else f'{mx:.2f}'} | {len(sx)} | "
                 f"{len(belowsx)} | {mt2:.0f} |")

    # classes
    L += ["", "## By two-qubit gate class (devices with at least one usable 2q gate)", "",
          "| class | devices | devices with any 2q below | median device frac | pooled frac | pooled sx frac |",
          "|---|---|---|---|---|---|"]
    cl = {}
    for c in ("cx", "ecr", "cz", "mixed"):
        ns = [n for n in stat if stat[n]["cls"] == c and stat[n]["n2"] > 0]
        if not ns:
            continue
        med = float(np.median([stat[n]["frac"] for n in ns]))
        pooled = sum(stat[n]["b2"] for n in ns) / sum(stat[n]["n2"] for n in ns)
        nsx = sum(stat[n]["nsx"] for n in ns)
        psx = sum(stat[n]["bsx"] for n in ns) / nsx if nsx else float("nan")
        cl[c] = dict(n=len(ns), med=med, pooled=pooled)
        L.append(f"| {c} | {len(ns)} | {sum(stat[n]['b2'] > 0 for n in ns)} | {med:.3f} | {pooled:.3f} | "
                 f"{psx:.3f} |")

    # predictions
    au, to, ki = (next((n for n in stat if devs[n]["cls"] == k), None) for k in NAMED)
    v = {}
    if au:
        s = stat[au]
        ncx = s["b2"] if s["cls"] == "cx" else None
        r1ok = ncx == 16 and s["max2"] is not None and 1.5 <= s["max2"] <= 1.7 and s["maxsx"] is not None and \
            2.0 <= s["maxsx"] <= 2.2
        r1bad = ncx is None or not (14 <= ncx <= 18)
        v["R1"] = (verdict(r1ok, r1bad), f"Auckland cx below {ncx} of {s['n2']}; max floor/err cx "
                                         f"{s['max2']:.3f}, sx {s['maxsx']:.3f}")
    if to:
        s = stat[to]
        v["R2"] = (verdict(s["b2"] == 0, s["b2"] > 0), f"Torino cz below {s['b2']} of {s['n2']}")
    if "cz" in cl:
        v["H1"] = (verdict(cl["cz"]["med"] <= 0.02, cl["cz"]["med"] > 0.10),
                   f"cz class median device frac {cl['cz']['med']:.3f} ({cl['cz']['n']} devices)")
    if "cx" in cl:
        v["H2"] = (verdict(cl["cx"]["med"] >= 0.05, cl["cx"]["med"] < 0.01),
                   f"cx class median device frac {cl['cx']['med']:.3f} ({cl['cx']['n']} devices)")
    if "cx" in cl and "cz" in cl:
        a, b = cl["cx"], cl["cz"]
        v["H3"] = (verdict(a["med"] > b["med"] and a["pooled"] > b["pooled"],
                           a["med"] <= b["med"] and a["pooled"] <= b["pooled"]),
                   f"cx vs cz: median {a['med']:.3f} vs {b['med']:.3f}, pooled {a['pooled']:.3f} vs {b['pooled']:.3f}")
    below = [(n, r) for n in stat for r in stat[n]["below_rows"]]
    if below:
        low = sum(min(r["t2"]) < devs[n]["median_t2"] for n, r in below) / len(below)
        v["H4"] = (verdict(low >= 0.80, low < 0.50),
                   f"{low:.3f} of {len(below)} below-floor 2q gates have a qubit with T2 below its device median")
    else:
        v["H4"] = ("AMBIGUOUS", "no 2q gate below floor")
    if ki:
        s = stat[ki]
        f = s["frac"] if s["frac"] is not None else float("nan")
        v["H5"] = (verdict(f <= 0.02, f > 0.10), f"Kingston cz below {s['b2']} of {s['n2']} ({f:.3f})")
    L += ["", "## Predictions", ""] + [f"- {k}: **{v[k][0]}** -- {v[k][1]}" for k in sorted(v)]

    # reported without prediction
    L += ["", "## Reported without prediction", ""]
    for n in sorted(stat):
        s = stat[n]
        if s["n2"]:
            tot = sum(r["err"] for r in per[n] if r["op"] in TWO_Q and usable(r))
            eff = sum(max(r["err"], r["floor"]) for r in per[n] if r["op"] in TWO_Q and usable(r))
            s["uplift"] = eff / tot - 1
    ups = {c: [stat[n]["uplift"] for n in stat if stat[n]["cls"] == c and "uplift" in stat[n]] for c in cl}
    for c, u in ups.items():
        L.append(f"- {c}: summed 2q error under max(reported, floor) / reported - 1: median {np.median(u):.4f}, "
                 f"max {max(u):.4f}")
    dated = [(devs[n]["snapshot"][:10], stat[n]["frac"]) for n in stat if devs[n]["snapshot"] and stat[n]["n2"]]
    if len(dated) > 3:
        x = np.argsort(np.argsort([d for d, _ in dated]))
        y = np.argsort(np.argsort([f for _, f in dated]))
        L.append(f"- Spearman rank correlation of snapshot date with 2q below-floor fraction: "
                 f"{np.corrcoef(x, y)[0, 1]:.3f} ({len(dated)} devices)")
    excl = sum(1 for r in rows if r["op"] in TWO_Q + ONE_Q and not usable(r))
    L.append(f"- rows excluded from fractions (no error, error >= {ERR_MAX}, no duration or missing T1/T2): {excl}")
    L.append(f"- qubits with T2 > 2*T1 (truncated, as qiskit-aer does): {sum(d['t2_clamped'] for d in devs.values())}")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "a0_score.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("run", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (run if args.part == "run" else score)(args)


if __name__ == "__main__":
    main()
