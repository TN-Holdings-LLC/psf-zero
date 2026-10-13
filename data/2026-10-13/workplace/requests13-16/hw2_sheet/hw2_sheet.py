"""hw2_sheet.py -- assembles HW2's results into one sheet in the form of the results addendum (Addendum 461, laid out
as HW1's Addendum 455: run, per-test table, prediction table with results, then the added predictors and the fixed
analysis). It computes nothing new: every verdict is read from the file that the pre-registered tool wrote; ratios
are recomputed from the same rows only to lay them side by side, and a mismatch with a tool's own number is flagged.

    python hw2_sheet.py --hw2 <hw2 dir> [--ns2 <dir with ns_hw2.jsonl, ns_hw2_score.json>]
                        [--pred <dir with hw2_predictors.jsonl, hw2_predictors_score.json>]
                        [--predc <dir with hw2_predictors_c.jsonl, hw2_predictors_score.json>]
                        [--within <hw2_within_test_v2.json>] [--within-v1 <hw2_within_test.json>]
                        [--out <sheet.md>]

Inputs that are not given (or missing) are listed as "not available"; the sheet is still written. It writes the
sheet (Markdown) and <sheet>.json beside it, and never overwrites an existing sheet."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os

READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")
TESTS_R = READING + ("tfim_n16", "ghz_n12", "tfimP_n12", "tfimP_n16")
TESTS_S = ("tfimP_n12", "tfimP_n16", "qaoa_n12_p2", "qaoa_n20_p2")
PRED_ORDER = ("ESP", "R3", "NS2", "P1", "P2", "P3", "P4", "P1c", "P2c")
TOL = 1e-6

TEXT_R = {
    "R1": "of the reading tests whose ESP ratio PSF / QK3 is >= 1.2, PSF's F >= QK3's on at most half",
    "R2": "rank correlation of ESP ratio and F ratio over the reading tests <= 0.3",
    "R3": "the estimate with idling picks the same winner as F on >= 5 of 7 reading tests, and its rank correlation "
          "with the F ratio is above ESP's",
    "R4": "on every test (the seven and the two tfimP) whose PSF duration is <= 0.8 x QK3's, PSF's F >= QK3's (at "
          "least two such tests)",
    "R5": "Barabasi-Albert: PSF's F <= 1.05 x QK3's on >= 3 of 4",
    "R6": "GHZ control: \\|F ratio - 1\\| <= 0.10",
    "S1": "tfimP: S1's F > S0's on every pair",
    "S2": "tfimP: geometric mean F ratio S1 / S0 >= 1.05",
    "NS2a": "the simulation picks the device's winner on >= 6 of every 7 reading tests scored",
    "NS2b": "rank correlation of simulated and measured F ratios on the reading tests >= 0.6",
    "NS2c": "simulated F within 0.5-2 times the measured F on >= 80% of part R's circuits (tfim_n16 left out)",
    "NS2d": "F_real / F_sim per arm: \\|ln(PSF / QK3)\\| < 0.153",
    "Q1": "at least one of P1-P4 is useful: the device's winner on >= 5 of 7 and rank correlation >= 0.5",
    "Q2": "H-T2*: P2's rank correlation >= P1's",
    "Q3": "P4 (state-blind) ranks at least as well as P3 (state-aware)",
    "Q4 (amendment 2)": "P1c ranks at least as well as P1",
    "Q2c": "P2c ranks at least as well as P1c",
}


def stamp():
    u = dt.datetime.now(dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return u.astimezone(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:  # noqa: BLE001
        return u.isoformat(timespec="seconds")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def jload(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def jlines(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    return None


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def f3(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def bold_v(v):
    v = str(v)
    head = v.split(" ")[0]
    return f"**{head}**" + v[len(head):] if head in ("CONFIRMED", "REFUTED") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw2", required=True)
    ap.add_argument("--ns2")
    ap.add_argument("--pred")
    ap.add_argument("--predc")
    ap.add_argument("--within")
    ap.add_argument("--within-v1", dest="within_v1")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.hw2, "hw2_sheet.md")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists: not overwritten")

    paths = dict(manifest=os.path.join(a.hw2, "hw2_manifest.json"), score=os.path.join(a.hw2, "hw2_score.json"),
                 raw=os.path.join(a.hw2, "hw2_raw_counts.json"),
                 ns2_rows=a.ns2 and os.path.join(a.ns2, "ns_hw2.jsonl"),
                 ns2_score=a.ns2 and os.path.join(a.ns2, "ns_hw2_score.json"),
                 pred_rows=a.pred and os.path.join(a.pred, "hw2_predictors.jsonl"),
                 pred_score=a.pred and os.path.join(a.pred, "hw2_predictors_score.json"),
                 predc_rows=a.predc and os.path.join(a.predc, "hw2_predictors_c.jsonl"),
                 predc_score=a.predc and os.path.join(a.predc, "hw2_predictors_score.json"),
                 within=a.within, within_v1=a.within_v1)
    man, sc, raw = jload(paths["manifest"]), jload(paths["score"]), jload(paths["raw"])
    if man is None or sc is None:
        raise SystemExit("hw2_manifest.json and hw2_score.json are required")
    ns2r, ns2s = jlines(paths["ns2_rows"]), jload(paths["ns2_score"])
    prr, prs = jlines(paths["pred_rows"]), jload(paths["pred_score"])
    pcr, pcs = jlines(paths["predc_rows"]), jload(paths["predc_score"])
    wv2, wv1 = jload(paths["within"]), jload(paths["within_v1"])
    F = sc["F"]
    flags = []
    L, J = [], dict(written=stamp(), inputs={})

    def g(part, t, arm):
        return F.get(f"{part}|{t}|{arm}")

    # ------------------------------------------------------------------ run
    ent = man["entries"]
    sub = [e for e in ent if e["submitted"]]
    nsub = [e for e in ent if not e["submitted"]]
    ex = [e["exact_maxdiff"] for e in sub if e.get("exact_maxdiff") is not None]
    n_na = sum(1 for e in sub if e.get("exact_maxdiff") is None)
    usage = ((raw or {}).get("metrics") or {}).get("usage")
    L += [f"# HW2 results sheet (assembled {stamp()} by hw2_sheet.py; the numbers are the tools' own)", "",
          "**Run**", "",
          f"- **Device:** {man['device']} (device rule: {man.get('device_rule')}); calibration "
          f"{man.get('calibration_time')}.",
          f"- **Prepare:** manifest written {man.get('written')}; {len(ent)} compiled circuits, {len(sub)} submitted "
          f"({len(man['pubs'])} PUBs with the second repeat); not submitted: "
          + (", ".join(f"{e['part']} {e['test']} {e['arm']}" for e in nsub) or "none") + ".",
          f"- **Checks:** ISA and failed elements passed on every submitted circuit "
          f"({sum(1 for e in sub if e['isa'] and not e['uses_failed'])} of {len(sub)}); exact (noiseless, <= 24 active "
          f"qubits) largest difference {max(ex) if ex else float('nan'):.1e}; above 24 active qubits (n/a): {n_na}.",
          f"- **Job:** {man.get('job_id')}; submitted {man.get('submitted')}; fetched {(raw or {}).get('fetched')}; "
          f"status {(raw or {}).get('status')}; usage {usage}.", ""]

    # ------------------------------------------------------------------ part R table
    L += ["**Part R (F: normalised linear XEB, mean of two repeats):**", "",
          "| test | ESP QK3 | ESP PSF | ESP ratio PSF/QK3 | idle-est ratio | duration PSF / QK3 (us) | F QK3 | F PSF | "
          "**F ratio** | largest repeat difference |", "|---|---|---|---|---|---|---|---|---|---|"]
    rp = sc.get("part_r") or {}
    for t in TESTS_R:
        x = rp.get(t)
        q, p = g("R", t, "QK3"), g("R", t, "PSF")
        if not x or not q or not p:
            continue
        L.append(f"| {t} | {q['esp']:.3f} | {p['esp']:.3f} | {x['esp']:.3f} | {x['idle']:.3f} | {p['dur']:.1f} / "
                 f"{q['dur']:.1f} | {q['f']:.3f} | {p['f']:.3f} | **{f3(x['f'])}** | {x['drift']:.3f} |")
    v = sc["verdict"]
    rd = [rp[t] for t in READING if rp.get(t) and rp[t]["f"] == rp[t]["f"]]
    big = [x for x in rd if x["esp"] >= 1.2]
    detail = {
        "R1": f"{sum(1 for x in big if x['f1'] >= x['f0'])} of {len(big)} (" + ", ".join(
            f"{x['test']} {x['f']:.2f}" for x in big) + ")",
        "R2": f"{sc['rho_esp']:+.3f} ({len(rd)} tests)",
        "R3": f"same winner {sum(1 for x in rd if (x['idle'] > 1) == (x['f'] > 1))} of {len(rd)} (ESP "
              f"{sum(1 for x in rd if (x['esp'] > 1) == (x['f'] > 1))}); rho {sc['rho_idle']:+.3f} vs ESP {sc['rho_esp']:+.3f}",
        "R4": ", ".join(f"{x['test']} duration {x['dur']:.2f} F {x['f']:.2f}" for t in READING + ("tfimP_n12", "tfimP_n16")
                        for x in [rp.get(t)] if x and x["f"] == x["f"] and x["dur"] <= 0.8) or "no such test",
        "R5": ", ".join(f"{t} {rp[t]['f']:.2f}" for t in ("ba_n14", "ba_n16", "ba_n18", "ba_n20") if rp.get(t)),
        "R6": f"F ratio {f3((rp.get('ghz_n12') or {}).get('f'))}",
    }
    sp = sc.get("part_s") or {}
    tp = [x for t, x in sp.items() if t.startswith("tfimP") and x and x["f"] == x["f"]]
    detail["S1"] = f"{sum(1 for x in tp if x['f1'] > x['f0'])} of {len(tp)}"
    gm = math.exp(sum(math.log(x["f"]) for x in tp) / len(tp)) if tp else float("nan")
    detail["S2"] = f"{f3(gm)}"
    L += ["", "| ID | prediction | result |", "|---|---|---|"]
    for k in ("R1", "R2", "R3", "R4", "R5", "R6"):
        L.append(f"| {k} | {TEXT_R[k]} | {bold_v(v.get(k))}: {detail[k]} |")
    L += ["", f"**Replication rule (R1, R3 and R4 all confirmed): {'REPLICATED' if sc.get('replicated') else 'NOT REPLICATED'}.**",
          ""]
    if rp.get("tfim_n16"):
        L.append(f"tfim_n16 is reported and not read (XEB failed there in HW1): F PSF {rp['tfim_n16']['f1']:.3f}, QK3 "
                 f"{rp['tfim_n16']['f0']:.3f}.")
        L.append("")

    # ------------------------------------------------------------------ part S
    L += ["**Part S (S1: item 83 on; S0: off; the same gate blocks):**", "",
          "| test | 2q S1 / S0 | ESP ratio | idle-est ratio | duration ratio | F S0 | F S1 | **F ratio S1/S0** | largest "
          "repeat difference |", "|---|---|---|---|---|---|---|---|---|"]
    for t in TESTS_S:
        x = sp.get(t)
        if not x:
            L.append(f"| {t} | not submitted (S0 and S1 identical or a circuit failed a check) | | | | | | | |")
            continue
        L.append(f"| {t} | {g('S', t, 'S1')['twoq']} / {g('S', t, 'S0')['twoq']} | {x['esp']:.3f} | {x['idle']:.3f} | "
                 f"{x['dur']:.3f} | {x['f0']:.3f} | {x['f1']:.3f} | **{f3(x['f'])}** | {x['drift']:.3f} |")
    L += ["", "| ID | prediction | result |", "|---|---|---|",
          f"| S1 | {TEXT_R['S1']} | {bold_v(v.get('S1'))}: {detail['S1']} |",
          f"| S2 | {TEXT_R['S2']} | {bold_v(v.get('S2'))}: {detail['S2']} |",
          "| S3 | reported: the QAOA pairs | " + ("; ".join(f"{t} duration {sp[t]['dur']:.3f}, F ratio {f3(sp[t]['f'])}"
                                                         for t in ("qaoa_n12_p2", "qaoa_n20_p2") if sp.get(t)) or "none submitted") + " |",
          "| S4 | reported: which estimate picked the device's winner | " + ("; ".join(
              f"{t}: ESP {'yes' if (x['esp'] > 1) == (x['f'] > 1) else 'no'}, idle-est {'yes' if (x['idle'] > 1) == (x['f'] > 1) else 'no'}"
              for t, x in sp.items() if x and x["f"] == x["f"]) or "none") + " |", ""]

    # ------------------------------------------------------------------ predictors side by side
    cols = {}

    def add(name, rows, field):
        if not rows:
            return
        d = {}
        for r in rows:
            val = r.get(field)
            if isinstance(val, (list, tuple)):
                val = val[0] if val else None
            if val is not None:
                d[(r["part"], r["test"], r["arm"])] = float(val)
        cols[name] = d
    cols["ESP"] = {tuple(k.split("|")): x["esp"] for k, x in F.items()}
    cols["R3"] = {tuple(k.split("|")): x["idle"] for k, x in F.items()}
    add("NS2", ns2r, "f_sim")
    for P in ("P1", "P2", "P3", "P4"):
        add(P, prr, P)
    for P in ("P1c", "P2c"):
        add(P, pcr, P)
    avail = [P for P in PRED_ORDER if P in cols]
    L += ["**Predictors on the seven reading pairs (predicted ratio PSF / QK3; device: F ratio):**", "",
          "| test | device | " + " | ".join(avail) + " |", "|---|---|" + "---|" * len(avail)]
    tab = {}
    for t in READING:
        x = rp.get(t)
        if not x:
            continue
        cells = []
        for P in avail:
            a1, a0 = cols[P].get(("R", t, "PSF")), cols[P].get(("R", t, "QK3"))
            r_ = a1 / a0 if a1 and a0 and a1 > 0 and a0 > 0 else None
            tab.setdefault(P, []).append((t, r_, x["f"]))
            cells.append(f3(r_))
        L.append(f"| {t} | **{f3(x['f'])}** | " + " | ".join(cells) + " |")
    summ_w, summ_r = [], []
    J["predictors"] = {}
    for P in avail:
        pr = [(t, r_, y) for t, r_, y in tab.get(P, []) if r_ is not None and y == y and y > 0]
        w = sum(1 for _, r_, y in pr if (r_ > 1) == (y > 1))
        rho = spearman([math.log(r_) for _, r_, _ in pr], [math.log(y) for _, _, y in pr])
        summ_w.append(f"{w} of {len(pr)}")
        summ_r.append(f"{rho:+.2f}" if rho == rho else "n/a")
        J["predictors"][P] = dict(winners=w, n=len(pr), spearman=rho)
    L += ["| same winner | | " + " | ".join(summ_w) + " |", "| rank correlation | | " + " | ".join(summ_r) + " |", ""]
    missing = [P for P in PRED_ORDER if P not in cols]
    if missing:
        L += [f"Not available: {', '.join(missing)}.", ""]
    # cross-check with the tools' own numbers
    if prs:
        for P, r in prs.items():
            if P in J["predictors"] and r.get("rho") is not None and r["rho"] == r["rho"]:
                if abs(r["rho"] - J["predictors"][P]["spearman"]) > TOL:
                    flags.append(f"{P}: rank correlation here {J['predictors'][P]['spearman']:+.4f}, tool {r['rho']:+.4f}"
                                 " (the tool ranks without tie correction)")
    if ns2s and "NS2" in J["predictors"] and ns2s.get("rho") == ns2s.get("rho"):
        if abs(ns2s["rho"] - J["predictors"]["NS2"]["spearman"]) > TOL:
            flags.append(f"NS2: rank correlation here {J['predictors']['NS2']['spearman']:+.4f}, tool {ns2s['rho']:+.4f}")
    if abs(sc["rho_esp"] - J["predictors"]["ESP"]["spearman"]) > TOL:
        flags.append(f"ESP: rank correlation here {J['predictors']['ESP']['spearman']:+.4f}, hw2_score {sc['rho_esp']:+.4f}")

    # ------------------------------------------------------------------ NS2 and HW2-PLUS predictions
    L += ["| ID | prediction | result |", "|---|---|---|"]
    if ns2s:
        nv = ns2s["verdict"]
        pr = ns2s.get("pairs", [])
        agree = sum(1 for x in pr if (x["sim"] > 1) == (x["real"] > 1))
        gmv = ns2s.get("gm", {})
        gap = abs(math.log(gmv["PSF"] / gmv["QK3"])) if "PSF" in gmv and "QK3" in gmv else float("nan")
        within = None
        if ns2r:
            allc = [r for r in ns2r if r["part"] == "R" and r["test"] != "tfim_n16" and r["f_real"] > 0]
            within = f"{sum(1 for r in allc if 0.5 <= r['f_sim'] / r['f_real'] <= 2.0)} of {len(allc)}"
        L += [f"| NS2a | {TEXT_R['NS2a']} | {bold_v(nv.get('NS2a'))}: {agree} of {len(pr)} |",
              f"| NS2b | {TEXT_R['NS2b']} | {bold_v(nv.get('NS2b'))}: {f3(ns2s.get('rho'))} (ESP's {f3(ns2s.get('rho_esp'))}) |",
              f"| NS2c | {TEXT_R['NS2c']} | {bold_v(nv.get('NS2c'))}: {within or 'n/a'} |",
              f"| NS2d | {TEXT_R['NS2d']} | {bold_v(nv.get('NS2d'))}: " + ", ".join(f"{k} {x:.3f}" for k, x in gmv.items())
              + f"; gap {f3(gap)} |",
              "| NS2e | reported: part S, simulated S1 / S0 against the device | " + ("; ".join(
                  f"{t} sim {x:.3f} real {y:.3f}" for t, x, y in ns2s.get("part_s", [])) or "none") + " |"]
    else:
        L.append("| NS2a-e | | not available |")
    if prs:
        useful = [P for P in ("P1", "P2", "P3", "P4") if (prs.get(P) or {}).get("useful")]
        q1 = "CONFIRMED" if useful else "REFUTED"

        def cmp(a_, b_):
            ra, rb = (prs.get(a_) or {}).get("rho"), (prs.get(b_) or {}).get("rho")
            if ra is None or rb is None or ra != ra or rb != rb:
                return "NOT TESTABLE", ra, rb
            return ("CONFIRMED" if ra >= rb else "REFUTED"), ra, rb
        q2, r2, r1 = cmp("P2", "P1")
        q3, r4, r3 = cmp("P4", "P3")
        L += [f"| Q1 | {TEXT_R['Q1']} | {bold_v(q1)}: useful " + (", ".join(useful) or "none") + "; " + ", ".join(
                  f"{P} {prs[P]['agree']} of {len(prs[P]['pairs'])}, {f3(prs[P]['rho'], 2)}" for P in ("P1", "P2", "P3", "P4")
                  if P in prs) + " |",
              f"| Q2 | {TEXT_R['Q2']} | {bold_v(q2)}: P2 {f3(r2)} vs P1 {f3(r1)} |",
              f"| Q3 | {TEXT_R['Q3']} | {bold_v(q3)}: P4 {f3(r4)} vs P3 {f3(r3)} |"]
    else:
        L.append("| Q1-Q3 | | not available |")
    if pcs:
        r1c = (pcs.get("P1c") or {}).get("rho")
        r2c = (pcs.get("P2c") or {}).get("rho")
        r1 = (prs or {}).get("P1", {}).get("rho") if prs else (pcs.get("P1") or {}).get("rho")
        q4 = "NOT TESTABLE" if r1c is None or r1 is None or r1c != r1c or r1 != r1 else ("CONFIRMED" if r1c >= r1 else "REFUTED")
        q2c = "NOT TESTABLE" if r1c is None or r2c is None or r1c != r1c or r2c != r2c else (
            "CONFIRMED" if r2c >= r1c else "REFUTED")
        L += [f"| Q4 (amendment 2) | {TEXT_R['Q4 (amendment 2)']} | {bold_v(q4)}: P1c {f3(r1c)} vs P1 {f3(r1)} |",
              f"| Q2c | {TEXT_R['Q2c']} | {bold_v(q2c)}: P2c {f3(r2c)} vs P1c {f3(r1c)} |",
              "| reported | P1c, P2c: winners, rank correlation, useful | " + "; ".join(
                  f"{P} {pcs[P]['agree']} of {len(pcs[P]['pairs'])}, {f3(pcs[P]['rho'], 2)}, useful "
                  f"{'yes' if pcs[P]['useful'] else 'no'}" for P in ("P1c", "P2c") if P in pcs) + " |"]
    else:
        L.append("| Q4 (amendment 2), Q2c | | not available |")
    L += ["", "(HW2-PLUS's own Q4 is \"reported: each predictor's winner count and rank correlation\" -- the table "
          "above; the amendment's Q4 is labelled \"Q4 (amendment 2)\" here to keep the two apart.)", ""]

    # ------------------------------------------------------------------ the fixed analysis (plan v2)
    if wv2:
        L += [f"**Fixed within-test analysis (plan v2: nine predictors, Holm over the nine, scope A; noise share "
              f"{f3(wv2.get('noise_share_A'), 2)}):**", "",
              "| predictor | n | winners | Spearman | perm p (one-sided) | Holm p | BF+0 | label (v2, Holm) | label (v1) |",
              "|---|---|---|---|---|---|---|---|---|"]
        got = {r["predictor"]: r for r in wv2.get("scope_A", [])}
        for P in PRED_ORDER:
            r = got.get(P)
            if not r:
                L.append(f"| {P} | - | missing (p = 1 in the family) | | | | | | |")
                continue
            L.append(f"| {P} | {r['n']} | {r['winners']} | {r['spearman']:+.2f} | {r['perm_p_one']:.4f} | "
                     f"{r['holm_p']:.4f} | {r['bf_plus0']:.2f} | {r['label_v2']} | {r['label_v1']} |")
        L.append("")
    else:
        L += ["**Fixed within-test analysis (plan v2):** not available.", ""]
    if wv1:
        L += ["(Plan v1's output was also given; its labels are the 'label (v1)' column.)", ""]

    # ------------------------------------------------------------------ flags and inputs
    if flags:
        L += ["**Cross-check flags (numbers here against the tools' own):**", ""] + [f"- {x}" for x in flags] + [""]
    L += ["**Inputs (SHA-256):**", "", "| file | sha256 |", "|---|---|"]
    for k, pth in paths.items():
        if pth and os.path.exists(pth):
            h = sha(pth)
            J["inputs"][k] = dict(path=os.path.basename(pth), sha256=h)
            L.append(f"| {k}: {os.path.basename(pth)} | `{h}` |")
        else:
            L.append(f"| {k} | not available |")
    txt = "\n".join(L) + "\n"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    J["flags"] = flags
    J["verdict"] = dict(hw2=v, replicated=sc.get("replicated"), ns2=(ns2s or {}).get("verdict"))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(J, f, indent=1, default=str)
    print(txt)


if __name__ == "__main__":
    main()
