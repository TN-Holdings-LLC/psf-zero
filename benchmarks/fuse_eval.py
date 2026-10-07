"""fuse_eval.py -- test FUSE (2026-10-07): does candidate psf_compile 2026-10-07.c19 (changelog item 46: fewer, larger
matrices in the state-vector estimates and checks) return exactly release 2026-10-06.4's circuit with the recommended
call, faster at 12-16 qubits, where those simulations run, and no slower above 16, where they do not?

Circuits, per device: SKIP's four families (skip_eval.family_circuit, Addendum 379, unchanged: ring, brick, pauli,
qft) at n = 8, 12, 14, 16 and 20, 3 circuits per (family, n), the first two with measure_all(), the third without:
60 per device, at seeds 82,000,000 + k (smoke: 82,500,000 + k, 1 per cell; none used before).
Devices: FakeTorino, FakeKingston (cz), FakeAuckland, FakeHanoiV2 (cx), FakeBrussels, FakeOsaka (ecr).
Per device: one warm-up call of each module on a 4-qubit ring (discarded). Per circuit: the release and c19 in
alternating order, each timed, with the README's recommended call; the outputs compared instruction by instruction
(with clbits, parameters, global phase and where the logical qubits start and end); c19's output checked for
instructions or couplings the target lacks and, for ring, brick and qft circuits of 8 qubits, for exactness (the
workplace probe's state infidelity, measurements removed).

    python benchmarks/fuse_eval.py run --device D --out DIR [--smoke]
    python benchmarks/fuse_eval.py score --out DIR
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for _p in (os.path.join(WORK, "depth1"), HERE, REPO):  # readout_eval imports depth_eval (DEPTH stage 1)
    sys.path.insert(0, _p)

CZ, CX, ECR = ("FakeTorino", "FakeKingston"), ("FakeAuckland", "FakeHanoiV2"), ("FakeBrussels", "FakeOsaka")
DEVICES = CZ + CX + ECR
C19_PATH = os.path.join(REPO, "patches", "psf_compile_c19_2026-10-07", "psf_compile.py")
SIZES = (8, 12, 14, 16, 20)
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
FAMILIES = ("ring", "brick", "pauli", "qft")  # skip_eval's


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    c19 = H.load_module(C19_PATH, "psf_compile_c19_fuse")
    RE = H.load_module(os.path.join(WORK, "readout", "readout_eval.py"), "readout_eval_fuse")
    if rel.VERSION != "2026-10-06.4" or c19.VERSION != "2026-10-07.c19":
        raise SystemExit(f"STOP: versions {rel.VERSION}, {c19.VERSION}")
    return rel, c19, RE


def family_circuit(name, n, rng):
    """SKIP's generator, imported unchanged (benchmarks/skip_eval.py, Addendum 379)."""
    import skip_eval
    return skip_eval.family_circuit(name, n, rng)


def circuits(device, smoke):
    """[(family, n, measured, circuit to compile, bare circuit)]"""
    base = 82_500_000 if smoke else 82_000_000
    per = 1 if smoke else 3
    out, k = [], 0
    for name in FAMILIES:
        for n in SIZES:
            for j in range(per):
                rng = np.random.default_rng(base + k)
                k += 1
                bare = family_circuit(name, n, rng)
                measured = j < 2
                qc = bare.copy()
                if measured:
                    qc.measure_all()
                out.append((name, n, measured, qc, bare))
    return out


def full_sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)) if c.layout else None,
         list(c.layout.final_index_layout(filter_ancillas=True)) if c.layout else None]


def off_target(out, t):
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        q = tuple(out.find_bit(b).index for b in ins.qubits)
        if ins.operation.name not in t.operation_names or q not in t[ins.operation.name]:
            return True
    return False


def git_state():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return "unknown"
    return git("rev-parse", "--short", "HEAD"), git("status", "--porcelain", "--untracked-files=no")


def run(args):
    import qiskit
    from qiskit_ibm_runtime import fake_provider
    rel, c19, RE = load()
    t = getattr(fake_provider, args.device)().target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]

    def call(M, qc):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            o = M.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                       layout_search=True, target=t, seed_transpiler=0, **RECOMMENDED)
            return o, time.perf_counter() - t0

    rows, t00 = [], time.time()
    warm = family_circuit("ring", 4, np.random.default_rng(1))
    call(rel, warm)  # warm-up of each module, discarded
    call(c19, warm)
    for k, (name, n, measured, qc, bare) in enumerate(circuits(args.device, args.smoke)):
        row = dict(family=name, n=n, measured=measured)
        try:
            if k % 2 == 0:
                (orl, trl) = call(rel, qc)
                (o19, t19) = call(c19, qc)
            else:
                (o19, t19) = call(c19, qc)
                (orl, trl) = call(rel, qc)
        except Exception as e:  # recorded; P0 requires none
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            print(f"{k:3d} {name}{n} ERROR {row['error'][:120]}", flush=True)
            continue
        row.update(t_rel=round(trl, 5), t_c19=round(t19, 5), identical=full_sig(orl) == full_sig(o19),
                   off19=off_target(o19, t), q2=sum(1 for i in o19.data if len(i.qubits) == 2))
        if name in ("ring", "brick", "qft") and n <= 8:
            row["exact19"] = RE.state_infid(bare, RE.strip_measure(o19))
        rows.append(row)
        print(f"{k:3d} {name}{n}{'m' if measured else ''} rel {trl:.3f} s c19 {t19:.3f} s identical "
              f"{row['identical']}", flush=True)
    head, dirty = git_state()
    meta = dict(device=args.device, smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty,
                qiskit=qiskit.__version__, versions=dict(release=rel.VERSION, c19=c19.VERSION),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         c19=norm_sha(C19_PATH), readout_eval=norm_sha(RE.__file__),
                         skip_eval=norm_sha(os.path.join(HERE, "skip_eval.py"))),
                wall_s=round(time.time() - t00, 1))
    os.makedirs(args.out, exist_ok=True)
    fn = f"fuse_{args.device}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"))
    print(f"wrote {fn}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    F, flags, heads = {}, [], set()
    for p in sorted(glob.glob(os.path.join(args.out, "fuse_*.json"))):
        if p.endswith("_smoke.json"):
            continue
        r = json.load(open(p))
        m = r["meta"]
        heads.add(m["git_head"])
        if m["smoke"] or m["dirty_tracked"] or m["versions"] != dict(release="2026-10-06.4", c19="2026-10-07.c19") \
                or len(r["rows"]) != 60:
            flags.append(os.path.basename(p))
        F[m["device"]] = r["rows"]
    rows = [x for rs in F.values() for x in rs]
    err = sum(1 for x in rows if "error" in x)
    ok = [x for x in rows if "error" not in x]
    off = sum(1 for x in ok if x["off19"])
    inexact = sum(1 for x in ok if x.get("exact19", 0.0) > 1e-6)
    checked = sum(1 for x in ok if "exact19" in x)
    p0 = sorted(F) == sorted(DEVICES) and not flags and not err and not off and not inexact and len(heads) == 1 and \
        checked == 9 * len(DEVICES)  # ring, brick and qft at 8 qubits, 3 each, on every device
    out = ["# FUSE score\n", f"files {len(F)}, git_head {sorted(heads)}, flags {flags}",
           f"P0 {'PASS' if p0 else 'FAIL'}: errors {err}, off-target {off}, inexact {inexact} "
           f"(checked {checked}, expected {9 * len(DEVICES)})"]
    if not p0:
        out.append("Nothing below is scored.")
        print("\n".join(out))
        open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")
        return

    def med(rs, sizes):
        return statistics.median(x["t_c19"] / x["t_rel"] for x in rs if x["n"] in sizes)

    out += ["", "| device | circuits | identical | n=16: median c19/release | n=12-14 | n=20 | n=8 |",
            "|---|---|---|---|---|---|---|"]
    r16, r1214, r20 = {}, {}, {}
    for d in DEVICES:
        rs = F[d]
        r16[d], r1214[d], r20[d] = med(rs, (16,)), med(rs, (12, 14)), med(rs, (20,))
        out.append(f"| {d} | {len(rs)} | {sum(x['identical'] for x in rs)} | {r16[d]:.3f} | {r1214[d]:.3f} | "
                   f"{r20[d]:.3f} | {med(rs, (8,)):.3f} |")
    differ = sum(1 for x in rows if not x["identical"])
    res = [("F1", "c19 returns the release's circuit", verdict(differ == 0, differ > 2)),
           ("F2", "n=16: median per-circuit time ratio <= 0.6 on every device",
            verdict(all(v <= 0.6 for v in r16.values()), any(v > 0.85 for v in r16.values()))),
           ("F3", "n=12-14: median per-circuit time ratio <= 0.9 on every device",
            verdict(all(v <= 0.9 for v in r1214.values()), any(v > 1.05 for v in r1214.values()))),
           ("F4", "n=20: median per-circuit time ratio <= 1.10 on every device",
            verdict(all(v <= 1.10 for v in r20.values()), any(v > 1.25 for v in r20.values())))]
    out += ["", "| | prediction | verdict |", "|---|---|---|"] + [f"| {a} | {b} | **{c}** |" for a, b, c in res]
    out += ["", f"Circuits that differ: {differ}.", "",
            "Reported without prediction (median c19/release per family at n=16 | n=12-14):"]
    for fam in FAMILIES:
        a = [x["t_c19"] / x["t_rel"] for x in rows if x["family"] == fam and x["n"] == 16]
        b = [x["t_c19"] / x["t_rel"] for x in rows if x["family"] == fam and x["n"] in (12, 14)]
        out.append(f"- {fam}: {statistics.median(a):.3f} ({len(a)}) | {statistics.median(b):.3f} ({len(b)})")
    for n in SIZES:
        tr = sum(x["t_rel"] for x in rows if x["n"] == n)
        tc = sum(x["t_c19"] for x in rows if x["n"] == n)
        out.append(f"- n={n}: total compile time release {tr:.1f} s, c19 {tc:.1f} s ({tc / tr:.3f})")
    print("\n".join(out))
    open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "score"))
    ap.add_argument("--device", choices=DEVICES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
