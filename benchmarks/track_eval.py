"""track_eval.py -- test TRACK (2026-10-07): does candidate psf_compile 2026-10-07.c22 (changelog items 47-49 on top of
candidate c19's item 46) return exactly candidate c19's circuit with the recommended call, on inputs without
instructions on more than two qubits, and is it faster at 12-16 qubits, where the recommended call simulates, and no
slower above 16, where it does not?

Item 48 (in c22) changes only inputs with an instruction on more than two qubits, so every input here is first
expanded with Qiskit's Unroll3qOrMore (SKIP's pauli and qft circuits have such instructions; ring and brick do not), and
both modules get the same expanded circuit. Items 47 and 49 do not change what is returned except where two estimates
lie within rounding of the tie band (ESTIMATE_TIE_TOL).

Circuits, per device: SKIP's four families (skip_eval.family_circuit, Addendum 379, unchanged: ring, brick, pauli,
qft) at n = 8, 12, 14, 16 and 20, 3 circuits per (family, n), the first two with measure_all(), the third without:
60 per device, at seeds 85,000,000 + k (smoke: 85,500,000 + k, 1 per cell; none used before). As FUSE (Addendum 383)
with new seeds and the expansion.
Devices: FakeTorino, FakeKingston (cz), FakeAuckland, FakeHanoiV2 (cx), FakeBrussels, FakeOsaka (ecr).
Per device: one warm-up call of each module on a 4-qubit ring (discarded). Per circuit: c19 and c22 in alternating
order, each timed, with the README's recommended call; the outputs compared instruction by instruction (with clbits,
parameters, global phase and where the logical qubits start and end); c22's output checked for instructions or
couplings the target lacks and, for ring, brick and qft circuits of 8 qubits, for exactness (the workplace probe's state
infidelity, measurements removed).

    python benchmarks/track_eval.py run --device D --out DIR [--smoke]
    python benchmarks/track_eval.py score --out DIR
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
C22_PATH = os.path.join(REPO, "patches", "psf_compile_c22_2026-10-07", "psf_compile.py")
VERSIONS = dict(c19="2026-10-07.c19", c22="2026-10-07.c22")
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
    c19 = H.load_module(C19_PATH, "psf_compile_c19_track")
    c22 = H.load_module(C22_PATH, "psf_compile_c22_track")
    RE = H.load_module(os.path.join(WORK, "readout", "readout_eval.py"), "readout_eval_track")
    if dict(c19=c19.VERSION, c22=c22.VERSION) != VERSIONS:
        raise SystemExit(f"STOP: versions {c19.VERSION}, {c22.VERSION}")
    return c19, c22, RE


def expanded(qc):
    """The input with every instruction on more than two qubits expanded by Qiskit's Unroll3qOrMore (no basis given:
    every such instruction); a circuit without one is returned as it is."""
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroll3qOrMore
    if not any(len(i.qubits) > 2 and i.operation.name != "barrier" for i in qc.data):
        return qc
    return PassManager([Unroll3qOrMore()]).run(qc)


def wide(qc):
    return sum(1 for i in qc.data if len(i.qubits) > 2 and i.operation.name != "barrier")


def circuits(device, smoke):
    """[(family, n, measured, circuit to compile, bare circuit)], all expanded"""
    import skip_eval
    base = 85_500_000 if smoke else 85_000_000
    per = 1 if smoke else 3
    out, k = [], 0
    for name in FAMILIES:
        for n in SIZES:
            for j in range(per):
                rng = np.random.default_rng(base + k)
                k += 1
                bare = expanded(skip_eval.family_circuit(name, n, rng))
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
    c19, c22, RE = load()
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

    import skip_eval
    rows, t00 = [], time.time()
    warm = skip_eval.family_circuit("ring", 4, np.random.default_rng(1))
    call(c19, warm)  # warm-up of each module, discarded
    call(c22, warm)
    for k, (name, n, measured, qc, bare) in enumerate(circuits(args.device, args.smoke)):
        row = dict(family=name, n=n, measured=measured, wide=wide(qc))
        try:
            if k % 2 == 0:
                (o19, t19) = call(c19, qc)
                (o22, t22) = call(c22, qc)
            else:
                (o22, t22) = call(c22, qc)
                (o19, t19) = call(c19, qc)
        except Exception as e:  # recorded; P0 requires none
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            print(f"{k:3d} {name}{n} ERROR {row['error'][:120]}", flush=True)
            continue
        row.update(t_c19=round(t19, 5), t_c22=round(t22, 5), identical=full_sig(o19) == full_sig(o22),
                   off22=off_target(o22, t), q2=sum(1 for i in o22.data if len(i.qubits) == 2))
        if name in ("ring", "brick", "qft") and n <= 8:
            row["exact22"] = RE.state_infid(bare, RE.strip_measure(o22))
        rows.append(row)
        print(f"{k:3d} {name}{n}{'m' if measured else ''} c19 {t19:.3f} s c22 {t22:.3f} s identical "
              f"{row['identical']}", flush=True)
    head, dirty = git_state()
    meta = dict(device=args.device, smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty,
                qiskit=qiskit.__version__, versions=dict(c19=c19.VERSION, c22=c22.VERSION),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), c19=norm_sha(C19_PATH), c22=norm_sha(C22_PATH),
                         readout_eval=norm_sha(RE.__file__), skip_eval=norm_sha(os.path.join(HERE, "skip_eval.py"))),
                wall_s=round(time.time() - t00, 1))
    os.makedirs(args.out, exist_ok=True)
    fn = f"track_{args.device}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"))
    print(f"wrote {fn}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    F, flags, heads, smoke = {}, [], set(), False
    paths = sorted(glob.glob(os.path.join(args.out, "track_*.json")))
    if paths and all(p.endswith("_smoke.json") for p in paths):
        smoke = True  # a smoke directory is scored as a smoke run (not counted)
    for p in paths:
        if p.endswith("_smoke.json") != smoke:
            continue
        r = json.load(open(p))
        m = r["meta"]
        heads.add(m["git_head"])
        if m["smoke"] != smoke or (m["dirty_tracked"] and not smoke) or m["versions"] != VERSIONS or \
                len(r["rows"]) != (20 if smoke else 60):
            flags.append(os.path.basename(p))
        F[m["device"]] = r["rows"]
    rows = [x for rs in F.values() for x in rs]
    err = sum(1 for x in rows if "error" in x)
    ok = [x for x in rows if "error" not in x]
    off = sum(1 for x in ok if x["off22"])
    wides = sum(1 for x in rows if x.get("wide", 1))
    inexact = sum(1 for x in ok if x.get("exact22", 0.0) > 1e-6)
    checked = sum(1 for x in ok if "exact22" in x)
    want = (3 if smoke else 9) * len(DEVICES)  # ring, brick and qft at 8 qubits, 1 or 3 each, on every device
    p0 = sorted(F) == sorted(DEVICES) and not flags and not err and not off and not wides and not inexact and \
        len(heads) == 1 and checked == want
    out = [f"# TRACK score{' (SMOKE: not counted)' if smoke else ''}\n",
           f"files {len(F)}, git_head {sorted(heads)}, flags {flags}",
           f"P0 {'PASS' if p0 else 'FAIL'}: errors {err}, off-target {off}, inputs with a wide instruction {wides}, "
           f"inexact {inexact} (checked {checked}, expected {want})"]
    if not p0:
        out.append("Nothing below is scored.")
        print("\n".join(out))
        open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")
        return

    def med(rs, sizes):
        return statistics.median(x["t_c22"] / x["t_c19"] for x in rs if x["n"] in sizes)

    out += ["", "| device | circuits | identical | n=16: median c22/c19 | n=12-14 | n=20 | n=8 |",
            "|---|---|---|---|---|---|---|"]
    r16, r1214, r20 = {}, {}, {}
    for d in DEVICES:
        rs = F[d]
        r16[d], r1214[d], r20[d] = med(rs, (16,)), med(rs, (12, 14)), med(rs, (20,))
        out.append(f"| {d} | {len(rs)} | {sum(x['identical'] for x in rs)} | {r16[d]:.3f} | {r1214[d]:.3f} | "
                   f"{r20[d]:.3f} | {med(rs, (8,)):.3f} |")
    differ = sum(1 for x in rows if not x["identical"])
    res = [("T1", "c22 returns c19's circuit", verdict(differ == 0, differ > 2)),
           ("T2", "n=16: median per-circuit time ratio <= 0.6 on every device",
            verdict(all(v <= 0.6 for v in r16.values()), any(v > 0.8 for v in r16.values()))),
           ("T3", "n=12-14: median per-circuit time ratio <= 0.85 on every device",
            verdict(all(v <= 0.85 for v in r1214.values()), any(v > 1.05 for v in r1214.values()))),
           ("T4", "n=20: median per-circuit time ratio <= 1.15 on every device",
            verdict(all(v <= 1.15 for v in r20.values()), any(v > 1.30 for v in r20.values())))]
    out += ["", "| | prediction | verdict |", "|---|---|---|"] + [f"| {a} | {b} | **{c}** |" for a, b, c in res]
    out += ["", f"Circuits that differ: {differ}.", "",
            "Reported without prediction (median c22/c19 per family at n=16 | n=12-14):"]
    for fam in FAMILIES:
        a = [x["t_c22"] / x["t_c19"] for x in rows if x["family"] == fam and x["n"] == 16]
        b = [x["t_c22"] / x["t_c19"] for x in rows if x["family"] == fam and x["n"] in (12, 14)]
        out.append(f"- {fam}: {statistics.median(a):.3f} ({len(a)}) | {statistics.median(b):.3f} ({len(b)})")
    for n in SIZES:
        ta = sum(x["t_c19"] for x in rows if x["n"] == n)
        tb = sum(x["t_c22"] for x in rows if x["n"] == n)
        out.append(f"- n={n}: total compile time c19 {ta:.1f} s, c22 {tb:.1f} s ({tb / ta:.3f})")
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
