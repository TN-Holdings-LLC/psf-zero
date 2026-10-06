"""skip_eval.py -- test SKIP (2026-10-06): does candidate psf_compile 2026-10-06.c17 (changelog item 45: alternatives
that item 39 cannot check are not built) return exactly release 2026-10-06.3's circuit with the recommended call,
faster above 16 logical qubits and no slower up to 16?

Circuits, per device (seeds 81,000,000 + k; smoke 81,500,000 + k, 1 per cell; none used before):
- four families x four sizes x 3 circuits = 48: sizes 10 and 16 (up to 16: c17 skips nothing) and 17 and L (above 16:
  c17 skips level 3 and the floor), L = 26 on the 27-qubit devices and 48 on the others;
  - ring: two layers of random RY on every qubit and CZ along a line (as test_c17.py);
  - brick: four brickwork layers of Haar-random two-qubit unitaries on a line;
  - pauli: one PauliEvolutionGate (time 1) of 3n random Pauli strings of weight 2-4, as Benchpress builds HamLib tests;
  - qft: random single-qubit unitaries, then QFTGate(n);
  the first two circuits of each cell with measure_all(), the third without;
- on the two 27-qubit devices, also 3 circuits of PL-GPU-REDO's family T at spare 0 (27 qubits, the full device;
  Addendum 375): 51 circuits there.
Devices: FakeTorino, FakeKingston (cz), FakeAuckland, FakeHanoiV2 (cx), FakeBrussels, FakeOsaka (ecr).
Per circuit: a warm-up call of the release (discarded), then the release and c17 in alternating order, each timed,
with the README's recommended call; the outputs compared instruction by instruction (with clbits, parameters, global
phase and where the logical qubits start and end); c17's SKIP_STATS before and after its call; c17's output checked
for instructions or couplings the target lacks and, for ring, brick and qft circuits of at most 12 qubits, for
exactness (the workplace probe's state infidelity, measurements removed).

    python benchmarks/skip_eval.py run --device D --out DIR [--smoke]
    python benchmarks/skip_eval.py score --out DIR
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
SMALL27 = ("FakeAuckland", "FakeHanoiV2")
C17_PATH = os.path.join(REPO, "patches", "psf_compile_c17_2026-10-06", "psf_compile.py")
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
FAMILIES = ("ring", "brick", "pauli", "qft")


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
    c17 = H.load_module(C17_PATH, "psf_compile_c17_skip")
    RE = H.load_module(os.path.join(WORK, "readout", "readout_eval.py"), "readout_eval_skip")
    if rel.VERSION != "2026-10-06.3" or c17.VERSION != "2026-10-06.c17":
        raise SystemExit(f"STOP: versions {rel.VERSION}, {c17.VERSION}")
    return rel, c17, RE


def family_circuit(name, n, rng):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate, QFTGate, UnitaryGate
    from qiskit.quantum_info import SparsePauliOp, random_unitary
    qc = QuantumCircuit(n)
    if name == "ring":
        for _ in range(2):
            for q in range(n):
                qc.ry(float(rng.uniform(-1, 1)), q)
            for q in range(n - 1):
                qc.cz(q, q + 1)
    elif name == "brick":
        for layer in range(4):
            for q in range(layer % 2, n - 1, 2):
                qc.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(2**31)))), [q, q + 1])
    elif name == "pauli":
        terms = []
        for _ in range(3 * n):
            w = int(rng.integers(2, 5))
            qs = sorted(int(x) for x in rng.choice(n, size=w, replace=False))
            terms.append(("".join(rng.choice(list("XYZ"), size=w)), qs, float(rng.uniform(-1, 1))))
        qc.append(PauliEvolutionGate(SparsePauliOp.from_sparse_list(terms, num_qubits=n), time=1.0), range(n))
    elif name == "qft":
        for q in range(n):
            qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
        qc.append(QFTGate(n), range(n))
    else:
        raise ValueError(name)
    return qc


def full_t(device, rng):
    """PL-GPU-REDO's family T at spare 0 (Addendum 375), without PennyLane: Haar-random two-qubit unitaries on the
    pair and triple blocks of a full heavy-hex layout."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    from qiskit_ibm_runtime import fake_provider
    import full_heavyhex_cliff as fh
    f = fh.graph_facts(getattr(fake_provider, device)().target)
    blocks = fh.layout_blocks("T", 0, f["nq"], f["matching"])
    qc = QuantumCircuit(sum(len(b) for b in blocks))
    for b in blocks:
        for a, c in ([(b[1], b[0])] if len(b) == 2 else [(b[1], b[0]), (b[2], b[1])]):
            for _ in range(10):
                qc.unitary(random_unitary(4, seed=int(rng.integers(2**31))).data, [a, c])
    return qc


def circuits(device, smoke):
    """[(family, n, measured, circuit to compile, bare circuit)]"""
    big = 26 if device in SMALL27 else 48
    base = 81_500_000 if smoke else 81_000_000
    per = 1 if smoke else 3
    out, k = [], 0
    for name in FAMILIES:
        for n in (10, 16, 17, big):
            for j in range(per):
                rng = np.random.default_rng(base + k)
                k += 1
                bare = family_circuit(name, n, rng)
                measured = j < 2
                qc = bare.copy()
                if measured:
                    qc.measure_all()
                out.append((name, n, measured, qc, bare))
    if device in SMALL27:
        for j in range(per):
            rng = np.random.default_rng(base + 900 + j)
            bare = full_t(device, rng)
            out.append(("fullT", bare.num_qubits, False, bare, bare))
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
    rel, c17, RE = load()
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
    for k, (name, n, measured, qc, bare) in enumerate(circuits(args.device, args.smoke)):
        row = dict(family=name, n=n, measured=measured)
        try:
            call(rel, qc)  # warm-up, discarded
            if k % 2 == 0:
                (orl, trl) = call(rel, qc)
                before = dict(c17.SKIP_STATS)
                (o17, t17) = call(c17, qc)
            else:
                before = dict(c17.SKIP_STATS)
                (o17, t17) = call(c17, qc)
                (orl, trl) = call(rel, qc)
            skipped = {key: c17.SKIP_STATS[key] - before[key] for key in before}
        except Exception as e:  # recorded; P0 requires none
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            print(f"{k:3d} {name}{n} ERROR {row['error'][:120]}", flush=True)
            continue
        row.update(t_rel=round(trl, 5), t_c17=round(t17, 5), identical=full_sig(orl) == full_sig(o17),
                   skipped=skipped, off17=off_target(o17, t), q2=sum(1 for i in o17.data if len(i.qubits) == 2))
        if name in ("ring", "brick", "qft") and n <= 12:
            row["exact17"] = RE.state_infid(bare, RE.strip_measure(o17))
        rows.append(row)
        print(f"{k:3d} {name}{n}{'m' if measured else ''} rel {trl:.3f} s c17 {t17:.3f} s identical "
              f"{row['identical']} skipped {skipped}", flush=True)
    head, dirty = git_state()
    meta = dict(device=args.device, smoke=bool(args.smoke), git_head=head, dirty_tracked=dirty,
                qiskit=qiskit.__version__, versions=dict(release=rel.VERSION, c17=c17.VERSION),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         c17=norm_sha(C17_PATH), readout_eval=norm_sha(RE.__file__),
                         full_heavyhex_cliff=norm_sha(os.path.join(HERE, "full_heavyhex_cliff.py"))),
                wall_s=round(time.time() - t00, 1))
    os.makedirs(args.out, exist_ok=True)
    fn = f"skip_{args.device}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"))
    print(f"wrote {fn}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    F, flags, heads = {}, [], set()
    for p in sorted(glob.glob(os.path.join(args.out, "skip_*.json"))):
        if p.endswith("_smoke.json"):
            continue
        r = json.load(open(p))
        m = r["meta"]
        heads.add(m["git_head"])
        want = 51 if m["device"] in SMALL27 else 48
        if m["smoke"] or m["dirty_tracked"] or m["versions"] != dict(release="2026-10-06.3", c17="2026-10-06.c17") \
                or len(r["rows"]) != want:
            flags.append(os.path.basename(p))
        F[m["device"]] = r["rows"]
    rows = [x for rs in F.values() for x in rs]
    err = sum(1 for x in rows if "error" in x)
    ok = [x for x in rows if "error" not in x]
    off = sum(1 for x in ok if x["off17"])
    inexact = sum(1 for x in ok if x.get("exact17", 0.0) > 1e-6)
    checked = sum(1 for x in ok if "exact17" in x)
    p0 = sorted(F) == sorted(DEVICES) and not flags and not err and not off and not inexact and len(heads) == 1 and \
        checked == 9 * len(DEVICES)  # ring, brick and qft at 10 qubits, 3 each, on every device
    out = [f"# SKIP score\n", f"files {len(F)}, git_head {sorted(heads)}, flags {flags}",
           f"P0 {'PASS' if p0 else 'FAIL'}: errors {err}, off-target {off}, inexact {inexact} "
           f"(checked {checked}, expected {9 * len(DEVICES)})"]
    if not p0:
        out.append("Nothing below is scored.")
        print("\n".join(out))
        open(os.path.join(args.out, "score.md"), "w").write("\n".join(out) + "\n")
        return
    out += ["", "| device | circuits | identical | above 16: median c17/release | up to 16: median c17/release |",
            "|---|---|---|---|---|"]
    above, upto, skips_ok = {}, {}, True
    for d in DEVICES:
        rs = F[d]
        a = [x["t_c17"] / x["t_rel"] for x in rs if x["n"] > 16]
        u = [x["t_c17"] / x["t_rel"] for x in rs if x["n"] <= 16]
        above[d], upto[d] = statistics.median(a), statistics.median(u)
        for x in rs:
            want = 1 if x["n"] > 16 else 0
            if x["skipped"]["floor"] != want or x["skipped"]["level3"] != want:
                skips_ok = False
        out.append(f"| {d} | {len(rs)} | {sum(x['identical'] for x in rs)} | {above[d]:.3f} | {upto[d]:.3f} |")
    ident = all(x["identical"] for x in rows)
    res = [("K1", "c17 returns the release's circuit", verdict(ident, not ident)),
           ("K2", "level 3 and the floor skipped once per circuit above 16 qubits, never up to 16",
            verdict(skips_ok, not skips_ok)),
           ("K3", "above 16: median per-circuit time ratio <= 0.5 on every device",
            verdict(all(v <= 0.5 for v in above.values()), any(v > 0.9 for v in above.values()))),
           ("K4", "up to 16: median per-circuit time ratio <= 1.10 on every device",
            verdict(all(v <= 1.10 for v in upto.values()), any(v > 1.25 for v in upto.values())))]
    out += ["", "| | prediction | verdict |", "|---|---|---|"] + [f"| {a} | {b} | **{c}** |" for a, b, c in res]
    out += ["", "Reported without prediction (median c17/release per family, above 16 | up to 16):"]
    for fam in FAMILIES + ("fullT",):
        a = [x["t_c17"] / x["t_rel"] for x in rows if x["family"] == fam and x["n"] > 16]
        u = [x["t_c17"] / x["t_rel"] for x in rows if x["family"] == fam and x["n"] <= 16]
        out.append(f"- {fam}: {statistics.median(a):.3f} ({len(a)}) | " +
                   (f"{statistics.median(u):.3f} ({len(u)})" if u else "-"))
    tot_r = sum(x["t_rel"] for x in rows)
    tot_c = sum(x["t_c17"] for x in rows)
    out.append(f"- total compile time: release {tot_r:.1f} s, c17 {tot_c:.1f} s ({tot_c / tot_r:.3f})")
    out.append(f"- resynthesis skipped on {sum(1 for x in rows if x['skipped']['resynthesis'])} circuits")
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
