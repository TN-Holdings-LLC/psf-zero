"""a12_eval.py -- test SPEED (2026-10-06): does candidate psf_ai_compile 2026-10-06.a12 (changelog item 17: cached
scoring terms and no repeated re-placement of identical candidates) return exactly the adopted front end a11's
circuit, faster?

Circuits: MODEL-RO2's model-style generator (ai10_eval2._family, _model_style, unchanged; Addendum 352), 6 per
(family, n) cell (96), each with measure_all(); plus the first 2 per cell (32) without measurements: 128 per device.
Seeds 76,000,000 + k (smoke: 76,500,000 + k, 1 per cell, 16 + 16), none used before.
Devices: FakeTorino, FakeKingston (cz), FakeAuckland, FakeHanoiV2 (cx), FakeBrussels, FakeOsaka (ecr).
Per circuit: a warm-up call of a11 (discarded), then a11 and a12 in alternating order, each timed; their outputs
compared instruction by instruction (with global phase, layouts and clbits); a12's output checked for exactness (the
workplace probe's state infidelity, measurements removed, <= 1e-6) and for off-target two-qubit gates.

    python benchmarks/a12_eval.py run --device D --out DIR [--smoke]
    python benchmarks/a12_eval.py score --out DIR
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
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
A12_PATH = os.path.join(REPO, "patches", "psf_ai_compile_a12_2026-10-06", "psf_ai_compile.py")


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")  # before anything imports it
    a11 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_a11_speed")
    a12 = H.load_module(A12_PATH, "psf_ai_compile_a12_speed")
    E2 = H.load_module(os.path.join(WORK, "model_ro2", "ai10_eval2.py"), "ai10_eval2_speed")
    RE = H.load_module(os.path.join(WORK, "readout", "readout_eval.py"), "readout_eval_speed")
    if rel.VERSION != "2026-10-06.1" or a11.AI_COMPILE_VERSION != "2026-10-05.a11" or \
            a12.AI_COMPILE_VERSION != "2026-10-06.a12":
        raise SystemExit(f"STOP: versions {rel.VERSION}, {a11.AI_COMPILE_VERSION}, {a12.AI_COMPILE_VERSION}")
    return rel, a11, a12, E2, RE


def circuits(E2, smoke):
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    out, k = [], 0
    base = 76_500_000 if smoke else 76_000_000
    per = 1 if smoke else 6
    for name, ns in E2.FAMILIES:
        for n in ns:
            for j in range(per):
                rng = np.random.default_rng(base + k)
                k += 1
                qc = E2._model_style(E2._family(name, n, rng), rng)
                for q in range(n):
                    qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
                m = qc.copy()
                m.measure_all()
                out.append((f"{name}{n}", True, m, qc))
                if j < (1 if smoke else 2):
                    out.append((f"{name}{n}", False, qc, qc))
    return out


def full_sig(c):
    """Instructions (name, qubits, clbits, parameters), global phase, and where the logical qubits start and end.
    The device positions given to ancilla qubits in the layout are left out: they do not change the circuit."""
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)) if c.layout else None,
         list(c.layout.final_index_layout(filter_ancillas=True)) if c.layout else None]


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(args):
    import qiskit
    from qiskit_ibm_runtime import fake_provider
    rel, a11, a12, E2, RE = load()
    be = getattr(fake_provider, args.device)()
    t = be.target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]

    def call(A, qc):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            o = A.compile_for_model_circuit(qc, cm, basis, target=t)
            return o, time.perf_counter() - t0

    rows, t00 = [], time.time()
    for k, (name, measured, qc, bare) in enumerate(circuits(E2, args.smoke)):
        row = dict(name=name, measured=measured, n=bare.num_qubits)
        try:
            call(a11, qc)  # warm-up, discarded
            if k % 2 == 0:
                (o11, t11), (o12, t12) = call(a11, qc), call(a12, qc)
            else:
                (o12, t12), (o11, t11) = call(a12, qc), call(a11, qc)
        except Exception as e:  # recorded; P0 requires none
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            continue
        row.update(t11=round(t11, 5), t12=round(t12, 5), identical=full_sig(o11) == full_sig(o12),
                   exact12=RE.state_infid(bare, RE.strip_measure(o12)), off12=a12._off_target_2q(o12, t))
        rows.append(row)
    meta = dict(device=args.device, smoke=bool(args.smoke), git_head=git_head(), qiskit=qiskit.__version__,
                versions=dict(release=rel.VERSION, a11=a11.AI_COMPILE_VERSION, a12=a12.AI_COMPILE_VERSION),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         a11=norm_sha(a11.__file__), a12=norm_sha(A12_PATH), ai10_eval2=norm_sha(E2.__file__),
                         readout_eval=norm_sha(RE.__file__)),
                wall_s=round(time.time() - t00, 1))
    os.makedirs(args.out, exist_ok=True)
    fn = f"a12_{args.device}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, fn), "w"))
    print(f"wrote {fn}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    F = {}
    for p in sorted(glob.glob(os.path.join(args.out, "a12_*.json"))):
        r = json.load(open(p))
        F[r["meta"]["device"]] = r
    smoke = any(r["meta"]["smoke"] for r in F.values())
    want = 32 if smoke else 128
    lines = [f"# SPEED score{' (SMOKE)' if smoke else ''}", ""]
    rows = {d: F[d]["rows"] for d in F}
    errors = sum(1 for d in rows for r in rows[d] if "error" in r)
    inexact = sum(1 for d in rows for r in rows[d] if "error" not in r and r["exact12"] > 1e-6)
    off = sum(1 for d in rows for r in rows[d] if "error" not in r and r["off12"])
    counts = all(len(rows[d]) == want for d in rows)
    p0 = sorted(F) == sorted(DEVICES) and not errors and not inexact and not off and counts
    lines.append(f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(F)} of {len(DEVICES)}; errors {errors}; inexact a12 "
                 f"outputs {inexact}; off-target a12 outputs {off}; counts ok {counts}")
    if not p0:
        lines.append("Nothing below is scored.")
        out = "\n".join(lines) + "\n"
        print(out)
        open(os.path.join(args.out, "score.md"), "w").write(out)
        return
    lines += ["", "| device | identical | median a11 s | median a12 s | ratio of medians | median of ratios |",
              "|---|---|---|---|---|---|"]
    S = {}
    for d in DEVICES:
        rs = rows[d]
        m11, m12 = float(np.median([r["t11"] for r in rs])), float(np.median([r["t12"] for r in rs]))
        S[d] = dict(ident=sum(r["identical"] for r in rs) / len(rs), ratio=m12 / m11,
                    mr=float(np.median([r["t12"] / r["t11"] for r in rs])))
        lines.append(f"| {d} | {sum(r['identical'] for r in rs)}/{len(rs)} | {m11:.3f} | {m12:.3f} | {S[d]['ratio']:.3f} | "
                     f"{S[d]['mr']:.3f} |")
    V = []
    s1 = {d: S[d]["ident"] for d in DEVICES}
    V.append(("S1", "a12 returns a11's circuit (identical on every circuit of every device; refuted if any differs)",
              verdict(all(v == 1.0 for v in s1.values()), any(v < 1.0 for v in s1.values())), s1))
    s2 = {d: S[d]["ratio"] for d in CZ}
    V.append(("S2", "faster on the cz devices (median a12 / median a11 <= 0.75 on both; refuted > 1.00 on either)",
              verdict(all(v <= 0.75 for v in s2.values()), any(v > 1.0 for v in s2.values())), s2))
    s3 = {d: S[d]["ratio"] for d in CX + ECR}
    V.append(("S3", "not slower elsewhere (median ratio <= 0.90 on the cx and ecr devices; refuted > 1.05 on any)",
              verdict(all(v <= 0.9 for v in s3.values()), any(v > 1.05 for v in s3.values())), s3))
    lines += ["", "## Predictions", ""]
    for qid, text, v, num in V:
        lines.append(f"- {qid} ({text}): **{v}** -- {json.dumps(num, default=lambda x: round(x, 4))}")
    lines += ["", "Reported: median of per-circuit ratios a12 / a11: " +
              json.dumps({d: round(S[d]["mr"], 3) for d in DEVICES})]
    out = "\n".join(lines) + "\n"
    print(out)
    open(os.path.join(args.out, "score.md"), "w").write(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", required=True, choices=DEVICES)
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
