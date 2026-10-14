"""p1_eval.py -- items 87-88 on c51 against c49, c51 (default call) and Qiskit level 3 given the same fractional
Target (QK3F: the Pauli evolution; QK3F-BW: the hand brickwork), each folded into IBM's rzz range where needed.
W2's 48 TFIM cells, Heisenberg chains, TFIM with random signs. Default call: compile_for_hardware(qc, target=F,
seed_transpiler=0). Correctness of item 87-88's output: exact distribution (n <= 20) or all one- and two-point Z
correlations (MPS, n > 20). Measured: correctness, two-qubit count (CZ + rzz), ASAP duration on the Target.
    python p1_eval.py <out dir> [quick]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit_ibm_runtime import fake_provider

out_dir = sys.argv[1]; quick = len(sys.argv) > 2
os.makedirs(out_dir, exist_ok=True)
NEW = compiler(os.environ.get("PSF_NEW", os.path.join(REPO, "patches", "psf_compile_c51_87_88_2026-10-14", "psf_compile.py")),
               "c51_87_88")
C51 = compiler(REPO + "/patches/psf_compile_c51_2026-10-14/psf_compile.py", "c51")
C49 = compiler(REPO + "/patches/psf_compile_c49_2026-10-14/psf_compile.py", "c49")
fh = open(os.path.join(out_dir, "p1_rows.jsonl"), "a")


def cases():
    for n in (12, 16, 24, 32, 50, 100):
        for s in (2, 4, 8, 10):
            yield "tfim", n, s, W2.tfim_pauli(n, s), W2.tfim_brickwork(n, s)
    for n in (12, 16, 24):
        for s in (2, 4):
            yield "heis", n, s, heis_bw(n, s), None
    for n in (12, 16, 24, 50):
        for s in (2, 4, 8):
            yield "tfim_signs", n, s, tfim_signs(n, s), None


def heis_bw(n, steps, seed=1):
    rng = np.random.default_rng(seed)
    terms = []
    for s0 in (0, 1):
        for q in range(s0, n - 1, 2):
            for p in "XYZ":
                terms.append((lab(n, {q: p, q + 1: p}), float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.5))))
    terms += [(lab(n, {q: "Z"}), float(rng.uniform(-1, 1))) for q in range(n)]
    return pauli_chain(n, steps, terms, t=0.8, order=1)


def run(label, fn):
    t0 = time.perf_counter()
    try:
        o = fn()
    except Exception as e:  # noqa: BLE001
        return None, dict(error=repr(e)[:200]), time.perf_counter() - t0
    return o, None, time.perf_counter() - t0


for dev in ("FakeFez", "FakeMarrakesh"):
    b = getattr(fake_provider, dev)(); T = b.target; F = fractional(T)
    for fam, n, s, qc, bw in cases():
        if quick and n > 24:
            continue
        if fam != "tfim" and dev == "FakeMarrakesh" and n > 24:
            continue
        row = dict(device=dev, family=fam, n=n, steps=s)
        u0 = dict(NEW.CHAIN_STATS)
        arms = [("new", lambda: NEW.compile_for_hardware(qc, target=F, seed_transpiler=0)),
                ("c51", lambda: C51.compile_for_hardware(qc, target=F, seed_transpiler=0)),
                ("c49", lambda: C49.compile_for_hardware(qc, target=T, seed_transpiler=0)),
                ("QK3F", lambda: qk3f_bw(qc, F)[1])]
        if bw is not None:
            arms.append(("QK3F-BW", lambda: qk3f_bw(bw, F)[1]))
        outs = {}
        for a, fn in arms:
            o, e, secs = run(a, fn)
            if o is None:
                row[a] = e; continue
            tgt = T if a == "c49" else F
            d, two, nr = dur_2q(o, tgt)
            row[a] = dict(dur_us=round(d, 4), two_q=two, rzz=nr, in_range=rzz_ok(o), secs=round(secs, 2))
            outs[a] = o
        row["new_path"] = {k: NEW.CHAIN_STATS[k] - u0[k] for k in ("kept", "kept_default", "frame") if NEW.CHAIN_STATS[k] != u0[k]}
        if "new" in outs:
            o = outs["new"]
            row["new"]["isa"] = bool(NEW._isa_ok(o, F))
            try:
                row["new"]["err"] = err_dist(o, qc, n) if n <= 50 else None
            except Exception as e:  # noqa: BLE001
                row["new"]["err"] = repr(e)[:100]
        fh.write(json.dumps(row) + "\n"); fh.flush()
        def g(a):
            r = row.get(a, {})
            return f"{r.get('dur_us', float('nan')):7.2f} ({r.get('two_q', -1):5d})" if "dur_us" in r else "   err      "
        print(f"{dev:13s} {fam:10s} n={n:3d} s={s:2d} | new {g('new')} | c51 {g('c51')} | c49 {g('c49')} | QK3F {g('QK3F')} | BW {g('QK3F-BW')} | err {row.get('new', {}).get('err')} {row['new_path']}", flush=True)
