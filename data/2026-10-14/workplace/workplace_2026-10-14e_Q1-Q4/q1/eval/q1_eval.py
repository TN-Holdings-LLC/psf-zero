"""q1_eval.py -- item 91 (embeddable interaction graphs by edge colouring) on c53, against c53, c52, Qiskit level 3
(QK3 on the device's Target, QK3F on the same Target with rzz and rx, folded). Default call compile_for_hardware(qc,
target=..., seed_transpiler=0). Correctness of the new output: exact distribution (n <= 20) or <Z_i>, <Z_i Z_i+1>
by MPS (n <= 80). python q1_eval.py <out dir> <device>"""
import json, os, sys, time, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_common import *  # noqa
from cliff_check import clifford_ising, check as cliff_check
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit_ibm_runtime import fake_provider
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
out_dir, dev = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
NEW = compiler(os.environ.get("PSF_NEW", REPO + "/patches/psf_compile_c53_91_2026-10-14/psf_compile.py"), "c53_91")
C53 = compiler(REPO + "/patches/psf_compile_c53_2026-10-14/psf_compile.py", "c53")
C52 = compiler(REPO + "/patches/psf_compile_c52_2026-10-14/psf_compile.py", "c52")
T = getattr(fake_provider, dev)().target; F = fractional(T)
K, kadj, fadj, _ = NEW._chain_lattice(T, 0.5)
start = min(K)
cases = []
EDGES = {}
for n in (18, 40, 80, len(K)):
    e = hh_patch(T, n, start, adj=kadj)
    EDGES[("hh_ising", n)] = e
    for s in (2, 4):
        cases.append((f"hh_ising", n, s, ising(n, e, s, signs=True)))
for n in (16, 40):
    e = bfs_tree(T, n, start, adj=kadj)
    EDGES[("tree_ising", n)] = e
    for s in (2, 4):
        cases.append(("tree_ising", n, s, ising(n, e, s, signs=True)))
for n in (18, 40):
    e = hh_patch(T, n, start, adj=kadj)
    for s in (1, 2):
        cases.append(("hh_xyz", n, s, heis_xyz(n, e, s)))
cases.append(("ring12_ising", 12, 2, ising(12, ring(12), 2, signs=True)))
cases.append(("ring12_xyz", 12, 2, heis_xyz(12, ring(12), 2)))
sys.path.insert(0, os.path.join(REPO, "data", "2026-10-12", "c48_dev"))
import gens
for nm in ("heis2d_n16_r1", "heis_n12_r2", "tfim_n12_r2"):
    cases.append((nm, int(nm.split("_n")[1].split("_")[0]), int(nm[-1]), gens.gen(nm)))
rows_path = os.path.join(out_dir, f"q1_rows_{dev}.jsonl")
done = set()
if os.path.exists(rows_path):
    for l in open(rows_path):
        r = json.loads(l); done.add((r["family"], r["n"], r["steps"], r["target"]))
fh = open(rows_path, "a")


def qk3(qc, tg, fold):
    o = transpile(qc, target=tg, optimization_level=3, seed_transpiler=0)
    if fold:
        raw_ok = rzz_ok(o)
        o2 = PassManager([FoldRzzAngle()]).run(o)
        ph = sum(float(i.operation.params[0]) for i in o2.data if i.operation.name == "global_phase")
        o2.data = [i for i in o2.data if i.operation.name != "global_phase"]; o2.global_phase += ph
        o2._layout = o._layout
        return o2
    return o


for fam, n, s, qc in cases:
    for tk, tg in (("T", T), ("F", F)):
        if (fam, n, s, tk) in done:
            continue
        row = dict(device=dev, family=fam, n=n, steps=s, target=tk)
        g0 = dict(NEW.GRAPH_STATS); k0 = (NEW.CHAIN_STATS["kept"], NEW.CHAIN_STATS["kept_default"])
        arms = [("new", lambda: NEW.compile_for_hardware(qc, target=tg, seed_transpiler=0)),
                ("c53", lambda: C53.compile_for_hardware(qc, target=tg, seed_transpiler=0)),
                ("c52", lambda: C52.compile_for_hardware(qc, target=tg, seed_transpiler=0)),
                ("QK3" if tk == "T" else "QK3F", lambda: qk3(qc, tg, tk == "F"))]
        outs = {}
        for a, fn in arms:
            t0 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    o = fn()
            except Exception as ex:  # noqa: BLE001
                row[a] = dict(error=repr(ex)[:160]); continue
            d, two, nr = dur_2q(o, tg)
            row[a] = dict(dur_us=round(d, 4), two_q=two, rzz=nr, in_range=rzz_ok(o), secs=round(time.perf_counter() - t0, 2))
            outs[a] = o
        row["item91"] = NEW.GRAPH_STATS["used"] - g0["used"]
        row["item87"] = NEW.CHAIN_STATS["used"] - 0 if False else None
        row["kept"] = NEW.CHAIN_STATS["kept"] - k0[0]; row["kept_default"] = NEW.CHAIN_STATS["kept_default"] - k0[1]
        row["graph_reason"] = {k: NEW.GRAPH_STATS[k] - g0[k] for k in ("degree", "no_embedding", "not_read") if NEW.GRAPH_STATS[k] != g0[k]}
        if "new" in outs:
            row["new"]["isa"] = bool(NEW._isa_ok(outs["new"], tg))
            try:
                row["new"]["err"] = W2.tvd(W2.probs(outs["new"]), W2.probs(W2.unrolled(qc))) if n <= 24 else None
            except Exception as ex:  # noqa: BLE001
                row["new"]["err"] = repr(ex)[:80]
            if n > 24 and fam in ("hh_ising", "tree_ising"):  # Clifford-angle twin, checked by stabilizer states
                try:
                    cq = clifford_ising(n, EDGES[(fam, n)], s)
                    with contextlib.redirect_stdout(io.StringIO()):
                        co = NEW.compile_for_hardware(cq, target=tg, seed_transpiler=0)
                    row["new"]["cliff_err"] = cliff_check(co, cq, EDGES[(fam, n)])
                except Exception as ex:  # noqa: BLE001
                    row["new"]["cliff_err"] = repr(ex)[:80]
        fh.write(json.dumps(row) + "\n"); fh.flush()
        def g(a):
            r = row.get(a, {})
            return f"{r['dur_us']:8.2f} ({r['two_q']:5d})" if "dur_us" in r else "   error       "
        q = "QK3" if tk == "T" else "QK3F"
        print(f"{dev} {fam:13s} n={n:3d} s={s} {tk} | new {g('new')} | c53 {g('c53')} | c52 {g('c52')} | {q} {g(q)} | 91:{row['item91']} kept {row['kept']}/{row['kept_default']} {row['graph_reason']} err {row.get('new', {}).get('err')} cliff {row.get('new', {}).get('cliff_err')}", flush=True)
