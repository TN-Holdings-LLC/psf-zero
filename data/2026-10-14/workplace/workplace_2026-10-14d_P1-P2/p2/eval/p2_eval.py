"""p2_eval.py -- W3's 64 classifier cells (FakeFez, FakeTorino; n = 4, 6; L = 1-16; W3's angles) through one front
end or Qiskit. python p2_eval.py <psf_compile.py> <front end .py | QK3> <label> <T|F> <out dir>
T: the device's own Target; F: with rzz and rx added (test_c51's construction). Rows: duration (ASAP on the Target),
two-qubit count, rzz count, exact outcome distribution (TVD), chosen candidate."""
import contextlib, io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
pcf, fe, label, tk, out = sys.argv[1:6]
P = load(pcf, "psf_compile"); P.WARN_WITHOUT_TARGET = False
A = None if fe == "QK3" else load(fe, "front_end")
D = load(REPO + "/data/2026-10-05/workplace/depth1/depth_eval.py", "depth_eval")
from qiskit import transpile
from qiskit_ibm_runtime import fake_provider
rng = np.random.default_rng(20261014)
angles = {(n, L): (rng.uniform(-1, 1, n), rng.uniform(-np.pi, np.pi, D.n_params(n, L))) for n in (4, 6) for L in range(1, 17)}
os.makedirs(out, exist_ok=True)
fh = open(os.path.join(out, f"p2_{label}_{tk}.jsonl"), "w")
for dev in ("FakeFez", "FakeTorino"):
    b = getattr(fake_provider, dev)(); T0 = b.target
    tg = fractional(T0) if tk == "F" else T0
    basis = [g for g in tg.operation_names if g in ("cz", "rz", "sx", "x")]
    for n in (4, 6):
        for L in range(1, 17):
            x, th = angles[(n, L)]
            qc = D.circuit(x, th, n, L); qc.measure_all()
            t0 = time.perf_counter(); chosen = None
            with contextlib.redirect_stdout(io.StringIO()):
                if A is None:
                    c = transpile(qc, target=tg, optimization_level=3, seed_transpiler=0)
                    if tk == "F":
                        from qiskit.transpiler import PassManager
                        from qiskit.transpiler.passes import Optimize1qGatesDecomposition
                        from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
                        raw_ok = rzz_ok(c)
                        c2 = PassManager([FoldRzzAngle(), Optimize1qGatesDecomposition(target=tg)]).run(c)
                        ph = sum(float(i.operation.params[0]) for i in c2.data if i.operation.name == "global_phase")
                        c2.data = [i for i in c2.data if i.operation.name != "global_phase"]
                        c2.global_phase += ph
                        c2._layout = c._layout; c = c2
                        chosen = f"raw_in_range={raw_ok}"
                else:
                    c, info = A.compile_for_model_circuit(qc, tg.build_coupling_map(), basis, target=tg, return_info=True)
                    chosen = info.get("chosen") if isinstance(info, dict) else None
            secs = time.perf_counter() - t0
            d, two, nr = dur_2q(c, tg)
            err = W2.tvd(W2.probs(c), W2.probs(qc))
            r = dict(device=dev, n=n, L=L, arm=label, target=tk, dur_us=round(d, 4), two_q=two, rzz=nr,
                     in_range=rzz_ok(c), err=err, chosen=chosen, secs=round(secs, 1))
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(json.dumps(r), flush=True)
