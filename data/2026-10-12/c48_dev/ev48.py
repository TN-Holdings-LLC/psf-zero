"""c48 development: PSF-Zero (c47 without mirror = c45's STEINER6 path) with and without the time-aware reordering,
against Qiskit level 3, on fake devices. Columns: two-qubit count, ESP, duration (ASAP on the Target's durations),
the idle-aware estimate (HW1's exploratory form), and the mover-level makespan in CX units."""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import os, sys, os, io, math, time, contextlib, importlib.util, warnings
warnings.simplefilter("ignore")
sys.path[:0] = ["benchmarks", ".", "data/2026-10-12/c48_dev"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M = load(os.environ.get("PSFC", "patches/psf_compile_c47_2026-10-12/psf_compile.py"), "psfm")
H = load("benchmarks/hw1_real.py", "hw1")
H.KIT = os.path.join("data", "2026-10-12", "hw_kit")  # its qasm/ (the kit as committed with Addendum 455)
import sched as S  # noqa: E402
import gens  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

S.install(M)
M._st_mirror_blocks = lambda triples, limit=4000: None  # c45's STEINER6 path (no item 82)
RULE = {"rule": None}
CHECK = os.environ.get("CHECK") == "1"
MOVER_LOG = []
_orig_build = M._StMover2.build


def post(self):
    before = S.asap(self.ops)
    if RULE["rule"]:
        new = S.reorder(self.ops, self.__dict__.get("blocks", []), RULE["rule"])
        if new is not None and S.asap(new) < before - 1e-9:
            self.ops = new
    MOVER_LOG.append((before, S.asap(self.ops)))


def build(self):
    _orig_build(self)
    post(self)
    return self


M._StMover2.build = build
BASE = M._StMover2
import mover3  # noqa: E402
C3 = mover3.make(M, 1.0)
_b3 = C3.build


def build3(self):
    _b3(self)
    post(self)
    return self


C3.build = build3


def set_arm(arm):
    """'PSF', 'PSF+start', 'PSF+k1', 'PSF+k1+start', ...: the reordering rule and the SWAP time weight."""
    toks = arm.split("+")[1:]
    RULE["rule"] = next((t for t in toks if t in ("start", "cp")), None)
    k = next((float(t[1:]) for t in toks if t.startswith("k")), None)
    if k is None:
        M._StMover2 = BASE
    else:
        C3.KAPPA = k
        M._StMover2 = C3


def metrics(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock, first = {}, {}
    n2 = 0
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = t[nm].get(qs) if nm in t.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s)
            clock[q] = s + d
        n2 += len(qs) == 2
    T = max(clock.values())
    ln_idle = 0.0
    for q in clock:
        p = t.qubit_properties[q]
        ln_idle += (T - first[q]) * (1 / (4 * p.t1) + 1 / (2 * p.t2))
    ln_esp = M._st_cost(c, t)
    return n2, math.exp(-ln_esp), T * 1e6, math.exp(-ln_esp - ln_idle)


def psf(qc, b):
    with contextlib.redirect_stdout(io.StringIO()):
        return M.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                                      level3_candidate=False)


if __name__ == "__main__":
    tests = sys.argv[1].split(",")
    devs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["FakeKingston"]
    rules = sys.argv[3].split(",") if len(sys.argv) > 3 else ["start"]
    print(f"{'test':12s} {'dev':8s} {'arm':10s} {'2q':>5s} {'ESP':>7s} {'dur us':>7s} {'idle-est':>8s} {'mover':>13s} {'s':>5s}")
    for d in devs:
        b = getattr(fake_provider, d)()
        t = b.target
        for name in tests:
            qc = H.build_test(name)[0] if name in H.TESTS else gens.build(name)
            q3 = generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
            base = None
            for arm in ["QK3", "PSF"] + ["PSF+" + r for r in rules]:  # rules: e.g. start,k1+start
                t0 = time.time()
                MOVER_LOG.clear()
                if arm == "QK3":
                    c = q3
                else:
                    set_arm(arm)
                    c = psf(qc, b)
                m = metrics(c, t)
                ex = ""
                if CHECK and arm != "QK3":
                    act = len({q for i in c.data if i.operation.name not in ("barrier",) for q in i.qubits})
                    if act <= 20:
                        p0, _ = H.measured_distribution(H._unrolled(qc)); p1, _ = H.measured_distribution(c)
                        ex = f" exact {max(abs(p0.get(k, 0) - p1.get(k, 0)) for k in set(p0) | set(p1)):.1e}"
                    else:
                        ex = " exact n/a" 
                mv = f"{MOVER_LOG[-1][0]:.0f}->{MOVER_LOG[-1][1]:.0f}" if MOVER_LOG and arm != "QK3" else ""
                if arm == "QK3":
                    base = m
                print(f"{name:12s} {d[4:]:8s} {arm:10s} {m[0]:5d} {m[1]:7.3f} {m[2]:7.2f} {m[3]:8.4f} {mv:>13s} "
                      f"{time.time()-t0:5.1f}  ratio ESP {m[1]/base[1]:.3f} idle {m[3]/base[3]:.3f}{ex}", flush=True)
