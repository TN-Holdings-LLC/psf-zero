"""c48 development: prototype c48 with item 83 on and off (off = c47), against Qiskit level 3, on fake devices.
Columns: two-qubit count, ESP, duration (ASAP on the Target's durations), idle-aware estimate, exactness."""
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
M = load(os.environ.get("PSFC", "patches/psf_compile_c48_2026-10-12/psf_compile.py"), "psfm")
M.WARN_WITHOUT_TARGET = False
H = load("benchmarks/hw1_real.py", "hw1")
H.KIT = os.path.join("data", "2026-10-12", "hw_kit")  # its qasm/ (the kit as committed with Addendum 455)
import gens  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402


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
    ln_idle = sum((T - first[q]) * (1 / (4 * t.qubit_properties[q].t1) + 1 / (2 * t.qubit_properties[q].t2))
                  for q in clock)
    ln_esp = M._st_cost(c, t)
    return n2, math.exp(-ln_esp), T * 1e6, math.exp(-ln_esp - ln_idle)


def build(name):
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def compile_arm(arm, qc, b):
    if arm == "QK3":
        return generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
    M.ST_SCHEDULE = arm == "C48"
    with contextlib.redirect_stdout(io.StringIO()):
        return M.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                                      level3_candidate=False)


def exactness(qc, c, limit=20):
    act = len({q for i in c.data if i.operation.name != "barrier" for q in i.qubits})
    if act > limit:
        return float("nan")
    p0, _ = H.measured_distribution(H._unrolled(qc))
    p1, _ = H.measured_distribution(c)
    return max(abs(p0.get(k, 0) - p1.get(k, 0)) for k in set(p0) | set(p1))


if __name__ == "__main__":
    tests, devs = sys.argv[1].split(","), sys.argv[2].split(",")
    check = os.environ.get("CHECK") == "1"
    for d in devs:
        b = getattr(fake_provider, d)()
        for name in tests:
            qc = build(name)
            base = None
            for arm in ("QK3", "C47", "C48"):
                t0 = time.time()
                c = compile_arm(arm, qc, b)
                m = metrics(c, b.target)
                base = base or m
                ex = f" exact {exactness(qc, c):.1e}" if check and arm != "QK3" else ""
                print(f"{name:16s} {d[4:]:9s} {arm:4s} 2q {m[0]:5d} ESP {m[1]:.3f} dur {m[2]:6.2f} idle-est {m[3]:.4f} | "
                      f"vs QK3 ESP {m[1]/base[1]:.3f} dur {m[2]/base[2]:.3f} idle {m[3]/max(base[3],1e-300):.3f}{ex} | "
                      f"{time.time()-t0:.1f}s", flush=True)
