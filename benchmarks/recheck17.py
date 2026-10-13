"""RECHECK-17: the 17 test-device pairs that C42-VAL's psf arms refused (FailedElementsError), compiled again by
QK3 (Qiskit level 3), R3 (release 2026-10-11.3), R3-noVF2 (2026-10-11.3 with Qiskit's VF2PostLayout replaced by nothing)
and R4 (release 2026-10-11.4). One pair per process.
    cd <psf-zero repository>; python <this file> run --bp <benchpress> --pairs PAIRS.json --tests C42_TESTS.json --out OUT.jsonl
"""
import argparse, contextlib, io, json, math, os, subprocess, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor

REPO = os.getcwd()


def one(a):
    warnings.simplefilter("ignore")
    sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
    os.environ["C42_TESTS"] = a.tests
    import c42_val as V
    from qiskit_ibm_runtime import fake_provider
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.transpiler.basepasses import AnalysisPass
    test, dev = json.loads(a.job)
    qc = V.build(a.bp, test)
    b = getattr(fake_provider, dev)()
    t = b.target
    V.load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    r3 = V.load(os.path.join(REPO, "patches", "psf_compile_release_2026-10-11.3", "psf_compile.py"), "r3")
    r4 = V.load(os.path.join(REPO, "psf_compile.py"), "r4")

    class Nothing(AnalysisPass):
        def run(self, dag):
            pass

    @contextlib.contextmanager
    def no_vf2():
        import qiskit.transpiler.preset_passmanagers.builtin_plugins as p1
        import qiskit.transpiler.preset_passmanagers.common as p2
        saved = (p1.VF2PostLayout, p2.VF2PostLayout)
        p1.VF2PostLayout = p2.VF2PostLayout = lambda *x, **k: Nothing()
        try:
            yield
        finally:
            p1.VF2PostLayout, p2.VF2PostLayout = saved

    for arm in ("QK3", "R3", "R3-noVF2", "R4"):
        rec = dict(test=test, device=dev, arm=arm)
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if arm == "QK3":
                    out = generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
                else:
                    mod = r4 if arm == "R4" else r3
                    ctx = no_vf2() if arm == "R3-noVF2" else contextlib.nullcontext()
                    with ctx:
                        out = mod.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True,
                                                       seed_transpiler=0)
            edges, bad = r4._failed_elements(t, 0.5)
            cost = r4._st_cost(out, t)
            rec.update(ok=True, failed=bool(r4._uses_failed(out, edges, bad)), esp=math.exp(-cost) if cost < math.inf else 0.0,
                       n2=sum(1 for i in out.data if len(i.qubits) == 2))
        except Exception as e:  # noqa: BLE001 - recorded
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:160])
        rec["t"] = round(time.perf_counter() - t0, 2)
        print(json.dumps(rec), flush=True)


def run(a):
    pairs = json.load(open(a.pairs))
    if os.path.exists(a.out):
        sys.exit("STOP: output exists")

    def go(p):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "one", "--bp", a.bp, "--tests", a.tests,
                            "--job", json.dumps(p)], capture_output=True, text=True, timeout=3600, cwd=REPO)
        return [l for l in r.stdout.splitlines() if l.startswith("{")] or [json.dumps(dict(test=p[0], device=p[1], error=r.stderr[-300:]))]
    with ThreadPoolExecutor(2) as ex, open(a.out, "w") as fh:
        for lines in ex.map(go, pairs):
            fh.write("\n".join(lines) + "\n")
            fh.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "one"))
    ap.add_argument("--bp"); ap.add_argument("--pairs"); ap.add_argument("--tests"); ap.add_argument("--out"); ap.add_argument("--job")
    a = ap.parse_args()
    {"run": run, "one": one}[a.mode](a)
