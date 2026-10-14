"""n5_ablate.py -- N5 (2026-10-14): c49's default call (C49-VAL's call: entangling_basis="cx", layout_search=True,
seed_transpiler=0, backend=...) with one estimate-based part removed at a time, on C48-VAL's 16 tests, FakeFez and
FakeTorino. Measured: CZ count and duration (measurements left out), plus whether the output changed.
  base       c49 as it is
  no33_60    placement_refine=False (item 33's re-placement by the Target's errors, default since item 60)
  no64       routing_optimization_level=1 (item 64 sets 3 when the device is given)
  no75score  item 75's re-placement applies nothing (as if every placement scored the same: the current one kept)
  no78       item 78's routing-stage re-placement off (item 75 gives up at the routing stage, as before c45)
    python n5_ablate.py <out>     (from the repository root)"""
import contextlib, hashlib, io, json, os, sys, time, warnings, importlib.util
warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-12", "c48_dev")]
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
P = load(os.path.join("patches", "psf_compile_c49_2026-10-14", "psf_compile.py"), "c49_n5"); P.WARN_WITHOUT_TARGET = False
H = load("benchmarks/hw1_real.py", "hw1"); H.KIT = os.path.join("data", "2026-10-12", "hw_kit")
import gens
from qiskit_ibm_runtime import fake_provider
TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12",
         "tfim_n20_r3", "heis_n12_r2", "heis2d_n16_r1", "chem_n12_r1", "qaoa3reg_n20_r3", "qaoa3reg_n40_r3", "tfim_n40_r2")
VARIANTS = ("base", "no33_60", "no64", "no75score", "no78")
_orig_run = P._DeterministicPostLayout.run
_orig_init = P._DeterministicVF2.__init__


def _no_apply(self, dag):
    from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason
    self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.NO_SOLUTION_FOUND


def _no_routing_stage(self, target, routing_stage=False):
    _orig_init(self, target, routing_stage=False)


def build(name):
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def metrics(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock, n2 = {}, 0
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = t[nm].get(qs) if nm in t.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    return n2, (max(clock.values()) if clock else 0.0) * 1e6


def sig(c):
    h = hashlib.sha256()
    for ins in c.data:
        h.update(ins.operation.name.encode()); h.update(repr([c.find_bit(q).index for q in ins.qubits]).encode())
        h.update(repr([round(float(p), 10) for p in ins.operation.params]).encode())
    return h.hexdigest()[:16]


def run(variant, qc, b):
    kw = dict(backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    if variant == "no33_60":
        kw["placement_refine"] = False
    if variant == "no64":
        kw["routing_optimization_level"] = 1
    P._DeterministicPostLayout.run = _no_apply if variant == "no75score" else _orig_run
    P._DeterministicVF2.__init__ = _no_routing_stage if variant == "no78" else _orig_init
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return P.compile_for_hardware(qc, **kw)
    finally:
        P._DeterministicPostLayout.run, P._DeterministicVF2.__init__ = _orig_run, _orig_init


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    fh = open(os.path.join(out, "n5_rows.jsonl"), "w")
    for dev in ("FakeFez", "FakeTorino"):
        b = getattr(fake_provider, dev)(); t = b.target
        for name in TESTS:
            qc = build(name)
            row = dict(device=dev, test=name)
            for v in VARIANTS:
                t0 = time.perf_counter()
                try:
                    c = run(v, qc, b)
                    n2, d = metrics(c, t)
                    row[v] = dict(cz=n2, dur_us=round(d, 3), sig=sig(c), secs=round(time.perf_counter() - t0, 1))
                except Exception as exc:  # noqa: BLE001
                    row[v] = dict(error=f"{type(exc).__name__}: {str(exc)[:100]}")
            fh.write(json.dumps(row) + "\n"); fh.flush(); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
