"""Home check (2026-11-13): psf_compile 2026-10-18.1 against the proposed 2026-10-18.2 patch, output fingerprints bit
for bit (C68-VAL's fp), on C68-VAL's 25 tests, the build input and 2 draws each, per device, with and without psfz.
Where outputs differ, the new output is checked for equivalence with the old one (Operator on the active qubits, up
to 12). usage (PSF-Zero root): python compare.py OLD.py NEW.py OUT.jsonl"""
import contextlib, hashlib, io, json, os, random, runpy, sys, tempfile, time, warnings
warnings.simplefilter("ignore")
OLD, NEW_, OUT = sys.argv[1:4]
_tmp = tempfile.mkdtemp()
_argv = sys.argv
sys.argv = ["psfz_val", os.path.join(_tmp, "none.jsonl")]
with contextlib.redirect_stdout(io.StringIO()):
    P = runpy.run_path(os.path.join("data", "2026-10-17", "psfz_val", "psfz_val.py"), run_name="cmp_import")
sys.argv = _argv
g = P["g"]
from qiskit_ibm_runtime import fake_provider
from qiskit.quantum_info import Operator
with contextlib.redirect_stdout(io.StringIO()):
    A = g["load"](OLD, "pc_old")
    B = g["load"](NEW_, "pc_new")
for M in (A, B):
    M.WARN_WITHOUT_TARGET = False
print("versions", A.VERSION, B.VERSION, flush=True)
FORMULA = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2", "hhI_n16",
           "hhI_n20", "ring12_tfim", "hhtree_heis_n16", "ghz_n12")
NATIVE = ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16")
TESTS = FORMULA + NATIVE + P["NEW"]


def build(name):
    if name in P["NEW"]:
        return P["build_new"](name)
    return g["build"](name) if name in FORMULA else P["TT"].build(name)


def fp(c):
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    ops = tuple((ci.operation.name, tuple(q[x] for x in ci.qubits), tuple(k[x] for x in ci.clbits),
                 tuple(str(p) if hasattr(p, "parameters") else float(p).hex() for p in ci.operation.params))
                for ci in c.data)
    lay = getattr(c, "_layout", None)
    il = None if lay is None else tuple(lay.initial_index_layout(filter_ancillas=False))
    fl = None if lay is None or lay.final_layout is None else tuple(lay.final_index_layout())
    return hashlib.sha256(repr((ops, float(c.global_phase).hex(), il, fl, c.num_qubits, c.num_clbits)).encode()).hexdigest()


def run(M, qc, t):
    s = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = M.compile_for_hardware(qc, target=t)
        return out, time.perf_counter() - s, None
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - s, f"{type(e).__name__}: {str(e)[:120]}"


def active_op(c):
    act = sorted({c.find_bit(q).index for ci in c.data if ci.operation.name not in ("barrier", "measure") for q in ci.qubits})
    if len(act) > 12:
        return None
    from qiskit import QuantumCircuit
    pos = {p: i for i, p in enumerate(act)}
    r = QuantumCircuit(len(act), global_phase=c.global_phase)
    for ci in c.data:
        if ci.operation.name in ("barrier", "measure"):
            continue
        r.append(ci.operation, [pos[c.find_bit(q).index] for q in ci.qubits])
    return act, Operator(r)


DEVICES = [("fez", fake_provider.FakeFez()), ("marrakesh", fake_provider.FakeMarrakesh()),
           ("sherbrooke", fake_provider.FakeSherbrooke()), ("brisbane", fake_provider.FakeBrisbane())]
BIN = os.path.abspath(os.path.join("rust", "psfz", "target", "release", "psfz"))
with open(OUT, "w") as f:
    for mode, binpath in (("psfz", BIN), ("nopsfz", "/nonexistent/psfz")):
        os.environ["PSFZ_BIN"] = binpath
        for dname, be in DEVICES:
            t = be.target
            for test in TESTS:
                base = build(test)
                rng = random.Random(f"home-cmp|{dname}|{test}")
                inputs = [("build", base), ("same", P["randomize"](base, rng, same=True)), ("free", P["randomize"](base, rng))]
                for kind, qc in inputs:
                    oa, ta, ea = run(A, qc, t)
                    ob, tb, eb = run(B, qc, t)
                    fa = None if oa is None else fp(oa)
                    fb = None if ob is None else fp(ob)
                    rec = dict(mode=mode, device=dname, test=test, kind=kind, same=(fa == fb and ea == eb),
                               err_old=ea, err_new=eb, t_old=round(ta, 4), t_new=round(tb, 4))
                    if oa is not None and ob is not None and fa != fb:
                        xa, xb = active_op(oa), active_op(ob)
                        if xa and xb and xa[0] == xb[0]:
                            rec["equivalent"] = bool(xa[1].equiv(xb[1]))
                        else:
                            rec["equivalent"] = None
                        rec["n_ops"] = (len(oa.data), len(ob.data))
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            print(mode, dname, "done", flush=True)
