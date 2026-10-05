"""Exploration on DEV data (split seed 2, the DEPTH dry run's thetas): does compiling the classifier WITH its final
measurement, and c13's readout term, move the measured qubit to a better-readout qubit? Not a test."""
import contextlib, io, json, sys, warnings, os, time
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "<workdir>/qml2")
import depth_eval as DE
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider as fp


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


sys.path.insert(0, HERE)
C12 = load("<workdir>/qml2/cand/psf_compile.py", "psf_c12")
C13 = load(os.path.join(HERE, "psf_compile.py"), "psf_c13")


def measured(qc):
    m = QuantumCircuit(qc.num_qubits, 1)
    m.compose(qc, inplace=True)
    m.measure(0, 0)
    return m


def run(dev, ds, n, L, npts):
    be = getattr(fp, dev)(); t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    full = dict(coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx", layout_search=True,
                seed_transpiler=0, target=t, placement_refine=True, final_resynthesis="select", compare_level3=True,
                compare_floor=True, candidate_score="hybrid")
    arms = {"C12": lambda qc: C12.compile_for_hardware(qc, **full),
            "C12M": lambda qc: C12.compile_for_hardware(measured(qc), **full),
            "C13M": lambda qc: C13.compile_for_hardware(measured(qc), **full),
            "L3TM": lambda qc: transpile(measured(qc), target=t, optimization_level=3, seed_transpiler=0,
                                         approximation_degree=1.0)}
    Xtr, ytr, Xte, yte = DE.data(ds, n, 2)
    th = np.load(f"<workdir>/qml2/dry1/theta_{ds}_n{n}_L{L}.npy")
    res = {}
    for a, f in arms.items():
        ro, n2 = [], []
        t0 = time.perf_counter()
        for x in Xte[:npts]:
            with contextlib.redirect_stdout(io.StringIO()):
                out = f(DE.circuit(x, th, n, L))
            mq = [out.find_bit(i.qubits[0]).index for i in out.data if i.operation.name == "measure"]
            q = mq[0] if mq else out.layout.final_index_layout(filter_ancillas=True)[0]
            ro.append(t["measure"][(q,)].error)
            n2.append(sum(1 for i in out.data if len(i.qubits) == 2))
        res[a] = dict(readout=float(np.mean(ro)), n2q=float(np.mean(n2)), s=round((time.perf_counter() - t0) / npts, 3))
    print(dev, ds, n, L, json.dumps(res), flush=True)


for dev in ("FakeTorino", "FakeKingston"):
    for n in (4, 6):
        for L in (4, 12):
            run(dev, "BC", n, L, 12)
