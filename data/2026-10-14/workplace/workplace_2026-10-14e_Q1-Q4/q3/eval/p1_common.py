"""p1_common.py -- shared helpers for the P1 checks (items 87-88 on c51)."""
import copy, importlib.util, math, os, sys, warnings
import numpy as np
warnings.simplefilter("ignore")
REPO = os.environ.get("PSF_REPO", os.getcwd())  # run from the repository root
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-14", "workplace", "workplace_2026-10-14_W1-W2-W3-W5-W7", "w2"),
                os.path.join(REPO, "benchmarks")]
import w2_tfim as W2  # noqa: E402


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m
    s.loader.exec_module(m); return m


load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")


def compiler(path, name):
    m = load(path, name); m.WARN_WITHOUT_TARGET = False; return m


def fractional(target):
    """test_c51's: rzz on every cz coupler (a CZ's duration and error), rx on every qubit (an sx's)."""
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    t.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in target["cz"].items()})
    t.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in target["sx"].items()})
    return t


def lab(n, d):
    s = ["I"] * n
    for q, p in d.items():
        s[n - 1 - q] = p
    return "".join(s)


def pauli_chain(n, steps, terms, t=1.0, measure=True, order=2):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter, LieTrotter
    qc = QuantumCircuit(n)
    syn = SuzukiTrotter(order=2, reps=steps) if order == 2 else LieTrotter(reps=steps)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=t, synthesis=syn), range(n))
    if measure:
        qc.measure_all()
    return qc


def heis(n, steps, measure=True, seed=1):
    rng = np.random.default_rng(seed)
    terms = []
    for q in range(n - 1):
        for p in "XYZ":
            terms.append((lab(n, {q: p, q + 1: p}), float(rng.uniform(0.5, 1.5))))
    for q in range(n):
        terms.append((lab(n, {q: "Z"}), float(rng.uniform(-1, 1))))
    return pauli_chain(n, steps, terms, t=0.8, measure=measure)


def tfim_signs(n, steps, measure=True, seed=2):
    """TFIM with random signs and sizes: J_i in +-[0.3, 2.5] (some above pi/2 per step after scaling), h_i in +-[0.2, 1.5]"""
    rng = np.random.default_rng(seed)
    terms = [(lab(n, {q: "Z", q + 1: "Z"}), float(rng.choice([-1, 1]) * rng.uniform(0.3, 2.5))) for q in range(n - 1)]
    terms += [(lab(n, {q: "X"}), float(rng.choice([-1, 1]) * rng.uniform(0.2, 1.5))) for q in range(n)]
    return pauli_chain(n, steps, terms, t=1.2, measure=measure)


def dur_2q(out, target):
    if any(len(i.qubits) == 0 for i in out.data):  # FoldRzzAngle may leave a GlobalPhaseGate (no qubits, no time)
        out = out.copy()
        out.data = [i for i in out.data if len(i.qubits) > 0]
    d, c, _ = W2.duration_and_cz(out, target)
    two = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
    nrzz = sum(1 for i in out.data if i.operation.name == "rzz")
    return d, two, nrzz


def rzz_ok(out):
    return all(0 < float(i.operation.params[0]) <= math.pi / 2 + 1e-12 for i in out.data if i.operation.name == "rzz")


def no_meas_flip(out):
    """no classical post-processing: the circuit has no classical control and no measurement into a register other
    than its own clbits in order (checked by the distribution test)."""
    return all(i.operation.name not in ("if_else", "c_if") for i in out.data)


def err_dist(out, qc, n):
    ref_c = W2.unrolled(qc)
    if n <= 20:
        return W2.tvd(W2.probs(out), W2.probs(ref_c))
    return float(np.max(np.abs(W2.corr(out, n) - W2.corr(ref_c, n))))


def qk3f_bw(qc_bw, target, seed=0):
    """Qiskit level 3 given the hand brickwork and the fractional Target, then folded (FoldRzzAngle) to be executable."""
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    raw = transpile(qc_bw, target=target, optimization_level=3, seed_transpiler=seed)
    fold = PassManager([FoldRzzAngle(), Optimize1qGatesDecomposition(target=target)]).run(raw)
    fold._layout = raw._layout
    return raw, fold
