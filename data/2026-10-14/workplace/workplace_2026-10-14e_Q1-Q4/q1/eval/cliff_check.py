"""Clifford-angle correctness check for large n: every rotation angle a multiple of pi/2, the compiled circuit and
the logical circuit compared by <Z_i> and <Z_a Z_b> (all qubits, all interaction edges) of their stabilizer states."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_common import *  # noqa
from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState, Pauli, Clifford


def clifford_ising(n, edges, steps, seed=11):
    rng = random.Random(seed)
    t = 1.0
    c = math.pi / 4 * steps / t  # Lie-Trotter: each rotation angle 2 c t / steps = pi / 2
    terms = [(lab(n, {a: "Z", b: "Z"}), rng.choice([-1, 1]) * c) for a, b in edges]
    terms += [(lab(n, {q: "X"}), rng.choice([-1, 1]) * c) for q in range(n)]
    return evo(n, terms, t, steps, order=1)


def _snap(circ):
    """the circuit with every rotation angle snapped to the nearest multiple of pi/2 (refused if one is off by more
    than 1e-9), measurements and barriers removed: (circuit, {clbit: qubit})"""
    out = QuantumCircuit(circ.num_qubits)
    meas = {}
    for ins in circ.data:
        nm = ins.operation.name
        qs = [circ.find_bit(q).index for q in ins.qubits]
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = qs[0]; continue
        if nm in ("barrier", "delay"):
            continue
        op = ins.operation
        if op.params:
            v = float(op.params[0]); k = round(v / (math.pi / 2))
            if abs(v - k * math.pi / 2) > 1e-9:
                raise ValueError(f"{nm} angle {v} is not a multiple of pi/2")
            op = op.__class__(k * math.pi / 2)
        out.append(op, qs)
    return out, meas


def check(compiled, logical, edges):
    """max |difference| over <Z_i>, <Z_a Z_b> (edges), read through each circuit's measurements"""
    from qiskit import transpile
    lg = transpile(logical, basis_gates=["rz", "rx", "rzz", "h", "s", "sdg", "x", "cx", "cz", "sx"], optimization_level=0)
    c1, m1 = _snap(compiled)
    c0, m0 = _snap(lg)
    s1 = StabilizerState(Clifford(c1)); s0 = StabilizerState(Clifford(c0))
    n = logical.num_qubits
    def ev(s, c, m, qs):
        lab_ = ["I"] * c.num_qubits
        for q in qs:
            lab_[c.num_qubits - 1 - m[q]] = "Z"
        return s.expectation_value(Pauli("".join(lab_)))
    worst = 0.0
    obs = [[i] for i in range(n)] + [[a, b] for a, b in edges]
    for o in obs:
        worst = max(worst, abs(ev(s1, c1, m1, o) - ev(s0, c0, m0, o)))
    return worst
