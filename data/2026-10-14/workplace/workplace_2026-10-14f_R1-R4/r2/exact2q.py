import os
"""R2: exact two-qubit synthesis for the CZ basis without the Weyl snap. Qiskit's TwoQubitBasisDecomposer picks the
number of CZ (and, after the snap, the structure); its single-qubit gates are then solved exactly (Levenberg-Marquardt
on the U3 angles of each single-qubit layer, started at Qiskit's own values, global phase free). An exact solution
with the same CZ count exists whenever the snap moved the coordinates within the same class (a = b vs a ~ b; b = c = 0
vs b, c ~ 2e-5 with c ~ 0 ... ); if the refinement does not reach 1e-14 the CZ count is raised by one."""
import numpy as np
from scipy.optimize import least_squares
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, UGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer, OneQubitEulerDecomposer

_DEC = TwoQubitBasisDecomposer(CZGate(), euler_basis="U")
_E = OneQubitEulerDecomposer("U")
CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _u3(t, p, l):
    return np.array([[np.cos(t / 2), -np.exp(1j * l) * np.sin(t / 2)],
                     [np.exp(1j * p) * np.sin(t / 2), np.exp(1j * (p + l)) * np.cos(t / 2)]])


def _layers(c):
    """Qiskit's circuit as [1q layer, CZ, 1q layer, CZ, ...]; each 1q layer = (U on q0, U on q1)"""
    layers, cur = [], [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
    for ins in c.data:
        q = [c.find_bit(x).index for x in ins.qubits]
        if len(q) == 2:
            layers.append(cur); cur = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
        else:
            cur[q[0]] = Operator(ins.operation).data @ cur[q[0]]
    layers.append(cur)
    return layers


def _build(params, k):
    M = np.eye(4, dtype=complex)
    for j in range(k + 1):
        a = _u3(*params[6 * j:6 * j + 3]); b = _u3(*params[6 * j + 3:6 * j + 6])
        M = np.kron(b, a) @ M  # qubit 0 is the least significant
        if j < k:
            M = CZ @ M
    return M


def _angles(u):
    t, p, l, ph = _E.angles_and_phase(u)
    return [t, p, l]


def exact_2q(U, k=None):
    """(QuantumCircuit with cz and u gates exactly equal to U up to global phase, CZ count, residual)"""
    U = np.asarray(U, dtype=complex)
    c = _DEC(U) if k is None else _DEC(U, _num_basis_uses=k)
    k = sum(1 for i in c.data if len(i.qubits) == 2)
    lay = _layers(c)
    x0 = np.array([a for l0, l1 in lay for a in _angles(l0) + _angles(l1)])

    def res(x):
        M = _build(x, k)
        ov = np.trace(M.conj().T @ U)
        ph = ov / abs(ov) if abs(ov) > 1e-12 else 1.0
        d = (M * ph - U).ravel()
        return np.concatenate([d.real, d.imag])
    sol = least_squares(res, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    r = float(np.abs(res(sol.x)).max())
    if r > 1e-12 and k < 3:
        return exact_2q(U, k + 1)
    out = QuantumCircuit(2)
    for j in range(k + 1):
        out.append(UGate(*sol.x[6 * j:6 * j + 3]), [0]); out.append(UGate(*sol.x[6 * j + 3:6 * j + 6]), [1])
        if j < k:
            out.cz(0, 1)
    M = _build(sol.x, k)
    ov = np.trace(M.conj().T @ U); out.global_phase = float(np.angle(ov / abs(ov))) if abs(ov) > 0 else 0.0
    return out, k, r


if __name__ == "__main__":
    import pickle
    from qiskit.quantum_info import average_gate_fidelity
    from qiskit import qpy
    sub = qpy.load(open("bad_block.qpy", "rb"))[0]  # Q3's snapped block
    U = Operator(sub).data
    q = _DEC(U)
    print("Qiskit:", sum(1 for i in q.data if len(i.qubits) == 2), "CZ, infid %.2e" % (1 - average_gate_fidelity(Operator(U), Operator(q))))
    c, k, r = exact_2q(U)
    print("exact_2q:", k, "CZ, residual %.1e, infid %.2e" % (r, 1 - average_gate_fidelity(Operator(U), Operator(c))))
    rng = np.random.default_rng(1)
    from qiskit.quantum_info import random_unitary
    worst = 0
    for i in range(200):
        V = random_unitary(4, seed=int(rng.integers(1 << 30))).data
        c, k, r = exact_2q(V); worst = max(worst, 1 - average_gate_fidelity(Operator(V), Operator(c)))
    print("200 random unitaries: worst infid %.1e" % worst)


# ------------------------------------------------------------------ passes
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import UnitarySynthesis, ConsolidateBlocks, TwoQubitPeepholeOptimization
from qiskit.circuit.library import RZZGate

EXACT_STATS = {"synthesised": 0, "refined": 0, "raised_count": 0, "already_exact": 0}


def _refine(small, U):
    """`small`: a 2-qubit circuit of single-qubit gates, cz and rzz(theta) approximating U. Returns the same structure
    (single-qubit layers as U gates, rzz angles free) solved to equal U up to global phase, or None."""
    lay, twos = [], []
    cur = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
    for ins in small.data:
        q = [small.find_bit(x).index for x in ins.qubits]
        if len(q) == 2:
            lay.append(cur); cur = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
            twos.append((ins.operation.name, float(ins.operation.params[0]) if ins.operation.params else None, q))
        else:
            cur[q[0]] = Operator(ins.operation).data @ cur[q[0]]
    lay.append(cur)
    x0 = [a for l0, l1 in lay for a in _angles(l0) + _angles(l1)] + [t[1] for t in twos if t[0] == "rzz"]
    x0 = np.array(x0, dtype=float)
    nl = 6 * len(lay)

    def mat(x):
        M = np.eye(4, dtype=complex); r = nl
        for j in range(len(lay)):
            M = np.kron(_u3(*x[6 * j + 3:6 * j + 6]), _u3(*x[6 * j:6 * j + 3])) @ M
            if j < len(twos):
                nm = twos[j][0]
                if nm == "cz":
                    M = CZ @ M
                elif nm == "rzz":
                    th = x[r]; r += 1
                    M = np.diag(np.exp(-1j * th / 2 * np.array([1, -1, -1, 1]))) @ M
                else:
                    return None
        return M

    def res(x):
        M = mat(x)
        ov = np.trace(M.conj().T @ U)
        ph = ov / abs(ov) if abs(ov) > 1e-12 else 1.0
        d = (M * ph - U).ravel()
        return np.concatenate([d.real, d.imag])
    if mat(x0) is None:
        return None
    sol = least_squares(res, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    if float(np.abs(res(sol.x)).max()) > 1e-12:
        return None
    out = QuantumCircuit(2)
    r = nl
    for j in range(len(lay)):
        out.append(UGate(*sol.x[6 * j:6 * j + 3]), [0]); out.append(UGate(*sol.x[6 * j + 3:6 * j + 6]), [1])
        if j < len(twos):
            if twos[j][0] == "cz":
                out.cz(0, 1)
            else:
                out.append(RZZGate(sol.x[r]), [0, 1]); r += 1
    M = mat(sol.x); ov = np.trace(M.conj().T @ U)
    out.global_phase = float(np.angle(ov))
    return out


class ExactUnitarySynthesis(TransformationPass):
    """Qiskit's UnitarySynthesis for each two-qubit `unitary` (its gate choice and count), then the result solved
    exactly (`_refine`); if no exact solution exists with that structure, the generic three-CZ form is solved instead.
    Other unitaries go to Qiskit's pass unchanged."""

    def __init__(self, target, **kw):
        super().__init__()
        self.target = target
        self.inner = UnitarySynthesis(target=target, approximation_degree=1.0, **kw)

    def run(self, dag):
        for node in dag.op_nodes():
            if node.op.name != "unitary" or len(node.qargs) != 2:
                continue
            U = node.op.to_matrix()
            phys = [dag.find_bit(q).index for q in node.qargs]
            n = dag.num_qubits()
            big = QuantumCircuit(n)
            big.unitary(U, phys)
            syn = dag_to_circuit(self.inner.run(circuit_to_dag(big)))
            small = QuantumCircuit(2, global_phase=syn.global_phase)
            for ins in syn.data:
                small.append(ins.operation, [phys.index(syn.find_bit(q).index) for q in ins.qubits])
            Uloc = U  # qubit order: phys[0] is little-endian bit 0 in both
            if np.allclose(Operator(small).data, Uloc * (np.trace(Operator(small).data.conj().T @ Uloc) / 4) / abs(np.trace(Operator(small).data.conj().T @ Uloc) / 4), atol=1e-13):
                EXACT_STATS["already_exact"] += 1
                rep = small
            else:
                rep = _refine(small, Uloc)
                if rep is None:
                    rep, _, _ = exact_2q(Uloc, 3)
                    EXACT_STATS["raised_count"] += 1
                else:
                    EXACT_STATS["refined"] += 1
            EXACT_STATS["synthesised"] += 1
            mini = circuit_to_dag(rep)
            dag.substitute_node_with_dag(node, mini, wires=mini.qubits)
        return self.inner.run(dag)  # anything else (unitaries on 1 or 3+ qubits)


def exact_level3(no_commutative_cancellation=False, **kw):
    """generate_preset_pass_manager(3, **kw) with every UnitarySynthesis replaced by ExactUnitarySynthesis and every
    TwoQubitPeepholeOptimization by ConsolidateBlocks + ExactUnitarySynthesis (Qiskit's pre-2.x level-3 pair)."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.passmanager.flow_controllers import FlowControllerLinear
    pm = generate_preset_pass_manager(3, **kw)
    tg = kw.get("target")

    def swap(t):
        if isinstance(t, UnitarySynthesis):
            return [ExactUnitarySynthesis(tg)]
        if no_commutative_cancellation and type(t).__name__ == "CommutativeCancellation":
            return []
        if isinstance(t, TwoQubitPeepholeOptimization):
            return [ConsolidateBlocks(target=tg, approximation_degree=1.0), ExactUnitarySynthesis(tg)]
        if hasattr(t, "tasks"):
            new = []
            for s in t.tasks:
                new += swap(s)
            t.tasks = tuple(new)
        return [t]
    for st in ("init", "translation", "optimization"):
        s = getattr(pm, st)
        if s is None:
            continue
        new = []
        for t in s._tasks:
            for x in (t if isinstance(t, list) else [t]):
                new += swap(x)
        s._tasks = [[x] for x in new]
    return pm
